#include "MpmForceAssembly.h"

#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <Ziran/CS/Util/Timer.h>
#include <Ziran/Math/Geometry/Elements.h>
#include <Ziran/Math/Geometry/Particles.h>
#include <Ziran/Physics/LagrangianForce/LagrangianForce.h>
#include <Ziran/Sim/Scene.h>

namespace ZIRAN {
template <class T, int dim>
MpmForceAssembly<T, dim>::MpmForceAssembly(
    const bool &mls_mpm, const T &dx, const T &dt, Particles<T, dim> &particles,
    Scene<T, dim> &scene, CONFIG_SCHEME &config_scheme,
    StdVector<TV> &scratch_xp, TVStack &scratch_vp, TVStack &scratch_fp,
    StdVector<TM> &scratch_gradV, StdVector<TM> &scratch_force_stress,
    const TVStack &dv, const TVStack &vn, MpmGrid<T, dim> &grid,
    StdVector<uint64_t> &particle_base_offset, StdVector<int> &particle_order,
    std::vector<std::pair<int, int>> &particle_group,
    std::vector<uint64_t> &block_offset, int &num_nodes)
    : mls_mpm(mls_mpm), D_inverse(0), dx(dx), dt(dt), dv(dv), vn(vn),
      particles(particles), scene(scene), config_scheme(config_scheme),
      has_lagrangian_contact(false), scratch_xp(scratch_xp), scratch_vp(scratch_vp),
      scratch_fp(scratch_fp), scratch_gradV(scratch_gradV),
      scratch_force_stress(scratch_force_stress), grid(grid),
      particle_base_offset(particle_base_offset),
      particle_order(particle_order), particle_group(particle_group),
      block_offset(block_offset), num_nodes(num_nodes) {}

template <class T, int dim> MpmForceAssembly<T, dim>::~MpmForceAssembly() {}

template <class T, int dim> void MpmForceAssembly<T, dim>::reinitialize() {
  if (mls_mpm && D_inverse == 0) {
    T dx2 = dx * dx;
    if (interpolation_degree == 2)
      D_inverse = 4 / dx2;
    else if (interpolation_degree == 3)
      D_inverse = 3 / dx2;
    else
      ZIRAN_ASSERT(false);
  }

  forEachForceTerm([](auto &h) { h.reinitialize(); });
}

template <class T, int dim> void MpmForceAssembly<T, dim>::backupStrain() {
  ZIRAN_QUIET_TIMER();
  forEachForceTerm([](auto &h) { h.backupStrain(); });
}

template <class T, int dim> void MpmForceAssembly<T, dim>::restoreStrain() {
  ZIRAN_QUIET_TIMER();
  forEachForceTerm([](auto &h) { h.restoreStrain(); });
}

template <class T, int dim> void MpmForceAssembly<T, dim>::computeVAndGradV() {
  ZIRAN_QUIET_TIMER();
  evalInterpolantAndGradient(
      [&](int node_id) -> TV { return vn.col(node_id) + dv.col(node_id); },
      scratch_vp, scratch_gradV);
}

template <class T, int dim>
void MpmForceAssembly<T, dim>::computeDvAndGradDv(const TVStack &dv) const {
  ZIRAN_QUIET_TIMER();
  evalInterpolantAndGradient([&](int node_id) -> TV { return dv.col(node_id); },
                             scratch_vp, scratch_gradV);
}

template <class T, int dim>
template <bool USE_MLS_MPM>
void MpmForceAssembly<T, dim>::rasterizeForceToTVStack(const T scale,
                                                   TVStack &force) const {
  ZIRAN_QUIET_TIMER();

  grid.iterateGrid(
      [&](IV node, GridState<T, dim> &g) { g.new_v = TV::Zero(); });

  auto &Xarray = particles.X.array;
  auto grid_array = grid.grid->Get_Array();

  if constexpr (dim == 3 && std::is_same<T, float>::value) {
#ifndef __FMA__
    ZIRAN_ASSERT(false, "FMA not supported");
#else
    for (uint64_t color = 0; color < (1 << dim); ++color) {
      tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
        if ((block_offset[group_idx] & ((1 << dim) - 1)) != color)
          return;
        int first_i = particle_order[particle_group[group_idx].first];
        CachedOffset<T, dim> cache(particle_base_offset[first_i]);
        for (int idx = particle_group[group_idx].first;
             idx <= particle_group[group_idx].second; ++idx) {
          int i = particle_order[idx];
          TV &Xp = Xarray[i];
          TM stress = scratch_force_stress[i] * scale;
          TV fp = scratch_fp.col(i) * scale;
          __m128 __dx = _mm_set_ps(0, dx, dx, dx);
          __m128 __Xp = _mm_set_ps(0, Xp(2), Xp(1), Xp(0));
          __m128 __stress0 =
              _mm_set_ps(0, stress(2, 0), stress(1, 0), stress(0, 0));
          __m128 __stress1 =
              _mm_set_ps(0, stress(2, 1), stress(1, 1), stress(0, 1));
          __m128 __stress2 =
              _mm_set_ps(0, stress(2, 2), stress(1, 2), stress(0, 2));
          __m128 __fp = _mm_set_ps(0, -fp(2), -fp(1), -fp(0));
          BSplineWeights<T, dim> spline(Xp, dx);
          uint64_t *offset_line = cache.getCachedLine(particle_base_offset[i]);
          if constexpr (USE_MLS_MPM) {
            __m128 __D_inverse = _mm_set_ps(0, D_inverse, D_inverse, D_inverse);
            grid.superIterateKernel(spline, [&](const int &cnt,
                                                const __m128 &__node,
                                                const __m128 &__dw_w) {
              float *addr =
                  reinterpret_cast<float *>(&grid_array(offset_line[cnt]));
              __m128 __new_v_and_idx = _mm_load_ps(addr + 4);

              __m128 __w = _mm_shuffle_ps(__dw_w, __dw_w, 0xff);
              __m128 __w_D_inverse = _mm_mul_ps(__w, __D_inverse);
              __m128 __xi_minus_xp = _mm_fmsub_ps(__node, __dx, __Xp);

              __m128 __xi_minus_xp0 =
                  _mm_shuffle_ps(__xi_minus_xp, __xi_minus_xp, 0x00);
              __m128 __xi_minus_xp1 =
                  _mm_shuffle_ps(__xi_minus_xp, __xi_minus_xp, 0x55);
              __m128 __xi_minus_xp2 =
                  _mm_shuffle_ps(__xi_minus_xp, __xi_minus_xp, 0xaa);

              __m128 __delta = _mm_fmadd_ps(__stress0, __xi_minus_xp0, __fp);
              __delta = _mm_fmadd_ps(__stress1, __xi_minus_xp1, __delta);
              __delta = _mm_fmadd_ps(__stress2, __xi_minus_xp2, __delta);
              __delta = _mm_mul_ps(__delta, __w_D_inverse);

              __new_v_and_idx = _mm_sub_ps(__new_v_and_idx, __delta);
              _mm_store_ps(addr + 4, __new_v_and_idx);
            });
          } else {
            grid.superIterateKernel(spline, [&](const int &cnt,
                                                const __m128 &__node,
                                                const __m128 &__dw_w) {
              float *addr =
                  reinterpret_cast<float *>(&grid_array(offset_line[cnt]));
              __m128 __new_v_and_idx = _mm_load_ps(addr + 4);

              __m128 __w = _mm_shuffle_ps(__dw_w, __dw_w, 0xff);

              __m128 __dw0 = _mm_shuffle_ps(__dw_w, __dw_w, 0x00);
              __m128 __dw1 = _mm_shuffle_ps(__dw_w, __dw_w, 0x55);
              __m128 __dw2 = _mm_shuffle_ps(__dw_w, __dw_w, 0xaa);
              __m128 __delta = _mm_mul_ps(__fp, __w);
              __delta = _mm_fmadd_ps(__stress0, __dw0, __delta);
              __delta = _mm_fmadd_ps(__stress1, __dw1, __delta);
              __delta = _mm_fmadd_ps(__stress2, __dw2, __delta);

              __new_v_and_idx = _mm_sub_ps(__new_v_and_idx, __delta);
              _mm_store_ps(addr + 4, __new_v_and_idx);
            });
          }
        }
      });
    }
#endif
  } else {
    // std::cout << particle_group.size() << "!!!!!!!!!!!!!!!!!!!!\n";
    for (uint64_t color = 0; color < (1 << dim); ++color) {
      tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
        if ((block_offset[group_idx] & ((1 << dim) - 1)) != color)
          return;
        for (int idx = particle_group[group_idx].first;
             idx <= particle_group[group_idx].second; ++idx) {
          int i = particle_order[idx];
          TV &Xp = Xarray[i];

          TM4 stress_density = TM4::Zero();
          TM &stress = scratch_force_stress[i];
          stress_density.template topLeftCorner<dim, dim>() =
              stress; // stress density
          const TV &fp = scratch_fp.col(i);
          stress_density.template topRightCorner<dim, 1>() =
              -fp; // fp (for meshed forces).

          BSplineWeights<T, dim> spline(Xp, dx);
          grid.iterateKernel(
              spline, particle_base_offset[i],
              [&](const IV &node, T w, const TV &dw, GridState<T, dim> &g) {
                TV4 weight = TV4::Zero();
                weight.template topLeftCorner<dim, 1>() = dw;
                weight(3) = w;
                if constexpr (USE_MLS_MPM) {
                  TV4 xi_minus_xp = TV4::Zero();
                  xi_minus_xp.template topLeftCorner<dim, 1>() =
                      node.template cast<T>() * dx -
                      Xp; // top dim entries non-zero
                  xi_minus_xp(3) = 1;
                  TV4 delta =
                      ((stress_density * xi_minus_xp) * (w * D_inverse));
                  g.new_v -= scale * delta.template topLeftCorner<
                                         dim, 1>(); // fi -= \sum_p (Ap (xi-xp)
                                                    // - fp )w_ip Dp_inv
                } else {
                  TV4 delta = (stress_density * weight);
                  g.new_v -= scale * delta.template topLeftCorner<dim, 1>();
                }
              });
        }
      });
    }
  }

  grid.iterateGrid(
      [&](IV node, GridState<T, dim> &g) { force.col(g.idx) += g.new_v; });
}

/**
   Implicit-step force-state update. Called once per Newton updateStateForDv()
   (via updateStrainAndPositions(), see below), at the trial position scratch_xp.

   Recomputes, into per-particle scratch, everything the residual/stiffness need:
     1. zero scratch_force_stress (P F^T) and scratch_fp (nodal force),
     2. each force helper accumulates its stress (force_stress) and force (fp)
        via updateImplicitState() -- forEachForceTerm over the sub-helpers.
   The filled scratch_force_stress is then read back by rasterizeForceToTVStack()
   during computeResidual()/multiply().
*/
template <class T, int dim>
void MpmForceAssembly<T, dim>::updateForcesAtState() {
  ZIRAN_QUIET_TIMER();
  // zero out scratch_fp and scratch_force_stress, prepare for adding lagrangian force
  tbb::parallel_for(tbb::blocked_range<size_t>(0, particles.count),
                    [&](const tbb::blocked_range<size_t> &range) {
                      for (size_t b = range.begin(), b_end = range.end();
                           b < b_end; ++b) {
                        scratch_fp.col(b) = TV::Zero();
                        scratch_force_stress[b] = TM::Zero();
                      }
                    });

  auto &force_stress = scratch_force_stress;
  auto &fp = scratch_fp;
  auto ranges = particles.X.ranges;
  tbb::parallel_for(ranges, [&](DisjointRanges &subrange) {
    forEachForceTerm([&](auto &h) {
      h.updateImplicitState(subrange, force_stress, fp); // this adds to force_stress and fp
    });
  });
}
/**
      Compute for a vector function f defined on active grid nodes, compute f
   and grad f on particles
      */
template <class T, int dim>
template <class Func>
void MpmForceAssembly<T, dim>::evalInterpolantAndGradient(
    Func &&f, TVStack &f_eval, StdVector<TM> &grad_f) const {
  grid.iterateTouchedGrid(
      [&](IV node, GridState<T, dim> &g) { g.new_v = TV::Zero(); });
  grid.iterateGrid(
      [&](IV node, GridState<T, dim> &g) { g.new_v = f((int)g.idx); });
  if constexpr (dim == 3 && std::is_same<T, float>::value) {
#ifndef __FMA__
    ZIRAN_ASSERT(false, "FMA not supported");
#else
    auto grid_array = grid.grid->Get_Array();
    for (uint64_t color = 0; color < (1 << dim); ++color) {
      tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
        if ((block_offset[group_idx] & ((1 << dim) - 1)) != color)
          return;
        int first_i = particle_order[particle_group[group_idx].first];
        CachedOffset<T, dim> cache(particle_base_offset[first_i]);
        for (int idx = particle_group[group_idx].first;
             idx <= particle_group[group_idx].second; ++idx) {
          int i = particle_order[idx];
          // if (grad_fp != grad_fp)
          //     continue;

          TV &Xp = particles.X[i];
          BSplineWeights<T, dim> spline(Xp, dx);

          uint64_t *offset_line = cache.getCachedLine(particle_base_offset[i]);
          __m128 __gradf_f0 = _mm_setzero_ps(), __gradf_f1 = _mm_setzero_ps(),
                 __gradf_f2 = _mm_setzero_ps();
          grid.superIterateKernel(spline, [&](const int &cnt,
                                              const __m128 &__node,
                                              const __m128 &__dw_w) {
            GridState<T, dim> &g = reinterpret_cast<GridState<T, dim> &>(
                grid_array(offset_line[cnt]));
            __m128 __new_v = _mm_set_ps(0, g.new_v[2], g.new_v[1], g.new_v[0]);
            __m128 __v0 = _mm_shuffle_ps(__new_v, __new_v, 0x00);
            __m128 __v1 = _mm_shuffle_ps(__new_v, __new_v, 0x55);
            __m128 __v2 = _mm_shuffle_ps(__new_v, __new_v, 0xaa);
            __gradf_f0 = _mm_fmadd_ps(__v0, __dw_w, __gradf_f0);
            __gradf_f1 = _mm_fmadd_ps(__v1, __dw_w, __gradf_f1);
            __gradf_f2 = _mm_fmadd_ps(__v2, __dw_w, __gradf_f2);
          });
          TM4 gradf_f = TM4::Zero();
          float *addr = (float *)&gradf_f;
          _mm_store_ps(addr, __gradf_f0);
          _mm_store_ps(addr + 4, __gradf_f1);
          _mm_store_ps(addr + 8, __gradf_f2);
          grad_f[i] = gradf_f.template topLeftCorner<dim, dim>().transpose();
          f_eval.col(i) =
              gradf_f.template bottomLeftCorner<1, dim>().transpose();
        }
      });
    }
#endif
  } else {
    for (uint64_t color = 0; color < (1 << dim); ++color) {
      tbb::parallel_for(0, (int)particle_group.size(), [&](int group_idx) {
        if ((block_offset[group_idx] & ((1 << dim) - 1)) != color)
          return;
        for (int idx = particle_group[group_idx].first;
             idx <= particle_group[group_idx].second; ++idx) {
          int i = particle_order[idx];
          TM &grad_fp = grad_f[i];
          // if (grad_fp != grad_fp)
          //     continue;

          TV &Xp = particles.X[i];
          BSplineWeights<T, dim> spline(Xp, dx);

          grad_fp = TM::Zero();
          f_eval.col(i) = TV::Zero();
          grid.iterateKernel(
              spline, particle_base_offset[i],
              [&](const IV &node, T w, const TV &dw, GridState<T, dim> &g) {
                grad_fp.noalias() += g.new_v * dw.transpose();
                f_eval.col(i) += g.new_v * w;
              });
        }
      });
    }
  }
}

template <class T, int dim>
void MpmForceAssembly<T, dim>::addScaledForces(const T scale,
                                           TVStack &forces) const {
  ZIRAN_QUIET_TIMER();
  if (mls_mpm) {
    rasterizeForceToTVStack<true>(scale, forces);
  } else {
    rasterizeForceToTVStack<false>(scale, forces);
  }
}

template <class T, int dim>
void MpmForceAssembly<T, dim>::addScaledForceDifferential(const T scale,
                                                      const TVStack &dv,
                                                      TVStack &df) const {
  ZIRAN_QUIET_TIMER();

  computeDvAndGradDv(dv);

  // scratch_gradV is now temporaraly used for storing gradDV (evaluated at
  // particles) scratch_vp is now temporaraly used for storing DV (evaluated at
  // particles)

  // zero out scratch_fp and scratch_force_stress, prepare for adding lagrangian force
  // differentials
  tbb::parallel_for(tbb::blocked_range<size_t>(0, particles.count),
                    [&](const tbb::blocked_range<size_t> &range) {
                      //   printf("----------------------%d--------------------\n",particles.count);
                      for (size_t b = range.begin(), b_end = range.end();
                           b < b_end; ++b) {
                        scratch_fp.col(b) = TV::Zero();
                        scratch_force_stress[b] = TM::Zero();
                      }
                    });

  auto ranges = particles.X.ranges;
  tbb::parallel_for(ranges, [&](DisjointRanges &subrange) {
    forEachForceTerm([&](auto &h) {
      h.computeStressDifferential(subrange, scratch_gradV, scratch_force_stress,
                                  scratch_vp, scratch_fp);
    });
    // scratch_force_stress is now V_p^0 dP (F_p^n)^T (dP is Ap in snow paper)
  });
  if (mls_mpm) {
    rasterizeForceToTVStack<true>(scale, df);
  } else {
    rasterizeForceToTVStack<false>(scale, df);
  }
}
/**
   Implicit-only. Called by the objective's updateStateForDv() once per Newton
   iteration to move the material state to the current grid-velocity guess.

   Steps:
     1. computeVAndGradV()      : interpolate the current grid velocity (and its
                                  gradient) back to particles into scratch_vp,
     2. restoreStrain()+evolveStrain(dt): rebuild F from the backed-up strain,
     3. scratch_xp = xn + dt*vp : trial particle positions,
     4. updateForcesAtState(): refill scratch_force_stress / scratch_fp at xp.
*/
template <class T, int dim>
void MpmForceAssembly<T, dim>::updateStrainAndPositions() {
  ZIRAN_QUIET_TIMER();
  computeVAndGradV();

  restoreStrain();
  evolveStrain(dt);

  auto *X0 = particles.DataManager::getPointer(X0_name<TV>());

  // xp = xn + dt * vp
  tbb::parallel_for(tbb::blocked_range<size_t>(0, particles.count),
                    [&](const tbb::blocked_range<size_t> &range) {
                      for (size_t b = range.begin(), b_end = range.end();
                           b < b_end; ++b) {
                        auto &xn = (config_scheme != CONFIG_UPDATE_LAGRANGIAN)
                                       ? (*X0)[b]
                                       : particles.X[b];
                        scratch_xp[b] = xn + dt * scratch_vp.col(b);
                      }
                    });

  if (config_scheme != CONFIG_UPDATE_LAGRANGIAN && has_lagrangian_contact) {
    T radius = 2 * dx;
    scratch_hash.rebuild(radius, scratch_xp);
  }

  updateForcesAtState();
}

template <class T, int dim> void MpmForceAssembly<T, dim>::evolveStrain(T dt) {
  // TODO: combinme evolve strain and update state?
  // printf("evloveStrain\n");
  ZIRAN_QUIET_TIMER();
  auto ranges = particles.X.ranges;
  tbb::parallel_for(ranges, [&](DisjointRanges &subrange) {
    forEachForceTerm([&](auto &h) { h.evolveStrain(subrange, dt, scratch_gradV); });
  });
}

template <class T, int dim>
int MpmForceAssembly<T, dim>::countNearlyIncompressibleParticles(T eps) {
  auto *F_sn_ptr = particles.DataManager::getPointer(F_sn_name<T, dim>());
  auto *phase_ptr = particles.DataManager::getPointer(material_phase_name());
  ZIRAN_ASSERT(F_sn_ptr && phase_ptr, "Missing J or material phase");

  const int n = particles.count;

  return tbb::parallel_reduce(
      tbb::blocked_range<int>(0, n), 0,
      [&](const tbb::blocked_range<int> &r, int local) -> int {
        for (int i = r.begin(); i != r.end(); ++i) {
          // 注意：若你的粒子有“空洞/已移除”索引，这种方式就需要你自己过滤
          auto &F_sn = (*F_sn_ptr)[i];
          const auto &ph = (*phase_ptr)[i];
          if (ph != MATERIAL_PHASE_FLUID &&
              std::abs(F_sn.determinant() - (T)1) > eps) {
            ++local;
          }
        }
        return local;
      },
      std::plus<int>());
}

template <class T, int dim>
void MpmForceAssembly<T, dim>::updateConfig(T eps, T eta) {
  ZIRAN_QUIET_TIMER();
  auto ranges = particles.X.ranges;
  if (config_scheme == CONFIG_AU_LAGRANGIAN) {

    // plasticity
    tbb::parallel_for(ranges, [&](DisjointRanges &subrange) {
      DisjointRanges subset(
          subrange, particles.commonRanges(F_name<T, dim>(), F_s_name<T, dim>(),
                                           F_sn_name<T, dim>(), J_name<T>(),
                                           X0_name<TV>()));
      const auto &X0Array = particles.DataManager::get(X0_name<TV>()).array;
      for (auto it = particles.subsetIter(
               subset, F_name<T, dim>(), F_s_name<T, dim>(),
               F_sn_name<T, dim>(), J_name<T>(), X0_name<TV>());
           it; ++it) {
        auto &F = it.template get<0>();
        auto &F_s = it.template get<1>();
        auto &F_sn = it.template get<2>();
        auto &J = it.template get<3>();
        const auto &X0 = it.template get<4>();
        F_sn = F * F_s.inverse();
      }
    });

    int number = countNearlyIncompressibleParticles(eps);
    printf("number of near incompressible particles: %d\n", number);
    T ratio = number / (T)particles.count;
    if (ratio > eta) {
      // printf("更新\n");
      ZIRAN_INFO("update Configuration ");
      tbb::parallel_for(ranges, [&](DisjointRanges &subrange) {
        DisjointRanges subset(
            subrange, particles.commonRanges(
                          F_name<T, dim>(), F_s_name<T, dim>(),
                          F_sn_name<T, dim>(), J_name<T>(), X0_name<TV>()));
        const auto &X0Array = particles.DataManager::get(X0_name<TV>()).array;
        for (auto it = particles.subsetIter(
                 subset, F_name<T, dim>(), F_s_name<T, dim>(),
                 F_sn_name<T, dim>(), J_name<T>(), X0_name<TV>());
             it; ++it) {
          auto &F = it.template get<0>();
          auto &F_s = it.template get<1>();
          auto &F_sn = it.template get<2>();
          auto &J = it.template get<3>();
          const auto &X0 = it.template get<4>();
          int p = it.entryId();
          F_s = F;
          F_sn = TM::Identity();
          particles.X[p] =
              X0; // 避免直接用 particles.X[p]，把 X/X0 加入 subset 保证线程安全
        }
      });
    }
  }
}

template <class T, int dim> T MpmForceAssembly<T, dim>::totalEnergy() const {
  ZIRAN_QUIET_TIMER();
  auto ranges = particles.X.ranges;
  T result = tbb::parallel_reduce(
      ranges, 0.0,
      [&](const DisjointRanges &subset, const double &e) -> double {
        double psi = e;
        forEachForceTerm([&](auto &h) { psi += h.totalEnergy(subset); });
        return psi;
      },
      [&](const double &x, const double &y) -> double { return x + y; });

  return result;
}
} // namespace ZIRAN
