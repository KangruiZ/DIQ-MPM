#include "SplittingSimulation.h"
#include "../coupling/GpuOperatorProbe.h"
#include <Ziran/Math/Linear/DenseExt.h>
#include <Ziran/Math/Linear/ImplicitQRSVD.h>
#include <optional>
#include <tbb/parallel_sort.h>

namespace ZIRAN {

template <class T, int dim>
template <int degree, int order>
void SplittingSimulation<T, dim>::solidKernel(MpmGrid<T, dim, degree> &grid3,
                                              MpmGrid<T, dim, order> &grid4,
                                              SM &M_inv, SM &G, SM &L, SM &D,
                                              SM &S, TStack &a, TStack &b) {
  ZIRAN_TIMER();
  auto &Xarray = particles.X.array;
  auto *J_pointer = particles.DataManager::getPointer(J_name<T>());
  auto *vol_pointer =
      particles.DataManager::getPointer(element_measure_name<T>());
  auto *F_pointer = particles.DataManager::getPointer(F_name<T, dim>());
  auto *F_sn_pointer = particles.DataManager::getPointer(F_sn_name<T, dim>());
  StdVector<TV> &X_for_p = particles.X.array;

  std::optional<GPU_STAGE3A::SolidQ2Q0AssemblyResult> gpu_assembled;
  gpu_device_assembly_valid = false;

  if (gpu_stage3b_assemble_blocks) {
    if constexpr (dim == 3 && degree == 2 && order == 0) {
      ZIRAN_TIMER();
      const bool resident_velocity_nodes =
          gpu_solid_final_grid_device_ready &&
          gpu_solid_p2g_node_map.size != 0;
      num_v = resident_velocity_nodes
          ? static_cast<int>(gpu_solid_p2g_node_map.size)
          : grid3.getNumNodes();
      GPU_STAGE3A::SolidQ2Q0AssemblyInput input;
      input.dx = static_cast<double>(Base::dx);
      input.dt = static_cast<double>(Base::dt);
      input.gravity = GPU_STAGE3A::Vec3d{
          static_cast<double>(Base::gravity[0]),
          static_cast<double>(Base::gravity[1]),
          static_cast<double>(Base::gravity[2])};
      input.velocity_node_count = num_v;
      input.pressure_node_count = num_p;

      auto push_node = [](auto &dst, const IV &node,
                          const GridState<T, dim> &g) {
        dst.push_back(GPU_STAGE3A::AssemblyGridNode{
            GPU_STAGE3A::packGridKey3D(node[0], node[1], node[2]),
            static_cast<int>(g.idx), static_cast<double>(g.m),
            GPU_STAGE3A::Vec3d{static_cast<double>(g.new_v[0]),
                               static_cast<double>(g.new_v[1]),
                               static_cast<double>(g.new_v[2])},
            GPU_STAGE3A::Vec3d{static_cast<double>(g.v[0]),
                               static_cast<double>(g.v[1]),
                               static_cast<double>(g.v[2])}});
      };
      if (resident_velocity_nodes) {
        input.resident_velocity_nodes = gpu_solid_p2g_node_map;
      } else {
        grid3.iterateGridSerial([&](IV node, GridState<T, dim> &g) {
          push_node(input.velocity_nodes, node, g);
        });
      }
      if (gpu_prebuilt_solid_pressure_valid &&
          gpu_pressure_node_map.storage &&
          gpu_pressure_node_map.size ==
              static_cast<std::size_t>(input.pressure_node_count))
        input.resident_pressure_nodes = gpu_pressure_node_map;
      else if (gpu_prebuilt_solid_pressure_valid)
        input.pressure_nodes = gpu_pressure_nodes;
      else
        grid4.iterateGridSerial([&](IV node, GridState<T, dim> &g) {
          push_node(input.pressure_nodes, node, g);
        });

      input.resident_particles =
          GPU_STAGE3A::getPersistentSolidParticleStateDeviceView(
              gpu_solid_particle_state);
      input.compute_f_inv_t =
          Base::config_scheme != CONFIG_UPDATE_LAGRANGIAN ? 1 : 0;
      if (input.resident_particles.size == 0) {
        input.particles.resize(static_cast<std::size_t>(particles.count));
        tbb::parallel_for(0, particles.count, [&](int i) {
          GPU_STAGE3A::SolidQ2Q0Particle particle;
          const TV &x = Xarray[i];
          const TV &x_for_p = X_for_p[i];
          particle.x = GPU_STAGE3A::Vec3d{
              static_cast<double>(x[0]), static_cast<double>(x[1]),
              static_cast<double>(x[2])};
          particle.x_for_p = GPU_STAGE3A::Vec3d{
              static_cast<double>(x_for_p[0]), static_cast<double>(x_for_p[1]),
              static_cast<double>(x_for_p[2])};
          particle.volume = static_cast<double>((*vol_pointer)[i]);
          particle.j = static_cast<double>((*J_pointer)[i]);
          particle.compute_f_inv_t = input.compute_f_inv_t;
          const TM F = Base::config_scheme == CONFIG_TOTAL_LAGRANGIAN
              ? (*F_pointer)[i]
              : (Base::config_scheme == CONFIG_UPDATE_LAGRANGIAN
                      ? TM::Identity()
                      : (*F_sn_pointer)[i]);
          for (int row = 0; row < dim; ++row)
            for (int col = 0; col < dim; ++col)
              particle.f.values[row * 3 + col] =
                  static_cast<double>(F(row, col));
          input.particles[static_cast<std::size_t>(i)] = particle;
        });
      }

      GPU_STAGE3A::SolidQ2Q0DeviceAssemblyResult device_assembled;
      GPU_STAGE3A::assembleSolidQ2Q0System3DDevice(input, device_assembled);
      gpu_m_inv = device_assembled.m_inv;
      gpu_g = device_assembled.g;
      gpu_s = device_assembled.s;
      gpu_a = device_assembled.a;
      gpu_b = device_assembled.b;
      gpu_velocity_node_map = device_assembled.velocity_nodes;
      gpu_pressure_node_map = device_assembled.pressure_nodes;
      gpu_device_assembly_valid = true;
      if (gpu_stage3b_validate_blocks) {
        GPU_STAGE3A::SolidQ2Q0AssemblyResult assembled;
        assembled.m_inv =
            GPU_STAGE3A::copyDeviceCsrToHost(device_assembled.m_inv);
        assembled.g = GPU_STAGE3A::copyDeviceCsrToHost(device_assembled.g);
        assembled.s = GPU_STAGE3A::copyDeviceCsrToHost(device_assembled.s);
        assembled.a = GPU_STAGE3A::copyDeviceVectorToHost(device_assembled.a);
        assembled.b = GPU_STAGE3A::copyDeviceVectorToHost(device_assembled.b);
        gpu_assembled.emplace(std::move(assembled));
      }
      ZIRAN_INFO(gpu_stage3b_validate_blocks
                     ? "GPU solid assembly oracle: num_v="
                     : "GPU solid device assembly: num_v=",
                 num_v,
                 " num_p=", num_p,
                 " resident_velocity_nodes=",
                 resident_velocity_nodes ? 1 : 0,
                 " M_inv_nnz=", device_assembled.m_inv.nonzeros,
                 " G_nnz=", device_assembled.g.nonzeros,
                 " S_nnz=", device_assembled.s.nonzeros);
    } else {
      ZIRAN_WARN("Stage3B GPU solid assembly only supports 3D Q2Q0; using CPU assembly");
    }
  }

  if (gpu_device_assembly_valid && !gpu_stage3b_validate_blocks) {
    M_inv = SM(num_v * dim, num_v * dim);
    G = SM(num_v * dim, num_p);
    D = SM(num_p, num_v * dim);
    S = SM(num_p, num_p);
    a = TStack::Zero(num_v * dim);
    b = TStack::Zero(num_p);
    return;
  }

  auto use_gpu_blocks = [&]() {
    ZIRAN_ASSERT(gpu_assembled.has_value(), "missing GPU solid assembly");
    M_inv = GPU_STAGE3A::makeEigenFromCsrLocal<T>(gpu_assembled->m_inv);
    G = GPU_STAGE3A::makeEigenFromCsrLocal<T>(gpu_assembled->g);
    D = Eigen::SparseMatrix<T>(-G.transpose());
    S = GPU_STAGE3A::makeEigenFromCsrLocal<T>(gpu_assembled->s);
    a = TStack::Zero(static_cast<int>(gpu_assembled->a.size()));
    b = TStack::Zero(static_cast<int>(gpu_assembled->b.size()));
    for (int i = 0; i < a.rows(); ++i)
      a(i) = static_cast<T>(gpu_assembled->a[static_cast<std::size_t>(i)]);
    for (int i = 0; i < b.rows(); ++i)
      b(i) = static_cast<T>(gpu_assembled->b[static_cast<std::size_t>(i)]);
  };

  // build M
  num_v = grid3.getNumNodes();
  M_inv = Eigen::SparseMatrix<T>(num_v * dim, num_v * dim);
  std::vector<Eigen::Triplet<T>> M_inv_tri;
  grid3.iterateGridSerial([&](IV node, GridState<T, dim> &g) {
    for (int d = 0; d < dim; ++d)
      M_inv_tri.emplace_back(g.idx * dim + d, g.idx * dim + d, Base::dt / g.m);
  });
  M_inv.setFromTriplets(M_inv_tri.begin(), M_inv_tri.end());

  // build G
  G = Eigen::SparseMatrix<T>(num_v * dim, num_p);
  std::vector<Eigen::Triplet<T>> G_tri(particles.count * grid3.kernel_size *
                                       grid4.kernel_size * dim);
  {
    ZIRAN_TIMER();
    tbb::parallel_for(0, particles.count, [&](int i) {
      TV Xp = Xarray[i];
      TV X_for_pp = X_for_p[i];
      T J = (*J_pointer)[i];
      T vol0 = (*vol_pointer)[i];

      TM F, F_inv_T;
      if (Base::config_scheme != CONFIG_UPDATE_LAGRANGIAN) {
        if (Base::config_scheme == CONFIG_TOTAL_LAGRANGIAN) {
          F = (*F_pointer)[i];
        } else {
          F = (*F_sn_pointer)[i];
        }
        TM U, V;
        TV sigma;
        singularValueDecomposition(F, U, sigma, V);
        for (int i = 0; i < dim; ++i) {
          if (sigma(i) >= 1e-12) {
            sigma(i) = 1.0 / sigma(i);
          }
          // else {
          //     printf("有问题\n");
          //     exit(0);
          // }
        }
        TM sigma_matrix = sigma.asDiagonal();
        F_inv_T = V * sigma_matrix * U.transpose();
        F_inv_T.transposeInPlace();
      }

      BSplineWeights<T, dim, degree> spline3(Xp, Base::dx);
      // BSplineWeights<T, dim, degree> spline3(degree == 2 ? Xp : (Xp -
      // TV::Ones() * Base::dx * 0.5), Base::dx); BSplineWeights<T, dim, order>
      // spline4(Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
      BSplineWeights<T, dim, order> spline4(
          X_for_pp - TV::Ones() * Base::dx * 0.5, Base::dx);
      uint64_t offset3 =
          Base::SparseMask::Linear_Offset(to_std_array(spline3.base_node));
      uint64_t offset4 =
          Base::SparseMask::Linear_Offset(to_std_array(spline4.base_node));
      int cnt = i * grid3.kernel_size * grid4.kernel_size * dim;
      grid3.iterateKernel(
          spline3, offset3, [&](IV node3, T w3, TV dw3, GridState<T, dim> &g3) {
            int idx3 = g3.idx;
            if (idx3 < 0)
              return;
            grid4.iterateKernel(
                spline4, offset4,
                [&](IV node4, T w4, TV dw4, GridState<T, dim> &g4) {
                  int idx4 = g4.idx;
                  if (idx4 < 0)
                    return;
                  for (int alpha = 0; alpha < dim; ++alpha) {
                    // G_tri[cnt++] = Eigen::Triplet<T>(idx3 * dim + alpha,
                    // idx4, -vol0 * w4 * dw3[alpha]);
                    if (Base::config_scheme != CONFIG_UPDATE_LAGRANGIAN) {
                      G_tri[cnt++] = Eigen::Triplet<T>(
                          idx3 * dim + alpha, idx4,
                          -vol0 * J * w4 * (F_inv_T * dw3)[alpha]);
                    } else {
                      G_tri[cnt++] =
                          Eigen::Triplet<T>(idx3 * dim + alpha, idx4,
                                            -vol0 * J * w4 * dw3[alpha]);
                    }
                  }
                });
          });
    });
  }
  G.setFromTriplets(G_tri.begin(), G_tri.end());
  // build D
  D = Eigen::SparseMatrix<T>(-G.transpose());
  // build S
  S = Eigen::SparseMatrix<T>(num_p, num_p);
  S.setZero();
  grid4.iterateGridSerial([&](IV node2, GridState<T, dim> &g2) {
    int idx = g2.idx;
    S.coeffRef(idx, idx) += g2.v[0] / Base::dt;
  });
  // S.setZero();
  std::cout << "------------------" << G.nonZeros() << "-------------------\n";
  // build a
  a = TStack::Zero(num_v * dim);
  grid3.iterateGrid([&](IV node, GridState<T, dim> &g) {
    for (int d = 0; d < dim; ++d)
      a(g.idx * dim + d) +=
          g.m / Base::dt * (g.new_v[d] + Base::dt * Base::gravity[d]);
  });
  // build b
  b = TStack::Zero(num_p);
  grid4.iterateGrid([&](IV node, GridState<T, dim> &g) {
    int idx = g.idx;
    b(idx) += g.v[0] * g.v[1] / Base::dt;
  });
  // b.setZero();

  if (gpu_assembled) {
    const Eigen::SparseMatrix<double> cpu_m_inv = M_inv.template cast<double>();
    const Eigen::SparseMatrix<double> cpu_g = G.template cast<double>();
    const Eigen::SparseMatrix<double> cpu_s = S.template cast<double>();
    const auto minv_stats = GPU_STAGE3A::compareCsrMatrices(
        "solid_Minv", gpu_assembled->m_inv,
        GPU_STAGE3A::makeCsrFromEigen(cpu_m_inv));
    const auto g_stats = GPU_STAGE3A::compareCsrMatrices(
        "solid_G", gpu_assembled->g, GPU_STAGE3A::makeCsrFromEigen(cpu_g));
    const auto s_stats = GPU_STAGE3A::compareCsrMatrices(
        "solid_S", gpu_assembled->s, GPU_STAGE3A::makeCsrFromEigen(cpu_s));
    ZIRAN_INFO("GPU solid base oracle M_inv: max_abs=", minv_stats.max_abs_error,
               " max_rel=", minv_stats.max_rel_error,
               " G: max_abs=", g_stats.max_abs_error,
               " max_rel=", g_stats.max_rel_error,
               " S: max_abs=", s_stats.max_abs_error,
               " max_rel=", s_stats.max_rel_error);

    T max_a_error = 0;
    T max_b_error = 0;
    ZIRAN_ASSERT(a.rows() == static_cast<int>(gpu_assembled->a.size()),
                 "GPU solid a size mismatch");
    ZIRAN_ASSERT(b.rows() == static_cast<int>(gpu_assembled->b.size()),
                 "GPU solid b size mismatch");
    for (int i = 0; i < a.rows(); ++i)
      max_a_error = std::max(
          max_a_error,
          std::abs(a(i) - static_cast<T>(gpu_assembled->a[static_cast<std::size_t>(i)])));
    for (int i = 0; i < b.rows(); ++i)
      max_b_error = std::max(
          max_b_error,
          std::abs(b(i) - static_cast<T>(gpu_assembled->b[static_cast<std::size_t>(i)])));
    ZIRAN_INFO("GPU solid base oracle a: max_abs=", max_a_error,
               " b: max_abs=", max_b_error);

    const bool oracle_passed =
        minv_stats.max_rel_error <= 1e-12 &&
        g_stats.max_rel_error <= 1e-12 &&
        s_stats.max_rel_error <= 1e-12 &&
        max_a_error <= (T)1e-12 && max_b_error <= (T)1e-12;
    if (oracle_passed) {
      use_gpu_blocks();
    } else {
      ZIRAN_WARN("GPU solid base assembly failed oracle; keeping CPU blocks");
    }
  }
}

template <class T, int dim>
template <int degree, int order>
void SplittingSimulation<T, dim>::buildSolidSystem() {
  ZIRAN_TIMER();
  auto &Xarray = particles.X.array;
  auto *J_pointer = particles.DataManager::getPointer(J_name<T>());
  auto *vol_pointer =
      particles.DataManager::getPointer(element_measure_name<T>());
  auto *j_model =
      particles.DataManager::getPointer(MaterialForce::j_constitutive_model_name());

  MpmGrid<T, dim, order> grid4;
  StdVector<TV> &X_for_p = particles.X.array;
  gpu_prebuilt_solid_pressure_valid = false;
  gpu_pressure_nodes.clear();
  gpu_pressure_node_map = {};
  if (gpu_build_system && dim == 3) {
    initializePersistentSolidParticleOwner(true);
    const GPU_STAGE3A::SolidParticleStateDeviceView resident_particles =
        GPU_STAGE3A::getPersistentSolidParticleStateDeviceView(
            gpu_solid_particle_state);
    std::vector<GPU_STAGE2::Vec3d> gpu_positions;
    std::vector<double> gpu_j;
    std::vector<double> gpu_volume;
    std::vector<double> gpu_lambda;
    if (resident_particles.size == 0) {
      gpu_positions.resize(particles.count);
      gpu_j.resize(particles.count);
      gpu_volume.resize(particles.count);
      gpu_lambda.resize(particles.count);
      tbb::parallel_for(0, particles.count, [&](int i) {
        const TV &x = X_for_p[i];
        const std::size_t p = static_cast<std::size_t>(i);
        gpu_positions[p] = GPU_STAGE2::Vec3d{static_cast<double>(x[0]),
            static_cast<double>(x[1]), static_cast<double>(x[2])};
        gpu_j[p] = static_cast<double>((*J_pointer)[i]);
        gpu_volume[p] = static_cast<double>((*vol_pointer)[i]);
        gpu_lambda[p] = static_cast<double>((*j_model)[i].lambda);
      });
    }
    std::vector<GPU_STAGE2::SolidPressureGridNode> pressure_nodes;
    GPU_STAGE2::FluidVelocityGridStats pressure_stats;
    GPU_STAGE2::rebuildSolidPressureGrid(
        gpu_fluid_analytic, gpu_positions, gpu_j, gpu_volume, gpu_lambda,
        static_cast<double>(Base::dx), pressure_nodes, &pressure_stats,
        resident_particles.size ? &resident_particles : nullptr,
        &gpu_pressure_node_map,
        /*materialize_host_nodes*/ !gpu_full_pipeline_strict);
    if (!gpu_full_pipeline_strict) {
      gpu_pressure_nodes.resize(pressure_nodes.size());
      for (std::size_t i = 0; i < pressure_nodes.size(); ++i) {
        gpu_pressure_nodes[i].key = pressure_nodes[i].key;
        gpu_pressure_nodes[i].idx = static_cast<int>(i);
        gpu_pressure_nodes[i].mass = pressure_nodes[i].mass;
        gpu_pressure_nodes[i].v = GPU_STAGE3A::Vec3d{
            pressure_nodes[i].compliance, pressure_nodes[i].pressure, 0.0};
      }
      if constexpr (dim == 3) {
      std::vector<std::pair<uint64_t, std::size_t>> dof_order;
      dof_order.reserve(gpu_pressure_nodes.size());
      for (std::size_t i = 0; i < gpu_pressure_nodes.size(); ++i) {
        int x, y, z;
        GPU_STAGE2::unpackGridKey3D(gpu_pressure_nodes[i].key, x, y, z);
        dof_order.emplace_back(Base::SparseMask::Linear_Offset(
                               std::array<int, 3>{x, y, z}),
                           i);
      }
      tbb::parallel_sort(dof_order.begin(), dof_order.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      for (std::size_t rank = 0; rank < dof_order.size(); ++rank)
        gpu_pressure_nodes[dof_order[rank].second].idx = static_cast<int>(rank);
      }
    }
    num_p = static_cast<int>(gpu_full_pipeline_strict
        ? gpu_pressure_node_map.size : gpu_pressure_nodes.size());
    if (gpu_full_pipeline_strict) {
      ZIRAN_ASSERT(gpu_pressure_node_map.storage &&
                       gpu_pressure_node_map.size ==
                           static_cast<std::size_t>(num_p) &&
                       pressure_nodes.empty(),
          "strict full-GPU solid pressure DOFs materialized on host");
    }
    gpu_prebuilt_solid_pressure_valid = true;
    ZIRAN_INFO("GPU solid pressure grid: particles=",
               pressure_stats.input_samples,
               " nodes=", pressure_stats.nodes,
               " device_dofs=", gpu_pressure_node_map.size,
               " host_nodes=", pressure_nodes.size(),
               " resident_particles=",
               pressure_stats.resident_position_mass ? 1 : 0,
               " reallocations=", pressure_stats.reallocations,
               " total_ms=", pressure_stats.total_ms);
  } else {
  grid4.activatePages(X_for_p, Base::dx, 0, 1, -TV::Ones() * Base::dx * 0.5);
  Base::parallel_for_updating_grid([&](int i) {
    TV Xp = X_for_p[i];
    // TV Xp = Xarray[i];
    T J = (*J_pointer)[i];
    T vol0 = (*vol_pointer)[i];
    // BSplineWeights<T, dim, order> spline(Xp, Base::dx);
    BSplineWeights<T, dim, order> spline(Xp - TV::Ones() * Base::dx * 0.5,
                                         Base::dx);
    uint64_t offset =
        Base::SparseMask::Linear_Offset(to_std_array(spline.base_node));
    grid4.iterateKernel(spline, offset,
                        [&](IV node4, T w4, TV dw4, GridState<T, dim> &g4) {
                          T lambda = (*j_model)[i].lambda;
                          T p = -lambda * (J - 1);
                          g4.v[0] += vol0 / lambda * w4;
                          g4.v[1] += vol0 * J * p * w4;
                          g4.m += vol0 * J * w4;
                          // save data in v
                        });
  });
  num_p = grid4.getNumNodes();
  grid4.iterateGrid([&](IV node, GridState<T, dim> &g) { g.v[1] /= g.m; });
  // Diagnostic-only: retain the exact Q0 pressure-grid key/DOF map produced
  // above.  CouplingSimulation uses it to dump solved_p *before*
  // constructVelocityFromSolvedP; rebuilding a new grid afterwards is not a
  // valid substitute because its virtual-block numbering may differ.
  if constexpr (dim == 3) {
    if (std::getenv("ZIRAN_SOLID_PRESSURE_FIELD_DUMP_PREFIX")) {
      gpu_pressure_nodes.resize(static_cast<std::size_t>(num_p));
      grid4.iterateGridSerial([&](IV node, GridState<T, dim> &g) {
        auto& pressure_node =
            gpu_pressure_nodes[static_cast<std::size_t>(g.idx)];
        pressure_node.key =
            GPU_STAGE3A::packGridKey3D(node[0], node[1], node[2]);
        pressure_node.idx = g.idx;
      });
    }
  }
  }

  solidKernel<degree, order>(grid, grid4, M_inv, G, L, D, S, a, b);
  const bool resident_velocity_grid = gpu_solid_final_grid_device_ready &&
      gpu_solid_p2g_node_map.size == static_cast<std::size_t>(num_v);
  if (resident_velocity_grid) {
    const bool device_w = buildDeviceInterfaceW(2, false);
    if (gpu_full_pipeline_strict) {
      ZIRAN_ASSERT(device_w,
          "strict full-GPU solid interface W fell back to host");
    }
    if (!device_w)
      buildWMatrixFromGpuSolidVelocity();
  } else {
    ZIRAN_ASSERT(!gpu_full_pipeline_strict,
        "strict full-GPU solid velocity grid fell back to host");
    buildWMatrix(grid);
    num_v = grid.getNumNodes();
  }
  num_p = num_p;
}

template void SplittingSimulation<float, 2>::buildSolidSystem<2, 0>();
template void SplittingSimulation<double, 2>::buildSolidSystem<2, 0>();
template void SplittingSimulation<float, 3>::buildSolidSystem<2, 0>();
template void SplittingSimulation<double, 3>::buildSolidSystem<2, 0>();

// template void SplittingSimulation<float, 2>::buildSolidSystem<1, 0>();
// template void SplittingSimulation<double, 2>::buildSolidSystem<1, 0>();
// template void SplittingSimulation<float, 3>::buildSolidSystem<1, 0>();
// template void SplittingSimulation<double, 3>::buildSolidSystem<1, 0>();

template void SplittingSimulation<float, 2>::buildSolidSystem<2, 1>();
template void SplittingSimulation<double, 2>::buildSolidSystem<2, 1>();
template void SplittingSimulation<float, 3>::buildSolidSystem<2, 1>();
template void SplittingSimulation<double, 3>::buildSolidSystem<2, 1>();

} // namespace ZIRAN
