#pragma once

#include "MpmForceTerm.h"

#include <Ziran/CS/DataStructure/DataArray.h>
#include <Ziran/CS/DataStructure/DataManager.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/Math/Geometry/Particles.h>

// Include the CRTP base first: several model headers (e.g. CorotatedIsotropic)
// are not self-contained and rely on HyperelasticTraits already being visible.
#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>
#include <Ziran/Physics/ConstitutiveModel/CorotatedIsotropic.h>
#include <Ziran/Physics/ConstitutiveModel/LinearCorotated.h>
#include <Ziran/Physics/ConstitutiveModel/PDIsotropic.h>
#include <Ziran/Physics/ConstitutiveModel/QuadraticVolumePenalty.h>

namespace ZIRAN {

/**
   Mixed F/J constitutive force: F is the deformation gradient and J is the
   volume ratio. Each operation dispatches to the elastic models present on
   the particles; FJMixedFModels.def defines that shared model list. The volume
   term uses QuadraticVolumePenalty. Attribute keys belong to the restart format.
*/
template <class T, int dim>
class FJMixedForce : public MpmForceTerm<T, dim> {
public:
  typedef Matrix<T, dim, dim> TM;
  typedef Vector<T, dim> TV;
  using TVStack = Matrix<T, dim, Eigen::Dynamic>;

  using JCONST = QuadraticVolumePenalty<T, dim>;
  using JScratch = typename JCONST::Scratch;

  // The supported F constitutive models live in the shared X-macro list
  // FJMixedFModels.def (also used by the addFJMaterial instantiations).
  // Each method below does:  #define FJMIXED_F_MODEL(M) <do something with M>
  //                          #include "FJMixedFModels.def"
  //                          #undef FJMIXED_F_MODEL

  Particles<T, dim> &particles;
  DataArray<T> &element_measure;
  DataArray<TM> &F;
  DataArray<TM> &F_s;
  DataArray<TM> &F_sn;
  DataArray<T> &J;
  StdVector<TM> Fn;
  StdVector<TM> Fn_sn;
  StdVector<T> Jn;
  CONFIG_SCHEME config_scheme;

  bool linear_corotated = false;

  explicit FJMixedForce(Particles<T, dim> &particles)
      : particles(particles),
        element_measure(particles.add(element_measure_name())),
        F(particles.add(F_name())), F_s(particles.add(F_s_name())),
        F_sn(particles.add(F_sn_name())), J(particles.add(J_name())) {}
  virtual ~FJMixedForce() {}

  void reinitialize() override {}

  // ---- attribute names (F ones are templated on the model) ----
  template <class FCONST>
  inline static AttributeName<FCONST> f_constitutive_model_name() {
    return AttributeName<FCONST>(FCONST::name());
  }
  template <class FCONST>
  inline static AttributeName<typename FCONST::Scratch>
  f_constitutive_model_scratch_name() {
    return AttributeName<typename FCONST::Scratch>(FCONST::scratch_name());
  }
  inline static AttributeName<JCONST> j_constitutive_model_name() {
    return AttributeName<JCONST>(JCONST::name());
  }
  inline static AttributeName<JScratch> j_constitutive_model_scratch_name() {
    return AttributeName<JScratch>(JCONST::scratch_name());
  }
  inline static AttributeName<T> element_measure_name() {
    return AttributeName<T>("element measure");
  }
  inline static AttributeName<TM> F_name() { return AttributeName<TM>("F"); }
  inline static AttributeName<TM> F_s_name() {
    return AttributeName<TM>("F_s");
  } // it means F^{0,s}
  inline static AttributeName<TM> F_sn_name() {
    return AttributeName<TM>("F_sn");
  } // it means F^{0,s}
  inline static AttributeName<T> J_name() { return AttributeName<T>("J"); }

  // ============================ public force interface ======================
  // Each method runs the per-model body for every F model present on particles.

  void backupStrain() override {
    if (config_scheme == CONFIG_AU_LAGRANGIAN) {
      Fn_sn.resize(F_sn.array.size());
      particles.parallel_for(
          [&](auto &FF, int FF_index) { Fn_sn[FF_index] = FF; }, F_sn_name(),
          valueIdOnly(F_sn_name()));
    } else {
      Fn.resize(F.array.size());
      particles.parallel_for(
          [&](auto &FF, int FF_index) { Fn[FF_index] = FF; }, F_name(),
          valueIdOnly(F_name()));
    }

    Jn = J.array;
    if (linear_corotated) {
#define FJMIXED_F_MODEL(M)                                                    \
  if (particles.exist(f_constitutive_model_name<M<T, dim>>()))                \
    rebuildRImpl<M<T, dim>>();
#include "FJMixedFModels.def"
#undef FJMIXED_F_MODEL
    }
  }

  void restoreStrain() {
    if (config_scheme == CONFIG_AU_LAGRANGIAN) {
      particles.parallel_for(
          [&](auto &FF, int FF_index) { FF = Fn_sn[FF_index]; }, F_sn_name(),
          valueIdOnly(F_sn_name()));
    } else {
      particles.parallel_for(
          [&](auto &FF, int FF_index) { FF = Fn[FF_index]; }, F_name(),
          valueIdOnly(F_name()));
    }
    J.array = Jn;
  }

  void updateState(const DisjointRanges &subrange, StdVector<TM> &force_stress,
                   TVStack &fp) {
#define FJMIXED_F_MODEL(M)                                                    \
  if (particles.exist(f_constitutive_model_name<M<T, dim>>()))                \
    updateStateFImpl<M<T, dim>>(subrange, force_stress, fp);
#include "FJMixedFModels.def"
#undef FJMIXED_F_MODEL
  }

  void updateImplicitState(const DisjointRanges &subrange,
                           StdVector<TM> &force_stress, TVStack &fp) {
#define FJMIXED_F_MODEL(M)                                                    \
  if (particles.exist(f_constitutive_model_name<M<T, dim>>()))                \
    updateImplicitStateFImpl<M<T, dim>>(subrange, force_stress, fp);
#include "FJMixedFModels.def"
#undef FJMIXED_F_MODEL
  }

  void evolveStrain(const DisjointRanges &subrange, T dt,
                    const StdVector<TM> &gradV) {
#define FJMIXED_F_MODEL(M)                                                    \
  if (particles.exist(f_constitutive_model_name<M<T, dim>>()))                \
    evolveStrainImpl<M<T, dim>>(subrange, dt, gradV);
#include "FJMixedFModels.def"
#undef FJMIXED_F_MODEL
  }

  double totalEnergy(const DisjointRanges &subrange) {
    double e = 0.0;
#define FJMIXED_F_MODEL(M)                                                    \
  if (particles.exist(f_constitutive_model_name<M<T, dim>>()))                \
    e += totalEnergyImpl<M<T, dim>>(subrange);
#include "FJMixedFModels.def"
#undef FJMIXED_F_MODEL
    return e;
  }

  void computeStressDifferential(const DisjointRanges &subrange,
                                 const StdVector<TM> &gradDv,
                                 StdVector<TM> &dstress, const TVStack &dvp,
                                 TVStack &dfp) {
#define FJMIXED_F_MODEL(M)                                                    \
  if (particles.exist(f_constitutive_model_name<M<T, dim>>()))                \
    computeStressDifferentialFImpl<M<T, dim>>(subrange, gradDv, dstress, dvp, dfp);
#include "FJMixedFModels.def"
#undef FJMIXED_F_MODEL
  }

private:
  // ======================= per-model implementations ========================
  // Bodies are unchanged from the old FJMixedMpmForceHelper<FCONST, QVP>.

  template <class FCONST> void rebuildRImpl() {
    auto ranges = particles.X.ranges;
    tbb::parallel_for(ranges, [&](DisjointRanges &subrange) {
      DisjointRanges subset(
          subrange,
          particles.commonRanges(f_constitutive_model_name<FCONST>(), F_name()));
      for (auto iter = particles.subsetIter(
               subset, f_constitutive_model_name<FCONST>(), F_name());
           iter; ++iter) {
        auto &constitutive_model = iter.template get<0>();
        const auto &F = iter.template get<1>();
        constitutive_model.rebuildR(F);
      }
    });
  }

  template <class FCONST>
  void updateStateFImpl(const DisjointRanges &subrange,
                        StdVector<TM> &force_stress, TVStack &fp) {
    DisjointRanges subset(
        subrange, particles.commonRanges(
                      f_constitutive_model_name<FCONST>(),
                      j_constitutive_model_name(), element_measure_name(),
                      F_name(), F_s_name(), J_name(), config_scheme_name()));
    for (auto iter = particles.subsetIter(
             subset, f_constitutive_model_name<FCONST>(),
             f_constitutive_model_scratch_name<FCONST>(),
             element_measure_name(), F_name(), F_s_name(),
             config_scheme_name());
         iter; ++iter) {
      auto &constitutive_model = iter.template get<0>();
      auto &scratch = iter.template get<1>();
      const auto &element_measure = iter.template get<2>();
      const auto &F = iter.template get<3>();
      const auto &F_s = iter.template get<4>();
      auto &config_scheme = iter.template get<5>();
      int p = iter.entryId();
      if (linear_corotated)
        constitutive_model.rebuildR(F);
      constitutive_model.updateScratch(F, scratch);
      TM force_stress_local = TM::Zero();
      if (config_scheme == CONFIG_TOTAL_LAGRANGIAN)
        constitutive_model.firstPiola(scratch, force_stress_local);
      else if (config_scheme == CONFIG_AU_LAGRANGIAN) {
        constitutive_model.firstPiola(scratch, force_stress_local);
        force_stress_local = force_stress_local * F_s.transpose();
      } else if (config_scheme == CONFIG_UPDATE_LAGRANGIAN)
        constitutive_model.kirchhoff(scratch, force_stress_local);
      force_stress_local *= element_measure;
      assert(force_stress_local == force_stress_local);
      force_stress[p] += force_stress_local;
    }
  }

  template <class FCONST>
  void updateImplicitStateFImpl(const DisjointRanges &subrange,
                                StdVector<TM> &force_stress, TVStack &fp) {
    DisjointRanges subset(
        subrange, particles.commonRanges(
                      f_constitutive_model_name<FCONST>(),
                      j_constitutive_model_name(), element_measure_name(),
                      J_name(), config_scheme_name(), F_s_name()));
    if (subset.size() == 0)
      return;
    for (auto iter = particles.subsetIter(
             subset, f_constitutive_model_name<FCONST>(),
             f_constitutive_model_scratch_name<FCONST>(),
             element_measure_name(), F_name(), valueIdOnly(F_name()),
             config_scheme_name(), F_s_name());
         iter; ++iter) {
      auto &constitutive_model = iter.template get<0>();
      auto &scratch = iter.template get<1>();
      const auto &element_measure = iter.template get<2>();
      const auto &F = iter.template get<3>();
      const int F_index = iter.template get<4>();
      auto &config_scheme = iter.template get<5>();
      const auto &F_s = iter.template get<6>();
      int p = iter.entryId();
      constitutive_model.updateScratch(F, scratch);
      TM force_stress_local;
      constitutive_model.firstPiola(scratch, force_stress_local);
      if (config_scheme == CONFIG_TOTAL_LAGRANGIAN) {
        force_stress_local = element_measure * force_stress_local;
      } else if (config_scheme == CONFIG_AU_LAGRANGIAN) {
        force_stress_local =
            element_measure * force_stress_local * F_s.transpose();
      } else if (config_scheme == CONFIG_UPDATE_LAGRANGIAN) {
        force_stress_local =
            element_measure * force_stress_local * Fn[F_index].transpose();
      }
      assert(force_stress_local == force_stress_local);
      force_stress[p] += force_stress_local;
    }
  }

  template <class FCONST>
  void evolveStrainImpl(const DisjointRanges &subrange, T dt,
                        const StdVector<TM> &gradV) {
    DisjointRanges subset(
        subrange,
        particles.commonRanges(
            f_constitutive_model_name<FCONST>(),
            f_constitutive_model_scratch_name<FCONST>(),
            j_constitutive_model_name(), j_constitutive_model_scratch_name(),
            element_measure_name(), F_name(), F_s_name(), F_sn_name(),
            material_phase_name(), config_scheme_name()));
    for (auto iter = particles.subsetIter(
             subset, f_constitutive_model_name<FCONST>(), F_name(), F_s_name(),
             F_sn_name(), J_name(), material_phase_name(),
             j_constitutive_model_name(), config_scheme_name());
         iter; ++iter) {
      auto &constitutive_model_F = iter.template get<0>();
      auto &F = iter.template get<1>();
      auto &F_s = iter.template get<2>();
      auto &F_sn = iter.template get<3>();
      auto &J = iter.template get<4>();
      auto &phase = iter.template get<5>();
      auto &constitutive_model_J = iter.template get<6>();
      auto &config_scheme_name = iter.template get<7>();
      int p = iter.entryId();
      if (phase == MATERIAL_PHASE_FLUID)
        continue;

      if (config_scheme_name == CONFIG_TOTAL_LAGRANGIAN) {
        F = (F + ((T)dt) * gradV[p]);
        J = F.determinant();
      } else if (config_scheme_name == CONFIG_AU_LAGRANGIAN) {
        F_sn = (F_sn + ((T)dt) * gradV[p]);
        F = F_sn * F_s;
        J = F.determinant();
      } else if (config_scheme_name == CONFIG_UPDATE_LAGRANGIAN) {
        F = (TM::Identity() + ((T)dt) * gradV[p]) * F;
        J = (1 + ((T)dt) * gradV[p].trace()) * J;
      }

      assert(gradV[p] == gradV[p]);
      constitutive_model_J.lambda =
          constitutive_model_F.lambda; // some plasticity modifies lambda in the
                                       // F model
    }
  }

  template <class FCONST>
  double totalEnergyImpl(const DisjointRanges &subrange) {
    DisjointRanges subset(
        subrange, particles.commonRanges(f_constitutive_model_name<FCONST>(),
                                         j_constitutive_model_name(),
                                         element_measure_name()));
    double e = 0.0;
    for (auto iter = particles.subsetIter(
             subset, f_constitutive_model_name<FCONST>(),
             f_constitutive_model_scratch_name<FCONST>(), F_name(),
             element_measure_name());
         iter; ++iter) {
      auto &constitutive_model = iter.template get<0>();
      auto &scratch = iter.template get<1>();
      const TM &F = iter.template get<2>();
      const auto &element_measure = iter.template get<3>();
      constitutive_model.updateScratch(F, scratch);
      e += element_measure * constitutive_model.psi(scratch);
    }
    return e;
  }

  template <class FCONST>
  void computeStressDifferentialFImpl(const DisjointRanges &subrange,
                                      const StdVector<TM> &gradDv,
                                      StdVector<TM> &dstress, const TVStack &dvp,
                                      TVStack &dfp) {
    DisjointRanges subset(
        subrange, particles.commonRanges(
                      f_constitutive_model_name<FCONST>(),
                      j_constitutive_model_name(), element_measure_name(),
                      config_scheme_name(), F_s_name()));
    if (subset.size() == 0)
      return;
    for (auto iter = particles.subsetIter(
             subset, f_constitutive_model_name<FCONST>(),
             f_constitutive_model_scratch_name<FCONST>(),
             element_measure_name(), valueIdOnly(F_name()),
             config_scheme_name(), F_s_name());
         iter; ++iter) {
      const auto &constitutive_model = iter.template get<0>();
      auto &scratch = iter.template get<1>();
      const auto &element_measure = iter.template get<2>();
      int F_index = iter.template get<3>();
      auto &config_scheme_name = iter.template get<4>();
      const auto &F_s = iter.template get<5>();
      int p = iter.entryId();
      TM dP;
      const auto &Fn_local = Fn[F_index];
      if (config_scheme_name == CONFIG_TOTAL_LAGRANGIAN) {
        constitutive_model.firstPiolaDifferential(scratch, gradDv[p], dP);
        assert(dP == dP);
        dstress[p] += dP * element_measure;
      } else if (config_scheme_name == CONFIG_AU_LAGRANGIAN) {
        constitutive_model.firstPiolaDifferential(scratch, gradDv[p] * F_s, dP);
        dstress[p] += dP * element_measure * F_s.transpose();
      } else if (config_scheme_name == CONFIG_UPDATE_LAGRANGIAN) {
        constitutive_model.firstPiolaDifferential(scratch, gradDv[p] * Fn_local,
                                                  dP);
        assert(dP == dP);
        dstress[p] += dP * element_measure * Fn_local.transpose();
      }
    }
  }

};
} // namespace ZIRAN
