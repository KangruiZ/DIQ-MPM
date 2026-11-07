#include "MpmParticleRange.h"

#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <Ziran/Math/Geometry/AnalyticLevelSet.h>
#include <Ziran/Math/Geometry/Particles.h>
#include <MPM/Force/FJMixedForce.h>
#include <MPM/Force/MpmMultiTargetPinningForce.h>
#include <MPM/Force/MpmPinningForce.h>
#include <MPM/Force/MpmLagrangianContactForce.h>

namespace ZIRAN {

template <class T, int dim>
MpmParticleRange<T, dim>::MpmParticleRange(
    Particles<T, dim> &particles,
    MpmForceAssembly<T, dim> *mpmforce,
    StdVector<TV> &scratch_xp, T &dt, Range particle_range, T total_volume)
    : particles(particles), mpmforce(mpmforce), scratch_xp(scratch_xp), dt(dt),
      particle_range(particle_range), total_volume(total_volume) {}

template <class T, int dim>
void MpmParticleRange<T, dim>::transform(
    const std::function<void(int, Ref<T>, Vector<T, dim> &, Vector<T, dim> &)>
        &mapping) {
  for (int i = particle_range.lower; i < particle_range.upper; ++i) {
    // lua does not support passing scalars by reference. This is a work around
    // to actually change mass.
    mapping(i - particle_range.lower, particles.mass[i], particles.X[i],
            particles.V[i]);
  }
}

template <class T, int dim>
void MpmParticleRange<T, dim>::captureRestPositions() {
  // Create X0 as attribute
  particles.add(X0_name<TV>(), particle_range, TV::Zero());
  printf("-------------------------------add "
         "X0--------------------------------\n");
  // Fill in X0 with current X
  for (auto iter = particles.iter(X0_name<TV>()); iter; ++iter) {
    TV &X0 = iter.template get<0>();
    X0 = particles.X(iter.entryId());
  }
}

template <class T, int dim>
template <class FCONST, class JCONST>
void MpmParticleRange<T, dim>::addFJMaterial(
    const FCONST &f_model_input, const JCONST &j_model,
    MATERIAL_PHASE_ENUM phase, CONFIG_SCHEME config_scheme,
    MATERIAL_PROPERTY_ENUM property, bool linear_corotated) {
  FJMixedForce<T, dim> &helper =
      mpmforce->materialForce(); // this will let mpmforce create the constitutive
                             // (FJMixed) force
  helper.config_scheme = config_scheme;
  helper.linear_corotated = linear_corotated;
  FCONST f_model = f_model_input;
  if (phase == MATERIAL_PHASE_FLUID) {
    f_model.mu = 0;
    f_model.lambda = 0;
  }
  particles.add(helper.template f_constitutive_model_name<FCONST>(),
                particle_range, f_model);
  particles.add(helper.j_constitutive_model_name(), particle_range, j_model);
  if (total_volume != 0) {
    particles.add(element_measure_name<T>(), particle_range,
                  total_volume / particle_range.length());
    particles.add(J_name<T>(), particle_range, (T)1);
    particles.add(F_name<T, dim>(), particle_range, TM::Identity());
    particles.add(F_sn_name<T, dim>(), particle_range, TM::Identity());
    particles.add(F_s_name<T, dim>(), particle_range, TM::Identity());
  }
  particles.add(helper.template f_constitutive_model_scratch_name<FCONST>(),
                particle_range, typename FCONST::Scratch());
  particles.add(helper.j_constitutive_model_scratch_name(), particle_range,
                typename JCONST::Scratch());
  particles.add(material_phase_name(), particle_range, phase);
  particles.add(material_property_name(), particle_range, property);
  particles.add(config_scheme_name(), particle_range, config_scheme);
}

template <class T, int dim>
void MpmParticleRange<T, dim>::addTotalLagrangianContacts(
    CONFIG_SCHEME config_scheme) {

  ZIRAN_ASSERT(config_scheme != CONFIG_UPDATE_LAGRANGIAN,
               "Do not allow UPDATE LAGRANGIAN CONFIG");
  ZIRAN_ASSERT(!already_add_tlcontacts_particles,
               "addTLContacts should only be called once per handle.");
  already_add_tlcontacts_particles = true;

  const StdVector<TV> &X0Array =
      particles.DataManager::get(X0_name<TV>()).array;
  if (mpmforce->lagrangian_contact == nullptr) {
    mpmforce->lagrangian_contact =
        std::make_unique<MpmLagrangianContactForce<T, dim>>(
            particles, mpmforce->scratch_hash, X0Array, scratch_xp, dt,
            mpmforce->dx);
    mpmforce->has_lagrangian_contact = true;
  }
  MpmLagrangianContactForce<T, dim> *helper =
      mpmforce->lagrangian_contact.get();

  DisjointRanges ranges;
  for (int i = particle_range.lower; i < particle_range.upper; i++) {
    ranges.append(Range{i, i + 1});
  }

  auto &scratch_neighbour =
      particles.DataManager::get(helper->neighbour_name());
  DisjointRanges dr = scratch_neighbour.ranges;
  dr.merge(ranges);
  scratch_neighbour.lazyResize(dr);
}

template <class T, int dim>
void MpmParticleRange<T, dim>::pinParticlesInLevelSet(
    AnalyticLevelSet<T, dim> &levelset, T stiffness, T damping) {
  DisjointRanges dr;
  int count = 0;
  for (int i = particle_range.lower; i < particle_range.upper; i++) {
    if (levelset.inside((particles.X[i]))) {
      dr.append(Range{i, i + 1});
      count++;
    }
  }
  ZIRAN_INFO("pinParticlesInLevelSet pinned ", count, " particles.");

  pinParticles(dr, stiffness, damping);
}

template <class T, int dim>
void MpmParticleRange<T, dim>::addMultiTargetPinningInLevelSet(
    AnalyticLevelSet<T, dim> &levelset, T stiffness, T damping,
    const bool is_collider_spring) {
  DisjointRanges dr;
  int count = 0;
  for (int i = particle_range.lower; i < particle_range.upper; i++) {
    if (levelset.inside((particles.X[i]))) {
      dr.append(Range{i, i + 1});
      count++;
    }
  }
  ZIRAN_INFO("pinParticlesInLevelSet pinned ", count, " particles.");

  addMultiTargetPinning(dr, stiffness, damping, is_collider_spring);
}

template <class T, int dim>
void MpmParticleRange<T, dim>::pinParticles(const DisjointRanges &ranges,
                                                 const T stiffness,
                                                 const T damping) {
  ZIRAN_ASSERT(!already_pinned_particles,
               "pinParticels should only be called once per handle.");
  already_pinned_particles = true;

  const StdVector<TV> &X0Array =
      particles.DataManager::get(X0_name<TV>()).array;
  if (mpmforce->pinning == nullptr)
    mpmforce->pinning = std::make_unique<MpmPinningForce<T, dim>>(
        particles, X0Array, scratch_xp, dt);
  MpmPinningForce<T, dim> *helper = mpmforce->pinning.get();

  auto &target = particles.DataManager::get(helper->target_name());
  DisjointRanges dr = target.ranges;
  dr.merge(ranges);
  target.lazyResize(dr);

  for (auto iter = particles.subsetIter(ranges, helper->target_name(),
                                        Particles<T, dim>::X_name(),
                                        Particles<T, dim>::mass_name());
       iter; ++iter) {
    PinningTarget<T, dim> &t = iter.template get<0>();
    const TV &x = iter.template get<1>();
    const T mass = iter.template get<2>();
    t.k = stiffness * mass;
    t.d = damping * mass;
    t.x = x;
    t.v.setZero();
    t.X = t.x;
  }
}

template <class T, int dim>
void MpmParticleRange<T, dim>::addMultiTargetPinning(
    const DisjointRanges &ranges, const T stiffness, const T damping,
    const bool is_collider_spring) {
  ZIRAN_ASSERT(!has_multi_target_pinning,
               "pinParticels should only be called once per handle.");
  has_multi_target_pinning = true;

  const StdVector<TV> &X0Array =
      particles.DataManager::get(X0_name<TV>()).array;
  if (mpmforce->multi_target_pinning == nullptr)
    mpmforce->multi_target_pinning =
        std::make_unique<MpmMultiTargetPinningForce<T, dim>>(
            particles, X0Array, scratch_xp, dt, is_collider_spring);
  MpmMultiTargetPinningForce<T, dim> *helper =
      mpmforce->multi_target_pinning.get();

  auto &target = particles.DataManager::get(helper->target_name());
  DisjointRanges dr = target.ranges;
  dr.merge(ranges);
  target.lazyResize(dr);

  for (auto iter = particles.subsetIter(ranges, helper->target_name(),
                                        Particles<T, dim>::X_name(),
                                        Particles<T, dim>::mass_name());
       iter; ++iter) {
    MultiTargetPinningTarget<T, dim> &t = iter.template get<0>();
    const TV &x = iter.template get<1>();
    const T mass = iter.template get<2>();
    t.k = stiffness * mass;
    t.d = damping * mass;
    t.x.clear();
    t.v.clear();
  }
}

} // namespace ZIRAN
