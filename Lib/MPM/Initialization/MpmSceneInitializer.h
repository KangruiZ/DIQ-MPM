#ifndef MPM_SCENE_INITIALIZER_H
#define MPM_SCENE_INITIALIZER_H

#include <MPM/Initialization/MpmParticleRange.h>
#include <MPM/Simulation/MpmSimulationBase.h>
#include <Ziran/Math/Geometry/CollisionObject.h>
#include <Ziran/Math/Geometry/PoissonDisk.h>
#include <Ziran/Math/Geometry/VdbLevelSet.h>

namespace ZIRAN {

// Samples scene particles and registers analytic colliders on a shared simulation.
template <class T, int dim> class MpmSceneInitializer {
public:
  using TV = Vector<T, dim>;
  MpmSimulationBase<T, dim>& mpm;

  MpmSceneInitializer(MpmSimulationBase<T, dim>& mpm) : mpm(mpm) {}

  MpmParticleRange<T, dim>
  sampleFromVdbFile(std::string filename, T density,
                    T particles_per_cell = (1 << dim)) {
    VdbLevelSet<float, dim> vdbls(
        filename); // houdini's vdb files are stored with floats
    Vector<float, dim> min_corner, max_corner;
    vdbls.getBounds(min_corner, max_corner);

    PoissonDisk<T, dim> pd(/*random seed*/ 123, 0,
                           min_corner.template cast<T>(),
                           max_corner.template cast<T>());
    pd.setDistanceByParticlesPerCell(mpm.dx, particles_per_cell * 2);
    AxisAlignedAnalyticBox<T, dim> box(min_corner.template cast<T>(),
                                       max_corner.template cast<T>());
    StdVector<TV> unscaled_samples;
    if (dim == 3)
      pd.sampleFromPeriodicData(unscaled_samples,
                                [&](const TV &x) { return box.inside(x); });
    else
      pd.sample(unscaled_samples, [&](const TV &x) { return box.inside(x); });
    T real_ppc = (T)unscaled_samples.size() /
                 ((max_corner - min_corner).template cast<T>() / mpm.dx).prod();
    ZIRAN_ASSERT(real_ppc > particles_per_cell,
                 "Samples particles should be more dense.");
    T scale = std::pow(real_ppc / particles_per_cell, (T)1 / (T)dim);
    StdVector<TV> full_samples;
    for (auto X : unscaled_samples) {
      TV center = (min_corner + max_corner).template cast<T>() / (T)2;
      TV scaled_X = (X - center) * scale + center;
      if (box.inside(scaled_X))
        full_samples.push_back(scaled_X);
    }
    T per_particle_volume = box.volume() / (T)full_samples.size();

    StdVector<TV> samples;
    for (auto x : full_samples)
      if (vdbls.inside(x.template cast<float>()))
        samples.push_back(x);
    size_t N = samples.size();
    Range particle_range = mpm.particles.getNextRange(samples.size());
    mpm.particles.add(mpm.particles.X_name(), particle_range,
                      std::move(samples));
    mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
    mpm.particles.add(mpm.particles.mass_name(), particle_range,
                      per_particle_volume * density);
    return MpmParticleRange<T, dim>(
        mpm.getParticles(), mpm.getMpmForce(),
        mpm.getScratchXp(), mpm.getDt(),
        particle_range, per_particle_volume * N);
  }

  T sampleInAnalyticLevelSetHelper(AnalyticLevelSet<T, dim> &levelset,
                                   T particles_per_cell,
                                   StdVector<Vector<T, dim>> &samples) {
    using TV = Vector<T, dim>;
    ZIRAN_INFO("Sampling ", particles_per_cell,
               " particles per cell in the levelset");
    TV min_corner, max_corner;
    levelset.getBounds(min_corner, max_corner);

    PoissonDisk<T, dim> pd(/*random seed*/ 123, 0, min_corner, max_corner);
    pd.setDistanceByParticlesPerCell(mpm.dx, particles_per_cell * 2);
    AxisAlignedAnalyticBox<T, dim> box(min_corner, max_corner);
    StdVector<TV> unscaled_samples;
    if (dim == 3)
      pd.sampleFromPeriodicData(unscaled_samples,
                                [&](const TV &x) { return box.inside(x); });
    else
      pd.sample(unscaled_samples, [&](const TV &x) { return box.inside(x); });
    T real_ppc = (T)unscaled_samples.size() /
                 ((max_corner - min_corner) / mpm.dx).prod();
    ZIRAN_ASSERT(real_ppc > particles_per_cell,
                 "Samples particles should be more dense.");
    T scale = std::pow(real_ppc / particles_per_cell, (T)1 / (T)dim);
    StdVector<TV> scaled_samples_in_box;
    for (auto X : unscaled_samples) {
      TV center = (min_corner + max_corner) / (T)2;
      TV scaled_X = (X - center) * scale + center;
      if (levelset.inside(scaled_X))
        samples.push_back(scaled_X);
      if (box.inside(scaled_X))
        scaled_samples_in_box.push_back(scaled_X);
    }
    T per_particle_volume = box.volume() / (T)scaled_samples_in_box.size();

    T total_volume = per_particle_volume * (T)samples.size();

    return total_volume;
  }

  MpmParticleRange<T, dim>
  sampleInAnalyticLevelSet(AnalyticLevelSet<T, dim> &levelset, T density,
                           T particles_per_cell) {
    StdVector<TV> samples;
    T total_volume =
        sampleInAnalyticLevelSetHelper(levelset, particles_per_cell, samples);
    size_t N = samples.size();
    Range particle_range = mpm.particles.getNextRange(samples.size());
    mpm.particles.add(mpm.particles.X_name(), particle_range,
                      std::move(samples));
    mpm.particles.add(mpm.particles.V_name(), particle_range, TV::Zero());
    mpm.particles.add(mpm.particles.mass_name(), particle_range,
                      total_volume * density / N);
    return MpmParticleRange<T, dim>(
        mpm.getParticles(), mpm.getMpmForce(),
        mpm.getScratchXp(), mpm.getDt(),
        particle_range, total_volume);
  }

  void addAnalyticCollisionObject(AnalyticCollisionObject<T, dim> &c) {
    mpm.addCollisionObject(std::move(c));
  }
};
} // namespace ZIRAN

#endif
