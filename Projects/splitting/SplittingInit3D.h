#pragma once

#include "SplittingInit.h"
#include "SplittingSimulation.h"
#include <Ziran/CS/Util/RandomNumber.h>
#include <stdexcept>

namespace ZIRAN {

template <class T, int dim> class SplittingInitBase;

template <class T> class SplittingInit3D : public SplittingInitBase<T, 3> {
public:
  static const int dim = 3;
  using Base = SplittingInitBase<T, dim>;
  using TV2 = Vector<T, 2>;
  using TVI2 = Vector<int, 2>;
  using TV = Vector<T, dim>;
  using TM = Eigen::Matrix<T, dim, dim>;
  using TVI = Vector<int, dim>;

  using Base::init_helper;
  using Base::sim;
  using Base::test_number;

  SplittingInit3D(SplittingSimulation<T, dim> &sim, const int test_number)
      : Base(sim, test_number) {}
#define SET_MPM_VALUE(A, B) sim.A = B

  void set_path(std::string path) { sim.output_dir.path = path; }

  void reload() override {
    if (test_number == 1) {
      set_path("output/splitting/1_colliding_armadillos");
      sim.step.frame_dt = 1. / 48.;
      SET_MPM_VALUE(end_frame, 200);
      SET_MPM_VALUE(dx, (T)1.5 / (T)128);
      SET_MPM_VALUE(cfl, 0.4);
      sim.apic_rpic_ratio = 1;
      T solid_density = 1000;
      T ppc = 8;
      T E = 60000, nu = 0.3;
      SET_MPM_VALUE(step.max_dt, 0.001);
      SET_MPM_VALUE(gravity, -9.8 * TV::Unit(1));
      SET_MPM_VALUE(mls_mpm, false);
      SET_MPM_VALUE(ignoreCollisionObject, true);
      sim.setSolver(1e-3, -1, 10000);
      sim.use_position_correction = true;
      sim.category = MATERIAL_PHASE_SOLID;
      sim.config_scheme = CONFIG_TOTAL_LAGRANGIAN;
      sim.symplectic = false;
      sim.use_vp = true;
      sim.newton.max_iterations = 5;
      sim.newton.tolerance = 1e-2;
      sim.newton.linesearch = false;
      sim.objective.switchLinearSolver(
          ZIRAN::SimulationBase::LinearSolverType(1)); // CG
      sim.objective.linear_solver->max_iterations = 10000;
      sim.objective.linear_solver->tolerance = 1e-6;
      sim.implicit_penalty_collision =
          solid_density * std::pow((T)sim.dx, (T)dim) / (T)ppc * 1000000;
      sim.boundary_ppc = ppc;
      PDIsotropic<T, dim> model_projective(E, nu);
      QuadraticVolumePenalty<T, dim> model2(E, nu);
      TV scene_shift = 5 * sim.dx * TV::Ones();

      {

        MpmParticleRange<T, dim> p_handle = init_helper.sampleFromVdbFile(
            "LevelSets/armadillo1.vdb", solid_density, ppc);
        p_handle.transform([&](int index, Ref<T> mass, TV &X, TV &V) {
          X += TV(-0.2, -0.1, 0.1);
        });
        p_handle.addFJMaterial(
            model_projective, model2, MATERIAL_PHASE_SOLID, sim.config_scheme,
            MATERIAL_PROPERTY_COMPRESSIBLE, /*linear corotated*/ true);
        p_handle.captureRestPositions();
        Sphere<T, dim> dummy(TV(0, 0), 100);
        p_handle.addMultiTargetPinningInLevelSet(dummy, 0, 0,
                                                /*collider spring mode*/ true);

        MpmParticleRange<T, dim> p_handle1 = init_helper.sampleFromVdbFile(
            "LevelSets/armadillo1.vdb", solid_density, ppc);
        p_handle1.transform([&](int index, Ref<T> mass, TV &X, TV &V) {
          X += TV(-0.2, 0.65, 0.1);
        });
        p_handle1.addFJMaterial(
            model_projective, model2, MATERIAL_PHASE_SOLID, sim.config_scheme,
            MATERIAL_PROPERTY_COMPRESSIBLE, /*linear corotated*/ true);
        p_handle1.captureRestPositions();
        p_handle1.addMultiTargetPinningInLevelSet(dummy, 0, 0,
                                                 /*collider spring mode*/ true);

        MpmParticleRange<T, dim> p_handle2 = init_helper.sampleFromVdbFile(
            "LevelSets/armadillo1.vdb", solid_density, ppc);
        p_handle2.transform([&](int index, Ref<T> mass, TV &X, TV &V) {
          X += TV(-0.2, 1.4, 0.1);
        });
        p_handle2.addFJMaterial(
            model_projective, model2, MATERIAL_PHASE_SOLID, sim.config_scheme,
            MATERIAL_PROPERTY_COMPRESSIBLE, /*linear corotated*/ true);
        p_handle2.captureRestPositions();
        p_handle2.addMultiTargetPinningInLevelSet(dummy, 0, 0,
                                                 /*collider spring mode*/ true);

        p_handle.addTotalLagrangianContacts(sim.config_scheme);
        p_handle1.addTotalLagrangianContacts(sim.config_scheme);
        p_handle2.addTotalLagrangianContacts(sim.config_scheme);
      }
      AxisAlignedAnalyticBox<T, dim> box1(
          TV(0.55, 0, 0.2) - 4 * sim.dx * TV::Ones() + scene_shift,
          TV(1.35, 4, 0.95) + 4 * sim.dx * TV::Ones() + scene_shift);
      AxisAlignedAnalyticBox<T, dim> box2(
          TV(0.55, 0, 0.2) + 4 * sim.dx * TV::Ones() + scene_shift,
          TV(1.35, 4, 0.95) - 4 * sim.dx * TV::Ones() + scene_shift);
      DifferenceLevelSet<T, dim> ls1, ls2;
      ls1.add(box1, box2);
      AnalyticCollisionObject<T, dim> container_object(
          ls1, AnalyticCollisionObject<T, dim>::GHOST);
      init_helper.addAnalyticCollisionObject(container_object);
      return;
    }
    throw std::invalid_argument(
        "Unknown example. Use -test 1 for the Armadillo collision.");
  }
};

#undef SET_MPM_VALUE

} // namespace ZIRAN
