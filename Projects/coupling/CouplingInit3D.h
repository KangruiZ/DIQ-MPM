#pragma once

#include "CouplingInit.h"
#include "CouplingSimulation.h"
#include <Ziran/CS/Util/RandomNumber.h>
#include <stdexcept>

namespace ZIRAN {

template <class T, int dim> class CouplingInitBase;

template <class T> class CouplingInit3D : public CouplingInitBase<T, 3> {
public:
  static const int dim = 3;
  using Base = CouplingInitBase<T, dim>;
  using TV2 = Vector<T, 2>;
  using TVI2 = Vector<int, 2>;
  using TV = Vector<T, dim>;
  using TM = Eigen::Matrix<T, dim, dim>;
  using TVI = Vector<int, dim>;

  using Base::init_helper1;
  using Base::init_helper2;
  using Base::sim1;
  using Base::sim2;
  using Base::sim_coordinator;
  using Base::test_number;
  CouplingInit3D(CouplingSimulation<T, dim> &sim, const int test_number)
      : Base(sim, test_number) {}

#define SET_SIM_VALUE(A, B) sim_coordinator.A = sim1.A = sim2.A = B
#define SET_MPM_VALUE(A, B) sim1.A = sim2.A = B

  void set_path(std::string path) {
    sim_coordinator.output_dir.path = path;
    sim1.output_dir.path = path + "/material1";
    sim2.output_dir.path = path + "/material2";
  }

  // Every solid phase uses the same CG linear solver (type 1) with these
  // fixed iteration/tolerance settings; the examples only vary the Newton
  // iterations/tolerance/linesearch around it.
  template <class Sim> void configureSolidLinearSolver(Sim &e) {
    e.objective.switchLinearSolver(ZIRAN::SimulationBase::LinearSolverType(1));
    e.objective.linear_solver->max_iterations = 10000;
    e.objective.linear_solver->tolerance = 1e-6;
  }

  // Fluid phase Minres settings shared by every example.
  template <class Sim> void configureFluidLinearSolver(Sim &e) {
    e.objective.minres.tolerance = 1e-6;
    e.objective.minres.max_iterations = 1000;
  }

  // Per-particle collision penalty stiffness used by every solid phase.
  template <class Sim> void setSolidPenalty(Sim &e, T density, T ppc) {
    e.implicit_penalty_collision =
        density * std::pow((T)e.dx, (T)dim) / (T)ppc * 1000000;
  }

  // Boundary particles-per-cell, set the same on both phases.
  void set_boundary_ppc(T ppc) {
    sim_coordinator.e1.boundary_ppc = ppc;
    sim_coordinator.e2.boundary_ppc = ppc;
  }

  // MPM knobs the examples all fix the same way.
  void set_common_mpm_defaults() {
    SET_MPM_VALUE(mls_mpm, false);
    SET_MPM_VALUE(ignoreCollisionObject, true);
  }

  void reload() override {
    // Armadillo with delayed fluid injection
    if (test_number == 1) {
      set_path("output/coupling/1_armadillo_fluid");
      SET_SIM_VALUE(end_frame, 200);
      SET_SIM_VALUE(step.max_dt, 0.003);
      SET_MPM_VALUE(dx, (T)1.5 / (T)128);
      sim_coordinator.e1.apic_rpic_ratio = 1;
      sim_coordinator.e2.apic_rpic_ratio = 1;
      T fluid_density = 1e3;
      T solid_density = 3e3;
      T ppc = 8;
      T E = 75000, nu = 0.3;
      SET_MPM_VALUE(gravity, -9.8 * TV::Unit(1));
      SET_MPM_VALUE(symplectic, false);
      SET_MPM_VALUE(cfl, 0.4);
      set_common_mpm_defaults();
      sim_coordinator.setSolver(1e-3, -1, 10000,
                                CouplingSimulation<T, dim>::AMGCL);
      sim_coordinator.use_position_correction = true;
      sim_coordinator.e1.category = MATERIAL_PHASE_FLUID;
      sim_coordinator.e1.config_scheme = CONFIG_UPDATE_LAGRANGIAN;
      sim_coordinator.e1.newton.max_iterations = 20;
      sim_coordinator.e1.newton.tolerance = 1e-1;
      configureFluidLinearSolver(sim_coordinator.e1);
      sim_coordinator.e2.category = MATERIAL_PHASE_SOLID;
      sim_coordinator.e2.config_scheme = CONFIG_TOTAL_LAGRANGIAN;
      sim_coordinator.e2.newton.max_iterations = 1;
      sim_coordinator.e2.newton.tolerance = 1e-2;
      sim_coordinator.e2.newton.linesearch = false;
      configureSolidLinearSolver(sim_coordinator.e2);
      setSolidPenalty(sim_coordinator.e2, solid_density, ppc);
      set_boundary_ppc(ppc);
      sim_coordinator.kkt_relax = ((T)1) / fluid_density / (T)(1e5);
      PDIsotropic<T, dim> model1(E, nu);
      QuadraticVolumePenalty<T, dim> model2(E, nu);
      TV scene_shift = 11.25 * sim_coordinator.e1.dx * TV::Ones();

      sim_coordinator.end_frame_callbacks.push_back(
          [this, fluid_density, ppc, model1, model2](int frame) {
            if (frame == 15) {
              AxisAlignedAnalyticBox<T, dim> box(TV(1.0, 1.4, 0.35),
                                                 TV(1.4, 1.9, 0.75));
              MpmParticleRange<T, dim> p_handle1 =
                  init_helper1.sampleInAnalyticLevelSet(box, fluid_density,
                                                        ppc);
              p_handle1.addFJMaterial(model1, model2, MATERIAL_PHASE_FLUID,
                                           sim_coordinator.e1.config_scheme,
                                           MATERIAL_PROPERTY_INCOMPRESSIBLE,
                                           /*linear corotated*/ false);
              p_handle1.captureRestPositions();
            }
          });

      {
        {
          MpmParticleRange<T, dim> p_handle =
              init_helper2.sampleFromVdbFile("LevelSets/armadillo1.vdb",
                                             solid_density, ppc);
          p_handle.transform([&](int index, Ref<T> mass, TV &X, TV &V) {
            X += TV(0.0, 0.4, 0.1);
          });
          p_handle.addFJMaterial(model1, model2, MATERIAL_PHASE_SOLID,
                                      sim_coordinator.e2.config_scheme,
                                      MATERIAL_PROPERTY_COMPRESSIBLE,
                                      /*linear corotated*/ true);
          p_handle.captureRestPositions();

          Sphere<T, dim> dummy(TV(0, 0), 100);
          p_handle.addMultiTargetPinningInLevelSet(
              dummy, 0, 0,
              /*collider spring mode*/ true);

          Sphere<T, dim> sphere_ls(TV(1.2, 1.3, 0.55), 0.15);
          AnalyticCollisionObject<T, dim> container_object(
              sphere_ls, AnalyticCollisionObject<T, dim>::GHOST);
          p_handle.pinParticlesInLevelSet(sphere_ls, (T)100000, (T)1000);

          p_handle.addTotalLagrangianContacts(sim_coordinator.e2.config_scheme);
        }
      }

      return;
    }

    // Dam break with a deformable bag
    if (test_number == 2) {
      set_path("output/coupling/2_dam_break_with_bag");
      SET_SIM_VALUE(end_frame, 300);
      SET_SIM_VALUE(step.frame_dt, 1. / 48.);
      SET_SIM_VALUE(step.max_dt, 0.003);
      SET_MPM_VALUE(dx, (T)2 / (T)128);
      sim_coordinator.e1.apic_rpic_ratio = 1;
      sim_coordinator.e2.apic_rpic_ratio = 0;
      T fluid_density = 1e3;
      T solid_density = 1000;
      T ppc = 8;
      T E = 120000, nu = 0.4;
      SET_MPM_VALUE(gravity, -9.8 * TV::Unit(1));
      SET_MPM_VALUE(symplectic, false);
      SET_MPM_VALUE(cfl, 0.4);
      set_common_mpm_defaults();
      sim_coordinator.setSolver(1e-3, -1, 10000,
                                CouplingSimulation<T, dim>::AMGCL);
      sim_coordinator.use_position_correction = true;
      sim_coordinator.e1.category = MATERIAL_PHASE_FLUID;
      sim_coordinator.e1.config_scheme = CONFIG_UPDATE_LAGRANGIAN;
      sim_coordinator.e1.newton.max_iterations = 20;
      sim_coordinator.e1.newton.tolerance = 1e-1;
      configureFluidLinearSolver(sim_coordinator.e1);
      sim_coordinator.e2.category = MATERIAL_PHASE_SOLID;
      sim_coordinator.e2.config_scheme = CONFIG_TOTAL_LAGRANGIAN;
      sim_coordinator.e2.newton.max_iterations = 10;
      sim_coordinator.e2.newton.tolerance = 1e-2;
      sim_coordinator.e2.newton.linesearch = true;
      configureSolidLinearSolver(sim_coordinator.e2);
      setSolidPenalty(sim_coordinator.e2, solid_density, ppc);
      set_boundary_ppc(ppc);
      sim_coordinator.kkt_relax = ((T)1) / fluid_density / (T)(1e5);
      PDIsotropic<T, dim> model1(E, nu);
      QuadraticVolumePenalty<T, dim> model2(E, nu);
      TV scene_shift = 111.5 * sim_coordinator.e1.dx * TV::Ones();

      {
        auto to_gpu_vec = [](const TV &x) {
          return GPU_STAGE2::Vec3d{static_cast<double>(x[0]),
                                   static_cast<double>(x[1]),
                                   static_cast<double>(x[2])};
        };
        auto &gpu_collision = sim_coordinator.e1.gpu_box_shell_collision;
        gpu_collision.outer_min = to_gpu_vec(
            TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift);
        gpu_collision.outer_max = to_gpu_vec(
            TV(2, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift);
        gpu_collision.inner_min = to_gpu_vec(TV(0, 0, 0) + scene_shift);
        gpu_collision.inner_max = to_gpu_vec(TV(2, 1, 1) + scene_shift);
        gpu_collision.hole_min =
            to_gpu_vec(TV(0.9, -8 * sim_coordinator.e1.dx, 0) + scene_shift);
        gpu_collision.hole_max = to_gpu_vec(TV(1.1, 0, 1) + scene_shift);
        gpu_collision.has_hole = 1;
        gpu_collision.valid = 1;
      }

      {
        {
          AxisAlignedAnalyticBox<T, dim> box(TV(0, 0, 0) + scene_shift,
                                             TV(0.5, 0.6, 1) + scene_shift);
          MpmParticleRange<T, dim> p_handle1 =
              init_helper1.sampleInAnalyticLevelSet(box, fluid_density, ppc);
          p_handle1.addFJMaterial(model1, model2, MATERIAL_PHASE_FLUID,
                                       sim_coordinator.e1.config_scheme,
                                       MATERIAL_PROPERTY_INCOMPRESSIBLE,
                                       /*linear corotated*/ false);
          p_handle1.captureRestPositions();
        }

        AxisAlignedAnalyticBox<T, dim> box1(
            TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift,
            TV(2, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift);
        AxisAlignedAnalyticBox<T, dim> box2(TV(0, 0, 0) + scene_shift,
                                            TV(2, 1, 1) + scene_shift);
        DifferenceLevelSet<T, dim> ls1, ls2;
        ls1.add(box1, box2);
        AxisAlignedAnalyticBox<T, dim> box_hole(
            TV(0.9, -8 * sim_coordinator.e1.dx, 0) + scene_shift,
            TV(1.1, 0, 1) + scene_shift);
        ls2.add(ls1, box_hole);
        AnalyticCollisionObject<T, dim> container_object(
            ls2, AnalyticCollisionObject<T, dim>::SLIP);
        sim1.sampleBoundaryParticlesInCollisionObjects(
            container_object,
            TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift,
            TV(2, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift,
            sim1.dx * 2);
        init_helper1.addAnalyticCollisionObject(container_object);
      }

      {
        {
          AxisAlignedAnalyticBox<T, dim> box1(
              TV(0.9, -0.6, 0.0) - TV(0.045, 0.045, 0.045) + scene_shift,
              TV(1.1, 0, 1.0) + TV(0.045, 0, 0.045) + scene_shift);
          AxisAlignedAnalyticBox<T, dim> box2(TV(0.9, -0.6, 0.0) + scene_shift,
                                              TV(1.1, 0.0, 1.0) + scene_shift);
          DifferenceLevelSet<T, dim> ls1, ls2;
          ls1.add(box1, box2);

          MpmParticleRange<T, dim> p_handle =
              init_helper2.sampleInAnalyticLevelSet(ls1, solid_density, 8);
          p_handle.addFJMaterial(model1, model2, MATERIAL_PHASE_SOLID,
                                      sim_coordinator.e2.config_scheme,
                                      MATERIAL_PROPERTY_COMPRESSIBLE,
                                      /*linear corotated*/ false);
          p_handle.captureRestPositions();
          Sphere<T, dim> dummy(TV(0, 0), 100);
          p_handle.addMultiTargetPinningInLevelSet(
              dummy, 0, 0,
              /*collider spring mode*/ true);

          AxisAlignedAnalyticBox<T, dim> box_fix(
              TV(0.9, 0, 0.0) - TV(0.1, 0.1, 0.1) + scene_shift,
              TV(1.1, 0, 1.0) + TV(0.1, 0, 0.1) + scene_shift);
          AnalyticCollisionObject<T, dim> container_object(
              box_fix, AnalyticCollisionObject<T, dim>::GHOST);
          p_handle.pinParticlesInLevelSet(box_fix, (T)100000, (T)1000);
        }
      }
      return;
    }

    // Dam break with a soft wall
    if (test_number == 3) {
      set_path("output/coupling/3_dam_break_with_soft_wall");
      SET_SIM_VALUE(end_frame, 400);
      SET_SIM_VALUE(step.max_dt, 0.003);
      SET_MPM_VALUE(dx, (T)2 / (T)128);
      sim_coordinator.e1.apic_rpic_ratio = 1;
      sim_coordinator.e2.apic_rpic_ratio = 0;
      T fluid_density = 1e3;
      T solid_density = 1e3;
      T ppc = 8;
      T E = 20000, nu = 0.4;
      SET_MPM_VALUE(gravity, -9.8 * TV::Unit(1));
      SET_MPM_VALUE(symplectic, false);
      SET_MPM_VALUE(cfl, 0.4);
      set_common_mpm_defaults();
      sim_coordinator.setSolver(1e-3, -1, 10000,
                                CouplingSimulation<T, dim>::AMGCL_CUDA);
      sim_coordinator.use_position_correction = true;
      sim_coordinator.e1.category = MATERIAL_PHASE_FLUID;
      sim_coordinator.e1.config_scheme = CONFIG_UPDATE_LAGRANGIAN;
      sim_coordinator.e1.newton.max_iterations = 20;
      sim_coordinator.e1.newton.tolerance = 1e-1;
      configureFluidLinearSolver(sim_coordinator.e1);
      sim_coordinator.e2.category = MATERIAL_PHASE_SOLID;
      sim_coordinator.e2.config_scheme = CONFIG_TOTAL_LAGRANGIAN;
      sim_coordinator.e2.newton.max_iterations = 10;
      sim_coordinator.e2.newton.tolerance = 1e-2;
      sim_coordinator.e2.newton.linesearch = true;
      configureSolidLinearSolver(sim_coordinator.e2);
      setSolidPenalty(sim_coordinator.e2, solid_density, ppc);
      sim_coordinator.e2.implicit_penalty_collision *= (T)100;
      ZIRAN_INFO("Soft-wall GHOST collision penalty stiffness (100x): ",
                 sim_coordinator.e2.implicit_penalty_collision);
      set_boundary_ppc(ppc);
      sim_coordinator.kkt_relax = ((T)1) / fluid_density / (T)(1e5);
      CorotatedIsotropic<T, dim> model1(E, nu);
      QuadraticVolumePenalty<T, dim> model2(E, nu);
      TV scene_shift = 77.5 * sim_coordinator.e1.dx * TV::Ones();

      {
        auto to_gpu_vec = [](const TV &x) {
          return GPU_STAGE2::Vec3d{static_cast<double>(x[0]),
                                   static_cast<double>(x[1]),
                                   static_cast<double>(x[2])};
        };
        auto &gpu_collision = sim_coordinator.e1.gpu_box_shell_collision;
        gpu_collision.outer_min = to_gpu_vec(
            TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift);
        gpu_collision.outer_max = to_gpu_vec(
            TV(2, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift);
        gpu_collision.inner_min = to_gpu_vec(TV(0, 0, 0) + scene_shift);
        gpu_collision.inner_max = to_gpu_vec(TV(2, 1, 1) + scene_shift);
        gpu_collision.has_hole = 0;
        gpu_collision.valid = 1;
      }

      {
        {
          AxisAlignedAnalyticBox<T, dim> box(TV(0, 0, 0) + scene_shift,
                                             TV(0.7, 0.6, 1) + scene_shift);
          MpmParticleRange<T, dim> p_handle1 =
              init_helper1.sampleInAnalyticLevelSet(box, fluid_density, ppc);
          p_handle1.addFJMaterial(model1, model2, MATERIAL_PHASE_FLUID,
                                       sim_coordinator.e1.config_scheme,
                                       MATERIAL_PROPERTY_INCOMPRESSIBLE,
                                       /*linear corotated*/ false);
          p_handle1.captureRestPositions();
        }

        AxisAlignedAnalyticBox<T, dim> box1(
            TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift,
            TV(2, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift);
        AxisAlignedAnalyticBox<T, dim> box2(TV(0, 0, 0) + scene_shift,
                                            TV(2, 1, 1) + scene_shift);
        DifferenceLevelSet<T, dim> ls1, ls2;
        ls1.add(box1, box2);
        AnalyticCollisionObject<T, dim> container_object(
            ls1, AnalyticCollisionObject<T, dim>::SLIP);
        sim1.sampleBoundaryParticlesInCollisionObjects(
            container_object,
            TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift,
            TV(2, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift,
            sim1.dx * 2);
        init_helper1.addAnalyticCollisionObject(container_object);
      }

      {
        {
          AxisAlignedAnalyticBox<T, dim> box1(
              TV(1.0, -0.03, -0.05) + scene_shift,
              TV(1.07, 0.4, 1.05) + scene_shift);

          MpmParticleRange<T, dim> p_handle =
              init_helper2.sampleInAnalyticLevelSet(box1, solid_density, ppc);
          p_handle.addFJMaterial(model1, model2, MATERIAL_PHASE_SOLID,
                                      sim_coordinator.e2.config_scheme,
                                      MATERIAL_PROPERTY_COMPRESSIBLE,
                                      /*linear corotated*/ false);
          p_handle.captureRestPositions();
          Sphere<T, dim> dummy(TV(0, 0), 100);
          p_handle.addMultiTargetPinningInLevelSet(
              dummy, 0, 0,
              /*collider spring mode*/ true);

          AxisAlignedAnalyticBox<T, dim> box_fix(
              TV(1.0, -0.03, -0.05) + scene_shift,
              TV(1.1, 0.01, 1.05) + scene_shift);
          AxisAlignedAnalyticBox<T, dim> box_fix_left(
              TV(1.0, -0.03, -0.05) + scene_shift,
              TV(1.1, 0.4, 0.01) + scene_shift);
          AxisAlignedAnalyticBox<T, dim> box_fix_right(
              TV(1.0, -0.03, 0.99) + scene_shift,
              TV(1.1, 0.4, 1.05) + scene_shift);

          DisjointUnionLevelSet<T, dim> stickers;
          stickers.add(box_fix);
          stickers.add(box_fix_left);
          stickers.add(box_fix_right);
          p_handle.pinParticlesInLevelSet(stickers, (T)1000000, (T)1000);

          AxisAlignedAnalyticBox<T, dim> box_boundary1(
              TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() +
                  scene_shift,
              TV(2, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() +
                  scene_shift);
          AxisAlignedAnalyticBox<T, dim> box_boundary2(
              TV(0, 0, 0) + scene_shift, TV(2, 1, 1) + scene_shift);
          DifferenceLevelSet<T, dim> ls1, ls2;
          ls1.add(box_boundary1, box_boundary2);
          AxisAlignedAnalyticBox<T, dim> box_hole(TV(1.0, -2, -2) + scene_shift,
                                                  TV(1.07, 2, 2) + scene_shift);
          ls2.add(ls1, box_hole);
          {
            auto to_gpu_vec = [](const TV &x) {
              return GPU_STAGE2::Vec3d{static_cast<double>(x[0]),
                                       static_cast<double>(x[1]),
                                       static_cast<double>(x[2])};
            };
            auto &gpu_ghost = sim_coordinator.e2.gpu_box_shell_collision;
            gpu_ghost.outer_min = to_gpu_vec(
                TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() +
                scene_shift);
            gpu_ghost.outer_max = to_gpu_vec(
                TV(2, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() +
                scene_shift);
            gpu_ghost.inner_min = to_gpu_vec(TV(0, 0, 0) + scene_shift);
            gpu_ghost.inner_max = to_gpu_vec(TV(2, 1, 1) + scene_shift);
            gpu_ghost.hole_min = to_gpu_vec(TV(1.0, -2, -2) + scene_shift);
            gpu_ghost.hole_max = to_gpu_vec(TV(1.07, 2, 2) + scene_shift);
            gpu_ghost.has_hole = 1;
            gpu_ghost.valid = 1;
          }
          AnalyticCollisionObject<T, dim> box_boundary(
              ls2, AnalyticCollisionObject<T, dim>::GHOST);
          init_helper2.addAnalyticCollisionObject(box_boundary);
        }
      }
      return;
    }

    // Buoyancy drop
    if (test_number == 4) {
      set_path("output/coupling/4_buoyancy_drop");
      SET_SIM_VALUE(end_frame, 300);
      SET_SIM_VALUE(step.max_dt, 0.003);
      SET_MPM_VALUE(dx, (T)2 / (T)128);
      sim_coordinator.e1.apic_rpic_ratio = 1;
      sim_coordinator.e2.apic_rpic_ratio = 0;
      T fluid_density = 1e3;
      T solid_density = 700;
      T ppc = 8;
      T E = 50000, nu = 0.4;
      SET_MPM_VALUE(gravity, -3 * TV::Unit(1));
      SET_MPM_VALUE(symplectic, false);
      SET_MPM_VALUE(cfl, 0.4);
      set_common_mpm_defaults();
      sim_coordinator.setSolver(1e-3, -1, 10000,
                                CouplingSimulation<T, dim>::AMGCL);
      sim_coordinator.use_position_correction = true;
      sim_coordinator.e1.category = MATERIAL_PHASE_FLUID;
      sim_coordinator.e1.newton.max_iterations = 20;
      sim_coordinator.e1.newton.tolerance = 1e-1;
      configureFluidLinearSolver(sim_coordinator.e1);
      sim_coordinator.e1.config_scheme = CONFIG_UPDATE_LAGRANGIAN;
      sim_coordinator.e2.category = MATERIAL_PHASE_SOLID;
      sim_coordinator.e2.newton.max_iterations = 1;
      sim_coordinator.e2.newton.tolerance = 1e-2;
      sim_coordinator.e2.newton.linesearch = true;
      configureSolidLinearSolver(sim_coordinator.e2);
      sim_coordinator.e2.config_scheme = CONFIG_TOTAL_LAGRANGIAN;
      setSolidPenalty(sim_coordinator.e2, solid_density, ppc);
      set_boundary_ppc(ppc);
      sim_coordinator.kkt_relax = ((T)1) / fluid_density / (T)(1e5);
      CorotatedIsotropic<T, dim> model1(E, nu);
      QuadraticVolumePenalty<T, dim> model2(E, nu);
      TV scene_shift = 11.5 * sim_coordinator.e1.dx * TV::Ones();

      {
        auto to_gpu_vec = [](const TV &x) {
          return GPU_STAGE2::Vec3d{static_cast<double>(x[0]),
                                   static_cast<double>(x[1]),
                                   static_cast<double>(x[2])};
        };
        auto &gpu_collision = sim_coordinator.e1.gpu_box_shell_collision;
        gpu_collision.outer_min = to_gpu_vec(
            TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift);
        gpu_collision.outer_max = to_gpu_vec(
            TV(1, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift);
        gpu_collision.inner_min = to_gpu_vec(TV(0, 0, 0) + scene_shift);
        gpu_collision.inner_max = to_gpu_vec(TV(1, 1, 1) + scene_shift);
        gpu_collision.has_hole = 0;
        gpu_collision.valid = 1;
      }

      {
        {
          AxisAlignedAnalyticBox<T, dim> box(TV(0, 0, 0) + scene_shift,
                                             TV(1, 0.5, 1) + scene_shift);
          MpmParticleRange<T, dim> p_handle1 =
              init_helper1.sampleInAnalyticLevelSet(box, fluid_density, ppc);
          p_handle1.addFJMaterial(model1, model2, MATERIAL_PHASE_FLUID,
                                       sim_coordinator.e1.config_scheme,
                                       MATERIAL_PROPERTY_INCOMPRESSIBLE,
                                       /*linear corotated*/ false);
          p_handle1.captureRestPositions();
        }

        AxisAlignedAnalyticBox<T, dim> box1(
            TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift,
            TV(1, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift);
        AxisAlignedAnalyticBox<T, dim> box2(TV(0, 0, 0) + scene_shift,
                                            TV(1, 1, 1) + scene_shift);
        DifferenceLevelSet<T, dim> ls1, ls2;
        ls1.add(box1, box2);
        AnalyticCollisionObject<T, dim> container_object(
            ls1, AnalyticCollisionObject<T, dim>::SLIP);
        sim1.sampleBoundaryParticlesInCollisionObjects(
            container_object,
            TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift,
            TV(1, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift,
            sim1.dx * 2);
        init_helper1.addAnalyticCollisionObject(container_object);
      }

      {
        {
          MpmParticleRange<T, dim> p_handle =
              init_helper2.sampleFromVdbFile("LevelSets/bear_move.vdb",
                                             solid_density, ppc);
          p_handle.addFJMaterial(model1, model2, MATERIAL_PHASE_SOLID,
                                      sim_coordinator.e2.config_scheme,
                                      MATERIAL_PROPERTY_COMPRESSIBLE,
                                      /*linear corotated*/ true);
          p_handle.captureRestPositions();
          Sphere<T, dim> dummy(TV(0, 0), 100);
          p_handle.addMultiTargetPinningInLevelSet(
              dummy, 0, 0,
              /*collider spring mode*/ true);
        }
        AxisAlignedAnalyticBox<T, dim> box1(
            TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift,
            TV(1, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() + scene_shift);
        AxisAlignedAnalyticBox<T, dim> box2(
            TV(0, 0, 0) + sim2.dx * TV::Ones() + scene_shift,
            TV(1, 1, 1) - sim2.dx * TV::Ones() + scene_shift);
        {
          auto to_gpu_vec = [](const TV &x) {
            return GPU_STAGE2::Vec3d{static_cast<double>(x[0]),
                                     static_cast<double>(x[1]),
                                     static_cast<double>(x[2])};
          };
          auto &gpu_collision = sim_coordinator.e2.gpu_box_shell_collision;
          gpu_collision.outer_min =
              to_gpu_vec(TV(0, 0, 0) - 4 * sim_coordinator.e1.dx * TV::Ones() +
                         scene_shift);
          gpu_collision.outer_max =
              to_gpu_vec(TV(1, 1, 1) + 4 * sim_coordinator.e1.dx * TV::Ones() +
                         scene_shift);
          gpu_collision.inner_min =
              to_gpu_vec(TV(0, 0, 0) + sim2.dx * TV::Ones() + scene_shift);
          gpu_collision.inner_max =
              to_gpu_vec(TV(1, 1, 1) - sim2.dx * TV::Ones() + scene_shift);
          gpu_collision.has_hole = 0;
          gpu_collision.valid = 1;
        }
        DifferenceLevelSet<T, dim> ls1, ls2;
        ls1.add(box1, box2);
        AnalyticCollisionObject<T, dim> container_object(
            ls1, AnalyticCollisionObject<T, dim>::GHOST);
        init_helper2.addAnalyticCollisionObject(container_object);
      }
      return;
    }

    throw std::invalid_argument("Unknown example. Use -test 1, 2, 3, or 4.");
  }
};

#undef SET_SIM_VALUE
#undef SET_MPM_VALUE

} // namespace ZIRAN
