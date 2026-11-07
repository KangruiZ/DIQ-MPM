#include "CouplingInit3D.h"
#include <Ziran/CS/Util/CommandLineFlags.h>
#include <Ziran/CS/Util/Debug.h>
#include <Ziran/CS/Util/Filesystem.h>
#include <Ziran/CS/Util/FloatingPointExceptions.h>
#include <Ziran/CS/Util/SignalHandler.h>
#include <stdexcept>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/tbb.h>

using namespace ZIRAN;

#define T double
#define dim 3

int main(int argc, char *argv[]) {
  {
    bool displayHelp = false;
    int test_number = -1;
    int restart = 0;
    bool run_diff_test = false;
    double diff_test_perturbation_scale = 1;
    bool disable_restart = false;
    int max_frame = -1;
    int restart_interval = -1;
    std::string solver_override;
    std::string output_dir_override;
    bool gpu_stage2_probe = false;
    bool gpu_stage3a_probe = false;
    bool gpu_stage3b_probe = false;
    bool gpu_stage3b_writeback = false;
    bool gpu_build_system = false;
    bool gpu_full_pipeline = false;
    bool gpu_solid_implicit_probe = false;
    bool gpu_solid_implicit_pcg = false;
    bool gpu_solid_implicit_state = false;
    bool gpu_position_corrector = false;
    bool gpu_position_corrector_density = false;
    bool gpu_position_corrector_pages = false;
    bool gpu_position_corrector_direct_topology_probe = false;
    bool gpu_position_corrector_direct_matrix_probe = false;
    bool gpu_position_corrector_direct_solver_probe = false;
    bool gpu_position_corrector_direct_legacy_order_probe = false;
    bool gpu_position_corrector_direct_legacy_solver_probe = false;
    bool gpu_position_corrector_direct_legacy_solver_writeback = false;
    bool gpu_position_corrector_direct_legacy_solver = false;
    bool gpu_position_corrector_direct_gpu_solver = false;
    bool gpu_position_corrector_direct_amgx_solver = false;
    bool gpu_position_corrector_direct_postsolve = false;
    bool gpu_coupling_amgx_solver = false;
    bool gpu_position_corrector_direct_pushout = false;
    bool gpu_build_system_cpu_numerical_grid = false;
    bool gpu_build_system_cpu_pressure_grid = false;
    bool gpu_build_system_cpu_velocity_grid = false;
    bool gpu_fluid_velocity_transfer = false;

    // Not checking for nan, because when constitutive model returns that,
    // MpmForceAssembly is skipping them (treating as zeros) FPE::WatchedScope
    // w(FPE::Mask::Overflow | FPE::Mask::DivZero); Unconmment the following to
    // catch division by 0 FPE::WatchedScope w(FPE::Mask::Overflow |
    // FPE::Mask::Invalid | FPE::Mask::DivZero);
    FPE::WatchedScope w(FPE::Mask::Invalid);

    FLAGS::Register helpflag("--help", "Print help (this message) and exit",
                             displayHelp);

    FLAGS::Register restartflag("-disablerestart", "Disable restart data",
                                disable_restart);

    FLAGS::Register test_number_flag(
        "-test", "Example: 1 Armadillo, 2 bag, 3 soft wall, 4 buoyancy",
        test_number);
    FLAGS::Register run_diff_test_flag(
        "--run_diff_test", "Run diff test (non-lua test)", run_diff_test);
    FLAGS::Register diff_test_perturbation_scale_flag(
        "-dtps", "diff_test_perturbation_scale (non-lua test)",
        diff_test_perturbation_scale);
    FLAGS::Register restart_flag("-restart", "Restart frame", restart);
    FLAGS::Register max_frame_flag(
        "-max_frame", "Override end_frame after scene initialization",
        max_frame);
    FLAGS::Register restart_interval_flag(
        "-restart_interval",
        "Override the frame interval for restart checkpoints",
        restart_interval);
    FLAGS::Register solver_override_flag(
        "-solver", "Override coupling solver: eigen, amgcl, or amgcl_cuda",
        solver_override);
    FLAGS::Register output_dir_override_flag(
        "-output_dir",
        "Override the coordinator output directory after scene initialization",
        output_dir_override);
    FLAGS::Register gpu_stage2_probe_flag(
        "-gpu_stage2_probe",
        "Run the stage 2 GPU sparse-grid probe without changing simulation "
        "state",
        gpu_stage2_probe);
    FLAGS::Register gpu_stage3a_probe_flag(
        "-gpu_stage3a_probe",
        "Run the stage 3A GPU operator probe without changing simulation state",
        gpu_stage3a_probe);
    FLAGS::Register gpu_stage3b_probe_flag(
        "-gpu_stage3b_probe", "Run the stage 3B assembled sparse-product probe",
        gpu_stage3b_probe);
    FLAGS::Register gpu_stage3b_writeback_flag(
        "-gpu_stage3b_writeback",
        "Use the stage 3B GPU sparse products for fluid/solid Schur blocks",
        gpu_stage3b_writeback);
    FLAGS::Register gpu_build_system_flag(
        "-gpu_build_system",
        "Use the persistent GPU sparse-grid buildSystem path",
        gpu_build_system);
    FLAGS::Register gpu_full_pipeline_flag(
        "-gpu_full_pipeline",
        "Enable every validated GPU coupling path: buildSystem, coupling "
        "AMGX, device-resident fluid Q1 state, fluid/solid device transfers, "
        "solid PD implicit state/PCG, and direct-AMGX PositionCorrector",
        gpu_full_pipeline);
    FLAGS::Register gpu_solid_implicit_probe_flag(
        "-gpu_solid_implicit_probe",
        "Run the 3D solid PD matrix-free GPU operator shadow probe",
        gpu_solid_implicit_probe);
    FLAGS::Register gpu_solid_implicit_pcg_flag(
        "-gpu_solid_implicit_pcg",
        "Use device-resident PCG for the supported 3D PD solid Newton solve",
        gpu_solid_implicit_pcg);
    FLAGS::Register gpu_solid_implicit_state_flag(
        "-gpu_solid_implicit_state",
        "Use CUDA PD trial-state evaluation (including GHOST/TL contact) "
        "before the solid Newton solve",
        gpu_solid_implicit_state);
    FLAGS::Register gpu_position_corrector_flag(
        "-gpu_position_corrector",
        "Use CUDA particle level-set construction in PositionCorrector",
        gpu_position_corrector);
    FLAGS::Register gpu_position_corrector_density_flag(
        "-gpu_position_corrector_density",
        "Use GPU density scatter in PositionCorrector",
        gpu_position_corrector_density);
    FLAGS::Register gpu_position_corrector_pages_flag(
        "-gpu_position_corrector_pages",
        "Use GPU current-frame page discovery in PositionCorrector",
        gpu_position_corrector_pages);
    FLAGS::Register gpu_position_corrector_direct_topology_probe_flag(
        "-gpu_position_corrector_direct_topology_probe",
        "Validate direct keyed PositionCorrector pressure topology",
        gpu_position_corrector_direct_topology_probe);
    FLAGS::Register gpu_position_corrector_direct_matrix_probe_flag(
        "-gpu_position_corrector_direct_matrix_probe",
        "Validate direct keyed PositionCorrector pressure CSR assembly",
        gpu_position_corrector_direct_matrix_probe);
    FLAGS::Register gpu_position_corrector_direct_solver_probe_flag(
        "-gpu_position_corrector_direct_solver_probe",
        "Compare CPU AMGCL on direct keyed PositionCorrector pressure CSR",
        gpu_position_corrector_direct_solver_probe);
    FLAGS::Register gpu_position_corrector_direct_legacy_order_probe_flag(
        "-gpu_position_corrector_direct_legacy_order_probe",
        "Validate SPGrid-free direct pressure DOF ordering",
        gpu_position_corrector_direct_legacy_order_probe);
    FLAGS::Register gpu_position_corrector_direct_legacy_solver_probe_flag(
        "-gpu_position_corrector_direct_legacy_solver_probe",
        "Compare CPU AMGCL on legacy-ordered direct pressure CSR",
        gpu_position_corrector_direct_legacy_solver_probe);
    FLAGS::Register gpu_position_corrector_direct_legacy_solver_writeback_flag(
        "-gpu_position_corrector_direct_legacy_solver_writeback",
        "Write legacy-ordered direct pressure solve back for validation",
        gpu_position_corrector_direct_legacy_solver_writeback);
    FLAGS::Register gpu_position_corrector_direct_legacy_solver_flag(
        "-gpu_position_corrector_direct_legacy_solver",
        "Use legacy-ordered direct PositionCorrector pressure solve",
        gpu_position_corrector_direct_legacy_solver);
    FLAGS::Register gpu_position_corrector_direct_gpu_solver_flag(
        "-gpu_position_corrector_direct_gpu_solver",
        "Use AMGCL_CUDA on the direct PositionCorrector pressure CSR",
        gpu_position_corrector_direct_gpu_solver);
    FLAGS::Register gpu_position_corrector_direct_amgx_solver_flag(
        "-gpu_position_corrector_direct_amgx_solver",
        "Use AMGX with device-resident direct PositionCorrector pressure CSR",
        gpu_position_corrector_direct_amgx_solver);
    FLAGS::Register gpu_position_corrector_direct_postsolve_flag(
        "-gpu_position_corrector_direct_postsolve",
        "Use CUDA pressure-to-particle postsolve with the resident AMGX "
        "solution",
        gpu_position_corrector_direct_postsolve);
    FLAGS::Register gpu_coupling_amgx_solver_flag(
        "-gpu_coupling_amgx_solver",
        "Use AMGX with the device-resident Stage3B coupling CSR",
        gpu_coupling_amgx_solver);
    FLAGS::Register gpu_position_corrector_direct_pushout_flag(
        "-gpu_position_corrector_direct_pushout",
        "Clone pressure-grid page topology for PositionCorrector solid_grid",
        gpu_position_corrector_direct_pushout);
    FLAGS::Register gpu_build_system_cpu_numerical_grid_flag(
        "-gpu_build_system_cpu_numerical_grid",
        "Validate GPU quadrature while retaining CPU numerical grids",
        gpu_build_system_cpu_numerical_grid);
    FLAGS::Register gpu_build_system_cpu_pressure_grid_flag(
        "-gpu_build_system_cpu_pressure_grid",
        "Validate GPU velocity grid while retaining CPU pressure grids",
        gpu_build_system_cpu_pressure_grid);
    FLAGS::Register gpu_build_system_cpu_velocity_grid_flag(
        "-gpu_build_system_cpu_velocity_grid",
        "Validate CPU fluid velocity grid while retaining GPU pressure "
        "topology",
        gpu_build_system_cpu_velocity_grid);
    FLAGS::Register gpu_fluid_velocity_transfer_flag(
        "-gpu_fluid_velocity_transfer",
        "Use CUDA Q1 fluid velocity-to-particle transfer after coupling solve",
        gpu_fluid_velocity_transfer);

    int num_threads = tbb::info::default_concurrency();

    FLAGS::Register thread_flag("-t", "Set number of threads", num_threads);

    try {
      FLAGS::ParseFlags(argc, argv);
      if (!displayHelp && (test_number < 1 || test_number > 4))
        throw std::invalid_argument(
            "Unknown example. Use -test 1, 2, 3, or 4.");
      if (gpu_full_pipeline) {
#if defined(ENABLE_AMGCL_CUDA) && defined(ENABLE_AMGX)
        gpu_build_system = true;
        gpu_coupling_amgx_solver = true;
        gpu_position_corrector = true;
        gpu_position_corrector_density = true;
        gpu_position_corrector_pages = true;
        gpu_position_corrector_direct_amgx_solver = true;
        gpu_position_corrector_direct_postsolve = true;
        gpu_position_corrector_direct_pushout = true;
        gpu_fluid_velocity_transfer = true;
        gpu_solid_implicit_pcg = true;
        gpu_solid_implicit_state = true;
#else
        ZIRAN_ASSERT(
            false,
            "-gpu_full_pipeline requires a CUDA build with AMGX enabled");
#endif
      }
      if (gpu_build_system)
        gpu_stage3b_writeback = true;
      MpmSimulationBase<T, dim>::disable_restart = disable_restart;
    } catch (std::exception &e) {
      std::cerr << e.what() << std::endl;
      FLAGS::PrintUsage(std::cerr);
      return 1;
    }
    if (displayHelp) {
      std::cout << "Usage:\n";
      FLAGS::PrintUsage(std::cout);
      return 0;
    }
    installSignalHandler();

    if (num_threads <= 0)
      num_threads = tbb::info::default_concurrency();
    tbb::global_control init(tbb::global_control::max_allowed_parallelism,
                             static_cast<size_t>(num_threads));

    {
      CouplingSimulation<T, dim> e;
      if (run_diff_test) {
        e.diff_test = true;
        e.diff_test_perturbation_scale = diff_test_perturbation_scale;
      }
      e.logger = LogWorker::initializeLogging();
      e.gpu_stage2_probe = gpu_stage2_probe;
      e.gpu_stage3a_probe = gpu_stage3a_probe;
      e.gpu_stage3b_probe = gpu_stage3b_probe;
      e.gpu_stage3b_writeback = gpu_stage3b_writeback;
      e.gpu_build_system = gpu_build_system;
      e.gpu_full_pipeline = gpu_full_pipeline;
      // All four release examples enforce the full-GPU numerical contract.
      e.gpu_full_pipeline_strict = gpu_full_pipeline;
      if (e.gpu_full_pipeline_strict)
        ZIRAN_INFO("Strict full-GPU contract enabled for test ", test_number);
      e.gpu_solid_implicit_probe = gpu_solid_implicit_probe;
      e.gpu_solid_implicit_pcg = gpu_solid_implicit_pcg;
      e.gpu_solid_implicit_state = gpu_solid_implicit_state;
      e.gpu_position_corrector = gpu_position_corrector;
      e.gpu_position_corrector_density = gpu_position_corrector_density;
      e.gpu_position_corrector_pages = gpu_position_corrector_pages;
      e.gpu_position_corrector_direct_topology_probe =
          gpu_position_corrector_direct_topology_probe;
      e.gpu_position_corrector_direct_matrix_probe =
          gpu_position_corrector_direct_matrix_probe;
      e.gpu_position_corrector_direct_solver_probe =
          gpu_position_corrector_direct_solver_probe;
      e.gpu_position_corrector_direct_legacy_order_probe =
          gpu_position_corrector_direct_legacy_order_probe;
      e.gpu_position_corrector_direct_legacy_solver_probe =
          gpu_position_corrector_direct_legacy_solver_probe;
      e.gpu_position_corrector_direct_legacy_solver_writeback =
          gpu_position_corrector_direct_legacy_solver_writeback;
      e.gpu_position_corrector_direct_legacy_solver =
          gpu_position_corrector_direct_legacy_solver;
      e.gpu_position_corrector_direct_gpu_solver =
          gpu_position_corrector_direct_gpu_solver;
      e.gpu_position_corrector_direct_amgx_solver =
          gpu_position_corrector_direct_amgx_solver;
      e.gpu_position_corrector_direct_postsolve =
          gpu_position_corrector_direct_postsolve;
      e.gpu_coupling_amgx_solver = gpu_coupling_amgx_solver;
      e.gpu_position_corrector_direct_pushout =
          gpu_position_corrector_direct_pushout;
      e.gpu_build_system_cpu_numerical_grid =
          gpu_build_system_cpu_numerical_grid;
      e.gpu_build_system_cpu_pressure_grid = gpu_build_system_cpu_pressure_grid;
      e.gpu_build_system_cpu_velocity_grid = gpu_build_system_cpu_velocity_grid;
      // The CUDA buildSystem path now owns a verified Q1 device velocity
      // transfer. Keep the explicit flag for A/B runs, but enable it for the
      // normal GPU pipeline so users do not silently fall back to a full
      // device-velocity download followed by CPU G2P.
      e.gpu_fluid_velocity_transfer =
          gpu_fluid_velocity_transfer || gpu_build_system;
      auto apply_solver_override = [&]() {
        if (solver_override.empty()) {
#ifdef ENABLE_AMGCL_CUDA
          if (gpu_stage3b_writeback) {
            e.solverType = CouplingSimulation<T, dim>::AMGCL_CUDA;
            ZIRAN_INFO("Using AMGCL_CUDA for -gpu_stage3b_writeback");
          }
#else
          if (gpu_stage3b_writeback)
            ZIRAN_WARN("-gpu_stage3b_writeback requested without "
                       "ENABLE_AMGCL_CUDA; keeping scene solver backend");
#endif
          return;
        }
        if (solver_override == "eigen")
          e.solverType = CouplingSimulation<T, dim>::EIGEN;
        else if (solver_override == "amgcl")
          e.solverType = CouplingSimulation<T, dim>::AMGCL;
        else if (solver_override == "amgcl_cuda")
          e.solverType = CouplingSimulation<T, dim>::AMGCL_CUDA;
        else
          ZIRAN_ASSERT(false, "Unknown -solver value. Use eigen, amgcl, or "
                              "amgcl_cuda.");
        ZIRAN_INFO("Overriding coupling solver:", solver_override);
      };
      auto apply_output_dir_override = [&]() {
        if (output_dir_override.empty())
          return;
        e.output_dir.path = output_dir_override;
        e.e1.output_dir.path = output_dir_override + "/material1";
        e.e2.output_dir.path = output_dir_override + "/material2";
        e.output_dir.createPath();
        e.e1.output_dir.createPath();
        e.e2.output_dir.createPath();
        ZIRAN_INFO("Overriding coupling output directory:",
                   output_dir_override);
      };
      CouplingInit3D<T> h(e, test_number);
      if (!restart) {
        if (max_frame >= 0 || !solver_override.empty() ||
            !output_dir_override.empty()) {
          h.initialize();
          if (max_frame >= 0)
            e.end_frame = max_frame;
          if (restart_interval > 0)
            e.restart_frame_interval = restart_interval;
          apply_solver_override();
          apply_output_dir_override();
          e.simulate();
        } else {
          h.start();
        }
      } else {
        if (max_frame >= 0 || !solver_override.empty() ||
            !output_dir_override.empty()) {
          h.initialize();
          if (max_frame >= 0)
            e.end_frame = max_frame;
          if (restart_interval > 0)
            e.restart_frame_interval = restart_interval;
          apply_solver_override();
          apply_output_dir_override();
          h.prepareRestart(restart);
          e.restart(restart);
          e.simulate();
        } else {
          h.restart(restart);
        }
      }
    }
  }
  return 0;
}
