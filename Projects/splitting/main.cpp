#include "SplittingInit3D.h"
#include <Ziran/CS/Util/CommandLineFlags.h>
#include <Ziran/CS/Util/Debug.h>
#include <Ziran/CS/Util/Filesystem.h>
#include <Ziran/CS/Util/FloatingPointExceptions.h>
#include <Ziran/CS/Util/SignalHandler.h>
#include <tbb/global_control.h>

#include <algorithm>
#include <stdexcept>
#include <thread>

using namespace ZIRAN;

#define T double
#define dim 3

int main(int argc, char *argv[]) {
  {
    bool displayHelp = false;
    int test_number = -1;                    // Example option.
    int max_frame = -1;                      // Optional smoke-test cap.
    int restart = 0;                         // Example option
    bool run_diff_test = false;              // Example option
    double diff_test_perturbation_scale = 1; // Example option

    // Not checking for nan, because when constitutive model returns that,
    // MpmForceAssembly is skipping them (treating as zeros) FPE::WatchedScope
    // w(FPE::Mask::Overflow | FPE::Mask::DivZero); Unconmment the following to
    // catch division by 0 FPE::WatchedScope w(FPE::Mask::Overflow |
    // FPE::Mask::Invalid | FPE::Mask::DivZero);
    FPE::WatchedScope w(FPE::Mask::Invalid);

    FLAGS::Register helpflag("--help", "Print help (this message) and exit",
                             displayHelp);

    // Non-lua command line options
    FLAGS::Register test_number_flag("-test", "Example: 1 colliding Armadillos",
                                     test_number);
    FLAGS::Register max_frame_flag(
        "-max_frame", "Override the example end frame (for smoke tests)",
        max_frame);
    FLAGS::Register run_diff_test_flag("--run_diff_test", "Run diff test",
                                       run_diff_test);
    FLAGS::Register diff_test_perturbation_scale_flag(
        "-dtps", "diff_test_perturbation_scale", diff_test_perturbation_scale);
    FLAGS::Register restart_flag("-restart", "Restart frame", restart);

    int num_threads = static_cast<int>(std::thread::hardware_concurrency());

    FLAGS::Register thread_flag("-t", "Set number of threads", num_threads);

    bool gpu_solid_implicit_state = false;
    bool gpu_solid_implicit_pcg = false;
    bool gpu_solid_implicit_probe = false;
    bool gpu_vp_pressure_system = false;
    bool gpu_vp_amgx_solver = false;
    bool gpu_full_pipeline = false;
    FLAGS::Register gpu_solid_implicit_state_flag(
        "-gpu_solid_implicit_state",
        "Use the CUDA solid trial-state (grid gather, stress, GHOST/TL "
        "contact)",
        gpu_solid_implicit_state);
    FLAGS::Register gpu_solid_implicit_pcg_flag(
        "-gpu_solid_implicit_pcg",
        "Use the CUDA matrix-free PCG for the solid Newton linear solve",
        gpu_solid_implicit_pcg);
    FLAGS::Register gpu_solid_implicit_probe_flag(
        "-gpu_solid_implicit_probe",
        "Run the 3D solid PD matrix-free GPU operator shadow probe",
        gpu_solid_implicit_probe);
    FLAGS::Register gpu_vp_pressure_system_flag(
        "-gpu_vp_pressure_system",
        "Assemble the splitting solid Q2Q0 VP pressure Schur system on CUDA",
        gpu_vp_pressure_system);
    FLAGS::Register gpu_vp_amgx_solver_flag(
        "-gpu_vp_amgx_solver",
        "Solve the CUDA-resident splitting VP pressure CSR with AMGX",
        gpu_vp_amgx_solver);
    FLAGS::Register gpu_full_pipeline_flag(
        "-gpu_full_pipeline",
        "Enable GPU solid trial state/PCG and the Q2Q0 VP pressure pipeline",
        gpu_full_pipeline);

    try {
      FLAGS::ParseFlags(argc, argv);
      if (!displayHelp && test_number != 1)
        throw std::invalid_argument(
            "Unknown example. Use -test 1 for the Armadillo collision.");
      if (gpu_full_pipeline) {
#if defined(ENABLE_CUDA_STAGE2_PROBE) && defined(ENABLE_AMGX)
        gpu_solid_implicit_state = true;
        gpu_solid_implicit_pcg = true;
        gpu_vp_pressure_system = true;
        gpu_vp_amgx_solver = true;
#else
        ZIRAN_ASSERT(
            false,
            "-gpu_full_pipeline requires a CUDA build with AMGX enabled");
#endif
      }
      if (gpu_vp_amgx_solver)
        gpu_vp_pressure_system = true;
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

    tbb::global_control tbb_thread_limit(
        tbb::global_control::max_allowed_parallelism,
        static_cast<std::size_t>(std::max(1, num_threads)));

    {
      SplittingSimulation<T, dim> e;
      e.gpu_solid_implicit_state = gpu_solid_implicit_state;
      e.gpu_solid_implicit_pcg = gpu_solid_implicit_pcg;
      e.gpu_solid_implicit_probe = gpu_solid_implicit_probe;
      e.gpu_vp_pressure_system = gpu_vp_pressure_system;
      e.gpu_vp_amgx_solver = gpu_vp_amgx_solver;
      e.gpu_solid_device_resident = gpu_full_pipeline;
      // The standalone VP Schur solve consumes the same device-resident base
      // blocks as coupling.  Enable both the GPU Q0 pressure-grid rebuild and
      // the Stage3B solid block assembler as one indivisible pipeline.
      e.gpu_build_system = gpu_vp_pressure_system;
      e.gpu_stage3b_assemble_blocks = gpu_vp_pressure_system;
      if (run_diff_test) {
        e.diff_test = true;
        e.diff_test_perturbation_scale = diff_test_perturbation_scale;
      }
      e.logger = LogWorker::initializeLogging();
      SplittingInit3D<T> h(e, test_number);
      if (!restart)
        h.start(max_frame);
      else
        h.restart(restart, max_frame);
    }
  }
  return 0;
}
