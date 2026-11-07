# GPU coupling interface

Run commands from the repository root, using the CUDA environment and the
executable in `build-ziran-cuda12/Projects/coupling/coupling`.
Build instructions and data setup are in the [README](../README.md).

## Production entry point

```bash
conda run -n ziran-cuda12 --no-capture-output \
  build-ziran-cuda12/Projects/coupling/coupling \
  -test 3 -max_frame 200 -gpu_full_pipeline -restart_interval 20 -t 4 \
  -output_dir output/coupling/3_dam_break_with_soft_wall
```

`-gpu_full_pipeline` enables GPU sparse-grid/system assembly, the coupling
AMGX solver, fluid Q1 transfers, solid trial-state/PCG, and all production GPU
position-corrector stages. Setting a scene's solver to `AMGCL_CUDA` alone does
not enable this pipeline or select AMGX.

For all four coupling examples (1–4), the full-pipeline flag also enables a strict
contract: a supported numerical stage falling back to a host implementation
stops the run. Initialization and output are explicit host boundaries.

## Supported numerical scope

- 3D double-precision Q1/Q0 fluids and TL/APIC/Q2 solids without plasticity.
- PDIsotropic TL solid implicit solves (1/2).
- Projected CorotatedIsotropic TL solid implicit solves (4/3).
- Validated open-domain or axis-aligned box-shell geometry; described GHOST,
  ordinary pinning, and supported TL self-contact.
- GPU interface quadrature, legacy-order pressure/face/velocity maps, coupled
  sparse CSR assembly, and AMGX solves.
- Resident fluid/solid particle state and solid Newton/PCG/line-search vectors.
- GPU level set, density, pressure-grid topology/CSR, AMGX, particle correction,
  and pushout in the position corrector.

Initialization still uses a CPU warm-up grid. Scheduling, logging, checkpoints,
and visualization use the host. General unsupported materials/collision objects
retain compatibility paths; they are not covered by the strict four-case
validation. Example 3 uses a 100x GHOST penalty multiplier in its initializer.

## Common options

| Flag | Meaning |
|---|---|
| `-test N` | Example 1, 2, 3, or 4 |
| `-gpu_full_pipeline` | Supported production GPU path |
| `-max_frame N` | Override final frame |
| `-restart N` | Read matching material checkpoints at frame N |
| `-restart_interval N` | Checkpoint interval; default 20 |
| `-output_dir PATH` | Coordinator output directory |
| `-t N` | TBB worker limit |
| `-solver eigen\|amgcl\|amgcl_cuda` | Compatibility solver; does not override an explicitly enabled AMGX path |
| `--help` | Complete list of options, including diagnostic controls |

Default output directories are:

| Example | Directory |
|---:|---|
| 1 | `output/coupling/1_armadillo_fluid` |
| 2 | `output/coupling/2_dam_break_with_bag` |
| 3 | `output/coupling/3_dam_break_with_soft_wall` |
| 4 | `output/coupling/4_buoyancy_drop` |

Other example numbers are rejected before scene initialization.

Use a separate output directory for each run or configuration. Both material
folders must contain a checkpoint for a restart. A scene with delayed particle
injection replays initialization callbacks to restore the required attributes
before loading restart state.

Example 1 starts with only solid particles and injects fluid at the end of
frame 15. Before injection, it advances the solid on the GPU with `J=det(F)`;
the coupling pressure solve and position corrector begin once fluid is present.

## Solver settings and diagnostics

The coupling AMGX iteration limit defaults to 30. The environment variable
`ZIRAN_AMGX_COUPLING_MAX_ITERS` changes it. Hold this fixed in performance and
state comparisons.

The source retains CPU bridge/oracle switches for numerical regression.
Examples include `ZIRAN_GPU_FLUID_VELOCITY_COMPARE=1`,
`ZIRAN_GPU_FLUID_DEVICE_DOF_COMPARE=1`,
`ZIRAN_GPU_SOLID_RESIDENT_DISABLE=1`, and
`ZIRAN_GPU_SOLID_PARTICLE_OWNER_ORACLE=1`.
They can download state or force compatibility work and should not be enabled
during production performance measurements. A forced fallback may intentionally
violate the strict full-pipeline contract.

A normal strict run logs `Strict full-GPU contract enabled`, verified
build/solve/position-corrector stages, and AMGX `status=0`.
For particle comparisons, fix the initial state, frame range, material/contact
parameters, and solver settings. Compare both phases' particle positions and
`xX0` state rather than requiring identical serialized files.

See [validation coverage](gpu_coupling_acceptance_report.md) for the existing
results. The standalone solid VP pressure system is documented
[separately](gpu_splitting_vp_interface.md).
