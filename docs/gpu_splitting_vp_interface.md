# Standalone solid VP GPU interface

The `splitting` executable solves a solid mixed Q2Q0 pressure system. It is
separate from the two-phase `coupling` executable and its coupled system.

The supported VP path assembles the Q0 sparse grid and Q2Q0 blocks on CUDA,
forms device Schur CSR, solves it with AMGX, and evaluates
`v = M^-1(a - Gp)` on the device. Solid nonlinear work uses CUDA trial-state
evaluation and matrix-free PCG. Compatibility particle writeback and some
host mirrors remain; this standalone path does not have the same particle
residency guarantee as the four strict coupling examples.

## Example 1: Colliding

This is the three-armadillo collision scene named **Colliding** in Fig. 3 of
the DIQ-MPM paper.

This is the only standalone splitting example; other numbers are rejected.
The scene contains three total-Lagrangian Armadillos with
`PDIsotropic + QuadraticVolumePenalty`, `use_vp=true`,
`frame_dt=1/48`, and `max_dt=0.001`.

Build with the [GPU configuration](../README.md#build-the-gpu-version), then run
from the repository root:

```bash
conda run -n ziran-cuda12 cmake --build build-ziran-cuda12 --target splitting -j 4

conda run -n ziran-cuda12 --no-capture-output \
  build-ziran-cuda12/Projects/splitting/splitting \
  -test 1 -max_frame 1 -gpu_full_pipeline -t 4
```

Output is written to `output/splitting/1_colliding_armadillos/`.
For a long run, set `-max_frame 200`. To continue an existing run, use
`-restart N` with a checkpoint in that directory. Repeated fresh runs overwrite
the example's existing outputs.

The CPU reference build uses AMGCL for the VP pressure solve. Build `splitting`
with the CPU configuration in the README and run
`build-ziran-dev/Projects/splitting/splitting -test 1 -max_frame 1 -t 4`
from the repository root, in the CPU environment. The GPU compatibility default
uses AMGCL_CUDA; `-gpu_full_pipeline` selects the AMGX production path.

## Options

| Flag | Path |
|---|---|
| `-gpu_full_pipeline` | GPU solid trial-state/PCG, sparse-grid/base blocks, Schur CSR, AMGX VP solve and device pressure-to-velocity algebra |
| `-gpu_vp_pressure_system` | GPU assembly followed by CSR download and AMGCL_CUDA compatibility solve |
| `-gpu_vp_amgx_solver` | GPU assembly and AMGX VP solve; does not by itself enable solid trial-state/PCG |
| `-gpu_solid_implicit_state` | CUDA solid trial-state evaluation |
| `-gpu_solid_implicit_pcg` | CUDA matrix-free solid Newton PCG |

The coupling flags `-gpu_build_system` and
`-gpu_coupling_amgx_solver` do not belong to this executable.

A working AMGX run logs `Splitting VP AMGX device CSR solver` with
`iterations=30 status=0`. Repeated VP solve output within one frame is normal:
the example has multiple substeps per frame.

## Validation

The recorded long run completed 0–50 and then continued from its own GPU
restart 40 through 200. All 3,677 AMGX solves in the continuation succeeded.
Later restart-100→101 transfer/residency optimizations preserved the recorded
Newton residuals and particle outputs. A full CPU/AMGX particle comparison is
not recorded. See the [validation summary](gpu_coupling_acceptance_report.md).
