# GPU validation coverage

This is the retained validation summary for the GPU implementation.
Historical evidence was collected before release cleanup; it does not imply
that every long sequence was rerun after each residency change. Generated
particles, checkpoints, and local experiment logs are not part of the source
distribution.

Scene numbers below use the release numbering in the [README](../README.md#supported-examples).
Historical runs used the original IDs shown in that mapping.

## Meaning of the checks

| Check | What it establishes |
|---|---|
| Local operator/oracle | A particular GPU stage agrees with a controlled host or previous implementation |
| Short end-to-end comparison | The pipeline advances from a fixed initial state or restart with checked particle state |
| Long physical run | A long trajectory was generated and, where stated, its animation/contact behavior was accepted |
| Strict GPU contract | Covered numerical stages execute on the device rather than silently using CPU fallback |

Long physical acceptance and CPU numerical equivalence are distinct claims.

## Coupling examples

| Test | Historical long run | Retained conclusion |
|---|---|---|
| 1: Armadillo bath | 0–150 | Long physical run accepted; PD solid trial-state, operator and PCG have local host-oracle comparisons |
| 2: Bag | 0–140, then continuation to 260 | Short state comparisons at startup, 20→21 and 120→121 passed; a final long physical acceptance is not recorded |
| 3: Wall | 0–200 | Long physical run accepted for the 100x GHOST penalty configuration; controlled fluid analytical-integration comparison passed |
| 4: Bear | Output through 300; GPU Corotated backend used for 61–300 | Stable checkpoint continuation and local Corotated tests; frames 0–60 originated from an earlier CPU-solid run |

For example 2, the checked fluid position difference is approximately
`2.38419e-7` in Partio float output, with solid position difference zero.
The 120→121 initial-only inactive-node fix removed the 130 previously stationary
boundary particles in the controlled comparison.

For example 3, the controlled two-frame GPU/CPU fluid analytical-integration check
had zero position difference for both phases. This is a local implementation
comparison, not proof that different global solver configurations produce the
same long trajectory. Its multi-Newton GHOST force oracle has a known limitation;
the entire solid chain is not claimed to have strict local CPU equivalence.

For example 1, restart-140 local checks found trial velocity/F/position errors at
approximately machine precision, matrix-free operator relative L2 error
`1.44e-15`, and PCG increment relative L2 error `6.62e-13`.
The later release verification below adds a two-frame CPU comparison before
fluid injection; it does not establish CPU equivalence after injection.

## Strict four-case regression: 2026-08-26

| Test | Range | Coupling substeps | Verified build / solve / position corrector |
|---|---|---:|---|
| 1 | 20→21 | 45 | 45 / 45 / 45 |
| 2 | 20→21 | 17 | 17 / 17 / 17 |
| 3 | 20→21 | 50 | 50 / 50 / 50 |
| 4 | 0→1 | 14 | 14 / 14 / 14 |

All coupling AMGX solves succeeded with device solutions. Solid pressure
mapping recorded `host_nodes=0`; nonzero solid PCG work used device vectors.
The example 4 first frame did not require nonzero PCG work, so its constitutive/PCG
coverage also relies on the independent fixture.

## Retained unit tests

The CUDA sparse-grid, sparse-operator, and solid-implicit tests passed in the
2026-08-26 record. The solid fixture includes projected Corotated stress and
Hessian comparisons, a matrix-free operator check, device PCG, pinning/GHOST
contact, and device Newton/line-search commits. A separate cuSPARSE SpGEMM
smoke target is included.

Build and run the tests with the commands in the [README](../README.md#tests).
These tests require no simulation data.

## Standalone solid VP

The splitting collision example completed 0–200 using GPU trial-state/PCG, sparse assembly, AMGX VP
solve, and device post-solve algebra. Restart-100→101 residency optimizations
preserved the recorded Newton residual sequence and particle outputs.
A complete CPU/AMGX particle comparison remains outside the recorded coverage;
the standalone path still has host particle writeback boundaries.

See the [splitting interface](gpu_splitting_vp_interface.md).

## Release cleanup verification before renumbering: 2026-11-06

The release checkout was configured and built independently, using the existing
validated conda environments and cached third-party source checkouts.

- CUDA `coupling` and `splitting` builds succeeded; all four CUDA CTest fixtures
  passed after cleanup.
- The CPU reference `coupling` build succeeded; all three CPU CTest fixtures
  passed. Device-only pressure-map checks are now conditional on a CUDA build;
  the CPU Q0 grid and weighted-mass checks remain active.
- The buoyancy scene (coupling example 4) was run for two frames with the pre-cleanup GPU executable and the
  release executable, both with `-gpu_full_pipeline -t 4`. Each run completed
  28 strict build/solve/position-corrector substeps and 56 successful AMGX solves.
- The two phases contained 1,048,503 fluid and 21,627 solid particles. All 18
  `partio`, `xX0`, and `J_pointer` files for frames 0, 1, and 2 were compared by
  particle and attribute. Positions and J agreed exactly; solid state agreed
  exactly. The maximum fluid `xX0.info` difference was `1.16415321827e-10`
  (relative L2 `5.18150689456e-11`), with no non-finite values.
- The standalone splitting collision example completed its first frame with all 21 VP
  AMGX solves successful; particle, xX0, and J outputs were written.
- An application-only CUDA configuration omitted the test targets while keeping
  AMGX/cuSPARSE enabled. Architecture 86 was honored as an explicit build
  override; the default was separately checked as 89. This configuration check
  is not a runtime validation of another GPU architecture.
- Documentation links and the source tree were checked for removed-file links,
  machine-specific documentation paths, temporary binaries, and generated data.

These are correctness and build checks. The GPU was shared with other running
processes, so their elapsed times are not used as performance benchmarks.

## Five-example release verification: 2026-11-06

Only the four coupling scenes and the standalone collision scene listed in the
README are retained. Coupling IDs are 1–4; the separate splitting executable
uses ID 1. Both executables reject removed IDs before scene initialization.
The retained scene bodies were checked against their original versions;
physical parameters, sampling, material models, contact, and callbacks are
unchanged. Output directories and example numbers were renamed.

The CPU and CUDA builds of both executables succeeded. All three CPU and four
CUDA CTest fixtures passed. The CUDA operator fixture now also checks a
solid-only affine Q2 velocity field, shuffled DOF indices, advection, and the
analytic deformation determinant without consuming a pressure map.

Starting coupling example 1 from frame 0 exposed a missing resident solid G2P
transfer before its delayed fluid injection. The no-fluid branch now consumes
the converged device grid velocity and updates J from det(F), retaining the
strict GPU assertions. For frames 0, 1, and 2, all 148,131 solid particles'
`partio`, `xX0`, and `J_pointer` attributes matched the CPU reference exactly
in the serialized float output, with no non-finite values.

Example 1 also completed frames 0–16 with the strict GPU contract enabled:
210 solid-only transfers before injection, followed by 14 verified coupled
build/solve/position-corrector substeps and 28 successful AMGX solves. The
other four scenes rely on the prior validation described above; no additional
long-run acceptance is claimed for this cleanup.

## Source dependency cleanup: 2026-11-06

The release retains only 3D double-precision simulation instantiations. Unused
2D/float simulation builds, plasticity and unselected material families, old
scene generators/importers, fiber initialization, unused Partio debug exports,
source/animated colliders, and empty scene/plugin wrappers were removed.
The unused libkdtree dependency and source-local Makefile/legacy CUDA build
helpers were also removed. GPU kernels, the CPU reference, shared numerical
support, and regression fixtures remain.

The retained sampling, force setup, pinning/contact, and Partio writer bodies
were checked against the preceding commit. Scene parameters and callbacks are
unchanged. Removing the plasticity stage removes an empty loop for these five
elastic/fluid scenes.

Both CPU and CUDA executables and all retained test targets compiled and linked
successfully after pruning. No unit tests or simulations were executed for this
pruning pass. The runtime
evidence above predates it; build checks do not add a new numerical-equivalence
or long-run acceptance claim.

## MPM structure and naming cleanup: 2026-11-07

The shared MPM layer is grouped into Grid, Simulation, Initialization, and
Force. Concrete particle/force classes have descriptive names, template method
definitions use `.tpp`, and the compiled library is named `mpm_core`. The
unused generic force factory and uncalled diagnostics were removed; the
existing grid kinetic-energy calculation is retained as a focused helper.
Unused LinearCorotated scalar/dimension instantiations were also removed.

Checkpoint target attribute keys and serialization order remain unchanged.
Source-token and include-dependency checks cover the declared renames and
cleanup. Scene physical parameters, transfers, material/contact calculations,
and CUDA numerical kernels retain their existing calculations.

The standalone collision initializer now selects an available default pressure
solver: AMGCL in a CPU build and AMGCL_CUDA in a CUDA build. The full GPU flag
continues to select AMGX. Requesting the CUDA solver through the shared
sparse-matrix interface from a CPU-only build now raises an explicit error
instead of returning without a solution.

Both CPU and CUDA executables and all retained test targets compiled and linked
after this refactor. No unit tests or simulations were run for this pass; these
checks add no new trajectory or restart-equivalence claim.
