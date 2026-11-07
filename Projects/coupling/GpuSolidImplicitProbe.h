#pragma once

#include "GpuOperatorProbe.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace ZIRAN::GPU_SOLID_IMPLICIT {

// Compact GPU-side representation of the material part of the 3D Q2
// matrix-free Backward-Euler operator. `idx` is the existing CPU grid DOF
// order, deliberately retained during the S1 oracle stage.
struct SolidNode {
    GPU_STAGE3A::GridKey key = 0;
    int idx = -1;
    double mass = 0;
    // Row-major collision projection. Identity for unconstrained nodes.
    double projection[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

struct PdParticle {
    GPU_STAGE3A::Vec3d x;
    double volume = 0;
    double mu = 0;
    double pinning_a = 0;
    // Positive particle-space tangent added to the PD Hessian. Legacy callers
    // leave this zero and pinning_a is interpreted as pinning_a * I.
    double pinning_matrix[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
};

struct PdOperatorInput {
    double dx = 0;
    double dt = 0;
    // When true the grid topology (node keys/indices, mass, projection) and the
    // per-particle static fields (x/volume/mu) already reside on the device from
    // an earlier call this step and are not re-uploaded.
    bool reuse_static = false;
    // The nonlinear trial-state context already owns this step's node
    // key/index/mass/projection arrays. Reuse those arrays while initializing
    // the operator's particle fields for its first solve of the step.
    bool reuse_topology = false;
    // Seed x/volume/mu from the resident nonlinear trial particles instead of
    // uploading the equivalent host PdParticle array.
    bool use_device_trial_particles = false;
    // When true the operator's per-particle tangent (pinning_a/pinning_matrix)
    // is taken directly from the device trial-state tangent buffers instead of
    // from the host particles array, avoiding a device->host->device round trip.
    bool use_device_tangent = false;
    std::vector<SolidNode> nodes;
    std::vector<PdParticle> particles;
};

// Inputs and outputs for the nonlinear trial-state part of a total-Lagrangian
// PD solid step.  This is deliberately separate from PdOperatorInput: the
// operator only needs the frozen Hessian, whereas the Newton residual needs
// the current grid velocity, polar rotation, unilateral GHOST springs and
// (when enabled) the TL self-contact neighbourhood.
struct PdTrialNode {
    SolidNode node;
    // v^n and dv are retained independently so a device Newton step can
    // reproduce the host update order exactly: dv += ddv; v = v^n + dv.
    GPU_STAGE3A::Vec3d base_velocity;
    GPU_STAGE3A::Vec3d velocity_increment;
    GPU_STAGE3A::Vec3d velocity;
};

struct PdTrialParticle {
    // x is the current physical/grid position used by the MPM stencil and the
    // TL contact reference test. x0 is the total-Lagrangian xn used for the
    // implicit trial position xp = xn + dt v.
    GPU_STAGE3A::Vec3d x;
    GPU_STAGE3A::Vec3d x0;
    double F[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double volume = 0;
    double mass = 0;
    double mu = 0;
    // Ordinary (bilateral) pinning: f = -a x + b.
    double pinning_a = 0;
    GPU_STAGE3A::Vec3d pinning_b;
    // When pinning_from_resident_x0 is set, CUDA reconstructs A/b from these
    // target coefficients after the resident owner has supplied the current
    // xn. This avoids downloading xn solely for the damping term.
    double pinning_k = 0;
    double pinning_d = 0;
    GPU_STAGE3A::Vec3d pinning_target_x;
    GPU_STAGE3A::Vec3d pinning_target_v;
    int pinning_from_resident_x0 = 0;
    // Slice of PdParallelPinningTarget in the flat target array below.
    int parallel_target_begin = 0;
    int parallel_target_count = 0;
};

struct PdParallelPinningTarget {
    GPU_STAGE3A::Vec3d x;
    double k = 0;
};

// Axis-aligned Difference(outer, inner[, hole]) GHOST geometry. For a
// resident TL solid the CUDA trial kernel can reconstruct the lagged collider
// target directly from current x0, avoiding host target vectors/current-X.
struct GhostBoxShell3D {
    GPU_STAGE3A::Vec3d outer_min;
    GPU_STAGE3A::Vec3d outer_max;
    GPU_STAGE3A::Vec3d inner_min;
    GPU_STAGE3A::Vec3d inner_max;
    GPU_STAGE3A::Vec3d hole_min;
    GPU_STAGE3A::Vec3d hole_max;
    int has_hole = 0;
    int valid = 0;
};

struct PdTrialStateInput {
    double dx = 0;
    double dt = 0;
    bool enable_tl_contact = false;
    // When true, the grid topology (node keys/indices), the per-particle static
    // fields (x/x0/F/volume/mass/mu) and the parallel-pinning targets are
    // assumed unchanged since the previous call and are NOT re-uploaded; only
    // the per-node trial velocity is refreshed. Valid across Newton iterations
    // of one implicit step, where only the trial grid velocity changes.
    bool reuse_static = false;
    // Reuse the device-owned v^n/dv/trial velocity as well. This is used after
    // a device PCG step has advanced them without a host round trip.
    bool reuse_velocity = false;
    // When true the device also scatters force_stress/fp into a node-indexed
    // residual force (output.node_residual), matching the CPU force raster. The
    // per-particle trial fields and tangent then remain device-resident; the
    // converged gradV is downloaded once through downloadPdTrialGradVelocity3D().
    bool scatter_residual = false;
    // Compatibility paths download the scattered node residual. A fully
    // device-resident Newton path leaves it in the persistent context for the
    // norm and following PCG solve.
    bool download_node_residual = true;
    // Optional canonical TL Q2 grid owner. On the first trial of a substep,
    // CUDA seeds key/idx/mass/vn directly from this map and initializes dv to
    // initial_velocity_increment. Identity node projection is required.
    GPU_STAGE3A::DeviceGridNodeMap resident_nodes;
    GPU_STAGE3A::Vec3d initial_velocity_increment;
    // Cross-substep particle owner. On the first Newton state of a new step,
    // dynamic x/x0/F/mass/volume are patched device-to-device into the
    // persistent trial metadata instead of re-uploading the full AoS.
    GPU_STAGE3A::SolidParticleStateDeviceView resident_particles;
    std::vector<PdTrialNode> nodes;
    std::vector<PdTrialParticle> particles;
    std::vector<PdParallelPinningTarget> parallel_targets;
};

struct PdTrialStateOutput {
    // AoS-compatible scalar arrays so the host bridge can write the existing
    // Eigen/DataArray scratch fields without changing their ownership.
    std::vector<double> particle_velocity;       // 3 * particles
    std::vector<double> particle_grad_velocity;  // 9 * particles, row major
    std::vector<double> deformation_gradient;    // 9 * particles, row major
    std::vector<double> determinant;
    std::vector<double> trial_position;          // 3 * particles
    std::vector<double> force_stress;            // 9 * particles, row major
    std::vector<double> force;                   // 3 * particles
    // Sum of every supported particle-space tangent coefficient per particle:
    // ordinary pinning, active GHOST springs and TL self-contact. The device
    // PCG consumes this as an isotropic particle pinning term.
    std::vector<double> tangent_a;
    std::vector<double> tangent_matrix;          // 9 * particles, row major
    // TL-only portion, retained for profiling/validation.
    std::vector<double> tl_contact_a;
    // Node-indexed residual force (3 * nodes, node.idx order) scattered on the
    // device from force_stress/fp, matching rasterizeForceToTVStack(scale=dt).
    // Empty unless PdTrialStateInput::scatter_residual is set.
    std::vector<double> node_residual;
    // Material plus supported pinning/contact force energy for the current
    // trial state. Inertia and gravity remain in BackwardEuler's grid term.
    double force_energy = 0;
};

struct PdTrialStateStats {
    double upload_ms = 0;
    double gather_ms = 0;
    double contact_ms = 0;
    double download_ms = 0;
    std::size_t active_tl_contacts = 0;
};

// Total-Lagrangian nonlinear CorotatedIsotropic trial state.  The common
// position/pinning/contact fields intentionally mirror PdTrialParticle, while
// lambda and the projected dP/dF are specific to CorotatedIsotropic.  Keeping
// this distinct from LinearCorotatedParticle is important: the latter freezes
// R, whereas CorotatedIsotropic uses the current SVD and its projected
// SvdBasedIsotropicHelper Hessian at every Newton state.
struct CorotatedTrialParticle {
    GPU_STAGE3A::Vec3d x;
    GPU_STAGE3A::Vec3d x0;
    double F[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double volume = 0;
    double mass = 0;
    double mu = 0;
    double lambda = 0;
    double pinning_a = 0;
    GPU_STAGE3A::Vec3d pinning_b;
    // Compact resident mode reconstructs A/b after the device owner supplies
    // the current xn, avoiding a host download for the damping term.
    double pinning_k = 0;
    double pinning_d = 0;
    GPU_STAGE3A::Vec3d pinning_target_x;
    GPU_STAGE3A::Vec3d pinning_target_v;
    int pinning_from_resident_x0 = 0;
    int parallel_target_begin = 0;
    int parallel_target_count = 0;
};

struct CorotatedTrialStateInput {
    double dx = 0;
    double dt = 0;
    bool enable_tl_contact = false;
    bool reuse_static = false;
    bool reuse_velocity = false;
    // Compact resident mode mirrors PdTrialStateInput: keep all particle
    // fields/Hessian/tangent on CUDA and optionally return only the scattered
    // node residual required by the host-owned line-search Newton loop.
    bool scatter_residual = false;
    bool download_node_residual = true;
    GhostBoxShell3D ghost_collision;
    double ghost_penalty = 0;
    GPU_STAGE3A::DeviceGridNodeMap resident_nodes;
    GPU_STAGE3A::Vec3d initial_velocity_increment;
    GPU_STAGE3A::SolidParticleStateDeviceView resident_particles;
    std::vector<PdTrialNode> nodes;
    std::vector<CorotatedTrialParticle> particles;
    std::vector<PdParallelPinningTarget> parallel_targets;
};

// `projected_hessian` is particle-major 9x9, row-major in both indices:
// dP[ij] = sum_rs H[ij,rs] dF[rs].  It is the CPU CorotatedIsotropic
// `firstPiolaDifferential(..., project=true)` linearization, not a finite
// difference and not a fixed-rotation approximation.
struct CorotatedParticle {
    GPU_STAGE3A::Vec3d x;
    double volume = 0;
    double projected_hessian[81] = {};
    double pinning_a = 0;
    double pinning_matrix[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
};

struct CorotatedOperatorInput {
    double dx = 0;
    double dt = 0;
    bool reuse_static = false;
    bool reuse_topology = false;
    // Consume x/volume/projected Hessian and contact/pinning tangent directly
    // from the latest Corotated trial buffers.
    bool use_device_trial = false;
    std::vector<SolidNode> nodes;
    std::vector<CorotatedParticle> particles;
};

// Fixed-rotation linear-corotated Hessian used by the current test-7 solid.
// During a Newton solve Ziran freezes R in backupStrain(), so the exact CPU
// differential is
//   dP = mu dF + mu R dF^T R + lambda (R:dF) R.
// The volume penalty belongs to the coupling pressure formulation; it is not
// part of FJMixedForce::computeStressDifferential() for this solid path.
struct LinearCorotatedParticle {
    GPU_STAGE3A::Vec3d x;
    double volume = 0;
    double mu = 0;
    double lambda = 0;
    double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double pinning_a = 0;
};

struct LinearCorotatedOperatorInput {
    double dx = 0;
    double dt = 0;
    std::vector<SolidNode> nodes;
    std::vector<LinearCorotatedParticle> particles;
};

struct PersistentPdOperator3D {
    std::shared_ptr<void> storage;
};

struct PdOperatorStats {
    std::size_t nodes = 0;
    std::size_t particles = 0;
    double upload_ms = 0;
    double apply_ms = 0;
    double download_ms = 0;
    double total_ms = 0;
};

struct PdPcgStats {
    int iterations = 0;
    double initial_preconditioned_norm = 0;
    double final_preconditioned_norm = 0;
    double upload_ms = 0;
    double solve_ms = 0;
    double download_ms = 0;
};

// Applies the material-plus-inertia portion of Ziran's current 3D PDIsotropic
// total-Lagrangian matrix-free Hessian:
//
//   y = M x + dt^2 G^T [ 2 mu_p V_p (Gx)_p ].
//
// The public interface intentionally uses the CPU grid DOF order. S1 uses it
// as a correctness oracle; a later phase replaces the upload with a persistent
// solid device sparse-grid state and keeps Krylov vectors on device.
void applyPdMatrixFree3D(PersistentPdOperator3D& context,
    const PdOperatorInput& input, const std::vector<double>& x,
    std::vector<double>& y, PdOperatorStats* stats = nullptr);

// Device-resident PCG for the same operator. The input vectors use the CPU
// grid DOF order only at the API boundary; r/p/Ap/z remain on device for the
// whole solve. This first S2 implementation is intentionally limited to an
// identity projection (the bag example has no collision nodes).
void solvePdPcg3D(PersistentPdOperator3D& context,
    const PdOperatorInput& input, const std::vector<double>& rhs,
    double relative_tolerance, double absolute_tolerance, int max_iterations,
    std::vector<double>& solution, PdPcgStats* stats = nullptr);

// Assembles gravity + material + inertia in the persistent device residual and
// returns only the mass-weighted Newton norm. No node vector crosses the API.
double preparePdNewtonResidual3D(PersistentPdOperator3D& context,
    const GPU_STAGE3A::Vec3d& gravity, double dt);

// PCG variant whose RHS is the persistent Newton residual and whose solution
// remains device-resident. applyPdNewtonIncrement3D then performs
//   dv += solution; velocity = v^n + dv
// without materializing either node vector on the host.
void solvePdPcgDeviceResidual3D(PersistentPdOperator3D& context,
    const PdOperatorInput& input, double relative_tolerance,
    double absolute_tolerance, int max_iterations,
    PdPcgStats* stats = nullptr);
void applyPdNewtonIncrement3D(PersistentPdOperator3D& context);

// Device line-search controls. set... evaluates vn + dv + alpha*step without
// mutating dv; commit... accepts that scale. totalEnergy includes the current
// force energy plus inertia and gravity for the same alpha.
void setPdNewtonLineSearchTrial3D(PersistentPdOperator3D& context,
    double alpha);
void commitPdNewtonLineSearchTrial3D(PersistentPdOperator3D& context,
    double alpha);
double computePdNewtonTotalEnergy3D(PersistentPdOperator3D& context,
    const GPU_STAGE3A::Vec3d& gravity, double dt, double step_scale = 0.0);

// Evaluates the nonlinear total-Lagrangian PD trial state on CUDA. It exactly
// follows the CPU force order for the supported terms: grid gather, F = Fn +
// dt grad(v), PD stress, bilateral pinning, unilateral parallel/GHOST springs,
// then TL self-contact. The compatibility mode materializes the legacy scratch;
// production scatters the residual on device and keeps the particle state there.
void evaluatePdTrialState3D(PersistentPdOperator3D& context,
    const PdTrialStateInput& input, PdTrialStateOutput& output,
    PdTrialStateStats* stats = nullptr);

// Materializes only the most recently evaluated PD trial gradV. Production
// calls this once after Newton convergence so evolveStrain keeps its existing
// host contract without a 9-double/particle D2H on every Newton iteration.
void downloadPdTrialGradVelocity3D(PersistentPdOperator3D& context,
    std::vector<double>& particle_grad_velocity);

// One compatibility writeback after Newton: the converged grid increment and
// particle gradV are copied together. Iteration-sized transfers stay removed.
void downloadPdNewtonState3D(PersistentPdOperator3D& context,
    std::vector<double>& velocity_increment,
    std::vector<double>& particle_grad_velocity);

// Publish the converged implicit velocity into the resident TL Q2 node map
// without staging the node array through host memory. The sorted P2G keys stay
// authoritative while idx/new_v retain the established SPGrid DOF contract.
void exportFinalGridVelocityToNodeMap3D(PersistentPdOperator3D& context,
    GPU_STAGE3A::DeviceGridNodeMap& nodes);

// CUDA SVD/state path for CorotatedIsotropic. `output` has the same legacy
// residual scratch fields as PD, while `projected_hessian` feeds the following
// device PCG.  This split keeps the existing CPU residual rasterizer intact
// for the A/B migration stage.
void evaluateCorotatedTrialState3D(PersistentPdOperator3D& context,
    const CorotatedTrialStateInput& input, PdTrialStateOutput& output,
    std::vector<double>& projected_hessian,
    PdTrialStateStats* stats = nullptr);

void applyCorotatedMatrixFree3D(PersistentPdOperator3D& context,
    const CorotatedOperatorInput& input, const std::vector<double>& x,
    std::vector<double>& y, PdOperatorStats* stats = nullptr);

void solveCorotatedPcg3D(PersistentPdOperator3D& context,
    const CorotatedOperatorInput& input, const std::vector<double>& rhs,
    double relative_tolerance, double absolute_tolerance, int max_iterations,
    std::vector<double>& solution, PdPcgStats* stats = nullptr);
void solveCorotatedPcgDeviceResidual3D(PersistentPdOperator3D& context,
    const CorotatedOperatorInput& input, double relative_tolerance,
    double absolute_tolerance, int max_iterations,
    PdPcgStats* stats = nullptr);

// Device PCG for the fixed-R linear-corotated material Hessian above.  The
// caller retains the CPU grid-DOF order at the boundary; all Krylov vectors
// remain on device.  It is opt-in until the CPU/GPU Newton-step oracle passes.
void solveLinearCorotatedPcg3D(PersistentPdOperator3D& context,
    const LinearCorotatedOperatorInput& input, const std::vector<double>& rhs,
    double relative_tolerance, double absolute_tolerance, int max_iterations,
    std::vector<double>& solution, PdPcgStats* stats = nullptr);

} // namespace ZIRAN::GPU_SOLID_IMPLICIT
