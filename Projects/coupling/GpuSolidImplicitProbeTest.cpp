#include "GpuSolidImplicitProbe.h"

#include <Ziran/Physics/ConstitutiveModel/HyperelasticConstitutiveModel.h>
#include <Ziran/Physics/ConstitutiveModel/CorotatedIsotropic.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace {

using ZIRAN::GPU_SOLID_IMPLICIT::CorotatedOperatorInput;
using ZIRAN::GPU_STAGE3A::GridKey;

void quadraticWeights(double coordinate, int& base, double weights[3],
    double derivatives[3])
{
    base = static_cast<int>(std::floor(coordinate - 0.5));
    const double d0 = coordinate - static_cast<double>(base);
    const double z = 1.5 - d0;
    weights[0] = 0.5 * z * z;
    const double d1 = d0 - 1.0;
    weights[1] = 0.75 - d1 * d1;
    const double d2 = 1.0 - d1;
    const double zz = 1.5 - d2;
    weights[2] = 0.5 * zz * zz;
    derivatives[0] = -z;
    derivatives[1] = -2.0 * d1;
    derivatives[2] = zz;
}

void applyCorotatedOnHost(const CorotatedOperatorInput& input,
    const std::vector<double>& x, std::vector<double>& y)
{
    using ZIRAN::GPU_STAGE3A::packGridKey3D;
    std::unordered_map<GridKey, int> node_index;
    node_index.reserve(input.nodes.size());
    y.assign(x.size(), 0.0);
    for (const auto& node : input.nodes) {
        node_index.emplace(node.key, node.idx);
        for (int axis = 0; axis < 3; ++axis)
            y[static_cast<std::size_t>(3 * node.idx + axis)] +=
                node.mass * x[static_cast<std::size_t>(3 * node.idx + axis)];
    }
    for (const auto& particle : input.particles) {
        int base[3];
        double w[3][3], dw[3][3];
        quadraticWeights(particle.x.x / input.dx, base[0], w[0], dw[0]);
        quadraticWeights(particle.x.y / input.dx, base[1], w[1], dw[1]);
        quadraticWeights(particle.x.z / input.dx, base[2], w[2], dw[2]);
        double dF[9] = {};
        double particle_x[3] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) {
                    const auto found = node_index.find(
                        packGridKey3D(base[0] + i, base[1] + j, base[2] + k));
                    if (found == node_index.end())
                        continue;
                    const int node = found->second;
                    const double gradient[3] = {
                        dw[0][i] * w[1][j] * w[2][k] / input.dx,
                        w[0][i] * dw[1][j] * w[2][k] / input.dx,
                        w[0][i] * w[1][j] * dw[2][k] / input.dx};
                    const double weight = w[0][i] * w[1][j] * w[2][k];
                    for (int row = 0; row < 3; ++row) {
                        const double value = x[static_cast<std::size_t>(3 * node + row)];
                        particle_x[row] += value * weight;
                        for (int col = 0; col < 3; ++col)
                            dF[3 * row + col] += value * gradient[col];
                    }
                }
        double dP[9] = {};
        for (int output = 0; output < 9; ++output)
            for (int source = 0; source < 9; ++source)
                dP[output] += particle.projected_hessian[9 * output + source] * dF[source];
        const double material_scale = input.dt * input.dt * particle.volume;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) {
                    const auto found = node_index.find(
                        packGridKey3D(base[0] + i, base[1] + j, base[2] + k));
                    if (found == node_index.end())
                        continue;
                    const int node = found->second;
                    const double gradient[3] = {
                        dw[0][i] * w[1][j] * w[2][k] / input.dx,
                        w[0][i] * dw[1][j] * w[2][k] / input.dx,
                        w[0][i] * w[1][j] * dw[2][k] / input.dx};
                    const double weight = w[0][i] * w[1][j] * w[2][k];
                    for (int row = 0; row < 3; ++row) {
                        double value = 0.0;
                        for (int col = 0; col < 3; ++col)
                            value += material_scale * dP[3 * row + col] * gradient[col];
                        double pinning = 0.0;
                        for (int col = 0; col < 3; ++col) {
                            double tangent = particle.pinning_matrix[3 * row + col];
                            if (row == col)
                                tangent += particle.pinning_a;
                            pinning += tangent * particle_x[col];
                        }
                        y[static_cast<std::size_t>(3 * node + row)] +=
                            value + input.dt * input.dt * pinning * weight;
                    }
                }
    }
}

struct ErrorNorm {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
};

ErrorNorm compareVectors(const std::vector<double>& reference,
    const std::vector<double>& candidate)
{
    if (reference.size() != candidate.size())
        throw std::runtime_error("Corotated oracle vector size mismatch");
    double reference_l2 = 0.0;
    double error_l2 = 0.0;
    ErrorNorm error;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double difference = candidate[i] - reference[i];
        error.max_abs = std::max(error.max_abs, std::abs(difference));
        reference_l2 += reference[i] * reference[i];
        error_l2 += difference * difference;
    }
    error.rel_l2 = std::sqrt(error_l2 / std::max(reference_l2, 1e-300));
    return error;
}

} // namespace

int main()
{
    using namespace ZIRAN;
    using namespace ZIRAN::GPU_SOLID_IMPLICIT;
    PdOperatorInput input;
    input.dx = 1.0;
    input.dt = 0.03;
    int idx = 0;
    // Deliberately use a non-key order to verify that the compact GPU key map
    // preserves the caller's existing DOF ordering.
    for (int z = 1; z >= -1; --z)
        for (int y = 1; y >= -1; --y)
            for (int x = 1; x >= -1; --x)
                input.nodes.push_back(SolidNode{
                    GPU_STAGE3A::packGridKey3D(x, y, z), idx++,
                    1.0 + 0.01 * idx});
    input.particles.push_back(PdParticle{
        GPU_STAGE3A::Vec3d{0.15, -0.20, 0.30}, 0.125, 37.0});
    input.particles.push_back(PdParticle{
        GPU_STAGE3A::Vec3d{-0.25, 0.10, -0.15}, 0.075, 19.0});
    std::vector<double> x(input.nodes.size() * 3);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const int sequence = static_cast<int>((17 * i + 3) % 23);
        x[i] = 0.03125 * static_cast<double>(sequence - 11);
    }

    PersistentPdOperator3D context;
    std::vector<double> result;
    applyPdMatrixFree3D(context, input, x, result);

    // The CUDA and CPU implementations are selected by CMake. This fixture
    // validates both finite output and the nontrivial mass+stiffness path; a
    // later S1 integration compares this device path against the production
    // BackwardEuler objective at an actual Newton state.
    if (result.size() != x.size())
        throw std::runtime_error("solid implicit operator result size mismatch");
    double squared_norm = 0.0;
    for (double value : result) {
        if (!std::isfinite(value))
            throw std::runtime_error("solid implicit operator produced non-finite value");
        squared_norm += value * value;
    }
    constexpr double expected_squared_norm = 4.494623679695346;
    if (std::abs(squared_norm - expected_squared_norm) > 1e-10)
        throw std::runtime_error("solid implicit operator fixture mismatch");

    // The CPU-only fallback intentionally has no device trial-state path.
    // Keep its existing PD fixture runnable while the CUDA build exercises the
    // SVD-based (non fixed-R) CorotatedIsotropic migration below.
#if defined(ENABLE_CUDA_STAGE2_PROBE)
    // Exercise the SVD-based (non fixed-R) CorotatedIsotropic migration as a
    // compact CUDA A/B fixture. It compares both particle H=dP/dF and the
    // whole mass-plus-matrix-free operator against the CPU implementation.
    CorotatedTrialStateInput corotated_trial;
    corotated_trial.dx = input.dx;
    corotated_trial.dt = input.dt;
    for (const SolidNode& node : input.nodes) {
        const int n = node.idx;
        PdTrialNode trial_node;
        trial_node.node = node;
        trial_node.velocity = {0.002 * (n - 7), -0.001 * (n + 3),
            0.0015 * (2 * n - 5)};
        corotated_trial.nodes.push_back(trial_node);
    }
    CorotatedTrialParticle trial_particle;
    trial_particle.x = {0.15, -0.20, 0.30};
    trial_particle.x0 = trial_particle.x;
    const double F[9] = {1.04, 0.03, -0.01,
                         -0.02, 0.97, 0.04,
                          0.01, -0.03, 1.02};
    for (int i = 0; i < 9; ++i)
        trial_particle.F[i] = F[i];
    trial_particle.volume = 0.125;
    trial_particle.mass = 0.8;
    trial_particle.mu = 37.0;
    trial_particle.lambda = 53.0;
    corotated_trial.particles.push_back(trial_particle);
    PdTrialStateOutput corotated_state;
    std::vector<double> projected_hessian;
    evaluateCorotatedTrialState3D(
        context, corotated_trial, corotated_state, projected_hessian);
    if (projected_hessian.size() != 81 || corotated_state.force_stress.size() != 9)
        throw std::runtime_error("solid Corotated trial-state result size mismatch");
    for (int row = 0; row < 9; ++row)
        for (int col = 0; col < 9; ++col) {
            const double value = projected_hessian[9 * row + col];
            if (!std::isfinite(value) ||
                std::abs(value - projected_hessian[9 * col + row]) > 1e-9)
                throw std::runtime_error("solid Corotated projected Hessian mismatch");
        }
    CorotatedIsotropic<double, 3> cpu_corotated;
    cpu_corotated.mu = trial_particle.mu;
    cpu_corotated.lambda = trial_particle.lambda;
    cpu_corotated.project = true;
    Matrix<double, 3, 3> cpu_F;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            cpu_F(row, col) = corotated_state.deformation_gradient[3 * row + col];
    CorotatedIsotropic<double, 3>::Scratch cpu_scratch;
    cpu_corotated.updateScratch(cpu_F, cpu_scratch);
    const double cpu_force_energy =
        trial_particle.volume * cpu_corotated.psi(cpu_scratch);
    if (std::abs(corotated_state.force_energy - cpu_force_energy) > 1e-10)
        throw std::runtime_error(
            "solid Corotated force-energy CPU/GPU oracle mismatch");
    // The resident GHOST path reconstructs the same lagged unilateral target
    // from x0 that addBoundarySpring() used to build on the host.  Exercise a
    // point inside outer-minus-inner, with an oblique inner-box normal so the
    // full rank-one tangent is checked rather than only one diagonal entry.
    CorotatedTrialStateInput ghost_trial = corotated_trial;
    ghost_trial.ghost_collision.outer_min = {-1.0, -1.0, -1.0};
    ghost_trial.ghost_collision.outer_max = {1.0, 1.0, 1.0};
    ghost_trial.ghost_collision.inner_min = {-0.1, -0.1, -0.1};
    ghost_trial.ghost_collision.inner_max = {0.1, 0.1, 0.1};
    ghost_trial.ghost_collision.valid = 1;
    ghost_trial.ghost_penalty = 123.0;
    PdTrialStateOutput ghost_state;
    std::vector<double> ghost_projected_hessian;
    evaluateCorotatedTrialState3D(
        context, ghost_trial, ghost_state, ghost_projected_hessian);
    const double outside[3] = {0.05, -0.10, 0.20};
    const double outside_norm = std::sqrt(
        outside[0] * outside[0] + outside[1] * outside[1] +
        outside[2] * outside[2]);
    const double normal[3] = {-outside[0] / outside_norm,
                              -outside[1] / outside_norm,
                              -outside[2] / outside_norm};
    const double x0[3] = {trial_particle.x0.x, trial_particle.x0.y,
                          trial_particle.x0.z};
    const double phi = -outside_norm;
    double signed_distance = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double target = x0[axis] - phi * normal[axis];
        signed_distance +=
            (ghost_state.trial_position[axis] - target) * normal[axis];
    }
    if (!(signed_distance < 0.0))
        throw std::runtime_error("solid resident GHOST fixture is not active");
    const double expected_energy_delta =
        0.5 * ghost_trial.ghost_penalty * signed_distance * signed_distance;
    if (std::abs((ghost_state.force_energy - corotated_state.force_energy) -
                     expected_energy_delta) > 1e-10 ||
        std::abs((ghost_state.tangent_a[0] - corotated_state.tangent_a[0]) -
                     ghost_trial.ghost_penalty) > 1e-10)
        throw std::runtime_error("solid resident GHOST scalar oracle mismatch");
    for (int row = 0; row < 3; ++row) {
        const double expected_force_delta =
            -ghost_trial.ghost_penalty * signed_distance * normal[row];
        if (std::abs((ghost_state.force[row] - corotated_state.force[row]) -
                         expected_force_delta) > 1e-10)
            throw std::runtime_error("solid resident GHOST force oracle mismatch");
        for (int col = 0; col < 3; ++col) {
            const double expected_tangent_delta =
                ghost_trial.ghost_penalty * normal[row] * normal[col];
            const int entry = 3 * row + col;
            if (std::abs((ghost_state.tangent_matrix[entry] -
                             corotated_state.tangent_matrix[entry]) -
                             expected_tangent_delta) > 1e-10)
                throw std::runtime_error(
                    "solid resident GHOST tangent oracle mismatch");
        }
    }
    // Ordinary pinning uses xn in its damping term. Verify that the device
    // reconstructs A/b from the resident owner rather than the deliberately
    // stale/zero host x0 carried by the compact metadata payload.
    GPU_STAGE3A::SolidParticleStateInput3D pinning_owner_input;
    GPU_STAGE3A::Mat3d identity;
    identity.values[0] = identity.values[4] = identity.values[8] = 1.0;
    pinning_owner_input.rest_x = {{0.15, -0.20, 0.30}};
    pinning_owner_input.current_x = {{0.18, -0.22, 0.31}};
    pinning_owner_input.velocity = {{0.0, 0.0, 0.0}};
    pinning_owner_input.affine = {GPU_STAGE3A::Mat3d{}};
    pinning_owner_input.grad_velocity = {GPU_STAGE3A::Mat3d{}};
    pinning_owner_input.f = {identity};
    pinning_owner_input.j = {1.0};
    pinning_owner_input.mass = {0.8};
    pinning_owner_input.volume = {0.125};
    pinning_owner_input.lambda = {0.0};
    GPU_STAGE3A::PersistentSolidParticleState3D pinning_owner;
    GPU_STAGE3A::initializePersistentSolidParticleState3D(
        pinning_owner, pinning_owner_input);
    PdTrialStateInput pd_trial;
    pd_trial.dx = input.dx;
    pd_trial.dt = input.dt;
    pd_trial.resident_particles =
        GPU_STAGE3A::getPersistentSolidParticleStateDeviceView(pinning_owner);
    for (const PdTrialNode& node : corotated_trial.nodes) {
        PdTrialNode pd_node = node;
        pd_node.base_velocity = {};
        pd_node.velocity_increment = node.velocity;
        pd_trial.nodes.push_back(pd_node);
    }
    PdTrialParticle pd_particle;
    pd_particle.mu = 37.0;
    pd_trial.particles.push_back(pd_particle);
    PdTrialStateOutput pd_unpinned_state;
    evaluatePdTrialState3D(context, pd_trial, pd_unpinned_state);
    PdTrialStateInput pd_pinned_trial = pd_trial;
    auto& pin = pd_pinned_trial.particles[0];
    pin.pinning_k = 17.0;
    pin.pinning_d = 2.0;
    pin.pinning_target_x = {0.10, -0.15, 0.25};
    pin.pinning_target_v = {0.03, -0.02, 0.01};
    pin.pinning_from_resident_x0 = 1;
    PdTrialStateOutput pd_pinned_state;
    evaluatePdTrialState3D(context, pd_pinned_trial, pd_pinned_state);
    const double pinning_a = pin.pinning_k + pin.pinning_d / pd_trial.dt;
    const double owner_x0[3] = {0.18, -0.22, 0.31};
    const double target_x[3] = {pin.pinning_target_x.x,
                                pin.pinning_target_x.y,
                                pin.pinning_target_x.z};
    const double target_v[3] = {pin.pinning_target_v.x,
                                pin.pinning_target_v.y,
                                pin.pinning_target_v.z};
    double expected_pinning_energy = 0.0;
    for (int row = 0; row < 3; ++row) {
        const double target_next = target_x[row] + pd_trial.dt * target_v[row];
        const double pinning_b = pin.pinning_k * target_next +
            (pin.pinning_d / pd_trial.dt) *
                (owner_x0[row] + target_next - target_x[row]);
        const double expected_force =
            -pinning_a * pd_pinned_state.trial_position[row] + pinning_b;
        if (std::abs((pd_pinned_state.force[row] -
                         pd_unpinned_state.force[row]) - expected_force) > 1e-10)
            throw std::runtime_error(
                "solid resident ordinary-pinning force oracle mismatch");
        const double centered =
            pd_pinned_state.trial_position[row] - pinning_b / pinning_a;
        expected_pinning_energy += 0.5 * pinning_a * centered * centered;
        for (int col = 0; col < 3; ++col) {
            const int entry = 3 * row + col;
            const double expected_tangent = row == col ? pinning_a : 0.0;
            if (std::abs((pd_pinned_state.tangent_matrix[entry] -
                             pd_unpinned_state.tangent_matrix[entry]) -
                             expected_tangent) > 1e-10)
                throw std::runtime_error(
                    "solid resident ordinary-pinning tangent oracle mismatch");
        }
    }
    if (std::abs((pd_pinned_state.force_energy -
                     pd_unpinned_state.force_energy) -
                     expected_pinning_energy) > 1e-10 ||
        std::abs((pd_pinned_state.tangent_a[0] -
                     pd_unpinned_state.tangent_a[0]) - pinning_a) > 1e-10)
        throw std::runtime_error(
            "solid resident ordinary-pinning scalar oracle mismatch");
    // Example 3 combines Corotated, resident GHOST contact, ordinary pinning and
    // line search. Check the combined path by differencing two otherwise
    // identical resident GHOST states so the pinning term is isolated.
    CorotatedTrialStateInput resident_ghost_trial = ghost_trial;
    resident_ghost_trial.resident_particles =
        GPU_STAGE3A::getPersistentSolidParticleStateDeviceView(pinning_owner);
    PdTrialStateOutput resident_ghost_state;
    std::vector<double> resident_ghost_hessian;
    evaluateCorotatedTrialState3D(context, resident_ghost_trial,
        resident_ghost_state, resident_ghost_hessian);
    CorotatedTrialStateInput combined_trial = resident_ghost_trial;
    auto& corotated_pin = combined_trial.particles[0];
    corotated_pin.pinning_k = pin.pinning_k;
    corotated_pin.pinning_d = pin.pinning_d;
    corotated_pin.pinning_target_x = pin.pinning_target_x;
    corotated_pin.pinning_target_v = pin.pinning_target_v;
    corotated_pin.pinning_from_resident_x0 = 1;
    PdTrialStateOutput combined_state;
    std::vector<double> combined_hessian;
    evaluateCorotatedTrialState3D(
        context, combined_trial, combined_state, combined_hessian);
    double expected_corotated_pinning_energy = 0.0;
    for (int row = 0; row < 3; ++row) {
        const double target_next =
            target_x[row] + combined_trial.dt * target_v[row];
        const double pinning_b = corotated_pin.pinning_k * target_next +
            (corotated_pin.pinning_d / combined_trial.dt) *
                (owner_x0[row] + target_next - target_x[row]);
        const double expected_force =
            -pinning_a * combined_state.trial_position[row] + pinning_b;
        if (std::abs((combined_state.force[row] -
                         resident_ghost_state.force[row]) -
                         expected_force) > 1e-10)
            throw std::runtime_error(
                "solid Corotated resident GHOST+pinning force oracle mismatch");
        const double centered =
            combined_state.trial_position[row] - pinning_b / pinning_a;
        expected_corotated_pinning_energy +=
            0.5 * pinning_a * centered * centered;
        for (int col = 0; col < 3; ++col) {
            const int entry = 3 * row + col;
            const double expected_tangent = row == col ? pinning_a : 0.0;
            if (std::abs((combined_state.tangent_matrix[entry] -
                             resident_ghost_state.tangent_matrix[entry]) -
                             expected_tangent) > 1e-10)
                throw std::runtime_error(
                    "solid Corotated resident GHOST+pinning tangent oracle mismatch");
        }
    }
    if (std::abs((combined_state.force_energy -
                     resident_ghost_state.force_energy) -
                     expected_corotated_pinning_energy) > 1e-10 ||
        std::abs((combined_state.tangent_a[0] -
                     resident_ghost_state.tangent_a[0]) - pinning_a) > 1e-10)
        throw std::runtime_error(
            "solid Corotated resident GHOST+pinning scalar oracle mismatch");
    PdTrialStateInput pd_scattered_trial = pd_pinned_trial;
    pd_scattered_trial.scatter_residual = true;
    pd_scattered_trial.download_node_residual = true;
    PdTrialStateOutput pd_scattered_state;
    evaluatePdTrialState3D(context, pd_scattered_trial, pd_scattered_state);
    if (pd_scattered_state.node_residual.size() != 3 * input.nodes.size() ||
        !pd_scattered_state.force.empty() ||
        !pd_scattered_state.deformation_gradient.empty() ||
        std::abs(pd_scattered_state.force_energy -
                     pd_pinned_state.force_energy) > 1e-10)
        throw std::runtime_error(
            "solid resident line-search residual compaction mismatch");
    GPU_STAGE3A::invalidatePersistentSolidParticleState3D(pinning_owner);
    Matrix<double, 3, 3> cpu_stress;
    CorotatedIsotropic<double, 3>::Hessian cpu_hessian;
    cpu_corotated.firstPiola(cpu_scratch, cpu_stress);
    cpu_corotated.firstPiolaDerivative(cpu_scratch, cpu_hessian);
    std::vector<double> cpu_projected_hessian(81);
    std::vector<double> cpu_force_stress(9);
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            const int output = 3 * row + col;
            cpu_force_stress[output] = trial_particle.volume * cpu_stress(row, col);
            for (int source_row = 0; source_row < 3; ++source_row)
                for (int source_col = 0; source_col < 3; ++source_col) {
                    const int source = 3 * source_row + source_col;
                    cpu_projected_hessian[9 * output + source] =
                        cpu_hessian(row + 3 * col, source_row + 3 * source_col);
                }
        }
    const ErrorNorm stress_error =
        compareVectors(cpu_force_stress, corotated_state.force_stress);
    const ErrorNorm hessian_error =
        compareVectors(cpu_projected_hessian, projected_hessian);
    if (stress_error.max_abs > 1e-8 || stress_error.rel_l2 > 1e-8 ||
        hessian_error.max_abs > 1e-7 || hessian_error.rel_l2 > 1e-8)
        throw std::runtime_error("solid Corotated CPU/GPU constitutive oracle mismatch");
    CorotatedOperatorInput corotated_operator;
    corotated_operator.dx = input.dx;
    corotated_operator.dt = input.dt;
    corotated_operator.nodes = input.nodes;
    CorotatedParticle operator_particle;
    operator_particle.x = trial_particle.x;
    operator_particle.volume = trial_particle.volume;
    for (int i = 0; i < 81; ++i)
        operator_particle.projected_hessian[i] = projected_hessian[i];
    corotated_operator.particles.push_back(operator_particle);
    std::vector<double> corotated_y;
    applyCorotatedMatrixFree3D(context, corotated_operator, x, corotated_y);
    std::vector<double> host_corotated_y;
    applyCorotatedOnHost(corotated_operator, x, host_corotated_y);
    const ErrorNorm operator_error = compareVectors(host_corotated_y, corotated_y);
    if (operator_error.max_abs > 1e-10 || operator_error.rel_l2 > 1e-10)
        throw std::runtime_error("solid Corotated matrix-free CPU/GPU oracle mismatch");
    CorotatedTrialStateInput compact_corotated_trial = corotated_trial;
    compact_corotated_trial.scatter_residual = true;
    compact_corotated_trial.download_node_residual = true;
    PdTrialStateOutput compact_corotated_state;
    std::vector<double> compact_projected_hessian;
    evaluateCorotatedTrialState3D(context, compact_corotated_trial,
        compact_corotated_state, compact_projected_hessian);
    if (compact_corotated_state.node_residual.size() != 3 * input.nodes.size() ||
        !compact_corotated_state.force.empty() ||
        !compact_projected_hessian.empty() ||
        std::abs(compact_corotated_state.force_energy -
                     corotated_state.force_energy) > 1e-10)
        throw std::runtime_error(
            "solid Corotated compact trial-state result mismatch");
    CorotatedOperatorInput device_trial_operator = corotated_operator;
    device_trial_operator.use_device_trial = true;
    std::vector<double> device_trial_y;
    applyCorotatedMatrixFree3D(
        context, device_trial_operator, x, device_trial_y);
    const ErrorNorm device_trial_error =
        compareVectors(host_corotated_y, device_trial_y);
    if (device_trial_error.max_abs > 1e-10 ||
        device_trial_error.rel_l2 > 1e-10)
        throw std::runtime_error(
            "solid Corotated device-trial operator oracle mismatch");
    double quadratic_form = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(corotated_y[i]))
            throw std::runtime_error("solid Corotated operator produced non-finite value");
        quadratic_form += x[i] * corotated_y[i];
    }
    if (!(quadratic_form > 0))
        throw std::runtime_error("solid Corotated projected operator is not positive");
    std::vector<double> corotated_solution;
    solveCorotatedPcg3D(context, corotated_operator, x, 1e-7, 1e-12, 64,
        corotated_solution);
    for (double value : corotated_solution)
        if (!std::isfinite(value))
            throw std::runtime_error("solid Corotated PCG produced non-finite value");
    Eigen::MatrixXd dense_operator(x.size(), x.size());
    for (std::size_t column = 0; column < x.size(); ++column) {
        std::vector<double> basis(x.size(), 0.0), host_column;
        basis[column] = 1.0;
        applyCorotatedOnHost(corotated_operator, basis, host_column);
        for (std::size_t row = 0; row < x.size(); ++row)
            dense_operator(static_cast<Eigen::Index>(row),
                static_cast<Eigen::Index>(column)) = host_column[row];
    }
    const Eigen::Map<const Eigen::VectorXd> rhs(x.data(),
        static_cast<Eigen::Index>(x.size()));
    const Eigen::VectorXd cpu_solution = dense_operator.ldlt().solve(rhs);
    std::vector<double> host_solution(corotated_solution.size());
    for (std::size_t i = 0; i < host_solution.size(); ++i)
        host_solution[i] = cpu_solution[static_cast<Eigen::Index>(i)];
    const ErrorNorm pcg_error = compareVectors(host_solution, corotated_solution);
    if (pcg_error.max_abs > 1e-9 || pcg_error.rel_l2 > 1e-9)
        throw std::runtime_error("solid Corotated PCG CPU/GPU oracle mismatch");

    // The example 4 first-frame fixture starts from equilibrium, so its Newton
    // residual is exactly zero.  Build a nonzero gravity/material residual in
    // this compact fixture and cover the production-only chain explicitly:
    // device residual -> device PCG -> device line-search trial/energy/commit.
    const GPU_STAGE3A::Vec3d newton_gravity{0.0, -3.0, 0.0};
    std::vector<double> newton_rhs(compact_corotated_state.node_residual);
    for (const SolidNode& node : input.nodes) {
        const std::size_t offset = static_cast<std::size_t>(3 * node.idx);
        newton_rhs[offset + 1] +=
            corotated_trial.dt * newton_gravity.y * node.mass;
    }
    std::vector<double> host_newton_solution;
    solveCorotatedPcg3D(context, corotated_operator, newton_rhs, 1e-10,
        1e-12, 128, host_newton_solution);
    const double device_residual_norm = preparePdNewtonResidual3D(
        context, newton_gravity, corotated_trial.dt);
    if (!(device_residual_norm > 0.0) || !std::isfinite(device_residual_norm))
        throw std::runtime_error(
            "solid Corotated device Newton residual fixture mismatch");
    CorotatedOperatorInput device_newton_operator;
    device_newton_operator.dx = corotated_operator.dx;
    device_newton_operator.dt = corotated_operator.dt;
    device_newton_operator.reuse_topology = true;
    device_newton_operator.use_device_trial = true;
    PdPcgStats device_newton_stats;
    solveCorotatedPcgDeviceResidual3D(context, device_newton_operator, 1e-10,
        1e-12, 128, &device_newton_stats);
    if (device_newton_stats.iterations <= 0 ||
        !std::isfinite(device_newton_stats.final_preconditioned_norm))
        throw std::runtime_error(
            "solid Corotated device-residual PCG did not iterate");
    constexpr double accepted_step = 0.5;
    setPdNewtonLineSearchTrial3D(context, accepted_step);
    CorotatedTrialStateInput device_line_search;
    device_line_search.dx = corotated_trial.dx;
    device_line_search.dt = corotated_trial.dt;
    device_line_search.reuse_static = true;
    device_line_search.reuse_velocity = true;
    device_line_search.scatter_residual = true;
    device_line_search.download_node_residual = false;
    PdTrialStateOutput device_line_search_state;
    std::vector<double> device_line_search_hessian;
    evaluateCorotatedTrialState3D(context, device_line_search,
        device_line_search_state, device_line_search_hessian);
    const double device_line_search_energy = computePdNewtonTotalEnergy3D(
        context, newton_gravity, corotated_trial.dt, accepted_step);
    if (!std::isfinite(device_line_search_energy) ||
        !device_line_search_state.node_residual.empty() ||
        !device_line_search_hessian.empty())
        throw std::runtime_error(
            "solid Corotated device line-search fixture mismatch");
    commitPdNewtonLineSearchTrial3D(context, accepted_step);
    std::vector<double> committed_increment;
    std::vector<double> committed_grad_velocity;
    downloadPdNewtonState3D(
        context, committed_increment, committed_grad_velocity);
    std::vector<double> expected_committed_increment(host_newton_solution);
    for (double& value : expected_committed_increment)
        value *= accepted_step;
    const ErrorNorm device_newton_error = compareVectors(
        expected_committed_increment, committed_increment);
    if (device_newton_error.max_abs > 1e-9 ||
        device_newton_error.rel_l2 > 1e-9 ||
        committed_grad_velocity.size() != 9)
        throw std::runtime_error(
            "solid Corotated resident Newton/line-search oracle mismatch");
    std::cout << "solid implicit PD operator fixture passed; norm_sq="
              << squared_norm << "; Corotated quadratic_form=" << quadratic_form
              << "; Corotated stress(max/rel)=" << stress_error.max_abs << '/'
              << stress_error.rel_l2 << "; hessian(max/rel)="
              << hessian_error.max_abs << '/' << hessian_error.rel_l2
              << "; operator(max/rel)=" << operator_error.max_abs << '/'
              << operator_error.rel_l2 << "; pcg(max/rel)="
              << pcg_error.max_abs << '/' << pcg_error.rel_l2
              << "; device_newton(max/rel)=" << device_newton_error.max_abs
              << '/' << device_newton_error.rel_l2
              << "; device_newton_iterations="
              << device_newton_stats.iterations << '\n';
#else
    std::cout << "solid implicit PD operator fixture passed; norm_sq="
              << squared_norm << '\n';
#endif
    return 0;
}
