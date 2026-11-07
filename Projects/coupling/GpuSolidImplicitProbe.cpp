#include "GpuSolidImplicitProbe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace ZIRAN::GPU_SOLID_IMPLICIT {
namespace {

using GridKey = GPU_STAGE3A::GridKey;

int findNode(const std::vector<std::pair<GridKey, int>>& lookup, GridKey key)
{
    const auto it = std::lower_bound(lookup.begin(), lookup.end(), key,
        [](const auto& entry, GridKey value) { return entry.first < value; });
    return it != lookup.end() && it->first == key ? it->second : -1;
}

void quadraticWeights(double coordinate, int& base, double w[3], double dw[3])
{
    base = static_cast<int>(std::floor(coordinate - 0.5));
    const double d0 = coordinate - static_cast<double>(base);
    const double z = 1.5 - d0;
    w[0] = 0.5 * z * z;
    const double d1 = d0 - 1.0;
    w[1] = 0.75 - d1 * d1;
    const double d2 = 1.0 - d1;
    const double zz = 1.5 - d2;
    w[2] = 0.5 * zz * zz;
    dw[0] = -z;
    dw[1] = -2.0 * d1;
    dw[2] = zz;
}

} // namespace

void evaluatePdTrialState3D(PersistentPdOperator3D&, const PdTrialStateInput&,
    PdTrialStateOutput&, PdTrialStateStats*)
{
    // This symbol is kept in the CPU-only build so the common coupling source
    // links unchanged. The caller only installs the device-state callback in
    // CUDA builds, therefore reaching this path is a configuration error.
    throw std::runtime_error("solid PD trial-state CUDA path is unavailable in this build");
}

void downloadPdTrialGradVelocity3D(PersistentPdOperator3D&,
    std::vector<double>&)
{
    throw std::runtime_error(
        "solid PD trial-state CUDA path is unavailable in this build");
}

double preparePdNewtonResidual3D(PersistentPdOperator3D&,
    const GPU_STAGE3A::Vec3d&, double)
{
    throw std::runtime_error(
        "solid PD device Newton residual is unavailable in this build");
}

void solvePdPcgDeviceResidual3D(PersistentPdOperator3D&,
    const PdOperatorInput&, double, double, int, PdPcgStats*)
{
    throw std::runtime_error(
        "solid PD device Newton PCG is unavailable in this build");
}

void applyPdNewtonIncrement3D(PersistentPdOperator3D&)
{
    throw std::runtime_error(
        "solid PD device Newton increment is unavailable in this build");
}

void setPdNewtonLineSearchTrial3D(PersistentPdOperator3D&, double)
{
    throw std::runtime_error(
        "solid PD device line search is unavailable in this build");
}

void commitPdNewtonLineSearchTrial3D(PersistentPdOperator3D&, double)
{
    throw std::runtime_error(
        "solid PD device line search is unavailable in this build");
}

double computePdNewtonTotalEnergy3D(PersistentPdOperator3D&,
    const GPU_STAGE3A::Vec3d&, double, double)
{
    throw std::runtime_error(
        "solid PD device total energy is unavailable in this build");
}

void downloadPdNewtonState3D(PersistentPdOperator3D&,
    std::vector<double>&, std::vector<double>&)
{
    throw std::runtime_error(
        "solid PD device Newton state is unavailable in this build");
}

void exportFinalGridVelocityToNodeMap3D(PersistentPdOperator3D&,
    GPU_STAGE3A::DeviceGridNodeMap&)
{
    throw std::runtime_error(
        "solid final grid export is unavailable in this build");
}

void evaluateCorotatedTrialState3D(PersistentPdOperator3D&,
    const CorotatedTrialStateInput&, PdTrialStateOutput&, std::vector<double>&,
    PdTrialStateStats*)
{
    throw std::runtime_error("solid Corotated trial-state CUDA path is unavailable in this build");
}

void applyCorotatedMatrixFree3D(PersistentPdOperator3D&,
    const CorotatedOperatorInput&, const std::vector<double>&,
    std::vector<double>&, PdOperatorStats*)
{
    throw std::runtime_error("solid Corotated CUDA operator path is unavailable in this build");
}

void solveCorotatedPcg3D(PersistentPdOperator3D&,
    const CorotatedOperatorInput&, const std::vector<double>&,
    double, double, int, std::vector<double>&, PdPcgStats*)
{
    throw std::runtime_error("solid Corotated CUDA PCG path is unavailable in this build");
}

void solveCorotatedPcgDeviceResidual3D(PersistentPdOperator3D&,
    const CorotatedOperatorInput&, double, double, int, PdPcgStats*)
{
    throw std::runtime_error(
        "solid Corotated device-residual CUDA PCG path is unavailable in this build");
}

void solveLinearCorotatedPcg3D(PersistentPdOperator3D&,
    const LinearCorotatedOperatorInput&, const std::vector<double>&,
    double, double, int, std::vector<double>&, PdPcgStats*)
{
    throw std::runtime_error(
        "solid linear-Corotated CUDA PCG path is unavailable in this build");
}

void applyPdMatrixFree3D(PersistentPdOperator3D&, const PdOperatorInput& input,
    const std::vector<double>& x, std::vector<double>& y, PdOperatorStats* stats)
{
    const auto begin = std::chrono::high_resolution_clock::now();
    if (!(input.dx > 0.0))
        throw std::runtime_error("solid PD operator requires positive dx");
    if (x.size() != input.nodes.size() * 3)
        throw std::runtime_error("solid PD operator vector size mismatch");

    std::vector<std::pair<GridKey, int>> lookup;
    lookup.reserve(input.nodes.size());
    for (const SolidNode& node : input.nodes) {
        if (node.idx < 0 || static_cast<std::size_t>(node.idx) >= input.nodes.size())
            throw std::runtime_error("solid PD operator node index out of range");
        lookup.emplace_back(node.key, node.idx);
    }
    std::sort(lookup.begin(), lookup.end());
    for (std::size_t i = 1; i < lookup.size(); ++i)
        if (lookup[i - 1].first == lookup[i].first)
            throw std::runtime_error("solid PD operator duplicate node key");

    y.assign(x.size(), 0.0);
    for (const SolidNode& node : input.nodes)
        for (int d = 0; d < 3; ++d)
            y[static_cast<std::size_t>(3 * node.idx + d)] =
                node.mass * x[static_cast<std::size_t>(3 * node.idx + d)];

    const double scale = input.dt * input.dt;
    for (const PdParticle& particle : input.particles) {
        int base[3];
        double w[3][3], dw[3][3];
        quadraticWeights(particle.x.x / input.dx, base[0], w[0], dw[0]);
        quadraticWeights(particle.x.y / input.dx, base[1], w[1], dw[1]);
        quadraticWeights(particle.x.z / input.dx, base[2], w[2], dw[2]);
        double grad_x[3][3] = {};
        double particle_x[3] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) {
                    const int node = findNode(lookup, GPU_STAGE3A::packGridKey3D(
                        base[0] + i, base[1] + j, base[2] + k));
                    if (node < 0)
                        continue;
                    const double grad[3] = {
                        dw[0][i] * w[1][j] * w[2][k] / input.dx,
                        w[0][i] * dw[1][j] * w[2][k] / input.dx,
                        w[0][i] * w[1][j] * dw[2][k] / input.dx};
                    for (int row = 0; row < 3; ++row)
                        for (int col = 0; col < 3; ++col)
                            grad_x[row][col] += x[static_cast<std::size_t>(3 * node + row)] * grad[col];
                    for (int row = 0; row < 3; ++row)
                        particle_x[row] += x[static_cast<std::size_t>(3 * node + row)] *
                            w[0][i] * w[1][j] * w[2][k];
                }
        const double stress_scale = scale * 2.0 * particle.mu * particle.volume;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) {
                    const int node = findNode(lookup, GPU_STAGE3A::packGridKey3D(
                        base[0] + i, base[1] + j, base[2] + k));
                    if (node < 0)
                        continue;
                    const double grad[3] = {
                        dw[0][i] * w[1][j] * w[2][k] / input.dx,
                        w[0][i] * dw[1][j] * w[2][k] / input.dx,
                        w[0][i] * w[1][j] * dw[2][k] / input.dx};
                    for (int row = 0; row < 3; ++row)
                        for (int col = 0; col < 3; ++col)
                            y[static_cast<std::size_t>(3 * node + row)] +=
                                stress_scale * grad_x[row][col] * grad[col];
                    for (int row = 0; row < 3; ++row)
                        y[static_cast<std::size_t>(3 * node + row)] +=
                            scale * particle.pinning_a * particle_x[row] *
                            w[0][i] * w[1][j] * w[2][k];
                }
    }
    if (stats) {
        stats->nodes = input.nodes.size();
        stats->particles = input.particles.size();
        stats->total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - begin).count();
    }
}

void solvePdPcg3D(PersistentPdOperator3D& context, const PdOperatorInput& input,
    const std::vector<double>& rhs, double relative_tolerance,
    double absolute_tolerance, int max_iterations, std::vector<double>& solution,
    PdPcgStats* stats)
{
    if (rhs.size() != input.nodes.size() * 3)
        throw std::runtime_error("solid PD PCG rhs size mismatch");
    const auto begin = std::chrono::high_resolution_clock::now();
    solution.assign(rhs.size(), 0.0);
    std::vector<double> r = rhs, z(rhs.size()), p(rhs.size()), q;
    for (const SolidNode& node : input.nodes)
        for (int d = 0; d < 3; ++d) {
            const std::size_t i = static_cast<std::size_t>(3 * node.idx + d);
            z[i] = r[i] / node.mass;
        }
    p = z;
    auto dot = [](const std::vector<double>& a, const std::vector<double>& b) {
        double result = 0;
        for (std::size_t i = 0; i < a.size(); ++i)
            result += a[i] * b[i];
        return result;
    };
    double rz = dot(r, z);
    const double initial = std::sqrt(std::max(0.0, rz));
    const double tolerance = std::min(relative_tolerance * initial, absolute_tolerance);
    int iteration = 0;
    for (; iteration < max_iterations && std::sqrt(std::max(0.0, rz)) > tolerance;
         ++iteration) {
        applyPdMatrixFree3D(context, input, p, q);
        const double alpha = rz / dot(p, q);
        for (std::size_t i = 0; i < solution.size(); ++i) {
            solution[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        for (const SolidNode& node : input.nodes)
            for (int d = 0; d < 3; ++d) {
                const std::size_t i = static_cast<std::size_t>(3 * node.idx + d);
                z[i] = r[i] / node.mass;
            }
        const double new_rz = dot(r, z);
        const double beta = new_rz / rz;
        for (std::size_t i = 0; i < p.size(); ++i)
            p[i] = z[i] + beta * p[i];
        rz = new_rz;
    }
    if (stats) {
        stats->iterations = iteration;
        stats->initial_preconditioned_norm = initial;
        stats->final_preconditioned_norm = std::sqrt(std::max(0.0, rz));
        stats->solve_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - begin).count();
    }
}

} // namespace ZIRAN::GPU_SOLID_IMPLICIT
