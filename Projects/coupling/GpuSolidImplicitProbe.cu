#include "GpuSolidImplicitProbe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

namespace ZIRAN::GPU_SOLID_IMPLICIT {
namespace {

using GridKey = GPU_STAGE3A::GridKey;

constexpr int kCoordBits = 21;
constexpr int kCoordBias = 1 << (kCoordBits - 1);

void checkCuda(cudaError_t error, const char* label)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(error));
}

struct DeviceState {
    thrust::device_vector<GridKey> node_keys;
    thrust::device_vector<int> node_indices;
    thrust::device_vector<double> mass;
    thrust::device_vector<double> projection;
    thrust::device_vector<PdParticle> particles;
    thrust::device_vector<LinearCorotatedParticle> linear_particles;
    thrust::device_vector<CorotatedParticle> corotated_particles;
    thrust::device_vector<PdTrialParticle> trial_particles;
    thrust::device_vector<CorotatedTrialParticle> corotated_trial_particles;
    thrust::device_vector<PdParallelPinningTarget> parallel_targets;
    struct TrialPinningPatch {
        double pinning_a = 0;
        GPU_STAGE3A::Vec3d pinning_b;
        double pinning_k = 0;
        double pinning_d = 0;
        GPU_STAGE3A::Vec3d pinning_target_x;
        GPU_STAGE3A::Vec3d pinning_target_v;
        int pinning_from_resident_x0 = 0;
        int parallel_target_begin = 0;
        int parallel_target_count = 0;
    };
    thrust::device_vector<TrialPinningPatch> trial_pinning_patches;
    thrust::device_vector<double> base_velocity;
    thrust::device_vector<double> newton_velocity_increment;
    thrust::device_vector<double> node_velocity;
    thrust::device_vector<double> trial_velocity;
    thrust::device_vector<double> trial_grad_velocity;
    thrust::device_vector<double> trial_F;
    thrust::device_vector<double> trial_J;
    thrust::device_vector<double> trial_position;
    thrust::device_vector<double> trial_force_stress;
    thrust::device_vector<double> trial_force;
    thrust::device_vector<double> trial_tangent_a;
    thrust::device_vector<double> trial_tangent_matrix;
    thrust::device_vector<double> trial_contact_a;
    thrust::device_vector<double> trial_energy;
    // Node-indexed residual force scattered on-device from force_stress/fp,
    // reproducing rasterizeForceToTVStack(scale=dt) so the host residual no
    // longer needs the full per-particle scratch download.
    thrust::device_vector<double> trial_node_residual;
    thrust::device_vector<double> trial_corotated_hessian;
    thrust::device_vector<GridKey> contact_keys;
    thrust::device_vector<int> contact_indices;
    thrust::device_vector<double> x;
    thrust::device_vector<double> y;
    thrust::device_vector<double> r;
    thrust::device_vector<double> z;
    thrust::device_vector<double> p;
    thrust::device_vector<double> q;
};

DeviceState& stateFor(PersistentPdOperator3D& context)
{
    if (!context.storage)
        context.storage = std::make_shared<DeviceState>();
    return *std::static_pointer_cast<DeviceState>(context.storage);
}

__device__ GridKey packGridKey(int x, int y, int z)
{
    return (static_cast<GridKey>(x + kCoordBias) << (2 * kCoordBits)) |
        (static_cast<GridKey>(y + kCoordBias) << kCoordBits) |
        static_cast<GridKey>(z + kCoordBias);
}

__device__ int lowerBoundKey(const GridKey* keys, int count, GridKey key)
{
    int first = 0;
    int last = count;
    while (first < last) {
        const int middle = first + ((last - first) >> 1);
        if (keys[middle] < key)
            first = middle + 1;
        else
            last = middle;
    }
    return first;
}

__device__ void quadraticWeights(double coordinate, int& base,
    double w[3], double dw[3])
{
    base = static_cast<int>(floor(coordinate - 0.5));
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

__device__ double axisAlignedBoxSignedDistance(const GPU_STAGE3A::Vec3d& x,
    const GPU_STAGE3A::Vec3d& lower,
    const GPU_STAGE3A::Vec3d& upper)
{
    const double center[3] = {0.5 * (lower.x + upper.x),
        0.5 * (lower.y + upper.y), 0.5 * (lower.z + upper.z)};
    const double half[3] = {0.5 * (upper.x - lower.x),
        0.5 * (upper.y - lower.y), 0.5 * (upper.z - lower.z)};
    const double value[3] = {x.x, x.y, x.z};
    double d[3], outside[3];
    for (int axis = 0; axis < 3; ++axis) {
        d[axis] = fabs(value[axis] - center[axis]) - half[axis];
        outside[axis] = fmax(d[axis], 0.0);
    }
    return fmin(fmax(d[0], fmax(d[1], d[2])), 0.0) +
        sqrt(outside[0] * outside[0] + outside[1] * outside[1] +
            outside[2] * outside[2]);
}

__device__ GPU_STAGE3A::Vec3d axisAlignedBoxNormal(
    const GPU_STAGE3A::Vec3d& x, const GPU_STAGE3A::Vec3d& lower,
    const GPU_STAGE3A::Vec3d& upper)
{
    const double center[3] = {0.5 * (lower.x + upper.x),
        0.5 * (lower.y + upper.y), 0.5 * (lower.z + upper.z)};
    const double half[3] = {0.5 * (upper.x - lower.x),
        0.5 * (upper.y - lower.y), 0.5 * (upper.z - lower.z)};
    const double value[3] = {x.x, x.y, x.z};
    double d[3], sign[3], outside[3];
    for (int axis = 0; axis < 3; ++axis) {
        sign[axis] = value[axis] - center[axis] >= 0.0 ? 1.0 : -1.0;
        d[axis] = fabs(value[axis] - center[axis]) - half[axis];
        outside[axis] = fmax(d[axis], 0.0);
    }
    const double norm = sqrt(outside[0] * outside[0] +
        outside[1] * outside[1] + outside[2] * outside[2]);
    if (norm > 0.0)
        return {sign[0] * outside[0] / norm,
            sign[1] * outside[1] / norm,
            sign[2] * outside[2] / norm};
    int axis = 0;
    if (d[1] > d[axis]) axis = 1;
    if (d[2] > d[axis]) axis = 2;
    GPU_STAGE3A::Vec3d normal;
    if (axis == 0) normal.x = sign[0];
    else if (axis == 1) normal.y = sign[1];
    else normal.z = sign[2];
    return normal;
}

__device__ double ghostBoxShellSignedDistanceAndNormal(
    const GPU_STAGE3A::Vec3d& x, const GhostBoxShell3D& collision,
    GPU_STAGE3A::Vec3d& normal)
{
    const double outer_distance = axisAlignedBoxSignedDistance(
        x, collision.outer_min, collision.outer_max);
    const GPU_STAGE3A::Vec3d outer_normal = axisAlignedBoxNormal(
        x, collision.outer_min, collision.outer_max);
    const double inner_distance = axisAlignedBoxSignedDistance(
        x, collision.inner_min, collision.inner_max);
    const GPU_STAGE3A::Vec3d inner_normal = axisAlignedBoxNormal(
        x, collision.inner_min, collision.inner_max);
    double distance = outer_distance;
    normal = outer_normal;
    if (-inner_distance > distance) {
        distance = -inner_distance;
        normal = {-inner_normal.x, -inner_normal.y, -inner_normal.z};
    }
    if (collision.has_hole) {
        const double hole_distance = axisAlignedBoxSignedDistance(
            x, collision.hole_min, collision.hole_max);
        const GPU_STAGE3A::Vec3d hole_normal = axisAlignedBoxNormal(
            x, collision.hole_min, collision.hole_max);
        if (-hole_distance > distance) {
            distance = -hole_distance;
            normal = {-hole_normal.x, -hole_normal.y, -hole_normal.z};
        }
    }
    return distance;
}

__device__ double determinant3x3(const double a[3][3])
{
    return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
        a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
        a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

__device__ bool inverseTranspose3x3(const double a[3][3], double out[3][3])
{
    const double det = determinant3x3(a);
    if (!isfinite(det) || fabs(det) < 1e-18)
        return false;
    const double inv_det = 1.0 / det;
    // inverse-transpose, directly in row-major form.
    out[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det;
    out[0][1] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det;
    out[0][2] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det;
    out[1][0] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det;
    out[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
    out[1][2] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det;
    out[2][0] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
    out[2][1] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det;
    out[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;
    return true;
}

// Newton polar iteration. The solid test path stays close to det(F)=1, so
// eight iterations are sufficient to reach double precision while avoiding a
// global SVD dependency in this CUDA translation unit.  A singular trial is
// conservatively mapped to identity; the host oracle rejects such a state.
__device__ void polarRotation3x3(const double F[3][3], double R[3][3])
{
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            R[row][col] = F[row][col];
    for (int iteration = 0; iteration < 8; ++iteration) {
        double inverse_transpose[3][3];
        if (!inverseTranspose3x3(R, inverse_transpose)) {
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    R[row][col] = row == col ? 1.0 : 0.0;
            return;
        }
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                R[row][col] = 0.5 * (R[row][col] + inverse_transpose[row][col]);
    }
}

__device__ double determinant3x3Flat(const double a[3][3])
{
    return determinant3x3(a);
}

// One-sided Jacobi SVD, adapted from the existing CUDA operator probe.  It
// intentionally returns U/S/V rather than only F^{-T}: CorotatedIsotropic's
// projected Hessian is expressed in this diagonal frame.  The sign convention
// matches Ziran's CPU SVD contract: singular values are sorted by magnitude
// and, for an inverted F, the smallest one carries the sign so U and V remain
// proper rotations.
__device__ void oneSidedJacobiSvd3x3(const double F[3][3], double U[3][3],
    double sigma[3], double V[3][3])
{
    double B[3][3];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            B[row][col] = F[row][col];
            V[row][col] = row == col ? 1.0 : 0.0;
        }
    for (int sweep = 0; sweep < 12; ++sweep) {
        for (int pair = 0; pair < 3; ++pair) {
            const int p = pair == 0 ? 0 : (pair == 1 ? 0 : 1);
            const int q = pair == 0 ? 1 : (pair == 1 ? 2 : 2);
            double alpha = 0, beta = 0, gamma = 0;
            for (int row = 0; row < 3; ++row) {
                alpha += B[row][p] * B[row][p];
                beta += B[row][q] * B[row][q];
                gamma += B[row][p] * B[row][q];
            }
            if (gamma == 0.0 || fabs(gamma) <= 1e-15 * sqrt(fmax(alpha * beta, 0.0)))
                continue;
            const double zeta = (beta - alpha) / (2.0 * gamma);
            const double sign = zeta >= 0.0 ? 1.0 : -1.0;
            const double t = sign / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
            const double c = 1.0 / sqrt(1.0 + t * t);
            const double s = c * t;
            for (int row = 0; row < 3; ++row) {
                const double bp = B[row][p], bq = B[row][q];
                B[row][p] = c * bp - s * bq;
                B[row][q] = s * bp + c * bq;
                const double vp = V[row][p], vq = V[row][q];
                V[row][p] = c * vp - s * vq;
                V[row][q] = s * vp + c * vq;
            }
        }
    }
    for (int col = 0; col < 3; ++col) {
        double norm_sq = 0;
        for (int row = 0; row < 3; ++row)
            norm_sq += B[row][col] * B[row][col];
        sigma[col] = sqrt(norm_sq);
        const double inv_sigma = sigma[col] > 1e-15 ? 1.0 / sigma[col] : 0.0;
        for (int row = 0; row < 3; ++row)
            U[row][col] = B[row][col] * inv_sigma;
    }
    // Sort decreasing absolute value (all values are nonnegative here).
    for (int i = 0; i < 2; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (sigma[j] > sigma[i]) {
                const double value = sigma[i];
                sigma[i] = sigma[j];
                sigma[j] = value;
                for (int row = 0; row < 3; ++row) {
                    const double u = U[row][i];
                    U[row][i] = U[row][j];
                    U[row][j] = u;
                    const double v = V[row][i];
                    V[row][i] = V[row][j];
                    V[row][j] = v;
                }
            }
    // The CPU helper keeps U/V rotations and places an inversion sign on the
    // smallest singular value. This also gives the closest proper rotation.
    if (determinant3x3Flat(F) < 0.0) {
        sigma[2] = -sigma[2];
        for (int row = 0; row < 3; ++row)
            U[row][2] = -U[row][2];
    }
}

__device__ double clampSmallMagnitude(double x, double eps)
{
    if (x < -eps)
        return x;
    if (x < 0.0)
        return -eps;
    return x < eps ? eps : x;
}

__device__ void cofactor3x3(const double F[3][3], double C[3][3])
{
    C[0][0] = F[1][1] * F[2][2] - F[1][2] * F[2][1];
    C[0][1] = F[1][2] * F[2][0] - F[1][0] * F[2][2];
    C[0][2] = F[1][0] * F[2][1] - F[1][1] * F[2][0];
    C[1][0] = F[0][2] * F[2][1] - F[0][1] * F[2][2];
    C[1][1] = F[0][0] * F[2][2] - F[0][2] * F[2][0];
    C[1][2] = F[0][1] * F[2][0] - F[0][0] * F[2][1];
    C[2][0] = F[0][1] * F[1][2] - F[0][2] * F[1][1];
    C[2][1] = F[0][2] * F[1][0] - F[0][0] * F[1][2];
    C[2][2] = F[0][0] * F[1][1] - F[0][1] * F[1][0];
}

__device__ void projectPSD2x2(double B[2][2])
{
    // Exact counterpart of EIGEN_EXT::makePD2d().
    const double a = B[0][0];
    const double b = 0.5 * (B[0][1] + B[1][0]);
    const double d = B[1][1];
    const double b2 = b * b;
    const double determinant = a * d - b2;
    const double half_trace = 0.5 * (a + d);
    const double root = sqrt(fabs(half_trace * half_trace - determinant));
    const double lambda2 = half_trace - root;
    if (lambda2 >= 0.0)
        return;
    const double lambda1 = half_trace + root;
    if (lambda1 <= 0.0) {
        B[0][0] = B[0][1] = B[1][0] = B[1][1] = 0.0;
        return;
    }
    if (b2 == 0.0) {
        B[0][0] = lambda1;
        B[0][1] = B[1][0] = B[1][1] = 0.0;
        return;
    }
    const double lambda1_minus_d = lambda1 - d;
    const double ratio = lambda1_minus_d / lambda1;
    B[0][0] = ratio * lambda1_minus_d;
    B[0][1] = B[1][0] = b * ratio;
    B[1][1] = b2 / lambda1;
}

__device__ void projectPSD3x3(double A[3][3])
{
    double Q[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    // Symmetric Jacobi eigensolve. The projected matrix is invariant to the
    // eigenvector signs/order, unlike an SVD output comparison.
    for (int sweep = 0; sweep < 10; ++sweep) {
        for (int pair = 0; pair < 3; ++pair) {
            const int p = pair == 0 ? 0 : (pair == 1 ? 0 : 1);
            const int q = pair == 0 ? 1 : (pair == 1 ? 2 : 2);
            const double apq = A[p][q];
            if (fabs(apq) <= 1e-15 * (fabs(A[p][p]) + fabs(A[q][q]) + 1.0))
                continue;
            const double tau = (A[q][q] - A[p][p]) / (2.0 * apq);
            const double sign = tau >= 0.0 ? 1.0 : -1.0;
            const double t = sign / (fabs(tau) + sqrt(1.0 + tau * tau));
            const double c = 1.0 / sqrt(1.0 + t * t);
            const double s = t * c;
            const double app = A[p][p], aqq = A[q][q];
            A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            A[p][q] = A[q][p] = 0.0;
            for (int r = 0; r < 3; ++r) {
                if (r == p || r == q)
                    continue;
                const double arp = A[r][p], arq = A[r][q];
                A[r][p] = A[p][r] = c * arp - s * arq;
                A[r][q] = A[q][r] = s * arp + c * arq;
            }
            for (int r = 0; r < 3; ++r) {
                const double qrp = Q[r][p], qrq = Q[r][q];
                Q[r][p] = c * qrp - s * qrq;
                Q[r][q] = s * qrp + c * qrq;
            }
        }
    }
    double result[3][3] = {};
    for (int k = 0; k < 3; ++k) {
        const double eigenvalue = fmax(A[k][k], 0.0);
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                result[row][col] += eigenvalue * Q[row][k] * Q[col][k];
    }
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            A[row][col] = result[row][col];
}

struct CorotatedProjectedBlocks {
    double A[3][3];
    double B01[2][2];
    double B12[2][2];
    double B20[2][2];
};

__device__ void buildCorotatedProjectedBlocks(const double sigma[3], double mu,
    double lambda, CorotatedProjectedBlocks& blocks)
{
    const double product[3] = {
        sigma[1] * sigma[2], sigma[0] * sigma[2], sigma[0] * sigma[1]};
    const double J = sigma[0] * sigma[1] * sigma[2];
    const double twice_mu = 2.0 * mu;
    const double lambda_j_minus_one = lambda * (J - 1.0);
    const double psi[3] = {
        twice_mu * (sigma[0] - 1.0) + lambda_j_minus_one * product[0],
        twice_mu * (sigma[1] - 1.0) + lambda_j_minus_one * product[1],
        twice_mu * (sigma[2] - 1.0) + lambda_j_minus_one * product[2]};
    blocks.A[0][0] = twice_mu + lambda * product[0] * product[0];
    blocks.A[1][1] = twice_mu + lambda * product[1] * product[1];
    blocks.A[2][2] = twice_mu + lambda * product[2] * product[2];
    blocks.A[0][1] = blocks.A[1][0] =
        lambda_j_minus_one * sigma[2] + lambda * product[0] * product[1];
    blocks.A[0][2] = blocks.A[2][0] =
        lambda_j_minus_one * sigma[1] + lambda * product[0] * product[2];
    blocks.A[1][2] = blocks.A[2][1] =
        lambda_j_minus_one * sigma[0] + lambda * product[1] * product[2];
    const double m01 = twice_mu - lambda_j_minus_one * sigma[2];
    const double m02 = twice_mu - lambda_j_minus_one * sigma[1];
    const double m12 = twice_mu - lambda_j_minus_one * sigma[0];
    const double p01 = (psi[0] + psi[1]) /
        clampSmallMagnitude(sigma[0] + sigma[1], 1e-6);
    const double p02 = (psi[0] + psi[2]) /
        clampSmallMagnitude(sigma[0] + sigma[2], 1e-6);
    const double p12 = (psi[1] + psi[2]) /
        clampSmallMagnitude(sigma[1] + sigma[2], 1e-6);
    blocks.B01[0][0] = blocks.B01[1][1] = 0.5 * (m01 + p01);
    blocks.B01[0][1] = blocks.B01[1][0] = 0.5 * (m01 - p01);
    blocks.B12[0][0] = blocks.B12[1][1] = 0.5 * (m12 + p12);
    blocks.B12[0][1] = blocks.B12[1][0] = 0.5 * (m12 - p12);
    blocks.B20[0][0] = blocks.B20[1][1] = 0.5 * (m02 + p02);
    blocks.B20[0][1] = blocks.B20[1][0] = 0.5 * (m02 - p02);
    projectPSD3x3(blocks.A);
    projectPSD2x2(blocks.B01);
    projectPSD2x2(blocks.B12);
    projectPSD2x2(blocks.B20);
}

__device__ void applyCorotatedProjectedBlocks(const double U[3][3],
    const double V[3][3], const CorotatedProjectedBlocks& blocks,
    const double dF[3][3], double dP[3][3])
{
    double D[3][3] = {};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            for (int q = 0; q < 3; ++q)
                for (int r = 0; r < 3; ++r)
                    D[row][col] += U[q][row] * dF[q][r] * V[r][col];
    double K[3][3] = {};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            K[row][row] += blocks.A[row][col] * D[col][col];
    K[0][1] = blocks.B01[0][0] * D[0][1] + blocks.B01[0][1] * D[1][0];
    K[1][0] = blocks.B01[1][0] * D[0][1] + blocks.B01[1][1] * D[1][0];
    K[1][2] = blocks.B12[0][0] * D[1][2] + blocks.B12[0][1] * D[2][1];
    K[2][1] = blocks.B12[1][0] * D[1][2] + blocks.B12[1][1] * D[2][1];
    K[0][2] = blocks.B20[0][0] * D[0][2] + blocks.B20[0][1] * D[2][0];
    K[2][0] = blocks.B20[1][0] * D[0][2] + blocks.B20[1][1] * D[2][0];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            dP[row][col] = 0.0;
            for (int q = 0; q < 3; ++q)
                for (int r = 0; r < 3; ++r)
                    dP[row][col] += U[row][q] * K[q][r] * V[col][r];
        }
}

__device__ void corotatedProjectedDifferential(const double U[3][3],
    const double sigma[3], const double V[3][3], double mu, double lambda,
    const double dF[3][3], double dP[3][3])
{
    double D[3][3] = {};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            for (int q = 0; q < 3; ++q)
                for (int r = 0; r < 3; ++r)
                    D[row][col] += U[q][row] * dF[q][r] * V[r][col];
    const double product[3] = {
        sigma[1] * sigma[2], sigma[0] * sigma[2], sigma[0] * sigma[1]};
    const double J = sigma[0] * sigma[1] * sigma[2];
    const double twice_mu = 2.0 * mu;
    const double lambda_j_minus_one = lambda * (J - 1.0);
    const double psi[3] = {
        twice_mu * (sigma[0] - 1.0) + lambda_j_minus_one * product[0],
        twice_mu * (sigma[1] - 1.0) + lambda_j_minus_one * product[1],
        twice_mu * (sigma[2] - 1.0) + lambda_j_minus_one * product[2]};
    double A[3][3] = {
        {twice_mu + lambda * product[0] * product[0],
         lambda_j_minus_one * sigma[2] + lambda * product[0] * product[1],
         lambda_j_minus_one * sigma[1] + lambda * product[0] * product[2]},
        {0.0, twice_mu + lambda * product[1] * product[1],
         lambda_j_minus_one * sigma[0] + lambda * product[1] * product[2]},
        {0.0, 0.0, twice_mu + lambda * product[2] * product[2]}};
    A[1][0] = A[0][1];
    A[2][0] = A[0][2];
    A[2][1] = A[1][2];
    const double m01 = twice_mu - lambda_j_minus_one * sigma[2];
    const double m02 = twice_mu - lambda_j_minus_one * sigma[1];
    const double m12 = twice_mu - lambda_j_minus_one * sigma[0];
    const double p01 = (psi[0] + psi[1]) /
        clampSmallMagnitude(sigma[0] + sigma[1], 1e-6);
    const double p02 = (psi[0] + psi[2]) /
        clampSmallMagnitude(sigma[0] + sigma[2], 1e-6);
    const double p12 = (psi[1] + psi[2]) /
        clampSmallMagnitude(sigma[1] + sigma[2], 1e-6);
    double B01[2][2] = {{0.5 * (m01 + p01), 0.5 * (m01 - p01)},
                         {0.5 * (m01 - p01), 0.5 * (m01 + p01)}};
    double B12[2][2] = {{0.5 * (m12 + p12), 0.5 * (m12 - p12)},
                         {0.5 * (m12 - p12), 0.5 * (m12 + p12)}};
    double B20[2][2] = {{0.5 * (m02 + p02), 0.5 * (m02 - p02)},
                         {0.5 * (m02 - p02), 0.5 * (m02 + p02)}};
    projectPSD3x3(A);
    projectPSD2x2(B01);
    projectPSD2x2(B12);
    projectPSD2x2(B20);
    double K[3][3] = {};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            K[row][row] += A[row][col] * D[col][col];
    K[0][1] = B01[0][0] * D[0][1] + B01[0][1] * D[1][0];
    K[1][0] = B01[1][0] * D[0][1] + B01[1][1] * D[1][0];
    K[1][2] = B12[0][0] * D[1][2] + B12[0][1] * D[2][1];
    K[2][1] = B12[1][0] * D[1][2] + B12[1][1] * D[2][1];
    K[0][2] = B20[0][0] * D[0][2] + B20[0][1] * D[2][0];
    K[2][0] = B20[1][0] * D[0][2] + B20[1][1] * D[2][0];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            dP[row][col] = 0.0;
            for (int q = 0; q < 3; ++q)
                for (int r = 0; r < 3; ++r)
                    dP[row][col] += U[row][q] * K[q][r] * V[col][r];
        }
}

__device__ void corotatedStateAndProjectedHessian(const double F[3][3],
    double mu, double lambda, double R[3][3], double P[3][3], double H[81],
    double& J)
{
    double U[3][3], sigma[3], V[3][3];
    oneSidedJacobiSvd3x3(F, U, sigma, V);
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            R[row][col] = 0.0;
            for (int q = 0; q < 3; ++q)
                R[row][col] += U[row][q] * V[col][q];
        }
    J = sigma[0] * sigma[1] * sigma[2];
    double cofactor[3][3];
    cofactor3x3(F, cofactor);
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            P[row][col] = 2.0 * mu * (F[row][col] - R[row][col]) +
                lambda * (J - 1.0) * cofactor[row][col];
    CorotatedProjectedBlocks blocks;
    buildCorotatedProjectedBlocks(sigma, mu, lambda, blocks);
    for (int basis = 0; basis < 9; ++basis) {
        double dF[3][3] = {};
        dF[basis / 3][basis % 3] = 1.0;
        double dP[3][3];
        applyCorotatedProjectedBlocks(U, V, blocks, dF, dP);
        for (int output = 0; output < 9; ++output)
            H[9 * output + basis] = dP[output / 3][output % 3];
    }
}

__global__ void pdTrialGatherKernel(const PdTrialParticle* particles,
    int particle_count, const PdParallelPinningTarget* parallel_targets,
    const GridKey* node_keys, const int* node_indices, int node_count,
    const double* node_velocity, double dx, double dt,
    double* particle_velocity, double* particle_grad_velocity, double* particle_F,
    double* particle_J, double* particle_position, double* particle_force_stress,
    double* particle_force, double* particle_tangent_a,
    double* particle_tangent_matrix, double* particle_energy)
{
    const int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= particle_count)
        return;
    const PdTrialParticle particle = particles[particle_id];
    int base[3];
    double w[3][3], dw[3][3];
    quadraticWeights(particle.x.x / dx, base[0], w[0], dw[0]);
    quadraticWeights(particle.x.y / dx, base[1], w[1], dw[1]);
    quadraticWeights(particle.x.z / dx, base[2], w[2], dw[2]);
    double velocity[3] = {};
    double grad_velocity[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const GridKey key = packGridKey(base[0] + i, base[1] + j,
                    base[2] + k);
                const int found = lowerBoundKey(node_keys, node_count, key);
                if (found == node_count || node_keys[found] != key)
                    continue;
                const int node = node_indices[found];
                const double weight = w[0][i] * w[1][j] * w[2][k];
                const double grad[3] = {
                    dw[0][i] * w[1][j] * w[2][k] / dx,
                    w[0][i] * dw[1][j] * w[2][k] / dx,
                    w[0][i] * w[1][j] * dw[2][k] / dx};
                for (int row = 0; row < 3; ++row) {
                    const double value = node_velocity[3 * node + row];
                    velocity[row] += value * weight;
                    for (int col = 0; col < 3; ++col)
                        grad_velocity[row][col] += value * grad[col];
                }
            }
    double F[3][3];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            F[row][col] = particle.F[3 * row + col] + dt * grad_velocity[row][col];
    double R[3][3];
    polarRotation3x3(F, R);
    double position[3] = {
        particle.x0.x + dt * velocity[0],
        particle.x0.y + dt * velocity[1],
        particle.x0.z + dt * velocity[2]};
    double force[3] = {
        -particle.pinning_a * position[0] + particle.pinning_b.x,
        -particle.pinning_a * position[1] + particle.pinning_b.y,
        -particle.pinning_a * position[2] + particle.pinning_b.z};
    double tangent_a = particle.pinning_a;
    double tangent_matrix[3][3] = {};
    for (int row = 0; row < 3; ++row)
        tangent_matrix[row][row] = particle.pinning_a;
    double energy = 0.0;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            const double difference = F[row][col] - R[row][col];
            energy += particle.volume * particle.mu * difference * difference;
        }
    if (particle.pinning_a > 0.0) {
        const double center[3] = {
            particle.pinning_b.x / particle.pinning_a,
            particle.pinning_b.y / particle.pinning_a,
            particle.pinning_b.z / particle.pinning_a};
        for (int row = 0; row < 3; ++row) {
            const double difference = position[row] - center[row];
            energy += 0.5 * particle.pinning_a * difference * difference;
        }
    }
    for (int target_id = particle.parallel_target_begin;
         target_id < particle.parallel_target_begin + particle.parallel_target_count;
         ++target_id) {
        const PdParallelPinningTarget target = parallel_targets[target_id];
        const double n_raw[3] = {target.x.x - particle.x0.x,
                                 target.x.y - particle.x0.y,
                                 target.x.z - particle.x0.z};
        const double n_norm = sqrt(n_raw[0] * n_raw[0] + n_raw[1] * n_raw[1] +
            n_raw[2] * n_raw[2]);
        if (n_norm < 1e-12 || target.k == 0)
            continue;
        const double n[3] = {n_raw[0] / n_norm, n_raw[1] / n_norm,
                             n_raw[2] / n_norm};
        const double signed_distance = (position[0] - target.x.x) * n[0] +
            (position[1] - target.x.y) * n[1] +
            (position[2] - target.x.z) * n[2];
        if (signed_distance < 0) {
            const double scalar = -target.k * signed_distance;
            energy += 0.5 * target.k * signed_distance * signed_distance;
            for (int row = 0; row < 3; ++row)
                force[row] += scalar * n[row];
            tangent_a += target.k;
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    tangent_matrix[row][col] += target.k * n[row] * n[col];
        }
    }
    for (int row = 0; row < 3; ++row) {
        particle_velocity[3 * particle_id + row] = velocity[row];
        particle_position[3 * particle_id + row] = position[row];
        particle_force[3 * particle_id + row] = force[row];
        for (int col = 0; col < 3; ++col) {
            particle_grad_velocity[9 * particle_id + 3 * row + col] =
                grad_velocity[row][col];
            particle_F[9 * particle_id + 3 * row + col] = F[row][col];
            particle_force_stress[9 * particle_id + 3 * row + col] =
                2.0 * particle.mu * particle.volume * (F[row][col] - R[row][col]);
            particle_tangent_matrix[9 * particle_id + 3 * row + col] =
                tangent_matrix[row][col];
        }
    }
    particle_J[particle_id] = determinant3x3(F);
    particle_tangent_a[particle_id] = tangent_a;
    particle_energy[particle_id] = energy;
}

__global__ void patchPdTrialParticlesFromOwnerKernel(
    PdTrialParticle* particles, int particle_count,
    GPU_STAGE3A::SolidParticleStateDeviceView owner)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    particles[p].x = owner.rest_x[p];
    particles[p].x0 = owner.current_x[p];
    particles[p].volume = owner.volume[p];
    particles[p].mass = owner.mass[p];
    for (int entry = 0; entry < 9; ++entry)
        particles[p].F[entry] = owner.f[p].values[entry];
}

__global__ void patchCorotatedTrialParticlesFromOwnerKernel(
    CorotatedTrialParticle* particles, int particle_count,
    GPU_STAGE3A::SolidParticleStateDeviceView owner)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    particles[p].x = owner.rest_x[p];
    particles[p].x0 = owner.current_x[p];
    particles[p].volume = owner.volume[p];
    particles[p].mass = owner.mass[p];
    for (int entry = 0; entry < 9; ++entry)
        particles[p].F[entry] = owner.f[p].values[entry];
}

__global__ void patchPdTrialPinningKernel(PdTrialParticle* particles,
    const DeviceState::TrialPinningPatch* patches, int particle_count,
    double dt)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    if (patches[p].pinning_from_resident_x0) {
        const double damping_over_dt = patches[p].pinning_d / dt;
        const GPU_STAGE3A::Vec3d target_next = {
            patches[p].pinning_target_x.x +
                dt * patches[p].pinning_target_v.x,
            patches[p].pinning_target_x.y +
                dt * patches[p].pinning_target_v.y,
            patches[p].pinning_target_x.z +
                dt * patches[p].pinning_target_v.z};
        particles[p].pinning_a = patches[p].pinning_k + damping_over_dt;
        particles[p].pinning_b = {
            patches[p].pinning_k * target_next.x + damping_over_dt *
                (particles[p].x0.x + target_next.x -
                    patches[p].pinning_target_x.x),
            patches[p].pinning_k * target_next.y + damping_over_dt *
                (particles[p].x0.y + target_next.y -
                    patches[p].pinning_target_x.y),
            patches[p].pinning_k * target_next.z + damping_over_dt *
                (particles[p].x0.z + target_next.z -
                    patches[p].pinning_target_x.z)};
    } else {
        particles[p].pinning_a = patches[p].pinning_a;
        particles[p].pinning_b = patches[p].pinning_b;
    }
    particles[p].parallel_target_begin = patches[p].parallel_target_begin;
    particles[p].parallel_target_count = patches[p].parallel_target_count;
}

__global__ void patchCorotatedTrialPinningKernel(
    CorotatedTrialParticle* particles,
    const DeviceState::TrialPinningPatch* patches, int particle_count,
    double dt)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    if (patches[p].pinning_from_resident_x0) {
        const double damping_over_dt = patches[p].pinning_d / dt;
        const GPU_STAGE3A::Vec3d target_next = {
            patches[p].pinning_target_x.x +
                dt * patches[p].pinning_target_v.x,
            patches[p].pinning_target_x.y +
                dt * patches[p].pinning_target_v.y,
            patches[p].pinning_target_x.z +
                dt * patches[p].pinning_target_v.z};
        particles[p].pinning_a = patches[p].pinning_k + damping_over_dt;
        particles[p].pinning_b = {
            patches[p].pinning_k * target_next.x + damping_over_dt *
                (particles[p].x0.x + target_next.x -
                    patches[p].pinning_target_x.x),
            patches[p].pinning_k * target_next.y + damping_over_dt *
                (particles[p].x0.y + target_next.y -
                    patches[p].pinning_target_x.y),
            patches[p].pinning_k * target_next.z + damping_over_dt *
                (particles[p].x0.z + target_next.z -
                    patches[p].pinning_target_x.z)};
    } else {
        particles[p].pinning_a = patches[p].pinning_a;
        particles[p].pinning_b = patches[p].pinning_b;
    }
    particles[p].parallel_target_begin = patches[p].parallel_target_begin;
    particles[p].parallel_target_count = patches[p].parallel_target_count;
}

__global__ void contactKeyKernel(const double* particle_position,
    int particle_count, double cell_size, GridKey* keys, int* indices)
{
    const int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= particle_count)
        return;
    const int x = static_cast<int>(floor(particle_position[3 * particle_id + 0] / cell_size));
    const int y = static_cast<int>(floor(particle_position[3 * particle_id + 1] / cell_size));
    const int z = static_cast<int>(floor(particle_position[3 * particle_id + 2] / cell_size));
    keys[particle_id] = packGridKey(x, y, z);
    indices[particle_id] = particle_id;
}

__global__ void corotatedTrialGatherKernel(const CorotatedTrialParticle* particles,
    int particle_count, const PdParallelPinningTarget* parallel_targets,
    const GridKey* node_keys, const int* node_indices, int node_count,
    const double* node_velocity, double dx, double dt,
    GhostBoxShell3D ghost_collision, double ghost_penalty,
    double* particle_velocity, double* particle_grad_velocity, double* particle_F,
    double* particle_J, double* particle_position, double* particle_force_stress,
    double* particle_force, double* particle_tangent_a, double* particle_tangent_matrix,
    double* particle_projected_hessian, double* particle_energy)
{
    const int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= particle_count)
        return;
    const CorotatedTrialParticle particle = particles[particle_id];
    int base[3];
    double w[3][3], dw[3][3];
    quadraticWeights(particle.x.x / dx, base[0], w[0], dw[0]);
    quadraticWeights(particle.x.y / dx, base[1], w[1], dw[1]);
    quadraticWeights(particle.x.z / dx, base[2], w[2], dw[2]);
    double velocity[3] = {};
    double grad_velocity[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const GridKey key = packGridKey(base[0] + i, base[1] + j,
                    base[2] + k);
                const int found = lowerBoundKey(node_keys, node_count, key);
                if (found == node_count || node_keys[found] != key)
                    continue;
                const int node = node_indices[found];
                const double weight = w[0][i] * w[1][j] * w[2][k];
                const double grad[3] = {
                    dw[0][i] * w[1][j] * w[2][k] / dx,
                    w[0][i] * dw[1][j] * w[2][k] / dx,
                    w[0][i] * w[1][j] * dw[2][k] / dx};
                for (int row = 0; row < 3; ++row) {
                    const double value = node_velocity[3 * node + row];
                    velocity[row] += value * weight;
                    for (int col = 0; col < 3; ++col)
                        grad_velocity[row][col] += value * grad[col];
                }
            }
    double F[3][3];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            F[row][col] = particle.F[3 * row + col] + dt * grad_velocity[row][col];
    double R[3][3], P[3][3], H[81], J = 1.0;
    corotatedStateAndProjectedHessian(F, particle.mu, particle.lambda, R, P, H, J);
    double position[3] = {
        particle.x0.x + dt * velocity[0],
        particle.x0.y + dt * velocity[1],
        particle.x0.z + dt * velocity[2]};
    double force[3] = {
        -particle.pinning_a * position[0] + particle.pinning_b.x,
        -particle.pinning_a * position[1] + particle.pinning_b.y,
        -particle.pinning_a * position[2] + particle.pinning_b.z};
    double tangent_a = particle.pinning_a;
    double tangent_matrix[3][3] = {};
    for (int row = 0; row < 3; ++row)
        tangent_matrix[row][row] = particle.pinning_a;
    double energy = 0.5 * particle.volume * particle.lambda
        * (J - 1.0) * (J - 1.0);
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            const double difference = F[row][col] - R[row][col];
            energy += particle.volume * particle.mu * difference * difference;
        }
    if (particle.pinning_a > 0.0) {
        const double center[3] = {
            particle.pinning_b.x / particle.pinning_a,
            particle.pinning_b.y / particle.pinning_a,
            particle.pinning_b.z / particle.pinning_a};
        for (int row = 0; row < 3; ++row) {
            const double difference = position[row] - center[row];
            energy += 0.5 * particle.pinning_a * difference * difference;
        }
    }
    for (int target_id = particle.parallel_target_begin;
         target_id < particle.parallel_target_begin + particle.parallel_target_count;
         ++target_id) {
        const PdParallelPinningTarget target = parallel_targets[target_id];
        const double n_raw[3] = {target.x.x - particle.x0.x,
                                 target.x.y - particle.x0.y,
                                 target.x.z - particle.x0.z};
        const double n_norm = sqrt(n_raw[0] * n_raw[0] + n_raw[1] * n_raw[1] +
            n_raw[2] * n_raw[2]);
        if (n_norm < 1e-12 || target.k == 0)
            continue;
        const double n[3] = {n_raw[0] / n_norm, n_raw[1] / n_norm,
                             n_raw[2] / n_norm};
        const double signed_distance = (position[0] - target.x.x) * n[0] +
            (position[1] - target.x.y) * n[1] +
            (position[2] - target.x.z) * n[2];
        if (signed_distance < 0) {
            const double scalar = -target.k * signed_distance;
            energy += 0.5 * target.k * signed_distance * signed_distance;
            for (int row = 0; row < 3; ++row)
                force[row] += scalar * n[row];
            tangent_a += target.k;
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    tangent_matrix[row][col] += target.k * n[row] * n[col];
        }
    }
    if (ghost_collision.valid && ghost_penalty != 0.0) {
        GPU_STAGE3A::Vec3d normal;
        const double phi = ghostBoxShellSignedDistanceAndNormal(
            particle.x0, ghost_collision, normal);
        // AnalyticCollisionObject::queryInside is inclusive. At phi==0 the
        // legacy target equals xn and is ignored by its 1e-12 normal guard.
        if (phi < 0.0) {
            const double target[3] = {
                particle.x0.x - phi * normal.x,
                particle.x0.y - phi * normal.y,
                particle.x0.z - phi * normal.z};
            const double n[3] = {normal.x, normal.y, normal.z};
            const double signed_distance =
                (position[0] - target[0]) * n[0] +
                (position[1] - target[1]) * n[1] +
                (position[2] - target[2]) * n[2];
            if (signed_distance < 0.0) {
                const double scalar = -ghost_penalty * signed_distance;
                energy += 0.5 * ghost_penalty * signed_distance * signed_distance;
                tangent_a += ghost_penalty;
                for (int row = 0; row < 3; ++row) {
                    force[row] += scalar * n[row];
                    for (int col = 0; col < 3; ++col)
                        tangent_matrix[row][col] +=
                            ghost_penalty * n[row] * n[col];
                }
            }
        }
    }
    for (int row = 0; row < 3; ++row) {
        particle_velocity[3 * particle_id + row] = velocity[row];
        particle_position[3 * particle_id + row] = position[row];
        particle_force[3 * particle_id + row] = force[row];
        for (int col = 0; col < 3; ++col) {
            particle_grad_velocity[9 * particle_id + 3 * row + col] =
                grad_velocity[row][col];
            particle_F[9 * particle_id + 3 * row + col] = F[row][col];
            particle_force_stress[9 * particle_id + 3 * row + col] =
                particle.volume * P[row][col];
        }
    }
    particle_J[particle_id] = J;
    particle_tangent_a[particle_id] = tangent_a;
    particle_energy[particle_id] = energy;
    for (int entry = 0; entry < 9; ++entry) {
        particle_tangent_matrix[9 * particle_id + entry] =
            tangent_matrix[entry / 3][entry % 3];
        for (int column = 0; column < 9; ++column)
            particle_projected_hessian[81 * particle_id + 9 * entry + column] =
                H[9 * entry + column];
    }
}

template <class TrialParticle>
__global__ void tlContactKernel(const TrialParticle* particles, int particle_count,
    const double* particle_position, const GridKey* sorted_keys,
    const int* sorted_indices, double cutoff, double dt, double* particle_force,
    double* contact_a, double* tangent_a, double* tangent_matrix)
{
    const int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= particle_count)
        return;
    const TrialParticle particle = particles[particle_id];
    const double x[3] = {particle_position[3 * particle_id + 0],
                         particle_position[3 * particle_id + 1],
                         particle_position[3 * particle_id + 2]};
    const int cell[3] = {static_cast<int>(floor(x[0] / cutoff)),
                         static_cast<int>(floor(x[1] / cutoff)),
                         static_cast<int>(floor(x[2] / cutoff))};
    double force[3] = {};
    double tangent = 0;
    const double reference_cutoff_sq = 2.25 * cutoff * cutoff;
    const double cutoff_sq = cutoff * cutoff;
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j)
            for (int k = -1; k <= 1; ++k) {
                const GridKey key = packGridKey(cell[0] + i, cell[1] + j, cell[2] + k);
                int begin = lowerBoundKey(sorted_keys, particle_count, key);
                for (int entry = begin; entry < particle_count && sorted_keys[entry] == key;
                     ++entry) {
                    const int q = sorted_indices[entry];
                    if (q == particle_id)
                        continue;
                    const TrialParticle other = particles[q];
                    const double reference_dx = particle.x.x - other.x.x;
                    const double reference_dy = particle.x.y - other.x.y;
                    const double reference_dz = particle.x.z - other.x.z;
                    if (reference_dx * reference_dx + reference_dy * reference_dy +
                        reference_dz * reference_dz <= reference_cutoff_sq)
                        continue;
                    const double dir[3] = {x[0] - particle_position[3 * q + 0],
                                           x[1] - particle_position[3 * q + 1],
                                           x[2] - particle_position[3 * q + 2]};
                    const double dist_sq = dir[0] * dir[0] + dir[1] * dir[1] +
                        dir[2] * dir[2];
                    if (!(dist_sq < cutoff_sq) || dist_sq < 1e-24)
                        continue;
                    const double dist = sqrt(dist_sq);
                    const double stiffness = particle.mass * other.mass /
                        (particle.mass + other.mass) / (dt * dt);
                    const double scale = -stiffness * (1.0 - cutoff / dist);
                    for (int row = 0; row < 3; ++row)
                        force[row] += scale * dir[row];
                    tangent += stiffness;
                }
            }
    for (int row = 0; row < 3; ++row)
        particle_force[3 * particle_id + row] += force[row];
    contact_a[particle_id] = tangent;
    tangent_a[particle_id] += tangent;
    for (int row = 0; row < 3; ++row)
        tangent_matrix[9 * particle_id + 3 * row + row] += tangent;
}

__global__ void massApplyKernel(const double* mass, const double* x,
    int nodes, double* y)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodes)
        return;
    for (int d = 0; d < 3; ++d)
        y[3 * node + d] = mass[node] * x[3 * node + d];
}

__global__ void pdApplyKernel(const PdParticle* particles, int particle_count,
    const GridKey* node_keys, const int* node_indices, int node_count,
    const double* x, double dx, double dt, double* y)
{
    const int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= particle_count)
        return;
    const PdParticle particle = particles[particle_id];
    int base[3];
    double w[3][3], dw[3][3];
    quadraticWeights(particle.x.x / dx, base[0], w[0], dw[0]);
    quadraticWeights(particle.x.y / dx, base[1], w[1], dw[1]);
    quadraticWeights(particle.x.z / dx, base[2], w[2], dw[2]);
    double grad_x[3][3] = {};
    double particle_x[3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const GridKey key = packGridKey(base[0] + i, base[1] + j,
                    base[2] + k);
                const int found = lowerBoundKey(node_keys, node_count, key);
                if (found == node_count || node_keys[found] != key)
                    continue;
                const int node = node_indices[found];
                const double grad[3] = {
                    dw[0][i] * w[1][j] * w[2][k] / dx,
                    w[0][i] * dw[1][j] * w[2][k] / dx,
                    w[0][i] * w[1][j] * dw[2][k] / dx};
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 3; ++col)
                        grad_x[row][col] += x[3 * node + row] * grad[col];
                for (int row = 0; row < 3; ++row)
                    particle_x[row] += x[3 * node + row] * w[0][i] * w[1][j] * w[2][k];
            }
    const double stress_scale = dt * dt * 2.0 * particle.mu * particle.volume;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const GridKey key = packGridKey(base[0] + i, base[1] + j,
                    base[2] + k);
                const int found = lowerBoundKey(node_keys, node_count, key);
                if (found == node_count || node_keys[found] != key)
                    continue;
                const int node = node_indices[found];
                const double grad[3] = {
                    dw[0][i] * w[1][j] * w[2][k] / dx,
                    w[0][i] * dw[1][j] * w[2][k] / dx,
                    w[0][i] * w[1][j] * dw[2][k] / dx};
                for (int row = 0; row < 3; ++row) {
                    double value = 0;
                    for (int col = 0; col < 3; ++col)
                        value += stress_scale * grad_x[row][col] * grad[col];
                    double pinning_value = 0;
                    for (int col = 0; col < 3; ++col) {
                        double tangent = particle.pinning_matrix[3 * row + col];
                        if (row == col)
                            tangent += particle.pinning_a;
                        pinning_value += tangent * particle_x[col];
                    }
                    value += dt * dt * pinning_value * w[0][i] * w[1][j] * w[2][k];
                    atomicAdd(&y[3 * node + row], value);
                }
            }
}

// Scatter per-particle force_stress/fp into node-indexed residual force,
// reproducing MpmForceAssembly::rasterizeForceToTVStack<false>(scale=dt): each node
// gets -dt * (stress * grad_w - fp * w), grad_w already carrying 1/dx.
template <class TrialParticle>
__global__ void residualScatterKernel(const TrialParticle* particles,
    int particle_count, const GridKey* node_keys, const int* node_indices,
    int node_count, const double* force_stress, const double* force, double dx,
    double dt, double* node_residual)
{
    const int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= particle_count)
        return;
    const TrialParticle particle = particles[particle_id];
    int base[3];
    double w[3][3], dw[3][3];
    quadraticWeights(particle.x.x / dx, base[0], w[0], dw[0]);
    quadraticWeights(particle.x.y / dx, base[1], w[1], dw[1]);
    quadraticWeights(particle.x.z / dx, base[2], w[2], dw[2]);
    const double* stress = force_stress + 9 * particle_id;
    const double fp[3] = {force[3 * particle_id + 0], force[3 * particle_id + 1],
        force[3 * particle_id + 2]};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const GridKey key = packGridKey(base[0] + i, base[1] + j, base[2] + k);
                const int found = lowerBoundKey(node_keys, node_count, key);
                if (found == node_count || node_keys[found] != key)
                    continue;
                const int node = node_indices[found];
                const double weight = w[0][i] * w[1][j] * w[2][k];
                const double grad[3] = {
                    dw[0][i] * w[1][j] * w[2][k] / dx,
                    w[0][i] * dw[1][j] * w[2][k] / dx,
                    w[0][i] * w[1][j] * dw[2][k] / dx};
                for (int row = 0; row < 3; ++row) {
                    const double delta = stress[3 * row + 0] * grad[0] +
                        stress[3 * row + 1] * grad[1] +
                        stress[3 * row + 2] * grad[2] - fp[row] * weight;
                    atomicAdd(&node_residual[3 * node + row], -dt * delta);
                }
            }
}

__global__ void linearCorotatedApplyKernel(
    const LinearCorotatedParticle* particles, int particle_count,
    const GridKey* node_keys, const int* node_indices, int node_count,
    const double* x, double dx, double dt, double* y)
{
    const int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= particle_count)
        return;
    const LinearCorotatedParticle particle = particles[particle_id];
    int base[3];
    double w[3][3], dw[3][3];
    quadraticWeights(particle.x.x / dx, base[0], w[0], dw[0]);
    quadraticWeights(particle.x.y / dx, base[1], w[1], dw[1]);
    quadraticWeights(particle.x.z / dx, base[2], w[2], dw[2]);
    double dF[3][3] = {};
    double particle_x[3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const int found = lowerBoundKey(node_keys, node_count,
                    packGridKey(base[0] + i, base[1] + j, base[2] + k));
                if (found == node_count || node_keys[found] !=
                    packGridKey(base[0] + i, base[1] + j, base[2] + k))
                    continue;
                const int node = node_indices[found];
                const double grad[3] = {
                    dw[0][i] * w[1][j] * w[2][k] / dx,
                    w[0][i] * dw[1][j] * w[2][k] / dx,
                    w[0][i] * w[1][j] * dw[2][k] / dx};
                const double weight = w[0][i] * w[1][j] * w[2][k];
                for (int row = 0; row < 3; ++row) {
                    const double value = x[3 * node + row];
                    particle_x[row] += value * weight;
                    for (int col = 0; col < 3; ++col)
                        dF[row][col] += value * grad[col];
                }
            }
    double RtdF[3][3] = {};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            for (int q = 0; q < 3; ++q)
                RtdF[row][col] += particle.R[3 * q + row] * dF[q][col];
    double dP[3][3] = {};
    double contraction = 0;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            contraction += particle.R[3 * row + col] * dF[row][col];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            double rotated_transpose = 0;
            for (int q = 0; q < 3; ++q)
                rotated_transpose += particle.R[3 * row + q] * RtdF[col][q];
            dP[row][col] = particle.mu * dF[row][col] +
                particle.mu * rotated_transpose +
                particle.lambda * contraction * particle.R[3 * row + col];
        }
    const double material_scale = dt * dt * particle.volume;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const int found = lowerBoundKey(node_keys, node_count,
                    packGridKey(base[0] + i, base[1] + j, base[2] + k));
                if (found == node_count || node_keys[found] !=
                    packGridKey(base[0] + i, base[1] + j, base[2] + k))
                    continue;
                const int node = node_indices[found];
                const double grad[3] = {
                    dw[0][i] * w[1][j] * w[2][k] / dx,
                    w[0][i] * dw[1][j] * w[2][k] / dx,
                    w[0][i] * w[1][j] * dw[2][k] / dx};
                const double weight = w[0][i] * w[1][j] * w[2][k];
                for (int row = 0; row < 3; ++row) {
                    double value = 0;
                    for (int col = 0; col < 3; ++col)
                        value += material_scale * dP[row][col] * grad[col];
                    value += dt * dt * particle.pinning_a * particle_x[row] * weight;
                    atomicAdd(&y[3 * node + row], value);
                }
            }
}

__global__ void corotatedApplyKernel(const CorotatedParticle* particles,
    int particle_count, const GridKey* node_keys, const int* node_indices,
    int node_count, const double* x, double dx, double dt, double* y)
{
    const int particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= particle_count)
        return;
    const CorotatedParticle particle = particles[particle_id];
    int base[3];
    double w[3][3], dw[3][3];
    quadraticWeights(particle.x.x / dx, base[0], w[0], dw[0]);
    quadraticWeights(particle.x.y / dx, base[1], w[1], dw[1]);
    quadraticWeights(particle.x.z / dx, base[2], w[2], dw[2]);
    double dF[3][3] = {};
    double particle_x[3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const GridKey key = packGridKey(base[0] + i, base[1] + j,
                    base[2] + k);
                const int found = lowerBoundKey(node_keys, node_count, key);
                if (found == node_count || node_keys[found] != key)
                    continue;
                const int node = node_indices[found];
                const double grad[3] = {
                    dw[0][i] * w[1][j] * w[2][k] / dx,
                    w[0][i] * dw[1][j] * w[2][k] / dx,
                    w[0][i] * w[1][j] * dw[2][k] / dx};
                const double weight = w[0][i] * w[1][j] * w[2][k];
                for (int row = 0; row < 3; ++row) {
                    const double value = x[3 * node + row];
                    particle_x[row] += value * weight;
                    for (int col = 0; col < 3; ++col)
                        dF[row][col] += value * grad[col];
                }
            }
    double dP[3][3] = {};
    for (int output = 0; output < 9; ++output)
        for (int input = 0; input < 9; ++input)
            dP[output / 3][output % 3] +=
                particle.projected_hessian[9 * output + input] *
                dF[input / 3][input % 3];
    const double material_scale = dt * dt * particle.volume;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const GridKey key = packGridKey(base[0] + i, base[1] + j,
                    base[2] + k);
                const int found = lowerBoundKey(node_keys, node_count, key);
                if (found == node_count || node_keys[found] != key)
                    continue;
                const int node = node_indices[found];
                const double grad[3] = {
                    dw[0][i] * w[1][j] * w[2][k] / dx,
                    w[0][i] * dw[1][j] * w[2][k] / dx,
                    w[0][i] * w[1][j] * dw[2][k] / dx};
                const double weight = w[0][i] * w[1][j] * w[2][k];
                for (int row = 0; row < 3; ++row) {
                    double value = 0.0;
                    for (int col = 0; col < 3; ++col)
                        value += material_scale * dP[row][col] * grad[col];
                    double pinning = 0.0;
                    for (int col = 0; col < 3; ++col) {
                        double tangent = particle.pinning_matrix[3 * row + col];
                        if (row == col)
                            tangent += particle.pinning_a;
                        pinning += tangent * particle_x[col];
                    }
                    atomicAdd(&y[3 * node + row],
                        value + dt * dt * pinning * weight);
                }
            }
}

__global__ void jacobiKernel(const double* mass, const double* r, int entries,
    double* z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries)
        z[i] = r[i] / mass[i / 3];
}

__global__ void projectKernel(const double* projection, double* v, int nodes)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodes)
        return;
    const double x0 = v[3 * node + 0];
    const double x1 = v[3 * node + 1];
    const double x2 = v[3 * node + 2];
    for (int row = 0; row < 3; ++row)
        v[3 * node + row] = projection[9 * node + 3 * row + 0] * x0 +
            projection[9 * node + 3 * row + 1] * x1 +
            projection[9 * node + 3 * row + 2] * x2;
}

__global__ void axpyKernel(double* x, const double* p, double alpha, int entries)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries)
        x[i] += alpha * p[i];
}

__global__ void residualKernel(double* r, const double* q, double alpha, int entries)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries)
        r[i] -= alpha * q[i];
}

__global__ void assembleNewtonResidualKernel(const double* material_residual,
    const double* mass, const double* velocity_increment, double dt,
    double gravity_x, double gravity_y, double gravity_z, int nodes,
    double* residual)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodes)
        return;
    const double gravity[3] = {gravity_x, gravity_y, gravity_z};
    for (int d = 0; d < 3; ++d) {
        double value = dt * gravity[d] * mass[node];
        value += material_residual[3 * node + d];
        value -= mass[node] * velocity_increment[3 * node + d];
        residual[3 * node + d] = value;
    }
}

__global__ void commitNewtonIncrementKernel(double* velocity_increment,
    const double* base_velocity, const double* step, double* node_velocity,
    int entries)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= entries)
        return;
    velocity_increment[i] += step[i];
    node_velocity[i] = base_velocity[i] + velocity_increment[i];
}

__global__ void setNewtonLineSearchTrialKernel(const double* base_velocity,
    const double* velocity_increment, const double* step, double alpha,
    double* node_velocity, int entries)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries)
        node_velocity[i] = base_velocity[i] + velocity_increment[i] +
            alpha * step[i];
}

__global__ void commitScaledNewtonIncrementKernel(double* velocity_increment,
    const double* step, double alpha, int entries)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries)
        velocity_increment[i] += alpha * step[i];
}

struct NewtonGridEnergyTransform {
    const double* mass;
    const double* velocity_increment;
    const double* step;
    double alpha;
    double dt;
    double gravity[3];

    __host__ __device__ double operator()(int entry) const
    {
        const int node = entry / 3;
        const int component = entry - 3 * node;
        const double dv = velocity_increment[entry] +
            (step ? alpha * step[entry] : 0.0);
        return 0.5 * mass[node] * dv * dv -
            dt * mass[node] * gravity[component] * dv;
    }
};

__global__ void exportFinalGridVelocityKernel(const GridKey* sorted_keys,
    const int* sorted_indices, const double* node_velocity,
    GPU_STAGE3A::AssemblyGridNode* nodes, int node_count)
{
    const int sorted = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted >= node_count)
        return;
    // Both arrays are canonical key-sorted views built from the same TL rest
    // positions. Keep an invalid row instead of silently pairing different
    // physical nodes if that invariant is ever broken by an unsupported path.
    if (nodes[sorted].key != sorted_keys[sorted]) {
        nodes[sorted].idx = -1;
        return;
    }
    const int idx = sorted_indices[sorted];
    nodes[sorted].idx = idx;
    nodes[sorted].new_v = {node_velocity[3 * idx + 0],
        node_velocity[3 * idx + 1], node_velocity[3 * idx + 2]};
}

__global__ void initializeTrialNodesFromResidentMapKernel(
    const GPU_STAGE3A::AssemblyGridNode* nodes, int node_count,
    double increment_x, double increment_y, double increment_z,
    GridKey* sorted_keys, int* sorted_indices, double* mass,
    double* projection, double* base_velocity,
    double* velocity_increment, double* node_velocity)
{
    const int sorted = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted >= node_count)
        return;
    const GPU_STAGE3A::AssemblyGridNode node = nodes[sorted];
    const int idx = node.idx;
    sorted_keys[sorted] = node.key;
    sorted_indices[sorted] = idx;
    if (idx < 0 || idx >= node_count)
        return;
    mass[idx] = node.mass;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            projection[9 * idx + 3 * r + c] = r == c ? 1.0 : 0.0;
    const double base[3] = {node.v.x, node.v.y, node.v.z};
    const double increment[3] = {increment_x, increment_y, increment_z};
    for (int d = 0; d < 3; ++d) {
        base_velocity[3 * idx + d] = base[d];
        velocity_increment[3 * idx + d] = increment[d];
        node_velocity[3 * idx + d] = base[d] + increment[d];
    }
}

__global__ void directionKernel(double* p, const double* z, double beta, int entries)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries)
        p[i] = z[i] + beta * p[i];
}

void applyDevice(DeviceState& state, const PdOperatorInput& input,
    const double* x, double* y)
{
    constexpr int block_size = 128;
    // Size from the device-resident buffers: on reuse_static iterations the host
    // input.particles is empty because the particles already live on the device.
    const int node_count = static_cast<int>(state.mass.size());
    const int particle_count = static_cast<int>(state.particles.size());
    const int node_blocks = (node_count + block_size - 1) / block_size;
    if (node_blocks)
        massApplyKernel<<<node_blocks, block_size>>>(
            thrust::raw_pointer_cast(state.mass.data()), x, node_count, y);
    const int particle_blocks = (particle_count + block_size - 1) / block_size;
    if (particle_blocks)
        pdApplyKernel<<<particle_blocks, block_size>>>(
            thrust::raw_pointer_cast(state.particles.data()), particle_count,
            thrust::raw_pointer_cast(state.node_keys.data()),
            thrust::raw_pointer_cast(state.node_indices.data()),
            node_count, x, input.dx, input.dt, y);
    checkCuda(cudaGetLastError(), "solid PD operator kernel launch");
}

__global__ void initializePdOperatorParticlesFromTrialKernel(
    const PdTrialParticle* trial_particles, PdParticle* particles,
    int particle_count)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    particles[p].x = trial_particles[p].x;
    particles[p].volume = trial_particles[p].volume;
    particles[p].mu = trial_particles[p].mu;
    particles[p].pinning_a = 0.0;
    for (int entry = 0; entry < 9; ++entry)
        particles[p].pinning_matrix[entry] = 0.0;
}

void uploadInput(DeviceState& state, const PdOperatorInput& input)
{
    // Within one implicit step the grid topology, mass, projection and the
    // per-particle static fields (x/volume/mu) are constant; only the tangent
    // changes each Newton iteration, and that is injected from the device
    // trial-state buffers separately. Skip the redundant host->device uploads.
    if (input.reuse_static)
        return;
    if (!input.reuse_topology) {
        std::vector<std::pair<GridKey, int>> keys;
        std::vector<double> mass(input.nodes.size());
        std::vector<double> projection(input.nodes.size() * 9);
        keys.reserve(input.nodes.size());
        for (const SolidNode& node : input.nodes) {
            if (node.idx < 0 ||
                static_cast<std::size_t>(node.idx) >= input.nodes.size())
                throw std::runtime_error(
                    "solid PD operator node index out of range");
            keys.emplace_back(node.key, node.idx);
            mass[static_cast<std::size_t>(node.idx)] = node.mass;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    projection[static_cast<std::size_t>(
                        9 * node.idx + 3 * r + c)] =
                        node.projection[3 * r + c];
        }
        std::sort(keys.begin(), keys.end());
        for (std::size_t i = 1; i < keys.size(); ++i)
            if (keys[i - 1].first == keys[i].first)
                throw std::runtime_error(
                    "solid PD operator duplicate node key");
        std::vector<GridKey> sorted_keys(keys.size());
        std::vector<int> sorted_indices(keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i) {
            sorted_keys[i] = keys[i].first;
            sorted_indices[i] = keys[i].second;
        }
        state.node_keys.assign(sorted_keys.begin(), sorted_keys.end());
        state.node_indices.assign(sorted_indices.begin(), sorted_indices.end());
        state.mass.assign(mass.begin(), mass.end());
        state.projection.assign(projection.begin(), projection.end());
    } else if (state.node_keys.empty() ||
        state.node_indices.size() != state.node_keys.size() ||
        state.mass.size() != state.node_keys.size() ||
        state.projection.size() != 9 * state.node_keys.size()) {
        throw std::runtime_error(
            "solid PD resident operator topology is incomplete");
    }
    if (input.use_device_trial_particles) {
        const std::size_t particle_count = state.trial_particles.size();
        if (particle_count == 0)
            throw std::runtime_error(
                "solid PD resident operator particles are empty");
        state.particles.resize(particle_count);
        constexpr int block_size = 128;
        const int blocks = static_cast<int>((particle_count + block_size - 1) /
            block_size);
        initializePdOperatorParticlesFromTrialKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(state.trial_particles.data()),
            thrust::raw_pointer_cast(state.particles.data()),
            static_cast<int>(particle_count));
        checkCuda(cudaGetLastError(),
            "solid PD resident operator particle initialize");
    } else {
        state.particles.assign(input.particles.begin(), input.particles.end());
    }
}

// Copy the device-resident trial-state tangent (produced by the most recent
// evaluatePdTrialState3D call this Newton iteration) into the operator's
// per-particle pinning fields, avoiding a device->host->device round trip.
// Matches the host packing in the PD PCG caller: when the trial-state tangent
// matrix is present (always true on this device path) the full tangent lives in
// pinning_matrix and pinning_a is zeroed, so the operator's
// (pinning_matrix + pinning_a*I) does not double-count the diagonal.
__global__ void injectDeviceTangentKernel(PdParticle* particles,
    int particle_count, const double* tangent_matrix)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    particles[p].pinning_a = 0.0;
    for (int e = 0; e < 9; ++e)
        particles[p].pinning_matrix[e] = tangent_matrix[9 * p + e];
}

__global__ void injectCorotatedTrialKernel(CorotatedParticle* particles,
    const CorotatedTrialParticle* trial_particles, int particle_count,
    const double* projected_hessian, const double* tangent_matrix)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    particles[p].x = trial_particles[p].x;
    particles[p].volume = trial_particles[p].volume;
    particles[p].pinning_a = 0.0;
    for (int entry = 0; entry < 9; ++entry)
        particles[p].pinning_matrix[entry] = tangent_matrix[9 * p + entry];
    for (int entry = 0; entry < 81; ++entry)
        particles[p].projected_hessian[entry] =
            projected_hessian[81 * p + entry];
}

void projectDevice(DeviceState& state, int nodes, double* vector)
{
    constexpr int block_size = 128;
    const int blocks = (nodes + block_size - 1) / block_size;
    if (blocks)
        projectKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(state.projection.data()), vector, nodes);
}

void uploadLinearInput(DeviceState& state, const LinearCorotatedOperatorInput& input)
{
    PdOperatorInput topology;
    topology.nodes = input.nodes;
    uploadInput(state, topology);
    state.linear_particles.assign(input.particles.begin(), input.particles.end());
}

void uploadCorotatedInput(DeviceState& state, const CorotatedOperatorInput& input)
{
    PdOperatorInput topology;
    topology.reuse_static = input.reuse_static;
    topology.reuse_topology = input.reuse_topology;
    if (!input.reuse_static && !input.reuse_topology)
        topology.nodes = input.nodes;
    uploadInput(state, topology);
    if (input.use_device_trial) {
        const std::size_t particle_count =
            state.corotated_trial_particles.size();
        if (state.corotated_trial_particles.size() != particle_count ||
            state.trial_corotated_hessian.size() != 81 * particle_count ||
            state.trial_tangent_matrix.size() != 9 * particle_count)
            throw std::runtime_error(
                "solid Corotated device trial operator size mismatch");
        state.corotated_particles.resize(particle_count);
        constexpr int block_size = 128;
        const int blocks = static_cast<int>((particle_count + block_size - 1) /
            block_size);
        if (blocks)
            injectCorotatedTrialKernel<<<blocks, block_size>>>(
                thrust::raw_pointer_cast(state.corotated_particles.data()),
                thrust::raw_pointer_cast(
                    state.corotated_trial_particles.data()),
                static_cast<int>(particle_count),
                thrust::raw_pointer_cast(
                    state.trial_corotated_hessian.data()),
                thrust::raw_pointer_cast(state.trial_tangent_matrix.data()));
        checkCuda(cudaGetLastError(),
            "solid Corotated device trial operator inject");
    } else if (!input.reuse_static) {
        state.corotated_particles.assign(
            input.particles.begin(), input.particles.end());
    }
}

void uploadTrialInput(DeviceState& state, const PdTrialStateInput& input)
{
    const bool resident_nodes = input.resident_nodes.size != 0;
    const std::size_t node_count = resident_nodes
        ? input.resident_nodes.size : input.nodes.size();
    if (!input.reuse_velocity) {
        if (resident_nodes) {
            state.node_keys.resize(node_count);
            state.node_indices.resize(node_count);
            state.mass.resize(node_count);
            state.projection.resize(9 * node_count);
            state.base_velocity.resize(3 * node_count);
            state.newton_velocity_increment.resize(3 * node_count);
            state.node_velocity.resize(3 * node_count);
            const auto view = GPU_STAGE3A::getDeviceGridNodeMapRawView(
                input.resident_nodes);
            constexpr int block_size = 128;
            const int blocks = static_cast<int>((node_count + block_size - 1) /
                block_size);
            if (blocks)
                initializeTrialNodesFromResidentMapKernel<<<blocks,
                    block_size>>>(view.nodes, static_cast<int>(node_count),
                    input.initial_velocity_increment.x,
                    input.initial_velocity_increment.y,
                    input.initial_velocity_increment.z,
                    thrust::raw_pointer_cast(state.node_keys.data()),
                    thrust::raw_pointer_cast(state.node_indices.data()),
                    thrust::raw_pointer_cast(state.mass.data()),
                    thrust::raw_pointer_cast(state.projection.data()),
                    thrust::raw_pointer_cast(state.base_velocity.data()),
                    thrust::raw_pointer_cast(
                        state.newton_velocity_increment.data()),
                    thrust::raw_pointer_cast(state.node_velocity.data()));
            checkCuda(cudaGetLastError(),
                "resident PD trial node initialize launch");
        } else {
            std::vector<double> base_velocity(node_count * 3);
            std::vector<double> velocity_increment(node_count * 3);
            std::vector<double> velocity(node_count * 3);
            for (const PdTrialNode& node : input.nodes) {
                if (node.node.idx < 0 ||
                    static_cast<std::size_t>(node.node.idx) >= node_count)
                    throw std::runtime_error(
                        "solid PD trial node index out of range");
                const std::size_t offset =
                    static_cast<std::size_t>(3 * node.node.idx);
                base_velocity[offset + 0] = node.base_velocity.x;
                base_velocity[offset + 1] = node.base_velocity.y;
                base_velocity[offset + 2] = node.base_velocity.z;
                velocity_increment[offset + 0] = node.velocity_increment.x;
                velocity_increment[offset + 1] = node.velocity_increment.y;
                velocity_increment[offset + 2] = node.velocity_increment.z;
                velocity[offset + 0] = node.velocity.x;
                velocity[offset + 1] = node.velocity.y;
                velocity[offset + 2] = node.velocity.z;
            }
            state.base_velocity.assign(
                base_velocity.begin(), base_velocity.end());
            state.newton_velocity_increment.assign(
                velocity_increment.begin(), velocity_increment.end());
            state.node_velocity.assign(velocity.begin(), velocity.end());
        }
    }
    // Only the trial grid velocity changes across Newton iterations of one
    // step; when reuse_static is set the topology, particles and parallel
    // targets already reside on the device from the first iteration.
    if (input.reuse_static)
        return;
    if (!resident_nodes) {
        std::vector<std::pair<GridKey, int>> keys;
        std::vector<double> mass(node_count);
        std::vector<double> projection(node_count * 9);
        keys.reserve(node_count);
        for (const PdTrialNode& node : input.nodes) {
            keys.emplace_back(node.node.key, node.node.idx);
            mass[static_cast<std::size_t>(node.node.idx)] = node.node.mass;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    projection[static_cast<std::size_t>(
                        9 * node.node.idx + 3 * r + c)] =
                        node.node.projection[3 * r + c];
        }
        std::sort(keys.begin(), keys.end());
        for (std::size_t i = 1; i < keys.size(); ++i)
            if (keys[i - 1].first == keys[i].first)
                throw std::runtime_error("solid PD trial duplicate node key");
        std::vector<GridKey> sorted_keys(keys.size());
        std::vector<int> sorted_indices(keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i) {
            sorted_keys[i] = keys[i].first;
            sorted_indices[i] = keys[i].second;
        }
        state.node_keys.assign(sorted_keys.begin(), sorted_keys.end());
        state.node_indices.assign(sorted_indices.begin(), sorted_indices.end());
        state.mass.assign(mass.begin(), mass.end());
        state.projection.assign(projection.begin(), projection.end());
    }
    const bool resident_particles = input.resident_particles.size != 0;
    if (resident_particles &&
        (!input.resident_particles.rest_x ||
            !input.resident_particles.current_x ||
            !input.resident_particles.f || !input.resident_particles.mass ||
            !input.resident_particles.volume))
        throw std::runtime_error("incomplete resident PD trial particle input");
    const bool reuse_particle_metadata = resident_particles &&
        state.trial_particles.size() == input.resident_particles.size;
    if (!reuse_particle_metadata) {
        state.trial_particles.assign(
            input.particles.begin(), input.particles.end());
    }
    if (input.particles.size() != state.trial_particles.size())
        throw std::runtime_error("PD trial pinning metadata size mismatch");
    std::vector<DeviceState::TrialPinningPatch> pinning_patches(
        input.particles.size());
    for (std::size_t p = 0; p < input.particles.size(); ++p) {
        pinning_patches[p].pinning_a = input.particles[p].pinning_a;
        pinning_patches[p].pinning_b = input.particles[p].pinning_b;
        pinning_patches[p].pinning_k = input.particles[p].pinning_k;
        pinning_patches[p].pinning_d = input.particles[p].pinning_d;
        pinning_patches[p].pinning_target_x =
            input.particles[p].pinning_target_x;
        pinning_patches[p].pinning_target_v =
            input.particles[p].pinning_target_v;
        pinning_patches[p].pinning_from_resident_x0 =
            input.particles[p].pinning_from_resident_x0;
        pinning_patches[p].parallel_target_begin =
            input.particles[p].parallel_target_begin;
        pinning_patches[p].parallel_target_count =
            input.particles[p].parallel_target_count;
    }
    state.trial_pinning_patches.assign(
        pinning_patches.begin(), pinning_patches.end());
    state.parallel_targets.assign(input.parallel_targets.begin(),
        input.parallel_targets.end());
    constexpr int block_size = 128;
    if (resident_particles) {
        if (state.trial_particles.size() != input.resident_particles.size)
            throw std::runtime_error(
                "resident PD trial metadata size mismatch");
        const int blocks = static_cast<int>((input.resident_particles.size +
            block_size - 1) / block_size);
        if (blocks)
            patchPdTrialParticlesFromOwnerKernel<<<blocks, block_size>>>(
                thrust::raw_pointer_cast(state.trial_particles.data()),
                static_cast<int>(input.resident_particles.size),
                input.resident_particles);
        checkCuda(cudaGetLastError(),
            "resident PD trial particle patch launch");
    }
    const int pinning_blocks = static_cast<int>((input.particles.size() +
        block_size - 1) / block_size);
    if (pinning_blocks)
        patchPdTrialPinningKernel<<<pinning_blocks, block_size>>>(
            thrust::raw_pointer_cast(state.trial_particles.data()),
            thrust::raw_pointer_cast(state.trial_pinning_patches.data()),
            static_cast<int>(input.particles.size()), input.dt);
    checkCuda(cudaGetLastError(), "PD trial pinning patch launch");
}

void uploadCorotatedTrialInput(DeviceState& state,
    const CorotatedTrialStateInput& input)
{
    const bool resident_nodes = input.resident_nodes.size != 0;
    const std::size_t node_count = resident_nodes
        ? input.resident_nodes.size : input.nodes.size();
    std::vector<std::pair<GridKey, int>> keys;
    if (!input.reuse_velocity) {
        if (resident_nodes) {
            state.node_keys.resize(node_count);
            state.node_indices.resize(node_count);
            state.mass.resize(node_count);
            state.projection.resize(9 * node_count);
            state.base_velocity.resize(3 * node_count);
            state.newton_velocity_increment.resize(3 * node_count);
            state.node_velocity.resize(3 * node_count);
            const auto view = GPU_STAGE3A::getDeviceGridNodeMapRawView(
                input.resident_nodes);
            constexpr int block_size = 128;
            const int blocks = static_cast<int>((node_count + block_size - 1) /
                block_size);
            if (blocks)
                initializeTrialNodesFromResidentMapKernel<<<blocks,
                    block_size>>>(view.nodes, static_cast<int>(node_count),
                    input.initial_velocity_increment.x,
                    input.initial_velocity_increment.y,
                    input.initial_velocity_increment.z,
                    thrust::raw_pointer_cast(state.node_keys.data()),
                    thrust::raw_pointer_cast(state.node_indices.data()),
                    thrust::raw_pointer_cast(state.mass.data()),
                    thrust::raw_pointer_cast(state.projection.data()),
                    thrust::raw_pointer_cast(state.base_velocity.data()),
                    thrust::raw_pointer_cast(
                        state.newton_velocity_increment.data()),
                    thrust::raw_pointer_cast(state.node_velocity.data()));
            checkCuda(cudaGetLastError(),
                "resident Corotated trial node initialize launch");
        } else {
            std::vector<double> base_velocity(node_count * 3);
            std::vector<double> velocity_increment(node_count * 3);
            std::vector<double> velocity(node_count * 3);
            for (const PdTrialNode& node : input.nodes) {
                if (node.node.idx < 0 ||
                    static_cast<std::size_t>(node.node.idx) >= node_count)
                    throw std::runtime_error(
                        "solid Corotated trial node index out of range");
                const std::size_t offset =
                    static_cast<std::size_t>(3 * node.node.idx);
                base_velocity[offset + 0] = node.base_velocity.x;
                base_velocity[offset + 1] = node.base_velocity.y;
                base_velocity[offset + 2] = node.base_velocity.z;
                velocity_increment[offset + 0] = node.velocity_increment.x;
                velocity_increment[offset + 1] = node.velocity_increment.y;
                velocity_increment[offset + 2] = node.velocity_increment.z;
                velocity[offset + 0] = node.velocity.x;
                velocity[offset + 1] = node.velocity.y;
                velocity[offset + 2] = node.velocity.z;
            }
            state.base_velocity.assign(
                base_velocity.begin(), base_velocity.end());
            state.newton_velocity_increment.assign(
                velocity_increment.begin(), velocity_increment.end());
            state.node_velocity.assign(velocity.begin(), velocity.end());
        }
    }
    auto upload_pinning = [&]() {
        if (input.particles.size() !=
            state.corotated_trial_particles.size())
            throw std::runtime_error(
                "Corotated trial pinning metadata size mismatch");
        std::vector<DeviceState::TrialPinningPatch> pinning_patches(
            input.particles.size());
        for (std::size_t p = 0; p < input.particles.size(); ++p) {
            pinning_patches[p].pinning_a = input.particles[p].pinning_a;
            pinning_patches[p].pinning_b = input.particles[p].pinning_b;
            pinning_patches[p].pinning_k = input.particles[p].pinning_k;
            pinning_patches[p].pinning_d = input.particles[p].pinning_d;
            pinning_patches[p].pinning_target_x =
                input.particles[p].pinning_target_x;
            pinning_patches[p].pinning_target_v =
                input.particles[p].pinning_target_v;
            pinning_patches[p].pinning_from_resident_x0 =
                input.particles[p].pinning_from_resident_x0;
            pinning_patches[p].parallel_target_begin =
                input.particles[p].parallel_target_begin;
            pinning_patches[p].parallel_target_count =
                input.particles[p].parallel_target_count;
        }
        state.trial_pinning_patches.assign(
            pinning_patches.begin(), pinning_patches.end());
        state.parallel_targets.assign(input.parallel_targets.begin(),
            input.parallel_targets.end());
        constexpr int block_size = 128;
        const int blocks = static_cast<int>((input.particles.size() +
            block_size - 1) / block_size);
        if (blocks) {
            patchCorotatedTrialPinningKernel<<<blocks, block_size>>>(
                thrust::raw_pointer_cast(
                    state.corotated_trial_particles.data()),
                thrust::raw_pointer_cast(
                    state.trial_pinning_patches.data()),
                static_cast<int>(input.particles.size()), input.dt);
            checkCuda(cudaGetLastError(),
                "Corotated trial pinning patch launch");
        }
    };
    if (input.reuse_static) {
        if (!input.particles.empty())
            upload_pinning();
        return;
    }
    if (!resident_nodes) {
        std::vector<double> mass(node_count);
        std::vector<double> projection(9 * node_count);
        keys.reserve(node_count);
        for (const PdTrialNode& node : input.nodes) {
            if (node.node.idx < 0 ||
                static_cast<std::size_t>(node.node.idx) >= node_count)
                throw std::runtime_error(
                    "solid Corotated trial node index out of range");
            keys.emplace_back(node.node.key, node.node.idx);
            mass[static_cast<std::size_t>(node.node.idx)] = node.node.mass;
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    projection[static_cast<std::size_t>(
                        9 * node.node.idx + 3 * row + column)] =
                        node.node.projection[3 * row + column];
        }
        std::sort(keys.begin(), keys.end());
        for (std::size_t i = 1; i < keys.size(); ++i)
            if (keys[i - 1].first == keys[i].first)
                throw std::runtime_error(
                    "solid Corotated trial duplicate node key");
        std::vector<GridKey> sorted_keys(keys.size());
        std::vector<int> sorted_indices(keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i) {
            sorted_keys[i] = keys[i].first;
            sorted_indices[i] = keys[i].second;
        }
        state.node_keys.assign(sorted_keys.begin(), sorted_keys.end());
        state.node_indices.assign(sorted_indices.begin(), sorted_indices.end());
        state.mass.assign(mass.begin(), mass.end());
        state.projection.assign(projection.begin(), projection.end());
    }
    const bool resident_particles = input.resident_particles.size != 0;
    if (resident_particles &&
        (!input.resident_particles.rest_x ||
            !input.resident_particles.current_x ||
            !input.resident_particles.f || !input.resident_particles.mass ||
            !input.resident_particles.volume))
        throw std::runtime_error(
            "incomplete resident Corotated trial particle input");
    const bool reuse_particle_metadata = resident_particles &&
        state.corotated_trial_particles.size() ==
            input.resident_particles.size;
    if (!reuse_particle_metadata) {
        state.corotated_trial_particles.assign(
            input.particles.begin(), input.particles.end());
    }
    if (resident_particles) {
        if (state.corotated_trial_particles.size() !=
            input.resident_particles.size)
            throw std::runtime_error(
                "resident Corotated trial metadata size mismatch");
        constexpr int block_size = 128;
        const int blocks = static_cast<int>((input.resident_particles.size +
            block_size - 1) / block_size);
        if (blocks)
            patchCorotatedTrialParticlesFromOwnerKernel<<<blocks, block_size>>>(
                thrust::raw_pointer_cast(
                    state.corotated_trial_particles.data()),
                static_cast<int>(input.resident_particles.size),
                input.resident_particles);
        checkCuda(cudaGetLastError(),
            "resident Corotated trial particle patch launch");
    }
    upload_pinning();
}

void applyLinearCorotatedDevice(DeviceState& state,
    const LinearCorotatedOperatorInput& input, const double* x, double* y)
{
    constexpr int block_size = 128;
    const int node_blocks = (static_cast<int>(input.nodes.size()) + block_size - 1) / block_size;
    if (node_blocks)
        massApplyKernel<<<node_blocks, block_size>>>(
            thrust::raw_pointer_cast(state.mass.data()), x,
            static_cast<int>(input.nodes.size()), y);
    const int particle_blocks =
        (static_cast<int>(input.particles.size()) + block_size - 1) / block_size;
    if (particle_blocks)
        linearCorotatedApplyKernel<<<particle_blocks, block_size>>>(
            thrust::raw_pointer_cast(state.linear_particles.data()),
            static_cast<int>(input.particles.size()),
            thrust::raw_pointer_cast(state.node_keys.data()),
            thrust::raw_pointer_cast(state.node_indices.data()),
            static_cast<int>(input.nodes.size()), x, input.dx, input.dt, y);
    checkCuda(cudaGetLastError(), "solid linear-corotated operator kernel launch");
}

void applyCorotatedDevice(DeviceState& state, const CorotatedOperatorInput& input,
    const double* x, double* y)
{
    constexpr int block_size = 128;
    const int node_count = static_cast<int>(state.mass.size());
    const int particle_count = static_cast<int>(state.corotated_particles.size());
    const int node_blocks = (node_count + block_size - 1) / block_size;
    if (node_blocks)
        massApplyKernel<<<node_blocks, block_size>>>(
            thrust::raw_pointer_cast(state.mass.data()), x,
            node_count, y);
    const int particle_blocks =
        (particle_count + block_size - 1) / block_size;
    if (particle_blocks)
        corotatedApplyKernel<<<particle_blocks, block_size>>>(
            thrust::raw_pointer_cast(state.corotated_particles.data()),
            particle_count,
            thrust::raw_pointer_cast(state.node_keys.data()),
            thrust::raw_pointer_cast(state.node_indices.data()),
            node_count, x, input.dx, input.dt, y);
    checkCuda(cudaGetLastError(), "solid Corotated operator kernel launch");
}

} // namespace

void evaluatePdTrialState3D(PersistentPdOperator3D& context,
    const PdTrialStateInput& input, PdTrialStateOutput& output,
    PdTrialStateStats* stats)
{
    if (!(input.dx > 0.0) || !(input.dt > 0.0) ||
        (input.nodes.empty() && input.resident_nodes.size == 0 &&
            !input.reuse_static))
        throw std::runtime_error("solid PD trial state requires nonempty grid and positive dx/dt");
    DeviceState& state = stateFor(context);
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    uploadTrialInput(state, input);
    // On reused Newton states the host particle array may be empty: the static
    // particle data already lives in state.trial_particles.
    const std::size_t particle_count = state.trial_particles.size();
    const int particles = static_cast<int>(particle_count);
    const int node_count = static_cast<int>(state.node_velocity.size() / 3);
    state.trial_velocity.assign(3 * particle_count, 0.0);
    state.trial_grad_velocity.assign(9 * particle_count, 0.0);
    state.trial_F.assign(9 * particle_count, 0.0);
    state.trial_J.assign(particle_count, 0.0);
    state.trial_position.assign(3 * particle_count, 0.0);
    state.trial_force_stress.assign(9 * particle_count, 0.0);
    state.trial_force.assign(3 * particle_count, 0.0);
    state.trial_tangent_a.assign(particle_count, 0.0);
    state.trial_tangent_matrix.assign(9 * particle_count, 0.0);
    state.trial_contact_a.assign(particle_count, 0.0);
    state.trial_energy.assign(particle_count, 0.0);
    checkCuda(cudaDeviceSynchronize(), "solid PD trial upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    constexpr int block_size = 128;
    const int blocks = (particles + block_size - 1) / block_size;
    const auto gather_begin = std::chrono::high_resolution_clock::now();
    if (blocks)
        pdTrialGatherKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(state.trial_particles.data()), particles,
            thrust::raw_pointer_cast(state.parallel_targets.data()),
            thrust::raw_pointer_cast(state.node_keys.data()),
            thrust::raw_pointer_cast(state.node_indices.data()),
            node_count,
            thrust::raw_pointer_cast(state.node_velocity.data()), input.dx,
            input.dt,
            thrust::raw_pointer_cast(state.trial_velocity.data()),
            thrust::raw_pointer_cast(state.trial_grad_velocity.data()),
            thrust::raw_pointer_cast(state.trial_F.data()),
            thrust::raw_pointer_cast(state.trial_J.data()),
            thrust::raw_pointer_cast(state.trial_position.data()),
            thrust::raw_pointer_cast(state.trial_force_stress.data()),
            thrust::raw_pointer_cast(state.trial_force.data()),
            thrust::raw_pointer_cast(state.trial_tangent_a.data()),
            thrust::raw_pointer_cast(state.trial_tangent_matrix.data()),
            thrust::raw_pointer_cast(state.trial_energy.data()));
    checkCuda(cudaGetLastError(), "solid PD trial gather launch");
    checkCuda(cudaDeviceSynchronize(), "solid PD trial gather synchronize");
    const auto gather_end = std::chrono::high_resolution_clock::now();

    const auto contact_begin = std::chrono::high_resolution_clock::now();
    if (input.enable_tl_contact && blocks) {
        const double cutoff = input.dx;
        state.contact_keys.resize(particle_count);
        state.contact_indices.resize(particle_count);
        contactKeyKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(state.trial_position.data()), particles, cutoff,
            thrust::raw_pointer_cast(state.contact_keys.data()),
            thrust::raw_pointer_cast(state.contact_indices.data()));
        checkCuda(cudaGetLastError(), "solid TL contact key launch");
        thrust::sort_by_key(state.contact_keys.begin(), state.contact_keys.end(),
            state.contact_indices.begin());
        tlContactKernel<PdTrialParticle><<<blocks, block_size>>>(
            thrust::raw_pointer_cast(state.trial_particles.data()), particles,
            thrust::raw_pointer_cast(state.trial_position.data()),
            thrust::raw_pointer_cast(state.contact_keys.data()),
            thrust::raw_pointer_cast(state.contact_indices.data()), cutoff, input.dt,
            thrust::raw_pointer_cast(state.trial_force.data()),
            thrust::raw_pointer_cast(state.trial_contact_a.data()),
            thrust::raw_pointer_cast(state.trial_tangent_a.data()),
            thrust::raw_pointer_cast(state.trial_tangent_matrix.data()));
        checkCuda(cudaGetLastError(), "solid TL contact launch");
        checkCuda(cudaDeviceSynchronize(), "solid TL contact synchronize");
    }
    const auto contact_end = std::chrono::high_resolution_clock::now();

    if (input.scatter_residual) {
        state.trial_node_residual.assign(3 * node_count, 0.0);
        if (blocks)
            residualScatterKernel<<<blocks, block_size>>>(
                thrust::raw_pointer_cast(state.trial_particles.data()), particles,
                thrust::raw_pointer_cast(state.node_keys.data()),
                thrust::raw_pointer_cast(state.node_indices.data()), node_count,
                thrust::raw_pointer_cast(state.trial_force_stress.data()),
                thrust::raw_pointer_cast(state.trial_force.data()), input.dx, input.dt,
                thrust::raw_pointer_cast(state.trial_node_residual.data()));
        checkCuda(cudaGetLastError(), "solid PD residual scatter launch");
        checkCuda(cudaDeviceSynchronize(), "solid PD residual scatter synchronize");
    }

    const auto download_begin = std::chrono::high_resolution_clock::now();
    output.force_energy = thrust::reduce(
        state.trial_energy.begin(), state.trial_energy.end(), 0.0);
    if (input.scatter_residual) {
        // Device residual and PCG own every per-iteration particle field. Keep
        // gradV and tangent on device too; gradV is materialized once after the
        // final Newton state and PCG injects the tangent device-to-device.
        output.particle_velocity.clear();
        output.particle_grad_velocity.clear();
        output.deformation_gradient.clear();
        output.determinant.clear();
        output.trial_position.clear();
        output.force_stress.clear();
        output.force.clear();
        output.tangent_a.clear();
        output.tangent_matrix.clear();
        output.tl_contact_a.clear();
        if (input.download_node_residual) {
            output.node_residual.resize(state.trial_node_residual.size());
            thrust::copy(state.trial_node_residual.begin(),
                state.trial_node_residual.end(), output.node_residual.begin());
        }
        else
            output.node_residual.clear();
    } else {
        output.particle_velocity.resize(3 * particle_count);
        output.particle_grad_velocity.resize(9 * particle_count);
        output.deformation_gradient.resize(9 * particle_count);
        output.determinant.resize(particle_count);
        output.trial_position.resize(3 * particle_count);
        output.force_stress.resize(9 * particle_count);
        output.force.resize(3 * particle_count);
        output.tangent_a.resize(particle_count);
        output.tangent_matrix.resize(9 * particle_count);
        output.tl_contact_a.resize(particle_count);
        thrust::copy(state.trial_velocity.begin(), state.trial_velocity.end(),
            output.particle_velocity.begin());
        thrust::copy(state.trial_grad_velocity.begin(), state.trial_grad_velocity.end(),
            output.particle_grad_velocity.begin());
        thrust::copy(state.trial_F.begin(), state.trial_F.end(),
            output.deformation_gradient.begin());
        thrust::copy(state.trial_J.begin(), state.trial_J.end(), output.determinant.begin());
        thrust::copy(state.trial_position.begin(), state.trial_position.end(),
            output.trial_position.begin());
        thrust::copy(state.trial_force_stress.begin(), state.trial_force_stress.end(),
            output.force_stress.begin());
        thrust::copy(state.trial_force.begin(), state.trial_force.end(), output.force.begin());
        thrust::copy(state.trial_tangent_a.begin(), state.trial_tangent_a.end(),
            output.tangent_a.begin());
        thrust::copy(state.trial_tangent_matrix.begin(), state.trial_tangent_matrix.end(),
            output.tangent_matrix.begin());
        thrust::copy(state.trial_contact_a.begin(), state.trial_contact_a.end(),
            output.tl_contact_a.begin());
        output.node_residual.clear();
    }
    checkCuda(cudaDeviceSynchronize(), "solid PD trial download synchronize");
    const auto download_end = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->upload_ms = std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
        stats->gather_ms = std::chrono::duration<double, std::milli>(gather_end - gather_begin).count();
        stats->contact_ms = std::chrono::duration<double, std::milli>(contact_end - contact_begin).count();
        stats->download_ms = std::chrono::duration<double, std::milli>(download_end - download_begin).count();
        std::size_t contacts = 0;
        for (double a : output.tl_contact_a)
            contacts += a != 0.0;
        stats->active_tl_contacts = contacts;
    }
}

void downloadPdTrialGradVelocity3D(PersistentPdOperator3D& context,
    std::vector<double>& particle_grad_velocity)
{
    DeviceState& state = stateFor(context);
    particle_grad_velocity.resize(state.trial_grad_velocity.size());
    thrust::copy(state.trial_grad_velocity.begin(), state.trial_grad_velocity.end(),
        particle_grad_velocity.begin());
    checkCuda(cudaDeviceSynchronize(), "solid PD final gradV download synchronize");
}

void downloadPdNewtonState3D(PersistentPdOperator3D& context,
    std::vector<double>& velocity_increment,
    std::vector<double>& particle_grad_velocity)
{
    DeviceState& state = stateFor(context);
    velocity_increment.resize(state.newton_velocity_increment.size());
    particle_grad_velocity.resize(state.trial_grad_velocity.size());
    thrust::copy(state.newton_velocity_increment.begin(),
        state.newton_velocity_increment.end(), velocity_increment.begin());
    thrust::copy(state.trial_grad_velocity.begin(), state.trial_grad_velocity.end(),
        particle_grad_velocity.begin());
    checkCuda(cudaDeviceSynchronize(),
        "solid PD final Newton state download synchronize");
}

void exportFinalGridVelocityToNodeMap3D(PersistentPdOperator3D& context,
    GPU_STAGE3A::DeviceGridNodeMap& nodes)
{
    DeviceState& state = stateFor(context);
    const std::size_t node_count = state.node_keys.size();
    if (nodes.size != node_count || state.node_indices.size() != node_count ||
        state.node_velocity.size() != 3 * node_count)
        throw std::runtime_error("solid final grid export size mismatch");
    auto node_view = GPU_STAGE3A::getDeviceGridNodeMapMutableRawView(nodes);
    constexpr int block_size = 128;
    const int blocks = static_cast<int>((node_count + block_size - 1) /
        block_size);
    if (blocks)
        exportFinalGridVelocityKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(state.node_keys.data()),
            thrust::raw_pointer_cast(state.node_indices.data()),
            thrust::raw_pointer_cast(state.node_velocity.data()),
            node_view.nodes, static_cast<int>(node_count));
    checkCuda(cudaGetLastError(), "solid final grid export launch");
}

void evaluateCorotatedTrialState3D(PersistentPdOperator3D& context,
    const CorotatedTrialStateInput& input, PdTrialStateOutput& output,
    std::vector<double>& projected_hessian, PdTrialStateStats* stats)
{
    if (!(input.dx > 0.0) || !(input.dt > 0.0) ||
        (input.nodes.empty() && input.resident_nodes.size == 0 &&
            !input.reuse_static))
        throw std::runtime_error("solid Corotated trial state requires nonempty grid and positive dx/dt");
    DeviceState& state = stateFor(context);
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    uploadCorotatedTrialInput(state, input);
    const std::size_t particle_count =
        state.corotated_trial_particles.size();
    const int particles = static_cast<int>(particle_count);
    const int node_count = static_cast<int>(state.node_velocity.size() / 3);
    state.trial_velocity.assign(3 * particle_count, 0.0);
    state.trial_grad_velocity.assign(9 * particle_count, 0.0);
    state.trial_F.assign(9 * particle_count, 0.0);
    state.trial_J.assign(particle_count, 0.0);
    state.trial_position.assign(3 * particle_count, 0.0);
    state.trial_force_stress.assign(9 * particle_count, 0.0);
    state.trial_force.assign(3 * particle_count, 0.0);
    state.trial_tangent_a.assign(particle_count, 0.0);
    state.trial_tangent_matrix.assign(9 * particle_count, 0.0);
    state.trial_contact_a.assign(particle_count, 0.0);
    state.trial_energy.assign(particle_count, 0.0);
    state.trial_corotated_hessian.assign(81 * particle_count, 0.0);
    checkCuda(cudaDeviceSynchronize(), "solid Corotated trial upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    constexpr int block_size = 128;
    const int blocks = (particles + block_size - 1) / block_size;
    const auto gather_begin = std::chrono::high_resolution_clock::now();
    if (blocks)
        corotatedTrialGatherKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(state.corotated_trial_particles.data()), particles,
            thrust::raw_pointer_cast(state.parallel_targets.data()),
            thrust::raw_pointer_cast(state.node_keys.data()),
            thrust::raw_pointer_cast(state.node_indices.data()),
            node_count,
            thrust::raw_pointer_cast(state.node_velocity.data()), input.dx,
            input.dt, input.ghost_collision, input.ghost_penalty,
            thrust::raw_pointer_cast(state.trial_velocity.data()),
            thrust::raw_pointer_cast(state.trial_grad_velocity.data()),
            thrust::raw_pointer_cast(state.trial_F.data()),
            thrust::raw_pointer_cast(state.trial_J.data()),
            thrust::raw_pointer_cast(state.trial_position.data()),
            thrust::raw_pointer_cast(state.trial_force_stress.data()),
            thrust::raw_pointer_cast(state.trial_force.data()),
            thrust::raw_pointer_cast(state.trial_tangent_a.data()),
            thrust::raw_pointer_cast(state.trial_tangent_matrix.data()),
            thrust::raw_pointer_cast(state.trial_corotated_hessian.data()),
            thrust::raw_pointer_cast(state.trial_energy.data()));
    checkCuda(cudaGetLastError(), "solid Corotated trial gather launch");
    checkCuda(cudaDeviceSynchronize(), "solid Corotated trial gather synchronize");
    const auto gather_end = std::chrono::high_resolution_clock::now();

    const auto contact_begin = std::chrono::high_resolution_clock::now();
    if (input.enable_tl_contact && blocks) {
        const double cutoff = input.dx;
        state.contact_keys.resize(particle_count);
        state.contact_indices.resize(particle_count);
        contactKeyKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(state.trial_position.data()), particles, cutoff,
            thrust::raw_pointer_cast(state.contact_keys.data()),
            thrust::raw_pointer_cast(state.contact_indices.data()));
        checkCuda(cudaGetLastError(), "solid Corotated TL contact key launch");
        thrust::sort_by_key(state.contact_keys.begin(), state.contact_keys.end(),
            state.contact_indices.begin());
        tlContactKernel<CorotatedTrialParticle><<<blocks, block_size>>>(
            thrust::raw_pointer_cast(state.corotated_trial_particles.data()), particles,
            thrust::raw_pointer_cast(state.trial_position.data()),
            thrust::raw_pointer_cast(state.contact_keys.data()),
            thrust::raw_pointer_cast(state.contact_indices.data()), cutoff, input.dt,
            thrust::raw_pointer_cast(state.trial_force.data()),
            thrust::raw_pointer_cast(state.trial_contact_a.data()),
            thrust::raw_pointer_cast(state.trial_tangent_a.data()),
            thrust::raw_pointer_cast(state.trial_tangent_matrix.data()));
        checkCuda(cudaGetLastError(), "solid Corotated TL contact launch");
        checkCuda(cudaDeviceSynchronize(), "solid Corotated TL contact synchronize");
    }
    const auto contact_end = std::chrono::high_resolution_clock::now();

    if (input.scatter_residual) {
        state.trial_node_residual.assign(3 * node_count, 0.0);
        if (blocks)
            residualScatterKernel<CorotatedTrialParticle>
                <<<blocks, block_size>>>(
                    thrust::raw_pointer_cast(
                        state.corotated_trial_particles.data()),
                    particles,
                    thrust::raw_pointer_cast(state.node_keys.data()),
                    thrust::raw_pointer_cast(state.node_indices.data()),
                    node_count,
                    thrust::raw_pointer_cast(
                        state.trial_force_stress.data()),
                    thrust::raw_pointer_cast(state.trial_force.data()),
                    input.dx, input.dt,
                    thrust::raw_pointer_cast(
                        state.trial_node_residual.data()));
        checkCuda(cudaGetLastError(),
            "solid Corotated residual scatter launch");
        checkCuda(cudaDeviceSynchronize(),
            "solid Corotated residual scatter synchronize");
    }

    const auto download_begin = std::chrono::high_resolution_clock::now();
    output.force_energy = thrust::reduce(
        state.trial_energy.begin(), state.trial_energy.end(), 0.0);
    if (input.scatter_residual) {
        output.particle_velocity.clear();
        output.particle_grad_velocity.clear();
        output.deformation_gradient.clear();
        output.determinant.clear();
        output.trial_position.clear();
        output.force_stress.clear();
        output.force.clear();
        output.tangent_a.clear();
        output.tangent_matrix.clear();
        output.tl_contact_a.clear();
        projected_hessian.clear();
        if (input.download_node_residual) {
            output.node_residual.resize(state.trial_node_residual.size());
            thrust::copy(state.trial_node_residual.begin(),
                state.trial_node_residual.end(), output.node_residual.begin());
        } else {
            output.node_residual.clear();
        }
        checkCuda(cudaDeviceSynchronize(),
            "solid Corotated compact residual download synchronize");
        const auto download_end = std::chrono::high_resolution_clock::now();
        if (stats) {
            stats->upload_ms = std::chrono::duration<double, std::milli>(
                upload_end - upload_begin).count();
            stats->gather_ms = std::chrono::duration<double, std::milli>(
                gather_end - gather_begin).count();
            stats->contact_ms = std::chrono::duration<double, std::milli>(
                contact_end - contact_begin).count();
            stats->download_ms = std::chrono::duration<double, std::milli>(
                download_end - download_begin).count();
            stats->active_tl_contacts = 0;
        }
        return;
    }
    output.particle_velocity.resize(3 * particle_count);
    output.particle_grad_velocity.resize(9 * particle_count);
    output.deformation_gradient.resize(9 * particle_count);
    output.determinant.resize(particle_count);
    output.trial_position.resize(3 * particle_count);
    output.force_stress.resize(9 * particle_count);
    output.force.resize(3 * particle_count);
    output.tangent_a.resize(particle_count);
    output.tangent_matrix.resize(9 * particle_count);
    output.tl_contact_a.resize(particle_count);
    projected_hessian.resize(81 * particle_count);
    thrust::copy(state.trial_velocity.begin(), state.trial_velocity.end(),
        output.particle_velocity.begin());
    thrust::copy(state.trial_grad_velocity.begin(), state.trial_grad_velocity.end(),
        output.particle_grad_velocity.begin());
    thrust::copy(state.trial_F.begin(), state.trial_F.end(),
        output.deformation_gradient.begin());
    thrust::copy(state.trial_J.begin(), state.trial_J.end(), output.determinant.begin());
    thrust::copy(state.trial_position.begin(), state.trial_position.end(),
        output.trial_position.begin());
    thrust::copy(state.trial_force_stress.begin(), state.trial_force_stress.end(),
        output.force_stress.begin());
    thrust::copy(state.trial_force.begin(), state.trial_force.end(), output.force.begin());
    thrust::copy(state.trial_tangent_a.begin(), state.trial_tangent_a.end(),
        output.tangent_a.begin());
    thrust::copy(state.trial_tangent_matrix.begin(), state.trial_tangent_matrix.end(),
        output.tangent_matrix.begin());
    thrust::copy(state.trial_contact_a.begin(), state.trial_contact_a.end(),
        output.tl_contact_a.begin());
    thrust::copy(state.trial_corotated_hessian.begin(), state.trial_corotated_hessian.end(),
        projected_hessian.begin());
    checkCuda(cudaDeviceSynchronize(), "solid Corotated trial download synchronize");
    const auto download_end = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->upload_ms = std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
        stats->gather_ms = std::chrono::duration<double, std::milli>(gather_end - gather_begin).count();
        stats->contact_ms = std::chrono::duration<double, std::milli>(contact_end - contact_begin).count();
        stats->download_ms = std::chrono::duration<double, std::milli>(download_end - download_begin).count();
        std::size_t contacts = 0;
        for (double a : output.tl_contact_a)
            contacts += a != 0.0;
        stats->active_tl_contacts = contacts;
    }
}

void applyPdMatrixFree3D(PersistentPdOperator3D& context,
    const PdOperatorInput& input, const std::vector<double>& x,
    std::vector<double>& y, PdOperatorStats* stats)
{
    const auto total_begin = std::chrono::high_resolution_clock::now();
    if (!(input.dx > 0.0))
        throw std::runtime_error("solid PD operator requires positive dx");
    if (x.size() != input.nodes.size() * 3)
        throw std::runtime_error("solid PD operator vector size mismatch");
    DeviceState& state = stateFor(context);
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    uploadInput(state, input);
    state.x.assign(x.begin(), x.end());
    state.y.resize(x.size());
    checkCuda(cudaDeviceSynchronize(), "solid PD operator upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    const auto apply_begin = std::chrono::high_resolution_clock::now();
    applyDevice(state, input, thrust::raw_pointer_cast(state.x.data()),
        thrust::raw_pointer_cast(state.y.data()));
    checkCuda(cudaDeviceSynchronize(), "solid PD operator kernel synchronize");
    const auto apply_end = std::chrono::high_resolution_clock::now();

    const auto download_begin = std::chrono::high_resolution_clock::now();
    y.resize(x.size());
    thrust::copy(state.y.begin(), state.y.end(), y.begin());
    checkCuda(cudaDeviceSynchronize(), "solid PD operator download synchronize");
    const auto download_end = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->nodes = input.nodes.size();
        stats->particles = input.particles.size();
        stats->upload_ms = std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
        stats->apply_ms = std::chrono::duration<double, std::milli>(apply_end - apply_begin).count();
        stats->download_ms = std::chrono::duration<double, std::milli>(download_end - download_begin).count();
        stats->total_ms = std::chrono::duration<double, std::milli>(download_end - total_begin).count();
    }
}

static void solvePdPcgInternal(PersistentPdOperator3D& context,
    const PdOperatorInput& input, const std::vector<double>* host_rhs,
    double relative_tolerance, double absolute_tolerance, int max_iterations,
    std::vector<double>* host_solution, PdPcgStats* stats)
{
    if (!(input.dx > 0.0))
        throw std::runtime_error("solid PD PCG input mismatch");
    DeviceState& state = stateFor(context);
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    uploadInput(state, input);
    if (input.use_device_tangent) {
        const int particle_count = static_cast<int>(state.particles.size());
        if (state.trial_tangent_matrix.size() !=
                static_cast<std::size_t>(9 * particle_count))
            throw std::runtime_error(
                "solid PD PCG device tangent size mismatch");
        constexpr int tblock = 128;
        const int tblocks = (particle_count + tblock - 1) / tblock;
        if (tblocks)
            injectDeviceTangentKernel<<<tblocks, tblock>>>(
                thrust::raw_pointer_cast(state.particles.data()), particle_count,
                thrust::raw_pointer_cast(state.trial_tangent_matrix.data()));
        checkCuda(cudaGetLastError(), "solid PD PCG device tangent inject");
    }
    const int entries = host_rhs ? static_cast<int>(host_rhs->size())
                                 : static_cast<int>(state.r.size());
    const int node_count = static_cast<int>(state.mass.size());
    if (entries != 3 * node_count)
        throw std::runtime_error("solid PD PCG vector size mismatch");
    state.x.assign(entries, 0.0);
    if (host_rhs)
        state.r.assign(host_rhs->begin(), host_rhs->end());
    state.z.resize(entries);
    state.p.resize(entries);
    state.q.resize(entries);
    constexpr int block_size = 128;
    const int blocks = (entries + block_size - 1) / block_size;
    projectDevice(state, node_count,
        thrust::raw_pointer_cast(state.r.data()));
    jacobiKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.mass.data()),
        thrust::raw_pointer_cast(state.r.data()), entries,
        thrust::raw_pointer_cast(state.z.data()));
    projectDevice(state, node_count,
        thrust::raw_pointer_cast(state.z.data()));
    thrust::copy(state.z.begin(), state.z.end(), state.p.begin());
    checkCuda(cudaGetLastError(), "solid PD PCG initialization");
    checkCuda(cudaDeviceSynchronize(), "solid PD PCG upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    double rz = thrust::inner_product(state.r.begin(), state.r.end(), state.z.begin(), 0.0);
    const double initial = std::sqrt(std::max(0.0, rz));
    const double tolerance = std::min(relative_tolerance * initial, absolute_tolerance);
    int iteration = 0;
    const auto solve_begin = std::chrono::high_resolution_clock::now();
    for (; iteration < max_iterations && std::sqrt(std::max(0.0, rz)) > tolerance;
         ++iteration) {
        applyDevice(state, input, thrust::raw_pointer_cast(state.p.data()),
            thrust::raw_pointer_cast(state.q.data()));
        projectDevice(state, node_count,
            thrust::raw_pointer_cast(state.q.data()));
        const double pq = thrust::inner_product(state.p.begin(), state.p.end(), state.q.begin(), 0.0);
        const double alpha = rz / pq;
        axpyKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.x.data()),
            thrust::raw_pointer_cast(state.p.data()), alpha, entries);
        residualKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.r.data()),
            thrust::raw_pointer_cast(state.q.data()), alpha, entries);
        projectDevice(state, node_count,
            thrust::raw_pointer_cast(state.r.data()));
        jacobiKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.mass.data()),
            thrust::raw_pointer_cast(state.r.data()), entries,
            thrust::raw_pointer_cast(state.z.data()));
        projectDevice(state, node_count,
            thrust::raw_pointer_cast(state.z.data()));
        const double new_rz = thrust::inner_product(
            state.r.begin(), state.r.end(), state.z.begin(), 0.0);
        directionKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.p.data()),
            thrust::raw_pointer_cast(state.z.data()), new_rz / rz, entries);
        projectDevice(state, node_count,
            thrust::raw_pointer_cast(state.p.data()));
        rz = new_rz;
    }
    checkCuda(cudaGetLastError(), "solid PD PCG iteration kernel");
    checkCuda(cudaDeviceSynchronize(), "solid PD PCG solve synchronize");
    const auto solve_end = std::chrono::high_resolution_clock::now();
    const auto download_begin = std::chrono::high_resolution_clock::now();
    if (host_solution) {
        host_solution->resize(static_cast<std::size_t>(entries));
        thrust::copy(state.x.begin(), state.x.end(), host_solution->begin());
        checkCuda(cudaDeviceSynchronize(), "solid PD PCG download synchronize");
    }
    const auto download_end = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->iterations = iteration;
        stats->initial_preconditioned_norm = initial;
        stats->final_preconditioned_norm = std::sqrt(std::max(0.0, rz));
        stats->upload_ms = std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
        stats->solve_ms = std::chrono::duration<double, std::milli>(solve_end - solve_begin).count();
        stats->download_ms = std::chrono::duration<double, std::milli>(download_end - download_begin).count();
    }
}

void solvePdPcg3D(PersistentPdOperator3D& context, const PdOperatorInput& input,
    const std::vector<double>& rhs, double relative_tolerance,
    double absolute_tolerance, int max_iterations, std::vector<double>& solution,
    PdPcgStats* stats)
{
    solvePdPcgInternal(context, input, &rhs, relative_tolerance,
        absolute_tolerance, max_iterations, &solution, stats);
}

double preparePdNewtonResidual3D(PersistentPdOperator3D& context,
    const GPU_STAGE3A::Vec3d& gravity, double dt)
{
    DeviceState& state = stateFor(context);
    const int nodes = static_cast<int>(state.mass.size());
    const int entries = 3 * nodes;
    if (!(dt > 0.0) || state.trial_node_residual.size() !=
            static_cast<std::size_t>(entries) ||
        state.newton_velocity_increment.size() != static_cast<std::size_t>(entries))
        throw std::runtime_error("solid PD device Newton residual state mismatch");
    state.r.resize(static_cast<std::size_t>(entries));
    constexpr int block_size = 128;
    const int blocks = (nodes + block_size - 1) / block_size;
    if (blocks)
        assembleNewtonResidualKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(state.trial_node_residual.data()),
            thrust::raw_pointer_cast(state.mass.data()),
            thrust::raw_pointer_cast(state.newton_velocity_increment.data()), dt,
            gravity.x, gravity.y, gravity.z, nodes,
            thrust::raw_pointer_cast(state.r.data()));
    checkCuda(cudaGetLastError(), "solid PD Newton residual assembly");
    projectDevice(state, nodes, thrust::raw_pointer_cast(state.r.data()));
    state.z.resize(static_cast<std::size_t>(entries));
    const int entry_blocks = (entries + block_size - 1) / block_size;
    if (entry_blocks)
        jacobiKernel<<<entry_blocks, block_size>>>(
            thrust::raw_pointer_cast(state.mass.data()),
            thrust::raw_pointer_cast(state.r.data()), entries,
            thrust::raw_pointer_cast(state.z.data()));
    const double norm_sq = thrust::inner_product(
        state.r.begin(), state.r.end(), state.z.begin(), 0.0);
    checkCuda(cudaGetLastError(), "solid PD Newton residual norm");
    return std::sqrt(std::max(0.0, norm_sq));
}

void solvePdPcgDeviceResidual3D(PersistentPdOperator3D& context,
    const PdOperatorInput& input, double relative_tolerance,
    double absolute_tolerance, int max_iterations, PdPcgStats* stats)
{
    solvePdPcgInternal(context, input, nullptr, relative_tolerance,
        absolute_tolerance, max_iterations, nullptr, stats);
}

void applyPdNewtonIncrement3D(PersistentPdOperator3D& context)
{
    DeviceState& state = stateFor(context);
    const int entries = static_cast<int>(state.newton_velocity_increment.size());
    if (entries == 0 || state.base_velocity.size() != static_cast<std::size_t>(entries) ||
        state.node_velocity.size() != static_cast<std::size_t>(entries) ||
        state.x.size() != static_cast<std::size_t>(entries))
        throw std::runtime_error("solid PD device Newton increment state mismatch");
    constexpr int block_size = 128;
    const int blocks = (entries + block_size - 1) / block_size;
    commitNewtonIncrementKernel<<<blocks, block_size>>>(
        thrust::raw_pointer_cast(state.newton_velocity_increment.data()),
        thrust::raw_pointer_cast(state.base_velocity.data()),
        thrust::raw_pointer_cast(state.x.data()),
        thrust::raw_pointer_cast(state.node_velocity.data()), entries);
    checkCuda(cudaGetLastError(), "solid PD Newton increment commit");
}

void setPdNewtonLineSearchTrial3D(PersistentPdOperator3D& context,
    double alpha)
{
    DeviceState& state = stateFor(context);
    const int entries = static_cast<int>(state.node_velocity.size());
    if (entries == 0 || state.base_velocity.size() !=
            static_cast<std::size_t>(entries) ||
        state.newton_velocity_increment.size() !=
            static_cast<std::size_t>(entries) ||
        state.x.size() != static_cast<std::size_t>(entries))
        throw std::runtime_error("solid PD device line-search state mismatch");
    constexpr int block_size = 128;
    const int blocks = (entries + block_size - 1) / block_size;
    setNewtonLineSearchTrialKernel<<<blocks, block_size>>>(
        thrust::raw_pointer_cast(state.base_velocity.data()),
        thrust::raw_pointer_cast(state.newton_velocity_increment.data()),
        thrust::raw_pointer_cast(state.x.data()), alpha,
        thrust::raw_pointer_cast(state.node_velocity.data()), entries);
    checkCuda(cudaGetLastError(), "solid PD line-search trial launch");
}

void commitPdNewtonLineSearchTrial3D(PersistentPdOperator3D& context,
    double alpha)
{
    DeviceState& state = stateFor(context);
    const int entries =
        static_cast<int>(state.newton_velocity_increment.size());
    if (entries == 0 || state.x.size() != static_cast<std::size_t>(entries))
        throw std::runtime_error("solid PD line-search commit state mismatch");
    constexpr int block_size = 128;
    const int blocks = (entries + block_size - 1) / block_size;
    commitScaledNewtonIncrementKernel<<<blocks, block_size>>>(
        thrust::raw_pointer_cast(state.newton_velocity_increment.data()),
        thrust::raw_pointer_cast(state.x.data()), alpha, entries);
    checkCuda(cudaGetLastError(), "solid PD line-search commit launch");
}

double computePdNewtonTotalEnergy3D(PersistentPdOperator3D& context,
    const GPU_STAGE3A::Vec3d& gravity, double dt, double step_scale)
{
    DeviceState& state = stateFor(context);
    const int entries =
        static_cast<int>(state.newton_velocity_increment.size());
    if (!(dt > 0.0) || entries == 0 || state.mass.size() * 3 !=
            static_cast<std::size_t>(entries) ||
        (step_scale != 0.0 && state.x.size() !=
            static_cast<std::size_t>(entries)))
        throw std::runtime_error("solid PD device total-energy state mismatch");
    const double* step = step_scale == 0.0 ? nullptr
        : thrust::raw_pointer_cast(state.x.data());
    NewtonGridEnergyTransform transform{
        thrust::raw_pointer_cast(state.mass.data()),
        thrust::raw_pointer_cast(state.newton_velocity_increment.data()),
        step, step_scale, dt, {gravity.x, gravity.y, gravity.z}};
    const double grid_energy = thrust::transform_reduce(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(entries), transform, 0.0,
        thrust::plus<double>());
    const double force_energy = thrust::reduce(
        state.trial_energy.begin(), state.trial_energy.end(), 0.0,
        thrust::plus<double>());
    return force_energy + grid_energy;
}

void applyCorotatedMatrixFree3D(PersistentPdOperator3D& context,
    const CorotatedOperatorInput& input, const std::vector<double>& x,
    std::vector<double>& y, PdOperatorStats* stats)
{
    const auto total_begin = std::chrono::high_resolution_clock::now();
    if (!(input.dx > 0.0) || x.size() != input.nodes.size() * 3)
        throw std::runtime_error("solid Corotated operator input mismatch");
    DeviceState& state = stateFor(context);
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    uploadCorotatedInput(state, input);
    state.x.assign(x.begin(), x.end());
    state.y.resize(x.size());
    checkCuda(cudaDeviceSynchronize(), "solid Corotated operator upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();
    const auto apply_begin = std::chrono::high_resolution_clock::now();
    applyCorotatedDevice(state, input, thrust::raw_pointer_cast(state.x.data()),
        thrust::raw_pointer_cast(state.y.data()));
    checkCuda(cudaDeviceSynchronize(), "solid Corotated operator kernel synchronize");
    const auto apply_end = std::chrono::high_resolution_clock::now();
    const auto download_begin = std::chrono::high_resolution_clock::now();
    y.resize(x.size());
    thrust::copy(state.y.begin(), state.y.end(), y.begin());
    checkCuda(cudaDeviceSynchronize(), "solid Corotated operator download synchronize");
    const auto download_end = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->nodes = input.nodes.size();
        stats->particles = input.particles.size();
        stats->upload_ms = std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
        stats->apply_ms = std::chrono::duration<double, std::milli>(apply_end - apply_begin).count();
        stats->download_ms = std::chrono::duration<double, std::milli>(download_end - download_begin).count();
        stats->total_ms = std::chrono::duration<double, std::milli>(download_end - total_begin).count();
    }
}

static void solveCorotatedPcgInternal(PersistentPdOperator3D& context,
    const CorotatedOperatorInput& input,
    const std::vector<double>* host_rhs, double relative_tolerance,
    double absolute_tolerance, int max_iterations,
    std::vector<double>* host_solution, PdPcgStats* stats)
{
    if (!(input.dx > 0.0))
        throw std::runtime_error("solid Corotated PCG input mismatch");
    DeviceState& state = stateFor(context);
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    uploadCorotatedInput(state, input);
    const int node_count = static_cast<int>(state.mass.size());
    const int entries = host_rhs ? static_cast<int>(host_rhs->size())
                                 : static_cast<int>(state.r.size());
    if (entries != 3 * node_count)
        throw std::runtime_error("solid Corotated PCG vector size mismatch");
    state.x.assign(entries, 0.0);
    if (host_rhs)
        state.r.assign(host_rhs->begin(), host_rhs->end());
    state.z.resize(entries);
    state.p.resize(entries);
    state.q.resize(entries);
    constexpr int block_size = 128;
    const int blocks = (entries + block_size - 1) / block_size;
    projectDevice(state, node_count,
        thrust::raw_pointer_cast(state.r.data()));
    jacobiKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.mass.data()),
        thrust::raw_pointer_cast(state.r.data()), entries,
        thrust::raw_pointer_cast(state.z.data()));
    projectDevice(state, node_count,
        thrust::raw_pointer_cast(state.z.data()));
    thrust::copy(state.z.begin(), state.z.end(), state.p.begin());
    checkCuda(cudaGetLastError(), "solid Corotated PCG initialization");
    checkCuda(cudaDeviceSynchronize(), "solid Corotated PCG upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();
    double rz = thrust::inner_product(state.r.begin(), state.r.end(), state.z.begin(), 0.0);
    const double initial = std::sqrt(std::max(0.0, rz));
    const double tolerance = std::min(relative_tolerance * initial, absolute_tolerance);
    int iteration = 0;
    const auto solve_begin = std::chrono::high_resolution_clock::now();
    for (; iteration < max_iterations && std::sqrt(std::max(0.0, rz)) > tolerance;
         ++iteration) {
        applyCorotatedDevice(state, input, thrust::raw_pointer_cast(state.p.data()),
            thrust::raw_pointer_cast(state.q.data()));
        projectDevice(state, node_count,
            thrust::raw_pointer_cast(state.q.data()));
        const double pq = thrust::inner_product(
            state.p.begin(), state.p.end(), state.q.begin(), 0.0);
        if (!(pq > 0.0) || !std::isfinite(pq))
            throw std::runtime_error("solid Corotated PCG lost SPD direction");
        const double alpha = rz / pq;
        axpyKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.x.data()),
            thrust::raw_pointer_cast(state.p.data()), alpha, entries);
        residualKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.r.data()),
            thrust::raw_pointer_cast(state.q.data()), alpha, entries);
        projectDevice(state, node_count,
            thrust::raw_pointer_cast(state.r.data()));
        jacobiKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.mass.data()),
            thrust::raw_pointer_cast(state.r.data()), entries,
            thrust::raw_pointer_cast(state.z.data()));
        projectDevice(state, node_count,
            thrust::raw_pointer_cast(state.z.data()));
        const double new_rz = thrust::inner_product(
            state.r.begin(), state.r.end(), state.z.begin(), 0.0);
        directionKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.p.data()),
            thrust::raw_pointer_cast(state.z.data()), new_rz / rz, entries);
        projectDevice(state, node_count,
            thrust::raw_pointer_cast(state.p.data()));
        rz = new_rz;
    }
    checkCuda(cudaGetLastError(), "solid Corotated PCG iteration kernel");
    checkCuda(cudaDeviceSynchronize(), "solid Corotated PCG solve synchronize");
    const auto solve_end = std::chrono::high_resolution_clock::now();
    const auto download_begin = std::chrono::high_resolution_clock::now();
    if (host_solution) {
        host_solution->resize(static_cast<std::size_t>(entries));
        thrust::copy(state.x.begin(), state.x.end(), host_solution->begin());
        checkCuda(cudaDeviceSynchronize(),
            "solid Corotated PCG download synchronize");
    }
    const auto download_end = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->iterations = iteration;
        stats->initial_preconditioned_norm = initial;
        stats->final_preconditioned_norm = std::sqrt(std::max(0.0, rz));
        stats->upload_ms = std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
        stats->solve_ms = std::chrono::duration<double, std::milli>(solve_end - solve_begin).count();
        stats->download_ms = std::chrono::duration<double, std::milli>(download_end - download_begin).count();
    }
}

void solveCorotatedPcg3D(PersistentPdOperator3D& context,
    const CorotatedOperatorInput& input, const std::vector<double>& rhs,
    double relative_tolerance, double absolute_tolerance, int max_iterations,
    std::vector<double>& solution, PdPcgStats* stats)
{
    solveCorotatedPcgInternal(context, input, &rhs, relative_tolerance,
        absolute_tolerance, max_iterations, &solution, stats);
}

void solveCorotatedPcgDeviceResidual3D(PersistentPdOperator3D& context,
    const CorotatedOperatorInput& input, double relative_tolerance,
    double absolute_tolerance, int max_iterations, PdPcgStats* stats)
{
    solveCorotatedPcgInternal(context, input, nullptr, relative_tolerance,
        absolute_tolerance, max_iterations, nullptr, stats);
}

void solveLinearCorotatedPcg3D(PersistentPdOperator3D& context,
    const LinearCorotatedOperatorInput& input, const std::vector<double>& rhs,
    double relative_tolerance, double absolute_tolerance, int max_iterations,
    std::vector<double>& solution, PdPcgStats* stats)
{
    if (!(input.dx > 0.0) || rhs.size() != input.nodes.size() * 3)
        throw std::runtime_error("solid linear-corotated PCG input mismatch");
    const auto begin = std::chrono::high_resolution_clock::now();
    DeviceState& state = stateFor(context);
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    uploadLinearInput(state, input);
    const int entries = static_cast<int>(rhs.size());
    state.x.assign(entries, 0.0);
    state.r.assign(rhs.begin(), rhs.end());
    state.z.resize(entries);
    state.p.resize(entries);
    state.q.resize(entries);
    constexpr int block_size = 128;
    const int blocks = (entries + block_size - 1) / block_size;
    projectDevice(state, static_cast<int>(input.nodes.size()),
        thrust::raw_pointer_cast(state.r.data()));
    jacobiKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.mass.data()),
        thrust::raw_pointer_cast(state.r.data()), entries,
        thrust::raw_pointer_cast(state.z.data()));
    projectDevice(state, static_cast<int>(input.nodes.size()),
        thrust::raw_pointer_cast(state.z.data()));
    thrust::copy(state.z.begin(), state.z.end(), state.p.begin());
    checkCuda(cudaGetLastError(), "solid linear-corotated PCG initialization");
    checkCuda(cudaDeviceSynchronize(), "solid linear-corotated PCG upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    double rz = thrust::inner_product(state.r.begin(), state.r.end(), state.z.begin(), 0.0);
    const double initial = std::sqrt(std::max(0.0, rz));
    const double tolerance = std::min(relative_tolerance * initial, absolute_tolerance);
    int iteration = 0;
    const auto solve_begin = std::chrono::high_resolution_clock::now();
    for (; iteration < max_iterations && std::sqrt(std::max(0.0, rz)) > tolerance;
         ++iteration) {
        applyLinearCorotatedDevice(state, input,
            thrust::raw_pointer_cast(state.p.data()), thrust::raw_pointer_cast(state.q.data()));
        projectDevice(state, static_cast<int>(input.nodes.size()),
            thrust::raw_pointer_cast(state.q.data()));
        const double pq = thrust::inner_product(state.p.begin(), state.p.end(), state.q.begin(), 0.0);
        if (!(pq > 0.0) || !std::isfinite(pq))
            throw std::runtime_error("solid linear-corotated PCG lost SPD direction");
        const double alpha = rz / pq;
        axpyKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.x.data()),
            thrust::raw_pointer_cast(state.p.data()), alpha, entries);
        residualKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.r.data()),
            thrust::raw_pointer_cast(state.q.data()), alpha, entries);
        projectDevice(state, static_cast<int>(input.nodes.size()),
            thrust::raw_pointer_cast(state.r.data()));
        jacobiKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.mass.data()),
            thrust::raw_pointer_cast(state.r.data()), entries,
            thrust::raw_pointer_cast(state.z.data()));
        projectDevice(state, static_cast<int>(input.nodes.size()),
            thrust::raw_pointer_cast(state.z.data()));
        const double new_rz = thrust::inner_product(
            state.r.begin(), state.r.end(), state.z.begin(), 0.0);
        directionKernel<<<blocks, block_size>>>(thrust::raw_pointer_cast(state.p.data()),
            thrust::raw_pointer_cast(state.z.data()), new_rz / rz, entries);
        projectDevice(state, static_cast<int>(input.nodes.size()),
            thrust::raw_pointer_cast(state.p.data()));
        rz = new_rz;
    }
    checkCuda(cudaGetLastError(), "solid linear-corotated PCG iteration kernel");
    checkCuda(cudaDeviceSynchronize(), "solid linear-corotated PCG solve synchronize");
    const auto solve_end = std::chrono::high_resolution_clock::now();
    const auto download_begin = std::chrono::high_resolution_clock::now();
    solution.resize(rhs.size());
    thrust::copy(state.x.begin(), state.x.end(), solution.begin());
    checkCuda(cudaDeviceSynchronize(), "solid linear-corotated PCG download synchronize");
    const auto download_end = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->iterations = iteration;
        stats->initial_preconditioned_norm = initial;
        stats->final_preconditioned_norm = std::sqrt(std::max(0.0, rz));
        stats->upload_ms = std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
        stats->solve_ms = std::chrono::duration<double, std::milli>(solve_end - solve_begin).count();
        stats->download_ms = std::chrono::duration<double, std::milli>(download_end - download_begin).count();
    }
}

} // namespace ZIRAN::GPU_SOLID_IMPLICIT
