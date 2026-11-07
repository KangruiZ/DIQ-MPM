#include "GpuSparseGridProbe.h"
#include "GpuOperatorProbe.h"
#include "GpuParticleLevelSet.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/set_operations.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#ifdef ENABLE_AMGX
#include <amgx_c.h>
#endif

namespace ZIRAN::GPU_STAGE2 {
namespace {

constexpr int kCoordBits = 21;
constexpr int kCoordBias = 1 << (kCoordBits - 1);
constexpr GridKey kCoordMask = (GridKey{1} << kCoordBits) - 1;
constexpr GridKey kInvalidGridKey = ~GridKey{0};

struct IsInvalidVirtualPage {
    __host__ __device__ bool operator()(
        const thrust::tuple<GridKey, std::uint64_t>& entry) const
    {
        return thrust::get<0>(entry) == kInvalidGridKey;
    }
};

struct IsPositive {
    __host__ __device__ bool operator()(double value) const
    {
        return value > 0.0;
    }
};

// MpmGrid::getNumNodes() assigns a velocity DOF only when the initial P2G
// mass is non-zero.  Q1 samples that lie exactly on a grid plane have some
// mathematically valid support keys with exactly zero weight; emitting those
// keys into the compact GPU field creates extra velocity rows which SPGrid
// never has.  Keep the test on the contribution mass rather than the key so
// the remaining reduction preserves the original per-key sample order.
struct HasZeroMomentumContributionMass {
    template <class Tuple>
    __host__ __device__ bool operator()(const Tuple& entry) const
    {
        return thrust::get<1>(entry) == 0.0;
    }
};

void checkCuda(cudaError_t error, const char* label)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(error));
}

__host__ __device__ int floorToIntDevice(double value)
{
    return static_cast<int>(floor(value));
}

__host__ __device__ int floorDivideDevice(int numerator, int denominator)
{
    int quotient = numerator / denominator;
    if (numerator % denominator && numerator < 0)
        --quotient;
    return quotient;
}

__host__ __device__ GridKey packGridKey3DUnchecked(int x, int y, int z)
{
    const int bx = x + kCoordBias;
    const int by = y + kCoordBias;
    const int bz = z + kCoordBias;
    return (static_cast<GridKey>(bx) << (2 * kCoordBits)) |
        (static_cast<GridKey>(by) << kCoordBits) |
        static_cast<GridKey>(bz);
}

__host__ __device__ void unpackGridKey3DUnchecked(GridKey key,
    int& x, int& y, int& z)
{
    x = static_cast<int>((key >> (2 * kCoordBits)) & kCoordMask) - kCoordBias;
    y = static_cast<int>((key >> kCoordBits) & kCoordMask) - kCoordBias;
    z = static_cast<int>(key & kCoordMask) - kCoordBias;
}

__device__ int binarySearchKey(const GridKey* keys, int count, GridKey key)
{
    int first = 0;
    int last = count;
    while (first < last) {
        const int mid = first + ((last - first) >> 1);
        if (keys[mid] < key)
            first = mid + 1;
        else
            last = mid;
    }
    return first;
}

struct IsActivePressureDof {
    const GridKey* sorted_source_keys = nullptr;
    int source_count = 0;

    __device__ bool operator()(
        const thrust::tuple<GridKey, unsigned char>& entry) const
    {
        if (thrust::get<1>(entry))
            return false;
        const GridKey key = thrust::get<0>(entry);
        const int found = binarySearchKey(sorted_source_keys, source_count, key);
        return found < source_count && sorted_source_keys[found] == key;
    }
};

__device__ int binarySearchAssemblyNode(
    const GPU_STAGE3A::AssemblyGridNode* nodes, int count, GridKey key)
{
    int first = 0;
    int last = count;
    while (first < last) {
        const int mid = first + ((last - first) >> 1);
        if (nodes[mid].key < key)
            first = mid + 1;
        else
            last = mid;
    }
    return first;
}

__device__ int upperBoundKey(const GridKey* keys, int count, GridKey key)
{
    int first = 0;
    int last = count;
    while (first < last) {
        const int mid = first + ((last - first) >> 1);
        if (!(key < keys[mid]))
            first = mid + 1;
        else
            last = mid;
    }
    return first;
}

__host__ __device__ GridKey order0KeyUnchecked(const Vec3d& x, double dx, double anchor)
{
    return packGridKey3DUnchecked(
        floorToIntDevice((x.x + anchor) / dx),
        floorToIntDevice((x.y + anchor) / dx),
        floorToIntDevice((x.z + anchor) / dx));
}

__global__ void generateLinearVelocityKeysKernel(
    const Vec3d* xq, int count, double dx, int halfDxAnchor, GridKey* keys)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count)
        return;

    const double anchor = halfDxAnchor ? -0.5 * dx : 0.0;
    const Vec3d x = xq[q];
    const int base_x = floorToIntDevice((x.x + anchor) / dx);
    const int base_y = floorToIntDevice((x.y + anchor) / dx);
    const int base_z = floorToIntDevice((x.z + anchor) / dx);
    const int out = q * 8;

    int n = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                keys[out + n++] = packGridKey3DUnchecked(base_x + i, base_y + j, base_z + k);
}

__global__ void generatePressureKeysKernel(
    const Vec3d* xq, int count, double dx, int halfDxAnchor, GridKey* keys)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count)
        return;

    const double anchor = halfDxAnchor ? -0.5 * dx : 0.0;
    keys[q] = order0KeyUnchecked(xq[q], dx, anchor);
}

// Exactly mirrors MpmGrid::activatePages(): a non-repeated adjacent particle base
// activates its own page first, then visits the full 3x3x3 neighbourhood in
// i/j/k loop order. Sorting stably by page key preserves this stream order
// within equal keys, so the first item of every group is the old Set_Page
// first-activation rank.
__global__ void generateVirtualBlockCandidatesKernel(
    const GridKey* source_base_keys, int source_count,
    int block_x, int block_y, int block_z,
    GridKey* page_keys, std::uint64_t* ranks)
{
    const int source = blockIdx.x * blockDim.x + threadIdx.x;
    if (source >= source_count)
        return;

    constexpr int candidates_per_source = 28;
    const int output = source * candidates_per_source;
    const GridKey base_key = source_base_keys[source];
    const bool repeated_adjacent = source > 0
        && base_key == source_base_keys[source - 1];
    if (repeated_adjacent) {
        for (int slot = 0; slot < candidates_per_source; ++slot) {
            page_keys[output + slot] = kInvalidGridKey;
            ranks[output + slot] = ~std::uint64_t{0};
        }
        return;
    }

    int x, y, z;
    unpackGridKey3DUnchecked(base_key, x, y, z);
    const int page_x = floorDivideDevice(x, block_x);
    const int page_y = floorDivideDevice(y, block_y);
    const int page_z = floorDivideDevice(z, block_z);
    const std::uint64_t source_rank = static_cast<std::uint64_t>(source)
        * candidates_per_source;
    page_keys[output] = packGridKey3DUnchecked(page_x, page_y, page_z);
    ranks[output] = source_rank;
    int slot = 1;
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j)
            for (int k = -1; k <= 1; ++k, ++slot) {
                page_keys[output + slot] = packGridKey3DUnchecked(
                    page_x + i, page_y + j, page_z + k);
                ranks[output + slot] = source_rank + static_cast<std::uint64_t>(slot);
            }
}

// Generalized form of the virtual-page stream above. Fluid Q0 pressure uses
// [0,1]^3 while the boundary-face grid uses [-1,1]^3. Slot zero is the
// explicit Set_Page(base) call which precedes MpmGrid::activatePages's nested loop.
__global__ void generateFluidDofPageCandidatesKernel(
    const GridKey* source_base_keys, int source_count,
    int block_x, int block_y, int block_z, int page_start, int page_end,
    int candidates_per_source, GridKey* page_keys, std::uint64_t* ranks)
{
    const int source = blockIdx.x * blockDim.x + threadIdx.x;
    if (source >= source_count)
        return;
    const int output = source * candidates_per_source;
    const GridKey base_key = source_base_keys[source];
    const bool repeated_adjacent = source > 0 &&
        base_key == source_base_keys[source - 1];
    if (repeated_adjacent) {
        for (int slot = 0; slot < candidates_per_source; ++slot) {
            page_keys[output + slot] = kInvalidGridKey;
            ranks[output + slot] = ~std::uint64_t{0};
        }
        return;
    }
    int x, y, z;
    unpackGridKey3DUnchecked(base_key, x, y, z);
    const int page_x = floorDivideDevice(x, block_x);
    const int page_y = floorDivideDevice(y, block_y);
    const int page_z = floorDivideDevice(z, block_z);
    const std::uint64_t source_rank = static_cast<std::uint64_t>(source) *
        static_cast<std::uint64_t>(candidates_per_source);
    page_keys[output] = packGridKey3DUnchecked(page_x, page_y, page_z);
    ranks[output] = source_rank;
    int slot = 1;
    for (int i = page_start; i <= page_end; ++i)
        for (int j = page_start; j <= page_end; ++j)
            for (int k = page_start; k <= page_end; ++k, ++slot) {
                page_keys[output + slot] = packGridKey3DUnchecked(
                    page_x + i, page_y + j, page_z + k);
                ranks[output + slot] = source_rank +
                    static_cast<std::uint64_t>(slot);
            }
}

__global__ void buildFluidLegacyNodeOrderKernel(const GridKey* node_keys,
    int node_count, const GridKey* sorted_page_keys,
    const int* sorted_page_ranks, int page_count,
    int block_x, int block_y, int block_z, std::uint64_t* order,
    GPU_STAGE3A::AssemblyGridNode* nodes, int* missing_pages)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count)
        return;
    int x, y, z;
    unpackGridKey3DUnchecked(node_keys[node], x, y, z);
    const int page_x = floorDivideDevice(x, block_x);
    const int page_y = floorDivideDevice(y, block_y);
    const int page_z = floorDivideDevice(z, block_z);
    const GridKey page_key = packGridKey3DUnchecked(page_x, page_y, page_z);
    const int found = binarySearchKey(sorted_page_keys, page_count, page_key);
    if (found >= page_count || sorted_page_keys[found] != page_key) {
        atomicAdd(missing_pages, 1);
        order[node] = ~std::uint64_t{0};
    } else {
        const int local_x = x - page_x * block_x;
        const int local_y = y - page_y * block_y;
        const int local_z = z - page_z * block_z;
        const int local_rank =
            (local_x * block_y + local_y) * block_z + local_z;
        order[node] = (static_cast<std::uint64_t>(sorted_page_ranks[found])
                          << 32) |
            static_cast<std::uint32_t>(local_rank);
    }
    GPU_STAGE3A::AssemblyGridNode assembled;
    assembled.key = node_keys[node];
    assembled.idx = -1;
    assembled.mass = 1.0;
    nodes[node] = assembled;
}

__global__ void scatterFluidLegacyNodeRanksKernel(const int* ordered_ids,
    int node_count, GPU_STAGE3A::AssemblyGridNode* nodes)
{
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank < node_count)
        nodes[ordered_ids[rank]].idx = rank;
}

__host__ __device__ std::uint64_t spgridBitSpread7(
    int coordinate, std::uint64_t mask)
{
    // Exact scalar equivalent of SPGrid::Bit_Spread<mask>(scoord_t). Signed
    // coordinates are sign-extended before bits are spread into the mask.
    std::uint64_t source = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(coordinate));
    std::uint64_t result = 0;
    for (std::uint64_t bit = 1; bit != 0; bit <<= 1) {
        if (mask & bit)
            result |= source & bit;
        else
            source <<= 1;
    }
    return result;
}

__host__ __device__ std::uint64_t spgridLinearOffset7(
    int x, int y, int z)
{
    // GridState<double,3> is 128 bytes (log2_struct/log2_field = 7), with a
    // 4 KiB page. These are SPGrid_Mask<7,7,3,12>'s aggregate masks.
    constexpr std::uint64_t xmask = 0x9249249249249800ULL;
    constexpr std::uint64_t ymask = 0x4924924924924600ULL;
    constexpr std::uint64_t zmask = 0x2492492492492180ULL;
    return spgridBitSpread7(x, xmask) |
        spgridBitSpread7(y, ymask) | spgridBitSpread7(z, zmask);
}

__global__ void buildFluidLegacyVelocityNodesKernel(const GridKey* keys,
    const double* mass, const Vec3d* velocity, int node_count,
    std::uint64_t* order, GPU_STAGE3A::AssemblyGridNode* nodes)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count)
        return;
    int x, y, z;
    unpackGridKey3DUnchecked(keys[node], x, y, z);
    order[node] = spgridLinearOffset7(x, y, z);
    GPU_STAGE3A::AssemblyGridNode assembled;
    assembled.key = keys[node];
    assembled.idx = -1;
    assembled.mass = mass[node];
    assembled.new_v = GPU_STAGE3A::Vec3d{
        velocity[node].x, velocity[node].y, velocity[node].z};
    nodes[node] = assembled;
}

__global__ void buildSolidLegacyPressureNodesKernel(const GridKey* keys,
    const double* mass, const double* compliance, const double* pressure,
    int node_count, std::uint64_t* order,
    GPU_STAGE3A::AssemblyGridNode* nodes)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count)
        return;
    int x, y, z;
    unpackGridKey3DUnchecked(keys[node], x, y, z);
    order[node] = spgridLinearOffset7(x, y, z);
    GPU_STAGE3A::AssemblyGridNode assembled;
    assembled.key = keys[node];
    assembled.idx = -1;
    assembled.mass = mass[node];
    assembled.v = GPU_STAGE3A::Vec3d{
        compliance[node], pressure[node], 0.0};
    nodes[node] = assembled;
}

__global__ void expandVirtualBlockKeysKernel(const GridKey* page_keys,
    int page_count, int block_x, int block_y, int block_z,
    GridKey* touched_keys)
{
    const int block_nodes = block_x * block_y * block_z;
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = page_count * block_nodes;
    if (flat >= total)
        return;
    const int page = flat / block_nodes;
    const int local = flat - page * block_nodes;
    const int i = local / (block_y * block_z);
    const int remainder = local - i * block_y * block_z;
    const int j = remainder / block_z;
    const int k = remainder - j * block_z;
    int page_x, page_y, page_z;
    unpackGridKey3DUnchecked(page_keys[page], page_x, page_y, page_z);
    touched_keys[flat] = packGridKey3DUnchecked(
        page_x * block_x + i, page_y * block_y + j, page_z * block_z + k);
}

__global__ void generateFaceKeysKernel(
    const Vec3d* face_xq, const Vec3d* face_n, int count, double dx,
    int halfDxAnchor, GridKey* keys)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count)
        return;

    const double anchor = halfDxAnchor ? -0.5 * dx : 0.0;
    const Vec3d x = face_xq[q];
    const Vec3d n = face_n[q];
    for (int delta_index = 0; delta_index < 2; ++delta_index) {
        const double delta = delta_index == 0 ? -1.0 : 1.0;
        const Vec3d sample{
            x.x + n.x * 0.5 * dx * delta,
            x.y + n.y * 0.5 * dx * delta,
            x.z + n.z * 0.5 * dx * delta};
        keys[q * 2 + delta_index] = order0KeyUnchecked(sample, dx, anchor);
    }
}

__global__ void scatterUnitCountsKernel(
    const GridKey* sample_keys, int sample_count,
    const GridKey* active_keys, int active_count, double* grid_count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sample_count)
        return;

    const GridKey key = sample_keys[i];
    const int dof = binarySearchKey(active_keys, active_count, key);
    if (dof < active_count && active_keys[dof] == key)
        atomicAdd(&grid_count[dof], 1.0);
}

__global__ void scatterLinearVelocityMassKernel(
    const Vec3d* xq, const double* q_mass, int count, double dx, int halfDxAnchor,
    const GridKey* active_keys, int active_count, double* grid_mass)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count)
        return;

    const double anchor = halfDxAnchor ? -0.5 * dx : 0.0;
    const Vec3d x = xq[q];
    int base[3];
    double w[3][2];

    const double gx = (x.x + anchor) / dx;
    const double gy = (x.y + anchor) / dx;
    const double gz = (x.z + anchor) / dx;
    base[0] = floorToIntDevice(gx);
    base[1] = floorToIntDevice(gy);
    base[2] = floorToIntDevice(gz);
    w[0][1] = gx - static_cast<double>(base[0]);
    w[1][1] = gy - static_cast<double>(base[1]);
    w[2][1] = gz - static_cast<double>(base[2]);
    w[0][0] = 1.0 - w[0][1];
    w[1][0] = 1.0 - w[1][1];
    w[2][0] = 1.0 - w[2][1];

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) {
                const GridKey key = packGridKey3DUnchecked(base[0] + i, base[1] + j, base[2] + k);
                const int dof = binarySearchKey(active_keys, active_count, key);
                if (dof < active_count && active_keys[dof] == key)
                    atomicAdd(&grid_mass[dof], q_mass[q] * w[0][i] * w[1][j] * w[2][k]);
    }
}

__global__ void scatterUniformLinearScalarKernel(
    const Vec3d* xq, int count, double value, double dx, int halfDxAnchor,
    const GridKey* active_keys, int active_count, double* grid_mass)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count)
        return;

    const double anchor = halfDxAnchor ? -0.5 * dx : 0.0;
    const Vec3d x = xq[q];
    const double gx = (x.x + anchor) / dx;
    const double gy = (x.y + anchor) / dx;
    const double gz = (x.z + anchor) / dx;
    const int base_x = floorToIntDevice(gx);
    const int base_y = floorToIntDevice(gy);
    const int base_z = floorToIntDevice(gz);
    const double wx[2] = {1.0 - (gx - base_x), gx - base_x};
    const double wy[2] = {1.0 - (gy - base_y), gy - base_y};
    const double wz[2] = {1.0 - (gz - base_z), gz - base_z};

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) {
                const GridKey key = packGridKey3DUnchecked(
                    base_x + i, base_y + j, base_z + k);
                const int node = binarySearchKey(active_keys, active_count, key);
                if (node < active_count && active_keys[node] == key)
                    atomicAdd(grid_mass + node, value * wx[i] * wy[j] * wz[k]);
            }
}

struct PersistentLinearScatterStorage {
    thrust::device_vector<Vec3d> positions;
    thrust::device_vector<GridKey> candidate_keys;
    thrust::device_vector<GridKey> active_keys;
    thrust::device_vector<double> grid_mass;
    std::size_t reallocations = 0;
};

struct PersistentVelocityTransferStorage {
    thrust::device_vector<Vec3d> particle_x;
    thrust::device_vector<GridKey> node_keys;
    thrust::device_vector<int> node_indices;
    thrust::device_vector<GridKey> g2p_fallback_keys;
    thrust::device_vector<Vec3d> g2p_fallback_velocity;
    thrust::device_vector<GridKey> pressure_keys;
    thrust::device_vector<int> pressure_indices;
    thrust::device_vector<Vec3d> output_x;
    thrust::device_vector<Vec3d> output_v;
    thrust::device_vector<Mat3d> output_grad_v;
    thrust::device_vector<double> output_pressure;
    thrust::device_vector<int> patch_indices;
    thrust::device_vector<Vec3d> patch_velocity;
    thrust::device_vector<Mat3d> patch_grad_v;
    thrust::device_vector<Vec3d> patch_x;
    thrust::device_vector<unsigned long long> boundary_projection_count;
    Vec3d* host_x = nullptr;
    Vec3d* host_v = nullptr;
    Mat3d* host_grad_v = nullptr;
    double* host_pressure = nullptr;
    std::size_t host_capacity = 0;
    std::size_t reallocations = 0;

    ~PersistentVelocityTransferStorage()
    {
        if (host_x) cudaFreeHost(host_x);
        if (host_v) cudaFreeHost(host_v);
        if (host_grad_v) cudaFreeHost(host_grad_v);
        if (host_pressure) cudaFreeHost(host_pressure);
    }
};

__global__ void patchFluidVelocityTransferParticlesKernel(
    const int* particle_indices, const Vec3d* velocity,
    const Mat3d* grad_velocity, const Vec3d* position, int patch_count,
    int particle_count, Vec3d* output_velocity, Mat3d* output_grad_velocity,
    Vec3d* output_position)
{
    const int patch = blockIdx.x * blockDim.x + threadIdx.x;
    if (patch >= patch_count)
        return;
    const int particle = particle_indices[patch];
    if (particle < 0 || particle >= particle_count)
        return;
    output_velocity[particle] = velocity[patch];
    if (grad_velocity)
        output_grad_velocity[particle] = grad_velocity[patch];
    if (position)
        output_position[particle] = position[patch];
}

__global__ void transferFluidVelocityQ1Kernel(const Vec3d* particle_x,
    int particle_count, const GridKey* node_keys, const int* node_indices,
    int node_count, const GPU_STAGE3A::AssemblyGridNode* resident_velocity_nodes,
    int resident_velocity_count, const GridKey* g2p_fallback_keys,
    const Vec3d* g2p_fallback_velocity, int g2p_fallback_count,
    const double* velocity, const GridKey* pressure_keys,
    const int* pressure_indices, int pressure_count,
    const GPU_STAGE3A::AssemblyGridNode* resident_pressure_nodes,
    int resident_pressure_count, const double* pressure,
    int pressure_value_count,
    double dx, double dt, Vec3d* output_x, Vec3d* output_v,
    Mat3d* output_grad_v, double* output_pressure)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    const Vec3d x = particle_x[p];
    const double gx = (x.x - 0.5 * dx) / dx;
    const double gy = (x.y - 0.5 * dx) / dx;
    const double gz = (x.z - 0.5 * dx) / dx;
    const int bx = floorToIntDevice(gx), by = floorToIntDevice(gy), bz = floorToIntDevice(gz);
    const double wx[2] = {1.0 - (gx - bx), gx - bx};
    const double wy[2] = {1.0 - (gy - by), gy - by};
    const double wz[2] = {1.0 - (gz - bz), gz - bz};
    Vec3d pic{0.0, 0.0, 0.0};
    Mat3d grad{};
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) for (int k = 0; k < 2; ++k) {
        const GridKey key = packGridKey3DUnchecked(bx + i, by + j, bz + k);
        Vec3d v;
        int velocity_index = -1;
        if (resident_velocity_nodes) {
            const int found = binarySearchAssemblyNode(
                resident_velocity_nodes, resident_velocity_count, key);
            if (found < resident_velocity_count &&
                resident_velocity_nodes[found].key == key)
                velocity_index = resident_velocity_nodes[found].idx;
        } else {
            const int found = binarySearchKey(node_keys, node_count, key);
            if (found < node_count && node_keys[found] == key)
                velocity_index = node_indices[found];
        }
        if (velocity_index >= 0) {
            const int row = velocity_index * 3;
            v = Vec3d{velocity[row], velocity[row + 1], velocity[row + 2]};
        } else {
            // CPU's final mass rebuild sets idx=-1 but deliberately leaves
            // `new_v` intact.  These initial-only nodes do not belong to the
            // pressure system, yet Q1 G2P still samples them.
            const int fallback = binarySearchKey(g2p_fallback_keys,
                g2p_fallback_count, key);
            if (fallback >= g2p_fallback_count ||
                g2p_fallback_keys[fallback] != key)
                continue;
            v = g2p_fallback_velocity[fallback];
        }
        const double weight = wx[i] * wy[j] * wz[k];
        pic.x += weight * v.x; pic.y += weight * v.y; pic.z += weight * v.z;
        const double d[3] = {(i ? 1.0 : -1.0) * wy[j] * wz[k] / dx,
            wx[i] * (j ? 1.0 : -1.0) * wz[k] / dx,
            wx[i] * wy[j] * (k ? 1.0 : -1.0) / dx};
        const double vv[3] = {v.x, v.y, v.z};
        for (int col = 0; col < 3; ++col) for (int row_i = 0; row_i < 3; ++row_i)
            grad.values[row_i + 3 * col] += vv[row_i] * d[col];
    }
    output_x[p] = {x.x + dt * pic.x, x.y + dt * pic.y, x.z + dt * pic.z};
    output_v[p] = pic;
    output_grad_v[p] = grad;
    const GridKey pressure_key = packGridKey3DUnchecked(bx, by, bz);
    int pressure_index = -1;
    if (resident_pressure_nodes) {
        const int found = binarySearchAssemblyNode(
            resident_pressure_nodes, resident_pressure_count, pressure_key);
        if (found < resident_pressure_count &&
            resident_pressure_nodes[found].key == pressure_key)
            pressure_index = resident_pressure_nodes[found].idx;
    } else {
        const int found = binarySearchKey(
            pressure_keys, pressure_count, pressure_key);
        if (found < pressure_count && pressure_keys[found] == pressure_key)
            pressure_index = pressure_indices[found];
    }
    output_pressure[p] = pressure_index >= 0 &&
            pressure_index < pressure_value_count
        ? pressure[pressure_index] : 0.0;
}

#ifdef ENABLE_AMGX
__device__ bool pressureSolidAtKey(const GridKey* sorted_solid_keys,
    const unsigned char* sorted_solid_flags, int solid_count, GridKey key)
{
    const int found = binarySearchKey(sorted_solid_keys, solid_count, key);
    // The CPU compatibility path falls back to a level-set query outside the
    // cached virtual blocks. The production virtual-block expansion contains
    // every pressure face queried below; treat a miss conservatively as solid
    // rather than manufacturing a face velocity.
    return found >= solid_count || sorted_solid_keys[found] != key
        || sorted_solid_flags[found] != 0;
}

__device__ Vec3d vec3ValueAtKey(const GridKey* sorted_keys,
    const Vec3d* values, int count, GridKey key)
{
    const int found = binarySearchKey(sorted_keys, count, key);
    return found < count && sorted_keys[found] == key
        ? values[found] : Vec3d{};
}

// Forward declaration: defined below (near pushoutFaceVelocityGatherKernel),
// which this kernel also needs for the same reason — see its own comment.
__device__ bool pushoutIsSolid(const GridKey* sorted_solid_keys,
    const unsigned char* sorted_solid_flags, int solid_count, int nx, int ny,
    int nz, const BoxShellCollision3D& collision,
    const GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D& level_set,
    double dx);

__global__ void applyPressurePushoutRhsKernel(
    const GridKey* ordered_dof_keys, int dof_count,
    const GridKey* sorted_solid_keys,
    const unsigned char* sorted_solid_flags, int solid_count,
    const GridKey* displacement_keys, const Vec3d* displacement,
    int displacement_count, double coefficient,
    BoxShellCollision3D collision,
    GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D level_set, double dx,
    double* rhs)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= dof_count)
        return;
    int x, y, z;
    unpackGridKey3DUnchecked(ordered_dof_keys[row], x, y, z);
    const Vec3d center = vec3ValueAtKey(displacement_keys, displacement,
        displacement_count, ordered_dof_keys[row]);
    double correction = 0.0;
    for (int d = 0; d < 3; ++d) {
        const int lx = x - (d == 0), ly = y - (d == 1), lz = z - (d == 2);
        // isSolid(), not the "miss=solid" pressureSolidAtKey convention: a
        // touched-key miss here is a real geometry query on CPU
        // (computeIsSolid()), same as pushoutFaceVelocityGatherKernel.
        const bool lower_solid = pushoutIsSolid(sorted_solid_keys,
            sorted_solid_flags, solid_count, lx, ly, lz, collision, level_set,
            dx);
        if (lower_solid)
            correction -= (d == 0 ? center.x : (d == 1 ? center.y : center.z));
        const int ux = x + (d == 0), uy = y + (d == 1), uz = z + (d == 2);
        const bool upper_solid = pushoutIsSolid(sorted_solid_keys,
            sorted_solid_flags, solid_count, ux, uy, uz, collision, level_set,
            dx);
        Vec3d upper{};
        if (upper_solid) {
            const GridKey upper_key = packGridKey3DUnchecked(ux, uy, uz);
            upper = vec3ValueAtKey(displacement_keys,
                displacement, displacement_count, upper_key);
            correction += d == 0 ? upper.x : (d == 1 ? upper.y : upper.z);
        }
    }
    rhs[row] += coefficient * correction;
}

__global__ void initializePressureFaceVelocityKernel(
    const GridKey* touched_keys, int touched_count,
    const GridKey* face_keys, const Vec3d* face_velocity, int face_count,
    Vec3d* velocity)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < touched_count)
        velocity[index] = vec3ValueAtKey(face_keys, face_velocity,
            face_count, touched_keys[index]);
}

__global__ void buildPressureFaceVelocityKernel(const GridKey* touched_keys,
    int touched_count, const GridKey* sorted_dof_keys,
    const int* sorted_dof_indices, int dof_count,
    const GridKey* sorted_solid_keys, const unsigned char* sorted_solid_flags,
    int solid_count, const double* pressure, double coefficient,
    double clamp_value, Vec3d* velocity)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= touched_count)
        return;
    int x, y, z;
    unpackGridKey3DUnchecked(touched_keys[index], x, y, z);
    const GridKey center_key = touched_keys[index];
    const bool center_solid = pressureSolidAtKey(sorted_solid_keys,
        sorted_solid_flags, solid_count, center_key);
    const int center_found = binarySearchKey(
        sorted_dof_keys, dof_count, center_key);
    const bool center_active = center_found < dof_count
        && sorted_dof_keys[center_found] == center_key;
    Vec3d initial = velocity[index];
    double value[3] = {initial.x, initial.y, initial.z};
    for (int d = 0; d < 3; ++d) {
        const GridKey lower_key = packGridKey3DUnchecked(
            x - (d == 0), y - (d == 1), z - (d == 2));
        const int lower_found = binarySearchKey(
            sorted_dof_keys, dof_count, lower_key);
        const bool lower_active = lower_found < dof_count
            && sorted_dof_keys[lower_found] == lower_key;
        // A pressure DOF on either side overwrites this face. If neither side
        // is active, or either side is solid, preserve the pushout seed.
        if (!center_solid && !pressureSolidAtKey(sorted_solid_keys,
                sorted_solid_flags, solid_count, lower_key)
            && (center_active || lower_active)) {
            const double center_value = center_active
                ? pressure[sorted_dof_indices[center_found]] : 0.0;
            const double lower_p = lower_active
                ? pressure[sorted_dof_indices[lower_found]] : 0.0;
            double face_v = coefficient * (center_value - lower_p);
            face_v = fmax(-clamp_value, fmin(clamp_value, face_v));
            value[d] = face_v;
        }
    }
    velocity[index] = Vec3d{value[0], value[1], value[2]};
}

__global__ void correctPressureParticlesKernel(const Vec3d* particle_x,
    int particle_count, const GridKey* touched_keys, int touched_count,
    const Vec3d* face_velocity, double dx, Vec3d* output_x)
{
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= particle_count)
        return;
    const Vec3d x = particle_x[particle];
    double corrected[3] = {x.x, x.y, x.z};
    for (int d = 0; d < 3; ++d) {
        // Match the CPU loop exactly: after component d is corrected, the
        // next component's face stencil observes that updated particle
        // position rather than the original Xp.
        double face[3] = {corrected[0] - dx, corrected[1] - dx,
            corrected[2] - dx};
        face[d] += 0.5 * dx;
        const int base_x = floorToIntDevice(face[0] / dx);
        const int base_y = floorToIntDevice(face[1] / dx);
        const int base_z = floorToIntDevice(face[2] / dx);
        const double wx[2] = {1.0 - (face[0] / dx - base_x),
            face[0] / dx - base_x};
        const double wy[2] = {1.0 - (face[1] / dx - base_y),
            face[1] / dx - base_y};
        const double wz[2] = {1.0 - (face[2] / dx - base_z),
            face[2] / dx - base_z};
        double displacement = 0.0;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    const GridKey key = packGridKey3DUnchecked(
                        base_x + i, base_y + j, base_z + k);
                    const int found = binarySearchKey(touched_keys,
                        touched_count, key);
                    if (found < touched_count && touched_keys[found] == key) {
                        const Vec3d v = face_velocity[found];
                        displacement += wx[i] * wy[j] * wz[k]
                            * (d == 0 ? v.x : (d == 1 ? v.y : v.z));
                    }
                }
        corrected[d] += displacement;
    }
    output_x[particle] = Vec3d{corrected[0], corrected[1], corrected[2]};
}
#endif

// Per-particle candidate emitted by the GPU pushout's particle-projection
// kernel; only entries flagged valid by `pushout_candidate_valid` matter.
// Declared ahead of PersistentPressureGridStorage, which caches a compacted
// buffer of these across substeps.
struct PushoutCandidate {
    GridKey key;
    double sq_disp;
    Vec3d disp;
    // Carried through compaction (but not the node-level max-displacement
    // reduction) so the corrected velocity can be sparsely synced back to
    // the CPU-authoritative particles.V array for every hit particle, not
    // just the per-node reduction winner.
    Vec3d corrected_v;
    int particle_id;
    int debug_source; // Bitmask of which check(s) triggered this hit: 1=box-shell, 2=particle level set, 3=both. Useful when narrowing a future divergence to one of the two geometric predicates.
};

struct PersistentPressureCsrStorage {
    thrust::device_vector<GridKey> row_keys;
    thrust::device_vector<GridKey> sorted_dof_keys;
    thrust::device_vector<int> sorted_dof_indices;
    thrust::device_vector<GridKey> sorted_solid_keys;
    thrust::device_vector<unsigned char> sorted_solid_flags;
    thrust::device_vector<double> rho_star;
    thrust::device_vector<int> entry_columns;
    thrust::device_vector<double> entry_values;
    thrust::device_vector<int> entry_counts;
    thrust::device_vector<double> rhs;
    thrust::device_vector<int> missing_solid;
    std::size_t reallocations = 0;
};

// One persistent compact pressure-field storage. The page stream and density
// field intentionally have separate source arrays, but they share ownership,
// lifetime and the direct topology fields needed by PositionCorrector.
struct PersistentPressureGridStorage {
    PersistentLinearScatterStorage pages;
    const Vec3d* page_source_positions = nullptr;
    std::size_t page_source_count = 0;
    PersistentLinearScatterStorage density;
    thrust::device_vector<Vec3d> density_static_positions;
    const Vec3d* density_static_host_data = nullptr;
    std::size_t density_static_count = 0;
    thrust::device_vector<GridKey> dof_source_keys;
    thrust::device_vector<GridKey> virtual_page_candidates;
    thrust::device_vector<std::uint64_t> virtual_page_ranks;
    thrust::device_vector<std::uint64_t> virtual_page_reduced_ranks;
    thrust::device_vector<GridKey> virtual_page_keys;
    thrust::device_vector<GridKey> virtual_touched_keys;
    thrust::device_vector<GridKey> ordered_dof_keys;
    thrust::device_vector<GridKey> touched_keys;
    thrust::device_vector<unsigned char> solid_flags;
    thrust::device_vector<GridKey> sorted_dof_keys;
    thrust::device_vector<int> sorted_dof_indices;
    thrust::device_vector<GridKey> sorted_solid_keys;
    thrust::device_vector<unsigned char> sorted_solid_flags;
    // Legacy `solid_grid.iterateGridSerial` visitation rank of each
    // `sorted_solid_keys` entry (i.e. its index within the *unsorted*
    // `touched_keys`, which already replicates that traversal order). Only
    // populated by the GPU pushout path, to deterministically resolve the
    // CPU "double write" tie in `pushoutPressureVelocity`.
    thrust::device_vector<int> sorted_solid_rank;
    thrust::device_vector<double> rho_star;
    thrust::device_vector<int> entry_columns;
    thrust::device_vector<double> entry_values;
    thrust::device_vector<int> entry_counts;
    thrust::device_vector<double> rhs;
    thrust::device_vector<int> missing_solid;
#ifdef ENABLE_AMGX
    // Connected-component metadata for pressure islands. The topology is
    // derived from the fixed-width device CSR; only enclosed components have
    // their RHS mean removed before AMGX, matching the legacy host policy.
    thrust::device_vector<int> row_touches_air;
    thrust::device_vector<int> island_parent;
    thrust::device_vector<int> island_touches_air;
    thrust::device_vector<double> island_rhs_sum;
    thrust::device_vector<int> island_row_count;
    std::size_t enclosed_island_count = 0;
    // AMGX owns a compact CSR. These buffers compact the seven-entry stencil
    // in device memory, avoiding the legacy CSR D2H solely for solver input.
    thrust::device_vector<int> amgx_row_offsets;
    thrust::device_vector<int> amgx_column_indices;
    thrust::device_vector<double> amgx_values;
    thrust::device_vector<double> amgx_solution;
    thrust::device_vector<GridKey> pushout_displacement_keys;
    thrust::device_vector<Vec3d> pushout_displacement;
    thrust::device_vector<GridKey> pushout_face_keys;
    thrust::device_vector<Vec3d> pushout_face_velocity;
    bool pushout_face_seed_valid = false;
    // GPU pushout scratch: one entry per fluid particle, compacted down to
    // the (typically tiny) set of particles actually displaced this
    // substep.
    thrust::device_vector<PushoutCandidate> pushout_candidates;
    thrust::device_vector<unsigned char> pushout_candidate_valid;
    thrust::device_vector<PushoutCandidate> pushout_candidates_compact;
    // Post-solve correction keeps its keyed velocity field and particle
    // staging alongside the AMGX pressure solution. This avoids routing the
    // solution through direct_pressure_new_v on the CPU.
    thrust::device_vector<Vec3d> pressure_velocity;
    thrust::device_vector<Vec3d> correction_particle_x;
    thrust::device_vector<Vec3d> correction_output_x;
    Vec3d* correction_host_x = nullptr;
    std::size_t correction_host_capacity = 0;
#endif
    std::size_t reallocations = 0;

    ~PersistentPressureGridStorage()
    {
#ifdef ENABLE_AMGX
        if (correction_host_x)
            cudaFreeHost(correction_host_x);
#endif
    }
};

template <class Vector>
void resizePressureCsr(Vector& values, std::size_t size,
    PersistentPressureCsrStorage& storage)
{
    const std::size_t old_capacity = values.capacity();
    values.resize(size);
    if (values.capacity() != old_capacity)
        ++storage.reallocations;
}

template <class Vector>
void resizeLinearScatter(Vector& values, std::size_t size,
    PersistentLinearScatterStorage& storage)
{
    const std::size_t old_capacity = values.capacity();
    values.resize(size);
    if (values.capacity() != old_capacity)
        ++storage.reallocations;
}

PersistentLinearScatterStorage& getLinearScatterStorage(
    PersistentLinearScatter3D& context)
{
    if (!context.storage)
        context.storage = std::make_shared<PersistentLinearScatterStorage>();
    return *std::static_pointer_cast<PersistentLinearScatterStorage>(context.storage);
}

PersistentLinearScatterStorage& getPagePolluteStorage(
    PersistentPagePollute3D& context)
{
    if (!context.storage)
        context.storage = std::make_shared<PersistentLinearScatterStorage>();
    return *std::static_pointer_cast<PersistentLinearScatterStorage>(context.storage);
}

PersistentVelocityTransferStorage& getVelocityTransferStorage(
    PersistentVelocityTransfer3D& context)
{
    if (!context.storage)
        context.storage = std::make_shared<PersistentVelocityTransferStorage>();
    return *std::static_pointer_cast<PersistentVelocityTransferStorage>(context.storage);
}

template <class Vector>
void resizeVelocityTransfer(Vector& values, std::size_t size,
    PersistentVelocityTransferStorage& storage)
{
    const std::size_t old_capacity = values.capacity();
    values.resize(size);
    if (values.capacity() != old_capacity)
        ++storage.reallocations;
}

void ensureVelocityTransferHostOutput(PersistentVelocityTransferStorage& storage,
    std::size_t size)
{
    if (size <= storage.host_capacity)
        return;
    if (storage.host_x) cudaFreeHost(storage.host_x);
    if (storage.host_v) cudaFreeHost(storage.host_v);
    if (storage.host_grad_v) cudaFreeHost(storage.host_grad_v);
    if (storage.host_pressure) cudaFreeHost(storage.host_pressure);
    storage.host_x = nullptr;
    storage.host_v = nullptr;
    storage.host_grad_v = nullptr;
    storage.host_pressure = nullptr;
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&storage.host_x),
                  size * sizeof(Vec3d), cudaHostAllocDefault),
        "velocity transfer pinned X allocation");
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&storage.host_v),
                  size * sizeof(Vec3d), cudaHostAllocDefault),
        "velocity transfer pinned V allocation");
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&storage.host_grad_v),
                  size * sizeof(Mat3d), cudaHostAllocDefault),
        "velocity transfer pinned grad allocation");
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&storage.host_pressure),
                  size * sizeof(double), cudaHostAllocDefault),
        "velocity transfer pinned pressure allocation");
    storage.host_capacity = size;
    ++storage.reallocations;
}

std::size_t downloadVelocityTransferHostOutput(
    PersistentVelocityTransferStorage& storage, std::size_t particle_count,
    VelocityTransferDownloadMask mask, VelocityTransferHostView& result,
    const VelocityTransferDeviceView* device_view = nullptr)
{
    result = {};
    result.size = particle_count;
    if (!mask.x && !mask.v && !mask.grad_v && !mask.pressure)
        return 0;
    ensureVelocityTransferHostOutput(storage, particle_count);
    const void* device_x = device_view
        ? device_view->x
        : static_cast<const void*>(
              thrust::raw_pointer_cast(storage.output_x.data()));
    const void* device_v = device_view
        ? device_view->v
        : static_cast<const void*>(
              thrust::raw_pointer_cast(storage.output_v.data()));
    const void* device_grad_v = device_view
        ? device_view->grad_v
        : static_cast<const void*>(
              thrust::raw_pointer_cast(storage.output_grad_v.data()));
    const void* device_pressure = device_view
        ? device_view->pressure
        : static_cast<const void*>(
              thrust::raw_pointer_cast(storage.output_pressure.data()));
    std::size_t bytes = 0;
    if (mask.x) {
        if (!device_x)
            throw std::runtime_error("velocity transfer X device view is null");
        checkCuda(cudaMemcpy(storage.host_x,
                      device_x,
                      particle_count * sizeof(Vec3d), cudaMemcpyDeviceToHost),
            "velocity transfer X pinned download");
        result.x = storage.host_x;
        bytes += particle_count * sizeof(Vec3d);
    }
    if (mask.v) {
        if (!device_v)
            throw std::runtime_error("velocity transfer V device view is null");
        checkCuda(cudaMemcpy(storage.host_v,
                      device_v,
                      particle_count * sizeof(Vec3d), cudaMemcpyDeviceToHost),
            "velocity transfer V pinned download");
        result.v = storage.host_v;
        bytes += particle_count * sizeof(Vec3d);
    }
    if (mask.grad_v) {
        if (!device_grad_v)
            throw std::runtime_error(
                "velocity transfer gradV device view is null");
        checkCuda(cudaMemcpy(storage.host_grad_v,
                      device_grad_v,
                      particle_count * sizeof(Mat3d), cudaMemcpyDeviceToHost),
            "velocity transfer grad pinned download");
        result.grad_v = storage.host_grad_v;
        bytes += particle_count * sizeof(Mat3d);
    }
    if (mask.pressure) {
        if (!device_pressure)
            throw std::runtime_error(
                "velocity transfer pressure device view is null");
        checkCuda(cudaMemcpy(storage.host_pressure,
                      device_pressure,
                      particle_count * sizeof(double), cudaMemcpyDeviceToHost),
            "velocity transfer pressure pinned download");
        result.pressure = storage.host_pressure;
        bytes += particle_count * sizeof(double);
    }
    return bytes;
}

PersistentPressureCsrStorage& getPressureCsrStorage(
    PersistentPressureCsr3D& context)
{
    if (!context.storage)
        context.storage = std::make_shared<PersistentPressureCsrStorage>();
    return *std::static_pointer_cast<PersistentPressureCsrStorage>(context.storage);
}

PersistentPressureGridStorage& getPressureGridStorage(
    PersistentPressureGrid3D& context)
{
    if (!context.storage)
        context.storage = std::make_shared<PersistentPressureGridStorage>();
    return *std::static_pointer_cast<PersistentPressureGridStorage>(context.storage);
}

template <class Vector>
void resizePressureGrid(Vector& values, std::size_t size,
    PersistentPressureGridStorage& storage)
{
    const std::size_t old_capacity = values.capacity();
    values.resize(size);
    if (values.capacity() != old_capacity)
        ++storage.reallocations;
}

std::size_t pressureGridReallocations(const PersistentPressureGridStorage& storage)
{
    return storage.reallocations + storage.pages.reallocations
        + storage.density.reallocations;
}

#ifdef ENABLE_AMGX
void checkAmgx(AMGX_RC result, const char* label)
{
    if (result == AMGX_RC_OK)
        return;
    char message[4096] = {};
    AMGX_get_error_string(result, message, sizeof(message));
    throw std::runtime_error(std::string(label) + ": " + message);
}

struct AmgxRuntime {
    AMGX_config_handle config = nullptr;
    AMGX_resources_handle resources = nullptr;
};

struct PersistentAmgxCsrStorage {
    thrust::device_vector<double> values;
    thrust::device_vector<double> rhs;
    thrust::device_vector<double> solution;
};

PersistentAmgxCsrStorage& getAmgxCsrStorage(PersistentAmgxCsrSolver& context)
{
    if (!context.storage)
        context.storage = std::make_shared<PersistentAmgxCsrStorage>();
    return *std::static_pointer_cast<PersistentAmgxCsrStorage>(context.storage);
}

int amgxEnvInt(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (!value || !*value)
        return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0)
        return fallback;
    return static_cast<int>(parsed);
}

double amgxEnvDouble(const char* name, double fallback)
{
    const char* value = std::getenv(name);
    if (!value || !*value)
        return fallback;
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || !(parsed > 0.0))
        return fallback;
    return parsed;
}

void initializeAmgxLibrary()
{
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        checkAmgx(AMGX_initialize(), "AMGX_initialize");
    });
}

AmgxRuntime& getAmgxRuntime()
{
    static AmgxRuntime runtime;
    static std::once_flag initialized;
    std::call_once(initialized, [&] {
        initializeAmgxLibrary();
        const double solver_tolerance = amgxEnvDouble(
            "ZIRAN_AMGX_TOLERANCE", 1e-3);
        const int solver_max_iters = amgxEnvInt(
            "ZIRAN_AMGX_MAX_ITERS", 30);
        const char* config_variant_env = std::getenv("ZIRAN_AMGX_CONFIG");
        const bool use_aggr_cheb4 = config_variant_env
            && std::strcmp(config_variant_env, "aggr_cheb4") == 0;
        // This is the closest AMGX equivalent of the existing AMGCL CUDA
        // setup: double PCG and aggregation AMG. The default tolerance and
        // iteration cap intentionally mirror the existing PositionCorrector
        // solver; environment overrides are for validation sweeps only.
        std::ostringstream config;
        if (use_aggr_cheb4) {
            config << R"json(
{
  "config_version": 2,
  "determinism_flag": 1,
  "solver": {
    "scope": "main",
    "solver": "AMG",
    "algorithm": "AGGREGATION",
    "selector": "SIZE_8",
    "smoother": {
      "solver": "CHEBYSHEV",
      "preconditioner": {
        "solver": "JACOBI_L1",
        "max_iters": 1
      },
      "max_iters": 1,
      "chebyshev_polynomial_order": 4,
      "chebyshev_lambda_estimate_mode": 2
    },
    "presweeps": 0,
    "postsweeps": 1,
    "coarsest_sweeps": 1,
    "coarse_solver": "NOSOLVER",
    "max_iters": )json"
                   << solver_max_iters
                   << R"json(,
    "monitor_residual": 0,
    "print_solve_stats": 0,
    "obtain_timings": 0,
    "min_coarse_rows": 2,
    "max_levels": 50,
    "convergence": "RELATIVE_INI",
    "tolerance": )json"
                   << solver_tolerance
                   << R"json(,
    "norm": "L2",
    "cycle": "V"
  }
}
)json";
        } else {
            config << R"json(
{
  "config_version": 2,
  "determinism_flag": 1,
  "solver": {
    "scope": "main",
    "solver": "PCG",
    "tolerance": )json"
               << solver_tolerance
               << R"json(,
    "max_iters": )json"
               << solver_max_iters
               << R"json(,
    "monitor_residual": 0,
    "print_solve_stats": 0,
    "obtain_timings": 0,
    "convergence": "RELATIVE_INI",
    "norm": "L2",
    "preconditioner": {
      "scope": "amg",
      "solver": "AMG",
      "algorithm": "AGGREGATION",
      "selector": "SIZE_2",
      "interpolator": "D2",
      "max_levels": 6,
      "max_iters": 1,
      "cycle": "V",
      "presweeps": 0,
      "postsweeps": 3,
      "coarse_solver": "NOSOLVER",
      "smoother": {
        "scope": "jacobi",
        "solver": "BLOCK_JACOBI",
        "max_iters": 1,
        "relaxation_factor": 0.8,
        "monitor_residual": 0,
        "print_solve_stats": 0
      }
    }
  }
}
)json";
        }
        const std::string config_string = config.str();
        checkAmgx(AMGX_config_create(&runtime.config, config_string.c_str()),
            "AMGX_config_create");
        checkAmgx(AMGX_resources_create_simple(&runtime.resources,
                      runtime.config),
            "AMGX_resources_create_simple");
    });
    return runtime;
}

AmgxRuntime& getCouplingAmgxRuntime()
{
    static AmgxRuntime runtime;
    static std::once_flag initialized;
    std::call_once(initialized, [&] {
        initializeAmgxLibrary();
        const double solver_tolerance = amgxEnvDouble(
            "ZIRAN_AMGX_COUPLING_TOLERANCE", 1e-3);
        const int solver_max_iters = amgxEnvInt(
            "ZIRAN_AMGX_COUPLING_MAX_ITERS", 30);
        // Keep the established PCG_AGGREGATION_JACOBI-style configuration as
        // the production default.  The two opt-in variants are deliberately
        // symmetric V/W cycles: paired pre/post smoothing, plus an exact
        // coarse solve.  They are for determining whether the old 0/3,
        // NOSOLVER cycle explains a discrepancy with the CPU AMGCL result.
        //
        // This setting is read while the AMGX singleton is constructed, so a
        // separate process is required for each A/B variant.
        const char* preconditioner_variant_env =
            std::getenv("ZIRAN_AMGX_COUPLING_PRECONDITIONER");
        const char* preconditioner_variant = preconditioner_variant_env
            ? preconditioner_variant_env : "legacy";
        int presweeps = 0;
        int postsweeps = 3;
        const char* cycle = "W";
        const char* coarse_solver = "NOSOLVER";
        if (std::strcmp(preconditioner_variant, "legacy") == 0) {
            // Existing default: AMGX's PCG aggregation/Jacobi template,
            // with the Chebyshev/Jacobi-L1 smoother used by this project.
        } else if (std::strcmp(preconditioner_variant, "symmetric_v") == 0) {
            presweeps = 1;
            postsweeps = 1;
            cycle = "V";
            coarse_solver = "DENSE_LU_SOLVER";
        } else if (std::strcmp(preconditioner_variant, "symmetric_w") == 0) {
            presweeps = 1;
            postsweeps = 1;
            cycle = "W";
            coarse_solver = "DENSE_LU_SOLVER";
        } else {
            throw std::runtime_error(
                "Unknown ZIRAN_AMGX_COUPLING_PRECONDITIONER value: " +
                std::string(preconditioner_variant) +
                " (expected legacy, symmetric_v, or symmetric_w)");
        }
        std::ostringstream config;
        // Closest available AMGX mapping of amgclCudaParams(): PCG with an
        // aggregation hierarchy, Chebyshev(5)/Jacobi-L1 relaxation, six
        // levels and a two-cycle (W) AMG preconditioner. AMGX does not expose
        // AMGCL's smoothed-aggregation implementation.  The diagnostic
        // variants above change only the cycle symmetry/coarse solve; the
        // matrix, RHS, diagonal regularization and outer PCG stay identical.
        config << R"json(
{
  "config_version": 2,
  "determinism_flag": 1,
  "solver": {
    "scope": "main",
    "solver": "PCG",
    "tolerance": )json"
               << solver_tolerance
               << R"json(,
    "max_iters": )json"
               << solver_max_iters
               << R"json(,
    "monitor_residual": 0,
    "print_solve_stats": 0,
    "obtain_timings": 0,
    "convergence": "RELATIVE_INI",
    "norm": "L2",
    "preconditioner": {
      "scope": "amg",
      "solver": "AMG",
      "algorithm": "AGGREGATION",
      "selector": "SIZE_2",
      "max_levels": 6,
      "max_iters": 1,
      "cycle": ")json"
               << cycle
               << R"json(",
      "presweeps": )json"
               << presweeps
               << R"json(,
      "postsweeps": )json"
               << postsweeps
               << R"json(,
      "coarse_solver": ")json"
               << coarse_solver
               << R"json(",
      "smoother": {
        "solver": "CHEBYSHEV",
        "preconditioner": {
          "solver": "JACOBI_L1",
          "max_iters": 1
        },
        "max_iters": 1,
        "chebyshev_polynomial_order": 5,
        "chebyshev_lambda_estimate_mode": 2
      }
    }
  }
}
)json";
        const std::string config_string = config.str();
        checkAmgx(AMGX_config_create(&runtime.config, config_string.c_str()),
            "AMGX coupling config create");
        checkAmgx(AMGX_resources_create_simple(&runtime.resources, runtime.config),
            "AMGX coupling resources create");
    });
    return runtime;
}

void destroyAmgxSolveHandles(AMGX_solver_handle solver, AMGX_vector_handle x,
    AMGX_vector_handle rhs, AMGX_matrix_handle matrix)
{
    if (solver)
        AMGX_solver_destroy(solver);
    if (x)
        AMGX_vector_destroy(x);
    if (rhs)
        AMGX_vector_destroy(rhs);
    if (matrix)
        AMGX_matrix_destroy(matrix);
}
#endif

__device__ int lookupSolid(const GridKey* sorted_keys,
    const unsigned char* sorted_flags, int count, GridKey key,
    int* missing_solid)
{
    const int index = binarySearchKey(sorted_keys, count, key);
    if (index >= count || sorted_keys[index] != key) {
        atomicExch(missing_solid, 1);
        return 0;
    }
    return sorted_flags[index] != 0;
}

__global__ void gatherPressureDensityKernel(const GridKey* row_keys, int rows,
    const GridKey* density_keys, const double* density_values,
    int density_count, double* rho_star)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    int x, y, z;
    unpackGridKey3DUnchecked(row_keys[row], x, y, z);
    const GridKey density_key = packGridKey3DUnchecked(x + 1, y + 1, z + 1);
    const int index = binarySearchKey(density_keys, density_count, density_key);
    rho_star[row] = (index < density_count && density_keys[index] == density_key)
        ? density_values[index]
        : 0.0;
}

__global__ void assemblePressureCsrKernel(const GridKey* row_keys, int rows,
    const GridKey* sorted_dof_keys, const int* sorted_dof_indices,
    const GridKey* sorted_solid_keys, const unsigned char* sorted_solid_flags,
    int solid_count, const double* rho_star_by_row, double rho_rest,
    double dx, double dt, int* entry_columns, double* entry_values,
    int* entry_counts, double* rhs, int* missing_solid,
    int* row_touches_air)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    int x, y, z;
    unpackGridKey3DUnchecked(row_keys[row], x, y, z);
    bool near_air = false;
    bool touches_air = false;
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j)
            for (int k = -1; k <= 1; ++k) {
                const GridKey key = packGridKey3DUnchecked(x + i, y + j, z + k);
                const bool solid = lookupSolid(sorted_solid_keys, sorted_solid_flags,
                    solid_count, key, missing_solid);
                const int found = binarySearchKey(sorted_dof_keys, rows, key);
                if (!solid && (found >= rows || sorted_dof_keys[found] != key))
                    near_air = true;
            }
    int diagonal = 0;
    int count = 1;
    const int base = row * 7;
    entry_columns[base] = row;
    for (int axis = 0; axis < 3; ++axis)
        for (int delta_index = 0; delta_index < 2; ++delta_index) {
            const int delta = delta_index ? 1 : -1;
            int nx = x, ny = y, nz = z;
            if (axis == 0) nx += delta;
            else if (axis == 1) ny += delta;
            else nz += delta;
            const GridKey key = packGridKey3DUnchecked(nx, ny, nz);
            if (!lookupSolid(sorted_solid_keys, sorted_solid_flags,
                    solid_count, key, missing_solid)) {
                ++diagonal;
                const int found = binarySearchKey(sorted_dof_keys, rows, key);
                if (found >= rows || sorted_dof_keys[found] != key) {
                    near_air = true;
                    touches_air = true;
                } else {
                    entry_columns[base + count] = sorted_dof_indices[found];
                    entry_values[base + count] = -1.0;
                    ++count;
                }
            }
        }
    entry_values[base] = static_cast<double>(diagonal);
    entry_counts[row] = count;
    if (row_touches_air)
        row_touches_air[row] = touches_air ? 1 : 0;
    double rho_star = rho_star_by_row[row];
    if (near_air)
        rho_star = fmax(rho_star, rho_rest);
    rho_star = fmax(rho_star, rho_rest * 0.5);
    rho_star = fmin(rho_star, rho_rest * 1.5);
    rhs[row] = -(rho_rest - rho_star) * dx * dx / dt / dt;
}

#ifdef ENABLE_AMGX
__device__ int pressureIslandRoot(int* parent, int node)
{
    int root = node;
    while (parent[root] != root)
        root = parent[root];
    while (parent[node] != node) {
        const int next = parent[node];
        parent[node] = root;
        node = next;
    }
    return root;
}

__device__ int pressureIslandRootNoCompress(const int* parent, int node)
{
    while (parent[node] != node)
        node = parent[node];
    return node;
}

__device__ void unitePressureIsland(int* parent, int a, int b)
{
    while (true) {
        // Do not path-compress while other threads are still linking roots.
        // Links always move from a larger row id to a smaller one, which
        // guarantees an acyclic forest. Compression happens in a separate
        // kernel after all unions have completed.
        a = pressureIslandRootNoCompress(parent, a);
        b = pressureIslandRootNoCompress(parent, b);
        if (a == b)
            return;
        const int lower = min(a, b);
        const int upper = max(a, b);
        if (atomicCAS(parent + upper, upper, lower) == upper)
            return;
    }
}

__global__ void unitePressureIslandsKernel(const int* entry_columns,
    const int* entry_counts, int rows, int* parent)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    const int base = row * 7;
    for (int entry = 1; entry < entry_counts[row]; ++entry)
        unitePressureIsland(parent, row, entry_columns[base + entry]);
}

__global__ void finalizePressureIslandsKernel(int rows, int* parent,
    const int* row_touches_air,
    int* island_touches_air)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    const int root = pressureIslandRoot(parent, row);
    parent[row] = root;
    if (row_touches_air[row])
        atomicExch(island_touches_air + root, 1);
}

struct IsEnclosedPressureIsland {
    const int* parent;
    const int* touches_air;

    __host__ __device__ bool operator()(int row) const
    {
        return parent[row] == row && touches_air[row] == 0;
    }
};

__global__ void accumulatePressureIslandRhsKernel(const double* rhs, int rows,
    const int* parent, const int* island_touches_air,
    double* island_sum, int* island_count)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    const int root = parent[row];
    if (island_touches_air[root])
        return;
    atomicAdd(island_sum + root, rhs[row]);
    atomicAdd(island_count + root, 1);
}

__global__ void projectPressureIslandRhsKernel(double* rhs, int rows,
    const int* parent, const int* island_touches_air,
    const double* island_sum, const int* island_count)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    const int root = parent[row];
    if (!island_touches_air[root] && island_count[root] > 0)
        rhs[row] -= island_sum[root] / static_cast<double>(island_count[root]);
}

__global__ void compactPressureCsrKernel(const int* entry_columns,
    const double* entry_values, const int* entry_counts,
    const int* row_offsets, int rows, double diagonal_epsilon,
    int* column_indices, double* values)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    const int source = row * 7;
    const int destination = row_offsets[row];
    const int count = entry_counts[row];
    for (int entry = 0; entry < count; ++entry) {
        const int column = entry_columns[source + entry];
        column_indices[destination + entry] = column;
        values[destination + entry] = entry_values[source + entry]
            + (column == row ? diagonal_epsilon : 0.0);
    }
}

__global__ void addDiagonalToCsrValuesKernel(const int* row_offsets,
    const int* column_indices, int rows, double diagonal_epsilon,
    double* values)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || diagonal_epsilon == 0.0)
        return;
    for (int entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry)
        if (column_indices[entry] == row) {
            values[entry] += diagonal_epsilon;
            return;
        }
}
#endif

struct PersistentSpatialSearchStorage {
    thrust::device_vector<Vec3d> positions;
    thrust::device_vector<GridKey> sorted_keys;
    thrust::device_vector<int> sorted_ids;
    thrust::device_vector<Vec3d> query_positions;
    thrust::device_vector<double> masses;
    thrust::device_vector<Vec3d> normals;
    thrust::device_vector<int> any_flag;
    double cell_size = 0.0;
    std::size_t reallocations = 0;
};

template <class Vector>
void resizeTracked(Vector& vector, std::size_t size,
    PersistentSpatialSearchStorage& storage)
{
    if (size > vector.capacity())
        ++storage.reallocations;
    vector.resize(size);
}

PersistentSpatialSearchStorage& getSpatialStorage(
    PersistentSpatialSearch3D& context)
{
    if (!context.storage)
        context.storage = std::make_shared<PersistentSpatialSearchStorage>();
    return *std::static_pointer_cast<PersistentSpatialSearchStorage>(context.storage);
}

struct PersistentInterfaceQuadStorage {
    thrust::device_vector<Vec3d> fluid_positions;
    thrust::device_vector<double> fluid_mass;
    std::vector<double> fluid_mass_host_cache;
    thrust::device_vector<Vec3d> fluid_velocity;
    thrust::device_vector<double> adjusted_fluid_mass;
    thrust::device_vector<double> solid_mass;
    thrust::device_vector<Vec3d> solid_momentum;
    thrust::device_vector<int> solid_eligible;
    thrust::device_vector<int> active_solid_ids;
    thrust::device_vector<double> active_solid_mass;
    thrust::device_vector<Vec3d> active_solid_momentum;
    thrust::device_vector<Vec3d> interface_rest_x;
    thrust::device_vector<Vec3d> interface_current_x;
    thrust::device_vector<Vec3d> interface_normal;
    thrust::device_vector<double> interface_length;
    // Interface transfer has a support radius of only 0.5 dx.  Keep a finer
    // (dx-wide) view of the same device solid positions so those queries do
    // not walk the much larger 2 dx SPH cells used by the normal estimator.
    thrust::device_vector<GridKey> fine_sorted_keys;
    thrust::device_vector<int> fine_sorted_ids;
    thrust::device_vector<GridKey> fine_cell_keys;
    thrust::device_vector<int> fine_cell_counts;
    thrust::device_vector<int> fine_cell_offsets;
    int fine_cell_count = 0;
    std::size_t reallocations = 0;
};

template <class Vector>
void resizeInterfaceQuad(Vector& vector, std::size_t size,
    PersistentInterfaceQuadStorage& storage)
{
    if (size > vector.capacity())
        ++storage.reallocations;
    vector.resize(size);
}

PersistentInterfaceQuadStorage& getInterfaceQuadStorage(
    PersistentInterfaceQuad3D& context)
{
    if (!context.storage)
        context.storage = std::make_shared<PersistentInterfaceQuadStorage>();
    return *std::static_pointer_cast<PersistentInterfaceQuadStorage>(context.storage);
}

__host__ __device__ bool boxShellCollision(const Vec3d& x,
    const BoxShellCollision3D& collision);

__device__ bool interfaceQuadParticleInSolid(const Vec3d& x, double dx,
    const BoxShellCollision3D& collision)
{
    const int bx = floorToIntDevice((x.x - 0.5 * dx) / dx);
    const int by = floorToIntDevice((x.y - 0.5 * dx) / dx);
    const int bz = floorToIntDevice((x.z - 0.5 * dx) / dx);
    return boxShellCollision(Vec3d{(bx + 1.0) * dx, (by + 1.0) * dx,
                                 (bz + 1.0) * dx}, collision);
}

__global__ void markInterfaceQuadSolidEligibleKernel(const Vec3d* positions,
    int count, double dx, BoxShellCollision3D collision, int* eligible)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        eligible[i] = interfaceQuadParticleInSolid(positions[i], dx, collision)
            ? 0 : 1;
}

__global__ void distributeInterfaceQuadAtomicKernel(const Vec3d* fluid_positions,
    const double* fluid_mass, const Vec3d* fluid_velocity, int fluid_count,
    const Vec3d* solid_positions, const GridKey* solid_cell_keys,
    const int* solid_cell_offsets, int solid_cell_count,
    const int* sorted_solid_ids, const int* solid_eligible,
    double cell_size, double dx,
    BoxShellCollision3D collision, double* adjusted_fluid_mass,
    double* solid_mass, Vec3d* solid_momentum)
{
    const int fluid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fluid >= fluid_count)
        return;

    const Vec3d x = fluid_positions[fluid];
    const double mass = fluid_mass[fluid];
    adjusted_fluid_mass[fluid] = mass;
    if (interfaceQuadParticleInSolid(x, dx, collision))
        return;

    const double radius = 0.5 * dx;
    const double radius_squared = radius * radius;
    const int cx = floorToIntDevice(x.x / cell_size);
    const int cy = floorToIntDevice(x.y / cell_size);
    const int cz = floorToIntDevice(x.z / cell_size);
    const int layers = static_cast<int>(ceil(radius / cell_size));
    int neighbour_count = 0;
    for (int di = -layers; di <= layers; ++di)
        for (int dj = -layers; dj <= layers; ++dj)
            for (int dk = -layers; dk <= layers; ++dk) {
                const GridKey key = packGridKey3DUnchecked(cx + di, cy + dj, cz + dk);
                const int cell = binarySearchKey(solid_cell_keys, solid_cell_count, key);
                if (cell >= solid_cell_count || solid_cell_keys[cell] != key)
                    continue;
                const int begin = solid_cell_offsets[cell];
                const int end = solid_cell_offsets[cell + 1];
                for (int entry = begin; entry < end; ++entry) {
                    const int solid = sorted_solid_ids[entry];
                    const Vec3d other = solid_positions[solid];
                    if (!solid_eligible[solid])
                        continue;
                    const double rx = x.x - other.x;
                    const double ry = x.y - other.y;
                    const double rz = x.z - other.z;
                    if (rx * rx + ry * ry + rz * rz < radius_squared)
                        ++neighbour_count;
                }
            }
    if (neighbour_count == 0)
        return;

    const double unit_mass = mass / static_cast<double>(neighbour_count + 1);
    adjusted_fluid_mass[fluid] = unit_mass;
    const Vec3d velocity = fluid_velocity[fluid];
    for (int di = -layers; di <= layers; ++di)
        for (int dj = -layers; dj <= layers; ++dj)
            for (int dk = -layers; dk <= layers; ++dk) {
                const GridKey key = packGridKey3DUnchecked(cx + di, cy + dj, cz + dk);
                const int cell = binarySearchKey(solid_cell_keys, solid_cell_count, key);
                if (cell >= solid_cell_count || solid_cell_keys[cell] != key)
                    continue;
                const int begin = solid_cell_offsets[cell];
                const int end = solid_cell_offsets[cell + 1];
                for (int entry = begin; entry < end; ++entry) {
                    const int solid = sorted_solid_ids[entry];
                    const Vec3d other = solid_positions[solid];
                    if (!solid_eligible[solid])
                        continue;
                    const double rx = x.x - other.x;
                    const double ry = x.y - other.y;
                    const double rz = x.z - other.z;
                    if (rx * rx + ry * ry + rz * rz >= radius_squared)
                        continue;
                    atomicAdd(&solid_mass[solid], unit_mass);
                    atomicAdd(&solid_momentum[solid].x, unit_mass * velocity.x);
                    atomicAdd(&solid_momentum[solid].y, unit_mass * velocity.y);
                    atomicAdd(&solid_momentum[solid].z, unit_mass * velocity.z);
                }
            }
}

__global__ void generateParticleKeysKernel(const Vec3d* positions, int count,
    double cell_size, GridKey* keys)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const Vec3d x = positions[i];
    keys[i] = packGridKey3DUnchecked(
        floorToIntDevice(x.x / cell_size),
        floorToIntDevice(x.y / cell_size),
        floorToIntDevice(x.z / cell_size));
}

__device__ bool pointHasNeighbor(const Vec3d& query, double radius_squared,
    double cell_size, const Vec3d* positions, const GridKey* sorted_keys,
    const int* sorted_ids, int particle_count)
{
    const int cx = floorToIntDevice(query.x / cell_size);
    const int cy = floorToIntDevice(query.y / cell_size);
    const int cz = floorToIntDevice(query.z / cell_size);
    const int layers = static_cast<int>(ceil(sqrt(radius_squared) / cell_size));
    for (int di = -layers; di <= layers; ++di)
        for (int dj = -layers; dj <= layers; ++dj)
            for (int dk = -layers; dk <= layers; ++dk) {
                const GridKey key = packGridKey3DUnchecked(cx + di, cy + dj, cz + dk);
                const int begin = binarySearchKey(sorted_keys, particle_count, key);
                const int end = upperBoundKey(sorted_keys, particle_count, key);
                for (int p = begin; p < end; ++p) {
                    const Vec3d other = positions[sorted_ids[p]];
                    const double dx = query.x - other.x;
                    const double dy = query.y - other.y;
                    const double dz = query.z - other.z;
                    const double distance_squared = dx * dx + dy * dy + dz * dz;
                    if (distance_squared < radius_squared)
                        return true;
                }
            }
    return false;
}

__global__ void anyPointWithinRadiusKernel(const Vec3d* queries, int query_count,
    double radius, double cell_size, const Vec3d* positions,
    const GridKey* sorted_keys, const int* sorted_ids, int particle_count,
    int* any_flag)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= query_count || atomicAdd(any_flag, 0) != 0)
        return;
    if (pointHasNeighbor(queries[i], radius * radius, cell_size, positions,
            sorted_keys, sorted_ids, particle_count))
        atomicExch(any_flag, 1);
}

__device__ double cubicBsplineGradientQ(double q)
{
    if (q <= 1.0)
        return (-12.0 * q + 9.0 * q * q) / 6.0;
    if (q <= 2.0) {
        const double t = q - 2.0;
        return -0.5 * t * t;
    }
    return 0.0;
}

__global__ void cubicSphNormalsKernel(const Vec3d* positions,
    const GridKey* sorted_keys, const int* sorted_ids, int particle_count,
    const double* masses, double radius, double cell_size, Vec3d* normals)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count)
        return;

    const Vec3d query = positions[i];
    const int cx = floorToIntDevice(query.x / cell_size);
    const int cy = floorToIntDevice(query.y / cell_size);
    const int cz = floorToIntDevice(query.z / cell_size);
    const int layers = static_cast<int>(ceil(radius / cell_size));
    double nx = 0.0;
    double ny = 0.0;
    double nz = 0.0;
    for (int di = -layers; di <= layers; ++di)
        for (int dj = -layers; dj <= layers; ++dj)
            for (int dk = -layers; dk <= layers; ++dk) {
                const GridKey key = packGridKey3DUnchecked(cx + di, cy + dj, cz + dk);
                const int begin = binarySearchKey(sorted_keys, particle_count, key);
                const int end = upperBoundKey(sorted_keys, particle_count, key);
                for (int p = begin; p < end; ++p) {
                    const int j = sorted_ids[p];
                    const Vec3d other = positions[j];
                    const double dx = query.x - other.x;
                    const double dy = query.y - other.y;
                    const double dz = query.z - other.z;
                    const double distance_squared = dx * dx + dy * dy + dz * dz;
                    if (distance_squared >= radius * radius || distance_squared <= 1e-16)
                        continue;
                    const double distance = sqrt(distance_squared);
                    const double scale = masses[i] *
                        cubicBsplineGradientQ(distance / radius * 2.0) /
                        (distance * radius);
                    nx += scale * dx;
                    ny += scale * dy;
                    nz += scale * dz;
                }
            }
    const double length = sqrt(nx * nx + ny * ny + nz * nz);
    if (length > 0.0)
        normals[i] = Vec3d{nx / length, ny / length, nz / length};
    else
        normals[i] = Vec3d{0.0, 0.0, 0.0};
}

struct PersistentFluidAnalyticStorage {
    thrust::device_vector<Vec3d> input_x;
    thrust::device_vector<double> input_mass;
    thrust::device_vector<GridKey> input_keys;
    thrust::device_vector<int> keep;
    thrust::device_vector<Vec3d> pruned_x;
    thrust::device_vector<double> pruned_mass;
    thrust::device_vector<GridKey> pruned_keys;
    thrust::device_vector<GridKey> active_keys;
    thrust::device_vector<int> cell_flags;
    thrust::device_vector<int> cell_first_sample;
    thrust::device_vector<int> face_cell_order;
    thrust::device_vector<int> face_cell_order_keys;
    thrust::device_vector<Vec3d> interface_x;
    thrust::device_vector<Vec3d> face_candidates;
    thrust::device_vector<Vec3d> face_normal_candidates;
    thrust::device_vector<int> face_keep;
    thrust::device_vector<Vec3d> face_x;
    thrust::device_vector<Vec3d> face_normal;
    thrust::device_vector<GridKey> face_node_keys;
    thrust::device_vector<GridKey> sorted_sample_keys;
    thrust::device_vector<int> sorted_sample_ids;
    thrust::device_vector<Vec3d> output_candidates;
    thrust::device_vector<double> output_mass_candidates;
    thrust::device_vector<int> output_keep;
    thrust::device_vector<Vec3d> output_x;
    thrust::device_vector<double> output_mass;
    thrust::device_vector<Vec3d> input_velocity;
    thrust::device_vector<Mat3d> input_affine;
    bool input_velocity_resident = false;
    bool input_affine_resident = false;
    thrust::device_vector<GridKey> velocity_candidate_keys;
    thrust::device_vector<double> velocity_contribution_mass;
    thrust::device_vector<Vec3d> velocity_contribution_momentum;
    thrust::device_vector<GridKey> velocity_keys;
    thrust::device_vector<GridKey> velocity_final_keys;
    thrust::device_vector<GridKey> velocity_g2p_fallback_keys;
    thrust::device_vector<double> velocity_mass;
    thrust::device_vector<Vec3d> velocity_momentum;
    thrust::device_vector<Vec3d> velocity_value;
    thrust::device_vector<Vec3d> velocity_final_value;
    thrust::device_vector<Vec3d> velocity_g2p_fallback_value;
    // Reused scratch for exact legacy Q0 page activation and local-node
    // traversal. Pressure and face maps are copied into Stage3A owners before
    // this workspace is reused for the next field.
    thrust::device_vector<GridKey> dof_source_base_keys;
    thrust::device_vector<GridKey> dof_page_candidates;
    thrust::device_vector<std::uint64_t> dof_page_candidate_ranks;
    thrust::device_vector<GridKey> dof_page_keys;
    thrust::device_vector<std::uint64_t> dof_page_first_ranks;
    thrust::device_vector<int> dof_page_activation_ranks;
    thrust::device_vector<std::uint64_t> dof_node_order;
    thrust::device_vector<int> dof_node_ids;
    thrust::device_vector<GPU_STAGE3A::AssemblyGridNode> dof_nodes;
    thrust::device_vector<int> dof_missing_pages;
    thrust::device_vector<double> solid_j;
    thrust::device_vector<double> solid_volume;
    thrust::device_vector<double> solid_lambda;
    thrust::device_vector<double> solid_compliance;
    thrust::device_vector<double> solid_pressure_numerator;
    std::size_t reallocations = 0;
};

template <class Vector>
void resizeFluid(Vector& vector, std::size_t size,
    PersistentFluidAnalyticStorage& storage)
{
    if (size > vector.capacity())
        ++storage.reallocations;
    vector.resize(size);
}

PersistentFluidAnalyticStorage& getFluidStorage(
    PersistentFluidAnalytic3D& context)
{
    if (!context.storage)
        context.storage = std::make_shared<PersistentFluidAnalyticStorage>();
    return *std::static_pointer_cast<PersistentFluidAnalyticStorage>(context.storage);
}

__host__ __device__ bool insideBoxInclusive(const Vec3d& x,
    const Vec3d& lower, const Vec3d& upper)
{
    return x.x >= lower.x && x.x <= upper.x &&
        x.y >= lower.y && x.y <= upper.y &&
        x.z >= lower.z && x.z <= upper.z;
}

__host__ __device__ bool boxShellCollision(const Vec3d& x,
    const BoxShellCollision3D& collision)
{
    return collision.valid &&
        insideBoxInclusive(x, collision.outer_min, collision.outer_max) &&
        !insideBoxInclusive(x, collision.inner_min, collision.inner_max) &&
        (!collision.has_hole ||
         !insideBoxInclusive(x, collision.hole_min, collision.hole_max));
}

__device__ double axisAlignedBoxSignedDistance(const Vec3d& x,
    const Vec3d& lower, const Vec3d& upper)
{
    const double center_x = 0.5 * (lower.x + upper.x);
    const double center_y = 0.5 * (lower.y + upper.y);
    const double center_z = 0.5 * (lower.z + upper.z);
    const double half_x = 0.5 * (upper.x - lower.x);
    const double half_y = 0.5 * (upper.y - lower.y);
    const double half_z = 0.5 * (upper.z - lower.z);
    const double dx = fabs(x.x - center_x) - half_x;
    const double dy = fabs(x.y - center_y) - half_y;
    const double dz = fabs(x.z - center_z) - half_z;
    const double outside_x = fmax(dx, 0.0);
    const double outside_y = fmax(dy, 0.0);
    const double outside_z = fmax(dz, 0.0);
    return fmin(fmax(dx, fmax(dy, dz)), 0.0) +
        sqrt(outside_x * outside_x + outside_y * outside_y +
            outside_z * outside_z);
}

__device__ bool boxShellInsideSignedDistance(const Vec3d& x,
    const BoxShellCollision3D& collision)
{
    if (!collision.valid)
        return false;
    const double outer = axisAlignedBoxSignedDistance(
        x, collision.outer_min, collision.outer_max);
    const double inner = axisAlignedBoxSignedDistance(
        x, collision.inner_min, collision.inner_max);
    double phi = fmax(outer, -inner);
    if (collision.has_hole) {
        const double hole = axisAlignedBoxSignedDistance(
            x, collision.hole_min, collision.hole_max);
        phi = fmax(phi, -hole);
    }
    return phi <= 0.0;
}

__device__ bool particleLevelSetInside(const Vec3d& x,
    const GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D& level_set)
{
    if (!level_set.phi || !(level_set.h > 0.0))
        return false;
    const int base_x = static_cast<int>(floor(x.x / level_set.h));
    const int base_y = static_cast<int>(floor(x.y / level_set.h));
    const int base_z = static_cast<int>(floor(x.z / level_set.h));
    const double tx = x.x / level_set.h - static_cast<double>(base_x);
    const double ty = x.y / level_set.h - static_cast<double>(base_y);
    const double tz = x.z / level_set.h - static_cast<double>(base_z);
    const double wx[2] = {1.0 - tx, tx};
    const double wy[2] = {1.0 - ty, ty};
    const double wz[2] = {1.0 - tz, tz};
    const auto* phi_values = static_cast<const double*>(level_set.phi);
    double phi = 0.0;
    for (int i = 0; i < 2; ++i) {
        const int lx = base_x + i - level_set.origin[0];
        for (int j = 0; j < 2; ++j) {
            const int ly = base_y + j - level_set.origin[1];
            for (int k = 0; k < 2; ++k) {
                const int lz = base_z + k - level_set.origin[2];
                if (lx < 0 || lx >= level_set.shape[0] ||
                    ly < 0 || ly >= level_set.shape[1] ||
                    lz < 0 || lz >= level_set.shape[2])
                    return false;
                const std::size_t index =
                    (static_cast<std::size_t>(lz) * level_set.shape[1] +
                        static_cast<std::size_t>(ly)) * level_set.shape[0] +
                    static_cast<std::size_t>(lx);
                const double value = phi_values[index];
                if (value >= DBL_MAX / 2.0)
                    return false;
                phi += wx[i] * wy[j] * wz[k] * value;
            }
        }
    }
    // ParticlesLevelSet::queryInside uses a strict comparison for this
    // reconstructed field (unlike AnalyticLevelSet's inclusive boundary).
    return phi < 0.0;
}

__global__ void classifyPressureSolidKernel(const GridKey* touched_keys,
    int touched_count,
    GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D level_set,
    BoxShellCollision3D collision, double dx, unsigned char* solid_flags)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= touched_count)
        return;
    int x, y, z;
    unpackGridKey3DUnchecked(touched_keys[index], x, y, z);
    const Vec3d position {
        (static_cast<double>(x) + 1.0) * dx,
        (static_cast<double>(y) + 1.0) * dx,
        (static_cast<double>(z) + 1.0) * dx};
    solid_flags[index] = boxShellInsideSignedDistance(position, collision) ||
            particleLevelSetInside(position, level_set)
        ? 1
        : 0;
}

// Gradient and signed-distance selection for the axis-aligned box-shell
// descriptor. These helpers are shared by the post-G2P boundary kernel and
// the AMGX-only PositionCorrector pushout below.
__device__ Vec3d axisAlignedBoxNormal(const Vec3d& x, const Vec3d& lower,
    const Vec3d& upper)
{
    const double center[3] = {0.5 * (lower.x + upper.x),
        0.5 * (lower.y + upper.y), 0.5 * (lower.z + upper.z)};
    const double half[3] = {0.5 * (upper.x - lower.x),
        0.5 * (upper.y - lower.y), 0.5 * (upper.z - lower.z)};
    const double xv[3] = {x.x, x.y, x.z};
    double d[3], sgn[3];
    for (int i = 0; i < 3; ++i) {
        sgn[i] = xv[i] - center[i] >= 0.0 ? 1.0 : -1.0;
        d[i] = fabs(xv[i] - center[i]) - half[i];
    }
    double outside[3] = {fmax(d[0], 0.0), fmax(d[1], 0.0), fmax(d[2], 0.0)};
    const double outside_norm = sqrt(outside[0] * outside[0] +
        outside[1] * outside[1] + outside[2] * outside[2]);
    if (outside_norm > 0.0) {
        return Vec3d{sgn[0] * outside[0] / outside_norm,
            sgn[1] * outside[1] / outside_norm,
            sgn[2] * outside[2] / outside_norm};
    }
    int argmax = 0;
    if (d[1] > d[argmax]) argmax = 1;
    if (d[2] > d[argmax]) argmax = 2;
    Vec3d n{0.0, 0.0, 0.0};
    if (argmax == 0) n.x = sgn[0];
    else if (argmax == 1) n.y = sgn[1];
    else n.z = sgn[2];
    return n;
}

__device__ double boxShellSignedDistanceAndNormal(const Vec3d& x,
    const BoxShellCollision3D& collision, Vec3d& normal)
{
    const double outer_sd = axisAlignedBoxSignedDistance(
        x, collision.outer_min, collision.outer_max);
    const Vec3d outer_n = axisAlignedBoxNormal(
        x, collision.outer_min, collision.outer_max);
    const double inner_sd = axisAlignedBoxSignedDistance(
        x, collision.inner_min, collision.inner_max);
    const Vec3d inner_n = axisAlignedBoxNormal(
        x, collision.inner_min, collision.inner_max);
    double sd;
    Vec3d n;
    if (-inner_sd > outer_sd) {
        sd = -inner_sd;
        n = Vec3d{-inner_n.x, -inner_n.y, -inner_n.z};
    } else {
        sd = outer_sd;
        n = outer_n;
    }
    if (collision.has_hole) {
        const double hole_sd = axisAlignedBoxSignedDistance(
            x, collision.hole_min, collision.hole_max);
        const Vec3d hole_n = axisAlignedBoxNormal(
            x, collision.hole_min, collision.hole_max);
        if (-hole_sd > sd) {
            sd = -hole_sd;
            n = Vec3d{-hole_n.x, -hole_n.y, -hole_n.z};
        }
    }
    normal = n;
    return sd;
}

__global__ void projectFluidBoundaryCollisionKernel(
    Vec3d* particle_x, Vec3d* particle_v, int particle_count,
    BoxShellCollision3D collision, double dx,
    unsigned long long* projected_count)
{
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= particle_count)
        return;
    Vec3d x = particle_x[particle];
    Vec3d normal;
    const double phi = boxShellSignedDistanceAndNormal(x, collision, normal);
    if (phi <= -3.0 * dx) {
        x.x -= (phi + 2.0 * dx) * normal.x;
        x.y -= (phi + 2.0 * dx) * normal.y;
        x.z -= (phi + 2.0 * dx) * normal.z;
        particle_x[particle] = x;
        particle_v[particle] = Vec3d{};
        atomicAdd(projected_count, 1ull);
    }
}

struct FluidParticleReduction {
    Vec3d min_corner;
    Vec3d max_corner;
    double max_speed_squared;
    unsigned long long outside_particles;
};

struct FluidParticleTransform {
    const Vec3d* x;
    const Vec3d* v;
    double domain_min;
    double domain_max;

    __device__ FluidParticleReduction operator()(std::size_t particle) const
    {
        const Vec3d xp = x[particle];
        const Vec3d vp = v[particle];
        const bool outside = xp.x < domain_min || xp.x > domain_max
            || xp.y < domain_min || xp.y > domain_max
            || xp.z < domain_min || xp.z > domain_max;
        return {xp, xp, vp.x * vp.x + vp.y * vp.y + vp.z * vp.z,
            outside ? 1ull : 0ull};
    }
};

struct FluidParticleReduce {
    __host__ __device__ FluidParticleReduction operator()(
        const FluidParticleReduction& a,
        const FluidParticleReduction& b) const
    {
        return {
            Vec3d{fmin(a.min_corner.x, b.min_corner.x),
                fmin(a.min_corner.y, b.min_corner.y),
                fmin(a.min_corner.z, b.min_corner.z)},
            Vec3d{fmax(a.max_corner.x, b.max_corner.x),
                fmax(a.max_corner.y, b.max_corner.y),
                fmax(a.max_corner.z, b.max_corner.z)},
            fmax(a.max_speed_squared, b.max_speed_squared),
            a.outside_particles + b.outside_particles};
    }
};

// The pushout kernels below call the pressureSolidAtKey/vec3ValueAtKey
// helpers, which only exist in the AMGX build (see the #ifdef ENABLE_AMGX
// block above); computePersistentPressureGridPushoutGpu3D, their only
// caller, is declared AMGX-only in GpuSparseGridProbe.h for the same
// reason.
#ifdef ENABLE_AMGX
// ---------------------------------------------------------------------------
// Device-resident PositionCorrector pushout.
//
// Reproduces PositionCorrector::pushout()'s particle projection against the
// analytic collision list (box-shell only; GHOST is never present in that
// list for supported scenes) and the resident particle level set, plus the
// node-level max-displacement reduction and the pushout RHS/face-velocity
// seeding that already exists as `applyPressurePushoutRhsKernel` /
// `initializePressureFaceVelocityKernel`. See Lib/Ziran/Math/Geometry/
// AnalyticLevelSet.cpp (AnalyticBox, DifferenceLevelSet) and
// Projects/coupling/ParticlesLevelSet.h::queryInside for the CPU formulas
// this mirrors.
// ---------------------------------------------------------------------------

// Matches ParticlesLevelSet::queryInside's gpu_result_active branch exactly:
// trilinear reconstruction of the dense phi field on the *unshifted*
// build_dx grid (not the Q1 pressure-node grid), plus the gradient of that
// same trilinear reconstruction as the normal. Returns false (not valid) on
// out-of-range or untouched (sentinel) samples, matching the CPU early
// `return false`.
__device__ bool particleLevelSetSignedDistanceAndNormal(const Vec3d& x,
    const GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D& level_set,
    double& phi, Vec3d& normal)
{
    if (!level_set.phi || !(level_set.h > 0.0))
        return false;
    const double h = level_set.h;
    const double one_over_h = 1.0 / h;
    const int base_x = floorToIntDevice(x.x * one_over_h);
    const int base_y = floorToIntDevice(x.y * one_over_h);
    const int base_z = floorToIntDevice(x.z * one_over_h);
    const double tx = x.x * one_over_h - static_cast<double>(base_x);
    const double ty = x.y * one_over_h - static_cast<double>(base_y);
    const double tz = x.z * one_over_h - static_cast<double>(base_z);
    const double wx[2] = {1.0 - tx, tx};
    const double wy[2] = {1.0 - ty, ty};
    const double wz[2] = {1.0 - tz, tz};
    const double dwx[2] = {-one_over_h, one_over_h};
    const double dwy[2] = {-one_over_h, one_over_h};
    const double dwz[2] = {-one_over_h, one_over_h};
    const auto* phi_values = static_cast<const double*>(level_set.phi);
    double acc_phi = 0.0;
    Vec3d acc_normal{0.0, 0.0, 0.0};
    for (int i = 0; i < 2; ++i) {
        const int lx = base_x + i - level_set.origin[0];
        for (int j = 0; j < 2; ++j) {
            const int ly = base_y + j - level_set.origin[1];
            for (int k = 0; k < 2; ++k) {
                const int lz = base_z + k - level_set.origin[2];
                if (lx < 0 || lx >= level_set.shape[0] ||
                    ly < 0 || ly >= level_set.shape[1] ||
                    lz < 0 || lz >= level_set.shape[2])
                    return false;
                const std::size_t index =
                    (static_cast<std::size_t>(lz) * level_set.shape[1] +
                        static_cast<std::size_t>(ly)) * level_set.shape[0] +
                    static_cast<std::size_t>(lx);
                const double value = phi_values[index];
                if (value >= DBL_MAX / 2.0)
                    return false;
                const double wijk = wx[i] * wy[j] * wz[k];
                acc_phi += wijk * value;
                acc_normal.x += dwx[i] * wy[j] * wz[k] * value;
                acc_normal.y += wx[i] * dwy[j] * wz[k] * value;
                acc_normal.z += wx[i] * wy[j] * dwz[k] * value;
            }
        }
    }
    const double norm = sqrt(acc_normal.x * acc_normal.x +
        acc_normal.y * acc_normal.y + acc_normal.z * acc_normal.z);
    if (norm > 0.0) {
        acc_normal.x /= norm;
        acc_normal.y /= norm;
        acc_normal.z /= norm;
    }
    phi = acc_phi;
    normal = acc_normal;
    return true;
}

// Reproduces PositionCorrector::isSolid()/directPressureIsSolid() exactly:
// a touched-key hit uses the cached classification flag (same as
// pressureSolidAtKey), but a miss falls back to a real geometric query
// (computeIsSolid()'s box-shell-or-level-set check at the node centre
// `(node+1)*dx`), not a blanket "solid". This is deliberately a *different*
// miss convention from pressureSolidAtKey. Both applyPressurePushoutRhsKernel
// and pushoutFaceVelocityGatherKernel use this — pressureSolidAtKey's
// "miss=solid" is only valid for the non-pushout CSR neighbour lookups
// (directPressureCsrGpu), where the touched set is built wide enough that a
// miss genuinely means solid; the pushout's own narrower touched/solid_count
// set does not have that property (a Stage 4 substep-11+ RHS divergence of
// up to ~1e-3 against the CPU oracle was traced to exactly this before the
// RHS kernel was switched over, matching the face-velocity kernel's earlier
// fix for the identical bug).
__device__ bool pushoutIsSolid(const GridKey* sorted_solid_keys,
    const unsigned char* sorted_solid_flags, int solid_count, int nx, int ny,
    int nz, const BoxShellCollision3D& collision,
    const GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D& level_set,
    double dx)
{
    const GridKey key = packGridKey3DUnchecked(nx, ny, nz);
    const int found = binarySearchKey(sorted_solid_keys, solid_count, key);
    if (found < solid_count && sorted_solid_keys[found] == key)
        return sorted_solid_flags[found] != 0;
    const Vec3d position{(static_cast<double>(nx) + 1.0) * dx,
        (static_cast<double>(ny) + 1.0) * dx,
        (static_cast<double>(nz) + 1.0) * dx};
    return boxShellInsideSignedDistance(position, collision)
        || particleLevelSetInside(position, level_set);
}

// One thread per resident fluid particle. Reproduces the body of
// PositionCorrector::pushout()'s `pushoutParticleProjection` timer block:
// classify the particle's Q1-shifted node against the resident solid flags,
// falling back to the same geometry query as CPU isSolid() on a cache miss,
// then run the same two sequential checks (analytic
// box-shell collision, then the particle level set) on the *same* running
// Xt/Vt the CPU loop uses. Directly patches the resident Q1 G2P owner's V and
// gradV in place; the position correction Xt is only ever used locally to
// derive the emitted per-node displacement, exactly like the CPU version.
__global__ void pushoutParticleProjectionKernel(const Vec3d* particle_x,
    int particle_count, Vec3d* output_v, Mat3d* output_grad_v,
    const GridKey* sorted_solid_keys, const unsigned char* sorted_solid_flags,
    int solid_count, BoxShellCollision3D collision,
    GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D level_set,
    double dx, PushoutCandidate* candidates, unsigned char* candidate_valid)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    candidate_valid[p] = 0;
    const Vec3d xp = particle_x[p];
    // Match BSplineWeights<T,dim,1>(Xp - 0.5*dx*Ones(), dx) exactly: CPU
    // precomputes one_over_dx = 1/dx and multiplies by it, which is not
    // bit-identical to a direct division here. Any mismatch shifts which
    // integer node a particle right on a grid boundary floors to, and
    // therefore its isSolid() classification.
    const double one_over_dx = 1.0 / dx;
    const double shifted_x = (xp.x - 0.5 * dx) * one_over_dx;
    const double shifted_y = (xp.y - 0.5 * dx) * one_over_dx;
    const double shifted_z = (xp.z - 0.5 * dx) * one_over_dx;
    const int nx = floorToIntDevice(shifted_x);
    const int ny = floorToIntDevice(shifted_y);
    const int nz = floorToIntDevice(shifted_z);
    const GridKey node_key = packGridKey3DUnchecked(nx, ny, nz);
    if (!pushoutIsSolid(sorted_solid_keys, sorted_solid_flags, solid_count,
            nx, ny, nz, collision, level_set, dx))
        return;

    Vec3d xt = xp;
    Vec3d vt = output_v[p];
    bool modified = false;
    int debug_source = 0;

    // Analytic collision list: only the box-shell SLIP descriptor is
    // present for supported scenes (GHOST objects are never added to the
    // fluid's collision_objects list, so there is nothing to skip here).
    if (collision.valid) {
        Vec3d normal;
        const double phi = boxShellSignedDistanceAndNormal(xt, collision, normal);
        if (phi <= 0.0) {
            xt.x -= phi * normal.x;
            xt.y -= phi * normal.y;
            xt.z -= phi * normal.z;
            const double vn = vt.x * normal.x + vt.y * normal.y + vt.z * normal.z;
            vt.x -= vn * normal.x;
            vt.y -= vn * normal.y;
            vt.z -= vn * normal.z;
            modified = true;
            debug_source |= 1;
        }
    }
    {
        double phi;
        Vec3d normal;
        const bool valid = particleLevelSetSignedDistanceAndNormal(
            xt, level_set, phi, normal);
        if (valid && phi < 0.0) {
            xt.x -= phi * normal.x;
            xt.y -= phi * normal.y;
            xt.z -= phi * normal.z;
            const double vn = vt.x * normal.x + vt.y * normal.y + vt.z * normal.z;
            vt.x -= vn * normal.x;
            vt.y -= vn * normal.y;
            vt.z -= vn * normal.z;
            modified = true;
            debug_source |= 2;
        }
    }
    if (!modified)
        return;
    output_v[p] = vt;
    output_grad_v[p] = Mat3d{};
    const Vec3d disp{xt.x - xp.x, xt.y - xp.y, xt.z - xp.z};
    const double sq_disp = disp.x * disp.x + disp.y * disp.y + disp.z * disp.z;
    candidates[p] = PushoutCandidate{node_key, sq_disp, disp, vt, p, debug_source};
    candidate_valid[p] = 1;
}

struct PushoutMaxDispOp {
    __device__ PushoutCandidate operator()(
        const PushoutCandidate& a, const PushoutCandidate& b) const
    {
        return b.sq_disp > a.sq_disp ? b : a;
    }
};

// Named functors (rather than `[] __device__(...)` lambdas, which would
// need --extended-lambda) for the thrust:: calls in
// computePersistentPressureGridPushoutGpu3D below.
struct PushoutValidPredicate {
    __device__ bool operator()(unsigned char v) const { return v != 0; }
};

struct PushoutKeyLess {
    __device__ bool operator()(
        const PushoutCandidate& a, const PushoutCandidate& b) const
    {
        return a.key < b.key;
    }
};

struct PushoutKeyEqual {
    __device__ bool operator()(
        const PushoutCandidate& a, const PushoutCandidate& b) const
    {
        return a.key == b.key;
    }
};

struct PushoutExtractKey {
    __device__ GridKey operator()(const PushoutCandidate& c) const { return c.key; }
};

struct PushoutExtractDisp {
    __device__ Vec3d operator()(const PushoutCandidate& c) const { return c.disp; }
};

__global__ void clampPushoutDisplacementKernel(
    Vec3d* displacement, int count, double clamp_value)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    Vec3d v = displacement[i];
    v.x = fmax(-clamp_value, fmin(clamp_value, v.x));
    v.y = fmax(-clamp_value, fmin(clamp_value, v.y));
    v.z = fmax(-clamp_value, fmin(clamp_value, v.z));
    displacement[i] = v;
}

// One thread per resident touched node. Deterministically reproduces the
// CPU `pushoutPressureVelocity` "double write" (a node's -d/+d face is
// written both by its own iteration and by its neighbour's) by resolving the
// ambiguity with each node's *legacy* touched-key rank (the order
// `solid_grid.iterateGridSerial` actually visits, which
// `direct_pressure_touched_keys` / `storage.touched_keys` already replicate
// by construction): whichever of {node, node-e_d} has the *later* legacy
// rank is the writer that would execute last on CPU, so its formula wins.
__global__ void pushoutFaceVelocityGatherKernel(
    const GridKey* sorted_solid_keys, const unsigned char* sorted_solid_flags,
    const int* sorted_solid_rank, int solid_count,
    const GridKey* displacement_keys, const Vec3d* displacement,
    int displacement_count, BoxShellCollision3D collision,
    GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D level_set, double dx,
    Vec3d* face_velocity)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= solid_count)
        return;
    const GridKey self_key = sorted_solid_keys[index];
    int x, y, z;
    unpackGridKey3DUnchecked(self_key, x, y, z);
    const int self_rank = sorted_solid_rank[index];
    const int self_disp_found = binarySearchKey(displacement_keys,
        displacement_count, self_key);
    // Presence in `displacement_keys` (the compact Stage 1/2 reduction
    // output), not `sorted_solid_keys` (the broader isSolid-classification
    // touched set), is what determines whether a node has its own write in
    // CPU's `solid_grid.iterateGridSerial` loop: that loop only visits nodes
    // that actually carry a pushout DOF, which is exactly the
    // `displacement_keys` set. A node can be classification-touched (queried
    // by isSolid, member of `sorted_solid_keys`) without ever being visited
    // as `node` there, and so never produces its own line-1 write.
    const bool self_has_dof = self_disp_found < displacement_count
        && displacement_keys[self_disp_found] == self_key;
    const Vec3d self_disp = self_has_dof
        ? displacement[self_disp_found] : Vec3d{};
    double value[3] = {0.0, 0.0, 0.0};
    for (int d = 0; d < 3; ++d) {
        const int lx = x - (d == 0), ly = y - (d == 1), lz = z - (d == 2);
        const GridKey lower_key = packGridKey3DUnchecked(lx, ly, lz);
        const int lower_disp_found = binarySearchKey(displacement_keys,
            displacement_count, lower_key);
        const bool lower_has_dof = lower_disp_found < displacement_count
            && displacement_keys[lower_disp_found] == lower_key;
        // isSolid(), not the "miss=solid" pressureSolidAtKey convention: a
        // touched-key miss here is a real geometry query on CPU
        // (computeIsSolid()), not an automatic solid classification.
        const bool lower_solid = pushoutIsSolid(sorted_solid_keys,
            sorted_solid_flags, solid_count, lx, ly, lz, collision, level_set,
            dx);
        const Vec3d lower_disp = lower_has_dof
            ? displacement[lower_disp_found] : Vec3d{};
        // Face between (self - e_d) and self. Whichever of the two sides
        // actually has its own DOF write is authoritative; if only one does,
        // the CPU loop never visited the other side at all so there is no
        // tie to break. Only fall back to legacy-order rank when BOTH sides
        // have a real write and genuinely raced for the last word.
        bool self_last;
        if (self_has_dof && lower_has_dof) {
            const int lower_found = binarySearchKey(
                sorted_solid_keys, solid_count, lower_key);
            const bool lower_found_ok = lower_found < solid_count
                && sorted_solid_keys[lower_found] == lower_key;
            self_last = !lower_found_ok
                || self_rank > sorted_solid_rank[lower_found];
        } else {
            self_last = self_has_dof;
        }
        double face;
        if (self_last) {
            // self's own line-1 write: avg(self, lower_solid ? lower : self)
            const double other = lower_solid
                ? (d == 0 ? lower_disp.x : (d == 1 ? lower_disp.y : lower_disp.z))
                : (d == 0 ? self_disp.x : (d == 1 ? self_disp.y : self_disp.z));
            const double mine =
                d == 0 ? self_disp.x : (d == 1 ? self_disp.y : self_disp.z);
            face = 0.5 * (mine + other);
        } else {
            // lower's line-2 write (lower writing to its +d neighbour=self):
            // avg(lower, self_solid ? self : lower)
            const bool self_solid = pushoutIsSolid(sorted_solid_keys,
                sorted_solid_flags, solid_count, x, y, z, collision, level_set,
                dx);
            const double other = self_solid
                ? (d == 0 ? self_disp.x : (d == 1 ? self_disp.y : self_disp.z))
                : (d == 0 ? lower_disp.x : (d == 1 ? lower_disp.y : lower_disp.z));
            const double mine =
                d == 0 ? lower_disp.x : (d == 1 ? lower_disp.y : lower_disp.z);
            face = 0.5 * (mine + other);
        }
        value[d] = face;
    }
    face_velocity[index] = Vec3d{value[0], value[1], value[2]};
}
#endif // ENABLE_AMGX

__device__ GridKey fluidCellKey(const Vec3d& x, double dx)
{
    return packGridKey3DUnchecked(
        floorToIntDevice((x.x - 0.5 * dx) / dx),
        floorToIntDevice((x.y - 0.5 * dx) / dx),
        floorToIntDevice((x.z - 0.5 * dx) / dx));
}

__device__ Vec3d fluidCellCenter(GridKey key, double dx)
{
    int x, y, z;
    unpackGridKey3DUnchecked(key, x, y, z);
    return Vec3d{(x + 1.0) * dx, (y + 1.0) * dx, (z + 1.0) * dx};
}

__global__ void markFluidPruneKernel(const Vec3d* xq, int count, double dx,
    BoxShellCollision3D collision, GridKey* keys, int* keep)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const GridKey key = fluidCellKey(xq[i], dx);
    keys[i] = key;
    keep[i] = boxShellCollision(fluidCellCenter(key, dx), collision) ? 0 : 1;
}

__global__ void classifyFluidCellsKernel(const GridKey* active_keys,
    int cell_count, double dx, BoxShellCollision3D collision, int* flags)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cell_count)
        return;
    int x, y, z;
    unpackGridKey3DUnchecked(active_keys[i], x, y, z);
    int boundary = 0;
    for (int di = -1; di <= 1 && !boundary; ++di)
        for (int dj = -1; dj <= 1 && !boundary; ++dj)
            for (int dk = -1; dk <= 1; ++dk) {
                const GridKey neighbor_key =
                    packGridKey3DUnchecked(x + di, y + dj, z + dk);
                const int found = binarySearchKey(active_keys, cell_count, neighbor_key);
                if ((found >= cell_count || active_keys[found] != neighbor_key) &&
                    !boxShellCollision(
                        Vec3d{(x + di + 1.0) * dx,
                            (y + dj + 1.0) * dx,
                            (z + dk + 1.0) * dx},
                        collision)) {
                    boundary = 1;
                    break;
                }
            }
    flags[i] = boundary;
}

__global__ void markInterfaceCellsKernel(const Vec3d* interface_x,
    int count, double dx, const GridKey* active_keys, int active_count,
    int* flags)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const GridKey key = fluidCellKey(interface_x[i], dx);
    const int cell = binarySearchKey(active_keys, active_count, key);
    if (cell < active_count && active_keys[cell] == key)
        flags[cell] = 1;
}

__global__ void markFirstFluidSamplesKernel(const GridKey* sample_keys,
    int sample_count, const GridKey* active_keys, int active_count,
    int* first_sample)
{
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= sample_count)
        return;
    const GridKey key = sample_keys[sample];
    const int cell = binarySearchKey(active_keys, active_count, key);
    if (cell < active_count && active_keys[cell] == key)
        atomicMin(first_sample + cell, sample);
}

__global__ void emitFluidFacesKernel(const GridKey* active_keys,
    const int* cell_order, int cell_count, double dx,
    BoxShellCollision3D collision,
    Vec3d* face_x, Vec3d* face_normal, int* keep)
{
    const int cell_rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_rank >= cell_count)
        return;
    const int cell = cell_order[cell_rank];
    int x, y, z;
    unpackGridKey3DUnchecked(active_keys[cell], x, y, z);
    for (int d = 0; d < 3; ++d) {
        for (int delta_index = 0; delta_index < 2; ++delta_index) {
            const int delta = delta_index == 0 ? -1 : 1;
            int neighbor[3] = {x, y, z};
            neighbor[d] += delta;
            const Vec3d query{(neighbor[0] + 1.0) * dx,
                (neighbor[1] + 1.0) * dx,
                (neighbor[2] + 1.0) * dx};
            const int out = cell_rank * 6 + d * 2 + delta_index;
            if (!boxShellCollision(query, collision)) {
                keep[out] = 0;
                continue;
            }
            double p[3] = {(neighbor[0] + 1.0) * dx,
                (neighbor[1] + 1.0) * dx,
                (neighbor[2] + 1.0) * dx};
            p[d] -= static_cast<double>(delta) * 0.5 * dx;
            face_x[out] = Vec3d{p[0], p[1], p[2]};
            double n[3] = {0.0, 0.0, 0.0};
            n[d] = static_cast<double>(delta);
            face_normal[out] = Vec3d{n[0], n[1], n[2]};
            keep[out] = 1;
        }
    }
}

__global__ void buildFluidOutputCandidatesKernel(const Vec3d* pruned_x,
    const double* pruned_mass, const GridKey* sample_keys, int count,
    const GridKey* active_keys, int active_count, const int* cell_flags,
    const int* cell_first_sample, double dx, double rho,
    Vec3d* output_x, double* output_mass, int* keep)
{
    const int source = blockIdx.x * blockDim.x + threadIdx.x;
    if (source >= count)
        return;
    const GridKey key = sample_keys[source];
    const int cell = binarySearchKey(active_keys, active_count, key);
    if (cell >= active_count || active_keys[cell] != key) {
        keep[source] = 0;
        return;
    }
    if (cell_flags[cell] != 0) {
        Vec3d x = pruned_x[source];
        int cx, cy, cz;
        unpackGridKey3DUnchecked(key, cx, cy, cz);
        const double lower[3] = {(cx + 0.6) * dx, (cy + 0.6) * dx,
            (cz + 0.6) * dx};
        const double upper[3] = {(cx + 1.4) * dx, (cy + 1.4) * dx,
            (cz + 1.4) * dx};
        x.x = fmin(fmax(x.x, lower[0]), upper[0]);
        x.y = fmin(fmax(x.y, lower[1]), upper[1]);
        x.z = fmin(fmax(x.z, lower[2]), upper[2]);
        output_x[source] = x;
        output_mass[source] = pruned_mass[source];
        keep[source] = 1;
        return;
    }
    const bool first = cell_first_sample[cell] == source;
    if (first) {
        output_x[source] = fluidCellCenter(key, dx);
        output_mass[source] = dx * dx * dx * rho;
        keep[source] = 1;
    } else {
        keep[source] = 0;
    }
}

__global__ void clampInterfaceKernel(Vec3d* xq, int count, double dx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const GridKey key = fluidCellKey(xq[i], dx);
    int cx, cy, cz;
    unpackGridKey3DUnchecked(key, cx, cy, cz);
    xq[i].x = fmin(fmax(xq[i].x, (cx + 0.6) * dx), (cx + 1.4) * dx);
    xq[i].y = fmin(fmax(xq[i].y, (cy + 0.6) * dx), (cy + 1.4) * dx);
    xq[i].z = fmin(fmax(xq[i].z, (cz + 0.6) * dx), (cz + 1.4) * dx);
}

__global__ void generateFluidMomentumContributionsKernel(const Vec3d* xq,
    const double* mass, const Vec3d* velocity, const Mat3d* affine,
    int count, double dx, GridKey* keys, double* contribution_mass,
    Vec3d* contribution_momentum)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count)
        return;
    const Vec3d shifted{xq[q].x - 0.5 * dx, xq[q].y - 0.5 * dx,
        xq[q].z - 0.5 * dx};
    const double gx[3] = {shifted.x / dx, shifted.y / dx, shifted.z / dx};
    int base[3] = {floorToIntDevice(gx[0]), floorToIntDevice(gx[1]),
        floorToIntDevice(gx[2])};
    double w[3][2];
    for (int d = 0; d < 3; ++d) {
        w[d][1] = gx[d] - static_cast<double>(base[d]);
        w[d][0] = 1.0 - w[d][1];
    }
    const double vp[3] = {velocity[q].x, velocity[q].y, velocity[q].z};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) {
                const int out = q * 8 + i * 4 + j * 2 + k;
                keys[out] = packGridKey3DUnchecked(
                    base[0] + i, base[1] + j, base[2] + k);
                const double weight = w[0][i] * w[1][j] * w[2][k];
                const double displacement[3] = {
                    (base[0] + i) * dx - shifted.x,
                    (base[1] + j) * dx - shifted.y,
                    (base[2] + k) * dx - shifted.z};
                double node_momentum[3] = {
                    mass[q] * vp[0], mass[q] * vp[1], mass[q] * vp[2]};
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 3; ++col)
                        node_momentum[row] += mass[q] *
                            affine[q].values[row * 3 + col] * displacement[col];
                contribution_mass[out] = mass[q] * weight;
                contribution_momentum[out] = Vec3d{
                    node_momentum[0] * weight,
                    node_momentum[1] * weight,
                    node_momentum[2] * weight};
            }
}

__global__ void gatherInterfaceQuadSolidVelocityKernel(
    const Vec3d* solid_momentum, const double* solid_mass,
    const int* active_solid_ids, int active_count, Vec3d* output_velocity)
{
    const int active = blockIdx.x * blockDim.x + threadIdx.x;
    if (active >= active_count)
        return;
    const int solid = active_solid_ids[active];
    const double mass = solid_mass[solid];
    const Vec3d momentum = solid_momentum[solid];
    output_velocity[active] = mass > 0.0
        ? Vec3d{momentum.x / mass, momentum.y / mass, momentum.z / mass}
        : Vec3d{};
}

__global__ void gatherInterfaceQuadratureFieldsKernel(
    const int* active_solid_ids, int active_count,
    const GPU_STAGE3A::Vec3d* solid_rest_x,
    const Vec3d* solid_current_x, const Vec3d* solid_normal,
    const double* solid_volume, Vec3d* interface_rest_x,
    Vec3d* interface_current_x, Vec3d* interface_normal,
    double* interface_length)
{
    const int active = blockIdx.x * blockDim.x + threadIdx.x;
    if (active >= active_count)
        return;
    const int particle = active_solid_ids[active];
    const auto rest = solid_rest_x[particle];
    interface_rest_x[active] = Vec3d{rest.x, rest.y, rest.z};
    interface_current_x[active] = solid_current_x[particle];
    interface_normal[active] = solid_normal[particle];
    interface_length[active] = 2.0 * sqrt(fmax(solid_volume[particle], 0.0)
        / 3.14159265358979323846);
}

__global__ void buildInterfaceQuadFluidAffineKernel(
    const Mat3d* grad_velocity, int fluid_count, double apic_rpic_ratio,
    Mat3d* output_affine)
{
    const int fluid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fluid >= fluid_count)
        return;
    const double direct = 0.5 * (apic_rpic_ratio + 1.0);
    const double transpose = 0.5 * (apic_rpic_ratio - 1.0);
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            output_affine[fluid].values[row * 3 + col] =
                direct * grad_velocity[fluid].values[row + 3 * col] +
                transpose * grad_velocity[fluid].values[col + 3 * row];
}

__global__ void reduceFluidMomentumContributionsKernel(
    const GridKey* contribution_keys, const double* contribution_mass,
    const Vec3d* contribution_momentum, int contribution_count,
    const GridKey* active_keys, int active_count, double* grid_mass,
    Vec3d* grid_momentum)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= active_count)
        return;
    const GridKey key = active_keys[node];
    const int begin = binarySearchKey(contribution_keys, contribution_count, key);
    const int end = upperBoundKey(contribution_keys, contribution_count, key);
    double m = 0.0;
    Vec3d momentum{0.0, 0.0, 0.0};
    for (int i = begin; i < end; ++i) {
        m += contribution_mass[i];
        momentum.x += contribution_momentum[i].x;
        momentum.y += contribution_momentum[i].y;
        momentum.z += contribution_momentum[i].z;
    }
    grid_mass[node] = m;
    grid_momentum[node] = momentum;
}

__global__ void normalizeFluidVelocityKernel(const double* mass,
    const Vec3d* momentum, int count, double dt, Vec3d gravity,
    Vec3d* velocity)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const double inv_mass = mass[i] != 0.0 ? 1.0 / mass[i] : 0.0;
    velocity[i] = Vec3d{momentum[i].x * inv_mass + dt * gravity.x,
        momentum[i].y * inv_mass + dt * gravity.y,
        momentum[i].z * inv_mass + dt * gravity.z};
}

__global__ void generateFluidMassContributionsKernel(const Vec3d* xq,
    const double* mass, int count, double dx, GridKey* keys,
    double* contribution_mass)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count)
        return;
    const double gx[3] = {(xq[q].x - 0.5 * dx) / dx,
        (xq[q].y - 0.5 * dx) / dx, (xq[q].z - 0.5 * dx) / dx};
    int base[3] = {floorToIntDevice(gx[0]), floorToIntDevice(gx[1]),
        floorToIntDevice(gx[2])};
    double w[3][2];
    for (int d = 0; d < 3; ++d) {
        w[d][1] = gx[d] - static_cast<double>(base[d]);
        w[d][0] = 1.0 - w[d][1];
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) {
                const int out = q * 8 + i * 4 + j * 2 + k;
                keys[out] = packGridKey3DUnchecked(
                    base[0] + i, base[1] + j, base[2] + k);
                contribution_mass[out] =
                    mass[q] * w[0][i] * w[1][j] * w[2][k];
            }
}

__global__ void reduceFluidMassContributionsKernel(
    const GridKey* contribution_keys, const double* contribution_mass,
    int contribution_count, const GridKey* active_keys, int active_count,
    double* grid_mass)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= active_count)
        return;
    const GridKey key = active_keys[node];
    const int begin = binarySearchKey(contribution_keys, contribution_count, key);
    const int end = upperBoundKey(contribution_keys, contribution_count, key);
    double mass = 0.0;
    for (int i = begin; i < end; ++i)
        mass += contribution_mass[i];
    grid_mass[node] = mass;
}

// Match the two-stage CPU grid semantics. Initial non-zero-mass nodes are the
// only ones that are normalized.  The same keyed gather serves both the final
// DOF set (final-only nodes retain zero velocity) and the initial-only G2P
// fallback set (whose velocity is never overwritten by the CPU solve).
__global__ void gatherInitialVelocityAtKeysKernel(
    const GridKey* initial_keys, const Vec3d* initial_velocity,
    int initial_count, const GridKey* output_keys, int output_count,
    Vec3d* output_velocity)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= output_count)
        return;
    const int initial = binarySearchKey(initial_keys, initial_count,
        output_keys[node]);
    output_velocity[node] = (initial < initial_count &&
        initial_keys[initial] == output_keys[node])
        ? initial_velocity[initial]
        : Vec3d{0.0, 0.0, 0.0};
}

__global__ void scatterSolidPressureKernel(const Vec3d* positions,
    const double* j, const double* volume, const double* lambda, int count,
    double dx, const GridKey* keys, int key_count, double* grid_mass,
    double* compliance, double* pressure_numerator)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= count)
        return;
    const Vec3d shifted{positions[p].x - 0.5 * dx,
        positions[p].y - 0.5 * dx, positions[p].z - 0.5 * dx};
    const GridKey key = packGridKey3DUnchecked(
        floorToIntDevice(shifted.x / dx), floorToIntDevice(shifted.y / dx),
        floorToIntDevice(shifted.z / dx));
    const int node = binarySearchKey(keys, key_count, key);
    if (node >= key_count || keys[node] != key)
        return;
    const double pressure = -lambda[p] * (j[p] - 1.0);
    atomicAdd(grid_mass + node, volume[p] * j[p]);
    atomicAdd(compliance + node, volume[p] / lambda[p]);
    atomicAdd(pressure_numerator + node, volume[p] * j[p] * pressure);
}

__global__ void normalizeSolidPressureKernel(const double* mass,
    const double* numerator, int count, double* pressure)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    pressure[i] = mass[i] != 0.0 ? numerator[i] / mass[i] : 0.0;
}

struct IsNonZero {
    __host__ __device__ bool operator()(int value) const { return value != 0; }
};

void finishUnitCountProbe(thrust::device_vector<GridKey>& d_sample_keys, ProbeGridResult& result)
{
    thrust::device_vector<GridKey> d_active_keys = d_sample_keys;
    thrust::sort(d_active_keys.begin(), d_active_keys.end());
    auto new_end = thrust::unique(d_active_keys.begin(), d_active_keys.end());
    d_active_keys.erase(new_end, d_active_keys.end());

    thrust::device_vector<double> d_grid_count(d_active_keys.size(), 0.0);
    if (!d_sample_keys.empty()) {
        const int block_size = 256;
        const int block_count = static_cast<int>((d_sample_keys.size() + block_size - 1) / block_size);
        scatterUnitCountsKernel<<<block_count, block_size>>>(
            thrust::raw_pointer_cast(d_sample_keys.data()), static_cast<int>(d_sample_keys.size()),
            thrust::raw_pointer_cast(d_active_keys.data()), static_cast<int>(d_active_keys.size()),
            thrust::raw_pointer_cast(d_grid_count.data()));
        checkCuda(cudaGetLastError(), "scatterUnitCountsKernel launch");
        checkCuda(cudaDeviceSynchronize(), "scatterUnitCountsKernel synchronize");
    }

    result.keys.resize(d_active_keys.size());
    result.mass.resize(d_grid_count.size());
    thrust::copy(d_active_keys.begin(), d_active_keys.end(), result.keys.begin());
    thrust::copy(d_grid_count.begin(), d_grid_count.end(), result.mass.begin());
}

} // namespace

GridKey packGridKey3D(int x, int y, int z)
{
    const int bx = x + kCoordBias;
    const int by = y + kCoordBias;
    const int bz = z + kCoordBias;
    if (bx < 0 || by < 0 || bz < 0 || bx > static_cast<int>(kCoordMask) || by > static_cast<int>(kCoordMask) || bz > static_cast<int>(kCoordMask))
        throw std::out_of_range("grid coordinate is outside packGridKey3D range");
    return packGridKey3DUnchecked(x, y, z);
}

void unpackGridKey3D(GridKey key, int& x, int& y, int& z)
{
    x = static_cast<int>((key >> (2 * kCoordBits)) & kCoordMask) - kCoordBias;
    y = static_cast<int>((key >> kCoordBits) & kCoordMask) - kCoordBias;
    z = static_cast<int>(key & kCoordMask) - kCoordBias;
}

void sortUniqueKeys(std::vector<GridKey>& keys)
{
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
}

ProbeCompareStats compareKeySets(std::vector<GridKey> cpu_keys,
    std::vector<GridKey> gpu_keys)
{
    sortUniqueKeys(cpu_keys);
    sortUniqueKeys(gpu_keys);

    ProbeCompareStats stats;
    stats.cpu_nodes = cpu_keys.size();
    stats.gpu_nodes = gpu_keys.size();

    std::size_t i = 0;
    std::size_t j = 0;
    while (i < cpu_keys.size() && j < gpu_keys.size()) {
        if (cpu_keys[i] == gpu_keys[j]) {
            ++i;
            ++j;
        }
        else if (cpu_keys[i] < gpu_keys[j]) {
            ++stats.missing_on_gpu;
            ++i;
        }
        else {
            ++stats.extra_on_gpu;
            ++j;
        }
    }
    stats.missing_on_gpu += cpu_keys.size() - i;
    stats.extra_on_gpu += gpu_keys.size() - j;
    return stats;
}

void runLinearScalarScatterProbe(const std::vector<Vec3d>& xq,
    const std::vector<double>& values, double dx, bool halfDxAnchor,
    ProbeGridResult& result)
{
    if (xq.size() != values.size())
        throw std::runtime_error("runLinearScalarScatterProbe input size mismatch");

    result.keys.clear();
    result.mass.clear();
    if (xq.empty())
        return;

    thrust::device_vector<Vec3d> d_xq(xq.begin(), xq.end());
    thrust::device_vector<double> d_values(values.begin(), values.end());
    thrust::device_vector<GridKey> d_keys(xq.size() * 8);

    const int block_size = 256;
    const int block_count = static_cast<int>((xq.size() + block_size - 1) / block_size);
    generateLinearVelocityKeysKernel<<<block_count, block_size>>>(
        thrust::raw_pointer_cast(d_xq.data()), static_cast<int>(xq.size()), dx,
        halfDxAnchor ? 1 : 0, thrust::raw_pointer_cast(d_keys.data()));
    checkCuda(cudaGetLastError(), "generateLinearVelocityKeysKernel launch");
    checkCuda(cudaDeviceSynchronize(), "generateLinearVelocityKeysKernel synchronize");

    thrust::sort(d_keys.begin(), d_keys.end());
    auto new_end = thrust::unique(d_keys.begin(), d_keys.end());
    d_keys.erase(new_end, d_keys.end());

    thrust::device_vector<double> d_grid_mass(d_keys.size(), 0.0);
    scatterLinearVelocityMassKernel<<<block_count, block_size>>>(
        thrust::raw_pointer_cast(d_xq.data()), thrust::raw_pointer_cast(d_values.data()),
        static_cast<int>(xq.size()), dx, halfDxAnchor ? 1 : 0,
        thrust::raw_pointer_cast(d_keys.data()), static_cast<int>(d_keys.size()),
        thrust::raw_pointer_cast(d_grid_mass.data()));
    checkCuda(cudaGetLastError(), "scatterLinearVelocityMassKernel launch");
    checkCuda(cudaDeviceSynchronize(), "scatterLinearVelocityMassKernel synchronize");

    result.keys.resize(d_keys.size());
    result.mass.resize(d_grid_mass.size());
    thrust::copy(d_keys.begin(), d_keys.end(), result.keys.begin());
    thrust::copy(d_grid_mass.begin(), d_grid_mass.end(), result.mass.begin());
}

void runPersistentUniformLinearScatter3D(
    PersistentLinearScatter3D& context, const std::vector<Vec3d>& xq,
    double value, double dx, bool halfDxAnchor, ProbeGridResult& result,
    LinearScatterStats* stats)
{
    if (!(dx > 0.0))
        throw std::runtime_error("persistent linear scatter requires positive dx");

    const auto total_begin = std::chrono::high_resolution_clock::now();
    LinearScatterStats local_stats;
    local_stats.sources = xq.size();
    local_stats.candidate_keys = xq.size() * 8;
    result.keys.clear();
    result.mass.clear();
    if (xq.empty()) {
        if (stats)
            *stats = local_stats;
        return;
    }

    auto& storage = getLinearScatterStorage(context);
    constexpr int block_size = 256;
    const int block_count = static_cast<int>((xq.size() + block_size - 1) / block_size);

    const auto upload_begin = std::chrono::high_resolution_clock::now();
    resizeLinearScatter(storage.positions, xq.size(), storage);
    resizeLinearScatter(storage.candidate_keys, xq.size() * 8, storage);
    checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.positions.data()),
                  xq.data(), xq.size() * sizeof(Vec3d), cudaMemcpyHostToDevice),
        "persistent linear scatter position upload");
    checkCuda(cudaDeviceSynchronize(), "persistent linear scatter upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    const auto sort_begin = std::chrono::high_resolution_clock::now();
    generateLinearVelocityKeysKernel<<<block_count, block_size>>>(
        thrust::raw_pointer_cast(storage.positions.data()),
        static_cast<int>(xq.size()), dx, halfDxAnchor ? 1 : 0,
        thrust::raw_pointer_cast(storage.candidate_keys.data()));
    checkCuda(cudaGetLastError(), "persistent linear scatter key launch");
    resizeLinearScatter(storage.active_keys, storage.candidate_keys.size(), storage);
    thrust::copy(storage.candidate_keys.begin(), storage.candidate_keys.end(),
        storage.active_keys.begin());
    thrust::sort(storage.active_keys.begin(), storage.active_keys.end());
    const auto unique_end = thrust::unique(storage.active_keys.begin(),
        storage.active_keys.end());
    storage.active_keys.erase(unique_end, storage.active_keys.end());
    checkCuda(cudaGetLastError(), "persistent linear scatter key sort");
    checkCuda(cudaDeviceSynchronize(), "persistent linear scatter key synchronize");
    const auto sort_end = std::chrono::high_resolution_clock::now();

    const auto scatter_begin = std::chrono::high_resolution_clock::now();
    resizeLinearScatter(storage.grid_mass, storage.active_keys.size(), storage);
    thrust::fill(storage.grid_mass.begin(), storage.grid_mass.end(), 0.0);
    scatterUniformLinearScalarKernel<<<block_count, block_size>>>(
        thrust::raw_pointer_cast(storage.positions.data()),
        static_cast<int>(xq.size()), value, dx, halfDxAnchor ? 1 : 0,
        thrust::raw_pointer_cast(storage.active_keys.data()),
        static_cast<int>(storage.active_keys.size()),
        thrust::raw_pointer_cast(storage.grid_mass.data()));
    checkCuda(cudaGetLastError(), "persistent linear scatter launch");
    checkCuda(cudaDeviceSynchronize(), "persistent linear scatter synchronize");
    const auto scatter_end = std::chrono::high_resolution_clock::now();

    const auto download_begin = std::chrono::high_resolution_clock::now();
    result.keys.resize(storage.active_keys.size());
    result.mass.resize(storage.grid_mass.size());
    checkCuda(cudaMemcpy(result.keys.data(),
                  thrust::raw_pointer_cast(storage.active_keys.data()),
                  result.keys.size() * sizeof(GridKey), cudaMemcpyDeviceToHost),
        "persistent linear scatter key download");
    checkCuda(cudaMemcpy(result.mass.data(),
                  thrust::raw_pointer_cast(storage.grid_mass.data()),
                  result.mass.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "persistent linear scatter value download");
    const auto download_end = std::chrono::high_resolution_clock::now();

    local_stats.nodes = result.keys.size();
    local_stats.reallocations = storage.reallocations;
    local_stats.upload_ms = std::chrono::duration<double, std::milli>(
        upload_end - upload_begin).count();
    local_stats.key_sort_ms = std::chrono::duration<double, std::milli>(
        sort_end - sort_begin).count();
    local_stats.scatter_ms = std::chrono::duration<double, std::milli>(
        scatter_end - scatter_begin).count();
    local_stats.download_ms = std::chrono::duration<double, std::milli>(
        download_end - download_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(
        download_end - total_begin).count();
    if (stats)
        *stats = local_stats;
}

void runPersistentLinearBaseKey3D(
    PersistentPagePollute3D& context, const std::vector<Vec3d>& xq,
    double dx, bool halfDxAnchor, std::vector<GridKey>& base_keys,
    PagePolluteStats* stats)
{
    if (!(dx > 0.0))
        throw std::runtime_error("persistent page pollute requires positive dx");

    const auto total_begin = std::chrono::high_resolution_clock::now();
    PagePolluteStats local_stats;
    local_stats.sources = xq.size();
    base_keys.clear();
    if (xq.empty()) {
        if (stats)
            *stats = local_stats;
        return;
    }

    auto& storage = getPagePolluteStorage(context);
    constexpr int block_size = 256;
    const int block_count = static_cast<int>((xq.size() + block_size - 1) / block_size);

    const auto upload_begin = std::chrono::high_resolution_clock::now();
    resizeLinearScatter(storage.positions, xq.size(), storage);
    resizeLinearScatter(storage.candidate_keys, xq.size(), storage);
    checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.positions.data()),
                  xq.data(), xq.size() * sizeof(Vec3d), cudaMemcpyHostToDevice),
        "persistent page pollute position upload");
    checkCuda(cudaDeviceSynchronize(), "persistent page pollute upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    const auto key_begin = std::chrono::high_resolution_clock::now();
    generatePressureKeysKernel<<<block_count, block_size>>>(
        thrust::raw_pointer_cast(storage.positions.data()),
        static_cast<int>(xq.size()), dx, halfDxAnchor ? 1 : 0,
        thrust::raw_pointer_cast(storage.candidate_keys.data()));
    checkCuda(cudaGetLastError(), "persistent page pollute key launch");
    checkCuda(cudaDeviceSynchronize(), "persistent page pollute key synchronize");
    const auto key_end = std::chrono::high_resolution_clock::now();

    const auto download_begin = std::chrono::high_resolution_clock::now();
    base_keys.resize(storage.candidate_keys.size());
    checkCuda(cudaMemcpy(base_keys.data(),
                  thrust::raw_pointer_cast(storage.candidate_keys.data()),
                  base_keys.size() * sizeof(GridKey), cudaMemcpyDeviceToHost),
        "persistent page pollute key download");
    const auto download_end = std::chrono::high_resolution_clock::now();

    local_stats.base_nodes = base_keys.size();
    local_stats.reallocations = storage.reallocations;
    local_stats.upload_ms = std::chrono::duration<double, std::milli>(
        upload_end - upload_begin).count();
    local_stats.key_generation_ms = std::chrono::duration<double, std::milli>(
        key_end - key_begin).count();
    local_stats.download_ms = std::chrono::duration<double, std::milli>(
        download_end - download_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(
        download_end - total_begin).count();
    if (stats)
        *stats = local_stats;
}

void transferFluidVelocityQ1FromDevice3D(
    PersistentVelocityTransfer3D& context,
    PersistentFluidAnalytic3D* resident_particle_context,
    const GPU_STAGE3A::DeviceGridNodeMap* resident_velocity_nodes,
    const GPU_STAGE3A::DeviceGridNodeMap* resident_pressure_nodes,
    const void* particle_x, std::size_t particle_count,
    const std::vector<GridKey>& sorted_velocity_keys,
    const std::vector<int>& velocity_node_indices,
    const std::vector<GridKey>& sorted_g2p_fallback_keys,
    const std::vector<Vec3d>& sorted_g2p_fallback_velocity,
    const GPU_STAGE3A::DeviceVector& device_velocity,
    const std::vector<GridKey>& sorted_pressure_keys,
    const std::vector<int>& pressure_node_indices,
    const GPU_STAGE3A::DeviceVector& device_pressure,
    double dx, double dt, VelocityTransferHostView& result,
    VelocityTransferDeviceView& device_result,
    VelocityTransferStats* stats,
    VelocityTransferDownloadMask download_mask)
{
    const bool resident_node_input = resident_velocity_nodes != nullptr &&
        resident_pressure_nodes != nullptr;
    const std::size_t velocity_node_count = resident_node_input
        ? resident_velocity_nodes->size : sorted_velocity_keys.size();
    if (!(dx > 0.0) || sorted_velocity_keys.size() != velocity_node_indices.size() ||
        sorted_g2p_fallback_keys.size() != sorted_g2p_fallback_velocity.size() ||
        device_velocity.size != velocity_node_count * 3 ||
        sorted_pressure_keys.size() != pressure_node_indices.size() ||
        std::any_of(pressure_node_indices.begin(), pressure_node_indices.end(),
            [&](int index) { return index < 0 ||
                static_cast<std::size_t>(index) >= device_pressure.size; }))
        throw std::runtime_error("invalid fluid GPU velocity transfer input");
    if (particle_count != 0 && !particle_x && !resident_particle_context)
        throw std::runtime_error("fluid GPU velocity transfer particle input is null");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    VelocityTransferStats local;
    local.particles = particle_count;
    local.velocity_nodes = velocity_node_count;
    local.g2p_fallback_nodes = sorted_g2p_fallback_keys.size();
    local.resident_particle_input = resident_particle_context != nullptr;
    local.resident_node_input = resident_node_input;
    result = {};
    device_result = {};
    if (particle_count == 0) {
        if (stats) *stats = local;
        return;
    }
    auto& storage = getVelocityTransferStorage(context);
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    const Vec3d* device_particle_x = nullptr;
    PersistentFluidAnalyticStorage* resident_fluid = nullptr;
    if (resident_particle_context) {
        if (!resident_particle_context->storage)
            throw std::runtime_error(
                "fluid GPU velocity transfer resident particle context is empty");
        auto& resident = *std::static_pointer_cast<PersistentFluidAnalyticStorage>(
            resident_particle_context->storage);
        resident_fluid = &resident;
        // The analytical/velocity-grid input keeps fluid particles as its
        // leading range. Interface quadrature may append solid samples, so
        // equality is intentionally not required here.
        if (resident.input_x.size() < particle_count)
            throw std::runtime_error(
                "fluid GPU velocity transfer resident particle range is too small");
        device_particle_x = thrust::raw_pointer_cast(resident.input_x.data());
    } else {
        resizeVelocityTransfer(storage.particle_x, particle_count, storage);
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.particle_x.data()),
                  particle_x, particle_count * sizeof(Vec3d),
                  cudaMemcpyHostToDevice),
            "velocity transfer particle upload");
        device_particle_x = thrust::raw_pointer_cast(storage.particle_x.data());
    }
    const auto particle_upload_end = std::chrono::high_resolution_clock::now();
    if (!local.resident_node_input) {
        resizeVelocityTransfer(storage.node_keys, sorted_velocity_keys.size(), storage);
        resizeVelocityTransfer(storage.node_indices, velocity_node_indices.size(), storage);
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.node_keys.data()), sorted_velocity_keys.data(),
                  sorted_velocity_keys.size() * sizeof(GridKey), cudaMemcpyHostToDevice), "velocity transfer key upload");
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.node_indices.data()), velocity_node_indices.data(),
                  velocity_node_indices.size() * sizeof(int), cudaMemcpyHostToDevice), "velocity transfer index upload");
    }
    const GridKey* device_fallback_keys = nullptr;
    const Vec3d* device_fallback_velocity = nullptr;
    local.resident_fallback_input = local.resident_node_input && resident_fluid &&
        resident_fluid->velocity_g2p_fallback_keys.size() ==
            resident_fluid->velocity_g2p_fallback_value.size();
    const std::size_t fallback_node_count = local.resident_fallback_input
        ? resident_fluid->velocity_g2p_fallback_keys.size()
        : sorted_g2p_fallback_keys.size();
    local.g2p_fallback_nodes = fallback_node_count;
    if (local.resident_fallback_input) {
        device_fallback_keys = fallback_node_count == 0 ? nullptr
            : thrust::raw_pointer_cast(
                resident_fluid->velocity_g2p_fallback_keys.data());
        device_fallback_velocity = fallback_node_count == 0 ? nullptr
            : thrust::raw_pointer_cast(
                resident_fluid->velocity_g2p_fallback_value.data());
    } else {
        resizeVelocityTransfer(storage.g2p_fallback_keys,
            sorted_g2p_fallback_keys.size(), storage);
        resizeVelocityTransfer(storage.g2p_fallback_velocity,
            sorted_g2p_fallback_velocity.size(), storage);
    }
    if (!local.resident_fallback_input && !sorted_g2p_fallback_keys.empty()) {
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.g2p_fallback_keys.data()),
                      sorted_g2p_fallback_keys.data(),
                      sorted_g2p_fallback_keys.size() * sizeof(GridKey),
                      cudaMemcpyHostToDevice),
            "velocity transfer fallback key upload");
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.g2p_fallback_velocity.data()),
                      sorted_g2p_fallback_velocity.data(),
                      sorted_g2p_fallback_velocity.size() * sizeof(Vec3d),
                      cudaMemcpyHostToDevice),
            "velocity transfer fallback velocity upload");
    }
    if (!local.resident_fallback_input) {
        device_fallback_keys = sorted_g2p_fallback_keys.empty() ? nullptr
            : thrust::raw_pointer_cast(storage.g2p_fallback_keys.data());
        device_fallback_velocity = sorted_g2p_fallback_velocity.empty() ? nullptr
            : thrust::raw_pointer_cast(storage.g2p_fallback_velocity.data());
    }
    if (!local.resident_node_input) {
        resizeVelocityTransfer(storage.pressure_keys, sorted_pressure_keys.size(), storage);
        resizeVelocityTransfer(storage.pressure_indices, pressure_node_indices.size(), storage);
    }
    if (!local.resident_node_input && !sorted_pressure_keys.empty()) {
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.pressure_keys.data()),
                      sorted_pressure_keys.data(),
                      sorted_pressure_keys.size() * sizeof(GridKey),
                      cudaMemcpyHostToDevice),
            "velocity transfer pressure key upload");
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.pressure_indices.data()),
                      pressure_node_indices.data(),
                      pressure_node_indices.size() * sizeof(int),
                      cudaMemcpyHostToDevice),
            "velocity transfer pressure index upload");
    }
    const auto node_upload_end = std::chrono::high_resolution_clock::now();
    resizeVelocityTransfer(storage.output_x, particle_count, storage);
    resizeVelocityTransfer(storage.output_v, particle_count, storage);
    resizeVelocityTransfer(storage.output_grad_v, particle_count, storage);
    resizeVelocityTransfer(storage.output_pressure, particle_count, storage);
    const auto raw = GPU_STAGE3A::getDeviceVectorRawView(device_velocity);
    const auto raw_pressure = GPU_STAGE3A::getDeviceVectorRawView(device_pressure);
    GPU_STAGE3A::DeviceGridNodeMapRawView raw_velocity_nodes;
    GPU_STAGE3A::DeviceGridNodeMapRawView raw_pressure_nodes;
    if (local.resident_node_input) {
        raw_velocity_nodes = GPU_STAGE3A::getDeviceGridNodeMapRawView(
            *resident_velocity_nodes);
        raw_pressure_nodes = GPU_STAGE3A::getDeviceGridNodeMapRawView(
            *resident_pressure_nodes);
    }
    constexpr int block_size = 256;
    transferFluidVelocityQ1Kernel<<<static_cast<int>((particle_count + block_size - 1) / block_size), block_size>>>(
        device_particle_x, static_cast<int>(particle_count),
        thrust::raw_pointer_cast(storage.node_keys.data()), thrust::raw_pointer_cast(storage.node_indices.data()),
        static_cast<int>(sorted_velocity_keys.size()),
        raw_velocity_nodes.nodes,
        resident_velocity_nodes ? static_cast<int>(resident_velocity_nodes->size) : 0,
        device_fallback_keys, device_fallback_velocity,
        static_cast<int>(fallback_node_count), raw.values,
        thrust::raw_pointer_cast(storage.pressure_keys.data()),
        thrust::raw_pointer_cast(storage.pressure_indices.data()),
        static_cast<int>(sorted_pressure_keys.size()), raw_pressure_nodes.nodes,
        resident_pressure_nodes ? static_cast<int>(resident_pressure_nodes->size) : 0,
        raw_pressure.values, static_cast<int>(device_pressure.size), dx, dt,
        thrust::raw_pointer_cast(storage.output_x.data()), thrust::raw_pointer_cast(storage.output_v.data()),
        thrust::raw_pointer_cast(storage.output_grad_v.data()),
        thrust::raw_pointer_cast(storage.output_pressure.data()));
    checkCuda(cudaGetLastError(), "fluid velocity transfer kernel launch");
    checkCuda(cudaDeviceSynchronize(), "fluid velocity transfer synchronize");
    device_result = {
        thrust::raw_pointer_cast(storage.output_x.data()),
        thrust::raw_pointer_cast(storage.output_v.data()),
        thrust::raw_pointer_cast(storage.output_grad_v.data()),
        thrust::raw_pointer_cast(storage.output_pressure.data()),
        particle_count};
    const auto kernel_end = std::chrono::high_resolution_clock::now();
    local.host_download_bytes = downloadVelocityTransferHostOutput(
        storage, particle_count, download_mask, result);
    const auto download_end = std::chrono::high_resolution_clock::now();
    local.reallocations = storage.reallocations;
    local.particle_upload_ms = std::chrono::duration<double, std::milli>(particle_upload_end - upload_begin).count();
    local.node_upload_ms = std::chrono::duration<double, std::milli>(node_upload_end - particle_upload_end).count();
    local.g2p_ms = std::chrono::duration<double, std::milli>(kernel_end - node_upload_end).count();
    local.download_ms = std::chrono::duration<double, std::milli>(download_end - kernel_end).count();
    local.total_ms = std::chrono::duration<double, std::milli>(download_end - total_begin).count();
    if (stats) *stats = local;
}

void downloadPersistentFluidVelocityTransfer3D(
    PersistentVelocityTransfer3D& context,
    VelocityTransferDownloadMask download_mask,
    VelocityTransferHostView& result,
    VelocityTransferDownloadStats* stats)
{
    const auto begin = std::chrono::high_resolution_clock::now();
    auto& storage = getVelocityTransferStorage(context);
    const std::size_t particle_count = storage.output_x.size();
    if (storage.output_v.size() != particle_count ||
        storage.output_grad_v.size() != particle_count ||
        storage.output_pressure.size() != particle_count)
        throw std::runtime_error(
            "fluid velocity-transfer output is incomplete");
    const std::size_t bytes = downloadVelocityTransferHostOutput(
        storage, particle_count, download_mask, result);
    const auto end = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->particles = particle_count;
        stats->bytes = bytes;
        stats->download_ms = std::chrono::duration<double, std::milli>(
            end - begin).count();
    }
}

void downloadFluidVelocityTransferDeviceView3D(
    PersistentVelocityTransfer3D& context,
    const VelocityTransferDeviceView& view,
    VelocityTransferDownloadMask download_mask,
    VelocityTransferHostView& result,
    VelocityTransferDownloadStats* stats)
{
    if (view.size != 0 && (!view.x || !view.v || !view.grad_v || !view.pressure))
        throw std::runtime_error(
            "fluid velocity-transfer device view is incomplete");
    const auto begin = std::chrono::high_resolution_clock::now();
    auto& storage = getVelocityTransferStorage(context);
    const std::size_t bytes = downloadVelocityTransferHostOutput(
        storage, view.size, download_mask, result, &view);
    const auto end = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->particles = view.size;
        stats->bytes = bytes;
        stats->download_ms = std::chrono::duration<double, std::milli>(
            end - begin).count();
    }
}

void projectFluidBoundaryCollision3D(
    PersistentVelocityTransfer3D& context,
    const BoxShellCollision3D& collision, double dx,
    FluidBoundaryProjectionStats* stats)
{
    if (!(dx > 0.0))
        throw std::runtime_error("invalid fluid boundary projection dx");
    const auto begin = std::chrono::high_resolution_clock::now();
    auto& storage = getVelocityTransferStorage(context);
    FluidBoundaryProjectionStats local;
    local.particles = storage.output_x.size();
    if (storage.output_v.size() != local.particles)
        throw std::runtime_error(
            "fluid boundary projection particle state is incomplete");
    if (collision.valid && local.particles != 0) {
        resizeVelocityTransfer(storage.boundary_projection_count, 1, storage);
        checkCuda(cudaMemset(
                      thrust::raw_pointer_cast(
                          storage.boundary_projection_count.data()),
                      0, sizeof(unsigned long long)),
            "fluid boundary projection counter clear");
        constexpr int block_size = 256;
        const int blocks = static_cast<int>(
            (local.particles + block_size - 1) / block_size);
        projectFluidBoundaryCollisionKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.output_x.data()),
            thrust::raw_pointer_cast(storage.output_v.data()),
            static_cast<int>(local.particles), collision, dx,
            thrust::raw_pointer_cast(
                storage.boundary_projection_count.data()));
        checkCuda(cudaGetLastError(),
            "fluid boundary projection kernel launch");
        unsigned long long projected = 0;
        checkCuda(cudaMemcpy(&projected,
                      thrust::raw_pointer_cast(
                          storage.boundary_projection_count.data()),
                      sizeof(projected), cudaMemcpyDeviceToHost),
            "fluid boundary projection counter download");
        local.projected_particles = static_cast<std::size_t>(projected);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    local.projection_ms = std::chrono::duration<double, std::milli>(
        end - begin).count();
    if (stats)
        *stats = local;
}

void reduceFluidParticleState3D(
    const VelocityTransferDeviceView& view,
    double domain_min, double domain_max,
    FluidParticleStateStats& result)
{
    if (view.size != 0 && (!view.x || !view.v))
        throw std::runtime_error(
            "fluid particle-state reduction requires resident X/V");
    if (domain_min > domain_max)
        throw std::runtime_error(
            "fluid particle-state reduction has invalid domain bounds");
    const auto begin = std::chrono::high_resolution_clock::now();
    result = {};
    result.particles = view.size;
    if (view.size != 0) {
        const double inf = std::numeric_limits<double>::infinity();
        const FluidParticleReduction init{
            Vec3d{inf, inf, inf}, Vec3d{-inf, -inf, -inf}, 0.0, 0ull};
        const FluidParticleReduction reduced = thrust::transform_reduce(
            thrust::counting_iterator<std::size_t>(0),
            thrust::counting_iterator<std::size_t>(view.size),
            FluidParticleTransform{
                static_cast<const Vec3d*>(view.x),
                static_cast<const Vec3d*>(view.v), domain_min, domain_max},
            init, FluidParticleReduce{});
        result.min_corner = reduced.min_corner;
        result.max_corner = reduced.max_corner;
        result.max_speed = std::sqrt(reduced.max_speed_squared);
        result.outside_particles =
            static_cast<std::size_t>(reduced.outside_particles);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    result.reduction_ms = std::chrono::duration<double, std::milli>(
        end - begin).count();
}

void patchFluidVelocityTransferParticles3D(
    PersistentVelocityTransfer3D& context,
    const std::vector<int>& particle_indices,
    const std::vector<Vec3d>& velocity,
    const std::vector<Mat3d>& grad_velocity,
    VelocityTransferPatchStats* stats,
    const std::vector<Vec3d>& position)
{
    if (particle_indices.size() != velocity.size() ||
        (!grad_velocity.empty() &&
            particle_indices.size() != grad_velocity.size()) ||
        (!position.empty() && particle_indices.size() != position.size()))
        throw std::runtime_error(
            "fluid velocity-transfer patch input size mismatch");
    auto& storage = getVelocityTransferStorage(context);
    const std::size_t particle_count = storage.output_v.size();
    if (storage.output_grad_v.size() != particle_count ||
        (!position.empty() && storage.output_x.size() != particle_count))
        throw std::runtime_error(
            "fluid velocity-transfer patch requires resident output");
    for (int particle : particle_indices)
        if (particle < 0 ||
            static_cast<std::size_t>(particle) >= particle_count)
            throw std::runtime_error(
                "fluid velocity-transfer patch index out of range");

    const auto total_begin = std::chrono::high_resolution_clock::now();
    const auto upload_begin = total_begin;
    resizeVelocityTransfer(storage.patch_indices, particle_indices.size(),
        storage);
    resizeVelocityTransfer(storage.patch_velocity, velocity.size(), storage);
    resizeVelocityTransfer(storage.patch_grad_v, grad_velocity.size(), storage);
    resizeVelocityTransfer(storage.patch_x, position.size(), storage);
    thrust::copy(particle_indices.begin(), particle_indices.end(),
        storage.patch_indices.begin());
    thrust::copy(velocity.begin(), velocity.end(), storage.patch_velocity.begin());
    if (!grad_velocity.empty())
        thrust::copy(grad_velocity.begin(), grad_velocity.end(),
            storage.patch_grad_v.begin());
    if (!position.empty())
        thrust::copy(position.begin(), position.end(), storage.patch_x.begin());
    const auto upload_end = std::chrono::high_resolution_clock::now();
    const auto patch_begin = upload_end;
    if (!particle_indices.empty()) {
        constexpr int block_size = 256;
        const int blocks = static_cast<int>(
            (particle_indices.size() + block_size - 1) / block_size);
        patchFluidVelocityTransferParticlesKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.patch_indices.data()),
            thrust::raw_pointer_cast(storage.patch_velocity.data()),
            grad_velocity.empty() ? nullptr
                                  : thrust::raw_pointer_cast(
                                        storage.patch_grad_v.data()),
            position.empty() ? nullptr
                              : thrust::raw_pointer_cast(storage.patch_x.data()),
            static_cast<int>(particle_indices.size()),
            static_cast<int>(particle_count),
            thrust::raw_pointer_cast(storage.output_v.data()),
            thrust::raw_pointer_cast(storage.output_grad_v.data()),
            position.empty() ? nullptr
                              : thrust::raw_pointer_cast(storage.output_x.data()));
        checkCuda(cudaGetLastError(),
            "fluid velocity-transfer particle patch launch");
        checkCuda(cudaDeviceSynchronize(),
            "fluid velocity-transfer particle patch synchronize");
    }
    const auto patch_end = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->particles = particle_indices.size();
        stats->reallocations = storage.reallocations;
        stats->upload_ms = std::chrono::duration<double, std::milli>(
            upload_end - upload_begin).count();
        stats->patch_ms = std::chrono::duration<double, std::milli>(
            patch_end - patch_begin).count();
        stats->total_ms = std::chrono::duration<double, std::milli>(
            patch_end - total_begin).count();
    }
}

void rebuildPersistentPressureGridPages3D(
    PersistentPressureGrid3D& context,
    const std::vector<Vec3d>& page_positions, double dx,
    std::vector<GridKey>& page_base_keys, PressureGridFieldStats* stats,
    bool download_base_keys, const void* resident_page_positions,
    std::size_t resident_page_count)
{
    if (!(dx > 0.0))
        throw std::runtime_error("persistent pressure grid requires positive dx");
    if ((resident_page_positions == nullptr) != (resident_page_count == 0))
        throw std::runtime_error("invalid resident pressure page input");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    PressureGridFieldStats local_stats;
    const bool use_resident_pages = resident_page_positions != nullptr;
    const std::size_t page_source_count = use_resident_pages
        ? resident_page_count : page_positions.size();
    local_stats.page_sources = page_source_count;
    local_stats.resident_sources = use_resident_pages ? page_source_count : 0;
    page_base_keys.clear();
    auto& storage = getPressureGridStorage(context);
    if (page_source_count == 0) {
        storage.page_source_positions = nullptr;
        storage.page_source_count = 0;
        storage.pages.candidate_keys.clear();
        if (stats)
            *stats = local_stats;
        return;
    }

    constexpr int block_size = 256;
    const int block_count = static_cast<int>((page_source_count + block_size - 1)
        / block_size);
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    resizeLinearScatter(storage.pages.candidate_keys, page_source_count, storage.pages);
    if (use_resident_pages) {
        storage.page_source_positions =
            static_cast<const Vec3d*>(resident_page_positions);
    } else {
        resizeLinearScatter(storage.pages.positions, page_source_count, storage.pages);
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.pages.positions.data()),
                      page_positions.data(), page_source_count * sizeof(Vec3d),
                      cudaMemcpyHostToDevice),
            "persistent pressure grid page position upload");
        storage.page_source_positions =
            thrust::raw_pointer_cast(storage.pages.positions.data());
    }
    storage.page_source_count = page_source_count;
    checkCuda(cudaDeviceSynchronize(),
        "persistent pressure grid page upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    const auto key_begin = std::chrono::high_resolution_clock::now();
    generatePressureKeysKernel<<<block_count, block_size>>>(
        storage.page_source_positions, static_cast<int>(page_source_count),
        dx, /*halfDxAnchor*/ 0,
        thrust::raw_pointer_cast(storage.pages.candidate_keys.data()));
    checkCuda(cudaGetLastError(), "persistent pressure grid page key launch");
    checkCuda(cudaDeviceSynchronize(),
        "persistent pressure grid page key synchronize");
    const auto key_end = std::chrono::high_resolution_clock::now();

    const auto download_begin = std::chrono::high_resolution_clock::now();
    if (download_base_keys) {
        page_base_keys.resize(storage.pages.candidate_keys.size());
        checkCuda(cudaMemcpy(page_base_keys.data(),
                      thrust::raw_pointer_cast(storage.pages.candidate_keys.data()),
                      page_base_keys.size() * sizeof(GridKey), cudaMemcpyDeviceToHost),
            "persistent pressure grid page key download");
    }
    const auto download_end = std::chrono::high_resolution_clock::now();
    local_stats.reallocations = pressureGridReallocations(storage);
    local_stats.upload_ms = std::chrono::duration<double, std::milli>(
        upload_end - upload_begin).count();
    local_stats.key_generation_ms = std::chrono::duration<double, std::milli>(
        key_end - key_begin).count();
    local_stats.download_ms = std::chrono::duration<double, std::milli>(
        download_end - download_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(
        download_end - total_begin).count();
    if (stats)
        *stats = local_stats;
}

void buildPersistentPressureGridVirtualBlocks3D(
    PersistentPressureGrid3D& context, int block_x, int block_y, int block_z,
    VirtualBlockOrderResult& result, VirtualBlockOrderStats* stats,
    bool download_result)
{
    if (block_x <= 0 || block_y <= 0 || block_z <= 0)
        throw std::runtime_error("persistent pressure virtual block has invalid size");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    VirtualBlockOrderStats local_stats;
    auto& storage = getPressureGridStorage(context);
    const std::size_t source_count = storage.pages.candidate_keys.size();
    local_stats.sources = source_count;
    result.page_keys.clear();
    result.touched_keys.clear();
    if (source_count == 0) {
        local_stats.reallocations = pressureGridReallocations(storage);
        if (stats)
            *stats = local_stats;
        return;
    }

    constexpr int candidates_per_source = 28;
    constexpr int block_size = 256;
    const std::size_t candidate_count = source_count * candidates_per_source;
    const int source_blocks = static_cast<int>((source_count + block_size - 1)
        / block_size);
    const auto ordering_begin = std::chrono::high_resolution_clock::now();
    resizePressureGrid(storage.virtual_page_candidates, candidate_count, storage);
    resizePressureGrid(storage.virtual_page_ranks, candidate_count, storage);
    generateVirtualBlockCandidatesKernel<<<source_blocks, block_size>>>(
        thrust::raw_pointer_cast(storage.pages.candidate_keys.data()),
        static_cast<int>(source_count), block_x, block_y, block_z,
        thrust::raw_pointer_cast(storage.virtual_page_candidates.data()),
        thrust::raw_pointer_cast(storage.virtual_page_ranks.data()));
    checkCuda(cudaGetLastError(), "persistent pressure virtual block candidate launch");

    auto candidate_zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
        storage.virtual_page_candidates.begin(), storage.virtual_page_ranks.begin()));
    auto candidate_zip_end = candidate_zip_begin + candidate_count;
    const auto valid_zip_end = thrust::remove_if(candidate_zip_begin, candidate_zip_end,
        IsInvalidVirtualPage{});
    const std::size_t valid_candidate_count = static_cast<std::size_t>(
        valid_zip_end - candidate_zip_begin);
    storage.virtual_page_candidates.resize(valid_candidate_count);
    storage.virtual_page_ranks.resize(valid_candidate_count);

    // Sort by page key, reduce each page to its first activation rank, then
    // sort by that rank. This is the exact page insertion order of the old
    // MpmGrid::activatePages() stream without a host hash table.
    thrust::sort_by_key(storage.virtual_page_candidates.begin(),
        storage.virtual_page_candidates.end(), storage.virtual_page_ranks.begin());
    resizePressureGrid(storage.virtual_page_keys, valid_candidate_count, storage);
    resizePressureGrid(storage.virtual_page_reduced_ranks,
        valid_candidate_count, storage);
    const auto reduced_end = thrust::reduce_by_key(
        storage.virtual_page_candidates.begin(), storage.virtual_page_candidates.end(),
        storage.virtual_page_ranks.begin(), storage.virtual_page_keys.begin(),
        storage.virtual_page_reduced_ranks.begin(), thrust::equal_to<GridKey>(),
        thrust::minimum<std::uint64_t>());
    const std::size_t page_count = static_cast<std::size_t>(
        reduced_end.first - storage.virtual_page_keys.begin());
    storage.virtual_page_keys.resize(page_count);
    storage.virtual_page_reduced_ranks.resize(page_count);
    thrust::sort_by_key(storage.virtual_page_reduced_ranks.begin(),
        storage.virtual_page_reduced_ranks.end(),
        storage.virtual_page_keys.begin());
    checkCuda(cudaDeviceSynchronize(), "persistent pressure virtual block order synchronize");
    const auto ordering_end = std::chrono::high_resolution_clock::now();

    const std::size_t nodes_per_page = static_cast<std::size_t>(block_x) * block_y * block_z;
    resizePressureGrid(storage.virtual_touched_keys, page_count * nodes_per_page, storage);
    const int touched_blocks = static_cast<int>((storage.virtual_touched_keys.size()
        + block_size - 1) / block_size);
    expandVirtualBlockKeysKernel<<<touched_blocks, block_size>>>(
        thrust::raw_pointer_cast(storage.virtual_page_keys.data()),
        static_cast<int>(page_count), block_x, block_y, block_z,
        thrust::raw_pointer_cast(storage.virtual_touched_keys.data()));
    checkCuda(cudaGetLastError(), "persistent pressure virtual block expand launch");
    checkCuda(cudaDeviceSynchronize(), "persistent pressure virtual block expand synchronize");
    const auto expansion_end = std::chrono::high_resolution_clock::now();

    const auto download_begin = std::chrono::high_resolution_clock::now();
    if (download_result) {
        result.page_keys.resize(page_count);
        result.touched_keys.resize(storage.virtual_touched_keys.size());
        checkCuda(cudaMemcpy(result.page_keys.data(),
                      thrust::raw_pointer_cast(storage.virtual_page_keys.data()),
                      result.page_keys.size() * sizeof(GridKey), cudaMemcpyDeviceToHost),
            "persistent pressure virtual page download");
        checkCuda(cudaMemcpy(result.touched_keys.data(),
                      thrust::raw_pointer_cast(storage.virtual_touched_keys.data()),
                      result.touched_keys.size() * sizeof(GridKey), cudaMemcpyDeviceToHost),
            "persistent pressure virtual touched-key download");
    }
    const auto download_end = std::chrono::high_resolution_clock::now();

    local_stats.candidate_pages = valid_candidate_count;
    local_stats.virtual_blocks = page_count;
    local_stats.touched_nodes = storage.virtual_touched_keys.size();
    local_stats.reallocations = pressureGridReallocations(storage);
    local_stats.ordering_ms = std::chrono::duration<double, std::milli>(
        ordering_end - ordering_begin).count();
    local_stats.expansion_ms = std::chrono::duration<double, std::milli>(
        expansion_end - ordering_end).count();
    local_stats.download_ms = std::chrono::duration<double, std::milli>(
        download_end - download_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(
        download_end - total_begin).count();
    if (stats)
        *stats = local_stats;
}

void extractPersistentPressureGridDofSourceKeys3D(
    PersistentPressureGrid3D& context, double dx,
    std::vector<GridKey>& dof_source_keys, PressureGridFieldStats* stats,
    bool download_keys)
{
    if (!(dx > 0.0))
        throw std::runtime_error("persistent pressure grid requires positive dx");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    PressureGridFieldStats local_stats;
    auto& storage = getPressureGridStorage(context);
    if (storage.page_source_count == 0) {
        dof_source_keys.clear();
        local_stats.reallocations = pressureGridReallocations(storage);
        if (stats)
            *stats = local_stats;
        return;
    }

    constexpr int block_size = 256;
    const std::size_t source_count = storage.page_source_count;
    const int block_count = static_cast<int>((source_count + block_size - 1)
        / block_size);
    const auto key_begin = std::chrono::high_resolution_clock::now();
    resizePressureGrid(storage.dof_source_keys, source_count, storage);
    generatePressureKeysKernel<<<block_count, block_size>>>(
        storage.page_source_positions,
        static_cast<int>(source_count), dx, /*halfDxAnchor*/ 1,
        thrust::raw_pointer_cast(storage.dof_source_keys.data()));
    checkCuda(cudaGetLastError(), "persistent pressure grid DOF-source key launch");
    thrust::sort(storage.dof_source_keys.begin(), storage.dof_source_keys.end());
    const auto unique_end = thrust::unique(storage.dof_source_keys.begin(),
        storage.dof_source_keys.end());
    storage.dof_source_keys.erase(unique_end, storage.dof_source_keys.end());
    checkCuda(cudaDeviceSynchronize(),
        "persistent pressure grid DOF-source key synchronize");
    const auto key_end = std::chrono::high_resolution_clock::now();

    const auto download_begin = std::chrono::high_resolution_clock::now();
    if (download_keys) {
        dof_source_keys.resize(storage.dof_source_keys.size());
        checkCuda(cudaMemcpy(dof_source_keys.data(),
                      thrust::raw_pointer_cast(storage.dof_source_keys.data()),
                      dof_source_keys.size() * sizeof(GridKey), cudaMemcpyDeviceToHost),
            "persistent pressure grid DOF-source key download");
    }
    const auto download_end = std::chrono::high_resolution_clock::now();
    local_stats.page_sources = source_count;
    local_stats.dofs = storage.dof_source_keys.size();
    local_stats.reallocations = pressureGridReallocations(storage);
    local_stats.key_generation_ms = std::chrono::duration<double, std::milli>(
        key_end - key_begin).count();
    local_stats.download_ms = std::chrono::duration<double, std::milli>(
        download_end - download_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(
        download_end - total_begin).count();
    if (stats)
        *stats = local_stats;
}

void classifyPersistentPressureGridSolid3D(
    PersistentPressureGrid3D& context,
    const GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D& level_set,
    const BoxShellCollision3D& collision, double dx,
    std::vector<unsigned char>& solid_flags,
    PressureGridFieldStats* stats, bool download_flags)
{
    if (!(dx > 0.0) || !level_set.phi || !(level_set.h > 0.0))
        throw std::runtime_error(
            "resident pressure solid classification has invalid input");
    const std::size_t expected_voxels =
        static_cast<std::size_t>(level_set.shape[0]) *
        static_cast<std::size_t>(level_set.shape[1]) *
        static_cast<std::size_t>(level_set.shape[2]);
    if (level_set.shape[0] <= 0 || level_set.shape[1] <= 0 ||
        level_set.shape[2] <= 0 || expected_voxels != level_set.voxels)
        throw std::runtime_error(
            "resident pressure solid classification level-set size mismatch");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    PressureGridFieldStats local_stats;
    auto& storage = getPressureGridStorage(context);
    const std::size_t touched_count = storage.virtual_touched_keys.size();
    resizePressureGrid(storage.solid_flags, touched_count, storage);
    if (touched_count != 0) {
        constexpr int block_size = 256;
        const int blocks = static_cast<int>(
            (touched_count + block_size - 1) / block_size);
        classifyPressureSolidKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.virtual_touched_keys.data()),
            static_cast<int>(touched_count), level_set, collision, dx,
            thrust::raw_pointer_cast(storage.solid_flags.data()));
        checkCuda(cudaGetLastError(),
            "persistent pressure solid classification launch");
    }
    const auto download_begin = std::chrono::high_resolution_clock::now();
    if (download_flags) {
        solid_flags.resize(touched_count);
        if (touched_count != 0)
            checkCuda(cudaMemcpy(solid_flags.data(),
                          thrust::raw_pointer_cast(storage.solid_flags.data()),
                          touched_count * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost),
                "persistent pressure solid flag download");
    } else {
        solid_flags.clear();
        checkCuda(cudaDeviceSynchronize(),
            "persistent pressure solid classification synchronize");
    }
    const auto download_end = std::chrono::high_resolution_clock::now();
    local_stats.touched_nodes = touched_count;
    local_stats.resident_solid_flags = true;
    local_stats.reallocations = pressureGridReallocations(storage);
    local_stats.download_ms = std::chrono::duration<double, std::milli>(
        download_end - download_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(
        download_end - total_begin).count();
    if (stats)
        *stats = local_stats;
}

void scatterPersistentPressureGridDensity3D(
    PersistentPressureGrid3D& context,
    const std::vector<Vec3d>& density_positions, double density_value,
    double dx, ProbeGridResult& density_result,
    PressureGridFieldStats* stats, bool download_result)
{
    if (!(dx > 0.0))
        throw std::runtime_error("persistent pressure grid requires positive dx");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    PressureGridFieldStats local_stats;
    local_stats.density_sources = density_positions.size();
    density_result.keys.clear();
    density_result.mass.clear();
    if (density_positions.empty()) {
        if (stats)
            *stats = local_stats;
        return;
    }

    auto& storage = getPressureGridStorage(context);
    constexpr int block_size = 256;
    const int block_count = static_cast<int>((density_positions.size() + block_size - 1)
        / block_size);
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    resizeLinearScatter(storage.density.positions, density_positions.size(), storage.density);
    resizeLinearScatter(storage.density.candidate_keys, density_positions.size() * 8,
        storage.density);
    checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.density.positions.data()),
                  density_positions.data(), density_positions.size() * sizeof(Vec3d),
                  cudaMemcpyHostToDevice),
        "persistent pressure grid density position upload");
    checkCuda(cudaDeviceSynchronize(),
        "persistent pressure grid density upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    const auto key_begin = std::chrono::high_resolution_clock::now();
    generateLinearVelocityKeysKernel<<<block_count, block_size>>>(
        thrust::raw_pointer_cast(storage.density.positions.data()),
        static_cast<int>(density_positions.size()), dx, /*halfDxAnchor*/ 0,
        thrust::raw_pointer_cast(storage.density.candidate_keys.data()));
    checkCuda(cudaGetLastError(), "persistent pressure grid density key launch");
    resizeLinearScatter(storage.density.active_keys,
        storage.density.candidate_keys.size(), storage.density);
    thrust::copy(storage.density.candidate_keys.begin(), storage.density.candidate_keys.end(),
        storage.density.active_keys.begin());
    thrust::sort(storage.density.active_keys.begin(), storage.density.active_keys.end());
    const auto unique_end = thrust::unique(storage.density.active_keys.begin(),
        storage.density.active_keys.end());
    storage.density.active_keys.erase(unique_end, storage.density.active_keys.end());
    checkCuda(cudaGetLastError(), "persistent pressure grid density key sort");
    checkCuda(cudaDeviceSynchronize(),
        "persistent pressure grid density key synchronize");
    const auto key_end = std::chrono::high_resolution_clock::now();

    const auto scatter_begin = std::chrono::high_resolution_clock::now();
    resizeLinearScatter(storage.density.grid_mass, storage.density.active_keys.size(),
        storage.density);
    thrust::fill(storage.density.grid_mass.begin(), storage.density.grid_mass.end(), 0.0);
    scatterUniformLinearScalarKernel<<<block_count, block_size>>>(
        thrust::raw_pointer_cast(storage.density.positions.data()),
        static_cast<int>(density_positions.size()), density_value, dx,
        /*halfDxAnchor*/ 0,
        thrust::raw_pointer_cast(storage.density.active_keys.data()),
        static_cast<int>(storage.density.active_keys.size()),
        thrust::raw_pointer_cast(storage.density.grid_mass.data()));
    checkCuda(cudaGetLastError(), "persistent pressure grid density launch");
    checkCuda(cudaDeviceSynchronize(),
        "persistent pressure grid density synchronize");
    const auto scatter_end = std::chrono::high_resolution_clock::now();

    const auto download_begin = std::chrono::high_resolution_clock::now();
    if (download_result) {
        density_result.keys.resize(storage.density.active_keys.size());
        density_result.mass.resize(storage.density.grid_mass.size());
        checkCuda(cudaMemcpy(density_result.keys.data(),
                      thrust::raw_pointer_cast(storage.density.active_keys.data()),
                      density_result.keys.size() * sizeof(GridKey), cudaMemcpyDeviceToHost),
            "persistent pressure grid density key download");
        checkCuda(cudaMemcpy(density_result.mass.data(),
                      thrust::raw_pointer_cast(storage.density.grid_mass.data()),
                      density_result.mass.size() * sizeof(double), cudaMemcpyDeviceToHost),
            "persistent pressure grid density value download");
    }
    else {
        density_result.keys.clear();
        density_result.mass.clear();
    }
    const auto download_end = std::chrono::high_resolution_clock::now();
    local_stats.density_nodes = storage.density.active_keys.size();
    local_stats.reallocations = pressureGridReallocations(storage);
    local_stats.upload_ms = std::chrono::duration<double, std::milli>(
        upload_end - upload_begin).count();
    local_stats.key_generation_ms = std::chrono::duration<double, std::milli>(
        key_end - key_begin).count();
    local_stats.density_scatter_ms = std::chrono::duration<double, std::milli>(
        scatter_end - scatter_begin).count();
    local_stats.download_ms = std::chrono::duration<double, std::milli>(
        download_end - download_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(
        download_end - total_begin).count();
    if (stats)
        *stats = local_stats;
}

void scatterPersistentPressureGridDensitySplit3D(
    PersistentPressureGrid3D& context,
    const std::vector<Vec3d>& static_density_positions,
    const std::vector<Vec3d>& dynamic_density_positions,
    double density_value, double dx, ProbeGridResult& density_result,
    PressureGridFieldStats* stats, bool download_result,
    const void* resident_dynamic_positions,
    std::size_t resident_dynamic_count,
    const void* second_resident_dynamic_positions,
    std::size_t second_resident_dynamic_count)
{
    if (!(dx > 0.0))
        throw std::runtime_error("persistent pressure grid requires positive dx");
    if ((resident_dynamic_positions == nullptr) !=
        (resident_dynamic_count == 0))
        throw std::runtime_error("invalid resident pressure density input");
    if ((second_resident_dynamic_positions == nullptr) !=
        (second_resident_dynamic_count == 0))
        throw std::runtime_error(
            "invalid second resident pressure density input");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    PressureGridFieldStats local_stats;
    const std::size_t source_count = static_density_positions.size()
        + resident_dynamic_count + second_resident_dynamic_count
        + dynamic_density_positions.size();
    local_stats.density_sources = source_count;
    local_stats.resident_sources = resident_dynamic_count
        + second_resident_dynamic_count;
    density_result.keys.clear();
    density_result.mass.clear();
    if (source_count == 0) {
        if (stats)
            *stats = local_stats;
        return;
    }

    auto& storage = getPressureGridStorage(context);
    constexpr int block_size = 256;
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    const Vec3d* static_data = static_density_positions.empty()
        ? nullptr : static_density_positions.data();
    const bool refresh_static = static_data != storage.density_static_host_data
        || static_density_positions.size() != storage.density_static_count;
    if (refresh_static) {
        resizePressureGrid(storage.density_static_positions,
            static_density_positions.size(), storage);
        if (!static_density_positions.empty())
            checkCuda(cudaMemcpy(
                          thrust::raw_pointer_cast(
                              storage.density_static_positions.data()),
                          static_density_positions.data(),
                          static_density_positions.size() * sizeof(Vec3d),
                          cudaMemcpyHostToDevice),
                "persistent pressure grid static density upload");
        storage.density_static_host_data = static_data;
        storage.density_static_count = static_density_positions.size();
    }
    // Keep the dynamic contributors in their legacy contiguous order. The
    // fluid and solid-owner prefixes may already be resident; copy those
    // device-to-device and upload only the remaining host-owned suffix. This
    // preserves the original key/scatter launch ordering exactly.
    const std::size_t dynamic_count = resident_dynamic_count
        + second_resident_dynamic_count + dynamic_density_positions.size();
    resizeLinearScatter(storage.density.positions,
        dynamic_count, storage.density);
    resizeLinearScatter(storage.density.candidate_keys, source_count * 8,
        storage.density);
    if (resident_dynamic_count != 0)
        checkCuda(cudaMemcpy(
                      thrust::raw_pointer_cast(storage.density.positions.data()),
                      resident_dynamic_positions,
                      resident_dynamic_count * sizeof(Vec3d),
                      cudaMemcpyDeviceToDevice),
            "persistent pressure grid resident density copy");
    if (second_resident_dynamic_count != 0)
        checkCuda(cudaMemcpy(
                      thrust::raw_pointer_cast(storage.density.positions.data())
                          + resident_dynamic_count,
                      second_resident_dynamic_positions,
                      second_resident_dynamic_count * sizeof(Vec3d),
                      cudaMemcpyDeviceToDevice),
            "persistent pressure grid second resident density copy");
    if (!dynamic_density_positions.empty())
        checkCuda(cudaMemcpy(
                      thrust::raw_pointer_cast(storage.density.positions.data())
                          + resident_dynamic_count
                          + second_resident_dynamic_count,
                      dynamic_density_positions.data(),
                      dynamic_density_positions.size() * sizeof(Vec3d),
                      cudaMemcpyHostToDevice),
            "persistent pressure grid dynamic density upload");
    checkCuda(cudaDeviceSynchronize(),
        "persistent pressure grid split density upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    const auto key_begin = std::chrono::high_resolution_clock::now();
    // Keep static boundary positions resident in their own device buffer.
    // Generating and scattering the two contiguous source ranges separately
    // preserves the same union-of-keys/density-field computation as a merged
    // input array, while avoiding an otherwise mandatory 100+ MB D2D copy at
    // every PositionCorrector substep.
    if (!static_density_positions.empty()) {
        const int static_block_count = static_cast<int>(
            (static_density_positions.size() + block_size - 1) / block_size);
        generateLinearVelocityKeysKernel<<<static_block_count, block_size>>>(
            thrust::raw_pointer_cast(storage.density_static_positions.data()),
            static_cast<int>(static_density_positions.size()), dx,
            /*halfDxAnchor*/ 0,
            thrust::raw_pointer_cast(storage.density.candidate_keys.data()));
        checkCuda(cudaGetLastError(),
            "persistent pressure grid static density key launch");
    }
    if (dynamic_count != 0) {
        const int dynamic_block_count = static_cast<int>(
            (dynamic_count + block_size - 1) / block_size);
        generateLinearVelocityKeysKernel<<<dynamic_block_count, block_size>>>(
            thrust::raw_pointer_cast(storage.density.positions.data()),
            static_cast<int>(dynamic_count), dx,
            /*halfDxAnchor*/ 0,
            thrust::raw_pointer_cast(storage.density.candidate_keys.data())
                + static_density_positions.size() * 8);
        checkCuda(cudaGetLastError(),
            "persistent pressure grid dynamic density key launch");
    }
    resizeLinearScatter(storage.density.active_keys,
        storage.density.candidate_keys.size(), storage.density);
    thrust::copy(storage.density.candidate_keys.begin(),
        storage.density.candidate_keys.end(), storage.density.active_keys.begin());
    thrust::sort(storage.density.active_keys.begin(), storage.density.active_keys.end());
    const auto unique_end = thrust::unique(storage.density.active_keys.begin(),
        storage.density.active_keys.end());
    storage.density.active_keys.erase(unique_end, storage.density.active_keys.end());
    checkCuda(cudaGetLastError(), "persistent pressure grid split density key sort");
    checkCuda(cudaDeviceSynchronize(),
        "persistent pressure grid split density key synchronize");
    const auto key_end = std::chrono::high_resolution_clock::now();

    const auto scatter_begin = std::chrono::high_resolution_clock::now();
    resizeLinearScatter(storage.density.grid_mass, storage.density.active_keys.size(),
        storage.density);
    thrust::fill(storage.density.grid_mass.begin(), storage.density.grid_mass.end(), 0.0);
    if (!static_density_positions.empty()) {
        const int static_block_count = static_cast<int>(
            (static_density_positions.size() + block_size - 1) / block_size);
        scatterUniformLinearScalarKernel<<<static_block_count, block_size>>>(
            thrust::raw_pointer_cast(storage.density_static_positions.data()),
            static_cast<int>(static_density_positions.size()), density_value, dx,
            /*halfDxAnchor*/ 0,
            thrust::raw_pointer_cast(storage.density.active_keys.data()),
            static_cast<int>(storage.density.active_keys.size()),
            thrust::raw_pointer_cast(storage.density.grid_mass.data()));
        checkCuda(cudaGetLastError(),
            "persistent pressure grid static density scatter launch");
    }
    if (dynamic_count != 0) {
        const int dynamic_block_count = static_cast<int>(
            (dynamic_count + block_size - 1) / block_size);
        scatterUniformLinearScalarKernel<<<dynamic_block_count, block_size>>>(
            thrust::raw_pointer_cast(storage.density.positions.data()),
            static_cast<int>(dynamic_count), density_value, dx,
            /*halfDxAnchor*/ 0,
            thrust::raw_pointer_cast(storage.density.active_keys.data()),
            static_cast<int>(storage.density.active_keys.size()),
            thrust::raw_pointer_cast(storage.density.grid_mass.data()));
        checkCuda(cudaGetLastError(),
            "persistent pressure grid dynamic density scatter launch");
    }
    checkCuda(cudaDeviceSynchronize(),
        "persistent pressure grid split density synchronize");
    const auto scatter_end = std::chrono::high_resolution_clock::now();

    const auto download_begin = std::chrono::high_resolution_clock::now();
    if (download_result) {
        density_result.keys.resize(storage.density.active_keys.size());
        density_result.mass.resize(storage.density.grid_mass.size());
        checkCuda(cudaMemcpy(density_result.keys.data(),
                      thrust::raw_pointer_cast(storage.density.active_keys.data()),
                      density_result.keys.size() * sizeof(GridKey), cudaMemcpyDeviceToHost),
            "persistent pressure grid split density key download");
        checkCuda(cudaMemcpy(density_result.mass.data(),
                      thrust::raw_pointer_cast(storage.density.grid_mass.data()),
                      density_result.mass.size() * sizeof(double), cudaMemcpyDeviceToHost),
            "persistent pressure grid split density value download");
    }
    const auto download_end = std::chrono::high_resolution_clock::now();
    local_stats.density_nodes = storage.density.active_keys.size();
    local_stats.reallocations = pressureGridReallocations(storage);
    local_stats.upload_ms = std::chrono::duration<double, std::milli>(
        upload_end - upload_begin).count();
    local_stats.key_generation_ms = std::chrono::duration<double, std::milli>(
        key_end - key_begin).count();
    local_stats.density_scatter_ms = std::chrono::duration<double, std::milli>(
        scatter_end - scatter_begin).count();
    local_stats.download_ms = std::chrono::duration<double, std::milli>(
        download_end - download_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(
        download_end - total_begin).count();
    if (stats)
        *stats = local_stats;
}

void updatePersistentPressureGridDirectFields3D(
    PersistentPressureGrid3D& context,
    const std::vector<GridKey>& ordered_dof_keys,
    const std::vector<GridKey>& touched_keys,
    const std::vector<unsigned char>& solid_flags,
    PressureGridFieldStats* stats,
    bool reuse_resident_virtual_touched_keys,
    bool rebuild_resident_legacy_dof_keys,
    bool reuse_resident_solid_flags)
{
    if (!reuse_resident_virtual_touched_keys
        && touched_keys.size() != solid_flags.size())
        throw std::runtime_error("persistent pressure grid solid field size mismatch");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    auto& storage = getPressureGridStorage(context);
    const std::size_t touched_count = reuse_resident_virtual_touched_keys
        ? storage.virtual_touched_keys.size() : touched_keys.size();
    const std::size_t solid_count = reuse_resident_solid_flags
        ? storage.solid_flags.size() : solid_flags.size();
    if (touched_count != solid_count)
        throw std::runtime_error("resident pressure topology size mismatch");
    if (rebuild_resident_legacy_dof_keys &&
        (!reuse_resident_virtual_touched_keys || storage.dof_source_keys.empty()))
        throw std::runtime_error(
            "resident legacy DOF rebuild requires current device topology");
#ifdef ENABLE_AMGX
    storage.pushout_face_seed_valid = false;
#endif
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    resizePressureGrid(storage.ordered_dof_keys,
        rebuild_resident_legacy_dof_keys ? touched_count
                                         : ordered_dof_keys.size(),
        storage);
    resizePressureGrid(storage.touched_keys, touched_count, storage);
    if (!reuse_resident_solid_flags)
        resizePressureGrid(storage.solid_flags, solid_count, storage);
    if (!ordered_dof_keys.empty() && !rebuild_resident_legacy_dof_keys)
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.ordered_dof_keys.data()),
                      ordered_dof_keys.data(), ordered_dof_keys.size() * sizeof(GridKey),
                      cudaMemcpyHostToDevice),
            "persistent pressure grid DOF upload");
    if (touched_count != 0) {
        if (reuse_resident_virtual_touched_keys)
            thrust::copy(storage.virtual_touched_keys.begin(),
                storage.virtual_touched_keys.end(), storage.touched_keys.begin());
        else
            checkCuda(cudaMemcpy(
                          thrust::raw_pointer_cast(storage.touched_keys.data()),
                          touched_keys.data(),
                          touched_keys.size() * sizeof(GridKey),
                          cudaMemcpyHostToDevice),
                "persistent pressure grid touched-key upload");
        if (!reuse_resident_solid_flags)
            checkCuda(cudaMemcpy(
                          thrust::raw_pointer_cast(storage.solid_flags.data()),
                          solid_flags.data(),
                          solid_flags.size() * sizeof(unsigned char),
                          cudaMemcpyHostToDevice),
                "persistent pressure grid solid-flag upload");
    }
    if (rebuild_resident_legacy_dof_keys) {
        auto stencil_begin = thrust::make_zip_iterator(thrust::make_tuple(
            storage.touched_keys.begin(), storage.solid_flags.begin()));
        const auto dof_end = thrust::copy_if(storage.touched_keys.begin(),
            storage.touched_keys.end(), stencil_begin,
            storage.ordered_dof_keys.begin(),
            IsActivePressureDof {
                thrust::raw_pointer_cast(storage.dof_source_keys.data()),
                static_cast<int>(storage.dof_source_keys.size()) });
        const std::size_t resident_dofs = static_cast<std::size_t>(
            dof_end - storage.ordered_dof_keys.begin());
        storage.ordered_dof_keys.resize(resident_dofs);
        if (!ordered_dof_keys.empty()
            && resident_dofs != ordered_dof_keys.size())
            throw std::runtime_error(
                "resident legacy DOF count does not match host topology");
    }
    checkCuda(cudaDeviceSynchronize(),
        "persistent pressure grid direct-field upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->dofs = storage.ordered_dof_keys.size();
        stats->touched_nodes = touched_count;
        stats->resident_touched_input =
            reuse_resident_virtual_touched_keys;
        stats->resident_dof_input = rebuild_resident_legacy_dof_keys;
        stats->resident_solid_flags = reuse_resident_solid_flags;
        stats->reallocations = pressureGridReallocations(storage);
        stats->upload_ms = std::chrono::duration<double, std::milli>(
            upload_end - upload_begin).count();
        stats->total_ms = std::chrono::duration<double, std::milli>(
            upload_end - total_begin).count();
    }
}

void assemblePersistentPressureGridCsr3D(
    PersistentPressureGrid3D& context, double rho_rest, double dx, double dt,
    PressureCsrResult& result, PressureCsrStats* stats,
    bool materialize_host_csr, bool materialize_host_rhs)
{
    if (!(dx > 0.0) || !(dt > 0.0))
        throw std::runtime_error("persistent pressure-grid CSR requires positive dx and dt");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    PressureCsrStats local_stats;
    auto& storage = getPressureGridStorage(context);
    const std::size_t rows_size = storage.ordered_dof_keys.size();
    local_stats.rows = rows_size;
    if (materialize_host_csr)
        result.row_offsets.assign(rows_size + 1, 0);
    else
        result.row_offsets.clear();
    result.column_indices.clear();
    result.values.clear();
    result.rhs.clear();
    if (rows_size == 0) {
        local_stats.reallocations = pressureGridReallocations(storage);
        if (stats)
            *stats = local_stats;
        return;
    }
    if (storage.touched_keys.empty()
        || storage.touched_keys.size() != storage.solid_flags.size()
        || storage.density.active_keys.empty()
        || storage.density.active_keys.size() != storage.density.grid_mass.size())
        throw std::runtime_error("persistent pressure-grid CSR fields are incomplete");

    constexpr int block_size = 256;
    const int rows = static_cast<int>(rows_size);
    const int blocks = (rows + block_size - 1) / block_size;
    const auto sort_begin = std::chrono::high_resolution_clock::now();
    resizePressureGrid(storage.sorted_dof_keys, rows_size, storage);
    resizePressureGrid(storage.sorted_dof_indices, rows_size, storage);
    resizePressureGrid(storage.sorted_solid_keys, storage.touched_keys.size(), storage);
    resizePressureGrid(storage.sorted_solid_flags, storage.solid_flags.size(), storage);
    resizePressureGrid(storage.rho_star, rows_size, storage);
    resizePressureGrid(storage.entry_columns, rows_size * 7, storage);
    resizePressureGrid(storage.entry_values, rows_size * 7, storage);
    resizePressureGrid(storage.entry_counts, rows_size, storage);
    resizePressureGrid(storage.rhs, rows_size, storage);
    resizePressureGrid(storage.missing_solid, 1, storage);
#ifdef ENABLE_AMGX
    resizePressureGrid(storage.row_touches_air, rows_size, storage);
    storage.enclosed_island_count = 0;
#endif
    thrust::copy(storage.ordered_dof_keys.begin(), storage.ordered_dof_keys.end(),
        storage.sorted_dof_keys.begin());
    thrust::sequence(storage.sorted_dof_indices.begin(), storage.sorted_dof_indices.end());
    thrust::sort_by_key(storage.sorted_dof_keys.begin(), storage.sorted_dof_keys.end(),
        storage.sorted_dof_indices.begin());
    thrust::copy(storage.touched_keys.begin(), storage.touched_keys.end(),
        storage.sorted_solid_keys.begin());
    thrust::copy(storage.solid_flags.begin(), storage.solid_flags.end(),
        storage.sorted_solid_flags.begin());
    thrust::sort_by_key(storage.sorted_solid_keys.begin(), storage.sorted_solid_keys.end(),
        storage.sorted_solid_flags.begin());
    checkCuda(cudaGetLastError(), "persistent pressure-grid CSR lookup sort");
    checkCuda(cudaDeviceSynchronize(),
        "persistent pressure-grid CSR lookup sort synchronize");
    const auto sort_end = std::chrono::high_resolution_clock::now();

    const auto assembly_begin = std::chrono::high_resolution_clock::now();
    gatherPressureDensityKernel<<<blocks, block_size>>>(
        thrust::raw_pointer_cast(storage.ordered_dof_keys.data()), rows,
        thrust::raw_pointer_cast(storage.density.active_keys.data()),
        thrust::raw_pointer_cast(storage.density.grid_mass.data()),
        static_cast<int>(storage.density.active_keys.size()),
        thrust::raw_pointer_cast(storage.rho_star.data()));
    checkCuda(cudaGetLastError(), "persistent pressure-grid density gather launch");
    thrust::fill(storage.missing_solid.begin(), storage.missing_solid.end(), 0);
    assemblePressureCsrKernel<<<blocks, block_size>>>(
        thrust::raw_pointer_cast(storage.ordered_dof_keys.data()), rows,
        thrust::raw_pointer_cast(storage.sorted_dof_keys.data()),
        thrust::raw_pointer_cast(storage.sorted_dof_indices.data()),
        thrust::raw_pointer_cast(storage.sorted_solid_keys.data()),
        thrust::raw_pointer_cast(storage.sorted_solid_flags.data()),
        static_cast<int>(storage.sorted_solid_keys.size()),
        thrust::raw_pointer_cast(storage.rho_star.data()), rho_rest, dx, dt,
        thrust::raw_pointer_cast(storage.entry_columns.data()),
        thrust::raw_pointer_cast(storage.entry_values.data()),
        thrust::raw_pointer_cast(storage.entry_counts.data()),
        thrust::raw_pointer_cast(storage.rhs.data()),
        thrust::raw_pointer_cast(storage.missing_solid.data()),
#ifdef ENABLE_AMGX
        thrust::raw_pointer_cast(storage.row_touches_air.data()));
#else
        nullptr);
#endif
    checkCuda(cudaGetLastError(), "persistent pressure-grid CSR assembly launch");
    checkCuda(cudaDeviceSynchronize(),
        "persistent pressure-grid CSR assembly synchronize");
#ifdef ENABLE_AMGX
    resizePressureGrid(storage.island_parent, rows_size, storage);
    resizePressureGrid(storage.island_touches_air, rows_size, storage);
    thrust::sequence(storage.island_parent.begin(), storage.island_parent.end());
    thrust::fill(storage.island_touches_air.begin(),
        storage.island_touches_air.end(), 0);
    unitePressureIslandsKernel<<<blocks, block_size>>>(
        thrust::raw_pointer_cast(storage.entry_columns.data()),
        thrust::raw_pointer_cast(storage.entry_counts.data()), rows,
        thrust::raw_pointer_cast(storage.island_parent.data()));
    checkCuda(cudaGetLastError(), "pressure island union launch");
    checkCuda(cudaDeviceSynchronize(), "pressure island union synchronize");
    finalizePressureIslandsKernel<<<blocks, block_size>>>(rows,
        thrust::raw_pointer_cast(storage.island_parent.data()),
        thrust::raw_pointer_cast(storage.row_touches_air.data()),
        thrust::raw_pointer_cast(storage.island_touches_air.data()));
    checkCuda(cudaGetLastError(), "pressure island finalize launch");
    checkCuda(cudaDeviceSynchronize(), "pressure island finalize synchronize");
    storage.enclosed_island_count = static_cast<std::size_t>(thrust::count_if(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(rows),
        IsEnclosedPressureIsland {
            thrust::raw_pointer_cast(storage.island_parent.data()),
            thrust::raw_pointer_cast(storage.island_touches_air.data()) }));
#endif
    const auto assembly_end = std::chrono::high_resolution_clock::now();

    const auto download_begin = std::chrono::high_resolution_clock::now();
    if (materialize_host_rhs)
        result.rhs.resize(rows_size);
    int missing_solid = 0;
    if (materialize_host_csr) {
        std::vector<int> counts(rows_size);
        std::vector<int> columns(rows_size * 7);
        std::vector<double> values(rows_size * 7);
        checkCuda(cudaMemcpy(counts.data(),
                      thrust::raw_pointer_cast(storage.entry_counts.data()),
                      counts.size() * sizeof(int), cudaMemcpyDeviceToHost),
            "persistent pressure-grid CSR count download");
        checkCuda(cudaMemcpy(columns.data(),
                      thrust::raw_pointer_cast(storage.entry_columns.data()),
                      columns.size() * sizeof(int), cudaMemcpyDeviceToHost),
            "persistent pressure-grid CSR column download");
        checkCuda(cudaMemcpy(values.data(),
                      thrust::raw_pointer_cast(storage.entry_values.data()),
                      values.size() * sizeof(double), cudaMemcpyDeviceToHost),
            "persistent pressure-grid CSR value download");
        for (std::size_t row = 0; row < rows_size; ++row) {
            if (counts[row] < 1 || counts[row] > 7)
                throw std::runtime_error(
                    "persistent pressure-grid CSR invalid row width");
            result.row_offsets[row + 1] = result.row_offsets[row] + counts[row];
            for (int entry = 0; entry < counts[row]; ++entry) {
                result.column_indices.push_back(columns[row * 7 + entry]);
                result.values.push_back(values[row * 7 + entry]);
            }
        }
        local_stats.nnz = result.values.size();
    } else {
        local_stats.nnz = static_cast<std::size_t>(thrust::reduce(
            storage.entry_counts.begin(), storage.entry_counts.end(), 0));
        if (local_stats.nnz < rows_size || local_stats.nnz > rows_size * 7)
            throw std::runtime_error(
                "persistent pressure-grid CSR invalid compact nnz");
    }
    if (materialize_host_rhs)
        checkCuda(cudaMemcpy(result.rhs.data(),
                      thrust::raw_pointer_cast(storage.rhs.data()),
                      result.rhs.size() * sizeof(double), cudaMemcpyDeviceToHost),
            "persistent pressure-grid CSR RHS download");
    checkCuda(cudaMemcpy(&missing_solid,
                  thrust::raw_pointer_cast(storage.missing_solid.data()), sizeof(int),
                  cudaMemcpyDeviceToHost),
        "persistent pressure-grid CSR missing-solid download");
    const auto download_end = std::chrono::high_resolution_clock::now();
    local_stats.solid_lookup_misses = static_cast<std::size_t>(missing_solid);
    local_stats.reallocations = pressureGridReallocations(storage);
    local_stats.lookup_sort_ms = std::chrono::duration<double, std::milli>(
        sort_end - sort_begin).count();
    local_stats.assembly_ms = std::chrono::duration<double, std::milli>(
        assembly_end - assembly_begin).count();
    local_stats.download_ms = std::chrono::duration<double, std::milli>(
        download_end - download_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(
        download_end - total_begin).count();
    if (stats)
        *stats = local_stats;
}

#ifdef ENABLE_AMGX
void updatePersistentPressureGridPushout3D(
    PersistentPressureGrid3D& context,
    const std::vector<GridKey>& displacement_keys,
    const std::vector<Vec3d>& displacement,
    const std::vector<GridKey>& face_keys,
    const std::vector<Vec3d>& face_velocity,
    double rho_rest, double dx, double dt,
    const GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D& level_set,
    const BoxShellCollision3D& collision,
    PressurePushoutStats* stats)
{
    if (!(rho_rest > 0.0) || !(dx > 0.0) || !(dt > 0.0)
        || displacement_keys.size() != displacement.size()
        || face_keys.size() != face_velocity.size()
        || !std::is_sorted(displacement_keys.begin(), displacement_keys.end())
        || !std::is_sorted(face_keys.begin(), face_keys.end()))
        throw std::runtime_error("invalid pressure pushout device input");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    PressurePushoutStats local_stats;
    local_stats.displacement_nodes = displacement_keys.size();
    local_stats.face_nodes = face_keys.size();
    auto& storage = getPressureGridStorage(context);
    const std::size_t dof_count = storage.ordered_dof_keys.size();
    const std::size_t touched_count = storage.sorted_solid_keys.size();
    if (storage.rhs.size() != dof_count
        || storage.sorted_solid_flags.size() != touched_count)
        throw std::runtime_error(
            "pressure pushout requires assembled device fields");

    const auto upload_begin = std::chrono::high_resolution_clock::now();
    resizePressureGrid(storage.pushout_displacement_keys,
        displacement_keys.size(), storage);
    resizePressureGrid(storage.pushout_displacement,
        displacement.size(), storage);
    resizePressureGrid(storage.pushout_face_keys, face_keys.size(), storage);
    resizePressureGrid(storage.pushout_face_velocity,
        face_velocity.size(), storage);
    if (!displacement_keys.empty()) {
        checkCuda(cudaMemcpy(
                      thrust::raw_pointer_cast(
                          storage.pushout_displacement_keys.data()),
                      displacement_keys.data(),
                      displacement_keys.size() * sizeof(GridKey),
                      cudaMemcpyHostToDevice),
            "pressure pushout displacement key upload");
        checkCuda(cudaMemcpy(
                      thrust::raw_pointer_cast(
                          storage.pushout_displacement.data()),
                      displacement.data(), displacement.size() * sizeof(Vec3d),
                      cudaMemcpyHostToDevice),
            "pressure pushout displacement upload");
    }
    if (!face_keys.empty()) {
        checkCuda(cudaMemcpy(
                      thrust::raw_pointer_cast(storage.pushout_face_keys.data()),
                      face_keys.data(), face_keys.size() * sizeof(GridKey),
                      cudaMemcpyHostToDevice),
            "pressure pushout face key upload");
        checkCuda(cudaMemcpy(
                      thrust::raw_pointer_cast(
                          storage.pushout_face_velocity.data()),
                      face_velocity.data(),
                      face_velocity.size() * sizeof(Vec3d),
                      cudaMemcpyHostToDevice),
            "pressure pushout face velocity upload");
    }
    checkCuda(cudaDeviceSynchronize(), "pressure pushout upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    constexpr int block_size = 256;
    const auto rhs_begin = std::chrono::high_resolution_clock::now();
    if (dof_count != 0) {
        const int blocks = static_cast<int>((dof_count + block_size - 1)
            / block_size);
        applyPressurePushoutRhsKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.ordered_dof_keys.data()),
            static_cast<int>(dof_count),
            thrust::raw_pointer_cast(storage.sorted_solid_keys.data()),
            thrust::raw_pointer_cast(storage.sorted_solid_flags.data()),
            static_cast<int>(touched_count),
            displacement_keys.empty() ? nullptr
                : thrust::raw_pointer_cast(
                    storage.pushout_displacement_keys.data()),
            displacement.empty() ? nullptr
                : thrust::raw_pointer_cast(storage.pushout_displacement.data()),
            static_cast<int>(displacement.size()), rho_rest * dx / (dt * dt),
            collision, level_set, dx,
            thrust::raw_pointer_cast(storage.rhs.data()));
        checkCuda(cudaGetLastError(), "pressure pushout RHS launch");
        checkCuda(cudaDeviceSynchronize(), "pressure pushout RHS synchronize");
    }
    const auto rhs_end = std::chrono::high_resolution_clock::now();

    const auto face_begin = std::chrono::high_resolution_clock::now();
    resizePressureGrid(storage.pressure_velocity, touched_count, storage);
    if (touched_count != 0) {
        const int blocks = static_cast<int>((touched_count + block_size - 1)
            / block_size);
        initializePressureFaceVelocityKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.sorted_solid_keys.data()),
            static_cast<int>(touched_count),
            face_keys.empty() ? nullptr
                : thrust::raw_pointer_cast(storage.pushout_face_keys.data()),
            face_velocity.empty() ? nullptr
                : thrust::raw_pointer_cast(storage.pushout_face_velocity.data()),
            static_cast<int>(face_velocity.size()),
            thrust::raw_pointer_cast(storage.pressure_velocity.data()));
        checkCuda(cudaGetLastError(), "pressure pushout face initialize launch");
        checkCuda(cudaDeviceSynchronize(),
            "pressure pushout face initialize synchronize");
    }
    const auto face_end = std::chrono::high_resolution_clock::now();
    local_stats.upload_ms = std::chrono::duration<double, std::milli>(
        upload_end - upload_begin).count();
    local_stats.rhs_update_ms = std::chrono::duration<double, std::milli>(
        rhs_end - rhs_begin).count();
    local_stats.face_initialize_ms = std::chrono::duration<double, std::milli>(
        face_end - face_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(
        face_end - total_begin).count();
    if (stats)
        *stats = local_stats;
    storage.pushout_face_seed_valid = true;
}

void computePersistentPressureGridPushoutGpu3D(
    PersistentPressureGrid3D& pressure_context,
    PersistentVelocityTransfer3D& velocity_context,
    const GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D& level_set,
    const BoxShellCollision3D& collision,
    std::size_t particle_count, double rho_rest, double dx, double dt,
    PressurePushoutGpuStats* stats, PushoutHitParticles* hit_particles,
    PushoutNodeReduction* node_reduction,
    PushoutFaceVelocityDebug* face_velocity_debug,
    PushoutRhsDebug* rhs_debug)
{
    if (!(rho_rest > 0.0) || !(dx > 0.0) || !(dt > 0.0))
        throw std::runtime_error("invalid GPU pressure pushout input");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    PressurePushoutGpuStats local_stats;
    local_stats.particles = particle_count;
    auto& pressure = getPressureGridStorage(pressure_context);
    auto& velocity = getVelocityTransferStorage(velocity_context);
    const std::size_t dof_count = pressure.ordered_dof_keys.size();
    const std::size_t touched_count = pressure.sorted_solid_keys.size();
    if (pressure.rhs.size() != dof_count
        || pressure.sorted_solid_flags.size() != touched_count
        || pressure.touched_keys.size() != touched_count)
        throw std::runtime_error(
            "GPU pressure pushout requires assembled device fields");
    if (velocity.output_v.size() != particle_count
        || velocity.output_grad_v.size() != particle_count
        || velocity.output_x.size() != particle_count)
        throw std::runtime_error(
            "GPU pressure pushout requires a resident Q1 G2P particle owner");

    constexpr int block_size = 256;

    // Legacy touched-key rank, co-sorted alongside sorted_solid_keys so its
    // index space matches sorted_solid_flags exactly.
    const auto rank_begin = std::chrono::high_resolution_clock::now();
    resizePressureGrid(pressure.sorted_solid_rank, touched_count, pressure);
    {
        thrust::device_vector<GridKey> rank_sort_keys(pressure.touched_keys);
        thrust::sequence(pressure.sorted_solid_rank.begin(),
            pressure.sorted_solid_rank.end());
        thrust::sort_by_key(rank_sort_keys.begin(), rank_sort_keys.end(),
            pressure.sorted_solid_rank.begin());
    }

    // Particle projection: in-place V/gradV patch plus compact candidates.
    const auto projection_begin = std::chrono::high_resolution_clock::now();
    resizePressureGrid(pressure.pushout_candidates, particle_count, pressure);
    resizePressureGrid(pressure.pushout_candidate_valid, particle_count,
        pressure);
    if (particle_count != 0) {
        const int blocks = static_cast<int>(
            (particle_count + block_size - 1) / block_size);
        pushoutParticleProjectionKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(velocity.output_x.data()),
            static_cast<int>(particle_count),
            thrust::raw_pointer_cast(velocity.output_v.data()),
            thrust::raw_pointer_cast(velocity.output_grad_v.data()),
            thrust::raw_pointer_cast(pressure.sorted_solid_keys.data()),
            thrust::raw_pointer_cast(pressure.sorted_solid_flags.data()),
            static_cast<int>(touched_count), collision, level_set, dx,
            thrust::raw_pointer_cast(pressure.pushout_candidates.data()),
            thrust::raw_pointer_cast(pressure.pushout_candidate_valid.data()));
        checkCuda(cudaGetLastError(), "pushout particle projection launch");
        checkCuda(cudaDeviceSynchronize(),
            "pushout particle projection synchronize");
    }
    const auto projection_end = std::chrono::high_resolution_clock::now();

    // Compact valid candidates, then reduce to one max-displacement winner
    // per node (ties broken identically to the CPU "if bigger, overwrite"
    // rule: strictly-greater squared displacement wins).
    resizePressureGrid(pressure.pushout_candidates_compact, particle_count,
        pressure);
    const auto compact_end_it = thrust::copy_if(
        pressure.pushout_candidates.begin(), pressure.pushout_candidates.end(),
        pressure.pushout_candidate_valid.begin(),
        pressure.pushout_candidates_compact.begin(), PushoutValidPredicate{});
    const std::size_t candidate_count = static_cast<std::size_t>(
        compact_end_it - pressure.pushout_candidates_compact.begin());
    local_stats.candidates = candidate_count;

    // Sparse host mirror for every hit particle (not just the per-node
    // reduction winners downloaded below): the CPU pushout() writes
    // sim.particles.V.array[i] directly for each of these, and downstream
    // CPU-side calculateDt()/evalMaxParticleSpeed() reads that host array.
    // Skip entirely when nothing was hit -- no D2H at all in that case.
    if (hit_particles) {
        hit_particles->particle_id.clear();
        hit_particles->corrected_v.clear();
        hit_particles->node_key.clear();
        hit_particles->sq_disp.clear();
        hit_particles->debug_source.clear();
        if (candidate_count != 0) {
            std::vector<PushoutCandidate> hits(candidate_count);
            checkCuda(cudaMemcpy(hits.data(),
                          thrust::raw_pointer_cast(
                              pressure.pushout_candidates_compact.data()),
                          candidate_count * sizeof(PushoutCandidate),
                          cudaMemcpyDeviceToHost),
                "pushout hit-particle download");
            hit_particles->particle_id.reserve(candidate_count);
            hit_particles->corrected_v.reserve(candidate_count);
            hit_particles->node_key.reserve(candidate_count);
            hit_particles->sq_disp.reserve(candidate_count);
            hit_particles->debug_source.reserve(candidate_count);
            for (const auto& c : hits) {
                hit_particles->particle_id.push_back(c.particle_id);
                hit_particles->corrected_v.push_back(c.corrected_v);
                hit_particles->node_key.push_back(c.key);
                hit_particles->sq_disp.push_back(c.sq_disp);
                hit_particles->debug_source.push_back(c.debug_source);
            }
        }
    }

    thrust::sort(pressure.pushout_candidates_compact.begin(),
        pressure.pushout_candidates_compact.begin() +
            static_cast<long>(candidate_count),
        PushoutKeyLess{});
    std::size_t displacement_count = 0;
    // Reduce (node_key, sq_disp, disp) candidates to one max-displacement
    // winner per node, into buffers distinct from the sorted input (thrust
    // requires the input and output ranges of reduce_by_key not to alias).
    {
        // Both the "keys" and "values" ranges below dereference to the same
        // PushoutCandidate type; only the reduced *values* are used
        // afterwards (each winning candidate already carries its own
        // group's key), but reduce_by_key still requires a same-typed keys
        // output range to report the reduced groups.
        thrust::device_vector<PushoutCandidate> reduced_keys(candidate_count);
        thrust::device_vector<PushoutCandidate> reduced_values(candidate_count);
        std::size_t reduced_count = 0;
        if (candidate_count != 0) {
            auto out_end = thrust::reduce_by_key(
                pressure.pushout_candidates_compact.begin(),
                pressure.pushout_candidates_compact.begin() +
                    static_cast<long>(candidate_count),
                pressure.pushout_candidates_compact.begin(), reduced_keys.begin(),
                reduced_values.begin(), PushoutKeyEqual{}, PushoutMaxDispOp{});
            reduced_count = static_cast<std::size_t>(
                out_end.first - reduced_keys.begin());
        }
        displacement_count = reduced_count;
        resizePressureGrid(pressure.pushout_displacement_keys,
            displacement_count, pressure);
        resizePressureGrid(pressure.pushout_displacement, displacement_count,
            pressure);
        if (displacement_count != 0) {
            thrust::transform(reduced_values.begin(),
                reduced_values.begin() +
                    static_cast<long>(displacement_count),
                pressure.pushout_displacement_keys.begin(), PushoutExtractKey{});
            thrust::transform(reduced_values.begin(),
                reduced_values.begin() +
                    static_cast<long>(displacement_count),
                pressure.pushout_displacement.begin(), PushoutExtractDisp{});
        }
        if (node_reduction) {
            node_reduction->node_key.clear();
            node_reduction->particle_id.clear();
            node_reduction->sq_disp.clear();
            node_reduction->disp.clear();
            node_reduction->clamped_disp.clear();
            if (displacement_count != 0) {
                std::vector<PushoutCandidate> winners(displacement_count);
                checkCuda(cudaMemcpy(winners.data(),
                              thrust::raw_pointer_cast(reduced_values.data()),
                              displacement_count * sizeof(PushoutCandidate),
                              cudaMemcpyDeviceToHost),
                    "pushout node-reduction debug download");
                node_reduction->node_key.reserve(displacement_count);
                node_reduction->particle_id.reserve(displacement_count);
                node_reduction->sq_disp.reserve(displacement_count);
                node_reduction->disp.reserve(displacement_count);
                for (const auto& w : winners) {
                    node_reduction->node_key.push_back(w.key);
                    node_reduction->particle_id.push_back(w.particle_id);
                    node_reduction->sq_disp.push_back(w.sq_disp);
                    node_reduction->disp.push_back(w.disp);
                }
            }
        }
    }
    local_stats.displacement_nodes = displacement_count;
    if (displacement_count != 0) {
        const int blocks = static_cast<int>(
            (displacement_count + block_size - 1) / block_size);
        clampPushoutDisplacementKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(pressure.pushout_displacement.data()),
            static_cast<int>(displacement_count), 0.5 * dx);
        checkCuda(cudaGetLastError(), "pushout displacement clamp launch");
        checkCuda(cudaDeviceSynchronize(),
            "pushout displacement clamp synchronize");
    }
    if (node_reduction && !node_reduction->node_key.empty()) {
        node_reduction->clamped_disp.resize(displacement_count);
        checkCuda(cudaMemcpy(node_reduction->clamped_disp.data(),
                      thrust::raw_pointer_cast(pressure.pushout_displacement.data()),
                      displacement_count * sizeof(Vec3d), cudaMemcpyDeviceToHost),
            "pushout clamped displacement debug download");
    }
    const auto reduce_end = std::chrono::high_resolution_clock::now();

    // RHS adjustment (same kernel as updatePersistentPressureGridPushout3D,
    // just fed device-resident pushout arrays with no host round trip).
    const auto rhs_begin = std::chrono::high_resolution_clock::now();
    std::vector<double> rhs_debug_before;
    if (rhs_debug && dof_count != 0) {
        rhs_debug_before.resize(dof_count);
        checkCuda(cudaMemcpy(rhs_debug_before.data(),
                      thrust::raw_pointer_cast(pressure.rhs.data()),
                      dof_count * sizeof(double), cudaMemcpyDeviceToHost),
            "pushout rhs debug pre-snapshot download");
    }
    if (dof_count != 0) {
        const int blocks = static_cast<int>(
            (dof_count + block_size - 1) / block_size);
        applyPressurePushoutRhsKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(pressure.ordered_dof_keys.data()),
            static_cast<int>(dof_count),
            thrust::raw_pointer_cast(pressure.sorted_solid_keys.data()),
            thrust::raw_pointer_cast(pressure.sorted_solid_flags.data()),
            static_cast<int>(touched_count),
            displacement_count == 0 ? nullptr
                : thrust::raw_pointer_cast(
                      pressure.pushout_displacement_keys.data()),
            displacement_count == 0 ? nullptr
                : thrust::raw_pointer_cast(pressure.pushout_displacement.data()),
            static_cast<int>(displacement_count), rho_rest * dx / (dt * dt),
            collision, level_set, dx,
            thrust::raw_pointer_cast(pressure.rhs.data()));
        checkCuda(cudaGetLastError(), "GPU pushout RHS launch");
        checkCuda(cudaDeviceSynchronize(), "GPU pushout RHS synchronize");
    }
    if (rhs_debug && dof_count != 0) {
        std::vector<double> rhs_after(dof_count);
        checkCuda(cudaMemcpy(rhs_after.data(),
                      thrust::raw_pointer_cast(pressure.rhs.data()),
                      dof_count * sizeof(double), cudaMemcpyDeviceToHost),
            "pushout rhs debug post-snapshot download");
        std::vector<GridKey> dof_keys_host(dof_count);
        checkCuda(cudaMemcpy(dof_keys_host.data(),
                      thrust::raw_pointer_cast(pressure.ordered_dof_keys.data()),
                      dof_count * sizeof(GridKey), cudaMemcpyDeviceToHost),
            "pushout rhs debug dof key download");
        rhs_debug->dof_key.clear();
        rhs_debug->delta.clear();
        for (std::size_t row = 0; row < dof_count; ++row) {
            const double delta = rhs_after[row] - rhs_debug_before[row];
            if (delta != 0.0) {
                rhs_debug->dof_key.push_back(dof_keys_host[row]);
                rhs_debug->delta.push_back(delta);
            }
        }
    }
    const auto rhs_end = std::chrono::high_resolution_clock::now();

    // Face velocity: deterministic gather (rank tie-break) directly into a
    // full per-touched-node array, matching what
    // initializePressureFaceVelocityKernel would seed from the CPU's sparse
    // nonzero-only face list.
    const auto face_begin = std::chrono::high_resolution_clock::now();
    resizePressureGrid(pressure.pressure_velocity, touched_count, pressure);
    if (touched_count != 0) {
        const int blocks = static_cast<int>(
            (touched_count + block_size - 1) / block_size);
        pushoutFaceVelocityGatherKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(pressure.sorted_solid_keys.data()),
            thrust::raw_pointer_cast(pressure.sorted_solid_flags.data()),
            thrust::raw_pointer_cast(pressure.sorted_solid_rank.data()),
            static_cast<int>(touched_count),
            displacement_count == 0 ? nullptr
                : thrust::raw_pointer_cast(
                      pressure.pushout_displacement_keys.data()),
            displacement_count == 0 ? nullptr
                : thrust::raw_pointer_cast(pressure.pushout_displacement.data()),
            static_cast<int>(displacement_count), collision, level_set, dx,
            thrust::raw_pointer_cast(pressure.pressure_velocity.data()));
        checkCuda(cudaGetLastError(), "GPU pushout face gather launch");
        checkCuda(cudaDeviceSynchronize(), "GPU pushout face gather synchronize");
    }
    if (face_velocity_debug) {
        face_velocity_debug->node_key.clear();
        face_velocity_debug->velocity.clear();
        if (touched_count != 0) {
            std::vector<GridKey> keys(touched_count);
            std::vector<Vec3d> values(touched_count);
            checkCuda(cudaMemcpy(keys.data(),
                          thrust::raw_pointer_cast(pressure.sorted_solid_keys.data()),
                          touched_count * sizeof(GridKey), cudaMemcpyDeviceToHost),
                "pushout face velocity debug key download");
            checkCuda(cudaMemcpy(values.data(),
                          thrust::raw_pointer_cast(pressure.pressure_velocity.data()),
                          touched_count * sizeof(Vec3d), cudaMemcpyDeviceToHost),
                "pushout face velocity debug value download");
            for (std::size_t i = 0; i < touched_count; ++i) {
                const Vec3d& v = values[i];
                if (v.x * v.x + v.y * v.y + v.z * v.z > 0.0) {
                    face_velocity_debug->node_key.push_back(keys[i]);
                    face_velocity_debug->velocity.push_back(v);
                }
            }
        }
    }
    const auto face_end = std::chrono::high_resolution_clock::now();
    local_stats.face_nodes = touched_count;
    local_stats.resident_fluid_input = true;
    local_stats.projection_ms = std::chrono::duration<double, std::milli>(
        projection_end - projection_begin).count();
    local_stats.reduce_ms = std::chrono::duration<double, std::milli>(
        reduce_end - rank_begin).count();
    local_stats.rhs_ms = std::chrono::duration<double, std::milli>(
        rhs_end - rhs_begin).count();
    local_stats.face_ms = std::chrono::duration<double, std::milli>(
        face_end - face_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(
        face_end - total_begin).count();
    if (stats)
        *stats = local_stats;
    pressure.pushout_face_seed_valid = true;
}

void solvePersistentPressureGridAmgx3D(
    PersistentPressureGrid3D& context,
    const std::vector<double>* rhs_override,
    double diagonal_epsilon,
    std::vector<double>& solution,
    PressureAmgxStats* stats, bool download_solution)
{
    const auto total_begin = std::chrono::high_resolution_clock::now();
    PressureAmgxStats local_stats;
    auto& storage = getPressureGridStorage(context);
    const std::size_t rows_size = storage.ordered_dof_keys.size();
    if (rows_size == 0) {
        solution.clear();
        if (stats)
            *stats = local_stats;
        return;
    }
    if (storage.entry_counts.size() != rows_size
        || storage.entry_columns.size() != rows_size * 7
        || storage.entry_values.size() != rows_size * 7
        || storage.rhs.size() != rows_size)
        throw std::runtime_error("AMGX pressure solve requires assembled device CSR");
    if (rhs_override && rhs_override->size() != rows_size)
        throw std::runtime_error("AMGX pressure solve RHS override size mismatch");

    const int rows = static_cast<int>(rows_size);
    constexpr int block_size = 256;
    const int blocks = (rows + block_size - 1) / block_size;
    const auto compact_begin = std::chrono::high_resolution_clock::now();
    resizePressureGrid(storage.amgx_row_offsets, rows_size + 1, storage);
    resizePressureGrid(storage.amgx_column_indices, rows_size * 7, storage);
    resizePressureGrid(storage.amgx_values, rows_size * 7, storage);
    resizePressureGrid(storage.amgx_solution, rows_size, storage);
    thrust::fill(storage.amgx_row_offsets.begin(), storage.amgx_row_offsets.begin() + 1,
        0);
    thrust::inclusive_scan(storage.entry_counts.begin(), storage.entry_counts.end(),
        storage.amgx_row_offsets.begin() + 1);
    int nnz = 0;
    checkCuda(cudaMemcpy(&nnz,
                  thrust::raw_pointer_cast(storage.amgx_row_offsets.data()) + rows,
                  sizeof(int), cudaMemcpyDeviceToHost),
        "AMGX pressure CSR nnz download");
    if (nnz <= 0 || nnz > rows * 7)
        throw std::runtime_error("AMGX pressure CSR has invalid device nnz");
    compactPressureCsrKernel<<<blocks, block_size>>>(
        thrust::raw_pointer_cast(storage.entry_columns.data()),
        thrust::raw_pointer_cast(storage.entry_values.data()),
        thrust::raw_pointer_cast(storage.entry_counts.data()),
        thrust::raw_pointer_cast(storage.amgx_row_offsets.data()), rows,
        diagonal_epsilon,
        thrust::raw_pointer_cast(storage.amgx_column_indices.data()),
        thrust::raw_pointer_cast(storage.amgx_values.data()));
    checkCuda(cudaGetLastError(), "AMGX pressure CSR compact launch");
    checkCuda(cudaDeviceSynchronize(), "AMGX pressure CSR compact synchronize");
    const auto compact_end = std::chrono::high_resolution_clock::now();

    const auto rhs_upload_begin = std::chrono::high_resolution_clock::now();
    if (rhs_override) {
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.rhs.data()),
                      rhs_override->data(), rows_size * sizeof(double),
                      cudaMemcpyHostToDevice),
            "AMGX pressure projected RHS upload");
        checkCuda(cudaDeviceSynchronize(),
            "AMGX pressure projected RHS upload synchronize");
    }
    const auto rhs_upload_end = std::chrono::high_resolution_clock::now();
    const double projected_rhs_upload_ms = std::chrono::duration<double, std::milli>(
        rhs_upload_end - rhs_upload_begin).count();

    const auto island_projection_begin = std::chrono::high_resolution_clock::now();
    if (!rhs_override && storage.enclosed_island_count != 0) {
        if (storage.island_parent.size() != rows_size
            || storage.island_touches_air.size() != rows_size)
            throw std::runtime_error(
                "AMGX pressure island projection requires current topology");
        resizePressureGrid(storage.island_rhs_sum, rows_size, storage);
        resizePressureGrid(storage.island_row_count, rows_size, storage);
        thrust::fill(storage.island_rhs_sum.begin(),
            storage.island_rhs_sum.end(), 0.0);
        thrust::fill(storage.island_row_count.begin(),
            storage.island_row_count.end(), 0);
        accumulatePressureIslandRhsKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.rhs.data()), rows,
            thrust::raw_pointer_cast(storage.island_parent.data()),
            thrust::raw_pointer_cast(storage.island_touches_air.data()),
            thrust::raw_pointer_cast(storage.island_rhs_sum.data()),
            thrust::raw_pointer_cast(storage.island_row_count.data()));
        checkCuda(cudaGetLastError(), "pressure island RHS accumulation launch");
        projectPressureIslandRhsKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.rhs.data()), rows,
            thrust::raw_pointer_cast(storage.island_parent.data()),
            thrust::raw_pointer_cast(storage.island_touches_air.data()),
            thrust::raw_pointer_cast(storage.island_rhs_sum.data()),
            thrust::raw_pointer_cast(storage.island_row_count.data()));
        checkCuda(cudaGetLastError(), "pressure island RHS projection launch");
        checkCuda(cudaDeviceSynchronize(),
            "pressure island RHS projection synchronize");
    }
    const auto island_projection_end = std::chrono::high_resolution_clock::now();

    const AmgxRuntime& runtime = getAmgxRuntime();
    AMGX_matrix_handle matrix = nullptr;
    AMGX_vector_handle rhs = nullptr;
    AMGX_vector_handle x = nullptr;
    AMGX_solver_handle solver = nullptr;
    try {
        checkAmgx(AMGX_matrix_create(&matrix, runtime.resources, AMGX_mode_dDDI),
            "AMGX_matrix_create");
        const auto matrix_upload_begin = std::chrono::high_resolution_clock::now();
        checkAmgx(AMGX_matrix_upload_all(matrix, rows, nnz, 1, 1,
                      thrust::raw_pointer_cast(storage.amgx_row_offsets.data()),
                      thrust::raw_pointer_cast(storage.amgx_column_indices.data()),
                      thrust::raw_pointer_cast(storage.amgx_values.data()), nullptr),
            "AMGX_matrix_upload_all device CSR");
        checkCuda(cudaDeviceSynchronize(), "AMGX matrix upload synchronize");
        const auto matrix_upload_end = std::chrono::high_resolution_clock::now();

        checkAmgx(AMGX_vector_create(&rhs, runtime.resources, AMGX_mode_dDDI),
            "AMGX RHS vector create");
        checkAmgx(AMGX_vector_create(&x, runtime.resources, AMGX_mode_dDDI),
            "AMGX solution vector create");
        const auto amgx_rhs_upload_begin = std::chrono::high_resolution_clock::now();
        checkAmgx(AMGX_vector_upload(rhs, rows, 1,
                      thrust::raw_pointer_cast(storage.rhs.data())),
            "AMGX_vector_upload device RHS");
        checkAmgx(AMGX_vector_set_zero(x, rows, 1), "AMGX_vector_set_zero");
        checkCuda(cudaDeviceSynchronize(), "AMGX vector upload synchronize");
        const auto amgx_rhs_upload_end = std::chrono::high_resolution_clock::now();

        checkAmgx(AMGX_solver_create(&solver, runtime.resources, AMGX_mode_dDDI,
                      runtime.config),
            "AMGX_solver_create");
        const auto setup_begin = std::chrono::high_resolution_clock::now();
        checkAmgx(AMGX_solver_setup(solver, matrix), "AMGX_solver_setup");
        checkCuda(cudaDeviceSynchronize(), "AMGX solver setup synchronize");
        const auto setup_end = std::chrono::high_resolution_clock::now();

        const auto solve_begin = std::chrono::high_resolution_clock::now();
        checkAmgx(AMGX_solver_solve(solver, rhs, x), "AMGX_solver_solve");
        checkCuda(cudaDeviceSynchronize(), "AMGX solver solve synchronize");
        const auto solve_end = std::chrono::high_resolution_clock::now();
        AMGX_SOLVE_STATUS solve_status = AMGX_SOLVE_FAILED;
        checkAmgx(AMGX_solver_get_status(solver, &solve_status),
            "AMGX_solver_get_status");
        checkAmgx(AMGX_solver_get_iterations_number(solver, &local_stats.iterations),
            "AMGX_solver_get_iterations_number");
        if (solve_status != AMGX_SOLVE_SUCCESS)
            throw std::runtime_error("AMGX pressure solve did not converge");

        const auto solution_download_begin = std::chrono::high_resolution_clock::now();
        checkAmgx(AMGX_vector_download(x,
                      thrust::raw_pointer_cast(storage.amgx_solution.data())),
            "AMGX_vector_download device solution");
        if (download_solution) {
            solution.resize(rows_size);
            checkCuda(cudaMemcpy(solution.data(),
                          thrust::raw_pointer_cast(storage.amgx_solution.data()),
                          rows_size * sizeof(double), cudaMemcpyDeviceToHost),
                "AMGX pressure solution host download");
        } else {
            solution.clear();
        }
        const auto solution_download_end = std::chrono::high_resolution_clock::now();

        local_stats.rows = rows_size;
        local_stats.nnz = static_cast<std::size_t>(nnz);
        local_stats.solve_status = static_cast<int>(solve_status);
        local_stats.projected_islands = rhs_override
            ? 0 : storage.enclosed_island_count;
        local_stats.diagonal_epsilon = diagonal_epsilon;
        local_stats.island_projection_ms =
            std::chrono::duration<double, std::milli>(
                island_projection_end - island_projection_begin).count();
        local_stats.compact_csr_ms = std::chrono::duration<double, std::milli>(
            compact_end - compact_begin).count();
        local_stats.matrix_upload_ms = std::chrono::duration<double, std::milli>(
            matrix_upload_end - matrix_upload_begin).count();
        local_stats.rhs_upload_ms = projected_rhs_upload_ms
            + std::chrono::duration<double, std::milli>(
                amgx_rhs_upload_end - amgx_rhs_upload_begin).count();
        local_stats.setup_ms = std::chrono::duration<double, std::milli>(
            setup_end - setup_begin).count();
        local_stats.solve_ms = std::chrono::duration<double, std::milli>(
            solve_end - solve_begin).count();
        local_stats.solution_download_ms = std::chrono::duration<double, std::milli>(
            solution_download_end - solution_download_begin).count();
        local_stats.total_ms = std::chrono::duration<double, std::milli>(
            solution_download_end - total_begin).count();
    } catch (...) {
        destroyAmgxSolveHandles(solver, x, rhs, matrix);
        throw;
    }
    destroyAmgxSolveHandles(solver, x, rhs, matrix);
    if (stats)
        *stats = local_stats;
}

void applyPersistentPressureGridAmgxCorrection3D(
    PersistentPressureGrid3D& context, const void* particle_x,
    std::size_t particle_count, double rho_rest, double dx, double dt,
    PressureParticleCorrectionView& result,
    PressureParticleCorrectionStats* stats, bool particle_x_on_device,
    bool download_positions)
{
    if (!(rho_rest > 0.0) || !(dx > 0.0) || !(dt > 0.0))
        throw std::runtime_error("invalid pressure particle correction scale");
    if (particle_count != 0 && !particle_x)
        throw std::runtime_error("pressure particle correction input is null");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    PressureParticleCorrectionStats local_stats;
    local_stats.particles = particle_count;
    local_stats.resident_particle_input = particle_x_on_device;
    result = {};
    auto& storage = getPressureGridStorage(context);
    const std::size_t touched_count = storage.touched_keys.size();
    const std::size_t dof_count = storage.ordered_dof_keys.size();
    local_stats.touched_nodes = touched_count;
    if (storage.amgx_solution.size() != dof_count ||
        storage.sorted_dof_keys.size() != dof_count ||
        storage.sorted_dof_indices.size() != dof_count ||
        storage.sorted_solid_keys.size() != storage.sorted_solid_flags.size())
        throw std::runtime_error("AMGX pressure correction requires assembled keyed fields");

    constexpr int block_size = 256;
    const auto velocity_begin = std::chrono::high_resolution_clock::now();
    resizePressureGrid(storage.pressure_velocity, touched_count, storage);
    if (!storage.pushout_face_seed_valid)
        thrust::fill(storage.pressure_velocity.begin(),
            storage.pressure_velocity.end(), Vec3d{});
    if (touched_count != 0) {
        const int blocks = static_cast<int>((touched_count + block_size - 1)
            / block_size);
        const double coefficient = -dt * dt / rho_rest / dx;
        // sorted_solid_keys is the canonical keyed pressure field. The
        // virtual touched stream intentionally retains legacy page order and
        // therefore is not valid for binary-search interpolation.
        buildPressureFaceVelocityKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.sorted_solid_keys.data()),
            static_cast<int>(touched_count),
            thrust::raw_pointer_cast(storage.sorted_dof_keys.data()),
            thrust::raw_pointer_cast(storage.sorted_dof_indices.data()),
            static_cast<int>(dof_count),
            thrust::raw_pointer_cast(storage.sorted_solid_keys.data()),
            thrust::raw_pointer_cast(storage.sorted_solid_flags.data()),
            static_cast<int>(storage.sorted_solid_keys.size()),
            thrust::raw_pointer_cast(storage.amgx_solution.data()), coefficient,
            0.5 * dx,
            thrust::raw_pointer_cast(storage.pressure_velocity.data()));
        checkCuda(cudaGetLastError(), "pressure face velocity launch");
        checkCuda(cudaDeviceSynchronize(), "pressure face velocity synchronize");
    }
    const auto velocity_end = std::chrono::high_resolution_clock::now();

    const auto upload_begin = std::chrono::high_resolution_clock::now();
    const Vec3d* device_particle_x = nullptr;
    if (particle_x_on_device) {
        device_particle_x = static_cast<const Vec3d*>(particle_x);
    } else {
        resizePressureGrid(storage.correction_particle_x, particle_count, storage);
        if (particle_count != 0) {
            checkCuda(cudaMemcpy(
                          thrust::raw_pointer_cast(
                              storage.correction_particle_x.data()),
                          particle_x, particle_count * sizeof(Vec3d),
                          cudaMemcpyHostToDevice),
                "pressure correction particle upload");
        }
        device_particle_x = particle_count == 0 ? nullptr
            : thrust::raw_pointer_cast(storage.correction_particle_x.data());
    }
    resizePressureGrid(storage.correction_output_x, particle_count, storage);
    const auto upload_end = std::chrono::high_resolution_clock::now();

    const auto correction_begin = std::chrono::high_resolution_clock::now();
    if (particle_count != 0) {
        const int blocks = static_cast<int>((particle_count + block_size - 1)
            / block_size);
        correctPressureParticlesKernel<<<blocks, block_size>>>(
            device_particle_x,
            static_cast<int>(particle_count),
            thrust::raw_pointer_cast(storage.sorted_solid_keys.data()),
            static_cast<int>(touched_count),
            thrust::raw_pointer_cast(storage.pressure_velocity.data()), dx,
            thrust::raw_pointer_cast(storage.correction_output_x.data()));
        checkCuda(cudaGetLastError(), "pressure particle correction launch");
        checkCuda(cudaDeviceSynchronize(), "pressure particle correction synchronize");
    }
    const auto correction_end = std::chrono::high_resolution_clock::now();

    const auto download_begin = std::chrono::high_resolution_clock::now();
    if (download_positions &&
        particle_count > storage.correction_host_capacity) {
        if (storage.correction_host_x)
            cudaFreeHost(storage.correction_host_x);
        storage.correction_host_x = nullptr;
        checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&storage.correction_host_x),
                      particle_count * sizeof(Vec3d), cudaHostAllocDefault),
            "pressure correction pinned output allocation");
        storage.correction_host_capacity = particle_count;
        ++storage.reallocations;
    }
    if (download_positions && particle_count != 0) {
        checkCuda(cudaMemcpy(storage.correction_host_x,
                      thrust::raw_pointer_cast(storage.correction_output_x.data()),
                      particle_count * sizeof(Vec3d), cudaMemcpyDeviceToHost),
            "pressure correction particle download");
    }
    const auto download_end = std::chrono::high_resolution_clock::now();

    result = {download_positions ? storage.correction_host_x : nullptr,
        particle_count == 0 ? nullptr
            : static_cast<const void*>(thrust::raw_pointer_cast(
                storage.correction_output_x.data())),
        particle_count};
    local_stats.particle_upload_ms = std::chrono::duration<double, std::milli>(
        upload_end - upload_begin).count();
    local_stats.velocity_update_ms = std::chrono::duration<double, std::milli>(
        velocity_end - velocity_begin).count();
    local_stats.particle_correction_ms = std::chrono::duration<double, std::milli>(
        correction_end - correction_begin).count();
    local_stats.download_ms = std::chrono::duration<double, std::milli>(
        download_end - download_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(
        download_end - total_begin).count();
    if (stats)
        *stats = local_stats;
}

void downloadPersistentPressureGridFaceVelocity3D(
    PersistentPressureGrid3D& context, std::vector<GridKey>& keys,
    std::vector<Vec3d>& velocity)
{
    auto& storage = getPressureGridStorage(context);
    if (storage.pressure_velocity.size() != storage.sorted_solid_keys.size())
        throw std::runtime_error("pressure face velocity field is unavailable");
    keys.resize(storage.sorted_solid_keys.size());
    velocity.resize(storage.pressure_velocity.size());
    if (keys.empty())
        return;
    checkCuda(cudaMemcpy(keys.data(),
                  thrust::raw_pointer_cast(storage.sorted_solid_keys.data()),
                  keys.size() * sizeof(GridKey), cudaMemcpyDeviceToHost),
        "pressure face velocity key download");
    checkCuda(cudaMemcpy(velocity.data(),
                  thrust::raw_pointer_cast(storage.pressure_velocity.data()),
                  velocity.size() * sizeof(Vec3d), cudaMemcpyDeviceToHost),
        "pressure face velocity download");
}

static void solveDeviceCsrAmgxInternal(PersistentAmgxCsrSolver& context,
    const GPU_STAGE3A::DeviceCsrMatrix& matrix,
    const std::vector<double>* rhs_host,
    const GPU_STAGE3A::DeviceVector* rhs_device, double diagonal_epsilon,
    std::vector<double>* solution_host, AmgxCsrStats* stats,
    GPU_STAGE3A::DeviceVector* solution_device)
{
    const auto total_begin = std::chrono::high_resolution_clock::now();
    AmgxCsrStats local_stats;
    if (matrix.rows <= 0 || matrix.rows != matrix.cols)
        throw std::runtime_error("AMGX coupling solver requires a square device CSR");
    const std::size_t rhs_size = rhs_host ? rhs_host->size()
        : (rhs_device ? rhs_device->size : 0);
    if (rhs_size != static_cast<std::size_t>(matrix.rows))
        throw std::runtime_error("AMGX coupling RHS size mismatch");
    const GPU_STAGE3A::DeviceCsrRawView view =
        GPU_STAGE3A::getDeviceCsrRawView(matrix);
    if (!view.row_offsets || !view.col_indices || !view.values)
        throw std::runtime_error("AMGX coupling device CSR view is empty");

    auto& storage = getAmgxCsrStorage(context);
    const std::size_t rows_size = static_cast<std::size_t>(matrix.rows);
    const std::size_t nnz = matrix.nonzeros;
    storage.values.resize(nnz);
    storage.solution.resize(rows_size);
    constexpr int block_size = 256;
    const int blocks = (matrix.rows + block_size - 1) / block_size;

    const auto values_copy_begin = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.values.data()), view.values,
                  nnz * sizeof(double), cudaMemcpyDeviceToDevice),
        "AMGX coupling CSR values copy");
    if (diagonal_epsilon != 0.0) {
        addDiagonalToCsrValuesKernel<<<blocks, block_size>>>(view.row_offsets,
            view.col_indices, matrix.rows, diagonal_epsilon,
            thrust::raw_pointer_cast(storage.values.data()));
        checkCuda(cudaGetLastError(), "AMGX coupling diagonal regularization launch");
    }
    checkCuda(cudaDeviceSynchronize(), "AMGX coupling values copy synchronize");
    const auto values_copy_end = std::chrono::high_resolution_clock::now();

    const auto rhs_upload_begin = std::chrono::high_resolution_clock::now();
    const double* rhs_values = nullptr;
    if (rhs_host) {
        storage.rhs.resize(rows_size);
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.rhs.data()),
                      rhs_host->data(), rows_size * sizeof(double),
                      cudaMemcpyHostToDevice),
            "AMGX coupling RHS upload");
        checkCuda(cudaDeviceSynchronize(),
            "AMGX coupling RHS upload synchronize");
        rhs_values = thrust::raw_pointer_cast(storage.rhs.data());
    }
    else {
        const GPU_STAGE3A::DeviceVectorRawView rhs_view =
            GPU_STAGE3A::getDeviceVectorRawView(*rhs_device);
        if (!rhs_view.values)
            throw std::runtime_error("AMGX coupling device RHS view is empty");
        rhs_values = rhs_view.values;
    }
    const auto rhs_upload_end = std::chrono::high_resolution_clock::now();

    const AmgxRuntime& runtime = getCouplingAmgxRuntime();
    AMGX_matrix_handle amgx_matrix = nullptr;
    AMGX_vector_handle amgx_rhs = nullptr;
    AMGX_vector_handle amgx_x = nullptr;
    AMGX_solver_handle amgx_solver = nullptr;
    try {
        checkAmgx(AMGX_matrix_create(&amgx_matrix, runtime.resources, AMGX_mode_dDDI),
            "AMGX coupling matrix create");
        const auto matrix_upload_begin = std::chrono::high_resolution_clock::now();
        checkAmgx(AMGX_matrix_upload_all(amgx_matrix, matrix.rows,
                      static_cast<int>(nnz), 1, 1, view.row_offsets,
                      view.col_indices,
                      thrust::raw_pointer_cast(storage.values.data()), nullptr),
            "AMGX coupling device CSR upload");
        checkCuda(cudaDeviceSynchronize(), "AMGX coupling matrix upload synchronize");
        const auto matrix_upload_end = std::chrono::high_resolution_clock::now();

        checkAmgx(AMGX_vector_create(&amgx_rhs, runtime.resources, AMGX_mode_dDDI),
            "AMGX coupling RHS vector create");
        checkAmgx(AMGX_vector_create(&amgx_x, runtime.resources, AMGX_mode_dDDI),
            "AMGX coupling solution vector create");
        checkAmgx(AMGX_vector_upload(amgx_rhs, matrix.rows, 1,
                      rhs_values),
            "AMGX coupling RHS vector upload");
        checkAmgx(AMGX_vector_set_zero(amgx_x, matrix.rows, 1),
            "AMGX coupling solution zero");

        checkAmgx(AMGX_solver_create(&amgx_solver, runtime.resources,
                      AMGX_mode_dDDI, runtime.config),
            "AMGX coupling solver create");
        const auto setup_begin = std::chrono::high_resolution_clock::now();
        checkAmgx(AMGX_solver_setup(amgx_solver, amgx_matrix),
            "AMGX coupling solver setup");
        checkCuda(cudaDeviceSynchronize(), "AMGX coupling solver setup synchronize");
        const auto setup_end = std::chrono::high_resolution_clock::now();

        const auto solve_begin = std::chrono::high_resolution_clock::now();
        checkAmgx(AMGX_solver_solve(amgx_solver, amgx_rhs, amgx_x),
            "AMGX coupling solver solve");
        checkCuda(cudaDeviceSynchronize(), "AMGX coupling solver solve synchronize");
        const auto solve_end = std::chrono::high_resolution_clock::now();
        AMGX_SOLVE_STATUS solve_status = AMGX_SOLVE_FAILED;
        checkAmgx(AMGX_solver_get_status(amgx_solver, &solve_status),
            "AMGX coupling solver status");
        checkAmgx(AMGX_solver_get_iterations_number(amgx_solver,
                      &local_stats.iterations),
            "AMGX coupling solver iteration count");
        if (solve_status != AMGX_SOLVE_SUCCESS)
            throw std::runtime_error("AMGX coupling solve did not converge");

        const auto download_begin = std::chrono::high_resolution_clock::now();
        checkAmgx(AMGX_vector_download(amgx_x,
                      thrust::raw_pointer_cast(storage.solution.data())),
            "AMGX coupling solution device download");
        if (solution_device) {
            *solution_device = GPU_STAGE3A::makeDeviceVectorFromDevice(
                thrust::raw_pointer_cast(storage.solution.data()), rows_size);
        }
        if (solution_host) {
            solution_host->resize(rows_size);
            checkCuda(cudaMemcpy(solution_host->data(),
                          thrust::raw_pointer_cast(storage.solution.data()),
                          rows_size * sizeof(double), cudaMemcpyDeviceToHost),
                "AMGX coupling solution host download");
        }
        const auto download_end = std::chrono::high_resolution_clock::now();

        local_stats.rows = rows_size;
        local_stats.nnz = nnz;
        local_stats.solve_status = static_cast<int>(solve_status);
        local_stats.diagonal_epsilon = diagonal_epsilon;
        local_stats.values_copy_ms = std::chrono::duration<double, std::milli>(
            values_copy_end - values_copy_begin).count();
        local_stats.rhs_upload_ms = std::chrono::duration<double, std::milli>(
            rhs_upload_end - rhs_upload_begin).count();
        local_stats.matrix_upload_ms = std::chrono::duration<double, std::milli>(
            matrix_upload_end - matrix_upload_begin).count();
        local_stats.setup_ms = std::chrono::duration<double, std::milli>(
            setup_end - setup_begin).count();
        local_stats.solve_ms = std::chrono::duration<double, std::milli>(
            solve_end - solve_begin).count();
        local_stats.solution_download_ms = std::chrono::duration<double, std::milli>(
            download_end - download_begin).count();
        local_stats.total_ms = std::chrono::duration<double, std::milli>(
            download_end - total_begin).count();
    } catch (...) {
        destroyAmgxSolveHandles(amgx_solver, amgx_x, amgx_rhs, amgx_matrix);
        throw;
    }
    destroyAmgxSolveHandles(amgx_solver, amgx_x, amgx_rhs, amgx_matrix);
    if (stats)
        *stats = local_stats;
}

void solveDeviceCsrAmgx3D(PersistentAmgxCsrSolver& context,
    const GPU_STAGE3A::DeviceCsrMatrix& matrix,
    const std::vector<double>& rhs_host, double diagonal_epsilon,
    std::vector<double>& solution, AmgxCsrStats* stats,
    GPU_STAGE3A::DeviceVector* solution_device)
{
    solveDeviceCsrAmgxInternal(context, matrix, &rhs_host, nullptr,
        diagonal_epsilon, &solution, stats, solution_device);
}

void solveDeviceCsrAmgx3D(PersistentAmgxCsrSolver& context,
    const GPU_STAGE3A::DeviceCsrMatrix& matrix,
    const GPU_STAGE3A::DeviceVector& rhs_device, double diagonal_epsilon,
    std::vector<double>* solution_host, AmgxCsrStats* stats,
    GPU_STAGE3A::DeviceVector* solution_device)
{
    solveDeviceCsrAmgxInternal(context, matrix, nullptr, &rhs_device,
        diagonal_epsilon, solution_host, stats, solution_device);
}
#endif

void assemblePersistentPressureCsr3D(
    PersistentPressureCsr3D& context,
    const std::vector<GridKey>& ordered_dof_keys,
    const std::vector<GridKey>& solid_keys,
    const std::vector<unsigned char>& solid_flags,
    const std::vector<double>& rho_star_by_row,
    double rho_rest, double dx, double dt, PressureCsrResult& result,
    PressureCsrStats* stats)
{
    if (solid_keys.size() != solid_flags.size()
        || ordered_dof_keys.size() != rho_star_by_row.size())
        throw std::runtime_error("pressure CSR input size mismatch");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    PressureCsrStats local_stats;
    local_stats.rows = ordered_dof_keys.size();
    result.row_offsets.assign(ordered_dof_keys.size() + 1, 0);
    result.column_indices.clear();
    result.values.clear();
    result.rhs.clear();
    if (ordered_dof_keys.empty()) {
        if (stats)
            *stats = local_stats;
        return;
    }
    if (solid_keys.empty())
        throw std::runtime_error("pressure CSR requires solid flags");

    auto& storage = getPressureCsrStorage(context);
    constexpr int block_size = 256;
    const int rows = static_cast<int>(ordered_dof_keys.size());
    const int blocks = (rows + block_size - 1) / block_size;

    const auto upload_begin = std::chrono::high_resolution_clock::now();
    resizePressureCsr(storage.row_keys, ordered_dof_keys.size(), storage);
    resizePressureCsr(storage.sorted_dof_keys, ordered_dof_keys.size(), storage);
    resizePressureCsr(storage.sorted_dof_indices, ordered_dof_keys.size(), storage);
    resizePressureCsr(storage.sorted_solid_keys, solid_keys.size(), storage);
    resizePressureCsr(storage.sorted_solid_flags, solid_flags.size(), storage);
    resizePressureCsr(storage.rho_star, rho_star_by_row.size(), storage);
    resizePressureCsr(storage.entry_columns, ordered_dof_keys.size() * 7, storage);
    resizePressureCsr(storage.entry_values, ordered_dof_keys.size() * 7, storage);
    resizePressureCsr(storage.entry_counts, ordered_dof_keys.size(), storage);
    resizePressureCsr(storage.rhs, ordered_dof_keys.size(), storage);
    resizePressureCsr(storage.missing_solid, 1, storage);
    checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.row_keys.data()),
                  ordered_dof_keys.data(), ordered_dof_keys.size() * sizeof(GridKey),
                  cudaMemcpyHostToDevice), "pressure CSR DOF upload");
    checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.sorted_solid_keys.data()),
                  solid_keys.data(), solid_keys.size() * sizeof(GridKey),
                  cudaMemcpyHostToDevice), "pressure CSR solid-key upload");
    checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.sorted_solid_flags.data()),
                  solid_flags.data(), solid_flags.size() * sizeof(unsigned char),
                  cudaMemcpyHostToDevice), "pressure CSR solid-flag upload");
    checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.rho_star.data()),
                  rho_star_by_row.data(), rho_star_by_row.size() * sizeof(double),
                  cudaMemcpyHostToDevice), "pressure CSR density upload");
    checkCuda(cudaDeviceSynchronize(), "pressure CSR upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    const auto sort_begin = std::chrono::high_resolution_clock::now();
    thrust::copy(storage.row_keys.begin(), storage.row_keys.end(),
        storage.sorted_dof_keys.begin());
    thrust::sequence(storage.sorted_dof_indices.begin(), storage.sorted_dof_indices.end());
    thrust::sort_by_key(storage.sorted_dof_keys.begin(), storage.sorted_dof_keys.end(),
        storage.sorted_dof_indices.begin());
    thrust::sort_by_key(storage.sorted_solid_keys.begin(), storage.sorted_solid_keys.end(),
        storage.sorted_solid_flags.begin());
    checkCuda(cudaGetLastError(), "pressure CSR lookup sort");
    checkCuda(cudaDeviceSynchronize(), "pressure CSR lookup sort synchronize");
    const auto sort_end = std::chrono::high_resolution_clock::now();

    const auto assembly_begin = std::chrono::high_resolution_clock::now();
    thrust::fill(storage.missing_solid.begin(), storage.missing_solid.end(), 0);
    assemblePressureCsrKernel<<<blocks, block_size>>>(
        thrust::raw_pointer_cast(storage.row_keys.data()), rows,
        thrust::raw_pointer_cast(storage.sorted_dof_keys.data()),
        thrust::raw_pointer_cast(storage.sorted_dof_indices.data()),
        thrust::raw_pointer_cast(storage.sorted_solid_keys.data()),
        thrust::raw_pointer_cast(storage.sorted_solid_flags.data()),
        static_cast<int>(solid_keys.size()),
        thrust::raw_pointer_cast(storage.rho_star.data()), rho_rest, dx, dt,
        thrust::raw_pointer_cast(storage.entry_columns.data()),
        thrust::raw_pointer_cast(storage.entry_values.data()),
        thrust::raw_pointer_cast(storage.entry_counts.data()),
        thrust::raw_pointer_cast(storage.rhs.data()),
        thrust::raw_pointer_cast(storage.missing_solid.data()), nullptr);
    checkCuda(cudaGetLastError(), "pressure CSR assembly launch");
    checkCuda(cudaDeviceSynchronize(), "pressure CSR assembly synchronize");
    const auto assembly_end = std::chrono::high_resolution_clock::now();

    const auto download_begin = std::chrono::high_resolution_clock::now();
    std::vector<int> counts(ordered_dof_keys.size());
    std::vector<int> columns(ordered_dof_keys.size() * 7);
    std::vector<double> values(ordered_dof_keys.size() * 7);
    result.rhs.resize(ordered_dof_keys.size());
    int missing_solid = 0;
    checkCuda(cudaMemcpy(counts.data(), thrust::raw_pointer_cast(storage.entry_counts.data()),
                  counts.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "pressure CSR count download");
    checkCuda(cudaMemcpy(columns.data(), thrust::raw_pointer_cast(storage.entry_columns.data()),
                  columns.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "pressure CSR column download");
    checkCuda(cudaMemcpy(values.data(), thrust::raw_pointer_cast(storage.entry_values.data()),
                  values.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "pressure CSR value download");
    checkCuda(cudaMemcpy(result.rhs.data(), thrust::raw_pointer_cast(storage.rhs.data()),
                  result.rhs.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "pressure CSR RHS download");
    checkCuda(cudaMemcpy(&missing_solid,
                  thrust::raw_pointer_cast(storage.missing_solid.data()), sizeof(int),
                  cudaMemcpyDeviceToHost), "pressure CSR missing-solid download");
    for (std::size_t row = 0; row < counts.size(); ++row) {
        if (counts[row] < 1 || counts[row] > 7)
            throw std::runtime_error("pressure CSR invalid row width");
        result.row_offsets[row + 1] = result.row_offsets[row] + counts[row];
        for (int entry = 0; entry < counts[row]; ++entry) {
            result.column_indices.push_back(columns[row * 7 + entry]);
            result.values.push_back(values[row * 7 + entry]);
        }
    }
    const auto download_end = std::chrono::high_resolution_clock::now();
    local_stats.nnz = result.values.size();
    local_stats.solid_lookup_misses = static_cast<std::size_t>(missing_solid);
    local_stats.reallocations = storage.reallocations;
    local_stats.upload_ms = std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
    local_stats.lookup_sort_ms = std::chrono::duration<double, std::milli>(sort_end - sort_begin).count();
    local_stats.assembly_ms = std::chrono::duration<double, std::milli>(assembly_end - assembly_begin).count();
    local_stats.download_ms = std::chrono::duration<double, std::milli>(download_end - download_begin).count();
    local_stats.total_ms = std::chrono::duration<double, std::milli>(download_end - total_begin).count();
    if (stats)
        *stats = local_stats;
}

void runVelocityGridMassProbe(const std::vector<Vec3d>& xq,
    const std::vector<double>& mass, double dx, bool halfDxAnchor,
    ProbeGridResult& result)
{
    runLinearScalarScatterProbe(xq, mass, dx, halfDxAnchor, result);
}

void runPressureGridCountProbe(const std::vector<Vec3d>& xq,
    double dx, bool halfDxAnchor, ProbeGridResult& result)
{
    result.keys.clear();
    result.mass.clear();
    if (xq.empty())
        return;

    thrust::device_vector<Vec3d> d_xq(xq.begin(), xq.end());
    thrust::device_vector<GridKey> d_sample_keys(xq.size());

    const int block_size = 256;
    const int block_count = static_cast<int>((xq.size() + block_size - 1) / block_size);
    generatePressureKeysKernel<<<block_count, block_size>>>(
        thrust::raw_pointer_cast(d_xq.data()), static_cast<int>(xq.size()), dx,
        halfDxAnchor ? 1 : 0, thrust::raw_pointer_cast(d_sample_keys.data()));
    checkCuda(cudaGetLastError(), "generatePressureKeysKernel launch");
    checkCuda(cudaDeviceSynchronize(), "generatePressureKeysKernel synchronize");

    finishUnitCountProbe(d_sample_keys, result);
}

void runFaceGridCountProbe(const std::vector<Vec3d>& face_xq,
    const std::vector<Vec3d>& face_n, double dx, bool halfDxAnchor,
    ProbeGridResult& result)
{
    if (face_xq.size() != face_n.size())
        throw std::runtime_error("runFaceGridCountProbe input size mismatch");

    result.keys.clear();
    result.mass.clear();
    if (face_xq.empty())
        return;

    thrust::device_vector<Vec3d> d_face_xq(face_xq.begin(), face_xq.end());
    thrust::device_vector<Vec3d> d_face_n(face_n.begin(), face_n.end());
    thrust::device_vector<GridKey> d_sample_keys(face_xq.size() * 2);

    const int block_size = 256;
    const int block_count = static_cast<int>((face_xq.size() + block_size - 1) / block_size);
    generateFaceKeysKernel<<<block_count, block_size>>>(
        thrust::raw_pointer_cast(d_face_xq.data()), thrust::raw_pointer_cast(d_face_n.data()),
        static_cast<int>(face_xq.size()), dx, halfDxAnchor ? 1 : 0,
        thrust::raw_pointer_cast(d_sample_keys.data()));
    checkCuda(cudaGetLastError(), "generateFaceKeysKernel launch");
    checkCuda(cudaDeviceSynchronize(), "generateFaceKeysKernel synchronize");

    finishUnitCountProbe(d_sample_keys, result);
}

ProbeCompareStats compareMassByKey(const std::vector<GridKey>& cpu_keys,
    const std::vector<double>& cpu_mass, const ProbeGridResult& gpu_result)
{
    if (cpu_keys.size() != cpu_mass.size())
        throw std::runtime_error("compareMassByKey CPU input size mismatch");
    if (gpu_result.keys.size() != gpu_result.mass.size())
        throw std::runtime_error("compareMassByKey GPU input size mismatch");

    ProbeCompareStats stats;
    stats.cpu_nodes = cpu_keys.size();
    stats.gpu_nodes = gpu_result.keys.size();

    std::size_t i = 0;
    std::size_t j = 0;
    double sum_abs = 0;
    std::size_t common = 0;
    while (i < cpu_keys.size() && j < gpu_result.keys.size()) {
        if (cpu_keys[i] == gpu_result.keys[j]) {
            const double err = std::abs(cpu_mass[i] - gpu_result.mass[j]);
            stats.max_abs_mass_error = std::max(stats.max_abs_mass_error, err);
            sum_abs += err;
            ++common;
            ++i;
            ++j;
        }
        else if (cpu_keys[i] < gpu_result.keys[j]) {
            ++stats.missing_on_gpu;
            ++i;
        }
        else {
            ++stats.extra_on_gpu;
            ++j;
        }
    }
    stats.missing_on_gpu += cpu_keys.size() - i;
    stats.extra_on_gpu += gpu_result.keys.size() - j;
    stats.mean_abs_mass_error = common ? sum_abs / static_cast<double>(common) : 0.0;
    return stats;
}

void rebuildPersistentSpatialSearch(PersistentSpatialSearch3D& context,
    const std::vector<Vec3d>& positions, double cell_size,
    SpatialSearchStats* stats, const void* resident_positions,
    std::size_t resident_position_count)
{
    if (cell_size <= 0.0)
        throw std::runtime_error("persistent spatial search requires positive cell size");
    if ((resident_positions == nullptr) != (resident_position_count == 0))
        throw std::runtime_error(
            "invalid resident spatial-search particle input");
    const bool use_resident_positions = resident_positions != nullptr;
    const std::size_t particle_count = use_resident_positions
        ? resident_position_count
        : positions.size();
    auto& storage = getSpatialStorage(context);
    storage.cell_size = cell_size;
    const auto t0 = std::chrono::high_resolution_clock::now();
    resizeTracked(storage.positions, particle_count, storage);
    if (use_resident_positions) {
        checkCuda(cudaMemcpy(
                      thrust::raw_pointer_cast(storage.positions.data()),
                      resident_positions, particle_count * sizeof(Vec3d),
                      cudaMemcpyDeviceToDevice),
            "resident spatial-search particle copy");
    }
    else {
        thrust::copy(
            positions.begin(), positions.end(), storage.positions.begin());
    }
    resizeTracked(storage.sorted_keys, particle_count, storage);
    resizeTracked(storage.sorted_ids, particle_count, storage);
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (particle_count != 0) {
        const int block_size = 256;
        const int block_count = static_cast<int>(
            (particle_count + block_size - 1) / block_size);
        generateParticleKeysKernel<<<block_count, block_size>>>(
            thrust::raw_pointer_cast(storage.positions.data()),
            static_cast<int>(particle_count), cell_size,
            thrust::raw_pointer_cast(storage.sorted_keys.data()));
        checkCuda(cudaGetLastError(), "generateParticleKeysKernel launch");
        thrust::sequence(storage.sorted_ids.begin(), storage.sorted_ids.end());
        thrust::stable_sort_by_key(storage.sorted_keys.begin(), storage.sorted_keys.end(),
            storage.sorted_ids.begin());
        checkCuda(cudaDeviceSynchronize(), "persistent spatial search sort synchronize");
    }
    const auto t2 = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->particles = particle_count;
        stats->particle_capacity = storage.positions.capacity();
        stats->reallocations = storage.reallocations;
        stats->resident_particle_positions = use_resident_positions;
        stats->upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->sort_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
}

bool anyPointWithinRadius(PersistentSpatialSearch3D& context,
    const std::vector<Vec3d>& query_positions, double radius,
    SpatialSearchStats* stats, const void* resident_query_positions,
    std::size_t resident_query_count)
{
    auto& storage = getSpatialStorage(context);
    if (radius <= 0.0 || storage.cell_size <= 0.0)
        throw std::runtime_error("invalid persistent spatial search radius");
    if ((resident_query_positions == nullptr) !=
        (resident_query_count == 0))
        throw std::runtime_error(
            "invalid resident spatial-search query input");
    const bool use_resident_queries = resident_query_positions != nullptr;
    const std::size_t query_count = use_resident_queries
        ? resident_query_count : query_positions.size();
    if (query_count == 0 || storage.positions.empty()) {
        if (stats)
            stats->resident_query_positions = use_resident_queries;
        return false;
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    const Vec3d* query_device = static_cast<const Vec3d*>(
        resident_query_positions);
    if (!use_resident_queries) {
        resizeTracked(storage.query_positions, query_count, storage);
        thrust::copy(query_positions.begin(), query_positions.end(),
            storage.query_positions.begin());
        query_device = thrust::raw_pointer_cast(
            storage.query_positions.data());
    }
    resizeTracked(storage.any_flag, 1, storage);
    storage.any_flag[0] = 0;
    const int block_size = 256;
    const int block_count = static_cast<int>(
        (query_count + block_size - 1) / block_size);
    anyPointWithinRadiusKernel<<<block_count, block_size>>>(
        query_device, static_cast<int>(query_count), radius, storage.cell_size,
        thrust::raw_pointer_cast(storage.positions.data()),
        thrust::raw_pointer_cast(storage.sorted_keys.data()),
        thrust::raw_pointer_cast(storage.sorted_ids.data()),
        static_cast<int>(storage.positions.size()),
        thrust::raw_pointer_cast(storage.any_flag.data()));
    checkCuda(cudaGetLastError(), "anyPointWithinRadiusKernel launch");
    checkCuda(cudaDeviceSynchronize(), "anyPointWithinRadiusKernel synchronize");
    int result = 0;
    checkCuda(cudaMemcpy(&result, thrust::raw_pointer_cast(storage.any_flag.data()),
                  sizeof(int), cudaMemcpyDeviceToHost),
        "anyPointWithinRadius result copy");
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->resident_query_positions = use_resident_queries;
        stats->query_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->reallocations = storage.reallocations;
    }
    return result != 0;
}

void computeCubicSphNormals(PersistentSpatialSearch3D& context,
    const std::vector<double>& masses, double radius,
    std::vector<Vec3d>& normals, SpatialSearchStats* stats,
    const void* resident_masses, std::size_t resident_mass_count,
    bool materialize_host_normals)
{
    auto& storage = getSpatialStorage(context);
    if ((resident_masses == nullptr) != (resident_mass_count == 0))
        throw std::runtime_error("invalid resident SPH normal mass input");
    const bool use_resident_masses = resident_masses != nullptr;
    const std::size_t particle_count = use_resident_masses
        ? resident_mass_count
        : masses.size();
    if (particle_count != storage.positions.size())
        throw std::runtime_error("SPH normal mass/position size mismatch");
    if (radius <= 0.0 || storage.cell_size <= 0.0)
        throw std::runtime_error("invalid SPH normal radius");
    const auto t0 = std::chrono::high_resolution_clock::now();
    resizeTracked(storage.normals, particle_count, storage);
    const double* mass_device = static_cast<const double*>(resident_masses);
    if (!use_resident_masses) {
        resizeTracked(storage.masses, particle_count, storage);
        thrust::copy(masses.begin(), masses.end(), storage.masses.begin());
        mass_device = thrust::raw_pointer_cast(storage.masses.data());
    }
    if (particle_count != 0) {
        const int block_size = 256;
        const int block_count = static_cast<int>(
            (particle_count + block_size - 1) / block_size);
        cubicSphNormalsKernel<<<block_count, block_size>>>(
            thrust::raw_pointer_cast(storage.positions.data()),
            thrust::raw_pointer_cast(storage.sorted_keys.data()),
            thrust::raw_pointer_cast(storage.sorted_ids.data()),
            static_cast<int>(storage.positions.size()),
            mass_device, radius,
            storage.cell_size, thrust::raw_pointer_cast(storage.normals.data()));
        checkCuda(cudaGetLastError(), "cubicSphNormalsKernel launch");
        checkCuda(cudaDeviceSynchronize(), "cubicSphNormalsKernel synchronize");
    }
    if (materialize_host_normals) {
        normals.resize(particle_count);
        thrust::copy(storage.normals.begin(), storage.normals.end(),
            normals.begin());
    } else {
        normals.clear();
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->resident_particle_mass = use_resident_masses;
        stats->query_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->reallocations = storage.reallocations;
    }
}

void copyPersistentSpatialSearchFieldsToHost3D(
    PersistentSpatialSearch3D& context, std::vector<Vec3d>& positions,
    std::vector<Vec3d>& normals, bool copy_normals)
{
    auto& storage = getSpatialStorage(context);
    positions.resize(storage.positions.size());
    thrust::copy(storage.positions.begin(), storage.positions.end(),
        positions.begin());
    if (!copy_normals) {
        normals.assign(storage.positions.size(), Vec3d{0.0, 0.0, 0.0});
        return;
    }
    if (storage.normals.size() != storage.positions.size())
        throw std::runtime_error(
            "persistent spatial-search output normals are unavailable");
    normals.resize(storage.normals.size());
    thrust::copy(storage.normals.begin(), storage.normals.end(),
        normals.begin());
}

void rebuildInterfaceQuadAtomic3D(PersistentInterfaceQuad3D& context,
    PersistentSpatialSearch3D& solid_search,
    const std::vector<Vec3d>& fluid_positions,
    const std::vector<double>& fluid_mass,
    const std::vector<Vec3d>& fluid_velocity,
    double dx, const BoxShellCollision3D& collision,
    InterfaceQuadAtomicResult& result, InterfaceQuadAtomicStats* stats,
    const void* resident_fluid_positions,
    std::size_t resident_fluid_count,
    const void* resident_fluid_velocity,
    std::size_t resident_fluid_velocity_count,
    bool compact_host_result,
    bool materialize_host_result)
{
    if ((resident_fluid_positions == nullptr) !=
        (resident_fluid_count == 0))
        throw std::runtime_error(
            "invalid resident interface quadrature position input");
    const bool use_resident_positions = resident_fluid_positions != nullptr;
    if ((resident_fluid_velocity == nullptr) !=
        (resident_fluid_velocity_count == 0))
        throw std::runtime_error(
            "invalid resident interface quadrature velocity input");
    const bool use_resident_velocity = resident_fluid_velocity != nullptr;
    const std::size_t fluid_count = use_resident_positions
        ? resident_fluid_count : fluid_positions.size();
    if (fluid_mass.size() != fluid_count
        || (use_resident_velocity
                ? resident_fluid_velocity_count : fluid_velocity.size())
            != fluid_count)
        throw std::runtime_error("GPU interface quadrature fluid input size mismatch");
    if (!(dx > 0.0))
        throw std::runtime_error("GPU interface quadrature requires positive dx");
    if (!compact_host_result && !materialize_host_result)
        throw std::runtime_error(
            "device-only interface quadrature requires compact mode");
    // `collision.valid == false` is the open-domain counterpart of the CPU
    // predicates: particleInSolid() is then false for both phases, so every
    // solid sample is eligible and every fluid sample can distribute mass.
    // A caller must still gate unknown nonempty collision lists.

    const auto total_begin = std::chrono::high_resolution_clock::now();
    auto& search = getSpatialStorage(solid_search);
    auto& storage = getInterfaceQuadStorage(context);
    const bool use_resident_mass =
        storage.fluid_mass.size() == fluid_count &&
        storage.fluid_mass_host_cache == fluid_mass;
    if (!(search.cell_size > 0.0))
        throw std::runtime_error("GPU interface quadrature requires a solid search");

    result.fluid_mass.clear();
    result.active_solid_ids.clear();
    result.solid_mass.clear();
    result.solid_momentum.clear();
    result.compact_host_result = compact_host_result;
    result.host_result_materialized = materialize_host_result;
    const std::size_t solid_count = search.positions.size();
    if (fluid_count == 0 || solid_count == 0) {
        // Preserve the legacy full-result contract for this degenerate path;
        // there is no meaningful transfer volume to compact and the caller
        // may still need the unmodified fluid masses.
        result.compact_host_result = false;
        result.host_result_materialized = true;
        result.fluid_mass = fluid_mass;
        result.solid_mass.assign(solid_count, 0.0);
        result.solid_momentum.assign(solid_count, Vec3d{0.0, 0.0, 0.0});
        if (stats) {
            stats->fluid_particles = fluid_count;
            stats->solid_particles = solid_count;
            stats->resident_fluid_positions = use_resident_positions;
            stats->resident_fluid_velocity = use_resident_velocity;
            stats->resident_fluid_mass = use_resident_mass;
            stats->reallocations = storage.reallocations;
        }
        return;
    }

    const auto upload_begin = std::chrono::high_resolution_clock::now();
    resizeInterfaceQuad(storage.fluid_positions, fluid_count, storage);
    resizeInterfaceQuad(storage.fluid_mass, fluid_count, storage);
    resizeInterfaceQuad(storage.fluid_velocity, fluid_count, storage);
    resizeInterfaceQuad(storage.adjusted_fluid_mass, fluid_count, storage);
    resizeInterfaceQuad(storage.solid_mass, solid_count, storage);
    resizeInterfaceQuad(storage.solid_momentum, solid_count, storage);
    resizeInterfaceQuad(storage.solid_eligible, solid_count, storage);
    if (use_resident_positions) {
        checkCuda(cudaMemcpy(
                      thrust::raw_pointer_cast(storage.fluid_positions.data()),
                      resident_fluid_positions, fluid_count * sizeof(Vec3d),
                      cudaMemcpyDeviceToDevice),
            "GPU interface quadrature resident position copy");
    } else {
        thrust::copy(fluid_positions.begin(), fluid_positions.end(),
            storage.fluid_positions.begin());
    }
    if (!use_resident_mass) {
        thrust::copy(fluid_mass.begin(), fluid_mass.end(),
            storage.fluid_mass.begin());
        storage.fluid_mass_host_cache = fluid_mass;
    }
    if (use_resident_velocity) {
        checkCuda(cudaMemcpy(
                      thrust::raw_pointer_cast(storage.fluid_velocity.data()),
                      resident_fluid_velocity, fluid_count * sizeof(Vec3d),
                      cudaMemcpyDeviceToDevice),
            "GPU interface quadrature resident velocity copy");
    } else {
        thrust::copy(fluid_velocity.begin(), fluid_velocity.end(),
            storage.fluid_velocity.begin());
    }
    thrust::fill(storage.solid_mass.begin(), storage.solid_mass.end(), 0.0);
    thrust::fill(storage.solid_momentum.begin(), storage.solid_momentum.end(),
        Vec3d{0.0, 0.0, 0.0});
    const int solid_blocks = static_cast<int>((solid_count + 255) / 256);
    markInterfaceQuadSolidEligibleKernel<<<solid_blocks, 256>>>(
        thrust::raw_pointer_cast(search.positions.data()),
        static_cast<int>(solid_count), dx, collision,
        thrust::raw_pointer_cast(storage.solid_eligible.data()));
    checkCuda(cudaGetLastError(), "GPU interface quadrature solid eligibility launch");
    checkCuda(cudaDeviceSynchronize(), "GPU interface quadrature upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();

    const auto fine_grid_begin = std::chrono::high_resolution_clock::now();
    resizeInterfaceQuad(storage.fine_sorted_keys, solid_count, storage);
    resizeInterfaceQuad(storage.fine_sorted_ids, solid_count, storage);
    generateParticleKeysKernel<<<solid_blocks, 256>>>(
        thrust::raw_pointer_cast(search.positions.data()),
        static_cast<int>(solid_count), dx,
        thrust::raw_pointer_cast(storage.fine_sorted_keys.data()));
    checkCuda(cudaGetLastError(), "GPU interface quadrature fine key launch");
    thrust::sequence(storage.fine_sorted_ids.begin(), storage.fine_sorted_ids.end());
    thrust::stable_sort_by_key(storage.fine_sorted_keys.begin(),
        storage.fine_sorted_keys.end(), storage.fine_sorted_ids.begin());
    resizeInterfaceQuad(storage.fine_cell_keys, solid_count, storage);
    resizeInterfaceQuad(storage.fine_cell_counts, solid_count, storage);
    const auto fine_cell_end = thrust::reduce_by_key(
        storage.fine_sorted_keys.begin(), storage.fine_sorted_keys.end(),
        thrust::make_constant_iterator(1), storage.fine_cell_keys.begin(),
        storage.fine_cell_counts.begin());
    storage.fine_cell_count = static_cast<int>(
        fine_cell_end.first - storage.fine_cell_keys.begin());
    storage.fine_cell_keys.resize(storage.fine_cell_count);
    storage.fine_cell_counts.resize(storage.fine_cell_count);
    resizeInterfaceQuad(storage.fine_cell_offsets,
        static_cast<std::size_t>(storage.fine_cell_count) + 1, storage);
    thrust::exclusive_scan(storage.fine_cell_counts.begin(),
        storage.fine_cell_counts.end(), storage.fine_cell_offsets.begin());
    const int fine_particle_count = static_cast<int>(solid_count);
    checkCuda(cudaMemcpy(
        thrust::raw_pointer_cast(storage.fine_cell_offsets.data())
            + storage.fine_cell_count,
        &fine_particle_count, sizeof(int), cudaMemcpyHostToDevice),
        "GPU interface quadrature fine offset copy");
    checkCuda(cudaDeviceSynchronize(), "GPU interface quadrature fine grid synchronize");
    const auto fine_grid_end = std::chrono::high_resolution_clock::now();

    const auto accumulation_begin = std::chrono::high_resolution_clock::now();
    constexpr int block_size = 256;
    const int blocks = static_cast<int>((fluid_count + block_size - 1) / block_size);
    distributeInterfaceQuadAtomicKernel<<<blocks, block_size>>>(
        thrust::raw_pointer_cast(storage.fluid_positions.data()),
        thrust::raw_pointer_cast(storage.fluid_mass.data()),
        thrust::raw_pointer_cast(storage.fluid_velocity.data()),
        static_cast<int>(fluid_count),
        thrust::raw_pointer_cast(search.positions.data()),
        thrust::raw_pointer_cast(storage.fine_cell_keys.data()),
        thrust::raw_pointer_cast(storage.fine_cell_offsets.data()), storage.fine_cell_count,
        thrust::raw_pointer_cast(storage.fine_sorted_ids.data()),
        thrust::raw_pointer_cast(storage.solid_eligible.data()),
        dx, dx, collision,
        thrust::raw_pointer_cast(storage.adjusted_fluid_mass.data()),
        thrust::raw_pointer_cast(storage.solid_mass.data()),
        thrust::raw_pointer_cast(storage.solid_momentum.data()));
    checkCuda(cudaGetLastError(), "GPU interface quadrature atomic launch");
    checkCuda(cudaDeviceSynchronize(), "GPU interface quadrature atomic synchronize");
    const auto accumulation_end = std::chrono::high_resolution_clock::now();

    const auto download_begin = std::chrono::high_resolution_clock::now();
    std::size_t active_solid_count = 0;
    if (compact_host_result) {
        resizeInterfaceQuad(storage.active_solid_ids, solid_count, storage);
        const auto ids_begin = thrust::make_counting_iterator<int>(0);
        const auto active_end = thrust::copy_if(
            ids_begin, ids_begin + solid_count, storage.solid_mass.begin(),
            storage.active_solid_ids.begin(), IsPositive{});
        active_solid_count = static_cast<std::size_t>(
            active_end - storage.active_solid_ids.begin());
        storage.active_solid_ids.resize(active_solid_count);
        resizeInterfaceQuad(storage.active_solid_mass, active_solid_count,
            storage);
        resizeInterfaceQuad(storage.active_solid_momentum, active_solid_count,
            storage);
        thrust::gather(storage.active_solid_ids.begin(),
            storage.active_solid_ids.end(), storage.solid_mass.begin(),
            storage.active_solid_mass.begin());
        thrust::gather(storage.active_solid_ids.begin(),
            storage.active_solid_ids.end(), storage.solid_momentum.begin(),
            storage.active_solid_momentum.begin());
        if (materialize_host_result) {
            result.active_solid_ids.resize(active_solid_count);
            result.solid_mass.resize(active_solid_count);
            result.solid_momentum.resize(active_solid_count);
            thrust::copy(storage.active_solid_ids.begin(),
                storage.active_solid_ids.end(), result.active_solid_ids.begin());
            thrust::copy(storage.active_solid_mass.begin(),
                storage.active_solid_mass.end(), result.solid_mass.begin());
            thrust::copy(storage.active_solid_momentum.begin(),
                storage.active_solid_momentum.end(), result.solid_momentum.begin());
        }
    } else {
        result.fluid_mass.resize(fluid_count);
        result.solid_mass.resize(solid_count);
        result.solid_momentum.resize(solid_count);
        thrust::copy(storage.adjusted_fluid_mass.begin(),
            storage.adjusted_fluid_mass.end(), result.fluid_mass.begin());
        thrust::copy(storage.solid_mass.begin(), storage.solid_mass.end(),
            result.solid_mass.begin());
        thrust::copy(storage.solid_momentum.begin(),
            storage.solid_momentum.end(), result.solid_momentum.begin());
        active_solid_count = static_cast<std::size_t>(std::count_if(
            result.solid_mass.begin(), result.solid_mass.end(),
            [](double mass) { return mass > 0.0; }));
    }
    const auto download_end = std::chrono::high_resolution_clock::now();

    if (stats) {
        stats->fluid_particles = fluid_count;
        stats->solid_particles = solid_count;
        stats->active_solid_particles = active_solid_count;
        stats->reallocations = storage.reallocations;
        stats->resident_fluid_positions = use_resident_positions;
        stats->resident_fluid_velocity = use_resident_velocity;
        stats->resident_fluid_mass = use_resident_mass;
        stats->fluid_upload_ms = std::chrono::duration<double, std::milli>(
            upload_end - upload_begin).count();
        stats->fine_grid_ms = std::chrono::duration<double, std::milli>(
            fine_grid_end - fine_grid_begin).count();
        stats->accumulation_ms = std::chrono::duration<double, std::milli>(
            accumulation_end - accumulation_begin).count();
        stats->download_ms = std::chrono::duration<double, std::milli>(
            download_end - download_begin).count();
        stats->total_ms = std::chrono::duration<double, std::milli>(
            download_end - total_begin).count();
    }
}

void buildPersistentInterfaceQuadratureFields3D(
    PersistentInterfaceQuad3D& context,
    PersistentSpatialSearch3D& solid_search,
    const GPU_STAGE3A::SolidParticleStateDeviceView& solid_particles,
    InterfaceQuadratureDeviceView3D& view)
{
    view = {};
    auto& storage = getInterfaceQuadStorage(context);
    auto& search = getSpatialStorage(solid_search);
    const std::size_t solid_count = search.positions.size();
    if (solid_particles.size != solid_count || !solid_particles.rest_x
        || !solid_particles.volume)
        throw std::runtime_error(
            "resident interface quadrature fields require current solid state");
    const std::size_t active_count = storage.active_solid_ids.size();
    resizeInterfaceQuad(storage.interface_rest_x, active_count, storage);
    resizeInterfaceQuad(storage.interface_current_x, active_count, storage);
    resizeInterfaceQuad(storage.interface_normal, active_count, storage);
    resizeInterfaceQuad(storage.interface_length, active_count, storage);
    if (active_count != 0) {
        if (!solid_particles.current_x ||
            search.normals.size() != solid_count)
            throw std::runtime_error(
                "resident interface quadrature fields require current solid state and normals");
        constexpr int block_size = 256;
        const int blocks = static_cast<int>((active_count + block_size - 1)
            / block_size);
        gatherInterfaceQuadratureFieldsKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.active_solid_ids.data()),
            static_cast<int>(active_count), solid_particles.rest_x,
            thrust::raw_pointer_cast(search.positions.data()),
            thrust::raw_pointer_cast(search.normals.data()),
            solid_particles.volume,
            thrust::raw_pointer_cast(storage.interface_rest_x.data()),
            thrust::raw_pointer_cast(storage.interface_current_x.data()),
            thrust::raw_pointer_cast(storage.interface_normal.data()),
            thrust::raw_pointer_cast(storage.interface_length.data()));
        checkCuda(cudaGetLastError(),
            "resident interface quadrature field gather launch");
        checkCuda(cudaDeviceSynchronize(),
            "resident interface quadrature field gather synchronize");
    }
    view.rest_x = active_count == 0 ? nullptr
        : thrust::raw_pointer_cast(storage.interface_rest_x.data());
    view.current_x = active_count == 0 ? nullptr
        : thrust::raw_pointer_cast(storage.interface_current_x.data());
    view.normal = active_count == 0 ? nullptr
        : thrust::raw_pointer_cast(storage.interface_normal.data());
    view.length = active_count == 0 ? nullptr
        : thrust::raw_pointer_cast(storage.interface_length.data());
    view.size = active_count;
}

void copyInterfaceQuadratureFieldsToHost3D(
    const InterfaceQuadratureDeviceView3D& view,
    std::vector<Vec3d>& rest_x, std::vector<Vec3d>& current_x,
    std::vector<Vec3d>& normal, std::vector<double>& length)
{
    if (view.size != 0 &&
        (!view.rest_x || !view.current_x || !view.normal || !view.length))
        throw std::runtime_error(
            "incomplete resident interface quadrature device view");
    rest_x.resize(view.size);
    current_x.resize(view.size);
    normal.resize(view.size);
    length.resize(view.size);
    if (view.size == 0)
        return;
    checkCuda(cudaMemcpy(rest_x.data(), view.rest_x,
                  view.size * sizeof(Vec3d), cudaMemcpyDeviceToHost),
        "resident interface rest-X diagnostic download");
    checkCuda(cudaMemcpy(current_x.data(), view.current_x,
                  view.size * sizeof(Vec3d), cudaMemcpyDeviceToHost),
        "resident interface current-X diagnostic download");
    checkCuda(cudaMemcpy(normal.data(), view.normal,
                  view.size * sizeof(Vec3d), cudaMemcpyDeviceToHost),
        "resident interface normal diagnostic download");
    checkCuda(cudaMemcpy(length.data(), view.length,
                  view.size * sizeof(double), cudaMemcpyDeviceToHost),
        "resident interface length diagnostic download");
}

void bindInterfaceQuadratureClampedCurrent3D(
    PersistentFluidAnalytic3D& fluid_context,
    InterfaceQuadratureDeviceView3D& view)
{
    auto& storage = getFluidStorage(fluid_context);
    if (storage.interface_x.size() != view.size)
        throw std::runtime_error(
            "clamped interface coordinate count does not match resident view");
    view.current_x = view.size == 0 ? nullptr
        : thrust::raw_pointer_cast(storage.interface_x.data());
}

void rebuildFluidAnalyticQuadratureDeviceInput(
    PersistentFluidAnalyticStorage& storage, std::size_t input_count,
    std::size_t interface_count, double dx, double rho,
    const BoxShellCollision3D& collision, FluidAnalyticResult& result,
    FluidAnalyticStats* stats, bool materialize_host_result)
{
    // `collision.valid == false` is the supported open-domain mode: it
    // exactly matches an empty CPU collision-object list, so prune and
    // solid-wall face emission both become no-ops.  Unknown nonempty
    // collision lists are gated by the caller and never reach this kernel.
    if (dx <= 0.0 || rho <= 0.0)
        throw std::runtime_error("invalid fluid analytic quadrature configuration");
    const auto t0 = std::chrono::high_resolution_clock::now();
    const int block_size = 256;

    resizeFluid(storage.input_keys, input_count, storage);
    resizeFluid(storage.keep, input_count, storage);
    if (input_count) {
        const int blocks = static_cast<int>((input_count + block_size - 1) / block_size);
        markFluidPruneKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.input_x.data()),
            static_cast<int>(input_count), dx, collision,
            thrust::raw_pointer_cast(storage.input_keys.data()),
            thrust::raw_pointer_cast(storage.keep.data()));
        checkCuda(cudaGetLastError(), "markFluidPruneKernel launch");
    }

    resizeFluid(storage.pruned_x, input_count, storage);
    resizeFluid(storage.pruned_mass, input_count, storage);
    resizeFluid(storage.pruned_keys, input_count, storage);
    auto input_begin = thrust::make_zip_iterator(thrust::make_tuple(
        storage.input_x.begin(), storage.input_mass.begin(), storage.input_keys.begin()));
    auto input_end = input_begin + input_count;
    auto pruned_begin = thrust::make_zip_iterator(thrust::make_tuple(
        storage.pruned_x.begin(), storage.pruned_mass.begin(), storage.pruned_keys.begin()));
    const auto pruned_end = thrust::copy_if(input_begin, input_end,
        storage.keep.begin(), pruned_begin, IsNonZero{});
    const std::size_t pruned_count =
        static_cast<std::size_t>(pruned_end - pruned_begin);
    storage.pruned_x.resize(pruned_count);
    storage.pruned_mass.resize(pruned_count);
    storage.pruned_keys.resize(pruned_count);

    resizeFluid(storage.active_keys, pruned_count, storage);
    thrust::copy(storage.pruned_keys.begin(), storage.pruned_keys.end(),
        storage.active_keys.begin());
    thrust::sort(storage.active_keys.begin(), storage.active_keys.end());
    const auto active_end = thrust::unique(
        storage.active_keys.begin(), storage.active_keys.end());
    const std::size_t active_count =
        static_cast<std::size_t>(active_end - storage.active_keys.begin());
    storage.active_keys.resize(active_count);
    resizeFluid(storage.cell_flags, active_count, storage);
    if (active_count) {
        const int blocks = static_cast<int>((active_count + block_size - 1) / block_size);
        classifyFluidCellsKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.active_keys.data()),
            static_cast<int>(active_count), dx, collision,
            thrust::raw_pointer_cast(storage.cell_flags.data()));
        checkCuda(cudaGetLastError(), "classifyFluidCellsKernel launch");
    }

    if (interface_count) {
        const int blocks = static_cast<int>((interface_count + block_size - 1) / block_size);
        markInterfaceCellsKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.interface_x.data()),
            static_cast<int>(interface_count), dx,
            thrust::raw_pointer_cast(storage.active_keys.data()),
            static_cast<int>(active_count),
            thrust::raw_pointer_cast(storage.cell_flags.data()));
        checkCuda(cudaGetLastError(), "markInterfaceCellsKernel launch");
        clampInterfaceKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.interface_x.data()),
            static_cast<int>(interface_count), dx);
        checkCuda(cudaGetLastError(), "clampInterfaceKernel launch");
    }

    resizeFluid(storage.cell_first_sample, active_count, storage);
    thrust::fill(storage.cell_first_sample.begin(),
        storage.cell_first_sample.end(), std::numeric_limits<int>::max());
    if (pruned_count) {
        const int blocks =
            static_cast<int>((pruned_count + block_size - 1) / block_size);
        markFirstFluidSamplesKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.pruned_keys.data()),
            static_cast<int>(pruned_count),
            thrust::raw_pointer_cast(storage.active_keys.data()),
            static_cast<int>(active_count),
            thrust::raw_pointer_cast(storage.cell_first_sample.data()));
        checkCuda(cudaGetLastError(), "markFirstFluidSamplesKernel launch");
    }
    resizeFluid(storage.face_cell_order, active_count, storage);
    resizeFluid(storage.face_cell_order_keys, active_count, storage);
    thrust::sequence(storage.face_cell_order.begin(),
        storage.face_cell_order.end());
    thrust::copy(storage.cell_first_sample.begin(),
        storage.cell_first_sample.end(), storage.face_cell_order_keys.begin());
    thrust::sort_by_key(storage.face_cell_order_keys.begin(),
        storage.face_cell_order_keys.end(), storage.face_cell_order.begin());

    const std::size_t face_candidate_count = active_count * 6;
    resizeFluid(storage.face_candidates, face_candidate_count, storage);
    resizeFluid(storage.face_normal_candidates, face_candidate_count, storage);
    resizeFluid(storage.face_keep, face_candidate_count, storage);
    if (active_count) {
        const int blocks = static_cast<int>((active_count + block_size - 1) / block_size);
        emitFluidFacesKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.active_keys.data()),
            thrust::raw_pointer_cast(storage.face_cell_order.data()),
            static_cast<int>(active_count), dx, collision,
            thrust::raw_pointer_cast(storage.face_candidates.data()),
            thrust::raw_pointer_cast(storage.face_normal_candidates.data()),
            thrust::raw_pointer_cast(storage.face_keep.data()));
        checkCuda(cudaGetLastError(), "emitFluidFacesKernel launch");
    }
    resizeFluid(storage.face_x, face_candidate_count, storage);
    resizeFluid(storage.face_normal, face_candidate_count, storage);
    auto face_in = thrust::make_zip_iterator(thrust::make_tuple(
        storage.face_candidates.begin(), storage.face_normal_candidates.begin()));
    auto face_out = thrust::make_zip_iterator(thrust::make_tuple(
        storage.face_x.begin(), storage.face_normal.begin()));
    const auto face_end = thrust::copy_if(face_in,
        face_in + face_candidate_count, storage.face_keep.begin(), face_out,
        IsNonZero{});
    const std::size_t face_count = static_cast<std::size_t>(face_end - face_out);
    storage.face_x.resize(face_count);
    storage.face_normal.resize(face_count);

    resizeFluid(storage.face_node_keys, face_count * 2, storage);
    if (face_count) {
        const int blocks =
            static_cast<int>((face_count + block_size - 1) / block_size);
        generateFaceKeysKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.face_x.data()),
            thrust::raw_pointer_cast(storage.face_normal.data()),
            static_cast<int>(face_count), dx, 1,
            thrust::raw_pointer_cast(storage.face_node_keys.data()));
        checkCuda(cudaGetLastError(), "fluid analytic face key launch");
        thrust::sort(storage.face_node_keys.begin(),
            storage.face_node_keys.end());
        const auto face_key_end = thrust::unique(
            storage.face_node_keys.begin(), storage.face_node_keys.end());
        storage.face_node_keys.resize(static_cast<std::size_t>(
            face_key_end - storage.face_node_keys.begin()));
    }

    resizeFluid(storage.output_candidates, pruned_count, storage);
    resizeFluid(storage.output_mass_candidates, pruned_count, storage);
    resizeFluid(storage.output_keep, pruned_count, storage);
    if (pruned_count) {
        const int blocks = static_cast<int>((pruned_count + block_size - 1) / block_size);
        buildFluidOutputCandidatesKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.pruned_x.data()),
            thrust::raw_pointer_cast(storage.pruned_mass.data()),
            thrust::raw_pointer_cast(storage.pruned_keys.data()),
            static_cast<int>(pruned_count),
            thrust::raw_pointer_cast(storage.active_keys.data()),
            static_cast<int>(active_count),
            thrust::raw_pointer_cast(storage.cell_flags.data()),
            thrust::raw_pointer_cast(storage.cell_first_sample.data()), dx, rho,
            thrust::raw_pointer_cast(storage.output_candidates.data()),
            thrust::raw_pointer_cast(storage.output_mass_candidates.data()),
            thrust::raw_pointer_cast(storage.output_keep.data()));
        checkCuda(cudaGetLastError(), "buildFluidOutputCandidatesKernel launch");
    }
    resizeFluid(storage.output_x, pruned_count, storage);
    resizeFluid(storage.output_mass, pruned_count, storage);
    auto output_in = thrust::make_zip_iterator(thrust::make_tuple(
        storage.output_candidates.begin(), storage.output_mass_candidates.begin()));
    auto output_out = thrust::make_zip_iterator(thrust::make_tuple(
        storage.output_x.begin(), storage.output_mass.begin()));
    const auto output_end = thrust::copy_if(output_in,
        output_in + pruned_count, storage.output_keep.begin(), output_out,
        IsNonZero{});
    const std::size_t output_count =
        static_cast<std::size_t>(output_end - output_out);
    storage.output_x.resize(output_count);
    storage.output_mass.resize(output_count);
    checkCuda(cudaDeviceSynchronize(), "fluid analytic quadrature synchronize");

    result = {};
    if (materialize_host_result) {
        result.volume_xq.resize(output_count);
        result.volume_mass.resize(output_count);
        result.face_xq.resize(face_count);
        result.face_normal.resize(face_count);
        result.interface_xq.resize(interface_count);
        result.pressure_keys.resize(active_count);
        result.face_keys.resize(storage.face_node_keys.size());
        thrust::copy(storage.output_x.begin(), storage.output_x.end(),
            result.volume_xq.begin());
        thrust::copy(storage.output_mass.begin(), storage.output_mass.end(),
            result.volume_mass.begin());
        thrust::copy(storage.face_x.begin(), storage.face_x.end(),
            result.face_xq.begin());
        thrust::copy(storage.face_normal.begin(), storage.face_normal.end(),
            result.face_normal.begin());
        thrust::copy(storage.interface_x.begin(), storage.interface_x.end(),
            result.interface_xq.begin());
        thrust::copy(storage.active_keys.begin(), storage.active_keys.end(),
            result.pressure_keys.begin());
        thrust::copy(storage.face_node_keys.begin(), storage.face_node_keys.end(),
            result.face_keys.begin());
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->input_samples = input_count;
        stats->pruned_samples = pruned_count;
        stats->active_cells = active_count;
        stats->output_samples = output_count;
        stats->face_samples = face_count;
        stats->reallocations = storage.reallocations;
        stats->total_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
}

void rebuildFluidAnalyticQuadrature(PersistentFluidAnalytic3D& context,
    const std::vector<Vec3d>& volume_xq,
    const std::vector<double>& volume_mass,
    const std::vector<Vec3d>& interface_xq, double dx, double rho,
    const BoxShellCollision3D& collision, FluidAnalyticResult& result,
    FluidAnalyticStats* stats, const void* resident_fluid_velocity,
    const void* resident_fluid_grad_velocity,
    std::size_t resident_fluid_count, double apic_rpic_ratio,
    bool materialize_host_result)
{
    if (volume_xq.size() != volume_mass.size())
        throw std::runtime_error("fluid analytic quadrature input size mismatch");
    const bool resident_velocity_affine = resident_fluid_velocity != nullptr ||
        resident_fluid_grad_velocity != nullptr || resident_fluid_count != 0;
    if (resident_velocity_affine &&
        (!resident_fluid_velocity || !resident_fluid_grad_velocity ||
            resident_fluid_count != volume_xq.size()))
        throw std::runtime_error(
            "invalid resident fluid velocity/gradient input");
    auto& storage = getFluidStorage(context);
    storage.input_velocity_resident = false;
    storage.input_affine_resident = false;
    resizeFluid(storage.input_x, volume_xq.size(), storage);
    resizeFluid(storage.input_mass, volume_mass.size(), storage);
    resizeFluid(storage.interface_x, interface_xq.size(), storage);
    thrust::copy(volume_xq.begin(), volume_xq.end(), storage.input_x.begin());
    thrust::copy(volume_mass.begin(), volume_mass.end(), storage.input_mass.begin());
    thrust::copy(interface_xq.begin(), interface_xq.end(), storage.interface_x.begin());
    if (resident_velocity_affine) {
        resizeFluid(storage.input_velocity, resident_fluid_count, storage);
        resizeFluid(storage.input_affine, resident_fluid_count, storage);
        if (resident_fluid_count) {
            checkCuda(cudaMemcpy(
                          thrust::raw_pointer_cast(storage.input_velocity.data()),
                          resident_fluid_velocity,
                          resident_fluid_count * sizeof(Vec3d),
                          cudaMemcpyDeviceToDevice),
                "resident analytic fluid velocity copy");
            constexpr int block_size = 256;
            const int blocks = static_cast<int>(
                (resident_fluid_count + block_size - 1) / block_size);
            buildInterfaceQuadFluidAffineKernel<<<blocks, block_size>>>(
                static_cast<const Mat3d*>(resident_fluid_grad_velocity),
                static_cast<int>(resident_fluid_count), apic_rpic_ratio,
                thrust::raw_pointer_cast(storage.input_affine.data()));
            checkCuda(cudaGetLastError(),
                "resident analytic fluid affine launch");
        }
        storage.input_velocity_resident = true;
        storage.input_affine_resident = true;
    }
    rebuildFluidAnalyticQuadratureDeviceInput(storage, volume_xq.size(),
        interface_xq.size(), dx, rho, collision, result, stats,
        materialize_host_result);
}

FluidAnalyticDeviceView3D getFluidAnalyticDeviceView3D(
    const PersistentFluidAnalytic3D& context)
{
    if (!context.storage)
        return {};
    const auto storage =
        std::static_pointer_cast<PersistentFluidAnalyticStorage>(context.storage);
    return {
        storage->output_x.empty() ? nullptr
            : thrust::raw_pointer_cast(storage->output_x.data()),
        storage->output_mass.empty() ? nullptr
            : thrust::raw_pointer_cast(storage->output_mass.data()),
        storage->output_x.size(),
        storage->face_x.empty() ? nullptr
            : thrust::raw_pointer_cast(storage->face_x.data()),
        storage->face_normal.empty() ? nullptr
            : thrust::raw_pointer_cast(storage->face_normal.data()),
        storage->face_x.size()};
}

void copyFluidAnalyticFaceFieldsToHost3D(
    const PersistentFluidAnalytic3D& context,
    std::vector<Vec3d>& face_xq, std::vector<Vec3d>& face_normal)
{
    if (!context.storage) {
        face_xq.clear();
        face_normal.clear();
        return;
    }
    const auto storage =
        std::static_pointer_cast<PersistentFluidAnalyticStorage>(context.storage);
    face_xq.resize(storage->face_x.size());
    face_normal.resize(storage->face_normal.size());
    thrust::copy(
        storage->face_x.begin(), storage->face_x.end(), face_xq.begin());
    thrust::copy(storage->face_normal.begin(), storage->face_normal.end(),
        face_normal.begin());
}

void buildFluidLegacyDofNodeMaps3D(PersistentFluidAnalytic3D& context,
    double dx, int block_x, int block_y, int block_z,
    GPU_STAGE3A::DeviceGridNodeMap& pressure_nodes,
    GPU_STAGE3A::DeviceGridNodeMap& face_nodes,
    FluidLegacyDofOrderStats* stats)
{
    if (!context.storage || !(dx > 0.0) || block_x <= 0 || block_y <= 0 ||
        block_z <= 0)
        throw std::runtime_error("invalid fluid legacy DOF map input");
    auto& storage = getFluidStorage(context);
    const auto total_begin = std::chrono::high_resolution_clock::now();
    FluidLegacyDofOrderStats local;
    constexpr int threads = 256;

    auto build_map = [&](const thrust::device_vector<Vec3d>& source_x,
                         const thrust::device_vector<GridKey>& node_keys,
                         int page_start, int page_end) {
        const std::size_t source_count = source_x.size();
        const std::size_t node_count = node_keys.size();
        const int side = page_end - page_start + 1;
        const int candidates_per_source = 1 + side * side * side;
        resizeFluid(storage.dof_source_base_keys, source_count, storage);
        if (source_count != 0) {
            const int blocks = static_cast<int>(
                (source_count + threads - 1) / threads);
            generatePressureKeysKernel<<<blocks, threads>>>(
                thrust::raw_pointer_cast(source_x.data()),
                static_cast<int>(source_count), dx, 1,
                thrust::raw_pointer_cast(storage.dof_source_base_keys.data()));
            checkCuda(cudaGetLastError(),
                "fluid legacy DOF source-key launch");
        }

        const std::size_t candidate_count =
            source_count * static_cast<std::size_t>(candidates_per_source);
        resizeFluid(storage.dof_page_candidates, candidate_count, storage);
        resizeFluid(storage.dof_page_candidate_ranks, candidate_count, storage);
        if (source_count != 0) {
            const int blocks = static_cast<int>(
                (source_count + threads - 1) / threads);
            generateFluidDofPageCandidatesKernel<<<blocks, threads>>>(
                thrust::raw_pointer_cast(storage.dof_source_base_keys.data()),
                static_cast<int>(source_count), block_x, block_y, block_z,
                page_start, page_end, candidates_per_source,
                thrust::raw_pointer_cast(storage.dof_page_candidates.data()),
                thrust::raw_pointer_cast(
                    storage.dof_page_candidate_ranks.data()));
            checkCuda(cudaGetLastError(),
                "fluid legacy DOF page candidate launch");
        }
        auto candidates = thrust::make_zip_iterator(thrust::make_tuple(
            storage.dof_page_candidates.begin(),
            storage.dof_page_candidate_ranks.begin()));
        const auto valid_end = thrust::remove_if(candidates,
            candidates + candidate_count, IsInvalidVirtualPage{});
        const std::size_t valid_count = static_cast<std::size_t>(
            valid_end - candidates);
        storage.dof_page_candidates.resize(valid_count);
        storage.dof_page_candidate_ranks.resize(valid_count);
        thrust::sort_by_key(storage.dof_page_candidates.begin(),
            storage.dof_page_candidates.end(),
            storage.dof_page_candidate_ranks.begin());
        resizeFluid(storage.dof_page_keys, valid_count, storage);
        resizeFluid(storage.dof_page_first_ranks, valid_count, storage);
        const auto reduced_end = thrust::reduce_by_key(
            storage.dof_page_candidates.begin(),
            storage.dof_page_candidates.end(),
            storage.dof_page_candidate_ranks.begin(),
            storage.dof_page_keys.begin(),
            storage.dof_page_first_ranks.begin(), thrust::equal_to<GridKey>(),
            thrust::minimum<std::uint64_t>());
        const std::size_t page_count = static_cast<std::size_t>(
            reduced_end.first - storage.dof_page_keys.begin());
        storage.dof_page_keys.resize(page_count);
        storage.dof_page_first_ranks.resize(page_count);

        // Convert unique first-stream ranks into dense page traversal ranks,
        // then sort the lookup back by page key for device binary search.
        thrust::sort_by_key(storage.dof_page_first_ranks.begin(),
            storage.dof_page_first_ranks.end(), storage.dof_page_keys.begin());
        resizeFluid(storage.dof_page_activation_ranks, page_count, storage);
        thrust::sequence(storage.dof_page_activation_ranks.begin(),
            storage.dof_page_activation_ranks.end());
        thrust::sort_by_key(storage.dof_page_keys.begin(),
            storage.dof_page_keys.end(),
            storage.dof_page_activation_ranks.begin());

        resizeFluid(storage.dof_node_order, node_count, storage);
        resizeFluid(storage.dof_node_ids, node_count, storage);
        resizeFluid(storage.dof_nodes, node_count, storage);
        resizeFluid(storage.dof_missing_pages, 1, storage);
        thrust::fill(storage.dof_missing_pages.begin(),
            storage.dof_missing_pages.end(), 0);
        if (node_count != 0) {
            const int blocks = static_cast<int>(
                (node_count + threads - 1) / threads);
            buildFluidLegacyNodeOrderKernel<<<blocks, threads>>>(
                thrust::raw_pointer_cast(node_keys.data()),
                static_cast<int>(node_count),
                thrust::raw_pointer_cast(storage.dof_page_keys.data()),
                thrust::raw_pointer_cast(
                    storage.dof_page_activation_ranks.data()),
                static_cast<int>(page_count), block_x, block_y, block_z,
                thrust::raw_pointer_cast(storage.dof_node_order.data()),
                thrust::raw_pointer_cast(storage.dof_nodes.data()),
                thrust::raw_pointer_cast(storage.dof_missing_pages.data()));
            checkCuda(cudaGetLastError(),
                "fluid legacy DOF node-order launch");
        }
        thrust::sequence(storage.dof_node_ids.begin(),
            storage.dof_node_ids.end());
        thrust::sort_by_key(storage.dof_node_order.begin(),
            storage.dof_node_order.end(), storage.dof_node_ids.begin());
        if (node_count != 0) {
            const int blocks = static_cast<int>(
                (node_count + threads - 1) / threads);
            scatterFluidLegacyNodeRanksKernel<<<blocks, threads>>>(
                thrust::raw_pointer_cast(storage.dof_node_ids.data()),
                static_cast<int>(node_count),
                thrust::raw_pointer_cast(storage.dof_nodes.data()));
            checkCuda(cudaGetLastError(),
                "fluid legacy DOF rank scatter launch");
        }
        checkCuda(cudaDeviceSynchronize(),
            "fluid legacy DOF map synchronize");
        if (storage.dof_missing_pages[0] != 0)
            throw std::runtime_error(
                "fluid legacy DOF node page missing from pollute order");
        return GPU_STAGE3A::makeDeviceGridNodeMapFromDeviceNodes(
            node_count == 0 ? nullptr
                            : thrust::raw_pointer_cast(storage.dof_nodes.data()),
            node_count);
    };

    const auto pressure_begin = std::chrono::high_resolution_clock::now();
    pressure_nodes = build_map(storage.output_x, storage.active_keys, 0, 1);
    const auto pressure_end = std::chrono::high_resolution_clock::now();
    face_nodes = build_map(storage.face_x, storage.face_node_keys, -1, 1);
    const auto face_end = std::chrono::high_resolution_clock::now();
    local.pressure_nodes = pressure_nodes.size;
    local.face_nodes = face_nodes.size;
    local.pressure_order_ms = std::chrono::duration<double, std::milli>(
        pressure_end - pressure_begin).count();
    local.face_order_ms = std::chrono::duration<double, std::milli>(
        face_end - pressure_end).count();
    local.total_ms = std::chrono::duration<double, std::milli>(
        face_end - total_begin).count();
    if (stats)
        *stats = local;
}

void rebuildFluidAnalyticQuadratureFromInterfaceQuad3D(
    PersistentFluidAnalytic3D& fluid_context,
    PersistentInterfaceQuad3D& interface_context,
    PersistentSpatialSearch3D& solid_search, double dx, double rho,
    const BoxShellCollision3D& collision, FluidAnalyticResult& result,
    FluidAnalyticStats* stats, const void* resident_fluid_grad_velocity,
    std::size_t resident_fluid_grad_velocity_count,
    double apic_rpic_ratio,
    const std::vector<Mat3d>& host_fluid_affine,
    bool materialize_host_result)
{
    if (dx <= 0.0 || rho <= 0.0)
        throw std::runtime_error("invalid interface analytic quadrature configuration");
    auto& interface_storage = getInterfaceQuadStorage(interface_context);
    auto& solid_storage = getSpatialStorage(solid_search);
    auto& fluid_storage = getFluidStorage(fluid_context);
    const std::size_t fluid_count = interface_storage.fluid_positions.size();
    const std::size_t solid_count = interface_storage.solid_mass.size();
    if (solid_storage.positions.size() != solid_count)
        throw std::runtime_error("interface analytic solid storage size mismatch");
    if ((resident_fluid_grad_velocity == nullptr) !=
            (resident_fluid_grad_velocity_count == 0) ||
        (resident_fluid_grad_velocity != nullptr &&
            resident_fluid_grad_velocity_count != fluid_count))
        throw std::runtime_error(
            "invalid resident interface fluid gradient input");
    if (!host_fluid_affine.empty() &&
        (resident_fluid_grad_velocity != nullptr ||
            host_fluid_affine.size() != fluid_count))
        throw std::runtime_error(
            "invalid host interface fluid affine input");

    // Keep solid particle order deterministic: copy_if over a counting range
    // emits ascending particle ids, exactly matching the previous host scan.
    resizeInterfaceQuad(interface_storage.active_solid_ids, solid_count,
        interface_storage);
    const auto ids_begin = thrust::make_counting_iterator<int>(0);
    const auto active_end = thrust::copy_if(ids_begin, ids_begin + solid_count,
        interface_storage.solid_mass.begin(),
        interface_storage.active_solid_ids.begin(), IsPositive{});
    const std::size_t active_count = static_cast<std::size_t>(
        active_end - interface_storage.active_solid_ids.begin());
    interface_storage.active_solid_ids.resize(active_count);

    const std::size_t input_count = fluid_count + active_count;
    resizeFluid(fluid_storage.input_x, input_count, fluid_storage);
    resizeFluid(fluid_storage.input_mass, input_count, fluid_storage);
    resizeFluid(fluid_storage.input_velocity, input_count, fluid_storage);
    resizeFluid(fluid_storage.interface_x, active_count, fluid_storage);
    thrust::copy(interface_storage.fluid_positions.begin(),
        interface_storage.fluid_positions.end(), fluid_storage.input_x.begin());
    thrust::copy(interface_storage.adjusted_fluid_mass.begin(),
        interface_storage.adjusted_fluid_mass.end(), fluid_storage.input_mass.begin());
    thrust::copy(interface_storage.fluid_velocity.begin(),
        interface_storage.fluid_velocity.end(),
        fluid_storage.input_velocity.begin());
    if (active_count) {
        thrust::gather(interface_storage.active_solid_ids.begin(),
            interface_storage.active_solid_ids.end(), solid_storage.positions.begin(),
            fluid_storage.input_x.begin() + fluid_count);
        thrust::gather(interface_storage.active_solid_ids.begin(),
            interface_storage.active_solid_ids.end(), interface_storage.solid_mass.begin(),
            fluid_storage.input_mass.begin() + fluid_count);
        thrust::copy(fluid_storage.input_x.begin() + fluid_count,
            fluid_storage.input_x.end(), fluid_storage.interface_x.begin());
        constexpr int block_size = 256;
        const int blocks = static_cast<int>(
            (active_count + block_size - 1) / block_size);
        gatherInterfaceQuadSolidVelocityKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(interface_storage.solid_momentum.data()),
            thrust::raw_pointer_cast(interface_storage.solid_mass.data()),
            thrust::raw_pointer_cast(
                interface_storage.active_solid_ids.data()),
            static_cast<int>(active_count),
            thrust::raw_pointer_cast(fluid_storage.input_velocity.data())
                + fluid_count);
        checkCuda(cudaGetLastError(),
            "gather interface quadrature solid velocity launch");
    }
    fluid_storage.input_velocity_resident = true;
    fluid_storage.input_affine_resident = false;
    if (resident_fluid_grad_velocity != nullptr) {
        resizeFluid(fluid_storage.input_affine, input_count, fluid_storage);
        thrust::fill(fluid_storage.input_affine.begin(),
            fluid_storage.input_affine.end(), Mat3d{});
        if (fluid_count) {
            constexpr int block_size = 256;
            const int blocks = static_cast<int>(
                (fluid_count + block_size - 1) / block_size);
            buildInterfaceQuadFluidAffineKernel<<<blocks, block_size>>>(
                static_cast<const Mat3d*>(resident_fluid_grad_velocity),
                static_cast<int>(fluid_count), apic_rpic_ratio,
                thrust::raw_pointer_cast(fluid_storage.input_affine.data()));
            checkCuda(cudaGetLastError(),
                "build resident interface fluid affine launch");
        }
        fluid_storage.input_affine_resident = true;
    } else if (!host_fluid_affine.empty()) {
        resizeFluid(fluid_storage.input_affine, input_count, fluid_storage);
        thrust::fill(fluid_storage.input_affine.begin(),
            fluid_storage.input_affine.end(), Mat3d{});
        thrust::copy(host_fluid_affine.begin(), host_fluid_affine.end(),
            fluid_storage.input_affine.begin());
        fluid_storage.input_affine_resident = true;
    }
    rebuildFluidAnalyticQuadratureDeviceInput(fluid_storage, input_count,
        active_count, dx, rho, collision, result, stats,
        materialize_host_result);
}

void rebuildFluidVelocityGrid(PersistentFluidAnalytic3D& context,
    const std::vector<Vec3d>& input_xq,
    const std::vector<double>& input_mass,
    const std::vector<Vec3d>& input_velocity,
    const std::vector<Mat3d>& input_affine,
    const std::vector<Vec3d>& final_xq,
    const std::vector<double>& final_mass, double dx, double dt,
    const Vec3d& gravity, std::vector<VelocityGridNode>& nodes,
    std::vector<VelocityGridNode>& g2p_fallback_nodes,
    FluidVelocityGridStats* stats,
    bool resident_analytic_position_mass,
    bool resident_analytic_velocity,
    bool resident_analytic_affine,
    bool materialize_host_nodes)
{
    auto& storage = getFluidStorage(context);
    const std::size_t input_count = resident_analytic_position_mass
        ? storage.input_x.size() : input_xq.size();
    const std::size_t final_count = resident_analytic_position_mass
        ? storage.output_x.size() : final_xq.size();
    const bool resident_sizes_valid =
        storage.input_x.size() == storage.input_mass.size() &&
        storage.output_x.size() == storage.output_mass.size();
    if ((!resident_analytic_position_mass &&
            (input_xq.size() != input_mass.size() ||
                final_xq.size() != final_mass.size())) ||
        (resident_analytic_position_mass && !resident_sizes_valid) ||
        (!resident_analytic_velocity &&
            input_count != input_velocity.size()) ||
        (resident_analytic_velocity &&
            (!storage.input_velocity_resident ||
                storage.input_velocity.size() != input_count)) ||
        (!resident_analytic_affine && input_count != input_affine.size()) ||
        (resident_analytic_affine &&
            (!storage.input_affine_resident ||
                storage.input_affine.size() != input_count)) ||
        dx <= 0.0 || dt <= 0.0)
        throw std::runtime_error("fluid velocity grid input size mismatch");
    const auto t0 = std::chrono::high_resolution_clock::now();
    const int block_size = 256;

    if (!resident_analytic_position_mass) {
        resizeFluid(storage.input_x, input_count, storage);
        resizeFluid(storage.input_mass, input_count, storage);
        thrust::copy(input_xq.begin(), input_xq.end(), storage.input_x.begin());
        thrust::copy(input_mass.begin(), input_mass.end(),
            storage.input_mass.begin());
    }
    if (!resident_analytic_velocity) {
        resizeFluid(storage.input_velocity, input_velocity.size(), storage);
        thrust::copy(input_velocity.begin(), input_velocity.end(),
            storage.input_velocity.begin());
        storage.input_velocity_resident = false;
    }
    if (!resident_analytic_affine) {
        resizeFluid(storage.input_affine, input_affine.size(), storage);
        thrust::copy(input_affine.begin(), input_affine.end(),
            storage.input_affine.begin());
        storage.input_affine_resident = false;
    }

    resizeFluid(storage.velocity_candidate_keys, input_count * 8, storage);
    resizeFluid(storage.velocity_contribution_mass, input_count * 8, storage);
    resizeFluid(storage.velocity_contribution_momentum, input_count * 8,
        storage);
    if (input_count) {
        const int blocks =
            static_cast<int>((input_count + block_size - 1) / block_size);
        generateFluidMomentumContributionsKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.input_x.data()),
            thrust::raw_pointer_cast(storage.input_mass.data()),
            thrust::raw_pointer_cast(storage.input_velocity.data()),
            thrust::raw_pointer_cast(storage.input_affine.data()),
            static_cast<int>(input_count), dx,
            thrust::raw_pointer_cast(storage.velocity_candidate_keys.data()),
            thrust::raw_pointer_cast(storage.velocity_contribution_mass.data()),
            thrust::raw_pointer_cast(
                storage.velocity_contribution_momentum.data()));
        checkCuda(cudaGetLastError(),
            "generateFluidMomentumContributionsKernel launch");
    }
    auto contribution_zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
        storage.velocity_candidate_keys.begin(),
        storage.velocity_contribution_mass.begin(),
        storage.velocity_contribution_momentum.begin()));
    const auto contribution_zip_end = contribution_zip_begin +
        storage.velocity_candidate_keys.size();
    const auto valid_contribution_zip_end = thrust::remove_if(
        contribution_zip_begin, contribution_zip_end,
        HasZeroMomentumContributionMass{});
    const std::size_t valid_contribution_count = static_cast<std::size_t>(
        valid_contribution_zip_end - contribution_zip_begin);
    storage.velocity_candidate_keys.resize(valid_contribution_count);
    storage.velocity_contribution_mass.resize(valid_contribution_count);
    storage.velocity_contribution_momentum.resize(valid_contribution_count);
    auto contribution_values = thrust::make_zip_iterator(thrust::make_tuple(
        storage.velocity_contribution_mass.begin(),
        storage.velocity_contribution_momentum.begin()));
    thrust::stable_sort_by_key(storage.velocity_candidate_keys.begin(),
        storage.velocity_candidate_keys.end(), contribution_values);
    resizeFluid(storage.velocity_keys,
        storage.velocity_candidate_keys.size(), storage);
    thrust::copy(storage.velocity_candidate_keys.begin(),
        storage.velocity_candidate_keys.end(), storage.velocity_keys.begin());
    const auto velocity_key_end = thrust::unique(
        storage.velocity_keys.begin(), storage.velocity_keys.end());
    const std::size_t node_count = static_cast<std::size_t>(
        velocity_key_end - storage.velocity_keys.begin());
    storage.velocity_keys.resize(node_count);
    resizeFluid(storage.velocity_mass, node_count, storage);
    resizeFluid(storage.velocity_momentum, node_count, storage);
    resizeFluid(storage.velocity_value, node_count, storage);
    if (node_count) {
        const int blocks =
            static_cast<int>((node_count + block_size - 1) / block_size);
        reduceFluidMomentumContributionsKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.velocity_candidate_keys.data()),
            thrust::raw_pointer_cast(storage.velocity_contribution_mass.data()),
            thrust::raw_pointer_cast(
                storage.velocity_contribution_momentum.data()),
            static_cast<int>(storage.velocity_candidate_keys.size()),
            thrust::raw_pointer_cast(storage.velocity_keys.data()),
            static_cast<int>(node_count),
            thrust::raw_pointer_cast(storage.velocity_mass.data()),
            thrust::raw_pointer_cast(storage.velocity_momentum.data()));
        checkCuda(cudaGetLastError(),
            "reduceFluidMomentumContributionsKernel launch");
    }
    if (node_count) {
        const int blocks =
            static_cast<int>((node_count + block_size - 1) / block_size);
        normalizeFluidVelocityKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.velocity_mass.data()),
            thrust::raw_pointer_cast(storage.velocity_momentum.data()),
            static_cast<int>(node_count), dt, gravity,
            thrust::raw_pointer_cast(storage.velocity_value.data()));
        checkCuda(cudaGetLastError(), "normalizeFluidVelocityKernel launch");
    }

    const Vec3d* final_x_device = nullptr;
    const double* final_mass_device = nullptr;
    if (resident_analytic_position_mass) {
        final_x_device = final_count == 0 ? nullptr
            : thrust::raw_pointer_cast(storage.output_x.data());
        final_mass_device = final_count == 0 ? nullptr
            : thrust::raw_pointer_cast(storage.output_mass.data());
    } else {
        resizeFluid(storage.pruned_x, final_count, storage);
        resizeFluid(storage.pruned_mass, final_count, storage);
        thrust::copy(final_xq.begin(), final_xq.end(), storage.pruned_x.begin());
        thrust::copy(final_mass.begin(), final_mass.end(),
            storage.pruned_mass.begin());
        final_x_device = final_count == 0 ? nullptr
            : thrust::raw_pointer_cast(storage.pruned_x.data());
        final_mass_device = final_count == 0 ? nullptr
            : thrust::raw_pointer_cast(storage.pruned_mass.data());
    }
    resizeFluid(storage.velocity_candidate_keys, final_count * 8, storage);
    resizeFluid(storage.velocity_contribution_mass, final_count * 8, storage);
    if (final_count) {
        const int blocks =
            static_cast<int>((final_count + block_size - 1) / block_size);
        generateFluidMassContributionsKernel<<<blocks, block_size>>>(
            final_x_device, final_mass_device, static_cast<int>(final_count), dx,
            thrust::raw_pointer_cast(storage.velocity_candidate_keys.data()),
            thrust::raw_pointer_cast(storage.velocity_contribution_mass.data()));
        checkCuda(cudaGetLastError(),
            "generateFluidMassContributionsKernel launch");
    }
    auto final_contribution_zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
        storage.velocity_candidate_keys.begin(),
        storage.velocity_contribution_mass.begin()));
    const auto final_contribution_zip_end = final_contribution_zip_begin +
        storage.velocity_candidate_keys.size();
    const auto valid_final_contribution_zip_end = thrust::remove_if(
        final_contribution_zip_begin, final_contribution_zip_end,
        HasZeroMomentumContributionMass{});
    const std::size_t valid_final_contribution_count = static_cast<std::size_t>(
        valid_final_contribution_zip_end - final_contribution_zip_begin);
    storage.velocity_candidate_keys.resize(valid_final_contribution_count);
    storage.velocity_contribution_mass.resize(valid_final_contribution_count);
    thrust::stable_sort_by_key(storage.velocity_candidate_keys.begin(),
        storage.velocity_candidate_keys.end(),
        storage.velocity_contribution_mass.begin());
    resizeFluid(storage.velocity_final_keys,
        storage.velocity_candidate_keys.size(), storage);
    thrust::copy(storage.velocity_candidate_keys.begin(),
        storage.velocity_candidate_keys.end(), storage.velocity_final_keys.begin());
    const auto final_velocity_key_end = thrust::unique(
        storage.velocity_final_keys.begin(), storage.velocity_final_keys.end());
    const std::size_t final_node_count = static_cast<std::size_t>(
        final_velocity_key_end - storage.velocity_final_keys.begin());
    storage.velocity_final_keys.resize(final_node_count);
    resizeFluid(storage.velocity_mass, final_node_count, storage);
    resizeFluid(storage.velocity_final_value, final_node_count, storage);
    if (final_node_count) {
        const int blocks =
            static_cast<int>((final_node_count + block_size - 1) / block_size);
        reduceFluidMassContributionsKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.velocity_candidate_keys.data()),
            thrust::raw_pointer_cast(storage.velocity_contribution_mass.data()),
            static_cast<int>(storage.velocity_candidate_keys.size()),
            thrust::raw_pointer_cast(storage.velocity_final_keys.data()),
            static_cast<int>(final_node_count),
            thrust::raw_pointer_cast(storage.velocity_mass.data()));
        checkCuda(cudaGetLastError(),
            "reduceFluidMassContributionsKernel launch");
        gatherInitialVelocityAtKeysKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.velocity_keys.data()),
            thrust::raw_pointer_cast(storage.velocity_value.data()),
            static_cast<int>(node_count),
            thrust::raw_pointer_cast(storage.velocity_final_keys.data()),
            static_cast<int>(final_node_count),
            thrust::raw_pointer_cast(storage.velocity_final_value.data()));
        checkCuda(cudaGetLastError(),
            "gatherInitialVelocityAtKeysKernel final launch");
    }

    // CPU `rebuild_grid3_mass_from_quadrature()` resets m and idx for every
    // existing grid entry, then only assigns a new idx to final-mass nodes.
    // It intentionally does *not* clear new_v.  Preserve precisely the
    // initial-only subset for G2P, while keeping it out of all system DOFs.
    resizeFluid(storage.velocity_g2p_fallback_keys, node_count, storage);
    const auto fallback_key_end = thrust::set_difference(
        storage.velocity_keys.begin(), storage.velocity_keys.end(),
        storage.velocity_final_keys.begin(), storage.velocity_final_keys.end(),
        storage.velocity_g2p_fallback_keys.begin());
    const std::size_t fallback_node_count = static_cast<std::size_t>(
        fallback_key_end - storage.velocity_g2p_fallback_keys.begin());
    storage.velocity_g2p_fallback_keys.resize(fallback_node_count);
    resizeFluid(storage.velocity_g2p_fallback_value, fallback_node_count,
        storage);
    if (fallback_node_count) {
        const int blocks = static_cast<int>((fallback_node_count + block_size - 1) /
            block_size);
        gatherInitialVelocityAtKeysKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.velocity_keys.data()),
            thrust::raw_pointer_cast(storage.velocity_value.data()),
            static_cast<int>(node_count),
            thrust::raw_pointer_cast(storage.velocity_g2p_fallback_keys.data()),
            static_cast<int>(fallback_node_count),
            thrust::raw_pointer_cast(storage.velocity_g2p_fallback_value.data()));
        checkCuda(cudaGetLastError(),
            "gatherInitialVelocityAtKeysKernel fallback launch");
    }
    checkCuda(cudaDeviceSynchronize(), "fluid velocity grid synchronize");

    nodes.clear();
    g2p_fallback_nodes.clear();
    if (materialize_host_nodes) {
        std::vector<GridKey> host_keys(final_node_count);
        std::vector<double> host_mass(final_node_count);
        std::vector<Vec3d> host_velocity(final_node_count);
        thrust::copy(storage.velocity_final_keys.begin(),
            storage.velocity_final_keys.end(), host_keys.begin());
        thrust::copy(storage.velocity_mass.begin(), storage.velocity_mass.end(),
            host_mass.begin());
        thrust::copy(storage.velocity_final_value.begin(),
            storage.velocity_final_value.end(), host_velocity.begin());
        nodes.resize(final_node_count);
        for (std::size_t i = 0; i < final_node_count; ++i)
            nodes[i] = VelocityGridNode{
                host_keys[i], host_mass[i], host_velocity[i]};
        std::vector<GridKey> host_fallback_keys(fallback_node_count);
        std::vector<Vec3d> host_fallback_velocity(fallback_node_count);
        thrust::copy(storage.velocity_g2p_fallback_keys.begin(),
            storage.velocity_g2p_fallback_keys.end(), host_fallback_keys.begin());
        thrust::copy(storage.velocity_g2p_fallback_value.begin(),
            storage.velocity_g2p_fallback_value.end(),
            host_fallback_velocity.begin());
        g2p_fallback_nodes.resize(fallback_node_count);
        for (std::size_t i = 0; i < fallback_node_count; ++i)
            g2p_fallback_nodes[i] = VelocityGridNode{
                host_fallback_keys[i], 0.0, host_fallback_velocity[i]};
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->input_samples = input_count;
        stats->final_samples = final_count;
        stats->nodes = final_node_count;
        stats->g2p_fallback_nodes = fallback_node_count;
        stats->reallocations = storage.reallocations;
        stats->resident_position_mass = resident_analytic_position_mass;
        stats->resident_velocity = resident_analytic_velocity;
        stats->resident_affine = resident_analytic_affine;
        stats->total_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
}

void buildFluidLegacyVelocityNodeMap3D(PersistentFluidAnalytic3D& context,
    GPU_STAGE3A::DeviceGridNodeMap& velocity_nodes, double* order_ms)
{
    if (!context.storage)
        throw std::runtime_error("fluid velocity owner is empty");
    auto& storage = getFluidStorage(context);
    const std::size_t node_count = storage.velocity_final_keys.size();
    if (storage.velocity_mass.size() != node_count ||
        storage.velocity_final_value.size() != node_count)
        throw std::runtime_error("fluid resident velocity field size mismatch");
    const auto begin = std::chrono::high_resolution_clock::now();
    resizeFluid(storage.dof_node_order, node_count, storage);
    resizeFluid(storage.dof_node_ids, node_count, storage);
    resizeFluid(storage.dof_nodes, node_count, storage);
    constexpr int threads = 256;
    if (node_count != 0) {
        const int blocks = static_cast<int>(
            (node_count + threads - 1) / threads);
        buildFluidLegacyVelocityNodesKernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(storage.velocity_final_keys.data()),
            thrust::raw_pointer_cast(storage.velocity_mass.data()),
            thrust::raw_pointer_cast(storage.velocity_final_value.data()),
            static_cast<int>(node_count),
            thrust::raw_pointer_cast(storage.dof_node_order.data()),
            thrust::raw_pointer_cast(storage.dof_nodes.data()));
        checkCuda(cudaGetLastError(),
            "fluid legacy velocity-node order launch");
    }
    thrust::sequence(storage.dof_node_ids.begin(), storage.dof_node_ids.end());
    thrust::sort_by_key(storage.dof_node_order.begin(),
        storage.dof_node_order.end(), storage.dof_node_ids.begin());
    if (node_count != 0) {
        const int blocks = static_cast<int>(
            (node_count + threads - 1) / threads);
        scatterFluidLegacyNodeRanksKernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(storage.dof_node_ids.data()),
            static_cast<int>(node_count),
            thrust::raw_pointer_cast(storage.dof_nodes.data()));
        checkCuda(cudaGetLastError(),
            "fluid legacy velocity-node rank scatter launch");
    }
    checkCuda(cudaDeviceSynchronize(),
        "fluid legacy velocity-node map synchronize");
    velocity_nodes = GPU_STAGE3A::makeDeviceGridNodeMapFromDeviceNodes(
        node_count == 0 ? nullptr
                        : thrust::raw_pointer_cast(storage.dof_nodes.data()),
        node_count);
    if (order_ms)
        *order_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - begin).count();
}

void rebuildSolidPressureGrid(PersistentFluidAnalytic3D& context,
    const std::vector<Vec3d>& positions, const std::vector<double>& j,
    const std::vector<double>& volume,
    const std::vector<double>& lambda, double dx,
    std::vector<SolidPressureGridNode>& nodes,
    FluidVelocityGridStats* stats,
    const GPU_STAGE3A::SolidParticleStateDeviceView* resident_particles,
    GPU_STAGE3A::DeviceGridNodeMap* device_nodes,
    bool materialize_host_nodes)
{
    const bool resident = resident_particles && resident_particles->size != 0;
    const std::size_t input_count = resident
        ? resident_particles->size : positions.size();
    if ((!resident && (positions.size() != j.size() ||
            positions.size() != volume.size() ||
            positions.size() != lambda.size())) ||
        (resident && (!resident_particles->rest_x ||
            !resident_particles->j || !resident_particles->volume ||
            !resident_particles->lambda)) || dx <= 0.0)
        throw std::runtime_error("solid pressure grid input size mismatch");
    auto& storage = getFluidStorage(context);
    const auto t0 = std::chrono::high_resolution_clock::now();
    const int block_size = 256;
    const Vec3d* device_positions = nullptr;
    const double* device_j = nullptr;
    const double* device_volume = nullptr;
    const double* device_lambda = nullptr;
    if (resident) {
        static_assert(sizeof(Vec3d) == sizeof(GPU_STAGE3A::Vec3d),
            "solid resident position layout mismatch");
        device_positions = reinterpret_cast<const Vec3d*>(
            resident_particles->rest_x);
        device_j = resident_particles->j;
        device_volume = resident_particles->volume;
        device_lambda = resident_particles->lambda;
    } else {
        resizeFluid(storage.input_x, input_count, storage);
        resizeFluid(storage.solid_j, input_count, storage);
        resizeFluid(storage.solid_volume, input_count, storage);
        resizeFluid(storage.solid_lambda, input_count, storage);
        thrust::copy(positions.begin(), positions.end(), storage.input_x.begin());
        thrust::copy(j.begin(), j.end(), storage.solid_j.begin());
        thrust::copy(volume.begin(), volume.end(), storage.solid_volume.begin());
        thrust::copy(lambda.begin(), lambda.end(), storage.solid_lambda.begin());
        device_positions = input_count
            ? thrust::raw_pointer_cast(storage.input_x.data()) : nullptr;
        device_j = input_count
            ? thrust::raw_pointer_cast(storage.solid_j.data()) : nullptr;
        device_volume = input_count
            ? thrust::raw_pointer_cast(storage.solid_volume.data()) : nullptr;
        device_lambda = input_count
            ? thrust::raw_pointer_cast(storage.solid_lambda.data()) : nullptr;
    }
    resizeFluid(storage.input_keys, input_count, storage);
    if (input_count != 0) {
        const int blocks =
            static_cast<int>((input_count + block_size - 1) / block_size);
        generatePressureKeysKernel<<<blocks, block_size>>>(
            device_positions, static_cast<int>(input_count), dx, 1,
            thrust::raw_pointer_cast(storage.input_keys.data()));
        checkCuda(cudaGetLastError(), "solid pressure key launch");
    }
    resizeFluid(storage.active_keys, storage.input_keys.size(), storage);
    thrust::copy(storage.input_keys.begin(), storage.input_keys.end(),
        storage.active_keys.begin());
    thrust::sort(storage.active_keys.begin(), storage.active_keys.end());
    const auto key_end = thrust::unique(
        storage.active_keys.begin(), storage.active_keys.end());
    const std::size_t node_count = static_cast<std::size_t>(
        key_end - storage.active_keys.begin());
    storage.active_keys.resize(node_count);
    resizeFluid(storage.velocity_mass, node_count, storage);
    resizeFluid(storage.solid_compliance, node_count, storage);
    resizeFluid(storage.solid_pressure_numerator, node_count, storage);
    resizeFluid(storage.output_mass, node_count, storage);
    thrust::fill(storage.velocity_mass.begin(), storage.velocity_mass.end(), 0.0);
    thrust::fill(storage.solid_compliance.begin(),
        storage.solid_compliance.end(), 0.0);
    thrust::fill(storage.solid_pressure_numerator.begin(),
        storage.solid_pressure_numerator.end(), 0.0);
    if (input_count != 0) {
        const int blocks =
            static_cast<int>((input_count + block_size - 1) / block_size);
        scatterSolidPressureKernel<<<blocks, block_size>>>(
            device_positions, device_j, device_volume, device_lambda,
            static_cast<int>(input_count), dx,
            thrust::raw_pointer_cast(storage.active_keys.data()),
            static_cast<int>(node_count),
            thrust::raw_pointer_cast(storage.velocity_mass.data()),
            thrust::raw_pointer_cast(storage.solid_compliance.data()),
            thrust::raw_pointer_cast(storage.solid_pressure_numerator.data()));
        checkCuda(cudaGetLastError(), "scatterSolidPressureKernel launch");
    }
    if (node_count) {
        const int blocks =
            static_cast<int>((node_count + block_size - 1) / block_size);
        normalizeSolidPressureKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.velocity_mass.data()),
            thrust::raw_pointer_cast(storage.solid_pressure_numerator.data()),
            static_cast<int>(node_count),
            thrust::raw_pointer_cast(storage.output_mass.data()));
        checkCuda(cudaGetLastError(), "normalizeSolidPressureKernel launch");
    }
    checkCuda(cudaDeviceSynchronize(), "solid pressure grid synchronize");
    if (device_nodes) {
        resizeFluid(storage.dof_node_order, node_count, storage);
        resizeFluid(storage.dof_node_ids, node_count, storage);
        resizeFluid(storage.dof_nodes, node_count, storage);
        if (node_count != 0) {
            const int blocks = static_cast<int>(
                (node_count + block_size - 1) / block_size);
            buildSolidLegacyPressureNodesKernel<<<blocks, block_size>>>(
                thrust::raw_pointer_cast(storage.active_keys.data()),
                thrust::raw_pointer_cast(storage.velocity_mass.data()),
                thrust::raw_pointer_cast(storage.solid_compliance.data()),
                thrust::raw_pointer_cast(storage.output_mass.data()),
                static_cast<int>(node_count),
                thrust::raw_pointer_cast(storage.dof_node_order.data()),
                thrust::raw_pointer_cast(storage.dof_nodes.data()));
            checkCuda(cudaGetLastError(),
                "solid legacy pressure-node order launch");
        }
        thrust::sequence(storage.dof_node_ids.begin(),
            storage.dof_node_ids.end());
        thrust::sort_by_key(storage.dof_node_order.begin(),
            storage.dof_node_order.end(), storage.dof_node_ids.begin());
        if (node_count != 0) {
            const int blocks = static_cast<int>(
                (node_count + block_size - 1) / block_size);
            scatterFluidLegacyNodeRanksKernel<<<blocks, block_size>>>(
                thrust::raw_pointer_cast(storage.dof_node_ids.data()),
                static_cast<int>(node_count),
                thrust::raw_pointer_cast(storage.dof_nodes.data()));
            checkCuda(cudaGetLastError(),
                "solid legacy pressure-node rank scatter launch");
        }
        checkCuda(cudaDeviceSynchronize(),
            "solid legacy pressure-node map synchronize");
        *device_nodes = GPU_STAGE3A::makeDeviceGridNodeMapFromDeviceNodes(
            node_count == 0 ? nullptr
                            : thrust::raw_pointer_cast(storage.dof_nodes.data()),
            node_count);
    }
    nodes.clear();
    if (materialize_host_nodes) {
        std::vector<GridKey> host_keys(node_count);
        std::vector<double> host_mass(node_count);
        std::vector<double> host_compliance(node_count);
        std::vector<double> host_pressure(node_count);
        thrust::copy(storage.active_keys.begin(), storage.active_keys.end(),
            host_keys.begin());
        thrust::copy(storage.velocity_mass.begin(), storage.velocity_mass.end(),
            host_mass.begin());
        thrust::copy(storage.solid_compliance.begin(),
            storage.solid_compliance.end(), host_compliance.begin());
        thrust::copy(storage.output_mass.begin(), storage.output_mass.end(),
            host_pressure.begin());
        nodes.resize(node_count);
        for (std::size_t i = 0; i < node_count; ++i)
            nodes[i] = SolidPressureGridNode{
                host_keys[i], host_mass[i], host_compliance[i], host_pressure[i]};
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->input_samples = input_count;
        stats->final_samples = input_count;
        stats->nodes = node_count;
        stats->reallocations = storage.reallocations;
        stats->resident_position_mass = resident;
        stats->total_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
}

} // namespace ZIRAN::GPU_STAGE2
