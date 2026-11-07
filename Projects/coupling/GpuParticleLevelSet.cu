#include "GpuParticleLevelSet.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace ZIRAN::GPU_POSITION_CORRECTOR {
namespace {

using Clock = std::chrono::steady_clock;

void checkCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": "
            + cudaGetErrorString(error));
}

double elapsedMs(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct ParticleLevelSetStorage {
    thrust::device_vector<Vec3d> positions;
    thrust::device_vector<int> bounds;
    thrust::device_vector<double> raw_phi;
    thrust::device_vector<double> distance;
    thrust::device_vector<double> final_phi;
    thrust::device_vector<std::uint8_t> touched;
    thrust::device_vector<std::uint8_t> frozen;
    int origin[3] = { 0, 0, 0 };
    int shape[3] = { 0, 0, 0 };
    double h = 0;
    std::size_t reallocations = 0;
};

__global__ void initializeParticleBoundsKernel(int* bounds)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    bounds[0] = bounds[1] = bounds[2] = INT_MAX;
    bounds[3] = bounds[4] = bounds[5] = INT_MIN;
}

__global__ void reduceParticleBoundsKernel(const Vec3d* positions,
    int particle_count, double h, int* bounds)
{
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= particle_count)
        return;
    const Vec3d x = positions[particle];
    const int base[3] = {
        static_cast<int>(floor(x.x / h)),
        static_cast<int>(floor(x.y / h)),
        static_cast<int>(floor(x.z / h))};
    for (int d = 0; d < 3; ++d) {
        atomicMin(bounds + d, base[d]);
        atomicMax(bounds + 3 + d, base[d]);
    }
}

ParticleLevelSetStorage& getStorage(PersistentParticleLevelSet3D& context)
{
    if (!context.storage)
        context.storage = std::make_shared<ParticleLevelSetStorage>();
    return *std::static_pointer_cast<ParticleLevelSetStorage>(context.storage);
}

template <class T>
void resizeTracked(thrust::device_vector<T>& values, std::size_t size,
    ParticleLevelSetStorage& storage)
{
    const std::size_t old_capacity = values.capacity();
    values.resize(size);
    if (values.capacity() != old_capacity)
        ++storage.reallocations;
}

__device__ double atomicMaxDouble(double* address, double value)
{
    auto* integer_address = reinterpret_cast<unsigned long long*>(address);
    unsigned long long observed = *integer_address;
    while (value > __longlong_as_double(observed)) {
        const unsigned long long assumed = observed;
        observed = atomicCAS(integer_address, assumed, __double_as_longlong(value));
        if (observed == assumed)
            break;
    }
    return __longlong_as_double(observed);
}

__global__ void initializeRasterKernel(double* raw_phi,
    std::uint8_t* touched, std::size_t voxel_count)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index >= voxel_count)
        return;
    raw_phi[index] = -DBL_MAX;
    touched[index] = 0;
}

__global__ void rasterizeParticlesKernel(const Vec3d* positions,
    int particle_count, double h, double blob_radius, int kernel_size,
    int low_offset, int origin_x, int origin_y, int origin_z,
    int shape_x, int shape_y, int shape_z, double* raw_phi)
{
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= particle_count)
        return;

    const Vec3d x = positions[particle];
    const int base_x = static_cast<int>(floor(x.x / h));
    const int base_y = static_cast<int>(floor(x.y / h));
    const int base_z = static_cast<int>(floor(x.z / h));
    for (int i = 0; i < kernel_size; ++i) {
        const int gx = base_x + i - low_offset;
        const double dx = x.x - static_cast<double>(gx) * h;
        const int lx = gx - origin_x;
        for (int j = 0; j < kernel_size; ++j) {
            const int gy = base_y + j - low_offset;
            const double dy = x.y - static_cast<double>(gy) * h;
            const int ly = gy - origin_y;
            for (int k = 0; k < kernel_size; ++k) {
                const int gz = base_z + k - low_offset;
                const double dz = x.z - static_cast<double>(gz) * h;
                const int lz = gz - origin_z;
                if (lx < 0 || lx >= shape_x || ly < 0 || ly >= shape_y
                    || lz < 0 || lz >= shape_z)
                    continue;
                const std::size_t index = (static_cast<std::size_t>(lz) * shape_y
                                               + static_cast<std::size_t>(ly))
                        * shape_x
                    + static_cast<std::size_t>(lx);
                const double value = blob_radius - sqrt(dx * dx + dy * dy + dz * dz);
                atomicMaxDouble(raw_phi + index, value);
            }
        }
    }
}

__global__ void initializeDistanceKernel(const double* raw_phi,
    std::uint8_t* touched, std::uint8_t* frozen, double* distance,
    std::size_t voxel_count)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index >= voxel_count)
        return;
    const bool is_touched = raw_phi[index] > -DBL_MAX / 2.0;
    touched[index] = is_touched ? 1 : 0;
    if (!is_touched) {
        frozen[index] = 0;
        distance[index] = DBL_MAX;
        return;
    }
    const bool is_frozen = raw_phi[index] <= 0.0;
    frozen[index] = is_frozen ? 1 : 0;
    distance[index] = is_frozen
        ? raw_phi[index]
        : DBL_MAX;
}

__device__ double solveEikonal(double a, double b, double c, double h)
{
    if (a > b) {
        const double tmp = a;
        a = b;
        b = tmp;
    }
    if (b > c) {
        const double tmp = b;
        b = c;
        c = tmp;
    }
    if (a > b) {
        const double tmp = a;
        a = b;
        b = tmp;
    }
    if (a > DBL_MAX / 2.0)
        return DBL_MAX;

    double result = a + h;
    if (a + h > b) {
        const double ab = ((a + b)
                              + sqrt(2.0 * h * h - (a - b) * (a - b)))
            / 2.0;
        result = fmin(result, ab);
        if (ab > c) {
            const double sum = -2.0 * a - 2.0 * b - 2.0 * c;
            const double abc = (sqrt(sum * sum
                                    - 12.0 * (a * a + b * b + c * c - h * h))
                                   + 2.0 * a + 2.0 * b + 2.0 * c)
                / 6.0;
            result = fmin(result, abc);
        }
    }
    return result;
}

// Nodes on one i+j+k plane have no six-neighbour edges between them, so this
// is an in-place Gauss-Seidel update without a data race. Kernels in the same
// stream serialize successive planes and preserve the sweep direction.
__global__ void sweepPlaneKernel(double* distance,
    const std::uint8_t* touched, const std::uint8_t* frozen,
    int shape_x, int shape_y, int shape_z, int level,
    int reverse_x, int reverse_y, int reverse_z, double h)
{
    const int interior_x = shape_x - 2;
    const int interior_y = shape_y - 2;
    const int interior_z = shape_z - 2;
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = level - u - v;
    if (u < 0 || u >= interior_x || v < 0 || v >= interior_y
        || w < 0 || w >= interior_z)
        return;

    const int x = 1 + (reverse_x ? interior_x - 1 - u : u);
    const int y = 1 + (reverse_y ? interior_y - 1 - v : v);
    const int z = 1 + (reverse_z ? interior_z - 1 - w : w);
    const std::size_t index = (static_cast<std::size_t>(z) * shape_y
                                  + static_cast<std::size_t>(y))
            * shape_x
        + static_cast<std::size_t>(x);
    if (!touched[index] || frozen[index])
        return;

    const double x_min = fmin(distance[index - 1], distance[index + 1]);
    const double y_min = fmin(distance[index - shape_x], distance[index + shape_x]);
    const std::size_t z_stride = static_cast<std::size_t>(shape_x) * shape_y;
    const double z_min = fmin(distance[index - z_stride], distance[index + z_stride]);
    const double candidate = solveEikonal(x_min, y_min, z_min, h);
    if (candidate < distance[index])
        distance[index] = candidate;
}

__global__ void finalizeDistanceKernel(const double* distance,
    const std::uint8_t* touched, double erosion, double* final_phi,
    std::size_t voxel_count)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index >= voxel_count)
        return;
    final_phi[index] = touched[index]
            && distance[index] < DBL_MAX / 2.0
        ? -distance[index] + erosion
        : DBL_MAX;
}

} // namespace

bool particleLevelSetCudaAvailable()
{
    return true;
}

bool buildParticleLevelSet3D(PersistentParticleLevelSet3D& context,
    const std::vector<Vec3d>& positions, double h, double blob_radius,
    double erosion, int sweep_cycles, DenseLevelSet3D& result,
    ParticleLevelSetStats* stats, const void* resident_positions,
    std::size_t resident_position_count, bool download_result)
{
    const auto total_begin = Clock::now();
    ParticleLevelSetStats local_stats;
    if ((resident_positions == nullptr) != (resident_position_count == 0))
        throw std::runtime_error("invalid resident particle level-set input");
    const bool use_resident_positions = resident_positions != nullptr;
    const std::size_t particle_count = use_resident_positions
        ? resident_position_count
        : positions.size();
    local_stats.particles = particle_count;
    local_stats.sweep_cycles = sweep_cycles;
    local_stats.resident_particle_input = use_resident_positions;
    if (!(h > 0) || !(blob_radius >= 0) || sweep_cycles < 1)
        throw std::runtime_error("invalid GPU particle level-set parameters");
    if (particle_count == 0) {
        result = DenseLevelSet3D {};
        if (context.storage) {
            ParticleLevelSetStorage& storage = getStorage(context);
            for (int d = 0; d < 3; ++d) {
                storage.origin[d] = 0;
                storage.shape[d] = 0;
            }
            storage.h = h;
            storage.final_phi.clear();
        }
        if (stats)
            *stats = local_stats;
        return true;
    }

    const int kernel_size = (static_cast<int>(std::ceil(blob_radius / h)) + 1) * 2;
    const int low_offset = kernel_size / 2 - 1;
    const int high_offset = kernel_size / 2;
    local_stats.kernel_size = kernel_size;

    ParticleLevelSetStorage& storage = getStorage(context);
    const auto upload_begin = Clock::now();
    const Vec3d* device_positions =
        static_cast<const Vec3d*>(resident_positions);
    if (!use_resident_positions) {
        resizeTracked(storage.positions, particle_count, storage);
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage.positions.data()),
                      positions.data(), particle_count * sizeof(Vec3d),
                      cudaMemcpyHostToDevice),
            "particle level-set position upload");
        device_positions = thrust::raw_pointer_cast(storage.positions.data());
    }
    resizeTracked(storage.bounds, 6, storage);
    constexpr int block_size = 256;
    initializeParticleBoundsKernel<<<1, 1>>>(
        thrust::raw_pointer_cast(storage.bounds.data()));
    const int particle_blocks = static_cast<int>(
        (particle_count + block_size - 1) / block_size);
    reduceParticleBoundsKernel<<<particle_blocks, block_size>>>(
        device_positions, static_cast<int>(particle_count), h,
        thrust::raw_pointer_cast(storage.bounds.data()));
    checkCuda(cudaGetLastError(), "particle level-set bounds launch");
    int bounds[6];
    checkCuda(cudaMemcpy(bounds,
                  thrust::raw_pointer_cast(storage.bounds.data()),
                  sizeof(bounds), cudaMemcpyDeviceToHost),
        "particle level-set bounds download");
    const int* min_base = bounds;
    const int* max_base = bounds + 3;

    // One untouched ghost voxel on every side makes all six-neighbour reads
    // valid, including at the edge of the particle rasterization support.
    for (int d = 0; d < 3; ++d) {
        result.origin[d] = min_base[d] - low_offset - 1;
        const int maximum = max_base[d] + high_offset + 1;
        result.shape[d] = maximum - result.origin[d] + 1;
        storage.origin[d] = result.origin[d];
        storage.shape[d] = result.shape[d];
    }
    storage.h = h;
    const std::size_t voxel_count = static_cast<std::size_t>(result.shape[0])
        * static_cast<std::size_t>(result.shape[1])
        * static_cast<std::size_t>(result.shape[2]);
    local_stats.voxels = voxel_count;

    resizeTracked(storage.raw_phi, voxel_count, storage);
    resizeTracked(storage.distance, voxel_count, storage);
    resizeTracked(storage.final_phi, voxel_count, storage);
    resizeTracked(storage.touched, voxel_count, storage);
    resizeTracked(storage.frozen, voxel_count, storage);
    checkCuda(cudaDeviceSynchronize(), "particle level-set upload synchronize");
    const auto upload_end = Clock::now();

    const int voxel_blocks = static_cast<int>((voxel_count + block_size - 1) / block_size);
    const auto raster_begin = Clock::now();
    initializeRasterKernel<<<voxel_blocks, block_size>>>(
        thrust::raw_pointer_cast(storage.raw_phi.data()),
        thrust::raw_pointer_cast(storage.touched.data()), voxel_count);
    rasterizeParticlesKernel<<<particle_blocks, block_size>>>(
        device_positions, static_cast<int>(particle_count), h, blob_radius, kernel_size,
        low_offset, result.origin[0], result.origin[1], result.origin[2],
        result.shape[0], result.shape[1], result.shape[2],
        thrust::raw_pointer_cast(storage.raw_phi.data()));
    initializeDistanceKernel<<<voxel_blocks, block_size>>>(
        thrust::raw_pointer_cast(storage.raw_phi.data()),
        thrust::raw_pointer_cast(storage.touched.data()),
        thrust::raw_pointer_cast(storage.frozen.data()),
        thrust::raw_pointer_cast(storage.distance.data()), voxel_count);
    checkCuda(cudaGetLastError(), "particle level-set rasterize launch");
    checkCuda(cudaDeviceSynchronize(), "particle level-set rasterize synchronize");
    const auto raster_end = Clock::now();

    const auto sweep_begin = Clock::now();
    const int interior_x = result.shape[0] - 2;
    const int interior_y = result.shape[1] - 2;
    const int interior_z = result.shape[2] - 2;
    const int max_level = interior_x + interior_y + interior_z - 3;
    const dim3 plane_block(16, 16, 1);
    const dim3 plane_grid(
        static_cast<unsigned>((interior_x + plane_block.x - 1) / plane_block.x),
        static_cast<unsigned>((interior_y + plane_block.y - 1) / plane_block.y), 1);
    for (int cycle = 0; cycle < sweep_cycles; ++cycle) {
        for (int direction = 0; direction < 8; ++direction) {
            const int reverse_x = direction & 1;
            const int reverse_y = (direction >> 1) & 1;
            const int reverse_z = (direction >> 2) & 1;
            for (int level = 0; level <= max_level; ++level) {
                sweepPlaneKernel<<<plane_grid, plane_block>>>(
                    thrust::raw_pointer_cast(storage.distance.data()),
                    thrust::raw_pointer_cast(storage.touched.data()),
                    thrust::raw_pointer_cast(storage.frozen.data()),
                    result.shape[0], result.shape[1], result.shape[2], level,
                    reverse_x, reverse_y, reverse_z, h);
            }
        }
    }
    finalizeDistanceKernel<<<voxel_blocks, block_size>>>(
        thrust::raw_pointer_cast(storage.distance.data()),
        thrust::raw_pointer_cast(storage.touched.data()), erosion,
        thrust::raw_pointer_cast(storage.final_phi.data()), voxel_count);
    checkCuda(cudaGetLastError(), "particle level-set sweep launch");
    checkCuda(cudaDeviceSynchronize(), "particle level-set sweep synchronize");
    const auto sweep_end = Clock::now();

    const auto download_begin = Clock::now();
    if (download_result) {
        result.phi.resize(voxel_count);
        checkCuda(cudaMemcpy(result.phi.data(),
                      thrust::raw_pointer_cast(storage.final_phi.data()),
                      voxel_count * sizeof(double), cudaMemcpyDeviceToHost),
            "particle level-set download");
    }
    else {
        result.phi.clear();
    }
    const auto download_end = Clock::now();

    local_stats.capacity_voxels = storage.distance.capacity();
    local_stats.reallocations = storage.reallocations;
    local_stats.upload_ms = elapsedMs(upload_begin, upload_end);
    local_stats.rasterize_ms = elapsedMs(raster_begin, raster_end);
    local_stats.sweep_ms = elapsedMs(sweep_begin, sweep_end);
    local_stats.download_ms = elapsedMs(download_begin, download_end);
    local_stats.total_ms = elapsedMs(total_begin, download_end);
    if (stats)
        *stats = local_stats;
    return true;
}

bool copyParticleLevelSetToHost3D(
    const PersistentParticleLevelSet3D& context, DenseLevelSet3D& result)
{
    if (!context.storage) {
        result = DenseLevelSet3D {};
        return false;
    }
    const auto& storage = *std::static_pointer_cast<ParticleLevelSetStorage>(
        context.storage);
    if (storage.final_phi.empty()) {
        result = DenseLevelSet3D {};
        return false;
    }
    for (int d = 0; d < 3; ++d) {
        result.origin[d] = storage.origin[d];
        result.shape[d] = storage.shape[d];
    }
    result.phi.resize(storage.final_phi.size());
    checkCuda(cudaMemcpy(result.phi.data(),
                  thrust::raw_pointer_cast(storage.final_phi.data()),
                  storage.final_phi.size() * sizeof(double),
                  cudaMemcpyDeviceToHost),
        "particle level-set compatibility download");
    return true;
}

ParticleLevelSetDeviceView3D getParticleLevelSetDeviceView3D(
    const PersistentParticleLevelSet3D& context)
{
    ParticleLevelSetDeviceView3D view;
    if (!context.storage)
        return view;
    const auto& storage = *std::static_pointer_cast<ParticleLevelSetStorage>(
        context.storage);
    if (storage.final_phi.empty())
        return view;
    view.phi = thrust::raw_pointer_cast(storage.final_phi.data());
    for (int d = 0; d < 3; ++d) {
        view.origin[d] = storage.origin[d];
        view.shape[d] = storage.shape[d];
    }
    view.h = storage.h;
    view.voxels = storage.final_phi.size();
    return view;
}

} // namespace ZIRAN::GPU_POSITION_CORRECTOR
