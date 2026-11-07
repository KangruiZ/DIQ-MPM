#include "GpuOperatorProbe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

#include <cuda_runtime.h>
#ifdef ZIRAN_ENABLE_CUSPARSE
#include <cusparse.h>
#endif
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <limits>

namespace ZIRAN::GPU_STAGE3A {
namespace {

constexpr int kCoordBits = 21;
constexpr int kCoordBias = 1 << (kCoordBits - 1);
constexpr GridKey kCoordMask = (GridKey{1} << kCoordBits) - 1;

struct CooEntry {
    int row = -1;
    int col = -1;
    double value = 0.0;
};

void checkCuda(cudaError_t error, const char* label)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(error));
}

template <class T>
void cudaMallocArray(T*& ptr, std::size_t count, const char* label)
{
    ptr = nullptr;
    if (count == 0)
        return;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), label);
}

template <class T>
void cudaCopyToDevice(T* dst, const std::vector<T>& src, const char* label)
{
    if (src.empty())
        return;
    checkCuda(cudaMemcpy(dst, src.data(), src.size() * sizeof(T),
                  cudaMemcpyHostToDevice),
        label);
}

template <class T>
void cudaCopyToHost(std::vector<T>& dst, const T* src, const char* label)
{
    if (dst.empty())
        return;
    checkCuda(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T),
                  cudaMemcpyDeviceToHost),
        label);
}

#ifdef ZIRAN_ENABLE_CUSPARSE
void checkCusparse(cusparseStatus_t status, const char* label)
{
    if (status != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string(label) + ": cuSPARSE error " +
            std::to_string(static_cast<int>(status)));
}
	#endif

__host__ __device__ int floorToInt(double value)
{
    return static_cast<int>(floor(value));
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

__host__ __device__ void linearWeights(double x, double dx, int& base,
    double w[2], double dw[2])
{
    const double gx = x / dx;
    base = floorToInt(gx);
    const double t = gx - static_cast<double>(base);
    w[0] = 1.0 - t;
    w[1] = t;
    dw[0] = -1.0;
    dw[1] = 1.0;
}

__host__ __device__ void quadraticWeights(double x, double dx, int& base,
    double w[3], double dw[3])
{
    const double gx = x / dx;
    base = floorToInt(gx - 0.5);
    const double d0 = gx - static_cast<double>(base);
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

__host__ __device__ GridKey order0Key(const Vec3d& x, double dx)
{
    return packGridKey3DUnchecked(
        floorToInt(x.x / dx),
        floorToInt(x.y / dx),
        floorToInt(x.z / dx));
}

void validateCsr(const CsrMatrix& matrix)
{
    if (matrix.rows < 0 || matrix.cols < 0)
        throw std::runtime_error("CSR matrix has negative dimensions");
    if (matrix.row_offsets.size() != static_cast<std::size_t>(matrix.rows + 1))
        throw std::runtime_error("CSR row_offsets size mismatch");
    if (matrix.col_indices.size() != matrix.values.size())
        throw std::runtime_error("CSR col/value size mismatch");
    if (!matrix.row_offsets.empty() && matrix.row_offsets.front() != 0)
        throw std::runtime_error("CSR row_offsets must start at zero");
    if (!matrix.row_offsets.empty() &&
        matrix.row_offsets.back() != static_cast<int>(matrix.values.size()))
        throw std::runtime_error("CSR row_offsets must end at nonzero count");
    for (int row = 0; row < matrix.rows; ++row) {
        if (matrix.row_offsets[row] > matrix.row_offsets[row + 1])
            throw std::runtime_error("CSR row_offsets must be monotone");
    }
    for (int col : matrix.col_indices) {
        if (col < 0 || col >= matrix.cols)
            throw std::runtime_error("CSR column index out of range: col=" +
                std::to_string(col) + " cols=" + std::to_string(matrix.cols));
    }
}

struct DeviceCsrStorage {
    thrust::device_vector<int> row_offsets;
    thrust::device_vector<int> col_indices;
    thrust::device_vector<double> values;
};

struct DeviceVectorStorage {
    thrust::device_vector<double> values;
};

struct DeviceGridNodeMapStorage {
    thrust::device_vector<AssemblyGridNode> nodes;
};

struct PersistentSolidParticleStateStorage {
    thrust::device_vector<Vec3d> rest_x;
    thrust::device_vector<Vec3d> current_x;
    thrust::device_vector<Vec3d> velocity;
    thrust::device_vector<Mat3d> affine;
    thrust::device_vector<Mat3d> grad_velocity;
    thrust::device_vector<Mat3d> f;
    thrust::device_vector<double> j;
    thrust::device_vector<double> mass;
    thrust::device_vector<double> volume;
    thrust::device_vector<double> lambda;
    thrust::device_vector<int> missing_velocity_nodes;
    thrust::device_vector<int> missing_pressure_nodes;
    thrust::device_vector<GridKey> p2g_candidate_keys;
    thrust::device_vector<double> p2g_candidate_mass;
    thrust::device_vector<Vec3d> p2g_candidate_momentum;
    thrust::device_vector<GridKey> p2g_keys;
    thrust::device_vector<double> p2g_mass;
    thrust::device_vector<Vec3d> p2g_momentum;
    std::shared_ptr<DeviceGridNodeMapStorage> p2g_node_map;
    thrust::device_vector<Vec3d> p2g_velocity;
    std::vector<AssemblyGridNode> p2g_host_topology;
    std::vector<Vec3d> p2g_host_velocity;
    bool p2g_device_topology_valid = false;
    bool valid = false;
};

struct PersistentVelocityAlgebraStorage {
    std::shared_ptr<DeviceVectorStorage> pressure_gradient;
    std::shared_ptr<DeviceVectorStorage> rhs;
    std::shared_ptr<DeviceVectorStorage> velocity;
};

PersistentVelocityAlgebraStorage& velocityAlgebraStorage(
    PersistentVelocityAlgebra& workspace)
{
    if (!workspace.storage)
        workspace.storage = std::make_shared<PersistentVelocityAlgebraStorage>();
    return *std::static_pointer_cast<PersistentVelocityAlgebraStorage>(
        workspace.storage);
}

DeviceVector resizePersistentVector(std::shared_ptr<DeviceVectorStorage>& storage,
    std::size_t size)
{
    if (!storage)
        storage = std::make_shared<DeviceVectorStorage>();
    storage->values.resize(size);
    DeviceVector result;
    result.size = size;
    result.storage = storage;
    return result;
}

const DeviceCsrStorage& storageFor(const DeviceCsrMatrix& matrix)
{
    if (!matrix.storage)
        throw std::runtime_error("device CSR storage is empty");
    return *std::static_pointer_cast<DeviceCsrStorage>(matrix.storage);
}

DeviceCsrStorage& storageFor(DeviceCsrMatrix& matrix)
{
    if (!matrix.storage)
        matrix.storage = std::make_shared<DeviceCsrStorage>();
    return *std::static_pointer_cast<DeviceCsrStorage>(matrix.storage);
}

void validateDeviceCsr(const DeviceCsrMatrix& matrix)
{
    if (matrix.rows < 0 || matrix.cols < 0)
        throw std::runtime_error("device CSR matrix has negative dimensions");
    const DeviceCsrStorage& storage = storageFor(matrix);
    if (storage.row_offsets.size() != static_cast<std::size_t>(matrix.rows + 1))
        throw std::runtime_error("device CSR row_offsets size mismatch");
    if (storage.col_indices.size() != matrix.nonzeros ||
        storage.values.size() != matrix.nonzeros)
        throw std::runtime_error("device CSR nonzero size mismatch");
}

const DeviceVectorStorage& storageFor(const DeviceVector& vector)
{
    if (!vector.storage)
        throw std::runtime_error("device vector storage is empty");
    const auto& storage =
        *std::static_pointer_cast<DeviceVectorStorage>(vector.storage);
    if (storage.values.size() != vector.size)
        throw std::runtime_error("device vector size mismatch");
    return storage;
}

// GPU equivalent of finalizeCoo: sorts + reduces the COO entries on the device
// (thrust sort_by_key + reduce_by_key), compacts zeros, and constructs CSR row
// offsets without downloading the intermediate COO/key arrays.
constexpr long long kCooSentinelKey = std::numeric_limits<long long>::max();

struct CooToKey {
    int rows;
    int cols;
    __host__ __device__ long long operator()(const CooEntry& e) const
    {
        if (e.row < 0 || e.row >= rows || e.col < 0 || e.col >= cols ||
            e.value == 0.0)
            return kCooSentinelKey;
        return static_cast<long long>(e.row) * cols + e.col;
    }
};

struct CooToVal {
    __host__ __device__ double operator()(const CooEntry& e) const
    {
        return e.value;
    }
};

struct InvalidReducedCoo {
    __host__ __device__ bool operator()(
        const thrust::tuple<long long, double>& entry) const
    {
        return thrust::get<0>(entry) == kCooSentinelKey ||
            thrust::get<1>(entry) == 0.0;
    }
};

struct KeyToColumn {
    int cols;
    __host__ __device__ int operator()(long long key) const
    {
        return static_cast<int>(key % cols);
    }
};

__global__ void countCsrRowsKernel(const long long* keys, std::size_t count,
    int cols, int* row_offsets)
{
    const std::size_t entry =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (entry >= count)
        return;
    const int row = static_cast<int>(keys[entry] / cols);
    atomicAdd(row_offsets + row + 1, 1);
}

DeviceCsrMatrix finalizeCooToDevice(int rows, int cols,
    const thrust::device_vector<CooEntry>& d_entries)
{
    DeviceCsrMatrix result;
    result.rows = rows;
    result.cols = cols;
    auto storage = std::make_shared<DeviceCsrStorage>();
    storage->row_offsets.assign(static_cast<std::size_t>(rows + 1), 0);
    result.storage = storage;
    if (d_entries.empty()) {
        result.nonzeros = 0;
        return result;
    }

    const std::size_t n = d_entries.size();
    thrust::device_vector<long long> d_keys(n);
    thrust::device_vector<double> d_vals(n);
    thrust::transform(d_entries.begin(), d_entries.end(), d_keys.begin(),
        CooToKey{rows, cols});
    thrust::transform(d_entries.begin(), d_entries.end(), d_vals.begin(),
        CooToVal{});
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

    thrust::device_vector<long long> d_ukeys(n);
    thrust::device_vector<double> d_uvals(n);
    auto ends = thrust::reduce_by_key(d_keys.begin(), d_keys.end(),
        d_vals.begin(), d_ukeys.begin(), d_uvals.begin());
    std::size_t unique_count =
        static_cast<std::size_t>(ends.first - d_ukeys.begin());

    auto zipped_begin = thrust::make_zip_iterator(
        thrust::make_tuple(d_ukeys.begin(), d_uvals.begin()));
    auto zipped_end = zipped_begin + unique_count;
    const auto valid_end =
        thrust::remove_if(zipped_begin, zipped_end, InvalidReducedCoo{});
    unique_count = static_cast<std::size_t>(valid_end - zipped_begin);

    result.nonzeros = unique_count;
    storage->col_indices.resize(unique_count);
    storage->values.assign(d_uvals.begin(), d_uvals.begin() + unique_count);
    thrust::transform(d_ukeys.begin(), d_ukeys.begin() + unique_count,
        storage->col_indices.begin(), KeyToColumn{cols});
    if (unique_count != 0) {
        constexpr int block_size = 256;
        const int block_count =
            static_cast<int>((unique_count + block_size - 1) / block_size);
        countCsrRowsKernel<<<block_count, block_size>>>(
            thrust::raw_pointer_cast(d_ukeys.data()), unique_count, cols,
            thrust::raw_pointer_cast(storage->row_offsets.data()));
        checkCuda(cudaGetLastError(), "countCsrRowsKernel launch");
        thrust::inclusive_scan(storage->row_offsets.begin(),
            storage->row_offsets.end(), storage->row_offsets.begin());
    }
    validateDeviceCsr(result);
    return result;
}

struct AssemblyGridNodeKeyLess {
    __host__ __device__ bool operator()(const AssemblyGridNode& left,
        const AssemblyGridNode& right) const
    {
        return left.key < right.key;
    }
};

struct SolidP2gTupleAdd {
    __host__ __device__ thrust::tuple<double, Vec3d> operator()(
        const thrust::tuple<double, Vec3d>& left,
        const thrust::tuple<double, Vec3d>& right) const
    {
        const Vec3d a = thrust::get<1>(left);
        const Vec3d b = thrust::get<1>(right);
        return thrust::make_tuple(thrust::get<0>(left) + thrust::get<0>(right),
            Vec3d{a.x + b.x, a.y + b.y, a.z + b.z});
    }
};

__global__ void generateSolidQ2P2gContributionsKernel(
    const Vec3d* rest_x, const Vec3d* velocity, const Mat3d* affine,
    const double* mass, int particle_count, double dx, GridKey* keys,
    double* contribution_mass, Vec3d* contribution_momentum)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    int base[3];
    double w[3][3];
    double ignored_dw[3][3];
    quadraticWeights(rest_x[p].x, dx, base[0], w[0], ignored_dw[0]);
    quadraticWeights(rest_x[p].y, dx, base[1], w[1], ignored_dw[1]);
    quadraticWeights(rest_x[p].z, dx, base[2], w[2], ignored_dw[2]);
    int local = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k, ++local) {
                const int out = p * 27 + local;
                const double weight = w[0][i] * w[1][j] * w[2][k];
                const double weighted_mass = weight * mass[p];
                const double displacement[3] = {
                    (static_cast<double>(base[0] + i) * dx) - rest_x[p].x,
                    (static_cast<double>(base[1] + j) * dx) - rest_x[p].y,
                    (static_cast<double>(base[2] + k) * dx) - rest_x[p].z};
                double apic_velocity[3] = {
                    velocity[p].x, velocity[p].y, velocity[p].z};
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 3; ++col)
                        apic_velocity[row] +=
                            affine[p].values[row * 3 + col] * displacement[col];
                keys[out] = packGridKey3DUnchecked(
                    base[0] + i, base[1] + j, base[2] + k);
                contribution_mass[out] = weighted_mass;
                contribution_momentum[out] = Vec3d{
                    weighted_mass * apic_velocity[0],
                    weighted_mass * apic_velocity[1],
                    weighted_mass * apic_velocity[2]};
            }
}

__global__ void normalizeSolidQ2P2gNodesKernel(const GridKey* keys,
    const double* mass, const Vec3d* momentum, int node_count,
    AssemblyGridNode* nodes, Vec3d* velocity, int index_mode)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= node_count)
        return;
    const double inverse_mass = mass[i] != 0.0 ? 1.0 / mass[i] : 0.0;
    AssemblyGridNode node;
    if (index_mode == 1)
        node.idx = nodes[i].idx;
    else if (index_mode == 2)
        node.idx = i;
    node.key = keys[i];
    node.mass = mass[i];
    node.v = Vec3d{momentum[i].x * inverse_mass,
        momentum[i].y * inverse_mass, momentum[i].z * inverse_mass};
    nodes[i] = node;
    velocity[i] = node.v;
}

struct SolidGridKeyMatchesNode {
    __host__ __device__ bool operator()(
        GridKey key, const AssemblyGridNode& node) const
    {
        return key == node.key;
    }
};

struct SolidParticleReductionValue {
    double min_x, min_y, min_z;
    double max_x, max_y, max_z;
    double max_speed_squared;
};

struct SolidParticleReductionTransform {
    const Vec3d* rest_x;
    const Vec3d* velocity;
    __host__ __device__ SolidParticleReductionValue operator()(int i) const
    {
        const Vec3d x = rest_x[i];
        const Vec3d v = velocity[i];
        return {x.x, x.y, x.z, x.x, x.y, x.z,
            v.x * v.x + v.y * v.y + v.z * v.z};
    }
};

struct SolidParticleReductionCombine {
    __host__ __device__ SolidParticleReductionValue operator()(
        const SolidParticleReductionValue& a,
        const SolidParticleReductionValue& b) const
    {
        return {fmin(a.min_x, b.min_x), fmin(a.min_y, b.min_y),
            fmin(a.min_z, b.min_z), fmax(a.max_x, b.max_x),
            fmax(a.max_y, b.max_y), fmax(a.max_z, b.max_z),
            fmax(a.max_speed_squared, b.max_speed_squared)};
    }
};

__device__ int findNodeIdxDevice(const AssemblyGridNode* nodes, int count, GridKey key)
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
    if (first < count && nodes[first].key == key)
        return nodes[first].idx;
    return -1;
}

__device__ int findNodePositionDevice(
    const AssemblyGridNode* nodes, int count, GridKey key)
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
    return first < count && nodes[first].key == key ? first : -1;
}

__device__ int findGridKeyPositionDevice(
    const GridKey* keys, int count, GridKey key)
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
    return first < count && keys[first] == key ? first : -1;
}

__global__ void generateInterfaceWRowKeysKernel(const Vec3d* current_x,
    int sample_count, double dx, GridKey* row_keys)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= sample_count)
        return;
    const Vec3d shifted{current_x[q].x - 0.5 * dx,
        current_x[q].y - 0.5 * dx,
        current_x[q].z - 0.5 * dx};
    row_keys[q] = order0Key(shifted, dx);
}

__global__ void buildInterfaceWKernel(const Vec3d* rest_x,
    const Vec3d* current_x, const Vec3d* normal, const double* length,
    int sample_count, double dx, int velocity_degree,
    int shift_velocity_half_dx, const GridKey* row_keys, int row_count,
    const AssemblyGridNode* velocity_nodes, int velocity_node_count,
    CooEntry* entries)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= sample_count)
        return;
    const Vec3d row_x{current_x[q].x - 0.5 * dx,
        current_x[q].y - 0.5 * dx,
        current_x[q].z - 0.5 * dx};
    const int row = findGridKeyPositionDevice(
        row_keys, row_count, order0Key(row_x, dx));
    if (row < 0)
        return;
    const double shift = shift_velocity_half_dx ? 0.5 * dx : 0.0;
    const Vec3d velocity_x{rest_x[q].x - shift,
        rest_x[q].y - shift, rest_x[q].z - shift};
    const double normal_value[3] = {
        normal[q].x, normal[q].y, normal[q].z};
    if (velocity_degree == 1) {
        int base[3];
        double w[3][2];
        double ignored_dw[3][2];
        linearWeights(velocity_x.x, dx, base[0], w[0], ignored_dw[0]);
        linearWeights(velocity_x.y, dx, base[1], w[1], ignored_dw[1]);
        linearWeights(velocity_x.z, dx, base[2], w[2], ignored_dw[2]);
        int out = q * 8 * 3;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    const int velocity_idx = findNodeIdxDevice(velocity_nodes,
                        velocity_node_count, packGridKey3DUnchecked(
                            base[0] + i, base[1] + j, base[2] + k));
                    const double weight = w[0][i] * w[1][j] * w[2][k];
                    for (int alpha = 0; alpha < 3; ++alpha, ++out)
                        if (velocity_idx >= 0)
                            entries[out] = CooEntry{row,
                                velocity_idx * 3 + alpha,
                                length[q] * weight * normal_value[alpha]};
                }
    } else {
        int base[3];
        double w[3][3];
        double ignored_dw[3][3];
        quadraticWeights(velocity_x.x, dx, base[0], w[0], ignored_dw[0]);
        quadraticWeights(velocity_x.y, dx, base[1], w[1], ignored_dw[1]);
        quadraticWeights(velocity_x.z, dx, base[2], w[2], ignored_dw[2]);
        int out = q * 27 * 3;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) {
                    const int velocity_idx = findNodeIdxDevice(velocity_nodes,
                        velocity_node_count, packGridKey3DUnchecked(
                            base[0] + i, base[1] + j, base[2] + k));
                    const double weight = w[0][i] * w[1][j] * w[2][k];
                    for (int alpha = 0; alpha < 3; ++alpha, ++out)
                        if (velocity_idx >= 0)
                            entries[out] = CooEntry{row,
                                velocity_idx * 3 + alpha,
                                length[q] * weight * normal_value[alpha]};
                }
    }
}

__global__ void buildFluidMinvAndRhsKernel(const AssemblyGridNode* nodes,
    int node_count, double dt, CooEntry* minv_entries, double* rhs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= node_count)
        return;
    const AssemblyGridNode node = nodes[i];
    if (node.idx < 0 || node.mass == 0.0)
        return;
    const double inv_mass_dt = dt / node.mass;
    const double mass_over_dt = node.mass / dt;
    const double v[3] = {node.new_v.x, node.new_v.y, node.new_v.z};
    for (int d = 0; d < 3; ++d) {
        const int row = node.idx * 3 + d;
        minv_entries[i * 3 + d] = CooEntry{row, row, inv_mass_dt};
        rhs[row] = mass_over_dt * v[d];
    }
}

__global__ void buildFluidVolumeGKernel(const Vec3d* volume_xq,
    const double* volume_mass, int q_count, double dx, double rho,
    const AssemblyGridNode* velocity_nodes, int velocity_count,
    const AssemblyGridNode* pressure_nodes, int pressure_count,
    CooEntry* g_entries)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= q_count)
        return;
    Vec3d x{volume_xq[q].x - 0.5 * dx,
        volume_xq[q].y - 0.5 * dx,
        volume_xq[q].z - 0.5 * dx};
    int base[3];
    double w[3][2];
    double dw[3][2];
    linearWeights(x.x, dx, base[0], w[0], dw[0]);
    linearWeights(x.y, dx, base[1], w[1], dw[1]);
    linearWeights(x.z, dx, base[2], w[2], dw[2]);
    const int pressure_idx =
        findNodeIdxDevice(pressure_nodes, pressure_count, order0Key(x, dx));
    if (pressure_idx < 0)
        return;
    const double vol = volume_mass[q] / rho;
    const double inv_dx = 1.0 / dx;
    int out = q * 8 * 3;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) {
                const int vel_idx = findNodeIdxDevice(velocity_nodes, velocity_count,
                    packGridKey3DUnchecked(base[0] + i, base[1] + j, base[2] + k));
                const double grad[3] = {
                    inv_dx * dw[0][i] * w[1][j] * w[2][k],
                    inv_dx * w[0][i] * dw[1][j] * w[2][k],
                    inv_dx * w[0][i] * w[1][j] * dw[2][k]};
                for (int alpha = 0; alpha < 3; ++alpha, ++out) {
                    if (vel_idx >= 0)
                        g_entries[out] = CooEntry{
                            vel_idx * 3 + alpha, pressure_idx,
                            -vol * grad[alpha]};
                }
            }
}

__global__ void buildFluidFaceGKernel(const Vec3d* face_xq,
    const Vec3d* face_normal, int q_count, double dx, int pressure_count,
    const AssemblyGridNode* velocity_nodes, int velocity_count,
    const AssemblyGridNode* face_nodes, int face_count, CooEntry* g_entries)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= q_count)
        return;
    Vec3d x{face_xq[q].x - 0.5 * dx,
        face_xq[q].y - 0.5 * dx,
        face_xq[q].z - 0.5 * dx};
    int base[3];
    double w[3][2];
    double dw[3][2];
    linearWeights(x.x, dx, base[0], w[0], dw[0]);
    linearWeights(x.y, dx, base[1], w[1], dw[1]);
    linearWeights(x.z, dx, base[2], w[2], dw[2]);
    const double normal[3] = {
        face_normal[q].x, face_normal[q].y, face_normal[q].z};
    const double face_len = dx * dx;
    int out = q * 8 * 2 * 3;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) {
                const int vel_idx = findNodeIdxDevice(velocity_nodes, velocity_count,
                    packGridKey3DUnchecked(base[0] + i, base[1] + j, base[2] + k));
                const double weight = w[0][i] * w[1][j] * w[2][k];
                for (int delta_index = 0; delta_index < 2; ++delta_index) {
                    const double delta = delta_index == 0 ? -1.0 : 1.0;
                    Vec3d sample{
                        face_xq[q].x + face_normal[q].x * 0.5 * dx * delta - 0.5 * dx,
                        face_xq[q].y + face_normal[q].y * 0.5 * dx * delta - 0.5 * dx,
                        face_xq[q].z + face_normal[q].z * 0.5 * dx * delta - 0.5 * dx};
                    const int face_idx =
                        findNodeIdxDevice(face_nodes, face_count, order0Key(sample, dx));
                    for (int alpha = 0; alpha < 3; ++alpha, ++out) {
                        if (vel_idx >= 0 && face_idx >= 0)
                            g_entries[out] = CooEntry{
                                vel_idx * 3 + alpha,
                                pressure_count + face_idx,
                                face_len * weight * normal[alpha]};
                    }
                }
            }
}

__global__ void buildSolidMinvAndRhsKernel(const AssemblyGridNode* nodes,
    int node_count, double dt, Vec3d gravity, CooEntry* minv_entries,
    double* rhs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= node_count)
        return;
    const AssemblyGridNode node = nodes[i];
    if (node.idx < 0 || node.mass == 0.0)
        return;
    const double inv_mass_dt = dt / node.mass;
    const double mass_over_dt = node.mass / dt;
    const double v[3] = {
        node.new_v.x + dt * gravity.x,
        node.new_v.y + dt * gravity.y,
        node.new_v.z + dt * gravity.z};
    for (int d = 0; d < 3; ++d) {
        const int row = node.idx * 3 + d;
        minv_entries[i * 3 + d] = CooEntry{row, row, inv_mass_dt};
        rhs[row] = mass_over_dt * v[d];
    }
}

__global__ void buildSolidSAndRhsKernel(const AssemblyGridNode* nodes,
    int node_count, double dt, CooEntry* s_entries, double* rhs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= node_count)
        return;
    const AssemblyGridNode node = nodes[i];
    if (node.idx < 0)
        return;
    s_entries[i] = CooEntry{node.idx, node.idx, node.v.x / dt};
    rhs[node.idx] = node.v.x * node.v.y / dt;
}

// Independent 3x3 one-sided Jacobi SVD.  B converges to F*V with orthogonal
// columns, so F^{-T} = B*diag(1/sigma^2)*V^T.  Singular values below the same
// 1e-12 cutoff used by the CPU reference are dropped.
__device__ Mat3d pseudoInverseTranspose3x3(const Mat3d& f)
{
    double b[3][3];
    double v[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            b[row][col] = f.values[row * 3 + col];

    for (int sweep = 0; sweep < 12; ++sweep) {
        for (int pair = 0; pair < 3; ++pair) {
            const int p = pair == 0 ? 0 : (pair == 1 ? 0 : 1);
            const int q = pair == 0 ? 1 : (pair == 1 ? 2 : 2);
            double alpha = 0.0;
            double beta = 0.0;
            double gamma = 0.0;
            for (int row = 0; row < 3; ++row) {
                alpha += b[row][p] * b[row][p];
                beta += b[row][q] * b[row][q];
                gamma += b[row][p] * b[row][q];
            }
            if (gamma == 0.0 ||
                fabs(gamma) <= 1e-15 * sqrt(fmax(alpha * beta, 0.0)))
                continue;
            const double zeta = (beta - alpha) / (2.0 * gamma);
            const double sign = zeta >= 0.0 ? 1.0 : -1.0;
            const double t = sign / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
            const double c = 1.0 / sqrt(1.0 + t * t);
            const double s = c * t;
            for (int row = 0; row < 3; ++row) {
                const double bp = b[row][p];
                const double bq = b[row][q];
                b[row][p] = c * bp - s * bq;
                b[row][q] = s * bp + c * bq;
                const double vp = v[row][p];
                const double vq = v[row][q];
                v[row][p] = c * vp - s * vq;
                v[row][q] = s * vp + c * vq;
            }
        }
    }

    double inv_sigma_squared[3];
    for (int col = 0; col < 3; ++col) {
        double sigma_squared = 0.0;
        for (int row = 0; row < 3; ++row)
            sigma_squared += b[row][col] * b[row][col];
        inv_sigma_squared[col] = sigma_squared >= 1e-24
            ? 1.0 / sigma_squared
            : 0.0;
    }

    Mat3d result;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            double value = 0.0;
            for (int singular = 0; singular < 3; ++singular)
                value += b[row][singular] * inv_sigma_squared[singular] *
                    v[col][singular];
            result.values[row * 3 + col] = value;
        }
    }
    return result;
}

__global__ void buildSolidGKernel(const SolidQ2Q0Particle* particles,
    int particle_count, double dx, const AssemblyGridNode* velocity_nodes,
    int velocity_count, const AssemblyGridNode* pressure_nodes,
    int pressure_count, CooEntry* g_entries)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    const SolidQ2Q0Particle particle = particles[p];
    Mat3d f_inv_t;
    f_inv_t.values[0] = 1.0;
    f_inv_t.values[4] = 1.0;
    f_inv_t.values[8] = 1.0;
    if (particle.compute_f_inv_t)
        f_inv_t = pseudoInverseTranspose3x3(particle.f);
    int base[3];
    double w[3][3];
    double dw[3][3];
    quadraticWeights(particle.x.x, dx, base[0], w[0], dw[0]);
    quadraticWeights(particle.x.y, dx, base[1], w[1], dw[1]);
    quadraticWeights(particle.x.z, dx, base[2], w[2], dw[2]);
    Vec3d pressure_x{particle.x_for_p.x - 0.5 * dx,
        particle.x_for_p.y - 0.5 * dx,
        particle.x_for_p.z - 0.5 * dx};
    const int pressure_idx =
        findNodeIdxDevice(pressure_nodes, pressure_count, order0Key(pressure_x, dx));
    if (pressure_idx < 0)
        return;
    const double inv_dx = 1.0 / dx;
    int out = p * 27 * 3;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) {
                const int velocity_idx = findNodeIdxDevice(velocity_nodes, velocity_count,
                    packGridKey3DUnchecked(base[0] + i, base[1] + j, base[2] + k));
                const double grad_index[3] = {
                    inv_dx * dw[0][i] * w[1][j] * w[2][k],
                    inv_dx * w[0][i] * dw[1][j] * w[2][k],
                    inv_dx * w[0][i] * w[1][j] * dw[2][k]};
                double grad[3] = {0.0, 0.0, 0.0};
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 3; ++col)
                        grad[row] += f_inv_t.values[row * 3 + col] *
                            grad_index[col];
                const double scale = -particle.volume * particle.j;
                for (int alpha = 0; alpha < 3; ++alpha, ++out) {
                    if (velocity_idx >= 0)
                        g_entries[out] = CooEntry{
                            velocity_idx * 3 + alpha, pressure_idx,
                            scale * grad[alpha]};
                }
            }
}

__global__ void buildSolidGResidentKernel(const Vec3d* rest_x,
    const double* volume, const double* j_value, const Mat3d* f,
    int particle_count, int compute_f_inv_t, double dx,
    const AssemblyGridNode* velocity_nodes, int velocity_count,
    const AssemblyGridNode* pressure_nodes, int pressure_count,
    CooEntry* g_entries)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    Mat3d f_inv_t;
    f_inv_t.values[0] = 1.0;
    f_inv_t.values[4] = 1.0;
    f_inv_t.values[8] = 1.0;
    if (compute_f_inv_t)
        f_inv_t = pseudoInverseTranspose3x3(f[p]);
    int base[3];
    double w[3][3];
    double dw[3][3];
    quadraticWeights(rest_x[p].x, dx, base[0], w[0], dw[0]);
    quadraticWeights(rest_x[p].y, dx, base[1], w[1], dw[1]);
    quadraticWeights(rest_x[p].z, dx, base[2], w[2], dw[2]);
    const Vec3d pressure_x{rest_x[p].x - 0.5 * dx,
        rest_x[p].y - 0.5 * dx, rest_x[p].z - 0.5 * dx};
    const int pressure_idx = findNodeIdxDevice(
        pressure_nodes, pressure_count, order0Key(pressure_x, dx));
    if (pressure_idx < 0)
        return;
    const double inv_dx = 1.0 / dx;
    int out = p * 27 * 3;
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l) {
                const int velocity_idx = findNodeIdxDevice(velocity_nodes,
                    velocity_count, packGridKey3DUnchecked(
                        base[0] + i, base[1] + k, base[2] + l));
                const double grad_index[3] = {
                    inv_dx * dw[0][i] * w[1][k] * w[2][l],
                    inv_dx * w[0][i] * dw[1][k] * w[2][l],
                    inv_dx * w[0][i] * w[1][k] * dw[2][l]};
                double grad[3] = {0.0, 0.0, 0.0};
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 3; ++col)
                        grad[row] += f_inv_t.values[row * 3 + col] *
                            grad_index[col];
                const double scale = -volume[p] * j_value[p];
                for (int alpha = 0; alpha < 3; ++alpha, ++out)
                    if (velocity_idx >= 0)
                        g_entries[out] = CooEntry{
                            velocity_idx * 3 + alpha, pressure_idx,
                            scale * grad[alpha]};
            }
}

__global__ void updateSolidParticlesQ2Kernel(Vec3d* current_x,
    Vec3d* particle_velocity, Mat3d* affine, Mat3d* grad_velocity,
    Mat3d* f, double* j_value, const double* lambda,
    const Vec3d* rest_x, int particle_count,
    const AssemblyGridNode* velocity_nodes, int velocity_node_count,
    const AssemblyGridNode* pressure_nodes, int pressure_node_count,
    const double* solved_velocity, int solved_velocity_count,
    const double* solved_pressure, int solved_pressure_count,
    double dx, double dt, double apic_rpic_ratio, double d_inverse,
    bool grid_only,
    int* missing_velocity_nodes, int* missing_pressure_nodes)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= particle_count)
        return;
    int base[3];
    double w[3][3];
    double dw[3][3];
    quadraticWeights(rest_x[p].x, dx, base[0], w[0], dw[0]);
    quadraticWeights(rest_x[p].y, dx, base[1], w[1], dw[1]);
    quadraticWeights(rest_x[p].z, dx, base[2], w[2], dw[2]);
    const double inv_dx = 1.0 / dx;
    double pic[3] = {0.0, 0.0, 0.0};
    double bp[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double grad[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l) {
                const double weight = w[0][i] * w[1][k] * w[2][l];
                if (weight == 0.0)
                    continue;
                const GridKey key = packGridKey3DUnchecked(
                    base[0] + i, base[1] + k, base[2] + l);
                const int found = findNodePositionDevice(
                    velocity_nodes, velocity_node_count, key);
                if (found < 0 ||
                    velocity_nodes[found].idx < 0 ||
                    (!grid_only && velocity_nodes[found].idx * 3 + 2 >=
                        solved_velocity_count)) {
                    atomicAdd(missing_velocity_nodes, 1);
                    continue;
                }
                const int row_offset = velocity_nodes[found].idx * 3;
                const Vec3d grid_v = velocity_nodes[found].new_v;
                const double node_v[3] = {
                    grid_only ? grid_v.x : solved_velocity[row_offset],
                    grid_only ? grid_v.y : solved_velocity[row_offset + 1],
                    grid_only ? grid_v.z : solved_velocity[row_offset + 2]};
                const double node_x[3] = {
                    static_cast<double>(base[0] + i) * dx,
                    static_cast<double>(base[1] + k) * dx,
                    static_cast<double>(base[2] + l) * dx};
                const double displacement[3] = {node_x[0] - rest_x[p].x,
                    node_x[1] - rest_x[p].y,
                    node_x[2] - rest_x[p].z};
                const double weight_gradient[3] = {
                    inv_dx * dw[0][i] * w[1][k] * w[2][l],
                    inv_dx * w[0][i] * dw[1][k] * w[2][l],
                    inv_dx * w[0][i] * w[1][k] * dw[2][l]};
                for (int row = 0; row < 3; ++row) {
                    pic[row] += weight * node_v[row];
                    for (int col = 0; col < 3; ++col) {
                        bp[row * 3 + col] += weight * node_v[row] *
                            displacement[col];
                        grad[row * 3 + col] += node_v[row] *
                            weight_gradient[col];
                    }
                }
            }
    particle_velocity[p] = Vec3d{pic[0], pic[1], pic[2]};
    current_x[p] = Vec3d{current_x[p].x + dt * pic[0],
        current_x[p].y + dt * pic[1], current_x[p].z + dt * pic[2]};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const int index = row * 3 + col;
            grad_velocity[p].values[index] = grad[index];
            f[p].values[index] += dt * grad[index];
            const double cc = bp[index] * d_inverse;
            const double cc_t = bp[col * 3 + row] * d_inverse;
            affine[p].values[index] =
                0.5 * (apic_rpic_ratio + 1.0) * cc +
                0.5 * (apic_rpic_ratio - 1.0) * cc_t;
        }
    }

    if (grid_only) {
        const double* ff = f[p].values;
        const double determinant =
            ff[0] * (ff[4] * ff[8] - ff[5] * ff[7]) -
            ff[1] * (ff[3] * ff[8] - ff[5] * ff[6]) +
            ff[2] * (ff[3] * ff[7] - ff[4] * ff[6]);
        j_value[p] = determinant < 0.3 ? 0.3 :
            (determinant > 3.0 ? 3.0 : determinant);
        return;
    }
    const Vec3d pressure_x{rest_x[p].x - 0.5 * dx,
        rest_x[p].y - 0.5 * dx, rest_x[p].z - 0.5 * dx};
    const GridKey pressure_key = order0Key(pressure_x, dx);
    const int found_pressure = findNodePositionDevice(
        pressure_nodes, pressure_node_count, pressure_key);
    if (found_pressure < 0 ||
        pressure_nodes[found_pressure].idx < 0 ||
        pressure_nodes[found_pressure].idx >= solved_pressure_count) {
        atomicAdd(missing_pressure_nodes, 1);
        return;
    }
    double updated_j = 1.0 -
        solved_pressure[pressure_nodes[found_pressure].idx] / lambda[p];
    updated_j = updated_j < 0.3 ? 0.3 : (updated_j > 3.0 ? 3.0 : updated_j);
    j_value[p] = updated_j;
}

std::vector<double> eigenProduct(const Eigen::SparseMatrix<double>& matrix,
    const std::vector<double>& x)
{
    Eigen::Map<const Eigen::VectorXd> x_map(x.data(), static_cast<Eigen::Index>(x.size()));
    Eigen::VectorXd y = matrix * x_map;
    return std::vector<double>(y.data(), y.data() + y.size());
}

__global__ void csrSpmvKernel(const int rows, const int* row_offsets,
    const int* col_indices, const double* values, const double* x, double* y)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    double sum = 0.0;
    for (int k = row_offsets[row]; k < row_offsets[row + 1]; ++k)
        sum += values[k] * x[col_indices[k]];
    y[row] = sum;
}

__global__ void subtractVectorKernel(const double* left, const double* right,
    int size, double* output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        output[i] = left[i] - right[i];
}

__global__ void csrBlockToCooKernel(int rows, const int* row_offsets,
    const int* col_indices, const double* values, std::size_t output_offset,
    int row_offset, int col_offset, double scale, CooEntry* entries)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    for (int entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry) {
        entries[output_offset + static_cast<std::size_t>(entry)] = CooEntry{
            row_offset + row, col_offset + col_indices[entry],
            scale * values[entry]};
    }
}

struct ScaleValue {
    double scale;
    __host__ __device__ double operator()(double value) const
    {
        return scale * value;
    }
};

struct AddScaledValue {
    double right_scale;
    __host__ __device__ double operator()(double left, double right) const
    {
        return left + right_scale * right;
    }
};

__global__ void buildJacobiInverseDiagonalKernel(const int rows, const int* row_offsets,
    const int* col_indices, const double* values, double* inv_diag)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    double diag = 0.0;
    for (int k = row_offsets[row]; k < row_offsets[row + 1]; ++k) {
        if (col_indices[k] == row) {
            diag = values[k];
            break;
        }
    }
    inv_diag[row] = fabs(diag) > 1e-30 ? 1.0 / diag : 1.0;
}

struct AxpyFunctor {
    double alpha;
    __host__ __device__ double operator()(const double x, const double y) const
    {
        return x + alpha * y;
    }
};

struct XmyFunctor {
    double alpha;
    __host__ __device__ double operator()(const double x, const double y) const
    {
        return x - alpha * y;
    }
};

struct ZPlusBetaPFunctor {
    double beta;
    __host__ __device__ double operator()(const double z, const double p) const
    {
        return z + beta * p;
    }
};

struct WeightedJacobiUpdateFunctor {
    double omega;
    __host__ __device__ double operator()(const thrust::tuple<double, double, double>& t) const
    {
        const double z = thrust::get<0>(t);
        const double inv_diag = thrust::get<1>(t);
        const double residual = thrust::get<2>(t);
        return z + omega * inv_diag * residual;
    }
};

double deviceDot(const thrust::device_vector<double>& a, const thrust::device_vector<double>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error("deviceDot input size mismatch");
    return thrust::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

void applyJacobiSweepPreconditioner(const CsrMatrix& matrix,
    const thrust::device_vector<int>& d_row_offsets,
    const thrust::device_vector<int>& d_col_indices,
    const thrust::device_vector<double>& d_values,
    const thrust::device_vector<double>& d_inv_diag,
    const thrust::device_vector<double>& d_r,
    int block_count, int block_size, int sweeps, double omega,
    thrust::device_vector<double>& d_z,
    thrust::device_vector<double>& d_Az,
    thrust::device_vector<double>& d_tmp)
{
    thrust::fill(d_z.begin(), d_z.end(), 0.0);
    if (sweeps <= 1) {
        thrust::transform(d_inv_diag.begin(), d_inv_diag.end(), d_r.begin(), d_z.begin(),
            thrust::multiplies<double>());
        return;
    }

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        csrSpmvKernel<<<block_count, block_size>>>(matrix.rows,
            thrust::raw_pointer_cast(d_row_offsets.data()),
            thrust::raw_pointer_cast(d_col_indices.data()),
            thrust::raw_pointer_cast(d_values.data()),
            thrust::raw_pointer_cast(d_z.data()),
            thrust::raw_pointer_cast(d_Az.data()));
        checkCuda(cudaGetLastError(), "csrSpmvKernel preconditioner launch");
        checkCuda(cudaDeviceSynchronize(), "csrSpmvKernel preconditioner synchronize");

        thrust::transform(d_r.begin(), d_r.end(), d_Az.begin(), d_tmp.begin(), thrust::minus<double>());
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(
            d_z.begin(), d_inv_diag.begin(), d_tmp.begin()));
        auto end = thrust::make_zip_iterator(thrust::make_tuple(
            d_z.end(), d_inv_diag.end(), d_tmp.end()));
        thrust::transform(begin, end, d_z.begin(), WeightedJacobiUpdateFunctor{omega});
    }
}

} // namespace

GridKey packGridKey3D(int x, int y, int z)
{
    const int bx = x + kCoordBias;
    const int by = y + kCoordBias;
    const int bz = z + kCoordBias;
    if (bx < 0 || by < 0 || bz < 0 ||
        bx > static_cast<int>(kCoordMask) ||
        by > static_cast<int>(kCoordMask) ||
        bz > static_cast<int>(kCoordMask))
        throw std::out_of_range("grid coordinate is outside GPU_STAGE3A::packGridKey3D range");
    return packGridKey3DUnchecked(x, y, z);
}

CsrMatrix makeCsrFromEigen(const Eigen::SparseMatrix<double>& matrix)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> row_major(matrix);
    row_major.makeCompressed();

    CsrMatrix csr;
    csr.rows = static_cast<int>(row_major.rows());
    csr.cols = static_cast<int>(row_major.cols());
    csr.row_offsets.assign(row_major.outerIndexPtr(),
        row_major.outerIndexPtr() + row_major.outerSize() + 1);
    csr.col_indices.assign(row_major.innerIndexPtr(),
        row_major.innerIndexPtr() + row_major.nonZeros());
    csr.values.assign(row_major.valuePtr(), row_major.valuePtr() + row_major.nonZeros());
    return csr;
}

Eigen::SparseMatrix<double> makeEigenFromCsr(const CsrMatrix& matrix)
{
    validateCsr(matrix);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(matrix.values.size());
    for (int row = 0; row < matrix.rows; ++row) {
        for (int k = matrix.row_offsets[row]; k < matrix.row_offsets[row + 1]; ++k) {
            triplets.emplace_back(row, matrix.col_indices[static_cast<std::size_t>(k)],
                matrix.values[static_cast<std::size_t>(k)]);
        }
    }
    Eigen::SparseMatrix<double> result(matrix.rows, matrix.cols);
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    return result;
}

DeviceCsrMatrix makeDeviceCsrFromHost(const CsrMatrix& matrix)
{
    validateCsr(matrix);
    DeviceCsrMatrix device;
    device.rows = matrix.rows;
    device.cols = matrix.cols;
    device.nonzeros = matrix.values.size();
    auto storage = std::make_shared<DeviceCsrStorage>();
    storage->row_offsets.assign(matrix.row_offsets.begin(), matrix.row_offsets.end());
    storage->col_indices.assign(matrix.col_indices.begin(), matrix.col_indices.end());
    storage->values.assign(matrix.values.begin(), matrix.values.end());
    device.storage = storage;
    validateDeviceCsr(device);
    return device;
}

DeviceCsrRawView getDeviceCsrRawView(const DeviceCsrMatrix& matrix)
{
    validateDeviceCsr(matrix);
    const DeviceCsrStorage& storage = storageFor(matrix);
    return {thrust::raw_pointer_cast(storage.row_offsets.data()),
        thrust::raw_pointer_cast(storage.col_indices.data()),
        thrust::raw_pointer_cast(storage.values.data())};
}

CsrMatrix copyDeviceCsrToHost(const DeviceCsrMatrix& matrix)
{
    validateDeviceCsr(matrix);
    const DeviceCsrStorage& storage = storageFor(matrix);
    CsrMatrix host;
    host.rows = matrix.rows;
    host.cols = matrix.cols;
    host.row_offsets.resize(storage.row_offsets.size());
    host.col_indices.resize(storage.col_indices.size());
    host.values.resize(storage.values.size());
    thrust::copy(storage.row_offsets.begin(), storage.row_offsets.end(),
        host.row_offsets.begin());
    thrust::copy(storage.col_indices.begin(), storage.col_indices.end(),
        host.col_indices.begin());
    thrust::copy(storage.values.begin(), storage.values.end(), host.values.begin());
    validateCsr(host);
    return host;
}

DeviceVector makeDeviceVectorFromHost(const std::vector<double>& values)
{
    DeviceVector result;
    result.size = values.size();
    auto storage = std::make_shared<DeviceVectorStorage>();
    storage->values.assign(values.begin(), values.end());
    result.storage = storage;
    return result;
}

DeviceVector makeDeviceVectorFromDevice(const double* values, std::size_t size)
{
    if (size != 0 && !values)
        throw std::runtime_error("device vector source pointer is null");
    DeviceVector result;
    result.size = size;
    auto storage = std::make_shared<DeviceVectorStorage>();
    storage->values.resize(size);
    if (size != 0)
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage->values.data()), values,
                      size * sizeof(double), cudaMemcpyDeviceToDevice),
            "device vector device-to-device copy");
    result.storage = storage;
    return result;
}

DeviceVectorRawView getDeviceVectorRawView(const DeviceVector& vector)
{
    const DeviceVectorStorage& storage = storageFor(vector);
    return DeviceVectorRawView{vector.size == 0 ? nullptr
        : thrust::raw_pointer_cast(storage.values.data())};
}

DeviceGridNodeMapRawView getDeviceGridNodeMapRawView(
    const DeviceGridNodeMap& map)
{
    if (!map.storage)
        throw std::runtime_error("device grid-node map storage is empty");
    const auto storage =
        std::static_pointer_cast<DeviceGridNodeMapStorage>(map.storage);
    if (storage->nodes.size() != map.size)
        throw std::runtime_error("device grid-node map size mismatch");
    return {map.size == 0 ? nullptr
        : thrust::raw_pointer_cast(storage->nodes.data())};
}

DeviceGridNodeMapMutableRawView getDeviceGridNodeMapMutableRawView(
    DeviceGridNodeMap& map)
{
    if (!map.storage)
        throw std::runtime_error("device grid-node map storage is empty");
    auto storage =
        std::static_pointer_cast<DeviceGridNodeMapStorage>(map.storage);
    if (storage->nodes.size() != map.size)
        throw std::runtime_error("device grid-node map size mismatch");
    return {map.size == 0 ? nullptr
        : thrust::raw_pointer_cast(storage->nodes.data())};
}

std::vector<AssemblyGridNode> copyDeviceGridNodeMapToHost(
    const DeviceGridNodeMap& map)
{
    if (!map.storage)
        return {};
    const auto storage =
        std::static_pointer_cast<DeviceGridNodeMapStorage>(map.storage);
    if (storage->nodes.size() != map.size)
        throw std::runtime_error("device grid-node map size mismatch");
    std::vector<AssemblyGridNode> result(map.size);
    thrust::copy(storage->nodes.begin(), storage->nodes.end(), result.begin());
    return result;
}

DeviceGridNodeMap makeDeviceGridNodeMapFromDeviceNodes(
    const AssemblyGridNode* nodes, std::size_t count)
{
    if (count != 0 && !nodes)
        throw std::runtime_error("device grid-node source pointer is null");
    DeviceGridNodeMap result;
    result.size = count;
    auto storage = std::make_shared<DeviceGridNodeMapStorage>();
    storage->nodes.resize(count);
    if (count != 0)
        checkCuda(cudaMemcpy(thrust::raw_pointer_cast(storage->nodes.data()),
                      nodes, count * sizeof(AssemblyGridNode),
                      cudaMemcpyDeviceToDevice),
            "device grid-node map device-to-device copy");
    result.storage = storage;
    return result;
}

bool persistentSolidParticleStateValid(
    const PersistentSolidParticleState3D& state, std::size_t particle_count)
{
    if (!state.storage)
        return false;
    const auto storage =
        std::static_pointer_cast<PersistentSolidParticleStateStorage>(
            state.storage);
    return storage->valid && storage->rest_x.size() == particle_count;
}

void initializePersistentSolidParticleState3D(
    PersistentSolidParticleState3D& state,
    const SolidParticleStateInput3D& input,
    SolidParticleStateStats* stats, bool force_upload)
{
    const std::size_t count = input.rest_x.size();
    const bool sizes_match = input.current_x.size() == count &&
        input.velocity.size() == count && input.affine.size() == count &&
        input.grad_velocity.size() == count && input.f.size() == count &&
        input.j.size() == count && input.mass.size() == count &&
        input.volume.size() == count && input.lambda.size() == count;
    if (!sizes_match)
        throw std::runtime_error("solid particle owner input size mismatch");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    if (!state.storage)
        state.storage = std::make_shared<PersistentSolidParticleStateStorage>();
    auto& storage = *std::static_pointer_cast<
        PersistentSolidParticleStateStorage>(state.storage);
    SolidParticleStateStats local;
    local.particles = count;
    if (!force_upload && storage.valid && storage.rest_x.size() == count) {
        local.reused_input = true;
        local.total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - total_begin).count();
        if (stats) *stats = local;
        return;
    }
    const auto upload_begin = std::chrono::high_resolution_clock::now();
    storage.rest_x = input.rest_x;
    storage.current_x = input.current_x;
    storage.velocity = input.velocity;
    storage.affine = input.affine;
    storage.grad_velocity = input.grad_velocity;
    storage.f = input.f;
    storage.j = input.j;
    storage.mass = input.mass;
    storage.volume = input.volume;
    storage.lambda = input.lambda;
    if (!storage.p2g_node_map)
        storage.p2g_node_map =
            std::make_shared<DeviceGridNodeMapStorage>();
    storage.p2g_node_map->nodes.clear();
    storage.p2g_host_topology.clear();
    storage.p2g_host_velocity.clear();
    storage.p2g_device_topology_valid = false;
    storage.missing_velocity_nodes.resize(1);
    storage.missing_pressure_nodes.resize(1);
    checkCuda(cudaDeviceSynchronize(), "solid particle owner upload synchronize");
    const auto upload_end = std::chrono::high_resolution_clock::now();
    storage.valid = true;
    local.upload_ms = std::chrono::duration<double, std::milli>(
        upload_end - upload_begin).count();
    local.total_ms = std::chrono::duration<double, std::milli>(
        upload_end - total_begin).count();
    if (stats) *stats = local;
}

void invalidatePersistentSolidParticleState3D(
    PersistentSolidParticleState3D& state)
{
    if (state.storage)
        std::static_pointer_cast<PersistentSolidParticleStateStorage>(
            state.storage)->valid = false;
}

SolidParticleStateDeviceView getPersistentSolidParticleStateDeviceView(
    const PersistentSolidParticleState3D& state)
{
    if (!state.storage)
        return {};
    const auto storage =
        std::static_pointer_cast<PersistentSolidParticleStateStorage>(
            state.storage);
    if (!storage->valid)
        return {};
    const std::size_t count = storage->rest_x.size();
    return {count ? thrust::raw_pointer_cast(storage->rest_x.data()) : nullptr,
        count ? thrust::raw_pointer_cast(storage->current_x.data()) : nullptr,
        count ? thrust::raw_pointer_cast(storage->velocity.data()) : nullptr,
        count ? thrust::raw_pointer_cast(storage->affine.data()) : nullptr,
        count ? thrust::raw_pointer_cast(storage->grad_velocity.data()) : nullptr,
        count ? thrust::raw_pointer_cast(storage->f.data()) : nullptr,
        count ? thrust::raw_pointer_cast(storage->j.data()) : nullptr,
        count ? thrust::raw_pointer_cast(storage->mass.data()) : nullptr,
        count ? thrust::raw_pointer_cast(storage->volume.data()) : nullptr,
        count ? thrust::raw_pointer_cast(storage->lambda.data()) : nullptr,
        count};
}

static void updatePersistentSolidParticlesQ2Impl(
    PersistentSolidParticleState3D& state,
    const DeviceGridNodeMap& velocity_nodes,
    const DeviceGridNodeMap& pressure_nodes,
    const DeviceVector& device_velocity,
    const DeviceVector& device_pressure, double dx, double dt,
    double apic_rpic_ratio, double d_inverse,
    SolidParticleStateStats* stats, bool grid_only)
{
    if (!(dx > 0.0) || !(dt > 0.0) || !state.storage)
        throw std::runtime_error("invalid solid particle Q2 update input");
    auto& storage = *std::static_pointer_cast<
        PersistentSolidParticleStateStorage>(state.storage);
    if (!storage.valid || (!grid_only &&
        (device_velocity.size != velocity_nodes.size * 3 ||
         device_pressure.size != pressure_nodes.size)))
        throw std::runtime_error("solid particle Q2 resident size mismatch");
    const std::size_t count = storage.rest_x.size();
    const auto total_begin = std::chrono::high_resolution_clock::now();
    thrust::fill(storage.missing_velocity_nodes.begin(),
        storage.missing_velocity_nodes.end(), 0);
    thrust::fill(storage.missing_pressure_nodes.begin(),
        storage.missing_pressure_nodes.end(), 0);
    const auto velocity_map = getDeviceGridNodeMapRawView(velocity_nodes);
    const auto pressure_map = grid_only ? DeviceGridNodeMapRawView{} :
        getDeviceGridNodeMapRawView(pressure_nodes);
    const auto velocity = grid_only ? DeviceVectorRawView{} :
        getDeviceVectorRawView(device_velocity);
    const auto pressure = grid_only ? DeviceVectorRawView{} :
        getDeviceVectorRawView(device_pressure);
    if (count != 0) {
        constexpr int block_size = 256;
        const int blocks = static_cast<int>((count + block_size - 1) /
            block_size);
        updateSolidParticlesQ2Kernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.current_x.data()),
            thrust::raw_pointer_cast(storage.velocity.data()),
            thrust::raw_pointer_cast(storage.affine.data()),
            thrust::raw_pointer_cast(storage.grad_velocity.data()),
            thrust::raw_pointer_cast(storage.f.data()),
            thrust::raw_pointer_cast(storage.j.data()),
            thrust::raw_pointer_cast(storage.lambda.data()),
            thrust::raw_pointer_cast(storage.rest_x.data()),
            static_cast<int>(count), velocity_map.nodes,
            static_cast<int>(velocity_nodes.size), pressure_map.nodes,
            static_cast<int>(pressure_nodes.size), velocity.values,
            static_cast<int>(device_velocity.size), pressure.values,
            static_cast<int>(device_pressure.size), dx, dt, apic_rpic_ratio,
            d_inverse, grid_only,
            thrust::raw_pointer_cast(storage.missing_velocity_nodes.data()),
            thrust::raw_pointer_cast(storage.missing_pressure_nodes.data()));
        checkCuda(cudaGetLastError(), "solid particle Q2 update launch");
    }
    SolidParticleStateStats local;
    local.particles = count;
    local.velocity_nodes = velocity_nodes.size;
    local.pressure_nodes = pressure_nodes.size;
    if (stats) {
        checkCuda(cudaDeviceSynchronize(),
            "solid particle Q2 diagnostic synchronize");
        thrust::copy_n(storage.missing_velocity_nodes.begin(), 1,
            &local.missing_velocity_nodes);
        thrust::copy_n(storage.missing_pressure_nodes.begin(), 1,
            &local.missing_pressure_nodes);
        local.g2p_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - total_begin).count();
        local.total_ms = local.g2p_ms;
        *stats = local;
    }
}

void updatePersistentSolidParticlesQ2FromDevice3D(
    PersistentSolidParticleState3D& state,
    const DeviceGridNodeMap& velocity_nodes,
    const DeviceGridNodeMap& pressure_nodes,
    const DeviceVector& device_velocity,
    const DeviceVector& device_pressure, double dx, double dt,
    double apic_rpic_ratio, double d_inverse, SolidParticleStateStats* stats)
{
    updatePersistentSolidParticlesQ2Impl(state, velocity_nodes, pressure_nodes,
        device_velocity, device_pressure, dx, dt, apic_rpic_ratio, d_inverse,
        stats, false);
}

void updatePersistentSolidParticlesQ2FromGrid3D(
    PersistentSolidParticleState3D& state,
    const DeviceGridNodeMap& velocity_nodes, double dx, double dt,
    double apic_rpic_ratio, double d_inverse, SolidParticleStateStats* stats)
{
    updatePersistentSolidParticlesQ2Impl(state, velocity_nodes, {}, {}, {},
        dx, dt, apic_rpic_ratio, d_inverse, stats, true);
}

void copyPersistentSolidParticleStateToHost(
    const PersistentSolidParticleState3D& state,
    SolidParticleStateInput3D& output)
{
    if (!state.storage)
        throw std::runtime_error("solid particle owner is empty");
    const auto storage =
        std::static_pointer_cast<PersistentSolidParticleStateStorage>(
            state.storage);
    if (!storage->valid)
        throw std::runtime_error("solid particle owner is invalid");
    output.rest_x.assign(storage->rest_x.begin(), storage->rest_x.end());
    output.current_x.assign(storage->current_x.begin(), storage->current_x.end());
    output.velocity.assign(storage->velocity.begin(), storage->velocity.end());
    output.affine.assign(storage->affine.begin(), storage->affine.end());
    output.grad_velocity.assign(storage->grad_velocity.begin(),
        storage->grad_velocity.end());
    output.f.assign(storage->f.begin(), storage->f.end());
    output.j.assign(storage->j.begin(), storage->j.end());
    output.mass.assign(storage->mass.begin(), storage->mass.end());
    output.volume.assign(storage->volume.begin(), storage->volume.end());
    output.lambda.assign(storage->lambda.begin(), storage->lambda.end());
}

void copyPersistentSolidDynamicStateToHost(
    const PersistentSolidParticleState3D& state,
    SolidParticleStateInput3D& output)
{
    if (!state.storage)
        throw std::runtime_error("solid particle owner is empty");
    const auto storage =
        std::static_pointer_cast<PersistentSolidParticleStateStorage>(
            state.storage);
    if (!storage->valid)
        throw std::runtime_error("solid particle owner is invalid");
    output = {};
    output.current_x.assign(storage->current_x.begin(), storage->current_x.end());
    output.velocity.assign(storage->velocity.begin(), storage->velocity.end());
    output.affine.assign(storage->affine.begin(), storage->affine.end());
    output.grad_velocity.assign(storage->grad_velocity.begin(),
        storage->grad_velocity.end());
    output.f.assign(storage->f.begin(), storage->f.end());
    output.j.assign(storage->j.begin(), storage->j.end());
}

void copyPersistentSolidCurrentXToHost(
    const PersistentSolidParticleState3D& state,
    std::vector<Vec3d>& current_x)
{
    if (!state.storage)
        throw std::runtime_error("solid particle owner is empty");
    const auto storage =
        std::static_pointer_cast<PersistentSolidParticleStateStorage>(
            state.storage);
    if (!storage->valid)
        throw std::runtime_error("solid particle owner is invalid");
    current_x.assign(storage->current_x.begin(), storage->current_x.end());
}

void buildPersistentSolidVelocityGridQ2ToHost(
    PersistentSolidParticleState3D& state, double dx,
    std::vector<AssemblyGridNode>& nodes, SolidVelocityGridStats* stats,
    bool download_dynamic_velocity,
    const std::vector<AssemblyGridNode>* legacy_topology_seed,
    bool assign_canonical_indices)
{
    if (!(dx > 0.0) || !state.storage)
        throw std::runtime_error("invalid solid particle Q2 P2G input");
    auto& storage = *std::static_pointer_cast<
        PersistentSolidParticleStateStorage>(state.storage);
    if (!storage.valid)
        throw std::runtime_error("solid particle owner is invalid");
    const auto total_begin = std::chrono::high_resolution_clock::now();
    const std::size_t particle_count = storage.rest_x.size();
    const std::size_t contribution_count = particle_count * 27;
    storage.p2g_candidate_keys.resize(contribution_count);
    storage.p2g_candidate_mass.resize(contribution_count);
    storage.p2g_candidate_momentum.resize(contribution_count);
    if (particle_count != 0) {
        constexpr int block_size = 256;
        const int blocks = static_cast<int>((particle_count + block_size - 1) /
            block_size);
        generateSolidQ2P2gContributionsKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.rest_x.data()),
            thrust::raw_pointer_cast(storage.velocity.data()),
            thrust::raw_pointer_cast(storage.affine.data()),
            thrust::raw_pointer_cast(storage.mass.data()),
            static_cast<int>(particle_count), dx,
            thrust::raw_pointer_cast(storage.p2g_candidate_keys.data()),
            thrust::raw_pointer_cast(storage.p2g_candidate_mass.data()),
            thrust::raw_pointer_cast(storage.p2g_candidate_momentum.data()));
        checkCuda(cudaGetLastError(), "solid Q2 P2G contribution launch");
    }
    auto candidate_values = thrust::make_zip_iterator(thrust::make_tuple(
        storage.p2g_candidate_mass.begin(),
        storage.p2g_candidate_momentum.begin()));
    thrust::stable_sort_by_key(storage.p2g_candidate_keys.begin(),
        storage.p2g_candidate_keys.end(), candidate_values);
    storage.p2g_keys.resize(contribution_count);
    storage.p2g_mass.resize(contribution_count);
    storage.p2g_momentum.resize(contribution_count);
    auto reduced_values = thrust::make_zip_iterator(thrust::make_tuple(
        storage.p2g_mass.begin(), storage.p2g_momentum.begin()));
    const auto reduced_end = thrust::reduce_by_key(
        storage.p2g_candidate_keys.begin(), storage.p2g_candidate_keys.end(),
        candidate_values, storage.p2g_keys.begin(), reduced_values,
        thrust::equal_to<GridKey>(), SolidP2gTupleAdd{});
    const std::size_t node_count = static_cast<std::size_t>(
        reduced_end.first - storage.p2g_keys.begin());
    storage.p2g_keys.resize(node_count);
    storage.p2g_mass.resize(node_count);
    storage.p2g_momentum.resize(node_count);
    if (!storage.p2g_node_map)
        storage.p2g_node_map =
            std::make_shared<DeviceGridNodeMapStorage>();
    bool seeded_legacy_topology = false;
    if (!storage.p2g_device_topology_valid && legacy_topology_seed &&
        legacy_topology_seed->size() == node_count) {
        std::vector<AssemblyGridNode> sorted_seed(*legacy_topology_seed);
        std::sort(sorted_seed.begin(), sorted_seed.end(),
            [](const AssemblyGridNode& a, const AssemblyGridNode& b) {
                return a.key < b.key;
            });
        bool valid_indices = true;
        std::vector<unsigned char> seen_indices(node_count, 0);
        for (std::size_t i = 0; i < sorted_seed.size(); ++i) {
            const int idx = sorted_seed[i].idx;
            valid_indices = valid_indices && idx >= 0 &&
                static_cast<std::size_t>(idx) < node_count &&
                (i == 0 || sorted_seed[i - 1].key != sorted_seed[i].key);
            if (idx >= 0 && static_cast<std::size_t>(idx) < node_count) {
                valid_indices = valid_indices && !seen_indices[idx];
                seen_indices[idx] = 1;
            }
        }
        if (valid_indices) {
            storage.p2g_node_map->nodes.assign(
                sorted_seed.begin(), sorted_seed.end());
            seeded_legacy_topology = thrust::equal(
                storage.p2g_keys.begin(), storage.p2g_keys.end(),
                storage.p2g_node_map->nodes.begin(),
                SolidGridKeyMatchesNode{});
            storage.p2g_device_topology_valid = seeded_legacy_topology;
        }
    }
    const bool reuse_topology =
        storage.p2g_device_topology_valid &&
        storage.p2g_node_map->nodes.size() == node_count;
    const bool assigned_canonical_topology =
        !reuse_topology && assign_canonical_indices;
    storage.p2g_node_map->nodes.resize(node_count);
    storage.p2g_velocity.resize(node_count);
    if (node_count != 0) {
        constexpr int block_size = 256;
        const int blocks = static_cast<int>((node_count + block_size - 1) /
            block_size);
        normalizeSolidQ2P2gNodesKernel<<<blocks, block_size>>>(
            thrust::raw_pointer_cast(storage.p2g_keys.data()),
            thrust::raw_pointer_cast(storage.p2g_mass.data()),
            thrust::raw_pointer_cast(storage.p2g_momentum.data()),
            static_cast<int>(node_count),
            thrust::raw_pointer_cast(storage.p2g_node_map->nodes.data()),
            thrust::raw_pointer_cast(storage.p2g_velocity.data()),
            reuse_topology ? 1 : (assigned_canonical_topology ? 2 : 0));
        checkCuda(cudaGetLastError(), "solid Q2 P2G normalize launch");
    }
    checkCuda(cudaDeviceSynchronize(), "solid Q2 P2G synchronize");
    if (assigned_canonical_topology)
        storage.p2g_device_topology_valid = true;
    const auto p2g_end = std::chrono::high_resolution_clock::now();
    std::size_t topology_download_bytes = 0;
    std::size_t velocity_download_bytes = 0;
    const bool device_topology_ready =
        reuse_topology || assigned_canonical_topology;
    const bool materialize_full_topology = !device_topology_ready ||
        (download_dynamic_velocity &&
         storage.p2g_host_topology.size() != node_count);
    if (materialize_full_topology) {
        storage.p2g_host_topology.resize(node_count);
        if (node_count != 0)
            thrust::copy(storage.p2g_node_map->nodes.begin(),
                storage.p2g_node_map->nodes.end(),
                storage.p2g_host_topology.begin());
        topology_download_bytes = node_count * sizeof(AssemblyGridNode);
        storage.p2g_device_topology_valid = true;
    } else if (download_dynamic_velocity) {
        storage.p2g_host_velocity.resize(node_count);
        if (node_count != 0)
            thrust::copy(storage.p2g_velocity.begin(),
                storage.p2g_velocity.end(),
                storage.p2g_host_velocity.begin());
        for (std::size_t i = 0; i < node_count; ++i)
            storage.p2g_host_topology[i].v = storage.p2g_host_velocity[i];
        velocity_download_bytes = node_count * sizeof(Vec3d);
    }
    if (download_dynamic_velocity || materialize_full_topology)
        nodes = storage.p2g_host_topology;
    else
        nodes.clear();
    const auto download_end = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->particles = particle_count;
        stats->nodes = node_count;
        stats->topology_download_bytes = topology_download_bytes;
        stats->velocity_download_bytes = velocity_download_bytes;
        stats->host_download_bytes =
            topology_download_bytes + velocity_download_bytes;
        stats->reused_static_topology = reuse_topology;
        stats->seeded_legacy_topology = seeded_legacy_topology;
        stats->assigned_canonical_topology = assigned_canonical_topology;
        stats->p2g_ms = std::chrono::duration<double, std::milli>(
            p2g_end - total_begin).count();
        stats->download_ms = std::chrono::duration<double, std::milli>(
            download_end - p2g_end).count();
        stats->total_ms = std::chrono::duration<double, std::milli>(
            download_end - total_begin).count();
    }
}

DeviceGridNodeMap getPersistentSolidVelocityGridQ2DeviceMap(
    const PersistentSolidParticleState3D& state)
{
    if (!state.storage)
        return {};
    const auto storage =
        std::static_pointer_cast<PersistentSolidParticleStateStorage>(
            state.storage);
    if (!storage->valid || !storage->p2g_node_map)
        return {};
    DeviceGridNodeMap result;
    result.size = storage->p2g_node_map->nodes.size();
    result.storage = storage->p2g_node_map;
    return result;
}

void reducePersistentSolidParticleState3D(
    const PersistentSolidParticleState3D& state,
    SolidParticleReductionStats& stats)
{
    if (!state.storage)
        throw std::runtime_error("solid particle owner is empty");
    const auto storage =
        std::static_pointer_cast<PersistentSolidParticleStateStorage>(
            state.storage);
    if (!storage->valid)
        throw std::runtime_error("solid particle owner is invalid");
    const auto begin = std::chrono::high_resolution_clock::now();
    const std::size_t count = storage->rest_x.size();
    stats = {};
    stats.particles = count;
    if (count != 0) {
        const double infinity = std::numeric_limits<double>::infinity();
        const SolidParticleReductionValue init{
            infinity, infinity, infinity, -infinity, -infinity, -infinity, 0.0};
        const SolidParticleReductionValue reduced = thrust::transform_reduce(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(static_cast<int>(count)),
            SolidParticleReductionTransform{
                thrust::raw_pointer_cast(storage->rest_x.data()),
                thrust::raw_pointer_cast(storage->velocity.data())},
            init, SolidParticleReductionCombine{});
        stats.min_corner = {reduced.min_x, reduced.min_y, reduced.min_z};
        stats.max_corner = {reduced.max_x, reduced.max_y, reduced.max_z};
        stats.max_speed = std::sqrt(reduced.max_speed_squared);
    }
    stats.reduction_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - begin).count();
}

DeviceVector sliceDeviceVector(const DeviceVector& vector,
    std::size_t offset, std::size_t size)
{
    const DeviceVectorStorage& input = storageFor(vector);
    if (offset > vector.size || size > vector.size - offset)
        throw std::runtime_error("device vector slice is out of range");
    DeviceVector result;
    result.size = size;
    auto output = std::make_shared<DeviceVectorStorage>();
    output->values.resize(size);
    if (size != 0)
        thrust::copy_n(input.values.begin() + static_cast<std::ptrdiff_t>(offset),
            static_cast<std::ptrdiff_t>(size), output->values.begin());
    result.storage = output;
    return result;
}

std::vector<double> copyDeviceVectorToHost(const DeviceVector& vector)
{
    const DeviceVectorStorage& storage = storageFor(vector);
    std::vector<double> result(vector.size);
    thrust::copy(storage.values.begin(), storage.values.end(), result.begin());
    return result;
}

DeviceCsrMatrix transposeDeviceCsrMatrix(
    const DeviceCsrMatrix& matrix, double scale)
{
    validateDeviceCsr(matrix);
    const DeviceCsrStorage& input = storageFor(matrix);
    DeviceCsrMatrix result;
    result.rows = matrix.cols;
    result.cols = matrix.rows;
    result.nonzeros = matrix.nonzeros;
    auto output = std::make_shared<DeviceCsrStorage>();
    output->row_offsets.assign(static_cast<std::size_t>(result.rows + 1), 0);
    output->col_indices.resize(result.nonzeros);
    output->values.resize(result.nonzeros);
    result.storage = output;
    if (matrix.nonzeros == 0) {
        validateDeviceCsr(result);
        return result;
    }
#ifdef ZIRAN_ENABLE_CUSPARSE
    cusparseHandle_t handle = nullptr;
    void* buffer = nullptr;
    checkCusparse(cusparseCreate(&handle), "cusparseCreate transpose");
    try {
        size_t buffer_size = 0;
        checkCusparse(cusparseCsr2cscEx2_bufferSize(handle, matrix.rows,
                          matrix.cols, static_cast<int>(matrix.nonzeros),
                          thrust::raw_pointer_cast(input.values.data()),
                          thrust::raw_pointer_cast(input.row_offsets.data()),
                          thrust::raw_pointer_cast(input.col_indices.data()),
                          thrust::raw_pointer_cast(output->values.data()),
                          thrust::raw_pointer_cast(output->row_offsets.data()),
                          thrust::raw_pointer_cast(output->col_indices.data()),
                          CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                          CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
                          &buffer_size),
            "cusparseCsr2cscEx2_bufferSize");
        if (buffer_size != 0)
            checkCuda(cudaMalloc(&buffer, buffer_size),
                "cudaMalloc transpose buffer");
        checkCusparse(cusparseCsr2cscEx2(handle, matrix.rows, matrix.cols,
                          static_cast<int>(matrix.nonzeros),
                          thrust::raw_pointer_cast(input.values.data()),
                          thrust::raw_pointer_cast(input.row_offsets.data()),
                          thrust::raw_pointer_cast(input.col_indices.data()),
                          thrust::raw_pointer_cast(output->values.data()),
                          thrust::raw_pointer_cast(output->row_offsets.data()),
                          thrust::raw_pointer_cast(output->col_indices.data()),
                          CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                          CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
                          buffer),
            "cusparseCsr2cscEx2");
        if (scale != 1.0)
            thrust::transform(output->values.begin(), output->values.end(),
                output->values.begin(), ScaleValue{scale});
        checkCuda(cudaDeviceSynchronize(), "device CSR transpose synchronize");
        if (buffer)
            cudaFree(buffer);
        cusparseDestroy(handle);
    } catch (...) {
        if (buffer)
            cudaFree(buffer);
        if (handle)
            cusparseDestroy(handle);
        throw;
    }
#else
    throw std::runtime_error("device CSR transpose requires cuSPARSE");
#endif
    validateDeviceCsr(result);
    return result;
}

DeviceVector multiplyDeviceCsrVector(
    const DeviceCsrMatrix& matrix, const DeviceVector& vector)
{
    validateDeviceCsr(matrix);
    const DeviceCsrStorage& matrix_storage = storageFor(matrix);
    const DeviceVectorStorage& vector_storage = storageFor(vector);
    if (matrix.cols != static_cast<int>(vector.size))
        throw std::runtime_error("device CSR/vector dimension mismatch");
    DeviceVector result;
    result.size = static_cast<std::size_t>(matrix.rows);
    auto output = std::make_shared<DeviceVectorStorage>();
    output->values.assign(result.size, 0.0);
    result.storage = output;
    if (matrix.rows != 0) {
        constexpr int block_size = 256;
        const int block_count = (matrix.rows + block_size - 1) / block_size;
        csrSpmvKernel<<<block_count, block_size>>>(matrix.rows,
            thrust::raw_pointer_cast(matrix_storage.row_offsets.data()),
            thrust::raw_pointer_cast(matrix_storage.col_indices.data()),
            thrust::raw_pointer_cast(matrix_storage.values.data()),
            thrust::raw_pointer_cast(vector_storage.values.data()),
            thrust::raw_pointer_cast(output->values.data()));
        checkCuda(cudaGetLastError(), "device CSR/vector SpMV launch");
    }
    return result;
}

DeviceVector addDeviceVectors(
    const DeviceVector& left, double right_scale, const DeviceVector& right)
{
    const DeviceVectorStorage& left_storage = storageFor(left);
    const DeviceVectorStorage& right_storage = storageFor(right);
    if (left.size != right.size)
        throw std::runtime_error("device vector add size mismatch");
    DeviceVector result;
    result.size = left.size;
    auto output = std::make_shared<DeviceVectorStorage>();
    output->values.resize(result.size);
    thrust::transform(left_storage.values.begin(), left_storage.values.end(),
        right_storage.values.begin(), output->values.begin(),
        AddScaledValue{right_scale});
    result.storage = output;
    return result;
}

DeviceVector concatenateDeviceVectors(const std::vector<DeviceVector>& blocks)
{
    DeviceVector result;
    for (const DeviceVector& block : blocks) {
        storageFor(block);
        result.size += block.size;
    }
    auto output = std::make_shared<DeviceVectorStorage>();
    output->values.resize(result.size);
    std::size_t offset = 0;
    for (const DeviceVector& block : blocks) {
        const DeviceVectorStorage& input = storageFor(block);
        if (block.size != 0)
            thrust::copy(input.values.begin(), input.values.end(),
                output->values.begin() + static_cast<std::ptrdiff_t>(offset));
        offset += block.size;
    }
    result.storage = output;
    return result;
}

DeviceVector computePressureVelocityDevice(
    PersistentVelocityAlgebra& workspace,
    const DeviceCsrMatrix& m_inv, const DeviceCsrMatrix& g,
    const DeviceVector& a, const DeviceVector& pressure)
{
    validateDeviceCsr(m_inv);
    validateDeviceCsr(g);
    const DeviceCsrStorage& minv_storage = storageFor(m_inv);
    const DeviceCsrStorage& g_storage = storageFor(g);
    const DeviceVectorStorage& a_storage = storageFor(a);
    const DeviceVectorStorage& pressure_storage = storageFor(pressure);
    if (g.cols != static_cast<int>(pressure.size) ||
        g.rows != static_cast<int>(a.size) ||
        m_inv.cols != g.rows)
        throw std::runtime_error("device pressure velocity algebra dimension mismatch");

    auto& storage = velocityAlgebraStorage(workspace);
    DeviceVector gradient = resizePersistentVector(
        storage.pressure_gradient, static_cast<std::size_t>(g.rows));
    DeviceVector rhs = resizePersistentVector(storage.rhs, a.size);
    DeviceVector velocity = resizePersistentVector(
        storage.velocity, static_cast<std::size_t>(m_inv.rows));
    DeviceVectorStorage& gradient_storage =
        *std::static_pointer_cast<DeviceVectorStorage>(gradient.storage);
    DeviceVectorStorage& rhs_storage =
        *std::static_pointer_cast<DeviceVectorStorage>(rhs.storage);
    DeviceVectorStorage& velocity_storage =
        *std::static_pointer_cast<DeviceVectorStorage>(velocity.storage);

    constexpr int block_size = 256;
    if (g.rows != 0) {
        const int block_count = (g.rows + block_size - 1) / block_size;
        csrSpmvKernel<<<block_count, block_size>>>(g.rows,
            thrust::raw_pointer_cast(g_storage.row_offsets.data()),
            thrust::raw_pointer_cast(g_storage.col_indices.data()),
            thrust::raw_pointer_cast(g_storage.values.data()),
            thrust::raw_pointer_cast(pressure_storage.values.data()),
            thrust::raw_pointer_cast(gradient_storage.values.data()));
        checkCuda(cudaGetLastError(), "persistent pressure gradient launch");
        subtractVectorKernel<<<block_count, block_size>>>(
            thrust::raw_pointer_cast(a_storage.values.data()),
            thrust::raw_pointer_cast(gradient_storage.values.data()), g.rows,
            thrust::raw_pointer_cast(rhs_storage.values.data()));
        checkCuda(cudaGetLastError(), "persistent pressure rhs launch");
    }
    if (m_inv.rows != 0) {
        const int block_count = (m_inv.rows + block_size - 1) / block_size;
        csrSpmvKernel<<<block_count, block_size>>>(m_inv.rows,
            thrust::raw_pointer_cast(minv_storage.row_offsets.data()),
            thrust::raw_pointer_cast(minv_storage.col_indices.data()),
            thrust::raw_pointer_cast(minv_storage.values.data()),
            thrust::raw_pointer_cast(rhs_storage.values.data()),
            thrust::raw_pointer_cast(velocity_storage.values.data()));
        checkCuda(cudaGetLastError(), "persistent pressure velocity launch");
    }
    return velocity;
}

DeviceCsrMatrix assembleDeviceCsrBlocks(int rows, int cols,
    const std::vector<DeviceCsrBlock>& blocks)
{
    std::size_t total_entries = 0;
    for (const DeviceCsrBlock& block : blocks) {
        validateDeviceCsr(block.matrix);
        if (block.row_offset < 0 || block.col_offset < 0 ||
            block.row_offset + block.matrix.rows > rows ||
            block.col_offset + block.matrix.cols > cols)
            throw std::runtime_error("device CSR block is outside destination");
        total_entries += block.matrix.nonzeros;
    }
    thrust::device_vector<CooEntry> entries(total_entries);
    std::size_t output_offset = 0;
    constexpr int block_size = 256;
    for (const DeviceCsrBlock& block : blocks) {
        const DeviceCsrStorage& input = storageFor(block.matrix);
        if (block.matrix.rows != 0 && block.matrix.nonzeros != 0) {
            const int block_count =
                (block.matrix.rows + block_size - 1) / block_size;
            csrBlockToCooKernel<<<block_count, block_size>>>(block.matrix.rows,
                thrust::raw_pointer_cast(input.row_offsets.data()),
                thrust::raw_pointer_cast(input.col_indices.data()),
                thrust::raw_pointer_cast(input.values.data()), output_offset,
                block.row_offset, block.col_offset, block.scale,
                thrust::raw_pointer_cast(entries.data()));
            checkCuda(cudaGetLastError(), "csrBlockToCooKernel launch");
        }
        output_offset += block.matrix.nonzeros;
    }
    return finalizeCooToDevice(rows, cols, entries);
}

std::vector<double> makeDeterministicVector(int size, int seed)
{
    if (size < 0)
        throw std::runtime_error("deterministic vector size must be non-negative");

    std::vector<double> result(static_cast<std::size_t>(size));
    unsigned int state = static_cast<unsigned int>(seed) * 747796405u + 2891336453u;
    for (int i = 0; i < size; ++i) {
        state = state * 1664525u + 1013904223u;
        const double unit = static_cast<double>(state & 0x00ffffffu) /
            static_cast<double>(0x00ffffffu);
        result[static_cast<std::size_t>(i)] = 2.0 * unit - 1.0;
    }
    return result;
}

void runCsrSpmvProbe(const CsrMatrix& matrix,
    const std::vector<double>& x, std::vector<double>& y)
{
    validateCsr(matrix);
    if (x.size() != static_cast<std::size_t>(matrix.cols))
        throw std::runtime_error("CSR SpMV input vector size mismatch");

    y.assign(static_cast<std::size_t>(matrix.rows), 0.0);
    if (matrix.rows == 0)
        return;

    thrust::device_vector<int> d_row_offsets(matrix.row_offsets.begin(), matrix.row_offsets.end());
    thrust::device_vector<int> d_col_indices(matrix.col_indices.begin(), matrix.col_indices.end());
    thrust::device_vector<double> d_values(matrix.values.begin(), matrix.values.end());
    thrust::device_vector<double> d_x(x.begin(), x.end());
    thrust::device_vector<double> d_y(static_cast<std::size_t>(matrix.rows), 0.0);

    const int block_size = 256;
    const int block_count = (matrix.rows + block_size - 1) / block_size;
    csrSpmvKernel<<<block_count, block_size>>>(matrix.rows,
        thrust::raw_pointer_cast(d_row_offsets.data()),
        thrust::raw_pointer_cast(d_col_indices.data()),
        thrust::raw_pointer_cast(d_values.data()),
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()));
    checkCuda(cudaGetLastError(), "csrSpmvKernel launch");
    checkCuda(cudaDeviceSynchronize(), "csrSpmvKernel synchronize");

    thrust::copy(d_y.begin(), d_y.end(), y.begin());
}

void runDeviceCsrSpmvProbe(const DeviceCsrMatrix& matrix,
    const std::vector<double>& x, std::vector<double>& y)
{
    validateDeviceCsr(matrix);
    if (x.size() != static_cast<std::size_t>(matrix.cols))
        throw std::runtime_error("device CSR SpMV input vector size mismatch");

    y.assign(static_cast<std::size_t>(matrix.rows), 0.0);
    if (matrix.rows == 0)
        return;

    const DeviceCsrStorage& storage = storageFor(matrix);
    thrust::device_vector<double> d_x(x.begin(), x.end());
    thrust::device_vector<double> d_y(static_cast<std::size_t>(matrix.rows), 0.0);

    const int block_size = 256;
    const int block_count = (matrix.rows + block_size - 1) / block_size;
    csrSpmvKernel<<<block_count, block_size>>>(matrix.rows,
        thrust::raw_pointer_cast(storage.row_offsets.data()),
        thrust::raw_pointer_cast(storage.col_indices.data()),
        thrust::raw_pointer_cast(storage.values.data()),
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()));
    checkCuda(cudaGetLastError(), "csrSpmvKernel device CSR launch");
    checkCuda(cudaDeviceSynchronize(), "csrSpmvKernel device CSR synchronize");

    thrust::copy(d_y.begin(), d_y.end(), y.begin());
}

OperatorCompareStats compareVectors(const std::string& label,
    const CsrMatrix& matrix, const std::vector<double>& cpu_y,
    const std::vector<double>& gpu_y)
{
    validateCsr(matrix);
    if (cpu_y.size() != gpu_y.size())
        throw std::runtime_error("compareVectors input size mismatch");
    if (cpu_y.size() != static_cast<std::size_t>(matrix.rows))
        throw std::runtime_error("compareVectors row count mismatch");

    OperatorCompareStats stats;
    stats.label = label;
    stats.rows = matrix.rows;
    stats.cols = matrix.cols;
    stats.nonzeros = matrix.values.size();

    double sum_abs = 0.0;
    double reference_norm_inf = 0.0;
    for (std::size_t i = 0; i < cpu_y.size(); ++i) {
        const double err = std::abs(cpu_y[i] - gpu_y[i]);
        stats.max_abs_error = std::max(stats.max_abs_error, err);
        reference_norm_inf = std::max(reference_norm_inf, std::abs(cpu_y[i]));
        sum_abs += err;
    }
    stats.max_rel_error = stats.max_abs_error / std::max(reference_norm_inf, 1e-30);
    stats.mean_abs_error = cpu_y.empty() ? 0.0 : sum_abs / static_cast<double>(cpu_y.size());
    return stats;
}

OperatorCompareStats runSparseMatrixProbe(const std::string& label,
    const Eigen::SparseMatrix<double>& matrix, int vector_seed)
{
    const CsrMatrix csr = makeCsrFromEigen(matrix);
    const std::vector<double> x = makeDeterministicVector(csr.cols, vector_seed);
    const std::vector<double> cpu_y = eigenProduct(matrix, x);
    std::vector<double> gpu_y;
    runCsrSpmvProbe(csr, x, gpu_y);
    return compareVectors(label, csr, cpu_y, gpu_y);
}

OperatorCompareStats compareSparseMatrixToReference(const std::string& label,
    const Eigen::SparseMatrix<double>& probe_matrix,
    const Eigen::SparseMatrix<double>& reference_matrix, int vector_seed)
{
    if (probe_matrix.rows() != reference_matrix.rows() ||
        probe_matrix.cols() != reference_matrix.cols())
        throw std::runtime_error("sparse matrix compare dimension mismatch");

    const CsrMatrix csr = makeCsrFromEigen(probe_matrix);
    const std::vector<double> x = makeDeterministicVector(csr.cols, vector_seed);
    const std::vector<double> cpu_y = eigenProduct(reference_matrix, x);
    std::vector<double> probe_y;
    runCsrSpmvProbe(csr, x, probe_y);
    return compareVectors(label, csr, cpu_y, probe_y);
}

OperatorCompareStats compareCsrMatrices(const std::string& label,
    const CsrMatrix& probe_matrix, const CsrMatrix& reference_matrix)
{
    validateCsr(probe_matrix);
    validateCsr(reference_matrix);
    if (probe_matrix.rows != reference_matrix.rows ||
        probe_matrix.cols != reference_matrix.cols)
        throw std::runtime_error("CSR compare dimension mismatch");

    OperatorCompareStats stats;
    stats.label = label;
    stats.rows = probe_matrix.rows;
    stats.cols = probe_matrix.cols;
    stats.nonzeros = probe_matrix.values.size();

    double sum_abs = 0.0;
    double reference_norm_inf = 0.0;
    std::size_t coefficient_count = 0;
    for (int row = 0; row < probe_matrix.rows; ++row) {
        int probe = probe_matrix.row_offsets[static_cast<std::size_t>(row)];
        const int probe_end =
            probe_matrix.row_offsets[static_cast<std::size_t>(row + 1)];
        int reference =
            reference_matrix.row_offsets[static_cast<std::size_t>(row)];
        const int reference_end =
            reference_matrix.row_offsets[static_cast<std::size_t>(row + 1)];
        while (probe < probe_end || reference < reference_end) {
            const int probe_col = probe < probe_end
                ? probe_matrix.col_indices[static_cast<std::size_t>(probe)]
                : probe_matrix.cols;
            const int reference_col = reference < reference_end
                ? reference_matrix.col_indices[static_cast<std::size_t>(reference)]
                : reference_matrix.cols;
            double probe_value = 0.0;
            double reference_value = 0.0;
            if (probe_col <= reference_col) {
                probe_value = probe_matrix.values[static_cast<std::size_t>(probe)];
                ++probe;
            }
            if (reference_col <= probe_col) {
                reference_value =
                    reference_matrix.values[static_cast<std::size_t>(reference)];
                ++reference;
            }
            const double error = std::abs(probe_value - reference_value);
            stats.max_abs_error = std::max(stats.max_abs_error, error);
            reference_norm_inf =
                std::max(reference_norm_inf, std::abs(reference_value));
            sum_abs += error;
            ++coefficient_count;
        }
    }
    stats.mean_abs_error = coefficient_count == 0
        ? 0.0
        : sum_abs / static_cast<double>(coefficient_count);
    stats.max_rel_error =
        stats.max_abs_error / std::max(reference_norm_inf, 1e-30);
    return stats;
}

CsrMatrix multiplyCsrMatrices(const CsrMatrix& left, const CsrMatrix& right)
{
    validateCsr(left);
    validateCsr(right);
    if (left.cols != right.rows)
        throw std::runtime_error("CSR multiply dimension mismatch");

    CsrMatrix result;
    result.rows = left.rows;
    result.cols = right.cols;
    result.row_offsets.assign(static_cast<std::size_t>(result.rows + 1), 0);
    if (left.rows == 0 || right.cols == 0 || left.values.empty() || right.values.empty())
        return result;

#ifdef ZIRAN_ENABLE_CUSPARSE
    int* d_left_rows = nullptr;
    int* d_left_cols = nullptr;
    double* d_left_vals = nullptr;
    int* d_right_rows = nullptr;
    int* d_right_cols = nullptr;
    double* d_right_vals = nullptr;
    int* d_result_rows = nullptr;
    int* d_result_cols = nullptr;
    double* d_result_vals = nullptr;
    void* buffer1 = nullptr;
    void* buffer2 = nullptr;

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat_left = nullptr;
    cusparseSpMatDescr_t mat_right = nullptr;
    cusparseSpMatDescr_t mat_result = nullptr;
    cusparseSpGEMMDescr_t spgemm_desc = nullptr;
    checkCusparse(cusparseCreate(&handle), "cusparseCreate");

    auto cleanup = [&]() {
        if (spgemm_desc)
            cusparseSpGEMM_destroyDescr(spgemm_desc);
        if (mat_left)
            cusparseDestroySpMat(mat_left);
        if (mat_right)
            cusparseDestroySpMat(mat_right);
        if (mat_result)
            cusparseDestroySpMat(mat_result);
        if (handle)
            cusparseDestroy(handle);
        if (buffer1)
            cudaFree(buffer1);
        if (buffer2)
            cudaFree(buffer2);
        if (d_left_rows)
            cudaFree(d_left_rows);
        if (d_left_cols)
            cudaFree(d_left_cols);
        if (d_left_vals)
            cudaFree(d_left_vals);
        if (d_right_rows)
            cudaFree(d_right_rows);
        if (d_right_cols)
            cudaFree(d_right_cols);
        if (d_right_vals)
            cudaFree(d_right_vals);
        if (d_result_rows)
            cudaFree(d_result_rows);
        if (d_result_cols)
            cudaFree(d_result_cols);
        if (d_result_vals)
            cudaFree(d_result_vals);
    };

    try {
        cudaMallocArray(d_left_rows, left.row_offsets.size(), "cudaMalloc left rows");
        cudaMallocArray(d_left_cols, left.col_indices.size(), "cudaMalloc left cols");
        cudaMallocArray(d_left_vals, left.values.size(), "cudaMalloc left vals");
        cudaMallocArray(d_right_rows, right.row_offsets.size(), "cudaMalloc right rows");
        cudaMallocArray(d_right_cols, right.col_indices.size(), "cudaMalloc right cols");
        cudaMallocArray(d_right_vals, right.values.size(), "cudaMalloc right vals");
        cudaMallocArray(d_result_rows, static_cast<std::size_t>(left.rows + 1),
            "cudaMalloc result rows");
        cudaCopyToDevice(d_left_rows, left.row_offsets, "cudaMemcpy left rows");
        cudaCopyToDevice(d_left_cols, left.col_indices, "cudaMemcpy left cols");
        cudaCopyToDevice(d_left_vals, left.values, "cudaMemcpy left vals");
        cudaCopyToDevice(d_right_rows, right.row_offsets, "cudaMemcpy right rows");
        cudaCopyToDevice(d_right_cols, right.col_indices, "cudaMemcpy right cols");
        cudaCopyToDevice(d_right_vals, right.values, "cudaMemcpy right vals");
        checkCuda(cudaMemset(d_result_rows, 0,
                      static_cast<std::size_t>(left.rows + 1) * sizeof(int)),
            "cudaMemset result rows");

        checkCusparse(cusparseCreateCsr(&mat_left, left.rows, left.cols,
                          static_cast<int64_t>(left.values.size()),
                          d_left_rows, d_left_cols, d_left_vals,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
            "cusparseCreateCsr left");
        checkCusparse(cusparseCreateCsr(&mat_right, right.rows, right.cols,
                          static_cast<int64_t>(right.values.size()),
                          d_right_rows, d_right_cols, d_right_vals,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
            "cusparseCreateCsr right");
        checkCusparse(cusparseCreateCsr(&mat_result, left.rows, right.cols, 0,
                          d_result_rows, nullptr, nullptr,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
            "cusparseCreateCsr result");

        const double alpha = 1.0;
        const double beta = 0.0;
        constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_DEFAULT;
        checkCusparse(cusparseSpGEMM_createDescr(&spgemm_desc),
            "cusparseSpGEMM_createDescr");

        size_t buffer_size1 = 0;
        checkCusparse(cusparseSpGEMM_workEstimation(handle, op, op, &alpha,
                          mat_left, mat_right, &beta, mat_result, CUDA_R_64F,
                          alg, spgemm_desc, &buffer_size1, nullptr),
            "cusparseSpGEMM_workEstimation size");
        if (buffer_size1)
            checkCuda(cudaMalloc(&buffer1, buffer_size1),
                "cudaMalloc SpGEMM buffer1");
        checkCusparse(cusparseSpGEMM_workEstimation(handle, op, op, &alpha,
                          mat_left, mat_right, &beta, mat_result, CUDA_R_64F,
                          alg, spgemm_desc, &buffer_size1, buffer1),
            "cusparseSpGEMM_workEstimation");

        size_t buffer_size2 = 0;
        checkCusparse(cusparseSpGEMM_compute(handle, op, op, &alpha,
                          mat_left, mat_right, &beta, mat_result, CUDA_R_64F,
                          alg, spgemm_desc, &buffer_size2, nullptr),
            "cusparseSpGEMM_compute size");
        if (buffer_size2)
            checkCuda(cudaMalloc(&buffer2, buffer_size2),
                "cudaMalloc SpGEMM buffer2");
        checkCusparse(cusparseSpGEMM_compute(handle, op, op, &alpha,
                          mat_left, mat_right, &beta, mat_result, CUDA_R_64F,
                          alg, spgemm_desc, &buffer_size2, buffer2),
            "cusparseSpGEMM_compute");
        checkCuda(cudaDeviceSynchronize(), "cusparseSpGEMM compute synchronize");

        int64_t rows = 0;
        int64_t cols = 0;
        int64_t nnz = 0;
        checkCusparse(cusparseSpMatGetSize(mat_result, &rows, &cols, &nnz),
            "cusparseSpMatGetSize");
        result.rows = static_cast<int>(rows);
        result.cols = static_cast<int>(cols);
        result.row_offsets.assign(static_cast<std::size_t>(result.rows + 1), 0);
        result.col_indices.assign(static_cast<std::size_t>(nnz), 0);
        result.values.assign(static_cast<std::size_t>(nnz), 0.0);

        cudaMallocArray(d_result_cols, static_cast<std::size_t>(nnz),
            "cudaMalloc result cols");
        cudaMallocArray(d_result_vals, static_cast<std::size_t>(nnz),
            "cudaMalloc result vals");
        checkCusparse(cusparseCsrSetPointers(mat_result,
                          d_result_rows, d_result_cols, d_result_vals),
            "cusparseCsrSetPointers");
        checkCusparse(cusparseSpGEMM_copy(handle, op, op, &alpha,
                          mat_left, mat_right, &beta, mat_result, CUDA_R_64F,
                          alg, spgemm_desc),
            "cusparseSpGEMM_copy");
        checkCuda(cudaDeviceSynchronize(), "cusparseSpGEMM synchronize");

        checkCusparse(cusparseSpGEMM_destroyDescr(spgemm_desc),
            "cusparseSpGEMM_destroyDescr");
        spgemm_desc = nullptr;
        checkCusparse(cusparseDestroySpMat(mat_left), "cusparseDestroySpMat left");
        mat_left = nullptr;
        checkCusparse(cusparseDestroySpMat(mat_right), "cusparseDestroySpMat right");
        mat_right = nullptr;
        checkCusparse(cusparseDestroySpMat(mat_result), "cusparseDestroySpMat result");
        mat_result = nullptr;
        checkCusparse(cusparseDestroy(handle), "cusparseDestroy");
        handle = nullptr;

        cudaCopyToHost(result.row_offsets, d_result_rows, "cudaMemcpy result rows");
        cudaCopyToHost(result.col_indices, d_result_cols, "cudaMemcpy result cols");
        cudaCopyToHost(result.values, d_result_vals, "cudaMemcpy result vals");
        validateCsr(result);
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
    return result;
#else
    result.col_indices.clear();
    result.values.clear();
    result.row_offsets[0] = 0;
    for (int row = 0; row < left.rows; ++row) {
        std::map<int, double> row_values;
        for (int left_k = left.row_offsets[row]; left_k < left.row_offsets[row + 1]; ++left_k) {
            const int middle = left.col_indices[static_cast<std::size_t>(left_k)];
            const double left_value = left.values[static_cast<std::size_t>(left_k)];
            for (int right_k = right.row_offsets[middle]; right_k < right.row_offsets[middle + 1]; ++right_k) {
                const int col = right.col_indices[static_cast<std::size_t>(right_k)];
                row_values[col] += left_value * right.values[static_cast<std::size_t>(right_k)];
            }
        }
        for (const auto& entry : row_values) {
            if (std::abs(entry.second) > 0.0) {
                result.col_indices.push_back(entry.first);
                result.values.push_back(entry.second);
            }
        }
        result.row_offsets[static_cast<std::size_t>(row + 1)] =
            static_cast<int>(result.values.size());
    }
    return result;
#endif
}

DeviceCsrMatrix multiplyDeviceCsrMatrices(const DeviceCsrMatrix& left,
    const DeviceCsrMatrix& right)
{
    validateDeviceCsr(left);
    validateDeviceCsr(right);
    if (left.cols != right.rows)
        throw std::runtime_error("device CSR multiply dimension mismatch");

    DeviceCsrMatrix result;
    result.rows = left.rows;
    result.cols = right.cols;
    result.storage = std::make_shared<DeviceCsrStorage>();
    DeviceCsrStorage& result_storage = storageFor(result);
    result_storage.row_offsets.assign(static_cast<std::size_t>(result.rows + 1), 0);
    if (left.rows == 0 || right.cols == 0 || left.nonzeros == 0 || right.nonzeros == 0) {
        result.nonzeros = 0;
        return result;
    }

#ifdef ZIRAN_ENABLE_CUSPARSE
    const DeviceCsrStorage& left_storage = storageFor(left);
    const DeviceCsrStorage& right_storage = storageFor(right);
    void* buffer1 = nullptr;
    void* buffer2 = nullptr;
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat_left = nullptr;
    cusparseSpMatDescr_t mat_right = nullptr;
    cusparseSpMatDescr_t mat_result = nullptr;
    cusparseSpGEMMDescr_t spgemm_desc = nullptr;

    checkCusparse(cusparseCreate(&handle), "cusparseCreate device CSR");
    auto cleanup = [&]() {
        if (spgemm_desc)
            cusparseSpGEMM_destroyDescr(spgemm_desc);
        if (mat_left)
            cusparseDestroySpMat(mat_left);
        if (mat_right)
            cusparseDestroySpMat(mat_right);
        if (mat_result)
            cusparseDestroySpMat(mat_result);
        if (handle)
            cusparseDestroy(handle);
        if (buffer1)
            cudaFree(buffer1);
        if (buffer2)
            cudaFree(buffer2);
    };

    try {
        checkCusparse(cusparseCreateCsr(&mat_left, left.rows, left.cols,
                          static_cast<int64_t>(left.nonzeros),
                          const_cast<int*>(thrust::raw_pointer_cast(left_storage.row_offsets.data())),
                          const_cast<int*>(thrust::raw_pointer_cast(left_storage.col_indices.data())),
                          const_cast<double*>(thrust::raw_pointer_cast(left_storage.values.data())),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
            "cusparseCreateCsr device left");
        checkCusparse(cusparseCreateCsr(&mat_right, right.rows, right.cols,
                          static_cast<int64_t>(right.nonzeros),
                          const_cast<int*>(thrust::raw_pointer_cast(right_storage.row_offsets.data())),
                          const_cast<int*>(thrust::raw_pointer_cast(right_storage.col_indices.data())),
                          const_cast<double*>(thrust::raw_pointer_cast(right_storage.values.data())),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
            "cusparseCreateCsr device right");
        checkCusparse(cusparseCreateCsr(&mat_result, left.rows, right.cols, 0,
                          thrust::raw_pointer_cast(result_storage.row_offsets.data()),
                          nullptr, nullptr, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
            "cusparseCreateCsr device result");

        const double alpha = 1.0;
        const double beta = 0.0;
        constexpr cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
        constexpr cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_DEFAULT;
        checkCusparse(cusparseSpGEMM_createDescr(&spgemm_desc),
            "cusparseSpGEMM_createDescr device");

        size_t buffer_size1 = 0;
        checkCusparse(cusparseSpGEMM_workEstimation(handle, op, op, &alpha,
                          mat_left, mat_right, &beta, mat_result, CUDA_R_64F,
                          alg, spgemm_desc, &buffer_size1, nullptr),
            "cusparseSpGEMM_workEstimation device size");
        if (buffer_size1)
            checkCuda(cudaMalloc(&buffer1, buffer_size1),
                "cudaMalloc device SpGEMM buffer1");
        checkCusparse(cusparseSpGEMM_workEstimation(handle, op, op, &alpha,
                          mat_left, mat_right, &beta, mat_result, CUDA_R_64F,
                          alg, spgemm_desc, &buffer_size1, buffer1),
            "cusparseSpGEMM_workEstimation device");

        size_t buffer_size2 = 0;
        checkCusparse(cusparseSpGEMM_compute(handle, op, op, &alpha,
                          mat_left, mat_right, &beta, mat_result, CUDA_R_64F,
                          alg, spgemm_desc, &buffer_size2, nullptr),
            "cusparseSpGEMM_compute device size");
        if (buffer_size2)
            checkCuda(cudaMalloc(&buffer2, buffer_size2),
                "cudaMalloc device SpGEMM buffer2");
        checkCusparse(cusparseSpGEMM_compute(handle, op, op, &alpha,
                          mat_left, mat_right, &beta, mat_result, CUDA_R_64F,
                          alg, spgemm_desc, &buffer_size2, buffer2),
            "cusparseSpGEMM_compute device");
        checkCuda(cudaDeviceSynchronize(), "cusparseSpGEMM device compute synchronize");

        int64_t rows = 0;
        int64_t cols = 0;
        int64_t nnz = 0;
        checkCusparse(cusparseSpMatGetSize(mat_result, &rows, &cols, &nnz),
            "cusparseSpMatGetSize device");
        result.rows = static_cast<int>(rows);
        result.cols = static_cast<int>(cols);
        result.nonzeros = static_cast<std::size_t>(nnz);
        result_storage.col_indices.assign(result.nonzeros, 0);
        result_storage.values.assign(result.nonzeros, 0.0);
        checkCusparse(cusparseCsrSetPointers(mat_result,
                          thrust::raw_pointer_cast(result_storage.row_offsets.data()),
                          thrust::raw_pointer_cast(result_storage.col_indices.data()),
                          thrust::raw_pointer_cast(result_storage.values.data())),
            "cusparseCsrSetPointers device");
        checkCusparse(cusparseSpGEMM_copy(handle, op, op, &alpha,
                          mat_left, mat_right, &beta, mat_result, CUDA_R_64F,
                          alg, spgemm_desc),
            "cusparseSpGEMM_copy device");
        checkCuda(cudaDeviceSynchronize(), "cusparseSpGEMM device synchronize");
        cleanup();
        validateDeviceCsr(result);
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
#else
    return makeDeviceCsrFromHost(
        multiplyCsrMatrices(copyDeviceCsrToHost(left), copyDeviceCsrToHost(right)));
#endif
}

void assembleFluidQ1Q0System3DDevice(const FluidQ1Q0AssemblyInput& input,
    FluidQ1Q0DeviceAssemblyResult& result)
{
    if (input.dx <= 0.0 || input.dt <= 0.0 || input.rho == 0.0)
        throw std::runtime_error("invalid fluid assembly scalar input");
    if (input.volume_xq.size() != input.volume_mass.size())
        throw std::runtime_error("fluid assembly volume input size mismatch");
    if (input.face_xq.size() != input.face_normal.size())
        throw std::runtime_error("fluid assembly face input size mismatch");
    const bool resident_volume = input.resident_volume_count != 0;
    const bool resident_face_quadrature = input.resident_face_count != 0;
    if ((resident_volume &&
            (!input.resident_volume_xq || !input.resident_volume_mass)) ||
        (resident_face_quadrature &&
            (!input.resident_face_xq || !input.resident_face_normal)))
        throw std::runtime_error(
            "fluid assembly resident quadrature pointer is null");
    const std::size_t volume_count = resident_volume
        ? input.resident_volume_count : input.volume_xq.size();
    const std::size_t face_count = resident_face_quadrature
        ? input.resident_face_count : input.face_xq.size();

    const int velocity_dofs = input.velocity_node_count * 3;
    const int pressure_dofs = input.pressure_node_count + input.face_node_count;
    const bool resident_velocity = input.resident_velocity_nodes.storage != nullptr;
    const bool resident_pressure = input.resident_pressure_nodes.storage != nullptr;
    const bool resident_face = input.resident_face_nodes.storage != nullptr;
    if ((resident_velocity ? input.resident_velocity_nodes.size
                           : input.velocity_nodes.size()) !=
            static_cast<std::size_t>(input.velocity_node_count) ||
        (resident_pressure ? input.resident_pressure_nodes.size
                           : input.pressure_nodes.size()) !=
            static_cast<std::size_t>(input.pressure_node_count) ||
        (resident_face ? input.resident_face_nodes.size
                       : input.face_nodes.size()) !=
            static_cast<std::size_t>(input.face_node_count))
        throw std::runtime_error("fluid assembly node count mismatch");

    thrust::device_vector<AssemblyGridNode> d_velocity_nodes;
    thrust::device_vector<AssemblyGridNode> d_pressure_nodes;
    thrust::device_vector<AssemblyGridNode> d_face_nodes;
    const AssemblyGridNode* velocity_nodes = nullptr;
    const AssemblyGridNode* pressure_nodes = nullptr;
    const AssemblyGridNode* face_nodes = nullptr;
    if (resident_velocity) {
        velocity_nodes = getDeviceGridNodeMapRawView(
            input.resident_velocity_nodes).nodes;
    } else {
        d_velocity_nodes.assign(
            input.velocity_nodes.begin(), input.velocity_nodes.end());
        thrust::sort(d_velocity_nodes.begin(), d_velocity_nodes.end(),
            AssemblyGridNodeKeyLess{});
        velocity_nodes = d_velocity_nodes.empty() ? nullptr
            : thrust::raw_pointer_cast(d_velocity_nodes.data());
    }
    if (resident_pressure) {
        pressure_nodes = getDeviceGridNodeMapRawView(
            input.resident_pressure_nodes).nodes;
    } else {
        d_pressure_nodes.assign(
            input.pressure_nodes.begin(), input.pressure_nodes.end());
        thrust::sort(d_pressure_nodes.begin(), d_pressure_nodes.end(),
            AssemblyGridNodeKeyLess{});
        pressure_nodes = d_pressure_nodes.empty() ? nullptr
            : thrust::raw_pointer_cast(d_pressure_nodes.data());
    }
    if (resident_face) {
        face_nodes = getDeviceGridNodeMapRawView(input.resident_face_nodes).nodes;
    } else {
        d_face_nodes.assign(input.face_nodes.begin(), input.face_nodes.end());
        thrust::sort(d_face_nodes.begin(), d_face_nodes.end(),
            AssemblyGridNodeKeyLess{});
        face_nodes = d_face_nodes.empty() ? nullptr
            : thrust::raw_pointer_cast(d_face_nodes.data());
    }

    thrust::device_vector<CooEntry> d_minv_entries(
        static_cast<std::size_t>(input.velocity_node_count) * 3);
    thrust::fill(d_minv_entries.begin(), d_minv_entries.end(),
        CooEntry{-1, -1, 0.0});
    thrust::device_vector<double> d_a(static_cast<std::size_t>(velocity_dofs), 0.0);
    const int block_size = 256;
    if (input.velocity_node_count != 0) {
        const int block_count =
            (input.velocity_node_count + block_size - 1) / block_size;
        buildFluidMinvAndRhsKernel<<<block_count, block_size>>>(
            velocity_nodes, input.velocity_node_count, input.dt,
            thrust::raw_pointer_cast(d_minv_entries.data()),
            thrust::raw_pointer_cast(d_a.data()));
        checkCuda(cudaGetLastError(), "buildFluidMinvAndRhsKernel launch");
        checkCuda(cudaDeviceSynchronize(), "buildFluidMinvAndRhsKernel synchronize");
    }
    result.m_inv =
        finalizeCooToDevice(velocity_dofs, velocity_dofs, d_minv_entries);
    result.a.size = static_cast<std::size_t>(velocity_dofs);
    auto a_storage = std::make_shared<DeviceVectorStorage>();
    a_storage->values = std::move(d_a);
    result.a.storage = a_storage;

    // Keep the G COO entries resident on the device: volume kernel fills the
    // first block, face kernel the second, then finalize on the GPU. No 2M-entry
    // round-trip to the host.
    const std::size_t volume_entry_count = volume_count * 8 * 3;
    const std::size_t face_entry_count = face_count * 8 * 2 * 3;
    thrust::device_vector<CooEntry> d_g_entries(
        volume_entry_count + face_entry_count);
    thrust::fill(d_g_entries.begin(), d_g_entries.end(), CooEntry{-1, -1, 0.0});
    thrust::device_vector<Vec3d> d_volume_xq;
    thrust::device_vector<double> d_volume_mass;
    const Vec3d* volume_xq = static_cast<const Vec3d*>(
        input.resident_volume_xq);
    const double* volume_mass = static_cast<const double*>(
        input.resident_volume_mass);
    if (!resident_volume) {
        d_volume_xq.assign(input.volume_xq.begin(), input.volume_xq.end());
        d_volume_mass.assign(input.volume_mass.begin(), input.volume_mass.end());
        volume_xq = d_volume_xq.empty() ? nullptr
            : thrust::raw_pointer_cast(d_volume_xq.data());
        volume_mass = d_volume_mass.empty() ? nullptr
            : thrust::raw_pointer_cast(d_volume_mass.data());
    }
    if (volume_count != 0) {
        const int block_count =
            static_cast<int>((volume_count + block_size - 1) / block_size);
        buildFluidVolumeGKernel<<<block_count, block_size>>>(
            volume_xq, volume_mass, static_cast<int>(volume_count),
            input.dx, input.rho,
            velocity_nodes, input.velocity_node_count,
            pressure_nodes, input.pressure_node_count,
            thrust::raw_pointer_cast(d_g_entries.data()));
        checkCuda(cudaGetLastError(), "buildFluidVolumeGKernel launch");
        checkCuda(cudaDeviceSynchronize(), "buildFluidVolumeGKernel synchronize");
    }
    thrust::device_vector<Vec3d> d_face_xq;
    thrust::device_vector<Vec3d> d_face_normal;
    const Vec3d* face_xq = static_cast<const Vec3d*>(input.resident_face_xq);
    const Vec3d* face_normal = static_cast<const Vec3d*>(
        input.resident_face_normal);
    if (!resident_face_quadrature) {
        d_face_xq.assign(input.face_xq.begin(), input.face_xq.end());
        d_face_normal.assign(
            input.face_normal.begin(), input.face_normal.end());
        face_xq = d_face_xq.empty() ? nullptr
            : thrust::raw_pointer_cast(d_face_xq.data());
        face_normal = d_face_normal.empty() ? nullptr
            : thrust::raw_pointer_cast(d_face_normal.data());
    }
    if (face_count != 0) {
        const int block_count =
            static_cast<int>((face_count + block_size - 1) / block_size);
        buildFluidFaceGKernel<<<block_count, block_size>>>(
            face_xq, face_normal, static_cast<int>(face_count), input.dx,
            input.pressure_node_count,
            velocity_nodes, input.velocity_node_count,
            face_nodes, input.face_node_count,
            thrust::raw_pointer_cast(d_g_entries.data()) + volume_entry_count);
        checkCuda(cudaGetLastError(), "buildFluidFaceGKernel launch");
        checkCuda(cudaDeviceSynchronize(), "buildFluidFaceGKernel synchronize");
    }
    result.g = finalizeCooToDevice(velocity_dofs, pressure_dofs, d_g_entries);
    result.b.size = static_cast<std::size_t>(pressure_dofs);
    auto b_storage = std::make_shared<DeviceVectorStorage>();
    b_storage->values.assign(result.b.size, 0.0);
    result.b.storage = b_storage;

    if (resident_velocity) {
        result.velocity_nodes = input.resident_velocity_nodes;
    } else {
        result.velocity_nodes.size = d_velocity_nodes.size();
        auto storage = std::make_shared<DeviceGridNodeMapStorage>();
        storage->nodes = std::move(d_velocity_nodes);
        result.velocity_nodes.storage = storage;
    }
    if (resident_pressure) {
        result.pressure_nodes = input.resident_pressure_nodes;
    } else {
        result.pressure_nodes.size = d_pressure_nodes.size();
        auto storage = std::make_shared<DeviceGridNodeMapStorage>();
        storage->nodes = std::move(d_pressure_nodes);
        result.pressure_nodes.storage = storage;
    }
}

void assembleFluidQ1Q0System3D(const FluidQ1Q0AssemblyInput& input,
    FluidQ1Q0AssemblyResult& result)
{
    FluidQ1Q0DeviceAssemblyResult device;
    assembleFluidQ1Q0System3DDevice(input, device);
    result.m_inv = copyDeviceCsrToHost(device.m_inv);
    result.g = copyDeviceCsrToHost(device.g);
    result.a = copyDeviceVectorToHost(device.a);
    result.b = copyDeviceVectorToHost(device.b);
}

void assembleSolidQ2Q0System3DDevice(const SolidQ2Q0AssemblyInput& input,
    SolidQ2Q0DeviceAssemblyResult& result)
{
    if (input.dx <= 0.0 || input.dt <= 0.0)
        throw std::runtime_error("invalid solid assembly scalar input");

    const int velocity_dofs = input.velocity_node_count * 3;
    const int pressure_dofs = input.pressure_node_count;
    const bool resident_particles = input.resident_particles.size != 0;
    const std::size_t particle_count = resident_particles
        ? input.resident_particles.size : input.particles.size();
    if (resident_particles &&
        (!input.resident_particles.rest_x || !input.resident_particles.f ||
            !input.resident_particles.j ||
            !input.resident_particles.volume))
        throw std::runtime_error("incomplete resident solid assembly input");
    const bool resident_velocity_nodes =
        input.resident_velocity_nodes.size != 0;
    if (resident_velocity_nodes &&
        input.resident_velocity_nodes.size !=
            static_cast<std::size_t>(input.velocity_node_count))
        throw std::runtime_error(
            "resident solid velocity-node count mismatch");
    if (!resident_velocity_nodes &&
        input.velocity_nodes.size() !=
            static_cast<std::size_t>(input.velocity_node_count))
        throw std::runtime_error("host solid velocity-node count mismatch");
    const bool resident_pressure_nodes =
        input.resident_pressure_nodes.storage != nullptr;
    if ((resident_pressure_nodes ? input.resident_pressure_nodes.size
                                 : input.pressure_nodes.size()) !=
        static_cast<std::size_t>(input.pressure_node_count))
        throw std::runtime_error("solid pressure-node count mismatch");
    thrust::device_vector<AssemblyGridNode> d_velocity_nodes;
    const AssemblyGridNode* velocity_nodes = nullptr;
    const std::size_t velocity_node_count = resident_velocity_nodes
        ? input.resident_velocity_nodes.size : input.velocity_nodes.size();
    if (resident_velocity_nodes) {
        velocity_nodes =
            getDeviceGridNodeMapRawView(input.resident_velocity_nodes).nodes;
    } else {
        d_velocity_nodes.assign(
            input.velocity_nodes.begin(), input.velocity_nodes.end());
        thrust::sort(d_velocity_nodes.begin(), d_velocity_nodes.end(),
            AssemblyGridNodeKeyLess{});
        velocity_nodes = d_velocity_nodes.empty() ? nullptr
            : thrust::raw_pointer_cast(d_velocity_nodes.data());
    }
    thrust::device_vector<AssemblyGridNode> d_pressure_nodes;
    const AssemblyGridNode* pressure_nodes = nullptr;
    const std::size_t pressure_node_count = resident_pressure_nodes
        ? input.resident_pressure_nodes.size : input.pressure_nodes.size();
    if (resident_pressure_nodes) {
        pressure_nodes = getDeviceGridNodeMapRawView(
            input.resident_pressure_nodes).nodes;
    } else {
        d_pressure_nodes.assign(
            input.pressure_nodes.begin(), input.pressure_nodes.end());
        thrust::sort(d_pressure_nodes.begin(), d_pressure_nodes.end(),
            AssemblyGridNodeKeyLess{});
        pressure_nodes = d_pressure_nodes.empty() ? nullptr
            : thrust::raw_pointer_cast(d_pressure_nodes.data());
    }

    const int block_size = 256;
    thrust::device_vector<CooEntry> d_minv_entries(
        velocity_node_count * 3);
    thrust::fill(d_minv_entries.begin(), d_minv_entries.end(),
        CooEntry{-1, -1, 0.0});
    thrust::device_vector<double> d_a(static_cast<std::size_t>(velocity_dofs), 0.0);
    if (velocity_node_count != 0) {
        const int block_count =
            static_cast<int>((velocity_node_count + block_size - 1) /
                block_size);
        buildSolidMinvAndRhsKernel<<<block_count, block_size>>>(
            velocity_nodes, static_cast<int>(velocity_node_count), input.dt,
            input.gravity, thrust::raw_pointer_cast(d_minv_entries.data()),
            thrust::raw_pointer_cast(d_a.data()));
        checkCuda(cudaGetLastError(), "buildSolidMinvAndRhsKernel launch");
        checkCuda(cudaDeviceSynchronize(), "buildSolidMinvAndRhsKernel synchronize");
    }
    result.m_inv =
        finalizeCooToDevice(velocity_dofs, velocity_dofs, d_minv_entries);
    result.a.size = static_cast<std::size_t>(velocity_dofs);
    auto a_storage = std::make_shared<DeviceVectorStorage>();
    a_storage->values = std::move(d_a);
    result.a.storage = a_storage;

    thrust::device_vector<CooEntry> d_s_entries(pressure_node_count);
    thrust::fill(d_s_entries.begin(), d_s_entries.end(), CooEntry{-1, -1, 0.0});
    thrust::device_vector<double> d_b(static_cast<std::size_t>(pressure_dofs), 0.0);
    if (pressure_node_count != 0) {
        const int block_count =
            static_cast<int>((pressure_node_count + block_size - 1) / block_size);
        buildSolidSAndRhsKernel<<<block_count, block_size>>>(
            pressure_nodes, static_cast<int>(pressure_node_count), input.dt,
            thrust::raw_pointer_cast(d_s_entries.data()),
            thrust::raw_pointer_cast(d_b.data()));
        checkCuda(cudaGetLastError(), "buildSolidSAndRhsKernel launch");
        checkCuda(cudaDeviceSynchronize(), "buildSolidSAndRhsKernel synchronize");
    }
    result.s = finalizeCooToDevice(pressure_dofs, pressure_dofs, d_s_entries);
    result.b.size = static_cast<std::size_t>(pressure_dofs);
    auto b_storage = std::make_shared<DeviceVectorStorage>();
    b_storage->values = std::move(d_b);
    result.b.storage = b_storage;

    const std::size_t g_entry_count = particle_count * 27 * 3;
    thrust::device_vector<CooEntry> d_g_entries(g_entry_count);
    thrust::fill(d_g_entries.begin(), d_g_entries.end(), CooEntry{-1, -1, 0.0});
    if (particle_count != 0) {
        const int block_count =
            static_cast<int>((particle_count + block_size - 1) / block_size);
        if (resident_particles) {
            buildSolidGResidentKernel<<<block_count, block_size>>>(
                input.resident_particles.rest_x,
                input.resident_particles.volume,
                input.resident_particles.j,
                input.resident_particles.f,
                static_cast<int>(particle_count), input.compute_f_inv_t,
                input.dx, velocity_nodes,
                static_cast<int>(velocity_node_count),
                pressure_nodes, static_cast<int>(pressure_node_count),
                thrust::raw_pointer_cast(d_g_entries.data()));
            checkCuda(cudaGetLastError(),
                "buildSolidGResidentKernel launch");
            checkCuda(cudaDeviceSynchronize(),
                "buildSolidGResidentKernel synchronize");
        } else {
            thrust::device_vector<SolidQ2Q0Particle> d_particles(
                input.particles.begin(), input.particles.end());
            buildSolidGKernel<<<block_count, block_size>>>(
                thrust::raw_pointer_cast(d_particles.data()),
                static_cast<int>(particle_count), input.dx,
                velocity_nodes, static_cast<int>(velocity_node_count),
                pressure_nodes, static_cast<int>(pressure_node_count),
                thrust::raw_pointer_cast(d_g_entries.data()));
            checkCuda(cudaGetLastError(), "buildSolidGKernel launch");
            checkCuda(cudaDeviceSynchronize(),
                "buildSolidGKernel synchronize");
        }
    }
    result.g = finalizeCooToDevice(velocity_dofs, pressure_dofs, d_g_entries);

    if (resident_velocity_nodes) {
        result.velocity_nodes = input.resident_velocity_nodes;
    } else {
        result.velocity_nodes.size = d_velocity_nodes.size();
        auto velocity_node_storage =
            std::make_shared<DeviceGridNodeMapStorage>();
        velocity_node_storage->nodes = std::move(d_velocity_nodes);
        result.velocity_nodes.storage = velocity_node_storage;
    }
    if (resident_pressure_nodes) {
        result.pressure_nodes = input.resident_pressure_nodes;
    } else {
        result.pressure_nodes.size = d_pressure_nodes.size();
        auto pressure_node_storage =
            std::make_shared<DeviceGridNodeMapStorage>();
        pressure_node_storage->nodes = std::move(d_pressure_nodes);
        result.pressure_nodes.storage = pressure_node_storage;
    }
}

void assembleInterfaceW3DDevice(const InterfaceWAssemblyInput3D& input,
    DeviceCsrMatrix& result)
{
    const bool resident_samples = input.resident_rest_x
        || input.resident_current_x || input.resident_normal
        || input.resident_length || input.resident_sample_count != 0;
    const std::size_t sample_count = resident_samples
        ? input.resident_sample_count : input.rest_x.size();
    if (!(input.dx > 0.0) ||
        (input.velocity_degree != 1 && input.velocity_degree != 2) ||
        (resident_samples
                ? (!input.resident_rest_x || !input.resident_current_x
                    || !input.resident_normal || !input.resident_length)
                : (input.current_x.size() != sample_count
                    || input.normal.size() != sample_count
                    || input.length.size() != sample_count)) ||
        input.velocity_node_count < 0 ||
        input.resident_velocity_nodes.size !=
            static_cast<std::size_t>(input.velocity_node_count))
        throw std::runtime_error("invalid interface W device assembly input");
    const AssemblyGridNode* velocity_nodes = input.velocity_node_count == 0
        ? nullptr
        : getDeviceGridNodeMapRawView(input.resident_velocity_nodes).nodes;
    thrust::device_vector<Vec3d> d_rest_x;
    thrust::device_vector<Vec3d> d_current_x;
    thrust::device_vector<Vec3d> d_normal;
    thrust::device_vector<double> d_length;
    const Vec3d* rest_x = static_cast<const Vec3d*>(input.resident_rest_x);
    const Vec3d* current_x =
        static_cast<const Vec3d*>(input.resident_current_x);
    const Vec3d* normal = static_cast<const Vec3d*>(input.resident_normal);
    const double* length = static_cast<const double*>(input.resident_length);
    if (!resident_samples) {
        d_rest_x.assign(input.rest_x.begin(), input.rest_x.end());
        d_current_x.assign(input.current_x.begin(), input.current_x.end());
        d_normal.assign(input.normal.begin(), input.normal.end());
        d_length.assign(input.length.begin(), input.length.end());
        rest_x = sample_count == 0 ? nullptr
            : thrust::raw_pointer_cast(d_rest_x.data());
        current_x = sample_count == 0 ? nullptr
            : thrust::raw_pointer_cast(d_current_x.data());
        normal = sample_count == 0 ? nullptr
            : thrust::raw_pointer_cast(d_normal.data());
        length = sample_count == 0 ? nullptr
            : thrust::raw_pointer_cast(d_length.data());
    }
    thrust::device_vector<GridKey> d_row_keys(sample_count);
    constexpr int block_size = 256;
    if (sample_count != 0) {
        const int block_count = static_cast<int>(
            (sample_count + block_size - 1) / block_size);
        generateInterfaceWRowKeysKernel<<<block_count, block_size>>>(
            current_x,
            static_cast<int>(sample_count), input.dx,
            thrust::raw_pointer_cast(d_row_keys.data()));
        checkCuda(cudaGetLastError(),
            "generateInterfaceWRowKeysKernel launch");
    }
    thrust::sort(d_row_keys.begin(), d_row_keys.end());
    d_row_keys.erase(thrust::unique(d_row_keys.begin(), d_row_keys.end()),
        d_row_keys.end());
    const int row_count = static_cast<int>(d_row_keys.size());
    const int stencil_size = input.velocity_degree == 1 ? 8 : 27;
    thrust::device_vector<CooEntry> entries(
        sample_count * static_cast<std::size_t>(stencil_size) * 3);
    thrust::fill(entries.begin(), entries.end(), CooEntry{-1, -1, 0.0});
    if (sample_count != 0) {
        const int block_count = static_cast<int>(
            (sample_count + block_size - 1) / block_size);
        buildInterfaceWKernel<<<block_count, block_size>>>(
            rest_x, current_x, normal, length,
            static_cast<int>(sample_count), input.dx,
            input.velocity_degree,
            input.shift_velocity_half_dx ? 1 : 0,
            thrust::raw_pointer_cast(d_row_keys.data()), row_count,
            velocity_nodes, input.velocity_node_count,
            thrust::raw_pointer_cast(entries.data()));
        checkCuda(cudaGetLastError(), "buildInterfaceWKernel launch");
    }
    result = finalizeCooToDevice(row_count,
        input.velocity_node_count * 3, entries);
}

void assembleSolidQ2Q0System3D(const SolidQ2Q0AssemblyInput& input,
    SolidQ2Q0AssemblyResult& result)
{
    SolidQ2Q0DeviceAssemblyResult device;
    assembleSolidQ2Q0System3DDevice(input, device);
    result.m_inv = copyDeviceCsrToHost(device.m_inv);
    result.g = copyDeviceCsrToHost(device.g);
    result.s = copyDeviceCsrToHost(device.s);
    result.a = copyDeviceVectorToHost(device.a);
    result.b = copyDeviceVectorToHost(device.b);
}

OperatorCompareStats runSparseProductProbe(const std::string& label,
    const Eigen::SparseMatrix<double>& left,
    const Eigen::SparseMatrix<double>& right,
    const Eigen::SparseMatrix<double>& reference_product, int vector_seed)
{
    const CsrMatrix product = multiplyCsrMatrices(makeCsrFromEigen(left), makeCsrFromEigen(right));
    return compareSparseMatrixToReference(label, makeEigenFromCsr(product),
        reference_product, vector_seed);
}

OperatorCompareStats runSparseTripleProductProbe(const std::string& label,
    const Eigen::SparseMatrix<double>& left,
    const Eigen::SparseMatrix<double>& middle,
    const Eigen::SparseMatrix<double>& right,
    const Eigen::SparseMatrix<double>& reference_product, int vector_seed)
{
    const CsrMatrix left_middle = multiplyCsrMatrices(makeCsrFromEigen(left), makeCsrFromEigen(middle));
    const CsrMatrix product = multiplyCsrMatrices(left_middle, makeCsrFromEigen(right));
    return compareSparseMatrixToReference(label, makeEigenFromCsr(product),
        reference_product, vector_seed);
}

void runConjugateGradientProbe(const CsrMatrix& matrix,
    const std::vector<double>& rhs, double tolerance, int max_iterations,
    std::vector<double>& x, ConjugateGradientStats& stats,
    int jacobi_sweeps, double jacobi_omega)
{
    validateCsr(matrix);
    if (matrix.rows != matrix.cols)
        throw std::runtime_error("CG probe requires a square matrix");
    if (rhs.size() != static_cast<std::size_t>(matrix.rows))
        throw std::runtime_error("CG rhs size mismatch");
    if (max_iterations < 0)
        throw std::runtime_error("CG max_iterations must be non-negative");
    if (jacobi_sweeps < 1)
        throw std::runtime_error("CG jacobi_sweeps must be positive");

    x.assign(rhs.size(), 0.0);
    stats = ConjugateGradientStats{};
    if (matrix.rows == 0) {
        stats.converged = true;
        return;
    }

    thrust::device_vector<int> d_row_offsets(matrix.row_offsets.begin(), matrix.row_offsets.end());
    thrust::device_vector<int> d_col_indices(matrix.col_indices.begin(), matrix.col_indices.end());
    thrust::device_vector<double> d_values(matrix.values.begin(), matrix.values.end());
    thrust::device_vector<double> d_x(rhs.size(), 0.0);
    thrust::device_vector<double> d_rhs(rhs.begin(), rhs.end());
    thrust::device_vector<double> d_r = d_rhs;
    thrust::device_vector<double> d_z(rhs.size(), 0.0);
    thrust::device_vector<double> d_p(rhs.size(), 0.0);
    thrust::device_vector<double> d_Ap(rhs.size(), 0.0);
    thrust::device_vector<double> d_Az(rhs.size(), 0.0);
    thrust::device_vector<double> d_tmp(rhs.size(), 0.0);
    thrust::device_vector<double> d_inv_diag(rhs.size(), 1.0);

    const int block_size = 256;
    const int block_count = (matrix.rows + block_size - 1) / block_size;
    buildJacobiInverseDiagonalKernel<<<block_count, block_size>>>(matrix.rows,
        thrust::raw_pointer_cast(d_row_offsets.data()),
        thrust::raw_pointer_cast(d_col_indices.data()),
        thrust::raw_pointer_cast(d_values.data()),
        thrust::raw_pointer_cast(d_inv_diag.data()));
    checkCuda(cudaGetLastError(), "buildJacobiInverseDiagonalKernel launch");
    checkCuda(cudaDeviceSynchronize(), "buildJacobiInverseDiagonalKernel synchronize");

    const double rhs_norm = std::sqrt(deviceDot(d_rhs, d_rhs));
    stats.initial_residual = rhs_norm;
    if (rhs_norm == 0.0) {
        stats.converged = true;
        return;
    }

    applyJacobiSweepPreconditioner(matrix, d_row_offsets, d_col_indices, d_values,
        d_inv_diag, d_r, block_count, block_size, jacobi_sweeps, jacobi_omega,
        d_z, d_Az, d_tmp);
    d_p = d_z;
    double rz_old = deviceDot(d_r, d_z);

    for (int iter = 0; iter < max_iterations; ++iter) {
        csrSpmvKernel<<<block_count, block_size>>>(matrix.rows,
            thrust::raw_pointer_cast(d_row_offsets.data()),
            thrust::raw_pointer_cast(d_col_indices.data()),
            thrust::raw_pointer_cast(d_values.data()),
            thrust::raw_pointer_cast(d_p.data()),
            thrust::raw_pointer_cast(d_Ap.data()));
        checkCuda(cudaGetLastError(), "csrSpmvKernel CG launch");
        checkCuda(cudaDeviceSynchronize(), "csrSpmvKernel CG synchronize");

        const double pAp = deviceDot(d_p, d_Ap);
        if (std::abs(pAp) < 1e-300)
            break;
        const double alpha = rz_old / pAp;
        thrust::transform(d_x.begin(), d_x.end(), d_p.begin(), d_x.begin(), AxpyFunctor{alpha});
        thrust::transform(d_r.begin(), d_r.end(), d_Ap.begin(), d_r.begin(), XmyFunctor{alpha});

        stats.iterations = iter + 1;
        stats.final_residual = std::sqrt(deviceDot(d_r, d_r));
        stats.relative_residual = stats.final_residual / rhs_norm;
        if (stats.relative_residual <= tolerance) {
            stats.converged = true;
            thrust::copy(d_x.begin(), d_x.end(), x.begin());
            return;
        }

        applyJacobiSweepPreconditioner(matrix, d_row_offsets, d_col_indices, d_values,
            d_inv_diag, d_r, block_count, block_size, jacobi_sweeps, jacobi_omega,
            d_z, d_Az, d_tmp);
        const double rz_new = deviceDot(d_r, d_z);
        if (std::abs(rz_old) < 1e-300)
            break;
        const double beta = rz_new / rz_old;
        thrust::transform(d_z.begin(), d_z.end(), d_p.begin(), d_p.begin(), ZPlusBetaPFunctor{beta});
        rz_old = rz_new;
    }

    stats.final_residual = std::sqrt(deviceDot(d_r, d_r));
    stats.relative_residual = stats.final_residual / rhs_norm;
    thrust::copy(d_x.begin(), d_x.end(), x.begin());
}

std::vector<double> concatenateBlockVector(const std::vector<double>& p1,
    const std::vector<double>& alpha, const std::vector<double>& p2)
{
    std::vector<double> block;
    block.reserve(p1.size() + alpha.size() + p2.size());
    block.insert(block.end(), p1.begin(), p1.end());
    block.insert(block.end(), alpha.begin(), alpha.end());
    block.insert(block.end(), p2.begin(), p2.end());
    return block;
}

void splitBlockVector(const std::vector<double>& block, int p1_size,
    int alpha_size, int p2_size, std::vector<double>& p1,
    std::vector<double>& alpha, std::vector<double>& p2)
{
    if (p1_size < 0 || alpha_size < 0 || p2_size < 0)
        throw std::runtime_error("block sizes must be non-negative");
    const std::size_t total = static_cast<std::size_t>(p1_size + alpha_size + p2_size);
    if (block.size() != total)
        throw std::runtime_error("block vector size mismatch");

    p1.assign(block.begin(), block.begin() + p1_size);
    alpha.assign(block.begin() + p1_size, block.begin() + p1_size + alpha_size);
    p2.assign(block.begin() + p1_size + alpha_size, block.end());
}

} // namespace ZIRAN::GPU_STAGE3A
