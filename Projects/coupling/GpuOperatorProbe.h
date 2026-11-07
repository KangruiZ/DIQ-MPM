#pragma once

#include <Eigen/Sparse>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ZIRAN::GPU_STAGE3A {

using GridKey = std::uint64_t;

struct Vec3d {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Mat3d {
    double values[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

struct CsrMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<double> values;
};

// Keep Eigen allocation and destruction in the caller's C++ translation unit.
// Returning an Eigen sparse object from an nvcc-compiled translation unit can
// mix Eigen alignment/allocator settings with the host compiler's settings.
template <class Scalar>
Eigen::SparseMatrix<Scalar> makeEigenFromCsrLocal(const CsrMatrix& matrix)
{
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(matrix.values.size());
    for (int row = 0; row < matrix.rows; ++row) {
        const int begin = matrix.row_offsets[static_cast<std::size_t>(row)];
        const int end = matrix.row_offsets[static_cast<std::size_t>(row + 1)];
        for (int entry = begin; entry < end; ++entry) {
            triplets.emplace_back(row,
                matrix.col_indices[static_cast<std::size_t>(entry)],
                static_cast<Scalar>(matrix.values[static_cast<std::size_t>(entry)]));
        }
    }
    Eigen::SparseMatrix<Scalar> result(matrix.rows, matrix.cols);
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    return result;
}

struct AssemblyGridNode {
    GridKey key = 0;
    int idx = -1;
    double mass = 0.0;
    Vec3d new_v;
    Vec3d v;
};

// Sorted key -> row metadata retained on CUDA. Device assembly and transfer
// stages share this handle without staging the node AoS through host memory.
struct DeviceGridNodeMap {
    std::size_t size = 0;
    std::shared_ptr<void> storage;
};

struct FluidQ1Q0AssemblyInput {
    double dx = 0.0;
    double dt = 0.0;
    double rho = 1.0;
    int velocity_node_count = 0;
    int pressure_node_count = 0;
    int face_node_count = 0;
    std::vector<AssemblyGridNode> velocity_nodes;
    DeviceGridNodeMap resident_velocity_nodes;
    std::vector<AssemblyGridNode> pressure_nodes;
    // Optional device-owned Q0 pressure map.  Like the velocity map it is
    // sorted by key while idx stores the legacy SPGrid DOF permutation.
    DeviceGridNodeMap resident_pressure_nodes;
    std::vector<AssemblyGridNode> face_nodes;
    DeviceGridNodeMap resident_face_nodes;
    std::vector<Vec3d> volume_xq;
    std::vector<double> volume_mass;
    const void* resident_volume_xq = nullptr;
    const void* resident_volume_mass = nullptr;
    std::size_t resident_volume_count = 0;
    std::vector<Vec3d> face_xq;
    std::vector<Vec3d> face_normal;
    const void* resident_face_xq = nullptr;
    const void* resident_face_normal = nullptr;
    std::size_t resident_face_count = 0;
};

struct FluidQ1Q0AssemblyResult {
    CsrMatrix m_inv;
    CsrMatrix g;
    std::vector<double> a;
    std::vector<double> b;
};

struct SolidQ2Q0Particle {
    Vec3d x;
    Vec3d x_for_p;
    double volume = 0.0;
    double j = 1.0;
    Mat3d f;
    int compute_f_inv_t = 0;
};

// Persistent particle owner for the supported 3D TL solid path.  The host
// arrays seed the owner once (or after an explicit invalidation); subsequent
// Q2 G2P updates remain on the device and are reused by pressure/base-system
// assembly in the following coupling substep.
struct SolidParticleStateInput3D {
    std::vector<Vec3d> rest_x;
    std::vector<Vec3d> current_x;
    std::vector<Vec3d> velocity;
    std::vector<Mat3d> affine;
    std::vector<Mat3d> grad_velocity;
    std::vector<Mat3d> f;
    std::vector<double> j;
    std::vector<double> mass;
    std::vector<double> volume;
    std::vector<double> lambda;
};

struct PersistentSolidParticleState3D {
    std::shared_ptr<void> storage;
};

struct SolidParticleStateDeviceView {
    const Vec3d* rest_x = nullptr;
    const Vec3d* current_x = nullptr;
    const Vec3d* velocity = nullptr;
    const Mat3d* affine = nullptr;
    const Mat3d* grad_velocity = nullptr;
    const Mat3d* f = nullptr;
    const double* j = nullptr;
    const double* mass = nullptr;
    const double* volume = nullptr;
    const double* lambda = nullptr;
    std::size_t size = 0;
};

struct SolidParticleStateStats {
    std::size_t particles = 0;
    std::size_t velocity_nodes = 0;
    std::size_t pressure_nodes = 0;
    std::size_t missing_velocity_nodes = 0;
    std::size_t missing_pressure_nodes = 0;
    bool reused_input = false;
    double upload_ms = 0.0;
    double g2p_ms = 0.0;
    double total_ms = 0.0;
};

// Compact compatibility result for the still-hosted implicit grid.  The
// expensive particle scatter is performed from the persistent device owner;
// only one record per active Q2 node is downloaded until the implicit grid
// solver itself is device-owned.
struct SolidVelocityGridStats {
    std::size_t particles = 0;
    std::size_t nodes = 0;
    std::size_t host_download_bytes = 0;
    std::size_t topology_download_bytes = 0;
    std::size_t velocity_download_bytes = 0;
    bool reused_static_topology = false;
    bool seeded_legacy_topology = false;
    bool assigned_canonical_topology = false;
    double p2g_ms = 0.0;
    double download_ms = 0.0;
    double total_ms = 0.0;
};

struct SolidParticleReductionStats {
    std::size_t particles = 0;
    Vec3d min_corner;
    Vec3d max_corner;
    double max_speed = 0.0;
    double reduction_ms = 0.0;
};

struct DeviceGridNodeMapRawView {
    const AssemblyGridNode* nodes = nullptr;
};

struct DeviceGridNodeMapMutableRawView {
    AssemblyGridNode* nodes = nullptr;
};

struct SolidQ2Q0AssemblyInput {
    double dx = 0.0;
    double dt = 0.0;
    Vec3d gravity;
    int velocity_node_count = 0;
    int pressure_node_count = 0;
    std::vector<AssemblyGridNode> velocity_nodes;
    // Optional device-owned replacement for velocity_nodes.  It must be
    // sorted by key and retain the legacy SPGrid idx values; this preserves
    // the established DOF order while removing the repeated node H2D.
    DeviceGridNodeMap resident_velocity_nodes;
    std::vector<AssemblyGridNode> pressure_nodes;
    // Device-owned Q0 pressure map, sorted by key with idx retaining the
    // legacy SPGrid pressure-DOF permutation.
    DeviceGridNodeMap resident_pressure_nodes;
    std::vector<SolidQ2Q0Particle> particles;
    SolidParticleStateDeviceView resident_particles;
    int compute_f_inv_t = 0;
};

struct InterfaceWAssemblyInput3D {
    double dx = 0.0;
    int velocity_degree = 1;
    bool shift_velocity_half_dx = true;
    int velocity_node_count = 0;
    DeviceGridNodeMap resident_velocity_nodes;
    std::vector<Vec3d> rest_x;
    std::vector<Vec3d> current_x;
    std::vector<Vec3d> normal;
    std::vector<double> length;
    const void* resident_rest_x = nullptr;
    const void* resident_current_x = nullptr;
    const void* resident_normal = nullptr;
    const void* resident_length = nullptr;
    std::size_t resident_sample_count = 0;
};

struct SolidQ2Q0AssemblyResult {
    CsrMatrix m_inv;
    CsrMatrix g;
    CsrMatrix s;
    std::vector<double> a;
    std::vector<double> b;
};

struct DeviceCsrMatrix {
    int rows = 0;
    int cols = 0;
    std::size_t nonzeros = 0;
    std::shared_ptr<void> storage;
};

// Non-owning device pointers for CUDA-side consumers such as AMGX. The view
// remains valid while the source DeviceCsrMatrix remains alive.
struct DeviceCsrRawView {
    const int* row_offsets = nullptr;
    const int* col_indices = nullptr;
    const double* values = nullptr;
};

struct DeviceVector {
    std::size_t size = 0;
    std::shared_ptr<void> storage;
};

// Non-owning device pointer view. The view remains valid while the source
// DeviceVector remains alive.
struct DeviceVectorRawView {
    const double* values = nullptr;
};

// Reusable device workspace for the post-coupling velocity algebra
//     v = M^{-1} (a - G p).
// The pressure gradient, rhs and velocity retain capacity across coupling
// substeps, so this path does not allocate transient DeviceVector storage.
struct PersistentVelocityAlgebra {
    std::shared_ptr<void> storage;
};

struct FluidQ1Q0DeviceAssemblyResult {
    DeviceCsrMatrix m_inv;
    DeviceCsrMatrix g;
    DeviceVector a;
    DeviceVector b;
    DeviceGridNodeMap velocity_nodes;
    DeviceGridNodeMap pressure_nodes;
};

struct SolidQ2Q0DeviceAssemblyResult {
    DeviceCsrMatrix m_inv;
    DeviceCsrMatrix g;
    DeviceCsrMatrix s;
    DeviceVector a;
    DeviceVector b;
    DeviceGridNodeMap velocity_nodes;
    DeviceGridNodeMap pressure_nodes;
};

struct DeviceCsrBlock {
    DeviceCsrMatrix matrix;
    int row_offset = 0;
    int col_offset = 0;
    double scale = 1.0;
};

GridKey packGridKey3D(int x, int y, int z);

struct OperatorCompareStats {
    std::string label;
    int rows = 0;
    int cols = 0;
    std::size_t nonzeros = 0;
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double max_rel_error = 0.0;
};

struct ConjugateGradientStats {
    int iterations = 0;
    bool converged = false;
    double initial_residual = 0.0;
    double final_residual = 0.0;
    double relative_residual = 0.0;
};

CsrMatrix makeCsrFromEigen(const Eigen::SparseMatrix<double>& matrix);

Eigen::SparseMatrix<double> makeEigenFromCsr(const CsrMatrix& matrix);

DeviceCsrMatrix makeDeviceCsrFromHost(const CsrMatrix& matrix);

DeviceCsrRawView getDeviceCsrRawView(const DeviceCsrMatrix& matrix);

CsrMatrix copyDeviceCsrToHost(const DeviceCsrMatrix& matrix);

DeviceVector makeDeviceVectorFromHost(const std::vector<double>& values);

// Copies a contiguous double array already resident on the CUDA device into
// an owning DeviceVector, avoiding a device-to-host-to-device round trip.
DeviceVector makeDeviceVectorFromDevice(const double* values,
    std::size_t size);

DeviceVectorRawView getDeviceVectorRawView(const DeviceVector& vector);

DeviceGridNodeMapRawView getDeviceGridNodeMapRawView(
    const DeviceGridNodeMap& map);

DeviceGridNodeMapMutableRawView getDeviceGridNodeMapMutableRawView(
    DeviceGridNodeMap& map);

std::vector<AssemblyGridNode> copyDeviceGridNodeMapToHost(
    const DeviceGridNodeMap& map);

// Copies a sorted AssemblyGridNode array between device owners. This is the
// narrow ownership bridge used when an upstream CUDA stage has already built
// exact legacy key/idx metadata and Stage3A must retain it past that stage's
// scratch-buffer lifetime.
DeviceGridNodeMap makeDeviceGridNodeMapFromDeviceNodes(
    const AssemblyGridNode* nodes, std::size_t count);

bool persistentSolidParticleStateValid(
    const PersistentSolidParticleState3D& state, std::size_t particle_count);

void initializePersistentSolidParticleState3D(
    PersistentSolidParticleState3D& state,
    const SolidParticleStateInput3D& input,
    SolidParticleStateStats* stats = nullptr, bool force_upload = false);

void invalidatePersistentSolidParticleState3D(
    PersistentSolidParticleState3D& state);

SolidParticleStateDeviceView getPersistentSolidParticleStateDeviceView(
    const PersistentSolidParticleState3D& state);

void updatePersistentSolidParticlesQ2FromDevice3D(
    PersistentSolidParticleState3D& state,
    const DeviceGridNodeMap& velocity_nodes,
    const DeviceGridNodeMap& pressure_nodes,
    const DeviceVector& device_velocity,
    const DeviceVector& device_pressure, double dx, double dt,
    double apic_rpic_ratio, double d_inverse,
    SolidParticleStateStats* stats = nullptr);

// Solid-only steps consume the converged Q2 node new_v directly. With no
// pressure constraint, J is the clamped determinant of the updated F.
void updatePersistentSolidParticlesQ2FromGrid3D(
    PersistentSolidParticleState3D& state,
    const DeviceGridNodeMap& velocity_nodes, double dx, double dt,
    double apic_rpic_ratio, double d_inverse,
    SolidParticleStateStats* stats = nullptr);

void copyPersistentSolidParticleStateToHost(
    const PersistentSolidParticleState3D& state,
    SolidParticleStateInput3D& output);

// Output/fallback synchronization for mutable fields only. Rest position,
// mass, volume and lambda are immutable after owner initialization.
void copyPersistentSolidDynamicStateToHost(
    const PersistentSolidParticleState3D& state,
    SolidParticleStateInput3D& output);

void copyPersistentSolidCurrentXToHost(
    const PersistentSolidParticleState3D& state,
    std::vector<Vec3d>& current_x);

void buildPersistentSolidVelocityGridQ2ToHost(
    PersistentSolidParticleState3D& state, double dx,
    std::vector<AssemblyGridNode>& nodes,
    SolidVelocityGridStats* stats = nullptr,
    bool download_dynamic_velocity = true,
    const std::vector<AssemblyGridNode>* legacy_topology_seed = nullptr,
    bool assign_canonical_indices = false);

// View of the most recently built resident Q2 P2G grid.  The map remains
// owned by state and is invalidated/rebuilt together with that particle owner.
DeviceGridNodeMap getPersistentSolidVelocityGridQ2DeviceMap(
    const PersistentSolidParticleState3D& state);

void reducePersistentSolidParticleState3D(
    const PersistentSolidParticleState3D& state,
    SolidParticleReductionStats& stats);

DeviceVector sliceDeviceVector(const DeviceVector& vector,
    std::size_t offset, std::size_t size);

std::vector<double> copyDeviceVectorToHost(const DeviceVector& vector);

DeviceCsrMatrix transposeDeviceCsrMatrix(
    const DeviceCsrMatrix& matrix, double scale = 1.0);

DeviceVector multiplyDeviceCsrVector(
    const DeviceCsrMatrix& matrix, const DeviceVector& vector);

DeviceVector addDeviceVectors(
    const DeviceVector& left, double right_scale, const DeviceVector& right);

DeviceVector concatenateDeviceVectors(
    const std::vector<DeviceVector>& blocks);

DeviceVector computePressureVelocityDevice(
    PersistentVelocityAlgebra& workspace,
    const DeviceCsrMatrix& m_inv, const DeviceCsrMatrix& g,
    const DeviceVector& a, const DeviceVector& pressure);

DeviceCsrMatrix assembleDeviceCsrBlocks(int rows, int cols,
    const std::vector<DeviceCsrBlock>& blocks);

std::vector<double> makeDeterministicVector(int size, int seed);

void runCsrSpmvProbe(const CsrMatrix& matrix,
    const std::vector<double>& x, std::vector<double>& y);

void runDeviceCsrSpmvProbe(const DeviceCsrMatrix& matrix,
    const std::vector<double>& x, std::vector<double>& y);

OperatorCompareStats compareVectors(const std::string& label,
    const CsrMatrix& matrix, const std::vector<double>& cpu_y,
    const std::vector<double>& gpu_y);

OperatorCompareStats runSparseMatrixProbe(const std::string& label,
    const Eigen::SparseMatrix<double>& matrix, int vector_seed);

OperatorCompareStats compareSparseMatrixToReference(const std::string& label,
    const Eigen::SparseMatrix<double>& probe_matrix,
    const Eigen::SparseMatrix<double>& reference_matrix, int vector_seed);

// Compares the actual CSR coefficients, including entries present in only one
// of the two sparsity patterns.  This is preferable to an SpMV probe when an
// assembly kernel is being used as a writeback path.
OperatorCompareStats compareCsrMatrices(const std::string& label,
    const CsrMatrix& probe_matrix, const CsrMatrix& reference_matrix);

CsrMatrix multiplyCsrMatrices(const CsrMatrix& left, const CsrMatrix& right);

DeviceCsrMatrix multiplyDeviceCsrMatrices(const DeviceCsrMatrix& left,
    const DeviceCsrMatrix& right);

void assembleFluidQ1Q0System3D(const FluidQ1Q0AssemblyInput& input,
    FluidQ1Q0AssemblyResult& result);

void assembleFluidQ1Q0System3DDevice(const FluidQ1Q0AssemblyInput& input,
    FluidQ1Q0DeviceAssemblyResult& result);

void assembleSolidQ2Q0System3D(const SolidQ2Q0AssemblyInput& input,
    SolidQ2Q0AssemblyResult& result);

void assembleSolidQ2Q0System3DDevice(const SolidQ2Q0AssemblyInput& input,
    SolidQ2Q0DeviceAssemblyResult& result);

// Builds the interface constraint matrix directly in the canonical device
// velocity-node space. Rows are canonical sorted Q0 keys derived from the
// current interface samples, so independently assembled fluid/solid W blocks
// share exactly the same interface DOF order.
void assembleInterfaceW3DDevice(const InterfaceWAssemblyInput3D& input,
    DeviceCsrMatrix& result);

OperatorCompareStats runSparseProductProbe(const std::string& label,
    const Eigen::SparseMatrix<double>& left,
    const Eigen::SparseMatrix<double>& right,
    const Eigen::SparseMatrix<double>& reference_product, int vector_seed);

OperatorCompareStats runSparseTripleProductProbe(const std::string& label,
    const Eigen::SparseMatrix<double>& left,
    const Eigen::SparseMatrix<double>& middle,
    const Eigen::SparseMatrix<double>& right,
    const Eigen::SparseMatrix<double>& reference_product, int vector_seed);

void runConjugateGradientProbe(const CsrMatrix& matrix,
    const std::vector<double>& rhs, double tolerance, int max_iterations,
    std::vector<double>& x, ConjugateGradientStats& stats,
    int jacobi_sweeps = 1, double jacobi_omega = 1.0);

std::vector<double> concatenateBlockVector(const std::vector<double>& p1,
    const std::vector<double>& alpha, const std::vector<double>& p2);

void splitBlockVector(const std::vector<double>& block, int p1_size,
    int alpha_size, int p2_size, std::vector<double>& p1,
    std::vector<double>& alpha, std::vector<double>& p2);

} // namespace ZIRAN::GPU_STAGE3A
