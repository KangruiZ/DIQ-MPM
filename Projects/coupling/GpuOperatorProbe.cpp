#include "GpuOperatorProbe.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>

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

int floorToInt(double value)
{
    return static_cast<int>(std::floor(value));
}

GridKey packGridKey3DUnchecked(int x, int y, int z)
{
    const int bx = x + kCoordBias;
    const int by = y + kCoordBias;
    const int bz = z + kCoordBias;
    return (static_cast<GridKey>(bx) << (2 * kCoordBits)) |
        (static_cast<GridKey>(by) << kCoordBits) |
        static_cast<GridKey>(bz);
}

void linearWeights(double x, double dx, int& base, double w[2], double dw[2])
{
    const double gx = x / dx;
    base = floorToInt(gx);
    const double t = gx - static_cast<double>(base);
    w[0] = 1.0 - t;
    w[1] = t;
    dw[0] = -1.0;
    dw[1] = 1.0;
}

void quadraticWeights(double x, double dx, int& base, double w[3], double dw[3])
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

GridKey order0Key(const Vec3d& x, double dx)
{
    return packGridKey3DUnchecked(
        floorToInt(x.x / dx),
        floorToInt(x.y / dx),
        floorToInt(x.z / dx));
}

std::vector<AssemblyGridNode> sortedNodes(std::vector<AssemblyGridNode> nodes)
{
    std::sort(nodes.begin(), nodes.end(),
        [](const AssemblyGridNode& a, const AssemblyGridNode& b) {
            return a.key < b.key;
        });
    return nodes;
}

int findNodeIdx(const std::vector<AssemblyGridNode>& nodes, GridKey key)
{
    auto it = std::lower_bound(nodes.begin(), nodes.end(), key,
        [](const AssemblyGridNode& node, GridKey needle) {
            return node.key < needle;
        });
    if (it == nodes.end() || it->key != key)
        return -1;
    return it->idx;
}

Mat3d pseudoInverseTranspose3x3(const Mat3d& f)
{
    double b[3][3];
    double v[3][3] = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            b[row][col] = f.values[row * 3 + col];
    for (int sweep = 0; sweep < 12; ++sweep) {
        for (int pair = 0; pair < 3; ++pair) {
            const int p = pair == 0 ? 0 : (pair == 1 ? 0 : 1);
            const int q = pair == 0 ? 1 : 2;
            double alpha = 0.0, beta = 0.0, gamma = 0.0;
            for (int row = 0; row < 3; ++row) {
                alpha += b[row][p] * b[row][p];
                beta += b[row][q] * b[row][q];
                gamma += b[row][p] * b[row][q];
            }
            if (gamma == 0.0 ||
                std::abs(gamma) <= 1e-15 * std::sqrt(std::max(alpha * beta, 0.0)))
                continue;
            const double zeta = (beta - alpha) / (2.0 * gamma);
            const double sign = zeta >= 0.0 ? 1.0 : -1.0;
            const double t = sign /
                (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
            const double c = 1.0 / std::sqrt(1.0 + t * t);
            const double s = c * t;
            for (int row = 0; row < 3; ++row) {
                const double bp = b[row][p], bq = b[row][q];
                b[row][p] = c * bp - s * bq;
                b[row][q] = s * bp + c * bq;
                const double vp = v[row][p], vq = v[row][q];
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
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            for (int singular = 0; singular < 3; ++singular)
                result.values[row * 3 + col] += b[row][singular] *
                    inv_sigma_squared[singular] * v[col][singular];
    return result;
}

CsrMatrix finalizeCoo(int rows, int cols, std::vector<CooEntry>& entries)
{
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                      [rows, cols](const CooEntry& entry) {
                          return entry.row < 0 || entry.row >= rows ||
                              entry.col < 0 || entry.col >= cols ||
                              entry.value == 0.0;
                      }),
        entries.end());
    std::sort(entries.begin(), entries.end(),
        [](const CooEntry& a, const CooEntry& b) {
            return std::tie(a.row, a.col) < std::tie(b.row, b.col);
        });

    CsrMatrix result;
    result.rows = rows;
    result.cols = cols;
    result.row_offsets.assign(static_cast<std::size_t>(rows + 1), 0);

    std::size_t cursor = 0;
    for (int row = 0; row < rows; ++row) {
        result.row_offsets[static_cast<std::size_t>(row)] =
            static_cast<int>(result.values.size());
        while (cursor < entries.size() && entries[cursor].row == row) {
            const int col = entries[cursor].col;
            double sum = 0.0;
            while (cursor < entries.size() && entries[cursor].row == row &&
                   entries[cursor].col == col)
                sum += entries[cursor++].value;
            if (sum != 0.0) {
                result.col_indices.push_back(col);
                result.values.push_back(sum);
            }
        }
    }
    result.row_offsets[static_cast<std::size_t>(rows)] =
        static_cast<int>(result.values.size());
    return result;
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
            throw std::runtime_error("CSR column index out of range");
    }
}

std::vector<double> eigenProduct(const Eigen::SparseMatrix<double>& matrix,
    const std::vector<double>& x)
{
    Eigen::Map<const Eigen::VectorXd> x_map(x.data(), static_cast<Eigen::Index>(x.size()));
    Eigen::VectorXd y = matrix * x_map;
    return std::vector<double>(y.data(), y.data() + y.size());
}

double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error("dot input size mismatch");
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        result += a[i] * b[i];
    return result;
}

std::vector<double> makeJacobiInverseDiagonal(const CsrMatrix& matrix)
{
    std::vector<double> inv_diag(static_cast<std::size_t>(matrix.rows), 1.0);
    for (int row = 0; row < matrix.rows; ++row) {
        double diag = 0.0;
        for (int k = matrix.row_offsets[row]; k < matrix.row_offsets[row + 1]; ++k) {
            if (matrix.col_indices[static_cast<std::size_t>(k)] == row) {
                diag = matrix.values[static_cast<std::size_t>(k)];
                break;
            }
        }
        if (std::abs(diag) > 1e-30)
            inv_diag[static_cast<std::size_t>(row)] = 1.0 / diag;
    }
    return inv_diag;
}

void applyJacobiSweepPreconditioner(const CsrMatrix& matrix,
    const std::vector<double>& inv_diag, const std::vector<double>& r,
    int sweeps, double omega, std::vector<double>& z)
{
    z.assign(r.size(), 0.0);
    if (sweeps <= 1) {
        for (std::size_t i = 0; i < r.size(); ++i)
            z[i] = inv_diag[i] * r[i];
        return;
    }

    std::vector<double> Az;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        runCsrSpmvProbe(matrix, z, Az);
        for (std::size_t i = 0; i < r.size(); ++i)
            z[i] += omega * inv_diag[i] * (r[i] - Az[i]);
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
    device.storage = std::make_shared<CsrMatrix>(matrix);
    return device;
}

CsrMatrix copyDeviceCsrToHost(const DeviceCsrMatrix& matrix)
{
    if (!matrix.storage)
        throw std::runtime_error("device CSR storage is empty");
    const auto host = std::static_pointer_cast<CsrMatrix>(matrix.storage);
    validateCsr(*host);
    return *host;
}

DeviceVector makeDeviceVectorFromHost(const std::vector<double>& values)
{
    DeviceVector result;
    result.size = values.size();
    result.storage = std::make_shared<std::vector<double>>(values);
    return result;
}

DeviceVector makeDeviceVectorFromDevice(const double*, std::size_t)
{
    throw std::runtime_error("device vector copy requires the CUDA implementation");
}

DeviceVectorRawView getDeviceVectorRawView(const DeviceVector&)
{
    throw std::runtime_error("device vector view requires the CUDA implementation");
}

DeviceGridNodeMapRawView getDeviceGridNodeMapRawView(
    const DeviceGridNodeMap&)
{
    throw std::runtime_error(
        "device grid-node map view requires the CUDA implementation");
}

DeviceGridNodeMapMutableRawView getDeviceGridNodeMapMutableRawView(
    DeviceGridNodeMap&)
{
    throw std::runtime_error(
        "mutable device grid-node map view requires the CUDA implementation");
}

std::vector<AssemblyGridNode> copyDeviceGridNodeMapToHost(
    const DeviceGridNodeMap&)
{
    throw std::runtime_error(
        "device grid-node map download requires the CUDA implementation");
}

DeviceGridNodeMap makeDeviceGridNodeMapFromDeviceNodes(
    const AssemblyGridNode*, std::size_t)
{
    throw std::runtime_error(
        "device grid-node map copy requires the CUDA implementation");
}

bool persistentSolidParticleStateValid(
    const PersistentSolidParticleState3D&, std::size_t)
{
    return false;
}

void initializePersistentSolidParticleState3D(
    PersistentSolidParticleState3D&, const SolidParticleStateInput3D&,
    SolidParticleStateStats*, bool)
{
    throw std::runtime_error(
        "persistent solid particle state requires the CUDA implementation");
}

void invalidatePersistentSolidParticleState3D(
    PersistentSolidParticleState3D& state)
{
    state.storage.reset();
}

SolidParticleStateDeviceView getPersistentSolidParticleStateDeviceView(
    const PersistentSolidParticleState3D&)
{
    return {};
}

void updatePersistentSolidParticlesQ2FromDevice3D(
    PersistentSolidParticleState3D&, const DeviceGridNodeMap&,
    const DeviceGridNodeMap&, const DeviceVector&, const DeviceVector&,
    double, double, double, double, SolidParticleStateStats*)
{
    throw std::runtime_error(
        "solid particle Q2 update requires the CUDA implementation");
}

void updatePersistentSolidParticlesQ2FromGrid3D(
    PersistentSolidParticleState3D&, const DeviceGridNodeMap&,
    double, double, double, double, SolidParticleStateStats*)
{
    throw std::runtime_error(
        "solid-only Q2 update requires the CUDA implementation");
}

void copyPersistentSolidParticleStateToHost(
    const PersistentSolidParticleState3D&, SolidParticleStateInput3D&)
{
    throw std::runtime_error(
        "solid particle state download requires the CUDA implementation");
}

void copyPersistentSolidDynamicStateToHost(
    const PersistentSolidParticleState3D&, SolidParticleStateInput3D&)
{
    throw std::runtime_error(
        "solid dynamic state download requires the CUDA implementation");
}

void copyPersistentSolidCurrentXToHost(
    const PersistentSolidParticleState3D&, std::vector<Vec3d>&)
{
    throw std::runtime_error(
        "solid particle current-X download requires the CUDA implementation");
}

void buildPersistentSolidVelocityGridQ2ToHost(
    PersistentSolidParticleState3D&, double,
    std::vector<AssemblyGridNode>&, SolidVelocityGridStats*, bool,
    const std::vector<AssemblyGridNode>*, bool)
{
    throw std::runtime_error(
        "solid particle Q2 P2G requires the CUDA implementation");
}

DeviceGridNodeMap getPersistentSolidVelocityGridQ2DeviceMap(
    const PersistentSolidParticleState3D&)
{
    return {};
}

void reducePersistentSolidParticleState3D(
    const PersistentSolidParticleState3D&, SolidParticleReductionStats&)
{
    throw std::runtime_error(
        "solid particle reduction requires the CUDA implementation");
}

DeviceVector sliceDeviceVector(const DeviceVector& vector,
    std::size_t offset, std::size_t size)
{
    const std::vector<double> values = copyDeviceVectorToHost(vector);
    if (offset > values.size() || size > values.size() - offset)
        throw std::runtime_error("device vector slice is out of range");
    return makeDeviceVectorFromHost(std::vector<double>(
        values.begin() + static_cast<std::ptrdiff_t>(offset),
        values.begin() + static_cast<std::ptrdiff_t>(offset + size)));
}

std::vector<double> copyDeviceVectorToHost(const DeviceVector& vector)
{
    if (!vector.storage)
        throw std::runtime_error("device vector storage is empty");
    const auto values =
        std::static_pointer_cast<std::vector<double>>(vector.storage);
    if (values->size() != vector.size)
        throw std::runtime_error("device vector size mismatch");
    return *values;
}

DeviceCsrMatrix transposeDeviceCsrMatrix(
    const DeviceCsrMatrix& matrix, double scale)
{
    const CsrMatrix host = copyDeviceCsrToHost(matrix);
    std::vector<CooEntry> entries;
    entries.reserve(host.values.size());
    for (int row = 0; row < host.rows; ++row)
        for (int entry = host.row_offsets[static_cast<std::size_t>(row)];
             entry < host.row_offsets[static_cast<std::size_t>(row + 1)]; ++entry)
            entries.push_back(CooEntry{
                host.col_indices[static_cast<std::size_t>(entry)], row,
                scale * host.values[static_cast<std::size_t>(entry)]});
    return makeDeviceCsrFromHost(finalizeCoo(host.cols, host.rows, entries));
}

DeviceVector multiplyDeviceCsrVector(
    const DeviceCsrMatrix& matrix, const DeviceVector& vector)
{
    const CsrMatrix host_matrix = copyDeviceCsrToHost(matrix);
    const std::vector<double> host_vector = copyDeviceVectorToHost(vector);
    std::vector<double> result;
    runCsrSpmvProbe(host_matrix, host_vector, result);
    return makeDeviceVectorFromHost(result);
}

DeviceVector addDeviceVectors(
    const DeviceVector& left, double right_scale, const DeviceVector& right)
{
    std::vector<double> result = copyDeviceVectorToHost(left);
    const std::vector<double> right_values = copyDeviceVectorToHost(right);
    if (result.size() != right_values.size())
        throw std::runtime_error("device vector add size mismatch");
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] += right_scale * right_values[i];
    return makeDeviceVectorFromHost(result);
}

DeviceVector concatenateDeviceVectors(const std::vector<DeviceVector>& blocks)
{
    std::vector<double> result;
    std::size_t total_size = 0;
    for (const DeviceVector& block : blocks)
        total_size += block.size;
    result.reserve(total_size);
    for (const DeviceVector& block : blocks) {
        const std::vector<double> values = copyDeviceVectorToHost(block);
        result.insert(result.end(), values.begin(), values.end());
    }
    return makeDeviceVectorFromHost(result);
}

DeviceVector computePressureVelocityDevice(
    PersistentVelocityAlgebra&, const DeviceCsrMatrix& m_inv,
    const DeviceCsrMatrix& g, const DeviceVector& a,
    const DeviceVector& pressure)
{
    const DeviceVector gradient = multiplyDeviceCsrVector(g, pressure);
    const DeviceVector rhs = addDeviceVectors(a, -1.0, gradient);
    return multiplyDeviceCsrVector(m_inv, rhs);
}

DeviceCsrMatrix assembleDeviceCsrBlocks(int rows, int cols,
    const std::vector<DeviceCsrBlock>& blocks)
{
    std::vector<CooEntry> entries;
    std::size_t total_entries = 0;
    for (const DeviceCsrBlock& block : blocks)
        total_entries += block.matrix.nonzeros;
    entries.reserve(total_entries);
    for (const DeviceCsrBlock& block : blocks) {
        const CsrMatrix host = copyDeviceCsrToHost(block.matrix);
        if (block.row_offset < 0 || block.col_offset < 0 ||
            block.row_offset + host.rows > rows ||
            block.col_offset + host.cols > cols)
            throw std::runtime_error("device CSR block is outside destination");
        for (int row = 0; row < host.rows; ++row)
            for (int entry = host.row_offsets[static_cast<std::size_t>(row)];
                 entry < host.row_offsets[static_cast<std::size_t>(row + 1)];
                 ++entry)
                entries.push_back(CooEntry{
                    block.row_offset + row,
                    block.col_offset +
                        host.col_indices[static_cast<std::size_t>(entry)],
                    block.scale * host.values[static_cast<std::size_t>(entry)]});
    }
    return makeDeviceCsrFromHost(finalizeCoo(rows, cols, entries));
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
    for (int row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        for (int k = matrix.row_offsets[row]; k < matrix.row_offsets[row + 1]; ++k)
            sum += matrix.values[static_cast<std::size_t>(k)] *
                x[static_cast<std::size_t>(matrix.col_indices[static_cast<std::size_t>(k)])];
        y[static_cast<std::size_t>(row)] = sum;
    }
}

void runDeviceCsrSpmvProbe(const DeviceCsrMatrix& matrix,
    const std::vector<double>& x, std::vector<double>& y)
{
    runCsrSpmvProbe(copyDeviceCsrToHost(matrix), x, y);
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

    Eigen::SparseMatrix<double> left_matrix = makeEigenFromCsr(left);
    Eigen::SparseMatrix<double> right_matrix = makeEigenFromCsr(right);
    Eigen::SparseMatrix<double> product = left_matrix * right_matrix;
    product.makeCompressed();
    return makeCsrFromEigen(product);
}

DeviceCsrMatrix multiplyDeviceCsrMatrices(const DeviceCsrMatrix& left,
    const DeviceCsrMatrix& right)
{
    return makeDeviceCsrFromHost(
        multiplyCsrMatrices(copyDeviceCsrToHost(left), copyDeviceCsrToHost(right)));
}

void assembleFluidQ1Q0System3D(const FluidQ1Q0AssemblyInput& input,
    FluidQ1Q0AssemblyResult& result)
{
    if (input.dx <= 0.0 || input.dt <= 0.0 || input.rho == 0.0)
        throw std::runtime_error("invalid fluid assembly scalar input");
    if (input.volume_xq.size() != input.volume_mass.size())
        throw std::runtime_error("fluid assembly volume input size mismatch");
    if (input.face_xq.size() != input.face_normal.size())
        throw std::runtime_error("fluid assembly face input size mismatch");

    const std::vector<AssemblyGridNode> velocity_nodes =
        sortedNodes(input.velocity_nodes);
    const std::vector<AssemblyGridNode> pressure_nodes =
        sortedNodes(input.pressure_nodes);
    const std::vector<AssemblyGridNode> face_nodes =
        sortedNodes(input.face_nodes);
    const int velocity_dofs = input.velocity_node_count * 3;
    const int pressure_dofs = input.pressure_node_count + input.face_node_count;

    std::vector<CooEntry> minv_entries;
    minv_entries.reserve(static_cast<std::size_t>(velocity_dofs));
    result.a.assign(static_cast<std::size_t>(velocity_dofs), 0.0);
    for (const AssemblyGridNode& node : input.velocity_nodes) {
        if (node.idx < 0 || node.idx >= input.velocity_node_count)
            continue;
        const double inv_mass_dt = input.dt / node.mass;
        const double mass_over_dt = node.mass / input.dt;
        const double v[3] = {node.new_v.x, node.new_v.y, node.new_v.z};
        for (int d = 0; d < 3; ++d) {
            const int row = node.idx * 3 + d;
            minv_entries.push_back(CooEntry{row, row, inv_mass_dt});
            result.a[static_cast<std::size_t>(row)] += mass_over_dt * v[d];
        }
    }
    result.m_inv = finalizeCoo(velocity_dofs, velocity_dofs, minv_entries);

    std::vector<CooEntry> g_entries;
    g_entries.reserve(input.volume_xq.size() * 8 * 3 +
        input.face_xq.size() * 8 * 2 * 3);
    for (std::size_t q = 0; q < input.volume_xq.size(); ++q) {
        Vec3d x{input.volume_xq[q].x - 0.5 * input.dx,
            input.volume_xq[q].y - 0.5 * input.dx,
            input.volume_xq[q].z - 0.5 * input.dx};
        int base[3];
        double w[3][2];
        double dw[3][2];
        linearWeights(x.x, input.dx, base[0], w[0], dw[0]);
        linearWeights(x.y, input.dx, base[1], w[1], dw[1]);
        linearWeights(x.z, input.dx, base[2], w[2], dw[2]);
        const int pressure_idx = findNodeIdx(pressure_nodes, order0Key(x, input.dx));
        if (pressure_idx < 0)
            continue;
        const double vol = input.volume_mass[q] / input.rho;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    const int vel_idx = findNodeIdx(velocity_nodes,
                        packGridKey3DUnchecked(base[0] + i, base[1] + j, base[2] + k));
                    if (vel_idx < 0)
                        continue;
                    const double inv_dx = 1.0 / input.dx;
                    const double grad[3] = {
                        inv_dx * dw[0][i] * w[1][j] * w[2][k],
                        inv_dx * w[0][i] * dw[1][j] * w[2][k],
                        inv_dx * w[0][i] * w[1][j] * dw[2][k]};
                    for (int alpha = 0; alpha < 3; ++alpha) {
                        g_entries.push_back(CooEntry{
                            vel_idx * 3 + alpha, pressure_idx,
                            -vol * grad[alpha]});
                    }
                }
    }

    const double face_len = input.dx * input.dx;
    for (std::size_t q = 0; q < input.face_xq.size(); ++q) {
        Vec3d x{input.face_xq[q].x - 0.5 * input.dx,
            input.face_xq[q].y - 0.5 * input.dx,
            input.face_xq[q].z - 0.5 * input.dx};
        int base[3];
        double w[3][2];
        double dw[3][2];
        linearWeights(x.x, input.dx, base[0], w[0], dw[0]);
        linearWeights(x.y, input.dx, base[1], w[1], dw[1]);
        linearWeights(x.z, input.dx, base[2], w[2], dw[2]);
        const double normal[3] = {
            input.face_normal[q].x,
            input.face_normal[q].y,
            input.face_normal[q].z};
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    const int vel_idx = findNodeIdx(velocity_nodes,
                        packGridKey3DUnchecked(base[0] + i, base[1] + j, base[2] + k));
                    if (vel_idx < 0)
                        continue;
                    const double weight = w[0][i] * w[1][j] * w[2][k];
                    for (int delta = -1; delta <= 1; delta += 2) {
                        Vec3d sample{
                            input.face_xq[q].x + input.face_normal[q].x * 0.5 * input.dx * delta - 0.5 * input.dx,
                            input.face_xq[q].y + input.face_normal[q].y * 0.5 * input.dx * delta - 0.5 * input.dx,
                            input.face_xq[q].z + input.face_normal[q].z * 0.5 * input.dx * delta - 0.5 * input.dx};
                        const int face_idx = findNodeIdx(face_nodes, order0Key(sample, input.dx));
                        if (face_idx < 0)
                            continue;
                        for (int alpha = 0; alpha < 3; ++alpha) {
                            g_entries.push_back(CooEntry{
                                vel_idx * 3 + alpha,
                                input.pressure_node_count + face_idx,
                                face_len * weight * normal[alpha]});
                        }
                    }
                }
    }
    result.g = finalizeCoo(velocity_dofs, pressure_dofs, g_entries);
    result.b.assign(static_cast<std::size_t>(pressure_dofs), 0.0);
}

void assembleSolidQ2Q0System3D(const SolidQ2Q0AssemblyInput& input,
    SolidQ2Q0AssemblyResult& result)
{
    if (input.dx <= 0.0 || input.dt <= 0.0)
        throw std::runtime_error("invalid solid assembly scalar input");

    const std::vector<AssemblyGridNode> velocity_nodes =
        sortedNodes(input.velocity_nodes);
    const std::vector<AssemblyGridNode> pressure_nodes =
        sortedNodes(input.pressure_nodes);
    const int velocity_dofs = input.velocity_node_count * 3;
    const int pressure_dofs = input.pressure_node_count;

    std::vector<CooEntry> minv_entries;
    minv_entries.reserve(static_cast<std::size_t>(velocity_dofs));
    result.a.assign(static_cast<std::size_t>(velocity_dofs), 0.0);
    for (const AssemblyGridNode& node : input.velocity_nodes) {
        if (node.idx < 0 || node.idx >= input.velocity_node_count || node.mass == 0.0)
            continue;
        const double inv_mass_dt = input.dt / node.mass;
        const double mass_over_dt = node.mass / input.dt;
        const double v[3] = {
            node.new_v.x + input.dt * input.gravity.x,
            node.new_v.y + input.dt * input.gravity.y,
            node.new_v.z + input.dt * input.gravity.z};
        for (int d = 0; d < 3; ++d) {
            const int row = node.idx * 3 + d;
            minv_entries.push_back(CooEntry{row, row, inv_mass_dt});
            result.a[static_cast<std::size_t>(row)] += mass_over_dt * v[d];
        }
    }
    result.m_inv = finalizeCoo(velocity_dofs, velocity_dofs, minv_entries);

    std::vector<CooEntry> s_entries;
    s_entries.reserve(static_cast<std::size_t>(pressure_dofs));
    result.b.assign(static_cast<std::size_t>(pressure_dofs), 0.0);
    for (const AssemblyGridNode& node : input.pressure_nodes) {
        if (node.idx < 0 || node.idx >= input.pressure_node_count)
            continue;
        s_entries.push_back(CooEntry{node.idx, node.idx, node.v.x / input.dt});
        result.b[static_cast<std::size_t>(node.idx)] = node.v.x * node.v.y / input.dt;
    }
    result.s = finalizeCoo(pressure_dofs, pressure_dofs, s_entries);

    std::vector<CooEntry> g_entries;
    g_entries.reserve(input.particles.size() * 27 * 3);
    for (const SolidQ2Q0Particle& particle : input.particles) {
        Mat3d f_inv_t;
        f_inv_t.values[0] = 1.0;
        f_inv_t.values[4] = 1.0;
        f_inv_t.values[8] = 1.0;
        if (particle.compute_f_inv_t)
            f_inv_t = pseudoInverseTranspose3x3(particle.f);
        int velocity_base[3];
        double w3[3][3];
        double dw3[3][3];
        quadraticWeights(particle.x.x, input.dx, velocity_base[0], w3[0], dw3[0]);
        quadraticWeights(particle.x.y, input.dx, velocity_base[1], w3[1], dw3[1]);
        quadraticWeights(particle.x.z, input.dx, velocity_base[2], w3[2], dw3[2]);
        Vec3d pressure_x{particle.x_for_p.x - 0.5 * input.dx,
            particle.x_for_p.y - 0.5 * input.dx,
            particle.x_for_p.z - 0.5 * input.dx};
        const int pressure_idx = findNodeIdx(pressure_nodes, order0Key(pressure_x, input.dx));
        if (pressure_idx < 0)
            continue;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) {
                    const int velocity_idx = findNodeIdx(velocity_nodes,
                        packGridKey3DUnchecked(velocity_base[0] + i,
                            velocity_base[1] + j, velocity_base[2] + k));
                    if (velocity_idx < 0)
                        continue;
                    const double inv_dx = 1.0 / input.dx;
                    const double grad_index[3] = {
                        inv_dx * dw3[0][i] * w3[1][j] * w3[2][k],
                        inv_dx * w3[0][i] * dw3[1][j] * w3[2][k],
                        inv_dx * w3[0][i] * w3[1][j] * dw3[2][k]};
                    double grad[3] = {0.0, 0.0, 0.0};
                    for (int row = 0; row < 3; ++row)
                        for (int col = 0; col < 3; ++col)
                            grad[row] += f_inv_t.values[row * 3 + col] *
                                grad_index[col];
                    const double scale = -particle.volume * particle.j;
                    for (int alpha = 0; alpha < 3; ++alpha)
                        g_entries.push_back(CooEntry{
                            velocity_idx * 3 + alpha, pressure_idx,
                            scale * grad[alpha]});
                }
    }
    result.g = finalizeCoo(velocity_dofs, pressure_dofs, g_entries);
}

void assembleFluidQ1Q0System3DDevice(const FluidQ1Q0AssemblyInput& input,
    FluidQ1Q0DeviceAssemblyResult& result)
{
    FluidQ1Q0AssemblyResult host;
    assembleFluidQ1Q0System3D(input, host);
    result.m_inv = makeDeviceCsrFromHost(host.m_inv);
    result.g = makeDeviceCsrFromHost(host.g);
    result.a = makeDeviceVectorFromHost(host.a);
    result.b = makeDeviceVectorFromHost(host.b);
}

void assembleSolidQ2Q0System3DDevice(const SolidQ2Q0AssemblyInput& input,
    SolidQ2Q0DeviceAssemblyResult& result)
{
    SolidQ2Q0AssemblyResult host;
    assembleSolidQ2Q0System3D(input, host);
    result.m_inv = makeDeviceCsrFromHost(host.m_inv);
    result.g = makeDeviceCsrFromHost(host.g);
    result.s = makeDeviceCsrFromHost(host.s);
    result.a = makeDeviceVectorFromHost(host.a);
    result.b = makeDeviceVectorFromHost(host.b);
}

void assembleInterfaceW3DDevice(const InterfaceWAssemblyInput3D&,
    DeviceCsrMatrix&)
{
    throw std::runtime_error(
        "interface W device assembly requires the CUDA implementation");
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
    const double rhs_norm = std::sqrt(dot(rhs, rhs));
    stats.initial_residual = rhs_norm;
    if (rhs_norm == 0.0) {
        stats.converged = true;
        return;
    }

    std::vector<double> r = rhs;
    const std::vector<double> inv_diag = makeJacobiInverseDiagonal(matrix);
    std::vector<double> z(rhs.size(), 0.0);
    std::vector<double> p(rhs.size(), 0.0);
    applyJacobiSweepPreconditioner(matrix, inv_diag, r, jacobi_sweeps, jacobi_omega, z);
    p = z;
    double rz_old = dot(r, z);

    std::vector<double> Ap;
    for (int iter = 0; iter < max_iterations; ++iter) {
        runCsrSpmvProbe(matrix, p, Ap);
        const double pAp = dot(p, Ap);
        if (std::abs(pAp) < 1e-300)
            break;
        const double alpha = rz_old / pAp;
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        stats.iterations = iter + 1;
        stats.final_residual = std::sqrt(dot(r, r));
        stats.relative_residual = stats.final_residual / rhs_norm;
        if (stats.relative_residual <= tolerance) {
            stats.converged = true;
            return;
        }

        applyJacobiSweepPreconditioner(matrix, inv_diag, r, jacobi_sweeps, jacobi_omega, z);
        const double rz_new = dot(r, z);
        if (std::abs(rz_old) < 1e-300)
            break;
        const double beta = rz_new / rz_old;
        for (std::size_t i = 0; i < rhs.size(); ++i)
            p[i] = z[i] + beta * p[i];
        rz_old = rz_new;
    }
    stats.final_residual = std::sqrt(dot(r, r));
    stats.relative_residual = stats.final_residual / rhs_norm;
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
