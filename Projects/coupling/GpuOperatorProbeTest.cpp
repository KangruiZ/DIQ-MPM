#include "GpuOperatorProbe.h"

#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}

Eigen::SparseMatrix<double> makeSparse(
    int rows, int cols, const std::vector<Eigen::Triplet<double>>& triplets)
{
    Eigen::SparseMatrix<double> matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
}

void addEigenBlock(int row_offset, int col_offset,
    const Eigen::SparseMatrix<double>& matrix,
    std::vector<std::vector<std::pair<int, double>>>& rows)
{
    for (int k = 0; k < matrix.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it)
            rows[static_cast<std::size_t>(row_offset + it.row())].emplace_back(
                col_offset + it.col(), it.value());
    }
}

void addCsrBlock(int row_offset, int col_offset,
    const ZIRAN::GPU_STAGE3A::CsrMatrix& matrix, double scale,
    std::vector<std::vector<std::pair<int, double>>>& rows)
{
    for (int row = 0; row < matrix.rows; ++row) {
        auto& dst = rows[static_cast<std::size_t>(row_offset + row)];
        for (int k = matrix.row_offsets[static_cast<std::size_t>(row)];
             k < matrix.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
            dst.emplace_back(
                col_offset + matrix.col_indices[static_cast<std::size_t>(k)],
                scale * matrix.values[static_cast<std::size_t>(k)]);
        }
    }
}

ZIRAN::GPU_STAGE3A::CsrMatrix finalizeRows(
    int rows_count, int cols_count,
    std::vector<std::vector<std::pair<int, double>>>& rows)
{
    ZIRAN::GPU_STAGE3A::CsrMatrix result;
    result.rows = rows_count;
    result.cols = cols_count;
    result.row_offsets.push_back(0);
    for (int row = 0; row < rows_count; ++row) {
        auto& entries = rows[static_cast<std::size_t>(row)];
        std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        int last_col = -1;
        double sum = 0.0;
        auto flush = [&]() {
            if (last_col >= 0 && sum != 0.0) {
                result.col_indices.push_back(last_col);
                result.values.push_back(sum);
            }
        };
        for (const auto& entry : entries) {
            if (entry.first != last_col) {
                flush();
                last_col = entry.first;
                sum = entry.second;
            } else {
                sum += entry.second;
            }
        }
        flush();
        result.row_offsets.push_back(static_cast<int>(result.values.size()));
    }
    return result;
}

} // namespace

int main()
{
    using namespace ZIRAN::GPU_STAGE3A;

    const CsrMatrix matrix{
        3,
        3,
        std::vector<int>{0, 2, 3, 5},
        std::vector<int>{0, 2, 1, 0, 2},
        std::vector<double>{2.0, -1.0, 4.0, 0.5, 3.0}};
    const std::vector<double> x{1.5, -2.0, 0.25};

    std::vector<double> y;
    runCsrSpmvProbe(matrix, x, y);
    require(y.size() == 3, "CSR SpMV should produce one value per row");
    require(std::abs(y[0] - 2.75) < 1e-12, "CSR SpMV row 0 mismatch");
    require(std::abs(y[1] + 8.0) < 1e-12, "CSR SpMV row 1 mismatch");
    require(std::abs(y[2] - 1.5) < 1e-12, "CSR SpMV row 2 mismatch");

    const std::vector<double> reference{2.75, -8.0, 1.5};
    const OperatorCompareStats stats = compareVectors("tiny_spmv", matrix, reference, y);
    require(stats.rows == 3, "compare stats rows mismatch");
    require(stats.cols == 3, "compare stats cols mismatch");
    require(stats.nonzeros == 5, "compare stats nonzeros mismatch");
    require(stats.max_abs_error == 0.0, "compare stats max abs error mismatch");
    require(stats.mean_abs_error == 0.0, "compare stats mean abs error mismatch");
    require(stats.max_rel_error == 0.0, "compare stats max relative error mismatch");

    const CsrMatrix right_matrix{
        3,
        2,
        std::vector<int>{0, 1, 3, 4},
        std::vector<int>{0, 0, 1, 1},
        std::vector<double>{1.0, -2.0, 0.5, 3.0}};
    const CsrMatrix product = multiplyCsrMatrices(matrix, right_matrix);
    require(product.rows == 3, "CSR product rows mismatch");
    require(product.cols == 2, "CSR product cols mismatch");

    const std::vector<double> product_x = makeDeterministicVector(product.cols, 29);
    const std::vector<double> reference_product_y{
        2.0 * product_x[0] - 3.0 * product_x[1],
        -8.0 * product_x[0] + 2.0 * product_x[1],
        0.5 * product_x[0] + 9.0 * product_x[1]};
    std::vector<double> product_y;
    runCsrSpmvProbe(product, product_x, product_y);
    const OperatorCompareStats product_stats = compareVectors(
        "tiny_spgemm", product, reference_product_y, product_y);
    require(product_stats.rows == 3, "CSR product compare rows mismatch");
    require(product_stats.cols == 2, "CSR product compare cols mismatch");
    require(product_stats.max_abs_error < 1e-12, "CSR product max abs error mismatch");
    require(product_stats.max_rel_error < 1e-12, "CSR product max relative error mismatch");

    const DeviceCsrMatrix device_matrix = makeDeviceCsrFromHost(matrix);
    std::vector<double> device_y;
    runDeviceCsrSpmvProbe(device_matrix, x, device_y);
    require(device_y == y, "device CSR SpMV should match host CSR SpMV");
    const DeviceCsrMatrix device_right = makeDeviceCsrFromHost(right_matrix);
    const DeviceCsrMatrix device_product =
        multiplyDeviceCsrMatrices(device_matrix, device_right);
    const CsrMatrix downloaded_product = copyDeviceCsrToHost(device_product);
    std::vector<double> downloaded_product_y;
    runCsrSpmvProbe(downloaded_product, product_x, downloaded_product_y);
    const OperatorCompareStats device_product_stats = compareVectors(
        "tiny_device_spgemm", downloaded_product, reference_product_y,
        downloaded_product_y);
    require(device_product_stats.max_abs_error < 1e-12,
        "device CSR product max abs error mismatch");
    require(device_product_stats.max_rel_error < 1e-12,
        "device CSR product max relative error mismatch");

    const DeviceCsrMatrix device_transpose =
        transposeDeviceCsrMatrix(device_matrix, -2.0);
    const CsrMatrix downloaded_transpose =
        copyDeviceCsrToHost(device_transpose);
    const CsrMatrix expected_transpose{
        3, 3, std::vector<int>{0, 2, 3, 5},
        std::vector<int>{0, 2, 1, 0, 2},
        std::vector<double>{-4.0, -1.0, -8.0, 2.0, -6.0}};
    const auto transpose_stats = compareCsrMatrices(
        "tiny_device_transpose", downloaded_transpose, expected_transpose);
    require(transpose_stats.max_abs_error < 1e-12,
        "device CSR transpose mismatch");

    const DeviceVector device_x = makeDeviceVectorFromHost(x);
    const DeviceVector device_ax = multiplyDeviceCsrVector(device_matrix, device_x);
    const DeviceVector device_sum = addDeviceVectors(device_ax, -1.0,
        makeDeviceVectorFromHost(reference));
    const std::vector<double> downloaded_sum =
        copyDeviceVectorToHost(device_sum);
    require(std::all_of(downloaded_sum.begin(), downloaded_sum.end(),
                [](double value) { return std::abs(value) < 1e-12; }),
        "device vector SpMV/add mismatch");

    const DeviceCsrMatrix block_device = assembleDeviceCsrBlocks(6, 6, {
        DeviceCsrBlock{device_matrix, 0, 0, 1.0},
        DeviceCsrBlock{device_transpose, 3, 3, 0.5}});
    const CsrMatrix block_host = copyDeviceCsrToHost(block_device);
    require(block_host.rows == 6 && block_host.cols == 6,
        "device block CSR dimensions mismatch");
    require(block_host.values.size() == 10,
        "device block CSR nonzero count mismatch");

    FluidQ1Q0AssemblyInput fluid_input;
    fluid_input.dx = 0.5;
    fluid_input.dt = 0.5;
    fluid_input.rho = 2.0;
    fluid_input.velocity_node_count = 8;
    fluid_input.pressure_node_count = 1;
    fluid_input.face_node_count = 0;
    int node_idx = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k, ++node_idx) {
                fluid_input.velocity_nodes.push_back(AssemblyGridNode{
                    packGridKey3D(i, j, k), node_idx, 2.0 + node_idx,
                    Vec3d{1.0 + node_idx, 2.0 + node_idx, 3.0 + node_idx},
                    Vec3d{}});
            }
    fluid_input.pressure_nodes.push_back(
        AssemblyGridNode{packGridKey3D(0, 0, 0), 0, 1.0, Vec3d{}, Vec3d{}});
    fluid_input.volume_xq.push_back(Vec3d{0.375, 0.375, 0.375});
    fluid_input.volume_mass.push_back(4.0);
    FluidQ1Q0AssemblyResult fluid_result;
    assembleFluidQ1Q0System3D(fluid_input, fluid_result);
    require(fluid_result.m_inv.rows == 24 && fluid_result.m_inv.cols == 24,
        "fluid assembly M_inv dimensions mismatch");
    require(fluid_result.g.rows == 24 && fluid_result.g.cols == 1,
        "fluid assembly G dimensions mismatch");
    require(fluid_result.m_inv.values.size() == 24,
        "fluid assembly M_inv nnz mismatch");
    require(fluid_result.g.values.size() == 24,
        "fluid assembly G nnz mismatch");
    require(fluid_result.a.size() == 24 && fluid_result.b.size() == 1,
        "fluid assembly rhs size mismatch");

    std::vector<double> assembled_g_y;
    runCsrSpmvProbe(fluid_result.g, std::vector<double>{1.0}, assembled_g_y);
    node_idx = 0;
    const double w0 = 0.75;
    const double w1 = 0.25;
    const double dw0 = -1.0;
    const double dw1 = 1.0;
    const double vol = 2.0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k, ++node_idx) {
                const double wx = i == 0 ? w0 : w1;
                const double wy = j == 0 ? w0 : w1;
                const double wz = k == 0 ? w0 : w1;
                const double dwx = i == 0 ? dw0 : dw1;
                const double dwy = j == 0 ? dw0 : dw1;
                const double dwz = k == 0 ? dw0 : dw1;
                const double inv_dx = 1.0 / fluid_input.dx;
                const double expected_g[3] = {
                    -vol * inv_dx * dwx * wy * wz,
                    -vol * inv_dx * wx * dwy * wz,
                    -vol * inv_dx * wx * wy * dwz};
                for (int d = 0; d < 3; ++d) {
                    const int row = node_idx * 3 + d;
                    require(std::abs(assembled_g_y[row] - expected_g[d]) < 1e-12,
                        "fluid assembly G value mismatch");
                    const double expected_a =
                        (2.0 + node_idx) / fluid_input.dt * (1.0 + node_idx + d);
                    require(std::abs(fluid_result.a[static_cast<std::size_t>(row)] -
                                     expected_a) < 1e-12,
                        "fluid assembly rhs value mismatch");
                }
            }

#ifdef ENABLE_CUDA_STAGE2_PROBE
    SolidParticleStateInput3D owner_input;
    Mat3d identity;
    identity.values[0] = identity.values[4] = identity.values[8] = 1.0;
    owner_input.rest_x = {Vec3d{0.75, 0.75, 0.75}};
    owner_input.current_x = {Vec3d{0.8, 0.9, 1.0}};
    owner_input.velocity = {Vec3d{}};
    owner_input.affine = {Mat3d{}};
    owner_input.grad_velocity = {Mat3d{}};
    owner_input.f = {identity};
    owner_input.j = {1.0};
    owner_input.mass = {2.0};
    owner_input.volume = {0.5};
    owner_input.lambda = {4.0};
    PersistentSolidParticleState3D owner;
    SolidParticleStateStats owner_upload_stats;
    initializePersistentSolidParticleState3D(
        owner, owner_input, &owner_upload_stats);
    require(owner_upload_stats.particles == 1,
        "solid particle owner upload count mismatch");
    require(persistentSolidParticleStateValid(owner, 1),
        "solid particle owner should be valid after upload");

    SolidQ2Q0AssemblyInput solid_input;
    solid_input.dx = 1.0;
    solid_input.dt = 0.1;
    solid_input.velocity_node_count = 27;
    solid_input.pressure_node_count = 1;
    solid_input.resident_particles =
        getPersistentSolidParticleStateDeviceView(owner);
    solid_input.compute_f_inv_t = 1;
    node_idx = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k, ++node_idx)
                solid_input.velocity_nodes.push_back(AssemblyGridNode{
                    packGridKey3D(i, j, k), node_idx, 1.0, Vec3d{}, Vec3d{}});
    solid_input.pressure_nodes.push_back(AssemblyGridNode{
        packGridKey3D(0, 0, 0), 0, 1.0, Vec3d{}, Vec3d{}});
    SolidQ2Q0DeviceAssemblyResult solid_device_result;
    assembleSolidQ2Q0System3DDevice(solid_input, solid_device_result);

    std::vector<double> constant_velocity(27 * 3);
    for (int node = 0; node < 27; ++node) {
        constant_velocity[static_cast<std::size_t>(3 * node)] = 1.0;
        constant_velocity[static_cast<std::size_t>(3 * node + 1)] = 2.0;
        constant_velocity[static_cast<std::size_t>(3 * node + 2)] = 3.0;
    }
    const DeviceVector solid_velocity =
        makeDeviceVectorFromHost(constant_velocity);
    const DeviceVector solid_pressure =
        makeDeviceVectorFromHost(std::vector<double>{1.0});
    SolidParticleStateStats owner_update_stats;
    updatePersistentSolidParticlesQ2FromDevice3D(owner,
        solid_device_result.velocity_nodes, solid_device_result.pressure_nodes,
        solid_velocity, solid_pressure, 1.0, 0.1, 1.0, 4.0,
        &owner_update_stats);
    require(owner_update_stats.missing_velocity_nodes == 0,
        "solid particle owner velocity lookup mismatch");
    require(owner_update_stats.missing_pressure_nodes == 0,
        "solid particle owner pressure lookup mismatch");
    SolidParticleStateInput3D owner_output;
    copyPersistentSolidParticleStateToHost(owner, owner_output);
    require(std::abs(owner_output.current_x[0].x - 0.9) < 1e-12 &&
            std::abs(owner_output.current_x[0].y - 1.1) < 1e-12 &&
            std::abs(owner_output.current_x[0].z - 1.3) < 1e-12,
        "solid particle owner position update mismatch");
    require(std::abs(owner_output.velocity[0].x - 1.0) < 1e-12 &&
            std::abs(owner_output.velocity[0].y - 2.0) < 1e-12 &&
            std::abs(owner_output.velocity[0].z - 3.0) < 1e-12,
        "solid particle owner velocity update mismatch");
    require(std::abs(owner_output.j[0] - 0.75) < 1e-12,
        "solid particle owner pressure/J update mismatch");
    for (int entry = 0; entry < 9; ++entry) {
        require(std::abs(owner_output.grad_velocity[0].values[entry]) < 1e-12,
            "solid particle owner constant-field gradient mismatch");
        require(std::abs(owner_output.affine[0].values[entry]) < 1e-12,
            "solid particle owner constant-field affine mismatch");
        require(std::abs(owner_output.f[0].values[entry] -
                         identity.values[entry]) < 1e-12,
            "solid particle owner deformation update mismatch");
    }
    // Before delayed fluid injection there is no pressure solve. Reproduce
    // an affine grid field with reversed DOF indices to check new_v lookup,
    // advection, and the analytic det(F) update without a pressure map.
    PersistentSolidParticleState3D grid_only_owner;
    initializePersistentSolidParticleState3D(grid_only_owner, owner_input);
    SolidQ2Q0AssemblyInput grid_only_input = solid_input;
    grid_only_input.resident_particles =
        getPersistentSolidParticleStateDeviceView(grid_only_owner);
    for (auto& node : grid_only_input.velocity_nodes) {
        const int original_idx = node.idx;
        const int x = original_idx / 9;
        const int y = (original_idx / 3) % 3;
        const int z = original_idx % 3;
        node.idx = 26 - original_idx;
        node.new_v = Vec3d{1.0 + x, 2.0 + 2.0 * y, 3.0 + 3.0 * z};
    }
    SolidQ2Q0DeviceAssemblyResult grid_only_assembly;
    assembleSolidQ2Q0System3DDevice(grid_only_input, grid_only_assembly);
    SolidParticleStateStats grid_only_stats;
    updatePersistentSolidParticlesQ2FromGrid3D(grid_only_owner,
        grid_only_assembly.velocity_nodes, 1.0, 0.1, 1.0, 4.0,
        &grid_only_stats);
    SolidParticleStateInput3D grid_only_output;
    copyPersistentSolidParticleStateToHost(grid_only_owner, grid_only_output);
    require(grid_only_stats.missing_velocity_nodes == 0 &&
            grid_only_stats.missing_pressure_nodes == 0 &&
            grid_only_stats.pressure_nodes == 0,
        "solid-only Q2 update must not require pressure nodes");
    require(std::abs(grid_only_output.current_x[0].x - 0.975) < 1e-12 &&
            std::abs(grid_only_output.current_x[0].y - 1.25) < 1e-12 &&
            std::abs(grid_only_output.current_x[0].z - 1.525) < 1e-12,
        "solid-only Q2 affine-field advection mismatch");
    require(std::abs(grid_only_output.j[0] - 1.716) < 1e-12,
        "solid-only Q2 J must equal the updated deformation determinant");

    std::vector<AssemblyGridNode> p2g_nodes;
    SolidVelocityGridStats p2g_stats;
    buildPersistentSolidVelocityGridQ2ToHost(
        owner, 1.0, p2g_nodes, &p2g_stats);
    require(p2g_nodes.size() == 27 && p2g_stats.nodes == 27,
        "solid particle owner Q2 P2G node count mismatch");
    double p2g_mass = 0.0;
    for (const auto& node : p2g_nodes) {
        p2g_mass += node.mass;
        require(std::abs(node.v.x - 1.0) < 1e-12 &&
                std::abs(node.v.y - 2.0) < 1e-12 &&
                std::abs(node.v.z - 3.0) < 1e-12,
            "solid particle owner Q2 P2G velocity mismatch");
    }
    require(std::abs(p2g_mass - 2.0) < 1e-12,
        "solid particle owner Q2 P2G mass mismatch");
    for (std::size_t i = 0; i < p2g_nodes.size(); ++i)
        p2g_nodes[i].idx = static_cast<int>(i);
    PersistentSolidParticleState3D seeded_owner;
    initializePersistentSolidParticleState3D(seeded_owner, owner_input);
    std::vector<AssemblyGridNode> seeded_p2g_nodes;
    SolidVelocityGridStats seeded_p2g_stats;
    buildPersistentSolidVelocityGridQ2ToHost(seeded_owner, 1.0,
        seeded_p2g_nodes, &seeded_p2g_stats, false, &p2g_nodes);
    require(seeded_p2g_stats.seeded_legacy_topology &&
            seeded_p2g_stats.reused_static_topology &&
            seeded_p2g_stats.host_download_bytes == 0 &&
            seeded_p2g_stats.topology_download_bytes == 0 &&
            seeded_p2g_stats.velocity_download_bytes == 0,
        "solid particle owner Q2 legacy topology seed should avoid D2H");
    const DeviceGridNodeMap seeded_p2g_map =
        getPersistentSolidVelocityGridQ2DeviceMap(seeded_owner);
    require(seeded_p2g_nodes.empty() &&
            seeded_p2g_map.size == p2g_nodes.size(),
        "solid particle owner seeded Q2 map size mismatch");
    std::vector<AssemblyGridNode> materialized_p2g_nodes;
    SolidVelocityGridStats materialized_p2g_stats;
    buildPersistentSolidVelocityGridQ2ToHost(seeded_owner, 1.0,
        materialized_p2g_nodes, &materialized_p2g_stats, true);
    require(materialized_p2g_nodes.size() == p2g_nodes.size() &&
            materialized_p2g_stats.reused_static_topology &&
            materialized_p2g_stats.topology_download_bytes ==
                p2g_nodes.size() * sizeof(AssemblyGridNode),
        "solid particle owner Q2 fallback topology materialization mismatch");
    invalidatePersistentSolidParticleState3D(seeded_owner);
    PersistentSolidParticleState3D canonical_owner;
    initializePersistentSolidParticleState3D(canonical_owner, owner_input);
    std::vector<AssemblyGridNode> canonical_p2g_nodes;
    SolidVelocityGridStats canonical_p2g_stats;
    buildPersistentSolidVelocityGridQ2ToHost(canonical_owner, 1.0,
        canonical_p2g_nodes, &canonical_p2g_stats, false, nullptr, true);
    require(canonical_p2g_nodes.empty() &&
            canonical_p2g_stats.assigned_canonical_topology &&
            !canonical_p2g_stats.reused_static_topology &&
            canonical_p2g_stats.host_download_bytes == 0,
        "solid particle owner canonical Q2 topology should stay on device");
    buildPersistentSolidVelocityGridQ2ToHost(canonical_owner, 1.0,
        canonical_p2g_nodes, &canonical_p2g_stats, true);
    require(canonical_p2g_nodes.size() == p2g_nodes.size() &&
            canonical_p2g_stats.reused_static_topology,
        "solid particle owner canonical Q2 topology reuse mismatch");
    for (std::size_t i = 0; i < canonical_p2g_nodes.size(); ++i)
        require(canonical_p2g_nodes[i].idx == static_cast<int>(i),
            "solid particle owner canonical Q2 index mismatch");
    InterfaceWAssemblyInput3D interface_w_input;
    interface_w_input.dx = 1.0;
    interface_w_input.velocity_degree = 2;
    interface_w_input.shift_velocity_half_dx = false;
    interface_w_input.velocity_node_count =
        static_cast<int>(canonical_p2g_nodes.size());
    interface_w_input.resident_velocity_nodes =
        getPersistentSolidVelocityGridQ2DeviceMap(canonical_owner);
    interface_w_input.rest_x = {Vec3d{0.75, 0.75, 0.75}};
    interface_w_input.current_x = {Vec3d{0.75, 0.75, 0.75}};
    interface_w_input.normal = {Vec3d{1.0, 2.0, 3.0}};
    interface_w_input.length = {2.0};
    DeviceCsrMatrix interface_w_device;
    assembleInterfaceW3DDevice(interface_w_input, interface_w_device);
    CsrMatrix interface_w = copyDeviceCsrToHost(interface_w_device);
    require(interface_w.rows == 1 && interface_w.cols == 81 &&
            interface_w.values.size() == 81,
        "solid canonical interface W shape mismatch");
    double interface_component_sum[3] = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < interface_w.values.size(); ++i)
        interface_component_sum[interface_w.col_indices[i] % 3] +=
            interface_w.values[i];
    require(std::abs(interface_component_sum[0] - 2.0) < 1e-12 &&
            std::abs(interface_component_sum[1] - 4.0) < 1e-12 &&
            std::abs(interface_component_sum[2] - 6.0) < 1e-12,
        "solid canonical interface W partition mismatch");
    invalidatePersistentSolidParticleState3D(canonical_owner);
    std::vector<Vec3d> owner_current_x;
    copyPersistentSolidCurrentXToHost(owner, owner_current_x);
    require(owner_current_x.size() == 1 &&
            std::abs(owner_current_x[0].x - 0.9) < 1e-12,
        "solid particle owner current-X selective download mismatch");
    SolidParticleReductionStats reduction_stats;
    reducePersistentSolidParticleState3D(owner, reduction_stats);
    require(reduction_stats.particles == 1 &&
            std::abs(reduction_stats.max_speed - std::sqrt(14.0)) < 1e-12 &&
            std::abs(reduction_stats.min_corner.x - 0.75) < 1e-12 &&
            std::abs(reduction_stats.max_corner.z - 0.75) < 1e-12,
        "solid particle owner reduction mismatch");
    invalidatePersistentSolidParticleState3D(owner);
    require(!persistentSolidParticleStateValid(owner, 1),
        "solid particle owner invalidation mismatch");
#endif

    const std::vector<double> p1{1.0, 2.0};
    const std::vector<double> alpha{3.0, 4.0, 5.0};
    const std::vector<double> p2{6.0};
    const std::vector<double> block = concatenateBlockVector(p1, alpha, p2);
    require((block == std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
        "block concatenation should preserve order");

    std::vector<double> out_p1;
    std::vector<double> out_alpha;
    std::vector<double> out_p2;
    splitBlockVector(block, 2, 3, 1, out_p1, out_alpha, out_p2);
    require(out_p1 == p1, "block split p1 mismatch");
    require(out_alpha == alpha, "block split alpha mismatch");
    require(out_p2 == p2, "block split p2 mismatch");

    const std::vector<double> deterministic_a = makeDeterministicVector(8, 17);
    const std::vector<double> deterministic_b = makeDeterministicVector(8, 17);
    require(deterministic_a == deterministic_b, "deterministic vectors should be reproducible");
    require(deterministic_a.size() == 8, "deterministic vector size mismatch");

    const CsrMatrix spd{
        3,
        3,
        std::vector<int>{0, 2, 5, 7},
        std::vector<int>{0, 1, 0, 1, 2, 1, 2},
        std::vector<double>{4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0}};
    const std::vector<double> rhs{1.0, 2.0, 3.0};
    std::vector<double> solved;
    ConjugateGradientStats cg_stats;
    runConjugateGradientProbe(spd, rhs, 1e-12, 100, solved, cg_stats);
    require(cg_stats.converged, "CG probe should converge on a tiny SPD system");
    require(cg_stats.iterations <= 3, "CG probe should solve 3x3 SPD in at most n iterations");
    require(std::abs(solved[0] - 0.2222222222222222) < 1e-10, "CG solution x0 mismatch");
    require(std::abs(solved[1] - 0.1111111111111111) < 1e-10, "CG solution x1 mismatch");
    require(std::abs(solved[2] - 1.4444444444444444) < 1e-10, "CG solution x2 mismatch");
    require(cg_stats.relative_residual < 1e-10, "CG residual should be small");

    std::vector<double> swept_solved;
    ConjugateGradientStats swept_cg_stats;
    runConjugateGradientProbe(spd, rhs, 1e-12, 100, swept_solved, swept_cg_stats,
        /*jacobi_sweeps*/ 4, /*jacobi_omega*/ 0.8);
    require(swept_cg_stats.converged, "swept Jacobi CG probe should converge on a tiny SPD system");
    require(std::abs(swept_solved[0] - 0.2222222222222222) < 1e-10, "swept CG solution x0 mismatch");
    require(std::abs(swept_solved[1] - 0.1111111111111111) < 1e-10, "swept CG solution x1 mismatch");
    require(std::abs(swept_solved[2] - 1.4444444444444444) < 1e-10, "swept CG solution x2 mismatch");

    const Eigen::SparseMatrix<double> S1 = makeSparse(2, 2, {
        {0, 0, 0.25}, {1, 1, 0.5}, {0, 1, -0.1}});
    const Eigen::SparseMatrix<double> D1 = makeSparse(2, 3, {
        {0, 0, 1.0}, {0, 2, -0.5}, {1, 1, 0.75}, {1, 2, 0.25}});
    const Eigen::SparseMatrix<double> Minv1 = makeSparse(3, 3, {
        {0, 0, 2.0}, {1, 1, 0.5}, {2, 2, 1.5}});
    const Eigen::SparseMatrix<double> G1 = makeSparse(3, 2, {
        {0, 0, -0.3}, {1, 0, 0.8}, {1, 1, -0.6}, {2, 1, 0.4}});
    const Eigen::SparseMatrix<double> W1 = makeSparse(2, 3, {
        {0, 0, 0.2}, {0, 2, -0.7}, {1, 1, 0.9}});

    const Eigen::SparseMatrix<double> S2 = makeSparse(2, 2, {
        {0, 0, 0.1}, {1, 1, 0.3}, {1, 0, 0.05}});
    const Eigen::SparseMatrix<double> D2 = makeSparse(2, 3, {
        {0, 0, -0.4}, {0, 1, 0.6}, {1, 1, -0.2}, {1, 2, 0.5}});
    const Eigen::SparseMatrix<double> Minv2 = makeSparse(3, 3, {
        {0, 0, 1.25}, {1, 1, 0.75}, {2, 2, 1.1}});
    const Eigen::SparseMatrix<double> G2 = makeSparse(3, 2, {
        {0, 0, 0.55}, {1, 0, -0.45}, {1, 1, 0.35}, {2, 1, -0.25}});
    const Eigen::SparseMatrix<double> W2 = makeSparse(2, 3, {
        {0, 1, -0.3}, {1, 0, 0.45}, {1, 2, 0.15}});

    Eigen::SparseMatrix<double> A00 = S1 - D1 * Minv1 * G1;
    Eigen::SparseMatrix<double> A01 = -D1 * Minv1 * W1.transpose();
    Eigen::SparseMatrix<double> A10 = W1 * Minv1 * G1;
    Eigen::SparseMatrix<double> A11 = W1 * Minv1 * W1.transpose() +
        W2 * Minv2 * W2.transpose();
    Eigen::SparseMatrix<double> A12 = -W2 * Minv2 * G2;
    Eigen::SparseMatrix<double> A21 = D2 * Minv2 * W2.transpose();
    Eigen::SparseMatrix<double> A22 = S2 - D2 * Minv2 * G2;

    std::vector<std::vector<std::pair<int, double>>> block_rows(6);
    addEigenBlock(0, 0, S1, block_rows);
    const CsrMatrix D1Minv = multiplyCsrMatrices(makeCsrFromEigen(D1), makeCsrFromEigen(Minv1));
    const CsrMatrix D1MinvG = multiplyCsrMatrices(D1Minv, makeCsrFromEigen(G1));
    Eigen::SparseMatrix<double> W1T = W1.transpose();
    const CsrMatrix D1MinvWT = multiplyCsrMatrices(D1Minv, makeCsrFromEigen(W1T));
    const CsrMatrix W1Minv = multiplyCsrMatrices(makeCsrFromEigen(W1), makeCsrFromEigen(Minv1));
    const CsrMatrix W1MinvG = multiplyCsrMatrices(W1Minv, makeCsrFromEigen(G1));
    const CsrMatrix W1MinvWT = multiplyCsrMatrices(W1Minv, makeCsrFromEigen(W1T));

    const CsrMatrix D2Minv = multiplyCsrMatrices(makeCsrFromEigen(D2), makeCsrFromEigen(Minv2));
    const CsrMatrix D2MinvG = multiplyCsrMatrices(D2Minv, makeCsrFromEigen(G2));
    Eigen::SparseMatrix<double> W2T = W2.transpose();
    const CsrMatrix D2MinvWT = multiplyCsrMatrices(D2Minv, makeCsrFromEigen(W2T));
    const CsrMatrix W2Minv = multiplyCsrMatrices(makeCsrFromEigen(W2), makeCsrFromEigen(Minv2));
    const CsrMatrix W2MinvG = multiplyCsrMatrices(W2Minv, makeCsrFromEigen(G2));
    const CsrMatrix W2MinvWT = multiplyCsrMatrices(W2Minv, makeCsrFromEigen(W2T));

    addCsrBlock(0, 0, D1MinvG, -1.0, block_rows);
    addCsrBlock(0, 2, D1MinvWT, -1.0, block_rows);
    addCsrBlock(2, 0, W1MinvG, 1.0, block_rows);
    addCsrBlock(2, 2, W1MinvWT, 1.0, block_rows);
    addCsrBlock(2, 2, W2MinvWT, 1.0, block_rows);
    addCsrBlock(2, 4, W2MinvG, -1.0, block_rows);
    addCsrBlock(4, 2, D2MinvWT, 1.0, block_rows);
    addEigenBlock(4, 4, S2, block_rows);
    addCsrBlock(4, 4, D2MinvG, -1.0, block_rows);
    const CsrMatrix assembled = finalizeRows(6, 6, block_rows);

    std::vector<std::vector<std::pair<int, double>>> ref_rows(6);
    addEigenBlock(0, 0, A00, ref_rows);
    addEigenBlock(0, 2, A01, ref_rows);
    addEigenBlock(2, 0, A10, ref_rows);
    addEigenBlock(2, 2, A11, ref_rows);
    addEigenBlock(2, 4, A12, ref_rows);
    addEigenBlock(4, 2, A21, ref_rows);
    addEigenBlock(4, 4, A22, ref_rows);
    const CsrMatrix reference_block = finalizeRows(6, 6, ref_rows);

    const std::vector<double> block_x = makeDeterministicVector(6, 401);
    std::vector<double> assembled_y;
    std::vector<double> reference_y;
    runCsrSpmvProbe(assembled, block_x, assembled_y);
    runCsrSpmvProbe(reference_block, block_x, reference_y);
    const OperatorCompareStats block_stats = compareVectors(
        "stage3b_nonzero_w_block", assembled, reference_y, assembled_y);
    require(block_stats.max_abs_error < 1e-12,
        "Stage3B nonzero W block max abs error mismatch");
    require(block_stats.max_rel_error < 1e-12,
        "Stage3B nonzero W block max relative error mismatch");

    return 0;
}
