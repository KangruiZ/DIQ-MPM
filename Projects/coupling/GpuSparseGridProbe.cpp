#include "GpuSparseGridProbe.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <map>
#include <set>
#include <stdexcept>

namespace ZIRAN::GPU_STAGE2 {
namespace {

constexpr int kCoordBits = 21;
constexpr int kCoordBias = 1 << (kCoordBits - 1);
constexpr GridKey kCoordMask = (GridKey{1} << kCoordBits) - 1;

int floorToInt(double value)
{
    return static_cast<int>(std::floor(value));
}

void linearWeights(double x, double dx, int& base, double w[2])
{
    const double grid_x = x / dx;
    base = floorToInt(grid_x);
    const double t = grid_x - static_cast<double>(base);
    w[0] = 1.0 - t;
    w[1] = t;
}

int lowerBoundKey(const std::vector<GridKey>& keys, GridKey key)
{
    return static_cast<int>(std::lower_bound(keys.begin(), keys.end(), key) - keys.begin());
}

GridKey order0Key(const Vec3d& x, double dx, double anchor)
{
    return packGridKey3D(
        floorToInt((x.x + anchor) / dx),
        floorToInt((x.y + anchor) / dx),
        floorToInt((x.z + anchor) / dx));
}

void scatterUnitCounts(const std::vector<GridKey>& samples, ProbeGridResult& result)
{
    result.keys = samples;
    sortUniqueKeys(result.keys);
    result.mass.assign(result.keys.size(), 0.0);
    for (GridKey key : samples) {
        const int dof = lowerBoundKey(result.keys, key);
        result.mass[dof] += 1.0;
    }
}

struct PersistentSpatialSearchStorage {
    std::vector<Vec3d> positions;
    double cell_size = 0.0;
    std::size_t reallocations = 0;
};

struct PersistentPressureGridStorage {
    PersistentPagePollute3D pages;
    PersistentLinearScatter3D density;
    PersistentPressureCsr3D csr;
    std::vector<Vec3d> page_positions;
    std::vector<GridKey> dof_source_keys;
    ProbeGridResult density_result;
    std::vector<GridKey> ordered_dof_keys;
    std::vector<GridKey> touched_keys;
    std::vector<unsigned char> solid_flags;
    std::size_t reallocations = 0;
};

PersistentSpatialSearchStorage& getSpatialStorage(
    PersistentSpatialSearch3D& context)
{
    if (!context.storage)
        context.storage = std::make_shared<PersistentSpatialSearchStorage>();
    return *std::static_pointer_cast<PersistentSpatialSearchStorage>(context.storage);
}

PersistentPressureGridStorage& getPressureGridStorage(
    PersistentPressureGrid3D& context)
{
    if (!context.storage)
        context.storage = std::make_shared<PersistentPressureGridStorage>();
    return *std::static_pointer_cast<PersistentPressureGridStorage>(context.storage);
}

template <class T>
void resizePressureField(std::vector<T>& values, std::size_t size,
    PersistentPressureGridStorage& storage)
{
    const std::size_t old_capacity = values.capacity();
    values.resize(size);
    if (values.capacity() != old_capacity)
        ++storage.reallocations;
}

} // namespace

GridKey packGridKey3D(int x, int y, int z)
{
    const int bx = x + kCoordBias;
    const int by = y + kCoordBias;
    const int bz = z + kCoordBias;
    if (bx < 0 || by < 0 || bz < 0 || bx > static_cast<int>(kCoordMask) || by > static_cast<int>(kCoordMask) || bz > static_cast<int>(kCoordMask))
        throw std::out_of_range("grid coordinate is outside packGridKey3D range");
    return (static_cast<GridKey>(bx) << (2 * kCoordBits)) |
        (static_cast<GridKey>(by) << kCoordBits) |
        static_cast<GridKey>(bz);
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

    std::vector<GridKey> candidate_keys;
    candidate_keys.reserve(xq.size() * 8);
    const double anchor = halfDxAnchor ? -0.5 * dx : 0.0;

    for (const Vec3d& x : xq) {
        int base[3];
        double w[3][2];
        linearWeights(x.x + anchor, dx, base[0], w[0]);
        linearWeights(x.y + anchor, dx, base[1], w[1]);
        linearWeights(x.z + anchor, dx, base[2], w[2]);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                    candidate_keys.push_back(packGridKey3D(base[0] + i, base[1] + j, base[2] + k));
    }

    sortUniqueKeys(candidate_keys);
    result.keys = candidate_keys;
    result.mass.assign(result.keys.size(), 0.0);

    for (std::size_t q = 0; q < xq.size(); ++q) {
        int base[3];
        double w[3][2];
        linearWeights(xq[q].x + anchor, dx, base[0], w[0]);
        linearWeights(xq[q].y + anchor, dx, base[1], w[1]);
        linearWeights(xq[q].z + anchor, dx, base[2], w[2]);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    const GridKey key = packGridKey3D(base[0] + i, base[1] + j, base[2] + k);
                    const int dof = lowerBoundKey(result.keys, key);
                    result.mass[dof] += values[q] * w[0][i] * w[1][j] * w[2][k];
                }
    }
}

void runPersistentUniformLinearScatter3D(
    PersistentLinearScatter3D&, const std::vector<Vec3d>& xq,
    double value, double dx, bool halfDxAnchor, ProbeGridResult& result,
    LinearScatterStats* stats)
{
    std::vector<double> values(xq.size(), value);
    runLinearScalarScatterProbe(xq, values, dx, halfDxAnchor, result);
    if (stats) {
        stats->sources = xq.size();
        stats->nodes = result.keys.size();
        stats->candidate_keys = xq.size() * 8;
    }
}

void runPersistentLinearBaseKey3D(
    PersistentPagePollute3D&, const std::vector<Vec3d>& xq,
    double dx, bool halfDxAnchor, std::vector<GridKey>& base_keys,
    PagePolluteStats* stats)
{
    if (!(dx > 0.0))
        throw std::runtime_error("persistent page pollute requires positive dx");

    const auto total_begin = std::chrono::high_resolution_clock::now();
    base_keys.clear();
    base_keys.reserve(xq.size());
    const double anchor = halfDxAnchor ? -0.5 * dx : 0.0;
    for (const Vec3d& x : xq)
        base_keys.push_back(order0Key(x, dx, anchor));

    if (stats) {
        stats->sources = xq.size();
        stats->base_nodes = base_keys.size();
        stats->total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - total_begin).count();
    }
}

void transferFluidVelocityQ1FromDevice3D(
    PersistentVelocityTransfer3D&, PersistentFluidAnalytic3D*,
    const GPU_STAGE3A::DeviceGridNodeMap*,
    const GPU_STAGE3A::DeviceGridNodeMap*, const void*, std::size_t,
    const std::vector<GridKey>&, const std::vector<int>&,
    const std::vector<GridKey>&, const std::vector<Vec3d>&,
    const GPU_STAGE3A::DeviceVector&, const std::vector<GridKey>&,
    const std::vector<int>&, const GPU_STAGE3A::DeviceVector&, double, double,
    VelocityTransferHostView&, VelocityTransferDeviceView&,
    VelocityTransferStats*, VelocityTransferDownloadMask)
{
    throw std::runtime_error("fluid velocity transfer requires the CUDA implementation");
}

void downloadPersistentFluidVelocityTransfer3D(
    PersistentVelocityTransfer3D&, VelocityTransferDownloadMask,
    VelocityTransferHostView&, VelocityTransferDownloadStats*)
{
    throw std::runtime_error(
        "fluid velocity-transfer download requires the CUDA implementation");
}

void downloadFluidVelocityTransferDeviceView3D(
    PersistentVelocityTransfer3D&, const VelocityTransferDeviceView&,
    VelocityTransferDownloadMask, VelocityTransferHostView&,
    VelocityTransferDownloadStats*)
{
    throw std::runtime_error(
        "fluid velocity-transfer view download requires the CUDA implementation");
}

void projectFluidBoundaryCollision3D(
    PersistentVelocityTransfer3D&, const BoxShellCollision3D&, double,
    FluidBoundaryProjectionStats*)
{
    throw std::runtime_error(
        "fluid boundary projection requires the CUDA implementation");
}

void reduceFluidParticleState3D(
    const VelocityTransferDeviceView&, double, double,
    FluidParticleStateStats&)
{
    throw std::runtime_error(
        "fluid particle-state reduction requires the CUDA implementation");
}

void patchFluidVelocityTransferParticles3D(
    PersistentVelocityTransfer3D&, const std::vector<int>&,
    const std::vector<Vec3d>&, const std::vector<Mat3d>&,
    VelocityTransferPatchStats*, const std::vector<Vec3d>&)
{
    throw std::runtime_error(
        "fluid velocity-transfer patch requires the CUDA implementation");
}

void rebuildPersistentPressureGridPages3D(
    PersistentPressureGrid3D& context, const std::vector<Vec3d>& page_positions,
    double dx, std::vector<GridKey>& page_base_keys,
    PressureGridFieldStats* stats, bool download_base_keys,
    const void* resident_page_positions, std::size_t resident_page_count)
{
    if (resident_page_positions || resident_page_count != 0)
        throw std::runtime_error(
            "resident pressure page input requires the CUDA implementation");
    auto& storage = getPressureGridStorage(context);
    storage.page_positions = page_positions;
    PagePolluteStats page_stats;
    runPersistentLinearBaseKey3D(storage.pages, page_positions, dx,
        /*halfDxAnchor*/ false, page_base_keys, &page_stats);
    if (!download_base_keys)
        page_base_keys.clear();
    if (stats) {
        stats->page_sources = page_stats.sources;
        stats->reallocations = storage.reallocations + page_stats.reallocations;
        stats->upload_ms = page_stats.upload_ms;
        stats->key_generation_ms = page_stats.key_generation_ms;
        stats->download_ms = page_stats.download_ms;
        stats->total_ms = page_stats.total_ms;
    }
}

void buildPersistentPressureGridVirtualBlocks3D(
    PersistentPressureGrid3D& context, int block_x, int block_y, int block_z,
    VirtualBlockOrderResult& result, VirtualBlockOrderStats* stats,
    bool download_result)
{
    // CPU fallback cannot infer dx from the context. It is only used in the
    // non-CUDA build, where callers retain the ordinary CPU/SPGrid path.
    (void)context;
    (void)block_x;
    (void)block_y;
    (void)block_z;
    (void)result;
    (void)stats;
    (void)download_result;
    throw std::runtime_error(
        "persistent GPU virtual blocks require the CUDA implementation");
}

void extractPersistentPressureGridDofSourceKeys3D(
    PersistentPressureGrid3D& context, double dx,
    std::vector<GridKey>& dof_source_keys, PressureGridFieldStats* stats,
    bool download_keys)
{
    auto& storage = getPressureGridStorage(context);
    PagePolluteStats page_stats;
    runPersistentLinearBaseKey3D(storage.pages, storage.page_positions, dx,
        /*halfDxAnchor*/ true, dof_source_keys, &page_stats);
    sortUniqueKeys(dof_source_keys);
    storage.dof_source_keys = dof_source_keys;
    if (!download_keys)
        dof_source_keys.clear();
    if (stats) {
        stats->page_sources = page_stats.sources;
        stats->reallocations = storage.reallocations + page_stats.reallocations;
        stats->upload_ms = page_stats.upload_ms;
        stats->key_generation_ms = page_stats.key_generation_ms;
        stats->download_ms = page_stats.download_ms;
        stats->total_ms = page_stats.total_ms;
    }
}

void classifyPersistentPressureGridSolid3D(
    PersistentPressureGrid3D&,
    const GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D&,
    const BoxShellCollision3D&, double,
    std::vector<unsigned char>&, PressureGridFieldStats*, bool)
{
    throw std::runtime_error(
        "resident pressure solid classification requires the CUDA implementation");
}

void scatterPersistentPressureGridDensity3D(
    PersistentPressureGrid3D& context,
    const std::vector<Vec3d>& density_positions, double density_value,
    double dx, ProbeGridResult& density_result,
    PressureGridFieldStats* stats, bool download_result)
{
    auto& storage = getPressureGridStorage(context);
    LinearScatterStats density_stats;
    runPersistentUniformLinearScatter3D(storage.density, density_positions,
        density_value, dx, /*halfDxAnchor*/ false, density_result,
        &density_stats);
    storage.density_result = density_result;
    if (!download_result) {
        density_result.keys.clear();
        density_result.mass.clear();
    }
    if (stats) {
        stats->density_sources = density_stats.sources;
        stats->density_nodes = density_stats.nodes;
        stats->reallocations = storage.reallocations + density_stats.reallocations;
        stats->upload_ms = density_stats.upload_ms;
        stats->key_generation_ms = density_stats.key_sort_ms;
        stats->density_scatter_ms = density_stats.scatter_ms;
        stats->download_ms = density_stats.download_ms;
        stats->total_ms = density_stats.total_ms;
    }
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
    if (resident_dynamic_positions || resident_dynamic_count != 0 ||
        second_resident_dynamic_positions ||
        second_resident_dynamic_count != 0)
        throw std::runtime_error(
            "resident pressure density input requires the CUDA implementation");
    std::vector<Vec3d> merged;
    merged.reserve(static_density_positions.size() + dynamic_density_positions.size());
    merged.insert(merged.end(), static_density_positions.begin(),
        static_density_positions.end());
    merged.insert(merged.end(), dynamic_density_positions.begin(),
        dynamic_density_positions.end());
    scatterPersistentPressureGridDensity3D(context, merged, density_value, dx,
        density_result, stats, download_result);
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
    if (reuse_resident_virtual_touched_keys || rebuild_resident_legacy_dof_keys ||
        reuse_resident_solid_flags)
        throw std::runtime_error(
            "resident pressure topology requires the CUDA implementation");
    if (touched_keys.size() != solid_flags.size())
        throw std::runtime_error("persistent pressure-grid solid field size mismatch");
    const auto begin = std::chrono::high_resolution_clock::now();
    auto& storage = getPressureGridStorage(context);
    resizePressureField(storage.ordered_dof_keys, ordered_dof_keys.size(), storage);
    resizePressureField(storage.touched_keys, touched_keys.size(), storage);
    resizePressureField(storage.solid_flags, solid_flags.size(), storage);
    storage.ordered_dof_keys = ordered_dof_keys;
    storage.touched_keys = touched_keys;
    storage.solid_flags = solid_flags;
    if (stats) {
        stats->dofs = ordered_dof_keys.size();
        stats->touched_nodes = touched_keys.size();
        stats->resident_touched_input = false;
        stats->resident_dof_input = false;
        stats->resident_solid_flags = false;
        stats->reallocations = storage.reallocations;
        stats->total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - begin).count();
    }
}

void assemblePersistentPressureGridCsr3D(
    PersistentPressureGrid3D& context, double rho_rest, double dx, double dt,
    PressureCsrResult& result, PressureCsrStats* stats, bool, bool)
{
    auto& storage = getPressureGridStorage(context);
    if (storage.touched_keys.size() != storage.solid_flags.size())
        throw std::runtime_error("persistent pressure-grid solid field size mismatch");
    std::vector<double> rho_star(storage.ordered_dof_keys.size(), 0.0);
    for (std::size_t row = 0; row < storage.ordered_dof_keys.size(); ++row) {
        int x, y, z;
        unpackGridKey3D(storage.ordered_dof_keys[row], x, y, z);
        const GridKey density_key = packGridKey3D(x + 1, y + 1, z + 1);
        const auto it = std::lower_bound(storage.density_result.keys.begin(),
            storage.density_result.keys.end(), density_key);
        if (it != storage.density_result.keys.end() && *it == density_key)
            rho_star[row] = storage.density_result.mass[static_cast<std::size_t>(
                it - storage.density_result.keys.begin())];
    }
    assemblePersistentPressureCsr3D(storage.csr, storage.ordered_dof_keys,
        storage.touched_keys, storage.solid_flags, rho_star, rho_rest, dx, dt,
        result, stats);
}

void assemblePersistentPressureCsr3D(
    PersistentPressureCsr3D&, const std::vector<GridKey>& ordered_dof_keys,
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
    std::vector<std::pair<GridKey, unsigned char>> solids;
    solids.reserve(solid_keys.size());
    for (std::size_t i = 0; i < solid_keys.size(); ++i)
        solids.emplace_back(solid_keys[i], solid_flags[i]);
    std::sort(solids.begin(), solids.end());
    std::vector<std::pair<GridKey, int>> dofs;
    dofs.reserve(ordered_dof_keys.size());
    for (std::size_t i = 0; i < ordered_dof_keys.size(); ++i)
        dofs.emplace_back(ordered_dof_keys[i], static_cast<int>(i));
    std::sort(dofs.begin(), dofs.end());
    auto lookup_dof = [&](GridKey key) {
        const auto it = std::lower_bound(dofs.begin(), dofs.end(),
            std::make_pair(key, std::numeric_limits<int>::min()));
        return it != dofs.end() && it->first == key ? it->second : -1;
    };
    std::size_t solid_misses = 0;
    auto is_solid = [&](GridKey key) {
        const auto it = std::lower_bound(solids.begin(), solids.end(),
            std::make_pair(key, static_cast<unsigned char>(0)));
        if (it == solids.end() || it->first != key) {
            ++solid_misses;
            return false;
        }
        return it->second != 0;
    };
    result.row_offsets.assign(ordered_dof_keys.size() + 1, 0);
    result.column_indices.clear();
    result.values.clear();
    result.rhs.assign(ordered_dof_keys.size(), 0.0);
    result.column_indices.reserve(ordered_dof_keys.size() * 7);
    result.values.reserve(ordered_dof_keys.size() * 7);
    for (std::size_t row = 0; row < ordered_dof_keys.size(); ++row) {
        int x, y, z;
        unpackGridKey3D(ordered_dof_keys[row], x, y, z);
        bool near_air = false;
        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k <= 1; ++k) {
                    const GridKey key = packGridKey3D(x + i, y + j, z + k);
                    if (!is_solid(key) && lookup_dof(key) < 0)
                        near_air = true;
                }
        int diagonal = 0;
        int cols[6];
        int col_count = 0;
        for (int axis = 0; axis < 3; ++axis)
            for (int delta : {-1, 1}) {
                int nx = x, ny = y, nz = z;
                if (axis == 0) nx += delta;
                else if (axis == 1) ny += delta;
                else nz += delta;
                const GridKey key = packGridKey3D(nx, ny, nz);
                if (!is_solid(key)) {
                    ++diagonal;
                    const int col = lookup_dof(key);
                    if (col < 0)
                        near_air = true;
                    else
                        cols[col_count++] = col;
                }
            }
        result.column_indices.push_back(static_cast<int>(row));
        result.values.push_back(static_cast<double>(diagonal));
        for (int i = 0; i < col_count; ++i) {
            result.column_indices.push_back(cols[i]);
            result.values.push_back(-1.0);
        }
        double rho_star = rho_star_by_row[row];
        if (near_air)
            rho_star = std::max(rho_star, rho_rest);
        rho_star = std::max(rho_star, rho_rest * 0.5);
        rho_star = std::min(rho_star, rho_rest * 1.5);
        result.rhs[row] = -(rho_rest - rho_star) * dx * dx / dt / dt;
        result.row_offsets[row + 1] = static_cast<int>(result.values.size());
    }
    if (stats) {
        stats->rows = ordered_dof_keys.size();
        stats->nnz = result.values.size();
        stats->solid_lookup_misses = solid_misses;
        stats->total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - total_begin).count();
    }
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
    std::vector<GridKey> samples;
    samples.reserve(xq.size());
    const double anchor = halfDxAnchor ? -0.5 * dx : 0.0;
    for (const Vec3d& x : xq)
        samples.push_back(order0Key(x, dx, anchor));
    scatterUnitCounts(samples, result);
}

void runFaceGridCountProbe(const std::vector<Vec3d>& face_xq,
    const std::vector<Vec3d>& face_n, double dx, bool halfDxAnchor,
    ProbeGridResult& result)
{
    if (face_xq.size() != face_n.size())
        throw std::runtime_error("runFaceGridCountProbe input size mismatch");

    std::vector<GridKey> samples;
    samples.reserve(face_xq.size() * 2);
    const double anchor = halfDxAnchor ? -0.5 * dx : 0.0;
    for (std::size_t i = 0; i < face_xq.size(); ++i) {
        for (int delta = -1; delta <= 1; delta += 2) {
            Vec3d sample{
                face_xq[i].x + face_n[i].x * 0.5 * dx * static_cast<double>(delta),
                face_xq[i].y + face_n[i].y * 0.5 * dx * static_cast<double>(delta),
                face_xq[i].z + face_n[i].z * 0.5 * dx * static_cast<double>(delta)};
            samples.push_back(order0Key(sample, dx, anchor));
        }
    }
    scatterUnitCounts(samples, result);
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
    if (resident_positions != nullptr || resident_position_count != 0)
        throw std::runtime_error(
            "resident spatial-search particles require the CUDA implementation");
    if (cell_size <= 0.0)
        throw std::runtime_error("persistent spatial search requires positive cell size");
    auto& storage = getSpatialStorage(context);
    const auto t0 = std::chrono::high_resolution_clock::now();
    if (positions.size() > storage.positions.capacity())
        ++storage.reallocations;
    storage.positions = positions;
    storage.cell_size = cell_size;
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->particles = positions.size();
        stats->particle_capacity = storage.positions.capacity();
        stats->reallocations = storage.reallocations;
        stats->resident_particle_positions = false;
        stats->upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
}

bool anyPointWithinRadius(PersistentSpatialSearch3D& context,
    const std::vector<Vec3d>& query_positions, double radius,
    SpatialSearchStats* stats, const void* resident_query_positions,
    std::size_t resident_query_count)
{
    if (resident_query_positions != nullptr || resident_query_count != 0)
        throw std::runtime_error(
            "resident spatial-search queries require the CUDA implementation");
    auto& storage = getSpatialStorage(context);
    if (radius <= 0.0)
        throw std::runtime_error("invalid persistent spatial search radius");
    const auto t0 = std::chrono::high_resolution_clock::now();
    const double radius_squared = radius * radius;
    bool any = false;
    for (const Vec3d& query : query_positions) {
        for (const Vec3d& other : storage.positions) {
            const double dx = query.x - other.x;
            const double dy = query.y - other.y;
            const double dz = query.z - other.z;
            if (dx * dx + dy * dy + dz * dz < radius_squared) {
                any = true;
                break;
            }
        }
        if (any)
            break;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->resident_query_positions = false;
        stats->query_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return any;
}

void computeCubicSphNormals(PersistentSpatialSearch3D& context,
    const std::vector<double>& masses, double radius,
    std::vector<Vec3d>& normals, SpatialSearchStats* stats,
    const void* resident_masses, std::size_t resident_mass_count,
    bool materialize_host_normals)
{
    if (resident_masses != nullptr || resident_mass_count != 0)
        throw std::runtime_error(
            "resident SPH normal masses require the CUDA implementation");
    auto& storage = getSpatialStorage(context);
    if (masses.size() != storage.positions.size())
        throw std::runtime_error("SPH normal mass/position size mismatch");
    const auto t0 = std::chrono::high_resolution_clock::now();
    normals.assign(materialize_host_normals ? masses.size() : 0,
        Vec3d{0.0, 0.0, 0.0});
    for (std::size_t i = 0; i < storage.positions.size(); ++i) {
        const Vec3d query = storage.positions[i];
        double nx = 0.0, ny = 0.0, nz = 0.0;
        for (std::size_t j = 0; j < storage.positions.size(); ++j) {
            const double dx = query.x - storage.positions[j].x;
            const double dy = query.y - storage.positions[j].y;
            const double dz = query.z - storage.positions[j].z;
            const double distance_squared = dx * dx + dy * dy + dz * dz;
            if (distance_squared >= radius * radius || distance_squared <= 1e-16)
                continue;
            const double distance = std::sqrt(distance_squared);
            const double q = distance / radius * 2.0;
            double gradient = 0.0;
            if (q <= 1.0)
                gradient = (-12.0 * q + 9.0 * q * q) / 6.0;
            else if (q <= 2.0)
                gradient = -0.5 * (q - 2.0) * (q - 2.0);
            const double scale = masses[i] * gradient / (distance * radius);
            nx += scale * dx;
            ny += scale * dy;
            nz += scale * dz;
        }
        const double length = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (materialize_host_normals && length > 0.0)
            normals[i] = Vec3d{nx / length, ny / length, nz / length};
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    if (stats) {
        stats->resident_particle_mass = false;
        stats->query_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
}

void copyPersistentSpatialSearchFieldsToHost3D(
    PersistentSpatialSearch3D&, std::vector<Vec3d>&,
    std::vector<Vec3d>&, bool)
{
    throw std::runtime_error(
        "resident spatial-search output copy requires the CUDA implementation");
}

void rebuildInterfaceQuadAtomic3D(PersistentInterfaceQuad3D&,
    PersistentSpatialSearch3D&, const std::vector<Vec3d>&,
    const std::vector<double>&, const std::vector<Vec3d>&,
    double, const BoxShellCollision3D&, InterfaceQuadAtomicResult&,
    InterfaceQuadAtomicStats*, const void*, std::size_t,
    const void*, std::size_t, bool, bool)
{
    throw std::runtime_error(
        "GPU interface quadrature requires the CUDA implementation");
}

void buildPersistentInterfaceQuadratureFields3D(
    PersistentInterfaceQuad3D&, PersistentSpatialSearch3D&,
    const GPU_STAGE3A::SolidParticleStateDeviceView&,
    InterfaceQuadratureDeviceView3D&)
{
    throw std::runtime_error(
        "resident interface fields require the CUDA implementation");
}

void copyInterfaceQuadratureFieldsToHost3D(
    const InterfaceQuadratureDeviceView3D&, std::vector<Vec3d>&,
    std::vector<Vec3d>&, std::vector<Vec3d>&, std::vector<double>&)
{
    throw std::runtime_error(
        "resident interface field copy requires the CUDA implementation");
}

void bindInterfaceQuadratureClampedCurrent3D(
    PersistentFluidAnalytic3D&, InterfaceQuadratureDeviceView3D&)
{
    throw std::runtime_error(
        "resident clamped interface binding requires the CUDA implementation");
}

// The caller gates this device-resident bridge on `gpu_build_system`, so a
// CPU-only executable never reaches it.  Keep an explicit stub nevertheless:
// the branch remains runtime-dispatched in SplittingSimulationWeakFormFast,
// and therefore its symbol must exist for the CPU reference build to link.
void rebuildFluidAnalyticQuadratureFromInterfaceQuad3D(
    PersistentFluidAnalytic3D&, PersistentInterfaceQuad3D&,
    PersistentSpatialSearch3D&, double, double, const BoxShellCollision3D&,
    FluidAnalyticResult&, FluidAnalyticStats*, const void*, std::size_t,
    double, const std::vector<Mat3d>&, bool)
{
    throw std::runtime_error(
        "device-resident fluid analytic quadrature requires the CUDA implementation");
}

void rebuildFluidAnalyticQuadrature(PersistentFluidAnalytic3D& context,
    const std::vector<Vec3d>& volume_xq,
    const std::vector<double>& volume_mass,
    const std::vector<Vec3d>& interface_xq, double dx, double rho,
    const BoxShellCollision3D& collision, FluidAnalyticResult& result,
    FluidAnalyticStats* stats, const void* resident_fluid_velocity,
    const void* resident_fluid_grad_velocity,
    std::size_t resident_fluid_count, double, bool materialize_host_result)
{
    if (resident_fluid_velocity || resident_fluid_grad_velocity ||
        resident_fluid_count != 0)
        throw std::runtime_error(
            "resident fluid velocity/gradient requires the CUDA implementation");
    (void)context;
    if (volume_xq.size() != volume_mass.size() || dx <= 0 || rho <= 0 || !collision.valid)
        throw std::runtime_error("invalid fluid analytic quadrature input");
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto inside_box = [](const Vec3d& x, const Vec3d& lo, const Vec3d& hi) {
        return x.x >= lo.x && x.x <= hi.x && x.y >= lo.y && x.y <= hi.y &&
            x.z >= lo.z && x.z <= hi.z;
    };
    auto in_solid = [&](const Vec3d& x) {
        return inside_box(x, collision.outer_min, collision.outer_max) &&
            !inside_box(x, collision.inner_min, collision.inner_max) &&
            (!collision.has_hole ||
             !inside_box(x, collision.hole_min, collision.hole_max));
    };
    auto cell = [&](const Vec3d& x) {
        return std::array<int, 3>{floorToInt((x.x - 0.5 * dx) / dx),
            floorToInt((x.y - 0.5 * dx) / dx),
            floorToInt((x.z - 0.5 * dx) / dx)};
    };
    auto center = [&](const std::array<int, 3>& c) {
        return Vec3d{(c[0] + 1.0) * dx, (c[1] + 1.0) * dx,
            (c[2] + 1.0) * dx};
    };
    std::vector<Vec3d> px;
    std::vector<double> pm;
    std::vector<std::array<int, 3>> pc;
    for (std::size_t i = 0; i < volume_xq.size(); ++i) {
        const auto c = cell(volume_xq[i]);
        if (!in_solid(center(c))) {
            px.push_back(volume_xq[i]);
            pm.push_back(volume_mass[i]);
            pc.push_back(c);
        }
    }
    std::set<std::array<int, 3>> active(pc.begin(), pc.end());
    std::map<std::array<int, 3>, int> boundary;
    for (const auto& c : active) {
        boundary[c] = 0;
        for (int i = -1; i <= 1; ++i)
            for (int j = -1; j <= 1; ++j)
                for (int k = -1; k <= 1; ++k) {
                    std::array<int, 3> n{c[0] + i, c[1] + j, c[2] + k};
                    if (!active.count(n) && !in_solid(center(n)))
                        boundary[c] = 1;
                }
    }
    result.interface_xq = interface_xq;
    for (Vec3d& x : result.interface_xq) {
        const auto c = cell(x);
        if (active.count(c))
            boundary[c] = 1;
        x.x = std::min(std::max(x.x, (c[0] + 0.6) * dx), (c[0] + 1.4) * dx);
        x.y = std::min(std::max(x.y, (c[1] + 0.6) * dx), (c[1] + 1.4) * dx);
        x.z = std::min(std::max(x.z, (c[2] + 0.6) * dx), (c[2] + 1.4) * dx);
    }
    result.face_xq.clear();
    result.face_normal.clear();
    for (const auto& c : active) {
        for (int d = 0; d < 3; ++d)
            for (int delta : {-1, 1}) {
                auto n = c;
                n[d] += delta;
                if (!in_solid(center(n)))
                    continue;
                Vec3d x = center(n);
                (&x.x)[d] -= delta * 0.5 * dx;
                Vec3d normal{0, 0, 0};
                (&normal.x)[d] = delta;
                result.face_xq.push_back(x);
                result.face_normal.push_back(normal);
            }
    }
    result.volume_xq.clear();
    result.volume_mass.clear();
    std::set<std::array<int, 3>> emitted;
    for (std::size_t i = 0; i < px.size(); ++i) {
        const auto c = pc[i];
        if (boundary[c]) {
            Vec3d x = px[i];
            x.x = std::min(std::max(x.x, (c[0] + 0.6) * dx), (c[0] + 1.4) * dx);
            x.y = std::min(std::max(x.y, (c[1] + 0.6) * dx), (c[1] + 1.4) * dx);
            x.z = std::min(std::max(x.z, (c[2] + 0.6) * dx), (c[2] + 1.4) * dx);
            result.volume_xq.push_back(x);
            result.volume_mass.push_back(pm[i]);
        } else if (emitted.insert(c).second) {
            result.volume_xq.push_back(center(c));
            result.volume_mass.push_back(dx * dx * dx * rho);
        }
    }
    result.pressure_keys.clear();
    for (const auto& c : active)
        result.pressure_keys.push_back(packGridKey3D(c[0], c[1], c[2]));
    result.face_keys.clear();
    for (std::size_t i = 0; i < result.face_xq.size(); ++i) {
        for (int delta : {-1, 1}) {
            Vec3d x{result.face_xq[i].x +
                    result.face_normal[i].x * 0.5 * dx * delta,
                result.face_xq[i].y +
                    result.face_normal[i].y * 0.5 * dx * delta,
                result.face_xq[i].z +
                    result.face_normal[i].z * 0.5 * dx * delta};
            const auto c = cell(x);
            result.face_keys.push_back(packGridKey3D(c[0], c[1], c[2]));
        }
    }
    sortUniqueKeys(result.face_keys);
    if (stats) {
        stats->input_samples = volume_xq.size();
        stats->pruned_samples = px.size();
        stats->active_cells = active.size();
        stats->output_samples = result.volume_xq.size();
        stats->face_samples = result.face_xq.size();
        stats->total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
    if (!materialize_host_result)
        result = {};
}

FluidAnalyticDeviceView3D getFluidAnalyticDeviceView3D(
    const PersistentFluidAnalytic3D&)
{
    return {};
}

void copyFluidAnalyticFaceFieldsToHost3D(const PersistentFluidAnalytic3D&,
    std::vector<Vec3d>&, std::vector<Vec3d>&)
{
    throw std::runtime_error(
        "fluid analytic face download requires the CUDA implementation");
}

void buildFluidLegacyDofNodeMaps3D(PersistentFluidAnalytic3D&, double, int,
    int, int, GPU_STAGE3A::DeviceGridNodeMap&,
    GPU_STAGE3A::DeviceGridNodeMap&, FluidLegacyDofOrderStats*)
{
    throw std::runtime_error(
        "fluid legacy DOF maps require the CUDA implementation");
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
    (void)context;
    if (resident_analytic_position_mass || resident_analytic_velocity ||
        resident_analytic_affine)
        throw std::runtime_error(
            "resident fluid velocity-grid input requires CUDA");
    if (input_xq.size() != input_mass.size() ||
        input_xq.size() != input_velocity.size() ||
        input_xq.size() != input_affine.size() ||
        final_xq.size() != final_mass.size())
        throw std::runtime_error("fluid velocity grid input size mismatch");
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto base_and_weights = [&](const Vec3d& x, int base[3], double w[3][2]) {
        const double values[3] = {x.x - 0.5 * dx, x.y - 0.5 * dx,
            x.z - 0.5 * dx};
        for (int d = 0; d < 3; ++d) {
            const double gx = values[d] / dx;
            base[d] = floorToInt(gx);
            w[d][1] = gx - base[d];
            w[d][0] = 1.0 - w[d][1];
        }
    };
    struct InitialNode {
        double mass = 0.0;
        Vec3d momentum{0.0, 0.0, 0.0};
    };
    std::map<GridKey, InitialNode> initial;
    for (std::size_t q = 0; q < input_xq.size(); ++q) {
        int base[3];
        double w[3][2];
        base_and_weights(input_xq[q], base, w);
        const double shifted[3] = {input_xq[q].x - 0.5 * dx,
            input_xq[q].y - 0.5 * dx, input_xq[q].z - 0.5 * dx};
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    const int offsets[3] = {i, j, k};
                    const GridKey key = packGridKey3D(
                        base[0] + i, base[1] + j, base[2] + k);
                    const double weight = w[0][i] * w[1][j] * w[2][k];
                    const double displacement[3] = {
                        (base[0] + offsets[0]) * dx - shifted[0],
                        (base[1] + offsets[1]) * dx - shifted[1],
                        (base[2] + offsets[2]) * dx - shifted[2]};
                    double v[3] = {input_velocity[q].x, input_velocity[q].y,
                        input_velocity[q].z};
                    for (int row = 0; row < 3; ++row)
                        for (int col = 0; col < 3; ++col)
                            v[row] += input_affine[q].values[row * 3 + col] *
                                displacement[col];
                    const double wm = input_mass[q] * weight;
                    if (wm == 0.0)
                        continue;
                    auto& node = initial[key];
                    node.mass += wm;
                    node.momentum.x += wm * v[0];
                    node.momentum.y += wm * v[1];
                    node.momentum.z += wm * v[2];
                }
    }
    std::map<GridKey, Vec3d> initial_velocity;
    for (const auto& [key, node] : initial)
        initial_velocity.emplace(key, Vec3d{
            node.momentum.x / node.mass + dt * gravity.x,
            node.momentum.y / node.mass + dt * gravity.y,
            node.momentum.z / node.mass + dt * gravity.z});

    std::map<GridKey, double> rebuilt_mass;
    for (std::size_t q = 0; q < final_xq.size(); ++q) {
        int base[3];
        double w[3][2];
        base_and_weights(final_xq[q], base, w);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    const GridKey key = packGridKey3D(
                        base[0] + i, base[1] + j, base[2] + k);
                    const double contribution =
                        final_mass[q] * w[0][i] * w[1][j] * w[2][k];
                    if (contribution != 0.0)
                        rebuilt_mass[key] += contribution;
                }
    }
    nodes.clear();
    nodes.reserve(rebuilt_mass.size());
    for (const auto& [key, mass] : rebuilt_mass) {
        const auto initial_it = initial_velocity.find(key);
        nodes.push_back(VelocityGridNode{key, mass,
            initial_it == initial_velocity.end()
                ? Vec3d{0.0, 0.0, 0.0}
                : initial_it->second});
    }
    g2p_fallback_nodes.clear();
    for (const auto& [key, velocity] : initial_velocity)
        if (rebuilt_mass.find(key) == rebuilt_mass.end())
            g2p_fallback_nodes.push_back(
                VelocityGridNode{key, 0.0, velocity});
    if (stats) {
        stats->input_samples = input_xq.size();
        stats->final_samples = final_xq.size();
        stats->nodes = nodes.size();
        stats->g2p_fallback_nodes = g2p_fallback_nodes.size();
        stats->resident_position_mass = false;
        stats->resident_velocity = false;
        stats->resident_affine = false;
        stats->total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
    if (!materialize_host_nodes) {
        nodes.clear();
        g2p_fallback_nodes.clear();
    }
}

void buildFluidLegacyVelocityNodeMap3D(PersistentFluidAnalytic3D&,
    GPU_STAGE3A::DeviceGridNodeMap&, double*)
{
    throw std::runtime_error(
        "fluid legacy velocity-node map requires the CUDA implementation");
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
    (void)device_nodes;
    (void)materialize_host_nodes;
    (void)context;
    if (resident_particles)
        throw std::runtime_error(
            "resident solid pressure input requires the CUDA implementation");
    if (positions.size() != j.size() || positions.size() != volume.size() ||
        positions.size() != lambda.size())
        throw std::runtime_error("solid pressure grid input size mismatch");
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::map<GridKey, SolidPressureGridNode> grid;
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const GridKey key = packGridKey3D(
            floorToInt((positions[i].x - 0.5 * dx) / dx),
            floorToInt((positions[i].y - 0.5 * dx) / dx),
            floorToInt((positions[i].z - 0.5 * dx) / dx));
        auto& node = grid[key];
        node.key = key;
        const double p = -lambda[i] * (j[i] - 1.0);
        node.mass += volume[i] * j[i];
        node.compliance += volume[i] / lambda[i];
        node.pressure += volume[i] * j[i] * p;
    }
    nodes.clear();
    nodes.reserve(grid.size());
    for (auto& [key, node] : grid) {
        node.pressure /= node.mass;
        nodes.push_back(node);
    }
    if (stats) {
        stats->input_samples = positions.size();
        stats->final_samples = positions.size();
        stats->nodes = nodes.size();
        stats->total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
}

} // namespace ZIRAN::GPU_STAGE2
