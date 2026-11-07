#pragma once

#include "../splitting/SplittingSimulation.h"
#include "GpuSparseGridProbe.h"
#include "ParticlesLevelSet.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>

namespace ZIRAN {

template <class T, int dim>
class PositionCorrector {
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using TStack = Vector<T, Eigen::Dynamic>;

    SplittingSimulation<T, dim>& sim;
    MpmGrid<T, dim, 1> pressure_grid;
    MpmGrid<T, dim, 1> solid_grid;
    TStack rhs, solved_p;
    bool solid_classification_cache_valid = false;
    GPU_STAGE2::PersistentPressureGrid3D gpu_pressure_grid;
    std::vector<GPU_STAGE2::Vec3d> gpu_density_sources;
    std::vector<GPU_STAGE2::Vec3d> gpu_density_static_sources;
    std::vector<GPU_STAGE2::Vec3d> gpu_density_dynamic_sources;
    const TV* gpu_density_boundary_data = nullptr;
    std::size_t gpu_density_boundary_count = 0;
    GPU_STAGE2::ProbeGridResult gpu_density_result;
    bool gpu_density_result_valid = false;
    GPU_STAGE2::PersistentPagePollute3D gpu_pressure_page_pollute;
    std::vector<GPU_STAGE2::Vec3d> gpu_pressure_page_sources;
    std::vector<IV> gpu_pressure_page_nodes;
    GPU_STAGE2::PersistentPagePollute3D gpu_pressure_topology;
    std::vector<GPU_STAGE2::GridKey> direct_pressure_dof_keys;
    std::vector<GPU_STAGE2::GridKey> direct_pressure_legacy_dof_keys;
    std::unordered_map<GPU_STAGE2::GridKey, int> direct_pressure_dof_lookup;
    std::vector<GPU_STAGE2::GridKey> direct_pressure_touched_keys;
    std::vector<unsigned char> direct_pressure_solid_flags;
    std::unordered_map<GPU_STAGE2::GridKey, int> direct_pressure_touched_lookup;
    // Compact host mirrors of the keyed pressure field. They replace
    // pressure_grid::new_v and solved_p in the direct production path; the
    // corresponding topology/density/CSR fields are resident in
    // gpu_pressure_grid.
    std::vector<IV> direct_pressure_page_bases;
    std::vector<IV> direct_pressure_solid_block_bases;
    std::unordered_set<GPU_STAGE2::GridKey> direct_pressure_seen_pages;
    std::vector<TV> direct_pressure_new_v;
    std::vector<GPU_STAGE2::GridKey> gpu_pushout_displacement_keys;
    std::vector<GPU_STAGE2::Vec3d> gpu_pushout_displacement;
    std::vector<GPU_STAGE2::GridKey> gpu_pushout_face_keys;
    std::vector<GPU_STAGE2::Vec3d> gpu_pushout_face_velocity;
    bool prepare_gpu_pressure_postsolve = false;
    bool gpu_pressure_pushout_ready = false;
    TStack direct_solved_p;
    bool direct_pressure_gridless_active = false;
    bool direct_pressure_resident_solid_flags = false;
    bool direct_pressure_device_topology_only = false;
    bool direct_pressure_device_fields_ready = false;
    std::size_t direct_pressure_dof_count = 0;
    std::size_t direct_pressure_touched_count = 0;
    struct DirectPressureSystem {
        std::vector<GPU_STAGE2::GridKey> dof_keys;
        Eigen::SparseMatrix<T> lhs;
        TStack rhs;
        // Preserved GPU-assembled row-major CSR. This avoids an Eigen CSR
        // re-export when the direct pressure solve itself uses AMGCL_CUDA.
        std::vector<int> row_offsets;
        std::vector<int> column_indices;
        std::vector<T> values;
        std::size_t nnz = 0;
        std::size_t rows = 0;
        bool host_csr_valid = false;
        bool host_rhs_valid = false;
        bool gpu_csr_valid = false;
        bool valid = false;
    } direct_pressure_system;

    GPU_STAGE2::GridKey packProbeKey(const IV& node) const
    {
        if constexpr (dim == 3)
            return GPU_STAGE2::packGridKey3D(node[0], node[1], node[2]);
        else
            return GPU_STAGE2::packGridKey3D(node[0], node[1], 0);
    }

    GPU_STAGE2::Vec3d toProbeVec(const TV& x) const
    {
        if constexpr (dim == 3) {
            return GPU_STAGE2::Vec3d{
                static_cast<double>(x[0]),
                static_cast<double>(x[1]),
                static_cast<double>(x[2])};
        } else {
            return GPU_STAGE2::Vec3d{
                static_cast<double>(x[0]),
                static_cast<double>(x[1]),
                0.0};
        }
    }

    bool hasResidentFluidPositions() const
    {
        return sim.gpu_velocity_transfer_device_x_valid
            && sim.gpu_velocity_transfer_device.x != nullptr
            && sim.gpu_velocity_transfer_device.size
                == static_cast<std::size_t>(sim.particles.count);
    }

    void rebuildGpuDensitySources(SplittingSimulation<T, dim>& Ssim,
        bool build_combined_shadow_sources = false,
        bool omit_resident_fluid = false,
        bool omit_resident_solid = false)
    {
        if constexpr (dim == 3) {
            ZIRAN_NAMED_TIMER("gpuDensitySourcePack");
            const TV* boundary_data = sim.boundary_positions.empty()
                ? nullptr : sim.boundary_positions.data();
            const std::size_t boundary_count = sim.boundary_positions.size();
            // Example 2's sampled container boundary is static. Keep its packed
            // Vec3d prefix across coupling substeps; only the fluid/solid
            // particle suffix changes. Pointer/count changes remain a safe
            // invalidation path for scenes that rebuild their boundary set.
            if (boundary_data != gpu_density_boundary_data
                || boundary_count != gpu_density_boundary_count) {
                gpu_density_static_sources.clear();
                gpu_density_static_sources.reserve(boundary_count);
                for (const TV& Xp : sim.boundary_positions)
                    gpu_density_static_sources.push_back(toProbeVec(Xp));
                gpu_density_boundary_data = boundary_data;
                gpu_density_boundary_count = boundary_count;
            }
            gpu_density_dynamic_sources.clear();
            gpu_density_dynamic_sources.reserve(
                (omit_resident_fluid ? 0
                                     : static_cast<std::size_t>(sim.particles.count))
                + (omit_resident_solid ? 0
                                       : static_cast<std::size_t>(Ssim.particles.count)));
            if (!omit_resident_fluid)
                for (int i = 0; i < sim.particles.count; ++i)
                    gpu_density_dynamic_sources.push_back(
                        toProbeVec(sim.particles.X.array[i]));
            if (!omit_resident_solid)
                for (int i = 0; i < Ssim.particles.count; ++i) {
                    const TV& Xp = (sim.solid_scheme != CONFIG_UPDATE_LAGRANGIAN)
                        ? sim.solid_cur_positions[i]
                        : Ssim.particles.X.array[i];
                    gpu_density_dynamic_sources.push_back(toProbeVec(Xp));
                }
            if (build_combined_shadow_sources) {
                gpu_density_sources.clear();
                gpu_density_sources.reserve(gpu_density_static_sources.size()
                    + gpu_density_dynamic_sources.size());
                gpu_density_sources.insert(gpu_density_sources.end(),
                    gpu_density_static_sources.begin(), gpu_density_static_sources.end());
                gpu_density_sources.insert(gpu_density_sources.end(),
                    gpu_density_dynamic_sources.begin(), gpu_density_dynamic_sources.end());
            }
        }
    }

    void rebuildGpuPressurePageSources(bool omit_resident_fluid = false)
    {
        if constexpr (dim == 3) {
            ZIRAN_NAMED_TIMER("gpuPressurePageSourcePack");
            gpu_pressure_page_sources.clear();
            if (omit_resident_fluid)
                return;
            gpu_pressure_page_sources.reserve(
                static_cast<std::size_t>(sim.particles.count));
            for (int i = 0; i < sim.particles.count; ++i)
                gpu_pressure_page_sources.push_back(
                    toProbeVec(sim.particles.X.array[i]));
        }
    }

    bool pollutePressureGridGpu()
    {
        if constexpr (dim != 3) {
            return false;
        } else {
#ifndef ENABLE_CUDA_STAGE2_PROBE
            return false;
#else
            rebuildGpuPressurePageSources();

            std::vector<GPU_STAGE2::GridKey> base_keys;
            GPU_STAGE2::PressureGridFieldStats stats;
            {
                ZIRAN_NAMED_TIMER("pressureGridPolluteGpu");
                GPU_STAGE2::rebuildPersistentPressureGridPages3D(
                    gpu_pressure_grid, gpu_pressure_page_sources,
                    static_cast<double>(sim.dx), base_keys, &stats);
            }

            gpu_pressure_page_nodes.clear();
            gpu_pressure_page_nodes.reserve(base_keys.size());
            for (const GPU_STAGE2::GridKey key : base_keys) {
                int x, y, z;
                GPU_STAGE2::unpackGridKey3D(key, x, y, z);
                const IV base_node { x, y, z };
                // Match MpmGrid::activatePages(): repeated *adjacent* base nodes
                // do not call Set_Page. Keep the first-occurrence order of
                // every run unchanged, since it fixes SPGrid block ordering.
                const bool same_as_previous = !gpu_pressure_page_nodes.empty()
                    && base_node[0] == gpu_pressure_page_nodes.back()[0]
                    && base_node[1] == gpu_pressure_page_nodes.back()[1]
                    && base_node[2] == gpu_pressure_page_nodes.back()[2];
                if (!same_as_previous)
                    gpu_pressure_page_nodes.emplace_back(base_node);
            }
            // This only replaces base-node discovery. MpmGrid retains its
            // original Clear/Set_Page/Update_Block_Offsets/zeroing sequence,
            // with base nodes in the original particle order.
            pressure_grid.activatePagesFromBaseNodes(gpu_pressure_page_nodes, -1, 1);
            ZIRAN_INFO("PositionCorrector GPU pressure pages: sources=",
                stats.page_sources, " base_nodes=", base_keys.size(),
                " page_activation_samples=", gpu_pressure_page_nodes.size(),
                " reallocations=", stats.reallocations,
                " upload_ms=", stats.upload_ms,
                " key_generation_ms=", stats.key_generation_ms,
                " download_ms=", stats.download_ms,
                " total_ms=", stats.total_ms);
            return true;
#endif
        }
    }

    // Build the direct active pressure keys.  The optional SPGrid comparison
    // is diagnostic-only: the production direct path must not traverse the
    // old grid merely to prove a result that has already been validated.
    void buildDirectPressureTopology(bool validate_against_spgrid)
    {
        if constexpr (dim != 3) {
            return;
        } else {
#ifndef ENABLE_CUDA_STAGE2_PROBE
            return;
#else
            rebuildGpuPressurePageSources();
            std::vector<GPU_STAGE2::GridKey> source_base_keys;
            GPU_STAGE2::PagePolluteStats stats;
            {
                ZIRAN_NAMED_TIMER("directPressureTopologyGpu");
                // The CPU pressure DOFs use the degree-1 base node of
                // Xp - 0.5 dx, rather than the unshifted density stencil.
                GPU_STAGE2::runPersistentLinearBaseKey3D(
                    gpu_pressure_topology, gpu_pressure_page_sources,
                    static_cast<double>(sim.dx), /*halfDxAnchor*/ true,
                    source_base_keys, &stats);
            }
            ZIRAN_NAMED_TIMER("directPressureTopologyHost");

            direct_pressure_dof_keys.clear();
            direct_pressure_dof_keys.reserve(source_base_keys.size());
            for (const GPU_STAGE2::GridKey key : source_base_keys) {
                int x, y, z;
                GPU_STAGE2::unpackGridKey3D(key, x, y, z);
                const IV node { x, y, z };
                // The pressure-grid solid cache is still the temporary
                // classification oracle for active-key eligibility. Calling
                // computeIsSolid() once per fluid source redoes expensive
                // level-set queries and regressed example 2 by about 2.2 s/frame.
                // Direct CSR itself uses direct_pressure_solid_flags below.
                if (!isSolid(node))
                    direct_pressure_dof_keys.push_back(key);
            }
            GPU_STAGE2::sortUniqueKeys(direct_pressure_dof_keys);

            ZIRAN_INFO("PositionCorrector direct pressure topology: sources=",
                stats.sources, " gpu_base_nodes=", stats.base_nodes,
                " active_dofs=", direct_pressure_dof_keys.size(),
                " upload_ms=", stats.upload_ms,
                " key_generation_ms=", stats.key_generation_ms,
                " download_ms=", stats.download_ms,
                " total_ms=", stats.total_ms,
                " spgrid_oracle=", validate_against_spgrid);
            if (validate_against_spgrid) {
                std::vector<GPU_STAGE2::GridKey> spgrid_dof_keys;
                spgrid_dof_keys.reserve(static_cast<std::size_t>(
                    pressure_grid.getNumNodes()));
                pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>&) {
                    spgrid_dof_keys.push_back(packProbeKey(node));
                });
                const GPU_STAGE2::ProbeCompareStats compare =
                    GPU_STAGE2::compareKeySets(spgrid_dof_keys,
                        direct_pressure_dof_keys);
                logProbeStats("direct_pressure_dof", compare);
            }
#endif
        }
    }

    int directPressureDof(const IV& node) const
    {
        const GPU_STAGE2::GridKey key = packProbeKey(node);
        if (!direct_pressure_dof_lookup.empty()) {
            const auto it = direct_pressure_dof_lookup.find(key);
            return it == direct_pressure_dof_lookup.end() ? -1 : it->second;
        }
        const auto it = std::lower_bound(direct_pressure_dof_keys.begin(),
            direct_pressure_dof_keys.end(), key);
        if (it == direct_pressure_dof_keys.end() || *it != key)
            return -1;
        return static_cast<int>(it - direct_pressure_dof_keys.begin());
    }

    T directPressureDensity(const IV& node) const
    {
        if (!gpu_density_result_valid)
            return pressure_grid[node].v[0];
        const GPU_STAGE2::GridKey key = packProbeKey(node);
        const auto it = std::lower_bound(gpu_density_result.keys.begin(),
            gpu_density_result.keys.end(), key);
        if (it == gpu_density_result.keys.end() || *it != key)
            return (T)0;
        const std::size_t index = static_cast<std::size_t>(it
            - gpu_density_result.keys.begin());
        return static_cast<T>(gpu_density_result.mass[index]);
    }

    bool directPressureIsSolid(const IV& node)
    {
        const auto it = direct_pressure_touched_lookup.find(packProbeKey(node));
        if (it != direct_pressure_touched_lookup.end())
            return direct_pressure_solid_flags[
                static_cast<std::size_t>(it->second)] != 0;
        return computeIsSolid(node);
    }

    int directPressureTouched(const IV& node) const
    {
        const auto it = direct_pressure_touched_lookup.find(packProbeKey(node));
        return it == direct_pressure_touched_lookup.end() ? -1 : it->second;
    }

    TV& directPressureVelocity(const IV& node)
    {
        const int index = directPressureTouched(node);
        ZIRAN_ASSERT(index >= 0,
            "direct pressure velocity query lies outside keyed virtual blocks");
        return direct_pressure_new_v[static_cast<std::size_t>(index)];
    }

    const TV& directPressureVelocity(const IV& node) const
    {
        const int index = directPressureTouched(node);
        ZIRAN_ASSERT(index >= 0,
            "direct pressure velocity query lies outside keyed virtual blocks");
        return direct_pressure_new_v[static_cast<std::size_t>(index)];
    }

    T directPressureValue(const IV& node) const
    {
        const auto it = direct_pressure_dof_lookup.find(packProbeKey(node));
        return it == direct_pressure_dof_lookup.end() || it->second >= direct_solved_p.rows()
            ? (T)0
            : direct_solved_p(it->second);
    }

    static int floorDivide(int numerator, int denominator)
    {
        ZIRAN_ASSERT(denominator > 0, "floorDivide requires a positive denominator");
        int quotient = numerator / denominator;
        const int remainder = numerator % denominator;
        if (remainder && numerator < 0)
            --quotient;
        return quotient;
    }

    // While pressure_grid remains necessary for mass, pushout and final
    // interpolation, it is also the exact current-frame oracle for the old
    // DOF/page order. Reuse that order in production rather than re-running
    // a second 1:1 particle-key scan solely to rebuild virtual blocks.
    void buildDirectPressureLegacyOrderFromPressureGrid()
    {
        if constexpr (dim != 3) {
            return;
        } else {
            ZIRAN_NAMED_TIMER("directPressureLegacyOrderFromPressureGrid");
            direct_pressure_legacy_dof_keys.clear();
            pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>&) {
                direct_pressure_legacy_dof_keys.push_back(packProbeKey(node));
            });

            direct_pressure_touched_keys.clear();
            direct_pressure_touched_keys.reserve(
                direct_pressure_legacy_dof_keys.size() * 6);
            pressure_grid.iterateTouchedGridSerial(
                [&](IV node, GridState<T, dim>&) {
                    direct_pressure_touched_keys.push_back(packProbeKey(node));
                });

            direct_pressure_touched_lookup.clear();
            direct_pressure_touched_lookup.reserve(
                direct_pressure_touched_keys.size() * 2);
            for (std::size_t i = 0; i < direct_pressure_touched_keys.size(); ++i)
                direct_pressure_touched_lookup.emplace(
                    direct_pressure_touched_keys[i], static_cast<int>(i));
            direct_pressure_solid_flags.resize(direct_pressure_touched_keys.size());
            {
                ZIRAN_NAMED_TIMER("directPressureSolidClassify");
                tbb::parallel_for(0, static_cast<int>(direct_pressure_touched_keys.size()),
                    [&](int i) {
                        int x, y, z;
                        GPU_STAGE2::unpackGridKey3D(
                            direct_pressure_touched_keys[static_cast<std::size_t>(i)],
                            x, y, z);
                        direct_pressure_solid_flags[static_cast<std::size_t>(i)] =
                            computeIsSolid(IV { x, y, z }) ? 1 : 0;
                    });
            }
            ZIRAN_INFO("PositionCorrector direct legacy order: source=pressure_grid",
                " direct_dofs=", direct_pressure_legacy_dof_keys.size(),
                " touched_nodes=", direct_pressure_touched_keys.size());
        }
    }

    // Reconstruct the pre-existing SPGrid DOF order with ordinary keyed
    // virtual blocks. This independent reconstruction is retained for probe
    // validation and for the future stage where pressure_grid is removed.
    void buildDirectPressureLegacyOrder(bool validate_against_spgrid,
        bool use_pressure_grid_order = false)
    {
        if (use_pressure_grid_order) {
            buildDirectPressureLegacyOrderFromPressureGrid();
            return;
        }
        if constexpr (dim != 3) {
            return;
        } else {
#ifndef ENABLE_CUDA_STAGE2_PROBE
            return;
#else
            ZIRAN_NAMED_TIMER("directPressureLegacyTopology");
            rebuildGpuPressurePageSources();
            std::vector<GPU_STAGE2::GridKey> source_base_keys;
            GPU_STAGE2::PagePolluteStats stats;
            GPU_STAGE2::runPersistentLinearBaseKey3D(
                gpu_pressure_page_pollute, gpu_pressure_page_sources,
                static_cast<double>(sim.dx), /*halfDxAnchor*/ false,
                source_base_keys, &stats);

            using PressureMask = typename MpmGrid<T, dim, 1>::SparseMask;
            constexpr int block_x = 1 << PressureMask::block_xbits;
            constexpr int block_y = 1 << PressureMask::block_ybits;
            constexpr int block_z = 1 << PressureMask::block_zbits;
            std::vector<IV> page_order;
            page_order.reserve(source_base_keys.size() / 4);
            std::unordered_set<GPU_STAGE2::GridKey> seen_pages;
            seen_pages.reserve(source_base_keys.size() / 2);
            auto activate_page = [&](const IV& page) {
                const GPU_STAGE2::GridKey page_key = packProbeKey(page);
                if (seen_pages.insert(page_key).second)
                    page_order.push_back(page);
            };

            IV previous_base = IV::Zero();
            bool have_previous_base = false;
            for (const GPU_STAGE2::GridKey key : source_base_keys) {
                int x, y, z;
                GPU_STAGE2::unpackGridKey3D(key, x, y, z);
                const IV base { x, y, z };
                if (have_previous_base
                    && base[0] == previous_base[0]
                    && base[1] == previous_base[1]
                    && base[2] == previous_base[2])
                    continue;
                previous_base = base;
                have_previous_base = true;
                const IV base_page {
                    floorDivide(base[0], block_x),
                    floorDivide(base[1], block_y),
                    floorDivide(base[2], block_z) };
                // Match MpmGrid::activatePages(): activate the base page first,
                // then visit the 3x3x3 page neighborhood in loop order.
                activate_page(base_page);
                for (int i = -1; i <= 1; ++i)
                    for (int j = -1; j <= 1; ++j)
                        for (int k = -1; k <= 1; ++k)
                            activate_page(base_page + IV { i, j, k });
            }

            std::unordered_set<GPU_STAGE2::GridKey> active_dofs(
                direct_pressure_dof_keys.begin(), direct_pressure_dof_keys.end());
            std::vector<GPU_STAGE2::GridKey> legacy_order_keys;
            legacy_order_keys.reserve(direct_pressure_dof_keys.size());
            direct_pressure_touched_keys.clear();
            direct_pressure_touched_keys.reserve(page_order.size()
                * static_cast<std::size_t>(block_x * block_y * block_z));
            for (const IV& page : page_order)
                for (int i = 0; i < block_x; ++i)
                    for (int j = 0; j < block_y; ++j)
                        for (int k = 0; k < block_z; ++k) {
                            const IV node {
                                page[0] * block_x + i,
                                page[1] * block_y + j,
                                page[2] * block_z + k };
                            const GPU_STAGE2::GridKey key = packProbeKey(node);
                            direct_pressure_touched_keys.push_back(key);
                            if (active_dofs.find(key) != active_dofs.end())
                                legacy_order_keys.push_back(key);
                        }

            direct_pressure_touched_lookup.clear();
            direct_pressure_touched_lookup.reserve(
                direct_pressure_touched_keys.size() * 2);
            for (std::size_t i = 0; i < direct_pressure_touched_keys.size(); ++i)
                direct_pressure_touched_lookup.emplace(
                    direct_pressure_touched_keys[i], static_cast<int>(i));
            direct_pressure_solid_flags.resize(direct_pressure_touched_keys.size());
            {
                ZIRAN_NAMED_TIMER("directPressureSolidClassify");
                tbb::parallel_for(0, static_cast<int>(direct_pressure_touched_keys.size()),
                    [&](int i) {
                        int x, y, z;
                        GPU_STAGE2::unpackGridKey3D(
                            direct_pressure_touched_keys[static_cast<std::size_t>(i)],
                            x, y, z);
                        direct_pressure_solid_flags[static_cast<std::size_t>(i)] =
                            computeIsSolid(IV { x, y, z }) ? 1 : 0;
                    });
            }

            std::size_t mismatches = 0;
            int first_mismatch = -1;
            std::size_t spgrid_dofs = 0;
            if (validate_against_spgrid) {
                std::vector<GPU_STAGE2::GridKey> spgrid_order_keys(
                    direct_pressure_dof_keys.size());
                pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                    spgrid_order_keys[g.idx] = packProbeKey(node);
                });
                spgrid_dofs = spgrid_order_keys.size();
                const std::size_t common = std::min(legacy_order_keys.size(),
                    spgrid_order_keys.size());
                for (std::size_t i = 0; i < common; ++i)
                    if (legacy_order_keys[i] != spgrid_order_keys[i]) {
                        ++mismatches;
                        if (first_mismatch < 0)
                            first_mismatch = static_cast<int>(i);
                    }
                mismatches += legacy_order_keys.size() > common
                    ? legacy_order_keys.size() - common
                    : spgrid_order_keys.size() - common;
            }
            direct_pressure_legacy_dof_keys = legacy_order_keys;
            ZIRAN_INFO("PositionCorrector direct legacy order: pages=",
                page_order.size(), " sources=", stats.sources,
                " direct_dofs=", legacy_order_keys.size(),
                " spgrid_dofs=", spgrid_dofs,
                " touched_nodes=", direct_pressure_touched_keys.size(),
                " mismatches=", mismatches,
                " first_mismatch=", first_mismatch,
                " spgrid_oracle=", validate_against_spgrid);
#endif
        }
    }

    // Production direct topology with no pressure MpmGrid. The old SPGrid
    // page/block sequence is reconstructed as an ordinary ordered list of
    // virtual blocks, then retained only for the independent solid pushout
    // grid. Pressure DOFs, solid flags and all pressure fields stay keyed.
    void buildGridlessDirectPressureTopology(bool allow_device_only)
    {
        if constexpr (dim != 3) {
            return;
        } else {
#ifndef ENABLE_CUDA_STAGE2_PROBE
            return;
#else
            ZIRAN_NAMED_TIMER("directPressureGridlessTopology");
            const bool resident_fluid_positions =
                hasResidentFluidPositions();
            const auto level_set_device =
                GPU_POSITION_CORRECTOR::getParticleLevelSetDeviceView3D(
                    levelset_builder.gpu_context);
            const bool supported_collision =
                sim.gpu_box_shell_collision.valid || sim.collision_objects.empty();
            const bool resident_pushout_particles =
                sim.gpu_velocity_transfer_device_x_valid
                && sim.gpu_velocity_transfer_device_v_grad_valid
                && sim.gpu_velocity_transfer_device.x != nullptr
                && sim.gpu_velocity_transfer_device.v != nullptr
                && sim.gpu_velocity_transfer_device.grad_v != nullptr
                && sim.gpu_velocity_transfer_device.size
                    == static_cast<std::size_t>(sim.particles.count);
            direct_pressure_device_topology_only = allow_device_only
                && resident_fluid_positions && resident_pushout_particles
                && levelset_builder.gpu_result_active && level_set_device.phi
                && supported_collision
                && std::getenv("ZIRAN_GPU_PRESSURE_TOPOLOGY_HOST_RESULT") == nullptr
                && std::getenv("ZIRAN_AMGX_COMPARE_AMGCL") == nullptr
                && std::getenv("ZIRAN_GPU_PRESSURE_POSTSOLVE_COMPARE") == nullptr
                && std::getenv("ZIRAN_GPU_PRESSURE_CSR_HOST_RESULT") == nullptr;
            direct_pressure_device_fields_ready = false;
            direct_pressure_dof_count = 0;
            direct_pressure_touched_count = 0;
            rebuildGpuPressurePageSources(resident_fluid_positions);
            std::vector<GPU_STAGE2::GridKey> page_base_keys;
            std::vector<GPU_STAGE2::GridKey> dof_source_keys;
            GPU_STAGE2::PressureGridFieldStats page_stats;
            GPU_STAGE2::PressureGridFieldStats dof_stats;
            {
                ZIRAN_NAMED_TIMER("gridlessPressureGpuKeys");
                GPU_STAGE2::rebuildPersistentPressureGridPages3D(
                    gpu_pressure_grid, gpu_pressure_page_sources,
                    static_cast<double>(sim.dx), page_base_keys, &page_stats,
                    /*download_base_keys*/ false,
                    resident_fluid_positions
                        ? sim.gpu_velocity_transfer_device.x : nullptr,
                    resident_fluid_positions
                        ? sim.gpu_velocity_transfer_device.size : 0);
                GPU_STAGE2::extractPersistentPressureGridDofSourceKeys3D(
                    gpu_pressure_grid, static_cast<double>(sim.dx),
                    dof_source_keys, &dof_stats,
                    /*download_keys*/ !direct_pressure_device_topology_only);
            }

            using PressureMask = typename MpmGrid<T, dim, 1>::SparseMask;
            constexpr int block_x = 1 << PressureMask::block_xbits;
            constexpr int block_y = 1 << PressureMask::block_ybits;
            constexpr int block_z = 1 << PressureMask::block_zbits;
            {
                ZIRAN_NAMED_TIMER("gridlessPressureVirtualBlocks");
                // Keep page activation and block expansion on the persistent
                // device sparse grid. The host receives only the compact,
                // legacy-ordered lists still required by CPU solid pushout.
                GPU_STAGE2::VirtualBlockOrderResult virtual_blocks;
                GPU_STAGE2::VirtualBlockOrderStats virtual_stats;
                GPU_STAGE2::buildPersistentPressureGridVirtualBlocks3D(
                    gpu_pressure_grid, block_x, block_y, block_z,
                    virtual_blocks, &virtual_stats,
                    /*download_result*/ !direct_pressure_device_topology_only);
                direct_pressure_page_bases.resize(virtual_blocks.page_keys.size());
                direct_pressure_solid_block_bases.resize(virtual_blocks.page_keys.size());
                for (std::size_t i = 0; i < virtual_blocks.page_keys.size(); ++i) {
                    int x, y, z;
                    GPU_STAGE2::unpackGridKey3D(virtual_blocks.page_keys[i], x, y, z);
                    direct_pressure_page_bases[i] = IV { x, y, z };
                    direct_pressure_solid_block_bases[i] = IV {
                        x * block_x, y * block_y, z * block_z };
                }
                direct_pressure_touched_keys = std::move(virtual_blocks.touched_keys);
                direct_pressure_touched_count = virtual_stats.touched_nodes;
                direct_pressure_touched_lookup.clear();
                direct_pressure_touched_lookup.reserve(
                    direct_pressure_touched_keys.size() * 2);
                for (std::size_t i = 0; i < direct_pressure_touched_keys.size(); ++i)
                    direct_pressure_touched_lookup.emplace(
                        direct_pressure_touched_keys[i], static_cast<int>(i));
                ZIRAN_INFO("PositionCorrector GPU virtual blocks: pages=",
                    virtual_stats.virtual_blocks, " touched=",
                    virtual_stats.touched_nodes, " order_ms=",
                    virtual_stats.ordering_ms, " expand_ms=",
                    virtual_stats.expansion_ms, " download_ms=",
                    virtual_stats.download_ms);
            }

            direct_pressure_resident_solid_flags = false;
            if (levelset_builder.gpu_result_active && level_set_device.phi
                && supported_collision) {
                ZIRAN_NAMED_TIMER("directPressureSolidClassifyGpu");
                GPU_STAGE2::PressureGridFieldStats classification_stats;
                GPU_STAGE2::classifyPersistentPressureGridSolid3D(
                    gpu_pressure_grid, level_set_device,
                    sim.gpu_box_shell_collision, static_cast<double>(sim.dx),
                    direct_pressure_solid_flags, &classification_stats,
                    /*download_flags*/ !direct_pressure_device_topology_only);
                direct_pressure_resident_solid_flags =
                    classification_stats.resident_solid_flags;
                ZIRAN_INFO("PositionCorrector GPU solid classification: touched=",
                    classification_stats.touched_nodes,
                    " resident_flags=",
                    classification_stats.resident_solid_flags ? 1 : 0,
                    " download_ms=", classification_stats.download_ms,
                    " total_ms=", classification_stats.total_ms);
            } else {
                direct_pressure_solid_flags.resize(
                    direct_pressure_touched_keys.size());
                ZIRAN_NAMED_TIMER("directPressureSolidClassify");
                tbb::parallel_for(0,
                    static_cast<int>(direct_pressure_touched_keys.size()),
                    [&](int i) {
                        int x, y, z;
                        GPU_STAGE2::unpackGridKey3D(
                            direct_pressure_touched_keys[static_cast<std::size_t>(i)],
                            x, y, z);
                        direct_pressure_solid_flags[static_cast<std::size_t>(i)] =
                            computeIsSolid(IV { x, y, z }) ? 1 : 0;
                    });
            }

            {
                ZIRAN_NAMED_TIMER("gridlessPressureActiveDofs");
                direct_pressure_legacy_dof_keys.clear();
                direct_pressure_dof_keys.clear();
                direct_pressure_dof_lookup.clear();
                direct_pressure_new_v.clear();
                if (direct_pressure_device_topology_only) {
                    GPU_STAGE2::PressureGridFieldStats field_stats;
                    GPU_STAGE2::updatePersistentPressureGridDirectFields3D(
                        gpu_pressure_grid, direct_pressure_legacy_dof_keys,
                        direct_pressure_touched_keys,
                        direct_pressure_solid_flags, &field_stats,
                        /*reuse_resident_virtual_touched_keys*/ true,
                        /*rebuild_resident_legacy_dof_keys*/ true,
                        /*reuse_resident_solid_flags*/ true);
                    direct_pressure_dof_count = field_stats.dofs;
                    direct_pressure_device_fields_ready = true;
                } else {
                    direct_pressure_dof_keys.reserve(dof_source_keys.size());
                    for (const GPU_STAGE2::GridKey key : dof_source_keys) {
                        const auto it = direct_pressure_touched_lookup.find(key);
                        ZIRAN_ASSERT(it != direct_pressure_touched_lookup.end(),
                            "direct pressure source key lies outside its virtual block set");
                        if (!direct_pressure_solid_flags[
                                static_cast<std::size_t>(it->second)])
                            direct_pressure_dof_keys.push_back(key);
                    }
                    ZIRAN_ASSERT(std::is_sorted(direct_pressure_dof_keys.begin(),
                            direct_pressure_dof_keys.end()),
                        "persistent pressure-grid DOF source keys must be sorted");
                    std::unordered_set<GPU_STAGE2::GridKey> active_dofs(
                        direct_pressure_dof_keys.begin(), direct_pressure_dof_keys.end());
                    direct_pressure_legacy_dof_keys.reserve(
                        direct_pressure_dof_keys.size());
                    for (const GPU_STAGE2::GridKey key : direct_pressure_touched_keys)
                        if (active_dofs.find(key) != active_dofs.end())
                            direct_pressure_legacy_dof_keys.push_back(key);
                    ZIRAN_ASSERT(direct_pressure_legacy_dof_keys.size()
                            == direct_pressure_dof_keys.size(),
                        "direct pressure virtual-block traversal lost an active DOF");
                    direct_pressure_dof_lookup.reserve(
                        direct_pressure_legacy_dof_keys.size() * 2);
                    for (std::size_t row = 0;
                         row < direct_pressure_legacy_dof_keys.size(); ++row)
                        direct_pressure_dof_lookup.emplace(
                            direct_pressure_legacy_dof_keys[row],
                            static_cast<int>(row));
                    direct_pressure_new_v.assign(
                        direct_pressure_touched_keys.size(), TV::Zero());
                    direct_pressure_dof_count =
                        direct_pressure_legacy_dof_keys.size();
                }
            }
            direct_solved_p.resize(0);
            ZIRAN_INFO("PositionCorrector gridless pressure topology: sources=",
                page_stats.page_sources, " virtual_blocks=",
                direct_pressure_device_topology_only
                    ? direct_pressure_touched_count / (block_x * block_y * block_z)
                    : direct_pressure_page_bases.size(), " touched_nodes=",
                direct_pressure_touched_count, " active_dofs=",
                direct_pressure_dof_count,
                " device_topology_only=",
                direct_pressure_device_topology_only ? 1 : 0,
                " page_key_ms=", page_stats.key_generation_ms,
                " dof_key_ms=", dof_stats.key_generation_ms,
                " resident_sources=", page_stats.resident_sources,
                " page_download_ms=", page_stats.download_ms,
                " dof_download_ms=", dof_stats.download_ms);
#endif
        }
    }

    void runDirectPressureMatrixProbe(
        const Eigen::SparseMatrix<T>* spgrid_lhs,
        const TStack* spgrid_rhs, bool use_legacy_order = false)
    {
        const std::vector<GPU_STAGE2::GridKey>& direct_dof_keys =
            use_legacy_order ? direct_pressure_legacy_dof_keys
                             : direct_pressure_dof_keys;
        const int dof = static_cast<int>(direct_dof_keys.size());
        if (spgrid_lhs || spgrid_rhs) {
            ZIRAN_ASSERT(spgrid_lhs && spgrid_rhs,
                "direct pressure matrix comparison requires both SPGrid inputs");
            ZIRAN_ASSERT(dof == spgrid_lhs->rows()
                    && dof == spgrid_rhs->rows(),
                "direct pressure topology and SPGrid DOF count differ");
        }
        direct_pressure_dof_lookup.clear();
        direct_pressure_dof_lookup.reserve(static_cast<std::size_t>(dof) * 2);
        for (int i = 0; i < dof; ++i)
            direct_pressure_dof_lookup.emplace(direct_dof_keys[i], i);

        direct_pressure_system.valid = false;
        direct_pressure_system.gpu_csr_valid = false;
        direct_pressure_system.host_csr_valid = false;
        direct_pressure_system.host_rhs_valid = false;
        direct_pressure_system.nnz = 0;
        direct_pressure_system.row_offsets.clear();
        direct_pressure_system.column_indices.clear();
        direct_pressure_system.values.clear();
        direct_pressure_system.dof_keys = direct_dof_keys;
        Eigen::SparseMatrix<T>& direct_lhs = direct_pressure_system.lhs;
        TStack& direct_rhs = direct_pressure_system.rhs;
        direct_lhs.resize(dof, dof);
        direct_rhs = TStack::Zero(dof);
        std::vector<Eigen::Triplet<T>> triplets;
        triplets.reserve(static_cast<std::size_t>(dof) * dim * 4);
        for (int row = 0; row < dof; ++row) {
            int x, y, z;
            GPU_STAGE2::unpackGridKey3D(direct_dof_keys[row], x, y, z);
            const IV node { x, y, z };
            bool near_air = false;
            const IV region = IV::Ones() * 3;
            iterateRegion(region, [&](const IV& offset) {
                const IV neighbor = node + offset - IV::Ones();
                if (!directPressureIsSolid(neighbor) && directPressureDof(neighbor) < 0)
                    near_air = true;
            });
            for (int d = 0; d < dim; ++d)
                for (int delta = -1; delta <= 1; delta += 2) {
                    IV neighbor = node;
                    neighbor[d] += delta;
                    if (!directPressureIsSolid(neighbor)) {
                        const int col = directPressureDof(neighbor);
                        if (col < 0) {
                            near_air = true;
                            triplets.emplace_back(row, row, (T)1);
                        } else {
                            triplets.emplace_back(row, row, (T)1);
                            triplets.emplace_back(row, col, (T)-1);
                        }
                    }
                }
            const T rho_rest = sim.particles.mass.array[0]
                / (sim.particles.DataManager::get(element_measure_name<T>())).array[0];
            T rho_star = directPressureDensity(node + IV::Ones());
            if (near_air)
                rho_star = std::max(rho_star, rho_rest);
            rho_star = std::max(rho_star, rho_rest * (T)0.5);
            rho_star = std::min(rho_star, rho_rest * (T)1.5);
            direct_rhs(row) = -(rho_rest - rho_star) * sim.dx * sim.dx
                / sim.dt / sim.dt;
        }
        direct_lhs.setFromTriplets(triplets.begin(), triplets.end());
        direct_pressure_system.nnz =
            static_cast<std::size_t>(direct_lhs.nonZeros());
        direct_pressure_system.host_csr_valid = true;
        direct_pressure_system.host_rhs_valid = true;

        if (!spgrid_lhs) {
            ZIRAN_INFO("PositionCorrector direct pressure matrix: dofs=", dof,
                " direct_nnz=", direct_lhs.nonZeros(),
                " legacy_order=", use_legacy_order,
                " spgrid_oracle=0");
            direct_pressure_system.valid = true;
            return;
        }

        std::vector<GPU_STAGE2::GridKey> spgrid_dof_keys(dof);
        pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
            spgrid_dof_keys[g.idx] = packProbeKey(node);
        });

        double rhs_max_abs = 0;
        for (int row = 0; row < dof; ++row) {
            const auto it = direct_pressure_dof_lookup.find(spgrid_dof_keys[row]);
            ZIRAN_ASSERT(it != direct_pressure_dof_lookup.end(),
                "SPGrid DOF missing from direct pressure topology");
            rhs_max_abs = std::max(rhs_max_abs, std::abs(
                static_cast<double>((*spgrid_rhs)(row)
                    - direct_rhs(it->second))));
        }

        using KeyPair = std::pair<GPU_STAGE2::GridKey, GPU_STAGE2::GridKey>;
        std::map<KeyPair, T> spgrid_entries;
        std::map<KeyPair, T> direct_entries;
        for (int outer = 0; outer < spgrid_lhs->outerSize(); ++outer)
            for (typename Eigen::SparseMatrix<T>::InnerIterator it(*spgrid_lhs, outer);
                 it; ++it)
                spgrid_entries.emplace(KeyPair { spgrid_dof_keys[it.row()],
                    spgrid_dof_keys[it.col()] }, it.value());
        for (int outer = 0; outer < direct_lhs.outerSize(); ++outer)
            for (typename Eigen::SparseMatrix<T>::InnerIterator it(direct_lhs, outer);
                 it; ++it)
                direct_entries.emplace(KeyPair { direct_dof_keys[it.row()],
                    direct_dof_keys[it.col()] }, it.value());

        std::size_t missing = 0;
        std::size_t extra = 0;
        double matrix_max_abs = 0;
        auto sp_it = spgrid_entries.begin();
        auto direct_it = direct_entries.begin();
        while (sp_it != spgrid_entries.end()
            && direct_it != direct_entries.end()) {
            if (sp_it->first == direct_it->first) {
                matrix_max_abs = std::max(matrix_max_abs, std::abs(
                    static_cast<double>(sp_it->second - direct_it->second)));
                ++sp_it;
                ++direct_it;
            } else if (sp_it->first < direct_it->first) {
                ++missing;
                ++sp_it;
            } else {
                ++extra;
                ++direct_it;
            }
        }
        while (sp_it != spgrid_entries.end()) {
            ++missing;
            ++sp_it;
        }
        while (direct_it != direct_entries.end()) {
            ++extra;
            ++direct_it;
        }
        ZIRAN_INFO("PositionCorrector direct pressure matrix: dofs=", dof,
            " spgrid_nnz=", spgrid_lhs->nonZeros(),
            " direct_nnz=", direct_lhs.nonZeros(),
            " missing=", missing, " extra=", extra,
            " max_abs=", matrix_max_abs,
            " rhs_max_abs=", rhs_max_abs,
            " legacy_order=", use_legacy_order);
        direct_pressure_system.valid = true;
    }

    TStack runDirectPressureSolverProbe(const TStack& spgrid_rhs_before_islands,
        bool compare_with_spgrid = true)
    {
        ZIRAN_ASSERT(direct_pressure_system.valid,
            "direct pressure solver probe requires a direct CSR system");
        ZIRAN_ASSERT(direct_pressure_system.host_csr_valid,
            "direct pressure solver probe requires a host CSR");
        const int dof = static_cast<int>(direct_pressure_system.dof_keys.size());
        ZIRAN_ASSERT(spgrid_rhs_before_islands.rows() == dof,
            "direct pressure solver probe RHS size mismatch");

        TStack direct_rhs = TStack::Zero(dof);
        if (direct_pressure_gridless_active) {
            direct_rhs = spgrid_rhs_before_islands;
        } else {
            pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                const int direct_row = directPressureDof(node);
                ZIRAN_ASSERT(direct_row >= 0,
                    "SPGrid pressure node missing from direct solver probe");
                direct_rhs(direct_row) = spgrid_rhs_before_islands(g.idx);
            });
        }
        const std::vector<T> null_spaces = detectIsland(
            direct_pressure_system.lhs, direct_rhs);
        const TStack direct_solution = DirectSolver<T>::solve(
            direct_pressure_system.lhs, direct_rhs, DirectSolver<T>::AMGCL,
            1e-3, -1, 10000, -1, 1, false, false, (T)1e-4,
            null_spaces);

        if (compare_with_spgrid) {
            double max_abs = 0;
            double mean_abs = 0;
            pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                const int direct_row = directPressureDof(node);
                const double error = std::abs(static_cast<double>(
                    solved_p(g.idx) - direct_solution(direct_row)));
                max_abs = std::max(max_abs, error);
                mean_abs += error;
            });
            if (dof)
                mean_abs /= static_cast<double>(dof);
            ZIRAN_INFO("PositionCorrector direct pressure solver: dofs=", dof,
                " max_abs=", max_abs, " mean_abs=", mean_abs);
        } else {
            ZIRAN_INFO("PositionCorrector direct pressure solver: dofs=", dof,
                " primary_writeback=1");
        }
        return direct_solution;
    }

    TStack gatherDirectPressureRhs(const TStack& spgrid_rhs)
    {
        ZIRAN_ASSERT(direct_pressure_system.host_rhs_valid,
            "direct pressure host RHS was not materialized");
        const int dof = static_cast<int>(direct_pressure_system.dof_keys.size());
        ZIRAN_ASSERT(spgrid_rhs.rows() == dof,
            "direct pressure solver RHS size mismatch");
        TStack direct_rhs = TStack::Zero(dof);
        if (direct_pressure_gridless_active) {
            direct_rhs = spgrid_rhs;
        } else {
            pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                const int direct_row = directPressureDof(node);
                ZIRAN_ASSERT(direct_row >= 0,
                    "SPGrid pressure node missing from direct solver RHS");
                direct_rhs(direct_row) = spgrid_rhs(g.idx);
            });
        }
        return direct_rhs;
    }

    // The direct CSR has already been assembled in the legacy DOF order by
    // CUDA. Only the compact CSR/RHS crosses to the host for AMGCL's VexCL
    // backend; this path deliberately does not rebuild an Eigen/SPGrid matrix.
    TStack runDirectPressureGpuSolver(const TStack& spgrid_rhs_before_islands)
    {
        ZIRAN_ASSERT(direct_pressure_system.valid
                && direct_pressure_system.gpu_csr_valid
                && direct_pressure_system.host_csr_valid,
            "GPU direct pressure solver requires GPU direct CSR");
        TStack direct_rhs = gatherDirectPressureRhs(spgrid_rhs_before_islands);
        const std::vector<T> null_spaces = detectIsland(
            direct_pressure_system.lhs, direct_rhs);
        constexpr T diagonal_epsilon = (T)1e-4;
        TStack direct_solution;
        {
            ZIRAN_NAMED_TIMER("directPressureSolveGpu");
            direct_solution = DirectSolver<T>::solveCsrCUDA(
                static_cast<int>(direct_pressure_system.dof_keys.size()),
                static_cast<int>(direct_pressure_system.dof_keys.size()),
                direct_pressure_system.row_offsets,
                direct_pressure_system.column_indices,
                direct_pressure_system.values, direct_rhs,
                (T)1e-3, -1, 10000, -1, 1, diagonal_epsilon, null_spaces);
        }
        ZIRAN_INFO("PositionCorrector direct GPU pressure solver: dofs=",
            direct_solution.rows(), " primary_writeback=1");
        return direct_solution;
    }

#ifdef ENABLE_AMGX
    TStack runDirectPressureAmgxSolver(const TStack& spgrid_rhs_before_islands)
    {
        ZIRAN_ASSERT(direct_pressure_system.valid
                && direct_pressure_system.gpu_csr_valid,
            "AMGX pressure solver requires GPU direct CSR");
        TStack direct_rhs;
        std::vector<T> null_spaces;
        if (direct_pressure_system.host_rhs_valid) {
            direct_rhs = gatherDirectPressureRhs(spgrid_rhs_before_islands);
            null_spaces = direct_pressure_system.host_csr_valid
                ? detectIsland(direct_pressure_system.lhs, direct_rhs)
                : detectDirectPressureIslands(direct_rhs);
        }
        const bool rhs_projected = !null_spaces.empty();
        const bool use_device_pushout_rhs = gpu_pressure_pushout_ready
            && !rhs_projected;
        ZIRAN_ASSERT(use_device_pushout_rhs
                || direct_pressure_system.host_rhs_valid,
            "pressure solve requires either device pushout RHS or host RHS");
        const bool compare_gpu_pressure_postsolve =
            std::getenv("ZIRAN_GPU_PRESSURE_POSTSOLVE_COMPARE") != nullptr;
        const bool download_solution = !prepare_gpu_pressure_postsolve
            || compare_gpu_pressure_postsolve || rhs_projected;
        std::vector<double> solve_rhs;
        if (!use_device_pushout_rhs) {
            solve_rhs.resize(static_cast<std::size_t>(direct_rhs.rows()));
            for (int row = 0; row < direct_rhs.rows(); ++row)
                solve_rhs[static_cast<std::size_t>(row)] =
                    static_cast<double>(direct_rhs(row));
        }
        // Keep AMGX on the exact operator used by the accepted AMGCL_CUDA
        // path: DirectSolver::solveCsrCUDA() regularizes every diagonal by
        // this amount before constructing its VexCL matrix.
        constexpr T diagonal_epsilon = (T)1e-4;
        GPU_STAGE2::PressureAmgxStats stats;
        std::vector<double> solution;
        {
            ZIRAN_NAMED_TIMER("directPressureSolveGpu");
            {
                ZIRAN_NAMED_TIMER("directPressureSolveAmgx");
                GPU_STAGE2::solvePersistentPressureGridAmgx3D(
                    gpu_pressure_grid,
                    use_device_pushout_rhs ? nullptr : &solve_rhs,
                    static_cast<double>(diagonal_epsilon), solution, &stats,
                    download_solution);
            }
        }
        if (!download_solution) {
            if (sim.gpu_full_pipeline_strict) {
                ZIRAN_ASSERT(stats.solve_status == 0 &&
                        use_device_pushout_rhs && solution.empty(),
                    "strict full-GPU PositionCorrector AMGX used host RHS/solution");
            }
            ZIRAN_INFO("PositionCorrector AMGX pressure solver: dofs=",
                stats.rows, " nnz=", stats.nnz,
                " iterations=", stats.iterations,
                " status=", stats.solve_status,
                " diagonal_epsilon=", stats.diagonal_epsilon,
                " current_rhs_h2d=0 device_projected_islands=",
                stats.projected_islands, " solution_host=0",
                " island_projection_ms=", stats.island_projection_ms,
                " compact_ms=", stats.compact_csr_ms,
                " matrix_d2d_ms=", stats.matrix_upload_ms,
                " rhs_transfer_ms=", stats.rhs_upload_ms,
                " setup_ms=", stats.setup_ms,
                " solve_ms=", stats.solve_ms,
                " solution_d2h_ms=", stats.solution_download_ms,
                " total_ms=", stats.total_ms);
            return TStack{};
        }
        ZIRAN_ASSERT(direct_pressure_system.host_csr_valid
                && direct_pressure_system.host_rhs_valid,
            "host pressure solution diagnostics require host CSR/RHS");
        ZIRAN_ASSERT(solution.size()
                == static_cast<std::size_t>(direct_pressure_system.dof_keys.size()),
            "AMGX pressure solution size mismatch");
        TStack direct_solution = TStack::Zero(static_cast<int>(solution.size()));
        for (std::size_t row = 0; row < solution.size(); ++row)
            direct_solution(static_cast<int>(row)) = static_cast<T>(solution[row]);
        const T rhs_norm = direct_rhs.norm();
        const auto relativeResidual = [&](const TStack& solution,
                                          bool include_diagonal_epsilon) {
            TStack residual = direct_pressure_system.lhs * solution - direct_rhs;
            if (include_diagonal_epsilon)
                residual += diagonal_epsilon * solution;
            return rhs_norm > (T)0 ? residual.norm() / rhs_norm : residual.norm();
        };
        // `raw_relative_residual` uses the assembled physical Laplacian A.
        // The solver-equivalent value below uses A + epsilon I, which is the
        // actual operator of both the AMGCL_CUDA and AMGX direct paths.
        const T raw_relative_residual = relativeResidual(direct_solution, false);
        const T regularized_relative_residual = relativeResidual(direct_solution, true);
        if (std::getenv("ZIRAN_AMGX_COMPARE_AMGCL")) {
            TStack amgcl_solution;
            {
                ZIRAN_NAMED_TIMER("directPressureAmgxCompareAmgcl");
                amgcl_solution = DirectSolver<T>::solveCsrCUDA(
                    static_cast<int>(direct_pressure_system.dof_keys.size()),
                    static_cast<int>(direct_pressure_system.dof_keys.size()),
                    direct_pressure_system.row_offsets,
                    direct_pressure_system.column_indices,
                    direct_pressure_system.values, direct_rhs,
                    (T)1e-3, -1, 10000, -1, 1, diagonal_epsilon, null_spaces);
            }
            const TStack difference = direct_solution - amgcl_solution;
            T max_abs_difference = (T)0;
            for (int row = 0; row < difference.rows(); ++row)
                max_abs_difference = std::max(max_abs_difference,
                    std::abs(difference(row)));
            const T amgcl_norm = amgcl_solution.norm();
            const T relative_solution_difference = amgcl_norm > (T)0
                ? difference.norm() / amgcl_norm
                : difference.norm();
            const T amgcl_raw_relative_residual = relativeResidual(
                amgcl_solution, false);
            const T amgcl_regularized_relative_residual = relativeResidual(
                amgcl_solution, true);
            ZIRAN_INFO("PositionCorrector AMGX/AMGCL pressure compare: max_abs=",
                max_abs_difference,
                " rel_l2=", relative_solution_difference,
                " amgx_raw_relative_residual=", raw_relative_residual,
                " amgx_regularized_relative_residual=", regularized_relative_residual,
                " amgcl_raw_relative_residual=", amgcl_raw_relative_residual,
                " amgcl_regularized_relative_residual=",
                amgcl_regularized_relative_residual);
        }
        ZIRAN_INFO("PositionCorrector AMGX pressure solver: dofs=",
            stats.rows, " nnz=", stats.nnz, " iterations=", stats.iterations,
            " status=", stats.solve_status,
            " diagonal_epsilon=", stats.diagonal_epsilon,
            " current_rhs_h2d=", use_device_pushout_rhs ? 0 : 1,
            " rhs_projected=", rhs_projected ? 1 : 0,
            " device_projected_islands=", stats.projected_islands,
            " island_projection_ms=", stats.island_projection_ms,
            " solution_host=1",
            " compact_ms=", stats.compact_csr_ms,
            " matrix_d2d_ms=", stats.matrix_upload_ms,
            " rhs_transfer_ms=", stats.rhs_upload_ms,
            " setup_ms=", stats.setup_ms,
            " solve_ms=", stats.solve_ms,
            " solution_d2h_ms=", stats.solution_download_ms,
            " total_ms=", stats.total_ms,
            " cpu_raw_relative_residual=", raw_relative_residual,
            " cpu_regularized_relative_residual=", regularized_relative_residual);
        return direct_solution;
    }
#endif

    // Assemble the direct pressure CSR on the GPU.  The caller-selected DOF
    // order is preserved in the returned CSR; only the temporary lookup
    // tables are sorted on device.  This is important because AMGCL's
    // iterations are not bit-invariant under a DOF permutation.
    void buildGpuDirectPressureCsr(bool use_legacy_order,
        bool compare_with_host_direct, bool materialize_host_csr,
        bool materialize_host_rhs)
    {
        const std::vector<GPU_STAGE2::GridKey>& keys =
            use_legacy_order ? direct_pressure_legacy_dof_keys
                             : direct_pressure_dof_keys;
        ZIRAN_ASSERT(!keys.empty() || direct_pressure_device_fields_ready,
            "GPU direct pressure CSR requires direct pressure topology");
        if (compare_with_host_direct) {
            ZIRAN_ASSERT(materialize_host_csr,
                "GPU direct CSR comparison requires a host CSR result");
            ZIRAN_ASSERT(materialize_host_rhs,
                "GPU direct CSR comparison requires a host RHS result");
            ZIRAN_ASSERT(direct_pressure_system.valid,
                "GPU direct CSR comparison requires a host direct pressure system");
            ZIRAN_ASSERT(direct_pressure_system.dof_keys == keys,
                "GPU and host direct pressure CSR use different DOF orders");
        }
        const T rho_rest = sim.particles.mass.array[0]
            / (sim.particles.DataManager::get(element_measure_name<T>())).array[0];
        GPU_STAGE2::PressureCsrResult gpu_result;
        GPU_STAGE2::PressureCsrStats stats;
        GPU_STAGE2::PressureGridFieldStats field_stats;
        if (!direct_pressure_device_fields_ready) {
            ZIRAN_NAMED_TIMER("directPressureGridFieldUpdate");
            GPU_STAGE2::updatePersistentPressureGridDirectFields3D(
                gpu_pressure_grid, keys, direct_pressure_touched_keys,
                direct_pressure_solid_flags, &field_stats,
                direct_pressure_gridless_active,
                direct_pressure_gridless_active && use_legacy_order,
                direct_pressure_resident_solid_flags);
        } else {
            field_stats.dofs = direct_pressure_dof_count;
            field_stats.touched_nodes = direct_pressure_touched_count;
            field_stats.resident_touched_input = true;
            field_stats.resident_dof_input = true;
            field_stats.resident_solid_flags = true;
        }
        {
            ZIRAN_NAMED_TIMER("directPressureCsrGpu");
            GPU_STAGE2::assemblePersistentPressureGridCsr3D(
                gpu_pressure_grid,
                static_cast<double>(rho_rest), static_cast<double>(sim.dx),
                static_cast<double>(sim.dt), gpu_result, &stats,
                materialize_host_csr, materialize_host_rhs);
        }
        ZIRAN_ASSERT(stats.solid_lookup_misses == 0,
            "GPU direct pressure CSR queried an uncached solid key");
        const std::size_t row_count = stats.rows;
        ZIRAN_ASSERT(row_count == direct_pressure_dof_count,
            "GPU direct pressure CSR row count changed after topology build");
        ZIRAN_ASSERT((!materialize_host_csr
                        || gpu_result.row_offsets.size() == row_count + 1)
                && (!materialize_host_rhs
                    || gpu_result.rhs.size() == row_count),
            "GPU direct pressure CSR returned an invalid row count");
        ZIRAN_ASSERT(!materialize_host_csr || keys.size() == row_count,
            "host pressure CSR materialization requires host topology");
        Eigen::SparseMatrix<T> gpu_lhs(static_cast<int>(row_count),
            static_cast<int>(row_count));
        std::vector<Eigen::Triplet<T>> triplets;
        if (materialize_host_csr) {
            triplets.reserve(gpu_result.values.size());
            for (std::size_t row = 0; row < row_count; ++row)
                for (int entry = gpu_result.row_offsets[row];
                     entry < gpu_result.row_offsets[row + 1]; ++entry)
                    triplets.emplace_back(static_cast<int>(row),
                        gpu_result.column_indices[static_cast<std::size_t>(entry)],
                        static_cast<T>(gpu_result.values[static_cast<std::size_t>(entry)]));
            gpu_lhs.setFromTriplets(triplets.begin(), triplets.end());
        }

        double rhs_max_abs = 0;
        T matrix_max_abs = 0;
        if (compare_with_host_direct) {
            for (std::size_t i = 0; i < gpu_result.rhs.size(); ++i)
                rhs_max_abs = std::max(rhs_max_abs, std::abs(gpu_result.rhs[i]
                    - static_cast<double>(direct_pressure_system.rhs(
                        static_cast<int>(i)))));
            const Eigen::SparseMatrix<T> difference = gpu_lhs
                - direct_pressure_system.lhs;
            for (int outer = 0; outer < difference.outerSize(); ++outer)
                for (typename Eigen::SparseMatrix<T>::InnerIterator it(difference, outer);
                     it; ++it)
                    matrix_max_abs = std::max(matrix_max_abs, std::abs(it.value()));
        }

        direct_pressure_dof_lookup.clear();
        direct_pressure_dof_lookup.reserve(keys.size() * 2);
        for (std::size_t row = 0; row < keys.size(); ++row)
            direct_pressure_dof_lookup.emplace(keys[row], static_cast<int>(row));
        direct_pressure_system.dof_keys = keys;
        direct_pressure_system.lhs = std::move(gpu_lhs);
        direct_pressure_system.row_offsets = std::move(gpu_result.row_offsets);
        direct_pressure_system.column_indices = std::move(gpu_result.column_indices);
        direct_pressure_system.values.resize(gpu_result.values.size());
        for (std::size_t entry = 0; entry < gpu_result.values.size(); ++entry)
            direct_pressure_system.values[entry] =
                static_cast<T>(gpu_result.values[entry]);
        direct_pressure_system.rhs = TStack::Zero(static_cast<int>(row_count));
        if (materialize_host_rhs)
            for (std::size_t row = 0; row < gpu_result.rhs.size(); ++row)
                direct_pressure_system.rhs(static_cast<int>(row)) =
                    static_cast<T>(gpu_result.rhs[row]);
        direct_pressure_system.gpu_csr_valid = true;
        direct_pressure_system.host_csr_valid = materialize_host_csr;
        direct_pressure_system.host_rhs_valid = materialize_host_rhs;
        direct_pressure_system.nnz = stats.nnz;
        direct_pressure_system.rows = row_count;
        direct_pressure_system.valid = true;

        if (compare_with_host_direct) {
            ZIRAN_INFO("PositionCorrector GPU direct pressure CSR: rows=", stats.rows,
                " nnz=", stats.nnz, " solid_misses=", stats.solid_lookup_misses,
                " host_matrix_max_abs=", matrix_max_abs,
                " host_rhs_max_abs=", rhs_max_abs,
                " host_oracle=1",
                " host_csr=1",
                " host_rhs=1",
                " resident_touched=",
                field_stats.resident_touched_input ? 1 : 0,
                " resident_dofs=",
                field_stats.resident_dof_input ? 1 : 0,
                " resident_solid_flags=",
                field_stats.resident_solid_flags ? 1 : 0,
                " field_upload_ms=", field_stats.upload_ms,
                " upload_ms=", stats.upload_ms,
                " sort_ms=", stats.lookup_sort_ms,
                " assembly_ms=", stats.assembly_ms,
                " download_ms=", stats.download_ms,
                " total_ms=", stats.total_ms);
        } else {
            ZIRAN_INFO("PositionCorrector GPU direct pressure CSR: rows=", stats.rows,
                " nnz=", stats.nnz, " solid_misses=", stats.solid_lookup_misses,
                " host_oracle=0",
                " host_csr=", materialize_host_csr ? 1 : 0,
                " host_rhs=", materialize_host_rhs ? 1 : 0,
                " resident_touched=",
                field_stats.resident_touched_input ? 1 : 0,
                " resident_dofs=",
                field_stats.resident_dof_input ? 1 : 0,
                " resident_solid_flags=",
                field_stats.resident_solid_flags ? 1 : 0,
                " field_upload_ms=", field_stats.upload_ms,
                " upload_ms=", stats.upload_ms,
                " sort_ms=", stats.lookup_sort_ms,
                " assembly_ms=", stats.assembly_ms,
                " download_ms=", stats.download_ms,
                " total_ms=", stats.total_ms);
        }
    }

    bool scatterDensityGpu(SplittingSimulation<T, dim>& Ssim, T rho,
        bool write_spgrid = true)
    {
        if constexpr (dim != 3) {
            return false;
        } else {
#ifndef ENABLE_CUDA_STAGE2_PROBE
            (void)Ssim;
            (void)rho;
            return false;
#else
            const bool resident_fluid_positions =
                hasResidentFluidPositions();
            const GPU_STAGE3A::SolidParticleStateDeviceView solid_device =
                GPU_STAGE3A::getPersistentSolidParticleStateDeviceView(
                    Ssim.gpu_solid_particle_state);
            const bool resident_solid_positions =
                sim.solid_scheme != CONFIG_UPDATE_LAGRANGIAN &&
                solid_device.current_x != nullptr &&
                solid_device.size ==
                    static_cast<std::size_t>(Ssim.particles.count);
            rebuildGpuDensitySources(Ssim, /*build_combined_shadow_sources*/ false,
                resident_fluid_positions, resident_solid_positions);
            gpu_density_result_valid = false;
            GPU_STAGE2::PressureGridFieldStats stats;
            {
                ZIRAN_NAMED_TIMER("pressureDensityScatterGpu");
                GPU_STAGE2::scatterPersistentPressureGridDensitySplit3D(
                    gpu_pressure_grid, gpu_density_static_sources,
                    gpu_density_dynamic_sources,
                    static_cast<double>(rho), static_cast<double>(sim.dx),
                    gpu_density_result, &stats, write_spgrid,
                    resident_fluid_positions
                        ? sim.gpu_velocity_transfer_device.x : nullptr,
                    resident_fluid_positions
                        ? sim.gpu_velocity_transfer_device.size : 0,
                    resident_solid_positions
                        ? solid_device.current_x : nullptr,
                    resident_solid_positions ? solid_device.size : 0);
            }

            std::size_t written_nodes = 0;
            if (write_spgrid) {
                for (std::size_t i = 0; i < gpu_density_result.keys.size(); ++i) {
                    int x, y, z;
                    GPU_STAGE2::unpackGridKey3D(gpu_density_result.keys[i], x, y, z);
                    const IV node { x, y, z };
                    const uint64_t offset =
                        SplittingSimulation<T, dim>::SparseMask::Linear_Offset(
                            to_std_array(node));
                    if (!pressure_grid.page_map->Test_Page(offset))
                        continue;
                    pressure_grid[node].v[0] = static_cast<T>(gpu_density_result.mass[i]);
                    ++written_nodes;
                }
            }
            gpu_density_result_valid = write_spgrid;
            ZIRAN_INFO("PositionCorrector GPU density: sources=", stats.density_sources,
                " nodes=", stats.density_nodes,
                " resident_sources=", stats.resident_sources,
                " resident_fluid=", resident_fluid_positions ? 1 : 0,
                " resident_solid=", resident_solid_positions ? 1 : 0,
                " written_nodes=", written_nodes,
                " reallocations=", stats.reallocations,
                " upload_ms=", stats.upload_ms,
                " key_sort_ms=", stats.key_generation_ms,
                " scatter_ms=", stats.density_scatter_ms,
                " download_ms=", stats.download_ms,
                " total_ms=", stats.total_ms);
            return true;
#endif
        }
    }

    void logProbeStats(const char* label, const GPU_STAGE2::ProbeCompareStats& stats) const
    {
        ZIRAN_INFO("Stage2 GPU position corrector probe ", label,
            ": cpu_nodes=", stats.cpu_nodes,
            " gpu_nodes=", stats.gpu_nodes,
            " missing_on_gpu=", stats.missing_on_gpu,
            " extra_on_gpu=", stats.extra_on_gpu,
            " max_abs_density_error=", stats.max_abs_mass_error,
            " mean_abs_density_error=", stats.mean_abs_mass_error);
    }

    GPU_STAGE2::ProbeCompareStats compareDensityOnQueryKeys(
        const std::vector<GPU_STAGE2::GridKey>& query_keys,
        const std::vector<double>& cpu_density,
        const GPU_STAGE2::ProbeGridResult& gpu_density) const
    {
        if (query_keys.size() != cpu_density.size())
            throw std::runtime_error("compareDensityOnQueryKeys input size mismatch");

        GPU_STAGE2::ProbeCompareStats stats;
        stats.cpu_nodes = query_keys.size();
        stats.gpu_nodes = query_keys.size();
        double sum_abs = 0.0;
        std::size_t common = 0;
        for (std::size_t i = 0; i < query_keys.size(); ++i) {
            const auto it = std::lower_bound(gpu_density.keys.begin(), gpu_density.keys.end(), query_keys[i]);
            if (it == gpu_density.keys.end() || *it != query_keys[i]) {
                ++stats.missing_on_gpu;
                continue;
            }
            const std::size_t index = static_cast<std::size_t>(it - gpu_density.keys.begin());
            const double err = std::abs(cpu_density[i] - gpu_density.mass[index]);
            stats.max_abs_mass_error = std::max(stats.max_abs_mass_error, err);
            sum_abs += err;
            ++common;
        }
        stats.mean_abs_mass_error = common ? sum_abs / static_cast<double>(common) : 0.0;
        stats.gpu_nodes -= stats.missing_on_gpu;
        return stats;
    }

    void runGpuStage2Probe(SplittingSimulation<T, dim>& Ssim)
    {
        if constexpr (dim != 3) {
            return;
        } else {
            ZIRAN_TIMER();
            StdVector<TV>& Xarray = sim.particles.X.array;

            std::vector<GPU_STAGE2::GridKey> active_gpu_keys;
            active_gpu_keys.reserve(Xarray.size());
            for (int i = 0; i < sim.particles.count; ++i) {
                const TV Xp = Xarray[i];
                BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * sim.dx * (T)0.5, sim.dx);
                if (!isSolid(spline.base_node))
                    active_gpu_keys.push_back(packProbeKey(spline.base_node));
            }

            std::vector<GPU_STAGE2::GridKey> active_cpu_keys;
            active_cpu_keys.reserve(static_cast<std::size_t>(pressure_grid.getNumNodes()));
            pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                active_cpu_keys.push_back(packProbeKey(node));
            });
            const GPU_STAGE2::ProbeCompareStats active_stats =
                GPU_STAGE2::compareKeySets(active_cpu_keys, active_gpu_keys);
            logProbeStats("active", active_stats);

            const double vol = dim == 2
                ? static_cast<double>(sim.dx * sim.dx)
                : static_cast<double>(sim.dx * sim.dx * sim.dx);
            const double rho = sim.particles.count > 0
                ? static_cast<double>(sim.particles.mass.array[0]) / vol
                : 0.0;
            rebuildGpuDensitySources(Ssim, /*build_combined_shadow_sources*/ true);

            GPU_STAGE2::ProbeGridResult density_result;
            GPU_STAGE2::PressureGridFieldStats density_gpu_stats;
            {
                ZIRAN_NAMED_TIMER("gpuDensityScatterShadow");
                GPU_STAGE2::scatterPersistentPressureGridDensity3D(
                    gpu_pressure_grid, gpu_density_sources, rho,
                    static_cast<double>(sim.dx), density_result,
                    &density_gpu_stats);
            }
            ZIRAN_INFO("PositionCorrector GPU density shadow: sources=",
                density_gpu_stats.density_sources,
                " nodes=", density_gpu_stats.density_nodes,
                " reallocations=", density_gpu_stats.reallocations,
                " upload_ms=", density_gpu_stats.upload_ms,
                " key_sort_ms=", density_gpu_stats.key_generation_ms,
                " scatter_ms=", density_gpu_stats.density_scatter_ms,
                " download_ms=", density_gpu_stats.download_ms,
                " total_ms=", density_gpu_stats.total_ms);

            std::vector<GPU_STAGE2::GridKey> query_keys;
            std::vector<double> cpu_density;
            query_keys.reserve(active_cpu_keys.size());
            cpu_density.reserve(active_cpu_keys.size());
            pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                const IV query_node = node + IV::Ones();
                query_keys.push_back(packProbeKey(query_node));
                cpu_density.push_back(static_cast<double>(pressure_grid[query_node].v[0]));
            });
            const GPU_STAGE2::ProbeCompareStats density_stats =
                compareDensityOnQueryKeys(query_keys, cpu_density, density_result);
            logProbeStats("density", density_stats);
        }
    }

public:
    ParticlesLevelSet<T, dim> levelset_builder;

    PositionCorrector(SplittingSimulation<T, dim>& sim)
        : sim(sim) {}

    bool computeIsSolid(IV node)
    {
        TV xi = node.template cast<T>() * sim.dx + TV::Ones() * sim.dx;
        TV vi = TV::Zero();
        TM normal_basis = TM::Zero();
        bool inside_boundary = AnalyticCollisionObject<T, dim>::multiObjectCollision(sim.collision_objects, xi, vi, normal_basis);
        T phi;
        TV normal;
        bool inside_solid = levelset_builder.queryInside(xi, phi, normal, 0);
        return inside_boundary || inside_solid;
    }

    bool isSolid(IV node)
    {
        if (direct_pressure_gridless_active)
            return directPressureIsSolid(node);
        if (solid_classification_cache_valid) {
            const T cached = pressure_grid[node].v[1];
            if (cached != 0)
                return cached > 0;
        }
        return computeIsSolid(node);
    }

    void rebuildSolidClassificationCache()
    {
        ZIRAN_TIMER();
        solid_classification_cache_valid = false;
        pressure_grid.iterateTouchedGrid([&](IV node, GridState<T, dim>& g) {
            g.v[1] = computeIsSolid(node) ? (T)1 : (T)-1;
        });
        solid_classification_cache_valid = true;
    }

    T getPressure(IV node)
    {
        if (direct_pressure_gridless_active)
            return directPressureValue(node);
        return pressure_grid[node].idx < 0 ? (T)0 : solved_p(pressure_grid[node].idx);
    }

    // Fully device-resident pushout: reads resident Q1 G2P fluid X/V/gradV
    // and the resident particle level set directly, patches V/gradV in
    // place for particles that hit, and feeds the pushout RHS/face-velocity
    // bridge without any host round trip for dense phi, solid flags,
    // touched/DOF keys or a sparse displacement/face upload. Only used for
    // the box-shell/open-domain collision gate already used by the device
    // solid classification; every other geometry (and any oracle/compare
    // run) keeps the CPU pushout() below untouched.
    bool pushoutGpu()
    {
        if constexpr (dim != 3) {
            return false;
        } else {
#if defined(ENABLE_CUDA_STAGE2_PROBE) && defined(ENABLE_AMGX)
            if (!direct_pressure_gridless_active || !prepare_gpu_pressure_postsolve)
                return false;
            if (std::getenv("ZIRAN_GPU_PRESSURE_PUSHOUT_DISABLE") != nullptr)
                return false;
            const bool supported_collision =
                sim.gpu_box_shell_collision.valid || sim.collision_objects.empty();
            if (!supported_collision)
                return false;
            if (!sim.gpu_velocity_transfer_device_x_valid
                || !sim.gpu_velocity_transfer_device_v_grad_valid
                || sim.gpu_velocity_transfer_device.x == nullptr
                || sim.gpu_velocity_transfer_device.v == nullptr
                || sim.gpu_velocity_transfer_device.grad_v == nullptr
                || sim.gpu_velocity_transfer_device.size
                    != static_cast<std::size_t>(sim.particles.count))
                return false;
            const auto level_set_device =
                GPU_POSITION_CORRECTOR::getParticleLevelSetDeviceView3D(
                    levelset_builder.gpu_context);
            if (!levelset_builder.gpu_result_active || !level_set_device.phi)
                return false;

            const T rho_rest = sim.particles.mass.array[0]
                / (sim.particles.DataManager::get(element_measure_name<T>())).array[0];
            GPU_STAGE2::PressurePushoutGpuStats stats;
            GPU_STAGE2::PushoutHitParticles hit_particles;
            GPU_STAGE2::PushoutNodeReduction node_reduction;
            const bool debug_pushout_gpu =
                std::getenv("ZIRAN_GPU_PUSHOUT_DEBUG") != nullptr;
            GPU_STAGE2::PushoutFaceVelocityDebug face_velocity_debug;
            GPU_STAGE2::PushoutRhsDebug rhs_debug;
            const bool mirror_hit_particles_to_host =
                !sim.gpu_fluid_device_resident || debug_pushout_gpu;
            GPU_STAGE2::computePersistentPressureGridPushoutGpu3D(
                gpu_pressure_grid, sim.gpu_velocity_transfer, level_set_device,
                sim.gpu_box_shell_collision,
                static_cast<std::size_t>(sim.particles.count),
                static_cast<double>(rho_rest), static_cast<double>(sim.dx),
                static_cast<double>(sim.dt), &stats,
                mirror_hit_particles_to_host ? &hit_particles : nullptr,
                debug_pushout_gpu ? &node_reduction : nullptr,
                debug_pushout_gpu ? &face_velocity_debug : nullptr,
                debug_pushout_gpu ? &rhs_debug : nullptr);
            gpu_pressure_pushout_ready = true;

            // The particle-projection kernel patches resident V/gradV in
            // place. Compatibility/debug paths still mirror the sparse hit
            // set into host V/gradV; the full-pipeline resident path keeps
            // the host arrays stale until an explicit output/fallback sync.
            if (mirror_hit_particles_to_host) {
                ZIRAN_ASSERT(
                    hit_particles.particle_id.size()
                        == hit_particles.corrected_v.size(),
                    "GPU pushout hit-particle/velocity size mismatch");
                for (std::size_t k = 0;
                     k < hit_particles.particle_id.size(); ++k) {
                    const int i = hit_particles.particle_id[k];
                    const auto& v = hit_particles.corrected_v[k];
                    sim.particles.V.array[i] = TV(static_cast<T>(v.x),
                        static_cast<T>(v.y), static_cast<T>(v.z));
                    sim.scratch_gradV[i] = TM::Zero();
                }
            }
            if (debug_pushout_gpu) {
                for (std::size_t k = 0; k < hit_particles.particle_id.size(); ++k)
                    ZIRAN_INFO("PositionCorrector GPU pushout hit: particle=",
                        hit_particles.particle_id[k],
                        " node_key=", hit_particles.node_key[k],
                        " sq_disp=", hit_particles.sq_disp[k],
                        " source=", hit_particles.debug_source[k],
                        " v=", sim.particles.V.array[
                            hit_particles.particle_id[k]].transpose());
                ZIRAN_INFO("PositionCorrector GPU pushout: hit_particles=",
                    hit_particles.particle_id.size());
                // Stage 1/2 oracle: per-node max-displacement reduction
                // winner, before and after the ±0.5dx clamp. Compare against
                // the CPU "per-node winner" log line for the same node_key.
                for (std::size_t k = 0; k < node_reduction.node_key.size(); ++k) {
                    const auto& d = node_reduction.disp[k];
                    const auto& cd = node_reduction.clamped_disp[k];
                    ZIRAN_INFO("PositionCorrector GPU pushout per-node winner: "
                        "node_key=", node_reduction.node_key[k],
                        " particle=", node_reduction.particle_id[k],
                        " sq_disp=", node_reduction.sq_disp[k],
                        " disp=", d.x, " ", d.y, " ", d.z,
                        " clamped_disp=", cd.x, " ", cd.y, " ", cd.z);
                }
                // Stage 3 oracle counterpart to "CPU pushout face velocity".
                for (std::size_t k = 0; k < face_velocity_debug.node_key.size(); ++k) {
                    const auto& v = face_velocity_debug.velocity[k];
                    ZIRAN_INFO("PositionCorrector GPU pushout face velocity: "
                        "node_key=", face_velocity_debug.node_key[k],
                        " v=", v.x, " ", v.y, " ", v.z);
                }
                ZIRAN_INFO("PositionCorrector GPU pushout face velocity: count=",
                    face_velocity_debug.node_key.size());
                // Stage 4 oracle counterpart to "CPU pushout rhs delta".
                for (std::size_t k = 0; k < rhs_debug.dof_key.size(); ++k)
                    ZIRAN_INFO("PositionCorrector GPU pushout rhs delta: "
                        "dof_key=", rhs_debug.dof_key[k],
                        " delta=", rhs_debug.delta[k]);
                ZIRAN_INFO("PositionCorrector GPU pushout rhs delta: count=",
                    rhs_debug.dof_key.size());
            }
            ZIRAN_INFO("PositionCorrector GPU pushout: particles=", stats.particles,
                " candidates=", stats.candidates,
                " hit_particles=", hit_particles.particle_id.size(),
                " displacement_nodes=", stats.displacement_nodes,
                " face_nodes=", stats.face_nodes,
                " projection_ms=", stats.projection_ms,
                " reduce_ms=", stats.reduce_ms,
                " rhs_ms=", stats.rhs_ms,
                " face_ms=", stats.face_ms,
                " total_ms=", stats.total_ms);
            return true;
#else
            return false;
#endif
        }
    }

    void pushout(bool reuse_pressure_grid_topology = false)
    {
        ZIRAN_TIMER();
        if (pushoutGpu())
            return;

        StdVector<TV>& Xarray = sim.particles.X.array;
#ifdef ENABLE_CUDA_STAGE2_PROBE
        const bool patch_resident_v_grad =
            sim.gpu_velocity_transfer_device_v_grad_valid
            && sim.gpu_velocity_transfer_device.v != nullptr
            && sim.gpu_velocity_transfer_device.grad_v != nullptr
            && sim.gpu_velocity_transfer_device.size
                == static_cast<std::size_t>(sim.particles.count);
#else
        const bool patch_resident_v_grad = false;
#endif
        std::vector<unsigned char> resident_v_grad_patch_flags(
            patch_resident_v_grad
                ? static_cast<std::size_t>(sim.particles.count) : 0,
            0);
        const bool debug_pushout = std::getenv("ZIRAN_GPU_PUSHOUT_DEBUG") != nullptr;
        std::vector<unsigned char> debug_modified_flags(
            debug_pushout ? static_cast<std::size_t>(sim.particles.count) : 0, 0);
        std::vector<GPU_STAGE2::GridKey> debug_node_key(
            debug_pushout ? static_cast<std::size_t>(sim.particles.count) : 0, 0);
        std::vector<T> debug_sq_disp(
            debug_pushout ? static_cast<std::size_t>(sim.particles.count) : 0, 0);
        std::vector<TV> debug_disp(
            debug_pushout ? static_cast<std::size_t>(sim.particles.count) : 0,
            TV::Zero());
        {
            ZIRAN_NAMED_TIMER("pushoutSolidGridPollute");
            if (direct_pressure_gridless_active)
                solid_grid.polluteFromBlockBases(direct_pressure_solid_block_bases);
            else if (reuse_pressure_grid_topology)
                solid_grid.polluteFromTopology(pressure_grid);
            else
                solid_grid.activatePages(Xarray, sim.dx, -1, 1);
        }

        {
            ZIRAN_NAMED_TIMER("pushoutParticleProjection");
            sim.parallel_for_updating_grid([&](int i) {
                TV Xp = sim.particles.X.array[i];
                BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * sim.dx * 0.5, sim.dx);
                IV node = spline.base_node;

                if (isSolid(node)) {
                    TV Xt = Xp, normal;
                    TV& Vt = sim.particles.V.array[i];
                    TM& Ct = sim.scratch_gradV[i];
                    bool v_grad_modified = false;
                    T phi;
                    for (int k = 0; k < (int)sim.collision_objects.size(); ++k) {
                        if (sim.collision_objects[k]->type == AnalyticCollisionObject<T, dim>::GHOST)
                            continue;
                        bool inside_boundary = sim.collision_objects[k]->queryInside(Xt, phi, normal, 0);
                        if (inside_boundary) {
                            Xt = Xt - phi * normal;
                            Vt -= normal * Vt.dot(normal);
                            Ct = TM::Zero();
                            v_grad_modified = true;
                        }
                    }
                    {
                        bool inside_solid = levelset_builder.queryInside(Xt, phi, normal, 0);
                        if (inside_solid) {
                            Xt = Xt - phi * normal;
                            Vt -= normal * Vt.dot(normal);
                            Ct = TM::Zero();
                            v_grad_modified = true;
                        }
                    }
                    if (v_grad_modified && patch_resident_v_grad)
                        resident_v_grad_patch_flags[
                            static_cast<std::size_t>(i)] = 1;
                    if (v_grad_modified && debug_pushout) {
                        debug_modified_flags[static_cast<std::size_t>(i)] = 1;
                        debug_node_key[static_cast<std::size_t>(i)] =
                            packProbeKey(node);
                        debug_sq_disp[static_cast<std::size_t>(i)] =
                            (Xt - Xp).squaredNorm();
                        debug_disp[static_cast<std::size_t>(i)] = Xt - Xp;
                    }
                    if ((Xt - Xp).squaredNorm() > solid_grid[node].v[0]) {
                        solid_grid[node].v[0] = (Xt - Xp).squaredNorm();
                        solid_grid[node].new_v = Xt - Xp;
                        solid_grid[node].m = 1;
                    }
                }
            });
        }
        if (debug_pushout) {
            std::size_t count = 0;
            for (int i = 0; i < sim.particles.count; ++i)
                if (debug_modified_flags[static_cast<std::size_t>(i)]) {
                    ZIRAN_INFO("PositionCorrector CPU pushout hit: particle=", i,
                        " node_key=", debug_node_key[static_cast<std::size_t>(i)],
                        " sq_disp=", debug_sq_disp[static_cast<std::size_t>(i)],
                        " v=", sim.particles.V.array[i].transpose());
                    ++count;
                }
            ZIRAN_INFO("PositionCorrector CPU pushout: hit_particles=", count);

            // Stage 7 oracle: parallel_for_updating_grid colors particles by
            // block so that same-color concurrent writers should never touch
            // the same node; the per-node max-displacement "if bigger,
            // overwrite" above therefore should NOT need synchronization.
            // Verify that claim directly instead of assuming it: replay the
            // exact same per-particle candidates single-threaded, in a fixed
            // (ascending particle index) order, and compare the resulting
            // per-node winner against what the actual colored-parallel loop
            // above wrote into solid_grid. Any mismatch here is real
            // evidence of a coloring/synchronization bug; agreement rules it
            // out as the source of the GPU/CPU drift.
            struct SequentialWinner {
                int particle_id;
                double sq_disp;
                TV disp;
            };
            std::unordered_map<GPU_STAGE2::GridKey, SequentialWinner> sequential_winner;
            for (int i = 0; i < sim.particles.count; ++i) {
                if (!debug_modified_flags[static_cast<std::size_t>(i)])
                    continue;
                const auto key = debug_node_key[static_cast<std::size_t>(i)];
                const double sq = static_cast<double>(
                    debug_sq_disp[static_cast<std::size_t>(i)]);
                auto it = sequential_winner.find(key);
                if (it == sequential_winner.end() || sq > it->second.sq_disp)
                    sequential_winner[key] = SequentialWinner{
                        i, sq, debug_disp[static_cast<std::size_t>(i)]};
            }
            std::size_t coloring_checked = 0, coloring_mismatches = 0;
            auto report_colored_node = [&](IV node, GridState<T, dim>& g) {
                if (g.m == 0)
                    return;
                const auto key = packProbeKey(node);
                const auto it = sequential_winner.find(key);
                ++coloring_checked;
                const bool matches = it != sequential_winner.end()
                    && it->second.sq_disp == static_cast<double>(g.v[0])
                    && (it->second.disp - g.new_v).squaredNorm() == (T)0;
                if (!matches) {
                    ++coloring_mismatches;
                    ZIRAN_INFO("PositionCorrector CPU pushout coloring MISMATCH: "
                        "node_key=", key, " colored_sq_disp=", g.v[0],
                        " colored_disp=", g.new_v.transpose(),
                        " sequential_sq_disp=",
                        it == sequential_winner.end() ? -1.0 : it->second.sq_disp,
                        " sequential_particle=",
                        it == sequential_winner.end() ? -1 : it->second.particle_id);
                }
                ZIRAN_INFO("PositionCorrector CPU pushout per-node winner: "
                    "node_key=", key,
                    " particle=", it == sequential_winner.end() ? -1 : it->second.particle_id,
                    " sq_disp=", g.v[0], " disp=", g.new_v.transpose());
            };
            // iterateGridSerial() only visits nodes with idx>=0, which
            // solid_grid does not assign until pushoutSolidGridAssignDofs()
            // runs later in this function; iterateTouchedGridSerial() visits
            // every node in every active page regardless of idx, and the
            // report_colored_node lambda already filters on g.m itself.
            solid_grid.iterateTouchedGridSerial(report_colored_node);
            ZIRAN_INFO("PositionCorrector CPU pushout coloring check: checked=",
                coloring_checked, " mismatches=", coloring_mismatches);
        }
#ifdef ENABLE_CUDA_STAGE2_PROBE
        if constexpr (dim == 3) if (patch_resident_v_grad) {
            ZIRAN_NAMED_TIMER("pushoutDeviceParticlePatch");
            std::vector<int> patch_indices;
            std::vector<GPU_STAGE2::Vec3d> patch_velocity;
            std::vector<GPU_STAGE2::Mat3d> patch_grad_velocity;
            for (int i = 0; i < sim.particles.count; ++i) {
                if (!resident_v_grad_patch_flags[static_cast<std::size_t>(i)])
                    continue;
                patch_indices.push_back(i);
                patch_velocity.push_back(toProbeVec(
                    sim.particles.V.array[i]));
                GPU_STAGE2::Mat3d grad;
                for (int col = 0; col < 3; ++col)
                    for (int row = 0; row < 3; ++row)
                        grad.values[row + 3 * col] = static_cast<double>(
                            sim.scratch_gradV[i](row, col));
                patch_grad_velocity.push_back(grad);
            }
            if (!patch_indices.empty()) {
                GPU_STAGE2::VelocityTransferPatchStats patch_stats;
                GPU_STAGE2::patchFluidVelocityTransferParticles3D(
                    sim.gpu_velocity_transfer, patch_indices, patch_velocity,
                    patch_grad_velocity, &patch_stats);
                ZIRAN_INFO(
                    "PositionCorrector GPU particle V/gradV patch: particles=",
                    patch_stats.particles, " upload_ms=", patch_stats.upload_ms,
                    " patch_ms=", patch_stats.patch_ms,
                    " total_ms=", patch_stats.total_ms);
            }
        }
#endif
        {
            ZIRAN_NAMED_TIMER("pushoutSolidGridAssignDofs");
            solid_grid.getNumNodes();
        }
        {
            ZIRAN_NAMED_TIMER("pushoutSolidGridClamp");
            solid_grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
                for (int d = 0; d < dim; ++d)
                    MATH_TOOLS::clamp(g.new_v[d], -(T)0.5 * sim.dx, (T)0.5 * sim.dx);
            });
        }
        if (debug_pushout) {
            // Stage 2 oracle: same nodes, after the ±0.5dx clamp.
            solid_grid.iterateTouchedGridSerial([&](IV node, GridState<T, dim>& g) {
                if (g.m == 0)
                    return;
                ZIRAN_INFO("PositionCorrector CPU pushout clamped: node_key=",
                    packProbeKey(node), " clamped_disp=", g.new_v.transpose());
            });
        }
        if (prepare_gpu_pressure_postsolve) {
            std::vector<std::pair<GPU_STAGE2::GridKey,
                GPU_STAGE2::Vec3d>> entries;
            solid_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                if (g.new_v.squaredNorm() > (T)0)
                    entries.emplace_back(packProbeKey(node), toProbeVec(g.new_v));
            });
            std::sort(entries.begin(), entries.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            gpu_pushout_displacement_keys.reserve(entries.size());
            gpu_pushout_displacement.reserve(entries.size());
            for (const auto& entry : entries) {
                gpu_pushout_displacement_keys.push_back(entry.first);
                gpu_pushout_displacement.push_back(entry.second);
            }
        }
        {
            ZIRAN_NAMED_TIMER("pushoutPressureVelocity");
            if (direct_pressure_gridless_active) {
                solid_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                    for (int d = 0; d < dim; ++d) {
                        IV neighbor = node;
                        neighbor[d] -= 1;
                        const bool lower_solid = isSolid(neighbor);
                        directPressureVelocity(node)[d] = (g.new_v[d]
                            + (lower_solid ? solid_grid[neighbor].new_v[d]
                                                 : g.new_v[d])) / (T)2;
                        if (debug_pushout)
                            ZIRAN_INFO("PositionCorrector CPU pushout face detail: "
                                "write_node_key=", packProbeKey(node), " d=", d,
                                " dir=lower neighbor_key=", packProbeKey(neighbor),
                                " neighbor_solid=", lower_solid ? 1 : 0,
                                " mine=", g.new_v[d], " neighbor_disp=",
                                solid_grid[neighbor].new_v[d],
                                " result=", directPressureVelocity(node)[d]);
                        neighbor = node;
                        neighbor[d] += 1;
                        const bool upper_solid = isSolid(neighbor);
                        directPressureVelocity(neighbor)[d] = (g.new_v[d]
                            + (upper_solid ? solid_grid[neighbor].new_v[d]
                                                 : g.new_v[d])) / (T)2;
                        if (debug_pushout)
                            ZIRAN_INFO("PositionCorrector CPU pushout face detail: "
                                "write_node_key=", packProbeKey(neighbor), " d=", d,
                                " dir=upper neighbor_key=", packProbeKey(neighbor),
                                " neighbor_solid=", upper_solid ? 1 : 0,
                                " mine=", g.new_v[d], " neighbor_disp=",
                                solid_grid[neighbor].new_v[d],
                                " result=", directPressureVelocity(neighbor)[d]);
                    }
                });
            } else {
                solid_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                    for (int d = 0; d < dim; ++d) {
                        IV neighbor = node;
                        neighbor[d] -= 1;
                        pressure_grid[node].new_v[d] = (g.new_v[d] + (isSolid(neighbor) ? solid_grid[neighbor].new_v[d] : g.new_v[d])) / (T)2;
                        neighbor = node;
                        neighbor[d] += 1;
                        pressure_grid[neighbor].new_v[d] = (g.new_v[d] + (isSolid(neighbor) ? solid_grid[neighbor].new_v[d] : g.new_v[d])) / (T)2;
                    }
                });
            }
        }
        if (prepare_gpu_pressure_postsolve) {
            std::vector<std::pair<GPU_STAGE2::GridKey,
                GPU_STAGE2::Vec3d>> entries;
            entries.reserve(direct_pressure_new_v.size() / 8);
            for (std::size_t i = 0; i < direct_pressure_new_v.size(); ++i) {
                if (direct_pressure_new_v[i].squaredNorm() > (T)0)
                    entries.emplace_back(direct_pressure_touched_keys[i],
                        toProbeVec(direct_pressure_new_v[i]));
            }
            std::sort(entries.begin(), entries.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            if (debug_pushout) {
                // Stage 3 oracle: every nonzero face velocity CPU's
                // pushoutPressureVelocity produced this substep.
                for (const auto& entry : entries)
                    ZIRAN_INFO("PositionCorrector CPU pushout face velocity: "
                        "node_key=", entry.first, " v=", entry.second.x, " ",
                        entry.second.y, " ", entry.second.z);
                ZIRAN_INFO("PositionCorrector CPU pushout face velocity: count=",
                    entries.size());
            }
            gpu_pushout_face_keys.reserve(entries.size());
            gpu_pushout_face_velocity.reserve(entries.size());
            for (const auto& entry : entries) {
                gpu_pushout_face_keys.push_back(entry.first);
                gpu_pushout_face_velocity.push_back(entry.second);
            }
        }
        {
            ZIRAN_NAMED_TIMER("pushoutPressureRhs");
            const T rho_rest = sim.particles.mass.array[0]
                / (sim.particles.DataManager::get(element_measure_name<T>())).array[0];
            if (direct_pressure_gridless_active) {
                // Stage 4 oracle: snapshot rhs before the pushout correction
                // so the per-row delta can be compared against the GPU
                // path's own applyPressurePushoutRhsKernel delta.
                std::vector<T> debug_rhs_before;
                if (debug_pushout)
                    debug_rhs_before.assign(rhs.data(),
                        rhs.data() + direct_pressure_legacy_dof_keys.size());
                for (std::size_t row = 0; row < direct_pressure_legacy_dof_keys.size();
                     ++row) {
                    int x, y, z;
                    GPU_STAGE2::unpackGridKey3D(
                        direct_pressure_legacy_dof_keys[row], x, y, z);
                    const IV node { x, y, z };
                    for (int d = 0; d < dim; ++d) {
                        IV neighbor = node;
                        neighbor[d] -= 1;
                        const bool lower_solid = isSolid(neighbor);
                        if (lower_solid)
                            rhs(static_cast<int>(row)) -= solid_grid[node].new_v[d]
                                * rho_rest * sim.dx / sim.dt / sim.dt;
                        IV upper_neighbor = node;
                        upper_neighbor[d] += 1;
                        const bool upper_solid = isSolid(upper_neighbor);
                        if (upper_solid)
                            rhs(static_cast<int>(row)) += solid_grid[upper_neighbor].new_v[d]
                                * rho_rest * sim.dx / sim.dt / sim.dt;
                    }
                }
                if (debug_pushout) {
                    std::size_t printed = 0;
                    for (std::size_t row = 0; row < direct_pressure_legacy_dof_keys.size();
                         ++row) {
                        const T delta = rhs(static_cast<int>(row)) - debug_rhs_before[row];
                        if (delta != (T)0) {
                            ZIRAN_INFO("PositionCorrector CPU pushout rhs delta: "
                                "dof_key=", direct_pressure_legacy_dof_keys[row],
                                " delta=", delta);
                            ++printed;
                        }
                    }
                    ZIRAN_INFO("PositionCorrector CPU pushout rhs delta: count=", printed);
                }
            } else {
                pressure_grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
                    for (int d = 0; d < dim; ++d) {
                        IV neighbor = node;
                        neighbor[d] -= 1;
                        if (isSolid(neighbor)) rhs(g.idx) -= solid_grid[node].new_v[d] * rho_rest * sim.dx / sim.dt / sim.dt;
                        neighbor = node;
                        neighbor[d] += 1;
                        if (isSolid(neighbor)) rhs(g.idx) += solid_grid[neighbor].new_v[d] * rho_rest * sim.dx / sim.dt / sim.dt;
                    }
                });
            }
        }
#ifdef ENABLE_AMGX
        if (prepare_gpu_pressure_postsolve) {
            const T rho_rest = sim.particles.mass.array[0]
                / (sim.particles.DataManager::get(
                    element_measure_name<T>())).array[0];
            GPU_STAGE2::PressurePushoutStats stats;
            const auto level_set_device =
                GPU_POSITION_CORRECTOR::getParticleLevelSetDeviceView3D(
                    levelset_builder.gpu_context);
            GPU_STAGE2::updatePersistentPressureGridPushout3D(
                gpu_pressure_grid, gpu_pushout_displacement_keys,
                gpu_pushout_displacement, gpu_pushout_face_keys,
                gpu_pushout_face_velocity, static_cast<double>(rho_rest),
                static_cast<double>(sim.dx), static_cast<double>(sim.dt),
                level_set_device, sim.gpu_box_shell_collision, &stats);
            gpu_pressure_pushout_ready = true;
            ZIRAN_INFO("PositionCorrector GPU pushout bridge: displacement_nodes=",
                stats.displacement_nodes, " face_nodes=", stats.face_nodes,
                " upload_ms=", stats.upload_ms,
                " rhs_ms=", stats.rhs_update_ms,
                " face_ms=", stats.face_initialize_ms,
                " total_ms=", stats.total_ms);
        }
#endif
    }

    std::vector<T> detectIsland(const Eigen::SparseMatrix<T>& lhs, TStack& rhs)
    {
        int dof = lhs.rows();
        std::vector<int> sum(dof, 0);
        std::vector<bool> vis(dof, false);
        std::vector<std::vector<int>> edges(dof);
        for (int k = 0; k < lhs.outerSize(); ++k)
            for (typename Eigen::SparseMatrix<T>::InnerIterator it(lhs, k); it; ++it) {
                int row = it.row();
                int col = it.col();
                T val = it.value();
                sum[row] += (int)std::round(val);
                edges[row].push_back(col);
            }
        std::vector<T> null_spaces;
        for (int i = 0; i < dof; ++i) {
            if (!vis[i]) {
                bool island = true;
                std::vector<int> line(dof, 0);
                line[0] = i;
                vis[i] = true;
                if (sum[i] != 0) island = false;
                int l = 0, r = 1;
                for (; l < r; ++l) {
                    for (auto nxt : edges[line[l]])
                        if (!vis[nxt]) {
                            line[r++] = nxt;
                            vis[nxt] = true;
                            if (sum[nxt] != 0) island = false;
                        }
                }
                if (island) {
                    std::vector<T> null_space(dof, (T)0);
                    T dot_product = 0;
                    for (int j = 0; j < r; ++j) {
                        null_space[line[j]] = (T)1;
                        dot_product += rhs(line[j]) / (T)r;
                    }
                    for (int j = 0; j < r; ++j)
                        rhs(line[j]) -= dot_product;
                    for (auto v : null_space)
                        null_spaces.push_back(v);
                }
            }
        }

        return null_spaces;
    }

    // The keyed direct grid already contains enough host topology to detect
    // enclosed pressure components. This preserves the island projection
    // policy without downloading the device CSR solely to walk its graph.
    bool processDirectPressureIslands(
        TStack* rhs, std::vector<T>* null_spaces)
    {
        const auto& keys = direct_pressure_legacy_dof_keys;
        const int dof = static_cast<int>(keys.size());
        if (rhs) {
            ZIRAN_ASSERT(rhs->rows() == dof,
                "direct pressure island RHS size mismatch");
        }
        std::vector<unsigned char> visited(static_cast<std::size_t>(dof), 0);
        std::vector<int> queue;
        queue.reserve(static_cast<std::size_t>(dof));
        bool found_island = false;
        for (int seed = 0; seed < dof; ++seed) {
            if (visited[static_cast<std::size_t>(seed)])
                continue;
            queue.clear();
            queue.push_back(seed);
            visited[static_cast<std::size_t>(seed)] = 1;
            bool touches_air = false;
            for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
                int x, y, z;
                GPU_STAGE2::unpackGridKey3D(
                    keys[static_cast<std::size_t>(queue[cursor])], x, y, z);
                for (int axis = 0; axis < dim; ++axis)
                    for (int delta = -1; delta <= 1; delta += 2) {
                        IV neighbor = IV::Zero();
                        neighbor[0] = x;
                        neighbor[1] = y;
                        if constexpr (dim == 3)
                            neighbor[2] = z;
                        neighbor[axis] += delta;
                        if (directPressureIsSolid(neighbor))
                            continue;
                        const int next = directPressureDof(neighbor);
                        if (next < 0) {
                            touches_air = true;
                            continue;
                        }
                        if (!visited[static_cast<std::size_t>(next)]) {
                            visited[static_cast<std::size_t>(next)] = 1;
                            queue.push_back(next);
                        }
                    }
            }
            if (touches_air)
                continue;
            found_island = true;
            if (!rhs || !null_spaces)
                return true;
            const std::size_t null_offset = null_spaces->size();
            null_spaces->resize(null_offset + static_cast<std::size_t>(dof),
                static_cast<T>(0));
            T mean = 0;
            for (const int row : queue) {
                (*null_spaces)[null_offset + static_cast<std::size_t>(row)] =
                    static_cast<T>(1);
                mean += (*rhs)(row) / static_cast<T>(queue.size());
            }
            for (const int row : queue)
                (*rhs)(row) -= mean;
        }
        return found_island;
    }

    std::vector<T> detectDirectPressureIslands(TStack& rhs)
    {
        std::vector<T> null_spaces;
        processDirectPressureIslands(&rhs, &null_spaces);
        return null_spaces;
    }

    void solve(SplittingSimulation<T, dim>& Ssim,
        bool gpu_stage2_probe = false, bool gpu_position_corrector = false,
        bool gpu_position_corrector_density = false,
        bool gpu_position_corrector_pages = false,
        bool gpu_position_corrector_direct_topology_probe = false,
        bool gpu_position_corrector_direct_matrix_probe = false,
        bool gpu_position_corrector_direct_solver_probe = false,
        bool gpu_position_corrector_direct_legacy_order_probe = false,
        bool gpu_position_corrector_direct_legacy_solver_probe = false,
        bool gpu_position_corrector_direct_legacy_solver_writeback = false,
        bool gpu_position_corrector_direct_legacy_solver = false,
        bool gpu_position_corrector_direct_gpu_solver = false,
        bool gpu_position_corrector_direct_amgx_solver = false,
        bool gpu_position_corrector_direct_postsolve = false,
        bool gpu_position_corrector_direct_pushout = false)
    {
        ZIRAN_TIMER();
        solid_classification_cache_valid = false;
        if (sim.gpu_full_pipeline_strict) {
            ZIRAN_ASSERT(gpu_position_corrector &&
                    gpu_position_corrector_density &&
                    gpu_position_corrector_pages &&
                    gpu_position_corrector_direct_amgx_solver &&
                    gpu_position_corrector_direct_postsolve &&
                    gpu_position_corrector_direct_pushout,
                "strict full-GPU PositionCorrector flags are incomplete");
        }
        const bool direct_pressure_primary_requested =
            gpu_position_corrector_direct_legacy_solver
            || gpu_position_corrector_direct_gpu_solver
            || gpu_position_corrector_direct_amgx_solver;
#ifndef ENABLE_AMGX
        ZIRAN_ASSERT(!gpu_position_corrector_direct_amgx_solver,
            "AMGX PositionCorrector solver requested without ENABLE_AMGX");
#endif
        direct_pressure_gridless_active = false;
        direct_pressure_resident_solid_flags = false;
        direct_pressure_device_topology_only = false;
        direct_pressure_device_fields_ready = false;
        direct_pressure_dof_count = 0;
        direct_pressure_touched_count = 0;
#ifdef ENABLE_CUDA_STAGE2_PROBE
        if constexpr (dim == 3)
            direct_pressure_gridless_active = direct_pressure_primary_requested
                && !gpu_position_corrector_direct_topology_probe
                && !gpu_position_corrector_direct_matrix_probe
                && !gpu_position_corrector_direct_solver_probe
                && !gpu_position_corrector_direct_legacy_order_probe
                && !gpu_position_corrector_direct_legacy_solver_probe
                && !gpu_position_corrector_direct_legacy_solver_writeback;
#endif
        if (sim.gpu_full_pipeline_strict) {
            ZIRAN_ASSERT(direct_pressure_gridless_active,
                "strict full-GPU PositionCorrector topology fell back to SPGrid");
        }
        prepare_gpu_pressure_postsolve = direct_pressure_gridless_active
            && gpu_position_corrector_direct_amgx_solver
            && gpu_position_corrector_direct_postsolve;
        gpu_pressure_pushout_ready = false;
        gpu_pushout_displacement_keys.clear();
        gpu_pushout_displacement.clear();
        gpu_pushout_face_keys.clear();
        gpu_pushout_face_velocity.clear();
        const GPU_STAGE3A::SolidParticleStateDeviceView levelset_solid_device =
            GPU_STAGE3A::getPersistentSolidParticleStateDeviceView(
                Ssim.gpu_solid_particle_state);
        const bool resident_levelset_solid =
            sim.solid_scheme != CONFIG_UPDATE_LAGRANGIAN &&
            levelset_solid_device.current_x != nullptr &&
            levelset_solid_device.size ==
                static_cast<std::size_t>(Ssim.particles.count);
        levelset_builder.build(Ssim, (T)0.4 * sim.dx, (T)0.5 * sim.dx,
            (T)0.4 * sim.dx, gpu_position_corrector,
            resident_levelset_solid ? levelset_solid_device.current_x : nullptr,
            resident_levelset_solid ? levelset_solid_device.size : 0,
            /*download_gpu_result*/ !direct_pressure_gridless_active);
        if (sim.gpu_full_pipeline_strict) {
            ZIRAN_ASSERT(resident_levelset_solid &&
                    levelset_builder.gpu_result_active,
                "strict full-GPU PositionCorrector level set fell back to host");
        }

        StdVector<TV>& Xarray = sim.particles.X.array;
        if (direct_pressure_gridless_active) {
            buildGridlessDirectPressureTopology(
                gpu_position_corrector_direct_pushout
                && prepare_gpu_pressure_postsolve);
            if (sim.gpu_full_pipeline_strict) {
                ZIRAN_ASSERT(direct_pressure_device_topology_only &&
                        direct_pressure_device_fields_ready,
                    "strict full-GPU PositionCorrector materialized topology/fields");
            }
        } else {
            {
                ZIRAN_NAMED_TIMER("pressureGridPollute");
                const bool used_gpu_pages = (gpu_position_corrector_pages
                                                || direct_pressure_primary_requested)
                    && pollutePressureGridGpu();
                if (!used_gpu_pages)
                    pressure_grid.activatePages(Xarray, sim.dx, -1, 1);
            }
            rebuildSolidClassificationCache();

            // compute m
            {
                ZIRAN_NAMED_TIMER("pressureComputeMass");
                sim.parallel_for_updating_grid([&](int i) {
                    TV Xp = Xarray[i];
                    BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * sim.dx * 0.5, sim.dx);
                    if (!isSolid(spline.base_node))
                        pressure_grid[spline.base_node].m = 1;
                });
            }
        }

        if (sim.particles.count > 0) {
            ZIRAN_NAMED_TIMER("pressureDensityScatter");
            gpu_density_result_valid = false;
            // compute rho
            T Vol = dim == 2 ? sim.dx * sim.dx : sim.dx * sim.dx * sim.dx;
            T rho = sim.particles.mass.array[0] / Vol;
            const bool used_gpu_density = (gpu_position_corrector_density
                                               || direct_pressure_gridless_active)
                && scatterDensityGpu(Ssim, rho,
                    /*write_spgrid*/ !direct_pressure_gridless_active);
            if (sim.gpu_full_pipeline_strict) {
                ZIRAN_ASSERT(used_gpu_density &&
                        !gpu_density_result_valid,
                    "strict full-GPU PositionCorrector density fell back to host");
            }
            if (!used_gpu_density) {
                ZIRAN_ASSERT(!direct_pressure_gridless_active,
                    "gridless direct pressure requires the persistent GPU density field");
                for (auto& Xp : sim.boundary_positions) {
                    BSplineWeights<T, dim, 1> spline(Xp, sim.dx);
                    uint64_t offset = SplittingSimulation<T, dim>::SparseMask::Linear_Offset(to_std_array(spline.base_node));
                    pressure_grid.iterateKernelWithValidation(spline, offset, [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                        g.v[0] += rho * w;
                    });
                }
                sim.parallel_for_updating_grid([&](int i) {
                    auto& Xp = sim.particles.X.array[i];
                    BSplineWeights<T, dim, 1> spline(Xp, sim.dx);
                    uint64_t offset = SplittingSimulation<T, dim>::SparseMask::Linear_Offset(to_std_array(spline.base_node));
                    pressure_grid.iterateKernelWithValidation(spline, offset, [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                        g.v[0] += rho * w;
                    });
                });
                Ssim.parallel_for_updating_grid([&](int i) {
                    auto& Xp = (sim.solid_scheme != CONFIG_UPDATE_LAGRANGIAN) ? sim.solid_cur_positions[i] : Ssim.particles.X.array[i];
                    BSplineWeights<T, dim, 1> spline(Xp, sim.dx);
                    uint64_t offset = SplittingSimulation<T, dim>::SparseMask::Linear_Offset(to_std_array(spline.base_node));
                    pressure_grid.iterateKernelWithValidation(spline, offset, [&](IV node, T w, TV dw, GridState<T, dim>& g) {
                        g.v[0] += rho * w;
                    });
                });
            }
        }

        int pressure_dof;
        {
            ZIRAN_NAMED_TIMER("pressureGridAssignDofs");
            pressure_dof = direct_pressure_gridless_active
                ? static_cast<int>(direct_pressure_dof_count)
                : pressure_grid.getNumNodes();
        }
        const bool needs_direct_pressure = gpu_position_corrector_direct_topology_probe
            || gpu_position_corrector_direct_matrix_probe
            || gpu_position_corrector_direct_solver_probe
            || gpu_position_corrector_direct_legacy_order_probe
            || gpu_position_corrector_direct_legacy_solver_probe
            || gpu_position_corrector_direct_legacy_solver_writeback
            || gpu_position_corrector_direct_legacy_solver
            || gpu_position_corrector_direct_gpu_solver
            || gpu_position_corrector_direct_amgx_solver;
        // All explicit probe/writeback modes retain their SPGrid oracle.
        // The direct primary mode alone is production and skips it.
        const bool validate_direct_topology =
            gpu_position_corrector_direct_topology_probe
            || gpu_position_corrector_direct_matrix_probe
            || gpu_position_corrector_direct_solver_probe
            || gpu_position_corrector_direct_legacy_order_probe
            || gpu_position_corrector_direct_legacy_solver_probe
            || gpu_position_corrector_direct_legacy_solver_writeback;
        if (needs_direct_pressure && !direct_pressure_gridless_active
            && (!direct_pressure_primary_requested || validate_direct_topology))
            buildDirectPressureTopology(validate_direct_topology);
        const bool needs_legacy_direct_order =
            gpu_position_corrector_direct_legacy_order_probe
            || gpu_position_corrector_direct_legacy_solver_probe
            || gpu_position_corrector_direct_legacy_solver_writeback
            || gpu_position_corrector_direct_legacy_solver
            || gpu_position_corrector_direct_gpu_solver
            || gpu_position_corrector_direct_amgx_solver;
        if (needs_legacy_direct_order && !direct_pressure_gridless_active)
            buildDirectPressureLegacyOrder(validate_direct_topology,
                /*use_pressure_grid_order*/ direct_pressure_primary_requested
                    && !validate_direct_topology);
        if (gpu_stage2_probe)
            runGpuStage2Probe(Ssim);

        const bool direct_pressure_primary = direct_pressure_primary_requested;
        Eigen::SparseMatrix<T> lhs = Eigen::SparseMatrix<T>(pressure_dof, pressure_dof);
        rhs = TStack::Zero(pressure_dof);
        // An explicit matrix probe is allowed to rebuild the legacy SPGrid
        // CSR as an oracle. Production direct mode never enters this block.
        if (!direct_pressure_primary || gpu_position_corrector_direct_matrix_probe) {
            ZIRAN_NAMED_TIMER("pressureMatrixAssembly");
            std::vector<Eigen::Triplet<T>> lhs_tri(pressure_dof * dim * 2 * 2);
            pressure_grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
                bool near_air = false;
                int row = g.idx;
                IV region = IV::Ones() * 3;
                iterateRegion(region, [&](const IV& offset) {
                    IV neighbor = node + offset - IV::Ones();
                    if (!isSolid(neighbor) && pressure_grid[neighbor].idx < 0)
                        near_air = true;
                });
                int cnt = row * dim * 2 * 2;
                for (int d = 0; d < dim; ++d)
                    for (int delta = -1; delta <= 1; delta += 2) {
                        IV neighbor = node;
                        neighbor[d] += delta;
                        if (!isSolid(neighbor)) {
                            int col = pressure_grid[neighbor].idx;
                            if (col < 0) {
                                near_air = true;
                                lhs_tri[cnt++] = Eigen::Triplet<T>(row, row, (T)1);
                            }
                            else {
                                lhs_tri[cnt++] = Eigen::Triplet<T>(row, row, (T)1);
                                lhs_tri[cnt++] = Eigen::Triplet<T>(row, col, (T)-1);
                            }
                        }
                    }
                T rho_rest = sim.particles.mass.array[0] / (sim.particles.DataManager::get(element_measure_name<T>())).array[0];
                T rho_star = pressure_grid[node + IV::Ones()].v[0];
                if (near_air) rho_star = std::max(rho_star, rho_rest);
                rho_star = std::max(rho_star, rho_rest * (T)0.5);
                rho_star = std::min(rho_star, rho_rest * (T)1.5);
                rhs(row) = -(rho_rest - rho_star) * sim.dx * sim.dx / sim.dt / sim.dt;
            });
            int cnt = 0;
            for (int i = 0; i < pressure_dof * dim * 2 * 2; ++i)
                if (lhs_tri[i].value() != 0)
                    lhs_tri[cnt++] = lhs_tri[i];
            lhs_tri.resize(cnt);
            lhs.setFromTriplets(lhs_tri.begin(), lhs_tri.end());
        }

        if (direct_pressure_primary) {
            // The primary direct path deliberately does not build the old
            // SPGrid-host CSR.  Topology still comes from the direct keyed
            // field in legacy DOF order, then CUDA assembles the actual
            // matrix.  The legacy order is retained solely for numerical
            // equivalence with the established pressure solve.
            if (gpu_position_corrector_direct_matrix_probe) {
                // Probe-only: first establish host direct == SPGrid, then
                // establish GPU direct == host direct before adopting GPU CSR.
                runDirectPressureMatrixProbe(&lhs, &rhs,
                    /*use_legacy_order*/ true);
                buildGpuDirectPressureCsr(/*use_legacy_order*/ true,
                    /*compare_with_host_direct*/ true,
                    /*materialize_host_csr*/ true,
                    /*materialize_host_rhs*/ true);
            } else {
                const bool host_csr_compatibility =
                    !prepare_gpu_pressure_postsolve
                    || std::getenv("ZIRAN_AMGX_COMPARE_AMGCL") != nullptr
                    || std::getenv("ZIRAN_GPU_PRESSURE_POSTSOLVE_COMPARE") != nullptr
                    || std::getenv("ZIRAN_GPU_PRESSURE_CSR_HOST_RESULT") != nullptr;
                const bool topology_has_island = !host_csr_compatibility
                    && processDirectPressureIslands(nullptr, nullptr);
                const bool materialize_host_csr =
                    host_csr_compatibility || topology_has_island;
                const bool materialize_host_rhs = materialize_host_csr
                    || !gpu_position_corrector_direct_pushout;
                ZIRAN_INFO("PositionCorrector direct primary: GPU CSR, host_spgrid_csr=0");
                buildGpuDirectPressureCsr(/*use_legacy_order*/ true,
                    /*compare_with_host_direct*/ false,
                    materialize_host_csr, materialize_host_rhs);
            }
            // Pushout appends its solid-boundary terms directly in legacy
            // keyed row order. The compatibility path still seeds the old
            // SPGrid vector; the gridless path has no such vector.
            if (direct_pressure_gridless_active) {
                rhs = direct_pressure_system.rhs;
            } else {
                pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                    const int direct_row = directPressureDof(node);
                    ZIRAN_ASSERT(direct_row >= 0,
                        "direct pressure RHS is missing an SPGrid DOF");
                    rhs(g.idx) = direct_pressure_system.rhs(direct_row);
                });
            }
        } else if (gpu_position_corrector_direct_matrix_probe
            || gpu_position_corrector_direct_solver_probe
            || gpu_position_corrector_direct_legacy_solver_probe
            || gpu_position_corrector_direct_legacy_solver_writeback
            || gpu_position_corrector_direct_legacy_solver) {
            runDirectPressureMatrixProbe(&lhs, &rhs,
                gpu_position_corrector_direct_legacy_solver_probe
                    || gpu_position_corrector_direct_legacy_solver_writeback
                    || gpu_position_corrector_direct_legacy_solver);
        }
        if (gpu_position_corrector_direct_matrix_probe && !direct_pressure_primary)
            buildGpuDirectPressureCsr(
                gpu_position_corrector_direct_legacy_solver_probe
                    || gpu_position_corrector_direct_legacy_solver_writeback
                    || gpu_position_corrector_direct_legacy_solver,
                /*compare_with_host_direct*/ true,
                /*materialize_host_csr*/ true,
                /*materialize_host_rhs*/ true);

        {
            // P0 diagnostics: per-solve size/topology fingerprint. Read-only,
            // does not change any simulation value.
            const std::size_t boundary_count = sim.boundary_positions.size();
            const std::size_t fluid_count = static_cast<std::size_t>(sim.particles.count);
            const std::size_t solid_count = static_cast<std::size_t>(Ssim.particles.count);
            const std::size_t source_count = boundary_count + fluid_count + solid_count;
            const std::size_t diagnostics_nnz = direct_pressure_primary
                ? direct_pressure_system.nnz
                : static_cast<std::size_t>(lhs.nonZeros());
            std::uint64_t key_hash = 1469598103934665603ull; // FNV-1a offset basis
            if (direct_pressure_gridless_active) {
                for (const GPU_STAGE2::GridKey key : direct_pressure_legacy_dof_keys) {
                    key_hash ^= static_cast<std::uint64_t>(key);
                    key_hash *= 1099511628211ull; // FNV-1a prime
                }
            } else {
                pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                    key_hash ^= static_cast<std::uint64_t>(packProbeKey(node));
                    key_hash *= 1099511628211ull; // FNV-1a prime
                });
            }
            ZIRAN_INFO("PositionCorrector pressure solve diagnostics: ",
                "pressure_dof=", pressure_dof,
                " csr_nnz=", static_cast<long long>(diagnostics_nnz),
                " source_count=", source_count,
                " (boundary=", boundary_count,
                " fluid=", fluid_count,
                " solid=", solid_count, ")",
                " active_key_hash=", key_hash);
        }

        if (gpu_position_corrector_direct_pushout)
            pushout(/*reuse_pressure_grid_topology*/ true);
        else
            pushout();
        if (sim.gpu_full_pipeline_strict) {
            ZIRAN_ASSERT(gpu_pressure_pushout_ready,
                "strict full-GPU PositionCorrector pushout RHS fell back to host");
        }

        bool gpu_pressure_particle_correction_ready = false;
        const bool compare_gpu_pressure_postsolve =
            std::getenv("ZIRAN_GPU_PRESSURE_POSTSOLVE_COMPARE") != nullptr;
#ifdef ENABLE_AMGX
        GPU_STAGE2::PressureParticleCorrectionView gpu_pressure_particle_correction;
        GPU_STAGE2::PressureParticleCorrectionStats
            gpu_pressure_particle_correction_stats;
#endif
        {
            ZIRAN_NAMED_TIMER("pressureSolve");
            const TStack rhs_before_islands = rhs;
            auto remap_direct_solution = [&](const TStack& direct_solution) {
                TStack remapped_solution = TStack::Zero(direct_solution.rows());
                pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                    const int direct_row = directPressureDof(node);
                    ZIRAN_ASSERT(direct_row >= 0,
                        "direct pressure writeback is missing an SPGrid DOF");
                    remapped_solution(g.idx) = direct_solution(direct_row);
                });
                return remapped_solution;
            };
            if (direct_pressure_primary) {
                TStack direct_solution;
                if (gpu_position_corrector_direct_amgx_solver) {
#ifdef ENABLE_AMGX
                    direct_solution = runDirectPressureAmgxSolver(rhs_before_islands);
#else
                    ZIRAN_ASSERT(false,
                        "AMGX PositionCorrector solver requested without ENABLE_AMGX");
#endif
                } else if (gpu_position_corrector_direct_gpu_solver) {
                    direct_solution = runDirectPressureGpuSolver(rhs_before_islands);
                } else {
                    direct_solution = runDirectPressureSolverProbe(rhs_before_islands,
                        /*compare_with_spgrid*/ false);
                }
                if (direct_pressure_gridless_active)
                    direct_solved_p = direct_solution;
                else
                    solved_p = remap_direct_solution(direct_solution);
            } else {
                std::vector<T> null_spaces = detectIsland(lhs, rhs);
                solved_p = DirectSolver<T>::solve(lhs, rhs, DirectSolver<T>::AMGCL,
                    1e-3, -1, 10000, -1, 1, false, false, (T)1e-4,
                    null_spaces);
                if (gpu_position_corrector_direct_solver_probe
                    || gpu_position_corrector_direct_legacy_solver_probe
                    || gpu_position_corrector_direct_legacy_solver_writeback) {
                    const TStack direct_solution = runDirectPressureSolverProbe(
                        rhs_before_islands);
                    if (gpu_position_corrector_direct_legacy_solver_writeback)
                        solved_p = remap_direct_solution(direct_solution);
                }
            }
        }

#ifdef ENABLE_AMGX
        // AMGX has already downloaded its solution into gpu_pressure_grid's
        // persistent device buffer.  Consume that buffer directly instead of
        // rebuilding direct_pressure_new_v and interpolating it on the CPU.
        // Full-pipeline mode keeps corrected X device-resident for the next
        // substep; compatibility/oracle paths may still request the D2H.
        if constexpr (dim == 3) {
            if (direct_pressure_gridless_active
                && gpu_position_corrector_direct_amgx_solver
                && gpu_position_corrector_direct_postsolve) {
                const T rho_rest = sim.particles.mass.array[0]
                    / (sim.particles.DataManager::get(element_measure_name<T>())).array[0];
                static_assert(sizeof(TV) == sizeof(GPU_STAGE2::Vec3d),
                    "GPU pressure correction requires tightly packed 3D positions");
                const bool resident_fluid_positions =
                    hasResidentFluidPositions();
                {
                    ZIRAN_NAMED_TIMER("pressureVelocityUpdateGpu");
                    GPU_STAGE2::applyPersistentPressureGridAmgxCorrection3D(
                        gpu_pressure_grid,
                        resident_fluid_positions
                            ? sim.gpu_velocity_transfer_device.x
                            : static_cast<const void*>(Xarray.data()),
                        static_cast<std::size_t>(sim.particles.count),
                        static_cast<double>(rho_rest), static_cast<double>(sim.dx),
                        static_cast<double>(sim.dt),
                        gpu_pressure_particle_correction,
                        &gpu_pressure_particle_correction_stats,
                        resident_fluid_positions,
                        !sim.gpu_fluid_device_resident ||
                            compare_gpu_pressure_postsolve);
                }
                gpu_pressure_particle_correction_ready = true;
                sim.gpu_velocity_transfer_device.x =
                    gpu_pressure_particle_correction.device_x;
                sim.gpu_velocity_transfer_device.size =
                    gpu_pressure_particle_correction.size;
                sim.gpu_velocity_transfer_device_x_valid =
                    gpu_pressure_particle_correction.device_x != nullptr
                    && gpu_pressure_particle_correction.size
                        == static_cast<std::size_t>(sim.particles.count);
                sim.gpu_velocity_transfer_host_x_valid = false;
                ZIRAN_INFO("PositionCorrector GPU postsolve: particles=",
                    gpu_pressure_particle_correction_stats.particles,
                    " touched=", gpu_pressure_particle_correction_stats.touched_nodes,
                    " resident_particles=",
                    gpu_pressure_particle_correction_stats.resident_particle_input
                        ? 1 : 0,
                    " particle_h2d_ms=",
                    gpu_pressure_particle_correction_stats.particle_upload_ms,
                    " velocity_ms=",
                    gpu_pressure_particle_correction_stats.velocity_update_ms,
                    " correction_ms=",
                    gpu_pressure_particle_correction_stats.particle_correction_ms,
                    " particle_d2h_ms=",
                    gpu_pressure_particle_correction_stats.download_ms,
                    " total_ms=", gpu_pressure_particle_correction_stats.total_ms);
            }
        }
#endif

        if (sim.gpu_full_pipeline_strict) {
            ZIRAN_ASSERT(gpu_pressure_particle_correction_ready &&
                    sim.gpu_velocity_transfer_device_x_valid,
                "strict full-GPU PositionCorrector postsolve fell back to host");
            ZIRAN_INFO("Strict full-GPU PositionCorrector verified: ",
                "topology=device density=device CSR=device AMGX=device ",
                "postsolve=device pushout=device");
        }

        if (!gpu_pressure_particle_correction_ready
            || compare_gpu_pressure_postsolve) {
        {
            ZIRAN_NAMED_TIMER("pressureVelocityUpdate");
            const T rho_rest = sim.particles.mass.array[0]
                / (sim.particles.DataManager::get(element_measure_name<T>())).array[0];
            const T coeff = -sim.dt * sim.dt / rho_rest / sim.dx;
            if (direct_pressure_gridless_active) {
                for (std::size_t row = 0; row < direct_pressure_legacy_dof_keys.size();
                     ++row) {
                    int x, y, z;
                    GPU_STAGE2::unpackGridKey3D(
                        direct_pressure_legacy_dof_keys[row], x, y, z);
                    const IV node { x, y, z };
                    const T center_p = direct_solved_p(static_cast<int>(row));
                    for (int d = 0; d < dim; ++d) {
                        IV neighbor = node;
                        neighbor[d] -= 1;
                        if (!isSolid(neighbor))
                            directPressureVelocity(node)[d] = coeff
                                * (center_p - getPressure(neighbor));
                        MATH_TOOLS::clamp(directPressureVelocity(node)[d],
                            -(T)0.5 * sim.dx, (T)0.5 * sim.dx);
                        neighbor = node;
                        neighbor[d] += 1;
                        if (!isSolid(neighbor))
                            directPressureVelocity(neighbor)[d] = coeff
                                * (getPressure(neighbor) - center_p);
                        MATH_TOOLS::clamp(directPressureVelocity(neighbor)[d],
                            -(T)0.5 * sim.dx, (T)0.5 * sim.dx);
                    }
                }
            } else {
                pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                    T center_p = solved_p(g.idx);
                    for (int d = 0; d < dim; ++d) {
                        IV neighbor = node;
                        neighbor[d] -= 1;
                        if (!isSolid(neighbor)) g.new_v[d] = coeff * (center_p - getPressure(neighbor));
                        MATH_TOOLS::clamp(g.new_v[d], -(T)0.5 * sim.dx, (T)0.5 * sim.dx);
                        neighbor = node;
                        neighbor[d] += 1;
                        if (!isSolid(neighbor)) pressure_grid[neighbor].new_v[d] = coeff * (getPressure(neighbor) - center_p);
                        MATH_TOOLS::clamp(pressure_grid[neighbor].new_v[d], -(T)0.5 * sim.dx, (T)0.5 * sim.dx);
                    }
                });
            }
        }
        }

#ifdef ENABLE_AMGX
        if (gpu_pressure_particle_correction_ready
            && compare_gpu_pressure_postsolve) {
            std::vector<GPU_STAGE2::GridKey> gpu_velocity_keys;
            std::vector<GPU_STAGE2::Vec3d> gpu_velocity;
            GPU_STAGE2::downloadPersistentPressureGridFaceVelocity3D(
                gpu_pressure_grid, gpu_velocity_keys, gpu_velocity);
            double max_velocity_error = 0;
            GPU_STAGE2::GridKey max_velocity_key = 0;
            TV max_cpu_velocity = TV::Zero();
            GPU_STAGE2::Vec3d max_gpu_velocity {};
            std::size_t missing_velocity_key = 0;
            for (std::size_t i = 0; i < gpu_velocity_keys.size(); ++i) {
                const auto it = direct_pressure_touched_lookup.find(
                    gpu_velocity_keys[i]);
                if (it == direct_pressure_touched_lookup.end()) {
                    ++missing_velocity_key;
                    continue;
                }
                const TV& cpu_v = direct_pressure_new_v[
                    static_cast<std::size_t>(it->second)];
                const auto& gpu_v = gpu_velocity[i];
                const double local_error = std::max({
                    std::abs(static_cast<double>(cpu_v[0]) - gpu_v.x),
                    std::abs(static_cast<double>(cpu_v[1]) - gpu_v.y),
                    std::abs(static_cast<double>(cpu_v[2]) - gpu_v.z)});
                if (local_error > max_velocity_error) {
                    max_velocity_error = local_error;
                    max_velocity_key = gpu_velocity_keys[i];
                    max_cpu_velocity = cpu_v;
                    max_gpu_velocity = gpu_v;
                }
            }
            ZIRAN_INFO("PositionCorrector GPU postsolve face oracle: keys=",
                gpu_velocity_keys.size(), " missing=", missing_velocity_key,
                " max_velocity_error=", max_velocity_error,
                " key=", max_velocity_key,
                " cpu=", max_cpu_velocity,
                " gpu=", max_gpu_velocity.x, ",", max_gpu_velocity.y,
                ",", max_gpu_velocity.z);
        }
#endif

        {
            ZIRAN_NAMED_TIMER("particleCorrection");
            if (gpu_pressure_particle_correction_ready) {
#ifdef ENABLE_AMGX
                const GPU_STAGE2::Vec3d* corrected_x =
                    gpu_pressure_particle_correction.x;
                ZIRAN_ASSERT((corrected_x != nullptr ||
                                 sim.gpu_fluid_device_resident)
                        && gpu_pressure_particle_correction.size
                            == static_cast<std::size_t>(sim.particles.count),
                    "GPU pressure correction output size mismatch");
                if (compare_gpu_pressure_postsolve) {
                    double max_particle_error = 0;
                    double mean_particle_error = 0;
                    for (int i = 0; i < sim.particles.count; ++i) {
                        TV cpu_x = Xarray[i];
                        for (int d = 0; d < dim; ++d) {
                            TV face_pos = cpu_x - TV::Ones() * sim.dx;
                            face_pos[d] += (T)0.5 * sim.dx;
                            BSplineWeights<T, dim, 1> spline(face_pos, sim.dx);
                            for (int ii = 0; ii < 2; ++ii)
                                for (int jj = 0; jj < 2; ++jj)
                                    for (int kk = 0; kk < 2; ++kk) {
                                        const IV node = spline.base_node + IV { ii, jj, kk };
                                        cpu_x[d] += spline.w[0](ii) * spline.w[1](jj)
                                            * spline.w[2](kk)
                                            * directPressureVelocity(node)[d];
                                    }
                        }
                        const auto& gpu_x = corrected_x[static_cast<std::size_t>(i)];
                        const double dx0 = static_cast<double>(cpu_x[0]) - gpu_x.x;
                        const double dx1 = static_cast<double>(cpu_x[1]) - gpu_x.y;
                        const double dx2 = static_cast<double>(cpu_x[2]) - gpu_x.z;
                        const double error = std::sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2);
                        max_particle_error = std::max(max_particle_error, error);
                        mean_particle_error += error;
                    }
                    ZIRAN_INFO("PositionCorrector GPU postsolve particle oracle: max_l2=",
                        max_particle_error, " mean_l2=",
                        mean_particle_error / std::max(1, sim.particles.count));
                }
                if (corrected_x) {
                    sim.parallel_for_updating_grid([&](int i) {
                        const auto& x = corrected_x[static_cast<std::size_t>(i)];
                        Xarray[i] = TV(static_cast<T>(x.x), static_cast<T>(x.y),
                            static_cast<T>(x.z));
                    });
                    sim.gpu_velocity_transfer_host_x_valid = true;
                }
#endif
            } else {
            sim.parallel_for_updating_grid([&](int i) {
                TV& Xp = Xarray[i];
                for (int d = 0; d < dim; ++d) {
                    TV face_pos = Xp - TV::Ones() * sim.dx;
                    face_pos[d] += (T)0.5 * sim.dx;
                    BSplineWeights<T, dim, 1> spline(face_pos, sim.dx);
                    if (direct_pressure_gridless_active) {
                        if constexpr (dim == 3) {
                            for (int ii = 0; ii < 2; ++ii) {
                                const T wi = spline.w[0](ii);
                                for (int jj = 0; jj < 2; ++jj) {
                                    const T wij = wi * spline.w[1](jj);
                                    for (int kk = 0; kk < 2; ++kk) {
                                        const IV node = spline.base_node + IV { ii, jj, kk };
                                        Xp[d] += wij * spline.w[2](kk)
                                            * directPressureVelocity(node)[d];
                                    }
                                }
                            }
                        }
                    } else {
                        uint64_t offset = MpmSimulationBase<T, dim>::SparseMask::Linear_Offset(to_std_array(spline.base_node));
                        pressure_grid.iterateKernel(spline, offset, [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) { Xp[d] += w * g.new_v[d]; });
                    }
                }
            });
            }
        }
    }
};

} // namespace ZIRAN
