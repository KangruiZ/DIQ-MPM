#include "GpuSparseGridProbe.h"
#include "GpuOperatorProbe.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}

} // namespace

int main()
{
    using namespace ZIRAN::GPU_STAGE2;

    const GridKey key = packGridKey3D(12, -3, 45);
    int x = 0, y = 0, z = 0;
    unpackGridKey3D(key, x, y, z);
    require(x == 12 && y == -3 && z == 45, "pack/unpack should preserve signed coordinates");

    std::vector<GridKey> keys = {
        packGridKey3D(1, 2, 3),
        packGridKey3D(1, 2, 3),
        packGridKey3D(2, 2, 3),
        packGridKey3D(0, 2, 3),
    };
    sortUniqueKeys(keys);
    require(keys.size() == 3, "sortUniqueKeys should remove duplicate keys");
    require(std::is_sorted(keys.begin(), keys.end()), "sortUniqueKeys should leave keys sorted");

    const std::vector<Vec3d> xq = {
        Vec3d{0.25, 0.25, 0.25},
        Vec3d{0.75, 0.25, 0.25},
    };
    const std::vector<double> mass = {2.0, 4.0};
    ProbeGridResult result;
    runVelocityGridMassProbe(xq, mass, 1.0, true, result);

    require(result.keys.size() == result.mass.size(), "keys/mass size mismatch");
    require(!result.keys.empty(), "probe should activate at least one grid node");

    double total_mass = 0;
    for (double m : result.mass)
        total_mass += m;
    require(std::abs(total_mass - 6.0) < 1e-12, "linear scatter should conserve mass");

    ProbeGridResult scalar_result;
    runLinearScalarScatterProbe(xq, mass, 1.0, false, scalar_result);
    double scalar_total = 0;
    for (double m : scalar_result.mass)
        scalar_total += m;
    require(std::abs(scalar_total - 6.0) < 1e-12, "generic scalar scatter should conserve scalar values");
    const GridKey scalar_key = packGridKey3D(0, 0, 0);
    auto scalar_it = std::lower_bound(scalar_result.keys.begin(), scalar_result.keys.end(), scalar_key);
    require(scalar_it != scalar_result.keys.end() && *scalar_it == scalar_key, "generic scalar scatter should contain query key");
    const std::size_t scalar_index = static_cast<std::size_t>(scalar_it - scalar_result.keys.begin());
    require(std::abs(scalar_result.mass[scalar_index] - 1.40625) < 1e-12, "generic scalar scatter should match query key value");

    const ProbeCompareStats key_stats = compareKeySets(
        {packGridKey3D(2, 0, 0), packGridKey3D(1, 0, 0), packGridKey3D(1, 0, 0)},
        {packGridKey3D(2, 0, 0), packGridKey3D(3, 0, 0), packGridKey3D(3, 0, 0)});
    require(key_stats.cpu_nodes == 2, "key-set compare should unique CPU keys");
    require(key_stats.gpu_nodes == 2, "key-set compare should unique GPU keys");
    require(key_stats.missing_on_gpu == 1, "key-set compare should count missing keys");
    require(key_stats.extra_on_gpu == 1, "key-set compare should count extra keys");

    ProbeGridResult pressure_result;
    runPressureGridCountProbe(xq, 1.0, true, pressure_result);
    require(pressure_result.keys.size() == pressure_result.mass.size(), "pressure keys/mass size mismatch");
    require(pressure_result.keys.size() == 2, "order-0 pressure probe should activate two cells");
    double pressure_count = 0;
    for (double m : pressure_result.mass)
        pressure_count += m;
    require(std::abs(pressure_count - 2.0) < 1e-12, "pressure probe should count quadrature samples");

    const std::vector<Vec3d> face_xq = {Vec3d{0.5, 0.5, 0.5}};
    const std::vector<Vec3d> face_n = {Vec3d{1.0, 0.0, 0.0}};
    ProbeGridResult face_result;
    runFaceGridCountProbe(face_xq, face_n, 1.0, true, face_result);
    require(face_result.keys.size() == face_result.mass.size(), "face keys/mass size mismatch");
    require(face_result.keys.size() == 2, "face probe should activate two delta-offset samples");
    double face_count = 0;
    for (double m : face_result.mass)
        face_count += m;
    require(std::abs(face_count - 2.0) < 1e-12, "face probe should count two face samples");

    PersistentSpatialSearch3D search;
    SpatialSearchStats search_stats;
    rebuildPersistentSpatialSearch(search,
        {Vec3d{0.0, 0.0, 0.0}, Vec3d{2.0, 0.0, 0.0}}, 1.0,
        &search_stats);
    require(search_stats.particles == 2,
        "persistent search should retain particle count");
    require(anyPointWithinRadius(search, {Vec3d{0.25, 0.0, 0.0}}, 0.5),
        "persistent search should find a close point");
    require(!anyPointWithinRadius(search, {Vec3d{1.0, 1.0, 0.0}}, 0.5),
        "persistent search should reject distant points");

#ifdef ENABLE_CUDA_STAGE2_PROBE
    ZIRAN::GPU_STAGE3A::SolidParticleStateInput3D owner_input;
    ZIRAN::GPU_STAGE3A::Mat3d identity;
    identity.values[0] = identity.values[4] = identity.values[8] = 1.0;
    owner_input.rest_x = {
        ZIRAN::GPU_STAGE3A::Vec3d{0.0, 0.0, 0.0},
        ZIRAN::GPU_STAGE3A::Vec3d{0.5, 0.0, 0.0}};
    owner_input.current_x = owner_input.rest_x;
    owner_input.velocity.resize(2);
    owner_input.affine.resize(2);
    owner_input.grad_velocity.resize(2);
    owner_input.f = {identity, identity};
    owner_input.j = {1.0, 1.0};
    owner_input.mass = {2.0, 3.0};
    owner_input.volume = {0.5, 0.5};
    owner_input.lambda = {4.0, 4.0};
    ZIRAN::GPU_STAGE3A::PersistentSolidParticleState3D owner;
    ZIRAN::GPU_STAGE3A::initializePersistentSolidParticleState3D(
        owner, owner_input);
    const auto owner_view =
        ZIRAN::GPU_STAGE3A::getPersistentSolidParticleStateDeviceView(owner);

    SpatialSearchStats resident_search_stats;
    rebuildPersistentSpatialSearch(search, {}, 1.0, &resident_search_stats,
        owner_view.current_x, owner_view.size);
    require(resident_search_stats.resident_particle_positions,
        "persistent search should accept resident solid positions");
    require(anyPointWithinRadius(search, {Vec3d{0.25, 0.0, 0.0}}, 0.3),
        "resident persistent search should retain owner positions");
    std::vector<Vec3d> resident_normals;
    SpatialSearchStats resident_normal_stats;
    computeCubicSphNormals(search, {}, 1.0, resident_normals,
        &resident_normal_stats, owner_view.mass, owner_view.size);
    require(resident_normal_stats.resident_particle_mass,
        "SPH normals should accept resident solid masses");
    require(resident_normals.size() == owner_view.size,
        "resident SPH normal output size mismatch");

    PersistentPressureGrid3D resident_density_grid;
    ProbeGridResult resident_density_result;
    PressureGridFieldStats resident_density_stats;
    scatterPersistentPressureGridDensitySplit3D(resident_density_grid, {}, {},
        1.0, 1.0, resident_density_result, &resident_density_stats,
        /*download_result*/ true, owner_view.current_x, 1,
        owner_view.current_x + 1, 1);
    require(resident_density_stats.density_sources == 2 &&
            resident_density_stats.resident_sources == 2,
        "split density should count both resident particle ranges");
    double resident_density_total = 0.0;
    for (double value : resident_density_result.mass)
        resident_density_total += value;
    require(std::abs(resident_density_total - 2.0) < 1e-12,
        "split resident density should conserve source values");

    PersistentPressureGrid3D resident_topology_grid;
    std::vector<GridKey> unused_page_keys;
    rebuildPersistentPressureGridPages3D(resident_topology_grid, xq, 1.0,
        unused_page_keys, nullptr, /*download_base_keys*/ false);
    std::vector<GridKey> source_dof_keys;
    extractPersistentPressureGridDofSourceKeys3D(
        resident_topology_grid, 1.0, source_dof_keys);
    VirtualBlockOrderResult virtual_topology;
    buildPersistentPressureGridVirtualBlocks3D(
        resident_topology_grid, 2, 2, 2, virtual_topology);
    std::vector<unsigned char> topology_solid_flags(
        virtual_topology.touched_keys.size(), 0);
    std::vector<GridKey> legacy_dof_keys;
    for (const GridKey topology_key : virtual_topology.touched_keys)
        if (std::binary_search(source_dof_keys.begin(), source_dof_keys.end(),
                topology_key))
            legacy_dof_keys.push_back(topology_key);
    PressureGridFieldStats resident_topology_stats;
    updatePersistentPressureGridDirectFields3D(resident_topology_grid,
        legacy_dof_keys, virtual_topology.touched_keys, topology_solid_flags,
        &resident_topology_stats, /*reuse_resident_virtual_touched_keys*/ true,
        /*rebuild_resident_legacy_dof_keys*/ true);
    require(resident_topology_stats.resident_touched_input &&
            resident_topology_stats.resident_dof_input,
        "direct pressure fields should reuse resident topology keys");

    PersistentPressureGrid3D device_only_topology_grid;
    rebuildPersistentPressureGridPages3D(device_only_topology_grid, xq, 1.0,
        unused_page_keys, nullptr, /*download_base_keys*/ false);
    std::vector<GridKey> no_host_source_dof_keys;
    PressureGridFieldStats no_host_source_stats;
    extractPersistentPressureGridDofSourceKeys3D(device_only_topology_grid,
        1.0, no_host_source_dof_keys, &no_host_source_stats,
        /*download_keys*/ false);
    VirtualBlockOrderResult no_host_virtual_topology;
    VirtualBlockOrderStats no_host_virtual_stats;
    buildPersistentPressureGridVirtualBlocks3D(device_only_topology_grid,
        2, 2, 2, no_host_virtual_topology, &no_host_virtual_stats,
        /*download_result*/ false);
    require(no_host_source_dof_keys.empty()
            && no_host_virtual_topology.page_keys.empty()
            && no_host_virtual_topology.touched_keys.empty(),
        "device-only pressure topology should not materialize host keys");
    std::vector<unsigned char> no_host_solid_flags(
        no_host_virtual_stats.touched_nodes, 0);
    PressureGridFieldStats device_only_topology_stats;
    updatePersistentPressureGridDirectFields3D(device_only_topology_grid,
        {}, {}, no_host_solid_flags, &device_only_topology_stats,
        /*reuse_resident_virtual_touched_keys*/ true,
        /*rebuild_resident_legacy_dof_keys*/ true,
        /*reuse_resident_solid_flags*/ false);
    require(device_only_topology_stats.dofs == legacy_dof_keys.size()
            && device_only_topology_stats.touched_nodes
                == no_host_virtual_stats.touched_nodes,
        "device-only pressure topology should preserve resident row counts");
#endif

    PersistentFluidAnalytic3D build_context;
    std::vector<VelocityGridNode> velocity_nodes;
    std::vector<VelocityGridNode> g2p_fallback_nodes;
    Mat3d zero_affine;
    FluidVelocityGridStats velocity_stats;
    rebuildFluidVelocityGrid(build_context,
        {Vec3d{0.75, 0.75, 0.75}}, {2.0}, {Vec3d{1.0, 2.0, 3.0}},
        {zero_affine}, {Vec3d{0.75, 0.75, 0.75}}, {2.0}, 1.0, 0.1,
        Vec3d{0.0, -10.0, 0.0}, velocity_nodes, g2p_fallback_nodes,
        &velocity_stats);
    require(velocity_stats.nodes == 8,
        "Q1 persistent velocity grid should contain eight candidate nodes");
    double rebuilt_mass = 0.0;
    for (const VelocityGridNode& node : velocity_nodes)
        rebuilt_mass += node.mass;
    require(std::abs(rebuilt_mass - 2.0) < 1e-12,
        "persistent velocity grid should conserve final mass");
    std::vector<VelocityGridNode> empty_final_nodes;
    std::vector<VelocityGridNode> empty_final_fallback;
    rebuildFluidVelocityGrid(build_context,
        {Vec3d{0.75, 0.75, 0.75}}, {2.0}, {Vec3d{1.0, 2.0, 3.0}},
        {zero_affine}, {}, {}, 1.0, 0.1, Vec3d{0.0, -10.0, 0.0},
        empty_final_nodes, empty_final_fallback);
    require(empty_final_nodes.empty() && empty_final_fallback.size() == 8,
        "initial-only Q1 nodes must be retained solely for G2P");

    // A box shell without an outlet must not interpret its unused hole bounds
    // as a real cutout.  Example 3 uses this geometry for the fluid container.
    BoxShellCollision3D closed_shell;
    closed_shell.outer_min = Vec3d{0.0, 0.0, 0.0};
    closed_shell.outer_max = Vec3d{4.0, 4.0, 4.0};
    closed_shell.inner_min = Vec3d{1.0, 1.0, 1.0};
    closed_shell.inner_max = Vec3d{3.0, 3.0, 3.0};
    closed_shell.hole_min = Vec3d{1.0, 1.0, 1.0};
    closed_shell.hole_max = Vec3d{3.0, 3.0, 3.0};
    closed_shell.has_hole = 0;
    closed_shell.valid = 1;
    FluidAnalyticResult closed_shell_result;
    rebuildFluidAnalyticQuadrature(build_context,
        {Vec3d{1.5, 1.5, 1.5}}, {1.0}, {}, 1.0, 1.0, closed_shell,
        closed_shell_result);
    require(!closed_shell_result.volume_xq.empty(),
        "closed-shell fluid sample must survive GPU analytical pruning");

    std::vector<SolidPressureGridNode> solid_pressure_nodes;
    ZIRAN::GPU_STAGE3A::DeviceGridNodeMap solid_pressure_device_nodes;
    rebuildSolidPressureGrid(build_context,
        {Vec3d{0.5, 0.5, 0.5}, Vec3d{0.6, 0.5, 0.5}},
        {1.1, 0.9}, {2.0, 3.0}, {10.0, 10.0}, 1.0,
        solid_pressure_nodes, nullptr, nullptr,
        &solid_pressure_device_nodes, true);
    require(solid_pressure_nodes.size() == 1,
        "solid Q0 pressure grid should merge particles in one cell");
    require(std::abs(solid_pressure_nodes[0].mass - 4.9) < 1e-12,
        "solid Q0 pressure grid should rebuild weighted mass");
#ifdef ENABLE_CUDA_STAGE2_PROBE
    // The CPU backend above still checks the Q0 grid and mass. Device map
    // download and hostless assembly require the CUDA implementation.
    const auto solid_pressure_device_copy =
        ZIRAN::GPU_STAGE3A::copyDeviceGridNodeMapToHost(
            solid_pressure_device_nodes);
    require(solid_pressure_device_copy.size() == solid_pressure_nodes.size(),
        "solid Q0 device pressure map size mismatch");
    require(solid_pressure_device_copy[0].key == solid_pressure_nodes[0].key
            && solid_pressure_device_copy[0].idx == 0,
        "solid Q0 device pressure map must retain the legacy DOF order");
    require(std::abs(solid_pressure_device_copy[0].mass
                         - solid_pressure_nodes[0].mass)
                < 1e-12
            && std::abs(solid_pressure_device_copy[0].v.x
                            - solid_pressure_nodes[0].compliance)
                < 1e-12
            && std::abs(solid_pressure_device_copy[0].v.y
                            - solid_pressure_nodes[0].pressure)
                < 1e-12,
        "solid Q0 device pressure fields must match the host oracle");

    std::vector<SolidPressureGridNode> hostless_solid_pressure_nodes;
    ZIRAN::GPU_STAGE3A::DeviceGridNodeMap hostless_solid_pressure_map;
    rebuildSolidPressureGrid(build_context,
        {Vec3d{0.5, 0.5, 0.5}, Vec3d{0.6, 0.5, 0.5}},
        {1.1, 0.9}, {2.0, 3.0}, {10.0, 10.0}, 1.0,
        hostless_solid_pressure_nodes, nullptr, nullptr,
        &hostless_solid_pressure_map, false);
    require(hostless_solid_pressure_nodes.empty()
            && hostless_solid_pressure_map.storage
            && hostless_solid_pressure_map.size == 1,
        "strict solid Q0 pressure rebuild must remain device-only");
#endif

    return 0;
}
