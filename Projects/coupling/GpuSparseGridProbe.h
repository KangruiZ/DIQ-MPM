#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace ZIRAN::GPU_STAGE3A {
struct DeviceCsrMatrix;
struct DeviceVector;
struct DeviceGridNodeMap;
struct SolidParticleStateDeviceView;
}

namespace ZIRAN::GPU_POSITION_CORRECTOR {
struct ParticleLevelSetDeviceView3D;
}

namespace ZIRAN::GPU_STAGE2 {

using GridKey = std::uint64_t;

struct Vec3d {
    double x;
    double y;
    double z;
};

struct Mat3d {
    double values[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

struct ProbeGridResult {
    std::vector<GridKey> keys;
    std::vector<double> mass;
};

struct ProbeCompareStats {
    std::size_t cpu_nodes = 0;
    std::size_t gpu_nodes = 0;
    std::size_t missing_on_gpu = 0;
    std::size_t extra_on_gpu = 0;
    double max_abs_mass_error = 0;
    double mean_abs_mass_error = 0;
};

// Persistent Q1 scalar scatter used by PositionCorrector. The API owns its
// device buffers and returns a compact, sorted key/value field for the host
// compatibility path. It intentionally has no SPGrid dependency.
struct PersistentLinearScatter3D {
    std::shared_ptr<void> storage;
};

struct LinearScatterStats {
    std::size_t sources = 0;
    std::size_t nodes = 0;
    std::size_t candidate_keys = 0;
    std::size_t reallocations = 0;
    double upload_ms = 0;
    double key_sort_ms = 0;
    double scatter_ms = 0;
    double download_ms = 0;
    double total_ms = 0;
};

// Persistent extractor for the degree-1 B-spline base nodes used by
// MpmGrid::activatePages(). Unlike the scalar scatter above it emits one key per
// particle, so the host can rebuild the *current* SPGrid page map with the
// original page-clear semantics.
struct PersistentPagePollute3D {
    std::shared_ptr<void> storage;
};

struct PagePolluteStats {
    std::size_t sources = 0;
    std::size_t base_nodes = 0;
    std::size_t reallocations = 0;
    double upload_ms = 0;
    double key_generation_ms = 0;
    double download_ms = 0;
    double total_ms = 0;
};

// Persistent device-side Q1 velocity lookup and particle transfer.  Node keys
// are sorted while their velocity rows retain the caller's DOF indices.
// Particle output stays in original particle order so the CPU collision and
// state-writeback compatibility path needs no permutation handling.
struct PersistentVelocityTransfer3D {
    std::shared_ptr<void> storage;
};

// Non-owning view of persistent pinned host staging. It remains valid until
// the next call on the same context; the compatibility writeback consumes it
// immediately, avoiding three temporary std::vector copies per substep.
struct VelocityTransferHostView {
    const Vec3d* x = nullptr;
    const Vec3d* v = nullptr;
    const Mat3d* grad_v = nullptr;
    const double* pressure = nullptr;
    std::size_t size = 0;
};

// Non-owning device view of the same post-G2P particle fields.  The
// PersistentVelocityTransfer3D context owns this storage; consumers may use
// it until the next transfer on that context.
struct VelocityTransferDeviceView {
    const void* x = nullptr;
    const void* v = nullptr;
    const void* grad_v = nullptr;
    const void* pressure = nullptr;
    std::size_t size = 0;
};

struct VelocityTransferDownloadMask {
    bool x = true;
    bool v = true;
    bool grad_v = true;
    bool pressure = true;
};

struct VelocityTransferDownloadStats {
    std::size_t particles = 0;
    std::size_t bytes = 0;
    double download_ms = 0;
};

struct VelocityTransferStats {
    std::size_t particles = 0;
    std::size_t velocity_nodes = 0;
    std::size_t g2p_fallback_nodes = 0;
    std::size_t reallocations = 0;
    bool resident_particle_input = false;
    bool resident_node_input = false;
    bool resident_fallback_input = false;
    std::size_t host_download_bytes = 0;
    double particle_upload_ms = 0;
    double node_upload_ms = 0;
    double g2p_ms = 0;
    double download_ms = 0;
    double total_ms = 0;
};

struct VelocityTransferPatchStats {
    std::size_t particles = 0;
    std::size_t reallocations = 0;
    double upload_ms = 0;
    double patch_ms = 0;
    double total_ms = 0;
};

struct FluidBoundaryProjectionStats {
    std::size_t particles = 0;
    std::size_t projected_particles = 0;
    double projection_ms = 0;
};

// Scalar host result of a device reduction over the current resident fluid
// particle owner. Only this fixed-size record crosses the PCIe boundary.
struct FluidParticleStateStats {
    std::size_t particles = 0;
    Vec3d min_corner;
    Vec3d max_corner;
    double max_speed = 0;
    std::size_t outside_particles = 0;
    double reduction_ms = 0;
};

struct PressureCsrResult;
struct PressureCsrStats;

// The PositionCorrector pressure field owns all current-frame sparse data in
// one persistent device context.  The key arrays are compact (not SPGrid
// pages): page_base_keys preserves the source order needed by the host virtual
// block reconstruction, density_keys/density are canonical sorted nodes,
// and the direct fields retain legacy DOF order plus the touched solid cache.
// Capacities survive across PositionCorrector substeps and frames.
struct PersistentPressureGrid3D {
    std::shared_ptr<void> storage;
};

struct PressureGridFieldStats {
    std::size_t page_sources = 0;
    std::size_t density_sources = 0;
    std::size_t resident_sources = 0;
    std::size_t density_nodes = 0;
    std::size_t dofs = 0;
    std::size_t touched_nodes = 0;
    bool resident_touched_input = false;
    bool resident_dof_input = false;
    bool resident_solid_flags = false;
    std::size_t reallocations = 0;
    double upload_ms = 0;
    double key_generation_ms = 0;
    double density_scatter_ms = 0;
    double download_ms = 0;
    double total_ms = 0;
};

// Ordered virtual blocks are the compact keyed equivalent of the old SPGrid
// page traversal. page_keys hold page coordinates (not node coordinates);
// touched_keys expand those pages in the legacy block-local loop order.
struct VirtualBlockOrderResult {
    std::vector<GridKey> page_keys;
    std::vector<GridKey> touched_keys;
};

struct VirtualBlockOrderStats {
    std::size_t sources = 0;
    std::size_t candidate_pages = 0;
    std::size_t virtual_blocks = 0;
    std::size_t touched_nodes = 0;
    std::size_t reallocations = 0;
    double ordering_ms = 0;
    double expansion_ms = 0;
    double download_ms = 0;
    double total_ms = 0;
};

// Fixed-width (diagonal plus up to six neighbours) pressure CSR assembly for
// the PositionCorrector direct keyed grid. Rows retain the caller's DOF order;
// lookup tables are sorted internally only for neighbour queries.
struct PersistentPressureCsr3D {
    std::shared_ptr<void> storage;
};

struct PressureCsrResult {
    std::vector<int> row_offsets;
    std::vector<int> column_indices;
    std::vector<double> values;
    std::vector<double> rhs;
};

struct PressureCsrStats {
    std::size_t rows = 0;
    std::size_t nnz = 0;
    std::size_t solid_lookup_misses = 0;
    std::size_t reallocations = 0;
    double upload_ms = 0;
    double lookup_sort_ms = 0;
    double assembly_ms = 0;
    double download_ms = 0;
    double total_ms = 0;
};

#ifdef ENABLE_AMGX
// Timings for the AMGX device-resident PositionCorrector pressure solve.
// `matrix_upload_ms` measures AMGX's device CSR ingestion. `rhs_upload_ms`
// includes the current host-side RHS copy required after pushout/island edits
// and AMGX's device-vector ingestion.
struct PressureAmgxStats {
    std::size_t rows = 0;
    std::size_t nnz = 0;
    int iterations = 0;
    int solve_status = -1;
    std::size_t projected_islands = 0;
    double diagonal_epsilon = 0;
    double island_projection_ms = 0;
    double compact_csr_ms = 0;
    double matrix_upload_ms = 0;
    double rhs_upload_ms = 0;
    double setup_ms = 0;
    double solve_ms = 0;
    double solution_download_ms = 0;
    double total_ms = 0;
};

struct PressurePushoutStats {
    std::size_t displacement_nodes = 0;
    std::size_t face_nodes = 0;
    double upload_ms = 0;
    double rhs_update_ms = 0;
    double face_initialize_ms = 0;
    double total_ms = 0;
};

// Stats for the fully device-resident PositionCorrector pushout (see
// computePersistentPressureGridPushoutGpu3D below).
struct PressurePushoutGpuStats {
    std::size_t particles = 0;
    std::size_t candidates = 0;
    std::size_t displacement_nodes = 0;
    std::size_t face_nodes = 0;
    bool resident_fluid_input = false;
    double projection_ms = 0;
    double reduce_ms = 0;
    double rhs_ms = 0;
    double face_ms = 0;
    double total_ms = 0;
};

// Non-owning host/device views of the final particle positions corrected by
// the device-resident pressure solution. The views are valid until the next
// call on the same PersistentPressureGrid3D context.
struct PressureParticleCorrectionView {
    const Vec3d* x = nullptr;
    const void* device_x = nullptr;
    std::size_t size = 0;
};

struct PressureParticleCorrectionStats {
    std::size_t particles = 0;
    std::size_t touched_nodes = 0;
    bool resident_particle_input = false;
    double particle_upload_ms = 0;
    double velocity_update_ms = 0;
    double particle_correction_ms = 0;
    double download_ms = 0;
    double total_ms = 0;
};

// Reusable device buffers for an AMGX solve whose matrix is already a
// GPU_STAGE3A device CSR (the coupling-system Stage3B product).
struct PersistentAmgxCsrSolver {
    std::shared_ptr<void> storage;
};

struct AmgxCsrStats {
    std::size_t rows = 0;
    std::size_t nnz = 0;
    int iterations = 0;
    int solve_status = -1;
    double diagonal_epsilon = 0;
    double values_copy_ms = 0;
    double rhs_upload_ms = 0;
    double matrix_upload_ms = 0;
    double setup_ms = 0;
    double solve_ms = 0;
    double solution_download_ms = 0;
    double total_ms = 0;
};
#endif

// Production spatial-search context used by the coupling buildSystem path.
// The implementation owns reusable device buffers; rebuilding changes the
// active sizes but retains capacity across coupling substeps.
struct PersistentSpatialSearch3D {
    std::shared_ptr<void> storage;
};

// Reuses a PersistentSpatialSearch3D solid cell list to distribute fluid
// quadrature mass/momentum to neighbouring solid particles. The result is
// deliberately a compact host compatibility view: later buildSystem stages
// still consume the existing quadrature vectors while their device-owner
// migration is in progress.
struct PersistentInterfaceQuad3D {
    std::shared_ptr<void> storage;
};

struct InterfaceQuadAtomicResult {
    std::vector<double> fluid_mass;
    std::vector<int> active_solid_ids;
    std::vector<double> solid_mass;
    std::vector<Vec3d> solid_momentum;
    bool compact_host_result = false;
    bool host_result_materialized = true;
};

// Contiguous device fields for the active solid interface samples. Entries
// retain increasing solid-particle id order, matching the legacy host
// quadrature order, and remain valid until the next rebuild on the context.
struct InterfaceQuadratureDeviceView3D {
    const void* rest_x = nullptr;
    const void* current_x = nullptr;
    const void* normal = nullptr;
    const void* length = nullptr;
    std::size_t size = 0;
};

struct InterfaceQuadAtomicStats {
    std::size_t fluid_particles = 0;
    std::size_t solid_particles = 0;
    std::size_t active_solid_particles = 0;
    std::size_t reallocations = 0;
    bool resident_fluid_positions = false;
    bool resident_fluid_velocity = false;
    bool resident_fluid_mass = false;
    double fluid_upload_ms = 0;
    double fine_grid_ms = 0;
    double accumulation_ms = 0;
    double download_ms = 0;
    double total_ms = 0;
};

struct SpatialSearchStats {
    std::size_t particles = 0;
    std::size_t particle_capacity = 0;
    std::size_t reallocations = 0;
    bool resident_particle_positions = false;
    bool resident_particle_mass = false;
    bool resident_query_positions = false;
    double upload_ms = 0;
    double sort_ms = 0;
    double query_ms = 0;
};

struct BoxShellCollision3D {
    Vec3d outer_min;
    Vec3d outer_max;
    Vec3d inner_min;
    Vec3d inner_max;
    Vec3d hole_min;
    Vec3d hole_max;
    // Example 2 has an outlet cut through the shell.  Other box-shell scenes,
    // such as example 3, have no such cutout; do not infer one from placeholder
    // coordinates.
    int has_hole = 0;
    int valid = 0;
};

struct PersistentFluidAnalytic3D {
    std::shared_ptr<void> storage;
};

struct FluidAnalyticResult {
    std::vector<Vec3d> volume_xq;
    std::vector<double> volume_mass;
    std::vector<Vec3d> face_xq;
    std::vector<Vec3d> face_normal;
    std::vector<Vec3d> interface_xq;
    std::vector<GridKey> pressure_keys;
    std::vector<GridKey> face_keys;
};

struct FluidAnalyticStats {
    std::size_t input_samples = 0;
    std::size_t pruned_samples = 0;
    std::size_t active_cells = 0;
    std::size_t output_samples = 0;
    std::size_t face_samples = 0;
    std::size_t reallocations = 0;
    double total_ms = 0.0;
};

struct FluidAnalyticDeviceView3D {
    const void* volume_xq = nullptr;
    const void* volume_mass = nullptr;
    std::size_t volume_count = 0;
    const void* face_xq = nullptr;
    const void* face_normal = nullptr;
    std::size_t face_count = 0;
};

struct FluidLegacyDofOrderStats {
    std::size_t pressure_nodes = 0;
    std::size_t face_nodes = 0;
    double pressure_order_ms = 0.0;
    double face_order_ms = 0.0;
    double total_ms = 0.0;
};

struct VelocityGridNode {
    GridKey key = 0;
    double mass = 0.0;
    Vec3d velocity;
};

struct FluidVelocityGridStats {
    std::size_t input_samples = 0;
    std::size_t final_samples = 0;
    std::size_t nodes = 0;
    // Nodes which had non-zero initial P2G mass but lost their final mass.
    // They are deliberately excluded from the pressure/velocity DOFs, but
    // CPU SPGrid keeps their pre-rebuild `new_v` for the subsequent G2P.
    std::size_t g2p_fallback_nodes = 0;
    std::size_t reallocations = 0;
    bool resident_position_mass = false;
    bool resident_velocity = false;
    bool resident_affine = false;
    double total_ms = 0.0;
};

struct SolidPressureGridNode {
    GridKey key = 0;
    double mass = 0.0;
    double compliance = 0.0;
    double pressure = 0.0;
};

GridKey packGridKey3D(int x, int y, int z);
void unpackGridKey3D(GridKey key, int& x, int& y, int& z);
void sortUniqueKeys(std::vector<GridKey>& keys);

ProbeCompareStats compareKeySets(std::vector<GridKey> cpu_keys,
    std::vector<GridKey> gpu_keys);

void runLinearScalarScatterProbe(const std::vector<Vec3d>& xq,
    const std::vector<double>& values, double dx, bool halfDxAnchor,
    ProbeGridResult& result);

// Every source contributes the same scalar value with degree-1 B-spline
// weights. The key/value result is sorted by GridKey and contains every node
// reached by the source stencils.
void runPersistentUniformLinearScatter3D(
    PersistentLinearScatter3D& context,
    const std::vector<Vec3d>& xq, double value, double dx,
    bool halfDxAnchor, ProbeGridResult& result,
    LinearScatterStats* stats = nullptr);

// Returns one current-frame degree-1 base node per input source, in the
// original source order. No SPGrid pages are persisted or unioned across
// calls. Preserving order is required because SPGrid records page activation
// order and consequently its host DOF traversal order.
void runPersistentLinearBaseKey3D(
    PersistentPagePollute3D& context,
    const std::vector<Vec3d>& xq, double dx, bool halfDxAnchor,
    std::vector<GridKey>& base_keys,
    PagePolluteStats* stats = nullptr);

// `device_velocity` is a contiguous 3*node_count double vector whose rows are
// addressed by velocity_node_indices.  The implementation performs the Q1
// PIC and velocity-gradient gather on CUDA and copies back only particle SoA.
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
    VelocityTransferStats* stats = nullptr,
    VelocityTransferDownloadMask download_mask = {});

// Explicit compatibility/output synchronization. Production consumers use
// VelocityTransferDeviceView and request only fields that must be materialized
// for host I/O or an unsupported fallback.
void downloadPersistentFluidVelocityTransfer3D(
    PersistentVelocityTransfer3D& context,
    VelocityTransferDownloadMask download_mask,
    VelocityTransferHostView& result,
    VelocityTransferDownloadStats* stats = nullptr);

// Synchronize an arbitrary current device view. In particular, PositionCorrector
// may replace `view.x` with its pressure-context output while V/gradV/pressure
// remain owned by the velocity-transfer context.
void downloadFluidVelocityTransferDeviceView3D(
    PersistentVelocityTransfer3D& context,
    const VelocityTransferDeviceView& view,
    VelocityTransferDownloadMask download_mask,
    VelocityTransferHostView& result,
    VelocityTransferDownloadStats* stats = nullptr);

// Reproduces the supported box-shell collision loop performed after Q1 G2P:
// queryInside(X, phi, n, -3*dx), X -= (phi + 2*dx)n and V = 0. The resident
// gradV field is intentionally unchanged, matching the existing CPU path.
void projectFluidBoundaryCollision3D(
    PersistentVelocityTransfer3D& context,
    const BoxShellCollision3D& collision, double dx,
    FluidBoundaryProjectionStats* stats = nullptr);

// Reduces bounds, max |V| and the count outside [domain_min, domain_max] from
// the current device view. This is used by CFL and the common no-cull fast path.
void reduceFluidParticleState3D(
    const VelocityTransferDeviceView& view,
    double domain_min, double domain_max,
    FluidParticleStateStats& result);

// Applies sparse CPU collision/pushout corrections to the resident Q1 G2P
// position, velocity and, when supplied, gradient fields without
// invalidating the full device owner. An empty grad_velocity vector patches
// position/velocity only; an empty position vector patches velocity (and
// gradient) only, leaving the resident position field untouched.
void patchFluidVelocityTransferParticles3D(
    PersistentVelocityTransfer3D& context,
    const std::vector<int>& particle_indices,
    const std::vector<Vec3d>& velocity,
    const std::vector<Mat3d>& grad_velocity,
    VelocityTransferPatchStats* stats = nullptr,
    const std::vector<Vec3d>& position = {});

// Persistent pressure-grid field updates. The two source sets are separate:
// page positions are fluid-only, while density positions include the current
// boundary/fluid/solid contributors. Their device fields live in the same
// PersistentPressureGrid3D instance.
void rebuildPersistentPressureGridPages3D(
    PersistentPressureGrid3D& context, const std::vector<Vec3d>& page_positions,
    double dx, std::vector<GridKey>& page_base_keys,
    PressureGridFieldStats* stats = nullptr, bool download_base_keys = true,
    const void* resident_page_positions = nullptr,
    std::size_t resident_page_count = 0);

// Uses the resident fluid base-key stream to reproduce MpmGrid::activatePages()
// page activation exactly on GPU. Each page retains the earliest particle and
// local activation rank, then pages are emitted in that rank order.
void buildPersistentPressureGridVirtualBlocks3D(
    PersistentPressureGrid3D& context, int block_x, int block_y, int block_z,
    VirtualBlockOrderResult& result,
    VirtualBlockOrderStats* stats = nullptr,
    bool download_result = true);

// Emits sorted unique degree-1 base keys for X - 0.5 dx from the page-source
// positions already resident in the persistent grid. This is the active-
// pressure-node source stream; GPU deduplication avoids downloading a second
// 1:1 particle key stream solely for DOF discovery.
void extractPersistentPressureGridDofSourceKeys3D(
    PersistentPressureGrid3D& context, double dx,
    std::vector<GridKey>& dof_source_keys,
    PressureGridFieldStats* stats = nullptr,
    bool download_keys = true);

// Classifies the resident virtual touched-key stream against the supported
// analytic box shell and resident particle level set. Optional host flags
// preserve the legacy CPU pushout; device CSR assembly reuses the same device
// allocation without a host round-trip.
void classifyPersistentPressureGridSolid3D(
    PersistentPressureGrid3D& context,
    const GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D& level_set,
    const BoxShellCollision3D& collision, double dx,
    std::vector<unsigned char>& solid_flags,
    PressureGridFieldStats* stats = nullptr,
    bool download_flags = true);

void scatterPersistentPressureGridDensity3D(
    PersistentPressureGrid3D& context,
    const std::vector<Vec3d>& density_positions, double density_value,
    double dx, ProbeGridResult& density_result,
    PressureGridFieldStats* stats = nullptr, bool download_result = true);

// Same density scatter, but the first source range is static across
// substeps. Two optional dynamic prefixes may already be resident (for
// example, fluid X after Q1 G2P followed by solid-owner current X); only the
// remaining host-owned dynamic suffix crosses PCIe. All dynamic ranges retain
// their original contiguous order before the exact key-sort/scatter pipeline.
void scatterPersistentPressureGridDensitySplit3D(
    PersistentPressureGrid3D& context,
    const std::vector<Vec3d>& static_density_positions,
    const std::vector<Vec3d>& dynamic_density_positions,
    double density_value, double dx, ProbeGridResult& density_result,
    PressureGridFieldStats* stats = nullptr, bool download_result = true,
    const void* resident_dynamic_positions = nullptr,
    std::size_t resident_dynamic_count = 0,
    const void* second_resident_dynamic_positions = nullptr,
    std::size_t second_resident_dynamic_count = 0);

// Updates the direct pressure keyed fields once per substep for device CSR
// assembly. Resident topology switches reuse device-generated touched keys
// and rebuild the caller's legacy-compatible DOF order without a host upload.
void updatePersistentPressureGridDirectFields3D(
    PersistentPressureGrid3D& context,
    const std::vector<GridKey>& ordered_dof_keys,
    const std::vector<GridKey>& touched_keys,
    const std::vector<unsigned char>& solid_flags,
    PressureGridFieldStats* stats = nullptr,
    bool reuse_resident_virtual_touched_keys = false,
    bool rebuild_resident_legacy_dof_keys = false,
    bool reuse_resident_solid_flags = false);

// Builds the direct pressure CSR entirely from the device fields previously
// populated in PersistentPressureGrid3D. The two materialization switches let
// production keep both matrix and RHS device-only while compatibility paths
// retain the complete host result.
void assemblePersistentPressureGridCsr3D(
    PersistentPressureGrid3D& context, double rho_rest, double dx, double dt,
    PressureCsrResult& result, PressureCsrStats* stats = nullptr,
    bool materialize_host_csr = true,
    bool materialize_host_rhs = true);

#ifdef ENABLE_AMGX
// Assemble the compact CSR from the persistent fixed-width CUDA stencil and
// hand its raw device pointers to AMGX. `rhs_override` carries host-side RHS
// edits made after assembly, such as pushout and CPU island projection.
// `diagonal_epsilon` is applied only to the compact AMGX copy, matching the
// regularization used by DirectSolver::solveCsrCUDA().
// The returned solution is copied to host solely for the still-CPU pressure
// velocity update.
void solvePersistentPressureGridAmgx3D(
    PersistentPressureGrid3D& context,
    const std::vector<double>* rhs_override,
    double diagonal_epsilon,
    std::vector<double>& solution,
    PressureAmgxStats* stats = nullptr,
    bool download_solution = true);

// Applies the compact CPU pushout result to the already assembled device RHS
// and seeds the exact pre-pressure face field. Keys must be sorted. This is a
// transitional boundary until pushout projection itself moves to CUDA; it
// avoids re-uploading the full pressure RHS and preserves boundary faces that
// the pressure gradient intentionally does not overwrite.
void updatePersistentPressureGridPushout3D(
    PersistentPressureGrid3D& context,
    const std::vector<GridKey>& displacement_keys,
    const std::vector<Vec3d>& displacement,
    const std::vector<GridKey>& face_keys,
    const std::vector<Vec3d>& face_velocity,
    double rho_rest, double dx, double dt,
    const GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D& level_set,
    const BoxShellCollision3D& collision,
    PressurePushoutStats* stats = nullptr);

// Optional sparse host mirror of every particle the GPU pushout modified this
// substep (not just the per-node max-displacement reduction winners). It is
// needed by compatibility fallbacks and debug oracles; the full-pipeline
// resident owner leaves it disabled. gradV is always exactly zero for a hit
// particle (see pushoutParticleProjectionKernel), so it is never downloaded.
struct PushoutHitParticles {
    std::vector<int> particle_id;
    std::vector<Vec3d> corrected_v;
    // Debug-only (always populated alongside the above; cheap given the
    // typically tiny hit count): the node key and squared displacement each
    // hit particle contributed, for comparing against the CPU pushout's
    // per-particle debug log without a separate download path.
    std::vector<GridKey> node_key;
    std::vector<double> sq_disp;
    std::vector<int> debug_source; // Bitmask of which check(s) triggered this hit: 1=box-shell, 2=particle level set, 3=both. Useful when narrowing a future divergence to one of the two geometric predicates.
};

// Debug-only mirror of the per-node max-displacement reduction winners (one
// entry per unique node with >=1 candidate), for comparing directly against
// the CPU pushout's own per-node winner (see PositionCorrector::pushout()'s
// `ZIRAN_GPU_PUSHOUT_DEBUG` "per-node winner" log). `disp` is the reduced
// value before the `±0.5dx` clamp; `clamped_disp` is after.
struct PushoutNodeReduction {
    std::vector<GridKey> node_key;
    std::vector<int> particle_id;
    std::vector<double> sq_disp;
    std::vector<Vec3d> disp;
    std::vector<Vec3d> clamped_disp;
};

// Debug-only: every touched node with a nonzero face velocity after
// pushoutFaceVelocityGatherKernel, for comparing against the CPU pushout's
// own nonzero `direct_pressure_new_v` entries (see
// PositionCorrector::pushout()'s "CPU pushout face velocity" debug log).
struct PushoutFaceVelocityDebug {
    std::vector<GridKey> node_key;
    std::vector<Vec3d> velocity;
};

// Debug-only: the RHS correction `applyPressurePushoutRhsKernel` added to
// each pressure DOF this substep (rhs-after minus rhs-before the kernel
// launch), for comparing against the CPU pushout's own
// `pushoutPressureRhs` correction (see PositionCorrector::pushout()'s
// "CPU pushout rhs delta" debug log). Only nonzero deltas are kept.
struct PushoutRhsDebug {
    std::vector<GridKey> dof_key;
    std::vector<double> delta;
};

// Fully device-resident PositionCorrector pushout. Projects the resident Q1
// G2P fluid particle owner (`velocity_context`'s output_x/output_v/
// output_grad_v) against the box-shell collision descriptor and the
// resident particle level set, in place updates V/gradV for particles that
// hit, reduces the per-node max displacement, clamps it, and feeds the same
// RHS/face-velocity kernels `updatePersistentPressureGridPushout3D` uses
// without any host round trip for phi, solid flags, touched/DOF keys, or
// the sparse displacement/face bridge. Only valid for the box-shell/
// open-domain collision gate already used by
// classifyPersistentPressureGridSolid3D (`collision.valid ||` an empty CPU
// collision list); callers must fall back to the CPU pushout() otherwise,
// and `level_set.phi` must be a resident device pointer (not a host one).
// `hit_particles`, when non-null, is populated with exactly the particles
// `pushout_candidate_valid` flagged this substep (empty when none did, in
// which case no D2H for this purpose occurs at all). When requested, the
// caller applies it to the compatibility host particle state.
void computePersistentPressureGridPushoutGpu3D(
    PersistentPressureGrid3D& pressure_context,
    PersistentVelocityTransfer3D& velocity_context,
    const GPU_POSITION_CORRECTOR::ParticleLevelSetDeviceView3D& level_set,
    const BoxShellCollision3D& collision,
    std::size_t particle_count, double rho_rest, double dx, double dt,
    PressurePushoutGpuStats* stats = nullptr,
    PushoutHitParticles* hit_particles = nullptr,
    PushoutNodeReduction* node_reduction = nullptr,
    PushoutFaceVelocityDebug* face_velocity_debug = nullptr,
    PushoutRhsDebug* rhs_debug = nullptr);

// Uses the AMGX solution already resident in `context` to form the exact
// direct-keyed pressure face velocities and correct fluid particle positions
// on CUDA. Corrected positions remain resident unless `download_positions`
// explicitly requests a pinned-host compatibility mirror.
void applyPersistentPressureGridAmgxCorrection3D(
    PersistentPressureGrid3D& context, const void* particle_x,
    std::size_t particle_count, double rho_rest, double dx, double dt,
    PressureParticleCorrectionView& result,
    PressureParticleCorrectionStats* stats = nullptr,
    bool particle_x_on_device = false,
    bool download_positions = true);

// Debug oracle for the post-solve face field. Keys are canonical sorted
// touched keys, matching the returned Vec3d velocity entries.
void downloadPersistentPressureGridFaceVelocity3D(
    PersistentPressureGrid3D& context, std::vector<GridKey>& keys,
    std::vector<Vec3d>& velocity);

// Solve a Stage3B device CSR directly with AMGX. `diagonal_epsilon` mirrors
// DirectSolver::solveCsrCUDA()'s KKT regularization without mutating the
// caller's persistent device matrix.
void solveDeviceCsrAmgx3D(
    PersistentAmgxCsrSolver& context,
    const GPU_STAGE3A::DeviceCsrMatrix& matrix,
    const std::vector<double>& rhs,
    double diagonal_epsilon,
    std::vector<double>& solution,
    AmgxCsrStats* stats = nullptr,
    GPU_STAGE3A::DeviceVector* solution_device = nullptr);

// Device-RHS overload. AMGX ingests the existing CUDA vector directly, so a
// Stage3B/VP assembly does not route RHS through host memory. `solution_host`
// remains optional for legacy CPU postsolve consumers.
void solveDeviceCsrAmgx3D(
    PersistentAmgxCsrSolver& context,
    const GPU_STAGE3A::DeviceCsrMatrix& matrix,
    const GPU_STAGE3A::DeviceVector& rhs_device,
    double diagonal_epsilon,
    std::vector<double>* solution_host,
    AmgxCsrStats* stats = nullptr,
    GPU_STAGE3A::DeviceVector* solution_device = nullptr);
#endif

void assemblePersistentPressureCsr3D(
    PersistentPressureCsr3D& context,
    const std::vector<GridKey>& ordered_dof_keys,
    const std::vector<GridKey>& solid_keys,
    const std::vector<unsigned char>& solid_flags,
    const std::vector<double>& rho_star_by_row,
    double rho_rest, double dx, double dt,
    PressureCsrResult& result,
    PressureCsrStats* stats = nullptr);

void runVelocityGridMassProbe(const std::vector<Vec3d>& xq,
    const std::vector<double>& mass, double dx, bool halfDxAnchor,
    ProbeGridResult& result);

void runPressureGridCountProbe(const std::vector<Vec3d>& xq,
    double dx, bool halfDxAnchor, ProbeGridResult& result);

void runFaceGridCountProbe(const std::vector<Vec3d>& face_xq,
    const std::vector<Vec3d>& face_n, double dx, bool halfDxAnchor,
    ProbeGridResult& result);

ProbeCompareStats compareMassByKey(const std::vector<GridKey>& cpu_keys,
    const std::vector<double>& cpu_mass, const ProbeGridResult& gpu_result);

// Builds the one and only particle-neighbour search grid. Equal-key particle
// ids retain input order so reductions are deterministic for a fixed input.
void rebuildPersistentSpatialSearch(PersistentSpatialSearch3D& context,
    const std::vector<Vec3d>& positions, double cell_size,
    SpatialSearchStats* stats = nullptr,
    const void* resident_positions = nullptr,
    std::size_t resident_position_count = 0);

// Conservative geometric query. A false result proves that no query point is
// within radius of any indexed point; collision predicates may further reject
// true candidates in the caller.
bool anyPointWithinRadius(PersistentSpatialSearch3D& context,
    const std::vector<Vec3d>& query_positions, double radius,
    SpatialSearchStats* stats = nullptr,
    const void* resident_query_positions = nullptr,
    std::size_t resident_query_count = 0);

// Reproduces the total-Lagrangian cubic B-spline SPH normal calculation using
// the already-built search topology. masses are indexed by query particle id.
void computeCubicSphNormals(PersistentSpatialSearch3D& context,
    const std::vector<double>& masses, double radius,
    std::vector<Vec3d>& normals, SpatialSearchStats* stats = nullptr,
    const void* resident_masses = nullptr,
    std::size_t resident_mass_count = 0,
    bool materialize_host_normals = true);

// Output-boundary copy of the solid positions and SPH normals retained by
// the persistent search. Production coupling steps keep both fields on the
// device and call this only when the legacy visualization files are written.
void copyPersistentSpatialSearchFieldsToHost3D(
    PersistentSpatialSearch3D& context, std::vector<Vec3d>& positions,
    std::vector<Vec3d>& normals, bool copy_normals = true);

// Example 2 interface-quadrature neighbour distribution. The solid search must
// already have been rebuilt for the current substep using the same solid
// position field as the CPU path. Accumulation uses double atomic adds; the
// caller retains its CPU fallback for scenes without the box-shell device
// collision predicate.
void rebuildInterfaceQuadAtomic3D(PersistentInterfaceQuad3D& context,
    PersistentSpatialSearch3D& solid_search,
    const std::vector<Vec3d>& fluid_positions,
    const std::vector<double>& fluid_mass,
    const std::vector<Vec3d>& fluid_velocity,
    double dx, const BoxShellCollision3D& collision,
    InterfaceQuadAtomicResult& result,
    InterfaceQuadAtomicStats* stats = nullptr,
    const void* resident_fluid_positions = nullptr,
    std::size_t resident_fluid_count = 0,
    const void* resident_fluid_velocity = nullptr,
    std::size_t resident_fluid_velocity_count = 0,
    bool compact_host_result = false,
    bool materialize_host_result = true);

// Gathers rest/current position, SPH normal and interface length for the
// active solid ids already compacted by rebuildInterfaceQuadAtomic3D. All
// source fields and the returned view stay on the device.
void buildPersistentInterfaceQuadratureFields3D(
    PersistentInterfaceQuad3D& context,
    PersistentSpatialSearch3D& solid_search,
    const GPU_STAGE3A::SolidParticleStateDeviceView& solid_particles,
    InterfaceQuadratureDeviceView3D& view);

// Diagnostic-only copy-out for checking the device-resident interface fields
// against the legacy host materialization. Production callers consume the
// device view above and never invoke this function.
void copyInterfaceQuadratureFieldsToHost3D(
    const InterfaceQuadratureDeviceView3D& view,
    std::vector<Vec3d>& rest_x, std::vector<Vec3d>& current_x,
    std::vector<Vec3d>& normal, std::vector<double>& length);

// The fluid analytical stage clamps its private interface coordinates before
// legacy W assembly. Rebind current_x to that device buffer so a resident W
// observes exactly the same post-clamp geometry without a host round trip.
void bindInterfaceQuadratureClampedCurrent3D(
    PersistentFluidAnalytic3D& fluid_context,
    InterfaceQuadratureDeviceView3D& view);

// Device-resident bridge for the example 2 interface path.  The interface
// distribution has already uploaded the fluid samples and accumulated solid
// samples; compact those samples on the device and feed them directly into
// the analytic quadrature pipeline.  This deliberately retains the existing
// host result as a compatibility view for the SPGrid/legacy stages, but
// avoids the intermediate host quadrature reconstruction and a second H2D
// upload before analytic integration.
void rebuildFluidAnalyticQuadratureFromInterfaceQuad3D(
    PersistentFluidAnalytic3D& fluid_context,
    PersistentInterfaceQuad3D& interface_context,
    PersistentSpatialSearch3D& solid_search, double dx, double rho,
    const BoxShellCollision3D& collision, FluidAnalyticResult& result,
    FluidAnalyticStats* stats = nullptr,
    const void* resident_fluid_grad_velocity = nullptr,
    std::size_t resident_fluid_grad_velocity_count = 0,
    double apic_rpic_ratio = 1.0,
    const std::vector<Mat3d>& host_fluid_affine = {},
    bool materialize_host_result = true);

// Example 2 Q1/Q0 analytical-integration pipeline: collision pruning, active-cell
// classification, face quadrature emission, interior mass reconstruction and
// clamping. The output remains on reusable device buffers until this API's
// compatibility copy-out; the production assembly path consumes the same
// topology in the next integration step.
void rebuildFluidAnalyticQuadrature(PersistentFluidAnalytic3D& context,
    const std::vector<Vec3d>& volume_xq,
    const std::vector<double>& volume_mass,
    const std::vector<Vec3d>& interface_xq, double dx, double rho,
    const BoxShellCollision3D& collision, FluidAnalyticResult& result,
    FluidAnalyticStats* stats = nullptr,
    const void* resident_fluid_velocity = nullptr,
    const void* resident_fluid_grad_velocity = nullptr,
    std::size_t resident_fluid_count = 0,
    double apic_rpic_ratio = 1.0,
    bool materialize_host_result = true);

FluidAnalyticDeviceView3D getFluidAnalyticDeviceView3D(
    const PersistentFluidAnalytic3D& context);

void copyFluidAnalyticFaceFieldsToHost3D(
    const PersistentFluidAnalytic3D& context,
    std::vector<Vec3d>& face_xq, std::vector<Vec3d>& face_normal);

// Reproduces MpmGrid::activatePages page-first traversal for the analytical Q0
// pressure and boundary-face grids without a host unordered_map or node AoS.
// The returned maps stay sorted by GridKey while idx retains the exact legacy
// SPGrid DOF permutation used by fixed-iteration AMGX runs.
void buildFluidLegacyDofNodeMaps3D(PersistentFluidAnalytic3D& context,
    double dx, int block_x, int block_y, int block_z,
    GPU_STAGE3A::DeviceGridNodeMap& pressure_nodes,
    GPU_STAGE3A::DeviceGridNodeMap& face_nodes,
    FluidLegacyDofOrderStats* stats = nullptr);

void rebuildFluidVelocityGrid(PersistentFluidAnalytic3D& context,
    const std::vector<Vec3d>& input_xq,
    const std::vector<double>& input_mass,
    const std::vector<Vec3d>& input_velocity,
    const std::vector<Mat3d>& input_affine,
    const std::vector<Vec3d>& final_xq,
    const std::vector<double>& final_mass, double dx, double dt,
    const Vec3d& gravity, std::vector<VelocityGridNode>& nodes,
    std::vector<VelocityGridNode>& g2p_fallback_nodes,
    FluidVelocityGridStats* stats = nullptr,
    bool resident_analytic_position_mass = false,
    bool resident_analytic_velocity = false,
    bool resident_analytic_affine = false,
    bool materialize_host_nodes = true);

// Builds the final-mass Q1 velocity map directly from the resident velocity
// grid. idx is the exact ascending SPGrid Linear_Offset rank used by the
// legacy host compatibility path.
void buildFluidLegacyVelocityNodeMap3D(PersistentFluidAnalytic3D& context,
    GPU_STAGE3A::DeviceGridNodeMap& velocity_nodes,
    double* order_ms = nullptr);

void rebuildSolidPressureGrid(PersistentFluidAnalytic3D& context,
    const std::vector<Vec3d>& positions, const std::vector<double>& j,
    const std::vector<double>& volume,
    const std::vector<double>& lambda, double dx,
    std::vector<SolidPressureGridNode>& nodes,
    FluidVelocityGridStats* stats = nullptr,
    const GPU_STAGE3A::SolidParticleStateDeviceView* resident_particles = nullptr,
    GPU_STAGE3A::DeviceGridNodeMap* device_nodes = nullptr,
    bool materialize_host_nodes = true);

} // namespace ZIRAN::GPU_STAGE2
