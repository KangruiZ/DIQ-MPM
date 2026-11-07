#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace ZIRAN::GPU_POSITION_CORRECTOR {

struct Vec3d {
    double x = 0;
    double y = 0;
    double z = 0;
};

// Device storage is retained across coupling substeps. The public wrapper is
// deliberately CUDA-free so the CPU reference build keeps compiling.
struct PersistentParticleLevelSet3D {
    std::shared_ptr<void> storage;
};

struct DenseLevelSet3D {
    int origin[3] = { 0, 0, 0 };
    int shape[3] = { 0, 0, 0 };
    // x is the fastest varying coordinate. Untouched voxels use a positive
    // maximum-value sentinel.
    std::vector<double> phi;
};

struct ParticleLevelSetDeviceView3D {
    const void* phi = nullptr;
    int origin[3] = { 0, 0, 0 };
    int shape[3] = { 0, 0, 0 };
    double h = 0;
    std::size_t voxels = 0;
};

struct ParticleLevelSetStats {
    std::size_t particles = 0;
    std::size_t voxels = 0;
    std::size_t capacity_voxels = 0;
    std::size_t reallocations = 0;
    int kernel_size = 0;
    int sweep_cycles = 0;
    bool resident_particle_input = false;
    double upload_ms = 0;
    double rasterize_ms = 0;
    double sweep_ms = 0;
    double download_ms = 0;
    double total_ms = 0;
};

bool particleLevelSetCudaAvailable();

// Reproduces ParticlesLevelSet's order-1 particle blob rasterization and its
// one-sided Eikonal solve. raw_phi <= 0 voxels remain frozen; raw_phi > 0
// voxels are redistanced with the same six-neighbour local equation used by
// the CPU priority-queue implementation. The final value is -distance+erosion.
bool buildParticleLevelSet3D(PersistentParticleLevelSet3D& context,
    const std::vector<Vec3d>& positions, double h, double blob_radius,
    double erosion, int sweep_cycles, DenseLevelSet3D& result,
    ParticleLevelSetStats* stats = nullptr,
    const void* resident_positions = nullptr,
    std::size_t resident_position_count = 0,
    bool download_result = true);

// Materializes the current device level set only when a CPU compatibility
// query actually needs it. Production PositionCorrector paths consume the
// device view directly and leave this dense array resident.
bool copyParticleLevelSetToHost3D(
    const PersistentParticleLevelSet3D& context, DenseLevelSet3D& result);

ParticleLevelSetDeviceView3D getParticleLevelSetDeviceView3D(
    const PersistentParticleLevelSet3D& context);

} // namespace ZIRAN::GPU_POSITION_CORRECTOR
