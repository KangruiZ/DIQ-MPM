#include "GpuParticleLevelSet.h"

namespace ZIRAN::GPU_POSITION_CORRECTOR {

bool particleLevelSetCudaAvailable()
{
    return false;
}

bool buildParticleLevelSet3D(PersistentParticleLevelSet3D&,
    const std::vector<Vec3d>&, double, double, double, int,
    DenseLevelSet3D& result, ParticleLevelSetStats* stats,
    const void*, std::size_t, bool)
{
    result = DenseLevelSet3D {};
    if (stats)
        *stats = ParticleLevelSetStats {};
    return false;
}

bool copyParticleLevelSetToHost3D(
    const PersistentParticleLevelSet3D&, DenseLevelSet3D& result)
{
    result = DenseLevelSet3D {};
    return false;
}

ParticleLevelSetDeviceView3D getParticleLevelSetDeviceView3D(
    const PersistentParticleLevelSet3D&)
{
    return {};
}

} // namespace ZIRAN::GPU_POSITION_CORRECTOR
