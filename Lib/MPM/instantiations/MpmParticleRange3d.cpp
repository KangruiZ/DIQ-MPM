#include <MPM/Initialization/MpmParticleRange.tpp>
namespace ZIRAN {
template class MpmParticleRange<double, 3>;
extern template class MpmForceAssembly<double, 3>;
// addFJMaterial<F, QuadraticVolumePenalty> for every F model in the list:
#define FJMIXED_F_MODEL(M)                                                     \
  template void MpmParticleRange<double, 3>::addFJMaterial<             \
      M<double, 3>, QuadraticVolumePenalty<double, 3>>(                              \
      M<double, 3> const &, QuadraticVolumePenalty<double, 3> const &,               \
      MATERIAL_PHASE_ENUM, CONFIG_SCHEME, MATERIAL_PROPERTY_ENUM, bool);
#include <MPM/Force/FJMixedFModels.def>
#undef FJMIXED_F_MODEL
} // namespace ZIRAN
