#ifndef MPM_PARTICLE_RANGE_H
#define MPM_PARTICLE_RANGE_H

#include <functional>
#include <MPM/Force/MpmForceAssembly.h>
#include <Ziran/CS/DataStructure/DisjointRanges.h>
#include <Ziran/CS/DataStructure/Ref.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/Physics/ConstitutiveModel/ConstitutiveModel.h>

namespace ZIRAN {
template <class T, int dim> class AnalyticLevelSet;
template <class T, int dim> class Particles;

// A view over scene particles; assigns materials, rest positions, and constraints.
template <class T, int dim> class MpmParticleRange {
public:
    using TV = Vector<T, dim>;
    using TM = Matrix<T, dim, dim>;

    Particles<T, dim>& particles;
    MpmForceAssembly<T, dim>* mpmforce;
    StdVector<TV>& scratch_xp;
    T& dt;
    Range particle_range;
    T total_volume;
    bool already_pinned_particles = false;
    bool already_add_tlcontacts_particles = false;
    bool has_multi_target_pinning = false;

    MpmParticleRange(Particles<T, dim>& particles, MpmForceAssembly<T, dim>* mpmforce,
        StdVector<TV>& scratch_xp, T& dt, Range particle_range, T total_volume);

    void transform(const std::function<void(int, Ref<T>, TV&, TV&)>& mapping);
    void captureRestPositions();

    template <class FCONST, class JCONST>
    void addFJMaterial(const FCONST& f_model, const JCONST& j_model,
        MATERIAL_PHASE_ENUM phase, CONFIG_SCHEME config_scheme,
        MATERIAL_PROPERTY_ENUM property, bool linear_corotated = false);

    void addTotalLagrangianContacts(CONFIG_SCHEME config_scheme);
    void pinParticlesInLevelSet(AnalyticLevelSet<T, dim>& levelset, T stiffness, T damping);
    void addMultiTargetPinningInLevelSet(AnalyticLevelSet<T, dim>& levelset,
        T stiffness, T damping, bool is_collider_spring = false);
    void pinParticles(const DisjointRanges& ranges, T stiffness, T damping);
    void addMultiTargetPinning(const DisjointRanges& ranges, T stiffness, T damping,
        bool is_collider_spring = false);
};
} // namespace ZIRAN
#endif
