#ifndef MPM_FORCE_ASSEMBLY_H
#define MPM_FORCE_ASSEMBLY_H

#include "MpmForceTerm.h"
#include "FJMixedForce.h"
#include "MpmMultiTargetPinningForce.h"
#include "MpmPinningForce.h"
#include "MpmLagrangianContactForce.h"

#include <MPM/Grid/MpmGrid.h>
#include <Ziran/Math/Geometry/Particles.h>
#include <Ziran/Physics/LagrangianForce/LagrangianForce.h>
#include <Ziran/CS/DataStructure/SpatialHash.h>
#include <Ziran/Sim/Scene.h>

namespace ZIRAN {

// Combines material, pinning, and contact terms for MPM force/tangent evaluation.
template <class T, int dim>
class MpmForceAssembly : public LagrangianForce<T, dim> {
public:
    static constexpr int interpolation_degree = ZIRAN_MPM_DEGREE;

    typedef Vector<T, 4> TV4;
    typedef Matrix<T, 4, 4> TM4;
    typedef Vector<T, dim> TV;
    typedef Vector<int, dim> IV;
    typedef Matrix<T, dim, dim> TM;

    using Base = LagrangianForce<T, dim>;
    using Base::addScaledForceDifferential;
    using Base::addScaledForces;
    using Base::reinitialize;
    using typename Base::TVStack;

    const bool& mls_mpm;
    T D_inverse;

    const T& dx;
    const T& dt;
    const TVStack& dv;
    const TVStack& vn;

    // using Base::addScaledForces;
    // using Base::addScaledForceDifferential;
    // using Base::gatherFromPadsToTVStack;

    // particle data
    Particles<T, dim>& particles;
    Scene<T, dim>& scene;

    const CONFIG_SCHEME& config_scheme;
    bool has_lagrangian_contact;
    SpatialHash<T, dim> scratch_hash;
    // Per-particle scratch shared with the owning simulation (references into it).
    // Refilled every implicit Newton iteration by updateForcesAtState():
    StdVector<TV>& scratch_xp;    // trial particle positions  xp = xn + dt*vp
    TVStack& scratch_vp;          // particle velocity interpolated from the grid
    TVStack& scratch_fp;          // per-particle non-stress force (pinning/contact),
                                  // later rasterized to grid nodes

    StdVector<TM>& scratch_gradV; // grad of the interpolated velocity, per particle
    StdVector<TM>& scratch_force_stress; // per-particle stress  V_p^0 * P * F^T, rasterized to nodal internal force

    // De-virtualized force set (was an anonymous `helpers` vector). Each MPM
    // force this simulation applies is now a named member, so the force-assembly
    // control flow is visible at the call site instead of a virtual loop.
    //   - material_force: the constitutive (FJMixed) force. Now a concrete,
    //     non-templated FJMixedForce (step B) -> its calls are non-virtual too;
    //     it dispatches internally to the present constitutive model(s).
    //   - the auxiliary forces below are concrete (non-templated) -> direct calls.
    std::unique_ptr<FJMixedForce<T, dim>> material_force;
    std::unique_ptr<MpmMultiTargetPinningForce<T, dim>> multi_target_pinning;
    std::unique_ptr<MpmPinningForce<T, dim>> pinning;
    std::unique_ptr<MpmLagrangianContactForce<T, dim>> lagrangian_contact;

    // Call `op` on every force that is present, in a fixed, visible order
    // (material first, then the auxiliary forces). Replaces `for (h : helpers)`.
    template <class Op>
    void forEachForceTerm(Op&& op) {
        if (material_force)   op(*material_force);
        if (multi_target_pinning) op(*multi_target_pinning);
        if (pinning)          op(*pinning);
        if (lagrangian_contact) op(*lagrangian_contact);
    }
    template <class Op>
    void forEachForceTerm(Op&& op) const {
        if (material_force)   op(*material_force);
        if (multi_target_pinning) op(*multi_target_pinning);
        if (pinning)          op(*pinning);
        if (lagrangian_contact) op(*lagrangian_contact);
    }

    MpmGrid<T, dim>& grid;
    StdVector<uint64_t>& particle_base_offset;
    StdVector<int>& particle_order;
    std::vector<std::pair<int, int>>& particle_group;
    std::vector<uint64_t>& block_offset;
    int& num_nodes;

    MpmForceAssembly(const bool& mls_mpm, const T& dx, const T& dt, Particles<T, dim>& particles,
        Scene<T, dim>& scene,
        CONFIG_SCHEME& config_scheme,
        StdVector<TV>& scratch_xp,
        TVStack& scratch_vp,
        TVStack& scratch_fp,
        StdVector<TM>& scratch_gradV,
        StdVector<TM>& scratch_force_stress,
        const TVStack& dv, const TVStack& vn,
        MpmGrid<T, dim>& grid, StdVector<uint64_t>& particle_base_offset, StdVector<int>& particle_order, std::vector<std::pair<int, int>>& particle_group, std::vector<uint64_t>& block_offset, int& num_nodes);

    virtual ~MpmForceAssembly();

    void reinitialize() override;

    void backupStrain();

    void restoreStrain();

    void computeVAndGradV();

    void computeDvAndGradDv(const TVStack& dv) const;

    template <bool USE_MLS_MPM>
    void rasterizeForceToTVStack(const T scale, TVStack& force) const;

    void updateForcesAtState();

    void addScaledForces(const T scale, TVStack& forces) const override;

    /**
      Compute for a vector function f defined on active grid nodes, compute f and grad f on particles
      */

    template <class Func>
    void evalInterpolantAndGradient(Func&& f, TVStack& f_eval, StdVector<TM>& grad_f) const;

    void addScaledForceDifferential(const T scale, const TVStack& dv, TVStack& df) const override;

    // Recompute strain/positions at the current Newton velocity guess (implicit).
    void updateStrainAndPositions();

    int countNearlyIncompressibleParticles(T eps);

    void updateConfig(T eps, T eta);

    void evolveStrain(T dt);

    T totalEnergy() const override;

    // Create the mixed F/J material term on first use.
    FJMixedForce<T, dim>& materialForce()
    {
        if (!material_force)
            material_force = std::make_unique<FJMixedForce<T, dim>>(particles);
        return *material_force;
    }

};
} // namespace ZIRAN
#endif
