#ifndef LAGRANGIAN_FORCE_H
#define LAGRANGIAN_FORCE_H

#include <Ziran/CS/Util/Forward.h>
#include <Ziran/Math/Geometry/ElementManager.h>
#include <Eigen/Sparse>

namespace ZIRAN {

/**
This is the base class for computing forces on a collection of particles. The forces are usually functions of the lagrangian mesh node positions and velocities. Derived examples are hyperelastic FEM,
mass/spring etc.
**/

template <class T, int dim>
class LagrangianForce {
protected:
public:
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using Vec = Vector<T, Eigen::Dynamic>;
    using VecBlock = Eigen::VectorBlock<Vec>;
    using TV = Vector<T, dim>;

    LagrangianForce() {}
    virtual ~LagrangianForce() {}

    /**
       TODO: A nice comment.
     **/
    virtual void reinitialize() {}

    /**
       For those have element manager, override with the proper function
     */
    virtual bool isThisMyElementManager(ElementManager<T, dim>& em)
    {
        return false;
    }

    virtual void splatToPad(const T scale, const int pad_id) const {}

    virtual void splatDifferentialToPad(const T scale, const int pad_id, const TVStack& pad_dx) const {}

    /**
Often, force, energy and or force derivatives need to be computed without changing the particle state. When this is the case, it is efficient to precompute quantities that are needed for each of these
routines. This function call precomputes position based quantities used for each of the energy, force and force derivative routines at a given particle position state.
**/
    virtual void updatePositionBasedState() {}

    virtual void updatePositionBasedState(const StdVector<TV>& x) {}

    virtual void updatePositionDifferentialBasedState(const TVStack& dx) {}

    /**
Matrix-free force derivative for the time-stepping solve.
df += scale*A*dx
**/
    virtual void addScaledForceDifferential(const T scale, const TVStack& dx, TVStack& df) const {}

    /**
This adds a scaled version of the force into the forces array. The Lagrangian force is typically only one of a few terms in the expression for the RHS used when
time stepping (e.g. backward Euler or quasistatics) the state of the particles. E.g. this can be used when setting up the linear system for both backward Euler and quasistatics.
**/
    virtual void addScaledForces(const T scale, TVStack& forces) const {}

    /**
This is mainly used for debugging tests, but it can be used for line search. It is not necessarily even defined for all Lagrangian forces.
**/
    virtual T totalEnergy() const { return (T)0; }
};
} // namespace ZIRAN
#endif
