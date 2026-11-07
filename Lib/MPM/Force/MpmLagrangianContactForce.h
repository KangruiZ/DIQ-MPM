#ifndef MPM_LAGRANGIAN_CONTACT_FORCE_H
#define MPM_LAGRANGIAN_CONTACT_FORCE_H

#include <Ziran/Physics/ConstitutiveModel/PinningTarget.h>
#include <Ziran/CS/DataStructure/DataArray.h>
#include <Ziran/CS/DataStructure/DataManager.h>
#include <Ziran/CS/DataStructure/DisjointRanges.h>
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/DataStructure/SpatialHash.h>
#include <Ziran/Math/Geometry/Particles.h>
#include <MPM/Force/MpmForceTerm.h>
namespace ZIRAN {

    
// Total-Lagrangian particle contact with reference-space neighbor queries.
template <class T, int dim>
class MpmLagrangianContactForce : public MpmForceTerm<T, dim> {
public:
    typedef Matrix<T, dim, dim> TM;
    typedef Vector<T, dim> TV;
    typedef Matrix<T, dim, Eigen::Dynamic> TVStack;

    // typedef SpatialHashScratch<T, dim> Scratch;

    Particles<T, dim>& particles; // contains Xn and Vn
    SpatialHash<T, dim>& scratch_hash;
    const StdVector<TV>& X0Array;
    const StdVector<TV>& xp; // this is particle position during implicit solve
    const T& dt;
    const T& dx;
    T radius;
    // DataArray<Scratch>& scratch; // contains A and b. no write.
    typedef TLNeighbour<T, dim> Neighbour;
    typedef PinningTarget<T, dim> Target;
    typedef PinningTargetScratch<T, dim> Scratch;
    // DataArray<Target>& target; // contains k, d, xt, vt. readwrite state.
    DataArray<Neighbour>& scratch_neighbour; // contains k, d, xt, vt. readwrite state.

    MpmLagrangianContactForce(Particles<T, dim>& particles, SpatialHash<T, dim>& scratch_hash1, const StdVector<TV>& X0Array, const StdVector<TV>& xp, const T& dt, const T& dx)
        : particles(particles)
        , scratch_hash(scratch_hash1)
        , X0Array(X0Array)
        , xp(xp) // this is particle position during implicit solve. use scratch_xp
        , dt(dt)
        , dx(dx)
        , scratch_neighbour(particles.add(neighbour_name()))
    {
        radius = 0.5 * dx;
        // X0Array = particles.DataManager::get(X0_name<TV>()).array;
        // auto * X0 = particles.DataManager::getPointer(X0_name<TV>());
    }

    virtual ~MpmLagrangianContactForce() {}

    void reinitialize() 
    {
        // StdVector<TV> X0Array = particles.DataManager::get(X0_name<TV>()).array;
        // scratch_hash.rebuild(radius, X0Array);
        printf("scratch_hash is run in reinitialize !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }

    // add to fp for explicit
    void updateState(const DisjointRanges& subrange, StdVector<TM>& force_stress, TVStack& fp) 
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(
                neighbour_name()));
        for (auto iter = particles.subsetIter(subset,
            neighbour_name()); iter; ++iter) {
            // config_scheme_name()); iter; ++iter) {
            // target_scratch_name()); iter; ++iter) {
            // target_name()); iter; ++iter) {
            // auto& config_scheme = iter.template get<0>();
            auto& s = iter.template get<0>(); // neighbour
            
            int p = iter.entryId();
            const TV& Xp = X0Array[p];
            // if( p == 100)
                // std::cout << Xp << std::endl;
            StdVector<int> unfiltered_neighbors;
            StdVector<int> neighbors;
            scratch_hash.oneLayerNeighbors(Xp, unfiltered_neighbors);
            for (auto &idx : unfiltered_neighbors)
            // TODO: partiicle in solid
                if ((Xp - X0Array[idx]).norm() < 2 * radius && (particles.X[p] - particles.X[idx]).norm() > 1.5 * 2 * radius)
                    neighbors.push_back(idx);

            for (auto idx : neighbors) {
                TV dir = (Xp - X0Array[idx]);
                TV temp_contact_f = - 1.0 / (dt) / dt * (particles.mass[p] * particles.mass[idx]) / (particles.mass[p] + particles.mass[idx]) * (1 - (radius + radius) / dir.norm()) * dir;
                fp.col(p) += temp_contact_f;
            }
        }
    }

    // add to fp for implicit
    void updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& force_stress, TVStack& fp)
    {
        // printf("updata updateImplicitState\n");
        DisjointRanges subset(subrange,
            particles.commonRanges(
                neighbour_name()));
        for (auto iter = particles.subsetIter(subset,
            neighbour_name()); iter; ++iter) {
            auto& s = iter.template get<0>(); // neighbour
            s.neighbors.clear();

            int p = iter.entryId();
            const TV& x = xp[p]; // from scratch_xp
            StdVector<int> unfiltered_neighbors;
            scratch_hash.oneLayerNeighbors(x, unfiltered_neighbors);
            for (auto &idx : unfiltered_neighbors)
            {
            // TODO: particle in solid
                const TV& xq = xp[idx]; // from scratch_xp
                if ((x - xq).norm() < 2 * radius && (particles.X[p] - particles.X[idx]).norm() > 1.5 * 2 * radius)
                {
                    s.neighbors.push_back(idx);
                }
            }

            for (auto idx : s.neighbors) {
                const TV& xq = xp[idx]; // from scratch_xp
                TV dir = (x - xq);
                TV temp_contact_f = - 1.0 / (dt) / dt * (particles.mass[p] * particles.mass[idx]) / (particles.mass[p] + particles.mass[idx]) * (1 - (radius + radius) / dir.norm()) * dir;
                fp.col(p) += temp_contact_f;
            }
        }
    }

    double totalEnergy(const DisjointRanges& subrange) 
    {
        // printf("updata totalEnergy\n");
        double e = 0.0;
        DisjointRanges subset(subrange,
            particles.commonRanges(
                neighbour_name()));
        for (auto iter = particles.subsetIter(subset,
            neighbour_name()); iter; ++iter) {
            auto& s = iter.template get<0>(); // neighbour

            int p = iter.entryId();
            const TV& x = xp[p]; // from scratch_xp

            for (auto idx : s.neighbors) {
                const TV& xq = xp[idx]; // from scratch_xp
                TV dir = (x - xq);
                T energy = - 0.5 * 1.0 / (dt) / (dt) * (particles.mass[p] * particles.mass[idx]) / (particles.mass[p] + particles.mass[idx]) * dir.squaredNorm();
                e += 1.0 / 2.0 * energy; // 因为对称的粒子有个等大的能量
            }
        }
        return e;
    }

    void computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp) 
    {
        DisjointRanges subset(subrange,
            particles.commonRanges(
                neighbour_name()));
        for (auto iter = particles.subsetIter(subset,
            neighbour_name()); iter; ++iter) {
            auto& s = iter.template get<0>(); // neighbour

            int p = iter.entryId();
            const TV& x = xp[p]; // from scratch_xp

            for (auto idx : s.neighbors) {
                const TV& xq = xp[idx]; // from scratch_xp
                TV dir = (x - xq);
                T k = 1.0 / (dt) / (dt) * (particles.mass[p] * particles.mass[idx]) / (particles.mass[p] + particles.mass[idx]);
                dfp.col(p) += (-k * dvp.col(p));
            }
        }
    }
    inline static AttributeName<Target> target_name()
    {
        return AttributeName<Target>(Target::name());
    }
    inline static AttributeName<Neighbour> neighbour_name()
    {
        return AttributeName<Neighbour>(Neighbour::name());
    }
    inline static AttributeName<Scratch> target_scratch_name()
    {
        return AttributeName<Scratch>(Scratch::name());
    }
    // inline static AttributeName<TV> X0_name()
    // {
    //     return AttributeName<TV>("X0");
    //     // return AttributeName<TV>("X0");
    // }
};
} // namespace ZIRAN
#endif
