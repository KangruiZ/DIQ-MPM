#include "MpmMultiTargetPinningForce.h"
#include <Ziran/Math/Geometry/Particles.h>
#include <atomic>
#include <cstdlib>

namespace ZIRAN {

template <class T, int dim>
void MpmMultiTargetPinningForce<T, dim>::
    reinitialize()
{
    // make sure scratch has the same size with target
    scratch.lazyResize(target.ranges);
}

// add to fp for explicit
template <class T, int dim>
void MpmMultiTargetPinningForce<T, dim>::
    updateState(const DisjointRanges& subrange, StdVector<TM>& force_stress, TVStack& fp)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(target_name(),
            target_scratch_name()
            ));
    auto * X0 = particles.DataManager::getPointer(X0_name<TV>());
    auto * config_scheme = particles.DataManager::getPointer(config_scheme_name());
    for (auto iter = particles.subsetIter(subset, target_name(), target_scratch_name()); iter; ++iter) {
        auto& t = iter.template get<0>();
        int p = iter.entryId();
        // const auto& xn = particles.X[p];
        auto& xn = ((*config_scheme)[p] != CONFIG_UPDATE_LAGRANGIAN) ? X0Array[p] : particles.X[p];
        const auto& vn = particles.V[p];

        for (int i = 0; i < (int)t.x.size(); ++i) {
            if (collider_spring) {
                TV n = (t.x[i] - xn);
                if (n.squaredNorm() < (T)1e-12) continue;
                n.normalize();
                TM M = n * n.transpose();
                fp.col(p) += -t.k * M * (xn - t.x[i]);
            }
            else {
                // f = -k(x - xt) - d(v - vt);
                fp.col(p) += (-t.k * (xn - t.x[i]) - t.d * (vn - t.v[i]));
            }
        }
    }
}

// add to fp for implicit
template <class T, int dim>
void MpmMultiTargetPinningForce<T, dim>::
    updateImplicitState(const DisjointRanges& subrange, StdVector<TM>& force_stress, TVStack& fp)
{
    const bool ghost_contact_probe =
        collider_spring && std::getenv("ZIRAN_SOLID_GHOST_CONTACT_PROBE") != nullptr;
    int active_contact_count = 0;
    double active_force_norm_sq = 0;
    DisjointRanges subset(subrange,
        particles.commonRanges(target_name(),
            target_scratch_name()));
    auto * config_scheme = particles.DataManager::getPointer(config_scheme_name());
    for (auto iter = particles.subsetIter(subset, target_name(), target_scratch_name()); iter; ++iter) {
        auto& t = iter.template get<0>(); // target
        auto& s = iter.template get<1>(); // scratch
        int p = iter.entryId();
        auto& xn = ((*config_scheme)[p] != CONFIG_UPDATE_LAGRANGIAN) ? X0Array[p] : particles.X[p];
        // const TV& xn = particles.X[p];
        const TV& x = xp[p]; // from scratch_xp
        for (int i = 0; i < (int)t.x.size(); ++i) {
            const TV& xtn = t.x[i];
            if (collider_spring) {
                TV n = (xtn - xn);
                if (n.squaredNorm() < (T)1e-12) continue;
                n.normalize();
                TM M = n * n.transpose();
                if ((x - xtn).dot(n) < 0) {
                    const TV force = -t.k * M * (x - xtn);
                    fp.col(p) += force;
                    if (ghost_contact_probe) {
                        ++active_contact_count;
                        active_force_norm_sq += static_cast<double>(force.squaredNorm());
                    }
                }
            }
            else {
                const TV& vtn = t.v[i];
                const TV& xt = xtn + vtn * dt; // candidate target n+1 position
                s.A = t.k + t.d / dt;
                s.b = t.k * xt - (t.d / dt) * (-xn - xt + xtn);
                fp.col(p) += (-s.A * x + s.b);
            }
        }
    }
    // Each Newton state is evaluated over several TBB subranges.  One
    // nonzero report is sufficient to establish that a GHOST target generated
    // an active unilateral penalty force, so suppress the remaining ranges.
    if (ghost_contact_probe && active_contact_count > 0) {
        static std::atomic<bool> reported{false};
        bool expected = false;
        if (reported.compare_exchange_strong(expected, true))
            ZIRAN_INFO("solid GHOST penalty force probe: active_contacts=",
                       active_contact_count, " local_force_l2=",
                       std::sqrt(active_force_norm_sq));
    }
}

template <class T, int dim>
double MpmMultiTargetPinningForce<T, dim>::
    totalEnergy(const DisjointRanges& subrange)
{
    double e = 0.0;
    DisjointRanges subset(subrange,
        particles.commonRanges(target_name(),
            target_scratch_name()));
    auto * config_scheme = particles.DataManager::getPointer(config_scheme_name());
    for (auto iter = particles.subsetIter(subset, target_name(), target_scratch_name()); iter; ++iter) {
        auto& t = iter.template get<0>(); // target
        auto& s = iter.template get<1>(); // scratch
        int p = iter.entryId();
        auto& x = xp[p];
        auto& xn = ((*config_scheme)[p] != CONFIG_UPDATE_LAGRANGIAN) ? X0Array[p] : particles.X[p];
        // auto& xn = particles.X[p];
        for (int i = 0; i < (int)t.x.size(); ++i) {
            if (collider_spring) {
                // const TV& xn = particles.X[p];
                const TV& xtn = t.x[i];
                TV n = (xtn - xn);
                if (n.squaredNorm() < (T)1e-12) continue;
                n.normalize();
                TM M = n * n.transpose();
                if ((x - xtn).dot(n) < 0)
                    e += t.k / 2 * (x - xtn).transpose() * M * (x - xtn);
            }
            else
                e += (s.A / 2) * (x - s.b / s.A).squaredNorm();
        }
    }
    return e;
}

template <class T, int dim>
void MpmMultiTargetPinningForce<T, dim>::
    computeStressDifferential(const DisjointRanges& subrange, const StdVector<TM>& gradDv, StdVector<TM>& dstress, const TVStack& dvp, TVStack& dfp)
{
    DisjointRanges subset(subrange,
        particles.commonRanges(target_name(),
            target_scratch_name()));
    auto * config_scheme = particles.DataManager::getPointer(config_scheme_name());
    for (auto iter = particles.subsetIter(subset, target_name(), target_scratch_name()); iter; ++iter) {
        auto& t = iter.template get<0>(); // target
        auto& s = iter.template get<1>(); // scratch
        int p = iter.entryId();
        // auto& xn = particles.X[p];
        auto& xn = ((*config_scheme)[p] != CONFIG_UPDATE_LAGRANGIAN) ? X0Array[p] : particles.X[p];
        for (int i = 0; i < (int)t.x.size(); ++i) {
            if (collider_spring) {
                // const TV& xn = particles.X[p];
                const TV& x = xp[p]; // from scratch_xp
                const TV& xtn = t.x[i];
                TV n = (xtn - xn);
                if (n.squaredNorm() < (T)1e-12) continue;
                n.normalize();
                TM M = n * n.transpose();
                if ((x - xtn).dot(n) < 0)
                    dfp.col(p) += -t.k * M * dvp.col(p);
            }
            else
                dfp.col(p) += (-s.A * dvp.col(p));
        }
    }
}

template class MpmMultiTargetPinningForce<double, 3>;
} // namespace ZIRAN
