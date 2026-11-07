#pragma once

#include <MPM/Grid/MpmGrid.h>
#include <mutex>

namespace ZIRAN {

// Shared host diagnostic used when reporting total implicit-step energy.
template <class T, int dim>
double computeGridKineticEnergy(MpmGrid<T, dim>& grid, bool use_updated_velocity = false)
{
    std::mutex mtx;
    double result = 0;
    grid.iterateGrid([&](Vector<int, dim> node, GridState<T, dim>& g) {
        double m = (double)g.m;
        Vector<double, dim> v = Vector<double, dim>::Zero();
        if (use_updated_velocity)
            v = g.new_v.template cast<double>();
        else
            v = g.v.template cast<double>();
        double energy = 0.5 * m * v.squaredNorm();
        mtx.lock();
        result += energy;
        mtx.unlock();
    });
    return result;
}

} // namespace ZIRAN
