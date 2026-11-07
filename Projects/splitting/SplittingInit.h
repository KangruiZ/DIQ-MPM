#pragma once

#include <Ziran/CS/Util/RandomNumber.h>
#include <MPM/Initialization/MpmSceneInitializer.h>

#include "SplittingSimulation.h"

namespace ZIRAN {

template <class T, int dim>
class SplittingInitBase {
public:
    using TV = Vector<T, dim>;
    using TVI = Vector<int, dim>;

    SplittingSimulation<T, dim>& sim; // This contains the real scene data.
    MpmSceneInitializer<T, dim> init_helper;
    const int test_number;

    SplittingInitBase(SplittingSimulation<T, dim>& sim, const int test_number)
        : sim(sim)
        , init_helper(sim)
        , test_number(test_number)
    {
    }

    // Normal start call.
    void start(int max_frame = -1)
    {
        printBasicInfo();
        initialize();
        if (max_frame > 0)
            sim.end_frame = max_frame;
        sim.simulate();
    }

    // Restart call.
    void restart(int frame, int max_frame = -1)
    {
        initialize();
        // reload() restores the example's normal end frame.  Apply a smoke-test
        // cap after initialization so a checkpoint validation can advance a
        // single frame rather than always running to the scene's final frame.
        if (max_frame > 0)
            sim.end_frame = max_frame;
        sim.restart(frame);
        sim.simulate();
    }

    void initialize()
    {
        reload();
        sim.initialize();
        sim.reinitialize();
    }

    virtual void reload() = 0;

    void printBasicInfo()
    {
        bool is_double = std::is_same<T, double>();
        ZIRAN_INFO("Simulation using double:", is_double);
        ZIRAN_INFO("Simulation dimension:", dim);
    }
};
} // namespace ZIRAN
