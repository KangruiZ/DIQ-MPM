#ifndef NEWTONS_METHOD_H
#define NEWTONS_METHOD_H
#include <Ziran/CS/Util/Debug.h>
#include <Ziran/CS/Util/ErrorContext.h>
#include <Ziran/CS/Util/Meta.h>
#include <Ziran/CS/Util/Timer.h>
#include <iostream>

namespace ZIRAN {
/**
  Newton's Method

  Templatized on the objective, which must implement:
    computeResidual(r)                 : residual (gradient) at the current state
    computeNorm(r)                     : convergence norm
    solveNewtonStep(step, r, tol)      : solve H*step = r for the Newton step
    updateStateForDv(dv)               : move material state to the guess dv
    totalEnergy()                      : energy (line search only)
  and the typedefs NewtonVector and Scalar.
*/
template <class Objective>
class NewtonsMethod {
    using Vec = typename Objective::NewtonVector;
    using T = typename Objective::Scalar;

    Objective& objective;

public:
    Vec step_direction;
    Vec residual;

    T tolerance;
    int max_iterations;
    T linear_solve_tolerance_scale;
    bool linesearch;

    NewtonsMethod(Objective& objective, const T tolerance = (T)1e-6, const int max_iterations = 5, const bool use_linesearch = false)
        : objective(objective)
        , tolerance(tolerance)
        , max_iterations(max_iterations)
        , linear_solve_tolerance_scale((T)1)
        , linesearch(use_linesearch)
    {
    }

    bool solve(Vec& dv, const bool verbose = false)
    {
        ZIRAN_TIMER();
        step_direction.resizeLike(dv);
        residual.resizeLike(dv);

        objective.updateStateForDv(dv);
        // printf("result in newton %f\n", objective.totalEnergy());
        T E0 = T(0.0); // totalEnergy of last iteration, for line search only
        Vec dv0; // dv of last iteration, for line search only
        if (linesearch) {
            E0 = objective.totalEnergy();
            dv0 = dv;
        }

        for (int it = 0; it < max_iterations; it++) {
            T residual_norm = T(0);
            const bool device_residual = objective.device_residual_norm &&
                objective.device_residual_norm(residual_norm);
            if (!device_residual) {
                objective.computeResidual(residual);
                residual_norm = objective.computeNorm(residual);
            }
            ZIRAN_VERB_IF(verbose, "Newton iter ", it, "; (designated norm) residual = ", residual_norm);
            ZIRAN_CONTEXT("Newton step", it);
            ZIRAN_CONTEXT(residual_norm);
            if (residual_norm < tolerance) {
                ZIRAN_VERB_IF(true, "Newton terminates at ", it, "; (designated norm) residual = ", residual_norm);
                return true;
            }
            else {
                ZIRAN_VERB_IF(true, "Newton residual = ", residual_norm);
            }
            T linear_solve_relative_tolerance = std::min((T)0.5, linear_solve_tolerance_scale * std::sqrt(std::max(residual_norm, tolerance)));
            const bool device_step = objective.solveNewtonStep(
                step_direction, residual, linear_solve_relative_tolerance);

            if (linesearch) {
                T stepSize = T(1.0), E;
                const bool device_committed = device_residual && device_step &&
                    objective.device_line_search_commit &&
                    objective.device_line_search_commit(E0, E, stepSize);
                ZIRAN_ASSERT(!device_residual || device_committed,
                    "device Newton residual requires a device line-search commit");
                if (!device_committed) {
                    do {
                        dv = dv0 + stepSize * step_direction;
                        objective.updateStateForDv(dv);
                        E = objective.totalEnergy();
                        stepSize /= T(2.0);
                    } while (E > E0);
                    stepSize *= T(2.0);
                    dv0 = dv;
                }
                ZIRAN_INFO("Newton it ", it, " stepSize = ", stepSize);
                E0 = E;
            }
            else {
                const bool device_committed = device_residual && device_step &&
                    objective.device_step_commit &&
                    objective.device_step_commit();
                ZIRAN_ASSERT(!device_residual || device_committed,
                    "device Newton residual requires a device solve/commit pair");
                if (!device_committed) {
                    dv += step_direction;
                    objective.updateStateForDv(dv);
                }
            }
        }
        // max iterations reached
        //NOTE: previously then x and state are not matched
        // now they are matched, will this be a problem?
        return false;
    }
};
} // namespace ZIRAN
#endif
