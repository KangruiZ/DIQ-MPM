#ifndef BACKWARD_EULER_H
#define BACKWARD_EULER_H

#include <Ziran/Math/Linear/DenseExt.h>
#include <Ziran/Math/Linear/KrylovSolvers.h>
#include <Ziran/Math/Linear/Minres.h>
#include <Ziran/Math/Linear/ConjugateGradient.h>
#include <Ziran/Math/MathTools.h>
#include <Ziran/Sim/SimulationBase.h>

#include <tbb/tbb.h>

namespace ZIRAN {

/**
   Simulation owns Objective and NewtonsMethod.
   This will allow options to do Newton.
   It stores the parameters for them. But it is a little weird.
   This is supposed to be used by both MPM and Elasticity.
 */
template <class Simulation>
class BackwardEulerLagrangianForceObjective {
public:
    using T = typename Simulation::Scalar;
    static const int dim = Simulation::dim;
    using Objective = BackwardEulerLagrangianForceObjective<Simulation>;
    using TV = Vector<T, dim>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using Vec = Vector<T, Eigen::Dynamic>;

    using Scalar = T;
    using NewtonVector = TVStack;

    Simulation& simulation;

    std::function<void(TVStack&)> project;
    std::function<void(const TVStack&, TVStack&)> precondition;
    // Optional application-provided device linear solve. It returns true only
    // when it fully populated `ddv`; otherwise the established CPU Krylov
    // solver remains the fallback.
    std::function<bool(TVStack&, const TVStack&, T)> device_linear_solve;
    // Optional fully device-resident Newton-vector path. The first callback
    // assembles the current residual and returns only its convergence norm;
    // the second commits the device linear solve result and advances the
    // device trial state. Together they let Newton skip materializing its
    // node-sized residual/step vectors on the host.
    std::function<bool(T&)> device_residual_norm;
    std::function<bool()> device_step_commit;
    // Optional device-resident backtracking line search. The callback tests
    // the already solved device Newton direction, commits the accepted scale,
    // and returns its total energy/step without materializing dv on the host.
    std::function<bool(T, T&, T&)> device_line_search_commit;
    // Optional nonlinear trial-state update. It is called after `dv` has been
    // copied to the simulation grid. It either populates the legacy force
    // scratch fields or pairs with device_residual_force/device_linear_solve to
    // keep the nonlinear state on device. Returning true bypasses the CPU
    // particle traversal while retaining the established inertia update.
    std::function<bool()> device_state_update;
    // A fully resident backend can make the host SPGrid node velocity a stale
    // compatibility mirror.  When this predicate is true, avoid the otherwise
    // redundant moveNodes() traversal before the device trial update.  If the
    // device update unexpectedly declines the state, moveNodes() is performed
    // before falling back to the CPU force path.
    std::function<bool()> device_node_state_owned;
    // Optional device-computed force raster. If set and it returns true, it has
    // already added the material/contact force contribution into the residual
    // (the same term force->addScaledForces(dt) would add), so the CPU force
    // rasterizer is skipped. Gravity, inertia and projection stay on the host.
    std::function<bool(TVStack&)> device_residual_force;
    // Optional force-only energy for line search. Material/pinning terms may
    // stay on the device while inertia and gravity retain their established
    // host grid evaluation below.
    std::function<bool(T&)> device_force_energy;
    // Optional complete objective energy (force + inertia + gravity). This is
    // needed when the authoritative Newton increment is device-resident and
    // the host simulation.dv is intentionally only a compatibility mirror.
    std::function<bool(T&)> device_total_energy;
    // Optional finalization after Newton has settled on its last trial state.
    // Device paths use this to materialize host compatibility state that is
    // needed once per implicit step, but not during every Newton iteration.
    std::function<void()> device_state_finalize;

    std::unique_ptr<LinearSolver<T, Objective, TVStack>> linear_solver;
    Minres<T, Objective, TVStack>& minres;

    BackwardEulerLagrangianForceObjective(Simulation& simulation)
        : simulation(simulation)
        , project([](TVStack&) {})
        , precondition([](const TVStack& x, TVStack& b) { b = x; })
        , linear_solver(std::make_unique<Minres<T, Objective, TVStack>>(20))
        , minres(*reinterpret_cast<Minres<T, Objective, TVStack>*>(linear_solver.get()))
    {
        linear_solver->setTolerance(1);
    }

    BackwardEulerLagrangianForceObjective(const BackwardEulerLagrangianForceObjective& objective) = delete;

    ~BackwardEulerLagrangianForceObjective() {}

    template <class Projection>
    void initialize(const Projection& projection_function)
    {
        project = projection_function;
    }

    void switchLinearSolver(SimulationBase::LinearSolverType p_linear_solver_type)
    {
        switch (p_linear_solver_type) {
        case SimulationBase::MINRES:
            linear_solver = std::make_unique<Minres<T, Objective, TVStack>>(20);
            break;

        case SimulationBase::CG:
            linear_solver = std::make_unique<ConjugateGradient<T, Objective, TVStack>>(20);
            break;

        default:
            ZIRAN_ASSERT(0 && "Invalid linear solver type!");
            break;
        }
        simulation.linear_solver_type = p_linear_solver_type;
    }

    // Called by NewtonsMethod::solve. Builds residual = dt*gravity*m +
    // force->addScaledForces(dt) + inertia, then project(). force is the sole
    // MpmForceAssembly; its addScaledForces -> rasterizeForceToTVStack, reading the
    // scratch_force_stress/scratch_fp that updateState() below has filled.
    // (See IMPLICIT TIME STEP MAP in MpmSimulationBase::backwardEulerStep.)
    void computeResidual(TVStack& residual)
    {
        ZIRAN_TIMER();
        ZIRAN_ASSERT(residual.cols() == simulation.num_nodes);
        tbb::parallel_for(tbb::blocked_range<int>(0, residual.cols(), 256),
            [&](const tbb::blocked_range<int>& range) {
                int start = range.begin();
                int length = (range.end() - range.begin());
                TV dtg = simulation.dt * simulation.gravity;
                residual.middleCols(start, length) = dtg * simulation.mass_matrix.segment(start, length).transpose();
            });

        if (!(device_residual_force && device_residual_force(residual)))
            simulation.force->addScaledForces(simulation.dt, residual);
        simulation.inertia->addScaledForces(simulation.dt, residual);
        project(residual);
    }

    // called by Newton
    T computeNorm(const TVStack& residual) const
    {
        // ZIRAN_TIMER();
        ZIRAN_ASSERT(residual.cols() == simulation.num_nodes);
        T norm_sq = tbb::parallel_reduce(tbb::blocked_range<int>(0, residual.cols(), 256), (T)0,
            [&](const tbb::blocked_range<int>& range, T ns) -> T {
                int start = range.begin();
                int length = (range.end() - range.begin());
                const auto& r_block = residual.middleCols(start, length);
                const auto& mass_block = simulation.mass_matrix.segment(start, length).transpose();
                return ns + ((r_block.colwise().squaredNorm()).array() / mass_block.array()).sum();
            },
            [](T a, T b) -> T { return a + b; });
        return std::sqrt(norm_sq);
    }

    T innerProduct(const TVStack& ddv, const TVStack& residual) const
    {
        ZIRAN_ASSERT(residual.cols() == ddv.cols());
        T result = tbb::parallel_reduce(tbb::blocked_range<int>(0, residual.cols(), 256), (T)0,
            [&](const tbb::blocked_range<int>& range, T ns) -> T {
                int start = range.begin();
                int length = (range.end() - range.begin());
                const auto& ddv_block = ddv.middleCols(start, length);
                const auto& r_block = residual.middleCols(start, length);
                const auto& mass_block = simulation.mass_matrix.segment(start, length);
                // ZIRAN_DEBUG(ddv_block);
                // ZIRAN_DEBUG(r_block);
                // ZIRAN_DEBUG(mass_block);

                T to_add = ((ddv_block.transpose() * r_block).diagonal().array() / mass_block.array()).sum();
                // ZIRAN_DEBUG(to_add);
                return ns += to_add;
            },
            [](T a, T b) -> T { return a + b; });
        return result;
    }

    // Called by NewtonsMethod::solve once per iteration to move the material
    // state to the current velocity guess: moveNodes(dv), then
    // force->updateStrainAndPositions() -> updateForcesAtState()
    // -> forEachForceTerm(updateImplicitState) over the sub-helpers.
    // (See IMPLICIT TIME STEP MAP in MpmSimulationBase::backwardEulerStep.)
    void updateStateForDv(const TVStack& dv)
    {
        ZIRAN_TIMER();
        ZIRAN_ASSERT(dv.cols() == simulation.num_nodes);
        const bool resident_nodes = device_node_state_owned &&
            device_node_state_owned();
        if (!resident_nodes)
            simulation.moveNodes(dv);
        if (device_state_update && device_state_update()) {
            simulation.inertia->updatePositionBasedState();
            return;
        }
        if (resident_nodes)
            simulation.moveNodes(dv);
        simulation.force->updateStrainAndPositions();
        simulation.inertia->updatePositionBasedState();
    }

    T totalEnergy()
    {
        ZIRAN_TIMER();
        T device_energy = 0;
        if (device_total_energy && device_total_energy(device_energy))
            return device_energy;
        T result = 0;
        T force_energy = 0;
        if (device_force_energy && device_force_energy(force_energy))
            result += force_energy;
        else
            result += simulation.force->totalEnergy();
        result += simulation.inertia->totalEnergy();

        result += tbb::parallel_reduce(tbb::blocked_range<int>(0, simulation.dv.cols(), 256), (T)0,
            [&](const tbb::blocked_range<int>& range, T ns) -> T {
                int start = range.begin();
                int length = (range.end() - range.begin());
                const auto& dv_block = simulation.dv.middleCols(start, length);
                const auto& mass_block = simulation.mass_matrix.segment(start, length);
                return ns - simulation.dt * ((simulation.gravity.transpose() * dv_block).dot(mass_block));
            },
            [](T a, T b) -> T { return a + b; });
        return result;
    }

    // Called by NewtonsMethod::solve to solve H*ddv = residual. MPM is always
    // matrix-free: a Krylov solver (Minres/CG) whose each iteration calls
    // multiply() -> force->addScaledForceDifferential()
    // -> forEachForceTerm(computeStressDifferential).
    // (See IMPLICIT TIME STEP MAP in MpmSimulationBase::backwardEulerStep.)
    bool solveNewtonStep(TVStack& ddv, const TVStack& residual,
        const T linear_solve_relative_tolerance)
    {
        ZIRAN_TIMER();
        ZIRAN_ASSERT(ddv.cols() == simulation.num_nodes);
        ddv.setZero();
        if (device_linear_solve &&
            device_linear_solve(ddv, residual, linear_solve_relative_tolerance))
            return true;
        linear_solver->setRelativeTolerance(linear_solve_relative_tolerance);
        linear_solver->solve(*this, ddv, residual, simulation.verbose);
        return false;
    }

    // Linear-operator interface required by the Krylov solvers (Minres/CG/GMRES):
    // computes b = H*x, where H is the Newton system matrix (mass + dt^2 stiffness).
    // Matrix-free: b = -dt^2 * (inertia + force) force differentials at x.
    void multiply(const TVStack& x, TVStack& b) const
    {
        ZIRAN_ASSERT(x.cols() == simulation.num_nodes);
        ZIRAN_ASSERT(b.cols() == simulation.num_nodes);
        using MATH_TOOLS::sqr;
        b.setZero();
        // Note the relationship H dx = - df, where H is the stiffness matrix
        simulation.inertia->addScaledForceDifferential(-sqr(simulation.dt), x, b);
        simulation.force->addScaledForceDifferential(-sqr(simulation.dt), x, b);
    }

    template <class Func>
    void setProjection(Func project_func)
    {
        project = project_func;
    }

    template <class Func>
    void setPreconditioner(Func preconditioner_func)
    {
        precondition = preconditioner_func;
    }
};
} // namespace ZIRAN

#endif
