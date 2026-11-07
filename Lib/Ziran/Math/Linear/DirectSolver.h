#pragma once

#include <stdexcept>

#include <tbb/tbb.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Ziran/CS/Util/Debug.h>
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/profiler.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#ifdef ENABLE_AMGCL_CUDA
#include <amgcl/backend/vexcl.hpp>
#endif
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/reorder.hpp>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <memory>
#include <boost/range/iterator_range.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <vector>

AMGCL_USE_EIGEN_VECTORS_WITH_BUILTIN_BACKEND()

namespace ZIRAN {

template <class T>
class DirectSolver {
    using TStack = Vector<T, Eigen::Dynamic>;

    static TStack eigenSolve(const Eigen::SparseMatrix<T>& matrix, const TStack& rhs)
    {
        ZIRAN_TIMER();
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> solverLDLT;
        solverLDLT.compute(matrix);
        return solverLDLT.solve(rhs);
    }

    template <int block_size>
    static TStack blockAmgclSolver(const Eigen::SparseMatrix<T>& matrix_col, const TStack& rhs, const T relativeTol, const T absTol, const int iter, const T eps_strong = -1)
    {
        ZIRAN_TIMER();
        using Backend = amgcl::backend::builtin<T>;
        using AmgclSolver = amgcl::make_solver<
            amgcl::runtime::preconditioner<Backend>,
            amgcl::solver::cg<Backend>>;
        // TODO: Is this really necessary?
        Eigen::SparseMatrix<T, Eigen::RowMajor> matrix(matrix_col);
        TStack x = TStack::Zero(matrix.rows());

        typedef amgcl::static_matrix<T, block_size, block_size> value_type;
        typedef amgcl::static_matrix<T, block_size, 1> rhs_type;
        typedef amgcl::backend::builtin<value_type> BBackend;
        using BlockAmgclSolver = amgcl::make_solver<
            // Use AMG as preconditioner:
            amgcl::amg<
                BBackend,
                amgcl::coarsening::smoothed_aggregation,
                amgcl::relaxation::gauss_seidel>,
            // And BiCGStab as iterative solver:
            amgcl::solver::cg<BBackend>>;

        typename BlockAmgclSolver::params prm;
        if (relativeTol > 0) prm.solver.tol = relativeTol;
        if (absTol > 0) prm.solver.abstol = absTol;
        if (iter > 0) prm.solver.maxiter = iter;
        if (eps_strong >= 0) prm.precond.coarsening.aggr.eps_strong = eps_strong;

        auto block_A = amgcl::adapter::block_matrix<value_type>(matrix);
        BlockAmgclSolver amgcl_solver(block_A, prm);

        std::cout << amgcl_solver.precond() << std::endl;
        size_t iters;
        double resid;

        rhs_type const* fptr = reinterpret_cast<rhs_type const*>(&rhs[0]);
        rhs_type* xptr = reinterpret_cast<rhs_type*>(&x[0]);

        amgcl::backend::numa_vector<rhs_type> F(fptr, fptr + x.rows() / block_size);
        amgcl::backend::numa_vector<rhs_type> X(xptr, xptr + x.rows() / block_size);
        std::tie(iters, resid) = amgcl_solver(F, X);
        std::copy(X.data(), X.data() + X.size(), xptr);

        ZIRAN_INFO("AMGCL Iterations: ", std::setw(20), iters, std::setw(20), "Error:", resid);
        return x;
    }

    static TStack amgclSolve(const Eigen::SparseMatrix<T>& matrix_col, const TStack& rhs, const T relativeTol, const T absTol, const int iter, const T eps_strong = -1, const int block_size = -1, const T add_diagonal_epsilon = 0, const std::vector<T> null_spaces = std::vector<T>())
    {
        ZIRAN_TIMER();
        using Backend = amgcl::backend::builtin<T>;
        using AmgclSolver = amgcl::make_solver<
            amgcl::runtime::preconditioner<Backend>,
            amgcl::solver::cg<Backend>>;

        Eigen::SparseMatrix<T, Eigen::RowMajor> matrix(matrix_col);
        if (add_diagonal_epsilon) {
            for (int i = 0; i < matrix.rows(); ++i)
                matrix.coeffRef(i, i) += add_diagonal_epsilon;
            // TStack diag_vec = TStack::Ones(matrix.rows());
            // diag_vec *= add_diagonal_epsilon;
            // matrix += diag_vec.asDiagonal();
        }
        matrix.makeCompressed();

        TStack x = TStack::Zero(matrix.rows());

        boost::property_tree::ptree prm;
        prm.put("solver.tol", relativeTol);
        prm.put("solver.maxiter", iter);
        prm.put("precond.class", "amg");
        prm.put("precond.relax.type", "chebyshev");
        prm.put("precond.relax.degree", 16);
        prm.put("precond.relax.power_iters", 100);
        prm.put("precond.relax.higher", 2.0f);
        prm.put("precond.relax.lower", 1.0f / 120.0f);
        prm.put("precond.relax.scale", true);
        prm.put("precond.max_levels", 6);
        prm.put("precond.direct_coarse", false);
        prm.put("precond.coarsening.type", "smoothed_aggregation");
        prm.put("precond.coarsening.estimate_spectral_radius", true);
        prm.put("precond.coarsening.relax", 1.0f);
        prm.put("precond.coarsening.power_iters", 100);
        if (!null_spaces.empty()) {
            prm.put("precond.coarsening.nullspace.cols", null_spaces.size() / matrix.rows());
            prm.put("precond.coarsening.nullspace.rows", matrix.rows());
            prm.put("precond.coarsening.nullspace.B", &null_spaces[0]);
        }
        prm.put("precond.ncycle", 2);

        size_t iters;
        double resid;
        {
            ZIRAN_NAMED_TIMER("pressureSolveSetup");
            AmgclSolver amgcl_solver(matrix, prm);

            std::cout << amgcl_solver.precond() << std::endl;

            {
                ZIRAN_NAMED_TIMER("pressureSolveIteration");
                std::tie(iters, resid) = amgcl_solver(rhs, x);
            }
        }

        ZIRAN_INFO("AMGCL Iterations: ", std::setw(20), iters, std::setw(20), "Error:", resid);
        return x;
    }

#ifdef ENABLE_AMGCL_CUDA
    static boost::property_tree::ptree amgclCudaParams(const T relativeTol, const int iter,
        const std::vector<T>& null_spaces, const size_t rows)
    {
        boost::property_tree::ptree prm;
        prm.put("solver.tol", relativeTol);
        prm.put("solver.maxiter", iter);
        prm.put("precond.class", "amg");
        prm.put("precond.relax.type", "chebyshev");
        prm.put("precond.relax.degree", 5);
        prm.put("precond.relax.power_iters", 100);
        prm.put("precond.relax.higher", 2.0f);
        prm.put("precond.relax.lower", 1.0f / 120.0f);
        prm.put("precond.relax.scale", true);
        prm.put("precond.max_levels", 6);
        prm.put("precond.direct_coarse", false);
        prm.put("precond.coarsening.type", "smoothed_aggregation");
        prm.put("precond.coarsening.estimate_spectral_radius", true);
        prm.put("precond.coarsening.relax", 1.0f);
        prm.put("precond.coarsening.power_iters", 100);
        if (!null_spaces.empty()) {
            prm.put("precond.coarsening.nullspace.cols", null_spaces.size() / rows);
            prm.put("precond.coarsening.nullspace.rows", rows);
            prm.put("precond.coarsening.nullspace.B", &null_spaces[0]);
        }
        prm.put("precond.ncycle", 2);
        return prm;
    }

    template <class Index>
    static TStack amgclSolveCUDAFromCsrTyped(const size_t rows,
        const std::vector<Index>& ptr,
        const std::vector<Index>& col,
        const std::vector<T>& val,
        const TStack& myrhs, const T relativeTol, const T absTol,
        const int iter, const T eps_strong = -1, const int block_size = -1,
        const std::vector<T> null_spaces = std::vector<T>())
    {
        (void)absTol;
        (void)eps_strong;
        (void)block_size;
        ZIRAN_TIMER();

        typedef amgcl::backend::vexcl<T, Index, Index> Backend;
        typedef amgcl::make_solver<
            amgcl::runtime::preconditioner<Backend>,
            amgcl::solver::cg<Backend>>
            Solver;

        ZIRAN_ASSERT(ptr.size() == rows + 1, "CSR row pointer size mismatch");
        ZIRAN_ASSERT(col.size() == val.size(), "CSR col/value size mismatch");
        ZIRAN_ASSERT(myrhs.rows() == (Eigen::Index)rows, "CSR rhs size mismatch");

        typename Backend::params bprm;
        vex::Context ctx(vex::Filter::Env);
        ZIRAN_INFO(ctx);
        bprm.q = ctx;

        std::vector<T> rhs(myrhs.data(), myrhs.data() + myrhs.rows() * myrhs.cols());
        std::vector<T> x(rows, 0);
        boost::property_tree::ptree prm =
            amgclCudaParams(relativeTol, iter, null_spaces, rows);

        std::unique_ptr<Solver> solve;
        {
            // This includes construction of the AMG hierarchy and its device
            // upload. It is the preconditioner cost, not a CG iteration.
            ZIRAN_NAMED_TIMER("amgclCudaPreconditionerSetup");
            solve = std::make_unique<Solver>(std::tie(rows, ptr, col, val), prm, bprm);
        }
        decltype(Backend::copy_vector(rhs, bprm)) f_b;
        decltype(Backend::copy_vector(x, bprm)) x_b;
        {
            ZIRAN_NAMED_TIMER("amgclCudaVectorUpload");
            f_b = Backend::copy_vector(rhs, bprm);
            x_b = Backend::copy_vector(x, bprm);
        }

        std::cout << solve->precond() << std::endl;

        size_t iters;
        double resid;
        {
            ZIRAN_NAMED_TIMER("amgclCudaIteration");
            std::tie(iters, resid) = (*solve)(*f_b, *x_b);
        }

        ZIRAN_INFO("AMGCL Iterations: ", std::setw(20), iters, std::setw(20), "Error:", resid);
        ZIRAN_WARN("FINISHED THIS CALCULATION!");

        TStack xvec = TStack::Zero(rows);
        {
            ZIRAN_NAMED_TIMER("amgclCudaSolutionDownload");
            vex::copy(*x_b, x);
            for (uint i = 0; i < x.size(); i++)
                xvec[i] = x[i];
        }
        return xvec;
    }

    static TStack amgclSolveCUDA(const Eigen::SparseMatrix<T>& matrix_col, const TStack& myrhs, const T relativeTol, const T absTol, const int iter, const T eps_strong = -1, const int block_size = -1, const T add_diagonal_epsilon = 0, const std::vector<T> null_spaces = std::vector<T>())
    {
        ZIRAN_TIMER();

        typedef amgcl::backend::vexcl<T> Backend;
        typedef amgcl::make_solver<
            amgcl::runtime::preconditioner<Backend>,
            amgcl::solver::cg<Backend>>
            Solver;
        Eigen::SparseMatrix<T, Eigen::RowMajor> matrix(matrix_col);
        {
            ZIRAN_TIMER();
            if (add_diagonal_epsilon) {
                for (int i = 0; i < matrix.rows(); ++i)
                    matrix.coeffRef(i, i) += add_diagonal_epsilon;
                // TStack diag_vec = TStack::Ones(matrix.rows());
                // diag_vec *= add_diagonal_epsilon;
                // matrix += diag_vec.asDiagonal();
            }
        }

        typename Backend::params bprm;

        vex::Context ctx(vex::Filter::Env);
        ZIRAN_INFO(ctx);
        bprm.q = ctx;

        //            std::string relax_type[3] = {"spai0", "chebyshev"};
        //            std::string coarse_type[4] = {"aggregation", "ruge_stuben", "smoothed_aggr_emin", "smoothed_aggregation"};

        size_t rows;
        std::vector<ptrdiff_t> ptr;
        ptr.reserve(matrix.nonZeros());
        std::vector<ptrdiff_t> col;
        col.reserve(matrix.cols());
        std::vector<T> val;
        val.reserve(matrix.nonZeros());
        std::vector<T> rhs(myrhs.data(), myrhs.data() + myrhs.rows() * myrhs.cols());
        std::vector<T> x(matrix.rows());
        rows = matrix.rows();

        {
            // TODO: NEED OPTIMIZE
            ZIRAN_TIMER();
            ptr.push_back(0);
            for (int k = 0; k < matrix.outerSize(); ++k) {
                for (typename Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator it(matrix, k); it; ++it) {
                    col.push_back(it.col());
                    val.push_back(it.value());
                }
                ptr.push_back(col.size());
            }
        }
        //                        ZIRAN_WARN("===== AMGCL RELAX TYPE: ", relax_type[i], ", COARSE TYPE: ", coarse_type[j], " =====");

        {
            ZIRAN_TIMER();
            boost::property_tree::ptree prm =
                amgclCudaParams(relativeTol, iter, null_spaces, matrix.rows());
            {
                ZIRAN_TIMER();
                Solver solve(std::tie(rows, ptr, col, val), prm, bprm);

                auto f_b = Backend::copy_vector(rhs, bprm);
                auto x_b = Backend::copy_vector(x, bprm);

                std::cout << solve.precond() << std::endl;

                size_t iters;
                double resid;

                {
                    ZIRAN_TIMER();
                    std::tie(iters, resid) = solve(*f_b, *x_b);
                }

                ZIRAN_INFO("AMGCL Iterations: ", std::setw(20), iters, std::setw(20), "Error:", resid);
                ZIRAN_WARN("FINISHED THIS CALCULATION!");

                TStack xvec = TStack::Zero(matrix.rows());
                vex::copy(*x_b, x);
                for (uint i = 0; i < x.size(); i++)
                    xvec[i] = x[i];
                return xvec;
            }
        }
    }
#endif

public:
    enum SolverType { EIGEN,
        AMGCL,
        AMGCL_CUDA };

    DirectSolver() = default;

    static TStack solveCsrCUDA(const int rows, const int cols,
        const std::vector<int>& row_offsets,
        const std::vector<int>& col_indices,
        const std::vector<T>& values,
        const TStack& rhs, const T relativeTol = 1e-3,
        const T absTol = -1, const int iter = 10000,
        const T eps_strong = -1, const int block_size = 1,
        const T add_diagonal_epsilon = 0,
        const std::vector<T> null_spaces = std::vector<T>())
    {
        ZIRAN_TIMER();
        ZIRAN_ASSERT(rows == cols && rows == rhs.rows(), "CSR dimensions don't match");
        ZIRAN_ASSERT((int)row_offsets.size() == rows + 1, "CSR row_offsets size mismatch");
        ZIRAN_ASSERT(col_indices.size() == values.size(), "CSR col/value size mismatch");
        if (!rows)
            return TStack::Zero(0);

        for (int row = 0; row < rows; ++row) {
            ZIRAN_ASSERT(row_offsets[static_cast<std::size_t>(row)] <=
                    row_offsets[static_cast<std::size_t>(row + 1)],
                "CSR row_offsets must be monotone");
            for (int k = row_offsets[static_cast<std::size_t>(row)];
                 k < row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
                const int column = col_indices[static_cast<std::size_t>(k)];
                ZIRAN_ASSERT(column >= 0 && column < cols, "CSR column out of range");
            }
        }

        std::vector<int> adjusted_row_offsets;
        std::vector<int> adjusted_col_indices;
        std::vector<T> adjusted_values;
        const std::vector<int>* solve_row_offsets = &row_offsets;
        const std::vector<int>* solve_col_indices = &col_indices;
        const std::vector<T>* solve_values = &values;
        if (add_diagonal_epsilon) {
            adjusted_row_offsets.reserve(row_offsets.size());
            adjusted_col_indices.reserve(values.size() + rows);
            adjusted_values.reserve(values.size() + rows);
            adjusted_row_offsets.push_back(0);
            for (int row = 0; row < rows; ++row) {
                bool found_diagonal = false;
                for (int k = row_offsets[static_cast<std::size_t>(row)];
                     k < row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
                    const int column = col_indices[static_cast<std::size_t>(k)];
                    adjusted_col_indices.push_back(column);
                    T value = values[static_cast<std::size_t>(k)];
                    if (column == row) {
                        value += add_diagonal_epsilon;
                        found_diagonal = true;
                    }
                    adjusted_values.push_back(value);
                }
                if (!found_diagonal) {
                    adjusted_col_indices.push_back(row);
                    adjusted_values.push_back(add_diagonal_epsilon);
                }
                adjusted_row_offsets.push_back(static_cast<int>(adjusted_values.size()));
            }
            solve_row_offsets = &adjusted_row_offsets;
            solve_col_indices = &adjusted_col_indices;
            solve_values = &adjusted_values;
        }

#ifdef ENABLE_AMGCL_CUDA
        return amgclSolveCUDAFromCsrTyped(static_cast<size_t>(rows),
            *solve_row_offsets, *solve_col_indices, *solve_values,
            rhs, relativeTol, absTol, iter, eps_strong, block_size, null_spaces);
#else
        ZIRAN_WARN("solveCsrCUDA requested without ENABLE_AMGCL_CUDA; falling back to AMGCL through Eigen CSR import");
        std::vector<Eigen::Triplet<T>> triplets;
        triplets.reserve(solve_values->size());
        for (int row = 0; row < rows; ++row) {
            for (int k = (*solve_row_offsets)[static_cast<std::size_t>(row)];
                 k < (*solve_row_offsets)[static_cast<std::size_t>(row + 1)]; ++k)
                triplets.emplace_back(row,
                    (*solve_col_indices)[static_cast<std::size_t>(k)],
                    (*solve_values)[static_cast<std::size_t>(k)]);
        }
        Eigen::SparseMatrix<T> matrix(rows, cols);
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        matrix.makeCompressed();
        return solve(matrix, rhs, AMGCL, relativeTol, absTol, iter, eps_strong,
            block_size, false, false, 0, null_spaces);
#endif
    }

    static TStack solve(const Eigen::SparseMatrix<T>& matrix, const TStack& rhs, SolverType solver_type, const T relativeTol = 1e-3, const T absTol = -1, const int iter = 10000, const T eps_strong = -1, const int block_size = 1, const bool block_backend = false, const bool diag_scaling = false, const T add_diagonal_epsilon = 0, const std::vector<T> null_spaces = std::vector<T>())
    {
        ZIRAN_ASSERT(matrix.rows() == matrix.cols() && matrix.rows() == rhs.rows(), "dimensions don't match");
        Eigen::SparseMatrix<T> matrix_scaled = matrix;
        TStack rhs_scaled = rhs;
        auto D_inv = (matrix.diagonal()).asDiagonal().inverse();
        if (diag_scaling) {
            ZIRAN_WARN("AMGCL SOLVE WITH DIAGONAL SCALING!");
            matrix_scaled = D_inv * matrix * D_inv;
            rhs_scaled = D_inv * rhs;
        }

        if (!matrix.rows())
            return TStack::Zero(0);
        if (solver_type == EIGEN)
            return eigenSolve(matrix, rhs);
        else if (solver_type == AMGCL) {
            if (block_backend && block_size == 2)
                return blockAmgclSolver<2>(matrix, rhs, relativeTol, absTol, iter, eps_strong);
            else if (block_backend && block_size == 3)
                return blockAmgclSolver<3>(matrix, rhs, relativeTol, absTol, iter, eps_strong);
            else {
                TStack x = amgclSolve(matrix_scaled, rhs_scaled, relativeTol, absTol, iter, eps_strong, block_size, add_diagonal_epsilon, null_spaces);
                if (diag_scaling) x = D_inv * x;
                return x;
            }
        }
#ifdef ENABLE_AMGCL_CUDA
        else {
            TStack x = amgclSolveCUDA(matrix_scaled, rhs_scaled, relativeTol, absTol, iter, eps_strong, block_size, add_diagonal_epsilon, null_spaces);
            if (diag_scaling) x = D_inv * x;
            return x;
        }
#else
        else {
            throw std::runtime_error("AMGCL_CUDA requires a CUDA-enabled build");
        }
#endif
    }
};

} // namespace ZIRAN
