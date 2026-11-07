#pragma once

#include "../splitting/SplittingSimulation.h"
#include "GpuOperatorProbe.h"
#include "GpuSparseGridProbe.h"
#include <Ziran/Sim/SimulationBase.h>
#include "PositionCorrector.h"
#include <Ziran/CS/Util/AttributeNamesForward.h>
#include <Ziran/Math/SphKernels.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace ZIRAN {

template <class T, int dim> class CouplingSimulation : public SimulationBase {
public:
  using Base = SimulationBase;

  typedef Vector<T, dim> TV;
  typedef Vector<int, dim> IV;
  typedef Matrix<T, dim, dim> TM;
  typedef Vector<T, Eigen::Dynamic> TStack;
  typedef Matrix<T, dim, Eigen::Dynamic> TVStack;

  SplittingSimulation<T, dim> e1;
  SplittingSimulation<T, dim> e2;

  TVStack surface_tension_delta1;
  TVStack surface_tension_delta2;
  PositionCorrector<T, dim> position_corrector1;
  PositionCorrector<T, dim> position_corrector2;

  /////////////////////////////////////////////////////////////////
  //////////////////////// INTERFACE START ////////////////////////
  /////////////////////////////////////////////////////////////////
  bool use_position_correction = true;
  bool use_surface_tension = false;
  bool gpu_stage2_probe = false;
  bool gpu_stage3a_probe = false;
  bool gpu_stage3b_probe = false;
  bool gpu_stage3b_writeback = false;
  bool gpu_build_system = false;
  bool gpu_full_pipeline = false;
  bool gpu_full_pipeline_strict = false;
  bool gpu_solid_implicit_probe = false;
  bool gpu_solid_implicit_pcg = false;
  bool gpu_solid_implicit_state = false;
  bool gpu_position_corrector = false;
  bool gpu_position_corrector_density = false;
  bool gpu_position_corrector_pages = false;
  bool gpu_position_corrector_direct_topology_probe = false;
  bool gpu_position_corrector_direct_matrix_probe = false;
  bool gpu_position_corrector_direct_solver_probe = false;
  bool gpu_position_corrector_direct_legacy_order_probe = false;
  bool gpu_position_corrector_direct_legacy_solver_probe = false;
  bool gpu_position_corrector_direct_legacy_solver_writeback = false;
  bool gpu_position_corrector_direct_legacy_solver = false;
  bool gpu_position_corrector_direct_gpu_solver = false;
  bool gpu_position_corrector_direct_amgx_solver = false;
  bool gpu_position_corrector_direct_postsolve = false;
  bool gpu_coupling_amgx_solver = false;
  bool gpu_position_corrector_direct_pushout = false;
  bool gpu_build_system_cpu_numerical_grid = false;
  bool gpu_build_system_cpu_pressure_grid = false;
  bool gpu_build_system_cpu_velocity_grid = false;
  bool gpu_fluid_velocity_transfer = false;
  GPU_STAGE2::PersistentSpatialSearch3D gpu_solid_search;
#ifdef ENABLE_AMGX
  GPU_STAGE2::PersistentAmgxCsrSolver gpu_coupling_amgx_context;
#endif

  T kkt_relax =
      0; // this should be set to 1/fluid_rho * penalty_stiffness_inverse.
  T relative_tol, abs_tol;
  int max_iter;
  StdVector<TV> wall_particles;
  enum SOLVER_TYPE { EIGEN, AMGCL, AMGCL_CUDA } solverType = AMGCL_CUDA;
  void setSolver(const T relTol, const T absTol, int maxIter,
                 SOLVER_TYPE type = AMGCL_CUDA) {
    relative_tol = relTol;
    abs_tol = absTol;
    max_iter = maxIter;
    solverType = type;
  }
  /////////////////////////////////////////////////////////////////
  //////////////////////// INTERFACE END //////////////////////////
  /////////////////////////////////////////////////////////////////

  CouplingSimulation()
      : Base(), position_corrector1(e1), position_corrector2(e2) {}

  double calculateDt() override {
    return std::min(e1.calculateDt(), e2.calculateDt());
  }

  void restart(int new_restart_frame) override {
    e1.restart(new_restart_frame);
    e2.restart(new_restart_frame);
    Base::restart(new_restart_frame);
  }

  void initialize() override {
    e1.autorestart = e2.autorestart = false;
    e1.initialize();
    e2.initialize();
    Base::initialize();
  }

  void reinitialize() override {
    e1.reinitialize();
    e2.reinitialize();
    Base::reinitialize();
  }

  int interface_cnt;
  struct HostCsrMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<T> values;
  };

  struct Stage3BCsrProducts {
    // These are the same W^T matrices used to form the Stage3B products.
    // Retain them so the post-solve interface-alpha update does not upload
    // and transpose W a second time on every coupling substep.
    GPU_STAGE3A::DeviceCsrMatrix e1_wt_device;
    GPU_STAGE3A::DeviceCsrMatrix e2_wt_device;
    GPU_STAGE3A::DeviceCsrMatrix e1_dminv_device;
    GPU_STAGE3A::DeviceCsrMatrix e1_dminvg_device;
    GPU_STAGE3A::DeviceCsrMatrix e1_dminv_wt_device;
    GPU_STAGE3A::DeviceCsrMatrix e1_w_minv_device;
    GPU_STAGE3A::DeviceCsrMatrix e1_w_minv_g_device;
    GPU_STAGE3A::DeviceCsrMatrix e1_w_minv_wt_device;
    GPU_STAGE3A::DeviceCsrMatrix e2_dminv_device;
    GPU_STAGE3A::DeviceCsrMatrix e2_dminvg_device;
    GPU_STAGE3A::DeviceCsrMatrix e2_dminv_wt_device;
    GPU_STAGE3A::DeviceCsrMatrix e2_w_minv_device;
    GPU_STAGE3A::DeviceCsrMatrix e2_w_minv_g_device;
    GPU_STAGE3A::DeviceCsrMatrix e2_w_minv_wt_device;

    GPU_STAGE3A::CsrMatrix e1_dminv;
    GPU_STAGE3A::CsrMatrix e1_dminvg;
    GPU_STAGE3A::CsrMatrix e1_dminv_wt;
    GPU_STAGE3A::CsrMatrix e1_w_minv;
    GPU_STAGE3A::CsrMatrix e1_w_minv_g;
    GPU_STAGE3A::CsrMatrix e1_w_minv_wt;
    GPU_STAGE3A::CsrMatrix e2_dminv;
    GPU_STAGE3A::CsrMatrix e2_dminvg;
    GPU_STAGE3A::CsrMatrix e2_dminv_wt;
    GPU_STAGE3A::CsrMatrix e2_w_minv;
    GPU_STAGE3A::CsrMatrix e2_w_minv_g;
    GPU_STAGE3A::CsrMatrix e2_w_minv_wt;
  };

  Eigen::SparseMatrix<double>
  toDoubleSparse(const Eigen::SparseMatrix<T> &matrix) const {
    return matrix.template cast<double>();
  }

  void logGpuStage3AStats(const GPU_STAGE3A::OperatorCompareStats &stats) {
    ZIRAN_INFO("Stage3A GPU operator probe ", stats.label,
               ": rows=", stats.rows, " cols=", stats.cols,
               " nonzeros=", stats.nonzeros,
               " max_abs_error=", stats.max_abs_error,
               " mean_abs_error=", stats.mean_abs_error,
               " max_rel_error=", stats.max_rel_error);
  }

  void logGpuStage3BAssemblyStats(const GPU_STAGE3A::OperatorCompareStats &stats) {
    ZIRAN_INFO("Stage3B GPU assembly probe ", stats.label,
               ": rows=", stats.rows, " cols=", stats.cols,
               " nonzeros=", stats.nonzeros,
               " max_abs_error=", stats.max_abs_error,
               " mean_abs_error=", stats.mean_abs_error,
               " max_rel_error=", stats.max_rel_error);
  }

  void logVectorDiffStats(const std::string &label, const TStack &reference,
                          const TStack &probe) const {
    ZIRAN_ASSERT(reference.rows() == probe.rows(), "vector size mismatch");
    T max_abs_error = 0;
    T mean_abs_error = 0;
    T reference_norm_inf = 0;
    for (int i = 0; i < reference.rows(); ++i) {
      const T err = std::abs(reference(i) - probe(i));
      max_abs_error = std::max(max_abs_error, err);
      mean_abs_error += err;
      reference_norm_inf = std::max(reference_norm_inf, std::abs(reference(i)));
    }
    if (reference.rows() > 0)
      mean_abs_error /= (T)reference.rows();
    ZIRAN_INFO(label, ": size=", reference.rows(),
               " max_abs_error=", max_abs_error,
               " mean_abs_error=", mean_abs_error,
               " max_rel_error=",
               max_abs_error / std::max(reference_norm_inf, (T)1e-30));
  }

  void logLinearResidualStats(const std::string &label,
                              const Eigen::SparseMatrix<T> &matrix,
                              const TStack &rhs, const TStack &x) const {
    ZIRAN_ASSERT(matrix.rows() == rhs.rows() && rhs.rows() == x.rows(),
                 "residual size mismatch");
    const TStack residual = matrix * x - rhs;
    const T residual_l2 = residual.norm();
    const T rhs_l2 = rhs.norm();
    T residual_inf = 0;
    for (int i = 0; i < residual.rows(); ++i)
      residual_inf = std::max(residual_inf, std::abs(residual(i)));
    ZIRAN_INFO(label, ": rows=", matrix.rows(), " nonzeros=", matrix.nonZeros(),
               " residual_l2=", residual_l2,
               " relative_l2=", residual_l2 / std::max(rhs_l2, (T)1e-30),
               " residual_inf=", residual_inf);
  }

  void logCsrResidualStats(const std::string &label,
                           const HostCsrMatrix &matrix, const TStack &rhs,
                           const TStack &x) const {
    ZIRAN_ASSERT(matrix.rows == rhs.rows() && rhs.rows() == x.rows(),
                 "CSR residual size mismatch");
    T residual_l2_sq = 0;
    T rhs_l2_sq = 0;
    T residual_inf = 0;
    for (int row = 0; row < matrix.rows; ++row) {
      T ax = 0;
      for (int k = matrix.row_offsets[static_cast<std::size_t>(row)];
           k < matrix.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
        ax += matrix.values[static_cast<std::size_t>(k)] *
              x(matrix.col_indices[static_cast<std::size_t>(k)]);
      }
      const T residual = ax - rhs(row);
      residual_l2_sq += residual * residual;
      rhs_l2_sq += rhs(row) * rhs(row);
      residual_inf = std::max(residual_inf, std::abs(residual));
    }
    const T residual_l2 = std::sqrt(residual_l2_sq);
    const T rhs_l2 = std::sqrt(rhs_l2_sq);
    ZIRAN_INFO(label, ": rows=", matrix.rows,
               " nonzeros=", matrix.values.size(),
               " residual_l2=", residual_l2,
               " relative_l2=", residual_l2 / std::max(rhs_l2, (T)1e-30),
               " residual_inf=", residual_inf);
  }

  GPU_STAGE3A::CsrMatrix
  buildGpuStage3BProduct(const std::string &label,
                         const GPU_STAGE3A::CsrMatrix &left,
                         const GPU_STAGE3A::CsrMatrix &right) const {
    ZIRAN_TIMER();
    GPU_STAGE3A::CsrMatrix product =
        GPU_STAGE3A::multiplyCsrMatrices(left, right);
    ZIRAN_INFO("Stage3B GPU writeback ", label,
               ": rows=", product.rows, " cols=", product.cols,
               " nonzeros=", product.values.size());
    return product;
  }

  GPU_STAGE3A::DeviceCsrMatrix
  buildGpuStage3BDeviceProduct(const std::string &label,
                               const GPU_STAGE3A::DeviceCsrMatrix &left,
                               const GPU_STAGE3A::DeviceCsrMatrix &right) const {
    ZIRAN_TIMER();
    GPU_STAGE3A::DeviceCsrMatrix product =
        GPU_STAGE3A::multiplyDeviceCsrMatrices(left, right);
    ZIRAN_INFO("Stage3B GPU writeback device ", label,
               ": rows=", product.rows, " cols=", product.cols,
               " nonzeros=", product.nonzeros);
    return product;
  }

  Stage3BCsrProducts
  buildGpuStage3BProducts(const Eigen::SparseMatrix<T> &d,
                          const Eigen::SparseMatrix<T> &minv,
                          const Eigen::SparseMatrix<T> &g,
                          const Eigen::SparseMatrix<T> &w,
                          const Eigen::SparseMatrix<T> &hat_d,
                          const Eigen::SparseMatrix<T> &hat_minv,
                          const Eigen::SparseMatrix<T> &hat_g,
                          const Eigen::SparseMatrix<T> &hat_w) const {
    ZIRAN_TIMER();
    Stage3BCsrProducts products;
    GPU_STAGE3A::DeviceCsrMatrix d_dev;
    GPU_STAGE3A::DeviceCsrMatrix minv_dev;
    GPU_STAGE3A::DeviceCsrMatrix g_dev;
    GPU_STAGE3A::DeviceCsrMatrix hat_d_dev;
    GPU_STAGE3A::DeviceCsrMatrix hat_minv_dev;
    GPU_STAGE3A::DeviceCsrMatrix hat_g_dev;

    const bool reuse_device_assembly =
        e1.gpu_device_assembly_valid && e2.gpu_device_assembly_valid;
    if (reuse_device_assembly) {
      minv_dev = e1.gpu_m_inv;
      g_dev = e1.gpu_g;
      d_dev = GPU_STAGE3A::transposeDeviceCsrMatrix(g_dev, -1.0);
      hat_minv_dev = e2.gpu_m_inv;
      hat_g_dev = e2.gpu_g;
      hat_d_dev = GPU_STAGE3A::transposeDeviceCsrMatrix(hat_g_dev, -1.0);
      ZIRAN_INFO("Stage3B reusing device-resident base assembly blocks");
    } else {
      const GPU_STAGE3A::CsrMatrix d_csr =
          GPU_STAGE3A::makeCsrFromEigen(toDoubleSparse(d));
      const GPU_STAGE3A::CsrMatrix minv_csr =
          GPU_STAGE3A::makeCsrFromEigen(toDoubleSparse(minv));
      const GPU_STAGE3A::CsrMatrix g_csr =
          GPU_STAGE3A::makeCsrFromEigen(toDoubleSparse(g));
      const GPU_STAGE3A::CsrMatrix hat_d_csr =
          GPU_STAGE3A::makeCsrFromEigen(toDoubleSparse(hat_d));
      const GPU_STAGE3A::CsrMatrix hat_minv_csr =
          GPU_STAGE3A::makeCsrFromEigen(toDoubleSparse(hat_minv));
      const GPU_STAGE3A::CsrMatrix hat_g_csr =
          GPU_STAGE3A::makeCsrFromEigen(toDoubleSparse(hat_g));
      d_dev = GPU_STAGE3A::makeDeviceCsrFromHost(d_csr);
      minv_dev = GPU_STAGE3A::makeDeviceCsrFromHost(minv_csr);
      g_dev = GPU_STAGE3A::makeDeviceCsrFromHost(g_csr);
      hat_d_dev = GPU_STAGE3A::makeDeviceCsrFromHost(hat_d_csr);
      hat_minv_dev = GPU_STAGE3A::makeDeviceCsrFromHost(hat_minv_csr);
      hat_g_dev = GPU_STAGE3A::makeDeviceCsrFromHost(hat_g_csr);
    }

    GPU_STAGE3A::DeviceCsrMatrix w_dev;
    GPU_STAGE3A::DeviceCsrMatrix hat_w_dev;
    const bool reuse_device_w = e1.gpu_w_valid && e2.gpu_w_valid;
    if (reuse_device_w) {
      ZIRAN_ASSERT(e1.gpu_w.rows == e2.gpu_w.rows &&
                       e1.gpu_w.cols == e1.num_v * dim &&
                       e2.gpu_w.cols == e2.num_v * dim,
                   "device interface W dimension mismatch");
      w_dev = e1.gpu_w;
      hat_w_dev = e2.gpu_w;
      ZIRAN_INFO("Stage3B reusing device-resident interface W blocks: rows=",
                 w_dev.rows, " fluid_nnz=", w_dev.nonzeros,
                 " solid_nnz=", hat_w_dev.nonzeros);
    } else {
      const GPU_STAGE3A::CsrMatrix w_csr =
          GPU_STAGE3A::makeCsrFromEigen(toDoubleSparse(w));
      const GPU_STAGE3A::CsrMatrix hat_w_csr =
          GPU_STAGE3A::makeCsrFromEigen(toDoubleSparse(hat_w));
      w_dev = GPU_STAGE3A::makeDeviceCsrFromHost(w_csr);
      hat_w_dev = GPU_STAGE3A::makeDeviceCsrFromHost(hat_w_csr);
    }
    const auto wt_dev = GPU_STAGE3A::transposeDeviceCsrMatrix(w_dev);
    const auto hat_wt_dev = GPU_STAGE3A::transposeDeviceCsrMatrix(hat_w_dev);

    auto e1_dminv_dev =
        buildGpuStage3BDeviceProduct("fluid_DMinv", d_dev, minv_dev);
    auto e1_dminvg_dev =
        buildGpuStage3BDeviceProduct("fluid_DMinvG", e1_dminv_dev, g_dev);
    auto e1_dminv_wt_dev =
        buildGpuStage3BDeviceProduct("fluid_DMinvWT", e1_dminv_dev, wt_dev);
    auto e1_w_minv_dev =
        buildGpuStage3BDeviceProduct("fluid_WMinv", w_dev, minv_dev);
    auto e1_w_minv_g_dev =
        buildGpuStage3BDeviceProduct("fluid_WMinvG", e1_w_minv_dev, g_dev);
    auto e1_w_minv_wt_dev =
        buildGpuStage3BDeviceProduct("fluid_WMinvWT", e1_w_minv_dev, wt_dev);

    auto e2_dminv_dev =
        buildGpuStage3BDeviceProduct("solid_DMinv", hat_d_dev, hat_minv_dev);
    auto e2_dminvg_dev =
        buildGpuStage3BDeviceProduct("solid_DMinvG", e2_dminv_dev, hat_g_dev);
    auto e2_dminv_wt_dev =
        buildGpuStage3BDeviceProduct("solid_DMinvWT", e2_dminv_dev, hat_wt_dev);
    auto e2_w_minv_dev =
        buildGpuStage3BDeviceProduct("solid_WMinv", hat_w_dev, hat_minv_dev);
    auto e2_w_minv_g_dev =
        buildGpuStage3BDeviceProduct("solid_WMinvG", e2_w_minv_dev, hat_g_dev);
    auto e2_w_minv_wt_dev =
        buildGpuStage3BDeviceProduct("solid_WMinvWT", e2_w_minv_dev, hat_wt_dev);

    const bool need_host_products = gpu_stage3b_probe || !reuse_device_assembly;
    if (need_host_products) {
      products.e1_dminv = GPU_STAGE3A::copyDeviceCsrToHost(e1_dminv_dev);
      products.e1_dminvg = GPU_STAGE3A::copyDeviceCsrToHost(e1_dminvg_dev);
      products.e1_dminv_wt =
          GPU_STAGE3A::copyDeviceCsrToHost(e1_dminv_wt_dev);
      products.e1_w_minv = GPU_STAGE3A::copyDeviceCsrToHost(e1_w_minv_dev);
      products.e1_w_minv_g =
          GPU_STAGE3A::copyDeviceCsrToHost(e1_w_minv_g_dev);
      products.e1_w_minv_wt =
          GPU_STAGE3A::copyDeviceCsrToHost(e1_w_minv_wt_dev);

      products.e2_dminv = GPU_STAGE3A::copyDeviceCsrToHost(e2_dminv_dev);
      products.e2_dminvg = GPU_STAGE3A::copyDeviceCsrToHost(e2_dminvg_dev);
      products.e2_dminv_wt =
          GPU_STAGE3A::copyDeviceCsrToHost(e2_dminv_wt_dev);
      products.e2_w_minv = GPU_STAGE3A::copyDeviceCsrToHost(e2_w_minv_dev);
      products.e2_w_minv_g =
          GPU_STAGE3A::copyDeviceCsrToHost(e2_w_minv_g_dev);
      products.e2_w_minv_wt =
          GPU_STAGE3A::copyDeviceCsrToHost(e2_w_minv_wt_dev);
    }

    products.e1_dminv_device = std::move(e1_dminv_dev);
    products.e1_wt_device = std::move(wt_dev);
    products.e1_dminvg_device = std::move(e1_dminvg_dev);
    products.e1_dminv_wt_device = std::move(e1_dminv_wt_dev);
    products.e1_w_minv_device = std::move(e1_w_minv_dev);
    products.e1_w_minv_g_device = std::move(e1_w_minv_g_dev);
    products.e1_w_minv_wt_device = std::move(e1_w_minv_wt_dev);
    products.e2_dminv_device = std::move(e2_dminv_dev);
    products.e2_wt_device = std::move(hat_wt_dev);
    products.e2_dminvg_device = std::move(e2_dminvg_dev);
    products.e2_dminv_wt_device = std::move(e2_dminv_wt_dev);
    products.e2_w_minv_device = std::move(e2_w_minv_dev);
    products.e2_w_minv_g_device = std::move(e2_w_minv_g_dev);
    products.e2_w_minv_wt_device = std::move(e2_w_minv_wt_dev);
    return products;
  }

  void fillInStage3BLeftTri(int offset, const Eigen::SparseMatrix<T> &s,
                            const GPU_STAGE3A::CsrMatrix &d_minv_g) {
    ZIRAN_TIMER();
    ZIRAN_ASSERT(s.rows() == d_minv_g.rows && s.cols() == d_minv_g.cols,
                 "Stage3B left block dimension mismatch");
    fillInCouplingTri(offset, offset, s);
    for (int row = 0; row < d_minv_g.rows; ++row) {
      for (int k = d_minv_g.row_offsets[static_cast<std::size_t>(row)];
           k < d_minv_g.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
        coupling_tri.emplace_back(
            offset + row,
            offset + d_minv_g.col_indices[static_cast<std::size_t>(k)],
            -static_cast<T>(d_minv_g.values[static_cast<std::size_t>(k)]));
      }
    }
  }

  void addEigenBlockToCsrRows(
      int row_offset, int col_offset, const Eigen::SparseMatrix<T> &matrix,
      std::vector<std::vector<std::pair<int, T>>> &rows) const {
    ZIRAN_TIMER();
    for (int k = 0; k < matrix.outerSize(); ++k) {
      for (typename Eigen::SparseMatrix<T>::InnerIterator it(matrix, k); it;
           ++it) {
        rows[static_cast<std::size_t>(row_offset + it.row())].emplace_back(
            col_offset + it.col(), it.value());
      }
    }
  }

  void addStage3BCsrBlockToRows(
      int row_offset, int col_offset, const GPU_STAGE3A::CsrMatrix &matrix,
      T scale, std::vector<std::vector<std::pair<int, T>>> &rows) const {
    ZIRAN_TIMER();
    for (int row = 0; row < matrix.rows; ++row) {
      auto &dst = rows[static_cast<std::size_t>(row_offset + row)];
      for (int k = matrix.row_offsets[static_cast<std::size_t>(row)];
           k < matrix.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
        dst.emplace_back(
            col_offset + matrix.col_indices[static_cast<std::size_t>(k)],
            scale * static_cast<T>(matrix.values[static_cast<std::size_t>(k)]));
      }
    }
  }

  TStack multiplyStage3BCsrVector(const GPU_STAGE3A::CsrMatrix &matrix,
                                  const TStack &x) const {
    ZIRAN_TIMER();
    ZIRAN_ASSERT(matrix.cols == x.rows(), "CSR SpMV vector size mismatch");
    TStack y = TStack::Zero(matrix.rows);
    for (int row = 0; row < matrix.rows; ++row) {
      T sum = 0;
      for (int k = matrix.row_offsets[static_cast<std::size_t>(row)];
           k < matrix.row_offsets[static_cast<std::size_t>(row + 1)]; ++k) {
        sum += static_cast<T>(matrix.values[static_cast<std::size_t>(k)]) *
               x(matrix.col_indices[static_cast<std::size_t>(k)]);
      }
      y(row) = sum;
    }
    return y;
  }

  TStack multiplyStage3BDeviceCsrVector(
      const GPU_STAGE3A::DeviceCsrMatrix &matrix, const TStack &x) const {
    ZIRAN_TIMER();
    ZIRAN_ASSERT(matrix.cols == x.rows(), "device CSR SpMV vector size mismatch");
    std::vector<double> x_host(static_cast<std::size_t>(x.rows()));
    for (int i = 0; i < x.rows(); ++i)
      x_host[static_cast<std::size_t>(i)] = static_cast<double>(x(i));
    std::vector<double> y_host;
    GPU_STAGE3A::runDeviceCsrSpmvProbe(matrix, x_host, y_host);
    TStack y = TStack::Zero(matrix.rows);
    for (int i = 0; i < matrix.rows; ++i)
      y(i) = static_cast<T>(y_host[static_cast<std::size_t>(i)]);
    return y;
  }

  HostCsrMatrix finalizeCsrRows(
      int rows_count, int cols_count,
      std::vector<std::vector<std::pair<int, T>>> &rows) const {
    ZIRAN_TIMER();
    HostCsrMatrix result;
    result.rows = rows_count;
    result.cols = cols_count;
    result.row_offsets.reserve(static_cast<std::size_t>(rows_count + 1));
    result.row_offsets.push_back(0);
    for (int row = 0; row < rows_count; ++row) {
      auto &entries = rows[static_cast<std::size_t>(row)];
      std::sort(entries.begin(), entries.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      int last_col = -1;
      T sum = 0;
      auto flush = [&]() {
        if (last_col >= 0 && sum != (T)0) {
          result.col_indices.push_back(last_col);
          result.values.push_back(sum);
        }
      };
      for (const auto &entry : entries) {
        if (entry.first != last_col) {
          flush();
          last_col = entry.first;
          sum = entry.second;
        } else {
          sum += entry.second;
        }
      }
      flush();
      result.row_offsets.push_back(static_cast<int>(result.values.size()));
    }
    return result;
  }

  void runGpuStage3AOperatorProbe(const Eigen::SparseMatrix<T> &coupling_m) {
    if constexpr (dim != 3) {
      return;
    } else {
      ZIRAN_TIMER();
      auto run = [&](const std::string &label,
                     const Eigen::SparseMatrix<T> &matrix, int seed) {
        logGpuStage3AStats(GPU_STAGE3A::runSparseMatrixProbe(
            label, toDoubleSparse(matrix), seed));
      };

      auto phase_prefix = [](const SplittingSimulation<T, dim> &sim) {
        return sim.category == MATERIAL_PHASE_FLUID ? std::string("fluid")
                                                    : std::string("solid");
      };
      const std::string e1_prefix = phase_prefix(e1);
      const std::string e2_prefix = phase_prefix(e2);

      run(e1_prefix + "_Minv", e1.M_inv, 11);
      run(e1_prefix + "_S", e1.S, 12);
      run(e1_prefix + "_G", e1.G, 13);
      run(e1_prefix + "_D", e1.D, 14);
      run(e2_prefix + "_Minv", e2.M_inv, 21);
      run(e2_prefix + "_S", e2.S, 22);
      run(e2_prefix + "_G", e2.G, 23);
      run(e2_prefix + "_D", e2.D, 24);

      run("W1", e1.W, 31);
      Eigen::SparseMatrix<T> W1T = e1.W.transpose();
      run("WT1", W1T, 32);
      run("W2", e2.W, 33);
      Eigen::SparseMatrix<T> W2T = e2.W.transpose();
      run("WT2", W2T, 34);
      run("coupling_block", coupling_m, 41);
    }
  }

  void runGpuStage3BProbe(const Eigen::SparseMatrix<T> &coupling_m,
                          const TStack &rhs, const TStack &cpu_solution) {
    if constexpr (dim != 3) {
      return;
    } else {
      ZIRAN_TIMER();
      auto to_double = [&](const Eigen::SparseMatrix<T> &matrix) {
        return toDoubleSparse(matrix);
      };
      auto gpu_product = [&](const Eigen::SparseMatrix<T> &left,
                             const Eigen::SparseMatrix<T> &middle,
                             const Eigen::SparseMatrix<T> &right) {
        const GPU_STAGE3A::CsrMatrix lm =
            GPU_STAGE3A::multiplyCsrMatrices(
                GPU_STAGE3A::makeCsrFromEigen(to_double(left)),
                GPU_STAGE3A::makeCsrFromEigen(to_double(middle)));
        return GPU_STAGE3A::multiplyCsrMatrices(
            lm, GPU_STAGE3A::makeCsrFromEigen(to_double(right)));
      };
      auto csr_spmv_host = [](const GPU_STAGE3A::CsrMatrix &matrix,
                              const std::vector<double> &x) {
        std::vector<double> y(static_cast<std::size_t>(matrix.rows), 0.0);
        for (int row = 0; row < matrix.rows; ++row) {
          double sum = 0.0;
          for (int k = matrix.row_offsets[row]; k < matrix.row_offsets[row + 1];
               ++k)
            sum += matrix.values[static_cast<std::size_t>(k)] *
                   x[static_cast<std::size_t>(
                       matrix.col_indices[static_cast<std::size_t>(k)])];
          y[static_cast<std::size_t>(row)] = sum;
        }
        return y;
      };
      auto log_vector_compare = [&](const std::string &label, int rows, int cols,
                                    std::size_t nonzeros,
                                    const std::vector<double> &reference_y,
                                    const std::vector<double> &probe_y) {
        GPU_STAGE3A::OperatorCompareStats stats;
        stats.label = label;
        stats.rows = rows;
        stats.cols = cols;
        stats.nonzeros = nonzeros;
        double sum_abs = 0.0;
        double reference_norm_inf = 0.0;
        for (std::size_t i = 0; i < reference_y.size(); ++i) {
          const double err = std::abs(reference_y[i] - probe_y[i]);
          stats.max_abs_error = std::max(stats.max_abs_error, err);
          reference_norm_inf =
              std::max(reference_norm_inf, std::abs(reference_y[i]));
          sum_abs += err;
        }
        stats.mean_abs_error =
            reference_y.empty() ? 0.0 : sum_abs / (double)reference_y.size();
        stats.max_rel_error =
            stats.max_abs_error / std::max(reference_norm_inf, 1e-30);
        logGpuStage3BAssemblyStats(stats);
      };
      auto compare_product = [&](const std::string &label,
                                 const GPU_STAGE3A::CsrMatrix &probe,
                                 const Eigen::SparseMatrix<double> &left,
                                 const Eigen::SparseMatrix<double> &middle,
                                 const Eigen::SparseMatrix<double> &right,
                                 int seed) {
        const std::vector<double> x = GPU_STAGE3A::makeDeterministicVector(
            probe.cols, seed);
        Eigen::Map<const Eigen::VectorXd> x_map(
            x.data(), static_cast<Eigen::Index>(x.size()));
        Eigen::VectorXd reference_y_eigen =
            left * (middle * (right * x_map));
        std::vector<double> reference_y(reference_y_eigen.data(),
                                        reference_y_eigen.data() +
                                            reference_y_eigen.size());
        log_vector_compare(label, probe.rows, probe.cols, probe.values.size(),
                           reference_y, csr_spmv_host(probe, x));
      };
      auto compare_left = [&](const std::string &label,
                              const GPU_STAGE3A::CsrMatrix &probe_dmg,
                              const Eigen::SparseMatrix<double> &s,
                              const Eigen::SparseMatrix<double> &d,
                              const Eigen::SparseMatrix<double> &minv,
                              const Eigen::SparseMatrix<double> &g, int seed) {
        const std::vector<double> x = GPU_STAGE3A::makeDeterministicVector(
            probe_dmg.cols, seed);
        Eigen::Map<const Eigen::VectorXd> x_map(
            x.data(), static_cast<Eigen::Index>(x.size()));
        Eigen::VectorXd sx = s * x_map;
        Eigen::VectorXd reference_y_eigen = sx - d * (minv * (g * x_map));
        std::vector<double> probe_dmg_x = csr_spmv_host(probe_dmg, x);
        std::vector<double> reference_y(reference_y_eigen.data(),
                                        reference_y_eigen.data() +
                                            reference_y_eigen.size());
        std::vector<double> probe_y(static_cast<std::size_t>(probe_dmg.rows));
        for (std::size_t i = 0; i < probe_y.size(); ++i)
          probe_y[i] = sx[static_cast<Eigen::Index>(i)] - probe_dmg_x[i];
        log_vector_compare(label, probe_dmg.rows, probe_dmg.cols,
                           probe_dmg.values.size(), reference_y, probe_y);
      };

      const Eigen::SparseMatrix<double> e1_d_double = toDoubleSparse(e1.D);
      const Eigen::SparseMatrix<double> e1_minv_double =
          toDoubleSparse(e1.M_inv);
      const Eigen::SparseMatrix<double> e1_g_double = toDoubleSparse(e1.G);
      const Eigen::SparseMatrix<double> e2_d_double = toDoubleSparse(e2.D);
      const Eigen::SparseMatrix<double> e2_minv_double =
          toDoubleSparse(e2.M_inv);
      const Eigen::SparseMatrix<double> e2_g_double = toDoubleSparse(e2.G);
      Eigen::SparseMatrix<double> e1_s_double = toDoubleSparse(e1.S);
      Eigen::SparseMatrix<double> e2_s_double = toDoubleSparse(e2.S);
      GPU_STAGE3A::CsrMatrix gpu_fluid_dmg =
          gpu_product(e1.D, e1.M_inv, e1.G);
      GPU_STAGE3A::CsrMatrix gpu_solid_dmg =
          gpu_product(e2.D, e2.M_inv, e2.G);
      compare_product("fluid_DMinvG", gpu_fluid_dmg, e1_d_double,
                      e1_minv_double, e1_g_double, 101);
      compare_product("solid_DMinvG", gpu_solid_dmg, e2_d_double,
                      e2_minv_double, e2_g_double, 102);
      compare_left("fluid_left", gpu_fluid_dmg, e1_s_double, e1_d_double,
                   e1_minv_double, e1_g_double, 103);
      compare_left("solid_left", gpu_solid_dmg, e2_s_double, e2_d_double,
                   e2_minv_double, e2_g_double, 104);

      if (interface_cnt == 0) {
        ZIRAN_INFO("Stage3B GPU assembly probe interface blocks skipped: "
                   "interface_cnt=0");
      } else {
        ZIRAN_WARN("Stage3B GPU assembly probe nonzero interface blocks are "
                   "not enabled in the CSR-only runtime probe yet");
      }

      (void)coupling_m;
      (void)rhs;
      (void)cpu_solution;
      ZIRAN_INFO("Stage3B GPU PCG probe skipped: assembled + AMGCL_CUDA is "
                 "now the stage 3B main path");
    }
  }

  void runGpuStage2Probe(SplittingSimulation<T, dim> &e_fluid) {
    if constexpr (dim != 3) {
      return;
    } else {
      ZIRAN_TIMER();
      std::vector<GPU_STAGE2::Vec3d> xq;
      std::vector<double> mass;
      xq.reserve(e_fluid.quadrature_state.volume_GQ.size());
      mass.reserve(e_fluid.quadrature_state.volume_mass.size());
      for (std::size_t i = 0; i < e_fluid.quadrature_state.volume_GQ.size(); ++i) {
        const TV &x = e_fluid.quadrature_state.volume_GQ[i];
        xq.push_back(GPU_STAGE2::Vec3d{static_cast<double>(x[0]),
            static_cast<double>(x[1]), static_cast<double>(x[2])});
        mass.push_back(static_cast<double>(e_fluid.quadrature_state.volume_mass[i]));
      }

      GPU_STAGE2::ProbeGridResult gpu_result;
      GPU_STAGE2::runVelocityGridMassProbe(xq, mass, static_cast<double>(e_fluid.dx),
          /*halfDxAnchor*/ true, gpu_result);

      auto make_count_reference = [](std::vector<GPU_STAGE2::GridKey> keys) {
        std::sort(keys.begin(), keys.end());
        std::vector<GPU_STAGE2::GridKey> unique_keys;
        std::vector<double> counts;
        for (GPU_STAGE2::GridKey key : keys) {
          if (unique_keys.empty() || unique_keys.back() != key) {
            unique_keys.push_back(key);
            counts.push_back(1.0);
          } else {
            counts.back() += 1.0;
          }
        }
        return std::make_pair(std::move(unique_keys), std::move(counts));
      };

      auto order0_key = [](const GPU_STAGE2::Vec3d &x, double dx) {
        return GPU_STAGE2::packGridKey3D(
            static_cast<int>(std::floor((x.x - 0.5 * dx) / dx)),
            static_cast<int>(std::floor((x.y - 0.5 * dx) / dx)),
            static_cast<int>(std::floor((x.z - 0.5 * dx) / dx)));
      };

      auto log_stats = [](const char *label, const GPU_STAGE2::ProbeCompareStats &stats) {
        ZIRAN_INFO("Stage2 GPU sparse grid probe ", label, ": cpu_nodes=", stats.cpu_nodes,
            " gpu_nodes=", stats.gpu_nodes,
            " missing_on_gpu=", stats.missing_on_gpu,
            " extra_on_gpu=", stats.extra_on_gpu,
            " max_abs_mass_error=", stats.max_abs_mass_error,
            " mean_abs_mass_error=", stats.mean_abs_mass_error);
      };

      std::vector<std::pair<GPU_STAGE2::GridKey, double>> cpu_pairs;
      e_fluid.grid_comm.iterateGridSerial([&](IV node, GridState<T, dim> &g) {
        cpu_pairs.emplace_back(
            GPU_STAGE2::packGridKey3D(node[0], node[1], node[2]),
            static_cast<double>(g.m));
      });
      std::sort(cpu_pairs.begin(), cpu_pairs.end(),
          [](const auto &a, const auto &b) { return a.first < b.first; });

      std::vector<GPU_STAGE2::GridKey> cpu_keys;
      std::vector<double> cpu_mass;
      cpu_keys.reserve(cpu_pairs.size());
      cpu_mass.reserve(cpu_pairs.size());
      for (const auto &entry : cpu_pairs) {
        cpu_keys.push_back(entry.first);
        cpu_mass.push_back(entry.second);
      }

      const GPU_STAGE2::ProbeCompareStats stats =
          GPU_STAGE2::compareMassByKey(cpu_keys, cpu_mass, gpu_result);
      log_stats("velocity", stats);

      GPU_STAGE2::ProbeGridResult pressure_result;
      GPU_STAGE2::runPressureGridCountProbe(xq, static_cast<double>(e_fluid.dx),
          /*halfDxAnchor*/ true, pressure_result);
      std::vector<GPU_STAGE2::GridKey> pressure_sample_keys;
      pressure_sample_keys.reserve(xq.size());
      for (const GPU_STAGE2::Vec3d &x : xq)
        pressure_sample_keys.push_back(order0_key(x, static_cast<double>(e_fluid.dx)));
      auto pressure_reference = make_count_reference(std::move(pressure_sample_keys));
      const GPU_STAGE2::ProbeCompareStats pressure_stats =
          GPU_STAGE2::compareMassByKey(pressure_reference.first, pressure_reference.second, pressure_result);
      log_stats("pressure", pressure_stats);
      ZIRAN_INFO("Stage2 GPU sparse grid probe pressure expected_nodes=",
          e_fluid.num_p - e_fluid.quadrature_state.total_id,
          " probed_nodes=", pressure_stats.gpu_nodes);

      std::vector<GPU_STAGE2::Vec3d> face_xq;
      std::vector<GPU_STAGE2::Vec3d> face_n;
      face_xq.reserve(e_fluid.quadrature_state.face_GQ.size());
      face_n.reserve(e_fluid.quadrature_state.face_N.size());
      for (std::size_t i = 0; i < e_fluid.quadrature_state.face_GQ.size(); ++i) {
        const TV &x = e_fluid.quadrature_state.face_GQ[i];
        const TV &n = e_fluid.quadrature_state.face_N[i];
        face_xq.push_back(GPU_STAGE2::Vec3d{static_cast<double>(x[0]),
            static_cast<double>(x[1]), static_cast<double>(x[2])});
        face_n.push_back(GPU_STAGE2::Vec3d{static_cast<double>(n[0]),
            static_cast<double>(n[1]), static_cast<double>(n[2])});
      }

      GPU_STAGE2::ProbeGridResult face_result;
      GPU_STAGE2::runFaceGridCountProbe(face_xq, face_n, static_cast<double>(e_fluid.dx),
          /*halfDxAnchor*/ true, face_result);
      std::vector<GPU_STAGE2::GridKey> face_sample_keys;
      face_sample_keys.reserve(face_xq.size() * 2);
      for (std::size_t i = 0; i < face_xq.size(); ++i) {
        for (int delta = -1; delta <= 1; delta += 2) {
          const double dx = static_cast<double>(e_fluid.dx);
          GPU_STAGE2::Vec3d sample{
              face_xq[i].x + face_n[i].x * 0.5 * dx * static_cast<double>(delta),
              face_xq[i].y + face_n[i].y * 0.5 * dx * static_cast<double>(delta),
              face_xq[i].z + face_n[i].z * 0.5 * dx * static_cast<double>(delta)};
          face_sample_keys.push_back(order0_key(sample, dx));
        }
      }
      auto face_reference = make_count_reference(std::move(face_sample_keys));
      const GPU_STAGE2::ProbeCompareStats face_stats =
          GPU_STAGE2::compareMassByKey(face_reference.first, face_reference.second, face_result);
      log_stats("face", face_stats);
      ZIRAN_INFO("Stage2 GPU sparse grid probe face expected_nodes=",
          e_fluid.quadrature_state.total_id,
          " probed_nodes=", face_stats.gpu_nodes);
    }
  }

  void buildSystem() {
    ZIRAN_TIMER();
    SplittingSimulation<T, dim> &e_solid =
        (e1.category == MATERIAL_PHASE_SOLID) ? e1 : e2;
    SplittingSimulation<T, dim> &e_fluid =
        (e1.category != MATERIAL_PHASE_SOLID) ? e1 : e2;
    e_fluid.solid_scheme = e_solid.config_scheme;

    GPU_STAGE3A::SolidParticleStateDeviceView solid_device;
    bool device_only_interface_fields = false;
    if constexpr (dim == 3) {
      solid_device = GPU_STAGE3A::getPersistentSolidParticleStateDeviceView(
          e_solid.gpu_solid_particle_state);
      const bool gpu_analytic_geometry_supported =
          e_fluid.gpu_box_shell_collision.valid ||
          e_fluid.collision_objects.empty();
      device_only_interface_fields = gpu_build_system &&
          e_fluid.solid_scheme != CONFIG_UPDATE_LAGRANGIAN &&
          solid_device.current_x != nullptr &&
          solid_device.rest_x != nullptr && solid_device.volume != nullptr &&
          solid_device.size ==
              static_cast<std::size_t>(e_solid.particles.count) &&
          e_fluid.fluid_Q1Q0 && gpu_analytic_geometry_supported &&
          !gpu_build_system_cpu_numerical_grid &&
          !gpu_build_system_cpu_velocity_grid &&
          std::getenv("ZIRAN_GPU_DISABLE_FLUID_ANALYTICAL") == nullptr &&
          std::getenv("ZIRAN_GPU_FLUID_VELOCITY_GRID_ORACLE") == nullptr &&
          std::getenv("ZIRAN_GPU_INTERFACE_QUAD_HOST_RESULT") == nullptr &&
          std::getenv("ZIRAN_GPU_INTERFACE_FIELDS_COMPARE") == nullptr &&
          std::getenv("ZIRAN_INTERFACE_QUAD_FIELD_DUMP_PREFIX") == nullptr &&
          std::getenv("ZIRAN_SOLID_NORMAL_FIELD_DUMP_PREFIX") == nullptr;
    }
    if (!device_only_interface_fields) {
      // A compatibility consumer of current X must never observe the stale
      // host mirror left behind by resident solid G2P.
      e_solid.synchronizeSolidCurrentXForHostContact();
      ZIRAN_NAMED_TIMER("couplingSolidStateCopy");
      e_fluid.solid_positions = e_solid.particles.X.array;
      e_fluid.solid_cur_positions =
          e_solid.particles.DataManager::get(X0_name<TV>())
              .array; // 在TOTAL LAGRANGIAN时是current的
    } else {
      e_fluid.solid_positions.clear();
      e_fluid.solid_cur_positions.clear();
      ZIRAN_INFO("Coupling solid interface fields remain device-resident: "
                 "particles=", solid_device.size);
    }

    bool gpu_no_interface_neighbors = false;
    bool gpu_sph_normals_ready = false;
    e_fluid.gpu_skip_interface_neighbor_search = false;
    e_fluid.gpu_interface_solid_particles = {};
    if (gpu_build_system) {
      if constexpr (dim == 3) {
        const bool resident_solid_positions =
            e_fluid.solid_scheme != CONFIG_UPDATE_LAGRANGIAN &&
            solid_device.current_x != nullptr &&
            solid_device.size ==
                static_cast<std::size_t>(e_solid.particles.count);
        if (device_only_interface_fields)
          e_fluid.gpu_interface_solid_particles = solid_device;
        std::vector<GPU_STAGE2::Vec3d> solid_positions;
        if (!resident_solid_positions) {
          const StdVector<TV> &search_positions =
              e_fluid.solid_scheme != CONFIG_UPDATE_LAGRANGIAN
                  ? e_fluid.solid_cur_positions
                  : e_fluid.solid_positions;
          solid_positions.resize(search_positions.size());
          ZIRAN_NAMED_TIMER("gpuCoordinatorSolidPack");
          tbb::parallel_for(std::size_t{0}, search_positions.size(),
              [&](std::size_t i) {
                const TV& x = search_positions[i];
                solid_positions[i] = GPU_STAGE2::Vec3d{
                    static_cast<double>(x[0]), static_cast<double>(x[1]),
                    static_cast<double>(x[2])};
              });
        }
        GPU_STAGE2::SpatialSearchStats build_stats;
        {
          ZIRAN_NAMED_TIMER("gpuCoordinatorSolidSearch");
          GPU_STAGE2::rebuildPersistentSpatialSearch(
              gpu_solid_search, solid_positions,
              static_cast<double>(e_solid.dx * (T)2), &build_stats,
              resident_solid_positions ? solid_device.current_x : nullptr,
              resident_solid_positions ? solid_device.size : 0);
        }

        const bool resident_fluid_positions =
            e_fluid.gpu_velocity_transfer_device_x_valid &&
            e_fluid.gpu_velocity_transfer_device.x != nullptr &&
            e_fluid.gpu_velocity_transfer_device.size ==
                static_cast<std::size_t>(e_fluid.particles.count);
        std::vector<GPU_STAGE2::Vec3d> fluid_positions;
        if (!resident_fluid_positions) {
          fluid_positions.resize(
              static_cast<std::size_t>(e_fluid.particles.count));
          ZIRAN_NAMED_TIMER("gpuCoordinatorFluidPack");
          tbb::parallel_for(std::size_t{0}, fluid_positions.size(),
              [&](std::size_t i) {
                const TV& x = e_fluid.particles.X.array[i];
                fluid_positions[i] = GPU_STAGE2::Vec3d{
                    static_cast<double>(x[0]), static_cast<double>(x[1]),
                    static_cast<double>(x[2])};
              });
        }
        GPU_STAGE2::SpatialSearchStats query_stats;
        bool has_geometric_interface_neighbor;
        {
          ZIRAN_NAMED_TIMER("gpuCoordinatorInterfaceProbe");
          has_geometric_interface_neighbor = GPU_STAGE2::anyPointWithinRadius(
              gpu_solid_search, fluid_positions,
              static_cast<double>(e_fluid.dx * (T)0.5), &query_stats,
              resident_fluid_positions
                  ? e_fluid.gpu_velocity_transfer_device.x : nullptr,
              resident_fluid_positions
                  ? e_fluid.gpu_velocity_transfer_device.size : 0);
        }
        gpu_no_interface_neighbors = !has_geometric_interface_neighbor;
        e_fluid.gpu_skip_interface_neighbor_search =
            gpu_no_interface_neighbors;

        if (has_geometric_interface_neighbor &&
            e_fluid.solid_scheme != CONFIG_UPDATE_LAGRANGIAN) {
          const bool resident_solid_mass =
              resident_solid_positions && solid_device.mass != nullptr;
          std::vector<double> masses;
          if (!resident_solid_mass) {
            masses.resize(e_solid.particles.count);
            ZIRAN_NAMED_TIMER("gpuCoordinatorSolidMassPack");
            tbb::parallel_for(std::size_t{0}, masses.size(),
                [&](std::size_t i) {
                  masses[i] = static_cast<double>(e_solid.particles.mass.array[i]);
                });
          }
          std::vector<GPU_STAGE2::Vec3d> gpu_normals;
          GPU_STAGE2::SpatialSearchStats normal_stats;
          {
            ZIRAN_NAMED_TIMER("gpuCoordinatorSphNormals");
            GPU_STAGE2::computeCubicSphNormals(
                gpu_solid_search, masses,
                static_cast<double>(e_solid.dx * (T)2), gpu_normals,
                &normal_stats,
                resident_solid_mass ? solid_device.mass : nullptr,
                resident_solid_mass ? solid_device.size : 0,
                /*materialize_host_normals*/ !device_only_interface_fields);
          }
          if (!device_only_interface_fields) {
            e_fluid.solid_normals.resize(gpu_normals.size());
            ZIRAN_NAMED_TIMER("gpuCoordinatorNormalMaterialize");
            tbb::parallel_for(std::size_t{0}, gpu_normals.size(),
                [&](std::size_t i) {
                  e_fluid.solid_normals[i] =
                      TV(static_cast<T>(gpu_normals[i].x),
                         static_cast<T>(gpu_normals[i].y),
                         static_cast<T>(gpu_normals[i].z));
                });
          }
          gpu_sph_normals_ready = true;
          if (!device_only_interface_fields && resident_solid_positions &&
              solid_device.rest_x != nullptr && solid_device.volume != nullptr)
            e_fluid.gpu_interface_solid_particles = solid_device;
          ZIRAN_INFO("GPU buildSystem solid search: particles=",
                     build_stats.particles,
                     " capacity=", build_stats.particle_capacity,
                     " reallocations=", normal_stats.reallocations,
                     " resident_solid_positions=",
                     build_stats.resident_particle_positions ? 1 : 0,
                     " resident_solid_mass=",
                     normal_stats.resident_particle_mass ? 1 : 0,
                     " upload_ms=", build_stats.upload_ms,
                     " sort_ms=", build_stats.sort_ms,
                     " interface_query_ms=", query_stats.query_ms,
                     " resident_fluid_queries=",
                     query_stats.resident_query_positions ? 1 : 0,
                     " sph_normal_ms=", normal_stats.query_ms,
                     " has_interface_candidate=1");
        } else {
          ZIRAN_INFO("GPU buildSystem solid search: particles=",
                     build_stats.particles,
                     " capacity=", build_stats.particle_capacity,
                     " reallocations=", query_stats.reallocations,
                     " resident_solid_positions=",
                     build_stats.resident_particle_positions ? 1 : 0,
                     " upload_ms=", build_stats.upload_ms,
                     " sort_ms=", build_stats.sort_ms,
                     " interface_query_ms=", query_stats.query_ms,
                     " resident_fluid_queries=",
                     query_stats.resident_query_positions ? 1 : 0,
                     " has_interface_candidate=",
                     has_geometric_interface_neighbor ? 1 : 0);
        }
      }
    }

    if (!device_only_interface_fields) {
      e_fluid.solid_normals.resize(e_solid.particles.count);
      e_fluid.solid_lens.resize(e_solid.particles.count);
      auto *vol_pointer =
          e_solid.particles.DataManager::getPointer(element_measure_name<T>());
      {
        ZIRAN_NAMED_TIMER("couplingSolidInterfaceFields");
        tbb::parallel_for(0, (int)e_solid.particles.count, [&](int i) {
          TV normal = TV::Zero();
          T vol = (*vol_pointer)[i];
          T len = (T)2 * std::sqrt(std::max(vol, (T)0) / M_PI);
          e_fluid.solid_lens[i] = len;
          if (gpu_no_interface_neighbors || gpu_sph_normals_ready) {
            if (gpu_no_interface_neighbors)
              e_fluid.solid_normals[i] = TV::Zero();
            return;
          }
          // printf("len %f\n",len);
          BSplineWeights<T, dim> spline(
              e_solid.particles.X.array[i], e_solid.dx);
          e_solid.grid.iterateKernel(
              spline, e_solid.particle_base_offset[i],
              [&](IV node, T w, TV dw, GridState<T, dim> &g) {
                normal += g.m * dw;
              });
          e_fluid.solid_normals[i] = normal.normalized();
        });
      }
    } else {
      e_fluid.solid_normals.clear();
      e_fluid.solid_lens.clear();
    }

    if constexpr (dim == 3) {
      if (const char* dump_prefix =
              std::getenv("ZIRAN_SOLID_NORMAL_FIELD_DUMP_PREFIX")) {
        // Particle ordering is fixed by the restart data. This dump compares
        // the CPU and GPU implementations of the same SPH-normal stage
        // before any interface quadrature or linear system is assembled.
        static int dump_index = 0;
        const std::string path = std::string(dump_prefix) + "_" +
            std::to_string(dump_index++) + ".bin";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        const std::uint64_t count = e_fluid.solid_normals.size();
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const TV& normal : e_fluid.solid_normals) {
          const double x = static_cast<double>(normal[0]);
          const double y = static_cast<double>(normal[1]);
          const double z = static_cast<double>(normal[2]);
          out.write(reinterpret_cast<const char*>(&x), sizeof(x));
          out.write(reinterpret_cast<const char*>(&y), sizeof(y));
          out.write(reinterpret_cast<const char*>(&z), sizeof(z));
        }
        ZIRAN_ASSERT(out.good(), "failed to write solid-normal diagnostic dump");
        ZIRAN_INFO("Solid normal-field dump: ", path,
                   " particles=", count);
      }
    }

    { // compute normal
      auto &marray = e_solid.particles.mass.array;
      if (e_fluid.solid_scheme != CONFIG_UPDATE_LAGRANGIAN &&
          !gpu_no_interface_neighbors && !gpu_sph_normals_ready) {
        // T radius = e_solid.dx * 1;
        T radius = e_solid.dx * 2;
        SpatialHash<T, dim> spatial_hash;
        spatial_hash.rebuild(radius, e_fluid.solid_cur_positions);
        e_solid.parallel_for_updating_grid([&](int i) {
          TV Xp = e_fluid.solid_cur_positions[i];
          T mass = marray[i];
          TV normal = TV::Zero();
          TV d_D = TV::Zero();
          T D = 0.0;
          StdVector<int> unfiltered_neighbors;
          StdVector<int> neighbors;
          spatial_hash.oneLayerNeighbors(Xp, unfiltered_neighbors);
          for (auto &idx : unfiltered_neighbors) {
            TV delta_pos = Xp - e_fluid.solid_cur_positions[idx];
            if (delta_pos.norm() < radius && delta_pos.norm() > 1e-8) {
              T d = mass;
              TV delta_pos_normalize = delta_pos.normalized();
              T weight = ZIRAN::SPH_KERNELS::cubicBSplineKernelQ(
                  delta_pos.norm() / radius * 2);
              TV d_weight = ZIRAN::SPH_KERNELS::cubicBSplineKernelGradientQ(
                                delta_pos.norm() / radius * 2) *
                            delta_pos_normalize / radius;
              D += d * weight;
              d_D += d * d_weight;
            }
          }
          normal = d_D;
          // normal = d_D / S - D * d_S / (S * S);
          e_fluid.solid_normals[i] = normal.normalized();
        });

        // 1: g2p
        // MpmGrid<T, dim> grid_normal;
        // auto& marray = e_solid.particles.mass.array;
        // grid_normal.activatePages(e_fluid.solid_cur_positions, e_solid.dx);
        // // grid_normal.activatePages(e_fluid.solid_cur_positions, Base::dx, 0, 1,
        // -TV::Ones() * Base::dx * 0.5); tbb::parallel_for(0,
        // (int)e_solid.particles.count, [&](int i) {
        //   TV Xp = e_fluid.solid_cur_positions[i];
        //   T mass = marray[i];
        //   BSplineWeights<T, dim> spline_n(Xp, e_solid.dx);
        //   // BSplineWeights<T, dim, order> spline(Xp - TV::Ones() * Base::dx
        //   * 0.5, Base::dx);
        //   // uint64_t offset =
        //   Base::SparseMask::Linear_Offset(to_std_array(spline_n.base_node));
        //   grid_normal.iterateKernel(spline_n,
        //   e_solid.particle_base_offset[i],
        //                       [&](const IV &node, T w_n, const TV &dw_n,
        //                           GridState<T, dim> &g_n) { g_n.m += mass *
        //                           w_n; });
        // });
        // // 2: compute
        // tbb::parallel_for(0, (int)e_solid.particles.count, [&](int i) {
        //   TV normal = TV::Zero();
        //   TV Xp = e_fluid.solid_cur_positions[i];
        //   BSplineWeights<T, dim> spline_n(Xp, e_solid.dx);
        //   grid_normal.iterateKernel(spline_n,
        //   e_solid.particle_base_offset[i],
        //                       [&](const IV &node, T w_n, const TV &dw_n,
        //                           GridState<T, dim> &g_n) { normal += g_n.m *
        //                           dw_n; });
        //   e_fluid.solid_normals[i] = normal.normalized();
        // });
      } else {
        // T radius = e_solid.dx * 2;
        // SpatialHash<T, dim> spatial_hash;
        // spatial_hash.rebuild(radius, e_fluid.solid_positions);
        // e_solid.parallel_for_updating_grid([&](int i) {
        //   TV Xp = e_fluid.solid_positions[i];
        //   T mass = marray[i];
        //   TV normal = TV::Zero();
        //   TV d_D = TV::Zero();
        //   T D = 0.0;
        //   StdVector<int> unfiltered_neighbors;
        //   StdVector<int> neighbors;
        //   spatial_hash.oneLayerNeighbors(Xp, unfiltered_neighbors);
        //   for (auto &idx : unfiltered_neighbors){
        //     TV delta_pos = Xp - e_fluid.solid_positions[idx];
        //     if (delta_pos.norm() < radius && delta_pos.norm() > 1e-8){
        //       T d = mass;
        //       TV delta_pos_normalize = delta_pos.normalized();
        //       T weight =
        //       ZIRAN::SPH_KERNELS::cubicBSplineKernelQ(delta_pos.norm()/radius
        //       * 2); TV d_weight =
        //       ZIRAN::SPH_KERNELS::cubicBSplineKernelGradientQ(delta_pos.norm()/radius
        //       * 2) * delta_pos_normalize / radius; D += d * weight; d_D += d
        //       * d_weight;
        //     }
        //   }
        //   normal = d_D;
        //   // normal = d_D / S - D * d_S / (S * S);
        //   e_fluid.solid_normals[i] = normal.normalized();
        // });

        // MpmGrid<T, dim> grid_normal;
        // grid_normal.activatePages(e_fluid.solid_positions, e_solid.dx);
        // // grid_normal.activatePages(e_fluid.solid_cur_positions, Base::dx, 0, 1,
        // -TV::Ones() * Base::dx * 0.5); tbb::parallel_for(0,
        // (int)e_solid.particles.count, [&](int i) {
        //   TV Xp = e_fluid.solid_positions[i];
        //   T mass = marray[i];
        //   BSplineWeights<T, dim> spline_n(Xp, e_solid.dx);
        //   grid_normal.iterateKernel(spline_n,
        //   e_solid.particle_base_offset[i],
        //                       [&](const IV &node, T w_n, const TV &dw_n,
        //                           GridState<T, dim> &g_n) { g_n.m += mass *
        //                           w_n; });
        // });
        // // 2: compute
        // tbb::parallel_for(0, (int)e_solid.particles.count, [&](int i) {
        //   TV normal = TV::Zero();
        //   TV Xp = e_fluid.solid_positions[i];
        //   BSplineWeights<T, dim> spline_n(Xp, e_solid.dx);
        //   grid_normal.iterateKernel(spline_n,
        //   e_solid.particle_base_offset[i],
        //                       [&](const IV &node, T w_n, const TV &dw_n,
        //                           GridState<T, dim> &g_n) { normal += g_n.m *
        //                           dw_n; });
        //   e_fluid.solid_normals[i] = normal.normalized();
        // });
      }
    }

    const bool use_gpu_block_assembly =
        gpu_stage3b_writeback || gpu_stage3b_probe;
    e_fluid.gpu_build_system = gpu_build_system;
    e_fluid.gpu_full_pipeline_strict = gpu_full_pipeline_strict;
    e_fluid.gpu_fluid_device_resident = gpu_full_pipeline &&
        std::getenv("ZIRAN_GPU_FLUID_RESIDENT_DISABLE") == nullptr;
    e_fluid.gpu_build_system_cpu_numerical_grid =
        gpu_build_system_cpu_numerical_grid;
    e_fluid.gpu_build_system_cpu_pressure_grid =
        gpu_build_system_cpu_pressure_grid;
    e_fluid.gpu_build_system_cpu_velocity_grid =
        gpu_build_system_cpu_velocity_grid;
    // Reuse the persistent solid cell list for interface quadrature on GPU.
    // The CUDA predicate supports both the existing box shell and an empty
    // collision list (open domain).  Other nonempty geometry lists remain on
    // the CPU until they have an equivalent device predicate.
    e_fluid.gpu_interface_quad = gpu_build_system &&
        (e_fluid.gpu_box_shell_collision.valid ||
         e_fluid.collision_objects.empty());
    // GPU analytic/velocity/Stage3B/AMGX all consume keyed node arrays.  Use
    // their native sorted GridKey order as one canonical fluid DOF space;
    // SPGrid page-first numbering remains only in the CPU fallback path.
    // Keep this as an explicit A/B path until the AMGX convergence policy is
    // made permutation-robust.  The keyed operators are already consistent,
    // but a fixed-iteration AMG hierarchy can follow a different numerical
    // trajectory after a symmetric DOF permutation.
    e_fluid.gpu_fluid_canonical_dof_order = gpu_build_system &&
        std::getenv("ZIRAN_GPU_FLUID_CANONICAL_DOF_ORDER") != nullptr;
    e_fluid.gpu_interface_solid_search = &gpu_solid_search;
    e_solid.gpu_build_system = gpu_build_system;
    e_solid.gpu_full_pipeline_strict = gpu_full_pipeline_strict;
    e_solid.gpu_solid_implicit_probe = gpu_solid_implicit_probe;
    e_solid.gpu_solid_implicit_pcg = gpu_solid_implicit_pcg;
    e_solid.gpu_solid_implicit_state = gpu_solid_implicit_state;
    // CouplingSimulation is the final owner of per-substep flags. Enable the
    // verified Q1 transfer whenever the persistent GPU buildSystem pipeline
    // is active; retain the explicit field for isolated A/B runs.
    e_fluid.gpu_fluid_velocity_transfer =
        gpu_fluid_velocity_transfer || gpu_build_system;
    e_fluid.gpu_stage3b_assemble_blocks = use_gpu_block_assembly;
    e_solid.gpu_stage3b_assemble_blocks = use_gpu_block_assembly;
    e_fluid.gpu_stage3b_validate_blocks = gpu_stage3b_probe;
    e_solid.gpu_stage3b_validate_blocks = gpu_stage3b_probe;
    e_fluid.buildSystem();
    if constexpr (dim == 3) {
      if (const char* dump_prefix =
              std::getenv("ZIRAN_INTERFACE_QUAD_FIELD_DUMP_PREFIX")) {
        // Entries are generated in increasing solid-particle order. Store
        // their physical fields, rather than any sparse-grid DOF, to identify
        // a first buildSystem divergence before solveSystem is involved.
        ZIRAN_ASSERT(e_fluid.interface_rest_GQ.size() ==
                         e_fluid.interface_GQ.size() &&
                         e_fluid.interface_GQ.size() ==
                         e_fluid.interface_N.size() &&
                         e_fluid.interface_N.size() ==
                         e_fluid.interface_L.size(),
                     "interface quadrature field sizes disagree");
        static int dump_index = 0;
        const std::string path = std::string(dump_prefix) + "_" +
            std::to_string(dump_index++) + ".bin";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        const std::uint64_t count = e_fluid.interface_GQ.size();
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (std::size_t i = 0; i < e_fluid.interface_GQ.size(); ++i) {
          const TV& rest = e_fluid.interface_rest_GQ[i];
          const TV& current = e_fluid.interface_GQ[i];
          const TV& normal = e_fluid.interface_N[i];
          const double length = static_cast<double>(e_fluid.interface_L[i]);
          for (int d = 0; d < 3; ++d) {
            const double value = static_cast<double>(rest[d]);
            out.write(reinterpret_cast<const char*>(&value), sizeof(value));
          }
          for (int d = 0; d < 3; ++d) {
            const double value = static_cast<double>(current[d]);
            out.write(reinterpret_cast<const char*>(&value), sizeof(value));
          }
          for (int d = 0; d < 3; ++d) {
            const double value = static_cast<double>(normal[d]);
            out.write(reinterpret_cast<const char*>(&value), sizeof(value));
          }
          out.write(reinterpret_cast<const char*>(&length), sizeof(length));
        }
        ZIRAN_ASSERT(out.good(),
                     "failed to write interface-quadrature diagnostic dump");
        ZIRAN_INFO("Interface quadrature-field dump: ", path,
                   " entries=", count);
      }
    }
    if (gpu_stage2_probe)
      runGpuStage2Probe(e_fluid);
    e_solid.interface_rest_GQ = e_fluid.interface_rest_GQ;
    e_solid.interface_GQ = e_fluid.interface_GQ;
    e_solid.interface_N = e_fluid.interface_N;
    e_solid.interface_L = e_fluid.interface_L;
    e_solid.gpu_interface_quad_context = e_fluid.gpu_interface_quad_context;
    e_solid.gpu_interface_fields = e_fluid.gpu_interface_fields;
    e_solid.gpu_interface_quad_device_ready =
        e_fluid.gpu_interface_quad_device_ready;
    e_solid.gpu_interface_quad_host_fields_valid =
        e_fluid.gpu_interface_quad_host_fields_valid;
    e_solid.buildSystem();
    if (gpu_full_pipeline_strict) {
      ZIRAN_ASSERT(device_only_interface_fields,
          "strict full-GPU coupling materialized solid interface fields");
      ZIRAN_ASSERT(e_fluid.gpu_interface_quad_device_ready &&
                       !e_fluid.gpu_interface_quad_host_fields_valid,
          "strict full-GPU interface quadrature fell back to host fields");
      ZIRAN_ASSERT(e_fluid.gpu_prebuilt_fluid_dofs_valid &&
                       e_fluid.gpu_prebuilt_velocity_valid &&
                       e_fluid.gpu_device_assembly_valid &&
                       e_fluid.gpu_w_valid,
          "strict full-GPU fluid buildSystem used a CPU numerical branch");
      ZIRAN_ASSERT(e_solid.gpu_prebuilt_solid_pressure_valid &&
                       e_solid.gpu_device_assembly_valid &&
                       e_solid.gpu_w_valid,
          "strict full-GPU solid buildSystem used a CPU numerical branch");
      ZIRAN_INFO("Strict full-GPU buildSystem verified: interface_fields=device",
                 " fluid_blocks=device solid_blocks=device W=device");
    }
    interface_cnt = e1.W.rows();
  }

  std::vector<Eigen::Triplet<T>> coupling_tri;
  void fillInCouplingTri(int offset0, int offset1,
                         const Eigen::SparseMatrix<T> &mat) {
    ZIRAN_TIMER();
    for (int k = 0; k < mat.outerSize(); ++k)
      for (typename Eigen::SparseMatrix<T>::InnerIterator it(mat, k); it; ++it)
        coupling_tri.emplace_back(offset0 + it.row(), offset1 + it.col(),
                                  it.value());
  }

  TStack coupling_rhs;
  void fillInCouplingRhs(int offset, const TStack &rhs) {
    ZIRAN_TIMER();
    for (int i = 0; i < rhs.rows(); ++i)
      coupling_rhs(offset + i) = rhs(i);
  }

  void solveSystem() {
    ZIRAN_TIMER();
    Eigen::SparseMatrix<T> &Minv = e1.M_inv;
    Eigen::SparseMatrix<T> &G = e1.G;
    Eigen::SparseMatrix<T> &D = e1.D;
    Eigen::SparseMatrix<T> &S = e1.S;
    Eigen::SparseMatrix<T> &W1 = e1.W;
    Eigen::SparseMatrix<T> &hat_Minv = e2.M_inv;
    Eigen::SparseMatrix<T> &hat_G = e2.G;
    Eigen::SparseMatrix<T> &hat_D = e2.D;
    Eigen::SparseMatrix<T> &hat_S = e2.S;
    Eigen::SparseMatrix<T> &W2 = e2.W;
    TStack &a = e1.a;
    TStack &b = e1.b;
    TStack &hat_a = e2.a;
    TStack &hat_b = e2.b;
    // build system
    int num_p1 = e1.num_p;
    int num_p2 = e2.num_p;
    bool use_stage3b_writeback = false;
    Stage3BCsrProducts stage3b_products;
    if (gpu_stage3b_writeback) {
      if constexpr (dim == 3) {
        use_stage3b_writeback = true;
        stage3b_products =
            buildGpuStage3BProducts(D, Minv, G, W1, hat_D, hat_Minv, hat_G,
                                    W2);
      } else {
        ZIRAN_WARN("Stage3B GPU writeback is only enabled for 3D coupling");
      }
    }

    auto assemble_coupling_matrix = [&](bool force_cpu_left_blocks) {
      ZIRAN_TIMER();
      coupling_tri.clear();
      if (use_stage3b_writeback && !force_cpu_left_blocks)
        fillInStage3BLeftTri(0, S, stage3b_products.e1_dminvg);
      else
        fillInCouplingTri(0, 0, S - D * Minv * G);
      fillInCouplingTri(0, num_p1, -D * Minv * W1.transpose());
      fillInCouplingTri(num_p1, 0, W1 * Minv * G);
      fillInCouplingTri(num_p1, num_p1,
                        W1 * Minv * W1.transpose() +
                            W2 * hat_Minv * W2.transpose());
      fillInCouplingTri(num_p1, num_p1 + interface_cnt, -W2 * hat_Minv * hat_G);
      fillInCouplingTri(num_p1 + interface_cnt, num_p1,
                        hat_D * hat_Minv * W2.transpose());
      if (use_stage3b_writeback && !force_cpu_left_blocks)
        fillInStage3BLeftTri(num_p1 + interface_cnt, hat_S,
                             stage3b_products.e2_dminvg);
      else
        fillInCouplingTri(num_p1 + interface_cnt, num_p1 + interface_cnt,
                          hat_S - hat_D * hat_Minv * hat_G);

      Eigen::SparseMatrix<T> matrix(num_p1 + interface_cnt + num_p2,
                                    num_p1 + interface_cnt + num_p2);
      matrix.setFromTriplets(coupling_tri.begin(), coupling_tri.end());
      matrix.makeCompressed();
      return matrix;
    };

    auto assemble_coupling_csr = [&]() {
      ZIRAN_TIMER();
      const int total_rows = num_p1 + interface_cnt + num_p2;
      std::vector<std::vector<std::pair<int, T>>> rows(
          static_cast<std::size_t>(total_rows));
      addEigenBlockToCsrRows(0, 0, S, rows);
      addStage3BCsrBlockToRows(0, 0, stage3b_products.e1_dminvg, (T)-1,
                               rows);

      addStage3BCsrBlockToRows(0, num_p1,
                               stage3b_products.e1_dminv_wt, (T)-1, rows);
      addStage3BCsrBlockToRows(num_p1, 0,
                               stage3b_products.e1_w_minv_g, (T)1, rows);
      addStage3BCsrBlockToRows(num_p1, num_p1,
                               stage3b_products.e1_w_minv_wt, (T)1, rows);
      addStage3BCsrBlockToRows(num_p1, num_p1,
                               stage3b_products.e2_w_minv_wt, (T)1, rows);
      addStage3BCsrBlockToRows(num_p1, num_p1 + interface_cnt,
                               stage3b_products.e2_w_minv_g, (T)-1, rows);
      addStage3BCsrBlockToRows(num_p1 + interface_cnt, num_p1,
                               stage3b_products.e2_dminv_wt, (T)1, rows);

      addEigenBlockToCsrRows(num_p1 + interface_cnt,
                             num_p1 + interface_cnt, hat_S, rows);
      addStage3BCsrBlockToRows(num_p1 + interface_cnt,
                               num_p1 + interface_cnt,
                               stage3b_products.e2_dminvg, (T)-1, rows);
      HostCsrMatrix csr = finalizeCsrRows(total_rows, total_rows, rows);
      ZIRAN_INFO("Stage3B GPU writeback global CSR: rows=", csr.rows,
                 " cols=", csr.cols, " nonzeros=", csr.values.size());
      return csr;
    };

    auto assemble_coupling_device_csr = [&]() {
      ZIRAN_TIMER();
      const int total_rows = num_p1 + interface_cnt + num_p2;
      std::vector<GPU_STAGE3A::DeviceCsrBlock> blocks;
      blocks.reserve(10);
      blocks.push_back({e1.gpu_s, 0, 0, 1.0});
      blocks.push_back({stage3b_products.e1_dminvg_device, 0, 0, -1.0});
      blocks.push_back({stage3b_products.e1_dminv_wt_device,
                        0, num_p1, -1.0});
      blocks.push_back({stage3b_products.e1_w_minv_g_device,
                        num_p1, 0, 1.0});
      blocks.push_back({stage3b_products.e1_w_minv_wt_device,
                        num_p1, num_p1, 1.0});
      blocks.push_back({stage3b_products.e2_w_minv_wt_device,
                        num_p1, num_p1, 1.0});
      blocks.push_back({stage3b_products.e2_w_minv_g_device,
                        num_p1, num_p1 + interface_cnt, -1.0});
      blocks.push_back({stage3b_products.e2_dminv_wt_device,
                        num_p1 + interface_cnt, num_p1, 1.0});
      blocks.push_back({e2.gpu_s, num_p1 + interface_cnt,
                        num_p1 + interface_cnt, 1.0});
      blocks.push_back({stage3b_products.e2_dminvg_device,
                        num_p1 + interface_cnt,
                        num_p1 + interface_cnt, -1.0});
      GPU_STAGE3A::DeviceCsrMatrix device_csr =
          GPU_STAGE3A::assembleDeviceCsrBlocks(
              total_rows, total_rows, blocks);
      ZIRAN_INFO("Stage3B device global CSR: rows=", device_csr.rows,
                 " cols=", device_csr.cols, " nonzeros=", device_csr.nonzeros);
      return device_csr;
    };

    Eigen::SparseMatrix<T> coupling_m_reference;
    HostCsrMatrix coupling_csr;
    GPU_STAGE3A::DeviceCsrMatrix coupling_device_csr;
    GPU_STAGE3A::DeviceVector coupling_solution_device;
    bool coupling_device_csr_valid = false;
    const bool use_stage3b_csr_cuda =
        use_stage3b_writeback &&
        (solverType == AMGCL_CUDA || gpu_coupling_amgx_solver);
    const bool compare_stage3b_writeback =
        use_stage3b_writeback && gpu_stage3b_probe;
    if (compare_stage3b_writeback)
      coupling_m_reference = assemble_coupling_matrix(true);
    if (use_stage3b_csr_cuda) {
      if (e1.gpu_device_assembly_valid && e2.gpu_device_assembly_valid) {
        coupling_device_csr = assemble_coupling_device_csr();
        coupling_device_csr_valid = true;
        if (!gpu_coupling_amgx_solver) {
          const GPU_STAGE3A::CsrMatrix downloaded =
              GPU_STAGE3A::copyDeviceCsrToHost(coupling_device_csr);
          coupling_csr.rows = downloaded.rows;
          coupling_csr.cols = downloaded.cols;
          coupling_csr.row_offsets = downloaded.row_offsets;
          coupling_csr.col_indices = downloaded.col_indices;
          coupling_csr.values.assign(downloaded.values.begin(), downloaded.values.end());
          ZIRAN_INFO("Stage3B device global CSR downloaded for AMGCL setup");
        }
      } else {
        coupling_csr = assemble_coupling_csr();
        if (gpu_coupling_amgx_solver) {
          GPU_STAGE3A::CsrMatrix upload;
          upload.rows = coupling_csr.rows;
          upload.cols = coupling_csr.cols;
          upload.row_offsets = coupling_csr.row_offsets;
          upload.col_indices = coupling_csr.col_indices;
          upload.values.assign(coupling_csr.values.begin(), coupling_csr.values.end());
          coupling_device_csr = GPU_STAGE3A::makeDeviceCsrFromHost(upload);
          coupling_device_csr_valid = true;
        }
      }
    }
    const bool need_coupling_m =
        !use_stage3b_csr_cuda || gpu_stage3a_probe || gpu_stage3b_probe;
    Eigen::SparseMatrix<T> coupling_m;
    if (need_coupling_m)
      coupling_m = assemble_coupling_matrix(false);
    if (gpu_stage3a_probe)
      runGpuStage3AOperatorProbe(coupling_m);
    // build rhs
    coupling_rhs = TStack::Zero(num_p1 + interface_cnt + num_p2);
    GPU_STAGE3A::DeviceVector coupling_rhs_device;
    bool coupling_rhs_device_valid = false;
    bool coupling_rhs_host_valid = true;
    if (use_stage3b_writeback) {
      if (e1.gpu_device_assembly_valid && e2.gpu_device_assembly_valid) {
        auto download_vector = [](const GPU_STAGE3A::DeviceVector &vector) {
          const std::vector<double> host =
              GPU_STAGE3A::copyDeviceVectorToHost(vector);
          TStack result = TStack::Zero(static_cast<int>(host.size()));
          for (int i = 0; i < result.rows(); ++i)
            result(i) = static_cast<T>(host[static_cast<std::size_t>(i)]);
          return result;
        };
        const auto fluid_dminv_a = GPU_STAGE3A::multiplyDeviceCsrVector(
            stage3b_products.e1_dminv_device, e1.gpu_a);
        const auto fluid_rhs_device = GPU_STAGE3A::addDeviceVectors(
            e1.gpu_b, -1.0, fluid_dminv_a);

        const auto e1_interface_rhs = GPU_STAGE3A::multiplyDeviceCsrVector(
            stage3b_products.e1_w_minv_device, e1.gpu_a);
        const auto e2_interface_rhs = GPU_STAGE3A::multiplyDeviceCsrVector(
            stage3b_products.e2_w_minv_device, e2.gpu_a);
        const auto interface_rhs_device = GPU_STAGE3A::addDeviceVectors(
            e1_interface_rhs, -1.0, e2_interface_rhs);

        const auto solid_dminv_a = GPU_STAGE3A::multiplyDeviceCsrVector(
            stage3b_products.e2_dminv_device, e2.gpu_a);
        const auto solid_rhs_device = GPU_STAGE3A::addDeviceVectors(
            e2.gpu_b, -1.0, solid_dminv_a);
        if (gpu_coupling_amgx_solver) {
          coupling_rhs_device = GPU_STAGE3A::concatenateDeviceVectors(
              {fluid_rhs_device, interface_rhs_device, solid_rhs_device});
          ZIRAN_ASSERT(coupling_rhs_device.size ==
                           static_cast<std::size_t>(coupling_rhs.rows()),
                       "device coupling RHS size mismatch");
          coupling_rhs_device_valid = true;
          coupling_rhs_host_valid = false;
        } else {
          fillInCouplingRhs(0, download_vector(fluid_rhs_device));
          fillInCouplingRhs(num_p1, download_vector(interface_rhs_device));
          fillInCouplingRhs(num_p1 + interface_cnt,
                            download_vector(solid_rhs_device));
        }
      } else {
        fillInCouplingRhs(
            0, b - multiplyStage3BDeviceCsrVector(
                       stage3b_products.e1_dminv_device, a));
        fillInCouplingRhs(num_p1,
                          multiplyStage3BDeviceCsrVector(
                              stage3b_products.e1_w_minv_device, a) -
                              multiplyStage3BDeviceCsrVector(
                                  stage3b_products.e2_w_minv_device, hat_a));
        fillInCouplingRhs(num_p1 + interface_cnt,
                          hat_b - multiplyStage3BDeviceCsrVector(
                                      stage3b_products.e2_dminv_device, hat_a));
      }
    } else {
      fillInCouplingRhs(0, b - D * Minv * a);
      fillInCouplingRhs(num_p1, W1 * Minv * a - W2 * hat_Minv * hat_a);
      fillInCouplingRhs(num_p1 + interface_cnt,
                        hat_b - hat_D * hat_Minv * hat_a);
    }
    if (gpu_full_pipeline_strict) {
      ZIRAN_ASSERT(use_stage3b_csr_cuda && coupling_device_csr_valid &&
                       coupling_rhs_device_valid &&
                       !coupling_rhs_host_valid && gpu_coupling_amgx_solver,
          "strict full-GPU coupling matrix/RHS/solver fell back to host");
    }

    auto materialize_coupling_rhs = [&]() {
      if (coupling_rhs_host_valid)
        return;
      ZIRAN_ASSERT(coupling_rhs_device_valid,
                   "coupling RHS has neither host nor device storage");
      const std::vector<double> host =
          GPU_STAGE3A::copyDeviceVectorToHost(coupling_rhs_device);
      ZIRAN_ASSERT(host.size() ==
                       static_cast<std::size_t>(coupling_rhs.rows()),
                   "materialized coupling RHS size mismatch");
      for (int i = 0; i < coupling_rhs.rows(); ++i)
        coupling_rhs(i) = static_cast<T>(host[static_cast<std::size_t>(i)]);
      coupling_rhs_host_valid = true;
    };

    const char* coupling_system_dump_prefix =
        std::getenv("ZIRAN_COUPLING_SYSTEM_DUMP_PREFIX");
    if (std::getenv("ZIRAN_COUPLING_SYSTEM_FINGERPRINT") ||
        coupling_system_dump_prefix) {
      materialize_coupling_rhs();
      // Read-only A/B fingerprint for identifying whether two build paths
      // feed the same ordered numerical system.  The CPU branch is included
      // deliberately: it is the oracle for a GPU-vs-CPU first-substep dump.
      // Both branches are opt-in because the device branch downloads a large
      // CSR solely for diagnostics.
      GPU_STAGE3A::CsrMatrix host_csr;
      if (coupling_device_csr_valid) {
        host_csr = GPU_STAGE3A::copyDeviceCsrToHost(coupling_device_csr);
      } else {
        ZIRAN_ASSERT(coupling_m.rows() > 0 && coupling_m.cols() > 0,
                     "CPU coupling matrix unavailable for system dump");
        host_csr = GPU_STAGE3A::makeCsrFromEigen(toDoubleSparse(coupling_m));
      }
      auto fnv_bytes = [](std::uint64_t hash, const void* data,
                          std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
          hash ^= static_cast<std::uint64_t>(bytes[i]);
          hash *= 1099511628211ull;
        }
        return hash;
      };
      std::uint64_t row_hash = 1469598103934665603ull;
      std::uint64_t col_hash = row_hash;
      std::uint64_t value_hash = row_hash;
      std::uint64_t rhs_hash = row_hash;
      for (const int value : host_csr.row_offsets)
        row_hash = fnv_bytes(row_hash, &value, sizeof(value));
      for (const int value : host_csr.col_indices)
        col_hash = fnv_bytes(col_hash, &value, sizeof(value));
      double value_l1 = 0, value_l2 = 0, rhs_l1 = 0, rhs_l2 = 0;
      for (const double value : host_csr.values) {
        value_hash = fnv_bytes(value_hash, &value, sizeof(value));
        value_l1 += std::abs(value);
        value_l2 += value * value;
      }
      for (int i = 0; i < coupling_rhs.rows(); ++i) {
        const double value = static_cast<double>(coupling_rhs(i));
        rhs_hash = fnv_bytes(rhs_hash, &value, sizeof(value));
        rhs_l1 += std::abs(value);
        rhs_l2 += value * value;
      }
      ZIRAN_INFO("Coupling system fingerprint: rows=", host_csr.rows,
          " nnz=", host_csr.values.size(), " row_hash=", row_hash,
          " col_hash=", col_hash, " value_hash=", value_hash,
          " rhs_hash=", rhs_hash, " value_l1=", value_l1,
          " value_l2=", std::sqrt(value_l2), " rhs_l1=", rhs_l1,
          " rhs_l2=", std::sqrt(rhs_l2));
      if (coupling_system_dump_prefix) {
        static int dump_index = 0;
        const std::string path = std::string(coupling_system_dump_prefix) + "_" +
            std::to_string(dump_index++) + ".bin";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        const std::uint64_t rows = static_cast<std::uint64_t>(host_csr.rows);
        const std::uint64_t nnz = static_cast<std::uint64_t>(host_csr.values.size());
        const std::uint64_t rhs_size =
            static_cast<std::uint64_t>(coupling_rhs.rows());
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
        out.write(reinterpret_cast<const char*>(&rhs_size), sizeof(rhs_size));
        out.write(reinterpret_cast<const char*>(host_csr.row_offsets.data()),
                  static_cast<std::streamsize>(host_csr.row_offsets.size() * sizeof(int)));
        out.write(reinterpret_cast<const char*>(host_csr.col_indices.data()),
                  static_cast<std::streamsize>(host_csr.col_indices.size() * sizeof(int)));
        out.write(reinterpret_cast<const char*>(host_csr.values.data()),
                  static_cast<std::streamsize>(host_csr.values.size() * sizeof(double)));
        for (int i = 0; i < coupling_rhs.rows(); ++i) {
          const double value = static_cast<double>(coupling_rhs(i));
          out.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
        out.flush();
        ZIRAN_ASSERT(out.good(), "failed to write coupling-system debug dump");
        out.close();
        ZIRAN_ASSERT(!out.fail(), "failed to close coupling-system debug dump");
        ZIRAN_INFO("Coupling system dump: ", path);
        if (std::getenv("ZIRAN_STOP_AFTER_COUPLING_SYSTEM_DUMP"))
          throw std::runtime_error(
              "stopped after requested coupling-system diagnostic dump");
      }
    }
    TStack solved_pap;
    bool coupling_solution_device_only = false;

    ZIRAN_ASSERT(e1.dt == e2.dt);
    ZIRAN_ASSERT(e1.dx == e2.dx);
    T kkt_relax_local = kkt_relax * std::pow(e1.dx, (T)dim) / e1.dt;
    // These opt-in overrides keep solver-convergence A/B tests out of scene
    // setup.  They affect every non-AMGX coupling solver invocation below;
    // AMGX has matching ZIRAN_AMGX_COUPLING_* controls in its device config.
    T coupling_relative_tol = relative_tol;
    int coupling_max_iter = max_iter;
    if (const char* value = std::getenv("ZIRAN_COUPLING_RELATIVE_TOLERANCE")) {
      char* end = nullptr;
      const double parsed = std::strtod(value, &end);
      if (end != value && parsed > 0)
        coupling_relative_tol = static_cast<T>(parsed);
    }
    if (const char* value = std::getenv("ZIRAN_COUPLING_MAX_ITERS")) {
      char* end = nullptr;
      const long parsed = std::strtol(value, &end, 10);
      if (end != value && parsed > 0)
        coupling_max_iter = static_cast<int>(parsed);
    }
    if (coupling_relative_tol != relative_tol || coupling_max_iter != max_iter)
      ZIRAN_INFO("Coupling solver A/B override: relative_tol=",
                 coupling_relative_tol, " max_iter=", coupling_max_iter);

    auto solve_coupling_matrix = [&](const Eigen::SparseMatrix<T> &matrix) {
      if (solverType == EIGEN)
        return DirectSolver<T>::solve(matrix, coupling_rhs,
                                      DirectSolver<T>::EIGEN);
      else if (solverType == AMGCL)
        return DirectSolver<T>::solve(
            matrix, coupling_rhs, DirectSolver<T>::AMGCL,
            coupling_relative_tol, abs_tol, coupling_max_iter, -1, 1, false,
            false, kkt_relax_local);
      else
        return DirectSolver<T>::solve(
            matrix, coupling_rhs, DirectSolver<T>::AMGCL_CUDA,
            coupling_relative_tol, abs_tol, coupling_max_iter, -1, 1, false,
            false, kkt_relax_local);
    };

    if (use_stage3b_csr_cuda) {
      if (std::getenv("ZIRAN_COUPLING_DEVICE_CSR_HOST_AMGCL")) {
        // A/B diagnostic: preserve the GPU-assembled CSR and its exact DOF
        // order, but solve it with the CPU AMGCL backend. This separates an
        // assembly/permutation defect from a GPU-solver convergence effect.
        ZIRAN_ASSERT(coupling_device_csr_valid,
                     "host AMGCL diagnostic requires a device coupling CSR");
        materialize_coupling_rhs();
        const GPU_STAGE3A::CsrMatrix downloaded =
            GPU_STAGE3A::copyDeviceCsrToHost(coupling_device_csr);
        const Eigen::SparseMatrix<T> host_matrix =
            GPU_STAGE3A::makeEigenFromCsrLocal<T>(downloaded);
        {
          ZIRAN_NAMED_TIMER("couplingDeviceCsrHostAmgclSolve");
          solved_pap = DirectSolver<T>::solve(
              host_matrix, coupling_rhs, DirectSolver<T>::AMGCL,
              coupling_relative_tol, abs_tol, coupling_max_iter, -1, 1, false,
              false, kkt_relax_local);
        }
        logLinearResidualStats("GPU CSR / host AMGCL coupling residual",
                               host_matrix, coupling_rhs, solved_pap);
      } else if (gpu_coupling_amgx_solver) {
#ifdef ENABLE_AMGX
        ZIRAN_ASSERT(coupling_device_csr_valid,
                     "AMGX coupling solver requires a device coupling CSR");
        std::vector<double> solution_host;
        const bool require_solution_host =
            std::getenv("ZIRAN_AMGX_COUPLING_COMPARE_AMGCL") != nullptr ||
            compare_stage3b_writeback || gpu_stage3b_probe ||
            std::getenv("ZIRAN_COUPLING_SOLUTION_DUMP_PREFIX") != nullptr ||
            std::getenv("ZIRAN_SOLID_PRESSURE_FIELD_DUMP_PREFIX") != nullptr ||
            std::getenv("ZIRAN_GPU_FLUID_VELOCITY_COMPARE") != nullptr;
        GPU_STAGE2::AmgxCsrStats amgx_stats;
        {
          ZIRAN_NAMED_TIMER("couplingAmgxSolve");
          if (coupling_rhs_device_valid) {
            GPU_STAGE2::solveDeviceCsrAmgx3D(
                gpu_coupling_amgx_context, coupling_device_csr,
                coupling_rhs_device, static_cast<double>(kkt_relax_local),
                require_solution_host ? &solution_host : nullptr,
                &amgx_stats, &coupling_solution_device);
          } else {
            materialize_coupling_rhs();
            std::vector<double> rhs_host(
                static_cast<std::size_t>(coupling_rhs.rows()));
            for (int i = 0; i < coupling_rhs.rows(); ++i)
              rhs_host[static_cast<std::size_t>(i)] =
                  static_cast<double>(coupling_rhs(i));
            GPU_STAGE2::solveDeviceCsrAmgx3D(
                gpu_coupling_amgx_context, coupling_device_csr, rhs_host,
                static_cast<double>(kkt_relax_local), solution_host,
                &amgx_stats, &coupling_solution_device);
          }
        }
        solved_pap = TStack::Zero(coupling_device_csr.rows);
        if (require_solution_host) {
          ZIRAN_ASSERT(solution_host.size() ==
                           static_cast<std::size_t>(solved_pap.rows()),
                       "coupling AMGX host solution size mismatch");
          for (std::size_t i = 0; i < solution_host.size(); ++i)
            solved_pap(static_cast<int>(i)) = static_cast<T>(solution_host[i]);
        } else {
          coupling_solution_device_only = true;
        }
        if (gpu_full_pipeline_strict) {
          ZIRAN_ASSERT(amgx_stats.solve_status == 0 &&
                           coupling_solution_device_only &&
                           coupling_solution_device.size ==
                               static_cast<std::size_t>(solved_pap.rows()),
              "strict full-GPU coupling AMGX solve failed or downloaded its solution");
        }
        ZIRAN_INFO("Coupling AMGX device CSR solver: rows=", amgx_stats.rows,
                   " nnz=", amgx_stats.nnz,
                   " iterations=", amgx_stats.iterations,
                   " status=", amgx_stats.solve_status,
                   " diagonal_epsilon=", amgx_stats.diagonal_epsilon,
                   " values_d2d_ms=", amgx_stats.values_copy_ms,
                   " rhs_h2d_ms=", amgx_stats.rhs_upload_ms,
                   " matrix_d2d_ms=", amgx_stats.matrix_upload_ms,
                   " setup_ms=", amgx_stats.setup_ms,
                   " solve_ms=", amgx_stats.solve_ms,
                   " solution_stage_ms=", amgx_stats.solution_download_ms,
                   " solution_host=", require_solution_host ? 1 : 0,
                   " total_ms=", amgx_stats.total_ms);
        if (std::getenv("ZIRAN_AMGX_COUPLING_COMPARE_AMGCL")) {
          materialize_coupling_rhs();
          const GPU_STAGE3A::CsrMatrix downloaded =
              GPU_STAGE3A::copyDeviceCsrToHost(coupling_device_csr);
          HostCsrMatrix debug_csr;
          debug_csr.rows = downloaded.rows;
          debug_csr.cols = downloaded.cols;
          debug_csr.row_offsets = downloaded.row_offsets;
          debug_csr.col_indices = downloaded.col_indices;
          debug_csr.values.assign(downloaded.values.begin(), downloaded.values.end());
          const TStack amgcl_solution = DirectSolver<T>::solveCsrCUDA(
              debug_csr.rows, debug_csr.cols, debug_csr.row_offsets,
              debug_csr.col_indices, debug_csr.values, coupling_rhs,
              coupling_relative_tol, abs_tol, coupling_max_iter, -1, 1,
              kkt_relax_local);
          const TStack difference = solved_pap - amgcl_solution;
          T max_abs_difference = 0;
          for (int i = 0; i < difference.rows(); ++i)
            max_abs_difference = std::max(max_abs_difference,
                                          std::abs(difference(i)));
          const T amgcl_norm = amgcl_solution.norm();
          const T relative_difference = amgcl_norm > (T)0
              ? difference.norm() / amgcl_norm : difference.norm();
          logCsrResidualStats("Coupling AMGX raw residual", debug_csr,
                              coupling_rhs, solved_pap);
          logCsrResidualStats("Coupling AMGCL raw residual", debug_csr,
                              coupling_rhs, amgcl_solution);
          ZIRAN_INFO("Coupling AMGX/AMGCL solution compare: max_abs=",
                     max_abs_difference, " rel_l2=", relative_difference);
        }
#else
        ZIRAN_ASSERT(false, "AMGX coupling solver requested without ENABLE_AMGX");
#endif
      } else {
        solved_pap = DirectSolver<T>::solveCsrCUDA(
            coupling_csr.rows, coupling_csr.cols, coupling_csr.row_offsets,
            coupling_csr.col_indices, coupling_csr.values, coupling_rhs,
            coupling_relative_tol, abs_tol, coupling_max_iter, -1, 1,
            kkt_relax_local);
        logCsrResidualStats("Stage3B GPU writeback CSR coupling solve residual",
                            coupling_csr, coupling_rhs, solved_pap);
      }
    } else {
      solved_pap = solve_coupling_matrix(coupling_m);
      if (use_stage3b_writeback)
        logLinearResidualStats("Stage3B GPU writeback coupling solve residual",
                               coupling_m, coupling_rhs, solved_pap);
    }
    TStack reference_solved_pap;
    if (compare_stage3b_writeback) {
      materialize_coupling_rhs();
      reference_solved_pap = solve_coupling_matrix(coupling_m_reference);
      logVectorDiffStats("Stage3B GPU writeback solution compare",
                         reference_solved_pap, solved_pap);
      logLinearResidualStats("Stage3B CPU reference coupling solve residual",
                             coupling_m_reference, coupling_rhs,
                             reference_solved_pap);
    }
    if (gpu_stage3b_probe) {
      materialize_coupling_rhs();
      runGpuStage3BProbe(coupling_m, coupling_rhs,
                         compare_stage3b_writeback ? reference_solved_pap
                                                   : solved_pap);
    }
    if (const char* dump_prefix =
            std::getenv("ZIRAN_COUPLING_SOLUTION_DUMP_PREFIX")) {
      static int dump_index = 0;
      const std::string path = std::string(dump_prefix) + "_" +
          std::to_string(dump_index++) + ".bin";
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      const std::uint64_t size = static_cast<std::uint64_t>(solved_pap.rows());
      out.write(reinterpret_cast<const char*>(&size), sizeof(size));
      for (int i = 0; i < solved_pap.rows(); ++i) {
        const double value = static_cast<double>(solved_pap(i));
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
      }
      ZIRAN_ASSERT(out.good(), "failed to write coupling-solution debug dump");
      ZIRAN_INFO("Coupling solution dump: ", path);
    }
    e1.solved_p = TStack(num_p1);
    for (int i = 0; i < num_p1; ++i)
      e1.solved_p(i) = solved_pap(i);
    e2.solved_p = TStack(num_p2);
    for (int i = 0; i < num_p2; ++i)
      e2.solved_p(i) = solved_pap(num_p1 + interface_cnt + i);
    const char* solid_pressure_dump_prefix =
        std::getenv("ZIRAN_SOLID_PRESSURE_FIELD_DUMP_PREFIX");
    if constexpr (dim == 3) {
      if (solid_pressure_dump_prefix) {
        // This is deliberately before interface-alpha application and before
        // either constructVelocityFromSolvedP call.  It compares the solved
        // pressure in physical grid-key space rather than raw DOF space.
        auto& solid = e1.category == MATERIAL_PHASE_SOLID ? e1 : e2;
        struct PressureEntry {
          std::uint64_t key;
          double value;
        };
        std::vector<PressureEntry> entries;
        if (solid.gpu_prebuilt_solid_pressure_valid ||
            !solid.gpu_pressure_nodes.empty()) {
          entries.reserve(solid.gpu_pressure_nodes.size());
          for (const auto& node : solid.gpu_pressure_nodes) {
            ZIRAN_ASSERT(node.idx >= 0 && node.idx < solid.solved_p.rows(),
                         "GPU solid pressure diagnostic index out of range");
            entries.push_back({node.key,
                static_cast<double>(solid.solved_p(node.idx))});
          }
        } else {
          // Recreate precisely the CPU Q2Q0 pressure grid used by
          // buildSolidSystem(): this grid is local there, so it must be
          // materialized again for a physical-key diagnostic.
          // buildSolidSystem() for the active Q2Q0 solid path creates
          // MpmGrid<T, dim, order> with order == 0.  This must stay a Q0
          // pressure grid; aod is the velocity/auxiliary order and creates a
          // different support if used here.
          MpmGrid<T, dim, 0> pressure_grid;
          pressure_grid.activatePages(solid.particles.X.array, solid.dx, 0, 1,
              -TV::Ones() * solid.dx * (T)0.5);
          ZIRAN_ASSERT(pressure_grid.getNumNodes() == solid.solved_p.rows(),
                       "CPU solid pressure diagnostic grid size disagrees");
          entries.reserve(static_cast<std::size_t>(pressure_grid.getNumNodes()));
          pressure_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
            entries.push_back({GPU_STAGE3A::packGridKey3D(
                node[0], node[1], node[2]),
                static_cast<double>(solid.solved_p(g.idx))});
          });
        }
        std::sort(entries.begin(), entries.end(),
                  [](const PressureEntry& a, const PressureEntry& b) {
                    return a.key < b.key;
                  });
        static int dump_index = 0;
        const std::string path = std::string(solid_pressure_dump_prefix) + "_" +
            std::to_string(dump_index++) + ".bin";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        const std::uint64_t count = entries.size();
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& entry : entries) {
          out.write(reinterpret_cast<const char*>(&entry.key), sizeof(entry.key));
          out.write(reinterpret_cast<const char*>(&entry.value),
                    sizeof(entry.value));
        }
        out.flush();
        ZIRAN_ASSERT(out.good(),
                     "failed to write solid pressure-field diagnostic dump");
        out.close();
        ZIRAN_ASSERT(!out.fail(),
                     "failed to close solid pressure-field diagnostic dump");
        ZIRAN_INFO("Solid pressure-field dump (pre-construct): ", path,
                   " nodes=", entries.size());
        // A/B solver probes need precisely this physical pressure field, not
        // a complete frame.  Stopping here also ensures later interface and
        // velocity updates cannot affect the diagnostic result.
        if (std::getenv("ZIRAN_STOP_AFTER_SOLID_PRESSURE_DUMP"))
          throw std::runtime_error(
              "stopped after requested solid-pressure diagnostic dump");
      }
    }
    if (gpu_coupling_amgx_solver && coupling_solution_device.size != 0) {
      e1.gpu_solved_p = GPU_STAGE3A::sliceDeviceVector(
          coupling_solution_device, 0, static_cast<std::size_t>(num_p1));
      e1.gpu_solved_p_valid = true;
      e2.gpu_solved_p = GPU_STAGE3A::sliceDeviceVector(
          coupling_solution_device,
          static_cast<std::size_t>(num_p1 + interface_cnt),
          static_cast<std::size_t>(num_p2));
      e2.gpu_solved_p_valid = true;
      if (coupling_solution_device_only) {
        // Both Q0 pressure gathers are device-side in the resident pipeline.
        // Keep the selective solid D2H only for compatibility/A-B paths.
        auto& solid = e1.category == MATERIAL_PHASE_SOLID ? e1 : e2;
        const bool resident_solid_pressure_consumer =
            solid.gpu_solid_device_resident &&
            GPU_STAGE3A::persistentSolidParticleStateValid(
                solid.gpu_solid_particle_state,
                static_cast<std::size_t>(solid.particles.count)) &&
            solid.gpu_pressure_node_map.size == solid.gpu_solved_p.size;
        if (resident_solid_pressure_consumer) {
          ZIRAN_INFO("Coupling AMGX solid pressure remains device-resident: "
                     "rows=", solid.gpu_solved_p.size);
        } else {
          const std::vector<double> solid_pressure =
              GPU_STAGE3A::copyDeviceVectorToHost(solid.gpu_solved_p);
          ZIRAN_ASSERT(solid_pressure.size() ==
                           static_cast<std::size_t>(solid.solved_p.rows()),
                       "solid coupling pressure slice size mismatch");
          for (int i = 0; i < solid.solved_p.rows(); ++i)
            solid.solved_p(i) = static_cast<T>(
                solid_pressure[static_cast<std::size_t>(i)]);
          ZIRAN_INFO("Coupling AMGX selective solution D2H: solid_rows=",
                     solid_pressure.size(), " full_rows=",
                     coupling_solution_device.size);
        }
      }
    } else {
      e1.gpu_solved_p_valid = false;
      e2.gpu_solved_p_valid = false;
    }
    if constexpr (dim == 3) {
      if (solid_pressure_dump_prefix) {
        auto& solid = e1.category == MATERIAL_PHASE_SOLID ? e1 : e2;
        if (solid.gpu_solved_p_valid) {
          const std::vector<double> device_pressure =
              GPU_STAGE3A::copyDeviceVectorToHost(solid.gpu_solved_p);
          ZIRAN_ASSERT(device_pressure.size() ==
                           static_cast<std::size_t>(solid.solved_p.rows()),
                       "solid AMGX pressure slice has wrong length");
          double max_abs = 0;
          double diff_l2 = 0;
          for (int i = 0; i < solid.solved_p.rows(); ++i) {
            const double difference = device_pressure[static_cast<std::size_t>(i)] -
                static_cast<double>(solid.solved_p(i));
            max_abs = std::max(max_abs, std::abs(difference));
            diff_l2 += difference * difference;
          }
          ZIRAN_INFO("Solid AMGX pressure device-slice consistency: rows=",
              solid.solved_p.rows(), " max_abs=", max_abs,
              " diff_l2=", std::sqrt(diff_l2));
        }
      }
    }
    TStack solved_alpha = TStack(interface_cnt);
    solved_alpha.setZero();
    if (!coupling_solution_device_only)
      for (int i = 0; i < interface_cnt; ++i)
        solved_alpha(i) = solved_pap(num_p1 + i);
    ZIRAN_ASSERT(W1.rows() == solved_alpha.rows(), "wrong");
    ZIRAN_ASSERT(W1.cols() == e1.a.rows(), "wrong");
    if (!coupling_solution_device_only)
      e1.a -= W1.transpose() * solved_alpha;
    auto apply_device_alpha = [&](SplittingSimulation<T, dim> &simulation,
                                  const GPU_STAGE3A::DeviceCsrMatrix &wt_device,
                                  const Eigen::SparseMatrix<T> &w,
                                  double scale) {
      if (!simulation.gpu_device_assembly_valid || interface_cnt == 0)
        return;
      GPU_STAGE3A::DeviceVector alpha_device;
      GPU_STAGE3A::DeviceCsrMatrix fallback_wt;
      const GPU_STAGE3A::DeviceCsrMatrix* active_wt = &wt_device;
      if (coupling_solution_device.size != 0) {
        alpha_device = GPU_STAGE3A::sliceDeviceVector(
            coupling_solution_device, static_cast<std::size_t>(num_p1),
            static_cast<std::size_t>(interface_cnt));
      } else {
        // Non-AMGX / host-solver fallback: retain the prior compatibility
        // behavior, including the compact alpha upload.
        const auto w_device = GPU_STAGE3A::makeDeviceCsrFromHost(
            GPU_STAGE3A::makeCsrFromEigen(toDoubleSparse(w)));
        fallback_wt = GPU_STAGE3A::transposeDeviceCsrMatrix(w_device);
        active_wt = &fallback_wt;
        std::vector<double> alpha(static_cast<std::size_t>(solved_alpha.rows()));
        for (int i = 0; i < solved_alpha.rows(); ++i)
          alpha[static_cast<std::size_t>(i)] =
              static_cast<double>(solved_alpha(i));
        alpha_device = GPU_STAGE3A::makeDeviceVectorFromHost(alpha);
      }
      ZIRAN_NAMED_TIMER("applyDeviceInterfaceAlpha");
      const auto correction = GPU_STAGE3A::multiplyDeviceCsrVector(
          *active_wt, alpha_device);
      simulation.gpu_a = GPU_STAGE3A::addDeviceVectors(
          simulation.gpu_a, scale, correction);
    };
    apply_device_alpha(e1, stage3b_products.e1_wt_device, W1, -1.0);
    e1.constructVelocityFromSolvedP();
    ZIRAN_ASSERT(W2.rows() == solved_alpha.rows(), "wrong");
    ZIRAN_ASSERT(W2.cols() == e2.a.rows(), "wrong");
    if (!coupling_solution_device_only)
      e2.a += W2.transpose() * solved_alpha;
    apply_device_alpha(e2, stage3b_products.e2_wt_device, W2, 1.0);
    e2.constructVelocityFromSolvedP();
    if (gpu_full_pipeline_strict) {
      auto& fluid = e1.category == MATERIAL_PHASE_FLUID ? e1 : e2;
      auto& solid = e1.category == MATERIAL_PHASE_SOLID ? e1 : e2;
      ZIRAN_ASSERT(fluid.gpu_velocity_transfer_device_x_valid &&
                       fluid.gpu_velocity_transfer_device_v_grad_valid &&
                       fluid.gpu_velocity_transfer_device.pressure != nullptr,
          "strict full-GPU fluid pressure/velocity G2P fell back to host");
      ZIRAN_ASSERT(solid.gpu_solid_particle_g2p_ready,
          "strict full-GPU solid pressure/velocity G2P fell back to host");
      ZIRAN_INFO("Strict full-GPU coupling solve verified: AMGX=device",
                 " fluid_G2P=device solid_G2P=device");
    }
  }

  void addBoundarySpring(SplittingSimulation<T, dim> &sim) {
    sim.gpu_solid_device_ghost_contact_active = false;
    sim.gpu_solid_device_ordinary_pinning_active = false;
    const bool ghost_contact_probe =
        std::getenv("ZIRAN_SOLID_GHOST_CONTACT_PROBE") != nullptr;
    auto *pin_pointer = sim.particles.DataManager::getPointer(
        MpmMultiTargetPinningForce<T, dim>::target_name());
    if (pin_pointer == nullptr) {
      if (ghost_contact_probe)
        ZIRAN_INFO("solid GHOST contact probe: parallel_pinning_target=absent");
      return;
    }
    const bool has_dynamic_contact_object = std::any_of(
        sim.collision_objects.begin(), sim.collision_objects.end(),
        [](const auto& object) {
          return object->type == AnalyticCollisionObject<T, dim>::GHOST ||
              object->type == AnalyticCollisionObject<T, dim>::STICKY;
        });
    const bool host_pinning_needs_current_x =
        sim.force && (sim.force->pinning ||
            (sim.force->multi_target_pinning &&
             !sim.force->multi_target_pinning->collider_spring));
    if (!has_dynamic_contact_object && !host_pinning_needs_current_x)
      return;
    if (!has_dynamic_contact_object &&
        sim.canUseResidentSolidOrdinaryPinning()) {
      sim.gpu_solid_device_ordinary_pinning_active = true;
      ZIRAN_INFO("Solid ordinary pinning coefficients remain device-resident");
      return;
    }
    if (sim.canUseResidentSolidGhostContact()) {
      sim.gpu_solid_device_ghost_contact_active = true;
      ZIRAN_INFO("Solid resident GHOST contact targets remain device-resident");
      return;
    }
    ZIRAN_ASSERT(!sim.gpu_full_pipeline_strict,
        "strict full-GPU solid boundary spring fell back to host contact");
    // Generic collision objects and the remaining CPU line-search/pinning
    // energy use current X. Materialize only that field; V/C/F/J remain
    // device-authoritative.
    sim.synchronizeSolidCurrentXForHostContact();
    std::atomic<int> ghost_inside_count{0};
    std::atomic<int> installed_ghost_target_count{0};
    sim.before_advect = sim.particles.X.array;
    tbb::parallel_for(0, sim.particles.count, [&](int i) {
      (*pin_pointer)[i].x.clear();
      (*pin_pointer)[i].v.clear();
      (*pin_pointer)[i].k = 0;
    });
    // addSpringWithSolid(another_sim);
    // auto *config_scheme =
    // sim.particles.DataManager::getPointer(config_scheme_name());
    auto *X0 = sim.particles.DataManager::getPointer(X0_name<TV>());
    tbb::parallel_for(0, sim.particles.count, [&](int i) {
      // TV& Xp = sim.particles.X.array[i];
      TV &Xp = (sim.config_scheme != CONFIG_UPDATE_LAGRANGIAN)
                   ? (*X0)[i]
                   : sim.particles.X.array[i];
      for (int k = 0; k < (int)sim.collision_objects.size(); ++k) {
        T phi;
        TV n;
        // if (sim.collision_objects[k]->type !=
        //     AnalyticCollisionObject<T, dim>::GHOST)
        //   continue;
        // bool collided = sim.collision_objects[k]->queryInside(Xp, phi, n, 0);
        // if (collided) {
        //   TV Xs = Xp - phi * n;
        //   (*pin_pointer)[i].x.push_back(Xs);
        //   (*pin_pointer)[i].v.push_back(TV::Zero());
        //   (*pin_pointer)[i].k = sim.implicit_penalty_collision;
        // }
        if (sim.collision_objects[k]->type ==
            AnalyticCollisionObject<T, dim>::GHOST) {
          bool collided = sim.collision_objects[k]->queryInside(Xp, phi, n, 0);
          if (collided) {
            if (ghost_contact_probe)
              ++ghost_inside_count;
            TV Xs = Xp - phi * n;
            (*pin_pointer)[i].x.push_back(Xs);
            (*pin_pointer)[i].v.push_back(TV::Zero());
            (*pin_pointer)[i].k = sim.implicit_penalty_collision;
            if (ghost_contact_probe)
              ++installed_ghost_target_count;
          }
        } else if (sim.collision_objects[k]->type ==
                   AnalyticCollisionObject<T, dim>::STICKY) {
          bool collided = sim.collision_objects[k]->queryInside(Xp, phi, n, 0);
          if (collided) {
            TV &Vp = sim.particles.V.array[i];
            TV Xs = Xp - phi * n;
            // Vp = Vp.dot(n) * n ;
            // Vp = Vp.dot(n) * n + 0.5 * (Vp - Vp.dot(n) * n);
            (*pin_pointer)[i].x.push_back(Xs);
            (*pin_pointer)[i].v.push_back(TV::Zero());
            (*pin_pointer)[i].k = sim.implicit_penalty_collision;
          }
        }
      }
    });
    if (ghost_contact_probe)
      ZIRAN_INFO("solid GHOST contact probe: particles_inside=",
                 ghost_inside_count.load(), " installed_ghost_targets=",
                 installed_ghost_target_count.load(), " penalty_k=",
                 sim.implicit_penalty_collision);
  }

  void addBoundarySpringSTICKY(SplittingSimulation<T, dim> &sim) {
    auto *pin_pointer = sim.particles.DataManager::getPointer(
        MpmMultiTargetPinningForce<T, dim>::target_name());
    if (pin_pointer == nullptr)
      return;
    sim.before_advect = sim.particles.X.array;
    // tbb::parallel_for(0, sim.particles.count, [&](int i) {
    //   (*pin_pointer)[i].x.clear();
    //   (*pin_pointer)[i].v.clear();
    //   (*pin_pointer)[i].k = 0;
    // });

    // addSpringWithSolid(another_sim);
    // auto *config_scheme =
    // sim.particles.DataManager::getPointer(config_scheme_name());
    auto *X0 = sim.particles.DataManager::getPointer(X0_name<TV>());
    tbb::parallel_for(0, sim.particles.count, [&](int i) {
      // TV& Xp = sim.particles.X.array[i];
      TV &Xp = (sim.config_scheme != CONFIG_UPDATE_LAGRANGIAN)
                   ? (*X0)[i]
                   : sim.particles.X.array[i];
      for (int k = 0; k < (int)sim.collision_objects.size(); ++k) {
        T phi;
        TV n;
        if (sim.collision_objects[k]->type ==
            AnalyticCollisionObject<T, dim>::STICKY) {
          if ((*pin_pointer)[i].x.size() == 0) {
            bool collided =
                sim.collision_objects[k]->queryInside(Xp, phi, n, 0);
            if (collided) {
              TV &Vp = sim.particles.V.array[i];
              TV Xs = Xp - phi * n;
              // Vp = Vp.dot(n) * n ;
              // Vp = Vp.dot(n) * n + 0.5 * (Vp - Vp.dot(n) * n);
              (*pin_pointer)[i].x.push_back(Xs);
              (*pin_pointer)[i].v.push_back(TV::Zero());
              (*pin_pointer)[i].k = sim.implicit_penalty_collision;
            }
          }
        }
      }
    });
  }

  StdVector<TV> points;
  StdVector<TV> solid_particles;
  StdVector<TV> before_advect;
  StdVector<TV> trajectory;
  StdVector<TV> normals;
  StdVector<Vector<int, 2>> segments;
  void advanceOneTimeStep(double dt) override {
    ZIRAN_INFO("Advance one time step with dt = ", dt);

    ZIRAN_ASSERT((e1.category == MATERIAL_PHASE_FLUID &&
                  e2.category == MATERIAL_PHASE_SOLID) ||
                 (e1.category == MATERIAL_PHASE_SOLID &&
                  e2.category == MATERIAL_PHASE_FLUID));
    ZIRAN_ASSERT((e2.config_scheme == CONFIG_UPDATE_LAGRANGIAN &&
                  e2.category == MATERIAL_PHASE_FLUID) ||
                 (e1.config_scheme == CONFIG_UPDATE_LAGRANGIAN &&
                  e1.category == MATERIAL_PHASE_FLUID));
    // ZIRAN_ASSERT((e2.config_scheme == CONFIG_UPDATE_LAGRANGIAN) ||
    //  (e1.config_scheme == CONFIG_UPDATE_LAGRANGIAN));

    e1.dt = dt;
    e1.setFrame(frame);
    e2.dt = dt;
    e2.setFrame(frame);

    const bool fluid_state_fingerprint =
        std::getenv("ZIRAN_FLUID_STATE_FINGERPRINT") != nullptr;
    auto fingerprint_fluid_state = [&](const char* phase,
                                       bool use_override_state) {
      if (!fluid_state_fingerprint)
        return;
      auto& fluid = e1.category == MATERIAL_PHASE_FLUID ? e1 : e2;
      const bool resident_state = fluid.gpu_fluid_device_resident &&
          fluid.gpu_velocity_transfer_device_x_valid &&
          fluid.gpu_velocity_transfer_device_v_grad_valid;
      if (resident_state)
        fluid.synchronizeFluidCompatibilityForHostOutput();
      const auto& x = use_override_state && !resident_state
          ? fluid.override_X : fluid.particles.X.array;
      const auto& v = use_override_state && !resident_state
          ? fluid.override_V : fluid.particles.V.array;
      if (x.size() != static_cast<std::size_t>(fluid.particles.count) ||
          v.size() != static_cast<std::size_t>(fluid.particles.count)) {
        ZIRAN_INFO("Fluid state fingerprint: phase=", phase,
                   " unavailable=1 x_size=", x.size(), " v_size=", v.size());
        return;
      }
      auto mix = [](std::uint64_t hash, double value) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
          hash ^= static_cast<std::uint64_t>(bytes[byte]);
          hash *= 1099511628211ull;
        }
        return hash;
      };
      std::uint64_t x_hash = 1469598103934665603ull;
      std::uint64_t v_hash = x_hash;
      double x_l2 = 0, v_l2 = 0;
      for (int p = 0; p < fluid.particles.count; ++p)
        for (int d = 0; d < dim; ++d) {
          const double xv = static_cast<double>(x[p][d]);
          const double vv = static_cast<double>(v[p][d]);
          x_hash = mix(x_hash, xv);
          v_hash = mix(v_hash, vv);
          x_l2 += xv * xv;
          v_l2 += vv * vv;
        }
      constexpr int tracked_particle = 84897;
      const TV& tracked_x = x[tracked_particle];
      const TV& tracked_v = v[tracked_particle];
      ZIRAN_INFO("Fluid state fingerprint: phase=", phase,
          " count=", fluid.particles.count, " x_hash=", x_hash,
          " v_hash=", v_hash, " x_l2=", std::sqrt(x_l2),
          " v_l2=", std::sqrt(v_l2), " tracked=", tracked_particle,
          " x=", tracked_x.transpose(), " v=", tracked_v.transpose());
    };

    // xX0 is the solid result that is actually written for the total-
    // Lagrangian example: it visualizes the X0 attribute, not particles.X.
    // Keep this probe separate from the fluid one so a CPU/GPU restart can
    // distinguish (a) a different post-coupling grid velocity from (b) a
    // later G2P/X0 writeback error. It is intentionally opt-in, as hashing
    // the entire solid state is a diagnostic-only host pass.
    const bool solid_state_fingerprint =
        std::getenv("ZIRAN_SOLID_STATE_FINGERPRINT") != nullptr;
    auto solid_mix = [](std::uint64_t hash, double value) {
      const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
      for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
        hash ^= static_cast<std::uint64_t>(bytes[byte]);
        hash *= 1099511628211ull;
      }
      return hash;
    };
    auto fingerprint_solid_particles = [&](const char* phase) {
      if (!solid_state_fingerprint)
        return;
      auto& solid = e1.category == MATERIAL_PHASE_SOLID ? e1 : e2;
      const auto* x0_ptr =
          solid.particles.DataManager::getPointer(X0_name<TV>());
      if (x0_ptr == nullptr ||
          x0_ptr->size() != static_cast<std::size_t>(solid.particles.count)) {
        ZIRAN_INFO("Solid X0 fingerprint: phase=", phase,
                   " unavailable=1 x0_size=",
                   x0_ptr == nullptr ? 0 : x0_ptr->size(),
                   " count=", solid.particles.count);
        return;
      }
      const auto& x0 = *x0_ptr;
      std::uint64_t x0_hash = 1469598103934665603ull;
      std::uint64_t v_hash = x0_hash;
      double x0_l2 = 0;
      double v_l2 = 0;
      for (int p = 0; p < solid.particles.count; ++p)
        for (int d = 0; d < dim; ++d) {
          const double x0v = static_cast<double>(x0[p][d]);
          const double vv = static_cast<double>(solid.particles.V.array[p][d]);
          x0_hash = solid_mix(x0_hash, x0v);
          v_hash = solid_mix(v_hash, vv);
          x0_l2 += x0v * x0v;
          v_l2 += vv * vv;
        }
      const int tracked_particle =
          std::min(20094, std::max(0, solid.particles.count - 1));
      ZIRAN_INFO("Solid X0 fingerprint: phase=", phase,
          " count=", solid.particles.count, " x0_hash=", x0_hash,
          " v_hash=", v_hash, " x0_l2=", std::sqrt(x0_l2),
          " v_l2=", std::sqrt(v_l2), " tracked=", tracked_particle,
          " x0=", x0[tracked_particle].transpose(),
          " v=", solid.particles.V.array[tracked_particle].transpose());
    };
    auto fingerprint_solid_postsolve_velocity = [&]() {
      if (!solid_state_fingerprint)
        return;
      auto& solid = e1.category == MATERIAL_PHASE_SOLID ? e1 : e2;
      std::uint64_t solved_v_hash = 1469598103934665603ull;
      double solved_v_l2 = 0;
      for (int i = 0; i < solid.solved_v.rows(); ++i) {
        const double value = static_cast<double>(solid.solved_v(i));
        solved_v_hash = solid_mix(solved_v_hash, value);
        solved_v_l2 += value * value;
      }

      double grid_new_v_l2 = 0;
      TV grid_new_v_sum = TV::Zero();
      solid.grid.iterateGridSerial([&](IV, GridState<T, dim>& g) {
        grid_new_v_l2 += g.new_v.squaredNorm();
        grid_new_v_sum += g.new_v;
      });

      const int tracked_particle =
          std::min(20094, std::max(0, solid.particles.count - 1));
      const TV& xp = solid.particles.X.array[tracked_particle];
      BSplineWeights<T, dim> spline(xp, solid.dx);
      const uint64_t offset =
          MpmSimulationBase<T, dim>::SparseMask::Linear_Offset(
              to_std_array(spline.base_node));
      TV sampled_new_v = TV::Zero();
      solid.grid.iterateKernel(
          spline, offset, [&](IV, T w, TV, GridState<T, dim>& g) {
            sampled_new_v += w * g.new_v;
          });
      ZIRAN_INFO("Solid postsolve velocity fingerprint: solved_v_rows=",
          solid.solved_v.rows(), " solved_v_hash=", solved_v_hash,
          " solved_v_l2=", std::sqrt(solved_v_l2),
          " grid_new_v_l2=", std::sqrt(grid_new_v_l2),
          " grid_new_v_sum=", grid_new_v_sum.transpose(),
          " tracked=", tracked_particle, " reference_x=", xp.transpose(),
          " sampled_new_v=", sampled_new_v.transpose());
    };
    auto fingerprint_solid_grid_update = [&]() {
      if (!solid_state_fingerprint)
        return;
      auto& solid = e1.category == MATERIAL_PHASE_SOLID ? e1 : e2;
      double grid_v_l2 = 0;
      double grid_new_v_l2 = 0;
      TV grid_v_sum = TV::Zero();
      TV grid_new_v_sum = TV::Zero();
      std::uint64_t grid_v_hash = 1469598103934665603ull;
      std::uint64_t grid_new_v_hash = grid_v_hash;
      solid.grid.iterateGridSerial([&](IV, GridState<T, dim>& g) {
        grid_v_l2 += g.v.squaredNorm();
        grid_new_v_l2 += g.new_v.squaredNorm();
        grid_v_sum += g.v;
        grid_new_v_sum += g.new_v;
        for (int d = 0; d < dim; ++d) {
          grid_v_hash = solid_mix(grid_v_hash, static_cast<double>(g.v[d]));
          grid_new_v_hash =
              solid_mix(grid_new_v_hash, static_cast<double>(g.new_v[d]));
        }
      });
      const int tracked_particle =
          std::min(20094, std::max(0, solid.particles.count - 1));
      const TV& xp = solid.particles.X.array[tracked_particle];
      BSplineWeights<T, dim> spline(xp, solid.dx);
      const uint64_t offset =
          MpmSimulationBase<T, dim>::SparseMask::Linear_Offset(
              to_std_array(spline.base_node));
      TV sampled_v = TV::Zero();
      TV sampled_new_v = TV::Zero();
      solid.grid.iterateKernel(
          spline, offset, [&](IV, T w, TV, GridState<T, dim>& g) {
            sampled_v += w * g.v;
            sampled_new_v += w * g.new_v;
          });
      ZIRAN_INFO("Solid gridUpdate fingerprint: grid_v_l2=",
          std::sqrt(grid_v_l2), " grid_v_sum=", grid_v_sum.transpose(),
          " grid_v_hash=", grid_v_hash,
          " grid_new_v_l2=", std::sqrt(grid_new_v_l2),
          " grid_new_v_sum=", grid_new_v_sum.transpose(),
          " grid_new_v_hash=", grid_new_v_hash,
          " tracked=", tracked_particle, " reference_x=", xp.transpose(),
          " sampled_v=", sampled_v.transpose(),
          " sampled_new_v=", sampled_new_v.transpose());
    };
    fingerprint_fluid_state("substep_input", false);
    fingerprint_solid_particles("substep_input");

    ZIRAN_TIMER();
    auto& fluid_phase = e1.category == MATERIAL_PHASE_FLUID ? e1 : e2;
    auto& solid_phase = e1.category == MATERIAL_PHASE_SOLID ? e1 : e2;
    e1.gpu_full_pipeline_strict = gpu_full_pipeline_strict;
    e2.gpu_full_pipeline_strict = gpu_full_pipeline_strict;
    solid_phase.gpu_solid_device_resident = gpu_full_pipeline &&
        std::getenv("ZIRAN_GPU_SOLID_RESIDENT_DISABLE") == nullptr;
    solid_phase.gpu_build_system = gpu_build_system;
    e1.gpu_solid_implicit_probe = gpu_solid_implicit_probe;
    e2.gpu_solid_implicit_probe = gpu_solid_implicit_probe;
    e1.gpu_solid_implicit_pcg = gpu_solid_implicit_pcg;
    e2.gpu_solid_implicit_pcg = gpu_solid_implicit_pcg;
    e1.gpu_solid_implicit_state = gpu_solid_implicit_state;
    e2.gpu_solid_implicit_state = gpu_solid_implicit_state;
    bool solid_owner_ready = false;
    if (solid_phase.gpu_solid_device_resident)
      solid_owner_ready =
          solid_phase.initializePersistentSolidParticleOwner(false);
    if (gpu_full_pipeline_strict) {
      ZIRAN_ASSERT(solid_owner_ready,
          "strict full-GPU solid particle owner is unavailable");
    }
    e1.reinitialize();
    e2.reinitialize();

    bool bypass_legacy_fluid_pregrid = false;
    if constexpr (dim == 3) {
      const bool supported_fluid_geometry =
          fluid_phase.gpu_box_shell_collision.valid ||
          fluid_phase.collision_objects.empty();
      bypass_legacy_fluid_pregrid = gpu_build_system &&
          !gpu_build_system_cpu_numerical_grid &&
          !gpu_build_system_cpu_pressure_grid &&
          !gpu_build_system_cpu_velocity_grid &&
          gpu_fluid_velocity_transfer && fluid_phase.fluid_Q1Q0 &&
          supported_fluid_geometry;
    }
    if (gpu_full_pipeline_strict) {
      ZIRAN_ASSERT(bypass_legacy_fluid_pregrid,
          "strict full-GPU fluid P2G/grid update fell back to CPU");
    }
    if (bypass_legacy_fluid_pregrid) {
      // The Q1 coupling path reconstructs its analytical quadrature and
      // velocity grid from the persistent fluid state.  The legacy fluid
      // MPM grid is not consumed by buildSystem(), and its Base G2P result is
      // overwritten by the CUDA Q1 transfer below.  Stop rebuilding that
      // redundant host grid so the next step can make the resident particle
      // state authoritative without a pre-P2G host synchronization.
      solid_phase.particlesToGrid();
      ZIRAN_INFO("Bypassing legacy fluid P2G/gridUpdate: resident Q1 path=1");
    } else {
      e1.particlesToGrid();
      e2.particlesToGrid();
    }

    {
      // These were previously part of advanceOneTimeStep's unattributed
      // time.  Keep the two phases together: only the solid phase can
      // actually contribute a penalty spring in a coupling scene.
      ZIRAN_NAMED_TIMER("couplingBoundarySpring");
    if (e1.category != MATERIAL_PHASE_FLUID) addBoundarySpring(e1);
    if (e2.category != MATERIAL_PHASE_FLUID) addBoundarySpring(e2);
    }
    // if (e1.category != MATERIAL_PHASE_FLUID)
    //   addBoundarySpringSTICKY(e1);
    // if (e2.category != MATERIAL_PHASE_FLUID)
    //   addBoundarySpringSTICKY(e2);

    // buildSystem() propagates these too, but gridUpdate precedes the first
    // buildSystem after a restart.  Set them here so the first BE substep
    // receives the same device-PCG policy as all later substeps.
    if (bypass_legacy_fluid_pregrid)
      solid_phase.gridUpdate();
    else {
      e1.gridUpdate();
      e2.gridUpdate();
    }
    if (gpu_full_pipeline_strict) {
      ZIRAN_ASSERT(solid_phase.gpu_solid_device_grid_active &&
                       solid_phase.gpu_solid_final_grid_device_ready,
          "strict full-GPU solid implicit solve/writeback fell back to CPU");
    }
    fingerprint_solid_grid_update();

    ZIRAN_ASSERT(e1.use_vp && e2.use_vp, "only solve coupling system here");
    const bool has_fluid_particles =
        (e1.category == MATERIAL_PHASE_FLUID ? e1 : e2).particles.count != 0;
    if (has_fluid_particles) {
      buildSystem();
      solveSystem();
      fingerprint_fluid_state("post_coupling_override", true);
      fingerprint_solid_postsolve_velocity();
    } else {
      // A few coupling examples deliberately inject their fluid phase from a
      // later frame callback.  Before that injection there is no fluid
      // pressure/interface constraint, so the physically correct operation is
      // the already-completed solid grid update followed by solid G2P.  In
      // particular, do not send the resulting zero-nnz coupling system to
      // AMGX just to preserve the normal pipeline shape.
      e1.gpu_solved_p_valid = false;
      e2.gpu_solved_p_valid = false;
      // Do not let a pressure override from an earlier, now-empty phase be
      // reused by solid G2P.  SplittingSimulation then uses det(F).
      e1.override_J.clear();
      e2.override_J.clear();
#ifdef ENABLE_CUDA_STAGE2_PROBE
      if constexpr (dim == 3) {
        if (solid_phase.gpu_solid_device_resident &&
            solid_phase.gpu_solid_final_grid_device_ready &&
            std::getenv("ZIRAN_GPU_SOLID_PARTICLE_OWNER_ORACLE") == nullptr) {
          GPU_STAGE3A::SolidParticleStateStats transfer_stats;
          GPU_STAGE3A::updatePersistentSolidParticlesQ2FromGrid3D(
              solid_phase.gpu_solid_particle_state,
              solid_phase.gpu_solid_p2g_node_map,
              static_cast<double>(solid_phase.dx), static_cast<double>(dt),
              static_cast<double>(solid_phase.apic_rpic_ratio),
              static_cast<double>(solid_phase.D_inverse), &transfer_stats);
          ZIRAN_ASSERT(transfer_stats.missing_velocity_nodes == 0,
              "solid-only resident Q2 velocity lookup failed");
          solid_phase.gpu_solid_particle_g2p_ready = true;
          solid_phase.gpu_solid_host_current_x_valid = false;
          solid_phase.gpu_solid_host_dynamic_state_valid = false;
          ZIRAN_INFO("GPU solid-only Q2 transfer: particles=",
              transfer_stats.particles, " J=det(F) vectors=device");
        }
      }
#endif
      static bool reported_empty_fluid_coupling_skip = false;
      if (!reported_empty_fluid_coupling_skip) {
        reported_empty_fluid_coupling_skip = true;
        ZIRAN_INFO("Skipping coupling solve and PositionCorrector: fluid "
                   "phase has no particles yet");
      }
    }

    {
      ZIRAN_NAMED_TIMER("couplingCollisionStateUpdate");
      for (size_t k = 0; k < e1.collision_objects.size(); ++k)
        if (e1.collision_objects[k]->updateState)
          e1.collision_objects[k]->updateState(Base::step.time + dt,
                                               *e1.collision_objects[k]);
      for (size_t k = 0; k < e2.collision_objects.size(); ++k)
        if (e2.collision_objects[k]->updateState)
          e2.collision_objects[k]->updateState(Base::step.time + dt,
                                               *e2.collision_objects[k]);
    }

    {
      ZIRAN_NAMED_TIMER("couplingFluidGridToParticles");
      (e1.category == MATERIAL_PHASE_FLUID ? e1 : e2).gridToParticles(dt);
    }
    {
      ZIRAN_NAMED_TIMER("couplingSolidGridToParticles");
      (e1.category == MATERIAL_PHASE_SOLID ? e1 : e2).gridToParticles(dt);
    }
    fingerprint_fluid_state("post_g2p", false);
    fingerprint_solid_particles("post_g2p");
    if (const char* requested_stop =
            std::getenv("ZIRAN_STOP_AFTER_FIRST_COUPLING_SUBSTEP")) {
      // `=1` retains the original single-step behavior. A larger positive
      // value gives a bounded CPU/GPU bisection across successive CFL
      // substeps without having to complete the output frame.
      static int completed_diagnostic_substeps = 0;
      const long requested_count = std::max(
          1L, std::strtol(requested_stop, nullptr, 10));
      ++completed_diagnostic_substeps;
      if (completed_diagnostic_substeps >= requested_count) {
        ZIRAN_INFO("Stopping after coupling diagnostic substep ",
                   completed_diagnostic_substeps, " of ", requested_count);
        throw std::runtime_error("requested coupling substep diagnostic stop");
      }
    }

    const bool disable_position_corrector_for_probe =
        std::getenv("ZIRAN_DISABLE_POSITION_CORRECTOR") != nullptr;
    if (has_fluid_particles && use_position_correction &&
        !disable_position_corrector_for_probe) {
      if (e1.category == MATERIAL_PHASE_FLUID) {
        position_corrector1.solve(e2, gpu_stage2_probe,
                                  gpu_position_corrector,
                                  gpu_position_corrector_density,
                                  gpu_position_corrector_pages,
                                  gpu_position_corrector_direct_topology_probe,
                                  gpu_position_corrector_direct_matrix_probe,
                                  gpu_position_corrector_direct_solver_probe,
                                  gpu_position_corrector_direct_legacy_order_probe,
                                  gpu_position_corrector_direct_legacy_solver_probe,
                                  gpu_position_corrector_direct_legacy_solver_writeback,
                                  gpu_position_corrector_direct_legacy_solver,
                                  gpu_position_corrector_direct_gpu_solver,
                                  gpu_position_corrector_direct_amgx_solver,
                                  gpu_position_corrector_direct_postsolve,
                                  gpu_position_corrector_direct_pushout);
        {
          ZIRAN_NAMED_TIMER("positionCorrectorMarchingSquare");
          position_corrector1.levelset_builder.marchingSquare(points, segments);
        }
      }
      if (e2.category == MATERIAL_PHASE_FLUID) {
        position_corrector2.solve(e1, gpu_stage2_probe,
                                  gpu_position_corrector,
                                  gpu_position_corrector_density,
                                  gpu_position_corrector_pages,
                                  gpu_position_corrector_direct_topology_probe,
                                  gpu_position_corrector_direct_matrix_probe,
                                  gpu_position_corrector_direct_solver_probe,
                                  gpu_position_corrector_direct_legacy_order_probe,
                                  gpu_position_corrector_direct_legacy_solver_probe,
                                  gpu_position_corrector_direct_legacy_solver_writeback,
                                  gpu_position_corrector_direct_legacy_solver,
                                  gpu_position_corrector_direct_gpu_solver,
                                  gpu_position_corrector_direct_amgx_solver,
                                  gpu_position_corrector_direct_postsolve,
                                  gpu_position_corrector_direct_pushout);
        {
          ZIRAN_NAMED_TIMER("positionCorrectorMarchingSquare");
          position_corrector2.levelset_builder.marchingSquare(points, segments);
        }
      }
    }
    else if (has_fluid_particles && use_position_correction &&
             disable_position_corrector_for_probe)
      ZIRAN_INFO("PositionCorrector disabled by ZIRAN_DISABLE_POSITION_CORRECTOR");
    fingerprint_fluid_state("post_position_corrector", false);

    {
      // This is intentionally a separate timer from G2P.  `remove()` moves
      // every registered particle attribute, so a nonzero out-of-domain
      // count can otherwise look like an unexplained coupling cost.
      ZIRAN_NAMED_TIMER("couplingParticleCull");
      SplittingSimulation<T, dim>& fluid_simulation =
          e1.category == MATERIAL_PHASE_FLUID ? e1 : e2;
      const int fluid_count_before_cull = fluid_simulation.particles.count;
      bool skip_fluid_host_cull = false;
#ifdef ENABLE_CUDA_STAGE2_PROBE
      if constexpr (dim == 3) {
        GPU_STAGE2::FluidParticleStateStats cull_stats;
        if (fluid_simulation.reduceResidentFluidParticleState(
                5.0 * static_cast<double>(fluid_simulation.dx),
                2028.0 * static_cast<double>(fluid_simulation.dx),
                cull_stats)) {
          skip_fluid_host_cull = cull_stats.outside_particles == 0;
          ZIRAN_INFO("Fluid resident cull check: particles=",
                     cull_stats.particles, " outside=",
                     cull_stats.outside_particles, " reduction_ms=",
                     cull_stats.reduction_ms);
          if (!skip_fluid_host_cull) {
            ZIRAN_ASSERT(!gpu_full_pipeline_strict,
                "strict full-GPU fluid cull requires host compaction");
            fluid_simulation.synchronizeFluidCompatibilityForHostOutput();
          }
        }
      }
#endif
      auto cull_host_particles = [&](SplittingSimulation<T, dim>& simulation) {
        if (&simulation == &fluid_simulation && skip_fluid_host_cull)
          return;
        bool has_outside_particle = false;
        for (int i = simulation.particles.count - 1;
             i >= 0 && !has_outside_particle; --i) {
          for (int d = 0; d < dim; ++d) {
            if (simulation.particles.X.array[i](d) < 5 * simulation.dx ||
                simulation.particles.X.array[i](d) >
                    2028. * simulation.dx) {
              has_outside_particle = true;
              break;
            }
          }
        }
#ifdef ENABLE_CUDA_STAGE2_PROBE
        // DataManager::remove() compacts every registered attribute.  If the
        // solid owner is authoritative, first materialize all stale dynamic
        // attributes so compaction cannot mix a fresh device state with stale
        // host arrays.  The particle count changes below, so the old owner
        // must then be invalidated and rebuilt from the compacted host state.
        if (has_outside_particle &&
            simulation.category == MATERIAL_PHASE_SOLID &&
            GPU_STAGE3A::persistentSolidParticleStateValid(
                simulation.gpu_solid_particle_state,
                static_cast<std::size_t>(simulation.particles.count))) {
          simulation.synchronizeSolidCompatibilityForHostOutput();
          GPU_STAGE3A::invalidatePersistentSolidParticleState3D(
              simulation.gpu_solid_particle_state);
          simulation.gpu_solid_p2g_node_map = {};
          simulation.gpu_solid_p2g_indices_ready = false;
          simulation.gpu_solid_legacy_q2_topology_seed.clear();
          simulation.gpu_solid_device_grid_active = false;
          simulation.gpu_solid_particle_g2p_ready = false;
        }
#endif
        for (int i = simulation.particles.count - 1; i >= 0; --i) {
          for (int d = 0; d < dim; ++d) {
            if (simulation.particles.X.array[i](d) < 5 * simulation.dx ||
                simulation.particles.X.array[i](d) > 2028. * simulation.dx) {
              simulation.particles.remove(i);
              break;
            }
          }
        }
      };
      cull_host_particles(e1);
      cull_host_particles(e2);
      if (fluid_simulation.particles.count != fluid_count_before_cull)
        fluid_simulation.gpu_velocity_transfer_device_x_valid = false;
      if (fluid_simulation.particles.count != fluid_count_before_cull)
        fluid_simulation.gpu_velocity_transfer_device_v_grad_valid = false;
    }
  }

  void write(const std::string &filename) override {
    e1.setFrame(frame);
    e2.setFrame(frame);
    e1.write(e1.outputFileName());
    e2.write(e2.outputFileName());
    return;
    visualize_points(e1, before_advect, trajectory, "trajectory");
    visualize_points(e1, solid_particles, normals, "solid_normal");
    visualize_segments(e2, points, segments, "marching_square");
    visualize_points(e1, wall_particles, "wall_points");
    visualize_g2p_vec(e1, surface_tension_delta1, "st");
    visualize_g2p_vec(e2, surface_tension_delta2, "st");
    // if (e1.quad_particles.size()) visualize_points(e1, e1.quad_particles,
    // "quad_particles"); if (e2.quad_particles.size()) visualize_points(e2,
    // e2.quad_particles, "quad_particles");
  }

  void writeFrameOutput() override {
    e1.setFrame(frame);
    e2.setFrame(frame);
    e1.writeFrameOutput();
    e2.writeFrameOutput();
  }

  void read(const std::string &filename) override {
    e1.setFrame(frame);
    e2.setFrame(frame);
    e1.read(e1.outputFileName());
    e2.read(e2.outputFileName());
  }

  void writeState(std::ostream &) override {}
  void readState(std::istream &) override {}
  bool useDouble() override { return std::is_same<T, double>::value; }
  int dimension() override { return dim; }
  const char *name() override { return "coupling"; }
};
} // namespace ZIRAN
