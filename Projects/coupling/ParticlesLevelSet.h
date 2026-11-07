#pragma once

#include "../splitting/GridIteration.h"
#include "../splitting/SparseIndexGrid.h"
#include "GpuParticleLevelSet.h"
#include <MPM/Grid/MpmGrid.h>
#include <MPM/Simulation/MpmSimulationBase.h>
#include <algorithm>
#include <utility>
#include <queue>

namespace ZIRAN {

template <class T, int dim>
class ParticlesLevelSet {
public:
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;

    std::priority_queue<std::pair<T, uint64_t>> pq;
    MpmGrid<T, dim, 1> phi_grid;
    T h;
    GPU_POSITION_CORRECTOR::PersistentParticleLevelSet3D gpu_context;
    GPU_POSITION_CORRECTOR::DenseLevelSet3D gpu_result;
    std::vector<GPU_POSITION_CORRECTOR::Vec3d> gpu_positions;
    bool gpu_result_active = false;
    bool gpu_host_result_valid = false;

    ParticlesLevelSet() = default;

    void insertHeap(IV node)
    {
        if (phi_grid[node].m > 0)
            return;
        T min_value[dim];
        for (int d = 0; d < dim; ++d) {
            min_value[d] = std::numeric_limits<T>::max();
            for (int delta = -1; delta <= 1; delta += 2) {
                IV new_node = node;
                new_node[d] += delta;
                if (phi_grid[new_node].m > 0)
                    min_value[d] = std::min(min_value[d], phi_grid[new_node].v[0]);
            }
        }
        std::sort(min_value, min_value + dim);
        if (min_value[0] > std::numeric_limits<T>::max() / (T)2)
            return;
        T result = std::numeric_limits<T>::max();
        if (dim == 2) {
            T a = min_value[0], b = min_value[1];
            result = std::min(result, a + h);
            if (a + h > b) {
                T ab = ((a + b) + std::sqrt(2.0 * h * h - (a - b) * (a - b))) / 2.0;
                result = std::min(result, ab);
            }
        }
        if (dim == 3) {
            T a = min_value[0], b = min_value[1], c = min_value[2];
            result = std::min(result, a + h);
            if (a + h > b) {
                T ab = ((a + b) + std::sqrt(2.0 * h * h - (a - b) * (a - b))) / 2.0;
                result = std::min(result, ab);
                if (ab > c) {
                    T abc = 1.0 / 6.0 * (std::sqrt((-2.0 * a - 2.0 * b - 2.0 * c) * (-2.0 * a - 2.0 * b - 2.0 * c) - 12.0 * (a * a + b * b + c * c - h * h)) + 2.0 * a + 2.0 * b + 2.0 * c);
                    result = std::min(result, abc);
                }
            }
        }
        GridState<T, dim>& state = phi_grid[node];
        if (state.idx == -1) {
            state.idx = -2;
            state.v[1] = std::numeric_limits<T>::max();
        }
        if (result >= state.v[1])
            return;
        state.v[1] = result;
        uint64_t offset = MpmSimulationBase<T, dim>::SparseMask::Linear_Offset(to_std_array(node));
        pq.push(std::make_pair(-result, offset));
    }

    void build(SplittingSimulation<T, dim>& Ssim, T build_dx, T blob_radius,
        T erosion, bool use_gpu = false,
        const void* resident_positions = nullptr,
        std::size_t resident_position_count = 0,
        bool download_gpu_result = true)
    {
        ZIRAN_TIMER();
        h = build_dx;
        gpu_result_active = false;
        gpu_host_result_valid = false;
        int kernel_size = ((int)std::ceil(blob_radius / h) + 1) * 2;
        StdVector<TV>& Xarray = (Ssim.config_scheme != CONFIG_UPDATE_LAGRANGIAN) ? Ssim.particles.DataManager::get(X0_name<TV>()).array : Ssim.particles.X.array;
        // StdVector<TV>& Xarray = Ssim.particles.X.array;

        if constexpr (dim == 3) {
            if (use_gpu && GPU_POSITION_CORRECTOR::particleLevelSetCudaAvailable()) {
                GPU_POSITION_CORRECTOR::ParticleLevelSetStats stats;
                {
                    ScopedTimer timer("levelsetGpuBuild", __FILE__, __LINE__, true);
                    if (!resident_positions) {
                        gpu_positions.resize(Xarray.size());
                        for (std::size_t i = 0; i < Xarray.size(); ++i) {
                            gpu_positions[i] = GPU_POSITION_CORRECTOR::Vec3d {
                                static_cast<double>(Xarray[i][0]),
                                static_cast<double>(Xarray[i][1]),
                                static_cast<double>(Xarray[i][2])
                            };
                        }
                    }
                    else {
                        gpu_positions.clear();
                    }
                    const bool built = GPU_POSITION_CORRECTOR::buildParticleLevelSet3D(
                        gpu_context, gpu_positions, static_cast<double>(h),
                        static_cast<double>(blob_radius), static_cast<double>(erosion),
                        2, gpu_result, &stats, resident_positions,
                        resident_position_count, download_gpu_result);
                    ZIRAN_ASSERT(built, "GPU particle level-set build failed");
                }
                gpu_result_active = true;
                gpu_host_result_valid = download_gpu_result;
                ZIRAN_INFO("GPU particle level set: particles=", stats.particles,
                    " voxels=", stats.voxels,
                    " kernel=", stats.kernel_size,
                    " cycles=", stats.sweep_cycles,
                    " resident_particles=",
                    stats.resident_particle_input ? 1 : 0,
                    " upload_ms=", stats.upload_ms,
                    " rasterize_ms=", stats.rasterize_ms,
                    " sweep_ms=", stats.sweep_ms,
                    " download_ms=", stats.download_ms,
                    " total_ms=", stats.total_ms);
                return;
            }
        }
        {
            ScopedTimer timer("levelsetPollute", __FILE__, __LINE__, true);
            phi_grid.activatePages(Xarray, h, -1, 1);
        }
        {
        ScopedTimer timer("levelsetRasterize", __FILE__, __LINE__, true);
        Ssim.parallel_for_updating_grid([&](int i) {
            auto& Xp = Xarray[i];
            BSplineWeights<T, dim, 1> spline(Xp, h);
            for (int d = 0; d < dim; ++d) {
                ZIRAN_ASSERT(spline.base_node[d] + kernel_size / 2 < phi_grid.spgrid_size, "Exceed SPGrid range");
            }
            IV region = IV::Ones() * kernel_size;
            iterateRegion(region, [&](const IV& offset) {
                IV node = spline.base_node + offset - IV::Ones() * (kernel_size / 2 - 1);
                TV pos = node.template cast<T>() * h;
                T phi = (Xp - pos).norm() - blob_radius;
                if (phi_grid[node].m > 0) {
                    phi_grid[node].v[0] = std::max(-phi, phi_grid[node].v[0]);
                }
                else {
                    phi_grid[node].v[0] = -phi;
                    phi_grid[node].m = 1;
                }
            });
        });
        }

        {
        ScopedTimer timer("levelsetSeed", __FILE__, __LINE__, true);
        int phi_dof = phi_grid.getNumNodes();
        StdVector<IV> ready_to_insert;
        phi_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
            if (g.v[0] > 0) {
                g.v[0] = 0;
                g.v[1] = std::numeric_limits<T>::max();
                g.m = 0;
                ready_to_insert.push_back(node);
            }
        });
        for (auto& node : ready_to_insert) {
            insertHeap(node);
        }
        }

        {
        ScopedTimer timer("levelsetFastMarch", __FILE__, __LINE__, true);
        while (!pq.empty()) {
            auto item = pq.top();
            pq.pop();
            auto coord = MpmSimulationBase<T, dim>::SparseMask::LinearToCoord(item.second);
            IV node;
            for (int d = 0; d < dim; ++d)
                node[d] = coord[d];
            if (phi_grid[node].m == 0) {
                const T result = -item.first;
                if (result != phi_grid[node].v[1])
                    continue;
                phi_grid[node].v[0] = result;
                phi_grid[node].m = 1;
                for (int d = 0; d < dim; ++d) {
                    for (int delta = -1; delta <= 1; delta += 2) {
                        IV new_node = node;
                        new_node[d] += delta;
                        insertHeap(new_node);
                    }
                }
            }
        }
        }
        {
        ScopedTimer timer("levelsetFinalize", __FILE__, __LINE__, true);
        phi_grid.iterateGrid([&](IV node, GridState<T, dim>& g) {
            g.v[0] = -g.v[0] + erosion;
        });
        }
    }

    bool queryInside(TV& particleX, T& phi, TV& normal, T isocontour)
    {
        BSplineWeights<T, dim, 1> spline(particleX, h);
        if constexpr (dim == 3) {
            if (gpu_result_active) {
                if (!gpu_host_result_valid) {
                    const bool copied =
                        GPU_POSITION_CORRECTOR::copyParticleLevelSetToHost3D(
                            gpu_context, gpu_result);
                    ZIRAN_ASSERT(copied,
                        "GPU particle level-set compatibility download failed");
                    gpu_host_result_valid = true;
                    ZIRAN_INFO("GPU particle level set lazy host sync: voxels=",
                        gpu_result.phi.size(), " bytes=",
                        gpu_result.phi.size() * sizeof(double));
                }
                const int ox = gpu_result.origin[0];
                const int oy = gpu_result.origin[1];
                const int oz = gpu_result.origin[2];
                const int sx = gpu_result.shape[0];
                const int sy = gpu_result.shape[1];
                const int sz = gpu_result.shape[2];
                const T invalid = std::numeric_limits<T>::max() / (T)2;
                phi = 0;
                normal = TV::Zero();
                for (int i = 0; i < 2; ++i) {
                    const T wi = spline.w[0](i);
                    const T dwidxi = spline.one_over_dx * spline.dw[0](i);
                    const int x = spline.base_node[0] + i - ox;
                    for (int j = 0; j < 2; ++j) {
                        const T wj = spline.w[1](j);
                        const T wij = wi * wj;
                        const T dwijdxi = dwidxi * wj;
                        const T dwijdxj = wi * spline.one_over_dx * spline.dw[1](j);
                        const int y = spline.base_node[1] + j - oy;
                        for (int k = 0; k < 2; ++k) {
                            const int z = spline.base_node[2] + k - oz;
                            if (x < 0 || x >= sx || y < 0 || y >= sy || z < 0 || z >= sz)
                                return false;
                            const std::size_t index =
                                (static_cast<std::size_t>(z) * sy
                                    + static_cast<std::size_t>(y))
                                    * sx
                                + static_cast<std::size_t>(x);
                            const T value = static_cast<T>(gpu_result.phi[index]);
                            if (value >= invalid)
                                return false;
                            const T wk = spline.w[2](k);
                            const T wijk = wij * wk;
                            const T wijkdxi = dwijdxi * wk;
                            const T wijkdxj = dwijdxj * wk;
                            const T wijkdxk = wij * spline.one_over_dx * spline.dw[2](k);
                            phi += wijk * value;
                            normal += TV { wijkdxi, wijkdxj, wijkdxk } * value;
                        }
                    }
                }
                normal.normalize();
                return phi < isocontour;
            }
        }
        uint64_t offset = MpmSimulationBase<T, dim>::SparseMask::Linear_Offset(to_std_array(spline.base_node));
        int cnt = 0;
        phi_grid.iterateKernelWithValidation(spline, offset, [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
            if (g.m > 0) ++cnt;
        });
        if (cnt != (1 << dim)) return false;
        phi = 0;
        normal = TV::Zero();
        phi_grid.iterateKernel(spline, offset, [&](const IV& node, T w, const TV& dw, GridState<T, dim>& g) {
            phi += w * g.v[0];
            normal += dw * g.v[0];
        });
        normal.normalize();
        return phi < isocontour;
    }

    void marchingSquare(StdVector<TV>& points, StdVector<Vector<int, 2>>& segments)
    {
        if constexpr (dim == 2) {
            points.clear();
            segments.clear();
            int count = 0;
            int phi_dof = phi_grid.getNumNodes();
            SparseIndexGrid<T, dim> cell_phi;
            cell_phi.clear();
            phi_grid.iterateGridSerial([&](IV node, GridState<T, dim>& g) {
                cell_phi(node[0], node[1]) = g.v[0];
            });
            T dx = h;
            cell_phi.iterateGrid([&](int x, int y, T value) {
                int idx_a = -1;
                int idx_b = -1;
                int idx_c = -1;
                int idx_d = -1;
                if (cell_phi.find(x + 1, y) && cell_phi.find(x, y + 1) && cell_phi.find(x + 1, y + 1)) {
                    if (cell_phi(x, y) * cell_phi(x + 1, y) < 0) {
                        TV shift = TV(std::abs(cell_phi(x, y) / (cell_phi(x, y) - cell_phi(x + 1, y))) * dx, 0);
                        TV pos = TV(x, y) * dx + shift;
                        points.push_back(pos);
                        idx_a = count;
                        count += 1;
                    }
                    if (cell_phi(x + 1, y) * cell_phi(x + 1, y + 1) < 0) {
                        TV shift = TV(0, std::abs(cell_phi(x + 1, y) / (cell_phi(x + 1, y) - cell_phi(x + 1, y + 1))) * dx);
                        TV pos = TV(x + 1, y) * dx + shift;
                        points.push_back(pos);
                        idx_b = count;
                        count += 1;
                    }
                    if (cell_phi(x, y + 1) * cell_phi(x + 1, y + 1) < 0) {
                        TV shift = TV(std::abs(cell_phi(x, y + 1) / (cell_phi(x, y + 1) - cell_phi(x + 1, y + 1))) * dx, 0);
                        TV pos = TV(x, y + 1) * dx + shift;
                        points.push_back(pos);
                        idx_c = count;
                        count += 1;
                    }
                    if (cell_phi(x, y) * cell_phi(x, y + 1) < 0) {
                        TV shift = TV(0, std::abs(cell_phi(x, y) / (cell_phi(x, y) - cell_phi(x, y + 1))) * dx);
                        TV pos = TV(x, y) * dx + shift;
                        points.push_back(pos);
                        idx_d = count;
                        count += 1;
                    }
                }
                if (idx_a != -1 && idx_b != -1) segments.push_back(Vector<int, 2>(idx_a, idx_b));
                if (idx_b != -1 && idx_c != -1) segments.push_back(Vector<int, 2>(idx_b, idx_c));
                if (idx_c != -1 && idx_d != -1) segments.push_back(Vector<int, 2>(idx_c, idx_d));
                if (idx_d != -1 && idx_a != -1) segments.push_back(Vector<int, 2>(idx_d, idx_a));
                if (idx_a != -1 && idx_c != -1) segments.push_back(Vector<int, 2>(idx_a, idx_c));
                if (idx_b != -1 && idx_d != -1) segments.push_back(Vector<int, 2>(idx_b, idx_d));
            });
        }
    }
};

} // namespace ZIRAN
