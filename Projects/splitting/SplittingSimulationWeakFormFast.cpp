#include <chrono>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <tbb/parallel_sort.h>
#include "GridIteration.h"
#include "SplittingSimulation.h"
#include "../coupling/GpuOperatorProbe.h"

namespace ZIRAN {

template <class T, int dim>
void SplittingSimulation<T, dim>::rebuildInterfaceQuad() {
  ZIRAN_TIMER();
  gpu_interface_quad_device_ready = false;
  gpu_interface_fields = {};
  gpu_interface_quad_host_fields_valid = true;

  StdVector<T> SVQ_mass = particles.mass.array;
  const StdVector<TV> &SVQ_v = particles.V.array;
  // const StdVector<TM>& SVQ_C =
  // particles.DataManager::get(C_name<TM>()).array;

  const std::size_t solid_count = gpu_interface_solid_particles.size != 0
      ? gpu_interface_solid_particles.size : solid_positions.size();
  StdVector<T> AVQ_mass(solid_count, 0);
  StdVector<TV> AVQ_v(solid_count, TV::Zero());
  // StdVector<TM> AVQ_C(solid_positions.size(), TM::Zero());

  T radius = Base::dx * 0.5;
  bool gpu_interface_quad_used = false;
#ifdef ENABLE_CUDA_STAGE2_PROBE
  if constexpr (dim == 3) {
    // The coupling coordinator has already rebuilt this persistent solid
    // cell list for the current substep. Reuse it for the full fluid->solid
    // quadrature distribution instead of merely asking its old boolean
    // "any neighbour" query.
    if (gpu_interface_quad && gpu_interface_solid_search &&
        (!gpu_skip_interface_neighbor_search || gpu_fluid_device_resident)) {
      const bool resident_fluid_positions =
          gpu_velocity_transfer_device_x_valid &&
          gpu_velocity_transfer_device.x != nullptr &&
          gpu_velocity_transfer_device.size ==
              static_cast<std::size_t>(particles.count);
      const bool resident_fluid_velocity =
          gpu_velocity_transfer_device_v_grad_valid &&
          gpu_velocity_transfer_device.v != nullptr &&
          gpu_velocity_transfer_device.size ==
              static_cast<std::size_t>(particles.count);
      std::vector<GPU_STAGE2::Vec3d> fluid_positions;
      std::vector<double> fluid_mass;
      std::vector<GPU_STAGE2::Vec3d> fluid_velocity;
      if (!resident_fluid_positions)
        fluid_positions.reserve(particles.count);
      fluid_mass.reserve(particles.count);
      if (!resident_fluid_velocity)
        fluid_velocity.reserve(particles.count);
      for (int i = 0; i < particles.count; ++i) {
        if (!resident_fluid_positions) {
          const TV& x = particles.X.array[i];
          fluid_positions.push_back(GPU_STAGE2::Vec3d{
              static_cast<double>(x[0]), static_cast<double>(x[1]),
              static_cast<double>(x[2])});
        }
        fluid_mass.push_back(static_cast<double>(SVQ_mass[i]));
        if (!resident_fluid_velocity) {
          const TV& v = SVQ_v[i];
          fluid_velocity.push_back(GPU_STAGE2::Vec3d{
              static_cast<double>(v[0]), static_cast<double>(v[1]),
              static_cast<double>(v[2])});
        }
      }
      GPU_STAGE2::InterfaceQuadAtomicResult gpu_result;
      GPU_STAGE2::InterfaceQuadAtomicStats gpu_stats;
      const bool gpu_analytic_geometry_supported =
          gpu_box_shell_collision.valid || Base::collision_objects.empty();
      const bool compact_interface_host_result =
          gpu_build_system && !gpu_build_system_cpu_numerical_grid &&
          !gpu_build_system_cpu_velocity_grid && fluid_Q1Q0 &&
          gpu_analytic_geometry_supported &&
          std::getenv("ZIRAN_GPU_DISABLE_FLUID_ANALYTICAL") == nullptr &&
          std::getenv("ZIRAN_GPU_FLUID_VELOCITY_GRID_ORACLE") == nullptr;
      const bool resident_interface_fields_supported =
          compact_interface_host_result &&
          solid_scheme != CONFIG_UPDATE_LAGRANGIAN &&
          gpu_interface_solid_particles.rest_x != nullptr &&
          gpu_interface_solid_particles.volume != nullptr &&
          gpu_interface_solid_particles.size == solid_count;
      const bool compare_interface_fields =
          resident_interface_fields_supported &&
          std::getenv("ZIRAN_GPU_INTERFACE_FIELDS_COMPARE") != nullptr;
      const bool resident_interface_fields =
          resident_interface_fields_supported &&
          std::getenv("ZIRAN_GPU_INTERFACE_QUAD_HOST_RESULT") == nullptr &&
          std::getenv("ZIRAN_INTERFACE_QUAD_FIELD_DUMP_PREFIX") == nullptr &&
          !compare_interface_fields;
      {
        ZIRAN_NAMED_TIMER("gpuInterfaceQuadAtomic");
        GPU_STAGE2::rebuildInterfaceQuadAtomic3D(
            gpu_interface_quad_context, *gpu_interface_solid_search,
            fluid_positions, fluid_mass, fluid_velocity,
            static_cast<double>(Base::dx), gpu_box_shell_collision,
            gpu_result, &gpu_stats,
            resident_fluid_positions
                ? gpu_velocity_transfer_device.x : nullptr,
            resident_fluid_positions
                ? gpu_velocity_transfer_device.size : 0,
            resident_fluid_velocity
                ? gpu_velocity_transfer_device.v : nullptr,
            resident_fluid_velocity
                ? gpu_velocity_transfer_device.size : 0,
            compact_interface_host_result,
            /*materialize_host_result*/ !resident_interface_fields);
      }
      if (resident_interface_fields || compare_interface_fields) {
        ZIRAN_NAMED_TIMER("gpuInterfaceQuadDeviceFields");
        GPU_STAGE2::buildPersistentInterfaceQuadratureFields3D(
            gpu_interface_quad_context, *gpu_interface_solid_search,
            gpu_interface_solid_particles, gpu_interface_fields);
      }
      ZIRAN_ASSERT(gpu_result.solid_mass.size() ==
                      gpu_result.solid_momentum.size(),
          "GPU interface quadrature solid output size mismatch");
      {
        ZIRAN_NAMED_TIMER("gpuInterfaceQuadHostResult");
        if (!gpu_result.host_result_materialized) {
          ZIRAN_ASSERT(gpu_result.compact_host_result &&
                  gpu_result.active_solid_ids.empty() &&
                  gpu_result.solid_mass.empty(),
              "device-only interface quadrature unexpectedly returned host data");
        } else if (gpu_result.compact_host_result) {
          ZIRAN_ASSERT(gpu_result.fluid_mass.empty() &&
                  gpu_result.active_solid_ids.size() ==
                      gpu_result.solid_mass.size(),
              "GPU compact interface quadrature output size mismatch");
          tbb::parallel_for(std::size_t{0}, gpu_result.solid_mass.size(),
              [&](std::size_t active) {
                const int i = gpu_result.active_solid_ids[active];
                ZIRAN_ASSERT(i >= 0 &&
                        static_cast<std::size_t>(i) < AVQ_mass.size(),
                    "GPU compact interface quadrature solid id out of range");
                AVQ_mass[i] = static_cast<T>(gpu_result.solid_mass[active]);
                const auto& momentum = gpu_result.solid_momentum[active];
                AVQ_v[i] = TV(static_cast<T>(momentum.x),
                              static_cast<T>(momentum.y),
                              static_cast<T>(momentum.z));
              });
        } else {
          ZIRAN_ASSERT(gpu_result.fluid_mass.size() == SVQ_mass.size() &&
                  gpu_result.solid_mass.size() == AVQ_mass.size(),
              "GPU full interface quadrature output size mismatch");
          tbb::parallel_for(std::size_t{0}, SVQ_mass.size(),
              [&](std::size_t i) {
                SVQ_mass[i] = static_cast<T>(gpu_result.fluid_mass[i]);
              });
          tbb::parallel_for(std::size_t{0}, AVQ_mass.size(),
              [&](std::size_t i) {
                AVQ_mass[i] = static_cast<T>(gpu_result.solid_mass[i]);
                const auto& momentum = gpu_result.solid_momentum[i];
                AVQ_v[i] = TV(static_cast<T>(momentum.x),
                              static_cast<T>(momentum.y),
                              static_cast<T>(momentum.z));
              });
        }
      }
      gpu_interface_quad_used = true;
      gpu_interface_quad_device_ready = true;
      gpu_interface_quad_host_fields_valid =
          gpu_result.host_result_materialized;
      ZIRAN_INFO("GPU interface quadrature atomic: fluid=",
          gpu_stats.fluid_particles, " solid=", gpu_stats.solid_particles,
          " active_solid=", gpu_stats.active_solid_particles,
          " resident_fluid_positions=",
          gpu_stats.resident_fluid_positions ? 1 : 0,
          " resident_fluid_velocity=",
          gpu_stats.resident_fluid_velocity ? 1 : 0,
          " resident_fluid_mass=",
          gpu_stats.resident_fluid_mass ? 1 : 0,
          " compact_host_result=",
          gpu_result.compact_host_result ? 1 : 0,
          " host_result_materialized=",
          gpu_result.host_result_materialized ? 1 : 0,
          " resident_interface_fields=",
          (resident_interface_fields || compare_interface_fields)
              ? gpu_interface_fields.size : 0,
          " reallocations=", gpu_stats.reallocations,
          " fluid_h2d_ms=", gpu_stats.fluid_upload_ms,
          " fine_grid_ms=", gpu_stats.fine_grid_ms,
          " atomic_ms=", gpu_stats.accumulation_ms,
          " result_d2h_ms=", gpu_stats.download_ms,
          " total_ms=", gpu_stats.total_ms);
    }
  }
#endif

  if (gpu_full_pipeline_strict) {
    ZIRAN_ASSERT(gpu_interface_quad_used &&
                     gpu_interface_quad_device_ready &&
                     !gpu_interface_quad_host_fields_valid,
        "strict full-GPU interface quadrature fell back to CPU/host fields");
  }

  if (!gpu_interface_quad_used && !gpu_skip_interface_neighbor_search) {
    if (solid_scheme != CONFIG_UPDATE_LAGRANGIAN) {
      spatial_hash.rebuild(radius, solid_cur_positions);
    } else {
      spatial_hash.rebuild(radius, solid_positions);
    }
  }
  // if(solid_scheme != CONFIG_UPDATE_LAGRANGIAN)
  //   spatial_hash.rebuild(radius, solid_cur_positions);
  // else
  //   spatial_hash.rebuild(radius, solid_positions);
  if (!gpu_interface_quad_used && !gpu_skip_interface_neighbor_search)
    Base::parallel_for_updating_grid([&](int i) {
    TV Xp = particles.X.array[i];
    StdVector<int> unfiltered_neighbors;
    StdVector<int> neighbors;
    spatial_hash.oneLayerNeighbors(Xp, unfiltered_neighbors);
    for (auto &idx : unfiltered_neighbors)
      if (solid_scheme != CONFIG_UPDATE_LAGRANGIAN) {
        if (!particleInSolid(Xp) && !particleInSolid(solid_cur_positions[idx]))
          if ((Xp - solid_cur_positions[idx]).norm() < radius)
            neighbors.push_back(idx);
      } else {
        if (!particleInSolid(Xp) && !particleInSolid(solid_positions[idx]))
          if ((Xp - solid_positions[idx]).norm() < radius)
            neighbors.push_back(idx);
      }
    // if(solid_scheme != CONFIG_UPDATE_LAGRANGIAN)
    //   if (!particleInSolid(Xp) && !particleInSolid(solid_cur_positions[idx]))
    //     if ((Xp - solid_cur_positions[idx]).norm() < radius)
    //       neighbors.push_back(idx);
    // else
    //   if (!particleInSolid(Xp) && !particleInSolid(solid_positions[idx]))
    //     if ((Xp - solid_positions[idx]).norm() < radius)
    //       neighbors.push_back(idx);
    T unit_mass = SVQ_mass[i] / (T)(neighbors.size() + 1);
    SVQ_mass[i] = unit_mass;
    for (auto idx : neighbors) {
      AVQ_mass[idx] += unit_mass;
      AVQ_v[idx] += unit_mass * SVQ_v[i];
      // AVQ_C[idx] += unit_mass * SVQ_C[i];
    }
    });

  // tbb::parallel_for(0, (int)solid_cur_positions.size(), [&](int I) {
  //   solid_normals[i] = TV::Zero()
  //   TV Xp = solid_cur_positions[i];
  //   StdVector<int> unfiltered_neighbors;
  //   StdVector<int> neighbors;
  //   spatial_hash.oneLayerNeighbors(Xp, unfiltered_neighbors);
  //   for (auto &idx : unfiltered_neighbors)
  //     if (!particleInSolid(Xp) && !particleInSolid(solid_cur_positions[idx]))
  //       if ((Xp - solid_cur_positions[idx]).norm() < radius)
  //         neighbors.push_back(idx);
  //   return;
  // });

  if (gpu_interface_quad_host_fields_valid)
    for (int i = 0; i < (int)solid_positions.size(); ++i)
      if (AVQ_mass[i] > 0) {
        AVQ_v[i] /= AVQ_mass[i];
        // AVQ_C[i] /= AVQ_mass[i];
      }

  {
  ZIRAN_NAMED_TIMER("interfaceQuadLegacyMaterialize");
  quadrature_state.volume_GQ.clear();
  quadrature_state.volume_mass.clear();
  quadrature_state.volume_v.clear();
  quadrature_state.volume_C.clear();
  quadrature_state.face_GQ.clear();
  quadrature_state.face_N.clear();
  interface_rest_GQ.clear();
  interface_GQ.clear();
  interface_N.clear();
  interface_L.clear();
  if (gpu_interface_quad_host_fields_valid) {
  ZIRAN_ASSERT(Base::interpolation_degree != 1, "do not support linear kernel");
  for (int i = 0; i < particles.count; ++i) {
    quadrature_state.volume_GQ.push_back(particles.X.array[i]);
    quadrature_state.volume_mass.push_back(SVQ_mass[i]);
    quadrature_state.volume_v.push_back(SVQ_v[i]);
    if (fluid_Q1Q0)
      quadrature_state.volume_C.push_back(((Base::apic_rpic_ratio + 1) * (T)0.5) *
                                  Base::scratch_gradV[i] +
                              ((Base::apic_rpic_ratio - 1) * (T)0.5) *
                                  Base::scratch_gradV[i].transpose());
    else {
      auto *Carray_pointer = particles.DataManager::getPointer(C_name<TM>());
      quadrature_state.volume_C.push_back((*Carray_pointer)[i]);
    }
  }
  for (int i = 0; i < (int)solid_positions.size(); ++i)
    if (AVQ_mass[i] > 0) {
      if (solid_scheme != CONFIG_UPDATE_LAGRANGIAN) {
        quadrature_state.volume_GQ.push_back(solid_cur_positions[i]);
      } else {
        quadrature_state.volume_GQ.push_back(solid_positions[i]);
      }
      // if(solid_scheme != CONFIG_UPDATE_LAGRANGIAN)
      //   quadrature_state.volume_GQ.push_back(solid_cur_positions[i]);
      // else
      //   quadrature_state.volume_GQ.push_back(solid_positions[i]);
      quadrature_state.volume_mass.push_back(AVQ_mass[i]);
      quadrature_state.volume_v.push_back(AVQ_v[i]);
      quadrature_state.volume_C.push_back(TM::Zero());

      interface_rest_GQ.push_back(solid_positions[i]);
      if (solid_scheme != CONFIG_UPDATE_LAGRANGIAN) {
        interface_GQ.push_back(solid_cur_positions[i]);
      } else {
        interface_GQ.push_back(solid_positions[i]);
      }
      // if(solid_scheme != CONFIG_UPDATE_LAGRANGIAN)
      //   interface_GQ.push_back(solid_cur_positions[i]);
      // else
      //   interface_GQ.push_back(solid_positions[i]);
      interface_N.push_back(solid_normals[i]);
      interface_L.push_back(solid_lens[i]);
    }
  }
  }
  if constexpr (dim == 3) {
    if (std::getenv("ZIRAN_GPU_INTERFACE_FIELDS_COMPARE") != nullptr &&
        gpu_interface_fields.size != 0) {
      std::vector<GPU_STAGE2::Vec3d> device_rest;
      std::vector<GPU_STAGE2::Vec3d> device_current;
      std::vector<GPU_STAGE2::Vec3d> device_normal;
      std::vector<double> device_length;
      GPU_STAGE2::copyInterfaceQuadratureFieldsToHost3D(
          gpu_interface_fields, device_rest, device_current,
          device_normal, device_length);
      ZIRAN_ASSERT(device_rest.size() == interface_GQ.size(),
          "interface field comparison size mismatch");
      double max_rest = 0.0;
      double max_current = 0.0;
      double max_normal = 0.0;
      double max_length = 0.0;
      for (std::size_t i = 0; i < device_rest.size(); ++i) {
        const double host_rest[3] = {
            static_cast<double>(interface_rest_GQ[i][0]),
            static_cast<double>(interface_rest_GQ[i][1]),
            static_cast<double>(interface_rest_GQ[i][2])};
        const double host_current[3] = {
            static_cast<double>(interface_GQ[i][0]),
            static_cast<double>(interface_GQ[i][1]),
            static_cast<double>(interface_GQ[i][2])};
        const double host_normal[3] = {
            static_cast<double>(interface_N[i][0]),
            static_cast<double>(interface_N[i][1]),
            static_cast<double>(interface_N[i][2])};
        const double device_rest_values[3] = {
            device_rest[i].x, device_rest[i].y, device_rest[i].z};
        const double device_current_values[3] = {
            device_current[i].x, device_current[i].y, device_current[i].z};
        const double device_normal_values[3] = {
            device_normal[i].x, device_normal[i].y, device_normal[i].z};
        for (int d = 0; d < 3; ++d) {
          max_rest = std::max(max_rest,
              std::abs(host_rest[d] - device_rest_values[d]));
          max_current = std::max(max_current,
              std::abs(host_current[d] - device_current_values[d]));
          max_normal = std::max(max_normal,
              std::abs(host_normal[d] - device_normal_values[d]));
        }
        max_length = std::max(max_length,
            std::abs(static_cast<double>(interface_L[i]) - device_length[i]));
      }
      ZIRAN_INFO("GPU interface field comparison: samples=",
          device_rest.size(), " max_rest=", max_rest,
          " max_current=", max_current, " max_normal=", max_normal,
          " max_length=", max_length);
    }
  }
}

// build M M_inv G D S a b
template <class T, int dim>
template <int degree, int order>
void SplittingSimulation<T, dim>::fluidKernel(MpmGrid<T, dim, degree> &grid3,
                                              MpmGrid<T, dim, order> &grid4,
                                              MpmGrid<T, dim, order> &grid5,
                                              SM &M_inv, SM &G, SM &D, SM &S,
                                              TStack &a, TStack &b) {
  ZIRAN_TIMER();
  auto *vol_pointer =
      particles.DataManager::getPointer(element_measure_name<T>());

  std::optional<GPU_STAGE3A::FluidQ1Q0AssemblyResult> gpu_assembled;
  gpu_device_assembly_valid = false;

  // Some coupling scenes (for example example 1) deliberately start with an
  // empty fluid phase and inject fluid particles in a later frame callback.
  // The device Q1Q0 input needs particle[0] to determine rho, whereas the
  // CPU empty-grid path already represents the valid zero-sized fluid block.
  // Keep that CPU representation until the phase actually owns particles.
  if (gpu_stage3b_assemble_blocks && particles.count != 0) {
    if constexpr (dim == 3 && degree == 1 && order == 0) {
      ZIRAN_TIMER();
      const bool resident_prebuilt_velocity = gpu_prebuilt_velocity_valid &&
          gpu_velocity_node_map.storage &&
          gpu_velocity_node_map.size != 0;
      num_v = resident_prebuilt_velocity
                  ? static_cast<int>(gpu_velocity_node_map.size)
                  : (gpu_prebuilt_velocity_valid
                        ? static_cast<int>(gpu_velocity_nodes.size())
                        : grid3.getNumNodes());
      const int pressure_node_count = num_p;
      const int face_node_count = quadrature_state.total_id;
      GPU_STAGE3A::FluidQ1Q0AssemblyInput input;
      input.dx = static_cast<double>(Base::dx);
      input.dt = static_cast<double>(Base::dt);
      input.rho = static_cast<double>(particles.mass.array[0] / (*vol_pointer)[0]);
      input.velocity_node_count = num_v;
      input.pressure_node_count = pressure_node_count;
      input.face_node_count = face_node_count;

      auto push_node = [](auto &dst, const IV &node,
                          const GridState<T, dim> &g) {
        dst.push_back(GPU_STAGE3A::AssemblyGridNode{
            GPU_STAGE3A::packGridKey3D(node[0], node[1], node[2]),
            static_cast<int>(g.idx), static_cast<double>(g.m),
            GPU_STAGE3A::Vec3d{static_cast<double>(g.new_v[0]),
                               static_cast<double>(g.new_v[1]),
                               static_cast<double>(g.new_v[2])},
            GPU_STAGE3A::Vec3d{}});
      };
      auto _gpu_t0 = std::chrono::high_resolution_clock::now();
      if (resident_prebuilt_velocity)
        input.resident_velocity_nodes = gpu_velocity_node_map;
      else if (gpu_prebuilt_velocity_valid)
        input.velocity_nodes = gpu_velocity_nodes;
      else
        grid3.iterateGridSerial([&](IV node, GridState<T, dim> &g) {
          push_node(input.velocity_nodes, node, g);
        });
      if (gpu_prebuilt_fluid_dofs_valid) {
        input.pressure_nodes = gpu_pressure_nodes;
        input.face_nodes = gpu_face_nodes;
        if (!gpu_fluid_canonical_dof_order &&
            gpu_pressure_node_map.size ==
                static_cast<std::size_t>(pressure_node_count) &&
            gpu_face_node_map.size ==
                static_cast<std::size_t>(face_node_count)) {
          input.resident_pressure_nodes = gpu_pressure_node_map;
          input.resident_face_nodes = gpu_face_node_map;
        }
      } else {
        grid4.iterateGridSerial([&](IV node, GridState<T, dim> &g) {
          push_node(input.pressure_nodes, node, g);
        });
        grid5.iterateGridSerial([&](IV node, GridState<T, dim> &g) {
          push_node(input.face_nodes, node, g);
        });
      }

      const bool resident_quadrature =
          input.resident_velocity_nodes.storage &&
          input.resident_pressure_nodes.storage &&
          input.resident_face_nodes.storage &&
          gpu_fluid_analytic_device_view.volume_count != 0 &&
          gpu_fluid_analytic_device_view.volume_xq &&
          gpu_fluid_analytic_device_view.volume_mass &&
          (gpu_fluid_analytic_device_view.face_count == 0 ||
              (gpu_fluid_analytic_device_view.face_xq &&
                  gpu_fluid_analytic_device_view.face_normal));
      if (resident_quadrature) {
        input.resident_volume_xq =
            gpu_fluid_analytic_device_view.volume_xq;
        input.resident_volume_mass =
            gpu_fluid_analytic_device_view.volume_mass;
        input.resident_volume_count =
            gpu_fluid_analytic_device_view.volume_count;
        input.resident_face_xq = gpu_fluid_analytic_device_view.face_xq;
        input.resident_face_normal =
            gpu_fluid_analytic_device_view.face_normal;
        input.resident_face_count =
            gpu_fluid_analytic_device_view.face_count;
      } else {
        input.volume_xq.reserve(quadrature_state.volume_GQ.size());
        input.volume_mass.reserve(quadrature_state.volume_mass.size());
        for (std::size_t i = 0; i < quadrature_state.volume_GQ.size(); ++i) {
          const TV &x = quadrature_state.volume_GQ[i];
          input.volume_xq.push_back(GPU_STAGE3A::Vec3d{
              static_cast<double>(x[0]), static_cast<double>(x[1]),
              static_cast<double>(x[2])});
          input.volume_mass.push_back(
              static_cast<double>(quadrature_state.volume_mass[i]));
        }
        input.face_xq.reserve(quadrature_state.face_GQ.size());
        input.face_normal.reserve(quadrature_state.face_N.size());
        for (std::size_t i = 0; i < quadrature_state.face_GQ.size(); ++i) {
          const TV &x = quadrature_state.face_GQ[i];
          const TV &n = quadrature_state.face_N[i];
          input.face_xq.push_back(GPU_STAGE3A::Vec3d{
              static_cast<double>(x[0]), static_cast<double>(x[1]),
              static_cast<double>(x[2])});
          input.face_normal.push_back(GPU_STAGE3A::Vec3d{
              static_cast<double>(n[0]), static_cast<double>(n[1]),
              static_cast<double>(n[2])});
        }
      }

      auto _gpu_t1 = std::chrono::high_resolution_clock::now();
      GPU_STAGE3A::FluidQ1Q0DeviceAssemblyResult device_assembled;
      GPU_STAGE3A::assembleFluidQ1Q0System3DDevice(input, device_assembled);
      if (gpu_fluid_canonical_dof_order &&
          std::getenv("ZIRAN_GPU_FLUID_CANONICAL_DOF_ORACLE")) {
        ZIRAN_ASSERT(gpu_canonical_dof_oracle_valid,
                     "canonical DOF oracle maps were not built");
        ZIRAN_ASSERT(gpu_canonical_velocity_to_legacy.size() ==
                         input.velocity_nodes.size() &&
                         gpu_canonical_pressure_to_legacy.size() ==
                         input.pressure_nodes.size() &&
                         gpu_canonical_face_to_legacy.size() ==
                         input.face_nodes.size(),
                     "canonical DOF oracle map dimensions mismatch");

        GPU_STAGE3A::FluidQ1Q0AssemblyInput legacy_input = input;
        for (auto& node : legacy_input.velocity_nodes)
          node.idx = gpu_canonical_velocity_to_legacy.at(
              static_cast<std::size_t>(node.idx));
        for (auto& node : legacy_input.pressure_nodes)
          node.idx = gpu_canonical_pressure_to_legacy.at(
              static_cast<std::size_t>(node.idx));
        for (auto& node : legacy_input.face_nodes)
          node.idx = gpu_canonical_face_to_legacy.at(
              static_cast<std::size_t>(node.idx));

        GPU_STAGE3A::FluidQ1Q0DeviceAssemblyResult legacy_assembled;
        GPU_STAGE3A::assembleFluidQ1Q0System3DDevice(legacy_input,
                                                      legacy_assembled);
        const auto canonical_m_inv =
            GPU_STAGE3A::copyDeviceCsrToHost(device_assembled.m_inv);
        const auto canonical_g =
            GPU_STAGE3A::copyDeviceCsrToHost(device_assembled.g);
        const auto legacy_m_inv =
            GPU_STAGE3A::copyDeviceCsrToHost(legacy_assembled.m_inv);
        const auto legacy_g =
            GPU_STAGE3A::copyDeviceCsrToHost(legacy_assembled.g);
        const auto canonical_a =
            GPU_STAGE3A::copyDeviceVectorToHost(device_assembled.a);
        const auto canonical_b =
            GPU_STAGE3A::copyDeviceVectorToHost(device_assembled.b);
        const auto legacy_a =
            GPU_STAGE3A::copyDeviceVectorToHost(legacy_assembled.a);
        const auto legacy_b =
            GPU_STAGE3A::copyDeviceVectorToHost(legacy_assembled.b);

        std::vector<int> velocity_dof_to_legacy(
            gpu_canonical_velocity_to_legacy.size() * dim);
        for (std::size_t node = 0;
             node < gpu_canonical_velocity_to_legacy.size(); ++node)
          for (int d = 0; d < dim; ++d)
            velocity_dof_to_legacy[node * dim + d] =
                gpu_canonical_velocity_to_legacy[node] * dim + d;
        std::vector<int> pressure_dof_to_legacy;
        pressure_dof_to_legacy.reserve(
            gpu_canonical_pressure_to_legacy.size() +
            gpu_canonical_face_to_legacy.size());
        pressure_dof_to_legacy.insert(pressure_dof_to_legacy.end(),
            gpu_canonical_pressure_to_legacy.begin(),
            gpu_canonical_pressure_to_legacy.end());
        const int pressure_count = static_cast<int>(
            gpu_canonical_pressure_to_legacy.size());
        for (const int face_legacy : gpu_canonical_face_to_legacy)
          pressure_dof_to_legacy.push_back(pressure_count + face_legacy);

        struct OracleEntry {
          int row;
          int col;
          double value;
        };
        auto compare_csr_permutation =
            [&](const char* label, const GPU_STAGE3A::CsrMatrix& canonical,
                const GPU_STAGE3A::CsrMatrix& legacy,
                const std::vector<int>& row_to_legacy,
                const std::vector<int>& col_to_legacy) {
              ZIRAN_ASSERT(canonical.rows == static_cast<int>(row_to_legacy.size()) &&
                               canonical.cols == static_cast<int>(col_to_legacy.size()) &&
                               legacy.rows == canonical.rows &&
                               legacy.cols == canonical.cols,
                           "canonical DOF oracle CSR dimensions mismatch");
              std::vector<OracleEntry> permuted;
              std::vector<OracleEntry> reference;
              permuted.reserve(canonical.values.size());
              reference.reserve(legacy.values.size());
              for (int row = 0; row < canonical.rows; ++row)
                for (int k = canonical.row_offsets[static_cast<std::size_t>(row)];
                     k < canonical.row_offsets[static_cast<std::size_t>(row + 1)]; ++k)
                  permuted.push_back({row_to_legacy[static_cast<std::size_t>(row)],
                      col_to_legacy[static_cast<std::size_t>(
                          canonical.col_indices[static_cast<std::size_t>(k)])],
                      canonical.values[static_cast<std::size_t>(k)]});
              for (int row = 0; row < legacy.rows; ++row)
                for (int k = legacy.row_offsets[static_cast<std::size_t>(row)];
                     k < legacy.row_offsets[static_cast<std::size_t>(row + 1)]; ++k)
                  reference.push_back({row, legacy.col_indices[static_cast<std::size_t>(k)],
                                       legacy.values[static_cast<std::size_t>(k)]});
              auto less_entry = [](const OracleEntry& a, const OracleEntry& b) {
                return a.row != b.row ? a.row < b.row : a.col < b.col;
              };
              std::sort(permuted.begin(), permuted.end(), less_entry);
              std::sort(reference.begin(), reference.end(), less_entry);
              ZIRAN_ASSERT(permuted.size() == reference.size(),
                           "canonical DOF oracle CSR nnz mismatch");
              double max_abs = 0;
              double max_ref = 0;
              for (std::size_t i = 0; i < reference.size(); ++i) {
                ZIRAN_ASSERT(permuted[i].row == reference[i].row &&
                                 permuted[i].col == reference[i].col,
                             "canonical DOF oracle CSR pattern mismatch");
                max_abs = std::max(max_abs,
                    std::abs(permuted[i].value - reference[i].value));
                max_ref = std::max(max_ref, std::abs(reference[i].value));
              }
              ZIRAN_INFO("GPU canonical DOF oracle ", label,
                         ": nnz=", reference.size(), " max_abs=", max_abs,
                         " max_rel=", max_abs / std::max(max_ref, 1e-30));
              ZIRAN_ASSERT(max_abs <= 1e-9 * std::max(max_ref, 1.0),
                           "canonical DOF oracle CSR value mismatch");
            };
        auto compare_vector_permutation =
            [&](const char* label, const std::vector<double>& canonical,
                const std::vector<double>& legacy,
                const std::vector<int>& to_legacy) {
              ZIRAN_ASSERT(canonical.size() == to_legacy.size() &&
                               legacy.size() == canonical.size(),
                           "canonical DOF oracle vector dimensions mismatch");
              double max_abs = 0;
              double max_ref = 0;
              for (std::size_t i = 0; i < canonical.size(); ++i) {
                const double reference = legacy.at(static_cast<std::size_t>(
                    to_legacy[i]));
                max_abs = std::max(max_abs, std::abs(canonical[i] - reference));
                max_ref = std::max(max_ref, std::abs(reference));
              }
              ZIRAN_INFO("GPU canonical DOF oracle ", label,
                         ": size=", canonical.size(), " max_abs=", max_abs,
                         " max_rel=", max_abs / std::max(max_ref, 1e-30));
              ZIRAN_ASSERT(max_abs <= 1e-9 * std::max(max_ref, 1.0),
                           "canonical DOF oracle vector value mismatch");
            };
        compare_csr_permutation("M_inv", canonical_m_inv, legacy_m_inv,
                                velocity_dof_to_legacy,
                                velocity_dof_to_legacy);
        compare_csr_permutation("G", canonical_g, legacy_g,
                                velocity_dof_to_legacy,
                                pressure_dof_to_legacy);
        compare_vector_permutation("a", canonical_a, legacy_a,
                                   velocity_dof_to_legacy);
        compare_vector_permutation("b", canonical_b, legacy_b,
                                   pressure_dof_to_legacy);
      }
      auto _gpu_t2 = std::chrono::high_resolution_clock::now();
      gpu_m_inv = device_assembled.m_inv;
      gpu_g = device_assembled.g;
      gpu_s = GPU_STAGE3A::assembleDeviceCsrBlocks(
          pressure_node_count + face_node_count,
          pressure_node_count + face_node_count, {});
      gpu_a = device_assembled.a;
      gpu_b = device_assembled.b;
      gpu_velocity_node_map = device_assembled.velocity_nodes;
      gpu_pressure_node_map = device_assembled.pressure_nodes;
      gpu_device_assembly_valid = true;
      if (gpu_stage3b_validate_blocks) {
        GPU_STAGE3A::FluidQ1Q0AssemblyResult assembled;
        assembled.m_inv =
            GPU_STAGE3A::copyDeviceCsrToHost(device_assembled.m_inv);
        assembled.g = GPU_STAGE3A::copyDeviceCsrToHost(device_assembled.g);
        assembled.a = GPU_STAGE3A::copyDeviceVectorToHost(device_assembled.a);
        assembled.b = GPU_STAGE3A::copyDeviceVectorToHost(device_assembled.b);
        gpu_assembled.emplace(std::move(assembled));
      }
      auto _ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      ZIRAN_INFO(gpu_stage3b_validate_blocks
                     ? "GPU fluid assembly oracle BREAKDOWN(ms): extract="
                     : "GPU fluid device assembly BREAKDOWN(ms): extract=",
                 _ms(_gpu_t0, _gpu_t1), " gpu_assemble=", _ms(_gpu_t1, _gpu_t2),
                 " | num_v=", num_v,
                 " G_nnz=", device_assembled.g.nonzeros);
    } else {
      ZIRAN_WARN("Stage3B GPU fluid assembly only supports 3D Q1Q0; using CPU assembly");
    }
  }

  if (gpu_device_assembly_valid && !gpu_stage3b_validate_blocks) {
    const int pressure_dofs = num_p + quadrature_state.total_id;
    M_inv = SM(num_v * dim, num_v * dim);
    G = SM(num_v * dim, pressure_dofs);
    D = SM(pressure_dofs, num_v * dim);
    S = SM(pressure_dofs, pressure_dofs);
    a = TStack::Zero(num_v * dim);
    b = TStack::Zero(pressure_dofs);
    return;
  }

  auto use_gpu_blocks = [&]() {
    ZIRAN_ASSERT(gpu_assembled.has_value(), "missing GPU fluid assembly");
    M_inv = GPU_STAGE3A::makeEigenFromCsrLocal<T>(gpu_assembled->m_inv);
    G = GPU_STAGE3A::makeEigenFromCsrLocal<T>(gpu_assembled->g);
    D = Eigen::SparseMatrix<T>(-G.transpose());
    S = Eigen::SparseMatrix<T>(num_p + quadrature_state.total_id,
                               num_p + quadrature_state.total_id);
    S.setZero();
    a = TStack::Zero(static_cast<int>(gpu_assembled->a.size()));
    b = TStack::Zero(static_cast<int>(gpu_assembled->b.size()));
    for (int i = 0; i < a.rows(); ++i)
      a(i) = static_cast<T>(gpu_assembled->a[static_cast<std::size_t>(i)]);
    for (int i = 0; i < b.rows(); ++i)
      b(i) = static_cast<T>(gpu_assembled->b[static_cast<std::size_t>(i)]);
  };

  // build M
  num_v = grid3.getNumNodes();
  M_inv = Eigen::SparseMatrix<T>(num_v * dim, num_v * dim);
  std::vector<Eigen::Triplet<T>> M_inv_tri;
  grid3.iterateGridSerial([&](IV node, GridState<T, dim> &g) {
    for (int d = 0; d < dim; ++d)
      M_inv_tri.emplace_back(g.idx * dim + d, g.idx * dim + d, Base::dt / g.m);
  });
  M_inv.setFromTriplets(M_inv_tri.begin(), M_inv_tri.end());
  // build G
  G = Eigen::SparseMatrix<T>(num_v * dim, num_p + quadrature_state.total_id);
  std::vector<Eigen::Triplet<T>> G_tri(
      quadrature_state.volume_GQ.size() * grid3.kernel_size * grid4.kernel_size * dim);
  {
    ZIRAN_TIMER();
    for (int i = 0; i < (int)quadrature_state.volume_GQ.size(); ++i) {
      TV Xp = quadrature_state.volume_GQ[i];
      T rho = particles.mass.array[0] / (*vol_pointer)[0];
      T vol = quadrature_state.volume_mass[i] / rho;
      BSplineWeights<T, dim, degree> spline3(
          degree == 2 ? Xp : (Xp - TV::Ones() * Base::dx * 0.5), Base::dx);
      BSplineWeights<T, dim, order> spline4(Xp - TV::Ones() * Base::dx * 0.5,
                                            Base::dx);
      uint64_t offset3 =
          Base::SparseMask::Linear_Offset(to_std_array(spline3.base_node));
      uint64_t offset4 =
          Base::SparseMask::Linear_Offset(to_std_array(spline4.base_node));
      int cnt = i * grid3.kernel_size * grid4.kernel_size * dim;
      grid3.iterateKernel(
          spline3, offset3, [&](IV node3, T w3, TV dw3, GridState<T, dim> &g3) {
            int idx3 = g3.idx;
            if (idx3 < 0)
              return;
            grid4.iterateKernel(
                spline4, offset4,
                [&](IV node4, T w4, TV dw4, GridState<T, dim> &g4) {
                  int idx4 = g4.idx;
                  if (idx4 < 0)
                    return;
                  for (int alpha = 0; alpha < dim; ++alpha)
                    G_tri[cnt++] = Eigen::Triplet<T>(idx3 * dim + alpha, idx4,
                                                     -vol * w4 * dw3[alpha]);
                });
          });
    }
  }
  {
    ZIRAN_TIMER();
    T len = (dim == 2 ? Base::dx : (Base::dx * Base::dx));
    for (int i = 0; i < (int)quadrature_state.face_GQ.size(); ++i) {
      TV Xp = quadrature_state.face_GQ[i];
      TV normal = quadrature_state.face_N[i];
      BSplineWeights<T, dim, degree> spline3(
          degree == 2 ? Xp : (Xp - TV::Ones() * Base::dx * 0.5), Base::dx);
      uint64_t offset3 =
          Base::SparseMask::Linear_Offset(to_std_array(spline3.base_node));
      grid3.iterateKernel(
          spline3, offset3, [&](IV node3, T w3, TV dw3, GridState<T, dim> &g3) {
            int idx3 = g3.idx;
            if (idx3 < 0)
              return;
            for (int delta = -1; delta <= 1; delta += 2) {
              TV tmp = quadrature_state.face_GQ[i] +
                       quadrature_state.face_N[i] * (T)0.5 * Base::dx * (T)delta;
              BSplineWeights<T, dim, order> spline5(
                  tmp - TV::Ones() * Base::dx * 0.5, Base::dx);
              int idx5 = grid5[spline5.base_node].idx;
              ZIRAN_ASSERT(idx5 >= 0, "dof not matching");
              T w5 = 1;
              for (int alpha = 0; alpha < dim; ++alpha)
                G_tri.emplace_back(idx3 * dim + alpha, num_p + idx5,
                                   len * w3 * w5 * normal[alpha]);
            }
          });
    }
    G.setFromTriplets(G_tri.begin(), G_tri.end());
  }

  // build D
  D = Eigen::SparseMatrix<T>(-G.transpose());
  // build S
  S = Eigen::SparseMatrix<T>(num_p + quadrature_state.total_id, num_p + quadrature_state.total_id);
  S.setZero();
  // Fanfu hacking condition
  if (0) {
    T rho = particles.mass.array[0] / (*vol_pointer)[0];
    std::vector<Eigen::Triplet<T>> S_tri;
    for (int i = 0; i < num_p + quadrature_state.total_id; i++)
      S_tri.emplace_back(
          i, i, (std::pow(Base::dx, (T)dim) / Base::dt / rho) / (T)100000);
    S.setFromTriplets(S_tri.begin(), S_tri.end());
  }

  // build a
  a = TStack::Zero(num_v * dim);
  grid3.iterateGridSerial([&](IV node, GridState<T, dim> &g) {
    for (int d = 0; d < dim; ++d)
      a(g.idx * dim + d) += g.m / Base::dt * g.new_v[d];
  });
  // build b
  b = TStack::Zero(num_p + quadrature_state.total_id);

  if (gpu_assembled) {
    const Eigen::SparseMatrix<double> cpu_m_inv = M_inv.template cast<double>();
    const Eigen::SparseMatrix<double> cpu_g = G.template cast<double>();
    const auto minv_stats = GPU_STAGE3A::compareCsrMatrices(
        "fluid_Minv", gpu_assembled->m_inv,
        GPU_STAGE3A::makeCsrFromEigen(cpu_m_inv));
    const auto g_stats = GPU_STAGE3A::compareCsrMatrices(
        "fluid_G", gpu_assembled->g, GPU_STAGE3A::makeCsrFromEigen(cpu_g));
    ZIRAN_INFO("GPU fluid base oracle M_inv: max_abs=", minv_stats.max_abs_error,
               " max_rel=", minv_stats.max_rel_error,
               " G: max_abs=", g_stats.max_abs_error,
               " max_rel=", g_stats.max_rel_error);

    T max_a_error = 0;
    ZIRAN_ASSERT(a.rows() == static_cast<int>(gpu_assembled->a.size()),
                 "GPU fluid a size mismatch");
    for (int i = 0; i < a.rows(); ++i)
      max_a_error = std::max(
          max_a_error,
          std::abs(a(i) - static_cast<T>(gpu_assembled->a[static_cast<std::size_t>(i)])));
    ZIRAN_INFO("GPU fluid base oracle a: max_abs=", max_a_error);

    const bool oracle_passed =
        minv_stats.max_rel_error <= 1e-12 &&
        g_stats.max_rel_error <= 1e-12 && max_a_error <= (T)1e-12;
    if (oracle_passed) {
      use_gpu_blocks();
    } else {
      ZIRAN_WARN("GPU fluid base assembly failed oracle; keeping CPU blocks");
    }
  }
}

template <class T, int dim>
template <int degree, int order>
void SplittingSimulation<T, dim>::buildFluidSystem() {
  ZIRAN_TIMER();
  gpu_prebuilt_fluid_dofs_valid = false;
  gpu_prebuilt_velocity_valid = false;
  gpu_velocity_nodes.clear();
  gpu_velocity_g2p_fallback_nodes.clear();
  gpu_pressure_nodes.clear();
  gpu_face_nodes.clear();
  gpu_velocity_node_map = {};
  gpu_pressure_node_map = {};
  gpu_face_node_map = {};
  gpu_fluid_analytic_device_view = {};
  gpu_canonical_dof_oracle_valid = false;
  gpu_canonical_velocity_to_legacy.clear();
  gpu_canonical_pressure_to_legacy.clear();
  gpu_canonical_face_to_legacy.clear();
  rebuildInterfaceQuad();

  // CUDA analytical integration also has a well-defined open-domain mode:
  // no fluid collision object means particleInSolid()/gridInSolid() are both
  // false in the CPU implementation.  Do not confuse that supported case
  // with an unknown collision geometry, which must still take the CPU path.
  const bool gpu_open_domain_fluid = Base::collision_objects.empty();
  const bool gpu_analytic_geometry_supported =
      gpu_box_shell_collision.valid || gpu_open_domain_fluid;
  const bool use_gpu_velocity_grid =
      gpu_build_system && !gpu_build_system_cpu_numerical_grid &&
      !gpu_build_system_cpu_velocity_grid && dim == 3 &&
      gpu_analytic_geometry_supported;
  bool do_analytical_integration = true;
  const bool gpu_analytical_forced_cpu =
      std::getenv("ZIRAN_GPU_DISABLE_FLUID_ANALYTICAL") != nullptr;
  const bool use_gpu_analytical =
      gpu_build_system && do_analytical_integration && fluid_Q1Q0 &&
      gpu_analytic_geometry_supported && !gpu_analytical_forced_cpu;
  const bool fluid_velocity_grid_oracle =
      std::getenv("ZIRAN_GPU_FLUID_VELOCITY_GRID_ORACLE") != nullptr;
  const bool compare_device_legacy_dofs =
      std::getenv("ZIRAN_GPU_FLUID_DEVICE_DOF_COMPARE") != nullptr;
  const bool use_device_legacy_dofs = use_gpu_analytical &&
      use_gpu_velocity_grid && gpu_stage3b_assemble_blocks &&
      !gpu_fluid_canonical_dof_order &&
      std::getenv("ZIRAN_GPU_FLUID_DEVICE_DOF_DISABLE") == nullptr;
  const bool materialize_fluid_analytic_host =
      !use_device_legacy_dofs || compare_device_legacy_dofs ||
      fluid_velocity_grid_oracle ||
      std::getenv("ZIRAN_GPU_INTERFACE_FIELDS_COMPARE") != nullptr;
  if (gpu_full_pipeline_strict) {
    ZIRAN_ASSERT(use_gpu_velocity_grid && use_gpu_analytical &&
                     use_device_legacy_dofs &&
                     !materialize_fluid_analytic_host,
        "strict full-GPU fluid analytical/DOF path fell back to host");
  }
  const bool use_device_interface_input =
      gpu_interface_quad_device_ready && gpu_interface_solid_search;
  const bool has_resident_fluid_v_grad =
      gpu_velocity_transfer_device_v_grad_valid &&
      gpu_velocity_transfer_device.v != nullptr &&
      gpu_velocity_transfer_device.grad_v != nullptr &&
      gpu_velocity_transfer_device.size ==
          static_cast<std::size_t>(particles.count);
  const bool use_resident_interface_grad =
      use_device_interface_input && has_resident_fluid_v_grad;
  // A supported scene may have no active interface samples (example 4 is the
  // important case). Its analytic input is then exactly the fluid particle
  // range. Keep V/gradV resident here too; otherwise the compatibility
  // velocity grid silently rebuilds APIC affine from host scratch_gradV and
  // prevents deferring the 9-double/particle gradV download.
  const bool use_resident_noninterface_v_grad =
      use_gpu_analytical && !use_device_interface_input &&
      has_resident_fluid_v_grad &&
      quadrature_state.volume_GQ.size() ==
          static_cast<std::size_t>(particles.count);
  const bool use_resident_analytic_velocity =
      use_device_interface_input || use_resident_noninterface_v_grad;
  const bool use_resident_analytic_affine =
      use_device_interface_input || use_resident_noninterface_v_grad;
  const bool pack_initial_position_mass =
      !use_gpu_analytical || fluid_velocity_grid_oracle;
  const bool pack_initial_velocity =
      !use_gpu_analytical || !use_resident_analytic_velocity ||
      fluid_velocity_grid_oracle;
  const bool pack_initial_affine =
      !use_gpu_analytical || !use_resident_analytic_affine ||
      fluid_velocity_grid_oracle;
  std::vector<GPU_STAGE2::Vec3d> gpu_initial_xq;
  std::vector<double> gpu_initial_mass;
  std::vector<GPU_STAGE2::Vec3d> gpu_initial_velocity;
  std::vector<GPU_STAGE2::Mat3d> gpu_initial_affine;
  std::vector<GPU_STAGE2::Mat3d> gpu_interface_host_affine;
  if constexpr (dim == 3) {
  if (use_device_interface_input && !use_resident_interface_grad) {
    ZIRAN_NAMED_TIMER("gpuInterfaceFluidAffinePack");
    gpu_interface_host_affine.resize(
        static_cast<std::size_t>(particles.count));
    tbb::parallel_for(std::size_t{0}, gpu_interface_host_affine.size(),
        [&](std::size_t i) {
          const TM affine = ((Base::apic_rpic_ratio + 1) * (T)0.5) *
                  Base::scratch_gradV[i] +
              ((Base::apic_rpic_ratio - 1) * (T)0.5) *
                  Base::scratch_gradV[i].transpose();
          for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
              gpu_interface_host_affine[i].values[row * 3 + col] =
                  static_cast<double>(affine(row, col));
        });
  }
  }
  if (use_gpu_velocity_grid) {
    ZIRAN_NAMED_TIMER("gpuFluidInitialStatePack");
    if (pack_initial_position_mass) {
      gpu_initial_xq.reserve(quadrature_state.volume_GQ.size());
      gpu_initial_mass.reserve(quadrature_state.volume_mass.size());
    }
    if (pack_initial_velocity)
      gpu_initial_velocity.reserve(quadrature_state.volume_v.size());
    if (pack_initial_affine)
      gpu_initial_affine.reserve(quadrature_state.volume_C.size());
    for (std::size_t i = 0; i < quadrature_state.volume_GQ.size(); ++i) {
      const TV &x = quadrature_state.volume_GQ[i];
      const TV &v = quadrature_state.volume_v[i];
      if (pack_initial_position_mass) {
        gpu_initial_xq.push_back(GPU_STAGE2::Vec3d{
            static_cast<double>(x[0]), static_cast<double>(x[1]),
            static_cast<double>(x[2])});
        gpu_initial_mass.push_back(
            static_cast<double>(quadrature_state.volume_mass[i]));
      }
      if (pack_initial_velocity)
        gpu_initial_velocity.push_back(GPU_STAGE2::Vec3d{
            static_cast<double>(v[0]), static_cast<double>(v[1]),
            static_cast<double>(v[2])});
      if (pack_initial_affine) {
        GPU_STAGE2::Mat3d affine;
        for (int row = 0; row < 3; ++row)
          for (int col = 0; col < 3; ++col)
            affine.values[row * 3 + col] =
                static_cast<double>(quadrature_state.volume_C[i](row, col));
        gpu_initial_affine.push_back(affine);
      }
    }
  }

  MpmGrid<T, dim, degree> &grid3 = grid_comm;
  if (!use_gpu_velocity_grid && degree == 1)
    grid3.activatePages(quadrature_state.volume_GQ, Base::dx, 0, 1, -TV::Ones() * Base::dx * 0.5);
  else if (!use_gpu_velocity_grid)
    grid3.activatePages(quadrature_state.volume_GQ, Base::dx);
  if (!use_gpu_velocity_grid)
  for (int i = 0; i < (int)quadrature_state.volume_GQ.size(); ++i) {
    TV Xp = quadrature_state.volume_GQ[i];
    if (degree == 1)
      Xp -= TV::Ones() * Base::dx * 0.5;
    T mass = quadrature_state.volume_mass[i];
    TV momentum = mass * quadrature_state.volume_v[i];
    TM C = mass * quadrature_state.volume_C[i];
    TM4 velocity_density = TM4::Zero();
    velocity_density.template topLeftCorner<dim, dim>() = C;
    velocity_density.template topRightCorner<dim, 1>() = momentum;
    velocity_density(3, 3) = mass;
    BSplineWeights<T, dim, degree> spline(Xp, Base::dx);
    uint64_t offset = MpmSimulationBase<T, dim>::SparseMask::Linear_Offset(
        to_std_array(spline.base_node));
    grid3.iterateKernel(
        spline, offset,
        [&](const IV &node, T w, const TV &dw, GridState<T, dim> &g) {
          TV4 xi_minus_xp = TV4::Zero();
          xi_minus_xp.template topLeftCorner<dim, 1>() =
              node.template cast<T>() * Base::dx -
              Xp; // top dim entries non-zero
          xi_minus_xp(3) = 1;
          TV4 velocity_delta = velocity_density * xi_minus_xp * w;
          g.m += velocity_delta(3);
          g.new_v += velocity_delta.template topLeftCorner<dim, 1>();
        });
  }
  if (!use_gpu_velocity_grid) {
    num_v = grid3.getNumNodes();
    grid3.iterateGrid([&](IV node, GridState<T, dim> &g) {
      g.new_v = g.new_v / g.m + Base::dt * Base::gravity;
      g.v = g.new_v;
    });
  }

  auto rebuild_grid3_mass_from_quadrature = [&]() {
    grid3.iterateGrid([&](IV node, GridState<T, dim> &g) {
      g.m = 0;
      g.idx = -1;
    });
    for (int i = 0; i < (int)quadrature_state.volume_GQ.size(); ++i) {
      TV Xp = quadrature_state.volume_GQ[i];
      T mass = quadrature_state.volume_mass[i];
      BSplineWeights<T, dim, degree> spline(
          Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
      uint64_t offset =
          Base::SparseMask::Linear_Offset(to_std_array(spline.base_node));
      grid3.iterateKernel(spline, offset,
                          [&](const IV &node, T w, const TV &dw,
                              GridState<T, dim> &g) { g.m += mass * w; });
    }
    num_v = grid3.getNumNodes();
  };

  // 计算加速,为了分出face和volume
  bool gpu_analytical_done = false;
  if (gpu_build_system && !use_gpu_analytical) {
    static bool reported_gpu_analytical_fallback = false;
    if (!reported_gpu_analytical_fallback) {
      reported_gpu_analytical_fallback = true;
      ZIRAN_WARN("GPU fluid analytical integration disabled: "
                 "do_analytical=", do_analytical_integration,
                 " fluid_Q1Q0=", fluid_Q1Q0,
                 " box_shell_valid=", gpu_box_shell_collision.valid,
                 " open_domain=", gpu_open_domain_fluid,
                 " forced_cpu=", gpu_analytical_forced_cpu);
    }
  }
  if (use_gpu_analytical) {
    if constexpr (dim == 3) {
      ZIRAN_TIMER();
      auto *vol_pointer =
          particles.DataManager::getPointer(element_measure_name<T>());
      const double rho = static_cast<double>(
          particles.mass.array[0] / (*vol_pointer)[0]);
      std::vector<GPU_STAGE2::Vec3d> volume_xq;
      std::vector<double> volume_mass;
      std::vector<GPU_STAGE2::Vec3d> interface_xq;
      if (!use_device_interface_input) {
        volume_xq.reserve(quadrature_state.volume_GQ.size());
        volume_mass.reserve(quadrature_state.volume_mass.size());
        interface_xq.reserve(interface_GQ.size());
        for (std::size_t i = 0; i < quadrature_state.volume_GQ.size(); ++i) {
          const TV &x = quadrature_state.volume_GQ[i];
          volume_xq.push_back(GPU_STAGE2::Vec3d{
              static_cast<double>(x[0]), static_cast<double>(x[1]),
              static_cast<double>(x[2])});
          volume_mass.push_back(
              static_cast<double>(quadrature_state.volume_mass[i]));
        }
        for (const TV &x : interface_GQ)
          interface_xq.push_back(GPU_STAGE2::Vec3d{
              static_cast<double>(x[0]), static_cast<double>(x[1]),
              static_cast<double>(x[2])});
      }
      GPU_STAGE2::FluidAnalyticResult gpu_result;
      GPU_STAGE2::FluidAnalyticStats gpu_stats;
      if (use_device_interface_input) {
        ZIRAN_NAMED_TIMER("gpuInterfaceQuadToAnalytic");
        GPU_STAGE2::rebuildFluidAnalyticQuadratureFromInterfaceQuad3D(
            gpu_fluid_analytic, gpu_interface_quad_context,
            *gpu_interface_solid_search, static_cast<double>(Base::dx), rho,
            gpu_box_shell_collision, gpu_result, &gpu_stats,
            use_resident_interface_grad
                ? gpu_velocity_transfer_device.grad_v : nullptr,
            use_resident_interface_grad
                ? gpu_velocity_transfer_device.size : 0,
            static_cast<double>(Base::apic_rpic_ratio),
            gpu_interface_host_affine, materialize_fluid_analytic_host);
        if (gpu_interface_fields.size != 0 ||
            gpu_result.interface_xq.empty())
          GPU_STAGE2::bindInterfaceQuadratureClampedCurrent3D(
              gpu_fluid_analytic, gpu_interface_fields);
      } else {
        GPU_STAGE2::rebuildFluidAnalyticQuadrature(
            gpu_fluid_analytic, volume_xq, volume_mass, interface_xq,
            static_cast<double>(Base::dx), rho, gpu_box_shell_collision,
            gpu_result, &gpu_stats,
            use_resident_noninterface_v_grad
                ? gpu_velocity_transfer_device.v : nullptr,
            use_resident_noninterface_v_grad
                ? gpu_velocity_transfer_device.grad_v : nullptr,
            use_resident_noninterface_v_grad
                ? gpu_velocity_transfer_device.size : 0,
            static_cast<double>(Base::apic_rpic_ratio),
            materialize_fluid_analytic_host);
      }
      gpu_fluid_analytic_device_view =
          GPU_STAGE2::getFluidAnalyticDeviceView3D(gpu_fluid_analytic);

      {
        ZIRAN_NAMED_TIMER("gpuFluidAnalyticHostMaterialize");
        quadrature_state.volume_GQ.resize(gpu_result.volume_xq.size());
        quadrature_state.volume_mass.resize(gpu_result.volume_mass.size());
        tbb::parallel_for(std::size_t{0}, gpu_result.volume_xq.size(),
            [&](std::size_t i) {
              quadrature_state.volume_GQ[i] =
                  TV(static_cast<T>(gpu_result.volume_xq[i].x),
                     static_cast<T>(gpu_result.volume_xq[i].y),
                     static_cast<T>(gpu_result.volume_xq[i].z));
              quadrature_state.volume_mass[i] =
                  static_cast<T>(gpu_result.volume_mass[i]);
            });
        quadrature_state.face_GQ.resize(gpu_result.face_xq.size());
        quadrature_state.face_N.resize(gpu_result.face_normal.size());
        tbb::parallel_for(std::size_t{0}, gpu_result.face_xq.size(),
            [&](std::size_t i) {
              quadrature_state.face_GQ[i] =
                  TV(static_cast<T>(gpu_result.face_xq[i].x),
                     static_cast<T>(gpu_result.face_xq[i].y),
                     static_cast<T>(gpu_result.face_xq[i].z));
              quadrature_state.face_N[i] =
                  TV(static_cast<T>(gpu_result.face_normal[i].x),
                     static_cast<T>(gpu_result.face_normal[i].y),
                     static_cast<T>(gpu_result.face_normal[i].z));
            });
        interface_GQ.resize(gpu_result.interface_xq.size());
        tbb::parallel_for(std::size_t{0}, gpu_result.interface_xq.size(),
            [&](std::size_t i) {
              interface_GQ[i] =
                  TV(static_cast<T>(gpu_result.interface_xq[i].x),
                     static_cast<T>(gpu_result.interface_xq[i].y),
                     static_cast<T>(gpu_result.interface_xq[i].z));
            });
      }
      if (!gpu_build_system_cpu_numerical_grid &&
          !gpu_build_system_cpu_pressure_grid) {
      if (!use_device_legacy_dofs || compare_device_legacy_dofs ||
          gpu_fluid_canonical_dof_order) {
      ZIRAN_NAMED_TIMER("gpuFluidDofNodeMaterialize");
      gpu_pressure_nodes.resize(gpu_result.pressure_keys.size());
      tbb::parallel_for(std::size_t{0}, gpu_result.pressure_keys.size(),
          [&](std::size_t i) {
            gpu_pressure_nodes[i].key = gpu_result.pressure_keys[i];
            gpu_pressure_nodes[i].idx = static_cast<int>(i);
            gpu_pressure_nodes[i].mass = 1.0;
          });
      gpu_face_nodes.resize(gpu_result.face_keys.size());
      tbb::parallel_for(std::size_t{0}, gpu_result.face_keys.size(),
          [&](std::size_t i) {
            gpu_face_nodes[i].key = gpu_result.face_keys[i];
            gpu_face_nodes[i].idx = static_cast<int>(i);
            gpu_face_nodes[i].mass = 1.0;
          });
      }
      auto assign_spgrid_dofs = [&](auto &nodes,
                                    const StdVector<TV> &pollute_positions,
                                    int page_start, int page_end) {
        std::unordered_map<uint64_t, int> page_rank;
        // The rank itself is order-sensitive, but computing the source page
        // offsets is not.  Precompute that expensive SPGrid bit-spread in
        // parallel and keep the original serial first-occurrence scan below.
        std::vector<uint64_t> source_offsets(pollute_positions.size());
        tbb::parallel_for(std::size_t{0}, pollute_positions.size(),
            [&](std::size_t position_i) {
              const TV& position = pollute_positions[position_i];
              IV base;
              for (int d = 0; d < 3; ++d)
                base[d] = static_cast<int>(std::floor(
                    (position[d] - Base::dx * (T)0.5) / Base::dx));
              source_offsets[position_i] = Base::SparseMask::Linear_Offset(
                  std::array<int, 3>{base[0], base[1], base[2]});
            });
        page_rank.reserve(std::max<std::size_t>(64, pollute_positions.size() / 8));
        int next_page_rank = 0;
        uint64_t previous_offset = 0;
        bool have_previous = false;
        constexpr uint64_t page_mask = ~((uint64_t{1} << 12) - 1);
        const int block_x = 1 << Base::SparseMask::block_xbits;
        const int block_y = 1 << Base::SparseMask::block_ybits;
        const int block_z = 1 << Base::SparseMask::block_zbits;
        auto add_page = [&](uint64_t offset) {
          const uint64_t page = offset & page_mask;
          if (page_rank.find(page) == page_rank.end())
            page_rank.emplace(page, next_page_rank++);
        };
        for (const uint64_t offset : source_offsets) {
          if (!have_previous || offset != previous_offset) {
            add_page(offset);
            for (int i = page_start; i <= page_end; ++i)
              for (int j = page_start; j <= page_end; ++j)
                for (int k = page_start; k <= page_end; ++k)
                  add_page(Base::SparseMask::Packed_Add(
                      offset, Base::SparseMask::Linear_Offset(
                                  block_x * i, block_y * j, block_z * k)));
          }
          previous_offset = offset;
          have_previous = true;
        }
        std::vector<std::tuple<int, uint64_t, std::size_t>> dof_order;
        dof_order.reserve(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
          int x, y, z;
          GPU_STAGE2::unpackGridKey3D(nodes[i].key, x, y, z);
          const uint64_t offset = Base::SparseMask::Linear_Offset(
              std::array<int, 3>{x, y, z});
          const auto page_it = page_rank.find(offset & page_mask);
          ZIRAN_ASSERT(page_it != page_rank.end(),
                       "GPU pressure node page missing from pollute order");
          dof_order.emplace_back(page_it->second, offset & ~page_mask, i);
        }
        tbb::parallel_sort(dof_order.begin(), dof_order.end(),
                  [](const auto &a, const auto &b) {
                    if (std::get<0>(a) != std::get<0>(b))
                      return std::get<0>(a) < std::get<0>(b);
                    return std::get<1>(a) < std::get<1>(b);
                  });
        for (std::size_t rank = 0; rank < dof_order.size(); ++rank)
          nodes[std::get<2>(dof_order[rank])].idx = static_cast<int>(rank);
      };
      const bool build_canonical_oracle =
          gpu_fluid_canonical_dof_order &&
          std::getenv("ZIRAN_GPU_FLUID_CANONICAL_DOF_ORACLE") != nullptr;
      if (!gpu_fluid_canonical_dof_order &&
          (!use_device_legacy_dofs || compare_device_legacy_dofs)) {
        ZIRAN_NAMED_TIMER("gpuFluidPressureDofOrder");
        assign_spgrid_dofs(gpu_pressure_nodes, quadrature_state.volume_GQ, 0, 1);
      } else if (build_canonical_oracle) {
        auto legacy_nodes = gpu_pressure_nodes;
        assign_spgrid_dofs(legacy_nodes, quadrature_state.volume_GQ, 0, 1);
        gpu_canonical_pressure_to_legacy.resize(legacy_nodes.size());
        for (std::size_t i = 0; i < legacy_nodes.size(); ++i) {
          ZIRAN_ASSERT(gpu_pressure_nodes[i].idx == static_cast<int>(i),
                       "canonical pressure node is not tightly indexed");
          gpu_canonical_pressure_to_legacy[i] = legacy_nodes[i].idx;
        }
      }
      if (!gpu_fluid_canonical_dof_order &&
          (!use_device_legacy_dofs || compare_device_legacy_dofs)) {
        ZIRAN_NAMED_TIMER("gpuFluidFaceDofOrder");
        assign_spgrid_dofs(gpu_face_nodes, quadrature_state.face_GQ, -1, 1);
      } else if (build_canonical_oracle) {
        auto legacy_nodes = gpu_face_nodes;
        assign_spgrid_dofs(legacy_nodes, quadrature_state.face_GQ, -1, 1);
        gpu_canonical_face_to_legacy.resize(legacy_nodes.size());
        for (std::size_t i = 0; i < legacy_nodes.size(); ++i) {
          ZIRAN_ASSERT(gpu_face_nodes[i].idx == static_cast<int>(i),
                       "canonical face node is not tightly indexed");
          gpu_canonical_face_to_legacy[i] = legacy_nodes[i].idx;
        }
      }
      if (use_device_legacy_dofs) {
        ZIRAN_NAMED_TIMER("gpuFluidDeviceLegacyDofOrder");
        GPU_STAGE2::FluidLegacyDofOrderStats device_dof_stats;
        GPU_STAGE2::buildFluidLegacyDofNodeMaps3D(
            gpu_fluid_analytic, static_cast<double>(Base::dx),
            1 << Base::SparseMask::block_xbits,
            1 << Base::SparseMask::block_ybits,
            1 << Base::SparseMask::block_zbits, gpu_pressure_node_map,
            gpu_face_node_map, &device_dof_stats);
        if (compare_device_legacy_dofs) {
          const auto device_pressure =
              GPU_STAGE3A::copyDeviceGridNodeMapToHost(gpu_pressure_node_map);
          const auto device_face =
              GPU_STAGE3A::copyDeviceGridNodeMapToHost(gpu_face_node_map);
          auto compare_nodes = [](const auto& host, const auto& device) {
            std::size_t key_mismatch = 0;
            std::size_t index_mismatch = 0;
            if (host.size() != device.size())
              key_mismatch = std::max(host.size(), device.size());
            const std::size_t count = std::min(host.size(), device.size());
            for (std::size_t i = 0; i < count; ++i) {
              key_mismatch += host[i].key != device[i].key;
              index_mismatch += host[i].idx != device[i].idx;
            }
            return std::make_pair(key_mismatch, index_mismatch);
          };
          const auto pressure_error =
              compare_nodes(gpu_pressure_nodes, device_pressure);
          const auto face_error = compare_nodes(gpu_face_nodes, device_face);
          ZIRAN_INFO("GPU fluid legacy DOF comparison: pressure_keys=",
              pressure_error.first, " pressure_indices=",
              pressure_error.second, " face_keys=", face_error.first,
              " face_indices=", face_error.second);
        }
        ZIRAN_INFO("GPU fluid device legacy DOFs: pressure=",
            device_dof_stats.pressure_nodes, " face=",
            device_dof_stats.face_nodes, " pressure_ms=",
            device_dof_stats.pressure_order_ms, " face_ms=",
            device_dof_stats.face_order_ms);
      }
      gpu_prebuilt_fluid_dofs_valid = true;
      }
      if (!use_gpu_velocity_grid)
        rebuild_grid3_mass_from_quadrature();
      gpu_analytical_done = true;
      ZIRAN_INFO("GPU fluid analytical integration: input=",
                 gpu_stats.input_samples,
                 " pruned=", gpu_stats.pruned_samples,
                 " active_cells=", gpu_stats.active_cells,
                 " output=", gpu_stats.output_samples,
                 " faces=", gpu_stats.face_samples,
                 " reallocations=", gpu_stats.reallocations,
                 " total_ms=", gpu_stats.total_ms);
    }
  }
  if (!gpu_analytical_done && do_analytical_integration && fluid_Q1Q0) {
    ZIRAN_TIMER();
    ZIRAN_ASSERT(degree == 1, "only work for FQ1Q0");
    auto *vol_pointer =
        particles.DataManager::getPointer(element_measure_name<T>());
    MpmGrid<T, dim, 1> cell_kind;
    cell_kind.activatePages(quadrature_state.volume_GQ, Base::dx, -1, 1,
                      -TV::Ones() * Base::dx * 0.5);
    // prune VQ outside boundary
    StdVector<TV> prune_volume_GQ;
    StdVector<T> prune_volume_mass;
    for (int i = 0; i < (int)quadrature_state.volume_GQ.size(); ++i)
      if (!particleInSolid(quadrature_state.volume_GQ[i])) {
        prune_volume_GQ.push_back(quadrature_state.volume_GQ[i]);
        prune_volume_mass.push_back(quadrature_state.volume_mass[i]);
      }
    quadrature_state.volume_GQ = prune_volume_GQ;
    quadrature_state.volume_mass = prune_volume_mass;
    // idx :
    //      0 inner fluid
    //      1 near free surface or near solid
    // m :
    for (auto &Xp : quadrature_state.volume_GQ) {
      BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * Base::dx * 0.5,
                                       Base::dx);
      cell_kind[spline.base_node].idx = 0;
    }
    cell_kind.iterateGrid([&](IV node, GridState<T, dim> &g) {
      IV region = IV::Ones() * 3;
      iterateRegion(region, [&](const IV &offset) {
        IV neighbor = node + offset - IV::Ones();
        if (!gridInSolid(neighbor) && cell_kind[neighbor].idx == -1)
          // if (!gridInSolid(neighbor) && cell_kind[neighbor].idx == -1)
          cell_kind[node].idx = 1;
      });
    });
    for (auto &Xp : interface_GQ) {
      BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * Base::dx * 0.5,
                                       Base::dx);
      cell_kind[spline.base_node].idx = 1;
    }

    //   //将体积和面积的GQ彻底分开
    StdVector<TV> tmp_volume_GQ = quadrature_state.volume_GQ;
    StdVector<T> tmp_volume_mass = quadrature_state.volume_mass;
    quadrature_state.volume_GQ.clear();
    quadrature_state.volume_mass.clear();
    for (int i = 0; i < (int)tmp_volume_GQ.size(); ++i) {
      TV Xp = tmp_volume_GQ[i];
      BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * Base::dx * 0.5,
                                       Base::dx);
      if (cell_kind[spline.base_node].m == 0) {
        for (int d = 0; d < dim; ++d)
          for (int delta = -1; delta <= 1; delta += 2) {
            IV node = spline.base_node;
            node[d] += delta;
            if (gridInSolid(node)) {
              TV tmp = (node + IV::Ones()).template cast<T>() * Base::dx;
              tmp[d] -= (T)delta * (T)0.5 * Base::dx;
              quadrature_state.face_GQ.push_back(tmp);
              quadrature_state.face_N.push_back(TV::Unit(d) * (T)delta);
            }
          }
        cell_kind[spline.base_node].m = 1;
      }
      if (cell_kind[spline.base_node].idx == 1) {
        quadrature_state.volume_GQ.push_back(tmp_volume_GQ[i]);
        quadrature_state.volume_mass.push_back(tmp_volume_mass[i]);
      }
      if (cell_kind[spline.base_node].idx == 0) {
        T vol = (dim == 2 ? (Base::dx * Base::dx)
                          : (Base::dx * Base::dx * Base::dx));
        T rho = particles.mass.array[0] / (*vol_pointer)[0];
        quadrature_state.volume_GQ.push_back(spline.base_node.template cast<T>() *
                                     Base::dx +
                                 TV::Ones() * Base::dx);
        quadrature_state.volume_mass.push_back(vol * rho);
        cell_kind[spline.base_node].idx = 2;
      }
    }

    auto clamp_position = [&](StdVector<TV> &Xarray) {
      for (auto &Xp : Xarray) {
        BSplineWeights<T, dim, 1> spline(Xp - TV::Ones() * Base::dx * 0.5,
                                         Base::dx);
        for (int d = 0; d < dim; ++d) {
          T lower = ((T)spline.base_node(d) + (T)0.6) * Base::dx;
          T upper = ((T)spline.base_node(d) + (T)1.4) * Base::dx;
          MATH_TOOLS::clamp(Xp(d), lower, upper);
        }
      }
    };
    clamp_position(quadrature_state.volume_GQ);
    // clamp_position(quadrature_state.face_GQ);
    clamp_position(interface_GQ);

    if (!use_gpu_velocity_grid)
      rebuild_grid3_mass_from_quadrature();
  }

  if (use_gpu_velocity_grid) {
    ZIRAN_NAMED_TIMER("gpuFluidVelocityGridCompatibility");
    std::vector<GPU_STAGE2::Vec3d> final_xq;
    std::vector<double> final_mass;
    if (!gpu_analytical_done || fluid_velocity_grid_oracle) {
      final_xq.reserve(quadrature_state.volume_GQ.size());
      final_mass.reserve(quadrature_state.volume_mass.size());
      for (std::size_t i = 0; i < quadrature_state.volume_GQ.size(); ++i) {
        const TV &x = quadrature_state.volume_GQ[i];
        final_xq.push_back(GPU_STAGE2::Vec3d{
            static_cast<double>(x[0]), static_cast<double>(x[1]),
            static_cast<double>(x[2])});
        final_mass.push_back(
            static_cast<double>(quadrature_state.volume_mass[i]));
      }
    }
    std::vector<GPU_STAGE2::VelocityGridNode> velocity_nodes;
    std::vector<GPU_STAGE2::VelocityGridNode> g2p_fallback_nodes;
    GPU_STAGE2::FluidVelocityGridStats velocity_stats;
    const bool materialize_host_velocity_nodes =
        !use_device_legacy_dofs || compare_device_legacy_dofs ||
        fluid_velocity_grid_oracle ||
        std::getenv("ZIRAN_GPU_FLUID_VELOCITY_COMPARE") != nullptr ||
        std::getenv("ZIRAN_FLUID_VELOCITY_FIELD_DUMP_PREFIX") != nullptr;
    GPU_STAGE2::rebuildFluidVelocityGrid(
        gpu_fluid_analytic, gpu_initial_xq, gpu_initial_mass,
        gpu_initial_velocity, gpu_initial_affine, final_xq, final_mass,
        static_cast<double>(Base::dx), static_cast<double>(Base::dt),
        GPU_STAGE2::Vec3d{static_cast<double>(Base::gravity[0]),
                          static_cast<double>(Base::gravity[1]),
                          static_cast<double>(Base::gravity[2])},
        velocity_nodes, g2p_fallback_nodes, &velocity_stats,
        gpu_analytical_done,
        gpu_analytical_done && use_resident_analytic_velocity,
        gpu_analytical_done && use_resident_analytic_affine,
        materialize_host_velocity_nodes);
    // The CPU reference deliberately uses two different quadrature states:
    // it scatters momentum from the pre-analytic samples, normalizes v there,
    // then overwrites only grid m using the post-prune/post-reconstruction
    // samples. Keep a small host oracle behind an opt-in environment switch
    // so changes to the CUDA implementation can be compared to that exact
    // (somewhat non-obvious) compatibility contract at a chosen frame.
    if (fluid_velocity_grid_oracle) {
      struct VelocityOracleNode {
        double initial_mass = 0;
        GPU_STAGE2::Vec3d momentum{0, 0, 0};
        double final_mass = 0;
      };
      std::unordered_map<GPU_STAGE2::GridKey, VelocityOracleNode> reference;
      reference.reserve(gpu_initial_xq.size() * 10);
      auto scatter = [&](const std::vector<GPU_STAGE2::Vec3d>& xq,
                         const std::vector<double>& mass,
                         bool momentum_phase) {
        for (std::size_t q = 0; q < xq.size(); ++q) {
          const auto& x = xq[q];
          const double gx[3] = {(x.x - 0.5 * Base::dx) / Base::dx,
                                (x.y - 0.5 * Base::dx) / Base::dx,
                                (x.z - 0.5 * Base::dx) / Base::dx};
          const int base[3] = {static_cast<int>(std::floor(gx[0])),
                               static_cast<int>(std::floor(gx[1])),
                               static_cast<int>(std::floor(gx[2]))};
          const double w[3][2] = {{1 - (gx[0] - base[0]), gx[0] - base[0]},
                                  {1 - (gx[1] - base[1]), gx[1] - base[1]},
                                  {1 - (gx[2] - base[2]), gx[2] - base[2]}};
          for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
              for (int k = 0; k < 2; ++k) {
                const double weight = w[0][i] * w[1][j] * w[2][k];
                auto& node = reference[GPU_STAGE2::packGridKey3D(
                    base[0] + i, base[1] + j, base[2] + k)];
                if (!momentum_phase) {
                  node.final_mass += mass[q] * weight;
                  continue;
                }
                node.initial_mass += mass[q] * weight;
                const double d[3] = {(base[0] + i) * Base::dx -
                                         (x.x - 0.5 * Base::dx),
                                     (base[1] + j) * Base::dx -
                                         (x.y - 0.5 * Base::dx),
                                     (base[2] + k) * Base::dx -
                                         (x.z - 0.5 * Base::dx)};
                double mv[3] = {mass[q] * gpu_initial_velocity[q].x,
                                mass[q] * gpu_initial_velocity[q].y,
                                mass[q] * gpu_initial_velocity[q].z};
                for (int row = 0; row < 3; ++row)
                  for (int col = 0; col < 3; ++col)
                    mv[row] += mass[q] *
                        gpu_initial_affine[q].values[row * 3 + col] * d[col];
                node.momentum.x += mv[0] * weight;
                node.momentum.y += mv[1] * weight;
                node.momentum.z += mv[2] * weight;
              }
        }
      };
      scatter(gpu_initial_xq, gpu_initial_mass, true);
      scatter(final_xq, final_mass, false);
      double max_mass_error = 0;
      double max_velocity_error = 0;
      std::size_t missing = 0;
      for (const auto& gpu_node : velocity_nodes) {
        const auto it = reference.find(gpu_node.key);
        if (it == reference.end()) {
          ++missing;
          continue;
        }
        const auto& node = it->second;
        max_mass_error = std::max(max_mass_error,
            std::abs(node.final_mass - gpu_node.mass));
        const double inv_mass = node.initial_mass ? 1 / node.initial_mass : 0;
        const GPU_STAGE2::Vec3d expected{
            node.momentum.x * inv_mass + Base::dt * Base::gravity[0],
            node.momentum.y * inv_mass + Base::dt * Base::gravity[1],
            node.momentum.z * inv_mass + Base::dt * Base::gravity[2]};
        max_velocity_error = std::max(max_velocity_error,
            std::max({std::abs(expected.x - gpu_node.velocity.x),
                      std::abs(expected.y - gpu_node.velocity.y),
                      std::abs(expected.z - gpu_node.velocity.z)}));
      }
      ZIRAN_INFO("GPU fluid velocity-grid CPU oracle: reference_nodes=",
          reference.size(), " gpu_nodes=", velocity_nodes.size(),
          " missing=", missing, " max_mass_abs=", max_mass_error,
          " max_velocity_abs=", max_velocity_error);
    }
    if (materialize_host_velocity_nodes) {
    gpu_velocity_nodes.resize(velocity_nodes.size());
    for (std::size_t i = 0; i < velocity_nodes.size(); ++i) {
      gpu_velocity_nodes[i].key = velocity_nodes[i].key;
      gpu_velocity_nodes[i].idx = static_cast<int>(i);
      gpu_velocity_nodes[i].mass = velocity_nodes[i].mass;
      gpu_velocity_nodes[i].new_v = GPU_STAGE3A::Vec3d{
          velocity_nodes[i].velocity.x, velocity_nodes[i].velocity.y,
          velocity_nodes[i].velocity.z};
    }
    gpu_velocity_g2p_fallback_nodes.resize(g2p_fallback_nodes.size());
    for (std::size_t i = 0; i < g2p_fallback_nodes.size(); ++i) {
      gpu_velocity_g2p_fallback_nodes[i].key = g2p_fallback_nodes[i].key;
      gpu_velocity_g2p_fallback_nodes[i].idx = -1;
      gpu_velocity_g2p_fallback_nodes[i].mass = 0.0;
      gpu_velocity_g2p_fallback_nodes[i].new_v = GPU_STAGE3A::Vec3d{
          g2p_fallback_nodes[i].velocity.x, g2p_fallback_nodes[i].velocity.y,
          g2p_fallback_nodes[i].velocity.z};
    }
    if constexpr (dim == 3) {
      if (!gpu_fluid_canonical_dof_order) {
      std::vector<std::pair<uint64_t, std::size_t>> dof_order;
      dof_order.reserve(gpu_velocity_nodes.size());
      for (std::size_t i = 0; i < gpu_velocity_nodes.size(); ++i) {
        int x, y, z;
        GPU_STAGE2::unpackGridKey3D(gpu_velocity_nodes[i].key, x, y, z);
        dof_order.emplace_back(Base::SparseMask::Linear_Offset(
                               std::array<int, 3>{x, y, z}),
                           i);
      }
      tbb::parallel_sort(dof_order.begin(), dof_order.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      for (std::size_t rank = 0; rank < dof_order.size(); ++rank)
        gpu_velocity_nodes[dof_order[rank].second].idx = static_cast<int>(rank);
      } else if (std::getenv("ZIRAN_GPU_FLUID_CANONICAL_DOF_ORACLE")) {
      auto legacy_nodes = gpu_velocity_nodes;
      std::vector<std::pair<uint64_t, std::size_t>> dof_order;
      dof_order.reserve(legacy_nodes.size());
      for (std::size_t i = 0; i < legacy_nodes.size(); ++i) {
        int x, y, z;
        GPU_STAGE2::unpackGridKey3D(legacy_nodes[i].key, x, y, z);
        dof_order.emplace_back(Base::SparseMask::Linear_Offset(
                               std::array<int, 3>{x, y, z}),
                           i);
      }
      tbb::parallel_sort(dof_order.begin(), dof_order.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      for (std::size_t rank = 0; rank < dof_order.size(); ++rank)
        legacy_nodes[dof_order[rank].second].idx = static_cast<int>(rank);
      gpu_canonical_velocity_to_legacy.resize(legacy_nodes.size());
      for (std::size_t i = 0; i < legacy_nodes.size(); ++i) {
        ZIRAN_ASSERT(gpu_velocity_nodes[i].idx == static_cast<int>(i),
                     "canonical velocity node is not tightly indexed");
        gpu_canonical_velocity_to_legacy[i] = legacy_nodes[i].idx;
      }
      gpu_canonical_dof_oracle_valid =
          gpu_canonical_velocity_to_legacy.size() == gpu_velocity_nodes.size() &&
          gpu_canonical_pressure_to_legacy.size() == gpu_pressure_nodes.size() &&
          gpu_canonical_face_to_legacy.size() == gpu_face_nodes.size();
      ZIRAN_INFO("GPU canonical DOF oracle maps: velocity=",
                 gpu_canonical_velocity_to_legacy.size(), " pressure=",
                 gpu_canonical_pressure_to_legacy.size(), " face=",
                 gpu_canonical_face_to_legacy.size());
      }
    }
    }
    if (use_device_legacy_dofs) {
      ZIRAN_NAMED_TIMER("gpuFluidDeviceLegacyVelocityOrder");
      double device_velocity_order_ms = 0.0;
      GPU_STAGE2::buildFluidLegacyVelocityNodeMap3D(
          gpu_fluid_analytic, gpu_velocity_node_map,
          &device_velocity_order_ms);
      if (compare_device_legacy_dofs) {
        const auto device_velocity =
            GPU_STAGE3A::copyDeviceGridNodeMapToHost(gpu_velocity_node_map);
        std::size_t key_mismatch = 0;
        std::size_t index_mismatch = 0;
        std::size_t mass_mismatch = 0;
        if (gpu_velocity_nodes.size() != device_velocity.size())
          key_mismatch = std::max(
              gpu_velocity_nodes.size(), device_velocity.size());
        const std::size_t count = std::min(
            gpu_velocity_nodes.size(), device_velocity.size());
        for (std::size_t i = 0; i < count; ++i) {
          key_mismatch += gpu_velocity_nodes[i].key != device_velocity[i].key;
          index_mismatch +=
              gpu_velocity_nodes[i].idx != device_velocity[i].idx;
          mass_mismatch +=
              gpu_velocity_nodes[i].mass != device_velocity[i].mass;
        }
        ZIRAN_INFO("GPU fluid legacy velocity DOF comparison: keys=",
            key_mismatch, " indices=", index_mismatch,
            " mass=", mass_mismatch);
      }
      ZIRAN_INFO("GPU fluid device legacy velocity DOFs: nodes=",
          gpu_velocity_node_map.size, " order_ms=",
          device_velocity_order_ms);
    }
    gpu_prebuilt_velocity_valid = true;
    num_v = static_cast<int>(use_device_legacy_dofs
            ? gpu_velocity_node_map.size : gpu_velocity_nodes.size());
    ZIRAN_INFO("GPU fluid velocity grid: input=", velocity_stats.input_samples,
               " final=", velocity_stats.final_samples,
               " nodes=", velocity_stats.nodes,
               " g2p_fallback_nodes=", velocity_stats.g2p_fallback_nodes,
               " resident_position_mass=",
               velocity_stats.resident_position_mass ? 1 : 0,
               " resident_velocity=",
               velocity_stats.resident_velocity ? 1 : 0,
               " resident_affine=",
               velocity_stats.resident_affine ? 1 : 0,
               " reallocations=", velocity_stats.reallocations,
               " total_ms=", velocity_stats.total_ms);
  }

  MpmGrid<T, dim, order> grid4;
  MpmGrid<T, dim, order> grid5; // 边界
  if (gpu_prebuilt_fluid_dofs_valid) {
    num_p = static_cast<int>(use_device_legacy_dofs
            ? gpu_pressure_node_map.size : gpu_pressure_nodes.size());
    quadrature_state.total_id = static_cast<int>(use_device_legacy_dofs
            ? gpu_face_node_map.size : gpu_face_nodes.size());
  } else {
    grid4.activatePages(quadrature_state.volume_GQ, Base::dx, 0, 1,
                  -TV::Ones() * Base::dx * 0.5);
    for (auto &Xp : quadrature_state.volume_GQ) {
      BSplineWeights<T, dim, order> spline4(
          Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
      uint64_t offset4 =
          Base::SparseMask::Linear_Offset(to_std_array(spline4.base_node));
      grid4.iterateKernel(
          spline4, offset4,
          [&](const IV &node4, T w4, const TV &dw4,
              GridState<T, dim> &g4) { g4.m += w4; });
    }
    num_p = grid4.getNumNodes();

    grid5.activatePages(quadrature_state.face_GQ, Base::dx, -1, 1,
                  -TV::Ones() * Base::dx * 0.5);
    for (int i = 0; i < (int)quadrature_state.face_GQ.size(); ++i)
      for (int delta = -1; delta <= 1; delta += 2) {
        TV Xp = quadrature_state.face_GQ[i] +
                quadrature_state.face_N[i] * (T)0.5 * Base::dx * (T)delta;
        BSplineWeights<T, dim, order> spline5(
            Xp - TV::Ones() * Base::dx * 0.5, Base::dx);
        grid5[spline5.base_node].m += (T)1;
      }
    quadrature_state.total_id = grid5.getNumNodes();
  }

  fluidKernel(grid3, grid4, grid5, M_inv, G, D, S, a, b);
  if (gpu_prebuilt_velocity_valid) {
    const bool device_w = buildDeviceInterfaceW(1, true);
    if (gpu_full_pipeline_strict) {
      ZIRAN_ASSERT(device_w,
          "strict full-GPU fluid interface W fell back to host");
    }
    if (!device_w) {
      if (interface_GQ.empty())
        W = Eigen::SparseMatrix<T>(0, num_v * dim);
      else
        buildWMatrixFromGpuVelocity();
    }
  } else {
    ZIRAN_ASSERT(!gpu_full_pipeline_strict,
        "strict full-GPU fluid velocity grid fell back to host");
    buildWMatrix(grid3);
  }
  if (!gpu_prebuilt_velocity_valid)
    num_v = grid.getNumNodes();
  num_p = num_p + quadrature_state.total_id;
}

template void SplittingSimulation<float, 2>::buildFluidSystem<1, 0>();
template void SplittingSimulation<double, 2>::buildFluidSystem<1, 0>();
template void SplittingSimulation<float, 3>::buildFluidSystem<1, 0>();
template void SplittingSimulation<double, 3>::buildFluidSystem<1, 0>();

// template void SplittingSimulation<float, 2>::buildFluidSystem<2, 1>();
// template void SplittingSimulation<double, 2>::buildFluidSystem<2, 1>();
// template void SplittingSimulation<float, 3>::buildFluidSystem<2, 1>();
// template void SplittingSimulation<double, 3>::buildFluidSystem<2, 1>();

} // namespace ZIRAN
