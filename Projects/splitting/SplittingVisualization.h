#pragma once

#include <Ziran/Math/Geometry/Visualizer.h>

namespace ZIRAN {

template <class T, int dim> class SplittingSimulation;

// Debug / visualization point-cloud output for one splitting phase. Pulled out
// of SplittingSimulation as a free function so the class stays focused on the
// solve; the commented-out lines are kept as debug scaffolding to re-enable.
template <class T, int dim>
void writeSplittingVisualization(SplittingSimulation<T, dim> &sim) {
  using TV = Vector<T, dim>;
  auto &particles = sim.particles;

  if (particles.DataManager::getPointer(X0_name<TV>()) != nullptr) {
    StdVector<TV> &X0 = particles.DataManager::get(X0_name<TV>()).array;
    visualize_points(sim, X0, particles.V.array, "xX0");
  }
  // visualize_points(*this, particles.X.array, particles.V.array, "vel");

  if (particles.DataManager::getPointer(J_name<T>()) != nullptr) {
    StdVector<T> &J_pointer = particles.DataManager::get(J_name<T>()).array;
    visualize_points(sim, particles.X.array, J_pointer, "J_pointer");
  }

  if (sim.category == MATERIAL_PHASE_FLUID) {
    if (sim.solid_scheme == CONFIG_UPDATE_LAGRANGIAN) {
      visualize_points(sim, sim.solid_positions, sim.solid_normals, "solid_normal");
    } else {
      visualize_points(sim, sim.solid_cur_positions, sim.solid_normals,
                       "solid_normal");
    }
  }

  // visualize_points(*this, particles.X.array, particles.V.array, "xv");
  static bool first_run = true;
  if (first_run) {
    visualize_points(sim, sim.boundary_positions, sim.boundary_normals, "boundary");
    first_run = false;
  }
  // return;
  // visualize_points(*this, quadrature_state.volume_GQ, "volume_GQ");
  visualize_points(sim, sim.quadrature_state.face_GQ, sim.quadrature_state.face_N, "face_GQ");
  // visualize_points(*this, interface_GQ, vis_interface_vel, "interface_GQ");
  // visualize_points(*this, interface_GQ, interface_N, "interface_normal");

  // visualize_points(*this, interface_rest_GQ, interface_N,
  // "interface_rest_GQ");

  // {
  //     auto* pin_pointer =
  //     particles.DataManager::getPointer(MpmPinningForce<T,
  //     dim>::target_name()); if (pin_pointer != nullptr) {
  //         StdVector<TV> positions;
  //         StdVector<TV> normals;
  //         for (int i = 0; i < particles.count; ++i)
  //             for (int c = 0; c < (*pin_pointer)[i].x.size(); ++c) {
  //                 positions.push_back(before_advect[i]);
  //                 normals.push_back((*pin_pointer)[i].x[c] -
  //                 before_advect[i]);
  //             }
  //         visualize_points(*this, before_advect, "before_advect");
  //         visualize_points(*this, positions, normals, "spring");
  //     }
  // }
  return;
  visualize_points(sim, sim.interface_GQ, sim.interface_N, "interface");

  if constexpr (dim == 2) {
    visualize_points(sim, sim.quadrature_state.face_GQ, "face_GQ");
    visualize_points(sim, sim.quadrature_state.volume_GQ, "volume_GQ");
    visualize_points(sim, sim.before_advect, "before_advect");
  }
}

} // namespace ZIRAN
