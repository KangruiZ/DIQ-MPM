#pragma once

#include <MPM/Simulation/MpmSimulationBase.h>

namespace ZIRAN {

template <class T, int dim> class CouplingQuadratureState {
  using TV = Vector<T, dim>;
  using TM = Matrix<T, dim, dim>;

public:
  int total_id = 0;

  StdVector<TV> volume_GQ;
  StdVector<T> volume_mass;
  StdVector<TV> volume_v;
  StdVector<TM> volume_C;
  StdVector<TV> face_GQ;
  StdVector<TV> face_N;
};

} // namespace ZIRAN
