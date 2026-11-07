#pragma once

#include <Ziran/CS/Util/Forward.h>

namespace ZIRAN {

template <typename OP>
void iterateRegion(const Vector<int, 2> &region, const OP &target) {
  for (int i = 0; i < region[0]; ++i)
    for (int j = 0; j < region[1]; ++j)
      target(Vector<int, 2>(i, j));
}

template <typename OP>
void iterateRegion(const Vector<int, 3> &region, const OP &target) {
  for (int i = 0; i < region[0]; ++i)
    for (int j = 0; j < region[1]; ++j)
      for (int k = 0; k < region[2]; ++k)
        target(Vector<int, 3>(i, j, k));
}

} // namespace ZIRAN
