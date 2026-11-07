#ifndef MULTI_TARGET_PINNING_TARGET_H
#define MULTI_TARGET_PINNING_TARGET_H
#include <Ziran/CS/Util/Forward.h>
#include <Ziran/CS/Util/BinaryIO.h>

namespace ZIRAN {

template <class T, int dim>
class MultiTargetPinningTarget {
public:
    using TV = Vector<T, dim>;
    T k, d; // stiffness and damping coefficient
    // TV x, v; // target position and target velocity
    StdVector<TV> x, v;

    MultiTargetPinningTarget()
    {
        k = 0;
        d = 0;
    }

    MultiTargetPinningTarget(const T k_in, const T d_in, const StdVector<TV>& x_in, const StdVector<TV>& v_in)
    {
        k = k_in;
        d = d_in;
        x = x_in;
        v = v_in;
    }

    void write(std::ostream& out) const
    {
        writeEntry(out, k);
        writeEntry(out, d);
        writeSTDVector(out, x);
        writeSTDVector(out, v);
    }

    static MultiTargetPinningTarget<T, dim> read(std::istream& in)
    {
        MultiTargetPinningTarget<T, dim> target;
        target.k = readEntry<T>(in);
        target.d = readEntry<T>(in);
        readSTDVector<TV>(in, target.x);
        readSTDVector<TV>(in, target.v);
        return target;
    }

    static const char* name()
    {
        // Keep the serialized attribute key used by existing checkpoints.
        return "ParallelPinningTarget";
    }
};

template <class T, int dim>
struct MultiTargetPinningTargetScratch {
    using TV = Vector<T, dim>;
    T A;
    TV b;

    MultiTargetPinningTargetScratch()
        : A(0)
        , b(TV::Zero())
    {
    }

    static const char* name()
    {
        // Scratch attributes retain the same registered key.
        return "ParallelPinningTargetScratch";
    }
};

template <class T, int dim>
struct RW<MultiTargetPinningTargetScratch<T, dim>> {
    using Tag = NoWriteTag<MultiTargetPinningTargetScratch<T, dim>>;
};

template <class T, int dim>
struct RW<MultiTargetPinningTarget<T, dim>> {
    using Tag = CustomTypeTag<MultiTargetPinningTarget<T, dim>>;
};
} // namespace ZIRAN

#endif
