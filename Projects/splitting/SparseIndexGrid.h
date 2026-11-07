#pragma once

#include <Ziran/CS/Util/Forward.h>
#include <array>
#include <cassert>
#include <tbb/tbb.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ZIRAN {

template <class T, int dim>
class SparseIndexGrid {
    using Key = std::array<int, dim>;

    struct KeyHash {
        size_t operator()(const Key& key) const
        {
            size_t seed = 0;
            for (int value : key) {
                size_t h = std::hash<int>{}(value);
                seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    std::unordered_map<Key, T, KeyHash> data;

    static Key keyFrom(int x, int y)
    {
        static_assert(dim == 2);
        return Key{x, y};
    }

    static Key keyFrom(int x, int y, int z)
    {
        static_assert(dim == 3);
        return Key{x, y, z};
    }

    static Key keyFrom(const Vector<int, dim>& v)
    {
        Key key{};
        for (int d = 0; d < dim; ++d)
            key[d] = v[d];
        return key;
    }

public:
    SparseIndexGrid() {}

    bool find(int x, int y) const
    {
        return data.find(keyFrom(x, y)) != data.end();
    }

    bool find(int x, int y, int z) const
    {
        return data.find(keyFrom(x, y, z)) != data.end();
    }

    bool find(const Vector<int, dim>& v) const
    {
        return data.find(keyFrom(v)) != data.end();
    }

    T& operator()(int x, int y)
    {
        return data[keyFrom(x, y)];
    }

    T& operator()(int x, int y, int z)
    {
        return data[keyFrom(x, y, z)];
    }

    const T& operator()(int x, int y) const
    {
        auto iter = data.find(keyFrom(x, y));
        assert(iter != data.end());
        return iter->second;
    }

    const T& operator()(int x, int y, int z) const
    {
        auto iter = data.find(keyFrom(x, y, z));
        assert(iter != data.end());
        return iter->second;
    }

    T& operator[](const Vector<int, dim>& v)
    {
        return data[keyFrom(v)];
    }

    template <typename OP>
    void iterateGrid(const OP& target) const
    {
        if constexpr (dim == 2) {
            for (const auto& [key, value] : data)
                target(key[0], key[1], value);
        }
        else {
            for (const auto& [key, value] : data)
                target(key[0], key[1], key[2], value);
        }
    }

    mutable std::vector<std::pair<Key, T>> caches;
    template <typename OP>
    void iterateGridParallel(const OP& target) const
    {
        if (caches.empty()) {
            for (const auto& [key, value] : data)
                caches.emplace_back(key, value);
        }

        tbb::parallel_for(0, (int)caches.size(), [&](int i) {
            const Key& key = caches[i].first;
            const T& value = caches[i].second;
            if constexpr (dim == 2) {
                target(key[0], key[1], value);
            }
            else {
                target(key[0], key[1], key[2], value);
            }
        });
    }

    void clear()
    {
        data.clear();
        caches.clear();
    }
};

} // namespace ZIRAN
