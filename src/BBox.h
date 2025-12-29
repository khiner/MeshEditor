#pragma once

#include "numeric/vec3.h"

#include <array>
#include <limits>

using uint = uint32_t;

struct BBox {
    vec3 Min, Max;

    BBox(vec3 min, vec3 max) : Min(std::move(min)), Max(std::move(max)) {}
    BBox() : Min(std::numeric_limits<float>::max()), Max(-std::numeric_limits<float>::max()) {}

    std::array<vec3, 8> Corners() const {
        return {{
            {Min.x, Min.y, Min.z},
            {Min.x, Min.y, Max.z},
            {Min.x, Max.y, Min.z},
            {Min.x, Max.y, Max.z},
            {Max.x, Min.y, Min.z},
            {Max.x, Min.y, Max.z},
            {Max.x, Max.y, Min.z},
            {Max.x, Max.y, Max.z},
        }};
    }

    static constexpr std::array<uint, 24> EdgeIndices{
        // clang-format off
        0, 1, 1, 3, 3, 2, 2, 0,
        4, 5, 5, 7, 7, 6, 6, 4,
        0, 4, 1, 5, 3, 7, 2, 6,
        // clang-format on
    };

    vec3 Center() const { return (Min + Max) * 0.5f; }

    vec3 Normal(const vec3 &p) const {
        static constexpr float eps = 1e-4f;

        if (std::abs(p.x - Min.x) < eps) return {-1, 0, 0};
        if (std::abs(p.x - Max.x) < eps) return {1, 0, 0};
        if (std::abs(p.y - Min.y) < eps) return {0, -1, 0};
        if (std::abs(p.y - Max.y) < eps) return {0, 1, 0};
        if (std::abs(p.z - Min.z) < eps) return {0, 0, -1};
        if (std::abs(p.z - Max.z) < eps) return {0, 0, 1};

        return {0, 0, 0}; // Failed to find a normal.
    }
};
