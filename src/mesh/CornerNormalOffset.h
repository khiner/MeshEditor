#pragma once

#include "numeric/VectorMath.h"
#include "numeric/vec2.h"

#include "gpu/Vertex.h"

#include <algorithm>
#include <cmath>
#include <span>

// Orthonormal frame anchoring a corner's authored-normal offset.
// Axes: the derived normal, the corner's first non-degenerate outgoing triangle edge projected off it, and their cross.
// Degenerate inputs take fixed fallback axes, deterministic from the same inputs, so encode and decode rebuild the same frame.
// The vertex shader rebuilds the frame from current local positions, so offsets follow the deformation.
struct CornerNormalFrame {
    vec3 Normal, Ref, Ortho;
};

inline CornerNormalFrame ComputeCornerFrame(vec3 normal, std::span<const uint32_t> indices, std::span<const Vertex> vertices, uint32_t ci) {
    const auto n = Length(normal) > 0.f ? normal : vec3{0, 0, 1};
    const auto tri = ci / 3 * 3;
    const auto k = ci - tri;
    const auto p0 = vertices[indices[tri + k]].Position;
    const auto ref = [&]() -> vec3 {
        for (uint32_t other = 1; other < 3; ++other) {
            const auto edge = vertices[indices[tri + (k + other) % 3]].Position - p0;
            const auto rejected = edge - n * Dot(edge, n);
            const auto len = Length(rejected);
            // Require a stable perpendicular component before using an edge to anchor the frame.
            if (len > 1e-3f * Length(edge)) return rejected / len;
        }
        const auto axis = std::abs(n.x) < 0.5f ? vec3{1, 0, 0} : vec3{0, 1, 0};
        return Normalize(Cross(n, axis));
    }();
    return {n, ref, Cross(n, ref)};
}

// A custom normal as (polar, azimuth) angles in the corner frame.
inline vec2 EncodeNormalOffset(vec3 custom, const CornerNormalFrame &frame) {
    const auto polar = std::acos(std::clamp(Dot(custom, frame.Normal), -1.f, 1.f));
    const auto azimuth = std::atan2(Dot(custom, frame.Ortho), Dot(custom, frame.Ref));
    return {polar, azimuth};
}

inline vec3 DecodeNormalOffset(vec2 offset, const CornerNormalFrame &frame) {
    return std::cos(offset.x) * frame.Normal + std::sin(offset.x) * (std::cos(offset.y) * frame.Ref + std::sin(offset.y) * frame.Ortho);
}
