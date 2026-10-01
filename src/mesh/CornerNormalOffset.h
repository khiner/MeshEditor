#pragma once

#include "numeric/VectorMath.h"

#include "mesh/Mesh.h"
#include "numeric/vec2.h"

#include <algorithm>
#include <cmath>

// Orthonormal frame anchoring a corner's authored-normal offset.
// Axes: the derived normal, the corner's first non-degenerate polygon edge projected off it, and their cross.
// Degenerate inputs take fixed fallback axes, deterministic from the same inputs, so encode and decode rebuild the same frame.
// The vertex shader rebuilds the frame from current local positions, so offsets follow the deformation.
struct CornerNormalFrame {
    vec3 Normal, Ref, Ortho;
};

inline CornerNormalFrame ComputeCornerFrame(vec3 normal, vec3 p0, vec3 next, vec3 previous) {
    const auto n = Length(normal) > 0.f ? normal : vec3{0, 0, 1};
    const auto ref = [&]() -> vec3 {
        for (const auto p : {next, previous}) {
            const auto edge = p - p0;
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

inline CornerNormalFrame ComputeCornerFrame(vec3 normal, const Mesh &mesh, uint32_t corner) {
    const Mesh::HH h{corner};
    const auto &c = mesh.GetConnectivity();
    const auto position = [&](Mesh::HH at) { return mesh.GetPosition(mesh.GetToVertex(at)); };
    return ComputeCornerFrame(normal, position(h), position(c.Next(h)), position(c.Previous(h)));
}

// A custom normal as (polar, azimuth) angles in the corner frame.
inline vec2 EncodeNormalOffset(vec3 custom, const CornerNormalFrame &frame) {
    const auto polar = std::acos(std::clamp(Dot(custom, frame.Normal), -1.f, 1.f));
    const auto x = Dot(custom, frame.Ref), y = Dot(custom, frame.Ortho);
    // Azimuth is undefined at the poles.
    // Keep a finite canonical value.
    const auto azimuth = x == 0.f && y == 0.f ? 0.f : std::atan2(y, x);
    return {polar, azimuth};
}

inline vec3 DecodeNormalOffset(vec2 offset, const CornerNormalFrame &frame) {
    return std::cos(offset.x) * frame.Normal + std::sin(offset.x) * (std::cos(offset.y) * frame.Ref + std::sin(offset.y) * frame.Ortho);
}
