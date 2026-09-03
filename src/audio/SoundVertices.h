#pragma once

#include "Range.h"

#include <cstdint>
#include <optional>
#include <ranges>
#include <span>

// Derived state, do not mutate directly.
// In Samples mode, mirrors keys of VertexSamples::PathByVertex.
// In Modal mode, mirrors ModalModes::Vertices.
// Rebuilt by a reactive handler in AudioSystem when any of those primary sources changes.
// The handles live in the mesh store's sound-vertex arena, which is the only copy.
struct SoundVertices {
    Range Vertices{};
};

inline std::optional<uint32_t> FindSoundVertexIndex(std::span<const uint32_t> vertices, uint32_t vertex) {
    if (auto it = std::ranges::find(vertices, vertex); it != vertices.end()) {
        return uint32_t(std::ranges::distance(vertices.begin(), it));
    }
    return {};
}

// Force `Force` is being applied to this entity's mesh at `Vertex`.
struct VertexForce {
    uint32_t Vertex;
    float Force;
    float ContactSpeed{1.f}; // Normal impact speed in m/s, driving the Hertz contact time. Collision events set this per strike.
};
