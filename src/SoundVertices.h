#pragma once

#include <ranges>
#include <vector>

// Derived state, do not mutate directly.
// In Samples mode, mirrors keys of VertexSamples::PathByVertex.
// In Modal mode, mirrors ModalModes::Vertices.
// Rebuilt by a reactive handler in AudioSystem when any of those primary sources changes.
struct SoundVertices {
    std::optional<uint32_t> FindVertexIndex(uint32_t vertex) const {
        if (auto it = std::ranges::find(Vertices, vertex); it != Vertices.end()) {
            return std::ranges::distance(Vertices.begin(), it);
        }
        return {};
    }

    std::vector<uint32_t> Vertices;
};

// Force `Force` is being applied to this entity's mesh at `Vertex`.
struct VertexForce {
    uint32_t Vertex;
    float Force;
};
