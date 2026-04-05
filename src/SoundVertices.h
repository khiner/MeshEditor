#pragma once

#include <entt_fwd.h>

#include <ranges>
#include <vector>

struct SoundVertices {
    uint32_t SelectedVertex() const { return Vertices[SelectedVertexIndex]; }
    std::optional<uint32_t> FindVertexIndex(uint32_t vertex) const {
        if (auto it = std::ranges::find(Vertices, vertex); it != Vertices.end()) {
            return std::ranges::distance(Vertices.begin(), it);
        }
        return {};
    }

    std::vector<uint32_t> Vertices;
    uint32_t SelectedVertexIndex; // The index in `Vertices` of the vertex currently selected for excitation.
};

// Force `Force` is being applied to this entity's mesh at `Vertex`.
struct VertexForce {
    uint32_t Vertex;
    float Force;
};
