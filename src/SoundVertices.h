#pragma once

#include <ranges>
#include <vector>

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
