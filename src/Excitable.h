#pragma once

#include <entt_fwd.h>

#include <ranges>
#include <vector>

struct Excitable {
    uint32_t SelectedVertex() const { return ExcitableVertices[SelectedVertexIndex]; }
    std::optional<uint32_t> FindVertexIndex(uint32_t vertex) const {
        if (auto it = std::ranges::find(ExcitableVertices, vertex); it != ExcitableVertices.end()) {
            return std::ranges::distance(ExcitableVertices.begin(), it);
        }
        return std::nullopt;
    }

    std::vector<uint32_t> ExcitableVertices;
    uint32_t SelectedVertexIndex; // The index in `ExcitableVertices` of the vertex currently selected for excitation.
};

// If an entity has this component, its mesh is being excited at this vertex.
struct ExcitedVertex {
    uint32_t Vertex;
    float Force;
};
