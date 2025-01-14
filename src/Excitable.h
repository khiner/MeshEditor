#pragma once

#include <entt/entity/entity.hpp>

#include <cstdint>
#include <ranges>
#include <vector>

struct Excitable {
    Excitable() = default;
    Excitable(std::vector<uint32_t> vertices) : ExcitableVertices(std::move(vertices)), SelectedVertexIndex(0) {}

    uint32_t SelectedVertex() const { return ExcitableVertices[SelectedVertexIndex]; }
    void SelectVertex(uint32_t vertex) {
        auto it = std::ranges::find(ExcitableVertices, vertex);
        if (it != ExcitableVertices.end()) SelectedVertexIndex = std::ranges::distance(ExcitableVertices.begin(), it);
    }

    std::vector<uint32_t> ExcitableVertices;
    uint32_t SelectedVertexIndex; // The index in `ExcitableVertices` of the vertex currently selected for excitation.
};

// If an entity has this component, its mesh is being excited at this vertex.
struct ExcitedVertex {
    uint32_t Vertex;
    float Force;
    entt::entity IndicatorEntity{};
};
