#pragma once

#include <cstdint>
#include <vector>

#include <entt/entity/entity.hpp>

struct Excitable {
    Excitable() = default;
    Excitable(std::vector<uint32_t> vertices)
        : ExcitableVertices(std::move(vertices)), SelectedVertex(!ExcitableVertices.empty() ? ExcitableVertices.front() : 0) {}

    std::vector<uint32_t> ExcitableVertices;
    uint32_t SelectedVertex; // The vertex currently selected for excitation.
};

// If an entity has this component, its mesh is being excited at this vertex.
struct ExcitedVertex {
    uint32_t Vertex;
    float Force;
    entt::entity IndicatorEntity{};
};
