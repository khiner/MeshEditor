#pragma once

#include <span>

#include <entt/entity/fwd.hpp>

// Fill the listed meshes' vertex CSR incidence tables on the GPU, then wait for the fill.
// Call before anything reads the tables, which for a new mesh means before its normals derive.
void BuildVertexAdjacencyNow(entt::registry &, std::span<const entt::entity> mesh_entities);
