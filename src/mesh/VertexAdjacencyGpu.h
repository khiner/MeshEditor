#pragma once

#include <span>

#include "state/Entity.h"

// Fill the listed meshes' vertex CSR incidence tables on the GPU, then wait for the fill.
// Call before anything reads the tables, which for a new mesh means before its normals derive.
void BuildVertexAdjacencyNow(state::Scene &, std::span<const state::Entity> mesh_entities);
