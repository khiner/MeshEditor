#pragma once

// todo Placeholder for entt registry.
// For now, including all component types here (matr/Mesh/...).
// Later with entt, consumers can include only what they use.

#include <vector>

#include "numeric/mat4.h"

#include "mesh/Mesh.h"

struct Registry {
    std::vector<mat4> Models{};
    std::vector<Mesh> Meshes;
};
