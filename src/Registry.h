#pragma once

// todo Placeholder for entt registry.
// For now, including all component types here (mat4/Mesh/...).
// Later with entt, consumers can include only what they use.

#include <vector>

#include "numeric/mat4.h"

#include "mesh/Mesh.h"
#include "mesh/VkMeshBuffers.h"

struct Registry {
    std::vector<mat4> Models;
    std::vector<Mesh> Meshes;
    std::vector<std::unordered_map<MeshElement, VkMeshBuffers>> NormalIndicatorBuffers;
    std::vector<std::unordered_map<MeshElement, VkMeshBuffers>> ElementBuffers;

    void AddMesh(Mesh &&mesh) {
        Meshes.emplace_back(std::move(mesh));
        Models.emplace_back(1);
        ElementBuffers.emplace_back();
    }
};