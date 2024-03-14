#pragma once

#include <unordered_map>
#include <vector>

#include "mesh/MeshBuffers.h"
#include "mesh/MeshElement.h"

// Contiguous vectors of mesh data for Vulkan.
struct MeshVkData {
    std::vector<Model> Models;
    std::vector<std::unordered_map<MeshElement, MeshBuffers>> ElementBuffers, NormalIndicatorBuffers;
};
