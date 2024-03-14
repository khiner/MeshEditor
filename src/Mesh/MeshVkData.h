#pragma once

#include <unordered_map>
#include <vector>

#include "mesh/MeshBuffers.h"
#include "mesh/MeshElement.h"

// Contiguous vectors of mesh data for Vulkan.
struct MeshVkData {
    using MeshElementBuffers = std::unordered_map<MeshElement, MeshBuffers>;
    std::vector<Model> Models;
    std::vector<MeshElementBuffers> ElementBuffers, NormalIndicatorBuffers;
};
