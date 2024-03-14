#pragma once

#include <unordered_map>
#include <vector>

#include "mesh/MeshBuffers.h"
#include "mesh/MeshElement.h"

struct MeshVkData {
    using MeshElementBuffers = std::unordered_map<MeshElement, MeshBuffers>;
    std::vector<MeshElementBuffers> ElementBuffers, NormalIndicatorBuffers;
};
