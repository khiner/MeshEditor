#pragma once

#include <unordered_map>
#include <vector>

#include "mesh/MeshElement.h"
#include "mesh/MeshBuffers.h"

struct MeshElementBuffers {
    std::vector<std::unordered_map<MeshElement, MeshBuffers>> ElementBuffers, NormalIndicatorBuffers;
};
