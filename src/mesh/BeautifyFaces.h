#pragma once
#include "mesh/MeshTopology.h"
#include <optional>
struct Mesh;
struct MeshStore;
// Plans only connectivity; positions remain in the canonical UMA vertex buffer.
std::optional<MeshTopologyTask> BeautifyFaceTask(const MeshStore &, const Mesh &, bool angle);
