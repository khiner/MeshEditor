#pragma once
#include "mesh/GeometrySelection.h"
#include "mesh/MeshTopology.h"
#include <optional>
struct Mesh;
// Plans only connectivity; positions remain in the canonical UMA vertex buffer.
std::optional<MeshTopologyTask> BeautifyFaceTask(const Mesh &, const GeometrySelection &, bool angle);
