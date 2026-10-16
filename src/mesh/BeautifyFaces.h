#pragma once
#include "mesh/MeshTopology.h"
struct Mesh;
// Plans only connectivity; positions remain in the canonical UMA vertex buffer.
std::optional<MeshTopologyTask> BeautifyFaceTask(const Mesh &, const GeometrySelection &, bool angle);
