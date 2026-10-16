#pragma once
#include "mesh/MeshTopology.h"
struct Mesh;
// Plans one alternating reduction from selected connectivity. Geometry and
// attributes stay in canonical UMA and transfer through the shared GPU edit.
std::optional<MeshTopologyTask> UnsubdivideTask(const Mesh &, const GeometrySelection &);
