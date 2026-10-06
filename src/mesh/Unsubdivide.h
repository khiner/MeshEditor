#pragma once
#include "mesh/GeometrySelection.h"
#include "mesh/MeshTopology.h"
#include <optional>
struct Mesh;
// Plans one alternating reduction from selected connectivity. Geometry and
// attributes stay in canonical UMA and transfer through the shared GPU edit.
std::optional<MeshTopologyTask> UnsubdivideTask(const Mesh &, const GeometrySelection &);
