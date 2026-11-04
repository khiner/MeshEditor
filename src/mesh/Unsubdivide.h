#pragma once
#include "mesh/MeshTopology.h"
struct Mesh;
namespace state {
struct Registry;
}
// Plans one alternating reduction of the selected vertices as a topology task, or none when nothing reduces.
std::optional<MeshTopologyTask> UnsubdivideTask(state::Registry &, const Mesh &, const GeometrySelection &);
