#pragma once
#include "mesh/MeshTopology.h"
struct Mesh;
namespace state {
struct Registry;
}
// Rotates the selected triangles' shared edges until each quad they form passes the Delaunay empty-circle test.
std::optional<MeshTopologyTask> BeautifyFaceTask(state::Registry &, const Mesh &, const GeometrySelection &);
