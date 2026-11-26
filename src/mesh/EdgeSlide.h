#pragma once
#include "gpu/VertexPositionEditPushConstants.h"
#include "mesh/GeometrySelection.h"
struct Mesh;
class LocalMesh;
// One pair of rails per selected vertex, in selection order, from the local mesh around the selected vertices.
std::vector<EdgeSlideDirections> PlanEdgeSlide(const Mesh &, const LocalMesh &, const GeometrySelection &, vec3 direction, vec3 scale, uint32_t reference);
