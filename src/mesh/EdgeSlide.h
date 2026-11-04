#pragma once
#include "gpu/VertexPositionEditPushConstants.h"
#include "mesh/GeometrySelection.h"
struct Mesh;
struct MeshStore;
struct MeshPipelines;
// One pair of rails per selected vertex, in selection order.
std::vector<EdgeSlideDirections> PlanEdgeSlide(const MeshStore &, const MeshPipelines &, const Mesh &, const GeometrySelection &, vec3 direction, vec3 scale, uint32_t reference);
