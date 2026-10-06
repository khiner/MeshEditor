#pragma once
#include "gpu/VertexPositionEditPushConstants.h"
#include "mesh/GeometrySelection.h"
#include <vector>
struct Mesh;
// One pair of rails per selected vertex, in selection order. Reads canonical UMA
// positions directly; no packed copy or traversal of unrelated mesh elements.
std::vector<EdgeSlideDirections> PlanEdgeSlide(const Mesh &, const GeometrySelection &, vec3 direction, vec3 scale, uint32_t reference);
