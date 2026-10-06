#pragma once
#include "gpu/VertexPositionEditPushConstants.h"
#include <vector>
struct Mesh;
struct MeshStore;
// One pair of rails per selected vertex, in selection order. Reads canonical UMA
// positions directly; no packed copy or traversal of unrelated mesh elements.
std::vector<EdgeSlideDirections> PlanEdgeSlide(const MeshStore &, const Mesh &, vec3 direction, vec3 scale, uint32_t reference);
