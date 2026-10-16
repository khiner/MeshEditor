#pragma once
#include "mesh/MeshTopology.h"
struct Mesh;
struct MeshStore;
// Plan valid, least-error edge collapses over selected incidence.
// Positions are read from canonical UMA; only proposed replacements and connectivity are stored.
std::optional<MeshTopologyTask> DecimateTask(const MeshStore &, const Mesh &, const GeometrySelection &, float ratio, std::span<const uint32_t> locked_faces = {});
