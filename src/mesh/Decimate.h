#pragma once
#include "mesh/GeometrySelection.h"
#include "mesh/MeshTopology.h"
#include <optional>
#include <span>
struct Mesh;
struct MeshStore;
// Plan valid, least-error edge collapses over selected incidence. Positions are
// read from canonical UMA; only proposed replacements and connectivity are stored.
std::optional<MeshTopologyTask> DecimateTask(const MeshStore &, const Mesh &, const GeometrySelection &, float ratio, std::span<const uint32_t> locked_faces = {});
