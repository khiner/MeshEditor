#pragma once
#include "mesh/GeometrySelection.h"
struct MeshStore;
struct MeshPipelines;

// The selected faces needing reversal for each closed manifold component holding a selected face to enclose positive signed volume.
std::vector<uint32_t> RecalculateFaceFlips(const MeshStore &, const MeshPipelines &, uint32_t id, const GeometrySelection &);
