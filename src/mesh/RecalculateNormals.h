#pragma once
#include "mesh/GeometrySelection.h"

#include <functional>
struct MeshStore;
struct MeshPipelines;
namespace mtl {
struct ComputeChain;
} // namespace mtl

// The selected faces needing reversal for each closed manifold component holding a selected face to enclose positive signed volume.
// Records the passes on the chain, each round of their label propagation once the chain completed the round before it, and calls `then` with the faces once the chain completed them.
void RecordFaceFlips(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, const GeometrySelection &, std::function<void(std::vector<uint32_t>)> then);
