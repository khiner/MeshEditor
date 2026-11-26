#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

struct MeshStore;
struct MeshPipelines;
namespace mtl {
struct ComputeChain;
}

// Records the passes that find the boundary halfedge loops through `edges`, or through every boundary edge when it is empty, at most `max_sides` long when positive, in lanes of the chain, which gather with the other lanes' passes.
// Calls `then` with the loops after the submits that complete them.
// Each loop pairs its vertices with the boundary halfedges leaving them, in face winding order from the loop's lowest halfedge.
// The loops come in order of those lowest halfedges.
// A face with no neighbor bounds no loop.
void RecordBoundaryLoops(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, std::span<const uint32_t> edges, uint32_t max_sides, std::function<void(std::vector<std::vector<std::array<uint32_t, 2>>>)> then);
