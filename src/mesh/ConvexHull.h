#pragma once
#include <cstdint>
#include <span>
#include <vector>

namespace mtl {
struct ComputeChain;
}
namespace state {
struct Registry;
}

// The outward faces of each point set's convex hull as loops of indices into the set.
// A hull is triangles, or one polygon wound around the normal of its first three seeds for a planar set, or empty for a degenerate or abandoned set.
// `positions` is one packed_float3 run in the chain's scratch, set s holding points first_point[s] to first_point[s + 1], then one lifted apex per set.
// The quickhull runs in rounds, each submitting the chain once.
std::vector<std::vector<std::vector<uint32_t>>> ConvexHulls(state::Registry &, mtl::ComputeChain &, uint32_t positions, std::span<const uint32_t> first_point);
