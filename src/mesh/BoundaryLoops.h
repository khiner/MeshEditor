#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace state {
struct Registry;
}

// The boundary halfedge loops through `edges`, or through every boundary edge when it is empty, at most `max_sides` long when positive.
// Each loop pairs its vertices with the boundary halfedges leaving them, in face winding order from the loop's lowest halfedge.
// The loops come in order of those lowest halfedges.
// A face with no neighbor bounds no loop.
std::vector<std::vector<std::array<uint32_t, 2>>> BoundaryLoops(state::Registry &, uint32_t id, std::span<const uint32_t> edges, uint32_t max_sides);
