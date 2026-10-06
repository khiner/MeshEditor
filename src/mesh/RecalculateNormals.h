#pragma once
#include <cstdint>
#include <span>
#include <vector>
namespace state {
struct Scene;
}

// Selected faces that need reversing, using only manifold components touched by the selection.
// Connectivity and parity are metadata; centers and orientation tests stay on the GPU.
std::vector<std::vector<uint32_t>> RecalculateFaceFlips(state::Scene &, std::span<const uint32_t> meshes, bool inside);
