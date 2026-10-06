#pragma once
#include "mesh/GeometrySelection.h"
#include <cstdint>
#include <span>
#include <vector>
struct MeshStore;
struct MeshPipelines;

// Faces needing reversal within manifold components touched by each explicit selection.
// Connectivity and parity are metadata; centers and orientation tests stay on the GPU.
std::vector<std::vector<uint32_t>> RecalculateFaceFlips(MeshStore &, const MeshPipelines &, std::span<const uint32_t> meshes, std::span<const GeometrySelection>, bool inside);
