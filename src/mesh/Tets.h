#pragma once

#include "mesh/TetMeshData.h"
#include "mesh/Tetrahedralize.h"

// Quadric edge-collapse the surface to `ratio` of its triangles in place, then drop unreferenced vertices.
// Preserves non-self-intersection and retains the original resolution in unsafe regions.
void SimplifySurface(std::vector<vec3> &positions, std::vector<uint32_t> &triangle_indices, float ratio);

// Fill the closed triangle surface with tetrahedra.
// The surface appears exactly in the output.
// The error string names the failure (e.g. a self-intersecting input surface).
// Simplify the surface first to request a lower resolution.
std::expected<tetra::Result, std::string> GenerateTets(std::vector<vec3> positions, std::vector<uint32_t> triangle_indices, tetra::Options options = {});

// Divides output positions by `scale`.
TetMeshData BuildTetMeshData(const TetMesh &, vec3 scale);
