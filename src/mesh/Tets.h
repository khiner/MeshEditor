#pragma once

#include "mesh/TetMeshData.h"
#include "mesh/Tetrahedralize.h"

// Quadric edge-collapse the surface to `ratio` of its triangles in place, then drop unreferenced vertices.
// A no-op when ratio >= 1. A non-self-intersecting input stays non-self-intersecting: collapses that fold the surface through itself are found and retried with that neighbourhood frozen, so a region that cannot be coarsened safely keeps its resolution.
void SimplifySurface(std::vector<vec3> &positions, std::vector<uint32_t> &triangle_indices, float ratio);

// Fill the closed triangle surface with tetrahedra.
// The surface appears exactly in the output.
// The error string names the failure (e.g. a self-intersecting input surface).
// Simplify the surface first with SimplifySurface if a lower resolution is wanted.
std::expected<tetra::Result, std::string> GenerateTets(std::vector<vec3> positions, std::vector<uint32_t> triangle_indices, tetra::Options options = {});

// positions are divided by `scale`.
TetMeshData BuildTetMeshData(const TetMesh &, vec3 scale);
