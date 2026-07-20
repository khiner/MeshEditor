#pragma once

// Constrained Delaunay tetrahedralization, after:
//
// Hang Si. "TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator." ACM Transactions on Mathematical Software 41(2), 2015.
//
// This is a port/rewrite of that paper's 1.6.0 reference implementation.
// The output matches the reference exactly, mesh and flip sequence, on every case of our corpus across configurations.
// `script/tet_dev/corpus_snapshot.txt` records that output.

#include "mesh/TetMesh.h"

#include <expected>
#include <span>
#include <string>

namespace tetra {
struct Options {
    // Insert interior points until tets meet a circumradius-to-shortest-edge ratio of 2, where the fixed surface allows.
    // This gates refinement alone: sliver repair and vertex optimisation run either way.
    bool Quality{false};
    // Split any tet whose volume exceeds this.
    // The bound is an absolute volume in the input's own coordinate units, so it scales with the model rather than with its bounding box.
    // Setting it turns Quality on as well.
    // Zero leaves element size unconstrained.
    double MaxVolume{0};
};

// Wall-clock seconds per stage, with size and effort counters.
struct Profile {
    double DelaunaySeconds{}, RecoverSeconds{}, CarveSeconds{}, RefineSeconds{};
    // Recovery, split by what its passes spend the time on.
    // What is left over is the constraint setup and the surface-marking sweep on either side of the pass loop.
    double SegmentSeconds{}, FaceSeconds{}, SuppressSeconds{};
    // Sizes describe the mesh the surviving build produced.
    uint32_t TetCount{}, SteinerCount{};
    // Interior tets in the Delaunay triangulation of the input points, before the surface is met.
    uint32_t DelaunayTetCount{};
    // Steiner points on the surface that recovery could not remove, and the interior points suppression placed instead of them.
    uint32_t BdrySteinerCount{}, VolSteinerCount{};
    uint32_t FlipCount{}, SplitCount{}, MissingEdgeCount{}, MissingFaceCount{}, Builds{};
};

struct Result {
    TetMesh Mesh;
    Profile Profile;
};

// Fill a triangulated surface with tetrahedra.
//
// The input must be:
//  - Closed: the triangles bound a solid with no holes, or the fill leaks out and fails.
//  - Non-self-intersecting: no triangle passes through another, or recovery fails.
// The input may be, unlike most tetrahedralizers:
//  - Non-manifold: an edge may be shared by more than two triangles (internal walls).
//  - Wound any way: triangle orientation is ignored.
// The input format is triangles only (three indices per face), never polygons.
//
// The surface appears exactly in the output: input vertex i keeps index i, every input triangle is a boundary face, and added (Steiner) points lie strictly inside.
// Returns an error string for open, self-intersecting, or otherwise unrecoverable surfaces.
std::expected<Result, std::string> Tetrahedralize(std::span<const dvec3> points, std::span<const uint32_t> triangle_indices, Options options = {});
} // namespace tetra
