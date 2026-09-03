#pragma once

// Constrained Delaunay tetrahedralization based on:
// Hang Si. "TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator." ACM Transactions on Mathematical Software 41(2), 2015.
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
    // Quality controls refinement.
    // Sliver repair and vertex optimization always run.
    bool Quality{false};
    // Split any tet whose volume exceeds this.
    // The bound is an absolute volume in the input's own coordinate units, so it scales with the model rather than with its bounding box.
    // A positive value also enables Quality.
    // Zero permits any element size.
    double MaxVolume{0};
};

struct Profile {
    double DelaunaySeconds{}, RecoverSeconds{}, CarveSeconds{}, RefineSeconds{};
    // Remaining recovery time covers constraint setup and the surrounding surface-marking sweeps.
    double SegmentSeconds{}, FaceSeconds{}, SuppressSeconds{};
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

// Fills a closed, non-self-intersecting triangle surface while preserving input vertex indices and boundary faces.
// Accepts non-manifold edges and arbitrary triangle winding.
// Returns an error string for open, self-intersecting, or otherwise unrecoverable surfaces.
std::expected<Result, std::string> Tetrahedralize(std::span<const dvec3> points, std::span<const uint32_t> triangle_indices, Options options = {});
} // namespace tetra
