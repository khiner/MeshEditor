#pragma once

#include "gpu/AABB.h"
#include "gpu/Vertex.h"
#include "numeric/vec3.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

// A point on a triangle, with barycentric weights over the triangle's three corners in order.
struct TrianglePoint {
    vec3 Position{0}, Weights{0};
};

// The point of triangle abc nearest `p`.
TrianglePoint ClosestPointOnTriangle(vec3 p, vec3 a, vec3 b, vec3 c);

// A point on a mesh's surface, in the mesh's own coordinates.
// The weights are barycentric over `Vertices`, so any per-vertex quantity interpolates at that point.
// Blending the positions recovers the point itself.
struct SurfacePoint {
    std::array<uint32_t, 3> Vertices{}; // The triangle the point lies in.
    vec3 Weights{0};
};

// Bounding volume hierarchy over a mesh's triangles, in the mesh's own coordinates, for closest-point queries.
// Geometry is rigid, so one hierarchy is built per mesh and serves every node instancing it at every scale.
// It holds only the tree, so queries pass the vertices and triangle indices in.
struct MeshBvh {
    // A leaf names a triangle, an interior node its two children in `Nodes`.
    struct Node {
        AABB Box;
        uint32_t Left{0}, Right{0};
        static constexpr uint32_t NoChild{~0u}; // Held in Right, which makes Left a triangle rather than a node.

        bool IsLeaf() const { return Right == NoChild; }
    };

    std::vector<Node> Nodes; // Emitted in post order, so the root is last.

    // Mean surface curvature (1/length, in the mesh's own units), one entry per vertex.
    // A SurfacePoint's weights interpolate it at the point.
    std::vector<float> MeanCurvature;

    // Volume the surface encloses, in the mesh's own units cubed. Empty unless it is closed and manifold.
    std::optional<double> EnclosedVolume;

    // The point of the surface nearest `point`, both in the mesh's own coordinates.
    // Takes the same vertices and triangle indices this was built over, since the nodes index them.
    // The mesh must have at least one triangle.
    SurfacePoint ClosestPoint(std::span<const Vertex> vertices, std::span<const uint32_t> triangle_indices, vec3 point) const;
};

MeshBvh BuildMeshBvh(std::span<const Vertex> vertices, std::span<const uint32_t> triangle_indices);
