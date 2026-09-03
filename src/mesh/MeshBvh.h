#pragma once

#include "gpu/AABB.h"
#include "gpu/Vertex.h"
#include "numeric/vec3.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

struct TrianglePoint {
    vec3 Position{0}, Weights{0};
};

// The point of triangle abc nearest `p`.
TrianglePoint ClosestPointOnTriangle(vec3 p, vec3 a, vec3 b, vec3 c);

struct SurfacePoint {
    std::array<uint32_t, 3> Vertices{};
    vec3 Weights{0};
};

// Stores a mesh-local triangle hierarchy independent of instance transforms.
struct MeshBvh {
    // A leaf names a triangle, an interior node its two children in `Nodes`.
    struct Node {
        AABB Box;
        uint32_t Left{0}, Right{0};
        static constexpr uint32_t NoChild{~0u};

        bool IsLeaf() const { return Right == NoChild; }
    };

    std::vector<Node> Nodes;

    // Mean surface curvature (1/length, in the mesh's own units), one entry per vertex.
    // A SurfacePoint's weights interpolate it at the point.
    std::vector<float> MeanCurvature;

    // Returns the enclosed volume in the mesh's coordinate units cubed.
    // Returns empty unless the surface is closed and manifold.
    std::optional<double> EnclosedVolume;

    // Returns the nearest mesh-local point and requires the source geometry used to build this nonempty hierarchy.
    SurfacePoint ClosestPoint(std::span<const Vertex> vertices, std::span<const uint32_t> triangle_indices, vec3 point) const;
};

MeshBvh BuildMeshBvh(std::span<const Vertex> vertices, std::span<const uint32_t> triangle_indices);
