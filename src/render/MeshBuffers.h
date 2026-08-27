#pragma once

#include "Range.h"
#include "SlottedRange.h"
#include "gpu/Element.h"

#include <unordered_map>

enum class IndexKind {
    Face,
    Edge,
    Vertex
};

struct RenderBuffers {
    RenderBuffers(Range vertices, SlottedRange indices, IndexKind index_type)
        : Vertices(vertices), Indices(indices), IndexType(index_type) {}

    Range Vertices;
    SlottedRange Indices;
    IndexKind IndexType;
};

struct MeshBuffers {
    MeshBuffers(SlottedRange vertices, SlottedRange face_indices, SlottedRange edge_indices, SlottedRange vertex_indices)
        : Vertices{vertices}, FaceIndices{face_indices}, EdgeIndices{edge_indices}, VertexIndices{vertex_indices} {}
    MeshBuffers(const MeshBuffers &) = delete;
    MeshBuffers &operator=(const MeshBuffers &) = delete;
    SlottedRange Vertices;
    SlottedRange FaceIndices, EdgeIndices, VertexIndices;
    Range Primitives, Meshlets, MeshletTriangles, MeshletVertices, MeshletLocalTriangles, MeshletEditEdges;
    // The cluster LOD DAG, whose coarse clusters take their own ranges in the meshlet geometry arenas.
    // A mesh whose clusters fit one partition carries no groups or coarse geometry, and its nodes are
    // one never-pruned span per primitive.
    Range ClusterGroups, LodNodes, CoarseVertices, CoarseLocalTriangles;
};

// Adjacency indices for bone silhouette edge detection (stored on armature object entities).
struct BoneAdjacencyIndices {
    SlottedRange Indices;
};
