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

// The render arena ranges of one mesh record, held by store id for the record's lifetime.
struct MeshBuffers {
    SlottedRange Vertices;
    SlottedRange FaceIndices, EdgeIndices, VertexIndices;
    Range MeshRecord, Primitives, Meshlets, MeshletTriangles, MeshletVertices, MeshletLocalTriangles, MeshletEditEdges;
    // Meshes without coarse geometry use one unpruned span node per primitive.
    Range ClusterGroups, LodNodes, CoarseVertices, CoarseLocalTriangles;
};

struct BoneAdjacencyIndices {
    SlottedRange Indices;
};
