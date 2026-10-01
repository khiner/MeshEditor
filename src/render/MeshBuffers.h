#pragma once

#include "Range.h"
#include "SlottedRange.h"
#include "gpu/Element.h"

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

// Render metadata for one store record. Sparse membership owns cluster storage.
struct MeshBuffers {
    SlottedRange Vertices;
    SlottedRange FaceIndices, EdgeIndices, VertexIndices;
    Range MeshRecord, Primitives, Meshlets, MeshletTriangles, MeshletVertices, MeshletLocalTriangles;
    // Published sparse roots own cluster/payload and primitive allocations.
    // Before publication, construction ranges own their provisional storage.
    uint32_t MeshletRoot{InvalidOffset}, PrimitiveRoot{InvalidOffset}, StoreId{InvalidOffset};
    uint32_t SpatialRoot{InvalidOffset};
    Range PrimitiveRoutes{}; // Render primitive handles indexed by source primitive.
    uint64_t MeshletRevision{};
    uint32_t Level0Count{};
    uint32_t RenderTopology{InvalidOffset};
    uint32_t ElementMeshletOrigin{InvalidOffset}, ElementMeshletBlockCount{};
    // Meshes without coarse geometry use one unpruned span node per primitive.
    // Ranges describe construction placement.
    // Roots own the live allocations.
    Range ClusterGroups, LodNodes, CoarseVertices, CoarseLocalTriangles;
    uint32_t GroupRoot{InvalidOffset}, NodeRoot{InvalidOffset};
    // Finest meshlets whose canonical positions changed since coarse repair.
    // Membership is tracked with the render history and survives cold restore.
    uint32_t PositionDirtyRoot{InvalidOffset};
    // Stale LOD groups, whose coarse clusters rebuild when the mesh leaves edit mode.
    // The root follows the mesh through preview and history.
    uint32_t DirtyGroupRoot{InvalidOffset};

    // Bindless slots are runtime bindings.
    // History stores canonical allocation identities.
    // RestoreMeshBindings resolves slots in the current context.
    static auto serialize(auto &archive,auto &self) {
        return archive(self.Vertices.Offset,self.Vertices.Count,
            self.FaceIndices.Offset,self.FaceIndices.Count,self.EdgeIndices.Offset,self.EdgeIndices.Count,
            self.VertexIndices.Offset,self.VertexIndices.Count,
            self.MeshRecord,self.Primitives,self.Meshlets,self.MeshletTriangles,self.MeshletVertices,self.MeshletLocalTriangles,
            self.MeshletRoot,self.PrimitiveRoot,self.StoreId,self.SpatialRoot,self.PrimitiveRoutes,self.MeshletRevision,self.Level0Count,
            self.RenderTopology,self.ElementMeshletOrigin,self.ElementMeshletBlockCount,
            self.ClusterGroups,self.LodNodes,self.CoarseVertices,self.CoarseLocalTriangles,self.GroupRoot,self.NodeRoot,
            self.PositionDirtyRoot,self.DirtyGroupRoot);
    }
};
