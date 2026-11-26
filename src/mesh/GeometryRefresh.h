#pragma once

#include "mesh/PositionOperations.h"

struct MeshRetessellation {
    uint32_t StoreId;
    // The storage of the Faces work, or null for every face of the mesh.
    const BufferArena<uint32_t> *Work;
    ElementWork Faces;
    ElementWork ChangedVertices;
};

// Canonical tessellation/tangent maintenance shared by procedural and editor publication.
// Each result is its input's changed triangles, work in chain.Scratch whose count the chain's next submit reports, or no work for an input without polygons to retessellate.
std::vector<ElementWork> EncodeRetessellation(state::Registry &, mtl::ComputeChain &, std::span<const MeshRetessellation>);
// A new record's faces of more than PolygonThreadCorners corners, each with its first corner in polygon scratch, and the chain scratch words its triangles take.
struct NewTriangles {
    uint32_t StoreId;
    std::vector<uvec2> Polygons{};
    uint32_t Corners{};
    uint64_t Words{};
};
// Lists the polygons of a new record whose element sets are dense, after its face loops.
NewTriangles PlanNewTriangles(const MeshStore &, uint32_t id);
// Records the first-triangle tables and render triangles of the planned records from their face loops and positions.
void EncodeNewTriangles(state::Registry &, mtl::ComputeChain &, std::span<const NewTriangles>);
// Records canonical tessellation, normals and bounds over every vertex of each record.
// Each result is its record's changed triangles, as EncodeRetessellation returns them.
std::vector<ElementWork> EncodeRecordRefresh(state::Registry &, mtl::ComputeChain &, std::span<const uint32_t> ids);
// A record whose vertices a position operation recorded on a chain moved.
struct PositionRefresh {
    uint32_t StoreId;
    const PositionOperationChange *Change;
};
// Records canonical tessellation, normals and bounds around the vertices each position operation moved, every record in one batch.
// Submits the chain once to read the neighborhoods of the operations that did not move every vertex.
// Returns each record's triangles whose tessellation changed, as work in chain.Scratch whose count the chain's next submit reports, or no work for a mesh without polygons to retessellate.
std::vector<ElementWork> EncodePositionRefresh(state::Registry &, mtl::ComputeChain &, std::span<const PositionRefresh>);
// Completes position operations, including sparse canonical tessellation, normals and bounds.
std::optional<PositionOperationChange> ExecuteGeometryPositions(state::Registry &, const PositionOperationTarget &, PositionEditOp, float factor, uint32_t repeat = 1u, const PositionOperationOptions & = {});
