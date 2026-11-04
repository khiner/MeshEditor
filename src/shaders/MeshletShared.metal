#pragma once
#include "CompactPresent.metal"

#include "Bindless.metal"
#include "gpu/MeshletGeometryEncoding.h"
#include "gpu/MeshletRecord.h"
#include "gpu/MeshPrimitiveTopology.h"
#include "gpu/PrimitiveRecord.h"

// Cluster payload pointers, offset by the record's range bases.
struct MeshletPayloads {
    device const uint *TriangleIds, *VertexCorners;
    device const MeshletTriangle *LocalTriangles;
};
inline MeshletPayloads ReadMeshletPayloads(device const BindlessSet &b, MeshletPayloadSlots slots, MeshletPayloadBases bases) {
    return {BindlessBuffer(uint, b.Buffer, slots.TriangleIds) + bases.TriangleIds, BindlessBuffer(uint, b.Buffer, slots.VertexCorners) + bases.VertexCorners, BindlessBuffer(MeshletTriangle, b.Buffer, slots.LocalTriangles) + bases.LocalTriangles};
}

inline uint MeshletSourceVertex(MeshletPayloads payloads, MeshletRecord meshlet, uint i) {
    return payloads.VertexCorners[meshlet.VertexOffset + i];
}

inline uint MeshletPrimitiveTopology(MeshletRecord meshlet) {
    return meshlet.Topology;
}

inline uint MeshletPrimitiveMaterialIndex(const thread Scene &scene, PrimitiveRecord primitive) {
    if (primitive.PrimitiveMaterialOffset == InvalidOffset) return 0u;
    return scene.PrimitiveMaterials(scene.View.PrimitiveMaterialSlot)[primitive.PrimitiveMaterialOffset + primitive.PrimitiveIndex];
}


// A cluster vertex is a corner handle for triangles and a vertex handle otherwise.
inline uint MeshletVertexId(
    const thread Scene &scene, DrawData draw, uint topology, uint source_vertex
) {
    if (topology != uint(MeshPrimitiveTopology::Triangle)) return source_vertex;
    return scene.CornerVertex(draw, source_vertex);
}

inline Transform MeshletWorld(const thread Scene &scene, DrawData draw) {
    const uint model_slot = scene.View.ModelSlotOverride != InvalidSlot ? scene.View.ModelSlotOverride : draw.ModelSlot;
    return scene.Models(model_slot)[draw.FirstInstance];
}
