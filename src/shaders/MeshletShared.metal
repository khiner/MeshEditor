#ifndef MESHLET_SHARED_MSL
#define MESHLET_SHARED_MSL

#include "CompactPresent.metal"

#include "Bindless.metal"
#include "gpu/MeshletGeometryEncoding.h"
#include "gpu/MeshletRecord.h"
#include "gpu/MeshPrimitiveTopology.h"
#include "gpu/PrimitiveRecord.h"

// A record's cluster payloads from the bases of its ranges, which payload offsets index.
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


// The vertex handle of a cluster vertex, which names a corner of a triangle cluster and a vertex otherwise.
inline uint MeshletVertexId(
    const thread Scene &scene, DrawData draw, uint topology, uint source_vertex
) {
    if (topology != uint(MeshPrimitiveTopology::Triangle)) return source_vertex;
    return scene.CornerVertex(draw, source_vertex);
}

// Returns the motion-blur model override or the draw's current world transform.
inline Transform MeshletWorld(const thread Scene &scene, DrawData draw) {
    const uint model_slot = scene.View.ModelSlotOverride != InvalidSlot ? scene.View.ModelSlotOverride : draw.ModelSlot;
    return scene.Models(model_slot)[draw.FirstInstance];
}

#endif
