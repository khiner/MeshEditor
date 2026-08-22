#ifndef MESHLET_SHARED_MSL
#define MESHLET_SHARED_MSL

// Decode helpers for the meshlet-local geometry streams.
#include "Bindless.metal"
#include "MeshletGeometryEncoding.metal"
#include "MeshletRecord.metal"
#include "MeshPrimitiveTopology.metal"

// The render vertex's representative corner packed with its flat-only bit.
inline uint MeshletPackedVertex(device const BindlessSet &bindless, uint vertex_slot, MeshletRecord meshlet, uint i) {
    return BindlessBuffer(uint, bindless.Buffer, vertex_slot)[meshlet.VertexOffset + i];
}

inline uint MeshletLocalTriangleOffset(MeshletRecord meshlet) {
    return meshlet.LocalTriangleOffset & MeshletGeometryEncoding_LocalTriangleOffsetMask;
}

// The canonical vertex id behind a packed render vertex.
inline uint MeshletVertexId(
    const thread Scene &scene, DrawData draw, uint topology, uint packed_vertex
) {
    if (topology != MeshPrimitiveTopology_Triangle) return packed_vertex;
    return scene.Indices(draw.IndexSlotOffset.Slot)[
        draw.IndexSlotOffset.Offset + (packed_vertex & MeshletGeometryEncoding_CornerMask)
    ];
}

// The world transform every meshlet stage reads, honoring the blur steps' model override.
inline Transform MeshletWorld(const thread Scene &scene, DrawData draw) {
    const uint model_slot = scene.View.ModelSlotOverride != INVALID_SLOT ? scene.View.ModelSlotOverride : draw.ModelSlot;
    return scene.Models(model_slot)[draw.FirstInstance];
}

#endif
