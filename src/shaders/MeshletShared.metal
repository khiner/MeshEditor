#ifndef MESHLET_SHARED_MSL
#define MESHLET_SHARED_MSL

#include "CompactPresent.metal"

#include "Bindless.metal"
#include "gpu/MeshletGeometryEncoding.h"
#include "gpu/MeshletRecord.h"
#include "gpu/MeshPrimitiveTopology.h"
#include "gpu/PrimitiveRecord.h"

inline uint MeshletSourceVertex(device const BindlessSet &bindless, uint vertex_slot, MeshletRecord meshlet, uint i) {
    return BindlessBuffer(uint, bindless.Buffer, vertex_slot)[meshlet.VertexOffset + i];
}

inline uint MeshletLocalTriangleOffset(MeshletRecord meshlet) {
    return meshlet.LocalTriangleOffset;
}

inline uint MeshletPrimitiveTopology(MeshletRecord meshlet) {
    return meshlet.Topology;
}

inline uint MeshletPrimitiveMaterialIndex(const thread Scene &scene, PrimitiveRecord primitive) {
    if (primitive.PrimitiveMaterialOffset == InvalidOffset) return 0u;
    return scene.PrimitiveMaterials(scene.View.PrimitiveMaterialSlot)[primitive.PrimitiveMaterialOffset + primitive.PrimitiveIndex];
}


inline uint MeshletVertexId(
    const thread Scene &scene, DrawData draw, uint topology, uint source_vertex
) {
    if (topology != uint(MeshPrimitiveTopology::Triangle)) return source_vertex;
    return scene.CornerVertexOrdinal(draw, source_vertex);
}

// Returns the motion-blur model override or the draw's current world transform.
inline Transform MeshletWorld(const thread Scene &scene, DrawData draw) {
    const uint model_slot = scene.View.ModelSlotOverride != InvalidSlot ? scene.View.ModelSlotOverride : draw.ModelSlot;
    return scene.Models(model_slot)[draw.FirstInstance];
}

#endif
