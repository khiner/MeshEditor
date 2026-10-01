#pragma once
#include "gpu/ConnectivityRef.h"
#include "gpu/ElementAttributeRef.h"

#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Stores one mesh's arena locations once, shared by every primitive and instance that draws it.
// Corner offsets locate the mesh's first corner, and ComposeDraw advances them to a primitive's.
struct MeshRecord {
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    SlotOffset IndexSlotOffset DEFAULT();
    uint32_t ModelSlot DEFAULT(InvalidSlot);
    uint32_t TriangleSlot DEFAULT(InvalidSlot);
    uint32_t CornerClassMode DEFAULT(InvalidOffset);
    ElementAttributeRef CustomNormals DEFAULT();
    ElementAttributeRef CornerTangent DEFAULT();
    ElementAttributeRef CornerColor DEFAULT();
    GpuArray<ElementAttributeRef, 4> CornerUvs DEFAULT();
    uint32_t TriangleOffset DEFAULT();
    ConnectivityRef Connectivity DEFAULT();
    uint32_t HalfedgeCount DEFAULT();
    uint32_t FaceCount DEFAULT();
    uint32_t VertexCountOrHeadImageSlot DEFAULT();
    uint32_t InstanceStateSlot DEFAULT(InvalidSlot);
    uint32_t VertexOffset DEFAULT();
    uint32_t MorphShadingAuthored DEFAULT();
    uint32_t PrimitiveMaterialOffset DEFAULT(InvalidOffset);
    ElementAttributeRef ElementPrimitives DEFAULT();
};
static_assert(sizeof(MeshRecord) == 180, "MeshRecord size");
