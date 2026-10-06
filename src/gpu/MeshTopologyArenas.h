#pragma once
#include "gpu/ElementAttributeRef.h"
#include "gpu/Types.h"

// Geometry bindings for one version of canonical topology data.
struct MeshTopologyArenas {
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    uint32_t CornerSlot DEFAULT(InvalidSlot);
    uint32_t FaceTriangleStartSlot DEFAULT(InvalidSlot);
    uint32_t TriangleSlot DEFAULT(InvalidSlot);
    uint32_t EdgeSharpnessSlot DEFAULT(InvalidSlot);
    uint32_t FaceSharpnessSlot DEFAULT(InvalidSlot);
    ElementAttributeRef FacePrimitives DEFAULT(), VertexPrimitives DEFAULT();
    ElementAttributeRef Skin DEFAULT(), Morph DEFAULT();
    ElementAttributeRef CornerTangent DEFAULT();
    ElementAttributeRef CornerColor DEFAULT();
    ElementAttributeRef VertexColor DEFAULT();
    GpuArray<ElementAttributeRef, 4> CornerUvs DEFAULT();
    ElementAttributeRef CustomNormals DEFAULT();
    ElementAttributeRef CornerSectors DEFAULT();
    ElementAttributeRef NormalSectors DEFAULT();
    uint32_t BaseVertexNormalSlot DEFAULT(InvalidSlot);
    uint32_t BaseFaceNormalSlot DEFAULT(InvalidSlot);
    uint32_t VertexHiddenSlot DEFAULT(InvalidSlot), EdgeHiddenSlot DEFAULT(InvalidSlot), FaceHiddenSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(MeshTopologyArenas) == 156, "MeshTopologyArenas size");
