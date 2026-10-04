#pragma once

#include "gpu/EditSelectionStorage.h"
#include "gpu/Types.h"

// The state of one mesh's draws that the scene's display, selection and pose set, apart from its arena locations.
// Every instance of the mesh draws with it.
// The cull reads the leading fields, which share one cache line.
struct MeshDisplay {
    uint32_t Flags DEFAULT(); // MeshletInstanceFlag bits, without the per-instance Silhouette bit.
    uint32_t PrimitiveRoot DEFAULT(InvalidOffset);
    uint32_t PrimitiveCount DEFAULT();
    // The slot of the instance that draws the mesh's element selection, in Edit mode only.
    // Without one, every instance draws the selection the record holds.
    uint32_t PrimaryEditInstanceIndex DEFAULT(InvalidOffset);
    uint32_t HasPendingVertexTransform DEFAULT();
    uint32_t MorphDeformOffset DEFAULT(InvalidOffset);
    // Pose namespaces shared by every instance of a mesh posed without per-instance deformation.
    uint32_t PositionNamespace DEFAULT(InvalidOffset);
    uint32_t MeshletBoundsNamespace DEFAULT(InvalidOffset);
    uint32_t MorphNormalNamespace DEFAULT(InvalidOffset);
    uint32_t VertexNormalNamespace DEFAULT(InvalidOffset);
    uint32_t SectorNamespace DEFAULT(InvalidOffset);
    uint32_t FaceNormalNamespace DEFAULT(InvalidOffset);
    uint32_t BoneDeformOffset DEFAULT(InvalidOffset);
    uint32_t MorphTargetCount DEFAULT();
    uint32_t ElementIdOffset DEFAULT();
    uint32_t EditEdgeSharpnessOffset DEFAULT(InvalidOffset);
    EditSelectionStorage Selection DEFAULT();
    uint32_t ActiveVertex DEFAULT(InvalidOffset); // The excited mesh's active vertex, in Excite mode.
};
static_assert(sizeof(MeshDisplay) == 100, "MeshDisplay size");
