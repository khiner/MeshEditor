#pragma once

#include "gpu/Types.h"

enum class MeshletInstanceFlag : uint32_t {
    Silhouette = 1u,
    ElementSelection = 2u,
    // Restrict edited and deformed instances to original source geometry.
    LodPinFinest = 4u,
    // Emit source edges and points from original meshlets for the primary edit instance.
    EditOverlay = 8u,
    // Provide canonical source edges from original meshlets for wire rasterization.
    Wire = 16u,
    // Route procedural and editor-only geometry outside material visibility.
    OverlayOnly = 32u,
    Bone = 64u,
    BoneWire = 128u,
    BoneJoint = 256u,
    BoneJointWire = 512u,
    FaceNormal = 1024u,
    VertexNormal = 2048u,
    EdgeOverlay = 4096u,
    PointOverlay = 8192u,
    // Use the persistent vertex-selection mask for excite-mode points.
    SoundPoint = 16384u,
};
