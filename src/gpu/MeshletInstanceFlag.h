#pragma once

#include "gpu/Types.h"

// Mesh records hold every flag except Silhouette, which an instance takes from its selection state.
// ElementSelection and EditOverlay apply to the mesh's primary edit instance only.
enum class MeshletInstanceFlag : uint32_t {
    // A selected instance of a SilhouetteEligible mesh, apart from its mesh's primary edit instance.
    Silhouette = 1u,
    ElementSelection = 2u,
    // Restrict edited and deformed instances and those showing vertex normals to original source geometry.
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
    EdgeOverlay = 2048u,
    // A face mesh, whose selected instances outline through the screen-space silhouette.
    SilhouetteEligible = 4096u,
};
