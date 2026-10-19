#pragma once

#include "gpu/Types.h"

// Mesh records contain every flag except Silhouette, which an instance takes from its selection state.
// A record takes a display flag when any visible viewport shows it, and each viewport's passes select theirs by the flags they require.
// ElementSelection and EditOverlay apply only to the mesh's primary edit instance.
enum class MeshletInstanceFlag : uint32_t {
    // A selected instance of a SilhouetteEligible mesh, apart from its mesh's primary edit instance.
    Silhouette = 1u,
    ElementSelection = 2u,
    // Restrict edited and deformed instances to original source geometry in every view.
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
    // A face-less Wire mesh, whose wire Solid shading draws.
    SolidWire = 8192u,
    // A face-less Wire mesh without materials, whose wire material shading draws.
    MaterialWire = 16384u,
    // A selected mesh showing vertex normals, which a view showing them draws from original source geometry.
    VertexNormal = 32768u,
};
