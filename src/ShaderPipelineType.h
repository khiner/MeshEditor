#pragma once

enum class ShaderPipelineType {
    Fill,
    Line,
    LineOverlayFaceNormals,
    LineOverlayVertexNormals,
    LineOverlayBBox,
    Point,
    Grid,
    Background,
    SilhouetteDepthObject,
    SilhouetteEdgeDepthObject,
    SilhouetteEdgeDepth,
    SilhouetteEdgeColor,
    SelectionElementFace,
    SelectionElementEdge,
    SelectionElementVertex,
    SelectionElementFaceBitsetBox,
    SelectionElementEdgeBitsetBox,
    SelectionElementVertexBitsetBox,
    SelectionElementFaceXRay,
    SelectionElementEdgeXRay,
    SelectionElementVertexXRay,
    SelectionElementFaceXRayBitsetBox,
    SelectionElementEdgeXRayBitsetBox,
    SelectionElementVertexXRayBitsetBox,
    SelectionElementEdgeXRayVerts, // ePointList pass paired with XRay edge lines to catch near/zero-length projected edges
    SelectionElementFaceXRayVerts, // ePointList pass paired with XRay triangle pass to handle edge-on faces
    SelectionElementEdgeXRayVertsBitsetBox, // Bitset-box variant of the XRay edge point fallback
    SelectionElementFaceXRayVertsBitsetBox, // Bitset-box variant of the XRay face point fallback
    SelectionFragmentTriangles,
    SelectionFragmentLines,
    SelectionFragmentPoints,
    SelectionFragmentBoneSphere,
    BoneFill,
    BoneWire,
    BoneSphereFill,
    BoneSphereWire,
    ObjectExtrasLine,
    SelectionObjectExtrasLines,
    DebugNormals,
};
