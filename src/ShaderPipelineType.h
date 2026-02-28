#pragma once

enum class ShaderPipelineType {
    Fill,
    PBRFill,
    PBRFillBlend,
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
    SelectionElementFaceXRay,
    SelectionElementEdgeXRay,
    SelectionElementVertexXRay,
    SelectionElementEdgeXRayVerts, // ePointList pass paired with XRay edge lines to catch near/zero-length projected edges
    SelectionElementFaceXRayVerts, // ePointList pass paired with XRay triangle pass to handle edge-on faces
    SelectionFragmentTriangles,
    SelectionFragmentLines,
    SelectionFragmentPoints,
    ObjectExtrasLine,
    SelectionObjectExtrasLines,
    DebugNormals,
};
