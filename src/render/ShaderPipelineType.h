#pragma once

enum class ShaderPipelineType {
    Grid,
    Background,
    BackgroundVelocity,
    TransmissionComposite,
    MotionBlurResolve,
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
    SelectionElementEdgeXRayVertsBitsetBox, // Bitset-box variant of the XRay edge point fallback
    SelectionElementFaceXRayVertsBitsetBox, // Bitset-box variant of the XRay face point fallback
};
