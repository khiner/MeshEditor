#pragma once
// The preview sphere's surface at a pixel of its triangle id image, decoded as the visibility resolve decodes a mesh's surface.
#include "VisibilityDecode.metal"
#include "gpu/MaterialPreview.h"

// The world corners of a triangle of the preview sphere and the pixel's weights over them.
struct PreviewTriangle {
    PreviewCorner Corners[3];
    PerspectiveWeights Weights;
};
inline PreviewTriangle ResolvePreviewTriangle(float2 pixel, uint triangle, device const PreviewCorner *corners, constant SceneViewUBO &view) {
    PreviewTriangle resolved;
    float4 clip[3];
    for (uint c = 0u; c < 3u; ++c) {
        resolved.Corners[c] = corners[3u * triangle + c];
        clip[c] = view.ViewProj.Unpack() * float4(float3(resolved.Corners[c].Position), 1.0f);
    }
    resolved.Weights = TriangleWeights(pixel, clip[0], clip[1], clip[2], float2(view.ViewportSize));
    return resolved;
}

// The surface of the material record at a pixel of a preview triangle, whose first texture coordinate set holds the sphere's coordinates.
inline DecodedSurface DecodePreviewSurface(float2 pixel, uint triangle, device const PreviewCorner *corners, constant SceneViewUBO &view, uint material) {
    const PreviewTriangle resolved = ResolvePreviewTriangle(pixel, triangle, corners, view);
    const thread PreviewCorner (&c)[3] = resolved.Corners;
    const PerspectiveWeights weights = resolved.Weights;
    const float3 p0 = float3(c[0].Position), p1 = float3(c[1].Position), p2 = float3(c[2].Position);
    DecodedSurface result{};
    result.GeometricNormal = normalize(cross(p1 - p0, p2 - p0));
    result.V.Position = float4(pixel, 0.0f, 1.0f);
    result.V.WorldNormal = PerspectiveValue(weights.Value, float3(c[0].Normal), float3(c[1].Normal), float3(c[2].Normal));
    result.V.WorldPosition = PerspectiveValue(weights.Value, p0, p1, p2);
    result.PositionDx = PerspectiveValue(weights.Dx, p0, p1, p2);
    result.PositionDy = PerspectiveValue(weights.Dy, p0, p1, p2);
    result.V.VertexColor = float4(1.0f);
    result.V.WorldTangent = PerspectiveValue(weights.Value, float4(float3(c[0].Tangent), c[0].TangentSign), float4(float3(c[1].Tangent), c[1].TangentSign), float4(float3(c[2].Tangent), c[2].TangentSign));
    DecodeUv(result.V.TexCoord0, result.UvDx[0], result.UvDy[0], weights, float2(c[0].Uv), float2(c[1].Uv), float2(c[2].Uv));
    result.V.FlatWorldNormal = result.GeometricNormal;
    result.V.MaterialIndex = material;
    result.V.WorldScale = 1.0f;
    result.InstanceSlot = InvalidOffset;
    result.Topology = uint(MeshPrimitiveTopology::Triangle);
    result.Valid = true;
    return result;
}
