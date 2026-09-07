#ifndef VISIBILITY_COVERAGE_MSL
#define VISIBILITY_COVERAGE_MSL

#include "Bindless.metal"
#include "Varyings.metal"
#include "MaterialAlphaMode.metal"
#include "SceneUBO.metal"
#include "VisibilityDecode.metal"

inline float2 VisibilityUvTransform(float2 uv, float2 offset, float2 scale, float rotation) {
    const float s = sin(rotation);
    const float c = cos(rotation);
    const float2 scaled = uv * scale;
    return float2(c * scaled.x - s * scaled.y, s * scaled.x + c * scaled.y) + offset;
}

inline float4 VisibilitySampleTexture(
    const thread Scene &scene, const thread ResolvedVisibility &resolved,
    const thread VisibilityCoverageValues &coverage, TextureInfo texture,
    VisibilityShadingPushConstants pc
) {
    const VisibilityTextureCoordinates coordinates = DecodeVisibilityTextureCoordinates(
        scene, resolved, coverage, texture.TexCoord, pc
    );
    const float2 transformed = VisibilityUvTransform(
        coordinates.Value, float2(texture.UvOffset), float2(texture.UvScale), texture.UvRotation
    );
    const float s = sin(texture.UvRotation);
    const float c = cos(texture.UvRotation);
    const float2 dx_scaled = coordinates.Dx * float2(texture.UvScale);
    const float2 dy_scaled = coordinates.Dy * float2(texture.UvScale);
    const float2 dx = float2(c * dx_scaled.x - s * dx_scaled.y, s * dx_scaled.x + c * dx_scaled.y);
    const float2 dy = float2(c * dy_scaled.x - s * dy_scaled.y, s * dy_scaled.x + c * dy_scaled.y);
    return scene.SampleTexGrad(texture.Slot, transformed, dx, dy);
}

// Shared raster coverage for visibility and all-depth object picking.
inline uint CoveredMeshletObject(
    const thread Scene &scene, VisibilityShadingPushConstants pc, uint primitive_id,
    float2 pixel, bool front_facing, bool transmission
) {
    device const BindlessSet &bindless = scene.B;
    constant SceneViewUBO &view = scene.View;
    ResolvedVisibility resolved = ResolveVisibilityPrimitive(primitive_id, bindless, pc);
    const uint topology = MeshletPrimitiveTopology(resolved.Meshlet);
    const uint material_index = MeshletPrimitiveMaterialIndex(scene, resolved.Primitive);
    device const PBRMaterial &material = scene.Materials(view.MaterialSlot)[material_index];
    if (topology == MeshPrimitiveTopology_Triangle) {
        const float3 scale = float3(MeshletWorld(scene, resolved.Draw).S);
        const bool authored_front_facing = scale.x * scale.y * scale.z < 0.0f ? !front_facing : front_facing;
        if (material.DoubleSided == 0u && !authored_front_facing) discard_fragment();
    }
    const bool alpha_mask = material.AlphaMode == MaterialAlphaMode_Mask;
    const bool transmission_mask = transmission && material.Unlit == 0u &&
        material.Transmission.Factor > 0.0f;
    const bool point_coverage = topology == MeshPrimitiveTopology_Point;
    if (!alpha_mask && !transmission_mask && !point_coverage) return resolved.Instance.ObjectId;

    if (!MeshletCoarse(resolved.Meshlet)) {
        const uint logical_element = topology == MeshPrimitiveTopology_Triangle ?
            resolved.LocalTriangle : resolved.LocalTriangle / 2u;
        resolved.Triangle = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot)[
            resolved.Meshlet.TriangleOffset + logical_element
        ];
    }
    const VisibilityCoverageValues coverage = DecodeVisibilityCoverage(scene, resolved, pixel, view, pc);
    if (topology != MeshPrimitiveTopology_Triangle && material.DoubleSided == 0u &&
        !IsFrontFacing(scene, coverage.WorldNormal, coverage.WorldPosition)) discard_fragment();
    if (topology == MeshPrimitiveTopology_Point &&
        length(coverage.PointCoord - float2(0.5f)) > 0.5f) discard_fragment();
    if (alpha_mask) {
        float4 base_color = float4(material.BaseColorFactor) * coverage.VertexColor;
        if (material.BaseColorTexture.Slot != INVALID_SLOT) {
            base_color *= VisibilitySampleTexture(scene, resolved, coverage, material.BaseColorTexture, pc);
        }
        if (base_color.a < material.AlphaCutoff) discard_fragment();
    }
    if (transmission_mask) {
        float transmission = material.Transmission.Factor;
        if (material.Transmission.Texture.Slot != INVALID_SLOT) {
            transmission *= VisibilitySampleTexture(scene, resolved, coverage, material.Transmission.Texture, pc).r;
        }
        if (transmission > 0.0f) discard_fragment();
    }
    return resolved.Instance.ObjectId;
}

#endif
