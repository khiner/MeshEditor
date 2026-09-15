#ifndef VISIBILITY_COVERAGE_MSL
#define VISIBILITY_COVERAGE_MSL

#include "Bindless.metal"
#include "Varyings.metal"
#include "gpu/MaterialAlphaMode.h"
#include "SceneUBO.metal"
#include "VisibilityDecode.metal"

inline float4 VisibilitySampleTexture(
    const thread Scene &scene, const thread ResolvedVisibility &resolved,
    const thread VisibilityCoverageValues &coverage, TextureInfo texture,
    VisibilityShadingPushConstants pc
) {
    const VisibilityTextureCoordinates coordinates = DecodeVisibilityTextureCoordinates(
        scene, resolved, coverage, texture.TexCoord, pc
    );
    const float2 scale = float2(texture.UvScale);
    return scene.SampleTexGrad(
        texture.Slot, ApplyUvTransform(coordinates.Value, float2(texture.UvOffset), scale, texture.UvRotation),
        TransformUvGradient(coordinates.Dx, scale, texture.UvRotation), TransformUvGradient(coordinates.Dy, scale, texture.UvRotation)
    );
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
    if (topology == uint(MeshPrimitiveTopology::Triangle)) {
        const float3 scale = float3(MeshletWorld(scene, resolved.Draw).S);
        const bool authored_front_facing = scale.x * scale.y * scale.z < 0.0f ? !front_facing : front_facing;
        if (material.DoubleSided == 0u && !authored_front_facing) discard_fragment();
    }
    const bool alpha_mask = material.AlphaMode == MaterialAlphaMode::Mask;
    const bool transmission_mask = transmission && material.Unlit == 0u &&
        material.Transmission.Factor > 0.0f;
    const bool point_coverage = topology == uint(MeshPrimitiveTopology::Point);
    if (!alpha_mask && !transmission_mask && !point_coverage) return resolved.Instance.ObjectId;

    if (!MeshletCoarse(resolved.Meshlet)) {
        const uint logical_element = topology == uint(MeshPrimitiveTopology::Triangle) ?
            resolved.LocalTriangle : resolved.LocalTriangle / 2u;
        resolved.Triangle = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot)[
            resolved.Meshlet.TriangleOffset + logical_element
        ];
    }
    const VisibilityCoverageValues coverage = DecodeVisibilityCoverage(scene, resolved, pixel, view, pc);
    if (topology != uint(MeshPrimitiveTopology::Triangle) && material.DoubleSided == 0u &&
        !IsFrontFacing(scene, coverage.WorldNormal, coverage.WorldPosition)) discard_fragment();
    if (topology == uint(MeshPrimitiveTopology::Point) &&
        length(coverage.PointCoord - float2(0.5f)) > 0.5f) discard_fragment();
    if (alpha_mask) {
        float4 base_color = float4(material.BaseColorFactor) * coverage.VertexColor;
        if (material.BaseColorTexture.Slot != InvalidSlot) {
            base_color *= VisibilitySampleTexture(scene, resolved, coverage, material.BaseColorTexture, pc);
        }
        if (base_color.a < material.AlphaCutoff) discard_fragment();
    }
    if (transmission_mask) {
        float transmission = material.Transmission.Factor;
        if (material.Transmission.Texture.Slot != InvalidSlot) {
            transmission *= VisibilitySampleTexture(scene, resolved, coverage, material.Transmission.Texture, pc).r;
        }
        if (transmission > 0.0f) discard_fragment();
    }
    return resolved.Instance.ObjectId;
}

#endif
