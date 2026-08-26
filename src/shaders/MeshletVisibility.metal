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

fragment uint MeshletVisibilityOpaqueFragment(uint primitive_id [[primitive_id]]) {
    return primitive_id;
}

fragment uint MeshletVisibilityPrimitiveFragment(
    float4 position [[position]],
    uint primitive_id [[primitive_id]],
    bool front_facing [[front_facing]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletDrawPushConstants &draw_pc [[buffer(BufferIndex_PushConstants)]]
) {
    // Raster-time coverage decodes the ID against the list this mesh draw is emitting. Alias both
    // phase slots so the encoded phase bit deliberately has no effect in this fragment.
    const VisibilityShadingPushConstants pc{
        .PrimitiveSlot = draw_pc.PrimitiveSlot,
        .InstanceSlot = draw_pc.InstanceSlot,
        .InstanceMapSlot = draw_pc.InstanceMapSlot,
        .MeshletSlot = draw_pc.MeshletSlot,
        .MeshletTriangleSlot = draw_pc.MeshletTriangleSlot,
        .MeshletLocalTriangleSlot = draw_pc.MeshletLocalTriangleSlot,
        .MeshletVertexSlot = draw_pc.MeshletVertexSlot,
        .VisibleMeshletSlot = draw_pc.VisibleMeshletSlot,
        .Phase2VisibleMeshletSlot = draw_pc.VisibleMeshletSlot,
    };
    const Scene scene{bindless, view, theme, workspace};
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
    const bool transmission_mask = draw_pc.VisibilityTransmission != 0u && material.Unlit == 0u &&
        material.Transmission.Factor > 0.0f;
    const bool point_coverage = topology == MeshPrimitiveTopology_Point;
    if (!alpha_mask && !transmission_mask && !point_coverage) return primitive_id;

    if (!MeshletCoarse(resolved.Meshlet)) {
        const uint logical_element = topology == MeshPrimitiveTopology_Triangle ?
            resolved.LocalTriangle : resolved.LocalTriangle / 2u;
        resolved.Triangle = BindlessBuffer(uint, bindless.Buffer, pc.MeshletTriangleSlot)[
            resolved.Meshlet.TriangleOffset + logical_element
        ];
    }
    const VisibilityCoverageValues coverage = DecodeVisibilityCoverage(scene, resolved, position.xy, view, pc);
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
    return primitive_id;
}
