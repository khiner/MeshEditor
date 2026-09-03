#ifndef BACKGROUND_MSL
#define BACKGROUND_MSL

#include "BackgroundConstant.metal"
#include "Bindless.metal"
#include "Varyings.metal"

constant float2 BackgroundPositions[4] = {float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1)};

vertex NdcVaryings BackgroundVertex(uint vertex_id [[vertex_id]]) {
    const float2 p = BackgroundPositions[vertex_id];
    // z=1 puts the quad on the far plane, so geometry overdraws it through the depth test.
    return NdcVaryings{float4(p, 1.0f, 1.0f), p};
}

fragment float4 BackgroundFragment(
    NdcVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    if (view.WorldOpacity <= 0.0f || view.Ibl.SpecularEnvSamplerSlot == 0xFFFFFFFFu) discard_fragment();

    const float3x3 view_rotation = view.ViewRotation.Unpack();
    const float4x4 view_proj = scene.ViewProj();
    const float3x3 proj3 = float3x3(view_proj[0].xyz, view_proj[1].xyz, view_proj[2].xyz) * transpose(view_rotation);
    const float3x3 inv_rot = transpose(view_rotation);
    const float3 world_dir = normalize(inv_rot * float3(in.Ndc.x / proj3[0][0], in.Ndc.y / proj3[1][1], -1.0f));
    const float3 env_dir = view.EnvRotation.Unpack() * world_dir;
    const uint mip_count = max(view.Ibl.SpecularEnvMipCount, 1u);
    const float lod = clamp(view.BackgroundBlur, 0.0f, 1.0f) * float(mip_count - 1u);
    const float3 linear_color = scene.SampleCubeLod(view.Ibl.SpecularEnvSamplerSlot, env_dir, lod).rgb * view.EnvIntensity;
    return float4(TransmissionPrepass ? linear_color : linear_color * view.Exposure, view.WorldOpacity);
}

#endif
