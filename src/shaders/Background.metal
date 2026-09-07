#ifndef BACKGROUND_MSL
#define BACKGROUND_MSL

#include "BackgroundConstant.metal"
#include "Bindless.metal"
#include "Varyings.metal"
#include "SceneUBO.metal"

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

    const float3 world_dir = normalize(WorldBackgroundDirection(scene, in.Ndc));
    const float3 env_dir = view.EnvRotation.Unpack() * world_dir;
    const uint mip_count = max(view.Ibl.SpecularEnvMipCount, 1u);
    const float lod = clamp(view.BackgroundBlur, 0.0f, 1.0f) * float(mip_count - 1u);
    const float3 linear_color = scene.SampleCubeLod(view.Ibl.SpecularEnvSamplerSlot, env_dir, lod).rgb * view.EnvIntensity;
    return float4(TransmissionPrepass ? linear_color : linear_color * view.Exposure, view.WorldOpacity);
}

#endif
