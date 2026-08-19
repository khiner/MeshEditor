#ifndef BACKGROUNDVELOCITY_MSL
#define BACKGROUNDVELOCITY_MSL

#include "Bindless.metal"
#include "Varyings.metal"
#include "Velocity.metal"

// The background sits at infinity, so only view rotation moves it. Projecting the view ray as a
// direction lands on the point at infinity along it, which is what the background shows.
inline float2 ProjectDirToNdc(float4x4 view_proj, float3 dir) {
    const float4 clip = view_proj * float4(dir, 0.0f);
    return clip.xy / clip.w;
}

// The scene pass's velocity attachment. The scene color at index 0 is masked off.
struct VelocityTarget {
    float4 Motion [[color(1)]];
};

fragment VelocityTarget BackgroundVelocityFragment(
    NdcVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const float3x3 view_rotation = view.ViewRotation.Unpack();
    const float4x4 view_proj = scene.ViewProj();
    const float3x3 proj3 = float3x3(view_proj[0].xyz, view_proj[1].xyz, view_proj[2].xyz) * transpose(view_rotation);
    const float3x3 inv_rot = transpose(view_rotation);
    const float3 world_dir = normalize(inv_rot * float3(in.Ndc.x / proj3[0][0], in.Ndc.y / proj3[1][1], -1.0f));

    const float2 curr_ndc = ProjectDirToNdc(view_proj, world_dir);
    const float2 prev_ndc = ProjectDirToNdc(scene.PrevViewProj(), world_dir);
    const float2 next_ndc = ProjectDirToNdc(scene.NextViewProj(), world_dir);
    return VelocityTarget{PackScreenMotion(prev_ndc, curr_ndc, next_ndc)};
}

#endif
