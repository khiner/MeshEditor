#ifndef SILHOUETTEEDGECOLOR_MSL
#define SILHOUETTEEDGECOLOR_MSL

#include "Bindless.metal"
#include "Varyings.metal"
#include "SilhouetteEdgeColorPushConstants.metal"

fragment float4 SilhouetteEdgeColorFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant SilhouetteEdgeColorPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const int2 texel = int2(in.TexCoord * float2(scene.TexSize(pc.ObjectSamplerIndex, 0)));
    const uint object_id = uint(scene.FetchTex(pc.ObjectSamplerIndex, texel, 0).r);
    if (object_id == 0u) discard_fragment();

    // UINT32_MAX marks every armature bone instance active because an armature has no single object ID.
    const bool is_active = pc.ActiveObjectId == 0xFFFFFFFFu || (pc.ActiveObjectId != 0u && object_id == pc.ActiveObjectId);
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    return pc.Manipulating != 0u ? float4(float3(colors.Transform), 1.0f) :
        is_active ? float4(float3(colors.ObjectActive), 1.0f) :
                    float4(float3(colors.ObjectSelected), 1.0f);
}

#endif
