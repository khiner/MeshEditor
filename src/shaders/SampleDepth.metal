#ifndef SAMPLEDEPTH_MSL
#define SAMPLEDEPTH_MSL

#include "Bindless.metal"
#include "Varyings.metal"

struct SampleDepthPushConstants {
    uint DepthSamplerIndex;
};

struct DepthOnly {
    float Depth [[depth(any)]];
};

fragment DepthOnly SampleDepthFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant SampleDepthPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const int2 texel = int2(in.TexCoord * float2(scene.TexSize(pc.DepthSamplerIndex, 0)));
    return DepthOnly{scene.FetchTex(pc.DepthSamplerIndex, texel, 0).r};
}

#endif
