#ifndef VERTEXPOINT_MSL
#define VERTEXPOINT_MSL

#include "Bindless.metal"
#include "OverlayFade.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"

fragment float4 VertexPointFragment(
    PointVaryings in [[stage_in]],
    float2 point_coord [[point_coord]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    if (length(point_coord - float2(0.5f)) > 0.5f) discard_fragment();
    const Scene scene{bindless, view, theme, workspace};
    return float4(in.Color.rgb, in.Color.a * OverlayBehindFade(scene, in.Position));
}

#endif
