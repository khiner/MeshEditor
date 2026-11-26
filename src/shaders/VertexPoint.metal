#pragma once
#include "OverlayFade.metal"
#include "Varyings.metal"

fragment float4 VertexPointFragment(
    PointVaryings in [[stage_in]],
    float2 point_coord [[point_coord]],
    SCENE_ARGS
) {
    if (length(point_coord - float2(0.5f)) > 0.5f) discard_fragment();
    const Scene scene{bindless, view, theme, workspace};
    return float4(in.Color.rgb, in.Color.a * OverlayBehindFade(scene, in.Position));
}
