#ifndef VERTEXCOLOR_MSL
#define VERTEXCOLOR_MSL

#include "Varyings.metal"

fragment OverlayTargets VertexColorFragment(LineVaryings in [[stage_in]]) {
    return OverlayTargets{in.Color, pack_line_data(in.Position.xy, in.EdgeStart, in.EdgePos)};
}

#endif
