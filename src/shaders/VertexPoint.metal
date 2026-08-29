#ifndef VERTEXPOINT_MSL
#define VERTEXPOINT_MSL

#include "Varyings.metal"

fragment float4 VertexPointFragment(PointVaryings in [[stage_in]], float2 point_coord [[point_coord]]) {
    if (length(point_coord - float2(0.5f)) > 0.5f) discard_fragment();
    return in.Color;
}

#endif
