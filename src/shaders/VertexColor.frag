#version 450

layout(location = 2) in vec4 InColor;
layout(location = 12) flat in vec2 EdgeStart; // Screen-space position of the first vertex (provoking vertex, flat)
layout(location = 13) in vec2 EdgePos;        // Screen-space position interpolated along the line (smooth)

layout(location = 0) out vec4 OutColor;
layout(location = 1) out vec4 OutLineData; // Packed line AA data: xy=perp direction, z=signed dist, w=1 (has line)

// Pack line edge data for the AA composite pass.
// Encodes the perpendicular direction and signed distance-to-line into a [0,1] float4.
// frag_co: fragment screen-space coordinate (pixels)
// edge_start: screen-space position of the first vertex of the line segment (flat)
// edge_pos: interpolated screen-space position along the line (smooth)
vec4 pack_line_data(vec2 frag_co, vec2 edge_start, vec2 edge_pos) {
    vec2 edge = edge_start - edge_pos;
    float len = length(edge);
    if (len > 0.0) {
        edge /= len;
        vec2 perp = vec2(-edge.y, edge.x);
        float dist = dot(perp, frag_co - edge_start);
        return vec4(perp * 0.5 + 0.5, dist * 0.25 + 0.6, 1.0);
    }
    // Zero-length edge: use a stable default (perp points right, dist=0)
    return vec4(1.0, 0.0, 0.6, 1.0);
}

void main() {
    OutColor = InColor;
    OutLineData = pack_line_data(gl_FragCoord.xy, EdgeStart, EdgePos);
}
