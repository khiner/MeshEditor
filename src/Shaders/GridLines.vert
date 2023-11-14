#version 450

layout(binding = 0) uniform ViewProjectionUBO {
    mat4 View;
    mat4 Projection;
} ViewProjection;

layout(location = 0) out vec3 NearPos;
layout(location = 1) out vec3 FarPos;

// Triangle strip.
const vec2 QuadPositions[4] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1));

vec3 Unproject(vec3 screen_pos, mat4 inv_view_proj) {
    vec4 unprojected_pos = inv_view_proj * vec4(screen_pos, 1);
    return unprojected_pos.xyz / unprojected_pos.w;
}

void main() {
    vec2 p = QuadPositions[gl_VertexIndex];
    mat4 inv_view_proj = inverse(ViewProjection.Projection * ViewProjection.View);
    NearPos = Unproject(vec3(p, 0), inv_view_proj).xyz;
    FarPos = Unproject(vec3(p, 1), inv_view_proj).xyz;
    gl_Position = vec4(p, 0, 1);
}
