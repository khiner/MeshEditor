#version 450

// Assume the input vertices are a clip-space quad.

layout(location = 0) in vec3 InPosition;

layout(location = 0) out vec3 NearPos;
layout(location = 1) out vec3 FarPos;

layout(binding = 0) uniform ViewProjectionUBO {
    mat4 View;
    mat4 Projection;
} ViewProjection;

vec3 Unproject(vec3 screen_pos, mat4 inv_view_proj) {
    vec4 unprojected_pos = inv_view_proj * vec4(screen_pos, 1);
    return unprojected_pos.xyz / unprojected_pos.w;
}

void main() {
    vec3 p = InPosition.xyz;
    mat4 inv_view_proj = inverse(ViewProjection.Projection * ViewProjection.View);
    NearPos = Unproject(vec3(p.xy, 0), inv_view_proj).xyz;
    FarPos = Unproject(vec3(p.xy, 1), inv_view_proj).xyz;
    gl_Position = vec4(InPosition, 1);
}
