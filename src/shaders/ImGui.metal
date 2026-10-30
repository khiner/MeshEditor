#include <metal_stdlib>
using namespace metal;

// Matches ImDrawVert: a packed position, texture coordinate and 8-bit color.
struct ImGuiDrawVertex {
    packed_float2 Position;
    packed_float2 Uv;
    uchar4 Color;
};

struct ImGuiVaryings {
    float4 Position [[position]];
    float2 Uv;
    float4 Color;
};

vertex ImGuiVaryings ImGuiVertex(uint id [[vertex_id]], device const ImGuiDrawVertex *vertices [[buffer(0)]], constant float4x4 &projection [[buffer(1)]]) {
    const ImGuiDrawVertex v = vertices[id];
    return {projection * float4(float2(v.Position), 0, 1), float2(v.Uv), float4(v.Color) / 255.0};
}

fragment half4 ImGuiFragment(ImGuiVaryings in [[stage_in]], texture2d<half> image [[texture(0)]], sampler image_sampler [[sampler(0)]]) {
    return half4(in.Color) * image.sample(image_sampler, in.Uv);
}
