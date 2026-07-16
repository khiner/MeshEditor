#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

#include "SceneUBO.glsl"
#include "MotionBlurGatherPushConstants.glsl"
#include "MotionBlurShared.glsl"

layout(location = 0) out vec4 OutColor;

layout(set = 0, binding = BINDING_Sampler) uniform sampler2D Samplers[];
layout(set = 0, binding = BINDING_Image, rgba16f) uniform image2D HdrImages[];
layout(set = 0, binding = BINDING_Buffer, scalar) buffer MotionBlurTileIndirection {
    uint Entries[];
} TileIndirections[];

const int GatherSampleCount = 8;

// Color and weight split by whether a sample sits in front of or behind the center pixel.
struct Accumulator {
    vec4 Fg;
    vec4 Bg;
    vec3 Weight; // x background, y foreground, z direction
};

// View-space depth, negative in front of the camera, so a smaller value is farther away.
float LinearDepth(float ndc_depth) {
    const float n = SceneViewUBO.CameraNear;
    const float f = SceneViewUBO.CameraFar;
    return -(2.0 * n * f) / (f + n - (ndc_depth * 2.0 - 1.0) * (f - n));
}

vec4 SampleVelocity(vec2 uv) {
    const vec4 velocity = texture(Samplers[pc.VelocitySamplerSlot], uv);
    return velocity * vec2(textureSize(Samplers[pc.VelocitySamplerSlot], 0)).xyxy * pc.MotionScale.xxyy;
}

// Whether each streak is long enough to reach across `offset_length`. The +1 gives the streak's
// tip a one pixel ramp rather than a hard edge.
vec2 SpreadCompare(float center_len, float sample_len, float offset_len) {
    return clamp(vec2(center_len, sample_len) - offset_len + 1.0, 0.0, 1.0);
}

// Classify the sample as behind (x) or in front of (y) the center.
vec2 DepthCompare(float center_depth, float sample_depth) {
    const vec2 depth_scale = vec2(-pc.DepthScale, pc.DepthScale);
    return clamp(0.5 + depth_scale * (sample_depth - center_depth), 0.0, 1.0);
}

// Keep only samples travelling the way we are gathering. Barely-moving samples always count.
float DirCompare(vec2 offset, vec2 sample_motion, float sample_len) {
    if (sample_len < 0.5) return 1.0;
    return dot(offset, sample_motion) > 0.0 ? 1.0 : 0.0;
}

void GatherSample(vec2 screen_uv, float center_depth, float center_len, vec2 offset, float offset_len, bool next, inout Accumulator accum) {
    const vec2 sample_uv = screen_uv - offset / vec2(textureSize(Samplers[pc.ColorSamplerSlot], 0));
    const vec4 sample_velocity = SampleVelocity(sample_uv);
    const vec2 sample_motion = next ? sample_velocity.zw : sample_velocity.xy;
    const float sample_len = length(sample_motion);
    const float sample_depth = LinearDepth(textureLod(Samplers[pc.DepthSamplerSlot], sample_uv, 0.0).r);
    const vec4 sample_color = textureLod(Samplers[pc.ColorSamplerSlot], sample_uv, 0.0);

    vec3 weights;
    weights.xy = DepthCompare(center_depth, sample_depth) * SpreadCompare(center_len, sample_len, offset_len);
    weights.z = DirCompare(offset, sample_motion, sample_len);
    weights.xy *= weights.z;

    accum.Fg += sample_color * weights.y;
    accum.Bg += sample_color * weights.x;
    accum.Weight += weights;
}

void GatherBlur(vec2 screen_uv, vec2 center_motion, float center_depth, vec2 max_motion, float ofs, bool next, inout Accumulator accum) {
    const float center_len = length(center_motion);
    float max_len = length(max_motion);
    // Jittering the tile lookup can land on a quieter tile than this pixel deserves.
    if (max_len < center_len) {
        max_len = center_len;
        max_motion = center_motion;
    }
    if (max_len < 0.5) return;

    const float inc = 1.0 / float(GatherSampleCount);
    float t = ofs * inc;
    for (int i = 0; i < GatherSampleCount; ++i, t += inc) {
        GatherSample(screen_uv, center_depth, center_len, max_motion * t, max_len * t, next, accum);
    }
    if (center_len < 0.5) return;
    // Walk our own motion too, which recovers detail where foreground and background disagree.
    t = ofs * inc;
    for (int i = 0; i < GatherSampleCount; ++i, t += inc) {
        GatherSample(screen_uv, center_depth, center_len, center_motion * t, center_len * t, next, accum);
    }
}

float InterleavedGradientNoise(vec2 pixel, float seed, float offset) {
    pixel += seed * (vec2(47, 17) * 0.695);
    return fract(offset + 52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

void main() {
    const ivec2 texel = ivec2(gl_FragCoord.xy);
    const ivec2 extent = textureSize(Samplers[pc.ColorSamplerSlot], 0);

    const vec2 uv = (vec2(texel) + 0.5) / vec2(extent);
    const float center_depth = LinearDepth(texelFetch(Samplers[pc.DepthSamplerSlot], texel, 0).r);
    const vec4 center_motion = SampleVelocity(uv);
    vec4 center_color = textureLod(Samplers[pc.ColorSamplerSlot], uv, 0.0);

    vec2 rand = vec2(
        InterleavedGradientNoise(vec2(texel), 0, pc.NoiseOffset),
        InterleavedGradientNoise(vec2(texel), 1, pc.NoiseOffset)
    );

    // Jitter the tile lookup by up to a quarter tile so tile edges do not show as banding.
    rand.x = rand.x * 2.0 - 1.0;
    const ivec2 tile_extent = imageSize(HdrImages[pc.TileImageSlot]);
    ivec2 tile = (texel + ivec2(int(rand.x * float(MotionBlurTileSize) * 0.25))) / MotionBlurTileSize;
    tile = clamp(tile, ivec2(0), tile_extent - 1);

    // Tile motion is already in pixels with both halves pointing forward.
    const ivec2 tile_prev = MotionTileUnpack(TileIndirections[pc.TileIndirectionSlot].Entries[MotionTileIndex(MotionPrev, uvec2(tile), uvec2(tile_extent))]);
    const ivec2 tile_next = MotionTileUnpack(TileIndirections[pc.TileIndirectionSlot].Entries[MotionTileIndex(MotionNext, uvec2(tile), uvec2(tile_extent))]);
    const vec4 max_motion = vec4(
        imageLoad(HdrImages[pc.TileImageSlot], tile_prev).xy,
        imageLoad(HdrImages[pc.TileImageSlot], tile_next).zw
    );

    Accumulator accum;
    accum.Fg = vec4(0.0);
    accum.Bg = vec4(0.0);
    accum.Weight = vec3(0.0, 0.0, 1.0); // The direction weight starts at one, which normalizes below.

    GatherBlur(uv, center_motion.xy, center_depth, max_motion.xy, rand.y, false, accum); // [T - delta, T]
    GatherBlur(uv, center_motion.zw, center_depth, max_motion.zw, rand.y, true, accum); // [T, T + delta]

    // A sliver of center weight keeps the division defined. A still pixel surrounded by fast
    // motion takes its full center color back, which keeps the background crisp.
    float w = 1.0 / (50.0 * float(GatherSampleCount) * 4.0);
    const bool no_motion = length(center_motion.xy) + length(center_motion.zw) < 0.5;
    if (accum.Weight.x < 1.0 && no_motion) w = 1.0;
    accum.Bg += center_color * w;
    accum.Weight.x += w;
    // The reconstructed background carries more information than the center sample for foreground
    // pixels that gathered too little weight.
    center_color = accum.Bg / accum.Weight.x;

    accum.Fg += accum.Bg;
    accum.Weight.y += accum.Weight.x;
    // Samples that passed the direction test but failed depth or spread leave a weight deficit.
    // Fill it with the background rather than darkening the pixel.
    const float blend_fac = clamp(1.0 - accum.Weight.y / accum.Weight.z, 0.0, 1.0);
    OutColor = (accum.Fg / accum.Weight.z) + center_color * blend_fac;
}
