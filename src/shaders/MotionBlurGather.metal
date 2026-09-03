#ifndef MOTIONBLURGATHER_MSL
#define MOTIONBLURGATHER_MSL

#include "Bindless.metal"
#include "Varyings.metal"
#include "MotionBlurShared.metal"
#include "Velocity.metal"
#include "MotionBlurGatherPushConstants.metal"

constant int GatherSampleCount = 8;

// Color and weight split by whether a sample sits in front of or behind the center pixel.
struct Accumulator {
    float4 Fg;
    float4 Bg;
    float3 Weight; // Components represent background, foreground, and direction weights.
};

struct GatherContext {
    Scene S;
    constant MotionBlurGatherPushConstants &Pc;

    // View-space depth, negative in front of the camera, so a smaller value is farther away.
    float LinearDepth(float ndc_depth) const {
        const float n = S.View.CameraNear;
        const float f = S.View.CameraFar;
        return -(2.0f * n * f) / (f + n - (ndc_depth * 2.0f - 1.0f) * (f - n));
    }

    float4 SampleVelocity(float2 uv) const {
        const float4 velocity = UnpackVelocity(S.SampleTex(Pc.VelocitySamplerSlot, uv));
        return velocity * float2(S.TexSize(Pc.VelocitySamplerSlot, 0)).xyxy * float2(Pc.MotionScale).xxyy;
    }

    // Add one pixel to produce a continuous ramp at each streak endpoint.
    float2 SpreadCompare(float center_len, float sample_len, float offset_len) const {
        return clamp(float2(center_len, sample_len) - offset_len + 1.0f, 0.0f, 1.0f);
    }

    float2 DepthCompare(float center_depth, float sample_depth) const {
        const float2 depth_scale = float2(-Pc.DepthScale, Pc.DepthScale);
        return clamp(0.5f + depth_scale * (sample_depth - center_depth), 0.0f, 1.0f);
    }

    // Accept samples moving in the current gather direction and include subpixel motion in both directions.
    float DirCompare(float2 offset, float2 sample_motion, float sample_len) const {
        if (sample_len < 0.5f) return 1.0f;
        return dot(offset, sample_motion) > 0.0f ? 1.0f : 0.0f;
    }

    void GatherSample(float2 screen_uv, float center_depth, float center_len, float2 offset, float offset_len, bool next, thread Accumulator &accum) const {
        const float2 sample_uv = screen_uv - offset / float2(S.TexSize(Pc.ColorSamplerSlot, 0));
        const float4 sample_velocity = SampleVelocity(sample_uv);
        const float2 sample_motion = next ? sample_velocity.zw : sample_velocity.xy;
        const float sample_len = length(sample_motion);
        const float sample_depth = LinearDepth(S.SampleTexLod(Pc.DepthSamplerSlot, sample_uv, 0.0f).r);
        const float4 sample_color = S.SampleTexLod(Pc.ColorSamplerSlot, sample_uv, 0.0f);

        float3 weights;
        weights.xy = DepthCompare(center_depth, sample_depth) * SpreadCompare(center_len, sample_len, offset_len);
        weights.z = DirCompare(offset, sample_motion, sample_len);
        weights.xy *= weights.z;

        accum.Fg += sample_color * weights.y;
        accum.Bg += sample_color * weights.x;
        accum.Weight += weights;
    }

    void GatherBlur(float2 screen_uv, float2 center_motion, float center_depth, float2 max_motion, float ofs, bool next, thread Accumulator &accum) const {
        const float center_len = length(center_motion);
        float max_len = length(max_motion);
        // Include the center tile to prevent jitter from selecting a lower-motion tile.
        if (max_len < center_len) {
            max_len = center_len;
            max_motion = center_motion;
        }
        if (max_len < 0.5f) return;

        const float inc = 1.0f / float(GatherSampleCount);
        float t = ofs * inc;
        for (int i = 0; i < GatherSampleCount; ++i, t += inc) {
            GatherSample(screen_uv, center_depth, center_len, max_motion * t, max_len * t, next, accum);
        }
        if (center_len < 0.5f) return;
        // Sample center motion to preserve detail across foreground-background discontinuities.
        t = ofs * inc;
        for (int i = 0; i < GatherSampleCount; ++i, t += inc) {
            GatherSample(screen_uv, center_depth, center_len, center_motion * t, center_len * t, next, accum);
        }
    }
};

inline float InterleavedGradientNoise(float2 pixel, float seed, float offset) {
    pixel += seed * (float2(47, 17) * 0.695f);
    return fract(offset + 52.9829189f * fract(0.06711056f * pixel.x + 0.00583715f * pixel.y));
}

fragment float4 MotionBlurGatherFragment(
    QuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MotionBlurGatherPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const GatherContext ctx{scene, pc};
    const int2 texel = int2(in.Position.xy);
    const int2 extent = int2(scene.TexSize(pc.ColorSamplerSlot, 0));

    const float2 uv = (float2(texel) + 0.5f) / float2(extent);
    const float center_depth = ctx.LinearDepth(scene.FetchTex(pc.DepthSamplerSlot, texel, 0).r);
    const float4 center_motion = ctx.SampleVelocity(uv);
    float4 center_color = scene.SampleTexLod(pc.ColorSamplerSlot, uv, 0.0f);

    float2 rand = float2(
        InterleavedGradientNoise(float2(texel), 0, pc.NoiseOffset),
        InterleavedGradientNoise(float2(texel), 1, pc.NoiseOffset)
    );

    // Jitter tile lookup by at most one quarter tile to suppress tile-edge banding.
    rand.x = rand.x * 2.0f - 1.0f;
    const int2 tile_extent = int2(bindless.Image[pc.TileImageSlot].get_width(), bindless.Image[pc.TileImageSlot].get_height());
    int2 tile = (texel + int2(int(rand.x * float(MotionBlurTileSize) * 0.25f))) / MotionBlurTileSize;
    tile = clamp(tile, int2(0), tile_extent - 1);

    device const uint *indirections = BindlessBuffer(uint, bindless.Buffer, pc.TileIndirectionSlot);
    const int2 tile_prev = MotionTileUnpack(indirections[MotionTileIndex(MotionPrev, uint2(tile), uint2(tile_extent))]);
    const int2 tile_next = MotionTileUnpack(indirections[MotionTileIndex(MotionNext, uint2(tile), uint2(tile_extent))]);
    const float4 max_motion = float4(
        bindless.Image[pc.TileImageSlot].read(uint2(tile_prev)).xy,
        bindless.Image[pc.TileImageSlot].read(uint2(tile_next)).zw
    );

    Accumulator accum;
    accum.Fg = float4(0.0f);
    accum.Bg = float4(0.0f);
    accum.Weight = float3(0.0f, 0.0f, 1.0f);

    ctx.GatherBlur(uv, center_motion.xy, center_depth, max_motion.xy, rand.y, false, accum);
    ctx.GatherBlur(uv, center_motion.zw, center_depth, max_motion.zw, rand.y, true, accum);

    // A small center weight keeps normalization defined and preserves stationary background detail.
    float w = 1.0f / (50.0f * float(GatherSampleCount) * 4.0f);
    const bool no_motion = length(center_motion.xy) + length(center_motion.zw) < 0.5f;
    if (accum.Weight.x < 1.0f && no_motion) w = 1.0f;
    accum.Bg += center_color * w;
    accum.Weight.x += w;
    // Prefer reconstructed background when foreground samples have insufficient weight.
    center_color = accum.Bg / accum.Weight.x;

    accum.Fg += accum.Bg;
    accum.Weight.y += accum.Weight.x;
    // Fill the rejected-sample weight with background color to preserve brightness.
    const float blend_fac = clamp(1.0f - accum.Weight.y / accum.Weight.z, 0.0f, 1.0f);
    return (accum.Fg / accum.Weight.z) + center_color * blend_fac;
}

#endif
