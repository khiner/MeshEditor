#ifndef MOTION_BLUR_SHARED_MSL
#define MOTION_BLUR_SHARED_MSL

// Shared by the motion blur tile passes and the gather.
// Ported from Blender EEVEE (eevee_motion_blur.bsl.hh), which follows
// "A Fast and Stable Feature-Aware Motion Blur Filter" by Guertin, McGuire, and Nowrouzezahrai.

#include <metal_stdlib>
using namespace metal;

constant int MotionBlurTileSize = 32;

constant uint MotionPrev = 0u;
constant uint MotionNext = 1u;

// Each entry holds the coordinates of the tile whose motion covers this one, with the motion's
// length in the high bits so an atomic max picks the fastest contributor.
//   bits 31..18: length in pixels, clamped to 16383
//   bits 17..9 : tile x
//   bits 8..0  : tile y
// Nine bits reach 512 tiles a side, which at 32 pixels per tile covers renders up to 16384 across.
inline uint MotionTilePack(float2 motion, uint2 tile) {
    const uint velocity = min(uint(ceil(length(motion))), 0x3FFFu);
    return (velocity << 18u) | ((tile.x & 0x1FFu) << 9u) | (tile.y & 0x1FFu);
}
inline int2 MotionTileUnpack(uint data) {
    return int2(int((data >> 9u) & 0x1FFu), int(data & 0x1FFu));
}
// The table holds one entry per tile, per motion direction, sized to the render's own tile grid.
inline uint MotionTileIndex(uint motion_step, uint2 tile, uint2 tile_extent) {
    return tile.x + tile.y * tile_extent.x + motion_step * tile_extent.x * tile_extent.y;
}

// Distance along `line_direction` at which a ray from `line_origin` leaves the [-1,1] square.
inline float LineUnitSquareIntersectDist(float2 line_origin, float2 line_direction) {
    const float2 first_plane = (float2(1.0f) - line_origin) / line_direction;
    const float2 second_plane = (float2(-1.0f) - line_origin) / line_direction;
    const float2 farthest_plane = max(first_plane, second_plane);
    return min(farthest_plane.x, farthest_plane.y);
}
inline float LineUnitSquareIntersectDistSafe(float2 line_origin, float2 line_direction) {
    const float2 safe_dir = max(float2(1e-8f), abs(line_direction)) *
        select(float2(1.0f), float2(-1.0f), line_direction < float2(0.0f));
    return LineUnitSquareIntersectDist(line_origin, safe_dir);
}

#endif
