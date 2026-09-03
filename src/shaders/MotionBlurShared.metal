#ifndef MOTION_BLUR_SHARED_MSL
#define MOTION_BLUR_SHARED_MSL

// Ported from Blender EEVEE's eevee_motion_blur.bsl.hh and Guertin, McGuire, and Nowrouzezahrai's feature-aware motion-blur filter.

#include <metal_stdlib>
using namespace metal;

constant int MotionBlurTileSize = 32;

constant uint MotionPrev = 0u;
constant uint MotionNext = 1u;

// Pack source-tile coordinates below motion length so atomic max selects the fastest contributor.
// Bits 31..18 store clamped pixel length, bits 17..9 store tile X, and bits 8..0 store tile Y.
// Nine coordinate bits cover 512 tiles or 16,384 pixels per dimension.
inline uint MotionTilePack(float2 motion, uint2 tile) {
    const uint velocity = min(uint(ceil(length(motion))), 0x3FFFu);
    return (velocity << 18u) | ((tile.x & 0x1FFu) << 9u) | (tile.y & 0x1FFu);
}
inline int2 MotionTileUnpack(uint data) {
    return int2(int((data >> 9u) & 0x1FFu), int(data & 0x1FFu));
}
inline uint MotionTileIndex(uint motion_step, uint2 tile, uint2 tile_extent) {
    return tile.x + tile.y * tile_extent.x + motion_step * tile_extent.x * tile_extent.y;
}

// Returns the distance along `line_direction` at which the ray exits the [-1, 1] square.
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
