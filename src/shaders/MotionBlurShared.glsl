#ifndef MOTION_BLUR_SHARED_GLSL
#define MOTION_BLUR_SHARED_GLSL

// Shared by the motion blur tile passes and the gather.
// Ported from Blender EEVEE (eevee_motion_blur.bsl.hh), which follows
// "A Fast and Stable Feature-Aware Motion Blur Filter" by Guertin, McGuire, and Nowrouzezahrai.

const int MotionBlurTileSize = 32;
// Tile coordinates pack into 9 bits each, so this is the widest tile grid the indirection
// table addresses. At 32 pixels per tile that covers renders up to 16384 pixels wide.
const int MotionBlurMaxTile = 512;

const uint MotionPrev = 0u;
const uint MotionNext = 1u;

// Each entry holds the coordinates of the tile whose motion covers this one, with the motion's
// length in the high bits so atomicMax picks the fastest contributor.
//   bits 31..18: length in pixels, clamped to 16383
//   bits 17..9 : tile x
//   bits 8..0  : tile y
uint MotionTilePack(vec2 motion, uvec2 tile) {
    const uint velocity = min(uint(ceil(length(motion))), 0x3FFFu);
    return (velocity << 18u) | ((tile.x & 0x1FFu) << 9u) | (tile.y & 0x1FFu);
}
ivec2 MotionTileUnpack(uint data) {
    return ivec2((data >> 9u) & 0x1FFu, data & 0x1FFu);
}
uint MotionTileIndex(uint motion_step, uvec2 tile) {
    return tile.x + tile.y * uint(MotionBlurMaxTile) + motion_step * uint(MotionBlurMaxTile) * uint(MotionBlurMaxTile);
}

// Distance along `line_direction` at which a ray from `line_origin` leaves the [-1,1] square.
float LineUnitSquareIntersectDist(vec2 line_origin, vec2 line_direction) {
    const vec2 first_plane = (vec2(1.0) - line_origin) / line_direction;
    const vec2 second_plane = (vec2(-1.0) - line_origin) / line_direction;
    const vec2 farthest_plane = max(first_plane, second_plane);
    return min(farthest_plane.x, farthest_plane.y);
}
float LineUnitSquareIntersectDistSafe(vec2 line_origin, vec2 line_direction) {
    const vec2 safe_dir = max(vec2(1e-8), abs(line_direction)) *
        mix(vec2(1.0), vec2(-1.0), lessThan(line_direction, vec2(0.0)));
    return LineUnitSquareIntersectDist(line_origin, safe_dir);
}

#endif
