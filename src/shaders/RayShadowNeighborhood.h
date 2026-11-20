#pragma once
#include "gpu/Types.h"

// Visibility is binary per emitter. Keep the current pixel's LTC and material
// evaluation, exchanging only shadow test results across coplanar receivers.
struct ShadowNeighborhood {
    uint Center, Axial, Diagonal;

    static ShadowNeighborhood make(uint triangle, uint2 pixel, uint lane) {
        const uint center = 1u << lane;
        // Tile8 covers eight columns. Partial edge groups can have a different
        // stride; only reuse a wave whose pixel coordinates confirm the layout.
        const uint2 first_tile = simd_broadcast_first(pixel / uint2(8, 4));
        if (!simd_all(all(pixel / uint2(8, 4) == first_tile) && (pixel.x % 8u + 8u * (pixel.y % 4u) == lane)))
            return {center, 0u, 0u};
        const uint sides = ((center & 0xfefefefeu) >> 1u) | ((center & 0x7f7f7f7fu) << 1u);
        const uint axial = sides | (center >> 8u) | (center << 8u);
        const uint diagonal = (sides >> 8u) | (sides << 8u);
        // A visibility ID identifies one rasterized triangle in one instance.
        // A wave on a large receiver usually has just one such ID.
        uint remaining = uint((simd_vote::vote_t)simd_ballot(true));
        uint surface = center;
        while (remaining) {
            const uint id = simd_shuffle(triangle, ctz(remaining));
            const uint matches = uint((simd_vote::vote_t)simd_ballot(triangle == id));
            if (triangle == id) surface = matches;
            remaining &= ~matches;
        }
        return {center, axial & surface, diagonal & surface};
    }

    float weight(uint votes) const {
        return float(9u * popcount(Center & votes) + 6u * popcount(Axial & votes) + 4u * popcount(Diagonal & votes));
    }

    float visibility(bool lit) const {
        const uint active = uint((simd_vote::vote_t)simd_ballot(true));
        const uint votes = uint((simd_vote::vote_t)simd_ballot(lit));
        if (votes == active || votes == 0u) return float(lit);
        return weight(votes) / weight(active);
    }
};
