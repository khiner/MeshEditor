/* SPDX-FileCopyrightText: 2022 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * EEVEE 222853b0d7e: tile bitmasks and Metal-coherent light iteration.
 * Native adaptation: explicit-range bounds, no depth clipping, original light order.
 */
#pragma once
#include "Bindless.metal"
#include "gpu/LightCulling.h"

inline bool LightSphereInTile(float3 center, float radius, float4x4 projection, uint2 tile, float2 extent) {
    const float margin = .02f * float(LightTileSize);
    const float2 low = (float2(tile * LightTileSize) - margin) / extent * 2.0f - 1.0f;
    const float2 high = (float2((tile + 1u) * LightTileSize) + margin) / extent * 2.0f - 1.0f;
    const float4x4 rows = transpose(projection);
    const float4 planes[4] = {rows[0] - low.x * rows[3], high.x * rows[3] - rows[0],
        rows[1] + high.y * rows[3], -low.y * rows[3] - rows[1]};
    for (uint i = 0; i < 4; ++i) {
        const float4 p = planes[i];
        const float distance = dot(p, float4(center, 1));
        const float rounding = 8.0f * FLT_EPSILON * dot(abs(p), float4(abs(center), 1));
        if (distance < -radius * length(p.xyz) - rounding) return false;
    }
    return true;
}


struct TiledLightIterator {
    uint Count, NextIndex, Word, WordIndex, TileOffset;
    device const uint *Words;

    static TiledLightIterator make(const thread Scene &scene, float3 position, uint count, bool enabled) {
        TiledLightIterator result{count, 0u, 0u, 0u, InvalidOffset, nullptr};
        if (!enabled || count == 0u || scene.View.LightTileSlot == InvalidSlot) return result;
        const float4 clip = scene.ViewProj() * float4(position, 1);
        const float2 uv = clip.xy / clip.w * float2(.5f, -.5f) + .5f;
        if (clip.w > 0.0f && all(isfinite(uv)) && all(uv >= 0.0f) && all(uv < 1.0f)) {
            const uint2 tile = uint2(uv * float2(scene.View.LightTileViewport)) / LightTileSize;
            const uint tiles_x = (uint(scene.View.LightTileViewport.x) + LightTileSize - 1u) / LightTileSize;
            result.TileOffset = (tile.y * tiles_x + tile.x) * ((count + 31u) / 32u);
        }
        result.Words = BindlessBuffer(uint, scene.B.Buffer, scene.View.LightTileSlot);
        return result;
    }

    uint next(bool enabled) {
        if (!enabled) return NextIndex++;
        while (Word == 0u && WordIndex < (Count + 31u) / 32u) {
            const uint mask = TileOffset == InvalidOffset ? ~0u : Words[TileOffset + WordIndex];
            // EEVEE's Metal iterator evaluates one common light set across active lanes.
            Word = simd_or(mask);
            ++WordIndex;
        }
        if (Word == 0u) return Count;
        const uint index = (WordIndex - 1u) * 32u + ctz(Word);
        Word &= Word - 1u;
        return min(index, Count);
    }
};
