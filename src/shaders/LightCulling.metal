// EEVEE tile bitmasks, one 32-lane Metal SIMD group per word.
#include "LightCullingLib.h"
#include "AreaLight.h"

kernel void LightCull(device const BindlessSet &bindless [[buffer(0)]], device uint *words [[buffer(1)]],
    constant SceneViewUBO &view [[buffer(2)]], uint index [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint words_per_tile = (view.LightCount + 31u) / 32u;
    if (words_per_tile == 0u) return;
    const uint tiles_x = (uint(view.ViewportSize.x) + LightTileSize - 1u) / LightTileSize;
    const uint tiles_y = (uint(view.ViewportSize.y) + LightTileSize - 1u) / LightTileSize;
    const uint word = index / 32u, tile_index = word / words_per_tile;
    if (tile_index >= tiles_x * tiles_y) return;
    const uint light_index = (word % words_per_tile) * 32u + lane;
    bool visible = false;
    if (light_index < view.LightCount) {
        const auto light = BindlessBuffer(LightRecord, bindless.LightBuffer, view.LightSlot)[light_index];
        visible = true;
        if (light.Type != LightType::Directional && light.Range > 0.0f && isfinite(light.Range)) {
            const uint slot = view.ModelSlotOverride != InvalidSlot ? view.ModelSlotOverride : light.TransformSlotOffset.Slot;
            const auto transform = BindlessBuffer(Transform, bindless.ModelBuffer, slot)[light.TransformSlotOffset.Offset];
            visible = LightSphereInTile(float3(transform.P), light.Range, view.ViewProj.Unpack(),
                uint2(tile_index % tiles_x, tile_index / tiles_x), float2(view.ViewportSize));
        }
    }
    const uint mask = uint((simd_vote::vote_t)simd_ballot(visible));
    if (lane == 0u) words[word] = mask;
}


kernel void LightPrepare(device const BindlessSet &bindless [[buffer(0)]], device PreparedAreaLight *prepared [[buffer(1)]],
    constant SceneViewUBO &view [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index >= view.LightCount) return;
    const auto light = BindlessBuffer(LightRecord, bindless.LightBuffer, view.LightSlot)[index];
    if (!IsAreaLight(light.Type)) { prepared[index] = {}; return; }
    const uint slot = view.ModelSlotOverride != InvalidSlot ? view.ModelSlotOverride : light.TransformSlotOffset.Slot;
    const auto world = BindlessBuffer(Transform, bindless.ModelBuffer, slot)[light.TransformSlotOffset.Offset];
    prepared[index] = PrepareAreaLight(light, world, index);
}
