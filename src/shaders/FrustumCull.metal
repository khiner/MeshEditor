#ifndef FRUSTUMCULL_MSL
#define FRUSTUMCULL_MSL

// Culls the draw list in place, one thread per indirect command (phase 0) or draw entry (phases
// 1 and 2). Phase 0 zeroes both command regions' instance counts. Phase 1 appends frustum-passing
// entries to region A and the visible-index remap, gated to previously-visible instances when
// VisibilitySlot is set. Phase 2 tests the deferred remainder against the depth pyramid, appends
// survivors to region B, and updates per-instance visibility.
#include "Bindless.metal"
#include "AABB.metal"
#include "CullEntry.metal"
#include "CullFlag.metal"
#include "FrustumCullPushConstants.metal"
#include "ScreenSpace.metal"

// Field for field, MTLDrawIndexedPrimitivesIndirectArguments.
struct IndirectCommand {
    uint IndexCount;
    uint InstanceCount;
    uint FirstIndex;
    int VertexOffset;
    uint FirstInstance;
};

// Row `i` of the view-projection, for Gribb-Hartmann plane extraction.
inline float4 vp_row(float4x4 m, uint i) {
    return float4(m[0][i], m[1][i], m[2][i], m[3][i]);
}

inline bool in_frustum(float4x4 view_proj, float3 center, float3 ax, float3 ay, float3 az) {
    const float4 r3 = vp_row(view_proj, 3);
    const float4 planes[6] = {
        r3 + vp_row(view_proj, 0), r3 - vp_row(view_proj, 0), // left, right
        r3 + vp_row(view_proj, 1), r3 - vp_row(view_proj, 1), // bottom, top
        vp_row(view_proj, 2), r3 - vp_row(view_proj, 2),      // near (zero-to-one depth), far
    };
    for (uint p = 0; p < 6; ++p) {
        const float3 n = planes[p].xyz;
        const float radius = abs(dot(n, ax)) + abs(dot(n, ay)) + abs(dot(n, az));
        if (dot(n, center) + planes[p].w < -radius) return false;
    }
    return true;
}

// True when every part of the world-space box is behind the pyramid's farthest depth over its
// screen footprint. The box's own depth from region A keeps it visible, so this never self-culls.
inline bool occluded(
    const thread Scene &scene, uint pyramid_slot, float3 center, float3 ax, float3 ay, float3 az
) {
    const float4x4 view_proj = scene.ViewProj();
    float2 uv_min = float2(1e30f), uv_max = float2(-1e30f);
    float min_depth = 1e30f;
    for (int c = 0; c < 8; ++c) {
        const float3 corner = center + ((c & 1) != 0 ? ax : -ax) + ((c & 2) != 0 ? ay : -ay) + ((c & 4) != 0 ? az : -az);
        const float4 clip = view_proj * float4(corner, 1.0f);
        if (clip.w <= 0.0f) return false; // Crosses the near plane: keep it visible.
        const float3 ndc = clip.xyz / clip.w;
        const float2 uv = ndc_to_uv(ndc.xy);
        uv_min = min(uv_min, uv);
        uv_max = max(uv_max, uv);
        min_depth = min(min_depth, ndc.z);
    }
    if (min_depth <= 0.0f) return false;

    // Pyramid level L texel x covers scene texels [x << (L + 1), (x + 1) << (L + 1)), so footprints
    // map through the scene extent. The pyramid image is padded to power-of-two dimensions, and
    // clamping to the last data texel keeps every fetch inside the written region.
    const float2 viewport_size = float2(scene.View.ViewportSize);
    const float2 half_scene = viewport_size * 0.5f;
    const float2 min_px = clamp(uv_min, 0.0f, 1.0f) * half_scene;
    const float2 max_px = clamp(uv_max, 0.0f, 1.0f) * half_scene;
    // The level where the footprint spans at most about two texels per dimension.
    const float max_dim = max(max_px.x - min_px.x, max_px.y - min_px.y);
    const int mip_count = int(scene.B.Sampler[pyramid_slot].Texture.get_num_mip_levels());
    const int level = clamp(int(ceil(log2(max(max_dim * 0.5f, 1.0f)))), 0, mip_count - 1);
    const int2 data_max = (int2(viewport_size) - 1) >> (level + 1);
    const int2 lo = clamp(int2(min_px) >> level, int2(0), data_max);
    const int2 hi = clamp(int2(max_px) >> level, int2(0), data_max);
    if (hi.x - lo.x > 3 || hi.y - lo.y > 3) return false; // Footprint outgrew the top level: keep it visible.
    float occluder = 0.0f;
    for (int y = lo.y; y <= hi.y; ++y) {
        for (int x = lo.x; x <= hi.x; ++x) {
            occluder = max(occluder, scene.FetchTex(pyramid_slot, int2(x, y), uint(level)).r);
        }
    }
    return min_depth > occluder;
}

kernel void FrustumCullKernel(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant FrustumCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    device IndirectCommand *commands = BindlessBufferMutable(IndirectCommand, bindless.Buffer, pc.CommandsSlot);

    if (pc.Phase == 0u) {
        if (i < 2u * pc.CommandCount) commands[i].InstanceCount = 0u;
        return;
    }
    if (i >= pc.EntryCount) return;

    const CullEntry entry = BindlessBuffer(CullEntry, bindless.Buffer, pc.CullEntrySlot)[i];
    const uint cmd = entry.Cmd & ~CullFlag_KeepOrder;
    const DrawData draw = scene.Draws(pc.DrawDataSlot)[i];
    // Deforming geometry always draws, at its original remap slot for deterministic instance order.
    // These predicates are uniform across a command's entries (one mesh's DrawData and bounds).
    // Fixed slots therefore never collide with compacted ones.
    const bool fixed_slot = (entry.Cmd & CullFlag_KeepOrder) != 0u || draw.ArmatureDeformOffset != INVALID_OFFSET || draw.MorphDeformOffset != INVALID_OFFSET;
    if (pc.Phase == 2u && fixed_slot) return; // Drew in region A.

    const float4x4 view_proj = scene.ViewProj();
    bool empty = fixed_slot;
    float3 center = float3(0), ax = float3(0), ay = float3(0), az = float3(0);
    if (!fixed_slot) {
        const AABB aabb = BindlessBuffer(AABB, bindless.Buffer, pc.BoundsSlot)[draw.FirstInstance];
        const float3 aabb_min = float3(aabb.Min), aabb_max = float3(aabb.Max);
        empty = aabb_min.x > aabb_max.x; // Extras and empty meshes: always drawn.
        if (!empty) {
            const Transform t = scene.Models(pc.ModelSlot)[draw.FirstInstance];
            const float3 half_local = (aabb_max - aabb_min) * 0.5f;
            const float3 scale = float3(t.S);
            const float4 rotation = float4(t.R);
            center = trs_transform_point(t, (aabb_min + aabb_max) * 0.5f);
            ax = quat_rotate(rotation, float3(scale.x * half_local.x, 0, 0));
            ay = quat_rotate(rotation, float3(0, scale.y * half_local.y, 0));
            az = quat_rotate(rotation, float3(0, 0, scale.z * half_local.z));
        }
    }

    device uint *visible_indices = BindlessBufferMutable(uint, bindless.Buffer, pc.VisibleIndexSlot);

    if (pc.Phase == 1u) {
        bool visible = fixed_slot || empty || in_frustum(view_proj, center, ax, ay, az);
        // Instances hidden last frame defer to the occlusion pass.
        if (visible && !fixed_slot && !empty && pc.VisibilitySlot != INVALID_SLOT &&
            BindlessBuffer(uchar, bindless.Buffer, pc.VisibilitySlot)[draw.FirstInstance] == uchar(0)) {
            visible = false;
        }
        if (visible) {
            device atomic_uint *instance_count = reinterpret_cast<device atomic_uint *>(&commands[cmd].InstanceCount);
            const uint k = atomic_fetch_add_explicit(instance_count, 1u, memory_order_relaxed);
            visible_indices[fixed_slot || empty ? i : entry.Base + k] = i;
        }
        return;
    }

    // Phase 2: settle visibility for every frustum-passing entry, and draw newly visible ones.
    if (empty) return; // Drew in region A.
    device uchar *visibility = BindlessBufferMutable(uchar, bindless.Buffer, pc.VisibilitySlot);
    if (!in_frustum(view_proj, center, ax, ay, az)) {
        visibility[draw.FirstInstance] = uchar(0);
        return;
    }
    const bool was_visible = visibility[draw.FirstInstance] != uchar(0);
    const bool now_hidden = occluded(scene, pc.PyramidSamplerSlot, center, ax, ay, az);
    visibility[draw.FirstInstance] = now_hidden ? uchar(0) : uchar(1);
    if (!was_visible && !now_hidden) {
        device atomic_uint *instance_count = reinterpret_cast<device atomic_uint *>(&commands[pc.CommandCount + cmd].InstanceCount);
        const uint k = atomic_fetch_add_explicit(instance_count, 1u, memory_order_relaxed);
        visible_indices[pc.EntryCount + entry.Base + k] = i;
    }
}

#endif
