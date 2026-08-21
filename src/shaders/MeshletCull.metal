#include "AABB.metal"
#include "Bindless.metal"
#include "Frustum.metal"
#include "InstanceRecord.metal"
#include "MaterialAlphaMode.metal"
#include "MeshDispatchArgs.metal"
#include "MeshletBlendBlockState.metal"
#include "MeshletCullBlockState.metal"
#include "MeshletCullPushConstants.metal"
#include "MeshletRecord.metal"
#include "MeshletRouteState.metal"
#include "MeshletWorkBlockState.metal"
#include "MeshletWorkRange.metal"
#include "MeshletWorkState.metal"
#include "PrimitiveRecord.metal"
#include "ScreenSpace.metal"
#include "VisibleMeshlet.metal"

constant uint CullBlockSize = 1024u;
constant uint CullRouteCount = 5u;
constant uint CullSimdGroups = 32u;
constant uint PrefixStride = CullSimdGroups + 1u;
constant uint ConeCullMinTriangles = 16u;
// The phase-2 cull kernels run one 32-lane simdgroup per threadgroup.
constant uint Phase2GroupSize = 32u;

struct RoutedMeshlet {
    uint Routes;
    uint BlendBucket;
};

struct OrientedBounds {
    float3 Center;
    float3 Ax, Ay, Az;
    bool Valid;
};

inline bool InstanceDeformed(InstanceRecord instance) {
    return instance.ArmatureDeformOffset != INVALID_OFFSET || instance.MorphDeformOffset != INVALID_OFFSET ||
        instance.PosedPositionOffset != INVALID_OFFSET || instance.HasPendingVertexTransform != 0u;
}

inline uint PrimitiveMaterialIndex(const thread Scene &scene, PrimitiveRecord primitive) {
    if (primitive.Draw.PrimitiveMaterialOffset == INVALID_OFFSET) return 0u;
    return scene.PrimitiveMaterials(scene.View.PrimitiveMaterialSlot)[primitive.Draw.PrimitiveMaterialOffset + primitive.PrimitiveIndex];
}

inline bool MeshletConeVisible(
    const thread Scene &scene, MeshletRecord meshlet, Transform world, bool deformed
) {
    const float3 scale = float3(world.S);
    if (meshlet.TriangleCount < ConeCullMinTriangles || deformed || any(scale < 0.0f)) return true;
    const float max_scale = max(scale.x, max(scale.y, scale.z));
    const float min_scale = min(scale.x, min(scale.y, scale.z));
    if (max_scale - min_scale > 1e-5f * max(max_scale, 1.0f)) return true;

    const int4 cone = int4(
        int(char(meshlet.ConeAxisCutoff & 0xffu)),
        int(char((meshlet.ConeAxisCutoff >> 8u) & 0xffu)),
        int(char((meshlet.ConeAxisCutoff >> 16u) & 0xffu)),
        int(char(meshlet.ConeAxisCutoff >> 24u))
    );
    if (cone.w >= 127) return true;
    const float3 axis = quat_rotate(float4(world.R), float3(cone.xyz) / 127.0f);
    const float cutoff = float(cone.w) / 127.0f;
    const float3 center = trs_transform_point(world, float3(meshlet.Center));
    const float3 camera_to_center = center - float3(scene.View.CameraPosition);
    return dot(camera_to_center, axis) < cutoff * length(camera_to_center) + meshlet.Radius * max_scale;
}

inline OrientedBounds TransformBounds(AABB bounds, Transform world) {
    const float3 lo = float3(bounds.Min), hi = float3(bounds.Max);
    if (lo.x > hi.x) return {};
    const float3 half_local = (hi - lo) * 0.5f;
    const float3 scale = float3(world.S);
    const float4 rotation = float4(world.R);
    return {
        trs_transform_point(world, (lo + hi) * 0.5f),
        quat_rotate(rotation, float3(scale.x * half_local.x, 0, 0)),
        quat_rotate(rotation, float3(0, scale.y * half_local.y, 0)),
        quat_rotate(rotation, float3(0, 0, scale.z * half_local.z)),
        true,
    };
}

inline OrientedBounds InstanceBounds(const thread Scene &scene, MeshletCullPushConstants pc, uint instance_slot) {
    const AABB bounds = BindlessBuffer(AABB, scene.B.Buffer, pc.BoundsSlot)[instance_slot];
    return TransformBounds(bounds, scene.Models(pc.ModelSlot)[instance_slot]);
}

inline OrientedBounds DeformedMeshletBounds(
    const thread Scene &scene, MeshletCullPushConstants pc, VisibleMeshlet candidate,
    InstanceRecord instance, Transform world
) {
    if (instance.PosedMeshletBoundsOffset == INVALID_OFFSET || pc.PosedMeshletBoundsSlot == INVALID_SLOT) return {};
    const PrimitiveRecord first = BindlessBuffer(PrimitiveRecord, scene.B.Buffer, pc.PrimitiveSlot)[instance.PrimitiveOffset];
    const uint local_meshlet = candidate.Meshlet - first.MeshletOffset;
    const AABB bounds = BindlessBuffer(AABB, scene.B.Buffer, pc.PosedMeshletBoundsSlot)
        [instance.PosedMeshletBoundsOffset + local_meshlet];
    return TransformBounds(bounds, world);
}

// A meshlet's world bounds: posed OBB for deformed instances (instance OBB when no posed bounds
// exist), the scaled bounding sphere as axis vectors otherwise. Invalid means no bounds are
// available and the meshlet must be treated as visible and unoccludable.
struct MeshletBounds {
    float3 Center;
    float3 Ax, Ay, Az;
    float Radius; // Set when Sphere, for the tighter sphere frustum test.
    bool Sphere;
    bool Valid;
};

inline MeshletBounds ResolveMeshletBounds(
    const thread Scene &scene, MeshletCullPushConstants pc, VisibleMeshlet candidate,
    uint instance_slot, InstanceRecord instance, MeshletRecord meshlet, Transform world
) {
    if (InstanceDeformed(instance)) {
        OrientedBounds bounds = DeformedMeshletBounds(scene, pc, candidate, instance, world);
        if (!bounds.Valid) bounds = InstanceBounds(scene, pc, instance_slot);
        if (!bounds.Valid) return {};
        return {bounds.Center, bounds.Ax, bounds.Ay, bounds.Az, 0.0f, false, true};
    }
    const float3 scale = abs(float3(world.S));
    const float radius = meshlet.Radius * max(scale.x, max(scale.y, scale.z));
    return {
        trs_transform_point(world, float3(meshlet.Center)),
        float3(radius, 0, 0), float3(0, radius, 0), float3(0, 0, radius),
        radius, true, true,
    };
}

inline bool MeshletBoundsInFrustum(const thread Scene &scene, MeshletBounds bounds) {
    return bounds.Sphere ?
        sphere_in_frustum(scene.ViewProj(), bounds.Center, bounds.Radius) :
        in_frustum(scene.ViewProj(), bounds.Center, bounds.Ax, bounds.Ay, bounds.Az);
}

inline bool MeshletOccluded(
    const thread Scene &scene, uint pyramid_slot, float4x4 view_proj,
    float3 center, float3 ax, float3 ay, float3 az
) {
    float2 uv_min = float2(1e30f), uv_max = float2(-1e30f);
    float min_depth = 1e30f;
    for (uint c = 0; c < 8; ++c) {
        const float3 corner = center + ((c & 1u) ? ax : -ax) + ((c & 2u) ? ay : -ay) + ((c & 4u) ? az : -az);
        const float4 clip = view_proj * float4(corner, 1.0f);
        if (clip.w <= 0.0f) return false;
        const float3 ndc = clip.xyz / clip.w;
        const float2 uv = ndc_to_uv(ndc.xy);
        uv_min = min(uv_min, uv);
        uv_max = max(uv_max, uv);
        min_depth = min(min_depth, ndc.z);
    }
    if (min_depth <= 0.0f) return false;
    const float2 viewport_size = float2(scene.View.ViewportSize);
    const float2 min_px = clamp(uv_min, 0.0f, 1.0f) * viewport_size * 0.5f;
    const float2 max_px = clamp(uv_max, 0.0f, 1.0f) * viewport_size * 0.5f;
    const float max_dim = max(max_px.x - min_px.x, max_px.y - min_px.y);
    const int mip_count = int(scene.B.Sampler[pyramid_slot].Texture.get_num_mip_levels());
    const int level = clamp(int(ceil(log2(max(max_dim * 0.5f, 1.0f)))), 0, mip_count - 1);
    const int2 data_max = (int2(viewport_size) - 1) >> (level + 1);
    const int2 lo = clamp(int2(min_px) >> level, int2(0), data_max);
    const int2 hi = clamp(int2(max_px) >> level, int2(0), data_max);
    if (hi.x - lo.x > 3 || hi.y - lo.y > 3) return false;
    float occluder = 0.0f;
    for (int y = lo.y; y <= hi.y; ++y) {
        for (int x = lo.x; x <= hi.x; ++x) occluder = max(occluder, scene.FetchTex(pyramid_slot, int2(x, y), uint(level)).r);
    }
    return min_depth > occluder;
}

// 0 rejects the instance, 1 expands its meshlets in phase 1, and 2 defers the whole
// instance range to the current-pyramid phase.  Deferring one instance id avoids materializing
// every meshlet behind a previous-frame occluder.
inline uint ClassifyInstanceRange(
    const thread Scene &scene, MeshletCullPushConstants pc, uint instance_slot, InstanceRecord instance
) {
    if (instance.PrimitiveCount == 0u || (instance.Flags & pc.RequiredInstanceFlags) != pc.RequiredInstanceFlags) return 0u;
    // Posed meshlet bounds are the authoritative deformed bounds; the instance AABB may belong to
    // another blur step, so it must not reject the range before per-meshlet classification.
    if (instance.PosedMeshletBoundsOffset != INVALID_OFFSET) return 1u;
    const OrientedBounds bounds = InstanceBounds(scene, pc, instance_slot);
    if (!bounds.Valid) return 1u;
    if (!in_frustum(scene.ViewProj(), bounds.Center, bounds.Ax, bounds.Ay, bounds.Az)) return 0u;
    if (pc.PyramidSamplerSlot == INVALID_SLOT ||
        !MeshletOccluded(scene, pc.PyramidSamplerSlot, pc.OcclusionViewProj.Unpack(), bounds.Center, bounds.Ax, bounds.Ay, bounds.Az)) return 1u;
    // Blend and silhouette routes are intentionally single-phase. Expand this instance so their
    // meshlets can remain in phase 1 while opaque meshlets are deferred independently.
    if (pc.TwoPhase != 0u && (pc.BlendBlockSlot != INVALID_SLOT ||
        (pc.ExtraInstanceFlags != 0u && (instance.Flags & pc.ExtraInstanceFlags) == pc.ExtraInstanceFlags))) return 1u;
    return pc.TwoPhase != 0u ? 2u : 0u;
}

inline RoutedMeshlet ClassifyMeshlet(
    const thread Scene &scene, MeshletCullPushConstants pc, VisibleMeshlet candidate,
    uint instance_slot, InstanceRecord instance
) {
    RoutedMeshlet result{0u, 0u};
    const MeshletRecord meshlet = BindlessBuffer(MeshletRecord, scene.B.Buffer, pc.MeshletSlot)[candidate.Meshlet];
    const Transform world = scene.Models(pc.ModelSlot)[instance_slot];
    const MeshletBounds bounds = ResolveMeshletBounds(scene, pc, candidate, instance_slot, instance, meshlet, world);
    if (bounds.Valid && !MeshletBoundsInFrustum(scene, bounds)) return result;
    const float3 world_center = bounds.Valid ? bounds.Center : float3(world.P);

    const PrimitiveRecord primitive = BindlessBuffer(PrimitiveRecord, scene.B.Buffer, pc.PrimitiveSlot)[meshlet.Primitive];
    // A one-meshlet instance already passed the conservative instance query. Repeating the
    // eight-corner pyramid query for its tighter sphere is optional extra rejection, not correctness.
    const bool can_occlude = bounds.Valid && !(instance.PrimitiveCount == 1u && primitive.MeshletCount == 1u);
    PBRMaterial material{};
    if (pc.RouteMode != 0u) material = scene.Materials(scene.View.MaterialSlot)[PrimitiveMaterialIndex(scene, primitive)];
    const bool cone_visible = pc.RouteMode == 0u || material.DoubleSided != 0u ||
        MeshletConeVisible(scene, meshlet, world, InstanceDeformed(instance));
    const bool occluded = can_occlude && pc.PyramidSamplerSlot != INVALID_SLOT &&
        MeshletOccluded(scene, pc.PyramidSamplerSlot, pc.OcclusionViewProj.Unpack(), world_center, bounds.Ax, bounds.Ay, bounds.Az);

    if (pc.RouteMode == 0u) {
        result.Routes = 1u;
    } else {
        if (material.AlphaMode == MaterialAlphaMode_Blend) {
            result.Routes = 2u;
            const float4 clip = scene.ViewProj() * float4(world_center, 1.0f);
            result.BlendBucket = clip.w > 0.0f ? uint(clamp(clip.z / clip.w, 0.0f, 1.0f) * 255.0f) : 0u;
        } else if (pc.RouteMode == 1u) {
            result.Routes = 1u;
        } else {
            const bool transmissive = material.Transmission.Factor > 0.0f;
            if (!transmissive || material.Transmission.Texture.Slot != INVALID_SLOT) result.Routes |= 1u;
            if (transmissive) result.Routes |= 4u;
        }
    }
    if (pc.ExtraInstanceFlags != 0u && (instance.Flags & pc.ExtraInstanceFlags) == pc.ExtraInstanceFlags) result.Routes |= 8u;
    // Cone culling applies only to the material routes.  Silhouette/id routes are two-sided and may
    // share this classification through ExtraInstanceFlags.
    if (!cone_visible) result.Routes &= 8u;
    if (result.Routes == 0u) return result;
    if (occluded) {
        if (pc.TwoPhase != 0u) {
            // Opaque route 0 moves to the current-pyramid phase. Blend and silhouette remain in
            // phase 1, globally sorted and drawn exactly once as in the single-phase path.
            result.Routes = (result.Routes & (2u | 8u)) | ((result.Routes & 1u) != 0u ? 16u : 0u);
        } else {
            result.Routes = 0u;
        }
    }
    return result;
}

inline VisibleMeshlet ResolveMeshlet(
    device const BindlessSet &bindless, MeshletCullPushConstants pc, uint block_id, uint work_index
) {
    device const MeshletWorkState *state = BindlessBuffer(MeshletWorkState, bindless.Buffer, pc.WorkStateSlot);
    if (work_index >= state->MeshletCount) return {INVALID_OFFSET, INVALID_OFFSET};
    device const uint *work_blocks = BindlessBuffer(uint, bindless.Buffer, pc.WorkBlockSlot);
    device const MeshletWorkRange *ranges = BindlessBuffer(MeshletWorkRange, bindless.Buffer, pc.WorkRangeSlot);
    uint lo = work_blocks[block_id];
    uint hi = block_id + 1u < state->CullBlockCount ? min(work_blocks[block_id + 1u] + 1u, state->RangeCount) : state->RangeCount;
    while (lo + 1u < hi) {
        const uint mid = (lo + hi) / 2u;
        if (ranges[mid].WorkOffset <= work_index) lo = mid;
        else hi = mid;
    }
    const MeshletWorkRange range = ranges[lo];
    return {range.Instance, range.MeshletOffset + work_index - range.WorkOffset};
}

struct InstanceWork {
    InstanceRecord Instance;
    uint RangeCount;
    uint MeshletCount;
    uint Phase2RangeCount;
};

inline InstanceWork ResolveInstanceWork(const thread Scene &scene, MeshletCullPushConstants pc, uint id) {
    if (id >= pc.InstanceCount) return {};
    const uint instance_slot = BindlessBuffer(uint, scene.B.Buffer, pc.InstanceMapSlot)[id];
    if (instance_slot == INVALID_OFFSET) return {};
    const InstanceRecord instance = BindlessBuffer(InstanceRecord, scene.B.Buffer, pc.InstanceSlot)[instance_slot];
    const uint visibility = ClassifyInstanceRange(scene, pc, instance_slot, instance);
    if (visibility == 0u) return {};
    uint range_count = 0u, meshlet_count = 0u;
    device const PrimitiveRecord *primitives = BindlessBuffer(PrimitiveRecord, scene.B.Buffer, pc.PrimitiveSlot);
    for (uint p = 0u; p < instance.PrimitiveCount; ++p) {
        const uint count = primitives[instance.PrimitiveOffset + p].MeshletCount;
        range_count += count != 0u;
        meshlet_count += count;
    }
    if (visibility == 2u) return {instance, 0u, 0u, range_count};
    return {instance, range_count, meshlet_count, 0u};
}

kernel void MeshletWorkBlockCount(
    uint lane [[thread_index_in_threadgroup]], uint block_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *group_prefixes [[threadgroup(0)]]
) {
    const uint id = block_id * CullBlockSize + lane;
    const Scene scene{bindless, view, theme, workspace};
    const InstanceWork work = ResolveInstanceWork(scene, pc, id);
    const uint range_sum = simd_sum(work.RangeCount);
    const uint meshlet_sum = simd_sum(work.MeshletCount);
    const uint phase2_range_sum = simd_sum(work.Phase2RangeCount);
    if (simd_lane == 0u) {
        group_prefixes[simd_group] = range_sum;
        group_prefixes[PrefixStride + simd_group] = meshlet_sum;
        group_prefixes[2u * PrefixStride + simd_group] = phase2_range_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u && simd_lane == 0u) {
        uint ranges = 0u, meshlets = 0u, phase2_ranges = 0u;
        for (uint group = 0u; group < CullSimdGroups; ++group) {
            ranges += group_prefixes[group];
            meshlets += group_prefixes[PrefixStride + group];
            phase2_ranges += group_prefixes[2u * PrefixStride + group];
        }
        BindlessBufferMutable(MeshletWorkBlockState, bindless.Buffer, pc.WorkBlockStateSlot)[block_id] = {
            ranges, meshlets, phase2_ranges
        };
    }
}

kernel void MeshletWorkPrefix(
    uint lane [[thread_index_in_threadgroup]], device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (lane != 0u) return;
    device MeshletWorkBlockState *blocks = BindlessBufferMutable(MeshletWorkBlockState, bindless.Buffer, pc.WorkBlockStateSlot);
    uint range_count = 0u, meshlet_count = 0u, phase2_range_count = 0u;
    for (uint block = 0u; block < pc.WorkBlockCount; ++block) {
        const MeshletWorkBlockState count = blocks[block];
        blocks[block] = {range_count, meshlet_count, phase2_range_count};
        range_count += count.RangeCount;
        meshlet_count += count.MeshletCount;
        phase2_range_count += count.Phase2RangeCount;
    }
    const uint cull_block_count = (meshlet_count + CullBlockSize - 1u) / CullBlockSize;
    BindlessBufferMutable(MeshletWorkState, bindless.Buffer, pc.WorkStateSlot)[0] = {
        range_count, meshlet_count, cull_block_count, phase2_range_count
    };
    BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.WorkDispatchArgsSlot)[0] = {cull_block_count, 1u, 1u};
    if (pc.TwoPhase != 0u) {
        BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.Phase2RangeCullArgsSlot)[0] = {
            phase2_range_count, 1u, 1u
        };
    }
}

kernel void MeshletWorkEmit(
    uint lane [[thread_index_in_threadgroup]], uint block_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *group_prefixes [[threadgroup(0)]]
) {
    const uint id = block_id * CullBlockSize + lane;
    const Scene scene{bindless, view, theme, workspace};
    const InstanceWork work = ResolveInstanceWork(scene, pc, id);
    const InstanceRecord instance = work.Instance;
    const uint range_count = work.RangeCount, meshlet_count = work.MeshletCount;
    const uint phase2_range_count = work.Phase2RangeCount;
    uint range_rank = simd_prefix_exclusive_sum(range_count);
    uint meshlet_rank = simd_prefix_exclusive_sum(meshlet_count);
    uint phase2_range_rank = simd_prefix_exclusive_sum(phase2_range_count);
    const uint range_sum = simd_sum(range_count);
    const uint meshlet_sum = simd_sum(meshlet_count);
    const uint phase2_range_sum = simd_sum(phase2_range_count);
    if (simd_lane == 0u) {
        group_prefixes[simd_group] = range_sum;
        group_prefixes[PrefixStride + simd_group] = meshlet_sum;
        group_prefixes[2u * PrefixStride + simd_group] = phase2_range_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        const uint ranges = simd_lane < CullSimdGroups ? group_prefixes[simd_lane] : 0u;
        const uint meshlets = simd_lane < CullSimdGroups ? group_prefixes[PrefixStride + simd_lane] : 0u;
        const uint phase2_ranges = simd_lane < CullSimdGroups ? group_prefixes[2u * PrefixStride + simd_lane] : 0u;
        if (simd_lane < CullSimdGroups) {
            group_prefixes[simd_lane] = simd_prefix_exclusive_sum(ranges);
            group_prefixes[PrefixStride + simd_lane] = simd_prefix_exclusive_sum(meshlets);
            group_prefixes[2u * PrefixStride + simd_lane] = simd_prefix_exclusive_sum(phase2_ranges);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const MeshletWorkBlockState block = BindlessBuffer(MeshletWorkBlockState, bindless.Buffer, pc.WorkBlockStateSlot)[block_id];
    if (phase2_range_count != 0u) {
        phase2_range_rank += group_prefixes[2u * PrefixStride + simd_group];
        uint output = block.Phase2RangeCount + phase2_range_rank;
        device const PrimitiveRecord *primitives = BindlessBuffer(PrimitiveRecord, bindless.Buffer, pc.PrimitiveSlot);
        device MeshletWorkRange *phase2_ranges = BindlessBufferMutable(
            MeshletWorkRange, bindless.Buffer, pc.Phase2RangeCandidateSlot
        );
        for (uint p = 0u; p < instance.PrimitiveCount; ++p) {
            const PrimitiveRecord primitive = primitives[instance.PrimitiveOffset + p];
            if (primitive.MeshletCount == 0u) continue;
            phase2_ranges[output++] = {id, primitive.MeshletOffset, primitive.MeshletCount, 0u};
        }
    }
    if (range_count == 0u) return;

    uint range_index = block.RangeCount + group_prefixes[simd_group] + range_rank;
    uint work_offset = block.MeshletCount + group_prefixes[PrefixStride + simd_group] + meshlet_rank;
    device const PrimitiveRecord *primitives = BindlessBuffer(PrimitiveRecord, bindless.Buffer, pc.PrimitiveSlot);
    device MeshletWorkRange *ranges = BindlessBufferMutable(MeshletWorkRange, bindless.Buffer, pc.WorkRangeSlot);
    device uint *work_blocks = BindlessBufferMutable(uint, bindless.Buffer, pc.WorkBlockSlot);
    for (uint p = 0u; p < instance.PrimitiveCount; ++p) {
        const PrimitiveRecord primitive = primitives[instance.PrimitiveOffset + p];
        if (primitive.MeshletCount == 0u) continue;
        ranges[range_index] = {id, primitive.MeshletOffset, primitive.MeshletCount, work_offset};
        const uint first_block = (work_offset + CullBlockSize - 1u) / CullBlockSize;
        const uint last_block = (work_offset + primitive.MeshletCount - 1u) / CullBlockSize;
        for (uint work_block = first_block; work_block <= last_block; ++work_block) work_blocks[work_block] = range_index;
        ++range_index;
        work_offset += primitive.MeshletCount;
    }
}

kernel void MeshletCullBlockCount(
    uint lane [[thread_index_in_threadgroup]], uint block_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *group_prefixes [[threadgroup(0)]]
) {
    device MeshletCullBlockState *blocks = BindlessBufferMutable(MeshletCullBlockState, bindless.Buffer, pc.BlockStateSlot);
    const bool sort_blend = pc.BlendBlockSlot != INVALID_SLOT;
    threadgroup ushort *group_blend_counts = reinterpret_cast<threadgroup ushort *>(group_prefixes + CullRouteCount * PrefixStride);
    if (sort_blend) {
        for (uint j = lane; j < CullSimdGroups * 256u; j += CullBlockSize) group_blend_counts[j] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint i = block_id * CullBlockSize + lane;
    const VisibleMeshlet work = ResolveMeshlet(bindless, pc, block_id, i);
    uint routes = 0u, blend_bucket = 0u;
    if (work.Instance != INVALID_OFFSET) {
        const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[work.Instance];
        if (instance_slot != INVALID_OFFSET) {
            const InstanceRecord instance = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[instance_slot];
            const Scene scene{bindless, view, theme, workspace};
            const RoutedMeshlet routed = ClassifyMeshlet(scene, pc, work, instance_slot, instance);
            routes = routed.Routes;
            blend_bucket = routed.BlendBucket;
        }
    }
    for (uint route = 0u; route < CullRouteCount; ++route) {
        const uint present = (routes >> route) & 1u;
        const uint count = simd_sum(present);
        if (simd_lane == 0u) group_prefixes[route * PrefixStride + simd_group] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u && simd_lane == 0u) {
        for (uint route = 0u; route < CullRouteCount; ++route) {
            uint total = 0u;
            for (uint group = 0u; group < CullSimdGroups; ++group) total += group_prefixes[route * PrefixStride + group];
            blocks[block_id].Routes[route] = total;
        }
    }

    if (sort_blend) {
        const uint present = (routes >> 1u) & 1u;
        uint blend_rank = 0u, blend_count = 0u;
        if (present != 0u) {
            for (uint source = 0u; source < 32u; ++source) {
                const bool match = simd_shuffle(present, source) != 0u && simd_shuffle(blend_bucket, source) == blend_bucket;
                blend_count += match;
                blend_rank += match && source < simd_lane;
            }
            if (blend_rank == 0u) group_blend_counts[simd_group * 256u + blend_bucket] = blend_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device MeshletBlendBlockState *blend_blocks = BindlessBufferMutable(MeshletBlendBlockState, bindless.Buffer, pc.BlendBlockSlot);
        for (uint bucket = lane; bucket < 256u; bucket += CullBlockSize) {
            uint count = 0u;
            for (uint group = 0u; group < CullSimdGroups; ++group) count += group_blend_counts[group * 256u + bucket];
            blend_blocks[block_id].Buckets[bucket] = count;
        }
    }
    if (work.Instance != INVALID_OFFSET) BindlessBufferMutable(ushort, bindless.Buffer, pc.ClassificationSlot)[i] = ushort(routes | (blend_bucket << 5u));
}

kernel void MeshletCullPrefix(
    uint lane [[thread_index_in_threadgroup]], device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]], threadgroup uint *route_totals [[threadgroup(0)]]
) {
    const uint block_count = BindlessBuffer(MeshletWorkState, bindless.Buffer, pc.WorkStateSlot)[0].CullBlockCount;
    device MeshletCullBlockState *blocks = BindlessBufferMutable(MeshletCullBlockState, bindless.Buffer, pc.BlockStateSlot);
    device MeshletRouteState *state = BindlessBufferMutable(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot);
    device MeshDispatchArgs *args = BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.DispatchArgsSlot);
    const bool sort_blend = pc.BlendBlockSlot != INVALID_SLOT;
    if (lane < CullRouteCount) {
        uint total = 0u;
        if (lane != 2u) {
            for (uint block = 0u; block < block_count; ++block) {
                const uint count = blocks[block].Routes[lane];
                blocks[block].Routes[lane] = total;
                total += count;
            }
        } else {
            for (uint block = block_count; block-- > 0u;) {
                const uint count = blocks[block].Routes[lane];
                blocks[block].Routes[lane] = total;
                total += count;
            }
        }
        route_totals[lane] = total;
    }
    if (sort_blend) {
        device MeshletBlendBlockState *blend_blocks = BindlessBufferMutable(MeshletBlendBlockState, bindless.Buffer, pc.BlendBlockSlot);
        uint total = 0u;
        for (uint block = 0u; block < block_count; ++block) {
            const uint count = blend_blocks[block].Buckets[lane];
            blend_blocks[block].Buckets[lane] = total;
            total += count;
        }
        state->BlendOffsets[lane] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (lane == 0u) {
        if (sort_blend) {
            uint blend_offset = 0u;
            for (int bucket = 255; bucket >= 0; --bucket) {
                const uint count = state->BlendOffsets[bucket];
                state->BlendOffsets[bucket] = blend_offset;
                blend_offset += count;
            }
        }
        uint route_offset = 0u;
        for (uint route = 0u; route < CullRouteCount; ++route) {
            state->Counts[route] = route_totals[route];
            state->Offsets[route] = route_offset;
            route_offset += route_totals[route];
            for (uint chunk = 0u; chunk < pc.DispatchChunkCount; ++chunk) {
                const uint begin = chunk * pc.DispatchChunkSize;
                const uint count = route_totals[route] > begin ? min(route_totals[route] - begin, pc.DispatchChunkSize) : 0u;
                args[route * pc.DispatchChunkCount + chunk] = {count, 1u, 1u};
            }
        }
        if (pc.TwoPhase != 0u) {
            BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.Phase2CullArgsSlot)[0] = {
                (route_totals[4] + Phase2GroupSize - 1u) / Phase2GroupSize, 1u, 1u
            };
        }
    }
}

kernel void MeshletCullEmit(
    uint lane [[thread_index_in_threadgroup]], uint block_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *group_prefixes [[threadgroup(0)]]
) {
    const uint i = block_id * CullBlockSize + lane;
    const VisibleMeshlet work = ResolveMeshlet(bindless, pc, block_id, i);
    const bool valid = work.Instance != INVALID_OFFSET;
    const bool sort_blend = pc.BlendBlockSlot != INVALID_SLOT;
    const uint classification = valid ? BindlessBuffer(ushort, bindless.Buffer, pc.ClassificationSlot)[i] : 0u;
    const uint routes = classification & 31u;
    const uint blend_bucket = classification >> 5u;
    threadgroup ushort *group_blend_counts = reinterpret_cast<threadgroup ushort *>(group_prefixes + CullRouteCount * PrefixStride);
    if (sort_blend) {
        for (uint j = lane; j < CullSimdGroups * 256u; j += CullBlockSize) group_blend_counts[j] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint present[CullRouteCount], rank[CullRouteCount];
    for (uint route = 0u; route < CullRouteCount; ++route) {
        present[route] = (routes >> route) & 1u;
        rank[route] = simd_prefix_exclusive_sum(present[route]);
        const uint count = simd_sum(present[route]);
        if (simd_lane == 0u) group_prefixes[route * PrefixStride + simd_group] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        for (uint route = 0u; route < CullRouteCount; ++route) {
            const uint count = simd_lane < CullSimdGroups ? group_prefixes[route * PrefixStride + simd_lane] : 0u;
            const uint total = simd_sum(count);
            if (simd_lane < CullSimdGroups) group_prefixes[route * PrefixStride + simd_lane] = simd_prefix_exclusive_sum(count);
            if (simd_lane == 0u) group_prefixes[route * PrefixStride + CullSimdGroups] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint blend_rank = 0u;
    if (sort_blend && present[1] != 0u) {
        uint blend_count = 0u;
        for (uint source = 0u; source < 32u; ++source) {
            const bool match = simd_shuffle(present[1], source) != 0u && simd_shuffle(blend_bucket, source) == blend_bucket;
            blend_count += match;
            blend_rank += match && source < simd_lane;
        }
        if (blend_rank == 0u) group_blend_counts[simd_group * 256u + blend_bucket] = blend_count;
    }
    if (sort_blend) threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sort_blend && present[1] != 0u) {
        for (uint group = 0u; group < simd_group; ++group) blend_rank += group_blend_counts[group * 256u + blend_bucket];
    }
    if (!valid) return;

    device const MeshletCullBlockState *blocks = BindlessBuffer(MeshletCullBlockState, bindless.Buffer, pc.BlockStateSlot);
    device const MeshletRouteState *state = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot);
    device VisibleMeshlet *visible = BindlessBufferMutable(VisibleMeshlet, bindless.Buffer, pc.VisibleSlot);
    for (uint route = 0u; route < CullRouteCount; ++route) {
        if (present[route] == 0u) continue;
        rank[route] += group_prefixes[route * PrefixStride + simd_group];
        if (route == 2u) rank[route] = group_prefixes[route * PrefixStride + CullSimdGroups] - rank[route] - 1u;
        uint output = state->Offsets[route] + blocks[block_id].Routes[route] + rank[route];
        if (sort_blend && route == 1u) {
            device const MeshletBlendBlockState *blend_blocks = BindlessBuffer(MeshletBlendBlockState, bindless.Buffer, pc.BlendBlockSlot);
            output = state->Offsets[1] + state->BlendOffsets[blend_bucket] + blend_blocks[block_id].Buckets[blend_bucket] + blend_rank;
        }
        visible[output] = work;
    }
}

inline bool Phase2ExpandedMeshletVisible(
    const thread Scene &scene, MeshletCullPushConstants pc, VisibleMeshlet candidate,
    uint instance_slot, InstanceRecord instance
) {
    const MeshletRecord meshlet = BindlessBuffer(MeshletRecord, scene.B.Buffer, pc.MeshletSlot)[candidate.Meshlet];
    const Transform world = scene.Models(pc.ModelSlot)[instance_slot];
    if (pc.RouteMode != 0u) {
        const PrimitiveRecord primitive = BindlessBuffer(PrimitiveRecord, scene.B.Buffer, pc.PrimitiveSlot)[meshlet.Primitive];
        const PBRMaterial material = scene.Materials(scene.View.MaterialSlot)[PrimitiveMaterialIndex(scene, primitive)];
        // Blend keeps the previous-frame single-phase behavior and is never redrawn in phase 2.
        if (material.AlphaMode == MaterialAlphaMode_Blend) return false;
        if (material.DoubleSided == 0u && !MeshletConeVisible(scene, meshlet, world, InstanceDeformed(instance))) return false;
    }
    const MeshletBounds bounds = ResolveMeshletBounds(scene, pc, candidate, instance_slot, instance, meshlet, world);
    if (!bounds.Valid) return true;
    if (!MeshletBoundsInFrustum(scene, bounds)) return false;
    return !MeshletOccluded(scene, pc.PyramidSamplerSlot, scene.ViewProj(), bounds.Center, bounds.Ax, bounds.Ay, bounds.Az);
}

// Both phase-2 cull kernels run twice over 32-lane groups: a count pass writes survivor counts,
// MeshletPhase2Prefix turns them into offsets, and an emit pass writes each survivor at its
// deterministic rank. Emission order is candidate order, independent of GPU scheduling.
kernel void MeshletPhase2Cull(
    uint lane [[thread_index_in_threadgroup]], uint block_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    device const MeshletRouteState *routes = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot);
    const uint i = block_id * Phase2GroupSize + lane;
    bool visible = false;
    VisibleMeshlet candidate{};
    if (i < routes->Counts[4]) {
        candidate = BindlessBuffer(VisibleMeshlet, bindless.Buffer, pc.VisibleSlot)[routes->Offsets[4] + i];
        const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[candidate.Instance];
        if (instance_slot != INVALID_OFFSET) {
            const InstanceRecord instance = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[instance_slot];
            const MeshletRecord meshlet = BindlessBuffer(MeshletRecord, bindless.Buffer, pc.MeshletSlot)[candidate.Meshlet];
            const Transform world = scene.Models(pc.ModelSlot)[instance_slot];
            const MeshletBounds bounds = ResolveMeshletBounds(scene, pc, candidate, instance_slot, instance, meshlet, world);
            visible = !bounds.Valid ||
                !MeshletOccluded(scene, pc.PyramidSamplerSlot, scene.ViewProj(), bounds.Center, bounds.Ax, bounds.Ay, bounds.Az);
        }
    }
    const uint present = visible ? 1u : 0u;
    const uint rank = simd_prefix_exclusive_sum(present);
    if (pc.Phase2Emit == 0u) {
        const uint count = simd_sum(present);
        if (lane == 0u) BindlessBufferMutable(uint, bindless.Buffer, pc.Phase2BlockCountSlot)[block_id] = count;
    } else if (visible) {
        const uint offset = BindlessBuffer(uint, bindless.Buffer, pc.Phase2BlockCountSlot)[block_id];
        BindlessBufferMutable(VisibleMeshlet, bindless.Buffer, pc.Phase2VisibleSlot)[offset + rank] = candidate;
    }
}

kernel void MeshletPhase2RangeCull(
    uint lane [[thread_index_in_threadgroup]], uint range_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWorkState work = BindlessBuffer(MeshletWorkState, bindless.Buffer, pc.WorkStateSlot)[0];
    if (range_id >= work.Phase2RangeCount) return;
    device MeshletWorkRange *ranges = BindlessBufferMutable(MeshletWorkRange, bindless.Buffer, pc.Phase2RangeCandidateSlot);
    const MeshletWorkRange range = ranges[range_id];
    const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[range.Instance];
    // Every lane evaluates the whole-instance test identically, so no broadcast is needed.
    bool range_visible = instance_slot != INVALID_OFFSET;
    InstanceRecord instance{};
    if (range_visible) {
        instance = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[instance_slot];
        const OrientedBounds instance_bounds = InstanceBounds(scene, pc, instance_slot);
        range_visible = !instance_bounds.Valid || !MeshletOccluded(
            scene, pc.PyramidSamplerSlot, scene.ViewProj(),
            instance_bounds.Center, instance_bounds.Ax, instance_bounds.Ay, instance_bounds.Az
        );
    }
    if (!range_visible) {
        // The count pass claims the range's slots even when the whole range stays occluded.
        if (pc.Phase2Emit == 0u && lane == 0u) ranges[range_id].WorkOffset = 0u;
        return;
    }
    // The count pass accumulates the range's survivors into its WorkOffset, which the prefix turns
    // into the emit pass's base offset.
    uint total = 0u;
    uint output = range.WorkOffset;
    for (uint base = 0u; base < range.MeshletCount; base += Phase2GroupSize) {
        const uint m = base + lane;
        const bool visible = m < range.MeshletCount &&
            Phase2ExpandedMeshletVisible(scene, pc, {range.Instance, range.MeshletOffset + m}, instance_slot, instance);
        const uint present = visible ? 1u : 0u;
        const uint rank = simd_prefix_exclusive_sum(present);
        if (pc.Phase2Emit != 0u && visible) {
            BindlessBufferMutable(VisibleMeshlet, bindless.Buffer, pc.Phase2VisibleSlot)[output + rank] = {
                range.Instance, range.MeshletOffset + m
            };
        }
        const uint iteration = simd_sum(present);
        total += iteration;
        output += iteration;
    }
    if (pc.Phase2Emit == 0u && lane == 0u) ranges[range_id].WorkOffset = total;
}

kernel void MeshletPhase2Prefix(
    uint lane [[thread_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (lane != 0u) return;
    device const MeshletRouteState *routes = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot);
    const uint block_count = (routes->Counts[4] + Phase2GroupSize - 1u) / Phase2GroupSize;
    device uint *blocks = BindlessBufferMutable(uint, bindless.Buffer, pc.Phase2BlockCountSlot);
    uint total = 0u;
    for (uint block = 0u; block < block_count; ++block) {
        const uint count = blocks[block];
        blocks[block] = total;
        total += count;
    }
    device MeshletWorkRange *ranges = BindlessBufferMutable(MeshletWorkRange, bindless.Buffer, pc.Phase2RangeCandidateSlot);
    const uint range_count = BindlessBuffer(MeshletWorkState, bindless.Buffer, pc.WorkStateSlot)[0].Phase2RangeCount;
    for (uint range = 0u; range < range_count; ++range) {
        const uint count = ranges[range].WorkOffset;
        ranges[range].WorkOffset = total;
        total += count;
    }
    device MeshletRouteState *phase2 = BindlessBufferMutable(MeshletRouteState, bindless.Buffer, pc.Phase2RouteStateSlot);
    phase2->Counts[0] = total;
    phase2->Offsets[0] = 0u;
    // Phase 2 draws route 0 alone, so its args buffer holds that route's chunks only.
    device MeshDispatchArgs *args = BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.Phase2DispatchArgsSlot);
    for (uint chunk = 0u; chunk < pc.DispatchChunkCount; ++chunk) {
        const uint begin = chunk * pc.DispatchChunkSize;
        args[chunk] = {total > begin ? min(total - begin, pc.DispatchChunkSize) : 0u, 1u, 1u};
    }
}
