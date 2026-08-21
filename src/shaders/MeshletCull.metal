#include "AABB.metal"
#include "Bindless.metal"
#include "Frustum.metal"
#include "InstanceRecord.metal"
#include "MaterialAlphaMode.metal"
#include "MeshDispatchArgs.metal"
#include "MeshletBlendBlockState.metal"
#include "MeshletCullPushConstants.metal"
#include "MeshletCullBlockState.metal"
#include "MeshletRecord.metal"
#include "MeshletRouteState.metal"
#include "PrimitiveRecord.metal"
#include "ScreenSpace.metal"
#include "VisibleMeshlet.metal"

constant uint CullBlockSize = 1024u;
constant uint CullRouteCount = 4u;
constant uint CullSimdGroups = 32u;

struct RoutedMeshlet {
    VisibleMeshlet Work;
    uint Routes;
    uint BlendBucket;
};

inline bool MeshletOccluded(const thread Scene &scene, uint pyramid_slot, float3 center, float3 ax, float3 ay, float3 az) {
    float2 uv_min = float2(1e30f), uv_max = float2(-1e30f);
    float min_depth = 1e30f;
    for (uint c = 0; c < 8; ++c) {
        const float3 corner = center + ((c & 1u) ? ax : -ax) + ((c & 2u) ? ay : -ay) + ((c & 4u) ? az : -az);
        const float4 clip = scene.ViewProj() * float4(corner, 1.0f);
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

inline RoutedMeshlet ClassifyMeshlet(
    const thread Scene &scene, MeshletCullPushConstants pc, VisibleMeshlet candidate,
    uint instance_slot, InstanceRecord instance
) {
    RoutedMeshlet result{{INVALID_OFFSET, INVALID_OFFSET}, 0u, 0u};
    if ((instance.Flags & pc.RequiredInstanceFlags) != pc.RequiredInstanceFlags) return result;
    const MeshletRecord meshlet = BindlessBuffer(MeshletRecord, scene.B.Buffer, pc.MeshletSlot)[candidate.Meshlet];
    const Transform world = scene.Models(pc.ModelSlot)[instance_slot];
    bool visible;
    bool can_occlude = true;
    float3 world_center;
    float3 ax, ay, az;
    const bool deformed = instance.ArmatureDeformOffset != INVALID_OFFSET || instance.MorphDeformOffset != INVALID_OFFSET || instance.PosedPositionOffset != INVALID_OFFSET;
    if (deformed) {
        const AABB bounds = BindlessBuffer(AABB, scene.B.Buffer, pc.BoundsSlot)[instance_slot];
        const float3 lo = float3(bounds.Min), hi = float3(bounds.Max);
        if (lo.x > hi.x) {
            world_center = float3(world.P);
            visible = true;
            can_occlude = false;
        } else {
            const float3 half_local = (hi - lo) * 0.5f;
            const float3 scale = float3(world.S);
            const float4 rotation = float4(world.R);
            world_center = trs_transform_point(world, (lo + hi) * 0.5f);
            ax = quat_rotate(rotation, float3(scale.x * half_local.x, 0, 0));
            ay = quat_rotate(rotation, float3(0, scale.y * half_local.y, 0));
            az = quat_rotate(rotation, float3(0, 0, scale.z * half_local.z));
            visible = in_frustum(scene.ViewProj(), world_center, ax, ay, az);
        }
    } else {
        const float3 scale = abs(float3(world.S));
        const float radius = meshlet.Radius * max(scale.x, max(scale.y, scale.z));
        ax = float3(radius, 0, 0);
        ay = float3(0, radius, 0);
        az = float3(0, 0, radius);
        world_center = trs_transform_point(world, float3(meshlet.Center));
        visible = sphere_in_frustum(scene.ViewProj(), world_center, radius);
    }
    if (visible && can_occlude && pc.PyramidSamplerSlot != INVALID_SLOT) visible = !MeshletOccluded(scene, pc.PyramidSamplerSlot, world_center, ax, ay, az);
    if (!visible) return result;

    result.Work = candidate;
    if (pc.RouteMode == 0u) {
        result.Routes = 1u;
    } else {
        const PrimitiveRecord primitive = BindlessBuffer(PrimitiveRecord, scene.B.Buffer, pc.PrimitiveSlot)[meshlet.Primitive];
        uint material_index = 0u;
        if (primitive.Draw.PrimitiveMaterialOffset != INVALID_OFFSET) {
            material_index = scene.PrimitiveMaterials(scene.View.PrimitiveMaterialSlot)[primitive.Draw.PrimitiveMaterialOffset + primitive.PrimitiveIndex];
        }
        const PBRMaterial material = scene.Materials(scene.View.MaterialSlot)[material_index];
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
    return result;
}

kernel void MeshletCullBlockCount(
    uint lane [[thread_index_in_threadgroup]],
    uint block_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *group_prefixes [[threadgroup(0)]]
) {
    if (block_id >= pc.BlockCount) return;
    device const VisibleMeshlet *candidates = BindlessBuffer(VisibleMeshlet, bindless.Buffer, pc.CandidateSlot);
    device MeshletCullBlockState *blocks = BindlessBufferMutable(MeshletCullBlockState, bindless.Buffer, pc.BlockStateSlot);
    const bool sort_blend = pc.BlendBlockSlot != INVALID_SLOT;
    threadgroup ushort *group_blend_counts = reinterpret_cast<threadgroup ushort *>(group_prefixes + CullRouteCount * (CullSimdGroups + 1u));
    if (sort_blend) {
        for (uint j = lane; j < CullSimdGroups * 256u; j += CullBlockSize) group_blend_counts[j] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint i = block_id * CullBlockSize + lane;
    uint routes = 0u;
    uint blend_bucket = 0u;
    if (i < pc.CandidateCount) {
        const VisibleMeshlet work = candidates[i];
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
    }
    uint present[CullRouteCount], rank[CullRouteCount];
    for (uint route = 0; route < CullRouteCount; ++route) {
        present[route] = (routes >> route) & 1u;
        rank[route] = simd_prefix_exclusive_sum(present[route]);
        const uint count = simd_sum(present[route]);
        if (simd_lane == 0u) group_prefixes[route * CullSimdGroups + simd_group] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u) {
        for (uint route = 0; route < CullRouteCount; ++route) {
            const uint count = simd_lane < CullSimdGroups ? group_prefixes[route * CullSimdGroups + simd_lane] : 0u;
            const uint total = simd_sum(count);
            const uint prefix = simd_prefix_exclusive_sum(count);
            if (simd_lane < CullSimdGroups) group_prefixes[route * CullSimdGroups + simd_lane] = prefix;
            if (simd_lane == 0u) {
                group_prefixes[CullRouteCount * CullSimdGroups + route] = total;
                blocks[block_id].Routes[route] = total;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sort_blend) {
        uint blend_rank = 0u;
        uint blend_count = 0u;
        if (present[1] != 0u) {
            for (uint source = 0u; source < 32u; ++source) {
                const bool match = simd_shuffle(present[1], source) != 0u && simd_shuffle(blend_bucket, source) == blend_bucket;
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
        if (present[1] != 0u) {
            for (uint group = 0u; group < simd_group; ++group) blend_rank += group_blend_counts[group * 256u + blend_bucket];
            rank[1] = blend_rank;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < pc.CandidateCount) {
        // Four route bits followed by one ten-bit rank per active non-transmission route.
        uint key = routes;
        uint field = 0u;
        for (uint route = 0; route < CullRouteCount; ++route) {
            if (!sort_blend || route != 1u) rank[route] += group_prefixes[route * CullSimdGroups + simd_group];
            if (route == 2u) rank[route] = group_prefixes[CullRouteCount * CullSimdGroups + route] - rank[route] - present[route];
            if (present[route] == 0u) continue;
            if (route == 2u) BindlessBufferMutable(ushort, bindless.Buffer, pc.TransmissionRankSlot)[i] = ushort(rank[route]);
            else key |= rank[route] << (4u + 10u * field++);
        }
        // Alpha uses at most two routes, leaving the next byte for its depth bucket.
        if (sort_blend && (routes & 2u) != 0u) key |= blend_bucket << (4u + 10u * field);
        BindlessBufferMutable(uint, bindless.Buffer, pc.RouteKeySlot)[i] = key;
    }
}

kernel void MeshletCullPrefix(
    uint lane [[thread_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *route_totals [[threadgroup(0)]]
) {
    device MeshletCullBlockState *blocks = BindlessBufferMutable(MeshletCullBlockState, bindless.Buffer, pc.BlockStateSlot);
    device MeshletRouteState *state = BindlessBufferMutable(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot);
    device MeshDispatchArgs *args = BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.DispatchArgsSlot);
    const bool sort_blend = pc.BlendBlockSlot != INVALID_SLOT;
    if (lane < CullRouteCount) {
        uint total = 0u;
        if (lane != 2u) {
            for (uint block = 0; block < pc.BlockCount; ++block) {
                const uint count = blocks[block].Routes[lane];
                blocks[block].Routes[lane] = total;
                total += count;
            }
        } else {
            for (uint block = pc.BlockCount; block-- > 0u;) {
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
        for (uint block = 0u; block < pc.BlockCount; ++block) {
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
        for (uint route = 0; route < CullRouteCount; ++route) {
            state->Counts[route] = route_totals[route];
            state->Offsets[route] = route_offset;
            route_offset += route_totals[route];
            for (uint chunk = 0u; chunk < pc.DispatchChunkCount; ++chunk) {
                const uint begin = chunk * pc.DispatchChunkSize;
                const uint count = route_totals[route] > begin ? min(route_totals[route] - begin, pc.DispatchChunkSize) : 0u;
                args[route * pc.DispatchChunkCount + chunk] = {count, 1u, 1u};
            }
        }
    }
}

kernel void MeshletCullEmit(
    uint lane [[thread_index_in_threadgroup]],
    uint block_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (block_id >= pc.BlockCount) return;
    device const VisibleMeshlet *candidates = BindlessBuffer(VisibleMeshlet, bindless.Buffer, pc.CandidateSlot);
    device MeshletCullBlockState *blocks = BindlessBufferMutable(MeshletCullBlockState, bindless.Buffer, pc.BlockStateSlot);
    const uint i = block_id * CullBlockSize + lane;
    if (i >= pc.CandidateCount) return;
    device const MeshletRouteState *state = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot);
    device VisibleMeshlet *visible = BindlessBufferMutable(VisibleMeshlet, bindless.Buffer, pc.VisibleSlot);
    const bool sort_blend = pc.BlendBlockSlot != INVALID_SLOT;
    const uint key = BindlessBuffer(uint, bindless.Buffer, pc.RouteKeySlot)[i];
    const uint routes = key & 15u;
    const uint blend_bucket = sort_blend && (routes & 2u) != 0u ? (key >> (4u + 10u * popcount(routes & ~(1u << 2u)))) & 255u : 0u;
    uint field = 0u;
    for (uint route = 0; route < CullRouteCount; ++route) {
        if ((routes & (1u << route)) != 0u) {
            const uint rank = route == 2u ?
                BindlessBuffer(ushort, bindless.Buffer, pc.TransmissionRankSlot)[i] :
                (key >> (4u + 10u * field++)) & 1023u;
            uint output = state->Offsets[route] + blocks[block_id].Routes[route] + rank;
            if (sort_blend && route == 1u) {
                device const MeshletBlendBlockState *blend_blocks = BindlessBuffer(MeshletBlendBlockState, bindless.Buffer, pc.BlendBlockSlot);
                output = state->Offsets[1] + state->BlendOffsets[blend_bucket] + blend_blocks[block_id].Buckets[blend_bucket] + rank;
            }
            visible[output] = candidates[i];
        }
    }
}
