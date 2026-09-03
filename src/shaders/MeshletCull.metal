#include "AABB.metal"
#include "TransformUtils.metal"
#include "Bindless.metal"
#include "ClusterGroup.metal"
#include "Frustum.metal"
#include "InstanceRecord.metal"
#include "MeshletInstanceFlag.metal"
#include "MaterialAlphaMode.metal"
#include "LodFrontierBlockState.metal"
#include "LodFrontierEntry.metal"
#include "LodFrontierState.metal"
#include "LodNode.metal"
#include "MeshDispatchArgs.metal"
#include "MeshletBlendBlockState.metal"
#include "MeshletCullBlockState.metal"
#include "MeshletCullPushConstants.metal"
#include "MeshletRecord.metal"
#include "MeshletGeometryEncoding.metal"
#include "MeshPrimitiveTopology.metal"
#include "MeshletRoute.metal"
#include "MeshletRouteState.metal"
#include "MeshletWorkRange.metal"
#include "MeshletWorkState.metal"
#include "PrimitiveRecord.metal"
#include "ScreenSpace.metal"
#include "VisibleMeshlet.metal"

constant uint CullBlockSize = 1024u;
constant uint CullRouteCount = MeshletRoute_Count;
constant uint CullSimdGroups = 32u;
constant uint PrefixStride = CullSimdGroups + 1u;
constant uint ConeCullMinTriangles = 16u;
// The phase-2 cull kernels run one 32-lane simdgroup per threadgroup.
constant uint Phase2GroupSize = 32u;

struct RoutedMeshlet {
    uint Routes;
    uint BlendBucket;
    bool Coarse;
};

inline uint RouteBit(uint route) { return 1u << route; }

inline uint PrimitiveTopology(MeshletRecord meshlet) {
    return meshlet.LocalTriangleOffset >> MeshletGeometryEncoding_TopologyShift;
}

inline uint OpaqueVisibilityRoute(PBRMaterial material, Transform world) {
    if (material.DoubleSided != 0u) return MeshletRoute_OpaqueDoubleSided;
    const float3 scale = float3(world.S);
    return scale.x * scale.y * scale.z < 0.0f ? MeshletRoute_OpaqueCullFront : MeshletRoute_OpaqueCullBack;
}

inline bool InstanceDeformed(InstanceRecord instance) {
    return instance.ArmatureDeformOffset != INVALID_OFFSET || instance.MorphDeformOffset != INVALID_OFFSET ||
        instance.PosedPositionOffset != INVALID_OFFSET || instance.HasPendingVertexTransform != 0u;
}

// Edited and deformed instances use original geometry covered by their posed bounds.
inline bool InstancePinsFinest(InstanceRecord instance) {
    return (instance.Flags & (MeshletInstanceFlag_LodPinFinest | MeshletInstanceFlag_Wire |
        MeshletInstanceFlag_FaceNormal | MeshletInstanceFlag_VertexNormal |
        MeshletInstanceFlag_EdgeOverlay | MeshletInstanceFlag_PointOverlay |
        MeshletInstanceFlag_SoundPoint)) != 0u ||
        InstanceDeformed(instance);
}

// Returns the primitive's full meshlet range or its original-geometry prefix.
inline uint PrimitiveWorkCount(PrimitiveRecord primitive, bool finest_only) {
    return finest_only ? primitive.Level0Count : primitive.MeshletCount;
}

// Nonpositive thresholds select original geometry.
inline bool InstanceFinestOnly(const thread Scene &scene, InstanceRecord instance) {
    return scene.View.LodErrorPixels <= 0.0f || InstancePinsFinest(instance);
}

// Returns a cluster group's projected simplification error in pixels.
inline float LodGroupErrorPixels(const thread Scene &scene, ClusterGroup group, Transform world) {
    const float3 scale = abs(float3(world.S));
    const float max_scale = max(scale.x, max(scale.y, scale.z));
    const float error = group.Error * max_scale;
    if (scene.View.ScreenPixelScale <= 0.0f) return error / -scene.View.ScreenPixelScale;
    const float3 center = trs_transform_point(world, float3(group.Center));
    const float distance = max(
        length(center - float3(scene.View.CameraPosition)) - group.Radius * max_scale, scene.View.CameraNear
    );
    return error / (distance * scene.View.ScreenPixelScale);
}

// Selects the cluster when its group exceeds the error threshold and its refining group does not.
inline bool LodClusterVisible(
    const thread Scene &scene, MeshletCullPushConstants pc, MeshletRecord meshlet, Transform world, bool finest_only
) {
    if (meshlet.GroupIndex == INVALID_OFFSET) return true;
    if (finest_only) return meshlet.RefinedGroup == INVALID_OFFSET;
    device const ClusterGroup *groups = BindlessBuffer(ClusterGroup, scene.B.Buffer, pc.ClusterGroupSlot);
    if (LodGroupErrorPixels(scene, groups[meshlet.GroupIndex], world) <= scene.View.LodErrorPixels) return false;
    return meshlet.RefinedGroup == INVALID_OFFSET ||
        LodGroupErrorPixels(scene, groups[meshlet.RefinedGroup], world) <= scene.View.LodErrorPixels;
}

inline uint PrimitiveMaterialIndex(const thread Scene &scene, PrimitiveRecord primitive) {
    if (primitive.Draw.PrimitiveMaterialOffset == INVALID_OFFSET) return 0u;
    return scene.PrimitiveMaterials(scene.View.PrimitiveMaterialSlot)[
        primitive.Draw.PrimitiveMaterialOffset + primitive.PrimitiveIndex
    ];
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

inline OrientedBounds InstanceBounds(const thread Scene &scene, MeshletCullPushConstants pc, uint instance_slot) {
    const AABB bounds = BindlessBuffer(AABB, scene.B.Buffer, pc.BoundsSlot)[instance_slot];
    return TransformBounds(bounds, scene.Models(pc.ModelSlot)[instance_slot]);
}

// Posed bounds cover original clusters in primitive order.
inline OrientedBounds DeformedMeshletBounds(
    const thread Scene &scene, MeshletCullPushConstants pc, VisibleMeshlet candidate,
    InstanceRecord instance, MeshletRecord meshlet, Transform world
) {
    if (instance.PosedMeshletBoundsOffset == INVALID_OFFSET || pc.PosedMeshletBoundsSlot == INVALID_SLOT) return {};
    device const PrimitiveRecord *primitives = BindlessBuffer(PrimitiveRecord, scene.B.Buffer, pc.PrimitiveSlot);
    uint local_meshlet = candidate.Meshlet - primitives[meshlet.Primitive].MeshletOffset;
    for (uint p = instance.PrimitiveOffset; p < meshlet.Primitive; ++p) local_meshlet += primitives[p].Level0Count;
    const AABB bounds = BindlessBuffer(AABB, scene.B.Buffer, pc.PosedMeshletBoundsSlot)
        [instance.PosedMeshletBoundsOffset + local_meshlet];
    return TransformBounds(bounds, world);
}

// Returns a posed OBB for deformed instances and a scaled sphere for static instances.
// Invalid bounds require visible, unoccludable treatment.
struct MeshletBounds {
    float3 Center;
    float3 Ax, Ay, Az;
    float Radius;
    bool Sphere;
    bool Valid;
};

inline MeshletBounds ResolveMeshletBounds(
    const thread Scene &scene, MeshletCullPushConstants pc, VisibleMeshlet candidate,
    uint instance_slot, InstanceRecord instance, MeshletRecord meshlet, Transform world
) {
    if (InstanceDeformed(instance)) {
        OrientedBounds bounds = DeformedMeshletBounds(scene, pc, candidate, instance, meshlet, world);
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

// Returns 0 to reject, 1 to expand in phase 1, or 2 to defer the complete instance to the current-pyramid phase.
inline uint ClassifyInstanceRange(
    const thread Scene &scene, MeshletCullPushConstants pc, uint instance_slot, InstanceRecord instance
) {
    if (instance.PrimitiveCount == 0u || (instance.Flags & pc.RequiredInstanceFlags) != pc.RequiredInstanceFlags) return 0u;
    // Posed meshlet bounds supersede the instance AABB, which may represent another motion-blur step.
    if (instance.PosedMeshletBoundsOffset != INVALID_OFFSET) return 1u;
    const OrientedBounds bounds = InstanceBounds(scene, pc, instance_slot);
    if (!bounds.Valid) return 1u;
    if (!in_frustum(scene.ViewProj(), bounds.Center, bounds.Ax, bounds.Ay, bounds.Az)) return 0u;
    if ((instance.Flags & MeshletInstanceFlag_OverlayOnly) != 0u) return 1u;
    if (pc.PyramidSamplerSlot == INVALID_SLOT ||
        !MeshletOccluded(scene, pc.PyramidSamplerSlot, pc.OcclusionViewProj.Unpack(), bounds.Center, bounds.Ax, bounds.Ay, bounds.Az)) return 1u;
    // Keep blend routes in phase 1 while opaque meshlets defer independently.
    if (pc.TwoPhase != 0u && pc.BlendBlockSlot != INVALID_SLOT) return 1u;
    return pc.TwoPhase != 0u ? 2u : 0u;
}

inline RoutedMeshlet ClassifyMeshlet(
    const thread Scene &scene, MeshletCullPushConstants pc, VisibleMeshlet candidate,
    uint instance_slot, InstanceRecord instance
) {
    RoutedMeshlet result{0u, 0u, false};
    const MeshletRecord meshlet = BindlessBuffer(MeshletRecord, scene.B.Buffer, pc.MeshletSlot)[candidate.Meshlet];
    const Transform world = scene.Models(pc.ModelSlot)[instance_slot];
    if (!LodClusterVisible(scene, pc, meshlet, world, InstanceFinestOnly(scene, instance))) return result;
    result.Coarse = meshlet.RefinedGroup != INVALID_OFFSET;
    const MeshletBounds bounds = ResolveMeshletBounds(scene, pc, candidate, instance_slot, instance, meshlet, world);
    if (bounds.Valid && !MeshletBoundsInFrustum(scene, bounds)) return result;
    const float3 world_center = bounds.Valid ? bounds.Center : float3(world.P);

    const PrimitiveRecord primitive = BindlessBuffer(PrimitiveRecord, scene.B.Buffer, pc.PrimitiveSlot)[meshlet.Primitive];
    const bool triangle_topology = PrimitiveTopology(meshlet) == MeshPrimitiveTopology_Triangle;
    // A one-meshlet instance already passed the conservative instance query.
    const bool can_occlude = bounds.Valid && !(instance.PrimitiveCount == 1u && primitive.MeshletCount == 1u);
    PBRMaterial material{};
    if (pc.RouteMode != 0u) material = scene.Materials(scene.View.MaterialSlot)[PrimitiveMaterialIndex(scene, primitive)];
    const bool edit_overlay = (instance.Flags & MeshletInstanceFlag_EditOverlay) != 0u;
    const bool overlay_only = (instance.Flags & MeshletInstanceFlag_OverlayOnly) != 0u;
    const bool cone_visible = pc.RouteMode == 0u || material.DoubleSided != 0u ||
        MeshletConeVisible(scene, meshlet, world, InstanceDeformed(instance));
    const bool occluded = !overlay_only && can_occlude && pc.PyramidSamplerSlot != INVALID_SLOT &&
        MeshletOccluded(scene, pc.PyramidSamplerSlot, pc.OcclusionViewProj.Unpack(), world_center, bounds.Ax, bounds.Ay, bounds.Az);

    if (overlay_only) {
        result.Routes = 0u;
    } else if (pc.RouteMode == 3u && !triangle_topology) {
        result.Routes = 0u;
    } else if (pc.RouteMode == 0u) {
        result.Routes = RouteBit(MeshletRoute_OpaqueCullBack);
    } else {
        const bool alpha_mask = material.AlphaMode == MaterialAlphaMode_Mask;
        const uint opaque_route = triangle_topology ? OpaqueVisibilityRoute(material, world) : MeshletRoute_Coverage;
        if (pc.RouteMode == 3u) {
            result.Routes = RouteBit(alpha_mask ? MeshletRoute_Coverage : opaque_route);
        } else if (material.AlphaMode == MaterialAlphaMode_Blend) {
            result.Routes = RouteBit(MeshletRoute_Blend);
            const float4 clip = scene.ViewProj() * float4(world_center, 1.0f);
            result.BlendBucket = clip.w > 0.0f ? uint(clamp(clip.z / clip.w, 0.0f, 1.0f) * 255.0f) : 0u;
        } else if (pc.RouteMode == 1u) {
            result.Routes = RouteBit(alpha_mask ? MeshletRoute_Coverage : opaque_route);
        } else {
            const bool transmissive = material.Transmission.Factor > 0.0f;
            if (!transmissive) {
                result.Routes = RouteBit(alpha_mask ? MeshletRoute_Coverage : opaque_route);
            } else {
                if (material.Transmission.Texture.Slot != INVALID_SLOT) result.Routes |= RouteBit(MeshletRoute_Coverage);
                result.Routes |= RouteBit(MeshletRoute_Transmission);
            }
        }
    }
    if (!cone_visible) result.Routes = 0u;
    if (edit_overlay) result.Routes |= RouteBit(MeshletRoute_EditOverlay);
    if ((instance.Flags & MeshletInstanceFlag_Wire) != 0u) result.Routes |= RouteBit(MeshletRoute_Wire);
    if ((instance.Flags & (MeshletInstanceFlag_Bone | MeshletInstanceFlag_BoneJoint |
        MeshletInstanceFlag_FaceNormal | MeshletInstanceFlag_VertexNormal |
        MeshletInstanceFlag_EdgeOverlay | MeshletInstanceFlag_PointOverlay |
        MeshletInstanceFlag_SoundPoint)) != 0u) {
        result.Routes |= RouteBit(MeshletRoute_Overlay);
    }
    result.Routes &= pc.RouteMask;
    if (result.Routes == 0u) return result;
    if (occluded) {
        if (pc.TwoPhase != 0u) {
            // Defer discard-free opaque routes; keep coverage and blend routes in phase 1.
            const uint fast = RouteBit(MeshletRoute_OpaqueCullBack) | RouteBit(MeshletRoute_OpaqueCullFront) |
                RouteBit(MeshletRoute_OpaqueDoubleSided) | RouteBit(MeshletRoute_EditOverlay) |
                RouteBit(MeshletRoute_Overlay);
            const uint keep = RouteBit(MeshletRoute_Blend) | RouteBit(MeshletRoute_Coverage) |
                RouteBit(MeshletRoute_Wire);
            result.Routes = (result.Routes & keep) |
                ((result.Routes & fast) != 0u ? RouteBit(MeshletRoute_Phase2Candidate) : 0u);
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

// Node error and bounds conservatively cover every record in the span, so pruning preserves classification results.
inline bool LodNodeVisible(const thread Scene &scene, LodNode node, Transform world) {
    // Infinite error disables span pruning because the associated bounds are undefined.
    if (isinf(node.Error)) return true;
    const ClusterGroup bound{node.Center, node.Radius, node.Error};
    if (LodGroupErrorPixels(scene, bound, world) <= scene.View.LodErrorPixels) return false;
    const float3 scale = abs(float3(world.S));
    const float radius = node.Radius * max(scale.x, max(scale.y, scale.z));
    return sphere_in_frustum(scene.ViewProj(), trs_transform_point(world, float3(node.Center)), radius);
}

// Stores one entry's child nodes, final-level record range, and phase-2 deferred instance range.
struct LodWork {
    uint Instance;
    uint Node;
    uint ChildCount;
    uint MeshletCount;
    uint Phase2RangeCount;
};

// Seeds traversal from instance IDs and writes primitive roots or complete phase-2 deferred ranges.
inline LodWork ResolveLodSeed(const thread Scene &scene, MeshletCullPushConstants pc, uint id) {
    if (id >= pc.InstanceCount) return {};
    const uint instance_slot = BindlessBuffer(uint, scene.B.Buffer, pc.InstanceMapSlot)[id];
    if (instance_slot == INVALID_OFFSET) return {};
    const InstanceRecord instance = BindlessBuffer(InstanceRecord, scene.B.Buffer, pc.InstanceSlot)[instance_slot];
    const uint visibility = ClassifyInstanceRange(scene, pc, instance_slot, instance);
    if (visibility == 0u) return {};
    const bool finest_only = InstanceFinestOnly(scene, instance);
    device const PrimitiveRecord *primitives = BindlessBuffer(PrimitiveRecord, scene.B.Buffer, pc.PrimitiveSlot);
    uint count = 0u;
    for (uint p = 0u; p < instance.PrimitiveCount; ++p) {
        count += PrimitiveWorkCount(primitives[instance.PrimitiveOffset + p], finest_only) != 0u;
    }
    if (visibility == 2u) return {id, INVALID_OFFSET, 0u, 0u, count};
    return {id, INVALID_OFFSET, count, 0u, 0u};
}

// Expands one frontier node into child nodes or its final-level record range.
inline LodWork ResolveLodNode(const thread Scene &scene, MeshletCullPushConstants pc, uint index) {
    device const LodFrontierState *states = BindlessBuffer(LodFrontierState, scene.B.Buffer, pc.LodFrontierStateSlot);
    if (index >= states[pc.LodFrontierIndex].NodeCount) return {};
    const LodFrontierEntry entry = BindlessBuffer(LodFrontierEntry, scene.B.Buffer, pc.LodFrontierSlot)[index];
    const uint instance_slot = BindlessBuffer(uint, scene.B.Buffer, pc.InstanceMapSlot)[entry.Instance];
    if (instance_slot == INVALID_OFFSET) return {};
    const LodNode node = BindlessBuffer(LodNode, scene.B.Buffer, pc.LodNodeSlot)[entry.Node];
    if (!LodNodeVisible(scene, node, scene.Models(pc.ModelSlot)[instance_slot])) return {};
    // The final level emits the complete range of nodes deeper than the recorded depth.
    if (pc.LodFinalLevel != 0u) return {entry.Instance, entry.Node, 1u, node.MeshletCount, 0u};
    // Repeat leaves through later levels to preserve frontier order.
    return {entry.Instance, entry.Node, max(node.ChildCount, 1u), 0u, 0u};
}

inline LodWork ResolveLodWork(const thread Scene &scene, MeshletCullPushConstants pc, uint index) {
    return pc.LodSeedLevel != 0u ? ResolveLodSeed(scene, pc, index) : ResolveLodNode(scene, pc, index);
}

// Writes each simdgroup's three totals into the corresponding prefix-row lane.
inline void WriteLodSimdGroupSums(
    threadgroup uint *group_prefixes, LodWork work, uint simd_lane, uint simd_group
) {
    const uint node_sum = simd_sum(work.ChildCount);
    const uint meshlet_sum = simd_sum(work.MeshletCount);
    const uint phase2_range_sum = simd_sum(work.Phase2RangeCount);
    if (simd_lane == 0u) {
        group_prefixes[simd_group] = node_sum;
        group_prefixes[PrefixStride + simd_group] = meshlet_sum;
        group_prefixes[2u * PrefixStride + simd_group] = phase2_range_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void LodFrontierCount(
    uint lane [[thread_index_in_threadgroup]], uint block_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *group_prefixes [[threadgroup(0)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const LodWork work = ResolveLodWork(scene, pc, block_id * CullBlockSize + lane);
    WriteLodSimdGroupSums(group_prefixes, work, simd_lane, simd_group);
    if (simd_group == 0u && simd_lane == 0u) {
        uint nodes = 0u, meshlets = 0u, phase2_ranges = 0u;
        for (uint group = 0u; group < CullSimdGroups; ++group) {
            nodes += group_prefixes[group];
            meshlets += group_prefixes[PrefixStride + group];
            phase2_ranges += group_prefixes[2u * PrefixStride + group];
        }
        BindlessBufferMutable(LodFrontierBlockState, bindless.Buffer, pc.LodFrontierBlockStateSlot)[block_id] = {
            nodes, meshlets, phase2_ranges
        };
    }
}

kernel void LodFrontierPrefix(
    uint lane [[thread_index_in_threadgroup]], device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (lane != 0u) return;
    device LodFrontierState *states = BindlessBufferMutable(LodFrontierState, bindless.Buffer, pc.LodFrontierStateSlot);
    const uint block_count = pc.LodSeedLevel != 0u ? pc.WorkBlockCount : states[pc.LodFrontierIndex].BlockCount;
    device LodFrontierBlockState *blocks = BindlessBufferMutable(LodFrontierBlockState, bindless.Buffer, pc.LodFrontierBlockStateSlot);
    uint node_count = 0u, meshlet_count = 0u, phase2_range_count = 0u;
    for (uint block = 0u; block < block_count; ++block) {
        const LodFrontierBlockState count = blocks[block];
        blocks[block] = {node_count, meshlet_count, phase2_range_count};
        node_count += count.NodeCount;
        meshlet_count += count.MeshletCount;
        phase2_range_count += count.Phase2RangeCount;
    }
    const uint next_block_count = (node_count + CullBlockSize - 1u) / CullBlockSize;
    states[pc.LodFrontierIndex ^ 1u] = {node_count, next_block_count};
    BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.LodExpandArgsSlot)[pc.LodFrontierIndex ^ 1u] = {
        next_block_count, 1u, 1u
    };
    device MeshletWorkState *state = BindlessBufferMutable(MeshletWorkState, bindless.Buffer, pc.WorkStateSlot);
    if (pc.LodSeedLevel != 0u) {
        state[0] = {0u, 0u, 0u, phase2_range_count};
        if (pc.CoarseCountSlot != INVALID_SLOT) BindlessBufferMutable(uint, bindless.Buffer, pc.CoarseCountSlot)[0] = 0u;
        if (pc.TwoPhase != 0u) {
            BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.Phase2RangeCullArgsSlot)[0] = {
                phase2_range_count, 1u, 1u
            };
        }
    }
    if (pc.LodFinalLevel != 0u) {
        const uint cull_block_count = (meshlet_count + CullBlockSize - 1u) / CullBlockSize;
        state[0].RangeCount = node_count;
        state[0].MeshletCount = meshlet_count;
        state[0].CullBlockCount = cull_block_count;
        BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.WorkDispatchArgsSlot)[0] = {cull_block_count, 1u, 1u};
    }
}

kernel void LodFrontierEmit(
    uint lane [[thread_index_in_threadgroup]], uint block_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    threadgroup uint *group_prefixes [[threadgroup(0)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const LodWork work = ResolveLodWork(scene, pc, block_id * CullBlockSize + lane);
    uint node_rank = simd_prefix_exclusive_sum(work.ChildCount);
    uint meshlet_rank = simd_prefix_exclusive_sum(work.MeshletCount);
    uint phase2_range_rank = simd_prefix_exclusive_sum(work.Phase2RangeCount);
    WriteLodSimdGroupSums(group_prefixes, work, simd_lane, simd_group);
    if (simd_group == 0u) {
        const uint nodes = simd_lane < CullSimdGroups ? group_prefixes[simd_lane] : 0u;
        const uint meshlets = simd_lane < CullSimdGroups ? group_prefixes[PrefixStride + simd_lane] : 0u;
        const uint phase2_ranges = simd_lane < CullSimdGroups ? group_prefixes[2u * PrefixStride + simd_lane] : 0u;
        if (simd_lane < CullSimdGroups) {
            group_prefixes[simd_lane] = simd_prefix_exclusive_sum(nodes);
            group_prefixes[PrefixStride + simd_lane] = simd_prefix_exclusive_sum(meshlets);
            group_prefixes[2u * PrefixStride + simd_lane] = simd_prefix_exclusive_sum(phase2_ranges);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const LodFrontierBlockState block = BindlessBuffer(LodFrontierBlockState, bindless.Buffer, pc.LodFrontierBlockStateSlot)[block_id];
    device const PrimitiveRecord *primitives = BindlessBuffer(PrimitiveRecord, bindless.Buffer, pc.PrimitiveSlot);
    if (work.Phase2RangeCount != 0u) {
        phase2_range_rank += group_prefixes[2u * PrefixStride + simd_group];
        uint output = block.Phase2RangeCount + phase2_range_rank;
        const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[work.Instance];
        const InstanceRecord instance = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[instance_slot];
        const bool finest_only = InstanceFinestOnly(scene, instance);
        device MeshletWorkRange *phase2_ranges = BindlessBufferMutable(
            MeshletWorkRange, bindless.Buffer, pc.Phase2RangeCandidateSlot
        );
        for (uint p = 0u; p < instance.PrimitiveCount; ++p) {
            const PrimitiveRecord primitive = primitives[instance.PrimitiveOffset + p];
            const uint count = PrimitiveWorkCount(primitive, finest_only);
            if (count == 0u) continue;
            phase2_ranges[output++] = {work.Instance, primitive.MeshletOffset, count, 0u};
        }
    }
    if (work.ChildCount == 0u) return;
    node_rank += group_prefixes[simd_group];
    uint output = block.NodeCount + node_rank;

    if (pc.LodFinalLevel != 0u) {
        meshlet_rank += group_prefixes[PrefixStride + simd_group];
        const uint work_offset = block.MeshletCount + meshlet_rank;
        const LodNode node = BindlessBuffer(LodNode, bindless.Buffer, pc.LodNodeSlot)[work.Node];
        BindlessBufferMutable(MeshletWorkRange, bindless.Buffer, pc.WorkRangeSlot)[output] = {
            work.Instance, node.FirstMeshlet, work.MeshletCount, work_offset
        };
        device uint *work_blocks = BindlessBufferMutable(uint, bindless.Buffer, pc.WorkBlockSlot);
        const uint first_block = (work_offset + CullBlockSize - 1u) / CullBlockSize;
        const uint last_block = (work_offset + work.MeshletCount - 1u) / CullBlockSize;
        for (uint b = first_block; b <= last_block; ++b) work_blocks[b] = output;
        return;
    }

    device LodFrontierEntry *next = BindlessBufferMutable(LodFrontierEntry, bindless.Buffer, pc.LodFrontierAltSlot);
    if (pc.LodSeedLevel != 0u) {
        const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[work.Instance];
        const InstanceRecord instance = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[instance_slot];
        const bool finest_only = InstanceFinestOnly(scene, instance);
        for (uint p = 0u; p < instance.PrimitiveCount; ++p) {
            const PrimitiveRecord primitive = primitives[instance.PrimitiveOffset + p];
            if (PrimitiveWorkCount(primitive, finest_only) == 0u) continue;
            next[output++] = {work.Instance, finest_only ? primitive.LodFinestNode : primitive.LodRootNode};
        }
        return;
    }
    const LodNode node = BindlessBuffer(LodNode, bindless.Buffer, pc.LodNodeSlot)[work.Node];
    // Repeat shallow leaves until the final level.
    if (node.ChildCount == 0u) {
        next[output] = {work.Instance, work.Node};
        return;
    }
    for (uint c = 0u; c < node.ChildCount; ++c) next[output + c] = {work.Instance, node.ChildOffset + c};
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
    uint routes = 0u, blend_bucket = 0u, coarse = 0u;
    if (work.Instance != INVALID_OFFSET) {
        const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[work.Instance];
        if (instance_slot != INVALID_OFFSET) {
            const InstanceRecord instance = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[instance_slot];
            const Scene scene{bindless, view, theme, workspace};
            const RoutedMeshlet routed = ClassifyMeshlet(scene, pc, work, instance_slot, instance);
            routes = routed.Routes;
            blend_bucket = routed.BlendBucket;
            coarse = routed.Routes != 0u && routed.Coarse ? 1u : 0u;
        }
    }
    // Accumulate one value per simdgroup because profiling records only the total.
    if (pc.CoarseCountSlot != INVALID_SLOT) {
        const uint coarse_count = simd_sum(coarse);
        if (simd_lane == 0u && coarse_count != 0u) {
            atomic_fetch_add_explicit(
                &BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.CoarseCountSlot)[0], coarse_count, memory_order_relaxed
            );
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
        const uint present = (routes >> MeshletRoute_Blend) & 1u;
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
    if (work.Instance != INVALID_OFFSET) {
        BindlessBufferMutable(uint, bindless.Buffer, pc.ClassificationSlot)[i] = routes | (blend_bucket << CullRouteCount);
    }
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
                (route_totals[MeshletRoute_Phase2Candidate] + Phase2GroupSize - 1u) / Phase2GroupSize, 1u, 1u
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
    const uint classification = valid ? BindlessBuffer(uint, bindless.Buffer, pc.ClassificationSlot)[i] : 0u;
    const uint routes = classification & ((1u << CullRouteCount) - 1u);
    const uint blend_bucket = classification >> CullRouteCount;
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
    if (sort_blend && present[MeshletRoute_Blend] != 0u) {
        uint blend_count = 0u;
        for (uint source = 0u; source < 32u; ++source) {
            const bool match = simd_shuffle(present[MeshletRoute_Blend], source) != 0u && simd_shuffle(blend_bucket, source) == blend_bucket;
            blend_count += match;
            blend_rank += match && source < simd_lane;
        }
        if (blend_rank == 0u) group_blend_counts[simd_group * 256u + blend_bucket] = blend_count;
    }
    if (sort_blend) threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sort_blend && present[MeshletRoute_Blend] != 0u) {
        for (uint group = 0u; group < simd_group; ++group) blend_rank += group_blend_counts[group * 256u + blend_bucket];
    }
    if (!valid) return;

    device const MeshletCullBlockState *blocks = BindlessBuffer(MeshletCullBlockState, bindless.Buffer, pc.BlockStateSlot);
    device const MeshletRouteState *state = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot);
    device VisibleMeshlet *visible = BindlessBufferMutable(VisibleMeshlet, bindless.Buffer, pc.VisibleSlot);
    for (uint route = 0u; route < CullRouteCount; ++route) {
        if (present[route] == 0u) continue;
        rank[route] += group_prefixes[route * PrefixStride + simd_group];
        if (route == MeshletRoute_Transmission) rank[route] = group_prefixes[route * PrefixStride + CullSimdGroups] - rank[route] - 1u;
        uint output = state->Offsets[route] + blocks[block_id].Routes[route] + rank[route];
        if (sort_blend && route == MeshletRoute_Blend) {
            device const MeshletBlendBlockState *blend_blocks = BindlessBuffer(MeshletBlendBlockState, bindless.Buffer, pc.BlendBlockSlot);
            output = state->Offsets[MeshletRoute_Blend] + state->BlendOffsets[blend_bucket] + blend_blocks[block_id].Buckets[blend_bucket] + blend_rank;
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
    if (!LodClusterVisible(scene, pc, meshlet, world, InstanceFinestOnly(scene, instance))) return false;
    if (pc.RouteMode != 0u) {
        const PrimitiveRecord primitive = BindlessBuffer(PrimitiveRecord, scene.B.Buffer, pc.PrimitiveSlot)[meshlet.Primitive];
        const bool triangle_topology = PrimitiveTopology(meshlet) == MeshPrimitiveTopology_Triangle;
        if (pc.RouteMode == 3u && !triangle_topology) return false;
        const PBRMaterial material = scene.Materials(scene.View.MaterialSlot)[PrimitiveMaterialIndex(scene, primitive)];
        // Solid visibility classifies rendered blend as opaque to match the primary visibility route.
        const bool edit_overlay = (instance.Flags & MeshletInstanceFlag_EditOverlay) != 0u;
        if (!edit_overlay && pc.RouteMode == 1u && material.AlphaMode == MaterialAlphaMode_Blend) return false;
        if (!edit_overlay && material.DoubleSided == 0u &&
            !MeshletConeVisible(scene, meshlet, world, InstanceDeformed(instance))) return false;
    }
    const MeshletBounds bounds = ResolveMeshletBounds(scene, pc, candidate, instance_slot, instance, meshlet, world);
    if (!bounds.Valid) return true;
    if (!MeshletBoundsInFrustum(scene, bounds)) return false;
    return !MeshletOccluded(scene, pc.PyramidSamplerSlot, scene.ViewProj(), bounds.Center, bounds.Ax, bounds.Ay, bounds.Az);
}

// The phase-2 count, prefix, and emit passes preserve candidate order independently of GPU scheduling.
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
    if (i < routes->Counts[MeshletRoute_Phase2Candidate]) {
        candidate = BindlessBuffer(VisibleMeshlet, bindless.Buffer, pc.VisibleSlot)[routes->Offsets[MeshletRoute_Phase2Candidate] + i];
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
    // Every lane evaluates the same complete-instance predicate.
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
        // Write zero because the prefix reads every range.
        if (pc.Phase2Emit == 0u && lane == 0u) ranges[range_id].WorkOffset = 0u;
        return;
    }
    // Store survivor counts in WorkOffset for conversion to emit offsets by the prefix pass.
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
    const uint block_count = (routes->Counts[MeshletRoute_Phase2Candidate] + Phase2GroupSize - 1u) / Phase2GroupSize;
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
    phase2->Counts[MeshletRoute_OpaqueCullBack] = total;
    phase2->Offsets[MeshletRoute_OpaqueCullBack] = 0u;
    phase2->Counts[MeshletRoute_EditOverlay] = total;
    phase2->Offsets[MeshletRoute_EditOverlay] = 0u;
    phase2->Counts[MeshletRoute_Overlay] = total;
    phase2->Offsets[MeshletRoute_Overlay] = 0u;
    device MeshDispatchArgs *args = BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.Phase2DispatchArgsSlot);
    for (uint chunk = 0u; chunk < pc.DispatchChunkCount; ++chunk) {
        const uint begin = chunk * pc.DispatchChunkSize;
        const MeshDispatchArgs dispatch = {total > begin ? min(total - begin, pc.DispatchChunkSize) : 0u, 1u, 1u};
        args[chunk] = dispatch;
        args[MeshletRoute_EditOverlay * pc.DispatchChunkCount + chunk] = dispatch;
        args[MeshletRoute_Overlay * pc.DispatchChunkCount + chunk] = dispatch;
    }
}
