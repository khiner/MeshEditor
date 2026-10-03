#include "MeshletIndexShared.metal"
#include "gpu/AABB.h"
#include "TransformUtils.metal"
#include "Bindless.metal"
#include "gpu/ClusterGroup.h"
#include "Frustum.metal"
#include "gpu/InstanceRecord.h"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/MeshletRouteMode.h"
#include "gpu/MaterialAlphaMode.h"
#include "gpu/LodFrontierBlockState.h"
#include "gpu/LodFrontierEntry.h"
#include "gpu/LodFrontierState.h"
#include "gpu/LodNode.h"
#include "gpu/MeshDispatchArgs.h"
#include "gpu/MeshletCullBlockState.h"
#include "gpu/MeshletCullPushConstants.h"
#include "gpu/MeshletRoute.h"
#include "MeshletShared.metal"
#include "gpu/MeshletRouteState.h"
#include "gpu/MeshletWorkRange.h"
#include "gpu/MeshletWorkState.h"
#include "gpu/PrimitiveRecord.h"
#include "PyramidOcclusion.metal"
#include "gpu/VisibleMeshlet.h"

constant uint CullBlockSize = 1024u;
constant uint CullRouteCount = uint(MeshletRoute::Count);
constant uint CullSimdGroups = 32u;
constant uint PrefixStride = CullSimdGroups + 1u;
constant uint ConeCullMinTriangles = 16u;
// The phase-2 cull kernels run one 32-lane simdgroup per threadgroup.

struct RoutedMeshlet {
    uint Routes;
    bool Coarse;
};

inline uint RouteBit(MeshletRoute route) { return 1u << uint(route); }

inline MeshletRoute OpaqueVisibilityRoute(PBRMaterial material, Transform world) {
    if (material.DoubleSided != 0u) return MeshletRoute::OpaqueDoubleSided;
    const float3 scale = float3(world.S);
    return scale.x * scale.y * scale.z < 0.0f ? MeshletRoute::OpaqueCullFront : MeshletRoute::OpaqueCullBack;
}

inline bool InstanceDeformed(InstanceRecord instance) {
    return instance.ArmatureDeformOffset != InvalidOffset || instance.MorphDeformOffset != InvalidOffset ||
        instance.PositionNamespace != InvalidOffset || instance.HasPendingVertexTransform != 0u;
}

// Edited and deformed instances use original geometry covered by their posed bounds.
inline bool InstancePinsFinest(InstanceRecord instance) {
    return (instance.Flags & (uint(MeshletInstanceFlag::LodPinFinest) | uint(MeshletInstanceFlag::Wire) |
        uint(MeshletInstanceFlag::FaceNormal) | uint(MeshletInstanceFlag::EdgeOverlay))) != 0u ||
        InstanceDeformed(instance);
}

// Nonpositive thresholds select original geometry.
inline bool InstanceFinestOnly(const thread Scene &scene, MeshletCullPushConstants pc, InstanceRecord instance) {
    const uint edit_flags = uint(MeshletInstanceFlag::ElementSelection) | uint(MeshletInstanceFlag::EditOverlay);
    return scene.View.LodErrorPixels <= 0.0f || InstancePinsFinest(instance) ||
        (pc.ExactEditGeometry != 0u && (instance.Flags & edit_flags) != 0u);
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
    if (meshlet.GroupIndex == InvalidOffset) return true;
    if (finest_only) return meshlet.RefinedGroup == InvalidOffset;
    device const ClusterGroup *groups = BindlessBuffer(ClusterGroup, scene.B.Buffer, pc.ClusterGroupSlot);
    if (LodGroupErrorPixels(scene, groups[meshlet.GroupIndex], world) <= scene.View.LodErrorPixels) return false;
    return meshlet.RefinedGroup == InvalidOffset ||
        LodGroupErrorPixels(scene, groups[meshlet.RefinedGroup], world) <= scene.View.LodErrorPixels;
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

// Each pose stores bounds at stable canonical cluster keys.
inline OrientedBounds DeformedMeshletBounds(
    const thread Scene &scene, MeshletCullPushConstants pc, VisibleMeshlet candidate,
    InstanceRecord instance, MeshletRecord meshlet, Transform world
) {
    if (instance.MeshletBoundsNamespace == InvalidOffset || pc.PosedMeshletBoundsSlot == InvalidSlot) return {};
    const uint index = PoseAttributeIndex(scene.B,pc.PosedMeshletBoundsNodesSlot,instance.MeshletBoundsNamespace,candidate.Meshlet);
    if (index == InvalidOffset) return {};
    const AABB bounds = BindlessBuffer(AABB,scene.B.Buffer,pc.PosedMeshletBoundsSlot)[index];
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

inline float EditEdgeMarginPixels(const thread Scene &scene, MeshletCullPushConstants pc) {
    if (pc.MinEditOverlayDiameterPixels <= 0.0f) return 0.0f;
    // StrokeQuadCorner extends both along and across an endpoint. The diagonal
    // plus half a pixel covers its raster footprint outside the meshlet bounds.
    const float width = scene.Theme.EdgeWidth;
    const float half_width = width + (pc.EditOverlayHasSharpEdges != 0u ? max(width, 1.0f) : 0.0f) + 0.5f;
    return 1.41421356f * half_width + 0.5f;
}

// Below one projected pixel, the original edges have no stable display footprint.
// Keep selection culls exact.
// This filter applies only to the visual edit route.
inline float MeshletDiameterPixels(const thread Scene &scene, MeshletBounds bounds) {
    if (!bounds.Valid) return INFINITY;
    return ProjectedDiameterPixels(scene, bounds.Center, bounds.Sphere ? bounds.Radius :
        length(bounds.Ax) + length(bounds.Ay) + length(bounds.Az));
}

// Edit edges are pulled toward the camera in clip space, so their meshlets test at the edges' nearest possible depth.
inline bool MeshletOccluded(
    const thread Scene &scene, uint pyramid_slot, float3 center, float3 ax, float3 ay, float3 az, float edge_margin
) {
    return BoxPastPyramid(scene, pyramid_slot, center, ax, ay, az, edge_margin, edge_margin > 0.0f ? scene.View.NdcOffsetFactor : 0.0f);
}

// Reject occluded instances before expanding their span trees.
inline uint ClassifyInstanceRange(
    const thread Scene &scene, MeshletCullPushConstants pc, uint instance_slot, InstanceRecord instance
) {
    if (instance.PrimitiveCount == 0u || (instance.Flags & pc.RequiredInstanceFlags) != pc.RequiredInstanceFlags) return 0u;
    // Posed meshlet bounds supersede the instance AABB, which may represent another motion-blur step.
    if (instance.MeshletBoundsNamespace != InvalidOffset) return 1u;
    const OrientedBounds bounds = InstanceBounds(scene, pc, instance_slot);
    if (!bounds.Valid) return 1u;
    if (!in_frustum(scene.ViewProj(), bounds.Center, bounds.Ax, bounds.Ay, bounds.Az)) return 0u;
    if ((instance.Flags & uint(MeshletInstanceFlag::OverlayOnly)) != 0u) return 1u;
    if (pc.PyramidSamplerSlot == InvalidSlot ||
        !MeshletOccluded(scene, pc.PyramidSamplerSlot, bounds.Center, bounds.Ax, bounds.Ay, bounds.Az,
                         EditEdgeMarginPixels(scene, pc))) return 1u;
    return 0u;
}

inline RoutedMeshlet ClassifyMeshlet(
    const thread Scene &scene, MeshletCullPushConstants pc, VisibleMeshlet candidate,
    uint instance_slot, InstanceRecord instance
) {
    RoutedMeshlet result{0u, false};
    const MeshletRecord meshlet = BindlessBuffer(MeshletRecord, scene.B.Buffer, pc.MeshletSlot)[candidate.Meshlet];
    const Transform world = scene.Models(pc.ModelSlot)[instance_slot];
    if (!LodClusterVisible(scene, pc, meshlet, world, InstanceFinestOnly(scene, pc, instance))) return result;
    result.Coarse = meshlet.RefinedGroup != InvalidOffset;
    const MeshletBounds bounds = ResolveMeshletBounds(scene, pc, candidate, instance_slot, instance, meshlet, world);
    if (bounds.Valid && !MeshletBoundsInFrustum(scene, bounds)) return result;
    const float3 world_center = bounds.Valid ? bounds.Center : float3(world.P);

    const PrimitiveRecord primitive = BindlessBuffer(PrimitiveRecord, scene.B.Buffer, pc.PrimitiveSlot)[meshlet.Primitive];
    const bool triangle_topology = MeshletPrimitiveTopology(meshlet) == uint(MeshPrimitiveTopology::Triangle);
    // A one-meshlet instance already passed the conservative instance query.
    const bool can_occlude = bounds.Valid && !(instance.PrimitiveCount == 1u && primitive.MeshletCount == 1u);
    PBRMaterial material{};
    const MeshletRouteMode mode = MeshletRouteMode(pc.RouteMode);
    if (mode != MeshletRouteMode::Single) material = scene.Materials(scene.View.MaterialSlot)[MeshletPrimitiveMaterialIndex(scene, primitive)];
    const bool edit_overlay = (instance.Flags & uint(MeshletInstanceFlag::EditOverlay)) != 0u;
    const bool overlay_only = (instance.Flags & uint(MeshletInstanceFlag::OverlayOnly)) != 0u;
    const bool cone_visible = mode == MeshletRouteMode::Single || material.DoubleSided != 0u ||
        MeshletConeVisible(scene, meshlet, world, InstanceDeformed(instance));
    const bool occluded = !overlay_only && can_occlude && pc.PyramidSamplerSlot != InvalidSlot &&
        MeshletOccluded(scene, pc.PyramidSamplerSlot, world_center, bounds.Ax, bounds.Ay, bounds.Az,
                        EditEdgeMarginPixels(scene, pc));

    if (overlay_only) {
        result.Routes = 0u;
    } else if (mode == MeshletRouteMode::Visibility && !triangle_topology) {
        result.Routes = 0u;
    } else if (mode == MeshletRouteMode::Single) {
        result.Routes = RouteBit(MeshletRoute::OpaqueCullBack);
    } else {
        const bool alpha_mask = material.AlphaMode == MaterialAlphaMode::Mask;
        const MeshletRoute opaque_route = triangle_topology ? OpaqueVisibilityRoute(material, world) : MeshletRoute::Coverage;
        if (mode == MeshletRouteMode::Visibility || mode == MeshletRouteMode::Selection) {
            result.Routes = RouteBit(alpha_mask ? MeshletRoute::Coverage : opaque_route);
        } else if (material.AlphaMode == MaterialAlphaMode::Blend) {
            result.Routes = RouteBit(MeshletRoute::Blend);
        } else if (mode == MeshletRouteMode::Material) {
            result.Routes = RouteBit(alpha_mask ? MeshletRoute::Coverage : opaque_route);
        } else {
            const bool transmissive = material.Transmission.Factor > 0.0f;
            if (!transmissive) {
                result.Routes = RouteBit(alpha_mask ? MeshletRoute::Coverage : opaque_route);
            } else {
                if (material.Transmission.Texture.Slot != InvalidSlot) result.Routes |= RouteBit(MeshletRoute::Coverage);
                result.Routes |= RouteBit(MeshletRoute::Transmission);
            }
        }
    }
    if (!cone_visible) result.Routes = 0u;
    if (edit_overlay && MeshletDiameterPixels(scene, bounds) >= pc.MinEditOverlayDiameterPixels) result.Routes |= RouteBit(MeshletRoute::EditOverlay);
    if ((instance.Flags & uint(MeshletInstanceFlag::Wire)) != 0u) result.Routes |= RouteBit(MeshletRoute::Wire);
    if ((instance.Flags & (uint(MeshletInstanceFlag::Bone) | uint(MeshletInstanceFlag::BoneJoint) |
        uint(MeshletInstanceFlag::FaceNormal) | uint(MeshletInstanceFlag::EdgeOverlay))) != 0u) {
        result.Routes |= RouteBit(MeshletRoute::Overlay);
    }
    result.Routes &= pc.RouteMask;
    if (result.Routes == 0u) return result;
    if (occluded) result.Routes = 0u;
    return result;
}

inline VisibleMeshlet ResolveMeshlet(
    device const BindlessSet &bindless, MeshletCullPushConstants pc, uint block_id, uint work_index
) {
    device const MeshletWorkState *state = BindlessBuffer(MeshletWorkState, bindless.Buffer, pc.WorkStateSlot);
    if (work_index >= state->MeshletCount) return {InvalidOffset, InvalidOffset, InvalidOffset};
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
    return {range.Instance, MeshletIndexSelect(bindless,{pc.MeshletIndexNodesSlot,pc.MeshletIndexLeavesSlot,range.MeshletRoot},work_index-range.WorkOffset), InvalidOffset};
}

// The mesh record of a candidate's instance, or InvalidOffset for an unmapped instance.
inline uint CandidateMesh(device const BindlessSet &bindless, MeshletCullPushConstants pc, VisibleMeshlet candidate) {
    if (candidate.Instance == InvalidOffset) return InvalidOffset;
    const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[candidate.Instance];
    return instance_slot == InvalidOffset ? InvalidOffset : BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[instance_slot].Mesh;
}

// The routes that draw surfaces.
constant MeshletRoute SurfaceRoutes[]{
    MeshletRoute::OpaqueCullBack, MeshletRoute::Blend, MeshletRoute::Transmission,
    MeshletRoute::OpaqueCullFront, MeshletRoute::OpaqueDoubleSided, MeshletRoute::Coverage,
};

// The source cull's surface route holding entry `i`, or CullRouteCount.
inline uint SourceSurfaceRoute(device const BindlessSet &bindless, MeshletCullPushConstants pc, uint i) {
    const MeshletRouteState source = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.SourceRouteStateSlot)[0];
    for (const MeshletRoute route : SurfaceRoutes) {
        if (i - source.Offsets[uint(route)] < source.Counts[uint(route)]) return uint(route);
    }
    return CullRouteCount;
}

// A cull filtering a source cull reads the source's surface entries in place of traversal work.
inline VisibleMeshlet ResolveCullEntry(device const BindlessSet &bindless, MeshletCullPushConstants pc, uint block_id, uint i) {
    if (pc.SourceVisibleSlot == InvalidSlot) return ResolveMeshlet(bindless, pc, block_id, i);
    if (SourceSurfaceRoute(bindless, pc, i) == CullRouteCount) return {InvalidOffset, InvalidOffset, InvalidOffset};
    return BindlessBuffer(VisibleMeshlet, bindless.Buffer, pc.SourceVisibleSlot)[i];
}

// Keeps an outlined surface entry in its route.
// The silhouette seed resolves every opaque outline except where an unoutlined surface may lie in front.
inline uint SilhouetteRoutes(
    const thread Scene &scene, MeshletCullPushConstants pc, VisibleMeshlet entry, uint instance_slot, InstanceRecord instance, uint i
) {
    if ((instance.Flags & uint(MeshletInstanceFlag::Silhouette)) == 0u) return 0u;
    const uint route = SourceSurfaceRoute(scene.B, pc, i);
    if (route == uint(MeshletRoute::Blend) || route == uint(MeshletRoute::Transmission)) return 1u << route;
    const MeshletRecord meshlet = BindlessBuffer(MeshletRecord, scene.B.Buffer, pc.MeshletSlot)[entry.Meshlet];
    const MeshletBounds bounds = ResolveMeshletBounds(scene, pc, entry, instance_slot, instance, meshlet, scene.Models(pc.ModelSlot)[instance_slot]);
    const bool hidden = !bounds.Valid ||
        !BoxPastPyramid(scene, pc.PyramidSamplerSlot, bounds.Center, bounds.Ax, bounds.Ay, bounds.Az, 0.0f, 0.0f, true);
    return hidden ? 1u << route : 0u;
}

// Sizes a filtering cull's block dispatches to the source cull's surface entries.
kernel void SilhouetteCullSize(
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant MeshletCullPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const MeshletRouteState source = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.SourceRouteStateSlot)[0];
    uint end = 0u;
    for (const MeshletRoute route : SurfaceRoutes) {
        if (source.Counts[uint(route)] > 0u) end = max(end, source.Offsets[uint(route)] + source.Counts[uint(route)]);
    }
    const uint blocks = (end + CullBlockSize - 1u) / CullBlockSize;
    BindlessBufferMutable(MeshletWorkState, bindless.Buffer, pc.WorkStateSlot)[0].CullBlockCount = blocks;
    BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.WorkDispatchArgsSlot)[0] = {blocks, 1u, 1u};
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

// Stores one entry's child nodes and final-level record range.
struct LodWork {
    uint Instance;
    uint Node;
    uint ChildCount;
    uint MeshletCount;
};

// Seeds traversal from instance IDs and writes primitive roots.
inline LodWork ResolveLodSeed(const thread Scene &scene, MeshletCullPushConstants pc, uint id) {
    if (id >= pc.InstanceCount) return {};
    const uint instance_slot = BindlessBuffer(uint, scene.B.Buffer, pc.InstanceMapSlot)[id];
    if (instance_slot == InvalidOffset) return {};
    const InstanceRecord instance = BindlessBuffer(InstanceRecord, scene.B.Buffer, pc.InstanceSlot)[instance_slot];
    const uint visibility = ClassifyInstanceRange(scene, pc, instance_slot, instance);
    if (visibility == 0u) return {};
    // Membership contains only primitives with finest geometry. Each emits
    // one root, so seeding needs no primitive traversal or record loads.
    return {id, InvalidOffset, instance.PrimitiveCount, 0u};
}

// Expands one frontier node into child nodes or its final-level record range.
inline LodWork ResolveLodNode(const thread Scene &scene, MeshletCullPushConstants pc, uint index) {
    device const LodFrontierState *states = BindlessBuffer(LodFrontierState, scene.B.Buffer, pc.LodFrontierStateSlot);
    if (index >= states[pc.LodFrontierIndex].NodeCount) return {};
    const LodFrontierEntry entry = BindlessBuffer(LodFrontierEntry, scene.B.Buffer, pc.LodFrontierSlot)[index];
    const uint instance_slot = BindlessBuffer(uint, scene.B.Buffer, pc.InstanceMapSlot)[entry.Instance];
    if (instance_slot == InvalidOffset) return {};
    const LodNode node = BindlessBuffer(LodNode, scene.B.Buffer, pc.LodNodeSlot)[entry.Node];
    if (!LodNodeVisible(scene, node, scene.Models(pc.ModelSlot)[instance_slot])) return {};
    // The final level emits the complete range of nodes deeper than the recorded depth.
    if (pc.LodFinalLevel != 0u) {
        const uint count = node.MeshletRoot == InvalidOffset ? 0u :
            BindlessBuffer(MeshletIndexNode,scene.B.Buffer,pc.MeshletIndexNodesSlot)[node.MeshletRoot].Count;
        return {entry.Instance,entry.Node,count ? 1u : 0u,count};
    }
    // Repeat leaves through later levels to preserve frontier order.
    return {entry.Instance, entry.Node, max(node.ChildCount, 1u), 0u};
}

inline LodWork ResolveLodWork(const thread Scene &scene, MeshletCullPushConstants pc, uint index) {
    return pc.LodSeedLevel != 0u ? ResolveLodSeed(scene, pc, index) : ResolveLodNode(scene, pc, index);
}

// Writes each simdgroup's two totals into the corresponding prefix-row lane.
inline void WriteLodSimdGroupSums(
    threadgroup uint *group_prefixes, LodWork work, uint simd_lane, uint simd_group
) {
    const uint node_sum = simd_sum(work.ChildCount);
    const uint meshlet_sum = simd_sum(work.MeshletCount);
    if (simd_lane == 0u) {
        group_prefixes[simd_group] = node_sum;
        group_prefixes[PrefixStride + simd_group] = meshlet_sum;
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
        uint nodes = 0u, meshlets = 0u;
        for (uint group = 0u; group < CullSimdGroups; ++group) {
            nodes += group_prefixes[group];
            meshlets += group_prefixes[PrefixStride + group];
        }
        BindlessBufferMutable(LodFrontierBlockState, bindless.Buffer, pc.LodFrontierBlockStateSlot)[block_id] = {
            nodes, meshlets
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
    uint node_count = 0u, meshlet_count = 0u;
    for (uint block = 0u; block < block_count; ++block) {
        const LodFrontierBlockState count = blocks[block];
        blocks[block] = {node_count, meshlet_count};
        node_count += count.NodeCount;
        meshlet_count += count.MeshletCount;
    }
    const uint next_block_count = (node_count + CullBlockSize - 1u) / CullBlockSize;
    states[pc.LodFrontierIndex ^ 1u] = {node_count, next_block_count};
    BindlessBufferMutable(MeshDispatchArgs, bindless.Buffer, pc.LodExpandArgsSlot)[pc.LodFrontierIndex ^ 1u] = {
        next_block_count, 1u, 1u
    };
    device MeshletWorkState *state = BindlessBufferMutable(MeshletWorkState, bindless.Buffer, pc.WorkStateSlot);
    if (pc.LodSeedLevel != 0u) {
        state[0] = {0u, 0u, 0u};
        if (pc.CoarseCountSlot != InvalidSlot) BindlessBufferMutable(uint, bindless.Buffer, pc.CoarseCountSlot)[0] = 0u;
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
    WriteLodSimdGroupSums(group_prefixes, work, simd_lane, simd_group);
    if (simd_group == 0u) {
        const uint nodes = simd_lane < CullSimdGroups ? group_prefixes[simd_lane] : 0u;
        const uint meshlets = simd_lane < CullSimdGroups ? group_prefixes[PrefixStride + simd_lane] : 0u;
        if (simd_lane < CullSimdGroups) {
            group_prefixes[simd_lane] = simd_prefix_exclusive_sum(nodes);
            group_prefixes[PrefixStride + simd_lane] = simd_prefix_exclusive_sum(meshlets);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const LodFrontierBlockState block = BindlessBuffer(LodFrontierBlockState, bindless.Buffer, pc.LodFrontierBlockStateSlot)[block_id];
    device const PrimitiveRecord *primitives = BindlessBuffer(PrimitiveRecord, bindless.Buffer, pc.PrimitiveSlot);
    if (work.ChildCount == 0u) return;
    node_rank += group_prefixes[simd_group];
    uint output = block.NodeCount + node_rank;

    if (pc.LodFinalLevel != 0u) {
        meshlet_rank += group_prefixes[PrefixStride + simd_group];
        const uint work_offset = block.MeshletCount + meshlet_rank;
        const LodNode node = BindlessBuffer(LodNode, bindless.Buffer, pc.LodNodeSlot)[work.Node];
        BindlessBufferMutable(MeshletWorkRange, bindless.Buffer, pc.WorkRangeSlot)[output] = {
            work.Instance, node.MeshletRoot, work.MeshletCount, work_offset
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
        const bool finest_only = InstanceFinestOnly(scene, pc, instance);
        for (uint p = 0u; p < instance.PrimitiveCount; ++p) {
            const PrimitiveRecord primitive = primitives[MeshletIndexSelect(scene.B,{pc.MeshletIndexNodesSlot,pc.MeshletIndexLeavesSlot,instance.PrimitiveRoot},p)];
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
    const uint i = block_id * CullBlockSize + lane;
    const VisibleMeshlet work = ResolveCullEntry(bindless, pc, block_id, i);
    uint routes = 0u, coarse = 0u;
    if (work.Instance != InvalidOffset) {
        const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[work.Instance];
        if (instance_slot != InvalidOffset) {
            const InstanceRecord instance = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[instance_slot];
            const Scene scene{bindless, view, theme, workspace};
            const RoutedMeshlet routed = pc.SourceVisibleSlot == InvalidSlot ? ClassifyMeshlet(scene, pc, work, instance_slot, instance) :
                                                                              RoutedMeshlet{SilhouetteRoutes(scene, pc, work, instance_slot, instance, i), false};
            routes = routed.Routes;
            coarse = routed.Routes != 0u && routed.Coarse ? 1u : 0u;
        }
    }
    // Accumulate one value per simdgroup because profiling records only the total.
    if (pc.CoarseCountSlot != InvalidSlot) {
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

    if (work.Instance != InvalidOffset) {
        BindlessBufferMutable(uint, bindless.Buffer, pc.ClassificationSlot)[i] = routes;
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
    if (lane < CullRouteCount) {
        uint total = 0u;
        for (uint block = 0u; block < block_count; ++block) {
            const uint count = blocks[block].Routes[lane];
            blocks[block].Routes[lane] = total;
            total += count;
        }
        route_totals[lane] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (lane == 0u) {
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
    const VisibleMeshlet work = ResolveCullEntry(bindless, pc, block_id, i);
    const bool valid = work.Instance != InvalidOffset;
    const uint mesh = CandidateMesh(bindless, pc, work);
    const uint classification = valid ? BindlessBuffer(uint, bindless.Buffer, pc.ClassificationSlot)[i] : 0u;
    const uint routes = classification & ((1u << CullRouteCount) - 1u);

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

    if (!valid) return;

    device const MeshletCullBlockState *blocks = BindlessBuffer(MeshletCullBlockState, bindless.Buffer, pc.BlockStateSlot);
    device const MeshletRouteState *state = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot);
    device VisibleMeshlet *visible = BindlessBufferMutable(VisibleMeshlet, bindless.Buffer, pc.VisibleSlot);
    for (uint route = 0u; route < CullRouteCount; ++route) {
        if (present[route] == 0u) continue;
        rank[route] += group_prefixes[route * PrefixStride + simd_group];
        uint output = state->Offsets[route] + blocks[block_id].Routes[route] + rank[route];
        visible[output] = {work.Instance, work.Meshlet, mesh};
    }
}
