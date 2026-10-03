#include "metal/AutoreleaseScope.h"
#include "numeric/uvec2.h"

#include "render/LightComponents.h"
#include "viewport/ViewportRenderGpu.h"

#include "Camera.h"
#include "Profile.h"
#include "Variant.h"
#include "animation/AnimationTimeline.h"
#include "animation/MorphWeights.h"
#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "gizmo/GizmoInteraction.h"
#include "gpu/BoundsEntry.h"
#include "gpu/BoundsReducePushConstants.h"
#include "gpu/CommitPosedGeometryPushConstants.h"
#include "gpu/DepthPyramidReducePushConstants.h"
#include "gpu/ExtrasLineKind.h"
#include "gpu/MeshletCullPushConstants.h"
#include "gpu/MeshletDrawPushConstants.h"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/MotionBlurAccumulatePushConstants.h"
#include "gpu/MotionBlurGatherPushConstants.h"
#include "gpu/MotionBlurResolvePushConstants.h"
#include "gpu/MotionBlurTilesFlattenPushConstants.h"
#include "gpu/NormalDeriveEntry.h"
#include "gpu/NormalDerivePushConstants.h"
#include "gpu/OverlayDispatch.h"
#include "gpu/OverlayJob.h"
#include "gpu/OverlayJobCullPushConstants.h"
#include "gpu/OverlayJobDrawPushConstants.h"
#include "gpu/OverlayJobKind.h"
#include "gpu/PosedMeshletBoundsPushConstants.h"
#include "gpu/SilhouetteEdgeColorPushConstants.h"
#include "gpu/VertexBlockPushConstants.h"
#include "gpu/ViewportCompositePushConstants.h"
#include "gpu/VisibilityId.h"
#include "gpu/WireRasterPushConstants.h"
#include "gpu/WireResolvePushConstants.h"
#include "mesh/MeshComponents.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshStore.h"
#include "mesh/NormalDeriveGpu.h"
#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"
#include "metal/MetalCpp.h"
#include "metal/PassChain.h"
#include "metal/RenderTarget.h"
#include "numeric/MatrixMath.h"
#include "physics/PhysicsTypes.h"
#include "render/ElementWorkOps.h"
#include "render/Encoding.h"
#include "render/GpuBufferOps.h"
#include "render/MeshletBoundsRefit.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/Pipelines.h"
#include "render/RenderTargets.h"
#include "render/SceneUpdates.h"
#include "scene/CameraLens.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionGpu.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCamera.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportInteractionState.h"

#include "state/Scene.h"

#include <cassert>
#include <cstring>
#include <numbers>

using std::ranges::any_of, std::ranges::to;

namespace {
// Wireframe form of an extras line source: which generator draws it, its parameters, and its line count.
struct ExtrasLine {
    ExtrasLineKind Kind{};
    vec4 Params{};
    uint32_t LineCount{0};
};

ExtrasLine ColliderWireParams(const PhysicsShape &shape) {
    constexpr auto CircleSegments = uint32_t(OverlayDispatch::ColliderCircleSegments);
    constexpr uint32_t CapLines = CircleSegments + 4 * (CircleSegments / 2);
    return std::visit(
        overloaded{
            [](const physics::Box &s) { return ExtrasLine{ExtrasLineKind::ColliderBox, vec4{s.Size, 0}, 12}; },
            [](const physics::Sphere &s) { return ExtrasLine{ExtrasLineKind::ColliderSphere, vec4{s.Radius, 0, 0, 0}, 3 * CircleSegments}; },
            [](const physics::Cylinder &s) { return ExtrasLine{ExtrasLineKind::ColliderCylinder, vec4{s.RadiusTop, s.RadiusBottom, s.Height, 0}, 2 * CircleSegments + 4}; },
            [](const physics::Capsule &s) { return ExtrasLine{ExtrasLineKind::ColliderCapsule, vec4{s.RadiusTop, s.RadiusBottom, s.Height, 0}, 2 * CapLines + 4}; },
            [](const auto &) { return ExtrasLine{}; },
        },
        shape
    );
}

ExtrasLine ExtrasGizmoParams(const state::Scene &r, state::Entity object, ObjectType type) {
    constexpr auto HaloLines = uint32_t(OverlayDispatch::ExtrasHaloLines);
    constexpr auto RangeSegments = uint32_t(OverlayDispatch::LightRangeSegments);
    constexpr auto SpotSegments = uint32_t(OverlayDispatch::SpotConeSegments);
    if (type == ObjectType::Empty) return {ExtrasLineKind::Empty, {}, 3};
    if (type == ObjectType::Camera) {
        // Matches Blender's overlay at default drawsize 1: the frame spans one unit on its dominant axis.
        constexpr float HalfExtent{0.5f};
        const auto camera = *LensOf(r, object);
        float depth{1.f}, half_w{HalfExtent}, half_h{HalfExtent};
        if (const auto *perspective = std::get_if<Perspective>(&camera)) {
            const float aspect = AspectRatio(camera);
            half_w = aspect >= 1.f ? HalfExtent : HalfExtent * aspect;
            half_h = aspect >= 1.f ? HalfExtent / aspect : HalfExtent;
            depth = half_h / std::tan(perspective->FieldOfViewRad * 0.5f);
        } else if (const auto *orthographic = std::get_if<Orthographic>(&camera)) {
            half_w = orthographic->Mag.x;
            half_h = orthographic->Mag.y;
        }
        return {ExtrasLineKind::Camera, vec4{half_w, half_h, depth, r.all_of<LookingThrough>(object) ? 1.f : 0.f}, 11};
    }

    if (type != ObjectType::Light) return {};

    const auto &light = r.get<const PunctualLight>(object);
    const uint32_t range_lines = light.Range > 0.f ? RangeSegments : 0;
    if (light.Type == PunctualLightType::Point) {
        return {ExtrasLineKind::LightPoint, vec4{light.Range, 0, 0, 0}, range_lines + HaloLines};
    }
    if (light.Type == PunctualLightType::Directional) {
        return {ExtrasLineKind::LightDirectional, vec4{light.Range, 0, 0, 0}, 8 * 2 + HaloLines};
    }
    constexpr float SpotDepth{2.f};
    const float outer_angle = std::min(light.OuterConeAngle, Radians(89.f));
    const float inner_angle = std::min(light.InnerConeAngle, outer_angle);
    const float outer_radius = SpotDepth * std::tan(outer_angle), inner_radius = SpotDepth * std::tan(inner_angle);
    const uint32_t inner_lines = inner_radius > 0.f ? SpotSegments : 0;
    return {
        ExtrasLineKind::LightSpot,
        vec4{light.Range, outer_radius, inner_radius, 0},
        range_lines + SpotSegments + inner_lines + SpotSegments + HaloLines,
    };
}

// Stores stable threadgroup chunks while the GPU filters dynamic settings and selection state at use time.
std::vector<OverlayJob> BuildOverlayJobs(const state::Scene &r) {
    constexpr uint32_t LinesPerJob{uint32_t(OverlayDispatch::LineGroupLines)};
    std::vector<OverlayJob> jobs;
    const auto append = [&](OverlayJob job, uint32_t element_count) {
        for (uint32_t first = 0u; first < element_count; first += LinesPerJob) {
            job.FirstElement = first;
            job.ElementCount = std::min(LinesPerJob, element_count - first);
            jobs.emplace_back(job);
        }
    };
    for (const auto [object, kind, instance, render_instance] : r.view<const ObjectKind, const Instance, const RenderInstance>().each()) {
        if (!r.all_of<ObjectExtrasTag>(instance.Entity) || r.all_of<Hidden>(object)) continue;
        const auto gizmo = ExtrasGizmoParams(r, object, kind.Value);
        if (gizmo.LineCount == 0) continue;
        append(OverlayJob{
                   .Kind = OverlayJobKind::Extras,
                   .InstanceIndex = render_instance.BufferIndex,
                   .ExtrasKind = gizmo.Kind,
                   .LocalOffset = vec3{0},
                   .Params = gizmo.Params,
               },
               gizmo.LineCount);
    }
    for (const auto [entity, shape, render_instance] : r.view<const ColliderShape, const RenderInstance>().each()) {
        const auto wire = ColliderWireParams(shape.Shape);
        if (wire.LineCount == 0) continue;
        append(OverlayJob{
                   .Kind = OverlayJobKind::Extras,
                   .InstanceIndex = render_instance.BufferIndex,
                   .ExtrasKind = wire.Kind,
                   .LocalOffset = shape.LocalOffset,
                   .Params = wire.Params,
               },
               wire.LineCount);
    }
    for (const auto [entity, instance, render_instance] : r.view<const Instance, const RenderInstance>().each()) {
        if (!HasMesh(r, instance.Entity)) continue;
        append(OverlayJob{
                   .Kind = OverlayJobKind::Bounds,
                   .InstanceIndex = render_instance.BufferIndex,
               },
               12u);
    }
    for (const auto [entity, instance, render_instance] : r.view<const Instance, const RenderInstance>().each()) {
        const auto *tets = r.try_get<const TetBuffers>(instance.Entity);
        if (!tets || tets->EdgeIndices.Count == 0u) continue;
        append(OverlayJob{
                   .Kind = OverlayJobKind::TetWire,
                   .InstanceIndex = render_instance.BufferIndex,
                   .SourceOffset = tets->Positions.Offset,
                   .IndexOffset = tets->EdgeIndices.Offset,
               },
               tets->EdgeIndices.Count / 2u);
    }
    return jobs;
}

void RecordSceneCounters(const GpuBuffers &buffers) {
    profile::RecordCounter("InstanceSlots", buffers.Instances.TransformBuffer.UsedSize / sizeof(Transform));
    profile::RecordCounter("MeshletRecords", buffers.Meshlets.Buffer.Count<MeshletRecord>());
    profile::RecordCounter("MeshletInstances", buffers.MeshletInstanceCount);
    profile::RecordCounter("MeshletRecordBytes", buffers.Meshlets.Buffer.UsedSize);
    profile::RecordCounter("MeshletTriangleIdBytes", buffers.MeshletTriangleIds.Buffer.UsedSize);
    profile::RecordCounter("PrimitiveRecordBytes", buffers.Primitives.Buffer.UsedSize);
    profile::RecordCounter("InstanceRecordBytes", buffers.Instances.RecordBuffer.UsedSize);
    profile::RecordCounter("OverlayJobs", buffers.OverlayJobs.Count<OverlayJob>());
    if (buffers.OverlayJobDispatchArgs.Contents().size() >= sizeof(MeshDispatchArgs)) {
        profile::RecordCounter(
            "VisibleOverlayJobs",
            reinterpret_cast<const MeshDispatchArgs *>(buffers.OverlayJobDispatchArgs.Contents().data())->ThreadgroupsX
        );
    }
    if (buffers.SceneCull.Routes.Contents().size() >= sizeof(MeshletRouteState)) {
        const auto &routes = *reinterpret_cast<const MeshletRouteState *>(buffers.SceneCull.Routes.Contents().data());
        const auto count = [&](MeshletRoute route) { return routes.Counts[uint32_t(route)]; };
        profile::RecordCounter("VisibleOpaqueMeshlets", count(MeshletRoute::OpaqueCullBack) + count(MeshletRoute::OpaqueCullFront) + count(MeshletRoute::OpaqueDoubleSided) + count(MeshletRoute::Coverage));
        profile::RecordCounter("VisibleCoverageMeshlets", count(MeshletRoute::Coverage));
        profile::RecordCounter("SelectedCoarseMeshlets", *reinterpret_cast<const uint32_t *>(buffers.MeshletCoarseCount.Contents().data()));
        profile::RecordCounter("VisibleBlendMeshlets", count(MeshletRoute::Blend));
        profile::RecordCounter("VisibleTransmissionMeshlets", count(MeshletRoute::Transmission));
    }
    if (buffers.EditCull.Routes.Contents().size() >= sizeof(MeshletRouteState)) {
        const auto &routes = *reinterpret_cast<const MeshletRouteState *>(buffers.EditCull.Routes.Contents().data());
        profile::RecordCounter("VisibleEditOverlayMeshlets", routes.Counts[uint32_t(MeshletRoute::EditOverlay)]);
    }
    profile::RecordCounter("DeviceAllocatedBytes", buffers.Ctx.Ctx.Device->currentAllocatedSize());
}

// Hashes every input that lacks explicit invalidation so matching records can be reused.
struct RecordInputs {
    uint64_t Value{0xcbf29ce484222325ull};
    void Mix(uint64_t v) { Value = (Value ^ v) * 0x100000001b3ull; }
    void Mix(SlotOffset range) {
        Mix(range.Slot);
        Mix(range.Offset);
    }
    void Mix(EditSelectionStorage selection) {
        Mix(selection.VertexBits);
        Mix(selection.EdgeBits);
        Mix(selection.FaceBits);
        Mix(selection.Summary);
    }
};

struct DeformSlots {
    uint32_t BoneDeformOffset{InvalidOffset}, ArmatureDeformOffset{InvalidOffset}, MorphDeformOffset{InvalidOffset};
    uint32_t MorphTargetCount{0};
    // Per-instance armature palette: buffer_index -> offset (instances of one mesh can bind different armatures)
    std::unordered_map<uint32_t, uint32_t> ArmatureDeformByBufferIndex;
    // Per-instance morph weights: buffer_index -> offset (weights are per-node in glTF)
    std::unordered_map<uint32_t, uint32_t> MorphWeightsByBufferIndex;
};

// `inputs` includes per-instance deform offsets absent from per-mesh fields.
std::unordered_map<state::Entity, DeformSlots> BuildDeformSlots(const state::Scene &r, const MeshStore &meshes, RecordInputs &inputs) {
    std::unordered_map<state::Entity, DeformSlots> result;
    for (const auto [instance_entity, instance, modifier] : r.view<const Instance, const ArmatureModifier>().each()) {
        const auto &mesh_record = meshes.Get(r.get<const MeshHandle>(instance.Entity).StoreId);
        if (!mesh_record.SkinBlocksReady) continue;
        const auto *pose_state = r.try_get<const ArmaturePoseState>(modifier.ArmatureEntity);
        if (!pose_state || modifier.SkinSlot >= pose_state->GpuDeformRanges.size()) continue;
        const auto deform_offset = pose_state->GpuDeformRanges[modifier.SkinSlot].Offset;
        auto &slots = result[instance.Entity];
        if (slots.BoneDeformOffset == InvalidOffset) {
            slots.BoneDeformOffset = 0u;
            slots.ArmatureDeformOffset = deform_offset;
        }
        if (const auto *ri = r.try_get<const RenderInstance>(instance_entity)) {
            slots.ArmatureDeformByBufferIndex[ri->BufferIndex] = deform_offset;
            inputs.Mix(ri->BufferIndex);
            inputs.Mix(deform_offset);
        }
    }
    for (const auto [instance_entity, instance, gpu_range, ri] : r.view<const Instance, const MorphWeightRange, const RenderInstance>().each()) {
        const auto mesh_entity = instance.Entity;
        const auto &record = meshes.Get(r.get<const MeshHandle>(mesh_entity).StoreId);
        if (!record.MorphBlocksReady) continue;
        auto &slots = result[mesh_entity];
        slots.MorphDeformOffset = 0u;
        slots.MorphTargetCount = record.MorphTargetCount;
        slots.MorphWeightsByBufferIndex[ri.BufferIndex] = gpu_range.Weights.Offset;
        inputs.Mix(ri.BufferIndex);
        inputs.Mix(gpu_range.Weights.Offset);
    }
    return result;
}

// Edit overlay draws skip geometry whose projected diameter falls below this many pixels.
constexpr float MinEditOverlayDiameterPixels{2.0f};

// Threadgroup memory lengths must be 16-byte multiples.
constexpr uint32_t AlignedThreadgroupBytes(uint32_t bytes) { return (bytes + 15u) & ~15u; }

// Slot of each prelude pass's args in GpuBuffers::PreludeDispatchArgs (PreludeGroups order).
enum class PreludeSlot : uint32_t { PosePrepass,
                                    PosedMeshletBounds,
                                    DeriveFaces,
                                    BoundsLevel1,
                                    DeriveGather,
                                    BoundsLevel2, BoundsLevel3 };

constexpr uint64_t PreludeArgsOffset(PreludeSlot slot) { return uint64_t(slot) * sizeof(MTL::DispatchThreadgroupsIndirectArguments); }

// Record one prelude pass's dispatch, reading its group count from the pass's indirect args slot.
void DispatchPrelude(MTL::ComputeCommandEncoder *encoder, const GpuBuffers &buffers, PreludeSlot slot) {
    encoder->dispatchThreadgroups(*buffers.PreludeDispatchArgs, PreludeArgsOffset(slot), ThreadgroupSize::Linear256);
}

// Address metadata only.
// Unchanged namespace revisions skip this enumeration.
template<typename T>
std::vector<uint32_t> PoseElementBlocks(const ElementArena<T> &arena, ElementSetRef set) {
    std::vector<uint32_t> blocks;
    for (auto b = set ? arena.Set(set).First : InvalidOffset; b != InvalidOffset; b = arena.Blocks.Get({b,1u})[0].Next)
        blocks.push_back(b);
    return blocks;
}

std::vector<uint32_t> NormalPayloadBlocks(const MeshStore &meshes, uint32_t store_id) {
    std::vector<uint32_t> blocks;
    for (const auto block : meshes.GetBlockList(store_id, MeshStore::ElementDomain::Halfedge).Blocks)
        if (const auto payload = meshes.Arenas().NormalSectors.PayloadBlock(block)) blocks.push_back(payload - 1u);
    return blocks;
}


} // namespace

namespace {
// Materialize posed positions and reduce every entry's canonical vertex blocks.
void RecordPosePrepass(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const Pipelines &pipelines, const GpuBuffers &buffers, uint32_t ubo_offset) {
    const auto &prepass = pipelines.PosePrepass;
    encode::BindCompute(encoder, prepass, slots, buffers, ubo_offset);
    const BoundsReducePushConstants pc{
        .BoundsEntrySlot = buffers.BoundsReduceEntries.Slot,
        .TileMapSlot = buffers.BoundsTiles.Slot,
        .ValuesSlot = buffers.VertexBounds.Values.Buffer.Slot,
        .NodesSlot = buffers.VertexBounds.Nodes.Buffer.Slot,
        .MembersSlot = buffers.VertexBounds.Members.Slot,
    };
    encode::SetPushConstants(encoder, pc);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector,0);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector,1);
    DispatchPrelude(encoder, buffers, PreludeSlot::PosePrepass);
}

// One derive dispatch over the tiles at `pc.FirstTile`, running the face or gather phase per pc.Phase.
// The tile count comes from `slot`'s indirect args.
void RecordNormalDerive(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const mtl::ComputePipeline &pipeline, const GpuBuffers &buffers, const NormalDerivePushConstants &pc, PreludeSlot slot, uint32_t ubo_offset) {
    encode::BindCompute(encoder, pipeline, slots, buffers, ubo_offset);
    encode::SetPushConstants(encoder, pc);
    DispatchPrelude(encoder, buffers, slot);
}

// Shared derive resources.
// Each entry selects base records or a posed namespace.
NormalDerivePushConstants MakeNormalDerivePc(const GpuBuffers &buffers, const MeshStore &meshes, uint32_t vertex_normal_slot, uint32_t face_normal_slot) {
    return {
        .EntriesSlot = buffers.NormalDeriveEntries.Slot,
        .CornerSectors = meshes.Slots().CornerSector,
        .EdgeSharpnessSlot = meshes.Slots().EdgeSharpness,
        .FaceSharpnessSlot = meshes.Slots().FaceSharpness,
        .TileMapSlot = buffers.DeriveTiles.Slot,
        .PositionSlot = buffers.PosedPositions.Values.Buffer.Slot,
        .PositionNodesSlot = buffers.PosedPositions.Nodes.Buffer.Slot,
        .VertexNormalSlot = vertex_normal_slot,
        .VertexNormalNodesSlot = buffers.PosedVertexNormals.Nodes.Buffer.Slot,
        .NormalSectors = meshes.Slots().NormalSector,
        .PosedSectorNodesSlot = buffers.PosedSectors.Nodes.Buffer.Slot,
        .PosedSectorValuesSlot = buffers.PosedSectors.Values.Buffer.Slot,
        .FaceNormalSlot = face_normal_slot,
        .FaceNormalNodesSlot = buffers.PosedFaceNormals.Nodes.Buffer.Slot,
        .BaseFaceNormalSlot = meshes.Slots().BaseFaceNormal,
    };
}

void RecordBoundsPass(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const mtl::ComputePipeline &pipeline, const GpuBuffers &buffers, PreludeSlot slot, uint32_t ubo_offset, BoundsReducePushConstants pc = {}) {
    pc.BoundsEntrySlot = buffers.BoundsReduceEntries.Slot;
    pc.BoundsSlot = buffers.Instances.BoundsBuffer.Slot;
    pc.TileMapSlot = buffers.BoundsTiles.Slot;
    pc.ValuesSlot = buffers.VertexBounds.Values.Buffer.Slot;
    pc.NodesSlot = buffers.VertexBounds.Nodes.Buffer.Slot;
    pc.MembersSlot = buffers.VertexBounds.Members.Slot;
    pc.FirstTile = buffers.BoundsFirstTiles[pc.Level];
    encode::BindCompute(encoder, pipeline, slots, buffers, ubo_offset);
    encode::SetPushConstants(encoder, pc);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector, 0);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector, 1);
    if (pc.Work.Storage.Slot == InvalidSlot) DispatchPrelude(encoder, buffers, slot);
    else encoder->dispatchThreadgroups(*buffers.GeometryWork.Buffer, WorkArgsOffset(pc.Work, true), ThreadgroupSize::Linear256);
}

void RecordPosedMeshletBounds(
    MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots,
    const Pipelines &pipelines, const GpuBuffers &buffers, uint32_t ubo_offset,
    PosedMeshletBoundsPushConstants pc = {}
) {
    pc.JobsSlot = buffers.PosedMeshletBoundsJobs.Slot;
    pc.JobCount = buffers.PosedMeshletBoundsJobs.Count<PosedMeshletBoundsJob>();
    pc.MeshletSlot = buffers.Meshlets.Buffer.Slot;
    pc.MeshletVertexSlot = buffers.MeshletVertexCorners.Buffer.Slot;
    pc.PosedMeshletBoundsSlot = buffers.PosedMeshletBounds.Values.Buffer.Slot;
    pc.PosedMeshletBoundsNodesSlot = buffers.PosedMeshletBounds.Nodes.Buffer.Slot;
    encode::BindCompute(encoder, pipelines.PosedMeshletBounds, slots, buffers, ubo_offset);
    encode::SetPushConstants(encoder, pc);
    if (pc.Work.Storage.Slot == InvalidSlot) {
        encoder->dispatchThreadgroups(*buffers.PreludeDispatchArgs, PreludeArgsOffset(PreludeSlot::PosedMeshletBounds), ThreadgroupSize::Linear32);
    }
    else encoder->dispatchThreadgroups(*buffers.GeometryWork.Buffer, WorkArgsOffset(pc.Work, true), ThreadgroupSize::Linear32);
}

// Buffer bindings shared by meshlet classification dispatches.
MeshletCullPushConstants MakeMeshletCullSlotsPc(const GpuBuffers &buffers, const MeshletCullOutput &output) {
    return {
        .WorkRangeSlot = buffers.MeshletWorkRanges.Slot,
        .WorkBlockSlot = buffers.MeshletWorkBlocks.Slot,
        .LodNodeSlot = buffers.LodNodes.Buffer.Slot,
        .LodFrontierBlockStateSlot = buffers.LodFrontierBlockStates.Slot,
        .LodExpandArgsSlot = buffers.LodExpandArgs.Slot,
        .WorkStateSlot = buffers.MeshletWorkState.Slot,
        .WorkDispatchArgsSlot = buffers.MeshletWorkDispatchArgs.Slot,
        .BlockStateSlot = buffers.MeshletCullBlocks.Slot,
        .ClassificationSlot = buffers.MeshletClassifications.Slot,
        .VisibleSlot = output.Visible.Slot,
        .InstanceMapSlot = buffers.GpuInstanceSlots.Slot,
        .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
        .PrimitiveSlot = buffers.Primitives.Buffer.Slot,
        .MeshletSlot = buffers.Meshlets.Buffer.Slot,
        .MeshletIndexNodesSlot = buffers.ActiveMeshlets.Nodes.Buffer.Slot,
        .MeshletIndexLeavesSlot = buffers.ActiveMeshlets.Leaves.Buffer.Slot,
        .ClusterGroupSlot = buffers.ClusterGroups.Buffer.Slot,
        .BoundsSlot = buffers.Instances.BoundsBuffer.Slot,
        .ModelSlot = buffers.Instances.TransformBuffer.Slot,
        .PosedMeshletBoundsSlot = buffers.PosedMeshletBounds.Values.Buffer.Slot,
        .PosedMeshletBoundsNodesSlot = buffers.PosedMeshletBounds.Nodes.Buffer.Slot,
        .RouteStateSlot = output.Routes.Slot,
        .DispatchArgsSlot = output.DispatchArgs.Slot,
        .DispatchChunkCount = output.ChunkCount,
        .DispatchChunkSize = GpuBuffers::MeshletDispatchChunkSize,
        .CoarseCountSlot = buffers.MeshletCoarseCount.Slot,
    };
}

// Classifies the work blocks, then compacts each route's entries in deterministic order.
void RecordMeshletCompaction(MTL::ComputeCommandEncoder *encoder, const Pipelines &pipelines, const GpuBuffers &buffers) {
    const auto dispatch_meshlets = [&](const mtl::ComputePipeline &pipeline) {
        encoder->setComputePipelineState(pipeline.State());
        const uint32_t prefix_bytes = GpuBuffers::MeshletRouteCount * (GpuBuffers::MeshletCullBlockSize / 32u + 1u) * sizeof(uint32_t);
        encoder->setThreadgroupMemoryLength(AlignedThreadgroupBytes(prefix_bytes), 0);
        encoder->dispatchThreadgroups(*buffers.MeshletWorkDispatchArgs, 0, MTL::Size(GpuBuffers::MeshletCullBlockSize, 1, 1));
    };
    dispatch_meshlets(pipelines.MeshletCullBlockCount);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    encoder->setComputePipelineState(pipelines.MeshletCullPrefix.State());
    encoder->setThreadgroupMemoryLength(AlignedThreadgroupBytes(GpuBuffers::MeshletRouteCount * sizeof(uint32_t)), 0);
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    dispatch_meshlets(pipelines.MeshletCullEmit);
}

MeshletDrawPushConstants MakeMeshletDrawPc(
    const GpuBuffers &buffers, const MeshletCullOutput &output,
    uint32_t route, uint32_t required_instance_flags,
    bool visibility_transmission, uint32_t edge_sharpness_slot,
    uint32_t edit_edge_corner = 0u, uint32_t instance_filter = InvalidOffset
) {
    return {
        .PrimitiveSlot = buffers.Primitives.Buffer.Slot,
        .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
        .InstanceMapSlot = buffers.GpuInstanceSlots.Slot,
        .MeshletSlot = buffers.Meshlets.Buffer.Slot,
        .MeshletTriangleSlot = buffers.MeshletTriangleIds.Buffer.Slot,
        .MeshletVertexSlot = buffers.MeshletVertexCorners.Buffer.Slot,
        .MeshletLocalTriangleSlot = buffers.MeshletLocalTriangles.Buffer.Slot,
        .VisibleMeshletSlot = output.Visible.Slot,
        .RouteStateSlot = output.Routes.Slot,
        .Route = route,
        .RequiredInstanceFlags = required_instance_flags,
        .InstanceFilter = instance_filter,
        .EditEdgeCorner = edit_edge_corner,
        .VisibilityTransmission = visibility_transmission,
        .EdgeSharpnessSlot = edge_sharpness_slot,
    };
}

void DrawMeshletList(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, uint32_t route, uint32_t required_instance_flags,
    bool visibility_transmission = false, bool fragment_pc = false,
    uint32_t edge_sharpness_slot = InvalidSlot,
    uint32_t mesh_threads = 160u, uint32_t edit_edge_corner = 0u,
    uint32_t instance_filter = InvalidOffset, const MeshletCullOutput *cull = nullptr
) {
    const auto &output = cull ? *cull : buffers.SceneCull;
    // Visibility IDs reserve a fixed bit range for the visible-list index.
    if (fragment_pc) {
        const auto visible_count = output.Visible.Count<VisibleMeshlet>();
        constexpr uint64_t index_limit = uint64_t{1} << uint32_t(VisibilityId::IndexBits);
        profile::RecordCounter("VisibleMeshletIndexOverflow", visible_count > index_limit ? double(visible_count - index_limit) : 0.0);
    }
    auto pc = MakeMeshletDrawPc(
        buffers, output, route, required_instance_flags,
        visibility_transmission, edge_sharpness_slot, edit_edge_corner, instance_filter
    );
    for (uint32_t chunk = 0; chunk < output.ChunkCount; ++chunk) {
        pc.VisibleOffset = chunk * GpuBuffers::MeshletDispatchChunkSize;
        if (fragment_pc) encode::SetPushConstants(encoder, pc);
        else encode::SetMeshPushConstants(encoder, pc);
        const auto args_offset = (route * output.ChunkCount + chunk) * sizeof(MeshDispatchArgs);
        encoder->drawMeshThreadgroups(*output.DispatchArgs, args_offset, MTL::Size(1, 1, 1), MTL::Size(mesh_threads, 1, 1));
    }
}

// Encodes every route with zero-sized dispatch arguments for routes without visible meshlets.
// The routes that raster into the visibility image, each with the cull mode it draws under.
constexpr std::pair<MeshletRoute, MTL::CullMode> VisibilityRoutes[]{
    {MeshletRoute::OpaqueCullBack, MTL::CullModeBack},
    {MeshletRoute::OpaqueCullFront, MTL::CullModeFront},
    {MeshletRoute::OpaqueDoubleSided, MTL::CullModeNone},
    {MeshletRoute::Coverage, MTL::CullModeNone},
};

void DrawVisibilityMeshlets(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, const MainPipeline &main,
    bool transmission
) {
    for (const auto [route, cull] : VisibilityRoutes) {
        (route == MeshletRoute::Coverage ? main.MeshletVisibilityCoverage : main.MeshletVisibilityOpaque).Bind(encoder);
        encoder->setCullMode(cull);
        DrawMeshletList(encoder, buffers, uint32_t(route), 0u, transmission, true);
    }
}

// Reduces a pyramid's levels from `first_level` on.
void RecordDepthPyramid(
    MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const GpuBuffers &buffers, const Pipelines &pipelines,
    const RenderTargets::ResourcesT::Pyramid &pyramid, uint32_t pyramid_sampler, uint32_t first_level,
    uint32_t source_sampler, mtl::Extent2D source_extent, bool nearest, uint32_t ubo_offset
) {
    encode::BindCompute(encoder, pipelines.DepthPyramidReduce, slots, buffers, ubo_offset);
    const auto &mips = pyramid.Mips;
    for (uint32_t base = first_level; base < uint32_t(mips.size()); base += 6) {
        // Add an explicit barrier between bindless mip dependencies.
        if (base > first_level) encoder->memoryBarrier(MTL::BarrierScopeTextures);
        const auto src_extent = base == 0 ? source_extent : mips[base - 1].Extent;
        const DepthPyramidReducePushConstants pc{
            .SrcSamplerSlot = base == 0 ? source_sampler : pyramid_sampler,
            .SrcLod = base == 0 ? 0 : base - 1,
            .SrcWidth = src_extent.Width,
            .SrcHeight = src_extent.Height,
            .DstSlots = [&] {
                std::array<uint32_t, 6> dst;
                for (uint32_t k = 0; k < dst.size(); ++k) dst[k] = base + k < mips.size() ? mips[base + k].Slot : InvalidSlot;
                return dst;
            }(),
            .Nearest = nearest,
        };
        encode::SetPushConstants(encoder, pc);
        encoder->setThreadgroupMemoryLength(ThreadgroupMemory::DepthPyramidTile, 0);
        encoder->dispatchThreadgroups(
            MTL::Size((mips[base].Extent.Width + 31) / 32, (mips[base].Extent.Height + 31) / 32, 1),
            ThreadgroupSize::Tile16
        );
    }
}

// The visibility/depth pair is still intact here; no shading variant needs velocity outputs.
void RecordMotionBlurPostFx(state::Scene &r, state::Entity viewport, mtl::PassChain &chain, uint32_t ubo_offset) {
    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    const auto &buffers = r.Context.get<const GpuBuffers>();
    const auto &main = GetPipelines(r).Main;
    const auto &targets = r.Context.get<const RenderTargets>();
    const auto &blur = *targets.MotionBlur;
    const auto &samplers = r.Context.get<const RenderSamplerSlots>();
    const auto extent = targets.Resources->SceneColorImage.Extent;
    const auto tiles = blur.TileImage.Extent;
    const auto &view = *reinterpret_cast<const SceneViewUBO *>(buffers.SceneViewUBO.Contents().data() + ubo_offset);
    auto *compute = chain.BeginCompute("BlurTiles", MTL::StageFragment);
    encode::BindCompute(compute, main.MotionBlurTilesFlatten, slots, buffers, ubo_offset);
    const auto &previous = *reinterpret_cast<const SceneViewUBO *>(buffers.SceneViewUBO.Contents().data() + buffers.SceneViewUboOffset(1));
    const auto &next = *reinterpret_cast<const SceneViewUBO *>(buffers.SceneViewUBO.Contents().data() + buffers.SceneViewUboOffset(2));
    const uint32_t camera_motion = view.ViewProj != previous.ViewProj || view.ViewProj != next.ViewProj;
    encode::SetPushConstants(compute, MotionBlurTilesFlattenPushConstants{encode::VisibilityDecodePc(buffers), Inverse(view.ViewProj), camera_motion});
    compute->setBuffer(*buffers.SceneViewUBO, buffers.SceneViewUboOffset(1), 5);
    compute->setBuffer(*buffers.SceneViewUBO, buffers.SceneViewUboOffset(2), 6);
    compute->setBuffer(blur.TileIndirection.get(), 0, 7);
    compute->setTexture(*targets.Resources->VisibilityImage, 0);
    compute->setTexture(*targets.Resources->VisibilityDepth, 1);
    compute->setTexture(*blur.VelocityImage, 2);
    compute->setTexture(*blur.TileImage, 3);
    compute->setThreadgroupMemoryLength(16, 0);
    compute->setThreadgroupMemoryLength(16, 1);
    compute->dispatchThreadgroups(MTL::Size(tiles.Width, tiles.Height, 1), ThreadgroupSize::Tile8);
    compute->memoryBarrier(MTL::BarrierScopeTextures | MTL::BarrierScopeBuffers);
    encode::BindCompute(compute, main.MotionBlurTilesDilate, slots, buffers, ubo_offset);
    compute->setBuffer(blur.TileIndirection.get(), 0, 5);
    compute->setTexture(*blur.TileImage, 0);
    compute->dispatchThreadgroups(MTL::Size((tiles.Width + 7) / 8, (tiles.Height + 7) / 8, 1), ThreadgroupSize::Tile8);

    const std::array colors{mtl::DiscardColor(*blur.OutputImage)};
    const auto pass = mtl::MakePassDescriptor(colors);
    auto *render = encode::BeginScenePass(chain, pass.get(), "BlurGather", {{MTL::StageDispatch | MTL::StageFragment, MTL::StageFragment}}, extent, slots, buffers, ubo_offset);
    main.MotionBlurGather.Bind(render);
    render->setFragmentBuffer(blur.TileIndirection.get(), 0, 5);
    render->setFragmentTexture(*blur.TileImage, 0);
    const auto inverse_projection = Inverse(r.get<const ViewCamera>(viewport).Projection(float(extent.Width) / float(extent.Height)));
    const float noise_phase = r.get<const PlaybackFrame>(viewport).Value * std::numbers::phi_v<float>;
    encode::SetPushConstants(render, MotionBlurGatherPushConstants{samplers.SceneDepth, samplers.Velocity, samplers.SceneColor, noise_phase - std::floor(noise_phase), {inverse_projection[2].z, inverse_projection[3].z, inverse_projection[2].w, inverse_projection[3].w}});
    render->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
}

void RecordSparseEditPrelude(state::Scene &, state::Entity, mtl::PassChain &);

// Record one phase's passes into `cb`, which is already begun with viewport and scissor set.
// `ubo_offset` selects the view UBO instance every bind in the phase reads.
void RecordPhase(state::Scene &r, state::Entity viewport, mtl::PassChain &chain, SceneUpdate update, RenderPhase phase, uint32_t ubo_offset, uint32_t sample_weight = 1u) {
    const profile::CpuScope scope{"RecordRenderCommandBuffer"};
    // Multi-step blur separates scene accumulation from sharp overlay rendering.
    const bool draw_scene = phase != RenderPhase::BlurResolve;
    const bool draw_overlays = !IsBlurAccumulate(phase);

    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &meshes = r.Context.get<MeshStore>();
    auto &pipelines = GetPipelines(r);
    auto &targets = r.Context.get<RenderTargets>();
    const auto &settings = r.get<const ViewportDisplay>(viewport);
    const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
    const auto edit_mode = r.get<const EditMode>(viewport).Value;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;
    const bool is_excite_mode = interaction_mode == InteractionMode::Excite;
    const bool is_wireframe_mode = settings.ViewportShading == ViewportShadingMode::Wireframe;
    const bool show_rendered = settings.ViewportShading == ViewportShadingMode::MaterialPreview || settings.ViewportShading == ViewportShadingMode::Rendered;
    const bool show_fill = !is_wireframe_mode;
    const bool xray = XRayActive(settings);
    const float overlay_behind = OverlayBehindOpacity(settings, interaction_mode);
    // Overlays draw through geometry and fade behind it instead of testing against scene depth.
    const bool overlays_through = overlay_behind > 0.f;
    // Overlays need scene depth to occlude or fade unless X-ray shows them at full strength.
    const bool overlay_scene_depth = overlay_behind < 1.f;
    const bool show_overlays = settings.ShowOverlays;
    const auto &active_lighting = GetActivePbrLighting(r, viewport, settings.ViewportShading);
    const bool real_transmission = show_rendered &&
        active_lighting.RealTransmission &&
        pipelines.Main.Compiler.HasFeature(PbrFeature::Transmission);

    const auto &samplers = r.Context.get<const RenderSamplerSlots>();
    auto &scene_state = r.Context.get<GpuSceneState>();

    RecordInputs record_inputs;
    record_inputs.Mix(uint32_t(interaction_mode) | uint32_t(edit_mode) << 8u);
    record_inputs.Mix(buffers.Instances.RecordBuffer.Count<InstanceRecord>());
    // Edit mode uses the rest pose, and reused blur phases use slots built by the rebuild phase.
    const auto mesh_deform_slots = is_edit_mode || update == SceneUpdate::Reuse ?
        std::unordered_map<state::Entity, DeformSlots>{} :
        BuildDeformSlots(r, meshes, record_inputs);
    static const DeformSlots no_deform{};
    const auto get_deform_slots = [&](state::Entity mesh_entity) -> const DeformSlots & {
        if (auto it = mesh_deform_slots.find(mesh_entity); it != mesh_deform_slots.end()) return it->second;
        return no_deform;
    };

    const auto is_silhouette_eligible = [&](state::Entity e) {
        if (!r.all_of<Instance, RenderInstance>(e)) return false;
        const auto buffer_entity = r.get<const Instance>(e).Entity;
        if (!r.valid(buffer_entity) || r.all_of<ObjectExtrasTag>(buffer_entity)) return false;
        // Bones get outlines from BoneWire/BoneSphereWire, not the screen-space silhouette system.
        if (r.all_of<ArmatureObject>(buffer_entity) || r.all_of<BoneJoint>(buffer_entity)) return false;
        const auto *mesh_buffers = TryMeshBuffers(r, buffer_entity);
        return mesh_buffers && mesh_buffers->FaceIndices.Count > 0;
    };
    const auto should_draw_armature_bones = [&](state::Entity armature) {
        if (is_wireframe_mode) return true;
        if (is_edit_mode || interaction_mode == InteractionMode::Pose) return r.all_of<Active>(armature);
        return r.all_of<Selected>(armature);
    };
    const bool show_normals = show_overlays && settings.NormalOverlays != 0u;
    const auto normal_meshes = show_normals ?
        selection::GetSelectedMeshEntities(r) :
        std::unordered_set<state::Entity>{};
    const bool show_face_normals = show_normals &&
        he::ElementMaskContains(settings.NormalOverlays, Element::Face);
    const bool show_vertex_normals = show_normals &&
        he::ElementMaskContains(settings.NormalOverlays, Element::Vertex);
    std::unordered_set<state::Entity> sound_meshes;
    if (is_excite_mode) {
        for (const auto entity : r.view<const Instance, const SoundVertices>()) {
            sound_meshes.insert(r.get<const Instance>(entity).Entity);
        }
    }

    selection::PrimaryEditInstanceMap primary_edit_instances, transform_instances;
    const bool has_pending_transform = is_edit_mode && r.all_of<PendingTransform>(viewport);
    record_inputs.Mix(has_pending_transform);
    if (is_edit_mode) {
        if (has_pending_transform) {
            auto primaries = selection::ComputePrimaryEditInstanceMaps(r);
            primary_edit_instances = std::move(primaries.All);
            transform_instances = std::move(primaries.Transformable);
        } else {
            primary_edit_instances = selection::ComputePrimaryEditInstances(r);
        }
    }
    std::unordered_set<state::Entity> silhouette_instances;
    if (is_edit_mode) {
        for (const auto [e, instance, ri] : r.view<const Instance, const Selected, const RenderInstance>().each()) {
            if (!is_silhouette_eligible(e)) continue;
            if (auto it = primary_edit_instances.find(instance.Entity); it == primary_edit_instances.end() || it->second != e) {
                silhouette_instances.insert(e);
            }
        }
    }
    if (update == SceneUpdate::Rebuild) {
        const profile::CpuScope build_scope{"UpdateGpuScene"};
        scene_state.PosedByEntity.clear();

        struct MeshEntityData {
            state::Entity Entity;
            const MeshBuffers &Buf;
            const ModelsBuffer &Mod;
            std::optional<Mesh> MeshComp;
            const DeformSlots &Deform;
            std::optional<uint32_t> PrimaryEditBufferIndex;
        };

        // Sort by descending entity ID for deterministic coincident-surface ordering across scene loads.
        const auto mesh_entity_order = SortedEntities(r.view<const ModelsBuffer>(), std::ranges::greater{});

        std::vector<MeshEntityData> mesh_entities;
        mesh_entities.reserve(mesh_entity_order.size());
        for (const auto entity : mesh_entity_order) {
            const auto *mesh_buffers = TryMeshBuffers(r, entity);
            if (!mesh_buffers) continue;
            const auto &models = r.get<const ModelsBuffer>(entity);
            std::optional<uint32_t> primary_bi;
            if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                primary_bi = r.get<RenderInstance>(it->second).BufferIndex;
            }
            mesh_entities.emplace_back(
                entity, *mesh_buffers, models, TryGetMesh(r, entity), get_deform_slots(entity), primary_bi
            );
        }

        // The mesh shades authored under morphing: rest normals plus weighted authored deltas.
        // Edit mode builds no deform slots, so edit-mode draws (including drags) derive.
        const auto morph_shading_authored = [&meshes](const MeshEntityData &e) {
            return e.Deform.MorphDeformOffset != InvalidOffset && e.MeshComp && meshes.Get(e.MeshComp->GetStoreId()).MorphShadingAuthored;
        };

        { // Bounds reduce entries.
            // Instances sharing one deform state share one entry, whose ElementIdOffset spans their consecutive slots.
            // Entries with morph, armature, or pending edit-transform deformation own poses.
            // Posed entries share a position namespace.
            // The same pass writes leaf bounds.
            struct BoundsEntrySpec {
                uint32_t Count{}, NormalVertexTiles{}, NormalFaceTiles{};
                uint64_t VertexLayoutRevision{}, FaceLayoutRevision{};
                bool PerInstanceDeform{}, Posed{}, Derive{};
                const RenderInstance *PendingPrimary{};
                NormalDeriveEntry Entry{}; // Derive-input fields, filled when Derive.
                PosedNamespaces Pose;
                const VertexBoundsStore::Keys *BoundsKeys{};
            };
            std::vector<BoundsEntrySpec> specs(mesh_entities.size());
            // Leaf work covers canonical blocks.
            // Parent work has three fixed levels.
            uint32_t entry_count = 0;
            uint32_t derive_entry_count = 0;
            std::array<uint32_t,VertexBoundsLevels> bounds_tile_counts{};
            uint32_t derive_face_tile_count = 0, derive_gather_tile_count = 0;
            uint32_t posed_meshlet_bounds_count = 0;
            RecordInputs prelude_layout, prelude_work;
            buffers.VertexBounds.BeginUpdate();
            buffers.PosedPositions.BeginUpdate();
            buffers.PosedMorphNormalDeltas.BeginUpdate();
            buffers.PosedVertexNormals.BeginUpdate();
            buffers.PosedFaceNormals.BeginUpdate();
            buffers.PosedSectors.BeginUpdate();
            buffers.PosedMeshletBounds.BeginUpdate();
            for (size_t mi = 0; mi < mesh_entities.size(); ++mi) {
                const auto &e = mesh_entities[mi];
                auto &spec = specs[mi];
                // Every mesh-keyed value an instance record reads, in the order the meshes come.
                record_inputs.Mix(state::Integral(e.Entity));
                record_inputs.Mix(e.Buf.PrimitiveRoot);
                record_inputs.Mix(buffers.PrimitiveCount(e.Buf));
                record_inputs.Mix(e.Buf.MeshletRoot);
                record_inputs.Mix(buffers.MeshletCount(e.Buf));
                record_inputs.Mix(e.Buf.Vertices.Count);
                // Face presence controls silhouette eligibility.
                record_inputs.Mix(e.Buf.FaceIndices.Count);
                record_inputs.Mix(e.Mod.InstanceRange.Offset);
                record_inputs.Mix(e.Mod.InstanceCount);
                record_inputs.Mix(e.Deform.BoneDeformOffset);
                record_inputs.Mix(e.Deform.ArmatureDeformOffset);
                record_inputs.Mix(e.Deform.MorphDeformOffset);
                record_inputs.Mix(e.Deform.MorphTargetCount);
                record_inputs.Mix(e.PrimaryEditBufferIndex.value_or(InvalidOffset));
                if (e.MeshComp) record_inputs.Mix(meshes.GetEditSelectionStorage(e.MeshComp->GetStoreId()));
                record_inputs.Mix(sound_meshes.contains(e.Entity));
                if (buffers.MeshletCount(e.Buf) != 0u) {
                    record_inputs.Mix(buffers.Meshlets.Buffer.GetSpan<MeshletRecord>({buffers.FirstMeshlet(e.Buf), 1}).front().LocalTriangleOffset);
                }
                if (!e.MeshComp || e.Mod.InstanceCount == 0) continue;
                if (has_pending_transform) {
                    if (const auto it = transform_instances.find(e.Entity); it != transform_instances.end()) {
                        spec.PendingPrimary = r.try_get<const RenderInstance>(it->second);
                    }
                }
                spec.PerInstanceDeform = !e.Deform.ArmatureDeformByBufferIndex.empty() || !e.Deform.MorphWeightsByBufferIndex.empty();
                spec.Count = spec.PerInstanceDeform ? e.Mod.InstanceCount : 1u;
                // Some canonical position edits need posed bounds until their
                // meshlets are refitted. Inset previews refit them directly.
                spec.Posed = e.Deform.BoneDeformOffset != InvalidOffset || e.Deform.MorphDeformOffset != InvalidOffset ||
                    (is_edit_mode && ((has_pending_transform && e.PrimaryEditBufferIndex.has_value()) ||
                        (scene_state.EditWork.contains(e.Entity) && scene_state.EditWork.at(e.Entity).RequiresPose)));
                entry_count += spec.Count;
                if (spec.Posed) {
                    spec.Pose.FirstInstance = e.Mod.InstanceRange.Offset;
                    spec.Pose.PerInstance = spec.PerInstanceDeform;
                    const auto store_id=e.MeshComp->GetStoreId();
                    const auto &vertex_arena=meshes.Arenas().Vertices;
                    const auto vertex_set=meshes.Get(store_id).Vertices;
                    const auto vertex_revision=vertex_set ? vertex_arena.Set(vertex_set).Revision : 0u;
                    const auto vertex_bounds=buffers.VertexBounds.Prepare(e.Entity,store_id,vertex_revision,spec.Count,
                        [&] { return PoseElementBlocks(vertex_arena,vertex_set); });
                    spec.BoundsKeys=&vertex_bounds.Nodes;
                    spec.VertexLayoutRevision=vertex_bounds.LayoutRevision;
                    spec.Pose.VertexBoundsNamespaces.assign(vertex_bounds.Roots.begin(),vertex_bounds.Roots.end());
                    for (uint32_t level=0u; level<VertexBoundsLevels; ++level)
                        bounds_tile_counts[level] += spec.Count*uint32_t(vertex_bounds.Nodes[level].size());
                    if (vertex_bounds.Changed) buffers.PreludeStale=true;
                    // Authored morph shading reads base normals.
                    const bool authored_morph = morph_shading_authored(e);
                    const auto &arena = meshes.Arenas().Vertices;
                    const auto set = meshes.Get(store_id).Vertices;
                    const auto vertex_blocks = [&] { return PoseElementBlocks(arena,set); };
                    const auto positions = buffers.PosedPositions.Prepare(e.Entity,store_id,set ? arena.Set(set).Revision : 0u,spec.Count,vertex_blocks);
                    spec.Pose.PositionNamespaces.assign(positions.Roots.begin(),positions.Roots.end());
                    for (const auto root : positions.Roots) record_inputs.Mix(root);
                    if (positions.Changed) buffers.PreludeStale = true;
                    if (authored_morph) {
                        const auto deltas = buffers.PosedMorphNormalDeltas.Prepare(e.Entity,store_id,set ? arena.Set(set).Revision : 0u,spec.Count,vertex_blocks);
                        spec.Pose.MorphNormalNamespaces.assign(deltas.Roots.begin(),deltas.Roots.end());
                        for (const auto root : deltas.Roots) record_inputs.Mix(root);
                        if (deltas.Changed) buffers.PreludeStale = true;
                    }
                    if (const auto derive_entry = authored_morph ? std::nullopt : MakeDeriveEntryInputs(meshes, e.MeshComp->GetStoreId())) {
                        spec.Derive = true;
                        spec.Entry = *derive_entry;
                        const auto vertex_normals = buffers.PosedVertexNormals.Prepare(e.Entity,store_id,set ? arena.Set(set).Revision : 0u,spec.Count,vertex_blocks);
                        const auto &faces = meshes.Arenas().FaceTriangles;
                        const auto face_set = meshes.Get(store_id).FaceData;
                        const auto face_normals = buffers.PosedFaceNormals.Prepare(e.Entity,store_id,faces.Set(face_set).Revision,spec.Count,
                            [&] { return PoseElementBlocks(faces,face_set); });
                        spec.FaceLayoutRevision=face_normals.LayoutRevision;
                        const auto prepared = buffers.PosedSectors.Prepare(e.Entity, store_id, meshes.GetDerived(store_id).NormalRevision, spec.Count,
                            [&] { return NormalPayloadBlocks(meshes, store_id); });
                        for (uint32_t i = 0u; i < spec.Count; ++i) {
                            spec.Pose.Normals.push_back({vertex_normals.Roots[i],prepared.Roots[i],face_normals.Roots[i]});
                            record_inputs.Mix(vertex_normals.Roots[i]);
                            record_inputs.Mix(prepared.Roots[i]);
                            record_inputs.Mix(face_normals.Roots[i]);
                        }
                        if (vertex_normals.Changed || face_normals.Changed || prepared.Changed) buffers.PreludeStale = true;
                        derive_entry_count += spec.Count;
                        spec.NormalVertexTiles = arena.Set(set).BlockCount;
                        spec.NormalFaceTiles = faces.Set(face_set).BlockCount;
                        derive_face_tile_count += spec.Count * spec.NormalFaceTiles;
                        derive_gather_tile_count += spec.Count * spec.NormalVertexTiles;
                    }
                    const auto bounds = buffers.PosedMeshletBounds.Prepare(e.Entity,e.MeshComp->GetStoreId(),e.Buf.MeshletRevision,spec.Count,[&] {
                        std::vector<uint32_t> blocks;
                        buffers.ForEachPrimitive(e.Buf,[&](uint32_t, const PrimitiveRecord &primitive) {
                            if (primitive.LodFinestNode == InvalidOffset) return;
                            const auto root = buffers.LodNodes.Get({primitive.LodFinestNode,1u})[0].MeshletRoot;
                            buffers.ActiveMeshlets.ForEachBlock(root,[&](uint32_t b) { blocks.push_back(b); });
                        });
                        return blocks;
                    },e.Buf.RenderTopology);
                    spec.Pose.MeshletBoundsNamespaces.assign(bounds.Roots.begin(),bounds.Roots.end());
                    for (const auto root : bounds.Roots) record_inputs.Mix(root);
                    if (bounds.Changed) buffers.PreludeStale = true;
                    posed_meshlet_bounds_count += spec.Count*e.Buf.Level0Count;
                }
                if (!spec.Posed) bounds_tile_counts.back()+=spec.Count;
                prelude_layout.Mix(state::Integral(e.Entity));
                prelude_layout.Mix(e.MeshComp->GetStoreId());
                prelude_layout.Mix(spec.Count);
                prelude_layout.Mix(uint32_t(spec.Posed) | uint32_t(spec.Derive) << 1u);
                prelude_layout.Mix(spec.VertexLayoutRevision);
                prelude_layout.Mix(spec.FaceLayoutRevision);
                prelude_layout.Mix(spec.NormalVertexTiles);
                prelude_layout.Mix(spec.NormalFaceTiles);
                prelude_work.Mix(state::Integral(e.Entity));
                prelude_work.Mix(e.Buf.MeshletRevision);
                prelude_work.Mix(e.Buf.Vertices.Count);
                prelude_work.Mix(spec.Entry.FaceCount);
                prelude_work.Mix(e.Mod.InstanceRange.Offset);
                prelude_work.Mix(e.Mod.InstanceCount);
                // Pose sharing and normal dispatch also depend on these inputs.
                record_inputs.Mix(spec.Count);
                record_inputs.Mix(uint32_t(spec.Posed) | uint32_t(spec.Derive) << 1u | uint32_t(spec.PerInstanceDeform) << 2u);
                record_inputs.Mix(spec.PendingPrimary ? spec.PendingPrimary->BufferIndex : InvalidOffset);
                record_inputs.Mix(spec.Entry.FaceCount);
                record_inputs.Mix(spec.Entry.VertexCount);
            }
            buffers.VertexBounds.EndUpdate();
            buffers.PosedPositions.EndUpdate();
            buffers.PosedMorphNormalDeltas.EndUpdate();
            buffers.PosedVertexNormals.EndUpdate();
            buffers.PosedFaceNormals.EndUpdate();
            buffers.PosedSectors.EndUpdate();
            buffers.PosedMeshletBounds.EndUpdate();
            const bool tiles_changed = prelude_layout.Value != scene_state.PreludeLayoutInputs;
            scene_state.PreludeLayoutInputs = prelude_layout.Value;
            if (prelude_work.Value != scene_state.PreludeWorkInputs) buffers.PreludeStale = true;
            scene_state.PreludeWorkInputs = prelude_work.Value;
            if (tiles_changed) buffers.PreludeStale = true;
            const auto entries = buffers.BoundsReduceEntries.SetCount<BoundsEntry>(entry_count);
            const auto derive_entries = buffers.NormalDeriveEntries.SetCount<NormalDeriveEntry>(derive_entry_count);
            uint32_t bounds_tile_count=0u;
            for (uint32_t level=0u; level<VertexBoundsLevels; ++level) {
                buffers.BoundsFirstTiles[level]=bounds_tile_count;
                bounds_tile_count+=bounds_tile_counts[level];
            }
            const auto bounds_tiles = buffers.BoundsTiles.SetCount<uvec2>(bounds_tile_count);
            const auto derive_tiles = buffers.DeriveTiles.SetCount<uvec2>(derive_face_tile_count + derive_gather_tile_count);
            std::vector<PosedMeshletBoundsJob> meshlet_jobs;
            buffers.Prelude = {
                .PosePrepass = bounds_tile_counts[0],
                .PosedMeshletBounds = posed_meshlet_bounds_count,
                .DeriveFaces = derive_face_tile_count,
                .DeriveGather = derive_gather_tile_count,
                .BoundsCombine = {bounds_tile_counts[1],bounds_tile_counts[2],bounds_tile_counts[3]},
            };

            uint32_t write = 0, derive_write = 0;
            uint32_t face_tile_write = 0, gather_tile_write = derive_face_tile_count;
            auto bounds_tile_write=buffers.BoundsFirstTiles;
            uint32_t meshlet_groups = 0u;
            for (size_t mi = 0; mi < mesh_entities.size(); ++mi) {
                const auto &e = mesh_entities[mi];
                auto &spec = specs[mi];
                if (spec.Count == 0) continue;
                BoundsEntry entry{
                    .FirstInstance = e.Mod.InstanceRange.Offset,
                    .InstanceCount = spec.PerInstanceDeform ? 1u : e.Mod.InstanceCount,
                    .Selection = meshes.GetEditSelectionStorage(e.MeshComp->GetStoreId()),
                    .VertexBlocksSlot = meshes.Arenas().Vertices.Blocks.Buffer.Slot,
                    .VertexOwner = meshes.Get(e.MeshComp->GetStoreId()).Vertices.Index,
                    .VertexRoot = meshes.Get(e.MeshComp->GetStoreId()).SelectionSummary.Count ? meshes.GetSelectionRoots(e.MeshComp->GetStoreId()) : SlotOffset{},
                };
                auto &pr = spec.Pose;
                NormalDeriveEntry derive_entry = spec.Entry;
                std::vector<uint32_t> face_blocks, vertex_blocks;
                if (tiles_changed && spec.Derive) {
                    const auto &record = meshes.Get(e.MeshComp->GetStoreId());
                    face_blocks = PoseElementBlocks(meshes.Arenas().FaceTriangles,record.FaceData);
                    vertex_blocks = PoseElementBlocks(meshes.Arenas().Vertices,record.Vertices);
                    assert(face_blocks.size() == spec.NormalFaceTiles && vertex_blocks.size() == spec.NormalVertexTiles);
                }
                for (uint32_t i = 0; i < spec.Count; ++i) {
                    if (const auto normals = pr.NormalsAt(i)) {
                        derive_entry.PositionNamespace = pr.PositionNamespace(i);
                        derive_entry.VertexNormalNamespace = normals->Vertex;
                        derive_entry.SectorNamespace = normals->Sector;
                        derive_entry.FaceNormalNamespace = normals->Face;
                        if (tiles_changed) {
                            for (const auto block : face_blocks) derive_tiles[face_tile_write++] = {derive_write,block};
                            for (const auto block : vertex_blocks) derive_tiles[gather_tile_write++] = {derive_write,block};
                        }
                        derive_entries[derive_write++] = derive_entry;
                    }
                    if (tiles_changed) {
                        if (spec.Posed) {
                            for (uint32_t level=0u; level<VertexBoundsLevels; ++level)
                                for (const auto key : (*spec.BoundsKeys)[level]) bounds_tiles[bounds_tile_write[level]++]={write,key};
                        } else bounds_tiles[bounds_tile_write.back()++]={write,0u};
                    }
                    auto instance_entry=entry;
                    if (spec.PerInstanceDeform) instance_entry.FirstInstance+=i;
                    if (spec.Posed) instance_entry.BoundsNamespace=spec.Pose.VertexBoundsNamespaces[i];
                    entries[write++]=instance_entry;
                }
                if (spec.Posed) scene_state.PosedByEntity.emplace(e.Entity,std::move(pr));
                // Descriptors enumerate canonical clusters on the GPU.
                // All instances sharing a pose use the same bounds namespace.
                if (spec.Posed) for (uint32_t i = 0u; i < spec.Count; ++i) {
                    const auto instance = spec.PerInstanceDeform ? entry.FirstInstance+i : entry.FirstInstance;
                    buffers.ForEachPrimitive(e.Buf,[&](uint32_t, const PrimitiveRecord &primitive) {
                        if (primitive.LodFinestNode == InvalidOffset) return;
                        const auto root = buffers.LodNodes.Get({primitive.LodFinestNode,1u})[0].MeshletRoot;
                        const auto count = buffers.ActiveMeshlets.Count(root);
                        if (!count) return;
                        meshlet_jobs.push_back({meshlet_groups,count,instance,buffers.ActiveMeshlets.Ref(root)});
                        meshlet_groups += count;
                    });
                }
            }
            assert(meshlet_groups == posed_meshlet_bounds_count);
            if (tiles_changed) assert(face_tile_write == derive_face_tile_count && gather_tile_write == derive_tiles.size());
            const auto jobs = buffers.PosedMeshletBoundsJobs.SetCount<PosedMeshletBoundsJob>(meshlet_jobs.size());
            std::ranges::copy(meshlet_jobs,jobs.begin());
        }

        // Reuse records and the topology mask when all hashed inputs match the previous rebuild.
        if (record_inputs.Value != scene_state.InstanceRecordInputs) {
            scene_state.InstanceRecordInputs = record_inputs.Value;
            MarkInstanceRecordsStale(scene_state);
        }
        if (scene_state.InstanceRecordsStale) {
            // Coplanar visibility follows entity order, independent of instance-buffer allocation history.
            const auto instance_order = SortedEntities(
                r.view<const RenderInstance>() |
                    std::views::filter([&](auto e) { return r.get<const RenderInstance>(e).MeshletCount > 0; }),
                std::ranges::greater{}
            );
            auto instance_slots = buffers.GpuInstanceSlots.SetCount<uint32_t>(uint32_t(instance_order.size()));
            for (uint32_t i = 0; i < instance_slots.size(); ++i) {
                instance_slots[i] = r.get<const RenderInstance>(instance_order[i]).BufferIndex;
            }

            buffers.MeshletTopologyMask = 0u;
            for (const auto [instance_entity, instance, ri] : r.view<const Instance, const RenderInstance>().each()) {
                if (ri.BufferIndex == UINT32_MAX) continue;
                const auto *mesh_buffers = TryMeshBuffers(r, instance.Entity);
                if (!mesh_buffers || buffers.PrimitiveCount(*mesh_buffers) == 0) continue;
                if (buffers.MeshletCount(*mesh_buffers) != 0u) {
                    const MeshletRecord &first_meshlet = buffers.Meshlets.Buffer.GetSpan<MeshletRecord>(
                                                                                    {buffers.FirstMeshlet(*mesh_buffers), 1u}
                    )
                                                             .front();
                    const uint32_t topology = first_meshlet.Topology;
                    buffers.MeshletTopologyMask |= 1u << topology;
                }
                InstanceRecord record{
                    .PrimitiveRoot = mesh_buffers->PrimitiveRoot,
                    .PrimitiveCount = buffers.PrimitiveCount(*mesh_buffers),
                    .Mesh = mesh_buffers->MeshRecord.Offset,
                    .ObjectId = ObjectId(instance_entity),
                };
                const auto &deform = get_deform_slots(instance.Entity);
                record.BoneDeformOffset = deform.BoneDeformOffset;
                record.ArmatureDeformOffset = deform.ArmatureDeformOffset;
                record.MorphDeformOffset = deform.MorphDeformOffset;
                record.MorphTargetCount = deform.MorphTargetCount;
                if (const auto it = deform.ArmatureDeformByBufferIndex.find(ri.BufferIndex); it != deform.ArmatureDeformByBufferIndex.end()) {
                    record.ArmatureDeformOffset = it->second;
                }
                if (const auto it = deform.MorphWeightsByBufferIndex.find(ri.BufferIndex); it != deform.MorphWeightsByBufferIndex.end()) {
                    record.MorphWeightsOffset = it->second;
                }
                if (const auto it = scene_state.PosedByEntity.find(instance.Entity); it != scene_state.PosedByEntity.end()) {
                    const auto &posed = it->second;
                    const auto i = posed.PerInstance ? ri.BufferIndex - posed.FirstInstance : 0u;
                    record.PositionNamespace = posed.PositionNamespace(i);
                    record.MorphNormalNamespace = posed.MorphNormalNamespace(i);
                    record.MeshletBoundsNamespace = posed.MeshletBoundsNamespace(i);
                    if (const auto normals = posed.NormalsAt(i)) {
                        record.VertexNormalNamespace = normals->Vertex;
                        record.SectorNamespace = normals->Sector;
                        record.FaceNormalNamespace = normals->Face;
                    }
                }
                const auto primary = primary_edit_instances.find(instance.Entity);
                if (has_pending_transform && primary != primary_edit_instances.end()) {
                    record.HasPendingVertexTransform = 1u;
                    record.PrimaryEditInstanceIndex = r.get<const RenderInstance>(primary->second).BufferIndex;
                }
                if (primary != primary_edit_instances.end() && primary->second == instance_entity) {
                    const uint32_t store_id = GetMesh(r, instance.Entity).GetStoreId();
                    record.Selection = meshes.GetEditSelectionStorage(store_id);
                    record.EditEdgeSharpnessOffset = meshes.Arenas().EdgeHalfedges.First(meshes.Get(store_id).EdgeData);
                    record.ElementIdOffset = meshes.GetSelectionBitOffset(store_id, edit_mode);
                } else if (is_excite_mode && sound_meshes.contains(instance.Entity)) {
                    const uint32_t store_id = GetMesh(r, instance.Entity).GetStoreId();
                    record.Selection = meshes.GetEditSelectionStorage(store_id);
                    const auto *active = r.try_get<const MeshActiveElement>(instance.Entity);
                    const auto *force = r.try_get<const VertexForce>(instance_entity);
                    record.ActiveVertex = active ? active->Handle : InvalidOffset;
                    record.ExcitedVertex = force ? force->Vertex : InvalidOffset;
                }
                buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>({ri.BufferIndex, 1}).front() = record;
            }
            scene_state.InstanceRecordsStale = false;
            // A fresh record carries no flags, so the object id and silhouette pass below must run.
            scene_state.InstanceFlagsStale = true;
        }

        scene_state.MeshletEditOverlayMeshes.clear();
        scene_state.MeshletEditHasSharpEdges = false;
        const bool meshlet_edit_overlay = show_overlays && is_edit_mode && draw_overlays;
        if (meshlet_edit_overlay) {
            for (const auto &e : mesh_entities) {
                if (!e.PrimaryEditBufferIndex || !e.MeshComp || buffers.MeshletCount(e.Buf) == 0u) continue;
                scene_state.MeshletEditOverlayMeshes.insert(e.Entity);
                scene_state.MeshletEditHasSharpEdges |= meshes.GetEdgeSharpnessSummary(e.MeshComp->GetStoreId()).Any;
            }
        }
        scene_state.InstanceFlagsStale = true;
        // Publish overlay jobs only after all RenderInstance slots are final.
        buffers.SetOverlayJobs(BuildOverlayJobs(r));
    }
    // Object ids and silhouette flags, with the silhouette cull's work totalled as the flags land.
    if (scene_state.InstanceFlagsStale) {
        const auto instance_records = buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>(
            {0, buffers.Instances.RecordBuffer.Count<InstanceRecord>()}
        );
        GpuBuffers::MeshletFlagWork silhouette_work{}, edit_overlay_work{}, element_selection_work{}, wire_work{};
        for (const auto flag : {
                 MeshletInstanceFlag::Bone,
                 MeshletInstanceFlag::BoneWire,
                 MeshletInstanceFlag::BoneJoint,
                 MeshletInstanceFlag::BoneJointWire,
                 MeshletInstanceFlag::FaceNormal,
                 MeshletInstanceFlag::EdgeOverlay,
             }) {
            buffers.FlagWork(uint32_t(flag)) = {};
        }
        scene_state.VertexOverlays.clear();
        for (const auto [instance_entity, ri] : r.view<const RenderInstance>().each()) {
            if (ri.BufferIndex == UINT32_MAX || ri.BufferIndex >= instance_records.size()) continue;
            auto &record = instance_records[ri.BufferIndex];
            record.ObjectId = ObjectId(instance_entity);
            const bool selected = r.all_of<Selected>(instance_entity) && is_silhouette_eligible(instance_entity);
            const bool silhouette = selected && (!is_edit_mode || silhouette_instances.contains(instance_entity));
            record.Flags = silhouette ? uint32_t(MeshletInstanceFlag::Silhouette) : 0u;
            const auto *instance = r.try_get<const Instance>(instance_entity);
            const auto *mesh_buffers = instance ? TryMeshBuffers(r, instance->Entity) : nullptr;
            if (instance && (EditPinsFinest(primary_edit_instances, scene_state, instance->Entity) ||
                (mesh_buffers && buffers.ActiveMeshlets.Count(mesh_buffers->PositionDirtyRoot) > 0u))) {
                record.Flags |= uint32_t(MeshletInstanceFlag::LodPinFinest);
            }
            const auto primary = instance ? primary_edit_instances.find(instance->Entity) : primary_edit_instances.end();
            if (instance && mesh_buffers && buffers.MeshletCount(*mesh_buffers) > 0u &&
                primary != primary_edit_instances.end() && primary->second == instance_entity &&
                GetMesh(r, instance->Entity).ElementCount(edit_mode) > 0u) {
                record.Flags |= uint32_t(MeshletInstanceFlag::ElementSelection);
                if (ri.MeshletCount > 0) {
                    element_selection_work.Nodes += ri.LodNodeCount;
                    element_selection_work.Meshlets += ri.MeshletCount;
                }
            }
            if (instance && scene_state.MeshletEditOverlayMeshes.contains(instance->Entity) &&
                primary != primary_edit_instances.end() && primary->second == instance_entity) {
                record.Flags |= uint32_t(MeshletInstanceFlag::EditOverlay);
                scene_state.VertexOverlays.push_back({instance->Entity, ri.BufferIndex, VertexOverlay::EditPoints});
                if (ri.MeshletCount > 0) {
                    edit_overlay_work.Nodes += ri.LodNodeCount;
                    edit_overlay_work.Meshlets += ri.MeshletCount;
                }
            }
            const auto mesh = instance ? TryGetMesh(r, instance->Entity) : std::nullopt;
            const bool shaded_face_less = mesh && show_rendered && mesh->FaceCount() == 0u &&
                meshes.Get(mesh->GetStoreId()).PrimitiveMaterials.Count > 0u;
            const bool wire = instance && mesh_buffers && buffers.MeshletCount(*mesh_buffers) > 0u &&
                !r.all_of<ArmatureObject>(instance->Entity) && !r.all_of<BoneJoint>(instance->Entity) &&
                !r.all_of<ObjectExtrasTag>(instance->Entity) && mesh && mesh->EdgeCount() > 0u &&
                (mesh->FaceCount() == 0u || is_wireframe_mode) && !shaded_face_less;
            if (wire) {
                record.Flags |= uint32_t(MeshletInstanceFlag::Wire);
                if (ri.MeshletCount > 0) {
                    wire_work.Nodes += ri.LodNodeCount;
                    wire_work.Meshlets += ri.MeshletCount;
                }
            }
            const bool bone = instance && r.all_of<ArmatureObject>(instance->Entity);
            const bool joint = instance && r.all_of<BoneJoint>(instance->Entity);
            if (bone || joint) record.Flags |= uint32_t(MeshletInstanceFlag::OverlayOnly);
            const auto mark = [&](MeshletInstanceFlag flag) {
                record.Flags |= uint32_t(flag);
                if (ri.MeshletCount > 0) {
                    auto &work = buffers.FlagWork(uint32_t(flag));
                    work.Nodes += ri.LodNodeCount;
                    work.Meshlets += ri.MeshletCount;
                }
            };
            if (show_overlays && settings.ShowBones) {
                if (bone) {
                    mark(MeshletInstanceFlag::Bone);
                    if (should_draw_armature_bones(instance->Entity)) mark(MeshletInstanceFlag::BoneWire);
                } else if (joint) {
                    mark(MeshletInstanceFlag::BoneJoint);
                    const auto *part = r.try_get<const BoneSubPartOf>(instance_entity);
                    const auto *owner = part ? r.try_get<const SubElementOf>(part->BoneEntity) : nullptr;
                    if (!owner || should_draw_armature_bones(owner->Parent)) mark(MeshletInstanceFlag::BoneJointWire);
                }
            }
            if (instance && mesh_buffers && normal_meshes.contains(instance->Entity)) {
                if (show_face_normals && mesh_buffers->FaceIndices.Count > 0u) {
                    mark(MeshletInstanceFlag::FaceNormal);
                }
                if (show_vertex_normals && mesh && mesh->EdgeCount() > 0u) {
                    record.Flags |= uint32_t(MeshletInstanceFlag::LodPinFinest);
                    scene_state.VertexOverlays.push_back({instance->Entity, ri.BufferIndex, VertexOverlay::Normals});
                }
            }
            if (instance && mesh_buffers && show_overlays && is_excite_mode &&
                sound_meshes.contains(instance->Entity) && mesh && mesh->EdgeCount() > 0u) {
                mark(MeshletInstanceFlag::EdgeOverlay);
            }
            if (instance && mesh_buffers && show_overlays && is_excite_mode &&
                sound_meshes.contains(instance->Entity) && buffers.MeshletCount(*mesh_buffers) > 0u) {
                scene_state.VertexOverlays.push_back({instance->Entity, ri.BufferIndex, VertexOverlay::SoundPoints});
            }
            const bool point_overlay = instance && mesh && mesh_buffers &&
                mesh->PrimitiveTopology() == uint32_t(MeshPrimitiveTopology::Point) &&
                !primary_edit_instances.contains(instance->Entity) && !shaded_face_less;
            if (point_overlay) scene_state.VertexOverlays.push_back({instance->Entity, ri.BufferIndex, VertexOverlay::Points});
            if (silhouette && ri.MeshletCount > 0) {
                silhouette_work.Nodes += ri.LodNodeCount;
                silhouette_work.Meshlets += ri.MeshletCount;
            }
        }
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::Silhouette)) = silhouette_work;
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::EditOverlay)) = edit_overlay_work;
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::ElementSelection)) = element_selection_work;
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::Wire)) = wire_work;
        scene_state.InstanceFlagsStale = false;
    }
    const bool has_object_silhouette_selection =
        any_of(r.view<const Selected, const Instance, const RenderInstance>().each(), [&](const auto &entry) { return is_silhouette_eligible(std::get<0>(entry)); });
    const bool render_silhouette = (show_overlays && settings.ShowOutlineSelected) && !is_excite_mode &&
        (is_edit_mode ? !silhouette_instances.empty() : has_object_silhouette_selection);

    // Specialize forward PBR during the authoritative rebuild scan to avoid a second registry traversal.
    if (show_rendered && update != SceneUpdate::Reuse) {
        pipelines.Main.Compiler.CompileTopologyPipelines((buffers.MeshletTopologyMask & ~1u) != 0u);
    }
    if (update != SceneUpdate::Reuse || phase == RenderPhase::Full) RecordSceneCounters(buffers);

    const bool transmission_active = real_transmission && targets.Transmission;
    // Reuse opaque transmission shading when neither edit tint nor debug output needs another shade.
    const bool composite_transmission = transmission_active && phase == RenderPhase::Full && !is_edit_mode && settings.DebugChannel == DebugChannel::None;
    const bool meshlet_fill = buffers.MeshletInstanceCount > 0;

    // Posed positions and bounds run before culling.
    // Every prelude pass dispatches indirectly.
    // A submit with unchanged deform inputs gets zero group counts, keeping the buffers' current results.
    if (buffers.Prelude.HasWork()) {
        const auto &prelude = buffers.Prelude;
        const bool bounds_work = prelude.BoundsCombine[2] > 0;
        auto *compute = chain.BeginCompute("Prelude", MTL::StageVertex | MTL::StageFragment | MTL::StageDispatch);
        // Bindless dependencies require explicit barriers between pose and bounds levels.
        if (prelude.PosePrepass > 0) {
            RecordPosePrepass(compute, slots, pipelines, buffers, ubo_offset);
            compute->memoryBarrier(MTL::BarrierScopeBuffers);
        }
        if (prelude.DeriveFaces > 0) {
            auto derive = MakeNormalDerivePc(buffers,meshes,buffers.PosedVertexNormals.Values.Buffer.Slot,buffers.PosedFaceNormals.Values.Buffer.Slot);
            const auto &normal_pipeline = GetMeshPipelines(r)[MeshPass::VertexNormalDerive];
            RecordNormalDerive(compute,slots,normal_pipeline,buffers,derive,PreludeSlot::DeriveFaces,ubo_offset);
            compute->memoryBarrier(MTL::BarrierScopeBuffers);
            derive.Phase = 1u;
            derive.FirstTile = prelude.DeriveFaces;
            RecordNormalDerive(compute,slots,normal_pipeline,buffers,derive,PreludeSlot::DeriveGather,ubo_offset);
            compute->memoryBarrier(MTL::BarrierScopeBuffers);
        }
        if (prelude.PosedMeshletBounds > 0) RecordPosedMeshletBounds(compute, slots, pipelines, buffers, ubo_offset);
        if (bounds_work) {
            RecordBoundsPass(compute, slots, pipelines.BoundsCombine, buffers, PreludeSlot::BoundsLevel1, ubo_offset, {.Level=1u});
            compute->memoryBarrier(MTL::BarrierScopeBuffers);
            RecordBoundsPass(compute, slots, pipelines.BoundsCombine, buffers, PreludeSlot::BoundsLevel2, ubo_offset, {.Level=2u});
            compute->memoryBarrier(MTL::BarrierScopeBuffers);
            RecordBoundsPass(compute, slots, pipelines.BoundsCombine, buffers, PreludeSlot::BoundsLevel3, ubo_offset, {.Level=3u});
        }
    }
    if (is_edit_mode && std::exchange(scene_state.EditPreludePending, false)) RecordSparseEditPrelude(r, viewport, chain);
    if (phase == RenderPhase::Prepare) return;
    MTL::RenderCommandEncoder *encoder = nullptr;
    auto draw_quad = [&] { encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4)); };

    const auto &main = pipelines.Main;
    const auto main_extent = targets.Resources->SceneColorImage.Extent;
    const bool has_silhouette = render_silhouette && meshlet_fill;
    // Populate visibility for wireframe selection outlines and overlay depth without loading its depth into the scene pass.
    const bool need_visibility = meshlet_fill && (show_fill || has_silhouette || overlay_scene_depth);
    const bool wire_meshlets = draw_overlays &&
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::Wire)).Meshlets > 0u;
    const uint64_t bone_meshlets = draw_overlays ?
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::Bone)).Meshlets +
            buffers.FlagWork(uint32_t(MeshletInstanceFlag::BoneJoint)).Meshlets :
        0u;
    const uint64_t normal_meshlets = draw_overlays ? buffers.FlagWork(uint32_t(MeshletInstanceFlag::FaceNormal)).Meshlets : 0u;
    const uint64_t element_overlay_meshlets = draw_overlays ? buffers.FlagWork(uint32_t(MeshletInstanceFlag::EdgeOverlay)).Meshlets : 0u;
    const bool vertex_overlays = draw_overlays && !scene_state.VertexOverlays.empty();
    const bool cull_scene_meshlets =
        (need_visibility || wire_meshlets || bone_meshlets > 0u || normal_meshlets > 0u ||
         element_overlay_meshlets > 0u);
    // X-ray blends every solid surface through the transparency layers.
    const bool xray_fill = xray && show_fill && meshlet_fill;
    const bool transparent = xray_fill || (show_rendered && (real_transmission || std::ranges::any_of(buffers.Materials.GetSpan<PBRMaterial>(), [](const auto &m) { return m.AlphaMode == MaterialAlphaMode::Blend; })));
    const auto view_bytes = buffers.SceneViewUBO.Contents().subspan(ubo_offset, sizeof(SceneViewUBO));
    const auto &current_view_proj = reinterpret_cast<const SceneViewUBO *>(view_bytes.data())->ViewProj;
    const bool disocclusion_possible = update != SceneUpdate::Reuse || buffers.PreludeStale || buffers.MeshletOcclusionStale ||
        std::memcmp(&current_view_proj, &buffers.PreviousFullCullViewProj, sizeof(mat4)) != 0;
    // Cached occlusion is valid only for the pose that produced it. Reordering opaque
    // surfaces across temporal phases changes the winner of equal-depth raster ties.
    if (cull_scene_meshlets) {
        // X-ray draws occluded surfaces, so it skips occlusion culling.
        const uint32_t pyramid = show_fill && !xray && phase == RenderPhase::Full && targets.Resources->DepthPyramidValid && !disocclusion_possible ?
            samplers.DepthPyramid :
            InvalidSlot;
        RecordMeshletCull(
            chain, slots, pipelines, buffers,
            {
                .Mode = show_rendered ? (real_transmission ? MeshletRouteMode::Transmission : MeshletRouteMode::Material) : MeshletRouteMode::Visibility,
                .RequiredInstanceFlags = show_fill || overlay_scene_depth || wire_meshlets || bone_meshlets > 0u ||
                        normal_meshlets > 0u || element_overlay_meshlets > 0u ?
                    0u :
                    uint32_t(MeshletInstanceFlag::Silhouette),
                .RouteMask = 0x1ffu & ~(1u << uint32_t(MeshletRoute::EditOverlay)),
                .UboOffset = ubo_offset,
                .PyramidSamplerSlot = pyramid,
            }
        );
    }
    // Overlay depth needs a cleared visibility depth even without meshlets.
    if (need_visibility || overlay_scene_depth || phase == RenderPhase::BlurFast) {
        RecordMeshletVisibilityPass(chain, slots, pipelines, targets, buffers, real_transmission, ubo_offset, {});
    }
    if (show_fill && phase == RenderPhase::Full && cull_scene_meshlets) {
        buffers.PreviousFullCullViewProj = current_view_proj;
        // Only visibility surfaces contribute to occlusion.
        auto *compute = chain.BeginCompute("DepthPyramidFinal", MTL::StageFragment);
        RecordDepthPyramid(
            compute, slots, buffers, pipelines, targets.Resources->DepthPyramid, samplers.DepthPyramid, 0u,
            samplers.SceneDepth, targets.Resources->VisibilityDepth.Extent, false, ubo_offset
        );
        targets.Resources->DepthPyramidValid = true;
    }
    if (has_silhouette) RecordSilhouetteDepthPass(chain, slots, pipelines, targets, samplers, buffers, ubo_offset);

    // Render background and opaque faces without exposure into TransmissionImage for refracted sampling.
    if (transmission_active && draw_scene) {
        // Refraction samples only the world buffer.
        // The overlay composite adds the display-referred viewport backdrop.
        const std::array colors{mtl::ClearColor(*targets.Transmission->Mip0View)};
        const auto pass = mtl::MakePassDescriptor(colors, mtl::LoadDepth(*targets.Resources->VisibilityDepth));
        encoder = encode::BeginScenePass(chain, pass.get(), "TransmissionPrepass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, ubo_offset);
        main.PrepassBackground.Bind(encoder);
        draw_quad();
        if (meshlet_fill && show_fill) {
            main.Compiler.BindVisibility(encoder, true);
            encoder->setFragmentTexture(*targets.Resources->VisibilityImage, 0u);
            encode::SetPushConstants(encoder, encode::VisibilityDecodePc(buffers));
            draw_quad();
        }

        // Generate the transmission mip chain sampled across roughness.
        if (targets.Transmission->Image.MipLevels > 1) {
            auto *blit = chain.BeginBlit("TransmissionMips", MTL::StageFragment);
            blit->generateMipmaps(*targets.Transmission->Image);
        }
    }

    { // Shade against immutable visibility depth.
        const std::array colors{
            mtl::ClearColor(*targets.Resources->SceneColorImage),
        };
        const auto depth = show_fill && meshlet_fill && !xray ? mtl::LoadDepth(*targets.Resources->VisibilityDepth) : mtl::ClearDepth(*targets.Resources->ScratchDepth);
        const auto pass = mtl::MakePassDescriptor(colors, depth);
        if (transparent) {
            pass->setImageblockSampleLength(std::max(main.TransparencyInit.ImageblockSampleLength(), main.TransparencyResolve.ImageblockSampleLength()));
            pass->setTileWidth(16);
            pass->setTileHeight(16);
        }
        encoder = encode::BeginScenePass(
            chain, pass.get(), draw_scene ? "ScenePass" : "SceneDepthPass",
            {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment | MTL::StageBlit, MTL::StageFragment}},
            main_extent, slots, buffers, ubo_offset
        );

        // The prepass covers the background and plain-opaque geometry, so the composite replaces both.
        if (composite_transmission) {
            main.TransmissionComposite.Bind(encoder);
            draw_quad();
        } else if (show_rendered && draw_scene) {
            // Draw the background environment only in PBR modes.
            // The shader discards when world opacity is zero or no environment slot exists.
            main.Background.Bind(encoder);
            draw_quad();
        }
        // Resolve shutter samples before drawing sharp overlays.
        if (phase == RenderPhase::BlurResolve) {
            main.MotionBlurResolve.Bind(encoder);
            const MotionBlurResolvePushConstants resolve_pc{.AccumSamplerSlot = samplers.MotionBlurOutput, .InvSteps = 1.f / float(MotionBlurSteps(settings))};
            encode::SetPushConstants(encoder, resolve_pc);
            draw_quad();
        }

        // Draw solid faces.
        if (show_fill) {
            if (meshlet_fill && draw_scene && show_rendered) {
                if (!composite_transmission) {
                    main.Compiler.BindVisibility(encoder);
                    encoder->setFragmentTexture(*targets.Resources->VisibilityImage, 0u);
                    encode::SetPushConstants(encoder, encode::VisibilityDecodePc(buffers));
                    draw_quad();
                }
                if (transparent) {
                    main.TransparencyInit.Bind(encoder);
                    draw_quad();
                    main.Compiler.BindMeshlets(encoder);
                    if (real_transmission) DrawMeshlets(encoder, buffers, uint32_t(MeshletRoute::Transmission));
                    DrawMeshlets(encoder, buffers, uint32_t(MeshletRoute::Blend));
                    main.TransparencyResolve.Bind(encoder);
                    draw_quad();
                }
            } else if (xray_fill && draw_scene) {
                // Every visibility route blends at the X-ray opacity instead of resolving the nearest surface.
                main.TransparencyInit.Bind(encoder);
                draw_quad();
                main.WorkspaceTransparent.Bind(encoder);
                for (const auto [route, cull] : VisibilityRoutes) {
                    encoder->setCullMode(cull);
                    DrawMeshlets(encoder, buffers, uint32_t(route));
                }
                main.TransparencyResolve.Bind(encoder);
                draw_quad();
            } else if (meshlet_fill && draw_scene) {
                main.WorkspaceVisibility.Bind(encoder);
                encoder->setFragmentTexture(*targets.Resources->VisibilityImage, 0u);
                encode::SetPushConstants(encoder, encode::VisibilityDecodePc(buffers));
                draw_quad();
            }
        }
    }

    if (!draw_overlays) { // Accumulate this shutter sample without overlays.
        const std::array colors{
            phase == RenderPhase::BlurAccumulateFirst ? mtl::ClearColor(*targets.MotionBlur->OutputImage) : mtl::LoadColor(*targets.MotionBlur->OutputImage)
        };
        const auto pass = mtl::MakePassDescriptor(colors);
        encoder = encode::BeginScenePass(chain, pass.get(), "BlurAccumulate", {{MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, ubo_offset);
        main.MotionBlurAccumulate.Bind(encoder);
        const MotionBlurAccumulatePushConstants accum_pc{.SceneSamplerSlot = samplers.SceneColor, .Weight = float(sample_weight)};
        encode::SetPushConstants(encoder, accum_pc);
        draw_quad();
        return;
    }

    const bool meshlet_edit_overlay_drawn =
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::EditOverlay)).Meshlets > 0;
    const bool overlay_jobs = show_overlays && buffers.OverlayJobs.UsedSize > 0u &&
        (settings.ShowExtras || settings.ShowBoundingBoxes || settings.ShowTetWireframe);
    if (wire_meshlets) {
        buffers.WireCoverageBuffer.SetCount<uint32_t>(main_extent.Width * main_extent.Height);
        { // Four independent 8-bit coverage maxima share one word.
            auto *blit = chain.BeginBlit("WireClear", MTL::StageDispatch);
            blit->fillBuffer(*buffers.WireCoverageBuffer, NS::Range::Make(0, buffers.WireCoverageBuffer.UsedSize), 0);
        }
        // Canonical meshlet edge owners accumulate with atomics, so threadgroups need no ordering.
        auto *wire = chain.BeginCompute("WireRaster", MTL::StageBlit | MTL::StageFragment, MTL::DispatchTypeConcurrent);
        encode::BindCompute(wire, pipelines.WireRaster, slots, buffers, ubo_offset);
        wire->setTexture(*targets.Resources->VisibilityDepth, 0u);
        WireRasterPushConstants wire_pc{
            .Meshlet = MakeMeshletDrawPc(
                buffers, buffers.SceneCull,
                uint32_t(MeshletRoute::Wire), uint32_t(MeshletInstanceFlag::Wire),
                false, InvalidSlot
            ),
            .CoverageSlot = buffers.WireCoverageBuffer.Slot,
            .TestDepth = overlay_scene_depth,
            .BehindOpacity = overlay_behind,
        };
        for (uint32_t chunk = 0; chunk < buffers.SceneCull.ChunkCount; ++chunk) {
            wire_pc.Meshlet.VisibleOffset = chunk * GpuBuffers::MeshletDispatchChunkSize;
            encode::SetPushConstants(wire, wire_pc);
            const auto args_offset = (uint32_t(MeshletRoute::Wire) * buffers.SceneCull.ChunkCount + chunk) *
                sizeof(MeshDispatchArgs);
            wire->dispatchThreadgroups(*buffers.SceneCull.DispatchArgs, args_offset, MTL::Size(160, 1, 1));
        }
    }
    if (overlay_jobs) RecordOverlayJobCull(chain, slots, pipelines, buffers, false, ubo_offset);
    // This frame's depth pyramid is valid here. Cull hidden edit edges and vertex overlays only
    // when their render pass tests against that same scene depth.
    const uint32_t overlay_pyramid = show_fill && !overlays_through && !xray && !has_silhouette &&
            phase == RenderPhase::Full && cull_scene_meshlets ?
        samplers.DepthPyramid : InvalidSlot;
    if (meshlet_edit_overlay_drawn) {
        RecordMeshletCull(chain, slots, pipelines, buffers, {
            .Mode = MeshletRouteMode::Visibility,
            .RequiredInstanceFlags = uint32_t(MeshletInstanceFlag::EditOverlay),
            .RouteMask = 1u << uint32_t(MeshletRoute::EditOverlay),
            .UboOffset = ubo_offset,
            .PyramidSamplerSlot = overlay_pyramid,
            .ExactEditGeometry = true,
            .MinEditOverlayDiameterPixels = MinEditOverlayDiameterPixels,
            .EditOverlayHasSharpEdges = scene_state.MeshletEditHasSharpEdges,
            .EditOutput = true,
        });
    }

    // Initialize overlays even when no geometry contributes color.
    const bool overlay_pass_needed = has_silhouette ||
        (show_overlays && settings.ShowGrid) ||
        meshlet_edit_overlay_drawn || element_overlay_meshlets > 0u || vertex_overlays || wire_meshlets ||
        overlay_jobs ||
        normal_meshlets > 0u || bone_meshlets > 0u;
    if (overlay_pass_needed) { // Display-referred overlays, depth-tested against scene surfaces and outlines.
        // The outline draw initializes color and scene-plus-outline depth across the target.
        const std::array overlay_colors{
            has_silhouette ? mtl::DiscardColor(*targets.Resources->OverlayColorImage) : mtl::ClearColor(*targets.Resources->OverlayColorImage),
        };
        const auto overlay_depth = has_silhouette ? mtl::DepthAttachment{*targets.Resources->ScratchDepth, MTL::LoadActionDontCare, MTL::StoreActionDontCare} :
            overlays_through                      ? mtl::ClearDepth(*targets.Resources->ScratchDepth) :
                                                    mtl::LoadDepth(*targets.Resources->VisibilityDepth);
        const auto overlay_pass = mtl::MakePassDescriptor(overlay_colors, overlay_depth);
        encoder = encode::BeginScenePass(chain, overlay_pass.get(), "OverlayPass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, ubo_offset);

        if (has_silhouette) {
            main.SilhouetteEdgeColor.Bind(encoder);
            // In mesh Edit mode, suppress active silhouette (element selection drives active state differently).
            // In armature Edit/Pose mode, the active bone gets the active-color silhouette.
            const auto active_entity = FindActiveEntity(r);
            const auto active_bone = FindActiveBone(r);
            const bool armature_mode = FindArmatureObject(r, active_entity) != state::Null;
            uint32_t active_object_id = 0;
            if (armature_mode && active_bone != state::Null) {
                if (r.all_of<RenderInstance>(active_bone)) active_object_id = ObjectId(active_bone);
            } else if (!is_edit_mode && active_entity != state::Null && r.all_of<RenderInstance>(active_entity)) {
                active_object_id = ObjectId(active_entity);
            }
            encode::SetPushConstants(encoder, SilhouetteEdgeColorPushConstants{
                                                  r.get<const GizmoInteraction>(viewport).IsUsing() && interaction_mode == InteractionMode::Object,
                                                  samplers.Silhouette,
                                                  active_object_id,
                                                  overlays_through ? InvalidSlot : samplers.SceneDepth,
                                              });
            draw_quad();
        }

        if (show_overlays && settings.ShowGrid) {
            main.Grid.Bind(encoder);
            encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(9));
        }

        const auto draw_meshlet_overlay = [&](
                                              const mtl::RenderPipeline &pipeline, MeshletRoute route, MeshletInstanceFlag flag,
                                              uint32_t threads, uint32_t corner = 0u, uint32_t sharpness_slot = InvalidSlot,
                                              const MeshletCullOutput *cull = nullptr
                                          ) {
            pipeline.Bind(encoder);
            DrawMeshletList(
                encoder, buffers,
                uint32_t(route), uint32_t(flag), false, false, sharpness_slot, threads, corner,
                InvalidOffset, cull
            );
        };

        if (meshlet_edit_overlay_drawn) {
            // Preserve the sharp-free specialization; the all-smooth Meshlets scene is measurably faster with it.
            const auto &edit_edges = scene_state.MeshletEditHasSharpEdges ? main.MeshletEditEdges : main.MeshletEditSmoothEdges;
            for (uint32_t corner = 0u; corner < 3u; ++corner) {
                draw_meshlet_overlay(
                    edit_edges, MeshletRoute::EditOverlay, MeshletInstanceFlag::EditOverlay,
                    160u, corner, meshes.Slots().EdgeSharpness, &buffers.EditCull
                );
            }
        }
        if (buffers.FlagWork(uint32_t(MeshletInstanceFlag::EdgeOverlay)).Meshlets > 0u) {
            for (uint32_t corner = 0u; corner < 3u; ++corner) {
                draw_meshlet_overlay(
                    main.MeshletEditSmoothEdges, MeshletRoute::Overlay,
                    MeshletInstanceFlag::EdgeOverlay, 160u, corner
                );
            }
        }
        if (wire_meshlets) {
            main.WireResolve.Bind(encoder);
            encode::SetPushConstants(encoder, WireResolvePushConstants{buffers.WireCoverageBuffer.Slot});
            draw_quad();
        }
        if (overlay_jobs) {
            main.OverlayJobLines.Bind(encoder);
            DrawOverlayJobs(encoder, buffers, meshes);
        }

        if (normal_meshlets > 0u) draw_meshlet_overlay(main.FaceNormalMesh, MeshletRoute::Overlay, MeshletInstanceFlag::FaceNormal, 64u);
        if (vertex_overlays) {
            const auto draw_vertex_overlays = [&](const mtl::RenderPipeline &pipeline, VertexOverlay kind) {
                bool bound = false;
                for (const auto &draw : scene_state.VertexOverlays) {
                    if (draw.Kind != kind) continue;
                    if (!std::exchange(bound, true)) pipeline.Bind(encoder);
                    DrawVertexBlocks(
                        encoder, r, draw.MeshEntity, draw.Instance, kind == VertexOverlay::SoundPoints, overlay_pyramid,
                        kind == VertexOverlay::EditPoints ? MinEditOverlayDiameterPixels : 0.0f
                    );
                }
            };
            draw_vertex_overlays(main.VertexNormalMesh, VertexOverlay::Normals);
            // Selected vertices composite above strokes.
            if (edit_mode == Element::Vertex) draw_vertex_overlays(main.VertexBlockPoints, VertexOverlay::EditPoints);
            draw_vertex_overlays(main.VertexBlockPoints, VertexOverlay::Points);
            draw_vertex_overlays(main.VertexBlockPoints, VertexOverlay::SoundPoints);
        }
        // Bone X-ray preserves overlay color and clears scratch depth to order bones against each other.
        if (bone_meshlets > 0u) {
            const std::array bone_colors{
                mtl::LoadColor(*targets.Resources->OverlayColorImage),
            };
            const auto bone_pass = mtl::MakePassDescriptor(bone_colors, {*targets.Resources->ScratchDepth, MTL::LoadActionClear, MTL::StoreActionDontCare});
            encoder = encode::BeginScenePass(chain, bone_pass.get(), "BoneXRay", {{MTL::StageDispatch, MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, ubo_offset);

            const auto draw_bones = [&](const mtl::RenderPipeline &pipeline, MeshletInstanceFlag flag, uint32_t threads, float depth_bias = 0.f) {
                if (buffers.FlagWork(uint32_t(flag)).Meshlets == 0u) return;
                pipeline.Bind(encoder);
                encoder->setDepthBias(depth_bias, 0.f, 0.f);
                DrawMeshlets(encoder, buffers, uint32_t(MeshletRoute::Overlay), uint32_t(flag), threads);
                if (depth_bias != 0.f) encoder->setDepthBias(0.f, 0.f, 0.f);
            };

            // In Object+wireframe mode, show only outlines (no fills).
            // Edit/Pose wireframe fills remain translucent; actual depth orders overlapping bones.
            const bool object_wireframe = is_wireframe_mode && interaction_mode == InteractionMode::Object;
            if (!object_wireframe) {
                draw_bones(main.BoneFillMesh, MeshletInstanceFlag::Bone, 24u, 2.f);
                draw_bones(main.BoneSphereFillMesh, MeshletInstanceFlag::BoneJoint, uint32_t(OverlayDispatch::BoneSphereVertices));
            }
            // In non-wireframe Object mode, "Outline selected" off suppresses bone wire outlines.
            // In wireframe+Object mode, wires are the only bone visualization so always show them.
            const bool hide_bone_outlines = !is_wireframe_mode && interaction_mode == InteractionMode::Object &&
                (!show_overlays || !settings.ShowOutlineSelected);
            if (!hide_bone_outlines) {
                draw_bones(main.BoneWireMesh, MeshletInstanceFlag::BoneWire, 24u);
                draw_bones(main.BoneSphereWireMesh, MeshletInstanceFlag::BoneJointWire, 64u);
            }
        }
    }

    if (phase == RenderPhase::BlurFast) {
        // Refraction and overlays have finished reading opaque visibility. Include glass in
        // the ID/depth pair for blur now, without changing the depth used to shade behind it.
        if (real_transmission && meshlet_fill) {
            const std::array colors{mtl::LoadColor(*targets.Resources->VisibilityImage)};
            const auto pass = mtl::MakePassDescriptor(colors, mtl::LoadDepth(*targets.Resources->VisibilityDepth));
            auto *visibility = encode::BeginScenePass(chain, pass.get(), "BlurTransmissionVisibility", {{MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, ubo_offset);
            main.MeshletVisibilityCoverage.Bind(visibility);
            visibility->setCullMode(MTL::CullModeNone);
            DrawMeshletList(visibility, buffers, uint32_t(MeshletRoute::Transmission), 0u, false, true);
        }
        RecordMotionBlurPostFx(r, viewport, chain, ubo_offset);
    }

    { // View-transform the scene, then composite display-referred overlays.
        const std::array colors{mtl::ClearColor(*targets.Resources->FinalColorImage, {0, 0, 0, 1})};
        const auto pass = mtl::MakePassDescriptor(colors);
        encoder = encode::BeginScenePass(chain, pass.get(), "Composite", {{MTL::StageFragment, MTL::StageFragment}}, targets.Resources->FinalColorImage.Extent, slots, buffers, ubo_offset);
        main.ViewportComposite.Bind(encoder);
        // Debug channels write their own already-viewable values, so they pass through untransformed.
        const uint32_t view_transform = settings.DebugChannel != DebugChannel::None ? 2u : show_rendered ? 1u :
                                                                                                           0u;
        const uint32_t scene_sampler = phase == RenderPhase::BlurFast ? samplers.MotionBlurOutput : samplers.SceneColor;
        const ViewportCompositePushConstants composite_pc{
            .SceneColorSamplerSlot = scene_sampler,
            .OverlayColorSamplerSlot = samplers.OverlayColor,
            .ViewTransform = view_transform,
            .HasOverlay = overlay_pass_needed,
            .Backdrop = settings.ClearColor,
        };
        encode::SetPushConstants(encoder, composite_pc);
        draw_quad();
    }
}

} // namespace

void RecordOverlayJobCull(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines,
    GpuBuffers &buffers, bool extras_only, uint32_t ubo_offset
) {
    const uint32_t job_count = buffers.OverlayJobs.Count<OverlayJob>();
    if (job_count == 0u) return;
    const OverlayJobCullPushConstants pc{
        .JobsSlot = buffers.OverlayJobs.Slot,
        .JobCount = job_count,
        .InstanceStateSlot = buffers.Instances.StateBuffer.Slot,
        .BlockStateSlot = buffers.OverlayJobBlocks.Slot,
        .VisibleSlot = buffers.VisibleOverlayJobs.Slot,
        .DispatchArgsSlot = buffers.OverlayJobDispatchArgs.Slot,
        .ExtrasOnly = extras_only,
    };
    auto *encoder = chain.BeginCompute("OverlayJobCull", MTL::StageDispatch | MTL::StageFragment);
    encode::BindScene(encoder, slots, buffers, ubo_offset);
    encode::SetPushConstants(encoder, pc);
    const auto blocks = MTL::Size(
        (job_count + GpuBuffers::OverlayJobBlockSize - 1u) / GpuBuffers::OverlayJobBlockSize, 1, 1
    );
    encoder->setComputePipelineState(pipelines.OverlayJobBlockCount.State());
    encoder->dispatchThreadgroups(blocks, ThreadgroupSize::Linear256);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    encoder->setComputePipelineState(pipelines.OverlayJobPrefix.State());
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    encoder->setComputePipelineState(pipelines.OverlayJobEmit.State());
    encoder->dispatchThreadgroups(blocks, ThreadgroupSize::Linear256);
}

void DrawOverlayJobs(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, const MeshStore &meshes
) {
    encode::SetMeshPushConstants(encoder, OverlayJobDrawPushConstants{
                                              .JobsSlot = buffers.OverlayJobs.Slot,
                                              .VisibleSlot = buffers.VisibleOverlayJobs.Slot,
                                              .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
                                              .BoundsSlot = buffers.Instances.BoundsBuffer.Slot,
                                              .ModelSlot = buffers.Instances.TransformBuffer.Slot,
                                              .StateSlot = buffers.Instances.StateBuffer.Slot,
                                              .TetPositionSlot = meshes.Slots().TetPosition,
                                              .TetEdgeIndexSlot = meshes.Slots().TetEdgeIndex,
                                          });
    encoder->drawMeshThreadgroups(
        *buffers.OverlayJobDispatchArgs, 0u, MTL::Size(1, 1, 1),
        MTL::Size(uint32_t(OverlayDispatch::LineGroupLines) * 2u, 1, 1)
    );
}

void RecordMeshletVisibilityPass(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines, const RenderTargets &targets,
    GpuBuffers &buffers, bool transmission, uint32_t ubo_offset, std::optional<PixelRect> scissor
) {
    const auto &main = pipelines.Main;
    const std::array colors{
        mtl::ClearColor(*targets.Resources->VisibilityImage, MTL::ClearColor{double(UINT32_MAX), 0, 0, 0})
    };
    const auto pass = mtl::MakePassDescriptor(colors, mtl::ClearDepth(*targets.Resources->VisibilityDepth));
    auto *encoder = encode::BeginScenePass(
        chain, pass.get(), "MeshletVisibility", {{MTL::StageDispatch, MTL::StageMesh}},
        targets.Resources->VisibilityImage.Extent, slots, buffers, ubo_offset
    );
    if (scissor) encoder->setScissorRect({scissor->Origin.x, scissor->Origin.y, scissor->Extent.x, scissor->Extent.y});
    // Without meshlets the pass only clears, which is what overlay depth needs.
    if (buffers.MeshletInstanceCount > 0) DrawVisibilityMeshlets(encoder, buffers, main, transmission);
    buffers.VisibilityGeneration = buffers.MeshletVisibleGeneration;
}

void RecordMeshletCull(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines,
    GpuBuffers &buffers, MeshletCullConfig config
) {
    auto &output = config.EditOutput ? buffers.EditCull : buffers.SceneCull;
    if (!config.EditOutput) ++buffers.MeshletVisibleGeneration;
    auto *encoder = chain.BeginCompute("MeshletCull", MTL::StageMesh | MTL::StageFragment);
    const bool transmission = config.Mode == MeshletRouteMode::Transmission;
    // The requested flag's maintained totals bound this cull.
    const auto primary = config.RequiredInstanceFlags == 0u ?
        GpuBuffers::MeshletFlagWork{buffers.LodNodeCount, buffers.MeshletInstanceCount} :
        buffers.FlagWork(config.RequiredInstanceFlags);
    buffers.EnsureMeshletVisibilityCapacity(
        output,
        primary.Meshlets * (1u + transmission), primary.Nodes, primary.Meshlets
    );
    const auto pc = [&] {
        auto pc = MakeMeshletCullSlotsPc(buffers, output);
        pc.InstanceCount = buffers.GpuInstanceSlots.Count<uint32_t>();
        pc.WorkBlockCount = (pc.InstanceCount + GpuBuffers::MeshletCullBlockSize - 1u) / GpuBuffers::MeshletCullBlockSize;
        pc.LodFrontierStateSlot = buffers.LodFrontierStates.Slot;
        pc.RouteMode = uint32_t(config.Mode);
        pc.RequiredInstanceFlags = config.RequiredInstanceFlags;
        pc.RouteMask = config.RouteMask;
        pc.PyramidSamplerSlot = config.PyramidSamplerSlot;
        pc.ExactEditGeometry = config.ExactEditGeometry;
        pc.MinEditOverlayDiameterPixels = config.MinEditOverlayDiameterPixels;
        pc.EditOverlayHasSharpEdges = config.EditOverlayHasSharpEdges;
        if (config.EditOutput) pc.CoarseCountSlot = InvalidSlot;
        return pc;
    }();
    encode::BindScene(encoder, slots, buffers, config.UboOffset);
    constexpr uint32_t simd_groups = GpuBuffers::MeshletCullBlockSize / 32u;
    constexpr uint32_t prefix_stride = simd_groups + 1u;
    // Descends every span tree in lockstep and emits surviving record runs in frontier order.
    const uint32_t level_count = buffers.MeshletLodDepth + 2u;
    for (uint32_t level = 0; level < level_count; ++level) {
        const uint32_t index = level & 1u;
        auto level_pc = pc;
        level_pc.LodFrontierSlot = buffers.LodFrontiers[index].Slot;
        level_pc.LodFrontierAltSlot = buffers.LodFrontiers[index ^ 1u].Slot;
        level_pc.LodFrontierIndex = index;
        level_pc.LodSeedLevel = level == 0u;
        level_pc.LodFinalLevel = level + 1u == level_count;
        encode::SetPushConstants(encoder, level_pc);
        // Seed from the host-known instance count and size later grids from the preceding frontier.
        const auto dispatch_level = [&](const mtl::ComputePipeline &pipeline) {
            if (level == 0u && pc.WorkBlockCount == 0u) return;
            encoder->setComputePipelineState(pipeline.State());
            encoder->setThreadgroupMemoryLength(AlignedThreadgroupBytes(2u * prefix_stride * sizeof(uint32_t)), 0);
            if (level == 0u) encoder->dispatchThreadgroups(MTL::Size(pc.WorkBlockCount, 1, 1), MTL::Size(GpuBuffers::MeshletCullBlockSize, 1, 1));
            else encoder->dispatchThreadgroups(*buffers.LodExpandArgs, index * sizeof(MeshDispatchArgs), MTL::Size(GpuBuffers::MeshletCullBlockSize, 1, 1));
        };
        dispatch_level(pipelines.LodFrontierCount);
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        encoder->setComputePipelineState(pipelines.LodFrontierPrefix.State());
        encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256);
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        dispatch_level(pipelines.LodFrontierEmit);
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    }
    encode::SetPushConstants(encoder, pc);
    RecordMeshletCompaction(encoder, pipelines, buffers);
}

void RecordSilhouetteDepthPass(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines, const RenderTargets &targets,
    const RenderSamplerSlots &samplers, GpuBuffers &buffers, uint32_t ubo_offset
) {
    // Compacts the outlined surfaces the visibility image cannot resolve.
    const auto &resources = *targets.Resources;
    auto decode_pc = encode::VisibilityDecodePc(buffers);
    auto *compute = chain.BeginCompute("SilhouetteCull", MTL::StageFragment);
    const auto &occluders = resources.OutlineOccluderPyramid;
    encode::BindCompute(compute, pipelines.OutlineOccluderSeed, slots, buffers, ubo_offset);
    compute->setTexture(*resources.VisibilityImage, 0u);
    compute->setTexture(*resources.VisibilityDepth, 1u);
    compute->setTexture(*occluders.Mips[0].View, 2u);
    encode::SetPushConstants(compute, decode_pc);
    const auto blocks = occluders.Mips[0].Extent;
    compute->dispatchThreadgroups(MTL::Size((blocks.Width + 15u) / 16u, (blocks.Height + 15u) / 16u, 1u), ThreadgroupSize::Tile16);
    compute->memoryBarrier(MTL::BarrierScopeTextures);
    RecordDepthPyramid(compute, slots, buffers, pipelines, occluders, samplers.OutlineOccluderPyramid, 1u, InvalidSlot, {}, true, ubo_offset);
    compute->memoryBarrier(MTL::BarrierScopeTextures);

    // Kept entries are a subset of the scene cull's.
    auto &output = buffers.SilhouetteCull;
    output.Visible.SetCount<VisibleMeshlet>(buffers.SceneCull.Visible.Count<VisibleMeshlet>());
    output.ChunkCount = buffers.SceneCull.ChunkCount;
    output.DispatchArgs.SetCount<MeshDispatchArgs>(GpuBuffers::MeshletRouteCount * output.ChunkCount);
    auto cull_pc = MakeMeshletCullSlotsPc(buffers, output);
    cull_pc.PyramidSamplerSlot = samplers.OutlineOccluderPyramid;
    cull_pc.SourceVisibleSlot = buffers.SceneCull.Visible.Slot;
    cull_pc.SourceRouteStateSlot = buffers.SceneCull.Routes.Slot;
    encode::SetPushConstants(compute, cull_pc);
    compute->setComputePipelineState(pipelines.SilhouetteCullSize.State());
    compute->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    compute->memoryBarrier(MTL::BarrierScopeBuffers);
    RecordMeshletCompaction(compute, pipelines, buffers);

    // Outlined surfaces occlude each other in a private depth, so unoutlined geometry never hides an outline.
    const std::array colors{mtl::ClearColor(*resources.SilhouetteImage)};
    const auto pass = mtl::MakePassDescriptor(colors, {*resources.ScratchDepth, MTL::LoadActionClear, MTL::StoreActionDontCare});
    auto *encoder = encode::BeginScenePass(
        chain, pass.get(), "SilhouetteDepth", {{MTL::StageDispatch, MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment, MTL::StageFragment}},
        resources.SilhouetteImage.Extent, slots, buffers, ubo_offset
    );
    pipelines.SilhouetteSeed.Bind(encoder);
    encoder->setFragmentTexture(*resources.VisibilityImage, 0u);
    encoder->setFragmentTexture(*resources.VisibilityDepth, 1u);
    encoder->setFragmentBytes(&decode_pc, sizeof(decode_pc), BufferIndex_PushConstants);
    encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
    pipelines.Silhouette.Bind(encoder);
    decode_pc.VisibleMeshletSlot = buffers.SilhouetteCull.Visible.Slot;
    encoder->setFragmentBytes(&decode_pc, sizeof(decode_pc), BufferIndex_PushConstants);
    const auto draw_route = [&](MeshletRoute route, MTL::CullMode cull) {
        encoder->setCullMode(cull);
        DrawMeshletList(encoder, buffers, uint32_t(route), 0u, false, false, InvalidSlot, 160u, 0u, InvalidOffset, &buffers.SilhouetteCull);
    };
    for (const auto [route, cull] : VisibilityRoutes) draw_route(route, cull);
    draw_route(MeshletRoute::Blend, MTL::CullModeNone);
    draw_route(MeshletRoute::Transmission, MTL::CullModeNone);
}


void DrawMeshlets(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, uint32_t route,
    uint32_t required_instance_flags, uint32_t mesh_threads, uint32_t edit_edge_corner,
    uint32_t instance_filter
) {
    DrawMeshletList(
        encoder, buffers,
        route, required_instance_flags, false, false, InvalidSlot,
        mesh_threads, edit_edge_corner, instance_filter
    );
}

void DrawVertexBlocks(
    MTL::RenderCommandEncoder *encoder, const state::Scene &r, state::Entity mesh_entity, uint32_t instance, bool sound_points,
    uint32_t pyramid_slot, float min_diameter_pixels
) {
    const auto &buffers = r.Context.get<const GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &scene = r.Context.get<const GpuSceneState>();
    const auto store_id = r.get<const MeshHandle>(mesh_entity).StoreId;
    VertexBlockPushConstants pc{
        .Instance = instance,
        .MembershipSlot = meshes.Arenas().Vertices.Blocks.Buffer.Slot,
        .SoundPoints = sound_points ? 1u : 0u,
        .PyramidSamplerSlot = pyramid_slot,
        .MinDiameterPixels = min_diameter_pixels,
    };
    if (const auto posed = scene.PosedByEntity.find(mesh_entity); posed != scene.PosedByEntity.end()) {
        pc.BoundsNamespace = posed->second.VertexBoundsNamespace(posed->second.PerInstance ? instance - posed->second.FirstInstance : 0u);
        pc.BoundsNodesSlot = buffers.VertexBounds.Nodes.Buffer.Slot;
        pc.BoundsValuesSlot = buffers.VertexBounds.Values.Buffer.Slot;
        pc.BoundsMembersSlot = buffers.VertexBounds.Members.Slot;
    } else if (meshes.Get(store_id).SelectionSummary.Count) {
        pc.LeafSlot = meshes.Arenas().VertexAggregates.Buffer.Slot;
    }
    // Each dispatch stays within the mesh grid's dimension limit.
    constexpr uint32_t BlocksPerDispatch{GpuBuffers::MeshletDispatchChunkSize / VertexBlockGroups};
    const auto list = meshes.GetBlockList(store_id, MeshStore::ElementDomain::Vertex);
    for (uint32_t first = 0u; first < list.Blocks.size(); first += BlocksPerDispatch) {
        pc.Blocks = {list.Gpu.Slot, list.Gpu.Offset + first};
        encode::SetMeshPushConstants(encoder, pc);
        const auto blocks = std::min(BlocksPerDispatch, uint32_t(list.Blocks.size()) - first);
        encoder->drawMeshThreadgroups(MTL::Size(blocks * VertexBlockGroups, 1, 1), MTL::Size(1, 1, 1), MTL::Size(MeshElementBlockSize / VertexBlockGroups, 1, 1));
    }
}

void RecordRenderCommandBuffer(state::Scene &r, state::Entity viewport, MTL::CommandBuffer *command_buffer, SceneUpdate update, RenderPhase phase) {
    const mtl::AutoreleaseScope native_scope;
    profile::BeginRecording();
    mtl::PassChain chain{command_buffer, profile::RecordingTimer()};
    RecordPhase(r, viewport, chain, update, phase, 0);
    profile::EndRecording();
}

void RecordBlurStepsCommandBuffer(state::Scene &r, state::Entity viewport, MTL::CommandBuffer *command_buffer, std::span<const uint32_t> sample_weights) {
    const mtl::AutoreleaseScope native_scope;
    const auto &buffers = r.Context.get<const GpuBuffers>();
    profile::BeginRecording();
    mtl::PassChain chain{command_buffer, profile::RecordingTimer()};
    for (uint32_t i = 0; i < sample_weights.size(); ++i) {
        RecordPhase(r, viewport, chain, i == 0 ? SceneUpdate::Rebuild : SceneUpdate::Reuse, i == 0 ? RenderPhase::BlurAccumulateFirst : RenderPhase::BlurAccumulate, buffers.SceneViewUboOffset(i + 1), sample_weights[i]);
    }
    RecordPhase(r, viewport, chain, SceneUpdate::Reuse, RenderPhase::BlurResolve, 0);
    profile::EndRecording();
}

namespace {
// Upload `entries` and their tiles, then record and submit one batched two-phase derive and wait for completion.
// The output slots select the target buffers.
void SubmitNormalDeriveNow(state::Scene &r, std::span<const NormalDeriveEntry> entries, uint32_t vertex_normal_slot, uint32_t face_normal_slot) {
    const auto &meshes = r.Context.get<const MeshStore>();
    auto &buffers = r.Context.get<GpuBuffers>();
    auto pc = MakeNormalDerivePc(buffers, meshes, vertex_normal_slot, face_normal_slot);
    DeriveNormalsNow(r, entries, pc);
}
} // namespace

void DeriveBaseNormalsNow(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    std::vector<uint32_t> ids;
    ids.reserve(mesh_entities.size());
    for (const auto entity : mesh_entities)
        if (const auto mesh = TryGetMesh(r, entity)) ids.push_back(mesh->GetStoreId());
    DeriveMeshNormalsNow(r, ids);
}

void UpdateAuthoredMorphShadingNow(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    auto &meshes = r.Context.get<MeshStore>();
    auto &buffers = r.Context.get<GpuBuffers>();
    // Each position-only target gets a derive entry reading its full-weight pose directly.
    struct PoseJob {
        state::Entity Entity;
    };
    std::vector<NormalDeriveEntry> entries;
    std::vector<PoseJob> jobs;
    PoseAttributeStore<vec3>::Temporary vertex_normals{buffers.PosedVertexNormals};
    PoseAttributeStore<vec3>::Temporary face_normals{buffers.PosedFaceNormals};
    PoseAttributeStore<vec3>::Temporary temporary{buffers.PosedSectors};
    for (const auto entity : mesh_entities) {
        const auto mesh = TryGetMesh(r, entity);
        if (!mesh) continue;
        const auto store_id = mesh->GetStoreId();
        const auto &record = meshes.Get(store_id);
        const auto target_count = record.MorphTargetCount;
        // A mesh without authored normals shades by derivation alone, under any morph weights.
        if (target_count == 0 || !record.HasAuthoredNormals) continue;
        const auto entry_inputs = MakeDeriveEntryInputs(meshes, store_id);
        if (!entry_inputs) continue;
        // Resolve the authored-normal gate from every morph target.
        UpdateMorphShadingAuthored(meshes, *mesh, {});
        if (meshes.Get(store_id).MorphShadingAuthored) continue;
        const auto normal_blocks = NormalPayloadBlocks(meshes, store_id);
        const auto vertex_blocks = PoseElementBlocks(meshes.Arenas().Vertices,record.Vertices);
        const auto face_blocks = PoseElementBlocks(meshes.Arenas().FaceTriangles,record.FaceData);
        const auto &morph = meshes.Arenas().Morph;
        for (uint32_t t = 0; t < target_count; ++t) {
            // Targets without position deltas use the rest pose and require no normal pinning.
            bool has_position_delta = false;
            meshes.Arenas().Vertices.ForEach(record.Vertices,[&](uint32_t vertex,uint32_t) {
                if (!has_position_delta && morph.Get(vertex, t).PositionDelta != vec3{0}) has_position_delta = true;
            });
            if (!has_position_delta) continue;
            auto entry = *entry_inputs;
            entry.Morph = meshes.Slots().Morph;
            entry.MorphTargetIndex = t;
            entry.VertexNormalNamespace = vertex_normals.Add(vertex_blocks);
            entry.SectorNamespace = temporary.Add(normal_blocks);
            entry.FaceNormalNamespace = face_normals.Add(face_blocks);
            entries.emplace_back(entry);
            jobs.emplace_back(entity);
        }
    }
    if (entries.empty()) return;

    // Derivation reads each full-weight morph directly from canonical base vertices and target deltas.
    // No temporary position buffer or CPU geometry copy.
    SubmitNormalDeriveNow(r, entries, buffers.PosedVertexNormals.Values.Buffer.Slot, buffers.PosedFaceNormals.Values.Buffer.Slot);

    // Compare per mesh over its contiguous run of jobs.
    for (size_t i = 0; i < jobs.size();) {
        const auto entity = jobs[i].Entity;
        std::vector<CornerNormalSources> poses;
        for (; i < jobs.size() && jobs[i].Entity == entity; ++i) {
            const auto &entry = entries[i];
            poses.push_back({
                .VertexNormals = meshes.Arenas().BaseVertexNormals.Buffer.GetSpan<vec3>(),
                .FaceNormals = meshes.Arenas().BaseFaceNormals.Buffer.GetSpan<vec3>(),
                .PosedVertexNormals = buffers.PosedVertexNormals.View(entry.VertexNormalNamespace),
                .PosedSectorNormals = buffers.PosedSectors.View(entry.SectorNamespace),
                .PosedFaceNormals = buffers.PosedFaceNormals.View(entry.FaceNormalNamespace),
            });
        }
        UpdateMorphShadingAuthored(meshes, GetMesh(r, entity), poses);
    }
}

void FinalizeNewMeshShadingNow(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    DeriveBaseNormalsNow(r, mesh_entities);
    auto &meshes = r.Context.get<MeshStore>();
    for (const auto entity : mesh_entities) {
        const auto *authored = r.try_get<const AuthoredCornerNormals>(entity);
        if (!authored) continue;
        EncodeAuthoredCornerNormals(meshes, GetMesh(r, entity), authored->Corners);
        r.remove<AuthoredCornerNormals>(entity);
    }
    UpdateAuthoredMorphShadingNow(r, mesh_entities);
}

namespace {
void DispatchWork(MTL::ComputeCommandEncoder *encoder, const GpuBuffers &buffers, ElementWork work) {
    encoder->dispatchThreadgroups(*buffers.GeometryWork.Buffer, WorkArgsOffset(work), ThreadgroupSize::Linear256);
}

void FinalizeWork(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const Pipelines &pipelines,
                  const GpuBuffers &buffers, std::initializer_list<ElementWork> work) {
    encode::BindCompute(encoder, pipelines.FinalizeElementWork, slots, buffers);
    encoder->setBytes(work.begin(), work.size() * sizeof(ElementWork), BufferIndex_PushConstants);
    encoder->dispatchThreadgroups(MTL::Size(work.size(), 1, 1), ThreadgroupSize::Linear256);
}

MeshEditWork &PrepareMeshEditWork(state::Scene &r, state::Entity entity) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto mesh = GetMesh(r, entity);
    const auto id = mesh.GetStoreId();
    auto &scene = r.Context.get<GpuSceneState>();
    auto &work = scene.EditWork;
    if (const auto it = work.find(entity); it != work.end()) {
        if (it->second.StoreId == id) return it->second;
        ReleaseMeshEditWork(r, entity);
    }
    // Edit work pins the mesh's instances to finest geometry.
    scene.InstanceFlagsStale = true;
    auto work_allocation = buffers.GeometryWork.BeginAllocation();
    MeshEditWork w{.StoreId = id};
    w.WorkBudget = buffers.GeometryWork.Allocate(3);
    w.Candidates = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Vertices = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Faces = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Normals = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Meshlets = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.BoundsTiles = AllocateElementWork(buffers.GeometryWork,1u<<24u);
    for (uint32_t level=0u; level<w.BoundsLevels.size(); ++level)
        w.BoundsLevels[level]=AllocateElementWork(buffers.GeometryWork,1u<<(16u-level*8u));
    auto &result = work.emplace(entity, std::move(w)).first->second;
    work_allocation.Commit();
    return result;
}
} // namespace

void ReleaseMeshEditWork(state::Scene &r, state::Entity entity) {
    auto *scene = r.Context.find<GpuSceneState>();
    if (!scene) return;
    auto &work = scene->EditWork;
    const auto it = work.find(entity);
    if (it == work.end()) return;
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &w = it->second;
    for (auto range : {w.Candidates, w.Vertices, w.Faces, w.Normals, w.Meshlets, w.BoundsTiles}) buffers.GeometryWork.Release(WorkStorageRange(range));
    for (const auto level : w.BoundsLevels) buffers.GeometryWork.Release(WorkStorageRange(level));
    buffers.GeometryWork.Release(w.WorkBudget);
    work.erase(it);
    scene->InstanceFlagsStale = true;
}

namespace {
using GeometryEditJobs = std::vector<std::pair<state::Entity, CommitPosedGeometryPushConstants>>;

// Prepares each job's reusable topological footprint before any geometry writes, one phase for every job per submit.
// Stale candidates seed from the vertex selection in the first submit.
// Counts bound sparse allocation between the phases.
// Repeated parameter changes reuse the same tables.
void PrepareGeometryFootprints(state::Scene &r, GeometryEditJobs &jobs) {
    struct Footprint {
        MeshEditWork *Work;
        CommitPosedGeometryPushConstants *Pc;
        uint64_t VertexBlocks{}, MeshletBlocks{};
    };
    auto &scene = r.Context.get<GpuSceneState>();
    std::vector<Footprint> footprints;
    for (auto &[entity, pc] : jobs)
        if (auto &work = scene.EditWork.at(entity); !work.FootprintReady) footprints.push_back({&work, &pc});
    if (footprints.empty()) return;
    const profile::CpuScope scope{"PrepareGeometryFootprint"};
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    const auto &pipelines = GetPipelines(r);
    const auto update = [](const Footprint &footprint) {
        const auto &work = *footprint.Work;
        auto &pc = *footprint.Pc;
        pc.Candidates = work.Candidates;
        pc.ChangedVertices = work.Vertices;
        pc.Faces = work.Faces;
        pc.Normals = work.Normals;
        pc.Meshlets = work.Meshlets;
        pc.BoundsTiles = work.BoundsTiles;
        pc.BudgetOffset = work.WorkBudget.Offset;
    };
    mtl::ComputeChain chain{meshes.BufferContext()};
    std::vector<ElementWorkSeedJob> seeds;
    std::vector<MeshEditWork *> seeded;
    for (const auto &footprint : footprints) {
        auto &work = *footprint.Work;
        if (work.CandidateReady) continue;
        std::vector<uint32_t> blocks;
        meshes.GetSelectedElements(work.StoreId, Element::Vertex).ForEachBlock([&](uint32_t block, uint32_t) { blocks.push_back(block); });
        ReserveElementWork(buffers.GeometryWork, work.Candidates, uint64_t(WorkBlockCount(buffers.GeometryWork, work.Candidates)) + 2u * blocks.size());
        seeds.push_back(PrepareBlockMembershipWork(chain.Scratch, meshes.Arenas().Vertices, meshes.Get(work.StoreId).Vertices, blocks, work.Candidates,
                                                   meshes.GetSelectionSlot(Element::Vertex)));
        seeded.push_back(&work);
    }
    EncodeElementMembershipWork(r, chain, seeds);
    if (!seeded.empty()) {
        chain.Encode([&](MTL::ComputeCommandEncoder *encoder) {
            for (const auto *work : seeded) FinalizeWork(encoder, slots, pipelines, buffers, {work->Candidates});
            encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        });
    }
    for (const auto &footprint : footprints) {
        const auto &work = *footprint.Work;
        for (auto item : {work.Faces, work.Normals, work.Meshlets, work.BoundsTiles}) ClearElementWork(buffers.GeometryWork, item);
        std::ranges::fill(buffers.GeometryWork.GetMutable(work.WorkBudget), 0u);
    }
    const auto submit = [&](std::initializer_list<uint32_t> phases) {
        for (const auto &footprint : footprints) update(footprint);
        chain.Encode([&](MTL::ComputeCommandEncoder *encoder) {
            for (const auto phase : phases) {
                for (const auto &footprint : footprints) {
                    auto &pc = *footprint.Pc;
                    pc.Phase = phase;
                    encode::BindCompute(encoder, pipelines.CommitPosedGeometry, slots, buffers);
                    encode::SetPushConstants(encoder, pc);
                    DispatchWork(encoder, buffers, phase < 2u ? pc.Candidates : pc.Faces);
                }
                encoder->memoryBarrier(MTL::BarrierScopeBuffers);
                if (phase != 1u && phase != 3u) continue;
                for (const auto &footprint : footprints) {
                    const auto &pc = *footprint.Pc;
                    if (phase == 1u) FinalizeWork(encoder, slots, pipelines, buffers, {pc.Faces, pc.Meshlets, pc.BoundsTiles});
                    else FinalizeWork(encoder, slots, pipelines, buffers, {pc.Normals, pc.Meshlets});
                }
                encoder->memoryBarrier(MTL::BarrierScopeBuffers);
            }
        });
        chain.Submit();
        for (const auto &footprint : footprints) {
            const auto &work = *footprint.Work;
            for (auto item : {work.Faces, work.Normals, work.Meshlets, work.BoundsTiles}) CheckElementWork(buffers.GeometryWork, item);
        }
    };
    submit({0u});
    for (auto *work : seeded) {
        CheckElementWork(buffers.GeometryWork, work->Candidates);
        work->CandidateReady = true;
    }
    const auto &a = meshes.Arenas();
    for (auto &footprint : footprints) {
        auto &work = *footprint.Work;
        const auto candidates = WorkBlockCount(buffers.GeometryWork, work.Candidates);
        ReserveElementWork(buffers.GeometryWork, work.Vertices, candidates);
        ReserveElementWork(buffers.GeometryWork, work.BoundsTiles, candidates);
        // A table's distinct blocks are at most the blocks its domain spans: the mesh's face and vertex sets, and its owner's meshlet index.
        const auto &record = meshes.Get(work.StoreId);
        const uint64_t face_blocks = record.FaceData ? a.FaceTriangles.Set(record.FaceData).BlockCount : 0u;
        footprint.VertexBlocks = a.Vertices.Set(record.Vertices).BlockCount;
        if (const auto *owner = buffers.TryMeshOf(work.StoreId)) buffers.ActiveMeshlets.ForEachBlock(owner->MeshletRoot, [&](uint32_t) { ++footprint.MeshletBlocks; });
        const auto budget = buffers.GeometryWork.Get(work.WorkBudget);
        ReserveElementWork(buffers.GeometryWork, work.Faces, std::min<uint64_t>(budget[0], face_blocks));
        ReserveElementWork(buffers.GeometryWork, work.Meshlets, std::min<uint64_t>(budget[2], footprint.MeshletBlocks));
    }
    submit({1u, 2u});
    for (const auto &footprint : footprints) {
        auto &work = *footprint.Work;
        const auto budget = buffers.GeometryWork.Get(work.WorkBudget);
        ReserveElementWork(buffers.GeometryWork, work.Normals, std::min<uint64_t>(budget[1], footprint.VertexBlocks));
        ReserveElementWork(buffers.GeometryWork, work.Meshlets, std::min<uint64_t>(budget[2], footprint.MeshletBlocks));
    }
    submit({3u});
    for (const auto &footprint : footprints) {
        auto &work = *footprint.Work;
        const auto bounds = WorkBlockCount(buffers.GeometryWork, work.BoundsTiles);
        for (auto &level : work.BoundsLevels) ReserveElementWork(buffers.GeometryWork, level, bounds);
        update(footprint);
        work.FootprintReady = true;
    }
}

CommitPosedGeometryPushConstants PrepareGeometryEdit(state::Scene &r, state::Entity entity, state::Entity primary, const PendingTransform *pending, const PosedNamespaces *pose = nullptr, std::span<const Range> changed = {}) {
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &meshes = r.Context.get<MeshStore>();
    auto &w = PrepareMeshEditWork(r, entity);
    const auto mesh = GetMesh(r, entity);
    const auto id = w.StoreId;
    if (changed.empty()) w.RefreshRanges.clear();
    if (!changed.empty()) {
        const bool same_ranges = w.CandidateReady && w.FootprintReady && w.PreviewActive &&
            std::ranges::equal(changed, w.RefreshRanges, [](Range a, Range b) {
                return a.Offset == b.Offset && a.Count == b.Count;
            });
        if (!same_ranges) {
            w.FootprintReady = false;
            SeedElementWorkRanges(buffers.GeometryWork, w.Candidates, changed, 0u, w.PreviewActive);
            w.RefreshRanges.assign(changed.begin(), changed.end());
        }
        w.CandidateReady = true;
    } else if (!w.CandidateReady || !w.PreviewActive || (!pose && r.Context.get<const GpuSceneState>().EditSelectionDirty)) {
        // The footprint preparation seeds the candidates from the vertex selection.
        w.CandidateReady = w.FootprintReady = false;
        if (!w.PreviewActive) ClearElementWork(buffers.GeometryWork, w.Candidates);
    }
    ClearElementWork(buffers.GeometryWork, w.Vertices);
    auto entry = MakeDeriveEntryInputs(meshes, id).value_or(NormalDeriveEntry{.VertexCount = mesh.VertexCount(), .Connectivity = meshes.GetConnectivityRef(id)});
    if (pose) {
        for (const auto &level : w.BoundsLevels) ClearElementWork(buffers.GeometryWork, level);
        entry.PositionNamespace = pose->PositionNamespace(0);
        if (const auto normals = pose->NormalsAt(0)) {
            entry.VertexNormalNamespace = normals->Vertex;
            entry.SectorNamespace = normals->Sector;
            entry.FaceNormalNamespace = normals->Face;
        }
        w.PreviewActive = pending != nullptr;
    }
    return {
        .Vertices = {meshes.Slots().Vertices, meshes.Arenas().Vertices.First(meshes.Get(id).Vertices)},
        .PositionSlot = buffers.PosedPositions.Values.Buffer.Slot,
        .PositionNodesSlot = buffers.PosedPositions.Nodes.Buffer.Slot,
        .SelectionSlot = meshes.GetSelectionSlot(Element::Vertex),
        .Candidates = w.Candidates,
        .ChangedVertices = w.Vertices,
        .Faces = w.Faces,
        .Normals = w.Normals,
        .Meshlets = w.Meshlets,
        .BoundsTiles = w.BoundsTiles,
        .Entry = entry,
        .Primary = pending ? static_cast<Transform>(r.get<const WorldTransform>(primary)) : Transform{},
        .Delta = pending ? pending->Delta : Transform{},
        .Pivot = pending ? pending->Pivot : vec3{},
        .FaceTriangleStartSlot = meshes.Slots().FaceTriangleStart,
        .Topology = mesh.PrimitiveTopology(),
        .ElementMeshlets = MeshBuffersOf(r,entity).RenderTopology == InvalidOffset ? ElementAttributeRef{} : buffers.ElementMeshlets[MeshBuffersOf(r,entity).RenderTopology].Ref(),
        .ApplyTransform = pending ? 1u : 0u,
        .Mode = !changed.empty() ? GeometryEditMode::Refresh : pose ? GeometryEditMode::Preview :
                                                                      GeometryEditMode::Commit,
    };
}

void RecordGeometryEditBatch(state::Scene &r, MTL::ComputeCommandEncoder *encoder, GeometryEditJobs &commits, bool posed) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &pipelines = GetPipelines(r);
    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    const auto entries = buffers.GeometryNormalEntries.SetCount<NormalDeriveEntry>(commits.size());
    for (uint32_t i = 0; i < commits.size(); ++i) entries[i] = commits[i].second.Entry;
    r.Context.get<const mtl::Context>().CommitResidency();
    for (auto &[_, pc] : commits) {
        // Posed outputs have no history. A commit writes the canonical base normals.
        if (!posed && pc.Entry.FaceCount) {
            auto entry=pc.Entry;
            entry.VerticesWork=pc.Normals; entry.FacesWork=pc.Faces;
            entry.VertexWorkCount=buffers.GeometryWork.Get({pc.Normals.Storage.Offset+5u,1u})[0];
            entry.FaceWorkCount=buffers.GeometryWork.Get({pc.Faces.Storage.Offset+5u,1u})[0];
            CaptureNormalWrites(r,entry,buffers.GeometryWork,buffers.GeometryWork);
        }
        pc.Phase = 4u;
        encode::BindCompute(encoder, pipelines.CommitPosedGeometry, slots, buffers);
        encode::SetPushConstants(encoder, pc);
        DispatchWork(encoder, buffers, pc.Candidates);
    }
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    for (const auto &[_, pc] : commits) FinalizeWork(encoder, slots, pipelines, buffers, {pc.ChangedVertices});
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    auto derive = MakeNormalDerivePc(buffers, meshes, posed ? buffers.PosedVertexNormals.Values.Buffer.Slot : meshes.Slots().BaseVertexNormal, posed ? buffers.PosedFaceNormals.Values.Buffer.Slot : meshes.Slots().BaseFaceNormal);
    derive.EntriesSlot = buffers.GeometryNormalEntries.Slot;
    for (uint32_t phase = 0; phase < 2; ++phase) {
        derive.Phase = phase;
        for (uint32_t i = 0; i < commits.size(); ++i) {
            if (commits[i].second.Entry.FaceCount == 0) continue;
            derive.EntryIndex = i;
            derive.Work = phase == 0 ? commits[i].second.Faces : commits[i].second.Normals;
            encode::BindCompute(encoder, GetMeshPipelines(r)[MeshPass::VertexNormalDerive], slots, buffers);
            encode::SetPushConstants(encoder, derive);
            DispatchWork(encoder, buffers, derive.Work);
        }
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    }
}
} // namespace

void RefreshEditedPositions(state::Scene &r, std::span<const MeshVertexChanges> changes) {
    const mtl::AutoreleaseScope native_scope;
    if (changes.empty()) return;
    const profile::CpuScope scope{"RefreshEditedPositions"};
    GeometryEditJobs jobs;
    for (const auto &[entity, ranges] : changes) jobs.emplace_back(entity, PrepareGeometryEdit(r, entity, state::Null, nullptr, nullptr, ranges));
    PrepareGeometryFootprints(r, jobs);
    const auto &ctx = r.Context.get<const mtl::Context>();
    auto *cb = ctx.Queue->commandBuffer();
    {
        mtl::PassChain chain{cb};
        RecordGeometryEditBatch(r, chain.BeginCompute("RefreshGeometry"), jobs, false);
    }
    cb->commit();
    cb->waitUntilCompleted();
    auto &scene = r.Context.get<GpuSceneState>();
    scene.EditPreludePending = true;
    for (const auto &[entity, ranges] : changes) {
        auto &work = scene.EditWork.at(entity);
        work.Modified = work.PreviewActive = work.RequiresPose = true;
    }
}

std::vector<state::Entity> CommitPosedGeometry(state::Scene &r, state::Entity viewport, std::span<const state::Entity> mesh_entities) {
    const mtl::AutoreleaseScope native_scope;
    const profile::CpuScope scope{"CommitGeometry"};
    const auto *pending = r.try_get<const PendingTransform>(viewport);
    if (!pending) return {};
    const auto primaries = selection::ComputePrimaryEditInstances(r, false);
    auto &buffers = r.Context.get<GpuBuffers>();
    GeometryEditJobs commits;
    for (const auto entity : mesh_entities) {
        if (const auto primary = primaries.find(entity); primary != primaries.end())
            commits.emplace_back(entity, PrepareGeometryEdit(r, entity, primary->second, pending));
    }
    if (commits.empty()) return {};
    PrepareGeometryFootprints(r, commits);
    auto &meshes = r.Context.get<MeshStore>();
    for (const auto &[entity, pc] : commits) meshes.CaptureVertexEdit(GetMesh(r, entity).GetStoreId());
    const auto &ctx = r.Context.get<const mtl::Context>();
    auto *cb = ctx.Queue->commandBuffer();
    {
        mtl::PassChain chain{cb};
        auto *encoder = chain.BeginCompute("CommitGeometry");
        RecordGeometryEditBatch(r, encoder, commits, false);
    }
    cb->commit();
    cb->waitUntilCompleted();
    std::vector<state::Entity> changed;
    std::vector<MeshStore::SelectionUpdate> aggregates;
    std::vector<MeshletBoundsRefitJob> refits;
    std::vector<MeshletIndexEdit> dirty_edits;
    std::vector<std::vector<uint32_t>> dirty_meshlets;
    std::vector<state::Entity> dirty_entities;
    for (const auto &[entity, pc] : commits) {
        if (!ElementWorkEmpty(buffers.GeometryWork, pc.ChangedVertices)) {
            changed.push_back(entity);
            const auto id = GetMesh(r, entity).GetStoreId();
            if (meshes.Get(id).SelectionSummary.Count) {
                auto &blocks = aggregates.emplace_back(MeshStore::SelectionUpdate{.StoreId = id}).Blocks[0];
                ForEachWorkBlock(buffers.GeometryWork, pc.ChangedVertices, [&](uint32_t block, auto) { blocks.push_back(block); });
            }
            auto &w = r.Context.get<GpuSceneState>().EditWork.at(entity);
            w.Modified = true;
            w.PreviewActive = true;
            w.RequiresPose = true;
            auto &owner=MeshBuffersOf(r,entity);
            refits.push_back({&owner,&buffers.GeometryWork,w.Meshlets});
            if (buffers.ClusterGroupCount(owner)>0u && !ElementWorkEmpty(buffers.GeometryWork,w.Meshlets)) {
                dirty_edits.push_back({.Root=owner.PositionDirtyRoot});
                auto &meshlets=dirty_meshlets.emplace_back();
                ForEachWorkElement(buffers.GeometryWork,w.Meshlets,[&](uint32_t id) { meshlets.push_back(id); });
                dirty_entities.push_back(entity);
            }
        }
    }
    RefitCanonicalMeshletBounds(r,refits);
    if (!dirty_edits.empty()) {
        for (uint32_t i=0u;i<dirty_edits.size();++i) dirty_edits[i].Added=dirty_meshlets[i];
        buffers.ActiveMeshlets.Update(dirty_edits);
        for (uint32_t i=0u;i<dirty_edits.size();++i) {
            MeshBuffersOf(r,dirty_entities[i]).PositionDirtyRoot=dirty_edits[i].Root;
            r.Context.get<GpuSceneState>().PositionDirty.insert(dirty_entities[i]);
        }
    }
    meshes.UpdateSelection(r, aggregates);
    RefreshElementSelectionSummaries(r, changed);
    return changed;
}

namespace {
void RecordSparseEditPrelude(state::Scene &r, state::Entity viewport, mtl::PassChain &chain) {
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &state = r.Context.get<GpuSceneState>();
    const auto &pipelines = GetPipelines(r);
    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    const auto *pending = r.try_get<const PendingTransform>(viewport);
    const auto primaries = selection::ComputePrimaryEditInstances(r, false);
    GeometryEditJobs jobs;
    for (const auto &[entity, pose] : state.PosedByEntity) {
        const auto primary = primaries.find(entity);
        const bool preview = pending && primary != primaries.end();
        const auto old = state.EditWork.find(entity);
        if (!preview && (old == state.EditWork.end() || !old->second.PreviewActive)) continue;
        jobs.emplace_back(entity, PrepareGeometryEdit(r, entity, preview ? primary->second : state::Null, preview ? pending : nullptr, &pose));
    }
    if (jobs.empty()) return;
    PrepareGeometryFootprints(r, jobs);
    auto *encoder = chain.BeginCompute("EditGeometry", MTL::StageDispatch);
    RecordGeometryEditBatch(r, encoder, jobs, true);
    const auto entries = buffers.BoundsReduceEntries.GetSpan<BoundsEntry>({0, buffers.BoundsReduceEntries.Count<BoundsEntry>()});
    for (const auto &[entity, job] : jobs) {
        auto &w = state.EditWork.at(entity);
        const auto &pose = state.PosedByEntity.at(entity);
        const auto entry_it = std::ranges::find(entries, pose.FirstInstance, &BoundsEntry::FirstInstance);
        assert(entry_it != entries.end());
        const auto entry_index = uint32_t(entry_it - entries.begin());
        RecordPosedMeshletBounds(encoder,slots,pipelines,buffers,0,{.Work=w.Meshlets,.Instance=pose.FirstInstance});
        RecordBoundsPass(encoder,slots,pipelines.BoundsReduce,buffers,PreludeSlot::PosePrepass,0,
            {.Work=w.BoundsTiles,.NextWork=w.BoundsLevels[0],.EntryIndex=entry_index});
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        for (uint32_t level=1u; level<VertexBoundsLevels; ++level) {
            const auto work=w.BoundsLevels[level-1u];
            FinalizeWork(encoder,slots,pipelines,buffers,{work});
            encoder->memoryBarrier(MTL::BarrierScopeBuffers);
            RecordBoundsPass(encoder,slots,pipelines.BoundsCombine,buffers,PreludeSlot::BoundsLevel1,0,
                {.Work=work,.NextWork=level+1u<VertexBoundsLevels ? w.BoundsLevels[level] : ElementWork{},.EntryIndex=entry_index,.Level=level});
            encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        }
    }
}
} // namespace

void SyncPreludeDispatchArgs(GpuBuffers &buffers) {
    const bool live = std::exchange(buffers.PreludeStale, false);
    buffers.MeshletOcclusionStale = false;
    const auto &groups = buffers.Prelude;
    // Array order is the PreludeSlot order.
    const std::array<MTL::DispatchThreadgroupsIndirectArguments, GpuBuffers::PreludeGroups::PassCount> args{{
        {live ? groups.PosePrepass : 0u, 1u, 1u},
        {live ? groups.PosedMeshletBounds : 0u, 1u, 1u},
        {live ? groups.DeriveFaces : 0u, 1u, 1u},
        {live ? groups.BoundsCombine[0] : 0u, 1u, 1u},
        {live ? groups.DeriveGather : 0u, 1u, 1u},
        {live ? groups.BoundsCombine[1] : 0u, 1u, 1u},
        {live ? groups.BoundsCombine[2] : 0u, 1u, 1u},
    }};
    buffers.PreludeDispatchArgs.Update(as_bytes(args));
}
