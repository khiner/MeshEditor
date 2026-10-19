#include "viewport/ViewportRenderGpu.h"
#include "render/LightComponents.h"

#include "Camera.h"
#include "Profile.h"
#include "SortUnique.h"
#include "Variant.h"
#include "animation/MorphWeights.h"
#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "gizmo/GizmoInteraction.h"
#include "gpu/BoundsReducePushConstants.h"
#include "gpu/CommitPosedGeometryPushConstants.h"
#include "gpu/DepthPyramidReducePushConstants.h"
#include "gpu/MeshletCullPushConstants.h"
#include "gpu/MotionBlurAccumulatePushConstants.h"
#include "gpu/MotionBlurGatherPushConstants.h"
#include "gpu/MotionBlurResolvePushConstants.h"
#include "gpu/MotionBlurTilesFlattenPushConstants.h"
#include "gpu/ObjectOriginPushConstants.h"
#include "gpu/OverlayDispatch.h"
#include "gpu/OverlayJobCullPushConstants.h"
#include "gpu/OverlayJobDrawPushConstants.h"
#include "gpu/PosedMeshletBoundsPushConstants.h"
#include "gpu/SilhouetteEdgeColorPushConstants.h"
#include "gpu/VertexBlockPushConstants.h"
#include "gpu/ViewportCompositePushConstants.h"
#include "gpu/VisibilityId.h"
#include "gpu/WireRasterPushConstants.h"
#include "gpu/WireResolvePushConstants.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "metal/MetalCpp.h"
#include "metal/PassChain.h"
#include "metal/RenderTarget.h"
#include "numeric/MatrixMath.h"
#include "object/PendingSync.h"
#include "procedure/Procedure.h"
#include "render/Encoding.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/MeshTopologyRepair.h"
#include "render/MeshletBoundsRefit.h"
#include "render/Pipelines.h"
#include "render/RenderTargets.h"
#include "render/SceneUpdates.h"
#include "render/Textures.h"
#include "scene/CameraLens.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "timeline/Timeline.h"
#include "viewport/InteractionComponents.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportInteractionState.h"

namespace {
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

ExtrasLine ExtrasGizmoParams(const state::Registry &r, state::Entity object, ObjectType type) {
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
        return {ExtrasLineKind::Camera, vec4{half_w, half_h, depth, 0.f}, 11};
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

// Write overlay jobs in stable threadgroup chunks; filter display and selection state on the GPU.
void BuildOverlayJobs(const state::Registry &r, GpuBuffers &buffers, bool bounding_boxes) {
    constexpr uint32_t LinesPerJob{uint32_t(OverlayDispatch::LineGroupLines)};
    const auto object_ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>();
    const auto for_each_instance = [&](state::Entity buffer_entity, auto &&visit) {
        const auto &models = r.get<const ModelsBuffer>(buffer_entity);
        for (uint32_t slot = models.InstanceRange.Offset; slot < models.InstanceRange.Offset + models.InstanceCount; ++slot) visit(r.EntityAt(ObjectIndex(object_ids[slot])), slot);
    };
    // Calls `emit` with each job before its split into line groups, and its line count.
    const auto for_each_job = [&](auto &&emit) {
        for (const auto buffer_entity : r.view<const ObjectExtrasTag, const ModelsBuffer>()) {
            for_each_instance(buffer_entity, [&](state::Entity object, uint32_t slot) {
                const auto gizmo = ExtrasGizmoParams(r, object, r.get<const ObjectKind>(object).Value);
                emit(OverlayJob{.Kind = OverlayJobKind::Extras, .InstanceIndex = slot, .ExtrasKind = gizmo.Kind, .LocalOffset = vec3{0}, .Params = gizmo.Params}, gizmo.LineCount);
            });
        }
        for (const auto [entity, shape, render_instance] : r.view<const ColliderShape, const RenderInstance>().each()) {
            const auto wire = ColliderWireParams(shape.Shape);
            emit(OverlayJob{.Kind = OverlayJobKind::Extras, .InstanceIndex = render_instance.BufferIndex, .ExtrasKind = wire.Kind, .LocalOffset = shape.LocalOffset, .Params = wire.Params}, wire.LineCount);
        }
        if (bounding_boxes) {
            for (const auto [entity, instance, render_instance] : r.view<const Instance, const RenderInstance>().each()) {
                if (HasMesh(r, instance.Entity)) emit(OverlayJob{.Kind = OverlayJobKind::Bounds, .InstanceIndex = render_instance.BufferIndex}, 12u);
            }
        }
        for (const auto [mesh_entity, tets] : r.view<const TetBuffers>().each()) {
            if (tets.EdgeIndices.Count == 0u || !r.all_of<ModelsBuffer>(mesh_entity)) continue;
            for_each_instance(mesh_entity, [&](state::Entity, uint32_t slot) {
                emit(OverlayJob{.Kind = OverlayJobKind::TetWire, .InstanceIndex = slot, .SourceOffset = tets.Positions.Offset, .IndexOffset = tets.EdgeIndices.Offset}, tets.EdgeIndices.Count / 2u);
            });
        }
    };
    uint32_t count = 0u;
    for_each_job([&](const OverlayJob &, uint32_t lines) { count += (lines + LinesPerJob - 1u) / LinesPerJob; });
    const auto jobs = buffers.ResizeOverlayJobs(count);
    uint32_t write = 0u;
    for_each_job([&](OverlayJob job, uint32_t lines) {
        for (uint32_t first = 0u; first < lines; first += LinesPerJob) {
            job.FirstElement = first;
            job.ElementCount = std::min(LinesPerJob, lines - first);
            jobs[write++] = job;
        }
    });
}

void RecordSceneCounters(const GpuBuffers &buffers, const Viewport &viewport) {
    profile::RecordCounter("InstanceSlots", buffers.Instances.TransformBuffer.Count<Transform>());
    profile::RecordCounter("MeshletRecords", buffers.Render->Meshlets.Buffer.Count<MeshletRecord>());
    profile::RecordCounter("MeshletInstances", buffers.MeshletInstanceCount);
    profile::RecordCounter("MeshletRecordBytes", buffers.Render->Meshlets.Buffer.UsedSize);
    profile::RecordCounter("MeshletTriangleIdBytes", buffers.Render->MeshletTriangleIds.Buffer.UsedSize);
    profile::RecordCounter("PrimitiveRecordBytes", buffers.Render->Primitives.Buffer.UsedSize);
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
        profile::RecordCounter("SelectedCoarseMeshlets", *reinterpret_cast<const uint32_t *>(viewport.MeshletCoarseCount.Contents().data()));
        profile::RecordCounter("VisibleBlendMeshlets", count(MeshletRoute::Blend));
        profile::RecordCounter("VisibleTransmissionMeshlets", count(MeshletRoute::Transmission));
    }
    if (buffers.EditCull.Routes.Contents().size() >= sizeof(MeshletRouteState)) {
        const auto &routes = *reinterpret_cast<const MeshletRouteState *>(buffers.EditCull.Routes.Contents().data());
        profile::RecordCounter("VisibleEditOverlayMeshlets", routes.Counts[uint32_t(MeshletRoute::EditOverlay)]);
    }
    profile::RecordCounter("DeviceAllocatedBytes", buffers.Ctx.Ctx.Device->currentAllocatedSize());
}

struct MeshDeform {
    uint32_t BoneDeformOffset{InvalidOffset}, ArmatureDeformOffset{InvalidOffset}, MorphDeformOffset{InvalidOffset};
    uint32_t MorphTargetCount{0};
    // The armature data entities whose poses deform the mesh.
    std::vector<state::Entity> Armatures;
    // By instance in slot order, InvalidOffset for an instance without that deformation, and empty when no instance has it.
    std::vector<uint32_t> ArmatureOffsets, WeightOffsets;
    // Instances of one mesh can bind different armatures and morph weights.
    bool PerInstance() const { return !ArmatureOffsets.empty() || !WeightOffsets.empty(); }
};

MeshDeform DeformOf(const state::Registry &r, const GpuBuffers &buffers, const MeshStore::Record &record, const ModelsBuffer &models) {
    MeshDeform deform;
    const auto object_ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>({models.InstanceRange.Offset, models.InstanceCount});
    for (uint32_t i = 0u; i < object_ids.size(); ++i) {
        const auto instance = r.EntityAt(ObjectIndex(object_ids[i]));
        if (const auto *modifier = record.SkinBlocksReady ? r.try_get<const ArmatureModifier>(instance) : nullptr) {
            const auto *pose_state = r.try_get<const ArmaturePoseState>(modifier->ArmatureEntity);
            if (pose_state && modifier->SkinSlot < pose_state->GpuDeformRanges.size()) {
                const auto offset = pose_state->GpuDeformRanges[modifier->SkinSlot].Offset;
                if (deform.BoneDeformOffset == InvalidOffset) {
                    deform.BoneDeformOffset = 0u;
                    deform.ArmatureDeformOffset = offset;
                }
                if (!std::ranges::contains(deform.Armatures, modifier->ArmatureEntity)) deform.Armatures.push_back(modifier->ArmatureEntity);
                if (deform.ArmatureOffsets.empty()) deform.ArmatureOffsets.assign(object_ids.size(), InvalidOffset);
                deform.ArmatureOffsets[i] = offset;
            }
        }
        if (const auto *weights = record.MorphBlocksReady ? r.try_get<const MorphWeightRange>(instance) : nullptr) {
            deform.MorphDeformOffset = 0u;
            deform.MorphTargetCount = record.MorphTargetCount;
            if (deform.WeightOffsets.empty()) deform.WeightOffsets.assign(object_ids.size(), InvalidOffset);
            deform.WeightOffsets[i] = weights->Weights.Offset;
        }
    }
    return deform;
}

// Edit overlay draws skip geometry whose projected diameter falls below this many pixels.
constexpr float MinEditOverlayDiameterPixels{2.0f};

// Threadgroup memory lengths must be 16-byte multiples.
constexpr uint32_t AlignedThreadgroupBytes(uint32_t bytes) { return (bytes + 15u) & ~15u; }

// Each prelude pass's slot in PreludeSource::Groups, in recorded order.
enum class PreludeSlot : uint32_t { PosePrepass,
                                    PosedMeshletBounds,
                                    DeriveFaces,
                                    BoundsLevel1,
                                    DeriveGather,
                                    BoundsLevel2,
                                    BoundsLevel3,
                                    Count };

struct PreludeSource {
    uint32_t MeshletJobCount;
    std::array<uint32_t, VertexBoundsLevels> FirstTiles;
    uint32_t DeriveGatherFirstTile;
    std::array<uint32_t, uint32_t(PreludeSlot::Count)> Groups; // In PreludeSlot order.

    bool Has(PreludeSlot slot) const { return Groups[uint32_t(slot)] > 0u; }
    bool Empty() const {
        return std::ranges::all_of(Groups, [](uint32_t groups) { return groups == 0u; });
    }
    void Dispatch(MTL::ComputeCommandEncoder *encoder, PreludeSlot slot, MTL::Size group_size) const {
        if (Has(slot)) encoder->dispatchThreadgroups(MTL::Size(Groups[uint32_t(slot)], 1, 1), group_size);
    }
};

// Address metadata only.
// Unchanged namespace revisions skip this enumeration.
template<typename T>
std::vector<uint32_t> PoseElementBlocks(const ElementArena<T> &arena, ElementSetRef set) {
    std::vector<uint32_t> blocks;
    arena.ForEachBlock(set, [&](uint32_t block, const auto &) { blocks.push_back(block); });
    return blocks;
}

// Index posed sector normals by corner-block addresses; use no blocks for records without sector roots.
std::vector<uint32_t> SectorBlocks(const MeshStore &meshes, uint32_t store_id) {
    const auto &record = meshes.Get(store_id);
    return record.NormalSectors.Count ? PoseElementBlocks(meshes.Arenas().FaceCorners, record.FaceCorners) : std::vector<uint32_t>{};
}

// Writes the pose's position and normal namespaces for instance `i` into `out`, InvalidOffset for a pose without derived normals.
void SetPoseNamespaces(auto &out, const PosedNamespaces &pose, uint32_t i) {
    const auto normals = pose.NormalsAt(i).value_or(PosedNamespaces::NormalNamespaces{});
    out.PositionNamespace = pose.PositionNamespace(i);
    out.VertexNormalNamespace = normals.Vertex;
    out.SectorNamespace = normals.Sector;
    out.FaceNormalNamespace = normals.Face;
}

// Require scene bindings on the encoder before recording.

void RecordPosePrepass(MTL::ComputeCommandEncoder *encoder, const Pipelines &pipelines, const GpuBuffers &buffers, const PreludeSource &source) {
    encoder->setComputePipelineState(pipelines.PosePrepass.State());
    const BoundsReducePushConstants pc{
        .BoundsEntrySlot = buffers.BoundsReduceEntries.Buffer.Slot,
        .TileMapSlot = buffers.SparseBoundsTiles.Slot,
        .ValuesSlot = buffers.VertexBounds.Values.Buffer.Slot,
        .NodesSlot = buffers.VertexBounds.Nodes.Buffer.Slot,
        .MembersSlot = buffers.VertexBounds.Members.Slot,
        .FirstTile = source.FirstTiles[0],
    };
    encode::SetPushConstants(encoder, pc);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector, 0);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector, 1);
    source.Dispatch(encoder, PreludeSlot::PosePrepass, ThreadgroupSize::Linear256);
}

void RecordNormalDerive(MTL::ComputeCommandEncoder *encoder, const mtl::ComputePipeline &pipeline, NormalDerivePushConstants pc, const PreludeSource &source, PreludeSlot slot) {
    encoder->setComputePipelineState(pipeline.State());
    encode::SetPushConstants(encoder, pc);
    source.Dispatch(encoder, slot, ThreadgroupSize::Linear256);
}

// Return resources for base or posed normal derivation in the namespace selected by each entry.
NormalDerivePushConstants MakeNormalDerivePc(const GpuBuffers &buffers, const MeshStore &meshes, bool posed) {
    return {
        .EntriesSlot = buffers.NormalDeriveEntries.Buffer.Slot,
        .EdgeSharpnessSlot = meshes.Slots().EdgeSharpness,
        .FaceSharpnessSlot = meshes.Slots().FaceSharpness,
        .TileMapSlot = buffers.SparseDeriveTiles.Slot,
        .PositionSlot = buffers.PosedPositions.Values.Buffer.Slot,
        .PositionNodesSlot = buffers.PosedPositions.Nodes.Buffer.Slot,
        .VertexNormalSlot = posed ? buffers.PosedVertexNormals.Values.Buffer.Slot : meshes.Slots().BaseVertexNormal,
        .VertexNormalNodesSlot = buffers.PosedVertexNormals.Nodes.Buffer.Slot,
        .PosedSectorNodesSlot = buffers.PosedSectors.Nodes.Buffer.Slot,
        .PosedSectorValuesSlot = buffers.PosedSectors.Values.Buffer.Slot,
        .FaceNormalSlot = posed ? buffers.PosedFaceNormals.Values.Buffer.Slot : meshes.Slots().BaseFaceNormal,
        .FaceNormalNodesSlot = buffers.PosedFaceNormals.Nodes.Buffer.Slot,
        .BaseFaceNormalSlot = meshes.Slots().BaseFaceNormal,
    };
}

// One bounds level over `pc.Work`'s elements when it names element work, else over the source's sparse tiles at `pc.Level`.
void RecordBoundsPass(MTL::ComputeCommandEncoder *encoder, const mtl::ComputePipeline &pipeline, const GpuBuffers &buffers, BoundsReducePushConstants pc, std::optional<const PreludeSource &> source = {}, PreludeSlot slot = PreludeSlot::BoundsLevel1) {
    pc.BoundsEntrySlot = buffers.BoundsReduceEntries.Buffer.Slot;
    pc.BoundsSlot = buffers.Instances.BoundsBuffer.Slot;
    pc.ValuesSlot = buffers.VertexBounds.Values.Buffer.Slot;
    pc.NodesSlot = buffers.VertexBounds.Nodes.Buffer.Slot;
    pc.MembersSlot = buffers.VertexBounds.Members.Slot;
    if (source) {
        pc.TileMapSlot = buffers.SparseBoundsTiles.Slot;
        pc.FirstTile = source->FirstTiles[pc.Level];
    }
    encoder->setComputePipelineState(pipeline.State());
    encode::SetPushConstants(encoder, pc);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector, 0);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector, 1);
    if (source) source->Dispatch(encoder, slot, ThreadgroupSize::Linear256);
    else encoder->dispatchThreadgroups(*buffers.GeometryWork.Buffer, WorkArgsOffset(pc.Work, true), ThreadgroupSize::Linear256);
}

// Posed meshlet bounds over `pc.MeshletAddresses` when it names element work, else over the source's sparse jobs.
void RecordPosedMeshletBounds(MTL::ComputeCommandEncoder *encoder, const Pipelines &pipelines, const GpuBuffers &buffers, PosedMeshletBoundsPushConstants pc, std::optional<const PreludeSource &> source = {}) {
    if (source) {
        pc.JobsSlot = buffers.SparsePosedMeshletBoundsJobs.Slot;
        pc.JobCount = source->MeshletJobCount;
    }
    pc.MeshletSlot = buffers.Render->Meshlets.Buffer.Slot;
    pc.MeshletVertexSlot = buffers.Render->MeshletVertexCorners.Buffer.Slot;
    pc.PosedMeshletBoundsSlot = buffers.PosedMeshletBounds.Values.Buffer.Slot;
    pc.PosedMeshletBoundsNodesSlot = buffers.PosedMeshletBounds.Nodes.Buffer.Slot;
    pc.Members = buffers.Render->LodMembers.Slots();
    pc.Meshlets = buffers.Render->Meshlets.Slots();
    pc.MemberSlot = buffers.Render->LodMembers.Buffer.Slot;
    encoder->setComputePipelineState(pipelines.PosedMeshletBounds.State());
    encode::SetPushConstants(encoder, pc);
    if (source) source->Dispatch(encoder, PreludeSlot::PosedMeshletBounds, ThreadgroupSize::Linear32);
    else encoder->dispatchThreadgroups(*buffers.GeometryWork.Buffer, WorkArgsOffset(pc.MeshletAddresses, true), ThreadgroupSize::Linear32);
}

// Records the posed prelude over the source: poses and leaf bounds, normal derivation, posed meshlet bounds, then the parent bounds levels.
// Bindless dependencies require explicit barriers between pose and bounds levels.
void RecordPrelude(MTL::ComputeCommandEncoder *compute, const Pipelines &pipelines, const MeshPipelines &mesh_pipelines, const GpuBuffers &buffers, const MeshStore &meshes, const PreludeSource &source) {
    if (source.Has(PreludeSlot::PosePrepass)) {
        RecordPosePrepass(compute, pipelines, buffers, source);
        compute->memoryBarrier(MTL::BarrierScopeBuffers);
    }
    if (source.Has(PreludeSlot::DeriveFaces)) {
        auto derive = MakeNormalDerivePc(buffers, meshes, true);
        const auto &normal_pipeline = mesh_pipelines[MeshPass::VertexNormalDerive];
        RecordNormalDerive(compute, normal_pipeline, derive, source, PreludeSlot::DeriveFaces);
        compute->memoryBarrier(MTL::BarrierScopeBuffers);
        derive.Phase = 1u;
        derive.FirstTile = source.DeriveGatherFirstTile;
        RecordNormalDerive(compute, normal_pipeline, derive, source, PreludeSlot::DeriveGather);
        compute->memoryBarrier(MTL::BarrierScopeBuffers);
    }
    if (source.Has(PreludeSlot::PosedMeshletBounds)) RecordPosedMeshletBounds(compute, pipelines, buffers, {}, source);
    if (source.Has(PreludeSlot::BoundsLevel3)) {
        RecordBoundsPass(compute, pipelines.BoundsCombine, buffers, {.Level = 1u}, source, PreludeSlot::BoundsLevel1);
        compute->memoryBarrier(MTL::BarrierScopeBuffers);
        RecordBoundsPass(compute, pipelines.BoundsCombine, buffers, {.Level = 2u}, source, PreludeSlot::BoundsLevel2);
        compute->memoryBarrier(MTL::BarrierScopeBuffers);
        RecordBoundsPass(compute, pipelines.BoundsCombine, buffers, {.Level = 3u}, source, PreludeSlot::BoundsLevel3);
    }
}

// Gather dirty bounds and optional posed work into sparse prelude lists, then consume the dirty marks.
PreludeSource GatherPrelude(state::Registry &r, bool every_posed) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto entries = buffers.BoundsReduceEntries.Buffer.GetSpan<BoundsEntry>();
    std::array<std::vector<uvec2>, VertexBoundsLevels> bounds;
    std::vector<uvec2> faces, gathers;
    std::vector<PosedMeshletBoundsJob> jobs;
    uint32_t groups = 0u;
    const auto gather = [&](state::Entity mesh_entity, const BoundsRun &run) {
        if (!run.Posed()) {
            bounds.back().push_back({run.Entries.Offset, 0u});
            return;
        }
        auto words = buffers.PreludeTiles.Get(run.Tiles);
        const auto take = [&](uint32_t count) {
            const auto taken = words.first(count);
            words = words.subspan(count);
            return taken;
        };
        for (uint32_t level = 0u; level < VertexBoundsLevels; ++level) {
            const auto keys = take(run.BoundsKeys[level]);
            for (uint32_t i = 0u; i < run.Entries.Count; ++i) {
                for (const auto key : keys) bounds[level].push_back({run.Entries.Offset + i, key});
            }
        }
        const auto face_blocks = take(run.FaceBlocks), vertex_blocks = take(run.VertexBlocks);
        for (uint32_t i = 0u; i < run.DeriveEntries.Count; ++i) {
            for (const auto block : face_blocks) faces.push_back({run.DeriveEntries.Offset + i, block});
            for (const auto block : vertex_blocks) gathers.push_back({run.DeriveEntries.Offset + i, block});
        }
        const auto job_words = take(2u * run.MeshletJobs);
        if (job_words.empty()) return;
        // Every instance sharing an entry's pose shares its bounds namespace, so a job poses its entry's first instance.
        const auto &record = RecordOf(r, mesh_entity);
        for (const auto &entry : entries.subspan(run.Entries.Offset, run.Entries.Count)) {
            for (uint32_t j = 0u; j < job_words.size(); j += 2u) {
                jobs.push_back({groups, job_words[j + 1u], entry.FirstInstance, job_words[j], record.MemberData.Index, record.MeshletData.Index});
                groups += job_words[j + 1u];
            }
        }
    };
    for (const auto [mesh_entity, run] : r.view<const MeshBoundsDirty, const BoundsRun>().each()) gather(mesh_entity, run);
    if (every_posed) {
        for (const auto [mesh_entity, run, _] : r.view<const BoundsRun, const PosedLayout>(state::Exclude<MeshBoundsDirty>).each()) gather(mesh_entity, run);
    }
    r.clear<MeshBoundsDirty>();
    std::array<uint32_t, VertexBoundsLevels> first{};
    uint32_t tile_count = 0u;
    for (uint32_t level = 0u; level < VertexBoundsLevels; ++level) {
        first[level] = tile_count;
        tile_count += uint32_t(bounds[level].size());
    }
    const auto bounds_tiles = buffers.SparseBoundsTiles.SetCount<uvec2>(tile_count);
    for (uint32_t level = 0u; level < VertexBoundsLevels; ++level) std::ranges::copy(bounds[level], bounds_tiles.begin() + first[level]);
    const auto derive_tiles = buffers.SparseDeriveTiles.SetCount<uvec2>(uint32_t(faces.size() + gathers.size()));
    std::ranges::copy(gathers, std::ranges::copy(faces, derive_tiles.begin()).out);
    std::ranges::copy(jobs, buffers.SparsePosedMeshletBoundsJobs.SetCount<PosedMeshletBoundsJob>(uint32_t(jobs.size())).begin());
    return {
        uint32_t(jobs.size()),
        first,
        uint32_t(faces.size()),
        {uint32_t(bounds[0].size()), groups, uint32_t(faces.size()), uint32_t(bounds[1].size()), uint32_t(gathers.size()), uint32_t(bounds[2].size()), uint32_t(bounds[3].size())},
    };
}

MeshletCullPushConstants MakeMeshletCullSlotsPc(const GpuBuffers &buffers, const Viewport &viewport, const MeshletCullOutput &output) {
    return {
        .WorkRangeSlot = buffers.MeshletWorkRanges.Slot,
        .WorkBlockSlot = buffers.MeshletWorkBlocks.Slot,
        .LodNodeSlot = buffers.Render->LodNodes.Buffer.Slot,
        .LodFrontierBlockStateSlot = buffers.LodFrontierBlockStates.Slot,
        .LodExpandArgsSlot = buffers.LodExpandArgs.Slot,
        .WorkStateSlot = buffers.MeshletWorkState.Slot,
        .WorkDispatchArgsSlot = buffers.MeshletWorkDispatchArgs.Slot,
        .BlockStateSlot = buffers.MeshletCullBlocks.Slot,
        .ClassificationSlot = buffers.MeshletClassifications.Slot,
        .VisibleSlot = output.Visible.Slot,
        .InstanceMapSlot = buffers.GpuInstanceSlots.Slot,
        .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
        .PrimitiveSlot = buffers.Render->Primitives.Buffer.Slot,
        .MeshletSlot = buffers.Render->Meshlets.Buffer.Slot,
        .MemberSlot = buffers.Render->LodMembers.Buffer.Slot,
        .Primitives = buffers.Render->Primitives.Slots(),
        .Nodes = buffers.Render->LodNodes.Slots(),
        .Groups = buffers.Render->ClusterGroups.Slots(),
        .Meshlets = buffers.Render->Meshlets.Slots(),
        .Members = buffers.Render->LodMembers.Slots(),
        .ClusterGroupSlot = buffers.Render->ClusterGroups.Buffer.Slot,
        .BoundsSlot = buffers.Instances.BoundsBuffer.Slot,
        .ModelSlot = buffers.Instances.TransformBuffer.Slot,
        .PosedMeshletBoundsSlot = buffers.PosedMeshletBounds.Values.Buffer.Slot,
        .PosedMeshletBoundsNodesSlot = buffers.PosedMeshletBounds.Nodes.Buffer.Slot,
        .RouteStateSlot = output.Routes.Slot,
        .DispatchArgsSlot = output.DispatchArgs.Slot,
        .DispatchChunkCount = output.ChunkCount,
        .DispatchChunkSize = GpuBuffers::MeshletDispatchChunkSize,
        .CoarseCountSlot = viewport.MeshletCoarseCount.Slot,
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
    uint32_t edit_edge_corner = 0u
) {
    return {
        .PrimitiveSlot = buffers.Render->Primitives.Buffer.Slot,
        .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
        .InstanceMapSlot = buffers.GpuInstanceSlots.Slot,
        .MeshletSlot = buffers.Render->Meshlets.Buffer.Slot,
        .Payloads = buffers.Render->PayloadSlots(),
        .VisibleMeshletSlot = output.Visible.Slot,
        .RouteStateSlot = output.Routes.Slot,
        .Route = route,
        .RequiredInstanceFlags = required_instance_flags,
        .EditEdgeCorner = edit_edge_corner,
        .VisibilityTransmission = visibility_transmission,
        .EdgeSharpnessSlot = edge_sharpness_slot,
    };
}

void DrawMeshletList(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, uint32_t route, uint32_t required_instance_flags,
    bool visibility_transmission = false, bool fragment_pc = false,
    uint32_t edge_sharpness_slot = InvalidSlot,
    uint32_t mesh_threads = 160u, uint32_t edit_edge_corner = 0u, const MeshletCullOutput *cull = nullptr
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
        visibility_transmission, edge_sharpness_slot, edit_edge_corner
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

void RecordDepthPyramid(
    MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const GpuBuffers &buffers, const Viewport &viewport, const Pipelines &pipelines,
    const RenderTargets::ResourcesT::Pyramid &pyramid, uint32_t pyramid_sampler, uint32_t first_level,
    uint32_t source_sampler, mtl::Extent2D source_extent, bool nearest, uint32_t ubo_offset
) {
    encode::BindCompute(encoder, pipelines.DepthPyramidReduce, slots, buffers, viewport, ubo_offset);
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
void RecordMotionBlurPostFx(state::Registry &r, const Viewport &viewport, mtl::PassChain &chain, uint32_t ubo_offset) {
    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    const auto &buffers = r.Context.get<const GpuBuffers>();
    const auto &main = GetPipelines(r).Main;
    const auto &targets = viewport.Targets;
    const auto &blur = *targets.MotionBlur;
    const auto &samplers = viewport.Samplers;
    const auto extent = targets.Resources->SceneColorImage.Extent;
    const auto tiles = blur.TileImage.Extent;
    const auto view_ubo = viewport.SceneViewUBO.Contents();
    const auto &view = *reinterpret_cast<const SceneViewUBO *>(view_ubo.data() + ubo_offset);
    auto *compute = chain.BeginCompute("BlurTiles", MTL::StageFragment);
    encode::BindCompute(compute, main.MotionBlurTilesFlatten, slots, buffers, viewport, ubo_offset);
    const auto &previous = *reinterpret_cast<const SceneViewUBO *>(view_ubo.data() + GpuBuffers::SceneViewUboOffset(1));
    const auto &next = *reinterpret_cast<const SceneViewUBO *>(view_ubo.data() + GpuBuffers::SceneViewUboOffset(2));
    const uint32_t camera_motion = view.ViewProj != previous.ViewProj || view.ViewProj != next.ViewProj;
    encode::SetPushConstants(compute, MotionBlurTilesFlattenPushConstants{encode::VisibilityDecodePc(buffers), Inverse(view.ViewProj), camera_motion});
    compute->setBuffer(*viewport.SceneViewUBO, GpuBuffers::SceneViewUboOffset(1), 5);
    compute->setBuffer(*viewport.SceneViewUBO, GpuBuffers::SceneViewUboOffset(2), 6);
    compute->setBuffer(blur.TileIndirection.get(), 0, 7);
    compute->setTexture(*targets.Resources->VisibilityImage, 0);
    compute->setTexture(*targets.Resources->VisibilityDepth, 1);
    compute->setTexture(*blur.VelocityImage, 2);
    compute->setTexture(*blur.TileImage, 3);
    compute->setThreadgroupMemoryLength(16, 0);
    compute->setThreadgroupMemoryLength(16, 1);
    compute->dispatchThreadgroups(MTL::Size(tiles.Width, tiles.Height, 1), ThreadgroupSize::Tile8);
    compute->memoryBarrier(MTL::BarrierScopeTextures | MTL::BarrierScopeBuffers);
    encode::BindCompute(compute, main.MotionBlurTilesDilate, slots, buffers, viewport, ubo_offset);
    compute->setBuffer(blur.TileIndirection.get(), 0, 5);
    compute->setTexture(*blur.TileImage, 0);
    compute->dispatchThreadgroups(MTL::Size((tiles.Width + 7) / 8, (tiles.Height + 7) / 8, 1), ThreadgroupSize::Tile8);

    const std::array colors{mtl::DiscardColor(*blur.OutputImage)};
    const auto pass = mtl::MakePassDescriptor(colors);
    auto *render = encode::BeginScenePass(chain, pass.get(), "BlurGather", {{MTL::StageDispatch | MTL::StageFragment, MTL::StageFragment}}, extent, slots, buffers, viewport, ubo_offset);
    main.MotionBlurGather.Bind(render);
    render->setFragmentBuffer(blur.TileIndirection.get(), 0, 5);
    render->setFragmentTexture(*blur.TileImage, 0);
    const auto inverse_projection = Inverse(viewport.State.Camera.Active.Projection(float(extent.Width) / float(extent.Height)));
    const float noise_phase = r.get<const PlaybackFrame>(state::Document).Value * std::numbers::phi_v<float>;
    encode::SetPushConstants(render, MotionBlurGatherPushConstants{samplers.SceneDepth, samplers.Velocity, samplers.SceneColor, noise_phase - std::floor(noise_phase), {inverse_projection[2].z, inverse_projection[3].z, inverse_projection[2].w, inverse_projection[3].w}});
    render->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
}

struct DisplayContext {
    DisplayContext(const state::Registry &r)
        : R(r), Buffers(r.Context.get<const GpuBuffers>()), Meshes(r.Context.get<const MeshStore>()),
          Displays(r.Context.get<const GpuSceneState>().Displays), Mode(r.get<const Interaction>(state::Document).Mode), EditElement(r.get<const EditMode>(state::Document).Value),
          Primaries(r.get<const EditPrimaries>(state::Document).All), SelectedMeshes(r.get<const SelectionFlags>(state::Document).Meshes),
          Transforming(Mode == InteractionMode::Edit && r.all_of<PendingTransform>(state::Document)) {
        if (Mode == InteractionMode::Excite) {
            for (const auto [_, instance, __] : r.view<const Instance, const SoundVertices>().each()) SoundMeshes.insert(instance.Entity);
        }
    }

    const state::Registry &R;
    const GpuBuffers &Buffers;
    const MeshStore &Meshes;
    const DisplayUnion &Displays;
    InteractionMode Mode;
    Element EditElement;
    const selection::PrimaryEditInstanceMap &Primaries;
    const std::vector<state::Entity> &SelectedMeshes;
    bool Transforming; // An Edit-mode transform is pending.
    std::unordered_set<state::Entity> SoundMeshes;

    // Edit and Pose modes outline the active armature's bones, and Object mode the selected armatures'.
    // Wireframe shading outlines every armature's, which its viewport selects through the Bone flags.
    bool OutlinesArmatureBones(state::Entity armature) const {
        if (Mode == InteractionMode::Edit || Mode == InteractionMode::Pose) return R.all_of<Active>(armature);
        return R.all_of<Selected>(armature);
    }
};

// The bounds entry covering a mesh's instances.
// Without a pose namespace, BoundsCombine copies the mesh's vertex selection root bounds to each instance.
BoundsEntry MeshBoundsEntry(const MeshStore &meshes, uint32_t store_id, const ModelsBuffer &models) {
    const auto &record = meshes.Get(store_id);
    return {
        .FirstInstance = models.InstanceRange.Offset,
        .InstanceCount = models.InstanceCount,
        .Selection = meshes.GetEditSelectionStorage(store_id),
        .VertexBlocksSlot = meshes.Arenas().Vertices.Blocks.Buffer.Slot,
        .VertexOwner = record.Vertices.Index,
        .VertexRoot = record.SelectionSummary.Count ? meshes.GetVertexSelectionRoot(store_id) : SlotOffset{},
    };
}

struct MeshDisplayState {
    MeshDisplay Display;
    uint8_t Overlays{0};
};

// A posed mesh's display fields name its deformation, and the pose namespaces its instances share unless they deform apart.
MeshDisplayState ComputeMeshDisplay(const DisplayContext &c, state::Entity mesh_entity, const MeshStore::Record &mb, const PosedLayout *pose) {
    const auto &r = c.R;
    const auto &meshes = c.Meshes;
    MeshDisplay d{
        .PrimitiveSet = mb.PrimitiveData.Index,
        .NodeSet = mb.NodeData.Index,
        .GroupSet = mb.GroupData.Index,
        .MeshletSet = mb.MeshletData.Index,
        .MemberSet = mb.MemberData.Index,
        .PrimitiveCount = c.Meshes.PrimitiveCount(mb),
        .MorphDeformOffset = pose ? pose->MorphDeformOffset : InvalidOffset,
        .BoneDeformOffset = pose ? pose->BoneDeformOffset : InvalidOffset,
        .MorphTargetCount = pose ? pose->MorphTargetCount : 0u,
        .Payloads = PayloadBasesOf(mb),
    };
    if (pose && !pose->Namespaces.PerInstance) {
        SetPoseNamespaces(d, pose->Namespaces, 0);
        d.MeshletBoundsNamespace = pose->Namespaces.MeshletBoundsNamespace(0);
        d.MorphNormalNamespace = pose->Namespaces.MorphNormalNamespace(0);
    }
    const bool mesh_record = r.all_of<MeshHandle>(mesh_entity);
    const auto &arenas = meshes.Arenas();
    const uint32_t faces = mesh_record ? arenas.FaceTriangles.Count(mb.FaceData) : 0u, edges = mesh_record ? arenas.EdgeHalfedges.Count(mb.EdgeData) : 0u;
    const bool meshlets = c.Meshes.MeshletCount(mb) > 0u;
    const auto primary = c.Primaries.find(mesh_entity);
    const bool has_primary = primary != c.Primaries.end();
    const bool sound = c.SoundMeshes.contains(mesh_entity);
    const auto &displays = c.Displays;
    const bool overlays = displays.Overlays;
    if (has_primary && mesh_record) {
        // A procedure mesh's transform moves its drawn record through the node it cooks.
        d.HasPendingVertexTransform = c.Transforming && !r.all_of<procedure::GeometryProcedure>(mesh_entity) ? 1u : 0u;
        d.PrimaryEditInstanceIndex = r.get<const RenderInstance>(primary->second).BufferIndex;
        d.Selection = meshes.GetEditSelectionStorage(mb.StoreId);
    } else if (sound && mesh_record) {
        d.Selection = meshes.GetEditSelectionStorage(mb.StoreId);
        const auto *active = r.try_get<const MeshActiveElement>(mesh_entity);
        d.ActiveVertex = active ? active->Handle : InvalidOffset;
    }

    const bool armature = r.all_of<ArmatureObject>(mesh_entity), joint = r.all_of<BoneJoint>(mesh_entity);
    // A face-less mesh with materials draws through its material routes in a material-shaded viewport.
    const bool material_face_less = mesh_record && faces == 0u && mb.PrimitiveMaterials.Count > 0u;
    const bool shaded_face_less = material_face_less && displays.AllMaterialShaded;
    const bool selected = std::ranges::binary_search(c.SelectedMeshes, mesh_entity);
    uint32_t flags = 0u;
    uint8_t vertex_overlays = 0u;
    const auto set = [&](MeshletInstanceFlag flag) { flags |= uint32_t(flag); };
    const auto draw = [&](VertexOverlay overlay) { vertex_overlays |= uint8_t(1u << uint32_t(overlay)); };
    if (IsSilhouetteEligible(r, mesh_entity)) set(MeshletInstanceFlag::SilhouetteEligible);
    if (EditPinsFinest(c.Primaries, r, mesh_entity) || c.Meshes.HasDirtyPositions(mb)) set(MeshletInstanceFlag::LodPinFinest);
    if (has_primary && meshlets && mesh_record && Mesh{meshes, mb.StoreId}.ElementCount(c.EditElement) > 0u) set(MeshletInstanceFlag::ElementSelection);
    if (overlays && c.Mode == InteractionMode::Edit && has_primary && mesh_record && meshlets) {
        set(MeshletInstanceFlag::EditOverlay);
        draw(VertexOverlay::EditPoints);
    }
    if (meshlets && !armature && !joint && !r.all_of<ObjectExtrasTag>(mesh_entity) && mesh_record && edges > 0u &&
        (faces == 0u || displays.Wireframe) && !shaded_face_less) {
        set(MeshletInstanceFlag::Wire);
        if (faces == 0u) set(MeshletInstanceFlag::SolidWire);
        if (faces == 0u && !material_face_less) set(MeshletInstanceFlag::MaterialWire);
    }
    if (armature || joint) set(MeshletInstanceFlag::OverlayOnly);
    if (displays.Bones) {
        if (armature) {
            set(MeshletInstanceFlag::Bone);
            if (c.OutlinesArmatureBones(mesh_entity)) set(MeshletInstanceFlag::BoneWire);
        } else if (joint) {
            set(MeshletInstanceFlag::BoneJoint);
            if (c.OutlinesArmatureBones(r.get<const BoneJoint>(mesh_entity).ArmatureObjectEntity)) set(MeshletInstanceFlag::BoneJointWire);
        }
    }
    if (displays.FaceNormals && selected && faces > 0u) set(MeshletInstanceFlag::FaceNormal);
    if (displays.VertexNormals && selected && edges > 0u) {
        set(MeshletInstanceFlag::VertexNormal);
        draw(VertexOverlay::Normals);
    }
    if (overlays && c.Mode == InteractionMode::Excite && sound) {
        if (edges > 0u) set(MeshletInstanceFlag::EdgeOverlay);
        if (meshlets) draw(VertexOverlay::SoundPoints);
    }
    if (mesh_record && faces == 0u && edges == 0u && !has_primary && !shaded_face_less) draw(VertexOverlay::Points);
    d.Flags = flags;
    return {d, vertex_overlays};
}

// Rederives the mesh's vertex overlays, and writes its record display fields when they changed, moving its flag totals to them.
// The tally also follows the edge sharpness an edit overlay draws, which changes apart from the display fields.
void RefreshMeshDisplay(state::Registry &r, const DisplayContext &c, state::Entity mesh_entity, const MeshStore::Record &mb, const PosedLayout *pose) {
    const auto [display, overlays] = ComputeMeshDisplay(c, mesh_entity, mb, pose);
    if (overlays) r.emplace_or_replace<MeshVertexOverlays>(mesh_entity, overlays);
    else r.remove<MeshVertexOverlays>(mesh_entity);
    if (mb.RenderTopologies == 0u) return;
    auto &records = r.Context.get<MeshStore>().Render().MeshRecords;
    const bool changed = std::memcmp(&records.Get({mb.StoreId, 1u})[0].Display, &display, sizeof(MeshDisplay)) != 0;
    if (changed) records.GetMutable({mb.StoreId, 1u})[0].Display = display;
    const auto &tallies = c.Buffers.FlagTallies;
    const bool sharp = (display.Flags & uint32_t(MeshletInstanceFlag::EditOverlay)) != 0u && c.Meshes.GetEdgeSharpnessSummary(mb.StoreId).Any;
    if (changed || sharp != (mb.StoreId < tallies.size() && tallies[mb.StoreId].SharpEdges)) RetallyMesh(r, mesh_entity);
}

void ReleasePoses(GpuBuffers &buffers, state::Entity mesh_entity) {
    for (auto *store : {&buffers.PosedPositions, &buffers.PosedMorphNormalDeltas, &buffers.PosedVertexNormals, &buffers.PosedFaceNormals, &buffers.PosedSectors}) store->Release(mesh_entity);
    buffers.VertexBounds.Release(mesh_entity);
    buffers.PosedMeshletBounds.Release(mesh_entity);
}

void ResizeRange(auto &arena, Range &range, uint32_t count) {
    if (range.Count == count) return;
    arena.Release(range);
    range = arena.Allocate(count);
}

struct LayoutContext {
    bool EditMode, Transforming;
    const selection::PrimaryEditInstanceMap &Primaries;
};

// Allocate bounds and pose storage from the mesh inputs and mark bounds for recomputation.
// Instances with the same deformation share one bounds entry; independently deformed instances use separate entries and poses.
// Keep an empty static entry for meshes without instances.
// Release bounds and pose storage when the mesh record or ModelsBuffer component is absent.
void LayOutMesh(state::Registry &r, const LayoutContext &c, state::Entity mesh_entity) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &render = meshes.Render();
    const auto *models = r.try_get<const ModelsBuffer>(mesh_entity);
    const auto *record = models && HasMesh(r, mesh_entity) ? TryRecordOf(r, mesh_entity) : nullptr;
    const auto *old_pose = r.try_get<const PosedLayout>(mesh_entity);
    const bool was_apart = old_pose && old_pose->Namespaces.PerInstance;
    const auto reset_records = [&] {
        if (!was_apart || !models) return;
        for (auto &instance_record : buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>({models->InstanceRange.Offset, models->InstanceCount})) {
            instance_record = {.Mesh = instance_record.Mesh, .ObjectId = instance_record.ObjectId, .ExcitedVertex = instance_record.ExcitedVertex};
        }
    };
    if (!record) {
        reset_records();
        r.remove<BoundsRun, PosedLayout>(mesh_entity);
        return;
    }
    const auto store_id = record->StoreId;
    // Edit mode uses the rest pose.
    const auto deform = c.EditMode ? MeshDeform{} : DeformOf(r, buffers, *record, *models);
    const bool per_instance = deform.PerInstance();
    const uint32_t count = per_instance ? models->InstanceCount : 1u;
    // Some canonical position edits need posed bounds until their meshlets are refitted.
    // Inset previews refit them directly.
    const auto *edit_work = r.try_get<const MeshEditWork>(mesh_entity);
    const bool posed = models->InstanceCount > 0u &&
        (deform.BoneDeformOffset != InvalidOffset || deform.MorphDeformOffset != InvalidOffset ||
         (c.EditMode && ((c.Transforming && c.Primaries.contains(mesh_entity)) || (edit_work && edit_work->RequiresPose))));

    const bool new_run = !r.all_of<BoundsRun>(mesh_entity);
    auto run = new_run ? BoundsRun{} : r.get<const BoundsRun>(mesh_entity);
    run.BoundsKeys = {};
    run.FaceBlocks = run.VertexBlocks = run.MeshletJobs = 0u;
    std::vector<uint32_t> words;
    std::optional<NormalDeriveEntry> derive_entry;
    PosedLayout pose{
        .Namespaces = {.FirstInstance = models->InstanceRange.Offset, .PerInstance = per_instance},
        .BoneDeformOffset = deform.BoneDeformOffset,
        .MorphDeformOffset = deform.MorphDeformOffset,
        .MorphTargetCount = deform.MorphTargetCount,
        .Armatures = deform.Armatures,
    };
    if (posed) {
        auto &namespaces = pose.Namespaces;
        const auto &arena = meshes.Arenas().Vertices;
        const auto set = record->Vertices;
        const auto vertex_revision = set ? arena.Set(set).Revision : 0u;
        const auto vertex_blocks = PoseElementBlocks(arena, set);
        const auto vertex_bounds = buffers.VertexBounds.Prepare(mesh_entity, store_id, vertex_revision, count, [&] { return std::span{vertex_blocks}; });
        namespaces.VertexBoundsNamespaces.assign(vertex_bounds.Roots.begin(), vertex_bounds.Roots.end());
        for (uint32_t level = 0u; level < VertexBoundsLevels; ++level) {
            run.BoundsKeys[level] = uint32_t(vertex_bounds.Nodes[level].size());
            words.append_range(vertex_bounds.Nodes[level]);
        }
        // Outside edit mode, which binds no deformation, authored morph shading reads rest normals plus weighted authored deltas.
        const bool authored_morph = deform.MorphDeformOffset != InvalidOffset && record->MorphShadingAuthored;
        const auto positions = buffers.PosedPositions.Prepare(mesh_entity, store_id, vertex_revision, count, [&] { return std::span{vertex_blocks}; });
        namespaces.PositionNamespaces.assign(positions.Roots.begin(), positions.Roots.end());
        if (authored_morph) {
            const auto deltas = buffers.PosedMorphNormalDeltas.Prepare(mesh_entity, store_id, vertex_revision, count, [&] { return std::span{vertex_blocks}; });
            namespaces.MorphNormalNamespaces.assign(deltas.Roots.begin(), deltas.Roots.end());
        } else {
            buffers.PosedMorphNormalDeltas.Release(mesh_entity);
        }
        if (!authored_morph) derive_entry = MakeDeriveEntryInputs(meshes, store_id);
        if (derive_entry) {
            const auto vertex_normals = buffers.PosedVertexNormals.Prepare(mesh_entity, store_id, vertex_revision, count, [&] { return std::span{vertex_blocks}; });
            const auto &faces = meshes.Arenas().FaceTriangles;
            const auto face_blocks = PoseElementBlocks(faces, record->FaceData);
            const auto face_normals = buffers.PosedFaceNormals.Prepare(mesh_entity, store_id, faces.Set(record->FaceData).Revision, count, [&] { return std::span{face_blocks}; });
            const auto sectors = buffers.PosedSectors.Prepare(mesh_entity, store_id, meshes.GetDerived(store_id).NormalRevision, count, [&] { return SectorBlocks(meshes, store_id); });
            for (uint32_t i = 0u; i < count; ++i) namespaces.Normals.push_back({vertex_normals.Roots[i], sectors.Roots[i], face_normals.Roots[i]});
            run.FaceBlocks = uint32_t(face_blocks.size());
            run.VertexBlocks = uint32_t(vertex_blocks.size());
            words.append_range(face_blocks);
            words.append_range(vertex_blocks);
        } else {
            buffers.PosedVertexNormals.Release(mesh_entity);
            buffers.PosedFaceNormals.Release(mesh_entity);
            buffers.PosedSectors.Release(mesh_entity);
        }
        const auto meshlet_bounds = buffers.PosedMeshletBounds.Prepare(mesh_entity, store_id, record->MeshletRevision, count, [&] {
            std::vector<uint32_t> blocks;
            const auto meshlets = render.Meshlets.Entries(record->MeshletData);
            meshes.ForEachPrimitive(*record, [&](ElementHandle, const PrimitiveRecord &primitive) {
                if (primitive.LodFinestNode == InvalidOffset) return;
                meshes.ForEachLodMember(*record, render.LodNodes.Get({render.LodNodes.Address(record->NodeData, ElementHandle{primitive.LodFinestNode}).Value(), 1u})[0], [&](uint32_t cluster) { blocks.push_back(Resolve<RenderBlockSize>(meshlets, ElementHandle{cluster}).Value() / MeshElementBlockSize); });
            });
            SortUnique(blocks);
            return blocks; }, record->RenderTopologies);
        namespaces.MeshletBoundsNamespaces.assign(meshlet_bounds.Roots.begin(), meshlet_bounds.Roots.end());
        meshes.ForEachPrimitive(*record, [&](ElementHandle, const PrimitiveRecord &primitive) {
            if (primitive.LodFinestNode == InvalidOffset) return;
            const auto &node = render.LodNodes.Get({render.LodNodes.Address(record->NodeData, ElementHandle{primitive.LodFinestNode}).Value(), 1u})[0];
            if (node.Members == InvalidOffset || !node.MeshletCount) return;
            words.insert(words.end(), {node.Members, node.MeshletCount});
            ++run.MeshletJobs;
        });
    } else if (old_pose) {
        ReleasePoses(buffers, mesh_entity);
    }

    ResizeRange(buffers.BoundsReduceEntries, run.Entries, count);
    ResizeRange(buffers.NormalDeriveEntries, run.DeriveEntries, derive_entry ? count : 0u);
    ResizeRange(buffers.PreludeTiles, run.Tiles, uint32_t(words.size()));
    std::ranges::copy(words, buffers.PreludeTiles.GetMutable(run.Tiles).begin());
    auto entry = MeshBoundsEntry(meshes, store_id, *models);
    if (per_instance) entry.InstanceCount = 1u;
    const auto entries = buffers.BoundsReduceEntries.GetMutable(run.Entries);
    const auto derive_entries = derive_entry ? buffers.NormalDeriveEntries.GetMutable(run.DeriveEntries) : std::span<NormalDeriveEntry>{};
    const auto instance_records = posed && per_instance ? buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>({models->InstanceRange.Offset, models->InstanceCount}) : std::span<InstanceRecord>{};
    for (uint32_t i = 0u; i < count; ++i) {
        auto &instance_entry = entries[i] = entry;
        if (per_instance) instance_entry.FirstInstance += i;
        if (posed) instance_entry.BoundsNamespace = pose.Namespaces.VertexBoundsNamespace(i);
        if (derive_entry) {
            derive_entries[i] = *derive_entry;
            SetPoseNamespaces(derive_entries[i], pose.Namespaces, i);
        }
        if (posed && per_instance) {
            auto &instance_record = instance_records[i];
            const auto armature = deform.ArmatureOffsets.empty() ? InvalidOffset : deform.ArmatureOffsets[i];
            instance_record.ArmatureDeformOffset = armature != InvalidOffset ? armature : deform.ArmatureDeformOffset;
            instance_record.MorphWeightsOffset = deform.WeightOffsets.empty() ? InvalidOffset : deform.WeightOffsets[i];
            SetPoseNamespaces(instance_record, pose.Namespaces, i);
            instance_record.MorphNormalNamespace = pose.Namespaces.MorphNormalNamespace(i);
            instance_record.MeshletBoundsNamespace = pose.Namespaces.MeshletBoundsNamespace(i);
        }
    }
    if (!(posed && per_instance)) reset_records();
    r.emplace_or_replace<BoundsRun>(mesh_entity, run);
    if (new_run || old_pose || posed) r.emplace_or_replace<MeshDisplayDirty>(mesh_entity);
    if (posed) r.emplace_or_replace<PosedLayout>(mesh_entity, std::move(pose));
    else if (old_pose) r.remove<PosedLayout>(mesh_entity);
    r.emplace_or_replace<MeshBoundsDirty>(mesh_entity);
}

void RecordSparseEditPrelude(state::Registry &, const Viewport &, mtl::PassChain &);

// Record one phase's passes into `cb`, which is already begun with viewport and scissor set.
// `ubo_offset` selects the view UBO instance every bind in the phase reads.
// The phase runs the prelude over the source its recording gathered.
// The records hold every displayed viewport's display flags, and this is the wire flag the display's shading draws.
constexpr MeshletInstanceFlag WireFlagOf(const ViewportDisplay &settings) {
    return settings.ViewportShading == ViewportShadingMode::Wireframe ? MeshletInstanceFlag::Wire :
        WorkbenchShading(settings.ViewportShading)                    ? MeshletInstanceFlag::SolidWire :
                                                                        MeshletInstanceFlag::MaterialWire;
}

// The instance flags whose geometry the display draws from original source geometry.
constexpr uint32_t LodPinFlagsOf(const ViewportDisplay &settings) {
    auto flags = uint32_t(MeshletInstanceFlag::LodPinFinest) | uint32_t(WireFlagOf(settings));
    if (settings.ShowOverlays) {
        flags |= uint32_t(MeshletInstanceFlag::EdgeOverlay);
        if (settings.Selection.FaceNormals) flags |= uint32_t(MeshletInstanceFlag::FaceNormal);
        if (settings.Selection.VertexNormals) flags |= uint32_t(MeshletInstanceFlag::VertexNormal);
    }
    return flags;
}

void RecordPhase(state::Registry &r, Viewport &viewport, mtl::PassChain &chain, const PreludeSource &prelude, RenderPhase phase, uint32_t ubo_offset, uint32_t sample_weight = 1u) {
    const profile::CpuScope scope{"RecordRenderCommandBuffer"};
    // Multi-step blur separates scene accumulation from sharp overlay rendering.
    const bool draw_scene = phase != RenderPhase::BlurResolve;
    const bool draw_overlays = !IsBlurAccumulate(phase);

    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<MeshStore>();
    auto &pipelines = GetPipelines(r);
    auto &targets = viewport.Targets;
    const auto &view = viewport.State;
    const auto &settings = view.Display;
    const auto interaction_mode = r.get<const Interaction>(state::Document).Mode;
    const auto edit_mode = r.get<const EditMode>(state::Document).Value;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;
    const bool is_excite_mode = interaction_mode == InteractionMode::Excite;
    const bool is_wireframe_mode = settings.ViewportShading == ViewportShadingMode::Wireframe;
    const bool show_rendered = !WorkbenchShading(settings.ViewportShading);
    const bool show_fill = !is_wireframe_mode;
    const bool xray = XRayActive(settings);
    const float overlay_behind = OverlayBehindOpacity(settings, interaction_mode);
    // Overlays draw through geometry and fade behind it instead of testing against scene depth.
    const bool overlays_through = overlay_behind > 0.f;
    // Overlays need scene depth to occlude or fade unless X-ray shows them at full strength.
    const bool overlay_scene_depth = overlay_behind < 1.f;
    const bool show_overlays = settings.ShowOverlays;
    auto &scene_state = r.Context.get<GpuSceneState>();
    // Forward PBR specializes on the viewport's lighting, the scene's material features, and whether a drawn mesh has line or point meshlets.
    const auto *pbr = show_rendered ? &pipelines.Main.Compiler.Variant(PbrFeaturesOf(view, scene_state.MeshPbrFeatures), buffers.DrawsNonTriangles()) : nullptr;
    const bool real_transmission = RealTransmission(view, scene_state.MeshPbrFeatures);
    const auto &samplers = viewport.Samplers;
    if (std::exchange(scene_state.OverlayJobsDirty, false)) BuildOverlayJobs(r, buffers, scene_state.Displays.BoundingBoxes);
    const bool render_silhouette = show_overlays && settings.Overlays.OutlineSelected && !is_excite_mode &&
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::Silhouette)).Meshlets > 0u;

    if (phase == RenderPhase::Full) RecordSceneCounters(buffers, viewport);

    const bool transmission_active = real_transmission && targets.Transmission;
    // Reuse opaque transmission shading when neither edit tint nor debug output needs another shade.
    const bool composite_transmission = transmission_active && phase == RenderPhase::Full && !is_edit_mode && settings.DebugChannel == DebugChannel::None;
    const bool meshlet_fill = buffers.MeshletInstanceCount > 0;

    // Posed positions and bounds run before culling, over the entries whose inputs changed, and every posed entry in each blur sample.
    const bool prelude_work = !prelude.Empty();
    if (prelude_work) {
        auto *compute = chain.BeginCompute("Prelude", MTL::StageVertex | MTL::StageFragment | MTL::StageDispatch);
        encode::BindScene(compute, slots, buffers, viewport, ubo_offset);
        RecordPrelude(compute, pipelines, GetMeshPipelines(r), buffers, meshes, prelude);
        // The moved geometry can reveal what any viewport's depth pyramid occludes.
        for (auto &other : r.Context.get<Viewports>().All) other.OcclusionStale = true;
    }
    if (is_edit_mode && std::exchange(scene_state.EditPreludePending, false)) RecordSparseEditPrelude(r, viewport, chain);
    if (phase == RenderPhase::Prepare) return;
    MTL::RenderCommandEncoder *encoder = nullptr;
    const auto draw_quad = [&] { encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4)); };

    const auto &main = pipelines.Main;
    const auto main_extent = targets.Resources->SceneColorImage.Extent;
    const bool has_silhouette = render_silhouette && meshlet_fill;
    // Populate visibility for wireframe selection outlines and overlay depth without loading its depth into the scene pass.
    const bool need_visibility = meshlet_fill && (show_fill || has_silhouette || overlay_scene_depth);
    // Each pass selects the flags this viewport's display shows.
    const auto wire_flag = uint32_t(WireFlagOf(settings));
    const bool wire_meshlets = draw_overlays && buffers.FlagWork(wire_flag).Meshlets > 0u;
    const uint64_t bone_meshlets = draw_overlays && show_overlays && settings.Overlays.Bones ?
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::Bone)).Meshlets +
            buffers.FlagWork(uint32_t(MeshletInstanceFlag::BoneJoint)).Meshlets :
        0u;
    const uint64_t normal_meshlets = draw_overlays && show_overlays && settings.Selection.FaceNormals ? buffers.FlagWork(uint32_t(MeshletInstanceFlag::FaceNormal)).Meshlets : 0u;
    const uint64_t element_overlay_meshlets = draw_overlays && show_overlays ? buffers.FlagWork(uint32_t(MeshletInstanceFlag::EdgeOverlay)).Meshlets : 0u;
    const bool vertex_overlays = draw_overlays && !r.view<const MeshVertexOverlays>().empty();
    const bool cull_scene_meshlets =
        (need_visibility || wire_meshlets || bone_meshlets > 0u || normal_meshlets > 0u ||
         element_overlay_meshlets > 0u);
    const bool xray_fill = xray && show_fill && meshlet_fill;
    const bool transparent = xray_fill || (show_rendered && (real_transmission || std::ranges::any_of(buffers.Materials.GetSpan<PBRMaterial>(), [](const auto &m) { return m.AlphaMode == MaterialAlphaMode::Blend; })));
    const auto view_bytes = viewport.SceneViewUBO.Contents().subspan(ubo_offset, sizeof(SceneViewUBO));
    const auto &current_view_proj = reinterpret_cast<const SceneViewUBO *>(view_bytes.data())->ViewProj;
    const bool disocclusion_possible = viewport.OcclusionStale ||
        std::memcmp(&current_view_proj, &targets.Resources->DepthPyramidViewProj, sizeof(mat4)) != 0; // NOLINT(bugprone-suspicious-memory-comparison)
    // Cached occlusion is valid only for the pose that produced it.
    // Reordering opaque surfaces across temporal phases changes the winner of equal-depth raster ties.
    if (cull_scene_meshlets) {
        // X-ray draws occluded surfaces, so it skips occlusion culling.
        const uint32_t pyramid = show_fill && !xray && phase == RenderPhase::Full && targets.Resources->DepthPyramidValid && !disocclusion_possible ?
            samplers.DepthPyramid :
            InvalidSlot;
        RecordMeshletCull(
            chain, slots, pipelines, buffers, viewport,
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
        RecordMeshletVisibilityPass(chain, slots, pipelines, viewport, buffers, real_transmission, ubo_offset, {});
    }
    if (show_fill && phase == RenderPhase::Full && cull_scene_meshlets) {
        targets.Resources->DepthPyramidViewProj = current_view_proj;
        // Only visibility surfaces contribute to occlusion.
        auto *compute = chain.BeginCompute("DepthPyramidFinal", MTL::StageFragment);
        RecordDepthPyramid(
            compute, slots, buffers, viewport, pipelines, targets.Resources->DepthPyramid, samplers.DepthPyramid, 0u,
            samplers.SceneDepth, targets.Resources->VisibilityDepth.Extent, false, ubo_offset
        );
        targets.Resources->DepthPyramidValid = true;
    }
    if (has_silhouette) RecordSilhouetteDepthPass(chain, slots, pipelines, viewport, buffers, ubo_offset);

    // Render background and opaque faces without exposure into TransmissionImage for refracted sampling.
    if (transmission_active && draw_scene) {
        // Refraction samples only the world buffer.
        // The overlay composite adds the display-referred viewport backdrop.
        const std::array colors{mtl::ClearColor(*targets.Transmission->Mip0View)};
        const auto pass = mtl::MakePassDescriptor(colors, mtl::LoadDepth(*targets.Resources->VisibilityDepth));
        encoder = encode::BeginScenePass(chain, pass.get(), "TransmissionPrepass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, viewport, ubo_offset);
        main.PrepassBackground.Bind(encoder);
        draw_quad();
        if (meshlet_fill && show_fill) {
            pbr->BindVisibility(encoder, true);
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
            main_extent, slots, buffers, viewport, ubo_offset
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

        if (show_fill) {
            if (meshlet_fill && draw_scene && show_rendered) {
                if (!composite_transmission) {
                    pbr->BindVisibility(encoder);
                    encoder->setFragmentTexture(*targets.Resources->VisibilityImage, 0u);
                    encode::SetPushConstants(encoder, encode::VisibilityDecodePc(buffers));
                    draw_quad();
                }
                if (transparent) {
                    main.TransparencyInit.Bind(encoder);
                    draw_quad();
                    pbr->BindMeshlets(encoder);
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
        encoder = encode::BeginScenePass(chain, pass.get(), "BlurAccumulate", {{MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, viewport, ubo_offset);
        main.MotionBlurAccumulate.Bind(encoder);
        const MotionBlurAccumulatePushConstants accum_pc{.SceneSamplerSlot = samplers.SceneColor, .Weight = float(sample_weight)};
        encode::SetPushConstants(encoder, accum_pc);
        draw_quad();
        return;
    }

    const bool meshlet_edit_overlay_drawn = show_overlays &&
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::EditOverlay)).Meshlets > 0;
    const bool overlay_jobs = show_overlays && buffers.OverlayJobs.UsedSize > 0u &&
        (settings.Overlays.Extras || settings.Selection.BoundingBoxes || settings.Selection.TetWireframe);
    if (wire_meshlets) {
        buffers.WireCoverageBuffer.SetCount<uint32_t>(main_extent.Width * main_extent.Height);
        { // Four independent 8-bit coverage maxima share one word.
            auto *blit = chain.BeginBlit("WireClear", MTL::StageDispatch);
            blit->fillBuffer(*buffers.WireCoverageBuffer, NS::Range::Make(0, buffers.WireCoverageBuffer.UsedSize), 0);
        }
        // Canonical meshlet edge owners accumulate with atomics, so threadgroups need no ordering.
        auto *wire = chain.BeginCompute("WireRaster", MTL::StageBlit | MTL::StageFragment, MTL::DispatchTypeConcurrent);
        encode::BindCompute(wire, pipelines.WireRaster, slots, buffers, viewport, ubo_offset);
        wire->setTexture(*targets.Resources->VisibilityDepth, 0u);
        WireRasterPushConstants wire_pc{
            .Meshlet = MakeMeshletDrawPc(
                buffers, buffers.SceneCull,
                uint32_t(MeshletRoute::Wire), wire_flag,
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
    if (overlay_jobs) RecordOverlayJobCull(chain, slots, pipelines, buffers, viewport, false, ubo_offset);
    // Cull hidden edit edges and vertex overlays only when their render pass tests against that same scene depth.
    const uint32_t overlay_pyramid = show_fill && !overlays_through && !xray && !has_silhouette &&
            phase == RenderPhase::Full && cull_scene_meshlets ?
        samplers.DepthPyramid :
        InvalidSlot;
    if (meshlet_edit_overlay_drawn) {
        RecordMeshletCull(chain, slots, pipelines, buffers, viewport, {
                                                                .Mode = MeshletRouteMode::Visibility,
                                                                .RequiredInstanceFlags = uint32_t(MeshletInstanceFlag::EditOverlay),
                                                                .RouteMask = 1u << uint32_t(MeshletRoute::EditOverlay),
                                                                .UboOffset = ubo_offset,
                                                                .PyramidSamplerSlot = overlay_pyramid,
                                                                .ExactEditGeometry = true,
                                                                .MinEditOverlayDiameterPixels = MinEditOverlayDiameterPixels,
                                                                .EditOverlayHasSharpEdges = buffers.SharpEditMeshes > 0u,
                                                                .EditOutput = true,
                                                            });
    }

    const bool draw_origins = show_overlays && settings.Overlays.Origins && (!r.view<const Selected>().empty() || !r.view<const Active>().empty());
    // Initialize overlays even when no geometry contributes color.
    const bool overlay_pass_needed = has_silhouette ||
        (show_overlays && settings.Overlays.Grid) ||
        meshlet_edit_overlay_drawn || element_overlay_meshlets > 0u || vertex_overlays || wire_meshlets ||
        overlay_jobs ||
        normal_meshlets > 0u || bone_meshlets > 0u || draw_origins;
    if (overlay_pass_needed) { // Display-referred overlays, depth-tested against scene surfaces and outlines.
        // The outline draw initializes color and scene-plus-outline depth across the target.
        const std::array overlay_colors{
            has_silhouette ? mtl::DiscardColor(*targets.Resources->OverlayColorImage) : mtl::ClearColor(*targets.Resources->OverlayColorImage),
        };
        const auto overlay_depth = has_silhouette ? mtl::DepthAttachment{*targets.Resources->ScratchDepth, MTL::LoadActionDontCare, MTL::StoreActionDontCare} :
            overlays_through                      ? mtl::ClearDepth(*targets.Resources->ScratchDepth) :
                                                    mtl::LoadDepth(*targets.Resources->VisibilityDepth);
        const auto overlay_pass = mtl::MakePassDescriptor(overlay_colors, overlay_depth);
        encoder = encode::BeginScenePass(chain, overlay_pass.get(), "OverlayPass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, viewport, ubo_offset);

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
                                                  r.get<const GizmoInteraction>(state::Document).IsUsing() && interaction_mode == InteractionMode::Object,
                                                  samplers.Silhouette,
                                                  active_object_id,
                                                  overlays_through ? InvalidSlot : samplers.SceneDepth,
                                              });
            draw_quad();
        }

        if (show_overlays && settings.Overlays.Grid) {
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
                uint32_t(route), uint32_t(flag), false, false, sharpness_slot, threads, corner, cull
            );
        };

        if (meshlet_edit_overlay_drawn) {
            // Preserve the sharp-free specialization; the all-smooth Meshlets scene is measurably faster with it.
            const auto &edit_edges = buffers.SharpEditMeshes > 0u ? main.MeshletEditEdges : main.MeshletEditSmoothEdges;
            for (uint32_t corner = 0u; corner < 3u; ++corner) {
                draw_meshlet_overlay(
                    edit_edges, MeshletRoute::EditOverlay, MeshletInstanceFlag::EditOverlay,
                    160u, corner, meshes.Slots().EdgeSharpness, &buffers.EditCull
                );
            }
        }
        if (element_overlay_meshlets > 0u) {
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
            const auto &primaries = r.get<const EditPrimaries>(state::Document).All;
            // Edit points draw on the mesh's primary edit instance, and other overlays on each of its instances.
            // Each kind draws where this viewport's display shows it, and points stay off the face-less meshes its materials shade.
            const auto shows = [&](VertexOverlay kind, state::Entity mesh_entity) {
                switch (kind) {
                    case VertexOverlay::Normals: return show_overlays && settings.Selection.VertexNormals;
                    case VertexOverlay::EditPoints:
                    case VertexOverlay::SoundPoints: return show_overlays;
                    case VertexOverlay::Points: return !show_rendered || RecordOf(r, mesh_entity).PrimitiveMaterials.Count == 0u;
                }
                return false;
            };
            const auto draw_vertex_overlays = [&](const mtl::RenderPipeline &pipeline, VertexOverlay kind) {
                bool bound = false;
                const auto draw = [&](state::Entity mesh_entity, uint32_t slot) {
                    if (!std::exchange(bound, true)) pipeline.Bind(encoder);
                    DrawVertexBlocks(
                        encoder, r, mesh_entity, slot, kind == VertexOverlay::SoundPoints, overlay_pyramid,
                        kind == VertexOverlay::EditPoints ? MinEditOverlayDiameterPixels : 0.0f
                    );
                };
                for (const auto [mesh_entity, overlays] : r.view<const MeshVertexOverlays>().each()) {
                    if (!(overlays.Mask & (1u << uint32_t(kind))) || !shows(kind, mesh_entity)) continue;
                    if (kind == VertexOverlay::EditPoints) {
                        const auto primary = primaries.find(mesh_entity);
                        if (primary != primaries.end()) draw(mesh_entity, r.get<const RenderInstance>(primary->second).BufferIndex);
                    } else if (const auto *models = r.try_get<const ModelsBuffer>(mesh_entity)) {
                        for (uint32_t i = 0u; i < models->InstanceCount; ++i) draw(mesh_entity, models->InstanceRange.Offset + i);
                    }
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
            encoder = encode::BeginScenePass(chain, bone_pass.get(), "BoneXRay", {{MTL::StageDispatch, MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, viewport, ubo_offset);

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
                (!show_overlays || !settings.Overlays.OutlineSelected);
            // Wireframe shading outlines every armature's bones, and other shading the highlighted armatures'.
            if (!hide_bone_outlines) {
                draw_bones(main.BoneWireMesh, is_wireframe_mode ? MeshletInstanceFlag::Bone : MeshletInstanceFlag::BoneWire, 24u);
                draw_bones(main.BoneSphereWireMesh, is_wireframe_mode ? MeshletInstanceFlag::BoneJoint : MeshletInstanceFlag::BoneJointWire, 64u);
            }
        }
        // Origins draw over every other overlay at a fixed size in logical pixels, one instance per instance slot.
        if (draw_origins) {
            const float scale = float(main_extent.Width) / float(std::max(view.Extent.x, 1u));
            main.ObjectOrigins.Bind(encoder);
            encode::SetPushConstants(encoder, ObjectOriginPushConstants{
                                                  .TransformSlot = buffers.Instances.TransformBuffer.Slot,
                                                  .RadiusPx = 4.f * scale,
                                                  .OutlinePx = scale,
                                              });
            encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4), NS::UInteger(buffers.Instances.StateBuffer.UsedSize));
        }
    }

    if (phase == RenderPhase::BlurFast) {
        // Refraction and overlays have finished reading opaque visibility.
        // Include glass in the ID/depth pair for blur now, without changing the depth used to shade behind it.
        if (real_transmission && meshlet_fill) {
            const std::array colors{mtl::LoadColor(*targets.Resources->VisibilityImage)};
            const auto pass = mtl::MakePassDescriptor(colors, mtl::LoadDepth(*targets.Resources->VisibilityDepth));
            auto *visibility = encode::BeginScenePass(chain, pass.get(), "BlurTransmissionVisibility", {{MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, viewport, ubo_offset);
            main.MeshletVisibilityCoverage.Bind(visibility);
            visibility->setCullMode(MTL::CullModeNone);
            DrawMeshletList(visibility, buffers, uint32_t(MeshletRoute::Transmission), 0u, false, true);
        }
        RecordMotionBlurPostFx(r, viewport, chain, ubo_offset);
    }

    { // View-transform the scene, then composite display-referred overlays.
        const std::array colors{mtl::ClearColor(*targets.Resources->FinalColorImage, {0, 0, 0, 1})};
        const auto pass = mtl::MakePassDescriptor(colors);
        encoder = encode::BeginScenePass(chain, pass.get(), "Composite", {{MTL::StageFragment, MTL::StageFragment}}, targets.Resources->FinalColorImage.Extent, slots, buffers, viewport, ubo_offset);
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
        };
        encode::SetPushConstants(encoder, composite_pc);
        draw_quad();
    }
}

} // namespace

void RecordOverlayJobCull(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines,
    GpuBuffers &buffers, const Viewport &viewport, bool extras_only, uint32_t ubo_offset
) {
    const uint32_t job_count = buffers.OverlayJobs.Count<OverlayJob>();
    if (job_count == 0u) return;
    const OverlayJobCullPushConstants pc{
        .JobsSlot = buffers.OverlayJobs.Slot,
        .JobCount = job_count,
        .BlockStateSlot = buffers.OverlayJobBlocks.Slot,
        .VisibleSlot = buffers.VisibleOverlayJobs.Slot,
        .DispatchArgsSlot = buffers.OverlayJobDispatchArgs.Slot,
        .ExtrasOnly = extras_only,
    };
    auto *encoder = chain.BeginCompute("OverlayJobCull", MTL::StageDispatch | MTL::StageFragment);
    encode::BindScene(encoder, slots, buffers, viewport, ubo_offset);
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
                                              .TetPositionSlot = meshes.Slots().TetPosition,
                                              .TetEdgeIndexSlot = meshes.Slots().TetEdgeIndex,
                                          });
    encoder->drawMeshThreadgroups(
        *buffers.OverlayJobDispatchArgs, 0u, MTL::Size(1, 1, 1),
        MTL::Size(uint32_t(OverlayDispatch::LineGroupLines) * 2u, 1, 1)
    );
}

void RecordMeshletVisibilityPass(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines, const Viewport &viewport,
    GpuBuffers &buffers, bool transmission, uint32_t ubo_offset, std::optional<PixelRect> scissor
) {
    const auto &main = pipelines.Main;
    const auto &targets = viewport.Targets;
    const std::array colors{
        mtl::ClearColor(*targets.Resources->VisibilityImage, MTL::ClearColor{double(UINT32_MAX), 0, 0, 0})
    };
    const auto pass = mtl::MakePassDescriptor(colors, mtl::ClearDepth(*targets.Resources->VisibilityDepth));
    auto *encoder = encode::BeginScenePass(
        chain, pass.get(), "MeshletVisibility", {{MTL::StageDispatch, MTL::StageMesh}},
        targets.Resources->VisibilityImage.Extent, slots, buffers, viewport, ubo_offset
    );
    if (scissor) encoder->setScissorRect({scissor->Origin.x, scissor->Origin.y, scissor->Extent.x, scissor->Extent.y});
    // Clear overlay depth even when no meshlets draw.
    if (buffers.MeshletInstanceCount > 0) DrawVisibilityMeshlets(encoder, buffers, main, transmission);
    buffers.VisibilityGeneration = buffers.MeshletVisibleGeneration;
}

void RecordMeshletCull(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines,
    GpuBuffers &buffers, const Viewport &viewport, MeshletCullConfig config
) {
    auto &output = config.EditOutput ? buffers.EditCull : buffers.SceneCull;
    if (!config.EditOutput) ++buffers.MeshletVisibleGeneration;
    auto *encoder = chain.BeginCompute("MeshletCull", MTL::StageMesh | MTL::StageFragment);
    const bool transmission = config.Mode == MeshletRouteMode::Transmission;
    // Bound this cull with maintained totals for the requested flag.
    const auto primary = config.RequiredInstanceFlags == 0u ?
        GpuBuffers::MeshletFlagWork{buffers.LodNodeCount, buffers.MeshletInstanceCount} :
        buffers.FlagWork(config.RequiredInstanceFlags);
    buffers.EnsureMeshletVisibilityCapacity(
        output,
        primary.Meshlets * (1u + transmission), primary.Nodes, primary.Meshlets
    );
    const auto pc = [&] {
        auto pc = MakeMeshletCullSlotsPc(buffers, viewport, output);
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
    encode::BindScene(encoder, slots, buffers, viewport, config.UboOffset);
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
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines, const Viewport &viewport,
    GpuBuffers &buffers, uint32_t ubo_offset
) {
    // Compacts the outlined surfaces the visibility image cannot resolve.
    const auto &resources = *viewport.Targets.Resources;
    const auto &samplers = viewport.Samplers;
    auto decode_pc = encode::VisibilityDecodePc(buffers);
    auto *compute = chain.BeginCompute("SilhouetteCull", MTL::StageFragment);
    const auto &occluders = resources.OutlineOccluderPyramid;
    encode::BindCompute(compute, pipelines.OutlineOccluderSeed, slots, buffers, viewport, ubo_offset);
    compute->setTexture(*resources.VisibilityImage, 0u);
    compute->setTexture(*resources.VisibilityDepth, 1u);
    compute->setTexture(*occluders.Mips[0].View, 2u);
    encode::SetPushConstants(compute, decode_pc);
    const auto blocks = occluders.Mips[0].Extent;
    compute->dispatchThreadgroups(MTL::Size((blocks.Width + 15u) / 16u, (blocks.Height + 15u) / 16u, 1u), ThreadgroupSize::Tile16);
    compute->memoryBarrier(MTL::BarrierScopeTextures);
    RecordDepthPyramid(compute, slots, buffers, viewport, pipelines, occluders, samplers.OutlineOccluderPyramid, 1u, InvalidSlot, {}, true, ubo_offset);
    compute->memoryBarrier(MTL::BarrierScopeTextures);

    auto &output = buffers.SilhouetteCull;
    output.Visible.SetCount<VisibleMeshlet>(buffers.SceneCull.Visible.Count<VisibleMeshlet>());
    output.ChunkCount = buffers.SceneCull.ChunkCount;
    output.DispatchArgs.SetCount<MeshDispatchArgs>(GpuBuffers::MeshletRouteCount * output.ChunkCount);
    auto cull_pc = MakeMeshletCullSlotsPc(buffers, viewport, output);
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
        resources.SilhouetteImage.Extent, slots, buffers, viewport, ubo_offset
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
        DrawMeshletList(encoder, buffers, uint32_t(route), 0u, false, false, InvalidSlot, 160u, 0u, &buffers.SilhouetteCull);
    };
    for (const auto [route, cull] : VisibilityRoutes) draw_route(route, cull);
    draw_route(MeshletRoute::Blend, MTL::CullModeNone);
    draw_route(MeshletRoute::Transmission, MTL::CullModeNone);
}

void DrawMeshlets(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, uint32_t route,
    uint32_t required_instance_flags, uint32_t mesh_threads, uint32_t edit_edge_corner
) {
    DrawMeshletList(
        encoder, buffers,
        route, required_instance_flags, false, false, InvalidSlot,
        mesh_threads, edit_edge_corner
    );
}

void DrawVertexBlocks(
    MTL::RenderCommandEncoder *encoder, const state::Registry &r, state::Entity mesh_entity, uint32_t instance, bool sound_points,
    uint32_t pyramid_slot, float min_diameter_pixels
) {
    const auto &buffers = r.Context.get<const GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto store_id = *DrawnStoreId(r, mesh_entity);
    VertexBlockPushConstants pc{
        .Instance = instance,
        .MembershipSlot = meshes.Arenas().Vertices.Blocks.Buffer.Slot,
        .SoundPoints = sound_points ? 1u : 0u,
        .PyramidSamplerSlot = pyramid_slot,
        .MinDiameterPixels = min_diameter_pixels,
    };
    if (const auto *posed = r.try_get<const PosedLayout>(mesh_entity)) {
        const auto &pose = posed->Namespaces;
        pc.BoundsNamespace = pose.VertexBoundsNamespace(pose.PerInstance ? instance - pose.FirstInstance : 0u);
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

void RecordRenderCommandBuffer(state::Registry &r, Viewport &viewport, MTL::CommandBuffer *command_buffer, RenderPhase phase) {
    const mtl::AutoreleaseScope native_scope;
    profile::BeginRecording();
    mtl::PassChain chain{command_buffer, profile::RecordingTimer()};
    // A blur sample's pose differs from the live one, so the fast blur recomputes every posed entry.
    RecordPhase(r, viewport, chain, GatherPrelude(r, phase == RenderPhase::BlurFast), phase, 0);
    profile::EndRecording();
}

void RecordBlurStepsCommandBuffer(state::Registry &r, Viewport &viewport, MTL::CommandBuffer *command_buffer, std::span<const uint32_t> sample_weights) {
    const mtl::AutoreleaseScope native_scope;
    profile::BeginRecording();
    mtl::PassChain chain{command_buffer, profile::RecordingTimer()};
    // Every step poses its own sample and the resolve the live pose, so each recomputes every posed entry over the lists gathered once.
    const auto prelude = GatherPrelude(r, true);
    for (uint32_t i = 0; i < sample_weights.size(); ++i) {
        RecordPhase(r, viewport, chain, prelude, i == 0 ? RenderPhase::BlurAccumulateFirst : RenderPhase::BlurAccumulate, GpuBuffers::SceneViewUboOffset(i + 1), sample_weights[i]);
    }
    RecordPhase(r, viewport, chain, prelude, RenderPhase::BlurResolve, 0);
    profile::EndRecording();
}

void ReserveFrameScratch(state::Registry &r, std::span<Viewport *const> viewports) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &scene_state = r.Context.get<const GpuSceneState>();
    uint64_t coverage = 0u;
    bool transmission = false;
    for (const auto *viewport : viewports) {
        const auto extent = viewport->Targets.Resources->SceneColorImage.Extent;
        coverage = std::max<uint64_t>(coverage, uint64_t(extent.Width) * extent.Height);
        transmission |= RealTransmission(viewport->State, scene_state.MeshPbrFeatures);
    }
    buffers.WireCoverageBuffer.Reserve(coverage * sizeof(uint32_t));
    const GpuBuffers::MeshletFlagWork all{buffers.LodNodeCount, buffers.MeshletInstanceCount};
    const auto visible = all.Meshlets * (1u + transmission);
    buffers.EnsureMeshletVisibilityCapacity(buffers.SceneCull, visible, all.Nodes, all.Meshlets);
    buffers.EnsureMeshletVisibilityCapacity(buffers.SilhouetteCull, visible, all.Nodes, all.Meshlets);
    const auto &edit = buffers.FlagWork(uint32_t(MeshletInstanceFlag::EditOverlay));
    buffers.EnsureMeshletVisibilityCapacity(buffers.EditCull, edit.Meshlets, edit.Nodes, edit.Meshlets);
}

SceneViewUBO SceneViewOf(state::Registry &r, const Viewport &viewport, const RenderView &render_view) {
    const auto &buffers = r.Context.get<const GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &render = meshes.Render();
    const auto &environments = r.Context.get<const EnvironmentStore>();
    const auto &targets = viewport.Targets;
    const auto &settings = viewport.State.Display;
    const auto interaction_mode = r.get<const Interaction>(state::Document).Mode;
    const bool is_pbr_mode = !WorkbenchShading(settings.ViewportShading);
    const auto &active_lighting = ActivePbrLighting(viewport.State);
    const bool use_scene_lights = is_pbr_mode && active_lighting.UseSceneLights;
    const bool use_scene_world = is_pbr_mode && active_lighting.UseSceneWorld;
    const auto ibl = use_scene_world ? environments.SceneWorld.Ibl : StudioIbl(r, viewport.State.Environment.Name);
    const auto *image_light = r.try_get<const ImageLight>(state::Document);
    const float env_intensity = use_scene_world && image_light ? image_light->Intensity : active_lighting.EnvIntensity;
    const mat3 env_rotation = [&]() -> mat3 {
        if (use_scene_world) return image_light ? ToMat3(image_light->Rotation) : mat3{1.f};
        const float radians = active_lighting.EnvRotationDegrees * (Pi / 180.f);
        const float s = std::sin(radians), c = std::cos(radians);
        return {c, 0, -s, 0, 1, 0, s, 0, c};
    }();
    const auto *pending = r.try_get<const PendingTransform>(state::Document);
    const auto &mesh_slots = meshes.Slots();
    const auto look_through = viewport.State.LookThroughEntity;
    SceneViewUBO view{
        .LightCount = buffers.Lights.Count<LightRecord>(),
        .LightSlot = buffers.Lights.Slot,
        .UseSceneLightsRender = use_scene_lights ? 1u : 0u,
        .EnvIntensity = env_intensity,
        .Exposure = std::exp2(active_lighting.ExposureEV),
        .EnvRotation = env_rotation,
        .BackgroundBlur = active_lighting.BackgroundBlur,
        .WorldOpacity = is_pbr_mode ? active_lighting.WorldOpacity : 0.f,
        .Ibl = ibl,
        .InteractionMode = interaction_mode,
        .EditElement = r.get<const EditMode>(state::Document).Value,
        .IsTransforming = pending ? 1u : 0u,
        .PendingPivot = pending ? pending->Pivot : vec3{},
        .PendingTranslation = pending ? pending->Delta.P : vec3{},
        .PendingRotation = pending ? pending->Delta.R : quat{1, 0, 0, 0},
        .PendingScale = pending ? pending->Delta.S : vec3{1},
        .LodErrorPixels = settings.LodErrorPixels,
        .LodPinFlags = LodPinFlagsOf(settings),
        .EdgeSharpnessSlot = mesh_slots.EdgeSharpness,
        .FaceSharpnessSlot = mesh_slots.FaceSharpness,
        .BaseVertexNormalSlot = mesh_slots.BaseVertexNormal,
        .BaseFaceNormalSlot = mesh_slots.BaseFaceNormal,
        .FaceTriangleStartSlot = mesh_slots.FaceTriangleStart,
        .ArmatureDeformSlot = buffers.ArmatureDeformBuffer.Buffer.Slot,
        .MorphWeightsSlot = buffers.MorphWeightBuffer.Buffer.Slot,
        .PosedPositionSlot = buffers.PosedPositions.Values.Buffer.Slot,
        .PosedPositionNodesSlot = buffers.PosedPositions.Nodes.Buffer.Slot,
        .PosedVertexNormalSlot = buffers.PosedVertexNormals.Values.Buffer.Slot,
        .PosedVertexNormalNodesSlot = buffers.PosedVertexNormals.Nodes.Buffer.Slot,
        .PosedSectorNodesSlot = buffers.PosedSectors.Nodes.Buffer.Slot,
        .PosedSectorValuesSlot = buffers.PosedSectors.Values.Buffer.Slot,
        .PosedFaceNormalSlot = buffers.PosedFaceNormals.Values.Buffer.Slot,
        .PosedFaceNormalNodesSlot = buffers.PosedFaceNormals.Nodes.Buffer.Slot,
        .PosedMorphNormalDeltaSlot = buffers.PosedMorphNormalDeltas.Values.Buffer.Slot,
        .PosedMorphNormalNodesSlot = buffers.PosedMorphNormalDeltas.Nodes.Buffer.Slot,
        .InstanceBoundsSlot = buffers.Instances.BoundsBuffer.Slot,
        .MaterialSlot = buffers.Materials.Slot,
        .PrimitiveMaterialSlot = mesh_slots.PrimitiveMaterial,
        .MeshRecordSlot = render.MeshRecords.Buffer.Slot,
        .InstanceRecordSlot = buffers.Instances.RecordBuffer.Slot,
        .InstanceStateSlot = buffers.Instances.StateBuffer.Slot,
        .BoneXRay = settings.ViewportShading == ViewportShadingMode::Wireframe ? 1u : 0u,
        .XRayAlpha = XRayActive(settings) && settings.ViewportShading == ViewportShadingMode::Solid ? XRayOpacity(settings) : 1.f,
        .OverlayBehindOpacity = OverlayBehindOpacity(settings, interaction_mode),
        .SceneDepthSamplerSlot = viewport.Samplers.SceneDepth,
        .ShowOverlays = settings.ShowOverlays ? 1u : 0u,
        .ShowExtras = settings.Overlays.Extras ? 1u : 0u,
        .ShowBoundingBoxes = settings.Selection.BoundingBoxes ? 1u : 0u,
        .ShowTetWireframe = settings.Selection.TetWireframe ? 1u : 0u,
        .TransmissionFramebufferSamplerSlot = viewport.Samplers.Transmission,
        .TransmissionFramebufferMipCount = targets.Transmission ? targets.Transmission->Image.MipLevels : 1u,
        .UseRealTransmission = (is_pbr_mode && active_lighting.RealTransmission && targets.Transmission) ? 1u : 0u,
        .DebugChannel = is_pbr_mode ? settings.DebugChannel : DebugChannel::None,
        .LookThroughObjectId = look_through != state::Null ? ObjectId(look_through) : 0u,
    };
    render_view.ApplyTo(view);
    return view;
}

void WriteViewUniforms(state::Registry &r, Viewport &viewport, const RenderView &view) {
    viewport.SceneViewUBO.Update(BytesOf(SceneViewOf(r, viewport, view)));
    viewport.WorkspaceLightsUBO.Update(BytesOf(viewport.State.Lights));
    viewport.FrameView = view;
}

void UpdateAuthoredMorphShading(state::Registry &r, mtl::ComputeChain &chain, std::span<const state::Entity> mesh_entities) {
    auto &meshes = r.Context.get<MeshStore>();
    auto &buffers = r.Context.get<GpuBuffers>();
    // Each position-only target gets a derive entry reading its full-weight pose directly.
    std::vector<NormalDeriveEntry> entries;
    std::vector<uint32_t> entry_ids;
    std::vector<state::Entity> job_entities;
    PoseAttributeStore<vec3>::Temporary vertex_normals{buffers.PosedVertexNormals};
    PoseAttributeStore<vec3>::Temporary face_normals{buffers.PosedFaceNormals};
    PoseAttributeStore<vec3>::Temporary sectors{buffers.PosedSectors};
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
        const auto normal_blocks = SectorBlocks(meshes, store_id);
        const auto vertex_blocks = PoseElementBlocks(meshes.Arenas().Vertices, record.Vertices);
        const auto face_blocks = PoseElementBlocks(meshes.Arenas().FaceTriangles, record.FaceData);
        const auto &morph = meshes.Arenas().Morph;
        for (uint32_t t = 0; t < target_count; ++t) {
            // Targets without position deltas use the rest pose and require no normal pinning.
            bool has_position_delta = false;
            meshes.Arenas().Vertices.ForEach(record.Vertices, [&](ElementHandle handle, ElementAddress) {
                if (!has_position_delta && morph.Get(record.Morph, handle.Value, t, target_count).PositionDelta != vec3{0}) has_position_delta = true;
            });
            if (!has_position_delta) continue;
            auto entry = *entry_inputs;
            entry.Morph = morph.Ref(record.Morph);
            entry.MorphTargetIndex = t;
            entry.MorphTargetCount = target_count;
            entry.VertexNormalNamespace = vertex_normals.Add(vertex_blocks);
            entry.SectorNamespace = sectors.Add(normal_blocks);
            entry.FaceNormalNamespace = face_normals.Add(face_blocks);
            entries.emplace_back(entry);
            entry_ids.emplace_back(store_id);
            job_entities.emplace_back(entity);
        }
    }
    if (entries.empty()) return;

    // Derivation reads each full-weight morph directly from canonical base vertices and target deltas.
    // No temporary position buffer or CPU geometry copy.
    EncodeDeriveNormals(r, chain, entry_ids, entries, MakeNormalDerivePc(buffers, meshes, true));
    chain.Submit();

    for (size_t i = 0; i < job_entities.size();) {
        const auto entity = job_entities[i];
        std::vector<CornerNormalSources> poses;
        for (; i < job_entities.size() && job_entities[i] == entity; ++i) {
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

void FinalizeAuthoredShading(state::Registry &r, mtl::ComputeChain &chain) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto authored = r.view<const AuthoredCornerNormals, const MeshHandle>() | std::ranges::to<std::vector>();
    for (const auto entity : authored) {
        EncodeAuthoredCornerNormals(meshes, GetMesh(r, entity), r.get<const AuthoredCornerNormals>(entity).Corners);
    }
    UpdateAuthoredMorphShading(r, chain, authored);
    for (const auto entity : authored) r.remove<AuthoredCornerNormals>(entity);
}

namespace {
void DispatchWork(MTL::ComputeCommandEncoder *encoder, const GpuBuffers &buffers, ElementWork work) {
    encoder->dispatchThreadgroups(*buffers.GeometryWork.Buffer, WorkArgsOffset(work), ThreadgroupSize::Linear256);
}

void FinalizeWork(MTL::ComputeCommandEncoder *encoder, const Pipelines &pipelines, std::initializer_list<ElementWork> work) {
    encoder->setComputePipelineState(pipelines.FinalizeElementWork.State());
    encoder->setBytes(work.begin(), work.size() * sizeof(ElementWork), BufferIndex_PushConstants);
    encoder->dispatchThreadgroups(MTL::Size(work.size(), 1, 1), ThreadgroupSize::Linear256);
}

MeshEditWork &PrepareMeshEditWork(state::Registry &r, state::Entity entity) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto mesh = GetDrawnMesh(r, entity);
    const auto id = mesh.GetStoreId();
    if (const auto *work = r.try_get<const MeshEditWork>(entity)) {
        if (work->StoreId == id) return r.edit<MeshEditWork>(entity);
        ReleaseMeshEditWork(r, entity);
    }
    // Edit work pins the mesh's instances to finest geometry.
    r.emplace_or_replace<MeshDisplayDirty>(entity);
    auto work_allocation = buffers.GeometryWork.BeginAllocation();
    MeshEditWork w{.StoreId = id};
    w.WorkBudget = buffers.GeometryWork.Allocate(3);
    w.Candidates = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Vertices = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Faces = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Normals = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Meshlets = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.BoundsTiles = AllocateElementWork(buffers.GeometryWork, 1u << 24u);
    for (uint32_t level = 0u; level < w.BoundsLevels.size(); ++level) {
        w.BoundsLevels[level] = AllocateElementWork(buffers.GeometryWork, 1u << (16u - level * 8u));
    }
    auto &result = r.emplace<MeshEditWork>(entity, std::move(w));
    work_allocation.Commit();
    return result;
}
} // namespace

void ReleaseMeshEditWork(state::Registry &r, state::Entity entity) {
    const auto *work = r.try_get<const MeshEditWork>(entity);
    if (!work) return;
    // Work that required a pose posed the mesh's bounds.
    if (work->RequiresPose) r.emplace_or_replace<MeshLayoutDirty>(entity);
    r.remove<MeshEditWork>(entity);
    r.emplace_or_replace<MeshDisplayDirty>(entity);
}

void ReleaseMeshEditWorkRanges(state::Registry &r, state::Entity entity) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &w = r.get<const MeshEditWork>(entity);
    for (const auto range : {w.Candidates, w.Vertices, w.Faces, w.Normals, w.Meshlets, w.BoundsTiles}) buffers.GeometryWork.Release(WorkStorageRange(range));
    for (const auto level : w.BoundsLevels) buffers.GeometryWork.Release(WorkStorageRange(level));
    buffers.GeometryWork.Release(w.WorkBudget);
}

namespace {
using GeometryEditJobs = std::vector<std::pair<state::Entity, CommitPosedGeometryPushConstants>>;

// Prepares each job's reusable topological footprint before any geometry writes, one phase for every job per submit.
// Stale candidates seed from the vertex selection in the first submit.
// Counts bound sparse allocation between the phases.
// Repeated parameter changes reuse the same tables.
void PrepareGeometryFootprints(state::Registry &r, mtl::ComputeChain &chain, GeometryEditJobs &jobs) {
    struct Footprint {
        MeshEditWork *Work;
        CommitPosedGeometryPushConstants *Pc;
        uint64_t VertexBlocks{}, MeshletBlocks{};
    };
    std::vector<Footprint> footprints;
    for (auto &[entity, pc] : jobs) {
        if (auto &work = r.edit<MeshEditWork>(entity); !work.FootprintReady) footprints.push_back({&work, &pc});
    }
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
        pc.MeshletAddresses = work.Meshlets;
        pc.BoundsTiles = work.BoundsTiles;
        pc.BudgetOffset = work.WorkBudget.Offset;
    };
    std::vector<ElementWorkSeedJob> seeds;
    std::vector<MeshEditWork *> seeded;
    for (const auto &footprint : footprints) {
        auto &work = *footprint.Work;
        if (work.CandidateReady) continue;
        std::vector<uint32_t> blocks;
        meshes.GetSelectedElements(work.StoreId, Element::Vertex).ForEachBlock([&](uint32_t block, uint32_t) { blocks.push_back(block); });
        ReserveElementWork(buffers.GeometryWork, work.Candidates, uint64_t(WorkBlockCount(buffers.GeometryWork, work.Candidates)) + 2u * blocks.size());
        seeds.push_back(PrepareBlockMembershipWork(chain.Scratch, meshes.Arenas().Vertices, meshes.Get(work.StoreId).Vertices, blocks, work.Candidates, meshes.GetSelectionSlot(Element::Vertex)));
        seeded.push_back(&work);
    }
    EncodeElementMembershipWork(r, chain, seeds);
    if (!seeded.empty()) {
        chain.Encode([&](MTL::ComputeCommandEncoder *encoder) {
            encode::BindBindless(encoder, slots);
            for (const auto *work : seeded) FinalizeWork(encoder, pipelines, {work->Candidates});
            encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        });
    }
    for (const auto &footprint : footprints) {
        const auto &work = *footprint.Work;
        for (const auto item : {work.Faces, work.Normals, work.Meshlets, work.BoundsTiles}) ClearElementWork(buffers.GeometryWork, item);
        std::ranges::fill(buffers.GeometryWork.GetMutable(work.WorkBudget), 0u);
    }
    const auto submit = [&](std::initializer_list<uint32_t> phases) {
        for (const auto &footprint : footprints) update(footprint);
        chain.Encode([&](MTL::ComputeCommandEncoder *encoder) {
            encode::BindBindless(encoder, slots);
            for (const auto phase : phases) {
                encoder->setComputePipelineState(pipelines.CommitPosedGeometry.State());
                for (const auto &footprint : footprints) {
                    auto &pc = *footprint.Pc;
                    pc.Phase = phase;
                    encode::SetPushConstants(encoder, pc);
                    DispatchWork(encoder, buffers, phase < 2u ? pc.Candidates : pc.Faces);
                }
                encoder->memoryBarrier(MTL::BarrierScopeBuffers);
                if (phase != 1u && phase != 3u) continue;
                for (const auto &footprint : footprints) {
                    const auto &pc = *footprint.Pc;
                    if (phase == 1u) FinalizeWork(encoder, pipelines, {pc.Faces, pc.MeshletAddresses, pc.BoundsTiles});
                    else FinalizeWork(encoder, pipelines, {pc.Normals, pc.MeshletAddresses});
                }
                encoder->memoryBarrier(MTL::BarrierScopeBuffers);
            }
        });
        chain.Submit();
        for (const auto &footprint : footprints) {
            const auto &work = *footprint.Work;
            for (const auto item : {work.Faces, work.Normals, work.Meshlets, work.BoundsTiles}) CheckElementWork(buffers.GeometryWork, item);
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
        // A table's distinct blocks are at most the blocks its domain spans: the mesh's face and vertex sets, and its owner's meshlet set.
        const auto &record = meshes.Get(work.StoreId);
        const uint64_t face_blocks = record.FaceData ? a.FaceTriangles.Set(record.FaceData).BlockCount : 0u;
        footprint.VertexBlocks = a.Vertices.Set(record.Vertices).BlockCount;
        if (const auto *owner = meshes.TryGet(work.StoreId); owner && owner->MeshletData) footprint.MeshletBlocks = meshes.Render().Meshlets.Set(owner->MeshletData).BlockCount;
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

CommitPosedGeometryPushConstants PrepareGeometryEdit(state::Registry &r, state::Entity entity, state::Entity primary, const PendingTransform *pending, const PosedNamespaces *pose = nullptr, std::span<const Range> changed = {}) {
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &meshes = r.Context.get<MeshStore>();
    auto &w = PrepareMeshEditWork(r, entity);
    const auto mesh = GetDrawnMesh(r, entity);
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
        w.CandidateReady = w.FootprintReady = false;
        if (!w.PreviewActive) ClearElementWork(buffers.GeometryWork, w.Candidates);
    }
    ClearElementWork(buffers.GeometryWork, w.Vertices);
    auto entry = MakeDeriveEntryInputs(meshes, id).value_or(NormalDeriveEntry{.VertexCount = mesh.VertexCount(), .Connectivity = meshes.GetConnectivityRef(id)});
    if (pose) {
        for (const auto &level : w.BoundsLevels) ClearElementWork(buffers.GeometryWork, level);
        SetPoseNamespaces(entry, *pose, 0);
        w.PreviewActive = pending != nullptr;
    }
    return {
        .VerticesSlot = meshes.Slots().Vertices,
        .PositionSlot = buffers.PosedPositions.Values.Buffer.Slot,
        .PositionNodesSlot = buffers.PosedPositions.Nodes.Buffer.Slot,
        .SelectionSlot = meshes.GetSelectionSlot(Element::Vertex),
        .Candidates = w.Candidates,
        .ChangedVertices = w.Vertices,
        .Faces = w.Faces,
        .Normals = w.Normals,
        .MeshletAddresses = w.Meshlets,
        .BoundsTiles = w.BoundsTiles,
        .Entry = entry,
        .Primary = pending ? *WorldTransformOf(r, primary) : Transform{},
        .Delta = pending ? pending->Delta : Transform{},
        .Pivot = pending ? pending->Pivot : vec3{},
        .FaceTriangleStartSlot = meshes.Slots().FaceTriangleStart,
        .ElementMeshlets = {meshes.Render().ElementMeshlets[0].Ref(meshes.Get(id).ElementMeshlets[0]), meshes.Render().ElementMeshlets[1].Ref(meshes.Get(id).ElementMeshlets[1]), meshes.Render().ElementMeshlets[2].Ref(meshes.Get(id).ElementMeshlets[2])},
        .MeshletSets = meshes.Render().Meshlets.Slots(),
        .MeshletSet = meshes.Get(id).MeshletData.Index,
        .ApplyTransform = pending ? 1u : 0u,
        .Mode = !changed.empty() ? GeometryEditMode::Refresh : pose ? GeometryEditMode::Preview :
                                                                      GeometryEditMode::Commit,
    };
}

// Binds the scene on the encoder once, so later passes on it set only their pipelines.
void RecordGeometryEditBatch(state::Registry &r, MTL::ComputeCommandEncoder *encoder, GeometryEditJobs &commits, bool posed) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &pipelines = GetPipelines(r);
    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    const auto entries = buffers.GeometryNormalEntries.SetCount<NormalDeriveEntry>(commits.size());
    for (uint32_t i = 0; i < commits.size(); ++i) entries[i] = commits[i].second.Entry;
    r.Context.get<const mtl::Context>().CommitResidency();
    encode::BindBindless(encoder, slots);
    encoder->setComputePipelineState(pipelines.CommitPosedGeometry.State());
    for (auto &[entity, pc] : commits) {
        // Posed outputs have no history.
        // A commit writes the canonical base normals.
        if (!posed && pc.Entry.FaceCount) {
            auto entry = pc.Entry;
            entry.VerticesWork = pc.Normals;
            entry.FacesWork = pc.Faces;
            entry.VertexWorkCount = buffers.GeometryWork.Get({pc.Normals.Storage.Offset + 5u, 1u})[0];
            entry.FaceWorkCount = buffers.GeometryWork.Get({pc.Faces.Storage.Offset + 5u, 1u})[0];
            CaptureNormalWrites(r, r.get<const MeshEditWork>(entity).StoreId, entry, buffers.GeometryWork, buffers.GeometryWork);
        }
        pc.Phase = 4u;
        encode::SetPushConstants(encoder, pc);
        DispatchWork(encoder, buffers, pc.Candidates);
    }
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    for (const auto &[_, pc] : commits) FinalizeWork(encoder, pipelines, {pc.ChangedVertices});
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    auto derive = MakeNormalDerivePc(buffers, meshes, posed);
    derive.EntriesSlot = buffers.GeometryNormalEntries.Slot;
    encoder->setComputePipelineState(GetMeshPipelines(r)[MeshPass::VertexNormalDerive].State());
    for (uint32_t phase = 0; phase < 2; ++phase) {
        derive.Phase = phase;
        for (uint32_t i = 0; i < commits.size(); ++i) {
            if (commits[i].second.Entry.FaceCount == 0) continue;
            derive.EntryIndex = i;
            derive.Work = phase == 0 ? commits[i].second.Faces : commits[i].second.Normals;
            encode::SetPushConstants(encoder, derive);
            DispatchWork(encoder, buffers, derive.Work);
        }
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    }
}
// Faces keep their identities and triangle ranges.
// Canonical position edits discard affected authored tangents; those render keys and changed diagonals share one repair.
bool RetessellateGeometry(state::Registry &r, mtl::ComputeChain &chain, GeometryEditJobs &geometry, bool preview) {
    const auto &work = r.Context.get<const GpuBuffers>().GeometryWork;
    std::vector<MeshRetessellation> inputs;
    for (const auto &[entity, edit] : geometry) {
        inputs.push_back({
            .StoreId = *DrawnStoreId(r, entity),
            .Work = &work,
            .Faces = edit.Faces,
            .Parameters = {.ChangedVertices = edit.ChangedVertices, .SelectionSlot = edit.SelectionSlot, .ApplyTransform = uint32_t(preview && edit.ApplyTransform), .Primary = edit.Primary, .Delta = edit.Delta, .Pivot = edit.Pivot},
            .Preview = preview,
        });
    }
    const auto changed = RetessellateMeshes(r, chain, inputs);
    std::vector<std::pair<state::Entity, FaceTriangles>> repairs;
    for (uint32_t i = 0u; i < changed.size(); ++i) {
        if (changed[i].Count) repairs.emplace_back(geometry[i].first, changed[i]);
    }
    RepairFaceRender(r, chain, repairs);
    if (!repairs.empty()) {
        // New triangle owners invalidate the cached render footprint.
        for (const auto &[entity, _] : repairs) r.edit<MeshEditWork>(entity).FootprintReady = false;
        PrepareGeometryFootprints(r, chain, geometry);
    }
    return !repairs.empty();
}

} // namespace

void RefreshEditedPositions(state::Registry &r, mtl::ComputeChain &chain, std::span<const MeshVertexChanges> changes, bool retessellate) {
    if (changes.empty()) return;
    const profile::CpuScope scope{"RefreshEditedPositions"};
    GeometryEditJobs jobs;
    for (const auto &[entity, ranges] : changes) jobs.emplace_back(entity, PrepareGeometryEdit(r, entity, state::Null, nullptr, nullptr, ranges));
    PrepareGeometryFootprints(r, chain, jobs);
    chain.Encode([&](MTL::ComputeCommandEncoder *encoder) { RecordGeometryEditBatch(r, encoder, jobs, false); });
    if (retessellate) RetessellateGeometry(r, chain, jobs, false);
    auto &scene = r.Context.get<GpuSceneState>();
    scene.EditPreludePending = true;
    for (const auto &[entity, ranges] : changes) {
        auto &work = r.edit<MeshEditWork>(entity);
        work.Modified = work.PreviewActive = work.RequiresPose = true;
        r.emplace_or_replace<MeshLayoutDirty>(entity);
    }
}

bool RefreshPreviewTessellation(state::Registry &r) {
    const auto *pending = r.try_get<const PendingTransform>(state::Document);
    GeometryEditJobs jobs;
    for (const auto &[entity, primary] : r.get<const EditPrimaries>(state::Document).Transformable) {
        // A procedure mesh's transform cooks into its drawn record instead of previewing over it.
        const auto *previewed = r.all_of<procedure::GeometryProcedure>(entity) ? nullptr : pending;
        const auto *old = r.try_get<const MeshEditWork>(entity);
        if (!previewed && (!old || !old->TessellationPreview)) continue;
        jobs.emplace_back(entity, PrepareGeometryEdit(r, entity, primary, previewed));
    }
    if (jobs.empty()) return false;
    mtl::ComputeChain chain{r.Context.get<MeshStore>().BufferContext()};
    PrepareGeometryFootprints(r, chain, jobs);
    const bool changed = RetessellateGeometry(r, chain, jobs, true);
    for (const auto &[entity, _] : jobs) r.edit<MeshEditWork>(entity).TessellationPreview = pending != nullptr;
    chain.Submit();
    return changed;
}

std::vector<state::Entity> CommitPosedGeometry(state::Registry &r, mtl::ComputeChain &chain, std::span<const state::Entity> mesh_entities) {
    const profile::CpuScope scope{"CommitGeometry"};
    const auto *pending = r.try_get<const PendingTransform>(state::Document);
    if (!pending) return {};
    const auto &primaries = r.get<const EditPrimaries>(state::Document).Transformable;
    GeometryEditJobs commits;
    for (const auto entity : mesh_entities) {
        if (const auto primary = primaries.find(entity); primary != primaries.end()) {
            commits.emplace_back(entity, PrepareGeometryEdit(r, entity, primary->second, pending));
        }
    }
    if (commits.empty()) return {};
    PrepareGeometryFootprints(r, chain, commits);
    auto &meshes = r.Context.get<MeshStore>();
    for (const auto &[entity, pc] : commits) meshes.CaptureVertexEdit(*DrawnStoreId(r, entity));
    chain.Encode([&](MTL::ComputeCommandEncoder *encoder) { RecordGeometryEditBatch(r, encoder, commits, false); });
    RetessellateGeometry(r, chain, commits, false);
    for (const auto &[entity, _] : commits) r.edit<MeshEditWork>(entity).TessellationPreview = false;
    std::vector<state::Entity> entities;
    for (const auto &[entity, pc] : commits) entities.push_back(entity);
    return entities;
}

std::vector<state::Entity> PublishEditedPositions(state::Registry &r, mtl::ComputeChain &chain, std::span<const state::Entity> entities, PositionPublication publication) {
    const bool preview = publication == PositionPublication::Preview;
    auto &meshes = r.Context.get<MeshStore>();
    const auto &buffers = r.Context.get<GpuBuffers>();
    std::vector<state::Entity> changed;
    std::vector<MeshStore::SelectionUpdate> aggregates;
    std::vector<MeshletBoundsRefitJob> refits;
    std::vector<std::vector<uint32_t>> dirty_meshlets;
    std::vector<state::Entity> dirty_entities;
    for (const auto entity : entities) {
        auto &w = r.edit<MeshEditWork>(entity);
        const auto vertices = preview ? w.Candidates : w.Vertices;
        if (!ElementWorkEmpty(buffers.GeometryWork, vertices)) {
            changed.push_back(entity);
            const auto id = *DrawnStoreId(r, entity);
            if (meshes.Get(id).SelectionSummary.Count) {
                auto &update = aggregates.emplace_back(MeshStore::SelectionUpdate{.StoreId = id});
                update.Blocks[0] = meshes.ArenaBlocks(id, MeshStore::ElementDomain::Vertex, buffers.GeometryWork, vertices);
                update.Blocks[2] = meshes.ArenaBlocks(id, MeshStore::ElementDomain::Face, buffers.GeometryWork, w.Faces);
            }
            w.Modified = w.PreviewActive = true;
            w.RequiresPose = !preview;
            r.emplace_or_replace<MeshLayoutDirty>(entity);
            const auto &owner = RecordOf(r, entity);
            refits.push_back({&owner, &buffers.GeometryWork, w.Meshlets});
            if (!preview && meshes.ClusterGroupCount(owner) > 0u && !ElementWorkEmpty(buffers.GeometryWork, w.Meshlets)) {
                auto &meshlets = dirty_meshlets.emplace_back();
                ForEachWorkElement(buffers.GeometryWork, w.Meshlets, [&](uint32_t id) { meshlets.push_back(id); });
                dirty_entities.push_back(entity);
            }
        }
    }
    {
        const profile::CpuScope scope{preview ? "InsetRefitPass" : "PositionRefitPass"};
        RefitCanonicalMeshletBounds(r, chain, refits);
    }
    if (!dirty_entities.empty()) {
        auto &dirty = meshes.Render().PositionDirty;
        for (uint32_t i = 0u; i < dirty_entities.size(); ++i) {
            for (const auto address : dirty_meshlets[i]) MaskSet(dirty, ElementAddress::Stored(address));
            // Position-dirty meshlets pin their mesh's instances to finest geometry.
            r.emplace_or_replace<MeshletPositionsDirty>(dirty_entities[i]);
            r.emplace_or_replace<MeshDisplayDirty>(dirty_entities[i]);
        }
    }
    meshes.UpdateSelection(r, chain, aggregates);
    chain.AfterSubmit([&r, changed] { RefreshElementSelectionSummaries(r, changed); });
    return changed;
}

namespace {
void RecordSparseEditPrelude(state::Registry &r, const Viewport &viewport, mtl::PassChain &chain) {
    const auto &buffers = r.Context.get<GpuBuffers>();
    const auto &pipelines = GetPipelines(r);
    const auto *pending = r.try_get<const PendingTransform>(state::Document);
    const auto &primaries = r.get<const EditPrimaries>(state::Document).Transformable;
    GeometryEditJobs jobs;
    for (const auto [entity, pose] : r.view<const PosedLayout>().each()) {
        const auto primary = primaries.find(entity);
        const bool preview = pending && primary != primaries.end() && !r.all_of<procedure::GeometryProcedure>(entity);
        const auto *old = r.try_get<const MeshEditWork>(entity);
        if (!preview && (!old || !old->PreviewActive)) continue;
        jobs.emplace_back(entity, PrepareGeometryEdit(r, entity, preview ? primary->second : state::Null, preview ? pending : nullptr, &pose.Namespaces));
    }
    if (jobs.empty()) return;
    {
        // The footprints submit before the frame records, since their counts size the edit work.
        mtl::ComputeChain footprints{r.Context.get<const MeshStore>().BufferContext()};
        PrepareGeometryFootprints(r, footprints, jobs);
        footprints.Submit();
    }
    auto *encoder = chain.BeginCompute("EditGeometry", MTL::StageDispatch);
    // The posed bounds passes read the viewport's view uniforms.
    encode::BindScene(encoder, r.Context.get<const mtl::BindlessSet>(), buffers, viewport);
    RecordGeometryEditBatch(r, encoder, jobs, true);
    // Each edited mesh's entry and work, with every bounds level dispatched for all meshes before the next level's barrier.
    struct EditBounds {
        const MeshEditWork &Work;
        uint32_t Entry, Instance;
    };
    std::vector<EditBounds> edits;
    edits.reserve(jobs.size());
    for (const auto &[entity, job] : jobs) {
        edits.push_back({r.get<const MeshEditWork>(entity), r.get<const BoundsRun>(entity).Entries.Offset, r.get<const PosedLayout>(entity).Namespaces.FirstInstance});
    }
    for (const auto &edit : edits) RecordPosedMeshletBounds(encoder, pipelines, buffers, {.MeshletAddresses = edit.Work.Meshlets, .Instance = edit.Instance});
    for (const auto &edit : edits) {
        RecordBoundsPass(encoder, pipelines.BoundsReduce, buffers, {.Work = edit.Work.BoundsTiles, .NextWork = edit.Work.BoundsLevels[0], .EntryIndex = edit.Entry});
    }
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    for (uint32_t level = 1u; level < VertexBoundsLevels; ++level) {
        for (const auto &edit : edits) FinalizeWork(encoder, pipelines, {edit.Work.BoundsLevels[level - 1u]});
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        for (const auto &edit : edits) {
            const auto work = edit.Work.BoundsLevels[level - 1u];
            RecordBoundsPass(encoder, pipelines.BoundsCombine, buffers, {.Work = work, .NextWork = level + 1u < VertexBoundsLevels ? edit.Work.BoundsLevels[level] : ElementWork{}, .EntryIndex = edit.Entry, .Level = level});
        }
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    }
}
} // namespace

void LayOutMeshes(state::Registry &r) {
    const profile::CpuScope scope{"LayOutMeshes"};
    const bool edit_mode = r.get<const Interaction>(state::Document).Mode == InteractionMode::Edit;
    const LayoutContext context{edit_mode, edit_mode && r.all_of<PendingTransform>(state::Document), r.get<const EditPrimaries>(state::Document).All};
    for (const auto mesh_entity : r.view<const MeshLayoutDirty>() | std::ranges::to<std::vector>()) LayOutMesh(r, context, mesh_entity);
    r.clear<MeshLayoutDirty>();
}

void ReleaseMeshLayout(state::Registry &r, state::Entity mesh_entity) {
    const auto &run = r.get<const BoundsRun>(mesh_entity);
    auto &pending = r.Context.emplace<PendingObjectRemovals>();
    pending.BoundsEntryRanges.push_back(run.Entries);
    pending.DeriveEntryRanges.push_back(run.DeriveEntries);
    pending.PreludeTileRanges.push_back(run.Tiles);
    // Only a posed run's mesh contains pose namespaces.
    if (run.Posed()) ReleasePoses(r.Context.get<GpuBuffers>(), mesh_entity);
}

void RefreshMeshDisplays(state::Registry &r) {
    const DisplayContext context{r};
    for (const auto mesh_entity : r.view<const MeshDisplayDirty>() | std::ranges::to<std::vector>()) {
        if (const auto *mb = TryRecordOf(r, mesh_entity)) RefreshMeshDisplay(r, context, mesh_entity, *mb, r.try_get<const PosedLayout>(mesh_entity));
    }
}
