#include "viewport/ViewportRenderGpu.h"
#include "gpu/FastGiPushConstants.h"
#include "graph/Graph.h"

#include "Camera.h"
#include "Profile.h"
#include "SortUnique.h"
#include "Variant.h"
#include "animation/Keying.h"
#include "animation/MorphWeights.h"
#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "gizmo/GizmoInteraction.h"
#include "gpu/AreaShadowBounds.h"
#include "gpu/BoundsReducePushConstants.h"
#include "gpu/DepthPyramidReducePushConstants.h"
#include "gpu/GeometryEditJob.h"
#include "gpu/LightCulling.h"
#include "gpu/LightRecord.h"
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
#include "gpu/RayLightingPushConstants.h"
#include "gpu/RayReconstructPushConstants.h"
#include "gpu/RayTracingRecord.h"
#include "gpu/SilhouetteEdgeColorPushConstants.h"
#include "gpu/SurfacePattern.h"
#include "gpu/VertexBlockPushConstants.h"
#include "gpu/ViewportCompositePushConstants.h"
#include "gpu/VisibilityId.h"
#include "gpu/WireRasterPushConstants.h"
#include "gpu/WireResolvePushConstants.h"
#include "mesh/ElementMembershipWork.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshStores.h"
#include "mesh/NormalDeriveGpu.h"
#include "metal/MetalCpp.h"
#include "metal/Mipmaps.h"
#include "metal/RenderTarget.h"
#include "numeric/MatrixMath.h"
#include "object/PendingSync.h"
#include "physics/PhysicsTypes.h"
#include "project/store/Hash.h"
#include "render/ClusterLod.h"
#include "render/Encoding.h"
#include "render/GpuSceneState.h"
#include "render/MaterialComponents.h"
#include "render/MaterialPatterns.h"
#include "render/MeshTopologyRepair.h"
#include "render/MeshletBoundsRefit.h"
#include "render/MeshletBuildGpu.h"
#include "render/ObjectGeometry.h"
#include "render/Pipelines.h"
#include "render/RayFilm.h"
#include "render/SceneUpdates.h"
#include "render/Textures.h"
#include "scene/CameraLens.h"
#include "scene/Entity.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "timeline/Timeline.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportComponents.h"
#include "viewport/ViewportEvents.h"
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

    const auto &light = r.get<const SceneLight>(object);
    const uint32_t range_lines = light.Range > 0.f ? RangeSegments : 0;
    if (light.Type == LightType::Point) {
        return {ExtrasLineKind::LightPoint, vec4{light.Range, 0, 0, 0}, range_lines + HaloLines};
    }
    if (light.Type == LightType::Directional) {
        return {ExtrasLineKind::LightDirectional, vec4{light.Range, 0, 0, 0}, 8 * 2 + HaloLines};
    }
    if (IsAreaLight(light.Type)) {
        const bool rectangle = light.Type == LightType::Rectangle;
        return {rectangle ? ExtrasLineKind::LightRectangle : ExtrasLineKind::LightEllipse, vec4{light.Range, light.Size.x * .5f, light.Size.y * .5f, 0}, range_lines + (rectangle ? 4u : SpotSegments) + 1u + HaloLines};
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

void BuildOverlayJobs(const state::Registry &r, GpuBuffers &buffers, bool bounding_boxes) {
    constexpr uint32_t LinesPerJob{uint32_t(OverlayDispatch::LineGroupLines)};
    const auto object_ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>();
    // The jobs follow the instances' drawing order, so coplanar lines resolve the same way across loads.
    const auto for_each_job = [&](auto &&emit) {
        for (const auto slot : buffers.GpuInstanceSlots.GetSpan<uint32_t>()) {
            const auto object = r.EntityAt(ObjectIndex(object_ids[slot]));
            const auto buffer_entity = r.get<const RenderInstance>(object).Entity;
            if (r.all_of<ObjectExtrasTag>(buffer_entity)) {
                const auto gizmo = ExtrasGizmoParams(r, object, r.get<const ObjectKind>(object).Value);
                emit(OverlayJob{.Kind = OverlayJobKind::Extras, .InstanceIndex = slot, .ExtrasKind = gizmo.Kind, .LocalOffset = vec3{0}, .Params = gizmo.Params}, gizmo.LineCount);
            }
            if (const auto *shape = r.try_get<const ColliderShape>(object)) {
                const auto wire = ColliderWireParams(shape->Shape);
                emit(OverlayJob{.Kind = OverlayJobKind::Extras, .InstanceIndex = slot, .ExtrasKind = wire.Kind, .LocalOffset = shape->LocalOffset, .Params = wire.Params}, wire.LineCount);
            }
            if (const auto *geometry = bounding_boxes ? r.try_get<const ObjectGeometry>(object) : nullptr; geometry && DrawsMesh(r, geometry->Entity)) {
                emit(OverlayJob{.Kind = OverlayJobKind::Bounds, .InstanceIndex = slot}, 12u);
            }
            if (const auto *tets = r.try_get<const TetBuffers>(buffer_entity); tets && tets->EdgeIndices.Count != 0u) {
                emit(OverlayJob{.Kind = OverlayJobKind::TetWire, .InstanceIndex = slot, .SourceOffset = tets->Positions.Offset, .IndexOffset = tets->EdgeIndices.Offset}, tets->EdgeIndices.Count / 2u);
            }
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
    if (!profile::Enabled) return;
    if (viewport.Targets.Screen) {
        const auto extent = viewport.Targets.Screen->SceneColorImage.Extent;
        profile::RecordCounter("RenderWidth", extent.Width);
        profile::RecordCounter("RenderHeight", extent.Height);
    }
    if (viewport.Targets.RayTracing) {
        const auto &rt = *viewport.Targets.RayTracing;
        const auto pixels = *reinterpret_cast<const uint32_t *>(rt.Work.Contents().data());
        profile::RecordCounter("RayActivePixels", pixels);
        profile::RecordCounter("RayQueueBytes", rt.RayQueue.Length());
        const auto extent = rt.Radiance.Extent;
        profile::RecordCounter("RayQueueUsedBytes", uint64_t(pixels) * (rt.RayQueue.Length() / (uint64_t(extent.Width) * extent.Height)));
        profile::RecordCounter("RayWorkBytes", rt.Work.Length());
    }
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
    std::vector<state::Entity> Armatures;
    // By instance in slot order, InvalidOffset for an instance without that deformation, and empty when no instance has it.
    std::vector<uint32_t> ArmatureOffsets, WeightOffsets;
    bool PerInstance() const { return !ArmatureOffsets.empty() || !WeightOffsets.empty(); }
};

MeshDeform DeformOf(const state::Registry &r, const GpuBuffers &buffers, const MeshStore::Record &record, const ModelsBuffer &models) {
    MeshDeform deform;
    const auto object_ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>({models.InstanceRange.Offset, models.InstanceCount});
    for (uint32_t i = 0u; i < object_ids.size(); ++i) {
        const auto instance = r.EntityAt(ObjectIndex(object_ids[i]));
        if (const auto *modifier = record.SkinBlocksReady ? r.try_get<const ArmatureModifier>(instance) : nullptr) {
            const auto *pose_state = r.try_get<const ArmaturePoseState>(modifier->ArmatureObjectEntity);
            if (pose_state && modifier->SkinSlot < pose_state->GpuDeformRanges.size()) {
                const auto offset = pose_state->GpuDeformRanges[modifier->SkinSlot].Offset;
                if (deform.BoneDeformOffset == InvalidOffset) {
                    deform.BoneDeformOffset = 0u;
                    deform.ArmatureDeformOffset = offset;
                }
                if (!std::ranges::contains(deform.Armatures, modifier->ArmatureObjectEntity)) deform.Armatures.push_back(modifier->ArmatureObjectEntity);
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
    std::array<uint32_t, uint32_t(PreludeSlot::Count)> Groups;

    bool Has(PreludeSlot slot) const { return Groups[uint32_t(slot)] > 0u; }
    bool Empty() const {
        return std::ranges::all_of(Groups, [](uint32_t groups) { return groups == 0u; });
    }
    void Dispatch(mtl::ComputeEncoder encoder, PreludeSlot slot, MTL::Size group_size) const {
        if (Has(slot)) encoder->dispatchThreadgroups(MTL::Size(Groups[uint32_t(slot)], 1, 1), group_size);
    }
};

template<typename T>
std::vector<uint32_t> PoseElementBlocks(const ElementArena<T> &arena, ElementSetRef set) {
    std::vector<uint32_t> blocks;
    arena.ForEachBlock(set, [&](uint32_t block, const auto &) { blocks.push_back(block); });
    return blocks;
}

std::vector<uint32_t> SectorBlocks(const MeshStore &meshes, uint32_t store_id) {
    const auto &record = meshes.Get(store_id);
    return record.NormalSectors.Count ? PoseElementBlocks(meshes.Arenas().FaceCorners, record.FaceCorners) : std::vector<uint32_t>{};
}

void SetPoseNamespaces(auto &out, const PosedNamespaces &pose, uint32_t i) {
    const auto normals = pose.NormalsAt(i).value_or(PosedNamespaces::NormalNamespaces{});
    out.PositionNamespace = pose.PositionNamespace(i);
    out.VertexNormalNamespace = normals.Vertex;
    out.SectorNamespace = normals.Sector;
    out.FaceNormalNamespace = normals.Face;
}

// The prelude recorders require scene bindings on the encoder.

void RecordPosePrepass(mtl::ComputeEncoder encoder, const Pipelines &pipelines, const GpuBuffers &buffers, const PreludeSource &source) {
    pipelines.PosePrepass.Bind(encoder);
    const BoundsReducePushConstants pc{
        .BoundsEntrySlot = buffers.BoundsReduceEntries.Buffer.Slot,
        .TileMapSlot = buffers.SparseBoundsTiles.Slot,
        .ValuesSlot = buffers.VertexBounds.Values.Buffer.Slot,
        .NodesSlot = buffers.VertexBounds.Nodes.Buffer.Slot,
        .MembersSlot = buffers.VertexBounds.Members.Slot,
        .FirstTile = source.FirstTiles[0],
    };
    encode::SetPushConstants(encoder, pc);
    SetThreadgroupMemoryLength(encoder, ThreadgroupMemory::BoundsFoldVector, 0);
    SetThreadgroupMemoryLength(encoder, ThreadgroupMemory::BoundsFoldVector, 1);
    source.Dispatch(encoder, PreludeSlot::PosePrepass, ThreadgroupSize::Linear256);
}

void RecordNormalDerive(mtl::ComputeEncoder encoder, const mtl::ComputePipeline &pipeline, NormalDerivePushConstants pc, const PreludeSource &source, PreludeSlot slot) {
    pipeline.Bind(encoder);
    encode::SetPushConstants(encoder, pc);
    source.Dispatch(encoder, slot, ThreadgroupSize::Linear256);
}

NormalDerivePushConstants MakeNormalDerivePc(const GpuBuffers &buffers, const MeshStore &meshes) {
    return {
        .EntriesSlot = buffers.NormalDeriveEntries.Buffer.Slot,
        .EdgeSharpnessSlot = meshes.Slots().EdgeSharpness,
        .FaceSharpnessSlot = meshes.Slots().FaceSharpness,
        .TileMapSlot = buffers.SparseDeriveTiles.Slot,
        .PositionSlot = buffers.PosedPositions.Values.Buffer.Slot,
        .PositionNodesSlot = buffers.PosedPositions.Nodes.Buffer.Slot,
        .VertexNormalSlot = buffers.PosedVertexNormals.Values.Buffer.Slot,
        .VertexNormalNodesSlot = buffers.PosedVertexNormals.Nodes.Buffer.Slot,
        .PosedSectorNodesSlot = buffers.PosedSectors.Nodes.Buffer.Slot,
        .PosedSectorValuesSlot = buffers.PosedSectors.Values.Buffer.Slot,
        .FaceNormalSlot = buffers.PosedFaceNormals.Values.Buffer.Slot,
        .FaceNormalNodesSlot = buffers.PosedFaceNormals.Nodes.Buffer.Slot,
        .BaseFaceNormalSlot = meshes.Slots().BaseFaceNormal,
    };
}

void RecordBoundsPass(mtl::ComputeEncoder encoder, const mtl::ComputePipeline &pipeline, const GpuBuffers &buffers, uint32_t level, const PreludeSource &source, PreludeSlot slot) {
    pipeline.Bind(encoder);
    encode::SetPushConstants(encoder, BoundsReducePushConstants{
                                          .BoundsEntrySlot = buffers.BoundsReduceEntries.Buffer.Slot,
                                          .BoundsSlot = buffers.Instances.BoundsBuffer.Slot,
                                          .TileMapSlot = buffers.SparseBoundsTiles.Slot,
                                          .ValuesSlot = buffers.VertexBounds.Values.Buffer.Slot,
                                          .NodesSlot = buffers.VertexBounds.Nodes.Buffer.Slot,
                                          .MembersSlot = buffers.VertexBounds.Members.Slot,
                                          .FirstTile = source.FirstTiles[level],
                                          .Level = level,
                                      });
    SetThreadgroupMemoryLength(encoder, ThreadgroupMemory::BoundsFoldVector, 0);
    SetThreadgroupMemoryLength(encoder, ThreadgroupMemory::BoundsFoldVector, 1);
    source.Dispatch(encoder, slot, ThreadgroupSize::Linear256);
}

void RecordPosedMeshletBounds(mtl::ComputeEncoder encoder, const Pipelines &pipelines, const GpuBuffers &buffers, const PreludeSource &source) {
    pipelines.PosedMeshletBounds.Bind(encoder);
    encode::SetPushConstants(encoder, PosedMeshletBoundsPushConstants{
                                          .JobsSlot = buffers.SparsePosedMeshletBoundsJobs.Slot,
                                          .JobCount = source.MeshletJobCount,
                                          .MeshletSlot = buffers.Render->Meshlets.Buffer.Slot,
                                          .MeshletVertexSlot = buffers.Render->MeshletVertexCorners.Buffer.Slot,
                                          .PosedMeshletBoundsSlot = buffers.PosedMeshletBounds.Values.Buffer.Slot,
                                          .PosedMeshletBoundsNodesSlot = buffers.PosedMeshletBounds.Nodes.Buffer.Slot,
                                          .Members = buffers.Render->LodMembers.Slots(),
                                          .Meshlets = buffers.Render->Meshlets.Slots(),
                                          .MemberSlot = buffers.Render->LodMembers.Buffer.Slot,
                                      });
    source.Dispatch(encoder, PreludeSlot::PosedMeshletBounds, ThreadgroupSize::Linear32);
}

// Bindless dependencies require explicit barriers between pose and bounds levels.
void RecordPrelude(mtl::ComputeEncoder compute, const Pipelines &pipelines, const MeshPipelines &mesh_pipelines, const GpuBuffers &buffers, const MeshStore &meshes, const PreludeSource &source) {
    if (source.Has(PreludeSlot::PosePrepass)) {
        RecordPosePrepass(compute, pipelines, buffers, source);
        mtl::ComputeBarrier(compute);
    }
    if (source.Has(PreludeSlot::DeriveFaces)) {
        auto derive = MakeNormalDerivePc(buffers, meshes);
        const auto &normal_pipeline = mesh_pipelines[MeshPass::VertexNormalDerive];
        RecordNormalDerive(compute, normal_pipeline, derive, source, PreludeSlot::DeriveFaces);
        mtl::ComputeBarrier(compute);
        derive.Phase = 1u;
        derive.FirstTile = source.DeriveGatherFirstTile;
        RecordNormalDerive(compute, normal_pipeline, derive, source, PreludeSlot::DeriveGather);
        mtl::ComputeBarrier(compute);
    }
    if (source.Has(PreludeSlot::PosedMeshletBounds)) RecordPosedMeshletBounds(compute, pipelines, buffers, source);
    if (source.Has(PreludeSlot::BoundsLevel3)) {
        RecordBoundsPass(compute, pipelines.BoundsCombine, buffers, 1u, source, PreludeSlot::BoundsLevel1);
        mtl::ComputeBarrier(compute);
        RecordBoundsPass(compute, pipelines.BoundsCombine, buffers, 2u, source, PreludeSlot::BoundsLevel2);
        mtl::ComputeBarrier(compute);
        RecordBoundsPass(compute, pipelines.BoundsCombine, buffers, 3u, source, PreludeSlot::BoundsLevel3);
    }
}

PreludeSource GatherPrelude(state::Registry &r, bool every_posed) {
    auto &buffers = r.Resources.get<GpuBuffers>();
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

void RecordMeshletCompaction(mtl::ComputeEncoder encoder, const Pipelines &pipelines, const GpuBuffers &buffers) {
    const auto dispatch_meshlets = [&](const mtl::ComputePipeline &pipeline) {
        pipeline.Bind(encoder);
        const uint32_t prefix_bytes = GpuBuffers::MeshletRouteCount * (GpuBuffers::MeshletCullBlockSize / 32u + 1u) * sizeof(uint32_t);
        SetThreadgroupMemoryLength(encoder, AlignedThreadgroupBytes(prefix_bytes), 0);
        encoder->dispatchThreadgroups((*buffers.MeshletWorkDispatchArgs)->gpuAddress(), MTL::Size(GpuBuffers::MeshletCullBlockSize, 1, 1));
    };
    dispatch_meshlets(pipelines.MeshletCullBlockCount);
    mtl::ComputeBarrier(encoder);
    pipelines.MeshletCullPrefix.Bind(encoder);
    SetThreadgroupMemoryLength(encoder, AlignedThreadgroupBytes(GpuBuffers::MeshletRouteCount * sizeof(uint32_t)), 0);
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256);
    mtl::ComputeBarrier(encoder);
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
    mtl::RenderEncoder encoder, const GpuBuffers &buffers, uint32_t route, uint32_t required_instance_flags,
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
        encoder->drawMeshThreadgroups((*output.DispatchArgs)->gpuAddress() + args_offset, MTL::Size(1, 1, 1), MTL::Size(mesh_threads, 1, 1));
    }
}

constexpr std::pair<MeshletRoute, MTL::CullMode> VisibilityRoutes[]{
    {MeshletRoute::OpaqueCullBack, MTL::CullModeBack},
    {MeshletRoute::OpaqueCullFront, MTL::CullModeFront},
    {MeshletRoute::OpaqueDoubleSided, MTL::CullModeNone},
    {MeshletRoute::Coverage, MTL::CullModeNone},
};

void DrawVisibilityMeshlets(
    mtl::RenderEncoder encoder, const GpuBuffers &buffers, const MainPipeline &main,
    bool transmission
) {
    for (const auto [route, cull] : VisibilityRoutes) {
        (route == MeshletRoute::Coverage ? main.MeshletVisibilityCoverage : main.MeshletVisibilityOpaque).Bind(encoder);
        SetCullMode(encoder, cull);
        DrawMeshletList(encoder, buffers, uint32_t(route), 0u, transmission, true);
    }
}

void RecordDepthPyramid(
    mtl::ComputeEncoder encoder, const mtl::BindlessSet &slots, const GpuBuffers &buffers, const Viewport &viewport, const Pipelines &pipelines,
    const RenderTargets::ScreenT::Pyramid &pyramid, uint32_t pyramid_sampler, uint32_t first_level,
    uint32_t source_sampler, mtl::Extent2D source_extent, bool nearest, uint32_t ubo_offset
) {
    encode::BindCompute(encoder, pipelines.DepthPyramidReduce, slots, buffers, viewport, ubo_offset);
    const auto &mips = pyramid.Mips;
    for (uint32_t base = first_level; base < uint32_t(mips.size()); base += 6) {
        // Bindless mip dependencies need an explicit barrier.
        if (base > first_level) mtl::ComputeBarrier(encoder);
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
        SetThreadgroupMemoryLength(encoder, ThreadgroupMemory::DepthPyramidTile, 0);
        encoder->dispatchThreadgroups(
            MTL::Size((mips[base].Extent.Width + 31) / 32, (mips[base].Extent.Height + 31) / 32, 1),
            ThreadgroupSize::Tile16
        );
    }
}

// The visibility and depth pair is intact here, and no shading variant needs velocity outputs.
void RecordMotionBlurPostFx(state::Registry &r, const Viewport &viewport, mtl::PassChain &chain, uint32_t ubo_offset) {
    const auto &slots = r.Resources.get<const mtl::BindlessSet>();
    const auto &buffers = r.Resources.get<const GpuBuffers>();
    const auto &main = GetPipelines(r).Main;
    const auto &targets = viewport.Targets;
    const auto &blur = *targets.MotionBlur;
    const auto &samplers = viewport.Samplers;
    const auto extent = targets.Screen->SceneColorImage.Extent;
    const auto tiles = blur.TileImage.Extent;
    const auto view_ubo = viewport.SceneViewUBO.Contents();
    const auto &view = *reinterpret_cast<const SceneViewUBO *>(view_ubo.data() + ubo_offset);
    const auto compute = chain.BeginCompute("BlurTiles", MTL::StageFragment);
    encode::BindCompute(compute, main.MotionBlurTilesFlatten, slots, buffers, viewport, ubo_offset);
    const auto &previous = *reinterpret_cast<const SceneViewUBO *>(view_ubo.data() + GpuBuffers::SceneViewUboOffset(1));
    const auto &next = *reinterpret_cast<const SceneViewUBO *>(view_ubo.data() + GpuBuffers::SceneViewUboOffset(2));
    const uint32_t camera_motion = view.ViewProj != previous.ViewProj || view.ViewProj != next.ViewProj;
    encode::SetPushConstants(compute, MotionBlurTilesFlattenPushConstants{encode::VisibilityDecodePc(buffers), Inverse(view.ViewProj), camera_motion});
    compute.Cmd->SetBuffer(*viewport.SceneViewUBO, GpuBuffers::SceneViewUboOffset(1), 5);
    compute.Cmd->SetBuffer(*viewport.SceneViewUBO, GpuBuffers::SceneViewUboOffset(2), 6);
    compute.Cmd->SetBuffer(*blur.TileIndirection, 0, 7);
    compute.Cmd->SetTexture(*targets.Screen->VisibilityImage, 0);
    compute.Cmd->SetTexture(*targets.Screen->VisibilityDepth, 1);
    compute.Cmd->SetTexture(*blur.VelocityImage, 2);
    compute.Cmd->SetTexture(*blur.TileImage, 3);
    SetThreadgroupMemoryLength(compute, 16, 0);
    SetThreadgroupMemoryLength(compute, 16, 1);
    compute->dispatchThreadgroups(MTL::Size(tiles.Width, tiles.Height, 1), ThreadgroupSize::Tile8);
    mtl::ComputeBarrier(compute);
    encode::BindCompute(compute, main.MotionBlurTilesDilate, slots, buffers, viewport, ubo_offset);
    compute.Cmd->SetBuffer(*blur.TileIndirection, 0, 5);
    compute.Cmd->SetTexture(*blur.TileImage, 0);
    compute->dispatchThreadgroups(MTL::Size((tiles.Width + 7) / 8, (tiles.Height + 7) / 8, 1), ThreadgroupSize::Tile8);

    const std::array colors{mtl::DiscardColor(*blur.OutputImage)};
    const auto pass = mtl::MakePassDescriptor(colors);
    const auto render = encode::BeginScenePass(chain, pass.get(), "BlurGather", {{MTL::StageDispatch | MTL::StageFragment | MTL::StageBlit, MTL::StageFragment}}, extent, slots, buffers, viewport, ubo_offset);
    main.MotionBlurGather.Bind(render);
    render.Cmd->SetBuffer(*blur.TileIndirection, 0, 5);
    render.Cmd->SetTexture(*blur.TileImage, 0);
    const auto inverse_projection = Inverse(viewport.FrameView.Camera.Projection(float(extent.Width) / float(extent.Height)));
    const float noise_phase = r.get<const PlaybackFrame>(state::Document).Value * std::numbers::phi_v<float>;
    encode::SetPushConstants(render, MotionBlurGatherPushConstants{samplers.SceneDepth, samplers.Velocity, samplers.SceneColor, noise_phase - std::floor(noise_phase), {inverse_projection[2].z, inverse_projection[3].z, inverse_projection[2].w, inverse_projection[3].w}});
    render->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
}

struct DisplayContext {
    DisplayContext(const state::Registry &r)
        : R(r), Buffers(r.Resources.get<const GpuBuffers>()), Meshes(r.Resources.get<const MeshStore>()),
          Displays(r.Resources.get<const GpuSceneState>().Displays), Mode(r.get<const Interaction>(state::Document).Mode),
          Primaries(r.get<const EditPrimaries>(state::Document).All), SelectedMeshes(r.get<const SelectionFlags>(state::Document).Meshes) {
        if (Mode == InteractionMode::Excite) {
            for (const auto [_, geometry, __] : r.view<const ObjectGeometry, const SoundVertices>().each()) SoundMeshes.insert(geometry.Entity);
        }
    }

    const state::Registry &R;
    const GpuBuffers &Buffers;
    const MeshStore &Meshes;
    const DisplayUnion &Displays;
    InteractionMode Mode;
    const selection::PrimaryEditObjectMap &Primaries;
    const std::vector<state::Entity> &SelectedMeshes;
    std::unordered_set<state::Entity> SoundMeshes;

    bool OutlinesArmatureBones(state::Entity armature) const {
        if (Mode == InteractionMode::Edit || Mode == InteractionMode::Pose) return R.all_of<Active>(armature);
        return R.all_of<Selected>(armature);
    }
};

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
    const bool mesh_record = DrawsMesh(r, mesh_entity);
    const auto &arenas = meshes.Arenas();
    const uint32_t faces = mesh_record ? arenas.FaceTriangles.Count(mb.FaceData) : 0u, edges = mesh_record ? arenas.EdgeHalfedges.Count(mb.EdgeData) : 0u;
    const bool meshlets = c.Meshes.MeshletCount(mb) > 0u;
    const auto primary = c.Primaries.find(mesh_entity);
    const bool has_primary = primary != c.Primaries.end();
    const bool sound = c.SoundMeshes.contains(mesh_entity);
    const auto &displays = c.Displays;
    const bool overlays = displays.Overlays;
    if (has_primary && mesh_record) {
        d.PrimaryEditInstanceIndex = r.get<const RenderInstance>(primary->second).BufferIndex;
        d.Selection = meshes.GetEditSelectionStorage(mb.StoreId);
    } else if (sound && mesh_record) {
        d.Selection = meshes.GetEditSelectionStorage(mb.StoreId);
        const auto *active = r.try_get<const MeshActiveElement>(mesh_entity);
        d.ActiveVertex = active ? active->Handle : InvalidOffset;
    }

    const bool armature = r.all_of<ArmatureObject>(mesh_entity), joint = r.all_of<BoneJoint>(mesh_entity);
    const bool material_face_less = mesh_record && faces == 0u && mb.PrimitiveMaterials.Count > 0u;
    const bool shaded_face_less = material_face_less && displays.AllMaterialShaded;
    const bool selected = std::ranges::binary_search(c.SelectedMeshes, mesh_entity);
    uint32_t flags = 0u;
    uint8_t vertex_overlays = 0u;
    const auto set = [&](MeshletInstanceFlag flag) { flags |= uint32_t(flag); };
    const auto draw = [&](VertexOverlay overlay) { vertex_overlays |= uint8_t(1u << uint32_t(overlay)); };
    if (IsSilhouetteEligible(r, mesh_entity)) set(MeshletInstanceFlag::SilhouetteEligible);
    if (c.Primaries.contains(mesh_entity) || c.Meshes.HasDirtyPositions(mb)) set(MeshletInstanceFlag::LodPinFinest);
    // A click can query another element domain before its mode action applies.
    if (has_primary && meshlets && mesh_record) set(MeshletInstanceFlag::ElementSelection);
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
    const auto *selection_edit = r.try_get<const graph::SelectionEditing>(state::Document);
    if (armature || joint || (selection_edit && selection_edit->Mesh == mesh_entity)) set(MeshletInstanceFlag::OverlayOnly);
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

void RefreshMeshDisplay(state::Registry &r, const DisplayContext &c, state::Entity mesh_entity, const MeshStore::Record &mb, const PosedLayout *pose) {
    const auto [display, overlays] = ComputeMeshDisplay(c, mesh_entity, mb, pose);
    if (overlays) r.emplace_or_replace<MeshVertexOverlays>(mesh_entity, overlays);
    else r.remove<MeshVertexOverlays>(mesh_entity);
    if (mb.RenderTopologies == 0u) return;
    auto &records = r.Resources.get<MeshStore>().Render().MeshRecords;
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

void LayOutMesh(state::Registry &r, bool edit_mode, state::Entity mesh_entity) {
    auto &buffers = r.Resources.get<GpuBuffers>();
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &render = meshes.Render();
    const auto *models = r.try_get<const ModelsBuffer>(mesh_entity);
    const auto *record = models && DrawsMesh(r, mesh_entity) ? TryRecordOf(r, mesh_entity) : nullptr;
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
    const auto deform = edit_mode ? MeshDeform{} : DeformOf(r, buffers, *record, *models);
    const bool per_instance = deform.PerInstance();
    const uint32_t count = per_instance ? models->InstanceCount : 1u;
    const bool posed = models->InstanceCount > 0u && (deform.BoneDeformOffset != InvalidOffset || deform.MorphDeformOffset != InvalidOffset);

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
        const auto vertex_bounds = buffers.VertexBounds.Prepare(mesh_entity, store_id, vertex_revision, count, vertex_blocks);
        namespaces.VertexBoundsNamespaces.assign(vertex_bounds.Roots.begin(), vertex_bounds.Roots.end());
        for (uint32_t level = 0u; level < VertexBoundsLevels; ++level) {
            run.BoundsKeys[level] = uint32_t(vertex_bounds.Nodes[level].size());
            words.append_range(vertex_bounds.Nodes[level]);
        }
        const bool authored_morph = deform.MorphDeformOffset != InvalidOffset && record->MorphShadingAuthored;
        const auto positions = buffers.PosedPositions.Prepare(mesh_entity, store_id, vertex_revision, count, vertex_blocks);
        namespaces.PositionNamespaces.assign(positions.Roots.begin(), positions.Roots.end());
        if (authored_morph) {
            const auto deltas = buffers.PosedMorphNormalDeltas.Prepare(mesh_entity, store_id, vertex_revision, count, vertex_blocks);
            namespaces.MorphNormalNamespaces.assign(deltas.Roots.begin(), deltas.Roots.end());
        } else {
            buffers.PosedMorphNormalDeltas.Release(mesh_entity);
        }
        if (!authored_morph) derive_entry = MakeDeriveEntryInputs(meshes, store_id);
        if (derive_entry) {
            const auto vertex_normals = buffers.PosedVertexNormals.Prepare(mesh_entity, store_id, vertex_revision, count, vertex_blocks);
            const auto &faces = meshes.Arenas().FaceTriangles;
            const auto face_blocks = PoseElementBlocks(faces, record->FaceData);
            const auto face_normals = buffers.PosedFaceNormals.Prepare(mesh_entity, store_id, faces.Set(record->FaceData).Revision, count, face_blocks);
            const auto sectors = buffers.PosedSectors.Prepare(mesh_entity, store_id, meshes.GetDerived(store_id).NormalRevision, count, SectorBlocks(meshes, store_id));
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
        std::vector<uint32_t> meshlet_blocks;
        const auto meshlets = render.Meshlets.Entries(record->MeshletData);
        meshes.ForEachPrimitive(*record, [&](ElementHandle, const PrimitiveRecord &primitive) {
            if (primitive.LodFinestNode == InvalidOffset) return;
            const auto &node = render.LodNodes.Get({render.LodNodes.Address(record->NodeData, ElementHandle{primitive.LodFinestNode}).Value(), 1u})[0];
            meshes.ForEachLodMember(*record, node, [&](uint32_t cluster) { meshlet_blocks.push_back(Resolve<RenderBlockSize>(meshlets, ElementHandle{cluster}).Value() / MeshElementBlockSize); });
            if (node.Members == InvalidOffset || !node.MeshletCount) return;
            words.insert(words.end(), {node.Members, node.MeshletCount});
            ++run.MeshletJobs;
        });
        SortUnique(meshlet_blocks);
        const auto meshlet_bounds = buffers.PosedMeshletBounds.Prepare(mesh_entity, store_id, record->MeshletRevision, count, meshlet_blocks, record->RenderTopologies);
        namespaces.MeshletBoundsNamespaces.assign(meshlet_bounds.Roots.begin(), meshlet_bounds.Roots.end());
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

constexpr MeshletInstanceFlag WireFlagOf(const ViewportDisplay &settings) {
    return settings.ViewportShading == ViewportShadingMode::Wireframe ? MeshletInstanceFlag::Wire :
        WorkbenchShading(settings.ViewportShading)                    ? MeshletInstanceFlag::SolidWire :
                                                                        MeshletInstanceFlag::MaterialWire;
}

constexpr uint32_t LodPinFlagsOf(const ViewportDisplay &settings) {
    auto flags = uint32_t(MeshletInstanceFlag::LodPinFinest) | uint32_t(WireFlagOf(settings));
    if (settings.ShowOverlays) {
        flags |= uint32_t(MeshletInstanceFlag::EdgeOverlay);
        if (settings.Selection.FaceNormals) flags |= uint32_t(MeshletInstanceFlag::FaceNormal);
        if (settings.Selection.VertexNormals) flags |= uint32_t(MeshletInstanceFlag::VertexNormal);
    }
    return flags;
}

// Consecutive texture arguments in the ray kernels' Metal signatures.
void BindTextures(mtl::Command &command, uint32_t first, std::initializer_list<MTL::Texture *> textures) {
    for (const auto *texture : textures) command.SetTexture(texture, first++);
}

void RecordFastGi(mtl::PassChain &chain, const MainPipeline &main, const RenderTargets::RayTracingResourcesT &rt, const RenderTargets::ScreenT &screen, const RenderTargets::RayTracingResourcesT::HistoryT &previous, const RenderTargets::RayTracingResourcesT::HistoryT &next, const FastGiPushConstants &pc) {
    const auto &gi = *rt.FastGI;
    // Each Fast GI kernel uses consecutive texture slots beginning at zero.
    const auto dispatch = [](mtl::ComputeEncoder encoder, const mtl::ComputePipeline &pipeline, mtl::Extent2D extent, std::initializer_list<MTL::Texture *> textures) {
        pipeline.Bind(encoder);
        BindTextures(*encoder.Cmd, 0, textures);
        encoder->dispatchThreads({extent.Width, extent.Height, 1}, ThreadgroupSize::Tile8);
    };
    const auto depth = chain.BeginCompute("FastGiDepth", MTL::StageDispatch | MTL::StageFragment);
    dispatch(depth, main.FastGiDepth, gi.Depth.Extent, {*screen.VisibilityDepth, *gi.DepthMips[0]});
    for (uint32_t mip = 1; mip < gi.Depth.MipLevels; ++mip) {
        mtl::ComputeBarrier(depth);
        dispatch(depth, main.FastGiDepthReduce, gi.DepthMips[mip].Extent, {*gi.DepthMips[mip - 1], *gi.DepthMips[mip]});
    }

    const auto setup = chain.BeginCompute("FastGiSetup", MTL::StageDispatch);
    setup.Cmd->SetBytes(pc, 1);
    dispatch(setup, main.FastGiSetup, gi.Radiance.Extent, {
                                                              *rt.Guide,
                                                              *screen.SceneColorImage,
                                                              *next.Geometry,
                                                              *rt.Feedback,
                                                              *previous.Geometry,
                                                              *previous.Normal,
                                                              *previous.MaterialValidity,
                                                              *rt.TransmissionWeight,
                                                              *rt.ReflectionWeight,
                                                              *gi.RadianceMips[0],
                                                              *gi.NormalMips[0],
                                                              *rt.ClosureNormal,
                                                              *(rt.CoatWeight ? rt.CoatWeight : rt.ReflectionWeight),
                                                          });
    for (uint32_t mip = 1; mip < gi.Radiance.MipLevels; ++mip) {
        mtl::ComputeBarrier(setup);
        dispatch(setup, main.FastGiReduce, gi.RadianceMips[mip].Extent, {*gi.RadianceMips[mip - 1], *gi.NormalMips[mip - 1], *gi.RadianceMips[mip], *gi.NormalMips[mip]});
    }

    dispatch(chain.BeginCompute("FastGiScan", MTL::StageDispatch), main.FastGiScan, gi.SH.Extent, {*gi.Depth, *gi.Radiance, *gi.Normal, *rt.Guide, *gi.SH});
    dispatch(chain.BeginCompute("FastGiDenoise", MTL::StageDispatch), main.FastGiDenoise, gi.FilteredSH.Extent, {*gi.SH, *gi.FilteredSH, *gi.NormalMips[0], *gi.DepthMips[0]});
    dispatch(chain.BeginCompute("FastGiResolve", MTL::StageDispatch), main.FastGiResolve, rt.Guide.Extent, {
                                                                                                               *gi.FilteredSH,
                                                                                                               *gi.NormalMips[0],
                                                                                                               *gi.DepthMips[0],
                                                                                                               *rt.Guide,
                                                                                                               *rt.TransmissionWeight,
                                                                                                               *rt.ReflectionWeight,
                                                                                                               *rt.ClosureNormal,
                                                                                                               *screen.SceneColorImage,
                                                                                                               *(rt.CoatWeight ? rt.CoatWeight : rt.ReflectionWeight),
                                                                                                           });
}

void RecordPhase(state::Registry &r, state::Entity viewport_entity, mtl::PassChain &chain, const PreludeSource &prelude, RenderPhase phase, uint32_t ubo_offset, uint32_t sample_weight = 1u) {
    const profile::CpuScope scope{"RecordRenderCommandBuffer"};
    const bool draw_scene = phase != RenderPhase::BlurResolve;
    const bool draw_overlays = !IsBlurAccumulate(phase);

    const auto &slots = r.Resources.get<const mtl::BindlessSet>();
    auto &buffers = r.Resources.get<GpuBuffers>();
    const auto &meshes = r.Resources.get<MeshStore>();
    auto &pipelines = GetPipelines(r);
    const auto logical_extent = r.get<const ViewportPanel>(viewport_entity).Extent;
    auto &viewport = r.edit<Viewport>(viewport_entity);
    const auto &settings = viewport.FrameView.Display;
    auto &targets = viewport.Targets;
    const auto interaction_mode = r.get<const Interaction>(state::Document).Mode;
    const auto edit_mode = r.get<const EditMode>(state::Document).Value;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;
    const bool is_excite_mode = interaction_mode == InteractionMode::Excite;
    const bool is_wireframe_mode = settings.ViewportShading == ViewportShadingMode::Wireframe;
    const bool show_rendered = !WorkbenchShading(settings.ViewportShading);
    const bool show_fill = !is_wireframe_mode;
    const bool xray = XRayActive(settings);
    const float overlay_behind = OverlayBehindOpacity(settings, interaction_mode);
    const bool overlays_through = overlay_behind > 0.f;
    const bool overlay_scene_depth = overlay_behind < 1.f;
    const bool show_overlays = settings.ShowOverlays;
    auto &scene_state = r.Resources.get<GpuSceneState>();
    const auto &lighting = ActivePbrLighting(r, viewport_entity, settings);
    auto pbr_features = scene_state.MeshPbrFeatures & MeshMaterialFeatures;
    if (lighting.UseSceneLights) pbr_features |= PbrFeature::SceneLights;
    const auto view_at = [&](uint32_t offset) -> SceneViewUBO & { return *reinterpret_cast<SceneViewUBO *>(viewport.SceneViewUBO.Contents().data() + offset); };
    auto &canonical = view_at(ubo_offset);
    RayTracingMaterials usage;
    bool ray_tracing = show_rendered && settings.RayTracing.Enabled && settings.DebugChannel == DebugChannel::None;
    viewport.RayTracingStatus.clear();
    if (!settings.RayTracing.Enabled) {
        viewport.Rays.reset();
        targets.RayTracing.reset();
    }
    if (ray_tracing) {
        const auto &ctx = r.Resources.get<mtl::Context>();
        if (!ctx.Device->supportsFamily(MTL::GPUFamilyApple10) || !ctx.Device->supportsRaytracing()) viewport.RayTracingStatus = "Ray tracing requires an M5 or newer GPU.";
        else {
            const bool classify = !is_edit_mode && !canonical.IsTransforming && canonical.RayShadowSamples > 1u && lighting.UseSceneLights && canonical.LightCount;
            usage = RayTracingMaterialUsage(meshes, buffers, canonical, classify);
            // Edit-mode hidden faces require per-hit rejection.
            if (!is_edit_mode && !HasFeature(pbr_features, PbrFeature::Transmission) && (usage.OpaqueSingleSided || usage.OpaqueDoubleSided)) {
                pbr_features |= PbrFeature::OpaqueShadows;
                if (usage.OpaqueSingleSided) pbr_features |= PbrFeature::CullShadowBackfaces;
            }
        }
        ray_tracing = viewport.RayTracingStatus.empty();
    }
    const bool ray_blended = usage.Coverage || usage.BlendedTransmission;
    const bool ray_glass = HasFeature(pbr_features, PbrFeature::Transmission);
    const bool traced_diffuse = settings.RayTracing.Diffuse == DiffuseLighting::RayTraced;
    const uint32_t canonical_ubo_offset = ubo_offset;
    const uint32_t canonical_view_index = uint32_t(canonical_ubo_offset / GpuBuffers::ViewUboStride());
    if (ray_tracing && draw_scene && phase != RenderPhase::Prepare) {
        canonical.RayActive = true;
        ubo_offset = GpuBuffers::RayViewUboOffset(canonical_view_index);
        std::memcpy(&view_at(ubo_offset), &canonical, sizeof(SceneViewUBO));
    }
    auto &ray_view = view_at(ubo_offset);
    ray_view.LightTileSlot = InvalidSlot;
    ray_view.RayAreaLightSlot = InvalidSlot;
    if (ray_tracing && lighting.UseSceneLights && ray_view.LightCount >= 32u &&
        std::ranges::any_of(buffers.LightsFor(ray_view), [](const auto &light) { return light.Type != LightType::Directional && light.Range > 0.f && std::isfinite(light.Range); }))
        pbr_features |= PbrFeature::TiledLights;
    if (lighting.UseSceneLights && ray_view.LightCount && std::ranges::any_of(buffers.LightsFor(ray_view), [](const auto &light) { return IsAreaLight(light.Type); }))
        pbr_features |= PbrFeature::AreaLights;
    if (ray_tracing && HasFeature(pbr_features, PbrFeature::AreaLights) && ray_view.RayShadowSamples > 1u)
        pbr_features |= PbrFeature::MultipleShadowSamples;
    if (HasFeature(pbr_features, PbrFeature::MultipleShadowSamples) && (ray_blended || ray_glass) && !usage.ComplexCasterSlots.empty())
        pbr_features |= PbrFeature::BoundedAreaShadows;
    const auto materials = buffers.MaterialsFor(ray_view);
    ray_view.RayActive = ray_tracing;
    const bool real_transmission = !ray_tracing && RealTransmission(r, viewport_entity, settings, scene_state.MeshPbrFeatures);
    const auto &samplers = viewport.Samplers;
    if (std::exchange(scene_state.OverlayJobsDirty, false)) BuildOverlayJobs(r, buffers, scene_state.Displays.BoundingBoxes);
    const bool render_silhouette = show_overlays && settings.Overlays.OutlineSelected && !is_excite_mode &&
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::Silhouette)).Meshlets > 0u;

    if (phase == RenderPhase::Full) RecordSceneCounters(buffers, viewport);

    const bool transmission_active = real_transmission && targets.Transmission;
    const bool composite_transmission = transmission_active && phase == RenderPhase::Full && !is_edit_mode && settings.DebugChannel == DebugChannel::None;
    const bool meshlet_fill = buffers.MeshletInstanceCount > 0;

    // Posed positions and bounds run before culling, over the entries whose inputs changed, and every posed entry in each blur sample.
    const bool prelude_work = !prelude.Empty();
    if (prelude_work) {
        const auto compute = chain.BeginCompute("Prelude", MTL::StageVertex | MTL::StageFragment | MTL::StageDispatch);
        encode::BindScene(compute, slots, buffers, viewport, ubo_offset);
        RecordPrelude(compute, pipelines, GetMeshPipelines(r), buffers, meshes, prelude);
        for (const auto [_, other] : r.view<Viewport>().each()) {
            other.OcclusionStale = true;
            if (other.Rays) other.Rays->InvalidatePositions();
        }
    }
    if (is_edit_mode && std::exchange(scene_state.EditPositionsPending, false)) {
        for (const auto [_, other] : r.view<Viewport>().each())
            if (other.Rays) other.Rays->InvalidatePositions(true);
    }
    if (phase == RenderPhase::Prepare) return;
    uint64_t hit_instances{};
    if (ray_tracing && draw_scene) {
        if (!viewport.Rays) viewport.Rays = std::make_unique<RayTracingScene>(chain.Cmd.Ctx);
        hit_instances = viewport.Rays->Build(chain, meshes, buffers, ray_view, r, pipelines.Main);
        ray_tracing = hit_instances != 0;
        ray_view.RayInstanceCount = viewport.Rays->InstanceCount();
    }
    ray_view.RayActive = ray_tracing;
    if (!ray_tracing && targets.RayTracing) targets.RayTracing->Frames = 0;
    if (buffers.DrawsNonTriangles()) pbr_features |= PbrFeature::NonTriangleTopology;
    if (ray_tracing) {
        pbr_features |= PbrFeature::RayTracing;
        if (usage.Opaque) pbr_features |= PbrFeature::OpaqueReceiver;
        if (ray_blended) pbr_features |= PbrFeature::AlphaBlend;
        if (traced_diffuse) pbr_features |= PbrFeature::TraceDiffuse;
        if (settings.RayTracing.Reflections) pbr_features |= PbrFeature::TraceReflections;
    }
    // A material in view that computes inputs makes its computed lobes live, and the points it shades read them from the pattern words.
    auto &patterns = r.Resources.get<MaterialPatterns>();
    const auto drawn_patterns = show_rendered ? DrawnPatternFeatures(patterns, meshes, buffers, canonical) : DrawnPatterns{};
    pbr_features |= drawn_patterns.Features;
    const auto main_extent = targets.Screen->SceneColorImage.Extent;
    auto *const pattern_targets = targets.EnsurePatternResources(r.Resources.get<const mtl::Context>(), buffers.Ctx, HasFeature(pbr_features, PbrFeature::Patterns) ? patterns.Words : 0u, ray_tracing ? main_extent.Width * main_extent.Height * RayLightingLobeCount(ray_glass, traced_diffuse) : 0u, drawn_patterns.Transparent);
    const auto timeline_frame = r.get<const TimelinePlayback>(state::Document).CurrentFrame;
    for (auto *view : {&canonical, &ray_view}) {
        view->PatternWordSlot = pattern_targets ? pattern_targets->Words->Slot : InvalidSlot;
        view->PatternStride = pattern_targets ? patterns.Words : 0u;
        view->PatternFrame = pattern_targets ? float(timeline_frame) : 0.f;
        view->PatternTime = pattern_targets ? animation::FrameSeconds(r, timeline_frame) : 0.f;
        view->PatternWidth = pattern_targets ? main_extent.Width : 0u;
        view->PatternRing = pattern_targets ? pattern_targets->Ring : 0u;
        const auto fragments = pattern_targets ? pattern_targets->Fragments : RenderTargets::PatternResourcesT::FragmentLists{};
        view->PatternFragmentHeads = fragments.Heads;
        view->PatternFragmentEntries = fragments.Entries;
        view->PatternFragmentWords = fragments.Words;
        view->PatternFragmentCapacity = fragments.Capacity;
    }
    if (pattern_targets && !patterns.Pipelines) {
        auto &libraries = r.Resources.get<mtl::LibraryCache>();
        const auto driver = [&](const char *name) { return mtl::ComputePipeline{libraries, {.Path = "SurfacePattern.metal", .Name = name, .BinaryLinking = true}}; };
        const auto kernel = [&](const char *name) { return mtl::ComputePipeline{libraries, {"SurfacePattern.metal", name}}; };
        auto &programs = r.Resources.get<mtl::ProgramRegistry>();
        patterns.Pipelines = std::make_unique<MaterialPatterns::Passes>(
            kernel("SurfacePatternClassify"), kernel("SurfacePatternArguments"), kernel("SurfacePatternHitArguments"),
            mtl::AddDriver(programs, driver("SurfacePatternDriver")), mtl::AddDriver(programs, driver("SurfacePatternHitDriver"))
        );
    }
    const auto *pbr = show_rendered ? &pipelines.Main.Compiler.Variant(pbr_features, r.Resources.find<mtl::ProgramRegistry>()) : nullptr;
    // Previous completed medium state schedules the cache; current GPU state still gates every lookup.
    const bool opaque_layer = pbr && pbr->OpaqueLayer && viewport.Rays->PreviousCameraInsideMedium();
    mtl::RenderEncoder encoder;
    const auto draw_quad = [&] { encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4)); };
    const auto decode_visibility = [&] {
        encoder.Cmd->SetTexture(*targets.Screen->VisibilityImage, 0u);
        encode::SetPushConstants(encoder, encode::VisibilityDecodePc(buffers));
        draw_quad();
    };

    const auto &main = pipelines.Main;
    const mat4 camera_view_proj = ray_view.ViewProj;
    const auto step = uint32_t(settings.RayTracing.Resolution);
    const bool denoise = settings.RayTracing.Denoise;
    const uint32_t samples = denoise ? ray_view.RaySamples : 1u;
    const bool quincunx_glass = ray_glass && denoise && step >= 4u;
    const uint32_t ray_lobes = RayLightingLobeCount(ray_glass, traced_diffuse);
    // The reflection ray carries the coat, the diffuse ray carries diffuse transmission, and sheen takes the diffuse ray or else the reflection ray.
    const bool trace_coat = HasFeature(pbr_features, PbrFeature::Clearcoat) && settings.RayTracing.Reflections;
    const bool trace_diffuse_transmission = HasFeature(pbr_features, PbrFeature::DiffuseTrans) && traced_diffuse;
    const bool trace_sheen = HasFeature(pbr_features, PbrFeature::Sheen) && (settings.RayTracing.Reflections || traced_diffuse);
    const bool trace_surface = ray_tracing && draw_scene && meshlet_fill;
    bool history_valid = false, object_motion = false, scene_changed = false;
    vec2 film_jitter{};
    if (trace_surface) {
        auto &rt = targets.EnsureRayTracingResources(chain.Cmd.Ctx, step, ray_lobes, ray_glass, trace_coat, trace_diffuse_transmission, trace_sheen);
        auto signature_view = ray_view;
        signature_view.ViewProj = mat4{1};
        signature_view.ViewRotation = mat3{1};
        signature_view.CameraPosition = {};
        signature_view.RayFrame = 0;
        signature_view.IsTransforming = 0;
        signature_view.PendingPivot = {};
        signature_view.PendingTranslation = {};
        signature_view.PendingRotation = quat{1, 0, 0, 0};
        signature_view.PendingScale = vec3{1};
        std::vector<std::byte> shading;
        const auto append = [&](const auto &value) { shading.append_range(BytesOf(value)); };
        append(signature_view);
        append(materials);
        append(buffers.ViewportThemeUBO.GetSpan<std::byte>());
        append(viewport.WorkspaceLightsUBO.GetSpan<std::byte>());
        for (const auto &states : buffers.Instances.States) append(states.GetSpan<std::byte>());
        for (const auto &texture : r.Resources.get<const TextureStore>().Textures) {
            append(std::array{(*texture.Image)->gpuResourceID()._impl, texture.Sampler->gpuResourceID()._impl});
        }
        append(meshes.Arenas().PrimitiveMaterials.Buffer.GetSpan<std::byte>());
        // A pattern's values are its inputs, as a material's record is its own.
        for (const auto values : patterns.Values) append(buffers.OpValues.Get(values));
        append(denoise);
        append(traced_diffuse);
        if (ray_view.RayFastGI) append(settings.RayTracing.FastGiResolution);
        const auto shading_hash = store::HashBytes(shading);
        shading.clear();
        const auto light_transforms = buffers.Instances.TransformBuffer.GetSpan<Transform>();
        for (const auto &lights : buffers.Lights) {
            append(lights.GetSpan<std::byte>());
            for (const auto &light : lights.GetSpan<LightRecord>())
                append(light_transforms[light.TransformSlotOffset.Offset]);
        }
        const auto light_hash = store::HashBytes(shading);
        object_motion = rt.SceneRevision != viewport.Rays->Revision();
        scene_changed = object_motion || rt.PreviousCameraProj != camera_view_proj || rt.LightHash != light_hash;
        if (!denoise || IsBlurAccumulate(phase) || rt.ShadingHash != shading_hash || rt.GeometryRevision != viewport.Rays->GeometryRevision()) rt.Frames = rt.SampleIndex = rt.JitterIndex = 0;
        history_valid = rt.Frames > 0;
        if (scene_changed) rt.Frames = 0;
        if (rt.Frames == 0) rt.FilmWeight = 0.f;
        ray_view.RayFrame = 0;
        if (denoise) {
            ray_view.RayFrame = rt.SampleIndex + (IsBlurAccumulate(phase) ? (canonical_view_index - 1u) * samples : 0u);
            rt.SampleIndex += samples;
            film_jitter = RayFilmJitter(IsBlurAccumulate(phase) ? canonical_view_index - 1u : rt.JitterIndex++);
            for (uint32_t column = 0; column < 4; ++column) {
                ray_view.ViewProj[column].x += 2.f * film_jitter.x / float(main_extent.Width) * camera_view_proj[column].w;
                ray_view.ViewProj[column].y -= 2.f * film_jitter.y / float(main_extent.Height) * camera_view_proj[column].w;
            }
        }
        rt.ShadingHash = shading_hash;
        rt.LightHash = light_hash;
        rt.SceneRevision = viewport.Rays->Revision();
        rt.GeometryRevision = viewport.Rays->GeometryRevision();
    }

    const bool film_layer = trace_surface && history_valid && scene_changed && usage.BlendedTransmission;
    // Both workspace buffers must fit every shutter pose before the first GPU consumer is encoded.
    const auto max_pose_lights = [&] {
        uint32_t max_lights = ray_view.LightCount;
        for (const auto &lights : buffers.Lights) max_lights = std::max(max_lights, lights.Count<LightRecord>());
        for (const auto &pose : buffers.BlurPoses)
            for (const auto &lights : pose.Lights) max_lights = std::max(max_lights, lights.Count<LightRecord>());
        return max_lights;
    };
    const auto reserve_lighting = [&](std::unique_ptr<mtl::Buffer> &buffer, uint64_t bytes) {
        if (!buffer) buffer = std::make_unique<mtl::Buffer>(buffers.Ctx, bytes, SlotType::Buffer, mtl::BufferLifetime::Workspace);
        else buffer->Reserve(bytes);
        return buffer->Slot;
    };
    if (trace_surface && HasFeature(pbr_features, PbrFeature::AreaLights)) {
        auto &prepared = targets.RayTracing->PreparedAreaLights;
        const uint64_t bytes = uint64_t(max_pose_lights()) * sizeof(PreparedAreaLight) + sizeof(AreaShadowBounds);
        ray_view.RayAreaLightSlot = reserve_lighting(prepared, bytes);
        const auto prepare = chain.BeginCompute("LightPrepare", MTL::StageAll);
        encode::BindCompute(prepare, main.LightPrepare, slots, buffers, viewport, ubo_offset);
        prepare.Cmd->SetBuffer(**prepared, 0, 1);
        prepare->dispatchThreads({ray_view.LightCount, 1, 1}, {32, 1, 1});
        if (HasFeature(pbr_features, PbrFeature::BoundedAreaShadows)) {
            const auto *pose = buffers.RenderPoseFor(ray_view);
            const auto &models = pose ? pose->Transforms : buffers.Instances.TransformBuffer;
            const AreaShadowBoundsParams params{uint32_t(usage.ComplexCasterSlots.size()), buffers.Instances.BoundsBuffer.Slot, models.Slot, uint32_t(usage.CullOpaqueBackfaces)};
            profile::RecordCounter("AreaShadowBoundedInstances", usage.ComplexCasterSlots.size());
            main.RayShadowBounds.Bind(prepare);
            prepare.Cmd->SetBuffer(**prepared, uint64_t(ray_view.LightCount) * sizeof(PreparedAreaLight), 1);
            prepare.Cmd->SetBytes(params, 2);
            prepare.Cmd->SetBytes(std::as_bytes(std::span{usage.ComplexCasterSlots}), 3);
            prepare->dispatchThreadgroups({1, 1, 1}, {32, 1, 1});
        }
        profile::RecordCounter("PreparedAreaLightBytes", bytes);
    }
    if (trace_surface && HasFeature(pbr_features, PbrFeature::TiledLights)) {
        auto &words = targets.RayTracing->LightTiles;
        const uint64_t tiles = uint64_t((main_extent.Width + LightTileSize - 1u) / LightTileSize) * ((main_extent.Height + LightTileSize - 1u) / LightTileSize);
        const uint64_t bytes = tiles * ((max_pose_lights() + 31u) / 32u) * sizeof(uint32_t);
        ray_view.LightTileSlot = reserve_lighting(words, bytes);
        ray_view.LightTileViewport = ray_view.ViewportSize;
        const auto cull = chain.BeginCompute("LightCull", MTL::StageAll);
        encode::BindCompute(cull, main.LightCull, slots, buffers, viewport, ubo_offset);
        cull.Cmd->SetBuffer(**words, 0, 1);
        cull->dispatchThreadgroups({tiles * ((ray_view.LightCount + 31u) / 32u), 1, 1}, {32, 1, 1});
        profile::RecordCounter("LightTileBytes", bytes);
    }
    const bool has_silhouette = render_silhouette && meshlet_fill;
    const bool need_visibility = meshlet_fill && (show_fill || has_silhouette || overlay_scene_depth);
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
    const bool transparent = xray_fill || (!ray_tracing && show_rendered && (real_transmission || std::ranges::any_of(buffers.Materials.GetSpan<PBRMaterial>(), [](const auto &m) { return m.AlphaMode == MaterialAlphaMode::Blend; })));
    const auto &current_view_proj = ray_view.ViewProj;
    const bool disocclusion_possible = viewport.OcclusionStale ||
        std::memcmp(&current_view_proj, &targets.Screen->DepthPyramidViewProj, sizeof(mat4)) != 0; // NOLINT(bugprone-suspicious-memory-comparison)
    // Cached occlusion is valid only for the pose that produced it.
    // Reordering opaque surfaces across temporal phases changes the winner of equal-depth raster ties.
    const auto record_cull = [&](uint32_t pyramid) {
        RecordMeshletCull(
            chain, slots, pipelines, buffers, viewport,
            {
                .Mode = !show_rendered ? MeshletRouteMode::Visibility : real_transmission ? MeshletRouteMode::Transmission :
                                                                                            MeshletRouteMode::Material,
                .RequiredInstanceFlags = show_fill || overlay_scene_depth || wire_meshlets || bone_meshlets > 0u ||
                        normal_meshlets > 0u || element_overlay_meshlets > 0u ?
                    0u :
                    uint32_t(MeshletInstanceFlag::Silhouette),
                .RouteMask = 0x1ffu & ~(1u << uint32_t(MeshletRoute::EditOverlay)),
                .UboOffset = ubo_offset,
                .PyramidSamplerSlot = pyramid,
            }
        );
    };
    if (cull_scene_meshlets) {
        const uint32_t pyramid = !opaque_layer && !film_layer && show_fill && !xray && phase == RenderPhase::Full && targets.Screen->DepthPyramidValid && !disocclusion_possible ? samplers.DepthPyramid : InvalidSlot;
        record_cull(pyramid);
    }
    // Overlay depth needs a cleared visibility depth even without meshlets.
    if (need_visibility || overlay_scene_depth || phase == RenderPhase::BlurFast) {
        RecordMeshletVisibilityPass(chain, slots, pipelines, viewport, buffers, real_transmission, ubo_offset, {});
    }
    // The programs of the patterned materials write their pixels' pattern words before the passes that shade the visibility image read them.
    if (pattern_targets && need_visibility) {
        const auto &passes = *patterns.Pipelines;
        const auto &work = pattern_targets->Work;
        chain.BeginBlit(MTL::StageFragment)->fillBuffer(*work, NS::Range::Make(0, sizeof(uint32_t)), 0);
        const auto compute = chain.BeginCompute("SurfacePattern", MTL::StageFragment | MTL::StageBlit);
        encode::BindScene(compute, slots, buffers, viewport, ubo_offset);
        encode::SetPushConstants(compute, encode::VisibilityDecodePc(buffers));
        compute.Cmd->SetTexture(*targets.Screen->VisibilityImage, 0u);
        compute.Cmd->SetBuffer(*work, 0, BufferIndex_PatternWork);
        passes.Classify.Bind(compute);
        compute->dispatchThreads({main_extent.Width, main_extent.Height, 1}, ThreadgroupSize::Tile8);
        mtl::ComputeBarrier(compute);
        passes.Arguments.Bind(compute);
        compute->dispatchThreads({1, 1, 1}, {1, 1, 1});
        mtl::ComputeBarrier(compute);
        mtl::Bind(compute, passes.Driver, BufferIndex_Programs);
        compute->dispatchThreadgroups(work.Address() + sizeof(uint32_t), {64, 1, 1});
    }
    if (show_fill && phase == RenderPhase::Full && cull_scene_meshlets) {
        targets.Screen->DepthPyramidViewProj = current_view_proj;
        const auto compute = chain.BeginCompute("DepthPyramidFinal", MTL::StageFragment);
        RecordDepthPyramid(
            compute, slots, buffers, viewport, pipelines, targets.Screen->DepthPyramid, samplers.DepthPyramid, 0u,
            samplers.SceneDepth, targets.Screen->VisibilityDepth.Extent, false, ubo_offset
        );
        targets.Screen->DepthPyramidValid = true;
    }
    if (has_silhouette) RecordSilhouetteDepthPass(chain, slots, pipelines, viewport, buffers, ubo_offset);

    if (transmission_active && draw_scene) {
        const std::array colors{mtl::ClearColor(*targets.Transmission->Mip0View)};
        const auto pass = mtl::MakePassDescriptor(colors, mtl::LoadDepth(*targets.Screen->VisibilityDepth));
        encoder = encode::BeginScenePass(chain, pass.get(), "TransmissionPrepass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, viewport, ubo_offset);
        main.PrepassBackground.Bind(encoder);
        draw_quad();
        if (meshlet_fill && show_fill) {
            pbr->Prepass->Bind(encoder);
            decode_visibility();
        }

        if (targets.Transmission->Image.MipLevels > 1) mtl::RecordMipmaps(chain.BeginCompute("TransmissionMips", MTL::StageFragment), pipelines.MipDownsample, targets.Transmission->Mips);
    }

    // The programs of the patterned transparent materials compute their fragments' inputs before the transparent passes shade the fragments.
    const bool pattern_fragments = pattern_targets && pattern_targets->Fragments.Capacity && show_fill && meshlet_fill && draw_scene && show_rendered && (transparent || usage.Coverage);
    if (pattern_fragments) {
        pattern_targets->Listed = true;
        auto *words = **pattern_targets->Words;
        const auto &lists = pattern_targets->Fragments;
        const auto blit = chain.BeginBlit(MTL::StageFragment);
        blit->fillBuffer(words, NS::Range::Make(0, sizeof(uint32_t)), 0);
        blit->fillBuffer(words, NS::Range::Make(uint64_t(lists.Heads) * sizeof(uint32_t), uint64_t(lists.Entries - lists.Heads) * sizeof(uint32_t)), 0xFF);
        const auto pass = mtl::MakePassDescriptor({}, !xray ? mtl::LoadDepth(*targets.Screen->VisibilityDepth) : mtl::ClearDepth(*targets.Screen->ScratchDepth));
        encoder = encode::BeginScenePass(chain, pass.get(), "SurfacePatternList", {{MTL::StageBlit | MTL::StageDispatch, MTL::StageMesh | MTL::StageFragment}}, main_extent, slots, buffers, viewport, ubo_offset);
        mtl::Bind(encoder, *pbr->PatternList, BufferIndex_PatternPrograms);
        // A fragment resolves its triangle from its key to read named attributes.
        encode::SetFragmentPushConstants(encoder, encode::VisibilityDecodePc(buffers));
        if (real_transmission) DrawMeshlets(encoder, buffers, uint32_t(MeshletRoute::Transmission));
        DrawMeshlets(encoder, buffers, uint32_t(MeshletRoute::Blend));
    }

    const auto configure_transparency = [&](auto &pass) {
        pass->setImageblockSampleLength(std::max(main.TransparencyInit.ImageblockSampleLength(), main.TransparencyResolve.ImageblockSampleLength()));
        pass->setTileWidth(16);
        pass->setTileHeight(16);
    };
    const auto draw_transparency = [&] {
        main.TransparencyInit.Bind(encoder);
        draw_quad();
        pbr->Transparent.Bind(encoder);
        if (real_transmission) DrawMeshlets(encoder, buffers, uint32_t(MeshletRoute::Transmission));
        DrawMeshlets(encoder, buffers, uint32_t(MeshletRoute::Blend));
        // In a frame with patterned materials the forward pipeline draws the other materials' triangles, and the pattern pipeline draws theirs.
        if (pattern_fragments) {
            pbr->TransparentPatterns->Bind(encoder);
            if (real_transmission) DrawMeshlets(encoder, buffers, uint32_t(MeshletRoute::Transmission));
            DrawMeshlets(encoder, buffers, uint32_t(MeshletRoute::Blend));
        }
        main.TransparencyResolve.Bind(encoder);
        draw_quad();
    };
    {
        const std::array colors{
            mtl::ClearColor(*targets.Screen->SceneColorImage),
        };
        const auto depth = show_fill && meshlet_fill && !xray ? mtl::LoadDepth(*targets.Screen->VisibilityDepth) : mtl::ClearDepth(*targets.Screen->ScratchDepth);
        const auto pass = mtl::MakePassDescriptor(colors, depth);
        if (transparent) configure_transparency(pass);
        encoder = encode::BeginScenePass(
            chain, pass.get(), draw_scene ? "ScenePass" : "SceneDepthPass",
            {{MTL::StageAccelerationStructure, MTL::StageFragment}, {MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment | MTL::StageBlit, MTL::StageFragment}},
            main_extent, slots, buffers, viewport, ubo_offset
        );

        if (composite_transmission) {
            main.TransmissionComposite.Bind(encoder);
            draw_quad();
        } else if (show_rendered && draw_scene) {
            main.Background.Bind(encoder);
            draw_quad();
        }
        if (phase == RenderPhase::BlurResolve) {
            main.MotionBlurResolve.Bind(encoder);
            const MotionBlurResolvePushConstants resolve_pc{.AccumSamplerSlot = samplers.MotionBlurOutput, .InvSteps = 1.f / float(MotionBlurSteps(settings))};
            encode::SetPushConstants(encoder, resolve_pc);
            draw_quad();
        }

        if (show_fill) {
            if (meshlet_fill && draw_scene && show_rendered) {
                if (!composite_transmission && !ray_tracing) {
                    pbr->Visibility.Bind(encoder);
                    decode_visibility();
                }
                if (transparent) draw_transparency();
            } else if (xray_fill && draw_scene) {
                main.TransparencyInit.Bind(encoder);
                draw_quad();
                main.WorkspaceTransparent.Bind(encoder);
                for (const auto [route, cull] : VisibilityRoutes) {
                    SetCullMode(encoder, cull);
                    DrawMeshlets(encoder, buffers, uint32_t(route));
                }
                main.TransparencyResolve.Bind(encoder);
                draw_quad();
            } else if (meshlet_fill && draw_scene) {
                main.WorkspaceVisibility.Bind(encoder);
                decode_visibility();
            }
        }
    }

    if (trace_surface) {
        auto &rt = *targets.RayTracing;
        const auto visibility_pc = encode::VisibilityDecodePc(buffers);
        const auto clip_to_world = Inverse(current_view_proj);
        const auto copy_image = [&](MTL::Stages after, const mtl::Texture &source, const mtl::Texture &destination) {
            chain.BeginBlit(after)->copyFromTexture(*source, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(main_extent.Width, main_extent.Height, 1), *destination, 0, 0, MTL::Origin(0, 0, 0));
        };
        const auto media = chain.BeginCompute("RayCameraMedia", MTL::StageAccelerationStructure | MTL::StageDispatch);
        encode::BindScene(media, slots, buffers, viewport, ubo_offset);
        main.RayCameraMedia.Bind(media);
        viewport.Rays->Bind(chain.Cmd, hit_instances);
        media->dispatchThreads({1, 1, 1}, {1, 1, 1});
        if (!usage.BlendedTransmission) rt.FilmLayer.reset();
        const auto draw_visibility_layer = [&](const RenderTargets::RayTracingResourcesT::VisibilityLayer &layer, const mtl::RenderPipeline &pipeline, std::string_view name, uint32_t offset) {
            const std::array colors{mtl::ClearColor(*layer.Visibility, MTL::ClearColor{double(UINT32_MAX), 0, 0, 0})};
            const auto pass = mtl::MakePassDescriptor(colors, mtl::ClearDepth(*layer.Depth));
            const auto visible = encode::BeginScenePass(chain, pass.get(), name, {{MTL::StageDispatch, MTL::StageMesh}}, layer.Visibility.Extent, slots, buffers, viewport, offset);
            pipeline.Bind(visible);
            for (const auto [route, cull] : VisibilityRoutes) {
                SetCullMode(visible, cull);
                DrawMeshletList(visible, buffers, uint32_t(route), 0u, false, true);
            }
        };
        if (film_layer) {
            if (!rt.FilmLayer) rt.FilmLayer = std::make_unique<RenderTargets::RayTracingResourcesT::VisibilityLayer>(chain.Cmd.Ctx, main_extent);
            draw_visibility_layer(*rt.FilmLayer, main.MeshletVisibilityMotion, "RayFilmVisibility", ubo_offset);
        }
        const auto bind_opaque_layer = [&] {
            const auto &layer = *rt.OpaqueLayer;
            chain.Cmd.SetTexture(*layer.Radiance, 3);
            BindTextures(chain.Cmd, 9, {*layer.Geometry, *layer.Normal, *layer.Identity});
        };
        if (opaque_layer) {
            const auto layer_extent = main_extent.Blocks(std::max(step, 4u));
            if (!rt.OpaqueLayer) rt.OpaqueLayer = std::make_unique<RenderTargets::RayTracingResourcesT::OpaqueLayerT>(chain.Cmd.Ctx, layer_extent);
            const auto &layer = *rt.OpaqueLayer;
            const auto extent = layer.Visibility.Extent;
            const uint32_t layer_offset = ubo_offset + GpuBuffers::ViewUboStride();
            auto &layer_view = view_at(layer_offset);
            layer_view = ray_view;
            layer_view.ViewportSize = {float(extent.Width), float(extent.Height)};
            draw_visibility_layer(layer, main.MeshletVisibilityLayer, "RayOpaqueVisibility", layer_offset);
            const auto shade = chain.BeginCompute("RayOpaqueLighting", MTL::StageFragment | MTL::StageAccelerationStructure);
            encode::BindScene(shade, slots, buffers, viewport, layer_offset);
            pbr->OpaqueLayer->Bind(shade);
            viewport.Rays->Bind(chain.Cmd, hit_instances);
            BindTextures(chain.Cmd, 0, {*layer.Visibility, *targets.Screen->VisibilityImage});
            chain.Cmd.SetTexture(*layer.Depth, 20);
            bind_opaque_layer();
            const RayLightingPushConstants layer_pc{visibility_pc, clip_to_world};
            shade.Cmd->SetBytes(layer_pc, BufferIndex_PushConstants);
            shade->dispatchThreads({extent.Width, extent.Height, 1}, ThreadgroupSize::Tile8);
        }
        if (samples > 1u) {
            if (!rt.SampleBase) rt.SampleBase = mtl::CreateTexture2D(chain.Cmd.Ctx, mtl::Format::HdrColor, main_extent, MTL::TextureUsageShaderRead);
            copy_image(MTL::StageDispatch | MTL::StageFragment, targets.Screen->SceneColorImage, rt.SampleBase);
        }
        const bool glass_direct = (HasFeature(pbr_features, PbrFeature::SceneLights) && ray_view.UseSceneLightsRender && ray_view.LightCount) ||
            std::ranges::any_of(materials, [](const auto &m) {
                                      return !m.Unlit && m.Transmission.Factor > 0.f && (m.Transmission.Factor < 1.f || m.Transmission.Texture.Slot != InvalidSlot || m.MetallicFactor > 0.f || (m.EmissiveStrength > 0.f && (m.EmissiveFactor.x > 0.f || m.EmissiveFactor.y > 0.f || m.EmissiveFactor.z > 0.f)));
                                  });
        const uint32_t lobes = ray_lobes + uint32_t(ray_glass && glass_direct);
        const auto tiles = RayTileOffset(main_extent.Width, main_extent.Height);
        const auto diffuse_tiles = !ray_glass && ray_lobes == 2u ? RayTileEnd(main_extent.Width, main_extent.Height) : 0u;
        for (uint32_t sample = 0; sample < samples; ++sample) {
            if (sample) {
                copy_image(MTL::StageDispatch | MTL::StageBlit, rt.SampleBase, targets.Screen->SceneColorImage);
            }
            // Later samples reuse geometry produced at this frame's jittered projection.
            const mat4 previous_view_proj = sample ? current_view_proj : rt.PreviousViewProj;
            const auto &previous = rt.History[rt.HistoryIndex];
            const auto &next = rt.History[rt.HistoryIndex ^ 1u];
            const auto trace = chain.BeginCompute("RayDirect", MTL::StageAccelerationStructure | MTL::StageFragment | MTL::StageDispatch | MTL::StageBlit);
            encode::BindScene(trace, slots, buffers, viewport, ubo_offset);
            viewport.Rays->Bind(chain.Cmd, hit_instances);
            trace.Cmd->SetBuffer(*rt.Work, 0, 9);
            BindTextures(chain.Cmd, 0, {*targets.Screen->VisibilityImage, *targets.Screen->SceneColorImage, *rt.Radiance, *rt.Guide, *next.MaterialValidity});
            BindTextures(chain.Cmd, 10, {*rt.ReflectionWeight, *rt.TransmissionWeight});
            BindTextures(chain.Cmd, 16, {*rt.ClosureNormal, *next.Geometry, *next.Normal});
            BindTextures(chain.Cmd, 20, {*targets.Screen->VisibilityDepth, *(rt.GlassShape ? rt.GlassShape : rt.ClosureNormal)});
            // Each kernel reads the extra lobes' weights at its own slots.
            const auto bind_extra_lobes = [&](uint32_t coat, uint32_t diffuse_transmission, uint32_t sheen) {
                if (rt.CoatWeight) BindTextures(chain.Cmd, coat, {*rt.CoatWeight, *rt.CoatNormal});
                if (rt.DiffuseTransmissionWeight) chain.Cmd.SetTexture(*rt.DiffuseTransmissionWeight, diffuse_transmission);
                if (rt.SheenWeight) chain.Cmd.SetTexture(*rt.SheenWeight, sheen);
            };
            bind_extra_lobes(5, 7, 8);
            const RayLightingPushConstants ray_pc{visibility_pc, clip_to_world, sample, uint32_t(denoise), previous_view_proj, uint32_t(denoise && (sample || history_valid)), uint32_t(sample == 0 && object_motion), Inverse(previous_view_proj), 0u, uint32_t(opaque_layer), uint32_t(quincunx_glass), diffuse_tiles};
            trace.Cmd->SetBytes(ray_pc, BufferIndex_PushConstants);
            if (pbr->PreparePrimary) {
                main.RayReset.Bind(trace);
                trace->dispatchThreads({1, 1, 1}, {1, 1, 1});
                mtl::ComputeBarrier(trace);
                pbr->PreparePrimary->Bind(trace);
                trace->dispatchThreads({main_extent.Width, main_extent.Height, 1}, ThreadgroupSize::Tile8);
                mtl::ComputeBarrier(trace);
                main.RayDispatch.Bind(trace);
                trace->dispatchThreads({1, 1, 1}, {1, 1, 1});
                mtl::ComputeBarrier(trace);
                pbr->RayDirect->Bind(trace);
                trace->dispatchThreadgroups(rt.Work.Address() + sizeof(uint32_t), {64, 1, 1});
                mtl::ComputeBarrier(trace);
                main.RayReset.Bind(trace);
                trace->dispatchThreads({1, 1, 1}, {1, 1, 1});
            } else {
                const bool filter_shadows = pbr->RayDirectFiltered && history_valid && scene_changed;
                (filter_shadows ? pbr->RayDirectFiltered : pbr->RayDirect)->Bind(trace);
                trace->dispatchThreads({main_extent.Width, main_extent.Height, 1}, ThreadgroupSize::Tile8);
            }
            if (ray_glass) {
                const auto weights = chain.BeginCompute("RayGlassWeights", MTL::StageDispatch);
                pbr->GlassWeights->Bind(weights);
                weights->dispatchThreads({main_extent.Width, main_extent.Height, 1}, ThreadgroupSize::Tile8);
            }
            const auto classify = chain.BeginCompute("RayClassify", MTL::StageDispatch);
            main.RayClassify.Bind(classify);
            classify.Cmd->SetTexture(*rt.Guide, 1);
            classify.Cmd->SetTexture(*targets.Screen->VisibilityDepth, 3);
            classify.Cmd->SetBuffer(*rt.Work, 0, 9);
            BindTextures(chain.Cmd, 4, {*next.Radiance, *next.Moments, *next.Transmission, *next.TransmissionMoments});
            classify->dispatchThreads({main_extent.Width, main_extent.Height, 1}, ThreadgroupSize::Tile8);
            mtl::ComputeBarrier(classify);
            main.RayDispatch.Bind(classify);
            classify->dispatchThreads({1, 1, 1}, {1, 1, 1});
            const auto indirect = chain.BeginCompute(pbr->Special ? "RaySpecial" : "RayGenerate", MTL::StageDispatch);
            if (opaque_layer) bind_opaque_layer();
            BindTextures(chain.Cmd, 4, {*rt.HitGuide, *rt.RayData, *rt.Incident, *rt.Transmission, *rt.TransmissionHit});
            BindTextures(chain.Cmd, 12, {*rt.TransmissionRayData, *rt.TransmissionIncident, *rt.GlassDirect});
            BindTextures(chain.Cmd, 16, {*rt.Feedback, *previous.Geometry, *previous.Normal, *previous.MaterialValidity});
            if (pbr->Special) {
                pbr->Special->Bind(indirect);
                indirect->dispatchThreadgroups(rt.Work.Address() + sizeof(uint32_t), {64, 1, 1});
            }
            indirect.Cmd->SetBuffer(*rt.RayQueue, 0, 11);
            const uint32_t staged_samples = denoise ? 1u : ray_view.RaySamples;
            for (uint32_t staged_sample = 0; staged_sample < staged_samples; ++staged_sample) {
                auto staged_pc = ray_pc;
                staged_pc.StagedSample = staged_sample;
                const auto generate = staged_sample == 0u && !pbr->Special ? indirect : chain.BeginCompute("RayGenerate", MTL::StageDispatch);
                generate.Cmd->SetBytes(staged_pc, BufferIndex_PushConstants);
                // Generation reads the direct pass's closure, with the diffuse ray's weight at the radiance image's slot, which shading binds back.
                BindTextures(chain.Cmd, 2, {*rt.TransmissionWeight, *rt.ReflectionWeight});
                chain.Cmd.SetTexture(*rt.ClosureNormal, 21);
                bind_extra_lobes(9, 11, 20);
                pbr->Generate->Bind(generate);
                generate->dispatchThreadgroups(rt.Work.Address() + sizeof(uint32_t), {64, 1, 1});
                const auto shade = chain.BeginCompute("RayShade", MTL::StageDispatch);
                chain.Cmd.SetTexture(*rt.Radiance, 2);
                if (opaque_layer) bind_opaque_layer();
                if (pattern_targets) {
                    shade.Cmd->SetBuffer(**pattern_targets->Hits, 0, BufferIndex_PatternHits);
                    shade.Cmd->SetBuffer(*rt.RayQueue, 0, BufferIndex_PatternQueue);
                }
                (opaque_layer ? pbr->ShadeOpaque : pbr->Shade)->Bind(shade);
                shade->dispatchThreadgroups(rt.Work.Address() + sizeof(uint32_t), {64, ray_lobes, 1});
                // The samples whose first surface computes inputs wait for the hit driver to write its words, and the resuming kernel shades them.
                if (pattern_targets) {
                    const auto &passes = *patterns.Pipelines;
                    const auto hits_arguments = pattern_targets->Hits->Address() + sizeof(uint32_t);
                    mtl::ComputeBarrier(shade);
                    passes.HitArguments.Bind(shade);
                    shade->dispatchThreads({1, 1, 1}, {1, 1, 1});
                    mtl::ComputeBarrier(shade);
                    mtl::Bind(shade, passes.HitDriver, BufferIndex_PatternPrograms);
                    shade->dispatchThreadgroups(hits_arguments, {64, 1, 1});
                    mtl::ComputeBarrier(shade);
                    (opaque_layer ? pbr->ShadeOpaqueResume : pbr->ShadeResume)->Bind(shade);
                    shade->dispatchThreadgroups(hits_arguments, {64, 1, 1});
                }
            }
            if (sample == 0 && history_valid && object_motion && viewport.Rays->HasDeformationHistory()) {
                // The tracing queue is dead until the next sample starts generating rays.
                const auto deformation = chain.BeginCompute("RayDeformation", MTL::StageDispatch);
                main.RayDeformation.Bind(deformation);
                deformation.Cmd->SetBytes(visibility_pc, 1);
                viewport.Rays->BindPreviousPositions(chain.Cmd);
                deformation->dispatchThreads({main_extent.Width, main_extent.Height, 1}, ThreadgroupSize::Tile8);
            }
            rt.HistoryIndex ^= 1u;
            for (uint32_t lobe = 0; lobe < lobes; ++lobe) {
                const auto &radiance = lobe ? rt.Transmission : rt.Radiance;
                const auto &filtered = lobe ? next.Transmission : next.Radiance;
                const auto &moments = lobe ? next.TransmissionMoments : next.Moments;
                const RayReconstructPushConstants reconstruction{
                    .ClipToWorld = clip_to_world,
                    .Denoise = uint32_t(denoise && lobe < 2u),
                    .Step = step,
                    .Frame = ray_view.RayFrame + sample,
                    .HistoryValid = ray_pc.HistoryValid,
                    .Moving = uint32_t(sample == 0),
                    .PreviousViewProj = previous_view_proj,
                    .Visibility = visibility_pc,
                    .Lobe = lobe,
                    .ApplyMotion = ray_pc.ApplyMotion,
                    .PreviousClipToWorld = ray_pc.PreviousClipToWorld,
                    .FastGI = ray_view.RayFastGI,
                    .Reflections = ray_view.RayReflections,
                    .MaxRoughness = ray_view.RayMaxRoughness,
                    .QuincunxGlass = uint32_t(quincunx_glass),
                    .SplitTiles = uint32_t(diffuse_tiles != 0u),
                };
                const auto compute = chain.BeginCompute("RayReconstruct", MTL::StageDispatch);
                const bool diffuse_list = lobe == 1u && diffuse_tiles != 0u;
                const auto dispatch = rt.Work.Address() + (diffuse_list ? diffuse_tiles : 5u) * sizeof(uint32_t);
                compute.Cmd->SetBuffer(*rt.Work, (diffuse_list ? diffuse_tiles + 3u : tiles) * sizeof(uint32_t), 9);
                pbr->Reconstruct->Bind(compute);
                BindTextures(chain.Cmd, 0, {*(lobe == 2u ? rt.GlassDirect : radiance), *rt.Guide, *targets.Screen->VisibilityDepth, *rt.Spatial});
                BindTextures(chain.Cmd, 4, {*(lobe ? rt.TransmissionHit : rt.HitGuide), *(lobe ? rt.TransmissionRayData : rt.RayData), *(lobe ? rt.TransmissionIncident : rt.Incident)});
                BindTextures(chain.Cmd, 7, {*rt.Radiance, *rt.SpatialInfo, *targets.Screen->VisibilityImage, *(lobe ? rt.TransmissionWeight : rt.ReflectionWeight), *rt.ClosureNormal, *(rt.GlassShape ? rt.GlassShape : rt.ClosureNormal)});
                bind_extra_lobes(13, 16, 17);
                compute.Cmd->SetBytes(reconstruction, 1);
                compute->dispatchThreadgroups(dispatch, ThreadgroupSize::Tile8);
                if (lobe < 2u) {
                    const auto temporal = chain.BeginCompute("RayTemporal", MTL::StageDispatch);
                    main.RayTemporal.Bind(temporal);
                    chain.Cmd.SetTexture(*rt.Spatial, 0);
                    BindTextures(chain.Cmd, 3, {*rt.SpatialInfo, *(lobe ? previous.Transmission : previous.Radiance), *previous.MaterialValidity, *(lobe ? previous.TransmissionMoments : previous.Moments), *filtered});
                    chain.Cmd.SetTexture(*moments, 9);
                    BindTextures(chain.Cmd, 11, {*next.Geometry, *previous.Geometry, *previous.Normal});
                    temporal->dispatchThreadgroups(dispatch, ThreadgroupSize::Tile8);
                }
                {
                    const auto bilateral = chain.BeginCompute("RayBilateral", MTL::StageDispatch);
                    main.RayBilateral.Bind(bilateral);
                    bilateral.Cmd->SetTexture(*(lobe == 2u ? rt.Spatial : filtered), 0);
                    bilateral.Cmd->SetTexture(*moments, 6);
                    bilateral.Cmd->SetTexture(*targets.Screen->SceneColorImage, 3);
                    bilateral->dispatchThreadgroups(dispatch, ThreadgroupSize::Tile8);
                }
            }
            if (ray_view.RayFastGI) {
                const auto gi_step = uint32_t(settings.RayTracing.FastGiResolution);
                const auto gi_extent = main_extent.Blocks(gi_step);
                if (!rt.FastGI || rt.FastGI->Radiance.Extent != gi_extent)
                    rt.FastGI = std::make_unique<RenderTargets::RayTracingResourcesT::FastGiT>(chain.Cmd.Ctx, main_extent, gi_step);
                const mat4 world_to_view = viewport.FrameView.Camera.View();
                const mat4 projection = current_view_proj * Inverse(world_to_view);
                const float radius = Length(vec2{2.f / (projection[0].x * main_extent.Width), 2.f / (projection[1].y * main_extent.Height)});
                const FastGiPushConstants gi_pc{clip_to_world, previous_view_proj, ray_pc.PreviousClipToWorld, projection, Inverse(projection), world_to_view, gi_step, ray_view.RayFrame + sample, ray_pc.HistoryValid, ray_pc.ApplyMotion, radius};
                RecordFastGi(chain, main, rt, *targets.Screen, previous, next, gi_pc);
            } else rt.FastGI.reset();
            if (denoise) copy_image(MTL::StageDispatch, targets.Screen->SceneColorImage, rt.Feedback);
            // Secondary hits reuse the unobscured background above. Composite the camera layers
            // at the same jittered pose and depth as the background before filtering the film.
            if (usage.Coverage) {
                const std::array colors{mtl::LoadColor(*targets.Screen->SceneColorImage)};
                const auto pass = mtl::MakePassDescriptor(colors, mtl::LoadDepth(*targets.Screen->VisibilityDepth));
                configure_transparency(pass);
                encoder = encode::BeginScenePass(chain, pass.get(), "RayTransparency", {{MTL::StageDispatch | MTL::StageBlit | MTL::StageFragment | MTL::StageAccelerationStructure, MTL::StageMesh | MTL::StageFragment}}, main_extent, slots, buffers, viewport, ubo_offset);
                viewport.Rays->Bind(chain.Cmd, hit_instances);
                draw_transparency();
            }
            if (denoise) {
                const auto accumulate = chain.BeginCompute("RayAccumulate", MTL::StageDispatch | MTL::StageBlit | MTL::StageFragment);
                const bool reproject_film = sample == 0 && history_valid && scene_changed;
                (reproject_film ? (film_layer ? main.RayFilmReprojectLayer : main.RayFilmReproject) : main.RayAccumulate).Bind(accumulate);
                BindTextures(chain.Cmd, 0, {*targets.Screen->SceneColorImage, *previous.Film, *rt.Spatial, *next.Film});
                const auto film = RayFilmFilter(film_jitter, rt.FilmWeight);
                if (reproject_film) {
                    if (film_layer) {
                        encode::BindScene(accumulate, slots, buffers, viewport, ubo_offset);
                        BindTextures(chain.Cmd, 11, {*rt.FilmLayer->Visibility, *rt.FilmLayer->Depth});
                        if (viewport.Rays->HasDeformationHistory()) viewport.Rays->BindPreviousPositions(chain.Cmd);
                        else accumulate.Cmd->SetBuffer(*rt.RayQueue, 0, 12);
                    }
                    accumulate.Cmd->SetBytes(RayFilmReprojectPushConstants{film, camera_view_proj, rt.PreviousCameraProj, rt.PreviousViewProj, clip_to_world, Inverse(rt.PreviousViewProj), uint32_t(object_motion), visibility_pc, uint32_t(viewport.Rays->HasDeformationHistory()), uint32_t(ray_glass || ray_blended)}, 1);
                    BindTextures(chain.Cmd, 4, {*next.Geometry, *previous.Geometry, *next.Normal, *previous.Normal, *targets.Screen->VisibilityDepth, *next.MaterialValidity, *previous.MaterialValidity});
                    accumulate.Cmd->SetTexture(*rt.Guide, 13);
                } else accumulate.Cmd->SetBytes(film, 1);
                rt.FilmWeight += film.Weight;
                accumulate->dispatchThreads({main_extent.Width, main_extent.Height, 1}, ThreadgroupSize::Tile8);
                copy_image(MTL::StageDispatch, rt.Spatial, targets.Screen->SceneColorImage);
            }
        }
        rt.PreviousViewProj = current_view_proj;
        rt.PreviousCameraProj = camera_view_proj;
        if (denoise) rt.Frames += samples;
    }

    if (ray_tracing && draw_scene) viewport.Rays->CapturePositions(chain, buffers);

    if (ubo_offset != canonical_ubo_offset) {
        ubo_offset = canonical_ubo_offset;
        if (film_jitter != vec2{}) {
            if (cull_scene_meshlets) record_cull(InvalidSlot);
            if (need_visibility || overlay_scene_depth) RecordMeshletVisibilityPass(chain, slots, pipelines, viewport, buffers, real_transmission, ubo_offset, {});
            if (has_silhouette) RecordSilhouetteDepthPass(chain, slots, pipelines, viewport, buffers, ubo_offset);
            targets.Screen->DepthPyramidValid = false;
        }
    }

    if (!draw_overlays) {
        if (ray_tracing && targets.RayTracing) targets.RayTracing->Frames = 0;
        const std::array colors{
            phase == RenderPhase::BlurAccumulateFirst ? mtl::ClearColor(*targets.MotionBlur->OutputImage) : mtl::LoadColor(*targets.MotionBlur->OutputImage)
        };
        const auto pass = mtl::MakePassDescriptor(colors);
        encoder = encode::BeginScenePass(chain, pass.get(), "BlurAccumulate", {{MTL::StageFragment | MTL::StageDispatch | MTL::StageBlit, MTL::StageFragment}}, main_extent, slots, buffers, viewport, ubo_offset);
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
            const auto blit = chain.BeginBlit(MTL::StageDispatch);
            blit->fillBuffer(*buffers.WireCoverageBuffer, NS::Range::Make(0, buffers.WireCoverageBuffer.UsedSize), 0);
        }
        // Canonical meshlet edge owners accumulate with atomics, so threadgroups need no ordering.
        const auto wire = chain.BeginCompute("WireRaster", MTL::StageBlit | MTL::StageFragment);
        encode::BindCompute(wire, pipelines.WireRaster, slots, buffers, viewport, ubo_offset);
        wire.Cmd->SetTexture(*targets.Screen->VisibilityDepth, 0u);
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
            wire->dispatchThreadgroups((*buffers.SceneCull.DispatchArgs)->gpuAddress() + args_offset, MTL::Size(160, 1, 1));
        }
    }
    if (overlay_jobs) RecordOverlayJobCull(chain, slots, pipelines, buffers, viewport, false, ubo_offset);
    // Hidden edit edges and vertex overlays cull only when their render pass tests against that same scene depth.
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
    const bool overlay_pass_needed = has_silhouette ||
        (show_overlays && settings.Overlays.Grid) ||
        meshlet_edit_overlay_drawn || element_overlay_meshlets > 0u || vertex_overlays || wire_meshlets ||
        overlay_jobs ||
        normal_meshlets > 0u || bone_meshlets > 0u || draw_origins;
    if (overlay_pass_needed) {
        const std::array overlay_colors{
            has_silhouette ? mtl::DiscardColor(*targets.Screen->OverlayColorImage) : mtl::ClearColor(*targets.Screen->OverlayColorImage),
        };
        const auto overlay_depth = has_silhouette ? mtl::DepthAttachment{*targets.Screen->ScratchDepth, MTL::LoadActionDontCare, MTL::StoreActionDontCare} :
            overlays_through                      ? mtl::ClearDepth(*targets.Screen->ScratchDepth) :
                                                    mtl::LoadDepth(*targets.Screen->VisibilityDepth);
        const auto overlay_pass = mtl::MakePassDescriptor(overlay_colors, overlay_depth);
        encoder = encode::BeginScenePass(chain, overlay_pass.get(), "OverlayPass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, viewport, ubo_offset);

        if (has_silhouette) {
            main.SilhouetteEdgeColor.Bind(encoder);
            const auto active_entity = FindActiveEntity(r, SceneOf(r, viewport_entity));
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
        if (bone_meshlets > 0u) {
            const std::array bone_colors{
                mtl::LoadColor(*targets.Screen->OverlayColorImage),
            };
            const auto bone_pass = mtl::MakePassDescriptor(bone_colors, {*targets.Screen->ScratchDepth, MTL::LoadActionClear, MTL::StoreActionDontCare});
            encoder = encode::BeginScenePass(chain, bone_pass.get(), "BoneXRay", {{MTL::StageDispatch, MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, viewport, ubo_offset);

            const auto draw_bones = [&](const mtl::RenderPipeline &pipeline, MeshletInstanceFlag flag, uint32_t threads, float depth_bias = 0.f) {
                if (buffers.FlagWork(uint32_t(flag)).Meshlets == 0u) return;
                pipeline.Bind(encoder);
                encoder->setDepthBias(depth_bias, 0.f, 0.f);
                DrawMeshlets(encoder, buffers, uint32_t(MeshletRoute::Overlay), uint32_t(flag), threads);
                if (depth_bias != 0.f) encoder->setDepthBias(0.f, 0.f, 0.f);
            };

            const bool object_wireframe = is_wireframe_mode && interaction_mode == InteractionMode::Object;
            if (!object_wireframe) {
                draw_bones(main.BoneFillMesh, MeshletInstanceFlag::Bone, 24u, 2.f);
                draw_bones(main.BoneSphereFillMesh, MeshletInstanceFlag::BoneJoint, uint32_t(OverlayDispatch::BoneSphereVertices));
            }
            const bool hide_bone_outlines = !is_wireframe_mode && interaction_mode == InteractionMode::Object &&
                (!show_overlays || !settings.Overlays.OutlineSelected);
            if (!hide_bone_outlines) {
                draw_bones(main.BoneWireMesh, is_wireframe_mode ? MeshletInstanceFlag::Bone : MeshletInstanceFlag::BoneWire, 24u);
                draw_bones(main.BoneSphereWireMesh, is_wireframe_mode ? MeshletInstanceFlag::BoneJoint : MeshletInstanceFlag::BoneJointWire, 64u);
            }
        }
        if (draw_origins) {
            const float scale = float(main_extent.Width) / float(std::max(logical_extent.x, 1u));
            main.ObjectOrigins.Bind(encoder);
            encode::SetPushConstants(encoder, ObjectOriginPushConstants{
                                                  .TransformSlot = buffers.Instances.TransformBuffer.Slot,
                                                  .InstanceMapSlot = buffers.GpuInstanceSlots.Slot,
                                                  .RadiusPx = 4.f * scale,
                                                  .OutlinePx = scale,
                                              });
            encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4), NS::UInteger(buffers.GpuInstanceSlots.Count<uint32_t>()));
        }
    }

    if (phase == RenderPhase::BlurFast) {
        // Refraction and overlays have finished reading opaque visibility.
        if ((real_transmission || (ray_tracing && usage.Coverage)) && meshlet_fill) {
            const std::array colors{mtl::LoadColor(*targets.Screen->VisibilityImage)};
            const auto pass = mtl::MakePassDescriptor(colors, mtl::LoadDepth(*targets.Screen->VisibilityDepth));
            const auto visibility = encode::BeginScenePass(chain, pass.get(), "BlurTransmissionVisibility", {{MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, viewport, ubo_offset);
            main.MeshletVisibilityCoverage.Bind(visibility);
            SetCullMode(visibility, MTL::CullModeNone);
            DrawMeshletList(visibility, buffers, uint32_t(real_transmission ? MeshletRoute::Transmission : MeshletRoute::Blend), 0u, false, true);
        }
        RecordMotionBlurPostFx(r, viewport, chain, ubo_offset);
    }

    {
        const std::array colors{mtl::ClearColor(*targets.Screen->FinalColorImage, {0, 0, 0, 1})};
        const auto pass = mtl::MakePassDescriptor(colors);
        encoder = encode::BeginScenePass(chain, pass.get(), "Composite", {{MTL::StageFragment | MTL::StageDispatch | MTL::StageBlit, MTL::StageFragment}}, targets.Screen->FinalColorImage.Extent, slots, buffers, viewport, ubo_offset);
        main.ViewportComposite.Bind(encoder);
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
    const auto encoder = chain.BeginCompute("OverlayJobCull", MTL::StageDispatch | MTL::StageFragment);
    encode::BindScene(encoder, slots, buffers, viewport, ubo_offset);
    encode::SetPushConstants(encoder, pc);
    const auto blocks = MTL::Size(
        (job_count + GpuBuffers::OverlayJobBlockSize - 1u) / GpuBuffers::OverlayJobBlockSize, 1, 1
    );
    pipelines.OverlayJobBlockCount.Bind(encoder);
    encoder->dispatchThreadgroups(blocks, ThreadgroupSize::Linear256);
    mtl::ComputeBarrier(encoder);
    pipelines.OverlayJobPrefix.Bind(encoder);
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256);
    mtl::ComputeBarrier(encoder);
    pipelines.OverlayJobEmit.Bind(encoder);
    encoder->dispatchThreadgroups(blocks, ThreadgroupSize::Linear256);
}

void DrawOverlayJobs(
    mtl::RenderEncoder encoder, const GpuBuffers &buffers, const MeshStore &meshes
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
    encoder->drawMeshThreadgroups((*buffers.OverlayJobDispatchArgs)->gpuAddress(), MTL::Size(1, 1, 1), MTL::Size(uint32_t(OverlayDispatch::LineGroupLines) * 2u, 1, 1));
}

void RecordMeshletVisibilityPass(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines, const Viewport &viewport,
    GpuBuffers &buffers, bool transmission, uint32_t ubo_offset, std::optional<PixelRect> scissor
) {
    const auto &main = pipelines.Main;
    const auto &targets = viewport.Targets;
    const std::array colors{
        mtl::ClearColor(*targets.Screen->VisibilityImage, MTL::ClearColor{double(UINT32_MAX), 0, 0, 0})
    };
    const auto pass = mtl::MakePassDescriptor(colors, mtl::ClearDepth(*targets.Screen->VisibilityDepth));
    const auto encoder = encode::BeginScenePass(
        chain, pass.get(), "MeshletVisibility", {{MTL::StageDispatch, MTL::StageMesh}},
        targets.Screen->VisibilityImage.Extent, slots, buffers, viewport, ubo_offset
    );
    if (scissor) encoder->setScissorRect({scissor->Origin.x, scissor->Origin.y, scissor->Extent.x, scissor->Extent.y});
    if (buffers.MeshletInstanceCount > 0) DrawVisibilityMeshlets(encoder, buffers, main, transmission);
    buffers.VisibilityGeneration = buffers.MeshletVisibleGeneration;
}

void RecordMeshletCull(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines,
    GpuBuffers &buffers, const Viewport &viewport, MeshletCullConfig config
) {
    auto &output = config.EditOutput ? buffers.EditCull : buffers.SceneCull;
    if (!config.EditOutput) ++buffers.MeshletVisibleGeneration;
    // The cull reads the instance bounds and posed geometry that earlier dispatches in the command write.
    const auto encoder = chain.BeginCompute("MeshletCull", MTL::StageDispatch | MTL::StageMesh | MTL::StageFragment);
    const bool transmission = config.Mode == MeshletRouteMode::Transmission;
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
        const auto dispatch_level = [&](const mtl::ComputePipeline &pipeline) {
            if (level == 0u && pc.WorkBlockCount == 0u) return;
            pipeline.Bind(encoder);
            SetThreadgroupMemoryLength(encoder, AlignedThreadgroupBytes(2u * prefix_stride * sizeof(uint32_t)), 0);
            if (level == 0u) encoder->dispatchThreadgroups(MTL::Size(pc.WorkBlockCount, 1, 1), MTL::Size(GpuBuffers::MeshletCullBlockSize, 1, 1));
            else encoder->dispatchThreadgroups((*buffers.LodExpandArgs)->gpuAddress() + index * sizeof(MeshDispatchArgs), MTL::Size(GpuBuffers::MeshletCullBlockSize, 1, 1));
        };
        dispatch_level(pipelines.LodFrontierCount);
        mtl::ComputeBarrier(encoder);
        pipelines.LodFrontierPrefix.Bind(encoder);
        encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256);
        mtl::ComputeBarrier(encoder);
        dispatch_level(pipelines.LodFrontierEmit);
        mtl::ComputeBarrier(encoder);
    }
    encode::SetPushConstants(encoder, pc);
    RecordMeshletCompaction(encoder, pipelines, buffers);
}

void RecordSilhouetteDepthPass(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines, const Viewport &viewport,
    GpuBuffers &buffers, uint32_t ubo_offset
) {
    const auto &screen = *viewport.Targets.Screen;
    const auto &samplers = viewport.Samplers;
    auto decode_pc = encode::VisibilityDecodePc(buffers);
    const auto compute = chain.BeginCompute("SilhouetteCull", MTL::StageFragment);
    const auto &occluders = screen.OutlineOccluderPyramid;
    encode::BindCompute(compute, pipelines.OutlineOccluderSeed, slots, buffers, viewport, ubo_offset);
    compute.Cmd->SetTexture(*screen.VisibilityImage, 0u);
    compute.Cmd->SetTexture(*screen.VisibilityDepth, 1u);
    compute.Cmd->SetTexture(*occluders.Mips[0].View, 2u);
    encode::SetPushConstants(compute, decode_pc);
    const auto blocks = occluders.Mips[0].Extent;
    compute->dispatchThreadgroups(MTL::Size((blocks.Width + 15u) / 16u, (blocks.Height + 15u) / 16u, 1u), ThreadgroupSize::Tile16);
    mtl::ComputeBarrier(compute);
    RecordDepthPyramid(compute, slots, buffers, viewport, pipelines, occluders, samplers.OutlineOccluderPyramid, 1u, InvalidSlot, {}, true, ubo_offset);
    mtl::ComputeBarrier(compute);

    auto &output = buffers.SilhouetteCull;
    output.Visible.SetCount<VisibleMeshlet>(buffers.SceneCull.Visible.Count<VisibleMeshlet>());
    output.ChunkCount = buffers.SceneCull.ChunkCount;
    output.DispatchArgs.SetCount<MeshDispatchArgs>(GpuBuffers::MeshletRouteCount * output.ChunkCount);
    auto cull_pc = MakeMeshletCullSlotsPc(buffers, viewport, output);
    cull_pc.PyramidSamplerSlot = samplers.OutlineOccluderPyramid;
    cull_pc.SourceVisibleSlot = buffers.SceneCull.Visible.Slot;
    cull_pc.SourceRouteStateSlot = buffers.SceneCull.Routes.Slot;
    encode::SetPushConstants(compute, cull_pc);
    pipelines.SilhouetteCullSize.Bind(compute);
    compute->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    mtl::ComputeBarrier(compute);
    RecordMeshletCompaction(compute, pipelines, buffers);

    const std::array colors{mtl::ClearColor(*screen.SilhouetteImage)};
    const auto pass = mtl::MakePassDescriptor(colors, {*screen.ScratchDepth, MTL::LoadActionClear, MTL::StoreActionDontCare});
    const auto encoder = encode::BeginScenePass(
        chain, pass.get(), "SilhouetteDepth", {{MTL::StageDispatch, MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment, MTL::StageFragment}},
        screen.SilhouetteImage.Extent, slots, buffers, viewport, ubo_offset
    );
    pipelines.SilhouetteSeed.Bind(encoder);
    encoder.Cmd->SetTexture(*screen.VisibilityImage, 0u);
    encoder.Cmd->SetTexture(*screen.VisibilityDepth, 1u);
    encode::SetFragmentPushConstants(encoder, decode_pc);
    encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
    pipelines.Silhouette.Bind(encoder);
    decode_pc.VisibleMeshletSlot = buffers.SilhouetteCull.Visible.Slot;
    encode::SetFragmentPushConstants(encoder, decode_pc);
    const auto draw_route = [&](MeshletRoute route, MTL::CullMode cull) {
        SetCullMode(encoder, cull);
        DrawMeshletList(encoder, buffers, uint32_t(route), 0u, false, false, InvalidSlot, 160u, 0u, &buffers.SilhouetteCull);
    };
    for (const auto [route, cull] : VisibilityRoutes) draw_route(route, cull);
    draw_route(MeshletRoute::Blend, MTL::CullModeNone);
    draw_route(MeshletRoute::Transmission, MTL::CullModeNone);
}

void DrawMeshlets(
    mtl::RenderEncoder encoder, const GpuBuffers &buffers, uint32_t route,
    uint32_t required_instance_flags, uint32_t mesh_threads, uint32_t edit_edge_corner
) {
    DrawMeshletList(
        encoder, buffers,
        route, required_instance_flags, false, false, InvalidSlot,
        mesh_threads, edit_edge_corner
    );
}

void DrawVertexBlocks(
    mtl::RenderEncoder encoder, const state::Registry &r, state::Entity mesh_entity, uint32_t instance, bool sound_points,
    uint32_t pyramid_slot, float min_diameter_pixels
) {
    const auto &buffers = r.Resources.get<const GpuBuffers>();
    const auto &meshes = r.Resources.get<const MeshStore>();
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

void RecordRenderCommandBuffer(state::Registry &r, state::Entity viewport, mtl::Command &command, RenderPhase phase) {
    const mtl::AutoreleaseScope native_scope;
    profile::BeginRecording();
    mtl::PassChain chain{command, profile::RecordingTimer()};
    RecordPhase(r, viewport, chain, GatherPrelude(r, phase == RenderPhase::BlurFast), phase, 0);
    profile::EndRecording();
}

void RecordBlurStepsCommandBuffer(state::Registry &r, state::Entity viewport, mtl::Command &command, std::span<const uint32_t> sample_weights) {
    const mtl::AutoreleaseScope native_scope;
    profile::BeginRecording();
    mtl::PassChain chain{command, profile::RecordingTimer()};
    const auto prelude = GatherPrelude(r, true);
    for (uint32_t i = 0; i < sample_weights.size(); ++i) {
        RecordPhase(r, viewport, chain, prelude, i == 0 ? RenderPhase::BlurAccumulateFirst : RenderPhase::BlurAccumulate, GpuBuffers::SceneViewUboOffset(i + 1), sample_weights[i]);
    }
    RecordPhase(r, viewport, chain, prelude, RenderPhase::BlurResolve, 0);
    profile::EndRecording();
}

void ReserveFrameScratch(state::Registry &r) {
    auto &buffers = r.Resources.get<GpuBuffers>();
    const auto &scene_state = r.Resources.get<const GpuSceneState>();
    uint64_t coverage = 0u;
    bool transmission = false;
    for (const auto [viewport, live] : r.view<const Viewport>().each()) {
        if (!ViewportImageReady(live)) continue;
        if (r.get<const ViewportDisplay>(viewport).RayTracing.Enabled)
            r.edit<Viewport>(viewport).SceneViewUBO.SetUsedSize(GpuBuffers::RayViewUboBytes());
        const auto extent = live.Targets.Screen->SceneColorImage.Extent;
        coverage = std::max<uint64_t>(coverage, uint64_t(extent.Width) * extent.Height);
        transmission |= RealTransmission(r, viewport, r.get<const ViewportDisplay>(viewport), scene_state.MeshPbrFeatures);
    }
    buffers.WireCoverageBuffer.Reserve(coverage * sizeof(uint32_t));
    const GpuBuffers::MeshletFlagWork all{buffers.LodNodeCount, buffers.MeshletInstanceCount};
    const auto visible = all.Meshlets * (1u + transmission);
    buffers.EnsureMeshletVisibilityCapacity(buffers.SceneCull, visible, all.Nodes, all.Meshlets);
    buffers.EnsureMeshletVisibilityCapacity(buffers.SilhouetteCull, visible, all.Nodes, all.Meshlets);
    const auto &edit = buffers.FlagWork(uint32_t(MeshletInstanceFlag::EditOverlay));
    buffers.EnsureMeshletVisibilityCapacity(buffers.EditCull, edit.Meshlets, edit.Nodes, edit.Meshlets);
}

SceneViewUBO SceneViewOf(state::Registry &r, state::Entity viewport_entity, const RenderView &render_view) {
    const auto &buffers = r.Resources.get<const GpuBuffers>();
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &render = meshes.Render();
    const auto &environments = r.Resources.get<const EnvironmentStore>();
    const auto &viewport = r.get<const Viewport>(viewport_entity);
    const auto &settings = render_view.Display;
    const auto look_through = LookThroughCamera(r, viewport_entity);
    const auto &targets = viewport.Targets;
    const auto interaction_mode = r.get<const Interaction>(state::Document).Mode;
    const bool is_pbr_mode = !WorkbenchShading(settings.ViewportShading);
    const auto &active_lighting = ActivePbrLighting(r, viewport_entity, settings);
    const bool use_scene_lights = is_pbr_mode && active_lighting.UseSceneLights;
    const bool use_scene_world = is_pbr_mode && active_lighting.UseSceneWorld;
    const auto ibl = use_scene_world ? environments.SceneWorld.Ibl : StudioIbl(r, r.get<const StudioEnvironment>(viewport_entity).Name);
    const auto scene = SceneOf(r, viewport_entity);
    const auto *image_light = r.try_get<const ImageLight>(scene);
    const float env_intensity = use_scene_world && image_light ? image_light->Intensity : active_lighting.EnvIntensity;
    const mat3 env_rotation = [&]() -> mat3 {
        if (use_scene_world) return image_light ? ToMat3(image_light->Rotation) : mat3{1.f};
        const float radians = active_lighting.EnvRotationDegrees * (Pi / 180.f);
        const float s = std::sin(radians), c = std::cos(radians);
        return {c, 0, -s, 0, 1, 0, s, 0, c};
    }();
    const auto *pending = r.try_get<const PendingTransform>(state::Document);
    const auto &mesh_slots = meshes.Slots();
    const auto scene_index = ShownSceneIndex(r, scene);
    const auto &patterns = r.Resources.get<const MaterialPatterns>();
    SceneViewUBO view{
        .RayMaxRoughness = settings.RayTracing.MaxRoughness,
        .RaySamples = std::clamp(uint32_t(settings.RayTracing.Samples), 1u, 64u),
        .RayStep = uint32_t(settings.RayTracing.Resolution),
        .RayFastGI = uint32_t(settings.RayTracing.Diffuse == DiffuseLighting::FastGI),
        .RayGiDistance = settings.RayTracing.FastGiDistance,
        .RayGiThickness = settings.RayTracing.FastGiThickness,
        .RayReflections = settings.RayTracing.Reflections,
        .RayTransmissionShadows = settings.RayTracing.TransmissionShadows,
        .LightCount = buffers.Lights[scene_index].Count<LightRecord>(),
        .LightSlot = buffers.Lights[scene_index].Slot,
        .UseSceneLightsRender = use_scene_lights ? 1u : 0u,
        .BackfaceCulling = is_pbr_mode                                                         ? BackfaceCullingMode::Material :
            settings.ViewportShading == ViewportShadingMode::Solid && settings.BackfaceCulling ? BackfaceCullingMode::Always :
                                                                                                 BackfaceCullingMode::None,
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
        .LodErrorPixels = settings.RayTracing.Enabled && is_pbr_mode ? 0.f : settings.LodErrorPixels,
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
        .MaterialSlot = buffers.Materials.Slot,
        .PrimitiveMaterialSlot = mesh_slots.PrimitiveMaterial,
        .MeshRecordSlot = render.MeshRecords.Buffer.Slot,
        .InstanceRecordSlot = buffers.Instances.RecordBuffer.Slot,
        .InstanceStateSlot = buffers.Instances.States[scene_index].Slot,
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
        .LodPinFlags = LodPinFlagsOf(settings),
        .RayShadowSamples = std::clamp(uint32_t(settings.RayTracing.ShadowSamples), 1u, 64u),
        .PatternSlot = patterns.Records ? patterns.Records->Slot : InvalidSlot,
    };
    render_view.ApplyTo(view);
    return view;
}

void WriteViewUniforms(state::Registry &r, state::Entity viewport_entity, const RenderView &view) {
    const auto scene_view = SceneViewOf(r, viewport_entity, view);
    auto &viewport = r.edit<Viewport>(viewport_entity);
    viewport.SceneViewUBO.Update(BytesOf(scene_view));
    viewport.WorkspaceLightsUBO.Update(BytesOf(r.get<const WorkspaceLights>(viewport_entity)));
    viewport.FrameView = view;
}

namespace {
bool PositionsOnly(const MeshStore::Change &change) { return (change.Bits & ~(MeshStore::SelectionChanged | MeshStore::EntryChanged)) == MeshStore::GeometryChanged; }

// A change to the positions alone of a record that a mesh of `deferred`, sorted, draws is left to that mesh's next refresh.
// Returns the meshes whose records the changes or a restore replaced, in order.
std::vector<state::Entity> ReconcileChanges(state::Registry &r, mtl::ComputeChain &chain, std::span<const MeshStore::Change> changes, std::vector<state::Entity> restored_meshes, std::span<const state::Entity> deferred) {
    const profile::CpuScope scope{"ReconcileChanges"};
    auto &meshes = r.Resources.get<MeshStore>();
    const auto deferred_positions = [&](const MeshStore::Change &change) { return PositionsOnly(change) && std::ranges::binary_search(deferred, MeshEntityOf(r, change.StoreId)); };
    std::vector<MeshStore::Change> reconciled;
    for (const auto &change : changes) {
        if (!deferred_positions(change)) reconciled.push_back(change);
    }
    meshes.ReconcileSelection(r, chain, reconciled);
    auto &buffers = r.Resources.get<GpuBuffers>();
    for (const auto &change : changes) {
        if (!(change.Bits & MeshStore::EntryChanged)) continue;
        RefreshMeshBinding(r, change.StoreId);
        if (!meshes.TryGet(change.StoreId)) buffers.Retally(change.StoreId, {});
        if (const auto entity = MeshEntityOf(r, change.StoreId); entity != state::Null) restored_meshes.push_back(entity);
    }
    for (const auto entity : r.view<const PosedLayout>()) r.emplace_or_replace<MeshBoundsDirty>(entity);
    auto &scene = r.Resources.get<GpuSceneState>();
    SortUnique(restored_meshes);
    for (const auto entity : restored_meshes) {
        r.remove<MeshletPositionsDirty, ClusterLodDirty>(entity);
        const auto *owner = TryRecordOf(r, entity);
        if (!owner || owner->RenderTopologies == 0u) continue;
        if (meshes.HasDirtyPositions(*owner)) r.emplace<MeshletPositionsDirty>(entity);
        if (meshes.HasDirtyGroups(*owner)) r.emplace<ClusterLodDirty>(entity);
        if (!meshes.ClusterGroupCount(*owner) && ClusterLodApplies(Mesh{meshes, owner->StoreId}.FaceCount() > 0u, meshes.MeshletCount(*owner))) r.emplace_or_replace<ClusterLodDemand>(entity);
    }
    RepointMeshInstances(r, restored_meshes);
    std::vector<MeshVertexChanges> positions;
    for (const auto &change : changes) {
        const auto entity = MeshEntityOf(r, change.StoreId);
        if (entity == state::Null) continue;
        if (!DrawsMesh(r, entity)) {
            r.emplace_or_replace<MeshDisplayDirty>(entity);
            r.emplace_or_replace<MeshLayoutDirty>(entity);
            continue;
        }
        if (change.Bits & ~MeshStore::SelectionChanged) {
            if (!deferred_positions(change)) {
                ReleaseMeshEditWork(r, entity);
                if (PositionsOnly(change) && !change.VertexRanges.empty() && !r.all_of<EvaluatedMesh>(entity)) positions.push_back({entity, change.VertexRanges});
            }
            RefreshMeshBinding(r, change.StoreId);
            r.emplace_or_replace<MeshGeometryDirty>(entity, EditSelectionAfter::Keep, true);
        }
        if (change.Bits & MeshStore::SelectionChanged) scene.EditSelectionDirty = true;
    }
    RefreshEditedPositions(r, chain, positions, false);
    return restored_meshes;
}
} // namespace

std::vector<state::Entity> ReconcileRestoredMeshes(state::Registry &r, std::vector<state::Entity> restored_meshes) {
    // An identity a restore brings back as another kind draws no mesh overlays.
    for (const auto entity : r.view<const MeshVertexOverlays>() | std::ranges::to<std::vector>()) {
        if (!DrawsMesh(r, entity)) r.remove<MeshVertexOverlays>(entity);
    }
    const auto changes = r.Resources.get<MeshStore>().TakeChanges();
    mtl::ComputeChain chain{r.Resources.get<MeshStore>().BufferContext()};
    auto restored = ReconcileChanges(r, chain, changes, std::move(restored_meshes), {});
    chain.Submit();
    for (const auto &change : changes) {
        if (const auto entity = MeshEntityOf(r, change.StoreId); entity != state::Null) restored.push_back(entity);
    }
    SortUnique(restored);
    return restored;
}

void ReconcileRestoredRecords(state::Registry &r, mtl::ComputeChain &chain, std::span<const MeshStore::Change> changes, std::span<const state::Entity> deferred) {
    (void)ReconcileChanges(r, chain, changes, {}, deferred);
}

bool RefreshesRestore(const state::Registry &r, uint32_t store_id, state::Entity mesh_entity) {
    return mesh_entity != state::Null && MeshEntityOf(r, store_id) == mesh_entity && DrawsMesh(r, mesh_entity);
}

MeshStore::Change DeferredRestore(const state::Registry &r, const MeshStore::Change &change, state::Entity mesh_entity) {
    if (!RefreshesRestore(r, change.StoreId, mesh_entity) || !(change.LiveRender || ((change.Bits & ~MeshStore::SelectionChanged) && PositionsOnly(change)))) return {};
    MeshStore::Change deferred{};
    deferred.VertexRanges = change.VertexRanges;
    deferred.Blocks = change.Blocks;
    deferred.LiveRender = change.LiveRender;
    return deferred;
}

MeshStore::Change ReconcileRestoredStore(state::Registry &r, state::Entity mesh_entity) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto changes = meshes.TakeChanges();
    mtl::ComputeChain chain{meshes.BufferContext()};
    (void)ReconcileChanges(r, chain, changes, {}, mesh_entity != state::Null ? std::span{&mesh_entity, 1u} : std::span<const state::Entity>{});
    chain.Submit();
    MeshStore::Change deferred{};
    for (const auto &change : changes) {
        const auto part = DeferredRestore(r, change, mesh_entity);
        deferred.VertexRanges.append_range(part.VertexRanges);
        for (uint32_t d = 0u; d < deferred.Blocks.size(); ++d) deferred.Blocks[d].append_range(part.Blocks[d]);
    }
    return deferred;
}

void UpdateAuthoredMorphShading(state::Registry &r, mtl::ComputeChain &chain, std::span<const state::Entity> mesh_entities) {
    auto &meshes = r.Resources.get<MeshStore>();
    auto &buffers = r.Resources.get<GpuBuffers>();
    std::vector<NormalDeriveEntry> entries;
    std::vector<uint32_t> entry_ids;
    PoseAttributeStore<vec3>::Temporary vertex_normals{buffers.PosedVertexNormals};
    PoseAttributeStore<vec3>::Temporary face_normals{buffers.PosedFaceNormals};
    PoseAttributeStore<vec3>::Temporary sectors{buffers.PosedSectors};
    for (const auto entity : mesh_entities) {
        const auto store_id = *DrawnStoreId(r, entity);
        const Mesh mesh{meshes, store_id};
        const auto &record = meshes.Get(store_id);
        const auto target_count = record.MorphTargetCount;
        if (target_count == 0 || !record.HasAuthoredNormals) continue;
        const auto entry_inputs = MakeDeriveEntryInputs(meshes, store_id);
        if (!entry_inputs) continue;
        UpdateMorphShadingAuthored(meshes, mesh, {});
        if (meshes.Get(store_id).MorphShadingAuthored) continue;
        const auto normal_blocks = SectorBlocks(meshes, store_id);
        const auto vertex_blocks = PoseElementBlocks(meshes.Arenas().Vertices, record.Vertices);
        const auto face_blocks = PoseElementBlocks(meshes.Arenas().FaceTriangles, record.FaceData);
        const auto &morph = meshes.Arenas().Morph;
        for (uint32_t t = 0; t < target_count; ++t) {
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
        }
    }
    if (entries.empty()) return;

    EncodeDeriveNormals(r, chain, entry_ids, entries, MakeNormalDerivePc(buffers, meshes));
    chain.Submit();

    for (size_t i = 0; i < entry_ids.size();) {
        const auto store_id = entry_ids[i];
        std::vector<CornerNormalSources> poses;
        for (; i < entry_ids.size() && entry_ids[i] == store_id; ++i) {
            const auto &entry = entries[i];
            poses.push_back({
                .VertexNormals = meshes.Arenas().BaseVertexNormals.Buffer.GetSpan<vec3>(),
                .FaceNormals = meshes.Arenas().BaseFaceNormals.Buffer.GetSpan<vec3>(),
                .PosedVertexNormals = buffers.PosedVertexNormals.View(entry.VertexNormalNamespace),
                .PosedSectorNormals = buffers.PosedSectors.View(entry.SectorNamespace),
                .PosedFaceNormals = buffers.PosedFaceNormals.View(entry.FaceNormalNamespace),
            });
        }
        UpdateMorphShadingAuthored(meshes, Mesh{meshes, store_id}, poses);
    }
}

void FinalizeAuthoredShading(state::Registry &r, mtl::ComputeChain &chain) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto authored = r.view<const AuthoredCornerNormals>() | std::ranges::to<std::vector>();
    for (const auto entity : authored) {
        EncodeAuthoredCornerNormals(meshes, GetMesh(r, entity), r.get<const AuthoredCornerNormals>(entity).Corners);
    }
    UpdateAuthoredMorphShading(r, chain, authored);
    for (const auto entity : authored) r.remove<AuthoredCornerNormals>(entity);
}

namespace {
// Records one pass that finishes each work's block list and its compact prefix, one threadgroup per work.
void FinalizeWork(state::Registry &r, mtl::ComputeChain &chain, std::span<const ElementWork> work) {
    if (work.empty()) return;
    const auto &pipeline = GetPipelines(r).FinalizeElementWork;
    chain.Encode([&](mtl::ComputeEncoder encoder) {
        pipeline.Bind(encoder);
        encoder.Cmd->SetBytes(std::as_bytes(work), BufferIndex_PushConstants);
        encoder->dispatchThreadgroups(MTL::Size(work.size(), 1, 1), ThreadgroupSize::Linear256);
        mtl::ComputeBarrier(encoder);
    });
}

MeshEditWork &PrepareMeshEditWork(state::Registry &r, state::Entity entity) {
    auto &buffers = r.Resources.get<GpuBuffers>();
    const auto mesh = GetMesh(r, entity);
    const auto id = mesh.GetStoreId();
    if (const auto *work = r.try_get<const MeshEditWork>(entity)) {
        if (work->StoreId == id) return r.edit<MeshEditWork>(entity);
        ReleaseMeshEditWork(r, entity);
    }
    r.emplace_or_replace<MeshDisplayDirty>(entity);
    auto work_allocation = buffers.GeometryWork.BeginAllocation();
    MeshEditWork w{.StoreId = id};
    w.Candidates = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Vertices = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Faces = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Normals = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Meshlets = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    auto &result = r.emplace<MeshEditWork>(entity, std::move(w));
    work_allocation.Commit();
    return result;
}
} // namespace

void ReleaseMeshEditWork(state::Registry &r, state::Entity entity) {
    if (!r.all_of<MeshEditWork>(entity)) return;
    r.remove<MeshEditWork>(entity);
    r.emplace_or_replace<MeshDisplayDirty>(entity);
}

void ReleaseMeshEditWorkRanges(state::Registry &r, state::Entity entity) {
    auto &buffers = r.Resources.get<GpuBuffers>();
    const auto &w = r.get<const MeshEditWork>(entity);
    for (const auto range : {w.Candidates, w.Vertices, w.Faces, w.Normals, w.Meshlets}) buffers.GeometryWork.Release(WorkStorageRange(range));
}

namespace {
using GeometryEditJobs = std::vector<std::pair<state::Entity, GeometryEditJob>>;

// Records each job's reusable topological footprint before any geometry writes.
void PrepareGeometryFootprints(state::Registry &r, mtl::ComputeChain &chain, GeometryEditJobs &jobs) {
    struct Footprint {
        MeshEditWork *Work;
        GeometryEditJob *Job;
        uint64_t Candidates{};
    };
    std::vector<Footprint> footprints;
    for (auto &[entity, job] : jobs) {
        if (auto &work = r.edit<MeshEditWork>(entity); !work.FootprintReady) footprints.push_back({&work, &job});
    }
    if (footprints.empty()) return;
    const profile::CpuScope scope{"PrepareGeometryFootprint"};
    auto &buffers = r.Resources.get<GpuBuffers>();
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetPipelines(r);
    std::vector<ElementWorkSeedJob> seeds;
    std::vector<ElementWork> seeded;
    for (auto &footprint : footprints) {
        auto &work = *footprint.Work;
        footprint.Candidates = WorkBlockCount(buffers.GeometryWork, work.Candidates);
        if (work.CandidateReady) continue;
        std::vector<uint32_t> blocks;
        meshes.GetSelectedElements(work.StoreId, Element::Vertex).ForEachBlock([&](uint32_t block, uint32_t) { blocks.push_back(block); });
        footprint.Candidates += blocks.size();
        ReserveElementWork(buffers.GeometryWork, work.Candidates, footprint.Candidates + blocks.size());
        seeds.push_back(PrepareBlockMembershipWork(chain.Scratch, meshes.Arenas().Vertices, meshes.Get(work.StoreId).Vertices, blocks, work.Candidates, meshes.GetSelectionSlot(Element::Vertex)));
        seeded.push_back(work.Candidates);
    }
    EncodeElementMembershipWork(r, chain, seeds);
    FinalizeWork(r, chain, seeded);
    const auto &a = meshes.Arenas();
    std::vector<GeometryEditJob> footprint_jobs;
    // A phase's tiles bound the occupied blocks of the work it reads: the candidates, or the faces their fans reach.
    std::array<std::vector<uint32_t>, 2> tiles;
    for (auto &footprint : footprints) {
        auto &work = *footprint.Work;
        for (const auto item : {work.Faces, work.Normals, work.Meshlets}) ClearElementWork(buffers.GeometryWork, item);
        // A table's distinct blocks are at most the blocks its domain spans: the candidates' blocks, the mesh's face and vertex sets, and its meshlet set.
        const auto &record = meshes.Get(work.StoreId);
        const uint64_t face_blocks = record.FaceData ? a.FaceTriangles.Set(record.FaceData).BlockCount : 0u;
        const uint64_t vertex_blocks = a.Vertices.Set(record.Vertices).BlockCount;
        const uint64_t meshlet_blocks = record.MeshletData ? meshes.Render().Meshlets.Set(record.MeshletData).BlockCount : 0u;
        ReserveElementWork(buffers.GeometryWork, work.Vertices, footprint.Candidates);
        ReserveElementWork(buffers.GeometryWork, work.Faces, face_blocks);
        ReserveElementWork(buffers.GeometryWork, work.Normals, vertex_blocks);
        ReserveElementWork(buffers.GeometryWork, work.Meshlets, meshlet_blocks);
        auto &job = *footprint.Job;
        job.Candidates = work.Candidates;
        job.ChangedVertices = work.Vertices;
        job.Faces = work.Faces;
        job.Normals = work.Normals;
        job.MeshletAddresses = work.Meshlets;
        work.CandidateReady = work.FootprintReady = true;
        footprint_jobs.push_back(job);
        tiles[0].push_back(uint32_t(footprint.Candidates));
        tiles[1].push_back(uint32_t(face_blocks));
    }
    for (const auto phase : {1u, 3u}) {
        mtl::GatherJobs(chain, &pipelines.GeometryEdit, std::span{footprint_jobs}, tiles[phase / 2u], [&chain, &pipelines, phase](mtl::JobTable table, auto) {
            table.Constants.Parameter = phase;
            chain.Jobs(pipelines.GeometryEdit, table, 256u);
        });
        std::vector<ElementWork> finished;
        for (const auto &job : footprint_jobs) {
            if (phase == 1u) finished.append_range(std::array{job.Faces, job.MeshletAddresses});
            else finished.append_range(std::array{job.Normals, job.MeshletAddresses});
        }
        FinalizeWork(r, chain, finished);
    }
}

GeometryEditJob PrepareGeometryEdit(state::Registry &r, state::Entity entity, std::span<const Range> changed) {
    auto &buffers = r.Resources.get<GpuBuffers>();
    auto &meshes = r.Resources.get<MeshStore>();
    auto &w = PrepareMeshEditWork(r, entity);
    const auto id = w.StoreId;
    if (changed.empty()) w.RefreshRanges.clear();
    if (!changed.empty()) {
        const bool same_ranges = w.FootprintReady && w.PreviewActive &&
            std::ranges::equal(changed, w.RefreshRanges, [](Range a, Range b) {
                                     return a.Offset == b.Offset && a.Count == b.Count;
                                 });
        if (!same_ranges) {
            w.FootprintReady = false;
            SeedElementWorkRanges(buffers.GeometryWork, w.Candidates, changed, 0u, w.PreviewActive);
            w.RefreshRanges.assign(changed.begin(), changed.end());
        }
        w.CandidateReady = true;
    } else if (!w.CandidateReady || !w.PreviewActive || r.Resources.get<const GpuSceneState>().EditSelectionDirty) {
        w.CandidateReady = w.FootprintReady = false;
        if (!w.PreviewActive) ClearElementWork(buffers.GeometryWork, w.Candidates);
    }
    ClearElementWork(buffers.GeometryWork, w.Vertices);
    return {
        .Candidates = w.Candidates,
        .ChangedVertices = w.Vertices,
        .Faces = w.Faces,
        .Normals = w.Normals,
        .MeshletAddresses = w.Meshlets,
        .Connectivity = meshes.GetConnectivityRef(id),
        .FaceTriangleStartSlot = meshes.Slots().FaceTriangleStart,
        .ElementMeshlets = {meshes.Render().ElementMeshlets[0].Ref(meshes.Get(id).ElementMeshlets[0]), meshes.Render().ElementMeshlets[1].Ref(meshes.Get(id).ElementMeshlets[1]), meshes.Render().ElementMeshlets[2].Ref(meshes.Get(id).ElementMeshlets[2])},
        .MeshletSets = meshes.Render().Meshlets.Slots(),
        .MeshletSet = meshes.Get(id).MeshletData.Index,
        .Mode = changed.empty() ? GeometryEditMode::Commit : GeometryEditMode::Refresh,
    };
}

// Records each pass of the edit once for all jobs, sized by the footprint counts a completed submit holds.
void RecordGeometryEditBatch(state::Registry &r, mtl::ComputeChain &chain, const GeometryEditJobs &commits) {
    const auto &work = r.Resources.get<const GpuBuffers>().GeometryWork;
    const auto &pipelines = GetPipelines(r);
    std::vector<GeometryEditJob> jobs;
    std::vector<uint32_t> tiles;
    std::vector<ElementWork> changed;
    std::vector<LocalNormalWork> normals;
    for (const auto &[entity, job] : commits) {
        jobs.push_back(job);
        tiles.push_back(WorkBlockCount(work, job.Candidates));
        changed.push_back(job.ChangedVertices);
        // The derive covers the changed faces and the vertices around them.
        normals.push_back({
            .StoreId = r.get<const MeshEditWork>(entity).StoreId,
            .Vertices = job.Normals,
            .VertexCount = work.Get({job.Normals.Storage.Offset + 5u, 1u})[0],
            .Faces = job.Faces,
            .FaceCount = work.Get({job.Faces.Storage.Offset + 5u, 1u})[0],
        });
    }
    mtl::GatherJobs(chain, &pipelines.GeometryEdit, std::span{jobs}, tiles, [&chain, &pipelines](mtl::JobTable table, auto) {
        table.Constants.Parameter = 4u;
        chain.Jobs(pipelines.GeometryEdit, table, 256u);
    });
    FinalizeWork(r, chain, changed);
    EncodeDeriveMeshNormals(r, chain, work, normals);
}
// Returns the repair, which reads the changed triangles the chain's next submit reports and records on the chain again, or nothing for jobs without polygons.
std::function<void()> RetessellateGeometry(state::Registry &r, mtl::ComputeChain &chain, const GeometryEditJobs &geometry) {
    const auto &work = r.Resources.get<const GpuBuffers>().GeometryWork;
    std::vector<MeshRetessellation> inputs;
    std::vector<state::Entity> entities;
    for (const auto &[entity, edit] : geometry) {
        inputs.push_back({
            .StoreId = *DrawnStoreId(r, entity),
            .Work = &work,
            .Faces = edit.Faces,
            .ChangedVertices = edit.ChangedVertices,
        });
        entities.push_back(entity);
    }
    auto changed = EncodeRetessellation(r, chain, inputs);
    if (std::ranges::none_of(changed, [](const ElementWork &triangles) { return triangles.Storage.Slot != InvalidSlot; })) return {};
    return [&r, &chain, entities = std::move(entities), changed = std::move(changed)] {
        std::vector<std::pair<state::Entity, FaceTriangles>> repairs;
        for (uint32_t i = 0u; i < changed.size(); ++i) {
            if (changed[i].Storage.Slot == InvalidSlot) continue;
            if (const auto count = ElementWorkCount(chain.Scratch, changed[i])) {
                FaceTriangles triangles;
                triangles.Triangles = changed[i];
                triangles.Count = count;
                repairs.emplace_back(entities[i], triangles);
            }
        }
        if (repairs.empty()) return;
        RepairFaceRender(r, chain, repairs);
        for (const auto &[entity, _] : repairs) r.edit<MeshEditWork>(entity).FootprintReady = false;
        chain.Submit();
    };
}

} // namespace

std::function<void()> RefreshEditedPositions(state::Registry &r, mtl::ComputeChain &chain, std::span<const MeshVertexChanges> changes, bool retessellate) {
    if (changes.empty()) return {};
    const profile::CpuScope scope{"RefreshEditedPositions"};
    GeometryEditJobs jobs;
    for (const auto &[entity, ranges] : changes) jobs.emplace_back(entity, PrepareGeometryEdit(r, entity, ranges));
    const bool footprints_pending = std::ranges::any_of(jobs, [&](const auto &job) { return !r.get<const MeshEditWork>(job.first).FootprintReady; });
    PrepareGeometryFootprints(r, chain, jobs);
    // The normal capture and the retessellation read the footprint tables on the host, so a footprint recorded now completes first.
    if (footprints_pending) chain.Submit();
    RecordGeometryEditBatch(r, chain, jobs);
    auto repair = retessellate ? RetessellateGeometry(r, chain, jobs) : std::function<void()>{};
    auto &scene = r.Resources.get<GpuSceneState>();
    scene.EditPositionsPending = true;
    for (const auto &[entity, ranges] : changes) {
        auto &work = r.edit<MeshEditWork>(entity);
        work.Modified = work.PreviewActive = true;
        r.emplace_or_replace<MeshLayoutDirty>(entity);
    }
    return repair;
}

void PublishEditedPositions(state::Registry &r, mtl::ComputeChain &chain, std::span<const state::Entity> entities) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto &buffers = r.Resources.get<GpuBuffers>();
    std::vector<state::Entity> changed;
    std::vector<MeshStore::SelectionUpdate> aggregates;
    std::vector<MeshletBoundsRefitJob> refits;
    std::vector<std::vector<uint32_t>> dirty_meshlets;
    std::vector<state::Entity> dirty_entities;
    for (const auto entity : entities) {
        auto &w = r.edit<MeshEditWork>(entity);
        const auto candidates = w.Candidates;
        if (!ElementWorkEmpty(buffers.GeometryWork, candidates)) {
            changed.push_back(entity);
            const auto id = *DrawnStoreId(r, entity);
            if (meshes.Get(id).SelectionSummary.Count) {
                auto &update = aggregates.emplace_back(MeshStore::SelectionUpdate{.StoreId = id});
                update.Blocks[0] = meshes.ArenaBlocks(id, MeshStore::ElementDomain::Vertex, buffers.GeometryWork, candidates);
                update.Blocks[2] = meshes.ArenaBlocks(id, MeshStore::ElementDomain::Face, buffers.GeometryWork, w.Faces);
            }
            w.Modified = w.PreviewActive = true;
            r.emplace_or_replace<MeshLayoutDirty>(entity);
            const auto &owner = RecordOf(r, entity);
            refits.push_back({&owner, &buffers.GeometryWork, w.Meshlets});
            if (meshes.ClusterGroupCount(owner) > 0u && !ElementWorkEmpty(buffers.GeometryWork, w.Meshlets)) {
                auto &meshlets = dirty_meshlets.emplace_back();
                ForEachWorkElement(buffers.GeometryWork, w.Meshlets, [&](uint32_t id) { meshlets.push_back(id); });
                dirty_entities.push_back(entity);
            }
        }
    }
    {
        const profile::CpuScope scope{"PositionRefitPass"};
        RefitCanonicalMeshletBounds(r, chain, refits);
    }
    if (!dirty_entities.empty()) {
        auto &dirty = meshes.Render().PositionDirty;
        for (uint32_t i = 0u; i < dirty_entities.size(); ++i) {
            for (const auto address : dirty_meshlets[i]) MaskSet(dirty, ElementAddress::Stored(address));
            r.emplace_or_replace<MeshletPositionsDirty>(dirty_entities[i]);
            r.emplace_or_replace<MeshDisplayDirty>(dirty_entities[i]);
        }
    }
    meshes.UpdateSelection(r, chain, aggregates);
    chain.AfterSubmit([&r, changed] { RefreshElementSelectionSummaries(r, changed); });
}

void LayOutMeshes(state::Registry &r) {
    const profile::CpuScope scope{"LayOutMeshes"};
    const bool edit_mode = r.get<const Interaction>(state::Document).Mode == InteractionMode::Edit;
    for (const auto mesh_entity : r.view<const MeshLayoutDirty>() | std::ranges::to<std::vector>()) LayOutMesh(r, edit_mode, mesh_entity);
    r.clear<MeshLayoutDirty>();
}

void ReleaseMeshLayout(state::Registry &r, state::Entity mesh_entity) {
    const auto &run = r.get<const BoundsRun>(mesh_entity);
    auto &pending = r.Resources.emplace<PendingObjectRemovals>();
    pending.BoundsEntryRanges.push_back(run.Entries);
    pending.DeriveEntryRanges.push_back(run.DeriveEntries);
    pending.PreludeTileRanges.push_back(run.Tiles);
    if (run.Posed()) ReleasePoses(r.Resources.get<GpuBuffers>(), mesh_entity);
}

void RefreshMeshDisplays(state::Registry &r) {
    const DisplayContext context{r};
    for (const auto mesh_entity : r.view<const MeshDisplayDirty>() | std::ranges::to<std::vector>()) {
        if (const auto *mb = TryRecordOf(r, mesh_entity)) RefreshMeshDisplay(r, context, mesh_entity, *mb, r.try_get<const PosedLayout>(mesh_entity));
    }
}
