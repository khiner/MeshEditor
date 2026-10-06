#include "metal/AutoreleaseScope.h"
#include "numeric/uvec2.h"

#include "render/LightComponents.h"
#include "viewport/ViewportRenderGpu.h"

#include "Camera.h"
#include "Profile.h"
#include "SortUnique.h"
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
#include "gpu/ObjectOriginPushConstants.h"
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
#include "mesh/ElementMembershipWork.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshClosure.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/NormalDeriveGpu.h"
#include "metal/Dispatch.h"
#include "metal/MetalCpp.h"
#include "metal/PassChain.h"
#include "metal/RenderTarget.h"
#include "numeric/MatrixMath.h"
#include "physics/PhysicsTypes.h"
#include "render/ElementWorkOps.h"
#include "render/Encoding.h"
#include "render/GpuBufferOps.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/MeshTopologyRepair.h"
#include "render/MeshletBoundsRefit.h"
#include "render/Pipelines.h"
#include "render/RenderTargets.h"
#include "render/SceneUpdates.h"
#include "scene/CameraLens.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCamera.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportInteractionState.h"

#include "state/Scene.h"

#include <cassert>
#include <cstring>
#include <numbers>
#include <tuple>

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

// Writes the overlay jobs into their buffer in stable threadgroup chunks: each extras gizmo, collider wire and tet wireframe, and each mesh instance's bounds while bounding boxes show.
// The GPU filters the display settings and selection state at use time.
void BuildOverlayJobs(const state::Scene &r, GpuBuffers &buffers, bool bounding_boxes) {
    constexpr uint32_t LinesPerJob{uint32_t(OverlayDispatch::LineGroupLines)};
    const auto object_ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>();
    // Visits each instance slot of the buffer entity with the instance's entity.
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

void RecordSceneCounters(const GpuBuffers &buffers) {
    profile::RecordCounter("InstanceSlots", buffers.Instances.TransformBuffer.UsedSize / sizeof(Transform));
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

// Hashes the inputs of the prelude layout, so an unchanged layout keeps its tiles and an unchanged workload keeps its results.
struct LayoutHash {
    uint64_t Value{0xcbf29ce484222325ull};
    void Mix(uint64_t v) { Value = (Value ^ v) * 0x100000001b3ull; }
};

// The display settings a layout rebuild reads, packed so a change to any of them rebuilds the layout.
uint64_t LayoutDisplayInputs(const ViewportDisplay &d) {
    return uint64_t(d.ShowOverlays) | uint64_t(d.ShowBones) << 1u | uint64_t(d.ShowBoundingBoxes) << 2u | uint64_t(d.ShowExtras) << 3u |
        uint64_t(d.ShowTetWireframe) << 4u | uint64_t(d.ShowOutlineSelected) << 5u | uint64_t(d.NormalOverlays) << 8u |
        uint64_t(d.ViewportShading) << 16u | uint64_t(d.FillMode) << 24u | uint64_t(d.DebugChannel) << 32u;
}

struct DeformSlots {
    uint32_t BoneDeformOffset{InvalidOffset}, ArmatureDeformOffset{InvalidOffset}, MorphDeformOffset{InvalidOffset};
    uint32_t MorphTargetCount{0};
    // Per-instance armature palette: buffer_index -> offset (instances of one mesh can bind different armatures)
    std::unordered_map<uint32_t, uint32_t> ArmatureDeformByBufferIndex;
    // Per-instance morph weights: buffer_index -> offset (weights are per-node in glTF)
    std::unordered_map<uint32_t, uint32_t> MorphWeightsByBufferIndex;
    // The armature data entities whose poses deform the mesh.
    std::vector<state::Entity> Armatures;
};
const DeformSlots NoDeform{};

std::unordered_map<state::Entity, DeformSlots> BuildDeformSlots(const state::Scene &r, const MeshStore &meshes) {
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
        if (!std::ranges::contains(slots.Armatures, modifier.ArmatureEntity)) slots.Armatures.push_back(modifier.ArmatureEntity);
        if (const auto *ri = r.try_get<const RenderInstance>(instance_entity)) slots.ArmatureDeformByBufferIndex[ri->BufferIndex] = deform_offset;
    }
    for (const auto [instance_entity, instance, gpu_range, ri] : r.view<const Instance, const MorphWeightRange, const RenderInstance>().each()) {
        const auto mesh_entity = instance.Entity;
        const auto &record = meshes.Get(r.get<const MeshHandle>(mesh_entity).StoreId);
        if (!record.MorphBlocksReady) continue;
        auto &slots = result[mesh_entity];
        slots.MorphDeformOffset = 0u;
        slots.MorphTargetCount = record.MorphTargetCount;
        slots.MorphWeightsByBufferIndex[ri.BufferIndex] = gpu_range.Weights.Offset;
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
                                    BoundsLevel2,
                                    BoundsLevel3 };

constexpr uint64_t PreludeArgsOffset(PreludeSlot slot) { return uint64_t(slot) * sizeof(MTL::DispatchThreadgroupsIndirectArguments); }

// Where the posed prelude's passes read their tiles and jobs, and how many groups each dispatches.
// The full prelude reads the layout's lists with the counts the submit writes, and a prelude over dirty entries reads their own lists with direct counts.
struct PreludeSource {
    uint32_t BoundsTilesSlot, DeriveTilesSlot, MeshletJobsSlot, MeshletJobCount;
    std::array<uint32_t, VertexBoundsLevels> BoundsFirstTiles;
    uint32_t DeriveGatherFirstTile;
    std::array<uint32_t, GpuBuffers::PreludePassCount> Groups; // In PreludeSlot order.
    bool Indirect;

    bool Has(PreludeSlot slot) const { return Groups[uint32_t(slot)] > 0u; }
    void Dispatch(MTL::ComputeCommandEncoder *encoder, const GpuBuffers &buffers, PreludeSlot slot, MTL::Size group_size) const {
        if (Indirect) encoder->dispatchThreadgroups(*buffers.PreludeDispatchArgs, PreludeArgsOffset(slot), group_size);
        else if (Has(slot)) encoder->dispatchThreadgroups(MTL::Size(Groups[uint32_t(slot)], 1, 1), group_size);
    }
};

PreludeSource LayoutPreludeSource(const GpuBuffers &buffers) {
    return {
        buffers.BoundsTiles.Slot,
        buffers.DeriveTiles.Slot,
        buffers.PosedMeshletBoundsJobs.Slot,
        buffers.PosedMeshletBoundsJobs.Count<PosedMeshletBoundsJob>(),
        buffers.BoundsFirstTiles,
        buffers.PreludeGroups[uint32_t(PreludeSlot::DeriveFaces)],
        buffers.PreludeGroups,
        true,
    };
}

// Address metadata only.
// Unchanged namespace revisions skip this enumeration.
template<typename T>
std::vector<uint32_t> PoseElementBlocks(const ElementArena<T> &arena, ElementSetRef set) {
    std::vector<uint32_t> blocks;
    for (auto b = set ? arena.Set(set).First : InvalidOffset; b != InvalidOffset; b = arena.Blocks.Get({b, 1u})[0].Next)
        blocks.push_back(b);
    return blocks;
}

std::vector<uint32_t> NormalPayloadBlocks(const MeshStore &meshes, uint32_t store_id) {
    std::vector<uint32_t> blocks;
    for (const auto block : meshes.GetBlockList(store_id, MeshStore::ElementDomain::Halfedge).Blocks)
        if (const auto payload = meshes.Arenas().NormalSectors.PayloadBlock(block)) blocks.push_back(payload - 1u);
    return blocks;
}

// Writes the pose's position and normal namespaces for instance `i` into `out`, InvalidOffset for a pose without derived normals.
void SetPoseNamespaces(auto &out, const PosedNamespaces &pose, uint32_t i) {
    const auto normals = pose.NormalsAt(i).value_or(PosedNamespaces::NormalNamespaces{});
    out.PositionNamespace = pose.PositionNamespace(i);
    out.VertexNormalNamespace = normals.Vertex;
    out.SectorNamespace = normals.Sector;
    out.FaceNormalNamespace = normals.Face;
}

// Each recorder below sets its pipeline over the scene bindings its encoder already holds.

// Materialize posed positions and reduce every entry's canonical vertex blocks.
void RecordPosePrepass(MTL::ComputeCommandEncoder *encoder, const Pipelines &pipelines, const GpuBuffers &buffers, const PreludeSource &source) {
    encoder->setComputePipelineState(pipelines.PosePrepass.State());
    const BoundsReducePushConstants pc{
        .BoundsEntrySlot = buffers.BoundsReduceEntries.Slot,
        .TileMapSlot = source.BoundsTilesSlot,
        .ValuesSlot = buffers.VertexBounds.Values.Buffer.Slot,
        .NodesSlot = buffers.VertexBounds.Nodes.Buffer.Slot,
        .MembersSlot = buffers.VertexBounds.Members.Slot,
        .FirstTile = source.BoundsFirstTiles[0],
    };
    encode::SetPushConstants(encoder, pc);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector, 0);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector, 1);
    source.Dispatch(encoder, buffers, PreludeSlot::PosePrepass, ThreadgroupSize::Linear256);
}

// One derive dispatch over the source's tiles from `pc.FirstTile`, running the face or gather phase per pc.Phase.
void RecordNormalDerive(MTL::ComputeCommandEncoder *encoder, const mtl::ComputePipeline &pipeline, const GpuBuffers &buffers, NormalDerivePushConstants pc, const PreludeSource &source, PreludeSlot slot) {
    encoder->setComputePipelineState(pipeline.State());
    pc.TileMapSlot = source.DeriveTilesSlot;
    encode::SetPushConstants(encoder, pc);
    source.Dispatch(encoder, buffers, slot, ThreadgroupSize::Linear256);
}

// Shared derive resources, writing the posed or the base normal outputs.
// Each entry selects base records or a posed namespace.
NormalDerivePushConstants MakeNormalDerivePc(const GpuBuffers &buffers, const MeshStore &meshes, bool posed) {
    return {
        .EntriesSlot = buffers.NormalDeriveEntries.Slot,
        .CornerSectors = meshes.Slots().CornerSector,
        .EdgeSharpnessSlot = meshes.Slots().EdgeSharpness,
        .FaceSharpnessSlot = meshes.Slots().FaceSharpness,
        .TileMapSlot = buffers.DeriveTiles.Slot,
        .PositionSlot = buffers.PosedPositions.Values.Buffer.Slot,
        .PositionNodesSlot = buffers.PosedPositions.Nodes.Buffer.Slot,
        .VertexNormalSlot = posed ? buffers.PosedVertexNormals.Values.Buffer.Slot : meshes.Slots().BaseVertexNormal,
        .VertexNormalNodesSlot = buffers.PosedVertexNormals.Nodes.Buffer.Slot,
        .NormalSectors = meshes.Slots().NormalSector,
        .PosedSectorNodesSlot = buffers.PosedSectors.Nodes.Buffer.Slot,
        .PosedSectorValuesSlot = buffers.PosedSectors.Values.Buffer.Slot,
        .FaceNormalSlot = posed ? buffers.PosedFaceNormals.Values.Buffer.Slot : meshes.Slots().BaseFaceNormal,
        .FaceNormalNodesSlot = buffers.PosedFaceNormals.Nodes.Buffer.Slot,
        .BaseFaceNormalSlot = meshes.Slots().BaseFaceNormal,
    };
}

// One bounds level over `pc.Work`'s elements when it names element work, else over the source's tiles at `pc.Level`.
void RecordBoundsPass(MTL::ComputeCommandEncoder *encoder, const mtl::ComputePipeline &pipeline, const GpuBuffers &buffers, BoundsReducePushConstants pc, const PreludeSource *source = nullptr, PreludeSlot slot = PreludeSlot::BoundsLevel1) {
    pc.BoundsEntrySlot = buffers.BoundsReduceEntries.Slot;
    pc.BoundsSlot = buffers.Instances.BoundsBuffer.Slot;
    pc.ValuesSlot = buffers.VertexBounds.Values.Buffer.Slot;
    pc.NodesSlot = buffers.VertexBounds.Nodes.Buffer.Slot;
    pc.MembersSlot = buffers.VertexBounds.Members.Slot;
    if (source) {
        pc.TileMapSlot = source->BoundsTilesSlot;
        pc.FirstTile = source->BoundsFirstTiles[pc.Level];
    }
    encoder->setComputePipelineState(pipeline.State());
    encode::SetPushConstants(encoder, pc);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector, 0);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector, 1);
    if (source) source->Dispatch(encoder, buffers, slot, ThreadgroupSize::Linear256);
    else encoder->dispatchThreadgroups(*buffers.GeometryWork.Buffer, WorkArgsOffset(pc.Work, true), ThreadgroupSize::Linear256);
}

// Posed meshlet bounds over `pc.Work`'s meshlets when it names element work, else over the source's jobs.
void RecordPosedMeshletBounds(MTL::ComputeCommandEncoder *encoder, const Pipelines &pipelines, const GpuBuffers &buffers, PosedMeshletBoundsPushConstants pc, const PreludeSource *source = nullptr) {
    if (source) {
        pc.JobsSlot = source->MeshletJobsSlot;
        pc.JobCount = source->MeshletJobCount;
    }
    pc.MeshletSlot = buffers.Render->Meshlets.Buffer.Slot;
    pc.MeshletVertexSlot = buffers.Render->MeshletVertexCorners.Buffer.Slot;
    pc.PosedMeshletBoundsSlot = buffers.PosedMeshletBounds.Values.Buffer.Slot;
    pc.PosedMeshletBoundsNodesSlot = buffers.PosedMeshletBounds.Nodes.Buffer.Slot;
    encoder->setComputePipelineState(pipelines.PosedMeshletBounds.State());
    encode::SetPushConstants(encoder, pc);
    if (source) source->Dispatch(encoder, buffers, PreludeSlot::PosedMeshletBounds, ThreadgroupSize::Linear32);
    else encoder->dispatchThreadgroups(*buffers.GeometryWork.Buffer, WorkArgsOffset(pc.Work, true), ThreadgroupSize::Linear32);
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
        RecordNormalDerive(compute, normal_pipeline, buffers, derive, source, PreludeSlot::DeriveFaces);
        compute->memoryBarrier(MTL::BarrierScopeBuffers);
        derive.Phase = 1u;
        derive.FirstTile = source.DeriveGatherFirstTile;
        RecordNormalDerive(compute, normal_pipeline, buffers, derive, source, PreludeSlot::DeriveGather);
        compute->memoryBarrier(MTL::BarrierScopeBuffers);
    }
    if (source.Has(PreludeSlot::PosedMeshletBounds)) RecordPosedMeshletBounds(compute, pipelines, buffers, {}, &source);
    if (source.Has(PreludeSlot::BoundsLevel3)) {
        RecordBoundsPass(compute, pipelines.BoundsCombine, buffers, {.Level = 1u}, &source, PreludeSlot::BoundsLevel1);
        compute->memoryBarrier(MTL::BarrierScopeBuffers);
        RecordBoundsPass(compute, pipelines.BoundsCombine, buffers, {.Level = 2u}, &source, PreludeSlot::BoundsLevel2);
        compute->memoryBarrier(MTL::BarrierScopeBuffers);
        RecordBoundsPass(compute, pipelines.BoundsCombine, buffers, {.Level = 3u}, &source, PreludeSlot::BoundsLevel3);
    }
}

// Gathers the dirty entries' tiles and posed meshlet jobs into the sparse lists, and returns them as a prelude source.
// Every dirty entry names an entry of the current layout, since a layout rebuild clears them.
PreludeSource DirtyPreludeSource(GpuBuffers &buffers, GpuSceneState &scene) {
    auto &dirty = scene.DirtyBoundsEntries;
    SortUnique(dirty);
    const auto layout_bounds = buffers.BoundsTiles.GetSpan<uvec2>();
    const auto layout_derive = buffers.DeriveTiles.GetSpan<uvec2>();
    const auto layout_jobs = buffers.PosedMeshletBoundsJobs.GetSpan<PosedMeshletBoundsJob>();
    std::array<std::vector<uvec2>, VertexBoundsLevels> bounds;
    std::vector<uvec2> faces, gathers;
    std::vector<PosedMeshletBoundsJob> jobs;
    uint32_t groups = 0u;
    for (const auto entry : dirty) {
        const auto &tiles = scene.BoundsEntryTiles[entry];
        for (uint32_t level = 0u; level < VertexBoundsLevels; ++level) bounds[level].append_range(layout_bounds.subspan(tiles.Bounds[level].Offset, tiles.Bounds[level].Count));
        faces.append_range(layout_derive.subspan(tiles.DeriveFaces.Offset, tiles.DeriveFaces.Count));
        gathers.append_range(layout_derive.subspan(tiles.DeriveGather.Offset, tiles.DeriveGather.Count));
        for (auto job : layout_jobs.subspan(tiles.MeshletJobs.Offset, tiles.MeshletJobs.Count)) {
            job.FirstGroup = groups;
            groups += job.Count;
            jobs.push_back(job);
        }
    }
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
        buffers.SparseBoundsTiles.Slot,
        buffers.SparseDeriveTiles.Slot,
        buffers.SparsePosedMeshletBoundsJobs.Slot,
        uint32_t(jobs.size()),
        first,
        uint32_t(faces.size()),
        {uint32_t(bounds[0].size()), groups, uint32_t(faces.size()), uint32_t(bounds[1].size()), uint32_t(gathers.size()), uint32_t(bounds[2].size()), uint32_t(bounds[3].size())},
        false,
    };
}

// Buffer bindings shared by meshlet classification dispatches.
MeshletCullPushConstants MakeMeshletCullSlotsPc(const GpuBuffers &buffers, const MeshletCullOutput &output) {
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
        .MeshletIndexNodesSlot = buffers.Render->ActiveMeshlets.Nodes.Buffer.Slot,
        .MeshletIndexLeavesSlot = buffers.Render->ActiveMeshlets.Leaves.Buffer.Slot,
        .ClusterGroupSlot = buffers.Render->ClusterGroups.Buffer.Slot,
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
    uint32_t edit_edge_corner = 0u
) {
    return {
        .PrimitiveSlot = buffers.Render->Primitives.Buffer.Slot,
        .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
        .InstanceMapSlot = buffers.GpuInstanceSlots.Slot,
        .MeshletSlot = buffers.Render->Meshlets.Buffer.Slot,
        .MeshletTriangleSlot = buffers.Render->MeshletTriangleIds.Buffer.Slot,
        .MeshletVertexSlot = buffers.Render->MeshletVertexCorners.Buffer.Slot,
        .MeshletLocalTriangleSlot = buffers.Render->MeshletLocalTriangles.Buffer.Slot,
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

// The scene state every mesh's display fields read beyond the mesh itself, gathered once per refresh.
struct DisplayContext {
    DisplayContext(const state::Scene &r, state::Entity viewport)
        : R(r), Buffers(r.Context.get<const GpuBuffers>()), Meshes(r.Context.get<const MeshStore>()), State(r.Context.get<const GpuSceneState>()),
          Settings(r.get<const ViewportDisplay>(viewport)), Mode(r.get<const Interaction>(viewport).Mode), EditElement(r.get<const EditMode>(viewport).Value),
          Primaries(r.get<const EditPrimaries>(viewport).All), SelectedMeshes(r.get<const SelectionFlags>(viewport).Meshes),
          Transforming(Mode == InteractionMode::Edit && r.all_of<PendingTransform>(viewport)) {
        if (Mode == InteractionMode::Excite) {
            for (const auto [_, instance, __] : r.view<const Instance, const SoundVertices>().each()) SoundMeshes.insert(instance.Entity);
        }
    }

    const state::Scene &R;
    const GpuBuffers &Buffers;
    const MeshStore &Meshes;
    const GpuSceneState &State;
    const ViewportDisplay &Settings;
    InteractionMode Mode;
    Element EditElement;
    const selection::PrimaryEditInstanceMap &Primaries;
    const std::vector<state::Entity> &SelectedMeshes;
    bool Transforming; // An Edit-mode transform is pending.
    std::unordered_set<state::Entity> SoundMeshes;

    bool Wireframe() const { return Settings.ViewportShading == ViewportShadingMode::Wireframe; }
    // Edit and Pose modes draw the active armature's bones, Object mode the selected armatures', and wireframe shading every armature's.
    bool DrawsArmatureBones(state::Entity armature) const {
        if (Wireframe()) return true;
        if (Mode == InteractionMode::Edit || Mode == InteractionMode::Pose) return R.all_of<Active>(armature);
        return R.all_of<Selected>(armature);
    }
};

// The armature object whose bones a joint buffer entity draws the joints of.
state::Entity JointArmature(const state::Scene &r, state::Entity joint_buffer) {
    for (const auto [armature, object] : r.view<const ArmatureObject>().each())
        if (object.JointEntity == joint_buffer) return armature;
    return state::Null;
}

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

// One mesh's record display fields and the VertexOverlay bits its instances draw.
struct MeshDisplayState {
    MeshDisplay Display;
    uint8_t Overlays{0};
};

MeshDisplayState ComputeMeshDisplay(const DisplayContext &c, state::Entity mesh_entity, const MeshStore::Record &mb, const DeformSlots &deform, const PosedNamespaces *pose) {
    const auto &r = c.R;
    const auto &meshes = c.Meshes;
    MeshDisplay d{
        .PrimitiveRoot = mb.PrimitiveRoot,
        .PrimitiveCount = c.Meshes.PrimitiveCount(mb),
        .MorphDeformOffset = deform.MorphDeformOffset,
        .BoneDeformOffset = deform.BoneDeformOffset,
        .MorphTargetCount = deform.MorphTargetCount,
    };
    if (pose && !pose->PerInstance) {
        SetPoseNamespaces(d, *pose, 0);
        d.MeshletBoundsNamespace = pose->MeshletBoundsNamespace(0);
        d.MorphNormalNamespace = pose->MorphNormalNamespace(0);
    }
    // A mesh object's canonical record, which bones, joints and extras lack.
    const auto *handle = r.try_get<const MeshHandle>(mesh_entity);
    const auto *record = handle ? &meshes.Get(handle->StoreId) : nullptr;
    const auto &arenas = meshes.Arenas();
    const uint32_t faces = record ? arenas.FaceTriangles.Count(record->FaceData) : 0u, edges = record ? arenas.EdgeHalfedges.Count(record->EdgeData) : 0u;
    const bool meshlets = c.Meshes.MeshletCount(mb) > 0u;
    const auto primary = c.Primaries.find(mesh_entity);
    const bool has_primary = primary != c.Primaries.end();
    const bool sound = c.SoundMeshes.contains(mesh_entity);
    const bool overlays = c.Settings.ShowOverlays;
    if (has_primary && record) {
        d.HasPendingVertexTransform = c.Transforming ? 1u : 0u;
        d.PrimaryEditInstanceIndex = r.get<const RenderInstance>(primary->second).BufferIndex;
        d.Selection = meshes.GetEditSelectionStorage(handle->StoreId);
        d.EditEdgeSharpnessOffset = arenas.EdgeHalfedges.First(record->EdgeData);
        d.ElementIdOffset = meshes.GetSelectionBitOffset(handle->StoreId, c.EditElement);
    } else if (sound && record) {
        d.Selection = meshes.GetEditSelectionStorage(handle->StoreId);
        const auto *active = r.try_get<const MeshActiveElement>(mesh_entity);
        d.ActiveVertex = active ? active->Handle : InvalidOffset;
    }

    const bool armature = r.all_of<ArmatureObject>(mesh_entity), joint = r.all_of<BoneJoint>(mesh_entity);
    const bool shaded_face_less = record && !WorkbenchShading(c.Settings.ViewportShading) && faces == 0u && record->PrimitiveMaterials.Count > 0u;
    const bool normals = overlays && c.Settings.NormalOverlays != 0u && std::ranges::binary_search(c.SelectedMeshes, mesh_entity);
    uint32_t flags = 0u;
    uint8_t vertex_overlays = 0u;
    const auto set = [&](MeshletInstanceFlag flag) { flags |= uint32_t(flag); };
    const auto draw = [&](VertexOverlay overlay) { vertex_overlays |= uint8_t(1u << uint32_t(overlay)); };
    if (IsSilhouetteEligible(r, mesh_entity)) set(MeshletInstanceFlag::SilhouetteEligible);
    if (EditPinsFinest(c.Primaries, c.State, mesh_entity) || c.Meshes.Render().ActiveMeshlets.Count(mb.PositionDirtyRoot) > 0u) set(MeshletInstanceFlag::LodPinFinest);
    if (has_primary && meshlets && record && GetMesh(r, mesh_entity).ElementCount(c.EditElement) > 0u) set(MeshletInstanceFlag::ElementSelection);
    if (overlays && c.Mode == InteractionMode::Edit && has_primary && record && meshlets) {
        set(MeshletInstanceFlag::EditOverlay);
        draw(VertexOverlay::EditPoints);
    }
    if (meshlets && !armature && !joint && !r.all_of<ObjectExtrasTag>(mesh_entity) && record && edges > 0u &&
        (faces == 0u || c.Wireframe()) && !shaded_face_less) set(MeshletInstanceFlag::Wire);
    if (armature || joint) set(MeshletInstanceFlag::OverlayOnly);
    if (overlays && c.Settings.ShowBones) {
        if (armature) {
            set(MeshletInstanceFlag::Bone);
            if (c.DrawsArmatureBones(mesh_entity)) set(MeshletInstanceFlag::BoneWire);
        } else if (joint) {
            set(MeshletInstanceFlag::BoneJoint);
            if (const auto owner = JointArmature(r, mesh_entity); owner == state::Null || c.DrawsArmatureBones(owner)) set(MeshletInstanceFlag::BoneJointWire);
        }
    }
    if (normals && he::ElementMaskContains(c.Settings.NormalOverlays, Element::Face) && faces > 0u) set(MeshletInstanceFlag::FaceNormal);
    if (normals && he::ElementMaskContains(c.Settings.NormalOverlays, Element::Vertex) && edges > 0u) {
        set(MeshletInstanceFlag::LodPinFinest);
        draw(VertexOverlay::Normals);
    }
    if (overlays && c.Mode == InteractionMode::Excite && sound) {
        if (edges > 0u) set(MeshletInstanceFlag::EdgeOverlay);
        if (meshlets) draw(VertexOverlay::SoundPoints);
    }
    // A mesh without faces or edges draws points.
    if (record && faces == 0u && edges == 0u && !has_primary && !shaded_face_less) draw(VertexOverlay::Points);
    d.Flags = flags;
    return {d, vertex_overlays};
}

// Rederives the mesh's vertex overlays and writes its record display fields when they changed, moving its flag totals to them.
void RefreshMeshDisplay(state::Scene &r, const DisplayContext &c, state::Entity mesh_entity, const MeshStore::Record &mb, const DeformSlots &deform, const PosedNamespaces *pose) {
    auto &scene = r.Context.get<GpuSceneState>();
    const auto [display, overlays] = ComputeMeshDisplay(c, mesh_entity, mb, deform, pose);
    if (display.Flags & uint32_t(MeshletInstanceFlag::EditOverlay)) scene.MeshletEditHasSharpEdges |= c.Meshes.GetEdgeSharpnessSummary(GetMesh(c.R, mesh_entity).GetStoreId()).Any;
    if (overlays) scene.VertexOverlays[mesh_entity] = overlays;
    else scene.VertexOverlays.erase(mesh_entity);
    if (mb.RenderTopologies == 0u) return;
    if (auto &record = r.Context.get<MeshStore>().Render().MeshRecords.GetMutable({mb.StoreId, 1u})[0]; std::memcmp(&record.Display, &display, sizeof(MeshDisplay)) != 0) {
        record.Display = display;
        RetallyMesh(r, mesh_entity);
    }
}

// Lays out the bounds entries, prelude tiles and pose namespaces of every mesh with instances, and rederives every mesh's record display fields.
// Instance records change only for meshes whose instances deform apart, which hold per-instance deform offsets and pose namespaces.
void RebuildMeshLayout(state::Scene &r, state::Entity viewport) {
    const profile::CpuScope build_scope{"UpdateGpuScene"};
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    auto &scene_state = r.Context.get<GpuSceneState>();
    const DisplayContext context{r, viewport};
    const bool is_edit_mode = context.Mode == InteractionMode::Edit;
    // Edit mode uses the rest pose.
    const auto mesh_deform_slots = is_edit_mode ? std::unordered_map<state::Entity, DeformSlots>{} : BuildDeformSlots(r, meshes);
    // Meshes whose instance records held per-instance deform state, which clears when the mesh stops deforming apart.
    std::vector<state::Entity> deformed_apart;
    for (const auto &[entity, pose] : scene_state.PosedByEntity)
        if (pose.PerInstance) deformed_apart.push_back(entity);
    scene_state.PosedByEntity.clear();
    const auto records = buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>();

    // A mesh buffer entity the layout covers, with its mesh and deform state.
    struct LayoutMesh {
        state::Entity Entity;
        const MeshStore::Record &Buf;
        const ModelsBuffer &Mod;
        std::optional<Mesh> MeshComp;
        const DeformSlots &Deform;
    };

    // Sort by descending entity ID for deterministic coincident-surface ordering across scene loads.
    const auto mesh_entity_order = SortedEntities(r.view<const ModelsBuffer>(), std::ranges::greater{});

    std::vector<LayoutMesh> mesh_entities;
    mesh_entities.reserve(mesh_entity_order.size());
    for (const auto entity : mesh_entity_order) {
        const auto *mesh_buffers = TryRecordOf(r, entity);
        if (!mesh_buffers) continue;
        const auto deform = mesh_deform_slots.find(entity);
        mesh_entities.emplace_back(
            entity, *mesh_buffers, r.get<const ModelsBuffer>(entity), TryGetMesh(r, entity), deform != mesh_deform_slots.end() ? deform->second : NoDeform
        );
    }

    { // Bounds reduce entries.
        // Instances sharing one deform state share one entry, whose run of slots spans their consecutive instances.
        // Entries with morph, armature, or pending edit-transform deformation own poses.
        // Posed entries share a position namespace.
        // A mesh without instances keeps an empty static entry, which its instances fill in place when they appear.
        // The same pass writes leaf bounds.
        struct BoundsEntrySpec {
            uint32_t Count{}, NormalVertexTiles{}, NormalFaceTiles{};
            uint64_t VertexLayoutRevision{}, FaceLayoutRevision{};
            bool PerInstanceDeform{}, Posed{}, Derive{};
            NormalDeriveEntry Entry{}; // Derive-input fields, filled when Derive.
            PosedNamespaces Pose;
            const VertexBoundsStore::Keys *BoundsKeys{};
        };
        std::vector<BoundsEntrySpec> specs(mesh_entities.size());
        // Leaf work covers canonical blocks.
        // Parent work has three fixed levels.
        uint32_t entry_count = 0;
        uint32_t derive_entry_count = 0;
        std::array<uint32_t, VertexBoundsLevels> bounds_tile_counts{};
        uint32_t derive_face_tile_count = 0, derive_gather_tile_count = 0;
        uint32_t posed_meshlet_bounds_count = 0;
        LayoutHash prelude_layout, prelude_work;
        const auto pose_stores = std::tie(buffers.VertexBounds, buffers.PosedPositions, buffers.PosedMorphNormalDeltas, buffers.PosedVertexNormals, buffers.PosedFaceNormals, buffers.PosedSectors, buffers.PosedMeshletBounds);
        std::apply([](auto &...stores) { (stores.BeginUpdate(), ...); }, pose_stores);
        for (size_t mi = 0; mi < mesh_entities.size(); ++mi) {
            const auto &e = mesh_entities[mi];
            auto &spec = specs[mi];
            if (!e.MeshComp) continue;
            const bool instanced = e.Mod.InstanceCount > 0u;
            spec.PerInstanceDeform = !e.Deform.ArmatureDeformByBufferIndex.empty() || !e.Deform.MorphWeightsByBufferIndex.empty();
            spec.Count = spec.PerInstanceDeform ? e.Mod.InstanceCount : 1u;
            // Some canonical position edits need posed bounds until their meshlets are refitted.
            // Inset previews refit them directly.
            spec.Posed = instanced && (e.Deform.BoneDeformOffset != InvalidOffset || e.Deform.MorphDeformOffset != InvalidOffset || (is_edit_mode && ((context.Transforming && context.Primaries.contains(e.Entity)) || (scene_state.EditWork.contains(e.Entity) && scene_state.EditWork.at(e.Entity).RequiresPose))));
            entry_count += spec.Count;
            if (spec.Posed) {
                spec.Pose.FirstInstance = e.Mod.InstanceRange.Offset;
                spec.Pose.PerInstance = spec.PerInstanceDeform;
                const auto store_id = e.MeshComp->GetStoreId();
                const auto &arena = meshes.Arenas().Vertices;
                const auto set = meshes.Get(store_id).Vertices;
                const auto vertex_revision = set ? arena.Set(set).Revision : 0u;
                const auto vertex_blocks = [&] { return PoseElementBlocks(arena, set); };
                const auto vertex_bounds = buffers.VertexBounds.Prepare(e.Entity, store_id, vertex_revision, spec.Count, vertex_blocks);
                spec.BoundsKeys = &vertex_bounds.Nodes;
                spec.VertexLayoutRevision = vertex_bounds.LayoutRevision;
                spec.Pose.VertexBoundsNamespaces.assign(vertex_bounds.Roots.begin(), vertex_bounds.Roots.end());
                for (uint32_t level = 0u; level < VertexBoundsLevels; ++level)
                    bounds_tile_counts[level] += spec.Count * uint32_t(vertex_bounds.Nodes[level].size());
                if (vertex_bounds.Changed) buffers.PreludeStale = true;
                // Outside edit mode, which builds no deform slots, authored morph shading reads rest normals plus weighted authored deltas.
                const bool authored_morph = e.Deform.MorphDeformOffset != InvalidOffset && meshes.Get(store_id).MorphShadingAuthored;
                const auto positions = buffers.PosedPositions.Prepare(e.Entity, store_id, vertex_revision, spec.Count, vertex_blocks);
                spec.Pose.PositionNamespaces.assign(positions.Roots.begin(), positions.Roots.end());
                if (positions.Changed) buffers.PreludeStale = true;
                if (authored_morph) {
                    const auto deltas = buffers.PosedMorphNormalDeltas.Prepare(e.Entity, store_id, vertex_revision, spec.Count, vertex_blocks);
                    spec.Pose.MorphNormalNamespaces.assign(deltas.Roots.begin(), deltas.Roots.end());
                    if (deltas.Changed) buffers.PreludeStale = true;
                }
                if (const auto derive_entry = authored_morph ? std::nullopt : MakeDeriveEntryInputs(meshes, store_id)) {
                    spec.Derive = true;
                    spec.Entry = *derive_entry;
                    const auto vertex_normals = buffers.PosedVertexNormals.Prepare(e.Entity, store_id, vertex_revision, spec.Count, vertex_blocks);
                    const auto &faces = meshes.Arenas().FaceTriangles;
                    const auto face_set = meshes.Get(store_id).FaceData;
                    const auto face_normals = buffers.PosedFaceNormals.Prepare(e.Entity, store_id, faces.Set(face_set).Revision, spec.Count, [&] { return PoseElementBlocks(faces, face_set); });
                    spec.FaceLayoutRevision = face_normals.LayoutRevision;
                    const auto prepared = buffers.PosedSectors.Prepare(e.Entity, store_id, meshes.GetDerived(store_id).NormalRevision, spec.Count, [&] { return NormalPayloadBlocks(meshes, store_id); });
                    for (uint32_t i = 0u; i < spec.Count; ++i) spec.Pose.Normals.push_back({vertex_normals.Roots[i], prepared.Roots[i], face_normals.Roots[i]});
                    if (vertex_normals.Changed || face_normals.Changed || prepared.Changed) buffers.PreludeStale = true;
                    derive_entry_count += spec.Count;
                    spec.NormalVertexTiles = arena.Set(set).BlockCount;
                    spec.NormalFaceTiles = faces.Set(face_set).BlockCount;
                    derive_face_tile_count += spec.Count * spec.NormalFaceTiles;
                    derive_gather_tile_count += spec.Count * spec.NormalVertexTiles;
                }
                const auto bounds = buffers.PosedMeshletBounds.Prepare(e.Entity, store_id, e.Buf.MeshletRevision, spec.Count, [&] {
                    std::vector<uint32_t> blocks;
                    meshes.ForEachPrimitive(e.Buf,[&](uint32_t, const PrimitiveRecord &primitive) {
                        if (primitive.LodFinestNode == InvalidOffset) return;
                        const auto root = render.LodNodes.Get({primitive.LodFinestNode,1u})[0].MeshletRoot;
                        render.ActiveMeshlets.ForEachBlock(root,[&](uint32_t b) { blocks.push_back(b); });
                    });
                    return blocks; }, e.Buf.RenderTopologies);
                spec.Pose.MeshletBoundsNamespaces.assign(bounds.Roots.begin(), bounds.Roots.end());
                if (bounds.Changed) buffers.PreludeStale = true;
                posed_meshlet_bounds_count += spec.Count * e.Buf.Level0Count;
            }
            if (!spec.Posed) bounds_tile_counts.back() += spec.Count;
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
            prelude_work.Mix(meshes.Arenas().Vertices.Count(e.Buf.Vertices));
            prelude_work.Mix(spec.Entry.FaceCount);
            prelude_work.Mix(e.Mod.InstanceRange.Offset);
            prelude_work.Mix(e.Mod.InstanceCount);
        }
        std::apply([](auto &...stores) { (stores.EndUpdate(), ...); }, pose_stores);
        const bool tiles_changed = std::exchange(scene_state.PreludeLayoutInputs, prelude_layout.Value) != prelude_layout.Value;
        const bool work_changed = std::exchange(scene_state.PreludeWorkInputs, prelude_work.Value) != prelude_work.Value;
        if (tiles_changed || work_changed) buffers.PreludeStale = true;
        const auto entries = buffers.BoundsReduceEntries.SetCount<BoundsEntry>(entry_count);
        const auto derive_entries = buffers.NormalDeriveEntries.SetCount<NormalDeriveEntry>(derive_entry_count);
        uint32_t bounds_tile_count = 0u;
        for (uint32_t level = 0u; level < VertexBoundsLevels; ++level) {
            buffers.BoundsFirstTiles[level] = bounds_tile_count;
            bounds_tile_count += bounds_tile_counts[level];
        }
        const auto bounds_tiles = buffers.BoundsTiles.SetCount<uvec2>(bounds_tile_count);
        const auto derive_tiles = buffers.DeriveTiles.SetCount<uvec2>(derive_face_tile_count + derive_gather_tile_count);
        std::vector<PosedMeshletBoundsJob> meshlet_jobs;
        buffers.PreludeGroups = {bounds_tile_counts[0], posed_meshlet_bounds_count, derive_face_tile_count, bounds_tile_counts[1], derive_gather_tile_count, bounds_tile_counts[2], bounds_tile_counts[3]};
        scene_state.BoundsRuns.clear();
        scene_state.BoundsEntryTiles.assign(entry_count, {});
        // Dirty entries name the previous layout's entries, so they recompute with every entry.
        if (!scene_state.DirtyBoundsEntries.empty()) buffers.PreludeStale = true;
        scene_state.DirtyBoundsEntries.clear();
        scene_state.MorphEntries.clear();
        scene_state.ArmatureEntries.clear();

        uint32_t write = 0, derive_write = 0;
        uint32_t face_tile_write = 0, gather_tile_write = derive_face_tile_count;
        auto bounds_tile_write = buffers.BoundsFirstTiles;
        uint32_t meshlet_groups = 0u;
        for (size_t mi = 0; mi < mesh_entities.size(); ++mi) {
            const auto &e = mesh_entities[mi];
            auto &spec = specs[mi];
            if (spec.Count == 0) continue;
            const auto first_entry = write;
            scene_state.BoundsRuns[e.Entity] = {first_entry, spec.Count, spec.Posed};
            auto entry = MeshBoundsEntry(meshes, e.MeshComp->GetStoreId(), e.Mod);
            if (spec.PerInstanceDeform) entry.InstanceCount = 1u;
            auto &pr = spec.Pose;
            NormalDeriveEntry derive_entry = spec.Entry;
            std::vector<uint32_t> face_blocks, vertex_blocks;
            if (tiles_changed && spec.Derive) {
                const auto &record = meshes.Get(e.MeshComp->GetStoreId());
                face_blocks = PoseElementBlocks(meshes.Arenas().FaceTriangles, record.FaceData);
                vertex_blocks = PoseElementBlocks(meshes.Arenas().Vertices, record.Vertices);
                assert(face_blocks.size() == spec.NormalFaceTiles && vertex_blocks.size() == spec.NormalVertexTiles);
            }
            for (uint32_t i = 0; i < spec.Count; ++i) {
                auto &tiles = scene_state.BoundsEntryTiles[write];
                if (spec.Derive) {
                    SetPoseNamespaces(derive_entry, pr, i);
                    tiles.DeriveFaces = {face_tile_write, spec.NormalFaceTiles};
                    tiles.DeriveGather = {gather_tile_write, spec.NormalVertexTiles};
                    if (tiles_changed) {
                        for (const auto block : face_blocks) derive_tiles[face_tile_write++] = {derive_write, block};
                        for (const auto block : vertex_blocks) derive_tiles[gather_tile_write++] = {derive_write, block};
                    } else {
                        face_tile_write += spec.NormalFaceTiles;
                        gather_tile_write += spec.NormalVertexTiles;
                    }
                    derive_entries[derive_write++] = derive_entry;
                }
                for (uint32_t level = 0u; level < VertexBoundsLevels; ++level) {
                    const auto count = spec.Posed ? uint32_t((*spec.BoundsKeys)[level].size()) : level + 1u == VertexBoundsLevels ? 1u :
                                                                                                                                    0u;
                    tiles.Bounds[level] = {bounds_tile_write[level], count};
                    if (tiles_changed) {
                        if (spec.Posed)
                            for (const auto key : (*spec.BoundsKeys)[level]) bounds_tiles[bounds_tile_write[level]++] = {write, key};
                        else if (count) bounds_tiles[bounds_tile_write[level]++] = {write, 0u};
                    } else bounds_tile_write[level] += count;
                }
                auto instance_entry = entry;
                if (spec.PerInstanceDeform) instance_entry.FirstInstance += i;
                if (spec.Posed) instance_entry.BoundsNamespace = spec.Pose.VertexBoundsNamespaces[i];
                if (e.Deform.MorphDeformOffset != InvalidOffset) scene_state.MorphEntries.push_back(write);
                for (const auto armature : e.Deform.Armatures) scene_state.ArmatureEntries[armature].push_back(write);
                entries[write++] = instance_entry;
                // An instance deformed apart holds its own deform offsets and pose namespaces.
                if (spec.Posed && spec.PerInstanceDeform) {
                    const auto slot = entry.FirstInstance + i;
                    auto &record = records[slot];
                    const auto armature = e.Deform.ArmatureDeformByBufferIndex.find(slot);
                    const auto weights = e.Deform.MorphWeightsByBufferIndex.find(slot);
                    record.ArmatureDeformOffset = armature != e.Deform.ArmatureDeformByBufferIndex.end() ? armature->second : e.Deform.ArmatureDeformOffset;
                    record.MorphWeightsOffset = weights != e.Deform.MorphWeightsByBufferIndex.end() ? weights->second : InvalidOffset;
                    SetPoseNamespaces(record, pr, i);
                    record.MorphNormalNamespace = pr.MorphNormalNamespace(i);
                    record.MeshletBoundsNamespace = pr.MeshletBoundsNamespace(i);
                }
            }
            // Descriptors enumerate canonical clusters on the GPU.
            // All instances sharing a pose use the same bounds namespace.
            if (spec.Posed)
                for (uint32_t i = 0u; i < spec.Count; ++i) {
                    const auto instance = spec.PerInstanceDeform ? entry.FirstInstance + i : entry.FirstInstance;
                    const auto first_job = uint32_t(meshlet_jobs.size());
                    meshes.ForEachPrimitive(e.Buf, [&](uint32_t, const PrimitiveRecord &primitive) {
                        if (primitive.LodFinestNode == InvalidOffset) return;
                        const auto root = render.LodNodes.Get({primitive.LodFinestNode, 1u})[0].MeshletRoot;
                        const auto count = render.ActiveMeshlets.Count(root);
                        if (!count) return;
                        meshlet_jobs.push_back({meshlet_groups, count, instance, render.ActiveMeshlets.Ref(root)});
                        meshlet_groups += count;
                    });
                    scene_state.BoundsEntryTiles[first_entry + i].MeshletJobs = {first_job, uint32_t(meshlet_jobs.size()) - first_job};
                }
            if (spec.Posed) scene_state.PosedByEntity.emplace(e.Entity, std::move(pr));
        }
        assert(meshlet_groups == posed_meshlet_bounds_count);
        assert(face_tile_write == derive_face_tile_count && gather_tile_write == derive_tiles.size());
        const auto jobs = buffers.PosedMeshletBoundsJobs.SetCount<PosedMeshletBoundsJob>(meshlet_jobs.size());
        std::ranges::copy(meshlet_jobs, jobs.begin());
    }

    // Every instance not deformed apart holds no deform offsets or pose namespaces of its own.
    for (const auto entity : deformed_apart) {
        const auto *models = r.try_get<const ModelsBuffer>(entity);
        if (!models || scene_state.PosedByEntity.contains(entity)) continue;
        for (auto &record : records.subspan(models->InstanceRange.Offset, models->InstanceCount)) record = {.Mesh = record.Mesh, .ObjectId = record.ObjectId, .ExcitedVertex = record.ExcitedVertex};
    }

    buffers.MeshletTopologyMask = 0u;
    scene_state.MeshletEditHasSharpEdges = false;
    scene_state.VertexOverlays.clear();
    for (const auto &e : mesh_entities) {
        if (e.Mod.InstanceCount > 0u && meshes.MeshletCount(e.Buf) != 0u) {
            buffers.MeshletTopologyMask |= e.Buf.RenderTopologies;
        }
    }
    for (const auto &e : mesh_entities) {
        const auto pose = scene_state.PosedByEntity.find(e.Entity);
        RefreshMeshDisplay(r, context, e.Entity, e.Buf, e.Deform, pose != scene_state.PosedByEntity.end() ? &pose->second : nullptr);
    }
    scene_state.DisplayDirty.clear();
    scene_state.OverlayJobsDirty = true;
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
    const bool show_rendered = !WorkbenchShading(settings.ViewportShading);
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

    // A change to a display setting the layout reads rebuilds the layout.
    const auto display_inputs = LayoutDisplayInputs(settings);
    const bool layout = update == SceneUpdate::Rebuild || display_inputs != scene_state.LayoutDisplayInputs;
    if (layout) {
        scene_state.LayoutDisplayInputs = display_inputs;
        RebuildMeshLayout(r, viewport);
    }
    if (std::exchange(scene_state.OverlayJobsDirty, false)) BuildOverlayJobs(r, buffers, settings.ShowBoundingBoxes);
    // A blur records several phases that share one set of prelude dispatch counts, so it recomputes every entry.
    if (!scene_state.DirtyBoundsEntries.empty() && phase != RenderPhase::Full && phase != RenderPhase::Prepare) buffers.PreludeStale = true;
    const bool render_silhouette = show_overlays && settings.ShowOutlineSelected && !is_excite_mode &&
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::Silhouette)).Meshlets > 0u;

    // Specialize forward PBR during the authoritative rebuild scan to avoid a second registry traversal.
    if (show_rendered && layout) {
        pipelines.Main.Compiler.CompileTopologyPipelines((buffers.MeshletTopologyMask & ~1u) != 0u);
    }
    if (layout || phase == RenderPhase::Full) RecordSceneCounters(buffers);

    const bool transmission_active = real_transmission && targets.Transmission;
    // Reuse opaque transmission shading when neither edit tint nor debug output needs another shade.
    const bool composite_transmission = transmission_active && phase == RenderPhase::Full && !is_edit_mode && settings.DebugChannel == DebugChannel::None;
    const bool meshlet_fill = buffers.MeshletInstanceCount > 0;

    // Posed positions and bounds run before culling.
    // The full prelude dispatches indirectly, and a submit with unchanged deform inputs gets zero group counts, keeping the buffers' current results.
    // Entries whose own inputs changed recompute apart from it.
    const bool dirty_prelude = !buffers.PreludeStale && !scene_state.DirtyBoundsEntries.empty();
    if (buffers.PreludeHasWork() || dirty_prelude) {
        auto *compute = chain.BeginCompute("Prelude", MTL::StageVertex | MTL::StageFragment | MTL::StageDispatch);
        encode::BindScene(compute, slots, buffers, ubo_offset);
        const auto &mesh_pipelines = GetMeshPipelines(r);
        if (buffers.PreludeHasWork()) RecordPrelude(compute, pipelines, mesh_pipelines, buffers, meshes, LayoutPreludeSource(buffers));
        if (dirty_prelude) RecordPrelude(compute, pipelines, mesh_pipelines, buffers, meshes, DirtyPreludeSource(buffers, scene_state));
    }
    scene_state.DirtyBoundsEntries.clear();
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
    const bool disocclusion_possible = layout || buffers.PreludeStale || dirty_prelude || buffers.MeshletOcclusionStale ||
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
        samplers.DepthPyramid :
        InvalidSlot;
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

    const bool draw_origins = show_overlays && settings.ShowOrigins && (!r.view<const Selected>().empty() || !r.view<const Active>().empty());
    // Initialize overlays even when no geometry contributes color.
    const bool overlay_pass_needed = has_silhouette ||
        (show_overlays && settings.ShowGrid) ||
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
                uint32_t(route), uint32_t(flag), false, false, sharpness_slot, threads, corner, cull
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
            const auto &primaries = r.get<const EditPrimaries>(viewport).All;
            // Edit points draw on the mesh's primary edit instance, and other overlays on each of its instances.
            const auto draw_vertex_overlays = [&](const mtl::RenderPipeline &pipeline, VertexOverlay kind) {
                bool bound = false;
                const auto draw = [&](state::Entity mesh_entity, uint32_t slot) {
                    if (!std::exchange(bound, true)) pipeline.Bind(encoder);
                    DrawVertexBlocks(
                        encoder, r, mesh_entity, slot, kind == VertexOverlay::SoundPoints, overlay_pyramid,
                        kind == VertexOverlay::EditPoints ? MinEditOverlayDiameterPixels : 0.0f
                    );
                };
                for (const auto &[mesh_entity, mask] : scene_state.VertexOverlays) {
                    if (!(mask & (1u << uint32_t(kind)))) continue;
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
        // Origins draw over every other overlay at a fixed size in logical pixels, one instance per instance slot.
        if (draw_origins) {
            const float scale = float(main_extent.Width) / float(std::max(r.Context.get<const ViewportExtent>().Value.x, 1u));
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

void UpdateAuthoredMorphShading(state::Scene &r, mtl::ComputeChain &chain, std::span<const state::Entity> mesh_entities) {
    auto &meshes = r.Context.get<MeshStore>();
    auto &buffers = r.Context.get<GpuBuffers>();
    // Each position-only target gets a derive entry reading its full-weight pose directly.
    std::vector<NormalDeriveEntry> entries;
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
        const auto normal_blocks = NormalPayloadBlocks(meshes, store_id);
        const auto vertex_blocks = PoseElementBlocks(meshes.Arenas().Vertices, record.Vertices);
        const auto face_blocks = PoseElementBlocks(meshes.Arenas().FaceTriangles, record.FaceData);
        const auto &morph = meshes.Arenas().Morph;
        for (uint32_t t = 0; t < target_count; ++t) {
            // Targets without position deltas use the rest pose and require no normal pinning.
            bool has_position_delta = false;
            meshes.Arenas().Vertices.ForEach(record.Vertices, [&](uint32_t vertex, uint32_t) {
                if (!has_position_delta && morph.Get(vertex, t).PositionDelta != vec3{0}) has_position_delta = true;
            });
            if (!has_position_delta) continue;
            auto entry = *entry_inputs;
            entry.Morph = meshes.Slots().Morph;
            entry.MorphTargetIndex = t;
            entry.VertexNormalNamespace = vertex_normals.Add(vertex_blocks);
            entry.SectorNamespace = sectors.Add(normal_blocks);
            entry.FaceNormalNamespace = face_normals.Add(face_blocks);
            entries.emplace_back(entry);
            job_entities.emplace_back(entity);
        }
    }
    if (entries.empty()) return;

    // Derivation reads each full-weight morph directly from canonical base vertices and target deltas.
    // No temporary position buffer or CPU geometry copy.
    EncodeDeriveNormals(r, chain, entries, MakeNormalDerivePc(buffers, meshes, true));
    chain.Submit();

    // Compare per mesh over its contiguous run of jobs.
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

void FinalizeNewMeshShading(state::Scene &r, mtl::ComputeChain &chain, std::span<const state::Entity> mesh_entities) {
    auto &meshes = r.Context.get<MeshStore>();
    std::vector<uint32_t> ids;
    ids.reserve(mesh_entities.size());
    for (const auto entity : mesh_entities)
        if (const auto mesh = TryGetMesh(r, entity)) ids.push_back(mesh->GetStoreId());
    EncodeDeriveAllNormals(r, chain, ids);
    // Authored corner normals encode as offsets from the derived ones, which the host reads.
    const auto authored = mesh_entities | std::views::filter([&](state::Entity entity) { return r.all_of<AuthoredCornerNormals>(entity); }) | std::ranges::to<std::vector>();
    if (!authored.empty()) chain.Submit();
    for (const auto entity : authored) {
        EncodeAuthoredCornerNormals(meshes, GetMesh(r, entity), r.get<const AuthoredCornerNormals>(entity).Corners);
        r.remove<AuthoredCornerNormals>(entity);
    }
    UpdateAuthoredMorphShading(r, chain, mesh_entities);
}

namespace {
void DispatchWork(MTL::ComputeCommandEncoder *encoder, const GpuBuffers &buffers, ElementWork work) {
    encoder->dispatchThreadgroups(*buffers.GeometryWork.Buffer, WorkArgsOffset(work), ThreadgroupSize::Linear256);
}

// Finalizes each work item over the scene bindings the encoder already holds.
void FinalizeWork(MTL::ComputeCommandEncoder *encoder, const Pipelines &pipelines, std::initializer_list<ElementWork> work) {
    encoder->setComputePipelineState(pipelines.FinalizeElementWork.State());
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
    scene.DisplayDirty.insert(entity);
    auto work_allocation = buffers.GeometryWork.BeginAllocation();
    MeshEditWork w{.StoreId = id};
    w.WorkBudget = buffers.GeometryWork.Allocate(3);
    w.Candidates = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Vertices = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Faces = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Normals = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.Meshlets = AllocateElementWork(buffers.GeometryWork, InvalidOffset);
    w.BoundsTiles = AllocateElementWork(buffers.GeometryWork, 1u << 24u);
    for (uint32_t level = 0u; level < w.BoundsLevels.size(); ++level)
        w.BoundsLevels[level] = AllocateElementWork(buffers.GeometryWork, 1u << (16u - level * 8u));
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
    scene->DisplayDirty.insert(entity);
}

namespace {
using GeometryEditJobs = std::vector<std::pair<state::Entity, CommitPosedGeometryPushConstants>>;

// Prepares each job's reusable topological footprint before any geometry writes, one phase for every job per submit.
// Stale candidates seed from the vertex selection in the first submit.
// Counts bound sparse allocation between the phases.
// Repeated parameter changes reuse the same tables.
void PrepareGeometryFootprints(state::Scene &r, mtl::ComputeChain &chain, GeometryEditJobs &jobs) {
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
            encode::BindScene(encoder, slots, buffers);
            for (const auto *work : seeded) FinalizeWork(encoder, pipelines, {work->Candidates});
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
            encode::BindScene(encoder, slots, buffers);
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
                    if (phase == 1u) FinalizeWork(encoder, pipelines, {pc.Faces, pc.Meshlets, pc.BoundsTiles});
                    else FinalizeWork(encoder, pipelines, {pc.Normals, pc.Meshlets});
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
        if (const auto *owner = meshes.TryGet(work.StoreId)) meshes.Render().ActiveMeshlets.ForEachBlock(owner->MeshletRoot, [&](uint32_t) { ++footprint.MeshletBlocks; });
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
        SetPoseNamespaces(entry, *pose, 0);
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
        .Primary = pending ? *WorldTransformOf(r, primary) : Transform{},
        .Delta = pending ? pending->Delta : Transform{},
        .Pivot = pending ? pending->Pivot : vec3{},
        .FaceTriangleStartSlot = meshes.Slots().FaceTriangleStart,
        .ElementMeshlets = {meshes.Render().ElementMeshlets[0].Ref(), meshes.Render().ElementMeshlets[1].Ref(), meshes.Render().ElementMeshlets[2].Ref()},
        .ApplyTransform = pending ? 1u : 0u,
        .Mode = !changed.empty() ? GeometryEditMode::Refresh : pose ? GeometryEditMode::Preview :
                                                                      GeometryEditMode::Commit,
    };
}

// Binds the scene on the encoder once, so later passes on it set only their pipelines.
void RecordGeometryEditBatch(state::Scene &r, MTL::ComputeCommandEncoder *encoder, GeometryEditJobs &commits, bool posed) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &pipelines = GetPipelines(r);
    const auto &slots = r.Context.get<const mtl::BindlessSet>();
    const auto entries = buffers.GeometryNormalEntries.SetCount<NormalDeriveEntry>(commits.size());
    for (uint32_t i = 0; i < commits.size(); ++i) entries[i] = commits[i].second.Entry;
    r.Context.get<const mtl::Context>().CommitResidency();
    encode::BindScene(encoder, slots, buffers);
    encoder->setComputePipelineState(pipelines.CommitPosedGeometry.State());
    for (auto &[_, pc] : commits) {
        // Posed outputs have no history. A commit writes the canonical base normals.
        if (!posed && pc.Entry.FaceCount) {
            auto entry = pc.Entry;
            entry.VerticesWork = pc.Normals;
            entry.FacesWork = pc.Faces;
            entry.VertexWorkCount = buffers.GeometryWork.Get({pc.Normals.Storage.Offset + 5u, 1u})[0];
            entry.FaceWorkCount = buffers.GeometryWork.Get({pc.Faces.Storage.Offset + 5u, 1u})[0];
            CaptureNormalWrites(r, entry, buffers.GeometryWork, buffers.GeometryWork);
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
// Faces keep their identities and triangle ranges. Canonical position edits discard
// affected authored tangents; those render keys and changed diagonals share one repair.
bool RetessellateGeometry(state::Scene &r, mtl::ComputeChain &chain, GeometryEditJobs &geometry, bool preview) {
    const auto &work = r.Context.get<const GpuBuffers>().GeometryWork;
    std::vector<MeshRetessellation> inputs;
    for (const auto &[entity, edit] : geometry) inputs.push_back({
        .StoreId = GetMesh(r, entity).GetStoreId(),
        .Work = &work,
        .Faces = edit.Faces,
        .Parameters = {.ChangedVertices = edit.ChangedVertices, .SelectionSlot = edit.SelectionSlot, .ApplyTransform = uint32_t(preview && edit.ApplyTransform), .Primary = edit.Primary, .Delta = edit.Delta, .Pivot = edit.Pivot},
        .Preview = preview,
    });
    const auto changed = RetessellateMeshes(r, chain, inputs);
    std::vector<std::pair<state::Entity, FaceTriangles>> repairs;
    for (uint32_t i = 0u; i < changed.size(); ++i)
        if (changed[i].Count) repairs.emplace_back(geometry[i].first, changed[i]);
    RepairFaceRender(r, chain, repairs);
    if (!repairs.empty()) {
        // New triangle owners invalidate the cached render footprint.
        auto &edit_work = r.Context.get<GpuSceneState>().EditWork;
        for (const auto &[entity, _] : repairs) edit_work.at(entity).FootprintReady = false;
        PrepareGeometryFootprints(r, chain, geometry);
    }
    return !repairs.empty();
}

} // namespace

void RefreshEditedPositions(state::Scene &r, mtl::ComputeChain &chain, std::span<const MeshVertexChanges> changes, bool retessellate) {
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
        auto &work = scene.EditWork.at(entity);
        work.Modified = work.PreviewActive = work.RequiresPose = true;
    }
}

bool RefreshPreviewTessellation(state::Scene &r, state::Entity viewport) {
    auto &scene = r.Context.get<GpuSceneState>();
    const auto *pending = r.try_get<const PendingTransform>(viewport);
    GeometryEditJobs jobs;
    for (const auto &[entity, primary] : r.get<const EditPrimaries>(viewport).Transformable) {
        const auto old = scene.EditWork.find(entity);
        if (!pending && (old == scene.EditWork.end() || !old->second.TessellationPreview)) continue;
        jobs.emplace_back(entity, PrepareGeometryEdit(r, entity, primary, pending));
    }
    if (jobs.empty()) return false;
    mtl::ComputeChain chain{r.Context.get<MeshStore>().BufferContext()};
    PrepareGeometryFootprints(r, chain, jobs);
    const bool changed = RetessellateGeometry(r, chain, jobs, true);
    for (const auto &[entity, _] : jobs) scene.EditWork.at(entity).TessellationPreview = pending != nullptr;
    chain.Submit();
    return changed;
}

std::vector<state::Entity> CommitPosedGeometry(state::Scene &r, mtl::ComputeChain &chain, state::Entity viewport, std::span<const state::Entity> mesh_entities) {
    const profile::CpuScope scope{"CommitGeometry"};
    const auto *pending = r.try_get<const PendingTransform>(viewport);
    if (!pending) return {};
    const auto &primaries = r.get<const EditPrimaries>(viewport).Transformable;
    GeometryEditJobs commits;
    for (const auto entity : mesh_entities) {
        if (const auto primary = primaries.find(entity); primary != primaries.end())
            commits.emplace_back(entity, PrepareGeometryEdit(r, entity, primary->second, pending));
    }
    if (commits.empty()) return {};
    PrepareGeometryFootprints(r, chain, commits);
    auto &meshes = r.Context.get<MeshStore>();
    for (const auto &[entity, pc] : commits) meshes.CaptureVertexEdit(GetMesh(r, entity).GetStoreId());
    chain.Encode([&](MTL::ComputeCommandEncoder *encoder) { RecordGeometryEditBatch(r, encoder, commits, false); });
    RetessellateGeometry(r, chain, commits, false);
    for (const auto &[entity, _] : commits) r.Context.get<GpuSceneState>().EditWork.at(entity).TessellationPreview = false;
    // The host reads which vertices the commit changed.
    chain.Submit();
    std::vector<state::Entity> entities;
    for (const auto &[entity, pc] : commits) entities.push_back(entity);
    return PublishEditedPositions(r, chain, entities);
}

std::vector<state::Entity> PublishEditedPositions(state::Scene &r, mtl::ComputeChain &chain, std::span<const state::Entity> entities, PositionPublication publication) {
    const bool preview = publication == PositionPublication::Preview;
    auto &meshes = r.Context.get<MeshStore>();
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &scene = r.Context.get<GpuSceneState>();
    std::vector<state::Entity> changed;
    std::vector<MeshStore::SelectionUpdate> aggregates;
    std::vector<MeshletBoundsRefitJob> refits;
    std::vector<MeshletIndexEdit> dirty_edits;
    std::vector<std::vector<uint32_t>> dirty_meshlets;
    std::vector<state::Entity> dirty_entities;
    for (const auto entity : entities) {
        auto &w = scene.EditWork.at(entity);
        const auto vertices = preview ? w.Candidates : w.Vertices;
        if (!ElementWorkEmpty(buffers.GeometryWork, vertices)) {
            changed.push_back(entity);
            const auto id = GetMesh(r, entity).GetStoreId();
            if (meshes.Get(id).SelectionSummary.Count) {
                auto &blocks = aggregates.emplace_back(MeshStore::SelectionUpdate{.StoreId = id}).Blocks[0];
                ForEachWorkBlock(buffers.GeometryWork, vertices, [&](uint32_t block, auto) { blocks.push_back(block); });
                ForEachWorkBlock(buffers.GeometryWork, w.Faces, [&](uint32_t block, auto) { aggregates.back().Blocks[2].push_back(block); });
            }
            w.Modified = w.PreviewActive = true;
            w.RequiresPose = !preview;
            const auto &owner = RecordOf(r, entity);
            refits.push_back({&owner, &buffers.GeometryWork, w.Meshlets});
            if (!preview && meshes.ClusterGroupCount(owner) > 0u && !ElementWorkEmpty(buffers.GeometryWork, w.Meshlets)) {
                dirty_edits.push_back({.Root = owner.PositionDirtyRoot});
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
    if (!dirty_edits.empty()) {
        for (uint32_t i = 0u; i < dirty_edits.size(); ++i) dirty_edits[i].Added = dirty_meshlets[i];
        meshes.Render().ActiveMeshlets.Update(dirty_edits);
        for (uint32_t i = 0u; i < dirty_edits.size(); ++i) {
            if (RecordOf(r, dirty_entities[i]).PositionDirtyRoot != dirty_edits[i].Root) EditRecordOf(r, dirty_entities[i]).PositionDirtyRoot = dirty_edits[i].Root;
            // Position-dirty meshlets pin their mesh's instances to finest geometry.
            scene.PositionDirty.insert(dirty_entities[i]);
            scene.DisplayDirty.insert(dirty_entities[i]);
        }
    }
    meshes.UpdateSelection(r, chain, aggregates);
    chain.AfterSubmit([&r, changed] { RefreshElementSelectionSummaries(r, changed); });
    return changed;
}

namespace {
void RecordSparseEditPrelude(state::Scene &r, state::Entity viewport, mtl::PassChain &chain) {
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &state = r.Context.get<GpuSceneState>();
    const auto &pipelines = GetPipelines(r);
    const auto *pending = r.try_get<const PendingTransform>(viewport);
    const auto &primaries = r.get<const EditPrimaries>(viewport).Transformable;
    GeometryEditJobs jobs;
    for (const auto &[entity, pose] : state.PosedByEntity) {
        const auto primary = primaries.find(entity);
        const bool preview = pending && primary != primaries.end();
        const auto old = state.EditWork.find(entity);
        if (!preview && (old == state.EditWork.end() || !old->second.PreviewActive)) continue;
        jobs.emplace_back(entity, PrepareGeometryEdit(r, entity, preview ? primary->second : state::Null, preview ? pending : nullptr, &pose));
    }
    if (jobs.empty()) return;
    {
        // The footprints submit before the frame records, since their counts size the edit work.
        mtl::ComputeChain footprints{r.Context.get<const MeshStore>().BufferContext()};
        PrepareGeometryFootprints(r, footprints, jobs);
        footprints.Submit();
    }
    auto *encoder = chain.BeginCompute("EditGeometry", MTL::StageDispatch);
    RecordGeometryEditBatch(r, encoder, jobs, true);
    // Each edited mesh's entry and work, with every bounds level dispatched for all meshes before the next level's barrier.
    struct EditBounds {
        const MeshEditWork &Work;
        uint32_t Entry, Instance;
    };
    std::vector<EditBounds> edits;
    edits.reserve(jobs.size());
    for (const auto &[entity, job] : jobs) {
        const auto &pose = state.PosedByEntity.at(entity);
        edits.push_back({state.EditWork.at(entity), state.BoundsRuns.at(entity).First, pose.FirstInstance});
    }
    for (const auto &edit : edits) RecordPosedMeshletBounds(encoder, pipelines, buffers, {.Work = edit.Work.Meshlets, .Instance = edit.Instance});
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

void SyncPreludeDispatchArgs(GpuBuffers &buffers) {
    const bool live = std::exchange(buffers.PreludeStale, false);
    buffers.MeshletOcclusionStale = false;
    std::array<MTL::DispatchThreadgroupsIndirectArguments, GpuBuffers::PreludePassCount> args;
    for (uint32_t i = 0u; i < args.size(); ++i) args[i] = {live ? buffers.PreludeGroups[i] : 0u, 1u, 1u};
    buffers.PreludeDispatchArgs.Update(as_bytes(args));
}

bool RefreshMeshDisplays(state::Scene &r, state::Entity viewport, std::span<const state::Entity> mesh_entities) {
    auto &scene_state = r.Context.get<GpuSceneState>();
    auto &buffers = r.Context.get<GpuBuffers>();
    const DisplayContext context{r, viewport};
    for (const auto mesh_entity : mesh_entities) {
        const auto *mb = r.valid(mesh_entity) ? TryRecordOf(r, mesh_entity) : nullptr;
        if (!mb) continue;
        // A posed mesh's display fields come with its pose layout.
        if (scene_state.PosedByEntity.contains(mesh_entity)) return true;
        // A static run's entry rereads the mesh's vertex selection root and recomputes its instances' bounds.
        const auto run = scene_state.BoundsRuns.find(mesh_entity);
        if (const auto *models = r.try_get<const ModelsBuffer>(mesh_entity); models && run != scene_state.BoundsRuns.end() && HasMesh(r, mesh_entity)) {
            buffers.BoundsReduceEntries.GetMutableSpan<BoundsEntry>({run->second.First, 1u})[0] = MeshBoundsEntry(context.Meshes, GetMesh(r, mesh_entity).GetStoreId(), *models);
            scene_state.DirtyBoundsEntries.push_back(run->second.First);
        }
        RefreshMeshDisplay(r, context, mesh_entity, *mb, NoDeform, nullptr);
    }
    return false;
}
