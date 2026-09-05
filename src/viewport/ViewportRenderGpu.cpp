#include "viewport/ViewportRenderGpu.h"
#include "Camera.h"
#include "ProcessEvents.h"
#include "Profile.h"
#include "Variant.h"
#include "animation/AnimationTimeline.h"
#include "animation/MorphWeightState.h"
#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "gizmo/TransformGizmoTypes.h"
#include "gpu/BoundsReducePushConstants.h"
#include "gpu/BoundsTreePushConstants.h"
#include "gpu/CommitPosedGeometryPushConstants.h"
#include "gpu/DepthPyramidReducePushConstants.h"
#include "gpu/ExtrasLineKind.h"
#include "gpu/MeshletCullPushConstants.h"
#include "gpu/MeshletDrawPushConstants.h"
#include "gpu/MeshletGeometryEncoding.h"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/MotionBlurGatherPushConstants.h"
#include "gpu/MotionBlurTilesDilatePushConstants.h"
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
#include "gpu/SilhouetteEdgeDepthObjectPushConstants.h"
#include "gpu/VisibilityId.h"
#include "gpu/WireRasterPushConstants.h"
#include "gpu/WireResolvePushConstants.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "metal/PassChain.h"
#include "metal/RenderTarget.h"
#include "numeric/Angles.h"
#include "physics/PhysicsTypes.h"
#include "render/ElementWorkOps.h"
#include "render/Encoding.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/Pipelines.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCamera.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportInteractionState.h"

#include <entt/entity/registry.hpp>

#include <cassert>
#include <cstdlib>
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

ExtrasLine ExtrasGizmoParams(const entt::registry &r, entt::entity object, ObjectType type) {
    constexpr auto HaloLines = uint32_t(OverlayDispatch::ExtrasHaloLines);
    constexpr auto RangeSegments = uint32_t(OverlayDispatch::LightRangeSegments);
    constexpr auto SpotSegments = uint32_t(OverlayDispatch::SpotConeSegments);
    if (type == ObjectType::Empty) return {ExtrasLineKind::Empty, {}, 3};
    if (type == ObjectType::Camera) {
        // Matches Blender's overlay at default drawsize 1: the frame spans one unit on its dominant axis.
        constexpr float HalfExtent{0.5f};
        const auto &camera = r.get<const Camera>(object);
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
    const auto angle_from_cos = [](float c) { return std::acos(std::clamp(c, -1.f, 1.f)); };
    const float outer_angle = std::min(angle_from_cos(light.OuterConeCos), numeric::Radians(89.f));
    const float inner_angle = std::min(angle_from_cos(light.InnerConeCos), outer_angle);
    const float outer_radius = SpotDepth * std::tan(outer_angle), inner_radius = SpotDepth * std::tan(inner_angle);
    const uint32_t inner_lines = inner_radius > 0.f ? SpotSegments : 0;
    return {
        ExtrasLineKind::LightSpot,
        vec4{light.Range, outer_radius, inner_radius, 0},
        range_lines + SpotSegments + inner_lines + SpotSegments + HaloLines,
    };
}

// Stores stable threadgroup chunks while the GPU filters dynamic settings and selection state at use time.
std::vector<OverlayJob> BuildOverlayJobs(const entt::registry &r) {
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
    if (buffers.MeshletRoutes.Contents().size() >= sizeof(MeshletRouteState)) {
        const auto &routes = *reinterpret_cast<const MeshletRouteState *>(buffers.MeshletRoutes.Contents().data());
        const auto count = [&](MeshletRoute route) { return routes.Counts[uint32_t(route)]; };
        profile::RecordCounter("VisibleOpaqueMeshlets", count(MeshletRoute::OpaqueCullBack) + count(MeshletRoute::OpaqueCullFront) + count(MeshletRoute::OpaqueDoubleSided) + count(MeshletRoute::Coverage));
        profile::RecordCounter("VisibleCoverageMeshlets", count(MeshletRoute::Coverage));
        profile::RecordCounter("SelectedCoarseMeshlets", *reinterpret_cast<const uint32_t *>(buffers.MeshletCoarseCount.Contents().data()));
        profile::RecordCounter("VisibleBlendMeshlets", count(MeshletRoute::Blend));
        profile::RecordCounter("VisibleTransmissionMeshlets", count(MeshletRoute::Transmission));
        static const bool record_routes = std::getenv("MESHEDITOR_MESHLET_ROUTE_COUNTERS") != nullptr;
        if (record_routes) {
            profile::RecordCounter("MeshletRoute OpaqueCullBack", count(MeshletRoute::OpaqueCullBack));
            profile::RecordCounter("MeshletRoute Blend", count(MeshletRoute::Blend));
            profile::RecordCounter("MeshletRoute Transmission", count(MeshletRoute::Transmission));
            profile::RecordCounter("MeshletRoute Phase2Candidate", count(MeshletRoute::Phase2Candidate));
            profile::RecordCounter("MeshletRoute OpaqueCullFront", count(MeshletRoute::OpaqueCullFront));
            profile::RecordCounter("MeshletRoute OpaqueDoubleSided", count(MeshletRoute::OpaqueDoubleSided));
            profile::RecordCounter("MeshletRoute Coverage", count(MeshletRoute::Coverage));
            profile::RecordCounter("MeshletRoute Overlay", count(MeshletRoute::Overlay));
        }
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
std::unordered_map<entt::entity, DeformSlots> BuildDeformSlots(const entt::registry &r, const MeshStore &meshes, RecordInputs &inputs) {
    std::unordered_map<entt::entity, DeformSlots> result;
    for (const auto [instance_entity, instance, modifier] : r.view<const Instance, const ArmatureModifier>().each()) {
        const auto &mesh = GetMesh(r, instance.Entity);
        const auto bone_deform = meshes.GetBoneDeformRange(mesh.GetStoreId());
        if (bone_deform.Count == 0) continue;
        const auto *pose_state = r.try_get<const ArmaturePoseState>(modifier.ArmatureEntity);
        if (!pose_state || modifier.SkinSlot >= pose_state->GpuDeformRanges.size()) continue;
        const auto deform_offset = pose_state->GpuDeformRanges[modifier.SkinSlot].Offset;
        auto &slots = result[instance.Entity];
        if (slots.BoneDeformOffset == InvalidOffset) {
            slots.BoneDeformOffset = bone_deform.Offset;
            slots.ArmatureDeformOffset = deform_offset;
        }
        if (const auto *ri = r.try_get<const RenderInstance>(instance_entity)) {
            slots.ArmatureDeformByBufferIndex[ri->BufferIndex] = deform_offset;
            inputs.Mix(ri->BufferIndex);
            inputs.Mix(deform_offset);
        }
    }
    for (const auto [instance_entity, instance, gpu_range, ri] : r.view<const Instance, const MorphWeightGpuRange, const RenderInstance>().each()) {
        const auto mesh_entity = instance.Entity;
        const auto &mesh = GetMesh(r, mesh_entity);
        const auto morph_range = meshes.GetMorphTargetRange(mesh.GetStoreId());
        if (morph_range.Count == 0) continue;
        auto &slots = result[mesh_entity];
        slots.MorphDeformOffset = morph_range.Offset;
        slots.MorphTargetCount = meshes.GetMorphTargetCount(mesh.GetStoreId());
        slots.MorphWeightsByBufferIndex[ri.BufferIndex] = gpu_range.Weights.Offset;
        inputs.Mix(ri.BufferIndex);
        inputs.Mix(gpu_range.Weights.Offset);
    }
    return result;
}

// Rewrite per-instance deform fields on `draws`, keyed by each draw's instance buffer index.
void PatchInstanceDeform(std::span<DrawData> draws, const DeformSlots &deform) {
    if (deform.MorphWeightsByBufferIndex.empty() && deform.ArmatureDeformByBufferIndex.empty()) return;
    for (auto &draw : draws) {
        if (auto it = deform.MorphWeightsByBufferIndex.find(draw.FirstInstance); it != deform.MorphWeightsByBufferIndex.end()) {
            draw.MorphWeightsOffset = it->second;
        }
        if (auto it = deform.ArmatureDeformByBufferIndex.find(draw.FirstInstance); it != deform.ArmatureDeformByBufferIndex.end()) {
            draw.ArmatureDeformOffset = it->second;
        }
    }
}
// Reduces and dilates tiled screen motion, then writes the blurred scene to GatherImage.
void RecordMotionBlurPostFx(entt::registry &r, mtl::PassChain &chain, const mtl::BindlessSet &slots, entt::entity viewport, mtl::Extent2D extent, uint32_t ubo_offset, float playback_frame) {
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &main = pipelines.Main;
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    const auto &settings = r.get<const ViewportDisplay>(viewport);
    const auto mb = EffectiveMotionBlur(settings);
    // The second half of each motion vector is stored pointing backward, which the negative y undoes.
    constexpr vec2 MotionScale{1.f, -1.f};
    // Golden-ratio stepping decorrelates the gather's dither across steps and frames.
    const float noise_phase = playback_frame * std::numbers::phi_v<float>;
    const float noise_offset = noise_phase - std::floor(noise_phase);

    const auto &buffers = r.ctx().get<const GpuBuffers>();
    auto *encoder = chain.BeginCompute("BlurTiles", MTL::StageFragment);
    const auto dispatch = [&](const mtl::ComputePipeline &compute, auto &&pc, uvec3 groups, MTL::Size threadgroup) {
        encode::BindCompute(encoder, compute, slots, buffers, ubo_offset);
        encode::SetPushConstants(encoder, pc);
        encoder->dispatchThreadgroups(MTL::Size(groups.x, groups.y, groups.z), threadgroup);
    };
    static constexpr auto divide_ceil = [](uint32_t v, uint32_t d) { return (v + d - 1) / d; };

    const auto tile_extent = main.MotionBlur->TileImage.Extent;
    { // One threadgroup per tile, which the flatten shader reduces to that tile's largest motion.
        encoder->setThreadgroupMemoryLength(ThreadgroupMemory::MotionBlurPayload, 0);
        encoder->setThreadgroupMemoryLength(ThreadgroupMemory::MotionBlurMaxMotion, 1);
        dispatch(
            pipelines.MotionBlurTilesFlatten,
            MotionBlurTilesFlattenPushConstants{sel_slots.VelocitySampler, sel_slots.MotionBlurTileImage, sel_slots.MotionBlurTileIndirection, MotionScale},
            {tile_extent.Width, tile_extent.Height, 1}, ThreadgroupSize::Tile8
        );
    }
    { // One thread per tile.
        dispatch(
            pipelines.MotionBlurTilesDilate,
            MotionBlurTilesDilatePushConstants{sel_slots.MotionBlurTileImage, sel_slots.MotionBlurTileIndirection},
            {divide_ceil(tile_extent.Width, 8), divide_ceil(tile_extent.Height, 8), 1}, ThreadgroupSize::Tile8
        );
    }

    { // One fullscreen pass, blurring the scene along its motion into the gather attachment.
        const std::array colors{mtl::DiscardColor(*main.MotionBlur->GatherImage)};
        const auto pass = mtl::MakePassDescriptor(colors);
        auto *render = encode::BeginScenePass(chain, pass, "BlurGather", {{MTL::StageFragment | MTL::StageDispatch, MTL::StageFragment}}, extent, slots, buffers, ubo_offset);
        main.MotionBlurGather.Bind(render);
        encode::SetPushConstants(render, MotionBlurGatherPushConstants{sel_slots.SceneDepthSampler, sel_slots.VelocitySampler, sel_slots.SceneColorSampler, sel_slots.MotionBlurTileImage, sel_slots.MotionBlurTileIndirection, MotionScale, mb.BleedingBias, noise_offset});
        render->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
    }
}

// Threadgroup memory lengths must be 16-byte multiples.
constexpr uint32_t AlignedThreadgroupBytes(uint32_t bytes) { return (bytes + 15u) & ~15u; }

// The tiled compute passes' threads per threadgroup.
constexpr uint32_t TileSize{256};
// Threadgroup count tiling `count` elements, min one so an empty entry still writes its outputs.
constexpr uint32_t TileCountFor(uint32_t count) { return std::max((count + TileSize - 1) / TileSize, 1u); }

// Slot of each prelude pass's args in GpuBuffers::PreludeDispatchArgs (PreludeGroups order).
enum class PreludeSlot : uint32_t { PosePrepass,
                                    PosedMeshletBounds,
                                    DeriveFaces,
                                    BoundsReduce,
                                    DeriveGather,
                                    BoundsCombine };

constexpr uint64_t PreludeArgsOffset(PreludeSlot slot) { return uint64_t(slot) * sizeof(MTL::DispatchThreadgroupsIndirectArguments); }

void WritePreludeArg(GpuBuffers &buffers, PreludeSlot slot, uint32_t groups) {
    const MTL::DispatchThreadgroupsIndirectArguments arg{groups, 1, 1};
    buffers.PreludeDispatchArgs.Update(as_bytes(arg), PreludeArgsOffset(slot));
}

// Record one prelude pass's dispatch, reading its group count from the pass's indirect args slot.
void DispatchPrelude(MTL::ComputeCommandEncoder *encoder, const GpuBuffers &buffers, PreludeSlot slot) {
    encoder->dispatchThreadgroups(*buffers.PreludeDispatchArgs, PreludeArgsOffset(slot), ThreadgroupSize::Linear256);
}

// The input fields of a mesh's normal-derive entry before assigning the position source and output offsets.
// Empty when the mesh has no triangles or adjacency.
std::optional<NormalDeriveEntry> MakeDeriveEntryInputs(const MeshStore &meshes, uint32_t store_id, SlottedRange face_indices) {
    if (face_indices.Count == 0) return {};
    const auto adjacency = meshes.GetVertexFanAdjacencyRange(store_id);
    if (adjacency.Count == 0) return {};
    const auto vertices = meshes.GetVerticesRange(store_id);
    const auto face_data = meshes.GetFaceDataRange(store_id);
    return NormalDeriveEntry{
        .Vertices = {vertices.Slot, vertices.Offset},
        .FaceIndices = face_indices,
        .VertexCount = vertices.Count,
        .VertexAdjacencyOffset = adjacency.Offset,
        .SeamFanOffset = meshes.GetSeamFanRange(store_id).Offset,
        .SeamCount = meshes.GetSeamCornerCount(store_id),
        .FaceDataOffset = face_data.Offset,
        .FaceCount = face_data.Count,
        .TriangleCount = meshes.GetTriangleCount(store_id),
    };
}

} // namespace

namespace {
// Materialize each posed entry's current-pose vertex positions.
void RecordPosePrepass(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const Pipelines &pipelines, const GpuBuffers &buffers, uint32_t ubo_offset) {
    const auto &prepass = pipelines.PosePrepass;
    encode::BindCompute(encoder, prepass, slots, buffers, ubo_offset);
    const BoundsReducePushConstants pc{
        .DrawDataSlot = buffers.BoundsReduceEntries.Slot,
        .TileMapSlot = buffers.BoundsTiles.Slot,
    };
    encode::SetPushConstants(encoder, pc);
    DispatchPrelude(encoder, buffers, PreludeSlot::PosePrepass);
}

// One derive dispatch over the tiles at `pc.FirstTile`, running the face or gather phase per pc.Phase.
// The tile count comes from `slot`'s indirect args.
void RecordNormalDerive(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const Pipelines &pipelines, const GpuBuffers &buffers, const NormalDerivePushConstants &pc, PreludeSlot slot, uint32_t ubo_offset) {
    const auto &pipeline = pipelines.VertexNormalDerive;
    encode::BindCompute(encoder, pipeline, slots, buffers, ubo_offset);
    encode::SetPushConstants(encoder, pc);
    DispatchPrelude(encoder, buffers, slot);
}

// The derive's shared input slots, plus the three output slots selecting the target buffers.
NormalDerivePushConstants MakeNormalDerivePc(const GpuBuffers &buffers, const MeshStore &meshes, uint32_t vertex_normal_slot, uint32_t seam_normal_slot, uint32_t face_normal_slot) {
    return {
        .EntriesSlot = buffers.NormalDeriveEntries.Slot,
        .AdjacencySlot = meshes.GetAdjacencySlot(),
        .TileMapSlot = buffers.DeriveTiles.Slot,
        .FaceFirstTriangleSlot = meshes.GetFaceFirstTriangleSlot(),
        .PositionSlot = buffers.PosedPositions.Slot,
        .VertexNormalSlot = vertex_normal_slot,
        .SeamNormalSlot = seam_normal_slot,
        .FaceNormalSlot = face_normal_slot,
    };
}

void RecordBoundsPass(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const mtl::ComputePipeline &pipeline, const GpuBuffers &buffers, PreludeSlot slot, uint32_t ubo_offset, BoundsReducePushConstants pc = {}) {
    pc.DrawDataSlot = buffers.BoundsReduceEntries.Slot;
    pc.BoundsSlot = buffers.Instances.BoundsBuffer.Slot;
    pc.TileMapSlot = buffers.BoundsTiles.Slot;
    pc.PartialBoundsSlot = buffers.BoundsPartials.Slot;
    pc.EntryFirstTileSlot = buffers.BoundsEntryFirstTiles.Slot;
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
    pc.DrawDataSlot = buffers.BoundsReduceEntries.Slot;
    pc.TileMapSlot = buffers.PosedMeshletBoundsTiles.Slot;
    pc.MeshletSlot = buffers.Meshlets.Buffer.Slot;
    pc.PrimitiveSlot = buffers.Primitives.Buffer.Slot;
    pc.MeshletVertexSlot = buffers.MeshletVertexCorners.Buffer.Slot;
    pc.PosedMeshletBoundsSlot = buffers.PosedMeshletBounds.Slot;
    encode::BindCompute(encoder, pipelines.PosedMeshletBounds, slots, buffers, ubo_offset);
    encode::SetPushConstants(encoder, pc);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::MeshletBoundsFoldVector, 0);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::MeshletBoundsFoldVector, 1);
    if (pc.Work.Storage.Slot == InvalidSlot)
        encoder->dispatchThreadgroups(*buffers.PreludeDispatchArgs, PreludeArgsOffset(PreludeSlot::PosedMeshletBounds), ThreadgroupSize::Linear64);
    else encoder->dispatchThreadgroups(*buffers.GeometryWork.Buffer, WorkArgsOffset(pc.Work, true), ThreadgroupSize::Linear64);
}

// The cull push constants' buffer-derived fields, shared by the phase-1 and phase-2 records.
MeshletCullPushConstants MakeMeshletCullSlotsPc(const GpuBuffers &buffers) {
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
        .VisibleSlot = buffers.VisibleMeshlets.Slot,
        .InstanceMapSlot = buffers.GpuInstanceSlots.Buffer.Slot,
        .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
        .PrimitiveSlot = buffers.Primitives.Buffer.Slot,
        .MeshletSlot = buffers.Meshlets.Buffer.Slot,
        .ClusterGroupSlot = buffers.ClusterGroups.Buffer.Slot,
        .BoundsSlot = buffers.Instances.BoundsBuffer.Slot,
        .ModelSlot = buffers.Instances.TransformBuffer.Slot,
        .PosedMeshletBoundsSlot = buffers.PosedMeshletBounds.Slot,
        .RouteStateSlot = buffers.MeshletRoutes.Slot,
        .DispatchArgsSlot = buffers.MeshletDispatchArgs.Slot,
        .DispatchChunkCount = buffers.MeshletDispatchChunkCount,
        .DispatchChunkSize = GpuBuffers::MeshletDispatchChunkSize,
        .OcclusionViewProj = buffers.PreviousFullCullViewProj,
        .Phase2VisibleSlot = buffers.MeshletPhase2Visible.Slot,
        .Phase2RouteStateSlot = buffers.MeshletPhase2Routes.Slot,
        .Phase2DispatchArgsSlot = buffers.MeshletPhase2DispatchArgs.Slot,
        .Phase2CullArgsSlot = buffers.MeshletPhase2CullArgs.Slot,
        .Phase2RangeCandidateSlot = buffers.MeshletPhase2RangeCandidates.Slot,
        .Phase2RangeCullArgsSlot = buffers.MeshletPhase2RangeCullArgs.Slot,
        .CoarseCountSlot = buffers.MeshletCoarseCount.Slot,
        .Phase2BlockCountSlot = buffers.MeshletPhase2CullBlockCounts.Slot,
    };
}

MeshletDrawPushConstants MakeMeshletDrawPc(
    const GpuBuffers &buffers, const mtl::Buffer &visible, const mtl::Buffer &routes,
    uint32_t route, uint32_t required_instance_flags, uint32_t visibility_phase,
    bool visibility_transmission, uint32_t edge_sharpness_slot,
    uint32_t edit_edge_corner = 0u, uint32_t instance_filter = InvalidOffset
) {
    return {
        .PrimitiveSlot = buffers.Primitives.Buffer.Slot,
        .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
        .InstanceMapSlot = buffers.GpuInstanceSlots.Buffer.Slot,
        .MeshletSlot = buffers.Meshlets.Buffer.Slot,
        .MeshletTriangleSlot = buffers.MeshletTriangleIds.Buffer.Slot,
        .MeshletVertexSlot = buffers.MeshletVertexCorners.Buffer.Slot,
        .MeshletLocalTriangleSlot = buffers.MeshletLocalTriangles.Buffer.Slot,
        .MeshletEditEdgeSlot = buffers.MeshletEditEdgeIds.Buffer.Slot,
        .VisibleMeshletSlot = visible.Slot,
        .RouteStateSlot = routes.Slot,
        .Route = route,
        .RequiredInstanceFlags = required_instance_flags,
        .InstanceFilter = instance_filter,
        .EditEdgeCorner = edit_edge_corner,
        .VisibilityPhase = visibility_phase,
        .VisibilityTransmission = visibility_transmission,
        .EdgeSharpnessSlot = edge_sharpness_slot,
    };
}

void DrawMeshletList(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, const mtl::Buffer &visible,
    const mtl::Buffer &routes, const mtl::Buffer &dispatch_args, uint32_t route, uint32_t required_instance_flags,
    uint32_t visibility_phase = 0u, bool visibility_transmission = false, bool fragment_pc = false,
    uint32_t edge_sharpness_slot = InvalidSlot,
    uint32_t mesh_threads = 160u, uint32_t edit_edge_corner = 0u,
    uint32_t instance_filter = InvalidOffset
) {
    // Visibility ids reserve 25 bits for the visible-list index, and overflowing aliases the phase bit.
    if (fragment_pc) {
        const auto visible_count = visible.UsedSize / sizeof(VisibleMeshlet);
        constexpr uint64_t index_limit = uint64_t{1} << uint32_t(VisibilityId::IndexBits);
        profile::RecordCounter("VisibleMeshletIndexOverflow", visible_count > index_limit ? double(visible_count - index_limit) : 0.0);
    }
    auto pc = MakeMeshletDrawPc(
        buffers, visible, routes, route, required_instance_flags, visibility_phase,
        visibility_transmission, edge_sharpness_slot, edit_edge_corner, instance_filter
    );
    for (uint32_t chunk = 0; chunk < buffers.MeshletDispatchChunkCount; ++chunk) {
        pc.VisibleOffset = chunk * GpuBuffers::MeshletDispatchChunkSize;
        if (fragment_pc) encode::SetPushConstants(encoder, pc);
        else encode::SetMeshPushConstants(encoder, pc);
        const auto args_offset = (route * buffers.MeshletDispatchChunkCount + chunk) * sizeof(MeshDispatchArgs);
        encoder->drawMeshThreadgroups(*dispatch_args, args_offset, MTL::Size(1, 1, 1), MTL::Size(mesh_threads, 1, 1));
    }
}

template<typename F>
void ForEachMeshletVisibilityList(const GpuBuffers &buffers, bool two_phase, F &&f) {
    f(buffers.VisibleMeshlets, buffers.MeshletRoutes, buffers.MeshletDispatchArgs);
    if (two_phase) f(buffers.MeshletPhase2Visible, buffers.MeshletPhase2Routes, buffers.MeshletPhase2DispatchArgs);
}

void DrawPhase2Meshlets(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, const MainPipeline &main
) {
    // Use one conservative two-sided coverage route before coarse candidates receive material classification.
    // Splitting its inner cull by route makes the serial prefix proportional to route count and regresses disocclusion-heavy scenes.
    encoder->setCullMode(MTL::CullModeNone);
    main.MeshletVisibilityCoverage.Bind(encoder);
    DrawMeshletList(
        encoder, buffers, buffers.MeshletPhase2Visible, buffers.MeshletPhase2Routes,
        buffers.MeshletPhase2DispatchArgs, uint32_t(MeshletRoute::OpaqueCullBack), 0u, 1u, false, true
    );
}

// Encodes every route with zero-sized dispatch arguments for routes without visible meshlets.
void DrawVisibilityMeshlets(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, const MainPipeline &main,
    bool transmission
) {
    const auto draw = [&](MeshletRoute route) {
        DrawMeshletList(
            encoder, buffers, buffers.VisibleMeshlets, buffers.MeshletRoutes, buffers.MeshletDispatchArgs,
            uint32_t(route), 0u, 0u, transmission, true
        );
    };
    main.MeshletVisibilityOpaque.Bind(encoder);
    encoder->setCullMode(MTL::CullModeBack);
    draw(MeshletRoute::OpaqueCullBack);
    encoder->setCullMode(MTL::CullModeFront);
    draw(MeshletRoute::OpaqueCullFront);
    encoder->setCullMode(MTL::CullModeNone);
    draw(MeshletRoute::OpaqueDoubleSided);
    main.MeshletVisibilityCoverage.Bind(encoder);
    draw(MeshletRoute::Coverage);
}

void RecordMeshletPhase2Cull(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines,
    GpuBuffers &buffers, uint32_t pyramid_sampler, uint32_t ubo_offset, MeshletRouteMode mode
) {
    ++buffers.MeshletVisibleGeneration;
    auto pc = MakeMeshletCullSlotsPc(buffers);
    pc.RouteMode = uint32_t(mode);
    pc.PyramidSamplerSlot = pyramid_sampler;
    pc.TwoPhase = 1u;
    auto *encoder = chain.BeginCompute("MeshletPhase2Cull", MTL::StageDispatch | MTL::StageFragment);
    encode::BindScene(encoder, slots, buffers, ubo_offset);
    const auto group = MTL::Size(GpuBuffers::MeshletPhase2GroupSize, 1, 1);
    // Count pass, prefix into deterministic offsets, then the emit pass with the same tests.
    const auto dispatch_culls = [&] {
        encode::SetPushConstants(encoder, pc);
        encoder->setComputePipelineState(pipelines.MeshletPhase2Cull.State());
        encoder->dispatchThreadgroups(*buffers.MeshletPhase2CullArgs, 0, group);
        encoder->setComputePipelineState(pipelines.MeshletPhase2RangeCull.State());
        encoder->dispatchThreadgroups(*buffers.MeshletPhase2RangeCullArgs, 0, group);
    };
    dispatch_culls();
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    encoder->setComputePipelineState(pipelines.MeshletPhase2Prefix.State());
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    pc.Phase2Emit = 1u;
    dispatch_culls();
}

void RecordDepthPyramid(
    MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const GpuBuffers &buffers,
    const Pipelines &pipelines, const SelectionSlots &sel_slots, uint32_t ubo_offset
) {
    const auto &main = pipelines.Main;
    encode::BindCompute(encoder, pipelines.DepthPyramidReduce, slots, buffers, ubo_offset);
    const auto &mips = main.Resources->DepthPyramidMips;
    const auto scene_extent = main.Resources->DepthImage.Extent;
    for (uint32_t base = 0; base < uint32_t(mips.size()); base += 6) {
        // Add an explicit barrier between bindless mip dependencies.
        if (base > 0) encoder->memoryBarrier(MTL::BarrierScopeTextures);
        const auto src_extent = base == 0 ? scene_extent : mips[base - 1].Extent;
        const DepthPyramidReducePushConstants pc{
            .SrcSamplerSlot = base == 0 ? sel_slots.SceneDepthSampler : sel_slots.DepthPyramidSampler,
            .SrcLod = base == 0 ? 0 : base - 1,
            .SrcWidth = src_extent.Width,
            .SrcHeight = src_extent.Height,
            .DstSlots = [&] {
                std::array<uint32_t, 6> dst;
                for (uint32_t k = 0; k < dst.size(); ++k) dst[k] = base + k < mips.size() ? mips[base + k].Slot : InvalidSlot;
                return dst;
            }(),
        };
        encode::SetPushConstants(encoder, pc);
        encoder->setThreadgroupMemoryLength(ThreadgroupMemory::DepthPyramidTile, 0);
        encoder->dispatchThreadgroups(
            MTL::Size((mips[base].Extent.Width + 31) / 32, (mips[base].Extent.Height + 31) / 32, 1),
            ThreadgroupSize::Tile16
        );
    }
}

void RecordSparseEditPrelude(entt::registry &, entt::entity, mtl::PassChain &);

// Record one phase's passes into `cb`, which is already begun with viewport and scissor set.
// `ubo_offset` selects the view UBO instance every bind in the phase reads.
void RecordPhase(entt::registry &r, entt::entity viewport, mtl::PassChain &chain, SceneUpdate update, RenderPhase phase, uint32_t ubo_offset, float playback_frame) {
    const profile::CpuScope scope{"RecordRenderCommandBuffer"};
    // Multi-step blur separates scene accumulation from sharp overlay rendering.
    const bool draw_scene = phase != RenderPhase::BlurResolve;
    const bool draw_overlays = !IsBlurAccumulate(phase);

    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &pipelines = r.ctx().get<Pipelines>();
    const auto &settings = r.get<const ViewportDisplay>(viewport);
    const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
    const auto edit_mode = r.get<const EditMode>(viewport).Value;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;
    const bool is_excite_mode = interaction_mode == InteractionMode::Excite;
    const bool is_wireframe_mode = settings.ViewportShading == ViewportShadingMode::Wireframe;
    const bool show_rendered = settings.ViewportShading == ViewportShadingMode::MaterialPreview || settings.ViewportShading == ViewportShadingMode::Rendered;
    const bool show_fill = !is_wireframe_mode;
    const bool show_overlays = settings.ShowOverlays;
    const auto &active_lighting = GetActivePbrLighting(r, viewport, settings.ViewportShading);
    const bool real_transmission = show_rendered &&
        active_lighting.RealTransmission &&
        pipelines.Main.Compiler.HasFeature(PbrFeature::Transmission);

    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &scene_state = r.ctx().get<GpuSceneState>();

    RecordInputs record_inputs;
    record_inputs.Mix(uint32_t(interaction_mode) | uint32_t(edit_mode) << 8u);
    record_inputs.Mix(buffers.Instances.RecordBuffer.Count<InstanceRecord>());
    // Edit mode uses the rest pose, and reused blur phases use slots built by the rebuild phase.
    const auto mesh_deform_slots = is_edit_mode || update == SceneUpdate::Reuse ?
        std::unordered_map<entt::entity, DeformSlots>{} :
        BuildDeformSlots(r, meshes, record_inputs);
    static const DeformSlots no_deform{};
    const auto get_deform_slots = [&](entt::entity mesh_entity) -> const DeformSlots & {
        if (auto it = mesh_deform_slots.find(mesh_entity); it != mesh_deform_slots.end()) return it->second;
        return no_deform;
    };

    const auto is_silhouette_eligible = [&](entt::entity e) {
        if (!r.all_of<Instance, RenderInstance>(e)) return false;
        const auto buffer_entity = r.get<const Instance>(e).Entity;
        if (!r.valid(buffer_entity) || r.all_of<ObjectExtrasTag>(buffer_entity)) return false;
        // Bones get outlines from BoneWire/BoneSphereWire, not the screen-space silhouette system.
        if (r.all_of<ArmatureObject>(buffer_entity) || r.all_of<BoneJoint>(buffer_entity)) return false;
        const auto *mesh_buffers = r.try_get<const MeshBuffers>(buffer_entity);
        return mesh_buffers && mesh_buffers->FaceIndices.Count > 0;
    };
    const auto should_draw_armature_bones = [&](entt::entity armature) {
        if (is_wireframe_mode) return true;
        if (is_edit_mode || interaction_mode == InteractionMode::Pose) return r.all_of<Active>(armature);
        return r.all_of<Selected>(armature);
    };
    const bool show_normals = show_overlays && settings.NormalOverlays != 0u;
    const auto normal_meshes = show_normals ?
        selection::GetSelectedMeshEntities(r) :
        std::unordered_set<entt::entity>{};
    const bool show_face_normals = show_normals &&
        he::ElementMaskContains(settings.NormalOverlays, Element::Face);
    const bool show_vertex_normals = show_normals &&
        he::ElementMaskContains(settings.NormalOverlays, Element::Vertex);
    std::unordered_set<entt::entity> sound_meshes;
    if (is_excite_mode) {
        for (const auto entity : r.view<const Instance, const SoundVertices>()) {
            sound_meshes.insert(r.get<const Instance>(entity).Entity);
        }
    }

    selection::PrimaryEditInstanceMap primary_edit_instances;
    EditTransformContext edit_transform_context;
    const bool has_pending_transform = is_edit_mode && r.all_of<PendingTransform>(viewport);
    record_inputs.Mix(has_pending_transform);
    if (is_edit_mode) {
        if (has_pending_transform) {
            auto primaries = selection::ComputePrimaryEditInstanceMaps(r);
            primary_edit_instances = std::move(primaries.All);
            edit_transform_context.TransformInstances = std::move(primaries.Transformable);
        } else {
            primary_edit_instances = selection::ComputePrimaryEditInstances(r);
        }
    }
    std::unordered_set<entt::entity> silhouette_instances;
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
            entt::entity Entity;
            const MeshBuffers &Buf;
            const ModelsBuffer &Mod;
            std::optional<Mesh> MeshComp;
            const DeformSlots &Deform;
            std::optional<uint32_t> PrimaryEditBufferIndex;
        };

        // Sort by descending entity ID for deterministic coincident-surface ordering across scene loads.
        auto mesh_entity_order = r.view<const MeshBuffers, const ModelsBuffer>() | to<std::vector>();
        std::ranges::sort(mesh_entity_order, std::ranges::greater{});

        std::vector<MeshEntityData> mesh_entities;
        mesh_entities.reserve(mesh_entity_order.size());
        for (const auto entity : mesh_entity_order) {
            const auto &mesh_buffers = r.get<const MeshBuffers>(entity);
            const auto &models = r.get<const ModelsBuffer>(entity);
            std::optional<uint32_t> primary_bi;
            if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                primary_bi = r.get<RenderInstance>(it->second).BufferIndex;
            }
            mesh_entities.emplace_back(
                entity, mesh_buffers, models, TryGetMesh(r, entity), get_deform_slots(entity), primary_bi
            );
        }

        // The mesh shades authored under morphing: rest normals plus weighted authored deltas.
        // Edit mode builds no deform slots, so edit-mode draws (including drags) derive.
        const auto morph_shading_authored = [&meshes](const MeshEntityData &e) {
            return e.Deform.MorphDeformOffset != InvalidOffset && e.MeshComp && meshes.GetMorphShadingAuthored(e.MeshComp->GetStoreId());
        };

        { // Bounds reduce entries.
            // Instances sharing one deform state share one entry, whose ElementIdOffset spans their consecutive slots.
            // Entries with morph, armature, or pending edit-transform deformation come first.
            // Each has a posed-position range the pose pre-pass materializes ahead of the bounds reduction.
            struct BoundsEntrySpec {
                uint32_t Count{};
                bool PerInstanceDeform{}, Posed{}, Derive{};
                const RenderInstance *PendingPrimary{};
                NormalDeriveEntry Entry{}; // Derive-input fields, filled when Derive.
                uint32_t Level0Count{}; // Original clusters the entry's posed bounds cover, filled when Posed.
            };
            std::vector<BoundsEntrySpec> specs(mesh_entities.size());
            // Posed bounds cover a mesh's original clusters alone, concatenated in primitive order.
            const auto mesh_level0_count = [&buffers](const MeshEntityData &e) {
                uint32_t count = 0;
                for (const auto &primitive : buffers.Primitives.Get(e.Buf.Primitives)) count += primitive.Level0Count;
                return count;
            };
            // Posed entries and their tiles come first: the pose pre-pass dispatches over that tile prefix.
            uint32_t entry_count = 0, posed_entry_count = 0, posed_vertex_count = 0;
            uint32_t derive_entry_count = 0, vertex_normal_count = 0, seam_normal_count = 0, face_normal_count = 0;
            uint32_t posed_tile_count = 0, bounds_tile_count = 0, derive_face_tile_count = 0, derive_gather_tile_count = 0;
            uint32_t posed_meshlet_bounds_count = 0;
            bool authored_morph_any = false;
            RecordInputs prelude_layout;
            for (size_t mi = 0; mi < mesh_entities.size(); ++mi) {
                const auto &e = mesh_entities[mi];
                auto &spec = specs[mi];
                // Every mesh-keyed value an instance record reads, in the order the meshes come.
                record_inputs.Mix(entt::to_integral(e.Entity));
                record_inputs.Mix(e.Buf.Primitives.Offset);
                record_inputs.Mix(e.Buf.Primitives.Count);
                record_inputs.Mix(e.Buf.Meshlets.Offset);
                record_inputs.Mix(e.Buf.Meshlets.Count);
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
                if (e.Buf.Meshlets.Count != 0u) {
                    record_inputs.Mix(buffers.Meshlets.Buffer.GetSpan<MeshletRecord>({e.Buf.Meshlets.Offset, 1}).front().LocalTriangleOffset);
                }
                if (!e.MeshComp || e.Mod.InstanceCount == 0) continue;
                if (has_pending_transform) {
                    if (const auto it = edit_transform_context.TransformInstances.find(e.Entity); it != edit_transform_context.TransformInstances.end()) {
                        spec.PendingPrimary = r.try_get<const RenderInstance>(it->second);
                    }
                }
                spec.PerInstanceDeform = !e.Deform.ArmatureDeformByBufferIndex.empty() || !e.Deform.MorphWeightsByBufferIndex.empty();
                spec.Count = spec.PerInstanceDeform ? e.Mod.InstanceCount : 1u;
                spec.Posed = e.Deform.BoneDeformOffset != InvalidOffset || e.Deform.MorphDeformOffset != InvalidOffset ||
                    (is_edit_mode && (e.PrimaryEditBufferIndex.has_value() || scene_state.EditWork.contains(e.Entity)));
                entry_count += spec.Count;
                bounds_tile_count += spec.Count * TileCountFor(e.Buf.Vertices.Count);
                if (spec.Posed) {
                    // Authored morph shading reads base normals.
                    const bool authored_morph = morph_shading_authored(e);
                    authored_morph_any |= authored_morph;
                    if (const auto derive_entry = authored_morph ? std::nullopt : MakeDeriveEntryInputs(meshes, e.MeshComp->GetStoreId(), e.Buf.FaceIndices)) {
                        spec.Derive = true;
                        spec.Entry = *derive_entry;
                        derive_entry_count += spec.Count;
                        derive_face_tile_count += spec.Count * TileCountFor(spec.Entry.FaceCount);
                        derive_gather_tile_count += spec.Count * TileCountFor(spec.Entry.VertexCount + spec.Entry.SeamCount);
                        vertex_normal_count += spec.Count * spec.Entry.VertexCount;
                        seam_normal_count += spec.Count * spec.Entry.SeamCount;
                        face_normal_count += spec.Count * spec.Entry.FaceCount;
                    }
                    posed_entry_count += spec.Count;
                    posed_tile_count += spec.Count * TileCountFor(e.Buf.Vertices.Count);
                    posed_vertex_count += spec.Count * e.Buf.Vertices.Count;
                    spec.Level0Count = mesh_level0_count(e);
                    posed_meshlet_bounds_count += spec.Count * spec.Level0Count;
                }
                prelude_layout.Mix(entt::to_integral(e.Entity));
                prelude_layout.Mix(spec.Count);
                prelude_layout.Mix(uint32_t(spec.Posed) | uint32_t(spec.Derive) << 1u);
                prelude_layout.Mix(e.Buf.Vertices.Count);
                prelude_layout.Mix(spec.Entry.FaceCount);
                prelude_layout.Mix(spec.Entry.SeamCount);
                prelude_layout.Mix(e.Mod.InstanceRange.Offset);
                prelude_layout.Mix(e.Mod.InstanceCount);
                for (const auto &primitive : buffers.Primitives.Get(e.Buf.Primitives)) {
                    prelude_layout.Mix(primitive.MeshletOffset);
                    prelude_layout.Mix(primitive.Level0Count);
                }
                // The posed-buffer offsets a record reads follow from these, given the mesh order above.
                record_inputs.Mix(spec.Count);
                record_inputs.Mix(uint32_t(spec.Posed) | uint32_t(spec.Derive) << 1u | uint32_t(spec.PerInstanceDeform) << 2u);
                record_inputs.Mix(spec.PendingPrimary ? spec.PendingPrimary->BufferIndex : InvalidOffset);
                record_inputs.Mix(spec.Entry.SeamCount);
                record_inputs.Mix(spec.Entry.FaceCount);
                record_inputs.Mix(spec.Entry.VertexCount);
            }
            const bool tiles_changed = prelude_layout.Value != scene_state.PreludeLayoutInputs;
            scene_state.PreludeLayoutInputs = prelude_layout.Value;
            if (tiles_changed) buffers.PreludeStale = true;
            const auto entries = buffers.BoundsReduceEntries.SetCount<DrawData>(entry_count);
            const auto derive_entries = buffers.NormalDeriveEntries.SetCount<NormalDeriveEntry>(derive_entry_count);
            const auto bounds_tiles = buffers.BoundsTiles.SetCount<uvec2>(bounds_tile_count);
            const auto derive_tiles = buffers.DeriveTiles.SetCount<uvec2>(derive_face_tile_count + derive_gather_tile_count);
            const auto entry_first_tiles = buffers.BoundsEntryFirstTiles.SetCount<uint32_t>(entry_count);
            buffers.BoundsPartials.SetCount<AABB>(bounds_tile_count);
            buffers.PosedPositions.SetCount<vec3>(posed_vertex_count);
            const auto posed_meshlet_tiles = buffers.PosedMeshletBoundsTiles.SetCount<uvec2>(posed_meshlet_bounds_count);
            buffers.PosedMeshletBounds.SetCount<AABB>(posed_meshlet_bounds_count);
            // Authored-morph entries index their deltas by posed-position offset.
            // The buffer spans the full posed range whenever any authored-morph entry exists.
            buffers.PosedMorphNormalDeltas.SetCount<vec3>(authored_morph_any ? posed_vertex_count : 0u);
            buffers.PosedVertexNormals.SetCount<vec3>(vertex_normal_count);
            buffers.PosedSeamNormals.SetCount<vec3>(seam_normal_count);
            buffers.PosedFaceNormals.SetCount<vec3>(face_normal_count);
            buffers.Prelude = {
                .PosePrepass = posed_tile_count,
                .PosedMeshletBounds = posed_meshlet_bounds_count,
                .DeriveFaces = derive_face_tile_count,
                .BoundsReduce = bounds_tile_count,
                .DeriveGather = derive_gather_tile_count,
                .BoundsCombine = entry_count,
            };

            uint32_t posed_write = 0, unposed_write = posed_entry_count, derive_write = 0;
            uint32_t posed_tile_write = 0, unposed_tile_write = posed_tile_count;
            uint32_t face_tile_write = 0, gather_tile_write = derive_face_tile_count;
            uint32_t posed_offset = 0, vertex_normal_offset = 0, seam_normal_offset = 0, face_normal_offset = 0;
            uint32_t meshlet_bounds_offset = 0, meshlet_tile_write = 0;
            for (size_t mi = 0; mi < mesh_entities.size(); ++mi) {
                const auto &e = mesh_entities[mi];
                const auto &spec = specs[mi];
                if (spec.Count == 0) continue;
                auto &write = spec.Posed ? posed_write : unposed_write;
                DrawData entry{
                    .VertexSlot = e.Buf.Vertices.Slot,
                    .ModelSlot = buffers.Instances.TransformBuffer.Slot,
                    .FirstInstance = e.Mod.InstanceRange.Offset,
                    .VertexCountOrHeadImageSlot = e.Buf.Vertices.Count,
                    .ElementIdOffset = spec.PerInstanceDeform ? 1u : e.Mod.InstanceCount,
                    .Selection = meshes.GetEditSelectionStorage(e.MeshComp->GetStoreId()),
                    .VertexOffset = e.Buf.Vertices.Offset,
                    .BoneDeformOffset = e.Deform.BoneDeformOffset,
                    .ArmatureDeformOffset = e.Deform.ArmatureDeformOffset,
                    .MorphDeformOffset = e.Deform.MorphDeformOffset,
                    .MorphTargetCount = e.Deform.MorphTargetCount,
                    .MorphShadingAuthored = morph_shading_authored(e) ? 1u : 0u,
                };
                if (spec.PendingPrimary) {
                    entry.HasPendingVertexTransform = 1u;
                    entry.PrimaryEditInstanceIndex = spec.PendingPrimary->BufferIndex;
                }
                // PosedRanges defines bases and per-instance offsets for the posed-buffer layout.
                PosedRanges pr{};
                NormalDeriveEntry derive_entry = spec.Entry;
                if (spec.Posed) {
                    pr = {
                        .FirstInstance = e.Mod.InstanceRange.Offset,
                        .PerInstance = spec.PerInstanceDeform,
                        .PositionBase = posed_offset,
                        .VertexCount = e.Buf.Vertices.Count,
                        .MeshletBoundsBase = meshlet_bounds_offset,
                        .Level0Count = spec.Level0Count,
                        .Normals = spec.Derive ?
                            std::optional{PosedRanges::NormalRanges{vertex_normal_offset, seam_normal_offset, face_normal_offset, spec.Entry.SeamCount, spec.Entry.FaceCount}} :
                            std::nullopt,
                    };
                    scene_state.PosedByEntity.emplace(e.Entity, pr);
                    posed_offset += spec.Count * pr.VertexCount;
                    meshlet_bounds_offset += spec.Count * pr.Level0Count;
                }
                if (spec.Derive) {
                    vertex_normal_offset += spec.Count * spec.Entry.VertexCount;
                    seam_normal_offset += spec.Count * spec.Entry.SeamCount;
                    face_normal_offset += spec.Count * spec.Entry.FaceCount;
                }
                const auto first = write;
                const auto bounds_tiles_per = TileCountFor(e.Buf.Vertices.Count);
                const auto face_tiles_per = TileCountFor(spec.Entry.FaceCount);
                const auto gather_tiles_per = TileCountFor(spec.Entry.VertexCount + spec.Entry.SeamCount);
                auto &tile_write = spec.Posed ? posed_tile_write : unposed_tile_write;
                for (uint32_t i = 0; i < spec.Count; ++i) {
                    if (spec.PerInstanceDeform) entry.FirstInstance = e.Mod.InstanceRange.Offset + i;
                    if (spec.Posed) entry.PosedPositionOffset = pr.PositionOffset(i);
                    if (const auto normals = pr.NormalsAt(i)) {
                        derive_entry.PosedPositionOffset = entry.PosedPositionOffset;
                        derive_entry.VertexNormalOffset = normals->VertexOffset;
                        derive_entry.SeamNormalOffset = normals->SeamOffset;
                        derive_entry.FaceNormalOffset = normals->FaceOffset;
                        if (tiles_changed) {
                            for (uint32_t t = 0; t < face_tiles_per; ++t) derive_tiles[face_tile_write++] = {derive_write, t};
                            for (uint32_t t = 0; t < gather_tiles_per; ++t) derive_tiles[gather_tile_write++] = {derive_write, t};
                        }
                        derive_entries[derive_write++] = derive_entry;
                    }
                    entry_first_tiles[write] = tile_write;
                    if (tiles_changed) {
                        for (uint32_t t = 0; t < bounds_tiles_per; ++t) bounds_tiles[tile_write++] = {write, t};
                    } else tile_write += bounds_tiles_per;
                    entries[write++] = entry;
                }
                // Shared-pose instances share one entry and one set of meshlet bounds, like positions.
                if (spec.Posed && tiles_changed) {
                    const auto mesh_primitives = buffers.Primitives.Get(e.Buf.Primitives);
                    for (uint32_t i = 0; i < spec.Count; ++i) {
                        for (const auto &primitive : mesh_primitives) {
                            for (uint32_t m = 0; m < primitive.Level0Count; ++m) {
                                posed_meshlet_tiles[meshlet_tile_write++] = {first + i, primitive.MeshletOffset + m};
                            }
                        }
                    }
                }
                if (spec.PerInstanceDeform) PatchInstanceDeform(entries.subspan(first, spec.Count), e.Deform);
            }
        }

        // Reuse records and the topology mask when all hashed inputs match the previous rebuild.
        if (record_inputs.Value != scene_state.InstanceRecordInputs) {
            scene_state.InstanceRecordInputs = record_inputs.Value;
            MarkInstanceRecordsStale(scene_state);
        }
        if (scene_state.InstanceRecordsStale) {
            buffers.MeshletTopologyMask = 0u;
            for (const auto [instance_entity, instance, ri] : r.view<const Instance, const RenderInstance>().each()) {
                if (ri.BufferIndex == UINT32_MAX) continue;
                const auto *mesh_buffers = r.try_get<const MeshBuffers>(instance.Entity);
                if (!mesh_buffers || mesh_buffers->Primitives.Count == 0) continue;
                if (mesh_buffers->Meshlets.Count != 0u) {
                    const MeshletRecord &first_meshlet = buffers.Meshlets.Buffer.GetSpan<MeshletRecord>(
                                                                                    {mesh_buffers->Meshlets.Offset, mesh_buffers->Meshlets.Count}
                    )
                                                             .front();
                    const uint32_t topology = first_meshlet.LocalTriangleOffset >> uint32_t(MeshletGeometryEncoding::TopologyShift);
                    buffers.MeshletTopologyMask |= 1u << topology;
                }
                InstanceRecord record{
                    .PrimitiveOffset = mesh_buffers->Primitives.Offset,
                    .PrimitiveCount = mesh_buffers->Primitives.Count,
                    .ObjectId = ri.ObjectId,
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
                    record.PosedPositionOffset = posed.PositionOffset(i);
                    record.PosedMeshletBoundsOffset = posed.MeshletBoundsOffset(i);
                    if (const auto normals = posed.NormalsAt(i)) {
                        record.PosedVertexNormalOffset = normals->VertexOffset;
                        record.PosedSeamNormalOffset = normals->SeamOffset;
                        record.PosedFaceNormalOffset = normals->FaceOffset;
                    }
                }
                const auto primary = primary_edit_instances.find(instance.Entity);
                if (has_pending_transform && primary != primary_edit_instances.end()) {
                    record.HasPendingVertexTransform = 1u;
                    record.PrimaryEditInstanceIndex = r.get<const RenderInstance>(primary->second).BufferIndex;
                }
                if (primary != primary_edit_instances.end() && primary->second == instance_entity) {
                    const uint32_t store_id = r.get<const MeshHandle>(instance.Entity).StoreId;
                    record.Selection = meshes.GetEditSelectionStorage(store_id);
                    record.EditEdgeSharpnessOffset = meshes.GetEdgeSharpnessRange(store_id).Offset;
                    record.ElementIdOffset = meshes.GetSelectionBitOffset(store_id, edit_mode);
                } else if (is_excite_mode && sound_meshes.contains(instance.Entity)) {
                    const uint32_t store_id = r.get<const MeshHandle>(instance.Entity).StoreId;
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
                if (!e.PrimaryEditBufferIndex || !e.MeshComp || e.Buf.Meshlets.Count == 0u) continue;
                scene_state.MeshletEditOverlayMeshes.insert(e.Entity);
                const auto sharpness = meshes.GetEdgeSharpness(e.MeshComp->GetStoreId());
                scene_state.MeshletEditHasSharpEdges |= std::memchr(sharpness.data(), 1, sharpness.size()) != nullptr;
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
                 MeshletInstanceFlag::VertexNormal,
                 MeshletInstanceFlag::EdgeOverlay,
                 MeshletInstanceFlag::PointOverlay,
                 MeshletInstanceFlag::SoundPoint,
             }) {
            buffers.FlagWork(uint32_t(flag)) = {};
        }
        for (const auto [instance_entity, ri] : r.view<const RenderInstance>().each()) {
            if (ri.BufferIndex == UINT32_MAX || ri.BufferIndex >= instance_records.size()) continue;
            auto &record = instance_records[ri.BufferIndex];
            record.ObjectId = ri.ObjectId;
            const bool selected = r.all_of<Selected>(instance_entity) && is_silhouette_eligible(instance_entity);
            const bool silhouette = selected && (!is_edit_mode || silhouette_instances.contains(instance_entity));
            record.Flags = silhouette ? uint32_t(MeshletInstanceFlag::Silhouette) : 0u;
            // Every instance of an edited mesh draws original geometry, since an element pick can land on any of them.
            const auto *instance = r.try_get<const Instance>(instance_entity);
            if (instance && (primary_edit_instances.contains(instance->Entity) || scene_state.EditWork.contains(instance->Entity))) {
                record.Flags |= uint32_t(MeshletInstanceFlag::LodPinFinest);
            }
            const auto primary = instance ? primary_edit_instances.find(instance->Entity) : primary_edit_instances.end();
            const auto *mesh_buffers = instance ? r.try_get<const MeshBuffers>(instance->Entity) : nullptr;
            if (instance && mesh_buffers && mesh_buffers->Meshlets.Count > 0u &&
                primary != primary_edit_instances.end() && primary->second == instance_entity &&
                selection::GetElementCount(GetMesh(r, instance->Entity), edit_mode) > 0u) {
                record.Flags |= uint32_t(MeshletInstanceFlag::ElementSelection);
                if (ri.GpuId != InvalidOffset) {
                    element_selection_work.Ranges += ri.MeshletRangeCount;
                    element_selection_work.Meshlets += ri.MeshletCount;
                }
            }
            if (instance && scene_state.MeshletEditOverlayMeshes.contains(instance->Entity) &&
                primary != primary_edit_instances.end() && primary->second == instance_entity) {
                record.Flags |= uint32_t(MeshletInstanceFlag::EditOverlay);
                if (ri.GpuId != InvalidOffset) {
                    edit_overlay_work.Ranges += ri.MeshletRangeCount;
                    edit_overlay_work.Meshlets += ri.MeshletCount;
                }
            }
            const auto mesh = instance ? TryGetMesh(r, instance->Entity) : std::nullopt;
            const bool shaded_face_less = mesh && show_rendered && mesh->FaceCount() == 0u &&
                meshes.GetPrimitiveMaterialRange(mesh->GetStoreId()).Count > 0u;
            const bool wire = instance && mesh_buffers && mesh_buffers->Meshlets.Count > 0u &&
                !r.all_of<ArmatureObject>(instance->Entity) && !r.all_of<BoneJoint>(instance->Entity) &&
                !r.all_of<ObjectExtrasTag>(instance->Entity) && mesh_buffers->EdgeIndices.Count > 0u &&
                (mesh_buffers->FaceIndices.Count == 0u || is_wireframe_mode) && !shaded_face_less;
            if (wire) {
                record.Flags |= uint32_t(MeshletInstanceFlag::Wire);
                if (ri.GpuId != InvalidOffset) {
                    wire_work.Ranges += ri.MeshletRangeCount;
                    wire_work.Meshlets += ri.MeshletCount;
                }
            }
            const bool bone = instance && r.all_of<ArmatureObject>(instance->Entity);
            const bool joint = instance && r.all_of<BoneJoint>(instance->Entity);
            if (bone || joint) record.Flags |= uint32_t(MeshletInstanceFlag::OverlayOnly);
            const auto mark = [&](MeshletInstanceFlag flag) {
                record.Flags |= uint32_t(flag);
                if (ri.GpuId != InvalidOffset) {
                    auto &work = buffers.FlagWork(uint32_t(flag));
                    work.Ranges += ri.MeshletRangeCount;
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
                if (show_vertex_normals && mesh_buffers->EdgeIndices.Count > 0u) {
                    mark(MeshletInstanceFlag::VertexNormal);
                }
            }
            if (instance && mesh_buffers && show_overlays && is_excite_mode &&
                sound_meshes.contains(instance->Entity) && mesh_buffers->EdgeIndices.Count > 0u) {
                mark(MeshletInstanceFlag::EdgeOverlay);
            }
            if (instance && mesh_buffers && show_overlays && is_excite_mode &&
                sound_meshes.contains(instance->Entity) && mesh_buffers->Meshlets.Count > 0u) {
                mark(MeshletInstanceFlag::SoundPoint);
            }
            const bool point_overlay = instance && mesh && mesh_buffers &&
                mesh->FaceCount() == 0u && mesh->EdgeCount() == 0u &&
                !primary_edit_instances.contains(instance->Entity) && !shaded_face_less;
            if (point_overlay) mark(MeshletInstanceFlag::PointOverlay);
            if (silhouette && ri.GpuId != InvalidOffset) {
                silhouette_work.Ranges += ri.MeshletRangeCount;
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
        pipelines.Main.Compiler.CompileTopologyPipelines(
            pipelines.Libraries, (buffers.MeshletTopologyMask & ~1u) != 0u
        );
    }
    if (update != SceneUpdate::Reuse || phase == RenderPhase::Full) RecordSceneCounters(buffers);

    const bool transmission_active = real_transmission && pipelines.Main.Transmission;
    // Composite transmission only when edit tint, velocity, and debug output do not require rerasterization.
    const bool composite_transmission = transmission_active && phase == RenderPhase::Full && !is_edit_mode && settings.DebugChannel == DebugChannel::None;
    const bool meshlet_fill = buffers.MeshletInstanceCount > 0;

    // The posed passes run every phase, since blur steps read their step's captured pose through the phase's UBO instance.
    // Bounds and cull run once per command buffer, and later blur phases reuse the culled buffers.
    // Derived normals feed only the scene's face-fill draws, so only scene-drawing phases record the derive.
    // Every prelude pass dispatches indirectly.
    // A submit with unchanged deform inputs gets zero group counts, keeping the buffers' current results.
    if (buffers.Prelude.HasWork()) {
        const auto &prelude = buffers.Prelude;
        const bool record_bounds = phase != RenderPhase::BlurAccumulate && phase != RenderPhase::BlurResolve;
        // Every derive entry contributes at least one face tile and one gather tile.
        const bool record_derive = draw_scene && prelude.DeriveFaces > 0;
        const bool bounds_work = record_bounds && prelude.BoundsCombine > 0;
        auto *compute = chain.BeginCompute("Prelude", MTL::StageVertex | MTL::StageFragment | MTL::StageDispatch);
        // Bindless dependencies require explicit barriers between pose, bounds, derive, gather, and combine dispatches.
        if (prelude.PosePrepass > 0) {
            RecordPosePrepass(compute, slots, pipelines, buffers, ubo_offset);
            compute->memoryBarrier(MTL::BarrierScopeBuffers);
        }
        if (prelude.PosedMeshletBounds > 0) RecordPosedMeshletBounds(compute, slots, pipelines, buffers, ubo_offset);
        if (record_derive || bounds_work) {
            auto derive_pc = MakeNormalDerivePc(buffers, meshes, buffers.PosedVertexNormals.Slot, buffers.PosedSeamNormals.Slot, buffers.PosedFaceNormals.Slot);
            if (record_derive) RecordNormalDerive(compute, slots, pipelines, buffers, derive_pc, PreludeSlot::DeriveFaces, ubo_offset);
            if (bounds_work) RecordBoundsPass(compute, slots, pipelines.BoundsReduce, buffers, PreludeSlot::BoundsReduce, ubo_offset);
            compute->memoryBarrier(MTL::BarrierScopeBuffers);
            if (record_derive) {
                derive_pc.Phase = 1;
                derive_pc.FirstTile = prelude.DeriveFaces;
                RecordNormalDerive(compute, slots, pipelines, buffers, derive_pc, PreludeSlot::DeriveGather, ubo_offset);
            }
            if (bounds_work) RecordBoundsPass(compute, slots, pipelines.BoundsCombine, buffers, PreludeSlot::BoundsCombine, ubo_offset);
        }
    }
    if (buffers.PreludeStale) {
        for (auto &[_, work] : scene_state.EditWork) work.BoundsInitialized = false;
    }
    if (is_edit_mode && std::exchange(scene_state.EditPreludePending, false)) RecordSparseEditPrelude(r, viewport, chain);
    MTL::RenderCommandEncoder *encoder = nullptr;
    const auto record_meshlets = [&](uint32_t route, auto &&bind_pipeline) {
        bind_pipeline();
        DrawMeshlets(encoder, buffers, route);
    };
    auto draw_quad = [&] { encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4)); };

    const auto &main = pipelines.Main;
    const auto main_extent = main.Resources->SceneColorImage.Extent;
    const bool has_silhouette = render_silhouette && meshlet_fill;
    // Populate visibility for wireframe selection outlines without loading its depth into the scene pass.
    const bool need_visibility = meshlet_fill && (show_fill || has_silhouette);
    const bool wire_meshlets = draw_overlays &&
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::Wire)).Meshlets > 0u;
    const uint64_t bone_meshlets = draw_overlays ?
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::Bone)).Meshlets +
            buffers.FlagWork(uint32_t(MeshletInstanceFlag::BoneJoint)).Meshlets :
        0u;
    const uint64_t normal_meshlets = draw_overlays ?
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::FaceNormal)).Meshlets +
            buffers.FlagWork(uint32_t(MeshletInstanceFlag::VertexNormal)).Meshlets :
        0u;
    const uint64_t element_overlay_meshlets = draw_overlays ?
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::EdgeOverlay)).Meshlets +
            buffers.FlagWork(uint32_t(MeshletInstanceFlag::PointOverlay)).Meshlets +
            buffers.FlagWork(uint32_t(MeshletInstanceFlag::SoundPoint)).Meshlets :
        0u;
    const bool cull_scene_meshlets = draw_scene &&
        (need_visibility || wire_meshlets || bone_meshlets > 0u || normal_meshlets > 0u ||
         element_overlay_meshlets > 0u);
    bool sort_blend = false;
    if (cull_scene_meshlets && show_rendered) {
        for (uint32_t i = 0; i < buffers.Materials.Count() && !sort_blend; ++i) {
            sort_blend = buffers.Materials.Get(i).AlphaMode == MaterialAlphaMode::Blend;
        }
    }
    const auto view_bytes = buffers.SceneViewUBO.Contents().subspan(ubo_offset, sizeof(SceneViewUBO));
    const auto &current_view_proj = reinterpret_cast<const SceneViewUBO *>(view_bytes.data())->ViewProj;
    const bool disocclusion_possible = update != SceneUpdate::Reuse || buffers.PreludeStale || buffers.MeshletOcclusionStale ||
        std::memcmp(&current_view_proj, &buffers.PreviousFullCullViewProj, sizeof(mat4)) != 0;
    // Keep real transmission single-phase because phase two omits textured transmission-hole coverage.
    const bool two_phase_meshlets = show_fill && cull_scene_meshlets && phase == RenderPhase::Full && !real_transmission &&
        main.Resources->DepthPyramidValid && disocclusion_possible;
    if (cull_scene_meshlets) {
        const bool stale_single_phase_transmission = real_transmission && disocclusion_possible;
        const uint32_t pyramid = show_fill && phase == RenderPhase::Full && main.Resources->DepthPyramidValid && !stale_single_phase_transmission ?
            sel_slots.DepthPyramidSampler :
            InvalidSlot;
        RecordMeshletCull(
            chain, slots, pipelines, buffers,
            {
                .Mode = show_rendered ? (real_transmission ? MeshletRouteMode::Transmission : MeshletRouteMode::Material) : MeshletRouteMode::Visibility,
                .RequiredInstanceFlags = show_fill || wire_meshlets || bone_meshlets > 0u ||
                        normal_meshlets > 0u || element_overlay_meshlets > 0u ?
                    0u :
                    uint32_t(MeshletInstanceFlag::Silhouette),
                .UboOffset = ubo_offset,
                .PyramidSamplerSlot = pyramid,
                .SortBlend = sort_blend,
                .TwoPhase = two_phase_meshlets,
            }
        );
    }
    if (need_visibility) {
        RecordMeshletVisibilityPass(chain, slots, pipelines, buffers, real_transmission, ubo_offset);
    }
    if (show_fill && phase == RenderPhase::Full && cull_scene_meshlets) {
        if (two_phase_meshlets) {
            auto *compute = chain.BeginCompute("DepthPyramidPhase1", MTL::StageFragment);
            RecordDepthPyramid(compute, slots, buffers, pipelines, sel_slots, ubo_offset);
            RecordMeshletPhase2Cull(
                chain, slots, pipelines, buffers, sel_slots.DepthPyramidSampler, ubo_offset,
                show_rendered ? MeshletRouteMode::Material : MeshletRouteMode::Visibility
            );
            const std::array colors{mtl::LoadColor(*main.Resources->VisibilityImage)};
            const auto pass = mtl::MakePassDescriptor(colors, mtl::LoadDepth(*main.Resources->DepthImage));
            encoder = encode::BeginScenePass(
                chain, pass, "MeshletVisibilityPhase2", {{MTL::StageDispatch, MTL::StageMesh}}, main_extent,
                slots, buffers, ubo_offset
            );
            DrawPhase2Meshlets(encoder, buffers, main);
            buffers.VisibilityIdGeneration = buffers.MeshletVisibleGeneration;
        }
        buffers.PreviousFullCullViewProj = current_view_proj;
    }
    if (has_silhouette) { // Silhouette depth/object pass
        RecordSilhouetteDepthPass(chain, slots, pipelines, buffers, true, ubo_offset);

        const auto &silhouette_edge = pipelines.SilhouetteEdge;
        {
            const auto extent = silhouette_edge.Resources->OffscreenImage.Extent;
            const std::array colors{mtl::ClearColor(*silhouette_edge.Resources->OffscreenImage)};
            const auto pass = mtl::MakePassDescriptor(colors, mtl::ClearDepth(*silhouette_edge.Resources->DepthImage));
            encoder = encode::BeginScenePass(chain, pass, "SilhouetteEdge", {{MTL::StageFragment, MTL::StageFragment}}, extent, slots, buffers, ubo_offset);
            silhouette_edge.Renderer.Bind(encoder, SPT::SilhouetteEdgeDepthObject);
            encode::SetPushConstants(encoder, SilhouetteEdgeDepthObjectPushConstants{sel_slots.SilhouetteSampler});
            draw_quad();
        }
    }

    // Render background and opaque faces without exposure into TransmissionImage for refracted sampling.
    if (transmission_active && draw_scene) {
        // Refraction samples only the world buffer.
        // The overlay composite adds the display-referred viewport backdrop.
        const std::array colors{mtl::ClearColor(*main.Transmission->Mip0View)};
        const auto pass = mtl::MakePassDescriptor(colors, mtl::LoadDepth(*main.Resources->DepthImage));
        encoder = encode::BeginScenePass(chain, pass, "TransmissionPrepass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, ubo_offset);
        main.PrepassBackground.Bind(encoder);
        draw_quad();
        if (meshlet_fill && show_fill) {
            main.Compiler.BindVisibility(encoder, PbrCompiler::Variant::OpaquePrepass);
            encoder->setFragmentTexture(*main.Resources->VisibilityImage, 0u);
            encode::SetPushConstants(encoder, encode::VisibilityDecodePc(buffers));
            draw_quad();
        }

        // Generate the transmission mip chain sampled across roughness.
        if (main.Transmission->Image.MipLevels > 1) {
            auto *blit = chain.BeginBlit("TransmissionMips", MTL::StageFragment);
            blit->generateMipmaps(*main.Transmission->Image);
        }
    }

    // The blur variant writes opaque color and screen motion together.
    const bool blur = phase == RenderPhase::BlurredFull || IsBlurAccumulate(phase);

    { // Scene pass: shaded scene into its own color target, and the depth the overlay pass occludes against.
        const auto &scene_renderer = blur ? main.SceneVelocityRenderer : main.SceneRenderer;
        // The composite path resumes over the transmission prepass's depth rather than clearing it.
        // Blurred steps add the velocity attachment the opaque geometry writes its screen motion into.
        const std::array colors{
            mtl::ClearColor(*main.Resources->SceneColorImage),
            blur ? mtl::ClearColor(*main.MotionBlur->VelocityImage) : mtl::ColorAttachment{},
        };
        const auto attachments = std::span{colors}.first(blur ? 2 : 1);
        const auto depth = show_fill && meshlet_fill ? mtl::LoadDepth(*main.Resources->DepthImage) : mtl::ClearDepth(*main.Resources->DepthImage);
        const auto pass = mtl::MakePassDescriptor(attachments, depth);
        encoder = encode::BeginScenePass(
            chain, pass, draw_scene ? "ScenePass" : "SceneDepthPass",
            {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment | MTL::StageBlit, MTL::StageFragment}},
            main_extent, slots, buffers, ubo_offset
        );

        // Initialize uncovered pixels with rotational background motion before geometry overwrites them.
        if (blur) {
            scene_renderer.Bind(encoder, SPT::BackgroundVelocity);
            draw_quad();
        }
        // The prepass covers the background and plain-opaque geometry, so the composite replaces both.
        if (composite_transmission) {
            scene_renderer.Bind(encoder, SPT::TransmissionComposite);
            draw_quad();
        } else if (show_rendered && draw_scene) {
            // Draw the background environment only in PBR modes.
            // The shader discards when world opacity is zero or no environment slot exists.
            scene_renderer.Bind(encoder, SPT::Background);
            draw_quad();
        }
        // Fill the scene target with the averaged steps, for the depth and overlays below to draw over.
        if (phase == RenderPhase::BlurResolve) {
            scene_renderer.Bind(encoder, SPT::MotionBlurResolve);
            const struct {
                uint32_t AccumSamplerSlot;
                float InvSteps;
            } resolve_pc{sel_slots.MotionBlurAccumSampler, 1.f / float(MotionBlurSteps(settings))};
            encode::SetPushConstants(encoder, resolve_pc);
            draw_quad();
        }

        // Seed silhouette depth before nearer mesh depth overwrites it.
        if (has_silhouette) {
            scene_renderer.Bind(encoder, SPT::SilhouetteEdgeDepth);
            const struct {
                uint32_t DepthSamplerIndex;
            } depth_pc{sel_slots.DepthSampler};
            encode::SetPushConstants(encoder, depth_pc);
            draw_quad();
        }

        // Draw solid faces.
        // BlurResolve writes depth for overlay occlusion because blended faces omit depth writes.
        if (show_fill) {
            if (meshlet_fill && draw_scene && show_rendered) {
                const auto opaque_variant = blur ? PbrCompiler::Variant::OpaqueVelocity : PbrCompiler::Variant::Opaque;
                const auto blend_variant = blur ? PbrCompiler::Variant::BlendVelocity : PbrCompiler::Variant::Blend;
                if (!composite_transmission) {
                    main.Compiler.BindVisibility(encoder, opaque_variant);
                    encoder->setFragmentTexture(*main.Resources->VisibilityImage, 0u);
                    encode::SetPushConstants(encoder, encode::VisibilityDecodePc(buffers));
                    draw_quad();
                }
                if (real_transmission) record_meshlets(uint32_t(MeshletRoute::Transmission), [&] { main.Compiler.BindMeshlets(encoder, opaque_variant); });
                record_meshlets(uint32_t(MeshletRoute::Blend), [&] { main.Compiler.BindMeshlets(encoder, blend_variant); });
            } else if (meshlet_fill && draw_scene) {
                main.WorkspaceVisibility.Bind(encoder);
                encoder->setFragmentTexture(*main.Resources->VisibilityImage, 0u);
                encode::SetPushConstants(encoder, encode::VisibilityDecodePc(buffers));
                draw_quad();
            }
        }
    }

    // Rebuild the persistent pyramid after indexed opaque draws while phase two uses the meshlet-only pyramid.
    if (show_fill && phase == RenderPhase::Full && cull_scene_meshlets) {
        auto *compute = chain.BeginCompute("DepthPyramidFinal", MTL::StageFragment);
        RecordDepthPyramid(compute, slots, buffers, pipelines, sel_slots, ubo_offset);
        main.Resources->DepthPyramidValid = true;
    }

    if (blur) RecordMotionBlurPostFx(r, chain, slots, viewport, main_extent, ubo_offset, playback_frame);

    if (!draw_overlays) { // BlurAccumulate adds this step's blurred scene without overlays.
        {
            const std::array colors{
                phase == RenderPhase::BlurAccumulateFirst ? mtl::ClearColor(*main.MotionBlur->AccumImage) : mtl::LoadColor(*main.MotionBlur->AccumImage)
            };
            const auto pass = mtl::MakePassDescriptor(colors);
            encoder = encode::BeginScenePass(chain, pass, "BlurAccumulate", {{MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, ubo_offset);
            main.MotionBlurAccumulate.Bind(encoder);
            const struct {
                uint32_t GatherSamplerSlot;
            } accum_pc{sel_slots.MotionBlurGatherSampler};
            encode::SetPushConstants(encoder, accum_pc);
            draw_quad();
        }
        return;
    }

    // Wireframe lines rasterize in compute before the overlay pass that resolves them.
    const bool wire_raster_drawn = wire_meshlets;
    const bool meshlet_edit_overlay_drawn =
        buffers.FlagWork(uint32_t(MeshletInstanceFlag::EditOverlay)).Meshlets > 0;
    const bool overlay_jobs = show_overlays && buffers.OverlayJobs.UsedSize > 0u &&
        (settings.ShowExtras || settings.ShowBoundingBoxes || settings.ShowTetWireframe);
    if (wire_raster_drawn) {
        { // Coverage sums and the complemented depth both start from zero.
            auto *blit = chain.BeginBlit("WireClear", MTL::StageDispatch);
            blit->fillBuffer(*buffers.WireCoverageBuffer, NS::Range::Make(0, buffers.WireCoverageBuffer.UsedSize), 0);
        }
        // Canonical meshlet edge owners accumulate with atomics, so threadgroups need no ordering.
        auto *wire = chain.BeginCompute("WireRaster", MTL::StageBlit | MTL::StageFragment, MTL::DispatchTypeConcurrent);
        encode::BindCompute(wire, pipelines.WireRaster, slots, buffers, ubo_offset);
        WireRasterPushConstants wire_pc{
            .Meshlet = MakeMeshletDrawPc(
                buffers, buffers.VisibleMeshlets, buffers.MeshletRoutes,
                uint32_t(MeshletRoute::Wire), uint32_t(MeshletInstanceFlag::Wire),
                0u, false, InvalidSlot
            ),
            .CoverageSlot = buffers.WireCoverageBuffer.Slot,
        };
        for (uint32_t chunk = 0; chunk < buffers.MeshletDispatchChunkCount; ++chunk) {
            wire_pc.Meshlet.VisibleOffset = chunk * GpuBuffers::MeshletDispatchChunkSize;
            encode::SetPushConstants(wire, wire_pc);
            const auto args_offset = (uint32_t(MeshletRoute::Wire) * buffers.MeshletDispatchChunkCount + chunk) *
                sizeof(MeshDispatchArgs);
            wire->dispatchThreadgroups(*buffers.MeshletDispatchArgs, args_offset, MTL::Size(160, 1, 1));
        }
    }
    if (overlay_jobs) RecordOverlayJobCull(chain, slots, pipelines, buffers, false, ubo_offset);

    // Skip transparent overlay color and line data when no overlay draw writes them.
    bool overlay_layer_drawn = false;
    // List every drawable category for this pass.
    // An empty list prevents the composite from reading either overlay layer.
    const bool overlay_pass_needed = has_silhouette ||
        (show_overlays && settings.ShowGrid) ||
        meshlet_edit_overlay_drawn || element_overlay_meshlets > 0u || wire_raster_drawn ||
        overlay_jobs ||
        normal_meshlets > 0u || bone_meshlets > 0u;
    if (overlay_pass_needed) { // Overlay pass: display-referred overlays over transparent, depth-tested against the scene above.
        // Transparent overlay color is composited over scene color by alpha.
        const std::array overlay_colors{
            mtl::ClearColor(*main.Resources->OverlayColorImage),
            mtl::ClearColor(*main.Resources->LineDataImage),
        };
        const auto overlay_pass = mtl::MakePassDescriptor(overlay_colors, mtl::LoadDepth(*main.Resources->DepthImage));
        encoder = encode::BeginScenePass(chain, overlay_pass, "OverlayPass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, ubo_offset);

        const auto draw_meshlet_overlay = [&](
                                              const mtl::MeshRenderPipeline &pipeline, MeshletRoute route, MeshletInstanceFlag flag,
                                              uint32_t threads, uint32_t corner = 0u, uint32_t sharpness_slot = InvalidSlot
                                          ) {
            pipeline.Bind(encoder);
            ForEachMeshletVisibilityList(
                buffers, two_phase_meshlets,
                [&](const auto &visible, const auto &routes, const auto &dispatch_args) {
                    DrawMeshletList(
                        encoder, buffers, visible, routes, dispatch_args,
                        uint32_t(route), uint32_t(flag), 0u, false, false,
                        sharpness_slot, threads, corner
                    );
                }
            );
        };

        {
            if (meshlet_edit_overlay_drawn) {
                overlay_layer_drawn = true;
                // Preserve the sharp-free specialization; the all-smooth Meshlets scene is measurably faster with it.
                const auto &edit_edges = scene_state.MeshletEditHasSharpEdges ? main.MeshletEditEdges : main.MeshletEditSmoothEdges;
                for (uint32_t corner = 0u; corner < 3u; ++corner) {
                    draw_meshlet_overlay(
                        edit_edges, MeshletRoute::EditOverlay, MeshletInstanceFlag::EditOverlay,
                        160u, corner, meshes.GetEdgeSharpnessSlot()
                    );
                }
            }
            if (buffers.FlagWork(uint32_t(MeshletInstanceFlag::EdgeOverlay)).Meshlets > 0u) {
                overlay_layer_drawn = true;
                for (uint32_t corner = 0u; corner < 3u; ++corner) {
                    draw_meshlet_overlay(
                        main.MeshletEditSmoothEdges, MeshletRoute::Overlay,
                        MeshletInstanceFlag::EdgeOverlay, 160u, corner
                    );
                }
            }
            if (wire_raster_drawn) {
                overlay_layer_drawn = true;
                main.WireResolve.Bind(encoder);
                encode::SetPushConstants(encoder, WireResolvePushConstants{buffers.WireCoverageBuffer.Slot});
                encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
            }
            // Points follow the wire resolve so selected vertices stay on top in wireframe mode too.
            if (meshlet_edit_overlay_drawn && edit_mode == Element::Vertex) {
                draw_meshlet_overlay(
                    main.MeshletEditPoint, MeshletRoute::EditOverlay,
                    MeshletInstanceFlag::EditOverlay, 64u
                );
            }
            if (buffers.FlagWork(uint32_t(MeshletInstanceFlag::PointOverlay)).Meshlets > 0u) {
                overlay_layer_drawn = true;
                draw_meshlet_overlay(
                    main.MeshletEditPoint, MeshletRoute::Overlay,
                    MeshletInstanceFlag::PointOverlay, 64u
                );
            }
            if (buffers.FlagWork(uint32_t(MeshletInstanceFlag::SoundPoint)).Meshlets > 0u) {
                overlay_layer_drawn = true;
                draw_meshlet_overlay(
                    main.MeshletEditPoint, MeshletRoute::Overlay,
                    MeshletInstanceFlag::SoundPoint, 64u
                );
            }
            if (overlay_jobs) {
                overlay_layer_drawn = true;
                main.OverlayJobLines.Bind(encoder);
                DrawOverlayJobs(encoder, buffers, meshes);
            }
        }

        if (has_silhouette) {
            overlay_layer_drawn = true;
            main.OverlayRenderer.Bind(encoder, SPT::SilhouetteEdgeColor);
            // In mesh Edit mode, suppress active silhouette (element selection drives active state differently).
            // In armature Edit/Pose mode, the active bone gets the active-color silhouette.
            const auto active_entity = FindActiveEntity(r);
            const auto active_bone = FindActiveBone(r);
            const bool armature_mode = FindArmatureObject(r, active_entity) != entt::null;
            uint32_t active_object_id = 0;
            if (armature_mode && active_bone != entt::null) {
                if (r.all_of<RenderInstance>(active_bone)) {
                    active_object_id = r.get<RenderInstance>(active_bone).ObjectId;
                }
            } else if (!is_edit_mode && active_entity != entt::null && r.all_of<RenderInstance>(active_entity)) {
                active_object_id = r.get<RenderInstance>(active_entity).ObjectId;
            }
            encode::SetPushConstants(encoder, SilhouetteEdgeColorPushConstants{TransformGizmo::IsUsing(r, viewport) && interaction_mode == InteractionMode::Object, sel_slots.ObjectIdSampler, active_object_id});
            draw_quad();
        }

        if (normal_meshlets > 0u) {
            overlay_layer_drawn = true;
            const auto draw_normals = [&](const mtl::MeshRenderPipeline &pipeline, MeshletInstanceFlag flag) {
                if (buffers.FlagWork(uint32_t(flag)).Meshlets == 0u) return;
                draw_meshlet_overlay(pipeline, MeshletRoute::Overlay, flag, 64u);
            };
            draw_normals(main.FaceNormalMesh, MeshletInstanceFlag::FaceNormal);
            draw_normals(main.VertexNormalMesh, MeshletInstanceFlag::VertexNormal);
        }

        // Grid plane (drawn before bone depth clear so grid remains depth-tested against scene meshes)
        if (show_overlays && settings.ShowGrid) {
            overlay_layer_drawn = true;
            main.OverlayRenderer.Bind(encoder, SPT::Grid);
            encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(9));
        }

        { // Bone X-ray: depth clears so bones are never occluded by scene meshes, only by each other.
            // Use a second pass to preserve overlay color while clearing bone depth at pass start.
            if (bone_meshlets > 0u) {
                const std::array bone_colors{
                    mtl::LoadColor(*main.Resources->OverlayColorImage),
                    mtl::LoadColor(*main.Resources->LineDataImage),
                };
                const auto bone_pass = mtl::MakePassDescriptor(bone_colors, mtl::ClearDepth(*main.Resources->DepthImage));
                encoder = encode::BeginScenePass(chain, bone_pass, "BoneXRay", {{MTL::StageDispatch, MTL::StageMesh | MTL::StageFragment}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, ubo_offset);

                const auto draw_bones = [&](const mtl::MeshRenderPipeline &pipeline, MeshletInstanceFlag flag, uint32_t threads, float depth_bias = 0.f) {
                    if (buffers.FlagWork(uint32_t(flag)).Meshlets == 0u) return;
                    overlay_layer_drawn = true;
                    pipeline.Bind(encoder);
                    encoder->setDepthBias(depth_bias, 0.f, 0.f);
                    DrawMeshlets(encoder, buffers, uint32_t(MeshletRoute::Overlay), uint32_t(flag), threads);
                    if (depth_bias != 0.f) encoder->setDepthBias(0.f, 0.f, 0.f);
                };

                // In Object+wireframe mode, show only outlines (no fills).
                // In Edit/Pose+wireframe, fills are semitransparent and write far-plane depth (via shader) so wires are never occluded.
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
    }

    { // Composite: anti-alias the overlay layer using LineDataImage, view-transform the scene, merge into FinalColorImage
        const std::array colors{mtl::ClearColor(*main.Resources->FinalColorImage, {0, 0, 0, 1})};
        const auto pass = mtl::MakePassDescriptor(colors);
        encoder = encode::BeginScenePass(chain, pass, "Composite", {{MTL::StageFragment, MTL::StageFragment}}, main.Resources->FinalColorImage.Extent, slots, buffers, ubo_offset);
        main.ViewportComposite.Bind(encoder);
        // Debug channels write their own already-viewable values, so they pass through untransformed.
        const uint32_t view_transform = settings.DebugChannel != DebugChannel::None ? 2u : show_rendered ? 1u :
                                                                                                           0u;
        // BlurredFull composites the finished scene from the gather target.
        const uint32_t scene_sampler = phase == RenderPhase::BlurredFull ? sel_slots.MotionBlurGatherSampler : sel_slots.SceneColorSampler;
        const struct {
            uint32_t SceneColorSamplerSlot, OverlayColorSamplerSlot, LineDataSamplerSlot, ViewTransform, HasOverlay;
            vec4 Backdrop;
        } composite_pc{scene_sampler, sel_slots.OverlayColorSampler, sel_slots.LineDataSampler, view_transform, overlay_layer_drawn, settings.ClearColor};
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
                                              .TetPositionSlot = meshes.GetTetPositionSlot(),
                                              .TetEdgeIndexSlot = meshes.GetTetEdgeIndexSlot(),
                                          });
    encoder->drawMeshThreadgroups(
        *buffers.OverlayJobDispatchArgs, 0u, MTL::Size(1, 1, 1),
        MTL::Size(uint32_t(OverlayDispatch::LineGroupLines) * 2u, 1, 1)
    );
}

void RecordMeshletVisibilityPass(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines,
    GpuBuffers &buffers, bool transmission, uint32_t ubo_offset
) {
    const auto &main = pipelines.Main;
    const std::array colors{
        mtl::ClearColor(*main.Resources->VisibilityImage, MTL::ClearColor{double(UINT32_MAX), 0, 0, 0})
    };
    const auto pass = mtl::MakePassDescriptor(colors, mtl::ClearDepth(*main.Resources->DepthImage));
    auto *encoder = encode::BeginScenePass(
        chain, pass, "MeshletVisibility", {{MTL::StageDispatch, MTL::StageMesh}},
        main.Resources->VisibilityImage.Extent, slots, buffers, ubo_offset
    );
    DrawVisibilityMeshlets(encoder, buffers, main, transmission);
    buffers.VisibilityIdGeneration = buffers.MeshletVisibleGeneration;
}

void RecordMeshletCull(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines,
    GpuBuffers &buffers, MeshletCullConfig config
) {
    ++buffers.MeshletVisibleGeneration;
    auto *encoder = chain.BeginCompute("MeshletCull", MTL::StageMesh | MTL::StageFragment);
    const bool transmission = config.Mode == MeshletRouteMode::Transmission;
    // The requested flag's maintained totals bound this cull.
    const auto primary = config.RequiredInstanceFlags == 0u ?
        GpuBuffers::MeshletFlagWork{buffers.MeshletRangeCount, buffers.MeshletInstanceCount} :
        buffers.FlagWork(config.RequiredInstanceFlags);
    buffers.EnsureMeshletVisibilityCapacity(
        primary.Meshlets * (1u + transmission), primary.Ranges, primary.Meshlets, primary.Meshlets,
        config.SortBlend, config.TwoPhase
    );
    const auto pc = [&] {
        auto pc = MakeMeshletCullSlotsPc(buffers);
        pc.InstanceCount = buffers.GpuInstanceSlots.Buffer.Count<uint32_t>();
        pc.WorkBlockCount = (pc.InstanceCount + GpuBuffers::MeshletCullBlockSize - 1u) / GpuBuffers::MeshletCullBlockSize;
        pc.LodFrontierStateSlot = buffers.LodFrontierStates.Slot;
        pc.BlendBlockSlot = config.SortBlend ? buffers.MeshletBlendBlocks.Slot : InvalidSlot;
        pc.RouteMode = uint32_t(config.Mode);
        pc.RequiredInstanceFlags = config.RequiredInstanceFlags;
        pc.RouteMask = config.RouteMask;
        pc.PyramidSamplerSlot = config.PyramidSamplerSlot;
        pc.TwoPhase = config.TwoPhase;
        return pc;
    }();
    encode::BindScene(encoder, slots, buffers, config.UboOffset);
    constexpr uint32_t simd_groups = GpuBuffers::MeshletCullBlockSize / 32u;
    constexpr uint32_t prefix_stride = simd_groups + 1u;
    const auto dispatch_meshlets = [&](const mtl::ComputePipeline &pipeline) {
        encoder->setComputePipelineState(pipeline.State());
        const uint32_t prefix_bytes = GpuBuffers::MeshletRouteCount * prefix_stride * sizeof(uint32_t);
        const uint32_t blend_bytes = config.SortBlend ? simd_groups * 256u * sizeof(uint16_t) : 0u;
        encoder->setThreadgroupMemoryLength(AlignedThreadgroupBytes(prefix_bytes + blend_bytes), 0);
        encoder->dispatchThreadgroups(*buffers.MeshletWorkDispatchArgs, 0, MTL::Size(GpuBuffers::MeshletCullBlockSize, 1, 1));
    };
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
            encoder->setThreadgroupMemoryLength(AlignedThreadgroupBytes(3u * prefix_stride * sizeof(uint32_t)), 0);
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
    dispatch_meshlets(pipelines.MeshletCullBlockCount);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    encoder->setComputePipelineState(pipelines.MeshletCullPrefix.State());
    encoder->setThreadgroupMemoryLength(AlignedThreadgroupBytes(GpuBuffers::MeshletRouteCount * sizeof(uint32_t)), 0);
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    dispatch_meshlets(pipelines.MeshletCullEmit);
}

void RecordSilhouetteDepthPass(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines,
    GpuBuffers &buffers, bool draw_meshlets, uint32_t ubo_offset
) {
    const bool draw = draw_meshlets && buffers.MeshletInstanceCount > 0;
    const auto &silhouette = pipelines.Silhouette;
    const auto extent = silhouette.Resources->OffscreenImage.Extent;
    const std::array colors{mtl::ClearColor(*silhouette.Resources->OffscreenImage)};
    const auto pass = mtl::MakePassDescriptor(colors, mtl::ClearDepth(*silhouette.Resources->DepthImage));
    // Element selection loads this cleared depth target even when there is no silhouette to draw.
    auto *encoder = encode::BeginScenePass(
        chain, pass, "SilhouetteDepth", {{MTL::StageFragment, MTL::StageFragment}},
        extent, slots, buffers, ubo_offset
    );
    if (!draw) return;
    silhouette.Visibility.Bind(encoder);
    encoder->setFragmentTexture(*pipelines.Main.Resources->VisibilityImage, 0u);
    encoder->setFragmentTexture(*pipelines.Main.Resources->DepthImage, 1u);
    encode::SetPushConstants(encoder, encode::VisibilityDecodePc(buffers));
    encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
}

void DrawMeshlets(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, uint32_t route,
    uint32_t required_instance_flags, uint32_t mesh_threads, uint32_t edit_edge_corner,
    uint32_t instance_filter
) {
    DrawMeshletList(
        encoder, buffers, buffers.VisibleMeshlets, buffers.MeshletRoutes, buffers.MeshletDispatchArgs,
        route, required_instance_flags, 0u, false, false, InvalidSlot,
        mesh_threads, edit_edge_corner, instance_filter
    );
}

void RecordRenderCommandBuffer(entt::registry &r, entt::entity viewport, MTL::CommandBuffer *command_buffer, SceneUpdate update, RenderPhase phase) {
    profile::BeginRecording();
    mtl::PassChain chain{command_buffer, profile::RecordingTimer()};
    RecordPhase(r, viewport, chain, update, phase, 0, r.get<const PlaybackFrame>(viewport).Value);
    profile::EndRecording();
}

void RecordBlurStepsCommandBuffer(entt::registry &r, entt::entity viewport, MTL::CommandBuffer *command_buffer, std::span<const float> step_frames) {
    const auto &buffers = r.ctx().get<const GpuBuffers>();
    profile::BeginRecording();
    mtl::PassChain chain{command_buffer, profile::RecordingTimer()};
    for (uint32_t i = 0; i < step_frames.size(); ++i) {
        RecordPhase(r, viewport, chain, i == 0 ? SceneUpdate::Rebuild : SceneUpdate::Reuse, i == 0 ? RenderPhase::BlurAccumulateFirst : RenderPhase::BlurAccumulate, buffers.SceneViewUboOffset(i + 1), step_frames[i]);
    }
    RecordPhase(r, viewport, chain, SceneUpdate::Reuse, RenderPhase::BlurResolve, 0, r.get<const PlaybackFrame>(viewport).Value);
    profile::EndRecording();
}

namespace {
// Upload `entries` and their tiles, then record and submit one batched two-phase derive and wait for completion.
// The output slots select the target buffers.
void SubmitNormalDeriveNow(entt::registry &r, std::span<const NormalDeriveEntry> entries, uint32_t vertex_normal_slot, uint32_t seam_normal_slot, uint32_t face_normal_slot) {
    const auto &meshes = r.ctx().get<const MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    std::vector<uvec2> face_tiles, gather_tiles;
    for (uint32_t entry_index = 0; entry_index < entries.size(); ++entry_index) {
        const auto &entry = entries[entry_index];
        for (uint32_t t = 0, n = TileCountFor(entry.FaceCount); t < n; ++t) face_tiles.emplace_back(entry_index, t);
        for (uint32_t t = 0, n = TileCountFor(entry.VertexCount + entry.SeamCount); t < n; ++t) gather_tiles.emplace_back(entry_index, t);
    }
    std::ranges::copy(entries, buffers.NormalDeriveEntries.SetCount<NormalDeriveEntry>(entries.size()).begin());
    const auto tiles = buffers.DeriveTiles.SetCount<uvec2>(face_tiles.size() + gather_tiles.size());
    std::ranges::copy(gather_tiles, std::ranges::copy(face_tiles, tiles.begin()).out);
    // The one-shot shares the frame prelude's indirect slots, then requests their rebuild.
    WritePreludeArg(buffers, PreludeSlot::DeriveFaces, uint32_t(face_tiles.size()));
    WritePreludeArg(buffers, PreludeSlot::DeriveGather, uint32_t(gather_tiles.size()));

    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    ctx.CommitResidency();
    auto *command_buffer = ctx.Queue->commandBuffer();
    auto *encoder = command_buffer->computeCommandEncoder();
    auto derive_pc = MakeNormalDerivePc(buffers, meshes, vertex_normal_slot, seam_normal_slot, face_normal_slot);
    RecordNormalDerive(encoder, slots, pipelines, buffers, derive_pc, PreludeSlot::DeriveFaces, 0);
    // The gather reads the face normals through bindless buffers the encoder cannot see.
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    derive_pc.Phase = 1;
    derive_pc.FirstTile = uint32_t(face_tiles.size());
    RecordNormalDerive(encoder, slots, pipelines, buffers, derive_pc, PreludeSlot::DeriveGather, 0);
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    // The one-shot rewrote per-frame derive inputs, so the next submit refreshes persistent scene descriptors.
    r.ctx().get<PendingRenderRequest>().Value = RenderRequest::Rebuild;
}
} // namespace

void DeriveBaseNormalsNow(entt::registry &r, std::span<const entt::entity> mesh_entities) {
    const auto &meshes = r.ctx().get<const MeshStore>();
    std::vector<NormalDeriveEntry> entries;
    entries.reserve(mesh_entities.size());
    for (const auto entity : mesh_entities) {
        const auto *mesh_buffers = r.try_get<const MeshBuffers>(entity);
        const auto mesh = TryGetMesh(r, entity);
        if (!mesh_buffers || !mesh) continue;
        const auto store_id = mesh->GetStoreId();
        auto entry = MakeDeriveEntryInputs(meshes, store_id, mesh_buffers->FaceIndices);
        if (!entry) continue;
        entry->VertexNormalOffset = entry->Vertices.Offset;
        entry->SeamNormalOffset = meshes.GetBaseSeamNormalRange(store_id).Offset;
        entry->FaceNormalOffset = entry->FaceDataOffset;
        entries.emplace_back(*entry);
    }
    if (entries.empty()) return;
    SubmitNormalDeriveNow(r, entries, meshes.GetBaseVertexNormalSlot(), meshes.GetBaseSeamNormalSlot(), meshes.GetBaseFaceNormalSlot());
}

namespace {
// Returns whether the listed mesh entities retain authored shading normals under morphing.
// The CPU resolves targets with authored normal deltas.
// Position-only targets derive their full-weight poses in one batched submit-and-wait.
// The derived pose tests whether derivation moves the normals authored shading would pin.
// Runs after the base derive, since the pin test compares against the base normal stores.
void UpdateAuthoredMorphShadingNow(entt::registry &r, std::span<const entt::entity> mesh_entities) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    // Each position-only target gets a derive entry at its full-weight pose, reading and writing the posed scratch.
    struct PoseJob {
        entt::entity Entity;
        uint32_t TargetIndex;
    };
    std::vector<NormalDeriveEntry> entries;
    std::vector<PoseJob> jobs;
    uint32_t vertex_count_total = 0, seam_count_total = 0, face_count_total = 0;
    for (const auto entity : mesh_entities) {
        const auto *mesh_buffers = r.try_get<const MeshBuffers>(entity);
        const auto mesh = TryGetMesh(r, entity);
        if (!mesh_buffers || !mesh) continue;
        const auto store_id = mesh->GetStoreId();
        const auto target_count = meshes.GetMorphTargetCount(store_id);
        // A mesh without authored normals shades by derivation alone, under any morph weights.
        if (target_count == 0 || !meshes.HasAuthoredNormals(store_id)) continue;
        const auto entry_inputs = MakeDeriveEntryInputs(meshes, store_id, mesh_buffers->FaceIndices);
        if (!entry_inputs) continue;
        // Resolve the authored-normal gate from every morph target.
        meshes.UpdateMorphShadingAuthored(*mesh, {});
        if (meshes.GetMorphShadingAuthored(store_id)) continue;
        const auto vertex_count = entry_inputs->VertexCount;
        const auto targets = meshes.GetMorphTargets(store_id);
        for (uint32_t t = 0; t < target_count; ++t) {
            // Targets without position deltas use the rest pose and require no normal pinning.
            const auto deltas = targets.subspan(size_t{t} * vertex_count, vertex_count);
            if (std::ranges::all_of(deltas, [](const auto &d) { return d.PositionDelta == vec3{0}; })) continue;
            auto entry = *entry_inputs;
            entry.PosedPositionOffset = vertex_count_total;
            entry.VertexNormalOffset = vertex_count_total;
            entry.SeamNormalOffset = seam_count_total;
            entry.FaceNormalOffset = face_count_total;
            entries.emplace_back(entry);
            jobs.emplace_back(entity, t);
            vertex_count_total += vertex_count;
            seam_count_total += entry.SeamCount;
            face_count_total += entry.FaceCount;
        }
    }
    if (entries.empty()) return;

    // Fill each job's scratch positions with the base positions plus its target's full-weight deltas.
    // Then derive the whole batch in one submit.
    const auto positions = buffers.PosedPositions.SetCount<vec3>(vertex_count_total);
    const auto vertex_normals = buffers.PosedVertexNormals.SetCount<vec3>(vertex_count_total);
    const auto seam_normals = buffers.PosedSeamNormals.SetCount<vec3>(seam_count_total);
    const auto face_normals = buffers.PosedFaceNormals.SetCount<vec3>(face_count_total);
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto store_id = r.get<const MeshHandle>(jobs[i].Entity).StoreId;
        const auto base_vertices = meshes.GetVertices(store_id);
        const auto vertex_count = uint32_t(base_vertices.size());
        const auto deltas = meshes.GetMorphTargets(store_id).subspan(size_t{jobs[i].TargetIndex} * vertex_count, vertex_count);
        for (uint32_t v = 0; v < vertex_count; ++v) {
            positions[entries[i].PosedPositionOffset + v] = base_vertices[v].Position + deltas[v].PositionDelta;
        }
    }
    SubmitNormalDeriveNow(r, entries, buffers.PosedVertexNormals.Slot, buffers.PosedSeamNormals.Slot, buffers.PosedFaceNormals.Slot);

    // Compare per mesh over its contiguous run of jobs.
    for (size_t i = 0; i < jobs.size();) {
        const auto entity = jobs[i].Entity;
        std::vector<CornerNormalSources> poses;
        for (; i < jobs.size() && jobs[i].Entity == entity; ++i) {
            const auto &entry = entries[i];
            poses.emplace_back(
                vertex_normals.subspan(entry.VertexNormalOffset, entry.VertexCount),
                seam_normals.subspan(entry.SeamNormalOffset, entry.SeamCount),
                face_normals.subspan(entry.FaceNormalOffset, entry.FaceCount)
            );
        }
        meshes.UpdateMorphShadingAuthored(GetMesh(r, entity), poses);
    }
}
} // namespace

void FinalizeNewMeshShadingNow(entt::registry &r, std::span<const entt::entity> mesh_entities) {
    DeriveBaseNormalsNow(r, mesh_entities);
    auto &meshes = r.ctx().get<MeshStore>();
    for (const auto entity : mesh_entities) meshes.EncodeAuthoredCornerNormals(GetMesh(r, entity));
    UpdateAuthoredMorphShadingNow(r, mesh_entities);
}

namespace {
void DispatchWork(MTL::ComputeCommandEncoder *encoder, const GpuBuffers &buffers, ElementWork work) {
    encoder->dispatchThreadgroups(*buffers.GeometryWork.Buffer, WorkArgsOffset(work), ThreadgroupSize::Linear256);
}

MeshEditWork &PrepareMeshEditWork(entt::registry &r, entt::entity entity) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    const auto mesh = GetMesh(r, entity);
    const auto id = mesh.GetStoreId();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &mb = r.get<MeshBuffers>(entity);
    auto &work = r.ctx().get<GpuSceneState>().EditWork;
    if (const auto it = work.find(entity); it != work.end() && it->second.StoreId != id) ReleaseMeshEditWork(r, entity);
    auto [it, inserted] = work.try_emplace(entity);
    auto &w = it->second;
    if (inserted) {
        w.StoreId = id;
        w.Candidates = AllocateElementWork(buffers.GeometryWork, mesh.VertexCount());
        w.Vertices = AllocateElementWork(buffers.GeometryWork, mesh.VertexCount());
        w.Faces = AllocateElementWork(buffers.GeometryWork, mesh.FaceCount());
        w.Normals = AllocateElementWork(buffers.GeometryWork, mesh.VertexCount() + meshes.GetSeamCornerCount(id));
        uint32_t level0 = 0;
        for (const auto &primitive : buffers.Primitives.Get(mb.Primitives)) level0 += primitive.Level0Count;
        w.Meshlets = AllocateElementWork(buffers.GeometryWork, level0);
        w.BoundsTiles = AllocateElementWork(buffers.GeometryWork, TileCountFor(mesh.VertexCount()));
        for (uint32_t n = w.BoundsTiles.Count;;) {
            n = (n + 255u) / 256u;
            w.BoundsLevels.push_back({AllocateElementWork(buffers.GeometryWork, n), buffers.BoundsParents.Allocate(n)});
            if (n <= 1u) break;
        }
        const uint32_t elements = mesh.FaceCount() ? meshes.GetTriangleCount(id) : mesh.EdgeCount() ? mesh.EdgeCount() :
                                                                                                      mesh.VertexCount();
        w.ElementMeshlets = buffers.ElementMeshlets.Allocate(elements);
        auto map = buffers.ElementMeshlets.GetMutable(w.ElementMeshlets);
        std::ranges::fill(map, InvalidOffset);
        uint32_t ordinal = 0;
        for (const auto &primitive : buffers.Primitives.Get(mb.Primitives)) {
            for (uint32_t m = 0; m < primitive.Level0Count; ++m, ++ordinal) {
                const auto index = primitive.MeshletOffset + m;
                const auto &record = buffers.Meshlets.Get({index, 1}).front();
                for (const auto element : buffers.MeshletTriangleIds.Get({record.TriangleOffset, record.TriangleCount})) map[element] = ordinal;
            }
        }
    }
    return w;
}
} // namespace

void ReleaseMeshEditWork(entt::registry &r, entt::entity entity) {
    auto *scene = r.ctx().find<GpuSceneState>();
    if (!scene) return;
    auto &work = scene->EditWork;
    const auto it = work.find(entity);
    if (it == work.end()) return;
    auto &buffers = r.ctx().get<GpuBuffers>();
    const auto &w = it->second;
    for (auto range : {w.Candidates, w.Vertices, w.Faces, w.Normals, w.Meshlets, w.BoundsTiles}) buffers.GeometryWork.Release(WorkStorageRange(range));
    for (const auto &level : w.BoundsLevels) {
        buffers.GeometryWork.Release(WorkStorageRange(level.Work));
        buffers.BoundsParents.Release(level.Values);
    }
    buffers.ElementMeshlets.Release(w.ElementMeshlets);
    work.erase(it);
}

namespace {
CommitPosedGeometryPushConstants PrepareGeometryEdit(entt::registry &r, entt::entity viewport, entt::entity entity, entt::entity primary, const PendingTransform *pending, const PosedRanges *pose = nullptr) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &w = PrepareMeshEditWork(r, entity);
    const auto mesh = GetMesh(r, entity);
    const auto id = w.StoreId;
    if (!w.CandidateReady || (!pose && r.all_of<EditSelectionDirty>(viewport)))
        SeedElementWork(buffers.GeometryWork, w.Candidates, meshes.GetSelectionBits(id, Element::Vertex), w.PreviewActive);
    else if (!w.PreviewActive)
        IntersectElementWork(buffers.GeometryWork, w.Candidates, meshes.GetSelectionBits(id, Element::Vertex));
    w.CandidateReady = true;
    for (auto work : {w.Vertices, w.Faces, w.Normals, w.Meshlets, w.BoundsTiles}) ClearElementWork(buffers.GeometryWork, work);
    auto entry = MakeDeriveEntryInputs(meshes, id, r.get<const MeshBuffers>(entity).FaceIndices).value_or(NormalDeriveEntry{.VertexCount = mesh.VertexCount()});
    entry.VertexNormalOffset = meshes.GetBaseVertexNormalRange(id).Offset;
    entry.SeamNormalOffset = meshes.GetBaseSeamNormalRange(id).Offset;
    entry.FaceNormalOffset = meshes.GetBaseFaceNormalRange(id).Offset;
    if (pose) {
        for (const auto &level : w.BoundsLevels) ClearElementWork(buffers.GeometryWork, level.Work);
        entry.PosedPositionOffset = pose->PositionBase;
        if (pose->Normals) {
            entry.VertexNormalOffset = pose->Normals->VertexOffset;
            entry.SeamNormalOffset = pose->Normals->SeamOffset;
            entry.FaceNormalOffset = pose->Normals->FaceOffset;
        }
        w.PreviewActive = pending != nullptr;
    }
    return {
        .Vertices = meshes.GetVerticesRange(id),
        .Output = pose ? SlotOffset{buffers.PosedPositions.Slot, pose->PositionBase} : SlotOffset{},
        .Selection = meshes.GetEditSelectionStorage(id).VertexBits,
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
        .AdjacencySlot = meshes.GetAdjacencySlot(),
        .FaceFirstTriangleSlot = meshes.GetFaceFirstTriangleSlot(),
        .CornerClassSlot = meshes.GetCornerClassSlot(),
        .CornerClassOffset = meshes.GetCornerClassOffset(id),
        .VertexEdgeAdjacencyOffset = OffsetOrInvalid(meshes.GetVertexEdgeAdjacencyRange(id)),
        .Topology = mesh.FaceCount() ? 0u : mesh.EdgeCount() ? 1u :
                                                               2u,
        .TriangleMeshlets = buffers.ElementMeshlets.Slotted(w.ElementMeshlets),
        .ApplyTransform = pending ? 1u : 0u,
        .Commit = pose ? 0u : 1u,
    };
}

void RecordGeometryEditBatch(entt::registry &r, MTL::ComputeCommandEncoder *encoder, std::vector<std::pair<entt::entity, CommitPosedGeometryPushConstants>> &commits, bool posed) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    const auto &meshes = r.ctx().get<const MeshStore>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto entries = buffers.GeometryNormalEntries.SetCount<NormalDeriveEntry>(commits.size());
    for (uint32_t i = 0; i < commits.size(); ++i) entries[i] = commits[i].second.Entry;
    r.ctx().get<const mtl::Context>().CommitResidency();
    for (uint32_t phase = 0; phase < 3; ++phase) {
        for (auto &[_, pc] : commits) {
            pc.Phase = phase;
            encode::BindCompute(encoder, pipelines.CommitPosedGeometry, slots, buffers);
            encode::SetPushConstants(encoder, pc);
            DispatchWork(encoder, buffers, phase == 0 ? pc.Candidates : phase == 1 ? pc.Faces :
                                                                                     pc.Normals);
        }
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        for (const auto &[_, pc] : commits) {
            encode::BindCompute(encoder, pipelines.GeometryWorkArgs, slots, buffers);
            encode::SetPushConstants(encoder, pc);
            encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256);
        }
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    }
    auto derive = MakeNormalDerivePc(buffers, meshes, posed ? buffers.PosedVertexNormals.Slot : meshes.GetBaseVertexNormalSlot(), posed ? buffers.PosedSeamNormals.Slot : meshes.GetBaseSeamNormalSlot(), posed ? buffers.PosedFaceNormals.Slot : meshes.GetBaseFaceNormalSlot());
    derive.EntriesSlot = buffers.GeometryNormalEntries.Slot;
    for (uint32_t phase = 0; phase < 2; ++phase) {
        derive.Phase = phase;
        for (uint32_t i = 0; i < commits.size(); ++i) {
            if (commits[i].second.Entry.FaceCount == 0) continue;
            derive.EntryIndex = i;
            derive.Work = phase == 0 ? commits[i].second.Faces : commits[i].second.Normals;
            encode::BindCompute(encoder, pipelines.VertexNormalDerive, slots, buffers);
            encode::SetPushConstants(encoder, derive);
            DispatchWork(encoder, buffers, derive.Work);
        }
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    }
}
} // namespace

std::vector<entt::entity> CommitPosedGeometry(entt::registry &r, entt::entity viewport, std::span<const entt::entity> mesh_entities) {
    const profile::CpuScope scope{"CommitGeometry"};
    const auto *pending = r.try_get<const PendingTransform>(viewport);
    if (!pending) return {};
    const auto primaries = selection::ComputePrimaryEditInstances(r, false);
    auto &buffers = r.ctx().get<GpuBuffers>();
    std::vector<std::pair<entt::entity, CommitPosedGeometryPushConstants>> commits;
    for (const auto entity : mesh_entities) {
        if (const auto primary = primaries.find(entity); primary != primaries.end())
            commits.emplace_back(entity, PrepareGeometryEdit(r, viewport, entity, primary->second, pending));
    }
    if (commits.empty()) return {};
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto *cb = ctx.Queue->commandBuffer();
    {
        mtl::PassChain chain{cb};
        auto *encoder = chain.BeginCompute("CommitGeometry");
        RecordGeometryEditBatch(r, encoder, commits, false);
    }
    cb->commit();
    cb->waitUntilCompleted();
    std::vector<entt::entity> changed;
    for (const auto &[entity, pc] : commits) {
        if (!ElementWorkEmpty(buffers.GeometryWork, pc.ChangedVertices)) {
            changed.push_back(entity);
            auto &w = r.ctx().get<GpuSceneState>().EditWork.at(entity);
            w.Modified = true;
            w.PreviewActive = true;
        }
    }
    return changed;
}

namespace {
void RecordSparseEditPrelude(entt::registry &r, entt::entity viewport, mtl::PassChain &chain) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &state = r.ctx().get<GpuSceneState>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto *pending = r.try_get<const PendingTransform>(viewport);
    const auto primaries = selection::ComputePrimaryEditInstances(r, false);
    std::vector<std::pair<entt::entity, CommitPosedGeometryPushConstants>> jobs;
    for (const auto &[entity, pose] : state.PosedByEntity) {
        const auto primary = primaries.find(entity);
        const bool preview = pending && primary != primaries.end();
        const auto old = state.EditWork.find(entity);
        if (!preview && (old == state.EditWork.end() || !old->second.PreviewActive)) continue;
        jobs.emplace_back(entity, PrepareGeometryEdit(r, viewport, entity, preview ? primary->second : entt::null, preview ? pending : nullptr, &pose));
    }
    if (jobs.empty()) return;
    auto *encoder = chain.BeginCompute("EditGeometry", MTL::StageDispatch);
    RecordGeometryEditBatch(r, encoder, jobs, true);
    const auto entries = buffers.BoundsReduceEntries.GetSpan<DrawData>({0, buffers.BoundsReduceEntries.Count<DrawData>()});
    for (const auto &[entity, job] : jobs) {
        auto &w = state.EditWork.at(entity);
        const auto &pose = state.PosedByEntity.at(entity);
        const auto entry_it = std::ranges::find(entries, pose.PositionBase, &DrawData::PosedPositionOffset);
        assert(entry_it != entries.end());
        const auto entry_index = uint32_t(entry_it - entries.begin());
        const auto first_tile = buffers.BoundsEntryFirstTiles.GetSpan<uint32_t>({entry_index, 1}).front();
        RecordPosedMeshletBounds(encoder, slots, pipelines, buffers, 0, {.Work = w.Meshlets, .FirstTile = pose.MeshletBoundsBase});
        RecordBoundsPass(encoder, slots, pipelines.BoundsReduce, buffers, PreludeSlot::BoundsReduce, 0, {.Work = w.BoundsTiles, .NextWork = w.BoundsLevels.front().Work, .EntryIndex = entry_index});
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        SlotOffset input{buffers.BoundsPartials.Slot, first_tile};
        uint32_t input_count = w.BoundsTiles.Count;
        for (uint32_t i = 0; i < w.BoundsLevels.size(); ++i) {
            const auto &level = w.BoundsLevels[i];
            const bool last = i + 1 == w.BoundsLevels.size();
            const CommitPosedGeometryPushConstants finish{.BoundsTiles = level.Work};
            encode::BindCompute(encoder, pipelines.GeometryWorkArgs, slots, buffers);
            encode::SetPushConstants(encoder, finish);
            encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256);
            encoder->memoryBarrier(MTL::BarrierScopeBuffers);
            const BoundsTreePushConstants tree{
                .Work = w.BoundsInitialized ? level.Work : ElementWork{.Count = level.Work.Count},
                .NextWork = last ? ElementWork{} : w.BoundsLevels[i + 1].Work,
                .Input = input,
                .Output = buffers.BoundsParents.Slotted(level.Values),
                .InputCount = input_count,
                .InstanceBounds = {buffers.Instances.BoundsBuffer.Slot, entry_it->FirstInstance},
                .InstanceCount = last ? entry_it->ElementIdOffset : 0u,
            };
            encode::BindCompute(encoder, pipelines.BoundsTree, slots, buffers);
            encode::SetPushConstants(encoder, tree);
            if (w.BoundsInitialized)
                encoder->dispatchThreadgroups(*buffers.GeometryWork.Buffer, WorkArgsOffset(level.Work, true), ThreadgroupSize::Linear256);
            else encoder->dispatchThreadgroups(MTL::Size(level.Work.Count, 1, 1), ThreadgroupSize::Linear256);
            encoder->memoryBarrier(MTL::BarrierScopeBuffers);
            input = tree.Output;
            input_count = level.Work.Count;
        }
        w.BoundsInitialized = true;
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
        {live ? groups.BoundsReduce : 0u, 1u, 1u},
        {live ? groups.DeriveGather : 0u, 1u, 1u},
        {live ? groups.BoundsCombine : 0u, 1u, 1u},
    }};
    buffers.PreludeDispatchArgs.Update(as_bytes(args));
}
