#include "viewport/ViewportRenderGpu.h"
#include "ProcessEvents.h"
#include "animation/AnimationTimeline.h"
#include "animation/MorphWeightState.h"
#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "gizmo/TransformGizmoTypes.h"
#include "gpu/BoundsBoxPushConstants.h"
#include "gpu/BoundsReducePushConstants.h"
#include "gpu/CornerClassEncoding.h"
#include "gpu/DepthPyramidReducePushConstants.h"
#include "gpu/FrustumCullPushConstants.h"
#include "gpu/MainDrawPushConstants.h"
#include "gpu/MeshPrimitiveTopology.h"
#include "gpu/MeshletCullPushConstants.h"
#include "gpu/MeshletDrawPushConstants.h"
#include "gpu/MeshletGeometryEncoding.h"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/MotionBlurGatherPushConstants.h"
#include "gpu/MotionBlurTilesDilatePushConstants.h"
#include "gpu/MotionBlurTilesFlattenPushConstants.h"
#include "gpu/NormalDeriveEntry.h"
#include "gpu/NormalDerivePushConstants.h"
#include "gpu/PosedMeshletBoundsPushConstants.h"
#include "gpu/SilhouetteEdgeColorPushConstants.h"
#include "gpu/SilhouetteEdgeDepthObjectPushConstants.h"
#include "gpu/VisibilityShadingPushConstants.h"
#include "mesh/MeshStore.h"
#include "metal/PassChain.h"
#include "metal/RenderTarget.h"
#include "render/Drawing.h"
#include "render/Encoding.h"
#include "render/Instance.h"
#include "render/Pipelines.h"
#include "render/Profile.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionGpu.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
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
MTL::PrimitiveType PipelinePrimitive(SPT spt) {
    switch (spt) {
        case SPT::Line:
        case SPT::LineOverlayFaceNormals:
        case SPT::LineOverlayVertexNormals:
        case SPT::ObjectExtrasLine:
        case SPT::BoundsBox:
        case SPT::BoneWire:
        case SPT::BoneSphereWire:
        case SPT::SelectionObjectExtrasLines:
            return MTL::PrimitiveTypeLine;
        case SPT::Point:
            return MTL::PrimitiveTypePoint;
        default:
            return MTL::PrimitiveTypeTriangle;
    }
}

DrawData MakeDrawData(const RenderBuffers &rb, uint32_t vertex_slot, const InstanceArena &instances) {
    return MakeDrawData(vertex_slot, rb.Vertices, rb.Indices, instances.TransformBuffer.Slot);
}
// The element a mesh draws from: triangulated faces, or the edges or vertices of a face-less mesh.
enum class ElementDomain {
    Face,
    Edge,
    Vertex
};
void RecordDrawCounters(const GpuBuffers &buffers, const DrawListBuilder &draw_list, bool selection) {
    const auto record = [&](const char *selection_name, const char *render_name, size_t value) { profile::RecordCounter(selection ? selection_name : render_name, value); };
    record("Selection DrawData", "Render DrawData", draw_list.Draws.size());
    record("Selection CullEntries", "Render CullEntries", draw_list.CullEntries.size());
    record("Selection IndirectCommands", "Render IndirectCommands", draw_list.IndirectCommands.size());
    record("Selection MaxIndexCount", "Render MaxIndexCount", draw_list.MaxIndexCount);
    record("Selection DrawDataBytes", "Render DrawDataBytes", draw_list.Draws.size() * sizeof(DrawData));
    record("Selection CullEntryBytes", "Render CullEntryBytes", draw_list.CullEntries.size() * sizeof(CullEntry));
    record("Selection IndirectCommandBytes", "Render IndirectCommandBytes", draw_list.IndirectCommands.size() * sizeof(MTL::DrawIndexedPrimitivesIndirectArguments));
    if (!selection) {
        profile::RecordCounter("InstanceSlots", buffers.Instances.TransformBuffer.UsedSize / sizeof(Transform));
        profile::RecordCounter("MeshletRecords", buffers.Meshlets.Buffer.Count<MeshletRecord>());
        profile::RecordCounter("MeshletInstances", buffers.MeshletInstanceCount);
        profile::RecordCounter("MeshletRecordBytes", buffers.Meshlets.Buffer.UsedSize);
        profile::RecordCounter("MeshletTriangleIdBytes", buffers.MeshletTriangleIds.Buffer.UsedSize);
        profile::RecordCounter("PrimitiveRecordBytes", buffers.Primitives.Buffer.UsedSize);
        profile::RecordCounter("InstanceRecordBytes", buffers.Instances.RecordBuffer.UsedSize);
        if (buffers.MeshletRoutes.Contents().size() >= sizeof(MeshletRouteState)) {
            const auto &routes = *reinterpret_cast<const MeshletRouteState *>(buffers.MeshletRoutes.Contents().data());
            const auto count = [&](MeshletRoute route) { return routes.Counts[uint32_t(route)]; };
            profile::RecordCounter("VisibleOpaqueMeshlets", count(MeshletRoute::OpaqueCullBack) + count(MeshletRoute::OpaqueCullFront) + count(MeshletRoute::OpaqueDoubleSided) + count(MeshletRoute::Coverage));
            profile::RecordCounter("VisibleCoverageMeshlets", count(MeshletRoute::Coverage));
            profile::RecordCounter("VisibleBlendMeshlets", count(MeshletRoute::Blend));
            profile::RecordCounter("VisibleTransmissionMeshlets", count(MeshletRoute::Transmission));
            static const bool record_routes = std::getenv("MESHEDITOR_MESHLET_ROUTE_COUNTERS") != nullptr;
            if (record_routes) {
                profile::RecordCounter("MeshletRoute OpaqueCullBack", count(MeshletRoute::OpaqueCullBack));
                profile::RecordCounter("MeshletRoute Blend", count(MeshletRoute::Blend));
                profile::RecordCounter("MeshletRoute Transmission", count(MeshletRoute::Transmission));
                profile::RecordCounter("MeshletRoute Silhouette", count(MeshletRoute::Silhouette));
                profile::RecordCounter("MeshletRoute Phase2Candidate", count(MeshletRoute::Phase2Candidate));
                profile::RecordCounter("MeshletRoute OpaqueCullFront", count(MeshletRoute::OpaqueCullFront));
                profile::RecordCounter("MeshletRoute OpaqueDoubleSided", count(MeshletRoute::OpaqueDoubleSided));
                profile::RecordCounter("MeshletRoute Coverage", count(MeshletRoute::Coverage));
            }
        }
        profile::RecordCounter("DeviceAllocatedBytes", buffers.Ctx.Ctx.Device->currentAllocatedSize());
    }
}
} // namespace

void FlushDrawList(entt::registry &r, const DrawListBuilder &draw_list, DrawBufferPair &pair) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    const bool selection = &pair == &buffers.SelectionDraw;
    if (selection) RecordDrawCounters(buffers, draw_list, true);
    buffers.EnsureIdentityIndexBuffer(std::max(draw_list.MaxIndexCount, uint32_t(draw_list.Draws.size())));
    if (!draw_list.Draws.empty() || buffers.Prelude.PosePrepass > 0 || buffers.Prelude.PosedMeshletBounds > 0 || buffers.Prelude.DeriveFaces > 0 || buffers.Prelude.BoundsReduce > 0) {
        pair.DrawData.Update(as_bytes(draw_list.Draws));
        pair.CullEntries.Update(as_bytes(draw_list.CullEntries));
        pair.VisibleIndices.Update(buffers.IdentityIndexBuffer.Contents().subspan(0, draw_list.Draws.size() * sizeof(uint32_t)));
    }
    if (!draw_list.IndirectCommands.empty()) pair.Indirect.Update(as_bytes(draw_list.IndirectCommands));
}

namespace {
struct DeformSlots {
    uint32_t BoneDeformOffset{InvalidOffset}, ArmatureDeformOffset{InvalidOffset}, MorphDeformOffset{InvalidOffset};
    uint32_t MorphTargetCount{0};
    // Per-instance armature palette: buffer_index -> offset (instances of one mesh can bind different armatures)
    std::unordered_map<uint32_t, uint32_t> ArmatureDeformByBufferIndex;
    // Per-instance morph weights: buffer_index -> offset (weights are per-node in glTF)
    std::unordered_map<uint32_t, uint32_t> MorphWeightsByBufferIndex;
};

std::unordered_map<entt::entity, DeformSlots> BuildDeformSlots(const entt::registry &r, const MeshStore &meshes) {
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
        }
    }
    // Add morph target slots for mesh instances with morph data (per-instance weights)
    for (const auto [instance_entity, instance, gpu_range, ri] : r.view<const Instance, const MorphWeightGpuRange, const RenderInstance>().each()) {
        const auto mesh_entity = instance.Entity;
        const auto &mesh = GetMesh(r, mesh_entity);
        const auto morph_range = meshes.GetMorphTargetRange(mesh.GetStoreId());
        if (morph_range.Count == 0) continue;
        auto &slots = result[mesh_entity];
        slots.MorphDeformOffset = morph_range.Offset;
        slots.MorphTargetCount = meshes.GetMorphTargetCount(mesh.GetStoreId());
        slots.MorphWeightsByBufferIndex[ri.BufferIndex] = gpu_range.Weights.Offset;
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
// Set per-draw posed ranges (positions and derived normals), keyed by the draw's instance.
void PatchPosedRanges(std::span<DrawData> draws, const PosedRanges &posed) {
    for (auto &d : draws) {
        const auto i = posed.PerInstance ? d.FirstInstance - posed.FirstInstance : 0u;
        d.PosedPositionOffset = posed.PositionOffset(i);
        if (const auto normals = posed.NormalsAt(i)) {
            d.PosedVertexNormalOffset = normals->VertexOffset;
            d.PosedSeamNormalOffset = normals->SeamOffset;
            d.PosedFaceNormalOffset = normals->FaceOffset;
        }
    }
}
// AppendExtrasDraw is templated on the customize_draw callable.
void AppendExtrasDraw(entt::registry &r, const InstanceArena &instances, DrawListBuilder &dl, DrawBatchInfo &batch, auto &&customize_draw) {
    batch = dl.BeginBatch();
    for (auto [entity, mesh_buffers, models] : r.view<ObjectExtrasTag, const MeshBuffers, const ModelsBuffer>().each()) {
        if (mesh_buffers.EdgeIndices.Count == 0) continue;
        auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, instances);
        if (const auto *vcr = r.try_get<VertexClass>(entity)) draw.VertexClassOffset = vcr->Offset;
        customize_draw(draw, instances);
        AppendDraw(dl, batch, mesh_buffers.EdgeIndices, models, draw);
    }
}
// Reduce this step's screen motion to tiles, spread each tile's motion over the tiles it crosses,
// then gather the scene along it. Leaves the blurred scene in GatherImage. The scene pass already
// wrote the motion into the velocity attachment.
void RecordMotionBlurPostFx(entt::registry &r, mtl::PassChain &chain, const mtl::BindlessSet &slots, entt::entity viewport, mtl::Extent2D extent, uint32_t ubo_offset, float playback_frame) {
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &main = pipelines.Main;
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    const auto &settings = r.get<const ViewportDisplay>(viewport);
    const auto mb = EffectiveMotionBlur(settings);
    // The second half of each motion vector is stored pointing backward, which the negative y undoes.
    constexpr vec2 MotionScale{1.f, -1.f};
    // Golden-ratio stepping decorrelates the gather's dither across steps and frames.
    const float noise_offset = glm::fract(playback_frame * std::numbers::phi_v<float>);

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

FrustumCullPushConstants MakeCullPushConstants(const GpuBuffers &buffers, const DrawBufferPair &pair, const DrawListBuilder &draw_list) {
    return {
        .CommandsSlot = pair.Indirect.Slot,
        .CullEntrySlot = pair.CullEntries.Slot,
        .DrawDataSlot = pair.DrawData.Slot,
        .VisibleIndexSlot = pair.VisibleIndices.Slot,
        .BoundsSlot = buffers.Instances.BoundsBuffer.Slot,
        .ModelSlot = buffers.Instances.TransformBuffer.Slot,
        .EntryCount = uint32_t(draw_list.Draws.size()),
        .CommandCount = uint32_t(draw_list.IndirectCommands.size()),
    };
}

void DispatchCull(MTL::ComputeCommandEncoder *encoder, const FrustumCullPushConstants &cull_pc, uint32_t count) {
    static constexpr uint32_t GroupSize{64};
    encode::SetPushConstants(encoder, cull_pc);
    encoder->dispatchThreadgroups(MTL::Size((count + GroupSize - 1) / GroupSize, 1, 1), ThreadgroupSize::Linear64);
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

// The input fields of a mesh's normal-derive entry, with the position source and output offsets left unset.
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

void RecordFrustumCull(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const Pipelines &pipelines, const GpuBuffers &buffers, const DrawBufferPair &pair, const DrawListBuilder &draw_list, uint32_t ubo_offset) {
    encode::BindCompute(encoder, pipelines.FrustumCull, slots, buffers, ubo_offset);
    auto cull_pc = MakeCullPushConstants(buffers, pair, draw_list);
    DispatchCull(encoder, cull_pc, cull_pc.CommandCount);
    // Phase 1 accumulates into the counts phase 0 zeroed, through bindless buffers.
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    cull_pc.Phase = 1;
    DispatchCull(encoder, cull_pc, cull_pc.EntryCount);
}
} // namespace

void RecordFrustumCull(mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines, const GpuBuffers &buffers, const DrawBufferPair &pair, const DrawListBuilder &draw_list) {
    // The cull overwrites the indirect commands and the visible-index remap that the draws read.
    auto *encoder = chain.BeginCompute("FrustumCull", MTL::StageVertex);
    RecordFrustumCull(encoder, slots, pipelines, buffers, pair, draw_list, 0);
}

namespace {
// Materialize each posed entry's current-pose vertex positions.
void RecordPosePrepass(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const Pipelines &pipelines, const GpuBuffers &buffers, uint32_t vertex_state_slot, uint32_t ubo_offset) {
    const auto &prepass = pipelines.PosePrepass;
    encode::BindCompute(encoder, prepass, slots, buffers, ubo_offset);
    const BoundsReducePushConstants pc{
        .VertexStateSlot = vertex_state_slot,
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

BoundsReducePushConstants MakeBoundsReducePc(const GpuBuffers &buffers) {
    return {
        .DrawDataSlot = buffers.BoundsReduceEntries.Slot,
        .BoundsSlot = buffers.Instances.BoundsBuffer.Slot,
        .TileMapSlot = buffers.BoundsTiles.Slot,
        .PartialBoundsSlot = buffers.BoundsPartials.Slot,
        .EntryFirstTileSlot = buffers.BoundsEntryFirstTiles.Slot,
    };
}

void RecordBoundsPass(MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const mtl::ComputePipeline &pipeline, const GpuBuffers &buffers, PreludeSlot slot, uint32_t ubo_offset) {
    const auto pc = MakeBoundsReducePc(buffers);
    encode::BindCompute(encoder, pipeline, slots, buffers, ubo_offset);
    encode::SetPushConstants(encoder, pc);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector, 0);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::BoundsFoldVector, 1);
    DispatchPrelude(encoder, buffers, slot);
}

void RecordPosedMeshletBounds(
    MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots,
    const Pipelines &pipelines, const GpuBuffers &buffers, uint32_t ubo_offset
) {
    const PosedMeshletBoundsPushConstants pc{
        .DrawDataSlot = buffers.BoundsReduceEntries.Slot,
        .TileMapSlot = buffers.PosedMeshletBoundsTiles.Slot,
        .MeshletSlot = buffers.Meshlets.Buffer.Slot,
        .PrimitiveSlot = buffers.Primitives.Buffer.Slot,
        .MeshletVertexSlot = buffers.MeshletVertexCorners.Buffer.Slot,
        .PosedMeshletBoundsSlot = buffers.PosedMeshletBounds.Slot,
    };
    encode::BindCompute(encoder, pipelines.PosedMeshletBounds, slots, buffers, ubo_offset);
    encode::SetPushConstants(encoder, pc);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::MeshletBoundsFoldVector, 0);
    encoder->setThreadgroupMemoryLength(ThreadgroupMemory::MeshletBoundsFoldVector, 1);
    encoder->dispatchThreadgroups(
        *buffers.PreludeDispatchArgs, PreludeArgsOffset(PreludeSlot::PosedMeshletBounds), ThreadgroupSize::Linear64
    );
}

// The cull push constants' buffer-derived fields, shared by the phase-1 and phase-2 records.
MeshletCullPushConstants MakeMeshletCullSlotsPc(const GpuBuffers &buffers) {
    return {
        .WorkRangeSlot = buffers.MeshletWorkRanges.Slot,
        .WorkBlockSlot = buffers.MeshletWorkBlocks.Slot,
        .WorkBlockStateSlot = buffers.MeshletWorkBlockStates.Slot,
        .WorkStateSlot = buffers.MeshletWorkState.Slot,
        .WorkDispatchArgsSlot = buffers.MeshletWorkDispatchArgs.Slot,
        .BlockStateSlot = buffers.MeshletCullBlocks.Slot,
        .ClassificationSlot = buffers.MeshletClassifications.Slot,
        .VisibleSlot = buffers.VisibleMeshlets.Slot,
        .InstanceMapSlot = buffers.GpuInstanceSlots.Buffer.Slot,
        .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
        .PrimitiveSlot = buffers.Primitives.Buffer.Slot,
        .MeshletSlot = buffers.Meshlets.Buffer.Slot,
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
        .Phase2BlockCountSlot = buffers.MeshletPhase2CullBlockCounts.Slot,
    };
}

void RecordMeshletCullImpl(
    MTL::ComputeCommandEncoder *encoder, const mtl::BindlessSet &slots, const Pipelines &pipelines,
    GpuBuffers &buffers, MeshletCullConfig config
) {
    const bool transmission = config.Mode == MeshletRouteMode::Transmission;
    struct WorkCapacity {
        uint64_t Ranges, Meshlets;
    };
    const auto work_with_flags = [&](uint32_t flags) -> WorkCapacity {
        if (flags == 0u) return {buffers.MeshletRangeCount, buffers.MeshletInstanceCount};
        const auto slots = buffers.GpuInstanceSlots.Buffer.GetSpan<uint32_t>({0, buffers.GpuInstanceSlots.Buffer.Count<uint32_t>()});
        const auto records = buffers.Instances.RecordBuffer.GetSpan<InstanceRecord>({0, buffers.Instances.RecordBuffer.Count<InstanceRecord>()});
        const auto primitives = buffers.Primitives.Buffer.GetSpan<PrimitiveRecord>({0, buffers.Primitives.Buffer.Count<PrimitiveRecord>()});
        WorkCapacity capacity{};
        for (const uint32_t slot : slots) {
            if (slot == InvalidOffset || slot >= records.size() || (records[slot].Flags & flags) != flags) continue;
            capacity.Ranges += records[slot].PrimitiveCount;
            for (uint32_t p = 0; p < records[slot].PrimitiveCount; ++p) {
                capacity.Meshlets += primitives[records[slot].PrimitiveOffset + p].MeshletCount;
            }
        }
        return capacity;
    };
    const WorkCapacity primary = work_with_flags(config.RequiredInstanceFlags);
    const WorkCapacity extra = config.ExtraRouteFlags != 0u ? work_with_flags(config.ExtraRouteFlags) : WorkCapacity{};
    buffers.EnsureMeshletVisibilityCapacity(
        primary.Meshlets * (1u + transmission) + extra.Meshlets,
        primary.Ranges,
        primary.Meshlets,
        std::max(primary.Meshlets, extra.Meshlets),
        config.SortBlend,
        config.TwoPhase
    );
    const auto pc = [&] {
        auto pc = MakeMeshletCullSlotsPc(buffers);
        pc.InstanceCount = buffers.GpuInstanceSlots.Buffer.Count<uint32_t>();
        pc.WorkBlockCount = buffers.MeshletWorkBlockStates.Count<MeshletWorkBlockState>();
        pc.BlendBlockSlot = config.SortBlend ? buffers.MeshletBlendBlocks.Slot : InvalidSlot;
        pc.RouteMode = uint32_t(config.Mode);
        pc.RequiredInstanceFlags = config.RequiredInstanceFlags;
        pc.ExtraInstanceFlags = config.ExtraRouteFlags;
        pc.PyramidSamplerSlot = config.PyramidSamplerSlot;
        pc.TwoPhase = config.TwoPhase;
        return pc;
    }();
    encode::BindScene(encoder, slots, buffers, config.UboOffset);
    encode::SetPushConstants(encoder, pc);
    constexpr uint32_t simd_groups = GpuBuffers::MeshletCullBlockSize / 32u;
    constexpr uint32_t prefix_stride = simd_groups + 1u;
    const auto dispatch_work = [&](const mtl::ComputePipeline &pipeline) {
        if (pc.WorkBlockCount == 0) return;
        encoder->setComputePipelineState(pipeline.State());
        encoder->setThreadgroupMemoryLength(AlignedThreadgroupBytes(3u * prefix_stride * sizeof(uint32_t)), 0);
        encoder->dispatchThreadgroups(MTL::Size(pc.WorkBlockCount, 1, 1), MTL::Size(GpuBuffers::MeshletCullBlockSize, 1, 1));
    };
    const auto dispatch_meshlets = [&](const mtl::ComputePipeline &pipeline) {
        encoder->setComputePipelineState(pipeline.State());
        const uint32_t prefix_bytes = GpuBuffers::MeshletRouteCount * prefix_stride * sizeof(uint32_t);
        const uint32_t blend_bytes = config.SortBlend ? simd_groups * 256u * sizeof(uint16_t) : 0u;
        encoder->setThreadgroupMemoryLength(AlignedThreadgroupBytes(prefix_bytes + blend_bytes), 0);
        encoder->dispatchThreadgroups(*buffers.MeshletWorkDispatchArgs, 0, MTL::Size(GpuBuffers::MeshletCullBlockSize, 1, 1));
    };
    dispatch_work(pipelines.MeshletWorkBlockCount);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    encoder->setComputePipelineState(pipelines.MeshletWorkPrefix.State());
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    dispatch_work(pipelines.MeshletWorkEmit);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    dispatch_meshlets(pipelines.MeshletCullBlockCount);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    encoder->setComputePipelineState(pipelines.MeshletCullPrefix.State());
    encoder->setThreadgroupMemoryLength(AlignedThreadgroupBytes(GpuBuffers::MeshletRouteCount * sizeof(uint32_t)), 0);
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256);
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    dispatch_meshlets(pipelines.MeshletCullEmit);
}

void DrawMeshletList(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, const mtl::Buffer &visible,
    const mtl::Buffer &routes, const mtl::Buffer &dispatch_args, uint32_t route, uint32_t required_instance_flags,
    uint32_t visibility_phase = 0u, bool visibility_transmission = false, bool fragment_pc = false
) {
    // Visibility IDs reserve 25 bits for the visible-list index; overflowing would alias the phase bit.
    if (fragment_pc) assert(visible.UsedSize / sizeof(VisibleMeshlet) <= (uint64_t{1} << 25u));
    MeshletDrawPushConstants pc{
        .PrimitiveSlot = buffers.Primitives.Buffer.Slot,
        .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
        .InstanceMapSlot = buffers.GpuInstanceSlots.Buffer.Slot,
        .MeshletSlot = buffers.Meshlets.Buffer.Slot,
        .MeshletTriangleSlot = buffers.MeshletTriangleIds.Buffer.Slot,
        .MeshletVertexSlot = buffers.MeshletVertexCorners.Buffer.Slot,
        .MeshletLocalTriangleSlot = buffers.MeshletLocalTriangles.Buffer.Slot,
        .VisibleMeshletSlot = visible.Slot,
        .RouteStateSlot = routes.Slot,
        .Route = route,
        .RequiredInstanceFlags = required_instance_flags,
        .VisibilityPhase = visibility_phase,
        .VisibilityTransmission = visibility_transmission,
    };
    for (uint32_t chunk = 0; chunk < buffers.MeshletDispatchChunkCount; ++chunk) {
        pc.VisibleOffset = chunk * GpuBuffers::MeshletDispatchChunkSize;
        if (fragment_pc) encode::SetPushConstants(encoder, pc);
        else encode::SetMeshPushConstants(encoder, pc);
        const auto args_offset = (route * buffers.MeshletDispatchChunkCount + chunk) * sizeof(MeshDispatchArgs);
        encoder->drawMeshThreadgroups(*dispatch_args, args_offset, MTL::Size(1, 1, 1), MTL::Size(160, 1, 1));
    }
}

bool DrawVisibilityRoute(uint32_t route_mask, MeshletRoute route) {
    return (route_mask & (1u << uint32_t(route))) != 0u;
}

uint32_t VisibilityRouteMask(const GpuBuffers &buffers, bool transmission) {
    uint32_t result = 1u << uint32_t(MeshletRoute::OpaqueCullBack);
    const uint32_t non_triangle_mask = (1u << uint32_t(MeshPrimitiveTopology::Line)) |
        (1u << uint32_t(MeshPrimitiveTopology::Point));
    if ((buffers.MeshletTopologyMask & non_triangle_mask) != 0u) {
        result |= 1u << uint32_t(MeshletRoute::Coverage);
    }
    const auto transforms = buffers.Instances.TransformBuffer.GetSpan<Transform>({0u, buffers.Instances.TransformBuffer.Count<Transform>()});
    if (std::ranges::any_of(transforms, [](const Transform &world) {
            return world.S.x * world.S.y * world.S.z < 0.0f;
        })) {
        result |= 1u << uint32_t(MeshletRoute::OpaqueCullFront);
    }
    for (uint32_t i = 0u; i < buffers.Materials.Count(); ++i) {
        const auto &material = buffers.Materials.Get(i);
        if (material.DoubleSided != 0u) result |= 1u << uint32_t(MeshletRoute::OpaqueDoubleSided);
        if (material.AlphaMode == MaterialAlphaMode::Mask ||
            (transmission && material.Unlit == 0u && material.Transmission.Factor > 0.0f &&
             material.Transmission.Texture.Slot != InvalidSlot)) {
            result |= 1u << uint32_t(MeshletRoute::Coverage);
        }
    }
    return result;
}

void DrawPhase2Meshlets(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, const MainPipeline &main
) {
    // Coarse whole-instance candidates have not yet seen material classification. Keep phase 2 on
    // one conservative two-sided, coverage-capable route; splitting its inner cull by route makes
    // the serial prefix proportional to route count and regresses disocclusion-heavy scenes.
    encoder->setCullMode(MTL::CullModeNone);
    main.MeshletVisibilityCoverage.Bind(encoder);
    DrawMeshletList(
        encoder, buffers, buffers.MeshletPhase2Visible, buffers.MeshletPhase2Routes,
        buffers.MeshletPhase2DispatchArgs, uint32_t(MeshletRoute::OpaqueCullBack), 0u, 1u, false, true
    );
}

void DrawVisibilityMeshlets(
    MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, const MainPipeline &main,
    bool transmission, uint32_t route_mask
) {
    const auto draw = [&](MeshletRoute route) {
        if (!DrawVisibilityRoute(route_mask, route)) return;
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

VisibilityShadingPushConstants MakeVisibilityShadingPc(const GpuBuffers &buffers) {
    return {
        .PrimitiveSlot = buffers.Primitives.Buffer.Slot,
        .InstanceSlot = buffers.Instances.RecordBuffer.Slot,
        .InstanceMapSlot = buffers.GpuInstanceSlots.Buffer.Slot,
        .MeshletSlot = buffers.Meshlets.Buffer.Slot,
        .MeshletTriangleSlot = buffers.MeshletTriangleIds.Buffer.Slot,
        .MeshletLocalTriangleSlot = buffers.MeshletLocalTriangles.Buffer.Slot,
        .MeshletVertexSlot = buffers.MeshletVertexCorners.Buffer.Slot,
        .VisibleMeshletSlot = buffers.VisibleMeshlets.Slot,
        .Phase2VisibleMeshletSlot = buffers.MeshletPhase2Visible.Slot,
    };
}

void RecordMeshletPhase2Cull(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines,
    GpuBuffers &buffers, uint32_t pyramid_sampler, uint32_t ubo_offset, MeshletRouteMode mode
) {
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
        // Each group reads the mip the previous group wrote, through bindless images the encoder
        // cannot see, so the groups need an explicit edge.
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

// Record one phase's passes into `cb`, which is already begun with viewport and scissor set.
// `ubo_offset` selects the view UBO instance every bind in the phase reads.
void RecordPhase(entt::registry &r, entt::entity viewport, mtl::PassChain &chain, DrawListUse use, RenderPhase phase, uint32_t ubo_offset, float playback_frame) {
    const profile::CpuScope scope{"RecordRenderCommandBuffer"};
    if (use == DrawListUse::Rebuild) r.ctx().get<DrawState>().SelectionStale = true;
    // The multi-step blur splits the scene and its overlays across two phases so the overlays stay
    // sharp over the averaged steps. Full and BlurredFull draw both in one.
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
    auto &draw = r.ctx().get<DrawState>();
    auto &draw_list = draw.List;

    // Edit mode uses the rest pose; reused blur phases use slots built by the rebuild phase.
    const auto mesh_deform_slots = is_edit_mode || use == DrawListUse::Reuse ?
        std::unordered_map<entt::entity, DeformSlots>{} :
        BuildDeformSlots(r, meshes);
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

    // In Edit mode, compute primary edit instances (for draw routing and silhouette filtering)
    // and edit transform context (for pending vertex transforms) in one pass.
    std::unordered_map<entt::entity, entt::entity> primary_edit_instances;
    EditTransformContext edit_transform_context;
    const bool has_pending_transform = is_edit_mode && r.all_of<PendingTransform>(viewport);
    if (is_edit_mode) {
        const auto active = FindActiveEntity(r);
        for (const auto [e, instance, ok, ri] : r.view<const Instance, const Selected, const ObjectKind, const RenderInstance>().each()) {
            if (ok.Value != ObjectType::Mesh) continue;
            if (auto &primary = primary_edit_instances[instance.Entity]; primary == entt::entity{} || e == active) primary = e;
            if (has_pending_transform && !r.all_of<ScaleLocked>(e)) {
                auto &primary_uf = edit_transform_context.TransformInstances[instance.Entity];
                if (primary_uf == entt::entity{} || e == active) primary_uf = e;
            }
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
    // Rewrite per-instance deform and posed-buffer fields on the draws appended since `draws_before`.
    const auto patch_mesh_draws = [&](DrawListBuilder &list, size_t draws_before, entt::entity mesh_entity, const DeformSlots &deform) {
        const auto draws = std::span{list.Draws}.subspan(draws_before);
        PatchInstanceDeform(draws, deform);
        if (const auto it = draw.PosedByEntity.find(mesh_entity); it != draw.PosedByEntity.end()) {
            PatchPosedRanges(draws, it->second);
        }
    };

    if (use == DrawListUse::Rebuild) {
        const profile::CpuScope build_scope{"BuildDrawList"};
        draw_list.Draws.clear();
        draw_list.CullEntries.clear();
        draw_list.IndirectCommands.clear();
        draw_list.MaxIndexCount = 0;
        draw.PosedByEntity.clear();

        std::unordered_set<entt::entity> excitable_mesh_entities;
        if (is_excite_mode) {
            for (const auto [e, instance, excitable] : r.view<const Instance, const SoundVertices>().each()) {
                excitable_mesh_entities.emplace(instance.Entity);
            }
        }

        // Single-pass entity data collection: avoids redundant ECS view iterations across fill, edge, wire, point, and normal batches.
        struct MeshEntityData {
            entt::entity Entity;
            const MeshBuffers &Buf;
            const ModelsBuffer &Mod;
            std::optional<Mesh> MeshComp;
            const DeformSlots &Deform;
            std::optional<uint32_t> PrimaryEditBufferIndex;
            ElementDomain Domain; // The element the mesh draws from.
            bool IsSoundVertices, IsBone, IsBoneJoint, IsExtras;
        };
        const auto element_domain = [](const MeshBuffers &b) {
            return b.FaceIndices.Count > 0 ? ElementDomain::Face : b.EdgeIndices.Count > 0 ? ElementDomain::Edge :
                                                                                             ElementDomain::Vertex;
        };

        // Draw order resolves coincident surfaces, and pool iteration order varies with scene-load
        // history, so draw in descending entity id order (a fresh registry's iteration order).
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
            const bool is_bone_joint = r.all_of<BoneJoint>(entity);
            mesh_entities.emplace_back(entity, mesh_buffers, models, TryGetMesh(r, entity), get_deform_slots(entity), primary_bi, element_domain(mesh_buffers), excitable_mesh_entities.contains(entity), r.all_of<ArmatureObject>(entity) || is_bone_joint, is_bone_joint, r.all_of<ObjectExtrasTag>(entity));
        }

        // The mesh shades authored under morphing: rest normals plus weighted authored deltas.
        // Edit mode builds no deform slots, so edit-mode draws (including drags) derive.
        const auto morph_shading_authored = [&meshes](const MeshEntityData &e) {
            return e.Deform.MorphDeformOffset != InvalidOffset && e.MeshComp && meshes.GetMorphShadingAuthored(e.MeshComp->GetStoreId());
        };

        // A face-less mesh with a material table is a point or line mesh, which material and rendered modes shade in the scene pass.
        // Solid and wireframe draw it through the wire and point overlays, as do face-less overlays (bones, sound vertices), which carry no material table.
        const auto shaded_in_scene_pass = [&meshes, show_rendered](const MeshEntityData &e) {
            return show_rendered && e.MeshComp && e.MeshComp->FaceCount() == 0 && meshes.GetPrimitiveMaterialRange(e.MeshComp->GetStoreId()).Count > 0;
        };
        // Shaded meshes take their selection feedback from the scene pass, which recolors selected line and point fills.
        // Edit mode shows element state through its own overlays.
        // Draws a mesh's wire or point overlay for every instance, unless the scene pass shades it.
        const auto append_overlay = [&](DrawBatchInfo &batch, const MeshEntityData &e, const SlottedRange &indices, const DrawData &dd) {
            if (shaded_in_scene_pass(e)) return;
            const auto draws_before = draw_list.Draws.size();
            AppendDraw(draw_list, batch, indices, e.Mod, dd);
            patch_mesh_draws(draw_list, draws_before, e.Entity, e.Deform);
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
            };
            std::vector<BoundsEntrySpec> specs(mesh_entities.size());
            // Posed entries and their tiles come first: the pose pre-pass dispatches over that tile prefix.
            uint32_t entry_count = 0, posed_entry_count = 0, posed_vertex_count = 0;
            uint32_t derive_entry_count = 0, vertex_normal_count = 0, seam_normal_count = 0, face_normal_count = 0;
            uint32_t posed_tile_count = 0, bounds_tile_count = 0, derive_face_tile_count = 0, derive_gather_tile_count = 0;
            uint32_t posed_meshlet_bounds_count = 0;
            bool authored_morph_any = false;
            for (size_t mi = 0; mi < mesh_entities.size(); ++mi) {
                const auto &e = mesh_entities[mi];
                auto &spec = specs[mi];
                if (!e.MeshComp || e.Mod.InstanceCount == 0) continue;
                if (has_pending_transform) {
                    if (const auto it = edit_transform_context.TransformInstances.find(e.Entity); it != edit_transform_context.TransformInstances.end()) {
                        spec.PendingPrimary = r.try_get<const RenderInstance>(it->second);
                    }
                }
                spec.PerInstanceDeform = !e.Deform.ArmatureDeformByBufferIndex.empty() || !e.Deform.MorphWeightsByBufferIndex.empty();
                spec.Count = spec.PerInstanceDeform ? e.Mod.InstanceCount : 1u;
                spec.Posed = e.Deform.BoneDeformOffset != InvalidOffset || e.Deform.MorphDeformOffset != InvalidOffset || spec.PendingPrimary != nullptr;
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
                    posed_meshlet_bounds_count += spec.Count * e.Buf.Meshlets.Count;
                }
            }
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
            // The rebuild rewrote the buffers the prelude reads and writes, so the next submit re-runs it.
            buffers.PreludeStale = true;
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
                // PosedRanges owns the posed-buffer layout: bases here, per-instance offsets via its accessors.
                PosedRanges pr{};
                NormalDeriveEntry derive_entry = spec.Entry;
                if (spec.Posed) {
                    pr = {
                        .FirstInstance = e.Mod.InstanceRange.Offset,
                        .PerInstance = spec.PerInstanceDeform,
                        .PositionBase = posed_offset,
                        .VertexCount = e.Buf.Vertices.Count,
                        .MeshletBoundsBase = meshlet_bounds_offset,
                        .MeshletCount = e.Buf.Meshlets.Count,
                        .Normals = spec.Derive ?
                            std::optional{PosedRanges::NormalRanges{vertex_normal_offset, seam_normal_offset, face_normal_offset, spec.Entry.SeamCount, spec.Entry.FaceCount}} :
                            std::nullopt,
                    };
                    draw.PosedByEntity.emplace(e.Entity, pr);
                    posed_offset += spec.Count * pr.VertexCount;
                    meshlet_bounds_offset += spec.Count * pr.MeshletCount;
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
                        for (uint32_t t = 0; t < face_tiles_per; ++t) derive_tiles[face_tile_write++] = {derive_write, t};
                        for (uint32_t t = 0; t < gather_tiles_per; ++t) derive_tiles[gather_tile_write++] = {derive_write, t};
                        derive_entries[derive_write++] = derive_entry;
                    }
                    entry_first_tiles[write] = tile_write;
                    for (uint32_t t = 0; t < bounds_tiles_per; ++t) bounds_tiles[tile_write++] = {write, t};
                    entries[write++] = entry;
                }
                // Shared-pose instances share one entry and one set of meshlet bounds, like positions.
                if (spec.Posed) {
                    for (uint32_t i = 0; i < spec.Count; ++i) {
                        for (uint32_t m = 0; m < e.Buf.Meshlets.Count; ++m) {
                            posed_meshlet_tiles[meshlet_tile_write++] = {first + i, e.Buf.Meshlets.Offset + m};
                        }
                    }
                }
                if (spec.PerInstanceDeform) PatchInstanceDeform(entries.subspan(first, spec.Count), e.Deform);
            }
        }

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
            if (const auto it = draw.PosedByEntity.find(instance.Entity); it != draw.PosedByEntity.end()) {
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
            if (!is_edit_mode || (primary != primary_edit_instances.end() && primary->second == instance_entity)) {
                record.ElementStateSlotOffset = meshes.GetFaceStateRange(GetMesh(r, instance.Entity).GetStoreId());
            }
            buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>({ri.BufferIndex, 1}).front() = record;
        }

        // Build bone batches for X-ray rendering (drawn after a mid-pass depth clear so bones are never occluded by scene meshes)
        // Only draw bones for: active armature in Edit/Pose mode, selected armatures in Object mode.
        const auto should_draw_armature_bones = [&](entt::entity arm_obj_entity) {
            if (is_wireframe_mode) return true; // Wireframe mode: always show bone outlines
            const bool is_bone_mode = is_edit_mode || interaction_mode == InteractionMode::Pose;
            if (is_bone_mode) return r.all_of<Active>(arm_obj_entity);
            return r.all_of<Selected>(arm_obj_entity);
        };
        // Map BoneJoint entities back to their owning armature object entities.
        std::unordered_map<entt::entity, entt::entity> joint_to_owner;
        for (const auto [e, arm_obj] : r.view<const ArmatureObject>().each()) {
            if (arm_obj.JointEntity != entt::null) joint_to_owner[arm_obj.JointEntity] = e;
        }
        draw.BoneFill = {};
        draw.BoneWire = {};
        draw.BoneSphereFill = {};
        draw.BoneSphereWire = {};
        if (show_overlays && settings.ShowBones) {
            draw.BoneFill = draw_list.BeginBatch();
            for (const auto [entity, arm_obj, mesh_buffers, models] : r.view<const ArmatureObject, const MeshBuffers, const ModelsBuffer>().each()) {
                if (mesh_buffers.FaceIndices.Count == 0) continue;
                auto fill_draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, buffers.Instances);
                fill_draw.InstanceStateSlot = buffers.Instances.StateBuffer.Slot;
                AppendDraw(draw_list, draw.BoneFill, mesh_buffers.FaceIndices, models, fill_draw);
            }
            draw.BoneWire = draw_list.BeginBatch();
            for (const auto [entity, arm_obj, mesh_buffers, models] : r.view<const ArmatureObject, const MeshBuffers, const ModelsBuffer>().each()) {
                if (!should_draw_armature_bones(entity)) continue;
                if (const auto *adj = r.try_get<const BoneAdjacencyIndices>(entity)) {
                    auto wire_draw = MakeDrawData(mesh_buffers.Vertices, adj->Indices, buffers.Instances);
                    wire_draw.InstanceStateSlot = buffers.Instances.StateBuffer.Slot;
                    AppendDraw(draw_list, draw.BoneWire, adj->Indices.Count / 2, models, wire_draw);
                }
            }

            // Joint sphere batches
            draw.BoneSphereFill = draw_list.BeginBatch();
            for (const auto [entity, mesh_buffers, models] : r.view<const BoneJoint, const MeshBuffers, const ModelsBuffer>().each()) {
                if (mesh_buffers.FaceIndices.Count == 0) continue;
                auto fill_draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, buffers.Instances);
                fill_draw.InstanceStateSlot = buffers.Instances.StateBuffer.Slot;
                AppendDraw(draw_list, draw.BoneSphereFill, mesh_buffers.FaceIndices, models, fill_draw);
            }
            draw.BoneSphereWire = draw_list.BeginBatch();
            for (const auto [entity, mesh_buffers, models] : r.view<const BoneJoint, const MeshBuffers, const ModelsBuffer>().each()) {
                if (mesh_buffers.EdgeIndices.Count == 0) continue;
                if (const auto it = joint_to_owner.find(entity); it != joint_to_owner.end() && !should_draw_armature_bones(it->second)) continue;
                auto wire_draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, buffers.Instances);
                wire_draw.InstanceStateSlot = buffers.Instances.StateBuffer.Slot;
                AppendDraw(draw_list, draw.BoneSphereWire, mesh_buffers.EdgeIndices, models, wire_draw);
            }
        }

        // Edge quad batch (edit/excite mode triangle quads with self-AA, matches Blender's overlay_edit_mesh_edge)
        draw.EdgeQuad = draw_list.BeginBatch();
        if (show_overlays && (is_edit_mode || is_excite_mode)) {
            for (const auto &e : mesh_entities) {
                if (e.IsBone || e.IsExtras || !e.MeshComp || e.Buf.EdgeIndices.Count == 0) continue;
                auto dd = MakeDrawData(e.Buf.Vertices, e.Buf.EdgeIndices, buffers.Instances, e.Deform.BoneDeformOffset, e.Deform.ArmatureDeformOffset, e.Deform.MorphDeformOffset, e.Deform.MorphTargetCount);
                dd.ElementStateSlotOffset = meshes.GetEdgeStateRange(e.MeshComp->GetStoreId());
                if (is_edit_mode) {
                    const auto sharpness = meshes.GetEdgeSharpnessRange(e.MeshComp->GetStoreId());
                    dd.EdgeSharpnessOffset = sharpness.Count > 0 ? sharpness.Offset : InvalidOffset;
                }
                const auto db = draw_list.Draws.size();
                if (e.PrimaryEditBufferIndex) AppendDraw(draw_list, draw.EdgeQuad, e.Buf.EdgeIndices.Count * 3, e.Mod, dd, e.PrimaryEditBufferIndex);
                else if (e.IsSoundVertices) AppendDraw(draw_list, draw.EdgeQuad, e.Buf.EdgeIndices.Count * 3, e.Mod, dd);
                patch_mesh_draws(draw_list, db, e.Entity, e.Deform);
            }
        }
        // Wire line batch (wireframe mode + line meshes, matches Blender's wireframe overlay)
        draw.WireLine = draw_list.BeginBatch();
        for (const auto &e : mesh_entities) {
            if (e.IsBone || e.IsExtras || !e.MeshComp || e.Buf.EdgeIndices.Count == 0) continue;
            if (e.Buf.FaceIndices.Count > 0 && !is_wireframe_mode) continue;
            auto dd = MakeDrawData(e.Buf.Vertices, e.Buf.EdgeIndices, buffers.Instances, e.Deform.BoneDeformOffset, e.Deform.ArmatureDeformOffset, e.Deform.MorphDeformOffset, e.Deform.MorphTargetCount);
            dd.ElementStateSlotOffset = meshes.GetEdgeStateRange(e.MeshComp->GetStoreId());
            append_overlay(draw.WireLine, e, e.Buf.EdgeIndices, dd);
        }

        draw.ExtrasLine = {};
        if (show_overlays && settings.ShowExtras) {
            AppendExtrasDraw(r, buffers.Instances, draw_list, draw.ExtrasLine, [](auto &, const auto &) {});
        }

        draw.Point = draw_list.BeginBatch();
        for (const auto &e : mesh_entities) {
            if (e.IsBone) continue;
            // Edit mode shows a mesh's vertices under vertex select, and excite mode its excitable ones.
            const bool draw_elements = show_overlays && ((is_edit_mode && edit_mode == Element::Vertex && e.PrimaryEditBufferIndex) || (is_excite_mode && e.IsSoundVertices));
            // A vertex-only mesh otherwise draws its points as the object itself, until it is the one being edited.
            const bool draw_object = e.Domain == ElementDomain::Vertex && !(is_edit_mode && e.PrimaryEditBufferIndex);
            if (!draw_elements && !draw_object) continue;
            auto dd = MakeDrawData(e.Buf.Vertices, e.Buf.VertexIndices, buffers.Instances, e.Deform.BoneDeformOffset, e.Deform.ArmatureDeformOffset, e.Deform.MorphDeformOffset, e.Deform.MorphTargetCount);
            dd.ElementStateSlotOffset = {meshes.GetVertexStateSlot(), e.Buf.Vertices.Offset};
            if (draw_elements) {
                const auto draws_before = draw_list.Draws.size();
                AppendDraw(draw_list, draw.Point, e.Buf.VertexIndices, e.Mod, dd, e.PrimaryEditBufferIndex);
                patch_mesh_draws(draw_list, draws_before, e.Entity, e.Deform);
            } else {
                append_overlay(draw.Point, e, e.Buf.VertexIndices, dd);
            }
        }

        { // Normal overlay + AABB batches
            const auto vertex_slot = buffers.VertexBuffer.Buffer.Slot;
            draw.OverlayFaceNormals = draw_list.BeginBatch();
            for (const auto &e : mesh_entities) {
                if (auto it = e.Buf.NormalIndicators.find(Element::Face); it != e.Buf.NormalIndicators.end()) {
                    auto dd = MakeDrawData(it->second, vertex_slot, buffers.Instances);
                    AppendDraw(draw_list, draw.OverlayFaceNormals, it->second.Indices, e.Mod, dd);
                }
            }
            draw.OverlayVertexNormals = draw_list.BeginBatch();
            for (const auto &e : mesh_entities) {
                if (auto it = e.Buf.NormalIndicators.find(Element::Vertex); it != e.Buf.NormalIndicators.end()) {
                    auto dd = MakeDrawData(it->second, vertex_slot, buffers.Instances);
                    AppendDraw(draw_list, draw.OverlayVertexNormals, it->second.Indices, e.Mod, dd);
                }
            }
        }

        { // Build selection draw list
            auto &sel_list = draw.SelectionList;
            sel_list = {};

            const auto run_sel_pass = [&](auto &&indices_of, auto &&skip) {
                auto batch = sel_list.BeginBatch(true);
                for (const auto &e : mesh_entities) {
                    if (e.IsExtras || e.IsBoneJoint || skip(e)) continue;
                    const auto &indices = indices_of(e);
                    auto dd = MakeDrawData(e.Buf.Vertices, indices, buffers.Instances, e.Deform.BoneDeformOffset, e.Deform.ArmatureDeformOffset, e.Deform.MorphDeformOffset, e.Deform.MorphTargetCount);
                    dd.ObjectIdSlot = buffers.Instances.ObjectIdBuffer.Slot;
                    const auto db = sel_list.Draws.size();
                    if (e.PrimaryEditBufferIndex) AppendDraw(sel_list, batch, indices, e.Mod, dd, e.PrimaryEditBufferIndex);
                    else AppendDraw(sel_list, batch, indices, e.Mod, dd);
                    patch_mesh_draws(sel_list, db, e.Entity, e.Deform);
                }
                return batch;
            };
            const auto sel_line = run_sel_pass(
                [](const auto &e) -> const auto & { return e.Buf.EdgeIndices; },
                [&](const auto &e) { return e.Buf.FaceIndices.Count > 0 || e.Buf.EdgeIndices.Count == 0 || shaded_in_scene_pass(e); }
            );
            const auto sel_point = run_sel_pass(
                [](const auto &e) -> const auto & { return e.Buf.VertexIndices; },
                [&](const auto &e) { return e.Buf.FaceIndices.Count > 0 || e.Buf.EdgeIndices.Count > 0 || shaded_in_scene_pass(e); }
            );

            DrawBatchInfo sel_bone_sphere;
            if (show_overlays && settings.ShowBones) {
                sel_bone_sphere = sel_list.BeginBatch(true);
                for (const auto [entity, mesh_buffers, models] : r.view<const BoneJoint, const MeshBuffers, const ModelsBuffer>().each()) {
                    if (mesh_buffers.FaceIndices.Count == 0) continue;
                    auto dd = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, buffers.Instances);
                    dd.ObjectIdSlot = buffers.Instances.ObjectIdBuffer.Slot;
                    AppendDraw(sel_list, sel_bone_sphere, mesh_buffers.FaceIndices, models, dd);
                }
            }

            DrawBatchInfo sel_extras;
            if (show_overlays && settings.ShowExtras) {
                AppendExtrasDraw(r, buffers.Instances, sel_list, sel_extras, [](auto &dd, const auto &instances) {
                    dd.ObjectIdSlot = instances.ObjectIdBuffer.Slot;
                });
            }

            draw.SelectionDraws = {
                {SPT::SelectionFragment, MTL::PrimitiveTypeLine, sel_line},
                {SPT::SelectionFragment, MTL::PrimitiveTypePoint, sel_point},
                {SPT::SelectionFragmentBoneSphere, MTL::PrimitiveTypeTriangle, sel_bone_sphere},
                {SPT::SelectionObjectExtrasLines, MTL::PrimitiveTypeLine, sel_extras},
            };
        }
    }
    const auto instance_records = buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>(
        {0, buffers.Instances.RecordBuffer.Count<InstanceRecord>()}
    );
    for (const auto [instance_entity, ri] : r.view<const RenderInstance>().each()) {
        if (ri.BufferIndex == UINT32_MAX || ri.BufferIndex >= instance_records.size()) continue;
        auto &record = instance_records[ri.BufferIndex];
        record.ObjectId = ri.ObjectId;
        const bool selected = r.all_of<Selected>(instance_entity) && is_silhouette_eligible(instance_entity);
        const bool silhouette = selected && (!is_edit_mode || silhouette_instances.contains(instance_entity));
        record.Flags = silhouette ? uint32_t(MeshletInstanceFlag::Silhouette) : 0u;
    }
    const bool has_object_silhouette_selection =
        any_of(r.view<const Selected, const Instance, const RenderInstance>().each(), [&](const auto &entry) { return is_silhouette_eligible(std::get<0>(entry)); });
    const bool render_silhouette = (show_overlays && settings.ShowOutlineSelected) && !is_excite_mode &&
        (is_edit_mode ? !silhouette_instances.empty() : has_object_silhouette_selection);

    if (use != DrawListUse::Reuse) {
        { // Bounding boxes draw straight from the instance arena: one box per selected mesh instance slot.
            auto &slots_buffer = buffers.BoundsBoxSlots;
            uint32_t box_count = 0;
            if (show_overlays && settings.ShowExtras && settings.ShowBoundingBoxes) {
                const auto boxes = r.view<const Selected, const Instance, const RenderInstance>();
                const auto slots = slots_buffer.SetCount<uint32_t>(uint32_t(boxes.size_hint()));
                for (const auto [e, instance, ri] : boxes.each()) {
                    if (ri.BufferIndex != UINT32_MAX && HasMesh(r, instance.Entity)) slots[box_count++] = ri.BufferIndex;
                }
            }
            slots_buffer.UsedSize = uint64_t(box_count) * sizeof(uint32_t);
        }

        FlushDrawList(r, draw_list, buffers.RenderDraw);
    }
    // A newly rebuilt draw list is the authoritative active-topology set. Specialize forward PBR
    // here so scene loading needs no second registry scan and the first mixed-topology frame is correct.
    if (show_rendered && use != DrawListUse::Reuse) {
        pipelines.Main.Compiler.CompileTopologyPipelines(
            pipelines.Libraries, (buffers.MeshletTopologyMask & ~1u) != 0u
        );
    }
    if (use != DrawListUse::Reuse || phase == RenderPhase::Full) RecordDrawCounters(buffers, draw_list, false);

    const bool transmission_active = real_transmission && pipelines.Main.Transmission;
    // The transmission composite path lays the prepass down as the scene's background and
    // plain-opaque pixels, loading the prepass's depth. Edit mode re-rasterizes for face tints,
    // blur steps for velocity, and debug channels carry values the composite's exposure would corrupt.
    const bool composite_transmission = transmission_active && phase == RenderPhase::Full && !is_edit_mode && settings.DebugChannel == DebugChannel::None;
    const bool meshlet_fill = buffers.MeshletInstanceCount > 0;

    const uint32_t transform_vertex_state_slot = is_edit_mode ? meshes.GetVertexStateSlot() : InvalidSlot;

    // The posed passes run every phase, since blur steps read their step's captured pose through the phase's UBO instance.
    // Bounds and cull run once per command buffer, and later blur phases reuse the culled buffers.
    // Derived normals feed only the scene's face-fill draws, so only scene-drawing phases record the derive.
    // Every prelude pass dispatches indirectly.
    // A submit with unchanged deform inputs gets zero group counts, keeping the buffers' current results.
    if (!draw_list.Draws.empty() || buffers.Prelude.PosePrepass > 0 || buffers.Prelude.PosedMeshletBounds > 0 || buffers.Prelude.DeriveFaces > 0 || buffers.Prelude.BoundsReduce > 0) {
        const auto &prelude = buffers.Prelude;
        const bool record_bounds = phase != RenderPhase::BlurAccumulate && phase != RenderPhase::BlurResolve;
        // A derive entry always holds at least one face tile and one gather tile, so one count decides both phases.
        const bool record_derive = draw_scene && prelude.DeriveFaces > 0;
        const bool bounds_work = record_bounds && prelude.BoundsCombine > 0;
        auto *compute = chain.BeginCompute("Prelude", MTL::StageVertex | MTL::StageFragment | MTL::StageDispatch);
        // The prelude's outputs flow through bindless buffers the encoder cannot see, so every
        // producing dispatch needs an explicit edge before its consumer: the pose pre-pass feeds
        // posed bounds, the derives, and the bounds reduce, the face derive feeds the gather, and
        // the bounds reduce feeds the combine.
        if (prelude.PosePrepass > 0) {
            RecordPosePrepass(compute, slots, pipelines, buffers, transform_vertex_state_slot, ubo_offset);
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
        if (record_bounds && !draw_list.Draws.empty()) {
            RecordFrustumCull(compute, slots, pipelines, buffers, buffers.RenderDraw, draw_list, ubo_offset);
        }
    }
    MTL::RenderCommandEncoder *encoder = nullptr;
    auto record_batch = [&](const DrawBatchInfo &batch, MTL::PrimitiveType primitive) {
        encode::SetPushConstants(encoder, MainDrawPushConstants{batch.DrawDataSlotOffset});
        encode::DrawIndexedIndirect(encoder, primitive, *buffers.IdentityIndexBuffer, *buffers.RenderDraw.Indirect, batch.IndirectOffset, batch.DrawCount);
    };
    auto record_draw_batch = [&](const PipelineRenderer &renderer, SPT spt, const DrawBatchInfo &batch) {
        if (batch.DrawCount == 0) return;
        renderer.Bind(encoder, spt);
        record_batch(batch, PipelinePrimitive(spt));
    };
    const auto record_meshlets = [&](uint32_t route, auto &&bind_pipeline) {
        bind_pipeline();
        DrawMeshlets(encoder, buffers, route);
    };
    auto draw_quad = [&] { encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4)); };

    const auto &main = pipelines.Main;
    const auto main_extent = main.Resources->SceneColorImage.Extent;
    const bool has_silhouette = render_silhouette && meshlet_fill;
    // Wireframe can still show selection outlines. Populate visibility for silhouette decode even
    // when face fill is hidden, then leave its depth out of the wireframe scene pass below.
    const bool need_visibility = meshlet_fill && (show_fill || has_silhouette);
    const bool cull_scene_meshlets = draw_scene && need_visibility;
    const uint32_t visibility_route_mask = VisibilityRouteMask(buffers, real_transmission);
    bool sort_blend = false;
    if (cull_scene_meshlets && show_rendered) {
        for (uint32_t i = 0; i < buffers.Materials.Count() && !sort_blend; ++i) {
            sort_blend = buffers.Materials.Get(i).AlphaMode == MaterialAlphaMode::Blend;
        }
    }
    const auto view_bytes = buffers.SceneViewUBO.Contents().subspan(ubo_offset, sizeof(SceneViewUBO));
    const auto &current_view_proj = reinterpret_cast<const SceneViewUBO *>(view_bytes.data())->ViewProj;
    const bool disocclusion_possible = use != DrawListUse::Reuse || buffers.PreludeStale || buffers.MeshletOcclusionStale ||
        std::memcmp(&current_view_proj, &buffers.PreviousFullCullViewProj, sizeof(mat4)) != 0;
    // Real transmission stays single-phase because phase-2 visibility does not run its per-pixel
    // textured transmission-hole coverage test.
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
                .RequiredInstanceFlags = show_fill ? 0u : uint32_t(MeshletInstanceFlag::Silhouette),
                .UboOffset = ubo_offset,
                .PyramidSamplerSlot = pyramid,
                .ExtraRouteFlags = 0u,
                .SortBlend = sort_blend,
                .TwoPhase = two_phase_meshlets,
            }
        );
    }
    if (need_visibility) {
        const std::array colors{mtl::ClearColor(*main.Resources->VisibilityImage, MTL::ClearColor{double(UINT32_MAX), 0, 0, 0})};
        const auto pass = mtl::MakePassDescriptor(colors, mtl::ClearDepth(*main.Resources->DepthImage));
        encoder = encode::BeginScenePass(
            chain, pass, "MeshletVisibility", {{MTL::StageDispatch, MTL::StageMesh}}, main_extent,
            slots, buffers, ubo_offset
        );
        DrawVisibilityMeshlets(encoder, buffers, main, real_transmission, visibility_route_mask);
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

    // Real-transmission pre-pass: render the background and opaque faces into TransmissionImage mip 0,
    // sampled by the scene pass at the refracted exit point. TRANSMISSION_PREPASS variants skip
    // exposure and drop transmission materials (no self-sampling).
    if (transmission_active && draw_scene) {
        // Refraction sees the world, and nothing where there is no world. The viewport backdrop is
        // display-referred UI drawn with the overlays, so it never reaches this buffer.
        const std::array colors{mtl::ClearColor(main.Transmission->Mip0View.get())};
        const auto pass = mtl::MakePassDescriptor(colors, mtl::LoadDepth(*main.Resources->DepthImage));
        encoder = encode::BeginScenePass(chain, pass, "TransmissionPrepass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, ubo_offset);
        main.PrepassBackground.Bind(encoder);
        draw_quad();
        if (meshlet_fill && show_fill) {
            main.Compiler.BindVisibility(encoder, PbrCompiler::Variant::OpaquePrepass);
            encoder->setFragmentTexture(*main.Resources->VisibilityImage, 0u);
            encode::SetPushConstants(encoder, MakeVisibilityShadingPc(buffers));
            draw_quad();
        }

        // Generate the transmission mip chain sampled across roughness.
        if (main.Transmission->Image.MipLevels > 1) {
            auto *blit = chain.BeginBlit("TransmissionMips", MTL::StageFragment);
            blit->generateMipmaps(*main.Transmission->Image);
        }
    }

    // Blurred steps render the scene through the velocity render pass variant, so opaque geometry
    // writes its screen motion alongside its color.
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

        // Screen motion for every pixel geometry leaves uncovered. The background sits at infinity,
        // so only view rotation moves it. Geometry overwrites it wherever it lands.
        if (blur) {
            scene_renderer.Bind(encoder, SPT::BackgroundVelocity);
            draw_quad();
        }
        // The prepass covers the background and plain-opaque geometry, so the composite replaces both.
        if (composite_transmission) {
            scene_renderer.Bind(encoder, SPT::TransmissionComposite);
            draw_quad();
        } else if (show_rendered && draw_scene) {
            // Background environment (PBR modes only). The shader discards with no world opacity or env slot.
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

        // Solid faces. BlurResolve writes depth only, for overlays to occlude against (blend faces never wrote depth).
        if (show_fill) {
            if (meshlet_fill && draw_scene && show_rendered) {
                const auto opaque_variant = blur ? PbrCompiler::Variant::OpaqueVelocity : PbrCompiler::Variant::Opaque;
                const auto blend_variant = blur ? PbrCompiler::Variant::BlendVelocity : PbrCompiler::Variant::Blend;
                if (!composite_transmission) {
                    main.Compiler.BindVisibility(encoder, opaque_variant);
                    encoder->setFragmentTexture(*main.Resources->VisibilityImage, 0u);
                    encode::SetPushConstants(encoder, MakeVisibilityShadingPc(buffers));
                    draw_quad();
                }
                if (real_transmission) record_meshlets(uint32_t(MeshletRoute::Transmission), [&] { main.Compiler.BindMeshlets(encoder, opaque_variant); });
                record_meshlets(uint32_t(MeshletRoute::Blend), [&] { main.Compiler.BindMeshlets(encoder, blend_variant); });
            } else if (meshlet_fill && draw_scene) {
                main.WorkspaceVisibility.Bind(encoder);
                encoder->setFragmentTexture(*main.Resources->VisibilityImage, 0u);
                encode::SetPushConstants(encoder, MakeVisibilityShadingPc(buffers));
                draw_quad();
            }
        }
    }

    // Phase 2 needs the meshlet-only pyramid above. The persistent next-frame pyramid is rebuilt
    // after the scene pass so classic opaque draw items contribute occlusion too.
    if (show_fill && phase == RenderPhase::Full && cull_scene_meshlets) {
        auto *compute = chain.BeginCompute("DepthPyramidFinal", MTL::StageFragment);
        RecordDepthPyramid(compute, slots, buffers, pipelines, sel_slots, ubo_offset);
        main.Resources->DepthPyramidValid = true;
    }

    if (blur) RecordMotionBlurPostFx(r, chain, slots, viewport, main_extent, ubo_offset, playback_frame);

    if (!draw_overlays) { // BlurAccumulate sums this step's blurred scene in. It draws no overlays to composite.
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

    // The layer clears transparent, so an untouched one composites to nothing. Track whether anything
    // reaches it, and the composite skips reading it and its line data at all. Every draw in the pass
    // below goes through here or sets the flag itself.
    bool overlay_layer_drawn = false;
    // Everything the pass could draw. With nothing to draw, the composite reads neither overlay layer.
    const bool overlay_pass_needed = has_silhouette ||
        (show_overlays && settings.ShowGrid) ||
        draw.EdgeQuad.DrawCount > 0 || draw.WireLine.DrawCount > 0 || draw.Point.DrawCount > 0 || draw.ExtrasLine.DrawCount > 0 || buffers.BoundsBoxSlots.UsedSize > 0 ||
        draw.OverlayFaceNormals.DrawCount > 0 || draw.OverlayVertexNormals.DrawCount > 0 ||
        draw.BoneFill.DrawCount > 0 || draw.BoneWire.DrawCount > 0 || draw.BoneSphereFill.DrawCount > 0 || draw.BoneSphereWire.DrawCount > 0;
    if (overlay_pass_needed) { // Overlay pass: display-referred overlays over transparent, depth-tested against the scene above.
        // Transparent overlay color is composited over scene color by alpha.
        const std::array overlay_colors{
            mtl::ClearColor(*main.Resources->OverlayColorImage),
            mtl::ClearColor(*main.Resources->LineDataImage),
        };
        const auto overlay_pass = mtl::MakePassDescriptor(overlay_colors, mtl::LoadDepth(*main.Resources->DepthImage));
        encoder = encode::BeginScenePass(chain, overlay_pass, "OverlayPass", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageFragment}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, ubo_offset);

        const auto record_overlay_batch = [&](SPT spt, const DrawBatchInfo &batch) {
            if (batch.DrawCount == 0) return;
            overlay_layer_drawn = true;
            record_draw_batch(main.OverlayRenderer, spt, batch);
        };

        if (buffers.IdentityIndexCount > 0) {
            // Edit mode edges as triangle quads with self-AA
            record_overlay_batch(SPT::EdgeQuad, draw.EdgeQuad);
            // Wireframe/line mesh edges as GPU lines (the composite handles AA)
            record_overlay_batch(SPT::Line, draw.WireLine);
            // Vertex points (always recorded — batch is empty when nothing qualifies)
            record_overlay_batch(SPT::Point, draw.Point);
            // Object extras (cameras, lights, empties)
            record_overlay_batch(SPT::ObjectExtrasLine, draw.ExtrasLine);
        }

        // Bounding boxes, generated in the vertex shader from the instance arena's bounds and transforms.
        if (const auto box_count = buffers.BoundsBoxSlots.Count<uint32_t>(); box_count > 0) {
            overlay_layer_drawn = true;
            main.OverlayRenderer.Bind(encoder, SPT::BoundsBox);
            encode::SetPushConstants(encoder, BoundsBoxPushConstants{
                                                  .SlotsSlot = buffers.BoundsBoxSlots.Slot,
                                                  .BoundsSlot = buffers.Instances.BoundsBuffer.Slot,
                                                  .ModelSlot = buffers.Instances.TransformBuffer.Slot,
                                                  .StateSlot = buffers.Instances.StateBuffer.Slot,
                                              });
            encoder->drawPrimitives(MTL::PrimitiveTypeLine, NS::UInteger(0), NS::UInteger(24), NS::UInteger(box_count));
        }

        // Silhouette edge color (rendered ontop of meshes)
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
            } else if (!is_edit_mode) {
                if (active_entity != entt::null && r.all_of<RenderInstance>(active_entity)) {
                    active_object_id = r.get<RenderInstance>(active_entity).ObjectId;
                }
            }
            encode::SetPushConstants(encoder, SilhouetteEdgeColorPushConstants{TransformGizmo::IsUsing(r, viewport) && interaction_mode == InteractionMode::Object, sel_slots.ObjectIdSampler, active_object_id});
            draw_quad();
        }

        if (buffers.IdentityIndexCount > 0) { // Selection overlays
            record_overlay_batch(SPT::LineOverlayFaceNormals, draw.OverlayFaceNormals);
            record_overlay_batch(SPT::LineOverlayVertexNormals, draw.OverlayVertexNormals);
        }

        // Grid plane (drawn before bone depth clear so grid remains depth-tested against scene meshes)
        if (show_overlays && settings.ShowGrid) {
            overlay_layer_drawn = true;
            main.OverlayRenderer.Bind(encoder, SPT::Grid);
            encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(9));
        }

        { // Bone X-ray: depth clears so bones are never occluded by scene meshes, only by each other.
            // Metal clears an attachment at pass start alone, so the bones take a second pass that
            // loads the overlay colors and clears the depth.
            if (draw.BoneFill.DrawCount > 0 || draw.BoneSphereFill.DrawCount > 0) {
                const std::array bone_colors{
                    mtl::LoadColor(*main.Resources->OverlayColorImage),
                    mtl::LoadColor(*main.Resources->LineDataImage),
                };
                const auto bone_pass = mtl::MakePassDescriptor(bone_colors, mtl::ClearDepth(*main.Resources->DepthImage));
                encoder = encode::BeginScenePass(chain, bone_pass, "BoneXRay", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageFragment}, {MTL::StageFragment, MTL::StageFragment}}, main_extent, slots, buffers, ubo_offset);

                // In Object+wireframe mode, show only outlines (no fills).
                // In Edit/Pose+wireframe, fills are semitransparent and write far-plane depth (via shader) so wires are never occluded.
                const bool object_wireframe = is_wireframe_mode && interaction_mode == InteractionMode::Object;
                if (!object_wireframe) {
                    record_overlay_batch(SPT::BoneFill, draw.BoneFill);
                    record_overlay_batch(SPT::BoneSphereFill, draw.BoneSphereFill);
                }
                // In non-wireframe Object mode, "Outline selected" off suppresses bone wire outlines.
                // In wireframe+Object mode, wires are the only bone visualization so always show them.
                const bool hide_bone_outlines = !is_wireframe_mode && interaction_mode == InteractionMode::Object &&
                    (!show_overlays || !settings.ShowOutlineSelected);
                if (!hide_bone_outlines) {
                    record_overlay_batch(SPT::BoneWire, draw.BoneWire);
                    record_overlay_batch(SPT::BoneSphereWire, draw.BoneSphereWire);
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
        // BlurredFull leaves the finished scene in the gather target, which holds the scene color it blurred.
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

void RecordMeshletCull(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const Pipelines &pipelines,
    GpuBuffers &buffers, MeshletCullConfig config
) {
    auto *encoder = chain.BeginCompute("MeshletCull", MTL::StageMesh | MTL::StageFragment);
    RecordMeshletCullImpl(encoder, slots, pipelines, buffers, config);
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
    encode::SetPushConstants(encoder, MakeVisibilityShadingPc(buffers));
    encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
}

void DrawMeshlets(MTL::RenderCommandEncoder *encoder, const GpuBuffers &buffers, uint32_t route, uint32_t required_instance_flags) {
    DrawMeshletList(encoder, buffers, buffers.VisibleMeshlets, buffers.MeshletRoutes, buffers.MeshletDispatchArgs, route, required_instance_flags);
}

void RecordRenderCommandBuffer(entt::registry &r, entt::entity viewport, MTL::CommandBuffer *command_buffer, DrawListUse use, RenderPhase phase) {
    profile::BeginRecording(command_buffer);
    mtl::PassChain chain{command_buffer, profile::RecordingTimer()};
    RecordPhase(r, viewport, chain, use, phase, 0, r.get<const PlaybackFrame>(viewport).Value);
    profile::EndRecording();
}

// Record every blur step and the resolve into one command buffer.
void RecordBlurStepsCommandBuffer(entt::registry &r, entt::entity viewport, MTL::CommandBuffer *command_buffer, std::span<const float> step_frames) {
    const auto &buffers = r.ctx().get<const GpuBuffers>();
    profile::BeginRecording(command_buffer);
    mtl::PassChain chain{command_buffer, profile::RecordingTimer()};
    for (uint32_t i = 0; i < step_frames.size(); ++i) {
        RecordPhase(r, viewport, chain, i == 0 ? DrawListUse::Rebuild : DrawListUse::Reuse, i == 0 ? RenderPhase::BlurAccumulateFirst : RenderPhase::BlurAccumulate, buffers.SceneViewUboOffset(i + 1), step_frames[i]);
    }
    RecordPhase(r, viewport, chain, DrawListUse::Reuse, RenderPhase::BlurResolve, 0, r.get<const PlaybackFrame>(viewport).Value);
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
    // The one-shot rewrote the per-frame derive entry and tile buffers, so the next submit rebuilds the draw list.
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
// Decide whether the listed mesh entities keep their authored shading normals under morphing.
// Targets authoring normal deltas decide on the CPU alone.
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
        // Targets authoring normal deltas settle the gate here.
        meshes.UpdateMorphShadingAuthored(*mesh, {});
        if (meshes.GetMorphShadingAuthored(store_id)) continue;
        const auto vertex_count = entry_inputs->VertexCount;
        const auto targets = meshes.GetMorphTargets(store_id);
        for (uint32_t t = 0; t < target_count; ++t) {
            // A target without position deltas leaves the pose at rest, pinning nothing.
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
        const auto store_id = GetMesh(r, jobs[i].Entity).GetStoreId();
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

// Copy the mesh's posed positions and derived normals from the last submitted frame (fenced complete) into the canonical stores.
// Returns true when any position changed.
bool CommitPosedGeometry(entt::registry &r, entt::entity mesh_entity) {
    const auto &posed_by_entity = r.ctx().get<const DrawState>().PosedByEntity;
    const auto it = posed_by_entity.find(mesh_entity);
    if (it == posed_by_entity.end()) return false;
    const auto &pr = it->second;
    auto &meshes = r.ctx().get<MeshStore>();
    const auto &buffers = r.ctx().get<const GpuBuffers>();
    const auto id = GetMesh(r, mesh_entity).GetStoreId();
    const auto posed_positions = buffers.PosedPositions.GetSpan<vec3>({pr.PositionOffset(0), pr.VertexCount});
    auto vertices = meshes.GetVertices(id);
    bool any_moved = false;
    for (uint32_t vi = 0; vi < pr.VertexCount; ++vi) {
        if (vertices[vi].Position != posed_positions[vi]) {
            vertices[vi].Position = posed_positions[vi];
            any_moved = true;
        }
    }
    if (any_moved) {
        if (const auto normals = pr.NormalsAt(0)) {
            std::ranges::copy(buffers.PosedVertexNormals.GetSpan<vec3>({normals->VertexOffset, pr.VertexCount}), meshes.GetBaseVertexNormals(id).begin());
            std::ranges::copy(buffers.PosedSeamNormals.GetSpan<vec3>({normals->SeamOffset, normals->SeamCount}), meshes.GetBaseSeamNormals(id).begin());
            std::ranges::copy(buffers.PosedFaceNormals.GetSpan<vec3>({normals->FaceOffset, normals->FaceCount}), meshes.GetBaseFaceNormals(id).begin());
        }
    }
    return any_moved;
}

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
