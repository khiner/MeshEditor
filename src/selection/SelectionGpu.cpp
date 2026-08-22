#include "selection/SelectionGpu.h"

#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "gpu/BoxSelectPushConstants.h"
#include "gpu/ElementPickPushConstants.h"
#include "gpu/MeshletInstanceFlag.h"
#include "gpu/ObjectPickPushConstants.h"
#include "gpu/SelectionDrawPushConstants.h"
#include "gpu/SelectionElementPushConstants.h"
#include "gpu/UpdateSelectionStatePushConstants.h"
#include "gpu/VisibilitySelectionPushConstants.h"
#include "mesh/MeshStore.h"
#include "metal/PassChain.h"
#include "metal/RenderTarget.h"
#include "object/ObjectComponents.h"
#include "render/Drawing.h"
#include "render/Encoding.h"
#include "render/Instance.h"
#include "render/Pipelines.h"
#include "render/Profile.h"
#include "scene/Entity.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionQueries.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewportRenderGpu.h"

#include <entt/entity/registry.hpp>

namespace {
MTL::PrimitiveType ElementPrimitive(Element element) {
    if (element == Element::Face) return MTL::PrimitiveTypeTriangle;
    if (element == Element::Edge) return MTL::PrimitiveTypeLine;
    return MTL::PrimitiveTypePoint;
}

void ResetObjectPickKeys(GpuBuffers &buffers) {
    std::fill_n(buffers.ObjectPickKeys.Data(), GpuBuffers::MaxSelectableObjects, std::numeric_limits<uint32_t>::max());
}

// Reset linked-list counters and fill each head with the all-bits-set sentinel.
void ResetSelectionFragmentState(mtl::PassChain &chain, const GpuBuffers &buffers, const Pipelines &pipelines) {
    const auto &heads = *pipelines.SelectionFragment.Resources;
    const auto head_bytes = uint64_t(heads.Extent.Width) * heads.Extent.Height * sizeof(uint32_t);
    auto *blit = chain.BeginBlit("SelectionReset", MTL::StageFragment);
    blit->fillBuffer(*buffers.SelectionCounter, NS::Range::Make(0, sizeof(SelectionCounters)), 0);
    blit->fillBuffer(*heads.HeadBuffer, NS::Range::Make(0, head_bytes), 0xFF);
}

void RecordSelectionCompute(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const GpuBuffers &buffers,
    const mtl::ComputePipeline &compute, const auto &pc, auto &&dispatch
) {
    auto *encoder = chain.BeginCompute("SelectionPick", MTL::StageFragment);
    encode::BindCompute(encoder, compute, slots, buffers);
    encode::SetPushConstants(encoder, pc);
    dispatch(encoder);
}

// Selection queries complete synchronously for immediate readback.
void SubmitAndWait(MTL::CommandBuffer *command_buffer) {
    const profile::CpuScope scope{"SelectionSubmit"};
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
}

// One thread group per candidate band, and a face picks from a single group.
uint32_t ElementPickGroupCount(Element element) { return element == Element::Face ? 1u : GpuBuffers::ElementPickGroupCount; }

void RecordElementPick(
    mtl::PassChain &chain, const mtl::BindlessSet &slots, const GpuBuffers &buffers,
    const mtl::ComputePipeline &compute,
    uint32_t head_slot, uvec2 head_extent, uint32_t selection_nodes_slot, uint32_t element_candidate_buffer_slot,
    uvec2 mouse_px, Element element
) {
    const uint32_t radius = element == Element::Face ? 0u : ElementSelectRadiusPx;
    const uint32_t group_count = ElementPickGroupCount(element);
    RecordSelectionCompute(
        chain, slots, buffers, compute,
        ElementPickPushConstants{
            .TargetPx = mouse_px,
            .Radius = radius,
            .HeadSlot = head_slot,
            .HeadExtent = head_extent,
            .SelectionNodesIndex = selection_nodes_slot,
            .ElementCandidateBufferIndex = element_candidate_buffer_slot,
        },
        [group_count](MTL::ComputeCommandEncoder *encoder) {
            encoder->setThreadgroupMemoryLength(ThreadgroupMemory::ElementPickCandidates, 0);
            encoder->dispatchThreadgroups(MTL::Size(group_count, 1, 1), ThreadgroupSize::Linear256);
        }
    );
}

std::optional<uint32_t> ReadNearestPickedElement(const GpuBuffers &buffers, uint32_t max_element_id, Element element) {
    const std::span candidates{buffers.ElementPickCandidates.Data(), ElementPickGroupCount(element)};
    auto valid = candidates | std::views::filter([](const auto &c) { return c.Id != 0; });
    const auto it = std::ranges::min_element(valid, [](const auto &a, const auto &b) {
        return std::tie(a.DistanceSq, a.Depth) < std::tie(b.DistanceSq, b.Depth);
    });
    if (it == valid.end() || it->Id > max_element_id) return {};
    return it->Id - 1;
}

uint32_t MaxElementBound(auto &&ranges) {
    return std::ranges::fold_left(ranges, uint32_t{0}, [](uint32_t total, const auto &r) { return std::max(total, r.Offset + r.Count); });
}

VisibilityShadingPushConstants SelectionVisibilityPc(const GpuBuffers &buffers) {
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

void BindVisibilitySelectionTextures(MTL::RenderCommandEncoder *encoder, const Pipelines &pipelines) {
    encoder->setFragmentTexture(*pipelines.Main.Resources->VisibilityImage, 0u);
    encoder->setFragmentTexture(*pipelines.Main.Resources->DepthImage, 1u);
}

// Cull, optionally reset linked-list state, then draw selection fragments into silhouette depth.
void RunSelectionPass(
    entt::registry &r, mtl::PassChain &chain, DrawListBuilder &draw_list,
    bool render_depth, bool render_silhouette, bool draw_meshlets, uint32_t meshlet_flags, bool reset_state, auto &&record_draws
) {
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    FlushDrawList(r, draw_list, buffers.SelectionDraw);
    buffers.SetSceneViewDrawSlots(buffers.SelectionDraw);

    if (!draw_list.Draws.empty()) RecordFrustumCull(chain, slots, pipelines, buffers, buffers.SelectionDraw, draw_list);
    if (reset_state) ResetSelectionFragmentState(chain, buffers, pipelines);
    if (render_depth) RecordSilhouetteDepthPass(chain, slots, pipelines, buffers, render_silhouette);
    if (draw_meshlets && buffers.MeshletInstanceCount > 0) {
        RecordMeshletCull(chain, slots, pipelines, buffers, {.RequiredInstanceFlags = meshlet_flags});
    }

    const auto extent = pipelines.Silhouette.Resources->DepthImage.Extent;
    const auto pass = mtl::MakePassDescriptor({}, mtl::LoadDepth(*pipelines.Silhouette.Resources->DepthImage, MTL::StoreActionDontCare));
    pass->setRenderTargetWidth(extent.Width);
    pass->setRenderTargetHeight(extent.Height);
    auto *encoder = encode::BeginScenePass(chain, pass, "SelectionDraws", {{MTL::StageDispatch, MTL::StageVertex | MTL::StageMesh}, {MTL::StageBlit, MTL::StageFragment}}, extent, slots, buffers);
    record_draws(encoder, extent);
}

void RenderElementSelectionPass(
    entt::registry &r, mtl::PassChain &chain, entt::entity viewport,
    std::span<const ElementRange> ranges, Element element, bool write_bitset,
    uvec2 box_min, uvec2 box_max
) {
    if (ranges.empty() || element == Element::None) return;
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &meshes = r.ctx().get<MeshStore>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    const auto primary_edit_instances = selection::ComputePrimaryEditInstances(r);
    const bool xray_selection = r.get<const SelectionXRay>(viewport).Value;
    const auto element_pipeline = [xray_selection, write_bitset](Element el) -> SPT {
        if (el == Element::Vertex) {
            if (xray_selection) return write_bitset ? SPT::SelectionElementVertexXRayBitsetBox : SPT::SelectionElementVertexXRay;
            return write_bitset ? SPT::SelectionElementVertexBitsetBox : SPT::SelectionElementVertex;
        }
        if (el == Element::Edge) {
            if (xray_selection) return write_bitset ? SPT::SelectionElementEdgeXRayBitsetBox : SPT::SelectionElementEdgeXRay;
            return write_bitset ? SPT::SelectionElementEdgeBitsetBox : SPT::SelectionElementEdge;
        }
        if (xray_selection) return write_bitset ? SPT::SelectionElementFaceXRayBitsetBox : SPT::SelectionElementFaceXRay;
        return write_bitset ? SPT::SelectionElementFaceBitsetBox : SPT::SelectionElementFace;
    };

    DrawListBuilder draw_list;
    const bool render_depth = !xray_selection;
    auto element_batch = draw_list.BeginBatch(true);
    auto records = buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>({0, buffers.Instances.RecordBuffer.Count<InstanceRecord>()});
    for (auto &record : records) record.Flags &= ~uint32_t(MeshletInstanceFlag::ElementSelection);
    const bool face_point_fallback = element == Element::Face && write_bitset && xray_selection;
    for (const auto &range : ranges) {
        const auto &mesh_buffers = r.get<MeshBuffers>(range.MeshEntity);
        const auto &models = r.get<ModelsBuffer>(range.MeshEntity);
        const auto primary = primary_edit_instances.find(range.MeshEntity);
        const auto mark_instance = [&](uint32_t slot) {
            if (slot >= records.size()) return;
            records[slot].Flags |= uint32_t(MeshletInstanceFlag::ElementSelection);
            records[slot].ElementIdOffset = range.Offset;
        };
        if (primary != primary_edit_instances.end()) {
            mark_instance(r.get<RenderInstance>(primary->second).BufferIndex);
        } else {
            for (uint32_t i = 0; i < models.InstanceCount; ++i) mark_instance(models.InstanceRange.Offset + i);
        }
        if (element == Element::Face && !face_point_fallback) continue;

        const auto &mesh = GetMesh(r, range.MeshEntity);
        const auto &indices = element == Element::Vertex ? mesh_buffers.VertexIndices :
            element == Element::Edge                     ? mesh_buffers.EdgeIndices :
                                                           mesh_buffers.FaceIndices;
        auto draw = MakeDrawData(mesh_buffers.Vertices, indices, buffers.Instances);
        if (element == Element::Face) {
            const auto face_id_buffer = meshes.GetFaceIdRange(mesh.GetStoreId());
            draw.ObjectIdSlot = face_id_buffer.Slot;
            draw.FaceIdOffset = face_id_buffer.Offset;
        } else {
            draw.ObjectIdSlot = InvalidSlot;
            draw.FaceIdOffset = 0;
        }
        draw.VertexCountOrHeadImageSlot = 0;
        draw.ElementIdOffset = range.Offset;
        if (primary != primary_edit_instances.end()) {
            AppendDraw(draw_list, element_batch, indices, models, draw, r.get<RenderInstance>(primary->second).BufferIndex);
        } else {
            AppendDraw(draw_list, element_batch, indices, models, draw);
        }
    }

    const auto &selection = pipelines.SelectionFragment;
    // Bitset writes append to prior linked-list state, so only the fresh-list path resets it.
    RunSelectionPass(
        r, chain, draw_list, render_depth, element != Element::Face, element == Element::Face && xray_selection,
        uint32_t(MeshletInstanceFlag::ElementSelection), !write_bitset,
        [&](auto *encoder, mtl::Extent2D extent) {
            const SelectionElementPushConstants element_pc{
                element_batch.DrawDataSlotOffset,
                {selection.Resources->HeadBuffer.Slot, {extent.Width, extent.Height}, buffers.SelectionNodeBuffer.Slot, sel_slots.SelectionCounter, buffers.SelectionNodeCapacity},
                {box_min.x, box_min.y, box_max.x, box_max.y},
                sel_slots.SelectionBitset,
            };
            auto draw_with = [&](SPT spt, MTL::PrimitiveType primitive) {
                selection.Renderer.Bind(encoder, spt);
                encode::SetPushConstants(encoder, element_pc);
                encode::DrawIndexedIndirect(encoder, primitive, *buffers.IdentityIndexBuffer, *buffers.SelectionDraw.Indirect, element_batch.IndirectOffset, element_batch.DrawCount);
            };
            if (element == Element::Face) {
                if (xray_selection) {
                    const auto &pipeline = write_bitset ? selection.MeshletFaceXRayBitsetBox : selection.MeshletFaceXRay;
                    pipeline.Bind(encoder);
                    encode::SetPushConstants(encoder, element_pc);
                    DrawMeshlets(encoder, buffers, 0u, uint32_t(MeshletInstanceFlag::ElementSelection));
                } else {
                    const auto &pipeline = write_bitset ? selection.VisibilityFaceBitsetBox : selection.VisibilityFace;
                    pipeline.Bind(encoder);
                    BindVisibilitySelectionTextures(encoder, pipelines);
                    encode::SetPushConstants(encoder, VisibilitySelectionPushConstants{SelectionVisibilityPc(buffers), element_pc.Selection, element_pc.Box, element_pc.BoxResultSlot});
                    encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
                }
            } else if (element_batch.DrawCount > 0) {
                draw_with(element_pipeline(element), ElementPrimitive(element));
            }
            if (write_bitset && xray_selection) {
                // X-Ray face: the point pass catches edge-on faces, whose projected triangle has zero area.
                if (element == Element::Face) draw_with(SPT::SelectionElementFaceXRayVertsBitsetBox, MTL::PrimitiveTypePoint);
                // X-Ray edge: the point pass catches near-zero-length projected edges.
                if (element == Element::Edge) draw_with(SPT::SelectionElementEdgeXRayVertsBitsetBox, MTL::PrimitiveTypePoint);
            }
        }
    );

    // Element selection pass overwrites the shared head image used for object selection.
    r.ctx().get<DrawState>().SelectionStale = true;
}

} // namespace

std::optional<std::pair<entt::entity, uint32_t>> RunElementPickFromRanges(
    entt::registry &r, entt::entity viewport,
    std::span<const ElementRange> ranges, Element element, uvec2 mouse_px
) {
    if (ranges.empty() || element == Element::None) return {};
    const auto element_count = MaxElementBound(ranges);
    if (element_count == 0) return {};

    const profile::CpuScope scope{"RunElementPick"};
    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    const auto &heads = *pipelines.SelectionFragment.Resources;
    auto *command_buffer = ctx.Queue->commandBuffer();
    { // The chain closes its last pass as it goes out of scope, which the submit below needs.
        mtl::PassChain chain{command_buffer};
        RenderElementSelectionPass(r, chain, viewport, ranges, element, false, {}, {});
        RecordElementPick(
            chain, slots, buffers, pipelines.ElementPick,
            heads.HeadBuffer.Slot, {heads.Extent.Width, heads.Extent.Height}, buffers.SelectionNodeBuffer.Slot, sel_slots.ElementPickCandidates,
            mouse_px, element
        );
    }
    SubmitAndWait(command_buffer);
    if (const auto index = ReadNearestPickedElement(buffers, element_count, element)) {
        for (const auto &range : ranges) {
            if (*index < range.Offset || *index >= range.Offset + range.Count) continue;
            return std::pair{range.MeshEntity, *index - range.Offset};
        }
    }
    return {};
}

void RenderSelectionPassWith(
    entt::registry &r, mtl::PassChain &chain, [[maybe_unused]] entt::entity viewport, bool render_depth,
    const SelectionBuildFn &build_fn, bool render_silhouette = true, bool meshlet_objects = false
) {
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    DrawListBuilder draw_list;
    const auto selection_draws = build_fn(draw_list);

    const auto &selection = pipelines.SelectionFragment;
    RunSelectionPass(r, chain, draw_list, render_depth, render_silhouette, meshlet_objects, 0, true, [&](auto *encoder, mtl::Extent2D extent) {
        const SelectionDrawPushConstants sel_pc{
            0u,
            {selection.Resources->HeadBuffer.Slot, {extent.Width, extent.Height}, buffers.SelectionNodeBuffer.Slot, sel_slots.SelectionCounter, buffers.SelectionNodeCapacity},
        };
        for (const auto &selection_draw : selection_draws) {
            if (selection_draw.Batch.DrawCount == 0) continue;
            selection.Renderer.Bind(encoder, selection_draw.Pipeline);
            auto pc = sel_pc;
            pc.DrawDataOffset = selection_draw.Batch.DrawDataSlotOffset;
            encode::SetPushConstants(encoder, pc);
            encode::DrawIndexedIndirect(encoder, selection_draw.Primitive, *buffers.IdentityIndexBuffer, *buffers.SelectionDraw.Indirect, selection_draw.Batch.IndirectOffset, selection_draw.Batch.DrawCount);
        }
        if (meshlet_objects && buffers.MeshletInstanceCount > 0) {
            selection.VisibilityObject.Bind(encoder);
            BindVisibilitySelectionTextures(encoder, pipelines);
            encode::SetPushConstants(encoder, VisibilitySelectionPushConstants{SelectionVisibilityPc(buffers), sel_pc.Selection, {}, InvalidSlot});
            encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip, NS::UInteger(0), NS::UInteger(4));
        }
    });
}

void RenderSelectionPass(entt::registry &r, mtl::PassChain &chain, entt::entity viewport) {
    // Render depth so the selection-fragment pass has a valid depth attachment to load, even right after a resize recreated it.
    // Object-pick ignores depth, so its contents and the silhouette draws don't matter.
    RenderSelectionPassWith(
        r, chain, viewport, /*render_depth=*/true,
        [&r](DrawListBuilder &draw_list) -> std::vector<SelectionDrawInfo> {
            const auto &draw = r.ctx().get<const DrawState>();
            draw_list = draw.SelectionList;
            return draw.SelectionDraws;
        },
        /*render_silhouette=*/false,
        /*meshlet_objects=*/true
    );
}

void RunBoxSelectElements(entt::registry &r, entt::entity viewport, std::span<const ElementRange> ranges, Element element, std::pair<uvec2, uvec2> box_px, bool is_additive) {
    if (ranges.empty()) return;

    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return;

    auto &buffers = r.ctx().get<GpuBuffers>();
    const profile::CpuScope scope{"RunBoxSelectElements"};
    const auto element_count = MaxElementBound(ranges);
    if (element_count == 0) return;

    const uint32_t bitset_words = (element_count + 31) / 32;
    if (bitset_words > GpuBuffers::SelectionBitsetWords) return;

    // Restore baseline bitset for additive mode, or clear for non-additive.
    auto *bits = buffers.SelectionBitset.Data();
    if (is_additive) {
        const auto *baseline = r.try_get<const AdditiveBoxSelectBaseline>(viewport);
        if (baseline && !baseline->ElementBitset.empty()) {
            const auto copy_words = std::min(bitset_words, uint32_t(baseline->ElementBitset.size()));
            memcpy(bits, baseline->ElementBitset.data(), copy_words * sizeof(uint32_t));
            if (copy_words < bitset_words) { // Zero any remaining words beyond the baseline
                memset(&bits[copy_words], 0, (bitset_words - copy_words) * sizeof(uint32_t));
            }
        }
    } else {
        memset(bits, 0, bitset_words * sizeof(uint32_t));
    }

    // Box-select writes element IDs directly from the selection fragment shader.
    auto *command_buffer = r.ctx().get<const mtl::Context>().Queue->commandBuffer();
    {
        mtl::PassChain chain{command_buffer};
        RenderElementSelectionPass(r, chain, viewport, ranges, element, true, box_min, box_max);
    }
    SubmitAndWait(command_buffer);
    ApplySelectionStateUpdate(r, viewport, ranges, element);
}

std::optional<uint32_t> RunSoundVerticesVertexPick(entt::registry &r, entt::entity viewport, entt::entity instance_entity, uvec2 mouse_px) {
    if (!r.all_of<SoundVertices>(instance_entity)) return {};
    const auto *instance = r.try_get<Instance>(instance_entity);
    if (!instance) return {};
    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &meshes = r.ctx().get<MeshStore>();
    const auto &pipelines = r.ctx().get<const Pipelines>();

    const profile::CpuScope scope{"RunSoundVerticesVertexPick"};
    const auto mesh_entity = instance->Entity;
    const auto &mesh = GetMesh(r, mesh_entity);
    const uint32_t vertex_count = mesh.VertexCount();
    if (vertex_count == 0) return {};

    const auto &mesh_buffers = r.get<MeshBuffers>(mesh_entity);
    const auto &models = r.get<ModelsBuffer>(mesh_entity);
    const auto model_index = r.get<RenderInstance>(instance_entity).BufferIndex;
    auto *command_buffer = ctx.Queue->commandBuffer();
    {
        mtl::PassChain chain{command_buffer};
        RenderSelectionPassWith(r, chain, viewport, true, [&](DrawListBuilder &draw_list) {
            auto batch = draw_list.BeginBatch(true);
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.VertexIndices, buffers.Instances);
            draw.VertexCountOrHeadImageSlot = 0;
            draw.ElementStateSlotOffset = {meshes.GetVertexStateSlot(), mesh_buffers.Vertices.Offset};
            AppendDraw(draw_list, batch, mesh_buffers.VertexIndices, models, draw, model_index);
            return std::vector{SelectionDrawInfo{SPT::SelectionElementVertex, MTL::PrimitiveTypePoint, batch}};
        });
        const auto &heads = *pipelines.SelectionFragment.Resources;
        RecordElementPick(
            chain, slots, buffers, pipelines.ElementPick,
            heads.HeadBuffer.Slot, {heads.Extent.Width, heads.Extent.Height}, buffers.SelectionNodeBuffer.Slot, sel_slots.ElementPickCandidates,
            mouse_px, Element::Vertex
        );
    }
    SubmitAndWait(command_buffer);
    r.ctx().get<DrawState>().SelectionStale = true;
    return ReadNearestPickedElement(buffers, vertex_count, Element::Vertex);
}

std::vector<entt::entity> RunObjectPick(entt::registry &r, entt::entity viewport, uint32_t &object_pick_epoch_tag, uvec2 mouse_px, uint32_t radius_px) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const uint32_t next_object_id = r.ctx().get<const ObjectIdCounter>().Next;
    if (next_object_id <= 1) return {}; // No objects have been assigned IDs yet
    const uint32_t max_object_id = std::min(next_object_id - 1, GpuBuffers::MaxSelectableObjects);
    if (max_object_id == 0) return {};

    const profile::CpuScope scope{"RunObjectPick"};
    const bool selection_stale = r.ctx().get<const DrawState>().SelectionStale;
    // ObjectPickKeyBuffer is persistent across clicks: high 8 bits of each packed key store
    // a per-click epoch tag. We therefore avoid clearing all keys every click and only do a
    // full reset when the 8-bit epoch wraps; stale keys are filtered out by epoch on readback.
    if (object_pick_epoch_tag == 0) {
        ResetObjectPickKeys(buffers);
        object_pick_epoch_tag = 255;
    }
    const uint32_t epoch_inv = object_pick_epoch_tag--;

    const auto &heads = *pipelines.SelectionFragment.Resources;
    auto *command_buffer = ctx.Queue->commandBuffer();
    { // The chain closes its last pass as it goes out of scope, which the submit below needs.
        mtl::PassChain chain{command_buffer};
        if (selection_stale) RenderSelectionPass(r, chain, viewport);
        RecordSelectionCompute(
            chain, slots, buffers, pipelines.ObjectPick,
            ObjectPickPushConstants{
                .TargetPx = mouse_px,
                .Radius = radius_px,
                .MaxId = max_object_id,
                .EpochInv = epoch_inv,
                .HeadSlot = heads.HeadBuffer.Slot,
                .HeadExtent = {heads.Extent.Width, heads.Extent.Height},
                .SelectionNodesIndex = buffers.SelectionNodeBuffer.Slot,
                .BestKeyIndex = sel_slots.ObjectPickKey,
                .SeenBitsIndex = sel_slots.ObjectPickSeenBits,
            },
            [](MTL::ComputeCommandEncoder *encoder) { encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), ThreadgroupSize::Linear256); }
        );
    }
    SubmitAndWait(command_buffer);
    if (selection_stale) r.ctx().get<DrawState>().SelectionStale = false;

    std::unordered_map<uint32_t, entt::entity> object_id_to_entity;
    for (const auto [e, ri] : r.view<RenderInstance>().each()) {
        if (ri.ObjectId > 0 && ri.ObjectId <= max_object_id) object_id_to_entity[ri.ObjectId] = e;
    }

    struct SortedHit {
        uint32_t DistSq;
        uint32_t Layer; // 0 = bone (on top in main pass), 1 = other
        uint32_t Depth;
        entt::entity Entity;
        auto operator<=>(const SortedHit &) const = default;
    };

    const auto *bits = buffers.ObjectPickSeenBitset.Data();
    const auto *keys = buffers.ObjectPickKeys.Data();
    std::vector<SortedHit> hits;
    for (uint32_t object_id = 1; object_id <= max_object_id; ++object_id) {
        const uint32_t idx = object_id - 1;
        if ((bits[idx / 32] & (1u << (idx % 32))) == 0) continue;
        const auto it = object_id_to_entity.find(object_id);
        if (it == object_id_to_entity.end()) continue;
        const uint32_t packed_key = keys[idx];
        if ((packed_key >> 24) == epoch_inv) {
            const uint32_t layer = r.any_of<BoneIndex, BoneSubPartOf>(it->second) ? 0u : 1u;
            hits.emplace_back(SortedHit{(packed_key >> 16) & 0xffu, layer, packed_key & 0xffffu, it->second});
        }
    }
    std::ranges::sort(hits);

    std::vector<entt::entity> entities;
    entities.reserve(hits.size());
    for (const auto &hit : hits) entities.emplace_back(hit.Entity);
    return entities;
}

namespace {
void DispatchBoxSelect(entt::registry &r, mtl::PassChain &chain, uvec2 box_min, uvec2 box_max, uint32_t max_id) {
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const uint32_t bitset_words = (max_id + 31) / 32;
    memset(buffers.SelectionBitset.Data(), 0, bitset_words * sizeof(uint32_t));

    const auto group_counts = glm::max((box_max - box_min + 15u) / 16u, uvec2{1, 1});
    const auto &heads = *pipelines.SelectionFragment.Resources;
    RecordSelectionCompute(
        chain, slots, buffers, pipelines.BoxSelect,
        BoxSelectPushConstants{
            .BoxMin = box_min,
            .BoxMax = box_max,
            .MaxId = max_id,
            .HeadSlot = heads.HeadBuffer.Slot,
            .HeadExtent = {heads.Extent.Width, heads.Extent.Height},
            .SelectionNodesIndex = buffers.SelectionNodeBuffer.Slot,
            .BoxResultIndex = sel_slots.SelectionBitset,
        },
        [group_counts](MTL::ComputeCommandEncoder *encoder) {
            encoder->dispatchThreadgroups(MTL::Size(group_counts.x, group_counts.y, 1), ThreadgroupSize::Tile16);
        }
    );
}
} // namespace

std::vector<entt::entity> RunBoxSelect(entt::registry &r, entt::entity viewport, std::pair<uvec2, uvec2> box_px) {
    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return {};
    auto &buffers = r.ctx().get<GpuBuffers>();
    const uint32_t next_object_id = r.ctx().get<const ObjectIdCounter>().Next;
    if (next_object_id <= 1) return {}; // No objects have been assigned IDs yet

    const uint32_t max_object_id = std::min(next_object_id - 1, GpuBuffers::MaxSelectableObjects);

    const profile::CpuScope scope{"RunBoxSelect"};
    const bool selection_stale = r.ctx().get<const DrawState>().SelectionStale;
    auto *command_buffer = r.ctx().get<const mtl::Context>().Queue->commandBuffer();
    { // The chain closes its last pass as it goes out of scope, which the submit below needs.
        mtl::PassChain chain{command_buffer};
        if (selection_stale) RenderSelectionPass(r, chain, viewport);
        DispatchBoxSelect(r, chain, box_min, box_max, max_object_id);
    }
    SubmitAndWait(command_buffer);
    if (selection_stale) r.ctx().get<DrawState>().SelectionStale = false;

    std::unordered_map<uint32_t, entt::entity> object_id_to_entity;
    for (const auto [e, ri] : r.view<RenderInstance>().each()) object_id_to_entity[ri.ObjectId] = e;

    const auto *bits = buffers.SelectionBitset.Data();
    std::vector<entt::entity> entities;
    for (uint32_t object_id = 1; object_id <= max_object_id; ++object_id) {
        const uint32_t bit_index = object_id - 1;
        const uint32_t mask = 1u << (bit_index % 32);
        if ((bits[bit_index / 32] & mask) != 0) {
            if (auto it = object_id_to_entity.find(object_id); it != object_id_to_entity.end()) {
                entities.emplace_back(it->second);
            }
        }
    }
    return entities;
}

void DispatchUpdateSelectionStates(
    entt::registry &r,
    std::span<const ElementRange> ranges, Element element
) {
    if (ranges.empty() || element == Element::None) return;
    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &slots = r.ctx().get<const mtl::BindlessSet>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &sel_slots = r.ctx().get<const SelectionSlots>();
    auto &meshes = r.ctx().get<MeshStore>();

    const auto &buffers = r.ctx().get<const GpuBuffers>();
    auto *command_buffer = ctx.Queue->commandBuffer();
    auto *encoder = command_buffer->computeCommandEncoder();
    encode::BindCompute(encoder, pipelines.UpdateSelectionState, slots, buffers);

    for (const auto &range : ranges) {
        const auto &mesh = GetMesh(r, range.MeshEntity);
        const auto &mesh_buffers = r.get<const MeshBuffers>(range.MeshEntity);
        const auto *active_element = r.try_get<const MeshActiveElement>(range.MeshEntity);

        uint32_t state_slot, state_offset;
        if (element == Element::Vertex) {
            state_slot = meshes.GetVertexStateSlot();
            state_offset = mesh_buffers.Vertices.Offset;
        } else if (element == Element::Edge) {
            const auto edge_range = meshes.GetEdgeStateRange(mesh.GetStoreId());
            state_slot = edge_range.Slot;
            state_offset = edge_range.Offset;
        } else {
            const auto face_range = meshes.GetFaceStateRange(mesh.GetStoreId());
            state_slot = face_range.Slot;
            state_offset = face_range.Offset;
        }

        const UpdateSelectionStatePushConstants pc{
            .BitsetSlot = sel_slots.SelectionBitset,
            .BitsetOffset = range.Offset,
            .StateSlot = state_slot,
            .StateOffset = state_offset,
            .ElementCount = range.Count,
            .ActiveHandle = active_element ? active_element->Handle : InvalidOffset,
            .EdgeMode = element == Element::Edge ? 1u : 0u,
        };
        encode::SetPushConstants(encoder, pc);
        encoder->dispatchThreadgroups(MTL::Size((range.Count + 255) / 256, 1, 1), ThreadgroupSize::Linear256);
    }

    encoder->endEncoding();
    command_buffer->commit();
    // The host reads the element states straight after this returns.
    command_buffer->waitUntilCompleted();
}

void ApplySelectionStateUpdate(
    entt::registry &r, entt::entity viewport,
    std::span<const ElementRange> ranges, Element element
) {
    auto &meshes = r.ctx().get<MeshStore>();
    DispatchUpdateSelectionStates(r, ranges, element);
    if (element == Element::Vertex) {
        for (const auto &range : ranges) {
            const auto &mesh = GetMesh(r, range.MeshEntity);
            meshes.UpdateEdgeStatesFromVertices(mesh);
            meshes.UpdateFaceStatesFromVertices(mesh);
        }
    } else if (element == Element::Face || element == Element::Edge) {
        for (const auto &range : ranges) {
            const auto &mesh = GetMesh(r, range.MeshEntity);
            std::optional<uint32_t> active_handle;
            if (const auto *active = r.try_get<const MeshActiveElement>(range.MeshEntity); active && active->Handle < range.Count) {
                active_handle = active->Handle;
            }
            if (element == Element::Face) {
                meshes.UpdateEdgeStatesFromFaces(mesh, active_handle);
                meshes.UpdateVertexStatesFromFaces(mesh, active_handle);
            } else {
                meshes.UpdateFaceStatesFromEdges(mesh);
                meshes.UpdateVertexStatesFromEdges(mesh, active_handle);
            }
        }
    }
    r.emplace_or_replace<ElementStatesDirty>(viewport);
}
