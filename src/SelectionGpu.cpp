#include "SelectionGpu.h"

#include "Apply.h"
#include "Armature.h"
#include "Drawing.h"
#include "Entity.h"
#include "Instance.h"
#include "ObjectComponents.h"
#include "Pipelines.h"
#include "Selection.h"
#include "SelectionComponents.h"
#include "SoundVertices.h"
#include "Timer.h"
#include "ViewportRenderGpu.h"
#include "VkFenceWait.h"
#include "VulkanResources.h"
#include "gpu/BoxSelectPushConstants.h"
#include "gpu/ElementPickPushConstants.h"
#include "gpu/MainDrawPushConstants.h"
#include "gpu/ObjectPickPushConstants.h"
#include "gpu/SelectionDrawPushConstants.h"
#include "gpu/SelectionElementPushConstants.h"
#include "mesh/MeshStore.h"

#include "imgui.h"

namespace {
constexpr vk::Extent2D ToExtent2D(vk::Extent3D extent) { return {extent.width, extent.height}; }
const vk::ClearColorValue Transparent{0, 0, 0, 0};
const std::vector<vk::ClearValue> SilhouetteClearValues{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};

std::optional<uint32_t> GetModelBufferIndex(const entt::registry &r, entt::entity e) {
    if (const auto *ri = r.try_get<RenderInstance>(e)) return ri->BufferIndex;
    return {};
}

void ResetObjectPickKeys(GpuBuffers &buffers) {
    std::fill_n(buffers.ObjectPickKeys.Data(), GpuBuffers::MaxSelectableObjects, std::numeric_limits<uint32_t>::max());
}

void RunSelectionCompute(
    vk::CommandBuffer cb, vk::Queue queue, vk::Fence fence, vk::Device device,
    const auto &compute, const auto &pc, auto &&dispatch, vk::Semaphore wait_semaphore = {}
) {
    const Timer timer{"RunSelectionCompute"};
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, *compute.Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *compute.PipelineLayout, 0, compute.GetDescriptorSet(), {});
    cb.pushConstants(*compute.PipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
    dispatch(cb);

    const vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead};
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, barrier, {}, {});
    cb.end();

    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    static constexpr vk::PipelineStageFlags wait_stage{vk::PipelineStageFlagBits::eComputeShader};
    if (wait_semaphore) {
        submit.setWaitSemaphores(wait_semaphore);
        submit.setWaitDstStageMask(wait_stage);
    }
    queue.submit(submit, fence);
    WaitFor(fence, device);
}

std::optional<uint32_t> FindNearestPickedElement(
    const GpuBuffers &buffers, const ComputePipeline &compute, vk::CommandBuffer cb,
    vk::Queue queue, vk::Fence fence, vk::Device device,
    uint32_t head_image_index, uint32_t selection_nodes_slot, uint32_t element_candidate_buffer_slot,
    uvec2 mouse_px, uint32_t max_element_id, Element element,
    vk::Semaphore wait_semaphore
) {
    const uint32_t radius = element == Element::Face ? 0u : ElementSelectRadiusPx;
    const uint32_t group_count = element == Element::Face ? 1u : GpuBuffers::ElementPickGroupCount;
    RunSelectionCompute(
        cb, queue, fence, device, compute,
        ElementPickPushConstants{
            .TargetPx = mouse_px,
            .Radius = radius,
            .HeadImageIndex = head_image_index,
            .SelectionNodesIndex = selection_nodes_slot,
            .ElementCandidateBufferIndex = element_candidate_buffer_slot,
        },
        [group_count](vk::CommandBuffer dispatch_cb) { dispatch_cb.dispatch(group_count, 1, 1); },
        wait_semaphore
    );

    const std::span candidates{buffers.ElementPickCandidates.Data(), group_count};
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

void AppendSelectedSilhouetteDraws(const entt::registry &R, DrawListBuilder &draw_list, DrawBatchInfo &silhouette_batch) {
    const auto &buffers = R.ctx().get<const GpuBuffers>();
    for (const auto e : R.view<Selected>()) {
        const auto *inst = R.try_get<Instance>(e);
        if (!inst) continue;
        const auto buffer_entity = inst->Entity;
        const auto &mesh_buffers = R.get<MeshBuffers>(buffer_entity);
        const auto &models = R.get<ModelsBuffer>(buffer_entity);
        if (const auto model_index = GetModelBufferIndex(R, e)) {
            auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, buffers.Instances);
            draw.ObjectIdSlot = buffers.Instances.ObjectIdBuffer.Slot;
            AppendDraw(draw_list, silhouette_batch, mesh_buffers.FaceIndices, models, draw, *model_index);
        }
    }
}

void RenderElementSelectionPass(
    entt::registry &R, entt::entity viewport,
    std::span<const ElementRange> ranges, Element element, bool write_bitset,
    uvec2 box_min, uvec2 box_max, vk::Semaphore signal_semaphore
) {
    if (ranges.empty() || element == Element::None) return;
    const auto &vk_res = R.ctx().get<const VulkanResources>();
    const auto &pipelines = R.ctx().get<const Pipelines>();
    const auto &one_shot = R.ctx().get<const OneShotGpu>();
    const auto &sel_slots = R.get<const SelectionSlots>(viewport);
    auto &meshes = R.ctx().get<MeshStore>();
    auto &buffers = R.ctx().get<GpuBuffers>();

    const auto primary_edit_instances = selection::ComputePrimaryEditInstances(R);
    const bool xray_selection = R.get<const SelectionXRay>(viewport).Value;
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
    DrawBatchInfo silhouette_batch{};
    const bool render_depth = !xray_selection;
    if (render_depth) {
        silhouette_batch = draw_list.BeginBatch();
        if (element != Element::Face) AppendSelectedSilhouetteDraws(R, draw_list, silhouette_batch);
    }
    auto element_batch = draw_list.BeginBatch();
    for (const auto &r : ranges) {
        const auto &mesh_buffers = R.get<MeshBuffers>(r.MeshEntity);
        const auto &models = R.get<ModelsBuffer>(r.MeshEntity);
        const auto &mesh = R.get<Mesh>(r.MeshEntity);
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
        draw.ElementIdOffset = r.Offset;
        if (auto it = primary_edit_instances.find(r.MeshEntity); it != primary_edit_instances.end()) {
            AppendDraw(draw_list, element_batch, indices, models, draw, R.get<RenderInstance>(it->second).BufferIndex);
        } else {
            AppendDraw(draw_list, element_batch, indices, models, draw);
        }
    }

    FlushDrawList(R, vk_res.Device, draw_list, buffers.SelectionDraw);
    buffers.SceneViewUBO.Update(as_bytes(buffers.SelectionDraw.DrawData.Slot), offsetof(SceneViewUBO, DrawDataSlot));

    auto cb = *one_shot.Cb;
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    if (!write_bitset) {
        // Reset linked-list state before writing selection fragments.
        buffers.SelectionCounter.Buffer.Write(as_bytes(SelectionCounters{}));

        const auto &head_image = pipelines.SelectionFragment.Resources->HeadImage;
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
            vk::ImageMemoryBarrier{{}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {}, {}, *head_image.Image, ColorSubresourceRange}
        );
        cb.clearColorImage(*head_image.Image, vk::ImageLayout::eGeneral, vk::ClearColorValue{std::array<uint32_t, 4>{InvalidSlot, 0, 0, 0}}, ColorSubresourceRange);
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
            vk::ImageMemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, {}, {}, *head_image.Image, ColorSubresourceRange}
        );
    }

    const auto render_extent = ComputeRenderExtentPx(R.get<const ViewportExtent>(viewport).Value, std::bit_cast<vec2>(ImGui::GetIO().DisplayFramebufferScale));
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(render_extent.width), float(render_extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, render_extent});

    if (render_depth) {
        const auto &silhouette = pipelines.Silhouette;
        const vk::Rect2D sil_rect{{0, 0}, ToExtent2D(silhouette.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette.Renderer.RenderPass, *silhouette.Resources->Framebuffer, sil_rect, SilhouetteClearValues}, vk::SubpassContents::eInline);
        cb.bindIndexBuffer(*buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        if (silhouette_batch.DrawCount > 0) {
            const auto &pipeline = silhouette.Renderer.Bind(cb, SPT::SilhouetteDepthObject);
            const MainDrawPushConstants sil_pc{{silhouette_batch.DrawDataSlotOffset, InvalidSlot}};
            cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(sil_pc), &sil_pc);
            cb.drawIndexedIndirect(*buffers.SelectionDraw.Indirect, silhouette_batch.IndirectOffset, silhouette_batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
        }
        cb.endRenderPass();
    }

    const auto &selection = pipelines.SelectionFragment;
    const vk::Rect2D sel_rect{{0, 0}, ToExtent2D(pipelines.Silhouette.Resources->DepthImage.Extent)};
    cb.beginRenderPass({*selection.Renderer.RenderPass, *selection.Resources->Framebuffer, sel_rect, {}}, vk::SubpassContents::eInline);
    cb.bindIndexBuffer(*buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
    if (element_batch.DrawCount > 0) {
        const SelectionElementPushConstants element_pc{
            {element_batch.DrawDataSlotOffset, InvalidSlot},
            {sel_slots.HeadImage, buffers.SelectionNodeBuffer.Slot, sel_slots.SelectionCounter, buffers.SelectionNodeCapacity},
            {box_min.x, box_min.y, box_max.x, box_max.y},
            sel_slots.SelectionBitset,
        };
        auto draw_with = [&](SPT spt) {
            const auto &pipeline = selection.Renderer.Bind(cb, spt);
            cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(element_pc), &element_pc);
            cb.drawIndexedIndirect(*buffers.SelectionDraw.Indirect, element_batch.IndirectOffset, element_batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
        };
        draw_with(element_pipeline(element));
        if (write_bitset && xray_selection) {
            // X-Ray face: point pass catches edge-on faces (zero projected triangle area).
            if (element == Element::Face) draw_with(SPT::SelectionElementFaceXRayVertsBitsetBox);
            // X-Ray edge: point pass catches near/zero-length projected edges.
            if (element == Element::Edge) draw_with(SPT::SelectionElementEdgeXRayVertsBitsetBox);
        }
    }
    cb.endRenderPass();

    if (write_bitset) {
        // Ensure fragment shader writes to the bitset are visible to the host after the fence.
        // Scope the barrier to the written range.
        const auto element_count = MaxElementBound(ranges);
        const vk::DeviceSize bitset_bytes = ((element_count + 31) / 32) * sizeof(uint32_t);
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eHost, {}, {},
            vk::BufferMemoryBarrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead, {}, {}, *buffers.SelectionBitset, 0, bitset_bytes},
            {}
        );
    }

    cb.end();
    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    if (signal_semaphore) submit.setSignalSemaphores(signal_semaphore);
    vk_res.Queue.submit(submit, *one_shot.Fence);
    WaitFor(*one_shot.Fence, vk_res.Device);

    // Element selection pass overwrites the shared head image used for object selection.
    R.emplace_or_replace<SelectionStale>(viewport);
}

} // namespace

std::optional<std::pair<entt::entity, uint32_t>> RunElementPickFromRanges(
    entt::registry &R, entt::entity viewport,
    std::span<const ElementRange> ranges, Element element, uvec2 mouse_px
) {
    if (ranges.empty() || element == Element::None) return {};
    const auto element_count = MaxElementBound(ranges);
    if (element_count == 0) return {};

    const Timer timer{"RunElementPick"};
    const auto &vk_res = R.ctx().get<const VulkanResources>();
    const auto &pipelines = R.ctx().get<const Pipelines>();
    const auto &one_shot = R.ctx().get<const OneShotGpu>();
    const auto &sel_slots = R.get<const SelectionSlots>(viewport);
    auto &buffers = R.ctx().get<GpuBuffers>();
    RenderElementSelectionPass(R, viewport, ranges, element, false, {}, {}, *one_shot.SelectionReady);
    if (const auto index = FindNearestPickedElement(
            buffers, pipelines.ElementPick, *one_shot.Cb,
            vk_res.Queue, *one_shot.Fence, vk_res.Device,
            sel_slots.HeadImage, buffers.SelectionNodeBuffer.Slot, sel_slots.ElementPickCandidates,
            mouse_px, element_count, element,
            *one_shot.SelectionReady
        )) {
        for (const auto &range : ranges) {
            if (*index < range.Offset || *index >= range.Offset + range.Count) continue;
            return std::pair{range.MeshEntity, *index - range.Offset};
        }
    }
    return {};
}

void RenderSelectionPassWith(entt::registry &R, entt::entity viewport, bool render_depth, const SelectionBuildFn &build_fn, vk::Semaphore signal_semaphore, bool render_silhouette) {
    const Timer timer{"RenderSelectionPassWith"};
    const auto &Vk = R.ctx().get<const VulkanResources>();
    const auto &one_shot = R.ctx().get<const OneShotGpu>();
    const auto &sel_slots = R.get<const SelectionSlots>(viewport);
    auto &Buffers = R.ctx().get<GpuBuffers>();
    auto &pipelines = R.ctx().get<const Pipelines>();
    DrawListBuilder draw_list;
    DrawBatchInfo silhouette_batch{};
    if (render_depth) {
        silhouette_batch = draw_list.BeginBatch();
        if (render_silhouette) AppendSelectedSilhouetteDraws(R, draw_list, silhouette_batch);
    }
    const auto selection_draws = build_fn(draw_list);

    FlushDrawList(R, Vk.Device, draw_list, Buffers.SelectionDraw);
    Buffers.SceneViewUBO.Update(as_bytes(Buffers.SelectionDraw.DrawData.Slot), offsetof(SceneViewUBO, DrawDataSlot));

    auto cb = *one_shot.Cb;
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    Buffers.SelectionCounter.Buffer.Write(as_bytes(SelectionCounters{}));

    // Transition head image to general layout and clear.
    const auto &head_image = pipelines.SelectionFragment.Resources->HeadImage;
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
        vk::ImageMemoryBarrier{{}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {}, {}, *head_image.Image, ColorSubresourceRange}
    );
    cb.clearColorImage(*head_image.Image, vk::ImageLayout::eGeneral, vk::ClearColorValue{std::array<uint32_t, 4>{InvalidSlot, 0, 0, 0}}, ColorSubresourceRange);
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
        vk::ImageMemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, {}, {}, *head_image.Image, ColorSubresourceRange}
    );

    const auto render_extent = ComputeRenderExtentPx(R.get<const ViewportExtent>(viewport).Value, std::bit_cast<vec2>(ImGui::GetIO().DisplayFramebufferScale));
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(render_extent.width), float(render_extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, render_extent});

    if (render_depth) {
        // Render selected meshes to silhouette depth buffer for element occlusion.
        // Open the pass even with no draws — its depth LoadOp::eClear seeds the selection-fragment pass's LoadOp::eLoad.
        const auto &silhouette = pipelines.Silhouette;
        const vk::Rect2D rect{{0, 0}, ToExtent2D(silhouette.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette.Renderer.RenderPass, *silhouette.Resources->Framebuffer, rect, SilhouetteClearValues}, vk::SubpassContents::eInline);
        if (silhouette_batch.DrawCount > 0) {
            cb.bindIndexBuffer(*Buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
            const auto &pipeline = silhouette.Renderer.Bind(cb, SPT::SilhouetteDepthObject);
            const MainDrawPushConstants sil_pc{{silhouette_batch.DrawDataSlotOffset, InvalidSlot}};
            cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(sil_pc), &sil_pc);
            cb.drawIndexedIndirect(*Buffers.SelectionDraw.Indirect, silhouette_batch.IndirectOffset, silhouette_batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
        }
        cb.endRenderPass();
    }

    const auto &selection = pipelines.SelectionFragment;
    const vk::Rect2D rect{{0, 0}, ToExtent2D(pipelines.Silhouette.Resources->DepthImage.Extent)};
    cb.beginRenderPass({*selection.Renderer.RenderPass, *selection.Resources->Framebuffer, rect, {}}, vk::SubpassContents::eInline);
    cb.bindIndexBuffer(*Buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
    const SelectionDrawPushConstants sel_pc{
        {0u, InvalidSlot},
        {sel_slots.HeadImage, Buffers.SelectionNodeBuffer.Slot, sel_slots.SelectionCounter, Buffers.SelectionNodeCapacity},
    };
    for (const auto &selection_draw : selection_draws) {
        if (selection_draw.Batch.DrawCount == 0) continue;
        const auto &pipeline = selection.Renderer.Bind(cb, selection_draw.Pipeline);
        auto pc = sel_pc;
        pc.VertexTransform.DrawDataOffset = selection_draw.Batch.DrawDataSlotOffset;
        cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        cb.drawIndexedIndirect(*Buffers.SelectionDraw.Indirect, selection_draw.Batch.IndirectOffset, selection_draw.Batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
    }
    cb.endRenderPass();

    cb.end();
    vk::SubmitInfo submit;
    submit.setCommandBuffers(cb);
    if (signal_semaphore) submit.setSignalSemaphores(signal_semaphore);
    Vk.Queue.submit(submit, *one_shot.Fence);
    WaitFor(*one_shot.Fence, Vk.Device);
}

void RenderSelectionPass(entt::registry &R, entt::entity viewport, vk::Semaphore signal_semaphore) {
    const Timer timer{"RenderSelectionPass"};

    // Selection draw list is pre-built by RecordRenderCommandBuffer.
    RenderSelectionPassWith(
        R, viewport, false,
        [&R, viewport](DrawListBuilder &draw_list) -> std::vector<SelectionDrawInfo> {
            const auto &draw = R.get<const DrawState>(viewport);
            draw_list = draw.SelectionList;
            return draw.SelectionDraws;
        },
        signal_semaphore
    );

    R.remove<SelectionStale>(viewport);
}

void RunBoxSelectElements(entt::registry &R, entt::entity viewport, std::span<const ElementRange> ranges, Element element, std::pair<uvec2, uvec2> box_px, bool is_additive) {
    if (ranges.empty()) return;

    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return;

    auto &Buffers = R.ctx().get<GpuBuffers>();
    const Timer timer{"RunBoxSelectElements"};
    const auto element_count = MaxElementBound(ranges);
    if (element_count == 0) return;

    const uint32_t bitset_words = (element_count + 31) / 32;
    if (bitset_words > GpuBuffers::SelectionBitsetWords) return;

    // Restore baseline bitset for additive mode, or clear for non-additive.
    auto *bits = Buffers.SelectionBitset.Data();
    if (is_additive) {
        const auto *baseline = R.try_get<const AdditiveBoxSelectBaseline>(viewport);
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
    RenderElementSelectionPass(R, viewport, ranges, element, true, box_min, box_max, {});
    // After RenderElementSelectionPass (which waits on fence), SelectionBitsetBuffer is populated.
    // Dispatch UpdateSelectionState compute shader to update element state buffers on GPU.
    ApplySelectionStateUpdate(R, viewport, ranges, element);
}

std::optional<uint32_t> RunSoundVerticesVertexPick(entt::registry &R, entt::entity viewport, entt::entity instance_entity, uvec2 mouse_px) {
    if (!R.all_of<SoundVertices>(instance_entity)) return {};
    const auto *instance = R.try_get<Instance>(instance_entity);
    if (!instance) return {};
    const auto &Vk = R.ctx().get<const VulkanResources>();
    const auto &one_shot = R.ctx().get<const OneShotGpu>();
    const auto &sel_slots = R.get<const SelectionSlots>(viewport);
    auto &Buffers = R.ctx().get<GpuBuffers>();
    auto &Meshes = R.ctx().get<MeshStore>();
    auto &pipelines = R.ctx().get<const Pipelines>();

    const Timer timer{"RunSoundVerticesVertexPick"};
    const auto mesh_entity = instance->Entity;
    const auto &mesh = R.get<Mesh>(mesh_entity);
    const uint32_t vertex_count = mesh.VertexCount();
    if (vertex_count == 0) return {};

    const auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
    const auto &models = R.get<ModelsBuffer>(mesh_entity);
    const auto model_index = R.get<RenderInstance>(instance_entity).BufferIndex;
    RenderSelectionPassWith(R, viewport, true, [&](DrawListBuilder &draw_list) {
        auto batch = draw_list.BeginBatch();
        auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.VertexIndices, Buffers.Instances);
        draw.VertexCountOrHeadImageSlot = 0;
        draw.ElementStateSlotOffset = {Meshes.GetVertexStateSlot(), mesh_buffers.Vertices.Offset};
        AppendDraw(draw_list, batch, mesh_buffers.VertexIndices, models, draw, model_index);
        return std::vector{SelectionDrawInfo{SPT::SelectionElementVertex, batch}}; }, *one_shot.SelectionReady);
    R.emplace_or_replace<SelectionStale>(viewport);

    return FindNearestPickedElement(
        Buffers, pipelines.ElementPick, *one_shot.Cb,
        Vk.Queue, *one_shot.Fence, Vk.Device,
        sel_slots.HeadImage, Buffers.SelectionNodeBuffer.Slot, sel_slots.ElementPickCandidates,
        mouse_px, vertex_count, Element::Vertex,
        *one_shot.SelectionReady
    );
}

std::vector<entt::entity> RunObjectPick(entt::registry &R, entt::entity viewport, uint32_t &object_pick_epoch_tag, uvec2 mouse_px, uint32_t radius_px) {
    const auto &Vk = R.ctx().get<const VulkanResources>();
    const auto &one_shot = R.ctx().get<const OneShotGpu>();
    const auto &sel_slots = R.get<const SelectionSlots>(viewport);
    auto &Buffers = R.ctx().get<GpuBuffers>();
    auto &pipelines = R.ctx().get<const Pipelines>();
    const uint32_t next_object_id = R.ctx().get<const ObjectIdCounter>().Next;
    if (next_object_id <= 1) return {}; // No objects have been assigned IDs yet
    const uint32_t max_object_id = std::min(next_object_id - 1, GpuBuffers::MaxSelectableObjects);
    if (max_object_id == 0) return {};

    const bool selection_rendered = R.all_of<SelectionStale>(viewport);
    if (selection_rendered) RenderSelectionPass(R, viewport, *one_shot.SelectionReady);

    const Timer timer{"RunObjectPick"};
    // ObjectPickKeyBuffer is persistent across clicks: high 8 bits of each packed key store
    // a per-click epoch tag. We therefore avoid clearing all keys every click and only do a
    // full reset when the 8-bit epoch wraps; stale keys are filtered out by epoch on readback.
    if (object_pick_epoch_tag == 0) {
        ResetObjectPickKeys(Buffers);
        object_pick_epoch_tag = 255;
    }
    const uint32_t epoch_inv = object_pick_epoch_tag--;

    auto cb = *one_shot.Cb;
    RunSelectionCompute(
        cb, Vk.Queue, *one_shot.Fence, Vk.Device, pipelines.ObjectPick,
        ObjectPickPushConstants{
            .TargetPx = mouse_px,
            .Radius = radius_px,
            .MaxId = max_object_id,
            .EpochInv = epoch_inv,
            .HeadImageIndex = sel_slots.HeadImage,
            .SelectionNodesIndex = Buffers.SelectionNodeBuffer.Slot,
            .BestKeyIndex = sel_slots.ObjectPickKey,
            .SeenBitsIndex = sel_slots.ObjectPickSeenBits,
        },
        [](vk::CommandBuffer dispatch_cb) { dispatch_cb.dispatch(1, 1, 1); }, // Single workgroup; threads cooperatively scan the radius.
        selection_rendered ? *one_shot.SelectionReady : vk::Semaphore{}
    );

    std::unordered_map<uint32_t, entt::entity> object_id_to_entity;
    for (const auto [e, ri] : R.view<RenderInstance>().each()) {
        if (ri.ObjectId > 0 && ri.ObjectId <= max_object_id) object_id_to_entity[ri.ObjectId] = e;
    }

    struct SortedHit {
        uint32_t DistSq;
        uint32_t Layer; // 0 = bone (on top in main pass), 1 = other
        uint32_t Depth;
        entt::entity Entity;
        auto operator<=>(const SortedHit &) const = default;
    };

    const auto *bits = Buffers.ObjectPickSeenBitset.Data();
    const auto *keys = Buffers.ObjectPickKeys.Data();
    std::vector<SortedHit> hits;
    for (uint32_t object_id = 1; object_id <= max_object_id; ++object_id) {
        const uint32_t idx = object_id - 1;
        if ((bits[idx / 32] & (1u << (idx % 32))) == 0) continue;
        const auto it = object_id_to_entity.find(object_id);
        if (it == object_id_to_entity.end()) continue;
        const uint32_t packed_key = keys[idx];
        if ((packed_key >> 24) != epoch_inv) continue;
        const uint32_t layer = R.any_of<BoneIndex, BoneSubPartOf>(it->second) ? 0u : 1u;
        hits.emplace_back(SortedHit{(packed_key >> 16) & 0xffu, layer, packed_key & 0xffffu, it->second});
    }
    std::ranges::sort(hits);

    std::vector<entt::entity> entities;
    entities.reserve(hits.size());
    for (const auto &hit : hits) entities.emplace_back(hit.Entity);
    return entities;
}

namespace {
void DispatchBoxSelect(entt::registry &R, entt::entity viewport, uvec2 box_min, uvec2 box_max, uint32_t max_id, vk::Semaphore wait_semaphore) {
    const auto &Vk = R.ctx().get<const VulkanResources>();
    const auto &one_shot = R.ctx().get<const OneShotGpu>();
    const auto &sel_slots = R.get<const SelectionSlots>(viewport);
    auto &Buffers = R.ctx().get<GpuBuffers>();
    auto &pipelines = R.ctx().get<const Pipelines>();
    const uint32_t bitset_words = (max_id + 31) / 32;
    memset(Buffers.SelectionBitset.Data(), 0, bitset_words * sizeof(uint32_t));

    const auto group_counts = glm::max((box_max - box_min + 15u) / 16u, uvec2{1, 1});
    RunSelectionCompute(
        *one_shot.Cb, Vk.Queue, *one_shot.Fence, Vk.Device, pipelines.BoxSelect,
        BoxSelectPushConstants{
            .BoxMin = box_min,
            .BoxMax = box_max,
            .MaxId = max_id,
            .HeadImageIndex = sel_slots.HeadImage,
            .SelectionNodesIndex = Buffers.SelectionNodeBuffer.Slot,
            .BoxResultIndex = sel_slots.SelectionBitset,
        },
        [group_counts](vk::CommandBuffer dispatch_cb) { dispatch_cb.dispatch(group_counts.x, group_counts.y, 1); },
        wait_semaphore
    );
}
} // namespace

std::vector<entt::entity> RunBoxSelect(entt::registry &R, entt::entity viewport, std::pair<uvec2, uvec2> box_px) {
    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return {};
    const auto &one_shot = R.ctx().get<const OneShotGpu>();
    auto &Buffers = R.ctx().get<GpuBuffers>();
    const uint32_t next_object_id = R.ctx().get<const ObjectIdCounter>().Next;
    if (next_object_id <= 1) return {}; // No objects have been assigned IDs yet

    const uint32_t max_object_id = std::min(next_object_id - 1, GpuBuffers::MaxSelectableObjects);

    const Timer timer{"RunBoxSelect"};
    const bool selection_rendered = R.all_of<SelectionStale>(viewport);
    if (selection_rendered) RenderSelectionPass(R, viewport, *one_shot.SelectionReady);
    DispatchBoxSelect(R, viewport, box_min, box_max, max_object_id, selection_rendered ? *one_shot.SelectionReady : vk::Semaphore{});

    std::unordered_map<uint32_t, entt::entity> object_id_to_entity;
    for (const auto [e, ri] : R.view<RenderInstance>().each()) object_id_to_entity[ri.ObjectId] = e;

    const auto *bits = Buffers.SelectionBitset.Data();
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
