#include "Scene.h"
#include "Armature.h"
#include "Bindless.h"
#include "EntityDestroyTracker.h"
#include "Instance.h"
#include "MeshComponents.h"
#include "Paths.h"
#include "Reactive.h"
#include "SceneApply.h"
#include "SceneChanges.h"
#include "SceneDefaults.h"
#include "ScenePipelines.h"
#include "SceneProcessEvents.h"
#include "SceneRenderGpu.h"
#include "SceneSelection.h"
#include "SceneStores.h"
#include "SceneTextures.h"
#include "SceneTree.h"
#include "SoundVertices.h"
#include "SvgResource.h"
#include "Timer.h"
#include "VideoRecorder.h"
#include "VkFenceWait.h"
#include "gltf/GltfScene.h"
#include "gpu/BoxSelectPushConstants.h"
#include "gpu/ElementPickPushConstants.h"
#include "gpu/MainDrawPushConstants.h"
#include "gpu/ObjectPickPushConstants.h"
#include "gpu/SelectionDrawPushConstants.h"
#include "gpu/SelectionElementPushConstants.h"
#include "gpu/SilhouetteEdgeColorPushConstants.h"
#include "gpu/SilhouetteEdgeDepthObjectPushConstants.h"
#include "mesh/MeshStore.h"
#include "physics/PhysicsWorld.h"

#include "imgui.h"

#include "AxisColors.h" // Must be after imgui.h

using std::ranges::any_of, std::ranges::find, std::ranges::fold_left;

namespace {
constexpr vk::Extent2D ToExtent2D(vk::Extent3D extent) { return {extent.width, extent.height}; }
const vk::ClearColorValue Transparent{0, 0, 0, 0};
const std::vector<vk::ClearValue> SilhouetteClearValues{{vk::ClearDepthStencilValue{1, 0}}, {Transparent}};

using namespace he;
} // namespace

#include "scene_impl/SceneComponents.h"

#include "scene_impl/SceneBuffers.h"
#include "scene_impl/SceneDrawing.h"
#include "scene_impl/SceneTransformUtils.h"

namespace {
// Returns `std::nullopt` if the entity does not have a RenderInstance (i.e., is not visible).
std::optional<uint32_t> GetModelBufferIndex(const entt::registry &r, entt::entity e) {
    if (const auto *ri = r.try_get<RenderInstance>(e)) return ri->BufferIndex;
    return {};
}

} // namespace

void AppendExtrasDraw(entt::registry &r, const InstanceArena &instances, DrawListBuilder &dl, DrawBatchInfo &batch, auto &&customize_draw) {
    batch = dl.BeginBatch();
    for (auto [entity, mesh_buffers, models] : r.view<ObjectExtrasTag, MeshBuffers, ModelsBuffer>().each()) {
        if (mesh_buffers.EdgeIndices.Count == 0) continue;
        auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, instances);
        if (const auto *vcr = r.try_get<VertexClass>(entity)) draw.VertexClassOffset = vcr->Offset;
        customize_draw(draw, instances);
        AppendDraw(dl, batch, mesh_buffers.EdgeIndices, models, draw);
    }
}

void ResetObjectPickKeys(SceneBuffers &buffers) {
    std::fill_n(buffers.ObjectPickKeys.Data(), SceneBuffers::MaxSelectableObjects, std::numeric_limits<uint32_t>::max());
}

// True if any component of type C has `C.*field == target`.
template<class C, class F>
bool AnyComponentRefersTo(entt::registry &R, F C::*field, entt::entity target) {
    return any_of(R.view<C>().each(), [=](const auto &entry) { return std::get<1>(entry).*field == target; });
}

namespace {

struct DeformSlots {
    uint32_t BoneDeformOffset{InvalidOffset}, ArmatureDeformOffset{InvalidOffset}, MorphDeformOffset{InvalidOffset};
    uint32_t MorphTargetCount{0};
    // Per-instance morph weights: buffer_index -> offset (weights are per-node in glTF)
    std::unordered_map<uint32_t, uint32_t> MorphWeightsByBufferIndex;
};

std::unordered_map<entt::entity, DeformSlots> BuildDeformSlots(const entt::registry &r, const MeshStore &meshes) {
    std::unordered_map<entt::entity, DeformSlots> result;
    for (const auto [_, instance, modifier] : r.view<const Instance, const ArmatureModifier>().each()) {
        if (result.contains(instance.Entity)) continue;
        const auto &mesh = r.get<const Mesh>(instance.Entity);
        const auto bone_deform = meshes.GetBoneDeformRange(mesh.GetStoreId());
        if (bone_deform.Count == 0) continue;
        if (const auto *pose_state = r.try_get<const ArmaturePoseState>(modifier.ArmatureEntity)) {
            result[instance.Entity] = {
                .BoneDeformOffset = bone_deform.Offset,
                .ArmatureDeformOffset = pose_state->GpuDeformRange.Offset,
                .MorphDeformOffset = InvalidOffset,
                .MorphTargetCount = 0,
                .MorphWeightsByBufferIndex = {},
            };
        }
    }
    // Add morph target slots for mesh instances with morph data (per-instance weights)
    for (const auto [instance_entity, instance, morph_state, ri] : r.view<const Instance, const MorphWeightState, const RenderInstance>().each()) {
        const auto mesh_entity = instance.Entity;
        const auto &mesh = r.get<const Mesh>(mesh_entity);
        const auto morph_range = meshes.GetMorphTargetRange(mesh.GetStoreId());
        if (morph_range.Count == 0) continue;
        auto &slots = result[mesh_entity];
        slots.MorphDeformOffset = morph_range.Offset;
        slots.MorphTargetCount = meshes.GetMorphTargetCount(mesh.GetStoreId());
        slots.MorphWeightsByBufferIndex[ri.BufferIndex] = morph_state.GpuWeightRange.Offset;
    }
    return result;
}

void PatchMorphWeights(DrawListBuilder &dl, size_t draws_before, const DeformSlots &deform) {
    if (deform.MorphWeightsByBufferIndex.empty()) return;
    for (size_t i = draws_before; i < dl.Draws.size(); ++i) {
        if (auto it = deform.MorphWeightsByBufferIndex.find(dl.Draws[i].FirstInstance); it != deform.MorphWeightsByBufferIndex.end()) {
            dl.Draws[i].MorphWeightsOffset = it->second;
        }
    }
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

// After rendering elements to selection buffer, dispatch compute shader to find the nearest element to mouse_px.
// Returns 0-based element index, or nullopt if no element found.
std::optional<uint32_t> FindNearestPickedElement(
    const SceneBuffers &buffers, const ComputePipeline &compute, vk::CommandBuffer cb,
    vk::Queue queue, vk::Fence fence, vk::Device device,
    uint32_t head_image_index, uint32_t selection_nodes_slot, uint32_t element_candidate_buffer_slot,
    uvec2 mouse_px, uint32_t max_element_id, Element element,
    vk::Semaphore wait_semaphore
) {
    const uint32_t radius = element == Element::Face ? 0u : ElementSelectRadiusPx;
    const uint32_t group_count = element == Element::Face ? 1u : SceneBuffers::ElementPickGroupCount;
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
    return fold_left(ranges, uint32_t{0}, [](uint32_t total, const auto &r) { return std::max(total, r.Offset + r.Count); });
}
} // namespace

namespace {

void AppendSelectedSilhouetteDraws(const entt::registry &R, entt::entity scene_entity, DrawListBuilder &draw_list, DrawBatchInfo &silhouette_batch) {
    const auto &buffers = R.get<const SceneBuffers>(scene_entity);
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
    entt::registry &R, entt::entity scene_entity,
    std::span<const ElementRange> ranges, Element element, bool write_bitset,
    uvec2 box_min, uvec2 box_max, vk::Semaphore signal_semaphore
) {
    if (ranges.empty() || element == Element::None) return;
    const auto &vk_res = R.ctx().get<const SceneVulkanResources>();
    const auto &pipelines = R.ctx().get<const ScenePipelines>();
    const auto &one_shot = R.get<const SceneOneShotGpu>(scene_entity);
    const auto &sel_slots = R.get<const SelectionSlots>(scene_entity);
    auto &meshes = R.ctx().get<MeshStore>();
    auto &buffers = R.get<SceneBuffers>(scene_entity);

    const auto primary_edit_instances = scene_selection::ComputePrimaryEditInstances(R);
    const bool xray_selection = R.get<const SelectionXRay>(scene_entity).Value;
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
        if (element != Element::Face) AppendSelectedSilhouetteDraws(R, scene_entity, draw_list, silhouette_batch);
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

    FlushDrawList(R, scene_entity, vk_res.Device, draw_list, buffers.SelectionDraw);
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

    const auto render_extent = ComputeRenderExtentPx(R.get<const ViewportExtent>(scene_entity).Value, std::bit_cast<vec2>(ImGui::GetIO().DisplayFramebufferScale));
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
    R.emplace_or_replace<SelectionStale>(scene_entity);
}

} // namespace

std::optional<std::pair<entt::entity, uint32_t>> RunElementPickFromRanges(
    entt::registry &R, entt::entity scene_entity,
    std::span<const ElementRange> ranges, Element element, uvec2 mouse_px
) {
    if (ranges.empty() || element == Element::None) return {};
    const auto element_count = MaxElementBound(ranges);
    if (element_count == 0) return {};

    const Timer timer{"RunElementPick"};
    const auto &vk_res = R.ctx().get<const SceneVulkanResources>();
    const auto &pipelines = R.ctx().get<const ScenePipelines>();
    const auto &one_shot = R.get<const SceneOneShotGpu>(scene_entity);
    const auto &sel_slots = R.get<const SelectionSlots>(scene_entity);
    auto &buffers = R.get<SceneBuffers>(scene_entity);
    RenderElementSelectionPass(R, scene_entity, ranges, element, false, {}, {}, *one_shot.SelectionReady);
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

Scene::Scene(SceneVulkanResources vc, entt::registry &r)
    : R{r},
      RenderFence{vc.Device.createFenceUnique({})} {
    InitSceneStoreCtx(R, vc);
    auto &Slots = R.ctx().get<DescriptorSlots>();
    auto &Pipelines = R.ctx().emplace<ScenePipelines>(vc.Device, vc.PhysicalDevice, Slots.GetSetLayout(), Slots.GetSet());
    auto &Physics = physics::Init(R);
    // Reactive storage subscriptions for deferred once-per-frame processing
    track<changes::TimelineRange>(R).on<TimelineRange>(On::Update);
    track<changes::Selected>(R).on<Selected>(On::Create | On::Destroy);
    track<changes::ActiveInstance>(R).on<Active>(On::Create | On::Destroy);
    track<changes::BoneSelection>(R).on<BoneSelection>(On::Create | On::Update | On::Destroy).on<BoneActive>(On::Create | On::Destroy);
    track<changes::Rerecord>(R)
        .on<RenderInstance>(On::Create | On::Destroy)
        .on<Active>(On::Create | On::Destroy)
        .on<StartTransform>(On::Create | On::Destroy)
        .on<SceneEditMode>(On::Create | On::Update)
        .on<SmoothShading>(On::Create | On::Destroy);
    track<changes::MeshActiveElement>(R).on<MeshActiveElement>(On::Create | On::Update);
    track<changes::MeshGeometry>(R).on<MeshGeometryDirty>(On::Create);
    track<changes::MeshMaterial>(R).on<MeshMaterialAssignment>(On::Create | On::Update);
    track<changes::SoundVertices>(R).on<SoundVertices>(On::Create | On::Destroy);
    track<changes::SoundVerticesUpdated>(R).on<SoundVertices>(On::Update);
    track<changes::VertexForce>(R).on<VertexForce>(On::Create | On::Destroy);
    track<changes::ModelsBuffer>(R).on<ModelsBuffer>(On::Update);
    track<changes::NewBufferEntity>(R).on<MeshBuffers>(On::Create);
    track<changes::RenderInstanceCreated>(R).on<RenderInstance>(On::Create);
    track<changes::SceneSettings>(R).on<SceneSettings>(On::Create | On::Update);
    track<changes::InteractionMode>(R).on<SceneInteraction>(On::Create | On::Update);
    track<changes::Submit>(R).on<SubmitDirty>(On::Create);
    track<changes::ViewportTheme>(R).on<ViewportTheme>(On::Create | On::Update);
    track<changes::Materials>(R).on<MaterialDirty>(On::Create | On::Update);
    track<changes::ActiveMaterialVariant>(R).on<MaterialVariants>(On::Create | On::Update);
    track<changes::PbrSpecialization>(R)
        .on<PbrMeshFeatures>(On::Create | On::Update | On::Destroy)
        .on<MaterialPreviewLighting>(On::Create | On::Update)
        .on<RenderedLighting>(On::Create | On::Update);
    track<changes::SceneView>(R)
        .on<ViewCamera>(On::Create | On::Update)
        .on<MaterialPreviewLighting>(On::Create | On::Update)
        .on<RenderedLighting>(On::Create | On::Update)
        .on<LightIndex>(On::Create | On::Destroy)
        .on<ViewportExtent>(On::Create | On::Update)
        .on<SceneEditMode>(On::Create | On::Update);
    track<changes::CameraLens>(R).on<Camera>(On::Create | On::Update).on<LookingThrough>(On::Create | On::Destroy);
    track<changes::Rotation>(R).on<Transform>(On::Create | On::Update);
    track<changes::WorldTransform>(R).on<WorldTransform>(On::Create | On::Update);
    track<changes::TransformPending>(R).on<PendingTransform>(On::Create | On::Update);
    track<changes::TransformEnd>(R).on<StartTransform>(On::Destroy);
    track<changes::TransformDirty>(R)
        .on<Transform>(On::Create | On::Update)
        .on<SceneNode>(On::Create | On::Update)
        .on<BoneDisplayScale>(On::Update);
    R.ctx().emplace<EntityDestroyTracker>().Bind(R);

    R.on_destroy<Name>().connect<&OnDestroyName>();
    R.on_construct<RenderInstance>().connect<&AssignRenderInstanceObjectId>();
    R.on_destroy<RenderInstance>().connect<&EmitPendingHideOnRenderInstanceDestroy>();
    // BoneConstraints edits change the resolved local Transform; poke it to drive the WorldTransform recompute.
    R.on_update<BoneConstraints>().connect<+[](entt::registry &r, entt::entity e) {
        r.patch<Transform>(e, [](auto &) {});
    }>();

    SceneEntity = WireSceneRegistry(R);
    auto &Buffers = R.get<SceneBuffers>(SceneEntity);
    R.emplace<SceneSettings>(SceneEntity);
    R.emplace<SceneInteraction>(SceneEntity);
    R.emplace<SceneEditMode>(SceneEntity);
    R.emplace<ViewportTheme>(SceneEntity, SceneDefaults::ViewportTheme);
    R.emplace<colors::AxesArray>(SceneEntity, colors::MakeAxes(SceneDefaults::ViewportTheme.AxisColors));
    R.emplace<ViewCamera>(SceneEntity, SceneDefaults::ViewCamera);
    R.emplace<MaterialPreviewLighting>(SceneEntity, false, false, 1.f, 0.f);
    R.emplace<RenderedLighting>(SceneEntity, true, true, 1.f, 0.f);
    R.emplace<ViewportExtent>(SceneEntity);
    R.emplace<PlaybackFrame>(SceneEntity);
    R.emplace<LastEvaluatedFrame>(SceneEntity);
    R.emplace<EnabledInteractionModes>(SceneEntity);
    R.emplace<SelectionBitsetRef>(SceneEntity, std::span<uint32_t>{Buffers.SelectionBitset.Data(), SceneBuffers::SelectionBitsetWords});
    R.emplace<SelectionSlots>(SceneEntity, Slots);
    R.emplace<SceneDrawState>(SceneEntity);
    const auto &one_shot = R.emplace<SceneOneShotGpu>(SceneEntity, MakeSceneOneShotGpu(vc.Device, vc.QueueFamily));
    R.emplace<ColliderShapeBuffers>(SceneEntity);
    RenderCommandBuffer = std::move(vc.Device.allocateCommandBuffersUnique({*one_shot.Pool, vk::CommandBufferLevel::ePrimary, 1}).front());
#ifdef MVK_FORCE_STAGED_TRANSFERS
    TransferCommandBuffer = std::move(vc.Device.allocateCommandBuffersUnique({*one_shot.Pool, vk::CommandBufferLevel::ePrimary, 1}).front());
#endif
    R.emplace<SelectionXRay>(SceneEntity);
    R.emplace<OrbitToActive>(SceneEntity);
    R.emplace<BoxSelectState>(SceneEntity);
    R.emplace<TransformGizmoState>(SceneEntity);
    R.emplace<AnimationTimelineView>(SceneEntity);
    R.emplace<SelectionStale>(SceneEntity); // Initial state: fragments need rendering on first selection use.
    Physics.ApplySimulationSettings(R.emplace<PhysicsSimulationSettings>(SceneEntity));

    Buffers.WorkspaceLightsUBO.Update(as_bytes(SceneDefaults::WorkspaceLights));
    ResetObjectPickKeys(Buffers);

    auto init_batch = BeginTextureUploadBatch(vc.Device, *one_shot.Pool, Buffers.Ctx);
    auto &environments = R.ctx().get<EnvironmentStore>();
    const auto images_dir = Paths::Res() / "images";
    environments.BrdfLut = CreateDefaultLutTexture(vc, init_batch, Slots, images_dir / "lut_ggx.png", "DefaultGGXBRDFLUT");
    environments.SheenELut = CreateDefaultLutTexture(vc, init_batch, Slots, images_dir / "lut_sheen_E.png", "DefaultSheenELUT");
    environments.CharlieLut = CreateDefaultLutTexture(vc, init_batch, Slots, images_dir / "lut_charlie.png", "DefaultCharlieLUT");
    // Blender's default world background color (linear RGB) - flat ambient-only IBL when no scene world is provided.
    environments.EmptySceneWorld = BuildFlatColorEnvironment(vc, init_batch, Slots, vec3{0.05f}, "EmptySceneWorld");
    SubmitTextureUploadBatch(init_batch, vc.Queue, *one_shot.Fence, vc.Device);

    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator{images_dir / "studiolights" / "world", ec}) {
        if (entry.path().extension() == ".hdr") {
            environments.Hdris.emplace_back(HdriEntry{.Name = entry.path().stem().string(), .Path = entry.path(), .Prefiltered = {}});
        }
    }
    std::ranges::sort(environments.Hdris, {}, &HdriEntry::Name);
    const auto forest_it = find(environments.Hdris, "forest", &HdriEntry::Name);
    environments.ActiveHdriIndex = forest_it != environments.Hdris.end() ? std::distance(environments.Hdris.begin(), forest_it) : 0;
    SetStudioEnvironment(R, SceneEntity, environments.ActiveHdriIndex);
    environments.SceneWorld = {.Ibl = MakeIblSamplers(environments.EmptySceneWorld, environments), .Name = environments.EmptySceneWorld.Name};

    Pipelines.CompileShaders();
}

Scene::~Scene() {
    // Free command buffers before their owning pool is torn down with SceneEntity.
    RenderCommandBuffer.reset();
#ifdef MVK_FORCE_STAGED_TRANSFERS
    TransferCommandBuffer.reset();
#endif
    if (R.valid(SceneEntity)) R.remove<MaterialStore>(SceneEntity);
    R.clear<Mesh>();
    R.ctx().erase<std::vector<ComponentEventHandler>>();
    R.ctx().erase<EntityDestroyTracker>();
    physics::Deinit(R);
    R.ctx().erase<ScenePipelines>();
    TearDownSceneStoreCtx(R);
}

void Scene::CreateSvgResource(std::unique_ptr<SvgResource> &svg, std::filesystem::path path) {
    const auto &Vk = R.ctx().get<const SceneVulkanResources>();
    const auto &one_shot = R.get<const SceneOneShotGpu>(SceneEntity);
    auto &Buffers = R.get<SceneBuffers>(SceneEntity);
    auto svg_batch = BeginTextureUploadBatch(Vk.Device, *one_shot.Pool, Buffers.Ctx);
    const auto RenderBitmap = [&Vk, &svg_batch](std::span<const std::byte> data, uint32_t width, uint32_t height) {
        return RenderBitmapToImage(Vk, svg_batch, data, width, height, Format::Color, ColorSubresourceRange);
    };
    svg = std::make_unique<SvgResource>(Vk.Device, RenderBitmap, std::move(path));
    SubmitTextureUploadBatch(svg_batch, Vk.Queue, *one_shot.Fence, Vk.Device);
}

void Scene::LoadIcons() {
    const auto &Vk = R.ctx().get<const SceneVulkanResources>();
    const auto &one_shot = R.get<const SceneOneShotGpu>(SceneEntity);
    auto &Buffers = R.get<SceneBuffers>(SceneEntity);
    const auto svg_path = Paths::Res() / "svg";
    auto batch = BeginTextureUploadBatch(Vk.Device, *one_shot.Pool, Buffers.Ctx);
    const auto RenderBitmap = [&Vk, &batch](std::span<const std::byte> data, uint32_t width, uint32_t height) {
        return RenderBitmapToImage(Vk, batch, data, width, height, Format::Color, ColorSubresourceRange);
    };

    const std::pair<std::unique_ptr<SvgResource> *, std::string_view> entries[] = {
        {&Icons.Select, "select.svg"},
        {&Icons.SelectBox, "select_box.svg"},
        {&Icons.Move, "move.svg"},
        {&Icons.Rotate, "rotate.svg"},
        {&Icons.Scale, "scale.svg"},
        {&Icons.Universal, "transform.svg"},
        {&ShadingIcons.Wireframe, "shading_wire.svg"},
        {&ShadingIcons.Solid, "shading_solid.svg"},
        {&ShadingIcons.MaterialPreview, "shading_texture.svg"},
        {&ShadingIcons.Rendered, "shading_rendered.svg"},
        {&OverlayIcon, "overlay.svg"},
        {&AnimIcons.Play, "play.svg"},
        {&AnimIcons.Pause, "pause.svg"},
        {&AnimIcons.JumpStart, "jump_start.svg"},
        {&AnimIcons.JumpEnd, "jump_end.svg"},
    };
    for (const auto &[svg, name] : entries) {
        *svg = std::make_unique<SvgResource>(Vk.Device, RenderBitmap, svg_path / name);
    }

    SubmitTextureUploadBatch(batch, Vk.Queue, *one_shot.Fence, Vk.Device);
}

namespace {
std::pair<vk::Offset3D, vk::Extent2D> GetCaptureRegion(const entt::registry &r) {
    auto &Pipelines = r.ctx().get<const ScenePipelines>();
    const auto full = ToExtent2D(Pipelines.Main.Resources->FinalColorImage.Extent);
    const auto camera = LookThroughCameraEntity(r);
    const auto *cd = camera != entt::null ? r.try_get<Camera>(camera) : nullptr;
    if (!cd) return {{0, 0, 0}, full};

    const auto cam_aspect = AspectRatio(*cd);
    const auto ratio = LookThroughFrameRatio(cam_aspect, float(full.width) / float(full.height));
    // yuv420p requires even width/height.
    const auto w = uint32_t(float(full.height) * cam_aspect * ratio) & ~1u;
    const auto h = uint32_t(float(full.height) * ratio) & ~1u;
    return {{int32_t(full.width - w) / 2, int32_t(full.height - h) / 2, 0}, {w, h}};
}
} // namespace

void Scene::StartRecording(std::filesystem::path path, int fps) {
    StopRecording();
    auto &Pipelines = R.ctx().get<const ScenePipelines>();
    if (!Pipelines.Main.Resources) {
        std::println(stderr, "Scene::StartRecording: render resources not ready");
        return;
    }
    RecordRegion = GetCaptureRegion(R);
    const auto &Vk = R.ctx().get<const SceneVulkanResources>();
    Recorder = std::make_unique<VideoRecorder>(Vk, std::move(path), RecordRegion.first, RecordRegion.second, fps);
}

void Scene::StopRecording() { Recorder.reset(); }
bool Scene::IsRecording() const { return Recorder && Recorder->IsActive(); }
uint64_t Scene::CapturedFrameCount() const { return Recorder ? Recorder->CapturedFrameCount() : 0; }

void Scene::CaptureRecordFrame() {
    auto &Pipelines = R.ctx().get<const ScenePipelines>();
    if (!IsRecording() || !Pipelines.Main.Resources) return;
    if (GetCaptureRegion(R) != RecordRegion) {
        std::println(stderr, "Scene: capture region changed; stopping recording.");
        StopRecording();
        return;
    }
    Recorder->CaptureFrame(*Pipelines.Main.Resources->FinalColorImage.Image);
}

std::string Scene::DebugBufferHeapUsage() const { return R.get<const SceneBuffers>(SceneEntity).Ctx.DebugHeapUsage(); }

void Scene::RecordRenderCommandBuffer(bool silhouette_only) {
    const Timer timer{silhouette_only ? "RecordRenderCommandBuffer (silhouette)" : "RecordRenderCommandBuffer"};
    if (!silhouette_only) R.emplace_or_replace<SelectionStale>(SceneEntity);

    const auto &Vk = R.ctx().get<const SceneVulkanResources>();
    auto &Buffers = R.get<SceneBuffers>(SceneEntity);
    auto &Meshes = R.ctx().get<MeshStore>();
    auto &Pipelines = R.ctx().get<ScenePipelines>();
    const auto &settings = R.get<const SceneSettings>(SceneEntity);
    const auto interaction_mode = R.get<const SceneInteraction>(SceneEntity).Mode;
    const auto edit_mode = R.get<const SceneEditMode>(SceneEntity).Value;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;
    const bool is_excite_mode = interaction_mode == InteractionMode::Excite;
    const bool is_wireframe_mode = settings.ViewportShading == ViewportShadingMode::Wireframe;
    const bool show_rendered = settings.ViewportShading == ViewportShadingMode::MaterialPreview || settings.ViewportShading == ViewportShadingMode::Rendered;
    const bool show_fill = !is_wireframe_mode, show_overlays = settings.ShowOverlays;
    const bool real_transmission = show_rendered &&
        GetActivePbrLighting(R, SceneEntity, settings.ViewportShading).RealTransmission &&
        Pipelines.Main.Compiler.HasFeature(PbrFeature::Transmission);

    const auto &sel_slots = R.get<const SelectionSlots>(SceneEntity);
    auto &draw = R.get<SceneDrawState>(SceneEntity);
    auto &draw_list = draw.List;

    // Build mesh_entity -> deform slots mapping for skinned meshes (edit mode shows rest pose)
    const auto mesh_deform_slots = is_edit_mode ? std::unordered_map<entt::entity, DeformSlots>{} : BuildDeformSlots(R, Meshes);
    static const DeformSlots no_deform{};
    const auto get_deform_slots = [&](entt::entity mesh_entity) -> const DeformSlots & {
        if (auto it = mesh_deform_slots.find(mesh_entity); it != mesh_deform_slots.end()) return it->second;
        return no_deform;
    };

    const auto is_silhouette_eligible = [&](entt::entity e) {
        if (!R.all_of<Instance, RenderInstance>(e)) return false;
        const auto buffer_entity = R.get<const Instance>(e).Entity;
        if (!R.valid(buffer_entity) || R.all_of<ObjectExtrasTag>(buffer_entity)) return false;
        // Bones get outlines from BoneWire/BoneSphereWire, not the screen-space silhouette system.
        if (R.all_of<ArmatureObject>(buffer_entity) || R.all_of<BoneJoint>(buffer_entity)) return false;
        const auto *mesh_buffers = R.try_get<const MeshBuffers>(buffer_entity);
        return mesh_buffers && mesh_buffers->FaceIndices.Count > 0;
    };

    // In Edit mode, compute primary edit instances (for draw routing and silhouette filtering)
    // and edit transform context (for pending vertex transforms) in one pass.
    std::unordered_map<entt::entity, entt::entity> primary_edit_instances;
    EditTransformContext edit_transform_context;
    const bool has_pending_transform = is_edit_mode && R.all_of<PendingTransform>(SceneEntity);
    if (is_edit_mode) {
        const auto active = FindActiveEntity(R);
        for (const auto [e, instance, ok, ri] : R.view<const Instance, const Selected, const ObjectKind, const RenderInstance>().each()) {
            if (ok.Value != ObjectType::Mesh) continue;
            auto &primary = primary_edit_instances[instance.Entity];
            if (primary == entt::entity{} || e == active) primary = e;
            if (has_pending_transform && !R.all_of<ScaleLocked>(e)) {
                auto &primary_uf = edit_transform_context.TransformInstances[instance.Entity];
                if (primary_uf == entt::entity{} || e == active) primary_uf = e;
            }
        }
    }
    const auto patch_edit_pending_local_transform = [&](size_t draws_before, entt::entity mesh_entity) {
        if (!has_pending_transform) return;
        const auto context_it = edit_transform_context.TransformInstances.find(mesh_entity);
        if (context_it == edit_transform_context.TransformInstances.end()) return;
        const auto *primary_ri = R.try_get<const RenderInstance>(context_it->second);
        if (!primary_ri) return;
        for (size_t i = draws_before; i < draw_list.Draws.size(); ++i) {
            draw_list.Draws[i].HasPendingVertexTransform = 1u;
            draw_list.Draws[i].PrimaryEditInstanceIndex = primary_ri->BufferIndex;
        }
    };

    if (!silhouette_only) {
        draw_list.Draws.clear();
        draw_list.IndirectCommands.clear();
        draw_list.MaxIndexCount = 0;

        std::unordered_set<entt::entity> excitable_mesh_entities;
        if (is_excite_mode) {
            for (const auto [e, instance, excitable] : R.view<const Instance, const SoundVertices>().each()) {
                excitable_mesh_entities.emplace(instance.Entity);
            }
        }

        std::vector<entt::entity> blend_mesh_order;
        if (show_rendered) {
            // Transparent pass ordering: sort mesh draws back-to-front by camera distance.
            // This is a mesh-level approximation; interpenetrating transparent geometry may still require
            // per-primitive sorting or OIT for fully correct compositing.
            const auto camera_position = R.get<const ViewCamera>(SceneEntity).Position();
            std::unordered_map<entt::entity, float> farthest_distance2_by_mesh;
            farthest_distance2_by_mesh.reserve(R.storage<RenderInstance>().size());
            for (const auto [entity, _, wt] : R.view<const RenderInstance, const WorldTransform>().each()) {
                entt::entity mesh_entity = entity;
                if (const auto *instance = R.try_get<const Instance>(entity)) mesh_entity = instance->Entity;
                if (!R.valid(mesh_entity) || !R.all_of<Mesh>(mesh_entity)) continue;
                const auto delta = wt.P - camera_position;
                const auto distance2 = dot(delta, delta);
                if (const auto it = farthest_distance2_by_mesh.find(mesh_entity); it != farthest_distance2_by_mesh.end()) {
                    it->second = std::max(it->second, distance2);
                } else {
                    farthest_distance2_by_mesh.emplace(mesh_entity, distance2);
                }
            }
            blend_mesh_order.reserve(farthest_distance2_by_mesh.size());
            for (const auto &[mesh_entity, _] : farthest_distance2_by_mesh) blend_mesh_order.emplace_back(mesh_entity);
            std::ranges::sort(
                blend_mesh_order,
                [&](const auto a, const auto b) {
                    return farthest_distance2_by_mesh.at(a) > farthest_distance2_by_mesh.at(b);
                }
            );
        }

        // Single-pass entity data collection: avoids redundant ECS view iterations across fill, edge, wire, point, and normal batches.
        struct MeshEntityData {
            entt::entity Entity;
            const MeshBuffers &Buf;
            const ModelsBuffer &Mod;
            const Mesh *MeshComp; // nullptr if entity has no Mesh component
            const DeformSlots &Deform;
            std::optional<uint32_t> PrimaryEditBufferIndex;
            bool IsSoundVertices, IsBone, IsBoneJoint, IsExtras, Smooth;
        };

        std::vector<MeshEntityData> mesh_entities;
        mesh_entities.reserve(R.storage<MeshBuffers>().size());
        for (auto [entity, mesh_buffers, models] : R.view<MeshBuffers, ModelsBuffer>().each()) {
            std::optional<uint32_t> primary_bi;
            if (auto it = primary_edit_instances.find(entity); it != primary_edit_instances.end()) {
                primary_bi = R.get<RenderInstance>(it->second).BufferIndex;
            }
            const bool is_bone_joint = R.all_of<BoneJoint>(entity);
            mesh_entities.emplace_back(entity, mesh_buffers, models, R.try_get<const Mesh>(entity), get_deform_slots(entity), primary_bi, excitable_mesh_entities.contains(entity), R.all_of<ArmatureObject>(entity) || is_bone_joint, is_bone_joint, R.all_of<ObjectExtrasTag>(entity), R.all_of<SmoothShading>(entity));
        }

        // Entity -> mesh_entities index map for blend_mesh_order lookup
        std::unordered_map<entt::entity, size_t> blend_entity_idx;
        if (show_rendered) {
            blend_entity_idx.reserve(mesh_entities.size());
            for (size_t i = 0; i < mesh_entities.size(); ++i) blend_entity_idx[mesh_entities[i].Entity] = i;
        }

        if (show_fill) {
            const auto append_fill_mesh = [&](DrawBatchInfo &batch, const MeshEntityData &e, std::optional<bool> blend_target) {
                if (!e.MeshComp || e.MeshComp->FaceCount() == 0) return;
                const auto &mesh_buffers = e.Buf;
                const auto &models = e.Mod;
                const auto &mesh = *e.MeshComp;
                const auto &deform = e.Deform;
                auto dd = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, Buffers.Instances, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
                const auto face_id_buffer = Meshes.GetFaceIdRange(mesh.GetStoreId());
                const auto face_state_buffer = Meshes.GetFaceStateRange(mesh.GetStoreId());
                const auto face_primitive_buffer = Meshes.GetFacePrimitiveRange(mesh.GetStoreId());
                const auto primitive_material_buffer = Meshes.GetPrimitiveMaterialRange(mesh.GetStoreId());
                dd.ObjectIdSlot = face_id_buffer.Slot;
                dd.FaceIdOffset = face_id_buffer.Offset;
                dd.FaceFirstTriOffset = e.Smooth ? InvalidOffset : Meshes.GetFaceFirstTriRange(mesh.GetStoreId()).Offset;
                dd.FacePrimitiveOffset = face_primitive_buffer.Count > 0 ? face_primitive_buffer.Offset : InvalidOffset;
                dd.PrimitiveMaterialOffset = primitive_material_buffer.Count > 0 ? primitive_material_buffer.Offset : InvalidOffset;
                const auto append_fill_draw = [&](const DrawData &dd, uint32_t index_count, std::optional<uint32_t> model_index) {
                    const auto db = draw_list.Draws.size();
                    AppendDraw(draw_list, batch, index_count, models, dd, model_index);
                    PatchMorphWeights(draw_list, db, deform);
                    patch_edit_pending_local_transform(db, e.Entity);
                };
                const auto append_fill_for_instances = [&](const DrawData &dd, uint32_t index_count) {
                    if (e.PrimaryEditBufferIndex) {
                        // Draw primary with element state first, then all without (depth LESS won't overwrite)
                        auto primary_draw = dd;
                        primary_draw.ElementStateSlotOffset = face_state_buffer;
                        append_fill_draw(primary_draw, index_count, *e.PrimaryEditBufferIndex);
                        auto other_draw = dd;
                        other_draw.ElementStateSlotOffset = {};
                        append_fill_draw(other_draw, index_count, std::nullopt);
                    } else {
                        auto all_draw = dd;
                        all_draw.ElementStateSlotOffset = face_state_buffer;
                        append_fill_draw(all_draw, index_count, std::nullopt);
                    }
                };

                if (show_rendered) {
                    const auto primitive_materials = Meshes.GetPrimitiveMaterialIndices(mesh.GetStoreId());
                    const auto primitive_ranges = Meshes.GetPrimitiveTriangleRanges(mesh.GetStoreId());
                    if (!primitive_materials.empty() && !primitive_ranges.empty()) {
                        const auto material_count = Buffers.Materials.Count();
                        // Merge adjacent primitives with the same blend mode into single draw calls.
                        struct BlendDrawRange {
                            bool Blend;
                            uint32_t FirstTriangle, TriangleCount;
                        };
                        std::vector<BlendDrawRange> blend_ranges;
                        blend_ranges.reserve(primitive_ranges.size());
                        for (const auto &pr : primitive_ranges) {
                            if (pr.TriangleCount == 0u) continue;
                            auto pi = pr.PrimitiveIndex;
                            if (pi >= primitive_materials.size()) pi = primitive_materials.size() - 1u;
                            const bool is_blend = primitive_materials[pi] < material_count && Buffers.Materials.Get(primitive_materials[pi]).AlphaMode == MaterialAlphaMode::Blend;
                            if (!blend_ranges.empty() && blend_ranges.back().Blend == is_blend) blend_ranges.back().TriangleCount += pr.TriangleCount;
                            else blend_ranges.emplace_back(is_blend, pr.FirstTriangle, pr.TriangleCount);
                        }
                        for (const auto &range : blend_ranges) {
                            if (blend_target && range.Blend != *blend_target) continue;
                            auto range_draw = dd;
                            range_draw.IndexSlotOffset.Offset += range.FirstTriangle * 3u;
                            range_draw.FaceIdOffset += range.FirstTriangle;
                            append_fill_for_instances(range_draw, range.TriangleCount * 3u);
                        }
                        return;
                    }
                }

                if (!blend_target || !*blend_target) append_fill_for_instances(dd, mesh_buffers.FaceIndices.Count);
            };

            if (show_rendered) {
                draw.FillOpaque = draw_list.BeginBatch();
                for (const auto &e : mesh_entities) {
                    if (!e.IsBone) append_fill_mesh(draw.FillOpaque, e, false);
                }
                draw.FillBlend = draw_list.BeginBatch();
                for (const auto mesh_entity : blend_mesh_order) {
                    if (auto it = blend_entity_idx.find(mesh_entity); it != blend_entity_idx.end()) {
                        append_fill_mesh(draw.FillBlend, mesh_entities[it->second], true);
                    }
                }
            } else {
                draw.FillOpaque = draw_list.BeginBatch();
                for (const auto &e : mesh_entities) {
                    if (!e.IsBone) append_fill_mesh(draw.FillOpaque, e, std::nullopt);
                }
            }
        }

        // Build bone batches for X-ray rendering (drawn after a mid-pass depth clear so bones are never occluded by scene meshes)
        // Only draw bones for: active armature in Edit/Pose mode, selected armatures in Object mode.
        const auto should_draw_armature_bones = [&](entt::entity arm_obj_entity) {
            if (is_wireframe_mode) return true; // Wireframe mode: always show bone outlines
            const bool is_bone_mode = is_edit_mode || interaction_mode == InteractionMode::Pose;
            if (is_bone_mode) return R.all_of<Active>(arm_obj_entity);
            return R.all_of<Selected>(arm_obj_entity);
        };
        // Map BoneJoint entities back to their owning armature object entities.
        std::unordered_map<entt::entity, entt::entity> joint_to_owner;
        for (const auto [e, arm_obj] : R.view<const ArmatureObject>().each()) {
            if (arm_obj.JointEntity != entt::null) joint_to_owner[arm_obj.JointEntity] = e;
        }
        draw.BoneFill = {};
        draw.BoneWire = {};
        draw.BoneSphereFill = {};
        draw.BoneSphereWire = {};
        if (show_overlays && settings.ShowBones) {
            draw.BoneFill = draw_list.BeginBatch();
            for (const auto [entity, arm_obj, mesh_buffers, models] : R.view<const ArmatureObject, const MeshBuffers, const ModelsBuffer>().each()) {
                if (mesh_buffers.FaceIndices.Count == 0) continue;
                auto fill_draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, Buffers.Instances);
                fill_draw.InstanceStateSlot = Buffers.Instances.StateBuffer.Slot;
                AppendDraw(draw_list, draw.BoneFill, mesh_buffers.FaceIndices, models, fill_draw);
            }
            draw.BoneWire = draw_list.BeginBatch();
            for (const auto [entity, arm_obj, mesh_buffers, models] : R.view<const ArmatureObject, const MeshBuffers, const ModelsBuffer>().each()) {
                if (!should_draw_armature_bones(entity)) continue;
                if (const auto *adj = R.try_get<const BoneAdjacencyIndices>(entity)) {
                    auto wire_draw = MakeDrawData(mesh_buffers.Vertices, adj->Indices, Buffers.Instances);
                    wire_draw.InstanceStateSlot = Buffers.Instances.StateBuffer.Slot;
                    AppendDraw(draw_list, draw.BoneWire, adj->Indices.Count / 2, models, wire_draw);
                }
            }

            // Joint sphere batches
            draw.BoneSphereFill = draw_list.BeginBatch();
            for (const auto [entity, mesh_buffers, models] : R.view<const BoneJoint, const MeshBuffers, const ModelsBuffer>().each()) {
                if (mesh_buffers.FaceIndices.Count == 0) continue;
                auto fill_draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, Buffers.Instances);
                fill_draw.InstanceStateSlot = Buffers.Instances.StateBuffer.Slot;
                AppendDraw(draw_list, draw.BoneSphereFill, mesh_buffers.FaceIndices, models, fill_draw);
            }
            draw.BoneSphereWire = draw_list.BeginBatch();
            for (const auto [entity, mesh_buffers, models] : R.view<const BoneJoint, const MeshBuffers, const ModelsBuffer>().each()) {
                if (mesh_buffers.EdgeIndices.Count == 0) continue;
                if (const auto it = joint_to_owner.find(entity); it != joint_to_owner.end() && !should_draw_armature_bones(it->second)) continue;
                auto wire_draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.EdgeIndices, Buffers.Instances);
                wire_draw.InstanceStateSlot = Buffers.Instances.StateBuffer.Slot;
                AppendDraw(draw_list, draw.BoneSphereWire, mesh_buffers.EdgeIndices, models, wire_draw);
            }
        }

        // Edge quad batch (edit/excite mode triangle quads with self-AA, matches Blender's overlay_edit_mesh_edge)
        draw.EdgeQuad = draw_list.BeginBatch();
        if (is_edit_mode || is_excite_mode) {
            for (const auto &e : mesh_entities) {
                // Line meshes use draw.WireLine
                if (e.IsBone || e.IsExtras || !e.MeshComp || e.Buf.EdgeIndices.Count == 0 || e.Buf.FaceIndices.Count == 0) continue;
                auto dd = MakeDrawData(e.Buf.Vertices, e.Buf.EdgeIndices, Buffers.Instances, e.Deform.BoneDeformOffset, e.Deform.ArmatureDeformOffset, e.Deform.MorphDeformOffset, e.Deform.MorphTargetCount);
                dd.ElementStateSlotOffset = Meshes.GetEdgeStateRange(e.MeshComp->GetStoreId());
                const auto db = draw_list.Draws.size();
                if (e.PrimaryEditBufferIndex) AppendDraw(draw_list, draw.EdgeQuad, e.Buf.EdgeIndices.Count * 3, e.Mod, dd, *e.PrimaryEditBufferIndex);
                else if (e.IsSoundVertices) AppendDraw(draw_list, draw.EdgeQuad, e.Buf.EdgeIndices.Count * 3, e.Mod, dd);
                PatchMorphWeights(draw_list, db, e.Deform);
                patch_edit_pending_local_transform(db, e.Entity);
            }
        }
        // Wire line batch (wireframe mode + line meshes, matches Blender's wireframe overlay)
        draw.WireLine = draw_list.BeginBatch();
        for (const auto &e : mesh_entities) {
            if (e.IsBone || e.IsExtras || !e.MeshComp || e.Buf.EdgeIndices.Count == 0) continue;
            if (e.Buf.FaceIndices.Count > 0 && !is_wireframe_mode) continue;
            auto dd = MakeDrawData(e.Buf.Vertices, e.Buf.EdgeIndices, Buffers.Instances, e.Deform.BoneDeformOffset, e.Deform.ArmatureDeformOffset, e.Deform.MorphDeformOffset, e.Deform.MorphTargetCount);
            dd.ElementStateSlotOffset = Meshes.GetEdgeStateRange(e.MeshComp->GetStoreId());
            const auto db = draw_list.Draws.size();
            AppendDraw(draw_list, draw.WireLine, e.Buf.EdgeIndices, e.Mod, dd);
            PatchMorphWeights(draw_list, db, e.Deform);
            patch_edit_pending_local_transform(db, e.Entity);
        }

        draw.ExtrasLine = {};
        if (show_overlays && settings.ShowExtras) {
            AppendExtrasDraw(R, Buffers.Instances, draw_list, draw.ExtrasLine, [](auto &, const auto &) {});
        }

        // Point batch
        draw.Point = draw_list.BeginBatch();
        for (const auto &e : mesh_entities) {
            if (e.IsBone) continue;
            const bool is_point_mesh = e.Buf.FaceIndices.Count == 0 && e.Buf.EdgeIndices.Count == 0;
            if (!is_point_mesh && !((is_edit_mode && edit_mode == Element::Vertex) || is_excite_mode)) continue;
            auto dd = MakeDrawData(e.Buf.Vertices, e.Buf.VertexIndices, Buffers.Instances, e.Deform.BoneDeformOffset, e.Deform.ArmatureDeformOffset, e.Deform.MorphDeformOffset, e.Deform.MorphTargetCount);
            dd.ElementStateSlotOffset = {Meshes.GetVertexStateSlot(), e.Buf.Vertices.Offset};
            const auto db = draw_list.Draws.size();
            if (is_point_mesh) AppendDraw(draw_list, draw.Point, e.Buf.VertexIndices, e.Mod, dd);
            else if (e.PrimaryEditBufferIndex) AppendDraw(draw_list, draw.Point, e.Buf.VertexIndices, e.Mod, dd, *e.PrimaryEditBufferIndex);
            else if (e.IsSoundVertices) AppendDraw(draw_list, draw.Point, e.Buf.VertexIndices, e.Mod, dd);
            PatchMorphWeights(draw_list, db, e.Deform);
            patch_edit_pending_local_transform(db, e.Entity);
        }

        { // Normal overlay + bbox batches
            const auto vertex_slot = Buffers.VertexBuffer.Buffer.Slot;
            draw.OverlayFaceNormals = draw_list.BeginBatch();
            for (const auto &e : mesh_entities) {
                if (auto it = e.Buf.NormalIndicators.find(Element::Face); it != e.Buf.NormalIndicators.end()) {
                    auto dd = MakeDrawData(it->second, vertex_slot, Buffers.Instances);
                    AppendDraw(draw_list, draw.OverlayFaceNormals, it->second.Indices, e.Mod, dd);
                }
            }
            draw.OverlayVertexNormals = draw_list.BeginBatch();
            for (const auto &e : mesh_entities) {
                if (auto it = e.Buf.NormalIndicators.find(Element::Vertex); it != e.Buf.NormalIndicators.end()) {
                    auto dd = MakeDrawData(it->second, vertex_slot, Buffers.Instances);
                    AppendDraw(draw_list, draw.OverlayVertexNormals, it->second.Indices, e.Mod, dd);
                }
            }
        }

        { // Build selection draw list
            auto &sel_list = draw.SelectionList;
            sel_list = {};

            const auto run_sel_pass = [&](auto &&indices_of, auto &&skip) {
                auto batch = sel_list.BeginBatch();
                for (const auto &e : mesh_entities) {
                    if (e.IsExtras || e.IsBoneJoint || skip(e)) continue;
                    const auto &indices = indices_of(e);
                    auto dd = MakeDrawData(e.Buf.Vertices, indices, Buffers.Instances, e.Deform.BoneDeformOffset, e.Deform.ArmatureDeformOffset, e.Deform.MorphDeformOffset, e.Deform.MorphTargetCount);
                    dd.ObjectIdSlot = Buffers.Instances.ObjectIdBuffer.Slot;
                    const auto db = sel_list.Draws.size();
                    if (e.PrimaryEditBufferIndex) AppendDraw(sel_list, batch, indices, e.Mod, dd, *e.PrimaryEditBufferIndex);
                    else AppendDraw(sel_list, batch, indices, e.Mod, dd);
                    PatchMorphWeights(sel_list, db, e.Deform);
                }
                return batch;
            };
            const auto sel_tri = run_sel_pass(
                [](const auto &e) -> const auto & { return e.Buf.FaceIndices; },
                [](const auto &e) { return e.Buf.FaceIndices.Count == 0; }
            );
            const auto sel_line = run_sel_pass(
                [](const auto &e) -> const auto & { return e.Buf.EdgeIndices; },
                [](const auto &e) { return e.Buf.FaceIndices.Count > 0 || e.Buf.EdgeIndices.Count == 0; }
            );
            const auto sel_point = run_sel_pass(
                [](const auto &e) -> const auto & { return e.Buf.VertexIndices; },
                [](const auto &e) { return e.Buf.FaceIndices.Count > 0 || e.Buf.EdgeIndices.Count > 0; }
            );

            DrawBatchInfo sel_bone_sphere;
            if (show_overlays && settings.ShowBones) {
                sel_bone_sphere = sel_list.BeginBatch();
                for (const auto [entity, mesh_buffers, models] : R.view<const BoneJoint, const MeshBuffers, const ModelsBuffer>().each()) {
                    if (mesh_buffers.FaceIndices.Count == 0) continue;
                    auto dd = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, Buffers.Instances);
                    dd.ObjectIdSlot = Buffers.Instances.ObjectIdBuffer.Slot;
                    AppendDraw(sel_list, sel_bone_sphere, mesh_buffers.FaceIndices, models, dd);
                }
            }

            DrawBatchInfo sel_extras;
            if (show_overlays && settings.ShowExtras) {
                AppendExtrasDraw(R, Buffers.Instances, sel_list, sel_extras, [](auto &dd, const auto &instances) {
                    dd.ObjectIdSlot = instances.ObjectIdBuffer.Slot;
                });
            }

            draw.SelectionDraws = {
                {SPT::SelectionFragmentTriangles, sel_tri},
                {SPT::SelectionFragmentLines, sel_line},
                {SPT::SelectionFragmentPoints, sel_point},
                {SPT::SelectionFragmentBoneSphere, sel_bone_sphere},
                {SPT::SelectionObjectExtrasLines, sel_extras},
            };
        }

        // Cache the main draw list state (everything before silhouette).
        draw.MainDrawCount = draw_list.Draws.size();
        draw.MainIndirectCount = draw_list.IndirectCommands.size();
    } else {
        // Silhouette-only: truncate to cached main portion, batch infos retained from last full build.
        draw_list.Draws.resize(draw.MainDrawCount);
        draw_list.IndirectCommands.resize(draw.MainIndirectCount);
    }

    // Silhouette batch (appended after main batches in both paths).
    std::unordered_set<entt::entity> silhouette_instances;
    if (is_edit_mode) {
        for (const auto [e, instance, ri] : R.view<const Instance, const Selected, const RenderInstance>().each()) {
            if (!is_silhouette_eligible(e)) continue;
            if (auto it = primary_edit_instances.find(instance.Entity); it == primary_edit_instances.end() || it->second != e) {
                silhouette_instances.insert(e);
            }
        }
    }

    const bool has_object_silhouette_selection =
        any_of(R.view<const Selected, const Instance, const RenderInstance>().each(), [&](const auto &entry) { return is_silhouette_eligible(std::get<0>(entry)); });
    const bool render_silhouette = (show_overlays && settings.ShowOutlineSelected) && !is_excite_mode &&
        (is_edit_mode ? !silhouette_instances.empty() : has_object_silhouette_selection);

    draw.Silhouette = {};
    if (render_silhouette) {
        draw.Silhouette = draw_list.BeginBatch();
        auto append_silhouette = [&](entt::entity e) {
            if (!is_silhouette_eligible(e)) return;
            const auto mesh_entity = R.get<Instance>(e).Entity;
            const auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
            const auto &models = R.get<ModelsBuffer>(mesh_entity);
            const auto deform = get_deform_slots(mesh_entity);
            auto dd = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.FaceIndices, Buffers.Instances, deform.BoneDeformOffset, deform.ArmatureDeformOffset, deform.MorphDeformOffset, deform.MorphTargetCount);
            dd.ObjectIdSlot = Buffers.Instances.ObjectIdBuffer.Slot;
            const auto draws_before = draw_list.Draws.size();
            AppendDraw(draw_list, draw.Silhouette, mesh_buffers.FaceIndices, models, dd, R.get<RenderInstance>(e).BufferIndex);
            PatchMorphWeights(draw_list, draws_before, deform);
            patch_edit_pending_local_transform(draws_before, mesh_entity);
        };
        if (is_edit_mode) {
            for (const auto e : silhouette_instances) append_silhouette(e);
        } else {
            for (const auto e : R.view<Selected, RenderInstance>()) append_silhouette(e);
        }
    }

    FlushDrawList(R, SceneEntity, Vk.Device, draw_list, Buffers.RenderDraw);
    const auto render_extent = ComputeRenderExtentPx(R.get<const ViewportExtent>(SceneEntity).Value, std::bit_cast<vec2>(ImGui::GetIO().DisplayFramebufferScale));
    const auto &cb = *RenderCommandBuffer;
    cb.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(render_extent.width), float(render_extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, render_extent});
    const uint32_t transform_vertex_state_slot = is_edit_mode ? Meshes.GetVertexStateSlot() : InvalidSlot;
    auto record_draw_batch = [&](const PipelineRenderer &renderer, SPT spt, const DrawBatchInfo &batch) {
        if (batch.DrawCount == 0) return;
        const auto &pipeline = renderer.Bind(cb, spt);
        const MainDrawPushConstants pc{{batch.DrawDataSlotOffset, transform_vertex_state_slot}};
        cb.pushConstants(*pipeline.PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        cb.drawIndexedIndirect(*Buffers.RenderDraw.Indirect, batch.IndirectOffset, batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
    };
    auto record_pbr_batch = [&](const DrawBatchInfo &batch, PbrCompiler::Variant variant) {
        if (batch.DrawCount == 0) return;
        const auto layout = Pipelines.Main.Compiler.Bind(cb, variant);
        const MainDrawPushConstants pc{{batch.DrawDataSlotOffset, transform_vertex_state_slot}};
        cb.pushConstants(layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        cb.drawIndexedIndirect(*Buffers.RenderDraw.Indirect, batch.IndirectOffset, batch.DrawCount, sizeof(vk::DrawIndexedIndirectCommand));
    };
    const auto make_shader_read_barrier = [](vk::AccessFlags src_access, vk::ImageLayout layout, vk::Image image, const vk::ImageSubresourceRange &range) {
        return vk::ImageMemoryBarrier{src_access, vk::AccessFlagBits::eShaderRead, layout, layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, range};
    };
    const auto sync_fragment_shader_reads = [&](vk::PipelineStageFlags src_stages, auto &&barriers) {
        cb.pipelineBarrier(src_stages, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barriers);
    };

    const bool has_silhouette = render_silhouette && draw.Silhouette.DrawCount > 0;
    if (has_silhouette) { // Silhouette depth/object pass
        const auto &silhouette = Pipelines.Silhouette;
        const vk::Rect2D rect{{0, 0}, ToExtent2D(silhouette.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette.Renderer.RenderPass, *silhouette.Resources->Framebuffer, rect, SilhouetteClearValues}, vk::SubpassContents::eInline);
        cb.bindIndexBuffer(*Buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        record_draw_batch(silhouette.Renderer, SPT::SilhouetteDepthObject, draw.Silhouette);
        cb.endRenderPass();

        // Silhouette pass offscreen color writes -> edge pass fragment sampling.
        const std::array silhouette_to_edge_barriers{
            make_shader_read_barrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eShaderReadOnlyOptimal, *silhouette.Resources->OffscreenImage.Image, ColorSubresourceRange),
        };
        sync_fragment_shader_reads(vk::PipelineStageFlagBits::eColorAttachmentOutput, silhouette_to_edge_barriers);

        const auto &silhouette_edge = Pipelines.SilhouetteEdge;
        const vk::Rect2D edge_rect{{0, 0}, ToExtent2D(silhouette_edge.Resources->OffscreenImage.Extent)};
        cb.beginRenderPass({*silhouette_edge.Renderer.RenderPass, *silhouette_edge.Resources->Framebuffer, edge_rect, SilhouetteClearValues}, vk::SubpassContents::eInline);
        const auto &silhouette_edo = silhouette_edge.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepthObject);
        const SilhouetteEdgeDepthObjectPushConstants edge_pc{sel_slots.SilhouetteSampler};
        cb.pushConstants(*silhouette_edo.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(edge_pc), &edge_pc);
        silhouette_edo.RenderQuad(cb);
        cb.endRenderPass();

        // Edge pass depth/color writes -> main pass silhouette sampling.
        const std::array edge_to_main_barriers{
            make_shader_read_barrier(vk::AccessFlagBits::eDepthStencilAttachmentWrite, vk::ImageLayout::eDepthStencilReadOnlyOptimal, *silhouette_edge.Resources->DepthImage.Image, DepthSubresourceRange),
            make_shader_read_barrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eShaderReadOnlyOptimal, *silhouette_edge.Resources->OffscreenImage.Image, ColorSubresourceRange),
        };
        sync_fragment_shader_reads(vk::PipelineStageFlagBits::eLateFragmentTests | vk::PipelineStageFlagBits::eColorAttachmentOutput, edge_to_main_barriers);
    }

    const auto &main = Pipelines.Main;
    const vk::Rect2D main_rect{{0, 0}, ToExtent2D(main.Resources->ColorImage.Extent)};
    const std::vector<vk::ClearValue> main_clear_values{
        {vk::ClearDepthStencilValue{1, 0}},
        {settings.ClearColor},
        {vk::ClearColorValue{std::array<float, 4>{0, 0, 0, 0}}},
    };

    // Real-transmission pre-pass: render Background + FillOpaque into TransmissionImage mip 0.
    // The PBR shader samples this in the main pass at the refracted exit point projected to NDC.
    // The OpaquePrepass spec-constant variant disables framebuffer sampling, so transmission
    // materials drawn here don't recursively read the attachment they're writing into.
    if (real_transmission && main.Transmission) {
        cb.beginRenderPass({*main.Renderer.RenderPass, *main.Transmission->Framebuffer, main_rect, main_clear_values}, vk::SubpassContents::eInline);
        main.Renderer.ShaderPipelines.at(SPT::Background).RenderQuad(cb);
        if (Buffers.IdentityIndexCount > 0 && show_fill) {
            cb.bindIndexBuffer(*Buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
            record_pbr_batch(draw.FillOpaque, PbrCompiler::Variant::OpaquePrepass);
        }
        cb.endRenderPass();

        // Generate mip chain via linear blits. After the render pass, mip 0 is in eShaderReadOnlyOptimal
        // (per attachment finalLayout); we re-transition it for blits, then leave all mips in eShaderReadOnlyOptimal.
        const auto mip_count = main.Transmission->MipCount;
        const auto image = *main.Transmission->Image.Image;
        // mip 0: shaderRO -> transferSrc
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
            {{vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferRead,
              vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal,
              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
              vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}}
        );
        // mips 1..N-1: undefined -> transferDst
        if (mip_count > 1) {
            cb.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
                {{{}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 1, mip_count - 1, 0, 1}}}
            );
        }
        int32_t mip_w = int32_t(main_rect.extent.width), mip_h = int32_t(main_rect.extent.height);
        for (uint32_t mip = 1; mip < mip_count; ++mip) {
            const int32_t next_w = std::max(1, mip_w / 2);
            const int32_t next_h = std::max(1, mip_h / 2);
            cb.blitImage(
                image, vk::ImageLayout::eTransferSrcOptimal,
                image, vk::ImageLayout::eTransferDstOptimal,
                vk::ImageBlit{
                    vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mip - 1, 0, 1},
                    {vk::Offset3D{0, 0, 0}, vk::Offset3D{mip_w, mip_h, 1}},
                    vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mip, 0, 1},
                    {vk::Offset3D{0, 0, 0}, vk::Offset3D{next_w, next_h, 1}},
                },
                vk::Filter::eLinear
            );
            // Promote this mip from transferDst to transferSrc for the next iteration.
            cb.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
                {{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead,
                  vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
                  vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, mip, 1, 0, 1}}}
            );
            mip_w = next_w;
            mip_h = next_h;
        }
        // All mips currently in transferSrc — flip to shaderReadOnly for sampling in the main pass.
        cb.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
            {{vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead,
              vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
              vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, mip_count, 0, 1}}}
        );
    }

    // Main rendering pass
    cb.beginRenderPass({*main.Renderer.RenderPass, *main.Resources->Framebuffer, main_rect, main_clear_values}, vk::SubpassContents::eInline);

    // Background environment (PBR modes only; shader discards when WorldOpacity == 0 or no env slot)
    if (show_rendered) main.Renderer.ShaderPipelines.at(SPT::Background).RenderQuad(cb);

    // Silhouette edge depth (not color! we render it before mesh depth to avoid overwriting closer depths with further ones)
    if (has_silhouette) {
        const auto &silhouette_depth = main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeDepth);
        const uint32_t depth_sampler_index = sel_slots.DepthSampler;
        cb.pushConstants(*silhouette_depth.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(depth_sampler_index), &depth_sampler_index);
        silhouette_depth.RenderQuad(cb);
    }

    if (Buffers.IdentityIndexCount > 0) { // Meshes
        cb.bindIndexBuffer(*Buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        // Solid faces
        if (show_fill) {
            if (show_rendered) {
                record_pbr_batch(draw.FillOpaque, PbrCompiler::Variant::Opaque);
                record_pbr_batch(draw.FillBlend, PbrCompiler::Variant::Blend);
            } else {
                record_draw_batch(main.Renderer, SPT::Fill, draw.FillOpaque);
            }
        }
        // Edit mode edges as triangle quads with self-AA
        record_draw_batch(main.Renderer, SPT::EdgeQuad, draw.EdgeQuad);
        // Wireframe/line mesh edges as GPU lines (LineAA composite handles AA)
        record_draw_batch(main.Renderer, SPT::Line, draw.WireLine);
        // Vertex points (always recorded — batch is empty when nothing qualifies)
        record_draw_batch(main.Renderer, SPT::Point, draw.Point);
        // Object extras (cameras, lights, empties)
        record_draw_batch(main.Renderer, SPT::ObjectExtrasLine, draw.ExtrasLine);
    }

    // Silhouette edge color (rendered ontop of meshes)
    if (has_silhouette) {
        const auto &silhouette_edc = main.Renderer.ShaderPipelines.at(SPT::SilhouetteEdgeColor);
        // In mesh Edit mode, suppress active silhouette (element selection drives active state differently).
        // In armature Edit/Pose mode, the active bone gets the active-color silhouette.
        const auto active_entity = FindActiveEntity(R);
        const auto active_bone = FindActiveBone(R);
        const bool armature_mode = FindArmatureObject(R, active_entity) != entt::null;
        uint32_t active_object_id = 0;
        if (armature_mode && active_bone != entt::null) {
            if (R.all_of<RenderInstance>(active_bone)) {
                active_object_id = R.get<RenderInstance>(active_bone).ObjectId;
            }
        } else if (!is_edit_mode) {
            if (active_entity != entt::null && R.all_of<RenderInstance>(active_entity)) {
                active_object_id = R.get<RenderInstance>(active_entity).ObjectId;
            }
        }
        const SilhouetteEdgeColorPushConstants pc{
            TransformGizmo::IsUsing() && interaction_mode == InteractionMode::Object, sel_slots.ObjectIdSampler, active_object_id
        };
        cb.pushConstants(*silhouette_edc.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        silhouette_edc.RenderQuad(cb);
    }

    if (Buffers.IdentityIndexCount > 0) { // Selection overlays
        cb.bindIndexBuffer(*Buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
        record_draw_batch(main.Renderer, SPT::LineOverlayFaceNormals, draw.OverlayFaceNormals);
        record_draw_batch(main.Renderer, SPT::LineOverlayVertexNormals, draw.OverlayVertexNormals);
    }

    // Grid lines texture (drawn before bone depth clear so grid remains depth-tested against scene meshes)
    if (show_overlays && settings.ShowGrid) {
        // MoltenVK/Metal workaround: the grid shader writes gl_FragDepth (disabling early-z),
        // and late fragment tests don't correctly read unresolved fast-cleared depth on tile-based GPUs.
        // Re-clear depth when no triangle draws with depth write have resolved the fast-clear state.
        if (!has_silhouette && draw.FillOpaque.DrawCount == 0) {
            const vk::ClearAttachment grid_depth_resolve{vk::ImageAspectFlagBits::eDepth, 0, vk::ClearDepthStencilValue{1.f, 0}};
            const vk::ClearRect grid_clear_rect{{{0, 0}, ToExtent2D(main.Resources->ColorImage.Extent)}, 0, 1};
            cb.clearAttachments(grid_depth_resolve, grid_clear_rect);
        }
        main.Renderer.ShaderPipelines.at(SPT::Grid).RenderQuad(cb);
    }

    { // Bone X-ray: clear depth so bones are never occluded by scene meshes (only mutually occlude each other)
        if (draw.BoneFill.DrawCount > 0 || draw.BoneSphereFill.DrawCount > 0) {
            cb.bindIndexBuffer(*Buffers.IdentityIndexBuffer, 0, vk::IndexType::eUint32);
            const vk::ClearAttachment depth_clear{vk::ImageAspectFlagBits::eDepth, 0, vk::ClearDepthStencilValue{1.f, 0}};
            const auto extent = main.Resources->ColorImage.Extent;
            const vk::ClearRect clear_rect{{{0, 0}, ToExtent2D(extent)}, 0, 1};
            cb.clearAttachments(depth_clear, clear_rect);

            // In Object+wireframe mode, show only outlines (no fills).
            // In Edit/Pose+wireframe, fills are semitransparent and write far-plane depth (via shader) so wires are never occluded.
            const bool object_wireframe = is_wireframe_mode && interaction_mode == InteractionMode::Object;
            if (!object_wireframe) {
                record_draw_batch(main.Renderer, SPT::BoneFill, draw.BoneFill);
                record_draw_batch(main.Renderer, SPT::BoneSphereFill, draw.BoneSphereFill);
            }
            // In non-wireframe Object mode, "Outline selected" off suppresses bone wire outlines.
            // In wireframe+Object mode, wires are the only bone visualization so always show them.
            const bool hide_bone_outlines = !is_wireframe_mode && interaction_mode == InteractionMode::Object &&
                (!show_overlays || !settings.ShowOutlineSelected);
            if (!hide_bone_outlines) {
                record_draw_batch(main.Renderer, SPT::BoneWire, draw.BoneWire);
                record_draw_batch(main.Renderer, SPT::BoneSphereWire, draw.BoneSphereWire);
            }
        }
    }

    cb.endRenderPass();

    // The render pass ExternalFragReadDependency should cover this, but MoltenVK needs an explicit barrier
    // to flush the Metal render encoder's color writes before the next encoder samples them.
    {
        const std::array main_to_lineaa_barriers{
            make_shader_read_barrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eShaderReadOnlyOptimal, *main.Resources->ColorImage.Image, ColorSubresourceRange),
            make_shader_read_barrier(vk::AccessFlagBits::eColorAttachmentWrite, vk::ImageLayout::eShaderReadOnlyOptimal, *main.Resources->LineDataImage.Image, ColorSubresourceRange),
        };
        sync_fragment_shader_reads(vk::PipelineStageFlagBits::eColorAttachmentOutput, main_to_lineaa_barriers);
    }

    { // Line AA composite pass: blends anti-aliased lines from LineDataImage onto ColorImage → FinalColorImage
        const vk::ClearValue clear_value{vk::ClearColorValue{std::array<float, 4>{0, 0, 0, 1}}};
        const vk::Rect2D rect{{0, 0}, ToExtent2D(main.Resources->FinalColorImage.Extent)};
        cb.beginRenderPass({*main.LineAARenderPass, *main.Resources->LineAAFramebuffer, rect, clear_value}, vk::SubpassContents::eInline);
        const struct {
            uint32_t ColorSamplerSlot, LineDataSamplerSlot;
        } line_aa_pc{sel_slots.ColorSampler, sel_slots.LineDataSampler};
        cb.pushConstants(*main.LineAAComposite.PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(line_aa_pc), &line_aa_pc);
        main.LineAAComposite.RenderQuad(cb);
        cb.endRenderPass();
    }

    cb.end();
}

void Scene::RenderSelectionPass(vk::Semaphore signal_semaphore) const {
    const Timer timer{"RenderSelectionPass"};

    // Selection draw list is pre-built by RecordRenderCommandBuffer.
    RenderSelectionPassWith(
        false,
        [this](DrawListBuilder &draw_list) -> std::vector<SelectionDrawInfo> {
            const auto &draw = R.get<const SceneDrawState>(SceneEntity);
            draw_list = draw.SelectionList;
            return draw.SelectionDraws;
        },
        signal_semaphore
    );

    R.remove<SelectionStale>(SceneEntity);
}

void Scene::RenderSelectionPassWith(bool render_depth, const SelectionBuildFn &build_fn, vk::Semaphore signal_semaphore, bool render_silhouette) const {
    const Timer timer{"RenderSelectionPassWith"};
    const auto &Vk = R.ctx().get<const SceneVulkanResources>();
    const auto &one_shot = R.get<const SceneOneShotGpu>(SceneEntity);
    const auto &sel_slots = R.get<const SelectionSlots>(SceneEntity);
    auto &Buffers = R.get<SceneBuffers>(SceneEntity);
    auto &Pipelines = R.ctx().get<const ScenePipelines>();
    DrawListBuilder draw_list;
    DrawBatchInfo silhouette_batch{};
    if (render_depth) {
        silhouette_batch = draw_list.BeginBatch();
        if (render_silhouette) AppendSelectedSilhouetteDraws(R, SceneEntity, draw_list, silhouette_batch);
    }
    const auto selection_draws = build_fn(draw_list);

    FlushDrawList(R, SceneEntity, Vk.Device, draw_list, Buffers.SelectionDraw);
    Buffers.SceneViewUBO.Update(as_bytes(Buffers.SelectionDraw.DrawData.Slot), offsetof(SceneViewUBO, DrawDataSlot));

    auto cb = *one_shot.Cb;
    cb.reset({});
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    Buffers.SelectionCounter.Buffer.Write(as_bytes(SelectionCounters{}));

    // Transition head image to general layout and clear.
    const auto &head_image = Pipelines.SelectionFragment.Resources->HeadImage;
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
        vk::ImageMemoryBarrier{{}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, {}, {}, *head_image.Image, ColorSubresourceRange}
    );
    cb.clearColorImage(*head_image.Image, vk::ImageLayout::eGeneral, vk::ClearColorValue{std::array<uint32_t, 4>{InvalidSlot, 0, 0, 0}}, ColorSubresourceRange);
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
        vk::ImageMemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, {}, {}, *head_image.Image, ColorSubresourceRange}
    );

    const auto render_extent = ComputeRenderExtentPx(R.get<const ViewportExtent>(SceneEntity).Value, std::bit_cast<vec2>(ImGui::GetIO().DisplayFramebufferScale));
    cb.setViewport(0, vk::Viewport{0.f, 0.f, float(render_extent.width), float(render_extent.height), 0.f, 1.f});
    cb.setScissor(0, vk::Rect2D{{0, 0}, render_extent});

    if (render_depth) {
        // Render selected meshes to silhouette depth buffer for element occlusion.
        // Open the pass even with no draws — its depth LoadOp::eClear seeds the selection-fragment pass's LoadOp::eLoad.
        const auto &silhouette = Pipelines.Silhouette;
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

    const auto &selection = Pipelines.SelectionFragment;
    const vk::Rect2D rect{{0, 0}, ToExtent2D(Pipelines.Silhouette.Resources->DepthImage.Extent)};
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

void Scene::RunBoxSelectElements(std::span<const ElementRange> ranges, Element element, std::pair<uvec2, uvec2> box_px, bool is_additive) {
    if (ranges.empty()) return;

    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return;

    auto &Buffers = R.get<SceneBuffers>(SceneEntity);
    const Timer timer{"RunBoxSelectElements"};
    const auto element_count = MaxElementBound(ranges);
    if (element_count == 0) return;

    const uint32_t bitset_words = (element_count + 31) / 32;
    if (bitset_words > SceneBuffers::SelectionBitsetWords) return;

    // Restore baseline bitset for additive mode, or clear for non-additive.
    auto *bits = Buffers.SelectionBitset.Data();
    if (is_additive) {
        const auto *baseline = R.try_get<const AdditiveBoxSelectBaseline>(SceneEntity);
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
    RenderElementSelectionPass(R, SceneEntity, ranges, element, true, box_min, box_max, {});
    // After RenderElementSelectionPass (which waits on fence), SelectionBitsetBuffer is populated.
    // Dispatch UpdateSelectionState compute shader to update element state buffers on GPU.
    ApplySelectionStateUpdate(R, SceneEntity, ranges, element);
}

std::optional<uint32_t> Scene::RunSoundVerticesVertexPick(entt::entity instance_entity, uvec2 mouse_px) {
    if (!R.all_of<SoundVertices>(instance_entity)) return {};
    const auto *instance = R.try_get<Instance>(instance_entity);
    if (!instance) return {};
    const auto &Vk = R.ctx().get<const SceneVulkanResources>();
    const auto &one_shot = R.get<const SceneOneShotGpu>(SceneEntity);
    const auto &sel_slots = R.get<const SelectionSlots>(SceneEntity);
    auto &Buffers = R.get<SceneBuffers>(SceneEntity);
    auto &Meshes = R.ctx().get<MeshStore>();
    auto &Pipelines = R.ctx().get<const ScenePipelines>();

    const Timer timer{"RunSoundVerticesVertexPick"};
    const auto mesh_entity = instance->Entity;
    const auto &mesh = R.get<Mesh>(mesh_entity);
    const uint32_t vertex_count = mesh.VertexCount();
    if (vertex_count == 0) return {};

    const auto &mesh_buffers = R.get<MeshBuffers>(mesh_entity);
    const auto &models = R.get<ModelsBuffer>(mesh_entity);
    const auto model_index = R.get<RenderInstance>(instance_entity).BufferIndex;
    RenderSelectionPassWith(true, [&](DrawListBuilder &draw_list) {
        auto batch = draw_list.BeginBatch();
        auto draw = MakeDrawData(mesh_buffers.Vertices, mesh_buffers.VertexIndices, Buffers.Instances);
        draw.VertexCountOrHeadImageSlot = 0;
        draw.ElementStateSlotOffset = {Meshes.GetVertexStateSlot(), mesh_buffers.Vertices.Offset};
        AppendDraw(draw_list, batch, mesh_buffers.VertexIndices, models, draw, model_index);
        return std::vector{SelectionDrawInfo{SPT::SelectionElementVertex, batch}}; }, *one_shot.SelectionReady);
    R.emplace_or_replace<SelectionStale>(SceneEntity);

    return FindNearestPickedElement(
        Buffers, Pipelines.ElementPick, *one_shot.Cb,
        Vk.Queue, *one_shot.Fence, Vk.Device,
        sel_slots.HeadImage, Buffers.SelectionNodeBuffer.Slot, sel_slots.ElementPickCandidates,
        mouse_px, vertex_count, Element::Vertex,
        *one_shot.SelectionReady
    );
}

// Returns unique object-hit entities sorted by (distance, depth, object id).
std::vector<entt::entity> Scene::RunObjectPick(uvec2 mouse_px, uint32_t radius_px) {
    const auto &Vk = R.ctx().get<const SceneVulkanResources>();
    const auto &one_shot = R.get<const SceneOneShotGpu>(SceneEntity);
    const auto &sel_slots = R.get<const SelectionSlots>(SceneEntity);
    auto &Buffers = R.get<SceneBuffers>(SceneEntity);
    auto &Pipelines = R.ctx().get<const ScenePipelines>();
    const uint32_t next_object_id = R.get<const ObjectIdCounter>(SceneEntity).Next;
    if (next_object_id <= 1) return {}; // No objects have been assigned IDs yet
    const uint32_t max_object_id = std::min(next_object_id - 1, SceneBuffers::MaxSelectableObjects);
    if (max_object_id == 0) return {};

    const bool selection_rendered = R.all_of<SelectionStale>(SceneEntity);
    if (selection_rendered) RenderSelectionPass(*one_shot.SelectionReady);

    const Timer timer{"RunObjectPick"};
    // ObjectPickKeyBuffer is persistent across clicks: high 8 bits of each packed key store
    // a per-click epoch tag. We therefore avoid clearing all keys every click and only do a
    // full reset when the 8-bit epoch wraps; stale keys are filtered out by epoch on readback.
    if (ObjectPickEpochTag == 0) {
        ResetObjectPickKeys(Buffers);
        ObjectPickEpochTag = 255;
    }
    const uint32_t epoch_inv = ObjectPickEpochTag--;

    auto cb = *one_shot.Cb;
    RunSelectionCompute(
        cb, Vk.Queue, *one_shot.Fence, Vk.Device, Pipelines.ObjectPick,
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

void Scene::DispatchBoxSelect(uvec2 box_min, uvec2 box_max, uint32_t max_id, vk::Semaphore wait_semaphore) {
    const auto &Vk = R.ctx().get<const SceneVulkanResources>();
    const auto &one_shot = R.get<const SceneOneShotGpu>(SceneEntity);
    const auto &sel_slots = R.get<const SelectionSlots>(SceneEntity);
    auto &Buffers = R.get<SceneBuffers>(SceneEntity);
    auto &Pipelines = R.ctx().get<const ScenePipelines>();
    const uint32_t bitset_words = (max_id + 31) / 32;
    memset(Buffers.SelectionBitset.Data(), 0, bitset_words * sizeof(uint32_t));

    const auto group_counts = glm::max((box_max - box_min + 15u) / 16u, uvec2{1, 1});
    RunSelectionCompute(
        *one_shot.Cb, Vk.Queue, *one_shot.Fence, Vk.Device, Pipelines.BoxSelect,
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

std::vector<entt::entity> Scene::RunBoxSelect(std::pair<uvec2, uvec2> box_px) {
    const auto [box_min, box_max] = box_px;
    if (box_min.x > box_max.x || box_min.y > box_max.y) return {};
    const auto &one_shot = R.get<const SceneOneShotGpu>(SceneEntity);
    auto &Buffers = R.get<SceneBuffers>(SceneEntity);
    const uint32_t next_object_id = R.get<const ObjectIdCounter>(SceneEntity).Next;
    if (next_object_id <= 1) return {}; // No objects have been assigned IDs yet

    const uint32_t max_object_id = std::min(next_object_id - 1, SceneBuffers::MaxSelectableObjects);

    const Timer timer{"RunBoxSelect"};
    const bool selection_rendered = R.all_of<SelectionStale>(SceneEntity);
    if (selection_rendered) RenderSelectionPass(*one_shot.SelectionReady);
    DispatchBoxSelect(box_min, box_max, max_object_id, selection_rendered ? *one_shot.SelectionReady : vk::Semaphore{});

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

void Scene::Render(vk::Fence viewportConsumerFence) {
    auto &dl = *ImGui::GetWindowDrawList();
    dl.ChannelsSetCurrent(0);
    if (SubmitViewport(viewportConsumerFence)) {
        const auto &Vk = R.ctx().get<const SceneVulkanResources>();
        auto &Pipelines = R.ctx().get<const ScenePipelines>();
        ViewportTexture = std::make_unique<mvk::ImGuiTexture>(Vk.Device, *Pipelines.Main.Resources->FinalColorImage.View, vec2{0, 1}, vec2{1, 0});
    }
    if (ViewportTexture) {
        const auto p = ImGui::GetCursorScreenPos();
        const auto extent = R.get<const ViewportExtent>(SceneEntity).Value;
        const auto &t = *ViewportTexture;
        dl.AddImage(ImTextureID(VkDescriptorSet(t.DescriptorSet)), p, p + ImVec2{float(extent.width), float(extent.height)}, std::bit_cast<ImVec2>(t.Uv0), std::bit_cast<ImVec2>(t.Uv1));
    }

    dl.ChannelsSetCurrent(1);
    DrawOverlay();
}

bool Scene::SubmitViewport(vk::Fence viewportConsumerFence) {
    const auto &Vk = R.ctx().get<const SceneVulkanResources>();
    const auto &sel_slots = R.get<const SelectionSlots>(SceneEntity);
    auto &Slots = R.ctx().get<DescriptorSlots>();
    auto &Buffers = R.get<SceneBuffers>(SceneEntity);
    auto &Pipelines = R.ctx().get<ScenePipelines>();
    auto &logical_extent = R.get<ViewportExtent>(SceneEntity).Value;
    const auto content_region = ImGui::GetContentRegionAvail();
    const vk::Extent2D new_logical_extent{
        uint32_t(std::max(content_region.x, 0.0f)),
        uint32_t(std::max(content_region.y, 0.0f))
    };
    const bool extent_changed = logical_extent.width != new_logical_extent.width || logical_extent.height != new_logical_extent.height;
    if (extent_changed) {
        logical_extent = new_logical_extent;
        R.patch<ViewportExtent>(SceneEntity, [](auto &) {});
    }
    const auto render_extent = ComputeRenderExtentPx(logical_extent, std::bit_cast<vec2>(ImGui::GetIO().DisplayFramebufferScale));
    const auto current_render_extent = Pipelines.Main.Resources ? ToExtent2D(Pipelines.Main.Resources->ColorImage.Extent) : vk::Extent2D{};
    const bool render_extent_changed = current_render_extent.width != render_extent.width || current_render_extent.height != render_extent.height;
    if (render_extent_changed && !extent_changed) {
        // Trigger SceneView update (projection, screen pixel scale) when DPI scale changes at fixed logical viewport size.
        R.patch<ViewportExtent>(SceneEntity, [](auto &) {});
    }

    const auto render_request = ProcessComponentEvents(R, SceneEntity);

    if (auto descriptor_updates = Buffers.Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
        const Timer timer{"SubmitViewport->UpdateBufferDescriptorSets"};
        Vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
        Buffers.Ctx.ClearDeferredDescriptorUpdates();
    }
    if (!extent_changed && !render_extent_changed && render_request == RenderRequest::None) return false;

    const Timer timer{"SubmitViewport"};
    if (extent_changed || render_extent_changed) {
        if (viewportConsumerFence) { // Wait for viewport consumer to finish sampling old resources
            std::ignore = Vk.Device.waitForFences(viewportConsumerFence, VK_TRUE, UINT64_MAX);
        }
        Pipelines.SetExtent(render_extent);
        {
            const auto shading = R.get<const SceneSettings>(SceneEntity).ViewportShading;
            const bool is_pbr = shading == ViewportShadingMode::MaterialPreview || shading == ViewportShadingMode::Rendered;
            const bool want_transmission = is_pbr && GetActivePbrLighting(R, SceneEntity, shading).RealTransmission && Pipelines.Main.Compiler.HasFeature(PbrFeature::Transmission);
            Pipelines.Main.EnsureTransmissionResources(render_extent, Vk.Device, Vk.PhysicalDevice, want_transmission);
        }
        Buffers.ResizeSelectionNodeBuffer(render_extent);
        {
            const Timer timer{"SubmitViewport->UpdateSelectionDescriptorSets"};
            const auto head_image_info = vk::DescriptorImageInfo{
                nullptr,
                *Pipelines.SelectionFragment.Resources->HeadImage.View,
                vk::ImageLayout::eGeneral
            };
            const auto selection_counter = Buffers.SelectionCounter.GetDescriptor();
            const auto object_pick_key = Buffers.ObjectPickKeys.GetDescriptor(SceneBuffers::MaxSelectableObjects);
            const auto element_pick_candidates = Buffers.ElementPickCandidates.GetDescriptor(SceneBuffers::ElementPickGroupCount);
            const auto &sil = Pipelines.Silhouette;
            const auto &sil_edge = Pipelines.SilhouetteEdge;
            const auto &main = Pipelines.Main;
            const vk::DescriptorImageInfo silhouette_sampler{*sil.Resources->ImageSampler, *sil.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo object_id_sampler{*sil_edge.Resources->ImageSampler, *sil_edge.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo depth_sampler{*sil_edge.Resources->DepthSampler, *sil_edge.Resources->DepthImage.View, vk::ImageLayout::eDepthStencilReadOnlyOptimal};
            const vk::DescriptorImageInfo color_sampler{*main.Resources->NearestSampler, *main.Resources->ColorImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo line_data_sampler{*main.Resources->NearestSampler, *main.Resources->LineDataImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            // Transmission resources are lazy. When unallocated, point the descriptor at ColorImage so the binding stays valid.
            // The shader's UseRealTransmission flag is 0 in that state, so it's never sampled.
            const vk::DescriptorImageInfo transmission_sampler = main.Transmission ? vk::DescriptorImageInfo{*main.Transmission->Sampler, *main.Transmission->Image.View, vk::ImageLayout::eShaderReadOnlyOptimal} : vk::DescriptorImageInfo{*main.Resources->NearestSampler, *main.Resources->ColorImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const auto selection_bitset = Buffers.SelectionBitset.GetDescriptor(SceneBuffers::SelectionBitsetWords);
            const auto object_pick_seen_bitset = Buffers.ObjectPickSeenBitset.GetDescriptor(SceneBuffers::ObjectPickBitsetWords);
            Vk.Device.updateDescriptorSets(
                {
                    Slots.MakeImageWrite(sel_slots.HeadImage, head_image_info),
                    Slots.MakeBufferWrite({SlotType::Buffer, sel_slots.SelectionCounter}, selection_counter),
                    Slots.MakeBufferWrite({SlotType::Buffer, sel_slots.ObjectPickKey}, object_pick_key),
                    Slots.MakeBufferWrite({SlotType::Buffer, sel_slots.ElementPickCandidates}, element_pick_candidates),
                    Slots.MakeBufferWrite({SlotType::Buffer, sel_slots.ObjectPickSeenBits}, object_pick_seen_bitset),
                    Slots.MakeBufferWrite({SlotType::Buffer, sel_slots.SelectionBitset}, selection_bitset),
                    Slots.MakeSamplerWrite(sel_slots.ObjectIdSampler, object_id_sampler),
                    Slots.MakeSamplerWrite(sel_slots.DepthSampler, depth_sampler),
                    Slots.MakeSamplerWrite(sel_slots.SilhouetteSampler, silhouette_sampler),
                    Slots.MakeSamplerWrite(sel_slots.ColorSampler, color_sampler),
                    Slots.MakeSamplerWrite(sel_slots.LineDataSampler, line_data_sampler),
                    Slots.MakeSamplerWrite(sel_slots.TransmissionSampler, transmission_sampler),
                },
                {}
            );
        }
        if (auto descriptor_updates = Buffers.Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
            Vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
            Buffers.Ctx.ClearDeferredDescriptorUpdates();
        }
    }

#ifdef MVK_FORCE_STAGED_TRANSFERS
    RecordTransferCommandBuffer(R, SceneEntity, *TransferCommandBuffer);
#endif

    if (render_request == RenderRequest::ReRecord || extent_changed || render_extent_changed) {
        RecordRenderCommandBuffer();
    } else if (render_request == RenderRequest::ReRecordSilhouette && R.get<const SceneDrawState>(SceneEntity).MainDrawCount > 0) {
        RecordRenderCommandBuffer(/*silhouette_only=*/true);
    }

    // Always ensure DrawDataSlot points to render draw data before submitting (may have been overwritten by a selection pass).
    Buffers.SceneViewUBO.Update(as_bytes(Buffers.RenderDraw.DrawData.Slot), offsetof(SceneViewUBO, DrawDataSlot));

    vk::SubmitInfo submit;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    const std::array command_buffers{*TransferCommandBuffer, *RenderCommandBuffer};
    submit.setCommandBuffers(command_buffers);
#else
    submit.setCommandBuffers(*RenderCommandBuffer);
#endif
    Vk.Queue.submit(submit, *RenderFence);
    RenderPending = true;
    return extent_changed || render_extent_changed;
}

void Scene::WaitForRender() {
    if (!RenderPending) return;

    const auto &Vk = R.ctx().get<const SceneVulkanResources>();
    const Timer timer{"WaitForRender"};
    WaitFor(*RenderFence, Vk.Device);
    R.get<SceneBuffers>(SceneEntity).Ctx.ReclaimRetiredBuffers();
    RenderPending = false;
}
