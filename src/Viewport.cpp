#include "Viewport.h"
#include "AnimationTimeline.h"
#include "Armature.h"
#include "Changes.h"
#include "Defaults.h"
#include "DrawState.h"
#include "EntityDestroyTracker.h"
#include "ExtrasComponents.h"
#include "FrameState.h"
#include "GizmoInteraction.h"
#include "InteractionComponents.h"
#include "MaterialComponents.h"
#include "ObjectOps.h"
#include "OneShotGpu.h"
#include "Paths.h"
#include "Pipelines.h"
#include "ProcessEvents.h"
#include "Reactive.h"
#include "SceneGraph.h"
#include "SelectionComponents.h"
#include "SoundVertices.h"
#include "Stores.h"
#include "SvgResource.h"
#include "Textures.h"
#include "Timer.h"
#include "TransformUtils.h"
#include "VideoRecorder.h"
#include "ViewportDisplay.h"
#include "ViewportEvents.h"
#include "ViewportInteractionState.h"
#include "ViewportOps.h"
#include "ViewportRenderGpu.h"
#include "ViewportUi.h"
#include "VkFenceWait.h"
#include "WorldTransform.h"
#include "mesh/Mesh.h"
#include "physics/PhysicsTypes.h"
#include "physics/PhysicsWorld.h"

#include "imgui.h"

#include "AxisColors.h" // Must be after imgui.h

#include "GpuBuffers.h"

using std::ranges::find;

namespace {
constexpr vk::Extent2D ToExtent2D(vk::Extent3D extent) { return {extent.width, extent.height}; }

// Per-viewport render command buffers, fence, and the ImGui texture handle for the final color image.
// Command buffers are allocated from OneShotGpu::Pool; DeinitViewport must remove this component
// before OneShotGpu so the buffers are freed before their owning pool.
struct ViewportRenderResources {
    vk::UniqueCommandBuffer RenderCommandBuffer;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    vk::UniqueCommandBuffer TransferCommandBuffer;
#endif
    vk::UniqueFence RenderFence;
    std::unique_ptr<mvk::ImGuiTexture> ViewportTexture;
};

// Present on the viewport entity iff recording is active.
struct VideoRecording {
    std::unique_ptr<VideoRecorder> Recorder;
    std::pair<vk::Offset3D, vk::Extent2D> Region; // Locked at StartRecording.
};

void ResetObjectPickKeys(GpuBuffers &buffers) {
    std::fill_n(buffers.ObjectPickKeys.Data(), GpuBuffers::MaxSelectableObjects, std::numeric_limits<uint32_t>::max());
}

std::pair<vk::Offset3D, vk::Extent2D> GetCaptureRegion(const entt::registry &r) {
    auto &pipelines = r.ctx().get<const Pipelines>();
    const auto full = ToExtent2D(pipelines.Main.Resources->FinalColorImage.Extent);
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

bool SubmitViewport(entt::registry &r, entt::entity viewport, vk::Fence viewport_consumer_fence) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    const auto &sel_slots = r.get<const SelectionSlots>(viewport);
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &pipelines = r.ctx().get<Pipelines>();
    auto &resources = r.get<ViewportRenderResources>(viewport);
    auto &frame = r.get<FrameState>(viewport);
    auto &logical_extent = r.get<ViewportExtent>(viewport).Value;
    const auto content_region = ImGui::GetContentRegionAvail();
    const vk::Extent2D new_logical_extent{
        uint32_t(std::max(content_region.x, 0.0f)),
        uint32_t(std::max(content_region.y, 0.0f))
    };
    const bool extent_changed = logical_extent.width != new_logical_extent.width || logical_extent.height != new_logical_extent.height;
    if (extent_changed) {
        logical_extent = new_logical_extent;
        r.patch<ViewportExtent>(viewport, [](auto &) {});
    }
    const auto render_extent = ComputeRenderExtentPx(logical_extent, std::bit_cast<vec2>(ImGui::GetIO().DisplayFramebufferScale));
    const auto current_render_extent = pipelines.Main.Resources ? ToExtent2D(pipelines.Main.Resources->ColorImage.Extent) : vk::Extent2D{};
    const bool render_extent_changed = current_render_extent.width != render_extent.width || current_render_extent.height != render_extent.height;
    if (render_extent_changed && !extent_changed) {
        // Trigger SceneView update (projection, screen pixel scale) when DPI scale changes at fixed logical viewport size.
        r.patch<ViewportExtent>(viewport, [](auto &) {});
    }

    const auto render_request = ProcessComponentEvents(r, viewport);

    if (auto descriptor_updates = buffers.Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
        const Timer timer{"SubmitViewport->UpdateBufferDescriptorSets"};
        vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
        buffers.Ctx.ClearDeferredDescriptorUpdates();
    }
    if (!extent_changed && !render_extent_changed && render_request == RenderRequest::None) return false;

    const Timer timer{"SubmitViewport"};
    if (extent_changed || render_extent_changed) {
        if (viewport_consumer_fence) { // Wait for viewport consumer to finish sampling old resources
            std::ignore = vk.Device.waitForFences(viewport_consumer_fence, VK_TRUE, UINT64_MAX);
        }
        pipelines.SetExtent(render_extent);
        {
            const auto shading = r.get<const ViewportDisplay>(viewport).ViewportShading;
            const bool is_pbr = shading == ViewportShadingMode::MaterialPreview || shading == ViewportShadingMode::Rendered;
            const bool want_transmission = is_pbr && GetActivePbrLighting(r, viewport, shading).RealTransmission && pipelines.Main.Compiler.HasFeature(PbrFeature::Transmission);
            pipelines.Main.EnsureTransmissionResources(render_extent, vk.Device, vk.PhysicalDevice, want_transmission);
        }
        buffers.ResizeSelectionNodeBuffer(render_extent);
        {
            const Timer timer{"SubmitViewport->UpdateSelectionDescriptorSets"};
            const auto head_image_info = vk::DescriptorImageInfo{
                nullptr,
                *pipelines.SelectionFragment.Resources->HeadImage.View,
                vk::ImageLayout::eGeneral
            };
            const auto selection_counter = buffers.SelectionCounter.GetDescriptor();
            const auto object_pick_key = buffers.ObjectPickKeys.GetDescriptor(GpuBuffers::MaxSelectableObjects);
            const auto element_pick_candidates = buffers.ElementPickCandidates.GetDescriptor(GpuBuffers::ElementPickGroupCount);
            const auto &sil = pipelines.Silhouette;
            const auto &sil_edge = pipelines.SilhouetteEdge;
            const auto &main = pipelines.Main;
            const vk::DescriptorImageInfo silhouette_sampler{*sil.Resources->ImageSampler, *sil.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo object_id_sampler{*sil_edge.Resources->ImageSampler, *sil_edge.Resources->OffscreenImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo depth_sampler{*sil_edge.Resources->DepthSampler, *sil_edge.Resources->DepthImage.View, vk::ImageLayout::eDepthStencilReadOnlyOptimal};
            const vk::DescriptorImageInfo color_sampler{*main.Resources->NearestSampler, *main.Resources->ColorImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo line_data_sampler{*main.Resources->NearestSampler, *main.Resources->LineDataImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            // Transmission resources are lazy. When unallocated, point the descriptor at ColorImage so the binding stays valid.
            // The shader's UseRealTransmission flag is 0 in that state, so it's never sampled.
            const vk::DescriptorImageInfo transmission_sampler = main.Transmission ? vk::DescriptorImageInfo{*main.Transmission->Sampler, *main.Transmission->Image.View, vk::ImageLayout::eShaderReadOnlyOptimal} : vk::DescriptorImageInfo{*main.Resources->NearestSampler, *main.Resources->ColorImage.View, vk::ImageLayout::eShaderReadOnlyOptimal};
            const auto selection_bitset = buffers.SelectionBitset.GetDescriptor(GpuBuffers::SelectionBitsetWords);
            const auto object_pick_seen_bitset = buffers.ObjectPickSeenBitset.GetDescriptor(GpuBuffers::ObjectPickBitsetWords);
            vk.Device.updateDescriptorSets(
                {
                    slots.MakeImageWrite(sel_slots.HeadImage, head_image_info),
                    slots.MakeBufferWrite({SlotType::Buffer, sel_slots.SelectionCounter}, selection_counter),
                    slots.MakeBufferWrite({SlotType::Buffer, sel_slots.ObjectPickKey}, object_pick_key),
                    slots.MakeBufferWrite({SlotType::Buffer, sel_slots.ElementPickCandidates}, element_pick_candidates),
                    slots.MakeBufferWrite({SlotType::Buffer, sel_slots.ObjectPickSeenBits}, object_pick_seen_bitset),
                    slots.MakeBufferWrite({SlotType::Buffer, sel_slots.SelectionBitset}, selection_bitset),
                    slots.MakeSamplerWrite(sel_slots.ObjectIdSampler, object_id_sampler),
                    slots.MakeSamplerWrite(sel_slots.DepthSampler, depth_sampler),
                    slots.MakeSamplerWrite(sel_slots.SilhouetteSampler, silhouette_sampler),
                    slots.MakeSamplerWrite(sel_slots.ColorSampler, color_sampler),
                    slots.MakeSamplerWrite(sel_slots.LineDataSampler, line_data_sampler),
                    slots.MakeSamplerWrite(sel_slots.TransmissionSampler, transmission_sampler),
                },
                {}
            );
        }
        if (auto descriptor_updates = buffers.Ctx.GetDeferredDescriptorUpdates(); !descriptor_updates.empty()) {
            vk.Device.updateDescriptorSets(std::move(descriptor_updates), {});
            buffers.Ctx.ClearDeferredDescriptorUpdates();
        }
    }

#ifdef MVK_FORCE_STAGED_TRANSFERS
    RecordTransferCommandBuffer(r, viewport, *resources.TransferCommandBuffer);
#endif

    if (render_request == RenderRequest::ReRecord || extent_changed || render_extent_changed) {
        RecordRenderCommandBuffer(r, viewport, *resources.RenderCommandBuffer);
    } else if (render_request == RenderRequest::ReRecordSilhouette && r.get<const DrawState>(viewport).MainDrawCount > 0) {
        RecordRenderCommandBuffer(r, viewport, *resources.RenderCommandBuffer, /*silhouette_only=*/true);
    }

    // Always ensure DrawDataSlot points to render draw data before submitting (may have been overwritten by a selection pass).
    buffers.SceneViewUBO.Update(as_bytes(buffers.RenderDraw.DrawData.Slot), offsetof(SceneViewUBO, DrawDataSlot));

    vk::SubmitInfo submit;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    const std::array command_buffers{*resources.TransferCommandBuffer, *resources.RenderCommandBuffer};
    submit.setCommandBuffers(command_buffers);
#else
    submit.setCommandBuffers(*resources.RenderCommandBuffer);
#endif
    vk.Queue.submit(submit, *resources.RenderFence);
    frame.RenderPending = true;
    return extent_changed || render_extent_changed;
}
} // namespace

entt::entity InitViewport(entt::registry &r, VulkanResources vc) {
    InitStoreCtx(r, vc);
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &pipelines = r.ctx().emplace<Pipelines>(vc.Device, vc.PhysicalDevice, slots.GetSetLayout(), slots.GetSet());
    auto &physics = physics::Init(r);
    // Reactive storage subscriptions for deferred once-per-frame processing
    track<changes::TimelineRange>(r).on<TimelineRange>(On::Update);
    track<changes::Selected>(r).on<Selected>(On::Create | On::Destroy);
    track<changes::ActiveInstance>(r).on<Active>(On::Create | On::Destroy);
    track<changes::BoneSelection>(r).on<BoneSelection>(On::Create | On::Update | On::Destroy).on<BoneActive>(On::Create | On::Destroy);
    track<changes::Rerecord>(r)
        .on<RenderInstance>(On::Create | On::Destroy)
        .on<Active>(On::Create | On::Destroy)
        .on<StartTransform>(On::Create | On::Destroy)
        .on<EditMode>(On::Create | On::Update)
        .on<SmoothShading>(On::Create | On::Destroy);
    track<changes::MeshActiveElement>(r).on<MeshActiveElement>(On::Create | On::Update);
    track<changes::MeshGeometry>(r).on<MeshGeometryDirty>(On::Create);
    track<changes::MeshMaterial>(r).on<MeshMaterialAssignment>(On::Create | On::Update);
    track<changes::SoundVertices>(r).on<SoundVertices>(On::Create | On::Destroy);
    track<changes::SoundVerticesUpdated>(r).on<SoundVertices>(On::Update);
    track<changes::VertexForce>(r).on<VertexForce>(On::Create | On::Destroy);
    track<changes::ModelsBuffer>(r).on<ModelsBuffer>(On::Update);
    track<changes::NewBufferEntity>(r).on<MeshBuffers>(On::Create);
    track<changes::RenderInstanceCreated>(r).on<RenderInstance>(On::Create);
    track<changes::ViewportDisplay>(r).on<ViewportDisplay>(On::Create | On::Update);
    track<changes::InteractionMode>(r).on<Interaction>(On::Create | On::Update);
    track<changes::Submit>(r).on<SubmitDirty>(On::Create);
    track<changes::ViewportTheme>(r).on<ViewportTheme>(On::Create | On::Update);
    track<changes::Materials>(r).on<MaterialDirty>(On::Create | On::Update);
    track<changes::ActiveMaterialVariant>(r).on<MaterialVariants>(On::Create | On::Update);
    track<changes::PbrSpecialization>(r)
        .on<PbrMeshFeatures>(On::Create | On::Update | On::Destroy)
        .on<MaterialPreviewLighting>(On::Create | On::Update)
        .on<RenderedLighting>(On::Create | On::Update);
    track<changes::SceneView>(r)
        .on<ViewCamera>(On::Create | On::Update)
        .on<MaterialPreviewLighting>(On::Create | On::Update)
        .on<RenderedLighting>(On::Create | On::Update)
        .on<LightIndex>(On::Create | On::Destroy)
        .on<ViewportExtent>(On::Create | On::Update)
        .on<EditMode>(On::Create | On::Update);
    track<changes::CameraLens>(r).on<Camera>(On::Create | On::Update).on<LookingThrough>(On::Create | On::Destroy);
    track<changes::Rotation>(r).on<Transform>(On::Create | On::Update);
    track<changes::WorldTransform>(r).on<WorldTransform>(On::Create | On::Update);
    track<changes::TransformPending>(r).on<PendingTransform>(On::Create | On::Update);
    track<changes::TransformEnd>(r).on<StartTransform>(On::Destroy);
    track<changes::TransformDirty>(r)
        .on<Transform>(On::Create | On::Update)
        .on<SceneNode>(On::Create | On::Update)
        .on<BoneDisplayScale>(On::Update);
    r.ctx().emplace<EntityDestroyTracker>().Bind(r);

    r.on_destroy<Name>().connect<&OnDestroyName>();
    r.on_construct<RenderInstance>().connect<&AssignRenderInstanceObjectId>();
    r.on_destroy<RenderInstance>().connect<&EmitPendingHideOnRenderInstanceDestroy>();
    // BoneConstraints edits change the resolved local Transform; poke it to drive the WorldTransform recompute.
    r.on_update<BoneConstraints>().connect<+[](entt::registry &r, entt::entity e) {
        r.patch<Transform>(e, [](auto &) {});
    }>();

    const auto viewport = WireRegistry(r);
    auto &buffers = r.ctx().get<GpuBuffers>();
    r.emplace<ViewportDisplay>(viewport);
    r.emplace<Interaction>(viewport);
    r.emplace<EditMode>(viewport);
    r.emplace<ViewportTheme>(viewport, Defaults::ViewportTheme);
    r.emplace<colors::AxesArray>(viewport, colors::MakeAxes(Defaults::ViewportTheme.AxisColors));
    r.emplace<ViewCamera>(viewport, Defaults::ViewCamera);
    r.emplace<MaterialPreviewLighting>(viewport, false, false, 1.f, 0.f);
    r.emplace<RenderedLighting>(viewport, true, true, 1.f, 0.f);
    r.emplace<ViewportExtent>(viewport);
    r.emplace<PlaybackFrame>(viewport);
    r.emplace<LastEvaluatedFrame>(viewport);
    r.emplace<EnabledInteractionModes>(viewport);
    r.emplace<SelectionBitsetRef>(viewport, std::span<uint32_t>{buffers.SelectionBitset.Data(), GpuBuffers::SelectionBitsetWords});
    r.emplace<SelectionSlots>(viewport, slots);
    r.emplace<DrawState>(viewport);
    r.emplace<FrameState>(viewport);
    const auto &one_shot = r.ctx().emplace<OneShotGpu>(MakeOneShotGpu(vc.Device, vc.QueueFamily));
    r.ctx().emplace<ColliderShapeBuffers>();
    r.emplace<ViewportRenderResources>(viewport, ViewportRenderResources{
                                                     .RenderCommandBuffer = std::move(vc.Device.allocateCommandBuffersUnique({*one_shot.Pool, vk::CommandBufferLevel::ePrimary, 1}).front()),
#ifdef MVK_FORCE_STAGED_TRANSFERS
                                                     .TransferCommandBuffer = std::move(vc.Device.allocateCommandBuffersUnique({*one_shot.Pool, vk::CommandBufferLevel::ePrimary, 1}).front()),
#endif
                                                     .RenderFence = vc.Device.createFenceUnique({}),
                                                     .ViewportTexture = {},
                                                 });
    r.emplace<SelectionXRay>(viewport);
    r.emplace<OrbitToActive>(viewport);
    r.emplace<BoxSelectState>(viewport);
    r.emplace<TransformGizmoState>(viewport);
    r.emplace<GizmoInteraction>(viewport);
    r.emplace<AnimationTimelineView>(viewport);
    r.emplace<SelectionStale>(viewport); // Initial state: fragments need rendering on first selection use.
    physics.ApplySimulationSettings(r.emplace<PhysicsSimulationSettings>(viewport));

    buffers.WorkspaceLightsUBO.Update(as_bytes(Defaults::WorkspaceLights));
    ResetObjectPickKeys(buffers);

    auto init_batch = BeginTextureUploadBatch(vc.Device, *one_shot.Pool, buffers.Ctx);
    auto &environments = r.ctx().get<EnvironmentStore>();
    const auto images_dir = Paths::Res() / "images";
    environments.BrdfLut = CreateDefaultLutTexture(vc, init_batch, slots, images_dir / "lut_ggx.png", "DefaultGGXBRDFLUT");
    environments.SheenELut = CreateDefaultLutTexture(vc, init_batch, slots, images_dir / "lut_sheen_E.png", "DefaultSheenELUT");
    environments.CharlieLut = CreateDefaultLutTexture(vc, init_batch, slots, images_dir / "lut_charlie.png", "DefaultCharlieLUT");
    // Blender's default world background color (linear RGB) - flat ambient-only IBL when no scene world is provided.
    environments.EmptySceneWorld = BuildFlatColorEnvironment(vc, init_batch, slots, vec3{0.05f}, "EmptySceneWorld");
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
    SetStudioEnvironment(r, environments.ActiveHdriIndex);
    environments.SceneWorld = {.Ibl = MakeIblSamplers(environments.EmptySceneWorld, environments), .Name = environments.EmptySceneWorld.Name};

    pipelines.CompileShaders();

    return viewport;
}

void DeinitViewport(entt::registry &r, entt::entity viewport) {
    // Free command buffers before their owning pool (OneShotGpu, in ctx) is torn down.
    if (r.valid(viewport)) {
        r.remove<ViewportRenderResources>(viewport);
        r.remove<VideoRecording>(viewport);
    }
    r.clear<Mesh>();
    r.ctx().erase<std::vector<ComponentEventHandler>>();
    r.ctx().erase<EntityDestroyTracker>();
    physics::Deinit(r);
    r.ctx().erase<Pipelines>();
    // Destroy the viewport entity (frees ViewportIcons/FaustDSP SVG images, which are VMA allocations
    // in GpuBuffers.Ctx) before TearDownStoreCtx erases GpuBuffers and while the device is alive.
    if (r.valid(viewport)) r.destroy(viewport);
    TearDownStoreCtx(r);
}

void MakeSvgResource(entt::registry &r, std::unique_ptr<SvgResource> &svg, std::filesystem::path path) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    const auto &one_shot = r.ctx().get<const OneShotGpu>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto svg_batch = BeginTextureUploadBatch(vk.Device, *one_shot.Pool, buffers.Ctx);
    const auto RenderBitmap = [&vk, &svg_batch](std::span<const std::byte> data, uint32_t width, uint32_t height) {
        return RenderBitmapToImage(vk, svg_batch, data, width, height, Format::Color, ColorSubresourceRange);
    };
    svg = std::make_unique<SvgResource>(vk.Device, RenderBitmap, std::move(path));
    SubmitTextureUploadBatch(svg_batch, vk.Queue, *one_shot.Fence, vk.Device);
}

void StartRecording(entt::registry &r, entt::entity viewport, std::filesystem::path path, int fps) {
    StopRecording(r, viewport);
    auto &pipelines = r.ctx().get<const Pipelines>();
    if (!pipelines.Main.Resources) {
        std::println(stderr, "StartRecording: render resources not ready");
        return;
    }
    const auto region = GetCaptureRegion(r);
    const auto &vk = r.ctx().get<const VulkanResources>();
    r.emplace<VideoRecording>(viewport, VideoRecording{
                                            .Recorder = std::make_unique<VideoRecorder>(vk, std::move(path), region.first, region.second, fps),
                                            .Region = region,
                                        });
}

void StopRecording(entt::registry &r, entt::entity viewport) {
    r.remove<VideoRecording>(viewport);
}

bool IsRecording(const entt::registry &r, entt::entity viewport) {
    const auto *rec = r.try_get<VideoRecording>(viewport);
    return rec && rec->Recorder && rec->Recorder->IsActive();
}

uint64_t CapturedFrameCount(const entt::registry &r, entt::entity viewport) {
    const auto *rec = r.try_get<VideoRecording>(viewport);
    return rec && rec->Recorder ? rec->Recorder->CapturedFrameCount() : 0;
}

void CaptureRecordFrame(entt::registry &r, entt::entity viewport) {
    auto &pipelines = r.ctx().get<const Pipelines>();
    auto *rec = r.try_get<VideoRecording>(viewport);
    if (!rec || !rec->Recorder || !rec->Recorder->IsActive() || !pipelines.Main.Resources) return;
    if (GetCaptureRegion(r) != rec->Region) {
        std::println(stderr, "Viewport: capture region changed; stopping recording.");
        StopRecording(r, viewport);
        return;
    }
    rec->Recorder->CaptureFrame(*pipelines.Main.Resources->FinalColorImage.Image);
}

std::string DebugBufferHeapUsage(const entt::registry &r) {
    return r.ctx().get<const GpuBuffers>().Ctx.DebugHeapUsage();
}

void RenderViewport(entt::registry &r, entt::entity viewport, vk::Fence viewport_consumer_fence) {
    auto &dl = *ImGui::GetWindowDrawList();
    dl.ChannelsSetCurrent(0);
    if (SubmitViewport(r, viewport, viewport_consumer_fence)) {
        const auto &vk = r.ctx().get<const VulkanResources>();
        auto &pipelines = r.ctx().get<const Pipelines>();
        r.get<ViewportRenderResources>(viewport).ViewportTexture =
            std::make_unique<mvk::ImGuiTexture>(vk.Device, *pipelines.Main.Resources->FinalColorImage.View, vec2{0, 1}, vec2{1, 0});
    }
    if (const auto &t_ptr = r.get<const ViewportRenderResources>(viewport).ViewportTexture) {
        const auto p = ImGui::GetCursorScreenPos();
        const auto extent = r.get<const ViewportExtent>(viewport).Value;
        const auto &t = *t_ptr;
        dl.AddImage(ImTextureID(VkDescriptorSet(t.DescriptorSet)), p, p + ImVec2{float(extent.width), float(extent.height)}, std::bit_cast<ImVec2>(t.Uv0), std::bit_cast<ImVec2>(t.Uv1));
    }

    dl.ChannelsSetCurrent(1);
    DrawOverlay(r, viewport, r.get<FrameState>(viewport));
}

void WaitForRender(entt::registry &r, entt::entity viewport) {
    auto &frame = r.get<FrameState>(viewport);
    if (!frame.RenderPending) return;

    const auto &vk = r.ctx().get<const VulkanResources>();
    const Timer timer{"WaitForRender"};
    WaitFor(*r.get<const ViewportRenderResources>(viewport).RenderFence, vk.Device);
    r.ctx().get<GpuBuffers>().Ctx.ReclaimRetiredBuffers();
    frame.RenderPending = false;
}
