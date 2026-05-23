#include "Scene.h"
#include "AnimationTimeline.h"
#include "Armature.h"
#include "Bindless.h"
#include "EntityDestroyTracker.h"
#include "MeshComponents.h"
#include "Paths.h"
#include "Reactive.h"
#include "SceneApply.h"
#include "SceneChanges.h"
#include "SceneDefaults.h"
#include "SceneDrawState.h"
#include "ScenePipelines.h"
#include "SceneProcessEvents.h"
#include "SceneRenderGpu.h"
#include "SceneSelection.h"
#include "SceneStores.h"
#include "SceneTextures.h"
#include "SceneTree.h"
#include "SceneUi.h"
#include "SoundVertices.h"
#include "SvgResource.h"
#include "Timer.h"
#include "VideoRecorder.h"
#include "VkFenceWait.h"
#include "gltf/GltfScene.h"
#include "mesh/Mesh.h"
#include "physics/PhysicsWorld.h"

#include "imgui.h"

#include "AxisColors.h" // Must be after imgui.h

using std::ranges::find;

namespace {
constexpr vk::Extent2D ToExtent2D(vk::Extent3D extent) { return {extent.width, extent.height}; }
using namespace he;
} // namespace

#include "SceneBuffers.h"

void ResetObjectPickKeys(SceneBuffers &buffers) {
    std::fill_n(buffers.ObjectPickKeys.Data(), SceneBuffers::MaxSelectableObjects, std::numeric_limits<uint32_t>::max());
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
    DrawOverlay(R, SceneEntity, Frame);
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
        RecordRenderCommandBuffer(R, SceneEntity, *RenderCommandBuffer);
    } else if (render_request == RenderRequest::ReRecordSilhouette && R.get<const SceneDrawState>(SceneEntity).MainDrawCount > 0) {
        RecordRenderCommandBuffer(R, SceneEntity, *RenderCommandBuffer, /*silhouette_only=*/true);
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
    Frame.RenderPending = true;
    return extent_changed || render_extent_changed;
}

void Scene::WaitForRender() {
    if (!Frame.RenderPending) return;

    const auto &Vk = R.ctx().get<const SceneVulkanResources>();
    const Timer timer{"WaitForRender"};
    WaitFor(*RenderFence, Vk.Device);
    R.get<SceneBuffers>(SceneEntity).Ctx.ReclaimRetiredBuffers();
    Frame.RenderPending = false;
}
