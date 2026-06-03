#include "viewport/Viewport.h"
#include "Camera.h"
#include "Changes.h"
#include "Paths.h"
#include "ProcessEvents.h"
#include "Reactive.h"
#include "Stores.h"
#include "Timer.h"
#include "VideoRecorder.h"
#include "animation/AnimationTimeline.h"
#include "armature/ArmatureComponents.h"
#include "armature/BoneConstraint.h"
#include "audio/AudioSystem.h"
#include "audio/FaustDSP.h"
#include "audio/SoundVertices.h"
#include "gizmo/GizmoInteraction.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "object/ExtrasComponents.h"
#include "object/ObjectComponents.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsSystem.h"
#include "physics/PhysicsTypes.h"
#include "render/DrawState.h"
#include "render/MaterialComponents.h"
#include "render/OneShotGpu.h"
#include "render/Pipelines.h"
#include "render/Textures.h"
#include "render/VkFenceWait.h"
#include "scene/Defaults.h"
#include "scene/EntityDestroyTracker.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "viewport/FrameState.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportConsumerFence.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportIcons.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"
#include "viewport/ViewportRenderGpu.h"
#include "viewport/ViewportUi.h"

#include "imgui.h"

#include "render/GpuBuffers.h"
#include "render/LightComponents.h"

#include <cassert>

using std::ranges::find, std::ranges::to;

namespace {
constexpr vk::Extent2D ToExtent2D(vk::Extent3D extent) { return {extent.width, extent.height}; }

// Per-viewport render command buffers, fence, and the ImGui texture handle for the final color image.
// Command buffers are allocated from OneShotGpu::Pool; DeinitViewport must erase this context
// singleton before OneShotGpu so the buffers are freed before their owning pool.
struct ViewportRenderResources {
    vk::UniqueCommandBuffer RenderCommandBuffer;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    vk::UniqueCommandBuffer TransferCommandBuffer;
#endif
    vk::UniqueFence RenderFence;
    std::unique_ptr<mvk::ImGuiTexture> ViewportTexture;
    vk::ImageView ViewportTextureView{}; // The FinalColorImage view ViewportTexture was built from; recreate when it changes.
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

// Submit the recorded command buffer. RenderFence must be unsignaled (reset by the prior WaitForRender).
void SubmitRecordedFrame(entt::registry &r) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &resources = r.ctx().get<ViewportRenderResources>();
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
    r.ctx().get<FrameState>().RenderPending = true;
}

// Re-record the command buffer for `render_request`. `force_full` records the full buffer even when the request wouldn't.
void RecordViewportFrame(entt::registry &r, entt::entity viewport, RenderRequest render_request, bool force_full = false) {
    auto &resources = r.ctx().get<ViewportRenderResources>();
    if (render_request == RenderRequest::ReRecord || force_full) {
        RecordRenderCommandBuffer(r, viewport, *resources.RenderCommandBuffer);
    } else if (render_request == RenderRequest::ReRecordSilhouette && r.ctx().get<const DrawState>().MainDrawCount > 0) {
        RecordRenderCommandBuffer(r, viewport, *resources.RenderCommandBuffer, /*silhouette_only=*/true);
    }
}

void SubmitViewport(entt::registry &r, entt::entity viewport) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    auto &buffers = r.ctx().get<GpuBuffers>();

    // A resize (a ProcessComponentEvents side effect) always yields RenderRequest::ReRecord, so it needs no
    // separate detection here.
    const auto render_request = ProcessComponentEvents(r, viewport);
    buffers.Ctx.FlushDeferredDescriptorUpdates(vk.Device);
    if (render_request == RenderRequest::None) return;
    // Recording a zero-size viewport is a Vulkan error; skip until it has a non-zero extent.
    const auto extent = r.ctx().get<const Pipelines>().BuiltColorExtent();
    if (extent.width == 0 || extent.height == 0) return;

    const Timer timer{"SubmitViewport"};
#ifdef MVK_FORCE_STAGED_TRANSFERS
    RecordTransferCommandBuffer(r, viewport, *r.ctx().get<ViewportRenderResources>().TransferCommandBuffer);
#endif

    RecordViewportFrame(r, viewport, render_request);
    SubmitRecordedFrame(r);
}

// Drain component events, resize as needed, and record the frame. Returns false (skipping the record) when
// the viewport has no non-zero extent yet. `force_full` records the full buffer even when the request wouldn't.
bool AdvanceAndRecord(entt::registry &r, entt::entity viewport, bool force_full) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    const auto render_request = ProcessComponentEvents(r, viewport);
    buffers.Ctx.FlushDeferredDescriptorUpdates(vk.Device);
    const auto extent = r.ctx().get<const Pipelines>().BuiltColorExtent();
    if (extent.width == 0 || extent.height == 0) return false; // Skip recording a zero-size viewport (no extent set yet).
    RecordViewportFrame(r, viewport, render_request, force_full);
    return true;
}
} // namespace

void SetStudioEnvironment(entt::registry &r, uint32_t index) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    const auto &one_shot = r.ctx().get<const OneShotGpu>();
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    auto &hdri = environments.Hdris[index];
    if (!hdri.Prefiltered) {
        hdri.Prefiltered = CreateIblFromHdri(vk, slots, pipelines.IblPrefilter, hdri.Path, hdri.Name, *one_shot.Pool, *one_shot.Fence, buffers.Ctx);
    }
    const auto &pre = *hdri.Prefiltered;
    environments.ActiveHdriIndex = index;
    environments.StudioWorld = {.Ibl = MakeIblSamplers(pre, environments), .Name = hdri.Name};
}

entt::entity InitEngine(entt::registry &r, VulkanResources vc, CreateSvgResource create_svg) {
    InitStoreCtx(r, vc);
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &pipelines = r.ctx().emplace<Pipelines>(vc.Device, vc.PhysicalDevice, slots.GetSetLayout(), slots.GetSet());
    physics::Init(r);
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
    track<changes::NewBufferEntity>(r).on<MeshBuffers>(On::Create);
    track<changes::RenderInstanceCreated>(r).on<RenderInstance>(On::Create);
    track<changes::ViewportDisplay>(r).on<ViewportDisplay>(On::Create | On::Update);
    track<changes::InteractionMode>(r).on<Interaction>(On::Create | On::Update);
    track<changes::WorkspaceLights>(r).on<WorkspaceLights>(On::Create | On::Update);
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
    // Engine-owned context singletons (process-lifetime). Document state lives in SetupScene.
    r.ctx().emplace<ViewportExtent>();
    r.ctx().emplace<ViewportConsumerFence>();
    r.ctx().emplace<SelectionBitsetRef>(std::span<uint32_t>{buffers.SelectionBitset.Data(), GpuBuffers::SelectionBitsetWords});
    r.ctx().emplace<SelectionSlots>(slots);
    r.ctx().emplace<DrawState>();
    r.ctx().emplace<FrameState>();
    const auto &one_shot = r.ctx().emplace<OneShotGpu>(MakeOneShotGpu(vc.Device, vc.QueueFamily));
    r.ctx().emplace<ColliderShapeBuffers>();
    r.ctx().emplace<ViewportRenderResources>(ViewportRenderResources{
        .RenderCommandBuffer = std::move(vc.Device.allocateCommandBuffersUnique({*one_shot.Pool, vk::CommandBufferLevel::ePrimary, 1}).front()),
#ifdef MVK_FORCE_STAGED_TRANSFERS
        .TransferCommandBuffer = std::move(vc.Device.allocateCommandBuffersUnique({*one_shot.Pool, vk::CommandBufferLevel::ePrimary, 1}).front()),
#endif
        .RenderFence = vc.Device.createFenceUnique({}),
        .ViewportTexture = {},
    });

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
    std::ranges::sort(environments.Hdris, {}, &HdriEntry::Name); // SetupScene selects the active one.

    pipelines.CompileShaders();

    LoadViewportIcons(r);
    r.ctx().emplace<FaustDSP>(std::move(create_svg));
    RegisterAudioComponentHandlers(r);

    return viewport;
}

void SetupScene(entt::registry &r, entt::entity viewport) {
    r.emplace_or_replace<ViewportDisplay>(viewport);
    r.emplace_or_replace<Interaction>(viewport);
    r.emplace_or_replace<EditMode>(viewport);
    r.emplace_or_replace<ViewportTheme>(viewport, Defaults::ViewportTheme);
    r.emplace_or_replace<ViewCamera>(viewport, Defaults::ViewCamera);
    r.emplace_or_replace<MaterialPreviewLighting>(viewport, false, false, 1.f, 0.f);
    r.emplace_or_replace<RenderedLighting>(viewport, true, true, 1.f, 0.f);
    r.emplace_or_replace<WorkspaceLights>(viewport, Defaults::WorkspaceLights);
    r.emplace_or_replace<PlaybackFrame>(viewport);
    r.emplace_or_replace<LastEvaluatedFrame>(viewport);
    r.emplace_or_replace<EnabledInteractionModes>(viewport);
    r.emplace_or_replace<SelectionXRay>(viewport);
    r.emplace_or_replace<OrbitToActive>(viewport);
    r.emplace_or_replace<BoxSelectState>(viewport);
    r.emplace_or_replace<TransformGizmoState>(viewport);
    r.emplace_or_replace<GizmoInteraction>(viewport);
    r.emplace_or_replace<AnimationTimelineView>(viewport);
    r.emplace_or_replace<TimelineRange>(viewport);
    r.emplace_or_replace<TimelinePlayback>(viewport);
    physics::ApplySimulationSettings(r, r.emplace_or_replace<PhysicsSimulationSettings>(viewport));

    const auto &hdris = r.ctx().get<EnvironmentStore>().Hdris;
    const auto forest = find(hdris, "forest", &HdriEntry::Name);
    SetStudioEnvironment(r, forest != hdris.end() ? std::distance(hdris.begin(), forest) : 0);
    r.emplace_or_replace<PendingSceneWorldClear>(viewport); // Release any IBL sampler slots and restore EmptySceneWorld next pass.
}

void AddDefaultSceneContent(entt::registry &r) {
    // Default scene: a cube, a light, and a camera (startup.blend layout).
    auto &meshes = r.ctx().get<MeshStore>();
    constexpr PrimitiveShape default_shape{primitive::Cuboid{}};
    const auto [mesh_entity, _] = ::AddMesh(r, meshes, meshes.CreateMesh(primitive::CreateMesh(default_shape), {}, {}), MeshInstanceCreateInfo{.Name = ToString(default_shape)});
    r.emplace<PrimitiveShape>(mesh_entity, default_shape);

    // startup.blend data, in Blender's frame (Z-up, -Y forward)
    constexpr vec3 LightLoc{4.07625, 1.00545, 5.90386}, CameraLoc{7.358891, -6.925791, 4.958309}, CameraEulerXYZ{1.109319, 0, 0.815801};
    constexpr float Lens{50}, SensorX{36}, RenderW{16}, RenderH{9};
    // Blender Z-up -> MeshEditor Y-up is a -90° rotation about +X: (x, y, z) -> (x, z, -y)
    const auto to_y_up_pos = [](vec3 v) { return vec3{v.x, v.z, -v.y}; };
    const quat to_y_up_rot = glm::angleAxis(-float(M_PI_2), vec3{1, 0, 0});
    // Matches Blender glTF exporter (cameras.py / yvof_blender_to_gltf): horizontal fit since render aspect > sensor aspect
    const float hfov = 2 * std::atan(SensorX / (2 * Lens));
    const float yfov = 2 * std::atan(std::tan(hfov * 0.5) * RenderH / RenderW);

    ::AddLight(r, meshes, {.Name = "Light", .Transform = {.P = to_y_up_pos(LightLoc)}, .Select = MeshInstanceCreateInfo::SelectBehavior::None});
    ::AddCamera(r, meshes, {.Name = "Camera", .Transform = {.P = to_y_up_pos(CameraLoc), .R = to_y_up_rot * quat{CameraEulerXYZ}}, .Select = MeshInstanceCreateInfo::SelectBehavior::None}, Perspective{.FieldOfViewRad = yfov, .FarClip = 1000, .NearClip = DefaultPerspectiveNearClip});
}

void ClearScene(entt::registry &r, entt::entity viewport) {
    ClearMeshes(r, viewport);
    for (const auto e : r.view<entt::entity>() | to<std::vector>()) {
        if (e != viewport) r.destroy(e);
    }
    r.destroy(viewport);
    r.ctx().get<ObjectIdCounter>() = {};

    // Reset the entity allocator for deterministic entity ids on replay.
    r.storage<entt::entity>().clear();
    r.storage<entt::entity>().start_from(entt::entity{0});

    [[maybe_unused]] const auto recreated = r.create();
    assert(recreated == viewport);
    SetupScene(r, viewport);
}

void DeinitViewport(entt::registry &r, entt::entity viewport) {
    // Free GPU-resource-owning context singletons before TearDownStoreCtx erases their backing stores:
    // command buffers before their pool (OneShotGpu), and the icon/Faust SVG images (VMA allocations in
    // GpuBuffers.Ctx) and descriptor slots while GpuBuffers/DescriptorSlots and the device are still alive.
    r.ctx().erase<ViewportRenderResources>();
    r.ctx().erase<ViewportIcons>();
    r.ctx().erase<FaustDSP>();
    r.ctx().erase<SelectionSlots>();
    r.ctx().erase<SelectionBitsetRef>();
    r.ctx().erase<FrameState>();
    r.ctx().erase<DrawState>();
    if (r.valid(viewport)) r.remove<VideoRecording>(viewport);
    r.clear<Mesh>();
    r.ctx().erase<std::vector<ComponentEventHandler>>();
    r.ctx().erase<EntityDestroyTracker>();
    physics::Deinit(r);
    r.ctx().erase<Pipelines>();
    if (r.valid(viewport)) r.destroy(viewport);
    TearDownStoreCtx(r);
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
    // Stash the live consumer fence so the resize path waits on it before recreating resources.
    // Cleared below so replay sees no live consumer.
    r.ctx().get<ViewportConsumerFence>().Value = viewport_consumer_fence;
    auto &dl = *ImGui::GetWindowDrawList();
    dl.ChannelsSetCurrent(0);
    SubmitViewport(r, viewport);
    // Recreate the ImGui texture when the final color image view changed (a resize swaps the image).
    auto &resources = r.ctx().get<ViewportRenderResources>();
    if (const auto &pipelines = r.ctx().get<const Pipelines>(); pipelines.Main.Resources) {
        if (const vk::ImageView view = *pipelines.Main.Resources->FinalColorImage.View; view != resources.ViewportTextureView) {
            resources.ViewportTexture = std::make_unique<mvk::ImGuiTexture>(r.ctx().get<const VulkanResources>().Device, view, vec2{0, 1}, vec2{1, 0});
            resources.ViewportTextureView = view;
        }
    }
    if (const auto &t_ptr = resources.ViewportTexture) {
        const auto p = ImGui::GetCursorScreenPos();
        const auto extent = r.ctx().get<ViewportExtent>().Value;
        const auto &t = *t_ptr;
        dl.AddImage(ImTextureID(VkDescriptorSet(t.DescriptorSet)), p, p + ImVec2{float(extent.x), float(extent.y)}, std::bit_cast<ImVec2>(t.Uv0), std::bit_cast<ImVec2>(t.Uv1));
    }

    r.ctx().get<ViewportConsumerFence>().Value = vk::Fence{}; // No live consumer outside this call.

    dl.ChannelsSetCurrent(1);
    DrawOverlay(r, viewport, r.ctx().get<FrameState>());
}

void AdvanceViewport(entt::registry &r, entt::entity viewport) {
    AdvanceAndRecord(r, viewport, /*force_full=*/false);
}

void PresentViewport(entt::registry &r, entt::entity viewport) {
    // Replay's ticks never render the color image and consumed the scene's reactive changes,
    // so force a full record to render the final state regardless.
    if (!AdvanceAndRecord(r, viewport, /*force_full=*/true)) return;
    SubmitRecordedFrame(r);
    WaitForRender(r);
}

void WaitForRender(entt::registry &r) {
    auto &frame = r.ctx().get<FrameState>();
    if (!frame.RenderPending) return;

    const auto &vk = r.ctx().get<const VulkanResources>();
    const Timer timer{"WaitForRender"};
    WaitFor(*r.ctx().get<const ViewportRenderResources>().RenderFence, vk.Device);
    r.ctx().get<GpuBuffers>().Ctx.ReclaimRetiredBuffers();
    frame.RenderPending = false;
}
