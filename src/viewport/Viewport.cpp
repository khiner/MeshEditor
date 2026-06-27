#include "viewport/Viewport.h"
#include "Camera.h"
#include "Changes.h"
#include "Paths.h"
#include "ProcessEvents.h"
#include "Reactive.h"
#include "Stores.h"
#include "Timer.h"
#include "animation/AnimationTimeline.h"
#include "armature/ArmatureComponents.h"
#include "armature/BoneConstraint.h"
#include "audio/SoundVertices.h"
#include "gizmo/GizmoInteraction.h"
#include "gltf/SourceAssets.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "object/ExtrasComponents.h"
#include "object/ObjectComponents.h"
#include "object/ObjectOps.h"
#include "object/PendingSync.h"
#include "physics/PhysicsSystem.h"
#include "physics/PhysicsTypes.h"
#include "render/DrawState.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/MaterialImport.h"
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
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"
#include "viewport/ViewportRenderGpu.h"

#include "render/GpuBuffers.h"
#include "render/LightComponents.h"

#include <cassert>

using std::ranges::find, std::ranges::to;

namespace {
struct ViewportRenderResources {
    vk::UniqueCommandBuffer RenderCommandBuffer;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    vk::UniqueCommandBuffer TransferCommandBuffer;
#endif
    vk::UniqueFence RenderFence;
};

void ResetObjectPickKeys(GpuBuffers &buffers) {
    std::fill_n(buffers.ObjectPickKeys.Data(), GpuBuffers::MaxSelectableObjects, std::numeric_limits<uint32_t>::max());
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

// Drain component events, resize as needed, and record the frame.
// Returns false (skipping the record) when the viewport has no non-zero extent yet.
// `force_full` records the full buffer regardless of the requested region.
bool AdvanceAndRecord(entt::registry &r, entt::entity viewport, bool force_full) {
    const auto render_request = ProcessComponentEvents(r, viewport);
    const auto extent = r.ctx().get<const Pipelines>().BuiltColorExtent();
    if (extent.width == 0 || extent.height == 0) return false;
    RecordViewportFrame(r, viewport, render_request, force_full);
    return true;
}
} // namespace

void SubmitViewport(entt::registry &r, entt::entity viewport, vk::Fence viewport_consumer_fence) {
    // Stash the consumer fence for the resize path to wait on before recreating resources. Cleared after so replay sees none.
    r.ctx().get<ViewportConsumerFence>().Value = viewport_consumer_fence;
    const auto render_request = ProcessComponentEvents(r, viewport);
    r.ctx().get<ViewportConsumerFence>().Value = vk::Fence{};
    if (render_request == RenderRequest::None) return;

    const auto extent = r.ctx().get<const Pipelines>().BuiltColorExtent();
    if (extent.width == 0 || extent.height == 0) return;

    const Timer timer{"SubmitViewport"};
#ifdef MVK_FORCE_STAGED_TRANSFERS
    RecordTransferCommandBuffer(r, viewport, *r.ctx().get<ViewportRenderResources>().TransferCommandBuffer);
#endif

    RecordViewportFrame(r, viewport, render_request);
    SubmitRecordedFrame(r);
}

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

void SetStudioEnvironment(entt::registry &r, std::string_view name) {
    const auto &hdris = r.ctx().get<const EnvironmentStore>().Hdris;
    const auto it = find(hdris, name, &HdriEntry::Name);
    SetStudioEnvironment(r, it != hdris.end() ? uint32_t(std::distance(hdris.begin(), it)) : 0u);
}

entt::entity InitEngine(entt::registry &r, VulkanResources vc) {
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
    track<changes::ObjectCreated>(r).on<ObjectKind>(On::Create);
    track<changes::RenderInstanceCreated>(r).on<RenderInstance>(On::Create);
    track<changes::ViewportDisplay>(r).on<ViewportDisplay>(On::Create | On::Update);
    track<changes::InteractionMode>(r).on<Interaction>(On::Create | On::Update);
    track<changes::WorkspaceLights>(r).on<WorkspaceLights>(On::Create | On::Update);
    track<changes::ViewportTheme>(r).on<ViewportTheme>(On::Create | On::Update);
    track<changes::Materials>(r).on<MaterialDirty>(On::Create | On::Update);
    track<changes::MaterializedTextures>(r).on<MaterializedTextures>(On::Create | On::Update);
    track<changes::StudioEnvironment>(r).on<StudioEnvironment>(On::Create | On::Update);
    track<changes::SceneWorld>(r).on<gltf::SourceAssets>(On::Create | On::Update);
    track<changes::PunctualLight>(r).on<PunctualLight>(On::Create | On::Update);
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
        .on<PosedLocal>(On::Create | On::Update)
        .on<SceneNode>(On::Create | On::Update)
        .on<BoneDisplayScale>(On::Update);
    r.ctx().emplace<EntityDestroyTracker>().Bind(r);

    r.on_destroy<Name>().connect<[](entt::registry &r, entt::entity e) {
        if (auto *registry = r.ctx().find<NameRegistry>()) registry->Names.erase(r.get<const Name>(e).Value);
    }>();
    // Assign a stable ObjectId (0 means unassigned) on RenderInstance construction.
    r.on_construct<RenderInstance>().connect<[](entt::registry &r, entt::entity e) {
        if (r.get<const RenderInstance>(e).ObjectId != 0) return;
        if (auto *counter = r.ctx().find<ObjectIdCounter>()) {
            r.patch<RenderInstance>(e, [counter](auto &ri) { ri.ObjectId = counter->Next++; });
        }
    }>();
    r.on_destroy<RenderInstance>().connect<[](entt::registry &r, entt::entity e) {
        const auto &ri = r.get<const RenderInstance>(e);
        if (ri.BufferIndex == UINT32_MAX) return; // Same-frame show+hide — never synced to GPU.
        r.get_or_emplace<PendingHide>(ri.Entity).BufferIndices.push_back(ri.BufferIndex);
    }>();
    // An instance renders unless Hidden: create its RenderInstance on construction, drop it when Hidden appears.
    // Together these keep RenderInstance in lockstep with Instance + !Hidden, including on snapshot restore
    // (which emplaces Instance and Hidden in either order).
    r.on_construct<Instance>().connect<[](entt::registry &r, entt::entity e) {
        if (!r.all_of<Hidden>(e) && !r.all_of<RenderInstance>(e)) r.emplace<RenderInstance>(e, r.get<Instance>(e).Entity, UINT32_MAX, 0u);
    }>();
    r.on_construct<Hidden>().connect<[](entt::registry &r, entt::entity e) {
        if (r.all_of<RenderInstance>(e)) r.remove<RenderInstance>(e);
    }>();
    // Build MeshBuffers when a vertex handle is constructed (MeshHandle = full meshes,
    // VertexStoreId = vertex-only extras, OverlayVertexStoreId = overlays). Index ranges fill in afterward.
    r.on_construct<MeshHandle>().connect<[](entt::registry &r, entt::entity e) {
        auto &meshes = r.ctx().get<MeshStore>();
        r.emplace<MeshBuffers>(e, meshes.GetVerticesRange(r.get<const MeshHandle>(e).StoreId), SlottedRange{}, SlottedRange{}, SlottedRange{});
    }>();
    r.on_construct<VertexStoreId>().connect<[](entt::registry &r, entt::entity e) {
        auto &meshes = r.ctx().get<MeshStore>();
        r.emplace<MeshBuffers>(e, meshes.GetVerticesRange(r.get<const VertexStoreId>(e).StoreId), SlottedRange{}, SlottedRange{}, SlottedRange{});
    }>();
    r.on_construct<OverlayVertexStoreId>().connect<[](entt::registry &r, entt::entity e) {
        auto &meshes = r.ctx().get<MeshStore>();
        r.emplace<MeshBuffers>(e, meshes.GetOverlayVerticesRange(r.get<const OverlayVertexStoreId>(e).StoreId), SlottedRange{}, SlottedRange{}, SlottedRange{});
    }>();
    // BoneConstraints edits change the resolved local Transform; poke it to drive the WorldTransform recompute.
    r.on_update<BoneConstraints>().connect<[](entt::registry &r, entt::entity e) {
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
    // Default scene world (no imported EXT-IBL). The reactive SceneWorld pass swaps in an imported world when
    // a glTF with EXT_lights_image_based is loaded or restored, and ClearScene restores this default.
    environments.SceneWorld = {.Ibl = MakeIblSamplers(environments.EmptySceneWorld, environments), .Name = environments.EmptySceneWorld.Name};
    // Safe placeholder until the reactive StudioEnvironment pass prefilters the selected HDRI on the first tick.
    environments.StudioWorld = environments.SceneWorld;

    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator{images_dir / "studiolights" / "world", ec}) {
        if (entry.path().extension() == ".hdr") {
            environments.Hdris.emplace_back(HdriEntry{.Name = entry.path().stem().string(), .Path = entry.path(), .Prefiltered = {}});
        }
    }
    std::ranges::sort(environments.Hdris, {}, &HdriEntry::Name); // SetupScene selects the active one.

    pipelines.CompileShaders();

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

    // Default studio-environment selection.
    r.emplace_or_replace<StudioEnvironment>(viewport, std::string{"forest"});
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
    // Clear physics while its components still exist, so the next load isn't tripped by stale entity keys.
    physics::Clear(r);
    ClearMeshes(r, viewport);

    // Release any imported (EXT-IBL) scene world and restore the empty default, so a subsequent restore starts
    // bare and its reactive SceneWorld pass rebuilds the imported world from the restored SourceAssets.
    auto &environments = r.ctx().get<EnvironmentStore>();
    if (environments.ImportedSceneWorld) {
        auto &slots = r.ctx().get<DescriptorSlots>();
        ReleaseCubeSamplerSlot(slots, environments.ImportedSceneWorld->DiffuseEnv.SamplerSlot);
        ReleaseCubeSamplerSlot(slots, environments.ImportedSceneWorld->SpecularEnv.SamplerSlot);
        environments.ImportedSceneWorld.reset();
        environments.SceneWorldRotation = mat3{1.f};
        environments.SceneWorld = {.Ibl = MakeIblSamplers(environments.EmptySceneWorld, environments), .Name = environments.EmptySceneWorld.Name};
    }

    // Reset imported textures + materials to the default. ClearMeshes does this only when the last instance is
    // destroyed, which skinned scenes never reach (bone-visual instances outlive the mesh), so do it explicitly.
    ResetImportedTexturesAndMaterials(r);

    // Lights live in a Derived GPU buffer keyed by LightIndex (also Derived). Clear it so restored lights are
    // re-registered from their (Persistent) PunctualLight starting at slot 0, with no stale entries.
    r.ctx().get<GpuBuffers>().Lights.SetCount(0);

    // Destroy instances before the buffer entities they reference.
    for (const auto e : r.view<RenderInstance>() | to<std::vector>()) r.destroy(e);
    for (const auto e : r.view<entt::entity>() | to<std::vector>()) {
        if (e != viewport) r.destroy(e);
    }
    r.destroy(viewport);
    r.ctx().get<ObjectIdCounter>() = {};

    // Drop ColliderShapeBuffers' cached handles to the wireframe buffer entities destroyed above, so
    // EnsureWireframes rebuilds them instead of mistaking a reused entity id for a live buffer.
    r.ctx().get<ColliderShapeBuffers>().Entities.fill(entt::entity{entt::null});

    // Reset the entity, mesh-store, and GPU-arena allocators to their fresh-start state, so replaying a scene from this
    // baseline re-allocates identical ids and GPU handles. Descriptor slots need no reset, since their RangeAllocator is order-independent.
    r.storage<entt::entity>().clear();
    r.storage<entt::entity>().start_from(entt::entity{0});
    r.ctx().get<MeshStore>().Clear();
    r.ctx().get<GpuBuffers>().ResetSceneArenas();

    [[maybe_unused]] const auto recreated = r.create();
    assert(recreated == viewport);
    SetupScene(r, viewport);
}

void DeinitViewport(entt::registry &r, entt::entity viewport) {
    r.ctx().erase<ViewportRenderResources>();
    r.ctx().erase<SelectionSlots>();
    r.ctx().erase<SelectionBitsetRef>();
    r.ctx().erase<FrameState>();
    r.ctx().erase<DrawState>();
    r.clear<Mesh>();
    r.ctx().erase<std::vector<ComponentEventHandler>>();
    r.ctx().erase<EntityDestroyTracker>();
    physics::Deinit(r);
    r.ctx().erase<Pipelines>();
    if (r.valid(viewport)) r.destroy(viewport);
    TearDownStoreCtx(r);
}

void AdvanceViewport(entt::registry &r, entt::entity viewport) { AdvanceAndRecord(r, viewport, /*force_full=*/false); }

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
