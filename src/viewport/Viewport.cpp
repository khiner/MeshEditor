#include "viewport/Viewport.h"

#include "render/ViewportSubmission.h"
#include <Metal/MTLCommandQueue.hpp>

#include "CameraTypes.h"
#include "Paths.h"
#include "ProcessEvents.h"
#include "Profile.h"
#include "Reactive.h"
#include "Stores.h"
#include "Window.h"
#include "action/ActionIndex.h"
#include "animation/AnimationTimeline.h"
#include "mesh/Mesh.h"
#include "mesh/MeshBatch.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "object/ObjectComponents.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsSystem.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuSceneState.h"
#include "render/MaterialImport.h"
#include "render/Pipelines.h"
#include "render/Textures.h"
#include "scene/Defaults.h"
#include "scene/EntityDestroyTracker.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionQueries.h"
#include "viewport/FrameState.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportConsumerFence.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"
#include "viewport/ViewportRenderGpu.h"
#include <numbers>

#include "render/GpuBuffers.h"

#include <cassert>

using std::ranges::find, std::ranges::to;

namespace {
RenderRequest TakeRenderRequest(entt::registry &r) {
    return std::exchange(r.ctx().get<PendingRenderRequest>().Value, RenderRequest::None);
}

SceneUpdate RequestedSceneUpdate(RenderRequest request) { return request == RenderRequest::Rebuild ? SceneUpdate::Rebuild : SceneUpdate::Reuse; }

void AdvanceViewport(entt::registry &r, entt::entity viewport, RenderPhase phase) {
    ProcessComponentEvents(r, viewport);
    if (!ViewportImageReady(r)) return;
    const auto request = TakeRenderRequest(r);
    if (phase == RenderPhase::Prepare && request == RenderRequest::None) return;
    RecordAndSubmitFrame(r, viewport, RequestedSceneUpdate(request), phase);
    WaitForRender(r);
}

// Motion blur applies in MaterialPreview/Rendered while playing, scrubbing, or capturing.
bool MotionBlurActive(const entt::registry &r, entt::entity viewport) {
    const auto &display = r.get<const ViewportDisplay>(viewport);
    if (display.ViewportShading != ViewportShadingMode::MaterialPreview && display.ViewportShading != ViewportShadingMode::Rendered) return false;
    const auto &frame_state = r.ctx().get<const FrameState>();
    if (!display.MotionBlur && !frame_state.Capturing) return false;
    return r.get<const TimelinePlayback>(viewport).Playing || frame_state.Scrubbing || frame_state.Capturing;
}

// Renders shutter samples with sharp overlays and restores the current frame afterward.
void RenderMotionBlurredFrame(entt::registry &r, entt::entity viewport) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &pipelines = r.ctx().get<Pipelines>();
    auto &resources = r.ctx().get<ViewportRenderResources>();
    auto &frame_state = r.ctx().get<FrameState>();

    const auto &display = r.get<const ViewportDisplay>(viewport);
    const auto mb = EffectiveMotionBlur(display);
    const bool fast = mb.Method == MotionBlurMethod::Fast;
    const auto count = fast ? 2u : MotionBlurSteps(display);
    const auto &range = r.get<const TimelineRange>(viewport);
    const auto &playback = r.get<const TimelinePlayback>(viewport);
    const int current_frame = playback.CurrentFrame;
    const float settled_pf = r.get<const PlaybackFrame>(viewport).Value;

    // Match Blender's centered shutter and clamp it to the timeline.
    const float half = mb.Shutter * 0.5f;
    const float lo = std::max(float(range.StartFrame), float(current_frame) - half);
    const float hi = std::min(float(range.EndFrame), float(current_frame) + half);

    const float last_sample = fast ? hi : lo + (hi - lo) * (1.f - 0.5f / float(count));
    physics::BakeThrough(r, viewport, int(std::ceil(last_sample)), range.Fps);

    auto &buffers = r.ctx().get<GpuBuffers>();
    if (pipelines.Main.EnsureMotionBlurResources(ctx, fast)) {
        auto &slots = r.ctx().get<mtl::BindlessSet>();
        const auto sampled = pipelines.Main.MotionBlurOutputSampler();
        slots.SetSampler({SlotType::Sampler, r.ctx().get<const SelectionSlots>().MotionBlurOutputSampler}, sampled.Texture, sampled.Sampler);
        const auto velocity = pipelines.Main.Nearest(fast ? &pipelines.Main.MotionBlur->VelocityImage : nullptr);
        slots.SetSampler({SlotType::Sampler, r.ctx().get<const SelectionSlots>().VelocitySampler}, velocity.Texture, velocity.Sampler);
    }

    // Evaluate animation, physics, and an animated look-through camera at `pf` into mapped pose buffers.
    const auto evaluate_at = [&](float pf) {
        {
            const profile::CpuScope scope{"SamplePoses"};
            physics::SamplePosesAtFrame(r, pf);
        }
        r.get<PlaybackFrame>(viewport).Value = pf;
        frame_state.MotionBlurSubFrame = true;
        ProcessComponentEvents(r, viewport);
        frame_state.MotionBlurSubFrame = false;
    };

    std::vector<uint32_t> sample_weights;
    sample_weights.reserve(count);
    buffers.BlurPoses.reserve(count);
    const auto same_buffer = [](const mtl::Buffer &a, const mtl::Buffer &b) {
        return a.UsedSize == b.UsedSize && (a.UsedSize == 0 || std::memcmp(a.Contents().data(), b.Contents().data(), a.UsedSize) == 0);
    };
    for (uint32_t i = 0; i < count; ++i) {
        const float time = fast ? (i == 0 ? lo : hi) : lo + (hi - lo) * (float(i) + 0.5f) / float(count);
        evaluate_at(time);
        // Consecutive identical samples need one render, including static scenes and clamped shutters.
        if (!fast && !sample_weights.empty()) {
            const auto &previous = buffers.BlurPoses[sample_weights.size() - 1];
            auto current_view = *reinterpret_cast<const SceneViewUBO *>(buffers.SceneViewUBO.Contents().data());
            const auto &previous_view = *reinterpret_cast<const SceneViewUBO *>(buffers.SceneViewUBO.Contents().data() + buffers.SceneViewUboOffset(sample_weights.size()));
            previous.ApplyTo(current_view);
            if (std::memcmp(&previous_view, &current_view, sizeof(current_view)) == 0 &&
                same_buffer(previous.Transforms, buffers.Instances.TransformBuffer) &&
                same_buffer(previous.ArmatureDeform, buffers.ArmatureDeformBuffer.Buffer) &&
                same_buffer(previous.MorphWeights, buffers.MorphWeightBuffer.Buffer) &&
                same_buffer(previous.Lights, buffers.Lights)) {
                ++sample_weights.back();
                continue;
            }
        }
        const uint32_t instance = uint32_t(sample_weights.size()) + 1u;
        if (buffers.BlurPoses.size() < instance) buffers.BlurPoses.emplace_back(buffers.Ctx);
        auto &pose = buffers.BlurPoses[instance - 1u];
        buffers.CaptureRenderPose(pose);
        auto view = *reinterpret_cast<const SceneViewUBO *>(buffers.SceneViewUBO.Contents().data());
        pose.ApplyTo(view);
        buffers.SceneViewUBO.Update(as_bytes(view), buffers.SceneViewUboOffset(instance));
        sample_weights.push_back(1u);
    }
    evaluate_at(settled_pf);
    std::ignore = TakeRenderRequest(r);
    auto *command_buffer = ctx.Queue->commandBuffer();
    if (fast) RecordRenderCommandBuffer(r, viewport, command_buffer, SceneUpdate::Rebuild, RenderPhase::BlurFast);
    else RecordBlurStepsCommandBuffer(r, viewport, command_buffer, sample_weights);
    resources.RecordedPhase = fast ? RenderPhase::BlurFast : RenderPhase::BlurAccumulate;
    SubmitRecordedFrame(r, command_buffer);
    WaitForRender(r);
}
} // namespace

void SubmitViewport(entt::registry &r, entt::entity viewport, MTL::CommandBuffer *viewport_consumer) {
    const profile::CpuScope scope{"SubmitViewport"};
    // Resize waits for this consumer before replacing its sampled texture.
    r.ctx().get<ViewportConsumerFence>().Value = viewport_consumer;
    ProcessComponentEvents(r, viewport);
    r.ctx().get<ViewportConsumerFence>().Value = nullptr;
    if (!ViewportImageReady(r)) return;
    auto &frame_state = r.ctx().get<FrameState>();
    if (MotionBlurActive(r, viewport)) {
        // A blurred frame costs several scene evaluations, so only run one when something changed.
        if (const auto request = TakeRenderRequest(r); request != RenderRequest::None) {
            // Preserve the request for the per-step render and any required framebuffer recreation.
            r.ctx().get<PendingRenderRequest>().Value = request;
            RenderMotionBlurredFrame(r, viewport);
            frame_state.MotionBlurred = true;
        }
        return;
    }
    // Blur just ended (playback stopped, or the playhead was released): replace the blurred frame with a sharp one.
    if (frame_state.MotionBlurred) {
        frame_state.MotionBlurred = false;
        r.ctx().get<PendingRenderRequest>().Value = RenderRequest::Rebuild;
    }
    const auto render_request = TakeRenderRequest(r);
    if (render_request == RenderRequest::None) return;

    RecordAndSubmitFrame(r, viewport, RequestedSceneUpdate(render_request));
}

entt::entity InitEngine(entt::registry &r) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    InitStoreCtx(r, ctx);
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &libraries = r.ctx().emplace<mtl::LibraryCache>(ctx, Paths::Shaders(), Paths::UserData() / "cache" / "Pipelines.mtl4a");
    r.ctx().emplace<Pipelines>(libraries);
    r.ctx().emplace<MeshPipelines>(libraries);
    physics::Init(r);
    RegisterSceneComponentHandlers(r);

    const auto viewport = WireRegistry(r);
    auto &buffers = r.ctx().get<GpuBuffers>();
    // These engine resources outlive documents.
    r.ctx().emplace<ViewportExtent>();
    r.ctx().emplace<ViewportConsumerFence>();
    const auto &sel_slots = r.ctx().emplace<SelectionSlots>(slots);
    // These selection buffers are engine-lifetime and never resized, so their bindless entries are bound once here.
    slots.SetBuffer({SlotType::Buffer, sel_slots.ObjectPickKey}, *buffers.ObjectPickKeys);
    slots.SetBuffer({SlotType::Buffer, sel_slots.ElementPickKey}, *buffers.ElementPickKey);
    slots.SetBuffer({SlotType::Buffer, sel_slots.ElementPickId}, *buffers.ElementPickId);
    slots.SetBuffer({SlotType::Buffer, sel_slots.ObjectPickSeenBits}, *buffers.ObjectPickSeenBitset);
    slots.SetBuffer({SlotType::Buffer, sel_slots.ObjectBoxBitset}, *buffers.ObjectBoxBitset);
    r.ctx().emplace<GpuSceneState>();
    r.ctx().emplace<FrameState>();
    r.ctx().emplace<PendingRenderRequest>();
    r.ctx().emplace<ViewportRenderResources>();
    r.ctx().emplace<WindowsState>();

    auto init_batch = BeginTextureUploadBatch(ctx, libraries);
    auto &environments = r.ctx().get<EnvironmentStore>();
    const auto images_dir = Paths::Res() / "images";
    environments.BrdfLut = CreateDefaultLutTexture(ctx, init_batch, slots, images_dir / "lut_ggx.png", "DefaultGGXBRDFLUT", r.ctx().get<const ActiveSamplerAnisotropy>().Value);
    environments.SheenELut = CreateDefaultLutTexture(ctx, init_batch, slots, images_dir / "lut_sheen_E.png", "DefaultSheenELUT", r.ctx().get<const ActiveSamplerAnisotropy>().Value);
    environments.CharlieLut = CreateDefaultLutTexture(ctx, init_batch, slots, images_dir / "lut_charlie.png", "DefaultCharlieLUT", r.ctx().get<const ActiveSamplerAnisotropy>().Value);
    // Blender's default world background color (linear RGB), a flat ambient-only IBL when no scene world is provided.
    environments.EmptySceneWorld = BuildFlatColorEnvironment(ctx, slots, vec3{0.05f}, "EmptySceneWorld");
    SubmitTextureUploadBatch(init_batch);
    // SceneWorld uses this default until reactive EXT_lights_image_based loading replaces it.
    environments.SceneWorld = {.Ibl = MakeIblSamplers(environments.EmptySceneWorld, environments), .Name = environments.EmptySceneWorld.Name};
    // Safe placeholder until the reactive StudioEnvironment pass prefilters the selected HDRI on the first tick.
    environments.StudioWorld = environments.SceneWorld;

    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator{images_dir / "studiolights" / "world", ec}) {
        if (entry.path().extension() == ".hdr") {
            environments.Hdris.emplace_back(HdriEntry{.Name = entry.path().stem().string(), .Path = entry.path(), .Prefiltered = {}});
        }
    }
    std::ranges::sort(environments.Hdris, {}, &HdriEntry::Name);

    return viewport;
}

void SetupScene(entt::registry &r, entt::entity viewport) {
    r.emplace_or_replace<ActionIndex>(viewport);
    r.emplace_or_replace<ViewportDisplay>(viewport);
    r.emplace_or_replace<Interaction>(viewport);
    r.emplace_or_replace<EditMode>(viewport);
    r.emplace_or_replace<ViewportTheme>(viewport, Defaults::ViewportTheme);
    r.emplace_or_replace<ViewCamera>(viewport, Defaults::ViewCamera);
    r.emplace_or_replace<MaterialPreviewLighting>(viewport, false, false, 1.f, 0.f);
    r.emplace_or_replace<RenderedLighting>(viewport, true, true, 1.f, 0.f);
    r.emplace_or_replace<WorkspaceLights>(viewport, Defaults::WorkspaceLights);
    r.emplace_or_replace<EnabledInteractionModes>(viewport);
    r.emplace_or_replace<OrbitToActive>(viewport);
    r.emplace_or_replace<TransformGizmoState>(viewport);
    physics::ApplySimulationSettings(r, r.emplace_or_replace<PhysicsSimulationSettings>(viewport));

    r.emplace_or_replace<StudioEnvironment>(viewport, std::string{"forest"});

    for (const auto &handler : r.ctx().get<SceneSetupHandlers>().Handlers) handler(r, viewport);
}

void AddDefaultSceneContent(entt::registry &r) {
    auto &meshes = r.ctx().get<MeshStore>();
    constexpr PrimitiveShape default_shape{primitive::Cuboid{}};
    const auto created = CreateMesh(r, {.Data = primitive::CreateMesh(default_shape), .FlatShaded = true});
    const auto [mesh_entity, _] = ::AddMesh(r, created.StoreId, MeshInstanceCreateInfo{.Name = ToString(default_shape)});
    r.emplace<PrimitiveShape>(mesh_entity, default_shape);

    // Match Blender's startup scene in its Z-up, negative-Y-forward frame.
    constexpr vec3 LightLoc{4.07625, 1.00545, 5.90386}, CameraLoc{7.358891, -6.925791, 4.958309}, CameraEulerXYZ{1.109319, 0, 0.815801};
    constexpr float Lens{50}, SensorX{36}, RenderW{16}, RenderH{9};
    // Blender Z-up -> MeshEditor Y-up is a -90° rotation about +X: (x, y, z) -> (x, z, -y)
    const auto to_y_up_pos = [](vec3 v) { return vec3{v.x, v.z, -v.y}; };
    const quat to_y_up_rot = numeric::AngleAxis(-std::numbers::pi_v<float> / 2.f, vec3{1, 0, 0});
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

    // Restore the default world so reactive loading can rebuild imported lighting from restored source assets.
    auto &environments = r.ctx().get<EnvironmentStore>();
    if (environments.ImportedSceneWorld) {
        auto &slots = r.ctx().get<mtl::BindlessSet>();
        ReleaseCubeSamplerSlot(slots, environments.ImportedSceneWorld->DiffuseEnv.SamplerSlot);
        ReleaseCubeSamplerSlot(slots, environments.ImportedSceneWorld->SpecularEnv.SamplerSlot);
        environments.ImportedSceneWorld.reset();
        environments.SceneWorldRotation = mat3{1.f};
        environments.SceneWorld = {.Ibl = MakeIblSamplers(environments.EmptySceneWorld, environments), .Name = environments.EmptySceneWorld.Name};
    }

    // Reset imported materials explicitly because bone visualization keeps skinned-scene mesh storage alive.
    ResetImportedTexturesAndMaterials(r);

    // Clear derived light slots so restored persistent lights register from slot zero.
    r.ctx().get<GpuBuffers>().Lights.SetCount(0);

    // Destroy instances before the buffer entities they reference.
    for (const auto e : r.view<RenderInstance>() | to<std::vector>()) r.destroy(e);
    for (const auto e : r.view<entt::entity>() | to<std::vector>()) {
        if (e != viewport) r.destroy(e);
    }
    r.destroy(viewport);
    r.ctx().get<ObjectIdCounter>() = {};

    // Reset domain caches keyed by the destroyed entities' ids, before the allocator reset lets the next scene reuse them.
    if (const auto *clear_handlers = r.ctx().find<SceneClearHandlers>()) {
        for (const auto &handler : clear_handlers->Handlers) handler(r);
    }

    // Reset ordered allocators so scene replay reproduces entity IDs and GPU handles.
    // Bindless allocation is order-independent and requires no reset.
    r.storage<entt::entity>().clear();
    r.storage<entt::entity>().start_from(entt::entity{0});
    r.ctx().get<MeshStore>().Clear();
    r.ctx().get<GpuBuffers>().ResetSceneArenas();
    r.ctx().get<GpuSceneState>() = {};
    // Disable occlusion until the new scene has produced a depth pyramid.
    if (auto &resources = r.ctx().get<Pipelines>().Main.Resources) resources->DepthPyramidValid = false;

    [[maybe_unused]] const auto recreated = r.create();
    assert(recreated == viewport);
    SetupScene(r, viewport);
}

void DeinitViewport(entt::registry &r, entt::entity viewport) {
    r.ctx().erase<ViewportRenderResources>();
    r.ctx().erase<SelectionSlots>();
    r.ctx().erase<FrameState>();
    r.ctx().erase<PendingRenderRequest>();
    r.ctx().erase<GpuSceneState>();
    r.clear<Mesh>();
    r.ctx().erase<std::vector<ComponentEventHandler>>();
    r.ctx().erase<EntityDestroyTracker>();
    physics::Deinit(r);
    r.ctx().erase<MeshPipelines>();
    r.ctx().erase<Pipelines>();
    if (r.valid(viewport)) r.destroy(viewport);
    TearDownStoreCtx(r);
}

void PresentViewport(entt::registry &r, entt::entity viewport) {
    if (MotionBlurActive(r, viewport)) {
        ProcessComponentEvents(r, viewport);
        if (!ViewportImageReady(r)) return;
        RenderMotionBlurredFrame(r, viewport);
        r.ctx().get<FrameState>().MotionBlurred = true;
    } else {
        AdvanceViewport(r, viewport, RenderPhase::Full);
    }
}
void PrepareViewport(entt::registry &r, entt::entity viewport) { AdvanceViewport(r, viewport, RenderPhase::Prepare); }
