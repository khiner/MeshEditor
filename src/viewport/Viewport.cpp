#include "viewport/Viewport.h"
#include "CameraTypes.h"
#include "Paths.h"
#include "ProcessEvents.h"
#include "Profile.h"
#include "Reactive.h"
#include "Stores.h"
#include "animation/AnimationTimeline.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "object/ObjectComponents.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsSystem.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuSceneState.h"
#include "render/MaterialImport.h"
#include "render/MeshBatch.h"
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

#include "render/GpuBuffers.h"

#include <cassert>

using std::ranges::find, std::ranges::to;

namespace {
// Metal command buffers are single-use, and RecordedPhase tracks the last build.
struct ViewportRenderResources {
    MTL::CommandBuffer *InFlight{nullptr}; // The submitted frame, until it completes.
    RenderPhase RecordedPhase{RenderPhase::Full};
};

void ResetObjectPickKeys(GpuBuffers &buffers) {
    std::fill_n(buffers.ObjectPickKeys.Data(), GpuBuffers::MaxSelectableObjects, std::numeric_limits<uint32_t>::max());
}

// Dispatch sizes follow scene recording because the rebuild determines their counts.
void SubmitRecordedFrame(entt::registry &r, MTL::CommandBuffer *command_buffer) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    SyncPreludeDispatchArgs(buffers);
    ctx.CommitResidency();
    {
        const profile::CpuScope scope{"QueueSubmit"};
        command_buffer->commit();
    }
    r.ctx().get<ViewportRenderResources>().InFlight = command_buffer;
    r.ctx().get<FrameState>().RenderPending = true;
}

void RecordAndSubmitFrame(entt::registry &r, entt::entity viewport, SceneUpdate update) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &resources = r.ctx().get<ViewportRenderResources>();
    auto *command_buffer = ctx.Queue->commandBuffer();
    RecordRenderCommandBuffer(r, viewport, command_buffer, update);
    resources.RecordedPhase = RenderPhase::Full;
    SubmitRecordedFrame(r, command_buffer);
}

RenderRequest TakeRenderRequest(entt::registry &r) {
    return std::exchange(r.ctx().get<PendingRenderRequest>().Value, RenderRequest::None);
}

SceneUpdate RequestedSceneUpdate(RenderRequest request, bool force_rebuild = false) {
    if (force_rebuild || request == RenderRequest::Rebuild) return SceneUpdate::Rebuild;
    return SceneUpdate::Reuse;
}

// Drain changes and render. Returns false while the viewport has no non-zero extent.
bool AdvanceAndRecord(entt::registry &r, entt::entity viewport, bool force_full) {
    ProcessComponentEvents(r, viewport);
    if (!ViewportImageReady(r)) return false;
    const auto render_request = TakeRenderRequest(r);
    RecordAndSubmitFrame(r, viewport, RequestedSceneUpdate(render_request, force_full));
    return true;
}

// Point view-UBO instance `instance` at the captured shutter poses, so the velocity pass reads them.
void StampShutterPoses(GpuBuffers &buffers, uint32_t instance, const GpuBuffers::VelocityPose &open, const GpuBuffers::VelocityPose &close) {
    const auto stamp = [&](const auto &value, size_t field_offset) {
        buffers.UpdateSceneViewUboField(instance, field_offset, as_bytes(value));
    };
    stamp(open.ViewProj, offsetof(SceneViewUBO, PrevViewProj));
    stamp(close.ViewProj, offsetof(SceneViewUBO, NextViewProj));
    stamp(open.Transforms.Slot, offsetof(SceneViewUBO, PrevModelSlot));
    stamp(close.Transforms.Slot, offsetof(SceneViewUBO, NextModelSlot));
    stamp(open.ArmatureDeform.Slot, offsetof(SceneViewUBO, PrevArmatureDeformSlot));
    stamp(close.ArmatureDeform.Slot, offsetof(SceneViewUBO, NextArmatureDeformSlot));
    stamp(open.MorphWeights.Slot, offsetof(SceneViewUBO, PrevMorphWeightsSlot));
    stamp(close.MorphWeights.Slot, offsetof(SceneViewUBO, NextMorphWeightsSlot));
}

// Motion blur applies in MaterialPreview/Rendered while playing, scrubbing, or capturing.
bool MotionBlurActive(const entt::registry &r, entt::entity viewport) {
    const auto &display = r.get<const ViewportDisplay>(viewport);
    if (display.ViewportShading != ViewportShadingMode::MaterialPreview && display.ViewportShading != ViewportShadingMode::Rendered) return false;
    const auto &frame_state = r.ctx().get<const FrameState>();
    if (!display.MotionBlur && !frame_state.Capturing) return false;
    return r.get<const TimelinePlayback>(viewport).Playing || frame_state.Scrubbing || frame_state.Capturing;
}

// Render the frame with motion blur across the shutter, centered on the current frame.
// Overlays stay sharp over the blur, and the settled frame is restored afterward.
void RenderMotionBlurredFrame(entt::registry &r, entt::entity viewport) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &pipelines = r.ctx().get<Pipelines>();
    auto &resources = r.ctx().get<ViewportRenderResources>();
    auto &frame_state = r.ctx().get<FrameState>();

    const auto &display = r.get<const ViewportDisplay>(viewport);
    const auto mb = EffectiveMotionBlur(display);
    const auto steps = MotionBlurSteps(display);
    const auto &range = r.get<const TimelineRange>(viewport);
    const auto &playback = r.get<const TimelinePlayback>(viewport);
    const int current_frame = playback.CurrentFrame;
    const float settled_pf = r.get<const PlaybackFrame>(viewport).Value;

    // Shutter centered on the current frame (Blender's default), clamped to the timeline range.
    const float half = mb.Shutter * 0.5f;
    const float lo = std::max(float(range.StartFrame), float(current_frame) - half);
    const float hi = std::min(float(range.EndFrame), float(current_frame) + half);

    // Cache physics through the shutter's forward half so centered sampling has both endpoints (forward playback only).
    if (playback.Playing) physics::BakeThrough(r, viewport, int(std::ceil(hi)), range.Fps);

    auto &buffers = r.ctx().get<GpuBuffers>();
    // Allocate the blur targets on first use, replacing the bindless fallback slots.
    if (pipelines.Main.EnsureMotionBlurResources(ctx)) {
        auto &slots = r.ctx().get<mtl::BindlessSet>();
        const auto &sel_slots = r.ctx().get<const SelectionSlots>();
        const auto &main = pipelines.Main;
        // The tile grid the targets were built around decides how many entries the table holds.
        buffers.ResizeMotionBlurTileIndirection(main.MotionBlur->TileImage.Extent);
        const auto accum = main.MotionBlurAccumSampler();
        const auto velocity = main.VelocitySampler();
        const auto gather = main.MotionBlurGatherSampler();
        slots.SetSampler({SlotType::Sampler, sel_slots.MotionBlurAccumSampler}, accum.Texture, accum.Sampler);
        slots.SetSampler({SlotType::Sampler, sel_slots.VelocitySampler}, velocity.Texture, velocity.Sampler);
        slots.SetSampler({SlotType::Sampler, sel_slots.MotionBlurGatherSampler}, gather.Texture, gather.Sampler);
        slots.SetTexture(sel_slots.MotionBlurTileImage, main.MotionBlurTileImage());
        slots.SetBuffer({SlotType::Buffer, sel_slots.MotionBlurTileIndirection}, *buffers.MotionBlurTileIndirection);
    }

    // Evaluate the scene at `pf` (animation + physics, which also moves the view when looking
    // through an animated camera). Each evaluation rewrites the mapped pose buffers in place.
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

    const auto render_at = [&](float pf, RenderPhase phase) {
        evaluate_at(pf);
        // Point the velocity pass at the captured shutter poses. ProcessComponentEvents rewrites the
        // whole UBO, so these have to land after it and before recording.
        StampShutterPoses(buffers, 0, buffers.ShutterOpen, buffers.ShutterClose);
        // Poses and view state reach the GPU through buffers the recorded commands already read,
        // so the recording goes stale only when the persistent scene or the phase changes.
        std::ignore = TakeRenderRequest(r);
        auto *command_buffer = ctx.Queue->commandBuffer();
        RecordRenderCommandBuffer(r, viewport, command_buffer, SceneUpdate::Rebuild, phase);
        resources.RecordedPhase = phase;
        SubmitRecordedFrame(r, command_buffer);
        WaitForRender(r);
    };

    // Evaluate the shutter's ends first so the velocity pass can reach them, then render between
    // them. Blender evaluates in this same open, close, render order.
    if (steps == 1) {
        // One step spans the whole shutter, so its blur is the finished frame: the gather output
        // goes straight to the composite, with no accumulation target to sum into and average.
        // The scene renders at the current frame, which is where the overlays draw, so both fit in
        // one recording. The shutter's ends still bound the blur, including where they clamp.
        evaluate_at(lo);
        buffers.CaptureVelocityPose(buffers.ShutterOpen);
        evaluate_at(hi);
        buffers.CaptureVelocityPose(buffers.ShutterClose);
        render_at(float(current_frame), RenderPhase::BlurredFull);
    } else {
        // Each step owns a slice of the shutter and blurs across it, rendering its centre once.
        // The first step clears the target it sums into, so the accumulation starts from it alone.
        // Every step's poses are captured up front, so all steps and the resolve record and submit
        // as one command buffer, each step reading its own view UBO instance and captured pose buffers.
        const auto step_count = std::min(uint32_t(steps), GpuBuffers::MaxBlurSteps);
        const float step_span = (hi - lo) / float(step_count);
        buffers.EnsureBlurPoses(2 * size_t(step_count) + 1);
        // Shutter boundaries at [2i]: step i opens at [2i] and closes at [2i+2], sharing each
        // interior boundary with its neighbor.
        for (uint32_t i = 0; i <= step_count; ++i) {
            evaluate_at(lo + step_span * float(i));
            buffers.CaptureVelocityPose(buffers.BlurPoses[2 * i]);
        }
        // Step centres at [2i+1], each snapshotting the step's evaluated view UBO into its instance.
        std::vector<float> step_frames(step_count);
        for (uint32_t i = 0; i < step_count; ++i) {
            const float centre = lo + step_span * float(i) + step_span * 0.5f;
            step_frames[i] = centre;
            evaluate_at(centre);
            auto &centre_pose = buffers.BlurPoses[2 * i + 1];
            buffers.CaptureVelocityPose(centre_pose);
            const uint32_t instance = i + 1;
            buffers.SnapshotSceneViewUbo(instance);
            StampShutterPoses(buffers, instance, buffers.BlurPoses[2 * i], buffers.BlurPoses[2 * i + 2]);
            // The step's own pose reads through the captured buffers, keeping draw data step-agnostic.
            const auto stamp = [&](const auto &value, size_t field_offset) {
                buffers.UpdateSceneViewUboField(instance, field_offset, as_bytes(value));
            };
            stamp(centre_pose.Transforms.Slot, offsetof(SceneViewUBO, ModelSlotOverride));
            stamp(centre_pose.ArmatureDeform.Slot, offsetof(SceneViewUBO, ArmatureDeformSlot));
            stamp(centre_pose.MorphWeights.Slot, offsetof(SceneViewUBO, MorphWeightsSlot));
        }
        // The resolve and the overlays read the live, settled state.
        evaluate_at(float(current_frame));
        std::ignore = TakeRenderRequest(r); // The recording below is always a full rebuild.
        auto *command_buffer = ctx.Queue->commandBuffer();
        RecordBlurStepsCommandBuffer(r, viewport, command_buffer, step_frames);
        // Not a single-phase recording: any later single-phase render must re-record.
        resources.RecordedPhase = RenderPhase::BlurAccumulate;
        SubmitRecordedFrame(r, command_buffer);
        WaitForRender(r);
    }

    r.get<PlaybackFrame>(viewport).Value = settled_pf;
    frame_state.RenderPending = false; // All motion blur submits were waited on internally.
}
} // namespace

bool ViewportImageReady(const entt::registry &r) {
    const auto extent = r.ctx().get<const Pipelines>().BuiltColorExtent();
    return extent.Width != 0 && extent.Height != 0;
}

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
            // Leave the request pending so the per-step render sees any re-record demand, like a resize recreating framebuffers.
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

void SetStudioEnvironment(entt::registry &r, uint32_t index) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    const auto &pipelines = r.ctx().get<const Pipelines>();
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    auto &hdri = environments.Hdris[index];
    if (!hdri.Prefiltered) {
        hdri.Prefiltered = CreateIblFromHdri(ctx, slots, pipelines.IblPrefilter, hdri.Path, hdri.Name);
    }
    const auto &pre = *hdri.Prefiltered;
    environments.ActiveHdriIndex = index;
    environments.StudioWorld = {.Ibl = MakeIblSamplers(pre, environments), .Name = hdri.Name};
}

void RebuildStudioEnvironments(entt::registry &r) {
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    if (environments.Hdris.empty()) return; // No studio environment to index into.
    const auto release = [&slots](uint32_t sampler_slot) {
        if (sampler_slot != InvalidSlot) slots.Release({SlotType::CubeSampler, sampler_slot});
    };
    for (auto &hdri : environments.Hdris) {
        if (!hdri.Prefiltered) continue;
        release(hdri.Prefiltered->DiffuseEnv.SamplerSlot);
        release(hdri.Prefiltered->SpecularEnv.SamplerSlot);
        hdri.Prefiltered.reset();
    }
    SetStudioEnvironment(r, environments.ActiveHdriIndex);
}

void SetStudioEnvironment(entt::registry &r, std::string_view name) {
    const auto &hdris = r.ctx().get<const EnvironmentStore>().Hdris;
    const auto it = find(hdris, name, &HdriEntry::Name);
    SetStudioEnvironment(r, it != hdris.end() ? uint32_t(std::distance(hdris.begin(), it)) : 0u);
}

entt::entity InitEngine(entt::registry &r) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    InitStoreCtx(r, ctx);
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &libraries = r.ctx().emplace<mtl::LibraryCache>(ctx, Paths::Shaders(), Paths::UserData() / "cache" / "Pipelines.mtl4a");
    r.ctx().emplace<Pipelines>(ctx, libraries);
    profile::Init(ctx);
    physics::Init(r);
    RegisterSceneComponentHandlers(r);

    const auto viewport = WireRegistry(r);
    auto &buffers = r.ctx().get<GpuBuffers>();
    // Engine-owned context singletons (process-lifetime). Document state lives in SetupScene.
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

    ResetObjectPickKeys(buffers);

    auto init_batch = BeginTextureUploadBatch(ctx, libraries);
    auto &environments = r.ctx().get<EnvironmentStore>();
    const auto images_dir = Paths::Res() / "images";
    environments.BrdfLut = CreateDefaultLutTexture(ctx, init_batch, slots, images_dir / "lut_ggx.png", "DefaultGGXBRDFLUT", r.ctx().get<const ActiveSamplerAnisotropy>().Value);
    environments.SheenELut = CreateDefaultLutTexture(ctx, init_batch, slots, images_dir / "lut_sheen_E.png", "DefaultSheenELUT", r.ctx().get<const ActiveSamplerAnisotropy>().Value);
    environments.CharlieLut = CreateDefaultLutTexture(ctx, init_batch, slots, images_dir / "lut_charlie.png", "DefaultCharlieLUT", r.ctx().get<const ActiveSamplerAnisotropy>().Value);
    // Blender's default world background color (linear RGB), a flat ambient-only IBL when no scene world is provided.
    environments.EmptySceneWorld = BuildFlatColorEnvironment(ctx, slots, vec3{0.05f}, "EmptySceneWorld");
    SubmitTextureUploadBatch(init_batch);
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
    r.emplace_or_replace<EnabledInteractionModes>(viewport);
    r.emplace_or_replace<OrbitToActive>(viewport);
    r.emplace_or_replace<TransformGizmoState>(viewport);
    physics::ApplySimulationSettings(r, r.emplace_or_replace<PhysicsSimulationSettings>(viewport));

    r.emplace_or_replace<StudioEnvironment>(viewport, std::string{"forest"});

    for (const auto &handler : r.ctx().get<SceneSetupHandlers>().Handlers) handler(r, viewport);
}

void AddDefaultSceneContent(entt::registry &r) {
    // Default scene: a cube, a light, and a camera (startup.blend layout).
    auto &meshes = r.ctx().get<MeshStore>();
    constexpr PrimitiveShape default_shape{primitive::Cuboid{}};
    const auto created = CreateMesh(r, {.Data = primitive::CreateMesh(default_shape), .FlatShaded = true});
    const auto [mesh_entity, _] = ::AddMesh(r, created.StoreId, MeshInstanceCreateInfo{.Name = ToString(default_shape)});
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
        auto &slots = r.ctx().get<mtl::BindlessSet>();
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

    // Reset domain caches keyed by the destroyed entities' ids, before the allocator reset lets the next scene reuse them.
    if (const auto *clear_handlers = r.ctx().find<SceneClearHandlers>()) {
        for (const auto &handler : clear_handlers->Handlers) handler(r);
    }

    // Reset the entity, mesh-store, and GPU-arena allocators to their fresh-start state, so replaying a scene from this
    // baseline re-allocates identical ids and GPU handles. Bindless slots need no reset because their allocator is order-independent.
    r.storage<entt::entity>().clear();
    r.storage<entt::entity>().start_from(entt::entity{0});
    r.ctx().get<MeshStore>().Clear();
    r.ctx().get<GpuBuffers>().ResetSceneArenas();
    // The depth pyramid still holds the cleared scene, so the next scene's first cull must not test against it.
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
    profile::Report();
    profile::Deinit();
    r.ctx().erase<Pipelines>();
    if (r.valid(viewport)) r.destroy(viewport);
    TearDownStoreCtx(r);
}

void PresentViewport(entt::registry &r, entt::entity viewport) {
    // Replay may reach this before the viewport has an extent.
    // Once it has one, record the complete state even if earlier zero-extent ticks consumed its reactive changes.
    if (!AdvanceAndRecord(r, viewport, /*force_full=*/true)) return;
    WaitForRender(r);
}

void WaitForRender(entt::registry &r) {
    auto &frame = r.ctx().get<FrameState>();
    if (!frame.RenderPending) return;

    auto &resources = r.ctx().get<ViewportRenderResources>();
    if (resources.InFlight) {
        const profile::CpuScope scope{"WaitGpu"};
        resources.InFlight->waitUntilCompleted();
    }
    profile::Resolve(resources.InFlight);
    resources.InFlight = nullptr;
    r.ctx().get<GpuBuffers>().Ctx.ReclaimRetiredBuffers();
    frame.RenderPending = false;

    const auto pending_box_stats = r.view<const BoxSelectStatsDirty>();
    const std::vector<entt::entity> completed_box_selections{pending_box_stats.begin(), pending_box_stats.end()};
    for (const auto viewport : completed_box_selections) PublishBoxSelectElementStats(r, viewport);
}
