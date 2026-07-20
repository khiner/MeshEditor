#include "viewport/Viewport.h"
#include "CameraTypes.h"
#include "Paths.h"
#include "ProcessEvents.h"
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
#include "render/DrawState.h"
#include "render/MaterialImport.h"
#include "render/OneShotGpu.h"
#include "render/Pipelines.h"
#include "render/Profile.h"
#include "render/Textures.h"
#include "render/VkFenceWait.h"
#include "scene/Defaults.h"
#include "scene/EntityDestroyTracker.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
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
struct ViewportRenderResources {
    vk::UniqueCommandBuffer RenderCommandBuffer;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    vk::UniqueCommandBuffer TransferCommandBuffer;
#endif
    vk::UniqueFence RenderFence;
    RenderPhase RecordedPhase{RenderPhase::Full}; // What RenderCommandBuffer currently holds.
};

void ResetObjectPickKeys(GpuBuffers &buffers) {
    std::fill_n(buffers.ObjectPickKeys.Data(), GpuBuffers::MaxSelectableObjects, std::numeric_limits<uint32_t>::max());
}

// Submit the recorded command buffer. RenderFence must be unsignaled (reset by the prior WaitForRender).
void SubmitRecordedFrame(entt::registry &r) {
    const auto &vk = r.ctx().get<const VulkanResources>();
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &resources = r.ctx().get<ViewportRenderResources>();
    // Always ensure the draw slots point to render draw data before submitting (a selection pass may have swapped them).
    buffers.SetSceneViewDrawSlots(buffers.RenderDraw);
    SyncPreludeDispatchArgs(buffers);
    vk::SubmitInfo submit;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    const std::array command_buffers{*resources.TransferCommandBuffer, *resources.RenderCommandBuffer};
    submit.setCommandBuffers(command_buffers);
#else
    submit.setCommandBuffers(*resources.RenderCommandBuffer);
#endif
    {
        const profile::CpuScope scope{"QueueSubmit"};
        vk.Queue.submit(submit, *resources.RenderFence);
    }
    r.ctx().get<FrameState>().RenderPending = true;
}

// Re-record the command buffer for `render_request`. `force_full` records the full buffer even when the request wouldn't.
void RecordViewportFrame(entt::registry &r, entt::entity viewport, RenderRequest render_request, bool force_full = false) {
    auto &resources = r.ctx().get<ViewportRenderResources>();
    if (render_request == RenderRequest::ReRecord || force_full) {
        RecordRenderCommandBuffer(r, viewport, *resources.RenderCommandBuffer);
        resources.RecordedPhase = RenderPhase::Full;
    } else if (render_request == RenderRequest::ReRecordSilhouette && r.ctx().get<const DrawState>().MainDrawCount > 0) {
        RecordRenderCommandBuffer(r, viewport, *resources.RenderCommandBuffer, DrawListUse::SilhouetteOnly);
        resources.RecordedPhase = RenderPhase::Full;
    }
}

// Take the pending render request, leaving none pending.
RenderRequest TakeRenderRequest(entt::registry &r) {
    return std::exchange(r.ctx().get<PendingRenderRequest>().Value, RenderRequest::None);
}

// Drain component events, resize as needed, and record the frame.
// Returns false (skipping the record) when the viewport has no non-zero extent yet.
// `force_full` records the full buffer regardless of the requested region.
bool AdvanceAndRecord(entt::registry &r, entt::entity viewport, bool force_full) {
    ProcessComponentEvents(r, viewport);
    if (!ViewportImageReady(r)) return false;
    RecordViewportFrame(r, viewport, TakeRenderRequest(r), force_full);
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

// Render the frame with motion blur across the shutter (centered on the current frame): each step
// renders once and blurs along its own screen motion, and several steps average together. Overlays
// stay sharp over the blur. Restores the settled frame afterward.
void RenderMotionBlurredFrame(entt::registry &r, entt::entity viewport) {
    const auto &vk = r.ctx().get<const VulkanResources>();
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

    const auto main_cb = *resources.RenderCommandBuffer;
    auto &buffers = r.ctx().get<GpuBuffers>();
    // Every blurred frame moves the scene under any selection data, recorded or not.
    r.ctx().get<DrawState>().SelectionStale = true;

    // Allocate the blur targets on first use, then point every descriptor that reaches them at the
    // new images. They are lazy, so these slots hold fallbacks until now.
    if (pipelines.Main.EnsureMotionBlurResources(vk.Device, vk.PhysicalDevice)) {
        auto &slots = r.ctx().get<DescriptorSlots>();
        const auto &sel_slots = r.ctx().get<const SelectionSlots>();
        const auto &main = pipelines.Main;
        // The tile grid the targets were built around decides how many entries the table holds.
        buffers.ResizeMotionBlurTileIndirection(main.MotionBlur->TileExtent);
        const auto accum = main.MotionBlurAccumSamplerInfo();
        const auto velocity = main.VelocitySamplerInfo();
        const auto gather_sampler = main.MotionBlurGatherSamplerInfo();
        const auto tile_image = main.MotionBlurTileImageInfo();
        const auto tile_indirection = buffers.MotionBlurTileIndirection.GetDescriptor();
        vk.Device.updateDescriptorSets(
            {
                slots.MakeSamplerWrite(sel_slots.MotionBlurAccumSampler, accum),
                slots.MakeSamplerWrite(sel_slots.VelocitySampler, velocity),
                slots.MakeSamplerWrite(sel_slots.MotionBlurGatherSampler, gather_sampler),
                slots.MakeImageWrite(sel_slots.MotionBlurTileImage, tile_image),
                slots.MakeBufferWrite({SlotType::Buffer, sel_slots.MotionBlurTileIndirection}, tile_indirection),
            },
            {}
        );
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
        // so the recording goes stale only when the draw list or the phase changes.
        if (TakeRenderRequest(r) >= RenderRequest::ReRecordSilhouette || resources.RecordedPhase != phase) {
            RecordRenderCommandBuffer(r, viewport, main_cb, DrawListUse::Rebuild, phase);
            resources.RecordedPhase = phase;
        }
        // Land any descriptor updates a pose-capture buffer growth deferred.
        buffers.Ctx.FlushDeferredDescriptorUpdates(vk.Device);
        SubmitRecordedFrame(r);
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
        RecordBlurStepsCommandBuffer(r, viewport, main_cb, step_frames);
        // Not a single-phase recording: any later single-phase render must re-record.
        resources.RecordedPhase = RenderPhase::BlurAccumulate;
        buffers.Ctx.FlushDeferredDescriptorUpdates(vk.Device);
        SubmitRecordedFrame(r);
        WaitForRender(r);
    }

    r.get<PlaybackFrame>(viewport).Value = settled_pf;
    frame_state.RenderPending = false; // All motion blur submits were waited on internally.
}
} // namespace

bool ViewportImageReady(const entt::registry &r) {
    const auto extent = r.ctx().get<const Pipelines>().BuiltColorExtent();
    return extent.width != 0 && extent.height != 0;
}

void SubmitViewport(entt::registry &r, entt::entity viewport, vk::Fence viewport_consumer_fence) {
    const profile::CpuScope scope{"SubmitViewport"};
    // Stash the consumer fence for the resize path to wait on before recreating resources. Cleared after so replay sees none.
    r.ctx().get<ViewportConsumerFence>().Value = viewport_consumer_fence;
    ProcessComponentEvents(r, viewport);
    r.ctx().get<ViewportConsumerFence>().Value = vk::Fence{};
    if (!ViewportImageReady(r)) return;
    auto &frame_state = r.ctx().get<FrameState>();
    if (MotionBlurActive(r, viewport)) {
        // A blurred frame costs several scene evaluations, so only run one when something changed.
        // (Otherwise the frame already on screen is what it would produce.)
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
        r.ctx().get<PendingRenderRequest>().Value = RenderRequest::ReRecord;
    }
    const auto render_request = TakeRenderRequest(r);
    if (render_request == RenderRequest::None) return;
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
    auto &pipelines = r.ctx().emplace<Pipelines>(vc.PhysicalDevice, PipelineContext{vc.Device, slots.GetSetLayout(), slots.GetSet(), slots.GetUboSetLayout(), slots.GetUboSet()});
    profile::Init(vc.Device, vc.PhysicalDevice);
    physics::Init(r);
    RegisterSceneComponentHandlers(r);

    const auto viewport = WireRegistry(r);
    auto &buffers = r.ctx().get<GpuBuffers>();
    // Engine-owned context singletons (process-lifetime). Document state lives in SetupScene.
    r.ctx().emplace<ViewportExtent>();
    r.ctx().emplace<ViewportConsumerFence>();
    r.ctx().emplace<SelectionBitsetRef>(std::span<uint32_t>{buffers.SelectionBitset.Data(), GpuBuffers::SelectionBitsetWords});
    r.ctx().emplace<SelectionSlots>(slots);
    r.ctx().emplace<DrawState>();
    r.ctx().emplace<FrameState>();
    r.ctx().emplace<PendingRenderRequest>();
    const auto &one_shot = r.ctx().emplace<OneShotGpu>(MakeOneShotGpu(vc.Device, vc.QueueFamily));
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
    r.emplace_or_replace<EnabledInteractionModes>(viewport);
    r.emplace_or_replace<OrbitToActive>(viewport);
    r.emplace_or_replace<TransformGizmoState>(viewport);
    physics::ApplySimulationSettings(r, r.emplace_or_replace<PhysicsSimulationSettings>(viewport));

    // Default studio-environment selection.
    r.emplace_or_replace<StudioEnvironment>(viewport, std::string{"forest"});

    // Domain-registered per-scene defaults.
    for (const auto &handler : r.ctx().get<SceneSetupHandlers>().Handlers) handler(r, viewport);
}

void AddDefaultSceneContent(entt::registry &r) {
    // Default scene: a cube, a light, and a camera (startup.blend layout).
    auto &meshes = r.ctx().get<MeshStore>();
    constexpr PrimitiveShape default_shape{primitive::Cuboid{}};
    const auto [mesh_entity, _] = ::AddMesh(r, meshes, meshes.CreateMesh(primitive::CreateMesh(default_shape), {}, {}, true), MeshInstanceCreateInfo{.Name = ToString(default_shape)});
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

    // Reset domain caches keyed by the destroyed entities' ids, before the allocator reset lets the next scene reuse them.
    for (const auto &handler : r.ctx().get<SceneClearHandlers>().Handlers) handler(r);

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
    r.ctx().erase<PendingRenderRequest>();
    r.ctx().erase<DrawState>();
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
    {
        const profile::CpuScope scope{"WaitGpu"};
        WaitFor(*r.ctx().get<const ViewportRenderResources>().RenderFence, vk.Device);
    }
    profile::Resolve(vk.Device);
    r.ctx().get<GpuBuffers>().Ctx.ReclaimRetiredBuffers();
    frame.RenderPending = false;
}
