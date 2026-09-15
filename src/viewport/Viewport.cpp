#include "viewport/Viewport.h"
#include "mesh/MeshComponents.h"
#include "state/Scene.h"

#include "render/ViewportSubmission.h"
#include <Metal/MTLCommandQueue.hpp>

#include "CameraTypes.h"
#include "Paths.h"
#include "ProcessEvents.h"
#include "Profile.h"
#include "action/Errors.h"
#include "Window.h"
#include "animation/AnimationTimeline.h"
#include "audio/AudioStores.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/SurfaceContact.h"
#include "gizmo/GizmoInteraction.h"
#include "mesh/Mesh.h"
#include "mesh/MeshBatch.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshStores.h"
#include "mesh/Primitives.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsStores.h"
#include "physics/PhysicsSystem.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuSceneState.h"
#include "render/MaterialImport.h"
#include "render/Pipelines.h"
#include "render/RenderTargets.h"
#include "render/RenderStores.h"
#include "render/Textures.h"
#include "scene/Defaults.h"
#include "scene/Entity.h"
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
RenderRequest TakeRenderRequest(state::Scene &r) {
    return std::exchange(r.ctx().get<PendingRenderRequest>().Value, RenderRequest::None);
}

SceneUpdate RequestedSceneUpdate(RenderRequest request) { return request == RenderRequest::Rebuild ? SceneUpdate::Rebuild : SceneUpdate::Reuse; }

// Motion blur applies in MaterialPreview/Rendered while playing, scrubbing, or capturing.
bool MotionBlurActive(const state::Scene &r, state::Entity viewport) {
    const auto &display = r.get<const ViewportDisplay>(viewport);
    if (display.ViewportShading != ViewportShadingMode::MaterialPreview && display.ViewportShading != ViewportShadingMode::Rendered) return false;
    const auto &frame_state = r.ctx().get<const FrameState>();
    if (!display.MotionBlur && !frame_state.Capturing) return false;
    return r.get<const TimelinePlayback>(viewport).Playing || frame_state.Scrubbing || frame_state.Capturing;
}

// Renders shutter samples with sharp overlays and restores the current frame afterward.
void RenderMotionBlurredFrame(state::Scene &r, state::Entity viewport) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &targets = r.ctx().get<RenderTargets>();
    auto &resources = r.ctx().get<ViewportRenderResources>();

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
    if (targets.EnsureMotionBlurResources(ctx, fast)) {
        auto &slots = r.ctx().get<mtl::BindlessSet>();
        const auto sampled = targets.MotionBlurOutputSampler();
        slots.SetSampler({SlotType::Sampler, r.ctx().get<const SelectionSlots>().MotionBlurOutputSampler}, sampled.Texture, sampled.Sampler);
        const auto velocity = targets.Nearest(fast ? &targets.MotionBlur->VelocityImage : nullptr);
        slots.SetSampler({SlotType::Sampler, r.ctx().get<const SelectionSlots>().VelocitySampler}, velocity.Texture, velocity.Sampler);
    }

    // Evaluate animation, physics, and an animated look-through camera at `pf` into mapped pose buffers.
    const auto evaluate_at = [&](float pf, EventPass pass) {
        {
            const profile::CpuScope scope{"SamplePoses"};
            physics::SamplePosesAtFrame(r, pf);
        }
        r.edit<PlaybackFrame>(viewport).Value = pf;
        ProcessComponentEvents(r, viewport, pass);
    };

    std::vector<uint32_t> sample_weights;
    sample_weights.reserve(count);
    buffers.BlurPoses.reserve(count);
    const auto same_buffer = [](const mtl::Buffer &a, const mtl::Buffer &b) {
        return a.UsedSize == b.UsedSize && (a.UsedSize == 0 || std::memcmp(a.Contents().data(), b.Contents().data(), a.UsedSize) == 0);
    };
    for (uint32_t i = 0; i < count; ++i) {
        const float time = fast ? (i == 0 ? lo : hi) : lo + (hi - lo) * (float(i) + 0.5f) / float(count);
        evaluate_at(time, EventPass::Sample);
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
    evaluate_at(float(current_frame), EventPass::Render);
    r.edit<PlaybackFrame>(viewport).Value = settled_pf;
    std::ignore = TakeRenderRequest(r);
    auto *command_buffer = ctx.Queue->commandBuffer();
    if (fast) RecordRenderCommandBuffer(r, viewport, command_buffer, SceneUpdate::Rebuild, RenderPhase::BlurFast);
    else RecordBlurStepsCommandBuffer(r, viewport, command_buffer, sample_weights);
    resources.RecordedPhase = fast ? RenderPhase::BlurFast : RenderPhase::BlurAccumulate;
    SubmitRecordedFrame(r, command_buffer);
    WaitForRender(r);
}
} // namespace

void SubmitViewport(state::Scene &r, state::Entity viewport) {
    const profile::CpuScope scope{"SubmitViewport"};
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

state::Entity InitEngine(state::Scene &r) {
    const auto &ctx = r.ctx().get<const mtl::Context>();
    InitRenderStoreContext(r, ctx);
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    r.ctx().emplace<mtl::LibraryCache>(ctx, Paths::Shaders(), Paths::UserData() / "cache" / "Pipelines.mtl4a");
    r.ctx().emplace<RenderTargets>();
    physics::Init(r);
    RegisterSceneComponentHandlers(r);

    RegisterMeshStoreHandlers(r);
    RegisterAudioStoreHandlers(r);
    RegisterPhysicsStoreHandlers(r);
    InitEntityNames(r);
    RegisterRenderStoreHandlers(r);
    const auto viewport = r.create();
    r.ctx().emplace<MeshStore>(InitRenderStores(r));
    auto &buffers = r.ctx().get<GpuBuffers>();
    r.ctx().emplace<action::Errors>();
    InitDefaultMaterial(r, viewport);
    // These engine resources outlive documents.
    r.ctx().emplace<ViewportExtent>();
    r.ctx().emplace<ViewportConsumerFence>();
    const auto &sel_slots = r.ctx().emplace<SelectionSlots>(slots);
    // Object picking grows on demand and refreshes its bindings; element picking uses fixed buffers.
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

    auto &environments = r.ctx().get<EnvironmentStore>();
    auto &textures = r.ctx().get<TextureStore>();
    const auto images_dir = Paths::Res() / "images";
    environments.BrdfLutSlot = QueueLutTexture(textures, slots, images_dir / "lut_ggx.png", "DefaultGGXBRDFLUT");
    environments.SheenELutSlot = QueueLutTexture(textures, slots, images_dir / "lut_sheen_E.png", "DefaultSheenELUT");
    environments.CharlieLutSlot = QueueLutTexture(textures, slots, images_dir / "lut_charlie.png", "DefaultCharlieLUT");
    // Blender's default world background color (linear RGB), a flat ambient-only IBL when no scene world is provided.
    environments.EmptySceneWorld = BuildFlatColorEnvironment(ctx, slots, vec3{0.05f}, "EmptySceneWorld");
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

void SetupScene(state::Scene &r, state::Entity viewport) {
    r.emplace_or_replace<ViewportDisplay>(viewport);
    r.emplace_or_replace<Interaction>(viewport);
    r.emplace_or_replace<EditMode>(viewport);
    r.emplace_or_replace<ViewportTheme>(viewport, Defaults::ViewportTheme);
    r.emplace_or_replace<ViewCamera>(viewport, Defaults::ViewCamera);
    r.emplace_or_replace<MaterialPreviewLighting>(viewport, PBRViewportLighting{false, false, 1.f, 0.f});
    r.emplace_or_replace<RenderedLighting>(viewport, PBRViewportLighting{true, true, 1.f, 0.f});
    r.emplace_or_replace<WorkspaceLights>(viewport, Defaults::WorkspaceLights);
    r.emplace_or_replace<EnabledInteractionModes>(viewport);
    r.emplace_or_replace<OrbitToActive>(viewport);
    r.emplace_or_replace<TransformGizmoState>(viewport);
    physics::ApplySimulationSettings(r, r.emplace_or_replace<PhysicsSimulationSettings>(viewport));

    r.emplace_or_replace<StudioEnvironment>(viewport, std::string{"forest"});
    r.emplace_or_replace<AudioOutputConfig>(viewport);
    r.emplace_or_replace<AudioOutputMix>(viewport);
    r.emplace_or_replace<Striker>(viewport);
    r.emplace_or_replace<ModalSoundControls>(viewport);
    r.emplace_or_replace<PlaybackFrame>(viewport);
    r.emplace_or_replace<LastEvaluatedFrame>(viewport);
    r.emplace_or_replace<AnimationTimelineView>(viewport);
    r.emplace_or_replace<TimelineRange>(viewport);
    r.emplace_or_replace<TimelinePlayback>(viewport);
    r.emplace_or_replace<SelectionXRay>(viewport);
    r.emplace_or_replace<ShadeSmoothAngle>(viewport);
    r.emplace_or_replace<BoxSelectState>(viewport);
    r.emplace_or_replace<GizmoInteraction>(viewport);
    SurfaceSetupScene(r, viewport);
}

void AddDefaultSceneContent(state::Scene &r) {
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

void ClearScene(state::Scene &r, state::Entity viewport) {
    // Clear physics while its components still exist, so the next load isn't tripped by stale entity keys.
    physics::Clear(r);
    ClearMeshes(r, viewport);

    ResetImportedEnvironment(r);

    // Reset imported materials explicitly because bone visualization keeps skinned-scene mesh storage alive.
    ResetImportedTexturesAndMaterials(r);

    // Clear derived light slots so restored persistent lights register from slot zero.
    r.ctx().get<GpuBuffers>().Lights.SetCount<PunctualLight>(0);
    r.ctx().get<GpuBuffers>().PendingLightRemovals.clear();
    // Raw-pixel uploads queued at engine init survive a clear that precedes their materialization.
    std::erase_if(r.ctx().get<TextureStore>().PendingUploads, [](const auto &upload) { return std::holds_alternative<PendingTextureUpload::GltfImageRef>(upload.Source); });
    r.ctx().get<EnvironmentStore>().PendingImport.reset();

    // Destroy instances before the buffer entities they reference.
    for (const auto e : r.view<RenderInstance>() | to<std::vector>()) r.destroy(e);
    for (const auto e : r.view<state::Entity>() | to<std::vector>()) {
        if (e != viewport) r.destroy(e);
    }
    r.destroy(viewport);

    // Reset ordered allocators so scene replay reproduces entity IDs and GPU handles.
    // Bindless allocation is order-independent and requires no reset.
    r.ResetEntities();
    r.ctx().get<MeshStore>().Clear();
    r.ctx().get<GpuBuffers>().ResetSceneArenas();
    r.ctx().get<GpuSceneState>() = {};
    // Disable occlusion until the new scene has produced a depth pyramid.
    if (auto &resources = r.ctx().get<RenderTargets>().Resources) resources->DepthPyramidValid = false;

    [[maybe_unused]] const auto recreated = r.create();
    assert(recreated == viewport);
    SetupScene(r, viewport);
}

void DeinitViewport(state::Scene &r, state::Entity viewport) {
    r.ctx().erase<ViewportRenderResources>();
    r.ctx().erase<SelectionSlots>();
    r.ctx().erase<FrameState>();
    r.ctx().erase<PendingRenderRequest>();
    r.ctx().erase<GpuSceneState>();
    r.ctx().erase<EntityDestroyTracker>();
    physics::Deinit(r);
    r.ctx().erase<MeshPipelines>();
    r.ctx().erase<Pipelines>();
    r.ctx().erase<RenderTargets>();
    if (r.valid(viewport)) r.destroy(viewport);
    // MeshHandle destruction needs the mesh store, and resource owners retire buffers into the render store.
    r.clear<MeshHandle>();
    DeinitTextureStores(r);
    r.ctx().erase<MeshStore>();
    DeinitRenderStores(r);
    DeinitEntityNames(r);
    DeinitRenderStoreContext(r);
}

void PresentViewport(state::Scene &r, state::Entity viewport) {
    ProcessComponentEvents(r, viewport, EventPass::Settle);
    if (!ViewportImageReady(r)) return;
    if (MotionBlurActive(r, viewport)) {
        RenderMotionBlurredFrame(r, viewport);
        r.ctx().get<FrameState>().MotionBlurred = true;
    } else {
        RecordAndSubmitFrame(r, viewport, RequestedSceneUpdate(TakeRenderRequest(r)));
        WaitForRender(r);
    }
}
