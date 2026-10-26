#include "viewport/Viewport.h"
#include "mesh/MeshComponents.h"

#include "render/ViewportSubmission.h"
#include <Metal/MTLCommandQueue.hpp>

#include "Panel.h"
#include "Paths.h"
#include "ProcessEvents.h"
#include "Profile.h"
#include "SortUnique.h"
#include "action/Errors.h"
#include "animation/AnimationData.h"
#include "audio/AudioStores.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/SurfaceContact.h"
#include "gizmo/GizmoInteraction.h"
#include "gpu/WorkspaceLights.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshStores.h"
#include "mesh/Primitives.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsStores.h"
#include "physics/PhysicsSystem.h"
#include "physics/PhysicsTypes.h"
#include "procedure/Cook.h"
#include "procedure/NodeTheme.h"
#include "render/GpuSceneState.h"
#include "render/Pipelines.h"
#include "render/RenderStores.h"
#include "render/SceneUpdates.h"
#include "render/Textures.h"
#include "scene/Defaults.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "snapshot/Workspace.h"
#include "state/Registry.h"
#include "timeline/Timeline.h"
#include "timeline/TimelineTheme.h"
#include "ui/InterfaceTheme.h"
#include "viewport/FrameState.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportComponents.h"
#include "viewport/ViewportConsumerFence.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportScene.h"

#include "render/GpuBuffers.h"

Viewport::Viewport(mtl::BufferContext &buffers, mtl::BindlessSet &slots)
    : Samplers(slots),
      SceneViewUBO(buffers, GpuBuffers::ViewUboStride() * (GpuBuffers::MaxBlurSteps + 1)),
      WorkspaceLightsUBO(buffers, sizeof(WorkspaceLights)),
      MeshletCoarseCount(buffers, sizeof(uint32_t), SlotType::Buffer) {}

bool Displayed(const state::Registry &r, state::Entity viewport) {
    return r.get<const ViewportPanel>(viewport).Open && r.get<const Viewport>(viewport).Shown;
}

state::Entity FirstViewport(const state::Registry &r) {
    const auto viewports = r.view<const Viewport>();
    return viewports.empty() ? state::Null : *viewports.begin();
}

state::Entity FindActiveViewport(const state::Registry &r) {
    const auto *active = r.try_get<const ActiveViewport>(state::Document);
    const auto *live = active ? r.try_get<const Viewport>(active->Viewport) : nullptr;
    if (live && Displayed(r, active->Viewport)) return active->Viewport;
    const auto viewports = r.view<const Viewport>();
    const auto displayed = std::ranges::find_if(viewports, [&](state::Entity viewport) { return Displayed(r, viewport); });
    return displayed != viewports.end() ? *displayed : FirstViewport(r);
}

state::Entity SceneOf(const state::Registry &r, state::Entity viewport) {
    return ResolveScene(r, r.get<const ViewportScene>(viewport).Scene);
}

state::Entity ActingScene(const state::Registry &r) {
    const auto viewport = FindActiveViewport(r);
    return viewport == state::Null ? ResolveScene(r, state::Null) : SceneOf(r, viewport);
}

std::vector<state::Entity> ShownScenes(const state::Registry &r) {
    std::vector<state::Entity> scenes;
    for (const auto viewport : r.view<const Viewport>()) {
        if (Displayed(r, viewport)) scenes.push_back(SceneOf(r, viewport));
    }
    if (scenes.empty()) scenes.push_back(ActingScene(r));
    SortUnique(scenes);
    return scenes;
}

DisplayUnion VisibleDisplays(const state::Registry &r) {
    DisplayUnion displays;
    for (const auto viewport : r.view<const Viewport>()) {
        if (!Displayed(r, viewport)) continue;
        const auto &display = r.get<const ViewportDisplay>(viewport);
        const bool overlays = display.ShowOverlays, material = !WorkbenchShading(display.ViewportShading);
        displays.Overlays |= overlays;
        displays.Bones |= overlays && display.Overlays.Bones;
        displays.VertexNormals |= overlays && display.Selection.VertexNormals;
        displays.FaceNormals |= overlays && display.Selection.FaceNormals;
        displays.BoundingBoxes |= display.Selection.BoundingBoxes;
        displays.Wireframe |= display.ViewportShading == ViewportShadingMode::Wireframe;
        displays.AllMaterialShaded &= material;
        if (material && display.DebugChannel != DebugChannel::None) displays.DebugChannels |= 1u << uint32_t(display.DebugChannel);
    }
    return displays;
}

const PBRViewportLighting &ActivePbrLighting(const state::Registry &r, state::Entity viewport, const ViewportDisplay &display) {
    if (display.ViewportShading == ViewportShadingMode::Rendered) return r.get<const RenderedLighting>(viewport).Value;
    return r.get<const MaterialPreviewLighting>(viewport).Value;
}
bool RealTransmission(const state::Registry &r, state::Entity viewport, const ViewportDisplay &display, uint32_t mesh_features) {
    return !WorkbenchShading(display.ViewportShading) && ActivePbrLighting(r, viewport, display).RealTransmission && HasFeature(mesh_features, PbrFeature::Transmission);
}

void RequestFrame(state::Registry &r, state::Entity viewport) {
    auto &live = r.edit<Viewport>(viewport);
    live.RenderRequested = live.OcclusionStale = true;
}

template<> WorkspaceLights ViewDefault<WorkspaceLights>() { return Defaults::WorkspaceLights; }

namespace {

template<typename C> void EmplaceAbsent(state::Registry &r, state::Entity e) {
    if (!r.all_of<C>(e)) r.emplace<C>(e, ViewDefault<C>());
}

// A viewport entity holds its workspace values and its live render state.
void EmplaceViewport(state::Registry &r, state::Entity viewport) {
    snapshot::ReturnRetained(r, viewport);
    [&]<typename... Cs>(TypeList<Cs...>) { (EmplaceAbsent<Cs>(r, viewport), ...); }(ViewValues{});
    r.emplace<Viewport>(viewport, r.Resources.get<GpuBuffers>().Ctx, r.Resources.get<mtl::BindlessSet>());
}

// An entity that stops being a viewport loses its workspace values and its live render state.
void RemoveViewport(state::Registry &r, state::Entity viewport) {
    [&]<typename... Cs>(TypeList<Cs...>) { r.remove<Cs..., Viewport>(viewport); }(ViewValues{});
}

// The live UI frame stops sampling the viewport's images before they go.
void WaitForViewportConsumer(state::Registry &r, state::Entity) {
    if (auto *consumer = r.Resources.get<const ViewportConsumerFence>().Value) consumer->waitUntilCompleted();
}

void RegisterViewportHandlers(state::Registry &r) {
    r.on_construct<ViewportScene, &EmplaceViewport>();
    r.on_destroy<ViewportScene, &RemoveViewport>();
    r.on_destroy<Viewport, &WaitForViewportConsumer>();
}

void ApplyPresentation(ViewportDisplay &display, MaterialPreviewLighting &preview, RenderedLighting &rendered, PresentationLighting lighting) {
    const auto mode = lighting == PresentationLighting::MaterialPreview ? ViewportShadingMode::MaterialPreview : ViewportShadingMode::Rendered;
    display.ViewportShading = display.FillMode = mode;
    display.ShowOverlays = false;
    switch (lighting) {
        case PresentationLighting::MaterialPreview:
            preview.Value.WorldOpacity = 1.f;
            break;
        case PresentationLighting::SceneImageLight:
            rendered.Value.BackgroundBlur = 0.f;
            break;
        case PresentationLighting::SceneLights:
            rendered.Value.UseSceneWorld = false;
            rendered.Value.WorldOpacity = 1.f;
            break;
    }
}
} // namespace

void TakePendingViewSetup(state::Registry &r, state::Entity viewport) {
    const auto *setup = r.try_get<const PendingViewSetup>(state::Document);
    if (!setup) return;
    auto camera = r.get<const ViewCameraState>(viewport);
    auto display = r.get<const ViewportDisplay>(viewport);
    auto preview = r.get<const MaterialPreviewLighting>(viewport);
    auto rendered = r.get<const RenderedLighting>(viewport);
    if (setup->WorldOpacity) rendered.Value.WorldOpacity = *setup->WorldOpacity;
    SetLookThrough(r, camera, setup->LookThroughEntity);
    if (setup->Presentation) ApplyPresentation(display, preview, rendered, *setup->Presentation);
    r.remove<PendingViewSetup>(state::Document);
    r.replace_if_changed(viewport, camera);
    r.replace_if_changed(viewport, display);
    r.replace_if_changed(viewport, preview);
    r.replace_if_changed(viewport, rendered);
}

namespace {
RenderView CurrentView(const state::Registry &r, state::Entity viewport) {
    return {r.get<const ViewCameraState>(viewport).Active, RenderExtentPx(r, viewport), r.get<const ViewportDisplay>(viewport)};
}

// The targets follow the view's extent and the lighting, and a scene or view that differs from the rendered one requests a frame.
// Returns the view the viewport's frame renders.
RenderView StepViewport(state::Registry &r, state::Entity viewport) {
    const auto view = CurrentView(r, viewport);
    const bool targets_changed = SyncViewportRenderResources(r, viewport, uvec2{view.Extent.x, view.Extent.y});
    const auto &live = r.get<const Viewport>(viewport);
    if (targets_changed || live.RenderedScene != SceneOf(r, viewport) || live.FrameView != view) RequestFrame(r, viewport);
    return view;
}

// Records and submits the viewport's frame at the view.
void SubmitFrame(state::Registry &r, state::Entity viewport, const RenderView &view) {
    WriteViewUniforms(r, viewport, view);
    RecordAndSubmitFrame(r, viewport);
}

// Motion blur applies in MaterialPreview/Rendered while playing, scrubbing, or capturing.
bool MotionBlurActive(const state::Registry &r, state::Entity viewport) {
    const auto &display = r.get<const ViewportDisplay>(viewport);
    if (display.ViewportShading != ViewportShadingMode::MaterialPreview && display.ViewportShading != ViewportShadingMode::Rendered) return false;
    const auto &frame_state = r.Resources.get<const FrameState>();
    if (!display.MotionBlur && !frame_state.Capturing) return false;
    return r.get<const TimelinePlayback>(state::Document).Playing || frame_state.Scrubbing || frame_state.Capturing;
}

constexpr bool BlurredPhase(RenderPhase phase) { return phase == RenderPhase::BlurFast || phase == RenderPhase::BlurAccumulate; }

// The shutter samples a blurred frame evaluates.
struct BlurSchedule {
    bool Fast;
    uint32_t Count;
    float Lo, Hi;
    bool operator==(const BlurSchedule &) const = default;
};

BlurSchedule ScheduleOf(const state::Registry &r, state::Entity viewport) {
    const auto &display = r.get<const ViewportDisplay>(viewport);
    const auto mb = EffectiveMotionBlur(display);
    const bool fast = mb.Method == MotionBlurMethod::Fast;
    const auto &range = r.get<const TimelineRange>(state::Document);
    const int current_frame = r.get<const TimelinePlayback>(state::Document).CurrentFrame;
    // Match Blender's centered shutter and clamp it to the timeline.
    const float half = mb.Shutter * 0.5f;
    return {
        .Fast = fast,
        .Count = fast ? 2u : MotionBlurSteps(display),
        .Lo = std::max(float(range.StartFrame), float(current_frame) - half),
        .Hi = std::min(float(range.EndFrame), float(current_frame) + half),
    };
}

// Renders each viewport sharing the schedule from its shutter samples, evaluating each sample once, with sharp overlays, and restores the current frame afterward.
void RenderMotionBlurredFrames(state::Registry &r, BlurSchedule schedule, std::span<const state::Entity> group) {
    const mtl::AutoreleaseScope native_scope;
    const auto &ctx = r.Resources.get<const mtl::Context>();
    const auto [fast, count, lo, hi] = schedule;
    const auto &range = r.get<const TimelineRange>(state::Document);
    const int current_frame = r.get<const TimelinePlayback>(state::Document).CurrentFrame;
    const float settled_pf = r.get<const PlaybackFrame>(state::Document).Value;

    const float last_sample = fast ? hi : lo + (hi - lo) * (1.f - 0.5f / float(count));
    physics::BakeThrough(r, int(std::ceil(last_sample)), range.Fps);

    auto &buffers = r.Resources.get<GpuBuffers>();
    auto &slots = r.Resources.get<mtl::BindlessSet>();
    for (const auto viewport : group) {
        auto &live = r.edit<Viewport>(viewport);
        auto &targets = live.Targets;
        if (!targets.EnsureMotionBlurResources(ctx, fast)) continue;
        const auto sampled = targets.MotionBlurOutputSampler();
        slots.SetSampler({SlotType::Sampler, live.Samplers.MotionBlurOutput}, sampled.Texture, sampled.Sampler);
        const auto velocity = targets.Nearest(fast ? &targets.MotionBlur->VelocityImage : nullptr);
        slots.SetSampler({SlotType::Sampler, live.Samplers.Velocity}, velocity.Texture, velocity.Sampler);
    }

    // Evaluate animation, physics, and each animated look-through camera at `pf` into mapped pose buffers.
    const auto evaluate_at = [&](float pf, EventPass pass) {
        {
            const profile::CpuScope scope{"SamplePoses"};
            physics::SamplePosesAtFrame(r, pf);
        }
        r.edit<PlaybackFrame>(state::Document).Value = pf;
        ProcessComponentEvents(r, pass);
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
        // Consecutive samples identical in every viewport need one render, including static scenes and clamped shutters.
        if (!fast && !sample_weights.empty()) {
            const auto &previous = buffers.BlurPoses[sample_weights.size() - 1];
            const auto previous_offset = GpuBuffers::SceneViewUboOffset(sample_weights.size());
            const bool same_views = std::ranges::all_of(group, [&](state::Entity viewport) {
                auto current_view = SceneViewOf(r, viewport, CurrentView(r, viewport));
                previous.ApplyTo(current_view, ShownSceneIndex(r, SceneOf(r, viewport)));
                const auto &previous_view = *reinterpret_cast<const SceneViewUBO *>(r.get<const Viewport>(viewport).SceneViewUBO.Contents().data() + previous_offset);
                return std::memcmp(&previous_view, &current_view, sizeof(current_view)) == 0; // NOLINT(bugprone-suspicious-memory-comparison)
            });
            if (same_views &&
                same_buffer(previous.Transforms, buffers.Instances.TransformBuffer) &&
                same_buffer(previous.ArmatureDeform, buffers.ArmatureDeformBuffer.Buffer) &&
                same_buffer(previous.MorphWeights, buffers.MorphWeightBuffer.Buffer) &&
                std::ranges::equal(previous.Lights, buffers.Lights, same_buffer)) {
                ++sample_weights.back();
                continue;
            }
        }
        const uint32_t instance = uint32_t(sample_weights.size()) + 1u;
        if (buffers.BlurPoses.size() < instance) buffers.BlurPoses.emplace_back(buffers.Ctx);
        auto &pose = buffers.BlurPoses[instance - 1u];
        buffers.CaptureRenderPose(pose);
        for (const auto viewport : group) {
            auto view = SceneViewOf(r, viewport, CurrentView(r, viewport));
            pose.ApplyTo(view, ShownSceneIndex(r, SceneOf(r, viewport)));
            r.edit<Viewport>(viewport).SceneViewUBO.Update(BytesOf(view), GpuBuffers::SceneViewUboOffset(instance));
        }
        sample_weights.push_back(1u);
    }
    evaluate_at(float(current_frame), EventPass::Render);
    r.edit<PlaybackFrame>(state::Document).Value = settled_pf;
    ReserveFrameScratch(r);
    for (const auto viewport : group) {
        WriteViewUniforms(r, viewport, CurrentView(r, viewport));
        auto *command_buffer = ctx.Queue->commandBuffer();
        if (fast) RecordRenderCommandBuffer(r, viewport, command_buffer, RenderPhase::BlurFast);
        else RecordBlurStepsCommandBuffer(r, viewport, command_buffer, sample_weights);
        SubmitRecordedFrame(r, viewport, command_buffer, fast ? RenderPhase::BlurFast : RenderPhase::BlurAccumulate);
    }
    WaitForRender(r);
}
} // namespace

void SubmitViewports(state::Registry &r) {
    const profile::CpuScope scope{"SubmitViewports"};
    // Each displayed viewport with the view its frame renders.
    std::vector<std::pair<state::Entity, RenderView>> displayed;
    for (const auto viewport : r.view<const Viewport>()) {
        if (Displayed(r, viewport)) displayed.emplace_back(viewport, StepViewport(r, viewport));
    }

    std::vector<state::Entity> blurring;
    for (const auto &[viewport, _] : displayed) {
        auto &live = r.edit<Viewport>(viewport);
        if (!ViewportImageReady(live)) continue;
        if (MotionBlurActive(r, viewport)) {
            // A blurred frame costs several scene evaluations, so it renders only when something changed.
            if (live.RenderRequested) blurring.push_back(viewport);
        } else if (BlurredPhase(live.RecordedPhase)) {
            // Blur ended (playback stopped, or the playhead was released), so a sharp frame replaces the blurred one.
            live.RenderRequested = true;
        }
    }
    if (!blurring.empty()) {
        // The shutter samples request every viewport's frame, and the evaluation they restore leaves the other viewports' requests as they were.
        std::vector<std::tuple<state::Entity, bool, bool>> requests;
        for (const auto [viewport, live] : r.view<const Viewport>().each()) requests.emplace_back(viewport, live.RenderRequested, live.OcclusionStale);
        for (auto remaining = blurring; !remaining.empty();) {
            const auto schedule = ScheduleOf(r, remaining.front());
            const auto shared = std::ranges::partition(remaining, [&](state::Entity viewport) { return ScheduleOf(r, viewport) != schedule; });
            const std::vector<state::Entity> group{shared.begin(), shared.end()};
            remaining.erase(shared.begin(), shared.end());
            RenderMotionBlurredFrames(r, schedule, group);
        }
        for (const auto &[viewport, requested, stale] : requests) {
            auto &live = r.edit<Viewport>(viewport);
            if (std::ranges::contains(blurring, viewport)) {
                live.RenderRequested = live.OcclusionStale = false;
            } else {
                live.RenderRequested = requested;
                live.OcclusionStale = stale;
            }
        }
    }
    std::vector<std::pair<state::Entity, RenderView>> sharp;
    for (const auto &[viewport, view] : displayed) {
        const auto &live = r.get<const Viewport>(viewport);
        if (live.RenderRequested && ViewportImageReady(live) && !MotionBlurActive(r, viewport)) sharp.emplace_back(viewport, view);
    }
    ReserveFrameScratch(r);
    for (const auto &[viewport, view] : sharp) SubmitFrame(r, viewport, view);
}

void InitEngine(state::Registry &r) {
    const auto &ctx = r.Resources.get<const mtl::Context>();
    InitRenderStoreContext(r, ctx);
    auto &slots = r.Resources.get<mtl::BindlessSet>();
    r.Resources.emplace<mtl::LibraryCache>(ctx, Paths::Shaders(), Paths::UserData() / "cache" / "Pipelines.mtl4a", Paths::Base() / "cache" / "Pipelines.mtl4a");
    GetMeshPipelines(r).PrewarmAsync();
    physics::Init(r);
    RegisterSceneComponentHandlers(r);
    snapshot::BindRetention(r);
    RegisterViewportHandlers(r);

    RegisterMeshStoreHandlers(r);
    RegisterProcedureHandlers(r);
    RegisterAudioStoreHandlers(r);
    RegisterPhysicsStoreHandlers(r);
    InitEntityNames(r);
    RegisterRenderStoreHandlers(r);
    [[maybe_unused]] const auto document = r.create();
    assert(document == state::Document);
    auto &meshes = r.Resources.emplace<MeshStore>(InitRenderStores(r));
    auto &buffers = r.Resources.get<GpuBuffers>();
    buffers.Render = &meshes.Render();
    r.Resources.emplace<action::Errors>();
    InitDefaultMaterial(r);
    // These engine resources outlive documents.
    r.Resources.emplace<ViewportConsumerFence>();
    const auto &sel_slots = r.Resources.emplace<SelectionSlots>(slots);
    // Object picking grows on demand and refreshes its bindings; element picking uses fixed buffers.
    slots.SetBuffer({SlotType::Buffer, sel_slots.ObjectPickKey}, *buffers.ObjectPickKeys);
    slots.SetBuffer({SlotType::Buffer, sel_slots.ElementPickKey}, *buffers.ElementPickKey);
    slots.SetBuffer({SlotType::Buffer, sel_slots.ElementPickId}, *buffers.ElementPickId);
    slots.SetBuffer({SlotType::Buffer, sel_slots.ObjectPickSeenBits}, *buffers.ObjectPickSeenBitset);
    slots.SetBuffer({SlotType::Buffer, sel_slots.ObjectBoxBitset}, *buffers.ObjectBoxBitset);
    r.Resources.emplace<GpuSceneState>();
    r.Resources.emplace<FrameState>();
    r.Resources.emplace<ViewportRenderResources>();
    r.Resources.emplace<PanelsState>();

    auto &environments = r.Resources.get<EnvironmentStore>();
    auto &textures = r.Resources.get<TextureStore>();
    const auto images_dir = Paths::Res() / "images";
    environments.BrdfLutSlot = QueueLutTexture(textures, slots, images_dir / "lut_ggx.png", "DefaultGGXBRDFLUT");
    environments.SheenELutSlot = QueueLutTexture(textures, slots, images_dir / "lut_sheen_E.png", "DefaultSheenELUT");
    environments.CharlieLutSlot = QueueLutTexture(textures, slots, images_dir / "lut_charlie.png", "DefaultCharlieLUT");
    // Blender's default world background color (linear RGB), a flat ambient-only IBL when no scene world is provided.
    environments.EmptySceneWorld = BuildFlatColorEnvironment(ctx, slots, vec3{0.05f}, "EmptySceneWorld");
    // SceneWorld uses this default until reactive EXT_lights_image_based loading replaces it.
    environments.SceneWorld = {.Ibl = MakeIblSamplers(environments.EmptySceneWorld, environments), .Name = environments.EmptySceneWorld.Name};

    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator{images_dir / "studiolights" / "world", ec}) {
        if (entry.path().extension() == ".hdr") {
            environments.Hdris.emplace_back(HdriEntry{.Name = entry.path().stem().string(), .Path = entry.path()});
        }
    }
    std::ranges::sort(environments.Hdris, {}, &HdriEntry::Name);
}

void SetupDocument(state::Registry &r) {
    r.emplace_or_replace<ActiveViewport>(state::Document);
    r.emplace_or_replace<Interaction>(state::Document);
    r.emplace_or_replace<EditMode>(state::Document);
    r.emplace_or_replace<ViewportTheme>(state::Document, Defaults::ViewportTheme);
    r.emplace_or_replace<procedure::NodeTheme>(state::Document);
    r.emplace_or_replace<InterfaceTheme>(state::Document);
    r.emplace_or_replace<TimelineTheme>(state::Document);
    r.emplace_or_replace<EnabledInteractionModes>(state::Document);
    r.emplace_or_replace<TransformGizmoState>(state::Document);
    r.emplace_or_replace<AnisotropicFilter>(state::Document);
    physics::ApplySimulationSettings(r, r.emplace_or_replace<PhysicsSimulationSettings>(state::Document));

    r.emplace_or_replace<AudioOutputMix>(state::Document);
    r.emplace_or_replace<Striker>(state::Document);
    r.emplace_or_replace<ModalSoundControls>(state::Document);
    r.emplace_or_replace<PlaybackFrame>(state::Document);
    r.emplace_or_replace<LastEvaluatedFrame>(state::Document);
    r.emplace_or_replace<TimelineView>(state::Document);
    r.emplace_or_replace<TimelineRange>(state::Document);
    r.emplace_or_replace<TimelinePlayback>(state::Document);
    r.emplace_or_replace<TimelineNavigation>(state::Document);
    r.emplace_or_replace<Animations>(state::Document);
    r.emplace_or_replace<ShadeSmoothAngle>(state::Document);
    r.emplace_or_replace<BoxSelectState>(state::Document);
    r.emplace_or_replace<GizmoInteraction>(state::Document);
    // The empty scene's selection aggregates, which the settle pass keeps current.
    r.emplace_or_replace<TransformRoots>(state::Document);
    r.emplace_or_replace<EditPrimaries>(state::Document);
    r.emplace_or_replace<SelectionFlags>(state::Document);
    r.emplace_or_replace<SelectedKeyframes>(state::Document);
    r.emplace_or_replace<OutlinerRows>(state::Document);
    SurfaceSetupDocument(r);
}

void AddDefaultSceneContent(state::Registry &r) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto created = CreateMesh(r, {.Data = primitive::CreateMesh(primitive::Cuboid{}), .FlatShaded = true});
    ::AddMesh(r, created.StoreId, MeshInstanceCreateInfo{.Name = "Cube"});

    // Match Blender's startup scene in its Z-up, negative-Y-forward frame.
    constexpr vec3 LightLoc{4.07625, 1.00545, 5.90386}, CameraLoc{7.358891, -6.925791, 4.958309}, CameraEulerXYZ{1.109319, 0, 0.815801};
    constexpr float Lens{50}, SensorX{36}, RenderW{16}, RenderH{9};
    // Blender Z-up -> MeshEditor Y-up is a -90° rotation about +X: (x, y, z) -> (x, z, -y)
    const auto to_y_up_pos = [](vec3 v) { return vec3{v.x, v.z, -v.y}; };
    const quat to_y_up_rot = AngleAxis(-std::numbers::pi_v<float> / 2.f, vec3{1, 0, 0});
    // Matches Blender glTF exporter (cameras.py / yvof_blender_to_gltf): horizontal fit since render aspect > sensor aspect
    const float hfov = 2 * std::atan(SensorX / (2 * Lens));
    const float yfov = 2 * std::atan(std::tan(hfov * 0.5) * RenderH / RenderW);

    ::AddLight(r, meshes, {.Name = "Light", .Transform = {.P = to_y_up_pos(LightLoc)}, .Select = MeshInstanceCreateInfo::SelectBehavior::None});
    ::AddCamera(r, meshes, {.Name = "Camera", .Transform = {.P = to_y_up_pos(CameraLoc), .R = to_y_up_rot * FromEulerAngles(CameraEulerXYZ)}, .Select = MeshInstanceCreateInfo::SelectBehavior::None}, Perspective{.FieldOfViewRad = yfov, .FarClip = 1000, .NearClip = DefaultPerspectiveNearClip});
}

void DeinitViewport(state::Registry &r) {
    r.Resources.erase<ViewportRenderResources>();
    r.Resources.erase<SelectionSlots>();
    r.clear<Viewport>();
    r.Resources.erase<FrameState>();
    r.Resources.erase<GpuSceneState>();
    physics::Deinit(r);
    r.Resources.erase<MeshPipelines>();
    r.Resources.erase<Pipelines>();
    if (r.valid(state::Document)) r.destroy(state::Document);
    r.Resources.erase<snapshot::Retention>();
    // The store releases every mesh in one batch, and MeshHandle destruction still needs it present.
    Reset(r.Resources.get<MeshStore>(), r);
    r.clear<MeshHandle>();
    DeinitTextureStores(r);
    r.Resources.erase<MeshStore>();
    DeinitRenderStores(r);
    DeinitEntityNames(r);
    DeinitRenderStoreContext(r);
}

void PresentViewport(state::Registry &r, state::Entity viewport) {
    ProcessComponentEvents(r, EventPass::Settle);
    const auto view = StepViewport(r, viewport);
    if (!ViewportImageReady(r.get<const Viewport>(viewport))) return;
    if (MotionBlurActive(r, viewport)) {
        const state::Entity group[]{viewport};
        RenderMotionBlurredFrames(r, ScheduleOf(r, viewport), group);
    } else {
        SubmitFrame(r, viewport, view);
        WaitForRender(r);
    }
}
