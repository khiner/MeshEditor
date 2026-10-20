#include "viewport/Viewport.h"
#include "mesh/MeshComponents.h"

#include "render/ViewportSubmission.h"
#include <Metal/MTLCommandQueue.hpp>

#include "Camera.h"
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
#include "scene/CameraLens.h"
#include "scene/Defaults.h"
#include "scene/SceneGraph.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "state/Registry.h"
#include "timeline/Timeline.h"
#include "timeline/TimelineTheme.h"
#include "ui/InterfaceTheme.h"
#include "viewport/FrameState.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewportConsumerFence.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportScene.h"

#include "render/GpuBuffers.h"

Viewport::Viewport(state::Entity entity, ViewportState &state, mtl::BufferContext &buffers, mtl::BindlessSet &slots)
    : Entity(entity), State(state), Samplers(slots),
      SceneViewUBO(buffers, GpuBuffers::ViewUboStride() * (GpuBuffers::MaxBlurSteps + 1)),
      WorkspaceLightsUBO(buffers, sizeof(WorkspaceLights)),
      MeshletCoarseCount(buffers, sizeof(uint32_t), SlotType::Buffer) {}

DisplayUnion VisibleDisplays(const Viewports &viewports) {
    DisplayUnion displays;
    for (const auto &viewport : viewports.All) {
        if (!viewport.Displayed()) continue;
        const auto &display = viewport.State.Display;
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

namespace {
// Moves the active viewport to the first displayed one when it is not displayed.
void KeepActiveDisplayed(Viewports &viewports) {
    if (viewports.All.empty() || viewports.Active().Displayed()) return;
    const auto displayed = std::ranges::find_if(viewports.All, &Viewport::Displayed);
    if (displayed != viewports.All.end()) viewports.ActiveIndex = size_t(displayed - viewports.All.begin());
}
} // namespace

bool SyncViewports(state::Registry &r) {
    auto &viewports = r.Resources.get<Viewports>();
    const auto entities = SortedEntities(r.view<const ViewportScene>());
    if (std::ranges::equal(viewports.All, entities, {}, &Viewport::Entity)) return false;
    const bool keeps_all = std::ranges::all_of(viewports.All, [&](const Viewport &v) { return std::ranges::binary_search(entities, v.Entity); });
    if (auto *consumer = r.Resources.get<const ViewportConsumerFence>().Value; consumer && !keeps_all) consumer->waitUntilCompleted();
    const auto active = viewports.All.empty() ? state::Null : viewports.Active().Entity;
    std::vector<Viewport> synced;
    synced.reserve(entities.size());
    for (const auto entity : entities) {
        if (auto *existing = viewports.Find(entity)) synced.push_back(std::move(*existing));
        else synced.emplace_back(entity, viewports.States.try_emplace(entity).first->second, r.Resources.get<GpuBuffers>().Ctx, r.Resources.get<mtl::BindlessSet>());
    }
    viewports.All = std::move(synced);
    const auto it = std::ranges::find(viewports.All, active, &Viewport::Entity);
    viewports.ActiveIndex = it != viewports.All.end() ? size_t(it - viewports.All.begin()) : 0;
    KeepActiveDisplayed(viewports);
    return true;
}

state::Entity SceneOf(const state::Registry &r, const Viewport &viewport) {
    const auto *shown = r.try_get<const ViewportScene>(viewport.Entity);
    return ResolveScene(r, shown ? shown->Scene : state::Null);
}

state::Entity ActingScene(const state::Registry &r) {
    const auto &viewports = r.Resources.get<const Viewports>();
    return viewports.All.empty() ? ResolveScene(r, state::Null) : SceneOf(r, viewports.Active());
}

std::vector<state::Entity> ShownScenes(const state::Registry &r) {
    std::vector<state::Entity> scenes;
    for (const auto &viewport : r.Resources.get<const Viewports>().All) {
        if (viewport.Displayed()) scenes.push_back(SceneOf(r, viewport));
    }
    if (scenes.empty()) scenes.push_back(ActingScene(r));
    SortUnique(scenes);
    return scenes;
}

namespace {
void ApplyPresentation(ViewportState &view, PresentationLighting lighting) {
    const auto mode = lighting == PresentationLighting::MaterialPreview ? ViewportShadingMode::MaterialPreview : ViewportShadingMode::Rendered;
    view.Display.ViewportShading = view.Display.FillMode = mode;
    view.Display.ShowOverlays = false;
    switch (lighting) {
        case PresentationLighting::MaterialPreview:
            view.PreviewLighting.Value.WorldOpacity = 1.f;
            break;
        case PresentationLighting::SceneImageLight:
            view.RenderLighting.Value.BackgroundBlur = 0.f;
            break;
        case PresentationLighting::SceneLights:
            view.RenderLighting.Value.UseSceneWorld = false;
            view.RenderLighting.Value.WorldOpacity = 1.f;
            break;
    }
}

// Leaves look-through when the look-through entity is no longer a camera in the view's scene, and otherwise widens the camera's lens to the viewport's aspect.
void FrameLookThrough(const state::Registry &r, Viewport &viewport) {
    auto &state = viewport.State;
    const auto camera = state.LookThroughEntity;
    if (camera == state::Null) return;
    if (!HasLens(r, camera) || !InScene(r, camera, SceneOf(r, viewport))) {
        ClearLookThrough(state);
        return;
    }
    const auto extent = RenderExtentPx(viewport, r.Resources.get<const FrameState>());
    const float aspect = extent.x == 0 || extent.y == 0 ? 1.f : float(extent.x) / float(extent.y);
    state.Camera.Active.Data = WidenForLookThrough(*LensOf(r, camera), aspect);
}

RenderView CurrentView(const state::Registry &r, const Viewport &viewport) {
    return {viewport.State.Camera.Active, RenderExtentPx(viewport, r.Resources.get<const FrameState>())};
}

// The active viewport takes the pending view setup, the targets follow the extent and lighting, and a view or scene that differs from the rendered one requests a frame.
void StepViewport(state::Registry &r, Viewport &viewport, bool active) {
    if (active) TakePendingViewSetup(r, viewport);
    FrameLookThrough(r, viewport);
    const bool targets_changed = SyncViewportRenderResources(r, viewport);
    if (targets_changed || viewport.Rendered != viewport.State || viewport.RenderedScene != SceneOf(r, viewport) || viewport.FrameView != CurrentView(r, viewport)) {
        viewport.RenderRequested = viewport.OcclusionStale = true;
    }
}

// Records and submits the viewport's frame at its current view.
void SubmitFrame(state::Registry &r, Viewport &viewport) {
    WriteViewUniforms(r, viewport, CurrentView(r, viewport));
    RecordAndSubmitFrame(r, viewport);
}

// Motion blur applies in MaterialPreview/Rendered while playing, scrubbing, or capturing.
bool MotionBlurActive(const state::Registry &r, const Viewport &viewport) {
    const auto &display = viewport.State.Display;
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

BlurSchedule ScheduleOf(const state::Registry &r, const Viewport &viewport) {
    const auto &display = viewport.State.Display;
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
void RenderMotionBlurredFrames(state::Registry &r, BlurSchedule schedule, std::span<Viewport *const> group) {
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
    for (auto *viewport : group) {
        auto &targets = viewport->Targets;
        if (!targets.EnsureMotionBlurResources(ctx, fast)) continue;
        const auto sampled = targets.MotionBlurOutputSampler();
        slots.SetSampler({SlotType::Sampler, viewport->Samplers.MotionBlurOutput}, sampled.Texture, sampled.Sampler);
        const auto velocity = targets.Nearest(fast ? &targets.MotionBlur->VelocityImage : nullptr);
        slots.SetSampler({SlotType::Sampler, viewport->Samplers.Velocity}, velocity.Texture, velocity.Sampler);
    }

    // Evaluate animation, physics, and each animated look-through camera at `pf` into mapped pose buffers.
    const auto evaluate_at = [&](float pf, EventPass pass) {
        {
            const profile::CpuScope scope{"SamplePoses"};
            physics::SamplePosesAtFrame(r, pf);
        }
        r.edit<PlaybackFrame>(state::Document).Value = pf;
        ProcessComponentEvents(r, pass);
        // The settle moves every look-through view to its camera, so each widens the camera's lens again.
        for (auto &viewport : r.Resources.get<Viewports>().All) FrameLookThrough(r, viewport);
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
            const bool same_views = std::ranges::all_of(group, [&](const Viewport *viewport) {
                auto current_view = SceneViewOf(r, *viewport, CurrentView(r, *viewport));
                previous.ApplyTo(current_view, ShownSceneIndex(r, SceneOf(r, *viewport)));
                const auto &previous_view = *reinterpret_cast<const SceneViewUBO *>(viewport->SceneViewUBO.Contents().data() + previous_offset);
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
        for (auto *viewport : group) {
            auto view = SceneViewOf(r, *viewport, CurrentView(r, *viewport));
            pose.ApplyTo(view, ShownSceneIndex(r, SceneOf(r, *viewport)));
            viewport->SceneViewUBO.Update(BytesOf(view), GpuBuffers::SceneViewUboOffset(instance));
        }
        sample_weights.push_back(1u);
    }
    evaluate_at(float(current_frame), EventPass::Render);
    r.edit<PlaybackFrame>(state::Document).Value = settled_pf;
    ReserveFrameScratch(r, group);
    for (auto *viewport : group) {
        WriteViewUniforms(r, *viewport, CurrentView(r, *viewport));
        auto *command_buffer = ctx.Queue->commandBuffer();
        if (fast) RecordRenderCommandBuffer(r, *viewport, command_buffer, RenderPhase::BlurFast);
        else RecordBlurStepsCommandBuffer(r, *viewport, command_buffer, sample_weights);
        SubmitRecordedFrame(r, *viewport, command_buffer, fast ? RenderPhase::BlurFast : RenderPhase::BlurAccumulate);
    }
    WaitForRender(r);
}
} // namespace

void TakePendingViewSetup(state::Registry &r, Viewport &viewport) {
    const auto *setup = r.try_get<const PendingViewSetup>(state::Document);
    if (!setup) return;
    auto &state = viewport.State;
    if (setup->WorldOpacity) state.RenderLighting.Value.WorldOpacity = *setup->WorldOpacity;
    if (const auto camera = setup->LookThroughEntity; camera != state::Null && HasLens(r, camera)) {
        const auto &wt = *WorldTransformOf(r, camera);
        SetLookThrough(r, state, camera);
        state.Camera.Active = ViewCamera{wt.P, wt.R, *LensOf(r, camera)};
    }
    if (setup->Presentation) ApplyPresentation(state, *setup->Presentation);
    r.remove<PendingViewSetup>(state::Document);
}

void SubmitViewports(state::Registry &r) {
    const profile::CpuScope scope{"SubmitViewports"};
    auto &viewports = r.Resources.get<Viewports>();
    KeepActiveDisplayed(viewports);
    for (auto &viewport : viewports.All) {
        if (viewport.Displayed()) StepViewport(r, viewport, &viewport == &viewports.Active());
    }
    // A view setup a step took can change the display settings the records derive from, and each displayed viewport binds its scene's buffers.
    if (VisibleDisplays(viewports) != r.Resources.get<const GpuSceneState>().Displays || ShownScenes(r) != r.Resources.get<const GpuBuffers>().Scenes) {
        ProcessComponentEvents(r, EventPass::Settle);
    }

    std::vector<Viewport *> blurring;
    for (auto &viewport : viewports.All) {
        if (!viewport.Displayed() || !ViewportImageReady(viewport)) continue;
        if (MotionBlurActive(r, viewport)) {
            // A blurred frame costs several scene evaluations, so it renders only when something changed.
            if (viewport.RenderRequested) blurring.push_back(&viewport);
        } else if (BlurredPhase(viewport.RecordedPhase)) {
            // Blur ended (playback stopped, or the playhead was released), so a sharp frame replaces the blurred one.
            viewport.RenderRequested = true;
        }
    }
    if (!blurring.empty()) {
        // The shutter samples request every viewport's frame, and the evaluation they restore leaves the other viewports' requests as they were.
        const auto requests = viewports.All | std::views::transform([](const Viewport &v) { return std::pair{v.RenderRequested, v.OcclusionStale}; }) | std::ranges::to<std::vector>();
        for (auto remaining = blurring; !remaining.empty();) {
            const auto schedule = ScheduleOf(r, *remaining.front());
            const auto shared = std::ranges::partition(remaining, [&](const Viewport *v) { return ScheduleOf(r, *v) != schedule; });
            const std::vector<Viewport *> group{shared.begin(), shared.end()};
            remaining.erase(shared.begin(), shared.end());
            RenderMotionBlurredFrames(r, schedule, group);
        }
        for (size_t i = 0; i < viewports.All.size(); ++i) {
            auto &viewport = viewports.All[i];
            if (std::ranges::contains(blurring, &viewport)) viewport.RenderRequested = viewport.OcclusionStale = false;
            else std::tie(viewport.RenderRequested, viewport.OcclusionStale) = requests[i];
        }
    }
    std::vector<Viewport *> sharp;
    for (auto &viewport : viewports.All) {
        if (viewport.Displayed() && viewport.RenderRequested && ViewportImageReady(viewport) && !MotionBlurActive(r, viewport)) sharp.push_back(&viewport);
    }
    ReserveFrameScratch(r, sharp);
    for (auto *viewport : sharp) SubmitFrame(r, *viewport);
}

void InitEngine(state::Registry &r) {
    const auto &ctx = r.Resources.get<const mtl::Context>();
    InitRenderStoreContext(r, ctx);
    auto &slots = r.Resources.get<mtl::BindlessSet>();
    r.Resources.emplace<mtl::LibraryCache>(ctx, Paths::Shaders(), Paths::UserData() / "cache" / "Pipelines.mtl4a", Paths::Base() / "cache" / "Pipelines.mtl4a");
    GetMeshPipelines(r).PrewarmAsync();
    physics::Init(r);
    RegisterSceneComponentHandlers(r);

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
    r.Resources.emplace<Viewports>();
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

    r.emplace_or_replace<AudioOutputConfig>(state::Document);
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
    constexpr PrimitiveShape default_shape{primitive::Cuboid{}};
    const auto created = CreateMesh(r, {.Data = primitive::CreateMesh(default_shape), .FlatShaded = true});
    const auto [mesh_entity, _] = ::AddMesh(r, created.StoreId, MeshInstanceCreateInfo{.Name = ToString(default_shape)});
    r.emplace<PrimitiveShape>(mesh_entity, default_shape);

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
    r.Resources.erase<Viewports>();
    r.Resources.erase<FrameState>();
    r.Resources.erase<GpuSceneState>();
    physics::Deinit(r);
    r.Resources.erase<MeshPipelines>();
    r.Resources.erase<Pipelines>();
    if (r.valid(state::Document)) r.destroy(state::Document);
    // The store releases every mesh in one batch, and MeshHandle destruction still needs it present.
    Reset(r.Resources.get<MeshStore>(), r);
    r.clear<MeshHandle>();
    DeinitTextureStores(r);
    r.Resources.erase<MeshStore>();
    DeinitRenderStores(r);
    DeinitEntityNames(r);
    DeinitRenderStoreContext(r);
}

void PresentViewport(state::Registry &r, Viewport &viewport) {
    ProcessComponentEvents(r, EventPass::Settle);
    StepViewport(r, viewport, &viewport == &r.Resources.get<Viewports>().Active());
    if (!ViewportImageReady(viewport)) return;
    if (MotionBlurActive(r, viewport)) {
        Viewport *const group[]{&viewport};
        RenderMotionBlurredFrames(r, ScheduleOf(r, viewport), group);
    } else {
        SubmitFrame(r, viewport);
        WaitForRender(r);
    }
}
