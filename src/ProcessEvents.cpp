#include "ProcessEvents.h"
#include "action/Errors.h"
#include "action/Nodes.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "physics/ColliderUpdate.h"
#include "render/SceneUpdates.h"

#include "Camera.h"
#include "Profile.h"
#include "SortUnique.h"
#include "TransformMath.h"
#include "action/Selection.h"
#include "animation/AnimationChanges.h"
#include "animation/AnimationData.h"
#include "animation/Evaluate.h"
#include "animation/MorphWeights.h"
#include "armature/Armature.h"
#include "armature/ArmatureChanges.h"
#include "armature/ArmatureComponents.h"
#include "audio/SoundVertexChanges.h"
#include "audio/SoundVertices.h"
#include "editor/AudioIntegration.h"
#include "gltf/GltfScene.h"
#include "gpu/LightRecord.h"
#include "mesh/MeshChanges.h"
#include "mesh/Primitives.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsChanges.h"
#include "physics/PhysicsSystem.h"
#include "procedure/Cook.h"
#include "procedure/Procedure.h"
#include "render/ClusterLodRepair.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/MeshletBoundsRefit.h"
#include "render/PickConstants.h"
#include "render/Pipelines.h"
#include "render/RenderChanges.h"
#include "render/RenderTargets.h"
#include "render/Textures.h"
#include "render/ViewportSubmission.h"
#include "scene/CameraLens.h"
#include "scene/Defaults.h"
#include "scene/SceneChanges.h"
#include "scene/SceneGraph.h"
#include "selection/SelectionChanges.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "timeline/Timeline.h"
#include "viewport/FrameState.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportChanges.h"
#include "viewport/ViewportComponents.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"

#include <format>

using state::On;

using std::ranges::to;
using std::views::iota;

namespace {
using namespace he;

vec3 ComputeElementLocalPosition(const Mesh &mesh, Element element, uint32_t handle, bool canonical) {
    if (element == Element::Vertex) return mesh.GetPosition(canonical ? Mesh::VH{handle} : mesh.VertexAt(handle));
    if (element == Element::Edge) {
        const auto heh = mesh.GetHalfedge(canonical ? Mesh::EH{handle} : mesh.EdgeAt(handle), 0);
        return (mesh.GetPosition(mesh.GetFromVertex(heh)) + mesh.GetPosition(mesh.GetToVertex(heh))) * 0.5f;
    }
    return mesh.CalcFaceCentroid(canonical ? Mesh::FH{handle} : mesh.FaceAt(handle));
}

vec3 ComputeElementWorldPosition(const state::Registry &r, state::Entity instance_entity, Element element, uint32_t handle, bool canonical) {
    const auto mesh = GetMesh(r, r.get<Instance>(instance_entity).Entity);
    const auto &wt = *WorldTransformOf(r, instance_entity);
    return {wt.P + Rotate(wt.R, wt.S * ComputeElementLocalPosition(mesh, element, handle, canonical))};
}

void SetEditMode(state::Registry &r, Element mode) {
    const auto current_mode = r.get<const EditMode>(state::Document).Value;
    if (current_mode == mode) return;

    // Convert edit-mode selections now and other selections on their next edit-mode entry.
    std::vector<state::Entity> editing;
    if (r.get<const Interaction>(state::Document).Mode == InteractionMode::Edit) {
        for (const auto mesh_entity : r.get<const SelectionFlags>(state::Document).Meshes) {
            if (r.all_of<MeshElementSelection>(mesh_entity)) editing.emplace_back(mesh_entity);
        }
    }
    r.patch<EditMode>(state::Document, [mode](auto &edit_mode) { edit_mode.Value = mode; });
    ConvertElementSelections(r, editing, mode);
    for (const auto mesh_entity : editing) action::nodes::CaptureActiveSelection(r, mesh_entity);
}

// Count visible silhouette instances, excluding primary edit instances.
void UpdateSilhouetteWork(state::Registry &r) {
    const auto &primaries = r.get<const EditPrimaries>(state::Document).All;
    auto &buffers = r.Resources.get<GpuBuffers>();
    const auto &meshes = r.Resources.get<const MeshStore>();
    GpuBuffers::MeshletFlagWork work{};
    for (const auto [entity, instance, render_instance] : r.view<const Selected, const Instance, const RenderInstance>(state::Exclude<Hidden>).each()) {
        if (render_instance.BufferIndex == UINT32_MAX || !IsSilhouetteEligible(r, instance.Entity)) continue;
        if (const auto primary = primaries.find(instance.Entity); primary != primaries.end() && primary->second == entity) continue;
        const auto &mb = RecordOf(r, instance.Entity);
        work.Nodes += meshes.LodNodeCount(mb);
        work.Meshlets += meshes.MeshletCount(mb);
    }
    buffers.FlagWork(uint32_t(MeshletInstanceFlag::Silhouette)) = work;
}
} // namespace

void RequestRender(state::Registry &r, bool disoccludes) {
    for (const auto viewport : r.view<const Viewport>()) {
        if (disoccludes) RequestFrame(r, viewport);
        else r.edit<Viewport>(viewport).RenderRequested = true;
    }
}

void ProcessComponentEvents(state::Registry &r, EventPass pass) {
    const bool rendering = pass == EventPass::Sample || pass == EventPass::Render;
    const auto &ctx = r.Resources.get<const mtl::Context>();
    auto &slots = r.Resources.get<mtl::BindlessSet>();
    auto &buffers = r.Resources.get<GpuBuffers>();
    auto &meshes = r.Resources.get<MeshStore>();
    auto &render = meshes.Render();
    auto &textures = r.Resources.get<TextureStore>();
    auto &environments = r.Resources.get<EnvironmentStore>();
    const auto &selection_changes = r.Resources.get<const SelectionChanges>();
    const auto &viewport_changes = r.Resources.get<const ViewportChanges>();
    const auto &scene_changes = r.Resources.get<const SceneChanges>();
    const auto &armature_changes = r.Resources.get<const ArmatureChanges>();
    const auto &animation_changes = r.Resources.get<const AnimationChanges>();
    const auto &mesh_changes = r.Resources.get<const MeshChanges>();
    const auto &render_changes = r.Resources.get<const RenderChanges>();
    const auto &sound_vertex_changes = r.Resources.get<const SoundVertexChanges>();
    const auto &physics_changes = r.Resources.get<const PhysicsChanges>();
    const profile::CpuScope profile_scope{"ProcessEvents"};

    // Submit before host reads of GPU results.
    // BuildMeshlets also submits for allocation and publication counts.
    mtl::ComputeChain chain{buffers.Ctx};
    // Classify: the new meshes' corner classification plans and normal membership.
    const auto classify = [&] { chain.Submit(); };
    // Construct: an edit commit's changed vertices, and the staged meshlet refits the hierarchy repair reads.
    const auto construct = [&] { chain.Submit(); };
    // Select: derived normals, selection aggregates and selection summaries.
    const auto select = [&] { chain.Submit(); };
    // End: every value the frame reads.
    const auto end = [&] { chain.Submit(); };

    const auto request = [&r](bool disoccludes = true) { RequestRender(r, disoccludes); };

    std::unordered_set<state::Entity> bone_state_dirty;

    ProcessObjectRemovals(r);

    // Replan restored procedures without regenerating their restored outputs.
    SettleProcedures(r, pass == EventPass::Frame || pass == EventPass::Settle);
    ProcessObjectRemovals(r);
    // A restored document acts on its active viewport's scene.
    if (pass == EventPass::Restore) r.Resources.get<FrameState>().Scene = ActingScene(r);
    // The active viewport takes an action's view setup before the records derive from its display.
    if (const auto viewport = FindActiveViewport(r); viewport != state::Null) TakePendingViewSetup(r, viewport);
    else r.remove<PendingViewSetup>(state::Document);
    // The records derive from the display settings the visible viewports share.
    auto &scene_state = r.Resources.get<GpuSceneState>();
    const auto displays = VisibleDisplays(r);
    const bool displays_changed = std::exchange(scene_state.Displays, displays) != displays;
    // A newly shown scene's states take each bone's state from the bone pass below.
    const bool scenes_changed = SyncShownScenes(r);
    if (scenes_changed) {
        for (const auto arm : r.view<const ArmatureObject>()) bone_state_dirty.insert(arm);
        request();
    }

    const bool recompiled = std::exchange(r.Resources.get<FrameState>().RecompileShaders, false);
    if (recompiled) {
        r.Resources.get<mtl::LibraryCache>().Clear();
        r.Resources.erase<Pipelines>();
        r.Resources.erase<MeshPipelines>();
        // Recompiled prefilter kernels must regenerate their cached cubemaps.
        RebuildStudioEnvironments(r);
        request();
    }

    // Restore texture data into free recorded slots while preserving textures materialized during import.
    if (!render_changes.MaterializedTextures.empty()) {
        if (const auto *manifest = r.try_get<const MaterializedTextures>(state::Document)) {
            for (const auto &t : manifest->Items) {
                if (!slots.Reserve(SlotType::Sampler, t.SamplerSlot)) continue;
                textures.PendingUploads.emplace_back(PendingTextureUpload{.SamplerSlot = t.SamplerSlot, .Source = PendingTextureUpload::GltfImageRef{t.SourceImageIndex}, .Params = t.Params});
            }
        }
    }
    if (!textures.PendingUploads.empty()) {
        const auto *src = r.try_get<const gltf::SourceAssets>(state::Document);
        static const std::vector<gltf::Image> empty_images;
        const auto &gltf_images = src ? src->Images : empty_images;
        TextureUploadBatch batch{ctx};
        for (const auto &item : textures.PendingUploads) {
            auto entry = MaterializeTextureEntry(r, batch, slots, item, gltf_images, r.Resources.get<const ActiveSamplerAnisotropy>().Value);
            if (!entry) {
                action::Fail(r, std::format("Cannot load texture '{}': {}", item.Params.Name, entry.error()));
                slots.Release({SlotType::Sampler, item.SamplerSlot});
                continue;
            }
            textures.Textures.emplace_back(std::move(*entry));
        }
        SubmitTextureUploadBatch(batch);
        textures.PendingUploads.clear();
    }
    // Rebuild restored EXT-IBL scene resources after a document clear releases their prefiltered cubemap.
    if (!render_changes.SceneWorld.empty()) {
        const auto *src = r.try_get<const gltf::SourceAssets>(state::Document);
        if (src && src->ImageBasedLight && !environments.ImportedSceneWorld && !environments.PendingImport) {
            const auto [diffuse_slot, specular_slot] = AllocateIblCubeSlots(slots);
            environments.PendingImport = PendingEnvironmentImport{*src->ImageBasedLight, diffuse_slot, specular_slot};
            environments.ClearRequested = false;
        }
    }
    // Cancel a stale pending EXT-IBL import before applying a later scene-world clear.
    if (environments.ClearRequested) Reset(environments, r);
    if (auto &pending_env = environments.PendingImport) {
        if (const auto *src = r.try_get<const gltf::SourceAssets>(state::Document)) {
            auto pre = MaterializeEnvironmentImport(r, slots, *pending_env, src->Images);
            if (pre) {
                auto &env = environments;
                if (env.ImportedSceneWorld) {
                    ReleaseCubeSamplerSlot(slots, env.ImportedSceneWorld->DiffuseEnv.SamplerSlot);
                    ReleaseCubeSamplerSlot(slots, env.ImportedSceneWorld->SpecularEnv.SamplerSlot);
                }
                env.ImportedSceneWorld = std::move(*pre);
                env.SceneWorld = {.Ibl = MakeIblSamplers(*env.ImportedSceneWorld, env), .Name = env.ImportedSceneWorld->Name};
            } else {
                action::Fail(r, std::format("Cannot load EXT_lights_image_based '{}': {}", pending_env->Source.Name, pre.error()));
                ReleaseCubeSamplerSlot(slots, pending_env->DiffuseCubeSlot);
                ReleaseCubeSamplerSlot(slots, pending_env->SpecularCubeSlot);
            }
        }
        pending_env.reset();
    }
    // Process pending handlers before consuming the reactive trackers they update.
    if (const auto *pending = r.try_get<const PendingSetEditMode>(state::Document)) {
        const auto mode = pending->Mode;
        r.remove<PendingSetEditMode>(state::Document);
        SetEditMode(r, mode);
    }
    if (const auto *pending = r.try_get<PendingImportMesh>(state::Document)) {
        const auto path = pending->Path;
        const auto info = pending->Info;
        r.remove<PendingImportMesh>(state::Document);
        if (const auto imported = ImportMesh(r, path, info); !imported) action::Fail(r, imported.error());
    }
    // A gesture resolves against the viewport it was emitted in, with its targets sized to the view it saw and that view's camera, which selection, culling and LOD use.
    const auto gesture_viewport = [&](state::Entity emitted_in, const RenderView &view) {
        const auto viewport = r.all_of<Viewport>(emitted_in) ? emitted_in : FirstViewport(r);
        if (SyncViewportRenderResources(r, viewport, uvec2{view.Extent.x, view.Extent.y})) RequestFrame(r, viewport);
        WriteViewUniforms(r, viewport, view);
        // Submit restored geometry and draw records before selection.
        if (r.get<const Viewport>(viewport).RenderRequested) {
            RecordAndSubmitFrame(r, viewport, RenderPhase::Prepare);
            WaitForRender(r);
        }
        return viewport;
    };
    const auto target_px = [&](const RenderView &view, vec2 fraction) {
        const auto extent = uvec2{view.Extent.x, view.Extent.y};
        const auto last = Max(extent, uvec2{1}) - uvec2{1};
        return Min(uvec2{std::lround(fraction.x * float(extent.x)), std::lround(fraction.y * float(extent.y))}, last);
    };
    if (const auto *pending = r.try_get<const PendingEditElementClick>(state::Document)) {
        const auto viewport = gesture_viewport(pending->Viewport, pending->View);
        const auto mouse_px = target_px(pending->View, pending->Mouse);
        const bool toggle = pending->Toggle;
        r.remove<PendingEditElementClick>(state::Document);

        const auto edit_mode = r.get<const EditMode>(state::Document).Value;
        const auto ranges = GetElementRangesForSelected(r);
        const auto hit = RunEditElementClick(r, viewport, ranges, edit_mode, mouse_px, toggle);
        if (!toggle) {
            for (const auto &range : ranges) r.remove<MeshActiveElement>(range.MeshEntity);
        }
        if (hit) {
            const auto mesh_entity = hit->first;
            const auto &summary = meshes.GetSelectionSummary(*DrawnStoreId(r, mesh_entity));
            if (summary.ActiveHandle == InvalidOffset) r.remove<MeshActiveElement>(mesh_entity);
            else r.emplace_or_replace<MeshActiveElement>(mesh_entity, summary.ActiveHandle);
        }
        for (const auto &range : ranges) action::nodes::CaptureActiveSelection(r, range.MeshEntity);
    }
    if (const auto *pending = r.try_get<const PendingBoxSelect>(state::Document)) {
        const auto viewport = gesture_viewport(pending->Viewport, pending->View);
        const auto box_px = std::pair{target_px(pending->View, pending->Box.first), target_px(pending->View, pending->Box.second)};
        const bool additive = pending->Additive;
        r.remove<PendingBoxSelect>(state::Document);

        const auto &interaction = r.get<const Interaction>(state::Document);
        const bool element_mode = interaction.Mode == InteractionMode::Edit && FindArmatureObject(r, FindActiveEntity(r)) == state::Null;
        // Capture the initial selection once for additive box selection.
        if (additive && !r.all_of<AdditiveBoxSelectBaseline>(state::Document)) {
            auto &baseline = r.emplace<AdditiveBoxSelectBaseline>(state::Document);
            if (interaction.Mode == InteractionMode::Pose || IsBoneEditMode(r)) {
                for (const auto e : r.view<BoneSelection>()) baseline.BoneSelections.emplace_back(e, r.get<BoneSelection>(e));
            } else if (!element_mode) {
                for (const auto e : r.view<Selected>()) baseline.SelectedEntities.emplace_back(e);
            }
        }
        if (element_mode) {
            const auto ranges = GetElementRangesForSelected(r);
            if (!additive) {
                for (const auto &range : ranges) r.remove<MeshActiveElement>(range.MeshEntity);
            }
            RunBoxSelectElements(r, viewport, ranges, r.get<const EditMode>(state::Document).Value, box_px, additive);
            for (const auto &range : ranges) action::nodes::CaptureActiveSelection(r, range.MeshEntity);
        } else {
            const bool bone_mode = interaction.Mode == InteractionMode::Pose || IsBoneEditMode(r);
            const auto hits = ResolveHits(r, RunBoxSelect(r, viewport, box_px), bone_mode, true);
            const auto *baseline = additive ? r.try_get<const AdditiveBoxSelectBaseline>(state::Document) : nullptr;
            // Union the hits with the baseline selection and update only entities whose selection changes.
            state::DirtySet target;
            if (bone_mode) {
                std::vector<std::pair<state::Entity, BoneSelection>> parts;
                if (baseline) {
                    for (const auto &[e, sel] : baseline->BoneSelections) {
                        if (!r.valid(e)) continue;
                        target.emplace(e);
                        parts.emplace_back(e, sel);
                    }
                }
                for (const auto &hit : hits) {
                    const auto sel = hit.Part ? BoneSelection::From(*hit.Part) : BoneSelection{};
                    if (target.contains(hit.Entity)) {
                        auto &part = parts[target.Positions[state::Index(hit.Entity)]].second;
                        part = additive ? part | sel : sel;
                    } else {
                        target.emplace(hit.Entity);
                        parts.emplace_back(hit.Entity, sel);
                    }
                }
                std::vector<state::Entity> deselected;
                for (const auto e : r.view<const BoneSelection>()) {
                    if (!target.contains(e)) deselected.emplace_back(e);
                }
                for (const auto e : deselected) r.remove<BoneSelection>(e);
                for (const auto &[e, sel] : parts) {
                    const auto *current = r.try_get<const BoneSelection>(e);
                    if (!current) r.emplace<BoneSelection>(e, sel);
                    else if (*current != sel) r.replace<BoneSelection>(e, sel);
                }
            } else {
                if (baseline) {
                    for (const auto e : baseline->SelectedEntities) {
                        if (r.valid(e)) target.emplace(e);
                    }
                }
                for (const auto &hit : hits) target.emplace(hit.Entity);
                std::vector<state::Entity> deselected;
                for (const auto e : r.view<const Selected>()) {
                    if (!target.contains(e)) deselected.emplace_back(e);
                }
                for (const auto e : deselected) r.remove<Selected>(e);
                for (const auto e : target.Entities) {
                    if (!r.all_of<Selected>(e)) r.emplace<Selected>(e);
                }
            }
        }
    }
    if (const auto *pending = r.try_get<const PendingPick>(state::Document)) {
        const auto viewport = gesture_viewport(pending->Viewport, pending->View);
        const auto mouse_px = target_px(pending->View, pending->Mouse);
        const bool shift = pending->Shift, cycle = pending->Cycle;
        r.remove<PendingPick>(state::Document);

        const bool bone_mode = r.get<const Interaction>(state::Document).Mode == InteractionMode::Pose || IsBoneEditMode(r);
        const auto active = bone_mode ? FindActiveBone(r) : FindActiveEntity(r);
        const auto scale = r.Resources.get<const FrameState>().DisplayFramebufferScale;
        const float render_scale = std::max(scale.x, scale.y);
        const auto radius = std::max(1u, uint32_t(std::lround(float(ObjectSelectRadiusPx) * render_scale)));
        const auto hits = ResolveHits(r, RunObjectPick(r, viewport, mouse_px, radius), bone_mode);
        const auto pick = hits.empty() ? std::optional<SelectionHit>{} : [&]() -> std::optional<SelectionHit> {
            if (!cycle) return hits.front();
            const auto *bs = r.try_get<const BoneSelection>(active);
            auto it = std::ranges::find_if(hits, [&](const SelectionHit &h) { return h.Entity == active && (!h.Part || (bs && bs->Has(*h.Part))); });
            return it != hits.end() && ++it != hits.end() ? *it : hits.front();
        }();
        using namespace action::selection;
        if (pick && shift) {
            if (active == pick->Entity && !bone_mode) Apply(r, ToggleSelected{pick->Entity});
            else if (bone_mode) Apply(r, ExtendBoneActive{pick->Entity, pick->Part, true});
            else Apply(r, ExtendActive{pick->Entity});
        } else if (pick || !shift) {
            if (pick && bone_mode) Apply(r, action::selection::SelectBone{pick->Entity, pick->Part, false});
            else if (pick) Apply(r, action::selection::Select{pick->Entity});
            else Apply(r, DeselectAll{});
        }
    }

    // Evaluate animation before synchronizing render state.
    bool anim_advanced;
    int evaluated_from;
    {
        const auto &range = r.get<const TimelineRange>(state::Document);
        const auto &playback = r.get<const TimelinePlayback>(state::Document);
        auto &pf = r.edit<PlaybackFrame>(state::Document).Value;
        auto &frame_state = r.Resources.get<FrameState>();
        anim_advanced = [&] {
            if (pass == EventPass::Restore) {
                pf = float(playback.CurrentFrame);
                return false;
            }
            if (rendering) return false;
            // A frame set by an action shows for one tick before playback advances.
            if (playback.Playing && pass == EventPass::Frame && playback.CurrentFrame == r.get<const LastEvaluatedFrame>(state::Document).Value) {
                const float step = frame_state.FixedFrameStep ? 1.f : frame_state.DeltaTime * range.Fps;
                pf += playback.Reverse ? -step : step;
                if (pf > float(range.EndFrame)) pf = float(range.StartFrame);
                else if (pf < float(range.StartFrame)) pf = float(range.EndFrame);
                const auto new_frame = int(std::floor(pf));
                if (new_frame != playback.CurrentFrame) r.patch<TimelinePlayback>(state::Document, [&](auto &p) { p.CurrentFrame = new_frame; });
            } else if (!playback.Playing) {
                pf = float(playback.CurrentFrame);
            }
            return playback.CurrentFrame != r.edit<LastEvaluatedFrame>(state::Document).Value || !animation_changes.Edited.empty();
        }();

        evaluated_from = r.edit<LastEvaluatedFrame>(state::Document).Value;
        // Re-evaluate animation after restoring the frame, animation data, or a pose seed transform.
        const bool restore_poses = pass == EventPass::Restore &&
            (playback.CurrentFrame != evaluated_from || AnyChanged(animation_changes.Edited, scene_changes.TransformDirty));
        if (anim_advanced || pass == EventPass::Restore) r.edit<LastEvaluatedFrame>(state::Document).Value = playback.CurrentFrame;
        // Convert 1-based display frames to animation time and preserve fractional motion-blur samples.
        const float frame_seconds = float(std::max(0, playback.CurrentFrame - 1)) / range.Fps;
        const float eval_seconds = pass == EventPass::Sample ? std::max(0.f, pf - 1.f) / range.Fps : frame_seconds;

        // A restore rebuilds only the derived node poses, since history restores every other animated value.
        if (anim_advanced || rendering || restore_poses) animation::Evaluate(r, eval_seconds, pass != EventPass::Restore);
        // Request a frame even when playback leaves animated values unchanged.
        if (anim_advanced) request();
    }

    // Derive render instances after animation updates visibility.
    // A hierarchy or membership change moves nodes between scenes, so it rescopes every instance.
    if (DeriveRenderInstances(r, AnyChanged(scene_changes.Hierarchy, scene_changes.Membership))) request();
    auto sync = SyncModelsBuffers(r);
    if (sync.SlotsChanged) {
        request();
        for (const auto &[mesh_entity, _] : r.get<const EditPrimaries>(state::Document).All) r.emplace_or_replace<MeshDisplayDirty>(mesh_entity);
    }
    if (!sync.NewlyInserted.empty()) {
        const auto records = buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>();
        for (const auto instance_entity : sync.NewlyInserted) {
            if (const auto *force = r.try_get<const VertexForce>(instance_entity)) records[r.get<const RenderInstance>(instance_entity).BufferIndex].ExcitedVertex = force->Vertex;
        }
    }

    // Reconstruct missing derived armature pose state from the canonical pose or rest state.
    std::vector<state::Entity> created_pose_armatures;
    for (const auto [arm_obj_entity, arm_obj_comp] : r.view<const ArmatureObject>().each()) {
        const auto data_entity = arm_obj_comp.Entity;
        const auto *armature = r.try_get<const Armature>(data_entity);
        if (!armature || armature->Bones.empty() || r.all_of<ArmaturePoseState>(data_entity)) continue;
        const auto n = armature->Bones.size();
        r.emplace<ArmaturePoseState>(data_entity, ArmaturePoseState{.BoneUserOffset = std::vector<Transform>(n), .BonePoseWorld = std::vector<mat4>(n, I4)});
        // Compose bone poses from rest and delta, preserving rest scale.
        for (uint32_t i = 0; i < n && i < arm_obj_comp.BoneEntities.size(); ++i) {
            const auto b = arm_obj_comp.BoneEntities[i];
            if (!r.all_of<BoneDelta>(b)) r.emplace<BoneDelta>(b);
            const auto &rest = armature->Bones[i].RestLocal;
            const auto posed = ComposeWithDelta(rest, r.get<const BoneDelta>(b).Value);
            r.emplace_or_replace<PosedLocal>(b, Transform{posed.P, posed.R, rest.S});
        }
        created_pose_armatures.emplace_back(arm_obj_entity);
    }

    {
        uint32_t total_joints = 0;
        std::vector<state::Entity> pending_armatures;
        for (const auto [arm_obj_entity, arm_obj_comp] : r.view<const ArmatureObject>().each()) {
            const auto *pose_state = r.try_edit<ArmaturePoseState>(arm_obj_comp.Entity);
            if (!pose_state || !pose_state->GpuDeformRanges.empty()) continue;
            const auto *armature = r.try_get<const Armature>(arm_obj_comp.Entity);
            if (armature && !armature->Skins.empty()) {
                for (const auto &skin : armature->Skins) total_joints += skin.OrderedJointNodeIndices.size();
                pending_armatures.emplace_back(arm_obj_comp.Entity);
            }
        }
        buffers.ArmatureDeformBuffer.ReserveAdditional(total_joints);
        for (const auto arm_data_entity : pending_armatures) {
            auto &pose_state = r.edit<ArmaturePoseState>(arm_data_entity);
            const auto &armature = r.get<const Armature>(arm_data_entity);
            pose_state.GpuDeformRanges.reserve(armature.Skins.size());
            for (const auto &skin : armature.Skins) {
                pose_state.GpuDeformRanges.emplace_back(buffers.ArmatureDeformBuffer.Allocate(skin.OrderedJointNodeIndices.size()));
            }
        }
        if (!pending_armatures.empty()) {
            for (const auto [_, instance, modifier] : r.view<const Instance, const ArmatureModifier>().each()) {
                if (std::ranges::contains(pending_armatures, modifier.ArmatureEntity)) r.emplace_or_replace<MeshLayoutDirty>(instance.Entity);
            }
            request();
        }
    }

    std::unordered_set<state::Entity> dirty_sound_selection_meshes;

    // Derive classification, normals, and selection for new meshes in one batch.
    if (auto shading = EncodeNewMeshShading(r, chain)) {
        classify();
        DeriveNewMeshShading(r, chain, *shading);
    }
    // Authored corner normals encode against the derived ones.
    if (!r.view<const AuthoredCornerNormals>().empty()) {
        select();
        FinalizeAuthoredShading(r, chain);
    }
    // New, restored and rebuilt meshes and new bone meshes build their meshlets in one batch after the geometry block.
    std::vector<state::Entity> meshlet_meshes{sync.NewMeshEntities}, bone_mesh_entities;
    if (!sync.NewMeshEntities.empty()) request();

    if (!sync.NewExtrasEntities.empty()) {
        static const auto bone = primitive::BoneOctahedron(1.f);
        static const auto &bone_faces = bone.Mesh.FaceCorners;
        static const auto bone_verts = iota(0u, uint32_t(bone.Mesh.Positions.size())) | to<std::vector>();
        static const auto sphere = primitive::BoneSphereDisc();
        static const auto &sphere_faces = sphere.Mesh.FaceCorners;
        static const auto sphere_verts = iota(0u, uint32_t(sphere.Mesh.Positions.size())) | to<std::vector>();

        uint32_t total_face = 0, total_edge = 0;
        for (const auto entity : sync.NewExtrasEntities) {
            if (r.all_of<ArmatureObject>(entity)) {
                total_face += bone_faces.size();
                total_edge += bone.AdjacencyIndices.size();
            } else if (r.all_of<BoneJoint>(entity)) {
                total_face += sphere_faces.size();
                total_edge += sphere.OutlineIndices.size();
            }
        }
        render.ExtrasFaces.ReserveAdditional(total_face);
        render.ExtrasEdges.ReserveAdditional(total_edge);

        for (auto entity : sync.NewExtrasEntities) {
            if (r.all_of<ArmatureObject>(entity)) {
                meshes.SetExtrasIndices(*DrawnStoreId(r, entity), bone_faces, bone.AdjacencyIndices);
                bone_mesh_entities.emplace_back(entity);
            } else if (r.all_of<BoneJoint>(entity)) {
                meshes.SetExtrasIndices(*DrawnStoreId(r, entity), sphere_faces, sphere.OutlineIndices);
                bone_mesh_entities.emplace_back(entity);
            }
        }
        request();
    }

    // Each shown scene's lights are its placed punctual lights not hidden in it, in entity index order.
    if (scenes_changed || sync.SlotsChanged ||
        AnyChanged(render_changes.PunctualLight, render_changes.InstanceVisibility, render_changes.InstanceDestroyed, scene_changes.Hierarchy, scene_changes.Membership)) {
        const auto light_entities = SortedEntities(r.view<const PunctualLight, const Instance, const RenderInstance>());
        // Overlay jobs size each light's wire from its type, range and cone.
        const auto same_shape = [](const LightRecord &a, const LightRecord &b) {
            return a.Type == b.Type && a.Range == b.Range && a.InnerConeCos == b.InnerConeCos && a.OuterConeCos == b.OuterConeCos;
        };
        std::vector<LightRecord> records;
        for (uint32_t i = 0u; i < buffers.Scenes.size(); ++i) {
            records.clear();
            for (const auto entity : light_entities) {
                const auto slot = r.get<const RenderInstance>(entity).BufferIndex;
                if (slot == UINT32_MAX || HiddenIn(r, entity, buffers.Scenes[i])) continue;
                const auto &light = r.get<const PunctualLight>(entity);
                records.push_back({
                    .TransformSlotOffset = {buffers.Instances.TransformBuffer.Slot, slot},
                    .Range = light.Range,
                    .Color = light.Color,
                    .Intensity = light.Intensity,
                    .InnerConeCos = std::cos(light.InnerConeAngle),
                    .OuterConeCos = std::cos(light.OuterConeAngle),
                    .Type = light.Type,
                });
            }
            auto &lights = buffers.Lights[i];
            if (!std::ranges::equal(lights.GetSpan<LightRecord>(), records, same_shape)) scene_state.OverlayJobsDirty = true;
            std::ranges::copy(records, lights.SetCount<LightRecord>(uint32_t(records.size())).begin());
        }
        request();
    }

    // An edit transform ends when StartTransform is cleared, and its nodes hold the transform.
    if (!viewport_changes.TransformEnd.empty() && r.get<const Interaction>(state::Document).Mode == InteractionMode::Edit && FindArmatureObject(r, FindActiveEntity(r)) == state::Null) {
        r.remove<PendingTransform>(state::Document);
    }

    UpdateMeshColliders(r);
    // History restores derived collider shapes directly.
    if (pass != EventPass::Restore) {
        std::vector<state::Entity> to_rederive;
        for (const auto e : physics_changes.ColliderPolicy) to_rederive.push_back(e);
        for (const auto mesh_entity : mesh_changes.Geometry) to_rederive.append_range(MeshCollidersOf(r, mesh_entity));
        SortUnique(to_rederive);
        // Fitted colliders read their meshes' vertex bounds from the selection aggregates.
        if (!to_rederive.empty()) select();
        RederiveColliders(r, to_rederive);
    }

    // Recompose non-bone nodes before physics and constraints read them.
    // Recompose bones after applying their poses below.
    const auto &transform_dirty = scene_changes.TransformDirty;
    const size_t first_pass_end = transform_dirty.size();
    std::vector<state::Entity> dirty_bones;
    {
        std::vector<state::Entity> roots;
        for (const auto e : transform_dirty.Entities) {
            if (r.all_of<BoneIndex>(e)) dirty_bones.push_back(e);
            else roots.push_back(e);
        }
        for (const auto e : sync.NewlyInserted) {
            if (!transform_dirty.contains(e) && !r.all_of<SubElementOf>(e)) roots.push_back(e);
        }
        // Physics writes the world transforms of the bodies it poses and of their subtrees.
        if (!roots.empty()) RecomputeWorldTransforms(r, roots, r.view<const BodyPoseCache>() | to<std::vector>(), {});
    }

    if (!rendering) {
        physics::ProcessChanges(r);
        ApplyCompletedModalSolves(r, pass);
    }

    { // Run before processing InteractionMode changes because selection may update the mode.
        const auto interaction_mode = r.get<const Interaction>(state::Document).Mode;
        auto &enabled_modes = r.edit<EnabledInteractionModes>(state::Document).Value;
        if (r.view<const SoundVertices>().empty()) {
            if (interaction_mode == InteractionMode::Excite) SetInteractionMode(r, *enabled_modes.begin());
            enabled_modes.erase(InteractionMode::Excite);
        } else if (!sound_vertex_changes.Vertices.empty()) {
            enabled_modes.insert(InteractionMode::Excite);
            if (interaction_mode == InteractionMode::Excite) {
                for (const auto e : sound_vertex_changes.Vertices) {
                    if (const auto *instance = r.valid(e) ? r.try_get<const Instance>(e) : nullptr) r.emplace_or_replace<MeshDisplayDirty>(instance->Entity);
                }
                request();
            } else {
                SetInteractionMode(r, InteractionMode::Excite);
            }
        }
    }

    {
        const auto &selected_tracker = selection_changes.Selected;
        const auto &active_tracker = selection_changes.ActiveInstance;
        const auto mode = r.get<const Interaction>(state::Document).Mode;
        if ((!selected_tracker.empty() || !active_tracker.empty()) && mode == InteractionMode::Edit) scene_state.EditPreludePending = true;
        // Convert remembered element selection when entering Edit mode with a different element mode.
        if (const auto element = r.get<const EditMode>(state::Document).Value; mode == InteractionMode::Edit && element != Element::None && !selected_tracker.empty()) {
            std::vector<state::Entity> converting;
            for (const auto e : selected_tracker) {
                const auto *instance = r.valid(e) && r.all_of<Selected>(e) ? r.try_get<const Instance>(e) : nullptr;
                if (!instance || !r.all_of<MeshElementSelection>(instance->Entity)) continue;
                const auto id = *DrawnStoreId(r, instance->Entity);
                if (meshes.Get(id).SelectionSummary.Count && meshes.GetSelectionSummary(id).Mode != element) converting.push_back(instance->Entity);
            }
            SortUnique(converting);
            if (!converting.empty()) select();
            ConvertElementSelections(r, converting, element);
        }
        // Refresh edit selections and pose wires.
        // Object-mode selection uses silhouette state bits and preserves occlusion.
        if (!selected_tracker.empty() || (!active_tracker.empty() && mode != InteractionMode::Object)) request(mode == InteractionMode::Edit);

        bool states_written = false;
        const auto write_instance_state = [&](state::Entity instance_entity) {
            if (const auto *ri = r.try_get<const RenderInstance>(instance_entity); ri && ri->BufferIndex != UINT32_MAX) {
                WriteInstanceState(r, buffers, instance_entity, ri->BufferIndex, InstanceStateBits(r, instance_entity));
                states_written = true;
            }
        };
        for (const auto *tracker : {&selected_tracker, &active_tracker}) {
            for (const auto instance_entity : *tracker) {
                write_instance_state(instance_entity);
                if (const auto arm = FindArmatureObject(r, instance_entity); arm != state::Null) bone_state_dirty.insert(arm);
            }
        }
        if (states_written) request();
    }
    {
        const auto &bone_sel_tracker = selection_changes.BoneSelection;
        if (!bone_sel_tracker.empty()) {
            request(false);
            for (const auto bone_entity : bone_sel_tracker) {
                if (const auto arm = FindArmatureObject(r, bone_entity); arm != state::Null) bone_state_dirty.insert(arm);
            }
        }
    }
    const auto interaction_mode = r.get<const Interaction>(state::Document).Mode;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;

    // Each viewport orbiting to the active element turns its view camera toward it.
    const auto orbit_to_active = [&](state::Entity instance_entity, Element element, uint32_t handle, bool canonical) {
        const auto world_pos = ComputeElementWorldPosition(r, instance_entity, element, handle, canonical);
        for (const auto [viewport, orbit, camera] : r.view<const OrbitToActive, const ViewCameraState>().each()) {
            if (!orbit.Value) continue;
            if (const auto dir = world_pos - camera.Active.Target; Dot(dir, dir) >= 1e-6f) {
                r.patch<ViewCameraState>(viewport, [&](ViewCameraState &state) { state.Active.SetTargetDirection(Normalize(dir)); });
            }
        }
    };

    if (const auto &tracker = selection_changes.MeshActiveElement; !tracker.empty()) {
        const auto edit_mode = r.get<const EditMode>(state::Document).Value;
        const auto active_entity = FindActiveEntity(r);
        const auto *active_instance = r.try_get<Instance>(active_entity);
        for (const auto mesh_entity : tracker) {
            if (const auto *active_element = r.try_get<MeshActiveElement>(mesh_entity);
                active_element && edit_mode != Element::None && active_instance && active_instance->Entity == mesh_entity) {
                orbit_to_active(active_entity, edit_mode, active_element->Handle, is_edit_mode);
            }
            if (interaction_mode == InteractionMode::Excite) dirty_sound_selection_meshes.insert(mesh_entity);
        }
    }
    for (const auto instance_entity : sound_vertex_changes.VertexForce) {
        if (interaction_mode == InteractionMode::Excite) {
            if (const auto *inst = r.try_get<Instance>(instance_entity)) dirty_sound_selection_meshes.insert(inst->Entity);
        }
        if (const auto *ev = r.try_get<VertexForce>(instance_entity)) orbit_to_active(instance_entity, Element::Vertex, ev->Vertex, false);
    }
    for (const auto instance_entity : sound_vertex_changes.Updated) {
        if (interaction_mode == InteractionMode::Excite) {
            if (const auto *inst = r.try_get<const Instance>(instance_entity)) dirty_sound_selection_meshes.insert(inst->Entity);
        }
    }
    // Camera overlays use the lenses evaluated by animation above.
    if (!render_changes.CameraLens.empty()) {
        scene_state.OverlayJobsDirty = true;
        request();
    }
    std::vector<state::Entity> material_meshes;
    if (const auto &tracker = mesh_changes.Material; !tracker.empty()) {
        for (auto mesh_entity : tracker) {
            const auto *assignment = r.try_get<const MeshMaterialAssignment>(mesh_entity);
            if (!assignment || !r.all_of<EvaluatedMesh>(mesh_entity)) continue;
            const auto material_count = buffers.Materials.Count<PBRMaterial>();
            if (material_count == 0u) continue;
            // A cook that replaces the drawn record carries its material table.
            const auto primitive_materials = meshes.EditPrimitiveMaterials(*DrawnStoreId(r, mesh_entity));
            if (assignment->PrimitiveIndex >= primitive_materials.size()) continue;
            primitive_materials[assignment->PrimitiveIndex] = std::min(assignment->MaterialIndex, material_count - 1u);
            material_meshes.emplace_back(mesh_entity);
        }
    }
    // Apply the selected variant mapping or the default material to each primitive with variant mappings.
    // Preserve assignments on meshes without variant mappings.
    if (!render_changes.ActiveMaterialVariant.empty()) {
        const auto *mv = r.try_get<const MaterialVariants>(state::Document);
        const auto active = mv ? mv->Active : std::nullopt;
        for (const auto [e, layout] : r.view<const HasVariantMappings, const MeshSourceLayout>().each()) {
            material_meshes.push_back(e);
            const auto primitive_materials = meshes.EditPrimitiveMaterials(*DrawnStoreId(r, e));
            for (size_t i = 0; i < std::min(layout.DefaultMaterials.size(), primitive_materials.size()); ++i) {
                const auto &mapping = layout.VariantMappings[i];
                primitive_materials[i] = active && *active < mapping.size() && mapping[*active] ? *mapping[*active] : layout.DefaultMaterials[i];
            }
        }
    }
    // Rebuild cluster attributes only when material requirements change.
    // Routing and shading read live materials each frame.
    const bool every_mesh = (AnyChanged(render_changes.Materials) || displays_changed) && RefreshMaterialLodAttributes(r);
    if (every_mesh) {
        material_meshes.clear();
        for (const auto entity : r.view<const EvaluatedMesh>()) material_meshes.push_back(entity);
        for (const auto [_, owned] : r.view<const procedure::ProcedureOwnedMesh>().each()) {
            if (r.valid(owned.MeshEntity)) material_meshes.push_back(owned.MeshEntity);
        }
    }
    if (!material_meshes.empty()) {
        if (!every_mesh) SortUnique(material_meshes);
        // History restores the hierarchy's attribute contract along with its geometry.
        if (pass != EventPass::Restore) RefreshClusterLodAttributes(r, chain, material_meshes);
        for (const auto e : material_meshes) r.emplace_or_replace<MeshDisplayDirty>(e);
        request();
    }
    if (!is_edit_mode && (!r.view<const MeshEditWork>().empty() || !r.view<const MeshletPositionsDirty>().empty() || !r.view<const ClusterLodDirty>().empty())) {
        auto edited = r.view<const MeshletPositionsDirty>() | to<std::vector>();
        if (!edited.empty() || !r.view<const ClusterLodDirty>().empty()) {
            StageDirtyPositionMeshlets(r, chain, edited);
            // Complete staged refits before repairing their traversal nodes and memberships.
            construct();
            // Stale coarse LOD rebuilds outside edit mode, including the groups staging marked.
            const auto repaired = r.view<const ClusterLodDirty>() | to<std::vector>();
            RepairDirtyClusterGroups(r, chain, repaired);
            edited.insert(edited.end(), repaired.begin(), repaired.end());
        }
        SortUnique(edited);
        RepointMeshInstances(r, edited);
        r.clear<MeshletPositionsDirty, ClusterLodDirty>();
        for (const auto e : r.view<const MeshEditWork>() | to<std::vector>()) ReleaseMeshEditWork(r, e);
        request();
    }
    // Overlay jobs reference tet arena ranges and contain collider parameters.
    if (AnyChanged(mesh_changes.TetMesh, physics_changes.BodyMesh)) {
        scene_state.OverlayJobsDirty = true;
        request();
    }
    if (const auto &tracker = mesh_changes.Geometry; !tracker.empty()) {
        const auto edit_mode = r.get<const EditMode>(state::Document).Value;
        std::vector<ElementRange> reset_ranges, carried_ranges;
        // Refresh instances of in-place repairs and restores.
        // Rebuild meshlets for other edited meshes with the new meshes.
        std::vector<state::Entity> edited, ready;
        for (const auto e : tracker) {
            const auto *dirty = r.try_get<const MeshGeometryDirty>(e);
            if (!dirty) continue;
            edited.push_back(e);
            if (dirty->RenderReady) ready.push_back(e);
            else if (!std::ranges::contains(sync.NewMeshEntities, e)) meshlet_meshes.push_back(e);
        }
        RepointMeshInstances(r, ready);
        // Resize selection bits after topology changes, then clear invalid selections or derive retained selections.
        const auto resets = [&](state::Entity mesh_entity) {
            return r.get<const MeshGeometryDirty>(mesh_entity).Selection != EditSelectionAfter::Keep && r.all_of<MeshElementSelection>(mesh_entity) && edit_mode != Element::None;
        };
        for (auto mesh_entity : edited) {
            if (!resets(mesh_entity)) continue;

            const auto selection_after = r.get<const MeshGeometryDirty>(mesh_entity).Selection;
            const auto mesh = GetMesh(r, mesh_entity);
            const uint32_t count = mesh.ElementCount(edit_mode);
            if (count == 0) continue;

            auto &ranges = selection_after == EditSelectionAfter::Reset ? reset_ranges : carried_ranges;
            ranges.emplace_back(mesh_entity, count);
        }
        if (!reset_ranges.empty() || !carried_ranges.empty()) select();
        if (!reset_ranges.empty()) ApplyEditSelectionCommand(r, reset_ranges, edit_mode, EditSelectionOperation::Clear);
        if (!carried_ranges.empty()) ApplyEditSelectionCommand(r, carried_ranges, edit_mode, EditSelectionOperation::Derive);
        request();
    }
    BuildMeshlets(r, chain, meshlet_meshes, bone_mesh_entities);
    if (!viewport_changes.Theme.empty()) {
        const auto &theme = r.get<const ViewportTheme>(state::Document);
        const ViewportThemeUbo ubo{
            .Colors = theme.Colors,
            .Derived = DeriveThemeColors(theme),
            .AxisColors = theme.AxisColors,
            .EdgeWidth = theme.Sizes.EdgeWidth * 0.5f * r.Resources.get<FrameState>().DisplayFramebufferScale.x,
            .SilhouetteEdgeWidth = theme.Sizes.SilhouetteEdgeWidth,
        };
        buffers.ViewportThemeUBO.Update(BytesOf(ubo));
        request();
    }
    if (displays_changed) {
        for (const auto e : r.view<const ModelsBuffer>()) r.emplace_or_replace<MeshDisplayDirty>(e);
        scene_state.OverlayJobsDirty = true;
        request();
    }
    if (const float requested = ClampMaxAnisotropy(ToMaxAnisotropy(r.get<const AnisotropicFilter>(state::Document).Value));
        requested != r.Resources.get<const ActiveSamplerAnisotropy>().Value) {
        r.Resources.get<ActiveSamplerAnisotropy>().Value = requested;
        RebuildTextureSamplers(ctx, slots, textures, requested);
        request();
    }
    const bool mode_changed = !viewport_changes.InteractionMode.empty();
    if (mode_changed) {
        // Use rest-pose geometry in Edit mode.
        for (const auto [_, instance, __] : r.view<const Instance, const ArmatureModifier>().each()) r.emplace_or_replace<MeshLayoutDirty>(instance.Entity);
        for (const auto [_, instance, __] : r.view<const Instance, const MorphWeightRange>().each()) r.emplace_or_replace<MeshLayoutDirty>(instance.Entity);
        request();
        for (const auto [_, instance, __] : r.view<const Instance, const SoundVertices>().each()) {
            r.emplace_or_replace<MeshDisplayDirty>(instance.Entity);
            if (interaction_mode == InteractionMode::Excite) dirty_sound_selection_meshes.insert(instance.Entity);
        }
        for (const auto arm : r.view<const ArmatureObject>()) bone_state_dirty.insert(arm);
    }

    {
        const auto &range = r.get<const TimelineRange>(state::Document);
        // Use interpolation instead of advancing physics during motion-blur sub-frames.
        if (!rendering && physics::AdvancePlayback(r, evaluated_from, r.get<const TimelinePlayback>(state::Document).CurrentFrame, range.StartFrame, range.EndFrame, range.Fps)) request();
    }
    // Evaluation writes materials and morph weights, so their consumers follow it.
    if (!render_changes.Materials.empty()) request();
    if (!render_changes.MorphWeights.empty()) {
        for (const auto [mesh_entity, pose] : r.view<const PosedLayout>().each()) {
            if (pose.MorphDeformOffset != InvalidOffset) r.emplace_or_replace<MeshBoundsDirty>(mesh_entity);
        }
        request();
    }
    {
        const bool is_object_mode = interaction_mode == InteractionMode::Object;
        for (const auto arm_obj_entity : bone_state_dirty) {
            if (!r.valid(arm_obj_entity) || !TryRecordOf(r, arm_obj_entity)) continue;
            const auto &arm_obj = r.get<const ArmatureObject>(arm_obj_entity);
            r.emplace_or_replace<MeshDisplayDirty>(arm_obj_entity);
            if (r.valid(arm_obj.JointEntity)) r.emplace_or_replace<MeshDisplayDirty>(arm_obj.JointEntity);
            const auto &bone_entities = arm_obj.BoneEntities;
            // Use object-level state in Object mode and per-bone state in Edit and Pose modes.
            uint8_t max_state = 0;
            if (is_object_mode) {
                // Use neutral wire colors for unselected armatures in wireframe mode.
                if (r.all_of<Selected>(arm_obj_entity)) {
                    max_state |= ElementStateSelected;
                    if (r.all_of<Active>(arm_obj_entity)) max_state |= ElementStateActive;
                }
            }
            const auto compute_state = [&](state::Entity b, BoneSel part) {
                if (is_object_mode) return max_state;
                const auto *parts = r.try_get<const BoneSelection>(b);
                const bool selected = is_edit_mode ?
                    parts && (part == BoneSel::Body ? parts->Body : part == BoneSel::Root ? parts->Root :
                                                                                            parts->Tip) :
                    r.all_of<BoneSelection>(b);
                uint8_t s = r.all_of<BoneActive>(b) ? ElementStateActive : 0;
                if (selected) s |= ElementStateSelected;
                return s;
            };

            const auto hidden = [&](state::Entity e) { return r.all_of<Hidden>(e) ? InstanceStateHidden : uint8_t{0}; };
            const bool joints_live = arm_obj.JointEntity != state::Null && r.valid(arm_obj.JointEntity);
            for (const auto b : bone_entities) {
                if (const auto *ri = r.try_get<RenderInstance>(b)) WriteInstanceState(r, buffers, b, ri->BufferIndex, uint8_t(compute_state(b, BoneSel::Body) | hidden(b)));
                const auto *joints = joints_live ? r.try_get<const BoneJointEntities>(b) : nullptr;
                if (!joints) continue;

                for (const auto &[je, part] : {std::pair{joints->Head, BoneSel::Root}, {joints->Tail, BoneSel::Tip}}) {
                    if (const auto *ri = je != state::Null ? r.try_get<const RenderInstance>(je) : nullptr) WriteInstanceState(r, buffers, je, ri->BufferIndex, uint8_t(compute_state(b, part) | hidden(je)));
                }
            }
            request();
        }

        const auto &constraint_changes = armature_changes.BoneConstraints;
        auto &constraint_targets = r.Resources.get<ConstraintTargets>().Armatures;
        if (!constraint_changes.empty() || pass == EventPass::Restore) {
            constraint_targets.clear();
            for (const auto [bone, constraints] : r.view<const BoneConstraints>().each()) {
                if (constraints.Stack.empty()) continue;

                const auto arm_obj_entity = r.get<const SubElementOf>(bone).Parent;
                const auto add = [&](state::Entity target) {
                    auto &armatures = constraint_targets[target];
                    if (!std::ranges::contains(armatures, arm_obj_entity)) armatures.push_back(arm_obj_entity);
                };
                add(arm_obj_entity);
                for (const auto &c : constraints.Stack) {
                    if (c.TargetEntity != state::Null) add(c.TargetEntity);
                }
            }
        }

        // Recompose bones after pose edits or constraint-target movement.
        // Restores publish the same changes.
        const auto &transform_end = viewport_changes.TransformEnd;
        const auto &pose_changes = armature_changes.BonePose;
        std::vector<state::Entity> posed_armatures;
        if (rendering || mode_changed) {
            for (const auto arm_obj_entity : r.view<const ArmatureObject>()) posed_armatures.push_back(arm_obj_entity);
        } else {
            posed_armatures = created_pose_armatures;
            const auto add_bone = [&](state::Entity e) {
                if (r.all_of<BoneIndex>(e)) posed_armatures.push_back(r.get<const SubElementOf>(e).Parent);
            };
            for (const auto b : dirty_bones) add_bone(b);
            for (const auto e : transform_end) add_bone(e);
            for (const auto b : pose_changes) add_bone(b);
            for (const auto b : constraint_changes) add_bone(b);
            const auto &moved = scene_changes.WorldTransform;
            for (const auto &[target, armatures] : constraint_targets) {
                if (moved.contains(target)) posed_armatures.append_range(armatures);
            }
            SortUnique(posed_armatures);
        }
        std::vector<state::Entity> synced_armatures;
        for (const auto arm_obj_entity : posed_armatures) {
            // The constraint target map can name an armature deleted since it was built.
            const auto *arm_obj = r.try_get<const ArmatureObject>(arm_obj_entity);
            if (!arm_obj) continue;
            const auto &arm_obj_comp = *arm_obj;
            auto *pose_state = r.try_edit<ArmaturePoseState>(arm_obj_comp.Entity);
            if (!pose_state) continue;

            const auto &armature = r.get<Armature>(arm_obj_comp.Entity);
            const bool created = std::ranges::contains(created_pose_armatures, arm_obj_entity);
            const bool has_any_constraint = std::ranges::any_of(arm_obj_comp.BoneEntities, [&](auto e) { return r.all_of<BoneConstraints>(e); });
            const mat4 armature_world_inv = has_any_constraint ? Inverse(ToMatrix(*WorldTransformOf(r, arm_obj_entity))) : I4;

            bool need_sync = has_any_constraint || created;
            bool rest_pose_edited = false;
            for (uint32_t i = 0; i < arm_obj_comp.BoneEntities.size(); ++i) {
                const auto b = arm_obj_comp.BoneEntities[i];
                if (!r.all_of<BoneDelta>(b)) continue;
                const auto &rest = armature.Bones[i].RestLocal;
                // Use identical pose composition for live edits and history restores.
                const auto posed = [&] { return ComposeWithDelta(rest, ComposeWithDelta(r.get<const BoneDelta>(b).Value, pose_state->BoneUserOffset[i])); };
                const auto &bt = r.get<const PosedLocal>(b).Value;
                Transform local{bt.P, bt.R, rest.S};
                bool should_patch = false;
                if (pass == EventPass::Restore) {
                    local = is_edit_mode ? rest : posed();
                    should_patch = need_sync = true;
                } else if (rendering) {
                    if (!is_edit_mode) {
                        local = posed();
                        should_patch = true;
                    }
                    need_sync = true;
                } else if (is_edit_mode) {
                    if (mode_changed) {
                        local = {rest.P, rest.R, rest.S};
                        should_patch = need_sync = true;
                    } else if (transform_end.contains(b) || (transform_dirty.contains(b) && !r.all_of<StartTransform>(b))) {
                        auto &edited = r.edit<Armature>(arm_obj_comp.Entity).Bones[i].RestLocal;
                        edited.P = bt.P;
                        edited.R = bt.R;
                        rest_pose_edited = need_sync = true;
                    }
                } else if (const auto *st = r.try_get<const StartTransform>(b)) {
                    // Apply an active drag as a user offset after animation.
                    const auto &pd = st->ParentDelta;
                    const auto grab_delta = AbsoluteToDelta(
                        rest,
                        {
                            .P = Conjugate(pd.R) * ((st->T.P - pd.P) / pd.S),
                            .R = Conjugate(pd.R) * st->T.R,
                            .S = st->T.S / pd.S,
                        }
                    );
                    const Transform gizmo_local{bt.P, bt.R, rest.S};
                    pose_state->BoneUserOffset[i] = AbsoluteToDelta(grab_delta, AbsoluteToDelta(rest, gizmo_local));
                    local = posed();
                    should_patch = need_sync = true;
                } else if (transform_end.contains(b)) {
                    // Commit the drag into the pose delta and reconstruct Transform from rest and delta.
                    r.edit<BoneDelta>(b).Value = AbsoluteToDelta(rest, {bt.P, bt.R, rest.S});
                    pose_state->BoneUserOffset[i] = {};
                    local = posed();
                    should_patch = need_sync = true;
                } else if (mode_changed || created || pose_changes.contains(b)) {
                    local = posed();
                    should_patch = need_sync = true;
                } else if (transform_dirty.contains(b)) {
                    if (const auto expected = posed();
                        bt.P != expected.P || bt.R != expected.R) {
                        r.edit<BoneDelta>(b).Value = AbsoluteToDelta(rest, {bt.P, bt.R, rest.S});
                        pose_state->BoneUserOffset[i] = {};
                        local = posed();
                        should_patch = need_sync = true;
                    }
                }

                const uint32_t parent_idx = armature.Bones[i].ParentIndex;
                const mat4 parent_pose_world = (parent_idx == InvalidBoneIndex) ? I4 : pose_state->BonePoseWorld[parent_idx];

                // Apply constraints to the unconstrained pose to avoid cumulative drift.
                if (!is_edit_mode) {
                    if (const auto *cs = r.try_get<const BoneConstraints>(b); cs && !cs->Stack.empty()) {
                        if (!should_patch) local = posed();
                        for (const auto &c : cs->Stack) {
                            if (c.TargetEntity == state::Null || !r.valid(c.TargetEntity)) continue;
                            const auto *twt = WorldTransformOf(r, c.TargetEntity);
                            if (twt) local = ApplyBoneConstraint(c, local, parent_pose_world, armature_world_inv, ToMatrix(*twt));
                        }
                        if (local.P != bt.P || local.R != bt.R) should_patch = true;
                    }
                }

                if (should_patch) r.patch<PosedLocal>(b, [&](auto &posed) { posed.Value.P = local.P; posed.Value.R = local.R; });
                pose_state->BonePoseWorld[i] = parent_pose_world * ToMatrix(local);
            }
            if (rest_pose_edited) {
                auto &edited = r.edit<Armature>(arm_obj_comp.Entity);
                // Recompute RestWorld in topological order while preserving unchanged bone positions.
                for (uint32_t i = 0; i < edited.Bones.size(); ++i) {
                    const auto parent = edited.Bones[i].ParentIndex;
                    const mat4 parent_world = (parent == InvalidBoneIndex) ? I4 : edited.Bones[parent].RestWorld;
                    const auto b = arm_obj_comp.BoneEntities[i];
                    if (transform_end.contains(b) || (transform_dirty.contains(b) && !r.all_of<StartTransform>(b))) {
                        edited.Bones[i].RestWorld = parent_world * ToMatrix(edited.Bones[i].RestLocal);
                    } else {
                        // Adjust RestLocal to preserve the previous world position after a parent change.
                        const mat4 new_local_mat = Inverse(parent_world) * edited.Bones[i].RestWorld;
                        edited.Bones[i].RestLocal.P = vec3(new_local_mat[3]);
                        edited.Bones[i].RestLocal.R = Normalize(ToQuat(ToMat3(new_local_mat)));
                        r.patch<PosedLocal>(b, [&](auto &posed) { posed.Value.P = edited.Bones[i].RestLocal.P; posed.Value.R = edited.Bones[i].RestLocal.R; });
                    }
                    edited.Bones[i].InvRestWorld = Inverse(edited.Bones[i].RestWorld);
                }
                edited.RecomputeInverseBindMatrices();
            }
            if (need_sync) {
                // A skinless armature has no deform ranges and no posed bounds entries.
                if (!is_edit_mode) {
                    for (uint32_t s = 0; s < pose_state->GpuDeformRanges.size(); ++s) {
                        ComputeDeformMatrices(armature, s, pose_state->BonePoseWorld, buffers.ArmatureDeformBuffer.GetMutable(pose_state->GpuDeformRanges[s]));
                    }
                    synced_armatures.push_back(arm_obj_comp.Entity);
                }
                request();
            }
        }
        // Recompute bounds for meshes skinned by the updated armatures.
        if (!synced_armatures.empty()) {
            SortUnique(synced_armatures);
            for (const auto [mesh_entity, pose] : r.view<const PosedLayout>().each()) {
                if (std::ranges::any_of(pose.Armatures, [&](state::Entity armature) { return std::ranges::binary_search(synced_armatures, armature); })) r.emplace_or_replace<MeshBoundsDirty>(mesh_entity);
            }
        }

        // The bones the block posed, any node dirtied since the first pass, and their descendants recompose.
        {
            auto roots = std::move(dirty_bones);
            roots.append_range(std::span{transform_dirty.Entities}.subspan(first_pass_end));
            if (!roots.empty()) {
                // A bone dragged in bone edit mode moves without its children.
                std::vector<state::Entity> held;
                if (is_edit_mode && FindArmatureObject(r, FindActiveEntity(r)) != state::Null) held = r.view<const StartTransform>() | to<std::vector>();
                RecomputeWorldTransforms(r, roots, r.view<const BodyPoseCache>() | to<std::vector>(), held);
            }
        }
        {
            const auto &moved = scene_changes.WorldTransform;
            std::span<Transform> transforms;
            const auto write = [&](state::Entity e) {
                const auto *display_scale = r.try_get<const BoneDisplayScale>(e);
                const auto *ri = display_scale ? r.try_get<const RenderInstance>(e) : nullptr;
                if (!ri || ri->BufferIndex == UINT32_MAX) return;
                const auto &wt = *WorldTransformOf(r, e);
                if (transforms.empty()) transforms = buffers.Instances.GetMutableTransforms();
                transforms[ri->BufferIndex] = Transform{wt.P, wt.R, vec3{display_scale->Value}};
                const auto *joints = r.try_get<const BoneJointEntities>(e);
                if (!joints) return;
                const float bone_length = display_scale->Value;
                const auto place_joint = [&](state::Entity joint, vec3 position) {
                    if (const auto *jri = joint != state::Null ? r.try_get<const RenderInstance>(joint) : nullptr) transforms[jri->BufferIndex] = Transform{position, {1, 0, 0, 0}, vec3{bone_length * 0.06f}};
                };
                place_joint(joints->Head, wt.P);
                place_joint(joints->Tail, wt.P + wt.R * vec3{0, bone_length, 0});
            };
            for (const auto e : moved) write(e);
            for (const auto e : sync.NewlyInserted) {
                if (!moved.contains(e)) write(e);
            }
            if (!moved.empty()) request();
        }
    }
    // The selection aggregates read this pass's world transforms.
    select();
    {
        // Refresh normal overlays when mesh selection changes.
        const bool normals = displays.VertexNormals || displays.FaceNormals;
        const auto selected_before = normals ? r.get<const SelectionFlags>(state::Document).Meshes : std::vector<state::Entity>{};
        for (const auto mesh_entity : UpdateSelectionState(r)) r.emplace_or_replace<MeshDisplayDirty>(mesh_entity);
        if (normals) {
            std::vector<state::Entity> changed;
            std::ranges::set_symmetric_difference(selected_before, r.get<const SelectionFlags>(state::Document).Meshes, std::back_inserter(changed));
            for (const auto mesh_entity : changed) {
                if (r.valid(mesh_entity)) r.emplace_or_replace<MeshDisplayDirty>(mesh_entity);
            }
        }
    }
    // Build demanded LODs after meshlet publication, preserving edit-mode pins.
    if (BuildDemandedClusterLods(r, chain)) request();
    // Each viewport looking through a camera views through its lens widened to the viewport's aspect and follows the camera when it moves.
    // A viewport whose camera is absent keeps its view until the camera returns.
    for (const auto [viewport, camera] : r.view<const ViewCameraState>().each()) {
        const auto look_through = LookThroughCamera(r, viewport);
        if (look_through == state::Null) continue;
        auto framed = camera;
        const auto lens = WidenForLookThrough(*LensOf(r, look_through), AspectRatio(RenderExtentPx(r, viewport)));
        if (scene_changes.WorldTransform.contains(look_through)) {
            const auto &wt = *WorldTransformOf(r, look_through);
            framed.Active = ViewCamera{wt.P, wt.R, lens};
        } else {
            framed.Active.Data = lens;
        }
        r.replace_if_changed(viewport, framed);
    }
    // Each viewport's frame specializes its PBR pipelines and keeps its transmission target for these features.
    if (!render_changes.PbrMeshFeatures.empty()) {
        scene_state.MeshPbrFeatures = 0u;
        for (const auto [_, feat] : r.view<const PbrMeshFeatures>().each()) scene_state.MeshPbrFeatures |= feat.Mask;
        request();
    }

    end();
    // An edit transform's update or end requests the edit prelude on the next render.
    // Object transforms preserve local instance bounds.
    if (is_edit_mode && AnyChanged(viewport_changes.TransformPending, viewport_changes.TransformEnd)) scene_state.EditPreludePending = true;

    // Every viewport's view uniforms carry the scene lighting, the edit mode and the pending transform.
    if (AnyChanged(render_changes.SceneLighting, viewport_changes.EditMode, viewport_changes.TransformPending, viewport_changes.TransformEnd)) request();

    // Publish changed sound-vertex selections in one GPU transaction.
    if (!dirty_sound_selection_meshes.empty()) {
        std::vector<std::pair<state::Entity, std::span<const uint32_t>>> sound_selections;
        sound_selections.reserve(dirty_sound_selection_meshes.size());
        // Each mesh's first sound instance supplies its vertex list.
        std::unordered_map<state::Entity, std::span<const uint32_t>> sound_vertices_by_mesh;
        for (const auto [entity, instance, excitable] : r.view<const Instance, const SoundVertices>().each()) {
            sound_vertices_by_mesh.try_emplace(instance.Entity, meshes.Arenas().SoundVertices.Get(excitable.Vertices));
        }
        for (const auto mesh_entity : dirty_sound_selection_meshes) {
            const auto it = sound_vertices_by_mesh.find(mesh_entity);
            sound_selections.emplace_back(mesh_entity, it != sound_vertices_by_mesh.end() ? it->second : std::span<const uint32_t>{});
        }
        ApplyEditSelectionLists(r, sound_selections, Element::Vertex);
        const auto records = buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>();
        const auto object_ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>();
        for (const auto mesh_entity : dirty_sound_selection_meshes) {
            if (r.valid(mesh_entity)) r.emplace_or_replace<MeshDisplayDirty>(mesh_entity);
            const auto *models = r.try_get<const ModelsBuffer>(mesh_entity);
            if (!models) continue;
            for (uint32_t slot = models->InstanceRange.Offset; slot < models->InstanceRange.Offset + models->InstanceCount; ++slot) {
                const auto *force = r.try_get<const VertexForce>(r.EntityAt(ObjectIndex(object_ids[slot])));
                records[slot].ExcitedVertex = force ? force->Vertex : InvalidOffset;
            }
        }
        request();
    }
    if (scene_state.EditSelectionDirty) {
        for (auto [_, work] : r.view<MeshEditWork>().each()) work.CandidateReady = false;
        scene_state.EditPreludePending = is_edit_mode;
        scene_state.EditSelectionDirty = false;
        request();
    }
    if (!r.view<const MeshLayoutDirty>().empty()) {
        LayOutMeshes(r);
        request();
    }
    const bool displays_refreshed = !r.view<const MeshDisplayDirty>().empty();
    if (displays_refreshed) {
        RefreshMeshDisplays(r);
        request();
    }
    if (displays_refreshed || mode_changed ||
        AnyChanged(selection_changes.Selected, selection_changes.ActiveInstance, render_changes.InstanceVisibility, render_changes.InstanceDestroyed, mesh_changes.Geometry)) {
        UpdateSilhouetteWork(r);
    }
    if (!rendering) {
        UpdateModalPlacement(r);
        UpdateAudioContacts(r);
    }
    // A changed view value requests its viewport's frame.
    for (const auto viewport : viewport_changes.View) RequestFrame(r, viewport);
    r.ClearChanges();
    r.clear<MeshGeometryDirty, MeshPositionsChanged, MeshMaterialAssignment, MeshDisplayDirty>();
    FinishCookPass(r);
}

void RegisterSceneComponentHandlers(state::Registry &r) {
    constexpr auto release_edit_work = [](state::Registry &r, state::Entity e) { r.remove<MeshEditWork>(e); };
    r.on_destroy<MeshHandle, release_edit_work>();
    r.on_destroy<EvaluatedMesh, release_edit_work>();
    r.on_destroy<MeshEditWork, &ReleaseMeshEditWorkRanges>();
    r.on_destroy<BoundsRun, &ReleaseMeshLayout>();
    auto &selection = r.Resources.emplace<SelectionChanges>();
    selection.Selected.bind(r).on<Selected>(On::Create | On::Destroy);
    selection.ActiveInstance.bind(r).on<Active>(On::Create | On::Destroy);
    selection.BoneSelection.bind(r).on<BoneSelection>(On::Create | On::Update | On::Destroy).on<BoneActive>(On::Create | On::Destroy);
    selection.MeshActiveElement.bind(r).on<MeshActiveElement>(On::Create | On::Update);
    auto &mesh = r.Resources.emplace<MeshChanges>();
    mesh.Geometry.bind(r).on<MeshGeometryDirty>(On::Create).on<MeshPositionsChanged>(On::Create);
    mesh.Material.bind(r).on<MeshMaterialAssignment>(On::Create | On::Update);
    mesh.TetMesh.bind(r).on<TetBuffers>(On::Create | On::Update | On::Destroy);
    mesh.NewBufferEntity.bind(r).on<EvaluatedMesh>(On::Create).on<VertexStoreId>(On::Create);
    auto &sound_vertices = r.Resources.emplace<SoundVertexChanges>();
    sound_vertices.Vertices.bind(r).on<SoundVertices>(On::Create | On::Destroy);
    sound_vertices.Updated.bind(r).on<SoundVertices>(On::Update);
    sound_vertices.VertexForce.bind(r).on<VertexForce>(On::Create | On::Destroy);
    auto &render = r.Resources.emplace<RenderChanges>();
    render.InstanceVisibility.bind(r).on<Instance>(On::Create | On::Update | On::Destroy).on<Hidden>(On::Create | On::Destroy);
    render.InstanceCreated.bind(r).on<RenderInstance>(On::Create);
    render.InstanceDestroyed.bind(r).on<RenderInstance>(On::Destroy);
    render.Materials.bind(r);
    render.PbrMeshFeatures.bind(r).on<PbrMeshFeatures>(On::Create | On::Update | On::Destroy);
    render.ActiveMaterialVariant.bind(r).on<MaterialVariants>(On::Create | On::Update);
    render.MaterializedTextures.bind(r).on<MaterializedTextures>(On::Create | On::Update);
    render.SceneWorld.bind(r).on<gltf::SourceAssets>(On::Create | On::Update);
    render.PunctualLight.bind(r).on<PunctualLight>(On::Create | On::Update).on<RenderInstance>(On::Create).on<Hidden>(On::Destroy);
    render.MorphWeights.bind(r);
    render.SceneLighting.bind(r).on<ImageLight>(On::Create | On::Update);
    render.CameraLens.bind(r).on<Perspective>(On::Create | On::Update).on<Orthographic>(On::Create | On::Update);
    auto &view = r.Resources.emplace<ViewportChanges>();
    view.InteractionMode.bind(r).on<Interaction>(On::Create | On::Update);
    view.EditMode.bind(r).on<EditMode>(On::Create | On::Update);
    view.TransformPending.bind(r).on<PendingTransform>(On::Create | On::Update | On::Destroy);
    view.TransformEnd.bind(r).on<StartTransform>(On::Destroy);
    view.Theme.bind(r).on<ViewportTheme>(On::Create | On::Update);
    view.View.bind(r);
    [&]<typename... Cs>(TypeList<Cs...>) { (view.View.on<Cs>(On::Create | On::Update), ...); }(ViewValues{});
    auto &scene = r.Resources.emplace<SceneChanges>();
    // Publish world-transform changes through SetWorldTransform so bone display scaling reaches instance transforms.
    scene.WorldTransform.bind(r).on<BoneDisplayScale>(On::Update);
    scene.Parent.bind(r).on<SceneParent>(On::Create | On::Update | On::Destroy);
    scene.Hierarchy.bind(r).on<SceneParent>(On::Create | On::Update | On::Destroy).on<SceneChildren>(On::Create | On::Update | On::Destroy);
    scene.Membership.bind(r).on<SceneMembership>(On::Create | On::Update | On::Destroy);
    scene.Names.bind(r).on<Name>(On::Create | On::Destroy);
    scene.Destroyed.bind(r, state::Clearing::Explicit).on<ObjectKind>(On::Destroy).on<BoneIndex>(On::Destroy);
    scene.ScaleLocked.bind(r).on<ScaleLocked>(On::Create | On::Destroy);
    scene.TransformDirty.bind(r)
        .on<Transform>(On::Create | On::Update)
        .on<PosedLocal>(On::Create | On::Update | On::Destroy)
        .on<SceneParent>(On::Create | On::Update | On::Destroy);
    auto &armature = r.Resources.emplace<ArmatureChanges>();
    armature.BonePose.bind(r).on<BoneDelta>(On::Update);
    armature.BoneConstraints.bind(r).on<BoneConstraints>(On::Create | On::Update | On::Destroy);
    r.Resources.emplace<ConstraintTargets>();
    auto &animation = r.Resources.emplace<AnimationChanges>();
    animation.Edited.bind(r)
        .on<AnimationClips>(On::Create | On::Update | On::Destroy)
        .on<Animations>(On::Update);
    animation.KeyframeSources.bind(r).on<TimelineRange>(On::Create | On::Update).on<MeshMaterialSlotSelection>(On::Create | On::Update | On::Destroy);
}
