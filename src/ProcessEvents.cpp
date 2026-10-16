#include "ProcessEvents.h"
#include "action/Errors.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "physics/ColliderUpdate.h"
#include "render/SceneUpdates.h"

#include "Camera.h"
#include "Profile.h"
#include "SortUnique.h"
#include "TransformMath.h"
#include "action/Selection.h"
#include "animation/AnimationData.h"
#include "animation/AnimationTimeline.h"
#include "animation/Evaluate.h"
#include "animation/MorphWeights.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "audio/SoundVertices.h"
#include "editor/AudioIntegration.h"
#include "gltf/GltfScene.h"
#include "mesh/Primitives.h"
#include "object/ObjectOps.h"
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
#include "render/RenderTargets.h"
#include "render/Textures.h"
#include "render/ViewportSubmission.h"
#include "scene/CameraLens.h"
#include "scene/Defaults.h"
#include "scene/SceneGraph.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "viewport/FrameState.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"

#include <format>

using state::Change;
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

vec3 ComputeElementWorldPosition(const state::Scene &r, state::Entity instance_entity, Element element, uint32_t handle, bool canonical) {
    const auto mesh = GetDrawnMesh(r, r.get<Instance>(instance_entity).Entity);
    const auto &wt = *WorldTransformOf(r, instance_entity);
    return {wt.P + Rotate(wt.R, wt.S * ComputeElementLocalPosition(mesh, element, handle, canonical))};
}

void SetEditMode(state::Scene &r, state::Entity viewport, Element mode) {
    const auto current_mode = r.get<const EditMode>(viewport).Value;
    if (current_mode == mode) return;

    // Convert edit-mode selections now and other selections on their next edit-mode entry.
    std::vector<state::Entity> editing;
    if (r.get<const Interaction>(viewport).Mode == InteractionMode::Edit) {
        for (const auto mesh_entity : r.get<const SelectionFlags>(viewport).Meshes) {
            if (r.all_of<MeshElementSelection>(mesh_entity)) editing.emplace_back(mesh_entity);
        }
    }
    r.patch<EditMode>(viewport, [mode](auto &edit_mode) { edit_mode.Value = mode; });
    ConvertElementSelections(r, editing, mode);
}

uint64_t MeshDisplaySettings(const ViewportDisplay &d) {
    return uint64_t(d.ShowOverlays) | uint64_t(d.ShowBones) << 1u | uint64_t(d.ShowBoundingBoxes) << 2u | uint64_t(d.NormalOverlays) << 8u | uint64_t(d.ViewportShading) << 16u;
}

// Count visible silhouette instances, excluding primary edit instances.
void UpdateSilhouetteWork(state::Scene &r, state::Entity viewport) {
    const auto &primaries = r.get<const EditPrimaries>(viewport).All;
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
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

void RequestRender(state::Scene &r, bool disoccludes) {
    r.Context.get<PendingRenderRequest>().Value = true;
    if (disoccludes) r.Context.get<GpuBuffers>().MeshletOcclusionStale = true;
}

void ProcessComponentEvents(state::Scene &r, state::Entity viewport, EventPass pass) {
    const bool rendering = pass == EventPass::Sample || pass == EventPass::Render;
    const auto &ctx = r.Context.get<const mtl::Context>();
    auto &slots = r.Context.get<mtl::BindlessSet>();
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &meshes = r.Context.get<MeshStore>();
    auto &render = meshes.Render();
    auto &textures = r.Context.get<TextureStore>();
    auto &environments = r.Context.get<EnvironmentStore>();
    auto &targets = r.Context.get<RenderTargets>();
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

    auto &pending_render = r.Context.get<PendingRenderRequest>().Value;
    const auto request = [&r](bool disoccludes = true) { RequestRender(r, disoccludes); };

    std::unordered_set<state::Entity> bone_state_dirty;

    ProcessObjectRemovals(r, viewport);

    // Preserve flat shading when regenerating primitives.
    // History restores their meshes directly.
    if (const auto &tracker = reactive(r, Change::PrimitiveShape); pass != EventPass::Restore && !tracker.empty()) {
        std::vector<state::Entity> primitives;
        for (const auto e : tracker) {
            if (r.valid(e) && r.all_of<PrimitiveShape, MeshHandle>(e)) primitives.emplace_back(e);
        }
        // Arena layout follows source order.
        std::ranges::sort(primitives);
        std::vector<MeshSource> sources;
        sources.reserve(primitives.size());
        for (const auto e : primitives) {
            sources.push_back({.Data = primitive::CreateMesh(r.get<const PrimitiveShape>(e)), .FlatShaded = meshes.GetFaceSharpnessSummary(r.get<const MeshHandle>(e).StoreId).All});
        }
        const auto created = CreateMeshes(r, sources);
        for (uint32_t i = 0u; i < primitives.size(); ++i) {
            // Erasing the handle queues the old store entry for release.
            r.remove<MeshHandle>(primitives[i]);
            r.emplace<MeshHandle>(primitives[i], created[i].StoreId);
            r.emplace_or_replace<MeshGeometryDirty>(primitives[i]);
        }
    }
    // Replan restored procedures without regenerating their restored outputs.
    SettleProcedures(r, pass == EventPass::Frame || pass == EventPass::Settle);
    ProcessObjectRemovals(r, viewport);

    // Resize render resources before the pick handlers below resolve against the rendered scene.
    const bool resized = SyncViewportRenderResources(r, viewport);
    if (resized) request();

    const bool recompiled = std::exchange(r.Context.get<FrameState>().RecompileShaders, false);
    if (recompiled) {
        r.Context.get<mtl::LibraryCache>().Clear();
        r.Context.erase<Pipelines>();
        r.Context.erase<MeshPipelines>();
        // Recompiled prefilter kernels must regenerate their cached cubemaps.
        RebuildStudioEnvironments(r);
        request();
    }

    // Restore texture data into free recorded slots while preserving textures materialized during import.
    if (!reactive(r, Change::MaterializedTextures).empty()) {
        if (const auto *manifest = r.try_get<const MaterializedTextures>(viewport)) {
            for (const auto &t : manifest->Items) {
                if (!slots.Reserve(SlotType::Sampler, t.SamplerSlot)) continue;
                textures.PendingUploads.emplace_back(PendingTextureUpload{.SamplerSlot = t.SamplerSlot, .Source = PendingTextureUpload::GltfImageRef{t.SourceImageIndex}, .Params = t.Params});
            }
        }
    }
    if (!textures.PendingUploads.empty()) {
        const auto *src = r.try_get<const gltf::SourceAssets>(viewport);
        static const std::vector<gltf::Image> empty_images;
        const auto &gltf_images = src ? src->Images : empty_images;
        TextureUploadBatch batch{ctx};
        for (const auto &item : textures.PendingUploads) {
            auto entry = MaterializeTextureEntry(r, batch, slots, item, gltf_images, r.Context.get<const ActiveSamplerAnisotropy>().Value);
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
    // Rebuild restored EXT-IBL scene resources after ClearScene releases their prefiltered cubemap.
    if (!reactive(r, Change::SceneWorld).empty()) {
        const auto *src = r.try_get<const gltf::SourceAssets>(viewport);
        if (src && src->ImageBasedLight && !environments.ImportedSceneWorld && !environments.PendingImport) {
            const auto [diffuse_slot, specular_slot] = AllocateIblCubeSlots(slots);
            environments.PendingImport = PendingEnvironmentImport{*src->ImageBasedLight, diffuse_slot, specular_slot};
            environments.ClearRequested = false;
        }
    }
    // Cancel a stale pending EXT-IBL import before applying a later scene-world clear.
    if (std::exchange(environments.ClearRequested, false)) {
        if (auto &imp = environments.PendingImport) {
            ReleaseCubeSamplerSlot(slots, imp->DiffuseCubeSlot);
            ReleaseCubeSamplerSlot(slots, imp->SpecularCubeSlot);
            imp.reset();
        }
        ResetImportedEnvironment(r);
    }
    if (auto &pending_env = environments.PendingImport) {
        if (const auto *src = r.try_get<const gltf::SourceAssets>(viewport)) {
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
    if (!reactive(r, Change::StudioEnvironment).empty()) {
        SetStudioEnvironment(r, r.get<const StudioEnvironment>(viewport).Name);
    }

    // Process pending handlers before consuming the reactive trackers they update.
    if (const auto *pending = r.try_get<const PendingSetEditMode>(viewport)) {
        const auto mode = pending->Mode;
        r.remove<PendingSetEditMode>(viewport);
        SetEditMode(r, viewport, mode);
    }
    if (const auto *pending = r.try_get<PendingImportMesh>(viewport)) {
        const auto path = pending->Path;
        auto info = pending->Info;
        r.remove<PendingImportMesh>(viewport);
        if (const auto imported = ImportMesh(r, viewport, path, std::move(info)); !imported) action::Fail(r, imported.error());
    }
    // Use the rendered camera for selection, culling, and LOD.
    const auto prepare_selection = [&](const RenderView &view) {
        auto &frame_view = *reinterpret_cast<SceneViewUBO *>(buffers.SceneViewUBO.Contents().data());
        buffers.FrameView = view;
        view.ApplyTo(frame_view);
        // Submit restored geometry and draw records before selection.
        if (pending_render) {
            RecordAndSubmitFrame(r, viewport, RenderPhase::Prepare);
            WaitForRender(r);
        }
    };
    const auto target_px = [&](vec2 fraction) {
        const auto extent = RenderExtentPx(r);
        const auto last = Max(extent, uvec2{1}) - uvec2{1};
        return Min(uvec2{std::lround(fraction.x * float(extent.x)), std::lround(fraction.y * float(extent.y))}, last);
    };
    if (const auto *pending = r.try_get<const PendingEditElementClick>(viewport)) {
        const auto mouse_px = target_px(pending->Mouse);
        const bool toggle = pending->Toggle;
        prepare_selection(pending->View);
        r.remove<PendingEditElementClick>(viewport);

        const auto edit_mode = r.get<const EditMode>(viewport).Value;
        const auto ranges = GetElementRangesForSelected(r, viewport);
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
    }
    if (const auto *pending = r.try_get<const PendingBoxSelect>(viewport)) {
        const auto box_px = std::pair{target_px(pending->Box.first), target_px(pending->Box.second)};
        const bool additive = pending->Additive;
        prepare_selection(pending->View);
        r.remove<PendingBoxSelect>(viewport);

        const auto &interaction = r.get<const Interaction>(viewport);
        const bool element_mode = interaction.Mode == InteractionMode::Edit && FindArmatureObject(r, FindActiveEntity(r)) == state::Null;
        // Capture the initial selection once for additive box selection.
        if (additive && !r.all_of<AdditiveBoxSelectBaseline>(viewport)) {
            auto &baseline = r.emplace<AdditiveBoxSelectBaseline>(viewport);
            if (interaction.Mode == InteractionMode::Pose || IsBoneEditMode(r, viewport)) {
                for (const auto e : r.view<BoneSelection>()) baseline.BoneSelections.emplace_back(e, r.get<BoneSelection>(e));
            } else if (!element_mode) {
                for (const auto e : r.view<Selected>()) baseline.SelectedEntities.emplace_back(e);
            }
        }
        if (element_mode) {
            const auto ranges = GetElementRangesForSelected(r, viewport);
            if (!additive) {
                for (const auto &range : ranges) r.remove<MeshActiveElement>(range.MeshEntity);
            }
            RunBoxSelectElements(r, viewport, ranges, r.get<const EditMode>(viewport).Value, box_px, additive);
        } else {
            const bool bone_mode = interaction.Mode == InteractionMode::Pose || IsBoneEditMode(r, viewport);
            const auto hits = ResolveHits(r, RunBoxSelect(r, viewport, box_px), bone_mode, true);
            const auto *baseline = additive ? r.try_get<const AdditiveBoxSelectBaseline>(viewport) : nullptr;
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
    if (const auto *pending = r.try_get<const PendingPick>(viewport)) {
        const auto mouse_px = target_px(pending->Mouse);
        const bool shift = pending->Shift, cycle = pending->Cycle;
        prepare_selection(pending->View);
        r.remove<PendingPick>(viewport);

        const bool bone_mode = r.get<const Interaction>(viewport).Mode == InteractionMode::Pose || IsBoneEditMode(r, viewport);
        const auto active = bone_mode ? FindActiveBone(r) : FindActiveEntity(r);
        const auto logical_extent = r.Context.get<ViewportExtent>().Value;
        const auto render_extent = RenderExtentPx(r);
        const float render_scale = std::max(
            logical_extent.x > 0u ? float(render_extent.x) / float(logical_extent.x) : 1.f,
            logical_extent.y > 0u ? float(render_extent.y) / float(logical_extent.y) : 1.f
        );
        const auto radius = std::max(1u, uint32_t(std::lround(float(ObjectSelectRadiusPx) * render_scale)));
        const auto hits = ResolveHits(r, RunObjectPick(r, mouse_px, radius), bone_mode);
        const auto pick = hits.empty() ? std::optional<SelectionHit>{} : [&]() -> std::optional<SelectionHit> {
            if (!cycle) return hits.front();
            const auto *bs = r.try_get<const BoneSelection>(active);
            auto it = std::ranges::find_if(hits, [&](const SelectionHit &h) { return h.Entity == active && (!h.Part || (bs && bs->Has(*h.Part))); });
            return it != hits.end() && ++it != hits.end() ? *it : hits.front();
        }();
        using namespace action::selection;
        if (pick && shift) {
            if (active == pick->Entity && !bone_mode) Apply(r, viewport, ToggleSelected{pick->Entity});
            else if (bone_mode) Apply(r, viewport, ExtendBoneActive{pick->Entity, pick->Part, true});
            else Apply(r, viewport, ExtendActive{pick->Entity});
        } else if (pick || !shift) {
            if (pick && bone_mode) Apply(r, viewport, action::selection::SelectBone{pick->Entity, pick->Part, false});
            else if (pick) Apply(r, viewport, action::selection::Select{pick->Entity});
            else Apply(r, viewport, DeselectAll{});
        }
    }

    // Evaluate animation before synchronizing render state.
    bool anim_advanced;
    int evaluated_from;
    {
        const auto &range = r.get<const TimelineRange>(viewport);
        const auto &playback = r.get<const TimelinePlayback>(viewport);
        auto &pf = r.edit<PlaybackFrame>(viewport).Value;
        auto &frame_state = r.Context.get<FrameState>();
        anim_advanced = [&] {
            if (pass == EventPass::Restore) {
                pf = float(playback.CurrentFrame);
                return false;
            }
            if (rendering) return false;
            // A frame set by an action shows for one tick before playback advances.
            if (playback.Playing && pass == EventPass::Frame && playback.CurrentFrame == r.get<const LastEvaluatedFrame>(viewport).Value) {
                const float step = frame_state.FixedFrameStep ? 1.f : frame_state.DeltaTime * range.Fps;
                pf += playback.Reverse ? -step : step;
                if (pf > float(range.EndFrame)) pf = float(range.StartFrame);
                else if (pf < float(range.StartFrame)) pf = float(range.EndFrame);
                const auto new_frame = int(std::floor(pf));
                if (new_frame != playback.CurrentFrame) r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = new_frame; });
            } else if (!playback.Playing) {
                pf = float(playback.CurrentFrame);
            }
            return playback.CurrentFrame != r.edit<LastEvaluatedFrame>(viewport).Value || !reactive(r, Change::AnimationEdited).empty();
        }();

        evaluated_from = r.edit<LastEvaluatedFrame>(viewport).Value;
        // Re-evaluate animation after restoring the frame, animation data, or a pose seed transform.
        const bool restore_poses = pass == EventPass::Restore &&
            (playback.CurrentFrame != evaluated_from || AnyChanged(r, Change::AnimationEdited, Change::TransformDirty));
        if (anim_advanced || pass == EventPass::Restore) r.edit<LastEvaluatedFrame>(viewport).Value = playback.CurrentFrame;
        // Convert 1-based display frames to animation time and preserve fractional motion-blur samples.
        const float frame_seconds = float(std::max(0, playback.CurrentFrame - 1)) / range.Fps;
        const float eval_seconds = pass == EventPass::Sample ? std::max(0.f, pf - 1.f) / range.Fps : frame_seconds;

        // A restore rebuilds only the derived node poses, since history restores every other animated value.
        if (anim_advanced || rendering || restore_poses) animation::Evaluate(r, viewport, eval_seconds, pass != EventPass::Restore);
        // Request a frame even when playback leaves animated values unchanged.
        if (anim_advanced) request();
    }

    // Derive render instances after animation updates visibility.
    if (DeriveRenderInstances(r)) request();
    auto sync = SyncModelsBuffers(r);
    if (sync.SlotsChanged) {
        request();
        for (const auto &[mesh_entity, _] : r.get<const EditPrimaries>(viewport).All) r.emplace_or_replace<MeshDisplayDirty>(mesh_entity);
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

    auto &scene_state = r.Context.get<GpuSceneState>();
    { // Register changed and shown lights into the GPU Lights buffer, the single path for both new and restored lights.
        bool synced = false;
        for (const auto entity : reactive(r, Change::PunctualLight)) {
            if (!r.all_of<PunctualLight, Instance>(entity) || r.all_of<Hidden>(entity)) continue;
            const auto *ri = r.try_get<const RenderInstance>(entity);
            if (!ri || ri->BufferIndex == UINT32_MAX) continue;
            const auto index = r.all_of<LightIndex>(entity) ? r.get<const LightIndex>(entity).Value : buffers.Lights.Count<LightRecord>();
            if (!r.all_of<LightIndex>(entity)) r.emplace<LightIndex>(entity, index);
            const auto &light = r.get<const PunctualLight>(entity);
            const LightRecord gpu_light{
                .TransformSlotOffset = {buffers.Instances.TransformBuffer.Slot, ri->BufferIndex},
                .Range = light.Range,
                .Color = light.Color,
                .Intensity = light.Intensity,
                .InnerConeCos = std::cos(light.InnerConeAngle),
                .OuterConeCos = std::cos(light.OuterConeAngle),
                .Type = light.Type,
            };
            if (index >= buffers.Lights.Count<LightRecord>()) scene_state.OverlayJobsDirty = true;
            else {
                const auto old = buffers.Lights.GetSpan<LightRecord>()[index];
                if (old.Type != gpu_light.Type || old.Range != gpu_light.Range || old.OuterConeCos != gpu_light.OuterConeCos || old.InnerConeCos != gpu_light.InnerConeCos) {
                    scene_state.OverlayJobsDirty = true;
                }
            }
            buffers.Lights.Update(as_bytes(gpu_light), uint64_t(index) * sizeof(LightRecord));
            synced = true;
        }
        if (synced) request();
    }

    const auto remove_light = [&](state::Entity entity) {
        if (const auto *light_index = r.try_get<const LightIndex>(entity)) {
            buffers.PendingLightRemovals.emplace_back(light_index->Value);
            r.remove<LightIndex>(entity);
        }
    };
    for (const auto entity : reactive(r, Change::RenderInstanceDestroyed)) remove_light(entity);
    for (const auto entity : reactive(r, Change::InstanceVisibility)) {
        if (r.valid(entity) && r.all_of<Hidden>(entity)) remove_light(entity);
    }
    if (auto &indices = buffers.PendingLightRemovals; !indices.empty()) {
        const auto before = buffers.Lights.Count<LightRecord>();
        SortUnique(indices);
        std::erase_if(indices, [before](uint32_t index) { return index >= before; });
        ForEachSurvivorRun({0u, before}, indices, [&](uint32_t from, uint32_t to, uint32_t count) {
            if (from != to && count) buffers.Lights.Move(uint64_t(from) * sizeof(LightRecord), uint64_t(to) * sizeof(LightRecord), uint64_t(count) * sizeof(LightRecord));
        });
        for (const auto [entity, index] : r.view<const LightIndex>().each()) {
            const auto shift = uint32_t(std::ranges::lower_bound(indices, index.Value) - indices.begin());
            if (shift) r.replace<LightIndex>(entity, index.Value - shift);
        }
        buffers.Lights.SetCount<LightRecord>(before - indices.size());
        indices.clear();
        request();
    }

    // Commit mesh edit transforms after StartTransform is cleared.
    if (!reactive(r, Change::TransformEnd).empty()) {
        if (r.get<const Interaction>(viewport).Mode == InteractionMode::Edit && FindArmatureObject(r, FindActiveEntity(r)) == state::Null) {
            if (const auto *pending = r.try_get<const PendingTransform>(viewport); pending && pending->Delta != Transform{}) {
                // Evaluate final edits before geometry and physics updates.
                // Preserve scale-locked geometry and record procedure transforms in their nodes.
                std::unordered_set<state::Entity> scale_locked;
                for (const auto [_, instance] : r.view<const Instance, const ScaleLocked>().each()) scale_locked.insert(instance.Entity);
                std::vector<state::Entity> commit_meshes;
                for (const auto &[mesh_entity, instance_entity] : r.get<const EditPrimaries>(viewport).Transformable) {
                    if (!scale_locked.contains(mesh_entity) && !r.all_of<procedure::GeometryProcedure>(mesh_entity)) commit_meshes.push_back(mesh_entity);
                }
                const auto committed = CommitPosedGeometry(r, chain, viewport, commit_meshes);
                if (!committed.empty()) construct();
                for (const auto mesh_entity : PublishEditedPositions(r, chain, committed)) {
                    r.remove<PrimitiveShape>(mesh_entity);
                    r.emplace_or_replace<MeshPositionsChanged>(mesh_entity);
                }
            }
            r.remove<PendingTransform>(viewport);
        }
    }

    UpdateMeshColliders(r);
    // History restores derived collider shapes directly.
    if (pass != EventPass::Restore) {
        std::vector<state::Entity> to_rederive;
        for (const auto e : reactive(r, Change::ColliderPolicy)) to_rederive.push_back(e);
        for (const auto mesh_entity : reactive(r, Change::MeshGeometry)) to_rederive.append_range(MeshCollidersOf(r, mesh_entity));
        SortUnique(to_rederive);
        // Fitted colliders read their meshes' vertex bounds from the selection aggregates.
        if (!to_rederive.empty()) select();
        RederiveColliders(r, to_rederive);
    }

    // Recompose non-bone nodes before physics and constraints read them.
    // Recompose bones after applying their poses below.
    const auto &transform_dirty = reactive(r, Change::TransformDirty);
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
        const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
        auto &enabled_modes = r.edit<EnabledInteractionModes>(viewport).Value;
        if (r.view<const SoundVertices>().empty()) {
            if (interaction_mode == InteractionMode::Excite) SetInteractionMode(r, viewport, *enabled_modes.begin());
            enabled_modes.erase(InteractionMode::Excite);
        } else if (!reactive(r, Change::SoundVertices).empty()) {
            enabled_modes.insert(InteractionMode::Excite);
            if (interaction_mode == InteractionMode::Excite) {
                for (const auto e : reactive(r, Change::SoundVertices)) {
                    if (const auto *instance = r.valid(e) ? r.try_get<const Instance>(e) : nullptr) r.emplace_or_replace<MeshDisplayDirty>(instance->Entity);
                }
                request();
            } else {
                SetInteractionMode(r, viewport, InteractionMode::Excite);
            }
        }
    }

    {
        auto &selected_tracker = reactive(r, Change::Selected);
        auto &active_tracker = reactive(r, Change::ActiveInstance);
        const auto mode = r.get<const Interaction>(viewport).Mode;
        if ((!selected_tracker.empty() || !active_tracker.empty()) && mode == InteractionMode::Edit) scene_state.EditPreludePending = true;
        // Convert remembered element selection when entering Edit mode with a different element mode.
        if (const auto element = r.get<const EditMode>(viewport).Value; mode == InteractionMode::Edit && element != Element::None && !selected_tracker.empty()) {
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

        std::span<uint8_t> states;
        const auto write_instance_state = [&](state::Entity instance_entity) {
            if (const auto *ri = r.try_get<const RenderInstance>(instance_entity); ri && ri->BufferIndex != UINT32_MAX) {
                if (states.empty()) states = buffers.Instances.GetMutableStates();
                states[ri->BufferIndex] = InstanceStateBits(r, instance_entity);
            }
        };
        for (const auto *tracker : {&selected_tracker, &active_tracker}) {
            for (const auto instance_entity : *tracker) {
                write_instance_state(instance_entity);
                if (const auto arm = FindArmatureObject(r, instance_entity); arm != state::Null) bone_state_dirty.insert(arm);
            }
        }
        if (!states.empty()) request();
    }
    {
        const auto &bone_sel_tracker = reactive(r, Change::BoneSelection);
        if (!bone_sel_tracker.empty()) {
            request(false);
            for (const auto bone_entity : bone_sel_tracker) {
                if (const auto arm = FindArmatureObject(r, bone_entity); arm != state::Null) bone_state_dirty.insert(arm);
            }
        }
    }
    const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;
    // Apply or remove the pending transform from primary-instance bounds when an edit transform starts or ends.
    if (is_edit_mode && AnyChanged(r, Change::TransformStart, Change::TransformEnd)) {
        for (const auto &[mesh_entity, _] : r.get<const EditPrimaries>(viewport).All) r.emplace_or_replace<MeshLayoutDirty>(mesh_entity);
        request();
    }

    const auto orbit_to_active = [&](state::Entity instance_entity, Element element, uint32_t handle, bool canonical) {
        if (!r.get<const OrbitToActive>(viewport).Value) return;
        const auto world_pos = ComputeElementWorldPosition(r, instance_entity, element, handle, canonical);
        r.patch<ViewCamera>(viewport, [&](auto &camera) {
            if (const auto dir = world_pos - camera.Target; Dot(dir, dir) >= 1e-6f) {
                camera.SetTargetDirection(Normalize(dir));
            }
        });
    };

    if (const auto &tracker = reactive(r, Change::MeshActiveElement); !tracker.empty()) {
        const auto edit_mode = r.get<const EditMode>(viewport).Value;
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
    for (const auto instance_entity : reactive(r, Change::VertexForce)) {
        if (interaction_mode == InteractionMode::Excite) {
            if (const auto *inst = r.try_get<Instance>(instance_entity)) dirty_sound_selection_meshes.insert(inst->Entity);
        }
        if (const auto *ev = r.try_get<VertexForce>(instance_entity)) orbit_to_active(instance_entity, Element::Vertex, ev->Vertex, false);
    }
    for (const auto instance_entity : reactive(r, Change::SoundVerticesUpdated)) {
        if (interaction_mode == InteractionMode::Excite) {
            if (const auto *inst = r.try_get<const Instance>(instance_entity)) dirty_sound_selection_meshes.insert(inst->Entity);
        }
    }
    // Camera overlays use the lenses evaluated by animation above.
    if (const auto &lenses = reactive(r, Change::CameraLens); !lenses.empty()) {
        scene_state.OverlayJobsDirty = true;
        request();
        for (const auto camera_entity : lenses) {
            if (HasLens(r, camera_entity) && r.all_of<LookingThrough>(camera_entity)) r.patch<ViewCamera>(viewport, [](auto &) {});
        }
    }
    bool light_count_changed = false;
    if (const uint32_t required_count = r.view<const LightIndex>().size();
        buffers.Lights.Count<LightRecord>() != required_count) {
        buffers.Lights.SetCount<LightRecord>(required_count);
        light_count_changed = true;
    }
    if (!reactive(r, Change::WorkspaceLights).empty()) {
        buffers.WorkspaceLightsUBO.Update(as_bytes(r.get<const WorkspaceLights>(viewport)));
        request();
    }
    std::vector<state::Entity> material_meshes;
    if (const auto &tracker = reactive(r, Change::MeshMaterial); !tracker.empty()) {
        for (auto mesh_entity : tracker) {
            const auto *assignment = r.try_get<const MeshMaterialAssignment>(mesh_entity);
            const auto mesh = TryGetMesh(r, mesh_entity);
            if (!assignment || !mesh) continue;
            const auto material_count = buffers.Materials.Count<PBRMaterial>();
            if (material_count == 0u) continue;

            const auto primitive_materials = meshes.EditPrimitiveMaterials(mesh->GetStoreId());
            if (assignment->PrimitiveIndex < primitive_materials.size()) {
                primitive_materials[assignment->PrimitiveIndex] = std::min(assignment->MaterialIndex, material_count - 1u);
                material_meshes.emplace_back(mesh_entity);
            }
        }
    }
    // Apply the selected variant mapping or the default material to each primitive with variant mappings.
    // Preserve assignments on meshes without variant mappings.
    if (!reactive(r, Change::ActiveMaterialVariant).empty()) {
        const auto *mv = r.try_get<const MaterialVariants>(viewport);
        const auto active = mv ? mv->Active : std::nullopt;
        for (const auto [e, layout, _] : r.view<const HasVariantMappings, const MeshSourceLayout, const MeshHandle>().each()) {
            const auto mesh = GetMesh(r, e);
            const auto primitive_materials = meshes.EditPrimitiveMaterials(mesh.GetStoreId());
            material_meshes.push_back(e);
            for (size_t i = 0; i < layout.DefaultMaterials.size(); ++i) {
                const auto &mapping = layout.VariantMappings[i];
                primitive_materials[i] = active && *active < mapping.size() && mapping[*active] ? *mapping[*active] : layout.DefaultMaterials[i];
            }
        }
    }
    // Rebuild cluster attributes only when material requirements change.
    // Routing and shading read live materials each frame.
    const bool every_mesh = AnyChanged(r, Change::Materials, Change::ViewportDisplay) && RefreshMaterialLodAttributes(r);
    if (every_mesh) {
        material_meshes.clear();
        material_meshes.reserve(r.view<const MeshHandle>().size());
        for (const auto entity : r.view<const MeshHandle>()) material_meshes.push_back(entity);
    }
    if (!material_meshes.empty()) {
        if (!every_mesh) SortUnique(material_meshes);
        // History restores the hierarchy's attribute contract along with its geometry.
        if (pass != EventPass::Restore) RefreshClusterLodAttributes(r, chain, material_meshes);
        for (const auto e : material_meshes) r.emplace_or_replace<MeshDisplayDirty>(e);
        request();
    }
    if (!is_edit_mode && (!r.view<const MeshEditWork>().empty() || !r.view<const MeshletPositionsDirty>().empty() || !r.view<const ClusterLodDirty>().empty())) {
        auto edited = r.view<const MeshletPositionsDirty, const MeshHandle>() | to<std::vector>();
        if (!edited.empty() || !r.view<const ClusterLodDirty>().empty()) {
            StageDirtyPositionMeshlets(r, chain, edited);
            // Complete staged refits before repairing their traversal nodes and memberships.
            construct();
            // Stale coarse LOD rebuilds outside edit mode, including the groups staging marked.
            const auto repaired = r.view<const ClusterLodDirty, const MeshHandle>() | to<std::vector>();
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
    if (AnyChanged(r, Change::TetMesh, Change::PhysicsBodyMesh)) {
        scene_state.OverlayJobsDirty = true;
        request();
    }
    if (const auto &tracker = reactive(r, Change::MeshGeometry); !tracker.empty()) {
        const auto edit_mode = r.get<const EditMode>(viewport).Value;
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
            const auto mesh = GetDrawnMesh(r, mesh_entity);
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
    if (is_edit_mode && pass != EventPass::Restore && !reactive(r, Change::TransformPending).empty() && RefreshPreviewTessellation(r, viewport)) request();
    if (!reactive(r, Change::ViewportTheme).empty()) {
        auto theme = r.get<const ViewportTheme>(viewport);
        UpdateDerivedColors(theme);
        theme.EdgeWidth *= r.Context.get<FrameState>().DisplayFramebufferScale.x;
        buffers.ViewportThemeUBO.Update(as_bytes(theme));
        request();
    }
    if (!reactive(r, Change::ViewportDisplay).empty()) {
        request();
        if (const auto settings = MeshDisplaySettings(r.get<const ViewportDisplay>(viewport)); std::exchange(scene_state.MeshDisplaySettings, settings) != settings) {
            for (const auto e : r.view<const ModelsBuffer>()) r.emplace_or_replace<MeshDisplayDirty>(e);
            scene_state.OverlayJobsDirty = true;
        }
        if (const float requested = ClampMaxAnisotropy(ToMaxAnisotropy(r.get<const ViewportDisplay>(viewport).AnisotropicFilter));
            requested != r.Context.get<const ActiveSamplerAnisotropy>().Value) {
            r.Context.get<ActiveSamplerAnisotropy>().Value = requested;
            RebuildTextureSamplers(ctx, slots, textures, requested);
        }
    }
    const bool mode_changed = !reactive(r, Change::InteractionMode).empty();
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
        const auto &range = r.get<const TimelineRange>(viewport);
        // Use interpolation instead of advancing physics during motion-blur sub-frames.
        if (!rendering && physics::AdvancePlayback(r, viewport, evaluated_from, r.get<const TimelinePlayback>(viewport).CurrentFrame, range.StartFrame, range.EndFrame, range.Fps)) request();
    }
    // Evaluation writes materials and morph weights, so their consumers follow it.
    if (!reactive(r, Change::Materials).empty()) request();
    if (!reactive(r, Change::MorphWeights).empty()) {
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
                if (const auto *ri = r.try_get<RenderInstance>(b)) buffers.Instances.UpdateState(ri->BufferIndex, uint8_t(compute_state(b, BoneSel::Body) | hidden(b)));
                const auto *joints = joints_live ? r.try_get<const BoneJointEntities>(b) : nullptr;
                if (!joints) continue;

                for (const auto &[je, part] : {std::pair{joints->Head, BoneSel::Root}, {joints->Tail, BoneSel::Tip}}) {
                    if (const auto *ri = je != state::Null ? r.try_get<const RenderInstance>(je) : nullptr) buffers.Instances.UpdateState(ri->BufferIndex, uint8_t(compute_state(b, part) | hidden(je)));
                }
            }
            request();
        }

        const auto &constraint_changes = reactive(r, Change::BoneConstraints);
        auto &constraint_targets = r.Context.get<ConstraintTargets>().Armatures;
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
        const auto &transform_end = reactive(r, Change::TransformEnd);
        const auto &pose_changes = reactive(r, Change::BonePose);
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
            const auto &moved = reactive(r, Change::WorldTransform);
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
            const auto &moved = reactive(r, Change::WorldTransform);
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
        const auto &display = r.get<const ViewportDisplay>(viewport);
        const bool normals = display.ShowOverlays && display.NormalOverlays != 0u;
        const auto selected_before = normals ? r.get<const SelectionFlags>(viewport).Meshes : std::vector<state::Entity>{};
        for (const auto mesh_entity : UpdateSelectionState(r, viewport)) r.emplace_or_replace<MeshDisplayDirty>(mesh_entity);
        if (normals) {
            std::vector<state::Entity> changed;
            std::ranges::set_symmetric_difference(selected_before, r.get<const SelectionFlags>(viewport).Meshes, std::back_inserter(changed));
            for (const auto mesh_entity : changed) {
                if (r.valid(mesh_entity)) r.emplace_or_replace<MeshDisplayDirty>(mesh_entity);
            }
        }
    }
    // Build demanded LODs after meshlet publication, preserving edit-mode pins.
    if (BuildDemandedClusterLods(r, chain, viewport)) request();
    // Update an active scene camera before processing SceneView changes.
    if (const auto camera = LookThroughCameraEntity(r); camera != state::Null &&
        reactive(r, Change::WorldTransform).contains(camera)) {
        const auto &wt = *WorldTransformOf(r, camera);
        r.replace<ViewCamera>(viewport, ViewCamera{wt.P, wt.R, *LensOf(r, camera)});
    }
    {
        // Update transmission specialization before the UBO reads its pipeline state.
        const auto shading = r.get<const ViewportDisplay>(viewport).ViewportShading;
        const bool mesh_features = !reactive(r, Change::PbrMeshFeatures).empty();
        if (mesh_features) {
            scene_state.MeshPbrFeatures = 0u;
            for (const auto [_, feat] : r.view<const PbrMeshFeatures>().each()) scene_state.MeshPbrFeatures |= feat.Mask;
        }
        if (recompiled || mesh_features || AnyChanged(r, Change::ViewportDisplay, Change::PbrSpecialization)) {
            // SubmitViewport refreshes all slots only on resize, so update this lazy sampler inline.
            const auto refresh_transmission_sampler = [&] {
                const auto info = targets.TransmissionSampler();
                slots.SetSampler({SlotType::Sampler, r.Context.get<const RenderSamplerSlots>().Transmission}, info.Texture, info.Sampler);
                request();
            };
            if (!WorkbenchShading(shading)) {
                PbrFeatureMask pbr_mask{scene_state.MeshPbrFeatures};
                const auto &active_lighting = GetActivePbrLighting(r, viewport, shading);
                if (active_lighting.UseSceneLights) pbr_mask |= PbrFeature::Punctual;
                if (GetPipelines(r).Main.Compiler.CompilePipelines(pbr_mask, buffers.DrawsNonTriangles())) request();
                const bool want_transmission = active_lighting.RealTransmission && HasFeature(pbr_mask, PbrFeature::Transmission);
                const auto te_px = RenderExtentPx(r);
                if (targets.EnsureTransmissionResources(ctx, std::bit_cast<mtl::Extent2D>(te_px), want_transmission)) refresh_transmission_sampler();
            } else if (targets.EnsureTransmissionResources(ctx, {}, false)) {
                refresh_transmission_sampler();
            }
        }
    }

    end();
    // Send pending edit deltas through the view UBO.
    // Object transforms preserve local instance bounds.
    if (is_edit_mode && AnyChanged(r, Change::TransformPending, Change::TransformEnd)) scene_state.EditPreludePending = true;

    const auto render_extent = RenderExtentPx(r);
    if (buffers.FrameView != RenderView{r.get<const ViewCamera>(viewport), render_extent} ||
        AnyChanged(r, Change::SceneView, Change::TransformPending, Change::ViewportDisplay, Change::TransformEnd) ||
        mode_changed ||
        light_count_changed ||
        resized) {
        const float aspect = render_extent.x == 0 || render_extent.y == 0 ? 1.f : float(render_extent.x) / float(render_extent.y);
        // Update widened scene-camera FOV after viewport aspect-ratio changes.
        if (const auto camera = LookThroughCameraEntity(r); camera != state::Null) {
            r.edit<ViewCamera>(viewport).Data = WidenForLookThrough(*LensOf(r, camera), aspect);
        }
        const auto &camera = r.get<const ViewCamera>(viewport);
        const auto &settings = r.get<const ViewportDisplay>(viewport);
        const bool is_pbr_mode = !WorkbenchShading(settings.ViewportShading);
        const auto &active_lighting = GetActivePbrLighting(r, viewport, settings.ViewportShading);
        const bool use_scene_lights = is_pbr_mode && active_lighting.UseSceneLights;
        const bool use_scene_world = is_pbr_mode && active_lighting.UseSceneWorld;
        const auto &active_environment = use_scene_world ? environments.SceneWorld : environments.StudioWorld;
        const auto *image_light = r.try_get<const ImageLight>(viewport);
        const float env_intensity = use_scene_world && image_light ? image_light->Intensity : active_lighting.EnvIntensity;
        const mat3 env_rotation = [&]() -> mat3 {
            if (use_scene_world) return image_light ? ToMat3(image_light->Rotation) : mat3{1.f};
            const float radians = active_lighting.EnvRotationDegrees * (Pi / 180.f);
            const float s = std::sin(radians), c = std::cos(radians);
            return {c, 0, -s, 0, 1, 0, s, 0, c};
        }();
        const float background_blur = active_lighting.BackgroundBlur;
        const float world_opacity = is_pbr_mode ? active_lighting.WorldOpacity : 0.f;
        const auto *pending = r.try_get<const PendingTransform>(viewport);
        buffers.FrameView = {camera, render_extent};
        const auto &mesh_slots = meshes.Slots();
        SceneViewUBO view{
            .LightCount = buffers.Lights.Count<LightRecord>(),
            .LightSlot = buffers.Lights.Slot,
            .UseSceneLightsRender = use_scene_lights ? 1u : 0u,
            .EnvIntensity = env_intensity,
            .Exposure = std::exp2(active_lighting.ExposureEV),
            .EnvRotation = env_rotation,
            .BackgroundBlur = background_blur,
            .WorldOpacity = world_opacity,
            .Ibl = active_environment.Ibl,
            .InteractionMode = interaction_mode,
            .EditElement = r.get<const EditMode>(viewport).Value,
            .IsTransforming = pending ? 1u : 0u,
            .PendingPivot = pending ? pending->Pivot : vec3{},
            .PendingTranslation = pending ? pending->Delta.P : vec3{},
            .PendingRotation = pending ? pending->Delta.R : quat{1, 0, 0, 0},
            .PendingScale = pending ? pending->Delta.S : vec3{1},
            .LodErrorPixels = settings.LodErrorPixels,
            .EdgeSharpnessSlot = mesh_slots.EdgeSharpness,
            .FaceSharpnessSlot = mesh_slots.FaceSharpness,
            .BaseVertexNormalSlot = mesh_slots.BaseVertexNormal,
            .BaseFaceNormalSlot = mesh_slots.BaseFaceNormal,
            .FaceTriangleStartSlot = mesh_slots.FaceTriangleStart,
            .ArmatureDeformSlot = buffers.ArmatureDeformBuffer.Buffer.Slot,
            .MorphWeightsSlot = buffers.MorphWeightBuffer.Buffer.Slot,
            .PosedPositionSlot = buffers.PosedPositions.Values.Buffer.Slot,
            .PosedPositionNodesSlot = buffers.PosedPositions.Nodes.Buffer.Slot,
            .PosedVertexNormalSlot = buffers.PosedVertexNormals.Values.Buffer.Slot,
            .PosedVertexNormalNodesSlot = buffers.PosedVertexNormals.Nodes.Buffer.Slot,
            .PosedSectorNodesSlot = buffers.PosedSectors.Nodes.Buffer.Slot,
            .PosedSectorValuesSlot = buffers.PosedSectors.Values.Buffer.Slot,
            .PosedFaceNormalSlot = buffers.PosedFaceNormals.Values.Buffer.Slot,
            .PosedFaceNormalNodesSlot = buffers.PosedFaceNormals.Nodes.Buffer.Slot,
            .PosedMorphNormalDeltaSlot = buffers.PosedMorphNormalDeltas.Values.Buffer.Slot,
            .PosedMorphNormalNodesSlot = buffers.PosedMorphNormalDeltas.Nodes.Buffer.Slot,
            .InstanceBoundsSlot = buffers.Instances.BoundsBuffer.Slot,
            .MaterialSlot = buffers.Materials.Slot,
            .PrimitiveMaterialSlot = mesh_slots.PrimitiveMaterial,
            .MeshRecordSlot = render.MeshRecords.Buffer.Slot,
            .InstanceRecordSlot = buffers.Instances.RecordBuffer.Slot,
            .InstanceStateSlot = buffers.Instances.StateBuffer.Slot,
            .BoneXRay = settings.ViewportShading == ViewportShadingMode::Wireframe ? 1u : 0u,
            .XRayAlpha = XRayActive(settings) && settings.ViewportShading == ViewportShadingMode::Solid ? XRayOpacity(settings) : 1.f,
            .OverlayBehindOpacity = OverlayBehindOpacity(settings, r.get<const Interaction>(viewport).Mode),
            .SceneDepthSamplerSlot = r.Context.get<const RenderSamplerSlots>().SceneDepth,
            .ShowOverlays = settings.ShowOverlays ? 1u : 0u,
            .ShowExtras = settings.ShowExtras ? 1u : 0u,
            .ShowBoundingBoxes = settings.ShowBoundingBoxes ? 1u : 0u,
            .ShowTetWireframe = settings.ShowTetWireframe ? 1u : 0u,
            .TransmissionFramebufferSamplerSlot = r.Context.get<const RenderSamplerSlots>().Transmission,
            .TransmissionFramebufferMipCount = targets.Transmission ? targets.Transmission->Image.MipLevels : 1u,
            .UseRealTransmission = (is_pbr_mode && active_lighting.RealTransmission && targets.Transmission) ? 1u : 0u,
            .DebugChannel = is_pbr_mode ? settings.DebugChannel : DebugChannel::None,
        };
        buffers.FrameView.ApplyTo(view);
        buffers.SceneViewUBO.Update(as_bytes(view));
        request();
    }

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
        LayOutMeshes(r, viewport);
        request();
    }
    const bool displays_refreshed = !r.view<const MeshDisplayDirty>().empty();
    if (displays_refreshed) {
        RefreshMeshDisplays(r, viewport);
        request();
    }
    if (displays_refreshed || mode_changed ||
        AnyChanged(r, Change::Selected, Change::ActiveInstance, Change::InstanceVisibility, Change::RenderInstanceDestroyed, Change::MeshGeometry)) {
        UpdateSilhouetteWork(r, viewport);
    }
    if (!rendering) {
        UpdateModalPlacement(r);
        UpdateAudioContacts(r);
    }
    r.ClearChanges();
    r.clear<MeshGeometryDirty, MeshPositionsChanged, MeshMaterialAssignment, MeshDisplayDirty>();
}

void RegisterSceneComponentHandlers(state::Scene &r) {
    r.on_destroy<MeshHandle, [](state::Scene &r, state::Entity e) { r.remove<MeshEditWork>(e); }>();
    r.on_destroy<MeshEditWork, &ReleaseMeshEditWorkRanges>();
    r.on_destroy<BoundsRun, &ReleaseMeshLayout>();
    reactive(r, Change::Selected).on<Selected>(On::Create | On::Destroy);
    reactive(r, Change::ActiveInstance).on<Active>(On::Create | On::Destroy);
    reactive(r, Change::BoneSelection).on<BoneSelection>(On::Create | On::Update | On::Destroy).on<BoneActive>(On::Create | On::Destroy);
    reactive(r, Change::MeshActiveElement).on<MeshActiveElement>(On::Create | On::Update);
    reactive(r, Change::MeshGeometry).on<MeshGeometryDirty>(On::Create).on<MeshPositionsChanged>(On::Create);
    reactive(r, Change::PhysicsBodyMesh).on<PhysicsBodyHandle>(On::Create | On::Destroy).on<ColliderShape>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::MeshMaterial).on<MeshMaterialAssignment>(On::Create | On::Update);
    reactive(r, Change::PrimitiveShape).on<PrimitiveShape>(On::Update);
    reactive(r, Change::SoundVertices).on<SoundVertices>(On::Create | On::Destroy);
    reactive(r, Change::SoundVerticesUpdated).on<SoundVertices>(On::Update);
    reactive(r, Change::VertexForce).on<VertexForce>(On::Create | On::Destroy);
    reactive(r, Change::TetMesh).on<TetBuffers>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::NewBufferEntity).on<MeshHandle>(On::Create).on<VertexStoreId>(On::Create);
    reactive(r, Change::InstanceVisibility).on<Instance>(On::Create | On::Update | On::Destroy).on<Hidden>(On::Create | On::Destroy);
    reactive(r, Change::RenderInstanceCreated).on<RenderInstance>(On::Create);
    reactive(r, Change::RenderInstanceDestroyed).on<RenderInstance>(On::Destroy);
    reactive(r, Change::ViewportDisplay).on<ViewportDisplay>(On::Create | On::Update);
    reactive(r, Change::InteractionMode).on<Interaction>(On::Create | On::Update);
    reactive(r, Change::WorkspaceLights).on<WorkspaceLights>(On::Create | On::Update);
    reactive(r, Change::ViewportTheme).on<ViewportTheme>(On::Create | On::Update);
    reactive(r, Change::MaterializedTextures).on<MaterializedTextures>(On::Create | On::Update);
    reactive(r, Change::StudioEnvironment).on<StudioEnvironment>(On::Create | On::Update);
    reactive(r, Change::SceneWorld).on<gltf::SourceAssets>(On::Create | On::Update);
    reactive(r, Change::PunctualLight).on<PunctualLight>(On::Create | On::Update).on<RenderInstance>(On::Create).on<Hidden>(On::Destroy);
    reactive(r, Change::ActiveMaterialVariant).on<MaterialVariants>(On::Create | On::Update);
    reactive(r, Change::PbrMeshFeatures).on<PbrMeshFeatures>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::PbrSpecialization)
        .on<MaterialPreviewLighting>(On::Create | On::Update)
        .on<RenderedLighting>(On::Create | On::Update);
    reactive(r, Change::SceneView)
        .on<ViewCamera>(On::Create | On::Update)
        .on<ImageLight>(On::Create | On::Update)
        .on<MaterialPreviewLighting>(On::Create | On::Update)
        .on<RenderedLighting>(On::Create | On::Update)
        .on<LightIndex>(On::Create | On::Destroy)
        .on<EditMode>(On::Create | On::Update);
    reactive(r, Change::CameraLens).on<Perspective>(On::Create | On::Update).on<Orthographic>(On::Create | On::Update).on<LookingThrough>(On::Create | On::Destroy);
    // Publish world-transform changes through SetWorldTransform so bone display scaling reaches instance transforms.
    reactive(r, Change::WorldTransform).on<BoneDisplayScale>(On::Update);
    reactive(r, Change::SceneParent).on<SceneParent>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::SceneHierarchy).on<SceneParent>(On::Create | On::Update | On::Destroy).on<SceneChildren>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::Names).on<Name>(On::Create | On::Destroy);
    reactive(r, Change::ScaleLocked).on<ScaleLocked>(On::Create | On::Destroy);
    reactive(r, Change::EditMode).on<EditMode>(On::Create | On::Update);
    reactive(r, Change::KeyframeSources).on<TimelineRange>(On::Create | On::Update).on<MeshMaterialSlotSelection>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::TransformPending).on<PendingTransform>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::TransformStart).on<StartTransform>(On::Create);
    reactive(r, Change::TransformEnd).on<StartTransform>(On::Destroy);
    reactive(r, Change::BonePose).on<BoneDelta>(On::Update);
    reactive(r, Change::BoneConstraints).on<BoneConstraints>(On::Create | On::Update | On::Destroy);
    r.Context.emplace<ConstraintTargets>();
    reactive(r, Change::TransformDirty)
        .on<Transform>(On::Create | On::Update)
        .on<PosedLocal>(On::Create | On::Update | On::Destroy)
        .on<SceneParent>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::AnimationEdited)
        .on<AnimationClips>(On::Create | On::Update | On::Destroy)
        .on<Animations>(On::Update);
}
