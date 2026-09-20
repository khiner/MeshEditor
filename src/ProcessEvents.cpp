#include "ProcessEvents.h"
#include "mesh/MeshComponents.h"
#include "numeric/VectorMath.h"
#include "physics/ColliderUpdate.h"
#include "render/MeshUpdates.h"
#include "render/SceneUpdates.h"
#include "state/Scene.h"

#include "Camera.h"
#include "File.h"
#include "Parallel.h"
#include "Profile.h"
#include "TransformMath.h"
#include "Variant.h"
#include "action/Selection.h"
#include "animation/AnimationData.h"
#include "animation/AnimationTimeline.h"
#include "animation/Evaluate.h"
#include "animation/MorphWeights.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "audio/AudioTypes.h"
#include "audio/ContactModel.h"
#include "audio/SoundVertices.h"
#include "editor/AudioIntegration.h"
#include "gizmo/GizmoInteraction.h"
#include "gltf/GltfScene.h"
#include "mesh/MeshBvh.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "mesh/TetBuffers.h"
#include "mesh/VertexAdjacencyGpu.h"
#include "object/ObjectOps.h"
#include "physics/PhysicsSystem.h"
#include "physics/PhysicsTypes.h"
#include "render/ElementWorkOps.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MaterialComponents.h"
#include "render/MeshletBuild.h"
#include "render/PickConstants.h"
#include "render/Pipelines.h"
#include "render/RenderTargets.h"
#include "render/Textures.h"
#include "render/ViewportSubmission.h"
#include "scene/CameraLens.h"
#include "scene/Defaults.h"
#include "scene/EntityDestroyTracker.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "viewport/FrameState.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportConsumerFence.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"
#include "viewport/ViewportRenderGpu.h"

#include <bit>
#include <iostream>
#include <numeric>
#include <print>

using state::Change;
using state::On;

using std::ranges::to;
using std::views::iota;

namespace {
using namespace he;

// Sort indexed writes, apply them to the target span, and return whether any were present.
bool FlushIndexedWrites(auto &writes, auto &&span_getter) {
    if (writes.empty()) return false;
    std::sort(writes.begin(), writes.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    auto span = span_getter();
    for (const auto &[index, value] : writes) span[index] = value;
    return true;
}

vec3 ComputeElementLocalPosition(const Mesh &mesh, Element element, uint32_t handle) {
    if (element == Element::Vertex) return mesh.GetPosition(VH{handle});
    if (element == Element::Edge) {
        const auto heh = mesh.GetHalfedge(EH{handle}, 0);
        return (mesh.GetPosition(mesh.GetFromVertex(heh)) + mesh.GetPosition(mesh.GetToVertex(heh))) * 0.5f;
    }
    return mesh.CalcFaceCentroid(FH{handle});
}

vec3 ComputeElementWorldPosition(const state::Scene &r, state::Entity instance_entity, Element element, uint32_t handle) {
    const auto &mesh = GetMesh(r, r.get<Instance>(instance_entity).Entity);
    const auto &wt = r.get<WorldTransform>(instance_entity);
    return {wt.P + Rotate(wt.R, wt.S * ComputeElementLocalPosition(mesh, element, handle))};
}

void SetEditMode(state::Scene &r, state::Entity viewport, Element mode) {
    const auto current_mode = r.get<const EditMode>(viewport).Value;
    if (current_mode == mode) return;

    auto &meshes = r.Context.get<MeshStore>();
    std::vector<ElementRange> ranges;
    for (const auto mesh_entity : r.view<const MeshElementSelection, const MeshHandle>()) {
        const auto mesh = GetMesh(r, mesh_entity);
        const auto id = mesh.GetStoreId();
        meshes.EnsureSelectionBits(mesh);
        r.remove<MeshActiveElement>(mesh_entity);
        const auto count = mesh.ElementCount(mode);
        if (count > 0) ranges.emplace_back(mesh_entity, meshes.GetSelectionBitOffset(id, mode), count);
    }

    r.patch<EditMode>(viewport, [mode](auto &edit_mode) { edit_mode.Value = mode; });
    if (!ranges.empty()) ApplyEditSelectionCommand(r, ranges, mode, EditSelectionOperation::ClearActive);
}

} // namespace

void ProcessComponentEvents(state::Scene &r, state::Entity viewport, EventPass pass) {
    const bool rendering = pass == EventPass::Sample || pass == EventPass::Render;
    const auto &ctx = r.Context.get<const mtl::Context>();
    auto &slots = r.Context.get<mtl::BindlessSet>();
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &meshes = r.Context.get<MeshStore>();
    auto &textures = r.Context.get<TextureStore>();
    auto &environments = r.Context.get<EnvironmentStore>();
    auto &targets = r.Context.get<RenderTargets>();
    const profile::CpuScope profile_scope{"ProcessEvents"};

    auto &pending_render = r.Context.get<PendingRenderRequest>().Value;
    auto request = [&pending_render, &buffers](RenderRequest req) {
        pending_render = std::max(pending_render, req);
        if (req != RenderRequest::None) buffers.MeshletOcclusionStale = true;
    };

    // Armature objects whose bone instance state resyncs this frame.
    std::unordered_set<state::Entity> bone_state_dirty;

    BuildMissingWorldTransforms(r);

    // Resize render resources before the pick handlers below resolve against the rendered scene.
    const bool resized = SyncViewportRenderResources(r, viewport);
    if (resized) request(RenderRequest::Reuse);

    // Dropping the pipeline sets recompiles every pipeline on its next use.
    const bool recompiled = std::exchange(r.Context.get<FrameState>().RecompileShaders, false);
    if (recompiled) {
        r.Context.get<mtl::LibraryCache>().Clear();
        r.Context.erase<Pipelines>();
        r.Context.erase<MeshPipelines>();
        // Recompiled prefilter kernels must regenerate their cached cubemaps.
        RebuildStudioEnvironments(r);
        request(RenderRequest::Reuse);
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
        auto batch = BeginTextureUploadBatch(ctx);
        for (const auto &item : textures.PendingUploads) {
            auto entry = MaterializeTextureEntry(r, batch, slots, item, gltf_images, r.Context.get<const ActiveSamplerAnisotropy>().Value);
            if (!entry) {
                std::cerr << std::format("Warning: Failed to materialize texture '{}': {}\n", item.Params.Name, entry.error());
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
                std::cerr << std::format("Warning: Failed to materialize EXT_lights_image_based '{}': {}\n", pending_env->Source.Name, pre.error());
                ReleaseCubeSamplerSlot(slots, pending_env->DiffuseCubeSlot);
                ReleaseCubeSamplerSlot(slots, pending_env->SpecularCubeSlot);
            }
        }
        pending_env.reset();
    }
    // Prefilter and activate the studio HDRI named by the StudioEnvironment selection whenever it changes.
    if (!reactive(r, Change::StudioEnvironment).empty()) {
        SetStudioEnvironment(r, r.get<const StudioEnvironment>(viewport).Name);
    }

    // Process pending handlers before consuming the reactive trackers they update.
    if (const auto *pending = r.try_get<const PendingSetEditMode>(viewport)) {
        const auto mode = pending->Mode;
        r.remove<PendingSetEditMode>(viewport);
        SetEditMode(r, viewport, mode);
    }
    if (auto *pending = r.try_get<PendingImportMesh>(viewport)) {
        auto path = std::move(pending->Path);
        auto info = std::move(pending->Info);
        r.remove<PendingImportMesh>(viewport);
        ImportMesh(r, viewport, path, std::move(info));
    }
    // Use the rendered camera for selection, culling, and LOD.
    const auto prepare_selection = [&](const RenderView &view) {
        auto &frame_view = *reinterpret_cast<SceneViewUBO *>(buffers.SceneViewUBO.Contents().data());
        buffers.FrameView = view;
        view.ApplyTo(frame_view);
        // Submit restored geometry and draw records before selection.
        if (pending_render != RenderRequest::None) {
            RecordAndSubmitFrame(r, viewport, pending_render == RenderRequest::Rebuild ? SceneUpdate::Rebuild : SceneUpdate::Reuse, RenderPhase::Prepare);
            WaitForRender(r);
            pending_render = RenderRequest::Reuse;
        }
    };
    // A view fraction as a pixel of the current render target.
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
            const auto &summary = meshes.GetSelectionSummary(GetMesh(r, mesh_entity).GetStoreId());
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
        // An additive drag unions each box with the selection it started from, recorded on the first update.
        // The GPU pass captures the element masks on its first run.
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
            if (bone_mode) {
                r.clear<BoneSelection>();
                if (baseline) {
                    for (const auto &[e, sel] : baseline->BoneSelections) {
                        if (r.valid(e)) r.emplace_or_replace<BoneSelection>(e, sel);
                    }
                }
                for (const auto &hit : hits) {
                    const auto sel = hit.Part ? BoneSelection::From(*hit.Part) : BoneSelection{};
                    const auto *cur = r.try_get<BoneSelection>(hit.Entity);
                    r.emplace_or_replace<BoneSelection>(hit.Entity, additive && cur ? *cur | sel : sel);
                }
            } else {
                r.clear<Selected>();
                if (baseline) {
                    for (const auto e : baseline->SelectedEntities) {
                        if (r.valid(e)) r.emplace_or_replace<Selected>(e);
                    }
                }
                for (const auto &hit : hits) r.emplace_or_replace<Selected>(hit.Entity);
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
            // Cycle to the next overlapping result after a repeated click.
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

    // Refresh camera overlay descriptors after lens changes.
    if (!reactive(r, Change::CameraLens).empty()) request(RenderRequest::Rebuild);

    // Advance playback and evaluate the animation before the render sync, so this frame renders what it writes.
    bool anim_advanced;
    int evaluated_from;
    float eval_seconds{}, frame_seconds{};
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
                const int new_frame = int(std::floor(pf));
                if (new_frame != playback.CurrentFrame) r.patch<TimelinePlayback>(viewport, [&](auto &p) { p.CurrentFrame = new_frame; });
            } else if (!playback.Playing) {
                pf = float(playback.CurrentFrame);
            }
            return playback.CurrentFrame != r.edit<LastEvaluatedFrame>(viewport).Value || !reactive(r, Change::AnimationEdited).empty();
        }();

        evaluated_from = r.edit<LastEvaluatedFrame>(viewport).Value;
        if (anim_advanced || pass == EventPass::Restore) r.edit<LastEvaluatedFrame>(viewport).Value = playback.CurrentFrame;
        // Convert 1-based display frames to animation time and preserve fractional motion-blur samples.
        frame_seconds = float(std::max(0, playback.CurrentFrame - 1)) / range.Fps;
        eval_seconds = pass == EventPass::Sample ? std::max(0.f, pf - 1.f) / range.Fps : frame_seconds;

        // A restore rebuilds only the derived node poses, since history restores every other animated value.
        if (anim_advanced || rendering || pass == EventPass::Restore) animation::Evaluate(r, viewport, eval_seconds, pass != EventPass::Restore);
    }

    auto sync = SyncModelsBuffers(r);
    if (!sync.NewlyInserted.empty() || sync.Compacted) {
        request(RenderRequest::Reuse);
        // Insertion and compaction both reassign the record slots instances write into.
        MarkInstanceRecordsStale(r.Context.get<GpuSceneState>());
    }
    const std::unordered_set<state::Entity> newly_inserted_set(sync.NewlyInserted.begin(), sync.NewlyInserted.end());
    const auto is_newly_inserted = [&](state::Entity e) { return newly_inserted_set.contains(e); };

    // Reconstruct missing derived armature pose state from the canonical pose or rest state.
    bool pose_state_created = false;
    for (const auto [arm_obj_entity, arm_obj_comp] : r.view<const ArmatureObject>().each()) {
        const auto data_entity = arm_obj_comp.Entity;
        const auto *armature = r.try_get<const Armature>(data_entity);
        if (!armature || armature->Bones.empty() || r.all_of<ArmaturePoseState>(data_entity)) continue;
        const auto n = armature->Bones.size();
        r.emplace<ArmaturePoseState>(data_entity, ArmaturePoseState{.BoneUserOffset = std::vector<Transform>(n), .BonePoseWorld = std::vector<mat4>(n, I4)});
        // Bone poses derive from rest + delta. Scale stays at rest.
        for (uint32_t i = 0; i < n && i < arm_obj_comp.BoneEntities.size(); ++i) {
            const auto b = arm_obj_comp.BoneEntities[i];
            if (!r.all_of<BoneDelta>(b)) r.emplace<BoneDelta>(b);
            const auto &rest = armature->Bones[i].RestLocal;
            const auto posed = ComposeWithDelta(rest, r.get<const BoneDelta>(b).Value);
            r.emplace_or_replace<PosedLocal>(b, Transform{posed.P, posed.R, rest.S});
        }
        pose_state_created = true;
    }

    {
        // Allocate deform storage for newly created armature pose state.
        uint32_t total_joints = 0;
        std::vector<state::Entity> pending_armatures;
        for (const auto [arm_obj_entity, arm_obj_comp] : r.view<const ArmatureObject>().each()) {
            auto *pose_state = r.try_edit<ArmaturePoseState>(arm_obj_comp.Entity);
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
        if (!pending_armatures.empty()) request(RenderRequest::Rebuild);
    }

    std::unordered_set<state::Entity> dirty_sound_selection_meshes;

    if (!sync.NewMeshEntities.empty()) {
        const bool overlay_indices = DrawsElementIndices(r, viewport);
        uint32_t total_face = 0, total_edge = 0, total_vertex = 0;
        for (auto entity : sync.NewMeshEntities) {
            const auto &mesh = GetMesh(r, entity);
            if (!DrawsStoredCorners(mesh)) total_face += mesh.TriangleIndexCount();
            if (!NeedsElementIndices(mesh, overlay_indices)) continue;
            total_edge += mesh.EdgeCount() * 2;
            total_vertex += mesh.VertexCount();
        }
        buffers.ReserveAdditionalIndices(total_face, total_edge, total_vertex);
        std::vector<ElementIndicesWork> work;
        for (auto entity : sync.NewMeshEntities) {
            const auto &mesh = GetMesh(r, entity);
            WriteElementIndices(buffers, meshes, mesh, buffers.MeshOf(mesh.GetStoreId()), overlay_indices, work);
        }
        WriteElementIndicesNow(r, work);
        // Fill adjacency tables before normal derivation reads the vertex-fan CSR.
        BuildVertexAdjacencyNow(r, sync.NewMeshEntities);
        // Derive shading state for all new and restored meshes in one batch.
        FinalizeNewMeshShadingNow(r, sync.NewMeshEntities);
        BuildMeshletsNow(r, sync.NewMeshEntities);
        request(RenderRequest::Rebuild);
    }

    if (!sync.NewExtrasEntities.empty()) {
        // Generate the shared bone and joint geometry once.
        static const auto bone = primitive::BoneOctahedron(1.f);
        static const auto &bone_faces = bone.Mesh.FaceCorners;
        static const auto bone_verts = iota(0u, uint32_t(bone.Mesh.Positions.size())) | to<std::vector>();
        static const auto sphere = primitive::BoneSphereDisc();
        static const auto &sphere_faces = sphere.Mesh.FaceCorners;
        static const auto sphere_verts = iota(0u, uint32_t(sphere.Mesh.Positions.size())) | to<std::vector>();

        uint32_t total_face = 0, total_edge = 0, total_vertex = 0;
        std::vector<state::Entity> bone_mesh_entities;
        for (auto entity : sync.NewExtrasEntities) {
            if (r.all_of<ArmatureObject>(entity)) {
                total_face += bone_faces.size();
                total_edge += bone.AdjacencyIndices.size();
                total_vertex += bone_verts.size();
            } else if (r.all_of<BoneJoint>(entity)) {
                total_face += sphere_faces.size();
                total_edge += sphere.OutlineIndices.size();
                total_vertex += sphere_verts.size();
            }
        }
        buffers.ReserveAdditionalIndices(total_face, total_edge, total_vertex);

        for (auto entity : sync.NewExtrasEntities) {
            if (r.all_of<ArmatureObject>(entity)) {
                auto &mb = MeshBuffersOf(r, entity);
                mb.FaceIndices = buffers.CreateIndices(bone_faces, IndexKind::Face);
                mb.VertexIndices = buffers.CreateIndices(bone_verts, IndexKind::Vertex);
                r.emplace_or_replace<BoneAdjacencyIndices>(entity, buffers.CreateIndices(bone.AdjacencyIndices, IndexKind::Edge));
                bone_mesh_entities.push_back(entity);
            } else if (r.all_of<BoneJoint>(entity)) {
                auto &mb = MeshBuffersOf(r, entity);
                mb.FaceIndices = buffers.CreateIndices(sphere_faces, IndexKind::Face);
                mb.EdgeIndices = buffers.CreateIndices(sphere.OutlineIndices, IndexKind::Edge);
                mb.VertexIndices = buffers.CreateIndices(sphere_verts, IndexKind::Vertex);
                bone_mesh_entities.push_back(entity);
            }
        }
        BuildBoneMeshletsNow(r, bone_mesh_entities);
        request(RenderRequest::Rebuild);
    }

    { // Register changed lights into the GPU Lights buffer, the single path for both new and restored lights.
        bool synced = false;
        for (const auto entity : reactive(r, Change::PunctualLight)) {
            if (!r.all_of<PunctualLight, Instance>(entity)) continue;
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
            if (index >= buffers.Lights.Count<LightRecord>()) request(RenderRequest::Rebuild);
            else {
                const auto old = buffers.Lights.GetSpan<LightRecord>()[index];
                if (old.Type != gpu_light.Type || old.Range != gpu_light.Range || old.OuterConeCos != gpu_light.OuterConeCos || old.InnerConeCos != gpu_light.InnerConeCos) {
                    request(RenderRequest::Rebuild);
                }
            }
            buffers.Lights.Update(as_bytes(gpu_light), uint64_t(index) * sizeof(LightRecord));
            synced = true;
        }
        if (synced) request(RenderRequest::Reuse);
    }

    // A hidden light leaves the light buffer.
    for (const auto entity : reactive(r, Change::RenderInstanceDestroyed)) {
        if (const auto *light_index = r.try_get<const LightIndex>(entity)) {
            buffers.PendingLightRemovals.emplace_back(light_index->Value);
            r.remove<LightIndex>(entity);
        }
    }
    // Compact destroyed light indices in one batch.
    if (auto &indices = buffers.PendingLightRemovals; !indices.empty()) {
        std::sort(indices.begin(), indices.end(), std::greater<>());
        auto buffer_count = buffers.Lights.Count<LightRecord>();
        for (const auto remove_index : indices) {
            if (remove_index >= buffer_count) continue;
            --buffer_count;
            if (remove_index != buffer_count) {
                buffers.Lights.Update(as_bytes(buffers.Lights.GetSpan<LightRecord>()[buffer_count]), uint64_t(remove_index) * sizeof(LightRecord));
                for (auto [other_entity, other_light_index] : r.view<LightIndex>().each()) {
                    if (other_light_index.Value == buffer_count) {
                        r.replace<LightIndex>(other_entity, remove_index);
                        break;
                    }
                }
            }
        }
        buffers.Lights.SetCount<LightRecord>(buffer_count);
        indices.clear();
        request(RenderRequest::Rebuild);
    }

    // Commit mesh edit transforms after StartTransform is cleared.
    if (!reactive(r, Change::TransformEnd).empty()) {
        if (r.get<const Interaction>(viewport).Mode == InteractionMode::Edit && FindArmatureObject(r, FindActiveEntity(r)) == state::Null) {
            if (const auto *pending = r.try_get<const PendingTransform>(viewport); pending && pending->Delta != Transform{}) {
                // Evaluate the final edit before geometry and physics consumers run.
                std::vector<state::Entity> commit_meshes;
                for (const auto &[mesh_entity, instance_entity] : selection::ComputePrimaryEditInstances(r, false)) {
                    if (!selection::HasScaleLockedInstance(r, mesh_entity)) commit_meshes.push_back(mesh_entity);
                }
                for (const auto mesh_entity : CommitPosedGeometry(r, viewport, commit_meshes)) {
                    r.remove<PrimitiveShape>(mesh_entity);
                    r.emplace_or_replace<MeshPositionsChanged>(mesh_entity);
                }
            }
            r.remove<PendingTransform>(viewport);
        }
    }

    for (auto entity : reactive(r, Change::MeshGeometry)) {
        if (!r.all_of<MeshPositionsChanged>(entity)) continue;
        const auto &work = r.Context.get<const GpuSceneState>().EditWork.at(entity);
        if (auto *bvh = r.try_edit<MeshBvh>(entity)) {
            const auto mesh = GetMesh(r, entity);
            const auto indices = GetFaceIndices(r, mesh);
            const auto first = meshes.Arenas().FaceFirstTriangles.Get(meshes.Get(mesh.GetStoreId()).FaceData);
            std::vector<uint32_t> triangles;
            ForEachWorkElement(buffers.GeometryWork, work.Faces, [&](uint32_t f) {
                const auto end = f + 1 < first.size() ? first[f + 1] : uint32_t(indices.size() / 3);
                for (auto t = first[f]; t < end; ++t) triangles.push_back(t);
            });
            bvh->Refit(mesh.GetVerticesSpan(), indices, triangles);
            ForEachWorkElement(buffers.GeometryWork, work.Normals, [&](uint32_t v) {
                if (v < mesh.VertexCount()) bvh->MeanCurvature[v] = mesh.CalcMeanCurvature(VH{v}, meshes.Arenas().EdgeSharpness.Get(meshes.Get(mesh.GetStoreId()).EdgeSharpness));
            });
            bvh->EnclosedVolume = mesh.CalcEnclosedVolume();
        }
    }
    // Restored collider shapes already hold their persisted derivation.
    if (pass != EventPass::Restore) {
        std::unordered_set<state::Entity> to_rederive;
        for (auto e : reactive(r, Change::ColliderPolicy)) to_rederive.insert(e);
        if (const auto &mesh_dirty = reactive(r, Change::MeshGeometry); !mesh_dirty.empty()) {
            for (auto [ce, cs] : r.view<const ColliderShape>().each()) {
                const auto me = cs.MeshEntity != state::Null ? cs.MeshEntity : FindMeshEntity(r, ce);
                if (mesh_dirty.contains(me)) to_rederive.insert(ce);
            }
        }
        for (auto e : to_rederive) RederiveCollider(r, e);
    }

    if (!rendering) {
        physics::ProcessChanges(r, pass);
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
            if (interaction_mode == InteractionMode::Excite) request(RenderRequest::Rebuild);
            else SetInteractionMode(r, viewport, InteractionMode::Excite);
        }
    }

    // Sound-model changes can enter Excite mode; selection derivation needs its indices in this pass.
    if (const bool draws_element_indices = DrawsElementIndices(r, viewport); draws_element_indices != buffers.DrewElementIndices) {
        buffers.DrewElementIndices = draws_element_indices;
        if (draws_element_indices) {
            uint32_t total_edge = 0, total_vertex = 0;
            for (const auto [entity, handle] : r.view<const MeshHandle>().each()) {
                const auto *mb = buffers.TryMeshOf(handle.StoreId);
                if (!mb) continue;
                const auto &mesh = GetMesh(r, entity);
                if (mb->EdgeIndices.Count == 0) total_edge += mesh.EdgeCount() * 2;
                if (mb->VertexIndices.Count == 0) total_vertex += mesh.VertexCount();
            }
            if (total_edge > 0 || total_vertex > 0) {
                buffers.ReserveAdditionalIndices(0, total_edge, total_vertex);
                std::vector<ElementIndicesWork> work;
                for (const auto [entity, handle] : r.view<const MeshHandle>().each()) {
                    if (auto *mb = buffers.TryMeshOf(handle.StoreId)) WriteElementIndices(buffers, meshes, GetMesh(r, entity), *mb, true, work);
                }
                WriteElementIndicesNow(r, work);
                request(RenderRequest::Rebuild);
            }
        }
    }

    {
        auto &selected_tracker = reactive(r, Change::Selected);
        auto &active_tracker = reactive(r, Change::ActiveInstance);
        if ((!selected_tracker.empty() || !active_tracker.empty()) && r.get<const Interaction>(viewport).Mode == InteractionMode::Edit)
            r.Context.get<GpuSceneState>().EditPreludePending = true;
        if (!selected_tracker.empty()) {
            // Edit-mode selection changes the fill, edge, and point batches.
            const auto mode = r.get<const Interaction>(viewport).Mode;
            request(mode == InteractionMode::Edit ? RenderRequest::Rebuild : RenderRequest::Silhouette);
            r.Context.get<GpuSceneState>().InstanceFlagsStale = true;
        }

        // SyncModelsBuffers writes the full initial state for newly inserted instances, so they are skipped here.
        std::vector<std::pair<uint32_t, uint8_t>> state_writes;
        const auto collect_instance_state = [&](state::Entity instance_entity) {
            if (is_newly_inserted(instance_entity)) return;
            if (const auto *ri = r.try_get<RenderInstance>(instance_entity); ri && ri->BufferIndex != UINT32_MAX) {
                state_writes.emplace_back(ri->BufferIndex, InstanceStateBits(r, instance_entity));
            }
        };
        for (auto instance_entity : selected_tracker) {
            collect_instance_state(instance_entity);
            if (const auto arm = FindArmatureObject(r, instance_entity); arm != state::Null) bone_state_dirty.insert(arm);
        }
        for (auto instance_entity : active_tracker) {
            collect_instance_state(instance_entity);
            if (const auto arm = FindArmatureObject(r, instance_entity); arm != state::Null) bone_state_dirty.insert(arm);
        }

        if (FlushIndexedWrites(state_writes, [&] { return buffers.Instances.GetMutableStates(); })) request(RenderRequest::Reuse);
    }
    {
        auto &bone_sel_tracker = reactive(r, Change::BoneSelection);
        if (!bone_sel_tracker.empty()) {
            request(RenderRequest::Silhouette);
            for (auto bone_entity : bone_sel_tracker) {
                if (const auto arm = FindArmatureObject(r, bone_entity); arm != state::Null) bone_state_dirty.insert(arm);
            }
        }
    }
    auto &destroy_tracker = r.Context.get<EntityDestroyTracker>();
    if (!reactive(r, Change::Rerecord).empty() || !destroy_tracker.Storage.empty()) {
        request(RenderRequest::Rebuild);
        // Instance lifecycle and slot changes invalidate meshlet records because mesh-keyed signatures omit instance slots and object IDs.
        MarkInstanceRecordsStale(r.Context.get<GpuSceneState>());
    }

    const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
    const bool is_edit_mode = interaction_mode == InteractionMode::Edit;

    const auto orbit_to_active = [&](state::Entity instance_entity, Element element, uint32_t handle) {
        if (!r.get<const OrbitToActive>(viewport).Value) return;
        const auto world_pos = ComputeElementWorldPosition(r, instance_entity, element, handle);
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
        for (auto mesh_entity : tracker) {
            if (const auto *active_element = r.try_get<MeshActiveElement>(mesh_entity);
                active_element && edit_mode != Element::None && active_instance && active_instance->Entity == mesh_entity) {
                orbit_to_active(active_entity, edit_mode, active_element->Handle);
            }
            if (interaction_mode == InteractionMode::Excite) dirty_sound_selection_meshes.insert(mesh_entity);
        }
    }
    for (auto instance_entity : reactive(r, Change::VertexForce)) {
        if (interaction_mode == InteractionMode::Excite) {
            if (const auto *inst = r.try_get<Instance>(instance_entity)) dirty_sound_selection_meshes.insert(inst->Entity);
        }
        if (const auto *ev = r.try_get<VertexForce>(instance_entity)) orbit_to_active(instance_entity, Element::Vertex, ev->Vertex);
    }
    for (auto instance_entity : reactive(r, Change::SoundVerticesUpdated)) {
        if (interaction_mode == InteractionMode::Excite) {
            if (const auto *inst = r.try_get<const Instance>(instance_entity)) dirty_sound_selection_meshes.insert(inst->Entity);
        }
    }
    for (auto camera_entity : reactive(r, Change::CameraLens)) {
        // Update the viewport FOV when it uses the changed scene camera.
        if (HasLens(r, camera_entity) && r.all_of<LookingThrough>(camera_entity)) r.patch<ViewCamera>(viewport, [](auto &) {});
    }
    bool light_count_changed = false;
    if (const uint32_t required_count = r.view<const LightIndex>().size();
        buffers.Lights.Count<LightRecord>() != required_count) {
        buffers.Lights.SetCount<LightRecord>(required_count);
        light_count_changed = true;
    }
    if (!reactive(r, Change::WorkspaceLights).empty()) {
        buffers.WorkspaceLightsUBO.Update(as_bytes(r.get<const WorkspaceLights>(viewport)));
        request(RenderRequest::Reuse);
    }
    if (!is_edit_mode && !r.Context.get<GpuSceneState>().EditWork.empty()) {
        auto &edit_work = r.Context.get<GpuSceneState>().EditWork;
        std::vector<state::Entity> edited;
        for (const auto &[e, work] : edit_work) {
            if (work.Modified && r.valid(e) && r.all_of<MeshHandle>(e)) edited.push_back(e);
        }
        std::ranges::sort(edited);
        BuildMeshletsNow(r, edited);
        for (auto e : edited)
            if (r.all_of<MeshBvh>(e)) UpdateMeshBvh(r, e);
        while (!edit_work.empty()) ReleaseMeshEditWork(r, edit_work.begin()->first);
        buffers.PreludeStale = true;
        request(RenderRequest::Rebuild);
    }
    if (auto &tracker = reactive(r, Change::MeshShading); !tracker.empty()) {
        // Reclassify corners and derive base normals after sharpness changes.
        std::vector<state::Entity> reclassified;
        for (auto mesh_entity : tracker) {
            if (const auto mesh = TryGetMesh(r, mesh_entity)) {
                const auto [any, all] = meshes.GetFaceSharpnessSummary(mesh->GetStoreId());
                r.emplace_or_replace<MeshShadingSummary>(mesh_entity, any, all);
                meshes.UpdateCornerClassification(*mesh);
                reclassified.emplace_back(mesh_entity);
            }
        }
        if (!reclassified.empty()) {
            DeriveBaseNormalsNow(r, reclassified);
            BuildMeshletsNow(r, reclassified);
            // Reclassification can reallocate arenas whose offsets persistent scene descriptors carry.
            request(RenderRequest::Rebuild);
        }
    }
    // Persistent overlay jobs reference tet arena ranges.
    if (!reactive(r, Change::TetMesh).empty()) request(RenderRequest::Rebuild);
    // Maintain closest-point hierarchies for meshes reachable from contact-reporting bodies.
    if (!reactive(r, Change::PhysicsBodyMesh).empty()) {
        // Collider parameters live in persistent overlay jobs.
        request(RenderRequest::Rebuild);

        std::vector<state::Entity> demanded;
        const auto is_body = [&r](state::Entity a) { return r.all_of<PhysicsBodyHandle>(a); };
        for (const auto [node, inst] : r.view<const Instance>().each()) {
            // Build hierarchies only for reachable mesh entities.
            if (FindAncestorIf(r, node, is_body) != state::Null && HasMesh(r, inst.Entity)) demanded.push_back(inst.Entity);
        }
        std::ranges::sort(demanded);
        const auto repeats = std::ranges::unique(demanded);
        demanded.erase(repeats.begin(), repeats.end());
        // Defer edited hierarchies to the geometry pass below.
        for (const auto mesh_entity : demanded) {
            if (!r.all_of<MeshBvh>(mesh_entity)) UpdateMeshBvh(r, mesh_entity);
        }
        std::vector<state::Entity> unreached;
        for (const auto mesh_entity : r.view<const MeshBvh>()) {
            if (!std::ranges::binary_search(demanded, mesh_entity)) unreached.push_back(mesh_entity);
        }
        for (const auto mesh_entity : unreached) r.remove<MeshBvh>(mesh_entity);
    }
    if (auto &tracker = reactive(r, Change::MeshGeometry); !tracker.empty()) {
        // Vertex-arena positions feed the pose pre-pass, so geometry edits re-run the prelude.
        if (std::ranges::any_of(tracker, [&](auto e) { return r.all_of<MeshGeometryDirty>(e); })) buffers.PreludeStale = true;
        const auto edit_mode = r.get<const EditMode>(viewport).Value;
        std::vector<ElementRange> reset_ranges, carried_ranges;
        // Rebuild edited meshlets before rendering the same frame.
        std::vector<state::Entity> edited, rebuilt;
        for (auto e : tracker)
            if (r.all_of<MeshGeometryDirty>(e)) edited.push_back(e);
        // A new mesh's meshlets were built above.
        for (auto e : edited)
            if (std::ranges::find(sync.NewMeshEntities, e) == sync.NewMeshEntities.end()) rebuilt.push_back(e);
        BuildMeshletsNow(r, rebuilt);
        for (auto mesh_entity : edited) {
            // Rebuild existing closest-point hierarchies after geometry edits.
            if (r.all_of<MeshBvh>(mesh_entity)) UpdateMeshBvh(r, mesh_entity);
            const auto selection_after = r.get<const MeshGeometryDirty>(mesh_entity).Selection;
            if (selection_after == EditSelectionAfter::Keep || !r.all_of<MeshElementSelection>(mesh_entity) || edit_mode == Element::None) continue;
            // Topology changed: size the bits to the new element counts, then drop a stale selection or derive a carried one.
            const auto mesh = GetMesh(r, mesh_entity);
            meshes.EnsureSelectionBits(mesh);
            const auto id = mesh.GetStoreId();
            const uint32_t count = mesh.ElementCount(edit_mode);
            if (count == 0) continue;
            auto &ranges = selection_after == EditSelectionAfter::Reset ? reset_ranges : carried_ranges;
            ranges.emplace_back(mesh_entity, meshes.GetSelectionBitOffset(id, edit_mode), count);
        }
        if (!reset_ranges.empty()) ApplyEditSelectionCommand(r, reset_ranges, edit_mode, EditSelectionOperation::Clear);
        if (!carried_ranges.empty()) ApplyEditSelectionCommand(r, carried_ranges, edit_mode, EditSelectionOperation::Derive);
        request(RenderRequest::Reuse);
    }
    // A preview change moves the entity's drawn record. A new preview took the new-record path above, and a dropped one returns the mesh.
    if (auto &tracker = reactive(r, Change::MeshPreview); !tracker.empty()) {
        std::vector<state::Entity> returned;
        for (auto e : tracker) {
            if (!r.valid(e) || !HasMesh(r, e)) continue;
            const auto [any, all] = meshes.GetFaceSharpnessSummary(GetMesh(r, e).GetStoreId());
            r.emplace_or_replace<MeshShadingSummary>(e, any, all);
            if (!r.all_of<MeshGeometryDirty>(e)) returned.push_back(e);
        }
        if (!returned.empty()) {
            std::ranges::sort(returned);
            for (auto e : returned)
                if (r.all_of<MeshBvh>(e)) UpdateMeshBvh(r, e);
            RepointMeshInstances(r, returned);
            buffers.PreludeStale = true;
            request(RenderRequest::Rebuild);
        }
    }
    // Every meshlet build in this pass has committed, so unpinned meshes take their hierarchy here.
    if (BuildDemandedClusterLods(r, is_edit_mode)) request(RenderRequest::Rebuild);
    if (auto &tracker = reactive(r, Change::MeshMaterial); !tracker.empty()) {
        for (auto mesh_entity : tracker) {
            const auto *assignment = r.try_get<const MeshMaterialAssignment>(mesh_entity);
            const auto mesh = TryGetMesh(r, mesh_entity);
            if (!assignment || !mesh) continue;
            const auto material_count = buffers.Materials.Count<PBRMaterial>();
            if (material_count == 0u) continue;
            auto primitive_materials = meshes.EditPrimitiveMaterials(mesh->GetStoreId());
            if (assignment->PrimitiveIndex < primitive_materials.size()) {
                primitive_materials[assignment->PrimitiveIndex] = std::min(assignment->MaterialIndex, material_count - 1u);
            }
        }
        request(RenderRequest::Rebuild);
    }
    if (!reactive(r, Change::ViewportTheme).empty()) {
        auto theme = r.get<const ViewportTheme>(viewport);
        UpdateDerivedColors(theme);
        theme.EdgeWidth *= r.Context.get<FrameState>().DisplayFramebufferScale.x;
        buffers.ViewportThemeUBO.Update(as_bytes(theme));
        request(RenderRequest::Reuse);
    }
    if (!reactive(r, Change::ActiveMaterialVariant).empty()) {
        const auto *mv = r.try_get<const MaterialVariants>(viewport);
        const auto active = mv ? mv->Active : std::nullopt;
        for (const auto [e, layout, _] : r.view<const MeshSourceLayout, const MeshHandle>().each()) {
            const auto mesh = GetMesh(r, e);
            auto primitive_materials = meshes.EditPrimitiveMaterials(mesh.GetStoreId());
            for (size_t i = 0; i < layout.DefaultMaterials.size(); ++i) {
                const auto &mapping = layout.VariantMappings[i];
                primitive_materials[i] = active && *active < mapping.size() && mapping[*active] ?
                    *mapping[*active] :
                    layout.DefaultMaterials[i];
            }
        }
        request(RenderRequest::Rebuild);
    }
    if (!reactive(r, Change::ViewportDisplay).empty()) {
        request(RenderRequest::Rebuild);
        if (const float requested = ClampMaxAnisotropy(ToMaxAnisotropy(r.get<const ViewportDisplay>(viewport).AnisotropicFilter));
            requested != r.Context.get<const ActiveSamplerAnisotropy>().Value) {
            r.Context.get<ActiveSamplerAnisotropy>().Value = requested;
            RebuildTextureSamplers(ctx, slots, textures, requested);
        }
    }
    if (!reactive(r, Change::InteractionMode).empty()) {
        // Entering edit mode replaces animation deformation with the rest pose even when storage is unchanged.
        buffers.PreludeStale = true;
        request(RenderRequest::Rebuild);
        if (interaction_mode == InteractionMode::Excite) {
            for (const auto [_, instance, __] : r.view<const Instance, const SoundVertices>().each()) {
                dirty_sound_selection_meshes.insert(instance.Entity);
            }
        }
        // Mark all armatures dirty for bone state + pose sync on mode change.
        for (const auto arm : r.view<const ArmatureObject>()) bone_state_dirty.insert(arm);
    }

    const bool mode_changed = !reactive(r, Change::InteractionMode).empty();
    {
        const auto &range = r.get<const TimelineRange>(viewport);
        // Use interpolation instead of advancing physics during motion-blur sub-frames.
        if (!rendering && physics::AdvancePlayback(r, viewport, evaluated_from, r.get<const TimelinePlayback>(viewport).CurrentFrame, range.StartFrame, range.EndFrame, range.Fps)) request(RenderRequest::Reuse);
    }
    // Evaluation writes materials and morph weights, so their consumers follow it.
    if (!reactive(r, Change::Materials).empty()) request(RenderRequest::Rebuild);
    if (!reactive(r, Change::MorphWeights).empty()) {
        buffers.PreludeStale = true;
        request(RenderRequest::Reuse);
    }
    {
        const bool is_object_mode = interaction_mode == InteractionMode::Object;
        for (const auto arm_obj_entity : bone_state_dirty) {
            if (!r.valid(arm_obj_entity) || !TryMeshBuffers(r, arm_obj_entity)) continue;
            const auto &arm_obj = r.get<const ArmatureObject>(arm_obj_entity);
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
            const bool is_edit = interaction_mode == InteractionMode::Edit;
            auto compute_state = [&](state::Entity b, BoneSel part) {
                if (is_object_mode) return max_state;
                const auto *parts = r.try_get<const BoneSelection>(b);
                const bool selected = is_edit ?
                    parts && (part == BoneSel::Body ? parts->Body : part == BoneSel::Root ? parts->Root :
                                                                                            parts->Tip) :
                    r.all_of<BoneSelection>(b);
                uint8_t s = r.all_of<BoneActive>(b) ? ElementStateActive : 0;
                if (selected) s |= ElementStateSelected;
                return s;
            };

            for (const auto b : bone_entities) {
                if (const auto *ri = r.try_get<RenderInstance>(b)) {
                    const auto state = compute_state(b, BoneSel::Body);
                    buffers.Instances.UpdateState(ri->BufferIndex, state);
                }
            }
            if (arm_obj.JointEntity != state::Null && r.valid(arm_obj.JointEntity)) {
                for (const auto b : bone_entities) {
                    const auto *joints = r.try_get<const BoneJointEntities>(b);
                    if (!joints) continue;
                    for (const auto &[je, part] : {std::pair{joints->Head, BoneSel::Root}, {joints->Tail, BoneSel::Tip}}) {
                        if (je != state::Null) {
                            if (const auto *ri = r.try_get<const RenderInstance>(je)) {
                                const auto state = compute_state(b, part);
                                buffers.Instances.UpdateState(ri->BufferIndex, state);
                            }
                        }
                    }
                }
            }
            request(RenderRequest::Reuse);
        }

        // Update bone pose state before WorldTransform consumes its Transform patches.
        const bool bones_need_refresh = rendering || pass == EventPass::Restore || anim_advanced || mode_changed || pose_state_created;
        if (bones_need_refresh || !reactive(r, Change::TransformDirty).empty() || !reactive(r, Change::TransformEnd).empty() || !reactive(r, Change::BonePose).empty()) {
            const auto &local_changes = reactive(r, Change::TransformDirty);
            const auto &transform_end = reactive(r, Change::TransformEnd);
            const auto &pose_changes = reactive(r, Change::BonePose);
            for (const auto [arm_obj_entity, arm_obj_comp] : r.view<const ArmatureObject>().each()) {
                auto *pose_state = r.try_edit<ArmaturePoseState>(arm_obj_comp.Entity);
                if (!pose_state) continue;
                const auto &armature = r.get<Armature>(arm_obj_comp.Entity);
                if (armature.Skins.empty()) continue;

                // Constraints can depend on external targets (e.g. physics bodies), so bone-dirty alone does not allow an early out.
                const bool has_any_constraint = std::any_of(
                    arm_obj_comp.BoneEntities.begin(), arm_obj_comp.BoneEntities.end(),
                    [&](auto e) { return r.all_of<BoneConstraints>(e); }
                );

                if (!bones_need_refresh && !has_any_constraint) {
                    bool has_dirty = false;
                    for (const auto b : arm_obj_comp.BoneEntities) {
                        if (local_changes.contains(b) || transform_end.contains(b) || pose_changes.contains(b)) {
                            has_dirty = true;
                            break;
                        }
                    }
                    if (!has_dirty) continue;
                }

                const mat4 armature_world_inv = has_any_constraint ? Inverse(ToMatrix(r.get<const WorldTransform>(arm_obj_entity))) : I4;

                bool need_sync = has_any_constraint || pose_state_created;
                bool rest_pose_edited = false;
                for (uint32_t i = 0; i < arm_obj_comp.BoneEntities.size(); ++i) {
                    const auto b = arm_obj_comp.BoneEntities[i];
                    if (!r.all_of<BoneDelta>(b)) continue;
                    const auto &rest = armature.Bones[i].RestLocal;
                    // Every pass composes the pose the same way so a restored pose matches a live one bit for bit.
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
                            // Start Edit mode from the rest pose.
                            local = {rest.P, rest.R, rest.S};
                            should_patch = need_sync = true;
                        } else if (transform_end.contains(b) || (local_changes.contains(b) && !r.all_of<StartTransform>(b))) {
                            // Commit an Edit-mode transform.
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
                    } else if (anim_advanced || mode_changed || pose_state_created || pose_changes.contains(b)) {
                        // Reconstruct entity position and rotation from the pose delta.
                        local = posed();
                        should_patch = need_sync = true;
                    } else if (local_changes.contains(b)) {
                        // Commit manual position or rotation changes into the pose delta.
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

                    // Apply pose constraints outside rest-pose editing.
                    if (!is_edit_mode) {
                        if (const auto *cs = r.try_get<const BoneConstraints>(b); cs && !cs->Stack.empty()) {
                            const auto before = local;
                            for (const auto &c : cs->Stack) {
                                if (c.TargetEntity == state::Null || !r.valid(c.TargetEntity)) continue;
                                const auto *twt = r.try_get<const WorldTransform>(c.TargetEntity);
                                if (twt) local = ApplyBoneConstraint(c, local, parent_pose_world, armature_world_inv, ToMatrix(*twt));
                            }
                            if (local.P != before.P || local.R != before.R) should_patch = true;
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
                        if (transform_end.contains(b) || (local_changes.contains(b) && !r.all_of<StartTransform>(b))) {
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
                    if (!is_edit_mode) {
                        for (uint32_t s = 0; s < pose_state->GpuDeformRanges.size(); ++s) {
                            ComputeDeformMatrices(armature, s, pose_state->BonePoseWorld, buffers.ArmatureDeformBuffer.GetMutable(pose_state->GpuDeformRanges[s]));
                        }
                        buffers.PreludeStale = true;
                    }
                    request(RenderRequest::Reuse);
                }
            }
        }
        // Recompute changed world transforms and their descendants.
        if (const auto &dirty = reactive(r, Change::TransformDirty); !dirty.empty()) {
            const bool bone_edit = is_edit_mode && FindArmatureObject(r, FindActiveEntity(r)) != state::Null;
            std::unordered_set<state::Entity> recompute;
            const auto collect = [&](this const auto &self, state::Entity e, bool propagate) -> void {
                if (!recompute.insert(e).second) return;
                if (propagate)
                    for (const auto child : Children{&r, e}) self(child, true);
            };
            for (const auto e : dirty) collect(e, !(bone_edit && r.all_of<StartTransform>(e)));

            std::unordered_set<state::Entity> done;
            const auto compute = [&](this const auto &self, state::Entity e) -> void {
                if (!done.insert(e).second) return;
                const auto *node = r.try_get<const SceneNode>(e);
                if (node && node->Parent != state::Null && (recompute.contains(node->Parent) || !r.all_of<WorldTransform>(node->Parent))) {
                    self(node->Parent); // Update the parent before reading its delta.
                }
                const Transform &t = *ComposedLocal(r, e);
                if (node && node->Parent != state::Null) r.emplace_or_replace<WorldTransform>(e, ToTransform(GetParentDelta(r, e) * ToMatrix(t)));
                else r.emplace_or_replace<WorldTransform>(e, t);
            };
            for (const auto e : recompute) compute(e);
        }
        {
            const auto &wt_reactive = reactive(r, Change::WorldTransform);
            std::vector<std::pair<uint32_t, WorldTransform>> wt_writes;
            wt_writes.reserve(wt_reactive.size() + sync.NewlyInserted.size());

            const auto collect_wt = [&](state::Entity e) {
                const auto *ri = r.try_get<const RenderInstance>(e);
                if (!ri || ri->BufferIndex == UINT32_MAX) return;

                const auto *wt = r.try_get<const WorldTransform>(e);
                if (!wt) return;

                auto display_wt = *wt;
                if (const auto *ds = r.try_get<BoneDisplayScale>(e)) display_wt.S = vec3{ds->Value};
                wt_writes.emplace_back(ri->BufferIndex, display_wt);
                if (const auto *joints = r.try_get<const BoneJointEntities>(e); joints && r.all_of<BoneDisplayScale>(e)) {
                    const float bone_length = r.get<BoneDisplayScale>(e).Value;
                    const float sphere_scale = bone_length * 0.06f;
                    if (joints->Head != state::Null) {
                        if (const auto *jri = r.try_get<const RenderInstance>(joints->Head)) {
                            wt_writes.emplace_back(jri->BufferIndex, Transform{wt->P, {1, 0, 0, 0}, vec3{sphere_scale}});
                        }
                    }
                    if (joints->Tail != state::Null) {
                        if (const auto *jri = r.try_get<const RenderInstance>(joints->Tail)) {
                            const vec3 tail_pos = wt->P + wt->R * vec3{0, bone_length, 0};
                            wt_writes.emplace_back(jri->BufferIndex, Transform{tail_pos, {1, 0, 0, 0}, vec3{sphere_scale}});
                        }
                    }
                }
            };
            for (auto e : wt_reactive) collect_wt(e);
            // Include newly visible entities absent from the reactive transform set.
            for (auto e : sync.NewlyInserted) {
                if (!wt_reactive.contains(e)) collect_wt(e);
            }
            if (FlushIndexedWrites(wt_writes, [&] { return buffers.Instances.GetMutableTransforms(); })) request(RenderRequest::Reuse);
        }
    }
    // Update an active scene camera before processing SceneView changes.
    if (const auto camera = LookThroughCameraEntity(r); camera != state::Null &&
        reactive(r, Change::WorldTransform).contains(camera)) {
        const auto &wt = r.get<WorldTransform>(camera);
        r.replace<ViewCamera>(viewport, ViewCamera{wt.P, wt.R, *LensOf(r, camera)});
    }
    {
        // Update transmission specialization before the UBO reads its pipeline state.
        const auto shading = r.get<const ViewportDisplay>(viewport).ViewportShading;
        if (recompiled || !reactive(r, Change::ViewportDisplay).empty() || !reactive(r, Change::PbrSpecialization).empty()) {
            // SubmitViewport refreshes all slots only on resize, so update this lazy sampler inline.
            const auto refresh_transmission_sampler = [&] {
                const auto info = targets.TransmissionSampler();
                slots.SetSampler({SlotType::Sampler, r.Context.get<const RenderSamplerSlots>().Transmission}, info.Texture, info.Sampler);
                request(RenderRequest::Rebuild);
            };
            if (shading == ViewportShadingMode::MaterialPreview || shading == ViewportShadingMode::Rendered) {
                PbrFeatureMask pbr_mask{0};
                const auto &active_lighting = GetActivePbrLighting(r, viewport, shading);
                if (active_lighting.UseSceneLights) pbr_mask |= PbrFeature::Punctual;
                for (const auto [_, feat] : r.view<const PbrMeshFeatures>().each()) pbr_mask |= feat.Mask;
                const bool non_triangle_topology = (buffers.MeshletTopologyMask & ~1u) != 0u;
                if (GetPipelines(r).Main.Compiler.CompilePipelines(pbr_mask, non_triangle_topology)) request(RenderRequest::Rebuild);
                const bool want_transmission = active_lighting.RealTransmission && HasFeature(pbr_mask, PbrFeature::Transmission);
                const auto te_px = RenderExtentPx(r);
                if (targets.EnsureTransmissionResources(ctx, std::bit_cast<mtl::Extent2D>(te_px), want_transmission)) refresh_transmission_sampler();
            } else if (targets.EnsureTransmissionResources(ctx, {}, false)) {
                refresh_transmission_sampler();
            }
        }
    }

    // Send pending pose deltas through the view UBO.
    if (!reactive(r, Change::TransformPending).empty() || !reactive(r, Change::TransformEnd).empty()) {
        if (is_edit_mode) r.Context.get<GpuSceneState>().EditPreludePending = true;
        else buffers.PreludeStale = true;
    }

    const auto render_extent = RenderExtentPx(r);
    if (buffers.FrameView != RenderView{r.get<const ViewCamera>(viewport), render_extent} ||
        !reactive(r, Change::SceneView).empty() ||
        !reactive(r, Change::TransformPending).empty() ||
        !reactive(r, Change::ViewportDisplay).empty() ||
        !reactive(r, Change::InteractionMode).empty() ||
        !reactive(r, Change::TransformEnd).empty() ||
        light_count_changed ||
        resized) {
        const float aspect = render_extent.x == 0 || render_extent.y == 0 ? 1.f : float(render_extent.x) / float(render_extent.y);
        // Update widened scene-camera FOV after viewport aspect-ratio changes.
        if (const auto camera = LookThroughCameraEntity(r); camera != state::Null) {
            r.edit<ViewCamera>(viewport).Data = WidenForLookThrough(*LensOf(r, camera), aspect);
        }
        const auto &camera = r.get<const ViewCamera>(viewport);
        const auto &settings = r.get<const ViewportDisplay>(viewport);
        const bool is_pbr_mode = settings.ViewportShading == ViewportShadingMode::MaterialPreview || settings.ViewportShading == ViewportShadingMode::Rendered;
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
            .CornerTangentSlot = mesh_slots.CornerTangent,
            .CornerColorSlot = mesh_slots.CornerColor,
            .CornerUvSlot = mesh_slots.CornerUv,
            .EdgeSharpnessSlot = mesh_slots.EdgeSharpness,
            .CornerClassSlot = mesh_slots.CornerClass,
            .CustomCornerMaskSlot = mesh_slots.CustomCornerMask,
            .CustomCornerNormalSlot = mesh_slots.CustomCornerNormal,
            .BaseSeamNormalSlot = mesh_slots.BaseSeamNormal,
            .BaseVertexNormalSlot = mesh_slots.BaseVertexNormal,
            .BaseFaceNormalSlot = mesh_slots.BaseFaceNormal,
            .FaceFirstTriangleSlot = mesh_slots.FaceFirstTriangle,
            .AdjacencySlot = mesh_slots.Adjacency,
            .BoneDeformSlot = mesh_slots.BoneDeform,
            .ArmatureDeformSlot = buffers.ArmatureDeformBuffer.Buffer.Slot,
            .MorphDeformSlot = mesh_slots.MorphTarget,
            .MorphWeightsSlot = buffers.MorphWeightBuffer.Buffer.Slot,
            .PosedPositionSlot = buffers.PosedPositions.Slot,
            .PosedVertexNormalSlot = buffers.PosedVertexNormals.Slot,
            .PosedSeamNormalSlot = buffers.PosedSeamNormals.Slot,
            .PosedFaceNormalSlot = buffers.PosedFaceNormals.Slot,
            .PosedMorphNormalDeltaSlot = buffers.PosedMorphNormalDeltas.Slot,
            .InstanceBoundsSlot = buffers.Instances.BoundsBuffer.Slot,
            .MaterialSlot = buffers.Materials.Slot,
            .PrimitiveMaterialSlot = mesh_slots.PrimitiveMaterial,
            .MeshRecordSlot = buffers.MeshRecords.Buffer.Slot,
            .InstanceRecordSlot = buffers.Instances.RecordBuffer.Slot,
            .ElementPrimitiveSlot = mesh_slots.ElementPrimitive,
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
        request(RenderRequest::Reuse);
    }

    // Publish dirty excite-mode vertex lists in one GPU selection transaction.
    if (interaction_mode == InteractionMode::Excite) {
        std::vector<std::pair<state::Entity, SlottedRange>> sound_selections;
        sound_selections.reserve(dirty_sound_selection_meshes.size());
        for (const auto mesh_entity : dirty_sound_selection_meshes) {
            SlottedRange sound_vertices{};
            for (const auto [entity, instance, excitable] : r.view<const Instance, const SoundVertices>().each()) {
                if (instance.Entity != mesh_entity) continue;
                sound_vertices = {excitable.Vertices, meshes.Slots().SoundVertex};
                break;
            }
            sound_selections.emplace_back(mesh_entity, sound_vertices);
        }
        ApplyEditSelectionLists(r, sound_selections, Element::Vertex);
        if (!dirty_sound_selection_meshes.empty()) {
            auto records = buffers.Instances.RecordBuffer.GetMutableSpan<InstanceRecord>(
                {0u, buffers.Instances.RecordBuffer.Count<InstanceRecord>()}
            );
            for (const auto [instance_entity, instance, render_instance] :
                 r.view<const Instance, const RenderInstance>().each()) {
                if (!dirty_sound_selection_meshes.contains(instance.Entity) ||
                    render_instance.BufferIndex >= records.size()) continue;
                auto &record = records[render_instance.BufferIndex];
                const auto *active = r.try_get<const MeshActiveElement>(instance.Entity);
                const auto *force = r.try_get<const VertexForce>(instance_entity);
                record.ActiveVertex = active ? active->Handle : InvalidOffset;
                record.ExcitedVertex = force ? force->Vertex : InvalidOffset;
            }
        }
    }
    if (!dirty_sound_selection_meshes.empty()) {
        request(RenderRequest::Reuse);
    }
    if (auto &state = r.Context.get<GpuSceneState>(); state.EditSelectionDirty) {
        for (auto &[_, work] : state.EditWork) work.CandidateReady = false;
        state.EditPreludePending = is_edit_mode;
        state.EditSelectionDirty = false;
        request(RenderRequest::Reuse);
    }
    if (!rendering) UpdateAudioContacts(r);
    r.ClearChanges();
    destroy_tracker.Storage.clear();
    r.clear<MeshGeometryDirty, MeshPositionsChanged, MeshMaterialAssignment>();
}

void RegisterSceneComponentHandlers(state::Scene &r) {
    r.on_destroy<MeshHandle, &ReleaseMeshEditWork>();
    reactive(r, Change::Selected).on<Selected>(On::Create | On::Destroy);
    reactive(r, Change::ActiveInstance).on<Active>(On::Create | On::Destroy);
    reactive(r, Change::BoneSelection).on<BoneSelection>(On::Create | On::Update | On::Destroy).on<BoneActive>(On::Create | On::Destroy);
    reactive(r, Change::Rerecord)
        .on<RenderInstance>(On::Create | On::Destroy)
        .on<Active>(On::Create | On::Destroy)
        .on<StartTransform>(On::Create | On::Destroy)
        .on<EditMode>(On::Create | On::Update);
    reactive(r, Change::MeshActiveElement).on<MeshActiveElement>(On::Create | On::Update);
    reactive(r, Change::MeshGeometry).on<MeshGeometryDirty>(On::Create).on<MeshPositionsChanged>(On::Create);
    // Refresh body-mesh reachability after collider or body changes.
    reactive(r, Change::PhysicsBodyMesh).on<PhysicsBodyHandle>(On::Create | On::Destroy).on<ColliderShape>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::MeshMaterial).on<MeshMaterialAssignment>(On::Create | On::Update);
    reactive(r, Change::SoundVertices).on<SoundVertices>(On::Create | On::Destroy);
    reactive(r, Change::SoundVerticesUpdated).on<SoundVertices>(On::Update);
    reactive(r, Change::VertexForce).on<VertexForce>(On::Create | On::Destroy);
    reactive(r, Change::TetMesh).on<TetBuffers>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::NewBufferEntity).on<MeshHandle>(On::Create).on<VertexStoreId>(On::Create).on<MeshPreview>(On::Create | On::Update);
    reactive(r, Change::MeshPreview).on<MeshPreview>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::RenderInstanceCreated).on<RenderInstance>(On::Create);
    reactive(r, Change::RenderInstanceDestroyed).on<RenderInstance>(On::Destroy);
    reactive(r, Change::ViewportDisplay).on<ViewportDisplay>(On::Create | On::Update);
    reactive(r, Change::InteractionMode).on<Interaction>(On::Create | On::Update);
    reactive(r, Change::WorkspaceLights).on<WorkspaceLights>(On::Create | On::Update);
    reactive(r, Change::ViewportTheme).on<ViewportTheme>(On::Create | On::Update);
    reactive(r, Change::MaterializedTextures).on<MaterializedTextures>(On::Create | On::Update);
    reactive(r, Change::StudioEnvironment).on<StudioEnvironment>(On::Create | On::Update);
    reactive(r, Change::SceneWorld).on<gltf::SourceAssets>(On::Create | On::Update);
    reactive(r, Change::PunctualLight).on<PunctualLight>(On::Create | On::Update).on<RenderInstance>(On::Create);
    reactive(r, Change::ActiveMaterialVariant).on<MaterialVariants>(On::Create | On::Update);
    reactive(r, Change::PbrSpecialization)
        .on<PbrMeshFeatures>(On::Create | On::Update | On::Destroy)
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
    reactive(r, Change::WorldTransform).on<WorldTransform>(On::Create | On::Update);
    reactive(r, Change::TransformPending).on<PendingTransform>(On::Create | On::Update | On::Destroy);
    reactive(r, Change::TransformEnd).on<StartTransform>(On::Destroy);
    reactive(r, Change::BonePose).on<BoneDelta>(On::Update);
    reactive(r, Change::TransformDirty)
        .on<Transform>(On::Create | On::Update)
        .on<PosedLocal>(On::Create | On::Update)
        .on<SceneNode>(On::Create | On::Update)
        .on<BoneDisplayScale>(On::Update);
    reactive(r, Change::AnimationEdited)
        .on<AnimationClips>(On::Create | On::Update | On::Destroy)
        .on<Animations>(On::Update);
    r.Context.emplace<EntityDestroyTracker>().Bind(r);

    // Mark local transforms after constraint edits to trigger world-transform recomputation.
    r.on_update<BoneConstraints, [](state::Scene &r, state::Entity e) {
        PatchEditedLocal(r, e, [](auto &) {});
    }>();
}
