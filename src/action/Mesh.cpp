#include "action/Mesh.h"
#include "Profile.h"
#include "SortUnique.h"
#include "action/Errors.h"
#include "action/InsetPreview.h"

#include "gpu/InsetPreviewPushConstants.h"
#include "gpu/InsetVertexBasis.h"

#include "TransformMath.h"
#include "mesh/BeautifyFaces.h"
#include "mesh/Decimate.h"
#include "mesh/EditVisibility.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshTopologyEdit.h"
#include "mesh/PositionOperations.h"
#include "mesh/RecalculateNormals.h"
#include "mesh/ScratchChunks.h"
#include "mesh/Unsubdivide.h"
#include "metal/Dispatch.h"
#include "object/ObjectOps.h"
#include "procedure/Procedure.h"
#include "project/Project.h"
#include "render/ElementWorkOps.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/MeshTopologyRepair.h"
#include "render/MeshletBuildGpu.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "state/Registry.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportRenderGpu.h"

#include <format>

namespace {
std::vector<uint32_t> EditorHiddenFaces(const MeshStore &meshes, uint32_t id) {
    std::vector<uint32_t> faces;
    meshes.GetHiddenElements(id, Element::Face).ForEach([&](uint32_t face) { faces.push_back(face); });
    return faces;
}

// Gather edited arena blocks before erasure releases empty blocks.
struct EditedBlocks {
    std::vector<uint32_t> Vertices, Faces, Corners;
};

void UpdatePoseMembership(state::Registry &r, const MeshTopologyEdit &edit, EditedBlocks edited) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(edit.StoreId);
    auto &[vertices, faces, corners] = edited;
    SortUnique(vertices);
    SortUnique(faces);
    SortUnique(corners);
    const auto update = [&](const auto &arena, ElementSetRef set, const auto &blocks, auto &...stores) {
        const auto members = arena.Blocks.Buffer.template GetSpan<MeshElementBlock>();
        const auto present = [&](uint32_t block) {
            return block < members.size() && members[block].Owner == set.Index && members[block].Count;
        };
        const auto revision = set ? arena.Set(set).Revision : 0u;
        (stores.UpdateBlocks(edit.StoreId, revision, blocks, present), ...);
    };
    update(a.Vertices, record.Vertices, vertices, buffers.VertexBounds, buffers.PosedPositions, buffers.PosedMorphNormalDeltas, buffers.PosedVertexNormals);
    update(a.FaceTriangles, record.FaceData, faces, buffers.PosedFaceNormals);
    // Posed sector normals key the corner blocks of a record with sector roots.
    const auto corner_members = a.FaceCorners.Blocks.Buffer.GetSpan<MeshElementBlock>();
    const auto has_sectors = [&](uint32_t block) {
        return record.NormalSectors.Count && block < corner_members.size() && corner_members[block].Owner == record.FaceCorners.Index && corner_members[block].Count;
    };
    buffers.PosedSectors.UpdateBlocks(edit.StoreId, meshes.GetDerived(edit.StoreId).NormalRevision, corners, has_sectors);
}

// Include newly created and removed faces when deciding whether triangle render data needs repair.
bool RepairsTriangleRender(const state::Registry &r, state::Entity entity, const MeshTopologyEdit &edit) {
    const auto *owner = TryRecordOf(r, entity);
    return owner && owner->StoreId == edit.SourceId && owner->RenderTopologies != 0u &&
        ((owner->RenderTopologies & 1u) != 0u || edit.AddedTriangleCount);
}

// Publish render, pose, and selection state.
// A missing render owner indicates a new canonical mesh.
void FinishTopologyEdit(state::Registry &r, state::Entity entity, const MeshTopologyTask &task, MeshTopologyEdit &edit, EditedBlocks edited) {
    const auto *render_owner = TryRecordOf(r, entity);
    const bool ready = render_owner && render_owner->StoreId == task.SourceId && render_owner->RenderTopologies != 0u;
    UpdatePoseMembership(r, edit, std::move(edited));
    if (ready && edit.InsetBasis.Count) {
        auto &session = project::Session(r);
        const auto output = edit.Chain.Scratch.Get(edit.Output->Vertices);
        std::vector<uint32_t> handles(output.begin(), output.end());
        SortUnique(handles);
        std::vector<Range> ranges;
        ForEachIndexRun(handles, [&](size_t first, size_t count) { ranges.push_back({handles[first], uint32_t(count)}); });
        session.InsetPreview->Entries.push_back({entity, task.SourceId, task.Op, task.Flags, edit.InsetBasis, std::move(ranges)});
        if (session.Previewing) {
            const auto &pipelines = GetMeshPipelines(r);
            (void)pipelines[MeshPass::InsetPreviewPositions].State();
            (void)pipelines[MeshPass::MeshletBoundsRefit].State();
        }
    }
    RefreshMeshBinding(r, task.SourceId);
    r.remove<PrimitiveShape, MeshActiveElement>(entity);
    r.emplace_or_replace<MeshGeometryDirty>(entity, EditSelectionAfter::Keep, ready);
    r.Context.get<GpuSceneState>().EditSelectionDirty = true;
}

// Apply all outputs of a topology action in one history transaction.
void RunTopologyAction(state::Registry &r, std::span<const state::Entity> mesh_entities, auto &&run) {
    auto &history = project::Session(r).History;
    auto before = history.Pin();
    try {
        for (const auto entity : mesh_entities) ReleaseMeshEditWork(r, entity);
        run();
    } catch (...) {
        (void)history.Restore(before);
        history.Release(before);
        throw;
    }
    history.Release(before);
}

// Batch render builds for copied outputs.
void CopiedOutputBuild(state::Registry &r, const MeshTopologyEdit &edit, std::vector<MeshletBuildSource> &sources) {
    auto &meshes = r.Context.get<MeshStore>();
    auto &record = meshes.WriteRecord(edit.StoreId);
    const Mesh mesh{meshes, edit.StoreId};
    if (mesh.FaceCount()) sources.push_back({.Destination = &record, .Topology = 0u, .ElementCount = edit.AddedTriangleCount, .Elements = edit.AddedTriangles});
    if (mesh.EdgeCount()) sources.push_back({.Destination = &record, .Topology = 1u, .ElementCount = mesh.EdgeCount()});
    sources.push_back({.Destination = &record, .Topology = 2u, .ElementCount = mesh.VertexCount()});
}

// Return an output record per task, or nullopt for an unchanged mesh.
// Repair render data before releasing reserved source identities.
std::vector<std::optional<uint32_t>> EditTopology(state::Registry &r, std::span<const state::Entity> mesh_entities, std::span<const MeshTopologyTask> tasks) {
    if (tasks.size() != mesh_entities.size()) throw std::invalid_argument("Topology task and entity counts differ.");
    auto &session = project::Session(r);
    auto &meshes = r.Context.get<MeshStore>();
    const bool insets = session.Previewing && std::ranges::any_of(tasks, [](const auto &task) {
                            return task.Op == MeshTopologyOp::InsetRegion || task.Op == MeshTopologyOp::InsetIndividual;
                        });
    if (insets && !session.InsetPreview) session.InsetPreview = std::make_unique<action::mesh::InsetPreviewCache>(meshes.BufferContext());
    mtl::ComputeChain chain{meshes.BufferContext(), TopologyScratchWords};
    auto edits = MeshTopologyEdit::Construct(r, chain, tasks, insets ? &session.InsetPreview->Basis : nullptr);
    MeshTopologyEdit::PublishAll(r, edits);
    std::vector<std::optional<uint32_t>> outputs(edits.size());
    std::vector<MeshletBuildSource> copies;
    std::vector<std::pair<state::Entity, const MeshTopologyEdit *>> repairs;
    std::vector<MeshTopologyEdit *> finished;
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        auto &edit = edits[i];
        if (!edit.Output) continue;
        outputs[i] = edit.StoreId;
        finished.push_back(&edit);
        if (edit.StoreId != edit.SourceId) CopiedOutputBuild(r, edit, copies);
        else if (RepairsTriangleRender(r, mesh_entities[i], edit)) repairs.emplace_back(mesh_entities[i], &edit);
    }
    RepairTopologyRender(r, chain, repairs, copies);
    // Resolve edited blocks before erasure.
    // Include surviving vertices and edges whose incidence changed.
    std::vector<ElementMeshletRepair> element_repairs;
    std::vector<EditedBlocks> edited(edits.size());
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        const auto &edit = edits[i];
        if (outputs[i] != edit.SourceId) continue;
        using Domain = MeshStore::ElementDomain;
        const auto &storage = edit.Chain.Scratch;
        const auto *render_owner = TryRecordOf(r, mesh_entities[i]);
        if (render_owner && render_owner->StoreId == edit.SourceId && render_owner->RenderTopologies && edit.Repair) {
            auto &points = element_repairs.emplace_back(ElementMeshletRepair{.StoreId = edit.SourceId, .Topology = 2u}).Elements;
            ForEachWorkElement(storage, edit.Repair->Elements[0], [&](uint32_t v) { points.push_back(v); });
            ForEachWorkElement(storage, edit.Output->Retired[0], [&](uint32_t v) { points.push_back(v); });
            auto &wires = element_repairs.emplace_back(ElementMeshletRepair{.StoreId = edit.SourceId, .Topology = 1u}).Elements;
            const auto edges = meshes.Arenas().HalfedgeEdges.Buffer.GetSpan<uint32_t>();
            ForEachWorkElement(storage, edit.RetiredEdges, [&](uint32_t e) { wires.push_back(e); });
            ForEachWorkElement(storage, edit.Repair->Elements[1], [&](uint32_t h) { wires.push_back(edges[meshes.HandleAddress(edit.SourceId, Domain::Halfedge, ElementHandle{h}).Value()]); });
        }
        auto &blocks = edited[i];
        blocks.Corners = edit.OldCornerBlocks;
        if (edit.Repair) {
            blocks.Vertices.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Vertex, storage, edit.Repair->Elements[0]));
            blocks.Faces.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Face, storage, edit.Repair->Elements[2]));
            if (meshes.Get(edit.StoreId).NormalSectors.Count) blocks.Corners.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Halfedge, storage, edit.Repair->Elements[1]));
        }
        blocks.Vertices.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Vertex, storage, edit.Output->Retired[0]));
        blocks.Faces.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Face, storage, edit.Output->Retired[1]));
    }
    MeshTopologyEdit::FinishAll(r, finished);
    std::vector<state::Entity> in_place;
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        if (outputs[i] != edits[i].SourceId) continue;
        FinishTopologyEdit(r, mesh_entities[i], tasks[i], edits[i], std::move(edited[i]));
        in_place.push_back(mesh_entities[i]);
    }
    RepairElementMeshlets(r, chain, element_repairs);
    chain.Submit();
    // Owner maps count only rendered elements, so topology changes need no mesh scan.
    for (const auto entity : in_place) {
        auto &owner = EditRecordOf(r, entity);
        if (!owner.RenderTopologies) continue;
        uint32_t mask = 0u;
        for (uint32_t t = 0u; t < 3u; ++t) {
            if (owner.ElementMeshletBlockCounts[t]) mask |= 1u << t;
        }
        owner.RenderTopologies = mask ? mask : 4u;
    }
    RefreshElementSelectionSummaries(r, in_place);
    return outputs;
}

// Run tasks in bounded batches to limit scratch memory.
void RunTasks(state::Registry &r, std::span<const state::Entity> mesh_entities, std::span<const MeshTopologyTask> tasks) {
    const profile::CpuScope scope{"TopologyAction"};
    if (tasks.empty()) return;
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto split = ChunkByScratch(uint32_t(tasks.size()), ScratchWordBudget, [&](uint32_t i) { return TopologyScratchBound(meshes, tasks[i]); });
    RunTopologyAction(r, mesh_entities, [&] {
        for (const auto span : split.Chunks) EditTopology(r, mesh_entities.subspan(span.Offset, span.Count), tasks.subspan(span.Offset, span.Count));
    });
}

void RunPerMesh(state::Registry &r, std::span<const state::Entity> mesh_entities, auto &&make) {
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<MeshTopologyTask> tasks;
    std::vector<state::Entity> entities;
    for (const auto e : mesh_entities) {
        if (std::optional<MeshTopologyTask> task = make(e, GetDrawnMesh(r, e))) {
            if (task->Op == MeshTopologyOp::AddPrimitives) task->Selection = EditorGeometrySelection(meshes, task->SourceId);
            PublishStoreSelection(meshes, *task);
            tasks.push_back(std::move(*task));
            entities.push_back(e);
        }
    }
    RunTasks(r, entities, tasks);
}

void RunOperator(state::Registry &r, std::span<const state::Entity> mesh_entities, MeshTopologyOp op, float param0 = 0.f, float param1 = 0.f, uint32_t flags = 0) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) { return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = op, .Param0 = param0, .Param1 = param1, .Flags = flags, .Selection = EditorGeometrySelection(r.Context.get<const MeshStore>(), mesh.GetStoreId())}; });
}

void SeparateSelected(state::Registry &r, std::span<const state::Entity> mesh_entities, const ::selection::PrimaryEditInstanceMap &primaries, action::mesh::SeparateMode mode) {
    std::vector<state::Entity> entities;
    std::vector<MeshTopologyTask> tasks;
    const auto &meshes = r.Context.get<const MeshStore>();
    for (const auto e : mesh_entities) {
        const auto id = *DrawnStoreId(r, e);
        const Mesh mesh{meshes, id};
        auto planned = SeparateGeometryTasks(meshes, mesh, EditorGeometrySelection(meshes, id), mode);
        for (auto &task : planned) {
            PublishStoreSelection(meshes, task);
            tasks.push_back(std::move(task));
            entities.push_back(e);
        }
    }
    if (tasks.empty()) return;
    RunTopologyAction(r, mesh_entities, [&] {
        const auto outputs = EditTopology(r, entities, tasks);
        std::vector<state::Entity> created;
        for (uint32_t i = 0u; i < tasks.size(); ++i) {
            if (tasks[i].Op != MeshTopologyOp::KeepSelectedFaces || !outputs[i]) continue;
            const auto primary = primaries.find(entities[i]);
            const auto instance = primary != primaries.end() ? primary->second : state::Null;
            MeshInstanceCreateInfo create{
                .Name = std::format("{}.001", instance != state::Null ? GetName(r, instance) : "Mesh"),
                .Transform = instance != state::Null ? *WorldTransformOf(r, instance) : Transform{},
                .Select = MeshInstanceCreateInfo::SelectBehavior::None,
            };
            created.push_back(::AddMesh(r, *outputs[i], std::move(create)).first);
        }
        if (created.empty()) return;
        RequestRender(r);
        for (const auto e : created) r.emplace_or_replace<ClusterLodDemand>(e);
        mtl::ComputeChain chain{r.Context.get<const MeshStore>().BufferContext()};
        UpdateAuthoredMorphShading(r, chain, created);
        chain.Submit();
    });
}

// Require exactly two selected objects without geometry procedures.
void CombineSelected(state::Registry &r, const action::mesh::Boolean &boolean) {
    const auto active = FindActiveEntity(r);
    std::vector<state::Entity> others;
    for (const auto e : r.view<const Selected>()) {
        if (e != active) others.push_back(e);
    }
    if (active == state::Null || others.size() != 1u || !r.all_of<Instance>(active) || !r.all_of<Instance>(others.front())) {
        return action::Fail(r, "A boolean combines the active object with one other selected object");
    }
    const auto first = r.get<const Instance>(active).Entity, second = r.get<const Instance>(others.front()).Entity;
    if (!HasMesh(r, first) || !HasMesh(r, second) || first == second) return action::Fail(r, "A boolean combines two different mesh objects");
    if (r.all_of<procedure::GeometryProcedure>(first) || r.all_of<procedure::GeometryProcedure>(second)) return action::Fail(r, "A boolean combines meshes without geometry procedures");
    const auto to_first = Inverse(ToMatrix(*WorldTransformOf(r, active))) * ToMatrix(*WorldTransformOf(r, others.front()));
    const auto rotation = ToMat3(to_first);
    if (!(Determinant(rotation) > 0.f)) return action::Fail(r, "A boolean cannot combine an object mirrored relative to the active one");
    const auto task = BooleanTask(GetMesh(r, first), GetMesh(r, second), boolean.Operation, rotation, vec3(to_first[3]));
    if (!task) return action::Fail(r, "A boolean combines closed surfaces");
    RunTopologyAction(r, std::array{first}, [&] {
        const auto output = ExecuteGeometryBoolean(r, *task);
        if (!output) return;
        // Erasing the handle queues the old record's release.
        r.remove<MeshHandle>(first);
        r.emplace<MeshHandle>(first, *output);
        r.remove<PrimitiveShape>(first);
        r.emplace_or_replace<MeshGeometryDirty>(first);
        RequestRender(r);
    });
}

std::optional<uint32_t> ActiveOrFirstSelectedEdge(const state::Registry &r, state::Entity mesh_entity, const Mesh &mesh) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selected = meshes.GetSelectedElements(mesh.GetStoreId(), Element::Edge);
    if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity);
        active && meshes.IsLiveElement(mesh.GetStoreId(), Element::Edge, ElementHandle{active->Handle}) && selected.Contains(active->Handle)) {
        return active->Handle;
    }
    if (const auto first = selected.First()) return *first;
    return {};
}

void LoopCutSelected(state::Registry &r, std::span<const state::Entity> mesh_entities, uint32_t cuts) {
    RunPerMesh(r, mesh_entities, [&](state::Entity e, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto edge = ActiveOrFirstSelectedEdge(r, e, mesh);
        return edge ? std::optional{LoopCutTask(mesh, *edge, cuts)} : std::nullopt;
    });
}

// Each stage reads the preceding stage's canonical output.
void RunTopologyStages(state::Registry &r, std::span<const state::Entity> mesh_entities, auto &&plan) {
    std::vector<std::vector<MeshTopologyTask>> plans;
    size_t stages = 0u;
    for (const auto entity : mesh_entities) {
        const auto mesh = GetDrawnMesh(r, entity);
        plans.push_back(plan(mesh));
        stages = std::max(stages, plans.back().size());
    }
    for (size_t stage = 0u; stage < stages; ++stage) {
        size_t at = 0u;
        RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &) -> std::optional<MeshTopologyTask> {
            auto &tasks = plans[at++];
            return stage < tasks.size() ? std::optional{std::move(tasks[stage])} : std::nullopt;
        });
    }
}

void KnifeSelected(state::Registry &r, std::span<const state::Entity> mesh_entities, const ::selection::PrimaryEditInstanceMap &primaries, vec2 start, vec2 end, const RenderView &view) {
    const float aspect = view.Extent.y > 0.f ? view.Extent.x / view.Extent.y : 1.f;
    const auto view_projection = view.Camera.Projection(aspect) * view.Camera.View();
    RunPerMesh(r, mesh_entities, [&](state::Entity e, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto primary = primaries.find(e);
        if (primary == primaries.end()) return {};
        auto task = KnifeTask(mesh, {}, view_projection * ToMatrix(*WorldTransformOf(r, primary->second)), view.Extent, start, end);
        task.Flags |= TopologyFlagSelectAll;
        return task;
    });
}

void EditSelectedFaceAttributes(state::Registry &r, std::span<const state::Entity> targets, bool colors, uint32_t uv_set, uint32_t operation) {
    auto &meshes = r.Context.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    std::vector<PositionOperationTarget> inputs;
    for (const auto entity : targets) {
        const auto id = *DrawnStoreId(r, entity);
        inputs.push_back({id, EditorGeometrySelection(meshes, id)});
    }
    const auto changes = EncodeFaceAttributeOperation(meshes, GetMeshPipelines(r), chain, inputs, colors, uv_set, operation);
    if (changes.empty()) return;
    uint64_t triangle_words = 0u;
    const auto &a = meshes.Arenas();
    for (const auto &change : changes) {
        const auto &record = meshes.Get(inputs[change.TargetIndex].StoreId);
        const auto bound = std::min(change.Faces.Incidence, a.Triangles.Set(record.TriangleData).BlockCount);
        const auto handles = a.Triangles.HandleBound(record.TriangleData);
        triangle_words += 1u + ElementWorkWords(handles, bound) + SortElementWorkWords(WorkCapacity(handles, bound));
    }
    chain.Scratch.ReserveAdditional(triangle_words);
    std::vector<std::pair<state::Entity, FaceTriangles>> repairs;
    repairs.reserve(changes.size());
    for (const auto &change : changes) {
        repairs.emplace_back(targets[change.TargetIndex], EncodeFaceTriangles(r, chain, inputs[change.TargetIndex].StoreId, change.Faces));
    }
    chain.Submit();
    for (auto &[entity, triangles] : repairs) triangles.Finish(chain);
    RepairFaceRender(r, chain, repairs);
    chain.Submit();
    RequestRender(r);
}

void EditSelectedPositions(state::Registry &r, std::span<const state::Entity> targets, PositionEditOp operation, float factor, uint32_t repeat, PositionOperationOptions options = {}) {
    const bool centered = operation == PositionEditOp::ToSphere || operation == PositionEditOp::PushPull || operation == PositionEditOp::Shear;
    const bool transformed = centered || options.Warp || options.Bend || options.Slide || options.EdgeSlide ||
        (operation == PositionEditOp::Copy && (options.Flags & PositionEditFlattenDirection));
    const auto &primaries = r.get<const EditPrimaries>(state::Document).Transformable;
    if (centered) options.Center = EditSelectionCenter(r);
    auto &meshes = r.Context.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    std::vector<PositionOperationTarget> inputs;
    std::vector<state::Entity> entities;
    for (const auto entity : targets) {
        if (transformed && !primaries.contains(entity)) continue;
        const auto id = *DrawnStoreId(r, entity);
        PositionOperationTarget input{id, EditorGeometrySelection(meshes, id)};
        if (transformed) input.World = *WorldTransformOf(r, primaries.at(entity));
        if (options.Curve || options.Circle) {
            const auto &a = meshes.Arenas();
            input.Excluded = GeometryMasks{{&a.VertexHidden}, {&a.EdgeHidden}, {&a.FaceHidden}};
        }
        if (const auto *active = r.try_get<const MeshActiveElement>(entity);
            active && r.get<const EditMode>(state::Document).Value == Element::Vertex) input.Reference = active->Handle;
        inputs.push_back(std::move(input));
        entities.push_back(entity);
    }
    const auto edited = EncodePositionOperations(meshes, GetMeshPipelines(r), chain, inputs, operation, factor, repeat, options);
    if (edited.empty()) return;
    std::vector<MeshVertexChanges> changes;
    std::vector<state::Entity> changed_entities;
    for (const auto &change : edited) {
        const auto entity = entities[change.TargetIndex];
        ReleaseMeshEditWork(r, entity);
        changes.push_back({entity, change.Ranges});
        changed_entities.push_back(entity);
    }
    RefreshEditedPositions(r, chain, changes);
    chain.Submit();
    for (const auto entity : PublishEditedPositions(r, chain, changed_entities)) {
        r.remove<PrimitiveShape>(entity);
        r.emplace_or_replace<MeshPositionsChanged>(entity);
    }
    chain.Submit();
}

} // namespace

namespace action::mesh {
std::vector<state::Entity> SelectedEditMeshes(const state::Registry &r) {
    std::vector<state::Entity> result;
    const auto element = r.get<const EditMode>(state::Document).Value;
    const auto &meshes = r.Context.get<const MeshStore>();
    for (const auto e : r.view<const MeshElementSelection>()) {
        if (!HasMesh(r, e)) continue;
        const auto &summary = meshes.GetSelectionSummary(*DrawnStoreId(r, e));
        if (summary.Mode == element && summary.SelectedCount > 0) result.push_back(e);
    }
    return result;
}

bool UpdateInsetPreview(state::Registry &r, const Inset &inset, InsetPreviewCache &cache) {
    const profile::CpuScope scope{"UpdateInsetPreview"};
    const auto targets = SelectedEditMeshes(r);
    if (targets.empty() || targets.size() != cache.Entries.size()) return false;
    const auto op = inset.Individual ? MeshTopologyOp::InsetIndividual : MeshTopologyOp::InsetRegion;
    const auto flags = inset.Even ? TopologyFlagEvenOffset : 0u;
    auto &meshes = r.Context.get<MeshStore>();
    // Retain the staged face selection for parameter updates.
    // Project::Record discards the cache for other actions.
    for (size_t i = 0u; i < targets.size(); ++i) {
        const auto &entry = cache.Entries[i];
        if (entry.Entity != targets[i] || entry.Op != op || entry.Flags != flags ||
            *DrawnStoreId(r, entry.Entity) != entry.StoreId || !entry.Basis.Count) return false;
    }
    {
        const profile::CpuScope capture{"InsetCaptureVertices"};
        const auto &arena = meshes.Arenas().Vertices;
        for (const auto &entry : cache.Entries) {
            const auto set = meshes.Get(entry.StoreId).Vertices;
            for (const auto run : entry.Ranges) arena.ForEachAddressRun(set, ElementHandle{run.Offset}, run.Count, [&](Range addresses) { arena.Buffer.CaptureWrite(uint64_t(addresses.Offset) * sizeof(Vertex), uint64_t(addresses.Count) * sizeof(Vertex)); });
        }
    }
    // Keep the position update and its dependent geometry work on one chain.
    mtl::ComputeChain chain{meshes.BufferContext()};
    {
        const profile::CpuScope positions{"InsetPositionPass"};
        const auto &pipeline = GetMeshPipelines(r)[MeshPass::InsetPreviewPositions];
        chain.Concurrent([&] {
            for (const auto &entry : cache.Entries) {
                const InsetPreviewPushConstants pc{
                    .Basis = {cache.Basis.Buffer.Slot, entry.Basis.Offset}, .VertexTable = meshes.Arenas().Vertices.TableRef(meshes.Get(entry.StoreId).Vertices), .VertexSlot = meshes.Slots().Vertices, .Count = uint32_t(uint64_t(entry.Basis.Count) * sizeof(uint32_t) / sizeof(InsetVertexBasis)), .Thickness = std::max(inset.Thickness, 0.f), .Depth = inset.Depth
                };
                chain.Groups(pipeline, pc, (pc.Count + 255u) / 256u);
            }
        });
    }
    std::vector<MeshVertexChanges> changed;
    changed.reserve(cache.Entries.size());
    for (const auto &entry : cache.Entries) changed.push_back({entry.Entity, entry.Ranges});
    RefreshEditedPositions(r, chain, changed);
    PublishEditedPositions(r, chain, targets, PositionPublication::Preview);
    chain.Submit();
    RequestRender(r);
    return true;
}

void Apply(state::Registry &r, const Action &action) {
    // Append operator nodes to procedure meshes and edit other meshes in place.
    // Hide and Reveal affect evaluated geometry; unsupported procedure operators fail.
    std::vector<state::Entity> targets, procedural;
    for (const auto e : SelectedEditMeshes(r)) (r.all_of<procedure::GeometryProcedure>(e) ? procedural : targets).push_back(e);
    const bool unformed = std::visit(
        [&]<typename A>(const A &a) {
            if constexpr (procedure::KindOf<A> < procedure::KindCount) {
                for (const auto e : procedural) {
                    const auto node = nodes::AppendNode(r, e, procedure::KindOf<A>);
                    if (node == state::Null) break;
                    r.replace<A>(node, a);
                    r.remove<MeshActiveElement>(e);
                }
                return false;
            } else {
                return !std::same_as<A, Hide> && !std::same_as<A, Reveal> && std::ranges::any_of(r.get<const EditPrimaries>(state::Document).All, [&](const auto &entry) { return r.all_of<procedure::GeometryProcedure>(entry.first); });
            }
        },
        action
    );
    if (unformed) return Fail(r, "The operator has no node form for a mesh with a geometry procedure");
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selected = [&](auto &&make) {
        RunPerMesh(r, targets, [&](state::Entity, const Mesh &mesh) { return make(mesh, EditorGeometrySelection(meshes, mesh.GetStoreId())); });
    };
    const auto visibility = [&](EditVisibilityOperation operation) {
        std::vector<uint32_t> ids;
        for (const auto &[entity, instance] : r.get<const EditPrimaries>(state::Document).All) {
            if (HasMesh(r, entity)) ids.push_back(*DrawnStoreId(r, entity));
        }
        if (EditVisibility(r, ids, r.get<const EditMode>(state::Document).Value, operation)) {
            r.Context.get<GpuSceneState>().EditSelectionDirty = true;
            for (const auto &[entity, instance] : r.get<const EditPrimaries>(state::Document).All) r.remove<MeshActiveElement>(entity);
        }
    };
    const auto latch_translate = [&] { r.emplace_or_replace<StartScreenTransform>(state::Document, TransformGizmo::TransformType::Translate); };
    std::visit(
        overloaded{
            [&](const Hide &a) { visibility(a.Unselected ? EditVisibilityOperation::HideUnselected : EditVisibilityOperation::HideSelected); },
            [&](const Reveal &a) { visibility(a.Select ? EditVisibilityOperation::RevealSelected : EditVisibilityOperation::Reveal); },
            [&](const RotateUVs &a) { EditSelectedFaceAttributes(r, targets, false, a.UVSet, a.CounterClockwise ? 1u : 0u); },
            [&](const ReverseUVs &a) { EditSelectedFaceAttributes(r, targets, false, a.UVSet, 2u); },
            [&](const RotateColors &a) { EditSelectedFaceAttributes(r, targets, true, 0u, a.CounterClockwise ? 1u : 0u); },
            [&](ReverseColors) { EditSelectedFaceAttributes(r, targets, true, 0u, 2u); },
            [&](const Delete &a) {
                RunOperator(r, targets, MeshTopologyOp(uint32_t(a.Mode)));
                if (a.Mode == DeleteMode::Loose) {
                    std::vector<std::pair<state::Entity, std::span<const uint32_t>>> empty;
                    empty.reserve(targets.size());
                    for (const auto entity : targets) empty.emplace_back(entity, std::span<const uint32_t>{});
                    ApplyEditSelectionLists(r, empty, r.get<const EditMode>(state::Document).Value);
                }
            },
            [&](const Merge &a) {
                selected([&](const Mesh &mesh, const GeometrySelection &selection) {
                    const auto &vertices = selection.Vertices;
                    const auto reference = vertices.empty() ? InvalidOffset : a.Mode == MergeMode::Last ? vertices.back() :
                                                                                                          vertices.front();
                    return MergeTask(meshes, mesh, selection, a.Mode, std::max(a.Distance, 0.f), reference);
                });
            },
            [&](const Extrude &a) {
                using Mode = ExtrudeMode;
                const auto op = a.Mode == Mode::Vertices ? MeshTopologyOp::ExtrudeVertices : a.Mode == Mode::Edges ? MeshTopologyOp::ExtrudeEdges :
                    a.Mode == Mode::FacesIndividual                                                                ? MeshTopologyOp::ExtrudeFacesIndividual :
                                                                                                                     MeshTopologyOp::ExtrudeRegion;
                RunOperator(r, targets, op);
                latch_translate();
            },
            [&](Duplicate) {
                RunOperator(r, targets, MeshTopologyOp::DuplicateGeometry);
                latch_translate();
            },
            [&](Split) { RunOperator(r, targets, MeshTopologyOp::SplitGeometry); },
            [&](const Separate &a) {
                const auto &primaries = r.get<const EditPrimaries>(state::Document).All;
                std::vector<state::Entity> sources;
                for (const auto &[entity, instance] : primaries) {
                    if (HasMesh(r, entity)) sources.push_back(entity);
                }
                SeparateSelected(r, sources, primaries, a.Mode);
            },
            [&](const Subdivide &a) { RunOperator(r, targets, MeshTopologyOp::Subdivide, float(std::max(a.Cuts, 1u))); },
            [&](const SnapSymmetry &a) {
                if (uint32_t(a.Axis) > 2u) return;
                EditSelectedPositions(r, targets, PositionEditOp::SnapSymmetry, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = 1u << uint32_t(a.Axis), .Flags = a.Negative ? PositionEditSymmetryNegative : 0u, .SnapSymmetry = PositionSymmetry{a.Threshold, a.Center}});
            },
            [&](const Decimate &a) { selected([&](const Mesh &mesh, const GeometrySelection &selection) { return DecimateTask(meshes, mesh, selection, a.Ratio, EditorHiddenFaces(meshes, mesh.GetStoreId())); }); },
            [&](const Unsubdivide &a) {
                for (uint32_t i = 0u; i < std::clamp(a.Iterations, 1u, 1000u); ++i) {
                    bool changed = false;
                    selected([&](const Mesh &mesh, const GeometrySelection &selection) {
                        auto task = UnsubdivideTask(mesh, selection);
                        changed |= task.has_value();
                        return task;
                    });
                    if (!changed) break;
                }
            },
            [&](Triangulate) { RunOperator(r, targets, MeshTopologyOp::Triangulate); },
            [&](const BeautifyFaces &a) {
                selected([&](const Mesh &mesh, const GeometrySelection &selection) {
                    return BeautifyFaceTask(mesh, selection, a.Method == BeautifyMethod::Angle);
                });
            },
            [&](TrisToQuads) { RunOperator(r, targets, MeshTopologyOp::TrisToQuads); },
            [&](const SmoothVertices &a) {
                EditSelectedPositions(r, targets, PositionEditOp::Smooth, a.Factor, std::clamp(a.Repeat, 1u, 1000u), {.Axes = uint32_t(a.X) | (uint32_t(a.Y) << 1u) | (uint32_t(a.Z) << 2u)});
            },
            [&](const SpaceEvenly &a) {
                EditSelectedPositions(r, targets, PositionEditOp::SpaceEvenly, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = uint32_t(a.X) | (uint32_t(a.Y) << 1u) | (uint32_t(a.Z) << 2u), .Flags = a.Interpolation == EdgeLoopInterpolation::Cubic ? PositionEditCurveCubic : 0u});
            },
            [&](const RelaxEdgeLoops &a) {
                if (!a.Iterations) return;
                EditSelectedPositions(r, targets, PositionEditOp::RelaxEdgeLoops, 1.f, std::min(a.Iterations, 1000u), {.Flags = (a.Interpolation == EdgeLoopInterpolation::Cubic ? PositionEditCurveCubic : 0u) | (a.EvenSpacing ? PositionEditRelaxEven : 0u)});
            },
            [&](const ToSphere &a) { EditSelectedPositions(r, targets, PositionEditOp::ToSphere, a.Factor, 1u); },
            [&](const PushPull &a) { EditSelectedPositions(r, targets, PositionEditOp::PushPull, a.Distance, 1u); },
            [&](const Shear &a) {
                const auto axis = uint32_t(a.Axis), ortho = uint32_t(a.AxisOrtho);
                if (axis > 2u || ortho > 2u || axis == ortho) return;
                vec3 normal{}, direction{};
                normal[axis] = 1.f;
                direction[ortho] = 1.f;
                if (a.Local) {
                    const auto &primaries = r.get<const EditPrimaries>(state::Document).Transformable;
                    const auto active = primaries.find(GetActiveMeshEntity(r));
                    if (active == primaries.end()) return;
                    const auto &world = *WorldTransformOf(r, active->second);
                    normal = world.R * normal * (world.S[axis] < 0.f ? -1.f : 1.f);
                    direction = world.R * direction * (world.S[ortho] < 0.f ? -1.f : 1.f);
                }
                EditSelectedPositions(r, targets, PositionEditOp::Shear, std::tan(a.Angle), 1u, {.Direction = direction, .Gradient = Cross(direction, normal)});
            },
            [&](const Warp &a) {
                if (!a.Orientation) return;
                EditSelectedPositions(r, targets, PositionEditOp::Warp, a.Angle, 1u, {.Warp = PositionWarp{*a.Orientation, a.Center, a.OffsetAngle, a.Min, a.Max, a.AutoRange}});
            },
            [&](const Bend &a) {
                if (!a.Orientation) return;
                EditSelectedPositions(r, targets, PositionEditOp::Bend, a.Angle, 1u, {.Flags = a.Clamp ? PositionEditBendClamp : 0u, .Bend = PositionBend{*a.Orientation, a.Center, a.OffsetAngle, a.Radius}});
            },
            [&](const Randomize &a) {
                EditSelectedPositions(r, targets, PositionEditOp::Randomize, a.Amount, 1u, {.Randomize = PositionRandomize{a.Uniform, a.Normal, a.Seed}});
            },
            [&](const VertexSlide &a) {
                EditSelectedPositions(r, targets, PositionEditOp::VertexSlide, a.Clamp ? std::clamp(a.Factor, 0.f, 1.f) : a.Factor, 1u, {.Flags = a.Clamp ? 0u : PositionEditSlideUnclamped, .Slide = PositionSlide{a.Direction, a.Even, a.Flipped}});
            },
            [&](const EdgeSlide &a) {
                EditSelectedPositions(r, targets, PositionEditOp::EdgeSlide, a.Clamp ? std::clamp(a.Factor, -1.f, 1.f) : a.Factor, 1u, {.Flags = a.Clamp ? 0u : PositionEditSlideUnclamped, .EdgeSlide = PositionSlide{a.Direction, a.Even, a.Flipped}});
            },
            [&](const Circularize &a) {
                EditSelectedPositions(r, targets, PositionEditOp::Copy, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = uint32_t(a.X) | (uint32_t(a.Y) << 1u) | (uint32_t(a.Z) << 2u), .Flags = (a.Regular ? PositionEditCurveRegular : 0u) | (a.Method == CircleFit::Contract ? PositionEditCircleContract : 0u), .Circle = PositionCircle{a.Radius, a.Angle}});
            },
            [&](const CurveBetweenSelected &a) {
                EditSelectedPositions(r, targets, PositionEditOp::Copy, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = uint32_t(a.X) | (uint32_t(a.Y) << 1u) | (uint32_t(a.Z) << 2u), .Flags = (a.Interpolation == EdgeLoopInterpolation::Cubic ? PositionEditCurveCubic : 0u) | (a.Regular ? PositionEditCurveRegular : 0u) | (a.Elevation == CurveElevation::Raise ? PositionEditCurveRaise : a.Elevation == CurveElevation::Lower ? PositionEditCurveLower :
                                                                                                                                                                                                                                                                                                                                                                                                                                     0u),
                                                                                                             .Curve = PositionCurve{a.Extend}});
            },
            [&](const MakePlanarFaces &a) { EditSelectedPositions(r, targets, PositionEditOp::Planar, a.Factor, std::clamp(a.Repeat, 1u, 10000u)); },
            [&](const Flatten &a) {
                EditSelectedPositions(r, targets, PositionEditOp::Copy, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = uint32_t(a.X) | (uint32_t(a.Y) << 1u) | (uint32_t(a.Z) << 2u), .Flags = a.Method == FlattenMethod::View ? PositionEditFlattenDirection : a.Method == FlattenMethod::FaceNormals ? PositionEditFlattenNormals :
                                                                                                                                                                                                                                                                                                       0u,
                                                                                                             .Direction = a.ViewNormal});
            },
            [&](const ShrinkFatten &a) {
                EditSelectedPositions(r, targets, PositionEditOp::ShrinkFatten, a.Distance, 1u, {.Flags = (a.Even ? PositionEditEvenOffset : 0u) | (r.get<const EditMode>(state::Document).Value == Element::Face ? PositionEditSelectedFaceNormals : 0u)});
            },
            [&](const SplitNonplanarFaces &a) { RunOperator(r, targets, MeshTopologyOp::SplitNonplanarFaces, std::clamp(a.Angle, 0.f, std::numbers::pi_v<float>)); },
            [&](SplitConcaveFaces) { RunOperator(r, targets, MeshTopologyOp::SplitConcaveFaces); },
            [&](const Poke &a) { RunOperator(r, targets, MeshTopologyOp::Poke, a.Offset); },
            [&](FlipNormals) { RunOperator(r, targets, MeshTopologyOp::FlipNormals); },
            [&](const RecalculateNormals &a) {
                std::vector<uint32_t> ids;
                ids.reserve(targets.size());
                for (const auto entity : targets) ids.push_back(*DrawnStoreId(r, entity));
                std::vector<GeometrySelection> selections;
                selections.reserve(ids.size());
                for (const auto id : ids) selections.push_back(EditorGeometrySelection(r.Context.get<const MeshStore>(), id));
                auto faces = RecalculateFaceFlips(r.Context.get<MeshStore>(), GetMeshPipelines(r), ids, selections, a.Inside);
                uint32_t i = 0u;
                RunPerMesh(r, targets, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
                    auto &list = faces[i++];
                    if (list.empty()) return {};
                    list.insert(list.begin(), uint32_t(list.size()));
                    return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::FlipNormals, .List = std::move(list), .Selection = EditorGeometrySelection(r.Context.get<const MeshStore>(), mesh.GetStoreId())};
                });
            },
            [&](EdgeSplit) { RunOperator(r, targets, MeshTopologyOp::EdgeSplit); },
            [&](const Inset &a) { RunOperator(r, targets, a.Individual ? MeshTopologyOp::InsetIndividual : MeshTopologyOp::InsetRegion, std::max(a.Thickness, 0.f), a.Depth, a.Even ? 1u : 0u); },
            [&](Fill) { selected([&](const Mesh &mesh, const GeometrySelection &selection) { return FillTask(meshes, mesh, selection); }); },
            [&](const LoopCut &a) { LoopCutSelected(r, targets, a.Cuts); },
            [&](const Spin &a) {
                if (Dot(a.Axis, a.Axis) <= 0.f) return;
                const auto axis = Normalize(a.Axis);
                selected([&](const Mesh &mesh, const GeometrySelection &selection) { return ExtrudeStepsTask(mesh, selection, Element::None, a.Steps, ToMat3(AngleAxis(a.Angle / float(std::max(a.Steps, 1u)), axis)), axis * (a.Offset / float(std::max(a.Steps, 1u))), a.Center); });
            },
            [&](const ExtrudeRepeat &a) { selected([&](const Mesh &mesh, const GeometrySelection &selection) { return ExtrudeStepsTask(mesh, selection, Element::None, a.Steps, mat3{1.f}, a.Offset, vec3{0.f}); }); },
            [&](const Bisect &a) {
                if (Dot(a.Normal, a.Normal) <= 0.f) return;
                RunTopologyStages(r, targets, [&](const Mesh &mesh) { return BisectTasks(mesh, a.Point, a.Normal, a.ClearInner, a.ClearOuter); });
            },
            [&](const Symmetrize &a) { RunTopologyStages(r, targets, [&](const Mesh &mesh) { return SymmetrizeTasks(mesh, uint8_t(a.Axis), a.Negative); }); },
            [&](const Solidify &a) { RunOperator(r, targets, MeshTopologyOp::Solidify, a.Thickness); },
            [&](const Wireframe &a) {
                RunOperator(r, targets, MeshTopologyOp::Wireframe, std::max(a.Thickness, 0.f), std::clamp(a.Offset, -1.f, 1.f), (a.Even ? TopologyFlagEvenOffset : 0u) | (a.Boundary ? TopologyFlagWireBoundary : 0u) | (a.Relative ? TopologyFlagWireRelative : 0u) | (a.Replace ? TopologyFlagWireReplace : 0u));
            },
            [&](ConnectVertices) { RunOperator(r, targets, MeshTopologyOp::ConnectVertices); },
            // A boolean combines whole objects rather than an edit selection.
            [&](const Boolean &a) { CombineSelected(r, a); },
            [&](const Knife &a) { KnifeSelected(r, targets, r.get<const EditPrimaries>(state::Document).All, a.Start, a.End, *a.View); },
            [&](BridgeEdgeLoops) { selected([&](const Mesh &mesh, const GeometrySelection &selection) { return BridgeEdgeLoopsTask(meshes, mesh, selection); }); },
            [&](const GridFill &a) { selected([&](const Mesh &mesh, const GeometrySelection &selection) { return GridFillTask(meshes, mesh, selection, a.Span); }); },
            [&](const FillHoles &a) { RunPerMesh(r, targets, [&](state::Entity, const Mesh &mesh) { return FillHolesTask(meshes, mesh, a.Sides); }); },
            [&](ConvexHull) { selected([&](const Mesh &mesh, const GeometrySelection &selection) { return ConvexHullTask(meshes, mesh, selection); }); },
            [&](EdgeRotate) { selected([&](const Mesh &mesh, const GeometrySelection &selection) { return RotateEdgesTask(mesh, selection); }); },
            [&](const Bevel &a) {
                selected([&](const Mesh &mesh, const GeometrySelection &selection) {
                    return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = a.Vertices ? MeshTopologyOp::BevelVertices : MeshTopologyOp::BevelEdges, .Param0 = std::max(a.Width, 0.f), .Steps = std::max(a.Segments, 1u), .Selection = selection};
                });
            },
            [&](Rip) {
                RunOperator(r, targets, MeshTopologyOp::EdgeSplit, 0.f, 0.f, TopologyFlagRipSelectCopies);
                latch_translate();
            },
            [&](const Dissolve &a) {
                using Mode = DissolveMode;
                switch (a.Mode) {
                    case Mode::Vertices: return RunOperator(r, targets, MeshTopologyOp::DissolveVertices);
                    case Mode::Edges: return RunOperator(r, targets, MeshTopologyOp::DissolveEdges, 0.f, 0.f, a.KeepVertices ? TopologyFlagKeepVertices : 0u);
                    case Mode::Faces: return RunOperator(r, targets, MeshTopologyOp::DissolveFaces);
                    case Mode::Limited: return RunOperator(r, targets, MeshTopologyOp::DissolveLimited, std::clamp(a.Angle, 0.f, .5f * std::numbers::pi_v<float>), 0.f, (r.get<const EditMode>(state::Document).Value == Element::Face ? TopologyFlagFaceSelection : 0u) | (a.AllBoundaries ? TopologyFlagAllBoundaries : 0u) | (a.DelimitMaterials ? TopologyFlagDelimitMaterial : 0u) | (a.DelimitSharpEdges ? TopologyFlagDelimitSharp : 0u) | (a.DelimitUVs ? TopologyFlagDelimitUV : 0u));
                    case Mode::Degenerate: return RunOperator(r, targets, MeshTopologyOp::DissolveDegenerate, std::max(a.Distance, 0.f));
                }
            },
        },
        action
    );
}
} // namespace action::mesh
