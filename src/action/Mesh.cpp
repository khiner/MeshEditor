#include "action/Mesh.h"
#include "ProcessEvents.h"
#include "Profile.h"
#include "SortUnique.h"
#include "action/InsetPreview.h"

#include "gpu/FaceAttributeEditPushConstants.h"
#include "gpu/InsetPreviewPushConstants.h"
#include "gpu/InsetVertexBasis.h"
#include "gpu/MeshTopologyOp.h"
#include "gpu/Vertex.h"
#include "gpu/VertexPositionEditPushConstants.h"

#include "TransformMath.h"
#include "Variant.h"
#include "mesh/BeautifyFaces.h"
#include "mesh/Decimate.h"
#include "mesh/EditVisibility.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopology.h"
#include "mesh/MeshTopologyEdit.h"
#include "mesh/PositionOperations.h"
#include "mesh/PrimitiveType.h"
#include "mesh/RecalculateNormals.h"
#include "mesh/ScratchChunks.h"
#include "mesh/TopologyOperations.h"
#include "mesh/Unsubdivide.h"
#include "metal/Dispatch.h"
#include "numeric/MatrixMath.h"
#include "numeric/QuaternionMath.h"
#include "object/ObjectOps.h"
#include "project/Project.h"
#include "render/ElementWorkOps.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/MeshTopologyRepair.h"
#include "render/MeshletBuildGpu.h"
#include "render/SceneUpdates.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "state/Scene.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportRenderGpu.h"

#include <format>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <optional>
#include <stdexcept>

namespace {
std::vector<uint32_t> EditorHiddenFaces(const MeshStore &meshes, uint32_t id) {
    std::vector<uint32_t> faces;
    meshes.GetHiddenElements(id, Element::Face).ForEach([&](uint32_t face) { faces.push_back(face); });
    return faces;
}

void UpdatePoseMembership(state::Scene &r, const MeshTopologyEdit &edit) {
    auto &buffers = r.Context.get<GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(edit.StoreId);
    std::vector<uint32_t> vertices, faces, normal_payloads = edit.OldNormalPayloadBlocks;
    const auto &storage = edit.Chain.Scratch;
    const auto gather = [&](std::vector<uint32_t> &blocks, ElementWork work) {
        ForEachWorkBlock(storage, work, [&](uint32_t block, auto) { blocks.push_back(block); });
    };
    if (edit.Repair) {
        gather(vertices, edit.Repair->Elements[0]);
        gather(faces, edit.Repair->Elements[2]);
        ForEachWorkBlock(storage, edit.Repair->Elements[1], [&](uint32_t block, auto) {
            const auto payload = a.NormalSectors.PayloadBlock(block);
            if (payload) normal_payloads.push_back(payload - 1u);
        });
    }
    if (edit.Output) {
        gather(vertices, edit.Output->Retired[0]);
        gather(faces, edit.Output->Retired[1]);
    }
    SortUnique(vertices);
    SortUnique(faces);
    SortUnique(normal_payloads);
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
    const auto normal_owners = a.NormalSectors.Owners.Buffer.GetSpan<uint32_t>();
    const auto has_normal = [&](uint32_t payload) {
        if (payload >= normal_owners.size() || !normal_owners[payload]) return false;
        const auto block = normal_owners[payload] - 1u;
        return a.FaceCorners.Blocks.Get({block, 1u})[0].Owner == record.FaceCorners.Index &&
            a.NormalSectors.PayloadBlock(block) == payload + 1u;
    };
    buffers.PosedSectors.UpdateBlocks(edit.StoreId, meshes.GetDerived(edit.StoreId).NormalRevision, normal_payloads, has_normal);
}

// The edit-mode meshes with a selection in the viewport's edit element domain.
std::vector<state::Entity> SelectedEditMeshes(const state::Scene &r, state::Entity viewport) {
    std::vector<state::Entity> result;
    const auto element = r.get<const EditMode>(viewport).Value;
    const auto &meshes = r.Context.get<const MeshStore>();
    for (const auto e : r.view<const MeshElementSelection>()) {
        if (!HasMesh(r, e)) continue;
        const auto &summary = meshes.GetSelectionSummary(r.get<const MeshHandle>(e).StoreId);
        if (summary.Mode == element && summary.SelectedCount > 0) result.push_back(e);
    }
    return result;
}

// A published owner repairs its old triangles and any new faces, including its first or last faces.
bool RepairsTriangleRender(const state::Scene &r, state::Entity entity, const MeshTopologyEdit &edit) {
    const auto *owner = TryRecordOf(r, entity);
    return owner && owner->StoreId == edit.SourceId && owner->RenderTopologies != 0u &&
        ((owner->RenderTopologies & 1u) != 0u || edit.AddedTriangleCount);
}

// Publishes a finished in-place edit's render, pose and selection summary state.
// A record without a render owner is a newly created canonical mesh.
// Affected vertices and edges join the point and wire repairs, including elements
// that gain or lose incidence while remaining live.
void FinishTopologyEdit(state::Scene &r, state::Entity entity, const MeshTopologyTask &task, MeshTopologyEdit &edit, std::vector<ElementMeshletRepair> &element_repairs) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto *render_owner = TryRecordOf(r, entity);
    const bool ready = render_owner && render_owner->StoreId == task.SourceId && render_owner->RenderTopologies != 0u;
    UpdatePoseMembership(r, edit);
    if (ready && edit.Repair) {
        const auto &storage = edit.Chain.Scratch;
        auto &points = element_repairs.emplace_back(ElementMeshletRepair{.StoreId = task.SourceId, .Topology = 2u}).Elements;
        ForEachWorkElement(storage, edit.Repair->Elements[0], [&](uint32_t v) { points.push_back(v); });
        ForEachWorkElement(storage, edit.Output->Retired[0], [&](uint32_t v) { points.push_back(v); });
        auto &wires = element_repairs.emplace_back(ElementMeshletRepair{.StoreId = task.SourceId, .Topology = 1u}).Elements;
        const auto edges = meshes.Arenas().HalfedgeEdges.Buffer.GetSpan<uint32_t>();
        ForEachWorkElement(storage, edit.RetiredEdges, [&](uint32_t e) { wires.push_back(e); });
        ForEachWorkElement(storage, edit.Repair->Elements[1], [&](uint32_t h) { wires.push_back(edges[h]); });
    }
    if (ready && edit.InsetBasis.Count) {
        // The staged edit captured its basis into the session's preview cache.
        auto &session = project::Session(r);
        const auto output = edit.Chain.Scratch.Get(edit.Output->Vertices);
        std::vector<uint32_t> handles(output.begin(), output.end());
        SortUnique(handles);
        std::vector<Range> ranges;
        ForEachIndexRun(handles, [&](size_t first, size_t count) { ranges.push_back({handles[first], uint32_t(count)}); });
        session.InsetPreview->Entries.push_back({entity, task.SourceId, task.Op, task.Flags, edit.InsetBasis, std::move(ranges)});
        if (session.Previewing) {
            auto &pipelines = GetMeshPipelines(r);
            (void)pipelines[MeshPass::InsetPreviewPositions].State();
            (void)pipelines[MeshPass::MeshletBoundsRefit].State();
        }
    }
    RefreshMeshBinding(r, task.SourceId);
    r.remove<PrimitiveShape, MeshActiveElement>(entity);
    r.emplace_or_replace<MeshGeometryDirty>(entity, EditSelectionAfter::Keep, ready);
    r.Context.get<GpuSceneState>().EditSelectionDirty = true;
}

// Every topology action prepares selection and edit work inside one history
// transaction, including actions that publish more than one mesh output.
void RunTopologyAction(state::Scene &r, std::span<const state::Entity> mesh_entities, auto &&run) {
    auto &history = project::Session(r).History;
    auto before = history.Pin();
    bool changed = false;
    try {
        for (const auto entity : mesh_entities) ReleaseMeshEditWork(r, entity);
        changed = run();
    } catch (...) {
        history.Restore(before);
        history.Release(before);
        throw;
    }
    history.Release(before);
    if (changed) r.Context.get<GpuBuffers>().PreludeStale = true;
}

// A copied output builds all its drawable domains through the same render batch.
// Its canonical geometry is new, so gathering it visits only the copied elements.
void CopiedOutputBuild(state::Scene &r, const MeshTopologyEdit &edit, std::vector<MeshletBuildSource> &sources) {
    auto &meshes = r.Context.get<MeshStore>();
    auto &record = meshes.WriteRecord(edit.StoreId);
    const Mesh mesh{meshes, edit.StoreId};
    if (mesh.FaceCount()) sources.push_back({.Destination = &record, .Topology = 0u, .ElementCount = edit.AddedTriangleCount, .Elements = edit.AddedTriangles});
    if (mesh.EdgeCount()) sources.push_back({.Destination = &record, .Topology = 1u, .ElementCount = mesh.EdgeCount()});
    sources.push_back({.Destination = &record, .Topology = 2u, .ElementCount = mesh.VertexCount()});
}

// Every mesh's edit shares the action's chain, construction's submits, each publication submit, one render repair and one selection update.
// Render repairs read the reserved source identities, so every repair precedes the edits' finish.
// A KeepSelectedFaces edit copies into a new canonical record, whose meshlet build rides the repair.
// Returns each task's output record, the source for an in-place edit, or none when the edit changed nothing.
std::vector<std::optional<uint32_t>> EditTopology(state::Scene &r, std::span<const state::Entity> mesh_entities, std::span<const MeshTopologyTask> tasks) {
    if (tasks.size() != mesh_entities.size()) throw std::invalid_argument("Topology task and entity counts differ.");
    auto &session = project::Session(r);
    auto &meshes = r.Context.get<MeshStore>();
    // A staged inset captures every mesh's basis into the session's preview cache.
    const bool insets = session.Previewing && std::ranges::any_of(tasks, [](const auto &task) {
                            return task.Op == MeshTopologyOp::InsetRegion || task.Op == MeshTopologyOp::InsetIndividual;
                        });
    if (insets && !session.InsetPreview) session.InsetPreview = std::make_unique<action::mesh::InsetPreviewCache>(meshes.BufferContext());
    mtl::ComputeChain chain{meshes.BufferContext(), TopologyScratchWords};
    auto edits = MeshTopologyEdit::Construct(r, chain, tasks, insets ? &session.InsetPreview->Basis : nullptr, TopologyPublication::Editor);
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
    MeshTopologyEdit::FinishAll(r, finished);
    std::vector<state::Entity> in_place;
    std::vector<ElementMeshletRepair> element_repairs;
    for (uint32_t i = 0u; i < edits.size(); ++i) {
        if (outputs[i] != edits[i].SourceId) continue;
        FinishTopologyEdit(r, mesh_entities[i], tasks[i], edits[i], element_repairs);
        in_place.push_back(mesh_entities[i]);
    }
    RepairElementMeshlets(r, chain, element_repairs);
    chain.Submit();
    // Owner maps count only rendered elements, so topology changes need no mesh scan.
    for (const auto entity : in_place) {
        auto &owner = EditRecordOf(r, entity);
        if (!owner.RenderTopologies) continue;
        uint32_t mask = 0u;
        for (uint32_t t = 0u; t < 3u; ++t)
            if (owner.ElementMeshletBlockCounts[t]) mask |= 1u << t;
        owner.RenderTopologies = mask ? mask : 4u;
    }
    RefreshElementSelectionSummaries(r, in_place);
    return outputs;
}

// Runs the tasks as one topology action.
// Spans of tasks edit one after another, each under the scratch budget, so a batch of large meshes holds one span's scratch at a time.
void RunTasks(state::Scene &r, std::span<const state::Entity> mesh_entities, std::span<const MeshTopologyTask> tasks) {
    const profile::CpuScope scope{"TopologyAction"};
    if (tasks.empty()) return;
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto split = ChunkByScratch(uint32_t(tasks.size()), ScratchWordBudget, [&](uint32_t i) { return TopologyScratchBound(meshes, tasks[i]); });
    RunTopologyAction(r, mesh_entities, [&] {
        bool changed = false;
        for (const auto span : split.Chunks) {
            const auto outputs = EditTopology(r, mesh_entities.subspan(span.Offset, span.Count), tasks.subspan(span.Offset, span.Count));
            changed |= std::ranges::any_of(outputs, [](const auto &output) { return output.has_value(); });
        }
        return changed;
    });
}

// Runs the task `make` builds for each mesh, skipping the meshes it returns nothing for.
void RunPerMesh(state::Scene &r, std::span<const state::Entity> mesh_entities, auto &&make) {
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<MeshTopologyTask> tasks;
    std::vector<state::Entity> entities;
    for (const auto e : mesh_entities) {
        if (std::optional<MeshTopologyTask> task = make(e, Mesh{meshes, r.get<const MeshHandle>(e).StoreId})) {
            if (task->Op == MeshTopologyOp::AddPrimitives) task->Selection = EditorGeometrySelection(meshes, task->SourceId);
            tasks.push_back(std::move(*task));
            entities.push_back(e);
        }
    }
    RunTasks(r, entities, tasks);
}

void RunOperator(state::Scene &r, std::span<const state::Entity> mesh_entities, MeshTopologyOp op, float param0 = 0.f, float param1 = 0.f, uint32_t flags = 0) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) { return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = op, .Param0 = param0, .Param1 = param1, .Flags = flags, .Selection = EditorGeometrySelection(r.Context.get<const MeshStore>(), mesh.GetStoreId())}; });
}

void SeparateSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, const ::selection::PrimaryEditInstanceMap &primaries, action::mesh::SeparateMode mode) {
    std::vector<state::Entity> entities;
    std::vector<MeshTopologyTask> tasks;
    const auto &meshes = r.Context.get<const MeshStore>();
    for (const auto e : mesh_entities) {
        const auto id = r.get<const MeshHandle>(e).StoreId;
        const Mesh mesh{meshes, id};
        auto planned = SeparateGeometryTasks(meshes, mesh, EditorGeometrySelection(meshes, id), mode);
        for (auto &task : planned) {
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
        if (created.empty()) return false;
        RequestRender(r, RenderRequest::Rebuild);
        r.Context.get<GpuSceneState>().LodDemand.insert(created.begin(), created.end());
        mtl::ComputeChain chain{r.Context.get<const MeshStore>().BufferContext()};
        UpdateAuthoredMorphShading(r, chain, created);
        chain.Submit();
        return true;
    });
}

std::optional<uint32_t> ActiveOrFirstSelectedEdge(const state::Scene &r, state::Entity mesh_entity, const Mesh &mesh) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selected = meshes.GetSelectedElements(mesh.GetStoreId(), Element::Edge);
    if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity);
        active && meshes.IsLiveElement(mesh.GetStoreId(), Element::Edge, active->Handle) && selected.Contains(active->Handle))
        return active->Handle;
    if (const auto first = selected.First()) return *first;
    return {};
}

void LoopCutSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, uint32_t cuts) {
    RunPerMesh(r, mesh_entities, [&](state::Entity e, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto edge = ActiveOrFirstSelectedEdge(r, e, mesh);
        return edge ? std::optional{LoopCutTask(mesh, *edge, cuts)} : std::nullopt;
    });
}

// Each stage reads the preceding stage's canonical output.
void RunTopologyStages(state::Scene &r, std::span<const state::Entity> mesh_entities, auto &&plan) {
    std::vector<std::vector<MeshTopologyTask>> plans;
    size_t stages = 0u;
    for (const auto entity : mesh_entities) {
        const auto mesh = GetMesh(r, entity);
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

void KnifeSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, const ::selection::PrimaryEditInstanceMap &primaries, vec2 start, vec2 end, const RenderView &view) {
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

// The editor resolves selection and reference frames, then publishes core writes.
void EditSelectedFaceAttributes(state::Scene &r, std::span<const state::Entity> targets, bool colors, uint32_t uv_set, uint32_t operation) {
    auto &meshes = r.Context.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    std::vector<PositionOperationTarget> inputs;
    for (const auto entity : targets) {
        const auto id = GetMesh(r, entity).GetStoreId();
        inputs.push_back({id, EditorGeometrySelection(meshes, id)});
    }
    const auto changes = EncodeFaceAttributeOperation(meshes, GetMeshPipelines(r), chain, inputs, colors, uv_set, operation);
    if (changes.empty()) return;
    uint64_t triangle_words = 0u;
    const auto &a = meshes.Arenas();
    for (const auto &change : changes) {
        const auto &record = meshes.Get(inputs[change.TargetIndex].StoreId);
        const auto bound = std::min(change.Faces.Incidence, a.Triangles.Set(record.TriangleData).BlockCount);
        triangle_words += 1u + ElementWorkWords(a.Triangles.Capacity(), bound) + SortElementWorkWords(WorkCapacity(a.Triangles.Capacity(), bound));
    }
    chain.Scratch.ReserveAdditional(triangle_words);
    std::vector<std::pair<state::Entity, FaceTriangles>> repairs;
    for (const auto &change : changes)
        repairs.emplace_back(targets[change.TargetIndex], EncodeFaceTriangles(r, chain, inputs[change.TargetIndex].StoreId, change.Faces));
    chain.Submit();
    for (auto &[entity, triangles] : repairs) triangles.Finish(chain);
    RepairFaceRender(r, chain, repairs);
    chain.Submit();
    RequestRender(r, RenderRequest::Rebuild);
}

void EditSelectedPositions(state::Scene &r, state::Entity viewport, std::span<const state::Entity> targets, PositionEditOp operation, float factor, uint32_t repeat, PositionOperationOptions options = {}) {
    const bool centered = operation == PositionEditOp::ToSphere || operation == PositionEditOp::PushPull || operation == PositionEditOp::Shear;
    const bool transformed = centered || options.Warp || options.Bend || options.Slide || options.EdgeSlide ||
        (operation == PositionEditOp::Copy && (options.Flags & PositionEditFlattenView));
    const auto &primaries = r.get<const EditPrimaries>(viewport).Transformable;
    if (centered) options.Center = EditSelectionCenter(r, viewport);
    auto &meshes = r.Context.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    std::vector<PositionOperationTarget> inputs;
    std::vector<state::Entity> entities;
    for (const auto entity : targets) {
        if (transformed && !primaries.contains(entity)) continue;
        const auto id = GetMesh(r, entity).GetStoreId();
        PositionOperationTarget input{id, EditorGeometrySelection(meshes, id)};
        if (transformed) input.World = *WorldTransformOf(r, primaries.at(entity));
        if (options.Curve || options.Circle) {
            meshes.GetHiddenElements(id, Element::Edge).ForEach([&](uint32_t edge) { input.Excluded.Edges.push_back(edge); });
            meshes.GetHiddenElements(id, Element::Face).ForEach([&](uint32_t face) { input.Excluded.Faces.push_back(face); });
        }
        if (const auto *active = r.try_get<const MeshActiveElement>(entity);
            active && r.get<const EditMode>(viewport).Value == Element::Vertex) input.Reference = active->Handle;
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
bool UpdateInsetPreview(state::Scene &r, state::Entity viewport, const Inset &inset, InsetPreviewCache &cache) {
    const profile::CpuScope scope{"UpdateInsetPreview"};
    const auto targets = SelectedEditMeshes(r, viewport);
    if (targets.empty() || targets.size() != cache.Entries.size()) return false;
    const auto op = inset.Individual ? MeshTopologyOp::InsetIndividual : MeshTopologyOp::InsetRegion;
    const auto flags = inset.Even ? TopologyFlagEvenOffset : 0u;
    auto &meshes = r.Context.get<MeshStore>();
    // Project::Record discards this cache for every other action, including
    // selection changes. Parameter updates retain the staged face selection.
    for (size_t i = 0u; i < targets.size(); ++i) {
        const auto &entry = cache.Entries[i];
        if (entry.Entity != targets[i] || entry.Op != op || entry.Flags != flags ||
            GetMesh(r, entry.Entity).GetStoreId() != entry.StoreId || !entry.Basis.Count) return false;
    }
    {
        const profile::CpuScope capture{"InsetCaptureVertices"};
        for (const auto &entry : cache.Entries)
            meshes.Arenas().Vertices.Buffer.CaptureWriteRanges(entry.Ranges, sizeof(Vertex));
    }
    // Keep the position update and its dependent geometry work on one chain.
    mtl::ComputeChain chain{meshes.BufferContext()};
    {
        const profile::CpuScope positions{"InsetPositionPass"};
        const auto &pipeline = GetMeshPipelines(r)[MeshPass::InsetPreviewPositions];
        // Each entry writes its own mesh's vertices.
        chain.Concurrent([&] {
            for (const auto &entry : cache.Entries) {
                const InsetPreviewPushConstants pc{
                    .Basis = {cache.Basis.Buffer.Slot, entry.Basis.Offset}, .VertexSlot = meshes.Slots().Vertices, .Count = uint32_t(uint64_t(entry.Basis.Count) * sizeof(uint32_t) / sizeof(InsetVertexBasis)), .Thickness = std::max(inset.Thickness, 0.f), .Depth = inset.Depth
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
    RequestRender(r, RenderRequest::Reuse);
    return true;
}

void Apply(state::Scene &r, state::Entity viewport, const Action &action) {
    const auto targets = SelectedEditMeshes(r, viewport);
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selected = [&](auto &&make) {
        RunPerMesh(r, targets, [&](state::Entity, const Mesh &mesh) { return make(mesh, EditorGeometrySelection(meshes, mesh.GetStoreId())); });
    };
    const auto visibility = [&](EditVisibilityOperation operation) {
        std::vector<uint32_t> ids;
        for (const auto &[entity, instance] : r.get<const EditPrimaries>(viewport).All)
            if (HasMesh(r, entity)) ids.push_back(GetMesh(r, entity).GetStoreId());
        if (EditVisibility(r, ids, r.get<const EditMode>(viewport).Value, operation)) {
            r.Context.get<GpuSceneState>().EditSelectionDirty = true;
            for (const auto &[entity, instance] : r.get<const EditPrimaries>(viewport).All) r.remove<MeshActiveElement>(entity);
        }
    };
    const auto latch_translate = [&] { r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate); };
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
                    for (const auto entity : targets) empty.emplace_back(entity, std::span<const uint32_t>{});
                    ApplyEditSelectionLists(r, empty, r.get<const EditMode>(viewport).Value);
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
                const auto &primaries = r.get<const EditPrimaries>(viewport).All;
                std::vector<state::Entity> sources;
                for (const auto &[entity, instance] : primaries)
                    if (HasMesh(r, entity)) sources.push_back(entity);
                SeparateSelected(r, sources, primaries, a.Mode);
            },
            [&](const Subdivide &a) { RunOperator(r, targets, MeshTopologyOp::Subdivide, float(std::max(a.Cuts, 1u))); },
            [&](const SnapSymmetry &a) {
                if (uint32_t(a.Axis) > 2u) return;
                EditSelectedPositions(r, viewport, targets, PositionEditOp::SnapSymmetry, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = 1u << uint32_t(a.Axis), .Flags = a.Negative ? PositionEditSymmetryNegative : 0u, .SnapSymmetry = PositionSymmetry{a.Threshold, a.Center}});
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
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Smooth, a.Factor, std::clamp(a.Repeat, 1u, 1000u), {.Axes = uint32_t(a.X) | (uint32_t(a.Y) << 1u) | (uint32_t(a.Z) << 2u)});
            },
            [&](const SpaceEvenly &a) {
                EditSelectedPositions(r, viewport, targets, PositionEditOp::SpaceEvenly, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = uint32_t(a.X) | (uint32_t(a.Y) << 1u) | (uint32_t(a.Z) << 2u), .Flags = a.Interpolation == EdgeLoopInterpolation::Cubic ? PositionEditCurveCubic : 0u});
            },
            [&](const RelaxEdgeLoops &a) {
                if (!a.Iterations) return;
                EditSelectedPositions(r, viewport, targets, PositionEditOp::RelaxEdgeLoops, 1.f, std::min(a.Iterations, 1000u), {.Flags = (a.Interpolation == EdgeLoopInterpolation::Cubic ? PositionEditCurveCubic : 0u) | (a.EvenSpacing ? PositionEditRelaxEven : 0u)});
            },
            [&](const ToSphere &a) { EditSelectedPositions(r, viewport, targets, PositionEditOp::ToSphere, a.Factor, 1u); },
            [&](const PushPull &a) { EditSelectedPositions(r, viewport, targets, PositionEditOp::PushPull, a.Distance, 1u); },
            [&](const Shear &a) {
                const auto axis = uint32_t(a.Axis), ortho = uint32_t(a.AxisOrtho);
                if (axis > 2u || ortho > 2u || axis == ortho) return;
                vec3 normal{}, direction{};
                normal[axis] = 1.f;
                direction[ortho] = 1.f;
                if (a.Local) {
                    const auto &primaries = r.get<const EditPrimaries>(viewport).Transformable;
                    const auto active = primaries.find(GetActiveMeshEntity(r));
                    if (active == primaries.end()) return;
                    const auto &world = *WorldTransformOf(r, active->second);
                    normal = world.R * normal * (world.S[axis] < 0.f ? -1.f : 1.f);
                    direction = world.R * direction * (world.S[ortho] < 0.f ? -1.f : 1.f);
                }
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Shear, std::tan(a.Angle), 1u, {.Direction = direction, .Gradient = Cross(direction, normal)});
            },
            [&](const Warp &a) {
                if (!a.Orientation) return;
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Warp, a.Angle, 1u, {.Warp = PositionWarp{*a.Orientation, a.Center, a.OffsetAngle, a.Min, a.Max, a.AutoRange}});
            },
            [&](const Bend &a) {
                if (!a.Orientation) return;
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Bend, a.Angle, 1u, {.Flags = a.Clamp ? PositionEditBendClamp : 0u, .Bend = PositionBend{*a.Orientation, a.Center, a.OffsetAngle, a.Radius}});
            },
            [&](const Randomize &a) {
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Randomize, a.Amount, 1u, {.Randomize = PositionRandomize{a.Uniform, a.Normal, a.Seed}});
            },
            [&](const VertexSlide &a) {
                EditSelectedPositions(r, viewport, targets, PositionEditOp::VertexSlide, a.Clamp ? std::clamp(a.Factor, 0.f, 1.f) : a.Factor, 1u, {.Flags = a.Clamp ? 0u : PositionEditSlideUnclamped, .Slide = PositionSlide{a.Direction, a.Even, a.Flipped}});
            },
            [&](const EdgeSlide &a) {
                EditSelectedPositions(r, viewport, targets, PositionEditOp::EdgeSlide, a.Clamp ? std::clamp(a.Factor, -1.f, 1.f) : a.Factor, 1u, {.Flags = a.Clamp ? 0u : PositionEditSlideUnclamped, .EdgeSlide = PositionSlide{a.Direction, a.Even, a.Flipped}});
            },
            [&](const Circularize &a) {
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Copy, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = uint32_t(a.X) | (uint32_t(a.Y) << 1u) | (uint32_t(a.Z) << 2u), .Flags = (a.Regular ? PositionEditCurveRegular : 0u) | (a.Method == CircleFit::Contract ? PositionEditCircleContract : 0u), .Circle = PositionCircle{a.Radius, a.Angle}});
            },
            [&](const CurveBetweenSelected &a) {
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Copy, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = uint32_t(a.X) | (uint32_t(a.Y) << 1u) | (uint32_t(a.Z) << 2u), .Flags = (a.Interpolation == EdgeLoopInterpolation::Cubic ? PositionEditCurveCubic : 0u) | (a.Regular ? PositionEditCurveRegular : 0u) | (a.Elevation == CurveElevation::Raise ? PositionEditCurveRaise : a.Elevation == CurveElevation::Lower ? PositionEditCurveLower :
                                                                                                                                                                                                                                                                                                                                                                                                                                               0u),
                                                                                                                       .Curve = PositionCurve{a.Extend}});
            },
            [&](const MakePlanarFaces &a) { EditSelectedPositions(r, viewport, targets, PositionEditOp::Planar, a.Factor, std::clamp(a.Repeat, 1u, 10000u)); },
            [&](const Flatten &a) {
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Copy, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = uint32_t(a.X) | (uint32_t(a.Y) << 1u) | (uint32_t(a.Z) << 2u), .Flags = a.Method == FlattenMethod::View ? PositionEditFlattenView : a.Method == FlattenMethod::FaceNormals ? PositionEditFlattenNormals :
                                                                                                                                                                                                                                                                                                            0u,
                                                                                                                       .Direction = a.ViewNormal});
            },
            [&](const ShrinkFatten &a) {
                EditSelectedPositions(r, viewport, targets, PositionEditOp::ShrinkFatten, a.Distance, 1u, {.Flags = (a.Even ? PositionEditEvenOffset : 0u) | (r.get<const EditMode>(viewport).Value == Element::Face ? PositionEditSelectedFaceNormals : 0u)});
            },
            [&](const SplitNonplanarFaces &a) { RunOperator(r, targets, MeshTopologyOp::SplitNonplanarFaces, std::clamp(a.Angle, 0.f, std::numbers::pi_v<float>)); },
            [&](SplitConcaveFaces) { RunOperator(r, targets, MeshTopologyOp::SplitConcaveFaces); },
            [&](const Poke &a) { RunOperator(r, targets, MeshTopologyOp::Poke, a.Offset); },
            [&](FlipNormals) { RunOperator(r, targets, MeshTopologyOp::FlipNormals); },
            [&](const RecalculateNormals &a) {
                std::vector<uint32_t> ids;
                for (const auto entity : targets) ids.push_back(GetMesh(r, entity).GetStoreId());
                std::vector<GeometrySelection> selections;
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
            [&](const Knife &a) { KnifeSelected(r, targets, r.get<const EditPrimaries>(viewport).All, a.Start, a.End, *a.View); },
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
                    case Mode::Limited: return RunOperator(r, targets, MeshTopologyOp::DissolveLimited, std::clamp(a.Angle, 0.f, .5f * std::numbers::pi_v<float>), 0.f, (r.get<const EditMode>(viewport).Value == Element::Face ? TopologyFlagFaceSelection : 0u) | (a.AllBoundaries ? TopologyFlagAllBoundaries : 0u) | (a.DelimitMaterials ? TopologyFlagDelimitMaterial : 0u) | (a.DelimitSharpEdges ? TopologyFlagDelimitSharp : 0u) | (a.DelimitUVs ? TopologyFlagDelimitUV : 0u));
                    case Mode::Degenerate: return RunOperator(r, targets, MeshTopologyOp::DissolveDegenerate, std::max(a.Distance, 0.f));
                }
            },
        },
        action
    );
}
} // namespace action::mesh
