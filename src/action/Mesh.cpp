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
#include "mesh/EdgeChains.h"
#include "mesh/EdgeSlide.h"
#include "mesh/EditVisibility.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/Flatten.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshEdgeUsers.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopology.h"
#include "mesh/MeshTopologyEdit.h"
#include "mesh/PrimitiveType.h"
#include "mesh/RecalculateNormals.h"
#include "mesh/ScratchChunks.h"
#include "mesh/SnapSymmetry.h"
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
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {
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
            tasks.push_back(std::move(*task));
            entities.push_back(e);
        }
    }
    RunTasks(r, entities, tasks);
}

void RunOperator(state::Scene &r, std::span<const state::Entity> mesh_entities, MeshTopologyOp op, float param0 = 0.f, float param1 = 0.f, uint32_t flags = 0) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) { return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = op, .Param0 = param0, .Param1 = param1, .Flags = flags}; });
}

// The lowest and highest selected vertex of a mesh.
std::pair<uint32_t, uint32_t> SelectedVertexSpan(const MeshStore &meshes, uint32_t id) {
    const auto selected = meshes.GetSelectedElements(id, Element::Vertex);
    const auto first = selected.First(), last = selected.Last();
    return {first.value_or(InvalidOffset), last.value_or(InvalidOffset)};
}

// Partition outputs copy before one removal per source; shared boundary vertices are retained by the transaction.
void SeparateSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, const ::selection::PrimaryEditInstanceMap &primaries, action::mesh::SeparateMode mode) {
    std::vector<state::Entity> entities;
    std::vector<MeshTopologyTask> tasks;
    const auto &meshes = r.Context.get<const MeshStore>();
    for (const auto e : mesh_entities) {
        const auto id = r.get<const MeshHandle>(e).StoreId;
        const Mesh mesh{meshes, id};
        Element element = Element::Vertex;
        std::vector<std::vector<uint32_t>> groups;
        if (mode == action::mesh::SeparateMode::Selected) {
            element = mesh.FaceCount() ? Element::Face : Element::Vertex;
            auto &group = groups.emplace_back();
            meshes.GetSelectedElements(id, element).ForEach([&](uint32_t h) { group.push_back(h); });
            if (group.empty()) groups.clear();
        } else if (mode == action::mesh::SeparateMode::LooseParts) {
            const auto incidence = mesh.GetVertexEdgeIncidence();
            std::unordered_set<uint32_t> visited;
            for (const auto v : mesh.vertices()) {
                if (!visited.insert(*v).second) continue;
                auto &group = groups.emplace_back(1u, *v);
                for (size_t at = 0u; at < group.size(); ++at)
                    for (const auto edge : incidence.Incident(group[at])) {
                        const auto h = mesh.GetHalfedge(he::EH{edge}, 0u);
                        const auto a = *mesh.GetFromVertex(h), b = *mesh.GetToVertex(h), other = a == group[at] ? b : a;
                        if (visited.insert(other).second) group.push_back(other);
                    }
            }
            // Blender keeps the first connected component in the original object.
            if (!groups.empty()) groups.erase(groups.begin());
        } else {
            element = Element::Face;
            const auto &a = meshes.Arenas();
            const auto palette = a.PrimitiveMaterials.Get(meshes.Get(id).PrimitiveMaterials);
            std::unordered_map<uint32_t, uint32_t> materials;
            for (const auto face : mesh.faces()) {
                const auto material = palette[a.FacePrimitives.Get(*face)];
                const auto [entry, inserted] = materials.try_emplace(material, uint32_t(groups.size()));
                if (inserted) groups.emplace_back();
                groups[entry->second].push_back(*face);
            }
            // Blender extracts successive material groups, leaving the last in place.
            if (!groups.empty()) groups.pop_back();
        }
        if (groups.empty()) continue;
        std::vector<uint32_t> removed;
        for (auto &group : groups) {
            std::ranges::sort(group);
            removed.insert(removed.end(), group.begin(), group.end());
            tasks.push_back({.SourceId = id, .Op = MeshTopologyOp::KeepSelectedFaces, .SelectionElement = element, .Selected = std::move(group)});
            entities.push_back(e);
        }
        std::ranges::sort(removed);
        tasks.push_back({.SourceId = id, .Op = element == Element::Face ? MeshTopologyOp::DeleteFaces : mesh.FaceCount() ? MeshTopologyOp::DeleteVertices :
                                                                                                                           MeshTopologyOp::DeleteEdges,
                         .SelectionElement = element,
                         .Selected = std::move(removed)});
        entities.push_back(e);
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

// The lowest selected edge of a mesh, or the active one when the active element is an edge.
std::optional<uint32_t> ActiveOrFirstSelectedEdge(const state::Scene &r, state::Entity mesh_entity, const Mesh &mesh) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selected = meshes.GetSelectedElements(mesh.GetStoreId(), Element::Edge);
    if (const auto *active = r.try_get<const MeshActiveElement>(mesh_entity);
        active && meshes.IsLiveElement(mesh.GetStoreId(), Element::Edge, active->Handle) && selected.Contains(active->Handle))
        return active->Handle;
    if (const auto first = selected.First()) return *first;
    return {};
}

// The ring of edges across quads from `edge`, walked both ways until a non-quad, a boundary, or the ring closes.
std::vector<uint32_t> EdgeRing(const Mesh &mesh, uint32_t edge) {
    std::vector<uint32_t> ring{edge};
    std::unordered_set<uint32_t> visited{edge};
    const auto &c = mesh.GetConnectivity();
    const auto start = mesh.GetHalfedge(he::EH{edge}, 0);
    for (const auto side : {start, c.Opposites[*start]}) {
        auto h = side;
        while (h) {
            const auto face = c.FaceOf(h);
            if (!face || mesh.GetValence(face) != 4) break;
            const auto across = c.Next(c.Next(h));
            const auto e = mesh.GetEdge(across);
            if (!visited.insert(*e).second) break;
            ring.push_back(*e);
            h = c.Opposites[*across];
        }
    }
    return ring;
}

// Closed boundary loops wound against the surface, ready to fill.
std::vector<std::vector<uint32_t>> BoundaryLoops(state::Scene &r, const Mesh &mesh, bool selected_only, uint32_t max_sides = 0u) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &c = mesh.GetConnectivity();
    // Both views visit their edges in ascending handle order.
    std::vector<uint32_t> edges;
    const auto collect = [&](const auto &view) { view.ForEach([&](uint32_t edge) { edges.push_back(edge); }); };
    if (selected_only) collect(meshes.GetSelectedElements(mesh.GetStoreId(), Element::Edge));
    else collect(meshes.GetBoundaryEdges(mesh.GetStoreId()));
    std::vector<uint32_t> starts;
    for (const auto edge : edges) {
        const auto h = mesh.GetHalfedge(he::EH{edge}, 0);
        if (!c.Opposites[*h]) starts.push_back(*h);
    }
    std::ranges::sort(starts);
    const auto candidate = [&](uint32_t h) {
        return h != InvalidOffset && !c.Opposites[h] && std::ranges::binary_search(edges, *mesh.GetEdge(Mesh::HH{h}));
    };
    std::unordered_map<uint32_t, uint32_t> selected_outgoing;
    if (selected_only)
        for (const auto h : starts) {
            const auto vertex = *mesh.GetFromVertex(Mesh::HH{h});
            const auto [it, unique] = selected_outgoing.emplace(vertex, h);
            if (!unique) it->second = InvalidOffset;
        }
    // Follow the face fan at the current boundary halfedge's destination to
    // find the next boundary halfedge on the same surface sheet. Vertex-based
    // pairing loses loops when distinct boundaries share a vertex.
    const auto successor = [&](uint32_t h) -> uint32_t {
        auto next = c.Next(Mesh::HH{h});
        const auto across = [&](Mesh::HH at) -> Mesh::HH {
            if (!at) return {};
            const auto opposite = c.Opposites[*at];
            return opposite ? c.Next(opposite) : Mesh::HH{};
        };
        auto fast = next;
        while (next && c.Opposites[*next]) {
            next = across(next);
            fast = across(across(fast));
            if (fast && next == fast) return InvalidOffset;
        }
        return next ? *next : InvalidOffset;
    };
    std::vector<std::vector<uint32_t>> loops;
    std::unordered_set<uint32_t> used;
    used.reserve(starts.size());
    for (const auto start : starts) {
        if (used.contains(start)) continue;
        std::vector<uint32_t> loop;
        auto h = start;
        bool closed = false;
        uint32_t length = 0u;
        while (candidate(h)) {
            if (!used.insert(h).second) {
                closed = h == start;
                break;
            }
            ++length;
            if (!max_sides || loop.size() < max_sides) loop.push_back(*mesh.GetFromVertex(Mesh::HH{h}));
            const auto next = successor(h);
            if (selected_only && !candidate(next)) {
                // A selected hole may touch an unselected boundary at one
                // vertex. Follow its sole selected outgoing edge there.
                const auto it = selected_outgoing.find(*mesh.GetToVertex(Mesh::HH{h}));
                h = it == selected_outgoing.end() ? InvalidOffset : it->second;
            } else h = next;
        }
        if (closed && length >= 3u && (!max_sides || length <= max_sides)) {
            std::ranges::reverse(loop);
            loops.push_back(std::move(loop));
        }
    }
    return loops;
}

// A face list task over canonical vertex handles. New vertices use handles
// starting at the current arena capacity, beyond every existing handle.
MeshTopologyTask PrimitiveListTask(state::Scene &r, const Mesh &mesh, std::span<const std::vector<uint32_t>> loops, const std::unordered_map<uint64_t, uint32_t> &edge_sources = {}, std::span<const uint32_t> grid_loop = {}, uint32_t grid_span = 0u) {
    const auto appended_base = r.Context.get<const MeshStore>().Arenas().Vertices.Capacity();
    const uint64_t vertices = grid_loop.empty() ? 0u : uint64_t(grid_span - 1u) * (grid_loop.size() / 2u - grid_span - 1u);
    if (uint64_t(appended_base) + vertices > UINT32_MAX) throw std::length_error("Face list exceeds the vertex handle address space.");
    MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::AddPrimitives, .AppendedBase = appended_base};
    uint32_t attribute_source = InvalidOffset;
    for (const auto &loop : loops)
        for (const auto vertex : loop)
            if (vertex < appended_base && attribute_source == InvalidOffset) attribute_source = vertex;
    if (attribute_source == InvalidOffset) throw std::invalid_argument("Face creation needs a source vertex for attributes.");
    task.List = {uint32_t(vertices), uint32_t(grid_loop.size()), grid_span, attribute_source};
    task.List.insert(task.List.end(), grid_loop.begin(), grid_loop.end());
    task.List.push_back(uint32_t(loops.size()));
    for (const auto &loop : loops) {
        task.List.push_back(uint32_t(loop.size()));
        for (uint32_t i = 0u; i < loop.size(); ++i) {
            task.List.push_back(loop[i]);
            const auto edge = edge_sources.find(MeshEdgeUsers::Key(loop[(i + loop.size() - 1u) % loop.size()], loop[i]));
            task.List.push_back(edge == edge_sources.end() ? InvalidOffset : edge->second);
        }
    }
    return task;
}

std::unordered_map<uint64_t, uint32_t> SelectedEdgeSources(const MeshStore &meshes, const Mesh &mesh, bool boundary_only) {
    std::unordered_map<uint64_t, uint32_t> sources;
    MeshEdgeUsers users{mesh};
    meshes.GetSelectedElements(mesh.GetStoreId(), Element::Edge).ForEach([&](uint32_t edge) {
        const auto h = mesh.GetHalfedge(he::EH{edge}, 0u);
        if (mesh.GetFromVertex(h) == mesh.GetToVertex(h)) return;
        const auto key = users.Key(h);
        // Retain the selected surface corner even when a coincident wire supplied a different radial corner first.
        if (!boundary_only && mesh.GetConnectivity().FaceOf(h)) {
            sources[key] = *h;
            return;
        }
        const auto adjacent = users.Get(h);
        if (!boundary_only || adjacent.Count <= 1u) sources.try_emplace(key, adjacent.Count ? adjacent.First : *h);
    });
    return sources;
}

struct SelectedChain {
    std::vector<uint32_t> Vertices;
    bool Closed{};
    int Winding{};
};

// Selected boundary and loose edges, with a corner source for each reusable edge.
std::vector<SelectedChain> SelectedChains(state::Scene &r, const Mesh &mesh, std::unordered_map<uint64_t, uint32_t> &sources) {
    sources = SelectedEdgeSources(r.Context.get<const MeshStore>(), mesh, true);
    const auto &c = mesh.GetConnectivity();
    EdgeGraph neighbors;
    for (const auto &[key, h] : sources) {
        const auto a = uint32_t(key >> 32u), b = uint32_t(key);
        neighbors[a].push_back(b);
        neighbors[b].push_back(a);
    }
    for (const auto &[v, adjacent] : neighbors)
        if (adjacent.size() > 2u) return {};
    std::vector<SelectedChain> chains;
    VisitEdgeChains(neighbors, [&](const auto &vertices, bool closed) {
        auto &chain = chains.emplace_back(SelectedChain{vertices, closed});
        for (size_t i = 0u; i < vertices.size() - (closed ? 0u : 1u); ++i) {
            const auto v = vertices[i], next = vertices[(i + 1u) % vertices.size()];
            const auto h = he::HH{sources.at(MeshEdgeUsers::Key(v, next))};
            if (c.FaceOf(h)) chain.Winding += *mesh.GetFromVertex(h) == v ? 1 : -1;
        }
    });
    return chains;
}

void BridgeSelected(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        std::unordered_map<uint64_t, uint32_t> sources;
        auto chains = SelectedChains(r, mesh, sources);
        if (chains.size() != 2u || chains[0].Closed != chains[1].Closed) return {};
        if (chains[0].Vertices.size() < chains[1].Vertices.size()) std::swap(chains[0], chains[1]);
        const bool closed = chains[0].Closed;
        auto &a = chains[0].Vertices, &b = chains[1].Vertices;
        const auto flip = [&](auto &chain) { std::reverse(chain.begin() + (closed ? 1u : 0u), chain.end()); };
        // New faces oppose the existing face along each rail.
        if (chains[0].Winding < 0) flip(a);
        if (chains[1].Winding > 0) flip(b);
        const auto position = [&](uint32_t v) { return mesh.GetPosition(he::VH{v}); };
        const auto normal = [&](const auto &loop) {
            vec3 n{};
            const auto origin = position(loop[0]);
            for (uint32_t i = 1u; i + 1u < loop.size(); ++i) n += Cross(position(loop[i]) - origin, position(loop[i + 1u]) - origin);
            return n;
        };
        const auto align = [&](const auto &fixed, auto &free) {
            if (closed) {
                if (Dot(normal(fixed), normal(free)) < 0.f) flip(free);
            } else {
                const auto same = Distance2(position(fixed.front()), position(free.front())) + Distance2(position(fixed.back()), position(free.back()));
                const auto crossed = Distance2(position(fixed.front()), position(free.back())) + Distance2(position(fixed.back()), position(free.front()));
                if (crossed < same) flip(free);
            }
        };
        if (!chains[1].Winding) align(a, b);
        else if (!chains[0].Winding) align(b, a);
        if (closed) {
            if (!chains[0].Winding && !chains[1].Winding) {
                vec3 separation{};
                for (const auto v : a) separation += position(v) / float(a.size());
                for (const auto v : b) separation -= position(v) / float(b.size());
                if (Dot(normal(a), separation) < 0.f) {
                    flip(a);
                    flip(b);
                }
            }
            uint32_t start = 0u;
            float best = std::numeric_limits<float>::max();
            for (uint32_t j = 0u; j < b.size(); ++j)
                if (const auto distance = Distance2(position(a[0]), position(b[j])); distance < best) {
                    best = distance;
                    start = j;
                }
            std::rotate(b.begin(), b.begin() + start, b.end());
        }
        const auto na = uint32_t(a.size()) - uint32_t(!closed), nb = uint32_t(b.size()) - uint32_t(!closed);
        std::vector<std::vector<uint32_t>> faces;
        faces.reserve(na);
        for (uint32_t i = 0u; i < na; ++i) {
            const auto j0 = uint32_t(uint64_t(i) * nb / na), j1 = uint32_t(uint64_t(i + 1u) * nb / na);
            const auto next = a[(i + 1u) % a.size()];
            if (j0 == j1) faces.push_back({next, a[i], b[j0]});
            else faces.push_back({next, a[i], b[j0], b[j1 % b.size()]});
        }
        return PrimitiveListTask(r, mesh, faces, sources);
    });
}

// Fills one closed loop of selected boundary edges with a Coons patch of quads, `span` edges along its first side.
void GridFillSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, uint32_t span) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        std::unordered_map<uint64_t, uint32_t> sources;
        auto chains = SelectedChains(r, mesh, sources);
        if (chains.size() != 1u || !chains[0].Closed || chains[0].Vertices.size() % 2u || chains[0].Vertices.size() < 4u) return {};
        auto &loop = chains[0].Vertices;
        if (chains[0].Winding < 0) std::reverse(loop.begin() + 1u, loop.end());
        const auto length = uint32_t(loop.size());
        const auto appended_base = r.Context.get<const MeshStore>().Arenas().Vertices.Capacity();
        const uint32_t s = std::clamp(span == 0 ? std::max(length / 4, 1u) : span, 1u, length / 2 - 1), t = length / 2 - s;
        const uint64_t interior = uint64_t(s - 1u) * (t - 1u), cells = uint64_t(s) * t;
        if (uint64_t(appended_base) + interior > UINT32_MAX || 5ull + length + 9u * cells > UINT32_MAX) {
            throw std::length_error("Grid fill exceeds the vertex or face-list address space.");
        }
        // Nodes run along the first side (u) and up the second (v), with the loop's four sides as the rails.
        const auto node = [&](uint32_t i, uint32_t j) {
            if (j == 0u) return loop[i];
            if (j == t) return loop[(2u * s + t - i) % length];
            if (i == 0u) return loop[(length - j) % length];
            if (i == s) return loop[s + j];
            return appended_base + (j - 1u) * (s - 1u) + i - 1u;
        };
        std::vector<std::vector<uint32_t>> faces;
        for (uint32_t j = 0; j < t; ++j) {
            for (uint32_t i = 0; i < s; ++i) faces.push_back({node(i + 1, j), node(i, j), node(i, j + 1), node(i + 1, j + 1)});
        }
        return PrimitiveListTask(r, mesh, faces, sources, loop, s);
    });
}

void FillHolesSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, uint32_t sides) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        auto loops = BoundaryLoops(r, mesh, false, sides);
        if (loops.empty()) return {};
        return PrimitiveListTask(r, mesh, loops);
    });
}

// The convex hull of the selected vertices as outward triangles, by quickhull.
// Each face keeps the points outside it, and adding a face's farthest point replaces only the faces that point sees.
void ConvexHullSelected(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    const auto &meshes = r.Context.get<const MeshStore>();
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        std::vector<uint32_t> points;
        meshes.GetSelectedElements(mesh.GetStoreId(), Element::Vertex).ForEach([&](uint32_t v) { points.push_back(v); });
        if (points.size() < 4) return {};
        const auto at = [&](uint32_t v) { return mesh.GetPosition(he::VH{v}); };
        // A starting tetrahedron from the first point, the farthest from it, the farthest from that line, and the farthest from that plane.
        std::array<uint32_t, 4> seed{points[0], points[0], points[0], points[0]};
        const auto farthest = [&](uint32_t &vertex, auto &&distance) {
            float best = 0.f;
            for (const auto v : points)
                if (const auto d = distance(v); d > best) {
                    best = d;
                    vertex = v;
                }
            return best;
        };
        const float extent = std::sqrt(farthest(seed[1], [&](uint32_t v) { return Distance2(at(v), at(seed[0])); }));
        farthest(seed[2], [&](uint32_t v) { return Length(Cross(at(seed[1]) - at(seed[0]), at(v) - at(seed[0]))); });
        const auto seed_normal = Cross(at(seed[1]) - at(seed[0]), at(seed[2]) - at(seed[0]));
        if (farthest(seed[3], [&](uint32_t v) { return std::abs(Dot(seed_normal, at(v) - at(seed[0]))); }) < 1e-12f) return {};

        struct Face {
            std::array<uint32_t, 3> V;
            // The face across each edge V[k] to V[k + 1].
            std::array<uint32_t, 3> Across{};
            vec3 Normal;
            float Offset, Tolerance;
            std::vector<uint32_t> Outside;
            bool Alive{true};
        };
        std::vector<Face> faces;
        // A point counts as outside a face beyond a sliver of the point set's extent, so coplanar points stay inside.
        const float tolerance = 1e-6f * extent;
        const auto make = [&](std::array<uint32_t, 3> v) {
            Face face{.V = v, .Normal = Cross(at(v[1]) - at(v[0]), at(v[2]) - at(v[0]))};
            face.Offset = Dot(face.Normal, at(v[0]));
            face.Tolerance = Length(face.Normal) * tolerance;
            return face;
        };
        const auto height = [&](const Face &face, uint32_t p) { return Dot(face.Normal, at(p)) - face.Offset; };
        const auto outside = [&](const Face &face, uint32_t p) { return height(face, p) > face.Tolerance; };
        const auto outward = [&](std::array<uint32_t, 3> tri, uint32_t inside) {
            const auto n = Cross(at(tri[1]) - at(tri[0]), at(tri[2]) - at(tri[0]));
            return Dot(n, at(inside) - at(tri[0])) > 0.f ? std::array{tri[0], tri[2], tri[1]} : tri;
        };
        // The slot of the edge `from` to `to` on a face, or three when the face lacks it.
        const auto edge_slot = [](const Face &face, uint32_t from, uint32_t to) {
            uint32_t j = 0;
            while (j < 3 && !(face.V[j] == from && face.V[(j + 1) % 3] == to)) ++j;
            return j;
        };
        // A point waits on the first face from `first` that sees it, or falls inside.
        const auto assign = [&](uint32_t v, uint32_t first) {
            for (uint32_t i = first; i < faces.size(); ++i) {
                if (outside(faces[i], v)) {
                    faces[i].Outside.push_back(v);
                    return;
                }
            }
        };
        std::vector<uint32_t> pending;
        const auto enqueue = [&](uint32_t first) {
            for (uint32_t i = first; i < faces.size(); ++i)
                if (!faces[i].Outside.empty()) pending.push_back(i);
        };
        faces.push_back(make(outward({seed[0], seed[1], seed[2]}, seed[3])));
        faces.push_back(make(outward({seed[0], seed[1], seed[3]}, seed[2])));
        faces.push_back(make(outward({seed[0], seed[2], seed[3]}, seed[1])));
        faces.push_back(make(outward({seed[1], seed[2], seed[3]}, seed[0])));
        // The tetrahedron's faces meet across each shared edge, in opposite directions.
        for (uint32_t a = 0; a < 4; ++a) {
            for (uint32_t k = 0; k < 3; ++k) {
                for (uint32_t b = 0; b < 4; ++b) {
                    if (edge_slot(faces[b], faces[a].V[(k + 1) % 3], faces[a].V[k]) < 3) faces[a].Across[k] = b;
                }
            }
        }
        for (const auto v : points)
            if (std::ranges::find(seed, v) == seed.end()) assign(v, 0);
        enqueue(0);

        struct HorizonEdge {
            uint32_t From, To, Neighbor, NeighborEdge;
        };
        std::vector<uint32_t> visible, stack;
        std::vector<HorizonEdge> horizon;
        // The search that last reached each face.
        std::vector<uint32_t> visited(faces.size(), 0u);
        uint32_t search = 0;
        while (!pending.empty()) {
            const auto start = pending.back();
            pending.pop_back();
            if (!faces[start].Alive || faces[start].Outside.empty()) continue;
            const auto p = *std::ranges::max_element(faces[start].Outside, {}, [&](uint32_t v) { return height(faces[start], v); });
            // The faces the point sees form one connected region, whose boundary edges are the horizon.
            visible.clear();
            horizon.clear();
            ++search;
            stack.assign(1, start);
            visited[start] = search;
            while (!stack.empty()) {
                const auto f = stack.back();
                stack.pop_back();
                visible.push_back(f);
                for (uint32_t k = 0; k < 3; ++k) {
                    const auto n = faces[f].Across[k];
                    const bool sees = outside(faces[n], p);
                    if (!sees) horizon.push_back({faces[f].V[k], faces[f].V[(k + 1) % 3], n, 0});
                    if (visited[n] == search) continue;
                    visited[n] = search;
                    if (sees) stack.push_back(n);
                }
            }
            for (auto &edge : horizon) edge.NeighborEdge = edge_slot(faces[edge.Neighbor], edge.To, edge.From);
            // Each horizon edge fans to the point, and consecutive fans meet along the point's spokes.
            const uint32_t first_new = uint32_t(faces.size());
            std::unordered_map<uint32_t, uint32_t> fan_from, fan_to;
            for (uint32_t i = 0; i < horizon.size(); ++i) {
                const auto &edge = horizon[i];
                faces.push_back(make({edge.From, edge.To, p}));
                fan_from[edge.From] = first_new + i;
                fan_to[edge.To] = first_new + i;
            }
            // A horizon that is not one simple loop means the tolerance split a nearly coplanar region, and the hull is abandoned.
            if (fan_from.size() != horizon.size() || fan_to.size() != horizon.size()) return {};
            visited.resize(faces.size(), 0u);
            for (uint32_t i = 0; i < horizon.size(); ++i) {
                const auto &edge = horizon[i];
                auto &face = faces[first_new + i];
                face.Across = {edge.Neighbor, fan_from.at(edge.To), fan_to.at(edge.From)};
                faces[edge.Neighbor].Across[edge.NeighborEdge] = first_new + i;
            }
            // The visible faces' outside points move to the new faces.
            for (const auto f : visible) {
                auto &face = faces[f];
                face.Alive = false;
                for (const auto v : face.Outside)
                    if (v != p) assign(v, first_new);
                face.Outside.clear();
            }
            enqueue(first_new);
        }
        std::vector<std::vector<uint32_t>> hull;
        for (const auto &face : faces)
            if (face.Alive) hull.push_back({face.V[0], face.V[1], face.V[2]});
        if (hull.empty()) return {};
        return PrimitiveListTask(r, mesh, hull);
    });
}

// Dissolves selected edges and connects their far vertices in one local topology transaction.
void EdgeRotateSelected(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    const auto &meshes = r.Context.get<const MeshStore>();
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto &c = mesh.GetConnectivity();
        MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::RotateEdges, .Flags = TopologyFlagListSelects, .List = {0}};
        meshes.GetSelectedElements(mesh.GetStoreId(), Element::Edge).ForEach([&](uint32_t edge) {
            const auto h = mesh.GetHalfedge(he::EH{edge}, 0);
            const auto opposite = c.Opposites[*h];
            if (!opposite) return;
            task.List.push_back(*mesh.GetToVertex(c.Next(h)));
            task.List.push_back(*mesh.GetToVertex(c.Next(opposite)));
            task.List[0] += 2;
        });
        if (task.List[0] == 0) return {};
        return task;
    });
}

void FillSelected(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto &meshes = r.Context.get<const MeshStore>();
        std::vector<uint32_t> vertices;
        meshes.GetSelectedElements(mesh.GetStoreId(), Element::Vertex).ForEach([&](uint32_t v) { vertices.push_back(v); });
        if (vertices.size() < 2u) return {};
        const auto sources = SelectedEdgeSources(meshes, mesh, false);
        const auto &c = mesh.GetConnectivity();
        if (vertices.size() == 2u) {
            if (sources.contains(MeshEdgeUsers::Key(vertices[0], vertices[1]))) return {};
            return PrimitiveListTask(r, mesh, std::array{vertices});
        }
        const auto selected_faces = meshes.GetSelectedElements(mesh.GetStoreId(), Element::Face).Count();
        if (selected_faces) {
            if (selected_faces == 1u) return {};
            return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::DissolveFaces};
        }
        auto loops = BoundaryLoops(r, mesh, true);
        if (loops.empty()) {
            EdgeGraph neighbors;
            for (const auto &[key, h] : sources) {
                const auto a = uint32_t(key >> 32u), b = uint32_t(key);
                neighbors[a].push_back(b);
                neighbors[b].push_back(a);
            }
            if (std::ranges::all_of(neighbors, [](const auto &entry) { return entry.second.size() <= 2u; })) {
                std::vector<std::vector<uint32_t>> chains;
                VisitEdgeChains(neighbors, [&](const auto &chain, bool) { chains.push_back(chain); });
                // Complete an open chain through the single free selected point.
                if (chains.size() == 1u && neighbors.size() + 1u == vertices.size()) {
                    for (const auto v : vertices)
                        if (!neighbors.contains(v)) chains[0].push_back(v);
                }
                for (auto &chain : chains)
                    if (chain.size() >= 3u) loops.push_back(std::move(chain));
            }
            if (loops.empty()) {
                // Blender's vertex-cloud fallback orders points radially in their plane.
                // Read canonical positions directly; only handles and angular keys are stored on the host.
                vec3 center{};
                for (const auto v : vertices) center += mesh.GetPosition(he::VH{v}) / float(vertices.size());
                vec3 tangent{};
                for (const auto v : vertices) {
                    const auto delta = mesh.GetPosition(he::VH{v}) - center;
                    if (Dot(delta, delta) > Dot(tangent, tangent)) tangent = delta;
                }
                if (Dot(tangent, tangent) == 0.f) return {};
                tangent = Normalize(tangent);
                vec3 across{};
                for (const auto v : vertices) {
                    auto delta = mesh.GetPosition(he::VH{v}) - center;
                    delta -= tangent * Dot(delta, tangent);
                    if (Dot(delta, delta) > Dot(across, across)) across = delta;
                }
                if (Dot(across, across) < 1e-20f) return {};
                across = Normalize(across);
                std::vector<std::pair<float, uint32_t>> angles;
                for (const auto v : vertices) {
                    const auto delta = mesh.GetPosition(he::VH{v}) - center;
                    angles.emplace_back(std::atan2(Dot(delta, across), Dot(delta, tangent)), v);
                }
                std::ranges::sort(angles);
                auto &loop = loops.emplace_back();
                for (const auto &[angle, v] : angles) loop.push_back(v);
            }
        }
        std::erase_if(loops, [&](const auto &loop) {
            const std::unordered_set<uint32_t> members(loop.begin(), loop.end());
            const auto fan = c.VertexCorners[loop.front()];
            for (uint32_t i = 0u; i < fan.y; ++i) {
                const auto face = c.FaceOf(he::HH{c.FanItems[fan.x + i].x});
                if (face && mesh.GetValence(face) == loop.size() &&
                    std::ranges::all_of(mesh.fv_range(face), [&](auto v) { return members.contains(*v); })) return true;
            }
            return false;
        });
        if (loops.empty()) return {};
        for (auto &loop : loops) {
            int winding = 0;
            for (uint32_t i = 0u; i < loop.size(); ++i) {
                const auto a = loop[(i + loop.size() - 1u) % loop.size()], b = loop[i];
                const auto found = sources.find(MeshEdgeUsers::Key(a, b));
                if (found != sources.end() && c.FaceOf(he::HH{found->second}))
                    winding += *mesh.GetFromVertex(he::HH{found->second}) == a ? 1 : -1;
            }
            if (winding > 0) std::ranges::reverse(loop);
        }
        return PrimitiveListTask(r, mesh, loops, sources);
    });
}

// Subdivides the ring through each mesh's active edge.
void LoopCutSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, uint32_t cuts) {
    RunPerMesh(r, mesh_entities, [&](state::Entity e, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto edge = ActiveOrFirstSelectedEdge(r, e, mesh);
        if (!edge) return {};
        MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::Subdivide, .Param0 = float(std::max(cuts, 1u)), .Flags = TopologyFlagLoopCutSelect | TopologyFlagListSelects, .List = EdgeRing(mesh, *edge)};
        task.List.insert(task.List.begin(), uint32_t(task.List.size()));
        return task;
    });
}

// Extrudes the selection in `steps` layers, each moved through the step's transform once more than the last.
void ExtrudeSteps(state::Scene &r, std::span<const state::Entity> mesh_entities, uint32_t steps, const mat3 &rotation, vec3 translation, vec3 center) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) {
        return MeshTopologyTask{
            .SourceId = mesh.GetStoreId(),
            .Op = MeshTopologyOp::ExtrudeRegion,
            .Flags = TopologyFlagTransformCopies,
            .Steps = std::max(steps, 1u),
            .CopyRotation = rotation,
            // Rotating about a center is a rotation about the origin followed by the center's own displacement.
            .CopyTranslation = center - rotation * center + translation,
        };
    });
}

// Cuts each mesh along a plane in its own space, then deletes the faces on the cleared sides.
void BisectSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, vec3 point, vec3 normal, bool clear_inner, bool clear_outer) {
    const auto n = Normalize(normal);
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) {
        return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::Subdivide, .Param0 = 1.f, .Flags = TopologyFlagPlaneCuts | TopologyFlagLoopCutSelect, .PlaneNormal = n, .PlaneOffset = Dot(n, point)};
    });
    for (const bool inner : {true, false}) {
        if (inner ? !clear_inner : !clear_outer) continue;
        const auto side = inner ? n : -n;
        RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) {
            return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::DeleteFaces, .Flags = TopologyFlagPlaneSide, .PlaneNormal = side, .PlaneOffset = Dot(side, point)};
        });
    }
}

// Mirrors the kept side across the mesh origin and welds the vertices on the plane.
void SymmetrizeSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, uint8_t axis, bool negative) {
    vec3 normal{0.f};
    normal[axis % 3] = negative ? -1.f : 1.f;
    BisectSelected(r, mesh_entities, vec3{0.f}, normal, true, false);
    mat3 mirror{};
    for (int i = 0; i < 3; ++i) mirror[i][i] = i == axis % 3 ? -1.f : 1.f;
    // The whole kept side duplicates, and the plane's vertices weld afterward.
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) {
        return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::DuplicateGeometry, .Flags = TopologyFlagTransformCopies | TopologyFlagFlipCopies | TopologyFlagSelectAll, .CopyRotation = mirror};
    });
    RunOperator(r, mesh_entities, MeshTopologyOp::MergeByDistance, 1e-5f, 0.f, TopologyFlagSelectAll);
}

// Cuts every edge whose screen segment crosses the knife segment, at the crossing.
void KnifeSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, const ::selection::PrimaryEditInstanceMap &primaries, vec2 start, vec2 end, const RenderView &view) {
    const float aspect = view.Extent.y > 0.f ? view.Extent.x / view.Extent.y : 1.f;
    const auto view_projection = view.Camera.Projection(aspect) * view.Camera.View();
    RunPerMesh(r, mesh_entities, [&](state::Entity e, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto primary = primaries.find(e);
        if (primary == primaries.end()) return {};
        return MeshTopologyTask{
            .SourceId = mesh.GetStoreId(),
            .Op = MeshTopologyOp::Subdivide,
            .Param0 = 1.f,
            .Flags = TopologyFlagScreenCuts | TopologyFlagLoopCutSelect,
            .ScreenTransform = view_projection * ToMatrix(*WorldTransformOf(r, primary->second)),
            .Extent = view.Extent,
            .KnifeStart = start,
            .KnifeEnd = end,
        };
    });
}

void MergeSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, action::mesh::MergeMode mode, float distance) {
    using Mode = action::mesh::MergeMode;
    if (mode == Mode::Collapse) return RunOperator(r, mesh_entities, MeshTopologyOp::MergeCollapse);
    if (mode == Mode::ByDistance) return RunOperator(r, mesh_entities, MeshTopologyOp::MergeByDistance, distance);
    const auto &meshes = r.Context.get<const MeshStore>();
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto id = mesh.GetStoreId();
        const auto [first, last] = SelectedVertexSpan(meshes, id);
        if (first == InvalidOffset || first == last) return {};
        const auto &summary = meshes.GetSelectionSummary(id);
        const auto target = mode == Mode::Last ? last : first;
        const vec3 position = mode == Mode::Center ? summary.PositionSum / float(std::max(summary.SelectedVertexCount, 1u)) : mesh.GetPosition(he::VH{target});
        return MeshTopologyTask{.SourceId = id, .Op = MeshTopologyOp::MergeAtTarget, .TargetVertex = target, .TargetPosition = position};
    });
}
// Only selected face ranges and attribute pages participate; vertex fans are irrelevant.
void EditSelectedFaceAttributes(state::Scene &r, std::span<const state::Entity> targets, bool colors, uint32_t uv_set, uint32_t operation) {
    if (uv_set >= MeshStore::MaxUvSets) return;
    auto &meshes = r.Context.get<MeshStore>();
    auto &a = meshes.Arenas();
    mtl::ComputeChain chain{meshes.BufferContext()};
    struct Job {
        state::Entity Entity;
        uint32_t Id;
        ClosureSeed Seed;
        FaceAttributeEditPushConstants Pc;
    };
    std::vector<Job> jobs;
    uint64_t triangle_words = 0u, corners = 0u, face_count = 0u;
    for (const auto entity : targets) {
        const auto mesh = GetMesh(r, entity);
        const auto id = mesh.GetStoreId();
        const auto &record = meshes.Get(id);
        if (!(record.CornerAttributes & (colors ? MeshAttributeBit_Color0 : MeshAttributeBit_TexCoord0 << uv_set))) continue;
        const bool clear_tangents = !colors && (record.CornerAttributes & MeshAttributeBit_Tangent);
        std::vector<uint32_t> faces;
        uint32_t incidence = 0u;
        meshes.GetSelectedElements(id, Element::Face).ForEach([&](uint32_t f) {
            const auto range = a.FaceRanges.Get({f, 1u})[0];
            const Range handles{range.x, range.y - range.x};
            if (colors) a.CornerColors.CaptureHandles(handles);
            else a.CornerUvs[uv_set].CaptureHandles(handles);
            if (clear_tangents) a.CornerTangents.CaptureHandles(handles);
            faces.push_back(f);
            incidence += handles.Count;
        });
        if (faces.empty()) continue;
        ClosureSeed seed{.Work = SeedElementWorkHandles(chain.Scratch, a.FaceTriangles.Capacity(), faces), .Count = uint32_t(faces.size()), .Incidence = incidence};
        jobs.push_back({entity, id, seed, {
                                              .Connectivity = meshes.GetConnectivityRef(id),
                                              .Faces = seed.Work,
                                              .Attribute = colors ? a.CornerColors.Ref() : a.CornerUvs[uv_set].Ref(),
                                              .Tangents = a.CornerTangents.Ref(clear_tangents),
                                              .FaceCount = mesh.FaceCount(),
                                              .Count = seed.Count,
                                              .Operation = operation,
                                          }});
        const auto bound = std::min(incidence, a.Triangles.Set(record.TriangleData).BlockCount);
        triangle_words += 1u + ElementWorkWords(a.Triangles.Capacity(), bound) + SortElementWorkWords(WorkCapacity(a.Triangles.Capacity(), bound));
        corners += incidence;
        face_count += seed.Count;
    }
    if (jobs.empty()) return;
    chain.Scratch.ReserveAdditional(triangle_words);
    const auto &pipeline = GetMeshPipelines(r)[colors ? MeshPass::EditFaceColors : MeshPass::EditFaceUvs];
    chain.Concurrent([&] { for (const auto &job : jobs) chain.Threads(pipeline, job.Pc, job.Pc.Count); });
    std::vector<std::pair<state::Entity, FaceTriangles>> repairs;
    for (const auto &job : jobs) repairs.emplace_back(job.Entity, EncodeFaceTriangles(r, chain, job.Id, job.Seed));
    chain.Submit();
    for (auto &[entity, triangles] : repairs) triangles.Finish(chain);
    RepairFaceRender(r, chain, repairs);
    chain.Submit();
    profile::RecordCounter("AttributeEditFaces", face_count);
    profile::RecordCounter("AttributeEditCorners", corners);
    RequestRender(r, RenderRequest::Rebuild);
}

struct PositionEditOptions {
    uint32_t Axes{7u}, Flags{};
    vec3 Direction{}, Gradient{};
    const action::mesh::Warp *Warp{};
    const action::mesh::Bend *Bend{};
    const action::mesh::Randomize *Randomize{};
    const action::mesh::VertexSlide *Slide{};
    const action::mesh::EdgeSlide *EdgeSlide{};
    const action::mesh::SnapSymmetry *SnapSymmetry{};
    const action::mesh::CurveBetweenSelected *Curve{};
    const action::mesh::Circularize *Circle{};
};

bool ValidPositionPlane(const quat *orientation, vec3 center, float roll) {
    if (!orientation) return false;
    const auto q = *orientation;
    const float norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    return norm > 0.f && std::isfinite(norm) && std::isfinite(roll) &&
        std::isfinite(center.x) && std::isfinite(center.y) && std::isfinite(center.z);
}

// Position operators share sparse capture, iterative GPU writes, and publication.
void EditSelectedPositions(state::Scene &r, state::Entity viewport, std::span<const state::Entity> targets, PositionEditOp operation, float factor, uint32_t repeat, PositionEditOptions options = {}) {
    const auto &[axes, flags, direction, gradient, warp, bend, randomize, slide, edge_slide, symmetry, curve, circle] = options;
    const bool zero_moves = symmetry || warp || (slide && slide->Even && slide->Flipped) || (edge_slide && edge_slide->Even);
    if (targets.empty() || !axes || (!zero_moves && factor == 0.f) || !std::isfinite(factor)) return;
    const bool planar = operation == PositionEditOp::Planar;
    const bool ordered_chains = curve || circle;
    const bool flatten = operation == PositionEditOp::Copy && !ordered_chains;
    const bool flatten_view = flatten && (flags & PositionEditFlattenView);
    const bool relax = operation == PositionEditOp::RelaxEdgeLoops;
    const bool edge_chains = relax || operation == PositionEditOp::SpaceEvenly;
    const bool sphere = operation == PositionEditOp::ToSphere;
    const auto gather = edge_chains ? (relax ? MeshPass::RelaxEdgeLoopsGather : MeshPass::SpaceEvenlyGather) : MeshPass::PositionVerticesGather;
    const bool centered = sphere || operation == PositionEditOp::PushPull || operation == PositionEditOp::Shear;
    const bool transformed = centered || warp || bend || slide || edge_slide || flatten_view;
    if (sphere) factor = std::clamp(factor, 0.f, 1.f);
    else if (planar || operation == PositionEditOp::Smooth) factor = std::clamp(factor, -10.f, 10.f);
    if (!zero_moves && factor == 0.f) return;
    const auto &primaries = r.get<const EditPrimaries>(viewport).Transformable;
    const vec3 center = centered ? EditSelectionCenter(r, viewport) : vec3{};
    vec3 plane_x{}, plane_y{}, plane_center{};
    float bend_pivot = 0.f;
    if (warp || bend) {
        const auto rotation = Normalize(warp ? *warp->Orientation : *bend->Orientation);
        const auto right = rotation * vec3{1, 0, 0}, up = rotation * vec3{0, 1, 0};
        const float roll = warp ? warp->OffsetAngle : bend->OffsetAngle;
        const float c = std::cos(roll), s = std::sin(roll);
        plane_x = c * right - s * up;
        plane_y = s * right + c * up;
        plane_center = warp ? warp->Center : bend->Center;
        if (bend) {
            // Equivalent to Blender's shell_angle_to_dist, with a stable
            // small-angle denominator and saturation past a quarter turn.
            const float angle = std::abs(factor);
            const float shell = angle >= std::numbers::pi_v<float> * .5f ? 1.f : 1.f / std::sin(angle);
            bend_pivot = -std::copysign(1.f, factor) * bend->Radius * shell;
            if (!std::isfinite(bend_pivot)) return;
        }
    }
    auto &meshes = r.Context.get<MeshStore>();
    auto &pipelines = GetMeshPipelines(r);
    mtl::ComputeChain chain{meshes.BufferContext()};
    uint64_t selected_count = 0u;
    uint32_t reduction_blocks = 0u;
    if (!planar) {
        for (const auto entity : targets) {
            if (transformed && !primaries.contains(entity)) continue;
            const auto count = meshes.GetSelectedElements(GetMesh(r, entity).GetStoreId(), Element::Vertex).Count();
            selected_count += count;
            if (sphere || (warp && warp->AutoRange)) reduction_blocks += (count + 255u) / 256u;
        }
        // Symmetry can also move an unselected partner for each selected vertex.
        const uint32_t vertex_words = edge_slide || symmetry ? 10u : 4u;
        const uint32_t parameter_words = warp || bend ? 18u : randomize ? 3u :
            slide                                                       ? 8u :
            edge_slide                                                  ? 1u :
                                                                          0u;
        chain.Scratch.ReserveAdditional(vertex_words * selected_count + 4ull * reduction_blocks + 1u + uint64_t(parameter_words) * targets.size());
    }
    const bool statistics = sphere || (warp && warp->AutoRange);
    const auto partials = chain.Scratch.Allocate(2u * reduction_blocks);
    uint32_t partial_offset = partials.Offset;
    std::vector<VertexPositionEditPushConstants> jobs;
    struct PositionBatch {
        uint32_t Index;
        std::vector<Range> Batches;
    };
    std::vector<PositionBatch> batches;
    std::vector<std::vector<Range>> ranges;
    std::vector<state::Entity> entities;
    for (const auto entity : targets) {
        if (transformed && !primaries.contains(entity)) continue;
        const auto mesh = GetMesh(r, entity);
        const auto id = mesh.GetStoreId();
        VertexPositionEditPushConstants pc{
            .Connectivity = meshes.GetConnectivityRef(id),
            .VertexSlot = meshes.Slots().Vertices,
            .CornerSlot = meshes.Arenas().FaceCorners.Buffer.Slot,
            .FaceCount = mesh.FaceCount(),
            .Axes = axes,
            .Factor = factor,
            .FaceNormalSlot = meshes.Arenas().BaseFaceNormals.Buffer.Slot,
            .VertexNormalSlot = meshes.Arenas().BaseVertexNormals.Buffer.Slot,
            .FaceSelectionSlot = meshes.GetSelectionSlot(Element::Face),
            .Flags = flags,
            .Operation = operation,
        };
        PositionBatch batch{.Index = uint32_t(jobs.size())};
        const auto world = transformed ? *WorldTransformOf(r, primaries.at(entity)) : Transform{};
        const auto inverse = Conjugate(world.R);
        const auto local_direction = [&](vec3 direction) {
            auto local = inverse * direction;
            for (uint32_t axis = 0u; axis < 3u; ++axis) local[axis] = world.S[axis] != 0.f ? local[axis] / world.S[axis] : 0.f;
            return local;
        };
        if (flatten_view) {
            pc.Direction = local_direction(direction);
            const float length = Length(pc.Direction);
            if (!(length > 0.f) || !std::isfinite(length)) continue;
            pc.Direction /= length;
        }
        if (centered) {
            pc.Center = local_direction(center - world.P);
            pc.Direction = local_direction(direction);
            pc.Gradient = (inverse * gradient) * world.S;
        }
        if (warp || bend) {
            const PositionPlane plane{(inverse * plane_x) * world.S, (inverse * plane_y) * world.S, local_direction(plane_x), local_direction(plane_y), {Dot(world.P - plane_center, plane_x), Dot(world.P - plane_center, plane_y)}};
            if (warp) pc.Parameters = chain.Upload(as_bytes(WarpParameters{plane, std::min(warp->Min, warp->Max), std::max(warp->Min, warp->Max)}));
            else pc.Parameters = chain.Upload(as_bytes(BendParameters{plane, bend->Radius, bend_pivot}));
        }
        if (randomize) pc.Parameters = chain.Upload(as_bytes(RandomizeParameters{std::clamp(randomize->Uniform, 0.f, 1.f), std::clamp(randomize->Normal, 0.f, 1.f), randomize->Seed}));
        if (slide || edge_slide) {
            const auto selected = meshes.GetSelectedElements(id, Element::Vertex);
            uint32_t reference = selected.First().value_or(InvalidOffset);
            if (const auto *active = r.try_get<const MeshActiveElement>(entity);
                active && r.get<const EditMode>(viewport).Value == Element::Vertex &&
                meshes.IsLiveElement(id, Element::Vertex, active->Handle) && selected.Contains(active->Handle)) reference = active->Handle;
            if (edge_slide) {
                const auto directions = PlanEdgeSlide(meshes, mesh, inverse * Normalize(edge_slide->Direction), world.S, reference);
                if (directions.empty()) continue;
                pc.Parameters = chain.Upload(as_bytes(directions));
                if (edge_slide->Even) {
                    uint32_t rank = 0u, reference_rank = 0u;
                    selected.ForEach([&](uint32_t v) { if (v==reference) reference_rank=rank; ++rank; });
                    const auto &ref = directions[reference_rank];
                    pc.ReductionResult = chain.Upload(as_bytes(Length(ref.Positive - ref.Negative)));
                }
            } else {
                const VertexSlideParameters parameters{inverse * Normalize(slide->Direction), world.S, reference};
                pc.Parameters = chain.Upload(as_bytes(parameters));
                if (slide->Even) pc.ReductionResult = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(1u).Offset};
            }
        }
        std::vector<uint32_t> vertices;
        if (flatten) {
            auto plan = PlanFlatten(meshes, mesh);
            if (plan.Vertices.empty()) continue;
            chain.Scratch.ReserveAdditional(plan.Words.size() + plan.Groups.size() + 4ull * plan.Vertices.size());
            // Flatten's parameters hold packed groups; Planes indexes their offsets.
            pc.Parameters = chain.Upload(as_bytes(plan.Words));
            pc.Planes = chain.Upload(as_bytes(plan.Groups));
            pc.PlaneCount = uint32_t(plan.Groups.size());
            batch.Batches = std::move(plan.Batches);
            vertices = std::move(plan.Vertices);
        } else if (planar) {
            std::vector<uint32_t> faces;
            meshes.GetSelectedElements(id, Element::Face).ForEach([&](uint32_t f) {
                if (mesh.GetValence(he::FH{f}) > 3u) faces.push_back(f);
            });
            if (faces.empty()) continue;
            // The existing face seed gathers only connectivity handles on the host.
            // Its vertex list also gives history the exact pages to capture.
            auto seed = ListSeed(r, chain, id, Element::Face, faces);
            pc.Faces = seed.Work;
            pc.PlaneCount = seed.Count;
            pc.Planes = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(4u * seed.Count).Offset};
            vertices = std::move(seed.Vertices);
        } else if (edge_chains || ordered_chains) {
            auto plan = circle ? PlanCircularize(meshes, mesh) : curve ? PlanCurveBetweenSelected(meshes, mesh, curve->Extend) :
                                                                         PlanSelectedEdgeChains(meshes, mesh, relax);
            if (plan.Outputs.empty()) continue;
            const uint32_t stride = circle ? 0u : curve ? 7u :
                relax                                   ? 10u :
                                                          6u;
            const uint32_t extra = circle ? 0u : curve ? 2u :
                                                         1u;
            chain.Scratch.ReserveAdditional(uint64_t(stride + 1u) * plan.Inputs.size() + (sizeof(EdgeChain) / sizeof(uint32_t) + extra) * plan.Chains.size() + 4ull * plan.Outputs.size() + plan.Phases.size());
            const auto inputs = chain.Scratch.Allocate(std::span<const uint32_t>{plan.Inputs});
            const auto phases = chain.Scratch.Allocate(std::span<const uint32_t>{plan.Phases});
            const auto work = chain.Scratch.Allocate(stride * uint32_t(plan.Inputs.size()) + extra * uint32_t(plan.Chains.size()));
            uint32_t work_offset = work.Offset;
            for (uint32_t i = 0u; i < plan.Chains.size(); ++i) {
                auto &descriptor = plan.Chains[i];
                descriptor.WorkOffset = work_offset;
                work_offset += stride * descriptor.Count + extra;
                descriptor.InputOffset += inputs.Offset;
                if (relax || curve) descriptor.PhaseOffset += phases.Offset;
            }
            pc.Parameters = chain.Upload(as_bytes(plan.Chains));
            pc.ChainCount = uint32_t(plan.Chains.size());
            if (ordered_chains) batch.Batches = std::move(plan.Batches);
            if (circle) pc.Direction = {circle->Radius, circle->Angle, 0.f};
            vertices = std::move(plan.Outputs);
        } else if (symmetry) {
            const auto plan = PlanSymmetrySnap(meshes, mesh, uint32_t(symmetry->Axis), symmetry->Threshold, symmetry->Center);
            if (plan.empty()) continue;
            std::vector<uint32_t> partners;
            for (const auto [v, partner] : plan) {
                vertices.push_back(v);
                partners.push_back(partner);
            }
            pc.Parameters = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(std::span<const uint32_t>{partners}).Offset};
        } else {
            meshes.GetSelectedElements(id, Element::Vertex).ForEach([&](uint32_t v) { vertices.push_back(v); });
        }
        if (vertices.empty()) continue;
        pc.Handles = chain.Upload(as_bytes(vertices));
        pc.Count = uint32_t(vertices.size());
        // Fitting jobs keep their own output order; only history capture needs sorted handles.
        if (edge_chains || ordered_chains) std::ranges::sort(vertices);
        std::vector<Range> runs;
        ForEachIndexRun(vertices, [&](size_t first, size_t count) { runs.push_back({vertices[first], uint32_t(count)}); });
        if (statistics) {
            pc.ReductionBlocks = {chain.Scratch.Buffer.Slot, partial_offset};
            partial_offset += 2u * ((pc.Count + 255u) / 256u);
        }
        pc.Positions = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(3u * pc.Count).Offset};
        meshes.Arenas().Vertices.Buffer.CaptureWriteRanges(runs, sizeof(Vertex));
        entities.push_back(entity);
        ranges.push_back(std::move(runs));
        jobs.push_back(pc);
        if (!batch.Batches.empty()) batches.push_back(std::move(batch));
        ReleaseMeshEditWork(r, entity);
    }
    if (jobs.empty()) return;
    uint64_t vertex_count = 0u, plane_count = 0u;
    for (const auto &pc : jobs) {
        vertex_count += pc.Count;
        plane_count += pc.PlaneCount;
    }
    profile::RecordCounter("PositionEditVertices", vertex_count);
    profile::RecordCounter("PositionEditPlanes", plane_count);
    uint64_t curve_chain_count = 0u, curve_batch_count = 0u;
    if (ordered_chains)
        for (const auto &job : batches) {
            curve_chain_count += jobs[job.Index].ChainCount;
            curve_batch_count += job.Batches.size();
        }
    profile::RecordCounter("CurveEditChains", curve_chain_count);
    profile::RecordCounter("CurveEditBatches", curve_batch_count);
    if (statistics) {
        chain.Concurrent([&] {
            for (const auto &pc : jobs) chain.Groups(pipelines[MeshPass::PositionStatisticsGather], pc, (pc.Count + 255u) / 256u);
        });
        const auto reduce = [&](SlotOffset input, uint32_t count) {
            // Sphere needs a final normalization even for one partial; bounds
            // already contain their result when a mesh fits in one group.
            if (sphere || count > 1u) {
                do {
                    const uint32_t next = (count + 255u) / 256u;
                    const SlotOffset output{chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(2u * next).Offset};
                    const PositionReducePushConstants pc{.Input = input, .Output = output, .Count = count, .Scale = sphere && next == 1u ? 1.f / float(selected_count) : 1.f, .Bounds = !sphere};
                    chain.Groups(pipelines[MeshPass::PositionStatisticsReduce], pc, next);
                    input = output;
                    count = next;
                } while (count > 1u);
            }
            return input;
        };
        if (sphere) {
            const auto result = reduce({chain.Scratch.Buffer.Slot, partials.Offset}, reduction_blocks);
            for (auto &pc : jobs) pc.ReductionResult = result;
        } else
            for (auto &pc : jobs) pc.ReductionResult = reduce(pc.ReductionBlocks, (pc.Count + 255u) / 256u);
        profile::RecordCounter(sphere ? "PositionRadiusBlocks" : "PositionWarpBoundsBlocks", reduction_blocks);
    }
    if (planar) chain.Concurrent([&] {
        for (const auto &pc : jobs) chain.Threads(pipelines[MeshPass::PlanarFacePlanes], pc, pc.PlaneCount);
    });
    if (slide && slide->Even) chain.Concurrent([&] {
        for (const auto &pc : jobs) chain.Threads(pipelines[MeshPass::VertexSlideReference], pc, 1u);
    });
    for (uint32_t step = 0u; step < repeat; ++step) {
        chain.Concurrent([&] {
            for (const auto &pc : jobs) chain.Threads(pipelines[gather], pc, edge_chains ? pc.ChainCount : pc.Count);
        });
        uint32_t depths = 0u;
        for (const auto &job : batches) depths = std::max(depths, uint32_t(job.Batches.size()));
        for (uint32_t depth = 0u; depth < depths; ++depth) chain.Concurrent([&] {
            for (const auto &job : batches)
                if (depth < job.Batches.size()) {
                    const auto batch = job.Batches[depth];
                    auto pc = jobs[job.Index];
                    if (flatten) {
                        pc.Planes.Offset += batch.Offset;
                        chain.Groups(pipelines[MeshPass::FlattenGroups], pc, batch.Count);
                    } else {
                        pc.Parameters.Offset += batch.Offset * uint32_t(sizeof(EdgeChain) / sizeof(uint32_t));
                        pc.ChainCount = batch.Count;
                        chain.Threads(pipelines[circle ? MeshPass::Circularize : MeshPass::CurveBetweenSelected], pc, pc.ChainCount);
                    }
                }
        });
        chain.Concurrent([&] {
            for (const auto &pc : jobs) chain.Threads(pipelines[MeshPass::WriteEditedPositions], pc, pc.Count);
        });
    }
    std::vector<MeshVertexChanges> changes;
    for (uint32_t i = 0u; i < entities.size(); ++i) changes.push_back({entities[i], ranges[i]});
    RefreshEditedPositions(r, chain, changes);
    chain.Submit();
    for (const auto entity : PublishEditedPositions(r, chain, entities)) {
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
            [&](const Merge &a) { MergeSelected(r, targets, a.Mode, std::max(a.Distance, 0.f)); },
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
                if (uint32_t(a.Axis) > 2u || !std::isfinite(a.Threshold) || a.Threshold <= 0.f) return;
                EditSelectedPositions(r, viewport, targets, PositionEditOp::SnapSymmetry, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = 1u << uint32_t(a.Axis), .Flags = a.Negative ? PositionEditSymmetryNegative : 0u, .SnapSymmetry = &a});
            },
            [&](const Decimate &a) { RunPerMesh(r, targets, [&](state::Entity, const Mesh &mesh) { return DecimateTask(r.Context.get<const MeshStore>(), mesh, a.Ratio); }); },
            [&](const Unsubdivide &a) {
                for (uint32_t i = 0u; i < std::clamp(a.Iterations, 1u, 1000u); ++i) {
                    bool changed = false;
                    RunPerMesh(r, targets, [&](state::Entity, const Mesh &mesh) {
                        auto task = UnsubdivideTask(r.Context.get<const MeshStore>(), mesh);
                        changed |= task.has_value();
                        return task;
                    });
                    if (!changed) break;
                }
            },
            [&](Triangulate) { RunOperator(r, targets, MeshTopologyOp::Triangulate); },
            [&](const BeautifyFaces &a) {
                RunPerMesh(r, targets, [&](state::Entity, const Mesh &mesh) {
                    return BeautifyFaceTask(r.Context.get<const MeshStore>(), mesh, a.Method == BeautifyMethod::Angle);
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
                if (!ValidPositionPlane(a.Orientation.get(), a.Center, a.OffsetAngle)) return;
                if (!a.AutoRange && (!std::isfinite(a.Min) || !std::isfinite(a.Max) || a.Min == a.Max)) return;
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Warp, a.Angle, 1u, {.Flags = a.AutoRange ? PositionEditWarpAutoRange : 0u, .Warp = &a});
            },
            [&](const Bend &a) {
                if (!ValidPositionPlane(a.Orientation.get(), a.Center, a.OffsetAngle) || a.Radius == 0.f || !std::isfinite(a.Radius)) return;
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Bend, a.Angle, 1u, {.Flags = a.Clamp ? PositionEditBendClamp : 0u, .Bend = &a});
            },
            [&](const Randomize &a) {
                if (!std::isfinite(a.Uniform) || !std::isfinite(a.Normal)) return;
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Randomize, a.Amount, 1u, {.Randomize = &a});
            },
            [&](const VertexSlide &a) {
                const float length = Length(a.Direction);
                if (!(length > 0.f) || !std::isfinite(length)) return;
                EditSelectedPositions(r, viewport, targets, PositionEditOp::VertexSlide, a.Clamp ? std::clamp(a.Factor, 0.f, 1.f) : a.Factor, 1u, {.Flags = (a.Even ? PositionEditSlideEven : 0u) | (a.Flipped ? PositionEditSlideFlipped : 0u), .Slide = &a});
            },
            [&](const EdgeSlide &a) {
                const float length = Length(a.Direction);
                if (!(length > 0.f) || !std::isfinite(length)) return;
                EditSelectedPositions(r, viewport, targets, PositionEditOp::EdgeSlide, a.Clamp ? std::clamp(a.Factor, -1.f, 1.f) : a.Factor, 1u, {.Flags = (a.Even ? PositionEditSlideEven : 0u) | (a.Flipped ? PositionEditSlideFlipped : 0u) | (a.Clamp ? 0u : PositionEditSlideUnclamped), .EdgeSlide = &a});
            },
            [&](const Circularize &a) {
                if (!std::isfinite(a.Radius) || !std::isfinite(a.Angle)) return;
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Copy, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = uint32_t(a.X) | (uint32_t(a.Y) << 1u) | (uint32_t(a.Z) << 2u), .Flags = (a.Regular ? PositionEditCurveRegular : 0u) | (a.Method == CircleFit::Contract ? PositionEditCircleContract : 0u), .Circle = &a});
            },
            [&](const CurveBetweenSelected &a) {
                EditSelectedPositions(r, viewport, targets, PositionEditOp::Copy, std::clamp(a.Factor, 0.f, 1.f), 1u, {.Axes = uint32_t(a.X) | (uint32_t(a.Y) << 1u) | (uint32_t(a.Z) << 2u), .Flags = (a.Interpolation == EdgeLoopInterpolation::Cubic ? PositionEditCurveCubic : 0u) | (a.Regular ? PositionEditCurveRegular : 0u) | (a.Elevation == CurveElevation::Raise ? PositionEditCurveRaise : a.Elevation == CurveElevation::Lower ? PositionEditCurveLower :
                                                                                                                                                                                                                                                                                                                                                                                                                                               0u),
                                                                                                                       .Curve = &a});
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
                auto faces = RecalculateFaceFlips(r, ids, a.Inside);
                uint32_t i = 0u;
                RunPerMesh(r, targets, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
                    auto &list = faces[i++];
                    if (list.empty()) return {};
                    list.insert(list.begin(), uint32_t(list.size()));
                    return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::FlipNormals, .List = std::move(list)};
                });
            },
            [&](EdgeSplit) { RunOperator(r, targets, MeshTopologyOp::EdgeSplit); },
            [&](const Inset &a) { RunOperator(r, targets, a.Individual ? MeshTopologyOp::InsetIndividual : MeshTopologyOp::InsetRegion, std::max(a.Thickness, 0.f), a.Depth, a.Even ? 1u : 0u); },
            [&](Fill) { FillSelected(r, targets); },
            [&](const LoopCut &a) { LoopCutSelected(r, targets, a.Cuts); },
            [&](const Spin &a) {
                if (Dot(a.Axis, a.Axis) <= 0.f) return;
                const auto axis = Normalize(a.Axis);
                ExtrudeSteps(r, targets, a.Steps, ToMat3(AngleAxis(a.Angle / float(std::max(a.Steps, 1u)), axis)), axis * (a.Offset / float(std::max(a.Steps, 1u))), a.Center);
            },
            [&](const ExtrudeRepeat &a) { ExtrudeSteps(r, targets, a.Steps, mat3{1.f}, a.Offset, vec3{0.f}); },
            [&](const Bisect &a) {
                if (Dot(a.Normal, a.Normal) <= 0.f) return;
                BisectSelected(r, targets, a.Point, a.Normal, a.ClearInner, a.ClearOuter);
            },
            [&](const Symmetrize &a) { SymmetrizeSelected(r, targets, uint8_t(a.Axis), a.Negative); },
            [&](const Solidify &a) { RunOperator(r, targets, MeshTopologyOp::Solidify, a.Thickness); },
            [&](const Wireframe &a) {
                RunOperator(r, targets, MeshTopologyOp::Wireframe, std::max(a.Thickness, 0.f), std::clamp(a.Offset, -1.f, 1.f), (a.Even ? TopologyFlagEvenOffset : 0u) | (a.Boundary ? TopologyFlagWireBoundary : 0u) | (a.Relative ? TopologyFlagWireRelative : 0u) | (a.Replace ? TopologyFlagWireReplace : 0u));
            },
            [&](ConnectVertices) { RunOperator(r, targets, MeshTopologyOp::ConnectVertices); },
            [&](const Knife &a) { KnifeSelected(r, targets, r.get<const EditPrimaries>(viewport).All, a.Start, a.End, *a.View); },
            [&](BridgeEdgeLoops) { BridgeSelected(r, targets); },
            [&](const GridFill &a) { GridFillSelected(r, targets, a.Span); },
            [&](const FillHoles &a) { FillHolesSelected(r, targets, a.Sides); },
            [&](ConvexHull) { ConvexHullSelected(r, targets); },
            [&](EdgeRotate) { EdgeRotateSelected(r, targets); },
            [&](const Bevel &a) {
                RunPerMesh(r, targets, [&](state::Entity, const Mesh &mesh) {
                    return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = a.Vertices ? MeshTopologyOp::BevelVertices : MeshTopologyOp::BevelEdges, .Param0 = std::max(a.Width, 0.f), .Steps = std::max(a.Segments, 1u)};
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
