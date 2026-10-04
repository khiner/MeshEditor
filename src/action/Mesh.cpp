#include "Profile.h"
#include "ProcessEvents.h"
#include "action/Mesh.h"
#include "action/InsetPreview.h"

#include "gpu/InsetPreviewPushConstants.h"
#include "gpu/InsetVertexBasis.h"
#include "gpu/MeshTopologyOp.h"
#include "gpu/Vertex.h"

#include "TransformMath.h"
#include "Variant.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/MeshTopology.h"
#include "mesh/MeshTopologyEdit.h"
#include "mesh/MeshPipelines.h"
#include "mesh/ScratchChunks.h"
#include "metal/Dispatch.h"
#include "render/MeshTopologyRepair.h"
#include "render/MeshletBuildGpu.h"
#include "render/MeshletBoundsRefit.h"
#include "render/GpuBuffers.h"
#include "render/SceneUpdates.h"
#include "render/GpuSceneState.h"
#include "render/ElementWorkOps.h"
#include "viewport/ViewportRenderGpu.h"
#include "mesh/PrimitiveType.h"
#include "numeric/MatrixMath.h"
#include "numeric/QuaternionMath.h"
#include "object/ObjectOps.h"
#include "project/Project.h"
#include "render/GpuBufferOps.h"
#include "render/MeshBuffers.h"
#include "scene/Entity.h"
#include "scene/WorldTransform.h"
#include "selection/SelectionState.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionComponents.h"
#include "state/Scene.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"

#include <format>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {
void UpdatePoseMembership(state::Scene &r,const MeshTopologyEdit &edit) {
    auto &buffers=r.Context.get<GpuBuffers>();
    const auto &meshes=r.Context.get<const MeshStore>();
    const auto &a=meshes.Arenas();
    const auto &record=meshes.Get(edit.StoreId);
    std::vector<uint32_t> vertices,faces,normal_payloads=edit.OldNormalPayloadBlocks;
    const auto &storage=edit.Chain.Scratch;
    const auto gather=[&](std::vector<uint32_t> &blocks,ElementWork work) {
        ForEachWorkBlock(storage,work,[&](uint32_t block,auto) { blocks.push_back(block); });
    };
    if (edit.Repair) {
        gather(vertices,edit.Repair->Elements[0]);
        gather(faces,edit.Repair->Elements[2]);
        ForEachWorkBlock(storage,edit.Repair->Elements[1],[&](uint32_t block,auto) {
            const auto payload=a.NormalSectors.PayloadBlock(block);
            if (payload) normal_payloads.push_back(payload-1u);
        });
    }
    if (edit.Output) {
        gather(vertices,edit.Output->Retired[0]);
        gather(faces,edit.Output->Retired[1]);
    }
    const auto unique=[](std::vector<uint32_t> &blocks) {
        std::ranges::sort(blocks);
        blocks.erase(std::unique(blocks.begin(),blocks.end()),blocks.end());
    };
    unique(vertices); unique(faces); unique(normal_payloads);
    const auto vertex_owner=record.Vertices.Index;
    const auto vertex_members=a.Vertices.Blocks.Buffer.GetSpan<MeshElementBlock>();
    const auto has_vertex=[&](uint32_t block) {
        return block<vertex_members.size() && vertex_members[block].Owner==vertex_owner && vertex_members[block].Count;
    };
    const auto vertex_revision=record.Vertices ? a.Vertices.Set(record.Vertices).Revision : 0u;
    buffers.VertexBounds.UpdateBlocks(edit.StoreId,vertex_revision,vertices,has_vertex);
    buffers.PosedPositions.UpdateBlocks(edit.StoreId,vertex_revision,vertices,has_vertex);
    buffers.PosedMorphNormalDeltas.UpdateBlocks(edit.StoreId,vertex_revision,vertices,has_vertex);
    buffers.PosedVertexNormals.UpdateBlocks(edit.StoreId,vertex_revision,vertices,has_vertex);
    const auto face_owner=record.FaceData.Index;
    const auto face_members=a.FaceTriangles.Blocks.Buffer.GetSpan<MeshElementBlock>();
    const auto has_face=[&](uint32_t block) {
        return block<face_members.size() && face_members[block].Owner==face_owner && face_members[block].Count;
    };
    const auto face_revision=record.FaceData ? a.FaceTriangles.Set(record.FaceData).Revision : 0u;
    buffers.PosedFaceNormals.UpdateBlocks(edit.StoreId,face_revision,faces,has_face);
    const auto normal_owners=a.NormalSectors.Owners.Buffer.GetSpan<uint32_t>();
    const auto has_normal=[&](uint32_t payload) {
        if (payload>=normal_owners.size() || !normal_owners[payload]) return false;
        const auto block=normal_owners[payload]-1u;
        return a.FaceCorners.Blocks.Get({block,1u})[0].Owner==record.FaceCorners.Index &&
            a.NormalSectors.PayloadBlock(block)==payload+1u;
    };
    buffers.PosedSectors.UpdateBlocks(edit.StoreId,meshes.GetDerived(edit.StoreId).NormalRevision,normal_payloads,has_normal);
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

// Whether a published in-place edit repairs its record's triangle render, which needs a triangle render owner and faces left after the edit.
// The unretired source faces still count, so the edit leaves faces when more than its retired faces are live.
bool RepairsTriangleRender(const state::Scene &r, state::Entity entity, const MeshTopologyEdit &edit) {
    const auto *owner=TryMeshBuffers(r,entity);
    return owner && owner->StoreId==edit.SourceId && owner->RenderTopology==0u &&
        Mesh{r.Context.get<const MeshStore>(),edit.SourceId}.FaceCount()>edit.Output->RetiredCounts[1];
}

// Publishes a finished in-place edit's render, pose and selection summary state.
// A record whose live elements now draw as another topology rebuilds through the settle pass's meshlet batch, since topologies never mix.
// A record without a render owner is a newly created canonical mesh.
// A point or line record's moved elements join `element_repairs`, which repair together.
void FinishTopologyEdit(state::Scene &r, state::Entity entity, const MeshTopologyTask &task, MeshTopologyEdit &edit, std::vector<ElementMeshletRepair> &element_repairs) {
    auto &buffers=r.Context.get<GpuBuffers>();
    auto &meshes=r.Context.get<MeshStore>();
    const auto *render_owner=TryMeshBuffers(r,entity);
    const bool ready=render_owner && render_owner->StoreId==task.SourceId;
    UpdatePoseMembership(r,edit);
    bool repaired=false;
    if (ready) {
        auto &owner=buffers.MeshOf(task.SourceId);
        const Mesh mesh{meshes,task.SourceId};
        owner.Vertices.Count=mesh.VertexCount();
        repaired=mesh.PrimitiveTopology()==owner.RenderTopology;
        // The drawn topology changes, so the instance flags that depend on it are rederived.
        if (!repaired) RequestRender(r,RenderRequest::Rebuild);
        else if (owner.RenderTopology!=0u && edit.Repair) {
            // A point record's repaired vertices, or a line record's retired edges and the edges of its repaired corners, move between its clusters.
            const auto &storage=edit.Chain.Scratch;
            auto &affected=element_repairs.emplace_back(ElementMeshletRepair{.StoreId=task.SourceId}).Elements;
            if (owner.RenderTopology==2u) ForEachWorkElement(storage,edit.Repair->Elements[0],[&](uint32_t v) { affected.push_back(v); });
            else {
                const auto edges=meshes.Arenas().HalfedgeEdges.Buffer.GetSpan<uint32_t>();
                ForEachWorkElement(storage,edit.RetiredEdges,[&](uint32_t e) { affected.push_back(e); });
                ForEachWorkElement(storage,edit.Repair->Elements[1],[&](uint32_t h) { affected.push_back(edges[h]); });
            }
        }
    }
    if (ready && edit.InsetBasis.Count) {
        // The staged edit captured its basis into the session's preview cache.
        auto &session=project::Session(r);
        const auto output=edit.Chain.Scratch.Get(edit.Output->Vertices);
        std::vector<uint32_t> handles(output.begin(),output.end());
        std::ranges::sort(handles);
        handles.erase(std::unique(handles.begin(),handles.end()),handles.end());
        std::vector<Range> ranges;
        ForEachIndexRun(handles, [&](size_t first, size_t count) { ranges.push_back({handles[first],uint32_t(count)}); });
        session.InsetPreview->Entries.push_back({entity,task.SourceId,task.Op,task.Flags,edit.InsetBasis,std::move(ranges)});
        if (session.Previewing) {
            auto &pipelines=GetMeshPipelines(r);
            (void)pipelines[MeshPass::InsetPreviewPositions].State();
            (void)pipelines[MeshPass::MeshletBoundsRefit].State();
        }
    }
    buffers.RefreshMeshBinding(r,task.SourceId);
    r.remove<PrimitiveShape,MeshActiveElement>(entity);
    r.emplace_or_replace<MeshGeometryDirty>(entity,EditSelectionAfter::Keep,repaired);
    r.Context.get<GpuSceneState>().EditSelectionDirty=true;
}

// Every topology action prepares selection and edit work inside one history
// transaction, including actions that publish more than one mesh output.
void RunTopologyAction(state::Scene &r, std::span<const state::Entity> mesh_entities, auto &&run) {
    auto &history=project::Session(r).History;
    auto before=history.Pin();
    bool changed=false;
    try {
        for (const auto entity:mesh_entities) ReleaseMeshEditWork(r,entity);
        changed=run();
    } catch (...) {
        history.Restore(before); history.Release(before); throw;
    }
    history.Release(before);
    if (changed) r.Context.get<GpuBuffers>().PreludeStale=true;
}

// Emplaces a published copied output's render buffers as a mesh gaining its first faces, and returns the build of its finest meshlets.
// The edit derived its normals and corner classes, so the output skips the new-mesh sync.
MeshletBuildSource CopiedOutputBuild(state::Scene &r, const MeshTopologyEdit &edit) {
    auto &buffers=r.Context.get<GpuBuffers>();
    const auto &meshes=r.Context.get<const MeshStore>();
    const auto &vertices=meshes.Arenas().Vertices;
    const auto set=meshes.Get(edit.StoreId).Vertices;
    auto &face=buffers.EmplaceMesh(edit.StoreId,{{vertices.First(set),vertices.Count(set)},vertices.Buffer.Slot});
    AssignFaceIndices(meshes,Mesh{meshes,edit.StoreId},face);
    return {.Destination=&face,.Mesh=BuildMeshRecord(buffers,face,meshes,edit.StoreId,true,false),
        .StoreId=edit.StoreId,.Topology=0u,.ElementCount=edit.AddedTriangleCount,.Elements=edit.AddedTriangles};
}

// Every mesh's edit shares the action's chain, construction's submits, each publication submit, one render repair and one selection update.
// Render repairs read the reserved source identities, so every repair precedes the edits' finish.
// A KeepSelectedFaces edit copies into a new canonical record, whose meshlet build rides the repair.
// Returns each task's output record, the source for an in-place edit, or none when the edit changed nothing.
std::vector<std::optional<uint32_t>> EditTopology(state::Scene &r, std::span<const state::Entity> mesh_entities, std::span<const MeshTopologyTask> tasks) {
    if (tasks.size()!=mesh_entities.size()) throw std::invalid_argument("Topology task and entity counts differ.");
    auto &session=project::Session(r);
    auto &meshes=r.Context.get<MeshStore>();
    // A staged inset captures every mesh's basis into the session's preview cache.
    const bool insets=session.Previewing && std::ranges::any_of(tasks,[](const auto &task) {
        return task.Op==MeshTopologyOp::InsetRegion || task.Op==MeshTopologyOp::InsetIndividual;
    });
    if (insets && !session.InsetPreview) session.InsetPreview=std::make_unique<action::mesh::InsetPreviewCache>(meshes.BufferContext());
    mtl::ComputeChain chain{meshes.BufferContext(),TopologyScratchWords};
    auto edits=MeshTopologyEdit::Construct(r,chain,tasks,insets ? &session.InsetPreview->Basis : nullptr);
    MeshTopologyEdit::PublishAll(r,edits);
    std::vector<std::optional<uint32_t>> outputs(edits.size());
    std::vector<MeshletBuildSource> copies;
    std::vector<std::pair<state::Entity,const MeshTopologyEdit *>> repairs;
    std::vector<MeshTopologyEdit *> finished;
    for (uint32_t i=0u;i<edits.size();++i) {
        auto &edit=edits[i];
        if (!edit.Output) continue;
        outputs[i]=edit.StoreId;
        finished.push_back(&edit);
        if (edit.StoreId!=edit.SourceId) copies.push_back(CopiedOutputBuild(r,edit));
        else if (RepairsTriangleRender(r,mesh_entities[i],edit)) repairs.emplace_back(mesh_entities[i],&edit);
    }
    RepairTopologyRender(r,chain,repairs,copies);
    MeshTopologyEdit::FinishAll(r,finished);
    std::vector<state::Entity> in_place;
    std::vector<ElementMeshletRepair> element_repairs;
    for (uint32_t i=0u;i<edits.size();++i) {
        if (outputs[i]!=edits[i].SourceId) continue;
        FinishTopologyEdit(r,mesh_entities[i],tasks[i],edits[i],element_repairs);
        in_place.push_back(mesh_entities[i]);
    }
    RepairElementMeshlets(r,chain,element_repairs);
    chain.Submit();
    RefreshElementSelectionSummaries(r,in_place);
    return outputs;
}

// Runs the tasks as one topology action.
// Spans of tasks edit one after another, each under the scratch budget, so a batch of large meshes holds one span's scratch at a time.
void RunTasks(state::Scene &r, std::span<const state::Entity> mesh_entities, std::span<const MeshTopologyTask> tasks) {
    const profile::CpuScope scope{"TopologyAction"};
    if (tasks.empty()) return;
    const auto &meshes=r.Context.get<const MeshStore>();
    const auto split=ChunkByScratch(uint32_t(tasks.size()),ScratchWordBudget,[&](uint32_t i) { return TopologyScratchBound(meshes,tasks[i]); });
    RunTopologyAction(r,mesh_entities,[&] {
        bool changed=false;
        for (const auto span:split.Chunks) {
            const auto outputs=EditTopology(r,mesh_entities.subspan(span.Offset,span.Count),tasks.subspan(span.Offset,span.Count));
            changed|=std::ranges::any_of(outputs,[](const auto &output) { return output.has_value(); });
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
    return {first.value_or(InvalidOffset),last.value_or(InvalidOffset)};
}

// Moves the selected faces of each mesh into a new mesh object placed over the mesh's primary instance.
// Each mesh copies its selected faces into a new record and deletes them from its source, and every mesh's edits share one batch.
void SeparateSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, const ::selection::PrimaryEditInstanceMap &primaries) {
    std::vector<state::Entity> entities;
    std::vector<MeshTopologyTask> tasks;
    for (const auto e:mesh_entities) {
        const auto id=r.get<const MeshHandle>(e).StoreId;
        tasks.push_back({.SourceId=id,.Op=MeshTopologyOp::KeepSelectedFaces});
        tasks.push_back({.SourceId=id,.Op=MeshTopologyOp::DeleteFaces});
        entities.insert(entities.end(),{e,e});
    }
    RunTopologyAction(r,mesh_entities,[&] {
        const auto outputs=EditTopology(r,entities,tasks);
        std::vector<state::Entity> created;
        for (uint32_t i=0u;i<tasks.size();++i) {
            if (tasks[i].Op!=MeshTopologyOp::KeepSelectedFaces || !outputs[i]) continue;
            const auto primary=primaries.find(entities[i]);
            const auto instance=primary!=primaries.end() ? primary->second : state::Null;
            MeshInstanceCreateInfo create{
                .Name=std::format("{}.001",instance!=state::Null ? GetName(r,instance) : "Mesh"),
                .Transform=instance!=state::Null ? *WorldTransformOf(r, instance) : Transform{},
                .Select=MeshInstanceCreateInfo::SelectBehavior::None,
            };
            created.push_back(::AddMesh(r,*outputs[i],std::move(create)).first);
        }
        if (created.empty()) return false;
        RequestRender(r,RenderRequest::Rebuild);
        r.Context.get<GpuSceneState>().LodDemand.insert(created.begin(),created.end());
        mtl::ComputeChain chain{r.Context.get<const MeshStore>().BufferContext()};
        UpdateAuthoredMorphShading(r,chain,created);
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

// Each closed loop of boundary edges, selected ones or all of them, as its vertices in the boundary's own direction.
std::vector<std::vector<uint32_t>> BoundaryChains(state::Scene &r, const Mesh &mesh, bool selected_only, uint32_t max_sides=0u) {
    const auto &meshes=r.Context.get<const MeshStore>();
    const auto &c = mesh.GetConnectivity();
    // Both views visit their edges in ascending handle order.
    std::vector<uint32_t> edges;
    const auto collect=[&](const auto &view) { view.ForEach([&](uint32_t edge) { edges.push_back(edge); }); };
    if (selected_only) collect(meshes.GetSelectedElements(mesh.GetStoreId(), Element::Edge));
    else collect(meshes.GetBoundaryEdges(mesh.GetStoreId()));
    std::vector<uint32_t> starts;
    for (const auto edge : edges) {
        const auto h = mesh.GetHalfedge(he::EH{edge}, 0);
        if (!c.Opposites[*h]) starts.push_back(*h);
    }
    std::ranges::sort(starts);
    const auto candidate=[&](uint32_t h) {
        return h!=InvalidOffset && !c.Opposites[h] && std::ranges::binary_search(edges,*mesh.GetEdge(Mesh::HH{h}));
    };
    std::unordered_map<uint32_t,uint32_t> selected_outgoing;
    if (selected_only) for (const auto h:starts) {
        const auto vertex=*mesh.GetFromVertex(Mesh::HH{h});
        const auto [it,unique]=selected_outgoing.emplace(vertex,h);
        if (!unique) it->second=InvalidOffset;
    }
    // Follow the face fan at the current boundary halfedge's destination to
    // find the next boundary halfedge on the same surface sheet. Vertex-based
    // pairing loses loops when distinct boundaries share a vertex.
    const auto successor = [&](uint32_t h) -> uint32_t {
        auto next=c.Next(Mesh::HH{h});
        const auto across=[&](Mesh::HH at) -> Mesh::HH {
            if (!at) return {};
            const auto opposite=c.Opposites[*at];
            return opposite ? c.Next(opposite) : Mesh::HH{};
        };
        auto fast=next;
        while (next && c.Opposites[*next]) {
            next=across(next);
            fast=across(across(fast));
            if (fast && next==fast) return InvalidOffset;
        }
        return next ? *next : InvalidOffset;
    };
    std::vector<std::vector<uint32_t>> loops;
    std::unordered_set<uint32_t> used;
    used.reserve(starts.size());
    for (const auto start : starts) {
        if (used.contains(start)) continue;
        std::vector<uint32_t> halfedges;
        auto h = start;
        bool closed = false;
        uint32_t length=0u;
        while (candidate(h)) {
            if (!used.insert(h).second) {
                closed = h == start;
                break;
            }
            ++length;
            if (!max_sides || halfedges.size()<max_sides) halfedges.push_back(h);
            const auto next=successor(h);
            if (selected_only && !candidate(next)) {
                // A selected hole may touch an unselected boundary at one
                // vertex. Follow its sole selected outgoing edge there.
                const auto it=selected_outgoing.find(*mesh.GetToVertex(Mesh::HH{h}));
                h=it==selected_outgoing.end() ? InvalidOffset : it->second;
            } else h=next;
        }
        if (closed && length>=3u && (!max_sides || length<=max_sides)) {
            auto &loop=loops.emplace_back();
            loop.reserve(halfedges.size());
            for (const auto edge:halfedges)
                loop.push_back(*mesh.GetFromVertex(Mesh::HH{edge}));
        }
    }
    return loops;
}

// Each closed boundary loop as the vertex loop of the face that fills it, wound against the boundary.
std::vector<std::vector<uint32_t>> BoundaryLoops(state::Scene &r, const Mesh &mesh, bool selected_only, uint32_t max_sides=0u) {
    auto loops = BoundaryChains(r, mesh, selected_only, max_sides);
    for (auto &loop : loops) std::ranges::reverse(loop);
    return loops;
}

// A face list task over canonical vertex handles. New vertices use handles
// starting at the current arena capacity, beyond every existing handle.
MeshTopologyTask FaceListTask(state::Scene &r, const Mesh &mesh,
                              std::span<const std::vector<uint32_t>> loops, std::span<const vec3> positions = {}) {
    const auto appended_base=r.Context.get<const MeshStore>().Arenas().Vertices.Capacity();
    if (uint64_t(appended_base)+positions.size()>UINT32_MAX) throw std::length_error("Face list exceeds the vertex handle address space.");
    MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::AddFaces, .AppendedBase=appended_base};
    task.List.push_back(uint32_t(positions.size()));
    for (const auto &p : positions) {
        task.List.push_back(std::bit_cast<uint32_t>(p.x));
        task.List.push_back(std::bit_cast<uint32_t>(p.y));
        task.List.push_back(std::bit_cast<uint32_t>(p.z));
    }
    uint32_t attribute_source=InvalidOffset;
    for (const auto &loop:loops) for (const auto vertex:loop)
        if (vertex<appended_base && attribute_source==InvalidOffset) attribute_source=vertex;
    if (attribute_source==InvalidOffset) throw std::invalid_argument("Face creation needs a source vertex for attributes.");
    task.List.push_back(attribute_source);
    task.List.push_back(uint32_t(loops.size()));
    for (const auto &loop : loops) {
        task.List.push_back(uint32_t(loop.size()));
        task.List.insert(task.List.end(), loop.begin(), loop.end());
    }
    return task;
}

// Bridges the two closed loops of selected boundary edges, pairing each vertex of the longer with its share of the shorter.
void BridgeSelected(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        auto chains = BoundaryChains(r, mesh, true);
        if (chains.size() != 2) return {};
        if (chains[0].size() < chains[1].size()) std::swap(chains[0], chains[1]);
        const auto &a = chains[0], &b = chains[1];
        // The strip runs along the longer loop and against the shorter one, starting at the shorter's nearest vertex.
        uint32_t start = 0;
        float best = std::numeric_limits<float>::max();
        for (uint32_t j = 0; j < b.size(); ++j) {
            if (const auto d = Distance2(mesh.GetPosition(Mesh::VH{a[0]}), mesh.GetPosition(Mesh::VH{b[j]})); d < best) {
                best = d;
                start = j;
            }
        }
        const auto na = uint32_t(a.size()), nb = uint32_t(b.size());
        const auto at_b = [&](uint32_t steps) { return b[(start + nb - steps % nb) % nb]; };
        std::vector<std::vector<uint32_t>> faces;
        for (uint32_t i = 0; i < na; ++i) {
            const uint32_t j0 = (i * nb) / na, j1 = ((i + 1) * nb) / na;
            if (j0 == j1) {
                faces.push_back({a[(i + 1) % na], a[i], at_b(j0)});
                continue;
            }
            faces.push_back({a[(i + 1) % na], a[i], at_b(j0), at_b(j0 + 1)});
            for (uint32_t j = j0 + 1; j < j1; ++j) faces.push_back({a[(i + 1) % na], at_b(j), at_b(j + 1)});
        }
        return FaceListTask(r, mesh, faces);
    });
}

// Fills one closed loop of selected boundary edges with a Coons patch of quads, `span` edges along its first side.
void GridFillSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, uint32_t span) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto chains = BoundaryChains(r, mesh, true);
        if (chains.size() != 1 || chains[0].size() % 2 != 0 || chains[0].size() < 4) return {};
        const auto &loop = chains[0];
        const auto length = uint32_t(loop.size());
        const auto appended_base=r.Context.get<const MeshStore>().Arenas().Vertices.Capacity();
        const uint32_t s = std::clamp(span == 0 ? std::max(length / 4, 1u) : span, 1u, length / 2 - 1), t = length / 2 - s;
        const uint64_t interior=uint64_t(s-1u)*(t-1u), cells=uint64_t(s)*t;
        if (uint64_t(appended_base)+interior>UINT32_MAX || 3ull+3u*interior+5u*cells>UINT32_MAX) {
            throw std::length_error("Grid fill exceeds the vertex or face-list address space.");
        }
        // Nodes run along the first side (u) and up the second (v), with the loop's four sides as the rails.
        const auto rail = [&](uint32_t k) { return mesh.GetPosition(Mesh::VH{loop[k % length]}); };
        std::vector<uint32_t> node((s + 1) * (t + 1), InvalidOffset);
        std::vector<vec3> positions;
        const auto index = [&](uint32_t i, uint32_t j) { return j * (s + 1) + i; };
        for (uint32_t i = 0; i <= s; ++i) {
            node[index(i, 0)] = loop[i];
            node[index(i, t)] = loop[(2 * s + t - i) % length];
        }
        for (uint32_t j = 0; j <= t; ++j) {
            node[index(s, j)] = loop[s + j];
            node[index(0, j)] = loop[(2 * s + 2 * t - j) % length];
        }
        for (uint32_t j = 1; j < t; ++j) {
            for (uint32_t i = 1; i < s; ++i) {
                const float u = float(i) / float(s), v = float(j) / float(t);
                const vec3 bottom = rail(i), top = rail(2 * s + t - i), right = rail(s + j), left = rail(2 * s + 2 * t - j);
                const vec3 p00 = rail(0), p10 = rail(s), p11 = rail(s + t), p01 = rail(2 * s + t);
                const vec3 p = bottom * (1.f - v) + top * v + left * (1.f - u) + right * u -
                    (p00 * ((1.f - u) * (1.f - v)) + p10 * (u * (1.f - v)) + p01 * ((1.f - u) * v) + p11 * (u * v));
                node[index(i, j)] = appended_base + uint32_t(positions.size());
                positions.push_back(p);
            }
        }
        std::vector<std::vector<uint32_t>> faces;
        for (uint32_t j = 0; j < t; ++j) {
            for (uint32_t i = 0; i < s; ++i) faces.push_back({node[index(i + 1, j)], node[index(i, j)], node[index(i, j + 1)], node[index(i + 1, j + 1)]});
        }
        return FaceListTask(r, mesh, faces, positions);
    });
}

void FillHolesSelected(state::Scene &r, std::span<const state::Entity> mesh_entities, uint32_t sides) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        auto loops = BoundaryLoops(r, mesh, false, sides);
        if (loops.empty()) return {};
        return FaceListTask(r, mesh, loops);
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
        float best = 0.f;
        for (const auto v : points)
            if (const auto d = Distance2(at(v), at(seed[0])); d > best) {
                best = d;
                seed[1] = v;
            }
        const float extent = std::sqrt(best);
        best = 0.f;
        for (const auto v : points)
            if (const auto d = Length(Cross(at(seed[1]) - at(seed[0]), at(v) - at(seed[0]))); d > best) {
                best = d;
                seed[2] = v;
            }
        best = 0.f;
        const auto seed_normal = Cross(at(seed[1]) - at(seed[0]), at(seed[2]) - at(seed[0]));
        for (const auto v : points)
            if (const auto d = std::abs(Dot(seed_normal, at(v) - at(seed[0]))); d > best) {
                best = d;
                seed[3] = v;
            }
        if (best < 1e-12f) return {};

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
        return FaceListTask(r, mesh, hull);
    });
}

// Dissolves selected edges and connects their far vertices in one local topology transaction.
void EdgeRotateSelected(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    const auto &meshes = r.Context.get<const MeshStore>();
    std::vector<MeshTopologyTask> tasks;
    std::vector<state::Entity> entities;
    for (const auto e : mesh_entities) {
        const auto id = r.get<const MeshHandle>(e).StoreId;
        const Mesh mesh{meshes, id};
        const auto &c = mesh.GetConnectivity();
        MeshTopologyTask task{.SourceId = id, .Op = MeshTopologyOp::RotateEdges, .Flags = TopologyFlagListSelects, .List = {0}};
        meshes.GetSelectedElements(id, Element::Edge).ForEach([&](uint32_t edge) {
            const auto h = mesh.GetHalfedge(he::EH{edge}, 0);
            const auto opposite = c.Opposites[*h];
            if (!opposite) return;
            task.List.push_back(*mesh.GetToVertex(c.Next(h)));
            task.List.push_back(*mesh.GetToVertex(c.Next(opposite)));
            task.List[0] += 2;
        });
        if (task.List[0] == 0) continue;
        tasks.push_back(std::move(task));
        entities.push_back(e);
    }
    RunTasks(r,entities,tasks);
}

void FillSelected(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto loops = BoundaryLoops(r, mesh, true);
        if (loops.empty()) return {};
        return FaceListTask(r, mesh, loops);
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
        return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::DuplicateFaces, .Flags = TopologyFlagTransformCopies | TopologyFlagFlipCopies | TopologyFlagSelectAll, .CopyRotation = mirror};
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

// Deletes each mesh's edges without a face, then its vertices without fan corners, as two transactions.
// A mesh without faces has only edges without a face, and a mesh with faces has none.
// Fans hold line corners too, so a vertex without fan corners has no edge.
void DeleteLoose(state::Scene &r, std::span<const state::Entity> mesh_entities) {
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        if (mesh.FaceCount() || !mesh.EdgeCount()) return {};
        return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::DeleteEdges, .Flags = TopologyFlagSelectAll};
    });
    RunPerMesh(r, mesh_entities, [&](state::Entity, const Mesh &mesh) -> std::optional<MeshTopologyTask> {
        const auto &fans = mesh.GetConnectivity().VertexCorners;
        MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::DeleteVertices, .Flags = TopologyFlagListSelects, .List = {0u}};
        for (const auto v : mesh.vertices()) if (!fans[*v].y) task.List.push_back(*v);
        if (task.List.size() == 1u) return {};
        task.List.front() = uint32_t(task.List.size() - 1u);
        return task;
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
} // namespace

namespace action::mesh {
bool UpdateInsetPreview(state::Scene &r, state::Entity viewport, const Inset &inset, InsetPreviewCache &cache) {
    const profile::CpuScope scope{"UpdateInsetPreview"};
    const auto targets=SelectedEditMeshes(r,viewport);
    if (targets.empty() || targets.size()!=cache.Entries.size()) return false;
    const auto op=inset.Individual ? MeshTopologyOp::InsetIndividual : MeshTopologyOp::InsetRegion;
    const auto flags=inset.Even ? TopologyFlagEvenOffset : 0u;
    auto &meshes=r.Context.get<MeshStore>();
    // Project::Record discards this cache for every other action, including
    // selection changes. Parameter updates retain the staged face selection.
    for (size_t i=0u;i<targets.size();++i) {
        const auto &entry=cache.Entries[i];
        if (entry.Entity!=targets[i] || entry.Op!=op || entry.Flags!=flags ||
            GetMesh(r,entry.Entity).GetStoreId()!=entry.StoreId || !entry.Basis.Count) return false;
    }
    {
        const profile::CpuScope capture{"InsetCaptureVertices"};
        for (const auto &entry:cache.Entries)
            meshes.Arenas().Vertices.Buffer.CaptureWriteRanges(entry.Ranges,sizeof(Vertex));
    }
    // The positions, their refresh, the selection aggregates and the meshlet refit share one submit.
    mtl::ComputeChain chain{meshes.BufferContext()};
    {
        const profile::CpuScope positions{"InsetPositionPass"};
        const auto &pipeline=GetMeshPipelines(r)[MeshPass::InsetPreviewPositions];
        // Each entry writes its own mesh's vertices.
        chain.Concurrent([&] {
            for (const auto &entry:cache.Entries) {
                const InsetPreviewPushConstants pc{
                    .Basis={cache.Basis.Buffer.Slot,entry.Basis.Offset},.VertexSlot=meshes.Slots().Vertices,
                    .Count=uint32_t(uint64_t(entry.Basis.Count)*sizeof(uint32_t)/sizeof(InsetVertexBasis)),.Thickness=std::max(inset.Thickness,0.f),.Depth=inset.Depth};
                chain.Groups(pipeline,pc,(pc.Count+255u)/256u);
            }
        });
    }
    std::vector<MeshVertexChanges> changed;
    changed.reserve(cache.Entries.size());
    for (const auto &entry:cache.Entries) changed.push_back({entry.Entity,entry.Ranges});
    RefreshEditedPositions(r,chain,changed);
    std::vector<MeshStore::SelectionUpdate> aggregates;
    for (const auto &entry:cache.Entries) {
        if (!meshes.Get(entry.StoreId).SelectionSummary.Count) continue;
        auto &blocks=aggregates.emplace_back(MeshStore::SelectionUpdate{.StoreId=entry.StoreId}).Blocks[0];
        for (const auto &range:entry.Ranges)
            for (uint64_t handle=range.Offset,end=uint64_t(range.Offset)+range.Count;handle<end;) {
                blocks.push_back(uint32_t(handle/MeshElementBlockSize));
                handle=(handle/MeshElementBlockSize+1u)*MeshElementBlockSize;
            }
    }
    meshes.UpdateSelection(r,chain,aggregates);
    // The canonical fine meshlet records are also used by static culling,
    // coarse repair, and replay. Refit their bounds and cones from current
    // positions without changing the meshlet topology or render vertex order.
    const auto &gpu=r.Context.get<const GpuBuffers>();
    const auto &scene=r.Context.get<const GpuSceneState>();
    std::vector<MeshletBoundsRefitJob> refits;
    for (const auto &entry:cache.Entries)
        refits.push_back({&MeshBuffersOf(r,entry.Entity),&gpu.GeometryWork,scene.EditWork.at(entry.Entity).Meshlets});
    {
        const profile::CpuScope refit_scope{"InsetRefitPass"};
        RefitCanonicalMeshletBounds(r,chain,refits);
    }
    chain.Submit();
    RefreshElementSelectionSummaries(r,targets);
    // This preview already repaired the canonical meshlet bounds. Its scratch
    // edit work still serves later parameter updates, without a full pose.
    for (const auto &entry:cache.Entries) r.Context.get<GpuSceneState>().EditWork.at(entry.Entity).RequiresPose=false;
    RequestRender(r, RenderRequest::Reuse);
    return true;
}

void Apply(state::Scene &r, state::Entity viewport, const Action &action) {
    const auto targets = SelectedEditMeshes(r, viewport);
    const auto latch_translate = [&] { r.emplace_or_replace<StartScreenTransform>(viewport, TransformGizmo::TransformType::Translate); };
    std::visit(
        overloaded{
            [&](const Delete &a) {
                if (a.Mode == DeleteMode::Loose) DeleteLoose(r, targets);
                else RunOperator(r, targets, MeshTopologyOp(uint32_t(a.Mode)));
            },
            [&](const Merge &a) { MergeSelected(r, targets, a.Mode, std::max(a.Distance, 0.f)); },
            [&](const Extrude &a) {
                using Mode = ExtrudeMode;
                const auto op = a.Mode == Mode::Edges ? MeshTopologyOp::ExtrudeEdges : a.Mode == Mode::FacesIndividual ? MeshTopologyOp::ExtrudeFacesIndividual :
                                                                                                                         MeshTopologyOp::ExtrudeRegion;
                RunOperator(r, targets, op);
                latch_translate();
            },
            [&](Duplicate) {
                RunOperator(r, targets, MeshTopologyOp::DuplicateFaces);
                latch_translate();
            },
            [&](Split) { RunOperator(r, targets, MeshTopologyOp::SplitFaces); },
            [&](Separate) { SeparateSelected(r, targets, r.get<const EditPrimaries>(viewport).All); },
            [&](const Subdivide &a) { RunOperator(r, targets, MeshTopologyOp::Subdivide, float(std::max(a.Cuts, 1u))); },
            [&](Triangulate) { RunOperator(r, targets, MeshTopologyOp::Triangulate); },
            [&](TrisToQuads) { RunOperator(r, targets, MeshTopologyOp::TrisToQuads); },
            [&](const Poke &a) { RunOperator(r, targets, MeshTopologyOp::Poke, a.Offset); },
            [&](FlipNormals) { RunOperator(r, targets, MeshTopologyOp::FlipNormals); },
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
            [&](ConnectVertices) { RunOperator(r, targets, MeshTopologyOp::ConnectVertices); },
            [&](const Knife &a) { KnifeSelected(r, targets, r.get<const EditPrimaries>(viewport).All, a.Start, a.End, *a.View); },
            [&](BridgeEdgeLoops) { BridgeSelected(r, targets); },
            [&](const GridFill &a) { GridFillSelected(r, targets, a.Span); },
            [&](const FillHoles &a) { FillHolesSelected(r, targets, a.Sides); },
            [&](ConvexHull) { ConvexHullSelected(r, targets); },
            [&](EdgeRotate) { EdgeRotateSelected(r, targets); },
            [&](const Bevel &a) {
                RunPerMesh(r, targets, [&](state::Entity, const Mesh &mesh) {
                    return MeshTopologyTask{.SourceId = mesh.GetStoreId(),
                        .Op = a.Vertices ? MeshTopologyOp::BevelVertices : MeshTopologyOp::BevelEdges,
                        .Param0 = std::max(a.Width, 0.f), .Steps = std::max(a.Segments, 1u)};
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
                    case Mode::Edges: return RunOperator(r, targets, MeshTopologyOp::DissolveEdges);
                    case Mode::Faces: return RunOperator(r, targets, MeshTopologyOp::DissolveFaces);
                    case Mode::Limited: return RunOperator(r, targets, MeshTopologyOp::DissolveLimited, std::clamp(a.Angle, 0.f, std::numbers::pi_v<float>));
                    case Mode::Degenerate: return RunOperator(r, targets, MeshTopologyOp::DissolveDegenerate, std::max(a.Distance, 0.f));
                }
            },
        },
        action
    );
}
} // namespace action::mesh
