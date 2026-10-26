#include "procedure/Cook.h"
#include "procedure/InlineSelection.h"
#include "procedure/Procedure.h"
#include "procedure/ProcedureSchema.h"

#include "Profile.h"
#include "SortUnique.h"
#include "TransformMath.h"
#include "mesh/BeautifyFaces.h"
#include "mesh/Decimate.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshTopologyEdit.h"
#include "mesh/PositionOperations.h"
#include "mesh/Primitives.h"
#include "mesh/RecalculateNormals.h"
#include "mesh/SharpnessOperations.h"
#include "mesh/Unsubdivide.h"
#include "numeric/MatrixMath.h"
#include "object/ObjectOps.h"
#include "object/PendingSync.h"
#include "render/ElementWorkOps.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/MeshTopologyRepair.h"
#include "render/MeshletBuildGpu.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportRenderGpu.h"

#include <format>

using state::On;

namespace procedure {
// The mesh the last cook pass cooked alone, and the kept states whose whole mesh storage restores exactly: the states kept while only that mesh's passes wrote mesh storage.
struct CookPass {
    state::Entity Mesh{state::Null};
    // The mesh storage state the pass ended with, once it ended.
    std::optional<store::Hash128> End;
    std::vector<store::Hash128> Exact;
};
void Reset(CookPass &pass, state::Registry &) { pass = {}; }

namespace {
using namespace action::mesh;
// Keep mesh storage as it stands as the state of each Source node whose pending state refers to `record`, or of every pending state for InvalidOffset.
void FinishPendingSourceStates(state::Registry &r, uint32_t record) {
    std::optional<store::Hash128> state;
    for (const auto [entity, source] : std::as_const(r).view<const SourceState>().each()) {
        if (source.State != store::Hash128{} || source.StoreId == InvalidOffset || (record != InvalidOffset && source.StoreId != record)) continue;
        if (!state) state = r.Resources.get<MeshStore>().KeepState();
        r.replace<SourceState>(entity, *state, source.StoreId);
    }
}

// Optional domain masks, where absent masks select every element.
struct Masks {
    std::optional<GeometryMasks> In;
    // The edit domain the node's inline selection was captured in.
    Element Domain{Element::Vertex};
    // The handle of the inline selection's active element, or InvalidOffset.
    uint32_t Active{InvalidOffset};
    // Copies the emitted selection into the output record.
    bool Publish{};
    // The mesh entity drawing the input record with current render records, which a node edits in place and repairs, setting Repaired.
    state::Entity Drawn{state::Null};
    bool *Repaired{};
    // The change a restore made to the drawn record, whose vertex ranges and blocks the next drawn position edit refreshes with its own.
    MeshStore::Change *Refresh{};
};

// Evaluated geometry, instances, selection masks, attribute, or constant value.
// Absent mask blocks select every element; InvalidOffset denotes an absent attribute.
struct SocketValue {
    SocketType Type{SocketType::Geometry};
    // Borrow records from MeshEntity for read-only use; own records when MeshEntity is null.
    // Borrowed records remain owned by their mesh and must never be edited or released by the cook.
    state::Entity MeshEntity{state::Null};
    uint32_t Instance{InvalidOffset}; // The record an instances value's points each place, or InvalidOffset for points placing nothing
    union {
        uint32_t Record{InvalidOffset};
        std::array<uint32_t, 3> MaskFirst;
        uint32_t Attribute;
        AttributeValue Value;
    };
    bool Borrowed() const { return Type == SocketType::Geometry && MeshEntity != state::Null; }
};
constexpr std::array<uint32_t, 3> EveryElement{InvalidOffset, InvalidOffset, InvalidOffset};
// Return the output record, the input for an unchanged operation, or the reason the operation failed.
using Outcome = std::expected<uint32_t, NodeFault>;

// Gather edited arena blocks before erasure releases empty blocks.
struct EditedBlocks {
    std::vector<uint32_t> Vertices, Faces, Corners;
};

void UpdatePoseMembership(state::Registry &r, const MeshTopologyEdit &edit, EditedBlocks edited) {
    auto &buffers = r.Resources.get<GpuBuffers>();
    const auto &meshes = r.Resources.get<const MeshStore>();
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

// Edit the record the mesh entity draws in place and repair its render, pose and selection state.
void EditDrawnTopology(state::Registry &r, state::Entity mesh_entity, const MeshTopologyTask &task) {
    auto &meshes = r.Resources.get<MeshStore>();
    ReleaseMeshEditWork(r, mesh_entity);
    mtl::ComputeChain chain{meshes.BufferContext(), TopologyScratchWords};
    auto edits = MeshTopologyEdit::Construct(r, chain, std::span{&task, 1u});
    MeshTopologyEdit::PublishAll(r, edits);
    auto &edit = edits.front();
    if (!edit.Output) {
        chain.Submit();
        return;
    }
    const auto &owner = meshes.Get(edit.StoreId);
    std::vector<std::pair<state::Entity, const MeshTopologyEdit *>> repairs;
    if ((owner.RenderTopologies & 1u) != 0u || edit.AddedTriangleCount) repairs.emplace_back(mesh_entity, &edit);
    RepairTopologyRender(r, chain, repairs);
    // Resolve edited blocks before erasure, including surviving vertices and edges whose incidence changed.
    std::vector<ElementMeshletRepair> element_repairs;
    using Domain = MeshStore::ElementDomain;
    const auto &storage = edit.Chain.Scratch;
    if (edit.Repair) {
        auto &points = element_repairs.emplace_back(ElementMeshletRepair{.StoreId = edit.StoreId, .Topology = 2u}).Elements;
        ForEachWorkElement(storage, edit.Repair->Elements[0], [&](uint32_t v) { points.push_back(v); });
        ForEachWorkElement(storage, edit.Output->Retired[0], [&](uint32_t v) { points.push_back(v); });
        auto &wires = element_repairs.emplace_back(ElementMeshletRepair{.StoreId = edit.StoreId, .Topology = 1u}).Elements;
        const auto edges = meshes.Arenas().HalfedgeEdges.Buffer.GetSpan<uint32_t>();
        ForEachWorkElement(storage, edit.RetiredEdges, [&](uint32_t e) { wires.push_back(e); });
        ForEachWorkElement(storage, edit.Repair->Elements[1], [&](uint32_t h) { wires.push_back(edges[meshes.HandleAddress(edit.StoreId, Domain::Halfedge, ElementHandle{h}).Value()]); });
    }
    EditedBlocks edited{.Corners = edit.OldCornerBlocks};
    if (edit.Repair) {
        edited.Vertices.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Vertex, storage, edit.Repair->Elements[0]));
        edited.Faces.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Face, storage, edit.Repair->Elements[2]));
        if (meshes.Get(edit.StoreId).NormalSectors.Count) edited.Corners.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Halfedge, storage, edit.Repair->Elements[1]));
    }
    edited.Vertices.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Vertex, storage, edit.Output->Retired[0]));
    edited.Faces.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Face, storage, edit.Output->Retired[1]));
    MeshTopologyEdit::FinishAll(r, std::array{&edit});
    UpdatePoseMembership(r, edit, std::move(edited));
    RefreshMeshBinding(r, edit.StoreId);
    r.Resources.get<GpuSceneState>().EditSelectionDirty = true;
    RepairElementMeshlets(r, chain, element_repairs);
    chain.Submit();
    // Owner maps count only rendered elements, so topology changes need no mesh scan.
    auto &repaired = meshes.WriteRecord(edit.StoreId);
    uint32_t mask = 0u;
    for (uint32_t t = 0u; t < 3u; ++t) {
        if (repaired.ElementMeshletBlockCounts[t]) mask |= 1u << t;
    }
    repaired.RenderTopologies = mask ? mask : 4u;
    RefreshElementSelectionSummaries(r, std::array{mesh_entity});
}

// Refreshes the selection aggregates of the blocks a restore changed and clears the change.
void PublishRestored(state::Registry &r, mtl::ComputeChain &chain, uint32_t id, MeshStore::Change &restored) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (meshes.Get(id).SelectionSummary.Count && std::ranges::any_of(std::span{restored.Blocks}.first(3u), [](const auto &blocks) { return !blocks.empty(); })) {
        MeshStore::SelectionUpdate update{.StoreId = id};
        for (uint32_t d = 0u; d < 3u; ++d) {
            update.Blocks[d] = restored.Blocks[d];
            SortUnique(update.Blocks[d]);
        }
        meshes.UpdateSelection(r, chain, std::array{update});
    }
    restored = {};
}

// Edit the positions of the record the mesh entity draws and refresh its normals, tessellation, render bounds and selection aggregates.
// The vertex ranges of `refresh`, which it then clears, join the edited ranges, and its blocks refresh their aggregates with them.
void EditDrawnPositions(state::Registry &r, state::Entity mesh_entity, const PositionOperationTarget &target, PositionEditOp op, float factor, uint32_t repeat, const PositionOperationOptions &options, MeshStore::Change *refresh) {
    auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto edited = EncodePositionOperations(meshes, GetMeshPipelines(r), chain, std::span{&target, 1u}, op, factor, repeat, options);
    if (edited.empty()) {
        chain.Submit();
        return;
    }
    auto ranges = edited.front().Ranges;
    if (refresh) {
        ranges.append_range(refresh->VertexRanges);
        std::ranges::sort(ranges, {}, &Range::Offset);
        // Overlapping and touching ranges merge.
        size_t count = 0u;
        for (const auto range : ranges) {
            if (!range.Count) continue;
            if (count && range.Offset <= ranges[count - 1u].Offset + ranges[count - 1u].Count) {
                auto &last = ranges[count - 1u];
                last.Count = std::max(last.Offset + last.Count, range.Offset + range.Count) - last.Offset;
            } else {
                ranges[count++] = range;
            }
        }
        ranges.resize(count);
    }
    // The publication refits the record's meshlets before the repair of changed triangles replaces any of them.
    auto repair = RefreshEditedPositions(r, chain, std::array{MeshVertexChanges{mesh_entity, ranges}});
    PublishEditedPositions(r, chain, std::array{mesh_entity});
    if (refresh) PublishRestored(r, chain, target.StoreId, *refresh);
    if (repair) chain.AfterSubmit(std::move(repair));
    chain.Submit();
}

// Permute the face attribute layer of the record the mesh entity draws and repair the triangles of the faces it permuted.
void EditDrawnFaceAttributes(state::Registry &r, state::Entity mesh_entity, const PositionOperationTarget &target, bool colors, uint32_t uv_set, uint32_t operation) {
    auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto changes = EncodeFaceAttributeOperation(meshes, GetMeshPipelines(r), chain, std::span{&target, 1u}, colors, uv_set, operation);
    if (changes.empty()) {
        chain.Submit();
        return;
    }
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(target.StoreId);
    const auto &faces = changes.front().Faces;
    const auto bound = std::min(faces.Incidence, a.Triangles.Set(record.TriangleData).BlockCount);
    const auto handles = a.Triangles.HandleBound(record.TriangleData);
    chain.Scratch.ReserveAdditional(1u + ElementWorkWords(handles, bound) + SortElementWorkWords(WorkCapacity(handles, bound)));
    std::vector<std::pair<state::Entity, FaceTriangles>> repairs{{mesh_entity, EncodeFaceTriangles(r, chain, target.StoreId, faces)}};
    chain.Submit();
    repairs.front().second.Finish(chain);
    RepairFaceRender(r, chain, repairs);
    chain.Submit();
}

// Set the sharpness of the record the mesh entity draws and repair the triangles whose shading it changed.
void EditDrawnSharpness(state::Registry &r, state::Entity mesh_entity, const SharpnessOperationTarget &target, EditSharpnessOperation operation, bool value, float angle) {
    auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto changes = ExecuteGeometrySharpness(r, chain, std::span{&target, 1u}, operation, value, angle);
    if (changes.empty()) {
        chain.Submit();
        return;
    }
    r.Resources.get<GpuSceneState>().EditSelectionDirty = true;
    const std::vector<std::pair<state::Entity, FaceTriangles>> repairs{{mesh_entity, changes.front().Triangles}};
    RepairFaceRender(r, chain, repairs);
    chain.Submit();
}

GeometrySelection Handles(const MeshStore &meshes, uint32_t id, const Masks &masks) { return masks.In ? ReadMasks(meshes, id, *masks.In) : AllHandles(Mesh{meshes, id}); }
SelectionSource SourceOf(const MeshStore &meshes, uint32_t id, const Masks &masks) { return masks.In ? SelectionSource{*masks.In} : SelectionSource{AllHandles(Mesh{meshes, id})}; }

MeshTopologyTask Task(uint32_t id, MeshTopologyOp op, const Masks &masks, float param0 = 0.f, float param1 = 0.f, uint32_t flags = 0u) {
    MeshTopologyTask task{.SourceId = id, .Op = op, .Param0 = param0, .Param1 = param1, .Flags = flags};
    if (masks.In) task.Selection = *masks.In;
    else task.Flags |= TopologyFlagSelectAll;
    return task;
}
// Select the input selection on the record a publishing node edits, which keeps the elements the edit leaves alone selected as the viewport operator does.
void PublishInput(state::Registry &r, uint32_t id, const Masks &masks) {
    if (!masks.Publish || !masks.In) return;
    auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    meshes.SelectMasks(r, chain, id, *masks.In);
    chain.Submit();
}
// Return the output record or a node fault.
// An edit in place of the drawn record repairs its render records.
Outcome Execute(state::Registry &r, std::optional<MeshTopologyTask> task, const Masks &masks) {
    if (!task) return std::unexpected(NodeFault::Selection);
    if (masks.Publish) PublishStoreSelection(r.Resources.get<const MeshStore>(), *task);
    if (masks.Drawn != state::Null && !TopologyCreatesRecord(task->Op)) {
        *masks.Repaired = true;
        EditDrawnTopology(r, masks.Drawn, *task);
        return task->SourceId;
    }
    const auto result = ExecuteGeometryTopology(r, std::span{&*task, 1u}).front();
    return result.Changed ? result.GeometryId : task->SourceId;
}
// Use planned primitive handles as the task selection.
Outcome ExecutePlanned(state::Registry &r, std::optional<MeshTopologyTask> task, const GeometrySelection &selection, const Masks &masks) {
    if (task && task->Op == MeshTopologyOp::AddPrimitives) task->Selection = selection;
    return Execute(r, std::move(task), masks);
}
void ToPoints(state::Registry &r, uint32_t id) {
    if (Mesh{r.Resources.get<const MeshStore>(), id}.EdgeCount()) {
        const auto task = Task(id, MeshTopologyOp(uint32_t(DeleteMode::OnlyEdgesAndFaces)), {});
        ExecuteGeometryTopology(r, std::span{&task, 1u});
    }
}
void Empty(state::Registry &r, uint32_t id) {
    if (Mesh{r.Resources.get<const MeshStore>(), id}.VertexCount()) {
        const auto task = Task(id, MeshTopologyOp(uint32_t(DeleteMode::Vertices)), {});
        ExecuteGeometryTopology(r, std::span{&task, 1u});
    }
}
Outcome ExecuteStages(state::Registry &r, std::vector<MeshTopologyTask> tasks, const Masks &masks) {
    if (masks.Publish) {
        for (auto &task : tasks) PublishStoreSelection(r.Resources.get<const MeshStore>(), task);
    }
    if (masks.Drawn != state::Null && std::ranges::none_of(tasks, [](const auto &task) { return TopologyCreatesRecord(task.Op); })) {
        *masks.Repaired = true;
        for (const auto &task : tasks) EditDrawnTopology(r, masks.Drawn, task);
        return tasks.front().SourceId;
    }
    const auto results = ExecuteGeometryTopologyStages(r, tasks);
    if (std::ranges::none_of(results, &GeometryTopologyResult::Changed)) return tasks.front().SourceId;
    return results.back().GeometryId;
}
// A vertex-domain selection's active element is the reference vertex.
Outcome Reposition(state::Registry &r, uint32_t id, const Masks &masks, PositionEditOp op, float factor, uint32_t repeat, const PositionOperationOptions &options = {}, const Transform &world = {}) {
    auto &meshes = r.Resources.get<MeshStore>();
    const PositionOperationTarget target{
        .StoreId = id,
        .Selection = SourceOf(meshes, id, masks),
        .World = world,
        .Reference = masks.Domain == Element::Vertex ? masks.Active : InvalidOffset,
    };
    if (masks.Drawn != state::Null) {
        *masks.Repaired = true;
        EditDrawnPositions(r, masks.Drawn, target, op, factor, repeat, options, masks.Refresh);
    } else {
        ExecuteGeometryPositions(r, std::span{&target, 1u}, op, factor, repeat, options);
    }
    return id;
}
// Rotate the selected faces' corner UVs of a set or colors clockwise for operation 0, counterclockwise for 1, or reverse them for 2.
Outcome EditFaceAttributes(state::Registry &r, uint32_t id, const Masks &masks, bool colors, uint32_t uv_set, uint32_t operation) {
    if (uv_set >= MeshStore::MaxUvSets) return std::unexpected(NodeFault::Parameter);
    auto &meshes = r.Resources.get<MeshStore>();
    const PositionOperationTarget target{.StoreId = id, .Selection = SourceOf(meshes, id, masks)};
    if (masks.Drawn != state::Null) {
        *masks.Repaired = true;
        EditDrawnFaceAttributes(r, masks.Drawn, target, colors, uv_set, operation);
        return id;
    }
    mtl::ComputeChain chain{meshes.BufferContext()};
    EncodeFaceAttributeOperation(meshes, GetMeshPipelines(r), chain, std::span{&target, 1u}, colors, uv_set, operation);
    chain.Submit();
    return id;
}
uint32_t Axes(bool x, bool y, bool z) { return uint32_t(x) | uint32_t(y) << 1u | uint32_t(z) << 2u; }

// Per-input attribute IDs; InvalidOffset selects the constant parameter.
struct Linked {
    uint16_t Kind;
    std::array<uint32_t, MaxInputs> Attributes;
    template<auto... Ms> uint32_t Of() const {
        constexpr std::array<uint8_t, sizeof...(Ms)> path{uint8_t(field::IndexOf<Ms>)...};
        const auto sockets = KindSockets(Kind);
        for (uint16_t s = 0u; s < InputCount[Kind]; ++s) {
            if (sockets[s].Carries(path[0], path.size() > 1u ? path[1] : NoField)) return Attributes[s];
        }
        return InvalidOffset;
    }
    bool Any() const {
        return std::ranges::any_of(Attributes, [](uint32_t a) { return a != InvalidOffset; });
    }
};
// Require the expected domain and type, or InvalidOffset for a constant.
std::expected<uint32_t, NodeFault> ElementParameter(const MeshStore &meshes, uint32_t attribute, AttributeDomain domain, AttributeType type = AttributeType::Float) {
    if (attribute == InvalidOffset) return InvalidOffset;
    const auto &a = meshes.GetAttribute(attribute);
    if (a.Domain != domain || a.Type != type) return std::unexpected(NodeFault::Attribute);
    return attribute;
}
NamedAttributeRef RefOf(const MeshStore &meshes, uint32_t attribute) { return attribute == InvalidOffset ? NamedAttributeRef{} : meshes.AttributeRef(attribute); }
Outcome ExecuteElement(state::Registry &r, MeshTopologyTask task, const Masks &masks, AttributeDomain domain, uint32_t param0, uint32_t param1 = InvalidOffset) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto first = ElementParameter(meshes, param0, domain), second = ElementParameter(meshes, param1, domain);
    if (!first || !second) return std::unexpected(NodeFault::Attribute);
    task.Param0Attribute = RefOf(meshes, *first);
    task.Param1Attribute = RefOf(meshes, *second);
    return Execute(r, std::move(task), masks);
}
Outcome RepositionBy(state::Registry &r, uint32_t id, const Masks &masks, PositionEditOp op, float factor, uint32_t repeat, uint32_t attribute, PositionOperationOptions options) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto linked = ElementParameter(meshes, attribute, AttributeDomain::Vertex);
    if (!linked) return std::unexpected(linked.error());
    options.Factor = RefOf(meshes, *linked);
    return Reposition(r, id, masks, op, factor, repeat, options);
}

// Apply operators to masked elements, or every element when masks are absent.
Outcome Run(state::Registry &r, const Delete &p, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp(uint32_t(p.Mode)), m), m); }
Outcome Run(state::Registry &r, const Merge &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    const auto &vertices = selection.Vertices;
    const auto reference = vertices.empty() ? InvalidOffset : p.Mode == MergeMode::Last ? vertices.back() :
                                                                                          vertices.front();
    return ExecutePlanned(r, MergeTask(meshes, Mesh{meshes, id}, selection, p.Mode, std::max(p.Distance, 0.f), reference), selection, m);
}
Outcome Run(state::Registry &r, const Extrude &p, uint32_t id, const Masks &m) {
    const auto op = p.Mode == ExtrudeMode::Vertices ? MeshTopologyOp::ExtrudeVertices : p.Mode == ExtrudeMode::Edges ? MeshTopologyOp::ExtrudeEdges :
        p.Mode == ExtrudeMode::FacesIndividual                                                                       ? MeshTopologyOp::ExtrudeFacesIndividual :
                                                                                                                       MeshTopologyOp::ExtrudeRegion;
    return Execute(r, Task(id, op, m), m);
}
Outcome Run(state::Registry &r, Duplicate, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::DuplicateGeometry, m), m); }
Outcome Run(state::Registry &r, Split, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::SplitGeometry, m), m); }
Outcome Run(state::Registry &r, const Dissolve &p, uint32_t id, const Masks &m) {
    switch (p.Mode) {
        case DissolveMode::Vertices: return Execute(r, Task(id, MeshTopologyOp::DissolveVertices, m), m);
        case DissolveMode::Edges: return Execute(r, Task(id, MeshTopologyOp::DissolveEdges, m, 0.f, 0.f, p.KeepVertices ? TopologyFlagKeepVertices : 0u), m);
        case DissolveMode::Faces: return Execute(r, Task(id, MeshTopologyOp::DissolveFaces, m), m);
        case DissolveMode::Limited: {
            const auto flags = (m.Domain == Element::Face ? TopologyFlagFaceSelection : 0u) | (p.AllBoundaries ? TopologyFlagAllBoundaries : 0u) | (p.DelimitMaterials ? TopologyFlagDelimitMaterial : 0u) | (p.DelimitSharpEdges ? TopologyFlagDelimitSharp : 0u) | (p.DelimitUVs ? TopologyFlagDelimitUV : 0u);
            return Execute(r, Task(id, MeshTopologyOp::DissolveLimited, m, std::clamp(p.Angle, 0.f, .5f * std::numbers::pi_v<float>), 0.f, flags), m);
        }
        case DissolveMode::Degenerate: return Execute(r, Task(id, MeshTopologyOp::DissolveDegenerate, m, std::max(p.Distance, 0.f)), m);
    }
    std::unreachable();
}
Outcome Run(state::Registry &r, const Subdivide &p, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::Subdivide, m, float(std::max(p.Cuts, 1u))), m); }
Outcome Run(state::Registry &r, Triangulate, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::Triangulate, m), m); }
Outcome Run(state::Registry &r, TrisToQuads, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::TrisToQuads, m), m); }
Outcome Run(state::Registry &r, const Poke &p, uint32_t id, const Masks &m, const Linked &linked) {
    return ExecuteElement(r, Task(id, MeshTopologyOp::Poke, m, p.Offset), m, AttributeDomain::Face, linked.Of<&Poke::Offset>());
}
Outcome Run(state::Registry &r, FlipNormals, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::FlipNormals, m), m); }
Outcome Run(state::Registry &r, EdgeSplit, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::EdgeSplit, m), m); }
Outcome Run(state::Registry &r, Rip, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::EdgeSplit, m, 0.f, 0.f, TopologyFlagRipSelectCopies), m); }
Outcome Run(state::Registry &r, const Inset &p, uint32_t id, const Masks &m, const Linked &linked) {
    const auto task = Task(id, p.Individual ? MeshTopologyOp::InsetIndividual : MeshTopologyOp::InsetRegion, m, std::max(p.Thickness, 0.f), p.Depth, p.Even ? TopologyFlagEvenOffset : 0u);
    return ExecuteElement(r, task, m, AttributeDomain::Vertex, linked.Of<&Inset::Thickness>(), linked.Of<&Inset::Depth>());
}
Outcome Run(state::Registry &r, const Solidify &p, uint32_t id, const Masks &m, const Linked &linked) {
    return ExecuteElement(r, Task(id, MeshTopologyOp::Solidify, m, p.Thickness), m, AttributeDomain::Vertex, linked.Of<&Solidify::Thickness>());
}
Outcome Run(state::Registry &r, const Wireframe &p, uint32_t id, const Masks &m, const Linked &linked) {
    const auto flags = (p.Even ? TopologyFlagEvenOffset : 0u) | (p.Boundary ? TopologyFlagWireBoundary : 0u) | (p.Relative ? TopologyFlagWireRelative : 0u) | (p.Replace ? TopologyFlagWireReplace : 0u);
    const auto task = Task(id, MeshTopologyOp::Wireframe, m, std::max(p.Thickness, 0.f), std::clamp(p.Offset, -1.f, 1.f), flags);
    return ExecuteElement(r, task, m, AttributeDomain::Vertex, linked.Of<&Wireframe::Thickness>());
}
// A vertex bevel reads its width at each vertex, and an edge bevel at each edge.
Outcome Run(state::Registry &r, const Bevel &p, uint32_t id, const Masks &m, const Linked &linked) {
    auto task = Task(id, p.Vertices ? MeshTopologyOp::BevelVertices : MeshTopologyOp::BevelEdges, m, std::max(p.Width, 0.f));
    task.Steps = std::max(p.Segments, 1u);
    return ExecuteElement(r, task, m, p.Vertices ? AttributeDomain::Vertex : AttributeDomain::Edge, linked.Of<&Bevel::Width>());
}
Outcome Run(state::Registry &r, const SmoothVertices &p, uint32_t id, const Masks &m, const Linked &linked) {
    return RepositionBy(r, id, m, PositionEditOp::Smooth, p.Factor, std::clamp(p.Repeat, 1u, 1000u), linked.Of<&SmoothVertices::Factor>(), {.Axes = Axes(p.X, p.Y, p.Z)});
}
Outcome Run(state::Registry &r, const ShrinkFatten &p, uint32_t id, const Masks &m, const Linked &linked) {
    return RepositionBy(r, id, m, PositionEditOp::ShrinkFatten, p.Distance, 1u, linked.Of<&ShrinkFatten::Distance>(), {.Flags = (p.Even ? PositionEditEvenOffset : 0u) | (m.Domain == Element::Face ? PositionEditSelectedFaceNormals : 0u)});
}
Outcome Run(state::Registry &r, const Randomize &p, uint32_t id, const Masks &m, const Linked &linked) {
    return RepositionBy(r, id, m, PositionEditOp::Randomize, p.Amount, 1u, linked.Of<&Randomize::Amount>(), {.Randomize = PositionRandomize{p.Uniform, p.Normal, p.Seed}});
}
Outcome Run(state::Registry &r, const SpaceEvenly &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::SpaceEvenly, std::clamp(p.Factor, 0.f, 1.f), 1u, {.Axes = Axes(p.X, p.Y, p.Z), .Flags = p.Interpolation == EdgeLoopInterpolation::Cubic ? PositionEditCurveCubic : 0u});
}
Outcome Run(state::Registry &r, const RelaxEdgeLoops &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::RelaxEdgeLoops, 1.f, std::min(p.Iterations, 1000u), {.Flags = (p.Interpolation == EdgeLoopInterpolation::Cubic ? PositionEditCurveCubic : 0u) | (p.EvenSpacing ? PositionEditRelaxEven : 0u)});
}
Outcome Run(state::Registry &r, const CurveBetweenSelected &p, uint32_t id, const Masks &m) {
    const auto elevation = p.Elevation == CurveElevation::Raise ? PositionEditCurveRaise : p.Elevation == CurveElevation::Lower ? PositionEditCurveLower :
                                                                                                                                  0u;
    const auto flags = (p.Interpolation == EdgeLoopInterpolation::Cubic ? PositionEditCurveCubic : 0u) | (p.Regular ? PositionEditCurveRegular : 0u) | elevation;
    return Reposition(r, id, m, PositionEditOp::Copy, std::clamp(p.Factor, 0.f, 1.f), 1u, {.Axes = Axes(p.X, p.Y, p.Z), .Flags = flags, .Curve = PositionCurve{p.Extend}});
}
Outcome Run(state::Registry &r, const Circularize &p, uint32_t id, const Masks &m) {
    const auto flags = (p.Regular ? PositionEditCurveRegular : 0u) | (p.Method == CircleFit::Contract ? PositionEditCircleContract : 0u);
    return Reposition(r, id, m, PositionEditOp::Copy, std::clamp(p.Factor, 0.f, 1.f), 1u, {.Axes = Axes(p.X, p.Y, p.Z), .Flags = flags, .Circle = PositionCircle{p.Radius, p.Angle}});
}
Outcome Run(state::Registry &r, const MakePlanarFaces &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::Planar, p.Factor, std::clamp(p.Repeat, 1u, 10000u));
}
Outcome Run(state::Registry &r, const Flatten &p, uint32_t id, const Masks &m) {
    const auto flags = p.Method == FlattenMethod::View ? PositionEditFlattenDirection : p.Method == FlattenMethod::FaceNormals ? PositionEditFlattenNormals :
                                                                                                                                 0u;
    return Reposition(r, id, m, PositionEditOp::Copy, std::clamp(p.Factor, 0.f, 1.f), 1u, {.Axes = Axes(p.X, p.Y, p.Z), .Flags = flags, .Direction = p.ViewNormal}, p.Frame);
}
Outcome Run(state::Registry &r, const ToSphere &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::ToSphere, p.Factor, 1u, {.Center = p.Center, .SphereRadius = p.Radius > 0.f ? std::optional{p.Radius} : std::nullopt}, p.Frame);
}
Outcome Run(state::Registry &r, const PushPull &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::PushPull, p.Distance, 1u, {.Center = p.Center}, p.Frame);
}
Outcome Run(state::Registry &r, const Shear &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::Shear, std::tan(p.Angle), 1u, {.Direction = p.Direction, .Gradient = Cross(p.Direction, p.Normal), .Center = p.Center}, p.Frame);
}
Outcome Run(state::Registry &r, const Warp &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::Warp, p.Angle, 1u, {.Warp = PositionWarp{p.Orientation, p.Center, p.OffsetAngle, p.Min, p.Max, p.AutoRange}}, p.Frame);
}
Outcome Run(state::Registry &r, const Bend &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::Bend, p.Angle, 1u, {.Flags = p.Clamp ? PositionEditBendClamp : 0u, .Bend = PositionBend{p.Orientation, p.Center, p.OffsetAngle, p.Radius}}, p.Frame);
}
Outcome Run(state::Registry &r, const VertexSlide &p, uint32_t id, const Masks &m) {
    const auto factor = p.Clamp ? std::clamp(p.Factor, 0.f, 1.f) : p.Factor;
    return Reposition(r, id, m, PositionEditOp::VertexSlide, factor, 1u, {.Flags = p.Clamp ? 0u : PositionEditSlideUnclamped, .Slide = PositionSlide{p.Direction, p.Even, p.Flipped}}, p.Frame);
}
Outcome Run(state::Registry &r, const EdgeSlide &p, uint32_t id, const Masks &m) {
    const auto factor = p.Clamp ? std::clamp(p.Factor, -1.f, 1.f) : p.Factor;
    return Reposition(r, id, m, PositionEditOp::EdgeSlide, factor, 1u, {.Flags = p.Clamp ? 0u : PositionEditSlideUnclamped, .EdgeSlide = PositionSlide{p.Direction, p.Even, p.Flipped}}, p.Frame);
}
Outcome Run(state::Registry &r, const SnapSymmetry &p, uint32_t id, const Masks &m) {
    if (uint32_t(p.Axis) > 2u) return std::unexpected(NodeFault::Parameter);
    return Reposition(r, id, m, PositionEditOp::SnapSymmetry, std::clamp(p.Factor, 0.f, 1.f), 1u, {.Axes = 1u << uint32_t(p.Axis), .Flags = p.Negative ? PositionEditSymmetryNegative : 0u, .SnapSymmetry = PositionSymmetry{p.Threshold, p.Center}});
}
Outcome Run(state::Registry &r, const Spin &p, uint32_t id, const Masks &m) {
    if (Dot(p.Axis, p.Axis) <= 0.f) return std::unexpected(NodeFault::Parameter);
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    const auto axis = Normalize(p.Axis);
    const auto steps = float(std::max(p.Steps, 1u));
    return ExecutePlanned(r, ExtrudeStepsTask(Mesh{meshes, id}, selection, Element::None, p.Steps, ToMat3(AngleAxis(p.Angle / steps, axis)), axis * (p.Offset / steps), p.Center), selection, m);
}
Outcome Run(state::Registry &r, const ExtrudeRepeat &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    return ExecutePlanned(r, ExtrudeStepsTask(Mesh{meshes, id}, selection, Element::None, p.Steps, mat3{1.f}, p.Offset, vec3{0.f}), selection, m);
}
Outcome Run(state::Registry &r, const Bisect &p, uint32_t id, const Masks &m) {
    if (Dot(p.Normal, p.Normal) <= 0.f) return std::unexpected(NodeFault::Parameter);
    return ExecuteStages(r, BisectTasks(Mesh{r.Resources.get<const MeshStore>(), id}, p.Point, p.Normal, p.ClearInner, p.ClearOuter), m);
}
Outcome Run(state::Registry &r, const Symmetrize &p, uint32_t id, const Masks &m) {
    return ExecuteStages(r, SymmetrizeTasks(Mesh{r.Resources.get<const MeshStore>(), id}, uint8_t(p.Axis), p.Negative), m);
}
Outcome Run(state::Registry &r, const FillHoles &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    return Execute(r, FillHolesTask(meshes, Mesh{meshes, id}, p.Sides), m);
}
Outcome Run(state::Registry &r, ConvexHull, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    return ExecutePlanned(r, ConvexHullTask(meshes, Mesh{meshes, id}, selection), selection, m);
}
Outcome Run(state::Registry &r, const Decimate &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    return ExecutePlanned(r, DecimateTask(meshes, Mesh{meshes, id}, selection, p.Ratio), selection, m);
}
Outcome Run(state::Registry &r, const RecalculateNormals &p, uint32_t id, const Masks &m) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto selection = Handles(meshes, id, m);
    auto faces = std::move(RecalculateFaceFlips(meshes, GetMeshPipelines(r), std::array{id}, std::span{&selection, 1u}, p.Inside).front());
    if (faces.empty()) return id;
    faces.insert(faces.begin(), uint32_t(faces.size()));
    return Execute(r, MeshTopologyTask{.SourceId = id, .Op = MeshTopologyOp::FlipNormals, .List = std::move(faces), .Selection = selection}, m);
}
Outcome Run(state::Registry &r, const SplitNonplanarFaces &p, uint32_t id, const Masks &m) {
    return Execute(r, Task(id, MeshTopologyOp::SplitNonplanarFaces, m, std::clamp(p.Angle, 0.f, std::numbers::pi_v<float>)), m);
}
Outcome Run(state::Registry &r, SplitConcaveFaces, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::SplitConcaveFaces, m), m); }
Outcome Run(state::Registry &r, ConnectVertices, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::ConnectVertices, m), m); }
Outcome Run(state::Registry &r, Fill, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    return ExecutePlanned(r, FillTask(meshes, Mesh{meshes, id}, selection), selection, m);
}
Outcome Run(state::Registry &r, BridgeEdgeLoops, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    return ExecutePlanned(r, BridgeEdgeLoopsTask(meshes, Mesh{meshes, id}, selection), selection, m);
}
Outcome Run(state::Registry &r, const GridFill &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    return ExecutePlanned(r, GridFillTask(meshes, Mesh{meshes, id}, selection, p.Span), selection, m);
}
Outcome Run(state::Registry &r, EdgeRotate, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    return Execute(r, RotateEdgesTask(Mesh{meshes, id}, Handles(meshes, id, m)), m);
}
Outcome Run(state::Registry &r, const BeautifyFaces &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    return Execute(r, BeautifyFaceTask(Mesh{meshes, id}, Handles(meshes, id, m), p.Method == BeautifyMethod::Angle), m);
}
// Repeat on the selected elements that survive each iteration, until an iteration changes nothing.
Outcome Run(state::Registry &r, const Unsubdivide &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    auto selection = Handles(meshes, id, m);
    auto edited = Execute(r, UnsubdivideTask(Mesh{meshes, id}, selection), m);
    for (uint32_t i = 1u; edited && i < std::clamp(p.Iterations, 1u, 1000u); ++i) {
        // A publishing node's later iterations read the selection the previous iteration published, as the viewport operator does.
        if (m.Publish) selection = EditorGeometrySelection(meshes, *edited);
        for (const auto [element, handles] : {std::pair{Element::Vertex, &selection.Vertices}, std::pair{Element::Edge, &selection.Edges}, std::pair{Element::Face, &selection.Faces}}) {
            std::erase_if(*handles, [&](uint32_t handle) { return !meshes.IsLiveElement(*edited, element, ElementHandle{handle}); });
        }
        auto task = UnsubdivideTask(Mesh{meshes, *edited}, selection);
        if (!task) break;
        edited = Execute(r, std::move(task), m);
    }
    return edited;
}
// Cut the ring through the active selected edge, or else the first selected edge.
Outcome Run(state::Registry &r, const LoopCut &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto edges = Handles(meshes, id, m).Edges;
    if (edges.empty()) return std::unexpected(NodeFault::Selection);
    const auto edge = m.Domain == Element::Edge && std::ranges::binary_search(edges, m.Active) ? m.Active : edges.front();
    return Execute(r, LoopCutTask(Mesh{meshes, id}, edge, p.Cuts), m);
}
Outcome Run(state::Registry &r, const Knife &p, uint32_t id, const Masks &m) {
    auto task = KnifeTask(Mesh{r.Resources.get<const MeshStore>(), id}, {}, p.ViewProjection * ToMatrix(p.Frame), p.Extent, p.Start, p.End);
    task.Flags |= TopologyFlagSelectAll;
    return Execute(r, std::move(task), m);
}
Outcome Run(state::Registry &r, const RotateUVs &p, uint32_t id, const Masks &m) { return EditFaceAttributes(r, id, m, false, p.UVSet, p.CounterClockwise ? 1u : 0u); }
Outcome Run(state::Registry &r, const ReverseUVs &p, uint32_t id, const Masks &m) { return EditFaceAttributes(r, id, m, false, p.UVSet, 2u); }
Outcome Run(state::Registry &r, const RotateColors &p, uint32_t id, const Masks &m) { return EditFaceAttributes(r, id, m, true, 0u, p.CounterClockwise ? 1u : 0u); }
Outcome Run(state::Registry &r, ReverseColors, uint32_t id, const Masks &m) { return EditFaceAttributes(r, id, m, true, 0u, 2u); }
// Run the sharpness operation over the selection.
Outcome EditSharpness(state::Registry &r, uint32_t id, const Masks &masks, SelectionSource selection, EditSharpnessOperation operation, bool value, float angle = 0.f) {
    auto &meshes = r.Resources.get<MeshStore>();
    const SharpnessOperationTarget target{id, std::move(selection)};
    if (masks.Drawn != state::Null) {
        *masks.Repaired = true;
        EditDrawnSharpness(r, masks.Drawn, target, operation, value, angle);
        return id;
    }
    mtl::ComputeChain chain{meshes.BufferContext()};
    ExecuteGeometrySharpness(r, chain, std::span{&target, 1u}, operation, value, angle);
    chain.Submit();
    return id;
}
Outcome Run(state::Registry &r, const MarkSharp &p, uint32_t id, const Masks &m) {
    const auto operation = m.Domain == Element::Face ? EditSharpnessOperation::SetSelectedFaces : m.Domain == Element::Edge ? EditSharpnessOperation::SetSelectedEdges :
                                                                                                                              EditSharpnessOperation::SetVertexEdges;
    return EditSharpness(r, id, m, SourceOf(r.Resources.get<const MeshStore>(), id, m), operation, p.Sharp);
}
// Whole-mesh sharpness operations read no selection.
Outcome Run(state::Registry &r, const ShadeSmooth &p, uint32_t id, const Masks &m) {
    return EditSharpness(r, id, m, GeometrySelection{}, p.Smooth ? EditSharpnessOperation::SmoothAll : EditSharpnessOperation::SetAllFaces, !p.Smooth);
}
Outcome Run(state::Registry &r, const ShadeSmoothByAngle &p, uint32_t id, const Masks &m) {
    return EditSharpness(r, id, m, GeometrySelection{}, EditSharpnessOperation::SmoothByAngle, false, p.Angle);
}
Outcome Run(state::Registry &r, const TransformGeometry &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{p.Delta, vec3{}}});
}
// An identity delta leaves the input unchanged.
Outcome Run(state::Registry &r, const TransformElements &p, uint32_t id, const Masks &m) {
    if (p.Delta == Transform{}) return id;
    return Reposition(r, id, m, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{p.Delta, p.Pivot}}, p.Frame);
}
Outcome Run(state::Registry &r, MeshToPoints, uint32_t id, const Masks &) {
    ToPoints(r, id);
    return id;
}
Outcome Run(state::Registry &r, const SetPosition &p, uint32_t id, uint32_t attribute, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    if (attribute == InvalidOffset || meshes.GetAttribute(attribute).Type != AttributeType::Vec3 || meshes.GetAttribute(attribute).Domain != AttributeDomain::Vertex) return std::unexpected(NodeFault::Attribute);
    return Reposition(r, id, m, PositionEditOp::Attribute, 1.f, 1u, {.Flags = p.Offset ? PositionEditAttributeOffset : 0u, .Attribute = meshes.AttributeRef(attribute)});
}

std::string_view NameOf(const AttributeName &name) { return {name.data(), strnlen(name.data(), name.size())}; }

AttributeValue Words(auto value) {
    AttributeValue words{};
    std::memcpy(words.data(), &value, sizeof(value));
    return words;
}
struct LinkedConstant {
    AttributeValue Words;
    AttributeType Type;
};
// Resolve an operand as a named attribute, a uniform value, or absent.
struct Operand {
    uint32_t Attribute{InvalidOffset};
    std::optional<LinkedConstant> Constant;
};

// Replace the named attribute with a filled constant, renamed temporary, or copied attribute.
Outcome Store(MeshStore &meshes, const StoreNamedAttribute &p, uint32_t id, const Operand &value, std::vector<uint32_t> &temporaries) {
    const auto name = NameOf(p.Name);
    const auto attribute = value.Attribute;
    if (attribute == InvalidOffset && !value.Constant) return std::unexpected(NodeFault::Attribute);
    if (name.empty()) return std::unexpected(NodeFault::Parameter);
    if (attribute != InvalidOffset && meshes.GetAttribute(attribute).NameView() == name) return id;
    if (const auto existing = meshes.FindAttribute(id, name); existing != InvalidOffset) meshes.ReleaseAttribute(existing);
    if (attribute == InvalidOffset) return meshes.AddAttribute(id, p.Domain, value.Constant->Type, name, value.Constant->Words) ? Outcome{id} : std::unexpected(NodeFault::Attribute);
    const auto temporary = std::ranges::find(temporaries, attribute);
    if (temporary == temporaries.end()) return meshes.CopyAttribute(attribute, name) ? Outcome{id} : std::unexpected(NodeFault::Attribute);
    temporaries.erase(temporary);
    return meshes.RenameAttribute(attribute, name) ? Outcome{id} : std::unexpected(NodeFault::Attribute);
}

// Replace the named Bool attribute in the domain with the selected elements, or with every element for absent masks.
Outcome Store(MeshStore &meshes, const StoreSelection &p, uint32_t id, const Masks &masks, AttributeDomain domain) {
    const auto name = NameOf(p.Name);
    if (name.empty()) return std::unexpected(NodeFault::Parameter);
    if (const auto existing = meshes.FindAttribute(id, name); existing != InvalidOffset) meshes.ReleaseAttribute(existing);
    const auto attribute = meshes.AddAttribute(id, domain, AttributeType::Bool, name, {masks.In ? 0u : 1u});
    if (!attribute) return std::unexpected(NodeFault::Attribute);
    if (!masks.In) return id;
    const auto element = domain == AttributeDomain::Face ? Element::Face : domain == AttributeDomain::Edge ? Element::Edge :
                                                                                                             Element::Vertex;
    const auto handles = ReadMasks(meshes, id, element, masks.In->Get(element));
    if (handles.empty()) return id;
    const auto count = handles.back() + 1u;
    const auto values = std::make_unique<bool[]>(count);
    for (const auto handle : handles) values[handle] = true;
    meshes.SetAttribute<bool>(*attribute, 0u, std::span<const bool>{values.get(), count});
    return id;
}

// Convert geometry to instance points using constant or linked vertex scale.
Outcome Run(state::Registry &r, const InstanceOnPoints &p, uint32_t id, const Masks &, const Linked &linked) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto rotation = ElementParameter(meshes, linked.Of<&InstanceOnPoints::Rotation>(), AttributeDomain::Vertex, AttributeType::Vec4);
    const auto scale = ElementParameter(meshes, linked.Of<&InstanceOnPoints::Scale>(), AttributeDomain::Vertex, AttributeType::Vec3);
    ToPoints(r, id);
    for (const auto &[name, type, value, attribute] : {std::tuple{InstanceRotation, AttributeType::Vec4, Words(p.Rotation), rotation.value_or(InvalidOffset)}, std::tuple{InstanceScale, AttributeType::Vec3, Words(p.Scale), scale.value_or(InvalidOffset)}}) {
        if (const auto existing = meshes.FindAttribute(id, name); existing != InvalidOffset) meshes.ReleaseAttribute(existing);
        if (attribute != InvalidOffset) (void)meshes.CopyAttribute(attribute, name);
        else (void)meshes.AddAttribute(id, AttributeDomain::Vertex, type, name, value);
    }
    if (!rotation) return std::unexpected(rotation.error());
    return scale ? Outcome{id} : std::unexpected(scale.error());
}

// Allocate zeroed masks through the last owned block, then call fill(domain, owned blocks, masks).
std::array<uint32_t, 3> ZeroedMasks(MeshStore &meshes, std::optional<BufferArena<SelectionBlock>> &arena, uint32_t id, auto &&fill) {
    if (!arena) arena.emplace(meshes.BufferContext(), SlotType::Buffer);
    using D = MeshStore::ElementDomain;
    constexpr std::array domains{D::Vertex, D::Edge, D::Face};
    std::array<uint32_t, 3> first;
    for (uint32_t d = 0u; d < 3u; ++d) {
        const auto owned = meshes.GetBlockList(id, domains[d]).Blocks;
        const auto range = arena->Allocate(owned.empty() ? 1u : std::ranges::max(owned) + 1u);
        const auto masks = arena->GetMutable(range);
        std::ranges::fill(masks, SelectionBlock{});
        fill(d, owned, masks);
        first[d] = range.Offset;
    }
    return first;
}
std::array<uint32_t, 3> UploadSelection(MeshStore &meshes, std::optional<BufferArena<SelectionBlock>> &arena, uint32_t id, const InlineSelection *selection) {
    if (!selection || Empty(*selection)) return EveryElement;
    const std::array lists{&selection->Vertices, &selection->Edges, &selection->Faces};
    constexpr std::array domains{MeshStore::ElementDomain::Vertex, MeshStore::ElementDomain::Edge, MeshStore::ElementDomain::Face};
    return ZeroedMasks(meshes, arena, id, [&](uint32_t d, std::span<const uint32_t>, std::span<SelectionBlock> masks) {
        const auto entries = WithDomain(meshes.Arenas(), domains[d], [&](const auto &elements) { return elements.Entries(DomainSet(meshes.Get(id), domains[d])); });
        ForEachSelectedBlock(*lists[d], [&](uint32_t local, const SelectionBlock &words) {
            if (local < entries.size() && entries[local] != InvalidOffset && entries[local] < masks.size()) masks[entries[local]] = words;
        });
    });
}

struct Operands {
    Operand A, B, C;
};
struct AttributeJob {
    AttributeDomain Domain;
    AttributeType Type;
    MeshPass Pass;
    AttributeNodePushConstants Pc;
};
AttributeInput NamedInput(const MeshStore &meshes, uint32_t attribute) {
    return {.Source = AttributeSource::Named, .Type = meshes.GetAttribute(attribute).Type, .Attribute = meshes.AttributeRef(attribute)};
}
AttributeInput ConstantInput(AttributeType type, auto value) { return {.Source = AttributeSource::Constant, .Type = type, .Value = Words(value)}; }
// Use the linked attribute or uniform value, with the constant as fallback for an absent input.
AttributeInput InputOf(const MeshStore &meshes, const Operand &operand, AttributeInput constant) {
    if (operand.Attribute != InvalidOffset) return NamedInput(meshes, operand.Attribute);
    if (operand.Constant) return {.Source = AttributeSource::Constant, .Type = operand.Constant->Type, .Value = operand.Constant->Words};
    return constant;
}
// Require matching operand domains and supported socket types.
// Constant-only nodes have no domain.
std::optional<AttributeDomain> OperandDomain(const MeshStore &meshes, uint16_t kind, std::initializer_list<std::pair<const Operand *, uint16_t>> operands) {
    const auto sockets = KindSockets(kind);
    std::optional<AttributeDomain> domain;
    for (const auto [operand, socket] : operands) {
        if (operand->Attribute == InvalidOffset) continue;
        const auto &a = meshes.GetAttribute(operand->Attribute);
        if (!(sockets[socket].Types & TypeBit(a.Type)) || (domain && *domain != a.Domain)) return {};
        domain = a.Domain;
    }
    return domain;
}

std::optional<AttributeJob> Prepare(const MeshStore &meshes, Position, Operands) {
    return AttributeJob{AttributeDomain::Vertex, AttributeType::Vec3, MeshPass::AttributeRead, {.A = {.Source = AttributeSource::Positions, .Type = AttributeType::Vec3, .Slot = meshes.Slots().Vertices}}};
}
std::optional<AttributeJob> Prepare(const MeshStore &meshes, const Normal &p, Operands) {
    if (p.Domain != AttributeDomain::Vertex && p.Domain != AttributeDomain::Face) return {};
    const auto &a = meshes.Arenas();
    const auto slot = p.Domain == AttributeDomain::Vertex ? a.BaseVertexNormals.Buffer.Slot : a.BaseFaceNormals.Buffer.Slot;
    return AttributeJob{p.Domain, AttributeType::Vec3, MeshPass::AttributeRead, {.A = {.Source = AttributeSource::Vectors, .Type = AttributeType::Vec3, .Slot = slot}}};
}
std::optional<AttributeJob> Prepare(const MeshStore &, const Index &p, Operands) {
    return AttributeJob{p.Domain, AttributeType::Int, MeshPass::AttributeRead, {.A = {.Source = AttributeSource::Ordinal, .Type = AttributeType::Int}}};
}
std::optional<AttributeJob> Prepare(const MeshStore &meshes, const VectorMath &p, const Operands &in) {
    const bool scale = p.Operation == VectorMathOp::Scale;
    const auto &b = scale ? in.C : in.B;
    const auto domain = OperandDomain(meshes, KindOf<VectorMath>, {{&in.A, 0u}, {&b, uint16_t(scale ? 2u : 1u)}});
    if (!domain) return {};
    const bool scalar = p.Operation == VectorMathOp::Dot || p.Operation == VectorMathOp::Length || p.Operation == VectorMathOp::Distance;
    const auto constant = scale ? ConstantInput(AttributeType::Float, p.Scale) : ConstantInput(AttributeType::Vec3, p.B);
    return AttributeJob{*domain, scalar ? AttributeType::Float : AttributeType::Vec3, MeshPass::AttributeVectorMath, {.A = InputOf(meshes, in.A, ConstantInput(AttributeType::Vec3, vec3{})), .B = InputOf(meshes, b, constant), .Operation = uint32_t(p.Operation)}};
}
std::optional<AttributeJob> Prepare(const MeshStore &meshes, const FloatMath &p, const Operands &in) {
    const auto domain = OperandDomain(meshes, KindOf<FloatMath>, {{&in.A, 0u}, {&in.B, 1u}});
    if (!domain) return {};
    return AttributeJob{*domain, AttributeType::Float, MeshPass::AttributeFloatMath, {.A = InputOf(meshes, in.A, ConstantInput(AttributeType::Float, 0.f)), .B = InputOf(meshes, in.B, ConstantInput(AttributeType::Float, p.B)), .Operation = uint32_t(p.Operation)}};
}
std::optional<AttributeJob> Prepare(const MeshStore &meshes, const Compare &p, const Operands &in) {
    const auto domain = OperandDomain(meshes, KindOf<Compare>, {{&in.A, 0u}, {&in.B, 1u}});
    if (!domain) return {};
    const auto pc = AttributeNodePushConstants{
        .A = InputOf(meshes, in.A, ConstantInput(AttributeType::Float, 0.f)),
        .B = InputOf(meshes, in.B, ConstantInput(AttributeType::Float, p.B)),
        .Operation = uint32_t(p.Operation),
        .Component = uint32_t(p.Component),
        .Epsilon = std::max(p.Epsilon, 0.f),
    };
    return AttributeJob{*domain, AttributeType::Bool, MeshPass::AttributeCompare, pc};
}
std::optional<AttributeJob> Prepare(const MeshStore &meshes, const BooleanMath &p, const Operands &in) {
    const auto domain = OperandDomain(meshes, KindOf<BooleanMath>, {{&in.A, 0u}, {&in.B, 1u}});
    if (!domain) return {};
    return AttributeJob{*domain, AttributeType::Bool, MeshPass::AttributeBooleanMath, {.A = InputOf(meshes, in.A, ConstantInput(AttributeType::Bool, 0u)), .B = InputOf(meshes, in.B, ConstantInput(AttributeType::Bool, uint32_t(p.B))), .Operation = uint32_t(p.Operation)}};
}
template<typename K>
concept Prepared = requires(const MeshStore &meshes, const K &p, const Operands &in) { Prepare(meshes, p, in); };

// Reserve internal names for temporary attributes.
constexpr std::string_view TemporaryPrefix = ".cook.";
std::string TemporaryName(uint32_t node, uint32_t socket) { return std::format("{}{}.{}", TemporaryPrefix, node, socket); }
// Release cook temporaries retained by an edit in its replacement record.
void ReleaseTemporaries(MeshStore &meshes, uint32_t id) {
    std::vector<uint32_t> temporaries;
    meshes.ForEachNamedAttribute(id, [&](uint32_t attribute, const auto &a) {
        if (a.NameView().starts_with(TemporaryPrefix)) temporaries.push_back(attribute);
    });
    for (const auto attribute : temporaries) meshes.ReleaseAttribute(attribute);
}

// Return the linked attribute or constant; return an empty operand for missing inputs.
Operand OperandOf(const GeometryProcedure &procedure, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node, uint32_t input) {
    if (input >= InputCount[procedure.Nodes[node].Kind]) return {};
    const auto link = procedure.Inputs[plan.InputFirst[node] + input];
    if (link.FromNode == NoNode) return {};
    const auto &source = values[plan.SocketFirst[link.FromNode] + link.FromSocket];
    if (source.Type == SocketType::Value) return {.Constant = LinkedConstant{source.Value, AttributeType(std::countr_zero(KindSockets(procedure.Nodes[link.FromNode].Kind)[link.FromSocket].Types))}};
    return {source.Type == SocketType::Attribute ? source.Attribute : InvalidOffset};
}
void AssignField(auto &p, Socket socket, const AttributeValue &words) {
    field::ForEach(p, [&]<size_t I>(auto &member, std::integral_constant<size_t, I>) {
        using F = std::remove_cvref_t<decltype(member)>;
        if constexpr (detail::TypesOf<F>() != 0u) {
            if (socket.Carries(I)) std::memcpy(&member, words.data(), sizeof(member));
        } else if constexpr (field::Walkable<F>) {
            field::ForEach(member, [&]<size_t J>(auto &nested, std::integral_constant<size_t, J>) {
                if constexpr (detail::TypesOf<std::remove_cvref_t<decltype(nested)>>() != 0u) {
                    if (socket.Carries(I, J)) std::memcpy(&nested, words.data(), sizeof(nested));
                }
            });
        }
    });
}
template<typename K> K Parameters(const state::Registry &r, const GeometryProcedure &procedure, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node) {
    const auto [entity, kind] = procedure.Nodes[node];
    auto p = r.get<const K>(entity);
    if constexpr (field::Walkable<K>) {
        const auto [first, end] = ParameterRange(kind);
        for (auto s = first; s < end; ++s) {
            if (const auto value = OperandOf(procedure, plan, values, node, s).Constant) AssignField(p, KindSockets(kind)[s], value->Words);
        }
    }
    return p;
}
Linked LinkedParameters(const GeometryProcedure &procedure, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node) {
    const auto kind = procedure.Nodes[node].Kind;
    const auto sockets = KindSockets(kind);
    const auto [first, end] = ParameterRange(kind);
    Linked linked{kind};
    linked.Attributes.fill(InvalidOffset);
    for (auto s = first; s < end; ++s) {
        if (sockets[s].Type == SocketType::Attribute) linked.Attributes[s] = OperandOf(procedure, plan, values, node, s).Attribute;
    }
    return linked;
}

struct Evaluation {
    std::vector<uint32_t> Temporaries; // The attributes the evaluation added to the record
    std::optional<std::array<uint32_t, 3>> MaskFirst; // The masks of a linked selection, complete in every domain
    bool Failed{}; // A linked selection reads no bool attribute of a selectable domain
};

// Evaluate attribute inputs into temporary attributes and derive selection masks.
// Mark invalid attribute nodes and pass through their first input.
Evaluation EvaluateInputs(state::Registry &r, const GeometryProcedure &procedure, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, std::optional<BufferArena<SelectionBlock>> &arena) {
    auto &meshes = r.Resources.get<MeshStore>();
    auto upstream = Upstream(procedure, plan.InputFirst, node, [&](uint32_t n) { return IsAttributeKind(procedure.Nodes[n].Kind); });
    upstream[node] = false;
    Evaluation evaluation;
    std::vector<AttributeJob> jobs;
    for (const auto n : plan.Order) {
        if (!upstream[n]) continue;
        const auto [entity, kind] = procedure.Nodes[n];
        const Operands in{OperandOf(procedure, plan, values, n, 0u), OperandOf(procedure, plan, values, n, 1u), OperandOf(procedure, plan, values, n, 2u)};
        auto &output = values[plan.SocketFirst[n] + InputCount[kind]];
        output.Type = SocketType::Attribute;
        output.Attribute = in.A.Attribute;
        if (kind == KindOf<NamedAttribute>) {
            const auto &p = r.get<const NamedAttribute>(entity);
            const auto found = meshes.FindAttribute(id, NameOf(p.Name));
            output.Attribute = found != InvalidOffset && meshes.GetAttribute(found).Type == p.Type ? found : InvalidOffset;
            if (output.Attribute == InvalidOffset) r.emplace_or_replace<NodeError>(entity, NodeFault::Attribute);
            continue;
        }
        std::optional<AttributeJob> job;
        VisitKind(kind, [&]<typename K> {
            if constexpr (Prepared<K>) job = Prepare(meshes, Parameters<K>(r, procedure, plan, values, n), in);
        });
        if (!job) {
            r.emplace_or_replace<NodeError>(entity, NodeFault::Attribute);
            continue;
        }
        const auto added = meshes.AddAttribute(id, job->Domain, job->Type, TemporaryName(n, InputCount[kind])).value();
        evaluation.Temporaries.push_back(added);
        output.Attribute = added;
        const auto domain = ElementDomainOf(job->Domain);
        const auto blocks = meshes.GetBlockList(id, domain);
        auto &pc = job->Pc;
        pc.Blocks = blocks.Gpu;
        pc.BlockCount = uint32_t(blocks.Blocks.size());
        pc.MembershipSlot = WithDomain(meshes.Arenas(), domain, [](const auto &arena) { return arena.Blocks.Buffer.Slot; });
        pc.Output = meshes.AttributeRef(added);
        pc.OutputType = job->Type;
        if (pc.BlockCount) jobs.push_back(*job);
    }
    std::vector<AttributeMaskPushConstants> scatters;
    std::array<Range, 3> mask_ranges;
    std::optional<Element> from;
    const auto sockets = KindSockets(procedure.Nodes[node].Kind);
    for (uint32_t s = 0u; s < InputCount[procedure.Nodes[node].Kind]; ++s) {
        if (sockets[s].Type != SocketType::Selection || procedure.Inputs[plan.InputFirst[node] + s].FromNode == NoNode) continue;
        const auto linked = OperandOf(procedure, plan, values, node, s);
        // A linked bool value selects every element, or none through empty masks.
        if (linked.Constant) {
            if (linked.Constant->Words[0] & 1u) {
                evaluation.MaskFirst = EveryElement;
                break;
            }
            evaluation.MaskFirst = ZeroedMasks(meshes, arena, id, [](uint32_t, std::span<const uint32_t>, std::span<SelectionBlock>) {});
            break;
        }
        const auto selection = linked.Attribute;
        const auto *attribute = selection != InvalidOffset ? &meshes.GetAttribute(selection) : nullptr;
        if (!attribute || attribute->Type != AttributeType::Bool || attribute->Domain == AttributeDomain::Corner) {
            evaluation.Failed = true;
            break;
        }
        if (!arena) arena.emplace(meshes.BufferContext(), SlotType::Buffer);
        using D = MeshStore::ElementDomain;
        constexpr std::array domains{D::Vertex, D::Edge, D::Face};
        constexpr std::array elements{Element::Vertex, Element::Edge, Element::Face};
        const auto ref = meshes.AttributeRef(selection);
        for (uint32_t d = 0u; d < 3u; ++d) {
            const auto owned = meshes.GetBlockList(id, domains[d]).Blocks;
            mask_ranges[d] = arena->Allocate(owned.empty() ? 1u : owned.back() + 1u);
            const bool scattered = ElementDomainOf(attribute->Domain) == domains[d];
            if (scattered) from = elements[d];
            const auto [membership, owner] = WithDomain(meshes.Arenas(), domains[d], [&](const auto &elements) { return std::pair{elements.Blocks.Buffer.Slot, DomainSet(meshes.Get(id), domains[d]).Index}; });
            scatters.push_back({.Attribute = scattered ? ref : NamedAttributeRef{}, .MembershipSlot = membership, .Owner = owner, .Masks = MaskBinding({&*arena, mask_ranges[d].Offset}), .Words = mask_ranges[d].Count * MeshElementBlockWords});
        }
        evaluation.MaskFirst = {mask_ranges[0].Offset, mask_ranges[1].Offset, mask_ranges[2].Offset};
    }
    if (jobs.empty() && scatters.empty()) return evaluation;
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto &pipelines = GetMeshPipelines(r);
    for (const auto &job : jobs) chain.Threads(pipelines[job.Pass], job.Pc, job.Pc.BlockCount * MeshElementBlockSize);
    chain.Concurrent([&] {
        for (const auto &pc : scatters) chain.Threads(pipelines[MeshPass::AttributeSelectionMasks], pc, pc.Words);
    });
    if (from) meshes.DeriveSelectionMasks(r, chain, id, {{&*arena, mask_ranges[0].Offset}, {&*arena, mask_ranges[1].Offset}, {&*arena, mask_ranges[2].Offset}}, *from);
    chain.Submit();
    return evaluation;
}

// Evaluate inputs and run the node, releasing temporary attributes after their last use.
// The trailing arguments become the node's Masks Drawn, Repaired and Refresh.
Outcome RunNode(state::Registry &r, const GeometryProcedure &procedure, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, std::optional<BufferArena<SelectionBlock>> &arena, state::Entity drawn = state::Null, bool *repaired = nullptr, MeshStore::Change *refresh = nullptr) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto [entity, kind] = procedure.Nodes[node];
    auto evaluation = EvaluateInputs(r, procedure, plan, values, node, id, arena);
    const auto release = [&] {
        for (const auto attribute : evaluation.Temporaries) meshes.ReleaseAttribute(attribute);
        evaluation.Temporaries.clear();
    };
    if (evaluation.Failed) {
        release();
        return std::unexpected(NodeFault::Attribute);
    }
    const auto linked = LinkedParameters(procedure, plan, values, node);
    const auto sockets = KindSockets(kind);
    const auto first = plan.SocketFirst[node];
    const auto selection_socket = uint32_t(std::ranges::find(sockets, SocketType::Selection, &Socket::Type) - sockets.begin());
    Masks masks;
    if (selection_socket < sockets.size()) {
        auto &selection = values[first + selection_socket];
        selection.Type = SocketType::Selection;
        const auto *inline_selection = r.try_get<const InlineSelection>(entity);
        selection.MaskFirst = evaluation.MaskFirst ? *evaluation.MaskFirst : UploadSelection(meshes, arena, id, inline_selection);
        if (!evaluation.MaskFirst && inline_selection) {
            masks.Domain = inline_selection->Domain;
            masks.Active = inline_selection->Active;
        }
        if (const auto &f = selection.MaskFirst; f != EveryElement) masks.In = GeometryMasks{{&*arena, f[0]}, {&*arena, f[1]}, {&*arena, f[2]}};
    }
    masks.Publish = r.all_of<PublishSelection>(entity);
    masks.Drawn = drawn;
    masks.Repaired = repaired;
    masks.Refresh = refresh;
    PublishInput(r, id, masks);
    Outcome edited = id;
    VisitKind(kind, [&]<typename K> {
        const auto p = Parameters<K>(r, procedure, plan, values, node);
        if constexpr (std::same_as<K, StoreNamedAttribute>) {
            edited = Store(meshes, p, id, OperandOf(procedure, plan, values, node, 1u), evaluation.Temporaries);
        } else if constexpr (std::same_as<K, StoreSelection>) {
            const auto linked = OperandOf(procedure, plan, values, node, 1u).Attribute;
            const auto domain = linked != InvalidOffset ? meshes.GetAttribute(linked).Domain : masks.Domain == Element::Face ? AttributeDomain::Face :
                masks.Domain == Element::Edge                                                                                ? AttributeDomain::Edge :
                                                                                                                               AttributeDomain::Vertex;
            edited = Store(meshes, p, id, masks, domain);
        } else if constexpr (std::same_as<K, SetPosition>) {
            auto position = OperandOf(procedure, plan, values, node, 1u);
            if (position.Attribute == InvalidOffset && position.Constant) {
                position.Attribute = meshes.AddAttribute(id, AttributeDomain::Vertex, position.Constant->Type, TemporaryName(node, 1u), position.Constant->Words).value();
                evaluation.Temporaries.push_back(position.Attribute);
            }
            edited = Run(r, p, id, position.Attribute, masks);
        } else if constexpr (requires(const K &p, const Masks &m) { Run(r, p, id, m, linked); }) {
            if (!linked.Any()) release();
            edited = Run(r, p, id, masks, linked);
            if (linked.Any() && edited && *edited != id) ReleaseTemporaries(meshes, *edited);
        } else if constexpr (requires(const K &p, const Masks &m) { Run(r, p, id, m); }) {
            release();
            edited = Run(r, p, id, masks);
        }
    });
    release();
    return edited;
}

// Derive generated shading before downstream nodes read it.
uint32_t Generate(state::Registry &r, const Generator auto &shape) {
    const auto id = CreateMesh(r, {.Data = primitive::CreateMesh(shape), .FlatShaded = true}).StoreId;
    ShadeNewMeshes(r);
    return id;
}

// Return the entity whose mesh record the referenced object draws, excluding the current procedure mesh.
state::Entity ReadMesh(const state::Registry &r, state::Entity mesh_entity, const ObjectInfo &p) {
    const auto *instance = r.valid(p.Object) ? r.try_get<const Instance>(p.Object) : nullptr;
    return instance && instance->Entity != mesh_entity && DrawsMesh(r, instance->Entity) ? instance->Entity : state::Null;
}

// Each point's instance transform, in vertex order.
std::vector<Transform> InstanceTransforms(const MeshStore &meshes, uint32_t points) {
    const Mesh mesh{meshes, points};
    const auto rotations = meshes.AttributeView<vec4>(meshes.FindAttribute(points, InstanceRotation));
    const auto scales = meshes.AttributeView<vec3>(meshes.FindAttribute(points, InstanceScale));
    std::vector<Transform> transforms;
    transforms.reserve(mesh.VertexCount());
    for (const auto v : mesh.vertices()) transforms.push_back({mesh.GetPosition(v), quat{rotations[*v]}, scales[*v]});
    return transforms;
}

// Realize instances into one mesh; return nullopt for an empty result.
std::optional<uint32_t> Realize(state::Registry &r, uint32_t points, uint32_t instance) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto transforms = InstanceTransforms(meshes, points);
    if (transforms.empty()) return {};
    const auto clones = CloneRecords(r, std::vector(transforms.size(), instance));
    for (size_t i = 0u; i < clones.size(); ++i) {
        const PositionOperationTarget target{.StoreId = clones[i], .Selection = AllHandles(Mesh{meshes, clones[i]})};
        ExecuteGeometryPositions(r, std::span{&target, 1u}, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = transforms[i]}});
    }
    for (size_t i = 1u; i < clones.size(); ++i) meshes.Join(r, clones.front(), clones[i]);
    meshes.Release(std::span{clones}.subspan(1u));
    return clones.front();
}

// Publish instance entities in transform order, retaining existing entities while the count permits.
void PublishInstances(state::Registry &r, state::Entity mesh_entity, state::Entity instanced, std::span<const Transform> transforms) {
    const auto *current = r.try_get<const ProcedureInstances>(mesh_entity);
    if (!current && transforms.empty()) return;
    const auto previous = current ? current->Entities : std::vector<state::Entity>{};
    std::vector<state::Entity> objects;
    if (!transforms.empty()) {
        for (const auto [e, instance] : std::as_const(r).view<const Instance>().each()) {
            if (instance.Entity == mesh_entity) objects.push_back(e);
        }
    }
    std::ranges::sort(objects);
    const auto count = uint32_t(transforms.size());
    std::vector<state::Entity> published(objects.size() * count, state::Null), extras;
    std::vector<uint32_t> kept(objects.size(), 0u);
    for (const auto e : previous) {
        if (!r.valid(e)) continue;
        const auto parent = ParentOrNull(r, e);
        const auto object = std::ranges::lower_bound(objects, parent);
        const auto o = uint32_t(object - objects.begin());
        if (object != objects.end() && *object == parent && kept[o] < count) published[o * count + kept[o]++] = e;
        else extras.push_back(e);
    }
    ClearParents(r, extras);
    r.destroy(extras);
    for (uint32_t o = 0u; o < objects.size(); ++o) {
        for (uint32_t i = 0u; i < count; ++i) {
            auto &e = published[o * count + i];
            if (e == state::Null) {
                e = r.create();
                r.emplace<Instance>(e, instanced);
                r.emplace<Transform>(e, transforms[i]);
                SetParent(r, e, objects[o]);
                EmplaceObject(r, e, ObjectType::Mesh);
                ApplyVisibility(r, e);
                continue;
            }
            if (r.get<const Instance>(e).Entity != instanced) r.replace<Instance>(e, instanced);
            if (r.get<const Transform>(e) != transforms[i]) r.replace<Transform>(e, transforms[i]);
        }
    }
    if (published.empty()) r.remove<ProcedureInstances>(mesh_entity);
    else if (previous != published) r.emplace_or_replace<ProcedureInstances>(mesh_entity, std::move(published));
}

// Assign the instance record to an owned mesh entity, creating or removing the entity as needed.
state::Entity OwnMesh(state::Registry &r, state::Entity mesh_entity, uint32_t record) {
    const auto *owned = r.try_get<const ProcedureOwnedMesh>(mesh_entity);
    if (record == InvalidOffset) {
        r.remove<ProcedureOwnedMesh>(mesh_entity);
        return state::Null;
    }
    // The owned mesh entity is gone when its published instances were deleted by hand.
    const bool kept = owned && r.valid(owned->MeshEntity);
    const auto e = kept ? owned->MeshEntity : r.create();
    if (!kept) {
        r.emplace<MeshHandle>(e, record);
        r.emplace_or_replace<ProcedureOwnedMesh>(mesh_entity, e);
    } else if (const auto held = r.get<const MeshHandle>(e).StoreId; held != record) {
        r.Resources.get<MeshStore>().Release(held);
        r.replace<MeshHandle>(e, record);
    }
    RefreshMeshBinding(r, record);
    r.emplace_or_replace<MeshGeometryDirty>(e);
    return e;
}

// Release a record the cook replaced, first keeping the mesh storage as the state of each Source node whose pending state refers to it.
void ReleaseReplaced(state::Registry &r, uint32_t record) {
    if (record == InvalidOffset) return;
    FinishPendingSourceStates(r, record);
    r.Resources.get<MeshStore>().Release(record);
}

// A new record holding a Source node's geometry, without selection, or none when no kept or recorded state holds it.
// An empty source emits an empty record, and a pending state's geometry is its record as it stands.
std::optional<uint32_t> MaterializeSource(state::Registry &r, const SourceState &source) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (source.StoreId == InvalidOffset) {
        const auto id = CreateMesh(r, {}).StoreId;
        ShadeNewMeshes(r);
        return id;
    }
    if (source.State == store::Hash128{}) {
        if (!meshes.TryGet(source.StoreId)) return std::nullopt;
        const auto id = CloneRecord(r, source.StoreId);
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.ResetEditState(r, chain, id, source.StoreId);
        chain.Submit();
        RefreshMeshBinding(r, id);
        return id;
    }
    const auto id = meshes.CloneStored(source.State, source.StoreId);
    if (!id) return std::nullopt;
    meshes.RebuildSelectionIndex(std::array{*id});
    mtl::ComputeChain chain{meshes.BufferContext()};
    meshes.EnsureSelectionState(r, chain, std::array{*id});
    chain.Submit();
    RefreshMeshBinding(r, *id);
    return id;
}

// A new record of the display path's root, which the cook owns: its Source node's geometry or its generated primitive.
std::optional<uint32_t> MaterializeRoot(state::Registry &r, NodeRecord root) {
    if (root.Kind == KindOf<Source>) return MaterializeSource(r, r.get<const SourceState>(root.Entity));
    std::optional<uint32_t> id;
    VisitKind(root.Kind, [&]<typename K> {
        if constexpr (Generator<K>) id = Generate(r, r.get<const K>(root.Entity));
    });
    return id;
}

// Copy the primitive material table of the record a cook replaces, which material assignments write, into its replacement.
void CarryMaterials(MeshStore &meshes, uint32_t from, uint32_t to) {
    if (from == InvalidOffset || from == to || !meshes.TryGet(from)) return;
    const auto stored = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(from).PrimitiveMaterials);
    const std::vector<uint32_t> materials{stored.begin(), stored.end()};
    const auto replaced = meshes.EditPrimitiveMaterials(to);
    std::copy_n(materials.begin(), std::min(materials.size(), replaced.size()), replaced.begin());
}

// Whether the display path is a chain its record holds node by node: its root is a Source node or a generator with unlinked parameters, no link reaches a path node except the main link from the node before it, and no path node after the root feeds anything but the next path node.
bool Chained(const GeometryProcedure &procedure, const CookPlan &plan, std::span<const uint32_t> path) {
    if (path.empty()) return false;
    const auto root = procedure.Nodes[path.front()].Kind;
    bool rooted = root == KindOf<Source>;
    VisitKind(root, [&]<typename K> { rooted |= Generator<K>; });
    if (!rooted) return false;
    for (const auto node : path.subspan(1u)) {
        const auto kind = procedure.Nodes[node].Kind;
        if (kind == KindOf<InstanceOnPoints> || kind == KindOf<RealizeInstances>) return false;
    }
    std::vector<uint32_t> position(procedure.Nodes.size(), NoNode);
    for (uint32_t i = 0u; i < path.size(); ++i) position[path[i]] = i;
    for (uint32_t node = 0u; node < procedure.Nodes.size(); ++node) {
        for (uint32_t input = 0u; input < InputCount[procedure.Nodes[node].Kind]; ++input) {
            const auto from = procedure.Inputs[plan.InputFirst[node] + input].FromNode;
            if (from == NoNode) continue;
            const bool path_link = input == 0u && position[from] != NoNode && position[node] == position[from] + 1u;
            if ((position[node] != NoNode || (position[from] != NoNode && position[from] != 0u)) && !path_link) return false;
        }
    }
    return true;
}

// The record after the display path's nodes from `first` edit it in place, each keeping the mesh storage state before it as its edited input, and whether its render records stayed current and any node faulted.
struct PathRun {
    uint32_t Record;
    bool Rendered, Faulted;
};
// The positions of `restored`, a restore's change to the record, refresh with the first drawn position edit or after the path.
PathRun RunPath(state::Registry &r, state::Entity mesh_entity, const GeometryProcedure &procedure, const CookPlan &plan, std::span<const uint32_t> path, size_t first, uint32_t record, bool rendered, std::optional<BufferArena<SelectionBlock>> &arena, MeshStore::Change &restored) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (first < path.size()) FinishPendingSourceStates(r, record);
    std::vector<SocketValue> values(plan.SocketFirst.back());
    bool faulted = false;
    const auto refresh_restored = [&] {
        if (restored.VertexRanges.empty() && std::ranges::all_of(restored.Blocks, [](const auto &blocks) { return blocks.empty(); })) return;
        mtl::ComputeChain chain{meshes.BufferContext()};
        RefreshEditedPositions(r, chain, std::array{MeshVertexChanges{mesh_entity, restored.VertexRanges}}, false);
        PublishRestored(r, chain, record, restored);
        chain.Submit();
    };
    for (auto i = first; i < path.size(); ++i) {
        const auto node = path[i];
        const auto [entity, kind] = procedure.Nodes[node];
        r.remove<NodeError>(entity);
        const EditedInput input{procedure.Nodes[path[i - 1u]].Entity, meshes.KeepState(), record};
        if (const auto *held = r.try_get<const EditedInput>(entity); !held || *held != input) r.emplace_or_replace<EditedInput>(entity, input);
        if (auto &pass = r.Resources.get<CookPass>(); pass.Mesh == mesh_entity && !std::ranges::contains(pass.Exact, input.State)) pass.Exact.push_back(input.State);
        // A chained Join or Boolean has no second input and passes its first through.
        if (kind == KindOf<Join> || kind == KindOf<Boolean>) continue;
        bool repaired = false;
        if (!rendered) refresh_restored();
        const auto edited = RunNode(r, procedure, plan, values, node, record, arena, rendered ? mesh_entity : state::Null, &repaired, &restored);
        if (!edited) {
            r.emplace<NodeError>(entity, edited.error());
            faulted = true;
        } else if (*edited != record) {
            restored = {};
            ReleaseReplaced(r, record);
            record = *edited;
            rendered = false;
        } else if (rendered && !repaired) {
            rendered = false;
        }
    }
    refresh_restored();
    return {record, rendered, faulted};
}

// Each element domain's set revision, which changes with the record's topology.
std::array<uint32_t, 4> TopologyRevisions(const MeshStore &meshes, uint32_t record) {
    using D = MeshStore::ElementDomain;
    std::array<uint32_t, 4> revisions{};
    for (uint32_t i = 0u; const auto domain : {D::Vertex, D::Halfedge, D::Edge, D::Face}) {
        revisions[i++] = WithDomain(meshes.Arenas(), domain, [&](const auto &elements) {
            const auto set = DomainSet(meshes.Get(record), domain);
            return set ? elements.Set(set).Revision : 0u;
        });
    }
    return revisions;
}

// Publish a cooked record the mesh draws, and request render records for a record without current ones.
// A cook that changed topology derives selection from the edit domain.
void PublishCooked(state::Registry &r, state::Entity mesh_entity, uint32_t record, state::Entity tip, bool rendered, bool topology) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto *live = r.try_get<const EvaluatedMesh>(mesh_entity);
    if (!live || live->StoreId != record || live->Tip != tip) r.emplace_or_replace<EvaluatedMesh>(mesh_entity, record, tip);
    RefreshMeshBinding(r, record);
    const auto &cooked = meshes.Get(record);
    const bool ready = rendered && cooked.RenderTopologies != 0u;
    if (ready) {
        if (meshes.HasDirtyPositions(cooked)) r.emplace_or_replace<MeshletPositionsDirty>(mesh_entity);
        if (meshes.HasDirtyGroups(cooked)) r.emplace_or_replace<ClusterLodDirty>(mesh_entity);
        if (!meshes.ClusterGroupCount(cooked)) r.emplace_or_replace<ClusterLodDemand>(mesh_entity);
    }
    r.emplace_or_replace<MeshGeometryDirty>(mesh_entity, topology ? EditSelectionAfter::Derive : EditSelectionAfter::Keep, ready);
}

// Cook a chained display path into the mesh's record.
// The record keeps the output of the path nodes up to its tip, and the nodes after the first changed one edit it in place.
// A changed node the record holds the output of restores the record to the state its edited input refers to, or else the path cooks from a new record of its root.
// The whole mesh storage restores while only this mesh's cooks wrote it since the state was kept, and the record alone restores otherwise, each followed by the same nodes.
// A copied node keeps the edited input its original holds, and the components it was created with leave it unchanged.
void CookChain(state::Registry &r, state::Entity mesh_entity, const GeometryProcedure &procedure, const CookPlan &plan, std::span<const uint32_t> path, std::optional<BufferArena<SelectionBlock>> &arena, const state::DirtySet &changed, const state::DirtySet &created) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto *live = r.try_get<const EvaluatedMesh>(mesh_entity);
    const auto prior = live && meshes.TryGet(live->StoreId) ? live->StoreId : InvalidOffset;
    // The path nodes the record holds the output of, from the root.
    std::vector<state::Entity> chain;
    if (prior != InvalidOffset && live->Tip != state::Null) {
        for (auto e = live->Tip;;) {
            chain.push_back(e);
            const auto *input = r.valid(e) ? r.try_get<const EditedInput>(e) : nullptr;
            if (!input) break;
            e = input->Input;
        }
        std::ranges::reverse(chain);
    }
    const auto entity_at = [&](size_t i) { return procedure.Nodes[path[i]].Entity; };
    const auto edited = [&](state::Entity e) { return changed.contains(e) && !created.contains(e); };
    size_t held = 0u;
    if (!chain.empty() && chain.front() == entity_at(0u) && !edited(chain.front())) {
        held = 1u;
        while (held < chain.size() && held < path.size() && chain[held] == entity_at(held) && !edited(chain[held])) ++held;
    }
    auto record = prior;
    // The record the result replaces, which a restore returns to the state before the nodes that replaced it.
    auto replaced = prior;
    bool rendered = prior != InvalidOffset && meshes.Get(prior).RenderTopologies != 0u;
    MeshStore::Change restored{};
    if (held == 0u || held < chain.size()) {
        const auto *input = held ? r.try_get<const EditedInput>(chain[held]) : nullptr;
        bool restored_record = false;
        if (input && input->StoreId == prior && input->State != store::Hash128{}) {
            WaitForRender(r);
            if (std::ranges::contains(r.Resources.get<const CookPass>().Exact, input->State) && meshes.RestoreKeptState(input->State)) {
                restored = ReconcileRestoredStore(r, mesh_entity);
                restored_record = true;
            } else if (const auto change = meshes.RestoreRecord(input->State, prior)) {
                restored = ReconcileRestoredRecord(r, *change, mesh_entity);
                restored_record = true;
            }
        }
        if (restored_record) {
            replaced = InvalidOffset;
            rendered = meshes.Get(record).RenderTopologies != 0u;
        } else {
            const auto root = MaterializeRoot(r, procedure.Nodes[path.front()]);
            if (!root) {
                r.emplace_or_replace<NodeError>(entity_at(0u), NodeFault::NoMesh);
                return;
            }
            record = *root;
            r.remove<NodeError>(entity_at(0u));
            CarryMaterials(meshes, prior, record);
            rendered = false;
            held = 1u;
        }
    } else if (held == path.size()) {
        return;
    }
    const auto topology = TopologyRevisions(meshes, record);
    const auto run = RunPath(r, mesh_entity, procedure, plan, path, held, record, rendered, arena, restored);
    // A run that replaced its own starting record released it.
    if (replaced != InvalidOffset && replaced != record && meshes.TryGet(replaced)) ReleaseReplaced(r, replaced);
    OwnMesh(r, mesh_entity, InvalidOffset);
    PublishInstances(r, mesh_entity, state::Null, {});
    PublishCooked(r, mesh_entity, run.Record, run.Faulted ? state::Null : entity_at(path.size() - 1u), run.Rendered, run.Record != record || TopologyRevisions(meshes, run.Record) != topology);
}

// Evaluate nodes in dependency order and publish the displayed node's result.
// Mark failed nodes and pass their inputs through.
// A chained display path edits the mesh's record in place, and other procedures evaluate into a new record, empty when no node is displayed or the displayed node has no result.
// Release intermediate records.
void Evaluate(state::Registry &r, state::Entity mesh_entity, std::optional<BufferArena<SelectionBlock>> &arena, const state::DirtySet &changed, const state::DirtySet &created_nodes) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto &procedure = r.get<const GeometryProcedure>(mesh_entity);
    const auto &plan = r.get<const CookPlan>(mesh_entity);
    // Track the first-input path, which carries the record the mesh draws from the root to the displayed node.
    const auto display = procedure.DisplayNode;
    std::vector<bool> on_path(procedure.Nodes.size());
    std::vector<uint32_t> path;
    for (auto node = display; node != NoNode && !on_path[node];) {
        on_path[node] = true;
        path.push_back(node);
        const auto kind = procedure.Nodes[node].Kind;
        node = InputCount[kind] && KindSockets(kind).front().Type == SocketType::Geometry ? procedure.Inputs[plan.InputFirst[node]].FromNode : NoNode;
    }
    std::ranges::reverse(path);
    if (Chained(procedure, plan, path)) return CookChain(r, mesh_entity, procedure, plan, path, arena, changed, created_nodes);
    // The record the mesh drew, which the cook releases once the result replaces it.
    const auto prior = r.all_of<EvaluatedMesh>(mesh_entity) ? r.get<const EvaluatedMesh>(mesh_entity).StoreId : InvalidOffset;
    // Evaluate only nodes reachable from the displayed node.
    const auto reached = display != NoNode ? Upstream(procedure, plan.InputFirst, display, [](uint32_t) { return true; }) : std::vector<bool>(procedure.Nodes.size());
    auto remaining = plan.Consumers;
    for (uint32_t node = 0u; node < procedure.Nodes.size(); ++node) {
        if (reached[node]) continue;
        for (uint32_t i = 0u; i < InputCount[procedure.Nodes[node].Kind]; ++i) {
            if (const auto link = procedure.Inputs[plan.InputFirst[node] + i]; link.FromNode != NoNode) --remaining[plan.SocketFirst[link.FromNode] + link.FromSocket];
        }
    }
    // The display reads the displayed node's result after every node, so the result's other consumers clone it.
    const auto result_socket = display != NoNode ? ResultSocket(procedure.Nodes[display].Kind) : std::optional<uint16_t>{};
    if (result_socket) ++remaining[plan.SocketFirst[display] + *result_socket];
    // Release temporary cook records at completion and retain the result.
    std::vector<uint32_t> created;
    std::vector<SocketValue> values(plan.SocketFirst.back());
    const auto slot_of = [&](uint32_t node, uint32_t input) {
        const auto link = procedure.Inputs[plan.InputFirst[node] + input];
        return plan.SocketFirst[link.FromNode] + link.FromSocket;
    };
    // Clone borrowed records before editing and owned records before additional readers.
    // Preserve the result-path record by cloning its producer instead.
    const auto take = [&](uint32_t node, uint32_t input) -> SocketValue {
        auto &taken = values[plan.SocketFirst[node] + input];
        if (procedure.Inputs[plan.InputFirst[node] + input].FromNode == NoNode) return taken;
        const auto slot = slot_of(node, input);
        taken = values[slot];
        const bool last = --remaining[slot] == 0u;
        if (taken.Record == InvalidOffset || (taken.Borrowed() ? KindSockets(procedure.Nodes[node].Kind)[input].Reads : last)) return taken;
        const auto clone = created.emplace_back(CloneRecord(r, taken.Record));
        if (taken.Borrowed() || !on_path[node] || input) {
            taken.MeshEntity = state::Null;
            taken.Record = clone;
            return taken;
        }
        values[slot].Record = clone;
        return taken;
    };
    const auto disown = [&](uint32_t record) { std::erase(created, record); };
    // Release a consumed record only when owned by this cook; borrowed records remain owned by their mesh entities.
    const auto consume = [&](const SocketValue &value) {
        if (value.Borrowed()) return;
        meshes.Release(value.Record);
        disown(value.Record);
    };
    uint32_t result = InvalidOffset;
    std::optional<SocketValue> published; // The instances the result reads
    for (const auto node : plan.Order) {
        const auto [entity, kind] = procedure.Nodes[node];
        const auto first = plan.SocketFirst[node];
        r.remove<NodeError>(entity);
        if (!reached[node] || IsAttributeKind(kind)) continue;
        if (kind == KindOf<Source>) {
            // A pending state's record is still the source, which its readers borrow.
            const auto &source = r.get<const SourceState>(entity);
            if (source.State == store::Hash128{} && source.StoreId != InvalidOffset && meshes.TryGet(source.StoreId)) values[first] = {.MeshEntity = mesh_entity, .Record = source.StoreId};
            else if (const auto materialized = MaterializeSource(r, source)) values[first].Record = created.emplace_back(*materialized);
            else r.emplace<NodeError>(entity, NodeFault::NoMesh);
            continue;
        }
        if (kind == KindOf<ObjectInfo>) {
            const auto read_mesh = ReadMesh(r, mesh_entity, r.get<const ObjectInfo>(entity));
            if (read_mesh != state::Null) values[first] = {.MeshEntity = read_mesh, .Record = *DrawnStoreId(r, read_mesh)};
            else r.emplace<NodeError>(entity, NodeFault::NoMesh);
            continue;
        }
        bool generated = false;
        VisitKind(kind, [&]<typename K> {
            if constexpr (Generator<K>) {
                values[first + InputCount[kind]].Record = created.emplace_back(Generate(r, Parameters<K>(r, procedure, plan, values, node)));
                generated = true;
            } else if constexpr (ValueKind<K>) {
                values[first] = {.Type = SocketType::Value, .Value = Words(r.get<const K>(entity).Value)};
                generated = true;
            }
        });
        if (generated) continue;
        const auto in = take(node, 0u);
        const auto input = in.Record;
        if (kind == KindOf<Join>) {
            const auto second = take(node, 1u);
            if (input != InvalidOffset && second.Record != InvalidOffset) {
                meshes.Join(r, input, second.Record);
                consume(second);
            }
            values[first + 2u] = input != InvalidOffset ? in : second;
            continue;
        }
        if (kind == KindOf<Boolean>) {
            const auto second = take(node, 1u);
            const auto p = Parameters<Boolean>(r, procedure, plan, values, node);
            // Map Mesh 2 into Mesh 1's space through both frames, which must keep Mesh 2 unmirrored.
            const auto to_first = Inverse(ToMatrix(p.Frame1)) * ToMatrix(p.Frame2);
            const auto rotation = ToMat3(to_first);
            const bool linked = input != InvalidOffset && second.Record != InvalidOffset, oriented = Determinant(rotation) > 0.f;
            const auto task = linked && oriented ? BooleanTask(Mesh{meshes, input}, Mesh{meshes, second.Record}, p.Operation, rotation, vec3(to_first[3])) : std::nullopt;
            const auto combined = task ? ExecuteGeometryBoolean(r, *task) : std::nullopt;
            auto &output = values[first + InputCount[kind]];
            if (combined) {
                created.push_back(*combined);
                output.Record = *combined;
            } else {
                if (linked) r.emplace<NodeError>(entity, !oriented ? NodeFault::Parameter : !task ? NodeFault::NotClosed :
                                                                                                    NodeFault::Empty);
                output = in;
            }
            continue;
        }
        if (kind == KindOf<InstanceOnPoints>) {
            const auto instance = take(node, 1u);
            auto &output = values[first + InputCount[kind]];
            // An unlinked Instance input faults the node, which passes its points through as geometry, as a failed placement does.
            output = in;
            if (input == InvalidOffset) continue;
            if (instance.Record == InvalidOffset) {
                r.emplace<NodeError>(entity, NodeFault::NoMesh);
                continue;
            }
            if (const auto placed = RunNode(r, procedure, plan, values, node, input, arena)) {
                output.Type = SocketType::Instances;
                output.MeshEntity = instance.MeshEntity;
                output.Instance = instance.Record;
            } else {
                r.emplace<NodeError>(entity, placed.error());
            }
            continue;
        }
        if (kind == KindOf<RealizeInstances>) {
            const auto realized = input != InvalidOffset && in.Instance != InvalidOffset ? Realize(r, input, in.Instance) : std::nullopt;
            if (realized) {
                consume(in);
                // Release the instance record after its last instance-list reader is removed.
                if (remaining[slot_of(node, 0u)] == 0u) consume({.MeshEntity = in.MeshEntity, .Record = in.Instance});
                created.push_back(*realized);
            }
            values[first + 1u].Record = realized.value_or(input);
            continue;
        }
        const auto edited = input != InvalidOffset ? RunNode(r, procedure, plan, values, node, input, arena) : Outcome{input};
        if (!edited) r.emplace<NodeError>(entity, edited.error());
        values[first + InputCount[kind]].Record = edited.value_or(input);
    }
    if (result_socket) {
        // The result is this cook's record, so a borrowed record is cloned.
        const auto &value = values[plan.SocketFirst[display] + *result_socket];
        if (value.Record != InvalidOffset && (value.Type == SocketType::Geometry || value.Type == SocketType::Instances)) {
            result = value.Borrowed() ? created.emplace_back(CloneRecord(r, value.Record)) : value.Record;
            if (value.Type == SocketType::Instances) published = value;
        }
    }
    std::vector<Transform> transforms;
    if (published && result != InvalidOffset) {
        transforms = InstanceTransforms(meshes, result);
        Empty(r, result);
    }
    // Instances of this mesh's pending source place a clone, since this mesh draws the points.
    if (!transforms.empty() && published->MeshEntity == mesh_entity) {
        published->MeshEntity = state::Null;
        published->Instance = CloneRecord(r, published->Instance);
    }
    const bool owns = !transforms.empty() && published->MeshEntity == state::Null;
    if (owns) disown(published->Instance);
    if (result == InvalidOffset) {
        result = CreateMesh(r, {}).StoreId;
        ShadeNewMeshes(r);
    }
    std::erase(created, result);
    meshes.Release(created);
    CarryMaterials(meshes, prior, result);
    if (prior != InvalidOffset && prior != result && meshes.TryGet(prior)) ReleaseReplaced(r, prior);
    const auto owned = OwnMesh(r, mesh_entity, owns ? published->Instance : InvalidOffset);
    PublishInstances(r, mesh_entity, owns ? owned : published ? published->MeshEntity :
                                                                state::Null,
                     transforms);
    PublishCooked(r, mesh_entity, result, state::Null, false, true);
}

// Select the active node's inline selection on the mesh's record while the node reads the displayed node, so the viewport shows the elements the node edits and a capture replaces them.
// An empty inline selection selects every element of its domain.
void ShowActiveSelection(state::Registry &r, state::Entity mesh_entity, std::optional<BufferArena<SelectionBlock>> &arena) {
    const auto &procedure = r.get<const GeometryProcedure>(mesh_entity);
    const auto active = procedure.ActiveNode, display = procedure.DisplayNode;
    if (active == NoNode || display == NoNode || active == display) return;
    const auto [entity, kind] = procedure.Nodes[active];
    const auto sockets = KindSockets(kind);
    const auto selection_socket = uint32_t(std::ranges::find(sockets, SocketType::Selection, &Socket::Type) - sockets.begin());
    if (selection_socket == sockets.size() || sockets.front().Type != SocketType::Geometry) return;
    const auto first = InputFirst(procedure, active);
    if (procedure.Inputs[first + selection_socket].FromNode != NoNode) return;
    const auto [from, socket] = procedure.Inputs[first];
    if (from != display || socket != ResultSocket(procedure.Nodes[display].Kind)) return;
    auto &meshes = r.Resources.get<MeshStore>();
    const auto *live = r.try_get<const EvaluatedMesh>(mesh_entity);
    if (!live || !meshes.TryGet(live->StoreId)) return;
    const auto *selection = r.try_get<const InlineSelection>(entity);
    if (!selection || Empty(*selection)) {
        ApplyEditSelectionCommand(r, std::array{ElementRange{mesh_entity, 0u}}, selection ? selection->Domain : Element::Vertex, EditSelectionOperation::Fill);
    } else {
        const auto masks = UploadSelection(meshes, arena, live->StoreId, selection);
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.SelectMasks(r, chain, live->StoreId, {{&*arena, masks[0]}, {&*arena, masks[1]}, {&*arena, masks[2]}});
        chain.Submit();
    }
    if (const auto handle = selection ? selection->Active : InvalidOffset; handle != InvalidOffset) r.emplace_or_replace<MeshActiveElement>(mesh_entity, handle);
    else r.remove<MeshActiveElement>(mesh_entity);
}

// Order nodes after their dependencies, preserving authoring order among ready nodes.
CookPlan Plan(const GeometryProcedure &procedure) {
    const auto count = uint32_t(procedure.Nodes.size());
    CookPlan plan{.InputFirst = InputFirsts(procedure), .SocketFirst = std::vector(count + 1u, 0u)};
    for (uint32_t n = 0u; n < count; ++n) plan.SocketFirst[n + 1u] = plan.SocketFirst[n] + SocketCount[procedure.Nodes[n].Kind];
    plan.Consumers.assign(plan.SocketFirst.back(), 0u);
    // Track unread inputs per node and consumers as contiguous runs in one list.
    std::vector<uint32_t> pending(count, 0u), consumer_first(count + 1u, 0u);
    const auto for_each_link = [&](auto &&fn) {
        for (uint32_t n = 0u; n < count; ++n) {
            for (uint32_t i = plan.InputFirst[n]; i < plan.InputFirst[n + 1u]; ++i) {
                if (const auto link = procedure.Inputs[i]; link.FromNode != NoNode) fn(n, link);
            }
        }
    };
    for_each_link([&](uint32_t n, SocketLink link) {
        ++plan.Consumers[plan.SocketFirst[link.FromNode] + link.FromSocket];
        ++pending[n];
        ++consumer_first[link.FromNode + 1u];
    });
    for (uint32_t n = 0u; n < count; ++n) consumer_first[n + 1u] += consumer_first[n];
    std::vector<uint32_t> consumers(consumer_first.back());
    auto fill = consumer_first;
    for_each_link([&](uint32_t n, SocketLink link) { consumers[fill[link.FromNode]++] = n; });
    for (uint32_t n = 0u; n < count; ++n) {
        if (!pending[n]) plan.Order.push_back(n);
    }
    for (size_t i = 0u; i < plan.Order.size(); ++i) {
        const auto n = plan.Order[i];
        for (uint32_t c = consumer_first[n]; c < consumer_first[n + 1u]; ++c) {
            if (!--pending[consumers[c]]) plan.Order.push_back(consumers[c]);
        }
    }
    return plan;
}

void EraseNode(GeometryProcedure &procedure, uint32_t node) {
    const auto first = procedure.Inputs.begin() + InputFirst(procedure, node);
    procedure.Inputs.erase(first, first + InputCount[procedure.Nodes[node].Kind]);
    procedure.Nodes.erase(procedure.Nodes.begin() + node);
    for (auto &link : procedure.Inputs) {
        if (link.FromNode == node) link = {};
        else if (link.FromNode != NoNode && link.FromNode > node) --link.FromNode;
    }
    for (auto *index : {&procedure.DisplayNode, &procedure.ActiveNode}) {
        if (*index == node) *index = NoNode;
        else if (*index != NoNode && *index > node) --*index;
    }
    // The displayed node, or else the last node, replaces a removed active node.
    if (procedure.ActiveNode == NoNode && !procedure.Nodes.empty()) procedure.ActiveNode = procedure.DisplayNode != NoNode ? procedure.DisplayNode : uint32_t(procedure.Nodes.size()) - 1u;
}

void RemoveNode(state::Registry &r, state::Entity node) {
    if (r.Restoring) return;
    for (const auto [mesh_entity, procedure] : std::as_const(r).view<const GeometryProcedure>().each()) {
        const auto it = std::ranges::find(procedure.Nodes, node, &NodeRecord::Entity);
        if (it == procedure.Nodes.end()) continue;
        const auto index = uint32_t(it - procedure.Nodes.begin());
        r.patch<GeometryProcedure>(mesh_entity, [&](GeometryProcedure &edited) { EraseNode(edited, index); });
        return;
    }
}

// Queue destroyed procedure nodes for the object-removal pass.
void QueueNodeRemovals(state::Registry &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    auto &dependents = r.Resources.emplace<PendingObjectRemovals>().Dependents;
    for (const auto &node : r.get<const GeometryProcedure>(mesh_entity).Nodes) dependents.push_back(node.Entity);
}
// Queue removed instance entities for the object-removal pass.
void QueueInstanceRemovals(state::Registry &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    r.Resources.emplace<PendingObjectRemovals>().Dependents.append_range(r.get<const ProcedureInstances>(mesh_entity).Entities);
}
// Queue the unused owned mesh entity for removal so its record is released with it.
void QueueOwnedMeshRemoval(state::Registry &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    r.Resources.emplace<PendingObjectRemovals>().Dependents.push_back(r.get<const ProcedureOwnedMesh>(mesh_entity).MeshEntity);
}
} // namespace

struct ProcedureChanges {
    state::DirtySet Graph; // A geometry procedure's graph changed.
    state::DirtySet Inputs; // A node's parameters or inline selection, a mesh's geometry or objects, or a published instance's parent changed.
    state::DirtySet Created; // A node was created, whose own components the inputs list with it.
};
} // namespace procedure

void RegisterProcedureHandlers(state::Registry &r) {
    using namespace procedure;
    r.Resources.emplace<CookCount>();
    auto &changes = r.Resources.emplace<ProcedureChanges>();
    changes.Graph.bind(r).on<GeometryProcedure>(On::Create | On::Update);
    changes.Created.bind(r).on<NodeLayout>(On::Create);
    r.on_destroy<GeometryProcedure, &QueueNodeRemovals>();
    r.on_destroy<ProcedureInstances, &QueueInstanceRemovals>();
    r.on_destroy<ProcedureOwnedMesh, &QueueOwnedMeshRemoval>();
    r.Resources.emplace<CookPass>();
    auto &inputs = changes.Inputs.bind(r);
    inputs.on<MeshGeometryDirty>(On::Create).on<MeshPositionsChanged>(On::Create);
    inputs.on<InlineSelection>(On::Create | On::Update | On::Destroy);
    // Update published instances when an object is created or a child is detached.
    inputs.on<Instance>(On::Create).on<SceneParent>(On::Destroy);
    ForEachKind([&]<typename K> {
        inputs.on<K>(On::Update);
        r.on_destroy<K, &RemoveNode>();
    });
}

void SettleProcedures(state::Registry &r, bool cook) {
    using namespace procedure;
    const auto &changes = r.Resources.get<const ProcedureChanges>();
    const auto &changed = changes.Graph;
    for (const auto e : changed) {
        if (const auto *procedure = r.try_get<const GeometryProcedure>(e)) r.emplace_or_replace<CookPlan>(e, Plan(*procedure));
    }
    if (!cook) return;
    std::vector<state::Entity> cooked, inputs;
    for (const auto e : changed) {
        if (r.all_of<GeometryProcedure>(e)) cooked.push_back(e);
    }
    for (const auto e : changes.Inputs) {
        if (r.all_of<GeometryProcedure>(e)) cooked.push_back(e);
        else inputs.push_back(e);
    }
    if (cooked.empty() && inputs.empty()) return;
    SortUnique(inputs);
    const auto input = [&](state::Entity e) { return std::ranges::binary_search(inputs, e); };
    // Index procedures by the mesh entities read by their Object Info nodes.
    std::vector<std::pair<state::Entity, state::Entity>> reads;
    for (const auto [e, procedure] : std::as_const(r).view<const GeometryProcedure>().each()) {
        const auto *published = r.try_get<const ProcedureInstances>(e);
        if (std::ranges::any_of(procedure.Nodes, input, &NodeRecord::Entity) || (published && std::ranges::any_of(published->Entities, input))) cooked.push_back(e);
        for (const auto &node : procedure.Nodes) {
            if (node.Kind == KindOf<ObjectInfo>) reads.emplace_back(e, ReadMesh(r, e, r.get<const ObjectInfo>(node.Entity)));
        }
    }
    for (const auto e : inputs) {
        if (const auto *instance = r.try_get<const Instance>(e); instance && r.all_of<ProcedureInstances>(instance->Entity)) cooked.push_back(instance->Entity);
    }
    SortUnique(cooked);
    // Cook dependencies before procedures that read them.
    const auto cooking = [&](state::Entity e) { return std::ranges::binary_search(cooked, e); };
    for (bool grew = true; grew;) {
        grew = false;
        for (const auto [e, read] : reads) {
            if (cooking(e) || (!cooking(read) && !input(read))) continue;
            cooked.insert(std::ranges::lower_bound(cooked, e), e);
            grew = true;
        }
    }
    std::vector<state::Entity> order, seen;
    const auto visit = [&](this const auto &self, state::Entity e) -> void {
        if (std::ranges::contains(seen, e)) return;
        seen.push_back(e);
        for (const auto [reader, read] : reads) {
            if (reader == e && cooking(read)) self(read);
        }
        order.push_back(e);
    };
    for (const auto e : cooked) visit(e);
    if (order.empty()) return;
    // The kept states restore the whole mesh storage exactly only while the passes of one mesh are its only writers.
    auto &pass = r.Resources.get<CookPass>();
    const auto mesh = order.size() == 1u ? order.front() : state::Null;
    if (!(pass.End && mesh != state::Null && mesh == pass.Mesh && r.Resources.get<MeshStore>().StateHash() == *pass.End)) pass.Exact.clear();
    pass.Mesh = mesh;
    pass.End.reset();
    // Derive new-mesh shading before cooking dependent procedures.
    ShadeNewMeshes(r);
    std::optional<BufferArena<SelectionBlock>> arena;
    for (const auto e : order) {
        Evaluate(r, e, arena, changes.Inputs, changes.Created);
        ShowActiveSelection(r, e, arena);
    }
    r.Resources.get<CookCount>().Value += order.size();
}

void FinishCookPass(state::Registry &r) {
    auto &pass = r.Resources.get<procedure::CookPass>();
    if (pass.End || pass.Mesh == state::Null) return;
    pass.End = r.Resources.get<MeshStore>().StateHash();
}

void FinishSourceStates(state::Registry &r) { procedure::FinishPendingSourceStates(r, InvalidOffset); }

namespace procedure {
namespace {
// The states the Source nodes and edited inputs refer to, ascending.
std::vector<store::Hash128> ReferencedStates(const state::Registry &r) {
    std::vector<store::Hash128> states;
    for (const auto [entity, source] : r.view<const SourceState>().each()) {
        if (source.State != store::Hash128{}) states.push_back(source.State);
    }
    for (const auto [entity, input] : r.view<const EditedInput>().each()) {
        if (input.State != store::Hash128{}) states.push_back(input.State);
    }
    std::ranges::sort(states, {}, [](const store::Hash128 &state) { return std::pair{state.A, state.B}; });
    states.erase(std::ranges::unique(states).begin(), states.end());
    return states;
}
} // namespace
} // namespace procedure

void EndMeshStates(state::Registry &r) {
    const profile::CpuScope scope{"EndMeshStates"};
    r.Resources.get<MeshStore>().EndStepStates(procedure::ReferencedStates(r));
}

void RecordMeshStates(state::Registry &r) { r.Resources.get<MeshStore>().RecordKeptStates(procedure::ReferencedStates(r)); }

void ReopenSourceState(state::Registry &r, state::Entity mesh_entity) {
    using namespace procedure;
    const auto *evaluated = r.try_get<const EvaluatedMesh>(mesh_entity);
    const auto *source = evaluated && r.valid(evaluated->Tip) ? r.try_get<const SourceState>(evaluated->Tip) : nullptr;
    if (source && source->StoreId == evaluated->StoreId && source->State != store::Hash128{}) r.replace<SourceState>(evaluated->Tip, store::Hash128{}, source->StoreId);
}

void EndCookStep(state::Registry &r) {
    r.clear<procedure::StagedNode>();
    r.Resources.get<procedure::CookPass>().Exact.clear();
}
