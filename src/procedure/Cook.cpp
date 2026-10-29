#include "procedure/Cook.h"
#include "procedure/InlineSelection.h"
#include "procedure/Procedure.h"
#include "procedure/ProcedureSchema.h"
#include "scene/HierarchyInstance.h"
#include "viewport/InteractionComponents.h"

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
#include "scene/SceneChanges.h"
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
    state::Entity Prototype{state::Null};
    state::Entity Object{state::Null}; // For geometry, the object whose local space the geometry is in, or null for this procedure's mesh space
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
// The element domains of a selection's masks, in mask order.
constexpr std::array SelectionDomains{MeshStore::ElementDomain::Vertex, MeshStore::ElementDomain::Edge, MeshStore::ElementDomain::Face};
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

// Edit topology in place, repairing render, pose and selection state when the record is drawn.
void EditTopology(state::Registry &r, state::Entity mesh_entity, const MeshTopologyTask &task, std::vector<ExtrusionSource> *sources = nullptr) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (mesh_entity != state::Null) ReleaseMeshEditWork(r, mesh_entity);
    mtl::ComputeChain chain{meshes.BufferContext(), TopologyScratchWords};
    auto edits = MeshTopologyEdit::Construct(r, chain, std::span{&task, 1u});
    if (sources) *sources = edits.front().ExtrusionSources(meshes);
    MeshTopologyEdit::PublishAll(r, edits);
    auto &edit = edits.front();
    if (!edit.Output) {
        if (mesh_entity == state::Null) MeshTopologyEdit::FinishAll(r, std::array{&edit});
        chain.Submit();
        return;
    }
    if (sources) {
        const auto handles = chain.Scratch.Get(edit.Output->Vertices);
        for (auto &source : *sources) source.Output = handles[source.Output];
    }
    if (mesh_entity == state::Null) {
        MeshTopologyEdit::FinishAll(r, std::array{&edit});
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
        EditTopology(r, masks.Drawn, *task);
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
        for (const auto &task : tasks) EditTopology(r, masks.Drawn, task);
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
store::Hash128 SelectionKey(const state::Registry &r, state::Entity node) {
    const auto *selection = r.try_get<const InlineSelection>(node);
    if (!selection || selection->All) return {};
    std::vector<uint32_t> words{uint32_t(selection->Domain), selection->Active};
    for (const auto *domain : {&selection->Vertices, &selection->Edges, &selection->Faces}) {
        words.push_back(uint32_t(domain->Handles.size()));
        words.append_range(domain->Handles);
        words.push_back(uint32_t(domain->Blocks.size()));
        for (const auto &[local, mask] : domain->Blocks) {
            words.push_back(local);
            words.append_range(mask);
        }
    }
    return store::HashBytes(std::as_bytes(std::span{words}));
}

enum class TopologyReuse { None,
                           Input,
                           Restored };

Outcome RunExtrude(state::Registry &r, state::Entity node, const Extrude &p, uint32_t id, const Masks &m, const Linked &linked, TopologyReuse reuse, std::vector<uint32_t> &temporaries, bool retain) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto domain = p.Mode == ExtrudeMode::FacesIndividual ? AttributeDomain::Face : AttributeDomain::Vertex;
    const auto offset = ElementParameter(meshes, linked.Of<&Extrude::Offset>(), domain, AttributeType::Vec3);
    const auto scale = ElementParameter(meshes, linked.Of<&Extrude::OffsetScale>(), domain);
    if (!offset || !scale) return std::unexpected(NodeFault::Attribute);
    const auto key = retain ? SelectionKey(r, node) : store::Hash128{};
    const auto *prior = retain ? r.try_get<const NodeResult>(node) : nullptr;
    const auto *cached = r.try_get<const NodeExtrusion>(node);
    const auto *stage = reuse != TopologyReuse::None && prior && cached && prior->StoreId == id && cached->Mode == p.Mode && cached->Selection == key ? cached : nullptr;
    NodeExtrusion computed{.Selection = key, .Mode = p.Mode};
    std::vector<std::pair<uint32_t, vec3>> offsets;
    const bool fields = *offset != InvalidOffset || *scale != InvalidOffset;
    if (!stage || fields) {
        const auto selection = Handles(meshes, id, m);
        const Mesh input{meshes, id};
        if (!stage) {
            for (const auto face : selection.Faces) computed.RegionNormal += input.GetNormal(he::FH{face});
            if (Length(computed.RegionNormal) > 0.f) computed.RegionNormal = Normalize(computed.RegionNormal);
            stage = &computed;
        }
        if (fields) {
            const auto &elements = domain == AttributeDomain::Face ? selection.Faces : selection.Vertices;
            for (const auto source : elements) {
                const auto normal = p.Mode == ExtrudeMode::Region ? stage->RegionNormal : domain == AttributeDomain::Face ? input.GetNormal(he::FH{source}) :
                                                                                                                            input.GetNormal(he::VH{source});
                const auto direction = *offset != InvalidOffset ? meshes.AttributeView<vec3>(*offset)[source] : p.AlongNormals ? normal :
                                                                                                                                 p.Offset;
                offsets.emplace_back(source, direction * (*scale != InvalidOffset ? meshes.AttributeView<float>(*scale)[source] : p.OffsetScale));
            }
            std::ranges::sort(offsets, {}, &std::pair<uint32_t, vec3>::first);
        }
    }
    for (const auto attribute : temporaries) meshes.ReleaseAttribute(attribute);
    temporaries.clear();
    bool restored = reuse == TopologyReuse::Restored;
    if (!restored && stage->State != store::Hash128{}) {
        WaitForRender(r);
        if (const auto change = meshes.RestoreRecord(stage->State, id)) {
            const auto refreshed = ReconcileRestoredRecord(r, *change, m.Drawn);
            if (m.Refresh) *m.Refresh = refreshed;
            restored = true;
        }
    }
    if (!restored) {
        computed.RegionNormal = stage->RegionNormal;
        stage = &computed;
        const auto op = p.Mode == ExtrudeMode::Vertices ? MeshTopologyOp::ExtrudeVertices : p.Mode == ExtrudeMode::Edges ? MeshTopologyOp::ExtrudeEdges :
            p.Mode == ExtrudeMode::FacesIndividual                                                                       ? MeshTopologyOp::ExtrudeFacesIndividual :
                                                                                                                           MeshTopologyOp::ExtrudeRegion;
        auto task = Task(id, op, m);
        if (m.Publish) PublishStoreSelection(meshes, task);
        EditTopology(r, m.Drawn, task, &computed.Vertices);
        if (retain) computed.State = meshes.KeepState();
    }
    GeometrySelection selected;
    for (const auto &vertex : stage->Vertices) selected.Vertices.push_back(vertex.Output);
    SortUnique(selected.Vertices);
    if (m.Drawn != state::Null) *m.Repaired = true;
    if (!selected.Vertices.empty() && (p.OffsetScale != 0.f || *scale != InvalidOffset)) {
        const PositionOperationTarget target{.StoreId = id, .Selection = std::move(selected)};
        const bool common = offsets.empty() && (!p.AlongNormals || p.Mode == ExtrudeMode::Region);
        uint32_t attribute = InvalidOffset;
        PositionOperationOptions options;
        if (common) {
            options.Transform = PositionTransform{.Delta = {.P = (p.AlongNormals ? stage->RegionNormal : p.Offset) * p.OffsetScale}};
        } else {
            attribute = meshes.AddAttribute(id, AttributeDomain::Vertex, AttributeType::Vec3, ".cook.extrude-offset").value();
            for (const auto &vertex : stage->Vertices) {
                const auto normal = p.Mode == ExtrudeMode::Region ? stage->RegionNormal : vertex.Normal;
                const vec3 displacement = offsets.empty() ? (p.AlongNormals ? normal : p.Offset) * p.OffsetScale : std::ranges::lower_bound(offsets, vertex.Source, {}, &std::pair<uint32_t, vec3>::first)->second;
                meshes.SetAttribute<vec3>(attribute, vertex.Output, std::span{&displacement, 1u});
            }
            options.Flags = PositionEditAttributeOffset;
            options.Attribute = meshes.AttributeRef(attribute);
        }
        const auto operation = common ? PositionEditOp::Transform : PositionEditOp::Attribute;
        if (m.Drawn != state::Null) EditDrawnPositions(r, m.Drawn, target, operation, 1.f, 1u, options, m.Refresh);
        else ExecuteGeometryPositions(r, std::span{&target, 1u}, operation, 1.f, 1u, options);
        if (attribute != InvalidOffset) meshes.ReleaseAttribute(attribute);
    }
    if (retain && stage == &computed) r.emplace_or_replace<NodeExtrusion>(node, std::move(computed));
    return id;
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

// The masks in the arena from each selection domain's first block.
GeometryMasks MasksAt(const BufferArena<SelectionBlock> &arena, const std::array<uint32_t, 3> &first) { return {{&arena, first[0]}, {&arena, first[1]}, {&arena, first[2]}}; }
// Allocate each selection domain's masks through the highest block the record owns in it.
std::array<Range, 3> AllocateMasks(MeshStore &meshes, std::optional<BufferArena<SelectionBlock>> &arena, uint32_t id) {
    if (!arena) arena.emplace(meshes.BufferContext(), SlotType::Buffer, mtl::BufferLifetime::Workspace);
    std::array<Range, 3> ranges;
    for (uint32_t d = 0u; d < 3u; ++d) {
        const auto owned = meshes.GetBlockList(id, SelectionDomains[d]).Blocks;
        ranges[d] = arena->Allocate(owned.empty() ? 1u : std::ranges::max(owned) + 1u);
    }
    return ranges;
}
// Allocate zeroed masks, then call fill(domain, masks).
std::array<uint32_t, 3> ZeroedMasks(MeshStore &meshes, std::optional<BufferArena<SelectionBlock>> &arena, uint32_t id, auto &&fill) {
    const auto ranges = AllocateMasks(meshes, arena, id);
    for (uint32_t d = 0u; d < 3u; ++d) {
        const auto masks = arena->GetMutable(ranges[d]);
        std::ranges::fill(masks, SelectionBlock{});
        fill(d, masks);
    }
    return {ranges[0].Offset, ranges[1].Offset, ranges[2].Offset};
}
std::array<uint32_t, 3> UploadSelection(MeshStore &meshes, std::optional<BufferArena<SelectionBlock>> &arena, uint32_t id, const InlineSelection *selection) {
    if (!selection || selection->All) return EveryElement;
    const std::array lists{&selection->Vertices, &selection->Edges, &selection->Faces};
    return ZeroedMasks(meshes, arena, id, [&](uint32_t d, std::span<SelectionBlock> masks) {
        const auto entries = WithDomain(meshes.Arenas(), SelectionDomains[d], [&](const auto &elements) { return elements.Entries(DomainSet(meshes.Get(id), SelectionDomains[d])); });
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
Operand OperandOf(const CookPlan &plan, std::span<const SocketValue> values, uint32_t node, uint32_t input) {
    if (input >= InputCount[plan.Nodes[node].Kind]) return {};
    const auto link = plan.Inputs[plan.InputFirst[node] + input];
    if (link.Constant) return {.Constant = LinkedConstant{link.Constant->Words, link.Constant->Type}};
    if (link.FromNode == NoNode) return {};
    const auto &source = values[plan.SocketFirst[link.FromNode] + link.FromSocket];
    if (source.Type == SocketType::Value) return {.Constant = LinkedConstant{source.Value, AttributeType(std::countr_zero(KindSockets(plan.Nodes[link.FromNode].Kind)[link.FromSocket].Types))}};
    return {source.Type == SocketType::Attribute ? source.Attribute : InvalidOffset};
}
template<typename K> K Parameters(const state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node) {
    const auto [entity, kind] = plan.Nodes[node];
    auto p = r.get<const K>(entity);
    if constexpr (field::Walkable<K>) {
        const auto [first, end] = ParameterRange(kind);
        for (auto s = first; s < end; ++s) {
            if (const auto value = OperandOf(plan, values, node, s).Constant) AssignSocketParameter(p, KindSockets(kind)[s], value->Words);
        }
    }
    return p;
}
Linked LinkedParameters(const CookPlan &plan, std::span<const SocketValue> values, uint32_t node) {
    const auto kind = plan.Nodes[node].Kind;
    const auto sockets = KindSockets(kind);
    const auto [first, end] = ParameterRange(kind);
    Linked linked{kind};
    linked.Attributes.fill(InvalidOffset);
    for (auto s = first; s < end; ++s) {
        if (sockets[s].Type == SocketType::Attribute) linked.Attributes[s] = OperandOf(plan, values, node, s).Attribute;
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
Evaluation EvaluateInputs(state::Registry &r, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, std::optional<BufferArena<SelectionBlock>> &arena) {
    auto &meshes = r.Resources.get<MeshStore>();
    auto upstream = Upstream(plan, node, [&](uint32_t n) { return IsAttributeKind(plan.Nodes[n].Kind); });
    upstream[node] = false;
    Evaluation evaluation;
    std::vector<AttributeJob> jobs;
    for (const auto n : plan.Order) {
        if (!upstream[n]) continue;
        const auto [entity, kind] = plan.Nodes[n];
        const Operands in{OperandOf(plan, values, n, 0u), OperandOf(plan, values, n, 1u), OperandOf(plan, values, n, 2u)};
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
            if constexpr (Prepared<K>) job = Prepare(meshes, Parameters<K>(r, plan, values, n), in);
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
    std::optional<Element> from;
    const auto sockets = KindSockets(plan.Nodes[node].Kind);
    for (uint32_t s = 0u; s < InputCount[plan.Nodes[node].Kind]; ++s) {
        if (sockets[s].Type != SocketType::Selection || plan.Inputs[plan.InputFirst[node] + s].FromNode == NoNode) continue;
        const auto linked = OperandOf(plan, values, node, s);
        // A linked bool value selects every element, or none through empty masks.
        if (linked.Constant) {
            if (linked.Constant->Words[0] & 1u) {
                evaluation.MaskFirst = EveryElement;
                break;
            }
            evaluation.MaskFirst = ZeroedMasks(meshes, arena, id, [](uint32_t, std::span<SelectionBlock>) {});
            break;
        }
        const auto selection = linked.Attribute;
        const auto *attribute = selection != InvalidOffset ? &meshes.GetAttribute(selection) : nullptr;
        if (!attribute || attribute->Type != AttributeType::Bool || attribute->Domain == AttributeDomain::Corner) {
            evaluation.Failed = true;
            break;
        }
        const auto ref = meshes.AttributeRef(selection);
        const auto mask_ranges = AllocateMasks(meshes, arena, id);
        for (uint32_t d = 0u; d < 3u; ++d) {
            const auto domain = SelectionDomains[d];
            const bool scattered = ElementDomainOf(attribute->Domain) == domain;
            if (scattered) from = he::Elements[d];
            const auto [membership, owner] = WithDomain(meshes.Arenas(), domain, [&](const auto &elements) { return std::pair{elements.Blocks.Buffer.Slot, DomainSet(meshes.Get(id), domain).Index}; });
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
    if (from) meshes.DeriveSelectionMasks(r, chain, id, MasksAt(*arena, *evaluation.MaskFirst), *from);
    chain.Submit();
    return evaluation;
}

enum class RunMode { Publish,
                     Transient,
                     Handle };

// Evaluate inputs and run the node, releasing temporary attributes after their last use.
// The trailing arguments become the node's Masks Drawn, Repaired and Refresh.
Outcome RunNode(state::Registry &r, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, std::optional<BufferArena<SelectionBlock>> &arena, state::Entity drawn = state::Null, bool *repaired = nullptr, MeshStore::Change *refresh = nullptr, TopologyReuse reuse_topology = TopologyReuse::None, RunMode mode = RunMode::Publish) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto [entity, kind] = plan.Nodes[node];
    auto evaluation = EvaluateInputs(r, plan, values, node, id, arena);
    const auto release = [&] {
        for (const auto attribute : evaluation.Temporaries) meshes.ReleaseAttribute(attribute);
        evaluation.Temporaries.clear();
    };
    if (evaluation.Failed) {
        release();
        return std::unexpected(NodeFault::Attribute);
    }
    const auto linked = LinkedParameters(plan, values, node);
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
        if (selection.MaskFirst != EveryElement) masks.In = MasksAt(*arena, selection.MaskFirst);
    }
    masks.Publish = r.all_of<PublishSelection>(entity);
    masks.Drawn = drawn;
    masks.Repaired = repaired;
    masks.Refresh = refresh;
    if (mode != RunMode::Transient && std::ranges::any_of(HandleSpecs, [&](const auto &spec) { return spec.Kind == kind; })) {
        if (reuse_topology == TopologyReuse::None || !r.all_of<NodeHandle>(entity)) {
            const auto selected = Handles(meshes, id, masks);
            const Mesh input{meshes, id};
            NodeHandle support;
            for (const auto vertex : selected.Vertices) support.Center += input.GetPosition(he::VH{vertex});
            if (!selected.Vertices.empty()) support.Center /= float(selected.Vertices.size());
            for (const auto face : selected.Faces) support.Normal += input.GetNormal(he::FH{face});
            if (Length(support.Normal) > 0.f) support.Normal = Normalize(support.Normal);
            r.emplace_or_replace<NodeHandle>(entity, support);
        }
    }
    if (mode == RunMode::Handle) {
        release();
        return id;
    }
    if (reuse_topology != TopologyReuse::Restored) PublishInput(r, id, masks);
    Outcome edited = id;
    VisitKind(kind, [&]<typename K> {
        const auto p = Parameters<K>(r, plan, values, node);
        if constexpr (std::same_as<K, Extrude>) {
            auto extrusion = p;
            for (uint32_t socket = 0u; socket < InputCount[kind]; ++socket) {
                if (!sockets[socket].Carries(uint8_t(field::IndexOf<&Extrude::Offset>))) continue;
                const auto &input = plan.Inputs[plan.InputFirst[node] + socket];
                if (input.FromNode != NoNode || input.Constant) extrusion.AlongNormals = false;
            }
            edited = RunExtrude(r, entity, extrusion, id, masks, linked, evaluation.MaskFirst ? TopologyReuse::None : reuse_topology, evaluation.Temporaries, mode == RunMode::Publish);
        } else if constexpr (std::same_as<K, StoreNamedAttribute>) {
            edited = Store(meshes, p, id, OperandOf(plan, values, node, 1u), evaluation.Temporaries);
        } else if constexpr (std::same_as<K, StoreSelection>) {
            const auto linked = OperandOf(plan, values, node, 1u).Attribute;
            const auto domain = linked != InvalidOffset ? meshes.GetAttribute(linked).Domain : masks.Domain == Element::Face ? AttributeDomain::Face :
                masks.Domain == Element::Edge                                                                                ? AttributeDomain::Edge :
                                                                                                                               AttributeDomain::Vertex;
            edited = Store(meshes, p, id, masks, domain);
        } else if constexpr (std::same_as<K, SetPosition>) {
            auto position = OperandOf(plan, values, node, 1u);
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

// The transform from the operand object's local space into Space's, by the rule the settle pass composes world transforms with, which takes each physics body's published pose.
// Both objects compose relative to their nearest common ancestor when no physics body lies below it and its world transform is a positive uniform scaling, rotation and translation.
// Such a world transform commutes with the rule, so the result equals the one through world transforms, and moving that ancestor leaves it unchanged bit for bit.
// An operand in this procedure's mesh space is already in Space's, and a dead object places by the identity.
mat4 ToFirst(const state::Registry &r, state::Entity space, state::Entity operand) {
    if (operand == state::Null) return I4;
    const auto owned = r.view<const BodyPoseCache>() | std::ranges::to<std::vector>();
    // Each object and its ancestors, nearest first.
    const auto lineage = [&](state::Entity e) {
        std::vector<state::Entity> entities;
        for (e = r.valid(e) ? e : state::Null; e != state::Null; e = ParentOrNull(r, e)) entities.push_back(e);
        return entities;
    };
    const auto spaces = lineage(space), operands = lineage(operand);
    const auto frame = [&] {
        const auto common = std::ranges::find_if(operands, [&](state::Entity e) { return std::ranges::contains(spaces, e); });
        if (common == operands.end()) return state::Null;
        const auto bodies_below = [&](const std::vector<state::Entity> &entities) {
            return std::ranges::any_of(entities | std::views::take_while([&](state::Entity e) { return e != *common; }), [&](state::Entity e) { return ContainsInIndexOrder(owned, e); });
        };
        const auto scale = ComposeTransform(r, *common, state::Null, owned).S;
        return !bodies_below(spaces) && !bodies_below(operands) && scale.x > 0.f && scale.x == scale.y && scale.x == scale.z ? *common : state::Null;
    }();
    const auto placed = [&](const std::vector<state::Entity> &entities) { return entities.empty() ? I4 : ToMatrix(ComposeTransform(r, entities.front(), frame, owned)); };
    return Inverse(placed(spaces)) * placed(operands);
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
void PublishInstances(state::Registry &r, state::Entity mesh_entity, state::Entity instanced, std::span<const Transform> transforms, state::Entity prototype = state::Null) {
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
    Destroy(r, extras);
    for (uint32_t o = 0u; o < objects.size(); ++o) {
        for (uint32_t i = 0u; i < count; ++i) {
            auto &e = published[o * count + i];
            if (e == state::Null) {
                e = prototype != state::Null ? CreateHierarchyInstance(r, prototype, transforms[i]) : r.create();
                if (prototype == state::Null) r.emplace<Instance>(e, instanced);
                r.emplace_or_replace<Transform>(e, transforms[i]);
                SetParent(r, e, objects[o]);
                if (prototype == state::Null) EmplaceObject(r, e, ObjectType::Mesh);
                ApplyVisibility(r, e);
                continue;
            }
            if (prototype != state::Null) BindHierarchyInstance(r, e, prototype);
            else if (r.all_of<HierarchyInstance>(e)) {
                const auto placement = r.get<const Transform>(e);
                Destroy(r, e);
                e = r.create();
                r.emplace<Transform>(e, placement);
                r.emplace<Instance>(e, instanced);
                SetParent(r, e, objects[o]);
                EmplaceObject(r, e, ObjectType::Mesh);
            } else if (r.get<const Instance>(e).Entity != instanced) r.replace<Instance>(e, instanced);
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
bool Chained(const CookPlan &plan, std::span<const uint32_t> path) {
    if (path.empty()) return false;
    const auto root = plan.Nodes[path.front()].Kind;
    bool rooted = root == KindOf<Source>;
    VisitKind(root, [&]<typename K> { rooted |= Generator<K>; });
    if (!rooted) return false;
    for (const auto node : path.subspan(1u)) {
        const auto kind = plan.Nodes[node].Kind;
        if (kind == KindOf<InstanceOnPoints> || kind == KindOf<RealizeInstances>) return false;
    }
    std::vector<uint32_t> position(plan.Nodes.size(), NoNode);
    for (uint32_t i = 0u; i < path.size(); ++i) position[path[i]] = i;
    for (uint32_t node = 0u; node < plan.Nodes.size(); ++node) {
        if (!plan.Demanded[node]) continue;
        for (uint32_t input = 0u; input < InputCount[plan.Nodes[node].Kind]; ++input) {
            const auto from = plan.Inputs[plan.InputFirst[node] + input].FromNode;
            if (from == NoNode) continue;
            bool field = IsAttributeKind(plan.Nodes[from].Kind);
            VisitKind(plan.Nodes[from].Kind, [&]<typename K> { field |= ValueKind<K>; });
            if (position[node] != NoNode && position[node] != 0u && field) continue;
            const bool path_link = input == 0u && position[from] != NoNode && position[node] == position[from] + 1u;
            if ((position[node] != NoNode || (position[from] != NoNode && position[from] != 0u)) && !path_link) return false;
        }
    }
    return true;
}

// The record after the display path's nodes from `first` edit it in place, and whether its render records stayed current and any node faulted.
struct PathRun {
    uint32_t Record;
    bool Rendered, Faulted;
};
// The record holds the output of the node before `first`.
// That node and each node that runs keep the mesh storage state their result leaves as their NodeResult, the path's last node keeps a pending one, and the nodes from the first faulted one hold none, so the next cook runs them again.
// The positions of `restored`, a restore's change to the record, refresh with the first drawn position edit or after the path.
PathRun RunPath(state::Registry &r, state::Entity mesh_entity, const CookPlan &plan, std::span<const uint32_t> path, size_t first, uint32_t record, bool rendered, std::optional<BufferArena<SelectionBlock>> &arena, MeshStore::Change &restored, TopologyReuse reuse_topology) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (first < path.size()) FinishPendingSourceStates(r, record);
    // Keep mesh storage as it stands as the result of the path node at `i`, which the record holds.
    const auto keep_result = [&](size_t i) {
        const bool last = i + 1u == path.size();
        const auto entity = plan.Nodes[path[i]].Entity;
        const NodeResult result{i ? plan.Nodes[path[i - 1u]].Entity : state::Null, last ? store::Hash128{} : meshes.KeepState(), record};
        if (const auto *held = r.try_get<const NodeResult>(entity); !held || *held != result) r.emplace_or_replace<NodeResult>(entity, result);
        if (auto &pass = r.Resources.get<CookPass>(); !last && pass.Mesh == mesh_entity && !std::ranges::contains(pass.Exact, result.State)) pass.Exact.push_back(result.State);
    };
    if (reuse_topology != TopologyReuse::Restored) keep_result(first - 1u);
    std::vector<SocketValue> values(plan.SocketFirst.back());
    for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
        VisitKind(plan.Nodes[n].Kind, [&]<typename K> {
            if constexpr (ValueKind<K>) values[plan.SocketFirst[n]] = {.Type = SocketType::Value, .Value = Words(r.get<const K>(plan.Nodes[n].Entity).Value)};
        });
    }
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
        const auto [entity, kind] = plan.Nodes[node];
        r.remove<NodeError>(entity);
        // A chained Join or Boolean has no second input and passes its first through.
        if (kind != KindOf<Join> && kind != KindOf<Boolean>) {
            bool repaired = false;
            if (!rendered) refresh_restored();
            const auto edited = RunNode(r, plan, values, node, record, arena, rendered ? mesh_entity : state::Null, &repaired, &restored, i == first ? reuse_topology : TopologyReuse::None);
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
        if (faulted) r.remove<NodeResult, NodeExtrusion, NodeHandle>(entity);
        else keep_result(i);
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
// The held nodes are the path's unchanged prefix whose results stand: the root, and each node after it whose result was kept on the result of the node before it.
// The nodes after the last held one edit the record in place: as it stands when it holds that node's output, else restored to that node's result state, else a new record of the root.
// The whole mesh storage restores while only this mesh's cooks wrote it since the state was kept, and the record alone restores otherwise, each followed by the same nodes.
// A copied procedure's nodes keep the results their originals hold, and the components they were created with leave them unchanged.
void CookChain(state::Registry &r, state::Entity mesh_entity, const CookPlan &plan, std::span<const uint32_t> path, std::optional<BufferArena<SelectionBlock>> &arena, const state::DirtySet &changed, const state::DirtySet &created) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto *live = r.try_get<const EvaluatedMesh>(mesh_entity);
    const auto prior = live && meshes.TryGet(live->StoreId) ? live->StoreId : InvalidOffset;
    const auto entity_at = [&](size_t i) { return plan.Nodes[path[i]].Entity; };
    std::vector<bool> invalid(plan.Nodes.size());
    for (const auto n : plan.Order) {
        invalid[n] = changed.contains(plan.Nodes[n].Entity) && !created.contains(plan.Nodes[n].Entity);
        for (const auto &input : std::span{plan.Inputs}.subspan(plan.InputFirst[n], InputsOf(plan, n)))
            if (input.FromNode != NoNode) invalid[n] = invalid[n] || invalid[input.FromNode];
    }
    const auto edited = [&](state::Entity e) { return invalid[IndexOf(plan, e)]; };
    size_t held = 0u;
    while (held < path.size() && !edited(entity_at(held))) {
        if (held) {
            const auto *result = r.try_get<const NodeResult>(entity_at(held));
            if (!result || result->Input != entity_at(held - 1u)) break;
        }
        ++held;
    }
    auto record = prior;
    // The record the result replaces, which a restore returns to the state before the nodes that replaced it.
    auto replaced = prior;
    bool rendered = prior != InvalidOffset && meshes.Get(prior).RenderTopologies != 0u;
    MeshStore::Change restored{};
    // Whether the record holds the output of the last held node.
    const bool continued = prior != InvalidOffset && held && live->Tip == entity_at(held - 1u);
    auto reuse_topology = continued ? TopologyReuse::Input : TopologyReuse::None;
    if (!continued) {
        const auto *result = held ? r.try_get<const NodeResult>(entity_at(held - 1u)) : nullptr;
        bool restored_record = false;
        store::Hash128 topology_state{};
        if (result && held < path.size() && plan.Nodes[path[held]].Kind == KindOf<Extrude> && result->StoreId == prior) {
            const auto node = path[held];
            const auto entity = entity_at(held);
            const auto *cached = r.try_get<const NodeResult>(entity);
            const auto *topology = r.try_get<const NodeExtrusion>(entity);
            const auto inputs = std::span{plan.Inputs}.subspan(plan.InputFirst[node], InputsOf(plan, node));
            const bool plain = std::ranges::all_of(inputs.subspan(1u), [](const auto &input) { return input.FromNode == NoNode && !input.Constant; });
            if (plain && cached && cached->Input == entity_at(held - 1u) && cached->StoreId == prior && topology &&
                topology->Mode == r.get<const Extrude>(entity).Mode && topology->Selection == SelectionKey(r, entity)) topology_state = topology->State;
        }
        if (topology_state != store::Hash128{}) {
            WaitForRender(r);
            if (const auto change = meshes.RestoreRecord(topology_state, prior)) {
                restored = ReconcileRestoredRecord(r, *change, mesh_entity);
                restored_record = true;
                reuse_topology = TopologyReuse::Restored;
            }
        }
        if (!restored_record && result && result->StoreId == prior && result->State != store::Hash128{}) {
            WaitForRender(r);
            if (std::ranges::contains(r.Resources.get<const CookPass>().Exact, result->State) && meshes.RestoreKeptState(result->State)) {
                restored = ReconcileRestoredStore(r, mesh_entity);
                restored_record = true;
            } else if (const auto change = meshes.RestoreRecord(result->State, prior)) {
                restored = ReconcileRestoredRecord(r, *change, mesh_entity);
                restored_record = true;
            }
        }
        if (restored_record) {
            if (reuse_topology != TopologyReuse::Restored) reuse_topology = TopologyReuse::Input;
            replaced = InvalidOffset;
            rendered = meshes.Get(record).RenderTopologies != 0u;
        } else {
            const auto root = MaterializeRoot(r, plan.Nodes[path.front()]);
            if (!root) {
                r.emplace_or_replace<NodeError>(entity_at(0u), NodeFault::NoMesh);
                for (const auto node : path.subspan(held)) r.remove<NodeResult, NodeExtrusion, NodeHandle>(plan.Nodes[node].Entity);
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
    const auto run = RunPath(r, mesh_entity, plan, path, held, record, rendered, arena, restored, reuse_topology);
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
uint32_t Evaluate(state::Registry &r, state::Entity mesh_entity, const CookPlan &plan, std::optional<BufferArena<SelectionBlock>> &arena, const state::DirtySet &changed, const state::DirtySet &created_nodes, RunMode mode = RunMode::Publish) {
    auto &meshes = r.Resources.get<MeshStore>();
    const bool publish = mode == RunMode::Publish;
    // Track the first-input path, which carries the record the mesh draws from the root to the displayed node.
    const auto display = plan.ResultNode;
    std::vector<bool> on_path(plan.Nodes.size());
    std::vector<uint32_t> path;
    for (auto node = display; node != NoNode && !on_path[node];) {
        on_path[node] = true;
        path.push_back(node);
        const auto kind = plan.Nodes[node].Kind;
        node = InputCount[kind] && KindSockets(kind).front().Type == SocketType::Geometry ? plan.Inputs[plan.InputFirst[node]].FromNode : NoNode;
    }
    std::ranges::reverse(path);
    const bool chained = publish && Chained(plan, path);
    // Only the nodes of a chained display path hold results, and only the Boolean nodes a cook evaluates hold placements, since the cooks that skip a node would leave its result or placement stale.
    // A chained path's Boolean nodes have no second input.
    for (uint32_t node = 0u; node < plan.Nodes.size(); ++node) {
        const auto entity = plan.Nodes[node].Entity;
        if (publish && (!chained || !on_path[node])) r.remove<NodeResult, NodeExtrusion, NodeHandle>(entity);
        if (chained) r.remove<BooleanPlacement>(entity);
    }
    if (chained) {
        CookChain(r, mesh_entity, plan, path, arena, changed, created_nodes);
        return DrawnStoreId(r, mesh_entity).value_or(InvalidOffset);
    }
    // The record the mesh drew, which the cook releases once the result replaces it.
    const auto prior = r.all_of<EvaluatedMesh>(mesh_entity) ? r.get<const EvaluatedMesh>(mesh_entity).StoreId : InvalidOffset;
    // Evaluate only nodes reachable from the displayed node.
    const auto &reached = plan.Demanded;
    auto remaining = plan.Consumers;
    // The display reads the displayed node's result after every node, so the result's other consumers clone it.
    const auto result_socket = display != NoNode ? ResultSocket(plan.Nodes[display].Kind) : std::optional<uint16_t>{};
    if (result_socket) ++remaining[plan.SocketFirst[display] + *result_socket];
    // Release temporary cook records at completion and retain the result.
    std::vector<uint32_t> created;
    std::vector<SocketValue> values(plan.SocketFirst.back());
    const auto slot_of = [&](uint32_t node, uint32_t input) {
        const auto link = plan.Inputs[plan.InputFirst[node] + input];
        return plan.SocketFirst[link.FromNode] + link.FromSocket;
    };
    // Clone borrowed records before editing and owned records before additional readers.
    // Preserve the result-path record by cloning its producer instead.
    const auto take = [&](uint32_t node, uint32_t input) -> SocketValue {
        auto &taken = values[plan.SocketFirst[node] + input];
        if (plan.Inputs[plan.InputFirst[node] + input].FromNode == NoNode) return taken;
        const auto slot = slot_of(node, input);
        taken = values[slot];
        const bool last = --remaining[slot] == 0u;
        if (taken.Record == InvalidOffset || (taken.Borrowed() ? KindSockets(plan.Nodes[node].Kind)[input].Reads : last)) return taken;
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
        const auto [entity, kind] = plan.Nodes[node];
        const auto fail = [&](NodeFault fault) { if (publish) r.emplace<NodeError>(entity, fault); };
        const auto first = plan.SocketFirst[node];
        if (publish) r.remove<NodeError>(entity);
        if (publish && !reached[node]) r.remove<BooleanPlacement>(entity);
        if (!reached[node] || IsAttributeKind(kind)) continue;
        if (kind == KindOf<Source>) {
            // A pending state's record is still the source, which its readers borrow.
            const auto &source = r.get<const SourceState>(entity);
            if (source.State == store::Hash128{} && source.StoreId != InvalidOffset && meshes.TryGet(source.StoreId)) values[first] = {.MeshEntity = mesh_entity, .Record = source.StoreId};
            else if (const auto materialized = MaterializeSource(r, source)) values[first].Record = created.emplace_back(*materialized);
            else fail(NodeFault::NoMesh);
            continue;
        }
        if (kind == KindOf<ObjectInfo>) {
            const auto &info = r.get<const ObjectInfo>(entity);
            const auto read_mesh = ReadMesh(r, mesh_entity, info);
            if (read_mesh != state::Null) values[first] = {.MeshEntity = read_mesh, .Object = info.Object, .Record = *DrawnStoreId(r, read_mesh)};
            else fail(NodeFault::NoMesh);
            continue;
        }
        bool generated = false;
        VisitKind(kind, [&]<typename K> {
            if constexpr (Generator<K>) {
                values[first + InputCount[kind]].Record = created.emplace_back(Generate(r, Parameters<K>(r, plan, values, node)));
                generated = true;
            } else if constexpr (ValueKind<K>) {
                values[first] = {.Type = SocketType::Value, .Value = Words(r.get<const K>(entity).Value)};
                generated = true;
            }
        });
        if (generated) continue;
        const auto in = take(node, 0u);
        const auto input = in.Record;
        if (mode == RunMode::Handle && node == display) {
            if (input != InvalidOffset) RunNode(r, plan, values, node, input, arena, state::Null, nullptr, nullptr, TopologyReuse::None, RunMode::Handle);
            break;
        }
        // A node's geometry output is in its first input's space.
        values[first + InputCount[kind]].Object = in.Object;
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
            const auto p = Parameters<Boolean>(r, plan, values, node);
            // Map Mesh 2 into Mesh 1's space, which must keep Mesh 2 unmirrored.
            const auto to_first = ToFirst(r, p.Space, second.Object);
            const BooleanPlacement placement{second.Object, to_first};
            if (const auto *held = r.try_get<const BooleanPlacement>(entity); publish && (!held || *held != placement)) r.emplace_or_replace<BooleanPlacement>(entity, placement);
            const auto rotation = ToMat3(to_first);
            const bool linked = input != InvalidOffset && second.Record != InvalidOffset, oriented = Determinant(rotation) > 0.f;
            const auto task = linked && oriented ? BooleanTask(Mesh{meshes, input}, Mesh{meshes, second.Record}, p.Operation, rotation, vec3(to_first[3])) : std::nullopt;
            const auto combined = task ? ExecuteGeometryBoolean(r, *task) : std::nullopt;
            auto &output = values[first + InputCount[kind]];
            if (combined) {
                created.push_back(*combined);
                output.Record = *combined;
            } else {
                if (linked) fail(!oriented ? NodeFault::Parameter : !task ? NodeFault::NotClosed : NodeFault::Empty);
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
            const auto prototype = r.get<const InstanceOnPoints>(entity).Prototype;
            if (prototype != state::Null && !r.all_of<ObjectKind>(prototype)) {
                fail(NodeFault::Parameter);
                continue;
            }
            if (instance.Record == InvalidOffset && prototype == state::Null) {
                fail(NodeFault::NoMesh);
                continue;
            }
            if (const auto placed = RunNode(r, plan, values, node, input, arena, state::Null, nullptr, nullptr, TopologyReuse::None, publish ? RunMode::Publish : RunMode::Transient)) {
                output.Type = SocketType::Instances;
                output.Prototype = prototype;
                output.MeshEntity = instance.MeshEntity;
                output.Instance = instance.Record;
            } else {
                fail(placed.error());
            }
            continue;
        }
        if (kind == KindOf<RealizeInstances>) {
            if (in.Prototype != state::Null) {
                fail(NodeFault::Parameter);
                values[first + 1u] = in;
                continue;
            }
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
        const auto edited = input != InvalidOffset ? RunNode(r, plan, values, node, input, arena, state::Null, nullptr, nullptr, TopologyReuse::None, publish ? RunMode::Publish : RunMode::Transient) : Outcome{input};
        if (!edited) fail(edited.error());
        values[first + InputCount[kind]].Record = edited.value_or(input);
    }
    if (mode != RunMode::Handle && result_socket) {
        // The result is this cook's record, so a borrowed record is cloned.
        const auto &value = values[plan.SocketFirst[display] + *result_socket];
        if (value.Record != InvalidOffset && (value.Type == SocketType::Geometry || value.Type == SocketType::Instances)) {
            result = value.Borrowed() ? created.emplace_back(CloneRecord(r, value.Record)) : value.Record;
            if (value.Type == SocketType::Instances) published = value;
        }
    }
    if (!publish) {
        disown(result);
        meshes.Release(created);
        return result;
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
    const bool owns = !transforms.empty() && published->Prototype == state::Null && published->MeshEntity == state::Null;
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
                     transforms, published ? published->Prototype : state::Null);
    PublishCooked(r, mesh_entity, result, state::Null, false, true);
    return result;
}

uint32_t EvaluateTarget(state::Registry &r, state::Entity node, std::optional<uint16_t> port, RunMode mode) {
    const auto mesh = r.get<const ProcedureNode>(node).Procedure;
    auto target = r.get<const GeometryProcedure>(mesh);
    target.DisplayNode = node;
    target.DisplayPort = port;
    const auto plan = Compile(r, target);
    std::optional<BufferArena<SelectionBlock>> arena;
    return Evaluate(r, mesh, plan, arena, {}, {}, mode);
}

// Queue destroyed procedure nodes for the object-removal pass.
void QueueNodeRemovals(state::Registry &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    auto &dependents = r.Resources.emplace<PendingObjectRemovals>().Dependents;
    for (auto node = r.get<const GeometryProcedure>(mesh_entity).First; node != state::Null; node = r.get<const ProcedureNode>(node).Next) dependents.push_back(node);
}
// Queue removed instance entities for the object-removal pass.
void QueueInstanceRemovals(state::Registry &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    r.Resources.emplace<PendingObjectRemovals>().Objects.append_range(r.get<const ProcedureInstances>(mesh_entity).Entities);
}
// Queue the unused owned mesh entity for removal so its record is released with it.
void QueueOwnedMeshRemoval(state::Registry &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    r.Resources.emplace<PendingObjectRemovals>().Dependents.push_back(r.get<const ProcedureOwnedMesh>(mesh_entity).MeshEntity);
}
} // namespace

void SetMeshSelection(state::Registry &r, state::Entity mesh_entity, const InlineSelection *selection) {
    std::optional<BufferArena<SelectionBlock>> arena;
    auto &meshes = r.Resources.get<MeshStore>();
    const auto *live = r.try_get<const EvaluatedMesh>(mesh_entity);
    if (!live || !meshes.TryGet(live->StoreId)) return;
    if (!selection || selection->All) {
        ApplyEditSelectionCommand(r, std::array{ElementRange{mesh_entity, 0u}}, selection ? selection->Domain : Element::Vertex, EditSelectionOperation::Fill);
    } else {
        const auto masks = UploadSelection(meshes, arena, live->StoreId, selection);
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.SelectMasks(r, chain, live->StoreId, MasksAt(*arena, masks));
        chain.Submit();
    }
    RefreshElementSelectionSummaries(r, std::array{mesh_entity}, selection ? selection->Domain : r.get<const EditMode>(state::Document).Value);
    const auto handle = selection && selection->Domain != Element::None && meshes.GetSelectedElements(live->StoreId, selection->Domain).Contains(selection->Active) ? selection->Active : InvalidOffset;
    meshes.WriteSelectionSummary(live->StoreId).ActiveHandle = handle;
    if (handle != InvalidOffset) r.emplace_or_replace<MeshActiveElement>(mesh_entity, handle);
    else r.remove<MeshActiveElement>(mesh_entity);
}

bool PrepareOperationHandle(state::Registry &r, state::Entity node) {
    if (r.all_of<NodeHandle>(node)) return true;
    EvaluateTarget(r, node, std::nullopt, RunMode::Handle);
    return r.all_of<NodeHandle>(node);
}

struct ProcedureChanges {
    state::DirtySet HandleInputs;
    state::DirtySet References;
    state::DirtySet Objects;
    uint64_t ReferencesEpoch{};
    std::vector<std::pair<state::Entity, state::Entity>> Readers;
    std::vector<std::pair<state::Entity, state::Entity>> PlacementReaders;
    state::DirtySet Graph; // A geometry procedure's graph changed, on its mesh entity or one of its node entities.
    state::DirtySet Inputs; // A node's parameters or inline selection, a mesh's geometry or objects, or a published instance's parent changed.
    state::DirtySet Created; // A node was created, whose own components the inputs list with it.
    state::DirtySet Moved; // An entity's local transform, pose, parent or published world transform changed since the last cook pass.
};
} // namespace procedure

namespace procedure {
uint32_t SelectionInput(state::Registry &r, state::Entity node) {
    const auto input = ResolveSocket(r, InputOf(r, node, 0u), true).Source;
    return input.FromNode == state::Null ? InvalidOffset : EvaluateTarget(r, input.FromNode, input.FromSocket, RunMode::Transient);
}

state::Entity CreateNode(state::Registry &r, uint16_t kind, uvec2 position) {
    const auto entity = r.create();
    VisitKind(kind, [&]<typename K> { r.emplace<K>(entity); });
    r.emplace<NodeLayout>(entity, position);
    return entity;
}

void RootProcedure(state::Registry &r, state::Entity mesh_entity, state::Entity root, uint16_t kind) {
    r.emplace<ProcedureNode>(root, mesh_entity, state::Null, kind);
    r.emplace<NodeInputs>(root, std::vector<SocketLink>(InputCount[kind]));
    r.emplace_or_replace<GeometryProcedure>(mesh_entity, root, root, root);
    r.emplace_or_replace<NetworkActive>(mesh_entity, NetworkItem{root}, std::vector<NetworkItem>{{root}});
}
} // namespace procedure

void RegisterProcedureHandlers(state::Registry &r) {
    using namespace procedure;
    r.Resources.emplace<CookCount>();
    auto &changes = r.Resources.emplace<ProcedureChanges>();
    changes.Graph.bind(r).on<InstanceOnPoints>(On::Update).on<GeometryProcedure>(On::Create | On::Update).on<ProcedureNode>(On::Create | On::Update).on<NodeInputs>(On::Update).on<NodeBypassed>(On::Create | On::Destroy).on<SubnetInterface>(On::Create | On::Update | On::Destroy).on<NodeContainer>(On::Create | On::Update | On::Destroy);
    changes.HandleInputs.bind(r).on<InlineSelection>(On::Create | On::Update | On::Destroy);
    changes.Created.bind(r).on<NodeLayout>(On::Create);
    changes.Objects.bind(r, state::Clearing::Explicit).on<Instance>(On::Create | On::Update | On::Destroy);
    changes.References.bind(r, state::Clearing::Explicit)
        .on<ObjectInfo>(On::Create | On::Update | On::Destroy)
        .on<procedure::Boolean>(On::Create | On::Update | On::Destroy)
        .on<Instance>(On::Create | On::Update | On::Destroy)
        .on<SceneParent>(On::Create | On::Update | On::Destroy);
    r.on_destroy<GeometryProcedure, &QueueNodeRemovals>();
    r.on_destroy<ProcedureInstances, &QueueInstanceRemovals>();
    r.on_destroy<ProcedureOwnedMesh, &QueueOwnedMeshRemoval>();
    r.Resources.emplace<CookPass>();
    auto &inputs = changes.Inputs.bind(r);
    inputs.on<MeshGeometryDirty>(On::Create).on<MeshPositionsChanged>(On::Create);
    inputs.on<InlineSelection>(On::Create | On::Update | On::Destroy);
    // Update published instances when an object is created or a child is detached.
    inputs.on<Instance>(On::Create).on<SceneParent>(On::Destroy);
    // Boolean nodes follow the objects whose world transforms place their operands.
    changes.Moved.bind(r, state::Clearing::Explicit).on<Transform>(On::Update).on<PosedLocal>(On::Create | On::Update | On::Destroy).on<SceneParent>(On::Create | On::Destroy);
    ForEachKind([&]<typename K> { inputs.on<K>(On::Update); });
}

void SettleProcedures(state::Registry &r, bool cook) {
    using namespace procedure;
    auto &changes = r.Resources.get<ProcedureChanges>();
    // The procedure a changed mesh or node entity belongs to, or null.
    const auto procedure_of = [&](state::Entity e) {
        if (r.all_of<GeometryProcedure>(e)) return e;
        const auto *node = r.try_get<const ProcedureNode>(e);
        return node && r.all_of<GeometryProcedure>(node->Procedure) ? node->Procedure : state::Null;
    };
    std::vector<state::Entity> planned, rewired;
    for (const auto e : changes.Graph) {
        if (const auto procedure = procedure_of(e); procedure != state::Null) {
            planned.push_back(procedure);
            if (e != procedure) rewired.push_back(procedure);
        }
    }
    SortUnique(planned);
    SortUnique(rewired);
    std::vector<state::Entity> cooked;
    for (const auto e : planned) {
        const auto &procedure = r.get<const GeometryProcedure>(e);
        const auto *previous_plan = r.try_get<const CookPlan>(e);
        if (previous_plan && !std::ranges::binary_search(rewired, e) &&
            (previous_plan->Nodes.empty() ? procedure.First == state::Null : previous_plan->Nodes.front().Entity == procedure.First && previous_plan->Nodes.back().Entity == procedure.Last)) {
            const auto result = EntityOf(*previous_plan, previous_plan->ResultNode);
            r.patch<CookPlan>(e, [&](auto &plan) { BindDisplay(r, procedure, plan); ++plan.Revision; });
            const auto &plan = r.get<const CookPlan>(e);
            if (cook && result != EntityOf(plan, plan.ResultNode)) cooked.push_back(e);
            continue;
        }
        auto compiled = Compile(r, procedure);
        compiled.Revision = previous_plan ? previous_plan->Revision + 1 : 1;
        if (cook && (!previous_plan || EntityOf(*previous_plan, previous_plan->ResultNode) != EntityOf(compiled, compiled.ResultNode))) cooked.push_back(e);
        if (cook && previous_plan) {
            for (uint32_t n = 0; n < compiled.Nodes.size(); ++n) {
                const auto entity = compiled.Nodes[n].Entity;
                const auto before = IndexOf(*previous_plan, entity);
                bool changed = before == NoNode;
                if (!changed) {
                    changed = InputsOf(compiled, n) != InputsOf(*previous_plan, before);
                    for (uint16_t i = 0; !changed && i < InputsOf(compiled, n); ++i) {
                        const auto a = compiled.Inputs[compiled.InputFirst[n] + i], b = previous_plan->Inputs[previous_plan->InputFirst[before] + i];
                        changed = EntityOf(compiled, a.FromNode) != EntityOf(*previous_plan, b.FromNode) || a.FromSocket != b.FromSocket || a.Constant != b.Constant;
                    }
                }
                if (changed) changes.Inputs.emplace(entity);
            }
        }
        r.emplace_or_replace<CookPlan>(e, std::move(compiled));
    }
    const auto demanded = [&](state::Entity e, state::Entity procedure) {
        const auto &plan = r.get<const CookPlan>(procedure);
        const auto node = IndexOf(plan, e);
        return node != NoNode && plan.Demanded[node];
    };
    if (changes.ReferencesEpoch != r.Epoch || !changes.References.empty() || !planned.empty()) {
        changes.Readers.clear();
        changes.PlacementReaders.clear();
        const auto placement = [&](state::Entity procedure, state::Entity object) {
            for (auto e = r.valid(object) ? object : state::Null; e != state::Null; e = ParentOrNull(r, e)) changes.PlacementReaders.emplace_back(e, procedure);
        };
        for (const auto [node, info] : std::as_const(r).view<const ObjectInfo>().each()) {
            const auto procedure = procedure_of(node);
            if (procedure == state::Null || !demanded(node, procedure)) continue;
            if (cook && changes.Objects.contains(info.Object)) {
                changes.Inputs.emplace(node);
                cooked.push_back(procedure);
            }
            if (const auto *instance = r.try_get<const Instance>(info.Object); instance && instance->Entity != procedure) changes.Readers.emplace_back(instance->Entity, procedure);
            placement(procedure, info.Object);
        }
        for (const auto [node, boolean] : std::as_const(r).view<const procedure::Boolean>().each()) {
            if (const auto procedure = procedure_of(node); procedure != state::Null && demanded(node, procedure)) placement(procedure, boolean.Space);
        }
        SortUnique(changes.Readers);
        SortUnique(changes.PlacementReaders);
        changes.References.clear();
        changes.Objects.clear();
        changes.ReferencesEpoch = r.Epoch;
    }
    if (!cook) return;
    std::vector<state::Entity> affected = planned;
    for (const auto e : changes.Inputs)
        if (const auto owner = procedure_of(e); owner != state::Null) affected.push_back(owner);
    SortUnique(affected);
    for (const auto owner : affected) {
        const auto &plan = r.get<const CookPlan>(owner);
        std::vector<bool> invalid(plan.Nodes.size());
        for (const auto n : plan.Order) {
            bool upstream = changes.Inputs.contains(owner);
            for (const auto &input : std::span{plan.Inputs}.subspan(plan.InputFirst[n], InputsOf(plan, n)))
                if (input.FromNode != NoNode) upstream |= invalid[input.FromNode];
            const auto e = plan.Nodes[n].Entity;
            invalid[n] = upstream || changes.Inputs.contains(e);
            if (plan.Demanded[n] || !invalid[n]) continue;
            r.remove<NodeResult, NodeExtrusion>(e);
            if (upstream || changes.HandleInputs.contains(e) || changes.Graph.contains(e)) r.remove<NodeHandle>(e);
        }
    }
    const bool handle_dirty = !affected.empty() || !changes.Moved.empty();
    const auto handle = [&] {
        if (!handle_dirty) return;
        if (const auto *active = r.try_get<const OperationHandle>(state::Document); active && r.all_of<ProcedureNode>(active->Node)) PrepareOperationHandle(r, active->Node);
    };
    std::vector<state::Entity> inputs;
    for (const auto e : changes.Inputs) {
        if (const auto procedure = procedure_of(e); procedure != state::Null) {
            if (e == procedure || demanded(e, procedure)) cooked.push_back(procedure);
        } else inputs.push_back(e);
    }
    SortUnique(cooked);
    if (cooked.empty() && inputs.empty() && changes.Moved.empty()) {
        handle();
        return;
    }
    SortUnique(inputs);
    const auto input = [&](state::Entity e) { return std::ranges::binary_search(inputs, e); };
    for (const auto [e, published] : std::as_const(r).view<const ProcedureInstances>().each()) {
        if (std::ranges::any_of(published.Entities, input)) cooked.push_back(e);
    }
    const auto readers_of = [](const auto &readers, state::Entity source) {
        const auto [first, last] = std::ranges::equal_range(readers, source, {}, &std::pair<state::Entity, state::Entity>::first);
        return std::ranges::subrange(first, last);
    };
    for (const auto e : inputs) {
        if (const auto *instance = r.try_get<const Instance>(e); instance && r.all_of<ProcedureInstances>(instance->Entity)) cooked.push_back(instance->Entity);
    }
    // A procedure whose placing objects moved cooks only when the motion changed one of its Boolean nodes' placements.
    // An edit that cooked in its own pass then does not cook again for the world transform it published, and moving a common ancestor of both operands cooks nothing.
    std::vector<state::Entity> moved;
    for (const auto source : changes.Moved) {
        for (const auto [_, reader] : readers_of(changes.PlacementReaders, source)) moved.push_back(reader);
    }
    changes.Moved.clear();
    SortUnique(moved);
    for (const auto [node, placement] : std::as_const(r).view<const BooleanPlacement>().each()) {
        const auto procedure = procedure_of(node);
        if (std::ranges::binary_search(moved, procedure) && ToFirst(r, r.get<const procedure::Boolean>(node).Space, placement.Operand) != placement.ToFirst) cooked.push_back(procedure);
    }
    SortUnique(cooked);
    // Propagate only through the displayed geometry dependencies.
    std::vector<state::Entity> queue = cooked;
    queue.append_range(inputs);
    for (size_t i = 0; i < queue.size(); ++i) {
        for (const auto [_, reader] : readers_of(changes.Readers, queue[i])) {
            const auto at = std::ranges::lower_bound(cooked, reader);
            if (at != cooked.end() && *at == reader) continue;
            cooked.insert(at, reader);
            queue.push_back(reader);
        }
    }
    const auto index_of = [&](state::Entity entity) {
        const auto at = std::ranges::lower_bound(cooked, entity);
        return at != cooked.end() && *at == entity ? uint32_t(at - cooked.begin()) : NoNode;
    };
    const auto order = DependencyOrder(uint32_t(cooked.size()), [&](auto visit) {
        for (const auto [source, reader] : changes.Readers) {
            const auto from = index_of(source), to = index_of(reader);
            if (from != NoNode && to != NoNode) visit(from, to);
        }
    });
    assert(order.size() == cooked.size());
    if (order.empty()) {
        handle();
        return;
    }
    // The kept states restore the whole mesh storage exactly only while the passes of one mesh are its only writers.
    auto &pass = r.Resources.get<CookPass>();
    const auto mesh = order.size() == 1u ? cooked[order.front()] : state::Null;
    if (!(pass.End && mesh != state::Null && mesh == pass.Mesh && r.Resources.get<MeshStore>().StateHash() == *pass.End)) pass.Exact.clear();
    pass.Mesh = mesh;
    pass.End.reset();
    // Derive new-mesh shading before cooking dependent procedures.
    ShadeNewMeshes(r);
    std::optional<BufferArena<SelectionBlock>> arena;
    for (const auto n : order) {
        const auto e = cooked[n];
        Evaluate(r, e, r.get<const CookPlan>(e), arena, changes.Inputs, changes.Created);
    }
    r.Resources.get<CookCount>().Value += order.size();
    handle();
}

void RecordWorldTransformChanges(state::Registry &r) {
    auto &moved = r.Resources.get<procedure::ProcedureChanges>().Moved;
    for (const auto e : r.Resources.get<const SceneChanges>().WorldTransform) moved.emplace(e);
}

void FinishCookPass(state::Registry &r) {
    auto &pass = r.Resources.get<procedure::CookPass>();
    if (pass.End || pass.Mesh == state::Null) return;
    pass.End = r.Resources.get<MeshStore>().StateHash();
}

void FinishSourceStates(state::Registry &r) { procedure::FinishPendingSourceStates(r, InvalidOffset); }

namespace procedure {
namespace {
// The states the Source nodes and node results refer to, ascending.
std::vector<store::Hash128> ReferencedStates(const state::Registry &r) {
    std::vector<store::Hash128> states;
    for (const auto [entity, source] : r.view<const SourceState>().each()) {
        if (source.State != store::Hash128{}) states.push_back(source.State);
    }
    for (const auto [entity, result] : r.view<const NodeResult>().each()) {
        if (result.State != store::Hash128{}) states.push_back(result.State);
    }
    for (const auto [_, extrusion] : r.view<const NodeExtrusion>().each()) states.push_back(extrusion.State);
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
