#include "procedure/Cook.h"
#include "procedure/Procedure.h"
#include "scene/HierarchyInstance.h"
#include "viewport/InteractionComponents.h"

#include "Profile.h"
#include "SortUnique.h"
#include "TransformMath.h"
#include "gpu/ElementRepairPushConstants.h"
#include "mesh/BeautifyFaces.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/LevelSet.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshTopologyEdit.h"
#include "mesh/Neighborhood.h"
#include "mesh/Primitives.h"
#include "mesh/RecalculateNormals.h"
#include "mesh/Region.h"
#include "mesh/ScratchPasses.h"
#include "mesh/SharpnessOperations.h"
#include "mesh/Simplify.h"
#include "mesh/Unsubdivide.h"
#include "object/ObjectOps.h"
#include "object/PendingSync.h"
#include "physics/PhysicsTypes.h"
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

#include <format>

using state::On;

namespace procedure {
// The mesh the last cook pass cooked alone, and the mesh storage states kept while only that mesh's passes wrote mesh storage.
struct CookPass {
    state::Entity Mesh{state::Null};
    std::optional<store::Hash128> End;
    std::vector<store::Hash128> Exact;
};
void Reset(CookPass &pass, state::Registry &) { pass = {}; }

using namespace action::mesh;
namespace {
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
    Element Domain{Element::Vertex};
    // Copies the emitted selection into the output record.
    bool Publish{};
    // The mesh entity drawing the input record with current render records, which a node edits in place and repairs, setting Repaired.
    state::Entity Drawn{state::Null};
    bool *Repaired{};
    // The change a restore made to the drawn record, whose vertex ranges and blocks the next drawn position edit refreshes with its own.
    MeshStore::Change *Refresh{};
};

struct Operand {
    uint32_t Attribute{InvalidOffset};
    std::optional<PortValue> Constant;
};
struct NodeSelection {
    // The operand linked into the socket or its constant, which overrides the literal.
    std::optional<Operand> Field;
    // The inline literal, or null for every element.
    const InlineSelection *Literal{};
    // The linked attribute's domain, Vertex for a linked value or constant, or else the domain the literal was captured in.
    Element Domain{Element::Vertex};
};

// InvalidOffset denotes an absent attribute.
struct SocketValue {
    SocketType Type{SocketType::Geometry};
    // The cook owns records when MeshEntity is null, and otherwise borrows them read-only from MeshEntity, which keeps owning them.
    state::Entity MeshEntity{state::Null};
    state::Entity Prototype{state::Null};
    state::Entity Object{state::Null}; // For geometry, the object whose local space the geometry is in, or null for this procedure's mesh space
    uint32_t Instance{InvalidOffset}; // The record an instances value's points each place, or InvalidOffset for points placing nothing
    union {
        uint32_t Record{InvalidOffset};
        uint32_t Attribute;
        AttributeValue Value;
    };
    bool Borrowed() const { return Type == SocketType::Geometry && MeshEntity != state::Null; }
};
// Return the output record, the input for an unchanged operation, or the reason the operation failed.
using Outcome = std::expected<uint32_t, NodeFault>;

// Edited arena blocks, gathered before erasure releases empty blocks.
struct EditedBlocks {
    std::vector<uint32_t> Vertices, Faces, Corners;
};

// Appends the arena blocks of a published edit, including surviving vertices and edges whose incidence changed, before it finishes.
void AppendEditedBlocks(EditedBlocks &edited, const MeshStore &meshes, const MeshTopologyEdit &edit) {
    using Domain = MeshStore::ElementDomain;
    const auto &storage = edit.Chain.Scratch;
    edited.Corners.append_range(edit.OldCornerBlocks);
    if (edit.Repair) {
        edited.Vertices.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Vertex, storage, edit.Repair->Elements[0]));
        edited.Faces.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Face, storage, edit.Repair->Elements[2]));
        if (meshes.Get(edit.StoreId).NormalSectors.Count) edited.Corners.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Halfedge, storage, edit.Repair->Elements[1]));
    }
    edited.Vertices.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Vertex, storage, edit.Output->Retired[0]));
    edited.Faces.append_range(meshes.ArenaBlocks(edit.StoreId, Domain::Face, storage, edit.Output->Retired[1]));
}

void UpdatePoseMembership(state::Registry &r, uint32_t store_id, EditedBlocks edited) {
    auto &buffers = r.Resources.get<GpuBuffers>();
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(store_id);
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
        (stores.UpdateBlocks(store_id, revision, blocks, present), ...);
    };
    update(a.Vertices, record.Vertices, vertices, buffers.VertexBounds, buffers.PosedPositions, buffers.PosedMorphNormalDeltas, buffers.PosedVertexNormals);
    update(a.FaceTriangles, record.FaceData, faces, buffers.PosedFaceNormals);
    const auto corner_members = a.FaceCorners.Blocks.Buffer.GetSpan<MeshElementBlock>();
    const auto has_sectors = [&](uint32_t block) {
        return record.NormalSectors.Count && block < corner_members.size() && corner_members[block].Owner == record.FaceCorners.Index && corner_members[block].Count;
    };
    buffers.PosedSectors.UpdateBlocks(store_id, meshes.GetDerived(store_id).NormalRevision, corners, has_sectors);
}

// The work lives on the repair chain's scratch, and Domain is its ElementRepairMark domain.
struct ElementMarks {
    ElementWork Work;
    uint32_t Domain;
};
// Requires an edit with a repair.
std::array<ElementMarks, 4> RepairMarks(const MeshTopologyEdit &edit) {
    return {{{edit.Repair->Elements[0], 0u}, {edit.Output->Retired[0], 0u}, {edit.RetiredEdges, 1u}, {edit.Repair->Elements[1], 2u}}};
}

// Refresh pose membership, the mesh binding, the marked elements' point and line owners and the selection summaries once the drawn record's edits finish.
void RepairDrawnElements(state::Registry &r, mtl::ComputeChain &chain, state::Entity mesh_entity, uint32_t store_id, EditedBlocks edited, std::span<const ElementMarks> marked) {
    auto &meshes = r.Resources.get<MeshStore>();
    UpdatePoseMembership(r, store_id, std::move(edited));
    RefreshMeshBinding(r, store_id);
    r.Resources.get<GpuSceneState>().EditSelectionDirty = true;
    const auto &owner = meshes.Get(store_id);
    const auto &a = meshes.Arenas();
    const std::array bounds{a.Vertices.HandleBound(owner.Vertices), a.EdgeHalfedges.HandleBound(owner.EdgeData)};
    ElementRepairPushConstants marks{
        .Connectivity = meshes.GetConnectivityRef(store_id),
        .MembershipSlot = {a.Vertices.Blocks.Buffer.Slot, a.EdgeHalfedges.Blocks.Buffer.Slot},
        .Bound = {bounds[0], bounds[1]},
        .Owners = {meshes.Render().ElementMeshlets[2].Ref(owner.ElementMeshlets[2]), meshes.Render().ElementMeshlets[1].Ref(owner.ElementMeshlets[1])},
        .OwnerBound = {owner.ElementMeshlets[2].Count * MeshElementBlockSize, owner.ElementMeshlets[1].Count * MeshElementBlockSize},
        .ScratchSlot = chain.Scratch.Buffer.Slot,
    };
    const std::array words{(std::max(marks.Bound[0], marks.OwnerBound[0]) + 31u) / 32u, (std::max(marks.Bound[1], marks.OwnerBound[1]) + 31u) / 32u};
    std::vector<ElementMeshletRepair> element_repairs;
    if (!marked.empty()) {
        const auto &pipelines = GetMeshPipelines(r);
        for (uint32_t d = 0u; d < 2u; ++d) {
            marks.Added[d] = chain.Scratch.Allocate(std::max(words[d], 1u)).Offset;
            marks.Removed[d] = chain.Scratch.Allocate(std::max(words[d], 1u)).Offset;
            for (const auto run : {marks.Added[d], marks.Removed[d]}) FillScratch(chain, pipelines, run, std::max(words[d], 1u), 0u);
        }
        for (const auto [work, domain] : marked) {
            marks.Work = work;
            marks.WorkCount = WorkBlockCount(chain.Scratch, work) * MeshElementBlockSize;
            marks.WorkDomain = domain;
            if (marks.WorkCount) chain.Threads(pipelines[MeshPass::ElementRepairMark], marks, marks.WorkCount);
        }
        chain.Submit();
        const auto handles = [&](uint32_t run, uint32_t count) { return SetBitHandles(chain.Scratch.Get({run, count})); };
        element_repairs.push_back({.StoreId = store_id, .Topology = 2u, .Added = handles(marks.Added[0], words[0]), .Removed = handles(marks.Removed[0], words[0])});
        element_repairs.push_back({.StoreId = store_id, .Topology = 1u, .Added = handles(marks.Added[1], words[1]), .Removed = handles(marks.Removed[1], words[1])});
    }
    RepairElementMeshlets(r, chain, element_repairs);
    chain.Submit();
    auto &repaired = meshes.WriteRecord(store_id);
    uint32_t mask = 0u;
    for (uint32_t t = 0u; t < 3u; ++t) {
        if (repaired.ElementMeshletBlockCounts[t]) mask |= 1u << t;
    }
    repaired.RenderTopologies = mask ? mask : 4u;
    RefreshElementSelectionSummaries(r, std::array{mesh_entity});
}

void EditTopology(state::Registry &r, state::Entity mesh_entity, const MeshTopologyTask &task, std::vector<ExtrusionSource> *sources = nullptr) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (mesh_entity != state::Null) ReleaseMeshEditWork(r, mesh_entity);
    mtl::ComputeChain chain{meshes.BufferContext(), TopologyScratchWords};
    auto edits = MeshTopologyEdit::Construct(r, chain, std::span{&task, 1u});
    if (sources) *sources = edits.front().ExtrusionSources(r);
    MeshTopologyEdit::PublishAll(r, edits);
    auto &edit = edits.front();
    if (sources && edit.Output) {
        const auto handles = chain.Scratch.Get(edit.Output->Vertices);
        for (auto &source : *sources) source.Output = handles[source.Output];
    }
    if (!edit.Output || mesh_entity == state::Null) {
        if (mesh_entity == state::Null) MeshTopologyEdit::FinishAll(r, std::array{&edit});
        chain.Submit();
        return;
    }
    std::vector<std::pair<state::Entity, const MeshTopologyEdit *>> repairs;
    if ((meshes.Get(edit.StoreId).RenderTopologies & 1u) != 0u || edit.AddedTriangleCount) repairs.emplace_back(mesh_entity, &edit);
    RepairTopologyRender(r, chain, repairs);
    EditedBlocks edited;
    AppendEditedBlocks(edited, meshes, edit);
    MeshTopologyEdit::FinishAll(r, std::array{&edit});
    std::vector<ElementMarks> marked;
    if (edit.Repair) marked = RepairMarks(edit) | std::ranges::to<std::vector>();
    RepairDrawnElements(r, chain, mesh_entity, edit.StoreId, std::move(edited), marked);
}

// Successive in-place topology edits of the record a mesh entity draws, gathered for one repair of the record the last edit leaves.
struct AccumulatedTopologyEdits {
    AccumulatedTriangleEdits Triangles;
    EditedBlocks Blocks;
    // Bit words by handle of the vertices, edges and halfedges standing for their edges that the edits may start or stop drawing as points and lines.
    std::array<std::vector<uint32_t>, 3> Marked;
};

bool RetiresAll(const MeshStore &meshes, uint32_t store_id, const AccumulatedTopologyEdits &accumulated) {
    return accumulated.Triangles.ClusterCount == meshes.Get(store_id).Level0Count;
}

void EditTopologyAccumulated(state::Registry &r, const MeshTopologyTask &task, AccumulatedTopologyEdits &accumulated) {
    auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext(), TopologyScratchWords};
    auto edits = MeshTopologyEdit::Construct(r, chain, std::span{&task, 1u});
    MeshTopologyEdit::PublishAll(r, edits);
    auto &edit = edits.front();
    if (edit.Output && !RetiresAll(meshes, edit.StoreId, accumulated)) {
        AccumulateTriangleEdits(r, accumulated.Triangles, edit);
        AppendEditedBlocks(accumulated.Blocks, meshes, edit);
        if (edit.Repair) {
            for (const auto [work, domain] : RepairMarks(edit)) {
                auto &words = accumulated.Marked[domain];
                ForEachWorkBlock(chain.Scratch, work, [&](uint32_t block, auto masks) {
                    if (words.size() < (block + 1u) * MeshElementBlockWords) words.resize((block + 1u) * MeshElementBlockWords);
                    for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) words[block * MeshElementBlockWords + w] |= masks[w];
                });
            }
        }
    }
    MeshTopologyEdit::FinishAll(r, std::array{&edit});
    chain.Submit();
}

void RepairAccumulated(state::Registry &r, state::Entity mesh_entity, uint32_t store_id, AccumulatedTopologyEdits accumulated) {
    auto &meshes = r.Resources.get<MeshStore>();
    ReleaseMeshEditWork(r, mesh_entity);
    mtl::ComputeChain chain{meshes.BufferContext(), TopologyScratchWords};
    const auto &triangles = accumulated.Triangles;
    if ((meshes.Get(store_id).RenderTopologies & 1u) != 0u || std::ranges::any_of(triangles.Emitted, [](uint32_t word) { return word != 0u; })) {
        RepairTopologyRender(r, chain, mesh_entity, triangles);
    }
    // Only halfedges live after the last edit mark their edges, and the edges of the others retired or took a halfedge an edit emitted.
    const auto &corners = meshes.Arenas().FaceCorners;
    const auto entries = corners.Entries(meshes.Get(store_id).FaceCorners);
    const auto blocks = corners.Blocks.Buffer.GetSpan<MeshElementBlock>();
    auto &halfedges = accumulated.Marked[2];
    for (size_t w = 0u; w < halfedges.size(); ++w) {
        const auto local = w / MeshElementBlockWords;
        halfedges[w] &= local < entries.size() && entries[local] != InvalidOffset ? blocks[entries[local]].Live[w % MeshElementBlockWords] : 0u;
    }
    std::vector<ElementMarks> marked;
    if (std::ranges::any_of(accumulated.Marked, [](const auto &words) { return !words.empty(); })) {
        for (uint32_t d = 0u; d < 3u; ++d) marked.push_back({SeedElementWorkWords(chain.Scratch, accumulated.Marked[d]), d});
    }
    RepairDrawnElements(r, chain, mesh_entity, store_id, std::move(accumulated.Blocks), marked);
}

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

// The vertex ranges of the masks' Refresh, which the edit then clears, join the edited ranges, and its blocks refresh their aggregates with them.
void EditPositions(state::Registry &r, const Masks &masks, const PositionOperationTarget &target, PositionEditOp op, float factor, uint32_t repeat, const PositionOperationOptions &options) {
    if (masks.Drawn == state::Null) {
        ExecuteGeometryPositions(r, target, op, factor, repeat, options);
        return;
    }
    *masks.Repaired = true;
    auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto edited = EncodePositionOperations(meshes, GetMeshPipelines(r), chain, target, op, factor, repeat, options);
    if (!edited) {
        chain.Submit();
        return;
    }
    auto ranges = edited->Ranges;
    if (masks.Refresh) {
        ranges.append_range(masks.Refresh->VertexRanges);
        CoalesceRanges(ranges);
    }
    // The publication refits the record's meshlets before the repair of changed triangles replaces any of them.
    auto repair = RefreshEditedPositions(r, chain, std::array{MeshVertexChanges{masks.Drawn, ranges}});
    PublishEditedPositions(r, chain, std::array{masks.Drawn});
    if (masks.Refresh) PublishRestored(r, chain, target.StoreId, *masks.Refresh);
    if (repair) chain.AfterSubmit(std::move(repair));
    chain.Submit();
}

GeometrySelection Handles(const MeshStore &meshes, uint32_t id, const Masks &masks) { return masks.In ? ReadMasks(meshes, id, *masks.In) : AllHandles(Mesh{meshes, id}); }
SelectionSource SourceOf(const MeshStore &meshes, uint32_t id, const Masks &masks) { return masks.In ? SelectionSource{*masks.In} : SelectionSource{AllHandles(Mesh{meshes, id})}; }
TopologyInput InputOf(const Masks &masks) { return {masks.In ? std::optional<SelectionSource>{*masks.In} : std::nullopt, masks.Domain}; }
GeometrySelection Handles(const MeshStore &meshes, uint32_t id, const TopologyInput &input) {
    if (!input.In) return AllHandles(Mesh{meshes, id});
    if (const auto *listed = std::get_if<GeometrySelection>(&*input.In)) return *listed;
    return ReadMasks(meshes, id, std::get<GeometryMasks>(*input.In));
}

MeshTopologyTask Task(uint32_t id, MeshTopologyOp op, const TopologyInput &input, float param0 = 0.f, float param1 = 0.f, uint32_t flags = 0u) {
    MeshTopologyTask task{.SourceId = id, .Op = op, .Param0 = param0, .Param1 = param1, .Flags = flags};
    if (input.In) task.Selection = *input.In;
    else task.Flags |= TopologyFlagSelectAll;
    return task;
}
void PublishInput(state::Registry &r, uint32_t id, const Masks &masks) {
    if (!masks.Publish || !masks.In) return;
    auto &meshes = r.Resources.get<MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    meshes.SelectMasks(r, chain, id, *masks.In);
    chain.Submit();
}
// Run the dependent tasks in order and return the last task's output record, or the input when no task changed it.
Outcome ExecuteStages(state::Registry &r, std::span<MeshTopologyTask> tasks, const Masks &masks) {
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
Outcome Execute(state::Registry &r, std::optional<MeshTopologyTask> task, const Masks &masks) {
    if (!task) return std::unexpected(NodeFault::Selection);
    return ExecuteStages(r, std::span{&*task, 1u}, masks);
}
// The task `plan` builds from the mesh and its input elements, which an AddPrimitives task takes as its selection.
std::optional<MeshTopologyTask> PlannedTask(state::Registry &r, uint32_t id, const TopologyInput &input, auto &&plan) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto selection = Handles(meshes, id, input);
    std::optional<MeshTopologyTask> task = plan(Mesh{meshes, id}, selection);
    if (task && task->Op == MeshTopologyOp::AddPrimitives) task->Selection = selection;
    return task;
}
Outcome Planned(state::Registry &r, uint32_t id, const Masks &masks, auto &&plan) { return Execute(r, PlannedTask(r, id, InputOf(masks), plan), masks); }
void DeleteEvery(state::Registry &r, uint32_t id, DeleteMode mode) {
    const Mesh mesh{r.Resources.get<const MeshStore>(), id};
    if (mode == DeleteMode::Vertices ? mesh.VertexCount() : mesh.EdgeCount()) {
        const auto task = Task(id, MeshTopologyOp(uint32_t(mode)), {});
        ExecuteGeometryTopology(r, std::span{&task, 1u});
    }
}
// A slide measures from the `reference` vertex, or else from the first selected vertex.
Outcome Reposition(state::Registry &r, uint32_t id, const Masks &masks, PositionEditOp op, float factor, uint32_t repeat, const PositionOperationOptions &options = {}, const Transform &world = {}, uint32_t reference = InvalidOffset) {
    auto &meshes = r.Resources.get<MeshStore>();
    const PositionOperationTarget target{.StoreId = id, .Selection = SourceOf(meshes, id, masks), .World = world, .Reference = reference};
    EditPositions(r, masks, target, op, factor, repeat, options);
    return id;
}
// Rotate the selected faces' corner UVs of a set or colors clockwise for operation 0, counterclockwise for 1, or reverse them for 2.
Outcome EditFaceAttributes(state::Registry &r, uint32_t id, const Masks &masks, bool colors, uint32_t uv_set, uint32_t operation) {
    if (uv_set >= MeshStore::MaxUvSets) return std::unexpected(NodeFault::Parameter);
    auto &meshes = r.Resources.get<MeshStore>();
    const PositionOperationTarget target{.StoreId = id, .Selection = SourceOf(meshes, id, masks)};
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto change = EncodeFaceAttributeOperation(meshes, GetMeshPipelines(r), chain, target, colors, uv_set, operation);
    if (masks.Drawn != state::Null) *masks.Repaired = true;
    if (masks.Drawn == state::Null || !change) {
        chain.Submit();
        return id;
    }
    const auto &a = meshes.Arenas();
    const auto &record = meshes.Get(id);
    const auto &faces = change->Faces;
    const auto bound = std::min(faces.Incidence, a.Triangles.Set(record.TriangleData).BlockCount);
    const auto handles = a.Triangles.HandleBound(record.TriangleData);
    chain.Scratch.ReserveAdditional(1u + ElementWorkWords(handles, bound) + SortElementWorkWords(WorkCapacity(handles, bound)));
    std::vector<std::pair<state::Entity, FaceTriangles>> repairs{{masks.Drawn, EncodeFaceTriangles(r, chain, id, faces)}};
    chain.Submit();
    repairs.front().second.Finish(chain);
    RepairFaceRender(r, chain, repairs);
    chain.Submit();
    return id;
}
uint32_t Axes(bool x, bool y, bool z) { return uint32_t(x) | uint32_t(y) << 1u | uint32_t(z) << 2u; }

// Per-input attribute IDs, where InvalidOffset selects the constant parameter.
struct LinkedAttributes {
    std::array<uint32_t, MaxInputs> Attributes;
    template<auto M> uint32_t Of() const { return Attributes[ParameterSocket<M>]; }
    bool Any() const {
        return std::ranges::any_of(Attributes, [](uint32_t a) { return a != InvalidOffset; });
    }
};
// Returns the attribute when its domain and type match, or InvalidOffset for a constant.
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

Outcome Run(state::Registry &r, const Delete &p, uint32_t id, const Masks &m) { return Execute(r, PlanTopology(r, p, id, InputOf(m)), m); }
Outcome Run(state::Registry &r, const Merge &p, uint32_t id, const Masks &m) { return Execute(r, PlanTopology(r, p, id, InputOf(m)), m); }
// The hash of a node's selection, which is zero for every element and differs between a linked field and any literal.
store::Hash128 SelectionKey(const NodeSelection &selection) {
    if (!selection.Field && !selection.Literal) return {};
    std::vector<uint32_t> words{uint32_t(selection.Domain), uint32_t(selection.Field.has_value()), uint32_t(!selection.Field && selection.Literal->Published)};
    if (selection.Field) return store::HashBytes(std::as_bytes(std::span{words}));
    for (const auto *domain : {&selection.Literal->Vertices, &selection.Literal->Edges, &selection.Literal->Faces}) {
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

vec3 NormalizedSum(std::span<const vec3> vectors) {
    vec3 sum{};
    for (const auto &v : vectors) sum += v;
    return Length(sum) > 0.f ? Normalize(sum) : sum;
}
Outcome RunExtrude(state::Registry &r, state::Entity node, const Extrude &p, uint32_t id, const Masks &m, const NodeSelection &resolved, const LinkedAttributes &linked, TopologyReuse reuse, std::vector<uint32_t> &temporaries, bool retain) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto domain = p.Mode == ExtrudeMode::FacesIndividual ? AttributeDomain::Face : AttributeDomain::Vertex;
    const auto offset = ElementParameter(meshes, linked.Of<&Extrude::Offset>(), domain, AttributeType::Vec3);
    const auto scale = ElementParameter(meshes, linked.Of<&Extrude::OffsetScale>(), domain);
    if (!offset || !scale) return std::unexpected(NodeFault::Attribute);
    const auto key = retain ? SelectionKey(resolved) : store::Hash128{};
    const auto *prior = retain ? r.try_get<const NodeResult>(node) : nullptr;
    const auto *cached = r.try_get<const NodeExtrusion>(node);
    const auto *stage = reuse != TopologyReuse::None && prior && cached && prior->StoreId == id && cached->Mode == p.Mode && cached->Selection == key ? cached : nullptr;
    NodeExtrusion computed{.Selection = key, .Mode = p.Mode};
    std::vector<std::pair<uint32_t, vec3>> offsets;
    const bool fields = *offset != InvalidOffset || *scale != InvalidOffset;
    const bool per_element = fields && p.Mode != ExtrudeMode::Region && p.AlongNormals;
    if (!stage || fields) {
        const auto selection = Handles(meshes, id, m);
        const auto &pipelines = GetMeshPipelines(r);
        const bool faces = domain == AttributeDomain::Face;
        const auto face_normals = !stage || (per_element && faces) ? FaceNormals(meshes, pipelines, id, selection.Faces) : std::vector<vec3>{};
        if (!stage) {
            computed.RegionNormal = NormalizedSum(face_normals);
            stage = &computed;
        }
        if (fields) {
            const auto &elements = faces ? selection.Faces : selection.Vertices;
            const auto vertex_normals = per_element && !faces ? VertexNormals(meshes, pipelines, id, selection.Vertices) : std::vector<vec3>{};
            const auto &normals = faces ? face_normals : vertex_normals;
            for (uint32_t i = 0u; i < elements.size(); ++i) {
                const auto source = elements[i];
                const auto normal = per_element ? normals[i] : stage->RegionNormal;
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
        auto task = *PlanTopology(r, p, id, InputOf(m));
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
        EditPositions(r, m, target, operation, 1.f, 1u, options);
        if (attribute != InvalidOffset) meshes.ReleaseAttribute(attribute);
    }
    if (retain && stage == &computed) r.emplace_or_replace<NodeExtrusion>(node, std::move(computed));
    return id;
}
bool Stepped(const Extrude &p) { return p.Steps > 1u || p.Angle != 0.f; }
} // namespace

MeshTopologyTask PlanTopology(state::Registry &, const Delete &p, uint32_t id, const TopologyInput &input) { return Task(id, MeshTopologyOp(uint32_t(p.Mode)), input); }
MeshTopologyTask PlanTopology(state::Registry &, const Dissolve &p, uint32_t id, const TopologyInput &input) {
    switch (p.Mode) {
        case DissolveMode::Vertices: return Task(id, MeshTopologyOp::DissolveVertices, input);
        case DissolveMode::Edges: return Task(id, MeshTopologyOp::DissolveEdges, input, 0.f, 0.f, p.KeepVertices ? TopologyFlagKeepVertices : 0u);
        case DissolveMode::Faces: return Task(id, MeshTopologyOp::DissolveFaces, input);
        case DissolveMode::Limited: {
            const auto flags = (input.Domain == Element::Face ? TopologyFlagFaceSelection : 0u) | (p.AllBoundaries ? TopologyFlagAllBoundaries : 0u) | (p.DelimitMaterials ? TopologyFlagDelimitMaterial : 0u) | (p.DelimitSharpEdges ? TopologyFlagDelimitSharp : 0u) | (p.DelimitUVs ? TopologyFlagDelimitUV : 0u);
            return Task(id, MeshTopologyOp::DissolveLimited, input, std::clamp(p.Angle, 0.f, .5f * std::numbers::pi_v<float>), 0.f, flags);
        }
        case DissolveMode::Degenerate: return Task(id, MeshTopologyOp::DissolveDegenerate, input, std::max(p.Distance, 0.f));
    }
    std::unreachable();
}
std::optional<MeshTopologyTask> PlanTopology(state::Registry &r, const Merge &p, uint32_t id, const TopologyInput &input) {
    return PlannedTask(r, id, input, [&](Mesh mesh, const GeometrySelection &selection) {
        const auto &vertices = selection.Vertices;
        const auto reference = vertices.empty() ? InvalidOffset : p.Mode == MergeMode::Last ? vertices.back() :
                                                                                              vertices.front();
        return MergeTask(r, mesh, selection, p.Mode, std::max(p.Distance, 0.f), reference);
    });
}
std::vector<MeshTopologyTask> PlanTopology(state::Registry &r, const ConvexHull &, uint32_t id, const TopologyInput &input) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    return ConvexHullTasks(r, Mesh{meshes, id}, Handles(meshes, id, input));
}
MeshTopologyTask PlanTopology(state::Registry &, const Wireframe &p, uint32_t id, const TopologyInput &input) {
    const auto flags = (p.Even ? TopologyFlagEvenOffset : 0u) | (p.Boundary ? TopologyFlagWireBoundary : 0u) | (p.Relative ? TopologyFlagWireRelative : 0u) | (p.Replace ? TopologyFlagWireReplace : 0u);
    return Task(id, MeshTopologyOp::Wireframe, input, std::max(p.Thickness, 0.f), std::clamp(p.Offset, -1.f, 1.f), flags);
}
MeshTopologyTask PlanTopology(state::Registry &, const Bevel &p, uint32_t id, const TopologyInput &input) {
    auto task = Task(id, p.Vertices ? MeshTopologyOp::BevelVertices : MeshTopologyOp::BevelEdges, input, std::max(p.Width, 0.f));
    task.Steps = std::max(p.Segments, 1u);
    return task;
}
MeshTopologyTask PlanTopology(state::Registry &, const TrisToQuads &, uint32_t id, const TopologyInput &input) { return Task(id, MeshTopologyOp::TrisToQuads, input); }
MeshTopologyTask PlanTopology(state::Registry &, const SplitFaces &p, uint32_t id, const TopologyInput &input) {
    switch (p.Target) {
        case SplitTarget::Triangle: return Task(id, MeshTopologyOp::Triangulate, input);
        case SplitTarget::Convex: return Task(id, MeshTopologyOp::SplitConcaveFaces, input);
        case SplitTarget::Planar: return Task(id, MeshTopologyOp::SplitNonplanarFaces, input, std::clamp(p.Angle, 0.f, std::numbers::pi_v<float>));
    }
    std::unreachable();
}
MeshTopologyTask PlanTopology(state::Registry &, const FlipNormals &, uint32_t id, const TopologyInput &input) { return Task(id, MeshTopologyOp::FlipNormals, input); }
MeshTopologyTask PlanTopology(state::Registry &, const Split &, uint32_t id, const TopologyInput &input) { return Task(id, MeshTopologyOp::SplitGeometry, input); }
MeshTopologyTask PlanTopology(state::Registry &, const Duplicate &, uint32_t id, const TopologyInput &input) { return Task(id, MeshTopologyOp::DuplicateGeometry, input); }
// Only a region extrudes in steps, about a nonzero axis.
std::optional<MeshTopologyTask> PlanTopology(state::Registry &r, const Extrude &p, uint32_t id, const TopologyInput &input) {
    if (!Stepped(p)) {
        const auto op = p.Mode == ExtrudeMode::Vertices ? MeshTopologyOp::ExtrudeVertices : p.Mode == ExtrudeMode::Edges ? MeshTopologyOp::ExtrudeEdges :
            p.Mode == ExtrudeMode::FacesIndividual                                                                       ? MeshTopologyOp::ExtrudeFacesIndividual :
                                                                                                                           MeshTopologyOp::ExtrudeRegion;
        return Task(id, op, input);
    }
    if (p.Mode != ExtrudeMode::Region || Dot(p.Axis, p.Axis) <= 0.f) return std::nullopt;
    const auto steps = std::max(p.Steps, 1u);
    return PlannedTask(r, id, input, [&](Mesh mesh, const GeometrySelection &selection) {
        const auto direction = p.AlongNormals ? NormalizedSum(FaceNormals(r.Resources.get<const MeshStore>(), GetMeshPipelines(r), id, selection.Faces)) : p.Offset;
        return ExtrudeStepsTask(mesh, selection, Element::None, steps, ToMat3(AngleAxis(p.Angle / float(steps), Normalize(p.Axis))), direction * p.OffsetScale / float(steps), p.Center);
    });
}
std::optional<MeshTopologyTask> PlanTopology(state::Registry &r, const Subdivide &p, uint32_t id, const TopologyInput &input) {
    if (p.MaxLength > 0.f) return PlannedTask(r, id, input, [&](Mesh mesh, const GeometrySelection &selection) { return SubdivideLengthTask(r, mesh, selection, p.MaxLength); });
    return Task(id, MeshTopologyOp::Subdivide, input, float(std::max(p.Cuts, 1u)), 0.f, p.NewLoopsOnly ? TopologyFlagNewLoopsOnly : 0u);
}
MeshTopologyTask PlanTopology(state::Registry &, const EdgeSplit &p, uint32_t id, const TopologyInput &input) { return Task(id, MeshTopologyOp::EdgeSplit, input, 0.f, 0.f, p.SelectCopies ? TopologyFlagSelectCopies : 0u); }
MeshTopologyTask PlanTopology(state::Registry &, const Poke &p, uint32_t id, const TopologyInput &input) { return Task(id, MeshTopologyOp::Poke, input, p.Offset); }
MeshTopologyTask PlanTopology(state::Registry &, const Inset &p, uint32_t id, const TopologyInput &input) { return Task(id, p.Individual ? MeshTopologyOp::InsetIndividual : MeshTopologyOp::InsetRegion, input, std::max(p.Thickness, 0.f), p.Depth, p.Even ? TopologyFlagEvenOffset : 0u); }
MeshTopologyTask PlanTopology(state::Registry &, const Solidify &p, uint32_t id, const TopologyInput &input) { return Task(id, MeshTopologyOp::Solidify, input, p.Thickness); }
MeshTopologyTask PlanTopology(state::Registry &, const ConnectVertices &, uint32_t id, const TopologyInput &input) { return Task(id, MeshTopologyOp::ConnectVertices, input); }
std::optional<MeshTopologyTask> PlanTopology(state::Registry &r, const BeautifyFaces &, uint32_t id, const TopologyInput &input) {
    return PlannedTask(r, id, input, [&](Mesh mesh, const GeometrySelection &selection) { return BeautifyFaceTask(r, mesh, selection); });
}

namespace {
Outcome RunExtrudeSteps(state::Registry &r, const Extrude &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    if (linked.Of<&Extrude::Offset>() != InvalidOffset || linked.Of<&Extrude::OffsetScale>() != InvalidOffset) return std::unexpected(NodeFault::Parameter);
    auto task = PlanTopology(r, p, id, InputOf(m));
    if (!task) return std::unexpected(NodeFault::Parameter);
    return Execute(r, std::move(task), m);
}
Outcome Run(state::Registry &r, const Duplicate &p, uint32_t id, const Masks &m) { return Execute(r, PlanTopology(r, p, id, InputOf(m)), m); }
Outcome Run(state::Registry &r, const Split &p, uint32_t id, const Masks &m) { return Execute(r, PlanTopology(r, p, id, InputOf(m)), m); }
Outcome Run(state::Registry &r, const Dissolve &p, uint32_t id, const Masks &m) { return Execute(r, PlanTopology(r, p, id, InputOf(m)), m); }
Outcome Run(state::Registry &r, const Subdivide &p, uint32_t id, const Masks &m) { return Execute(r, PlanTopology(r, p, id, InputOf(m)), m); }
Outcome Run(state::Registry &r, const SubdivisionSurface &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    if (!Mesh{meshes, id}.FaceCount() || !p.Levels) return id;
    std::vector<MeshTopologyTask> levels;
    for (uint32_t level = 0u; level < p.Levels; ++level) levels.push_back(Task(id, MeshTopologyOp::Subdivide, {}, 1.f, 0.f, TopologyFlagFacePoints));
    return ExecuteStages(r, levels, m);
}
Outcome Run(state::Registry &r, const SmoothSurface &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    if (!Mesh{meshes, id}.FaceCount()) return id;
    if (!Mesh{meshes, id}.HasClosedSurface()) return std::unexpected(NodeFault::NotClosed);
    auto tasks = SmoothSurfaceTasks(r, Mesh{meshes, id}, p.Cuts);
    return ExecuteStages(r, tasks, m);
}
Outcome Run(state::Registry &r, const SplitFaces &p, uint32_t id, const Masks &m) { return Execute(r, PlanTopology(r, p, id, InputOf(m)), m); }
Outcome Run(state::Registry &r, const TrisToQuads &p, uint32_t id, const Masks &m) { return Execute(r, PlanTopology(r, p, id, InputOf(m)), m); }
Outcome Run(state::Registry &r, const Poke &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    return ExecuteElement(r, PlanTopology(r, p, id, InputOf(m)), m, AttributeDomain::Face, linked.Of<&Poke::Offset>());
}
Outcome Run(state::Registry &r, const FlipNormals &p, uint32_t id, const Masks &m) { return Execute(r, PlanTopology(r, p, id, InputOf(m)), m); }
Outcome Run(state::Registry &r, const EdgeSplit &p, uint32_t id, const Masks &m) { return Execute(r, PlanTopology(r, p, id, InputOf(m)), m); }
Outcome Run(state::Registry &r, const Inset &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    return ExecuteElement(r, PlanTopology(r, p, id, InputOf(m)), m, AttributeDomain::Vertex, linked.Of<&Inset::Thickness>(), linked.Of<&Inset::Depth>());
}
Outcome Run(state::Registry &r, const Solidify &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    return ExecuteElement(r, PlanTopology(r, p, id, InputOf(m)), m, AttributeDomain::Vertex, linked.Of<&Solidify::Thickness>());
}
Outcome Run(state::Registry &r, const Wireframe &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    return ExecuteElement(r, PlanTopology(r, p, id, InputOf(m)), m, AttributeDomain::Vertex, linked.Of<&Wireframe::Thickness>());
}
Outcome Run(state::Registry &r, const Bevel &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    return ExecuteElement(r, PlanTopology(r, p, id, InputOf(m)), m, p.Vertices ? AttributeDomain::Vertex : AttributeDomain::Edge, linked.Of<&Bevel::Width>());
}
Outcome Run(state::Registry &r, const SmoothVertices &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    return RepositionBy(r, id, m, PositionEditOp::Smooth, p.Factor, std::clamp(p.Repeat, 1u, 1000u), linked.Of<&SmoothVertices::Factor>(), {.Axes = Axes(p.X, p.Y, p.Z)});
}
Outcome Run(state::Registry &r, const ShrinkFatten &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    return RepositionBy(r, id, m, PositionEditOp::ShrinkFatten, p.Distance, 1u, linked.Of<&ShrinkFatten::Distance>(), {.Flags = (p.Even ? PositionEditEvenOffset : 0u) | (m.Domain == Element::Face ? PositionEditSelectedFaceNormals : 0u)});
}
Outcome Run(state::Registry &r, const Randomize &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
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
    return Reposition(r, id, m, PositionEditOp::VertexSlide, factor, 1u, {.Flags = p.Clamp ? 0u : PositionEditSlideUnclamped, .Slide = PositionSlide{p.Direction, p.Even, p.Flipped}}, p.Frame, p.Reference);
}
Outcome Run(state::Registry &r, const EdgeSlide &p, uint32_t id, const Masks &m) {
    const auto factor = p.Clamp ? std::clamp(p.Factor, -1.f, 1.f) : p.Factor;
    return Reposition(r, id, m, PositionEditOp::EdgeSlide, factor, 1u, {.Flags = p.Clamp ? 0u : PositionEditSlideUnclamped, .EdgeSlide = PositionSlide{p.Direction, p.Even, p.Flipped}}, p.Frame, p.Reference);
}
Outcome Run(state::Registry &r, const SnapSymmetry &p, uint32_t id, const Masks &m) {
    if (uint32_t(p.Axis) > 2u) return std::unexpected(NodeFault::Parameter);
    return Reposition(r, id, m, PositionEditOp::SnapSymmetry, std::clamp(p.Factor, 0.f, 1.f), 1u, {.Axes = 1u << uint32_t(p.Axis), .Flags = p.Negative ? PositionEditSymmetryNegative : 0u, .SnapSymmetry = PositionSymmetry{p.Threshold, p.Center}});
}
Outcome Run(state::Registry &r, const Bisect &p, uint32_t id, const Masks &m) {
    if (Dot(p.Normal, p.Normal) <= 0.f) return std::unexpected(NodeFault::Parameter);
    const auto &meshes = r.Resources.get<const MeshStore>();
    auto cut = Execute(r, PlaneCutTask(Mesh{meshes, id}, p.Point, p.Normal), m);
    for (const auto [clear, side] : {std::pair{p.ClearInner, PlaneSideKind::Negative}, std::pair{p.ClearOuter, PlaneSideKind::Positive}}) {
        if (!cut || !clear) continue;
        if (auto task = PlaneSideDeleteTask(r, Mesh{meshes, *cut}, p.Point, p.Normal, side, PlaneSideTolerance)) cut = Execute(r, std::move(task), m);
    }
    return cut;
}
Outcome Run(state::Registry &r, const Project &p, uint32_t id, const Masks &) {
    if (Dot(p.Normal, p.Normal) <= 0.f) return std::unexpected(NodeFault::Parameter);
    const auto plane = PlaneAt(p.Point, p.Normal);
    const auto loops = RegionLoops(r, plane, std::array{RegionOperand{.Record = id, .Silhouette = true}}, RegionFill::Positive);
    if (!loops || loops->empty()) return std::unexpected(NodeFault::Empty);
    return CreateShadedMesh(r, {.Data = RegionData(plane, *loops)});
}
Outcome Run(state::Registry &r, const ConvexHull &p, uint32_t id, const Masks &m) {
    auto tasks = PlanTopology(r, p, id, InputOf(m));
    if (tasks.empty()) return std::unexpected(NodeFault::Selection);
    return ExecuteStages(r, tasks, m);
}
Outcome Run(state::Registry &r, const Simplify &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    if (m.Drawn == state::Null) {
        if (!SimplifyGeometry(r, id, selection, p.Tolerance, p.Ratio, [&](const MeshTopologyTask &task) { Execute(r, task, m); })) return std::unexpected(NodeFault::Selection);
        return id;
    }
    AccumulatedTopologyEdits accumulated;
    const auto round = [&](MeshTopologyTask task) {
        if (m.Publish) PublishStoreSelection(meshes, task);
        EditTopologyAccumulated(r, task, accumulated);
    };
    if (!SimplifyGeometry(r, id, selection, p.Tolerance, p.Ratio, round)) return std::unexpected(NodeFault::Selection);
    if (!RetiresAll(meshes, id, accumulated)) {
        *m.Repaired = true;
        RepairAccumulated(r, m.Drawn, id, std::move(accumulated));
    }
    return id;
}
Outcome Run(state::Registry &r, RecalculateNormals, uint32_t id, const Masks &m) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto selection = Handles(meshes, id, m);
    auto faces = RecalculateFaceFlips(meshes, GetMeshPipelines(r), id, selection);
    if (faces.empty()) return id;
    faces.insert(faces.begin(), uint32_t(faces.size()));
    return Execute(r, MeshTopologyTask{.SourceId = id, .Op = MeshTopologyOp::FlipNormals, .List = std::move(faces), .Selection = selection}, m);
}
Outcome Run(state::Registry &r, const ConnectVertices &p, uint32_t id, const Masks &m) { return Execute(r, PlanTopology(r, p, id, InputOf(m)), m); }
Outcome Run(state::Registry &r, const Fill &p, uint32_t id, const Masks &m) {
    return Planned(r, id, m, [&](Mesh mesh, const GeometrySelection &selection) { return FillTask(r, mesh, selection, p.Sides); });
}
Outcome Run(state::Registry &r, BridgeEdgeLoops, uint32_t id, const Masks &m) {
    return Planned(r, id, m, [&](Mesh mesh, const GeometrySelection &selection) { return BridgeEdgeLoopsTask(r, mesh, selection); });
}
Outcome Run(state::Registry &r, const GridFill &p, uint32_t id, const Masks &m) {
    return Planned(r, id, m, [&](Mesh mesh, const GeometrySelection &selection) { return GridFillTask(r, mesh, selection, p.Span); });
}
// The task selects only edges because the kernel selects each rotated edge's apexes as the vertices to connect.
Outcome Run(state::Registry &r, EdgeRotate, uint32_t id, const Masks &m) {
    return Execute(r, MeshTopologyTask{.SourceId = id, .Op = MeshTopologyOp::RotateEdges, .SelectionElement = Element::Edge, .Selection = GeometrySelection{.Edges = Handles(r.Resources.get<const MeshStore>(), id, m).Edges}}, m);
}
Outcome Run(state::Registry &r, const BeautifyFaces &p, uint32_t id, const Masks &m) { return Execute(r, PlanTopology(r, p, id, InputOf(m)), m); }
// Repeat on the selected elements that survive each iteration, until an iteration changes nothing.
Outcome Run(state::Registry &r, const Unsubdivide &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    auto selection = Handles(meshes, id, m);
    auto edited = Execute(r, UnsubdivideTask(r, Mesh{meshes, id}, selection), m);
    for (uint32_t i = 1u; edited && i < std::clamp(p.Iterations, 1u, 1000u); ++i) {
        if (m.Publish) selection = EditorGeometrySelection(meshes, *edited);
        for (const auto [element, handles] : {std::pair{Element::Vertex, &selection.Vertices}, std::pair{Element::Edge, &selection.Edges}, std::pair{Element::Face, &selection.Faces}}) {
            std::erase_if(*handles, [&](uint32_t handle) { return !meshes.IsLiveElement(*edited, element, ElementHandle{handle}); });
        }
        auto task = UnsubdivideTask(r, Mesh{meshes, *edited}, selection);
        if (!task) break;
        edited = Execute(r, std::move(task), m);
    }
    return edited;
}
Outcome Run(state::Registry &r, const Knife &p, uint32_t id, const Masks &m) {
    auto task = KnifeTask(Mesh{r.Resources.get<const MeshStore>(), id}, p.ViewProjection * ToMatrix(p.Frame), p.Extent, p.Start, p.End);
    task.Flags |= TopologyFlagSelectAll;
    return Execute(r, std::move(task), m);
}
Outcome Run(state::Registry &r, const CycleCornerAttribute &p, uint32_t id, const Masks &m) {
    return EditFaceAttributes(r, id, m, p.Attribute == CornerAttribute::Colors, p.UVSet, uint32_t(p.Cycle));
}
Outcome EditSharpness(state::Registry &r, uint32_t id, const Masks &masks, SelectionSource selection, EditSharpnessOperation operation, bool value, float angle = 0.f) {
    auto &meshes = r.Resources.get<MeshStore>();
    const SharpnessOperationTarget target{id, std::move(selection)};
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto change = ExecuteGeometrySharpness(r, chain, target, operation, value, angle);
    if (masks.Drawn != state::Null) *masks.Repaired = true;
    if (masks.Drawn != state::Null && change) {
        r.Resources.get<GpuSceneState>().EditSelectionDirty = true;
        RepairFaceRender(r, chain, std::array{std::pair{masks.Drawn, change->Triangles}});
    }
    chain.Submit();
    return id;
}
Outcome Run(state::Registry &r, const MarkSharp &p, uint32_t id, const Masks &m) {
    const auto operation = m.Domain == Element::Face ? EditSharpnessOperation::SetSelectedFaces : m.Domain == Element::Edge ? EditSharpnessOperation::SetSelectedEdges :
                                                                                                                              EditSharpnessOperation::SetVertexEdges;
    return EditSharpness(r, id, m, SourceOf(r.Resources.get<const MeshStore>(), id, m), operation, p.Sharp);
}
Outcome Run(state::Registry &r, const Shade &p, uint32_t id, const Masks &m) {
    if (p.Mode == ShadeMode::ByAngle) return EditSharpness(r, id, m, GeometrySelection{}, EditSharpnessOperation::SmoothByAngle, false, p.Angle);
    return EditSharpness(r, id, m, GeometrySelection{}, p.Mode == ShadeMode::Smooth ? EditSharpnessOperation::SmoothAll : EditSharpnessOperation::SetAllFaces, p.Mode == ShadeMode::Flat);
}
Outcome Run(state::Registry &r, const TransformElements &p, uint32_t id, const Masks &m) {
    const bool mirrors = Dot(p.Mirror, p.Mirror) > 0.f;
    if (p.Delta == Transform{} && !mirrors) return id;
    return Reposition(r, id, m, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{p.Delta, p.Pivot, mirrors ? Normalize(p.Mirror) : vec3{}}}, p.Frame);
}
Outcome Run(state::Registry &r, MeshToPoints, uint32_t id, const Masks &) {
    DeleteEvery(r, id, DeleteMode::OnlyEdgesAndFaces);
    return id;
}
// The record holds the lattice points the Value attribute was evaluated on.
Outcome Run(state::Registry &r, const LevelSet &p, uint32_t id, uint32_t attribute) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    if (attribute == InvalidOffset || !ElementParameter(meshes, attribute, AttributeDomain::Vertex)) return std::unexpected(NodeFault::Attribute);
    const auto lattice = *LatticeOver(p.Min, p.Max, p.EdgeLength);
    const auto task = LevelSetTask(meshes, id, lattice, attribute, p.Level);
    const auto result = ExecuteGeometryTopology(r, std::span{&task, 1u}).front();
    if (!result.Changed) return std::unexpected(NodeFault::Empty);
    return result.GeometryId;
}
Outcome Run(state::Registry &r, const SetPosition &p, uint32_t id, uint32_t attribute, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    if (attribute == InvalidOffset || !ElementParameter(meshes, attribute, AttributeDomain::Vertex, AttributeType::Vec3)) return std::unexpected(NodeFault::Attribute);
    return Reposition(r, id, m, PositionEditOp::Attribute, 1.f, 1u, {.Flags = p.Offset ? PositionEditAttributeOffset : 0u, .Attribute = meshes.AttributeRef(attribute)});
}

std::string_view NameOf(const AttributeName &name) { return {name.data(), strnlen(name.data(), name.size())}; }

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

// Add the Bool attribute in the domain holding the masked elements, or every element for absent masks.
std::expected<uint32_t, MeshStore::AttributeError> AddSelectionAttribute(MeshStore &meshes, uint32_t id, AttributeDomain domain, std::string_view name, const std::optional<GeometryMasks> &masks) {
    const auto attribute = meshes.AddAttribute(id, domain, AttributeType::Bool, name, {masks ? 0u : 1u});
    if (!attribute || !masks) return attribute;
    const auto element = ElementOf(domain);
    const auto handles = ReadMasks(meshes, id, element, masks->Get(element));
    if (handles.empty()) return attribute;
    const auto count = handles.back() + 1u;
    const auto values = std::make_unique<bool[]>(count);
    for (const auto handle : handles) values[handle] = true;
    meshes.SetAttribute<bool>(*attribute, 0u, std::span<const bool>{values.get(), count});
    return attribute;
}
Outcome Store(MeshStore &meshes, const StoreSelection &p, uint32_t id, const Masks &masks) {
    const auto name = NameOf(p.Name);
    if (name.empty()) return std::unexpected(NodeFault::Parameter);
    if (const auto existing = meshes.FindAttribute(id, name); existing != InvalidOffset) meshes.ReleaseAttribute(existing);
    return AddSelectionAttribute(meshes, id, AttributeDomainOf(masks.Domain), name, masks.In) ? Outcome{id} : std::unexpected(NodeFault::Attribute);
}

Outcome Run(state::Registry &r, const InstanceOnPoints &p, uint32_t id, const Masks &, const LinkedAttributes &linked) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto rotation = ElementParameter(meshes, linked.Of<&InstanceOnPoints::Rotation>(), AttributeDomain::Vertex, AttributeType::Vec4);
    const auto scale = ElementParameter(meshes, linked.Of<&InstanceOnPoints::Scale>(), AttributeDomain::Vertex, AttributeType::Vec3);
    DeleteEvery(r, id, DeleteMode::OnlyEdgesAndFaces);
    for (const auto &[name, type, value, attribute] : {std::tuple{InstanceRotation, AttributeType::Vec4, Words(p.Rotation), rotation.value_or(InvalidOffset)}, std::tuple{InstanceScale, AttributeType::Vec3, Words(p.Scale), scale.value_or(InvalidOffset)}}) {
        if (const auto existing = meshes.FindAttribute(id, name); existing != InvalidOffset) meshes.ReleaseAttribute(existing);
        if (attribute != InvalidOffset) (void)meshes.CopyAttribute(attribute, name);
        else (void)meshes.AddAttribute(id, AttributeDomain::Vertex, type, name, value);
    }
    if (!rotation) return std::unexpected(rotation.error());
    return scale ? Outcome{id} : std::unexpected(scale.error());
}

GeometryMasks MasksAt(const BufferArena<SelectionBlock> &arena, const std::array<uint32_t, 3> &first) { return {{&arena, first[0]}, {&arena, first[1]}, {&arena, first[2]}}; }
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
GeometryMasks ZeroedMasks(MeshStore &meshes, std::optional<BufferArena<SelectionBlock>> &arena, uint32_t id, auto &&fill) {
    const auto ranges = AllocateMasks(meshes, arena, id);
    for (uint32_t d = 0u; d < 3u; ++d) {
        const auto masks = arena->GetMutable(ranges[d]);
        std::ranges::fill(masks, SelectionBlock{});
        fill(d, masks);
    }
    return MasksAt(*arena, {ranges[0].Offset, ranges[1].Offset, ranges[2].Offset});
}
// The masks of the inline selection, or none for every element.
std::optional<GeometryMasks> UploadSelection(state::Registry &r, std::optional<BufferArena<SelectionBlock>> &arena, uint32_t id, const InlineSelection *selection) {
    if (!selection) return {};
    auto &meshes = r.Resources.get<MeshStore>();
    const auto held = selection->Published ? EditorGeometrySelection(meshes, id) : GeometrySelection{};
    const std::array lists{&selection->Vertices, &selection->Edges, &selection->Faces};
    const auto uploaded = ZeroedMasks(meshes, arena, id, [&](uint32_t d, std::span<SelectionBlock> masks) {
        const auto entries = WithDomain(meshes.Arenas(), SelectionDomains[d], [&](const auto &elements) { return elements.Entries(DomainSet(meshes.Get(id), SelectionDomains[d])); });
        // A literal can predate the record, so its blocks past the record's table select nothing.
        const auto write = [&](uint32_t local, const SelectionBlock &words) {
            if (local < entries.size() && entries[local] != InvalidOffset) masks[entries[local]] = words;
        };
        if (!selection->Published) ForEachSelectedBlock(*lists[d], write);
        else if (SelectionElements[d] == selection->Domain) ForEachHandleBlock(held.Get(selection->Domain), write);
    });
    if (selection->Published) {
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.DeriveSelectionMasks(r, chain, id, uploaded, selection->Domain);
        chain.Submit();
    }
    return uploaded;
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
AttributeInput ConstantInput(AttributeType type, auto value) { return {.Source = AttributeSource::Constant, .Type = type, .Value = Words(value)}; }
AttributeInput InputOf(const MeshStore &meshes, const Operand &operand, AttributeInput constant) {
    if (operand.Attribute != InvalidOffset) return {.Source = AttributeSource::Named, .Type = meshes.GetAttribute(operand.Attribute).Type, .Attribute = meshes.AttributeRef(operand.Attribute)};
    if (operand.Constant) return {.Source = AttributeSource::Constant, .Type = operand.Constant->Type, .Value = operand.Constant->Words};
    return constant;
}
// The linked operands' shared domain, or none for mismatched domains, an unsupported type or constants only.
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

std::optional<uint32_t> Compute(state::Registry &r, uint32_t id, const Curvature &p, std::string_view name, const Operands &) { return CurvatureField(r, id, p.Kind, name); }
std::optional<uint32_t> SeedAttribute(const MeshStore &meshes, const Operand &operand) {
    if (operand.Attribute == InvalidOffset) return {};
    const auto &attribute = meshes.GetAttribute(operand.Attribute);
    if (attribute.Type != AttributeType::Bool || attribute.Domain == AttributeDomain::Corner) return {};
    return operand.Attribute;
}
std::optional<uint32_t> Compute(state::Registry &r, uint32_t id, const Boundary &p, std::string_view name, const Operands &) {
    if (p.Domain == AttributeDomain::Corner) return {};
    return BoundaryField(r, id, p.Domain, name);
}
std::optional<uint32_t> Compute(state::Registry &r, uint32_t id, const PlaneSide &p, std::string_view name, const Operands &) {
    if (p.Domain == AttributeDomain::Corner) return {};
    return PlaneSideField(r, id, p.Domain, p.Point, p.Normal, p.Side, p.Tolerance, name);
}
std::optional<uint32_t> Compute(state::Registry &r, uint32_t id, Linked, std::string_view name, const Operands &in) {
    const auto seed = SeedAttribute(r.Resources.get<const MeshStore>(), in.A);
    return seed ? LinkedField(r, id, *seed, name) : std::nullopt;
}
std::optional<uint32_t> Compute(state::Registry &r, uint32_t id, const Adjacent &p, std::string_view name, const Operands &in) {
    const auto seed = SeedAttribute(r.Resources.get<const MeshStore>(), in.A);
    return seed ? AdjacentField(r, id, *seed, p.Mode == AdjacentMode::More, p.Steps, name) : std::nullopt;
}
std::optional<uint32_t> Compute(state::Registry &r, uint32_t id, EdgeLoop, std::string_view name, const Operands &in) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto seed = SeedAttribute(meshes, in.A);
    return seed && meshes.GetAttribute(*seed).Domain == AttributeDomain::Edge ? EdgeLoopField(r, id, *seed, name) : std::nullopt;
}
std::optional<uint32_t> Compute(state::Registry &r, uint32_t id, EdgeRing, std::string_view name, const Operands &in) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto seed = SeedAttribute(meshes, in.A);
    return seed && meshes.GetAttribute(*seed).Domain == AttributeDomain::Edge ? EdgeRingField(r, id, *seed, name) : std::nullopt;
}
template<typename K>
concept Computed = requires(state::Registry &r, uint32_t id, const K &p, std::string_view name, const Operands &in) { Compute(r, id, p, name, in); };

constexpr std::string_view TemporaryPrefix = ".cook.";
std::string TemporaryName(uint32_t node, uint32_t socket) { return std::format("{}{}.{}", TemporaryPrefix, node, socket); }
void ReleaseTemporaries(MeshStore &meshes, uint32_t id) {
    std::vector<uint32_t> temporaries;
    meshes.ForEachNamedAttribute(id, [&](uint32_t attribute, const auto &a) {
        if (a.NameView().starts_with(TemporaryPrefix)) temporaries.push_back(attribute);
    });
    for (const auto attribute : temporaries) meshes.ReleaseAttribute(attribute);
}

Operand OperandOf(const CookPlan &plan, std::span<const SocketValue> values, uint32_t node, uint32_t input) {
    if (input >= InputCount[plan.Nodes[node].Kind]) return {};
    const auto link = plan.Inputs[plan.InputFirst[node] + input];
    if (link.Constant) return {.Constant = link.Constant};
    if (link.FromNode == NoNode) return {};
    const auto &source = values[plan.SocketFirst[link.FromNode] + link.FromSocket];
    if (source.Type == SocketType::Value) return {.Constant = PortValue{source.Value, AttributeType(std::countr_zero(KindSockets(plan.Nodes[link.FromNode].Kind)[link.FromSocket].Types))}};
    return {source.Type == SocketType::Attribute ? source.Attribute : InvalidOffset};
}
// A socket constant, such as a subnet port's default, is a Bool field operand.
// A kind without a Selection socket selects every element.
NodeSelection ResolveSelection(const state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node) {
    const auto [entity, kind] = plan.Nodes[node];
    const auto sockets = KindSockets(kind);
    const auto socket = uint32_t(std::ranges::find(sockets, SocketType::Selection, &Socket::Type) - sockets.begin());
    if (socket == sockets.size()) return {};
    const auto *literal = r.try_get<const InlineSelection>(entity);
    NodeSelection selection{.Literal = literal && !literal->All ? literal : nullptr, .Domain = literal ? literal->Domain : Element::Vertex};
    if (const auto &link = plan.Inputs[plan.InputFirst[node] + socket]; link.FromNode == NoNode && !link.Constant) return selection;
    selection.Field = OperandOf(plan, values, node, socket);
    const auto attribute = selection.Field->Attribute;
    selection.Domain = attribute != InvalidOffset ? ElementOf(r.Resources.get<const MeshStore>().GetAttribute(attribute).Domain) : Element::Vertex;
    return selection;
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
LinkedAttributes LinkedParameters(const CookPlan &plan, std::span<const SocketValue> values, uint32_t node) {
    const auto kind = plan.Nodes[node].Kind;
    const auto sockets = KindSockets(kind);
    const auto [first, end] = ParameterRange(kind);
    LinkedAttributes linked;
    linked.Attributes.fill(InvalidOffset);
    for (auto s = first; s < end; ++s) {
        if (sockets[s].Type == SocketType::Attribute) linked.Attributes[s] = OperandOf(plan, values, node, s).Attribute;
    }
    return linked;
}

struct Evaluation {
    std::vector<uint32_t> Temporaries;
    NodeSelection Selection;
    std::optional<GeometryMasks> In; // The selection's masks, complete in every domain, or none for every element
    bool Failed{}; // A linked selection reads no bool attribute of a selectable domain
};

// Mark invalid attribute nodes and pass through their first input.
Evaluation EvaluateInputs(state::Registry &r, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, std::optional<BufferArena<SelectionBlock>> &arena) {
    auto &meshes = r.Resources.get<MeshStore>();
    auto upstream = Upstream(plan, node, [&](uint32_t n) { return IsAttributeKind(plan.Nodes[n].Kind); });
    upstream[node] = false;
    Evaluation evaluation;
    std::vector<AttributeJob> jobs;
    // A computed field reads its seed attributes on the host side of a submit, so the prepared jobs before it run first.
    const auto flush = [&] {
        if (jobs.empty()) return;
        mtl::ComputeChain chain{meshes.BufferContext()};
        const auto &pipelines = GetMeshPipelines(r);
        for (const auto &job : jobs) chain.Threads(pipelines[job.Pass], job.Pc, job.Pc.Blocks.Count * MeshElementBlockSize);
        chain.Submit();
        jobs.clear();
    };
    for (const auto n : plan.Order) {
        if (!upstream[n]) continue;
        const auto [entity, kind] = plan.Nodes[n];
        const Operands in{OperandOf(plan, values, n, 0u), OperandOf(plan, values, n, 1u), OperandOf(plan, values, n, 2u)};
        auto &output = values[plan.SocketFirst[n] + InputCount[kind]];
        output.Type = SocketType::Attribute;
        output.Attribute = in.A.Attribute;
        if (kind == KindOf<SelectionAttribute>) {
            const auto selection = ResolveSelection(r, plan, values, n);
            const auto &field = selection.Field;
            if (field && !field->Constant) continue;
            const auto domain = AttributeDomainOf(selection.Domain);
            const auto name = TemporaryName(n, InputCount[kind]);
            const auto attribute = (field ? meshes.AddAttribute(id, domain, AttributeType::Bool, name, {field->Constant->Words[0] & 1u}) : AddSelectionAttribute(meshes, id, domain, name, UploadSelection(r, arena, id, selection.Literal))).value();
            evaluation.Temporaries.push_back(attribute);
            output.Attribute = attribute;
            continue;
        }
        if (kind == KindOf<NamedAttribute>) {
            const auto &p = r.get<const NamedAttribute>(entity);
            const auto found = meshes.FindAttribute(id, NameOf(p.Name));
            output.Attribute = found != InvalidOffset && meshes.GetAttribute(found).Type == p.Type ? found : InvalidOffset;
            if (output.Attribute == InvalidOffset) r.emplace_or_replace<NodeError>(entity, NodeFault::Attribute);
            continue;
        }
        std::optional<AttributeJob> job;
        std::optional<uint32_t> computed;
        VisitKind(kind, [&]<typename K> {
            if constexpr (Prepared<K>) job = Prepare(meshes, Parameters<K>(r, plan, values, n), in);
            else if constexpr (Computed<K>) {
                flush();
                computed = Compute(r, id, Parameters<K>(r, plan, values, n), TemporaryName(n, InputCount[kind]), in);
            }
        });
        if (computed) {
            evaluation.Temporaries.push_back(*computed);
            output.Attribute = *computed;
            continue;
        }
        if (!job) {
            r.emplace_or_replace<NodeError>(entity, NodeFault::Attribute);
            continue;
        }
        const auto added = meshes.AddAttribute(id, job->Domain, job->Type, TemporaryName(n, InputCount[kind])).value();
        evaluation.Temporaries.push_back(added);
        output.Attribute = added;
        auto &pc = job->Pc;
        pc.Blocks = BlocksOf(meshes, id, ElementDomainOf(job->Domain));
        pc.Output = meshes.AttributeRef(added);
        pc.OutputType = job->Type;
        if (pc.Blocks.Count) jobs.push_back(*job);
    }
    std::vector<AttributeMaskPushConstants> scatters;
    evaluation.Selection = ResolveSelection(r, plan, values, node);
    const auto &field = evaluation.Selection.Field;
    if (!field) {
        evaluation.In = UploadSelection(r, arena, id, evaluation.Selection.Literal);
    } else if (field->Constant) {
        if (!(field->Constant->Words[0] & 1u)) evaluation.In = ZeroedMasks(meshes, arena, id, [](uint32_t, std::span<SelectionBlock>) {});
    } else if (!SeedAttribute(meshes, *field)) {
        evaluation.Failed = true;
    } else {
        const auto ref = meshes.AttributeRef(field->Attribute);
        const auto mask_ranges = AllocateMasks(meshes, arena, id);
        for (uint32_t d = 0u; d < 3u; ++d) {
            const auto domain = SelectionDomains[d];
            const bool scattered = SelectionElements[d] == evaluation.Selection.Domain;
            const auto [membership, owner] = WithDomain(meshes.Arenas(), domain, [&](const auto &elements) { return std::pair{elements.Blocks.Buffer.Slot, DomainSet(meshes.Get(id), domain).Index}; });
            scatters.push_back({.Attribute = scattered ? ref : NamedAttributeRef{}, .MembershipSlot = membership, .Owner = owner, .Masks = MaskBinding({&*arena, mask_ranges[d].Offset}), .Words = mask_ranges[d].Count * MeshElementBlockWords});
        }
        evaluation.In = MasksAt(*arena, {mask_ranges[0].Offset, mask_ranges[1].Offset, mask_ranges[2].Offset});
    }
    if (jobs.empty() && scatters.empty()) return evaluation;
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto &pipelines = GetMeshPipelines(r);
    for (const auto &job : jobs) chain.Threads(pipelines[job.Pass], job.Pc, job.Pc.Blocks.Count * MeshElementBlockSize);
    chain.Concurrent([&] {
        for (const auto &pc : scatters) chain.Threads(pipelines[MeshPass::AttributeSelectionMasks], pc, pc.Words);
    });
    if (!scatters.empty()) meshes.DeriveSelectionMasks(r, chain, id, *evaluation.In, evaluation.Selection.Domain);
    chain.Submit();
    return evaluation;
}

enum class RunMode { Publish,
                     Transient,
                     Handle };

Outcome RunNode(state::Registry &r, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, std::optional<BufferArena<SelectionBlock>> &arena, RunMode mode, state::Entity drawn = state::Null, bool *repaired = nullptr, MeshStore::Change *refresh = nullptr, TopologyReuse reuse_topology = TopologyReuse::None) {
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
    const Masks masks{.In = evaluation.In, .Domain = evaluation.Selection.Domain, .Publish = r.all_of<PublishSelection>(entity), .Drawn = drawn, .Repaired = repaired, .Refresh = refresh};
    if (mode != RunMode::Transient && std::ranges::any_of(HandleSpecs, [&](const auto &spec) { return spec.Kind == kind; })) {
        if (reuse_topology == TopologyReuse::None || !r.all_of<NodeHandle>(entity)) {
            const auto selected = Handles(meshes, id, masks);
            const auto &pipelines = GetMeshPipelines(r);
            NodeHandle support;
            for (const auto &position : VertexPositions(meshes, pipelines, id, selected.Vertices)) support.Center += position;
            if (!selected.Vertices.empty()) support.Center /= float(selected.Vertices.size());
            support.Normal = NormalizedSum(FaceNormals(meshes, pipelines, id, selected.Faces));
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
        auto p = Parameters<K>(r, plan, values, node);
        if constexpr (std::same_as<K, Extrude>) {
            const auto &input = plan.Inputs[plan.InputFirst[node] + ParameterSocket<&Extrude::Offset>];
            if (input.FromNode != NoNode || input.Constant) p.AlongNormals = false;
            edited = Stepped(p) ? RunExtrudeSteps(r, p, id, masks, linked) : RunExtrude(r, entity, p, id, masks, evaluation.Selection, linked, evaluation.Selection.Field ? TopologyReuse::None : reuse_topology, evaluation.Temporaries, mode == RunMode::Publish);
        } else if constexpr (std::same_as<K, StoreNamedAttribute>) {
            edited = Store(meshes, p, id, OperandOf(plan, values, node, 1u), evaluation.Temporaries);
        } else if constexpr (std::same_as<K, StoreSelection>) {
            edited = Store(meshes, p, id, masks);
        } else if constexpr (std::same_as<K, LevelSet>) {
            edited = Run(r, p, id, OperandOf(plan, values, node, 0u).Attribute);
            if (edited && *edited != id) ReleaseTemporaries(meshes, *edited);
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

uint32_t Generate(state::Registry &r, const Generator auto &shape) {
    return CreateShadedMesh(r, {.Data = primitive::CreateMesh(shape), .FlatShaded = true});
}

// Return the entity whose mesh record the referenced object draws, excluding the current procedure mesh.
state::Entity ReadMesh(const state::Registry &r, state::Entity mesh_entity, const ObjectInfo &p) {
    const auto *instance = r.valid(p.Object) ? r.try_get<const Instance>(p.Object) : nullptr;
    return instance && instance->Entity != mesh_entity && DrawsMesh(r, instance->Entity) ? instance->Entity : state::Null;
}

// The transform from the operand object's local space into Space's, composed by the settle pass rule, which takes each physics body's published pose.
// Both objects compose relative to their nearest common ancestor when no physics body lies below it and its world transform has positive uniform scale.
// Such a transform commutes with the rule, so moving that ancestor leaves the result unchanged bit for bit.
// A null operand is already in Space's space, and a dead object places by the identity.
mat4 ToFirst(const state::Registry &r, state::Entity space, state::Entity operand) {
    if (operand == state::Null) return I4;
    const auto owned = r.view<const BodyPoseCache>() | std::ranges::to<std::vector>();
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

std::vector<Transform> InstanceTransforms(state::Registry &r, uint32_t points) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const Mesh mesh{meshes, points};
    const auto rotations = meshes.AttributeView<vec4>(meshes.FindAttribute(points, InstanceRotation));
    const auto scales = meshes.AttributeView<vec3>(meshes.FindAttribute(points, InstanceScale));
    const auto vertices = HandleList(mesh.vertices());
    const auto positions = VertexPositions(meshes, GetMeshPipelines(r), points, vertices);
    std::vector<Transform> transforms;
    transforms.reserve(vertices.size());
    for (uint32_t i = 0u; i < vertices.size(); ++i) transforms.push_back({positions[i], quat{rotations[vertices[i]]}, scales[vertices[i]]});
    return transforms;
}

std::optional<uint32_t> Realize(state::Registry &r, uint32_t points, uint32_t instance) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto transforms = InstanceTransforms(r, points);
    if (transforms.empty()) return {};
    const auto clones = CloneRecords(r, std::vector(transforms.size(), instance));
    for (size_t i = 0u; i < clones.size(); ++i) {
        const PositionOperationTarget target{.StoreId = clones[i], .Selection = AllHandles(Mesh{meshes, clones[i]})};
        ExecuteGeometryPositions(r, target, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = transforms[i]}});
    }
    for (size_t i = 1u; i < clones.size(); ++i) meshes.Join(r, clones.front(), clones[i]);
    meshes.Release(std::span{clones}.subspan(1u));
    return clones.front();
}

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

void ReleaseReplaced(state::Registry &r, uint32_t record) {
    if (record == InvalidOffset) return;
    FinishPendingSourceStates(r, record);
    r.Resources.get<MeshStore>().Release(record);
}

// A new record holding a Source node's geometry, without selection, or none when no kept or recorded state holds it.
std::optional<uint32_t> MaterializeSource(state::Registry &r, const SourceState &source) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (source.StoreId == InvalidOffset) return CreateShadedMesh(r, {});
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

std::optional<uint32_t> MaterializeRoot(state::Registry &r, NodeRecord root) {
    if (root.Kind == KindOf<Source>) return MaterializeSource(r, r.get<const SourceState>(root.Entity));
    std::optional<uint32_t> id;
    VisitKind(root.Kind, [&]<typename K> {
        if constexpr (Generator<K>) id = Generate(r, r.get<const K>(root.Entity));
    });
    return id;
}

void CarryMaterials(MeshStore &meshes, uint32_t from, uint32_t to) {
    if (from == InvalidOffset || from == to || !meshes.TryGet(from)) return;
    const auto stored = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(from).PrimitiveMaterials);
    const std::vector<uint32_t> materials{stored.begin(), stored.end()};
    const auto replaced = meshes.EditPrimitiveMaterials(to);
    std::copy_n(materials.begin(), std::min(materials.size(), replaced.size()), replaced.begin());
}

// The root is a Source node or a generator with unlinked parameters, and no node after it instances.
// Only the main link from the node before it and field or value links reach a path node after the root, and such a node feeds only the next path node.
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
// That node and each node that runs keep the mesh storage state their result leaves as their NodeResult, and the path's last node keeps a pending one.
// The nodes from the first faulted one hold none, so the next cook runs them again.
// The positions of `restored`, a restore's change to the record, refresh with the first drawn position edit or after the path.
PathRun RunPath(state::Registry &r, state::Entity mesh_entity, const CookPlan &plan, std::span<const uint32_t> path, size_t first, uint32_t record, bool rendered, std::optional<BufferArena<SelectionBlock>> &arena, MeshStore::Change &restored, TopologyReuse reuse_topology) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (first < path.size()) FinishPendingSourceStates(r, record);
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
        // A chained Join, Boolean or Minkowski has no second input and passes its first through.
        if (kind != KindOf<Join> && kind != KindOf<Boolean> && kind != KindOf<Minkowski>) {
            bool repaired = false;
            if (!rendered) refresh_restored();
            const auto edited = RunNode(r, plan, values, node, record, arena, RunMode::Publish, rendered ? mesh_entity : state::Null, &repaired, &restored, i == first ? reuse_topology : TopologyReuse::None);
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

// The held nodes are the root and each following node whose result was kept on the result of the node before it.
// The record starts as it stands when it holds the last held node's output, else restored to that node's result state, else as a new record of the root.
// The whole mesh storage restores while only this mesh's cooks wrote it since the state was kept, and the record alone restores otherwise.
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
    size_t held = 0u;
    while (held < path.size() && !invalid[path[held]]) {
        if (held) {
            const auto *result = r.try_get<const NodeResult>(entity_at(held));
            if (!result || result->Input != entity_at(held - 1u)) break;
        }
        ++held;
    }
    auto record = prior;
    auto replaced = prior;
    bool rendered = prior != InvalidOffset && meshes.Get(prior).RenderTopologies != 0u;
    MeshStore::Change restored{};
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
            // A plain node's selection is its literal, which reads no socket values.
            if (plain && cached && cached->Input == entity_at(held - 1u) && cached->StoreId == prior && topology &&
                topology->Mode == r.get<const Extrude>(entity).Mode && !Stepped(r.get<const Extrude>(entity)) && topology->Selection == SelectionKey(ResolveSelection(r, plan, {}, node))) topology_state = topology->State;
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
    if (replaced != InvalidOffset && replaced != record && meshes.TryGet(replaced)) ReleaseReplaced(r, replaced);
    OwnMesh(r, mesh_entity, InvalidOffset);
    PublishInstances(r, mesh_entity, state::Null, {});
    PublishCooked(r, mesh_entity, run.Record, run.Faulted ? state::Null : entity_at(path.size() - 1u), run.Rendered, run.Record != record || TopologyRevisions(meshes, run.Record) != topology);
}

// A chained display path edits the mesh's record in place.
// Other procedures evaluate into a new record, empty when no node is displayed or the displayed node has no result.
uint32_t Evaluate(state::Registry &r, state::Entity mesh_entity, const CookPlan &plan, std::optional<BufferArena<SelectionBlock>> &arena, const state::DirtySet &changed, const state::DirtySet &created_nodes, RunMode mode = RunMode::Publish) {
    auto &meshes = r.Resources.get<MeshStore>();
    const bool publish = mode == RunMode::Publish;
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
    // Only the nodes of a chained display path hold results, and only the Boolean nodes a cook evaluates hold placements.
    // A cook that skips a node would leave its result or placement stale.
    for (uint32_t node = 0u; node < plan.Nodes.size(); ++node) {
        const auto entity = plan.Nodes[node].Entity;
        if (publish && (!chained || !on_path[node])) r.remove<NodeResult, NodeExtrusion, NodeHandle>(entity);
        if (chained) r.remove<BooleanPlacement>(entity);
    }
    if (chained) {
        CookChain(r, mesh_entity, plan, path, arena, changed, created_nodes);
        return DrawnStoreId(r, mesh_entity).value_or(InvalidOffset);
    }
    const auto prior = r.all_of<EvaluatedMesh>(mesh_entity) ? r.get<const EvaluatedMesh>(mesh_entity).StoreId : InvalidOffset;
    const auto &reached = plan.Demanded;
    auto remaining = plan.Consumers;
    // The display reads the displayed node's result after every node, so the result's other consumers clone it.
    const auto result_socket = display != NoNode ? ResultSocket(plan.Nodes[display].Kind) : std::optional<uint16_t>{};
    if (result_socket) ++remaining[plan.SocketFirst[display] + *result_socket];
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
    const auto consume = [&](const SocketValue &value) {
        if (value.Borrowed()) return;
        meshes.Release(value.Record);
        disown(value.Record);
    };
    uint32_t result = InvalidOffset;
    std::optional<SocketValue> published;
    const auto node_mode = publish ? RunMode::Publish : RunMode::Transient;
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
        if (kind == KindOf<LevelSet>) {
            const auto p = Parameters<LevelSet>(r, plan, values, node);
            const auto lattice = LatticeOver(p.Min, p.Max, p.EdgeLength);
            if (!lattice || lattice->Count() > (1u << 22u)) {
                fail(NodeFault::Parameter);
                continue;
            }
            const auto grid = created.emplace_back(CreateShadedMesh(r, {.Data = MeshData{lattice->Points()}}));
            if (const auto marched = RunNode(r, plan, values, node, grid, arena, node_mode)) values[first + InputCount[kind]].Record = created.emplace_back(*marched);
            else fail(marched.error());
            continue;
        }
        if (generated) continue;
        const auto in = take(node, 0u);
        const auto input = in.Record;
        const auto emit = [&](std::optional<uint32_t> record, bool linked, NodeFault fault) {
            auto &output = values[first + InputCount[kind]];
            if (record) {
                created.push_back(*record);
                output.Record = *record;
            } else {
                if (linked) fail(fault);
                output = in;
            }
        };
        if (mode == RunMode::Handle && node == display) {
            if (input != InvalidOffset) RunNode(r, plan, values, node, input, arena, RunMode::Handle);
            break;
        }
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
            // The map into Mesh 1's space must keep Mesh 2 unmirrored.
            const auto to_first = ToFirst(r, p.Space, second.Object);
            const BooleanPlacement placement{second.Object, to_first};
            if (const auto *held = r.try_get<const BooleanPlacement>(entity); publish && (!held || *held != placement)) r.emplace_or_replace<BooleanPlacement>(entity, placement);
            const auto rotation = ToMat3(to_first);
            const bool linked = input != InvalidOffset && second.Record != InvalidOffset, oriented = Determinant(rotation) > 0.f;
            std::optional<uint32_t> combined;
            auto fault = oriented ? NodeFault::Empty : NodeFault::Parameter;
            if (linked && oriented) {
                const Mesh first_mesh{meshes, input}, second_mesh{meshes, second.Record};
                if (const auto task = BooleanTask(first_mesh, second_mesh, p.Operation, rotation, vec3(to_first[3]))) combined = ExecuteGeometryBoolean(r, *task);
                else if (first_mesh.FaceCount() && second_mesh.FaceCount()) {
                    const auto plane = *FacePlane(r, first_mesh);
                    const auto fill = p.Operation == BooleanOperation::Union ? RegionFill::Union : p.Operation == BooleanOperation::Difference ? RegionFill::Difference :
                                                                                                                                                 RegionFill::Intersection;
                    const auto loops = RegionLoops(r, plane, std::array{RegionOperand{.Record = input}, RegionOperand{.Record = second.Record, .Rotation = rotation, .Translation = vec3(to_first[3])}}, fill);
                    if (!loops) fault = NodeFault::NotCoplanar;
                    else if (!loops->empty()) combined = CreateShadedMesh(r, {.Data = RegionData(plane, *loops)});
                } else fault = NodeFault::NotClosed;
            }
            emit(combined, linked, fault);
            continue;
        }
        if (kind == KindOf<Minkowski>) {
            const auto second = take(node, 1u);
            const bool linked = input != InvalidOffset && second.Record != InvalidOffset;
            const bool closed = linked && Mesh{meshes, input}.FaceCount() && Mesh{meshes, second.Record}.FaceCount();
            const auto swept = closed ? MinkowskiRecord(r, Mesh{meshes, input}, Mesh{meshes, second.Record}, Parameters<Minkowski>(r, plan, values, node).Operation) : std::nullopt;
            emit(swept, linked, closed ? NodeFault::Empty : NodeFault::NotClosed);
            continue;
        }
        if (kind == KindOf<InstanceOnPoints>) {
            const auto instance = take(node, 1u);
            auto &output = values[first + InputCount[kind]];
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
            if (const auto placed = RunNode(r, plan, values, node, input, arena, node_mode)) {
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
        const auto edited = input != InvalidOffset ? RunNode(r, plan, values, node, input, arena, node_mode) : Outcome{input};
        if (!edited) fail(edited.error());
        else if (*edited != input) created.push_back(*edited);
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
        transforms = InstanceTransforms(r, result);
        DeleteEvery(r, result, DeleteMode::Vertices);
    }
    // Instances of this mesh's pending source place a clone, since this mesh draws the points.
    if (!transforms.empty() && published->MeshEntity == mesh_entity) {
        published->MeshEntity = state::Null;
        published->Instance = CloneRecord(r, published->Instance);
    }
    const bool owns = !transforms.empty() && published->Prototype == state::Null && published->MeshEntity == state::Null;
    if (owns) disown(published->Instance);
    if (result == InvalidOffset) result = CreateShadedMesh(r, {});
    disown(result);
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

void QueueNodeRemovals(state::Registry &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    auto &dependents = r.Resources.emplace<PendingObjectRemovals>().Dependents;
    for (auto node = r.get<const GeometryProcedure>(mesh_entity).First; node != state::Null; node = r.get<const ProcedureNode>(node).Next) dependents.push_back(node);
}
void QueueInstanceRemovals(state::Registry &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    r.Resources.emplace<PendingObjectRemovals>().Objects.append_range(r.get<const ProcedureInstances>(mesh_entity).Entities);
}
void QueueOwnedMeshRemoval(state::Registry &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    r.Resources.emplace<PendingObjectRemovals>().Dependents.push_back(r.get<const ProcedureOwnedMesh>(mesh_entity).MeshEntity);
}
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

void SetMeshSelection(state::Registry &r, state::Entity mesh_entity, const InlineSelection *selection) {
    std::optional<BufferArena<SelectionBlock>> arena;
    auto &meshes = r.Resources.get<MeshStore>();
    const auto *live = r.try_get<const EvaluatedMesh>(mesh_entity);
    if (!live || !meshes.TryGet(live->StoreId)) return;
    if (!selection || selection->All) {
        ApplyEditSelectionCommand(r, std::array{ElementRange{mesh_entity, 0u}}, selection ? selection->Domain : Element::Vertex, EditSelectionOperation::Fill);
    } else {
        const auto masks = *UploadSelection(r, arena, live->StoreId, selection);
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.SelectMasks(r, chain, live->StoreId, masks);
        chain.Submit();
    }
    RefreshElementSelectionSummaries(r, std::array{mesh_entity}, selection ? selection->Domain : r.get<const EditMode>(state::Document).Value);
    const auto *active = r.try_get<const MeshActiveElement>(mesh_entity);
    const auto handle = active && selection && selection->Domain != Element::None && meshes.GetSelectedElements(live->StoreId, selection->Domain).Contains(active->Handle) ? active->Handle : InvalidOffset;
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
    inputs.on<Instance>(On::Create).on<SceneParent>(On::Destroy);
    changes.Moved.bind(r, state::Clearing::Explicit).on<Transform>(On::Update).on<PosedLocal>(On::Create | On::Update | On::Destroy).on<SceneParent>(On::Create | On::Destroy);
    ForEachKind([&]<typename K> { inputs.on<K>(On::Update); });
}

void SettleProcedures(state::Registry &r, bool cook) {
    using namespace procedure;
    auto &changes = r.Resources.get<ProcedureChanges>();
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
    // An edit that cooked in its own pass then does not cook again for the world transform it published.
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
