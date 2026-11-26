#include "procedure/Cook.h"
#include "procedure/OccurrenceImages.h"
#include "ProcessEvents.h"
#include "animation/Keying.h"
#include "procedure/OpNodes.h"
#include "scene/HierarchyInstance.h"
#include "timeline/Timeline.h"
#include "viewport/InteractionComponents.h"

#include "SortUnique.h"
#include "TransformMath.h"
#include "Variant.h"
#include "action/Core.h"
#include "gpu/ElementRepairJob.h"
#include "gpu/InstanceJob.h"
#include "mesh/AttributeFields.h"
#include "mesh/BeautifyFaces.h"
#include "mesh/ElementWorkSort.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/LevelSet.h"
#include "mesh/MeshChanges.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshTopologyEdit.h"
#include "mesh/Neighborhood.h"
#include "mesh/NormalDeriveGpu.h"
#include "mesh/Primitives.h"
#include "mesh/RecalculateNormals.h"
#include "mesh/Region.h"
#include "mesh/ScratchPasses.h"
#include "mesh/SharpnessOperations.h"
#include "mesh/Simplify.h"
#include "mesh/Unsubdivide.h"
#include "object/ObjectOps.h"
#include "object/PendingSync.h"
#include "ops/Evaluate.h"
#include "physics/PhysicsTypes.h"
#include "render/ElementWorkOps.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/MeshTopologyRepair.h"
#include "render/MeshletBuildGpu.h"
#include "render/Pipelines.h"
#include "render/RenderChanges.h"
#include "render/SceneUpdates.h"
#include "render/Textures.h"
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
    bool DependsOnEarlierCooks{}; // A cook took a path whose result depends on the cooks before it
};
void Reset(CookPass &pass, state::Registry &) { pass = {}; }
// The timeline frame and frame rate the procedures' cooks hold.
struct CookTime {
    int Frame{};
    float Fps{};
};
void Reset(CookTime &time, state::Registry &) { time = {}; }
// The Source nodes that may hold a pending state, by the record it refers to.
// Changed lists the Source nodes whose state changed since the last finish, which the next finish indexes, so a finish costs what its pending states do.
struct PendingSources {
    state::DirtySet Changed;
    std::map<uint32_t, std::vector<state::Entity>> ByRecord;
};
void Reset(PendingSources &pending, state::Registry &) { pending.ByRecord.clear(); }
void Reset(OccurrenceImages &images, state::Registry &) { images.Images.clear(); }
OccurrenceImages &CurrentImages(state::Registry &r) {
    auto &images = r.Resources.get<OccurrenceImages>();
    if (images.Epoch != r.Epoch) {
        Reset(images, r);
        images.Epoch = r.Epoch;
    }
    return images;
}

using namespace action::mesh;
namespace {
// Keep mesh storage as it stands as the state of each Source node whose pending state refers to `record`, or of every pending state for InvalidOffset.
void FinishPendingSourceStates(state::Registry &r, uint32_t record) {
    auto &pending = r.Resources.get<PendingSources>();
    const auto pending_record = [&](state::Entity entity) {
        const auto *source = r.valid(entity) ? r.try_get<const SourceState>(entity) : nullptr;
        return source && source->State == store::Hash128{} ? source->StoreId : InvalidOffset;
    };
    for (const auto entity : pending.Changed) {
        if (const auto id = pending_record(entity); id != InvalidOffset) pending.ByRecord[id].push_back(entity);
    }
    std::optional<store::Hash128> state;
    const auto finish = [&](uint32_t id, std::span<const state::Entity> entities) {
        for (const auto entity : entities) {
            // An entry whose source changed record or finished since it was indexed is stale.
            if (pending_record(entity) != id) continue;
            if (!state) state = r.Resources.get<MeshStore>().KeepState();
            r.replace<SourceState>(entity, *state, id);
        }
    };
    if (record == InvalidOffset) {
        for (const auto &[id, entities] : std::exchange(pending.ByRecord, {})) finish(id, entities);
    } else if (const auto found = pending.ByRecord.find(record); found != pending.ByRecord.end()) {
        const auto entities = std::move(found->second);
        pending.ByRecord.erase(found);
        finish(record, entities);
    }
    // The finished sources' own changes leave them no longer pending.
    pending.Changed.clear();
}

struct DrawnEdits;
// A drawn record the round's edits moved, with its edits, and whether they moved its vertices since its normals were derived.
// The moves of a record with stale render records refresh on their own, over Unrefreshed's vertices.
struct DrawnRecord {
    uint32_t Record;
    DrawnEdits *Edits;
    bool NormalsStale{};
    std::optional<PositionOperationChange> Unrefreshed{};
};
struct CookRound;
// Return the output record, the input for an unchanged operation, or the reason the operation failed.
using Outcome = std::expected<uint32_t, NodeFault>;
// An in-place topology task of a path step, with the edits of the record it edits when a mesh entity draws that record.
// Extruded receives the vertices an extrusion emits, or is null.
// Kept edits repair the render records of a drawn record, which the state kept after them holds.
// Created receives the outcome of a task that creates a record, or is null.
struct StagedTask {
    MeshTopologyTask Task;
    DrawnEdits *Drawn{};
    std::vector<ExtrusionSource> *Extruded{};
    bool Kept{};
    std::optional<Outcome> *Created{};
};
// The stages of one round of a step's in-place topology edits, and the plans of the next round's stages, which read this round's results.
// Keeps asks for the mesh storage state this round's stages leave, which the next round's Kept holds.
struct StageRound {
    std::vector<std::vector<StagedTask>> Stages;
    std::vector<std::function<void(StageRound &)>> Later;
    bool Keeps{};
    store::Hash128 Kept{};
};
// A node of a step, with the outcome its run returned, the outcome a plan after the round or after reads set, and what takes the outcome once the step's work completed.
struct StepNode {
    Outcome Ran{};
    std::optional<Outcome> Planned{};
    std::function<void(const Outcome &)> Take{};
};
// The work of one step of the cooks: the chain their plans read on, the first round of their topology stages, and their nodes.
// Reads is the step's chain, whose lanes record the passes the plans read and whose submits run the plans that wait for the round.
// Waits says a plan waits for the round or for reads.
// Temporaries holds the temporary attributes the step's passes and plans read, which the round releases once it submits after the step.
struct Step {
    mtl::BufferContext &Buffers;
    // The step's chain, which exists from the step's first read or plan, so a step with none allocates nothing.
    std::optional<mtl::ComputeChain> Reads{};
    bool Waits{};
    std::vector<uint32_t> Temporaries{};
    // The changed triangles of the drawn records whose edits on the step's chain repair their render records, all in one repair once their edits complete.
    std::vector<std::pair<state::Entity, FaceTriangles>> Repairs{};
    StageRound First{};
    std::deque<StepNode> Nodes{};
};

mtl::ComputeChain &ReadsOf(Step &step) {
    if (!step.Reads) step.Reads.emplace(step.Buffers);
    return *step.Reads;
}

// A node's selection masks, where absent masks select every element, and the step its run records into.
struct Masks {
    std::optional<GeometryMasks> In;
    Element Domain{Element::Vertex};
    // Copies the emitted selection into the output record.
    bool Publish{};
    // The edits of the input record a mesh entity draws, which a node edits in place, or null for a record without current render records.
    DrawnEdits *Drawn{};
    // The round the node's passes record on, each in its record's lane.
    CookRound &Round;
    Step &Step;
    // The round of in-place topology edits the node stages its tasks in, each stage executing with the same stage of the other meshes of the step.
    StageRound &Topology;
    // Receives the node's outcome when a plan completes after the round or its reads, and is null for a later round.
    std::optional<Outcome> *Planned{};
};
// The masks with the stages of a later round, whose plans set no outcome.
Masks Later(const Masks &m, StageRound &later) { return {.In = m.In, .Domain = m.Domain, .Publish = m.Publish, .Drawn = m.Drawn, .Round = m.Round, .Step = m.Step, .Topology = later}; }

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
    DataType Type{DataType::Geometry}; // A record's type, Geometry or Instances, an image's, or the type of an op output that does not vary
    // The cook owns records when MeshEntity is null, and otherwise borrows them read-only from MeshEntity, which keeps owning them.
    state::Entity MeshEntity{state::Null};
    state::Entity Prototype{state::Null};
    state::Entity Object{state::Null}; // For geometry, the object whose local space the geometry is in, or null for this procedure's mesh space
    uint32_t Instance{InvalidOffset}; // The record an instances value's points each place, or InvalidOffset for points placing nothing
    union {
        uint32_t Record{InvalidOffset};
        uint32_t Attribute;
        ImageValue Image;
        AttributeValue Value;
    };
    bool Borrowed() const { return Type == DataType::Geometry && MeshEntity != state::Null; }
};
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

// The finished edits of a record a mesh entity draws, with the elements whose point and line owners they changed.
struct DrawnRepair {
    state::Entity Entity;
    uint32_t StoreId;
    EditedBlocks Edited;
    std::vector<ElementMarks> Marked;
};
// Refresh pose membership, the mesh binding, the marked elements' point and line owners and the selection summaries once the drawn records' edits finish.
void RepairDrawnElements(state::Registry &r, mtl::ComputeChain &chain, std::span<DrawnRepair> repairs) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto &a = meshes.Arenas();
    const auto &pipelines = GetMeshPipelines(r);
    // Each marked repair's bit runs of the vertices and edges that start and stop drawing, all zeroed in one fill, and their word counts.
    std::vector<ElementRepairJob> marks(repairs.size());
    std::vector<std::array<uint32_t, 2>> counts(repairs.size());
    uint32_t words = 0u;
    for (size_t i = 0u; i < repairs.size(); ++i) {
        auto &repair = repairs[i];
        UpdatePoseMembership(r, repair.StoreId, std::move(repair.Edited));
        RefreshMeshBinding(r, repair.StoreId);
        if (repair.Marked.empty()) continue;
        const auto &owner = meshes.Get(repair.StoreId);
        auto &mark = marks[i];
        mark = {
            .Connectivity = meshes.GetConnectivityRef(repair.StoreId),
            .MembershipSlot = {a.Vertices.Blocks.Buffer.Slot, a.EdgeHalfedges.Blocks.Buffer.Slot},
            .Bound = {a.Vertices.HandleBound(owner.Vertices), a.EdgeHalfedges.HandleBound(owner.EdgeData)},
            .Owners = {meshes.Render().ElementMeshlets[2].Ref(owner.ElementMeshlets[2]), meshes.Render().ElementMeshlets[1].Ref(owner.ElementMeshlets[1])},
            .OwnerBound = {owner.ElementMeshlets[2].Count * MeshElementBlockSize, owner.ElementMeshlets[1].Count * MeshElementBlockSize},
        };
        for (uint32_t d = 0u; d < 2u; ++d) {
            counts[i][d] = (std::max(mark.Bound[d], mark.OwnerBound[d]) + 31u) / 32u;
            mark.Added[d] = words;
            mark.Removed[d] = words + std::max(counts[i][d], 1u);
            words += 2u * std::max(counts[i][d], 1u);
        }
    }
    if (words) {
        const auto first = chain.Scratch.Allocate(words).Offset;
        FillScratch(chain, pipelines, std::array{ScratchFillRun{first, words, 0u}});
        // Every marked work is one job, with one tile per occupied block.
        std::vector<ElementRepairJob> jobs;
        std::vector<uint32_t> tiles;
        for (size_t i = 0u; i < repairs.size(); ++i) {
            if (repairs[i].Marked.empty()) continue;
            auto &mark = marks[i];
            for (uint32_t d = 0u; d < 2u; ++d) {
                mark.Added[d] += first;
                mark.Removed[d] += first;
            }
            for (const auto [work, domain] : repairs[i].Marked) {
                const auto blocks = WorkBlockCount(chain.Scratch, work);
                if (!blocks) continue;
                mark.Work = work;
                mark.WorkDomain = domain;
                jobs.push_back(mark);
                tiles.push_back(blocks);
            }
        }
        mtl::GatherPass(chain, pipelines[MeshPass::ElementRepairMark], std::span{jobs}, tiles, MeshElementBlockSize);
    }
    r.Resources.get<GpuSceneState>().EditSelectionDirty = true;
    std::vector<ElementMeshletRepair> element_repairs;
    if (words) {
        chain.Submit();
        const auto handles = [&](uint32_t offset, uint32_t count) { return SetBitHandles(chain.Scratch.Get({offset, count})); };
        for (size_t i = 0u; i < repairs.size(); ++i) {
            if (repairs[i].Marked.empty()) continue;
            for (const uint32_t d : {0u, 1u}) element_repairs.push_back({.StoreId = repairs[i].StoreId, .Topology = 2u - d, .Added = handles(marks[i].Added[d], counts[i][d]), .Removed = handles(marks[i].Removed[d], counts[i][d])});
        }
    }
    RepairElementMeshlets(r, chain, element_repairs);
    chain.Submit();
    std::vector<state::Entity> entities;
    for (const auto &repair : repairs) {
        auto &repaired = meshes.WriteRecord(repair.StoreId);
        uint32_t mask = 0u;
        for (uint32_t t = 0u; t < 3u; ++t) {
            if (repaired.ElementMeshletBlockCounts[t]) mask |= 1u << t;
        }
        repaired.RenderTopologies = mask ? mask : 4u;
        entities.push_back(repair.Entity);
    }
    RefreshElementSelectionSummaries(r, entities);
}

// Successive in-place topology edits of the record a mesh entity draws, gathered for one repair of the record the last edit leaves.
struct AccumulatedTopologyEdits {
    AccumulatedTriangleEdits Triangles;
    EditedBlocks Blocks;
    // Bit words by handle of the vertices, edges and halfedges standing for their edges that the edits may start or stop drawing as points and lines.
    std::array<std::vector<uint32_t>, 3> Marked;
    // An edit changed the record.
    bool Edited{};
};

// The unsubmitted work of the cooks of several meshes, on one chain they share, each record's passes in its own lane.
// The restored records reconcile, the materialized records derive their selection state, and the records the edits moved refresh their tessellation, normals and bounds, all together when the chain submits.
// The chain exists from the round's first work, so a cook with none allocates nothing.
struct CookRound {
    explicit CookRound(mtl::BufferContext &buffers) : Buffers(buffers) {}
    mtl::BufferContext &Buffers;
    std::optional<mtl::ComputeChain> Pending;
    // The readers of the states the round cloned and restored records from, which hold until it submits.
    MeshStore::StateReaders Readers;
    // The changes of the restored records awaiting their reconciliation, and the meshes whose refresh takes their positions.
    std::vector<MeshStore::Change> Restored;
    std::vector<state::Entity> Deferred;
    // The materialized records awaiting their selection state.
    std::vector<uint32_t> Unselected;
    // The records the chain's edits moved, sorted, whose normals are stale and whose tessellation and bounds have not refreshed.
    std::vector<uint32_t> Moved;
    // The records edits planned from the mesh or from selection masks moved, which refresh around the vertices they moved.
    std::vector<std::pair<uint32_t, PositionOperationChange>> Partial;
    // The drawn records the chain's edits moved, whose positions refresh and whose render records repair.
    std::vector<DrawnRecord> Drawn;
    // The temporary attributes the chain's passes read, which release once it submits.
    std::vector<uint32_t> Temporaries;
    // The buffers the chain's image passes write texels to, each filled from its start, which never grow, so the copies into the images bind them directly.
    std::vector<mtl::Buffer> TexelPages;
    // The texel words the pages hold.
    uint64_t Texels{};
    // The scratch bytes the chain held at the last budget check since its submit.
    uint64_t Scratch{};
    // The records the chain clones, whose selection indexes rebuild once it submits.
    std::vector<uint32_t> Cloned;
    // Records the round created whose shading derives before its work.
    bool Unshaded{};
    // Clones of pending Source records, by clone and source, whose edit state resets from the source's once the chain submits.
    std::vector<std::pair<uint32_t, uint32_t>> Reset;
    // The changed triangles of the drawn records whose edits on the chain repair their render records, all in one repair once the chain submits.
    std::vector<std::pair<state::Entity, FaceTriangles>> Repairs;
};

// The in-place edits of the record a mesh entity draws since its render records were current.
struct DrawnEdits {
    state::Entity Entity{state::Null};
    AccumulatedTopologyEdits Topology;
    // Handle ranges of the vertices the edits moved.
    std::vector<Range> Moved;
    // The change a restore made to the record, whose positions refresh with the moved vertices.
    MeshStore::Change Restored;
    // The running node is the last before the render records repair.
    bool Last{};
    // The running node recorded its edits.
    bool Recorded{};
};
// A node's topology or shading edit repairs the render records itself when it is the last before their repair and no earlier edit changed the record's topology or triangles.
bool RepairsOwnEdit(const DrawnEdits &drawn) { return drawn.Last && !drawn.Topology.Edited; }

bool RetiresAll(const MeshStore &meshes, uint32_t store_id, const AccumulatedTopologyEdits &accumulated) {
    return accumulated.Triangles.ClusterCount == meshes.Get(store_id).Level0Count;
}

// Adds triangles whose shading or tessellation changed to the gathered edits.
// Edits that retire every finest cluster leave every later change to the new build, since the triangles later edits emit hold no render owner.
void AccumulateChanged(state::Registry &r, DrawnEdits &drawn, uint32_t id, const BufferArena<uint32_t> &storage, ElementWork triangles) {
    if (!RetiresAll(r.Resources.get<const MeshStore>(), id, drawn.Topology)) AccumulateChangedTriangles(r, drawn.Topology.Triangles, id, storage, triangles);
    drawn.Topology.Edited = true;
}

// Adds a published edit of the record to the accumulated edits before it finishes.
void AccumulateEdit(state::Registry &r, AccumulatedTopologyEdits &accumulated, const MeshTopologyEdit &edit) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    accumulated.Edited |= bool(edit.Output);
    if (!edit.Output || RetiresAll(meshes, edit.StoreId, accumulated)) return;
    AccumulateTriangleEdits(r, accumulated.Triangles, edit);
    AppendEditedBlocks(accumulated.Blocks, meshes, edit);
    if (!edit.Repair) return;
    for (const auto [work, domain] : RepairMarks(edit)) {
        auto &words = accumulated.Marked[domain];
        ForEachWorkBlock(edit.Chain.Scratch, work, [&](uint32_t block, auto masks) {
            if (words.size() < (block + 1u) * MeshElementBlockWords) words.resize((block + 1u) * MeshElementBlockWords);
            for (uint32_t w = 0u; w < MeshElementBlockWords; ++w) words[block * MeshElementBlockWords + w] |= masks[w];
        });
    }
}

// The selection update of the blocks a restore changed, when the record has selection.
std::optional<MeshStore::SelectionUpdate> RestoredSelection(const MeshStore &meshes, uint32_t id, const MeshStore::Change &restored) {
    if (!meshes.Get(id).SelectionSummary.Count || std::ranges::all_of(std::span{restored.Blocks}.first(3u), [](const auto &blocks) { return blocks.empty(); })) return {};
    MeshStore::SelectionUpdate update{.StoreId = id};
    for (uint32_t d = 0u; d < 3u; ++d) {
        update.Blocks[d] = restored.Blocks[d];
        SortUnique(update.Blocks[d]);
    }
    return update;
}

// Records the refresh of the live vertices among each drawn record's moves and restore, and returns the render repair to run after the chain submits.
// `retessellate` covers the faces of the moves the chain recorded without their refresh.
std::function<void()> RecordDrawnRefresh(state::Registry &r, mtl::ComputeChain &chain, std::span<const DrawnRecord> drawn, bool retessellate) {
    auto &meshes = r.Resources.get<MeshStore>();
    std::vector<std::vector<Range>> ranges(drawn.size());
    std::vector<MeshVertexChanges> changes;
    std::vector<state::Entity> moved;
    std::vector<MeshStore::SelectionUpdate> restored;
    for (size_t i = 0u; i < drawn.size(); ++i) {
        const auto record = drawn[i].Record;
        auto *edits = drawn[i].Edits;
        auto &own = ranges[i];
        own = std::exchange(edits->Moved, {});
        const bool any_moved = !own.empty();
        own.append_range(edits->Restored.VertexRanges);
        const bool ranged = !own.empty();
        CoalesceRanges(own);
        own = meshes.LiveVertexRanges(record, own);
        // A restore's blocks refresh without ranges.
        if (!ranged || !own.empty()) changes.push_back({edits->Entity, own});
        if (any_moved && !own.empty()) moved.push_back(edits->Entity);
        if (auto update = RestoredSelection(meshes, record, edits->Restored)) restored.push_back(std::move(*update));
        edits->Restored = {};
    }
    auto repair = RefreshEditedPositions(r, chain, changes, retessellate);
    // The publication refits the records' meshlets before the repair of changed triangles replaces any of them.
    if (!moved.empty()) PublishEditedPositions(r, chain, moved);
    if (!restored.empty()) meshes.UpdateSelection(r, chain, restored);
    return repair;
}

// Whether the drawn edits hold moves or a restore whose positions have not refreshed.
bool RefreshPending(const DrawnEdits &drawn) {
    return !drawn.Moved.empty() || !drawn.Restored.VertexRanges.empty() || std::ranges::any_of(drawn.Restored.Blocks, [](const auto &blocks) { return !blocks.empty(); });
}

// Refreshes the moves and the restore of each drawn record where either is pending, on one chain.
// A restore that kept the live render records retessellates around its positions, and `current` says the records' render records are current, which then repair the changed triangles.
void RefreshDrawn(state::Registry &r, std::span<const DrawnRecord> drawn, bool current) {
    const auto pending = drawn | std::views::filter([](const DrawnRecord &record) { return RefreshPending(*record.Edits); }) | std::ranges::to<std::vector>();
    if (pending.empty()) return;
    mtl::ComputeChain chain{r.Resources.get<MeshStore>().BufferContext()};
    const bool retessellate = std::ranges::any_of(pending, [](const DrawnRecord &record) { return record.Edits->Restored.LiveRender; });
    if (auto repair = RecordDrawnRefresh(r, chain, pending, retessellate); repair && current) chain.AfterSubmit(std::move(repair));
    chain.Submit();
}

// Whether the edit reads base normals.
bool ReadsNormals(PositionEditOp op, const PositionOperationOptions &options) {
    const auto *program = std::get_if<PositionProgram>(&options.Input);
    return op == PositionEditOp::ShrinkFatten || (options.Randomize && options.Randomize->Normal > 0.f) || (program && (program->Call.Program.Needs & OpNeedsNormal));
}

mtl::ComputeChain &ChainOf(CookRound &round) {
    if (!round.Pending) round.Pending.emplace(round.Buffers);
    return *round.Pending;
}
// Leaves a restore's reconciliation to the round and returns the part of its change the mesh's refresh takes.
MeshStore::Change JoinRestore(state::Registry &r, CookRound &round, const MeshStore::Change &change, state::Entity mesh_entity) {
    round.Restored.push_back(change);
    round.Deferred.push_back(mesh_entity);
    return DeferredRestore(r, change, mesh_entity);
}

// Merges a later edit's moves into the moves of a record awaiting its refresh, where moves of every vertex cover any other.
void MergeMoves(PositionOperationChange &into, const PositionOperationChange &change) {
    if (into.Every) return;
    if (change.Every) {
        into = change;
        return;
    }
    into.Vertices.append_range(change.Vertices);
    SortUnique(into.Vertices);
    into.Ranges.append_range(change.Ranges);
}

// Shades the round's new records, reconciles its restored records, records the selection state of its materialized records and one refresh of every record its edits moved, as each record's own work would leave it, and submits the round.
// The moves of drawn records with stale render records refresh as undrawn ones do, and the triangles each refresh retessellated join their records' accumulated edits.
// Then rebuilds the selection indexes of its clones and resets the edit state of its clones of pending sources.
void SubmitRound(state::Registry &r, CookRound &round) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (std::exchange(round.Unshaded, false)) ShadeNewMeshes(r);
    round.Readers = {};
    if (!round.Restored.empty()) {
        SortUnique(round.Deferred);
        ReconcileRestoredRecords(r, ChainOf(round), round.Restored, round.Deferred);
        round.Restored.clear();
        round.Deferred.clear();
    }
    if (!round.Unselected.empty()) meshes.EnsureSelectionState(r, ChainOf(round), round.Unselected);
    std::vector<DrawnRecord> current, stale;
    for (auto &drawn : std::exchange(round.Drawn, {})) (drawn.Unrefreshed ? stale : current).push_back(std::move(drawn));
    // The stale records' edits and records, partial refreshes first, in the order of their refreshes' retessellated triangles.
    std::vector<std::pair<DrawnEdits *, uint32_t>> accumulated;
    std::vector<PositionRefresh> partial;
    std::vector<uint32_t> every;
    std::ranges::stable_partition(stale, [](const DrawnRecord &drawn) { return !drawn.Unrefreshed->Every; });
    for (const auto &drawn : stale) {
        if (drawn.Unrefreshed->Every) every.push_back(drawn.Record);
        else partial.push_back({drawn.Record, &*drawn.Unrefreshed});
        accumulated.emplace_back(drawn.Edits, drawn.Record);
    }
    const auto stale_partial = partial.size();
    for (const auto &[id, change] : round.Partial) partial.push_back({id, &change});
    std::vector<ElementWork> retessellated;
    if (!partial.empty()) retessellated = EncodePositionRefresh(r, ChainOf(round), partial);
    retessellated.resize(stale_partial);
    round.Partial.clear();
    if (!round.Moved.empty() || !every.empty()) {
        // The refresh plans from the moved records' faces on the host, which a clone holds once the chain has copied it.
        if (std::ranges::any_of(round.Moved, [&](uint32_t id) { return std::ranges::contains(round.Cloned, id); })) round.Pending->Submit();
        auto ids = std::exchange(round.Moved, {});
        ids.append_range(every);
        const auto refreshed = EncodeRecordRefresh(r, ChainOf(round), ids);
        retessellated.append_range(std::span{refreshed}.last(every.size()));
    }
    if (!accumulated.empty()) {
        ChainOf(round).AfterSubmit([&r, &chain = ChainOf(round), accumulated = std::move(accumulated), retessellated = std::move(retessellated)] {
            for (size_t i = 0u; i < accumulated.size(); ++i) {
                const auto [edits, record] = accumulated[i];
                if (retessellated[i].Storage.Slot != InvalidSlot && ElementWorkCount(chain.Scratch, retessellated[i])) AccumulateChanged(r, *edits, record, chain.Scratch, retessellated[i]);
            }
        });
    }
    if (!current.empty()) {
        if (auto repair = RecordDrawnRefresh(r, ChainOf(round), current, true)) ChainOf(round).AfterSubmit(std::move(repair));
    }
    if (round.Pending) {
        // The pages hold until the passes that read them complete.
        for (auto &page : std::exchange(round.TexelPages, {})) round.Pending->Retain(std::move(page));
        round.Pending->AfterSubmit([&r, &round] {
            if (auto repairs = std::exchange(round.Repairs, {}); !repairs.empty()) {
                RepairFaceRender(r, *round.Pending, repairs);
                round.Pending->Submit();
            }
        });
        round.Pending->Submit();
        round.Pending->ReleaseScratch();
    }
    round.Texels = round.Scratch = 0u;
    if (!round.Cloned.empty()) meshes.RebuildSelectionIndex(std::exchange(round.Cloned, {}));
    if (!round.Reset.empty()) {
        mtl::ComputeChain chain{meshes.BufferContext()};
        for (const auto [id, source] : round.Reset) meshes.ResetEditState(r, chain, id, source);
        chain.Submit();
        for (const auto [id, source] : std::exchange(round.Reset, {})) RefreshMeshBinding(r, id);
    }
    for (const auto attribute : std::exchange(round.Temporaries, {})) meshes.ReleaseAttribute(attribute);
    for (const auto id : std::exchange(round.Unselected, {})) RefreshMeshBinding(r, id);
}

// Submits the round before another mesh as wide as the last would pass the budget, so the chain's scratch stays within one workspace of the budget's size.
// The scratch counts what the pending lanes' gathered values allocate when they record.
void SubmitPastBudget(state::Registry &r, CookRound &round) {
    if (!round.Pending) return;
    const uint64_t used = round.Pending->Scratch.Buffer.UsedSize + round.Pending->GatheredWords() * sizeof(uint32_t), last = used - std::exchange(round.Scratch, used);
    if (used + last > uint64_t(ScratchWordBudget) * sizeof(uint32_t)) SubmitRound(r, round);
}

// Records the refresh of a record the round moved, in the record's lane, before passes that read its normals or triangulation.
// The round submits first for a record it cloned, whose faces the refresh plans from once the round copied them, and for a record it refreshes around listed vertices.
void RefreshMoved(state::Registry &r, CookRound &round, uint32_t id) {
    if (std::ranges::contains(round.Partial, id, &std::pair<uint32_t, PositionOperationChange>::first)) return SubmitRound(r, round);
    const auto moved = std::ranges::lower_bound(round.Moved, id);
    if (moved == round.Moved.end() || *moved != id) return;
    if (std::ranges::contains(round.Cloned, id)) return SubmitRound(r, round);
    round.Moved.erase(moved);
    auto &chain = ChainOf(round);
    chain.Lane(id, [&] { EncodeRecordRefresh(r, chain, std::span{&id, 1u}); });
}

// A clone of the record, which the round's chain copies in the record's lane, where the clone's later passes record too.
// A record the round moved copies after its refresh, so the clone holds its normals and triangulation.
uint32_t CloneOnRound(state::Registry &r, CookRound &round, uint32_t record) {
    RefreshMoved(r, round, record);
    auto &chain = ChainOf(round);
    uint32_t clone;
    chain.Lane(record, [&] { clone = RecordClones(r, chain, std::array{record}).front(); });
    chain.JoinLane(clone, record);
    round.Cloned.push_back(clone);
    return clone;
}

// Runs `then` once the round completed the passes recorded before the step's plans.
void AfterRound(const Masks &m, std::function<void()> then) {
    m.Step.Waits = true;
    ReadsOf(m.Step).AfterSubmit(std::move(then));
}
// Runs `plan` once the round completed the passes recorded before it, and returns the input record, which the outcome `plan` returns replaces as the node's planned outcome.
Outcome PlanAfterRound(const Masks &m, uint32_t id, std::function<Outcome()> plan) {
    AfterRound(m, [planned = m.Planned, plan = std::move(plan)] {
        const auto outcome = plan();
        if (planned) *planned = outcome;
    });
    return id;
}

// Records the edit on the round in its record's lane.
// An edit that reads normals the round's earlier edits left stale reads them after a derive for a drawn record, and after the refresh of an undrawn one.
// A drawn record with stale render records refreshes its moves on its own, and an undrawn record moved by an edit planned after the round refreshes around the moved vertices.
void RecordPositions(state::Registry &r, const Masks &masks, const PositionOperationTarget &target, PositionEditOp op, float factor, uint32_t repeat, const PositionOperationOptions &options, bool planned, const PositionPlan *plan = nullptr) {
    auto &meshes = r.Resources.get<MeshStore>();
    auto &round = masks.Round;
    const auto id = target.StoreId;
    auto *drawn = masks.Drawn;
    if (drawn) drawn->Recorded = true;
    auto listed = drawn ? std::ranges::find(round.Drawn, drawn, &DrawnRecord::Edits) : round.Drawn.end();
    if (ReadsNormals(op, options)) {
        if (listed != round.Drawn.end() && listed->NormalsStale) {
            ChainOf(round).Lane(id, [&] { EncodeDeriveRecordNormals(r, ChainOf(round), std::span{&id, 1u}); });
            listed->NormalsStale = false;
        } else if (!drawn) {
            RefreshMoved(r, round, id);
        }
    }
    auto &chain = ChainOf(round);
    std::optional<PositionOperationChange> edited;
    chain.Lane(id, [&] { edited = EncodePositionOperations(meshes, GetMeshPipelines(r), chain, target, op, factor, repeat, options, plan); });
    if (!edited) return;
    if (!drawn) {
        if (planned && !edited->Every) {
            round.Partial.emplace_back(id, std::move(*edited));
        } else if (const auto at = std::ranges::lower_bound(round.Moved, id); at == round.Moved.end() || *at != id) {
            round.Moved.insert(at, id);
        }
        return;
    }
    drawn->Moved.append_range(edited->Ranges);
    if (listed == round.Drawn.end()) listed = round.Drawn.insert(round.Drawn.end(), DrawnRecord{id, drawn});
    listed->NormalsStale = true;
    if (!drawn->Topology.Edited) return;
    if (listed->Unrefreshed) MergeMoves(*listed->Unrefreshed, *edited);
    else listed->Unrefreshed = std::move(*edited);
}

// An edit records at once when it reads no GPU state to plan, and an edit over its selection's masks or one that plans from the mesh records once the round completed the passes before it.
// An edit planned from the mesh records once its plan's reads on the step's chain completed.
void EditPositions(state::Registry &r, const Masks &masks, const PositionOperationTarget &target, PositionEditOp op, float factor, uint32_t repeat, const PositionOperationOptions &options) {
    const bool plans = PlansFromMesh(op, options);
    if (!std::holds_alternative<GeometryMasks>(target.Selection) && !plans) return RecordPositions(r, masks, target, op, factor, repeat, options, false);
    AfterRound(masks, [&r, masks, target, op, factor, repeat, options, plans] {
        if (!plans) return RecordPositions(r, masks, target, op, factor, repeat, options, true);
        RecordPositionPlan(r.Resources.get<const MeshStore>(), GetMeshPipelines(r), ReadsOf(masks.Step), target, op, options, [&r, masks, target, op, factor, repeat, options](std::shared_ptr<const PositionPlan> plan) { RecordPositions(r, masks, target, op, factor, repeat, options, true, plan.get()); });
    });
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
// Adds the triangles whose shading changed to `repairs` where the edit repairs its own render records, and to the drawn record's accumulated edits otherwise.
void TakeShaded(state::Registry &r, const mtl::ComputeChain &chain, std::vector<std::pair<state::Entity, FaceTriangles>> &repairs, DrawnEdits &drawn, uint32_t id, const FaceTriangles &triangles) {
    if (RepairsOwnEdit(drawn)) repairs.emplace_back(drawn.Entity, triangles);
    else if (triangles.Count) AccumulateChanged(r, drawn, id, chain.Scratch, triangles.Triangles);
}
// Executes one stage's tasks together, fills each extrusion's vertices, gathers the edits of each drawn record for one repair, repairing at once those whose state is kept and those that repair their own edit, and sets the outcome of each task that creates a record.
// The edits that repair their records' render records share one render repair and one element repair.
void ExecuteStage(state::Registry &r, std::vector<StagedTask> &stage) {
    const auto tasks = stage | std::views::transform([](StagedTask &staged) { return std::move(staged.Task); }) | std::ranges::to<std::vector>();
    const auto &meshes = r.Resources.get<const MeshStore>();
    // An extrusion's vertices are read before publication and take their handles after it.
    const auto extruded = [&](std::span<const MeshTopologyEdit> edits, uint32_t first) {
        for (uint32_t i = 0u; i < edits.size(); ++i) {
            if (auto *vertices = stage[first + i].Extruded) *vertices = edits[i].ExtrusionSources(r);
        }
    };
    const auto results = ExecuteGeometryTopology(r, tasks, [&](mtl::ComputeChain &chain, std::span<MeshTopologyEdit> edits, uint32_t first) {
        std::vector<std::pair<state::Entity, const MeshTopologyEdit *>> renders;
        std::vector<std::pair<DrawnEdits *, const MeshTopologyEdit *>> accumulated;
        std::vector<DrawnRepair> repairs;
        std::vector<const MeshTopologyEdit *> repaired;
        for (uint32_t i = 0u; i < edits.size(); ++i) {
            const auto &edit = edits[i];
            const auto &staged = stage[first + i];
            if (staged.Extruded && edit.Output) {
                const auto handles = chain.Scratch.Get(edit.Output->Vertices);
                for (auto &vertex : *staged.Extruded) vertex.Output = handles[vertex.Output];
            }
            auto *drawn = staged.Drawn;
            if (!drawn || !edit.Output) continue;
            if (!staged.Kept && !RepairsOwnEdit(*drawn)) {
                accumulated.emplace_back(drawn, &edit);
                continue;
            }
            if ((meshes.Get(edit.StoreId).RenderTopologies & 1u) != 0u || edit.AddedTriangleCount) renders.emplace_back(drawn->Entity, &edit);
            repairs.push_back({drawn->Entity, edit.StoreId});
            repaired.push_back(&edit);
        }
        if (!renders.empty()) RepairTopologyRender(r, chain, renders);
        for (const auto &[drawn, edit] : accumulated) AccumulateEdit(r, drawn->Topology, *edit);
        for (size_t i = 0u; i < repairs.size(); ++i) AppendEditedBlocks(repairs[i].Edited, meshes, *repaired[i]);
        MeshTopologyEdit::FinishAll(r, edits | std::views::transform([](MeshTopologyEdit &edit) { return &edit; }) | std::ranges::to<std::vector>());
        for (size_t i = 0u; i < repairs.size(); ++i) {
            if (repaired[i]->Repair) repairs[i].Marked = RepairMarks(*repaired[i]) | std::ranges::to<std::vector>();
        }
        if (!repairs.empty()) RepairDrawnElements(r, chain, repairs);
    }, extruded);
    for (size_t i = 0u; i < stage.size(); ++i) {
        if (auto *created = stage[i].Created) *created = results[i].Changed ? Outcome{results[i].GeometryId} : std::unexpected(NodeFault::Empty);
    }
}
// Copies the node's selection into its input record's selection once the round completed the passes that wrote the masks, on the round in the record's lane.
void PublishInput(state::Registry &r, uint32_t id, const Masks &masks) {
    if (!masks.Publish || !masks.In) return;
    AfterRound(masks, [&r, id, masks] {
        auto &chain = ChainOf(masks.Round);
        chain.Lane(id, [&] { r.Resources.get<MeshStore>().SelectMasks(r, chain, id, *masks.In); });
    });
}
// Stages the dependent tasks in order, each executing with the same stage of the other meshes' nodes of the step, and returns the input record, which they edit in place.
// An extrusion task fills `extruded`, and `kept` keeps the state the tasks' stages leave.
Outcome ExecuteStages(state::Registry &r, std::span<MeshTopologyTask> tasks, const Masks &masks, std::vector<ExtrusionSource> *extruded = nullptr, bool kept = false) {
    if (masks.Publish) {
        for (auto &task : tasks) PublishStoreSelection(r.Resources.get<const MeshStore>(), task);
    }
    auto &stages = masks.Topology.Stages;
    if (stages.size() < tasks.size()) stages.resize(tasks.size());
    for (size_t stage = 0u; stage < tasks.size(); ++stage) stages[stage].push_back({tasks[stage], masks.Drawn, extruded, kept});
    // The mesh entity's position edit work refers to the topology before the edits.
    if (masks.Drawn) {
        masks.Drawn->Recorded = true;
        ReleaseMeshEditWork(r, masks.Drawn->Entity);
    }
    return tasks.front().SourceId;
}
Outcome Stage(state::Registry &r, std::optional<MeshTopologyTask> task, const Masks &masks) {
    if (!task) return std::unexpected(NodeFault::Selection);
    return ExecuteStages(r, std::span{&*task, 1u}, masks);
}
// A topology node's run, which stages the task `plan` returns once the round completed the passes before it.
Outcome Execute(state::Registry &r, uint32_t id, const Masks &masks, std::function<std::optional<MeshTopologyTask>()> plan) {
    return PlanAfterRound(masks, id, [&r, masks, plan = std::move(plan)] { return Stage(r, plan(), masks); });
}
// The task `plan` builds from the mesh and its input elements, which an AddPrimitives task takes as its selection.
std::optional<MeshTopologyTask> PlannedTask(state::Registry &r, uint32_t id, const TopologyInput &input, auto &&plan) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto selection = Handles(meshes, id, input);
    std::optional<MeshTopologyTask> task = plan(Mesh{meshes, id}, selection);
    if (task && task->Op == MeshTopologyOp::AddPrimitives) task->Selection = selection;
    return task;
}
// Records reads through `record(chain, then)` on the step's chain, and returns the input record, which the outcome `take` makes of what `then` receives replaces as the node's planned outcome.
template<typename Read> Outcome AfterReads(uint32_t id, const Masks &masks, auto &&record, std::function<Outcome(Read)> take) {
    masks.Step.Waits = true;
    record(ReadsOf(masks.Step), [planned = masks.Planned, take = std::move(take)](Read read) {
        const auto outcome = take(std::move(read));
        if (planned) *planned = outcome;
    });
    return id;
}
// Deletes every vertex of each record in one topology call.
void DeleteVertices(state::Registry &r, std::span<const uint32_t> ids) {
    std::vector<MeshTopologyTask> tasks;
    for (const auto id : ids) {
        if (Mesh{r.Resources.get<const MeshStore>(), id}.VertexCount()) tasks.push_back(Task(id, MeshTopologyOp(uint32_t(DeleteMode::Vertices)), {}));
    }
    if (!tasks.empty()) ExecuteGeometryTopology(r, tasks);
}
// A slide measures from the `reference` vertex, or else from the first selected vertex.
Outcome Reposition(state::Registry &r, uint32_t id, const Masks &masks, PositionEditOp op, float factor, uint32_t repeat, const PositionOperationOptions &options = {}, const Transform &world = {}, uint32_t reference = InvalidOffset) {
    const PositionOperationTarget target{.StoreId = id, .Selection = masks.In ? SelectionSource{*masks.In} : SelectionSource{}, .World = world, .Reference = reference, .Every = !masks.In};
    EditPositions(r, masks, target, op, factor, repeat, options);
    return id;
}
// Rotate the selected faces' corner UVs of a set or colors clockwise for operation 0, counterclockwise for 1, or reverse them for 2.
// The edit plans from its selection once the round completed the passes before it and records on the round in its record's lane, where a drawn record's changed triangles join the round's repair of its render records or its accumulated edits once the round submits.
Outcome EditFaceAttributes(state::Registry &r, uint32_t id, const Masks &masks, bool colors, uint32_t uv_set, uint32_t operation) {
    if (uv_set >= MeshStore::MaxUvSets) return std::unexpected(NodeFault::Parameter);
    if (masks.Drawn) masks.Drawn->Recorded = true;
    AfterRound(masks, [&r, id, masks, colors, uv_set, operation] {
        auto &meshes = r.Resources.get<MeshStore>();
        auto &chain = ChainOf(masks.Round);
        const PositionOperationTarget target{.StoreId = id, .Selection = SourceOf(meshes, id, masks)};
        chain.Lane(id, [&] {
            const auto change = EncodeFaceAttributeOperation(meshes, GetMeshPipelines(r), chain, target, colors, uv_set, operation);
            if (!masks.Drawn || !change) return;
            const auto &a = meshes.Arenas();
            const auto &record = meshes.Get(id);
            const auto &faces = change->Faces;
            const auto bound = std::min(faces.Incidence, a.Triangles.Set(record.TriangleData).BlockCount);
            const auto handles = a.Triangles.HandleBound(record.TriangleData);
            chain.Scratch.ReserveAdditional(1u + ElementWorkWords(handles, bound) + SortElementWorkWords(WorkCapacity(handles, bound)));
            chain.AfterSubmit([&r, &chain, &round = masks.Round, drawn = masks.Drawn, id, triangles = EncodeFaceTriangles(r, chain, id, faces)] mutable {
                triangles.Finish(chain);
                TakeShaded(r, chain, round.Repairs, *drawn, id, triangles);
            });
        });
    });
    return id;
}
uint32_t Axes(bool x, bool y, bool z) { return uint32_t(x) | uint32_t(y) << 1u | uint32_t(z) << 2u; }

// Per-input attribute IDs, where InvalidOffset selects the constant parameter, and the op program a position edit's input calls at each vertex instead.
struct LinkedAttributes {
    std::array<uint32_t, MaxInputs> Attributes;
    std::array<std::optional<PositionProgram>, MaxInputs> Programs;
    template<auto M> uint32_t Of() const { return Attributes[ParameterSocket<M>]; }
};
// The input sockets of the kind's per-element parameters, in declaration order.
template<typename K> constexpr auto ElementSockets = [] {
    constexpr auto sockets = KindSockets(KindOf<K>);
    constexpr auto per_element = [](const Socket &socket) { return socket.Varies && socket.Field != NoField; };
    std::array<uint16_t, std::ranges::count_if(sockets, per_element)> found{};
    for (uint16_t s = 0u, n = 0u; s < sockets.size(); ++s) {
        if (per_element(sockets[s])) found[n++] = s;
    }
    return found;
}();
// Returns the attribute when its domain and type match, or InvalidOffset for a constant.
std::expected<uint32_t, NodeFault> ElementParameter(const MeshStore &meshes, uint32_t attribute, AttributeDomain domain, AttributeType type = AttributeType::Float) {
    if (attribute == InvalidOffset) return InvalidOffset;
    const auto &a = meshes.GetAttribute(attribute);
    if (a.Domain != domain || a.Type != type) return std::unexpected(NodeFault::Attribute);
    return attribute;
}
// The domain a kind evaluates its varying parameters on.
AttributeDomain FieldDomain(const Extrude &p) { return p.Mode == ExtrudeMode::FacesIndividual ? AttributeDomain::Face : AttributeDomain::Vertex; }
AttributeDomain FieldDomain(const Poke &) { return AttributeDomain::Face; }
AttributeDomain FieldDomain(const Bevel &p) { return p.Vertices ? AttributeDomain::Vertex : AttributeDomain::Edge; }
AttributeDomain FieldDomain(const auto &) { return AttributeDomain::Vertex; }
// The elements an operator acts on, which its linked selection evaluates on without a selection of its own.
Element NaturalDomain(const Delete &p) {
    switch (p.Mode) {
        case DeleteMode::Edges:
        case DeleteMode::OnlyEdgesAndFaces: return Element::Edge;
        case DeleteMode::Faces:
        case DeleteMode::OnlyFaces: return Element::Face;
        case DeleteMode::Vertices:
        case DeleteMode::Loose: return Element::Vertex;
    }
    std::unreachable();
}
Element NaturalDomain(const Dissolve &p) {
    if (p.Mode == DissolveMode::Edges) return Element::Edge;
    if (p.Mode == DissolveMode::Faces) return Element::Face;
    return Element::Vertex;
}
Element NaturalDomain(const Extrude &p) {
    if (p.Mode == ExtrudeMode::Region || p.Mode == ExtrudeMode::FacesIndividual) return Element::Face;
    if (p.Mode == ExtrudeMode::Edges) return Element::Edge;
    return Element::Vertex;
}
Element NaturalDomain(const Bevel &p) { return p.Vertices ? Element::Vertex : Element::Edge; }
template<typename K> Element NaturalDomain(const K &) { return Kind<K>.Natural; }
NamedAttributeRef RefOf(const MeshStore &meshes, uint32_t attribute) { return attribute == InvalidOffset ? NamedAttributeRef{} : meshes.AttributeRef(attribute); }
Outcome ExecuteElement(state::Registry &r, MeshTopologyTask task, const Masks &masks, AttributeDomain domain, uint32_t param0, uint32_t param1 = InvalidOffset) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto first = ElementParameter(meshes, param0, domain), second = ElementParameter(meshes, param1, domain);
    if (!first || !second) return std::unexpected(NodeFault::Attribute);
    task.Param0Attribute = RefOf(meshes, *first);
    task.Param1Attribute = RefOf(meshes, *second);
    return Stage(r, std::move(task), masks);
}
// The edit takes its per-vertex factor from the program the kind's per-element parameter calls at each vertex, or else from its linked float vertex attribute.
template<typename K> Outcome RepositionBy(state::Registry &r, uint32_t id, const Masks &masks, PositionEditOp op, float factor, uint32_t repeat, const LinkedAttributes &linked, PositionOperationOptions options) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto socket = ElementSockets<K>[0];
    if (const auto &program = linked.Programs[socket]) {
        options.Input = *program;
    } else {
        const auto attribute = ElementParameter(meshes, linked.Attributes[socket], AttributeDomain::Vertex);
        if (!attribute) return std::unexpected(attribute.error());
        if (*attribute != InvalidOffset) options.Input = meshes.AttributeRef(*attribute);
    }
    return Reposition(r, id, masks, op, factor, repeat, options);
}

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
// The run of a node of a display path's chained part.
// Drawn holds the edits of the input record a mesh entity draws, or is null for a record without current render records.
// Reuse is how the node's topology reuses an earlier cook's, and KeepsExtrusion keeps the state an Extrude node's topology leaves.
struct PathNodeRun {
    DrawnEdits *Drawn;
    TopologyReuse Reuse;
    bool KeepsExtrusion;
};

vec3 NormalizedSum(std::span<const vec3> vectors) {
    vec3 sum{};
    for (const auto &v : vectors) sum += v;
    return Length(sum) > 0.f ? Normalize(sum) : sum;
}
// Keeps the occurrence's handle over the selected elements of the input record: the mean position of the selected vertices and the normalized sum of the selected faces' normals.
void KeepHandle(state::Registry &r, std::span<const state::Entity> path, uint32_t id, const Masks &masks) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto selection = Handles(meshes, id, masks);
    NodeHandle support;
    for (const auto &position : VertexPositions(meshes, id, selection.Vertices)) support.Center += position;
    if (!selection.Vertices.empty()) support.Center /= float(selection.Vertices.size());
    support.Normal = NormalizedSum(FaceNormals(meshes, id, selection.Faces));
    Keep(r, path, support);
}
// The material of each primitive of the record.
std::vector<uint32_t> MaterialsOf(const MeshStore &meshes, uint32_t id) {
    const auto materials = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(id).PrimitiveMaterials);
    return {materials.begin(), materials.end()};
}
// Gives the record's first primitives the materials.
void TakeMaterials(MeshStore &meshes, std::span<const uint32_t> materials, uint32_t to) {
    const auto held = MaterialsOf(meshes, to);
    const auto count = std::min(materials.size(), held.size());
    if (std::ranges::equal(materials.first(count), std::span{held}.first(count))) return;
    std::ranges::copy(materials.first(count), meshes.EditPrimitiveMaterials(to).begin());
}
// A path node keeps its computed extrusion, and a path node without a linked selection keeps the extruded topology's mesh storage state where its run asks and its record has current render records, which a later run that reuses topology restores.
// The extrusion plans once the round completed the passes before it, executes with the stage of the other meshes' nodes, a restore joins the round, and the offsets move the extruded vertices on the round once the stages executed.
Outcome RunExtrude(state::Registry &r, std::span<const state::Entity> node, const Extrude &p, uint32_t id, const Masks &m, const NodeSelection &resolved, const LinkedAttributes &linked, std::vector<uint32_t> &temporaries, const std::optional<PathNodeRun> &path_run) {
    const bool retain = path_run.has_value();
    const auto reuse = path_run && !resolved.Field ? path_run->Reuse : TopologyReuse::None;
    const bool keep = path_run && path_run->KeepsExtrusion && !resolved.Field && m.Drawn;
    const auto domain = FieldDomain(p);
    const auto offset = ElementParameter(r.Resources.get<const MeshStore>(), linked.Of<&Extrude::Offset>(), domain, AttributeType::Vec3);
    const auto scale = ElementParameter(r.Resources.get<const MeshStore>(), linked.Of<&Extrude::OffsetScale>(), domain);
    if (!offset || !scale) return std::unexpected(NodeFault::Attribute);
    const auto key = retain ? SelectionKey(resolved) : store::Hash128{};
    if (m.Drawn) m.Drawn->Recorded = true;
    return PlanAfterRound(m, id, [&r, node, p, id, m, domain, offset = *offset, scale = *scale, key, reuse, temporaries = std::exchange(temporaries, {}), retain, keep] -> Outcome {
        auto &meshes = r.Resources.get<MeshStore>();
        const auto *prior = retain ? CookedOf<NodeResult>(r, node) : nullptr;
        const auto *cached = CookedOf<NodeExtrusion>(r, node);
        const auto *stage = reuse != TopologyReuse::None && prior && cached && prior->StoreId == id && cached->Mode == p.Mode && cached->Selection == key ? cached : nullptr;
        // The extrusion's topology fills the computed extrusion, which the offsets read after it.
        const auto computed = std::make_shared<NodeExtrusion>(NodeExtrusion{.Selection = key, .Mode = p.Mode});
        std::vector<std::pair<uint32_t, vec3>> offsets;
        const bool fields = offset != InvalidOffset || scale != InvalidOffset;
        const bool per_element = fields && p.Mode != ExtrudeMode::Region && p.AlongNormals;
        if (!stage || fields) {
            const auto selection = Handles(meshes, id, m);
            const bool faces = domain == AttributeDomain::Face;
            const auto face_normals = !stage || (per_element && faces) ? FaceNormals(meshes, id, selection.Faces) : std::vector<vec3>{};
            if (!stage) {
                computed->RegionNormal = NormalizedSum(face_normals);
                stage = computed.get();
            }
            if (fields) {
                const auto &elements = faces ? selection.Faces : selection.Vertices;
                const auto vertex_normals = per_element && !faces ? VertexNormals(meshes, id, selection.Vertices) : std::vector<vec3>{};
                const auto &normals = faces ? face_normals : vertex_normals;
                for (uint32_t i = 0u; i < elements.size(); ++i) {
                    const auto source = elements[i];
                    const auto normal = per_element ? normals[i] : stage->RegionNormal;
                    const auto direction = offset != InvalidOffset ? meshes.AttributeView<vec3>(offset)[source] : p.AlongNormals ? normal :
                                                                                                                                 p.Offset;
                    offsets.emplace_back(source, direction * (scale != InvalidOffset ? meshes.AttributeView<float>(scale)[source] : p.OffsetScale));
                }
                std::ranges::sort(offsets, {}, &std::pair<uint32_t, vec3>::first);
            }
        }
        for (const auto attribute : temporaries) meshes.ReleaseAttribute(attribute);
        bool restored = reuse == TopologyReuse::Restored;
        if (!restored && stage->State != store::Hash128{}) {
            WaitForRender(r);
            // The restore reads with a reader of its own, since mesh storage changes between the restores of a step's plans.
            MeshStore::StateReaders readers;
            const auto materials = MaterialsOf(meshes, id);
            if (const auto change = meshes.RestoreRecord(stage->State, id, readers, m.Drawn && RefreshesRestore(r, id, m.Drawn->Entity))) {
                TakeMaterials(meshes, materials, id);
                // The restore reconciles with the round's other restores before the offsets move the vertices.
                if (m.Drawn) m.Drawn->Restored = JoinRestore(r, m.Round, *change, m.Drawn->Entity);
                else m.Round.Restored.push_back(*change);
                restored = true;
            }
        }
        // Moves the extruded vertices by their offsets and keeps a computed extrusion.
        // A restored run reads the extrusion the node keeps, which no other node's run replaces.
        const auto edit = [&r, node, p, id, m, scale, offsets = std::move(offsets), computed = !restored || stage == computed.get() ? computed : nullptr, retain] {
            auto &meshes = r.Resources.get<MeshStore>();
            const auto &extrusion = computed ? *computed : *CookedOf<NodeExtrusion>(r, node);
            GeometrySelection selected;
            for (const auto &vertex : extrusion.Vertices) selected.Vertices.push_back(vertex.Output);
            SortUnique(selected.Vertices);
            if (!selected.Vertices.empty() && (p.OffsetScale != 0.f || scale != InvalidOffset)) {
                const PositionOperationTarget target{.StoreId = id, .Selection = std::move(selected)};
                const bool common = offsets.empty() && (!p.AlongNormals || p.Mode == ExtrudeMode::Region);
                uint32_t attribute = InvalidOffset;
                PositionOperationOptions options;
                if (common) {
                    options.Transform = PositionTransform{.Delta = {.P = (p.AlongNormals ? extrusion.RegionNormal : p.Offset) * p.OffsetScale}};
                } else {
                    attribute = meshes.AddAttribute(id, AttributeDomain::Vertex, AttributeType::Vec3, ".cook.extrude-offset").value();
                    for (const auto &vertex : extrusion.Vertices) {
                        const auto normal = p.Mode == ExtrudeMode::Region ? extrusion.RegionNormal : vertex.Normal;
                        const vec3 displacement = offsets.empty() ? (p.AlongNormals ? normal : p.Offset) * p.OffsetScale : std::ranges::lower_bound(offsets, vertex.Source, {}, &std::pair<uint32_t, vec3>::first)->second;
                        meshes.SetAttribute<vec3>(attribute, vertex.Output, std::span{&displacement, 1u});
                    }
                    options.Flags = PositionEditAttributeOffset;
                    options.Input = meshes.AttributeRef(attribute);
                }
                // The round releases the offsets once it submits.
                EditPositions(r, m, target, common ? PositionEditOp::Transform : PositionEditOp::Attribute, 1.f, 1u, options);
                if (attribute != InvalidOffset) m.Round.Temporaries.push_back(attribute);
                SubmitPastBudget(r, m.Round);
            }
            if (retain && computed) Keep(r, node, std::move(*computed));
        };
        if (restored) {
            m.Topology.Later.push_back([edit](StageRound &) { edit(); });
            return id;
        }
        computed->RegionNormal = stage->RegionNormal;
        auto task = *PlanTopology(r, p, id, InputOf(m));
        (void)ExecuteStages(r, std::span{&task, 1u}, m, &computed->Vertices, keep);
        m.Topology.Keeps = m.Topology.Keeps || keep;
        m.Topology.Later.push_back([edit, computed, keep](StageRound &later) {
            if (keep) computed->State = later.Kept;
            edit();
        });
        return id;
    });
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
        const auto direction = p.AlongNormals ? NormalizedSum(FaceNormals(r.Resources.get<const MeshStore>(), id, selection.Faces)) : p.Offset;
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

namespace {
Outcome RunExtrudeSteps(state::Registry &r, const Extrude &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    if (linked.Of<&Extrude::Offset>() != InvalidOffset || linked.Of<&Extrude::OffsetScale>() != InvalidOffset) return std::unexpected(NodeFault::Parameter);
    return PlanAfterRound(m, id, [&r, p, id, m] -> Outcome {
        auto task = PlanTopology(r, p, id, InputOf(m));
        if (!task) return std::unexpected(NodeFault::Parameter);
        return Stage(r, std::move(task), m);
    });
}
Outcome Run(state::Registry &r, const SubdivisionSurface &p, uint32_t id, const Masks &m) {
    return PlanAfterRound(m, id, [&r, p, id, m] -> Outcome {
        if (!Mesh{r.Resources.get<const MeshStore>(), id}.FaceCount() || !p.Levels) return id;
        std::vector<MeshTopologyTask> levels;
        for (uint32_t level = 0u; level < p.Levels; ++level) levels.push_back(Task(id, MeshTopologyOp::Subdivide, {}, 1.f, 0.f, TopologyFlagFacePoints));
        return ExecuteStages(r, levels, m);
    });
}
Outcome Run(state::Registry &r, const SmoothSurface &p, uint32_t id, const Masks &m) {
    return PlanAfterRound(m, id, [&r, p, id, m] -> Outcome {
        const Mesh mesh{r.Resources.get<const MeshStore>(), id};
        if (!mesh.FaceCount()) return id;
        if (!mesh.HasClosedSurface()) return std::unexpected(NodeFault::NotClosed);
        auto tasks = SmoothSurfaceTasks(mesh, p.Cuts);
        return ExecuteStages(r, tasks, m);
    });
}
// A topology node's run whose task takes the linked fields of the kind's per-element parameters as its parameter attributes, in their order, read at each element of the kind's field domain once the round completed the passes before it.
template<typename K> Outcome RunElement(state::Registry &r, const K &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    constexpr auto &sockets = ElementSockets<K>;
    const auto field = [&](size_t i) { return i < sockets.size() ? linked.Attributes[sockets[i]] : InvalidOffset; };
    return PlanAfterRound(m, id, [&r, p, id, m, first = field(0u), second = field(1u)] { return ExecuteElement(r, PlanTopology(r, p, id, InputOf(m)), m, FieldDomain(p), first, second); });
}
Outcome Run(state::Registry &r, const SmoothVertices &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    return RepositionBy<SmoothVertices>(r, id, m, PositionEditOp::Smooth, p.Factor, std::clamp(p.Repeat, 1u, 1000u), linked, {.Axes = Axes(p.X, p.Y, p.Z)});
}
Outcome Run(state::Registry &r, const ShrinkFatten &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    return RepositionBy<ShrinkFatten>(r, id, m, PositionEditOp::ShrinkFatten, p.Distance, 1u, linked, {.Flags = (p.Even ? PositionEditEvenOffset : 0u) | (m.Domain == Element::Face ? PositionEditSelectedFaceNormals : 0u)});
}
Outcome Run(state::Registry &r, const Randomize &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    return RepositionBy<Randomize>(r, id, m, PositionEditOp::Randomize, p.Amount, 1u, linked, {.Randomize = PositionRandomize{p.Uniform, p.Normal, p.Seed}});
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
// Deletes each side of the cut the Bisect clears, from side `first` on.
// A delete reads the plane sides of the record the edit before it leaves, so each one plans in the next round.
Outcome ClearSides(state::Registry &r, const Bisect &p, Outcome cut, uint32_t first, const Masks &m) {
    const std::array sides{std::pair{p.ClearInner, PlaneSideKind::Negative}, std::pair{p.ClearOuter, PlaneSideKind::Positive}};
    const auto side = std::ranges::find(sides.begin() + first, sides.end(), true, &std::pair<bool, PlaneSideKind>::first);
    if (!cut || side == sides.end()) return cut;
    m.Topology.Later.push_back([&r, p, next = uint32_t(side - sides.begin()) + 1u, kind = side->second, id = *cut, m](StageRound &later) {
        const auto staged = Later(m, later);
        const auto record = [&](mtl::ComputeChain &chain, auto then) { RecordPlaneSideDeleteTask(r, chain, Mesh{r.Resources.get<const MeshStore>(), id}, p.Point, p.Normal, kind, PlaneSideTolerance, std::move(then)); };
        (void)AfterReads<std::optional<MeshTopologyTask>>(id, staged, record, [&r, p, next, id, staged](std::optional<MeshTopologyTask> task) { return ClearSides(r, p, task ? Stage(r, std::move(task), staged) : Outcome{id}, next, staged); });
    });
    return cut;
}
Outcome Run(state::Registry &r, const Bisect &p, uint32_t id, const Masks &m) {
    if (Dot(p.Normal, p.Normal) <= 0.f) return std::unexpected(NodeFault::Parameter);
    return PlanAfterRound(m, id, [&r, p, id, m] { return ClearSides(r, p, Stage(r, PlaneCutTask(Mesh{r.Resources.get<const MeshStore>(), id}, p.Point, p.Normal), m), 0u, m); });
}
// The projection creates its record once the round completed the passes before it.
Outcome Run(state::Registry &r, const Project &p, uint32_t id, const Masks &m) {
    if (Dot(p.Normal, p.Normal) <= 0.f) return std::unexpected(NodeFault::Parameter);
    // The projection plans on the step's chain with the other meshes' projections.
    return PlanAfterRound(m, id, [&r, p, id, m] {
        const auto record = [&](mtl::ComputeChain &chain, auto then) { RecordProjection(r, chain, id, PlaneAt(p.Point, p.Normal), std::move(then)); };
        return AfterReads<std::optional<uint32_t>>(id, m, record, [](std::optional<uint32_t> projected) -> Outcome {
            if (!projected) return std::unexpected(NodeFault::Empty);
            return *projected;
        });
    });
}
// The hull plans on the step's chain with the other meshes' hulls.
Outcome Run(state::Registry &r, const ConvexHull &, uint32_t id, const Masks &m) {
    return PlanAfterRound(m, id, [&r, id, m] {
        const auto &meshes = r.Resources.get<const MeshStore>();
        const auto record = [&](mtl::ComputeChain &chain, auto then) { RecordConvexHullTasks(r, chain, Mesh{meshes, id}, Handles(meshes, id, m), std::move(then)); };
        return AfterReads<std::vector<MeshTopologyTask>>(id, m, record, [&r, m](std::vector<MeshTopologyTask> tasks) -> Outcome {
            if (tasks.empty()) return std::unexpected(NodeFault::Selection);
            return ExecuteStages(r, tasks, m);
        });
    });
}
// Each round of the simplify reads the record its previous round's Decimate leaves, so its passes record on the step's chain and each Decimate executes with the same stage of the other meshes' nodes.
// The outcome is the node's fault when no round changed the record.
Outcome Run(state::Registry &r, const Simplify &p, uint32_t id, const Masks &m) {
    // The rounds' edits join the accumulated edits, which one repair covers.
    if (m.Drawn) m.Drawn->Last = false;
    return PlanAfterRound(m, id, [&r, p, id, m] -> Outcome {
        const auto selection = Handles(r.Resources.get<const MeshStore>(), id, m);
        // The stages of the round a Decimate joins.
        const auto staged = std::make_shared<StageRound *>(&m.Topology);
        const auto decimate = [&r, m, staged](const MeshTopologyTask &task, std::function<void()> next) {
            (void)Stage(r, task, Later(m, **staged));
            (*staged)->Later.push_back([staged, next = std::move(next)](StageRound &later) {
                *staged = &later;
                next();
            });
        };
        const auto done = [id, planned = m.Planned](bool changed) {
            if (planned) *planned = changed ? Outcome{id} : Outcome{std::unexpected(NodeFault::Selection)};
        };
        if (!RecordSimplify(r, ReadsOf(m.Step), id, selection, p.Tolerance, p.Ratio, decimate, done)) return std::unexpected(NodeFault::Selection);
        return id;
    });
}
// The faces to flip read on the step's chain, each round of their components once the round before it completed.
Outcome Run(state::Registry &r, RecalculateNormals, uint32_t id, const Masks &m) {
    return PlanAfterRound(m, id, [&r, id, m] {
        const auto selection = Handles(r.Resources.get<const MeshStore>(), id, m);
        const auto record = [&](mtl::ComputeChain &chain, auto then) { RecordFaceFlips(r.Resources.get<const MeshStore>(), GetMeshPipelines(r), chain, id, selection, std::move(then)); };
        return AfterReads<std::vector<uint32_t>>(id, m, record, [&r, id, m, selection](std::vector<uint32_t> faces) -> Outcome {
            if (faces.empty()) return id;
            faces.insert(faces.begin(), uint32_t(faces.size()));
            return Stage(r, MeshTopologyTask{.SourceId = id, .Op = MeshTopologyOp::FlipNormals, .List = std::move(faces), .Selection = selection}, m);
        });
    });
}

// The planner reads the record's chains on the step's chain once the round completed the passes before it.
Outcome PlanAfterReads(state::Registry &r, uint32_t id, const Masks &m, auto &&record) {
    return PlanAfterRound(m, id, [&r, id, m, record] {
        const auto selection = Handles(r.Resources.get<const MeshStore>(), id, m);
        return AfterReads<std::optional<MeshTopologyTask>>(id, m, [&](mtl::ComputeChain &chain, auto then) { record(chain, Mesh{r.Resources.get<const MeshStore>(), id}, selection, std::move(then)); }, [&r, m, selection](std::optional<MeshTopologyTask> task) {
            if (task && task->Op == MeshTopologyOp::AddPrimitives) task->Selection = selection;
            return Stage(r, std::move(task), m);
        });
    });
}
Outcome Run(state::Registry &r, const Fill &p, uint32_t id, const Masks &m) {
    return PlanAfterReads(r, id, m, [&r, sides = p.Sides](mtl::ComputeChain &chain, const Mesh &mesh, const GeometrySelection &selection, auto then) { RecordFillTask(r, chain, mesh, selection, sides, std::move(then)); });
}
Outcome Run(state::Registry &r, BridgeEdgeLoops, uint32_t id, const Masks &m) {
    return PlanAfterReads(r, id, m, [&r](mtl::ComputeChain &chain, const Mesh &mesh, const GeometrySelection &selection, auto then) { RecordBridgeEdgeLoopsTask(r, chain, mesh, selection, std::move(then)); });
}
Outcome Run(state::Registry &r, const GridFill &p, uint32_t id, const Masks &m) {
    return PlanAfterReads(r, id, m, [&r, span = p.Span](mtl::ComputeChain &chain, const Mesh &mesh, const GeometrySelection &selection, auto then) { RecordGridFillTask(r, chain, mesh, selection, span, std::move(then)); });
}
// The task selects only edges because the kernel selects each rotated edge's apexes as the vertices to connect.
Outcome Run(state::Registry &r, EdgeRotate, uint32_t id, const Masks &m) {
    return Execute(r, id, m, [&r, id, m] { return std::optional{MeshTopologyTask{.SourceId = id, .Op = MeshTopologyOp::RotateEdges, .SelectionElement = Element::Edge, .Selection = GeometrySelection{.Edges = Handles(r.Resources.get<const MeshStore>(), id, m).Edges}}}; });
}
Outcome Run(state::Registry &r, const BeautifyFaces &, uint32_t id, const Masks &m) {
    return PlanAfterReads(r, id, m, [&r](mtl::ComputeChain &chain, const Mesh &mesh, const GeometrySelection &selection, auto then) { RecordBeautifyFaceTask(r, chain, mesh, selection, std::move(then)); });
}
// Repeats on the selected elements that survive each iteration, `remaining` more times or until an iteration changes nothing.
// An iteration plans from the record the iteration before it leaves, so each one plans in the next round.
Outcome Unsubdivided(state::Registry &r, Outcome edited, GeometrySelection selection, uint32_t remaining, const Masks &m) {
    if (!edited || !remaining) return edited;
    m.Topology.Later.push_back([&r, selection = std::move(selection), remaining, id = *edited, m](StageRound &later) {
        const auto staged = Later(m, later);
        const auto &meshes = r.Resources.get<const MeshStore>();
        auto surviving = staged.Publish ? EditorGeometrySelection(meshes, id) : selection;
        for (const auto [element, handles] : {std::pair{Element::Vertex, &surviving.Vertices}, std::pair{Element::Edge, &surviving.Edges}, std::pair{Element::Face, &surviving.Faces}}) {
            std::erase_if(*handles, [&](uint32_t handle) { return !meshes.IsLiveElement(id, element, ElementHandle{handle}); });
        }
        const auto record = [&](mtl::ComputeChain &chain, auto then) { RecordUnsubdivideTask(r, chain, Mesh{meshes, id}, surviving, std::move(then)); };
        (void)AfterReads<std::optional<MeshTopologyTask>>(id, staged, record, [&r, surviving, remaining, id, staged](std::optional<MeshTopologyTask> task) {
            return task ? Unsubdivided(r, Stage(r, std::move(task), staged), surviving, remaining - 1u, staged) : Outcome{id};
        });
    });
    return edited;
}
// Each iteration's plan reads on the step's chain.
Outcome Run(state::Registry &r, const Unsubdivide &p, uint32_t id, const Masks &m) {
    return PlanAfterRound(m, id, [&r, p, id, m] {
        const auto selection = Handles(r.Resources.get<const MeshStore>(), id, m);
        const auto record = [&](mtl::ComputeChain &chain, auto then) { RecordUnsubdivideTask(r, chain, Mesh{r.Resources.get<const MeshStore>(), id}, selection, std::move(then)); };
        return AfterReads<std::optional<MeshTopologyTask>>(id, m, record, [&r, m, selection, remaining = std::clamp(p.Iterations, 1u, 1000u) - 1u](std::optional<MeshTopologyTask> task) { return Unsubdivided(r, Stage(r, std::move(task), m), selection, remaining, m); });
    });
}
Outcome Run(state::Registry &r, const Knife &p, uint32_t id, const Masks &m) {
    return Execute(r, id, m, [&r, p, id] {
        auto task = KnifeTask(Mesh{r.Resources.get<const MeshStore>(), id}, p.ViewProjection * ToMatrix(p.Frame), p.Extent, p.Start, p.End);
        task.Flags |= TopologyFlagSelectAll;
        return std::optional{std::move(task)};
    });
}
Outcome Run(state::Registry &r, const CycleCornerAttribute &p, uint32_t id, const Masks &m) {
    return EditFaceAttributes(r, id, m, p.Attribute == CornerAttribute::Colors, p.UVSet, uint32_t(p.Cycle));
}
// The edit plans from its selection once the round completed the passes before it, and records its parts on the step's chain, each once the part before it completed.
// A drawn record's changed triangles join the step's repair of its render records or its accumulated edits.
Outcome EditSharpness(state::Registry &r, uint32_t id, const Masks &masks, SelectionSource selection, EditSharpnessOperation operation, bool value, float angle = 0.f) {
    if (masks.Drawn) masks.Drawn->Recorded = true;
    AfterRound(masks, [&r, id, masks, selection = std::move(selection), operation, value, angle] {
        auto &chain = ReadsOf(masks.Step);
        RecordSharpness(r, chain, {id, selection}, operation, value, angle, [&r, &chain, id, masks](std::optional<SharpnessOperationChange> change) {
            if (!masks.Drawn || !change) return;
            r.Resources.get<GpuSceneState>().EditSelectionDirty = true;
            TakeShaded(r, chain, masks.Step.Repairs, *masks.Drawn, id, change->Triangles);
        });
    });
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
Outcome Run(state::Registry &r, MeshToPoints, uint32_t id, const Masks &m) {
    return PlanAfterRound(m, id, [&r, id, m] -> Outcome {
        if (!Mesh{r.Resources.get<const MeshStore>(), id}.EdgeCount()) return id;
        auto task = Task(id, MeshTopologyOp(uint32_t(DeleteMode::OnlyEdgesAndFaces)), {});
        return ExecuteStages(r, std::span{&task, 1u}, Masks{.Round = m.Round, .Step = m.Step, .Topology = m.Topology});
    });
}
// The record holds the lattice points the Value attribute was evaluated on.
// The march adds its task to the first stage and leaves the record in place.
Outcome Run(state::Registry &r, const LevelSet &p, uint32_t id, uint32_t attribute, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    if (attribute == InvalidOffset || !ElementParameter(meshes, attribute, AttributeDomain::Vertex)) return std::unexpected(NodeFault::Attribute);
    const auto lattice = *LatticeOver(p.Min, p.Max, p.EdgeLength);
    auto &stages = m.Topology.Stages;
    if (stages.empty()) stages.emplace_back();
    stages.front().push_back({.Task = LevelSetTask(meshes, id, lattice, attribute, p.Level), .Created = m.Planned});
    return id;
}
// The positions come from the program the Position input calls at each vertex, or else from its vec3 vertex attribute.
Outcome Run(state::Registry &r, const SetPosition &p, uint32_t id, uint32_t attribute, const std::optional<PositionProgram> &program, const Masks &m) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    if (!program && (attribute == InvalidOffset || !ElementParameter(meshes, attribute, AttributeDomain::Vertex, AttributeType::Vec3))) return std::unexpected(NodeFault::Attribute);
    using Input = decltype(PositionOperationOptions::Input);
    return Reposition(r, id, m, PositionEditOp::Attribute, 1.f, 1u, {.Flags = p.Offset ? PositionEditAttributeOffset : 0u, .Input = program ? Input{*program} : Input{meshes.AttributeRef(attribute)}});
}

std::string_view NameOf(const AttributeName &name) { return {name.data(), strnlen(name.data(), name.size())}; }

Outcome Store(MeshStore &meshes, const StoreNamedAttribute &p, uint32_t id, const Operand &value, std::vector<uint32_t> &temporaries) {
    const auto name = NameOf(p.Name);
    const auto attribute = value.Attribute;
    if (attribute == InvalidOffset && !value.Constant) return std::unexpected(NodeFault::Attribute);
    if (name.empty()) return std::unexpected(NodeFault::Parameter);
    if (attribute != InvalidOffset && meshes.GetAttribute(attribute).NameView() == name) return id;
    if (const auto existing = meshes.FindAttribute(id, name); existing != InvalidOffset) meshes.ReleaseAttribute(existing);
    if (attribute == InvalidOffset) return meshes.AddAttribute(id, p.Domain, StorageOf(value.Constant->Type), name, value.Constant->Words) ? Outcome{id} : std::unexpected(NodeFault::Attribute);
    const auto temporary = std::ranges::find(temporaries, attribute);
    if (temporary == temporaries.end()) return meshes.CopyAttribute(attribute, name) ? Outcome{id} : std::unexpected(NodeFault::Attribute);
    temporaries.erase(temporary);
    return meshes.RenameAttribute(attribute, name) ? Outcome{id} : std::unexpected(NodeFault::Attribute);
}

// Sets the Bool attribute true at the masked elements of its domain.
void WriteSelection(MeshStore &meshes, uint32_t attribute, const GeometryMasks &masks) {
    const auto &held = meshes.GetAttribute(attribute);
    const auto element = ElementOf(held.Domain);
    const auto handles = ReadMasks(meshes, held.Record, element, masks.Get(element));
    if (handles.empty()) return;
    const auto count = handles.back() + 1u;
    const auto values = std::make_unique<bool[]>(count);
    for (const auto handle : handles) values[handle] = true;
    meshes.SetAttribute<bool>(attribute, 0u, std::span<const bool>{values.get(), count});
}
// The Bool attribute holds the masked elements of the selection's domain, or every element for absent masks.
Outcome Store(MeshStore &meshes, const StoreSelection &p, uint32_t id, const Masks &masks) {
    const auto name = NameOf(p.Name);
    if (name.empty()) return std::unexpected(NodeFault::Parameter);
    if (const auto existing = meshes.FindAttribute(id, name); existing != InvalidOffset) meshes.ReleaseAttribute(existing);
    const auto attribute = meshes.AddAttribute(id, AttributeDomainOf(masks.Domain), AttributeType::Bool, name, {masks.In ? 0u : 1u});
    if (!attribute) return std::unexpected(NodeFault::Attribute);
    if (masks.In) WriteSelection(meshes, *attribute, *masks.In);
    return id;
}

// The points take their instance attributes once the round completed the passes that evaluated them.
Outcome Run(state::Registry &r, const InstanceOnPoints &p, uint32_t id, const Masks &m, const LinkedAttributes &linked) {
    return PlanAfterRound(m, id, [&r, p, id, m, linked] -> Outcome {
        auto &meshes = r.Resources.get<MeshStore>();
        const auto rotation = ElementParameter(meshes, linked.Of<&InstanceOnPoints::Rotation>(), AttributeDomain::Vertex, AttributeType::Vec4);
        const auto scale = ElementParameter(meshes, linked.Of<&InstanceOnPoints::Scale>(), AttributeDomain::Vertex, AttributeType::Vec3);
        if (Mesh{meshes, id}.EdgeCount()) {
            auto task = Task(id, MeshTopologyOp(uint32_t(DeleteMode::OnlyEdgesAndFaces)), {});
            (void)ExecuteStages(r, std::span{&task, 1u}, Masks{.Round = m.Round, .Step = m.Step, .Topology = m.Topology});
        }
        for (const auto &[name, type, value, attribute] : {std::tuple{InstanceRotation, AttributeType::Vec4, Words(p.Rotation), rotation.value_or(InvalidOffset)}, std::tuple{InstanceScale, AttributeType::Vec3, Words(p.Scale), scale.value_or(InvalidOffset)}}) {
            if (const auto existing = meshes.FindAttribute(id, name); existing != InvalidOffset) meshes.ReleaseAttribute(existing);
            if (attribute != InvalidOffset) (void)meshes.CopyAttribute(attribute, name);
            else (void)meshes.AddAttribute(id, AttributeDomain::Vertex, type, name, value);
        }
        if (!rotation) return std::unexpected(rotation.error());
        return scale ? Outcome{id} : std::unexpected(scale.error());
    });
}

// The selection masks of a cook pass's nodes, one mirror of each selectable domain's element blocks.
// A record's masks are the entries of the blocks it owns, which hold one node's selection at a time.
struct MaskArenas {
    explicit MaskArenas(mtl::BufferContext &buffers)
        : Domains{BufferArena<SelectionBlock>{buffers, SlotType::Buffer, mtl::BufferLifetime::Workspace}, BufferArena<SelectionBlock>{buffers, SlotType::Buffer, mtl::BufferLifetime::Workspace}, BufferArena<SelectionBlock>{buffers, SlotType::Buffer, mtl::BufferLifetime::Workspace}} {}
    std::array<BufferArena<SelectionBlock>, 3> Domains;
};
// The record's masks, whose mirrors cover the blocks it owns.
GeometryMasks MasksOf(MeshStore &meshes, std::optional<MaskArenas> &arenas, uint32_t id) {
    if (!arenas) arenas.emplace(meshes.BufferContext());
    auto &domains = arenas->Domains;
    for (uint32_t d = 0u; d < 3u; ++d) {
        const auto owned = meshes.GetBlockList(id, SelectionDomains[d]).Blocks;
        if (!owned.empty()) domains[d].Mirror({0u, std::ranges::max(owned) + 1u});
    }
    return {{&domains[0], 0u}, {&domains[1], 0u}, {&domains[2], 0u}};
}
// The record's masks with the blocks it owns zeroed, after `fill(domain, masks)` sets them, where the span holds the domain's mask of each block.
GeometryMasks ZeroedMasks(MeshStore &meshes, std::optional<MaskArenas> &arenas, uint32_t id, auto &&fill) {
    const auto geometry = MasksOf(meshes, arenas, id);
    for (uint32_t d = 0u; d < 3u; ++d) {
        const auto owned = meshes.GetBlockList(id, SelectionDomains[d]).Blocks;
        const auto masks = owned.empty() ? std::span<SelectionBlock>{} : arenas->Domains[d].GetMutable({0u, std::ranges::max(owned) + 1u});
        for (const auto block : owned) masks[block] = SelectionBlock{};
        fill(d, masks);
    }
    return geometry;
}
// The masks of the inline selection, or none for every element.
// A published selection's other domains derive on the chain, in the record's lane.
std::optional<GeometryMasks> UploadSelection(state::Registry &r, std::optional<MaskArenas> &arena, uint32_t id, const InlineSelection *selection, mtl::ComputeChain &chain) {
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
    if (selection->Published) chain.Lane(id, [&] { meshes.DeriveSelectionMasks(r, chain, id, uploaded, selection->Domain); });
    return uploaded;
}

constexpr std::string_view TemporaryPrefix = ".cook.";
std::string TemporaryName(uint32_t node, uint32_t socket) { return std::format("{}{}.{}", TemporaryPrefix, node, socket); }
void ReleaseTemporaries(MeshStore &meshes, uint32_t id) {
    std::vector<uint32_t> temporaries;
    meshes.ForEachNamedAttribute(id, [&](uint32_t attribute, const auto &a) {
        if (a.NameView().starts_with(TemporaryPrefix)) temporaries.push_back(attribute);
    });
    for (const auto attribute : temporaries) meshes.ReleaseAttribute(attribute);
}

// Whether the input socket takes a single value of the type.
bool Takes(const CookPlan &plan, uint32_t node, uint16_t socket, DataType type) { return NodeSockets(plan, node)[socket].Types & TypeBit(type); }
// The value an input reads: its constant, a single value an op output holds with its type, or the stored field `attributes` holds for a varying one by input socket.
// An input reads no value of a type it does not take, or from an op whose evaluation faulted.
Operand OperandOf(const CookPlan &plan, std::span<const SocketValue> values, std::span<const uint32_t> attributes, uint32_t node, uint32_t input) {
    if (input >= InputCount[plan.Nodes[node].Kind]) return {};
    const auto link = plan.Inputs[plan.InputFirst[node] + input];
    if (link.Constant) return Takes(plan, node, uint16_t(input), link.Constant->Type) ? Operand{.Constant = link.Constant} : Operand{};
    if (link.FromNode == NoNode) return {};
    if (NodeSockets(plan, link.FromNode)[link.FromSocket].Varies) return {attributes.empty() ? InvalidOffset : attributes[input]};
    const auto &source = values[plan.SocketFirst[link.FromNode] + link.FromSocket];
    if (!Takes(plan, node, uint16_t(input), source.Type)) return {};
    return {.Constant = PortValue{source.Value, source.Type}};
}
// A value a port or an op passes to a parameter is clamped to the parameter's range, and an integer one is a signed MaterialX integer.
template<typename K> K Parameters(const state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node) {
    const auto &[entity, kind, parent, builtin] = plan.Nodes[node];
    auto p = Authored(r, plan.Nodes[node]).get<const K>(entity);
    if constexpr (field::Walkable<K>) {
        const auto [first, end] = ParameterRange(kind);
        for (auto s = first; s < end; ++s) {
            const auto value = OperandOf(plan, values, {}, node, s).Constant;
            if (!value) continue;
            AssignSocketParameter(p, KindSockets(kind)[s], value->Words);
        }
    }
    return p;
}
// The inline selection the node's selection input reads: a subnet node's own selection of the port it reads, or else the node's own, or null.
const InlineSelection *SelectionLiteral(const state::Registry &r, const CookPlan &plan, uint32_t node, uint16_t socket) {
    const auto &link = plan.Inputs[plan.InputFirst[node] + socket];
    if (link.SelectionSubnet != NoNode) return &*InstancePortOf(Authored(r, plan.Nodes[link.SelectionSubnet]), plan.Nodes[link.SelectionSubnet].Entity, link.SelectionPort)->Selection;
    return Authored(r, plan.Nodes[node]).try_get<const InlineSelection>(plan.Nodes[node].Entity);
}
// The domain the node's selection applies to: the edit domain its inline selection was captured in, or else the elements its operator acts on for a linked selection, and vertices for every element.
Element SelectionDomain(const state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node, uint16_t socket) {
    if (const auto *literal = SelectionLiteral(r, plan, node, socket)) return literal->Domain;
    const auto &link = plan.Inputs[plan.InputFirst[node] + socket];
    if (link.FromNode == NoNode && !link.Constant) return Element::Vertex;
    const auto &[entity, kind, parent, builtin] = plan.Nodes[node];
    auto domain = Element::Vertex;
    VisitKind(kind, [&]<typename K> { domain = NaturalDomain(values.empty() ? Authored(r, plan.Nodes[node]).get<const K>(entity) : Parameters<K>(r, plan, values, node)); });
    return domain;
}
// A socket constant, such as a subnet port's default, is a Bool field operand, and a linked field is the stored field `attributes` holds for it.
// A kind without a selection input selects every element.
// An unlinked selection input reading a subnet node's selection port takes that node's selection in place of its own.
NodeSelection ResolveSelection(const state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, std::span<const uint32_t> attributes, uint32_t node) {
    const auto sockets = KindSockets(plan.Nodes[node].Kind);
    const auto socket = uint16_t(std::ranges::find(sockets, true, &Socket::Selection) - sockets.begin());
    if (socket == sockets.size()) return {};
    const auto *literal = SelectionLiteral(r, plan, node, socket);
    const auto &link = plan.Inputs[plan.InputFirst[node] + socket];
    const bool linked = link.FromNode != NoNode || link.Constant;
    return {.Field = linked ? std::optional{OperandOf(plan, values, attributes, node, socket)} : std::nullopt, .Literal = literal && !literal->All ? literal : nullptr, .Domain = SelectionDomain(r, plan, values, node, socket)};
}
LinkedAttributes LinkedParameters(const CookPlan &plan, std::span<const SocketValue> values, std::span<const uint32_t> attributes, uint32_t node, const std::array<std::optional<PositionProgram>, MaxInputs> &programs) {
    const auto kind = plan.Nodes[node].Kind;
    const auto sockets = KindSockets(kind);
    const auto [first, end] = ParameterRange(kind);
    LinkedAttributes linked{.Programs = programs};
    linked.Attributes.fill(InvalidOffset);
    for (auto s = first; s < end; ++s) {
        if (sockets[s].Varies) linked.Attributes[s] = OperandOf(plan, values, attributes, node, s).Attribute;
    }
    return linked;
}

// An op program a cook pass evaluates on one domain.
struct OpProgram {
    ops::Domain Domain;
    OpGraph Graph;
    const ops::Compiled *Compiled;
};
// A program's plan shape, its consumer node or the op node of an output evaluated alone, its domain, its outputs, and the filters that held no image when it was requested.
using OpProgramKey = std::tuple<uint32_t, uint32_t, ops::Domain, ops::Domain, std::vector<std::pair<uint32_t, uint16_t>>, std::vector<uint32_t>>;
// What a plan's op layout and programs read of the plan: its structure, and the entities of its op nodes and of the nodes that read them.
struct PlanShape {
    std::vector<NodeRecord> Nodes; // Every other node holds no entity
    std::vector<PlanLink> Inputs;
    std::vector<Socket> Sockets;
    std::vector<uint32_t> Order, InputFirst, SocketFirst;
    bool operator==(const PlanShape &) const = default;
};
PlanShape ShapeOf(const CookPlan &plan) {
    const auto op = [&](uint32_t node) { return node != NoNode && plan.Nodes[node].Kind == KindOf<Op>; };
    PlanShape shape{.Inputs = plan.Inputs, .Sockets = plan.ViewSockets, .Order = plan.Order, .InputFirst = plan.InputFirst, .SocketFirst = plan.SocketFirst};
    shape.Nodes.reserve(plan.Nodes.size());
    for (uint32_t n = 0u; n < plan.Nodes.size(); ++n) {
        const auto &node = plan.Nodes[n];
        const bool reads = op(n) || std::ranges::any_of(NodeLinks(plan, n), [&](const PlanLink &link) { return op(link.FromNode); });
        shape.Nodes.push_back({reads ? node.Entity : state::Null, node.Kind, node.Parent, node.Builtin});
    }
    return shape;
}
uint64_t HashOf(const PlanShape &shape) {
    uint64_t h = shape.Nodes.size();
    const auto mix = [&h](uint64_t value) { h = store::Mix64(h ^ value); };
    for (const auto &node : shape.Nodes) mix(uint64_t(state::Integral(node.Entity)) << 32 | uint64_t(node.Kind) << 16 | node.Builtin), mix(node.Parent);
    for (const auto &link : shape.Inputs) {
        mix(uint64_t(link.FromNode) << 32 | link.FromSocket);
        if (link.Constant) {
            mix(uint64_t(link.Constant->Type));
            for (const auto word : link.Constant->Words) mix(word);
        }
    }
    for (const auto &socket : shape.Sockets) mix(uint64_t(socket.Types) << 8 | socket.Varies);
    for (const auto *values : {&shape.Order, &shape.InputFirst, &shape.SocketFirst})
        for (const auto value : *values) mix(value);
    return h;
}
// The op layouts and programs of one cook pass by plan shape, and the op nodes whose faults it reset.
// The plans of the meshes that instance one definition share one shape, which lays out and compiles once.
struct OpPass {
    std::vector<PlanShape> Shapes;
    std::unordered_multimap<uint64_t, uint32_t> ShapesByHash;
    std::deque<OpLayout> Layouts; // By shape
    std::map<state::Entity, uint32_t> MeshShapes; // The shape of each mesh's plan
    std::map<OpProgramKey, OpProgram> Programs;
    std::map<std::pair<uint32_t, std::string>, OpProgram> Splits; // By shape and temporary name, the program of a split on the domain it evaluates on
    std::set<std::pair<uint32_t, std::vector<bool>>> Prepared; // The shapes, with the nodes their plans demand, whose programs compiled
    std::map<state::Entity, std::vector<bool>> Reset; // By mesh, its plan nodes whose faults reset
    float Frame, Time; // The timeline's frame and time its programs read
    std::map<state::Entity, uint32_t> Records; // The records an evaluation at another frame holds for meshes, which Object Info nodes read in place of the drawn ones
    std::optional<MaskArenas> Masks; // The selection masks of the pass's nodes
};
// A cook pass at the timeline frame.
OpPass PassAt(const state::Registry &r, int frame) { return {.Frame = float(frame), .Time = animation::FrameSeconds(r, frame)}; }
// A procedure's op evaluation in one cook pass.
// Its programs read its block of the op value store.
// A published evaluation and one for display keep the images of the occurrences they evaluate at the cooks' frame.
// Unheld lists the filters that held no image, in the order they ran, which their readers read as their inputs.
struct OpCook {
    OpPass &Pass;
    state::Entity Mesh;
    uint32_t Shape;
    const OpLayout &Layout;
    SlotOffset Values;
    bool Publish, Keeps;
    std::vector<uint32_t> Unheld{};
};
// The program reads its values from its graph's first.
SlotOffset ValuesOf(const OpCook &cook, const OpProgram &program) { return {cook.Values.Slot, cook.Values.Offset + program.Graph.Base}; }
bool HasOps(const CookPlan &plan) {
    return std::ranges::any_of(plan.Nodes, [](const NodeRecord &node) { return node.Kind == KindOf<Op>; });
}
// The pass's shape of the mesh's plan, which lays out when the pass first meets it.
uint32_t ShapeIn(OpPass &pass, const state::Registry &r, state::Entity mesh, const CookPlan &plan) {
    if (const auto found = pass.MeshShapes.find(mesh); found != pass.MeshShapes.end()) return found->second;
    auto shape = ShapeOf(plan);
    const auto hash = HashOf(shape);
    const auto [first, last] = pass.ShapesByHash.equal_range(hash);
    const auto match = std::find_if(first, last, [&](const auto &entry) { return pass.Shapes[entry.second] == shape; });
    const auto index = match != last ? match->second : uint32_t(pass.Shapes.size());
    if (match == last) {
        pass.Shapes.push_back(std::move(shape));
        pass.Layouts.push_back(LayoutOf(r, plan));
        pass.ShapesByHash.emplace(hash, index);
    }
    pass.MeshShapes.emplace(mesh, index);
    return index;
}
const OpLayout &LayoutIn(OpPass &pass, const state::Registry &r, state::Entity mesh, const CookPlan &plan) {
    // A plan without op nodes lays out nothing.
    static const OpLayout None{};
    return HasOps(plan) ? pass.Layouts[ShapeIn(pass, r, mesh, plan)] : None;
}

// The mesh domain a domain input's text names.
ops::Domain DomainNamed(std::string_view name) {
    if (name == "edge") return ops::Domain::Edge;
    if (name == "face") return ops::Domain::Face;
    if (name == "corner") return ops::Domain::Corner;
    return ops::Domain::Vertex;
}
// The domain a stored-field op's value inputs are stored on, and the domain its output is stored on.
struct StoredDomains {
    ops::Domain Input, Output;
};
StoredDomains StoredDomainsOf(const state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node) {
    const auto &op = Authored(r, plan.Nodes[node]).get<const Op>(plan.Nodes[node].Entity);
    // Every declaration of a stored-field op has the same text inputs.
    const auto declaration = mx::NodeDeclarations(mx::Libraries(), op.Node).front();
    const auto domain = [&](std::string_view input) { return DomainNamed(TextOf(op, input, declaration)); };
    switch (StoredOpOf(op.Node)->Domains) {
        case StoredDomainRule::Text: return {domain("domain"), domain("domain")};
        case StoredDomainRule::SourceTarget: return {domain("source"), domain("target")};
        case StoredDomainRule::Selection: {
            const auto selected = ops::DomainOf(AttributeDomainOf(SelectionDomain(r, plan, values, node, 0u)));
            return {selected, selected};
        }
        case StoredDomainRule::Vertex: return {ops::Domain::Vertex, ops::Domain::Vertex};
        case StoredDomainRule::Edge: return {ops::Domain::Edge, ops::Domain::Edge};
    }
    std::unreachable();
}
// The domain a consumer's varying input evaluates on, which the consumer decides.
// A selection evaluates on the domain its node's selection applies to, a stored-field op's input on its input domain, a filter's on UV texels, Store Named Attribute's value on its domain, Bake's value on texels, and every other input on the elements its kind evaluates its varying parameters on.
// Without socket values, the consumer's own parameters decide.
ops::Domain InputDomain(const state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node, uint16_t socket) {
    const auto &[entity, kind, parent, builtin] = plan.Nodes[node];
    const auto &at = Authored(r, plan.Nodes[node]);
    if (kind == KindOf<Op>) return IsFilterOp(r, plan, node) ? ops::Domain::Uv : StoredDomainsOf(r, plan, values, node).Input;
    if (KindSockets(kind)[socket].Selection) return ops::DomainOf(AttributeDomainOf(SelectionDomain(r, plan, values, node, socket)));
    if (kind == KindOf<StoreNamedAttribute>) return ops::DomainOf(at.get<const StoreNamedAttribute>(entity).Domain);
    if (kind == KindOf<Bake>) return ops::Domain::Texel;
    auto domain = ops::Domain::Vertex;
    VisitKind(kind, [&]<typename K> { domain = ops::DomainOf(FieldDomain(values.empty() ? at.get<const K>(entity) : Parameters<K>(r, plan, values, node))); });
    return domain;
}

struct OpRequest {
    ops::Domain Domain;
    std::vector<std::pair<uint32_t, uint16_t>> Outputs;
    std::vector<std::pair<uint16_t, uint32_t>> Inputs{}; // Each consumer input socket the request evaluates, with the index of the output it reads
    std::optional<uint16_t> Called{}; // The consumer input that calls the program at each element, for a request of its output alone
    ops::Domain Colors{ops::Domain::Corner}; // The domain the evaluated record holds its colors on
};
// Whether the consumer input can call the op program it reads at each element: a selection of a node other than an op, an input its node's position edit evaluates at each vertex, or Bake's value at each texel.
bool CallsAtElements(uint16_t kind, uint16_t socket) {
    if (kind == KindOf<Op>) return false;
    const auto input = KindSockets(kind)[socket];
    return input.Selection || input.Called || (input.Varies && kind == KindOf<Bake>);
}
// The op outputs the consumer's varying inputs read, grouped by the domain they evaluate on.
// A stored-field op's input reads an op output on its input domain whether or not the output varies, and a filter's input reads one that varies.
// An input that can call its program at each element has a request of its own.
// Each program reads the colors of a record that holds them on `colors`.
std::vector<OpRequest> RequestsOf(const state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node, ops::Domain colors) {
    std::vector<OpRequest> requests;
    const auto kind = plan.Nodes[node].Kind;
    const bool stored = kind == KindOf<Op>;
    for (uint16_t slot = 0; slot < InputsOf(plan, node); ++slot) {
        const auto &link = plan.Inputs[plan.InputFirst[node] + slot];
        const auto socket = SlotSocket(kind, slot);
        if (!IsInlineOp(r, plan, link.FromNode) || !NodeSockets(plan, node)[socket].Varies) continue;
        if ((!stored || IsFilterOp(r, plan, node)) && !NodeSockets(plan, link.FromNode)[link.FromSocket].Varies) continue;
        const auto domain = InputDomain(r, plan, values, node, socket);
        const std::pair output{link.FromNode, link.FromSocket};
        if (CallsAtElements(kind, socket)) {
            requests.push_back({domain, {output}, {{socket, 0u}}, socket, colors});
            continue;
        }
        auto request = std::ranges::find_if(requests, [&](const OpRequest &request) { return request.Domain == domain && !request.Called; });
        if (request == requests.end()) request = requests.insert(requests.end(), OpRequest{domain, {}, {}, {}, colors});
        auto read = std::ranges::find(request->Outputs, output);
        if (read == request->Outputs.end()) read = request->Outputs.insert(read, output);
        request->Inputs.emplace_back(socket, uint32_t(read - request->Outputs.begin()));
    }
    return requests;
}

// The name of the stored attribute a graph node reads: a stored field, or a geompropvalue op's property.
// None for a node that reads no attribute.
std::optional<std::string_view> ReadName(const mx::Library &lib, const ops::Node &node) {
    const auto *named = std::get_if<std::string>(&node.Op);
    if ((named ? std::string_view{*named} : lib.Declarations[std::get<uint32_t>(node.Op)].Node) != "geompropvalue") return {};
    const auto input = std::ranges::find(node.Inputs, "geomprop", &ops::Input::Name);
    return input == node.Inputs.end() ? std::string_view{} : std::string_view{std::get<std::string>(std::get<ops::Literal>(input->Source).Value)};
}

// The pass's program for the request's outputs, compiled when no graph of its shape was, where the filters in `unheld` hold no image.
OpProgram &ProgramOf(state::Registry &r, OpPass &pass, uint32_t shape, const CookPlan &plan, const OpLayout &layout, uint32_t consumer, const OpRequest &request, std::span<const uint32_t> unheld = {}) {
    OpProgramKey key{shape, consumer, request.Domain, request.Colors, request.Outputs, {unheld.begin(), unheld.end()}};
    if (const auto found = pass.Programs.find(key); found != pass.Programs.end()) return found->second;
    auto graph = GraphOf(r, plan, layout, request.Outputs, unheld);
    graph.Graph.Colors = request.Colors;
    for (const auto &read : graph.Reads) std::get<std::string>(std::get<ops::Literal>(graph.Graph.Nodes[read.Node].Inputs.front().Source).Value) = TemporaryName(read.From, read.Socket);
    const auto &compiled = ops::CompileShared(ops::GetDrivers(r), graph.Graph, request.Domain);
    return pass.Programs.emplace(std::move(key), OpProgram{request.Domain, std::move(graph), &compiled}).first->second;
}
// The split's graph reading its values relative to its first, and the word of its first in the procedure's block, for a split of a program whose values start at `base`.
std::pair<ops::Graph, uint32_t> Rebased(const ops::Split &split, uint32_t base) {
    auto graph = split.Graph;
    const auto first = Rebase(graph);
    return {std::move(graph), base + first};
}
// Whether op programs evaluate the node's inputs: a demanded node other than an inline op, or a demanded filter.
bool EvaluatesPrograms(const state::Registry &r, const CookPlan &plan, uint32_t n) {
    return (!IsInlineOp(r, plan, n) || IsFilterOp(r, plan, n)) && (plan.Demanded.empty() || plan.Demanded[n]);
}
// Visits each input of a demanded node other than an op that a program evaluates which reads an op output that does not vary, or that takes a single value, as (reading node, op node, output socket, taken).
// An input takes the output unless the output varies or has a type the input does not take.
// A stored-field op's and a filter's varying inputs read whole fields.
void ForEachSingle(const state::Registry &r, const CookPlan &plan, const OpLayout &layout, auto &&visit) {
    for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
        if (!EvaluatesPrograms(r, plan, n)) continue;
        const bool stored = plan.Nodes[n].Kind == KindOf<Op>;
        for (uint16_t slot = 0; slot < InputsOf(plan, n); ++slot) {
            const auto &link = plan.Inputs[plan.InputFirst[n] + slot];
            const auto socket = SlotSocket(plan.Nodes[n].Kind, slot);
            if (link.FromNode == NoNode || plan.Nodes[link.FromNode].Kind != KindOf<Op> || (stored && NodeSockets(plan, n)[socket].Varies)) continue;
            if (NodeSockets(plan, link.FromNode)[link.FromSocket].Varies) {
                if (!NodeSockets(plan, n)[socket].Varies) visit(n, link.FromNode, link.FromSocket, false);
                continue;
            }
            const auto type = OutputType(r, plan, layout, link.FromNode, link.FromSocket);
            visit(n, link.FromNode, link.FromSocket, !type || Takes(plan, n, socket, *DataTypeOf(*type)));
        }
    }
}
// The value a chain of constant and dot ops passes to the op output, which is a literal, a default or a copied value, and the chain's nodes.
// None for an output that evaluates at one point.
struct PassedValue {
    std::vector<uint32_t> Nodes;
    const OpLayout::Slot *Slot;
};
std::optional<PassedValue> Passed(const state::Registry &r, const CookPlan &plan, const OpLayout &layout, uint32_t node) {
    std::vector<uint32_t> nodes;
    for (;;) {
        const auto &op = Authored(r, plan.Nodes[node]).get<const Op>(plan.Nodes[node].Entity);
        if (!layout.Nodes[node].Declaration || (op.Node != "constant" && op.Node != "dot")) return {};
        nodes.push_back(node);
        const auto from = plan.Inputs[plan.InputFirst[node]].FromNode;
        if (from == NoNode) break;
        if (plan.Nodes[from].Kind != KindOf<Op>) return {};
        node = from;
    }
    const std::string_view input{DefinitionOf(Authored(r, plan.Nodes[node]).get<const Op>(plan.Nodes[node].Entity).Node)->Sockets.front().Name};
    const auto slot = std::ranges::find_if(layout.Slots, [&](const OpLayout::Slot &s) { return s.Node == node && s.Input == input; });
    if (slot == layout.Slots.end()) return {};
    return PassedValue{std::move(nodes), &*slot};
}
// Compiles the programs the procedure's cook can evaluate before it runs a node: the consumers', stored-field ops' and filters' inputs for records with faces, the splits of the sources they read on other domains, and the single values.
void PrepareOpPrograms(state::Registry &r, OpPass &pass, state::Entity mesh, const CookPlan &plan) {
    if (!HasOps(plan)) return;
    const auto shape = ShapeIn(pass, r, mesh, plan);
    if (!pass.Prepared.emplace(shape, plan.Demanded).second) return;
    const auto &layout = pass.Layouts[shape];
    auto &drivers = ops::GetDrivers(r);
    const auto prepare = [&](this const auto &self, const ops::Compiled &compiled) -> void {
        for (const auto &attribute : compiled.Program ? std::span{compiled.Program->Attributes} : std::span<const ops::Attribute>{}) {
            if (attribute.Source) self(ops::CompileShared(drivers, Rebased(*attribute.Source, 0u).first, attribute.Source->Domain));
        }
    };
    for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
        if (!EvaluatesPrograms(r, plan, n)) continue;
        for (const auto &request : RequestsOf(r, plan, {}, n, ops::Domain::Corner)) prepare(*ProgramOf(r, pass, shape, plan, layout, n, request).Compiled);
    }
    ForEachSingle(r, plan, layout, [&](uint32_t, uint32_t from, uint16_t socket, bool taken) {
        if (taken && !Passed(r, plan, layout, from)) ProgramOf(r, pass, shape, plan, layout, from, {ops::Domain::Point, {{from, socket}}});
    });
}
// Whether the program, or a split of a field it reads on another domain, reads texture coordinates.
bool ReadsTexcoords(ops::Drivers &drivers, const ops::Compiled &compiled) {
    if (!compiled.Program) return false;
    if (compiled.Program->Needs & OpNeedsTexcoords) return true;
    return std::ranges::any_of(compiled.Program->Attributes, [&](const ops::Attribute &attribute) { return attribute.Source && ReadsTexcoords(drivers, ops::CompileShared(drivers, Rebased(*attribute.Source, 0u).first, attribute.Source->Domain)); });
}
// The demanded occurrences among the generator nodes of the mesh's plan that a demanded node after them reads texture coordinate set 0 through, sorted.
// A Bake node reads the set, and so does a node whose inputs an op program reading texture coordinates evaluates.
std::vector<NodePath> ReadGenerators(state::Registry &r, OpPass &pass, state::Entity mesh, const CookPlan &plan, std::span<const uint32_t> generators) {
    const auto demanded = [&](uint32_t n) { return plan.Demanded.empty() || plan.Demanded[n]; };
    std::vector<int8_t> reads(plan.Nodes.size(), -1);
    const auto read = [&](uint32_t n) {
        if (reads[n] >= 0) return reads[n] != 0;
        reads[n] = plan.Nodes[n].Kind == KindOf<Bake> || (HasOps(plan) && EvaluatesPrograms(r, plan, n) && std::ranges::any_of(RequestsOf(r, plan, {}, n, ops::Domain::Corner), [&](const OpRequest &request) {
            const auto shape = ShapeIn(pass, r, mesh, plan);
            return ReadsTexcoords(ops::GetDrivers(r), *ProgramOf(r, pass, shape, plan, pass.Layouts[shape], n, request).Compiled);
        }));
        return reads[n] != 0;
    };
    std::vector<NodePath> read_generators;
    for (const auto g : generators) {
        if (!demanded(g)) continue;
        std::vector<bool> seed(plan.Nodes.size());
        seed[g] = true;
        const auto after = Downstream(plan, std::move(seed));
        if (std::ranges::any_of(plan.Order, [&](uint32_t n) { return n != g && after[n] && demanded(n) && read(n); })) read_generators.emplace_back(std::from_range, PathOf(plan, g));
    }
    std::ranges::sort(read_generators);
    return read_generators;
}
// Resets the faults of the op nodes once per pass.
void ResetFaults(state::Registry &r, OpCook &cook, const CookPlan &plan, std::span<const uint32_t> nodes) {
    if (!cook.Publish) return;
    auto &reset = cook.Pass.Reset[cook.Mesh];
    reset.resize(plan.Nodes.size());
    for (const auto n : nodes) {
        if (!reset[n]) Drop<NodeError>(r, PathOf(plan, n));
        reset[n] = true;
    }
}
// A fault of a graph node of a program.
using GraphFault = std::pair<uint32_t, NodeFault>;
// Resets the faults of the program's op nodes, then records its compile faults and the fault of each graph node in `faults`.
void ApplyFaults(state::Registry &r, OpCook &cook, const CookPlan &plan, const OpProgram &program, std::span<const GraphFault> faults) {
    if (!cook.Publish) return;
    ResetFaults(r, cook, plan, program.Graph.Nodes);
    if (!program.Compiled->Program) KeepFaults(r, plan, program.Graph, program.Compiled->Program.error());
    for (const auto [node, fault] : faults) Keep(r, PathOf(plan, program.Graph.Nodes[node]), NodeError{fault});
}
// The image an image input reads: the image a node's output holds, or the document texture the input holds, or none.
ImageValue ImageOf(const state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, const OpImage &image) {
    if (image.From == NoNode) return DocumentImage(r, image.Texture);
    const auto &value = values[plan.SocketFirst[image.From] + image.Socket];
    return value.Type == DataType::Image ? value.Image : NoImage;
}
// The texture of each image of the graph's image table, or none.
std::vector<OpImageTexture> ImageTable(const state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, const OpGraph &graph) {
    return graph.Images | std::views::transform([&](const OpImage &image) { return ImageOf(r, plan, values, image).Texture; }) | std::ranges::to<std::vector>();
}

// An op output that does not vary which nodes other than ops read, the chain of constant and dot ops that passes its value or else its program at one point, and its readers.
struct Single {
    uint32_t Node;
    uint16_t Socket;
    std::optional<PassedValue> Passed;
    OpProgram *Program;
    std::vector<uint32_t> Readers;
};
// The singles of a cook, with the socket values that hold them, and the op nodes whose outputs an input does not take.
struct SingleWork {
    const CookPlan *Plan;
    std::vector<SocketValue> *Values;
    OpCook *Cook;
    std::vector<Single> Singles;
    std::vector<uint32_t> Mismatched{};
};
// Holds each single's value and type in its cook's socket values, a passed value from the value store and the others evaluated at one point, all in one submit.
// Then faults the mismatched op nodes.
void EvaluateSingles(state::Registry &r, std::span<SingleWork> work) {
    const auto &store = r.Resources.get<const GpuBuffers>().OpValues;
    for (auto &[plan, values, cook, singles, mismatched] : work) {
        for (const auto &[node, socket, passed, program, readers] : singles) {
            if (!passed) continue;
            ResetFaults(r, *cook, *plan, passed->Nodes);
            auto &value = (*values)[plan->SocketFirst[node] + socket];
            value = {};
            value.Type = *DataTypeOf(passed->Slot->Type);
            std::ranges::copy(store.Get({cook->Values.Offset + passed->Slot->Offset, mx::Words(passed->Slot->Type)}), value.Value.begin());
        }
    }
    if (std::ranges::any_of(work, [](const SingleWork &cook) { return std::ranges::any_of(cook.Singles, &Single::Program); })) {
        auto &drivers = ops::GetDrivers(r);
        ops::LinkCompiled(r, drivers);
        mtl::ComputeChain chain{r.Resources.get<MeshStore>().BufferContext()};
        std::vector<std::vector<Range>> results;
        for (auto &[plan, values, cook, singles, mismatched] : work) {
            auto &ranges = results.emplace_back(singles.size());
            for (size_t i = 0; i < singles.size(); ++i) {
                if (!singles[i].Program) continue;
                ApplyFaults(r, *cook, *plan, *singles[i].Program, {});
                if (const auto &compiled = *singles[i].Program->Compiled; compiled.Program) ranges[i] = ops::RecordPoint(chain, drivers, compiled, ValuesOf(*cook, *singles[i].Program), ImageTable(r, *plan, *values, singles[i].Program->Graph), cook->Pass.Frame, cook->Pass.Time);
            }
        }
        chain.Submit();
        for (size_t w = 0; w < work.size(); ++w) {
            auto &[plan, values, cook, singles, mismatched] = work[w];
            for (size_t i = 0; i < singles.size(); ++i) {
                if (!singles[i].Program) continue;
                auto &value = (*values)[plan->SocketFirst[singles[i].Node] + singles[i].Socket];
                value = {};
                if (const auto &program = singles[i].Program->Compiled->Program) {
                    value.Type = *DataTypeOf(program->Outputs.front());
                    std::ranges::copy(chain.Scratch.Get(results[w][i]), value.Value.begin());
                }
            }
        }
    }
    for (auto &[plan, values, cook, singles, mismatched] : work) {
        ResetFaults(r, *cook, *plan, mismatched);
        if (cook->Publish) {
            for (const auto node : mismatched) Keep(r, PathOf(*plan, node), NodeError{NodeFault::Mismatch});
        }
    }
}
// The singles of a cook: those it evaluates as it starts, and those whose programs read an image a node's output holds, which evaluate before their first reader runs.
// Mismatched holds the op nodes whose outputs an input does not take.
struct CookSingles {
    std::vector<Single> Start, Later;
    std::vector<uint32_t> Mismatched;
};
// The singles of each op output that does not vary and that a node other than an op reads.
// A chain of constant and dot ops passes its value from the value store, and any other output evaluates at one point.
CookSingles SinglesOf(state::Registry &r, const CookPlan &plan, OpCook &cook) {
    std::vector<Single> singles;
    std::vector<uint32_t> mismatched;
    ForEachSingle(r, plan, cook.Layout, [&](uint32_t reader, uint32_t from, uint16_t socket, bool taken) {
        if (!taken) return mismatched.push_back(from);
        if (const auto held = std::ranges::find_if(singles, [&](const Single &single) { return single.Node == from && single.Socket == socket; }); held != singles.end()) return held->Readers.push_back(reader);
        auto passed = Passed(r, plan, cook.Layout, from);
        auto *program = passed ? nullptr : &ProgramOf(r, cook.Pass, cook.Shape, plan, cook.Layout, from, {ops::Domain::Point, {{from, socket}}}, cook.Unheld);
        singles.push_back({from, socket, std::move(passed), program, {reader}});
    });
    const auto reads_images = [](const Single &single) { return single.Program && std::ranges::any_of(single.Program->Graph.Images, [](const OpImage &image) { return image.From != NoNode; }); };
    const auto later = std::ranges::stable_partition(singles, std::not_fn(reads_images));
    std::vector<Single> reading{std::make_move_iterator(later.begin()), std::make_move_iterator(later.end())};
    singles.erase(later.begin(), later.end());
    return {std::move(singles), std::move(reading), std::move(mismatched)};
}
// Moves the later singles the node reads, which evaluate before it runs, into `work`.
void TakeLater(const CookPlan &plan, std::vector<SocketValue> &values, OpCook &cook, std::vector<Single> &later, uint32_t node, std::vector<SingleWork> &work) {
    const auto read = std::ranges::stable_partition(later, [&](const Single &single) { return !std::ranges::contains(single.Readers, node); });
    if (read.empty()) return;
    work.push_back({&plan, &values, &cook, {std::make_move_iterator(read.begin()), std::make_move_iterator(read.end())}});
    later.erase(read.begin(), read.end());
}

// Whether the consumer's input `socket` calls the evaluating program at each element: Bake's value, or an input that reads the program's output with its storage type, where a repeated smooth reads no positions its earlier passes moved.
bool Calls(const state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node, uint16_t socket, const OpProgram &program) {
    const auto kind = plan.Nodes[node].Kind;
    if (kind == KindOf<Bake>) return true;
    const auto &compiled = *program.Compiled->Program;
    const auto input = KindSockets(kind)[socket];
    if (ops::StorageOf(compiled.Outputs.front()) != StorageOf(DataType(std::countr_zero(input.Types)))) return false;
    return input.Selection || kind != KindOf<SmoothVertices> || !(compiled.Needs & OpNeedsPosition) || std::clamp(Parameters<SmoothVertices>(r, plan, values, node).Repeat, 1u, 1000u) == 1u;
}

// An op program evaluated at every element of its domain into the attributes it writes, with its attribute and image tables.
struct OpJob {
    const OpProgram *Program;
    std::vector<OpAttributeRef> Attributes;
    std::vector<OpImageTexture> Images;
    std::vector<NamedAttributeRef> Outputs;
};
// A stored field reduced onto another domain into an attribute.
struct ConversionJob {
    uint32_t Source, Output;
    DomainReduction Reduction;
};
// The stored-field ops' passes into the attribute `Output` on its domain, from the Bool field `Seed` on that domain.
struct CurvatureJob {
    uint32_t Output;
    CurvatureKind Kind;
};
struct BoundaryJob {
    uint32_t Output;
};
struct AdjacentJob {
    uint32_t Seed, Output;
    bool More;
    uint32_t Steps;
};
// The components of Linked, Edge Loop or Edge Ring, whose rounds read their changes.
struct ComponentsJob {
    uint32_t Seed, Output;
    FieldAdjacency Adjacency;
};
using FieldJob = std::variant<OpJob, ConversionJob, CurvatureJob, BoundaryJob, AdjacentJob, ComponentsJob>;
// The passes an evaluation records before the nodes that read their outputs, in order, and the temporary attributes they write.
// Reused holds the temporaries of the same names an earlier node of the record left, which the passes write again.
struct FieldWork {
    std::vector<FieldJob> Jobs;
    std::vector<uint32_t> Temporaries, Reused;
    std::map<std::pair<uint32_t, ops::Domain>, uint32_t> Converted; // Each stored field's conversion to a domain
    std::map<std::string, uint32_t, std::less<>> Splits; // Each split's evaluation, by its temporary's name
};
// The record's temporary attribute of the name, which the work holds as written or reused.
uint32_t TemporaryOf(MeshStore &meshes, uint32_t id, FieldWork &work, std::string_view name, AttributeDomain domain, AttributeType type, AttributeValue fallback = {}) {
    if (const auto held = meshes.FindAttribute(id, name); held != InvalidOffset) {
        work.Reused.push_back(held);
        return held;
    }
    const auto attribute = meshes.AddAttribute(id, domain, type, name, fallback).value();
    work.Temporaries.push_back(attribute);
    return attribute;
}
void Record(state::Registry &r, mtl::ComputeChain &chain, const OpCook &cook, uint32_t id, const FieldJob &job) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    std::visit(
        overloaded{
            [&](const OpJob &elements) {
                const auto &program = *elements.Program;
                ops::RecordElements(chain, ops::GetDrivers(r), meshes, id, program.Domain, *program.Compiled, ValuesOf(cook, program), elements.Attributes, elements.Images, elements.Outputs, cook.Pass.Frame, cook.Pass.Time);
            },
            [&](const ConversionJob &conversion) { RecordDomainConversion(meshes, pipelines, chain, conversion.Source, conversion.Output, conversion.Reduction); },
            [&](const CurvatureJob &curvature) { RecordCurvature(meshes, pipelines, chain, curvature.Output, curvature.Kind); },
            [&](const BoundaryJob &boundary) { RecordBoundary(meshes, pipelines, chain, boundary.Output); },
            [&](const AdjacentJob &adjacent) { RecordAdjacent(meshes, pipelines, chain, adjacent.Seed, adjacent.Output, adjacent.More, adjacent.Steps); },
            [](const ComponentsJob &) { std::unreachable(); }, // A components job records its rounds through RecordJobs
        },
        job
    );
}
// Records the jobs from `first` on in the record's lane, then calls `then`, where the jobs after a components job record once its rounds completed.
void RecordJobs(state::Registry &r, mtl::ComputeChain &chain, const OpCook &cook, uint32_t id, std::shared_ptr<const std::vector<FieldJob>> jobs, size_t first, std::function<void()> then) {
    for (auto i = first; i < jobs->size(); ++i) {
        if (const auto *components = std::get_if<ComponentsJob>(&(*jobs)[i])) {
            chain.Lane(id, [&] { RecordComponents(r.Resources.get<const MeshStore>(), GetMeshPipelines(r), chain, components->Seed, components->Output, components->Adjacency, [&r, &chain, &cook, id, jobs, i, then] { RecordJobs(r, chain, cook, id, jobs, i + 1u, then); }); });
            return;
        }
        chain.Lane(id, [&] { Record(r, chain, cook, id, (*jobs)[i]); });
    }
    then();
}

// The stored field a program on the domain reads: the field itself when the domain stores it, or one a program reads through its vertex or face when `through`, and else its conversion by its type's default reduction into a temporary on the storing domain.
uint32_t ConvertedTo(MeshStore &meshes, uint32_t id, FieldWork &work, uint32_t attribute, ops::Domain domain, bool through) {
    const auto [held, type] = [&] {
        const auto &a = meshes.GetAttribute(attribute);
        return std::pair{ops::DomainOf(a.Domain), a.Type};
    }();
    const auto stored = ops::StoringDomain(domain);
    if (held == stored || (through && ops::Through(held, domain))) return attribute;
    if (const auto found = work.Converted.find({attribute, stored}); found != work.Converted.end()) return found->second;
    const auto converted = TemporaryOf(meshes, id, work, std::format("{}a{}.{}", TemporaryPrefix, attribute, uint32_t(stored)), ops::AttributeDomainOf(stored), type);
    work.Jobs.push_back(ConversionJob{attribute, converted, ops::DefaultReduction(type, held, stored)});
    work.Converted.emplace(std::pair{attribute, stored}, converted);
    return converted;
}

// The program's outputs evaluated at every element of its domain with the attribute and image tables into the temporaries of the names.
std::vector<uint32_t> EvaluateElements(MeshStore &meshes, uint32_t id, FieldWork &work, const OpProgram &program, std::span<const std::string> names, std::vector<OpAttributeRef> table, std::vector<OpImageTexture> images) {
    const auto domain = ops::AttributeDomainOf(program.Domain);
    OpJob job{&program, std::move(table), std::move(images)};
    std::vector<uint32_t> attributes;
    for (size_t o = 0; o < names.size(); ++o) {
        const auto attribute = TemporaryOf(meshes, id, work, names[o], domain, *ops::StorageOf(program.Compiled->Program->Outputs[o]));
        attributes.push_back(attribute);
        job.Outputs.push_back(meshes.AttributeRef(attribute));
    }
    if (BlocksOf(meshes, id, ElementDomainOf(domain)).Count) work.Jobs.push_back(std::move(job));
    return attributes;
}
std::optional<std::vector<OpAttributeRef>> ResolveReads(state::Registry &, const CookPlan &, std::span<const SocketValue>, uint32_t id, OpCook &, const OpProgram &, FieldWork &);
// The record's temporary holding a split the program reads, evaluated on the domain that stores its source, or none when the split faults.
// Its name holds the split's shape and the word its values start at, so the programs of one evaluation that read one source read one temporary.
std::optional<uint32_t> EvaluateSplit(state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t id, OpCook &cook, const OpProgram &program, const ops::Split &split, FieldWork &work) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto domain = split.Domain;
    auto [graph, base] = Rebased(split, program.Graph.Base);
    const auto shape = ops::ShapeOf(graph, domain);
    auto name = std::format("{}split.{:016x}{:016x}.{}", TemporaryPrefix, shape.A, shape.B, base);
    if (const auto found = work.Splits.find(name); found != work.Splits.end()) return found->second;
    auto evaluated = cook.Pass.Splits.find({cook.Shape, name});
    if (evaluated == cook.Pass.Splits.end()) {
        OpGraph split_graph{.Nodes = split.Nodes | std::views::transform([&](uint32_t node) { return program.Graph.Nodes[node]; }) | std::ranges::to<std::vector>(), .Images = program.Graph.Images, .Base = base};
        const auto &compiled = ops::CompileShared(ops::GetDrivers(r), graph, domain);
        split_graph.Graph = std::move(graph);
        evaluated = cook.Pass.Splits.emplace(std::pair{cook.Shape, name}, OpProgram{domain, std::move(split_graph), &compiled}).first;
    }
    auto table = ResolveReads(r, plan, values, id, cook, evaluated->second, work);
    if (!table) return {};
    const auto attribute = EvaluateElements(meshes, id, work, evaluated->second, std::array{name}, std::move(*table), ImageTable(r, plan, values, evaluated->second.Graph)).front();
    work.Splits.emplace(std::move(name), attribute);
    return attribute;
}
// The attribute table of the program on record `id`, after recording the passes that store what it reads, or none when it faults.
// A stored field converts to the program's domain, a split evaluates on the domain that stores its source, and a geompropvalue read takes the record's attribute of its name and storage type.
// Records the program's faults: its compile faults, a field read of no stored field, and a geompropvalue read that finds no attribute, which reads its default.
std::optional<std::vector<OpAttributeRef>> ResolveReads(state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t id, OpCook &cook, const OpProgram &program, FieldWork &work) {
    ops::LinkCompiled(r, ops::GetDrivers(r));
    std::vector<GraphFault> faults;
    if (!program.Compiled->Program) {
        ApplyFaults(r, cook, plan, program, faults);
        return {};
    }
    auto &meshes = r.Resources.get<MeshStore>();
    const auto &lib = mx::Libraries();
    const auto &graph = program.Graph.Graph.Nodes;
    const auto &read_attributes = program.Compiled->Program->Attributes;
    const auto slot_of = [&](std::string_view name, mx::Type type) { return size_t(std::ranges::find_if(read_attributes, [&](const ops::Attribute &a) { return a.Name == name && a.Type == type; }) - read_attributes.begin()); };
    std::vector<uint32_t> attributes(read_attributes.size(), InvalidOffset);
    std::vector<bool> stored(read_attributes.size());
    bool evaluates = true;
    for (const auto &read : program.Graph.Reads) {
        const auto &reader = graph[read.Node];
        const auto name = *ReadName(lib, reader);
        const auto slot = slot_of(name, *reader.Output);
        stored[slot] = true;
        const auto attribute = values[plan.SocketFirst[read.From] + read.Socket].Attribute;
        if (attribute == InvalidOffset || meshes.GetAttribute(attribute).NameView() != name || meshes.GetAttribute(attribute).Type != ops::StorageOf(*reader.Output)) {
            faults.emplace_back(read.Node, NodeFault::Attribute);
            evaluates = false;
            continue;
        }
        attributes[slot] = ConvertedTo(meshes, id, work, attribute, program.Domain, true);
    }
    for (size_t slot = 0; slot < read_attributes.size(); ++slot) {
        if (stored[slot]) continue;
        const auto &[name, type, split] = read_attributes[slot];
        if (split) {
            const auto evaluated = EvaluateSplit(r, plan, values, id, cook, program, *split, work);
            if (evaluated) attributes[slot] = ConvertedTo(meshes, id, work, *evaluated, program.Domain, true);
            evaluates &= evaluated.has_value();
            continue;
        }
        const auto found = meshes.FindAttribute(id, name);
        if (found != InvalidOffset && meshes.GetAttribute(found).Type == ops::StorageOf(type)) {
            attributes[slot] = ConvertedTo(meshes, id, work, found, program.Domain, true);
            continue;
        }
        // A geompropvalue op's declaration has the type it reads as its output.
        for (uint32_t n = 0; n < graph.size(); ++n) {
            const auto *declaration = std::get_if<uint32_t>(&graph[n].Op);
            if (declaration && ReadName(lib, graph[n]) == name && lib.Declarations[*declaration].Outputs.front().Type == type) faults.emplace_back(n, NodeFault::NotFound);
        }
    }
    ApplyFaults(r, cook, plan, program, faults);
    if (!evaluates) return {};
    return ops::AttributeTable(meshes, program.Domain, attributes);
}

// An op program a consumer input calls at each element in place of a stored attribute, with its attribute and image tables.
struct OpCalled {
    uint16_t Socket;
    const OpProgram *Program;
    std::vector<OpAttributeRef> Attributes;
    std::vector<OpImageTexture> Images;
};
// Prepares each varying input of the node to read its stored field on the input's domain: an op output evaluated on it, or a stored field from another node converted to it.
// The op outputs evaluate into temporaries, except for the inputs that call their programs at each element.
// Where an op's evaluation faults, an input reads the stored field its op's first input reads from a node other than an op on its domain, or else none.
void EvaluateFieldInputs(state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node, uint32_t id, OpCook &cook, FieldWork &work, std::span<uint32_t> attributes, std::vector<OpCalled> &calls) {
    auto &meshes = r.Resources.get<MeshStore>();
    for (const auto &request : RequestsOf(r, plan, values, node, ops::ColorDomain(meshes, id))) {
        const auto *program = &ProgramOf(r, cook.Pass, cook.Shape, plan, cook.Layout, node, request, cook.Unheld);
        auto table = ResolveReads(r, plan, values, id, cook, *program, work);
        if (table && request.Called && Calls(r, plan, values, node, *request.Called, *program)) {
            calls.push_back({*request.Called, program, std::move(*table), ImageTable(r, plan, values, program->Graph)});
            continue;
        }
        const auto names = request.Outputs | std::views::transform([&](const auto &output) { return std::format("{}.{}", TemporaryName(output.first, output.second), uint32_t(request.Domain)); }) | std::ranges::to<std::vector>();
        const auto outputs = table ? std::optional{EvaluateElements(meshes, id, work, *program, names, std::move(*table), ImageTable(r, plan, values, program->Graph))} : std::nullopt;
        for (const auto [socket, output] : request.Inputs) {
            if (outputs) {
                attributes[socket] = (*outputs)[output];
                continue;
            }
            const auto from = request.Outputs[output].first;
            const auto &first = plan.Inputs[plan.InputFirst[from]];
            const bool field = InputsOf(plan, from) && first.FromNode != NoNode && !IsInlineOp(r, plan, first.FromNode) && NodeSockets(plan, first.FromNode)[first.FromSocket].Varies;
            const auto passed = field ? values[plan.SocketFirst[first.FromNode] + first.FromSocket].Attribute : InvalidOffset;
            attributes[socket] = passed == InvalidOffset ? InvalidOffset : ConvertedTo(meshes, id, work, passed, program->Domain, false);
        }
    }
    const auto kind = plan.Nodes[node].Kind;
    const auto sockets = NodeSockets(plan, node);
    for (uint16_t socket = 0; socket < (MultiInput[kind] == NoSocket ? InputsOf(plan, node) : InputCount[kind]); ++socket) {
        const auto &link = plan.Inputs[plan.InputFirst[node] + socket];
        if (link.FromNode == NoNode || IsInlineOp(r, plan, link.FromNode) || !NodeSockets(plan, link.FromNode)[link.FromSocket].Varies || !sockets[socket].Varies) continue;
        const auto attribute = values[plan.SocketFirst[link.FromNode] + link.FromSocket].Attribute;
        if (attribute == InvalidOffset) continue;
        attributes[socket] = ConvertedTo(meshes, id, work, attribute, InputDomain(r, plan, values, node, socket), false);
    }
}
// The reduction a Convert Domain node of the type names in its mode, or none for a mode the type does not admit.
std::optional<DomainReduction> ReductionNamed(std::string_view mode, mx::Type type) {
    const bool boolean = type == mx::Type::Boolean;
    if (boolean && mode == "any") return DomainReduction::Any;
    if (boolean && mode == "all") return DomainReduction::All;
    if (!boolean && mode == "mean") return DomainReduction::Mean;
    if (!boolean && mode == "min") return DomainReduction::Min;
    if (!boolean && mode == "max") return DomainReduction::Max;
    return {};
}
// The words of the single value a stored-field op's input takes: the value of the op output it reads, or else its slot of the procedure's value block, which holds its constant, literal or default.
// None for a value of a type the input does not take.
std::optional<AttributeValue> SingleOf(const state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, const OpCook &cook, uint32_t node, uint16_t socket) {
    const auto &link = plan.Inputs[plan.InputFirst[node] + socket];
    if (link.FromNode != NoNode) {
        const auto &source = values[plan.SocketFirst[link.FromNode] + link.FromSocket];
        return Takes(plan, node, socket, source.Type) ? std::optional{source.Value} : std::nullopt;
    }
    const std::string_view name{NodeSockets(plan, node)[socket].Name};
    const auto slot = std::ranges::find_if(cook.Layout.Slots, [&](const OpLayout::Slot &s) { return s.Node == node && s.Input == name && s.Kind != OpSlotKind::Read; });
    AttributeValue words{};
    std::ranges::copy(r.Resources.get<const GpuBuffers>().OpValues.Get({cook.Values.Offset + slot->Offset, mx::Words(slot->Type)}), words.begin());
    return words;
}
// Prepares a stored-field op's pass into a temporary on its output domain, which its output holds.
// Its field input reads the stored field linked into it on its input domain, or else its single value at every element there.
// Selection's unlinked input reads the node's selection.
// A stored-field op whose input type no declaration takes faults, as does one whose mode its type does not admit.
void EvaluateStored(state::Registry &r, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, OpCook &cook, FieldWork &work, std::span<const uint32_t> attributes) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto path = PathOf(plan, node);
    const auto &op = Authored(r, plan.Nodes[node]).get<const Op>(plan.Nodes[node].Entity);
    const auto &definition = *DefinitionOf(op.Node);
    auto &output = values[plan.SocketFirst[node] + definition.Inputs].Attribute;
    output = InvalidOffset;
    const auto &resolved = cook.Layout.Nodes[node];
    const auto fail = [&](NodeFault fault) {
        if (cook.Publish) Keep(r, path, NodeError{fault});
    };
    if (!resolved.Declaration) {
        // An op whose field input reads a faulting op shows that op's fault alone.
        if (!definition.Inputs || !IsInlineOp(r, plan, plan.Inputs[plan.InputFirst[node]].FromNode)) fail(NodeFault::Unresolved);
        return;
    }
    const auto declaration = *resolved.Declaration;
    const auto type = resolved.Outputs.front();
    const auto storage = *ops::StorageOf(type);
    const auto [input_domain, output_domain] = StoredDomainsOf(r, plan, values, node);
    const auto stored = [&](AttributeValue fallback = {}) { return TemporaryOf(meshes, id, work, TemporaryName(node, definition.Inputs), ops::AttributeDomainOf(output_domain), storage, fallback); };
    const auto link = definition.Inputs ? plan.Inputs[plan.InputFirst[node]] : PlanLink{};
    const auto &stored_op = *StoredOpOf(op.Node);
    // An unlinked selection reads the node's selection.
    if (stored_op.Job == StoredJob::Selection && link.FromNode == NoNode && !link.Constant) {
        const auto *literal = SelectionLiteral(r, plan, node, 0u);
        // The attribute reads the masks on the host once their derive completed.
        mtl::ComputeChain chain{meshes.BufferContext()};
        const auto masks = UploadSelection(r, cook.Pass.Masks, id, literal && !literal->All ? literal : nullptr, chain);
        chain.Submit();
        output = stored({masks ? 0u : 1u});
        if (masks) WriteSelection(meshes, output, *masks);
        return;
    }
    // The field input of every job other than Curvature and Boundary, which a single value fills on the input domain.
    const bool fielded = stored_op.Job != StoredJob::Curvature && stored_op.Job != StoredJob::Boundary;
    auto field = attributes[0];
    if (fielded && link.FromNode == NoNode) {
        const auto words = SingleOf(r, plan, values, cook, node, 0u);
        field = TemporaryOf(meshes, id, work, TemporaryName(node, 0u), ops::AttributeDomainOf(input_domain), storage, *words);
    }
    if (fielded && field == InvalidOffset) return;
    switch (stored_op.Job) {
        case StoredJob::Curvature:
            output = stored();
            work.Jobs.push_back(CurvatureJob{output, TextOf(op, "kind", declaration) == "gaussian" ? CurvatureKind::Gaussian : CurvatureKind::Mean});
            return;
        case StoredJob::Boundary:
            output = stored();
            work.Jobs.push_back(BoundaryJob{output});
            return;
        case StoredJob::Selection:
            output = field;
            return;
        case StoredJob::Convert: {
            const auto reduction = ReductionNamed(TextOf(op, "mode", declaration), type);
            if (!reduction) return fail(NodeFault::Parameter);
            output = stored();
            work.Jobs.push_back(ConversionJob{field, output, *reduction});
            return;
        }
        case StoredJob::Adjacent: {
            const auto steps = SingleOf(r, plan, values, cook, node, *InputSocket(definition, "steps"));
            if (!steps) return;
            output = stored();
            work.Jobs.push_back(AdjacentJob{field, output, TextOf(op, "mode", declaration) == "more", uint32_t(std::clamp(std::bit_cast<int32_t>((*steps)[0]), 1, 64))});
            return;
        }
        case StoredJob::Components:
            output = stored();
            work.Jobs.push_back(ComponentsJob{field, output, stored_op.Adjacency});
            return;
    }
}

struct Evaluation {
    std::vector<uint32_t> Temporaries;
    std::array<uint32_t, MaxInputs> Attributes{}; // By input socket, the stored field a varying input reads on its domain, or InvalidOffset
    NodeSelection Selection;
    std::optional<GeometryMasks> In; // The selection's masks, complete in every domain, or none for every element
    bool Failed{}; // A linked selection reads no bool attribute of a selectable domain
    std::array<std::optional<PositionProgram>, MaxInputs> Programs{}; // By input socket, the program a position edit's input calls at each vertex
};

// Prepares the stored-field ops the node reads and its own varying inputs, with the inputs that call their programs at each element in `calls`.
// Each stored-field op and the node itself read the varying fields they take on their inputs' domains, evaluated or converted before them.
void PrepareFields(state::Registry &r, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, OpCook &cook, FieldWork &work, std::span<uint32_t> attributes, std::vector<OpCalled> &calls) {
    auto upstream = Upstream(plan, node, [&](uint32_t n) { return plan.Nodes[n].Kind == KindOf<Op>; });
    upstream[node] = false;
    for (const auto n : plan.Order) {
        if (!upstream[n] || IsInlineOp(r, plan, n)) continue;
        std::array<uint32_t, MaxInputs> inputs;
        inputs.fill(InvalidOffset);
        EvaluateFieldInputs(r, plan, values, n, id, cook, work, inputs, calls);
        if (cook.Publish) Drop<NodeError>(r, PathOf(plan, n));
        EvaluateStored(r, plan, values, n, id, cook, work, inputs);
    }
    EvaluateFieldInputs(r, plan, values, node, id, cook, work, attributes, calls);
}

// Records the passes that evaluate the node's varying inputs and its selection's masks on the round in the record's lane, and returns the evaluation.
// The passes that read the normals or triangulation of a record the round moved follow its refresh.
// An evaluation whose stored-field ops read the changes of their own rounds records on the step's chain and returns none, and `later` receives it once its passes completed.
std::optional<Evaluation> EvaluateInputs(state::Registry &r, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, OpCook &cook, CookRound &round, Step &step, std::function<void(Evaluation)> later) {
    auto &meshes = r.Resources.get<MeshStore>();
    Evaluation evaluation;
    evaluation.Attributes.fill(InvalidOffset);
    FieldWork work;
    std::vector<OpCalled> calls;
    PrepareFields(r, plan, values, node, id, cook, work, evaluation.Attributes, calls);
    evaluation.Temporaries = std::move(work.Temporaries);
    const auto kind = plan.Nodes[node].Kind;
    // A copy of the call that computes the selection, which the masks can record after the calls are gone.
    std::optional<OpCalled> masked;
    for (const auto &call : calls) {
        if (KindSockets(kind)[call.Socket].Selection) {
            masked = call;
            continue;
        }
        evaluation.Programs[call.Socket] = ops::PositionInput(ops::GetDrivers(r), meshes, id, *call.Program->Compiled, ValuesOf(cook, *call.Program), call.Attributes, call.Images, cook.Pass.Frame, cook.Pass.Time);
    }
    // Each domain's scatter of the selection's attribute into the masks of the record's blocks, with those blocks.
    std::vector<std::pair<AttributeMaskJob, std::vector<uint32_t>>> scatters;
    evaluation.Selection = ResolveSelection(r, plan, values, evaluation.Attributes, node);
    const auto &field = evaluation.Selection.Field;
    if (!field) {
        evaluation.In = UploadSelection(r, cook.Pass.Masks, id, evaluation.Selection.Literal, ChainOf(round));
    } else if (masked) {
        evaluation.In = ZeroedMasks(meshes, cook.Pass.Masks, id, [](uint32_t, std::span<SelectionBlock>) {});
    } else if (field->Constant) {
        if (!(field->Constant->Words[0] & 1u)) evaluation.In = ZeroedMasks(meshes, cook.Pass.Masks, id, [](uint32_t, std::span<SelectionBlock>) {});
    } else if (field->Attribute == InvalidOffset || meshes.GetAttribute(field->Attribute).Type != AttributeType::Bool || meshes.GetAttribute(field->Attribute).Domain == AttributeDomain::Corner) {
        evaluation.Failed = true;
    } else {
        const auto ref = meshes.AttributeRef(field->Attribute);
        evaluation.In = MasksOf(meshes, cook.Pass.Masks, id);
        for (uint32_t d = 0u; d < 3u; ++d) {
            const auto domain = SelectionDomains[d];
            const auto blocks = meshes.GetBlockList(id, domain).Blocks;
            if (blocks.empty()) continue;
            const auto membership = WithDomain(meshes.Arenas(), domain, [](const auto &elements) { return elements.Blocks.Buffer.Slot; });
            const bool scattered = SelectionElements[d] == evaluation.Selection.Domain;
            scatters.push_back({{.Attribute = scattered ? ref : NamedAttributeRef{}, .MembershipSlot = membership, .Masks = MaskBinding(evaluation.In->Get(SelectionElements[d])), .Words = uint32_t(blocks.size()) * MeshElementBlockWords}, {blocks.begin(), blocks.end()}});
        }
    }
    if (work.Jobs.empty() && scatters.empty() && !masked) return evaluation;
    {
        // The node holds the temporaries it reuses until the round submits after its edit.
        for (const auto attribute : work.Reused) {
            if (std::erase(round.Temporaries, attribute)) evaluation.Temporaries.push_back(attribute);
        }
        // The passes read normals and the rendered triangulation only after the round refreshed those its edits moved.
        const auto reads_surface = [](const FieldJob &job) {
            const auto *elements = std::get_if<OpJob>(&job);
            return std::holds_alternative<CurvatureJob>(job) || (elements && (elements->Program->Compiled->Program->Needs & OpNeedsNormal));
        };
        if (std::ranges::any_of(work.Jobs, reads_surface) || (masked && (masked->Program->Compiled->Program->Needs & OpNeedsNormal))) {
            RefreshMoved(r, round, id);
            if (const auto listed = std::ranges::find(round.Drawn, id, &DrawnRecord::Record); listed != round.Drawn.end() && listed->NormalsStale) {
                ChainOf(round).Lane(id, [&] { EncodeDeriveRecordNormals(r, ChainOf(round), std::span{&id, 1u}); });
                listed->NormalsStale = false;
            }
        }
    }
    // Rounds of label propagation read their changes, so their evaluation records on the step's chain.
    const bool rounds = std::ranges::any_of(work.Jobs, [](const FieldJob &job) { return std::holds_alternative<ComponentsJob>(job); });
    auto &chain = rounds ? ReadsOf(step) : ChainOf(round);
    // The scatters and the masks, which write only the record's own masks, so they record in the record's lane, which its edit's passes continue.
    const auto masks = [&r, &chain, &cook, id, scatters = std::move(scatters), masked, in = evaluation.In, domain = evaluation.Selection.Domain] {
        auto &meshes = r.Resources.get<MeshStore>();
        const auto &pipelines = GetMeshPipelines(r);
        chain.Lane(id, [&] {
            if (!scatters.empty()) {
                std::vector<AttributeMaskJob> jobs;
                std::vector<uint32_t> tiles;
                for (auto [job, blocks] : scatters) {
                    job.Blocks = chain.Upload(std::as_bytes(std::span{blocks}));
                    jobs.push_back(job);
                    tiles.push_back(TileCount(job.Words, JobTileThreads));
                }
                mtl::GatherPass(chain, pipelines[MeshPass::AttributeSelectionMasks], std::span{jobs}, tiles, JobTileThreads);
            }
            if (masked) {
                ops::RecordMasks(chain, ops::GetDrivers(r), meshes, id, masked->Program->Domain, *masked->Program->Compiled, ValuesOf(cook, *masked->Program), masked->Attributes, masked->Images, MaskBinding(in->Get(domain)), cook.Pass.Frame, cook.Pass.Time);
            }
            if (!scatters.empty() || masked) meshes.DeriveSelectionMasks(r, chain, id, *in, domain);
        });
    };
    auto jobs = std::make_shared<const std::vector<FieldJob>>(std::move(work.Jobs));
    if (!rounds) {
        RecordJobs(r, chain, cook, id, jobs, 0u, masks);
        return evaluation;
    }
    step.Waits = true;
    RecordJobs(r, chain, cook, id, jobs, 0u, [masks, later = std::move(later), evaluation = std::make_shared<Evaluation>(std::move(evaluation))] {
        masks();
        later(std::move(*evaluation));
    });
    return std::nullopt;
}

// Whether the node holds an image: a Bake node or a filter.
bool HoldsImage(const state::Registry &r, const CookPlan &plan, uint32_t node) { return plan.Nodes[node].Kind == KindOf<Bake> || IsFilterOp(r, plan, node); }
// The output socket of a Bake node or a filter that holds its image.
uint16_t ImageSocket(const state::Registry &r, const CookPlan &plan, uint32_t node) {
    const auto &record = plan.Nodes[node];
    return record.Kind == KindOf<Bake> ? InputCount[KindOf<Bake>] : DefinitionOf(Authored(r, record).get<const Op>(record.Entity).Node)->Inputs;
}
// The texture format of an image of values of the components, and the swizzles of its view whose channels past them read zero and of its opaque view.
struct BakeFormat {
    MTL::PixelFormat Format;
    MTL::TextureSwizzleChannels Sampled, Shown;
};
BakeFormat BakeFormatOf(uint32_t components) {
    using enum MTL::TextureSwizzle;
    if (components == 1u) return {MTL::PixelFormatR32Float, {TextureSwizzleRed, TextureSwizzleZero, TextureSwizzleZero, TextureSwizzleZero}, {TextureSwizzleRed, TextureSwizzleRed, TextureSwizzleRed, TextureSwizzleOne}};
    if (components == 2u) return {MTL::PixelFormatRG32Float, {TextureSwizzleRed, TextureSwizzleGreen, TextureSwizzleZero, TextureSwizzleZero}, {TextureSwizzleRed, TextureSwizzleGreen, TextureSwizzleZero, TextureSwizzleOne}};
    return {MTL::PixelFormatRGBA32Float, {TextureSwizzleRed, TextureSwizzleGreen, TextureSwizzleBlue, components == 3u ? TextureSwizzleZero : TextureSwizzleAlpha}, {TextureSwizzleRed, TextureSwizzleGreen, TextureSwizzleBlue, TextureSwizzleOne}};
}
ImageValue ValueOf(const OccurrenceImage &image) { return {ImageTextureOf(*image.Views->Sampled), image.Views->Texture.Extent.Width, image.Views->Texture.Extent.Height, image.Components}; }

// Where a node's evaluations keep its image: an evaluation at the cooks' frame that publishes, or that evaluates for display, keeps the image of each occurrence it evaluates, which later evaluations there reuse while nothing it reads changes.
// The evaluation holds every other image until it ends.
struct ImageHolder {
    std::pair<state::Entity, NodePath> Key;
    bool Keeps, Reuses;
};
ImageHolder HolderOf(const state::Registry &r, const CookPlan &plan, uint32_t node, const OpCook &cook) {
    const bool current = cook.Pass.Frame == float(r.Resources.get<const CookTime>().Frame) && cook.Pass.Records.empty();
    return {{cook.Mesh, NodePath{std::from_range, PathOf(plan, node)}}, current && cook.Keeps, current};
}
// The image the occurrence keeps from an earlier evaluation, where the evaluation reuses one.
std::optional<ImageValue> HeldImage(state::Registry &r, const ImageHolder &holder) {
    if (!holder.Reuses) return {};
    const auto &images = CurrentImages(r).Images;
    const auto held = images.find(holder.Key);
    if (held == images.end() || !held->second.Views) return {};
    return ValueOf(held->second);
}
// Keeps that the occurrence holds no image, where the evaluation keeps images.
void HoldNone(state::Registry &r, const ImageHolder &holder) {
    if (holder.Keeps) CurrentImages(r).Images[holder.Key] = {};
}
// A new Width by Height image of values of the components, which the occurrence keeps or the evaluation holds until it ends.
// `record(chain, place)` writes its texels row by row from the top to the round's texel page, which the chain then copies to the image's first level before it fills the mip chain.
// The image records on the round's chain, which submits first when the image's texels would take the round's texels and its chain's scratch past the scratch budget.
// Its texels start on a 256-byte boundary of the round's last texel page, or on a new page at least twice as large when they do not fit there.
ImageValue StoreImage(state::Registry &r, CookRound &round, const ImageHolder &holder, uint32_t width, uint32_t height, uint32_t components, std::vector<OccurrenceImage> &transient, auto &&record) {
    if (holder.Keeps) ++r.Resources.get<CookCount>().Images;
    const auto words = uint64_t(width) * height * ops::TexelStride(components);
    const auto scratch = round.Pending ? round.Pending->Scratch.Buffer.UsedSize / sizeof(float) + round.Pending->GatheredWords() : 0u;
    if (round.Texels && round.Texels + scratch + words > ScratchWordBudget) SubmitRound(r, round);
    round.Texels += words;
    constexpr uint64_t PageAlignment{64u}, FirstPageWords{1u << 16u};
    const auto capacity = [](const mtl::Buffer &page) { return uint64_t(page.Contents().size()) / sizeof(float); };
    auto offset = round.TexelPages.empty() ? 0u : (round.TexelPages.back().UsedSize / sizeof(float) + PageAlignment - 1u) / PageAlignment * PageAlignment;
    if (round.TexelPages.empty() || offset + words > capacity(round.TexelPages.back())) {
        const auto page_words = std::max({words, FirstPageWords, round.TexelPages.empty() ? 0u : 2u * capacity(round.TexelPages.back())});
        round.TexelPages.emplace_back(round.Buffers, page_words * sizeof(float), SlotType::Buffer, mtl::BufferLifetime::Workspace);
        offset = 0u;
    }
    auto &page = round.TexelPages.back();
    // The used size covers the texels, which a recycled page clears.
    page.SetUsedSize((offset + words) * sizeof(float));
    const auto &ctx = r.Resources.get<mtl::LibraryCache>().Ctx;
    const auto format = BakeFormatOf(components);
    OccurrenceImage image{.Views = std::make_shared<ImageViews>(), .Components = components};
    auto &views = *image.Views;
    views.Texture = mtl::CreateTexture2D(ctx, format.Format, {width, height}, MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite | MTL::TextureUsagePixelFormatView, mtl::MipLevelCount(width, height));
    const auto view = [&](MTL::TextureSwizzleChannels swizzle) {
        const auto levels = NS::Range::Make(0, views.Texture.MipLevels), layers = NS::Range::Make(0, 1);
        return mtl::Texture{ctx, NS::TransferPtr(views.Texture.Handle->newTextureView(format.Format, MTL::TextureType2D, levels, layers, swizzle)), views.Texture.Extent, views.Texture.MipLevels};
    };
    views.Sampled = view(format.Sampled);
    views.Shown = view(format.Shown);
    views.Levels = mtl::CreateMipChain(views.Texture);
    const auto row = width * ops::TexelStride(components) * uint32_t(sizeof(float));
    // The image's passes read and write only its texels, its texture and their own scratch, so they record as a lane of the chain with those of the other images.
    auto &chain = ChainOf(round);
    chain.Lane([&] {
        record(chain, SlotOffset{page.Slot, uint32_t(offset)});
        chain.Encode([buffer = *page, views = image.Views, offset, row, width, height](mtl::ComputeEncoder encoder) {
            encoder->copyFromBuffer(buffer, uint64_t(offset) * sizeof(float), row, uint64_t(row) * height, MTL::Size{width, height, 1u}, *views->Texture, 0u, 0u, MTL::Origin{0u, 0u, 0u});
        });
        for (uint32_t mip = 1u; mip < views.Texture.MipLevels; ++mip) {
            chain.Encode([views = image.Views, &pipeline = GetPipelines(r).MipDownsample, mip](mtl::ComputeEncoder encoder) { mtl::RecordMipLevel(encoder, pipeline, views->Levels, mip); });
        }
    });
    const auto value = ValueOf(image);
    if (holder.Keeps) CurrentImages(r).Images[holder.Key] = std::move(image);
    else transient.push_back(std::move(image));
    return value;
}

// The image of a Bake node: its value evaluated at each texel of the grid over record `id`'s layout, or none when the value's program faults.
// Faults when the record has no layout in the UV set.
std::expected<ImageValue, NodeFault> BakeNode(state::Registry &r, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, OpCook &cook, CookRound &round, std::vector<OccurrenceImage> &transient) {
    const auto holder = HolderOf(r, plan, node, cook);
    if (const auto held = HeldImage(r, holder)) return *held;
    auto &meshes = r.Resources.get<MeshStore>();
    const auto p = Parameters<Bake>(r, plan, values, node);
    if (p.UvSet >= 4u || !meshes.Get(id).CornerUvs[p.UvSet].Count) {
        HoldNone(r, holder);
        return std::unexpected{p.UvSet >= 4u ? NodeFault::Parameter : NodeFault::Attribute};
    }
    // The image's passes read the normals and triangulation of a record the round moved after its refresh.
    RefreshMoved(r, round, id);
    FieldWork work;
    std::array<uint32_t, MaxInputs> attributes;
    attributes.fill(InvalidOffset);
    std::vector<OpCalled> calls;
    PrepareFields(r, plan, values, node, id, cook, work, attributes, calls);
    // The value's program: the op program it calls, or the single value or a read of the stored field linked to it, or zero.
    const auto &link = plan.Inputs[plan.InputFirst[node] + 1u];
    const auto called = std::ranges::find(calls, uint16_t{1u}, &OpCalled::Socket);
    auto &drivers = ops::GetDrivers(r);
    const auto compile = [&](ops::Graph graph) -> const ops::Compiled * {
        const auto &compiled = ops::CompileShared(drivers, graph, ops::Domain::Texel);
        ops::LinkCompiled(r, drivers);
        return &compiled;
    };
    const auto constant = [&](mx::Type type, std::vector<uint32_t> words) { return compile({.Nodes = {{.Op = std::string{"dot"}, .Inputs = {{"in", ops::Literal{type, std::move(words)}}}}}, .Outputs = {{0u, 0u}}}); };
    const ops::Compiled *compiled{};
    SlotOffset program_values{};
    std::vector<OpAttributeRef> table;
    std::vector<OpImageTexture> images;
    if (called != calls.end()) {
        compiled = called->Program->Compiled;
        program_values = ValuesOf(cook, *called->Program);
        table = std::move(called->Attributes);
        images = std::move(called->Images);
    } else if (const auto single = OperandOf(plan, values, {}, node, 1u).Constant) {
        const auto type = ValueTypeOf(single->Type);
        compiled = constant(type, {single->Words.begin(), single->Words.begin() + mx::Words(type)});
    } else if (link.FromNode != NoNode && IsInlineOp(r, plan, link.FromNode)) {
        // The op's program faulted, or its single value has a type the image does not hold.
    } else if (attributes[1] != InvalidOffset) {
        const auto &attribute = meshes.GetAttribute(attributes[1]);
        const auto type = ValueTypeOf(DataTypeOf(attribute.Type));
        compiled = compile({.Nodes = {{.Op = std::string{"geompropvalue"}, .Output = type, .Inputs = {{"geomprop", ops::Literal{mx::Type::String, std::string{attribute.NameView()}}}, {"default", ops::Literal{type, std::vector<uint32_t>(mx::Words(type))}}}}}, .Outputs = {{0u, 0u}}});
        table = ops::AttributeTable(meshes, ops::Domain::Texel, std::array{attributes[1]});
    } else if (link.FromNode == NoNode && !link.Constant) {
        compiled = constant(mx::Type::Float, {0u});
    }
    const auto image = [&] {
        if (!compiled || !compiled->Program) {
            HoldNone(r, holder);
            return NoImage;
        }
        for (const auto &job : work.Jobs) Record(r, ChainOf(round), cook, id, job);
        return StoreImage(r, round, holder, p.Width, p.Height, mx::Words(compiled->Program->Outputs.front()), transient, [&](mtl::ComputeChain &chain, SlotOffset output) {
            ops::RecordTexels(chain, drivers, meshes, id, *compiled, program_values, table, images, {p.UvSet, p.Width, p.Height, p.Margin}, output, cook.Pass.Frame, cook.Pass.Time);
        });
    }();
    // The round's passes read the temporaries until it submits.
    round.Temporaries.append_range(work.Temporaries);
    return image;
}

// The image of a blur: its input, an image or a field of texture coordinates, evaluated on a grid over their unit square and blurred by a kernel `size` wide in texture coordinates.
// The grid holds 32 texels across the kernel, at most the image's texels or 1024 for a field along each axis, and reaches past the square by the kernel's half width, so the blur near the square's sides reads the field beyond them and an image's edge texels.
// A blur of size zero gives its input image, and a blur of a field gives none, so the programs reading it read its input.
// Faults when its input field reads more than one set of texture coordinates, and when its size exceeds 32.
std::expected<ImageValue, NodeFault> FilterNode(state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node, OpCook &cook, CookRound &round, std::vector<OccurrenceImage> &transient) {
    const auto &op = Authored(r, plan.Nodes[node]).get<const Op>(plan.Nodes[node].Entity);
    const auto &definition = *DefinitionOf(op.Node);
    const auto &resolved = cook.Layout.Nodes[node];
    const auto holder = HolderOf(r, plan, node, cook);
    const auto none = [&] {
        HoldNone(r, holder);
        return NoImage;
    };
    const auto size_words = resolved.Declaration ? SingleOf(r, plan, values, cook, node, *InputSocket(definition, "size")) : std::nullopt;
    if (!size_words) return none();
    const auto size = std::bit_cast<float>((*size_words)[0]);
    // A wider kernel spans more than 64 texels of the coarsest grid, of one texel across the square.
    if (size > 32.f) {
        none();
        return std::unexpected{NodeFault::Parameter};
    }
    const bool gaussian = TextOf(op, "filtertype", *resolved.Declaration) == "gaussian";
    auto &drivers = ops::GetDrivers(r);
    // The grid's texels along an axis of at most `limit` texels.
    const auto texels = [&](uint32_t limit) { return uint32_t(std::clamp(std::ceil(32.f / size), 1.f, float(limit))); };
    const auto blur = [&](const ops::Compiled &compiled, SlotOffset program_values, std::vector<OpImageTexture> images, uint32_t width, uint32_t height) {
        const ops::UvGrid grid{width, height, uint32_t(std::ceil(.5f * size * float(std::max(width, height)) + .5f))};
        const auto components = mx::Words(compiled.Program->Outputs.front());
        const auto stride = ops::TexelStride(components);
        return StoreImage(r, round, holder, width, height, components, transient, [&](mtl::ComputeChain &chain, SlotOffset output) {
            const SlotOffset evaluated{chain.Scratch.Buffer.Slot, TakeScratch(chain, uint64_t(ops::GridTexels(grid)) * stride).Offset};
            ops::RecordGrid(chain, drivers, compiled, program_values, images, grid, evaluated, cook.Pass.Frame, cook.Pass.Time);
            ops::RecordBlur(chain, drivers.Pipelines, grid, evaluated, stride, size, gaussian, output);
        });
    };
    const auto &link = plan.Inputs[plan.InputFirst[node]];
    if (resolved.Outputs.front() == mx::Type::Filename) {
        const auto texture = op.Images.find(definition.Sockets.front().Name);
        const auto input = ImageOf(r, plan, values, {link.FromNode, link.FromSocket, link.FromNode == NoNode && texture != op.Images.end() ? texture->second : state::Null});
        if (!input.Texture.Texture || !(size > 0.f)) {
            HoldNone(r, holder);
            return input;
        }
        if (const auto held = HeldImage(r, holder)) return *held;
        // The image read at the grid's texels, filtered over each texel's footprint, and its edge texels past its edges.
        const auto type = input.Components == 1u ? mx::Type::Float : input.Components == 2u ? mx::Type::Vector2 : input.Components == 3u ? mx::Type::Color3 : mx::Type::Color4;
        const auto &compiled = ops::CompileShared(drivers, {.Nodes = {ImageRead(type, 0u)}, .Outputs = {{0u, 0u}}}, ops::Domain::Uv);
        ops::LinkCompiled(r, drivers);
        return blur(compiled, {}, {input.Texture}, texels(input.Width), texels(input.Height));
    }
    if (!NodeSockets(plan, node)[definition.Inputs].Varies || !(size > 0.f)) return none();
    if (const auto held = HeldImage(r, holder)) return *held;
    // A field from a node other than an op is a stored field, which depends on the mesh, and the grid holds one set of coordinates.
    const auto requests = RequestsOf(r, plan, values, node, ops::Domain::Corner);
    if (requests.empty() || !ReadsOneCoordinateSet(r, plan, cook.Layout, node, cook.Unheld)) {
        none();
        return std::unexpected{NodeFault::Coordinates};
    }
    const auto &program = ProgramOf(r, cook.Pass, cook.Shape, plan, cook.Layout, node, requests.front(), cook.Unheld);
    ops::LinkCompiled(r, drivers);
    const auto &compiled = *program.Compiled;
    if (!compiled.Program) {
        none();
        // The programs reading the filter, which read its input, show the input's other faults.
        const auto lacks = [](const ops::NodeFault &fault) { return fault.Fault == ops::Fault::Unavailable || fault.Fault == ops::Fault::Ambiguous; };
        if (std::ranges::any_of(compiled.Program.error(), lacks)) return std::unexpected{NodeFault::Coordinates};
        return NoImage;
    }
    return blur(compiled, ValuesOf(cook, program), ImageTable(r, plan, values, program.Graph), texels(1024u), texels(1024u));
}

// Whether the node publishes its selection: it holds PublishSelection, or it is the source of the first linked output port of a subnet node that holds it.
bool Publishes(const state::Registry &r, const CookPlan &plan, uint32_t node) {
    const auto &record = plan.Nodes[node];
    if (Authored(r, record).all_of<PublishSelection>(record.Entity)) return true;
    if (record.Parent == NoNode) return false;
    const auto &subnet = plan.Nodes[record.Parent];
    const auto &at = Authored(r, subnet);
    if (!at.all_of<PublishSelection>(subnet.Entity)) return false;
    for (const auto &port : InterfaceOf(at, subnet.Entity)->Ports) {
        if (port.Direction == SocketDirection::Out && port.Source().FromNode != state::Null) return port.Source().FromNode == record.Entity;
    }
    return false;
}

// A display evaluation publishes nothing and keeps the images it stores, as a published one does.
enum class RunMode { Publish,
                     Transient,
                     Handle,
                     Display };

// Whether the node's inputs read the editor's selection of its input record: its own selection literal, or that of a stored-field op before it, follows the editor.
bool ReadsEditorSelection(const state::Registry &r, const CookPlan &plan, uint32_t node) {
    const auto published = [&](uint32_t n, uint16_t socket) {
        const auto *literal = SelectionLiteral(r, plan, n, socket);
        return literal && literal->Published;
    };
    const auto sockets = KindSockets(plan.Nodes[node].Kind);
    const auto socket = uint16_t(std::ranges::find(sockets, true, &Socket::Selection) - sockets.begin());
    if (socket < sockets.size() && published(node, socket)) return true;
    const auto upstream = Upstream(plan, node, [&](uint32_t n) { return plan.Nodes[n].Kind == KindOf<Op>; });
    for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
        if (upstream[n] && n != node && IsStoredOp(r, plan, n) && Authored(r, plan.Nodes[n]).get<const Op>(plan.Nodes[n].Entity).Node == "selection" && published(n, 0u)) return true;
    }
    return false;
}

// A run of a node kind after its inputs evaluated, which leaves the temporaries the edit reads in `evaluation`.
Outcome RunKind(state::Registry &r, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node, uint32_t id, Evaluation &evaluation, const Masks &masks, RunMode mode, const std::optional<PathNodeRun> &path_run) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto kind = plan.Nodes[node].Kind;
    const auto path = PathOf(plan, node);
    const auto linked = LinkedParameters(plan, values, evaluation.Attributes, node, evaluation.Programs);
    if (mode == RunMode::Handle) return id;
    if (!path_run || path_run->Reuse != TopologyReuse::Restored) PublishInput(r, id, masks);
    Outcome edited = id;
    VisitKind(kind, [&]<typename K> {
        auto p = Parameters<K>(r, plan, values, node);
        if constexpr (std::same_as<K, Extrude>) {
            const auto &input = plan.Inputs[plan.InputFirst[node] + ParameterSocket<&Extrude::Offset>];
            if (input.FromNode != NoNode || input.Constant) p.AlongNormals = false;
            edited = Stepped(p) ? RunExtrudeSteps(r, p, id, masks, linked) : RunExtrude(r, path, p, id, masks, evaluation.Selection, linked, evaluation.Temporaries, path_run);
        } else if constexpr (std::same_as<K, StoreNamedAttribute>) {
            // The attribute stores the value once the round completed the passes that evaluated it, and a temporary it renames leaves the step's temporaries.
            edited = PlanAfterRound(masks, id, [&meshes, p, id, value = OperandOf(plan, values, evaluation.Attributes, node, 1u), &temporaries = masks.Step.Temporaries] { return Store(meshes, p, id, value, temporaries); });
        } else if constexpr (std::same_as<K, StoreSelection>) {
            edited = PlanAfterRound(masks, id, [&meshes, p, id, masks] { return Store(meshes, p, id, masks); });
        } else if constexpr (std::same_as<K, LevelSet>) {
            edited = Run(r, p, id, OperandOf(plan, values, evaluation.Attributes, node, 0u).Attribute, masks);
        } else if constexpr (std::same_as<K, SetPosition>) {
            auto position = OperandOf(plan, values, evaluation.Attributes, node, 1u);
            if (position.Attribute == InvalidOffset && position.Constant) {
                position.Attribute = meshes.AddAttribute(id, AttributeDomain::Vertex, StorageOf(position.Constant->Type), TemporaryName(node, 1u), position.Constant->Words).value();
                evaluation.Temporaries.push_back(position.Attribute);
            }
            edited = Run(r, p, id, position.Attribute, evaluation.Programs[1u], masks);
        } else if constexpr (requires(const K &p, const Masks &m) { Run(r, p, id, m, linked); }) {
            edited = Run(r, p, id, masks, linked);
        } else if constexpr (requires(const K &p, const Masks &m) { Run(r, p, id, m); }) {
            edited = Run(r, p, id, masks);
        } else if constexpr (requires(const K &p) { std::optional<MeshTopologyTask>{PlanTopology(r, p, id, TopologyInput{})}; }) {
            if constexpr (ElementSockets<K>.empty()) edited = Execute(r, id, masks, [&r, p, id, masks] { return std::optional<MeshTopologyTask>{PlanTopology(r, p, id, InputOf(masks))}; });
            else edited = RunElement(r, p, id, masks, linked);
        }
    });
    return edited;
}

// Whether the occurrence in the mesh's plan is the one the document's operation handle refers to.
bool HandleActive(const state::Registry &r, state::Entity mesh, std::span<const state::Entity> path) {
    const auto *active = r.try_get<const OperationHandle>(state::Document);
    const auto *object = active ? r.try_get<const Instance>(active->Object) : nullptr;
    return object && object->Entity == mesh && path.back() == active->Node && std::ranges::equal(path.first(path.size() - 1u), active->Path);
}
// Runs the node in the step and adds it to the step, which passes its outcome to `take` once the step's work completed.
// Its inputs' passes record on the round in the record's lane, then its kind runs, recording passes on the round, staging topology tasks in the step's stages and planning after the round or after reads.
// The kind runs once its inputs' passes recorded, or once they completed when they read the changes of their own rounds.
// A kept handle reads the input record once the round completed the passes before it, and the kind runs after it.
// The step holds the temporaries the node's passes and plans read.
// A graph cook's node runs without `path_run`.
void RunNode(state::Registry &r, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, OpCook &cook, RunMode mode, CookRound &round, Step &step, std::function<void(const Outcome &)> take = {}, std::optional<PathNodeRun> path_run = {}) {
    if (mode == RunMode::Publish) ++r.Resources.get<CookCount>().Nodes;
    auto &added = step.Nodes.emplace_back(StepNode{.Take = std::move(take)});
    auto *planned = &added.Planned;
    const auto run = [&r, &plan, &values, node, id, &cook, mode, &round, &step, planned, path_run](Evaluation evaluation) -> Outcome {
        if (evaluation.Failed) {
            step.Temporaries.append_range(evaluation.Temporaries);
            return std::unexpected(NodeFault::Attribute);
        }
        const Masks masks{.In = evaluation.In, .Domain = evaluation.Selection.Domain, .Publish = Publishes(r, plan, node), .Drawn = path_run ? path_run->Drawn : nullptr, .Round = round, .Step = step, .Topology = step.First, .Planned = planned};
        // A kind with a handle keeps one of its input's selection for a handle evaluation and for the occurrence the document's operation handle refers to, unless the occurrence holds one.
        const auto kind = plan.Nodes[node].Kind;
        const auto path = PathOf(plan, node);
        const bool keeps_handle = std::ranges::any_of(HandleSpecs, [&](const auto &spec) { return spec.Kind == kind; }) && (mode == RunMode::Handle || (mode != RunMode::Transient && HandleActive(r, cook.Mesh, path))) && !CookedOf<NodeHandle>(r, path);
        if (!keeps_handle) {
            const auto edited = RunKind(r, plan, values, node, id, evaluation, masks, mode, path_run);
            step.Temporaries.append_range(evaluation.Temporaries);
            return edited;
        }
        AfterRound(masks, [&r, &plan, &values, node, id, mode, path_run, masks, path, evaluation = std::move(evaluation)] mutable {
            KeepHandle(r, path, id, masks);
            const auto edited = RunKind(r, plan, values, node, id, evaluation, masks, mode, path_run);
            masks.Step.Temporaries.append_range(evaluation.Temporaries);
            if (masks.Planned) *masks.Planned = edited;
        });
        return id;
    };
    auto evaluation = EvaluateInputs(r, plan, values, node, id, cook, round, step, [run, planned](Evaluation evaluation) {
        const auto edited = run(std::move(evaluation));
        if (planned) *planned = edited;
    });
    added.Ran = evaluation ? run(std::move(*evaluation)) : Outcome{id};
}

// Whether the generator occurrence writes texture coordinate set 0, by the writers of its procedure, where a procedure without writers has no generator.
bool Writes(const TexcoordWriters *writers, std::span<const state::Entity> path) {
    return writers && (writers->All || std::ranges::binary_search(writers->Occurrences, NodePath{path.begin(), path.end()}));
}
// Whether a material a record of the mesh draws reads texture coordinate set 0, or another procedure reads the mesh, where `readers` holds each read mesh with its reader, sorted.
bool TexcoordsRead(const state::Registry &r, state::Entity mesh, std::span<const std::pair<state::Entity, state::Entity>> readers) {
    if (std::ranges::binary_search(readers, mesh, {}, &std::pair<state::Entity, state::Entity>::first)) return true;
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto *owned = r.try_get<const ProcedureOwnedMesh>(mesh);
    for (const auto drawing : {mesh, owned && r.valid(owned->MeshEntity) ? owned->MeshEntity : state::Null}) {
        const auto id = drawing == state::Null ? std::nullopt : DrawnStoreId(r, drawing);
        if (!id) continue;
        for (const auto material : meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(*id).PrimitiveMaterials)) {
            if (MaterialAttributes(r, material, 0u, false) & MeshAttributeBit_TexCoord0) return true;
        }
    }
    return false;
}
// The source of the record a generator node of the mesh's plan makes, or none for another kind.
std::optional<MeshSource> GeneratedSourceOf(const state::Registry &r, state::Entity mesh, const CookPlan &plan, std::span<const SocketValue> values, uint32_t node) {
    std::optional<MeshSource> source;
    VisitKind(plan.Nodes[node].Kind, [&]<typename K> {
        if constexpr (Generator<K>) source = primitive::CreateMesh(Parameters<K>(r, plan, values, node), Writes(r.try_get<const TexcoordWriters>(mesh), PathOf(plan, node)));
    });
    return source;
}
// Creates the records of the sources in one batch, which the round shades before its work.
std::vector<uint32_t> CreateUnshaded(state::Registry &r, CookRound &round, std::vector<MeshSource> sources) {
    if (sources.empty()) return {};
    round.Unshaded = true;
    return CreateMeshes(r, sources) | std::views::transform(&CreatedMesh::StoreId) | std::ranges::to<std::vector>();
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

// Each points record's instance transforms, read in one submit.
std::vector<std::vector<Transform>> InstanceTransforms(state::Registry &r, std::span<const uint32_t> points) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto vertices = points | std::views::transform([&](uint32_t id) { return HandleList(Mesh{meshes, id}.vertices()); }) | std::ranges::to<std::vector>();
    std::vector<RecordElements> listed;
    for (uint32_t i = 0u; i < points.size(); ++i) listed.push_back({points[i], vertices[i]});
    const auto positions = VertexPositions(meshes, GetMeshPipelines(r), listed);
    std::vector<std::vector<Transform>> transforms(points.size());
    for (uint32_t p = 0u; p < points.size(); ++p) {
        const auto rotations = meshes.AttributeView<vec4>(meshes.FindAttribute(points[p], InstanceRotation));
        const auto scales = meshes.AttributeView<vec3>(meshes.FindAttribute(points[p], InstanceScale));
        transforms[p].reserve(vertices[p].size());
        for (uint32_t i = 0u; i < vertices[p].size(); ++i) transforms[p].push_back({positions[p][i], quat{rotations[vertices[p][i]]}, scales[vertices[p][i]]});
    }
    return transforms;
}

// A points record and the instance record placed on its points.
struct Instancing {
    uint32_t Points, Instance;
};
// Each instancing's new record holding a copy of its instance per point in point order, each placed by its point's transform, or none without points, with every instancing in one batch.
std::vector<std::optional<uint32_t>> Realize(state::Registry &r, std::span<const Instancing> instancings) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto &vertices = meshes.Arenas().Vertices;
    mtl::ComputeChain chain{meshes.BufferContext()};
    std::vector<std::vector<ConcatPart>> parts;
    std::vector<InstanceTransformsJob> jobs;
    std::vector<uint32_t> tiles;
    for (const auto &[points, instance] : instancings) {
        const auto count = vertices.Count(meshes.Get(points).Vertices);
        if (!count) continue;
        const auto blocks = meshes.GetBlockList(points, MeshStore::ElementDomain::Vertex);
        const auto &job = jobs.emplace_back(InstanceTransformsJob{
            .Blocks = blocks.Gpu,
            .Transforms = {chain.Scratch.Buffer.Slot, chain.Scratch.Allocate(uint32_t(count * sizeof(Transform) / sizeof(uint32_t))).Offset},
            .BlockCount = uint32_t(blocks.Blocks.size()),
            .VertexSlot = meshes.Slots().Vertices,
            .MembershipSlot = vertices.Blocks.Buffer.Slot,
            .Rotations = meshes.AttributeRef(meshes.FindAttribute(points, InstanceRotation)),
            .Scales = meshes.AttributeRef(meshes.FindAttribute(points, InstanceScale)),
        });
        tiles.push_back(job.BlockCount);
        parts.push_back({{.StoreId = instance, .Copies = count, .Transforms = job.Transforms}});
    }
    mtl::GatherPass(chain, pipelines[MeshPass::InstanceTransforms], std::span{jobs}, tiles, MeshElementBlockSize);
    const auto placed = meshes.Concatenate(r, chain, parts);
    std::vector<MeshRetessellation> tessellations;
    for (const auto id : placed) tessellations.push_back({.StoreId = id, .ChangedVertices = {.Count = vertices.HandleBound(meshes.Get(id).Vertices)}});
    (void)EncodeRetessellation(r, chain, tessellations);
    EncodeDeriveRecordNormals(r, chain, placed);
    chain.Submit();
    std::vector<std::optional<uint32_t>> realized(instancings.size());
    for (uint32_t i = 0u, k = 0u; i < instancings.size(); ++i) {
        if (vertices.Count(meshes.Get(instancings[i].Points).Vertices)) realized[i] = placed[k++];
    }
    return realized;
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
// A record of a kept or recorded state derives its selection state when `round` submits, a clone of a pending source's record resets its edit state after, and a new empty record shades before its work.
std::optional<uint32_t> MaterializeSource(state::Registry &r, const SourceState &source, CookRound &round) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (source.StoreId == InvalidOffset) {
        round.Unshaded = true;
        return CreateMesh(r, {}).StoreId;
    }
    if (source.State == store::Hash128{}) {
        if (!meshes.TryGet(source.StoreId)) return std::nullopt;
        const auto id = CloneOnRound(r, round, source.StoreId);
        round.Reset.emplace_back(id, source.StoreId);
        return id;
    }
    const auto id = meshes.CloneStored(source.State, source.StoreId, round.Readers);
    if (!id) return std::nullopt;
    meshes.RebuildSelectionIndex(std::array{*id});
    round.Unselected.push_back(*id);
    return id;
}

void CarryMaterials(MeshStore &meshes, uint32_t from, uint32_t to) {
    if (from == InvalidOffset || from == to || !meshes.TryGet(from)) return;
    TakeMaterials(meshes, MaterialsOf(meshes, from), to);
}

// Whether the node roots a chain of a record it makes itself: a Source node or a generator with unlinked parameters.
bool MakesRoot(const CookPlan &plan, uint32_t node) {
    const auto kind = plan.Nodes[node].Kind;
    const bool rooted = kind == KindOf<Source> || GeneratorKind[kind];
    return rooted && std::ranges::all_of(NodeLinks(plan, node), [](const PlanLink &link) { return link.FromNode == NoNode; });
}
// The Bake nodes and filters whose images the display path's nodes from its node `root` on read, or none when the path does not chain from there.
// The path chains when the nodes after the root edit the record of the node before them in place, and the root's record is its own or the output of a graph cook of the nodes it reads.
// No node after the root instances, and only the main link from the node before it and field, value or image links reach one.
// A node outside the root's graph cook reads no record of a node after the root, and the images are those such a node other than a filter reads.
std::optional<std::vector<uint32_t>> ChainImages(const state::Registry &r, const CookPlan &plan, std::span<const uint32_t> path, size_t root) {
    for (const auto node : path.subspan(root)) {
        if (plan.Nodes[node].Kind == KindOf<InstanceOnPoints>) return {};
    }
    for (const auto node : path.subspan(root + 1u)) {
        if (plan.Nodes[node].Kind == KindOf<RealizeInstances>) return {};
    }
    // The root outputs geometry, which a Realize Instances node of a prototype's instances does not.
    const auto first = path[root];
    if (std::ranges::any_of(NodeSockets(plan, first), [](const Socket &socket) { return socket.Direction == SocketDirection::Out && (socket.Types & TypeBit(DataType::Instances)); })) return {};
    if (plan.Nodes[first].Kind == KindOf<RealizeInstances>) {
        const auto from = plan.Inputs[plan.InputFirst[first]].FromNode;
        if (from == NoNode || plan.Nodes[from].Kind != KindOf<InstanceOnPoints> || Authored(r, plan.Nodes[from]).get<const InstanceOnPoints>(plan.Nodes[from].Entity).Prototype != state::Null) return {};
    }
    std::vector<uint32_t> position(plan.Nodes.size(), NoNode);
    for (size_t i = root; i < path.size(); ++i) position[path[i]] = uint32_t(i - root);
    const auto cooked = Upstream(plan, path[root], [](uint32_t) { return true; });
    std::vector<uint32_t> images;
    for (uint32_t node = 0u; node < plan.Nodes.size(); ++node) {
        if (!plan.Demanded[node] || cooked[node]) continue;
        for (uint32_t input = 0u; input < InputsOf(plan, node); ++input) {
            const auto &link = plan.Inputs[plan.InputFirst[node] + input];
            const auto from = link.FromNode;
            if (from == NoNode) continue;
            const auto types = NodeSockets(plan, from)[link.FromSocket].Types;
            if ((types & TypeBit(DataType::Image)) && HoldsImage(r, plan, from) && !HoldsImage(r, plan, node)) images.push_back(from);
            if (position[node] != NoNode && position[node] != 0u && (types & (NumericTypes | TypeBit(DataType::Image)))) continue;
            const bool path_link = input == 0u && position[from] != NoNode && position[node] == position[from] + 1u;
            if ((position[node] != NoNode || (position[from] != NoNode && position[from] != 0u)) && !path_link) return {};
        }
    }
    SortUnique(images);
    return images;
}

// A display path's run from its first node to run, which edits the record in place.
// The record holds the output of the node before First, which keeps the mesh storage state as its NodeResult where KeepsInput holds and a pending result otherwise.
// The node before Resume, the first node the cook found changed, keeps the state its result leaves, so the next cook of that change restarts there.
// Every other node that runs holds a pending result, and the nodes from the first faulted one hold none, so the next cook runs them again.
// The render records repair once for the nodes before each kept state and once for the rest, with the positions of a restore's change to the record.
// Start is the record the path starts on, Replaced the record the cook replaces, and Topology the topology revisions of Start.
struct PathRun {
    state::Entity Mesh;
    const CookPlan *Plan;
    std::vector<uint32_t> Path;
    size_t First, Resume;
    uint32_t Record, Start, Replaced;
    std::array<uint32_t, 4> Topology;
    bool Rendered, Faulted{};
    bool DrewRender; // The record the mesh drew before the cook had render records
    TopologyReuse Reuse;
    bool Kept{}; // The record restored the state the node before First keeps
    // The node before First keeps the state the record holds: the record continues an earlier cook's, or the cook found the node at First changed.
    // A new record of the root keeps it only for that change, and its root holds a pending result otherwise.
    bool KeepsInput{true};
    std::vector<SocketValue> Values{};
    DrawnEdits Drawn{};
};

// Holds node `i`'s result, with the kept state `state` or pending without one.
void HoldResult(state::Registry &r, const PathRun &run, size_t i, store::Hash128 state) {
    const auto &plan = *run.Plan;
    const NodeResult result{i ? NodePath{std::from_range, PathOf(plan, run.Path[i - 1u])} : NodePath{}, state, run.Record};
    Keep(r, PathOf(plan, run.Path[i]), result);
    if (auto &pass = r.Resources.get<CookPass>(); state != store::Hash128{} && pass.Mesh == run.Mesh && !std::ranges::contains(pass.Exact, state)) pass.Exact.push_back(state);
}

// Holds the result of the node before the path's first node to run, and returns the single op values its nodes take.
SingleWork BeginPath(state::Registry &r, PathRun &run, OpCook &cook) {
    const auto &plan = *run.Plan;
    run.Values.resize(plan.SocketFirst.back());
    // A path with no node to run was restored to the state its last node keeps.
    if (run.First >= run.Path.size()) return {&plan, &run.Values, &cook, {}};
    FinishPendingSourceStates(r, run.Record);
    // A record restored from that state keeps it, except in a pass that cooks this mesh alone, which keeps the whole mesh storage as it stands for the next pass to restore exactly.
    if (run.Reuse != TopologyReuse::Restored && (!run.Kept || r.Resources.get<const CookPass>().Mesh == run.Mesh)) HoldResult(r, run, run.First - 1u, run.KeepsInput ? r.Resources.get<MeshStore>().KeepState() : store::Hash128{});
    auto [start, later, mismatched] = SinglesOf(r, plan, cook);
    // The values hold the images the path reads, so the singles that read them evaluate with the others.
    start.append_range(std::move(later));
    // The path evaluates the singles its nodes that run read, and the root's graph cook those its nodes read.
    const auto runs = std::span{run.Path}.subspan(run.First);
    std::erase_if(start, [&](const Single &single) { return std::ranges::none_of(single.Readers, [&](uint32_t reader) { return std::ranges::contains(runs, reader); }); });
    return {&plan, &run.Values, &cook, std::move(start), std::move(mismatched)};
}

// Takes the outcome of the run of the path's node `i`: its fault, or the record it leaves.
void TakeOutcome(state::Registry &r, PathRun &run, size_t i, const Outcome &edited) {
    if (!edited) {
        Keep(r, PathOf(*run.Plan, run.Path[i]), NodeError{edited.error()});
        run.Faulted = true;
    } else if (*edited != run.Record) {
        run.Drawn = {.Entity = run.Mesh};
        ReleaseReplaced(r, run.Record);
        run.Record = *edited;
        run.Rendered = false;
    } else if (run.Rendered && !run.Drawn.Recorded) {
        run.Drawn = {.Entity = run.Mesh, .Restored = std::move(run.Drawn.Restored)};
        run.Rendered = false;
    }
    // Position edit work refers to render records an unrepaired edit left stale.
    if (!run.Rendered) ReleaseMeshEditWork(r, run.Mesh);
}

// Holds the result of the path's node `i` unless the run faulted, and returns whether it keeps the state its result leaves, which KeepPathNodes keeps.
bool HoldPathNode(state::Registry &r, const PathRun &run, size_t i, bool keeps) {
    if (run.Faulted) {
        Drop<NodeResult, NodeExtrusion, NodeHandle>(r, PathOf(*run.Plan, run.Path[i]));
        return false;
    }
    if (!keeps) HoldResult(r, run, i, {});
    return keeps;
}

// A path node that keeps the state its result leaves.
struct KeptNode {
    PathRun *Run;
    size_t Index;
};

// Completes a step after its nodes ran: the round submits the passes they recorded, the plans that wait for it or for reads run, the round submits what they recorded, each round of topology stages executes for every mesh together, and each node takes its outcome.
// A round's stages asked to keep the state they leave keep it once they executed.
// The round releases the step's temporaries once it submits after the step.
void RunStep(state::Registry &r, CookRound &round, Step &step) {
    if (step.Waits || !step.First.Stages.empty() || !step.First.Later.empty()) {
        SubmitRound(r, round);
        auto &reads = ReadsOf(step);
        reads.Drain();
        if (auto repairs = std::exchange(step.Repairs, {}); !repairs.empty()) {
            RepairFaceRender(r, reads, repairs);
            reads.Submit();
        }
        SubmitRound(r, round);
        auto *stages = &step.First;
        std::unique_ptr<StageRound> held;
        while (!stages->Stages.empty() || !stages->Later.empty()) {
            for (auto &stage : stages->Stages) ExecuteStage(r, stage);
            auto next = std::make_unique<StageRound>();
            if (stages->Keeps) next->Kept = r.Resources.get<MeshStore>().KeepState();
            for (const auto &later : stages->Later) later(*next);
            reads.Drain();
            held = std::move(next);
            stages = held.get();
        }
        reads.ReleaseScratch();
    }
    step.First = {};
    step.Waits = false;
    round.Temporaries.append_range(std::exchange(step.Temporaries, {}));
    for (auto &node : step.Nodes) {
        if (node.Take) node.Take(node.Planned.value_or(node.Ran));
    }
    step.Nodes.clear();
}

// Runs node `i` of the path on the round and the step's stages, and adds it to the step, which takes its outcome once the step's work completed.
// A node that keeps the state its result leaves joins `kept`, which KeepPathNodes keeps after the step.
void RunPathNode(state::Registry &r, PathRun &run, size_t i, OpCook &cook, CookRound &round, Step &step, std::vector<KeptNode> &kept) {
    const auto &plan = *run.Plan;
    const auto node = run.Path[i];
    Drop<NodeError>(r, PathOf(plan, node));
    const bool keeps = i + 1u == run.Resume && i + 1u < run.Path.size();
    // A chained Join, Boolean or Minkowski has no second input and passes its first through.
    const auto passes_through = [&](uint32_t n) { return plan.Nodes[n].Kind == KindOf<Join> || plan.Nodes[n].Kind == KindOf<Boolean> || plan.Nodes[n].Kind == KindOf<Minkowski>; };
    if (passes_through(node)) {
        if (HoldPathNode(r, run, i, keeps)) kept.push_back({&run, i});
        return;
    }
    run.Drawn.Last = keeps || std::ranges::all_of(std::span{run.Path}.subspan(i + 1u), passes_through);
    run.Drawn.Recorded = false;
    const auto take = [&r, &run, i, keeps, &kept](const Outcome &edited) {
        TakeOutcome(r, run, i, edited);
        if (HoldPathNode(r, run, i, keeps)) kept.push_back({&run, i});
    };
    RunNode(r, plan, run.Values, node, run.Record, cook, RunMode::Publish, round, step, take, PathNodeRun{run.Rendered ? &run.Drawn : nullptr, i == run.First ? run.Reuse : TopologyReuse::None, i == run.First});
}

// Repairs the render records of each rendered run's record for its accumulated edits and refreshes its moves and restore, every record in one repair.
// Edits that retire every finest cluster leave their record's render records to a new build, which needs only the restore's positions, and their run no longer rendered.
void RepairDrawn(state::Registry &r, std::span<PathRun *const> runs) {
    auto &meshes = r.Resources.get<MeshStore>();
    std::vector<AccumulatedTopologyEdits> repaired;
    repaired.reserve(runs.size());
    std::vector<std::pair<state::Entity, const AccumulatedTriangleEdits *>> renders;
    std::vector<DrawnRepair> repairs;
    std::vector<DrawnRecord> refreshed;
    std::optional<mtl::ComputeChain> chain;
    for (auto *run : runs) {
        if (!run->Rendered) continue;
        const auto record = run->Record;
        auto &edits = run->Drawn;
        if (edits.Topology.Edited && RetiresAll(meshes, record, edits.Topology)) {
            edits = {.Entity = edits.Entity, .Restored = std::move(edits.Restored)};
            run->Rendered = false;
            continue;
        }
        refreshed.push_back({record, &edits});
        if (!edits.Topology.Edited) continue;
        if (!chain) chain.emplace(meshes.BufferContext(), TopologyScratchWords);
        auto &accumulated = repaired.emplace_back(std::exchange(edits.Topology, {}));
        const auto &triangles = accumulated.Triangles;
        if ((meshes.Get(record).RenderTopologies & 1u) != 0u || std::ranges::any_of(triangles.Emitted, [](uint32_t word) { return word != 0u; })) renders.emplace_back(edits.Entity, &triangles);
        // Only halfedges live after the last edit mark their edges, and the edges of the others retired or took a halfedge an edit emitted.
        const auto &corners = meshes.Arenas().FaceCorners;
        const auto entries = corners.Entries(meshes.Get(record).FaceCorners);
        const auto blocks = corners.Blocks.Buffer.GetSpan<MeshElementBlock>();
        auto &halfedges = accumulated.Marked[2];
        for (size_t w = 0u; w < halfedges.size(); ++w) {
            const auto local = w / MeshElementBlockWords;
            halfedges[w] &= local < entries.size() && entries[local] != InvalidOffset ? blocks[entries[local]].Live[w % MeshElementBlockWords] : 0u;
        }
        auto &repair = repairs.emplace_back(DrawnRepair{edits.Entity, record, std::move(accumulated.Blocks)});
        if (std::ranges::any_of(accumulated.Marked, [](const auto &words) { return !words.empty(); })) {
            for (uint32_t d = 0u; d < 3u; ++d) repair.Marked.push_back({SeedElementWorkWords(chain->Scratch, accumulated.Marked[d]), d});
        }
    }
    if (!renders.empty()) RepairTopologyRender(r, *chain, renders);
    if (!repairs.empty()) RepairDrawnElements(r, *chain, repairs);
    RefreshDrawn(r, refreshed, true);
}

// Keeps one state for the nodes of one path step of several meshes, after the round's work.
// A kept state holds current render records, which the render repair of each record's edits leaves, or else a new build of the records the meshes draw.
void KeepPathNodes(state::Registry &r, CookRound &round, std::span<const KeptNode> kept) {
    SubmitRound(r, round);
    RepairDrawn(r, kept | std::views::transform(&KeptNode::Run) | std::ranges::to<std::vector>());
    std::vector<state::Entity> built;
    std::vector<DrawnRecord> refreshed;
    for (const auto &[run, i] : kept) {
        if (run->Rendered || DrawnStoreId(r, run->Mesh) != run->Record) continue;
        refreshed.push_back({run->Record, &run->Drawn});
        built.push_back(run->Mesh);
    }
    RefreshDrawn(r, refreshed, false);
    if (!built.empty()) {
        SortUnique(built);
        mtl::ComputeChain chain{r.Resources.get<MeshStore>().BufferContext()};
        BuildMeshlets(r, chain, built, {});
        chain.Submit();
        for (const auto &[run, i] : kept) run->Rendered = run->Rendered || std::ranges::binary_search(built, run->Mesh);
    }
    const auto state = std::ranges::any_of(kept, [](const KeptNode &node) { return node.Run->Rendered; }) ? r.Resources.get<MeshStore>().KeepState() : store::Hash128{};
    for (const auto &[run, i] : kept) HoldResult(r, *run, i, run->Rendered ? state : store::Hash128{});
}

// Repairs the render records for each path's edits, every path in one repair, or leaves them to a new build when they are stale.
void EndPaths(state::Registry &r, std::span<PathRun *const> runs) {
    RepairDrawn(r, runs);
    std::vector<DrawnRecord> refreshed;
    for (auto *run : runs) {
        if (run->Rendered) continue;
        ReleaseMeshEditWork(r, run->Mesh);
        refreshed.push_back({run->Record, &run->Drawn});
    }
    RefreshDrawn(r, refreshed, false);
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

// A cook that changed topology, or whose record holds its selection in another domain than the edit domain, derives selection from the edit domain.
void PublishCooked(state::Registry &r, state::Entity mesh_entity, uint32_t record, const NodePath &tip, bool rendered, bool topology) {
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
    } else {
        // A new build replaces the render records an earlier node of the cook marked for repair.
        r.remove<MeshletPositionsDirty, ClusterLodDirty>(mesh_entity);
    }
    const bool other_domain = cooked.SelectionSummary.Count && meshes.GetSelectionSummary(record).Mode != r.get<const EditMode>(state::Document).Value;
    r.emplace_or_replace<MeshGeometryDirty>(mesh_entity, topology || other_domain ? EditSelectionAfter::Derive : EditSelectionAfter::Keep, ready);
}

// The held nodes are the root and each following node that holds a result on the result of the node before it.
// The cook restarts after the last held node whose output the record holds, whose state is kept, or, for a held Extrude's topology, whose extrusion state is kept, and from a new record of the root otherwise.
// The whole mesh storage restores while only this mesh's cooks wrote it since the state was kept, and the record alone restores otherwise.
// A copied procedure's nodes keep the results their originals hold, and the components they were created with leave them unchanged.
// The cook's start: the path run from the held nodes, or none when the record holds the result or the root has no mesh.
// A root's materialization and a restore of the record join `round`, which the caller submits, and a restore of the whole mesh storage submits `round` before it.
std::optional<PathRun> StartChain(state::Registry &r, state::Entity mesh_entity, const CookPlan &plan, std::span<const uint32_t> path, const std::vector<bool> &changed, CookRound &round) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto *live = r.try_get<const EvaluatedMesh>(mesh_entity);
    const auto prior = live && meshes.TryGet(live->StoreId) ? live->StoreId : InvalidOffset;
    const auto entity_at = [&](size_t i) { return PathOf(plan, path[i]); };
    const auto invalid = Downstream(plan, changed.empty() ? std::vector<bool>(plan.Nodes.size()) : changed);
    size_t held = 0u;
    while (held < path.size() && !invalid[path[held]]) {
        if (held) {
            const auto *result = CookedOf<NodeResult>(r, entity_at(held));
            if (!result || !std::ranges::equal(result->Input, entity_at(held - 1u))) break;
        }
        ++held;
    }
    // The first node the cook found changed, or the path's end.
    const auto resume = held;
    auto record = prior;
    auto replaced = prior;
    const bool drawn = prior != InvalidOffset && meshes.Get(prior).RenderTopologies != 0u;
    bool rendered = drawn;
    MeshStore::Change restored{};
    // Whether the record holds the output of the last of `count` held nodes.
    const auto holds = [&](size_t count) { return prior != InvalidOffset && count && std::ranges::equal(live->Tip, entity_at(count - 1u)); };
    auto reuse_topology = TopologyReuse::None;
    bool kept = false, keeps_input = true, restored_store = false;
    if (holds(held)) {
        if (held == path.size()) return std::nullopt;
        reuse_topology = TopologyReuse::Input;
    } else {
        bool restored_record = false;
        // A restored record keeps the materials it holds, which an assignment after the kept state set.
        const auto materials = prior != InvalidOffset ? MaterialsOf(meshes, prior) : std::vector<uint32_t>{};
        store::Hash128 topology_state{};
        if (const auto *result = held ? CookedOf<NodeResult>(r, entity_at(held - 1u)) : nullptr; result && held < path.size() && plan.Nodes[path[held]].Kind == KindOf<Extrude> && result->StoreId == prior) {
            const auto node = path[held];
            const auto entity = entity_at(held);
            const auto *cached = CookedOf<NodeResult>(r, entity);
            const auto *topology = CookedOf<NodeExtrusion>(r, entity);
            const auto &extrude = Authored(r, plan.Nodes[node]).get<const Extrude>(plan.Nodes[node].Entity);
            const bool plain = std::ranges::all_of(NodeLinks(plan, node).subspan(1u), [](const auto &input) { return input.FromNode == NoNode && !input.Constant; });
            // A plain node's selection is its literal, which reads no socket values.
            if (plain && cached && std::ranges::equal(cached->Input, entity_at(held - 1u)) && cached->StoreId == prior && topology &&
                topology->Mode == extrude.Mode && !Stepped(extrude) && topology->Selection == SelectionKey(ResolveSelection(r, plan, {}, {}, node))) topology_state = topology->State;
        }
        if (topology_state != store::Hash128{}) {
            WaitForRender(r);
            if (const auto change = meshes.RestoreRecord(topology_state, prior, round.Readers, RefreshesRestore(r, prior, mesh_entity))) {
                restored = JoinRestore(r, round, *change, mesh_entity);
                restored_record = true;
                reuse_topology = TopologyReuse::Restored;
            }
        }
        while (!restored_record && held && !holds(held)) {
            if (const auto *result = CookedOf<NodeResult>(r, entity_at(held - 1u)); result && result->StoreId == prior && result->State != store::Hash128{}) {
                // The whole mesh storage restores after the round's other work.
                const bool exact = std::ranges::contains(r.Resources.get<const CookPass>().Exact, result->State);
                if (exact) SubmitRound(r, round);
                WaitForRender(r);
                if (exact && meshes.RestoreKeptState(result->State)) {
                    restored = ReconcileRestoredStore(r, mesh_entity);
                    restored_record = restored_store = true;
                } else if (const auto change = meshes.RestoreRecord(result->State, prior, round.Readers, false)) {
                    // The record takes the state's own tessellation and render records.
                    restored = JoinRestore(r, round, *change, mesh_entity);
                    restored_record = true;
                }
            }
            if (!restored_record) --held;
        }
        if (restored_record) {
            TakeMaterials(meshes, materials, record);
            kept = reuse_topology != TopologyReuse::Restored;
            if (kept) reuse_topology = TopologyReuse::Input;
            replaced = InvalidOffset;
            rendered = meshes.Get(record).RenderTopologies != 0u;
        } else if (held) {
            reuse_topology = TopologyReuse::Input;
        } else {
            // A generator root's record creates with the other cooks' generated roots, and the record of a root that makes none of its own is the output of the root's graph cook.
            const auto root = plan.Nodes[path.front()];
            const auto materialized = root.Kind == KindOf<Source> ? MaterializeSource(r, Authored(r, root).get<const SourceState>(root.Entity), round) :
                                                                    std::optional{InvalidOffset};
            if (!materialized) {
                Keep(r, entity_at(0u), NodeError{NodeFault::NoMesh});
                for (const auto node : path.subspan(resume)) Drop<NodeResult, NodeExtrusion, NodeHandle>(r, PathOf(plan, node));
                return std::nullopt;
            }
            record = *materialized;
            Drop<NodeError>(r, entity_at(0u));
            if (record != InvalidOffset) CarryMaterials(meshes, prior, record);
            rendered = false;
            held = 1u;
            keeps_input = resume == held && held < path.size() && invalid[path[held]];
        }
    }
    // A node whose last run read another input has no topology of this input to reuse.
    if (reuse_topology == TopologyReuse::Input && held && held < path.size()) {
        const auto *result = CookedOf<NodeResult>(r, entity_at(held));
        if (result && !std::ranges::equal(result->Input, entity_at(held - 1u))) reuse_topology = TopologyReuse::None;
    }
    // Only a restore of the whole mesh storage to a state the step kept makes the cook independent of the storage and records the cooks before it left.
    if (!restored_store) r.Resources.get<CookPass>().DependsOnEarlierCooks = true;
    return PathRun{.Mesh = mesh_entity, .Plan = &plan, .Path = {path.begin(), path.end()}, .First = held, .Resume = resume, .Record = record, .Start = record, .Replaced = replaced, .Topology = record == InvalidOffset ? std::array<uint32_t, 4>{} : TopologyRevisions(meshes, record), .Rendered = rendered, .DrewRender = drawn, .Reuse = reuse_topology, .Kept = kept, .KeepsInput = keeps_input, .Drawn = {.Entity = mesh_entity, .Restored = std::move(restored)}};
}

// Releases the record the cook replaced and publishes the cooked record.
void FinishChain(state::Registry &r, const PathRun &run) {
    auto &meshes = r.Resources.get<MeshStore>();
    if (run.Replaced != InvalidOffset && run.Replaced != run.Start && meshes.TryGet(run.Replaced)) ReleaseReplaced(r, run.Replaced);
    OwnMesh(r, run.Mesh, InvalidOffset);
    PublishInstances(r, run.Mesh, state::Null, {});
    PublishCooked(r, run.Mesh, run.Record, run.Faulted ? NodePath{} : NodePath{std::from_range, PathOf(*run.Plan, run.Path.back())}, run.Rendered, run.Record != run.Start || TopologyRevisions(meshes, run.Record) != run.Topology);
}

// A started chained cook of one mesh.
// Graph is the plan of the graph cook the chain takes values from, or null when the chain needs none.
// The chain starts from its output where Rooted holds, which displays the chain's root through the display path's nodes in Prefix, and takes the images of the nodes in Images, which the path's nodes read.
struct ChainCook {
    OpCook Ops;
    PathRun Run;
    std::unique_ptr<CookPlan> Graph;
    bool Rooted;
    std::vector<uint32_t> Prefix, Images;
};

// Builds the render records of each path's input record that lacks them while its mesh drew render records before the cook.
// Only a path that starts at the node the cook found changed builds them, since the next cook of that change restores the input state BeginPath keeps.
// A record the mesh draws then runs its path with current render records.
void BuildInputRender(state::Registry &r, std::span<ChainCook> cooks) {
    auto &meshes = r.Resources.get<MeshStore>();
    std::vector<ChainCook *> built;
    std::vector<DrawnRecord> refreshed;
    for (auto &cook : cooks) {
        auto &run = cook.Run;
        if (!run.DrewRender || run.Rendered || !run.KeepsInput || run.First != run.Resume || run.First >= run.Path.size() || run.Reuse == TopologyReuse::Restored) continue;
        refreshed.push_back({run.Record, &run.Drawn});
        built.push_back(&cook);
    }
    if (built.empty()) return;
    RefreshDrawn(r, refreshed, false);
    std::vector<MeshletBuildSource> sources;
    for (const auto *cook : built) AppendRecordSources(sources, meshes, cook->Run.Record);
    mtl::ComputeChain chain{meshes.BufferContext()};
    BuildGpuMeshlets(r, chain, sources);
    chain.Submit();
    for (auto *cook : built) {
        auto &run = cook->Run;
        run.Kept = false;
        run.Rendered = DrawnStoreId(r, run.Mesh) == run.Record;
        if (run.Rendered) ReleaseMeshEditWork(r, run.Mesh);
    }
}

// Cooks the started meshes after the work `round` holds, each with the result it cooks to alone.
// Every mesh keeps its path's input state, then the meshes run their path nodes step by step on the round, and the nodes of a step complete together.
void CookChains(state::Registry &r, CookRound &round, std::span<ChainCook> cooks) {
    // The cooks' generated roots create together, and the round shades them before the cooks run.
    std::vector<MeshSource> sources;
    std::vector<PathRun *> generated;
    for (auto &cook : cooks) {
        if (cook.Run.Record != InvalidOffset) continue;
        sources.push_back(*GeneratedSourceOf(r, cook.Run.Mesh, *cook.Run.Plan, {}, cook.Run.Path.front()));
        generated.push_back(&cook.Run);
    }
    auto &meshes = r.Resources.get<MeshStore>();
    const auto created = CreateUnshaded(r, round, std::move(sources));
    for (size_t i = 0u; i < created.size(); ++i) {
        auto &run = *generated[i];
        run.Record = run.Start = created[i];
        CarryMaterials(meshes, run.Replaced, created[i]);
        run.Topology = TopologyRevisions(meshes, created[i]);
    }
    SubmitRound(r, round);
    if (cooks.empty()) return;
    BuildInputRender(r, cooks);
    std::vector<SingleWork> singles;
    for (auto &cook : cooks) singles.push_back(BeginPath(r, cook.Run, cook.Ops));
    EvaluateSingles(r, singles);
    // The meshes run each path step together, so the nodes of one step that keep their states keep one.
    std::vector<KeptNode> kept;
    Step step{meshes.BufferContext()};
    for (size_t s = 0u;; ++s) {
        // A record without current render records refreshes a restore's positions before its node reads them.
        std::vector<DrawnRecord> restored;
        for (auto &[ops, run, graph, rooted, prefix, images] : cooks) {
            if (run.First + s < run.Path.size() && !run.Rendered) restored.push_back({run.Record, &run.Drawn});
        }
        RefreshDrawn(r, restored, false);
        bool ran = false;
        for (auto &[ops, run, graph, rooted, prefix, images] : cooks) {
            const auto i = run.First + s;
            if (i >= run.Path.size()) continue;
            ran = true;
            RunPathNode(r, run, i, ops, round, step, kept);
            SubmitPastBudget(r, round);
        }
        RunStep(r, round, step);
        // A record's lanes of one step record before those of its next.
        if (round.Pending) round.Pending->RecordLanes();
        if (!kept.empty()) KeepPathNodes(r, round, kept);
        kept.clear();
        if (!ran) break;
    }
    SubmitRound(r, round);
    EndPaths(r, cooks | std::views::transform([](ChainCook &cook) { return &cook.Run; }) | std::ranges::to<std::vector>());
    for (const auto &cook : cooks) FinishChain(r, cook.Run);
}

// A procedure's cook through its value graph, which runs the nodes of its plan's order one at a time.
// The cook owns the records in Created, and the display reads the result socket after every node, so the result's other consumers clone it.
struct GraphCook {
    state::Entity Mesh;
    const CookPlan *Plan;
    OpCook Ops;
    RunMode Mode;
    std::vector<bool> OnPath; // The display path's nodes
    uint32_t Prior;
    std::optional<uint16_t> ResultSocket;
    std::vector<uint32_t> Remaining; // By socket, its consumers yet to take it
    std::vector<SocketValue> Values;
    std::vector<Single> Later;
    std::vector<uint32_t> Created;
    std::vector<OccurrenceImage> Transient;
    bool Stopped{}; // A handle evaluation stopped after the displayed node
    bool Root{}; // The cook serves a chained cook of the mesh's display path, which takes its result and images, and publishes nothing
};

GraphCook StartGraph(state::Registry &r, state::Entity mesh_entity, const CookPlan &plan, OpCook ops, RunMode mode, std::vector<bool> on_path) {
    const auto display = plan.ResultNode;
    GraphCook cook{
        .Mesh = mesh_entity,
        .Plan = &plan,
        .Ops = std::move(ops),
        .Mode = mode,
        .OnPath = std::move(on_path),
        .Prior = r.all_of<EvaluatedMesh>(mesh_entity) ? r.get<const EvaluatedMesh>(mesh_entity).StoreId : InvalidOffset,
        .ResultSocket = display != NoNode ? ResultSocket(plan.Nodes[display].Kind) : std::optional<uint16_t>{},
        .Remaining = plan.Consumers,
        .Values = std::vector<SocketValue>(plan.SocketFirst.back()),
    };
    if (cook.ResultSocket) ++cook.Remaining[plan.SocketFirst[display] + *cook.ResultSocket];
    return cook;
}

uint32_t SlotOf(const CookPlan &plan, uint32_t node, uint32_t input) {
    const auto link = plan.Inputs[plan.InputFirst[node] + input];
    return plan.SocketFirst[link.FromNode] + link.FromSocket;
}
// The value linked into the node's input, which clones a borrowed record before editing and an owned record before additional consumers, and preserves the result-path record by cloning its producer instead.
// A node that only reads the input and outputs no geometry takes an owned record itself, whose later consumers record their edits after its reads.
// A slot past the node's input sockets is an extra link of its multi-input socket, whose value has no entry in the socket values.
// A clone copies on the round's chain, which submits before anything reads the clone on the host.
SocketValue Take(state::Registry &r, GraphCook &cook, CookRound &round, uint32_t node, uint32_t input) {
    const auto &plan = *cook.Plan;
    const auto kind = plan.Nodes[node].Kind;
    SocketValue further;
    auto &taken = input < InputCount[kind] ? cook.Values[plan.SocketFirst[node] + input] : further;
    if (plan.Inputs[plan.InputFirst[node] + input].FromNode == NoNode) return taken;
    const auto slot = SlotOf(plan, node, input);
    taken = cook.Values[slot];
    const bool last = --cook.Remaining[slot] == 0u;
    if (taken.Record == InvalidOffset || (!taken.Borrowed() && last)) return taken;
    const auto sockets = KindSockets(kind);
    const auto outputs_geometry = [&] { return std::ranges::any_of(sockets, [](const Socket &socket) { return socket.Direction == SocketDirection::Out && (socket.Types & (TypeBit(DataType::Geometry) | TypeBit(DataType::Instances))); }); };
    if (sockets[SlotSocket(kind, uint16_t(input))].Reads && (taken.Borrowed() || !outputs_geometry())) return taken;
    const auto clone = cook.Created.emplace_back(CloneOnRound(r, round, taken.Record));
    if (taken.Borrowed() || !cook.OnPath[node] || input) {
        taken.MeshEntity = state::Null;
        taken.Record = clone;
        return taken;
    }
    cook.Values[slot].Record = clone;
    return taken;
}
void Disown(GraphCook &cook, uint32_t record) { std::erase(cook.Created, record); }
// Keeps the fault of the graph cook's node where the cook publishes.
void Fail(state::Registry &r, const GraphCook &cook, uint32_t node, NodeFault fault) {
    if (cook.Mode == RunMode::Publish) Keep(r, PathOf(*cook.Plan, node), NodeError{fault});
}
void Consume(MeshStore &meshes, GraphCook &cook, const SocketValue &value) {
    if (value.Borrowed()) return;
    meshes.Release(value.Record);
    Disown(cook, value.Record);
}

// A Join node of a graph cook, whose sources join into its output with those of the other cooks' Join nodes of its step.
struct JoiningNode {
    GraphCook *Cook;
    uint32_t Node;
    std::vector<SocketValue> Sources;
};
// Joins the step's Join nodes after the round, every node in one batch, sets their outputs and releases the sources they own.
void RunJoins(state::Registry &r, CookRound &round, std::vector<JoiningNode> &joining) {
    if (joining.empty()) return;
    SubmitRound(r, round);
    auto &meshes = r.Resources.get<MeshStore>();
    const auto parts = joining | std::views::transform([](const JoiningNode &node) {
                           return node.Sources | std::views::transform([](const SocketValue &source) { return ConcatPart{source.Record}; }) | std::ranges::to<std::vector>();
                       }) |
        std::ranges::to<std::vector>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    const auto joined = meshes.Concatenate(r, chain, parts);
    chain.Submit();
    // The sources the nodes own release in one call.
    std::vector<uint32_t> released;
    for (uint32_t k = 0u; const auto &[cook, node, sources] : joining) {
        for (const auto &source : sources) {
            if (source.Borrowed()) continue;
            released.push_back(source.Record);
            Disown(*cook, source.Record);
        }
        auto &output = cook->Values[cook->Plan->SocketFirst[node] + InputCount[KindOf<Join>]];
        output.Record = cook->Created.emplace_back(joined[k++]);
    }
    meshes.Release(released);
    joining.clear();
}

// A Boolean node of a graph cook, whose operands apply in order, each with those of the other cooks' Boolean nodes at the same operand.
struct BooleanNode {
    struct Operand {
        uint32_t Record;
        mat4 ToFirst;
    };
    GraphCook *Cook;
    uint32_t Node;
    SocketValue In;
    BooleanOperation Operation;
    uint32_t Current; // The result of the operands applied so far
    std::vector<Operand> Operands{};
    bool Linked{};
    std::optional<NodeFault> Fault{};
};
// Combines the step's Boolean nodes after the round copies their operands, the k-th operand of every node in one batch, and sets their outputs.
void RunBooleans(state::Registry &r, CookRound &round, std::vector<BooleanNode> &booleans) {
    if (booleans.empty()) return;
    SubmitRound(r, round);
    auto &meshes = r.Resources.get<MeshStore>();
    const auto combine = [](BooleanNode &boolean, std::optional<uint32_t> combined) {
        if (!combined) {
            if (!boolean.Fault) boolean.Fault = NodeFault::Empty;
            return;
        }
        boolean.Cook->Created.push_back(*combined);
        boolean.Current = *combined;
    };
    for (size_t k = 0u; std::ranges::any_of(booleans, [k](const BooleanNode &boolean) { return k < boolean.Operands.size(); }); ++k) {
        std::vector<MeshTopologyTask> tasks;
        std::vector<BooleanNode *> tasked;
        for (auto &boolean : booleans) {
            if (k >= boolean.Operands.size()) continue;
            const auto [second, to_first] = boolean.Operands[k];
            if (boolean.Fault || boolean.In.Record == InvalidOffset || second == InvalidOffset) continue;
            boolean.Linked = true;
            const auto rotation = ToMat3(to_first);
            if (Determinant(rotation) <= 0.f) {
                boolean.Fault = NodeFault::Parameter;
                continue;
            }
            const Mesh first_mesh{meshes, boolean.Current}, second_mesh{meshes, second};
            if (auto task = BooleanTask(first_mesh, second_mesh, boolean.Operation, rotation, vec3(to_first[3]))) {
                tasks.push_back(std::move(*task));
                tasked.push_back(&boolean);
            } else if (first_mesh.FaceCount() && second_mesh.FaceCount() && !first_mesh.HasClosedSurface() && !second_mesh.HasClosedSurface()) {
                // Two open operands combine as regions in the first one's plane.
                const auto fill = boolean.Operation == BooleanOperation::Union ? RegionFill::Union : boolean.Operation == BooleanOperation::Difference ? RegionFill::Difference :
                                                                                                                                                      RegionFill::Intersection;
                const auto plane = *FacePlane(r, first_mesh);
                const auto loops = RegionLoops(r, std::array{RegionTask{plane, {RegionOperand{.Record = boolean.Current}, RegionOperand{.Record = second, .Rotation = rotation, .Translation = vec3(to_first[3])}}, fill}}).front();
                if (!loops) boolean.Fault = NodeFault::NotCoplanar;
                combine(boolean, loops && !loops->empty() ? std::optional{CreateShadedMesh(r, {.Data = RegionData(plane, *loops)})} : std::nullopt);
            } else {
                boolean.Fault = NodeFault::NotClosed;
            }
        }
        if (tasks.empty()) continue;
        for (uint32_t i = 0u; const auto combined : ExecuteGeometryBooleans(r, tasks)) combine(*tasked[i++], combined);
    }
    for (auto &boolean : booleans) {
        auto &output = boolean.Cook->Values[boolean.Cook->Plan->SocketFirst[boolean.Node] + InputCount[KindOf<Boolean>]];
        if (boolean.Linked && !boolean.Fault) {
            output.Record = boolean.Current;
        } else {
            if (boolean.Linked) Fail(r, *boolean.Cook, boolean.Node, *boolean.Fault);
            output = boolean.In;
        }
    }
    booleans.clear();
}

// A Minkowski node of a graph cook, whose sweep runs with those of the other cooks' Minkowski nodes of its step.
struct MinkowskiNode {
    GraphCook *Cook;
    uint32_t Node;
    SocketValue In;
    uint32_t Second;
    MinkowskiOperation Operation;
};
// Sweeps the step's Minkowski nodes after the round copies their inputs, every node in one batch, and sets their outputs.
void RunMinkowskis(state::Registry &r, CookRound &round, std::vector<MinkowskiNode> &minkowskis) {
    if (minkowskis.empty()) return;
    SubmitRound(r, round);
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto linked = [](const MinkowskiNode &node) { return node.In.Record != InvalidOffset && node.Second != InvalidOffset; };
    // Two closed operands sweep as solids and two open ones as regions in a plane.
    const auto paired = minkowskis | std::views::transform([&](const MinkowskiNode &node) {
                            if (!linked(node)) return false;
                            const Mesh first{meshes, node.In.Record}, second{meshes, node.Second};
                            return first.FaceCount() && second.FaceCount() && first.HasClosedSurface() == second.HasClosedSurface();
                        }) |
        std::ranges::to<std::vector>();
    std::vector<MinkowskiTask> tasks;
    for (uint32_t i = 0u; i < minkowskis.size(); ++i) {
        if (paired[i]) tasks.push_back({minkowskis[i].In.Record, minkowskis[i].Second, minkowskis[i].Operation});
    }
    const auto swept = MinkowskiRecords(r, tasks);
    for (uint32_t i = 0u, k = 0u; i < minkowskis.size(); ++i) {
        auto &node = minkowskis[i];
        auto &output = node.Cook->Values[node.Cook->Plan->SocketFirst[node.Node] + InputCount[KindOf<Minkowski>]];
        const auto record = paired[i] ? swept[k++] : std::nullopt;
        if (record) {
            node.Cook->Created.push_back(*record);
            output.Record = *record;
        } else {
            if (linked(node)) Fail(r, *node.Cook, node.Node, paired[i] ? NodeFault::Empty : NodeFault::NotClosed);
            output = node.In;
        }
    }
    minkowskis.clear();
}

// A generator node of a graph cook, whose record creates with those of the other cooks' generator nodes of its step.
struct GeneratedNode {
    GraphCook *Cook;
    uint32_t Node;
    MeshSource Source;
};

// A Level Set node of a graph cook, whose lattice record creates and marches with those of the other cooks' Level Set nodes of its step.
struct LevelSetNode {
    GraphCook *Cook;
    uint32_t Node;
    Lattice Grid;
};
// Creates the step's lattice records in one batch, records their fields' passes on the round, and marches every lattice in one topology call of `step`, which the lattices run in alone.
void RunLevelSets(state::Registry &r, CookRound &round, std::vector<LevelSetNode> &levelsets, Step &step) {
    if (levelsets.empty()) return;
    const auto grids = CreateUnshaded(r, round, levelsets | std::views::transform([](const LevelSetNode &node) { return MeshSource{.Data = MeshData{node.Grid.Points()}}; }) | std::ranges::to<std::vector>());
    // The fields evaluate on the shaded lattices.
    SubmitRound(r, round);
    for (uint32_t i = 0u; i < levelsets.size(); ++i) {
        auto &[cook, node, grid] = levelsets[i];
        cook->Created.push_back(grids[i]);
        RunNode(r, *cook->Plan, cook->Values, node, grids[i], cook->Ops, cook->Mode == RunMode::Publish ? RunMode::Publish : RunMode::Transient, round, step, [&r, cook, node, grid = grids[i]](const Outcome &marched) {
            if (!marched || *marched == grid) return Fail(r, *cook, node, marched ? NodeFault::Empty : marched.error());
            ReleaseTemporaries(r.Resources.get<MeshStore>(), *marched);
            cook->Values[cook->Plan->SocketFirst[node] + InputCount[KindOf<LevelSet>]].Record = cook->Created.emplace_back(*marched);
        });
        SubmitPastBudget(r, round);
    }
    RunStep(r, round, step);
    levelsets.clear();
}

// A Realize Instances node of a graph cook, whose instances realize with those of the other cooks' Realize Instances nodes of its step.
struct RealizingNode {
    GraphCook *Cook;
    uint32_t Node;
    SocketValue In;
};
// Realizes the step's Realize Instances nodes after the round, every node in one batch, and sets their outputs.
void RunRealizes(state::Registry &r, CookRound &round, std::vector<RealizingNode> &realizing) {
    if (realizing.empty()) return;
    SubmitRound(r, round);
    auto &meshes = r.Resources.get<MeshStore>();
    const auto instanced = [](const RealizingNode &node) { return node.In.Record != InvalidOffset && node.In.Instance != InvalidOffset; };
    std::vector<Instancing> instancings;
    for (const auto &node : realizing) {
        if (instanced(node)) instancings.push_back({node.In.Record, node.In.Instance});
    }
    const auto realized = Realize(r, instancings);
    for (uint32_t k = 0u; auto &realize : realizing) {
        auto &[cook, node, in] = realize;
        const auto record = instanced(realize) ? realized[k++] : std::nullopt;
        if (record) {
            Consume(meshes, *cook, in);
            // Release the instance record after its last instance-list reader is removed.
            if (cook->Remaining[SlotOf(*cook->Plan, node, 0u)] == 0u) Consume(meshes, *cook, {.MeshEntity = in.MeshEntity, .Record = in.Instance});
            cook->Created.push_back(*record);
        }
        cook->Values[cook->Plan->SocketFirst[node] + 1u].Record = record.value_or(in.Record);
    }
    realizing.clear();
}

// Creates the records of the step's generator nodes in one batch, which the round shades before its work, and sets their outputs.
void RunGenerated(state::Registry &r, CookRound &round, std::vector<GeneratedNode> &generated) {
    const auto created = CreateUnshaded(r, round, generated | std::views::transform([](GeneratedNode &node) { return std::move(node.Source); }) | std::ranges::to<std::vector>());
    for (size_t i = 0u; i < created.size(); ++i) {
        auto &[cook, node, _] = generated[i];
        cook->Values[cook->Plan->SocketFirst[node] + InputCount[cook->Plan->Nodes[node].Kind]].Record = cook->Created.emplace_back(created[i]);
    }
    generated.clear();
}

// The nodes of one step of the graph cooks that complete together after the step: the nodes that run on the round, and the nodes of each kind that makes a record from its inputs with those of the other cooks.
struct StepWork {
    Step Nodes;
    std::vector<JoiningNode> Joins;
    std::vector<BooleanNode> Booleans;
    std::vector<MinkowskiNode> Minkowskis;
    std::vector<LevelSetNode> LevelSets;
    std::vector<RealizingNode> Realizes;
    std::vector<GeneratedNode> Generated;
};
// Runs the step's nodes that make records after its nodes that run on the round completed, each kind's nodes in one batch of its own, kind by kind in the order StepWork lists them.
void RunMaking(state::Registry &r, CookRound &round, StepWork &step) {
    RunJoins(r, round, step.Joins);
    RunBooleans(r, round, step.Booleans);
    RunMinkowskis(r, round, step.Minkowskis);
    RunLevelSets(r, round, step.LevelSets, step.Nodes);
    RunRealizes(r, round, step.Realizes);
    // A record's lanes of one step record before those of its next.
    if (round.Pending) round.Pending->RecordLanes();
    RunGenerated(r, round, step.Generated);
}

// Runs the node of the graph cook on `round` and the step's stages, or adds it to the work of its step that runs with the same kind of node of the other cooks.
// Bake and filter images, clones and Source materializations record on `round`, and a generator, Join, Boolean, Minkowski, Level Set or Realize Instances node takes its inputs into the step's work, which runs after the step's nodes completed.
void RunGraphNode(state::Registry &r, GraphCook &cook, uint32_t node, CookRound &round, StepWork &step) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto &plan = *cook.Plan;
    auto &values = cook.Values;
    const bool publish = cook.Mode == RunMode::Publish;
    const auto &[entity, kind, parent, builtin] = plan.Nodes[node];
    const auto &at = Authored(r, plan.Nodes[node]);
    const auto first = plan.SocketFirst[node];
    const auto &reached = plan.Demanded;
    // A filter runs before the programs that read its image.
    if (kind == KindOf<Op> && reached[node] && IsFilterOp(r, plan, node)) {
        ResetFaults(r, cook.Ops, plan, std::array{node});
        const auto image = FilterNode(r, plan, values, node, cook.Ops, round, cook.Transient);
        const auto output = ImageSocket(r, plan, node);
        // A filter that holds no image leaves its output as it is, which is the single value of a filter of one, and the programs reading a filter of a field read its input.
        if (image && image->Texture.Texture) values[first + output] = {.Type = DataType::Image, .Image = *image};
        else if (NodeSockets(plan, node)[output].Varies) cook.Ops.Unheld.push_back(node);
        if (!image && publish) Keep(r, PathOf(plan, node), NodeError{image.error()});
    }
    if (kind == KindOf<Op>) return;
    const auto occurrence = PathOf(plan, node);
    const auto fail = [&r, &cook, node](NodeFault fault) { Fail(r, cook, node, fault); };
    if (!reached[node]) return;
    if (publish) Drop<NodeError>(r, occurrence);
    // The editor's selection of a clone is current once the round copied it.
    const auto take_input = [&] {
        const auto taken = Take(r, cook, round, node, 0u);
        if (ReadsEditorSelection(r, plan, node)) SubmitRound(r, round);
        return taken;
    };
    if (kind == KindOf<Bake>) {
        const auto in = take_input();
        auto &output = values[first + InputCount[kind]];
        output = {.Type = DataType::Image, .Image = NoImage};
        if (in.Record == InvalidOffset) return;
        if (const auto image = BakeNode(r, plan, values, node, in.Record, cook.Ops, round, cook.Transient)) output.Image = *image;
        else fail(image.error());
        return;
    }
    if (kind == KindOf<Source>) {
        // A pending state's record is still the source, which its readers borrow.
        const auto &source = at.get<const SourceState>(entity);
        if (source.State == store::Hash128{} && source.StoreId != InvalidOffset && meshes.TryGet(source.StoreId)) values[first] = {.MeshEntity = cook.Mesh, .Record = source.StoreId};
        else if (const auto materialized = MaterializeSource(r, source, round)) values[first].Record = cook.Created.emplace_back(*materialized);
        else fail(NodeFault::NoMesh);
        return;
    }
    if (kind == KindOf<ObjectInfo>) {
        // The node reads the mesh the object draws, other than this procedure's mesh.
        const auto object = ContextObject(r, cook.Mesh, at.get<const ObjectInfo>(entity).Object);
        const auto *instance = r.valid(object) ? r.try_get<const Instance>(object) : nullptr;
        const auto read_mesh = instance && instance->Entity != cook.Mesh && DrawsMesh(r, instance->Entity) ? instance->Entity : state::Null;
        const auto &records = cook.Ops.Pass.Records;
        const auto held = records.find(read_mesh);
        if (read_mesh != state::Null) values[first] = {.MeshEntity = read_mesh, .Object = object, .Record = held != records.end() ? held->second : *DrawnStoreId(r, read_mesh)};
        else fail(NodeFault::NoMesh);
        return;
    }
    if (auto source = GeneratedSourceOf(r, cook.Mesh, plan, values, node)) {
        step.Generated.push_back({&cook, node, std::move(*source)});
        return;
    }
    const auto node_mode = publish ? RunMode::Publish : RunMode::Transient;
    if (kind == KindOf<LevelSet>) {
        const auto p = Parameters<LevelSet>(r, plan, values, node);
        const auto lattice = LatticeOver(p.Min, p.Max, p.EdgeLength);
        if (!lattice || lattice->Count() > (1u << 22u)) return fail(NodeFault::Parameter);
        step.LevelSets.push_back({&cook, node, *lattice});
        return;
    }
    const auto in = take_input();
    const auto input = in.Record;
    if (cook.Mode == RunMode::Handle && node == plan.ResultNode) {
        if (input != InvalidOffset) RunNode(r, plan, values, node, input, cook.Ops, RunMode::Handle, round, step.Nodes);
        cook.Stopped = true;
        return;
    }
    values[first + InputCount[kind]].Object = in.Object;
    // A Join reads its links and joins them into a new record with the step's other joins, and a single link passes through.
    if (kind == KindOf<Join>) {
        std::vector<SocketValue> sources;
        if (input != InvalidOffset) sources.push_back(in);
        for (uint16_t slot = InputCount[kind]; slot < InputsOf(plan, node); ++slot) {
            if (const auto next = Take(r, cook, round, node, slot); next.Record != InvalidOffset) sources.push_back(next);
        }
        if (sources.size() == 1u) values[first + InputCount[kind]] = sources.front();
        if (sources.size() > 1u) {
            values[first + InputCount[kind]].Object = sources.front().Object;
            step.Joins.push_back({&cook, node, std::move(sources)});
        }
        return;
    }
    // A Boolean takes its operands, whose clones copy on the round, and combines them with the step's other Boolean nodes.
    if (kind == KindOf<Boolean>) {
        auto p = Parameters<Boolean>(r, plan, values, node);
        p.Space = ContextObject(r, cook.Mesh, p.Space);
        BooleanNode boolean{.Cook = &cook, .Node = node, .In = in, .Operation = p.Operation, .Current = input};
        BooleanPlacement placement;
        for (auto slot = MultiInput[kind]; slot < InputsOf(plan, node); slot = NextMultiSlot(kind, slot)) {
            const auto second = Take(r, cook, round, node, slot);
            // The map into Mesh 1's space must keep Mesh 2 unmirrored.
            const auto to_first = ToFirst(r, p.Space, second.Object);
            placement.Operands.push_back({second.Object, to_first});
            boolean.Operands.push_back({second.Record, to_first});
        }
        if (publish) Keep(r, occurrence, placement);
        step.Booleans.push_back(std::move(boolean));
        return;
    }
    // A Minkowski takes its second input, whose clone copies on the round, and sweeps with the step's other Minkowski nodes.
    if (kind == KindOf<Minkowski>) {
        step.Minkowskis.push_back({.Cook = &cook, .Node = node, .In = in, .Second = Take(r, cook, round, node, 1u).Record, .Operation = Parameters<Minkowski>(r, plan, values, node).Operation});
        return;
    }
    // An Instance on Points takes its instance, whose clone copies on the round.
    if (kind == KindOf<InstanceOnPoints>) {
        const auto instance = Take(r, cook, round, node, 1u);
        values[first + InputCount[kind]] = in;
        if (input == InvalidOffset) return;
        const auto prototype = ContextObject(r, cook.Mesh, at.get<const InstanceOnPoints>(entity).Prototype);
        if (prototype != state::Null && !r.all_of<ObjectKind>(prototype)) return fail(NodeFault::Parameter);
        if (instance.Record == InvalidOffset && prototype == state::Null) return fail(NodeFault::NoMesh);
        const auto place = [&output = values[first + InputCount[kind]], prototype, instance, fail](const Outcome &placed) {
            if (!placed) return fail(placed.error());
            output.Type = DataType::Instances;
            output.Prototype = prototype;
            output.MeshEntity = instance.MeshEntity;
            output.Instance = instance.Record;
        };
        return RunNode(r, plan, values, node, input, cook.Ops, node_mode, round, step.Nodes, place);
    }
    // A Realize Instances realizes with the step's other Realize Instances nodes.
    if (kind == KindOf<RealizeInstances>) {
        if (in.Prototype != state::Null) {
            fail(NodeFault::Parameter);
            values[first + 1u] = in;
            return;
        }
        step.Realizes.push_back({&cook, node, in});
        return;
    }
    // A node that edits its input in place runs on the round, and its output takes the record its outcome leaves once the step completed.
    values[first + InputCount[kind]].Record = input;
    if (input == InvalidOffset) return;
    RunNode(r, plan, values, node, input, cook.Ops, node_mode, round, step.Nodes, [&cook, output = first + InputCount[kind], input, fail](const Outcome &edited) {
        cook.Values[output].Record = edited.value_or(input);
        if (!edited) fail(edited.error());
        else if (*edited != input) cook.Created.push_back(*edited);
    });
}

// The graph cook's result, a record it owns, which clones a borrowed record on the round's chain, and the instances value it publishes.
struct GraphResult {
    uint32_t Record{InvalidOffset};
    std::optional<SocketValue> Published;
};
GraphResult ResultOf(state::Registry &r, GraphCook &cook, CookRound &round) {
    const auto &plan = *cook.Plan;
    if (cook.Mode == RunMode::Handle || !cook.ResultSocket) return {};
    const auto &value = cook.Values[plan.SocketFirst[plan.ResultNode] + *cook.ResultSocket];
    if (value.Record == InvalidOffset) return {};
    return {value.Borrowed() ? cook.Created.emplace_back(CloneOnRound(r, round, value.Record)) : value.Record, value.Type == DataType::Instances ? std::optional{value} : std::nullopt};
}
// Publishes each graph cook's result with the transforms of the instances it publishes, or returns it without publishing for an evaluation that does not publish.
// The created records that hold no result release in one call, then the records the results replace in another.
std::vector<uint32_t> FinishGraphs(state::Registry &r, std::span<GraphCook> cooks, std::vector<GraphResult> results, std::vector<std::vector<Transform>> transforms) {
    auto &meshes = r.Resources.get<MeshStore>();
    std::vector<bool> owns(cooks.size());
    std::vector<uint32_t> released;
    const auto publishes = [](const GraphCook &cook) { return cook.Mode == RunMode::Publish && !cook.Root; };
    for (uint32_t i = 0u; i < cooks.size(); ++i) {
        auto &cook = cooks[i];
        auto &[result, published] = results[i];
        // A published cook's records are new, and their identities depend on the records the cooks before it allocated.
        if (cook.Mode == RunMode::Publish) r.Resources.get<CookPass>().DependsOnEarlierCooks = true;
        if (publishes(cook)) {
            // Instances of this mesh's pending source place a clone, since this mesh draws the points.
            if (!transforms[i].empty() && published->MeshEntity == cook.Mesh) {
                published->MeshEntity = state::Null;
                published->Instance = CloneRecord(r, published->Instance);
            }
            owns[i] = !transforms[i].empty() && published->Prototype == state::Null && published->MeshEntity == state::Null;
            if (owns[i]) Disown(cook, published->Instance);
            if (result == InvalidOffset) result = CreateShadedMesh(r, {});
        }
        Disown(cook, result);
        released.append_range(cook.Created);
    }
    meshes.Release(released);
    released.clear();
    for (uint32_t i = 0u; i < cooks.size(); ++i) {
        const auto &cook = cooks[i];
        const auto result = results[i].Record;
        if (!publishes(cook)) continue;
        CarryMaterials(meshes, cook.Prior, result);
        if (cook.Prior != InvalidOffset && cook.Prior != result && meshes.TryGet(cook.Prior)) {
            FinishPendingSourceStates(r, cook.Prior);
            released.push_back(cook.Prior);
        }
    }
    meshes.Release(released);
    for (uint32_t i = 0u; i < cooks.size(); ++i) {
        const auto &cook = cooks[i];
        const auto &[result, published] = results[i];
        if (!publishes(cook)) continue;
        const auto owned = OwnMesh(r, cook.Mesh, owns[i] ? published->Instance : InvalidOffset);
        PublishInstances(r, cook.Mesh, owns[i] ? owned : published ? published->MeshEntity :
                                                                     state::Null,
                         transforms[i], published ? published->Prototype : state::Null);
        PublishCooked(r, cook.Mesh, result, {}, false, true);
    }
    return results | std::views::transform(&GraphResult::Record) | std::ranges::to<std::vector>();
}

// Cooks the graph cooks after the work `round` holds, each with the result it cooks to alone, and returns their results.
// Each cook evaluates its starting singles in one submit, then the cooks run the nodes of their plans' orders step by step, and the work that joins the round submits together.
std::vector<uint32_t> CookGraphs(state::Registry &r, CookRound &round, std::span<GraphCook> cooks) {
    std::vector<SingleWork> singles;
    for (auto &cook : cooks) {
        const auto &plan = *cook.Plan;
        auto [start, later, mismatched] = SinglesOf(r, plan, cook.Ops);
        cook.Later = std::move(later);
        singles.push_back({&plan, &cook.Values, &cook.Ops, std::move(start), std::move(mismatched)});
    }
    EvaluateSingles(r, singles);
    StepWork work{.Nodes{round.Buffers}};
    for (size_t step = 0u;; ++step) {
        // The later singles the step's nodes read evaluate in one submit, after the round wrote the images they read.
        std::vector<SingleWork> later;
        for (auto &cook : cooks) {
            if (cook.Stopped || cook.Later.empty() || step >= cook.Plan->Order.size()) continue;
            const auto &plan = *cook.Plan;
            const auto node = plan.Order[step];
            if (plan.Demanded[node] && (plan.Nodes[node].Kind != KindOf<Op> || IsFilterOp(r, plan, node))) TakeLater(plan, cook.Values, cook.Ops, cook.Later, node, later);
        }
        if (!later.empty()) {
            SubmitRound(r, round);
            EvaluateSingles(r, later);
        }
        bool ran = false;
        for (auto &cook : cooks) {
            if (cook.Stopped || step >= cook.Plan->Order.size()) continue;
            ran = true;
            RunGraphNode(r, cook, cook.Plan->Order[step], round, work);
            SubmitPastBudget(r, round);
        }
        RunStep(r, round, work.Nodes);
        RunMaking(r, round, work);
        // The next step reads the records this step created shaded and reset.
        if (round.Unshaded || !round.Reset.empty()) SubmitRound(r, round);
        if (!ran) break;
    }
    auto results = cooks | std::views::transform([&](GraphCook &cook) { return ResultOf(r, cook, round); }) | std::ranges::to<std::vector>();
    SubmitRound(r, round);
    // The published instances read their transforms in one submit, and their points records then lose every vertex in one topology call.
    std::vector<uint32_t> placing, points;
    for (uint32_t i = 0u; i < cooks.size(); ++i) {
        if (cooks[i].Mode != RunMode::Publish || !results[i].Published || results[i].Record == InvalidOffset) continue;
        placing.push_back(i);
        points.push_back(results[i].Record);
    }
    auto placed = InstanceTransforms(r, points);
    DeleteVertices(r, points);
    std::vector<std::vector<Transform>> transforms(cooks.size());
    for (uint32_t k = 0u; k < placing.size(); ++k) transforms[placing[k]] = std::move(placed[k]);
    return FinishGraphs(r, cooks, std::move(results), std::move(transforms));
}

// The chained and graph cooks of several meshes in one settle pass, each started when its mesh evaluates, with its materialization joining the round.
struct ChainBatch {
    explicit ChainBatch(mtl::BufferContext &buffers) : Round(buffers) {}
    CookRound Round;
    std::vector<ChainCook> Cooks;
    std::vector<GraphCook> Graphs;
};

// Starts the chained cook from the record its root's graph cook made, a new record the cook publishes whatever topology its path leaves.
// A root without a record leaves an empty record, which no node of the path runs on, as a graph cook leaves.
void TakeRoot(state::Registry &r, PathRun &run, uint32_t record) {
    if (record == InvalidOffset) {
        record = CreateShadedMesh(r, {});
        // The root keeps its fault and no result of an earlier input.
        Drop<NodeResult, NodeExtrusion, NodeHandle>(r, PathOf(*run.Plan, run.Path.front()));
        for (const auto node : std::span{run.Path}.subspan(1u)) Drop<NodeError, NodeResult, NodeExtrusion, NodeHandle>(r, PathOf(*run.Plan, node));
        run.First = run.Path.size();
        run.Faulted = true;
    }
    CarryMaterials(r.Resources.get<MeshStore>(), run.Replaced, record);
    run.Record = record;
    run.Start = InvalidOffset;
}

// Takes the values of the image nodes the chain's path reads from its graph cook, and lists the filters among them that held no image.
void TakeImages(const state::Registry &r, ChainCook &cook, const GraphCook &graph) {
    const auto &plan = *graph.Plan;
    auto &values = cook.Run.Values;
    values.resize(plan.SocketFirst.back());
    for (const auto node : cook.Images) {
        const auto socket = plan.SocketFirst[node] + ImageSocket(r, plan, node);
        values[socket] = graph.Values[socket];
    }
    for (const auto node : graph.Ops.Unheld) {
        if (std::ranges::contains(cook.Images, node)) cook.Ops.Unheld.push_back(node);
    }
}

// Cooks the batch's graph cooks with the graph cooks the chained cooks take values from, then the chained cooks.
void CookBatch(state::Registry &r, ChainBatch &batch) {
    const auto graphs = batch.Graphs.size();
    for (auto &cook : batch.Cooks) {
        if (!cook.Graph) continue;
        std::vector<bool> on_path(cook.Graph->Nodes.size());
        for (const auto node : cook.Prefix) on_path[node] = true;
        auto &graph = batch.Graphs.emplace_back(StartGraph(r, cook.Run.Mesh, *cook.Graph, cook.Ops, RunMode::Publish, std::move(on_path)));
        graph.Root = true;
    }
    const auto results = batch.Graphs.empty() ? std::vector<uint32_t>{} : CookGraphs(r, batch.Round, batch.Graphs);
    for (auto k = graphs; auto &cook : batch.Cooks) {
        if (!cook.Graph) continue;
        if (cook.Rooted) TakeRoot(r, cook.Run, results[k]);
        TakeImages(r, cook, batch.Graphs[k++]);
    }
    batch.Graphs.clear();
    CookChains(r, batch.Round, batch.Cooks);
    batch.Cooks.clear();
}

// The chained part of a published display path edits the mesh's record in place, from the record its root makes or a graph cook of the nodes the root reads makes.
// Other procedures evaluate into a new record, empty when no node is displayed or the displayed node has no result.
// A node the display does not demand holds no fault or placement.
// A cook joins `batch` when one is given, and a cook that reads other meshes cooks the batch first, then a graph cook of it cooks alone.
uint32_t Evaluate(state::Registry &r, state::Entity mesh_entity, const CookPlan &plan, OpPass &pass, const std::vector<bool> &changed, RunMode mode = RunMode::Publish, ChainBatch *batch = nullptr) {
    auto &meshes = r.Resources.get<MeshStore>();
    const bool publish = mode == RunMode::Publish;
    // A cook that runs nodes reads the mesh's op values.
    const auto op_cook = [&] {
        const auto *block = r.try_get<const OpValueBlock>(mesh_entity);
        const auto shape = HasOps(plan) ? ShapeIn(pass, r, mesh_entity, plan) : NoNode;
        return OpCook{pass, mesh_entity, shape, LayoutIn(pass, r, mesh_entity, plan), {r.Resources.get<const GpuBuffers>().OpValues.Buffer.Slot, block ? block->Block.Offset : 0u}, publish, publish || mode == RunMode::Display};
    };
    const auto display = plan.ResultNode;
    std::vector<bool> on_path(plan.Nodes.size());
    std::vector<uint32_t> path;
    for (auto node = display; node != NoNode && !on_path[node];) {
        on_path[node] = true;
        path.push_back(node);
        const auto kind = plan.Nodes[node].Kind;
        node = InputCount[kind] && KindSockets(kind).front().Types == TypeBit(DataType::Geometry) ? plan.Inputs[plan.InputFirst[node]].FromNode : NoNode;
    }
    std::ranges::reverse(path);
    // A published cook's display path chains from its first node it can chain from on.
    std::optional<size_t> root;
    std::vector<uint32_t> images;
    for (size_t i = 0u; batch && !root && i < path.size(); ++i) {
        if (auto read = ChainImages(r, plan, path, i)) {
            root = i;
            images = std::move(*read);
        }
    }
    const auto chain = root ? std::span{path}.subspan(*root) : std::span<const uint32_t>{};
    std::vector<bool> chained(plan.Nodes.size());
    for (const auto node : chain) chained[node] = true;
    // Only the nodes of a chained part of a display path hold results, and only the Boolean nodes a cook evaluates hold placements.
    // A cook that skips a node would leave its result or placement stale.
    for (uint32_t node = 0u; node < plan.Nodes.size(); ++node) {
        if (!publish) continue;
        const auto occurrence = PathOf(plan, node);
        if (!plan.Demanded[node]) Drop<NodeError, BooleanPlacement>(r, occurrence);
        if (!chained[node]) Drop<NodeResult, NodeExtrusion, NodeHandle>(r, occurrence);
        if (chained[node] && node != chain.front()) Drop<BooleanPlacement>(r, occurrence);
    }
    // A cook that reads other meshes runs after the batch cooked the meshes before it.
    const bool reads_objects = std::ranges::any_of(plan.Nodes, [](const NodeRecord &node) { return node.Kind == KindOf<ObjectInfo>; });
    if (root) {
        if (reads_objects) CookBatch(r, *batch);
        auto started = StartChain(r, mesh_entity, plan, chain, changed, batch->Round);
        if (!started) return InvalidOffset;
        auto ops = op_cook();
        // A root that makes no record of its own takes the output of a graph cook of the nodes it reads, which runs with the batch's graph cooks.
        // The path's nodes read the images of Bake nodes the evaluations keep, and that graph cook makes the images of the others.
        const bool rooted = started->Record == InvalidOffset && !MakesRoot(plan, chain.front());
        const bool kept = std::ranges::all_of(images, [&](uint32_t node) { return plan.Nodes[node].Kind == KindOf<Bake> && CurrentImages(r).Images.contains(HolderOf(r, plan, node, ops).Key); });
        if (kept) {
            auto &values = started->Values;
            values.resize(plan.SocketFirst.back());
            for (const auto node : images) {
                const auto &image = CurrentImages(r).Images.at(HolderOf(r, plan, node, ops).Key);
                values[plan.SocketFirst[node] + ImageSocket(r, plan, node)] = {.Type = DataType::Image, .Image = image.Views ? ValueOf(image) : NoImage};
            }
            images.clear();
        }
        auto demanded = images;
        if (rooted) demanded.push_back(chain.front());
        auto graph = demanded.empty() ? nullptr : std::make_unique<CookPlan>(plan);
        if (graph) {
            graph->ResultNode = rooted ? chain.front() : NoNode;
            Demand(*graph, demanded);
        }
        batch->Cooks.push_back({std::move(ops), std::move(*started), std::move(graph), rooted, {path.begin(), path.begin() + *root + 1}, std::move(images)});
        return InvalidOffset;
    }
    if (batch && !reads_objects) {
        batch->Graphs.push_back(StartGraph(r, mesh_entity, plan, op_cook(), mode, std::move(on_path)));
        return InvalidOffset;
    }
    if (batch) CookBatch(r, *batch);
    CookRound round{meshes.BufferContext()};
    GraphCook one{StartGraph(r, mesh_entity, plan, op_cook(), mode, std::move(on_path))};
    return CookGraphs(r, round, std::span{&one, 1u}).front();
}

// Evaluates the mesh's procedure displaying the node's occurrence in the subnet nodes of `path`.
uint32_t EvaluateTarget(state::Registry &r, state::Entity mesh, NodePath path, state::Entity node, std::optional<uint16_t> port, RunMode mode) {
    auto target = r.get<const GeometryProcedure>(mesh);
    target.DisplayNode = node;
    target.DisplayPort = port;
    target.DisplayPath = std::move(path);
    auto plan = Compile(r, mesh);
    BindDisplay(r, target, plan);
    auto pass = PassAt(r, r.get<const TimelinePlayback>(state::Document).CurrentFrame);
    LayoutOpValues(r, mesh, plan, LayoutIn(pass, r, mesh, plan));
    return Evaluate(r, mesh, plan, pass, {}, mode);
}

void QueueNodeRemovals(state::Registry &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    auto &dependents = r.Resources.emplace<PendingObjectRemovals>().Dependents;
    dependents.append_range(NodesOf(r, mesh_entity));
}
void QueueInstanceRemovals(state::Registry &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    r.Resources.emplace<PendingObjectRemovals>().Objects.append_range(r.get<const ProcedureInstances>(mesh_entity).Entities);
}
void QueueOwnedMeshRemoval(state::Registry &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    r.Resources.emplace<PendingObjectRemovals>().Dependents.push_back(r.get<const ProcedureOwnedMesh>(mesh_entity).MeshEntity);
}
} // namespace

void SetMeshSelection(state::Registry &r, state::Entity mesh_entity, const InlineSelection *selection) {
    std::optional<MaskArenas> arena;
    auto &meshes = r.Resources.get<MeshStore>();
    const auto *live = r.try_get<const EvaluatedMesh>(mesh_entity);
    if (!live || !meshes.TryGet(live->StoreId)) return;
    if (!selection || selection->All) {
        ApplyEditSelectionCommand(r, std::array{ElementRange{mesh_entity, 0u}}, selection ? selection->Domain : Element::Vertex, EditSelectionOperation::Fill);
    } else {
        mtl::ComputeChain chain{meshes.BufferContext()};
        const auto masks = *UploadSelection(r, arena, live->StoreId, selection, chain);
        chain.Submit();
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

Element NaturalDomainOf(const state::Registry &r, state::Entity node) {
    auto domain = Element::Vertex;
    VisitKind(r.get<const ProcedureNode>(node).Kind, [&]<typename K> { domain = NaturalDomain(r.get<const K>(node)); });
    return domain;
}

bool PrepareOperationHandle(state::Registry &r, state::Entity mesh_entity, std::span<const state::Entity> path, state::Entity node) {
    NodePath occurrence{path.begin(), path.end()};
    occurrence.push_back(node);
    if (CookedOf<NodeHandle>(r, occurrence)) return true;
    EvaluateTarget(r, mesh_entity, {path.begin(), path.end()}, node, std::nullopt, RunMode::Handle);
    return CookedOf<NodeHandle>(r, occurrence);
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
    state::DirtySet Textures; // A document texture changed, which op nodes holding it read.
    state::DirtySet States; // A component that refers to kept mesh storage states changed since the history last recorded the states.
};
// The components that refer to kept mesh storage states.
using StateHolders = TypeList<SourceState, NodeResult, NodeExtrusion, NodeResults, NodeExtrusions>;

uint32_t SelectionInput(state::Registry &r, state::Entity mesh_entity, std::span<const state::Entity> path, state::Entity node, uint16_t slot) {
    const auto input = ResolveSocket(r, {path.begin(), path.end()}, InputOf(r, node, slot), true);
    return input.Source.FromNode == state::Null ? InvalidOffset : EvaluateTarget(r, mesh_entity, input.Path, input.Source.FromNode, input.Source.FromSocket, RunMode::Transient);
}

state::Entity CreateNode(state::Registry &r, uint16_t kind) {
    const auto entity = r.create();
    VisitKind(kind, [&]<typename K> { r.emplace<K>(entity); });
    return entity;
}

void RootProcedure(state::Registry &r, state::Entity mesh_entity, state::Entity root, uint16_t kind) {
    r.emplace<ProcedureNode>(root, mesh_entity, state::Null, kind);
    r.emplace<NodeInputs>(root, std::vector<SocketLink>(InputCount[kind]));
    r.emplace_or_replace<GeometryProcedure>(mesh_entity, root);
    r.emplace_or_replace<NetworkNodes>(mesh_entity, root, root);
    r.emplace_or_replace<NetworkActive>(mesh_entity, root, std::vector{root});
}
} // namespace procedure

void RegisterProcedureHandlers(state::Registry &r) {
    using namespace procedure;
    r.Resources.emplace<CookCount>();
    auto &changes = r.Resources.emplace<ProcedureChanges>();
    changes.Graph.bind(r).on<InstanceOnPoints>(On::Update).on<Op>(On::Update).on<GeometryProcedure>(On::Create | On::Update | On::Destroy).on<ProcedureNode>(On::Create | On::Update).on<NodeInputs>(On::Update).on<NodeBypassed>(On::Create | On::Destroy).on<SubnetInterface>(On::Create | On::Update | On::Destroy).on<Subnet>(On::Create | On::Update | On::Destroy).on<NetworkNodes>(On::Create | On::Update);
    changes.HandleInputs.bind(r).on<InlineSelection>(On::Create | On::Update | On::Destroy);
    changes.Created.bind(r).on<ProcedureNode>(On::Create);
    changes.Objects.bind(r, state::Clearing::Explicit).on<Instance>(On::Create | On::Update | On::Destroy);
    changes.References.bind(r, state::Clearing::Explicit)
        .on<ObjectInfo>(On::Create | On::Update | On::Destroy)
        .on<procedure::Boolean>(On::Create | On::Update | On::Destroy)
        .on<Instance>(On::Create | On::Update | On::Destroy)
        .on<SceneParent>(On::Create | On::Update | On::Destroy);
    r.on_destroy<GeometryProcedure, &QueueNodeRemovals>();
    r.on_destroy<ProcedureInstances, &QueueInstanceRemovals>();
    r.on_destroy<ProcedureOwnedMesh, &QueueOwnedMeshRemoval>();
    // History restores the op value store's allocator, so a restore releases nothing.
    r.on_destroy<OpValueBlock, [](state::Registry &r, state::Entity e) {
        if (!r.Restoring) r.Resources.get<GpuBuffers>().OpValues.Release(r.get<const OpValueBlock>(e).Block);
    }>();
    r.Resources.emplace<CookPass>();
    r.Resources.emplace<CookTime>();
    changes.Textures.bind(r).on<MaterializedTexture>(On::Create | On::Update | On::Destroy);
    r.Resources.emplace<OccurrenceImages>();
    r.Resources.emplace<TimeCooks>();
    r.Resources.emplace<PendingSources>().Changed.bind(r, state::Clearing::Explicit).on<SourceState>(On::Create | On::Update);
    [&]<typename... Ts>(TypeList<Ts...>) { (changes.States.bind(r, state::Clearing::Explicit).on<Ts>(On::Create | On::Update), ...); }(StateHolders{});
    auto &inputs = changes.Inputs.bind(r);
    inputs.on<MeshGeometryDirty>(On::Create).on<MeshPositionsChanged>(On::Create);
    inputs.on<InlineSelection>(On::Create | On::Update | On::Destroy);
    inputs.on<Instance>(On::Create).on<SceneParent>(On::Destroy);
    changes.Moved.bind(r, state::Clearing::Explicit).on<Transform>(On::Update).on<PosedLocal>(On::Create | On::Update | On::Destroy).on<SceneParent>(On::Create | On::Destroy);
    ForEachKind([&]<typename K> { inputs.on<K>(On::Update); });
}

namespace procedure {
namespace {
// Adds each mesh storage state the component refers to.
void AddStates(const auto &held, std::vector<store::Hash128> &states) {
    if constexpr (requires { held.Values; }) {
        for (const auto &[path, value] : held.Values) AddStates(value, states);
    } else if (held.State != store::Hash128{}) {
        states.push_back(held.State);
    }
}
void SortStates(std::vector<store::Hash128> &states) {
    std::ranges::sort(states, {}, [](const store::Hash128 &state) { return std::pair{state.A, state.B}; });
    states.erase(std::ranges::unique(states).begin(), states.end());
}

// The previous plan's node for each node of a plan compiled since: the same occurrence, or else the only occurrence of the same node in both plans.
std::vector<uint32_t> Matches(const CookPlan &previous, const CookPlan &compiled) {
    using Key = std::tuple<bool, state::Entity, uint32_t>;
    const auto keys = [](const CookPlan &plan) {
        std::vector<Key> keys;
        for (uint32_t n = 0; n < plan.Nodes.size(); ++n) keys.emplace_back(plan.Nodes[n].Builtin, plan.Nodes[n].Entity, n);
        std::ranges::sort(keys);
        return keys;
    };
    const auto before = keys(previous), after = keys(compiled);
    const auto only = [](const std::vector<Key> &keys, const NodeRecord &node) {
        const auto [first, last] = std::ranges::equal_range(keys, std::pair{node.Builtin, node.Entity}, {}, [](const Key &key) { return std::pair{std::get<0>(key), std::get<1>(key)}; });
        return last - first == 1 ? std::get<2>(*first) : NoNode;
    };
    std::vector<uint32_t> matches(compiled.Nodes.size(), NoNode);
    for (uint32_t n = 0; n < compiled.Nodes.size(); ++n) {
        const auto &node = compiled.Nodes[n];
        if (const auto same = IndexOf(previous, PathOf(compiled, n)); same != NoNode && previous.Nodes[same].Builtin == node.Builtin) matches[n] = same;
        else if (only(after, node) != NoNode) matches[n] = only(before, node);
    }
    return matches;
}
// Whether each occurrence of the compiled plan differs from its match in the previous plan, which `matches` gives: it has none or reads other inputs.
// An unlinked input reads its constant, or else its node's parameter.
// An occurrence of a node created since has no match and reads nothing changed.
std::vector<bool> Rewired(const state::Registry &r, const CookPlan &previous, const CookPlan &compiled, std::span<const uint32_t> matches, const state::DirtySet &created) {
    const auto mapped = [&](uint32_t node) { return node == NoNode ? NoNode : matches[node]; };
    std::vector<bool> rewired(compiled.Nodes.size());
    for (uint32_t n = 0; n < compiled.Nodes.size(); ++n) {
        const auto before = matches[n];
        if (before == NoNode) {
            rewired[n] = compiled.Nodes[n].Builtin || !created.contains(compiled.Nodes[n].Entity);
            continue;
        }
        const auto &at = Authored(r, compiled.Nodes[n]);
        const auto entity = compiled.Nodes[n].Entity;
        bool changed = InputsOf(compiled, n) != InputsOf(previous, before);
        for (uint16_t i = 0; !changed && i < InputsOf(compiled, n); ++i) {
            const auto &a = compiled.Inputs[compiled.InputFirst[n] + i], &b = previous.Inputs[previous.InputFirst[before] + i];
            const bool same_constant = a.Constant == b.Constant ||
                (a.FromNode == NoNode && a.Constant.has_value() != b.Constant.has_value() && ParameterDefault(at, entity, SlotSocketOf(at, entity, i)) == (a.Constant ? a.Constant : b.Constant));
            changed = mapped(a.FromNode) != b.FromNode || a.FromSocket != b.FromSocket || !same_constant || mapped(a.SelectionSubnet) != b.SelectionSubnet || a.SelectionPort != b.SelectionPort;
        }
        rewired[n] = changed;
    }
    return rewired;
}
// Drops the images of the Bake nodes and filters of the mesh's plan that read a `reread` occurrence, which run again.
void DropReaderImages(state::Registry &r, state::Entity mesh, const std::vector<bool> &reread) {
    const auto &plan = r.get<const CookPlan>(mesh);
    const auto invalid = Downstream(plan, reread);
    for (const auto n : plan.Order) {
        if (invalid[n] && HoldsImage(r, plan, n)) CurrentImages(r).Images.erase({mesh, NodePath{std::from_range, PathOf(plan, n)}});
    }
}
// Keeps the demanded op occurrences of the mesh's plan that read the timeline's frame or time on the mesh.
// The results of the nodes off the display path drop when the mesh cooks, so no undemanded read reaches a kept state.
void SetTimeReads(state::Registry &r, state::Entity mesh) {
    const auto &plan = r.get<const CookPlan>(mesh);
    TimeReads reads;
    for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
        if (plan.Nodes[n].Kind != KindOf<Op> || !plan.Demanded[n]) continue;
        const auto *definition = DefinitionOf(Authored(r, plan.Nodes[n]).get<const Op>(plan.Nodes[n].Entity).Node);
        if (definition && definition->ReadsFrame) reads.Frame.push_back(n);
        if (definition && definition->ReadsTime) reads.Time.push_back(n);
    }
    if (reads.Frame.empty() && reads.Time.empty()) r.remove<TimeReads>(mesh);
    else r.emplace_or_replace<TimeReads>(mesh, std::move(reads));
}
// The sorted meshes in an order where each follows the meshes it reads.
std::vector<state::Entity> ReadOrder(std::span<const state::Entity> meshes, std::span<const std::pair<state::Entity, state::Entity>> readers) {
    const auto index_of = [&](state::Entity entity) {
        const auto at = std::ranges::lower_bound(meshes, entity);
        return at != meshes.end() && *at == entity ? uint32_t(at - meshes.begin()) : NoNode;
    };
    const auto order = DependencyOrder(uint32_t(meshes.size()), [&](auto visit) {
        for (const auto [source, reader] : readers) {
            const auto from = index_of(source), to = index_of(reader);
            if (from != NoNode && to != NoNode) visit(from, to);
        }
    });
    return order | std::views::transform([&](uint32_t n) { return meshes[n]; }) | std::ranges::to<std::vector>();
}
// Evaluates the meshes' plans in order at the frame without publishing, passes the records they evaluate by mesh to `read`, then restores mesh storage exactly.
void EvaluateAndRestore(state::Registry &r, std::span<const std::pair<state::Entity, const CookPlan *>> plans, int frame, RunMode mode, auto &&read) {
    auto &store = r.Resources.get<MeshStore>();
    WaitForRender(r);
    auto pin = store.Pin();
    // The geometry changes the evaluation and the restore report leave the next cooks' inputs as they were.
    auto &inputs = r.Resources.get<ProcedureChanges>().Inputs;
    const auto changed = inputs.Entities;
    auto pass = PassAt(r, frame);
    for (const auto [mesh, plan] : plans) PrepareOpPrograms(r, pass, mesh, *plan);
    if (!pass.Programs.empty()) ops::LinkCompiled(r, ops::GetDrivers(r));
    for (const auto [mesh, plan] : plans) pass.Records.emplace(mesh, Evaluate(r, mesh, *plan, pass, {}, mode));
    read(std::as_const(pass.Records));
    WaitForRender(r);
    store.Restore(pin);
    store.Release(pin);
    ReconcileRestoredStore(r, state::Null);
    for (const auto e : inputs.Entities | std::ranges::to<std::vector>()) {
        if (!std::ranges::contains(changed, e)) inputs.remove(e);
    }
}
} // namespace
} // namespace procedure

namespace {
void SettleChanges(state::Registry &r, EventPass pass) {
    using namespace procedure;
    const bool cook = pass == EventPass::Frame || pass == EventPass::Settle;
    auto &changes = r.Resources.get<ProcedureChanges>();
    r.Resources.get<TimeCooks>().Meshes.clear();
    // The network holding the entity: a node's network, or the entity itself.
    const auto network_of = [&](state::Entity e) {
        const auto *node = r.valid(e) ? r.try_get<const ProcedureNode>(e) : nullptr;
        return node ? node->Network : e;
    };
    // The meshes whose plans instance any of the definitions.
    const auto instancing = [&](std::vector<state::Entity> definitions) {
        SortUnique(definitions);
        std::vector<state::Entity> meshes;
        if (definitions.empty()) return meshes;
        for (const auto [mesh, plan] : r.view<const CookPlan>().each())
            if (std::ranges::any_of(plan.Definitions, [&](state::Entity d) { return std::ranges::binary_search(definitions, d); })) meshes.push_back(mesh);
        return meshes;
    };
    std::vector<state::Entity> planned, rewired, definitions;
    // Each procedure's changed entities, which hold the procedure itself when it changed.
    std::map<state::Entity, std::vector<state::Entity>> changed_nodes;
    if (!changes.Graph.empty()) ++r.Resources.get<NetworkChanges>().GraphRevision;
    for (const auto e : changes.Graph) {
        // An identity a restore brings back without its procedure keeps no plan of it.
        if (r.valid(e) && !r.all_of<GeometryProcedure>(e)) r.remove<CookPlan, TimeReads, TexcoordWriters>(e);
        const auto network = network_of(e);
        if (!r.valid(network)) continue;
        if (r.all_of<GeometryProcedure>(network)) {
            planned.push_back(network);
            if (e != network) rewired.push_back(network);
            changed_nodes[network].push_back(e);
        } else if (r.all_of<SubnetDefinition>(network)) definitions.push_back(network);
    }
    const auto replanned = instancing(std::move(definitions));
    planned.append_range(replanned);
    rewired.append_range(replanned);
    SortUnique(planned);
    SortUnique(rewired);
    std::vector<state::Entity> cooked;
    // Each replanned mesh's occurrences whose inputs differ from its previous plan's.
    std::map<state::Entity, std::vector<bool>> rewired_nodes;
    for (const auto e : planned) {
        const auto &procedure = r.get<const GeometryProcedure>(e);
        const auto &ends = r.get<const NetworkNodes>(e);
        const auto *previous_plan = r.try_get<const CookPlan>(e);
        if (previous_plan && !std::ranges::binary_search(rewired, e) &&
            (previous_plan->Nodes.empty() ? ends.First == state::Null : previous_plan->Nodes.front().Entity == ends.First && IndexOf(*previous_plan, ends.Last) != NoNode)) {
            const auto result = previous_plan->ResultNode;
            r.patch<CookPlan>(e, [&](auto &plan) { BindDisplay(r, procedure, plan); ++plan.Revision; });
            SetTimeReads(r, e);
            if (cook && result != r.get<const CookPlan>(e).ResultNode) cooked.push_back(e);
            continue;
        }
        // A change of the values of input ports alone takes them into the previous plan.
        const auto &nodes_changed = changed_nodes[e];
        const bool own = !nodes_changed.empty() && !std::ranges::contains(nodes_changed, e) && !std::ranges::contains(replanned, e);
        auto revalued = previous_plan && own ? Revalued(r, *previous_plan, nodes_changed) : std::nullopt;
        auto compiled = revalued ? std::move(*revalued) : Compile(r, e);
        compiled.Revision = previous_plan ? previous_plan->Revision + 1 : 1;
        if (cook && previous_plan) {
            const auto matches = Matches(*previous_plan, compiled);
            auto nodes = Rewired(r, *previous_plan, compiled, matches, changes.Created);
            const auto result = compiled.ResultNode == NoNode ? NoNode : matches[compiled.ResultNode];
            // A record that holds the new result's occurrence, which an edit that copied the displayed node keeps, needs no cook for it.
            const auto *evaluated = r.try_get<const EvaluatedMesh>(e);
            const bool holds = evaluated && !evaluated->Tip.empty() && compiled.ResultNode != NoNode && std::ranges::equal(evaluated->Tip, PathOf(compiled, compiled.ResultNode));
            if (!holds && (result != previous_plan->ResultNode || (compiled.ResultNode != NoNode) != (previous_plan->ResultNode != NoNode))) cooked.push_back(e);
            rewired_nodes.emplace(e, std::move(nodes));
        } else if (cook) cooked.push_back(e);
        r.emplace_or_replace<CookPlan>(e, std::move(compiled));
        SetTimeReads(r, e);
        if (!cook) continue;
        // Occurrences inside a subnet node of the mesh's network that left the plan keep nothing.
        // The settle pass drops them directly, outside an action's Apply, since a plan derives which occurrences exist.
        const auto &plan = r.get<const CookPlan>(e);
        for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
            if (plan.Nodes[n].Parent != NoNode || plan.Nodes[n].Kind != KindOf<Subnet>) continue;
            const auto subnet = plan.Nodes[n].Entity;
            const auto gone = [&](const auto &entry) {
                auto at = n;
                for (const auto entity : entry.first) at = at == NoNode ? NoNode : IndexOf(plan, at, entity);
                return at == NoNode;
            };
            const auto prune = [&]<typename T> {
                const auto *held = r.try_get<const Inner<T>>(subnet);
                if (!held || std::ranges::none_of(held->Values, gone)) return;
                r.patch<Inner<T>>(subnet, [&](auto &kept) { std::erase_if(kept.Values, gone); });
                if (r.get<const Inner<T>>(subnet).Values.empty()) r.remove<Inner<T>>(subnet);
            };
            [&]<typename... T>(TypeList<T...>) { (prune.template operator()<T>(), ...); }(OccurrenceStates{});
        }
    }
    // An occurrence of a Bake or a filter that left its mesh's plan, or whose mesh is gone, keeps no image.
    if (!changes.Graph.empty()) {
        const auto occurs = [&](state::Entity mesh, std::span<const state::Entity> path) {
            const auto *plan = r.valid(mesh) ? r.try_get<const CookPlan>(mesh) : nullptr;
            const auto n = plan ? IndexOf(*plan, path) : NoNode;
            return n != NoNode && HoldsImage(r, *plan, n);
        };
        std::erase_if(CurrentImages(r).Images, [&](const auto &entry) { return !occurs(entry.first.first, entry.first.second); });
    }
    const auto rewired_of = [&](state::Entity mesh) -> std::vector<bool> & { return rewired_nodes.try_emplace(mesh, r.get<const CookPlan>(mesh).Nodes.size()).first->second; };
    const bool references = changes.ReferencesEpoch != r.Epoch || !changes.References.empty() || !planned.empty();
    if (references) {
        changes.Readers.clear();
        changes.PlacementReaders.clear();
        const auto placement = [&](state::Entity procedure, state::Entity object) {
            for (auto e = r.valid(object) ? object : state::Null; e != state::Null; e = ParentOrNull(r, e)) changes.PlacementReaders.emplace_back(e, procedure);
        };
        // The objects each demanded occurrence of a node with an object reference reads, as its mesh's hierarchy instance contexts map them.
        const auto read = [&](state::Entity mesh, uint32_t n, state::Entity node) {
            if (const auto *info = r.try_get<const ObjectInfo>(node)) {
                const auto object = ContextObject(r, mesh, info->Object);
                if (cook && changes.Objects.contains(object)) {
                    rewired_of(mesh)[n] = true;
                    cooked.push_back(mesh);
                }
                if (const auto *instance = r.try_get<const Instance>(object); instance && instance->Entity != mesh) changes.Readers.emplace_back(instance->Entity, mesh);
                placement(mesh, object);
            } else {
                placement(mesh, ContextObject(r, mesh, r.get<const procedure::Boolean>(node).Space));
            }
        };
        // A node of a mesh's network has one occurrence, and the occurrences of a definition's nodes are found in one pass over the plans that list their definitions.
        std::vector<state::Entity> inner, held;
        const auto visit = [&](state::Entity node) {
            if (!r.all_of<ProcedureNode>(node)) return;
            const auto network = NetworkOf(r, node);
            if (!r.all_of<GeometryProcedure>(network)) {
                inner.push_back(node);
                held.push_back(network);
                return;
            }
            const auto &plan = r.get<const CookPlan>(network);
            if (const auto n = IndexOf(plan, node); n != NoNode && plan.Demanded[n]) read(network, n, node);
        };
        for (const auto node : std::as_const(r).view<const ObjectInfo>()) visit(node);
        for (const auto node : std::as_const(r).view<const procedure::Boolean>()) visit(node);
        if (!inner.empty()) {
            SortUnique(inner);
            for (const auto procedure : instancing(std::move(held))) {
                const auto &plan = r.get<const CookPlan>(procedure);
                for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
                    if (!plan.Nodes[n].Builtin && plan.Demanded[n] && std::ranges::binary_search(inner, plan.Nodes[n].Entity)) read(procedure, n, plan.Nodes[n].Entity);
                }
            }
        }
        SortUnique(changes.Readers);
        SortUnique(changes.PlacementReaders);
        changes.References.clear();
        changes.Objects.clear();
        changes.ReferencesEpoch = r.Epoch;
    }
    // An op occurrence reading a changed texture changed.
    if (cook && !changes.Textures.empty()) {
        for (const auto [mesh, plan] : std::as_const(r).view<const CookPlan>().each()) {
            for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
                if (plan.Nodes[n].Kind != KindOf<Op> || !plan.Demanded[n]) continue;
                const auto &images = Authored(r, plan.Nodes[n]).get<const Op>(plan.Nodes[n].Entity).Images;
                if (std::ranges::any_of(images, [&](const auto &image) { return changes.Textures.contains(image.second); })) rewired_of(mesh)[n] = true;
            }
        }
    }
    auto &cooked_time = r.Resources.get<CookTime>();
    const CookTime now{r.get<const TimelinePlayback>(state::Document).CurrentFrame, r.get<const TimelineRange>(state::Document).Fps};
    auto ops = PassAt(r, now.Frame);
    // A generator occurrence whose write of texture coordinate set 0 changed runs again.
    // The settle pass keeps each procedure's writers directly, outside an action's Apply, since the document derives them.
    // A restore, a change of what any procedure reads or of the materials, and a replanned mesh refresh them.
    const auto &render_changes = r.Resources.get<const RenderChanges>();
    const bool refresh = pass == EventPass::Restore || references || !render_changes.Materials.empty() || !render_changes.ActiveMaterialVariant.empty() || !r.Resources.get<const MeshChanges>().Material.empty();
    for (const auto mesh : refresh ? SortedEntities(r.view<const CookPlan>()) : planned) {
        const auto *held = r.try_get<const TexcoordWriters>(mesh);
        const bool replanned = std::ranges::binary_search(planned, mesh);
        if (!held && !replanned) continue;
        const bool all = TexcoordsRead(r, mesh, changes.Readers);
        if (!replanned && held->All == all) continue;
        const auto &plan = r.get<const CookPlan>(mesh);
        std::vector<uint32_t> generators;
        for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
            if (GeneratorKind[plan.Nodes[n].Kind]) generators.push_back(n);
        }
        if (generators.empty()) {
            if (held) r.remove<TexcoordWriters>(mesh);
            continue;
        }
        TexcoordWriters writers{.All = all, .Occurrences = all ? std::vector<NodePath>{} : ReadGenerators(r, ops, mesh, plan, generators)};
        for (const auto n : generators) {
            if (cook && Writes(held, PathOf(plan, n)) != Writes(&writers, PathOf(plan, n))) rewired_of(mesh)[n] = true;
        }
        r.emplace_or_replace<TexcoordWriters>(mesh, std::move(writers));
    }
    // A restore holds the cooks of the restored frame.
    if (pass == EventPass::Restore) cooked_time = now;
    if (!cook) return;
    // The occurrences that read the frame or the time where it changed, by mesh, sorted.
    std::map<state::Entity, std::vector<bool>> timed;
    if (now.Frame != cooked_time.Frame || now.Fps != cooked_time.Fps) {
        const bool frame = now.Frame != cooked_time.Frame;
        cooked_time = now;
        for (const auto [mesh, reads] : std::as_const(r).view<const TimeReads>().each()) {
            if (!frame && reads.Time.empty()) continue;
            std::vector<bool> nodes(r.get<const CookPlan>(mesh).Nodes.size());
            for (const auto n : reads.Time) nodes[n] = true;
            if (frame) {
                for (const auto n : reads.Frame) nodes[n] = true;
            }
            timed.emplace(mesh, std::move(nodes));
        }
    }
    std::vector<state::Entity> affected = planned, inputs;
    definitions.clear();
    for (const auto e : changes.Inputs) {
        const auto network = network_of(e);
        if (r.valid(network) && r.all_of<GeometryProcedure>(network)) affected.push_back(network);
        else if (r.valid(network) && r.all_of<SubnetDefinition>(network)) definitions.push_back(network);
        else if (!(r.valid(e) && r.any_of<ProcedureNode, SubnetDefinition>(e))) inputs.push_back(e);
    }
    affected.append_range(instancing(std::move(definitions)));
    for (const auto &[mesh, nodes] : rewired_nodes) affected.push_back(mesh);
    for (const auto &[mesh, nodes] : timed) affected.push_back(mesh);
    SortUnique(affected);
    // The occurrences that changed of each affected mesh with any: rewired, or of a node whose own inputs changed since it was created.
    std::map<state::Entity, std::vector<bool>> changed;
    // The procedures with an authored change of their own.
    std::vector<state::Entity> authored = planned;
    for (const auto owner : affected) {
        const auto &plan = r.get<const CookPlan>(owner);
        LayoutOpValues(r, owner, plan, LayoutIn(ops, r, owner, plan));
        const auto rewired_at = rewired_nodes.find(owner), timed_at = timed.find(owner);
        const auto *nodes = rewired_at != rewired_nodes.end() ? &rewired_at->second : nullptr;
        const auto *times = timed_at != timed.end() ? &timed_at->second : nullptr;
        std::vector<bool> own;
        std::vector<bool> invalid(plan.Nodes.size());
        const bool owner_changed = changes.Inputs.contains(owner);
        bool demanded = owner_changed;
        if (demanded) authored.push_back(owner);
        for (const auto n : plan.Order) {
            const auto &node = plan.Nodes[n];
            const bool input = !node.Builtin && changes.Inputs.contains(node.Entity);
            const bool edit = (nodes && (*nodes)[n]) || (input && !changes.Created.contains(node.Entity));
            const bool own_change = edit || (times && (*times)[n]);
            if (edit) authored.push_back(owner);
            if (own_change) {
                if (own.empty()) own.resize(plan.Nodes.size());
                own[n] = true;
            }
            bool upstream = owner_changed;
            for (const auto &link : NodeLinks(plan, n))
                if (link.FromNode != NoNode) upstream = upstream || invalid[link.FromNode];
            invalid[n] = upstream || own_change;
            demanded = demanded || (plan.Demanded[n] && own_change);
            // A Bake or a filter whose input changed runs again.
            if (invalid[n] && HoldsImage(r, plan, n)) CurrentImages(r).Images.erase({owner, NodePath{std::from_range, PathOf(plan, n)}});
            if (!invalid[n]) continue;
            const auto path = PathOf(plan, n);
            // A handle follows the node's input and selection, so a cook that restarts at the node keeps one only for a change of its own parameters.
            if (upstream || (!node.Builtin && (changes.HandleInputs.contains(node.Entity) || changes.Graph.contains(node.Entity)))) Drop<NodeHandle>(r, path);
            if (!plan.Demanded[n]) Drop<NodeResult, NodeExtrusion>(r, path);
        }
        if (demanded) cooked.push_back(owner);
        if (!own.empty()) changed.emplace(owner, std::move(own));
    }
    const bool handle_dirty = !affected.empty() || !changes.Moved.empty();
    const auto handle = [&] {
        if (!handle_dirty) return;
        const auto *active = r.try_get<const OperationHandle>(state::Document);
        const auto *object = active && r.all_of<ProcedureNode>(active->Node) ? r.try_get<const Instance>(active->Object) : nullptr;
        if (object && r.all_of<GeometryProcedure>(object->Entity)) PrepareOperationHandle(r, object->Entity, active->Path, active->Node);
    };
    SortUnique(cooked);
    if (cooked.empty() && inputs.empty() && changes.Moved.empty()) {
        handle();
        return;
    }
    SortUnique(inputs);
    const auto input = [&](state::Entity e) { return std::ranges::binary_search(inputs, e); };
    for (const auto [e, published] : std::as_const(r).view<const ProcedureInstances>().each()) {
        if (std::ranges::any_of(published.Entities, input)) {
            cooked.push_back(e);
            authored.push_back(e);
        }
    }
    const auto readers_of = [](const auto &readers, state::Entity source) {
        const auto [first, last] = std::ranges::equal_range(readers, source, {}, &std::pair<state::Entity, state::Entity>::first);
        return std::ranges::subrange(first, last);
    };
    for (const auto e : inputs) {
        if (const auto *instance = r.try_get<const Instance>(e); instance && r.all_of<ProcedureInstances>(instance->Entity)) {
            cooked.push_back(instance->Entity);
            authored.push_back(instance->Entity);
        }
    }
    // A procedure whose placing objects moved cooks only when the motion changed one of its Boolean nodes' placements.
    // An edit that cooked in its own pass then does not cook again for the world transform it published.
    std::vector<state::Entity> moved;
    for (const auto source : changes.Moved) {
        for (const auto [_, reader] : readers_of(changes.PlacementReaders, source)) moved.push_back(reader);
    }
    changes.Moved.clear();
    SortUnique(moved);
    // The mesh's changed occurrences, which an occurrence whose object changed what it reads joins.
    const auto changed_of = [&](state::Entity mesh) -> std::vector<bool> & { return changed.try_emplace(mesh, r.get<const CookPlan>(mesh).Nodes.size()).first->second; };
    // The occurrences whose objects changed what they read join the changed ones: the Boolean nodes whose placements changed, and the Object Info nodes whose objects' meshes cook or changed.
    // The meshes among their procedures that keep an image drop the images of their readers.
    std::vector<state::Entity> rereaders;
    const auto &kept = CurrentImages(r).Images;
    const auto reread = [&](state::Entity mesh, uint32_t node) {
        changed_of(mesh)[node] = true;
        const auto first = kept.lower_bound({mesh, NodePath{}});
        if (first != kept.end() && first->first.first == mesh) rereaders.push_back(mesh);
    };
    for (const auto procedure : moved) {
        const auto &plan = r.get<const CookPlan>(procedure);
        for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
            const auto *placement = plan.Nodes[n].Kind == KindOf<procedure::Boolean> ? CookedOf<BooleanPlacement>(r, PathOf(plan, n)) : nullptr;
            const auto space = placement ? ContextObject(r, procedure, Authored(r, plan.Nodes[n]).get<const procedure::Boolean>(plan.Nodes[n].Entity).Space) : state::Null;
            if (placement && std::ranges::any_of(placement->Operands, [&](const auto &operand) { return ToFirst(r, space, operand.Operand) != operand.ToFirst; })) {
                cooked.push_back(procedure);
                authored.push_back(procedure);
                reread(procedure, n);
            }
        }
    }
    SortUnique(cooked);
    SortUnique(authored);
    std::vector<state::Entity> queue = cooked;
    queue.append_range(inputs);
    for (size_t i = 0; i < queue.size(); ++i) {
        const bool edit = i >= cooked.size() || std::ranges::binary_search(authored, queue[i]);
        for (const auto [_, reader] : readers_of(changes.Readers, queue[i])) {
            if (edit && !std::ranges::binary_search(authored, reader)) authored.insert(std::ranges::lower_bound(authored, reader), reader);
            const auto &plan = r.get<const CookPlan>(reader);
            for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
                if (plan.Nodes[n].Kind != KindOf<ObjectInfo>) continue;
                const auto *instance = r.try_get<const Instance>(ContextObject(r, reader, Authored(r, plan.Nodes[n]).get<const ObjectInfo>(plan.Nodes[n].Entity).Object));
                if (instance && instance->Entity == queue[i]) reread(reader, n);
            }
            const auto at = std::ranges::lower_bound(cooked, reader);
            if (at != cooked.end() && *at == reader) continue;
            cooked.insert(at, reader);
            queue.push_back(reader);
        }
    }
    SortUnique(rereaders);
    for (const auto mesh : rereaders) DropReaderImages(r, mesh, changed_of(mesh));
    const auto order = ReadOrder(cooked, changes.Readers);
    assert(order.size() == cooked.size());
    // A procedure that the time alone reaches, without an authored change of its own or of what it reads, is a time cook.
    auto &time_cooks = r.Resources.get<TimeCooks>().Meshes;
    std::ranges::set_difference(cooked, authored, std::back_inserter(time_cooks));
    if (order.empty()) {
        handle();
        return;
    }
    // The kept states restore the whole mesh storage exactly only while the passes of one mesh are its only writers.
    auto &cook_pass = r.Resources.get<CookPass>();
    const auto mesh = order.size() == 1u ? order.front() : state::Null;
    if (!(cook_pass.End && mesh != state::Null && mesh == cook_pass.Mesh && r.Resources.get<MeshStore>().StateHash() == *cook_pass.End)) cook_pass.Exact.clear();
    cook_pass.Mesh = mesh;
    cook_pass.End.reset();
    // Derive new-mesh shading before cooking dependent procedures.
    ShadeNewMeshes(r);
    for (const auto e : order) PrepareOpPrograms(r, ops, e, r.get<const CookPlan>(e));
    if (!ops.Programs.empty()) ops::LinkCompiled(r, ops::GetDrivers(r));
    ChainBatch batch{r.Resources.get<MeshStore>().BufferContext()};
    for (const auto e : order) {
        static const std::vector<bool> unchanged;
        const auto at = changed.find(e);
        const auto &own = at != changed.end() ? at->second : unchanged;
        Evaluate(r, e, r.get<const CookPlan>(e), ops, own, RunMode::Publish, &batch);
    }
    CookBatch(r, batch);
    r.Resources.get<CookCount>().Value += order.size();
    handle();
}
} // namespace

void SettleProcedures(state::Registry &r, EventPass pass) {
    SettleChanges(r, pass);
    // A write later in the same event pass cooks in the next settle.
    auto &changes = r.Resources.get<procedure::ProcedureChanges>();
    for (auto *changed : {&changes.Graph, &changes.Inputs, &changes.HandleInputs, &changes.Created}) changed->clear();
}

std::vector<state::Entity> TimeDependentMeshes(const state::Registry &r) {
    using namespace procedure;
    std::vector<state::Entity> meshes;
    for (const auto mesh : r.view<const TimeReads>()) meshes.push_back(mesh);
    const auto &readers = r.Resources.get<const ProcedureChanges>().Readers;
    for (size_t i = 0; i < meshes.size(); ++i) {
        const auto [first, last] = std::ranges::equal_range(readers, meshes[i], {}, &std::pair<state::Entity, state::Entity>::first);
        for (const auto &[_, reader] : std::ranges::subrange(first, last)) {
            if (!std::ranges::contains(meshes, reader)) meshes.push_back(reader);
        }
    }
    std::ranges::sort(meshes);
    return meshes;
}

void ReadAtFrame(state::Registry &r, std::span<const state::Entity> meshes, int frame, const std::function<void(std::span<const uint32_t>)> &read) {
    using namespace procedure;
    const auto dependent = TimeDependentMeshes(r);
    const auto &readers = r.Resources.get<const ProcedureChanges>().Readers;
    // The time-dependent meshes the requested ones read, with them.
    std::vector<state::Entity> evaluated{meshes.begin(), meshes.end()};
    for (size_t i = 0; i < evaluated.size(); ++i) {
        for (const auto &[source, reader] : readers) {
            if (reader == evaluated[i] && std::ranges::binary_search(dependent, source) && !std::ranges::contains(evaluated, source)) evaluated.push_back(source);
        }
    }
    SortUnique(evaluated);
    const auto plans = ReadOrder(evaluated, readers) | std::views::transform([&](state::Entity e) { return std::pair{e, &r.get<const CookPlan>(e)}; }) | std::ranges::to<std::vector>();
    EvaluateAndRestore(r, plans, frame, RunMode::Transient, [&](const auto &records) { read(meshes | std::views::transform([&](state::Entity e) { return records.at(e); }) | std::ranges::to<std::vector>()); });
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

std::vector<store::Hash128> DocumentStates(const state::Registry &r) {
    using namespace procedure;
    std::vector<store::Hash128> states;
    [&]<typename... Ts>(TypeList<Ts...>) { (..., [&] { for (const auto [_, held] : r.view<const Ts>().each()) AddStates(held, states); }()); }(StateHolders{});
    SortStates(states);
    return states;
}

void RecordMeshStates(state::Registry &r, std::span<const store::Hash128> states) {
    using namespace procedure;
    auto &changed = r.Resources.get<ProcedureChanges>().States;
    std::vector<store::Hash128> referenced{states.begin(), states.end()};
    for (const auto entity : changed) {
        if (!r.valid(entity)) continue;
        [&]<typename... Ts>(TypeList<Ts...>) { (..., [&] { if (const auto *held = r.try_get<const Ts>(entity)) AddStates(*held, referenced); }()); }(StateHolders{});
    }
    SortStates(referenced);
    if (r.Resources.get<MeshStore>().RecordKeptStates(referenced)) changed.clear();
}

void ReopenSourceState(state::Registry &r, state::Entity mesh_entity) {
    using namespace procedure;
    const auto *evaluated = r.try_get<const EvaluatedMesh>(mesh_entity);
    const auto tip = evaluated && evaluated->Tip.size() == 1u && r.valid(evaluated->Tip.front()) ? evaluated->Tip.front() : state::Null;
    const auto *source = tip != state::Null ? r.try_get<const SourceState>(tip) : nullptr;
    if (source && source->StoreId == evaluated->StoreId && source->State != store::Hash128{}) r.replace<SourceState>(tip, store::Hash128{}, source->StoreId);
}

void InvalidateNode(state::Registry &r, state::Entity node) { r.Resources.get<procedure::ProcedureChanges>().Inputs.emplace(node); }

bool CooksDependOnEarlierCooks(const state::Registry &r) { return r.Resources.get<const procedure::CookPass>().DependsOnEarlierCooks; }
void ClearCookDependence(state::Registry &r) { r.Resources.get<procedure::CookPass>().DependsOnEarlierCooks = false; }

void EndCookStep(state::Registry &r) {
    r.clear<procedure::StagedNode>();
    r.Resources.get<procedure::CookPass>().Exact.clear();
}

namespace procedure {
std::optional<ImageTextures> OccurrenceImageOf(state::Registry &r, state::Entity mesh, std::span<const state::Entity> path) {
    const std::pair key{mesh, NodePath{path.begin(), path.end()}};
    if (!CurrentImages(r).Images.contains(key)) {
        // The nodes the occurrence reads evaluate at the cooks' frame, and mesh storage returns to its state before them.
        const auto *compiled = r.try_get<const CookPlan>(mesh);
        const auto node = compiled ? IndexOf(*compiled, path) : NoNode;
        if (node == NoNode || !HoldsImage(r, *compiled, node)) return {};
        auto plan = *compiled;
        plan.ResultNode = NoNode;
        Demand(plan, std::array{node});
        EvaluateAndRestore(r, std::array{std::pair<state::Entity, const CookPlan *>{mesh, &plan}}, r.Resources.get<const CookTime>().Frame, RunMode::Display, [](const auto &) {});
        // An occurrence the evaluation did not run, which lacks its geometry, holds no image.
        CurrentImages(r).Images.try_emplace(key);
    }
    const auto &image = CurrentImages(r).Images.at(key);
    if (!image.Views) return {};
    const auto &views = *image.Views;
    return ImageTextures{*views.Texture, *views.Shown, views.Texture.Extent.Width, views.Texture.Extent.Height, image.Views};
}
} // namespace procedure
