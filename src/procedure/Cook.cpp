#include "procedure/Cook.h"
#include "procedure/Procedure.h"

#include "SortUnique.h"
#include "mesh/Decimate.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "mesh/RecalculateNormals.h"
#include "mesh/TopologyOperations.h"
#include "metal/Dispatch.h"
#include "numeric/MatrixMath.h"
#include "numeric/QuaternionMath.h"
#include "numeric/VectorMath.h"
#include "object/ObjectOps.h"
#include "object/PendingSync.h"
#include "render/Instance.h"
#include "render/MeshletBuildGpu.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "state/Scene.h"
#include "viewport/ViewportEvents.h"

#include <cstring>
#include <expected>
#include <format>
#include <numbers>
#include <optional>
#include <string>

using state::Change;
using state::On;

namespace procedure {
namespace {
using namespace action::mesh;
// A node's selection input as each domain's masks, or none for every element, and whether its edit publishes its emitted selection into its record's masks.
struct Masks {
    std::optional<GeometryMasks> In;
    bool Publish{};
};

// A socket's value during a cook: a record for geometry, points and the record they place for instances, each domain's first mask block in the cook's arena for a selection, or an attribute of the record its consumer reads.
// A selection without mask blocks selects every element, and an absent attribute is InvalidOffset.
struct SocketValue {
    SocketType Type{SocketType::Geometry};
    // The mesh entity whose drawn record the value borrows: a geometry value's record, or an instances value's instance record.
    // The cook reads a borrowed record in place and never edits or releases it, and it owns the records of a value without a mesh entity.
    state::Entity MeshEntity{state::Null};
    uint32_t Instance{InvalidOffset}; // The record an instances value's points each place, or InvalidOffset for points placing nothing
    union {
        uint32_t Record{InvalidOffset};
        std::array<uint32_t, 3> MaskFirst;
        uint32_t Attribute;
    };
    bool Borrowed() const { return Type == SocketType::Geometry && MeshEntity != state::Null; }
};
static_assert(sizeof(SocketValue) == 24u);
constexpr std::array<uint32_t, 3> EveryElement{InvalidOffset, InvalidOffset, InvalidOffset};
// A node's output record, which is its input when the operation changed nothing, or why it could not run.
using Outcome = std::expected<uint32_t, NodeFault>;

// Copies of the records without render records, since a cook output builds them anew.
std::vector<uint32_t> Clones(state::Scene &r, std::span<const uint32_t> ids) {
    auto &meshes = r.Context.get<MeshStore>();
    for (const auto id : ids) meshes.PlanClone(Mesh{meshes, id});
    meshes.CommitReserves();
    CloneCopies copies;
    auto clones = meshes.CloneMeshes(copies, ids, false);
    mtl::ComputeChain chain{meshes.BufferContext()};
    copies.Encode(chain, GetMeshPipelines(r));
    chain.Submit();
    meshes.RebuildSelectionIndex(clones);
    return clones;
}
uint32_t Clone(state::Scene &r, uint32_t id) { return Clones(r, std::array{id}).front(); }

// The record's local blocks among the ascending arena blocks of a domain, skipping the blocks it does not own, ascending.
std::vector<uint32_t> LocalBlocks(const MeshStore &meshes, uint32_t id, MeshStore::ElementDomain domain, std::span<const uint32_t> blocks) {
    std::vector<uint32_t> locals;
    WithDomain(meshes.Arenas(), domain, [&](const auto &arena) {
        const auto owner = DomainSet(meshes.Get(id), domain).Index;
        const auto members = arena.Blocks.Buffer.template GetSpan<MeshElementBlock>();
        for (const auto block : blocks) {
            if (owner != InvalidOffset && block < members.size() && members[block].Owner == owner) locals.push_back(members[block].Local);
        }
    });
    std::ranges::sort(locals);
    return locals;
}

// The source's copy that the Source node emits, which `evaluated` and `resync` then describe.
// With resync state, the previous output restores to the source at the local blocks either wrote since its last cook and wherever their tables differ in shape.
uint32_t SourceCopy(state::Scene &r, uint32_t source, EvaluatedMesh &evaluated, std::optional<ResyncState> &resync, const MeshStore::Writes &between) {
    auto &meshes = r.Context.get<MeshStore>();
    if (resync && evaluated.Source == source && meshes.TryGet(evaluated.StoreId)) {
        auto written = resync->Written;
        for (const auto domain : MeshStore::ElementDomains) {
            const auto d = uint32_t(domain);
            written[d].append_range(LocalBlocks(meshes, source, domain, between.Blocks[d]));
            SortUnique(written[d]);
        }
        CloneCopies copies;
        const auto change = meshes.Resync(copies, evaluated.StoreId, source, written, between.Attributes, resync->Attributes);
        mtl::ComputeChain chain{meshes.BufferContext()};
        copies.Encode(chain, GetMeshPipelines(r));
        chain.Submit();
        meshes.ReconcileSelection(r, chain, std::span{&change, 1u});
        chain.Submit();
        ++resync->Resyncs;
        return evaluated.StoreId;
    }
    meshes.Release(evaluated.StoreId);
    resync.emplace();
    evaluated = {.StoreId = Clone(r, source), .Source = source};
    meshes.ForEachNamedAttribute(evaluated.StoreId, [&](uint32_t attribute, const auto &) { resync->Attributes.push_back(attribute); });
    return evaluated.StoreId;
}

// Keeps the record's local blocks among the arena blocks the cook wrote.
void KeepWritten(const MeshStore &meshes, uint32_t id, ResyncState &resync, const MeshStore::DomainBlocks &written) {
    for (const auto domain : MeshStore::ElementDomains) resync.Written[uint32_t(domain)] = LocalBlocks(meshes, id, domain, written[uint32_t(domain)]);
}

GeometrySelection Handles(const MeshStore &meshes, uint32_t id, const Masks &masks) { return masks.In ? ReadMasks(meshes, id, *masks.In) : AllHandles(Mesh{meshes, id}); }

MeshTopologyTask Task(uint32_t id, MeshTopologyOp op, const Masks &masks, float param0 = 0.f, float param1 = 0.f, uint32_t flags = 0u) {
    MeshTopologyTask task{.SourceId = id, .Op = op, .Param0 = param0, .Param1 = param1, .Flags = flags};
    if (masks.In) task.Selection = *masks.In;
    else task.Flags |= TopologyFlagSelectAll;
    return task;
}
// The record a task leaves, or none when it changed nothing.
Outcome Execute(state::Scene &r, std::optional<MeshTopologyTask> task, const Masks &masks) {
    if (!task) return std::unexpected(NodeFault::Selection);
    if (masks.Publish) PublishStoreSelection(r.Context.get<const MeshStore>(), *task);
    const auto result = ExecuteGeometryTopology(r, std::span{&*task, 1u}).front();
    return result.Changed ? result.GeometryId : task->SourceId;
}
// A planned primitive list takes the handles it was planned from as its selection.
Outcome ExecutePlanned(state::Scene &r, std::optional<MeshTopologyTask> task, const GeometrySelection &selection, const Masks &masks) {
    if (task && task->Op == MeshTopologyOp::AddPrimitives) task->Selection = selection;
    return Execute(r, std::move(task), masks);
}
// Deletes the record's edges and faces in place, leaving its vertices as points.
void ToPoints(state::Scene &r, uint32_t id) {
    if (Mesh{r.Context.get<const MeshStore>(), id}.EdgeCount()) {
        const auto task = Task(id, MeshTopologyOp(uint32_t(DeleteMode::OnlyEdgesAndFaces)), {});
        ExecuteGeometryTopology(r, std::span{&task, 1u});
    }
}
// Deletes every element of the record in place.
void Empty(state::Scene &r, uint32_t id) {
    if (Mesh{r.Context.get<const MeshStore>(), id}.VertexCount()) {
        const auto task = Task(id, MeshTopologyOp(uint32_t(DeleteMode::Vertices)), {});
        ExecuteGeometryTopology(r, std::span{&task, 1u});
    }
}
Outcome ExecuteStages(state::Scene &r, std::vector<MeshTopologyTask> tasks, const Masks &masks) {
    if (masks.Publish) {
        for (auto &task : tasks) PublishStoreSelection(r.Context.get<const MeshStore>(), task);
    }
    const auto results = ExecuteGeometryTopologyStages(r, tasks);
    if (std::ranges::none_of(results, &GeometryTopologyResult::Changed)) return tasks.front().SourceId;
    return results.back().GeometryId;
}
// A publishing position edit keeps every element, so its record selects the edit's selection.
Outcome Reposition(state::Scene &r, uint32_t id, const Masks &masks, PositionEditOp op, float factor, uint32_t repeat, const PositionOperationOptions &options = {}, const Transform &world = {}) {
    auto &meshes = r.Context.get<MeshStore>();
    const PositionOperationTarget target{.StoreId = id, .Selection = masks.In ? SelectionSource{*masks.In} : SelectionSource{AllHandles(Mesh{meshes, id})}, .World = world};
    if (ExecuteGeometryPositions(r, std::span{&target, 1u}, op, factor, repeat, options).empty()) return id;
    if (masks.Publish && masks.In) {
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.SelectMasks(r, chain, id, *masks.In);
        chain.Submit();
    }
    return id;
}
uint32_t Axes(bool x, bool y, bool z) { return uint32_t(x) | uint32_t(y) << 1u | uint32_t(z) << 2u; }

// Each operator edits record `id` where `masks` select, or every element without masks.
Outcome Run(state::Scene &r, const Delete &p, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp(uint32_t(p.Mode)), m), m); }
Outcome Run(state::Scene &r, const Merge &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    const auto &vertices = selection.Vertices;
    const auto reference = vertices.empty() ? InvalidOffset : p.Mode == MergeMode::Last ? vertices.back() :
                                                                                          vertices.front();
    return ExecutePlanned(r, MergeTask(meshes, Mesh{meshes, id}, selection, p.Mode, std::max(p.Distance, 0.f), reference), selection, m);
}
Outcome Run(state::Scene &r, const Extrude &p, uint32_t id, const Masks &m) {
    const auto op = p.Mode == ExtrudeMode::Vertices ? MeshTopologyOp::ExtrudeVertices : p.Mode == ExtrudeMode::Edges ? MeshTopologyOp::ExtrudeEdges :
        p.Mode == ExtrudeMode::FacesIndividual                                                                       ? MeshTopologyOp::ExtrudeFacesIndividual :
                                                                                                                       MeshTopologyOp::ExtrudeRegion;
    return Execute(r, Task(id, op, m), m);
}
Outcome Run(state::Scene &r, Duplicate, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::DuplicateGeometry, m), m); }
Outcome Run(state::Scene &r, Split, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::SplitGeometry, m), m); }
// A limited dissolve reads every selected domain, since a node has no face edit mode to protect unselected faces by.
Outcome Run(state::Scene &r, const Dissolve &p, uint32_t id, const Masks &m) {
    switch (p.Mode) {
        case DissolveMode::Vertices: return Execute(r, Task(id, MeshTopologyOp::DissolveVertices, m), m);
        case DissolveMode::Edges: return Execute(r, Task(id, MeshTopologyOp::DissolveEdges, m, 0.f, 0.f, p.KeepVertices ? TopologyFlagKeepVertices : 0u), m);
        case DissolveMode::Faces: return Execute(r, Task(id, MeshTopologyOp::DissolveFaces, m), m);
        case DissolveMode::Limited: {
            const auto flags = (p.AllBoundaries ? TopologyFlagAllBoundaries : 0u) | (p.DelimitMaterials ? TopologyFlagDelimitMaterial : 0u) | (p.DelimitSharpEdges ? TopologyFlagDelimitSharp : 0u) | (p.DelimitUVs ? TopologyFlagDelimitUV : 0u);
            return Execute(r, Task(id, MeshTopologyOp::DissolveLimited, m, std::clamp(p.Angle, 0.f, .5f * std::numbers::pi_v<float>), 0.f, flags), m);
        }
        case DissolveMode::Degenerate: return Execute(r, Task(id, MeshTopologyOp::DissolveDegenerate, m, std::max(p.Distance, 0.f)), m);
    }
    std::unreachable();
}
Outcome Run(state::Scene &r, const Subdivide &p, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::Subdivide, m, float(std::max(p.Cuts, 1u))), m); }
Outcome Run(state::Scene &r, Triangulate, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::Triangulate, m), m); }
Outcome Run(state::Scene &r, TrisToQuads, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::TrisToQuads, m), m); }
Outcome Run(state::Scene &r, const Poke &p, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::Poke, m, p.Offset), m); }
Outcome Run(state::Scene &r, FlipNormals, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::FlipNormals, m), m); }
Outcome Run(state::Scene &r, EdgeSplit, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::EdgeSplit, m), m); }
Outcome Run(state::Scene &r, Rip, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::EdgeSplit, m, 0.f, 0.f, TopologyFlagRipSelectCopies), m); }
Outcome Run(state::Scene &r, const Inset &p, uint32_t id, const Masks &m) {
    return Execute(r, Task(id, p.Individual ? MeshTopologyOp::InsetIndividual : MeshTopologyOp::InsetRegion, m, std::max(p.Thickness, 0.f), p.Depth, p.Even ? TopologyFlagEvenOffset : 0u), m);
}
Outcome Run(state::Scene &r, const Solidify &p, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::Solidify, m, p.Thickness), m); }
Outcome Run(state::Scene &r, const Wireframe &p, uint32_t id, const Masks &m) {
    const auto flags = (p.Even ? TopologyFlagEvenOffset : 0u) | (p.Boundary ? TopologyFlagWireBoundary : 0u) | (p.Relative ? TopologyFlagWireRelative : 0u) | (p.Replace ? TopologyFlagWireReplace : 0u);
    return Execute(r, Task(id, MeshTopologyOp::Wireframe, m, std::max(p.Thickness, 0.f), std::clamp(p.Offset, -1.f, 1.f), flags), m);
}
Outcome Run(state::Scene &r, const Bevel &p, uint32_t id, const Masks &m) {
    auto task = Task(id, p.Vertices ? MeshTopologyOp::BevelVertices : MeshTopologyOp::BevelEdges, m, std::max(p.Width, 0.f));
    task.Steps = std::max(p.Segments, 1u);
    return Execute(r, task, m);
}
Outcome Run(state::Scene &r, const SmoothVertices &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::Smooth, p.Factor, std::clamp(p.Repeat, 1u, 1000u), {.Axes = Axes(p.X, p.Y, p.Z)});
}
// Offsets follow vertex normals, since a node has no face edit mode to follow selected face normals by.
Outcome Run(state::Scene &r, const ShrinkFatten &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::ShrinkFatten, p.Distance, 1u, {.Flags = p.Even ? PositionEditEvenOffset : 0u});
}
Outcome Run(state::Scene &r, const Randomize &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::Randomize, p.Amount, 1u, {.Randomize = PositionRandomize{p.Uniform, p.Normal, p.Seed}});
}
Outcome Run(state::Scene &r, const Spin &p, uint32_t id, const Masks &m) {
    if (Dot(p.Axis, p.Axis) <= 0.f) return std::unexpected(NodeFault::Parameter);
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    const auto axis = Normalize(p.Axis);
    const auto steps = float(std::max(p.Steps, 1u));
    return ExecutePlanned(r, ExtrudeStepsTask(Mesh{meshes, id}, selection, Element::None, p.Steps, ToMat3(AngleAxis(p.Angle / steps, axis)), axis * (p.Offset / steps), p.Center), selection, m);
}
Outcome Run(state::Scene &r, const ExtrudeRepeat &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    return ExecutePlanned(r, ExtrudeStepsTask(Mesh{meshes, id}, selection, Element::None, p.Steps, mat3{1.f}, p.Offset, vec3{0.f}), selection, m);
}
Outcome Run(state::Scene &r, const Bisect &p, uint32_t id, const Masks &m) {
    if (Dot(p.Normal, p.Normal) <= 0.f) return std::unexpected(NodeFault::Parameter);
    return ExecuteStages(r, BisectTasks(Mesh{r.Context.get<const MeshStore>(), id}, p.Point, p.Normal, p.ClearInner, p.ClearOuter), m);
}
Outcome Run(state::Scene &r, const Symmetrize &p, uint32_t id, const Masks &m) {
    return ExecuteStages(r, SymmetrizeTasks(Mesh{r.Context.get<const MeshStore>(), id}, uint8_t(p.Axis), p.Negative), m);
}
Outcome Run(state::Scene &r, const FillHoles &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Context.get<const MeshStore>();
    return Execute(r, FillHolesTask(meshes, Mesh{meshes, id}, p.Sides), m);
}
Outcome Run(state::Scene &r, ConvexHull, uint32_t id, const Masks &m) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    return ExecutePlanned(r, ConvexHullTask(meshes, Mesh{meshes, id}, selection), selection, m);
}
Outcome Run(state::Scene &r, const Decimate &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    return ExecutePlanned(r, DecimateTask(meshes, Mesh{meshes, id}, selection, p.Ratio), selection, m);
}
Outcome Run(state::Scene &r, const RecalculateNormals &p, uint32_t id, const Masks &m) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto selection = Handles(meshes, id, m);
    auto faces = std::move(RecalculateFaceFlips(meshes, GetMeshPipelines(r), std::array{id}, std::span{&selection, 1u}, p.Inside).front());
    if (faces.empty()) return id;
    faces.insert(faces.begin(), uint32_t(faces.size()));
    return Execute(r, MeshTopologyTask{.SourceId = id, .Op = MeshTopologyOp::FlipNormals, .List = std::move(faces), .Selection = selection}, m);
}
Outcome Run(state::Scene &r, const TransformElements &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{p.Delta, p.Pivot}}, p.Frame);
}
Outcome Run(state::Scene &r, MeshToPoints, uint32_t id, const Masks &) {
    ToPoints(r, id);
    return id;
}
Outcome Run(state::Scene &r, const SetPosition &p, uint32_t id, uint32_t attribute, const Masks &m) {
    const auto &meshes = r.Context.get<const MeshStore>();
    if (attribute == InvalidOffset || meshes.GetAttribute(attribute).Type != AttributeType::Vec3 || meshes.GetAttribute(attribute).Domain != AttributeDomain::Vertex) return std::unexpected(NodeFault::Attribute);
    return Reposition(r, id, m, PositionEditOp::Attribute, 1.f, 1u, {.Flags = p.Offset ? PositionEditAttributeOffset : 0u, .Attribute = meshes.AttributeRef(attribute)});
}

std::string_view NameOf(const AttributeName &name) { return {name.data(), strnlen(name.data(), name.size())}; }

// Stores the attribute under the name, renaming one of the evaluation's temporaries and copying any other attribute.
Outcome Store(MeshStore &meshes, const StoreNamedAttribute &p, uint32_t id, uint32_t attribute, std::vector<uint32_t> &temporaries) {
    const auto name = NameOf(p.Name);
    if (attribute == InvalidOffset) return std::unexpected(NodeFault::Attribute);
    if (name.empty()) return std::unexpected(NodeFault::Parameter);
    if (meshes.GetAttribute(attribute).NameView() == name) return id;
    if (const auto existing = meshes.FindAttribute(id, name); existing != InvalidOffset) meshes.ReleaseAttribute(existing);
    const auto temporary = std::ranges::find(temporaries, attribute);
    if (temporary == temporaries.end()) return meshes.CopyAttribute(attribute, name) ? Outcome{id} : std::unexpected(NodeFault::Attribute);
    temporaries.erase(temporary);
    return meshes.RenameAttribute(attribute, name) ? Outcome{id} : std::unexpected(NodeFault::Attribute);
}

// Writes a node's inline selection into the cook's arena over its input record's blocks, and returns each domain's first mask block.
// Mask block b of a domain holds the record's block b, so it covers every block through the record's last.
std::array<uint32_t, 3> UploadSelection(MeshStore &meshes, std::optional<BufferArena<SelectionBlock>> &arena, uint32_t id, const InlineSelection *selection) {
    if (!selection || (selection->Vertices.empty() && selection->Edges.empty() && selection->Faces.empty())) return EveryElement;
    if (!arena) arena.emplace(meshes.BufferContext(), SlotType::Buffer);
    using D = MeshStore::ElementDomain;
    const std::array domains{std::pair{D::Vertex, &selection->Vertices}, std::pair{D::Edge, &selection->Edges}, std::pair{D::Face, &selection->Faces}};
    std::array<uint32_t, 3> first;
    for (uint32_t d = 0u; d < 3u; ++d) {
        const auto owned = meshes.GetBlockList(id, domains[d].first).Blocks;
        const auto &blocks = *domains[d].second;
        const auto range = arena->Allocate(owned.empty() ? 1u : owned.back() + 1u);
        const auto masks = arena->GetMutable(range);
        std::ranges::fill(masks, SelectionBlock{});
        for (size_t i = 0u; i < std::min(owned.size(), blocks.size()); ++i) masks[owned[i]] = blocks[i];
        first[d] = range.Offset;
    }
    return first;
}

// An attribute node's input attributes, each InvalidOffset when absent.
struct Operands {
    uint32_t A, B;
};
// An attribute node's dispatch: its output's domain and type, its pipeline, and its inputs and parameters.
struct AttributeJob {
    AttributeDomain Domain;
    AttributeType Type;
    MeshPass Pass;
    AttributeNodePushConstants Pc;
};
AttributeInput NamedInput(const MeshStore &meshes, uint32_t attribute) {
    return {.Source = AttributeSource::Named, .Type = meshes.GetAttribute(attribute).Type, .Attribute = meshes.AttributeRef(attribute)};
}
// The value's bits as attribute words, zero beyond its size.
AttributeValue Words(auto value) {
    AttributeValue words{};
    std::memcpy(words.data(), &value, sizeof(value));
    return words;
}
AttributeInput ConstantInput(AttributeType type, auto value) { return {.Source = AttributeSource::Constant, .Type = type, .Value = Words(value)}; }
// The second operand, or the constant when it is absent.
AttributeInput OperandB(const MeshStore &meshes, Operands in, AttributeInput constant) { return in.B != InvalidOffset ? NamedInput(meshes, in.B) : constant; }
// The domain of A, when A is present, each present operand has one of its socket's types, and B shares A's domain.
std::optional<AttributeDomain> OperandDomain(const MeshStore &meshes, uint16_t kind, Operands in) {
    if (in.A == InvalidOffset) return {};
    const auto sockets = KindSockets(kind);
    const auto &a = meshes.GetAttribute(in.A);
    if (!(sockets[0].Types & TypeBit(a.Type))) return {};
    if (in.B == InvalidOffset) return a.Domain;
    const auto &b = meshes.GetAttribute(in.B);
    if (b.Domain != a.Domain || !(sockets[1].Types & TypeBit(b.Type))) return {};
    return a.Domain;
}

// Each attribute node's job, or none when its parameters or inputs do not fit it.
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
// Scale takes a float B, and the other operations a vec3 B.
std::optional<AttributeJob> Prepare(const MeshStore &meshes, const VectorMath &p, Operands in) {
    const auto domain = OperandDomain(meshes, KindOf<VectorMath>, in);
    const bool scale = p.Operation == VectorMathOp::Scale;
    if (!domain || (in.B != InvalidOffset && (meshes.GetAttribute(in.B).Type == AttributeType::Float) != scale)) return {};
    const bool scalar = p.Operation == VectorMathOp::Dot || p.Operation == VectorMathOp::Length || p.Operation == VectorMathOp::Distance;
    const auto constant = scale ? ConstantInput(AttributeType::Float, p.Scale) : ConstantInput(AttributeType::Vec3, p.B);
    return AttributeJob{*domain, scalar ? AttributeType::Float : AttributeType::Vec3, MeshPass::AttributeVectorMath, {.A = NamedInput(meshes, in.A), .B = OperandB(meshes, in, constant), .Operation = uint32_t(p.Operation)}};
}
std::optional<AttributeJob> Prepare(const MeshStore &meshes, const FloatMath &p, Operands in) {
    const auto domain = OperandDomain(meshes, KindOf<FloatMath>, in);
    if (!domain) return {};
    return AttributeJob{*domain, AttributeType::Float, MeshPass::AttributeFloatMath, {.A = NamedInput(meshes, in.A), .B = OperandB(meshes, in, ConstantInput(AttributeType::Float, p.B)), .Operation = uint32_t(p.Operation)}};
}
// A vector operand compares its chosen component.
std::optional<AttributeJob> Prepare(const MeshStore &meshes, const Compare &p, Operands in) {
    const auto domain = OperandDomain(meshes, KindOf<Compare>, in);
    if (!domain || p.Component > 2u) return {};
    const auto pc = AttributeNodePushConstants{
        .A = NamedInput(meshes, in.A),
        .B = OperandB(meshes, in, ConstantInput(AttributeType::Float, p.B)),
        .Operation = uint32_t(p.Operation),
        .Component = p.Component,
        .Epsilon = std::max(p.Epsilon, 0.f),
    };
    return AttributeJob{*domain, AttributeType::Bool, MeshPass::AttributeCompare, pc};
}
std::optional<AttributeJob> Prepare(const MeshStore &meshes, const BooleanMath &p, Operands in) {
    const auto domain = OperandDomain(meshes, KindOf<BooleanMath>, in);
    if (!domain) return {};
    return AttributeJob{*domain, AttributeType::Bool, MeshPass::AttributeBooleanMath, {.A = NamedInput(meshes, in.A), .B = OperandB(meshes, in, ConstantInput(AttributeType::Bool, uint32_t(p.B))), .Operation = uint32_t(p.Operation)}};
}
template<typename K>
concept Prepared = requires(const MeshStore &meshes, const K &p, Operands in) { Prepare(meshes, p, in); };

// The record's attributes a node writes, named so no user attribute takes the name.
std::string TemporaryName(uint32_t node, uint32_t socket) { return std::format(".cook.{}.{}", node, socket); }

// The attribute an input reads, or InvalidOffset when it is unlinked or its producer has none.
uint32_t LinkedAttribute(const GeometryProcedure &procedure, const CookPlan &plan, const std::vector<SocketValue> &values, uint32_t node, uint32_t input) {
    const auto link = procedure.Inputs[plan.InputFirst[node] + input];
    return link.FromNode == NoNode ? InvalidOffset : values[plan.SocketFirst[link.FromNode] + link.FromSocket].Attribute;
}

// A geometry node's attribute and selection inputs, evaluated on its input record.
struct Evaluation {
    std::vector<uint32_t> Temporaries; // The attributes the evaluation added to the record
    std::optional<std::array<uint32_t, 3>> MaskFirst; // The masks of a linked selection, complete in every domain
    bool Failed{}; // A linked selection reads no bool attribute of a selectable domain
};

// Evaluates the attribute nodes upstream of the node's inputs on record `id` in plan order, each writing a temporary attribute.
// A linked selection then scatters into its domain's masks in the cook's arena, and the other two domains derive from them.
// An attribute node whose inputs or parameters do not fit it is marked and forwards its first input.
Evaluation EvaluateInputs(state::Scene &r, const GeometryProcedure &procedure, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, std::optional<BufferArena<SelectionBlock>> &arena) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto input = [&](uint32_t n, uint32_t i) { return LinkedAttribute(procedure, plan, values, n, i); };
    auto upstream = Upstream(procedure, plan.InputFirst, node, [&](uint32_t n) { return IsAttributeKind(procedure.Nodes[n].Kind); });
    upstream[node] = false;
    Evaluation evaluation;
    std::vector<AttributeJob> jobs;
    for (const auto n : plan.Order) {
        if (!upstream[n]) continue;
        const auto [entity, kind] = procedure.Nodes[n];
        const Operands in{InputCount[kind] > 0u ? input(n, 0u) : InvalidOffset, InputCount[kind] > 1u ? input(n, 1u) : InvalidOffset};
        auto &output = values[plan.SocketFirst[n] + InputCount[kind]];
        output.Type = SocketType::Attribute;
        output.Attribute = in.A;
        if (kind == KindOf<NamedAttribute>) {
            const auto &p = r.get<const NamedAttribute>(entity);
            const auto found = meshes.FindAttribute(id, NameOf(p.Name));
            output.Attribute = found != InvalidOffset && meshes.GetAttribute(found).Type == p.Type ? found : InvalidOffset;
            if (output.Attribute == InvalidOffset) r.emplace_or_replace<NodeError>(entity, NodeFault::Attribute);
            continue;
        }
        std::optional<AttributeJob> job;
        VisitKind(kind, [&]<typename K> {
            if constexpr (Prepared<K>) job = Prepare(meshes, r.get<const K>(entity), in);
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
        const auto selection = input(node, s);
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

// Runs a geometry node on record `id` after evaluating its inputs on it, releasing the temporaries it does not store.
// Operators release them first, so the edit carries none of them to its output.
Outcome RunNode(state::Scene &r, const GeometryProcedure &procedure, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, std::optional<BufferArena<SelectionBlock>> &arena) {
    auto &meshes = r.Context.get<MeshStore>();
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
    const auto sockets = KindSockets(kind);
    const auto first = plan.SocketFirst[node];
    const auto selection_socket = uint32_t(std::ranges::find(sockets, SocketType::Selection, &Socket::Type) - sockets.begin());
    Masks masks;
    if (selection_socket < sockets.size()) {
        auto &selection = values[first + selection_socket];
        selection.Type = SocketType::Selection;
        selection.MaskFirst = evaluation.MaskFirst ? *evaluation.MaskFirst : UploadSelection(meshes, arena, id, r.try_get<const InlineSelection>(entity));
        if (const auto &f = selection.MaskFirst; f != EveryElement) masks.In = GeometryMasks{{&*arena, f[0]}, {&*arena, f[1]}, {&*arena, f[2]}};
    }
    masks.Publish = r.all_of<PublishSelection>(entity);
    Outcome edited = id;
    VisitKind(kind, [&]<typename K> {
        if constexpr (std::same_as<K, StoreNamedAttribute>) {
            edited = Store(meshes, r.get<const K>(entity), id, LinkedAttribute(procedure, plan, values, node, 1u), evaluation.Temporaries);
        } else if constexpr (std::same_as<K, SetPosition>) {
            edited = Run(r, r.get<const K>(entity), id, LinkedAttribute(procedure, plan, values, node, 1u), masks);
        } else if constexpr (requires(const K &p, const Masks &m) { Run(r, p, id, m); }) {
            release();
            edited = Run(r, r.get<const K>(entity), id, masks);
        }
    });
    release();
    return edited;
}

// The generated record's shading derives at once, since the cook's next nodes read it.
uint32_t Generate(state::Scene &r, const Generator auto &shape) {
    const auto id = CreateMesh(r, {.Data = primitive::CreateMesh(shape), .FlatShaded = true}).StoreId;
    ShadeNewMeshes(r);
    return id;
}

// Turns record `id` into points, each placing an instance rotated and scaled by the node's parameters.
void InstancePoints(state::Scene &r, const InstanceOnPoints &p, uint32_t id) {
    auto &meshes = r.Context.get<MeshStore>();
    ToPoints(r, id);
    for (const auto &[name, type, value] : {std::tuple{InstanceRotation, AttributeType::Vec4, Words(p.Rotation)}, std::tuple{InstanceScale, AttributeType::Vec3, Words(p.Scale)}}) {
        if (const auto existing = meshes.FindAttribute(id, name); existing != InvalidOffset) meshes.ReleaseAttribute(existing);
        (void)meshes.AddAttribute(id, AttributeDomain::Vertex, type, name, value);
    }
}

// The mesh entity an Object Info node reads: its object's mesh when the object has one other than the procedure's own, or null.
// A procedure-owned mesh has no object, so no node reads one.
state::Entity ReadMesh(const state::Scene &r, state::Entity mesh_entity, const ObjectInfo &p) {
    const auto read = r.valid(p.Object) ? GetMeshEntity(r, p.Object) : state::Null;
    return read != mesh_entity ? read : state::Null;
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

// Clones the instance record once per point, moves each clone by its point's instance transform, and joins the clones into the first.
// Returns none for points placing no instance.
std::optional<uint32_t> Realize(state::Scene &r, uint32_t points, uint32_t instance) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto transforms = InstanceTransforms(meshes, points);
    if (transforms.empty()) return {};
    const auto clones = Clones(r, std::vector(transforms.size(), instance));
    for (size_t i = 0u; i < clones.size(); ++i) {
        const PositionOperationTarget target{.StoreId = clones[i], .Selection = AllHandles(Mesh{meshes, clones[i]})};
        ExecuteGeometryPositions(r, std::span{&target, 1u}, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = transforms[i]}});
    }
    for (size_t i = 1u; i < clones.size(); ++i) meshes.Join(r, clones.front(), clones[i]);
    meshes.Release(std::span{clones}.subspan(1u));
    return clones.front();
}

// Publishes one unnamed instance entity of `instanced` per transform under each object instancing the mesh, as children holding the transforms.
// Each object keeps its published entities in list order, extras are destroyed and missing ones are created, so an unchanged count keeps every entity.
void PublishInstances(state::Scene &r, state::Entity mesh_entity, state::Entity instanced, std::span<const Transform> transforms) {
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
                r.emplace<ObjectKind>(e, ObjectType::Mesh);
                r.emplace<Transform>(e, transforms[i]);
                SetParent(r, e, objects[o]);
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

// Holds the instance record in the procedure's owned mesh entity, created on first need, releasing the record it held.
// Without a record, the owned mesh entity leaves with its record.
// Returns the owned mesh entity, or null.
state::Entity OwnMesh(state::Scene &r, state::Entity mesh_entity, uint32_t record) {
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
        r.Context.get<MeshStore>().Release(held);
        r.replace<MeshHandle>(e, record);
    }
    RefreshMeshBinding(r, record);
    r.emplace_or_replace<MeshGeometryDirty>(e);
    return e;
}

// Evaluates the plan's nodes in order, each on one record, and publishes the result as the mesh's evaluated record.
// The Source node emits the previous output restored to the source when the source kept its blocks, so the output record persists across cooks.
// An Object Info node emits another mesh entity's drawn record, which the cook borrows.
// An input owns the record it reads once its producer has no other consumer left, and otherwise reads a clone.
// A node that cannot run is marked with its fault and forwards its input, as does an operation that changes nothing.
// A boolean emits a new record, and its inputs release with the cook's other records.
// A missing result is the source's copy, and every other record the cook made releases.
// A result reading instances empties its record and publishes the instances as entities.
void Cook(state::Scene &r, state::Entity mesh_entity, std::optional<BufferArena<SelectionBlock>> &arena, const MeshStore::Writes &between) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto &procedure = r.get<const GeometryProcedure>(mesh_entity);
    const auto &plan = r.get<const CookPlan>(mesh_entity);
    const auto source = r.get<const MeshHandle>(mesh_entity).StoreId;
    auto evaluated = r.all_of<EvaluatedMesh>(mesh_entity) ? r.get<const EvaluatedMesh>(mesh_entity) : EvaluatedMesh{.StoreId = InvalidOffset};
    std::optional<ResyncState> resync;
    if (const auto *state = r.try_get<const ResyncState>(mesh_entity)) resync = *state;
    // The nodes the result reads through any input, which are the ones the cook evaluates and marks.
    // A node outside them is inert, and the inputs it would read count as unread.
    const auto reached = procedure.OutputNode != NoNode ? Upstream(procedure, plan.InputFirst, procedure.OutputNode, [](uint32_t) { return true; }) : std::vector<bool>(procedure.Nodes.size());
    auto remaining = plan.Consumers;
    for (uint32_t node = 0u; node < procedure.Nodes.size(); ++node) {
        if (reached[node]) continue;
        for (uint32_t i = 0u; i < InputCount[procedure.Nodes[node].Kind]; ++i) {
            if (const auto link = procedure.Inputs[plan.InputFirst[node] + i]; link.FromNode != NoNode) --remaining[plan.SocketFirst[link.FromNode] + link.FromSocket];
        }
    }
    const auto read = [&](uint32_t node) { return remaining[plan.SocketFirst[node]] > 0u; };
    const bool unlinked = procedure.OutputNode == NoNode || procedure.Inputs[plan.InputFirst[procedure.OutputNode]].FromNode == NoNode;
    const bool sourced = unlinked || std::ranges::any_of(plan.Order, [&](uint32_t node) { return procedure.Nodes[node].Kind == KindOf<Source> && read(node); });
    // The records the cook made, which release at its end apart from the result.
    std::vector<uint32_t> created;
    if (sourced) {
        created.push_back(SourceCopy(r, source, evaluated, resync, between));
        // The copy selects nothing and hides what the source hides whether it resynced or cloned anew, so the result selects only what its publishing nodes select.
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.ResetEditState(r, chain, evaluated.StoreId, source);
        chain.Submit();
    } else {
        meshes.Release(evaluated.StoreId);
        evaluated = {.StoreId = InvalidOffset};
        resync.reset();
    }
    // The source copy's restore is no change of its own.
    meshes.TakeWrites();
    // The nodes whose first input the result's record passes through, so the source's copy reaches the result in place.
    std::vector<bool> on_path(procedure.Nodes.size());
    for (auto node = procedure.OutputNode; node != NoNode && !on_path[node];) {
        on_path[node] = true;
        node = InputCount[procedure.Nodes[node].Kind] ? procedure.Inputs[plan.InputFirst[node]].FromNode : NoNode;
    }
    std::vector<SocketValue> values(plan.SocketFirst.back());
    const auto slot_of = [&](uint32_t node, uint32_t input) {
        const auto link = procedure.Inputs[plan.InputFirst[node] + input];
        return plan.SocketFirst[link.FromNode] + link.FromSocket;
    };
    // Takes the value linked into the node's input, which the input's socket then holds.
    // A borrowed record reaches a reading input as is and an editing input as a clone.
    // A record the cook owns reaches its last reader as is.
    // A reader the record's later readers follow takes a clone, apart from a reader on the result's path, which takes the record and leaves its producer a clone.
    const auto take = [&](uint32_t node, uint32_t input) -> SocketValue {
        auto &taken = values[plan.SocketFirst[node] + input];
        if (procedure.Inputs[plan.InputFirst[node] + input].FromNode == NoNode) return taken;
        const auto slot = slot_of(node, input);
        taken = values[slot];
        const bool last = --remaining[slot] == 0u;
        if (taken.Record == InvalidOffset || (taken.Borrowed() ? KindSockets(procedure.Nodes[node].Kind)[input].Reads : last)) return taken;
        const auto clone = created.emplace_back(Clone(r, taken.Record));
        if (taken.Borrowed() || !on_path[node] || input) {
            taken.MeshEntity = state::Null;
            taken.Record = clone;
            return taken;
        }
        values[slot].Record = clone;
        return taken;
    };
    // Takes a record out of the cook's ownership, so a source copy leaving it leaves the output to a new record.
    const auto disown = [&](uint32_t record) {
        std::erase(created, record);
        if (record == evaluated.StoreId) {
            evaluated = {.StoreId = InvalidOffset};
            resync.reset();
        }
    };
    // Releases a record a node consumed, apart from a borrowed record, which stays with its mesh entity.
    const auto consume = [&](const SocketValue &value) {
        if (value.Borrowed()) return;
        meshes.Release(value.Record);
        disown(value.Record);
    };
    bool copied = false;
    uint32_t result = InvalidOffset;
    std::optional<SocketValue> published; // The instances the result reads
    for (const auto node : plan.Order) {
        const auto [entity, kind] = procedure.Nodes[node];
        const auto first = plan.SocketFirst[node];
        r.remove<NodeError>(entity);
        if (!reached[node] || IsAttributeKind(kind)) continue;
        if (kind == KindOf<Source>) {
            if (!read(node)) continue;
            values[first].Record = std::exchange(copied, true) ? created.emplace_back(Clone(r, source)) : evaluated.StoreId;
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
                values[first].Record = created.emplace_back(Generate(r, r.get<const K>(entity)));
                generated = true;
            }
        });
        if (generated) continue;
        const auto in = take(node, 0u);
        const auto input = in.Record;
        if (kind == KindOf<Output>) {
            // Only the result's Output node needs an input.
            if (node == procedure.OutputNode) {
                if (input == InvalidOffset) r.emplace<NodeError>(entity, NodeFault::Unlinked);
                result = input;
                if (in.Type == SocketType::Instances) published = in;
            }
            continue;
        }
        if (kind == KindOf<Join>) {
            const auto second = take(node, 1u);
            if (input == InvalidOffset || second.Record == InvalidOffset) r.emplace<NodeError>(entity, NodeFault::Unlinked);
            if (input != InvalidOffset && second.Record != InvalidOffset) {
                meshes.Join(r, input, second.Record);
                consume(second);
            }
            values[first + 2u] = input != InvalidOffset ? in : second;
            continue;
        }
        if (kind == KindOf<Boolean>) {
            const auto second = take(node, 1u);
            const bool linked = input != InvalidOffset && second.Record != InvalidOffset;
            const auto task = linked ? BooleanTask(Mesh{meshes, input}, Mesh{meshes, second.Record}, r.get<const Boolean>(entity).Operation, mat3{1.f}, vec3{0.f}) : std::nullopt;
            const auto combined = task ? ExecuteGeometryBoolean(r, *task) : std::nullopt;
            if (combined) {
                created.push_back(*combined);
                values[first + 2u].Record = *combined;
            } else {
                r.emplace<NodeError>(entity, !linked ? NodeFault::Unlinked : !task ? NodeFault::NotClosed :
                                                                                     NodeFault::Empty);
                values[first + 2u] = in;
            }
            continue;
        }
        if (kind == KindOf<InstanceOnPoints>) {
            const auto instance = take(node, 1u);
            auto &output = values[first + 2u];
            output.Type = SocketType::Instances;
            output.Record = input;
            if (input != InvalidOffset && instance.Record != InvalidOffset) {
                InstancePoints(r, r.get<const InstanceOnPoints>(entity), input);
                output.MeshEntity = instance.MeshEntity;
                output.Instance = instance.Record;
                continue;
            }
            r.emplace<NodeError>(entity, NodeFault::Unlinked);
            if (input != InvalidOffset) Empty(r, input);
            continue;
        }
        if (kind == KindOf<RealizeInstances>) {
            if (input == InvalidOffset) r.emplace<NodeError>(entity, NodeFault::Unlinked);
            const auto realized = input != InvalidOffset && in.Instance != InvalidOffset ? Realize(r, input, in.Instance) : std::nullopt;
            if (realized) {
                consume(in);
                // The instance record releases with the last reader of the instances carrying it.
                if (remaining[slot_of(node, 0u)] == 0u) consume({.MeshEntity = in.MeshEntity, .Record = in.Instance});
                created.push_back(*realized);
            }
            values[first + 1u].Record = realized.value_or(input);
            continue;
        }
        const auto edited = input != InvalidOffset ? RunNode(r, procedure, plan, values, node, input, arena) : Outcome{std::unexpected(NodeFault::Unlinked)};
        if (!edited) r.emplace<NodeError>(entity, edited.error());
        values[first + InputCount[kind]].Record = edited.value_or(input);
    }
    // The result's points place instances of a mesh entity's drawn record, or of an instance record the procedure's owned mesh entity takes from the cook.
    std::vector<Transform> transforms;
    if (published && result != InvalidOffset) {
        transforms = InstanceTransforms(meshes, result);
        Empty(r, result);
    }
    const bool owns = !transforms.empty() && published->MeshEntity == state::Null;
    if (owns) disown(published->Instance);
    auto writes = meshes.TakeWrites();
    if (result == InvalidOffset) result = sourced && !copied ? evaluated.StoreId : created.emplace_back(Clone(r, source));
    if (result == evaluated.StoreId) {
        KeepWritten(meshes, result, *resync, writes.Blocks);
        std::erase_if(resync->Attributes, [&](uint32_t attribute) { return std::ranges::binary_search(writes.Attributes, attribute); });
        r.emplace_or_replace<ResyncState>(mesh_entity, std::move(*resync));
    } else {
        evaluated = {.StoreId = result};
        r.remove<ResyncState>(mesh_entity);
    }
    std::erase(created, result);
    meshes.Release(created);
    r.emplace_or_replace<EvaluatedMesh>(mesh_entity, evaluated);
    const auto owned = OwnMesh(r, mesh_entity, owns ? published->Instance : InvalidOffset);
    PublishInstances(r, mesh_entity, owns ? owned : published ? published->MeshEntity :
                                                                state::Null,
                     transforms);
    RefreshMeshBinding(r, result);
    // The editor derives the record's selection anew from its edit domain, which also refreshes its summary.
    r.get_or_emplace<MeshGeometryDirty>(mesh_entity, EditSelectionAfter::Derive).RenderReady = false;
}

// Socket slots and input slots in node order, the consumers of each socket slot, and an order in which every node follows the nodes it reads.
// Nodes ready together keep their authoring order.
CookPlan Plan(const GeometryProcedure &procedure) {
    const auto count = uint32_t(procedure.Nodes.size());
    CookPlan plan{.InputFirst = InputFirsts(procedure), .SocketFirst = std::vector(count + 1u, 0u)};
    for (uint32_t n = 0u; n < count; ++n) plan.SocketFirst[n + 1u] = plan.SocketFirst[n] + SocketCount[procedure.Nodes[n].Kind];
    plan.Consumers.assign(plan.SocketFirst.back(), 0u);
    // Each node's unread inputs, and its consumers as runs of one flat list.
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
    for (auto *index : {&procedure.OutputNode, &procedure.ActiveNode}) {
        if (*index == node) *index = NoNode;
        else if (*index != NoNode && *index > node) --*index;
    }
}

// A destroyed node leaves its procedure, and the inputs reading it unlink.
void RemoveNode(state::Scene &r, state::Entity node) {
    if (r.Restoring) return;
    for (const auto [mesh_entity, procedure] : std::as_const(r).view<const GeometryProcedure>().each()) {
        const auto it = std::ranges::find(procedure.Nodes, node, &NodeRecord::Entity);
        if (it == procedure.Nodes.end()) continue;
        const auto index = uint32_t(it - procedure.Nodes.begin());
        r.patch<GeometryProcedure>(mesh_entity, [&](GeometryProcedure &edited) { EraseNode(edited, index); });
        return;
    }
}

// A destroyed procedure's nodes destroy with the settle's object removals.
void QueueNodeRemovals(state::Scene &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    auto &dependents = r.Context.emplace<PendingObjectRemovals>().Dependents;
    for (const auto &node : r.get<const GeometryProcedure>(mesh_entity).Nodes) dependents.push_back(node.Entity);
}
// A dropped instance list's entities destroy with the settle's object removals.
void QueueInstanceRemovals(state::Scene &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    r.Context.emplace<PendingObjectRemovals>().Dependents.append_range(r.get<const ProcedureInstances>(mesh_entity).Entities);
}
// A dropped owned mesh entity destroys with the settle's object removals, which release its record.
void QueueOwnedMeshRemoval(state::Scene &r, state::Entity mesh_entity) {
    if (r.Restoring) return;
    r.Context.emplace<PendingObjectRemovals>().Dependents.push_back(r.get<const ProcedureOwnedMesh>(mesh_entity).MeshEntity);
}
} // namespace
} // namespace procedure

void RegisterProcedureHandlers(state::Scene &r) {
    using namespace procedure;
    r.Context.emplace<CookCount>();
    reactive(r, Change::Procedure).on<GeometryProcedure>(On::Create | On::Update);
    r.on_destroy<GeometryProcedure, &QueueNodeRemovals>();
    r.on_destroy<ProcedureInstances, &QueueInstanceRemovals>();
    r.on_destroy<ProcedureOwnedMesh, &QueueOwnedMeshRemoval>();
    auto &inputs = reactive(r, Change::ProcedureInputs);
    inputs.on<MeshHandle>(On::Create | On::Update).on<MeshGeometryDirty>(On::Create).on<MeshPositionsChanged>(On::Create);
    inputs.on<InlineSelection>(On::Create | On::Update | On::Destroy);
    // A new object of a publishing mesh takes its instances, and a published instance leaving its object returns to it.
    inputs.on<Instance>(On::Create).on<SceneParent>(On::Destroy);
    ForEachKind([&]<typename K> {
        inputs.on<K>(On::Update);
        r.on_destroy<K, &RemoveNode>();
    });
}

void SettleProcedures(state::Scene &r, bool cook) {
    using namespace procedure;
    const auto &changed = reactive(r, Change::Procedure);
    for (const auto e : changed) {
        if (const auto *procedure = r.try_get<const GeometryProcedure>(e)) r.emplace_or_replace<CookPlan>(e, Plan(*procedure));
    }
    // The writes since the last settle name the blocks each resync restores, and a restore pass restores outputs with their sources.
    const auto writes = r.Context.get<MeshStore>().TakeWrites();
    if (!cook) return;
    // The procedures whose graph or source changed, and the other entities whose change a procedure reads.
    std::vector<state::Entity> cooked, inputs;
    for (const auto e : changed) {
        if (r.all_of<GeometryProcedure, MeshHandle>(e)) cooked.push_back(e);
    }
    for (const auto e : reactive(r, Change::ProcedureInputs)) {
        if (r.all_of<GeometryProcedure, MeshHandle>(e)) cooked.push_back(e);
        else inputs.push_back(e);
    }
    if (cooked.empty() && inputs.empty()) return;
    SortUnique(inputs);
    const auto input = [&](state::Entity e) { return std::ranges::binary_search(inputs, e); };
    // Each procedure beside each mesh entity its Object Info nodes read.
    std::vector<std::pair<state::Entity, state::Entity>> reads;
    for (const auto [e, procedure] : std::as_const(r).view<const GeometryProcedure>().each()) {
        if (!r.all_of<MeshHandle>(e)) continue;
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
    // A procedure reading a changed or cooking mesh cooks after it.
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
    // A cook reads the meshes created since the last settle, such as a regenerated primitive source, so their shading derives first.
    if (!order.empty()) ShadeNewMeshes(r);
    std::optional<BufferArena<SelectionBlock>> arena;
    for (const auto e : order) Cook(r, e, arena, writes);
    r.Context.get<CookCount>().Value += order.size();
}
