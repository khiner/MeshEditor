#include "procedure/Cook.h"
#include "procedure/Procedure.h"
#include "procedure/ProcedureSchema.h"

#include "SortUnique.h"
#include "mesh/Decimate.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/Primitives.h"
#include "mesh/RecalculateNormals.h"
#include "numeric/MatrixMath.h"
#include "object/ObjectOps.h"
#include "object/PendingSync.h"
#include "render/Instance.h"
#include "render/MeshletBuildGpu.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "viewport/ViewportEvents.h"

#include <format>

using state::On;

namespace procedure {
namespace {
using namespace action::mesh;
// Optional domain masks; absent masks select every element.
// Publish copies emitted selection into the output record.
struct Masks {
    std::optional<GeometryMasks> In;
    bool Publish{};
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

// Copy canonical records only; cooking rebuilds render records for the output.
std::vector<uint32_t> Clones(state::Registry &r, std::span<const uint32_t> ids) {
    auto &meshes = r.Resources.get<MeshStore>();
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
uint32_t Clone(state::Registry &r, uint32_t id) { return Clones(r, std::array{id}).front(); }

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

// Reuse the prior Source output by restoring changed blocks and table entries from the source.
uint32_t SourceCopy(state::Registry &r, uint32_t source, EvaluatedMesh &evaluated, std::optional<ResyncState> &resync, const MeshStore::Writes &between) {
    auto &meshes = r.Resources.get<MeshStore>();
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
// Return the output record or a node fault.
Outcome Execute(state::Registry &r, std::optional<MeshTopologyTask> task, const Masks &masks) {
    if (!task) return std::unexpected(NodeFault::Selection);
    if (masks.Publish) PublishStoreSelection(r.Resources.get<const MeshStore>(), *task);
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
    const auto results = ExecuteGeometryTopologyStages(r, tasks);
    if (std::ranges::none_of(results, &GeometryTopologyResult::Changed)) return tasks.front().SourceId;
    return results.back().GeometryId;
}
// Publish the input selection after position edits, which preserve membership.
Outcome Reposition(state::Registry &r, uint32_t id, const Masks &masks, PositionEditOp op, float factor, uint32_t repeat, const PositionOperationOptions &options = {}, const Transform &world = {}) {
    auto &meshes = r.Resources.get<MeshStore>();
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
// Nodes lack face edit mode, so limited dissolve reads all selected domains.
Outcome Run(state::Registry &r, const Dissolve &p, uint32_t id, const Masks &m) {
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
// Use vertex normals for node offsets, independently of viewport selection mode.
Outcome Run(state::Registry &r, const ShrinkFatten &p, uint32_t id, const Masks &m, const Linked &linked) {
    return RepositionBy(r, id, m, PositionEditOp::ShrinkFatten, p.Distance, 1u, linked.Of<&ShrinkFatten::Distance>(), {.Flags = p.Even ? PositionEditEvenOffset : 0u});
}
Outcome Run(state::Registry &r, const Randomize &p, uint32_t id, const Masks &m, const Linked &linked) {
    return RepositionBy(r, id, m, PositionEditOp::Randomize, p.Amount, 1u, linked.Of<&Randomize::Amount>(), {.Randomize = PositionRandomize{p.Uniform, p.Normal, p.Seed}});
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
Outcome Run(state::Registry &r, const TransformGeometry &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{p.Delta, vec3{}}});
}
Outcome Run(state::Registry &r, const TransformElements &p, uint32_t id, const Masks &m) {
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
        const auto range = arena->Allocate(owned.empty() ? 1u : owned.back() + 1u);
        const auto masks = arena->GetMutable(range);
        std::ranges::fill(masks, SelectionBlock{});
        fill(d, owned, masks);
        first[d] = range.Offset;
    }
    return first;
}
std::array<uint32_t, 3> UploadSelection(MeshStore &meshes, std::optional<BufferArena<SelectionBlock>> &arena, uint32_t id, const InlineSelection *selection) {
    if (!selection || (selection->Vertices.empty() && selection->Edges.empty() && selection->Faces.empty())) return EveryElement;
    const std::array lists{&selection->Vertices, &selection->Edges, &selection->Faces};
    return ZeroedMasks(meshes, arena, id, [&](uint32_t d, std::span<const uint32_t> owned, std::span<SelectionBlock> masks) {
        const auto &blocks = *lists[d];
        for (size_t i = 0u; i < std::min(owned.size(), blocks.size()); ++i) masks[owned[i]] = blocks[i];
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
Outcome RunNode(state::Registry &r, const GeometryProcedure &procedure, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, std::optional<BufferArena<SelectionBlock>> &arena) {
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
        selection.MaskFirst = evaluation.MaskFirst ? *evaluation.MaskFirst : UploadSelection(meshes, arena, id, r.try_get<const InlineSelection>(entity));
        if (const auto &f = selection.MaskFirst; f != EveryElement) masks.In = GeometryMasks{{&*arena, f[0]}, {&*arena, f[1]}, {&*arena, f[2]}};
    }
    masks.Publish = r.all_of<PublishSelection>(entity);
    Outcome edited = id;
    VisitKind(kind, [&]<typename K> {
        const auto p = Parameters<K>(r, procedure, plan, values, node);
        if constexpr (std::same_as<K, StoreNamedAttribute>) {
            edited = Store(meshes, p, id, OperandOf(procedure, plan, values, node, 1u), evaluation.Temporaries);
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

// Return the referenced object mesh, excluding the current procedure mesh and meshes without objects.
state::Entity ReadMesh(const state::Registry &r, state::Entity mesh_entity, const ObjectInfo &p) {
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

// Realize instances into one mesh; return nullopt for an empty result.
std::optional<uint32_t> Realize(state::Registry &r, uint32_t points, uint32_t instance) {
    auto &meshes = r.Resources.get<MeshStore>();
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

// Evaluate nodes in dependency order and publish the result.
// Mark failed nodes and pass their inputs through.
// Release intermediate records; retain the source copy when no result is linked.
void Cook(state::Registry &r, state::Entity mesh_entity, std::optional<BufferArena<SelectionBlock>> &arena, const MeshStore::Writes &between) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto &procedure = r.get<const GeometryProcedure>(mesh_entity);
    const auto &plan = r.get<const CookPlan>(mesh_entity);
    const auto source = r.get<const MeshHandle>(mesh_entity).StoreId;
    auto evaluated = r.all_of<EvaluatedMesh>(mesh_entity) ? r.get<const EvaluatedMesh>(mesh_entity) : EvaluatedMesh{.StoreId = InvalidOffset};
    std::optional<ResyncState> resync;
    if (const auto *state = r.try_get<const ResyncState>(mesh_entity)) resync = *state;
    // Evaluate only nodes reachable from the output.
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
    // Release temporary cook records at completion and retain the result.
    std::vector<uint32_t> created;
    if (sourced) {
        created.push_back(SourceCopy(r, source, evaluated, resync, between));
        // Clear copied selection and preserve source visibility; publishing nodes determine result selection.
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.ResetEditState(r, chain, evaluated.StoreId, source);
        chain.Submit();
    } else {
        meshes.Release(evaluated.StoreId);
        evaluated = {.StoreId = InvalidOffset};
        resync.reset();
    }
    // Exclude source restoration from cook writes.
    meshes.TakeWrites();
    // Track the first-input path to preserve the source-copy record through the result.
    std::vector<bool> on_path(procedure.Nodes.size());
    for (auto node = procedure.OutputNode; node != NoNode && !on_path[node];) {
        on_path[node] = true;
        node = KindSockets(procedure.Nodes[node].Kind).front().Type == SocketType::Geometry ? procedure.Inputs[plan.InputFirst[node]].FromNode : NoNode;
    }
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
        const auto clone = created.emplace_back(Clone(r, taken.Record));
        if (taken.Borrowed() || !on_path[node] || input) {
            taken.MeshEntity = state::Null;
            taken.Record = clone;
            return taken;
        }
        values[slot].Record = clone;
        return taken;
    };
    const auto disown = [&](uint32_t record) {
        std::erase(created, record);
        if (record == evaluated.StoreId) {
            evaluated = {.StoreId = InvalidOffset};
            resync.reset();
        }
    };
    // Release a consumed record only when owned by this cook; borrowed records remain owned by their mesh entities.
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
        if (kind == KindOf<Output>) {
            // Use the Output node result, or a source copy when the Output input is unlinked.
            if (node == procedure.OutputNode) {
                result = input;
                if (in.Type == SocketType::Instances) published = in;
            }
            continue;
        }
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
            const bool linked = input != InvalidOffset && second.Record != InvalidOffset;
            const auto task = linked ? BooleanTask(Mesh{meshes, input}, Mesh{meshes, second.Record}, r.get<const Boolean>(entity).Operation, mat3{1.f}, vec3{0.f}) : std::nullopt;
            const auto combined = task ? ExecuteGeometryBoolean(r, *task) : std::nullopt;
            if (combined) {
                created.push_back(*combined);
                values[first + 2u].Record = *combined;
            } else {
                if (linked) r.emplace<NodeError>(entity, !task ? NodeFault::NotClosed : NodeFault::Empty);
                values[first + 2u] = in;
            }
            continue;
        }
        if (kind == KindOf<InstanceOnPoints>) {
            const auto instance = take(node, 1u);
            auto &output = values[first + InputCount[kind]];
            output.Type = SocketType::Instances;
            output.Record = input;
            if (input != InvalidOffset && instance.Record != InvalidOffset) {
                if (const auto placed = RunNode(r, procedure, plan, values, node, input, arena); !placed) r.emplace<NodeError>(entity, placed.error());
                output.MeshEntity = instance.MeshEntity;
                output.Instance = instance.Record;
                continue;
            }
            if (input != InvalidOffset) Empty(r, input);
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
    // Recompute selection and its summary from the edit domain.
    r.get_or_emplace<MeshGeometryDirty>(mesh_entity, EditSelectionAfter::Derive).RenderReady = false;
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
    for (auto *index : {&procedure.OutputNode, &procedure.ActiveNode}) {
        if (*index == node) *index = NoNode;
        else if (*index != NoNode && *index > node) --*index;
    }
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
    state::DirtySet Inputs; // A procedure's source record, a node's parameters or inline selection, a mesh's objects or a published instance's parent changed.
};
} // namespace procedure

void RegisterProcedureHandlers(state::Registry &r) {
    using namespace procedure;
    r.Resources.emplace<CookCount>();
    auto &changes = r.Resources.emplace<ProcedureChanges>();
    changes.Graph.bind(r).on<GeometryProcedure>(On::Create | On::Update);
    r.on_destroy<GeometryProcedure, &QueueNodeRemovals>();
    r.on_destroy<ProcedureInstances, &QueueInstanceRemovals>();
    r.on_destroy<ProcedureOwnedMesh, &QueueOwnedMeshRemoval>();
    auto &inputs = changes.Inputs.bind(r);
    inputs.on<MeshHandle>(On::Create | On::Update).on<MeshGeometryDirty>(On::Create).on<MeshPositionsChanged>(On::Create);
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
    // Track changed blocks for resynchronization; history restores both sources and outputs.
    const auto writes = r.Resources.get<MeshStore>().TakeWrites();
    if (!cook) return;
    std::vector<state::Entity> cooked, inputs;
    for (const auto e : changed) {
        if (r.all_of<GeometryProcedure, MeshHandle>(e)) cooked.push_back(e);
    }
    for (const auto e : changes.Inputs) {
        if (r.all_of<GeometryProcedure, MeshHandle>(e)) cooked.push_back(e);
        else inputs.push_back(e);
    }
    if (cooked.empty() && inputs.empty()) return;
    SortUnique(inputs);
    const auto input = [&](state::Entity e) { return std::ranges::binary_search(inputs, e); };
    // Index procedures by the mesh entities read by their Object Info nodes.
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
    // Derive new-mesh shading before cooking dependent procedures.
    if (!order.empty()) ShadeNewMeshes(r);
    std::optional<BufferArena<SelectionBlock>> arena;
    for (const auto e : order) Cook(r, e, arena, writes);
    r.Resources.get<CookCount>().Value += order.size();
}
