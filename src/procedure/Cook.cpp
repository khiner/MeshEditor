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
#include "object/PendingSync.h"
#include "render/MeshletBuildGpu.h"
#include "state/Scene.h"
#include "viewport/ViewportEvents.h"

#include <cstring>
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

// A socket's value during a cook: a record for geometry, each domain's first mask block in the cook's arena for a selection, or an attribute of the record its consumer reads.
// A selection without mask blocks selects every element, and an absent attribute is InvalidOffset.
struct SocketValue {
    SocketType Type{SocketType::Geometry};
    union {
        uint32_t Record{InvalidOffset};
        std::array<uint32_t, 3> MaskFirst;
        uint32_t Attribute;
    };
};
static_assert(sizeof(SocketValue) == 16u);
constexpr std::array<uint32_t, 3> EveryElement{InvalidOffset, InvalidOffset, InvalidOffset};

// Records the copies and runs them.
void Submit(state::Scene &r, CloneCopies &copies) {
    mtl::ComputeChain chain{r.Context.get<MeshStore>().BufferContext()};
    copies.Encode(chain, GetMeshPipelines(r));
    chain.Submit();
}
// A copy of the record without render records, since a cook output builds them anew, and its block pairs from the record when `pairs` holds one list.
uint32_t Clone(state::Scene &r, uint32_t id, std::span<MeshStore::BlockPairs> pairs) {
    auto &meshes = r.Context.get<MeshStore>();
    meshes.PlanClone(Mesh{meshes, id});
    meshes.CommitReserves();
    CloneCopies copies;
    const auto clone = meshes.CloneMeshes(copies, std::array{id}, false, pairs).front();
    Submit(r, copies);
    meshes.RebuildSelectionIndex(std::array{clone});
    return clone;
}

// The source's copy that the Source node emits, which `evaluated` and `resync` then describe.
// With resync state, the previous output restores to the source at the blocks either wrote since its last cook.
// Otherwise, or when the source gained or lost blocks, the previous output releases for a new copy.
uint32_t SourceCopy(state::Scene &r, uint32_t source, EvaluatedMesh &evaluated, std::optional<ResyncState> &resync, const MeshStore::Writes &between) {
    auto &meshes = r.Context.get<MeshStore>();
    if (resync && evaluated.Source == source && meshes.TryGet(evaluated.StoreId)) {
        auto written = resync->Written;
        for (uint32_t d = 0u; d < written.size(); ++d) {
            written[d].append_range(between.Blocks[d]);
            SortUnique(written[d]);
        }
        CloneCopies copies;
        if (const auto change = meshes.Resync(copies, evaluated.StoreId, source, resync->Blocks, written, between.Attributes, resync->Attributes)) {
            mtl::ComputeChain chain{meshes.BufferContext()};
            copies.Encode(chain, GetMeshPipelines(r));
            chain.Submit();
            meshes.ReconcileSelection(r, chain, std::span{&*change, 1u});
            chain.Submit();
            ++resync->Resyncs;
            return evaluated.StoreId;
        }
    }
    meshes.Release(evaluated.StoreId);
    resync.emplace();
    evaluated = {.StoreId = Clone(r, source, std::span{&resync->Blocks, 1u}), .Source = source};
    meshes.ForEachNamedAttribute(evaluated.StoreId, [&](uint32_t attribute, const auto &) { resync->Attributes.push_back(attribute); });
    return evaluated.StoreId;
}

// Keeps the written blocks the record still holds or pairs with its source.
void KeepWritten(const MeshStore &meshes, uint32_t id, ResyncState &resync, MeshStore::DomainBlocks written) {
    const auto &record = meshes.Get(id);
    for (const auto domain : MeshStore::ElementDomains) {
        const auto d = uint32_t(domain);
        const auto &pairs = resync.Blocks[d];
        const auto owner = DomainSet(record, domain).Index;
        WithDomain(meshes.Arenas(), domain, [&](const auto &arena) {
            const auto members = arena.Blocks.Buffer.template GetSpan<MeshElementBlock>();
            std::erase_if(written[d], [&](uint32_t block) {
                const auto paired = std::ranges::lower_bound(pairs, block, {}, &uvec2::y);
                const bool owned = owner != InvalidOffset && block < members.size() && members[block].Owner == owner;
                return !owned && (paired == pairs.end() || paired->y != block);
            });
        });
        resync.Written[d] = std::move(written[d]);
    }
}

GeometrySelection Handles(const MeshStore &meshes, uint32_t id, const Masks &masks) { return masks.In ? ReadMasks(meshes, id, *masks.In) : AllHandles(Mesh{meshes, id}); }

MeshTopologyTask Task(uint32_t id, MeshTopologyOp op, const Masks &masks, float param0 = 0.f, float param1 = 0.f, uint32_t flags = 0u) {
    MeshTopologyTask task{.SourceId = id, .Op = op, .Param0 = param0, .Param1 = param1, .Flags = flags};
    if (masks.In) task.Selection = *masks.In;
    else task.Flags |= TopologyFlagSelectAll;
    return task;
}
// The record a task leaves, or none when it changed nothing.
std::optional<uint32_t> Execute(state::Scene &r, std::optional<MeshTopologyTask> task, const Masks &masks) {
    if (!task) return {};
    if (masks.Publish) PublishStoreSelection(r.Context.get<const MeshStore>(), *task);
    const auto result = ExecuteGeometryTopology(r, std::span{&*task, 1u}).front();
    return result.Changed ? std::optional{result.GeometryId} : std::nullopt;
}
// A planned primitive list takes the handles it was planned from as its selection.
std::optional<uint32_t> ExecutePlanned(state::Scene &r, std::optional<MeshTopologyTask> task, const GeometrySelection &selection, const Masks &masks) {
    if (task && task->Op == MeshTopologyOp::AddPrimitives) task->Selection = selection;
    return Execute(r, std::move(task), masks);
}
std::optional<uint32_t> ExecuteStages(state::Scene &r, std::vector<MeshTopologyTask> tasks, const Masks &masks) {
    if (masks.Publish)
        for (auto &task : tasks) PublishStoreSelection(r.Context.get<const MeshStore>(), task);
    const auto results = ExecuteGeometryTopologyStages(r, tasks);
    if (std::ranges::none_of(results, &GeometryTopologyResult::Changed)) return {};
    return results.back().GeometryId;
}
// A publishing position edit keeps every element, so its record selects the edit's selection.
std::optional<uint32_t> Reposition(state::Scene &r, uint32_t id, const Masks &masks, PositionEditOp op, float factor, uint32_t repeat, const PositionOperationOptions &options = {}, const Transform &world = {}) {
    auto &meshes = r.Context.get<MeshStore>();
    const PositionOperationTarget target{.StoreId = id, .Selection = masks.In ? SelectionSource{*masks.In} : SelectionSource{AllHandles(Mesh{meshes, id})}, .World = world};
    if (ExecuteGeometryPositions(r, std::span{&target, 1u}, op, factor, repeat, options).empty()) return {};
    if (masks.Publish && masks.In) {
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.SelectMasks(r, chain, id, *masks.In);
        chain.Submit();
    }
    return id;
}
uint32_t Axes(bool x, bool y, bool z) { return uint32_t(x) | uint32_t(y) << 1u | uint32_t(z) << 2u; }

// Each operator edits record `id` where `masks` select, or every element without masks.
std::optional<uint32_t> Run(state::Scene &r, const Delete &p, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp(uint32_t(p.Mode)), m), m); }
std::optional<uint32_t> Run(state::Scene &r, const Merge &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    const auto &vertices = selection.Vertices;
    const auto reference = vertices.empty() ? InvalidOffset : p.Mode == MergeMode::Last ? vertices.back() :
                                                                                          vertices.front();
    return ExecutePlanned(r, MergeTask(meshes, Mesh{meshes, id}, selection, p.Mode, std::max(p.Distance, 0.f), reference), selection, m);
}
std::optional<uint32_t> Run(state::Scene &r, const Extrude &p, uint32_t id, const Masks &m) {
    const auto op = p.Mode == ExtrudeMode::Vertices ? MeshTopologyOp::ExtrudeVertices : p.Mode == ExtrudeMode::Edges ? MeshTopologyOp::ExtrudeEdges :
        p.Mode == ExtrudeMode::FacesIndividual                                                                       ? MeshTopologyOp::ExtrudeFacesIndividual :
                                                                                                                       MeshTopologyOp::ExtrudeRegion;
    return Execute(r, Task(id, op, m), m);
}
std::optional<uint32_t> Run(state::Scene &r, Duplicate, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::DuplicateGeometry, m), m); }
std::optional<uint32_t> Run(state::Scene &r, Split, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::SplitGeometry, m), m); }
// A limited dissolve reads every selected domain, since a node has no face edit mode to protect unselected faces by.
std::optional<uint32_t> Run(state::Scene &r, const Dissolve &p, uint32_t id, const Masks &m) {
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
std::optional<uint32_t> Run(state::Scene &r, const Subdivide &p, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::Subdivide, m, float(std::max(p.Cuts, 1u))), m); }
std::optional<uint32_t> Run(state::Scene &r, Triangulate, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::Triangulate, m), m); }
std::optional<uint32_t> Run(state::Scene &r, TrisToQuads, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::TrisToQuads, m), m); }
std::optional<uint32_t> Run(state::Scene &r, const Poke &p, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::Poke, m, p.Offset), m); }
std::optional<uint32_t> Run(state::Scene &r, FlipNormals, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::FlipNormals, m), m); }
std::optional<uint32_t> Run(state::Scene &r, EdgeSplit, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::EdgeSplit, m), m); }
std::optional<uint32_t> Run(state::Scene &r, Rip, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::EdgeSplit, m, 0.f, 0.f, TopologyFlagRipSelectCopies), m); }
std::optional<uint32_t> Run(state::Scene &r, const Inset &p, uint32_t id, const Masks &m) {
    return Execute(r, Task(id, p.Individual ? MeshTopologyOp::InsetIndividual : MeshTopologyOp::InsetRegion, m, std::max(p.Thickness, 0.f), p.Depth, p.Even ? TopologyFlagEvenOffset : 0u), m);
}
std::optional<uint32_t> Run(state::Scene &r, const Solidify &p, uint32_t id, const Masks &m) { return Execute(r, Task(id, MeshTopologyOp::Solidify, m, p.Thickness), m); }
std::optional<uint32_t> Run(state::Scene &r, const Wireframe &p, uint32_t id, const Masks &m) {
    const auto flags = (p.Even ? TopologyFlagEvenOffset : 0u) | (p.Boundary ? TopologyFlagWireBoundary : 0u) | (p.Relative ? TopologyFlagWireRelative : 0u) | (p.Replace ? TopologyFlagWireReplace : 0u);
    return Execute(r, Task(id, MeshTopologyOp::Wireframe, m, std::max(p.Thickness, 0.f), std::clamp(p.Offset, -1.f, 1.f), flags), m);
}
std::optional<uint32_t> Run(state::Scene &r, const Bevel &p, uint32_t id, const Masks &m) {
    auto task = Task(id, p.Vertices ? MeshTopologyOp::BevelVertices : MeshTopologyOp::BevelEdges, m, std::max(p.Width, 0.f));
    task.Steps = std::max(p.Segments, 1u);
    return Execute(r, task, m);
}
std::optional<uint32_t> Run(state::Scene &r, const SmoothVertices &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::Smooth, p.Factor, std::clamp(p.Repeat, 1u, 1000u), {.Axes = Axes(p.X, p.Y, p.Z)});
}
// Offsets follow vertex normals, since a node has no face edit mode to follow selected face normals by.
std::optional<uint32_t> Run(state::Scene &r, const ShrinkFatten &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::ShrinkFatten, p.Distance, 1u, {.Flags = p.Even ? PositionEditEvenOffset : 0u});
}
std::optional<uint32_t> Run(state::Scene &r, const Randomize &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::Randomize, p.Amount, 1u, {.Randomize = PositionRandomize{p.Uniform, p.Normal, p.Seed}});
}
std::optional<uint32_t> Run(state::Scene &r, const Spin &p, uint32_t id, const Masks &m) {
    if (Dot(p.Axis, p.Axis) <= 0.f) return {};
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    const auto axis = Normalize(p.Axis);
    const auto steps = float(std::max(p.Steps, 1u));
    return ExecutePlanned(r, ExtrudeStepsTask(Mesh{meshes, id}, selection, Element::None, p.Steps, ToMat3(AngleAxis(p.Angle / steps, axis)), axis * (p.Offset / steps), p.Center), selection, m);
}
std::optional<uint32_t> Run(state::Scene &r, const ExtrudeRepeat &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    return ExecutePlanned(r, ExtrudeStepsTask(Mesh{meshes, id}, selection, Element::None, p.Steps, mat3{1.f}, p.Offset, vec3{0.f}), selection, m);
}
std::optional<uint32_t> Run(state::Scene &r, const Bisect &p, uint32_t id, const Masks &m) {
    if (Dot(p.Normal, p.Normal) <= 0.f) return {};
    return ExecuteStages(r, BisectTasks(Mesh{r.Context.get<const MeshStore>(), id}, p.Point, p.Normal, p.ClearInner, p.ClearOuter), m);
}
std::optional<uint32_t> Run(state::Scene &r, const Symmetrize &p, uint32_t id, const Masks &m) {
    return ExecuteStages(r, SymmetrizeTasks(Mesh{r.Context.get<const MeshStore>(), id}, uint8_t(p.Axis), p.Negative), m);
}
std::optional<uint32_t> Run(state::Scene &r, const FillHoles &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Context.get<const MeshStore>();
    return Execute(r, FillHolesTask(meshes, Mesh{meshes, id}, p.Sides), m);
}
std::optional<uint32_t> Run(state::Scene &r, ConvexHull, uint32_t id, const Masks &m) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    return ExecutePlanned(r, ConvexHullTask(meshes, Mesh{meshes, id}, selection), selection, m);
}
std::optional<uint32_t> Run(state::Scene &r, const Decimate &p, uint32_t id, const Masks &m) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto selection = Handles(meshes, id, m);
    return ExecutePlanned(r, DecimateTask(meshes, Mesh{meshes, id}, selection, p.Ratio), selection, m);
}
std::optional<uint32_t> Run(state::Scene &r, const RecalculateNormals &p, uint32_t id, const Masks &m) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto selection = Handles(meshes, id, m);
    auto faces = std::move(RecalculateFaceFlips(meshes, GetMeshPipelines(r), std::array{id}, std::span{&selection, 1u}, p.Inside).front());
    if (faces.empty()) return {};
    faces.insert(faces.begin(), uint32_t(faces.size()));
    return Execute(r, MeshTopologyTask{.SourceId = id, .Op = MeshTopologyOp::FlipNormals, .List = std::move(faces), .Selection = selection}, m);
}
std::optional<uint32_t> Run(state::Scene &r, const TransformElements &p, uint32_t id, const Masks &m) {
    return Reposition(r, id, m, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{p.Delta, p.Pivot}}, p.Frame);
}
std::optional<uint32_t> Run(state::Scene &r, const SetPosition &p, uint32_t id, uint32_t attribute, const Masks &m) {
    const auto &meshes = r.Context.get<const MeshStore>();
    if (attribute == InvalidOffset || meshes.GetAttribute(attribute).Type != AttributeType::Vec3 || meshes.GetAttribute(attribute).Domain != AttributeDomain::Vertex) return {};
    return Reposition(r, id, m, PositionEditOp::Attribute, 1.f, 1u, {.Flags = p.Offset ? PositionEditAttributeOffset : 0u, .Attribute = meshes.AttributeRef(attribute)});
}

std::string_view NameOf(const AttributeName &name) { return {name.data(), strnlen(name.data(), name.size())}; }

// Stores the attribute under the name, renaming one of the evaluation's temporaries and copying any other attribute.
std::optional<uint32_t> Store(MeshStore &meshes, const StoreNamedAttribute &p, uint32_t id, uint32_t attribute, std::vector<uint32_t> &temporaries) {
    const auto name = NameOf(p.Name);
    if (attribute == InvalidOffset || name.empty()) return {};
    if (meshes.GetAttribute(attribute).NameView() == name) return id;
    if (const auto existing = meshes.FindAttribute(id, name); existing != InvalidOffset) meshes.ReleaseAttribute(existing);
    const auto temporary = std::ranges::find(temporaries, attribute);
    if (temporary == temporaries.end()) return meshes.CopyAttribute(attribute, name) ? std::optional{id} : std::nullopt;
    temporaries.erase(temporary);
    return meshes.RenameAttribute(attribute, name) ? std::optional{id} : std::nullopt;
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
        auto masks = arena->GetMutable(range);
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
AttributeInput ConstantInput(AttributeType type, auto value) {
    std::array<uint32_t, 4> words{};
    std::memcpy(words.data(), &value, sizeof(value));
    return {.Source = AttributeSource::Constant, .Type = type, .Value = words};
}
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
            if (output.Attribute == InvalidOffset) r.emplace_or_replace<NodeError>(entity);
            continue;
        }
        std::optional<AttributeJob> job;
        VisitKind(kind, [&]<typename K> {
            if constexpr (Prepared<K>) job = Prepare(meshes, r.get<const K>(entity), in);
        });
        if (!job) {
            r.emplace_or_replace<NodeError>(entity);
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
            scatters.push_back({.Attribute = ref, .TableCount = scattered ? attribute->Table.Count : 0u, .Masks = MaskBinding({&*arena, mask_ranges[d].Offset}), .Words = mask_ranges[d].Count * MeshElementBlockWords});
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
std::optional<uint32_t> RunNode(state::Scene &r, const GeometryProcedure &procedure, const CookPlan &plan, std::vector<SocketValue> &values, uint32_t node, uint32_t id, std::optional<BufferArena<SelectionBlock>> &arena) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto [entity, kind] = procedure.Nodes[node];
    auto evaluation = EvaluateInputs(r, procedure, plan, values, node, id, arena);
    const auto release = [&] {
        for (const auto attribute : evaluation.Temporaries) meshes.ReleaseAttribute(attribute);
        evaluation.Temporaries.clear();
    };
    if (evaluation.Failed) {
        release();
        return {};
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
    std::optional<uint32_t> edited;
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

uint32_t Generate(state::Scene &r, const Generator auto &shape) { return CreateMesh(r, {.Data = primitive::CreateMesh(shape), .FlatShaded = true}).StoreId; }

// Evaluates the plan's nodes in order, each on one record, and publishes the result as the mesh's evaluated record.
// The Source node emits the previous output restored to the source when the source kept its blocks, so the output record persists across cooks.
// An input owns the record it reads once its producer has no other consumer left, and otherwise reads a clone.
// A node without an input record, or whose operation changes nothing, is marked and forwards its input.
// A boolean emits a new record, and its inputs release with the cook's other records.
// A missing result is the source's copy, and every other record the cook made releases.
void Cook(state::Scene &r, state::Entity mesh_entity, std::optional<BufferArena<SelectionBlock>> &arena, const MeshStore::Writes &between) {
    auto &meshes = r.Context.get<MeshStore>();
    const auto &procedure = r.get<const GeometryProcedure>(mesh_entity);
    const auto &plan = r.get<const CookPlan>(mesh_entity);
    const auto source = r.get<const MeshHandle>(mesh_entity).StoreId;
    auto evaluated = r.all_of<EvaluatedMesh>(mesh_entity) ? r.get<const EvaluatedMesh>(mesh_entity) : EvaluatedMesh{.StoreId = InvalidOffset};
    std::optional<ResyncState> resync;
    if (const auto *state = r.try_get<const ResyncState>(mesh_entity)) resync = *state;
    const auto read = [&](uint32_t node) { return plan.Consumers[plan.SocketFirst[node]] > 0u; };
    const bool unlinked = procedure.OutputNode == NoNode || procedure.Inputs[plan.InputFirst[procedure.OutputNode]].FromNode == NoNode;
    const bool sourced = unlinked || std::ranges::any_of(plan.Order, [&](uint32_t node) { return procedure.Nodes[node].Kind == KindOf<Source> && read(node); });
    // The records the cook made, which release at its end apart from the result.
    std::vector<uint32_t> created;
    if (sourced) {
        created.push_back(SourceCopy(r, source, evaluated, resync, between));
        // The copy selects nothing and hides what the source hides whether it resynced or cloned anew, so the result selects only what its publishing nodes select.
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.ResetEditState(r, chain, evaluated.StoreId, resync->Blocks);
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
    auto remaining = plan.Consumers;
    // A reader the record's later readers follow takes a clone, apart from a reader on the result's path, which takes the record and leaves its producer a clone.
    const auto take = [&](uint32_t node, uint32_t input) {
        const auto link = procedure.Inputs[plan.InputFirst[node] + input];
        if (link.FromNode == NoNode) return InvalidOffset;
        const auto slot = plan.SocketFirst[link.FromNode] + link.FromSocket;
        const auto record = values[slot].Record;
        if (--remaining[slot] == 0u || record == InvalidOffset) return record;
        const auto clone = created.emplace_back(Clone(r, record, {}));
        if (!on_path[node] || input) return clone;
        values[slot].Record = clone;
        return record;
    };
    bool copied = false;
    uint32_t result = InvalidOffset;
    for (const auto node : plan.Order) {
        const auto [entity, kind] = procedure.Nodes[node];
        const auto first = plan.SocketFirst[node];
        r.remove<NodeError>(entity);
        if (IsAttributeKind(kind)) continue;
        if (kind == KindOf<Source>) {
            if (!read(node)) continue;
            values[first].Record = std::exchange(copied, true) ? created.emplace_back(Clone(r, source, {})) : evaluated.StoreId;
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
        const auto input = take(node, 0u);
        values[first].Record = input;
        if (kind == KindOf<Output>) {
            if (input == InvalidOffset) r.emplace<NodeError>(entity);
            if (node == procedure.OutputNode) result = input;
            continue;
        }
        if (kind == KindOf<Join>) {
            const auto second = take(node, 1u);
            values[first + 1u].Record = second;
            if (input == InvalidOffset || second == InvalidOffset) r.emplace<NodeError>(entity);
            if (input != InvalidOffset && second != InvalidOffset) {
                meshes.Join(r, input, second);
                meshes.Release(second);
                std::erase(created, second);
                if (second == evaluated.StoreId) {
                    evaluated = {.StoreId = InvalidOffset};
                    resync.reset();
                }
            }
            values[first + 2u].Record = input != InvalidOffset ? input : second;
            continue;
        }
        if (kind == KindOf<Boolean>) {
            const auto second = take(node, 1u);
            values[first + 1u].Record = second;
            const auto task = input != InvalidOffset && second != InvalidOffset ? BooleanTask(Mesh{meshes, input}, Mesh{meshes, second}, r.get<const Boolean>(entity).Operation, mat3{1.f}, vec3{0.f}) : std::nullopt;
            const auto combined = task ? ExecuteGeometryBoolean(r, *task) : std::nullopt;
            if (combined) created.push_back(*combined);
            else r.emplace<NodeError>(entity);
            values[first + 2u].Record = combined.value_or(input);
            continue;
        }
        if (input == InvalidOffset) r.emplace<NodeError>(entity);
        const auto edited = input != InvalidOffset ? RunNode(r, procedure, plan, values, node, input, arena) : std::nullopt;
        if (input != InvalidOffset && !edited) r.emplace<NodeError>(entity);
        values[first + InputCount[kind]].Record = edited.value_or(input);
    }
    auto writes = meshes.TakeWrites();
    if (result == InvalidOffset) result = sourced && !copied ? evaluated.StoreId : created.emplace_back(Clone(r, source, {}));
    if (result == evaluated.StoreId) {
        KeepWritten(meshes, result, *resync, std::move(writes.Blocks));
        std::erase_if(resync->Attributes, [&](uint32_t attribute) { return std::ranges::binary_search(writes.Attributes, attribute); });
        r.emplace_or_replace<ResyncState>(mesh_entity, std::move(*resync));
    } else {
        evaluated = {.StoreId = result};
        r.remove<ResyncState>(mesh_entity);
    }
    std::erase(created, result);
    meshes.Release(created);
    r.emplace_or_replace<EvaluatedMesh>(mesh_entity, evaluated);
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
        for (uint32_t n = 0u; n < count; ++n)
            for (uint32_t i = plan.InputFirst[n]; i < plan.InputFirst[n + 1u]; ++i)
                if (const auto link = procedure.Inputs[i]; link.FromNode != NoNode) fn(n, link);
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
    for (uint32_t n = 0u; n < count; ++n)
        if (!pending[n]) plan.Order.push_back(n);
    for (size_t i = 0u; i < plan.Order.size(); ++i) {
        const auto n = plan.Order[i];
        for (uint32_t c = consumer_first[n]; c < consumer_first[n + 1u]; ++c)
            if (!--pending[consumers[c]]) plan.Order.push_back(consumers[c]);
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
    if (procedure.OutputNode == node) procedure.OutputNode = NoNode;
    else if (procedure.OutputNode != NoNode && procedure.OutputNode > node) --procedure.OutputNode;
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
} // namespace
} // namespace procedure

void RegisterProcedureHandlers(state::Scene &r) {
    using namespace procedure;
    r.Context.emplace<CookCount>();
    reactive(r, Change::Procedure).on<GeometryProcedure>(On::Create | On::Update);
    r.on_destroy<GeometryProcedure, &QueueNodeRemovals>();
    auto &inputs = reactive(r, Change::ProcedureInputs);
    inputs.on<MeshHandle>(On::Create | On::Update).on<MeshGeometryDirty>(On::Create).on<MeshPositionsChanged>(On::Create);
    inputs.on<InlineSelection>(On::Create | On::Update | On::Destroy);
    ForEachKind([&]<typename K> {
        inputs.on<K>(On::Update);
        r.on_destroy<K, &RemoveNode>();
    });
}

void SettleProcedures(state::Scene &r, bool cook) {
    using namespace procedure;
    const auto &changed = reactive(r, Change::Procedure);
    for (const auto e : changed)
        if (const auto *procedure = r.try_get<const GeometryProcedure>(e)) r.emplace_or_replace<CookPlan>(e, Plan(*procedure));
    // The writes since the last settle name the blocks each resync restores, and a restore pass restores outputs with their sources.
    const auto writes = r.Context.get<MeshStore>().TakeWrites();
    if (!cook) return;
    std::vector<state::Entity> cooked, nodes;
    for (const auto e : changed)
        if (r.all_of<GeometryProcedure, MeshHandle>(e)) cooked.push_back(e);
    for (const auto e : reactive(r, Change::ProcedureInputs)) {
        if (r.all_of<GeometryProcedure, MeshHandle>(e)) cooked.push_back(e);
        else nodes.push_back(e);
    }
    if (!nodes.empty()) {
        SortUnique(nodes);
        for (const auto [e, procedure] : std::as_const(r).view<const GeometryProcedure>().each())
            if (r.all_of<MeshHandle>(e) && std::ranges::any_of(procedure.Nodes, [&](const NodeRecord &node) { return std::ranges::binary_search(nodes, node.Entity); })) cooked.push_back(e);
    }
    SortUnique(cooked);
    std::optional<BufferArena<SelectionBlock>> arena;
    for (const auto e : cooked) Cook(r, e, arena, writes);
    r.Context.get<CookCount>().Value += cooked.size();
}
