#include "ProcedureFixture.h"
#include "TopologyCases.h"
#include "animation/Fields.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "audio/ModalModes.h"
#include "mesh/Primitives.h"
#include "mesh/Unsubdivide.h"
#include "object/ObjectOps.h"
#include "procedure/NodeEditor.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "scene/HierarchyInstance.h"
#include "scene/SceneGraph.h"
#include <imgui_internal.h>

using boost::ut::expect, boost::ut::fatal;

namespace {
using namespace procedure;
using action::UpdateOf;
using action::nodes::AddNode, action::nodes::CollapseProcedure, action::nodes::DeleteNodes, action::nodes::DuplicateNodes, action::nodes::InsertNode, action::nodes::Link, action::nodes::SelectItems, action::nodes::SetAttributeName, action::nodes::SetDisplay, action::nodes::SetInlineSelection;
using ::test::Op, ::test::Patchwork, ::test::Region, ::test::TopologyCases;

// Each vertex's position, sorted.
std::vector<Corner> Points(const MeshStore &meshes, uint32_t id, auto &&transform) {
    const Mesh mesh{meshes, id};
    std::vector<Corner> points;
    for (const auto v : mesh.vertices()) {
        const auto p = transform(*v, mesh.GetPosition(v));
        points.push_back({p.x, p.y, p.z});
    }
    std::ranges::sort(points);
    return points;
}
std::vector<Corner> Positions(const MeshStore &meshes, uint32_t id) {
    return Points(meshes, id, [](uint32_t, vec3 p) { return p; });
}
// The transformed positions of the geometry of the mesh's Source node, sorted.
std::vector<Corner> SourcePoints(ProcedureFixture &f, state::Entity mesh_entity, auto &&transform) {
    return f.WithSource(mesh_entity, [&](uint32_t id) { return Points(f.Store(), id, transform); });
}
bool Near(const std::vector<Corner> &a, const std::vector<Corner> &b) {
    return a.size() == b.size() && std::ranges::equal(a, b, [](const Corner &p, const Corner &q) {
               return std::abs(p[0] - q[0]) < 1e-5f && std::abs(p[1] - q[1]) < 1e-5f && std::abs(p[2] - q[2]) < 1e-5f;
           });
}
// The primitive's vertex positions, sorted.
std::vector<Corner> ShapePositions(const auto &shape) {
    std::vector<Corner> points;
    for (const auto p : primitive::CreateMesh(shape).Positions) points.push_back({p.x, p.y, p.z});
    std::ranges::sort(points);
    return points;
}
// Runs the tasks on a clone of `id`, each selecting every element, and returns the record they leave.
Shape SelectAllReference(ProcedureFixture &f, uint32_t id, std::vector<MeshTopologyTask> tasks) {
    for (auto &task : tasks) task.Flags |= TopologyFlagSelectAll;
    return Reference(f, id, [&](uint32_t clone) {
               f.Run(clone, tasks);
               return GeometrySelection{};
           })
        .Output;
}
// Runs the tasks on the geometry of the mesh's Source node, each selecting every element, and returns the record they leave.
Shape SelectAllSourceReference(ProcedureFixture &f, state::Entity mesh_entity, std::vector<MeshTopologyTask> tasks) {
    for (auto &task : tasks) task.Flags |= TopologyFlagSelectAll;
    return SourceReference(f, mesh_entity, [&](uint32_t id) {
               f.Run(id, tasks);
               return GeometrySelection{};
           })
        .Output;
}
// The inline selection of the handles in each domain.
std::unique_ptr<InlineSelection> SelectionOfHandles(GeometrySelection handles, Element domain = Element::Vertex) {
    for (auto *listed : {&handles.Vertices, &handles.Edges, &handles.Faces}) std::ranges::sort(*listed);
    return std::make_unique<InlineSelection>(SelectionOf(handles, domain));
}
// The vertices `inside` accepts, and the edges and faces whose vertices it all accepts.
GeometrySelection SelectionWhere(const Mesh &mesh, auto &&inside) {
    GeometrySelection selection;
    for (const auto v : mesh.vertices())
        if (inside(v)) selection.Vertices.push_back(*v);
    for (const auto e : mesh.edges()) {
        const auto h = mesh.GetHalfedge(e, 0u);
        if (inside(mesh.GetFromVertex(h)) && inside(mesh.GetToVertex(h))) selection.Edges.push_back(*e);
    }
    for (const auto face : mesh.faces())
        if (std::ranges::all_of(mesh.fv_range(face), inside)) selection.Faces.push_back(*face);
    return selection;
}

// The node kind whose cook plans the task's op and flags, or none for an op without a node.
// A merge at a target merges onto the selection's first vertex at its center, as the Center mode does.
// A face replacement is the area beautify that plans it.
std::optional<uint16_t> NodeOf(const MeshTopologyTask &task) {
    using enum MeshTopologyOp;
    using namespace action::mesh;
    switch (task.Op) {
        case DeleteVertices:
        case DeleteEdges:
        case DeleteFaces:
        case DeleteOnlyEdgesFaces:
        case DeleteOnlyFaces:
        case DeleteLoose: return KindOf<Delete>;
        case DissolveVertices:
        case DissolveEdges:
        case DissolveFaces:
        case DissolveLimited:
        case DissolveDegenerate: return KindOf<Dissolve>;
        case MergeAtTarget:
        case MergeByDistance:
        case MergeCollapse: return KindOf<Merge>;
        case Decimate: return KindOf<action::mesh::Decimate>;
        case AddPrimitives: return KindOf<ConvexHull>;
        case Wireframe: return KindOf<action::mesh::Wireframe>;
        case BevelEdges:
        case BevelVertices: return KindOf<Bevel>;
        case TrisToQuads: return KindOf<action::mesh::TrisToQuads>;
        case Triangulate: return KindOf<action::mesh::Triangulate>;
        case FlipNormals: return KindOf<action::mesh::FlipNormals>;
        case SplitGeometry: return KindOf<Split>;
        case DuplicateGeometry: return KindOf<Duplicate>;
        case ExtrudeVertices:
        case ExtrudeEdges:
        case ExtrudeFacesIndividual: return KindOf<Extrude>;
        case ExtrudeRegion: return task.Flags & TopologyFlagTransformCopies ? KindOf<ExtrudeRepeat> : KindOf<Extrude>;
        case Subdivide: return KindOf<action::mesh::Subdivide>;
        case EdgeSplit: return task.Flags & TopologyFlagRipSelectCopies ? KindOf<Rip> : KindOf<action::mesh::EdgeSplit>;
        case Poke: return KindOf<action::mesh::Poke>;
        case InsetRegion:
        case InsetIndividual: return KindOf<Inset>;
        case Solidify: return KindOf<action::mesh::Solidify>;
        case ConnectVertices: return KindOf<action::mesh::ConnectVertices>;
        case SplitNonplanarFaces: return KindOf<action::mesh::SplitNonplanarFaces>;
        case SplitConcaveFaces: return KindOf<action::mesh::SplitConcaveFaces>;
        case ReplaceFaces: return KindOf<BeautifyFaces>;
        default: return std::nullopt;
    }
}
// Writes each scalar parameter of the node's kind one step up and one step down from its value through the field update the node panel emits, each clamped by the field's spec.
template<typename K> void SweepParameters(ProcedureFixture &f, state::Entity node) {
    const auto before = f.R.get<const K>(node);
    field::ForEach(before, [&]<size_t I>(const auto &member, std::integral_constant<size_t, I>) {
        using T = std::remove_cvref_t<decltype(member)>;
        const auto offset = uint16_t(reinterpret_cast<const std::byte *>(&member) - reinterpret_cast<const std::byte *>(&before));
        const auto update = [&](auto value) { f.Do(action::Update<decltype(value)>{node, state::Key<K>(), offset, value}); };
        if constexpr (std::same_as<T, bool>) {
            update(!member);
        } else if constexpr (std::is_enum_v<T>) {
            if constexpr (field::EnumCount<T> > 0u) {
                const auto &values = field::EnumValues<T>;
                const auto next = values[size_t(std::ranges::find(values, member) - values.begin() + 1u) % values.size()];
                update(std::underlying_type_t<T>(next));
            }
        } else if constexpr (std::is_arithmetic_v<T>) {
            const auto step = std::is_floating_point_v<T> ? T(Specs<K>.ByIndex[I].Speed * 25.f) : T(1);
            update(T(member + step));
            update(T(member - step));
        }
    });
}
void SweepParameters(ProcedureFixture &f, state::Entity node, uint16_t kind) {
    VisitKind(kind, [&]<typename K> {
        if constexpr (!std::is_empty_v<K>) SweepParameters<K>(f, node);
    });
}
// Whether a node of the kind runs one topology task, as the kinds NodeOf returns do.
bool RunsTask(uint16_t kind) {
    using namespace action::mesh;
    static constexpr std::array Kinds{KindOf<Delete>, KindOf<Dissolve>, KindOf<Merge>, KindOf<action::mesh::Decimate>, KindOf<ConvexHull>, KindOf<action::mesh::Wireframe>, KindOf<Bevel>, KindOf<action::mesh::TrisToQuads>, KindOf<action::mesh::Triangulate>, KindOf<action::mesh::FlipNormals>, KindOf<Split>, KindOf<Duplicate>, KindOf<Extrude>, KindOf<ExtrudeRepeat>, KindOf<action::mesh::Subdivide>, KindOf<action::mesh::EdgeSplit>, KindOf<Rip>, KindOf<action::mesh::Poke>, KindOf<Inset>, KindOf<action::mesh::Solidify>, KindOf<action::mesh::ConnectVertices>, KindOf<action::mesh::SplitNonplanarFaces>, KindOf<action::mesh::SplitConcaveFaces>, KindOf<BeautifyFaces>};
    return std::ranges::contains(Kinds, kind);
}
// The live elements of the record that the inline selection selects, ascending.
GeometrySelection LiveHandles(const MeshStore &meshes, uint32_t id, const InlineSelection &selection) {
    GeometrySelection handles;
    for (const auto [element, selected, listed] : {std::tuple{Element::Vertex, &selection.Vertices, &handles.Vertices}, std::tuple{Element::Edge, &selection.Edges, &handles.Edges}, std::tuple{Element::Face, &selection.Faces, &handles.Faces}}) {
        ForEachSelectedBlock(*selected, [&](uint32_t local, const SelectionBlock &words) {
            for (uint32_t slot = 0u; slot < MeshElementBlockSize; ++slot) {
                const auto handle = local * MeshElementBlockSize + slot;
                if (words[slot / 32u] >> (slot % 32u) & 1u && meshes.IsLiveElement(id, element, ElementHandle{handle})) listed->push_back(handle);
            }
        });
    }
    return handles;
}
// The task a node of a RunsTask kind runs on record `id`, planned from its parameters and inline selection as its cook plans it, or none where the cook faults the node.
std::optional<MeshTopologyTask> TaskOf(const state::Registry &r, state::Entity node, uint16_t kind, uint32_t id) {
    using enum MeshTopologyOp;
    using namespace action::mesh;
    const auto &meshes = r.Resources.get<const MeshStore>();
    const Mesh mesh{meshes, id};
    const auto *inline_selection = r.try_get<const InlineSelection>(node);
    const bool masked = inline_selection && !inline_selection->All;
    const auto domain = inline_selection ? inline_selection->Domain : Element::Vertex;
    const auto selection = masked ? LiveHandles(meshes, id, *inline_selection) : AllHandles(mesh);
    const auto task = [&](MeshTopologyOp op, float param0 = 0.f, float param1 = 0.f, uint32_t flags = 0u) {
        MeshTopologyTask planned{.SourceId = id, .Op = op, .Param0 = param0, .Param1 = param1, .Flags = flags};
        if (masked) planned.Selection = selection;
        else planned.Flags |= TopologyFlagSelectAll;
        return planned;
    };
    std::optional<MeshTopologyTask> planned;
    VisitKind(kind, [&]<typename K> {
        if constexpr (std::same_as<K, Delete>) {
            planned = task(MeshTopologyOp(uint32_t(r.get<const K>(node).Mode)));
        } else if constexpr (std::same_as<K, Dissolve>) {
            const auto &p = r.get<const K>(node);
            if (p.Mode == DissolveMode::Vertices) planned = task(DissolveVertices);
            else if (p.Mode == DissolveMode::Edges) planned = task(DissolveEdges, 0.f, 0.f, p.KeepVertices ? TopologyFlagKeepVertices : 0u);
            else if (p.Mode == DissolveMode::Faces) planned = task(DissolveFaces);
            else if (p.Mode == DissolveMode::Degenerate) planned = task(DissolveDegenerate, std::max(p.Distance, 0.f));
            else {
                const auto flags = (domain == Element::Face ? TopologyFlagFaceSelection : 0u) | (p.AllBoundaries ? TopologyFlagAllBoundaries : 0u) | (p.DelimitMaterials ? TopologyFlagDelimitMaterial : 0u) | (p.DelimitSharpEdges ? TopologyFlagDelimitSharp : 0u) | (p.DelimitUVs ? TopologyFlagDelimitUV : 0u);
                planned = task(DissolveLimited, std::clamp(p.Angle, 0.f, .5f * std::numbers::pi_v<float>), 0.f, flags);
            }
        } else if constexpr (std::same_as<K, Merge>) {
            const auto &p = r.get<const K>(node);
            const auto &vertices = selection.Vertices;
            const auto reference = vertices.empty() ? InvalidOffset : p.Mode == MergeMode::Last ? vertices.back() :
                                                                                                  vertices.front();
            planned = MergeTask(meshes, mesh, selection, p.Mode, std::max(p.Distance, 0.f), reference);
        } else if constexpr (std::same_as<K, action::mesh::Decimate>) {
            planned = DecimateTask(meshes, mesh, selection, r.get<const K>(node).Ratio);
        } else if constexpr (std::same_as<K, ConvexHull>) {
            planned = ConvexHullTask(meshes, mesh, selection);
            if (planned && planned->Op == AddPrimitives) planned->Selection = selection;
        } else if constexpr (std::same_as<K, action::mesh::Wireframe>) {
            const auto &p = r.get<const K>(node);
            const auto flags = (p.Even ? TopologyFlagEvenOffset : 0u) | (p.Boundary ? TopologyFlagWireBoundary : 0u) | (p.Relative ? TopologyFlagWireRelative : 0u) | (p.Replace ? TopologyFlagWireReplace : 0u);
            planned = task(MeshTopologyOp::Wireframe, std::max(p.Thickness, 0.f), std::clamp(p.Offset, -1.f, 1.f), flags);
        } else if constexpr (std::same_as<K, Bevel>) {
            const auto &p = r.get<const K>(node);
            planned = task(p.Vertices ? BevelVertices : BevelEdges, std::max(p.Width, 0.f));
            planned->Steps = std::max(p.Segments, 1u);
        } else if constexpr (std::same_as<K, action::mesh::TrisToQuads>) {
            planned = task(MeshTopologyOp::TrisToQuads);
        } else if constexpr (std::same_as<K, action::mesh::Triangulate>) {
            planned = task(MeshTopologyOp::Triangulate);
        } else if constexpr (std::same_as<K, action::mesh::FlipNormals>) {
            planned = task(MeshTopologyOp::FlipNormals);
        } else if constexpr (std::same_as<K, Split>) {
            planned = task(SplitGeometry);
        } else if constexpr (std::same_as<K, Duplicate>) {
            planned = task(DuplicateGeometry);
        } else if constexpr (std::same_as<K, Extrude>) {
            const auto mode = r.get<const K>(node).Mode;
            planned = task(mode == ExtrudeMode::Vertices ? ExtrudeVertices : mode == ExtrudeMode::Edges ? ExtrudeEdges :
                               mode == ExtrudeMode::FacesIndividual                                     ? ExtrudeFacesIndividual :
                                                                                                          ExtrudeRegion);
        } else if constexpr (std::same_as<K, ExtrudeRepeat>) {
            const auto &p = r.get<const K>(node);
            planned = ExtrudeStepsTask(mesh, selection, Element::None, p.Steps, mat3{1.f}, p.Offset, vec3{0.f});
        } else if constexpr (std::same_as<K, action::mesh::Subdivide>) {
            planned = task(MeshTopologyOp::Subdivide, float(std::max(r.get<const K>(node).Cuts, 1u)));
        } else if constexpr (std::same_as<K, action::mesh::EdgeSplit>) {
            planned = task(MeshTopologyOp::EdgeSplit);
        } else if constexpr (std::same_as<K, Rip>) {
            planned = task(MeshTopologyOp::EdgeSplit, 0.f, 0.f, TopologyFlagRipSelectCopies);
        } else if constexpr (std::same_as<K, action::mesh::Poke>) {
            planned = task(MeshTopologyOp::Poke, r.get<const K>(node).Offset);
        } else if constexpr (std::same_as<K, Inset>) {
            const auto &p = r.get<const K>(node);
            planned = task(p.Individual ? InsetIndividual : InsetRegion, std::max(p.Thickness, 0.f), p.Depth, p.Even ? TopologyFlagEvenOffset : 0u);
        } else if constexpr (std::same_as<K, action::mesh::Solidify>) {
            planned = task(MeshTopologyOp::Solidify, r.get<const K>(node).Thickness);
        } else if constexpr (std::same_as<K, action::mesh::ConnectVertices>) {
            planned = task(MeshTopologyOp::ConnectVertices);
        } else if constexpr (std::same_as<K, action::mesh::SplitNonplanarFaces>) {
            planned = task(MeshTopologyOp::SplitNonplanarFaces, std::clamp(r.get<const K>(node).Angle, 0.f, std::numbers::pi_v<float>));
        } else if constexpr (std::same_as<K, action::mesh::SplitConcaveFaces>) {
            planned = task(MeshTopologyOp::SplitConcaveFaces);
        } else if constexpr (std::same_as<K, BeautifyFaces>) {
            planned = BeautifyFaceTask(mesh, selection, r.get<const K>(node).Method == BeautifyMethod::Angle);
        }
    });
    return planned;
}

} // namespace

// Each procedure mesh whose display path runs only topology tasks on one Source node draws what those tasks make of a clone of the Source, where a faulted node passes its input through.
void ProcedureFixture::CheckStep() {
    auto &meshes = Store();
    bool cloned = false;
    for (const auto [mesh, plan] : R.view<const CookPlan>().each()) {
        if (plan.ResultNode == NoNode) continue;
        const auto upstream = Upstream(plan, plan.ResultNode, [](uint32_t) { return true; });
        std::optional<SourceState> source;
        std::vector<NodeRecord> path;
        bool runs = true;
        for (const auto n : plan.Order) {
            if (!upstream[n]) continue;
            if (std::ranges::any_of(std::span{plan.Inputs}.subspan(plan.InputFirst[n], plan.InputFirst[n + 1] - plan.InputFirst[n]), [](const auto &input) { return input.Constant.has_value(); })) runs = false;
            const auto [node, kind] = plan.Nodes[n];
            if (kind == KindOf<action::mesh::Extrude> && R.get<const action::mesh::Extrude>(node).OffsetScale != 0.f) runs = false;
            if (kind == KindOf<procedure::Source> && !source) source = R.get<const SourceState>(node);
            else if (RunsTask(kind)) path.push_back(plan.Nodes[n]);
            else runs = false;
        }
        if (!runs || !source || path.empty()) continue;
        const auto id = CloneSource(*source);
        cloned = true;
        for (const auto [node, kind] : path) {
            if (R.all_of<NodeError>(node)) continue;
            const auto task = TaskOf(R, node, kind, id);
            expect(task.has_value());
            if (task) ExecuteGeometryTopology(R, std::span{&*task, 1u});
        }
        expect(Describe(meshes, id) == Shaped(mesh));
        meshes.Release(id);
    }
    // The present node holds mesh storage as it stood before the clones.
    if (cloned) P->History.Revert();
}

namespace {
// The output must be the kernel's, with the Source node's state unchanged.
void ExpectCooksLike(ProcedureFixture &f, state::Entity mesh, const SourceState &source, const Shape &reference) {
    expect(f.Shaped(mesh) == reference);
    expect(f.SourceOf(mesh) == source);
}

// Every operator kind cooks like its kernel on the patchwork at every parameter value, each generator kind like its primitive, and a fan-out cooks the read branch alone.
void OperatorKinds(ProcedureFixture &f, Fixture &other) {
    for (const auto &c : TopologyCases) {
        const auto kind = NodeOf(c.Task);
        if (!kind) continue;
        Scenario(f, other, std::string{c.Name}, [kind](ProcedureFixture &f) {
            auto &r = f.R;
            const auto &meshes = f.Store();
            const auto mesh = AddMesh(r, CreateMesh(r, {.Data = Patchwork()}).StoreId, MeshInstanceCreateInfo{}).first;
            f.Commit("Add patchwork");
            const auto source = f.SourceOf(mesh);
            const auto entity = f.Append(mesh, *kind);
            // The region by the source's handles, as CaptureSelection captures a drawn selection.
            f.Do(SetInlineSelection{entity, SelectionOfHandles(Region(Mesh{meshes, f.Output(mesh)}))});
            f.Route(mesh, {f.Node(mesh, 0u), entity});
            SweepParameters(f, entity, *kind);
            expect(f.SourceOf(mesh) == source);
        });
    }
    // Generator kinds follow PrimitiveParameters' alternatives.
    for (const PrimitiveParameters shape : {PrimitiveParameters{primitive::Plane{}}, PrimitiveParameters{primitive::Circle{}}, PrimitiveParameters{primitive::Cube{}}, PrimitiveParameters{primitive::IcoSphere{}}, PrimitiveParameters{primitive::UVSphere{}}, PrimitiveParameters{primitive::Torus{}}, PrimitiveParameters{primitive::Cylinder{}}, PrimitiveParameters{primitive::Cone{}}}) {
        Scenario(f, other, std::visit([]<typename K>(const K &) { return std::string{Title<K>.c_str()}; }, shape), [shape](ProcedureFixture &f) {
            auto &meshes = f.Store();
            const auto mesh = f.AddMesh(primitive::Plane{});
            const auto source = f.SourceOf(mesh);
            const auto generator = f.Append(mesh, uint16_t(KindOf<primitive::Plane> + shape.index()));
            const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
            f.Route(mesh, {generator, subdivide});
            const auto generated = CreateMesh(f.R, {.Data = primitive::CreateMesh(shape), .FlatShaded = true}).StoreId;
            const auto reference = SelectAllReference(f, generated, {Op(MeshTopologyOp::Subdivide, 1.f)});
            meshes.Release(generated);
            f.Commit("Release generated");
            ExpectCooksLike(f, mesh, source, reference);
        });
    }
    Scenario(f, other, "FanOut", [](ProcedureFixture &f) {
        const auto mesh = f.AddMesh(primitive::Plane{});
        const auto source = f.SourceOf(mesh);
        const auto first = f.Append(mesh, KindOf<action::mesh::Extrude>);
        const auto second = f.Append(mesh, KindOf<action::mesh::Extrude>);
        f.Do(Link{mesh, f.Node(mesh, 0u), 0u, first, 0u});
        f.Do(Link{mesh, f.Node(mesh, 0u), 0u, second, 0u});
        for (const auto read : {first, second}) {
            f.Do(SetDisplay{mesh, read});
            expect(!f.R.any_of<NodeError>(first) && !f.R.any_of<NodeError>(second));
            expect(f.SourceOf(mesh) == source);
        }
    });
}

// A graph on `Mesh`, an oracle for its output beyond the kernels each step checks, and an edit that changes a copy's output.
struct Graph {
    state::Entity Mesh;
    std::function<void(ProcedureFixture &)> Oracle{};
    std::function<void(ProcedureFixture &, state::Entity)> Edit{};
};
struct GraphRow {
    std::string_view Name;
    Graph (*Build)(ProcedureFixture &);
};

// Each graph's output matches its oracle, a duplicate cooks the same and edits apart, and deleting and collapsing the procedure leave its output.
void Survives(ProcedureFixture &f, Fixture &other, std::span<const GraphRow> rows) {
    for (const auto &row : rows) {
        Scenario(f, other, std::string{row.Name}, [&row](ProcedureFixture &f) {
            auto &r = f.R;
            const auto graph = row.Build(f);
            const auto mesh = graph.Mesh;
            if (graph.Oracle) graph.Oracle(f);
            const auto output = f.Shaped(mesh);
            const auto display = r.get<const GeometryProcedure>(mesh);
            f.Do(SetDisplay{mesh});
            expect(f.Shaped(mesh) == Shape{});
            const auto *instances = r.try_get<const ProcedureInstances>(mesh);
            expect(!instances || instances->Entities.empty());
            f.Do(SetDisplay{mesh, display.DisplayNode, display.DisplayPort});
            expect(f.Shaped(mesh) == output);
            f.Do(AddNode{mesh, KindOf<action::mesh::Bevel>, {}});
            const auto unused = r.get<const GeometryProcedure>(mesh).Last;
            expect(r.get<const GeometryProcedure>(mesh).DisplayNode == display.DisplayNode);
            expect(f.Shaped(mesh) == output);
            const auto cooks = Cooks(r);
            f.Do(UpdateOf<&action::mesh::Bevel::Width>(unused, .2f));
            std::vector<state::Entity> groups;
            auto member = r.get<const GeometryProcedure>(mesh).DisplayNode;
            for (int depth = 0; depth < 2; ++depth) {
                f.Do(action::nodes::GroupNodes{mesh, {member}});
                member = r.get<const GeometryProcedure>(mesh).Last;
                groups.push_back(member);
                expect(f.Shaped(mesh) == output && Cooks(r) == cooks);
            }
            if (!graph.Edit) {
                for (const auto group : groups | std::views::reverse) {
                    f.Do(action::nodes::UngroupNodes{group});
                    expect(f.Shaped(mesh) == output && Cooks(r) == cooks);
                }
                return;
            }
            f.Do(action::selection::Select{f.Object(mesh)});
            f.Do(action::object::Duplicate{});
            const auto copy = r.get<const Instance>(FindActiveEntity(r)).Entity;
            expect(copy != mesh && f.Output(copy) != f.Output(mesh) && f.Shaped(copy) == output);
            graph.Edit(f, copy);
            expect(f.Shaped(copy) != output && f.Shaped(mesh) == output);
            f.Undo();
            f.Undo();
            const auto procedures = r.view<const GeometryProcedure>().size();
            f.Do(action::object::Delete{});
            expect(!r.valid(mesh) && r.view<const GeometryProcedure>().size() + 1u == procedures);
            f.Undo();
            expect(f.Shaped(mesh) == output);
            f.Do(CollapseProcedure{mesh});
            const auto &collapsed = r.get<const CookPlan>(mesh);
            expect(collapsed.Nodes.size() == 1u && collapsed.Nodes[0].Kind == KindOf<procedure::Source>);
            expect(f.SourceOf(mesh).StoreId == f.Output(mesh) && f.Shaped(mesh) == output);
        });
    }
}

Graph ExtrudeSubdivide(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto extrude = f.Append(mesh, KindOf<action::mesh::Extrude>);
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Route(mesh, {f.Node(mesh, 0u), extrude, subdivide});
    const auto once = f.Shaped(mesh);
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide, 2u));
    expect(f.Shaped(mesh) != once);
    return {
        mesh,
        [mesh](ProcedureFixture &f) { expect(f.Shaped(mesh) != f.SourceShape(mesh)); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 2u), 3u)); },
    };
}

void ExtrudeDisplacement(ProcedureFixture &f) {
    using E = action::mesh::Extrude;
    using M = action::mesh::ExtrudeMode;
    for (const auto mode : {M::Region, M::Edges, M::FacesIndividual, M::Vertices}) {
        const auto mesh = f.AddMesh(primitive::Plane{});
        const Mesh plane{f.Store(), f.Output(mesh)};
        const auto normal = plane.GetNormal(*plane.faces().begin());
        const auto extrude = f.Append(mesh, KindOf<E>);
        f.Do(UpdateOf<&E::Mode>(extrude, mode));
        f.Route(mesh, {f.Node(mesh, 0u), extrude});
        const auto topology = f.R.get<const NodeExtrusion>(extrude).State;
        const auto zero = f.Shaped(mesh);
        for (const float distance : {.25f, .75f, -.5f, 0.f}) {
            f.Do(UpdateOf<&E::OffsetScale>(extrude, distance));
            expect(f.R.get<const NodeExtrusion>(extrude).State == topology);
            const Mesh result{f.Store(), f.Output(mesh)};
            uint32_t displaced = 0u;
            for (const auto vertex : result.vertices()) {
                const float height = Dot(result.GetPosition(vertex), normal);
                expect(std::abs(height) < .0001f || std::abs(height - distance) < .0001f);
                if (std::abs(height - distance) < .0001f) ++displaced;
            }
            expect(displaced >= 4u);
            if (distance == 0.f) expect(f.Shaped(mesh) == zero);
        }
        f.Do(UpdateOf<&E::AlongNormals>(extrude, false));
        f.Do(UpdateOf<&E::Offset>(extrude, vec3{.3f, -.2f, .7f}));
        f.Do(UpdateOf<&E::OffsetScale>(extrude, 2.f));
        expect(f.Shaped(mesh) != zero);
        const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
        f.Route(mesh, {f.Node(mesh, 0u), extrude, subdivide});
        const auto downstream = f.Shaped(mesh);
        f.Do(action::nodes::SetOperationHandle{extrude, f.Object(mesh), HandleParameter::ExtrudeScale});
        f.Do(UpdateOf<&E::OffsetScale>(extrude, 1.f));
        f.Do(action::nodes::SetOperationHandle{});
        expect(f.R.get<const GeometryProcedure>(mesh).DisplayNode == subdivide);
        expect(f.Shaped(mesh) != downstream);
        expect(f.R.get<const NodeExtrusion>(extrude).State == topology);
    }
}

// A sphere's cap above y = 0.9, with a vertex moved before its Source node's state is kept, insets through Position, Compare and Inset.
Graph CapInset(ProcedureFixture &f) {
    auto &r = f.R;
    const auto &meshes = f.Store();
    const auto source = CreateMesh(r, {.Data = primitive::CreateMesh(primitive::UVSphere{.Slices = 128u, .Stacks = 64u}), .FlatShaded = true}).StoreId;
    f.P->Settle();
    const PositionOperationTarget target{.StoreId = source, .Selection = GeometrySelection{.Vertices = {**Mesh{meshes, source}.vertices().begin()}}};
    ExecuteGeometryPositions(r, std::span{&target, 1u}, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = {.3f, -.5f, .1f}}}});
    const auto mesh = AddMesh(r, source, MeshInstanceCreateInfo{}).first;
    f.Commit("Add moved sphere");
    const auto position = f.Append(mesh, KindOf<Position>);
    const auto compare = f.Append(mesh, KindOf<Compare>);
    const auto inset = f.Append(mesh, KindOf<action::mesh::Inset>);
    f.Do(UpdateOf<&Compare::Component>(compare, VectorComponent::Y));
    f.Do(UpdateOf<&Compare::B>(compare, .9f));
    f.Do(Link{mesh, position, 0u, compare, 0u});
    f.Do(Link{mesh, compare, f.Out(compare), inset, 1u});
    f.Route(mesh, {f.Node(mesh, 0u), inset});
    const auto first = f.Shaped(mesh);
    const auto set = [&](float thickness) {
        f.Do(UpdateOf<&action::mesh::Inset::Thickness>(inset, thickness));
        return f.Shaped(mesh);
    };
    set(.05f);
    set(.2f);
    const auto thin = set(.01f);
    expect(set(action::mesh::Inset{}.Thickness) == first);
    f.Undo();
    expect(f.Shaped(mesh) == thin);
    set(.03f);
    set(.06f);
    return {
        mesh,
        // Compare output against the kernel insetting the cap of the Source node's geometry.
        [mesh](ProcedureFixture &f) {
            const auto kernel = SourceReference(f, mesh, [&](uint32_t clone) {
                const Mesh sphere{f.Store(), clone};
                const auto cap = SelectionWhere(sphere, [&](he::VH v) { return sphere.GetPosition(v).y > .9f; });
                f.Run(clone, {{.Op = MeshTopologyOp::InsetRegion, .Param0 = .06f, .Flags = TopologyFlagEvenOffset, .Selection = cap}});
                return GeometrySelection{};
            });
            expect(f.Shaped(mesh) == kernel.Output);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Inset::Thickness>(f.Node(mesh, 3u), .2f)); },
    };
}

// Vector Math feeding Set Position doubles vertex positions.
Graph ScalePositions(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto position = f.Append(mesh, KindOf<Position>);
    const auto math = f.Append(mesh, KindOf<VectorMath>);
    const auto set = f.Append(mesh, KindOf<SetPosition>);
    f.Do(UpdateOf<&VectorMath::Operation>(math, VectorMathOp::Scale));
    f.Do(UpdateOf<&VectorMath::Scale>(math, 2.f));
    f.Do(UpdateOf<&SetPosition::Offset>(set, false));
    f.Do(Link{mesh, position, 0u, math, 0u});
    f.Do(Link{mesh, math, f.Out(math), set, 1u});
    f.Route(mesh, {f.Node(mesh, 0u), set});
    return {
        mesh,
        [mesh](ProcedureFixture &f) { expect(Near(Positions(f.Store(), f.Output(mesh)), SourcePoints(f, mesh, [](uint32_t, vec3 p) { return 2.f * p; }))); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&VectorMath::Scale>(f.Node(mesh, 2u), 3.f)); },
    };
}

// Position.z compared greater than zero selects the cube's upper face for Extrude.
Graph CompareSelectsExtrusion(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto position = f.Append(mesh, KindOf<Position>);
    const auto compare = f.Append(mesh, KindOf<Compare>);
    const auto extrude = f.Append(mesh, KindOf<action::mesh::Extrude>);
    f.Do(UpdateOf<&Compare::Component>(compare, VectorComponent::Z));
    f.Do(Link{mesh, position, 0u, compare, 0u});
    f.Do(Link{mesh, compare, f.Out(compare), extrude, 1u});
    f.Route(mesh, {f.Node(mesh, 0u), extrude});
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            const auto kernel = SourceReference(f, mesh, [&](uint32_t clone) {
                const Mesh cube{meshes, clone};
                const auto upper = SelectionWhere(cube, [&](he::VH v) { return cube.GetPosition(v).z > 0.f; });
                expect(upper.Faces.size() == 1u);
                f.Run(clone, {{.Op = MeshTopologyOp::ExtrudeRegion, .Selection = upper}});
                return GeometrySelection{};
            });
            expect(f.Shaped(mesh) == kernel.Output);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&Compare::Component>(f.Node(mesh, 2u), VectorComponent::X)); },
    };
}

vec3 OffsetOf(uint32_t v) { return {float(v % 7u), 1.f, -float(v % 3u)}; }
// A mesh of the primitive whose Source node refers to a record that `add` gives named attributes before the node's state is kept.
state::Entity AddAttributed(ProcedureFixture &f, const PrimitiveParameters &shape, auto &&add) {
    auto &r = f.R;
    const auto source = CreateMesh(r, {.Data = std::visit([](const auto &parameters) { return primitive::CreateMesh(parameters); }, shape), .FlatShaded = true}).StoreId;
    f.P->Settle();
    add(source);
    const auto mesh = AddMesh(r, source, MeshInstanceCreateInfo{}).first;
    f.Commit("Add attributed mesh");
    return mesh;
}
// A named vertex attribute offsets each vertex through Set Position.
Graph NamedAttributeReads(ProcedureFixture &f) {
    auto &meshes = f.Store();
    const auto mesh = AddAttributed(f, primitive::Plane{}, [&](uint32_t source) {
        const auto offset = meshes.AddAttribute(source, AttributeDomain::Vertex, AttributeType::Vec3, "offset").value();
        for (const auto v : Mesh{meshes, source}.vertices()) {
            const auto value = OffsetOf(*v);
            meshes.SetAttribute<vec3>(offset, *v, std::span{&value, 1u});
        }
    });
    const auto named = f.Append(mesh, KindOf<procedure::NamedAttribute>);
    const auto set = f.Append(mesh, KindOf<SetPosition>);
    f.Do(SetAttributeName{named, "offset"});
    f.Do(UpdateOf<&procedure::NamedAttribute::Type>(named, AttributeType::Vec3));
    f.Do(Link{mesh, named, 0u, set, 1u});
    f.Route(mesh, {f.Node(mesh, 0u), set});
    return {
        mesh,
        [mesh](ProcedureFixture &f) { expect(Near(Positions(f.Store(), f.Output(mesh)), SourcePoints(f, mesh, [](uint32_t v, vec3 p) { return p + OffsetOf(v); }))); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&SetPosition::Offset>(f.Node(mesh, 2u), false)); },
    };
}

// Whether the output's "rest" attribute holds each vertex's position.
bool RestMatches(ProcedureFixture &f, state::Entity mesh) {
    const auto &meshes = f.Store();
    const auto id = f.Output(mesh);
    const auto rest = meshes.FindAttribute(id, "rest");
    if (rest == InvalidOffset) return false;
    const auto view = meshes.AttributeView<vec3>(rest);
    const Mesh output{meshes, id};
    for (const auto v : output.vertices()) {
        const auto d = view[*v] - output.GetPosition(v);
        if (std::abs(d.x) >= 1e-5f || std::abs(d.y) >= 1e-5f || std::abs(d.z) >= 1e-5f) return false;
    }
    return true;
}
// Store Named Attribute keeps Position under its name through a downstream subdivision and a re-cook with more cuts.
Graph StorePromotes(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto position = f.Append(mesh, KindOf<Position>);
    const auto store = f.Append(mesh, KindOf<StoreNamedAttribute>);
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Do(SetAttributeName{store, "rest"});
    f.Do(Link{mesh, position, 0u, store, 1u});
    f.Route(mesh, {f.Node(mesh, 0u), store, subdivide});
    expect(RestMatches(f, mesh));
    const auto faces = f.Shaped(mesh).Faces.size();
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide, 2u));
    expect(f.Shaped(mesh).Faces.size() > faces);
    return {
        mesh,
        [mesh](ProcedureFixture &f) { expect(RestMatches(f, mesh)); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 3u), 3u)); },
    };
}

// A Store Selection of the cube's upper face keeps it through a subdivision inserted after the store, so a Delete reading it through Named Attribute removes the four faces it became.
Graph StoreSelectionDelete(ProcedureFixture &f) {
    const auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto store = f.Append(mesh, KindOf<StoreSelection>);
    const auto named = f.Append(mesh, KindOf<procedure::NamedAttribute>);
    const auto remove = f.Append(mesh, KindOf<action::mesh::Delete>);
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Do(SetAttributeName{store, "upper"});
    f.Do(SetAttributeName{named, "upper"});
    f.Do(UpdateOf<&procedure::NamedAttribute::Type>(named, AttributeType::Bool));
    f.Do(UpdateOf<&action::mesh::Delete::Mode>(remove, action::mesh::DeleteMode::OnlyFaces));
    // The upper face of the source the mesh draws.
    const Mesh cube{meshes, f.Output(mesh)};
    GeometrySelection upper;
    for (const auto face : cube.faces())
        if (cube.CalcFaceCentroid(face).y > .5f) upper.Faces.push_back(*face);
    f.Do(SetInlineSelection{store, SelectionOfHandles(upper, Element::Face)});
    f.Do(Link{mesh, named, 0u, remove, 1u});
    f.Route(mesh, {f.Node(mesh, 0u), store, remove});
    expect(Mesh{meshes, f.Output(mesh)}.FaceCount() == 5u);
    f.Do(InsertNode{mesh, subdivide, remove, 0u, 0u});
    return {
        mesh,
        [mesh, named, remove](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            expect(!f.R.any_of<NodeError>(named) && !f.R.any_of<NodeError>(remove));
            const auto kernel = SourceReference(f, mesh, [&](uint32_t clone) {
                f.Run(clone, {Op(MeshTopologyOp::Subdivide, 1.f, TopologyFlagSelectAll)});
                const Mesh subdivided{meshes, clone};
                GeometrySelection upper;
                for (const auto face : subdivided.faces())
                    if (subdivided.CalcFaceCentroid(face).y > .99f) upper.Faces.push_back(*face);
                expect(upper.Faces.size() == 4u);
                f.Run(clone, {{.Op = MeshTopologyOp(uint32_t(action::mesh::DeleteMode::OnlyFaces)), .Selection = upper}});
                return GeometrySelection{};
            });
            expect(f.Shaped(mesh) == kernel.Output);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 4u), 2u)); },
    };
}

// Vector Math on a vertex and a face attribute errs and forwards A, so Set Position offsets each vertex by its position.
Graph DomainMismatch(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto position = f.Append(mesh, KindOf<Position>);
    const auto normal = f.Append(mesh, KindOf<Normal>);
    const auto math = f.Append(mesh, KindOf<VectorMath>);
    const auto set = f.Append(mesh, KindOf<SetPosition>);
    f.Do(UpdateOf<&Normal::Domain>(normal, AttributeDomain::Face));
    f.Do(Link{mesh, position, 0u, math, 0u});
    f.Do(Link{mesh, normal, 0u, math, 1u});
    f.Do(Link{mesh, math, f.Out(math), set, 1u});
    f.Route(mesh, {f.Node(mesh, 0u), set});
    return {
        mesh,
        [mesh, math](ProcedureFixture &f) {
            expect(f.R.any_of<NodeError>(math));
            expect(Near(Positions(f.Store(), f.Output(mesh)), SourcePoints(f, mesh, [](uint32_t, vec3 p) { return 2.f * p; })));
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&Normal::Domain>(f.Node(mesh, 2u), AttributeDomain::Vertex)); },
    };
}

// A join of the source and a generated cube doubles the elements, and the generated vertices take the source attribute's default.
Graph JoinGenerated(ProcedureFixture &f) {
    auto &meshes = f.Store();
    const auto mesh = AddAttributed(f, primitive::Cube{}, [&](uint32_t source) {
        const auto weight = meshes.AddAttribute(source, AttributeDomain::Vertex, AttributeType::Float, "weight", AttributeValue{std::bit_cast<uint32_t>(-1.f), 0u, 0u, 0u}).value();
        for (const auto v : Mesh{meshes, source}.vertices()) {
            const float value = 5.f;
            meshes.SetAttribute<float>(weight, *v, std::span{&value, 1u});
        }
    });
    const auto cube = f.Append(mesh, KindOf<primitive::Cube>);
    const auto join = f.Append(mesh, KindOf<Join>);
    f.Do(Link{mesh, f.Node(mesh, 0u), 0u, join, 0u});
    f.Do(Link{mesh, cube, f.Out(cube), join, 1u});
    f.Do(SetDisplay{mesh, join});
    const auto joined = f.Shaped(mesh);
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(cube, vec3{2.f, .5f, 1.f}));
    expect(f.Shaped(mesh) != joined);
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            const auto id = f.Output(mesh);
            const auto [vertices, faces] = f.WithSource(mesh, [&](uint32_t source) {
                const Mesh single{meshes, source};
                return std::pair{single.VertexCount(), single.FaceCount()};
            });
            const Mesh joined{meshes, id};
            expect(joined.VertexCount() == 2u * vertices && joined.FaceCount() == 2u * faces);
            const auto view = meshes.AttributeView<float>(meshes.FindAttribute(id, "weight"));
            std::array<uint32_t, 2> weights{};
            for (const auto v : joined.vertices()) {
                if (view[*v] == 5.f) ++weights[0];
                else if (view[*v] == -1.f) ++weights[1];
            }
            expect(weights[0] == vertices && weights[1] == vertices);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(mesh, 1u), vec3{.5f, .5f, .5f})); },
    };
}

std::map<uint32_t, float> Heat(const MeshStore &meshes, uint32_t id) {
    std::map<uint32_t, float> heat;
    const auto view = meshes.AttributeView<float>(meshes.FindAttribute(id, "heat"));
    meshes.Arenas().Vertices.ForEach(meshes.Get(id).Vertices, [&](ElementHandle v, ElementAddress) { heat.emplace(v.Value, view[v.Value]); });
    return heat;
}
// A named attribute's values travel through an edit-mode subdivision of a cube.
// Retained vertices keep their heat, and an edge midpoint takes the mean of two corners and a face center the mean of four.
Graph HeatSubdivide(ProcedureFixture &f) {
    auto &meshes = f.Store();
    const auto mesh = AddAttributed(f, primitive::Cube{}, [&](uint32_t id) {
        const auto heat = meshes.AddAttribute(id, AttributeDomain::Vertex, AttributeType::Float, "heat").value();
        const auto vertices = meshes.Arenas().Vertices.Dense(meshes.Get(id).Vertices);
        std::vector<float> values(vertices.Count);
        for (uint32_t i = 0u; i < vertices.Count; ++i) values[i] = float(i * i);
        meshes.SetAttribute<float>(heat, 0u, values);
    });
    f.Do(action::selection::Select{f.Object(mesh)});
    const auto original = Heat(meshes, f.Output(mesh));
    f.EnterEdit(Element::Vertex);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{1u});
    return {mesh, [mesh, original](ProcedureFixture &f) {
                const auto split = Heat(f.Store(), f.Output(mesh));
                expect(split.size() > original.size());
                for (const auto [v, value] : original) expect(split.at(v) == value);
                std::vector<float> corners;
                corners.reserve(original.size());
                for (const auto [v, value] : original) corners.push_back(value);
                std::set<float> means;
                for (size_t a = 0u; a < corners.size(); ++a) {
                    for (size_t b = a + 1u; b < corners.size(); ++b) {
                        means.insert(.5f * (corners[a] + corners[b]));
                        for (size_t c = b + 1u; c < corners.size(); ++c)
                            for (size_t d = c + 1u; d < corners.size(); ++d) means.insert(.25f * (corners[a] + corners[b] + corners[c] + corners[d]));
                    }
                }
                for (const auto [v, value] : split)
                    if (!original.contains(v)) expect(means.contains(value)) << "vertex" << v << "heat" << value;
            }};
}

// A boolean intersection of the cube with a generated bar keeps the bar's middle.
Graph BooleanIntersection(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto cube = f.Shaped(mesh);
    const auto box = f.Append(mesh, KindOf<primitive::Cube>);
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(box, vec3{2.f, .5f, .5f}));
    const auto boolean = f.Append(mesh, KindOf<procedure::Boolean>);
    f.Do(UpdateOf<&procedure::Boolean::Operation>(boolean, BooleanOperation::Intersection));
    f.Route(mesh, {f.Node(mesh, 0u), boolean});
    // An unlinked operand passes the cube through.
    expect(f.Shaped(mesh) == cube);
    f.Do(Link{mesh, box, f.Out(box), boolean, 1u});
    expect(std::abs(test::EnclosedVolume(Mesh{f.Store(), f.Output(mesh)}).value_or(-1.0) - 2.0) < 1e-4);
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(box, vec3{2.f, .25f, .75f}));
    return {
        mesh,
        [mesh](ProcedureFixture &f) { expect(std::abs(test::EnclosedVolume(Mesh{f.Store(), f.Output(mesh)}).value_or(-1.0) - 1.5) < 1e-4); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&procedure::Boolean::Operation>(f.Node(mesh, 2u), BooleanOperation::Union)); },
    };
}

// The boolean action makes the active object's mesh the active cube minus the other selected cube, in the active object's space.
Graph BooleanAction(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto target = f.AddMesh(primitive::Cube{});
    const auto active = FindActiveEntity(r);
    f.Do(action::object::AddMeshPrimitive{primitive::Cube{}, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = {.P = {1.f, 1.f, 1.f}}, .Select = MeshInstanceCreateInfo::SelectBehavior::Additive})});
    expect(FindActiveEntity(r) == active);
    f.Do(action::mesh::Boolean{BooleanOperation::Difference});
    return {target, [target, active](ProcedureFixture &f) {
                expect(std::abs(test::EnclosedVolume(Mesh{f.Store(), f.Output(target)}).value_or(-1.0) - 7.0) < 1e-4);
                for (const auto e : f.R.view<const Instance>())
                    if (e != active) expect(Mesh{f.Store(), f.Output(f.R.get<const Instance>(e).Entity)}.VertexCount() == 8u);
            }};
}

// Generator parameter edits regenerate the record.
Graph GeneratorEdit(ProcedureFixture &f) {
    const auto mesh = f.AddGenerated(primitive::Cube{});
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(mesh, 0u), vec3{2.f, .5f, 1.f}));
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(mesh, 0u), vec3{1.f, 1.f, 3.f}));
    return {mesh, [mesh](ProcedureFixture &f) { expect(Near(Positions(f.Store(), f.Output(mesh)), ShapePositions(primitive::Cube{{1.f, 1.f, 3.f}}))); }};
}

// An Object Info node reading its own procedure's mesh errs and forwards the source.
// One reading a mesh with a procedure of its own cooks after that procedure, so a resize there reaches the join.
Graph ObjectInfo(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto plane = f.Shaped(mesh);
    const auto info = f.Append(mesh, KindOf<procedure::ObjectInfo>);
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(info, f.Object(mesh)));
    const auto join = f.Append(mesh, KindOf<Join>);
    f.Do(Link{mesh, f.Node(mesh, 0u), 0u, join, 0u});
    f.Do(Link{mesh, info, 0u, join, 1u});
    f.Do(SetDisplay{mesh, join});
    expect(r.all_of<NodeError>(info) && !r.any_of<NodeError>(join));
    expect(f.Shaped(mesh) == plane);
    const auto cube = f.AddMesh(primitive::Cube{});
    const auto box = f.Append(cube, KindOf<primitive::Cube>);
    f.Do(SetDisplay{cube, box});
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(info, f.Object(cube)));
    expect(!r.any_of<NodeError>(info));
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(box, vec3{.5f, .25f, .5f}));
    return {mesh, [mesh](ProcedureFixture &f) {
                auto points = ShapePositions(primitive::Plane{});
                points.append_range(ShapePositions(primitive::Cube{{.5f, .25f, .5f}}));
                std::ranges::sort(points);
                expect(Mesh{f.Store(), f.Output(mesh)}.FaceCount() == 7u && Near(Positions(f.Store(), f.Output(mesh)), points));
            }};
}

// Instances on the plane's points publish as entities or realize as geometry, following scale, plane and cube edits.
template<bool Borrowed, bool Realize> Graph Instances(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddGenerated(primitive::Plane{});
    const auto object = FindActiveEntity(r);
    const auto cube_mesh = f.AddGenerated(primitive::Cube{});
    const auto to_points = f.Append(mesh, KindOf<MeshToPoints>);
    const auto instancer = f.Append(mesh, KindOf<InstanceOnPoints>);
    const auto instance = f.Append(mesh, Borrowed ? KindOf<procedure::ObjectInfo> : KindOf<primitive::Cube>);
    if constexpr (Borrowed) f.Do(UpdateOf<&procedure::ObjectInfo::Object>(instance, f.Object(cube_mesh)));
    f.Route(mesh, {f.Node(mesh, 0u), to_points, instancer});
    f.Do(Link{mesh, instance, f.Out(instance), instancer, 1u});
    if constexpr (Realize) {
        const auto realizer = f.Append(mesh, KindOf<RealizeInstances>);
        f.Route(mesh, {instancer, realizer});
    }
    f.Do(UpdateOf<&InstanceOnPoints::Scale>(instancer, vec3{.5f, .5f, .5f}));
    f.Do(UpdateOf<&primitive::Plane::HalfExtents>(f.Node(mesh, 0u), vec2{2.f, 3.f}));
    if constexpr (Borrowed) f.Do(UpdateOf<&primitive::Cube::HalfExtents>(f.Node(cube_mesh, 0u), vec3{.5f, .25f, .5f}));
    else f.Do(UpdateOf<&primitive::Cube::HalfExtents>(instance, vec3{.5f, .25f, .5f}));
    Graph graph{mesh, [=](ProcedureFixture &f) {
                    const auto &r = f.R;
                    const auto &meshes = f.Store();
                    const auto plane = ShapePositions(primitive::Plane{{2.f, 3.f}}), cube = ShapePositions(primitive::Cube{{.5f, .25f, .5f}});
                    if constexpr (Realize) {
                        std::vector<Corner> realized;
                        for (const auto &p : plane)
                            for (const auto &c : cube) realized.push_back({p[0] + .5f * c[0], p[1] + .5f * c[1], p[2] + .5f * c[2]});
                        std::ranges::sort(realized);
                        expect(Near(Positions(meshes, f.Output(mesh)), realized));
                    } else {
                        const auto &entities = r.get<const ProcedureInstances>(mesh).Entities;
                        expect(entities.size() == 4u);
                        std::vector<Corner> points;
                        for (const auto e : entities) {
                            expect(ParentOrNull(r, e) == object);
                            const auto &t = r.get<const Transform>(e);
                            expect(t.S == vec3{.5f, .5f, .5f});
                            points.push_back({t.P.x, t.P.y, t.P.z});
                            expect(Near(Positions(meshes, f.Output(r.get<const Instance>(e).Entity)), cube));
                        }
                        std::ranges::sort(points);
                        expect(Near(points, plane));
                    }
                }};
    if constexpr (Realize) graph.Edit = [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&InstanceOnPoints::Scale>(f.Node(mesh, 2u), vec3{.25f, .25f, .25f})); };
    return graph;
}

// Selecting, duplicating, inserting and deleting nodes leaves the output the chain they form.
Graph NodeEditing(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto source = f.Shaped(mesh);
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    const auto triangulate = f.Append(mesh, KindOf<action::mesh::Triangulate>);
    const auto position = f.Append(mesh, KindOf<Position>);
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide, 3u));
    // Duplicates keep their parameters and only the links among them.
    const auto cooks = r.Resources.get<const CookCount>().Value;
    const auto shown = r.get<const GeometryProcedure>(mesh).DisplayNode;
    f.Do(SelectItems{mesh, {subdivide}, {{subdivide}, {triangulate}}});
    f.Do(UpdateOf<&NodeLayout::Position>(subdivide, uvec2{12u, 4u}));
    expect(r.Resources.get<const CookCount>().Value == cooks);
    expect(r.get<const GeometryProcedure>(mesh).DisplayNode == shown && f.Shaped(mesh) == source);
    f.Do(DuplicateNodes{mesh, {subdivide, triangulate}});
    const auto &procedure = r.get<const CookPlan>(mesh);
    expect(procedure.Nodes.size() == 6u && r.get<const action::mesh::Subdivide>(procedure.Nodes[4].Entity).Cuts == 3u);
    expect(f.Shaped(mesh) == source);
    // Displaying a node cooks the chain it ends, inserting a node into the displayed link cooks through it, and a node without a geometry input inserts nothing.
    f.Route(mesh, {f.Node(mesh, 0u), triangulate});
    f.Fails(InsertNode{mesh, position, triangulate, 0u, 0u});
    f.Do(SelectItems{mesh, {subdivide}, {{subdivide}}});
    f.Stage(InsertNode{mesh, subdivide, triangulate, 0u, 40u});
    f.CommitGesture();
    const auto refined = f.Shaped(mesh);
    f.Do(action::nodes::SetBypass{subdivide, true});
    expect(f.Shaped(mesh) != refined);
    expect(r.get<const CookPlan>(mesh).Wires[r.get<const CookPlan>(mesh).InputFirst[2u]].FromNode == 1u);
    f.Do(action::nodes::SetBypass{subdivide, false});
    expect(f.Shaped(mesh) == refined);
    // Deleting a node heals the chain through it, and the display follows a deleted displayed node's chain.
    f.Do(DeleteNodes{mesh, {subdivide}});
    f.Undo();
    f.Do(DeleteNodes{mesh, {subdivide, triangulate}});
    expect(f.Shaped(mesh) == source && r.get<const CookPlan>(mesh).DisplayNode == 0u);
    f.Undo();
    return {mesh};
}

template<auto M, auto... Ms> uint16_t SocketOf() {
    const auto sockets = KindSockets(KindOf<field::Owner<M>>);
    constexpr std::array<uint8_t, sizeof...(Ms)> nested{uint8_t(field::IndexOf<Ms>)...};
    return uint16_t(std::ranges::find_if(sockets, [&](Socket socket) { return socket.Carries(uint8_t(field::IndexOf<M>), nested.empty() ? NoField : nested[0]); }) - sockets.begin());
}
template<AttributeDomain D> NamedAttributeRef FloatAttribute(MeshStore &meshes, uint32_t id, auto &&value) {
    const auto attribute = meshes.AddAttribute(id, D, AttributeType::Float, "parameter").value();
    const Mesh mesh{meshes, id};
    const auto set = [&](auto handle) {
        const float v = value(mesh, handle);
        meshes.SetAttribute<float>(attribute, *handle, std::span{&v, 1u});
    };
    if constexpr (D == AttributeDomain::Vertex) {
        for (const auto v : mesh.vertices()) set(v);
    } else {
        for (const auto face : mesh.faces()) set(face);
    }
    return meshes.AttributeRef(attribute);
}
float WidthAt(const Mesh &mesh, he::VH v) {
    const auto p = mesh.GetPosition(v);
    return .05f * (p.x + p.y + p.z) + .2f;
}

// Linked values override parameter constants, linked attributes set each element's parameter, and an attribute from another domain is rejected.
void ParameterSockets(ProcedureFixture &f) {
    auto &r = f.R;
    auto &meshes = f.Store();
    using action::mesh::Bevel, action::mesh::ShrinkFatten, action::mesh::Poke;
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto source = f.Shaped(mesh);
    const auto output = [&] { return f.Shaped(mesh); };
    const auto real = f.Append(mesh, KindOf<FloatValue>);
    {
        const auto bevel = f.Append(mesh, KindOf<Bevel>);
        f.Route(mesh, {f.Node(mesh, 0u), bevel});
        const auto one = output();
        f.Do(UpdateOf<&Bevel::Segments>(bevel, 3u));
        const auto three = output();
        expect(three != one);
        f.Do(UpdateOf<&Bevel::Segments>(bevel, 1u));
        const auto integer = f.Append(mesh, KindOf<IntegerValue>);
        f.Do(UpdateOf<&IntegerValue::Value>(integer, 3u));
        const auto segments = SocketOf<&Bevel::Segments>(), width = SocketOf<&Bevel::Width>();
        f.Do(Link{mesh, integer, 0u, bevel, segments});
        expect(output() == three);
        f.Do(UpdateOf<&Bevel::Segments>(bevel, 5u));
        expect(output() == three);
        f.Do(UpdateOf<&IntegerValue::Value>(integer, 1u));
        expect(output() == one);
        f.Do(action::nodes::Unlink{mesh, bevel, segments});
        f.Do(UpdateOf<&Bevel::Segments>(bevel, 3u));
        expect(output() == three);
        const auto position = f.Append(mesh, KindOf<Position>);
        f.Fails(Link{mesh, real, 0u, bevel, segments});
        f.Fails(Link{mesh, position, 0u, bevel, segments});
        f.Do(UpdateOf<&FloatValue::Value>(real, .2f));
        f.Do(Link{mesh, real, 0u, bevel, width});
        f.Do(UpdateOf<&Bevel::Width>(bevel, .3f));
        expect(output() == SelectAllSourceReference(f, mesh, {Op(MeshTopologyOp::BevelEdges, .2f, 0u, 3u)}));
        f.Do(action::nodes::Unlink{mesh, bevel, width});
        f.Do(UpdateOf<&Bevel::Width>(bevel, .1f));
        const auto flag = f.Append(mesh, KindOf<BooleanValue>);
        f.Do(Link{mesh, flag, 0u, bevel, 1u});
        expect(output() == source);
        f.Do(UpdateOf<&BooleanValue::Value>(flag, true));
        expect(output() == three);
        f.Do(DeleteNodes{mesh, {bevel, position}});
    }
    const auto position = f.Append(mesh, KindOf<Position>);
    const auto dot = f.Append(mesh, KindOf<VectorMath>);
    const auto add = f.Append(mesh, KindOf<FloatMath>);
    f.Do(UpdateOf<&VectorMath::Operation>(dot, VectorMathOp::Dot));
    f.Do(UpdateOf<&VectorMath::B>(dot, vec3{.05f, .05f, .05f}));
    f.Do(UpdateOf<&FloatMath::B>(add, .2f));
    f.Do(Link{mesh, position, 0u, dot, 0u});
    f.Do(Link{mesh, dot, f.Out(dot), add, 0u});
    // Run the operator on a source clone with all elements selected and Param0 read from the added parameter attribute.
    const auto reference = [&](MeshTopologyOp op, auto &&parameter) {
        return SourceReference(f, mesh, [&](uint32_t clone) {
                   auto task = Op(op, 0.f, TopologyFlagSelectAll);
                   task.Param0Attribute = parameter(clone);
                   f.Run(clone, {task});
                   return GeometrySelection{};
               })
            .Output;
    };
    const auto widths = [&](uint32_t clone) { return FloatAttribute<AttributeDomain::Vertex>(meshes, clone, WidthAt); };
    const auto add_operator = [&](uint16_t kind, uint16_t socket) {
        const auto node = f.Append(mesh, kind);
        f.Do(Link{mesh, add, f.Out(add), node, socket});
        f.Route(mesh, {f.Node(mesh, 0u), node});
        return node;
    };
    {
        const auto bevel = add_operator(KindOf<Bevel>, SocketOf<&Bevel::Width>());
        f.Do(UpdateOf<&Bevel::Vertices>(bevel, true));
        expect(output() == reference(MeshTopologyOp::BevelVertices, widths));
        f.Do(DeleteNodes{mesh, {bevel}});
    }
    {
        const auto shrink = add_operator(KindOf<ShrinkFatten>, SocketOf<&ShrinkFatten::Distance>());
        // The kernel reads the factor at each vertex from the attribute, past the constant factor's zero check.
        const auto kernel = SourceReference(f, mesh, [&](uint32_t clone) {
            const PositionOperationTarget target{.StoreId = clone, .Selection = AllHandles(Mesh{meshes, clone})};
            ExecuteGeometryPositions(r, std::span{&target, 1u}, PositionEditOp::ShrinkFatten, 1.f, 1u, {.Factor = widths(clone)});
            return GeometrySelection{};
        });
        expect(kernel.Output != source && output() == kernel.Output);
        f.Do(DeleteNodes{mesh, {shrink}});
    }
    {
        // A vertex attribute does not fit the poke's per-face offset, and a face normal's z does.
        const auto poke = add_operator(KindOf<Poke>, SocketOf<&Poke::Offset>());
        expect(output() == source);
        const auto normal = f.Append(mesh, KindOf<Normal>);
        f.Do(UpdateOf<&Normal::Domain>(normal, AttributeDomain::Face));
        f.Do(UpdateOf<&VectorMath::B>(dot, vec3{0.f, 0.f, .3f}));
        f.Do(Link{mesh, normal, 0u, dot, 0u});
        f.Do(Link{mesh, dot, f.Out(dot), poke, SocketOf<&Poke::Offset>()});
        const auto offsets = [&](uint32_t clone) { return FloatAttribute<AttributeDomain::Face>(meshes, clone, [](const Mesh &m, he::FH face) { return .3f * m.GetNormal(face).z; }); };
        expect(output() == reference(MeshTopologyOp::Poke, offsets));
        f.Do(DeleteNodes{mesh, {poke}});
    }
    {
        // Link vector and rotation values into nested transform members while editing scale directly.
        const auto transform = f.Append(mesh, KindOf<TransformGeometry>);
        const auto vector = f.Append(mesh, KindOf<VectorValue>);
        const auto rotation = f.Append(mesh, KindOf<RotationValue>);
        // The rotation leaves no two moved corners a GPU rounding apart in sort order.
        const Transform delta{{1.f, 2.f, 3.f}, numeric::AngleAxis(.5f, Normalize(vec3{1.f, 2.f, 3.f})), {2.f, 1.f, .5f}};
        f.Do(UpdateOf<&VectorValue::Value>(vector, delta.P));
        f.Do(UpdateOf<&RotationValue::Value>(rotation, delta.R));
        f.Do(UpdateOf<&TransformGeometry::Delta, &Transform::S>(transform, delta.S));
        f.Do(Link{mesh, vector, 0u, transform, SocketOf<&TransformGeometry::Delta, &Transform::P>()});
        f.Do(Link{mesh, rotation, 0u, transform, SocketOf<&TransformGeometry::Delta, &Transform::R>()});
        f.Route(mesh, {f.Node(mesh, 0u), transform});
        expect(Near(Positions(meshes, f.Output(mesh)), SourcePoints(f, mesh, [&](uint32_t, vec3 p) { return delta.R * (delta.S * p) + delta.P; })));
        f.Do(DeleteNodes{mesh, {transform, vector, rotation}});
    }
    {
        const auto store = f.Append(mesh, KindOf<StoreNamedAttribute>);
        f.Do(UpdateOf<&FloatValue::Value>(real, 2.5f));
        f.Do(UpdateOf<&StoreNamedAttribute::Domain>(store, AttributeDomain::Face));
        f.Do(SetAttributeName{store, "heat"});
        f.Do(Link{mesh, real, 0u, store, 1u});
        f.Route(mesh, {f.Node(mesh, 0u), store});
        const auto heat = meshes.FindAttribute(f.Output(mesh), "heat");
        expect(fatal(heat != InvalidOffset));
        expect(meshes.GetAttribute(heat).Domain == AttributeDomain::Face && meshes.GetAttribute(heat).Type == AttributeType::Float);
        const auto values = meshes.AttributeView<float>(heat);
        for (const auto face : Mesh{meshes, f.Output(mesh)}.faces()) expect(values[*face] == 2.5f);
    }
}

// A chained display path edits the record its Source node refers to in place, and the Source node's state keeps the geometry a cook from the root reads.
// Disconnecting the displayed path restores its upstream result, and collapsing captures the displayed geometry.
void SourceStates(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto source = f.SourceOf(mesh);
    const auto record = f.Output(mesh);
    const auto shape = f.Shaped(mesh);
    expect(source.StoreId == record && source.State != store::Hash128{});
    const auto extrude = f.Append(mesh, KindOf<action::mesh::Extrude>);
    f.Route(mesh, {f.Node(mesh, 0u), extrude});
    expect(f.Output(mesh) == record && f.SourceOf(mesh) == source);
    const auto extruded = f.Shaped(mesh);
    expect(extruded != shape);
    // Deleting the extrude restores the record to the geometry before it.
    f.Do(DeleteNodes{mesh, {extrude}});
    expect(f.Shaped(mesh) == shape && f.SourceOf(mesh) == source);
    f.Undo();
    expect(f.Shaped(mesh) == extruded);
    f.Redo();
    expect(f.Shaped(mesh) == shape);
    // A Boolean without a second mesh reads its first and passes it through.
    const auto boolean = f.Append(mesh, KindOf<procedure::Boolean>);
    f.Route(mesh, {f.Node(mesh, 0u), boolean});
    expect(f.Shaped(mesh) == shape && !r.any_of<NodeError>(boolean));
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Route(mesh, {boolean, subdivide});
    expect(Mesh{f.Store(), f.Output(mesh)}.FaceCount() == 24u);
    f.Do(action::nodes::Unlink{mesh, subdivide, 0u});
    expect(r.get<const GeometryProcedure>(mesh).DisplayNode == boolean && f.Shaped(mesh) == shape);
    f.Undo();
    expect(Mesh{f.Store(), f.Output(mesh)}.FaceCount() == 24u);
    f.Undo();
    expect(f.Shaped(mesh) == shape);
    f.Do(CollapseProcedure{mesh});
    expect(r.get<const CookPlan>(mesh).Nodes.size() == 1u && f.SourceOf(mesh) != source);
    expect(f.SourceOf(mesh).StoreId == f.Output(mesh) && f.Shaped(mesh) == shape);
}

// Instances of the source on the source's own points place a copy of the source.
Graph SourceInstances(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto source = f.SourceOf(mesh);
    const auto plane = Positions(f.Store(), f.Output(mesh));
    const auto to_points = f.Append(mesh, KindOf<MeshToPoints>);
    const auto instancer = f.Append(mesh, KindOf<InstanceOnPoints>);
    f.Route(mesh, {f.Node(mesh, 0u), to_points, instancer});
    f.Do(Link{mesh, f.Node(mesh, 0u), 0u, instancer, 1u});
    return {
        mesh,
        [mesh, source, plane](ProcedureFixture &f) {
            const auto &r = f.R;
            const auto &entities = r.get<const ProcedureInstances>(mesh).Entities;
            expect(fatal(entities.size() == plane.size()));
            const auto instanced = r.get<const Instance>(entities.front()).Entity;
            const auto placed = r.get<const MeshHandle>(instanced).StoreId;
            expect(instanced != mesh && placed != f.Output(mesh) && Near(Positions(f.Store(), placed), plane));
            expect(f.SourceOf(mesh) == source && Positions(f.Store(), f.Output(mesh)) != plane);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(action::nodes::Unlink{mesh, f.Node(mesh, 2u), 1u}); },
    };
}
// A viewport operator appends after the displayed node and displays its node, moving the displayed node's readers right.
// Removing the active node activates the displayed node, and deleting the displayed node displays the node its chain reads, or a Join's first input reads.
void DisplayedNode(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto source = f.Shaped(mesh);
    const auto extrude = f.Append(mesh, KindOf<action::mesh::Extrude>);
    f.Route(mesh, {f.Node(mesh, 0u), extrude});
    f.EnterEdit(Element::Face);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{1u});
    {
        const auto &procedure = r.get<const CookPlan>(mesh);
        expect(procedure.Nodes.size() == 3u && procedure.DisplayNode == 2u && ActiveIndex(r, procedure) == 2u);
        expect(procedure.Nodes[procedure.Inputs[procedure.InputFirst[2u]].FromNode].Entity == extrude);
    }
    const auto extrude_x = r.get<const NodeLayout>(extrude).Position.x;
    f.Do(SetDisplay{mesh, f.Node(mesh, 0u)});
    expect(f.Shaped(mesh) == source);
    // Selection preserves display; an explicit display action switches the stage.
    f.Do(SelectItems{mesh, {extrude}, {{extrude}}});
    f.Do(SetDisplay{mesh, extrude});
    expect(r.get<const GeometryProcedure>(mesh).DisplayNode == extrude && f.Shaped(mesh) != source);
    f.Do(SetDisplay{mesh, f.Node(mesh, 0u)});
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{1u});
    {
        const auto &procedure = r.get<const CookPlan>(mesh);
        expect(procedure.Nodes.size() == 4u && procedure.DisplayNode == 3u && ActiveIndex(r, procedure) == 3u);
        expect(procedure.Inputs[procedure.InputFirst[3u]].FromNode == 0u && procedure.Inputs[procedure.InputFirst[1u]].FromNode == 0u);
        expect(r.get<const NodeLayout>(extrude).Position.x == extrude_x + AppendGap);
    }
    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
    f.Do(SelectItems{mesh, {extrude}, {{extrude}}});
    f.Do(DeleteNodes{mesh, {extrude}});
    {
        const auto &procedure = r.get<const CookPlan>(mesh);
        expect(procedure.Nodes.size() == 3u && procedure.DisplayNode == 2u && ActiveIndex(r, procedure) == 2u);
    }
    f.Do(DeleteNodes{mesh, {f.Node(mesh, 2u)}});
    {
        const auto &procedure = r.get<const CookPlan>(mesh);
        expect(procedure.Nodes.size() == 2u && procedure.DisplayNode == 0u && ActiveIndex(r, procedure) == 0u);
    }
    expect(f.Shaped(mesh) == source);
    // Deleting a displayed Join displays the node its first input reads.
    const auto box = f.Append(mesh, KindOf<primitive::Cube>);
    const auto join = f.Append(mesh, KindOf<Join>);
    f.Do(Link{mesh, f.Node(mesh, 0u), 0u, join, 0u});
    f.Do(Link{mesh, box, f.Out(box), join, 1u});
    f.Do(SetDisplay{mesh, join});
    expect(f.Shaped(mesh) != source);
    f.Do(DeleteNodes{mesh, {join}});
    expect(r.get<const GeometryProcedure>(mesh).DisplayNode == f.Node(mesh, 0u) && f.Shaped(mesh) == source);

    // Connecting extends the displayed path; disconnecting returns to the severed wire's source, including through nested ports and bypass.
    for (const int depth : {0, 2}) {
        const auto owner = f.AddMesh(primitive::Cube{}), root = f.Node(owner, 0u);
        const auto cube = f.Shaped(owner);
        const auto bevel = f.Append(owner, KindOf<action::mesh::Bevel>), tail = f.Append(owner, KindOf<TransformGeometry>);
        f.Do(UpdateOf<&TransformGeometry::Delta, &Transform::P>(tail, vec3{1, 2, 3}));
        f.Do(Link{owner, bevel, f.Out(bevel), tail, 0u});
        auto input = bevel;
        uint16_t socket = 0u;
        for (int level = 0; level < depth; ++level) {
            f.Do(action::nodes::GroupNodes{owner, level == 0 ? std::vector{bevel, tail} : std::vector{input}});
            const auto group = r.get<const GeometryProcedure>(owner).Last;
            f.Do(action::nodes::ExposeSocket{group, input, socket, "Geometry"});
            input = group;
            socket = r.get<const SubnetInterface>(group).Ports.back().Id;
        }
        const auto displayed = [&] { return ResolveSocket(r, DisplayLink(r, r.get<const GeometryProcedure>(owner))).Source.FromNode; };
        for (const bool bypass : {false, true}) {
            f.Do(action::nodes::SetBypass{bevel, bypass});
            expect(r.get<const GeometryProcedure>(owner).DisplayNode == root && f.Shaped(owner) == cube);
            f.Do(Link{owner, root, f.Out(root), input, socket});
            expect(displayed() == tail && f.Shaped(owner) != cube);
            const auto result = f.Shaped(owner);
            f.Do(action::nodes::Unlink{owner, tail, 0u});
            expect(r.get<const GeometryProcedure>(owner).DisplayNode == bevel);
            f.Do(Link{owner, bevel, f.Out(bevel), tail, 0u});
            expect(displayed() == tail && f.Shaped(owner) == result);
            f.Do(action::nodes::Unlink{owner, input, socket});
            expect(r.get<const GeometryProcedure>(owner).DisplayNode == root && f.Shaped(owner) == cube);
        }
        if (depth) {
            f.Do(action::nodes::SetBypass{input, true});
            f.Do(Link{owner, root, f.Out(root), input, socket});
            f.Do(action::nodes::Unlink{owner, tail, 0u});
            expect(r.get<const GeometryProcedure>(owner).DisplayNode == input && f.Shaped(owner) == cube);
            f.Do(Link{owner, bevel, f.Out(bevel), tail, 0u});
            f.Do(action::nodes::Unlink{owner, input, socket});
            expect(r.get<const GeometryProcedure>(owner).DisplayNode == root && f.Shaped(owner) == cube);
            f.Do(action::nodes::SetBypass{input, false});
        }
        f.Do(AddNode{owner, KindOf<Join>, {}, ContainerOf(r, bevel)});
        const auto branch = r.get<const GeometryProcedure>(owner).Last;
        f.Do(Link{owner, bevel, f.Out(bevel), branch, 0u});
        f.Do(Link{owner, root, f.Out(root), input, socket});
        expect(r.get<const GeometryProcedure>(owner).DisplayNode == bevel);
        f.Do(Link{owner, tail, f.Out(tail), branch, 1u});
        f.Do(action::nodes::Unlink{owner, branch, 1u});
        expect(r.get<const GeometryProcedure>(owner).DisplayNode == bevel);
    }
}

// Editing input selections keeps Display and the viewport selection independent, and commits or cancels the whole task.
void ViewportSelection(ProcedureFixture &f) {
    auto &r = f.R;
    for (const auto [kind, vertices] : {std::pair{KindOf<action::mesh::Extrude>, false}, {KindOf<action::mesh::Bevel>, false}, {KindOf<action::mesh::Bevel>, true}, {KindOf<TransformElements>, false}}) {
        const auto mesh = kind == KindOf<action::mesh::Bevel> ? f.AddMesh(primitive::Cube{}) : f.AddGenerated(primitive::Cube{});
        const auto node = f.Append(mesh, kind);
        if (vertices) f.Do(UpdateOf<&action::mesh::Bevel::Vertices>(node, true));
        f.Route(mesh, {f.Node(mesh, 0u), node});
        if (kind == KindOf<TransformElements>) f.Do(action::nodes::GroupNodes{mesh, {node}});
        const auto display = r.get<const GeometryProcedure>(mesh).DisplayNode;
        for (const auto mode : {InteractionMode::Object, InteractionMode::Edit}) {
            f.Do(action::view::SetInteractionMode{mode});
            if (mode == InteractionMode::Edit) {
                f.Do(action::view::SetEditMode{Element::Vertex});
                f.Do(action::selection::ApplyEditElementClick{false, ElementHit{mesh, *GetMesh(r, mesh).VertexAt(0u)}});
            }
            for (const bool cancel : {true, false}) {
                const auto before = f.Shaped(mesh);
                const auto prior_elements = EditorGeometrySelection(f.Store(), f.Output(mesh));
                const auto present = f.P->History.Present;
                const auto selected = SelectedIn(r);
                const auto element = r.get<const EditMode>(state::Document).Value;
                const auto edit = [&](auto a) {
                    action::Emit(std::move(a));
                    f.P->Frame(action::Drain());
                    f.Audit();
                    expect(f.P->HasStaged() && f.P->History.Present == present);
                    expect(r.get<const GeometryProcedure>(mesh).DisplayNode == display);
                };
                edit(action::nodes::EditSelection{node, f.Object(mesh)});
                const auto input = r.get<const SelectionEditing>(state::Document).Mesh;
                expect(GetMesh(r, input).FaceCount() == 6u);
                edit(action::view::SetEditMode{Element::Face});
                edit(action::selection::DeselectAll{});
                expect(SelectedCount(r.get<const InlineSelection>(node).Faces) == 0u);
                f.Render();
                const Mesh cage = GetMesh(r, input);
                he::FH face;
                for (const auto candidate : cage.faces()) {
                    if (Dot(cage.GetNormal(candidate), f.View().CameraPosition - cage.CalcFaceCentroid(candidate)) > 0.f) {
                        face = candidate;
                        break;
                    }
                }
                const auto clip = f.View().ViewProj * vec4{cage.CalcFaceCentroid(face), 1.f};
                const auto hit = RunEditElementPick(r, FirstViewport(r), Element::Face, SelectionPixel(f.View().ViewportSize, {.5f + .5f * clip.x / clip.w, .5f - .5f * clip.y / clip.w}));
                expect(fatal(hit.has_value()));
                expect(hit->MeshEntity == input);
                edit(action::selection::ApplyEditElementClick{false, hit});
                expect(SelectedCount(r.get<const InlineSelection>(node).Faces) == 1u);
                const auto output = GetMesh(r, mesh);
                expect(output.HasClosedSurface());
                if (kind == KindOf<action::mesh::Bevel>) {
                    const auto shaped = f.Shaped(mesh);
                    const auto selected = f.Store().GetSelectedElements(f.Output(input), Element::Vertex);
                    for (const auto v : cage.vertices()) {
                        if (selected.Contains(*v)) continue;
                        const auto p = cage.GetPosition(v);
                        expect(std::ranges::contains(shaped.Vertices, Corner{p.x, p.y, p.z}));
                    }
                }
                edit(action::selection::SelectAll{});
                expect(SelectedCount(r.get<const InlineSelection>(node).Faces) == 6u);
                if (cancel) {
                    action::Cancel();
                    f.P->Frame(action::Drain());
                    expect(f.P->History.Present == present && f.Shaped(mesh) == before);
                } else {
                    f.Do(action::nodes::FinishSelection{});
                    expect(f.P->History.Nodes[*f.P->History.Present].Parent == present);
                }
                expect(!f.P->HasStaged() && !r.all_of<SelectionEditing>(state::Document));
                expect(SelectedIn(r) == selected && r.get<const Interaction>(state::Document).Mode == mode && r.get<const EditMode>(state::Document).Value == element);
                expect(r.get<const GeometryProcedure>(mesh).DisplayNode == display);
                const auto restored = EditorGeometrySelection(f.Store(), f.Output(mesh));
                expect(restored.Vertices == prior_elements.Vertices && restored.Edges == prior_elements.Edges && restored.Faces == prior_elements.Faces);
            }
        }
    }
}

// An Unsubdivide node's first pass publishes the selection the kernel publishes, and its second pass reads that selection as the viewport operator's second iteration does.
// The step derives the edges and faces of the last pass's vertex selection.
Graph UnsubdivideSelection(ProcedureFixture &f) {
    const auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Plane{});
    f.EnterEdit(Element::Vertex);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{.Cuts = 3u});
    const auto drawn = f.Output(mesh);
    const auto selected = EditorGeometrySelection(meshes, drawn);
    const auto pass = [&](uint32_t clone, const GeometrySelection &input) {
        auto task = UnsubdivideTask(Mesh{meshes, clone}, input);
        expect(fatal(task.has_value()));
        return PublishedKernel(f, clone, std::move(*task));
    };
    GeometrySelection first, second;
    const auto once = Reference(f, drawn, [&](uint32_t clone) { return first = pass(clone, selected); });
    const auto twice = Reference(f, drawn, [&](uint32_t clone) { return second = pass(clone, pass(clone, selected)); });
    f.Do(action::mesh::Unsubdivide{1u});
    const auto published = EditorGeometrySelection(meshes, f.Output(mesh));
    expect(published.Vertices == first.Vertices && published.Edges == first.Edges && published.Faces == first.Faces);
    expect(f.Shaped(mesh) == once.Output);
    f.Undo();
    f.Do(action::mesh::Unsubdivide{2u});
    return {
        mesh,
        [mesh, twice, second](ProcedureFixture &f) { expect(f.Shaped(mesh) == twice.Output && EditorGeometrySelection(f.Store(), f.Output(mesh)).Vertices == second.Vertices); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Unsubdivide::Iterations>(f.Node(mesh, 2u), 1u)); },
    };
}

// An instancer without an Instance input faults and passes its points through, and publishes instances once the input is linked.
Graph UnlinkedInstance(ProcedureFixture &f) {
    const auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto vertices = Mesh{f.Store(), f.Output(mesh)}.VertexCount();
    const auto to_points = f.Append(mesh, KindOf<MeshToPoints>);
    const auto instancer = f.Append(mesh, KindOf<InstanceOnPoints>);
    f.Route(mesh, {f.Node(mesh, 0u), to_points, instancer});
    expect(r.all_of<NodeError>(instancer) && r.get<const NodeError>(instancer).Fault == NodeFault::NoMesh);
    const Mesh points{f.Store(), f.Output(mesh)};
    const auto *instances = r.try_get<const ProcedureInstances>(mesh);
    expect(points.VertexCount() == vertices && points.FaceCount() == 0u && (!instances || instances->Entities.empty()));
    f.Do(Link{mesh, f.Node(mesh, 0u), 0u, instancer, 1u});
    return {
        mesh,
        [mesh, instancer, vertices](ProcedureFixture &f) { expect(!f.R.any_of<NodeError>(instancer) && f.R.get<const ProcedureInstances>(mesh).Entities.size() == vertices); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(action::nodes::Unlink{mesh, f.Node(mesh, 2u), 1u}); },
    };
}

// The source clone takes lower blocks than the generated cube and the Join appends them after the cube's, so the joined block table is not in ascending order.
// A Compare linked into a Delete still selects the cube's upper vertices in its higher block.
Graph JoinedSelection(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto cube = f.Append(mesh, KindOf<primitive::Cube>);
    const auto join = f.Append(mesh, KindOf<Join>);
    const auto position = f.Append(mesh, KindOf<Position>);
    const auto compare = f.Append(mesh, KindOf<Compare>);
    const auto remove = f.Append(mesh, KindOf<action::mesh::Delete>);
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(cube, vec3{1.f, 2.f, 1.f}));
    f.Do(UpdateOf<&Compare::Component>(compare, VectorComponent::Y));
    f.Do(UpdateOf<&Compare::B>(compare, 1.5f));
    f.Do(Link{mesh, cube, f.Out(cube), join, 0u});
    f.Do(Link{mesh, f.Node(mesh, 0u), 0u, join, 1u});
    f.Do(Link{mesh, position, 0u, compare, 0u});
    f.Do(Link{mesh, compare, f.Out(compare), remove, 1u});
    f.Route(mesh, {join, remove});
    return {
        mesh,
        [mesh, compare, remove](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            expect(!f.R.any_of<NodeError>(compare) && !f.R.any_of<NodeError>(remove));
            const auto id = f.Output(mesh);
            const auto blocks = meshes.GetBlockList(id, MeshStore::ElementDomain::Vertex).Blocks;
            expect(fatal(blocks.size() == 2u));
            expect(blocks.back() < blocks.front());
            const Mesh output{meshes, id};
            expect(output.VertexCount() == 12u);
            for (const auto v : output.vertices()) expect(output.GetPosition(v).y < 1.5f);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&Compare::B>(f.Node(mesh, 4u), .5f)); },
    };
}

// A change to a node committed in an earlier step restores the record to the result of the node before it.
void ColdRecook(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    const auto extrude = f.Append(mesh, KindOf<action::mesh::Extrude>);
    f.Route(mesh, {f.Node(mesh, 0u), subdivide, extrude});
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide, 2u));
}

// A change to the first node after the Source restores the record to the Source node's result, with the position edit and the topology node after it run again.
void NodeResultRecook(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    const auto to_sphere = f.Append(mesh, KindOf<ToSphere>);
    const auto extrude = f.Append(mesh, KindOf<action::mesh::Extrude>);
    f.Do(UpdateOf<&ToSphere::Factor>(to_sphere, .5f));
    f.Route(mesh, {f.Node(mesh, 0u), subdivide, to_sphere, extrude});
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide, 2u));
}

// Deleting the displayed tip node restores the record in place to its input node's result with one cook, which leaves the Source node's state untouched.
void DeleteTipRestores(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cube{});
    const auto root = f.Node(mesh, 0u);
    const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    const auto extrude = f.Append(mesh, KindOf<action::mesh::Extrude>);
    f.Route(mesh, {root, subdivide});
    f.Route(mesh, {subdivide, extrude});
    const auto record = f.Output(mesh);
    const auto source = f.SourceOf(mesh);
    const auto cooks = Cooks(f.R);
    f.Do(DeleteNodes{mesh, {extrude}});
    expect(Cooks(f.R) == cooks + 1u);
    expect(f.Output(mesh) == record && f.SourceOf(mesh) == source);
}

// Grouping changes presentation only, and nested interfaces edit the same underlying fields.
void SceneNetworks(ProcedureFixture &f) {
    using namespace action::nodes;
    const auto cube = f.AddCube();
    const auto light = f.Add(action::object::AddLight{std::make_unique<ObjectCreateInfo>()});
    f.Do(AssignReference{{{light}, ReferenceKind::Parent}, {cube}});
    f.Do(action::animation::AddAnimation{"Animation"});
    for (const auto network : {FrameScene(f.R), state::Document}) {
        const auto items = NetworkItems(f.R, network);
        const auto shapes = Live::DrawnShapes(f);
        const auto cooks = Cooks(f.R);
        std::vector<state::Entity> groups;
        auto members = items;
        for (int depth = 0; depth < 2; ++depth) {
            f.Do(GroupItems{network, members});
            const auto group = f.R.get<const NetworkActive>(network).Item.Entity;
            groups.push_back(group);
            members = {{group}};
            f.Do(MoveNodes{network, {{{group}, {uint32_t(depth + 2), 3u}}}});
            expect(NetworkLeaves(f.R, group) == items);
            if (network != state::Document) {
                f.Do(ExposePort{group, PromotedField{light, state::Key<PunctualLight>(), action::detail::FieldOffset<&PunctualLight::Intensity>()}, std::make_unique<std::string>("Intensity")});
                f.Do(SetPortValue{group, f.R.get<const SubnetInterface>(group).Ports.back().Id, {std::bit_cast<uint32_t>(17.f + depth)}});
                expect(f.R.get<const PunctualLight>(light).Intensity == 17.f + depth);
            }
        }
        for (const auto group : groups | std::views::reverse) f.Do(UngroupItems{group});
        expect(NetworkItems(f.R, network) == items && Live::DrawnShapes(f) == shapes && Cooks(f.R) == cooks);
        expect(ParentOrNull(f.R, light) == cube);
    }
}

// Interactive and read-only drawing produce the same UI on every page, including after history navigation.
void CanvasEquivalence(ProcedureFixture &f) {
    const auto mesh = f.AddGenerated(primitive::Cube{});
    const auto node = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Route(mesh, {f.Node(mesh, 0), node});
    f.Do(action::nodes::GroupNodes{mesh, {node}});
    const auto subnet = f.R.get<const GeometryProcedure>(mesh).Last;
    f.Do(action::physics::AddJointDef{});
    f.Do(action::animation::AddAnimation{"Animation"});
    auto *context = ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = {1400, 900};
    io.Fonts->Build();
    NodesPanel panel{{"Nodes"}};
    Panel parameters{"Parameters"};
    const auto frame = [&](bool interactive = false) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({950, 850});
        DrawNodesPanel(f.R, panel, interactive);
        ImGui::SetNextWindowPos({950, 0});
        ImGui::SetNextWindowSize({450, 850});
        DrawParametersPanel(f.R, panel, parameters, interactive);
        ImGui::Render();
        std::vector<std::byte> drawn;
        for (const auto *list : ImGui::GetDrawData()->CmdLists) {
            drawn.append_range(std::as_bytes(std::span{list->VtxBuffer.Data, size_t(list->VtxBuffer.Size)}));
            drawn.append_range(std::as_bytes(std::span{list->IdxBuffer.Data, size_t(list->IdxBuffer.Size)}));
        }
        expect(!drawn.empty());
        return drawn;
    };
    const auto cooks = Cooks(f.R);
    for (const auto network : {FrameScene(f.R), state::Document, mesh, subnet}) {
        EnterNetwork(panel, network);
        frame();
        frame();
        const auto drawn = frame(true);
        expect(frame() == drawn);
        f.Undo();
        f.Redo();
        expect(frame() == drawn);
        if (network == state::Document) {
            for (const auto item : NetworkItems(f.R, network)) {
                f.Do(SelectItems{network, item, {item}});
                frame();
            }
        }
    }
    expect(Cooks(f.R) == cooks);
    ImGui::DestroyContext(context);
}

state::Entity AnimatedAssembly(ProcedureFixture &f) {
    const auto root = f.AddEmpty("Assembly");
    const auto mesh = f.AddGenerated(primitive::Cube{});
    const auto operand = f.Object(mesh);
    const auto result = f.AddGenerated(primitive::Plane{});
    const auto info = f.Append(result, KindOf<procedure::ObjectInfo>);
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(info, operand));
    f.Do(SetDisplay{result, info});
    const auto light = f.Add(action::object::AddLight{std::make_unique<ObjectCreateInfo>()});
    for (const auto child : {operand, f.Object(result), light}) f.Do(action::object::SetParent{child, root});
    f.R.patch<Animations>(state::Document, [](auto &a) { a.Names = {"Animation"}; a.Active = 0; });
    f.R.emplace<AnimationClips>(root, std::vector<AnimationClip>{{0, {{animation::Target<&Transform::P>(), AnimationInterpolation::Linear, {0, 1}, {0, 0, 0, 2, 0, 0}, {0, 0, 0}}}}});
    f.R.emplace<AnimationClips>(light, std::vector<AnimationClip>{{0, {{animation::Target<&PunctualLight::Intensity>(), AnimationInterpolation::Linear, {0, 1}, {2, 8}, {2}}}}});
    f.Commit("Animate assembly");
    return root;
}
state::Entity Mechanism(ProcedureFixture &f) {
    const auto root = f.AddEmpty("Mechanism"), a = f.AddCube(), b = f.AddCube();
    for (const auto child : {a, b}) {
        f.Do(action::object::SetParent{child, root});
        f.Do(action::physics::SetMotionType{action::physics::SetMotionType::Type::Kinematic, child});
    }
    f.Do(UpdateOf<&Transform::P>(b, vec3{3, 0, 0}));
    const auto definition = f.R.create();
    f.R.emplace<PhysicsJointDef>(definition);
    f.R.emplace<PhysicsJoint>(a, b, definition);
    f.R.emplace<TriggerNodes>(root, std::vector{a, b});
    ModalModes modes;
    modes.Freqs = {440};
    modes.T60s = {1};
    modes.Shapes = {{{1, 0, 0}}};
    modes.Positions = {{-1, -1, -1}};
    modes.Vertices = {0};
    f.R.emplace<ModalModes>(a, modes);
    f.R.emplace<SoundVerticesModel>(a, SoundVerticesModel::Modal);
    f.R.emplace<ModalGain>(a);
    f.Commit("Bind mechanism domains");
    return root;
}
state::Entity ImportedRig(ProcedureFixture &f) {
    f.Do(action::io::LoadGltf{std::filesystem::path{MESHEDITOR_SOURCE_DIR} / "external/glTF-Sample-Assets/Models/RiggedSimple/glTF/RiggedSimple.gltf"});
    auto root = *f.R.view<const ArmatureObject>().begin();
    while (ParentOrNull(f.R, root) != state::Null) root = ParentOrNull(f.R, root);
    return root;
}

// Detaching a nested instance preserves its geometry and animated appearance; source edits then leave it independent.
void Hierarchies(ProcedureFixture &f, Fixture &other) {
    for (const auto [name, build] : std::array<std::pair<const char *, state::Entity (*)(ProcedureFixture &)>, 3>{{{"Assembly", AnimatedAssembly}, {"Mechanism", Mechanism}, {"Rig", ImportedRig}}}) {
        Scenario(f, other, std::string{"Hierarchy"} + name, [&](ProcedureFixture &f) {
            const auto source = build(f);
            const auto inner = f.Add(action::object::InstantiateHierarchy{source});
            const auto root = f.Add(action::object::InstantiateHierarchy{inner});
            f.Do(UpdateOf<&Transform::P>(root, vec3{2, 0, 0}));
            const auto members = f.R.get<const HierarchyInstance>(root).Members;
            auto edited = state::Null;
            for (const auto &member : members) {
                if (const auto *object = f.R.try_get<const Instance>(member.Entity); object && DrawnStoreId(f.R, object->Entity)) {
                    expect(f.Shaped(object->Entity) == f.Shaped(f.R.get<const Instance>(member.Source).Entity));
                    edited = HierarchySource(f.R, member.Entity);
                }
            }
            for (const auto &member : members) f.Do(action::object::SetVisible{false, HierarchySource(f.R, member.Entity)});
            f.Do(action::selection::Select{inner});
            f.Do(action::object::Delete{});
            f.Do(action::selection::DeselectAll{});
            std::array<std::vector<std::byte>, 2> images;
            for (int frame = 0; frame < 2; ++frame) {
                f.Do(action::timeline::SetFrame{1 + frame * 30});
                if (Render) images[frame] = f.Image();
            }
            f.Do(action::object::MakeIndependent{root});
            for (int frame = 0; frame < 2; ++frame) {
                f.Do(action::timeline::SetFrame{1 + frame * 30});
                if (Render) expect(f.Image() == images[frame]);
            }
            const auto independent = Live::DrawnShapes(f);
            f.Do(UpdateOf<&Transform::P>(edited, vec3{9, 0, 0}));
            expect(Live::DrawnShapes(f) == independent);
            if (Render) expect(f.Image() == images.back());
        });
    }
}

void ObjectDependencies(ProcedureFixture &f) {
    const auto a = f.AddGenerated(primitive::Cube{});
    const auto root = f.Node(a, 0u);
    const auto b = f.AddGenerated(primitive::Plane{});
    const auto bi = f.Append(b, KindOf<procedure::ObjectInfo>);
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(bi, f.Object(a)));
    f.Do(SetDisplay{b, bi});
    const auto c = f.AddGenerated(primitive::Plane{});
    const auto ci = f.Append(c, KindOf<procedure::ObjectInfo>);
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(ci, f.Object(b)));
    f.Do(SetDisplay{c, ci});
    auto cooks = Cooks(f.R);
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(root, vec3{2.f, .5f, 1.f}));
    expect(Cooks(f.R) == cooks + 3u);
    expect(f.Shaped(a) == f.Shaped(b) && f.Shaped(a) == f.Shaped(c));
    const auto dormant = f.Append(a, KindOf<primitive::Cube>);
    cooks = Cooks(f.R);
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(dormant, vec3{4.f, 1.f, 2.f}));
    expect(Cooks(f.R) == cooks);
    const auto cycle = f.Append(a, KindOf<procedure::ObjectInfo>);
    cooks = Cooks(f.R);
    f.Fails(UpdateOf<&procedure::ObjectInfo::Object>(cycle, f.Object(c)));
    expect(f.R.get<const procedure::ObjectInfo>(cycle).Object == state::Null && Cooks(f.R) == cooks);
    f.Do(action::selection::Select{f.Object(c)});
    f.Do(action::selection::ExtendActive{f.Object(a)});
    const auto node_count = f.R.get<const CookPlan>(a).Nodes.size();
    f.Fails(action::mesh::Boolean{BooleanOperation::Difference});
    expect(f.R.get<const CookPlan>(a).Nodes.size() == node_count && Cooks(f.R) == cooks);
    f.Do(SetDisplay{b, f.Node(b, 0u)});
    cooks = Cooks(f.R);
    f.Do(UpdateOf<&primitive::Cube::HalfExtents>(root, vec3{1.f, 1.f, 3.f}));
    expect(Cooks(f.R) == cooks + 1u);
    expect(f.Shaped(b) == f.Shaped(c));
    f.Do(SetDisplay{b, bi});
    expect(f.Shaped(a) == f.Shaped(b) && f.Shaped(a) == f.Shaped(c));
    const auto object_a = f.Object(a);
    f.Do(action::selection::Select{object_a});
    f.Do(action::object::Delete{});
    expect(f.R.all_of<NodeError>(bi));
}

void HiddenSharedGeometry(ProcedureFixture &f) {
    const auto original = f.AddCube();
    const auto mesh = f.R.get<const Instance>(original).Entity;
    const auto shape = f.Shaped(mesh);
    const auto hidden = f.Add(action::object::DuplicateLinked{});
    f.Do(action::object::SetVisible{false, hidden});
    f.Do(action::selection::Select{original});
    f.Do(action::object::Delete{});
    expect(f.R.valid(mesh) && f.R.get<const Instance>(hidden).Entity == mesh);
    expect(f.Shaped(mesh) == shape);
    f.Do(action::object::SetVisible{true, hidden});
    expect(f.Shaped(mesh) == shape);
    f.Do(action::selection::Select{hidden});
    f.Do(action::object::Delete{});
    expect(!f.R.valid(mesh));
}

void ProceduralHierarchies(ProcedureFixture &f) {
    auto &r = f.R;
    const auto source = f.AddEmpty("Prototype");
    const auto light = f.Add(action::object::AddLight{std::make_unique<ObjectCreateInfo>()});
    f.Do(action::object::SetParent{light, source});
    const auto mesh = f.AddGenerated(primitive::Plane{});
    const auto producer = f.Object(mesh);
    const auto points = f.Append(mesh, KindOf<MeshToPoints>);
    const auto instances = f.Append(mesh, KindOf<InstanceOnPoints>);
    f.Do(UpdateOf<&InstanceOnPoints::Prototype>(instances, source));
    f.Route(mesh, {f.Node(mesh, 0u), points, instances});
    const auto roots = r.get<const ProcedureInstances>(mesh).Entities;
    expect(roots.size() == 4u);
    for (const auto root : roots) {
        expect(r.all_of<HierarchyInstance>(root));
        expect(ParentOrNull(r, root) == producer);
        expect(r.get<const HierarchyInstance>(root).Members.size() == 2u);
    }
    f.Do(UpdateOf<&PunctualLight::Intensity>(light, 23.f));
    expect(r.get<const ProcedureInstances>(mesh).Entities == roots);
    for (const auto root : roots) {
        const auto &members = r.get<const HierarchyInstance>(root).Members;
        const auto copied = std::ranges::find(members, light, &HierarchyInstance::Member::Source)->Entity;
        expect(r.get<const PunctualLight>(copied).Intensity == 23.f);
    }
    f.Fails(UpdateOf<&InstanceOnPoints::Prototype>(instances, producer));
    const auto realize = f.Append(mesh, KindOf<RealizeInstances>);
    f.Do(Link{mesh, instances, f.Out(instances), realize, 0});
    f.Do(SetDisplay{mesh, realize});
    expect(r.all_of<NodeError>(realize));
    expect(r.get<const ProcedureInstances>(mesh).Entities == roots);
    f.Do(SetDisplay{mesh, f.Node(mesh, 0u)});
    for (const auto root : roots) expect(!r.valid(root));
    f.Do(action::selection::Select{source});
    f.Do(action::object::Delete{});
    expect(r.all_of<HierarchyDefinition>(source));
    expect(r.valid(light));
    f.Do(SetDisplay{mesh, instances});
    expect(r.get<const ProcedureInstances>(mesh).Entities.size() == 4u);
    f.Do(SetDisplay{mesh, f.Node(mesh, 0u)});
    f.Do(UpdateOf<&InstanceOnPoints::Prototype>(instances, state::Null));
    expect(!r.valid(source) && !r.valid(light));
}

void LinkedSkinPoses(ProcedureFixture &f) {
    f.Do(action::io::LoadGltf{std::filesystem::path{MESHEDITOR_SOURCE_DIR} / "external/glTF-Sample-Assets/Models/RiggedSimple/glTF/RiggedSimple.gltf"});
    auto &r = f.R;
    const auto skinned = *r.view<const ArmatureModifier>().begin();
    const auto modifier = r.get<const ArmatureModifier>(skinned);
    const auto first = modifier.ArmatureObjectEntity;
    f.Do(action::selection::Select{first});
    const auto second = f.Add(action::object::DuplicateLinked{});
    if (r.all_of<HierarchyInstance>(second)) f.Do(action::object::MakeIndependent{second});
    f.Do(action::selection::Select{skinned});
    const auto mesh_copy = f.Add(action::object::DuplicateLinked{});
    r.patch<ArmatureModifier>(mesh_copy, [&](auto &binding) { binding.ArmatureObjectEntity = second; });
    f.Commit("Bind the copied skin to its armature occurrence");
    const auto target = f.Add(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Transform = {.P = {3.f, 0.f, 0.f}}})});
    const auto second_bone = r.get<const ArmatureObject>(second).BoneEntities.front();
    f.Do(action::bone::AddConstraint{action::bone::BoneConstraintKind::CopyTransforms, second_bone});
    f.Do(action::bone::SetConstraintTarget{0u, target, second_bone});
    const auto check = [&] {
        const auto &a = r.get<const ArmaturePoseState>(first), &b = r.get<const ArmaturePoseState>(second);
        const auto &layout = r.get<const PosedLayout>(r.get<const Instance>(skinned).Entity);
        expect(std::ranges::contains(layout.Armatures, first) && std::ranges::contains(layout.Armatures, second));
        expect(!a.GpuDeformRanges.empty() && a.GpuDeformRanges.size() == b.GpuDeformRanges.size()) << fatal;
        const auto &buffers = r.Resources.get<const GpuBuffers>();
        for (uint32_t skin = 0; skin < a.GpuDeformRanges.size(); ++skin) {
            expect(a.GpuDeformRanges[skin].Offset != b.GpuDeformRanges[skin].Offset);
            const auto one = buffers.ArmatureDeformBuffer.Get(a.GpuDeformRanges[skin]);
            const auto two = buffers.ArmatureDeformBuffer.Get(b.GpuDeformRanges[skin]);
            std::vector<mat4> expected(two.size());
            ComputeDeformMatrices(r.get<const Armature>(modifier.ArmatureEntity), skin, b.BonePoseWorld, expected);
            expect(std::ranges::equal(two, expected));
            expect(!std::ranges::equal(one, two));
        }
    };
    check();
    f.Undo();
    f.Redo();
    check();
}

constexpr std::array<GraphRow, 21> Graphs{{
    {"ExtrudeSubdivide", ExtrudeSubdivide},
    {"CapInset", CapInset},
    {"ScalePositions", ScalePositions},
    {"CompareSelectsExtrusion", CompareSelectsExtrusion},
    {"NamedAttributeReads", NamedAttributeReads},
    {"StorePromotes", StorePromotes},
    {"StoreSelectionDelete", StoreSelectionDelete},
    {"DomainMismatch", DomainMismatch},
    {"JoinGenerated", JoinGenerated},
    {"HeatSubdivide", HeatSubdivide},
    {"BooleanIntersection", BooleanIntersection},
    {"BooleanAction", BooleanAction},
    {"GeneratorEdit", GeneratorEdit},
    {"ObjectInfo", ObjectInfo},
    {"InstancesBorrowedPublish", Instances<true, false>},
    {"InstancesGeneratedRealize", Instances<false, true>},
    {"NodeEditing", NodeEditing},
    {"SourceInstances", SourceInstances},
    {"UnlinkedInstance", UnlinkedInstance},
    {"UnsubdivideSelection", UnsubdivideSelection},
    {"JoinedSelection", JoinedSelection},
}};
} // namespace

void TestProcedures() {
    ProcedureFixture f;
    Fixture other;
    OperatorKinds(f, other);
    Survives(f, other, Graphs);
    TestOperators(f, other);
    Hierarchies(f, other);
    Scenario(f, other, "SceneNetworks", SceneNetworks);
    Scenario(f, other, "CanvasEquivalence", CanvasEquivalence);
    Scenario(f, other, "ParameterSockets", ParameterSockets);
    Scenario(f, other, "SourceStates", SourceStates);
    Scenario(f, other, "DisplayedNode", DisplayedNode);
    Scenario(f, other, "ViewportSelection", ViewportSelection);
    Scenario(f, other, "ColdRecook", ColdRecook);
    Scenario(f, other, "NodeResultRecook", NodeResultRecook);
    Scenario(f, other, "ExtrudeDisplacement", ExtrudeDisplacement);
    Scenario(f, other, "DeleteTipRestores", DeleteTipRestores);
    Scenario(f, other, "HiddenSharedGeometry", HiddenSharedGeometry);
    Scenario(f, other, "ProceduralHierarchies", ProceduralHierarchies);
    Scenario(f, other, "LinkedSkinPoses", LinkedSkinPoses);
    Scenario(f, other, "ObjectDependencies", ObjectDependencies);
}
