#include "procedure/Procedure.h"
#include "ProjectFixture.h"
#include "TestMeshOrdinals.h"
#include "TopologyCases.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshClone.h"
#include "mesh/Primitives.h"
#include "numeric/MatrixMath.h"
#include "object/ObjectOps.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "scene/SceneGraph.h"
#include "viewport/ViewportEvents.h"

using boost::ut::expect, boost::ut::fatal;

namespace {
using namespace procedure;
using action::UpdateOf;
using action::nodes::AddNode, action::nodes::AddProcedure, action::nodes::ApplyProcedure, action::nodes::DeleteNodes, action::nodes::DuplicateNodes, action::nodes::InsertNode, action::nodes::Link, action::nodes::SelectNodes, action::nodes::SetAttributeName, action::nodes::SetInlineSelection;
using ::test::Op, ::test::Patchwork, ::test::PlanCase, ::test::Region, ::test::TopologyCases;

struct ProcedureFixture : Fixture {
    // Does the action, which must fail.
    void Fails(auto a) {
        P->Do(action::MakeAction(std::move(a)));
        expect(!R.Resources.get<action::Errors>().Messages.empty());
        R.Resources.get<action::Errors>().Messages.clear();
    }
    state::Entity AddMesh(PrimitiveShape shape) {
        Do(action::object::AddMeshPrimitive{std::move(shape), std::make_unique<MeshInstanceCreateInfo>()});
        return R.get<const Instance>(FindActiveEntity(R)).Entity;
    }
    std::pair<uint32_t, state::Entity> Append(state::Entity mesh_entity, uint16_t kind) {
        Do(AddNode{mesh_entity, kind, {}});
        const auto &nodes = R.get<const GeometryProcedure>(mesh_entity).Nodes;
        return {uint32_t(nodes.size() - 1u), nodes.back().Entity};
    }
    // Links each node's first output into the next node's first input.
    void Route(state::Entity mesh_entity, std::initializer_list<uint32_t> nodes) {
        for (auto from = nodes.begin(); from + 1 != nodes.end(); ++from)
            Do(Link{mesh_entity, *from, InputCount[R.get<const GeometryProcedure>(mesh_entity).Nodes[*from].Kind], *(from + 1), 0u});
    }
    state::Entity Node(state::Entity mesh_entity, uint32_t node) const { return R.get<const GeometryProcedure>(mesh_entity).Nodes[node].Entity; }
    uint16_t Out(state::Entity mesh_entity, uint32_t node) const { return InputCount[R.get<const GeometryProcedure>(mesh_entity).Nodes[node].Kind]; }
    void Stage(auto a) {
        action::Emit(std::move(a), action::Phase::Stage);
        P->Frame(action::Drain());
        expect(R.Resources.get<action::Errors>().Messages.empty());
    }
    void CommitGesture() {
        action::Commit();
        P->Frame(action::Drain());
        expect(!P->HasStaged());
        Audit();
        if (Step) Step(true);
    }
    void EnterEdit(Element element) {
        Do(action::view::SetInteractionMode{InteractionMode::Edit});
        Do(action::view::SetEditMode{.Mode = element});
    }
    void Render() {
        SubmitViewports(R);
        WaitForRender(R);
    }
    const SceneViewUBO &View() const { return *reinterpret_cast<const SceneViewUBO *>(R.get<const Viewport>(FirstViewport(R)).SceneViewUBO.Contents().data()); }
    // Renders a frame and clicks the element nearest the world position, adding it to the selection with `toggle`.
    void Click(vec3 position, bool toggle) {
        Render();
        const auto clip = View().ViewProj * vec4{position.x, position.y, position.z, 1.f};
        Do(action::selection::ApplyEditElementClick{{.5f + .5f * clip.x / clip.w, .5f - .5f * clip.y / clip.w}, toggle, std::make_unique<RenderView>(R.get<const Viewport>(FirstViewport(R)).FrameView)});
    }
    uint32_t Source(state::Entity mesh_entity) const { return R.get<const MeshHandle>(mesh_entity).StoreId; }
    state::Entity Object(state::Entity mesh_entity) const {
        for (const auto [e, instance] : R.view<const Instance>().each())
            if (instance.Entity == mesh_entity) return e;
        return state::Null;
    }
    // Return the evaluated procedure record when present, otherwise the source record.
    uint32_t Output(state::Entity mesh_entity) const { return *DrawnStoreId(R, mesh_entity); }
    Shape Shaped(state::Entity mesh_entity) { return Describe(Store(), Output(mesh_entity)); }
    uint32_t Clone(uint32_t id) {
        auto &meshes = Store();
        meshes.PlanClone(Mesh{meshes, id});
        meshes.CommitReserves();
        CloneCopies copies;
        const auto clone = meshes.CloneMeshes(copies, std::array{id}, true).front();
        mtl::ComputeChain chain{meshes.BufferContext()};
        copies.Encode(chain, GetMeshPipelines(R));
        chain.Submit();
        meshes.RebuildSelectionIndex(std::array{clone});
        Audit();
        return clone;
    }
    void Run(uint32_t id, std::vector<MeshTopologyTask> tasks) {
        for (auto &task : tasks) task.SourceId = id;
        expect(std::ranges::all_of(ExecuteGeometryTopologyStages(R, tasks), &GeometryTopologyResult::Changed));
    }
};

// The record and the selected elements a kernel edit leaves on a clone of `id`, where `edit` runs the kernel on the clone and returns the elements it selects.
struct Kernel {
    Shape Output, Selected;
};
Kernel Reference(ProcedureFixture &f, uint32_t id, auto &&edit) {
    auto &meshes = f.Store();
    const auto clone = f.Clone(id);
    const GeometrySelection selected = edit(clone);
    Kernel kernel{Describe(meshes, clone), Describe(meshes, clone, selected)};
    meshes.Release(clone);
    f.Commit("Release reference");
    return kernel;
}
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
GeometrySelection Selection(const MeshStore &meshes, uint32_t id) {
    GeometrySelection selection;
    for (const auto [element, handles] : {std::pair{Element::Vertex, &selection.Vertices}, std::pair{Element::Edge, &selection.Edges}, std::pair{Element::Face, &selection.Faces}})
        meshes.GetSelectedElements(id, element).ForEach([&](uint32_t handle) { handles->push_back(handle); });
    return selection;
}

struct OperatorNode {
    uint16_t Kind;
    void (*Set)(ProcedureFixture &, state::Entity node, const MeshTopologyTask &){};
};
// The node whose cook plans the task, with parameters from the task's op, Param0, Param1, flags, steps and copy translation.
// A merge at a target merges onto the selection's first vertex at its center, as the Center mode does.
// The planned decimation carries no ratio, and its row's ratio is the node's default.
std::optional<OperatorNode> NodeOf(const MeshTopologyTask &task) {
    using enum MeshTopologyOp;
    using namespace action::mesh;
    switch (task.Op) {
        case DeleteVertices:
        case DeleteEdges:
        case DeleteFaces:
        case DeleteOnlyEdgesFaces:
        case DeleteOnlyFaces:
        case DeleteLoose: return OperatorNode{KindOf<Delete>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&Delete::Mode>(n, DeleteMode(t.Op))); }};
        case DissolveVertices:
        case DissolveEdges:
        case DissolveFaces:
        case DissolveLimited:
        case DissolveDegenerate:
            return OperatorNode{KindOf<Dissolve>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    const auto mode = t.Op == DissolveVertices ? DissolveMode::Vertices : t.Op == DissolveEdges ? DissolveMode::Edges :
                                        t.Op == DissolveFaces                                                                   ? DissolveMode::Faces :
                                        t.Op == DissolveLimited                                                                 ? DissolveMode::Limited :
                                                                                                                                  DissolveMode::Degenerate;
                                    f.Do(UpdateOf<&Dissolve::Mode>(n, mode));
                                    f.Do(UpdateOf<&Dissolve::KeepVertices>(n, (t.Flags & TopologyFlagKeepVertices) != 0u));
                                    if (mode == DissolveMode::Limited) f.Do(UpdateOf<&Dissolve::Angle>(n, t.Param0));
                                    if (mode == DissolveMode::Degenerate) f.Do(UpdateOf<&Dissolve::Distance>(n, t.Param0));
                                }};
        case MergeAtTarget:
        case MergeByDistance:
        case MergeCollapse:
            return OperatorNode{KindOf<Merge>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&Merge::Mode>(n, t.Op == MergeByDistance ? MergeMode::ByDistance : t.Op == MergeCollapse ? MergeMode::Collapse :
                                                                                                                                             MergeMode::Center));
                                    if (t.Op == MergeByDistance) f.Do(UpdateOf<&Merge::Distance>(n, t.Param0));
                                }};
        case Decimate: return OperatorNode{KindOf<action::mesh::Decimate>};
        case AddPrimitives: return OperatorNode{KindOf<ConvexHull>};
        case Wireframe:
            return OperatorNode{KindOf<action::mesh::Wireframe>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    using W = action::mesh::Wireframe;
                                    f.Do(UpdateOf<&W::Thickness>(n, t.Param0));
                                    f.Do(UpdateOf<&W::Offset>(n, t.Param1));
                                    f.Do(UpdateOf<&W::Even>(n, (t.Flags & TopologyFlagEvenOffset) != 0u));
                                    f.Do(UpdateOf<&W::Boundary>(n, (t.Flags & TopologyFlagWireBoundary) != 0u));
                                    f.Do(UpdateOf<&W::Relative>(n, (t.Flags & TopologyFlagWireRelative) != 0u));
                                    f.Do(UpdateOf<&W::Replace>(n, (t.Flags & TopologyFlagWireReplace) != 0u));
                                }};
        case BevelEdges:
        case BevelVertices:
            return OperatorNode{KindOf<Bevel>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&Bevel::Width>(n, t.Param0));
                                    f.Do(UpdateOf<&Bevel::Segments>(n, t.Steps));
                                    f.Do(UpdateOf<&Bevel::Vertices>(n, t.Op == BevelVertices));
                                }};
        case TrisToQuads: return OperatorNode{KindOf<action::mesh::TrisToQuads>};
        case Triangulate: return OperatorNode{KindOf<action::mesh::Triangulate>};
        case FlipNormals: return OperatorNode{KindOf<action::mesh::FlipNormals>};
        case SplitGeometry: return OperatorNode{KindOf<Split>};
        case DuplicateGeometry: return OperatorNode{KindOf<Duplicate>};
        case ExtrudeVertices:
        case ExtrudeEdges:
        case ExtrudeFacesIndividual:
            return OperatorNode{KindOf<Extrude>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&Extrude::Mode>(n, t.Op == ExtrudeVertices ? ExtrudeMode::Vertices : t.Op == ExtrudeEdges ? ExtrudeMode::Edges :
                                                                                                                                              ExtrudeMode::FacesIndividual));
                                }};
        case ExtrudeRegion:
            if (!(task.Flags & TopologyFlagTransformCopies)) return OperatorNode{KindOf<Extrude>};
            return OperatorNode{KindOf<ExtrudeRepeat>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&ExtrudeRepeat::Steps>(n, t.Steps));
                                    f.Do(UpdateOf<&ExtrudeRepeat::Offset>(n, t.CopyTranslation));
                                }};
        case Subdivide: return OperatorNode{KindOf<action::mesh::Subdivide>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(n, uint32_t(t.Param0))); }};
        case EdgeSplit: return OperatorNode{task.Flags & TopologyFlagRipSelectCopies ? KindOf<Rip> : KindOf<action::mesh::EdgeSplit>};
        case Poke: return OperatorNode{KindOf<action::mesh::Poke>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&action::mesh::Poke::Offset>(n, t.Param0)); }};
        case InsetRegion:
        case InsetIndividual:
            return OperatorNode{KindOf<Inset>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&Inset::Thickness>(n, t.Param0));
                                    f.Do(UpdateOf<&Inset::Depth>(n, t.Param1));
                                    f.Do(UpdateOf<&Inset::Individual>(n, t.Op == InsetIndividual));
                                    f.Do(UpdateOf<&Inset::Even>(n, (t.Flags & TopologyFlagEvenOffset) != 0u));
                                }};
        case Solidify: return OperatorNode{KindOf<action::mesh::Solidify>, [](ProcedureFixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&action::mesh::Solidify::Thickness>(n, t.Param0)); }};
        default: return std::nullopt;
    }
}

// The output must be the kernel's, from a record of its own, with the source record unchanged.
void ExpectCooksLike(ProcedureFixture &f, state::Entity mesh, uint32_t source_id, const Shape &source, const Shape &reference) {
    expect(f.Shaped(mesh) == reference);
    expect(f.Output(mesh) != source_id && f.Source(mesh) == source_id);
    expect(Describe(f.Store(), source_id) == source);
}

// Every operator kind cooks like its kernel on the patchwork, each generator kind like its primitive, and a fan-out cooks the read branch alone.
void OperatorKinds(ProcedureFixture &f, Fixture &other) {
    constexpr std::array<std::string_view, 6> WithoutNode{"RotateEdges", "ReplaceFaces", "ConnectVertices", "SplitNonplanarFaces", "SplitConcaveFaces", "KeepSelectedFaces"};
    for (const auto &c : TopologyCases) {
        if (std::ranges::contains(WithoutNode, c.Name)) continue;
        Scenario(f, other, std::string{c.Name}, [&c](ProcedureFixture &f) {
            auto &r = f.R;
            auto &meshes = f.Store();
            const auto mesh = AddMesh(r, CreateMesh(r, {.Data = Patchwork()}).StoreId, MeshInstanceCreateInfo{}).first;
            f.Commit("Add patchwork");
            const auto source_id = f.Source(mesh);
            const auto source = Describe(meshes, source_id);
            auto task = PlanCase(c, meshes, Mesh{meshes, source_id});
            expect(fatal(task.has_value()));
            const auto node = NodeOf(*task);
            expect(fatal(node.has_value()));
            f.Do(AddProcedure{mesh});
            const auto [index, entity] = f.Append(mesh, node->Kind);
            if (node->Set) node->Set(f, entity, *task);
            // The region as one mask block per block the source owns in each domain, as CaptureSelection captures a drawn selection.
            const auto region = Region(Mesh{meshes, source_id});
            SetInlineSelection selection{.Node = entity};
            using D = MeshStore::ElementDomain;
            for (const auto [domain, handles] : {std::pair{D::Vertex, &region.Vertices}, std::pair{D::Edge, &region.Edges}, std::pair{D::Face, &region.Faces}}) {
                for (const auto block : meshes.GetBlockList(source_id, domain).Blocks) {
                    auto &words = selection.Blocks.emplace_back();
                    for (const auto h : *handles)
                        if (h / MeshElementBlockSize == block) words[(h % MeshElementBlockSize) / 32u] |= 1u << (h % 32u);
                }
                if (domain == D::Vertex) selection.Vertices = uint32_t(selection.Blocks.size());
                if (domain == D::Edge) selection.Edges = uint32_t(selection.Blocks.size()) - selection.Vertices;
            }
            f.Do(std::move(selection));
            f.Route(mesh, {0u, index, 1u});
            const auto reference = [&](float param0) {
                return Reference(f, source_id, [&](uint32_t clone) {
                           auto planned = *PlanCase(c, meshes, Mesh{meshes, clone});
                           planned.Param0 = param0;
                           f.Run(clone, {planned});
                           return GeometrySelection{};
                       })
                    .Output;
            };
            ExpectCooksLike(f, mesh, source_id, source, reference(task->Param0));
            if (task->Param0 == 0.f) return;
            task->Param0 *= 2.f;
            node->Set(f, entity, *task);
            ExpectCooksLike(f, mesh, source_id, source, reference(task->Param0));
        });
    }
    // Generator kinds follow PrimitiveShape's alternatives.
    for (const PrimitiveShape shape : {PrimitiveShape{primitive::Plane{}}, PrimitiveShape{primitive::Circle{}}, PrimitiveShape{primitive::Cuboid{}}, PrimitiveShape{primitive::IcoSphere{}}, PrimitiveShape{primitive::UVSphere{}}, PrimitiveShape{primitive::Torus{}}, PrimitiveShape{primitive::Cylinder{}}, PrimitiveShape{primitive::Cone{}}}) {
        Scenario(f, other, ToString(shape), [shape](ProcedureFixture &f) {
            auto &meshes = f.Store();
            const auto mesh = f.AddMesh(primitive::Plane{});
            const auto source_id = f.Source(mesh);
            const auto source = Describe(meshes, source_id);
            f.Do(AddProcedure{mesh});
            const auto generator = f.Append(mesh, uint16_t(KindOf<primitive::Plane> + shape.index())).first;
            const auto subdivide = f.Append(mesh, KindOf<action::mesh::Subdivide>).first;
            f.Route(mesh, {generator, subdivide, 1u});
            const auto generated = CreateMesh(f.R, {.Data = primitive::CreateMesh(shape), .FlatShaded = true}).StoreId;
            const auto reference = SelectAllReference(f, generated, {Op(MeshTopologyOp::Subdivide, 1.f)});
            meshes.Release(generated);
            f.Commit("Release generated");
            ExpectCooksLike(f, mesh, source_id, source, reference);
        });
    }
    Scenario(f, other, "FanOut", [](ProcedureFixture &f) {
        const auto mesh = f.AddMesh(primitive::Plane{});
        const auto source_id = f.Source(mesh);
        const auto source = Describe(f.Store(), source_id);
        f.Do(AddProcedure{mesh});
        const auto [first, first_entity] = f.Append(mesh, KindOf<action::mesh::Extrude>);
        const auto [second, second_entity] = f.Append(mesh, KindOf<action::mesh::Extrude>);
        f.Do(Link{mesh, 0u, 0u, first, 0u});
        f.Do(Link{mesh, 0u, 0u, second, 0u});
        const auto reference = SelectAllReference(f, source_id, {Op(MeshTopologyOp::ExtrudeRegion)});
        for (const auto read : {first, second}) {
            f.Do(Link{mesh, read, 2u, 1u, 0u});
            expect(!f.R.any_of<NodeError>(first_entity) && !f.R.any_of<NodeError>(second_entity));
            ExpectCooksLike(f, mesh, source_id, source, reference);
        }
    });
}

// A graph on `Mesh`, an oracle for its output, and an edit that changes a copy's output.
struct Graph {
    state::Entity Mesh;
    std::function<void(ProcedureFixture &)> Oracle;
    std::function<void(ProcedureFixture &, state::Entity)> Edit{};
};
struct GraphRow {
    std::string_view Name;
    Graph (*Build)(ProcedureFixture &);
};

// Each graph's output matches its oracle, a duplicate cooks the same and edits apart, and deleting and applying the procedure leave its output.
void Survives(ProcedureFixture &f, Fixture &other, std::span<const GraphRow> rows) {
    for (const auto &row : rows) {
        Scenario(f, other, std::string{row.Name}, [&row](ProcedureFixture &f) {
            auto &r = f.R;
            const auto graph = row.Build(f);
            const auto mesh = graph.Mesh;
            graph.Oracle(f);
            if (!graph.Edit) return;
            const auto output = f.Shaped(mesh);
            f.Do(action::selection::Select{f.Object(mesh)});
            f.Do(action::object::Duplicate{});
            const auto copy = r.get<const Instance>(FindActiveEntity(r)).Entity;
            expect(copy != mesh && f.Output(copy) != f.Output(mesh) && f.Shaped(copy) == output);
            graph.Edit(f, copy);
            expect(f.Shaped(copy) != output && f.Shaped(mesh) == output);
            f.Undo();
            f.Undo();
            f.Do(action::object::Delete{});
            expect(!r.valid(mesh) && r.view<const GeometryProcedure>().empty());
            f.Undo();
            expect(f.Shaped(mesh) == output);
            f.Do(ApplyProcedure{mesh});
            expect(!r.any_of<GeometryProcedure, EvaluatedMesh, ResyncState, CookPlan>(mesh));
            expect(Describe(f.Store(), f.Source(mesh)) == output);
        });
    }
}

Graph ExtrudeSubdivide(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Plane{});
    f.Do(AddProcedure{mesh});
    const auto extrude = f.Append(mesh, KindOf<action::mesh::Extrude>).first;
    const auto [subdivide, subdivide_entity] = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Route(mesh, {0u, extrude, subdivide, 1u});
    const auto once = f.Shaped(mesh);
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide_entity, 2u));
    expect(f.Shaped(mesh) != once);
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto output = f.Shaped(mesh);
            expect(output != Describe(f.Store(), f.Source(mesh)));
            expect(output == SelectAllReference(f, f.Source(mesh), {Op(MeshTopologyOp::ExtrudeRegion), Op(MeshTopologyOp::Subdivide, 2.f)}));
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 3u), 3u)); },
    };
}

// A sphere's cap above y = 0.9 insets through Position, Compare and Inset, and follows a moved source vertex.
Graph CapResync(ProcedureFixture &f) {
    auto &r = f.R;
    const auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::UVSphere{.Slices = 128u, .Stacks = 64u});
    f.Do(AddProcedure{mesh});
    const auto position = f.Append(mesh, KindOf<Position>).first;
    const auto [compare, compare_entity] = f.Append(mesh, KindOf<Compare>);
    const auto [inset, inset_entity] = f.Append(mesh, KindOf<action::mesh::Inset>);
    f.Do(UpdateOf<&Compare::Component>(compare_entity, VectorComponent::Y));
    f.Do(UpdateOf<&Compare::B>(compare_entity, .9f));
    f.Do(Link{mesh, position, 0u, compare, 0u});
    f.Do(Link{mesh, compare, f.Out(mesh, compare), inset, 1u});
    f.Route(mesh, {0u, inset, 1u});
    const auto first = f.Shaped(mesh);
    const auto set = [&](float thickness) {
        f.Do(UpdateOf<&action::mesh::Inset::Thickness>(inset_entity, thickness));
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
    const auto source = f.Source(mesh);
    const PositionOperationTarget target{.StoreId = source, .Selection = GeometrySelection{.Vertices = {**Mesh{meshes, source}.vertices().begin()}}};
    for (const auto offset : {vec3{0.f, -.5f, 0.f}, vec3{.3f, 0.f, .1f}}) {
        ExecuteGeometryPositions(r, std::span{&target, 1u}, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = offset}}});
        r.emplace_or_replace<MeshPositionsChanged>(mesh);
        f.Commit("Move source vertex");
    }
    return {
        mesh,
        // Compare output against the kernel insetting the moved source cap on a clone.
        [mesh](ProcedureFixture &f) {
            const auto kernel = Reference(f, f.Source(mesh), [&](uint32_t clone) {
                const Mesh sphere{f.Store(), clone};
                const auto above = [&](he::VH v) { return sphere.GetPosition(v).y > .9f; };
                GeometrySelection cap;
                for (const auto v : sphere.vertices())
                    if (above(v)) cap.Vertices.push_back(*v);
                for (const auto e : sphere.edges()) {
                    const auto h = sphere.GetHalfedge(e, 0u);
                    if (above(sphere.GetFromVertex(h)) && above(sphere.GetToVertex(h))) cap.Edges.push_back(*e);
                }
                for (const auto face : sphere.faces())
                    if (std::ranges::all_of(sphere.fv_range(face), above)) cap.Faces.push_back(*face);
                f.Run(clone, {{.Op = MeshTopologyOp::InsetRegion, .Param0 = .06f, .Flags = TopologyFlagEvenOffset, .Selection = cap}});
                return GeometrySelection{};
            });
            expect(f.Shaped(mesh) == kernel.Output);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Inset::Thickness>(f.Node(mesh, 4u), .2f)); },
    };
}

// Vector Math feeding Set Position doubles vertex positions.
Graph ScalePositions(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto position = f.Append(mesh, KindOf<Position>).first;
    const auto [math, math_entity] = f.Append(mesh, KindOf<VectorMath>);
    const auto [set, set_entity] = f.Append(mesh, KindOf<SetPosition>);
    f.Do(UpdateOf<&VectorMath::Operation>(math_entity, VectorMathOp::Scale));
    f.Do(UpdateOf<&VectorMath::Scale>(math_entity, 2.f));
    f.Do(UpdateOf<&SetPosition::Offset>(set_entity, false));
    f.Do(Link{mesh, position, 0u, math, 0u});
    f.Do(Link{mesh, math, f.Out(mesh, math), set, 1u});
    f.Route(mesh, {0u, set, 1u});
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            expect(Near(Positions(meshes, f.Output(mesh)), Points(meshes, f.Source(mesh), [](uint32_t, vec3 p) { return 2.f * p; })));
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&VectorMath::Scale>(f.Node(mesh, 3u), 3.f)); },
    };
}

// Position.z compared greater than zero selects the cube's upper face for Extrude.
Graph CompareSelectsExtrusion(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto position = f.Append(mesh, KindOf<Position>).first;
    const auto [compare, compare_entity] = f.Append(mesh, KindOf<Compare>);
    const auto extrude = f.Append(mesh, KindOf<action::mesh::Extrude>).first;
    f.Do(UpdateOf<&Compare::Component>(compare_entity, VectorComponent::Z));
    f.Do(Link{mesh, position, 0u, compare, 0u});
    f.Do(Link{mesh, compare, f.Out(mesh, compare), extrude, 1u});
    f.Route(mesh, {0u, extrude, 1u});
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            const auto kernel = Reference(f, f.Source(mesh), [&](uint32_t clone) {
                const Mesh cube{meshes, clone};
                const auto above = [&](he::VH v) { return cube.GetPosition(v).z > 0.f; };
                GeometrySelection upper;
                for (const auto v : cube.vertices())
                    if (above(v)) upper.Vertices.push_back(*v);
                for (const auto e : cube.edges()) {
                    const auto h = cube.GetHalfedge(e, 0u);
                    if (above(cube.GetFromVertex(h)) && above(cube.GetToVertex(h))) upper.Edges.push_back(*e);
                }
                for (const auto face : cube.faces())
                    if (std::ranges::all_of(cube.fv_range(face), above)) upper.Faces.push_back(*face);
                expect(upper.Faces.size() == 1u);
                f.Run(clone, {{.Op = MeshTopologyOp::ExtrudeRegion, .Selection = upper}});
                return GeometrySelection{};
            });
            expect(f.Shaped(mesh) == kernel.Output);
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&Compare::Component>(f.Node(mesh, 3u), VectorComponent::X)); },
    };
}

vec3 OffsetOf(uint32_t v) { return {float(v % 7u), 1.f, -float(v % 3u)}; }
// A named vertex attribute offsets each vertex through Set Position.
Graph NamedAttributeReads(ProcedureFixture &f) {
    auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto source = f.Source(mesh);
    const auto offset = meshes.AddAttribute(source, AttributeDomain::Vertex, AttributeType::Vec3, "offset").value();
    for (const auto v : Mesh{meshes, source}.vertices()) {
        const auto value = OffsetOf(*v);
        meshes.SetAttribute<vec3>(offset, *v, std::span{&value, 1u});
    }
    f.Commit("Add offset");
    f.Do(AddProcedure{mesh});
    const auto [named, named_entity] = f.Append(mesh, KindOf<procedure::NamedAttribute>);
    const auto set = f.Append(mesh, KindOf<SetPosition>).first;
    f.Do(SetAttributeName{named_entity, "offset"});
    f.Do(UpdateOf<&procedure::NamedAttribute::Type>(named_entity, AttributeType::Vec3));
    f.Do(Link{mesh, named, 0u, set, 1u});
    f.Route(mesh, {0u, set, 1u});
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            expect(Near(Positions(meshes, f.Output(mesh)), Points(meshes, f.Source(mesh), [](uint32_t v, vec3 p) { return p + OffsetOf(v); })));
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&SetPosition::Offset>(f.Node(mesh, 3u), false)); },
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
    f.Do(AddProcedure{mesh});
    const auto position = f.Append(mesh, KindOf<Position>).first;
    const auto [store, store_entity] = f.Append(mesh, KindOf<StoreNamedAttribute>);
    const auto [subdivide, subdivide_entity] = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    f.Do(SetAttributeName{store_entity, "rest"});
    f.Do(Link{mesh, position, 0u, store, 1u});
    f.Route(mesh, {0u, store, subdivide, 1u});
    expect(RestMatches(f, mesh));
    const auto faces = f.Shaped(mesh).Faces.size();
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide_entity, 2u));
    expect(f.Shaped(mesh).Faces.size() > faces);
    return {
        mesh,
        [mesh](ProcedureFixture &f) { expect(RestMatches(f, mesh)); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 4u), 3u)); },
    };
}

// Vector Math on a vertex and a face attribute errs and forwards A, so Set Position offsets each vertex by its position.
Graph DomainMismatch(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto position = f.Append(mesh, KindOf<Position>).first;
    const auto [normal, normal_entity] = f.Append(mesh, KindOf<Normal>);
    const auto [math, math_entity] = f.Append(mesh, KindOf<VectorMath>);
    const auto [set, set_entity] = f.Append(mesh, KindOf<SetPosition>);
    f.Do(UpdateOf<&Normal::Domain>(normal_entity, AttributeDomain::Face));
    f.Do(Link{mesh, position, 0u, math, 0u});
    f.Do(Link{mesh, normal, 0u, math, 1u});
    f.Do(Link{mesh, math, f.Out(mesh, math), set, 1u});
    f.Route(mesh, {0u, set, 1u});
    return {
        mesh,
        [mesh, math_entity](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            expect(f.R.any_of<NodeError>(math_entity));
            expect(Near(Positions(meshes, f.Output(mesh)), Points(meshes, f.Source(mesh), [](uint32_t, vec3 p) { return 2.f * p; })));
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&Normal::Domain>(f.Node(mesh, 3u), AttributeDomain::Vertex)); },
    };
}

// A join of the source and a generated cube doubles the elements, and the generated vertices take the source attribute's default.
Graph JoinGenerated(ProcedureFixture &f) {
    auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    const auto source = f.Source(mesh);
    const auto weight = meshes.AddAttribute(source, AttributeDomain::Vertex, AttributeType::Float, "weight", AttributeValue{std::bit_cast<uint32_t>(-1.f), 0u, 0u, 0u}).value();
    for (const auto v : Mesh{meshes, source}.vertices()) {
        const float value = 5.f;
        meshes.SetAttribute<float>(weight, *v, std::span{&value, 1u});
    }
    f.Commit("Add weight");
    f.Do(AddProcedure{mesh});
    const auto [cube, cube_entity] = f.Append(mesh, KindOf<primitive::Cuboid>);
    const auto join = f.Append(mesh, KindOf<Join>).first;
    f.Do(Link{mesh, 0u, 0u, join, 0u});
    f.Do(Link{mesh, cube, f.Out(mesh, cube), join, 1u});
    f.Do(Link{mesh, join, 2u, 1u, 0u});
    const auto joined = f.Shaped(mesh);
    f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(cube_entity, vec3{2.f, .5f, 1.f}));
    expect(f.Shaped(mesh) != joined);
    return {
        mesh,
        [mesh](ProcedureFixture &f) {
            const auto &meshes = f.Store();
            const auto id = f.Output(mesh), source = f.Source(mesh);
            const Mesh joined{meshes, id}, single{meshes, source};
            expect(joined.VertexCount() == 2u * single.VertexCount() && joined.FaceCount() == 2u * single.FaceCount());
            const auto view = meshes.AttributeView<float>(meshes.FindAttribute(id, "weight"));
            std::array<uint32_t, 2> weights{};
            for (const auto v : joined.vertices()) {
                if (view[*v] == 5.f) ++weights[0];
                else if (view[*v] == -1.f) ++weights[1];
            }
            expect(weights[0] == single.VertexCount() && weights[1] == single.VertexCount());
        },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(f.Node(mesh, 2u), vec3{.5f, .5f, .5f})); },
    };
}

std::map<uint32_t, float> Heat(const MeshStore &meshes, uint32_t id) {
    std::map<uint32_t, float> heat;
    const auto view = meshes.AttributeView<float>(meshes.FindAttribute(id, "heat"));
    meshes.Arenas().Vertices.ForEach(meshes.Get(id).Vertices, [&](ElementHandle v, ElementAddress) { heat.emplace(v.Value, view[v.Value]); });
    return heat;
}
// A named attribute's values travel through an edit-mode subdivision of a plain cube.
// Retained vertices keep their heat, and an edge midpoint takes the mean of two corners and a face center the mean of four.
Graph HeatSubdivide(ProcedureFixture &f) {
    auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    const auto id = f.Source(mesh);
    const auto heat = meshes.AddAttribute(id, AttributeDomain::Vertex, AttributeType::Float, "heat").value();
    const auto vertices = meshes.Arenas().Vertices.Dense(meshes.Get(id).Vertices);
    std::vector<float> values(vertices.Count);
    for (uint32_t i = 0u; i < vertices.Count; ++i) values[i] = float(i * i);
    meshes.SetAttribute<float>(heat, 0u, values);
    f.Commit("Heat cube");
    const auto original = Heat(meshes, id);
    f.EnterEdit(Element::Vertex);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{1u});
    return {mesh, [id, original](ProcedureFixture &f) {
                const auto split = Heat(f.Store(), id);
                expect(split.size() > original.size());
                for (const auto [v, value] : original) expect(split.at(v) == value);
                std::vector<float> corners;
                for (const auto [v, value] : original) corners.push_back(value);
                std::set<float> means;
                for (size_t a = 0u; a < corners.size(); ++a)
                    for (size_t b = a + 1u; b < corners.size(); ++b) {
                        means.insert(.5f * (corners[a] + corners[b]));
                        for (size_t c = b + 1u; c < corners.size(); ++c)
                            for (size_t d = c + 1u; d < corners.size(); ++d) means.insert(.25f * (corners[a] + corners[b] + corners[c] + corners[d]));
                    }
                for (const auto [v, value] : split)
                    if (!original.contains(v)) expect(means.contains(value)) << "vertex" << v << "heat" << value;
            }};
}

// A boolean intersection of the cube with a generated bar keeps the bar's middle.
Graph BooleanIntersection(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto [box, box_entity] = f.Append(mesh, KindOf<primitive::Cuboid>);
    f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(box_entity, vec3{2.f, .5f, .5f}));
    const auto [boolean, boolean_entity] = f.Append(mesh, KindOf<procedure::Boolean>);
    f.Do(UpdateOf<&procedure::Boolean::Operation>(boolean_entity, BooleanOperation::Intersection));
    f.Route(mesh, {0u, boolean, 1u});
    // An unlinked operand passes the cube through.
    expect(f.Shaped(mesh) == Describe(f.Store(), f.Source(mesh)));
    f.Do(Link{mesh, box, f.Out(mesh, box), boolean, 1u});
    expect(std::abs(test::EnclosedVolume(Mesh{f.Store(), f.Output(mesh)}).value_or(-1.0) - 2.0) < 1e-4);
    f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(box_entity, vec3{2.f, .25f, .75f}));
    return {
        mesh,
        [mesh](ProcedureFixture &f) { expect(std::abs(test::EnclosedVolume(Mesh{f.Store(), f.Output(mesh)}).value_or(-1.0) - 1.5) < 1e-4); },
        [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&procedure::Boolean::Operation>(f.Node(mesh, 3u), BooleanOperation::Union)); },
    };
}

// The boolean action makes the active object's mesh the active cube minus the other selected cube, in the active object's space.
Graph BooleanAction(ProcedureFixture &f) {
    auto &r = f.R;
    const auto target = f.AddMesh(primitive::Cuboid{});
    const auto active = FindActiveEntity(r);
    f.Do(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = {.P = {1.f, 1.f, 1.f}}, .Select = MeshInstanceCreateInfo::SelectBehavior::Additive})});
    expect(FindActiveEntity(r) == active);
    f.Do(action::mesh::Boolean{BooleanOperation::Difference});
    return {target, [target, active](ProcedureFixture &f) {
                expect(std::abs(test::EnclosedVolume(Mesh{f.Store(), f.Source(target)}).value_or(-1.0) - 7.0) < 1e-4);
                for (const auto e : f.R.view<const Instance>())
                    if (e != active) expect(Mesh{f.Store(), f.Source(f.R.get<const Instance>(e).Entity)}.VertexCount() == 8u);
            }};
}

// Primitive shape edits regenerate the record.
Graph PrimitiveShapeEdit(ProcedureFixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(action::UpdateOfShape(mesh, &primitive::Cuboid::HalfExtents, vec3{2.f, .5f, 1.f}));
    f.Do(action::UpdateOfShape(mesh, &primitive::Cuboid::HalfExtents, vec3{1.f, 1.f, 3.f}));
    return {mesh, [mesh](ProcedureFixture &f) { expect(Near(Positions(f.Store(), f.Output(mesh)), ShapePositions(primitive::Cuboid{{1.f, 1.f, 3.f}}))); }};
}

// An Object Info node reading its own procedure's mesh errs and forwards the source.
// One reading a mesh with a procedure of its own cooks after that procedure, so a resize there reaches the join.
Graph ObjectInfo(ProcedureFixture &f) {
    auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Plane{});
    f.Do(AddProcedure{mesh});
    const auto [info, info_entity] = f.Append(mesh, KindOf<procedure::ObjectInfo>);
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(info_entity, f.Object(mesh)));
    const auto [join, join_entity] = f.Append(mesh, KindOf<Join>);
    f.Do(Link{mesh, 0u, 0u, join, 0u});
    f.Do(Link{mesh, info, 0u, join, 1u});
    f.Do(Link{mesh, join, 2u, 1u, 0u});
    expect(r.all_of<NodeError>(info_entity) && !r.any_of<NodeError>(join_entity));
    expect(f.Shaped(mesh) == Describe(f.Store(), f.Source(mesh)));
    const auto cube = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{cube});
    const auto [box, box_entity] = f.Append(cube, KindOf<primitive::Cuboid>);
    f.Do(Link{cube, box, f.Out(cube, box), 1u, 0u});
    f.Do(UpdateOf<&procedure::ObjectInfo::Object>(info_entity, f.Object(cube)));
    expect(!r.any_of<NodeError>(info_entity));
    f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(box_entity, vec3{.5f, .25f, .5f}));
    return {mesh, [mesh](ProcedureFixture &f) {
                auto points = ShapePositions(primitive::Plane{});
                points.append_range(ShapePositions(primitive::Cuboid{{.5f, .25f, .5f}}));
                std::ranges::sort(points);
                expect(Mesh{f.Store(), f.Output(mesh)}.FaceCount() == 7u && Near(Positions(f.Store(), f.Output(mesh)), points));
            }};
}

// Instances on the plane's points publish as entities or realize as geometry, following scale, plane and cube edits.
template<bool Borrowed, bool Realize> Graph Instances(ProcedureFixture &f) {
    auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto object = FindActiveEntity(r);
    const auto cube_mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto to_points = f.Append(mesh, KindOf<MeshToPoints>).first;
    const auto [instancer, instancer_entity] = f.Append(mesh, KindOf<InstanceOnPoints>);
    const auto [instance, instance_entity] = f.Append(mesh, Borrowed ? KindOf<procedure::ObjectInfo> : KindOf<primitive::Cuboid>);
    if constexpr (Borrowed) f.Do(UpdateOf<&procedure::ObjectInfo::Object>(instance_entity, f.Object(cube_mesh)));
    f.Route(mesh, {0u, to_points, instancer});
    f.Do(Link{mesh, instance, f.Out(mesh, instance), instancer, 1u});
    if constexpr (Realize) {
        const auto realizer = f.Append(mesh, KindOf<RealizeInstances>).first;
        f.Route(mesh, {instancer, realizer, 1u});
    } else {
        f.Do(Link{mesh, instancer, f.Out(mesh, instancer), 1u, 0u});
    }
    f.Do(UpdateOf<&InstanceOnPoints::Scale>(instancer_entity, vec3{.5f, .5f, .5f}));
    f.Do(action::UpdateOfShape(mesh, &primitive::Plane::HalfExtents, vec2{2.f, 3.f}));
    if constexpr (Borrowed) f.Do(action::UpdateOfShape(cube_mesh, &primitive::Cuboid::HalfExtents, vec3{.5f, .25f, .5f}));
    else f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(instance_entity, vec3{.5f, .25f, .5f}));
    Graph graph{mesh, [=](ProcedureFixture &f) {
                    const auto &r = f.R;
                    const auto &meshes = f.Store();
                    const auto plane = ShapePositions(primitive::Plane{{2.f, 3.f}}), cube = ShapePositions(primitive::Cuboid{{.5f, .25f, .5f}});
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
                            expect(Near(Positions(meshes, f.Source(r.get<const Instance>(e).Entity)), cube));
                        }
                        std::ranges::sort(points);
                        expect(Near(points, plane));
                    }
                }};
    if constexpr (Realize) graph.Edit = [](ProcedureFixture &f, state::Entity mesh) { f.Do(UpdateOf<&InstanceOnPoints::Scale>(f.Node(mesh, 3u), vec3{.25f, .25f, .25f})); };
    return graph;
}

// Selecting, duplicating, inserting and deleting nodes leaves the output the chain they form.
Graph NodeEditing(ProcedureFixture &f) {
    auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto [subdivide, subdivide_entity] = f.Append(mesh, KindOf<action::mesh::Subdivide>);
    const auto [triangulate, triangulate_entity] = f.Append(mesh, KindOf<action::mesh::Triangulate>);
    const auto position = f.Append(mesh, KindOf<Position>).first;
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide_entity, 3u));
    const auto source = Describe(f.Store(), f.Source(mesh));
    const auto subdivided = SelectAllReference(f, f.Source(mesh), {Op(MeshTopologyOp::Subdivide, 3.f)});
    const auto triangulated = SelectAllReference(f, f.Source(mesh), {Op(MeshTopologyOp::Triangulate)});
    const auto both = SelectAllReference(f, f.Source(mesh), {Op(MeshTopologyOp::Subdivide, 3.f), Op(MeshTopologyOp::Triangulate)});
    // Duplicates keep their parameters and only the links among them.
    f.Do(SelectNodes{mesh, {subdivide, triangulate}, subdivide});
    f.Do(DuplicateNodes{mesh, {subdivide, triangulate}});
    const auto &procedure = r.get<const GeometryProcedure>(mesh);
    expect(procedure.Nodes.size() == 7u && r.get<const action::mesh::Subdivide>(procedure.Nodes[5].Entity).Cuts == 3u);
    expect(f.Shaped(mesh) == source);
    // Inserting a node into the output link cooks through it, and a node without a geometry input inserts nothing.
    f.Fails(InsertNode{mesh, position, 1u, 0u, 0u});
    f.Do(SelectNodes{mesh, {subdivide}, subdivide});
    f.Stage(InsertNode{mesh, subdivide, 1u, 0u, 40u});
    f.CommitGesture();
    expect(f.Shaped(mesh) == subdivided);
    f.Do(InsertNode{mesh, triangulate, 1u, 0u, 0u});
    expect(f.Shaped(mesh) == both);
    // Deleting a node heals the chain through it.
    f.Do(DeleteNodes{mesh, {subdivide}});
    expect(f.Shaped(mesh) == triangulated);
    f.Undo();
    f.Do(DeleteNodes{mesh, {subdivide, triangulate}});
    expect(f.Shaped(mesh) == source);
    f.Undo();
    return {mesh, [mesh, both](ProcedureFixture &f) { expect(f.Shaped(mesh) == both); }};
}

constexpr std::array<GraphRow, 16> Graphs{{
    {"ExtrudeSubdivide", ExtrudeSubdivide},
    {"CapResync", CapResync},
    {"ScalePositions", ScalePositions},
    {"CompareSelectsExtrusion", CompareSelectsExtrusion},
    {"NamedAttributeReads", NamedAttributeReads},
    {"StorePromotes", StorePromotes},
    {"DomainMismatch", DomainMismatch},
    {"JoinGenerated", JoinGenerated},
    {"HeatSubdivide", HeatSubdivide},
    {"BooleanIntersection", BooleanIntersection},
    {"BooleanAction", BooleanAction},
    {"PrimitiveShapeEdit", PrimitiveShapeEdit},
    {"ObjectInfo", ObjectInfo},
    {"InstancesBorrowedPublish", Instances<true, false>},
    {"InstancesGeneratedRealize", Instances<false, true>},
    {"NodeEditing", NodeEditing},
}};

struct ViewportRow {
    std::string_view Name;
    Element Domain;
    uint32_t Clicks;
    uint16_t Kind;
    std::vector<float> Values;
    void (*Gesture)(ProcedureFixture &, float value, bool first);
    // Run the operator on the clone at the clicked handles and return its selected elements.
    GeometrySelection (*Kernel)(ProcedureFixture &, uint32_t clone, const GeometrySelection &clicked, float value);
};
// Run the topology task on the clone, publish its emitted selection, and return that selection.
GeometrySelection PublishedKernel(ProcedureFixture &f, uint32_t clone, MeshTopologyTask task) {
    task.SourceId = clone;
    PublishStoreSelection(f.Store(), task);
    f.Run(clone, {task});
    return Selection(f.Store(), clone);
}

// Each viewport edit appends one node cooking like the kernel at the clicked elements, and a drag commits as one history node.
void ViewportOperators(ProcedureFixture &f, Fixture &other) {
    using D = MeshStore::ElementDomain;
    static const std::array<ViewportRow, 3> rows{{
        {"Extrude", Element::Face, 2u, KindOf<action::mesh::Extrude>, {0.f}, [](ProcedureFixture &f, float, bool) { f.Do(action::mesh::Extrude{}); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float) { return PublishedKernel(f, clone, {.Op = MeshTopologyOp::ExtrudeRegion, .Selection = clicked}); }},
        {"BevelDrag", Element::Edge, 1u, KindOf<action::mesh::Bevel>, {.05f, .1f, .2f, .3f}, [](ProcedureFixture &f, float width, bool first) {
             if (first) f.Stage(action::mesh::Bevel{.Width = width});
             else f.Stage(UpdateOf<&action::mesh::Bevel::Width>(action::OnSelected{}, width)); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float width) { return PublishedKernel(f, clone, {.Op = MeshTopologyOp::BevelEdges, .Param0 = width, .Steps = 1u, .Selection = clicked}); }},
        {"TransformDrag", Element::Vertex, 1u, KindOf<TransformElements>, {.5f, 1.f, 1.5f}, [](ProcedureFixture &f, float z, bool) { f.Stage(action::view::TransformElements{{.P = {0.f, 0.f, z}}}); }, [](ProcedureFixture &f, uint32_t clone, const GeometrySelection &clicked, float z) {
             const PositionOperationTarget target{.StoreId = clone, .Selection = clicked};
             expect(!ExecuteGeometryPositions(f.R, std::span{&target, 1u}, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = {0.f, 0.f, z}}}}).empty());
             return clicked; }},
    }};
    for (const auto &row : rows) {
        Scenario(f, other, std::string{row.Name}, [&row](ProcedureFixture &f) {
            auto &r = f.R;
            const auto &meshes = f.Store();
            const auto domains = [](GeometrySelection &selection) {
                return std::array{std::tuple{D::Vertex, Element::Vertex, &selection.Vertices}, std::tuple{D::Edge, Element::Edge, &selection.Edges}, std::tuple{D::Face, Element::Face, &selection.Faces}};
            };
            const auto mesh = f.AddMesh(primitive::Cuboid{});
            f.Do(AddProcedure{mesh});
            f.EnterEdit(row.Domain);
            f.Render();
            {
                // The row clicks the centers of the first elements of its domain whose faces all look toward the eye.
                const Mesh cube{meshes, f.Output(mesh)};
                const auto eye = f.View().CameraPosition;
                const auto facing = [&](he::FH face) { return Dot(cube.GetNormal(face), eye - cube.CalcFaceCentroid(face)) > 0.f; };
                std::vector<vec3> centers;
                if (row.Domain == Element::Vertex) {
                    for (const auto v : cube.vertices())
                        if (std::ranges::all_of(cube.voh_range(v), [&](he::HH h) { return facing(cube.GetFace(h)); })) centers.push_back(cube.GetPosition(v));
                } else if (row.Domain == Element::Edge) {
                    for (const auto e : cube.edges()) {
                        const auto h = cube.GetHalfedge(e, 0u);
                        if (facing(cube.GetFace(h)) && facing(cube.GetFace(cube.GetOppositeHalfedge(h)))) centers.push_back(.5f * (cube.GetPosition(cube.GetFromVertex(h)) + cube.GetPosition(cube.GetToVertex(h))));
                    }
                } else {
                    for (const auto face : cube.faces())
                        if (facing(face)) centers.push_back(cube.CalcFaceCentroid(face));
                }
                expect(fatal(centers.size() >= row.Clicks));
                for (uint32_t i = 0u; i < row.Clicks; ++i) f.Click(centers[i], i > 0u);
            }
            GeometrySelection ordinals;
            for (const auto [domain, element, list] : domains(ordinals))
                meshes.GetSelectedElements(f.Output(mesh), element).ForEach([&](uint32_t handle) { list->push_back(meshes.LiveElementOrdinal(f.Output(mesh), domain, ElementHandle{handle})); });
            expect(ordinals.Get(row.Domain).size() == row.Clicks);
            const auto source = Describe(meshes, f.Source(mesh));
            const auto start = f.P->History.Present.value();
            const auto node_count = uint32_t(r.get<const GeometryProcedure>(mesh).Nodes.size());
            auto previous = f.Shaped(mesh);
            for (uint32_t i = 0u; i < row.Values.size(); ++i) {
                row.Gesture(f, row.Values[i], i == 0u);
                expect(r.get<const GeometryProcedure>(mesh).Nodes.size() == node_count + 1u);
                const auto shape = f.Shaped(mesh);
                expect(shape != previous);
                previous = shape;
            }
            if (f.P->HasStaged()) f.CommitGesture();
            const auto &procedure = r.get<const GeometryProcedure>(mesh);
            const auto [node, kind] = procedure.Nodes.back();
            expect(procedure.Nodes.size() == node_count + 1u && kind == row.Kind);
            expect(r.all_of<PublishSelection>(node) && !r.any_of<NodeError>(node));
            expect(f.P->History.Nodes[f.P->History.Present.value()].Parent == start);
            const auto selection = Selection(meshes, f.Output(mesh));
            const auto selected = Describe(meshes, f.Output(mesh), selection);
            expect(meshes.GetSelectionSummary(f.Output(mesh)).SelectedCount == selection.Get(row.Domain).size());
            f.Do(action::view::SetInteractionMode{InteractionMode::Object});
            expect(f.Shaped(mesh) == previous);
            expect(Describe(meshes, f.Source(mesh)) == source);
            const auto kernel = Reference(f, f.Source(mesh), [&](uint32_t clone) {
                GeometrySelection clicked;
                for (const auto [domain, element, list] : domains(clicked)) {
                    for (const auto ordinal : ordinals.Get(element)) list->push_back(meshes.LiveElementAt(clone, domain, ordinal).Value);
                    std::ranges::sort(*list);
                }
                return row.Kernel(f, clone, clicked, row.Values.back());
            });
            expect(kernel.Output == previous);
            expect(kernel.Selected == selected);
        });
    }
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
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto source_id = f.Source(mesh);
    const auto source = Describe(meshes, source_id);
    const auto output = [&] { return f.Shaped(mesh); };
    const auto [real, real_entity] = f.Append(mesh, KindOf<FloatValue>);
    {
        const auto [bevel, bevel_entity] = f.Append(mesh, KindOf<Bevel>);
        f.Route(mesh, {0u, bevel, 1u});
        const auto one = output();
        f.Do(UpdateOf<&Bevel::Segments>(bevel_entity, 3u));
        const auto three = output();
        expect(three != one);
        f.Do(UpdateOf<&Bevel::Segments>(bevel_entity, 1u));
        const auto [integer, integer_entity] = f.Append(mesh, KindOf<IntegerValue>);
        f.Do(UpdateOf<&IntegerValue::Value>(integer_entity, 3u));
        const auto segments = SocketOf<&Bevel::Segments>(), width = SocketOf<&Bevel::Width>();
        f.Do(Link{mesh, integer, 0u, bevel, segments});
        expect(output() == three);
        f.Do(UpdateOf<&Bevel::Segments>(bevel_entity, 5u));
        expect(output() == three);
        f.Do(UpdateOf<&IntegerValue::Value>(integer_entity, 1u));
        expect(output() == one);
        f.Do(action::nodes::Unlink{mesh, bevel, segments});
        f.Do(UpdateOf<&Bevel::Segments>(bevel_entity, 3u));
        expect(output() == three);
        const auto position = f.Append(mesh, KindOf<Position>).first;
        f.Fails(Link{mesh, real, 0u, bevel, segments});
        f.Fails(Link{mesh, position, 0u, bevel, segments});
        f.Do(UpdateOf<&FloatValue::Value>(real_entity, .2f));
        f.Do(Link{mesh, real, 0u, bevel, width});
        f.Do(UpdateOf<&Bevel::Width>(bevel_entity, .3f));
        expect(output() == SelectAllReference(f, source_id, {Op(MeshTopologyOp::BevelEdges, .2f, 0u, 3u)}));
        f.Do(action::nodes::Unlink{mesh, bevel, width});
        f.Do(UpdateOf<&Bevel::Width>(bevel_entity, .1f));
        const auto [flag, flag_entity] = f.Append(mesh, KindOf<BooleanValue>);
        f.Do(Link{mesh, flag, 0u, bevel, 1u});
        expect(output() == source);
        f.Do(UpdateOf<&BooleanValue::Value>(flag_entity, true));
        expect(output() == three);
        f.Do(DeleteNodes{mesh, {bevel, position}});
    }
    const auto position = f.Append(mesh, KindOf<Position>).first;
    const auto [dot, dot_entity] = f.Append(mesh, KindOf<VectorMath>);
    const auto [add, add_entity] = f.Append(mesh, KindOf<FloatMath>);
    f.Do(UpdateOf<&VectorMath::Operation>(dot_entity, VectorMathOp::Dot));
    f.Do(UpdateOf<&VectorMath::B>(dot_entity, vec3{.05f, .05f, .05f}));
    f.Do(UpdateOf<&FloatMath::B>(add_entity, .2f));
    f.Do(Link{mesh, position, 0u, dot, 0u});
    f.Do(Link{mesh, dot, f.Out(mesh, dot), add, 0u});
    // Run the operator on a source clone with all elements selected and Param0 read from the added parameter attribute.
    const auto reference = [&](MeshTopologyOp op, auto &&parameter) {
        return Reference(f, source_id, [&](uint32_t clone) {
                   auto task = Op(op, 0.f, TopologyFlagSelectAll);
                   task.Param0Attribute = parameter(clone);
                   f.Run(clone, {task});
                   return GeometrySelection{};
               })
            .Output;
    };
    const auto widths = [&](uint32_t clone) { return FloatAttribute<AttributeDomain::Vertex>(meshes, clone, WidthAt); };
    const auto add_operator = [&](uint16_t kind, uint16_t socket) {
        const auto [node, entity] = f.Append(mesh, kind);
        f.Do(Link{mesh, add, f.Out(mesh, add), node, socket});
        f.Route(mesh, {0u, node, 1u});
        return std::pair{node, entity};
    };
    {
        const auto [bevel, bevel_entity] = add_operator(KindOf<Bevel>, SocketOf<&Bevel::Width>());
        f.Do(UpdateOf<&Bevel::Vertices>(bevel_entity, true));
        expect(output() == reference(MeshTopologyOp::BevelVertices, widths));
        f.Do(DeleteNodes{mesh, {bevel}});
    }
    {
        const auto [shrink, shrink_entity] = add_operator(KindOf<ShrinkFatten>, SocketOf<&ShrinkFatten::Distance>());
        // The kernel reads the factor at each vertex from the attribute, past the constant factor's zero check.
        const auto kernel = Reference(f, source_id, [&](uint32_t clone) {
            const PositionOperationTarget target{.StoreId = clone, .Selection = AllHandles(Mesh{meshes, clone})};
            ExecuteGeometryPositions(r, std::span{&target, 1u}, PositionEditOp::ShrinkFatten, 1.f, 1u, {.Factor = widths(clone)});
            return GeometrySelection{};
        });
        expect(kernel.Output != source && output() == kernel.Output);
        f.Do(DeleteNodes{mesh, {shrink}});
    }
    {
        // A vertex attribute does not fit the poke's per-face offset, and a face normal's z does.
        const auto [poke, poke_entity] = add_operator(KindOf<Poke>, SocketOf<&Poke::Offset>());
        expect(output() == source);
        const auto [normal, normal_entity] = f.Append(mesh, KindOf<Normal>);
        f.Do(UpdateOf<&Normal::Domain>(normal_entity, AttributeDomain::Face));
        f.Do(UpdateOf<&VectorMath::B>(dot_entity, vec3{0.f, 0.f, .3f}));
        f.Do(Link{mesh, normal, 0u, dot, 0u});
        f.Do(Link{mesh, dot, f.Out(mesh, dot), poke, SocketOf<&Poke::Offset>()});
        const auto offsets = [&](uint32_t clone) { return FloatAttribute<AttributeDomain::Face>(meshes, clone, [](const Mesh &m, he::FH face) { return .3f * m.GetNormal(face).z; }); };
        expect(output() == reference(MeshTopologyOp::Poke, offsets));
        f.Do(DeleteNodes{mesh, {poke}});
    }
    {
        // Link vector and rotation values into nested transform members while editing scale directly.
        const auto [transform, transform_entity] = f.Append(mesh, KindOf<TransformGeometry>);
        const auto [vector, vector_entity] = f.Append(mesh, KindOf<VectorValue>);
        const auto [rotation, rotation_entity] = f.Append(mesh, KindOf<RotationValue>);
        // The rotation leaves no two moved corners a GPU rounding apart in sort order.
        const Transform delta{{1.f, 2.f, 3.f}, numeric::AngleAxis(.5f, Normalize(vec3{1.f, 2.f, 3.f})), {2.f, 1.f, .5f}};
        f.Do(UpdateOf<&VectorValue::Value>(vector_entity, delta.P));
        f.Do(UpdateOf<&RotationValue::Value>(rotation_entity, delta.R));
        f.Do(UpdateOf<&TransformGeometry::Delta, &Transform::S>(transform_entity, delta.S));
        f.Do(Link{mesh, vector, 0u, transform, SocketOf<&TransformGeometry::Delta, &Transform::P>()});
        f.Do(Link{mesh, rotation, 0u, transform, SocketOf<&TransformGeometry::Delta, &Transform::R>()});
        f.Route(mesh, {0u, transform, 1u});
        expect(Near(Positions(meshes, f.Output(mesh)), Points(meshes, source_id, [&](uint32_t, vec3 p) { return delta.R * (delta.S * p) + delta.P; })));
        f.Do(DeleteNodes{mesh, {transform, vector, rotation}});
    }
    {
        const auto [store, store_entity] = f.Append(mesh, KindOf<StoreNamedAttribute>);
        f.Do(UpdateOf<&FloatValue::Value>(real_entity, 2.5f));
        f.Do(UpdateOf<&StoreNamedAttribute::Domain>(store_entity, AttributeDomain::Face));
        f.Do(SetAttributeName{store_entity, "heat"});
        f.Do(Link{mesh, real, 0u, store, 1u});
        f.Route(mesh, {0u, store, 1u});
        const auto heat = meshes.FindAttribute(f.Output(mesh), "heat");
        expect(fatal(heat != InvalidOffset));
        expect(meshes.GetAttribute(heat).Domain == AttributeDomain::Face && meshes.GetAttribute(heat).Type == AttributeType::Float);
        const auto values = meshes.AttributeView<float>(heat);
        for (const auto face : Mesh{meshes, f.Output(mesh)}.faces()) expect(values[*face] == 2.5f);
    }
}
} // namespace

void TestProcedures() {
    ProcedureFixture f;
    Fixture other;
    OperatorKinds(f, other);
    Survives(f, other, Graphs);
    ViewportOperators(f, other);
    Scenario(f, other, "ParameterSockets", ParameterSockets);
}
