#include "procedure/Procedure.h"
#include "TestMeshOrdinals.h"
#include "TestPaths.h"
#include "TopologyCases.h"
#include "action/Build.h"
#include "action/Errors.h"
#include "editor/Engine.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/Primitives.h"
#include "numeric/MatrixMath.h"
#include "object/ObjectOps.h"
#include "procedure/Cook.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/RenderTargets.h"
#include "render/Textures.h"
#include "scene/SceneGraph.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportEvents.h"

#include <boost/ut.hpp>

#include <map>
#include <print>

using boost::ut::expect, boost::ut::fatal;

namespace {
using namespace procedure;
using action::UpdateOf;
using action::nodes::AddNode, action::nodes::AddProcedure, action::nodes::ApplyProcedure, action::nodes::DeleteNodes, action::nodes::DuplicateNodes, action::nodes::InsertNode, action::nodes::Link, action::nodes::SelectNodes, action::nodes::SetAttributeName, action::nodes::SetInlineSelection;
using ::test::Op, ::test::Patchwork, ::test::PlanCase, ::test::Region, ::test::TopologyCases;

// One engine whose tests each begin a new project in a scratch directory of their own, which clears after the engine closes.
struct Fixture : TestDir, Engine {
    // Runs after each action and each direct commit, told whether the present node replays from its actions.
    std::function<void(bool replays)> Observe;
    explicit Fixture(const char *dir) : TestDir{dir}, Engine{true} { R.Context.get<ViewportExtent>().Value = {64, 64}; }
    void Reset(std::string_view name) {
        Observe = {};
        expect(P->New(Path / name));
    }
    uint32_t Do(auto a) {
        const auto node = P->Do(action::MakeAction(std::move(a)));
        Audit();
        if (Observe) Observe(true);
        return node.value();
    }
    // Does the action, which must fail.
    void Fails(auto a) {
        P->Do(action::MakeAction(std::move(a)));
        expect(!R.Context.get<action::Errors>().Messages.empty());
        R.Context.get<action::Errors>().Messages.clear();
    }
    state::Entity AddMesh(PrimitiveShape shape) {
        Do(action::object::AddMeshPrimitive{std::move(shape), std::make_unique<MeshInstanceCreateInfo>()});
        return R.get<const Instance>(FindActiveEntity(R)).Entity;
    }
    std::pair<uint32_t, state::Entity> Add(state::Entity mesh_entity, uint16_t kind) {
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
    std::vector<state::Entity> NodeEntities(state::Entity mesh_entity) const {
        std::vector<state::Entity> entities;
        for (const auto &node : R.get<const GeometryProcedure>(mesh_entity).Nodes) entities.push_back(node.Entity);
        return entities;
    }
    void Stage(auto a) {
        action::Emit(std::move(a), action::Phase::Stage);
        P->Frame(action::Drain());
        expect(R.Context.get<action::Errors>().Messages.empty());
    }
    void CommitGesture() {
        action::Commit();
        P->Frame(action::Drain());
        expect(!P->HasStaged());
    }
    void EnterEdit(Element element) {
        Do(action::view::SetInteractionMode{InteractionMode::Edit});
        Do(action::view::SetEditMode{.Mode = element});
    }
    void Render() {
        SubmitViewport(R, Viewport);
        WaitForRender(R);
    }
    std::vector<std::byte> Image() {
        Render();
        const auto &image = R.Context.get<const RenderTargets>().Resources->FinalColorImage;
        return ReadbackImageRgba8(R.Context.get<const mtl::Context>(), image, 0, 0, image.Extent);
    }
    const SceneViewUBO &View() const { return *reinterpret_cast<const SceneViewUBO *>(R.Context.get<const GpuBuffers>().SceneViewUBO.Contents().data()); }
    // Renders a frame and clicks the element nearest the world position, adding it to the selection with `toggle`.
    void Click(vec3 position, bool toggle) {
        Render();
        const auto clip = View().ViewProj * vec4{position.x, position.y, position.z, 1.f};
        Do(action::selection::ApplyEditElementClick{{.5f + .5f * clip.x / clip.w, .5f - .5f * clip.y / clip.w}, toggle, std::make_unique<RenderView>(R.Context.get<const GpuBuffers>().FrameView)});
    }
    // Process direct scene writes and commit them as one history node.
    void Commit(std::string label) {
        P->Settle();
        P->History.Commit(std::move(label), {});
        Audit();
        if (Observe) Observe(false);
    }
    void Audit() {
        std::string why;
        const bool valid = P->Audit(why);
        if (!valid) std::println("audit: {}", why);
        expect(valid);
        expect(R.Context.get<action::Errors>().Messages.empty());
    }
    // Replays the present node's actions from its parent's state, which must reach the node's state.
    void ValidateReplay(std::source_location at = std::source_location::current()) {
        const auto why = P->History.ValidateReplay();
        if (!why.empty()) std::println("replay differs in {} at {}", why, at.line());
        expect(why.empty());
    }
    // Replays every recorded action, which must reach the present persistent state.
    void ExpectReplay() {
        const auto persistent = P->History.MaterializeLive();
        expect(P->Replay());
        expect(P->History.MaterializeLive() == persistent);
    }
    MeshStore &Store() { return R.Context.get<MeshStore>(); }
    uint32_t Source(state::Entity mesh_entity) const { return R.get<const MeshHandle>(mesh_entity).StoreId; }
    state::Entity Object(state::Entity mesh_entity) const {
        for (const auto [e, instance] : R.view<const Instance>().each())
            if (instance.Entity == mesh_entity) return e;
        return state::Null;
    }
    // Return the evaluated procedure record when present, otherwise the source record.
    uint32_t Output(state::Entity mesh_entity) const { return *DrawnStoreId(R, mesh_entity); }
    uint64_t Cooks() const { return R.Context.get<const CookCount>().Value; }
    uint32_t Resyncs(state::Entity mesh_entity) const { return R.get<const ResyncState>(mesh_entity).Resyncs; }
    uint32_t AttributeCount(uint32_t id) {
        uint32_t count = 0u;
        Store().ForEachNamedAttribute(id, [&](uint32_t, const auto &) { ++count; });
        return count;
    }
    uint32_t LiveRecords() {
        uint32_t count = 0u;
        for (uint32_t id = 0u; id < 4096u; ++id) count += Store().TryGet(id) != nullptr;
        return count;
    }
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

// Each domain's elements as their sorted corner positions, sorted.
using Point = std::array<float, 3>;
struct Shape {
    std::vector<Point> Vertices;
    std::vector<std::array<Point, 2>> Edges;
    std::vector<std::vector<Point>> Faces;
    bool operator==(const Shape &) const = default;
};
Shape Describe(const MeshStore &meshes, uint32_t id, const GeometrySelection &elements) {
    const Mesh mesh{meshes, id};
    const auto point = [&](he::VH v) {
        const auto p = mesh.GetPosition(v);
        return Point{p.x, p.y, p.z};
    };
    Shape shape;
    for (const auto v : elements.Vertices) shape.Vertices.push_back(point(he::VH{v}));
    for (const auto e : elements.Edges) {
        const auto h = mesh.GetHalfedge(he::EH{e}, 0u);
        const auto from = point(mesh.GetFromVertex(h)), to = point(mesh.GetToVertex(h));
        shape.Edges.push_back({std::min(from, to), std::max(from, to)});
    }
    for (const auto f : elements.Faces) {
        auto &corners = shape.Faces.emplace_back();
        for (const auto v : mesh.fv_range(he::FH{f})) corners.push_back(point(v));
        std::ranges::sort(corners);
    }
    std::ranges::sort(shape.Vertices);
    std::ranges::sort(shape.Edges);
    std::ranges::sort(shape.Faces);
    return shape;
}
Shape Describe(const MeshStore &meshes, uint32_t id) { return Describe(meshes, id, AllHandles(Mesh{meshes, id})); }
// Capture persistent state, mesh output, and rendered image for a history node and track whether its actions replay.
struct Step {
    uint32_t Node;
    std::vector<std::byte> Persistent;
    Shape Output;
    std::vector<std::byte> Image;
    bool Replays{true};
};
Step PresentStep(Fixture &f, state::Entity mesh, bool replays = true) { return {*f.P->History.Present, f.P->History.MaterializeLive(), Describe(f.Store(), f.Output(mesh)), f.Image(), replays}; }
// Check restored node, persistent state, output, image, and the restored predicate without cooking.
// Check action replay here because its restore clears cook marks used during graph construction.
void ExpectRestores(Fixture &f, state::Entity mesh, std::span<const Step> steps, auto &&restored) {
    auto &p = *f.P;
    const auto move = [&](const Step &step) {
        const auto cooks = f.Cooks();
        p.Navigate(step.Node);
        expect(f.Cooks() == cooks);
        expect(p.History.Present == step.Node);
        expect(p.History.MaterializeLive() == step.Persistent);
        expect(Describe(f.Store(), f.Output(mesh)) == step.Output);
        expect(f.Image() == step.Image);
        restored();
        f.Audit();
    };
    for (size_t i = steps.size() - 1u; i-- > 0u;) move(steps[i]);
    for (size_t i = 0u; i < steps.size(); ++i) {
        if (i > 0u) move(steps[i]);
        if (steps[i].Replays) f.ValidateReplay();
    }
}

// The record and the selected elements a kernel edit leaves on a clone of `id`, where `edit` runs the kernel on the clone and returns the elements it selects.
// The clone releases, and its release commits.
struct Kernel {
    Shape Output, Selected;
};
Kernel Reference(Fixture &f, uint32_t id, auto &&edit) {
    auto &meshes = f.Store();
    const auto clone = f.Clone(id);
    const GeometrySelection selected = edit(clone);
    Kernel kernel{Describe(meshes, clone), Describe(meshes, clone, selected)};
    meshes.Release(clone);
    f.Commit("Release reference");
    return kernel;
}
// Each vertex's position, sorted.
std::vector<Point> Points(const MeshStore &meshes, uint32_t id, auto &&transform) {
    const Mesh mesh{meshes, id};
    std::vector<Point> points;
    for (const auto v : mesh.vertices()) {
        const auto p = transform(*v, mesh.GetPosition(v));
        points.push_back({p.x, p.y, p.z});
    }
    std::ranges::sort(points);
    return points;
}
std::vector<Point> Positions(const MeshStore &meshes, uint32_t id) {
    return Points(meshes, id, [](uint32_t, vec3 p) { return p; });
}
bool Near(const std::vector<Point> &a, const std::vector<Point> &b) {
    return a.size() == b.size() && std::ranges::equal(a, b, [](const Point &p, const Point &q) {
               return std::abs(p[0] - q[0]) < 1e-5f && std::abs(p[1] - q[1]) < 1e-5f && std::abs(p[2] - q[2]) < 1e-5f;
           });
}

size_t Faces(const Shape &shape) { return shape.Faces.size(); }
// Runs the tasks on a clone of `id`, each selecting every element, and returns the record they leave.
Shape SelectAllReference(Fixture &f, uint32_t id, std::vector<MeshTopologyTask> tasks) {
    for (auto &task : tasks) task.Flags |= TopologyFlagSelectAll;
    return Reference(f, id, [&](uint32_t clone) {
               f.Run(clone, tasks);
               return GeometrySelection{};
           })
        .Output;
}

struct OperatorNode {
    uint16_t Kind;
    void (*Set)(Fixture &, state::Entity node, const MeshTopologyTask &){};
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
        case DeleteLoose: return OperatorNode{KindOf<Delete>, [](Fixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&Delete::Mode>(n, DeleteMode(t.Op))); }};
        case DissolveVertices:
        case DissolveEdges:
        case DissolveFaces:
        case DissolveLimited:
        case DissolveDegenerate:
            return OperatorNode{KindOf<Dissolve>, [](Fixture &f, state::Entity n, const MeshTopologyTask &t) {
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
            return OperatorNode{KindOf<Merge>, [](Fixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&Merge::Mode>(n, t.Op == MergeByDistance ? MergeMode::ByDistance : t.Op == MergeCollapse ? MergeMode::Collapse :
                                                                                                                                             MergeMode::Center));
                                    if (t.Op == MergeByDistance) f.Do(UpdateOf<&Merge::Distance>(n, t.Param0));
                                }};
        case Decimate: return OperatorNode{KindOf<action::mesh::Decimate>};
        case AddPrimitives: return OperatorNode{KindOf<ConvexHull>};
        case Wireframe:
            return OperatorNode{KindOf<action::mesh::Wireframe>, [](Fixture &f, state::Entity n, const MeshTopologyTask &t) {
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
            return OperatorNode{KindOf<Bevel>, [](Fixture &f, state::Entity n, const MeshTopologyTask &t) {
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
            return OperatorNode{KindOf<Extrude>, [](Fixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&Extrude::Mode>(n, t.Op == ExtrudeVertices ? ExtrudeMode::Vertices : t.Op == ExtrudeEdges ? ExtrudeMode::Edges :
                                                                                                                                              ExtrudeMode::FacesIndividual));
                                }};
        case ExtrudeRegion:
            if (!(task.Flags & TopologyFlagTransformCopies)) return OperatorNode{KindOf<Extrude>};
            return OperatorNode{KindOf<ExtrudeRepeat>, [](Fixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&ExtrudeRepeat::Steps>(n, t.Steps));
                                    f.Do(UpdateOf<&ExtrudeRepeat::Offset>(n, t.CopyTranslation));
                                }};
        case Subdivide: return OperatorNode{KindOf<action::mesh::Subdivide>, [](Fixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(n, uint32_t(t.Param0))); }};
        case EdgeSplit: return OperatorNode{task.Flags & TopologyFlagRipSelectCopies ? KindOf<Rip> : KindOf<action::mesh::EdgeSplit>};
        case Poke: return OperatorNode{KindOf<action::mesh::Poke>, [](Fixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&action::mesh::Poke::Offset>(n, t.Param0)); }};
        case InsetRegion:
        case InsetIndividual:
            return OperatorNode{KindOf<Inset>, [](Fixture &f, state::Entity n, const MeshTopologyTask &t) {
                                    f.Do(UpdateOf<&Inset::Thickness>(n, t.Param0));
                                    f.Do(UpdateOf<&Inset::Depth>(n, t.Param1));
                                    f.Do(UpdateOf<&Inset::Individual>(n, t.Op == InsetIndividual));
                                    f.Do(UpdateOf<&Inset::Even>(n, (t.Flags & TopologyFlagEvenOffset) != 0u));
                                }};
        case Solidify: return OperatorNode{KindOf<action::mesh::Solidify>, [](Fixture &f, state::Entity n, const MeshTopologyTask &t) { f.Do(UpdateOf<&action::mesh::Solidify::Thickness>(n, t.Param0)); }};
        default: return std::nullopt;
    }
}

// Compare node output with direct kernel execution on a clone, including parameter edits and generated inputs.
// Verify unused branches remain unevaluated, record counts stay stable, and source records remain unchanged.
void TestOperatorKinds(Fixture &f) {
    auto &r = f.R;
    auto &meshes = f.Store();
    const auto cooks_like = [&](state::Entity mesh, uint32_t source_id, const Shape &source, uint32_t records, const Shape &reference) {
        expect(Describe(meshes, f.Output(mesh)) == reference);
        expect(f.Output(mesh) != source_id);
        expect(f.Source(mesh) == source_id);
        expect(Describe(meshes, source_id) == source);
        expect(f.LiveRecords() == records);
        f.Audit();
    };
    const auto observe = [&] { f.Observe = [&](bool replays) { if (replays) f.ValidateReplay(); }; };
    constexpr std::array<std::string_view, 6> WithoutNode{"RotateEdges", "ReplaceFaces", "ConnectVertices", "SplitNonplanarFaces", "SplitConcaveFaces", "KeepSelectedFaces"};
    for (const auto &c : TopologyCases) {
        if (std::ranges::contains(WithoutNode, c.Name)) continue;
        boost::ut::test(std::string{c.Name}) = [&] {
            f.Reset(c.Name);
            observe();
            const auto mesh = AddMesh(r, CreateMesh(r, {.Data = Patchwork()}).StoreId, MeshInstanceCreateInfo{}).first;
            f.Commit("Add patchwork");
            const auto source_id = f.Source(mesh);
            const auto source = Describe(meshes, source_id);
            auto task = PlanCase(c, meshes, Mesh{meshes, source_id});
            expect(fatal(task.has_value()));
            const auto node = NodeOf(*task);
            expect(fatal(node.has_value()));
            f.Do(AddProcedure{mesh});
            const auto records = f.LiveRecords();
            const auto [index, entity] = f.Add(mesh, node->Kind);
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
            cooks_like(mesh, source_id, source, records, reference(task->Param0));
            if (task->Param0 == 0.f) return;
            task->Param0 *= 2.f;
            node->Set(f, entity, *task);
            cooks_like(mesh, source_id, source, records, reference(task->Param0));
        };
    }
    // Generator kinds follow PrimitiveShape's alternatives.
    for (const PrimitiveShape shape : {PrimitiveShape{primitive::Plane{}}, PrimitiveShape{primitive::Circle{}}, PrimitiveShape{primitive::Cuboid{}}, PrimitiveShape{primitive::IcoSphere{}}, PrimitiveShape{primitive::UVSphere{}}, PrimitiveShape{primitive::Torus{}}, PrimitiveShape{primitive::Cylinder{}}, PrimitiveShape{primitive::Cone{}}}) {
        const auto name = ToString(shape);
        boost::ut::test(name) = [&] {
            f.Reset(name);
            observe();
            const auto mesh = f.AddMesh(primitive::Plane{});
            const auto source_id = f.Source(mesh);
            const auto source = Describe(meshes, source_id);
            f.Do(AddProcedure{mesh});
            const auto records = f.LiveRecords();
            const auto generator = f.Add(mesh, uint16_t(KindOf<primitive::Plane> + shape.index())).first;
            const auto subdivide = f.Add(mesh, KindOf<action::mesh::Subdivide>).first;
            f.Route(mesh, {generator, subdivide, 1u});
            const auto generated = CreateMesh(r, {.Data = primitive::CreateMesh(shape), .FlatShaded = true}).StoreId;
            const auto reference = SelectAllReference(f, generated, {Op(MeshTopologyOp::Subdivide, 1.f)});
            meshes.Release(generated);
            f.Commit("Release generated");
            cooks_like(mesh, source_id, source, records, reference);
        };
    }
    boost::ut::test("FanOut") = [&] {
        f.Reset("FanOut");
        observe();
        const auto mesh = f.AddMesh(primitive::Plane{});
        const auto source_id = f.Source(mesh);
        const auto source = Describe(meshes, source_id);
        f.Do(AddProcedure{mesh});
        const auto records = f.LiveRecords();
        const auto [first, first_entity] = f.Add(mesh, KindOf<action::mesh::Extrude>);
        const auto [second, second_entity] = f.Add(mesh, KindOf<action::mesh::Extrude>);
        f.Do(Link{mesh, 0u, 0u, first, 0u});
        f.Do(Link{mesh, 0u, 0u, second, 0u});
        const auto reference = SelectAllReference(f, source_id, {Op(MeshTopologyOp::ExtrudeRegion)});
        // A replay's validation restores its node, which clears the cook marks, so each link's marks check first.
        f.Observe = {};
        for (const auto read : {first, second}) {
            f.Do(Link{mesh, read, 2u, 1u, 0u});
            expect(!r.any_of<NodeError>(first_entity) && !r.any_of<NodeError>(second_entity));
            f.ValidateReplay();
            cooks_like(mesh, source_id, source, records, reference);
        }
    };
}

struct Graph {
    state::Entity Mesh;
    std::function<void(Fixture &)> Oracle;
    std::function<void(Fixture &, state::Entity)> Edit{};
};
struct GraphRow {
    std::string_view Name;
    Graph (*Build)(Fixture &);
};

// Compare action-driven outputs against independent kernel results and restored history.
// Skip full replay for cases that write scene state directly.
void Survives(Fixture &f, std::span<const GraphRow> rows) {
    auto &r = f.R;
    const auto &meshes = f.Store();
    for (const auto &row : rows) {
        boost::ut::test(std::string{row.Name}) = [&] {
            f.Reset(row.Name);
            auto &p = *f.P;
            std::vector<Step> steps;
            bool replays = true;
            auto cooks = f.Cooks();
            f.Observe = [&](bool replayable) {
                const auto mesh = r.get<const Instance>(FindActiveEntity(r)).Entity;
                expect(f.Cooks() == cooks + (r.all_of<GeometryProcedure>(mesh) ? 1u : 0u));
                cooks = f.Cooks();
                replays = replays && replayable;
                f.Audit();
                steps.push_back(PresentStep(f, mesh, replayable));
            };
            const auto graph = row.Build(f);
            f.Observe = {};
            const auto mesh = graph.Mesh;
            graph.Oracle(f);
            ExpectRestores(f, mesh, steps, [] {});
            if (replays) f.ExpectReplay();
            expect(Describe(meshes, f.Output(mesh)) == steps.back().Output);
            if (!graph.Edit) return;

            const auto output = Describe(meshes, f.Output(mesh));
            const auto nodes = f.NodeEntities(mesh);
            const auto records = f.LiveRecords();
            f.Do(action::object::Duplicate{});
            f.ValidateReplay();
            const auto copy = r.get<const Instance>(FindActiveEntity(r)).Entity;
            const auto &copied = r.get<const GeometryProcedure>(copy).Nodes;
            expect(copy != mesh && f.Output(copy) != f.Output(mesh));
            expect(std::ranges::equal(copied, r.get<const GeometryProcedure>(mesh).Nodes, {}, &NodeRecord::Kind, &NodeRecord::Kind));
            expect(std::ranges::none_of(copied, [&](const NodeRecord &node) { return std::ranges::contains(nodes, node.Entity); }));
            expect(Describe(meshes, f.Output(copy)) == output);
            graph.Edit(f, copy);
            f.ValidateReplay();
            expect(Describe(meshes, f.Output(copy)) != output);
            expect(Describe(meshes, f.Output(mesh)) == output);
            f.Audit();
            p.Undo();
            p.Undo();

            const auto present = [&] {
                return r.valid(mesh) && r.all_of<GeometryProcedure>(mesh) && std::ranges::all_of(nodes, [&](state::Entity e) { return r.valid(e); });
            };
            const auto absent = [&] {
                return !r.valid(mesh) && r.view<const GeometryProcedure>().empty() && std::ranges::none_of(nodes, [&](state::Entity e) { return r.valid(e); });
            };
            f.Do(action::object::Delete{});
            f.ValidateReplay();
            expect(absent());
            f.Audit();
            p.Undo();
            expect(present());
            expect(Describe(meshes, f.Output(mesh)) == output);
            f.Audit();
            p.Redo();
            expect(absent());
            f.Audit();
            p.Undo();

            const auto output_id = f.Output(mesh);
            f.Do(ApplyProcedure{mesh});
            f.ValidateReplay();
            expect(!r.any_of<GeometryProcedure, EvaluatedMesh, ResyncState, CookPlan>(mesh));
            expect(std::ranges::none_of(nodes, [&](state::Entity e) { return r.valid(e); }));
            expect(f.Source(mesh) == output_id);
            expect(Describe(meshes, f.Source(mesh)) == output);
            expect(f.LiveRecords() == records - 1u);
            f.Audit();
            if (replays) f.ExpectReplay();
            p.Undo();
            expect(present());
            expect(Describe(meshes, f.Output(mesh)) == output);
            expect(f.LiveRecords() == records);
            f.Audit();
            p.Redo();
            expect(!r.any_of<GeometryProcedure, EvaluatedMesh>(mesh));
            expect(Describe(meshes, f.Source(mesh)) == output);
            f.Audit();
        };
    }
}

Graph ExtrudeSubdivide(Fixture &f) {
    const auto mesh = f.AddMesh(primitive::Plane{});
    f.Do(AddProcedure{mesh});
    const auto extrude = f.Add(mesh, KindOf<action::mesh::Extrude>).first;
    const auto [subdivide, subdivide_entity] = f.Add(mesh, KindOf<action::mesh::Subdivide>);
    f.Route(mesh, {0u, extrude, subdivide, 1u});
    const auto once = Describe(f.Store(), f.Output(mesh));
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide_entity, 2u));
    expect(Describe(f.Store(), f.Output(mesh)) != once);
    return {
        mesh,
        [mesh](Fixture &f) {
            const auto output = Describe(f.Store(), f.Output(mesh));
            expect(output != Describe(f.Store(), f.Source(mesh)));
            expect(output == SelectAllReference(f, f.Source(mesh), {Op(MeshTopologyOp::ExtrudeRegion), Op(MeshTopologyOp::Subdivide, 2.f)}));
        },
        [](Fixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 3u), 3u)); },
    };
}

// A sphere's cap above y = 0.9 insets through Position, Compare and Inset.
// Thickness edits after the first cook resync the output record in place and keep the record count, and the default thickness returns the first output.
// An edit after an undo resyncs from the restored state, and moving a source vertex re-cooks through the output record, which follows the moved source.
Graph CapResync(Fixture &f) {
    auto &r = f.R;
    const auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::UVSphere{.Slices = 128u, .Stacks = 64u});
    f.Do(AddProcedure{mesh});
    const auto position = f.Add(mesh, KindOf<Position>).first;
    const auto [compare, compare_entity] = f.Add(mesh, KindOf<Compare>);
    const auto [inset, inset_entity] = f.Add(mesh, KindOf<action::mesh::Inset>);
    f.Do(UpdateOf<&Compare::Component>(compare_entity, VectorComponent::Y));
    f.Do(UpdateOf<&Compare::B>(compare_entity, .9f));
    f.Do(Link{mesh, position, 0u, compare, 0u});
    f.Do(Link{mesh, compare, f.Out(mesh, compare), inset, 1u});
    f.Route(mesh, {0u, inset, 1u});
    const auto output = f.Output(mesh);
    const auto records = f.LiveRecords();
    const auto first = Describe(meshes, output);
    // Each edit resyncs once and keeps the output record and the record count.
    const auto resyncs = [&](auto &&edit) {
        const auto before = f.Resyncs(mesh);
        edit();
        expect(f.Resyncs(mesh) == before + 1u);
        expect(f.Output(mesh) == output);
        expect(f.LiveRecords() == records);
        return Describe(meshes, output);
    };
    const auto set = [&](float thickness) { return resyncs([&] { f.Do(UpdateOf<&action::mesh::Inset::Thickness>(inset_entity, thickness)); }); };
    set(.05f);
    set(.2f);
    const auto thin = set(.01f);
    expect(set(action::mesh::Inset{}.Thickness) == first);
    f.P->Undo();
    expect(r.all_of<ResyncState>(mesh));
    expect(Describe(meshes, f.Output(mesh)) == thin);
    set(.03f);
    set(.06f);
    const auto source = f.Source(mesh);
    const PositionOperationTarget target{.StoreId = source, .Selection = GeometrySelection{.Vertices = {**Mesh{meshes, source}.vertices().begin()}}};
    for (const auto offset : {vec3{0.f, -.5f, 0.f}, vec3{.3f, 0.f, .1f}}) {
        resyncs([&] {
            ExecuteGeometryPositions(r, std::span{&target, 1u}, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = offset}}});
            r.emplace_or_replace<MeshPositionsChanged>(mesh);
            f.Commit("Move source vertex");
        });
    }
    return {
        mesh,
        // Compare output against the kernel insetting the moved source cap on a clone.
        [mesh](Fixture &f) {
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
            expect(Describe(f.Store(), f.Output(mesh)) == kernel.Output);
        },
        [](Fixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Inset::Thickness>(f.Node(mesh, 4u), .2f)); },
    };
}

// Verify Vector Math feeding Set Position doubles vertex positions and releases temporary output attributes.
Graph ScalePositions(Fixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto position = f.Add(mesh, KindOf<Position>).first;
    const auto [math, math_entity] = f.Add(mesh, KindOf<VectorMath>);
    const auto [set, set_entity] = f.Add(mesh, KindOf<SetPosition>);
    f.Do(UpdateOf<&VectorMath::Operation>(math_entity, VectorMathOp::Scale));
    f.Do(UpdateOf<&VectorMath::Scale>(math_entity, 2.f));
    f.Do(UpdateOf<&SetPosition::Offset>(set_entity, false));
    f.Do(Link{mesh, position, 0u, math, 0u});
    f.Do(Link{mesh, math, f.Out(mesh, math), set, 1u});
    f.Route(mesh, {0u, set, 1u});
    return {
        mesh,
        [mesh, math_entity, set_entity](Fixture &f) {
            const auto &meshes = f.Store();
            expect(Near(Positions(meshes, f.Output(mesh)), Points(meshes, f.Source(mesh), [](uint32_t, vec3 p) { return 2.f * p; })));
            expect(!f.R.any_of<NodeError>(math_entity) && !f.R.any_of<NodeError>(set_entity));
            expect(f.AttributeCount(f.Output(mesh)) == 0u);
            expect(f.AttributeCount(f.Source(mesh)) == 0u);
        },
        [](Fixture &f, state::Entity mesh) { f.Do(UpdateOf<&VectorMath::Scale>(f.Node(mesh, 3u), 3.f)); },
    };
}

// Position.z compared greater than zero selects the cube's upper face for Extrude.
Graph CompareSelectsExtrusion(Fixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto position = f.Add(mesh, KindOf<Position>).first;
    const auto [compare, compare_entity] = f.Add(mesh, KindOf<Compare>);
    const auto extrude = f.Add(mesh, KindOf<action::mesh::Extrude>).first;
    f.Do(UpdateOf<&Compare::Component>(compare_entity, VectorComponent::Z));
    f.Do(Link{mesh, position, 0u, compare, 0u});
    f.Do(Link{mesh, compare, f.Out(mesh, compare), extrude, 1u});
    f.Route(mesh, {0u, extrude, 1u});
    return {
        mesh,
        [mesh](Fixture &f) {
            const auto &meshes = f.Store();
            const auto output = Describe(meshes, f.Output(mesh));
            expect(Faces(output) == 10u);
            expect(f.AttributeCount(f.Output(mesh)) == 0u);
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
            expect(output == kernel.Output);
        },
        [](Fixture &f, state::Entity mesh) { f.Do(UpdateOf<&Compare::Component>(f.Node(mesh, 3u), VectorComponent::X)); },
    };
}

vec3 OffsetOf(uint32_t v) { return {float(v % 7u), 1.f, -float(v % 3u)}; }
Graph NamedAttributeReads(Fixture &f) {
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
    const auto [named, named_entity] = f.Add(mesh, KindOf<procedure::NamedAttribute>);
    const auto set = f.Add(mesh, KindOf<SetPosition>).first;
    f.Do(SetAttributeName{named_entity, "offset"});
    f.Do(UpdateOf<&procedure::NamedAttribute::Type>(named_entity, AttributeType::Vec3));
    f.Do(Link{mesh, named, 0u, set, 1u});
    f.Route(mesh, {0u, set, 1u});
    return {
        mesh,
        [mesh, named_entity](Fixture &f) {
            const auto &meshes = f.Store();
            expect(!f.R.any_of<NodeError>(named_entity));
            expect(Near(Positions(meshes, f.Output(mesh)), Points(meshes, f.Source(mesh), [](uint32_t v, vec3 p) { return p + OffsetOf(v); })));
            expect(f.AttributeCount(f.Output(mesh)) == 1u);
        },
        [](Fixture &f, state::Entity mesh) { f.Do(UpdateOf<&SetPosition::Offset>(f.Node(mesh, 3u), false)); },
    };
}

bool RestMatches(Fixture &f, state::Entity mesh) {
    const auto &meshes = f.Store();
    const auto id = f.Output(mesh);
    const auto rest = meshes.FindAttribute(id, "rest");
    if (rest == InvalidOffset || f.AttributeCount(id) != 1u) return false;
    const auto view = meshes.AttributeView<vec3>(rest);
    const Mesh output{meshes, id};
    for (const auto v : output.vertices()) {
        const auto d = view[*v] - output.GetPosition(v);
        if (std::abs(d.x) >= 1e-5f || std::abs(d.y) >= 1e-5f || std::abs(d.z) >= 1e-5f) return false;
    }
    return true;
}
// Store Named Attribute keeps Position under its name through a downstream subdivision and a re-cook with more cuts.
Graph StorePromotes(Fixture &f) {
    const auto mesh = f.AddMesh(primitive::Plane{});
    f.Do(AddProcedure{mesh});
    const auto position = f.Add(mesh, KindOf<Position>).first;
    const auto [store, store_entity] = f.Add(mesh, KindOf<StoreNamedAttribute>);
    const auto [subdivide, subdivide_entity] = f.Add(mesh, KindOf<action::mesh::Subdivide>);
    f.Do(SetAttributeName{store_entity, "rest"});
    f.Do(Link{mesh, position, 0u, store, 1u});
    f.Route(mesh, {0u, store, subdivide, 1u});
    expect(RestMatches(f, mesh));
    const auto faces = Faces(Describe(f.Store(), f.Output(mesh)));
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide_entity, 2u));
    expect(Faces(Describe(f.Store(), f.Output(mesh))) > faces);
    return {
        mesh,
        [mesh, store_entity](Fixture &f) {
            expect(!f.R.any_of<NodeError>(store_entity));
            expect(RestMatches(f, mesh));
            expect(f.AttributeCount(f.Source(mesh)) == 0u);
        },
        [](Fixture &f, state::Entity mesh) { f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(f.Node(mesh, 4u), 3u)); },
    };
}

// Vector Math on a vertex and a face attribute marks its node and forwards A, so Set Position offsets each vertex by its position.
Graph DomainMismatch(Fixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto position = f.Add(mesh, KindOf<Position>).first;
    const auto [normal, normal_entity] = f.Add(mesh, KindOf<Normal>);
    const auto [math, math_entity] = f.Add(mesh, KindOf<VectorMath>);
    const auto [set, set_entity] = f.Add(mesh, KindOf<SetPosition>);
    f.Do(UpdateOf<&Normal::Domain>(normal_entity, AttributeDomain::Face));
    f.Do(Link{mesh, position, 0u, math, 0u});
    f.Do(Link{mesh, normal, 0u, math, 1u});
    f.Do(Link{mesh, math, f.Out(mesh, math), set, 1u});
    f.Route(mesh, {0u, set, 1u});
    return {
        mesh,
        [mesh, normal_entity, math_entity, set_entity](Fixture &f) {
            const auto &r = f.R;
            const auto &meshes = f.Store();
            expect(r.any_of<NodeError>(math_entity));
            expect(!r.any_of<NodeError>(normal_entity) && !r.any_of<NodeError>(set_entity));
            expect(Near(Positions(meshes, f.Output(mesh)), Points(meshes, f.Source(mesh), [](uint32_t, vec3 p) { return 2.f * p; })));
            expect(f.AttributeCount(f.Output(mesh)) == 0u);
        },
        [](Fixture &f, state::Entity mesh) { f.Do(UpdateOf<&Normal::Domain>(f.Node(mesh, 3u), AttributeDomain::Vertex)); },
    };
}

// Verify joined cubes retain both material tables and default source-only attributes on generated elements.
// Resize the generated cube and verify the joined output changes.
Graph JoinGenerated(Fixture &f) {
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
    const auto [cube, cube_entity] = f.Add(mesh, KindOf<primitive::Cuboid>);
    const auto join = f.Add(mesh, KindOf<Join>).first;
    f.Do(Link{mesh, 0u, 0u, join, 0u});
    f.Do(Link{mesh, cube, f.Out(mesh, cube), join, 1u});
    f.Do(Link{mesh, join, 2u, 1u, 0u});
    const auto joined = Describe(meshes, f.Output(mesh));
    f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(cube_entity, vec3{2.f, .5f, 1.f}));
    expect(Describe(meshes, f.Output(mesh)) != joined);
    return {
        mesh,
        [mesh, join_entity = f.Node(mesh, join)](Fixture &f) {
            const auto &meshes = f.Store();
            expect(!f.R.any_of<NodeError>(join_entity));
            const auto id = f.Output(mesh), source = f.Source(mesh);
            const Mesh joined{meshes, id}, single{meshes, source};
            expect(joined.VertexCount() == 2u * single.VertexCount());
            expect(joined.EdgeCount() == 2u * single.EdgeCount());
            expect(joined.FaceCount() == 2u * single.FaceCount());
            expect(Faces(Describe(meshes, id)) == 2u * Faces(Describe(meshes, source)));
            const auto materials = meshes.Get(id).PrimitiveMaterials.Count;
            expect(materials == 2u * meshes.Get(source).PrimitiveMaterials.Count);
            std::array<uint32_t, 2> primitives{};
            for (const auto face : joined.faces()) {
                const auto primitive = meshes.Arenas().FacePrimitives.Get(meshes.Get(joined.GetStoreId()).FacePrimitives, *face);
                expect(primitive < materials);
                ++primitives[primitive < materials / 2u ? 0u : 1u];
            }
            expect(primitives[0] == single.FaceCount() && primitives[1] == single.FaceCount());
            const auto joined_weight = meshes.FindAttribute(id, "weight");
            expect(joined_weight != InvalidOffset);
            std::array<uint32_t, 2> weights{};
            const auto view = meshes.AttributeView<float>(joined_weight);
            for (const auto v : joined.vertices()) {
                if (view[*v] == 5.f) ++weights[0];
                else if (view[*v] == -1.f) ++weights[1];
            }
            expect(weights[0] == single.VertexCount() && weights[1] == single.VertexCount());
        },
        [](Fixture &f, state::Entity mesh) { f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(f.Node(mesh, 2u), vec3{.5f, .5f, .5f})); },
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
Graph HeatSubdivide(Fixture &f) {
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
    return {mesh, [id, original](Fixture &f) {
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

// The rows that write attributes or positions directly, the cap, named attribute, join and heat rows, skip the whole-history replays.
// The heat row edits a plain mesh, which has no procedure to duplicate, delete or apply.
constexpr std::array<GraphRow, 9> AttributeGraphs{{
    {"ExtrudeSubdivide", ExtrudeSubdivide},
    {"CapResync", CapResync},
    {"ScalePositions", ScalePositions},
    {"CompareSelectsExtrusion", CompareSelectsExtrusion},
    {"NamedAttributeReads", NamedAttributeReads},
    {"StorePromotes", StorePromotes},
    {"DomainMismatch", DomainMismatch},
    {"JoinGenerated", JoinGenerated},
    {"HeatSubdivide", HeatSubdivide},
}};

// Joining meshes with and without material tables resolves every face's primitive to the material its own mesh gave it.
void TestJoinPrimitiveTables(Fixture &f) {
    f.Reset("join-primitives");
    auto &meshes = f.Store();
    const auto cube = primitive::CreateMesh(primitive::Cuboid{});
    // A cube whose first three faces take material 3 and the others material 5, a cube with one material, and a wire square without a primitive table.
    const auto create = [&](MeshSource source) { return CreateMesh(f.R, std::move(source)).StoreId; };
    const auto palette = [&] { return create({.Data = cube, .Primitives = {.ElementPrimitiveIndices = {0u, 0u, 0u, 1u, 1u, 1u}, .MaterialIndices = {3u, 5u}}}); };
    const auto plain = [&] { return create({.Data = cube}); };
    const auto wire = [&] {
        MeshData data{{{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}}};
        data.Edges = {{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 0u}};
        return create({.Data = std::move(data)});
    };
    const auto points = [&] { return create({.Data = MeshData{{{0.f, 0.f, 1.f}, {1.f, 0.f, 1.f}, {0.f, 1.f, 1.f}}}, .Primitives = {.ElementPrimitiveIndices = {0u, 0u, 0u}, .MaterialIndices = {7u}}}); };
    expect(meshes.Get(wire()).PrimitiveMaterials.Count == 0u);
    // Each element's material through its primitive layer, sorted.
    const auto resolve = [&](uint32_t id, const auto &layer, Range values, auto &&elements) {
        const auto table = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(id).PrimitiveMaterials);
        std::vector<uint32_t> resolved;
        for (const auto element : elements) {
            const auto primitive = layer.Get(values, *element);
            expect(primitive < table.size());
            resolved.push_back(primitive < table.size() ? table[primitive] : InvalidOffset);
        }
        std::ranges::sort(resolved);
        return resolved;
    };
    const auto materials = [&](uint32_t id) { return resolve(id, meshes.Arenas().FacePrimitives, meshes.Get(id).FacePrimitives, Mesh{meshes, id}.faces()); };
    const std::vector<uint32_t> both{0u, 0u, 0u, 0u, 0u, 0u, 3u, 3u, 3u, 5u, 5u, 5u}, authored{3u, 3u, 3u, 5u, 5u, 5u};
    for (const auto &[first, second, expected] : {std::tuple{palette(), plain(), both}, std::tuple{plain(), palette(), both}, std::tuple{wire(), palette(), authored}, std::tuple{palette(), wire(), authored}}) {
        meshes.Join(f.R, first, second);
        f.Audit();
        expect(materials(first) == expected);
        meshes.Release(std::array{first, second});
        f.Audit();
    }
    // The wire square's vertices read the default material of their own table entry, and the points keep theirs.
    const auto lines = wire(), dots = points();
    meshes.Join(f.R, lines, dots);
    f.Audit();
    expect(resolve(lines, meshes.Arenas().VertexPrimitives, meshes.Get(lines).VertexPrimitives, Mesh{meshes, lines}.vertices()) == std::vector<uint32_t>{0u, 0u, 0u, 0u, 7u, 7u, 7u});
    meshes.Release(std::array{lines, dots});
    f.Commit("Release joins");
    f.Audit();
}

GeometrySelection Selection(const MeshStore &meshes, uint32_t id) {
    GeometrySelection selection;
    for (const auto [element, handles] : {std::pair{Element::Vertex, &selection.Vertices}, std::pair{Element::Edge, &selection.Edges}, std::pair{Element::Face, &selection.Faces}})
        meshes.GetSelectedElements(id, element).ForEach([&](uint32_t handle) { handles->push_back(handle); });
    return selection;
}

struct ViewportRow {
    std::string_view Name;
    Element Domain;
    uint32_t Clicks;
    uint16_t Kind;
    std::vector<float> Values;
    void (*Gesture)(Fixture &, float value, bool first);
    // Run the operator on the clone at the clicked handles and return its selected elements.
    GeometrySelection (*Kernel)(Fixture &, uint32_t clone, const GeometrySelection &clicked, float value);
};
// Run the topology task on the clone, publish its emitted selection, and return that selection.
GeometrySelection PublishedKernel(Fixture &f, uint32_t clone, MeshTopologyTask task) {
    task.SourceId = clone;
    PublishStoreSelection(f.Store(), task);
    f.Run(clone, {task});
    return Selection(f.Store(), clone);
}

// Compare viewport edits and output selection with kernel operations on a clone at the clicked ordinals.
// Verify drag steps update the output in place and commit as one history node.
// Undo and redo restore the graph without cooking, and the source record remains unchanged.
void TestViewportOperators(Fixture &f) {
    auto &r = f.R;
    const auto &meshes = f.Store();
    using D = MeshStore::ElementDomain;
    const auto domains = [](GeometrySelection &selection) {
        return std::array{std::tuple{D::Vertex, Element::Vertex, &selection.Vertices}, std::tuple{D::Edge, Element::Edge, &selection.Edges}, std::tuple{D::Face, Element::Face, &selection.Faces}};
    };
    const std::array<ViewportRow, 3> rows{{
        {"Extrude", Element::Face, 2u, KindOf<action::mesh::Extrude>, {0.f}, [](Fixture &f, float, bool) { f.Do(action::mesh::Extrude{}); }, [](Fixture &f, uint32_t clone, const GeometrySelection &clicked, float) { return PublishedKernel(f, clone, {.Op = MeshTopologyOp::ExtrudeRegion, .Selection = clicked}); }},
        {"BevelDrag", Element::Edge, 1u, KindOf<action::mesh::Bevel>, {.05f, .1f, .2f, .3f}, [](Fixture &f, float width, bool first) {
             if (first) f.Stage(action::mesh::Bevel{.Width = width});
             else f.Stage(UpdateOf<&action::mesh::Bevel::Width>(action::OnSelected{}, width)); }, [](Fixture &f, uint32_t clone, const GeometrySelection &clicked, float width) { return PublishedKernel(f, clone, {.Op = MeshTopologyOp::BevelEdges, .Param0 = width, .Steps = 1u, .Selection = clicked}); }},
        {"TransformDrag", Element::Vertex, 1u, KindOf<TransformElements>, {.5f, 1.f, 1.5f}, [](Fixture &f, float z, bool) { f.Stage(action::view::TransformElements{{.P = {0.f, 0.f, z}}}); }, [](Fixture &f, uint32_t clone, const GeometrySelection &clicked, float z) {
             const PositionOperationTarget target{.StoreId = clone, .Selection = clicked};
             expect(!ExecuteGeometryPositions(f.R, std::span{&target, 1u}, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = {0.f, 0.f, z}}}}).empty());
             return clicked; }},
    }};
    for (const auto &row : rows) {
        boost::ut::test(std::string{row.Name}) = [&] {
            f.Reset(row.Name);
            f.Observe = [&](bool replays) { if (replays) f.ValidateReplay(); };
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
            // A replay's validation restores its node, which clears the cook marks, so the gesture's marks check first.
            f.Observe = {};
            GeometrySelection ordinals;
            for (const auto [domain, element, list] : domains(ordinals))
                meshes.GetSelectedElements(f.Output(mesh), element).ForEach([&](uint32_t handle) { list->push_back(meshes.LiveElementOrdinal(f.Output(mesh), domain, ElementHandle{handle})); });
            expect(ordinals.Get(row.Domain).size() == row.Clicks);
            const auto source = Describe(meshes, f.Source(mesh));
            const auto start = PresentStep(f, mesh);
            const auto node_count = uint32_t(r.get<const GeometryProcedure>(mesh).Nodes.size());
            auto previous = start.Output;
            for (uint32_t i = 0u; i < row.Values.size(); ++i) {
                const auto resyncs = f.Resyncs(mesh);
                row.Gesture(f, row.Values[i], i == 0u);
                expect(r.get<const GeometryProcedure>(mesh).Nodes.size() == node_count + 1u);
                if (i > 0u) expect(f.Resyncs(mesh) == resyncs + 1u);
                const auto shape = Describe(meshes, f.Output(mesh));
                expect(shape != previous);
                previous = shape;
            }
            if (f.P->HasStaged()) f.CommitGesture();
            const auto &procedure = r.get<const GeometryProcedure>(mesh);
            const auto [node, kind] = procedure.Nodes.back();
            expect(procedure.Nodes.size() == node_count + 1u && kind == row.Kind);
            expect(procedure.Inputs[InputFirst(procedure, procedure.OutputNode)].FromNode == node_count);
            expect(procedure.Inputs[InputFirst(procedure, node_count)].FromNode == 0u);
            const auto &inline_selection = r.get<const InlineSelection>(node);
            uint32_t inline_count = 0u;
            for (const auto &block : row.Domain == Element::Vertex ? inline_selection.Vertices : row.Domain == Element::Edge ? inline_selection.Edges :
                                                                                                                               inline_selection.Faces)
                for (const auto word : block) inline_count += uint32_t(std::popcount(word));
            expect(inline_count == row.Clicks);
            expect(r.all_of<PublishSelection>(node) && !r.any_of<NodeError>(node));
            const auto done = PresentStep(f, mesh);
            expect(done.Output == previous);
            expect(f.P->History.Nodes[done.Node].Parent == start.Node);
            const auto selection = Selection(meshes, f.Output(mesh));
            const auto selected = Describe(meshes, f.Output(mesh), selection);
            expect(meshes.GetSelectionSummary(f.Output(mesh)).SelectedCount == selection.Get(row.Domain).size());
            f.Audit();
            f.Fails(action::mesh::ToSphere{});
            expect(Describe(meshes, f.Output(mesh)) == done.Output);
            ExpectRestores(f, mesh, std::array{start, done}, [&] { expect(r.valid(node) == (f.P->History.Present == done.Node)); });
            state::DirtySet cooks;
            cooks.bind(r, state::Clearing::Explicit);
            cooks.on<EvaluatedMesh>(state::On::Create | state::On::Update);
            f.Do(action::view::SetInteractionMode{InteractionMode::Object});
            expect(cooks.empty());
            expect(Describe(meshes, f.Output(mesh)) == done.Output);
            f.ValidateReplay();
            f.ExpectReplay();
            expect(Describe(meshes, f.Source(mesh)) == source);
            const auto kernel = Reference(f, f.Source(mesh), [&](uint32_t clone) {
                GeometrySelection clicked;
                for (const auto [domain, element, list] : domains(clicked)) {
                    for (const auto ordinal : ordinals.Get(element)) list->push_back(meshes.LiveElementAt(clone, domain, ordinal).Value);
                    std::ranges::sort(*list);
                }
                return row.Kernel(f, clone, clicked, row.Values.back());
            });
            expect(kernel.Output == done.Output);
            expect(kernel.Selected == selected);
            f.Audit();
        };
    }
}

Graph BooleanIntersection(Fixture &f) {
    const auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto [box, box_entity] = f.Add(mesh, KindOf<primitive::Cuboid>);
    f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(box_entity, vec3{2.f, .5f, .5f}));
    const auto [boolean, boolean_entity] = f.Add(mesh, KindOf<procedure::Boolean>);
    f.Do(UpdateOf<&procedure::Boolean::Operation>(boolean_entity, BooleanOperation::Intersection));
    f.Route(mesh, {0u, boolean, 1u});
    expect(!f.R.any_of<NodeError>(boolean_entity));
    expect(Describe(meshes, f.Output(mesh)) == Describe(meshes, f.Source(mesh)));
    f.Do(Link{mesh, box, f.Out(mesh, box), boolean, 1u});
    expect(!f.R.any_of<NodeError>(boolean_entity));
    // The cube keeps the bar's middle: 2 x 1 x 1.
    expect(std::abs(test::EnclosedVolume(Mesh{meshes, f.Output(mesh)}).value_or(-1.0) - 2.0) < 1e-4);
    f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(box_entity, vec3{2.f, .25f, .75f}));
    return {
        mesh,
        [mesh, boolean_entity](Fixture &f) {
            expect(!f.R.any_of<NodeError>(boolean_entity));
            expect(std::abs(test::EnclosedVolume(Mesh{f.Store(), f.Output(mesh)}).value_or(-1.0) - 1.5) < 1e-4);
        },
        [](Fixture &f, state::Entity mesh) { f.Do(UpdateOf<&procedure::Boolean::Operation>(f.Node(mesh, 3u), BooleanOperation::Union)); },
    };
}
constexpr std::array<GraphRow, 1> BooleanGraphs{{{"BooleanIntersection", BooleanIntersection}}};

// The boolean action makes the active object's mesh the active cube minus the other selected cube, in the active object's space, and undo restores it.
void TestBooleanAction(Fixture &f) {
    f.Reset("boolean-action");
    auto &r = f.R;
    const auto target = f.AddMesh(primitive::Cuboid{});
    const auto active = FindActiveEntity(r);
    f.Do(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = {.P = {1.f, 1.f, 1.f}}, .Select = MeshInstanceCreateInfo::SelectBehavior::Additive})});
    expect(FindActiveEntity(r) == active);
    const auto source = f.Source(target);
    f.Do(action::mesh::Boolean{BooleanOperation::Difference});
    expect(f.Source(target) != source);
    expect(std::abs(test::EnclosedVolume(Mesh{f.Store(), f.Source(target)}).value_or(-1.0) - 7.0) < 1e-4);
    for (const auto e : r.view<const Instance>())
        if (e != active) expect(Mesh{f.Store(), f.Source(r.get<const Instance>(e).Entity)}.VertexCount() == 8u);
    f.ValidateReplay();
    f.Audit();
    f.P->Undo();
    expect(f.Source(target) == source);
    f.Audit();
}

// The primitive's vertex positions, sorted.
std::vector<Point> ShapePositions(const auto &shape) {
    std::vector<Point> points;
    for (const auto p : primitive::CreateMesh(shape).Positions) points.push_back({p.x, p.y, p.z});
    std::ranges::sort(points);
    return points;
}

// Release the old primitive record in the same transaction that regenerates it after a shape edit.
Graph PrimitiveShapeEdit(Fixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(action::UpdateOfShape(mesh, &primitive::Cuboid::HalfExtents, vec3{2.f, .5f, 1.f}));
    f.Do(action::UpdateOfShape(mesh, &primitive::Cuboid::HalfExtents, vec3{1.f, 1.f, 3.f}));
    return {mesh, [mesh](Fixture &f) { expect(Near(Positions(f.Store(), f.Output(mesh)), ShapePositions(primitive::Cuboid{{1.f, 1.f, 3.f}}))); }};
}
constexpr std::array<GraphRow, 1> PlainGraphs{{{"PrimitiveShapeEdit", PrimitiveShapeEdit}}};

// Verify published instances retain their entities across edits and borrowed geometry remains unchanged.
// Generated geometry belongs to a separate mesh entity released with its final owner.
// Realization replaces published instances with geometry, and undo/redo restores each state without cooking.
void TestInstances(Fixture &f) {
    auto &r = f.R;
    const auto &meshes = f.Store();
    for (const auto instance_kind : {KindOf<ObjectInfo>, KindOf<primitive::Cuboid>}) {
        for (const bool realize : {false, true}) {
            const bool borrowed = instance_kind == KindOf<ObjectInfo>;
            const auto name = std::string{borrowed ? "ObjectInfo" : "Cuboid"} + (realize ? "Realize" : "Publish");
            boost::ut::test(name) = [&] {
                f.Reset(name);
                f.Observe = [&](bool) { f.ValidateReplay(); };
                const auto mesh = f.AddMesh(primitive::Plane{});
                const auto object = FindActiveEntity(r);
                const auto cube_mesh = f.AddMesh(primitive::Cuboid{});
                f.Do(AddProcedure{mesh});
                const auto to_points = f.Add(mesh, KindOf<MeshToPoints>).first;
                const auto [instancer, instancer_entity] = f.Add(mesh, KindOf<InstanceOnPoints>);
                const auto [instance, instance_entity] = f.Add(mesh, instance_kind);
                if (borrowed) f.Do(UpdateOf<&ObjectInfo::Object>(instance_entity, f.Object(cube_mesh)));
                f.Route(mesh, {0u, to_points, instancer});
                f.Do(Link{mesh, instance, f.Out(mesh, instance), instancer, 1u});
                const auto cube_record = f.Source(cube_mesh);
                const auto cube_shape = Describe(meshes, cube_record);
                // Later actions record steps, whose replays the restores validate after the checks read each action's cook marks.
                std::vector steps{PresentStep(f, mesh)};
                f.Observe = [&](bool) {
                    f.Audit();
                    steps.push_back(PresentStep(f, mesh));
                };
                const auto records = f.LiveRecords();
                uint32_t realizer = NoNode;
                if (realize) {
                    realizer = f.Add(mesh, KindOf<RealizeInstances>).first;
                    f.Route(mesh, {instancer, realizer, 1u});
                } else {
                    f.Do(Link{mesh, instancer, f.Out(mesh, instancer), 1u, 0u});
                }
                const auto published = [&] { return r.get<const ProcedureInstances>(mesh).Entities; };
                const auto entities = realize ? std::vector<state::Entity>{} : published();
                const auto owned = realize || borrowed ? state::Null : r.get<const ProcedureOwnedMesh>(mesh).MeshEntity;
                const auto instanced = borrowed ? cube_mesh : owned;
                const auto placed = [&](vec2 plane, vec3 cube, vec3 scale) {
                    expect(std::ranges::none_of(f.NodeEntities(mesh), [&](state::Entity e) { return r.any_of<NodeError>(e); }));
                    if (realize) {
                        expect(!r.any_of<ProcedureInstances, ProcedureOwnedMesh>(mesh));
                        expect(Mesh{meshes, f.Output(mesh)}.FaceCount() == 4u * 6u);
                        std::vector<Point> realized;
                        for (const auto &p : ShapePositions(primitive::Plane{plane}))
                            for (const auto &c : ShapePositions(primitive::Cuboid{cube})) realized.push_back({p[0] + scale.x * c[0], p[1] + scale.y * c[1], p[2] + scale.z * c[2]});
                        std::ranges::sort(realized);
                        expect(Near(Positions(meshes, f.Output(mesh)), realized));
                        return;
                    }
                    expect(published() == entities);
                    expect(entities.size() == 4u);
                    std::vector<Point> points;
                    for (const auto e : entities) {
                        expect(r.get<const Instance>(e).Entity == instanced && ParentOrNull(r, e) == object);
                        const auto &t = r.get<const Transform>(e);
                        expect(t.S == scale);
                        points.push_back({t.P.x, t.P.y, t.P.z});
                    }
                    std::ranges::sort(points);
                    expect(Near(points, ShapePositions(primitive::Plane{plane})));
                    expect(Mesh{meshes, f.Source(instanced)}.FaceCount() == 6u);
                    expect(Near(Positions(meshes, f.Source(instanced)), ShapePositions(primitive::Cuboid{cube})));
                    expect(Mesh{meshes, f.Output(mesh)}.VertexCount() == 0u);
                    if (borrowed) expect(!r.all_of<ProcedureOwnedMesh>(mesh));
                    else {
                        expect(r.get<const ProcedureOwnedMesh>(mesh).MeshEntity == owned);
                        expect(!r.any_of<Name, Transform, ObjectKind, Instance>(owned));
                    }
                };
                placed({1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f});
                expect(f.LiveRecords() == records + (owned != state::Null ? 1u : 0u));
                expect(f.Source(cube_mesh) == cube_record && Describe(meshes, cube_record) == cube_shape);
                // A realized result is a new record, which no source resyncs.
                const auto resyncs = realize ? 0u : f.Resyncs(mesh);
                f.Do(UpdateOf<&InstanceOnPoints::Scale>(instancer_entity, vec3{.5f, .5f, .5f}));
                if (!realize) expect(f.Resyncs(mesh) == resyncs + 1u);
                placed({1.f, 1.f}, {1.f, 1.f, 1.f}, {.5f, .5f, .5f});
                f.Do(action::UpdateOfShape(mesh, &primitive::Plane::HalfExtents, vec2{2.f, 3.f}));
                placed({2.f, 3.f}, {1.f, 1.f, 1.f}, {.5f, .5f, .5f});
                const auto record = owned != state::Null ? f.Source(owned) : InvalidOffset;
                if (borrowed) f.Do(action::UpdateOfShape(cube_mesh, &primitive::Cuboid::HalfExtents, vec3{.5f, .25f, .5f}));
                else f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(instance_entity, vec3{.5f, .25f, .5f}));
                placed({2.f, 3.f}, {.5f, .25f, .5f}, {.5f, .5f, .5f});
                if (owned != state::Null) expect(f.Source(owned) != record && !meshes.TryGet(record));
                f.Observe = {};
                ExpectRestores(f, mesh, steps, [] {});
                f.ExpectReplay();
                if (realize) {
                    const auto realized = f.LiveRecords();
                    f.Do(Link{mesh, instancer, f.Out(mesh, instancer), 1u, 0u});
                    f.ValidateReplay();
                    const auto published_owned = borrowed ? state::Null : r.get<const ProcedureOwnedMesh>(mesh).MeshEntity;
                    const auto published_record = borrowed ? InvalidOffset : f.Source(published_owned);
                    f.Do(Link{mesh, realizer, 1u, 1u, 0u});
                    placed({2.f, 3.f}, {.5f, .25f, .5f}, {.5f, .5f, .5f});
                    if (!borrowed) expect(!r.valid(published_owned) && !meshes.TryGet(published_record));
                    expect(f.LiveRecords() == realized);
                    f.ValidateReplay();
                    f.Audit();
                    return;
                }
                f.Do(action::selection::Select{object});
                f.Do(action::object::DuplicateLinked{});
                f.ValidateReplay();
                const auto linked = FindActiveEntity(r);
                expect(published().size() == 8u);
                expect(std::ranges::count(published(), linked, [&](state::Entity e) { return ParentOrNull(r, e); }) == 4);
                expect(std::ranges::all_of(entities, [&](state::Entity e) { return std::ranges::contains(published(), e); }));
                f.Do(action::object::Delete{});
                placed({2.f, 3.f}, {.5f, .25f, .5f}, {.5f, .5f, .5f});
                f.ValidateReplay();
                f.Do(action::selection::Select{object});
                f.Do(action::object::Duplicate{});
                f.ValidateReplay();
                const auto object_copy = FindActiveEntity(r);
                const auto mesh_copy = r.get<const Instance>(object_copy).Entity;
                const auto instanced_copy = borrowed ? cube_mesh : r.get<const ProcedureOwnedMesh>(mesh_copy).MeshEntity;
                if (!borrowed) {
                    expect(instanced_copy != owned && r.valid(owned) && r.valid(instanced_copy));
                    expect(f.Source(instanced_copy) != f.Source(owned) && Describe(meshes, f.Source(instanced_copy)) == Describe(meshes, f.Source(owned)));
                }
                const auto &instances_copy = r.get<const ProcedureInstances>(mesh_copy).Entities;
                expect(instances_copy.size() == 4u && std::ranges::all_of(instances_copy, [&](state::Entity e) { return r.get<const Instance>(e).Entity == instanced_copy && ParentOrNull(r, e) == object_copy; }));
                f.Do(ApplyProcedure{mesh});
                f.ValidateReplay();
                expect(!r.any_of<ProcedureInstances, ProcedureOwnedMesh>(mesh));
                expect(r.valid(instanced) && meshes.TryGet(f.Source(instanced)));
                expect(std::ranges::all_of(entities, [&](state::Entity e) { return r.valid(e) && r.get<const Instance>(e).Entity == instanced; }));
                f.P->Undo();
                const auto held = f.Source(instanced);
                f.Do(action::selection::Select{object});
                f.Do(action::object::Delete{});
                f.ValidateReplay();
                expect(!r.valid(mesh));
                expect(std::ranges::none_of(entities, [&](state::Entity e) { return r.valid(e); }));
                expect(r.valid(instanced) == borrowed && (meshes.TryGet(held) != nullptr) == borrowed);
                f.Audit();
                f.ExpectReplay();
            };
        }
    }
}

// An Object Info node reading its own procedure's mesh errs and emits nothing, so a join of it errs and forwards the source.
// One reading a mesh with a procedure of its own cooks after that procedure, so a resize there reaches the join in the same settle.
// An editing node reads a clone of the borrowed record and leaves the cube's drawn record as it was.
void TestObjectInfo(Fixture &f) {
    f.Reset("object-info");
    auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Plane{});
    const auto &meshes = f.Store();
    f.Do(AddProcedure{mesh});
    const auto [info, info_entity] = f.Add(mesh, KindOf<ObjectInfo>);
    f.Do(UpdateOf<&ObjectInfo::Object>(info_entity, f.Object(mesh)));
    const auto [join, join_entity] = f.Add(mesh, KindOf<Join>);
    f.Do(Link{mesh, 0u, 0u, join, 0u});
    f.Do(Link{mesh, info, 0u, join, 1u});
    f.Do(Link{mesh, join, 2u, 1u, 0u});
    expect(r.all_of<NodeError>(info_entity) && !r.any_of<NodeError>(join_entity));
    expect(Describe(meshes, f.Output(mesh)) == Describe(meshes, f.Source(mesh)));
    f.ValidateReplay();
    const auto cube = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{cube});
    const auto [box, box_entity] = f.Add(cube, KindOf<primitive::Cuboid>);
    f.Do(Link{cube, box, f.Out(cube, box), 1u, 0u});
    f.Do(UpdateOf<&ObjectInfo::Object>(info_entity, f.Object(cube)));
    expect(!r.any_of<NodeError>(info_entity) && !r.any_of<NodeError>(join_entity));
    f.ValidateReplay();
    const auto joined = [&](vec3 half_extents) {
        auto points = ShapePositions(primitive::Plane{});
        points.append_range(ShapePositions(primitive::Cuboid{half_extents}));
        std::ranges::sort(points);
        return Mesh{meshes, f.Output(mesh)}.FaceCount() == 7u && Near(Positions(meshes, f.Output(mesh)), points);
    };
    expect(joined(primitive::Cuboid{}.HalfExtents));
    const auto cooks = f.Cooks();
    f.Do(UpdateOf<&primitive::Cuboid::HalfExtents>(box_entity, vec3{.5f, .25f, .5f}));
    expect(f.Cooks() == cooks + 2u);
    expect(joined({.5f, .25f, .5f}));
    f.ValidateReplay();
    const auto cube_record = f.Output(cube);
    const auto cube_shape = Describe(meshes, cube_record);
    const auto subdivide = f.Add(mesh, KindOf<action::mesh::Subdivide>).first;
    f.Route(mesh, {info, subdivide, 1u});
    expect(!r.any_of<NodeError>(info_entity));
    expect(Mesh{meshes, f.Output(mesh)}.FaceCount() == 24u);
    expect(f.Output(cube) == cube_record && Describe(meshes, cube_record) == cube_shape);
    f.ValidateReplay();
    f.Audit();
}

// Node selection is the procedure's own: it tags node entities, names the active node, follows a node's removal and leaves the object selection alone.
void TestSelectNodes(Fixture &f) {
    f.Reset("select-nodes");
    auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto [subdivide, subdivide_entity] = f.Add(mesh, KindOf<action::mesh::Subdivide>);
    const auto [triangulate, triangulate_entity] = f.Add(mesh, KindOf<action::mesh::Triangulate>);
    expect(r.get<const GeometryProcedure>(mesh).ActiveNode == triangulate && !r.all_of<NodeSelected>(subdivide_entity) && r.all_of<NodeSelected>(triangulate_entity));
    const auto selected_objects = r.view<const Selected>().size();
    const auto active_object = FindActiveEntity(r);
    f.Do(SelectNodes{mesh, {subdivide, triangulate}, triangulate});
    expect(r.all_of<NodeSelected>(subdivide_entity) && r.all_of<NodeSelected>(triangulate_entity));
    expect(r.get<const GeometryProcedure>(mesh).ActiveNode == triangulate);
    expect(r.view<const Selected>().size() == selected_objects);
    expect(FindActiveEntity(r) == active_object);
    f.Do(SelectNodes{mesh, {triangulate}, triangulate});
    expect(!r.all_of<NodeSelected>(subdivide_entity));
    // Removing the node before the active one shifts the active index down with the table.
    f.Do(action::DestroyEntity{subdivide_entity});
    expect(r.get<const GeometryProcedure>(mesh).ActiveNode == triangulate - 1u);
    expect(r.all_of<NodeSelected>(triangulate_entity));
    f.P->Undo();
    expect(r.get<const GeometryProcedure>(mesh).ActiveNode == triangulate);
    f.P->Redo();
    f.Do(action::DestroyEntity{triangulate_entity});
    expect(r.get<const GeometryProcedure>(mesh).ActiveNode == NoNode);
    f.ValidateReplay();
    f.Audit();
}

// Duplicating nodes appends moved copies that keep only the links among them, and the copies take the selection.
void TestDuplicateNodes(Fixture &f) {
    f.Reset("duplicate-nodes");
    auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto [subdivide, subdivide_entity] = f.Add(mesh, KindOf<action::mesh::Subdivide>);
    const auto [triangulate, triangulate_entity] = f.Add(mesh, KindOf<action::mesh::Triangulate>);
    f.Route(mesh, {0u, subdivide, triangulate});
    f.Do(UpdateOf<&action::mesh::Subdivide::Cuts>(subdivide_entity, 3u));
    f.Do(SelectNodes{mesh, {subdivide, triangulate}, subdivide});
    f.Do(DuplicateNodes{mesh, {subdivide, triangulate}});
    const auto &procedure = r.get<const GeometryProcedure>(mesh);
    expect(procedure.Nodes.size() == 6u);
    const auto [copy_subdivide, copy_triangulate] = std::pair{procedure.Nodes[4], procedure.Nodes[5]};
    expect(copy_subdivide.Kind == KindOf<action::mesh::Subdivide> && copy_triangulate.Kind == KindOf<action::mesh::Triangulate>);
    expect(r.get<const action::mesh::Subdivide>(copy_subdivide.Entity).Cuts == 3u);
    expect(r.get<const NodeLayout>(copy_subdivide.Entity).Position == r.get<const NodeLayout>(subdivide_entity).Position);
    // Discard external links into copied nodes and retain links between the copies.
    expect(procedure.Inputs[InputFirst(procedure, 4u)].FromNode == NoNode);
    expect(procedure.Inputs[InputFirst(procedure, 5u)].FromNode == 4u);
    expect(!r.all_of<NodeSelected>(subdivide_entity) && r.all_of<NodeSelected>(copy_subdivide.Entity) && r.all_of<NodeSelected>(copy_triangulate.Entity));
    expect(procedure.ActiveNode == 4u);
    f.P->Undo();
    expect(r.get<const GeometryProcedure>(mesh).Nodes.size() == 4u && r.all_of<NodeSelected>(subdivide_entity));
    f.P->Redo();
    f.ValidateReplay();
    f.Audit();
}

// Deleting nodes heals each chain through them, and a reader of a deleted node without one linked source of its output's type unlinks.
// Each row deletes nodes and names inputs with the node they then read, or null, and the output it then cooks, and undo restores every node and link.
void TestDeleteNodes(Fixture &f) {
    f.Reset("delete-nodes");
    auto &r = f.R;
    const auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto subdivide = f.Add(mesh, KindOf<action::mesh::Subdivide>).first;
    const auto [triangulate, triangulate_entity] = f.Add(mesh, KindOf<action::mesh::Triangulate>);
    const auto loose = f.Add(mesh, KindOf<action::mesh::Subdivide>).first;
    const auto [boolean, boolean_entity] = f.Add(mesh, KindOf<procedure::Boolean>);
    const auto [reader, reader_entity] = f.Add(mesh, KindOf<action::mesh::Triangulate>);
    f.Route(mesh, {0u, subdivide, triangulate, 1u});
    f.Route(mesh, {loose, boolean, reader});
    f.Do(Link{mesh, 0u, 0u, boolean, 1u});
    const auto source = f.Node(mesh, 0u), output = f.Node(mesh, 1u);
    const auto reads = [&](state::Entity node, uint16_t input) {
        const auto &procedure = r.get<const GeometryProcedure>(mesh);
        const auto index = uint32_t(std::ranges::find(procedure.Nodes, node, &NodeRecord::Entity) - procedure.Nodes.begin());
        const auto from = procedure.Inputs[InputFirst(procedure, index) + input].FromNode;
        return from == NoNode ? state::Null : procedure.Nodes[from].Entity;
    };
    const auto links = [&] {
        std::vector<std::pair<uint32_t, uint16_t>> all;
        for (const auto link : r.get<const GeometryProcedure>(mesh).Inputs) all.emplace_back(link.FromNode, link.FromSocket);
        return all;
    };
    const auto nodes = f.NodeEntities(mesh);
    const auto linked = links();
    const auto cooked = Describe(meshes, f.Output(mesh));
    const auto triangulated = SelectAllReference(f, f.Source(mesh), {Op(MeshTopologyOp::Triangulate)});
    const auto unchanged = Describe(meshes, f.Source(mesh));
    struct Row {
        std::vector<uint32_t> Deleted;
        std::vector<std::tuple<state::Entity, uint16_t, state::Entity>> Reads;
        const Shape &Output;
    };
    const Row rows[]{
        {{subdivide}, {{triangulate_entity, 0u, source}, {output, 0u, triangulate_entity}}, triangulated},
        {{subdivide, triangulate}, {{output, 0u, source}}, unchanged},
        {{boolean}, {{reader_entity, 0u, state::Null}}, cooked},
        {{loose}, {{boolean_entity, 0u, state::Null}, {boolean_entity, 1u, source}}, cooked},
        {{loose, boolean}, {{reader_entity, 0u, source}}, cooked},
    };
    for (const auto &row : rows) {
        f.Do(DeleteNodes{mesh, row.Deleted});
        for (const auto &[node, input, from] : row.Reads) expect(reads(node, input) == from);
        expect(Describe(meshes, f.Output(mesh)) == row.Output);
        f.ValidateReplay();
        f.Audit();
        f.P->Undo();
        expect(f.NodeEntities(mesh) == nodes && links() == linked);
        expect(Describe(meshes, f.Output(mesh)) == cooked);
    }
}

// Insert dragged nodes through their first geometry sockets and shift downstream nodes right.
// A node without a geometry input, an unlinked input and a link whose source reads the node insert nothing.
void TestInsertNode(Fixture &f) {
    f.Reset("insert-node");
    auto &r = f.R;
    const auto &meshes = f.Store();
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto subdivide = f.Add(mesh, KindOf<action::mesh::Subdivide>).first;
    const auto triangulate = f.Add(mesh, KindOf<action::mesh::Triangulate>).first;
    const auto position = f.Add(mesh, KindOf<procedure::Position>).first;
    const auto output = f.Node(mesh, 1u);
    const auto subdivided = SelectAllReference(f, f.Source(mesh), {Op(MeshTopologyOp::Subdivide, 1.f)});
    const auto reads = [&](uint32_t node) { return r.get<const GeometryProcedure>(mesh).Inputs[InputFirst(r.get<const GeometryProcedure>(mesh), node)].FromNode; };
    f.Fails(InsertNode{mesh, position, 1u, 0u, 0u});
    f.Fails(InsertNode{mesh, subdivide, triangulate, 0u, 0u});
    const auto x = r.get<const NodeLayout>(output).Position.x;
    f.Do(SelectNodes{mesh, {subdivide}, subdivide});
    f.Stage(UpdateOf<&NodeLayout::Position>(action::OnSelectedDelta{}, uvec2{100u, 0u}));
    f.Stage(InsertNode{mesh, subdivide, 1u, 0u, 40u});
    f.CommitGesture();
    expect(reads(1u) == subdivide && reads(subdivide) == 0u && r.get<const NodeLayout>(f.Node(mesh, subdivide)).Position == uvec2{100u, 0u});
    expect(r.get<const NodeLayout>(output).Position.x == x + 40u && r.get<const NodeLayout>(f.Node(mesh, 0u)).Position.x == 0u);
    expect(Describe(meshes, f.Output(mesh)) == subdivided);
    f.ValidateReplay();
    f.Audit();
    f.P->Undo();
    expect(reads(1u) == 0u && reads(subdivide) == NoNode && r.get<const NodeLayout>(output).Position.x == x);
    f.P->Redo();
    f.Do(Link{mesh, triangulate, 2u, subdivide, 0u});
    f.Fails(InsertNode{mesh, triangulate, 1u, 0u, 0u});
}

// Verify empty bevel selections and unlinked boolean inputs pass through without errors.
// Verify an unresolved Object Info node regains its error on the next cook after restore clears it.
void TestNodeError(Fixture &f) {
    f.Reset("node-error");
    auto &r = f.R;
    const auto mesh = f.AddMesh(primitive::Plane{});
    f.Do(AddProcedure{mesh});
    const auto [bevel, bevel_entity] = f.Add(mesh, KindOf<action::mesh::Bevel>);
    f.Route(mesh, {0u, bevel, 1u});
    const auto &meshes = f.Store();
    const auto source = Describe(meshes, f.Source(mesh));
    expect(!r.any_of<NodeError>(bevel_entity));
    expect(Describe(meshes, f.Output(mesh)) != source);
    f.Do(SetInlineSelection{bevel_entity, std::vector<SelectionBlock>(3u), 1u, 1u});
    expect(!r.any_of<NodeError>(bevel_entity));
    expect(Describe(meshes, f.Output(mesh)) == source);
    // Leave nodes outside the output dependency graph unevaluated.
    const auto [_, loose_entity] = f.Add(mesh, KindOf<action::mesh::Subdivide>);
    expect(!r.any_of<NodeError>(loose_entity));
    const auto [boolean, boolean_entity] = f.Add(mesh, KindOf<procedure::Boolean>);
    f.Route(mesh, {bevel, boolean, 1u});
    expect(!r.any_of<NodeError>(boolean_entity));
    expect(Describe(meshes, f.Output(mesh)) == source);
    const auto [info, info_entity] = f.Add(mesh, KindOf<ObjectInfo>);
    const auto join = f.Add(mesh, KindOf<Join>).first;
    f.Route(mesh, {bevel, join, 1u});
    f.Do(Link{mesh, info, 0u, join, 1u});
    const auto fault = [&] { return r.all_of<NodeError>(info_entity) ? std::optional{r.get<const NodeError>(info_entity).Fault} : std::nullopt; };
    expect(fault() == NodeFault::NoMesh);
    expect(Describe(meshes, f.Output(mesh)) == source);
    f.ValidateReplay();
    f.Do(UpdateOf<&action::mesh::Bevel::Width>(bevel_entity, .2f));
    expect(fault() == NodeFault::NoMesh);
    f.P->Undo();
    expect(!fault());
    f.P->Redo();
    expect(!fault());
    f.Do(UpdateOf<&action::mesh::Bevel::Width>(bevel_entity, .3f));
    expect(fault() == NodeFault::NoMesh);
    f.ValidateReplay();
    f.Audit();
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

// Verify linked values override parameter constants and boolean values select all or no elements.
// Compare per-element attributes with kernel execution and reject attributes from incompatible domains.
// Verify temporary cook attributes are removed from output.
void TestParameterSockets(Fixture &f) {
    f.Reset("parameter-sockets");
    auto &r = f.R;
    auto &meshes = f.Store();
    using action::mesh::Bevel, action::mesh::ShrinkFatten, action::mesh::Poke;
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto source_id = f.Source(mesh);
    const auto source = Describe(meshes, source_id);
    const auto output = [&] { return Describe(meshes, f.Output(mesh)); };
    const auto fault = [&](state::Entity node) { return r.all_of<NodeError>(node) ? std::optional{r.get<const NodeError>(node).Fault} : std::nullopt; };
    const auto [real, real_entity] = f.Add(mesh, KindOf<FloatValue>);
    boost::ut::test("value") = [&] {
        const auto [bevel, bevel_entity] = f.Add(mesh, KindOf<Bevel>);
        f.Route(mesh, {0u, bevel, 1u});
        const auto one = output();
        f.Do(UpdateOf<&Bevel::Segments>(bevel_entity, 3u));
        const auto three = output();
        expect(three != one);
        f.Do(UpdateOf<&Bevel::Segments>(bevel_entity, 1u));
        const auto [integer, integer_entity] = f.Add(mesh, KindOf<IntegerValue>);
        f.Do(UpdateOf<&IntegerValue::Value>(integer_entity, 3u));
        const auto segments = SocketOf<&Bevel::Segments>(), width = SocketOf<&Bevel::Width>();
        expect(KindSockets(KindOf<Bevel>)[segments].Type == SocketType::Value && KindSockets(KindOf<Bevel>)[width].Type == SocketType::Attribute);
        f.Do(Link{mesh, integer, 0u, bevel, segments});
        expect(output() == three);
        f.ValidateReplay();
        f.Do(UpdateOf<&Bevel::Segments>(bevel_entity, 5u));
        expect(output() == three);
        f.Do(UpdateOf<&IntegerValue::Value>(integer_entity, 1u));
        expect(output() == one);
        f.P->Undo();
        expect(output() == three);
        f.P->Redo();
        expect(output() == one);
        f.Do(action::nodes::Unlink{mesh, bevel, segments});
        f.Do(UpdateOf<&Bevel::Segments>(bevel_entity, 3u));
        expect(output() == three);
        const auto position = f.Add(mesh, KindOf<Position>).first;
        f.Fails(Link{mesh, real, 0u, bevel, segments});
        f.Fails(Link{mesh, position, 0u, bevel, segments});
        f.Do(UpdateOf<&FloatValue::Value>(real_entity, .2f));
        f.Do(Link{mesh, real, 0u, bevel, width});
        f.Do(UpdateOf<&Bevel::Width>(bevel_entity, .3f));
        expect(output() == SelectAllReference(f, source_id, {Op(MeshTopologyOp::BevelEdges, .2f, 0u, 3u)}));
        f.Do(action::nodes::Unlink{mesh, bevel, width});
        f.Do(UpdateOf<&Bevel::Width>(bevel_entity, .1f));
        const auto [flag, flag_entity] = f.Add(mesh, KindOf<BooleanValue>);
        f.Do(Link{mesh, flag, 0u, bevel, 1u});
        expect(output() == source && !fault(bevel_entity));
        f.Do(UpdateOf<&BooleanValue::Value>(flag_entity, true));
        expect(output() == three);
        f.ValidateReplay();
        f.Do(DeleteNodes{mesh, {bevel}});
    };
    const auto position = f.Add(mesh, KindOf<Position>).first;
    const auto [dot, dot_entity] = f.Add(mesh, KindOf<VectorMath>);
    const auto [add, add_entity] = f.Add(mesh, KindOf<FloatMath>);
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
        const auto [node, entity] = f.Add(mesh, kind);
        f.Do(Link{mesh, add, f.Out(mesh, add), node, socket});
        f.Route(mesh, {0u, node, 1u});
        return std::pair{node, entity};
    };
    boost::ut::test("vertex bevel") = [&] {
        const auto [bevel, bevel_entity] = add_operator(KindOf<Bevel>, SocketOf<&Bevel::Width>());
        f.Do(UpdateOf<&Bevel::Vertices>(bevel_entity, true));
        expect(!fault(bevel_entity));
        f.ValidateReplay();
        expect(output() == reference(MeshTopologyOp::BevelVertices, widths));
        expect(output() != SelectAllReference(f, source_id, {Op(MeshTopologyOp::BevelVertices, .2f)}));
        expect(f.AttributeCount(f.Output(mesh)) == 0u && f.AttributeCount(source_id) == 0u);
        f.Do(DeleteNodes{mesh, {bevel}});
    };
    boost::ut::test("shrink fatten") = [&] {
        const auto [shrink, shrink_entity] = add_operator(KindOf<ShrinkFatten>, SocketOf<&ShrinkFatten::Distance>());
        expect(!fault(shrink_entity));
        f.ValidateReplay();
        // The kernel reads the factor at each vertex from the attribute, past the constant factor's zero check.
        const auto kernel = Reference(f, source_id, [&](uint32_t clone) {
            const PositionOperationTarget target{.StoreId = clone, .Selection = AllHandles(Mesh{meshes, clone})};
            ExecuteGeometryPositions(r, std::span{&target, 1u}, PositionEditOp::ShrinkFatten, 1.f, 1u, {.Factor = widths(clone)});
            return GeometrySelection{};
        });
        expect(kernel.Output != source && output() == kernel.Output);
        expect(f.AttributeCount(f.Output(mesh)) == 0u);
        f.Do(DeleteNodes{mesh, {shrink}});
    };
    boost::ut::test("poke") = [&] {
        // A vertex attribute does not fit the poke's per-face offset, and a face normal's z does.
        const auto [poke, poke_entity] = add_operator(KindOf<Poke>, SocketOf<&Poke::Offset>());
        expect(fault(poke_entity) == NodeFault::Attribute && output() == source);
        const auto [normal, normal_entity] = f.Add(mesh, KindOf<Normal>);
        f.Do(UpdateOf<&Normal::Domain>(normal_entity, AttributeDomain::Face));
        f.Do(UpdateOf<&VectorMath::B>(dot_entity, vec3{0.f, 0.f, .3f}));
        f.Do(Link{mesh, normal, 0u, dot, 0u});
        f.Do(Link{mesh, dot, f.Out(mesh, dot), poke, SocketOf<&Poke::Offset>()});
        expect(!fault(poke_entity));
        f.ValidateReplay();
        const auto offsets = [&](uint32_t clone) { return FloatAttribute<AttributeDomain::Face>(meshes, clone, [](const Mesh &m, he::FH face) { return .3f * m.GetNormal(face).z; }); };
        expect(output() == reference(MeshTopologyOp::Poke, offsets));
        expect(f.AttributeCount(f.Output(mesh)) == 0u);
        f.Do(DeleteNodes{mesh, {poke}});
    };
    // Link vector and rotation values into nested transform members while editing scale directly.
    boost::ut::test("transform geometry") = [&] {
        const auto [transform, transform_entity] = f.Add(mesh, KindOf<TransformGeometry>);
        const auto [vector, vector_entity] = f.Add(mesh, KindOf<VectorValue>);
        const auto [rotation, rotation_entity] = f.Add(mesh, KindOf<RotationValue>);
        // The rotation leaves no two moved corners a GPU rounding apart in sort order.
        const Transform delta{{1.f, 2.f, 3.f}, numeric::AngleAxis(.5f, Normalize(vec3{1.f, 2.f, 3.f})), {2.f, 1.f, .5f}};
        f.Do(UpdateOf<&VectorValue::Value>(vector_entity, delta.P));
        f.Do(UpdateOf<&RotationValue::Value>(rotation_entity, delta.R));
        f.Do(UpdateOf<&TransformGeometry::Delta, &Transform::S>(transform_entity, delta.S));
        f.Do(Link{mesh, vector, 0u, transform, SocketOf<&TransformGeometry::Delta, &Transform::P>()});
        f.Do(Link{mesh, rotation, 0u, transform, SocketOf<&TransformGeometry::Delta, &Transform::R>()});
        f.Route(mesh, {0u, transform, 1u});
        expect(!fault(transform_entity));
        expect(Near(Positions(meshes, f.Output(mesh)), Points(meshes, source_id, [&](uint32_t, vec3 p) { return delta.R * (delta.S * p) + delta.P; })));
        f.ValidateReplay();
        f.Do(DeleteNodes{mesh, {transform, vector, rotation}});
    };
    boost::ut::test("store") = [&] {
        const auto [store, store_entity] = f.Add(mesh, KindOf<StoreNamedAttribute>);
        f.Do(UpdateOf<&FloatValue::Value>(real_entity, 2.5f));
        f.Do(UpdateOf<&StoreNamedAttribute::Domain>(store_entity, AttributeDomain::Face));
        f.Do(SetAttributeName{store_entity, "heat"});
        f.Do(Link{mesh, real, 0u, store, 1u});
        f.Route(mesh, {0u, store, 1u});
        expect(!fault(store_entity));
        const auto heat = meshes.FindAttribute(f.Output(mesh), "heat");
        expect(fatal(heat != InvalidOffset));
        expect(meshes.GetAttribute(heat).Domain == AttributeDomain::Face && meshes.GetAttribute(heat).Type == AttributeType::Float);
        const auto values = meshes.AttributeView<float>(heat);
        for (const auto face : Mesh{meshes, f.Output(mesh)}.faces()) expect(values[*face] == 2.5f);
        expect(f.AttributeCount(f.Output(mesh)) == 1u && f.AttributeCount(source_id) == 0u);
        f.ValidateReplay();
    };
    f.Audit();
}
} // namespace

void TestProcedures() {
    Fixture f{"/tmp/mesheditor-scratch/procedure"};
    boost::ut::test("every operator kind cooks like its kernel") = [&] { TestOperatorKinds(f); };
    boost::ut::test("every attribute graph survives history") = [&] { Survives(f, AttributeGraphs); };
    boost::ut::test("join primitive tables") = [&] { TestJoinPrimitiveTables(f); };
    boost::ut::test("viewport operators append nodes") = [&] { TestViewportOperators(f); };
    boost::ut::test("every boolean graph survives history") = [&] { Survives(f, BooleanGraphs); };
    boost::ut::test("boolean action") = [&] { TestBooleanAction(f); };
    boost::ut::test("instances publish or realize") = [&] { TestInstances(f); };
    boost::ut::test("every plain graph survives history") = [&] { Survives(f, PlainGraphs); };
    boost::ut::test("object info") = [&] { TestObjectInfo(f); };
    boost::ut::test("select nodes") = [&] { TestSelectNodes(f); };
    boost::ut::test("duplicate nodes") = [&] { TestDuplicateNodes(f); };
    boost::ut::test("delete nodes") = [&] { TestDeleteNodes(f); };
    boost::ut::test("insert node") = [&] { TestInsertNode(f); };
    boost::ut::test("node error") = [&] { TestNodeError(f); };
    boost::ut::test("parameter sockets") = [&] { TestParameterSockets(f); };
}
