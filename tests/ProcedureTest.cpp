#include "procedure/Procedure.h"
#include "TestMeshOrdinals.h"
#include "TestPaths.h"
#include "TopologyCases.h"
#include "action/ActionDrain.h"
#include "action/Build.h"
#include "action/Emit.h"
#include "action/Errors.h"
#include "editor/Engine.h"
#include "gpu/SceneViewUBO.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "metal/Dispatch.h"
#include "numeric/MatrixMath.h"
#include "object/ObjectOps.h"
#include "procedure/Cook.h"
#include "project/Project.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/RenderTargets.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "state/Scene.h"
#include "viewport/RenderExtent.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportEvents.h"

#include <boost/ut.hpp>

#include <bit>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <source_location>
#include <string>
#include <tuple>

using boost::ut::expect, boost::ut::fatal;

namespace {
using namespace procedure;
using action::UpdateOf;
using action::nodes::AddNode, action::nodes::AddProcedure, action::nodes::ApplyProcedure, action::nodes::Link, action::nodes::SetAttributeName, action::nodes::SetInlineSelection;
using ::test::Op, ::test::Patchwork, ::test::PlanCase, ::test::Region, ::test::TopologyCases;

// One engine whose tests each begin a new project in a scratch directory of their own, which clears after the engine closes.
struct Fixture : TestDir, Engine {
    // Runs after each action and each direct commit, told whether the present node replays from its actions.
    std::function<void(bool replays)> Observe;
    explicit Fixture(const char *dir) : TestDir{dir}, Engine{true} { R.Context.get<ViewportExtent>().Value = {64, 64}; }
    // Begins a new project in the directory named for the test, observing nothing.
    void Reset(std::string_view name) {
        Observe = {};
        expect(P->New(Path / name));
    }
    template<typename A> uint32_t Do(A a) {
        const auto node = P->Do(action::MakeAction(std::move(a)));
        expect(R.Context.get<action::Errors>().Messages.empty());
        if (Observe) Observe(true);
        return node.value();
    }
    // Adds a primitive object and returns its mesh entity.
    state::Entity AddMesh(PrimitiveShape shape) {
        Do(action::object::AddMeshPrimitive{std::move(shape), std::make_unique<MeshInstanceCreateInfo>()});
        return R.get<const Instance>(FindActiveEntity(R)).Entity;
    }
    // Adds a node and returns its index and entity.
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
    std::vector<state::Entity> NodeEntities(state::Entity mesh_entity) const {
        std::vector<state::Entity> entities;
        for (const auto &node : R.get<const GeometryProcedure>(mesh_entity).Nodes) entities.push_back(node.Entity);
        return entities;
    }
    // Stages the action as the open gesture's next step.
    template<typename A> void Stage(A a) {
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
    // Settles direct scene writes and commits them as one history node.
    void Commit(std::string label) {
        P->Settle();
        P->History.Commit(std::move(label), {});
        if (Observe) Observe(false);
    }
    void Audit() {
        std::string why;
        const bool valid = P->Audit(why);
        if (!valid) std::printf("audit: %s\n", why.c_str());
        expect(valid);
        expect(R.Context.get<action::Errors>().Messages.empty());
    }
    // Replays the present node's actions from its parent's state, which must reach the node's state.
    void ValidateReplay(std::source_location at = std::source_location::current()) {
        const auto why = P->History.ValidateReplay();
        if (!why.empty()) std::printf("replay differs in %s at %u\n", why.c_str(), at.line());
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
    // The record the mesh entity draws: its procedure's evaluated record, or its source.
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
    // A clone of the record that the kernel edits directly.
    uint32_t Clone(uint32_t id) {
        auto &meshes = Store();
        meshes.PlanClone(Mesh{meshes, id});
        meshes.CommitReserves();
        CloneCopies copies;
        const auto clone = meshes.CloneMeshes(copies, std::array{id}, true, {}).front();
        mtl::ComputeChain chain{meshes.BufferContext()};
        copies.Encode(chain, GetMeshPipelines(R));
        chain.Submit();
        meshes.RebuildSelectionIndex(std::array{clone});
        return clone;
    }
    // Runs the tasks as dependent stages on the record, each of which changes it.
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
// A history node with its persistent state, the mesh's output and the rendered image there, and whether the node replays from its actions.
struct Step {
    uint32_t Node;
    std::vector<std::byte> Persistent;
    Shape Output;
    std::vector<std::byte> Image;
    bool Replays{true};
};
// The present node as a step of the mesh.
Step PresentStep(Fixture &f, state::Entity mesh, bool replays = true) { return {*f.P->History.Present, f.P->History.MaterializeLive(), Describe(f.Store(), f.Output(mesh)), f.Image(), replays}; }
// Navigates back through the steps to the first and forward again to the last.
// Each move cooks nothing and restores its step's node, persistent state, output and image, and `restored` holds.
// Each step that replays reaches its state from its parent's, checked here because the check's restore clears the cook marks a builder reads.
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

// A node kind, and the parameter updates that make its cook plan a task.
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

// Each operator row's node, given the row's region as its inline selection, cooks the patchwork like the kernel running the row's task on a clone, and again after its Param0 doubles.
// Each generator kind cooks a subdivided mesh like the kernel subdividing its primitive mesh, without reading the source.
// A source feeding two extrusions serves the one the result reads, the other stays inert, and the cooks keep the record count.
// Every cook leaves the source record as it was in a new output record.
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
    // These rows name operators without a node kind: edge rotation, face replacement, vertex connection, nonplanar and concave splits, and keeping selected faces.
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

// A graph a row builds through actions: its mesh entity, the check of its final output, and whether the mesh has a procedure.
struct Graph {
    state::Entity Mesh;
    std::function<void(Fixture &)> Oracle;
    bool Procedure{true};
};
struct GraphRow {
    std::string_view Name;
    Graph (*Build)(Fixture &);
};

// Each row's graph, built action by action, cooks its procedure once per action, and its final output passes the row's oracle.
// Navigating its history restores every step's state, output and image without cooking, each action replays from its parent's state, and replaying the whole history reproduces it.
// Deleting a procedure mesh's object destroys its nodes until undo, and applying its procedure makes the output its source until undo.
// A row that writes the scene directly skips the whole-history replays, which replay only actions.
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
            if (!graph.Procedure) return;

            const auto output = Describe(meshes, f.Output(mesh));
            const auto nodes = f.NodeEntities(mesh);
            const auto records = f.LiveRecords();
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

// A plane through Extrude and Subdivide, whose cuts rise to two, matches the kernel's extrusion and subdivision.
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
    f.Do(UpdateOf<&Compare::Component>(compare_entity, 1u));
    f.Do(UpdateOf<&Compare::B>(compare_entity, .9f));
    f.Do(Link{mesh, position, 0u, compare, 0u});
    f.Do(Link{mesh, compare, 2u, inset, 1u});
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
        // The output matches the kernel insetting the moved source's cap on a clone.
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
    };
}

// Position scaled by Vector Math into Set Position moves every vertex to twice its position, and the cook's temporaries leave the output.
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
    f.Do(Link{mesh, math, 2u, set, 1u});
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
    };
}

// Position.z compared greater than zero selects the cube's upper face for Extrude.
Graph CompareSelectsExtrusion(Fixture &f) {
    const auto mesh = f.AddMesh(primitive::Cuboid{});
    f.Do(AddProcedure{mesh});
    const auto position = f.Add(mesh, KindOf<Position>).first;
    const auto [compare, compare_entity] = f.Add(mesh, KindOf<Compare>);
    const auto extrude = f.Add(mesh, KindOf<action::mesh::Extrude>).first;
    f.Do(UpdateOf<&Compare::Component>(compare_entity, 2u));
    f.Do(Link{mesh, position, 0u, compare, 0u});
    f.Do(Link{mesh, compare, 2u, extrude, 1u});
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
    };
}

vec3 OffsetOf(uint32_t v) { return {float(v % 7u), 1.f, -float(v % 3u)}; }
// Named Attribute reads a vec3 attribute set on the source, which Set Position adds to each vertex.
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
    };
}

// Whether the output holds exactly one attribute, "rest", equal to each vertex's position.
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
    f.Do(Link{mesh, math, 2u, set, 1u});
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
    };
}

// A cube joined with a generated cube holds both cubes' elements and material tables, and an attribute only the source has holds its default on the generated cube.
// Resizing the generated cube changes the joined output.
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
    f.Do(Link{mesh, cube, 0u, join, 1u});
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
                const auto primitive = meshes.Arenas().FacePrimitives.Get(*face);
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
    };
}

// Each live vertex's handle and heat.
std::map<uint32_t, float> Heat(const MeshStore &meshes, uint32_t id) {
    std::map<uint32_t, float> heat;
    const auto view = meshes.AttributeView<float>(meshes.FindAttribute(id, "heat"));
    meshes.Arenas().Vertices.ForEach(meshes.Get(id).Vertices, [&](uint32_t v, uint32_t) { heat.emplace(v, view[v]); });
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
    meshes.SetAttribute<float>(heat, vertices.Offset, values);
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
            },
            false};
}

// The rows that write attributes or positions directly, the cap, named attribute, join and heat rows, skip the whole-history replays.
// The heat row edits a plain mesh, which has no procedure to delete or apply.
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
    // Three points drawn with material 7.
    const auto points = [&] { return create({.Data = MeshData{{{0.f, 0.f, 1.f}, {1.f, 0.f, 1.f}, {0.f, 1.f, 1.f}}}, .Primitives = {.ElementPrimitiveIndices = {0u, 0u, 0u}, .MaterialIndices = {7u}}}); };
    expect(meshes.Get(wire()).PrimitiveMaterials.Count == 0u);
    // Each element's material through its primitive layer, sorted.
    const auto resolve = [&](uint32_t id, const auto &layer, auto &&elements) {
        const auto table = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(id).PrimitiveMaterials);
        std::vector<uint32_t> resolved;
        for (const auto element : elements) {
            const auto primitive = layer.Get(*element);
            expect(primitive < table.size());
            resolved.push_back(primitive < table.size() ? table[primitive] : InvalidOffset);
        }
        std::ranges::sort(resolved);
        return resolved;
    };
    const auto materials = [&](uint32_t id) { return resolve(id, meshes.Arenas().FacePrimitives, Mesh{meshes, id}.faces()); };
    const std::vector<uint32_t> both{0u, 0u, 0u, 0u, 0u, 0u, 3u, 3u, 3u, 5u, 5u, 5u}, authored{3u, 3u, 3u, 5u, 5u, 5u};
    for (const auto &[first, second, expected] : {std::tuple{palette(), plain(), both}, std::tuple{plain(), palette(), both}, std::tuple{wire(), palette(), authored}, std::tuple{palette(), wire(), authored}}) {
        meshes.Join(f.R, first, second);
        expect(materials(first) == expected);
        meshes.Release(std::array{first, second});
    }
    // The wire square's vertices read the default material of their own table entry, and the points keep theirs.
    const auto lines = wire(), dots = points();
    meshes.Join(f.R, lines, dots);
    expect(resolve(lines, meshes.Arenas().VertexPrimitives, Mesh{meshes, lines}.vertices()) == std::vector<uint32_t>{0u, 0u, 0u, 0u, 7u, 7u, 7u});
    meshes.Release(std::array{lines, dots});
    f.Commit("Release joins");
    f.Audit();
}

// The elements the record selects, in each domain.
GeometrySelection Selection(const MeshStore &meshes, uint32_t id) {
    GeometrySelection selection;
    for (const auto [element, handles] : {std::pair{Element::Vertex, &selection.Vertices}, std::pair{Element::Edge, &selection.Edges}, std::pair{Element::Face, &selection.Faces}})
        meshes.GetSelectedElements(id, element).ForEach([&](uint32_t handle) { handles->push_back(handle); });
    return selection;
}

// A viewport operator on the clicked elements of a procedure cube: the domain and count it clicks, the node kind it appends, its gesture's step values, and the kernel it plans.
struct ViewportRow {
    std::string_view Name;
    Element Domain;
    uint32_t Clicks;
    uint16_t Kind;
    std::vector<float> Values;
    // Takes the gesture's step with the value, the first step appending the node.
    void (*Gesture)(Fixture &, float value, bool first);
    // Runs the operator with the value on the clone at the clicked handles, and returns the elements it selects.
    GeometrySelection (*Kernel)(Fixture &, uint32_t clone, const GeometrySelection &clicked, float value);
};
// Runs the topology task on the clone, publishing its emitted selection, and returns the clone's selection.
GeometrySelection PublishedKernel(Fixture &f, uint32_t clone, MeshTopologyTask task) {
    task.SourceId = clone;
    PublishStoreSelection(f.Store(), task);
    f.Run(clone, {task});
    return Selection(f.Store(), clone);
}

// Each operator appends its node kind between the source and the output, holding the clicked selection and publishing its edit's selection, and each later gesture step resyncs the output in place.
// The commit leaves one history node after the start, an operator without a node kind fails and leaves the mesh, and undo and redo restore both nodes without cooking.
// Leaving edit mode cooks nothing, replay reproduces the history, the source stays as it was, and the output and its selection match the kernel on a clone at the clicked ordinals.
void TestViewportOperators(Fixture &f) {
    auto &r = f.R;
    const auto &meshes = f.Store();
    using D = MeshStore::ElementDomain;
    // The domains of each element type, with their lists in a selection.
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
                meshes.GetSelectedElements(f.Output(mesh), element).ForEach([&](uint32_t handle) { list->push_back(meshes.LiveElementOrdinal(f.Output(mesh), domain, handle)); });
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
            f.P->Do(action::MakeAction(action::mesh::ToSphere{}));
            expect(!r.Context.get<action::Errors>().Messages.empty());
            r.Context.get<action::Errors>().Messages.clear();
            expect(Describe(meshes, f.Output(mesh)) == done.Output);
            ExpectRestores(f, mesh, std::array{start, done}, [&] { expect(r.valid(node) == (f.P->History.Present == done.Node)); });
            state::DirtySet cooks;
            cooks.bind(r);
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
                    for (const auto ordinal : ordinals.Get(element)) list->push_back(meshes.LiveElementAt(clone, domain, ordinal));
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

// A bevel that selects nothing marks its node and forwards its input, a restore clears the mark, and the next cook marks the node again.
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
    r.emplace<InlineSelection>(bevel_entity, std::vector<SelectionBlock>(1u), std::vector<SelectionBlock>(1u), std::vector<SelectionBlock>(1u));
    f.Commit("Select nothing");
    expect(r.any_of<NodeError>(bevel_entity));
    expect(Describe(meshes, f.Output(mesh)) == source);
    f.ValidateReplay();
    f.Do(UpdateOf<&action::mesh::Bevel::Width>(bevel_entity, .2f));
    expect(r.any_of<NodeError>(bevel_entity));
    f.P->Undo();
    expect(!r.any_of<NodeError>(bevel_entity));
    f.P->Redo();
    expect(!r.any_of<NodeError>(bevel_entity));
    f.Do(UpdateOf<&action::mesh::Bevel::Width>(bevel_entity, .3f));
    expect(r.any_of<NodeError>(bevel_entity));
    f.ValidateReplay();
    f.Audit();
}
} // namespace

void TestProcedures() {
    Fixture f{"/tmp/mesheditor-scratch/procedure"};
    boost::ut::test("every operator kind cooks like its kernel") = [&] { TestOperatorKinds(f); };
    boost::ut::test("every attribute graph survives history") = [&] { Survives(f, AttributeGraphs); };
    boost::ut::test("join primitive tables") = [&] { TestJoinPrimitiveTables(f); };
    boost::ut::test("viewport operators append nodes") = [&] { TestViewportOperators(f); };
    boost::ut::test("node error") = [&] { TestNodeError(f); };
}
