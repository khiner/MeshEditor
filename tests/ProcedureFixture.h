#pragma once

#include "MeshProperties.h"
#include "ProjectFixture.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshClone.h"
#include "mesh/Primitives.h"
#include "numeric/MatrixMath.h"
#include "procedure/Procedure.h"

// A project fixture for procedure scenarios, with node authoring, viewport clicks and kernel references on clones of a mesh's source.
struct ProcedureFixture : Fixture {
    void Fails(auto a) {
        P->Do(action::MakeAction(std::move(a)));
        boost::ut::expect(!R.Resources.get<action::Errors>().Messages.empty());
        R.Resources.get<action::Errors>().Messages.clear();
    }
    // Adds a mesh whose procedure displays a generator of the primitive.
    state::Entity AddGenerated(PrimitiveParameters parameters) {
        Do(action::object::AddMeshPrimitive{std::move(parameters), std::make_unique<MeshInstanceCreateInfo>()});
        return R.get<const Instance>(FindActiveEntity(R)).Entity;
    }
    // Adds a mesh whose procedure displays a Source node holding the primitive's record.
    state::Entity AddMesh(PrimitiveParameters parameters) {
        const auto mesh = AddGenerated(std::move(parameters));
        Do(action::nodes::CollapseProcedure{mesh});
        return mesh;
    }
    // Adds a mesh of the primitive whose Source node refers to a record that `prepare` edits before the node's state is kept.
    state::Entity AddPrepared(const PrimitiveParameters &shape, auto &&prepare) {
        const auto source = CreateMesh(R, {.Data = std::visit([](const auto &parameters) { return primitive::CreateMesh(parameters); }, shape), .FlatShaded = true}).StoreId;
        P->Settle();
        prepare(*this, source);
        const auto mesh = ::AddMesh(R, source, MeshInstanceCreateInfo{}).first;
        Commit("Add prepared mesh");
        return mesh;
    }
    // Adds an unlinked node without changing the displayed stage.
    state::Entity Append(state::Entity mesh_entity, uint16_t kind) {
        Do(action::nodes::AddNode{mesh_entity, kind, {}});
        return R.get<const procedure::GeometryProcedure>(mesh_entity).Last;
    }
    // Links each node's first output into the next node's first input and displays the last node.
    void Route(state::Entity mesh_entity, std::initializer_list<state::Entity> nodes) {
        for (const auto *from = nodes.begin(); from + 1 != nodes.end(); ++from) Do(action::nodes::Link{mesh_entity, *from, Out(*from), *(from + 1), 0u});
        Do(action::nodes::SetDisplay{mesh_entity, *(nodes.end() - 1)});
    }
    // The node at the index in the procedure's authoring order.
    state::Entity Node(state::Entity mesh_entity, uint32_t node) const { return R.get<const procedure::CookPlan>(mesh_entity).Nodes[node].Entity; }
    // The node's first output socket.
    uint16_t Out(state::Entity node) const { return procedure::InputCount[R.get<const procedure::ProcedureNode>(node).Kind]; }
    void Render() {
        SubmitViewports(R);
        WaitForRender(R);
    }
    const SceneViewUBO &View() const { return *reinterpret_cast<const SceneViewUBO *>(R.get<const Viewport>(FirstViewport(R)).SceneViewUBO.Contents().data()); }
    // The world position's fraction of the viewport across from its left edge and down from its top edge, as the last frame projects it.
    vec2 ScreenFraction(vec3 world) const {
        const auto clip = View().ViewProj * vec4{world.x, world.y, world.z, 1.f};
        return {.5f + .5f * clip.x / clip.w, .5f - .5f * clip.y / clip.w};
    }
    // Renders a frame and picks the element of the domain nearest the world position.
    std::optional<ElementHit> Pick(vec3 position, Element domain) {
        Render();
        return RunEditElementPick(R, FirstViewport(R), domain, SelectionPixel(View().ViewportSize, ScreenFraction(position)));
    }
    // Clicks the element nearest the world position, adding it to the selection with `toggle`.
    void Click(vec3 position, bool toggle) { Do(action::selection::ApplyEditElementClick{toggle, Pick(position, R.get<const EditMode>(state::Document).Value)}); }
    // The state of the mesh's first Source node.
    procedure::SourceState SourceOf(state::Entity mesh_entity) const {
        for (const auto &node : R.get<const procedure::CookPlan>(mesh_entity).Nodes) {
            if (node.Kind == procedure::KindOf<procedure::Source>) return R.get<const procedure::SourceState>(node.Entity);
        }
        return {};
    }
    // A new record holding the Source node's geometry, with selection state.
    uint32_t CloneSource(const procedure::SourceState &source) {
        auto &meshes = Store();
        const auto id = source.State == store::Hash128{} ? CloneRecord(R, source.StoreId) : meshes.CloneStored(source.State, source.StoreId).value();
        meshes.RebuildSelectionIndex(std::array{id});
        mtl::ComputeChain chain{meshes.BufferContext()};
        meshes.EnsureSelectionState(R, chain, std::array{id});
        chain.Submit();
        return id;
    }
    // Calls `read` with a new record holding the geometry of the mesh's first Source node, which it then releases with a direct commit.
    auto WithSource(state::Entity mesh_entity, auto &&read) {
        const auto id = CloneSource(SourceOf(mesh_entity));
        auto result = read(id);
        Store().Release(id);
        Commit("Release source");
        return result;
    }
    state::Entity Object(state::Entity mesh_entity) const {
        for (const auto [e, instance] : R.view<const Instance>().each())
            if (instance.Entity == mesh_entity) return e;
        return state::Null;
    }
    // The record the mesh draws.
    uint32_t Output(state::Entity mesh_entity) const { return *DrawnStoreId(R, mesh_entity); }
    Shape Shaped(state::Entity mesh_entity) { return Describe(Store(), Output(mesh_entity)); }
    void Run(uint32_t id, std::vector<MeshTopologyTask> tasks) {
        for (auto &task : tasks) task.SourceId = id;
        boost::ut::expect(std::ranges::all_of(ExecuteGeometryTopologyStages(R, tasks), &GeometryTopologyResult::Changed));
    }
    void CheckStep();
};

// A write the fixture does on a node once it is added.
using NodeUpdate = std::function<void(ProcedureFixture &, state::Entity)>;
template<auto... Ms> NodeUpdate Set(action::detail::last_field<Ms...> value) {
    return [value](ProcedureFixture &f, state::Entity node) { f.Do(action::UpdateOf<Ms...>(node, value)); };
}
// The attribute name of a Named Attribute or Store node.
inline NodeUpdate Named(std::string name) {
    return [name = std::move(name)](ProcedureFixture &f, state::Entity node) { f.Do(action::nodes::SetAttributeName{node, name}); };
}

// A procedure graph on a mesh of a primitive, whose Source node is node 0 and whose listed nodes follow it in authoring order.
struct GraphDescription {
    struct Node {
        uint16_t Kind;
        std::vector<NodeUpdate> Updates{};
    };
    // A link from the first output of node `From` into socket `Socket` of node `To`.
    struct Link {
        uint32_t From, To;
        uint16_t Socket;
    };
    PrimitiveParameters Shape;
    std::vector<Node> Nodes;
    std::vector<Link> Links;
    uint32_t Display;
    // Edits the source record before the Source node keeps its state.
    std::function<void(ProcedureFixture &, uint32_t)> Prepare{};
};
struct BuiltGraph {
    state::Entity Mesh;
    // The node entities in description order, starting with the Source node.
    std::vector<state::Entity> Nodes;
};
inline BuiltGraph Build(ProcedureFixture &f, const GraphDescription &graph) {
    const auto mesh = graph.Prepare ? f.AddPrepared(graph.Shape, graph.Prepare) : f.AddMesh(graph.Shape);
    BuiltGraph built{mesh, {f.Node(mesh, 0u)}};
    for (const auto &[kind, updates] : graph.Nodes) {
        built.Nodes.push_back(f.Append(mesh, kind));
        for (const auto &update : updates) update(f, built.Nodes.back());
    }
    for (const auto [from, to, socket] : graph.Links) f.Do(action::nodes::Link{mesh, built.Nodes[from], f.Out(built.Nodes[from]), built.Nodes[to], socket});
    f.Do(action::nodes::SetDisplay{mesh, built.Nodes[graph.Display]});
    return built;
}

// The record and the selected elements a kernel edit leaves on a clone of `id`, where `edit` runs the kernel on the clone and returns the elements it selects.
struct Kernel {
    Shape Output, Selected;
};
Kernel Reference(ProcedureFixture &f, uint32_t id, auto &&edit) {
    auto &meshes = f.Store();
    meshes.PlanClone(Mesh{meshes, id});
    meshes.CommitReserves();
    CloneCopies copies;
    const auto clone = meshes.CloneMeshes(copies, std::array{id}, true).front();
    mtl::ComputeChain chain{meshes.BufferContext()};
    copies.Encode(chain, GetMeshPipelines(f.R));
    chain.Submit();
    meshes.RebuildSelectionIndex(std::array{clone});
    f.Audit();
    const GeometrySelection selected = edit(clone);
    Kernel kernel{Describe(meshes, clone), Describe(meshes, clone, selected)};
    meshes.Release(clone);
    f.Commit("Release reference");
    return kernel;
}
// The kernel edit on a record holding the geometry of the mesh's Source node.
Kernel SourceReference(ProcedureFixture &f, state::Entity mesh_entity, auto &&edit) {
    return f.WithSource(mesh_entity, [&](uint32_t id) {
        const GeometrySelection selected = edit(id);
        return Kernel{Describe(f.Store(), id), Describe(f.Store(), id, selected)};
    });
}

inline bool Facing(const Mesh &mesh, he::FH face, vec3 eye) { return face && Dot(test::Normal(mesh, face), eye - test::Centroid(mesh, face)) > 0.f; }
inline std::vector<vec3> FacingFaceCenters(const Mesh &mesh, vec3 eye) {
    std::vector<vec3> centers;
    for (const auto face : mesh.faces())
        if (Facing(mesh, face, eye)) centers.push_back(test::Centroid(mesh, face));
    return centers;
}

// The volume the record the mesh draws encloses, or -1 when its surface is open.
inline double OutputVolume(ProcedureFixture &f, state::Entity mesh) { return test::EnclosedVolume(Mesh{f.Store(), f.Output(mesh)}).value_or(-1.0); }

// Runs the topology task on the clone, publishes its emitted selection and returns that selection.
inline GeometrySelection PublishedKernel(ProcedureFixture &f, uint32_t clone, MeshTopologyTask task) {
    task.SourceId = clone;
    PublishStoreSelection(f.Store(), task);
    f.Run(clone, {task});
    return EditorGeometrySelection(f.Store(), clone);
}

// Runs the viewport gesture rows, the recorded operator rows and the mesh edit scenarios.
void TestOperators(ProcedureFixture &f, Fixture &other);
