#pragma once

#include "MeshProperties.h"
#include "ProjectFixture.h"
#include "graph/NodeEditor.h"
#include "graph/OpNodes.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshClone.h"
#include "mesh/Primitives.h"
#include "numeric/MatrixMath.h"
#include <imgui.h>

using Matrix33 = std::array<float, 9>;
// A literal of an op input, of its own type.
using OpLiteral = std::variant<float, int32_t, bool, vec2, vec3, vec4, Matrix33>;
struct GraphFixture;

// An ImGui context of its own, with a Nodes and a Properties panel.
struct PanelContext {
    ImGuiContext *Context{ImGui::CreateContext()};
    NodesPanel Nodes{{"Nodes"}};
    Panel Properties{"Properties"};
    PanelContext() {
        ImGui::SetCurrentContext(Context);
        auto &io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.DisplaySize = {1400, 900};
        io.Fonts->Build();
    }
    ~PanelContext() { ImGui::DestroyContext(Context); }
};
// The geometry the Nodes and Properties panels draw in one frame.
inline std::vector<std::byte> DrawPanels(Fixture &f, PanelContext &panels, bool interactive = false) {
    ImGui::SetCurrentContext(panels.Context);
    ImGui::NewFrame();
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({950, 850});
    graph::DrawNodesPanel(f.R, panels.Nodes, interactive);
    ImGui::SetNextWindowPos({950, 0});
    ImGui::SetNextWindowSize({450, 850});
    graph::DrawPropertiesPanel(f.R, panels.Nodes, panels.Properties, interactive);
    ImGui::Render();
    std::vector<std::byte> drawn;
    for (const auto *list : ImGui::GetDrawData()->CmdLists) {
        drawn.append_range(std::as_bytes(std::span{list->VtxBuffer.Data, size_t(list->VtxBuffer.Size)}));
        drawn.append_range(std::as_bytes(std::span{list->IdxBuffer.Data, size_t(list->IdxBuffer.Size)}));
    }
    boost::ut::expect(!drawn.empty());
    return drawn;
}
// A write the fixture does on a node once it is added.
using NodeUpdate = std::function<void(GraphFixture &, state::Entity)>;

// A project fixture for graph scenarios, with node authoring, viewport clicks and kernel references on clones of a mesh's source.
struct GraphFixture : Fixture {
    // The panels the step checks draw, whose Nodes panel keeps the page it shows across steps.
    std::unique_ptr<PanelContext> Panels{std::make_unique<PanelContext>()};
    // Whether the fresh-cook and canvas checks run at every step, or once at the end of a script.
    bool EveryStep{};
    void Begin(const std::filesystem::path &dir) {
        Fixture::Begin(dir);
        Panels->Nodes = NodesPanel{{"Nodes"}};
        EveryStep = false;
    }
    void Fails(auto a) {
        project::Do(*P, action::MakeAction(std::move(a)));
        boost::ut::expect(!R.Resources.get<action::Errors>().Messages.empty());
        R.Resources.get<action::Errors>().Messages.clear();
    }
    // Adds a mesh whose graph displays a generator of the primitive.
    state::Entity AddGenerated(PrimitiveParameters parameters) {
        Do(action::object::AddMeshPrimitive{std::move(parameters), std::make_unique<MeshObjectCreateInfo>()});
        return R.get<const ObjectGeometry>(FindActiveEntity(R)).Entity;
    }
    // Adds a mesh whose graph displays a Source node holding the primitive's record.
    state::Entity AddMesh(PrimitiveParameters parameters) {
        const auto mesh = AddGenerated(std::move(parameters));
        Do(action::nodes::FreezeGraph{mesh});
        return mesh;
    }
    // Adds a mesh of the primitive whose Source node refers to a record that `prepare` edits before the node's state is kept.
    state::Entity AddPrepared(const PrimitiveParameters &shape, auto &&prepare) {
        const auto source = CreateMesh(R, primitive::CreateMesh(shape)).StoreId;
        P->Settle();
        prepare(*this, source);
        const auto mesh = ::AddMesh(R, source, MeshObjectCreateInfo{}).first;
        Commit("Add prepared mesh");
        return mesh;
    }
    // Adds an unlinked node of the kind, or an op node evaluating the MaterialX node, to a graph without changing the displayed stage, and applies the updates to it.
    state::Entity Append(state::Entity graph, std::variant<uint16_t, std::string_view> kind, const std::vector<NodeUpdate> &updates = {}) {
        const auto spec = std::visit(overloaded{[](uint16_t k) -> action::nodes::NodeSpec { return action::nodes::KindNode{k}; }, [](std::string_view op) -> action::nodes::NodeSpec { return action::nodes::OpNode{std::string{op}}; }}, kind);
        Do(action::nodes::AddNode{graph, spec});
        const auto node = graph::NodesOf(R, graph).back();
        for (const auto &update : updates) update(*this, node);
        return node;
    }
    // The op node's input socket of the name, or else its output socket of the name.
    uint16_t OpSocket(state::Entity node, std::string_view name) const {
        const auto sockets = graph::DefinitionOf(R.get<const graph::Op>(node).Node)->Sockets;
        return uint16_t(std::ranges::find(sockets, name, [](const graph::Socket &socket) { return std::string_view{socket.Name}; }) - sockets.begin());
    }
    // Links each node's first output into the next node's first input and displays the last node.
    void Route(state::Entity mesh_entity, std::initializer_list<state::Entity> nodes) {
        for (const auto *from = nodes.begin(); from + 1 != nodes.end(); ++from) Do(action::nodes::Link{mesh_entity, *from, Out(*from), *(from + 1), 0u});
        Do(action::nodes::SetDisplay{mesh_entity, *(nodes.end() - 1)});
    }
    // The node at the index in the graph's authoring order.
    state::Entity Node(state::Entity mesh_entity, uint32_t node) const { return R.get<const graph::CookPlan>(mesh_entity).Nodes[node].Entity; }
    // The node's first output socket.
    uint16_t Out(state::Entity node) const { return graph::InputCountOf(R, node); }
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
    graph::SourceState SourceOf(state::Entity mesh_entity) const {
        for (const auto &node : R.get<const graph::CookPlan>(mesh_entity).Nodes) {
            if (node.Kind == graph::KindOf<graph::Source>) return R.get<const graph::SourceState>(node.Entity);
        }
        return {};
    }
    // A new record holding the Source node's geometry, with selection state.
    uint32_t CloneSource(const graph::SourceState &source) {
        auto &meshes = Store();
        MeshStore::StateReaders readers;
        const auto id = source.State == store::Hash128{} ? CloneRecord(R, source.StoreId) : meshes.CloneStored(source.State, source.StoreId, readers).value();
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
        for (const auto [e, geometry] : R.view<const ObjectGeometry>().each())
            if (geometry.Entity == mesh_entity) return e;
        return state::Null;
    }
    // The fault the cook of the node's mesh holds for the node.
    std::optional<graph::NodeFault> Fault(state::Entity node) const {
        const auto *error = graph::CookedOf<graph::NodeError>(R, std::array{node});
        return error ? std::optional{error->Fault} : std::nullopt;
    }
    // The record the mesh draws.
    uint32_t Output(state::Entity mesh_entity) const { return *DrawnStoreId(R, mesh_entity); }
    Shape Shaped(state::Entity mesh_entity) { return Describe(Store(), Output(mesh_entity)); }
    void Run(uint32_t id, std::vector<MeshTopologyTask> tasks) {
        for (auto &task : tasks) task.SourceId = id;
        boost::ut::expect(std::ranges::all_of(ExecuteGeometryTopologyStages(R, tasks), &GeometryTopologyResult::Changed));
    }
    void CheckStep();
    void CheckFreshCooks();
    void CheckCanvas();
    // Runs the fresh-cook and canvas checks on the state a script ends with.
    void CheckEnd() {
        CheckFreshCooks();
        CheckCanvas();
    }
};

// The action that sets the op node's input of the name to the value.
inline action::nodes::SetOpInput OpInputOf(state::Entity node, std::string_view input, action::nodes::OpInputValue value) {
    return {node, std::string{input}, std::make_unique<action::nodes::OpInputValue>(std::move(value))};
}
template<auto... Ms> NodeUpdate Set(action::detail::last_field<Ms...> value) {
    return [value](GraphFixture &f, state::Entity node) { f.Do(action::UpdateOf<Ms...>(node, value)); };
}
// The attribute name of a Store node.
inline NodeUpdate Named(std::string name) {
    return [name = std::move(name)](GraphFixture &f, state::Entity node) { f.Do(action::nodes::SetAttributeName{node, name}); };
}
inline void SetLiteral(GraphFixture &f, state::Entity node, std::string_view input, const OpLiteral &value) {
    const auto word = [](float v) { return std::bit_cast<uint32_t>(v); };
    const auto [type, words] = std::visit(
        overloaded{
            [&](float v) { return std::pair{mx::Type::Float, std::vector{word(v)}}; },
            [](int32_t v) { return std::pair{mx::Type::Integer, std::vector{std::bit_cast<uint32_t>(v)}}; },
            [](bool v) { return std::pair{mx::Type::Boolean, std::vector{uint32_t(v)}}; },
            [&](vec2 v) { return std::pair{mx::Type::Vector2, std::vector{word(v.x), word(v.y)}}; },
            [&](vec3 v) { return std::pair{mx::Type::Vector3, std::vector{word(v.x), word(v.y), word(v.z)}}; },
            [&](vec4 v) { return std::pair{mx::Type::Vector4, std::vector{word(v.x), word(v.y), word(v.z), word(v.w)}}; },
            [&](const Matrix33 &m) { return std::pair{mx::Type::Matrix33, m | std::views::transform(word) | std::ranges::to<std::vector>()}; },
        },
        value
    );
    f.Do(OpInputOf(node, input, action::nodes::OpLiteral{type, words}));
}
// A link from the first output of `from` into the op node's input of the name.
inline NodeUpdate From(std::string_view input, state::Entity from) {
    return [input, from](GraphFixture &f, state::Entity node) { f.Do(action::nodes::Link{graph::GraphOf(f.R, node), from, f.Out(from), node, f.OpSocket(node, input)}); };
}
inline NodeUpdate Literal(std::string_view input, OpLiteral value) {
    return [input, value](GraphFixture &f, state::Entity node) { SetLiteral(f, node, input, value); };
}
// The text of an op node's string input of the name.
inline NodeUpdate Text(std::string_view input, std::string text) {
    return [input, text = std::move(text)](GraphFixture &f, state::Entity node) { f.Do(OpInputOf(node, input, text)); };
}

// A geometry graph on a mesh of a primitive or a source, whose Source node is node 0 and whose listed nodes follow it in authoring order.
struct GraphDescription {
    // A node of a kind, or an op node of a MaterialX node.
    struct Node {
        std::variant<uint16_t, std::string_view> Kind;
        std::vector<NodeUpdate> Updates{};
    };
    // A link from the first output of node `From` into socket `Socket` of node `To`, or into the op input named `Input`.
    struct Link {
        uint32_t From, To;
        uint16_t Socket;
        std::string_view Input{};
    };
    std::variant<PrimitiveParameters, MeshSource> Shape;
    std::vector<Node> Nodes;
    std::vector<Link> Links;
    uint32_t Display;
    // Edits the primitive's record before the Source node keeps its state.
    std::function<void(GraphFixture &, uint32_t)> Prepare{};
};
struct BuiltGraph {
    state::Entity Mesh;
    // The node entities in description order, starting with the Source node.
    std::vector<state::Entity> Nodes;
};
inline BuiltGraph Build(GraphFixture &f, const GraphDescription &graph) {
    const auto *primitive = std::get_if<PrimitiveParameters>(&graph.Shape);
    const auto mesh = !primitive ? f.AddSourced(std::get<MeshSource>(graph.Shape)).first : graph.Prepare ? f.AddPrepared(*primitive, graph.Prepare) :
                                                                                                           f.AddMesh(*primitive);
    BuiltGraph built{mesh, {f.Node(mesh, 0u)}};
    for (const auto &[kind, updates] : graph.Nodes) built.Nodes.push_back(f.Append(mesh, kind, updates));
    for (const auto [from, to, socket, input] : graph.Links) f.Do(action::nodes::Link{mesh, built.Nodes[from], f.Out(built.Nodes[from]), built.Nodes[to], input.empty() ? socket : f.OpSocket(built.Nodes[to], input)});
    f.Do(action::nodes::SetDisplay{mesh, built.Nodes[graph.Display]});
    return built;
}

// The record and the selected elements a kernel edit leaves on a clone of `id`, where `edit` runs the kernel on the clone and returns the elements it selects.
struct Kernel {
    Shape Output, Selected;
};
Kernel Reference(GraphFixture &f, uint32_t id, auto &&edit) {
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
// The shape the edit leaves on a record holding the mesh's Source geometry.
Shape EditedSource(GraphFixture &f, state::Entity mesh, auto &&edit) {
    return f.WithSource(mesh, [&](uint32_t id) {
        edit(id);
        return Describe(f.Store(), id);
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
inline double OutputVolume(GraphFixture &f, state::Entity mesh) { return test::EnclosedVolume(Mesh{f.Store(), f.Output(mesh)}).value_or(-1.0); }

// Runs the topology task on the clone, publishes its emitted selection and returns that selection.
inline GeometrySelection PublishedKernel(GraphFixture &f, uint32_t clone, MeshTopologyTask task) {
    task.SourceId = clone;
    PublishStoreSelection(f.Store(), task);
    f.Run(clone, {task});
    return EditorGeometrySelection(f.Store(), clone);
}

// Runs the viewport gesture rows, the recorded operator rows and the mesh edit scenarios.
void TestOperators(GraphFixture &f, Fixture &other);
