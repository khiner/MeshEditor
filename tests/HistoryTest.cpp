#include "GraphFixture.h"
#include "audio/AudioOutputConfig.h"
#include "graph/BuiltinSubgraphs.h"
#include "materialx/Library.h"
#include "render/MaterialComponents.h"
#include "scene/Instance.h"
#include "scene/SceneGraph.h"
#include "viewport/FrameState.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportScene.h"

#include <imgui_internal.h>
#include <random>

using boost::ut::expect;

namespace {
struct HistoryFixture : GraphFixture {
    // The fixture's images size the first viewport's panel, so a first viewport's panel is left out under rendering.
    state::Entity AddViewport() {
        const auto before = R.view<const ViewportScene>() | std::ranges::to<std::vector>();
        Do(action::view::AddViewport{});
        const auto viewport = *std::ranges::find_if(R.view<const ViewportScene>(), [&](state::Entity e) { return !std::ranges::contains(before, e); });
        const bool sized = ::Render && viewport == FirstViewport(R);
        [&]<typename... Cs>(TypeList<Cs...>) { (expect((std::same_as<Cs, ViewportPanel> && sized) || R.get<const Cs>(viewport) == ViewDefault<Cs>()), ...); }(ViewValues{});
        return viewport;
    }
    void MoveCamera(state::Entity viewport, vec3 target) {
        Do(action::view::SetViewCameraTarget{viewport, target});
        expect(Camera(viewport) != ViewCameraState{});
    }
    ViewCameraState Camera(state::Entity viewport) const { return R.get<const ViewCameraState>(viewport); }
};

// The walk vocabulary, in groups: objects and views, scene and document items, nodes, gestures and the node editor's commands.
enum class WalkStep {
    AddPrimitive,
    AddEmpty,
    AddLight,
    AddViewport,
    RemoveViewport,
    ChangeView,
    SelectObject,
    ExtendActive,
    ParentToActive,
    MoveActive,
    Delete,
    Duplicate,
    LookThroughCamera,
    SetVisible,
    Instantiate,
    MakeInstanceUnique,
    LoadRig,
    SetFrame,
    AddMaterial,
    ReadTexcoords,
    AssignReference,
    DeleteItems,
    CollapseItems,
    ExpandItems,
    ExposePort,
    InstancePortValue,
    AddNode,
    Link,
    Unlink,
    InsertNode,
    DeleteNodes,
    DuplicateNodes,
    CollapseNodes,
    ExpandNodes,
    MakeUnique,
    ExposeSocket,
    NumericValue,
    SetDisplay,
    SetBypass,
    EditParameter,
    SetOpInput,
    ChangeOp,
    MaterialPage,
    MoveNodes,
    FreezeGraph,
    EnterPage,
    StageParameter,
    ExtrudeDrag,
    ViewportOperator,
    EditSelection,
    MakeLinks,
    AddSearch,
    LinkingAdd,
    Count,
};
constexpr std::array<std::string_view, size_t(WalkStep::Count)> WalkStepNames{
    "add primitive", "add empty", "add light", "add viewport", "remove viewport", "change view", "select object", "extend active", "parent to active", "move active", "delete", "duplicate", "look through camera", "set visible", "instantiate", "make instance unique", "load rig", "set frame",
    "add material", "read texcoords", "assign reference", "delete items", "collapse items", "expand items", "expose port", "instance port value",
    "add node", "link", "unlink", "insert node", "delete nodes", "duplicate nodes", "collapse nodes", "expand nodes", "make unique", "expose socket", "port value", "set display", "set bypass", "edit parameter", "set op input", "change op", "material page", "move nodes", "freeze graph", "enter page",
    "stage parameter", "extrude drag", "viewport operator", "edit selection", "make links", "add search", "linking add"
};

// The rules a walk reaches at least once, each counted at a step that exercises it.
constexpr std::array WalkRules{"index reuse", "fresh identity", "kind change", "name collision", "loop in parents", "viewport removal", "look through", "rejected", "cancelled"};

// The MaterialX nodes a geometry graph offers, in name order.
const std::vector<std::string_view> &GeometryOps() {
    static const auto nodes = [] {
        std::vector<std::string_view> found;
        for (const auto &[node, declarations] : mx::Libraries().DeclarationsByNode) {
            const auto *definition = graph::DefinitionOf(node);
            if (definition && definition->Domains && definition->Sockets.size() > definition->Inputs) found.push_back(node);
        }
        return found;
    }();
    return nodes;
}

// The geometry ops that read texture coordinates: texcoord, and each op with an input that defaults to them.
const std::vector<std::string_view> &TexcoordOps() {
    static const auto nodes = [] {
        const auto &lib = mx::Libraries();
        const auto defaults_texcoord = [&](const mx::Port &port) { return port.DefaultGeomProp && lib.Declarations[lib.GeomProps[*port.DefaultGeomProp].Declaration].Node == "texcoord"; };
        std::vector<std::string_view> found;
        for (const auto node : GeometryOps()) {
            const auto &declarations = lib.DeclarationsByNode.find(node)->second;
            if (node == "texcoord" || std::ranges::any_of(declarations, [&](uint32_t d) { return std::ranges::any_of(lib.Declarations[d].Inputs, defaults_texcoord); })) found.push_back(node);
        }
        return found;
    }();
    return nodes;
}

// A socket of a node on a graph's page by the id a link names it with.
struct PageSocket {
    state::Entity Node;
    uint16_t Id;
    graph::Socket Socket;
};
// The sockets of the node facing the direction on the graph's page: a subgraph node's ports, the subgraph's own ports at its boundary, or another node's sockets.
std::vector<PageSocket> SocketsFacing(const state::Registry &r, state::Entity graph, state::Entity node, graph::SocketDirection direction) {
    std::vector<PageSocket> sockets;
    if (node == graph || r.all_of<graph::SubgraphNode>(node)) {
        // A boundary's input ports pass values into the graph, and its output ports take them.
        const bool boundary = node == graph;
        for (const auto &port : graph::InterfaceOf(r, node)->Ports) {
            if ((port.Direction == direction) != boundary) sockets.push_back({node, port.Id, graph::PortSocket(r, node, port, direction)});
        }
        return sockets;
    }
    const auto all = graph::SocketsOf(r, node);
    for (uint16_t s = 0; s < all.size(); ++s) {
        if (all[s].Direction == direction) sockets.push_back({node, s, all[s]});
    }
    return sockets;
}
// Each linked input of the graph's nodes and of its boundary, as the node, its socket and the link's index among the socket's links.
std::vector<std::tuple<state::Entity, uint16_t, uint16_t>> LinkedInputs(const state::Registry &r, state::Entity graph) {
    std::vector<std::tuple<state::Entity, uint16_t, uint16_t>> linked;
    std::map<std::pair<state::Entity, uint16_t>, uint16_t> counts;
    graph::ForEachGraphInput(r, graph, [&](state::Entity node, uint16_t slot, graph::SocketLink link) {
        const auto socket = node == graph || r.all_of<graph::SubgraphNode>(node) ? slot : graph::SlotSocketOf(r, node, slot);
        const auto index = counts[{node, socket}]++;
        if (link.FromNode != state::Null) linked.emplace_back(node, socket, index);
    });
    return linked;
}
// The nodes holding C of each graph, in graph order.
template<typename C> std::vector<state::Entity> NodesWith(const state::Registry &r, const std::vector<state::Entity> &graphs) {
    std::vector<state::Entity> found;
    for (const auto graph : graphs) {
        for (const auto node : graph::NodesOf(r, graph))
            if (r.all_of<C>(node)) found.push_back(node);
    }
    return found;
}
// The screen point of a position in cells on the page the Nodes panel last drew, at the canvas's first offset and zoom.
ImVec2 CanvasPoint(numeric::vec2 cells) {
    const auto *canvas = *std::ranges::find_if(GImGui->Windows, [](const ImGuiWindow *window) { return std::string_view{window->Name}.starts_with("Nodes/Canvas"); });
    const float cell = ImGui::GetFrameHeight();
    return canvas->Pos + ImVec2{(cells.x + .5f) * cell, (cells.y + .5f) * cell};
}

// Takes `steps` steps drawn from the walk vocabulary, or from undo, redo and navigation, checking after each action and navigation that:
// * An index freed along the present's path returns only at a higher generation.
// * Sibling names are unique, and an empty's taken name takes the suffix past every live one.
// * An identity carries only its kind's components, object or viewport.
// * A viewport added again after its add is undone takes the same identity and starts from defaults.
// * No node is its own ancestor, and parenting an object to its descendant reports the loop and leaves the hierarchy unchanged.
// * A removed viewport is invalid, every other viewport keeps its camera, and a view change records no node.
// * A viewport keeps looking through a camera whose creation is undone, and resolves it again on redo.
// * An action that reports an error, and a cancelled gesture, leave the document and the history unchanged.
// Each node action draws its graph, nodes, sockets and values from the live document.
// The step checks of the fixture judge every state the walk reaches.
void RandomHistoryWalk(HistoryFixture &f, uint32_t seed, uint32_t steps) {
    f.EveryStep = true;
    using action::nodes::AddNode, action::nodes::AssignReference, action::nodes::AutoLayout, action::nodes::ChangeOp, action::nodes::ClearPortValue, action::nodes::FreezeGraph, action::nodes::UserSubgraphNode, action::nodes::DeleteItems, action::nodes::DeleteNodes, action::nodes::DuplicateNodes;
    using action::nodes::EditSelection, action::nodes::ExposeSocket, action::nodes::FinishSelection, action::nodes::Collapse, action::nodes::ExposePort, action::nodes::InsertNode, action::nodes::KindNode, action::nodes::Link, action::nodes::BuiltinSubgraphNode, action::nodes::MakeUnique, action::nodes::MoveNodes;
    using action::nodes::OpNode, action::nodes::SelectItems, action::nodes::SetBypass, action::nodes::SetDisplay, action::nodes::SetOpInput, action::nodes::SetOpOutputType, action::nodes::SetPortValue, action::nodes::Expand, action::nodes::Unlink;
    using graph::SocketDirection;
    auto &r = f.R;
    std::mt19937 rng{seed};
    const auto below = [&](size_t n) { return std::uniform_int_distribution<size_t>{0u, n - 1u}(rng); };
    const auto coordinate = [&] { return std::uniform_real_distribution<float>{-4.f, 4.f}(rng); };
    uint32_t step = 0;
    const auto at = [&] { return std::format("seed {} step {}", seed, step); };
    std::map<std::string_view, uint32_t> reached;
    // Per node, its live entities and the lowest generation each index freed along its path may return at, or zero.
    std::map<uint32_t, std::vector<state::Entity>> live;
    std::map<uint32_t, std::vector<uint32_t>> fresh_generation;
    std::map<state::Entity, bool> viewport_kind;
    const auto check = [&] {
        const auto node = f.P->History.Present.value();
        std::vector<state::Entity> alive;
        for (uint32_t i = 0; i < r.EntityCapacity(); ++i) {
            if (const auto e = r.EntityAt(i); e != state::Null) alive.push_back(e);
        }
        if (!fresh_generation.contains(node)) {
            std::vector<uint32_t> generations;
            // A node editor command can record several nodes, of which the walk checks the last, so the nearest checked ancestor stands in for the parent.
            auto parent = f.P->History.Nodes[node].Parent;
            while (parent && !fresh_generation.contains(*parent)) parent = f.P->History.Nodes[*parent].Parent;
            if (parent) {
                generations = fresh_generation.at(*parent);
                for (const auto e : live.at(*parent)) {
                    if (r.valid(e)) continue;
                    generations.resize(std::max<size_t>(generations.size(), state::Index(e) + 1u));
                    generations[state::Index(e)] = std::max(generations[state::Index(e)], state::Generation(e) + 1u);
                }
            }
            fresh_generation.emplace(node, std::move(generations));
            live.emplace(node, alive);
        }
        const auto &generations = fresh_generation.at(node);
        for (const auto e : alive) {
            if (state::Index(e) < generations.size() && generations[state::Index(e)] != 0u) {
                expect(state::Generation(e) >= generations[state::Index(e)]) << at() << "index" << state::Index(e);
                ++reached["index reuse"];
            }
            const bool object = r.all_of<ObjectKind>(e), viewport = r.any_of<ViewportScene, ViewCameraState, Viewport>(e);
            expect(!(object && viewport)) << at() << "kinds of" << state::Index(e);
            if (!object && !viewport) continue;
            if (const auto [it, added] = viewport_kind.try_emplace(e, viewport); !added && it->second != viewport) {
                ++reached["kind change"];
                it->second = viewport;
            }
        }
        std::set<std::pair<state::Entity, std::string_view>> names;
        for (const auto [e, name] : r.view<const Name>().each()) expect(names.emplace(ParentOrNull(r, e), name.Value).second) << at() << "name" << name.Value;
        for (const auto e : r.view<const SceneParent>()) {
            uint32_t depth = 0;
            for (auto ancestor = e; ancestor != state::Null && depth <= r.EntityCapacity(); ancestor = ParentOrNull(r, ancestor)) ++depth;
            expect(depth <= r.EntityCapacity()) << at() << "parent cycle at" << state::Index(e);
        }
    };
    f.Step = [record = std::move(f.Step), &check](bool replays) {
        record(replays);
        check();
    };
    check();
    auto &messages = r.Resources.get<action::Errors>().Messages;
    // Captures the document and the history, and returns a check that expects them as captured.
    const auto unchanged_since = [&] {
        return [&, document = f.P->History.MaterializeLive(), present = f.P->History.Present, nodes = f.P->History.Nodes.size()] {
            expect(f.P->History.MaterializeLive() == document && f.P->History.Present == present && f.P->History.Nodes.size() == nodes && !f.P->HasStaged()) << at() << "a rejected step changed the document";
        };
    };
    // Ends a rejected action or a cancelled gesture, which leaves the document and the history unchanged, counts the rule and returns false.
    // The refused step's inputs stay in the workspace, which the present's record follows.
    const auto refused = [&](const auto &unchanged, std::string_view rule) {
        messages.clear();
        unchanged();
        ++reached[rule];
        f.Step(true);
        return false;
    };
    // Does the action and returns whether it succeeded, where a rejected action leaves the document and the history unchanged.
    const auto attempt = [&](auto a) {
        const auto unchanged = unchanged_since();
        project::Do(*f.P, action::MakeAction(std::move(a)));
        if (!messages.empty()) return refused(unchanged, "rejected");
        f.Audit();
        f.Step(true);
        return true;
    };
    // Stages each action `steps` passes to its argument as one gesture's step, then commits the gesture or cancels it.
    // A rejected step cancels the gesture, and a cancelled gesture leaves the document and the history unchanged.
    const auto gesture = [&](bool commit, auto &&steps) {
        const auto unchanged = unchanged_since();
        bool rejected = false;
        steps([&](auto a) {
            if (rejected) return;
            action::Emit(action::Phase::Stage, std::move(a));
            f.P->Frame();
            rejected = !messages.empty();
        });
        if (rejected || !commit) {
            action::Cancel();
            f.P->Frame();
            return refused(unchanged, rejected ? "rejected" : "cancelled");
        }
        action::Commit();
        f.P->Frame();
        f.Audit();
        f.Step(true);
        return true;
    };
    const auto pick = [&](const auto &entities) { return entities.empty() ? state::Null : entities[below(entities.size())]; };
    const auto objects = [&] { return SortedEntities(r.view<const ObjectKind>()); };
    const auto viewports = [&] { return SortedEntities(r.view<const ViewportScene>()); };
    // The meshes whose graphs the walk edits, and the graphs it edits: theirs and the user subgraphs'.
    const auto meshes = [&] { return SortedEntities(r.view<const graph::GeometryGraph>()); };
    // Selects a random object drawing a mesh the walk edits, and returns the mesh.
    const auto select_mesh = [&] {
        const auto edited = meshes();
        std::vector<state::Entity> drawn;
        for (const auto [e, geometry] : r.view<const ObjectGeometry>().each())
            if (std::ranges::contains(edited, geometry.Entity)) drawn.push_back(e);
        std::ranges::sort(drawn);
        const auto object = pick(drawn);
        if (object == state::Null || !attempt(action::selection::Select{object})) return state::Null;
        return r.get<const ObjectGeometry>(object).Entity;
    };
    const auto subgraphs = [&] { return SortedEntities(r.view<const graph::Subgraph, const graph::GraphNodes>()); };
    const auto scene_subgraphs = [&] {
        auto found = SortedEntities(r.view<const graph::SubgraphItems>());
        std::erase_if(found, [&](state::Entity e) { return !IsSceneSubgraph(r, e); });
        return found;
    };
    const auto graphs = [&] {
        auto found = meshes();
        found.append_range(subgraphs());
        return found;
    };
    const auto materials = [&] { return SortedEntities(r.view<const graph::MaterialGraph>()); };
    const auto nodes_of = [&](state::Entity graph) { return graph == state::Null ? std::vector<state::Entity>{} : graph::NodesOf(r, graph); };
    const auto uses = [&] { return NodesWith<graph::SubgraphNode>(r, graphs()); };
    const auto some = [&](std::vector<state::Entity> entities, size_t most) {
        std::ranges::shuffle(entities, rng);
        entities.resize(std::min(entities.size(), 1u + below(most)));
        return entities;
    };
    // A random scene subgraph, which is a new one of some items of the frame's scene when the document has none.
    const auto some_scene_subgraph = [&] {
        if (scene_subgraphs().empty()) {
            const auto items = graph::GraphItems(r, FrameScene(r));
            if (items.empty() || !attempt(Collapse{FrameScene(r), some(items, 2u)})) return state::Null;
        }
        return pick(scene_subgraphs());
    };
    // Exposes the position of a random object the scene subgraph holds as a port.
    const auto expose_position = [&](state::Entity subgraph) {
        auto items = graph::GraphItems(r, subgraph);
        std::erase_if(items, [&](state::Entity e) { return !r.all_of<Transform>(e); });
        return !items.empty() && attempt(ExposePort{subgraph, graph::PromotedField{pick(items), state::Key<Transform>(), uint16_t(offsetof(Transform, P))}, std::make_unique<std::string>("Position")});
    };
    // A random object with an Instance, placed in a scene, which is a new one when the document has none.
    const auto some_instance = [&] {
        auto instances = SortedEntities(r.view<const Instance>());
        std::erase_if(instances, [&](state::Entity e) { return r.all_of<CopyOf>(e) || !RootMembership(r, e) || ParentOrNull(r, e) != state::Null; });
        if (!instances.empty()) return pick(instances);
        const auto subgraph = some_scene_subgraph();
        return subgraph != state::Null && attempt(action::object::Instantiate{subgraph}) ? FindActiveEntity(r) : state::Null;
    };
    // Each output of `from` and input of a node of `tos` that link on the graph's page.
    const auto linkable = [&](state::Entity graph, state::Entity from, std::span<const state::Entity> tos) {
        std::vector<std::pair<PageSocket, PageSocket>> pairs;
        for (const auto &output : SocketsFacing(r, graph, from, SocketDirection::Out)) {
            for (const auto to : tos) {
                for (const auto &input : SocketsFacing(r, graph, to, SocketDirection::In))
                    if (graph::Links(output.Socket, input.Socket)) pairs.emplace_back(output, input);
            }
        }
        return pairs;
    };
    // Links a random one of the output and input pairs, if there are any.
    const auto link_any = [&](state::Entity graph, const std::vector<std::pair<PageSocket, PageSocket>> &pairs) {
        if (pairs.empty()) return false;
        const auto &[from, to] = pairs[below(pairs.size())];
        return attempt(Link{graph, from.Node, from.Id, to.Node, to.Id});
    };
    // Links a random output into an input it links with on the graph's page, one of them the node's when given.
    const auto link_in = [&](state::Entity graph, state::Entity node) {
        if (graph == state::Null) return false;
        auto nodes = nodes_of(graph);
        if (r.all_of<graph::Subgraph>(graph)) nodes.push_back(graph);
        std::vector<std::pair<PageSocket, PageSocket>> pairs;
        for (const auto from : nodes) {
            const auto tos = nodes | std::views::filter([&](state::Entity to) { return to != from && (node == state::Null || from == node || to == node); }) | std::ranges::to<std::vector>();
            pairs.append_range(linkable(graph, from, tos));
        }
        return link_any(graph, pairs);
    };
    // Calls `use` with an update of a random field of the node's parameters, if its kind has any.
    const auto parameter_update = [&](state::Entity node, auto &&use) {
        graph::VisitKind(r.get<const graph::GraphNode>(node).Kind, [&]<typename K> {
            if constexpr (!std::is_empty_v<K> && action::Updatable<K>) {
                const auto &value = r.get<const K>(node);
                size_t count = 0;
                field::ForEach(value, [&](const auto &, auto) { ++count; });
                const auto chosen = below(count);
                field::ForEach(value, [&]<size_t I>(const auto &member, std::integral_constant<size_t, I>) {
                    using T = std::remove_cvref_t<decltype(member)>;
                    const auto offset = uint16_t(reinterpret_cast<const std::byte *>(&member) - reinterpret_cast<const std::byte *>(&value));
                    const auto update = [&](auto v) { use(action::Update<decltype(v)>{node, state::Key<K>(), offset, v}); };
                    if (I != chosen) return;
                    if constexpr (std::same_as<T, bool>) {
                        update(!member);
                    } else if constexpr (std::is_enum_v<T>) {
                        if constexpr (field::EnumCount<T> > 0u) update(std::underlying_type_t<T>(field::EnumValues<T>[below(field::EnumValues<T>.size())]));
                    } else if constexpr (std::floating_point<T>) {
                        update(member == T(0) ? T(below(2) ? .25 : -.25) : T(member * (below(2) ? 1.25 : .8)));
                    } else if constexpr (std::is_arithmetic_v<T>) {
                        update(T(member > T(0) && (member > T(3) || below(2)) ? member - T(1) : member + T(1)));
                    } else if constexpr (std::same_as<T, vec3>) {
                        update(vec3{coordinate(), coordinate(), coordinate()} * .25f);
                    } else if constexpr (std::same_as<T, state::Entity>) {
                        update(below(4) ? pick(objects()) : state::Null);
                    }
                });
            }
        });
    };
    // A literal of a type the op input takes, or a choice of a text input.
    const auto op_input = [&](state::Entity node) -> std::optional<SetOpInput> {
        const auto &op = r.get<const graph::Op>(node);
        const auto &definition = *graph::DefinitionOf(op.Node);
        if (!definition.Texts.empty() && (definition.Inputs == 0u || below(3) == 0u)) {
            const auto name = definition.Texts[below(definition.Texts.size())].first;
            const auto &lib = mx::Libraries();
            for (const auto d : lib.DeclarationsByNode.find(op.Node)->second) {
                const auto &inputs = lib.Declarations[d].Inputs;
                const auto input = std::ranges::find(inputs, name, &mx::Port::Name);
                if (input != inputs.end() && !input->EnumNames.empty()) return OpInputOf(node, name, input->EnumNames[below(input->EnumNames.size())]);
            }
            return {};
        }
        if (definition.Inputs == 0u) return {};
        const auto &socket = definition.Sockets[below(definition.Inputs)];
        std::vector<mx::Type> types;
        for (const auto type : {mx::Type::Boolean, mx::Type::Integer, mx::Type::Float, mx::Type::Vector2, mx::Type::Vector3, mx::Type::Vector4, mx::Type::Color3, mx::Type::Color4}) {
            if (const auto data = graph::DataTypeOf(type); data && (socket.Types & graph::TypeBit(*data))) types.push_back(type);
        }
        if (types.empty()) return {};
        const auto type = types[below(types.size())];
        std::vector<uint32_t> words(mx::Words(type));
        for (auto &word : words) word = type == mx::Type::Boolean ? uint32_t(below(2)) : type == mx::Type::Integer ? std::bit_cast<uint32_t>(int32_t(below(5)) - 1) :
                                                                                                                     std::bit_cast<uint32_t>(float(int32_t(below(9)) - 4) * .25f);
        return OpInputOf(node, socket.Name, action::nodes::OpLiteral{type, std::move(words)});
    };
    // Adds a node of the spec to the graph and returns it, or null on a rejection.
    const auto add_node = [&](state::Entity graph, action::nodes::NodeSpec spec) { return attempt(AddNode{graph, std::move(spec)}) ? graph::NodesOf(r, graph).back() : state::Null; };
    auto &panels = *f.Panels;
    ImGui::SetCurrentContext(panels.Context);
    auto &io = ImGui::GetIO();
    // Draws an interactive frame of the panels and runs the app's frame after it.
    const auto ui_frame = [&] {
        DrawPanels(f, panels, true);
        f.P->Frame();
    };
    const auto press = [&](ImGuiKey key) {
        io.AddKeyEvent(key, true);
        ui_frame();
        io.AddKeyEvent(key, false);
        ui_frame();
    };
    // Shows a random geometry graph's page under the mouse, settled.
    const auto canvas_on = [&](state::Entity graph) {
        graph::EnterGraph(panels.Nodes, graph);
        io.AddMousePosEvent(500.f, 500.f);
        for (int i = 0; i < 2; ++i) ui_frame();
    };
    // Ends a node editor command, whose actions are recorded as the editor emits them, and returns whether it recorded a node.
    const auto finish_command = [&](size_t nodes) {
        io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
        ui_frame();
        messages.clear();
        f.Audit();
        // A command that records nothing can still change the workspace, which the present's record follows.
        f.Step(true);
        return f.P->History.Nodes.size() != nodes;
    };
    // A search text of a random kind's or op's name.
    const auto search_text = [&] {
        std::string text;
        if (below(2)) graph::VisitKind(uint16_t(below(graph::KindCount)), [&]<typename K> { text = field::TypeLabel<K>.c_str(); });
        else text = std::string{GeometryOps()[below(GeometryOps().size())]};
        return text;
    };
    // A request known to be a loop is rejected.
    const auto parent_to_active = [&](bool loop) {
        const auto unchanged = unchanged_since();
        project::Do(*f.P, action::MakeAction(action::object::ParentToActive{}));
        if (messages.empty()) {
            expect(!loop) << at() << "a parent loop was accepted";
            f.Audit();
            f.Step(true);
            return true;
        }
        expect(messages == std::vector<std::string>{"Loop in parents"}) << at();
        return refused(unchanged, "loop in parents");
    };
    for (; step < steps; ++step) {
        // A quarter of the steps move through the history.
        if (below(4) == 0u) {
            boost::ut::log << std::format("{} history", at());
            if (const auto choice = below(3); choice == 0u) f.Undo();
            else if (choice == 1u) f.Redo();
            else f.Navigate(uint32_t(below(f.P->History.Nodes.size())));
            continue;
        }
        // The node entries take twice the weight of the others.
        constexpr auto first_node = size_t(WalkStep::AddNode), node_entries = size_t(WalkStep::Count) - first_node;
        const auto draw = below(first_node + 2u * node_entries);
        const auto entry = WalkStep(draw < first_node ? draw : first_node + (draw - first_node) / 2u);
        // Each step names itself, so a failure of a check it runs follows its seed and step.
        boost::ut::log << std::format("{} {}", at(), WalkStepNames[size_t(entry)]);
        // Collapsing into a subgraph, expanding, making unique, exposing a socket, freezing a graph and entering a page leave the shape of every mesh drawn before and after unchanged.
        bool keeps_shapes = entry == WalkStep::CollapseNodes || entry == WalkStep::ExpandNodes || entry == WalkStep::MakeUnique || entry == WalkStep::ExposeSocket || entry == WalkStep::FreezeGraph || entry == WalkStep::EnterPage;
        const auto shapes = keeps_shapes ? Live::DrawnShapes(f) : std::map<state::Entity, Shape>{};
        const bool done = [&] {
            switch (entry) {
                case WalkStep::AddPrimitive: {
                    const std::array<PrimitiveParameters, 8> primitives{primitive::Plane{.Subdivisions = {2u, 2u}}, primitive::Circle{.Segments = 8}, primitive::Cube{}, primitive::IcoSphere{.Subdivisions = 1}, primitive::UVSphere{.Slices = 12, .Stacks = 6}, primitive::QuadSphere{.Segments = 2}, primitive::Torus{.MajorSegments = 12, .MinorSegments = 6}, primitive::Cylinder{.Slices = 8}};
                    const auto shape = primitives[below(primitives.size())];
                    if (!attempt(action::object::AddMeshPrimitive{shape, std::make_unique<MeshObjectCreateInfo>(MeshObjectCreateInfo{.Transform = {.P = {coordinate(), 0.f, coordinate()}}})})) return false;
                    // Half the meshes start from a Source node holding the primitive's record.
                    return below(2) || attempt(FreezeGraph{r.get<const ObjectGeometry>(FindActiveEntity(r)).Entity});
                }
                case WalkStep::AddEmpty: {
                    const std::string requested = below(2) ? "Thing" : "Other";
                    bool taken = false;
                    uint64_t past = 0;
                    for (const auto [e, name] : r.view<const Name>().each()) {
                        taken |= name.Value == requested;
                        if (name.Value.size() > requested.size() && NameStem(name.Value) == requested) past = std::max(past, std::stoull(name.Value.substr(requested.size() + 1u)));
                    }
                    const auto expected = taken ? std::format("{}_{}", requested, past + 1u) : requested;
                    expect(r.get<const Name>(f.AddEmpty(requested)).Value == expected) << at() << expected;
                    if (taken) ++reached["name collision"];
                    return true;
                }
                case WalkStep::AddLight: return attempt(action::object::AddLight{std::make_unique<ObjectCreateInfo>()});
                case WalkStep::AddViewport: {
                    const auto viewport = f.AddViewport();
                    if (below(2)) return true;
                    // The camera move records no node, so the undo removes the viewport.
                    f.MoveCamera(viewport, {coordinate(), coordinate(), 1.f});
                    f.Undo();
                    expect(!r.valid(viewport) && f.AddViewport() == viewport) << at();
                    ++reached["fresh identity"];
                    return true;
                }
                case WalkStep::RemoveViewport: {
                    const auto viewport = pick(viewports());
                    std::map<state::Entity, ViewCameraState> cameras;
                    for (const auto e : viewports())
                        if (e != viewport) cameras.emplace(e, f.Camera(e));
                    if (cameras.empty()) return false;
                    f.Do(action::view::RemoveViewport{viewport});
                    expect(!r.valid(viewport)) << at();
                    for (const auto &[e, camera] : cameras) expect(f.Camera(e) == camera) << at() << "camera of" << state::Index(e);
                    ++reached["viewport removal"];
                    return true;
                }
                case WalkStep::ChangeView: {
                    const auto viewport = pick(viewports());
                    const auto node = f.P->History.Present;
                    const auto nodes = f.P->History.Nodes.size();
                    if (below(2)) {
                        f.MoveCamera(viewport, {coordinate(), coordinate(), 1.f});
                    } else {
                        const auto lod_error = 1.f + float(below(4));
                        f.Do(action::UpdateOf<&ViewportDisplay::LodErrorPixels>(viewport, lod_error));
                        expect(r.get<const ViewportDisplay>(viewport).LodErrorPixels == lod_error) << at();
                    }
                    expect(f.P->History.Present == node && f.P->History.Nodes.size() == nodes) << at() << "a view change recorded a node";
                    return true;
                }
                case WalkStep::SelectObject: {
                    const auto e = pick(objects());
                    return e != state::Null && attempt(action::selection::Select{e});
                }
                case WalkStep::ExtendActive: {
                    const auto e = pick(objects());
                    return e != state::Null && attempt(action::selection::ExtendActive{e});
                }
                case WalkStep::ParentToActive: {
                    bool loop = false;
                    if (const auto child = pick(objects()), parent = pick(objects()); child != parent && below(2)) {
                        f.Do(action::selection::Select{child});
                        f.Do(action::selection::ExtendActive{parent});
                        parent_to_active(false);
                        if (ParentOrNull(r, child) == parent) {
                            f.Do(action::selection::Select{parent});
                            f.Do(action::selection::ExtendActive{child});
                            // An instance's copy parents through its source, so only two plain objects make a known loop.
                            loop = !r.any_of<CopyOf>(child) && !r.any_of<CopyOf>(parent);
                        }
                    }
                    return parent_to_active(loop);
                }
                case WalkStep::MoveActive: return FindActiveEntity(r) != state::Null && attempt(action::UpdateOf<&Transform::P>(action::OnActive{}, vec3{coordinate(), coordinate(), coordinate()}));
                case WalkStep::Delete: return attempt(action::object::Delete{});
                case WalkStep::Duplicate: return below(2) ? attempt(action::object::Duplicate{}) : attempt(action::object::DuplicateLinked{});
                case WalkStep::LookThroughCamera: {
                    const auto viewport = pick(viewports());
                    const CameraLens lens{Perspective{.FieldOfViewRad = .9f, .FarClip = 1000.f, .NearClip = .1f}};
                    const auto camera = f.Add(action::object::AddCamera{std::make_unique<ObjectCreateInfo>(), lens});
                    f.Do(action::view::SetLookThroughCamera{viewport, camera});
                    expect(LookThroughCamera(r, viewport) == camera) << at();
                    f.Undo();
                    expect(LookThroughCamera(r, viewport) == state::Null && f.Camera(viewport).LookThrough == camera) << at();
                    f.Redo();
                    expect(LookThroughCamera(r, viewport) == camera) << at();
                    ++reached["look through"];
                    return true;
                }
                case WalkStep::SetVisible: {
                    const auto e = pick(objects());
                    return e != state::Null && attempt(action::object::SetVisible{below(2) == 0u, e});
                }
                case WalkStep::Instantiate: {
                    const auto subgraph = some_scene_subgraph();
                    return subgraph != state::Null && attempt(action::object::Instantiate{subgraph});
                }
                case WalkStep::MakeInstanceUnique: {
                    // An instance takes objects of its own by make unique, by expand, or by expand of all levels.
                    const auto e = some_instance();
                    if (e == state::Null) return false;
                    const auto way = below(3);
                    return way == 0u ? attempt(MakeUnique{e}) : attempt(Expand{e, way == 2u});
                }
                case WalkStep::LoadRig: return r.view<const ArmatureObject>().size() < 2u && attempt(action::io::LoadGltf{RiggedSimple});
                case WalkStep::SetFrame: return attempt(action::timeline::SetFrame{1 + int(below(30))});
                case WalkStep::AddMaterial: {
                    const auto mesh = pick(meshes());
                    return mesh != state::Null && attempt(action::object::AddMaterial{0u, mesh});
                }
                case WalkStep::ReadTexcoords: {
                    const auto generators_of = [&](state::Entity mesh) {
                        std::vector<state::Entity> found;
                        for (const auto node : nodes_of(mesh)) {
                            graph::VisitKind(r.get<const graph::GraphNode>(node).Kind, [&]<typename K> { if (graph::Generator<K>) found.push_back(node); });
                        }
                        return found;
                    };
                    auto generated = meshes();
                    std::erase_if(generated, [&](state::Entity mesh) { return generators_of(mesh).empty(); });
                    const auto mesh = pick(generated);
                    if (mesh == state::Null) return false;
                    const auto op = OpNode{std::string{TexcoordOps()[below(TexcoordOps().size())]}};
                    // A Store Named Attribute stores the op's field on a generator's output and shows it, or a material the mesh draws computes an input from the op.
                    if (below(2)) {
                        const auto generator = pick(generators_of(mesh));
                        const auto store = add_node(mesh, KindNode{graph::KindOf<graph::StoreNamedAttribute>});
                        if (store == state::Null) return false;
                        const auto read = add_node(mesh, op);
                        if (read == state::Null) return false;
                        // Links the first output of `from` that links into an input of `to`.
                        const auto link = [&](state::Entity from, state::Entity to) {
                            const auto pairs = linkable(mesh, from, std::array{to});
                            return !pairs.empty() && attempt(Link{mesh, from, pairs[0].first.Id, to, pairs[0].second.Id});
                        };
                        return link(generator, store) && link(read, store) && attempt(SetDisplay{mesh, store});
                    }
                    // The material the mesh's first primitive draws, or null for the default.
                    const auto drawn = [&] {
                        const auto &store = f.Store();
                        const auto id = DrawnStoreId(r, mesh);
                        const auto materials = id ? store.Arenas().PrimitiveMaterials.Get(store.Get(*id).PrimitiveMaterials) : std::span<const uint32_t>{};
                        return materials.empty() ? state::Null : MaterialAt(r, materials[0]);
                    };
                    if (drawn() == state::Null && !attempt(action::object::AddMaterial{0u, mesh})) return false;
                    const auto material = drawn();
                    if (material == state::Null) return false;
                    if (!r.all_of<graph::MaterialGraph>(material) && !attempt(action::nodes::AddMaterialGraph{material})) return false;
                    const auto node = add_node(material, op);
                    if (node == state::Null) return false;
                    return link_any(material, linkable(material, node, std::array{graph::SurfaceNodeOf(r, material)}));
                }
                case WalkStep::AssignReference: {
                    const auto object = pick(objects());
                    if (object == state::Null) return false;
                    const std::array kinds{graph::ReferenceKind::Parent, graph::ReferenceKind::Geometry, graph::ReferenceKind::Material};
                    const graph::SceneReference reference{object, kinds[below(kinds.size())], 0u};
                    auto candidates = graph::ReferenceCandidates(r, reference);
                    candidates.push_back(state::Null);
                    return attempt(AssignReference{reference, pick(candidates)});
                }
                case WalkStep::DeleteItems: {
                    // An item of the document's page, or a scene subgraph, which stays in the document while an instance uses it.
                    const auto item = pick(below(4) ? graph::GraphItems(r, state::Document) : scene_subgraphs());
                    return item != state::Null && attempt(DeleteItems{{item}});
                }
                case WalkStep::CollapseItems: {
                    const auto items = graph::GraphItems(r, FrameScene(r));
                    return !items.empty() && attempt(Collapse{FrameScene(r), some(items, 2u)});
                }
                case WalkStep::ExpandItems: {
                    const auto subgraph = pick(SortedEntities(r.view<const graph::SubgraphItems>()));
                    return subgraph != state::Null && attempt(Expand{subgraph});
                }
                case WalkStep::ExposePort: {
                    const auto subgraph = some_scene_subgraph();
                    return subgraph != state::Null && expose_position(subgraph);
                }
                case WalkStep::InstancePortValue: {
                    // An instance's own value of a port, set at the port or through the field of its copy, or cleared.
                    const auto instance = some_instance();
                    if (instance == state::Null) return false;
                    // An instance whose subgraph has no port takes one for the position of an object the subgraph holds.
                    const auto subgraph = r.get<const Instance>(instance).SubgraphEntity;
                    if (r.get<const graph::SubgraphInterface>(subgraph).Ports.empty() && !expose_position(subgraph)) return false;
                    const auto &ports = r.get<const graph::SubgraphInterface>(subgraph).Ports;
                    const auto port = ports[below(ports.size())].Id;
                    const vec3 position{coordinate(), coordinate(), coordinate()};
                    const auto copy = CopyFor(r.get<const Instance>(instance).Copies, std::get<graph::PromotedField>(graph::PortOf(r, instance, port)->Binding).Entity);
                    const auto held = [&] { return r.get<const Transform>(copy).P; };
                    switch (below(3)) {
                        case 0u: return attempt(ClearPortValue{instance, port});
                        case 1u: {
                            std::array<uint32_t, 4> words{};
                            std::memcpy(words.data(), &position, sizeof(position));
                            return attempt(SetPortValue{instance, port, words}) && (expect(held() == position) << at(), true);
                        }
                        default: return attempt(action::UpdateOf<&Transform::P>(copy, position)) && (expect(held() == position) << at(), true);
                    }
                }
                case WalkStep::AddNode: {
                    const auto graph = pick(graphs());
                    if (graph == state::Null) return false;
                    const auto user = pick(subgraphs());
                    const auto spec = [&]() -> action::nodes::NodeSpec {
                        if (const auto choice = below(8); choice < 3u) {
                            auto kind = uint16_t(below(graph::KindCount - 2u));
                            if (kind >= graph::KindOf<graph::Op>) kind += 2u;
                            return KindNode{kind};
                        } else if (choice < 6u) {
                            return OpNode{std::string{GeometryOps()[below(GeometryOps().size())]}};
                        } else if (choice == 6u || user == state::Null) {
                            return BuiltinSubgraphNode{graph::BuiltinSubgraphId(below(graph::BuiltinSubgraphCount))};
                        } else {
                            return UserSubgraphNode{user};
                        }
                    }();
                    const auto node = add_node(graph, spec);
                    // Most added nodes join the graph through a link.
                    if (node != state::Null && below(4)) link_in(graph, node);
                    return node != state::Null;
                }
                case WalkStep::Link: return link_in(pick(graphs()), state::Null);
                case WalkStep::Unlink: {
                    const auto graph = pick(graphs());
                    if (graph == state::Null) return false;
                    const auto linked = LinkedInputs(r, graph);
                    if (linked.empty()) return false;
                    const auto [node, socket, index] = linked[below(linked.size())];
                    return attempt(Unlink{graph, node, socket, index});
                }
                case WalkStep::InsertNode: {
                    const auto graph = pick(graphs());
                    if (graph == state::Null) return false;
                    const auto linked = LinkedInputs(r, graph);
                    if (linked.empty()) return false;
                    // An insert takes a node no link touches into a geometry link.
                    std::set<state::Entity> touched;
                    graph::ForEachGraphInput(r, graph, [&](state::Entity node, uint16_t, graph::SocketLink link) {
                        if (link.FromNode != state::Null) touched.insert({node, link.FromNode});
                    });
                    auto free = nodes_of(graph);
                    std::erase_if(free, [&](state::Entity node) { return touched.contains(node); });
                    std::vector<std::tuple<state::Entity, uint16_t, uint16_t>> geometry;
                    for (const auto &input : linked) {
                        const auto &[to, socket, index] = input;
                        const auto sockets = SocketsFacing(r, graph, to, SocketDirection::In);
                        const auto found = std::ranges::find(sockets, socket, &PageSocket::Id);
                        if (found != sockets.end() && (found->Socket.Types & graph::TypeBit(graph::DataType::Geometry))) geometry.push_back(input);
                    }
                    if (geometry.empty()) return false;
                    // Without a node no link touches, the insert takes a new Subdivide node.
                    const auto node = free.empty() ? add_node(graph, KindNode{graph::KindOf<action::mesh::Subdivide>}) : pick(free);
                    if (node == state::Null) return false;
                    const auto [to, socket, index] = geometry[below(geometry.size())];
                    return attempt(InsertNode{graph, node, to, socket, uint32_t(below(4)), index});
                }
                case WalkStep::DeleteNodes: {
                    const auto graph = pick(graphs());
                    const auto nodes = nodes_of(graph);
                    return !nodes.empty() && attempt(DeleteNodes{graph, some(nodes, 2u)});
                }
                case WalkStep::DuplicateNodes: {
                    const auto graph = pick(graphs());
                    const auto nodes = nodes_of(graph);
                    return !nodes.empty() && attempt(DuplicateNodes{graph, some(nodes, 2u)});
                }
                case WalkStep::CollapseNodes: {
                    const auto graph = pick(graphs());
                    const auto nodes = nodes_of(graph);
                    return !nodes.empty() && attempt(Collapse{graph, some(nodes, 3u)});
                }
                case WalkStep::ExpandNodes: {
                    const auto use = pick(uses());
                    return use != state::Null && attempt(Expand{use, below(2) == 0u});
                }
                case WalkStep::MakeUnique: {
                    const auto use = pick(uses());
                    return use != state::Null && attempt(MakeUnique{use});
                }
                case WalkStep::ExposeSocket: {
                    const auto subgraph = pick(subgraphs());
                    const auto node = pick(nodes_of(subgraph));
                    if (node == state::Null) return false;
                    auto sockets = SocketsFacing(r, subgraph, node, SocketDirection::In);
                    sockets.append_range(SocketsFacing(r, subgraph, node, SocketDirection::Out));
                    if (sockets.empty()) return false;
                    const auto &socket = sockets[below(sockets.size())];
                    // A port that replaces an input's link changes what the node reads.
                    if (socket.Socket.Direction == SocketDirection::In && !socket.Socket.Multi && graph::InputOf(r, node, socket.Id).FromNode != state::Null) keeps_shapes = false;
                    return attempt(ExposeSocket{subgraph, node, socket.Id, {}});
                }
                case WalkStep::NumericValue: {
                    auto holders = uses();
                    holders.append_range(subgraphs());
                    const auto holder = pick(holders);
                    if (holder == state::Null) return false;
                    std::vector<const graph::SubgraphPort *> ports;
                    for (const auto &port : graph::InterfaceOf(r, holder)->Ports)
                        if (graph::ReadsDefault(port)) ports.push_back(&port);
                    if (ports.empty()) return false;
                    const auto &port = *ports[below(ports.size())];
                    if (below(4) == 0u) return attempt(ClearPortValue{holder, port.Id});
                    const bool real = port.Types & (graph::NumericTypes & ~(graph::TypeBit(graph::DataType::Int) | graph::TypeBit(graph::DataType::Bool)));
                    std::array<uint32_t, 4> words{};
                    for (auto &word : words) word = real ? std::bit_cast<uint32_t>(.25f * float(1u + below(8))) : uint32_t(below(4));
                    return attempt(SetPortValue{holder, port.Id, words});
                }
                case WalkStep::SetDisplay: {
                    const auto mesh = pick(meshes());
                    if (mesh == state::Null) return false;
                    std::vector<SetDisplay> displays{{mesh}};
                    for (const auto node : nodes_of(mesh)) {
                        if (const auto port = graph::ResultSocket(r, node)) displays.push_back({mesh, node, port});
                        if (!r.all_of<graph::SubgraphNode>(node)) continue;
                        const auto inner = graph::SubgraphOf(r, node);
                        for (const auto n : graph::NodesOf(*inner.Registry, inner.Entity))
                            if (const auto port = graph::ResultSocket(*inner.Registry, n)) displays.push_back({mesh, n, port, {node}});
                    }
                    return attempt(std::move(displays[below(displays.size())]));
                }
                case WalkStep::SetBypass: {
                    // A bypass passes a node's input through, so it takes a node that has one.
                    auto bypassed = nodes_of(pick(graphs()));
                    std::erase_if(bypassed, [&](state::Entity node) { return !r.all_of<graph::SubgraphNode>(node) && !graph::BypassInput(r.get<const graph::GraphNode>(node).Kind); });
                    const auto node = pick(bypassed);
                    return node != state::Null && attempt(SetBypass{node, !r.all_of<graph::NodeBypassed>(node)});
                }
                case WalkStep::EditParameter: {
                    const auto node = pick(nodes_of(pick(graphs())));
                    if (node == state::Null) return false;
                    bool done = false;
                    parameter_update(node, [&](auto update) { done = attempt(std::move(update)); });
                    return done;
                }
                case WalkStep::SetOpInput: {
                    const auto node = pick(NodesWith<graph::Op>(r, graphs()));
                    if (node == state::Null) return false;
                    auto input = op_input(node);
                    return input && attempt(std::move(*input));
                }
                case WalkStep::ChangeOp: {
                    const auto node = pick(NodesWith<graph::Op>(r, graphs()));
                    if (node == state::Null) return false;
                    const auto &definition = *graph::DefinitionOf(r.get<const graph::Op>(node).Node);
                    if (definition.StoresOutput && below(2)) return attempt(SetOpOutputType{node, definition.OutputTypes[below(definition.OutputTypes.size())]});
                    return attempt(ChangeOp{node, std::string{GeometryOps()[below(GeometryOps().size())]}});
                }
                case WalkStep::MaterialPage: {
                    // A new op of a random material's page, then a literal of the op, a link from the op into an input of the page's Surface node, or a group of the page's nodes.
                    // A document without a graph first gives one to a random material, and a document without a material first adds one to a random mesh.
                    if (materials().empty()) {
                        if (r.view<const Material>().empty() && (meshes().empty() || !attempt(action::object::AddMaterial{0u, pick(meshes())}))) return false;
                        if (!attempt(action::nodes::AddMaterialGraph{pick(SortedEntities(r.view<const Material>()))})) return false;
                    }
                    const auto material = pick(materials());
                    const auto op = add_node(material, OpNode{std::string{GeometryOps()[below(GeometryOps().size())]}});
                    if (op == state::Null) return false;
                    if (const auto choice = below(3); choice == 0u) {
                        auto input = op_input(op);
                        return input && attempt(std::move(*input));
                    } else if (choice == 1u) {
                        return link_any(material, linkable(material, op, std::array{graph::SurfaceNodeOf(r, material)}));
                    }
                    auto collapsed = nodes_of(material);
                    std::erase(collapsed, graph::SurfaceNodeOf(r, material));
                    return attempt(Collapse{material, some(collapsed, 3u)});
                }
                case WalkStep::MoveNodes: {
                    const auto graph = pick(graphs());
                    const auto node = pick(nodes_of(graph));
                    if (node == state::Null) return false;
                    if (below(3) == 0u) return attempt(AutoLayout{graph});
                    return attempt(MoveNodes{graph, {{node, {uint32_t(below(24)), uint32_t(below(16))}}}});
                }
                case WalkStep::FreezeGraph: {
                    const auto mesh = pick(meshes());
                    return mesh != state::Null && attempt(FreezeGraph{mesh});
                }
                case WalkStep::EnterPage: {
                    // The pages the node editor shows: the document's, each scene's, each graph's, each subgraph node's, each scene or document subgraph's and each instance's.
                    auto pages = graphs();
                    pages.push_back(state::Document);
                    pages.append_range(SortedEntities(r.view<const Scene>()));
                    pages.append_range(SortedEntities(r.view<const graph::SubgraphItems>()));
                    pages.append_range(SortedEntities(r.view<const Instance>()));
                    const auto nodes = uses();
                    pages.append_range(nodes);
                    const auto page = pick(pages);
                    if (std::ranges::contains(nodes, page)) graph::EnterGraph(panels.Nodes, graph::GraphOf(r, page), state::Null, {page});
                    else graph::EnterGraph(panels.Nodes, page);
                    return true;
                }
                case WalkStep::StageParameter: {
                    const auto node = pick(nodes_of(pick(graphs())));
                    if (node == state::Null) return false;
                    return gesture(below(3) != 0u, [&](auto &&stage) {
                        for (size_t i = 0, count = 2u + below(2); i < count; ++i) parameter_update(node, stage);
                    });
                }
                case WalkStep::ExtrudeDrag: {
                    if (select_mesh() == state::Null) return false;
                    f.EnterEdit(Element::Face);
                    f.Do(action::selection::SelectAll{});
                    const auto mode = field::EnumValues<action::mesh::ExtrudeMode>[below(field::EnumValues<action::mesh::ExtrudeMode>.size())];
                    const bool done = gesture(below(3) != 0u, [&](auto &&stage) {
                        stage(action::mesh::Extrude{.Mode = mode});
                        for (size_t i = 0, count = 1u + below(3); i < count; ++i) stage(action::view::TransformElements{{.P = {0.f, 0.f, .1f * float(i + 1u)}}});
                    });
                    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
                    return done;
                }
                case WalkStep::ViewportOperator: {
                    if (select_mesh() == state::Null) return false;
                    f.EnterEdit(std::array{Element::Vertex, Element::Edge, Element::Face}[below(3)]);
                    f.Do(action::selection::SelectAll{});
                    const bool done = [&] {
                        switch (below(4)) {
                            case 0: return attempt(action::mesh::Subdivide{});
                            case 1: return attempt(action::mesh::Extrude{});
                            case 2: return attempt(action::mesh::Delete{action::mesh::DeleteMode::Faces});
                            default: return attempt(action::mesh::Unsubdivide{1u});
                        }
                    }();
                    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
                    return done;
                }
                case WalkStep::EditSelection: {
                    const auto mesh = select_mesh();
                    if (mesh == state::Null) return false;
                    std::vector<state::Entity> selecting;
                    for (const auto node : nodes_of(mesh)) {
                        const auto sockets = graph::SocketsOf(r, node);
                        if (!r.all_of<graph::SubgraphNode>(node) && std::ranges::contains(sockets, true, &graph::Socket::Selection)) selecting.push_back(node);
                    }
                    const auto node = pick(selecting);
                    if (node == state::Null) return false;
                    const auto commit = below(3) != 0u;
                    const auto unchanged = unchanged_since();
                    action::Emit(EditSelection{node, FindActiveEntity(r)});
                    f.P->Frame();
                    if (!messages.empty()) {
                        action::Cancel();
                        f.P->Frame();
                        return refused(unchanged, "rejected");
                    }
                    action::Emit(action::view::SetEditMode{Element::Face});
                    action::Emit(below(2) ? action::selection::Action{action::selection::SelectAll{}} : action::selection::Action{action::selection::DeselectAll{}});
                    f.P->Frame();
                    if (commit) return attempt(FinishSelection{});
                    action::Cancel();
                    f.P->Frame();
                    return refused(unchanged, "cancelled");
                }
                case WalkStep::MakeLinks: {
                    const auto graph = pick(meshes());
                    auto nodes = nodes_of(graph);
                    if (nodes.size() < 2u) return false;
                    // Make Links chains the selected nodes, so the selection favors geometry nodes whose inputs are free.
                    const auto linked = LinkedInputs(r, graph);
                    auto free = nodes;
                    std::erase_if(free, [&](state::Entity node) {
                        const auto sockets = SocketsFacing(r, graph, node, SocketDirection::In);
                        return std::ranges::contains(linked, node, [](const auto &input) { return std::get<0>(input); }) || std::ranges::none_of(sockets, [](const PageSocket &socket) { return bool(socket.Socket.Types & graph::TypeBit(graph::DataType::Geometry)); });
                    });
                    auto chosen = free.size() >= 2u && below(4) ? free : nodes;
                    std::ranges::shuffle(chosen, rng);
                    chosen.resize(std::min<size_t>(chosen.size(), 3u));
                    if (!attempt(SelectItems{graph, chosen.back(), chosen})) return false;
                    const auto count = f.P->History.Nodes.size();
                    const auto selected = f.P->History.Present;
                    canvas_on(graph);
                    const bool replace = below(2);
                    const auto planned = graph::MakeLinksOf(r, panels.Nodes, replace);
                    if (replace) io.AddKeyEvent(ImGuiMod_Shift, true);
                    press(ImGuiKey_J);
                    io.AddKeyEvent(ImGuiMod_Shift, false);
                    // A link the graph refuses, such as a field reaching a single-value input inside a subgraph, refuses the whole command.
                    const bool rejected = !messages.empty();
                    const bool completed = finish_command(count);
                    // Make Links records one node holding each link it plans, or returns to a node of the history that holds the state it reaches, and each linked input reads its new source.
                    if (planned.empty() || rejected) {
                        expect(f.P->History.Nodes.size() == count) << at() << "make links recorded a node without a link to make";
                        if (rejected) ++reached["rejected"];
                        return false;
                    }
                    const bool recorded_node = f.P->History.Nodes.size() == count + 1u;
                    expect(recorded_node || (f.P->History.Nodes.size() == count && f.P->History.Present != selected)) << at() << "make links recorded" << f.P->History.Nodes.size() - count << "nodes";
                    const auto &recorded = project::DraftOf(*f.P, *f.P->History.Present).RecordedActions;
                    expect(!recorded_node || recorded.size() == planned.size()) << at() << "make links planned" << planned.size() << "links and recorded" << recorded.size();
                    for (const auto &link : planned) {
                        bool linked = false;
                        graph::ForEachGraphInput(r, graph, [&](state::Entity node, uint16_t, graph::SocketLink from) { linked |= node == link.ToNode && from.FromNode == link.FromNode && from.FromSocket == link.FromSocket; });
                        expect(linked) << at() << "make links left a planned link out";
                    }
                    return completed;
                }
                case WalkStep::AddSearch: {
                    const auto graph = pick(graphs());
                    if (graph == state::Null) return false;
                    const auto count = f.P->History.Nodes.size();
                    canvas_on(graph);
                    io.AddKeyEvent(ImGuiMod_Shift, true);
                    press(ImGuiKey_A);
                    io.AddKeyEvent(ImGuiMod_Shift, false);
                    ui_frame();
                    io.AddInputCharactersUTF8(search_text().c_str());
                    ui_frame();
                    press(ImGuiKey_Enter);
                    // The grab of the added node ends without moving it.
                    press(ImGuiKey_Escape);
                    return finish_command(count);
                }
                case WalkStep::LinkingAdd: {
                    const auto graph = pick(graphs());
                    if (graph == state::Null) return false;
                    const auto count = f.P->History.Nodes.size();
                    canvas_on(graph);
                    const auto boxes = graph::DrawnGeometry(r, panels.Nodes).Nodes;
                    if (boxes.empty()) return finish_command(count);
                    // A wire dropped from a node's first output row on empty canvas opens the linking add.
                    const auto box = boxes[below(boxes.size())];
                    const auto output = CanvasPoint({box.Max.x - .5f, box.Min.y + 1.5f}), empty = CanvasPoint({box.Max.x + 12.f, box.Max.y + 12.f});
                    io.AddMousePosEvent(output.x, output.y);
                    ui_frame();
                    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
                    ui_frame();
                    io.AddMousePosEvent(empty.x, empty.y);
                    ui_frame();
                    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
                    for (int i = 0; i < 3; ++i) ui_frame();
                    io.AddInputCharactersUTF8(search_text().c_str());
                    ui_frame();
                    press(ImGuiKey_Enter);
                    press(ImGuiKey_Escape);
                    return finish_command(count);
                }
                case WalkStep::Count: break;
            }
            return false;
        }();
        if (done) ++reached[WalkStepNames[size_t(entry)]];
        if (done && keeps_shapes) {
            for (const auto &[mesh, shape] : Live::DrawnShapes(f)) {
                if (const auto before = shapes.find(mesh); before != shapes.end()) expect(shape == before->second) << at() << "changed the shape of mesh" << state::Index(mesh);
            }
        }
    }
    for (const auto rule : WalkRules) expect(reached[rule] > 0u) << "seed" << seed << "never reached" << rule;
    for (const auto name : WalkStepNames) expect(reached[name] > 0u) << "seed" << seed << "never completed" << name;
}

// Switching documents starts every workspace value from defaults and keeps the output device setting.
void TestDocumentSwitch(HistoryFixture &f, HistoryFixture &other) {
    const TestDir dir{"/tmp/mesheditor-scratch/history-switch"}, other_dir{"/tmp/mesheditor-scratch/history-switch-other"}, archives{"/tmp/mesheditor-scratch/history-switch-archive"}, opened{"/tmp/mesheditor-scratch/history-switch-opened"}, next{"/tmp/mesheditor-scratch/history-switch-next"};
    const auto archive = archives.Path / "Scene.actions";
    other.Begin(other_dir.Path);
    const auto viewport = other.AddViewport();
    const auto added = other.P->History.Present.value();
    other.Undo();
    const auto other_workspace = snapshot::SerializeWorkspace(other.R);
    expect(other.P->SaveArchive(archive, project::ArchiveForm::Actions, other_workspace));
    expect(Decompress(archive, opened.Path));

    f.Begin(dir.Path);
    f.MoveCamera(FirstViewport(f.R), {1, 2, 3});
    f.R.replace<ActiveViewport>(state::Document, ActiveViewport{FirstViewport(f.R)});
    f.Do(action::audio::SetOutputDevice{.SampleRate = 96000u});
    expect(f.P->New(next));
    expect(f.Camera(FirstViewport(f.R)) == ViewCameraState{});
    expect(f.R.get<const ActiveViewport>(state::Document) == ActiveViewport{});
    expect(f.R.Resources.get<const AudioOutputConfig>().SampleRate == 96000u);

    f.MoveCamera(FirstViewport(f.R), {1, 2, 3});
    expect(f.AddViewport() == viewport);
    f.MoveCamera(viewport, {4, 5, 6});
    f.Do(action::view::RemoveViewport{viewport});
    expect(f.P->Open(opened));
    expect(f.Camera(FirstViewport(f.R)) == ViewCameraState{});
    f.Navigate(added);
    expect(f.Camera(viewport) == ViewCameraState{});
    expect(f.R.Resources.get<const AudioOutputConfig>().SampleRate == 96000u);

    f.Navigate(0);
    f.AddCube();
    expect(f.R.all_of<primitive::Cube>(viewport));
    snapshot::ApplyWorkspace(f.R, other_workspace);
    expect(!f.R.all_of<ViewCameraState>(viewport));
    expect(snapshot::SerializeWorkspace(f.R) == other_workspace);
    f.Audit();
}

// The exported action log, then each node's stamps and manifest roots, of the history `script` records in a new project.
std::vector<std::byte> RecordedHistory(HistoryFixture &f, auto &&script) {
    const TestDir dir{"/tmp/mesheditor-scratch/history-reset"}, exported{"/tmp/mesheditor-scratch/history-reset-export"};
    f.Begin(dir.Path);
    script(f);
    const auto &history = f.P->History;
    std::filesystem::create_directories(exported.Path);
    expect(f.P->Save());
    expect(history.WriteActions(exported.Path / store::History::ActionsLogName).has_value());
    std::vector<std::byte> bytes;
    for (const auto *name : {store::History::ActionsLogName, history.NodeLog.Name, history.LeafLog.Name}) bytes.append_range(File::Read(exported.Path / name).value());
    for (const auto &node : history.Nodes) {
        bytes.append_range(std::as_bytes(std::span{node.Stamps}));
        bytes.append_range(std::as_bytes(std::span{node.Roots}));
    }
    return bytes;
}

// A document begun after a reset records the same history as one begun in a fresh engine.
void TestResetRecordsFresh(HistoryFixture &f) {
    const auto script = [](HistoryFixture &h) {
        h.AddCube();
        h.MoveCamera(FirstViewport(h.R), {1, 2, 3});
        h.MoveCamera(h.AddViewport(), {4, 5, 6});
        h.AddEmpty("Thing");
    };
    HistoryFixture fresh;
    const auto expected = RecordedHistory(fresh, script);
    const TestDir larger{"/tmp/mesheditor-scratch/history-reset-larger"};
    f.Begin(larger.Path);
    f.Do(action::io::LoadGltf{RiggedSimple});
    f.MoveCamera(f.AddViewport(), {7, 8, 9});
    expect(RecordedHistory(f, script) == expected);
}
} // namespace

void TestHistory() {
    HistoryFixture f, other;
    for (const auto phase : {action::Phase::Record, action::Phase::Stage}) {
        Scenario(f, other, phase == action::Phase::Record ? "loop cut gesture" : "staged loop cut gesture", [phase](HistoryFixture &f) {
            const auto cube = f.AddCube();
            f.EnterEdit(Element::Face);
            f.Image();
            const auto hit = RunEditElementPick(f.R, FirstViewport(f.R), Element::Edge, {32, 32});
            expect(hit.has_value());
            const auto before = f.P->History.MaterializeLive();
            const auto nodes = f.P->History.Nodes.size();
            const auto emit = [&](uint32_t cuts) {
                action::Emit(phase, action::view::SetEditMode{Element::Edge}, action::selection::ApplyEditElementClick{false, hit}, action::nodes::AppendSubgraphNode{graph::BuiltinSubgraphId::LoopCut, {{cuts, 0u, 0u, 0u}}});
                action::Emit(action::Phase::Record, action::view::SetEditMode{Element::Vertex}, action::object::Delete{});
                f.P->Frame();
                f.Audit();
            };
            emit(1);
            expect(f.R.get<const EditMode>(state::Document).Value == Element::Edge);
            expect(GetMesh(f.R, cube).VertexCount() == 12u);
            if (phase == action::Phase::Stage) {
                emit(2);
                expect(GetMesh(f.R, cube).VertexCount() == 16u);
                expect(f.P->History.Nodes.size() == nodes);
                action::Cancel();
                f.P->Frame();
                expect(f.P->History.MaterializeLive() == before);
                emit(1);
                action::Commit();
                f.P->Frame();
            }
            expect(f.P->History.Nodes.size() == nodes + 1);
            expect(project::DraftOf(*f.P, *f.P->History.Present).RecordedActions.size() == 3u);
            f.Step(true);
            f.Undo();
            expect(f.P->History.MaterializeLive() == before);
            f.Redo();
            expect(GetMesh(f.R, cube).VertexCount() == 12u);
        });
    }
    Scenario(f, other, "gesture failure and background completion", [](HistoryFixture &f) {
        const auto cube = f.AddCube();
        const auto before = f.P->History.MaterializeLive();
        const auto nodes = f.P->History.Nodes.size();
        action::Emit(action::Phase::Record, action::UpdateOf<&Transform::P>(cube, vec3{1, 0, 0}), action::mesh::Boolean{}, action::object::Delete{});
        f.P->Frame();
        auto &errors = f.R.Resources.get<action::Errors>().Messages;
        expect(errors.size() == 1u);
        errors.clear();
        expect(f.P->History.MaterializeLive() == before);
        expect(f.P->History.Nodes.size() == nodes);
        action::Emit(action::Phase::Record, action::UpdateOf<&Transform::P>(cube, vec3{1, 0, 0}), action::UpdateOf<&Transform::P>(cube, vec3{2, 0, 0}));
        action::Post(action::UpdateOf<&Transform::P>(cube, vec3{3, 0, 0}));
        f.P->Frame();
        expect(f.R.get<const Transform>(cube).P == vec3{3, 0, 0});
        expect(f.P->History.Nodes.size() == nodes + 2);
        f.Audit();
        f.Step(true);
        f.Undo();
        expect(f.R.get<const Transform>(cube).P == vec3{2, 0, 0});
        expect(project::DraftOf(*f.P, *f.P->History.Present).RecordedActions.size() == 2u);
        f.Redo();
    });
    for (const bool orthographic : {false, true}) {
        Scenario(f, other, orthographic ? "orthographic selection replay" : "perspective selection replay", [orthographic](HistoryFixture &f) {
            auto &r = f.R;
            const auto viewport = FirstViewport(r);
            if (orthographic) f.Do(action::view::SetViewCameraLens{viewport, Orthographic{{6.f, 6.f}, 1000.f, .1f}});
            const auto cube = f.AddCube();
            f.AddCube();
            f.Image();
            const auto first = PickObject(r, viewport, {32, 32});
            f.Do(action::selection::Pick{false, first});
            r.Resources.get<FrameState>().DisplayFramebufferScale = {2, 2};
            f.ValidateReplay();
            r.Resources.get<FrameState>().DisplayFramebufferScale = {1, 1};
            f.Image();
            const auto next = PickObject(r, viewport, {32, 32}, true);
            expect(first && next && HitEntity(*first) != HitEntity(*next));
            f.Do(action::selection::Pick{true, next});
            expect(PickObject(r, viewport, {32, 32}, true) == first);
            f.Do(action::selection::DeselectAll{});
            for (const bool through : {false, true}) {
                f.Do(action::UpdateOf<&ViewportDisplay::XRaySolid>(viewport, through));
                f.Do(action::selection::ApplyBoxSelect{{vec2{.45f}, vec2{.55f}}, false});
            }
            const auto box_node = *f.P->History.Present;
            const auto target = f.Camera(viewport).Active.Target;
            f.MoveCamera(viewport, {100, 50, 25});
            auto &draft = project::DraftOf(*f.P, box_node);
            auto &box = std::get<action::selection::ApplyBoxSelect>(std::get<action::selection::Action>(draft.RecordedActions.front()));
            box.Box = {vec2{0}, vec2{1}};
            f.P->RequestRestage();
            f.P->Frame();
            expect(r.all_of<Selected>(cube));
            expect(f.Camera(viewport).Active.Target == vec3{100, 50, 25});
            f.P->CancelGesture();
            f.Do(action::view::SetViewCameraTarget{viewport, target});
            f.Do(action::selection::Select{cube});
            f.EnterEdit(Element::Vertex);
            f.Do(action::selection::ApplyBoxSelect{{vec2{0}, vec2{1}}, false});
            f.Do(action::selection::ApplyEditElementClick{false, RunEditElementPick(r, viewport, Element::Vertex, {32, 32})});
            f.Do(action::view::SetViewportShading{viewport, ViewportShadingMode::Wireframe});
            f.MoveCamera(viewport, {100, 50, 25});
        });
    }
    // A node reached by another path draws as it did when made.
    // Two linked cameras with crossing lines are placed in another order when the first reuses a freed index at a later generation.
    // An empty of one branch holds the identity of a bone of another, with the bone's mesh as its geometry.
    Scenario(f, other, "paths to one node", [](HistoryFixture &f) {
        auto &r = f.R;
        const auto add_camera = [&] { return f.Add(action::object::AddCamera{std::make_unique<ObjectCreateInfo>(), {}}); };
        add_camera();
        f.Do(action::object::Delete{});
        const auto camera = add_camera();
        f.Do(action::object::DuplicateLinked{});
        const auto linked = FindActiveEntity(r);
        expect(state::Generation(camera) > state::Generation(linked) && r.get<const ObjectGeometry>(camera).Entity == r.get<const ObjectGeometry>(linked).Entity);
        for (const auto e : {camera, linked}) f.Do(action::UpdateOf<&Transform::S>(e, vec3{6.f}));
        f.Do(action::UpdateOf<&Transform::P>(linked, vec3{.3f, .15f, .05f}));
        f.Do(action::UpdateOf<&Transform::R>(linked, AngleAxis(.3f, Normalize(vec3{1, 2, 3}))));
        f.Do(action::selection::Select{camera});
        const auto rig = f.Do(action::io::LoadGltf{RiggedSimple});
        const auto bone = r.get<const ArmatureObject>(*r.view<const ArmatureObject>().begin()).BoneEntities.front();
        const auto bone_mesh = r.get<const ObjectGeometry>(bone).Entity;
        f.Undo();
        auto empty = f.AddEmpty("Empty");
        while (state::Index(empty) < state::Index(bone)) empty = f.AddEmpty("Empty");
        expect(boost::ut::fatal(empty == bone && r.get<const ObjectGeometry>(empty).Entity == bone_mesh));
        const auto branch = *f.P->History.Present;
        for (const auto node : {rig, branch, rig}) f.Navigate(node);
    });
    // Deleting a copy deletes the item it stands for, and with it each copy of that item, through copies of copies.
    Scenario(f, other, "nested occurrence removal", [](HistoryFixture &f) {
        const auto parent = f.AddEmpty("Thing"), child = f.AddEmpty("Other");
        f.Do(action::selection::Select{child});
        f.Do(action::selection::ExtendActive{parent});
        f.Do(action::object::ParentToActive{});
        f.Do(action::selection::Select{parent});
        const auto inner = f.Instantiate(parent);
        const auto outer = f.Instantiate(inner);
        const auto copies = f.R.get<const Instance>(outer).Copies;
        const auto occurrence = std::ranges::find_if(copies, [&](const auto &pair) { return SourceOfCopy(f.R, pair.CopyEntity) == child; })->CopyEntity;
        f.Do(action::selection::Select{occurrence});
        f.Do(action::object::Delete{});
        expect(!f.R.valid(child) && !f.R.valid(occurrence));
        // The inner instance keeps its copy of the parent, and the outer one its copies of the inner instance and of that copy.
        expect(f.R.get<const Instance>(inner).Copies.size() == 1u && f.R.get<const Instance>(outer).Copies.size() == 2u);
    });
    // The default seed's walk reaches every rule and completes every vocabulary entry.
    for (auto seed = WalkSeeds[0]; seed <= WalkSeeds[1]; ++seed) {
        Scenario(f, other, std::format("random history walk {}", seed), [seed](HistoryFixture &f) { RandomHistoryWalk(f, seed, 1200u); });
    }
    boost::ut::test("switching documents") = [&] { TestDocumentSwitch(f, other); };
    boost::ut::test("reset records like a fresh engine") = [&] { TestResetRecordsFresh(f); };
}
