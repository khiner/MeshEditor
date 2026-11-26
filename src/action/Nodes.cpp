#include "action/Nodes.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "procedure/Cook.h"
#include "procedure/Macros.h"
#include "procedure/NodeView.h"
#include "procedure/OpNodes.h"
#include "render/MaterialComponents.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"
#include "state/EntityReferences.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"

namespace action::nodes {
namespace {
using namespace procedure;

InlineSelection MeshSelection(const state::Registry &r, state::Entity mesh_entity) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto id = *DrawnStoreId(r, mesh_entity);
    InlineSelection selection{.Domain = r.get<const EditMode>(state::Document).Value};
    for (const auto [element, captured] : {std::pair{Element::Vertex, &selection.Vertices}, std::pair{Element::Edge, &selection.Edges}, std::pair{Element::Face, &selection.Faces}}) {
        const auto selected = meshes.GetSelectedElements(id, element);
        std::vector<SelectedBlock> blocks;
        selected.ForEachBlock([&](uint32_t block, uint32_t) {
            SelectionBlock words;
            std::ranges::copy(selected.Bits.subspan(block * MeshElementBlockWords, MeshElementBlockWords), words.begin());
            blocks.push_back({selected.Membership[block].Local, words});
        });
        *captured = CompactSelection(std::move(blocks));
    }
    return selection;
}
// A kind with a Reference captures the mesh's active vertex in Vertex mode, and no vertex otherwise.
void CaptureSelection(state::Registry &r, state::Entity node, state::Entity mesh_entity) {
    r.emplace_or_replace<InlineSelection>(node, MeshSelection(r, mesh_entity));
    const auto *active = r.try_get<const MeshActiveElement>(mesh_entity);
    const auto reference = active && r.get<const EditMode>(state::Document).Value == Element::Vertex ? active->Handle : InvalidOffset;
    VisitKind(r.get<const ProcedureNode>(node).Kind, [&]<typename K> {
        if constexpr (requires { &K::Reference; }) r.patch<K>(node, [&](K &p) { p.Reference = reference; });
    });
}

// The subnet node's selection port takes the drawn selection as its own selection.
void CaptureSubnetSelection(state::Registry &r, state::Entity subnet, uint16_t port, state::Entity mesh_entity) {
    r.patch<Subnet>(subnet, [&](Subnet &s) {
        auto found = std::ranges::lower_bound(s.Ports, port, {}, &InstancePort::Id);
        if (found == s.Ports.end() || found->Id != port) found = s.Ports.insert(found, {.Id = port});
        found->Selection = MeshSelection(r, mesh_entity);
    });
}

// The registry and network holding the node occurrences that the subnet nodes of the path enclose in the mesh's procedure, the mesh itself for an empty path, or none for a path that is not one.
std::optional<DefinitionRef> NetworkAt(const state::Registry &r, state::Entity mesh_entity, std::span<const state::Entity> path) {
    DefinitionRef at{&r, mesh_entity};
    for (const auto subnet : path) {
        const auto *record = at.Registry->valid(subnet) ? at.Registry->try_get<const ProcedureNode>(subnet) : nullptr;
        if (!record || record->Network != at.Entity || !at.Registry->all_of<Subnet>(subnet)) return {};
        at = InstanceDefinition(*at.Registry, subnet);
    }
    return at;
}
// The network NetworkAt finds, or null for a path that is not one or that enters a built-in definition.
state::Entity NetworkAlong(const state::Registry &r, state::Entity mesh_entity, std::span<const state::Entity> path) {
    const auto at = NetworkAt(r, mesh_entity, path);
    return at && at->Registry == &r ? at->Entity : state::Null;
}

// Whether the handle parameter of the node's occurrence in the subnet nodes of `path` reads no link and no port value.
bool HandleWritable(const state::Registry &r, std::span<const state::Entity> path, state::Entity node, HandleParameter parameter) {
    const auto *record = r.try_get<const ProcedureNode>(node);
    if (!record || r.all_of<NodeBypassed>(node) || uint8_t(parameter) >= HandleSpecs.size()) return false;
    const auto &spec = HandleSpecs[uint8_t(parameter)];
    if (record->Kind != spec.Kind) return false;
    const auto sockets = KindSockets(record->Kind);
    for (uint32_t socket = 0; socket < InputCount[record->Kind]; ++socket) {
        if (!spec.Reads(sockets[socket])) continue;
        const auto input = ResolveSocket(r, {path.begin(), path.end()}, InputOf(r, node, uint16_t(socket)), true);
        if (input.Source.FromNode != state::Null || input.Constant) return false;
    }
    return true;
}

// A copy's link to a node that is not copied is removed.
const LinkMap Unlinked = [](SocketLink) { return SocketLink{}; };

// A plan with a cycle orders fewer nodes than it holds.
bool Cyclic(const CookPlan &plan) { return plan.Order.size() != plan.Nodes.size(); }
// The plans an edit of the network must keep acyclic: the network's own and each mesh's that instances it.
std::vector<state::Entity> PlanOwners(const state::Registry &r, state::Entity network) {
    auto planned = MeshesUsing(r, network);
    if (!std::ranges::contains(planned, network)) planned.push_back(network);
    return planned;
}
// The socket a display of the node shows, its port or else its result socket, when it outputs geometry or instances.
std::optional<uint16_t> DisplayedSocket(const state::Registry &r, state::Entity node, std::optional<uint16_t> port) {
    const auto socket = port ? port : ResultSocket(r, node);
    if (!socket) return {};
    const auto descriptor = SocketOf(r, node, *socket);
    return descriptor.Direction == SocketDirection::Out && (descriptor.Types & ResultTypes) ? socket : std::nullopt;
}
bool IsNodeOf(const state::Registry &r, state::Entity network, state::Entity node) {
    const auto *record = r.valid(node) ? r.try_get<const ProcedureNode>(node) : nullptr;
    return record && record->Network == network;
}
bool ListsNodes(const state::Registry &r, state::Entity network, std::span<const state::Entity> nodes) {
    if (!std::ranges::all_of(nodes, [&](state::Entity node) { return IsNodeOf(r, network, node); })) return false;
    std::vector<state::Entity> sorted{nodes.begin(), nodes.end()};
    std::ranges::sort(sorted);
    return std::ranges::adjacent_find(sorted) == sorted.end();
}
std::vector<state::Entity> SelectedNodes(const state::Registry &r, state::Entity network) {
    std::vector<state::Entity> selected;
    if (const auto *active = r.try_get<const NetworkActive>(network))
        for (const auto node : active->Selection)
            if (IsNodeOf(r, network, node)) selected.push_back(node);
    return selected;
}
void Select(state::Registry &r, state::Entity network, std::span<const state::Entity> nodes, state::Entity active) {
    NetworkActive selection{active, {nodes.begin(), nodes.end()}};
    std::ranges::sort(selection.Selection);
    r.emplace_or_replace<NetworkActive>(network, std::move(selection));
}

std::optional<uint16_t> PrimaryInput(uint16_t kind) {
    if (!ResultSocket(kind)) return {};
    const auto sockets = KindSockets(kind);
    for (uint16_t s = 0; s < InputCount[kind]; ++s)
        if (sockets[s].Types & ResultTypes) return s;
    return {};
}

// The subnet nodes at which the document's input socket lies on the displayed result's primary input path, which continues through the ports of subnet nodes.
std::optional<NodePath> OnDisplayPath(const state::Registry &r, const GeometryProcedure &procedure, state::Entity node, uint16_t socket) {
    auto path = procedure.DisplayPath;
    auto display = DisplayLink(r, procedure);
    std::vector<std::pair<NodePath, SocketLink>> visited;
    while (display.FromNode != state::Null) {
        if (std::ranges::any_of(visited, [&](const auto &v) { return v.second == display && v.first == path; })) return {};
        visited.emplace_back(path, display);
        const auto &at = RegistryAt(r, path);
        const bool own = &at == &r;
        const auto from = display.FromNode;
        if (at.all_of<SubnetDefinition>(from)) {
            if (path.empty()) return {};
            const auto subnet = path.back();
            path.pop_back();
            const auto &outer = RegistryAt(r, path);
            if (&outer == &r && subnet == node && display.FromSocket == socket) return path;
            display = InputOf(outer, subnet, display.FromSocket);
        } else if (at.all_of<Subnet>(from) && at.all_of<NodeBypassed>(from)) {
            const auto input = SubnetBypassInput(at, from, display.FromSocket);
            if (!input) return {};
            if (own && from == node && *input == socket) return path;
            display = InputOf(at, from, *input);
        } else if (at.all_of<Subnet>(from)) {
            const auto definition = InstanceDefinition(at, from);
            const auto *port = PortOf(at, from, display.FromSocket);
            if (!port) return {};
            path.push_back(from);
            if (definition.Registry == &r && definition.Entity == node && display.FromSocket == socket) return path;
            display = port->Source();
        } else {
            const auto input = PrimaryInput(at.get<const ProcedureNode>(from).Kind);
            if (!input) return {};
            if (own && from == node && *input == socket) return path;
            display = InputOf(at, from, *input);
        }
    }
    return {};
}

// The display that follows a link of the displayed result into a new consumer: the end of the chain the consumer starts, shown through the output ports that pass it, in their subnet nodes.
std::optional<std::pair<NodePath, SocketLink>> ExtendedDisplay(const state::Registry &r, const GeometryProcedure &procedure, const CookPlan &plan, const SocketEdit &edit) {
    const auto display = ResolveSocket(r, procedure.DisplayPath, DisplayLink(r, procedure), false);
    if (display.Source.FromNode == state::Null || OnDisplayPath(r, procedure, edit.Node, edit.Socket)) return {};
    auto shown_path = display.Path;
    shown_path.push_back(display.Source.FromNode);
    const auto shown = IndexOf(plan, shown_path);
    if (shown == NoNode) return {};
    const auto count = uint32_t(plan.Nodes.size());
    std::vector<uint32_t> next(count, NoNode);
    uint32_t start = NoNode;
    for (uint32_t n = 0; n < count; ++n) {
        const auto input = plan.Nodes[n].Kind == KindOf<Subnet> ? std::nullopt : PrimaryInput(plan.Nodes[n].Kind);
        if (!input) continue;
        const auto link = plan.Inputs[plan.InputFirst[n] + *input];
        if (link.FromNode == NoNode) continue;
        if (ResultSocket(plan.Nodes[link.FromNode].Kind) != link.FromSocket) continue;
        auto &reader = next[link.FromNode];
        reader = reader == NoNode ? n : count;
        if (link.FromNode != shown || link.FromSocket != display.Source.FromSocket) continue;
        // The consumer whose input the edit makes read the displayed result.
        NodePath path{std::from_range, PathOf(plan, n)};
        path.pop_back();
        const auto before = ResolveSocket(r, path, InputOf(Authored(r, plan.Nodes[n]), plan.Nodes[n].Entity, *input), false);
        if (before.Path != display.Path || before.Source != display.Source) start = start == NoNode ? n : count;
    }
    if (start >= count) return {};
    while (next[start] < count) start = next[start];
    NodePath path{std::from_range, PathOf(plan, start)};
    path.pop_back();
    SocketLink output{plan.Nodes[start].Entity, *ResultSocket(plan.Nodes[start].Kind)};
    const auto result_path = path;
    const auto result = output;
    while (!path.empty()) {
        const auto subnet = path.back();
        auto outer = path;
        outer.pop_back();
        const auto &at = RegistryAt(r, outer);
        std::optional<uint16_t> port;
        for (const auto &p : InterfaceOf(at, subnet)->Ports) {
            if (p.Direction != SocketDirection::Out) continue;
            const auto resolved = ResolveSocket(r, outer, {subnet, p.Id}, false, &edit);
            if (resolved.Path != result_path || resolved.Source != result) continue;
            if (port) return std::pair{path, output};
            port = p.Id;
        }
        if (!port) break;
        output = {subnet, *port};
        path = std::move(outer);
    }
    return std::pair{path, output};
}

template<typename K> bool Rename(state::Registry &r, state::Entity node, std::string_view name) {
    if (!r.valid(node) || !r.all_of<K>(node)) return false;
    r.patch<K>(node, [&](K &p) {
        p.Name = {};
        std::ranges::copy(name, p.Name.begin());
    });
    return true;
}
SetDisplay DisplayOf(state::Entity mesh_entity, const NodePath &path, SocketLink link, const state::Registry &at) {
    return {mesh_entity, link.FromNode, link.FromNode != state::Null && at.all_of<Subnet>(link.FromNode) ? std::optional{link.FromSocket} : std::nullopt, path};
}
} // namespace

void Apply(state::Registry &r, const Action &action) {
    // The network, or false after failing the action.
    const auto network_of = [&](state::Entity network) {
        if (!IsNodeNetwork(r, network)) Fail(r, "The network is not a geometry procedure, a user definition or a material's graph");
        return IsNodeNetwork(r, network);
    };
    // Whether the nodes include a material's Surface node, which stays in its graph, after failing the action with one.
    const auto holds_surface = [&](std::span<const state::Entity> nodes) {
        if (std::ranges::none_of(nodes, [&](state::Entity node) { return r.get<const ProcedureNode>(node).Kind == KindOf<Surface>; })) return false;
        Fail(r, "A material's graph keeps its Surface node");
        return true;
    };
    const auto kind_of = [&](state::Entity node) { return r.get<const ProcedureNode>(node).Kind; };
    std::visit(
        overloaded{
            [&](const AddNode &a) {
                if (!network_of(a.Network)) return;
                // The added node, or null after failing the action.
                const auto fail = [&](std::string message) {
                    Fail(r, std::move(message));
                    return state::Null;
                };
                const auto node = std::visit(
                    overloaded{
                        [&](const KindNode &spec) {
                            if (spec.Kind >= KindCount) return fail("Unknown node kind");
                            if (r.all_of<MaterialGraph>(a.Network)) return fail("Add op nodes to a material's graph");
                            bool added = false, operable = !spec.Operation;
                            VisitKind(spec.Kind, [&]<typename K> {
                                added = Kind<K>.Added;
                                if constexpr (Operated<K>) operable |= spec.Operation && std::ranges::contains(field::EnumValues<decltype(K::Operation)>, decltype(K::Operation)(*spec.Operation));
                            });
                            if (!added) return fail("Add an op node by its MaterialX node, and a Surface node with a material's graph");
                            if (!operable) return fail("Operation is not one of the node kind's operations");
                            if (spec.Kind == KindOf<Subnet>) return CreateSubnet(r, a.Network, {CreateDefinition(r, "Subnet")});
                            const auto created = CreateNode(r, spec.Kind);
                            if (spec.Operation) VisitKind(spec.Kind, [&]<typename K> {
                                if constexpr (Operated<K>) r.patch<K>(created, [&](K &p) { p.Operation = decltype(K::Operation)(*spec.Operation); });
                            });
                            if (spec.Kind == KindOf<Source>) r.emplace<SourceState>(created);
                            AddToNetwork(r, a.Network, {created, spec.Kind}, std::vector<SocketLink>(InputCount[spec.Kind]));
                            return created;
                        },
                        [&](const OpNode &spec) {
                            const auto *definition = DefinitionOf(spec.Node);
                            if (!definition || definition->Sockets.size() == definition->Inputs) return fail("Name a MaterialX node with a value output");
                            const auto &lib = mx::Libraries();
                            const auto created = r.create();
                            r.emplace<Op>(created, spec.Node, definition->StoresOutput ? std::optional{lib.Declarations[mx::NodeDeclarations(lib, spec.Node).front()].Outputs.front().Type} : std::nullopt);
                            AddToNetwork(r, a.Network, {created, KindOf<Op>}, std::vector<SocketLink>(definition->Inputs));
                            return created;
                        },
                        [&](const MacroNode &spec) {
                            if (!r.any_of<GeometryProcedure, SubnetDefinition>(a.Network)) return fail("Add a macro to a geometry network");
                            if (spec.Macro >= Macros.size()) return fail("Unknown macro");
                            return CreateSubnet(r, a.Network, {state::Null, Macros[spec.Macro].Name});
                        },
                        [&](const DefinitionNode &spec) {
                            if (!r.valid(spec.Definition) || !r.all_of<SubnetDefinition>(spec.Definition)) return fail("The definition does not exist");
                            // A definition never contains itself, directly or through other definitions.
                            if (Uses(r, spec.Definition, a.Network)) return fail("A definition cannot contain itself");
                            return CreateSubnet(r, a.Network, {spec.Definition});
                        },
                    },
                    a.Node
                );
                if (node != state::Null) Select(r, a.Network, std::array{node}, node);
            },
            [&](const Link &a) {
                if (!network_of(a.Network)) return;
                // A definition's own ports are its input ports, which nodes read, and its output ports, which nodes link into.
                const auto here = [&](state::Entity node) { return node == a.Network ? r.all_of<SubnetDefinition>(node) : IsNodeOf(r, a.Network, node); };
                if (!here(a.FromNode) || !here(a.ToNode)) return Fail(r, "Connect nodes of one network through their ports");
                if ((r.any_of<Subnet, SubnetDefinition>(a.FromNode) ? !PortOf(r, a.FromNode, a.FromSocket) : a.FromSocket >= SocketsOf(r, a.FromNode).size()) ||
                    (r.any_of<Subnet, SubnetDefinition>(a.ToNode) ? !PortOf(r, a.ToNode, a.ToSocket) : a.ToSocket >= InputCountOf(r, a.ToNode))) return Fail(r, "A link refers to a missing socket");
                // A definition's port read inside its network faces the other way.
                const auto facing = [&](state::Entity node, uint16_t socket) {
                    const auto *port = PortOf(r, node, socket);
                    if (!port || node != a.Network) return SocketOf(r, node, socket);
                    return PortSocket(r, node, *port, port->Direction == SocketDirection::In ? SocketDirection::Out : SocketDirection::In);
                };
                const auto from = facing(a.FromNode, a.FromSocket);
                const auto to = facing(a.ToNode, a.ToSocket);
                if (from.Direction != SocketDirection::Out || to.Direction != SocketDirection::In || !Links(from, to)) return Fail(r, "Link compatible output and input sockets");
                // A multi-input socket that already has this source moves that link to Index.
                // Otherwise, a multi-input socket that has links gets the new link in a new slot.
                const auto links = InputLinks(r, a.ToNode, a.ToSocket);
                const SocketLink source{a.FromNode, a.FromSocket};
                if (to.Multi && std::ranges::contains(links, source)) return InsertInput(r, a.ToNode, source, a.Index);
                const SocketEdit edit{a.ToNode, to.Multi && !links.empty() ? uint16_t(r.get<const NodeInputs>(a.ToNode).Links.size()) : a.ToSocket, source};
                // Each plan of the network must stay acyclic, and each resolved link must admit its source.
                std::vector<std::pair<state::Entity, std::pair<NodePath, SocketLink>>> displays;
                for (const auto owner : PlanOwners(r, a.Network)) {
                    const auto plan = Plan(r, owner, false, &edit);
                    // The bypasses in effect resolve other sources, which must stay acyclic too.
                    if (Cyclic(plan) || Cyclic(Plan(r, owner, true, &edit))) return Fail(r, "A link cannot close a cycle");
                    for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
                        if (plan.Nodes[n].Kind == KindOf<Subnet>) continue;
                        for (uint16_t slot = 0; slot < InputsOf(plan, n); ++slot) {
                            const auto &link = plan.Inputs[plan.InputFirst[n] + slot];
                            if (link.FromNode == NoNode) continue;
                            if (!Links(NodeSockets(plan, link.FromNode)[link.FromSocket], NodeSockets(plan, n)[SlotSocket(plan.Nodes[n].Kind, slot)])) return Fail(r, "A field cannot link into an input that takes a single value");
                        }
                    }
                    if (const auto *procedure = r.try_get<const GeometryProcedure>(owner)) {
                        if (auto display = ExtendedDisplay(r, *procedure, plan, edit)) displays.emplace_back(owner, std::move(*display));
                    }
                }
                if (to.Multi) InsertInput(r, a.ToNode, edit.Source, a.Index);
                else SetInput(r, a.ToNode, a.ToSocket, edit.Source);
                for (const auto &[mesh, display] : displays) Apply(r, Action{DisplayOf(mesh, display.first, display.second, RegistryAt(r, display.first))});
            },
            [&](const Unlink &a) {
                if (!network_of(a.Network)) return;
                if (!(a.ToNode == a.Network || IsNodeOf(r, a.Network, a.ToNode)) || (r.any_of<Subnet, SubnetDefinition>(a.ToNode) ? !PortOf(r, a.ToNode, a.ToSocket) : a.ToSocket >= InputCountOf(r, a.ToNode))) return Fail(r, "An unlink refers to a missing input");
                const auto links = InputLinks(r, a.ToNode, a.ToSocket).size();
                if (a.Index > 0u && a.Index >= links) return Fail(r, "Link index is out of range for the input");
                const auto slot = r.any_of<Subnet, SubnetDefinition>(a.ToNode) ? a.ToSocket : SlotOf(kind_of(a.ToNode), a.ToSocket, a.Index);
                const auto source = InputOf(r, a.ToNode, slot);
                // Move the display to the source only when this is the socket's last link.
                // A definition's input port read passes the link into its subnet node's port.
                std::vector<SetDisplay> displays;
                for (const auto mesh : links == 1u ? MeshesUsing(r, a.Network) : std::vector<state::Entity>{}) {
                    auto path = OnDisplayPath(r, r.get<const GeometryProcedure>(mesh), a.ToNode, a.ToSocket);
                    if (!path) continue;
                    auto display = source;
                    while (display.FromNode != state::Null && RegistryAt(r, *path).all_of<SubnetDefinition>(display.FromNode) && !path->empty()) {
                        const auto subnet = path->back();
                        path->pop_back();
                        display = InputOf(RegistryAt(r, *path), subnet, display.FromSocket);
                    }
                    if (display.FromNode != state::Null && !RegistryAt(r, *path).all_of<SubnetDefinition>(display.FromNode)) displays.push_back(DisplayOf(mesh, *path, display, RegistryAt(r, *path)));
                }
                SetInput(r, a.ToNode, slot, {});
                for (const auto &display : displays) Apply(r, Action{display});
            },
            [&](const SetDisplay &a) {
                const auto *procedure = r.try_get<const GeometryProcedure>(a.MeshEntity);
                if (!procedure) return Fail(r, "The mesh has no geometry procedure");
                std::optional<uint16_t> port;
                if (a.Node != state::Null) {
                    const auto at = NetworkAt(r, a.MeshEntity, a.Path);
                    const auto *record = at && at->Registry->valid(a.Node) ? at->Registry->try_get<const ProcedureNode>(a.Node) : nullptr;
                    if (!record || record->Network != at->Entity) return Fail(r, "A display refers to a missing node");
                    const auto &held = *at->Registry;
                    const auto socket = DisplayedSocket(held, a.Node, a.Port);
                    if (!socket) return Fail(r, "Only geometry or instances can be displayed");
                    if (held.all_of<Subnet>(a.Node)) port = socket;
                }
                const auto path = a.Node == state::Null ? std::vector<state::Entity>{} : a.Path;
                if (procedure->DisplayNode != a.Node || procedure->DisplayPort != port || procedure->DisplayPath != path) {
                    r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &edited) {
                        edited.DisplayNode = a.Node;
                        edited.DisplayPort = port;
                        edited.DisplayPath = path;
                    });
                }
            },
            [&](const SelectItems &a) {
                if (!r.valid(a.Network)) return Fail(r, "The network does not exist");
                auto items = a.Items;
                std::ranges::sort(items);
                if (std::ranges::adjacent_find(items) != items.end()) return Fail(r, "The selection must list distinct items");
                if (!IsNodeNetwork(r, a.Network)) return Fail(r, "Select nodes of a geometry network or a material's graph");
                const auto valid = [&](state::Entity node) { return IsNodeOf(r, a.Network, node) || (node == a.Network && r.all_of<SubnetDefinition>(node)); };
                if (!std::ranges::all_of(items, valid) || (a.Active != state::Null && !valid(a.Active))) return Fail(r, "Select nodes of the network");
                r.emplace_or_replace<NetworkActive>(a.Network, a.Active != state::Null ? a.Active : ActiveNodeOf(r, a.Network), std::move(items));
            },
            [&](const MoveNodes &a) {
                if (!r.valid(a.Network)) return Fail(r, "The network does not exist");
                if (!std::ranges::all_of(a.Positions, [&](const auto &p) { return ShowsItem(r, a.Network, p.Node) && (!p.Output || p.Node == a.Network); })) return Fail(r, "A layout item is outside this network");
                for (const auto &p : a.Positions) SetNetworkPosition(r, a.Network, p.Node, p.Position, p.Output);
            },
            [&](const AutoLayout &a) {
                if (!r.valid(a.Network)) return Fail(r, "The network does not exist");
                r.remove<NetworkLayout>(a.Network);
            },
            [&](const UnplaceNode &a) {
                if (!r.valid(a.Network)) return Fail(r, "The network does not exist");
                ClearNetworkPosition(r, a.Network, a.Node, a.Output);
            },
            [&](const SetExtrudeOffset &a) {
                if (!r.all_of<procedure::Extrude>(a.Node)) return Fail(r, "The operation is not an extrusion");
                const auto *handle = r.try_get<const OperationHandle>(state::Document);
                if (!HandleWritable(r, handle && handle->Node == a.Node ? std::span{handle->Path} : std::span<const state::Entity>{}, a.Node, HandleParameter::ExtrudeOffset)) return Fail(r, "Disconnect the extrusion offset inputs before editing the offset handle");
                r.patch<procedure::Extrude>(a.Node, [&](auto &p) {
                    p.Offset = a.Value;
                    p.OffsetScale = 1.f;
                    p.AlongNormals = false;
                });
            },
            [&](const EditSelection &a) {
                const auto *node = r.try_get<const ProcedureNode>(a.Node);
                const auto *object = r.try_get<const Instance>(a.Object);
                if (!node || !object || r.all_of<SelectionEditing>(state::Document)) return Fail(r, "Selection editing needs a node and its scene placement");
                if (!r.all_of<GeometryProcedure>(object->Entity) || node->Network != NetworkAlong(r, object->Entity, a.Path)) return Fail(r, "Selection editing needs the node's occurrence in the object's procedure");
                // The input slot whose geometry the selection is edited over, and the selection being edited.
                std::optional<uint16_t> geometry;
                std::optional<InlineSelection> current;
                if (a.Port) {
                    const auto *port = r.all_of<Subnet>(a.Node) ? PortOf(r, a.Node, *a.Port) : nullptr;
                    if (!port || !port->Selection || port->Direction != SocketDirection::In || InputOf(r, a.Node, *a.Port).FromNode != state::Null) return Fail(r, "Selection editing needs an unconnected selection port");
                    for (const auto &p : InterfaceOf(r, a.Node)->Ports) {
                        if (p.Direction == SocketDirection::In && p.Types == TypeBit(DataType::Geometry)) {
                            geometry = p.Id;
                            break;
                        }
                    }
                    if (const auto *own = InstancePortOf(r, a.Node, *a.Port)) current = own->Selection;
                } else {
                    const auto sockets = SocketsOf(r, a.Node);
                    const auto selection = std::ranges::find(sockets, true, &Socket::Selection);
                    if (selection == sockets.end() || InputOf(r, a.Node, uint16_t(selection - sockets.begin())).FromNode != state::Null) return Fail(r, "Selection editing needs an unconnected Selection input");
                    if (sockets.front().Types == TypeBit(DataType::Geometry)) geometry = 0u;
                    if (const auto *inline_selection = r.try_get<const InlineSelection>(a.Node)) current = *inline_selection;
                }
                if (!geometry) return Fail(r, "Selection editing needs a geometry input");
                const auto *world = WorldTransformOf(r, a.Object);
                const auto *membership = RootMembership(r, a.Object);
                if (!world || !membership) return Fail(r, "Selection editing needs a placed scene object");
                const auto id = SelectionInput(r, object->Entity, a.Path, a.Node, *geometry);
                if (id == InvalidOffset) return Fail(r, "Connect input geometry before editing selection");
                const auto mesh = r.create(), placement = r.create();
                r.emplace<EvaluatedMesh>(mesh, id, NodePath{});
                r.emplace<MeshGeometryDirty>(mesh, EditSelectionAfter::Keep);
                r.emplace<MeshElementSelection>(mesh);
                r.emplace<SceneMembership>(placement, *membership);
                r.emplace<Transform>(placement, *world);
                r.emplace<Instance>(placement, mesh);
                std::optional<InlineSelection> prior;
                if (r.all_of<MeshElementSelection>(object->Entity)) {
                    prior = MeshSelection(r, object->Entity);
                    prior->Domain = r.Resources.get<const MeshStore>().GetSelectionSummary(*DrawnStoreId(r, object->Entity)).Mode;
                }
                r.emplace<SelectionEditing>(state::Document, a.Node, object->Entity, mesh, placement, r.get<const Interaction>(state::Document).Mode, r.get<const EditMode>(state::Document).Value, std::move(prior), a.Port);
                r.patch<Interaction>(state::Document, [](auto &s) { s.Mode = InteractionMode::Edit; });
                r.patch<EditMode>(state::Document, [&](auto &s) { if (current) s.Value = current->Domain; });
                SetMeshSelection(r, mesh, current ? &*current : nullptr);
                if (a.Port || r.get<const EditMode>(state::Document).Value != Element::Vertex) return;
                VisitKind(node->Kind, [&]<typename K> {
                    if constexpr (requires { &K::Reference; }) {
                        const auto reference = r.get<const K>(a.Node).Reference;
                        auto &meshes = r.Resources.get<MeshStore>();
                        if (!meshes.IsLiveElement(id, Element::Vertex, ElementHandle{reference})) return;
                        meshes.WriteSelectionSummary(id).ActiveHandle = reference;
                        r.emplace<MeshActiveElement>(mesh, reference);
                    }
                });
            },
            [&](FinishSelection) {
                if (!r.all_of<SelectionEditing>(state::Document)) return;
                const auto saved = std::move(r.edit<SelectionEditing>(state::Document));
                if (saved.PriorSelection) SetMeshSelection(r, saved.Procedure, &*saved.PriorSelection);
                r.remove<SelectionEditing>(state::Document);
                r.destroy(saved.Placement);
                r.destroy(saved.Mesh);
                r.patch<Interaction>(state::Document, [&](auto &s) { s.Mode = saved.PriorMode; });
                r.patch<EditMode>(state::Document, [&](auto &s) { s.Value = saved.PriorElement; });
            },
            [&](const SetOperationHandle &a) {
                if (a.Node == state::Null) {
                    r.remove<OperationHandle>(state::Document);
                    return;
                }
                const auto *node = r.try_get<const ProcedureNode>(a.Node);
                const auto *object = r.try_get<const Instance>(a.Object);
                if (!node || !object || !r.all_of<GeometryProcedure>(object->Entity) || node->Network != NetworkAlong(r, object->Entity, a.Path) || !HandleWritable(r, a.Path, a.Node, a.Parameter)) {
                    Fail(r, "The operation needs a scene placement and a cooked input for its viewport handle");
                    return;
                }
                PrepareOperationHandle(r, object->Entity, a.Path, a.Node);
                r.emplace_or_replace<OperationHandle>(state::Document, a.Node, a.Object, a.Parameter, a.Path);
            },
            [&](const SetBypass &a) {
                const auto *node = r.try_get<const ProcedureNode>(a.Node);
                if (!node) return Fail(r, "The node does not exist");
                if (node->Kind == KindOf<Subnet>) {
                    const auto *interface = InterfaceOf(r, a.Node);
                    const auto &ports = interface ? interface->Ports : std::vector<SubnetPort>{};
                    if (a.Enabled && (std::ranges::none_of(ports, [](const auto &p) { return p.Direction == SocketDirection::Out; }) || std::ranges::any_of(ports, [&](const auto &p) { return p.Direction == SocketDirection::Out && !SubnetBypassInput(r, a.Node, p.Id); }))) return Fail(r, "Each bypassed output needs one compatible input");
                } else if (!BypassInput(node->Kind)) return Fail(r, "The node has no compatible bypass input");
                const bool bypassed = r.all_of<NodeBypassed>(a.Node);
                if (a.Enabled == bypassed) return;
                // A bypass changes what the node's outputs read, which must leave every plan of its network acyclic.
                const auto set = [&](bool enabled) {
                    if (enabled) r.emplace<NodeBypassed>(a.Node);
                    else r.remove<NodeBypassed>(a.Node);
                };
                set(a.Enabled);
                if (std::ranges::any_of(PlanOwners(r, node->Network), [&](state::Entity owner) { return Cyclic(Plan(r, owner, true)); })) {
                    set(bypassed);
                    return Fail(r, "The bypass would close a cycle");
                }
            },
            [&](const SetAttributeName &a) {
                if (a.Name.empty() || a.Name.size() >= AttributeName{}.size() || a.Name.front() == '.') return Fail(r, "An attribute name needs 1 to 63 characters and no leading '.'");
                if (!Rename<StoreNamedAttribute>(r, a.Node, a.Name) && !Rename<StoreSelection>(r, a.Node, a.Name)) Fail(r, "Only attribute name nodes take a name");
            },
            [&](const SetInlineSelection &a) {
                const auto *node = r.try_get<const ProcedureNode>(a.Node);
                if (!node || !std::ranges::contains(SocketsOf(r, a.Node), true, &Socket::Selection)) return Fail(r, "Only a node with a selection input takes an inline selection");
                r.emplace_or_replace<InlineSelection>(a.Node, *a.Selection);
            },
            [&](const CollapseProcedure &a) {
                const auto *evaluated = r.all_of<GeometryProcedure>(a.MeshEntity) ? r.try_get<const EvaluatedMesh>(a.MeshEntity) : nullptr;
                if (!evaluated) return;
                // The new Source node takes the evaluated record's state where its history step ends, and the mesh keeps drawing it.
                const auto record = evaluated->StoreId;
                const auto nodes = NodesOf(r, a.MeshEntity);
                // Detach published instance lists before destroying the nodes to retain the objects.
                if (r.all_of<ProcedureInstances>(a.MeshEntity)) r.replace<ProcedureInstances>(a.MeshEntity);
                if (r.all_of<ProcedureOwnedMesh>(a.MeshEntity)) r.replace<ProcedureOwnedMesh>(a.MeshEntity, state::Null);
                r.remove<ProcedureInstances, ProcedureOwnedMesh>(a.MeshEntity);
                const auto source = CreateNode(r, KindOf<Source>);
                r.emplace<SourceState>(source, store::Hash128{}, record);
                RootProcedure(r, a.MeshEntity, source, KindOf<Source>);
                r.replace<EvaluatedMesh>(a.MeshEntity, record, NodePath{source});
                r.destroy(nodes);
            },
            [&](const DuplicateNodes &a) {
                if (!network_of(a.Network)) return;
                if (a.Nodes.empty() || !ListsNodes(r, a.Network, a.Nodes)) return Fail(r, "A duplicate lists distinct nodes of the network");
                if (holds_surface(a.Nodes)) return;
                // A copied Source node's state refers to the original's geometry, so a pending state is first kept as it stands.
                FinishSourceStates(r);
                std::vector<state::Entity> originals;
                for (const auto node : NodesOf(r, a.Network))
                    if (std::ranges::contains(a.Nodes, node)) originals.push_back(node);
                const auto active = ActiveNodeOf(r, a.Network);
                const auto copies = CopyInto(r, a.Network, r, originals, Unlinked);
                std::vector<state::Entity> selected;
                for (const auto node : originals) selected.push_back(copies(node));
                const auto copied_active = copies(active);
                Select(r, a.Network, selected, copied_active != state::Null ? copied_active : copies(originals.back()));
            },
            [&](const DeleteNodes &a) {
                if (!network_of(a.Network)) return;
                if (a.Nodes.empty() || !ListsNodes(r, a.Network, a.Nodes)) return Fail(r, "A delete lists distinct nodes of the network");
                if (holds_surface(a.Nodes)) return;
                const auto nodes = NodesOf(r, a.Network);
                auto removed_sorted = a.Nodes;
                std::ranges::sort(removed_sorted);
                const auto is_removed = [&](state::Entity node) { return std::ranges::binary_search(removed_sorted, node); };
                // The socket of the value a link's source passes.
                const auto carried = [&](SocketLink link) {
                    const auto *port = PortOf(r, link.FromNode, link.FromSocket);
                    return port ? PortSocket(r, link.FromNode, *port, SocketDirection::Out) : SocketOf(r, link.FromNode, link.FromSocket);
                };
                // A removed subnet node passes the link into its bypass input, and any other removed node the one link into it whose value matches its output.
                const auto through = [&](this const auto &self, SocketLink link) -> SocketLink {
                    if (link.FromNode == state::Null || !is_removed(link.FromNode)) return link;
                    if (r.all_of<Subnet>(link.FromNode)) {
                        const auto input = SubnetBypassInput(r, link.FromNode, link.FromSocket);
                        return input ? self(InputOf(r, link.FromNode, *input)) : SocketLink{};
                    }
                    const auto output = SocketOf(r, link.FromNode, link.FromSocket);
                    SocketLink source{};
                    uint32_t sources = 0;
                    for (const auto input : r.get<const NodeInputs>(link.FromNode).Links) {
                        const auto read = self(input);
                        if (read.FromNode == state::Null) continue;
                        const auto value = carried(read);
                        if (!(value.Types & output.Types) || value.Varies != output.Varies) continue;
                        source = read;
                        ++sources;
                    }
                    return sources == 1 ? source : SocketLink{};
                };
                const auto reconnected = [&](state::Entity node, uint16_t socket, SocketLink link) {
                    const auto source = through(link);
                    const auto *port = PortOf(r, node, socket);
                    const auto into = port ? PortSocket(r, node, *port, SocketDirection::In) : SocketOf(r, node, socket);
                    return source.FromNode != state::Null && Links(carried(source), into) ? source : SocketLink{};
                };
                for (const auto node : nodes) {
                    if (is_removed(node)) continue;
                    if (const auto *subnet = r.try_get<const Subnet>(node)) {
                        const auto ports = subnet->Ports;
                        for (const auto &port : ports) {
                            if (is_removed(port.Source.FromNode)) SetInput(r, node, port.Id, reconnected(node, port.Id, port.Source));
                        }
                    } else {
                        auto inputs = r.get<const NodeInputs>(node).Links;
                        bool changed = false;
                        for (uint16_t i = 0; i < inputs.size(); ++i) {
                            if (!is_removed(inputs[i].FromNode)) continue;
                            inputs[i] = reconnected(node, SlotSocketOf(r, node, i), inputs[i]);
                            changed = true;
                        }
                        if (changed) r.replace<NodeInputs>(node, NodeInputs{NormalizedInputs(kind_of(node), std::move(inputs))});
                    }
                }
                if (r.all_of<SubnetDefinition>(a.Network)) {
                    const auto ports = r.get<const SubnetInterface>(a.Network).Ports;
                    for (const auto &port : ports)
                        if (port.Direction == SocketDirection::Out && is_removed(port.Source().FromNode)) SetInput(r, a.Network, port.Id, through(port.Source()));
                }
                // A display whose occurrence holds a removed node moves to what passes through it, at the removed node's place.
                for (const auto mesh : MeshesUsing(r, a.Network)) {
                    const auto &procedure = r.get<const GeometryProcedure>(mesh);
                    const auto shown = DisplayedPath(procedure);
                    const auto at = std::ranges::find_if(shown, is_removed);
                    if (at == shown.end()) continue;
                    NodePath path{shown.begin(), at};
                    const auto node = *at;
                    auto display = through(at + 1 == shown.end() ? DisplayLink(r, procedure) : SocketLink{node, ResultSocket(r, node).value_or(0u)});
                    if (display.FromNode == state::Null && !r.all_of<Subnet>(node) && !r.get<const NodeInputs>(node).Links.empty()) display = through(r.get<const NodeInputs>(node).Links.front());
                    while (display.FromNode != state::Null && display.FromNode == a.Network && !path.empty()) {
                        const auto subnet = path.back();
                        path.pop_back();
                        display = InputOf(RegistryAt(r, path), subnet, display.FromSocket);
                    }
                    if (display.FromNode == a.Network) display = {};
                    // A display that would show a value other than geometry or instances shows nothing.
                    const auto &held = RegistryAt(r, path);
                    if (display.FromNode != state::Null && !DisplayedSocket(held, display.FromNode, held.all_of<Subnet>(display.FromNode) ? std::optional{display.FromSocket} : std::nullopt)) display = {};
                    Apply(r, Action{DisplayOf(mesh, path, display, held)});
                }
                std::vector<state::Entity> remaining;
                for (const auto node : nodes)
                    if (!is_removed(node)) remaining.push_back(node);
                OrderNetwork(r, a.Network, remaining);
                auto active = ActiveNodeOf(r, a.Network);
                if (!std::ranges::contains(nodes, active) || is_removed(active)) {
                    const auto *procedure = r.try_get<const GeometryProcedure>(a.Network);
                    const auto display = procedure && procedure->DisplayPath.empty() ? procedure->DisplayNode : state::Null;
                    active = display != state::Null ? display : remaining.empty() ? state::Null : remaining.back();
                }
                auto selected = SelectedNodes(r, a.Network);
                std::erase_if(selected, is_removed);
                Select(r, a.Network, selected, active);
                r.destroy(removed_sorted);
            },
            [&](const InsertNode &a) {
                if (!network_of(a.Network)) return;
                if (!IsNodeOf(r, a.Network, a.Node) || !(a.ToNode == a.Network || IsNodeOf(r, a.Network, a.ToNode))) return Fail(r, "An insert refers to a missing node or input");
                const auto view = ProjectNetwork(r, a.Network, {});
                const auto count = uint32_t(view.Nodes.size());
                // A link into a definition's output port is a link into the page's output boundary.
                const auto node = NodeOf(view, a.Node), to_node = a.ToNode == a.Network && count > 0u && OutputBoundary(view, count - 1u) ? count - 1u : NodeOf(view, a.ToNode);
                if (node == NoNode || to_node == NoNode) return Fail(r, "Insert within one network");
                // A subnet node's or a boundary's slots are its input ports in page order, and any other node's slots are those of its NodeInputs.
                const auto slots = SlotsOf(view, to_node);
                const auto slot = [&] {
                    if (r.any_of<Subnet, SubnetDefinition>(a.ToNode)) {
                        const auto ids = std::span{view.SocketIds}.subspan(view.SocketFirst[to_node], slots);
                        return uint16_t(std::ranges::find(ids, a.ToSocket) - ids.begin());
                    }
                    const auto to_kind = kind_of(a.ToNode);
                    return a.ToSocket < InputCountOf(r, a.ToNode) && (a.Index == 0u || a.ToSocket == MultiInput[to_kind]) ? SlotOf(to_kind, a.ToSocket, a.Index) : slots;
                }();
                if (slot >= slots || !Inserts(view, node, to_node, slot)) return Fail(r, "Insert needs a node with unlinked geometry sockets that are compatible with the link, and must not create a cycle");
                if (a.Shift) {
                    // The reader and each node downstream of it, in an order that visits each node after the nodes it reads.
                    std::vector<bool> moved(count);
                    moved[to_node] = true;
                    const auto edges = [&](auto visit) {
                        for (uint32_t n = 0u; n < count; ++n)
                            for (auto i = view.InputFirst[n]; i < view.InputFirst[n + 1u]; ++i)
                                if (const auto from = view.Wires[i].FromNode; from != NoNode) visit(from, n);
                    };
                    for (const auto n : DependencyOrder(count, edges)) {
                        for (auto i = view.InputFirst[n]; i < view.InputFirst[n + 1u] && !moved[n]; ++i) moved[n] = view.Wires[i].FromNode != NoNode && moved[view.Wires[i].FromNode];
                    }
                    for (uint32_t n = 0u; n < count; ++n) {
                        const auto item = view.Nodes[n].Item;
                        const bool output = OutputBoundary(view, n);
                        if (const auto cell = moved[n] ? NetworkPosition(r, a.Network, item, output) : std::nullopt) SetNetworkPosition(r, a.Network, item, {cell->x + a.Shift, cell->y}, output);
                    }
                }
                const auto kind = view.Nodes[node].Kind;
                const auto link = view.Wires[view.InputFirst[to_node] + slot];
                SetInput(r, a.Node, *MainSocket(kind, SocketDirection::In), {view.Nodes[link.FromNode].Item, SocketId(view, link.FromNode, link.FromSocket)});
                SetInput(r, a.ToNode, r.any_of<Subnet, SubnetDefinition>(a.ToNode) ? a.ToSocket : slot, {a.Node, *MainSocket(kind, SocketDirection::Out)});
            },
            [&](const AddMaterialGraph &a) {
                if (!r.valid(a.Material) || !r.all_of<Material>(a.Material)) return Fail(r, "The material does not exist");
                if (r.all_of<MaterialGraph>(a.Material)) return Fail(r, "The material already has a node graph");
                r.emplace<MaterialGraph>(a.Material);
                r.emplace<NetworkNodes>(a.Material);
                const auto surface = CreateNode(r, KindOf<Surface>);
                AddToNetwork(r, a.Material, {surface, KindOf<Surface>}, std::vector<SocketLink>(InputCount[KindOf<Surface>]));
                Select(r, a.Material, std::array{surface}, surface);
            },
            [&](const auto &a) { Apply(r, a); },
        },
        action
    );
}

void CaptureActiveSelection(state::Registry &r, state::Entity mesh_entity) {
    const auto *task = r.try_get<const SelectionEditing>(state::Document);
    if (!task || task->Mesh != mesh_entity) return;
    if (task->Port) CaptureSubnetSelection(r, task->Node, *task->Port, mesh_entity);
    else CaptureSelection(r, task->Node, mesh_entity);
}

state::Entity CopyProcedure(state::Registry &r, state::Entity from, uint32_t drawn) {
    FinishSourceStates(r);
    const auto procedure = r.get<const GeometryProcedure>(from);
    const auto originals = NodesOf(r, from);
    const auto to = r.create();
    r.emplace<GeometryProcedure>(to);
    r.emplace<NetworkNodes>(to);
    const auto copies = CopyInto(r, to, r, originals, Unlinked);
    // The copy's occurrences are the originals' with their outermost node copied.
    const auto copied = [&](NodePath path) {
        if (!path.empty()) path.front() = copies(path.front());
        return path;
    };
    for (const auto node : originals) {
        const auto copy = copies(node);
        if (const auto *result = r.try_get<const NodeResult>(node)) r.emplace<NodeResult>(copy, NodeResult{copied(result->Input), result->State, result->StoreId});
        if (const auto *placement = r.try_get<const BooleanPlacement>(node)) r.emplace<BooleanPlacement>(copy, BooleanPlacement{*placement});
        if (const auto *results = r.try_get<const NodeResults>(node)) {
            NodeResults kept;
            for (const auto &[path, result] : results->Values) kept.Values.emplace(path, NodeResult{copied(result.Input), result.State, result.StoreId});
            r.emplace<NodeResults>(copy, std::move(kept));
        }
        if (const auto *placements = r.try_get<const BooleanPlacements>(node)) r.emplace<BooleanPlacements>(copy, BooleanPlacements{*placements});
    }
    auto display = copied(DisplayedPath(procedure));
    r.replace<GeometryProcedure>(to, display.empty() ? state::Null : display.back(), procedure.DisplayPort, display.empty() ? NodePath{} : NodePath{display.begin(), display.end() - 1});
    if (const auto *layout = r.try_get<const NetworkLayout>(from)) {
        auto copied_layout = *layout;
        for (auto &cell : copied_layout.Cells) cell.Item = copies(cell.Item);
        std::ranges::sort(copied_layout.Cells, {}, [](const NetworkCell &cell) { return std::pair{cell.Item, cell.Output}; });
        r.emplace<NetworkLayout>(to, std::move(copied_layout));
    }
    auto selected = SelectedNodes(r, from);
    for (auto &node : selected) node = copies(node);
    Select(r, to, selected, copies(ActiveNodeOf(r, from)));
    r.emplace<EvaluatedMesh>(to, drawn, copied(r.get<const EvaluatedMesh>(from).Tip));
    return to;
}

namespace {
// The displayed geometry result and the network that holds its node, or a null link after failing without one.
struct Displayed {
    SocketLink Link;
    state::Entity Network{state::Null};
    NodePath Path;
};
Displayed DisplayedGeometry(state::Registry &r, state::Entity mesh_entity) {
    const auto &procedure = r.get<const GeometryProcedure>(mesh_entity);
    const auto network = NetworkAlong(r, mesh_entity, procedure.DisplayPath);
    if (network == state::Null) {
        Fail(r, "Make the built-in subnet unique to append to its network");
        return {};
    }
    const auto link = DisplayLink(r, procedure);
    if (link.FromNode != state::Null && Links(SocketOf(r, link.FromNode, link.FromSocket), Socket{TypeBit(DataType::Geometry), SocketDirection::In})) return {link, network, procedure.DisplayPath};
    Fail(r, "Display a node with a geometry result to append to it");
    return {};
}
// The mesh's displayed node, when its network is the mesh's or a user definition's, or null.
state::Entity DisplayedNode(const state::Registry &r, state::Entity mesh_entity) {
    const auto &procedure = r.get<const GeometryProcedure>(mesh_entity);
    return procedure.DisplayNode != state::Null && &RegistryAt(r, procedure.DisplayPath) == &r ? procedure.DisplayNode : state::Null;
}
} // namespace

void Apply(state::Registry &r, const AppendMacro &a) {
    if (a.Macro >= Macros.size()) return Fail(r, "Unknown macro");
    for (const auto mesh_entity : mesh::SelectedEditMeshes(r)) {
        const auto set_values = [&](state::Entity subnet) {
            uint32_t next = 0u;
            for (const auto &port : InterfaceOf(r, subnet)->Ports) {
                if (!ReadsDefault(port) || next >= a.Values.size()) continue;
                Apply(r, Action{SetPortValue{subnet, port.Id, a.Values[next++]}});
            }
        };
        if (const auto staged = StagedResultNode(r, mesh_entity, KindOf<Subnet>); staged != state::Null && r.get<const Subnet>(staged).Builtin == Macros[a.Macro].Name) {
            set_values(staged);
            continue;
        }
        const auto displayed = DisplayedGeometry(r, mesh_entity);
        if (displayed.Link.FromNode == state::Null) return;
        Apply(r, Action{AddNode{displayed.Network, MacroNode{a.Macro}}});
        const auto subnet = ActiveNodeOf(r, displayed.Network);
        if (subnet == state::Null || !r.all_of<Subnet>(subnet)) return;
        const auto ports = InterfaceOf(r, subnet)->Ports;
        const auto first_port = [&](auto &&matches) { return std::ranges::find_if(ports, matches); };
        if (const auto p = first_port([](const auto &port) { return port.Direction == SocketDirection::In && port.Types == TypeBit(DataType::Geometry); }); p != ports.end()) {
            Apply(r, Action{Link{displayed.Network, displayed.Link.FromNode, displayed.Link.FromSocket, subnet, p->Id}});
        }
        if (const auto p = first_port([](const auto &port) { return port.Direction == SocketDirection::In && port.Selection; }); p != ports.end()) CaptureSubnetSelection(r, subnet, p->Id, mesh_entity);
        r.emplace_or_replace<PublishSelection>(subnet);
        set_values(subnet);
        Apply(r, Action{SetDisplay{mesh_entity, subnet, {}, displayed.Path}});
        r.emplace_or_replace<StagedNode>(subnet);
        r.remove<MeshActiveElement>(mesh_entity);
    }
}

state::Entity AppendNode(state::Registry &r, state::Entity mesh_entity, uint16_t kind) {
    const auto displayed = DisplayedGeometry(r, mesh_entity);
    if (displayed.Link.FromNode == state::Null) return state::Null;
    Apply(r, Action{AddNode{displayed.Network, KindNode{kind}}});
    const auto entity = ActiveNodeOf(r, displayed.Network);
    if (std::ranges::contains(KindSockets(kind), true, &Socket::Selection)) CaptureSelection(r, entity, mesh_entity);
    Apply(r, Link{displayed.Network, displayed.Link.FromNode, displayed.Link.FromSocket, entity, 0u});
    Apply(r, SetDisplay{mesh_entity, entity, {}, displayed.Path});
    r.emplace<PublishSelection>(entity);
    return entity;
}

state::Entity StagedResultNode(const state::Registry &r, state::Entity mesh_entity, uint16_t kind) {
    const auto display = DisplayedNode(r, mesh_entity);
    return display != state::Null && r.get<const ProcedureNode>(display).Kind == kind && r.all_of<StagedNode>(display) ? display : state::Null;
}

void ForEachResultNode(const state::Registry &r, const std::function<void(state::Entity)> &fn) {
    for (const auto mesh_entity : mesh::SelectedEditMeshes(r)) {
        if (!r.all_of<GeometryProcedure>(mesh_entity)) continue;
        if (const auto display = DisplayedNode(r, mesh_entity); display != state::Null) fn(display);
    }
}
} // namespace action::nodes
