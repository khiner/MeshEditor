#include "Variant.h"
#include "action/Core.h"
#include "action/Errors.h"
#include "action/Nodes.h"
#include "graph/NodeView.h"
#include "graph/OpNodes.h"
#include "mesh/MeshComponents.h"
#include "scene/Entity.h"
#include "scene/Instance.h"
#include "state/Registry.h"

#include <numeric>

namespace graph {
namespace {
// Whether the value a port passes varies, which its resolved source decides.
bool PortVaries(const state::Registry &r, state::Entity use, uint16_t port) {
    const auto resolved = ResolveSocket(r, {use, port});
    return resolved.Source.FromNode != state::Null && SocketOf(*resolved.Registry, resolved.Source.FromNode, resolved.Source.FromSocket).Varies;
}
// The non-null links of the kind's multi-input socket, in order and without duplicates.
std::vector<SocketLink> MultiLinks(uint16_t kind, std::span<const SocketLink> links) {
    std::vector<SocketLink> listed;
    for (uint16_t slot = MultiInput[kind]; slot < links.size(); slot = NextMultiSlot(kind, slot)) {
        if (links[slot].FromNode != state::Null && !std::ranges::contains(listed, links[slot])) listed.push_back(links[slot]);
    }
    return listed;
}
void SetMultiLinks(uint16_t kind, std::vector<SocketLink> &links, std::span<const SocketLink> listed) {
    links.resize(InputCount[kind] + std::max<size_t>(listed.size(), 1u) - 1u);
    links[MultiInput[kind]] = listed.empty() ? SocketLink{} : listed.front();
    std::ranges::copy(listed | std::views::drop(1), links.begin() + InputCount[kind]);
}
} // namespace

const state::Registry &RegistryAt(const state::Registry &r, std::span<const state::Entity> path) {
    const auto *at = &r;
    for (const auto use : path) at = SubgraphOf(*at, use).Registry;
    return *at;
}

SubgraphRef SubgraphOf(const state::Registry &r, state::Entity node) {
    return std::visit(overloaded{
                          [&](state::Entity subgraph) { return SubgraphRef{&r, subgraph}; },
                          [](BuiltinSubgraphId builtin) { return BuiltinSubgraph(builtin); },
                      },
                      r.get<const SubgraphNode>(node).SubgraphId);
}
state::Entity FirstNode(const state::Registry &r, state::Entity graph) { return r.get<const GraphNodes>(graph).First; }
std::vector<state::Entity> NodesOf(const state::Registry &r, state::Entity graph) {
    std::vector<state::Entity> nodes;
    for (auto node = FirstNode(r, graph); node != state::Null; node = r.get<const GraphNode>(node).Next) nodes.push_back(node);
    return nodes;
}
bool Uses(const state::Registry &r, state::Entity graph, state::Entity subgraph) {
    if (graph == subgraph) return true;
    for (auto node = FirstNode(r, graph); node != state::Null; node = r.get<const GraphNode>(node).Next) {
        const auto *use = r.try_get<const SubgraphNode>(node);
        if (const auto used = use ? UserSubgraph(use->SubgraphId) : state::Null; used != state::Null && Uses(r, used, subgraph)) return true;
    }
    return false;
}
std::vector<std::pair<state::Entity, state::Entity>> SubgraphUses(const state::Registry &r) {
    std::vector<std::pair<state::Entity, state::Entity>> uses;
    for (const auto mesh : SortedEntities(r.view<const GeometryGraph>())) {
        std::vector<state::Entity> pending{mesh}, reached;
        while (!pending.empty()) {
            const auto graph = pending.back();
            pending.pop_back();
            for (auto node = FirstNode(r, graph); node != state::Null; node = r.get<const GraphNode>(node).Next) {
                const auto *use = r.try_get<const SubgraphNode>(node);
                const auto used = use ? UserSubgraph(use->SubgraphId) : state::Null;
                if (used == state::Null || std::ranges::contains(reached, used)) continue;
                reached.push_back(used);
                pending.push_back(used);
                uses.emplace_back(used, mesh);
            }
        }
    }
    std::ranges::sort(uses);
    return uses;
}
std::vector<state::Entity> MeshesUsing(const state::Registry &r, std::span<const std::pair<state::Entity, state::Entity>> uses, state::Entity graph) {
    if (r.all_of<GeometryGraph>(graph)) return {graph};
    const auto [first, last] = std::ranges::equal_range(uses, graph, {}, &std::pair<state::Entity, state::Entity>::first);
    return std::ranges::subrange(first, last) | std::views::values | std::ranges::to<std::vector>();
}
std::vector<state::Entity> MeshesUsing(const state::Registry &r, state::Entity graph) {
    return r.all_of<GeometryGraph>(graph) ? std::vector{graph} : MeshesUsing(r, SubgraphUses(r), graph);
}
const SubgraphNodePort *NodePortOf(const state::Registry &r, state::Entity node, uint16_t port) {
    const auto &ports = r.get<const SubgraphNode>(node).Ports;
    const auto found = std::ranges::find(ports, port, &SubgraphNodePort::PortId);
    return found == ports.end() ? nullptr : &*found;
}
std::string SubgraphNodeName(const state::Registry &r, state::Entity use) {
    const auto subgraph = SubgraphOf(r, use);
    return subgraph.Entity == state::Null ? std::string{} : GetName(*subgraph.Registry, subgraph.Entity);
}
state::Entity GraphOf(const state::Registry &r, state::Entity node) { return r.get<const GraphNode>(node).Graph; }
const SubgraphInterface *InterfaceOf(const state::Registry &r, state::Entity use) {
    if (r.all_of<SubgraphNode>(use)) {
        const auto subgraph = SubgraphOf(r, use);
        return subgraph.Entity == state::Null ? nullptr : subgraph.Registry->try_get<const SubgraphInterface>(subgraph.Entity);
    }
    if (const auto *instance = r.try_get<const Instance>(use)) return r.try_get<const SubgraphInterface>(instance->SubgraphEntity);
    return r.try_get<const SubgraphInterface>(use);
}
const SubgraphPort *PortOf(const state::Registry &r, state::Entity use, uint16_t port) {
    const auto *interface = InterfaceOf(r, use);
    if (!interface) return nullptr;
    const auto found = std::ranges::find(interface->Ports, port, &SubgraphPort::Id);
    return found == interface->Ports.end() ? nullptr : &*found;
}
Socket PortSocket(const state::Registry &r, state::Entity use, const SubgraphPort &port, SocketDirection facing) {
    return {.Types = port.Types, .Direction = facing, .Varies = facing == SocketDirection::In || PortVaries(r, use, port.Id), .Name = port.Name.c_str(), .Selection = port.Selection};
}
Socket SocketOf(const state::Registry &r, state::Entity node, uint16_t socket) {
    if (const auto *port = PortOf(r, node, socket)) return PortSocket(r, node, *port, port->Direction);
    if (!r.all_of<GraphNode>(node) || socket >= SocketsOf(r, node).size()) return {DataTypes{}, SocketDirection::In};
    auto descriptor = SocketsOf(r, node)[socket];
    if (descriptor.Direction == SocketDirection::Out && r.all_of<Op>(node)) descriptor.Varies = OpVaries(r, node);
    return descriptor;
}
std::span<const Socket> SocketsOf(const state::Registry &r, state::Entity node) {
    if (const auto *op = r.try_get<const Op>(node)) {
        const auto *definition = DefinitionOf(op->Node);
        return definition ? std::span{definition->Sockets} : std::span<const Socket>{};
    }
    return KindSockets(r.get<const GraphNode>(node).Kind);
}
uint16_t InputCountOf(const state::Registry &r, state::Entity node) {
    if (const auto *op = r.try_get<const Op>(node)) {
        const auto *definition = DefinitionOf(op->Node);
        return definition ? definition->Inputs : uint16_t{};
    }
    return InputCount[r.get<const GraphNode>(node).Kind];
}
uint16_t SlotSocketOf(const state::Registry &r, state::Entity node, uint16_t slot) { return SlotSocket(r.get<const GraphNode>(node).Kind, slot); }
void ForEachInput(const state::Registry &r, state::Entity node, const std::function<void(uint16_t, SocketLink)> &f) {
    if (const auto *use = r.try_get<const SubgraphNode>(node)) {
        for (const auto ports = use->Ports; const auto &port : ports) f(port.PortId, port.Source);
        return;
    }
    const auto links = r.get<const NodeInputs>(node).Links;
    for (uint16_t slot = 0; slot < links.size(); ++slot) f(slot, links[slot]);
}
void ForEachGraphInput(const state::Registry &r, state::Entity graph, const std::function<void(state::Entity, uint16_t, SocketLink)> &f) {
    for (const auto node : NodesOf(r, graph)) ForEachInput(r, node, [&](uint16_t slot, SocketLink link) { f(node, slot, link); });
    if (!r.all_of<Subgraph>(graph)) return;
    for (const auto ports = r.get<const SubgraphInterface>(graph).Ports; const auto &port : ports)
        if (port.Direction == SocketDirection::Out) f(graph, port.Id, port.Source());
}
std::optional<uint16_t> SubgraphBypassInput(const state::Registry &r, state::Entity use, uint16_t output) {
    const auto *port = PortOf(r, use, output);
    if (!port || port->Direction != SocketDirection::Out) return {};
    std::optional<uint16_t> input;
    for (const auto &p : InterfaceOf(r, use)->Ports) {
        if (p.Direction != SocketDirection::In || p.Types != port->Types || p.Selection != port->Selection || PortVaries(r, use, p.Id) != PortVaries(r, use, output)) continue;
        if (input) return {};
        input = p.Id;
    }
    return input;
}
std::optional<uint16_t> ResultSocket(const state::Registry &r, state::Entity node) {
    if (r.all_of<SubgraphNode>(node)) {
        const auto *interface = InterfaceOf(r, node);
        if (!interface) return {};
        for (const auto &port : interface->Ports) {
            if (port.Direction == SocketDirection::Out && (port.Types & ResultTypes)) return port.Id;
        }
        return {};
    }
    const auto *record = r.try_get<const GraphNode>(node);
    return record ? ResultSocket(record->Kind) : std::optional<uint16_t>{};
}
SocketLink DisplayLink(const state::Registry &r, const GeometryGraph &graph) {
    if (graph.DisplayNode == state::Null) return {};
    const auto &at = RegistryAt(r, graph.DisplayPath);
    const auto socket = graph.DisplayPort ? graph.DisplayPort : ResultSocket(at, graph.DisplayNode);
    return socket ? SocketLink{graph.DisplayNode, *socket} : SocketLink{};
}
NodePath DisplayedPath(const GeometryGraph &graph) {
    if (graph.DisplayNode == state::Null) return {};
    auto path = graph.DisplayPath;
    path.push_back(graph.DisplayNode);
    return path;
}
std::vector<SocketLink> NormalizedInputs(uint16_t kind, std::vector<SocketLink> links) {
    if (MultiInput[kind] != NoSocket) SetMultiLinks(kind, links, MultiLinks(kind, links));
    return links;
}
void SetInput(state::Registry &r, state::Entity node, uint16_t slot, SocketLink source) {
    if (r.all_of<SubgraphNode>(node)) {
        r.patch<SubgraphNode>(node, [&](SubgraphNode &use) {
            auto found = std::ranges::lower_bound(use.Ports, slot, {}, &SubgraphNodePort::PortId);
            if (found == use.Ports.end() || found->PortId != slot) found = use.Ports.insert(found, {.PortId = slot});
            found->Source = source;
            if (source.FromNode == state::Null && !found->Value && !found->Selection) use.Ports.erase(found);
        });
    } else if (r.all_of<Subgraph>(node)) {
        r.patch<SubgraphInterface>(node, [&](auto &interface) {
            for (auto &port : interface.Ports)
                if (port.Id == slot) port.Source() = source;
        });
    } else {
        const auto kind = r.get<const GraphNode>(node).Kind;
        r.patch<NodeInputs>(node, [&](auto &inputs) {
            inputs.Links[slot] = source;
            inputs.Links = NormalizedInputs(kind, std::move(inputs.Links));
        });
    }
}
void InsertInput(state::Registry &r, state::Entity node, SocketLink source, uint16_t index) {
    const auto kind = r.get<const GraphNode>(node).Kind;
    r.patch<NodeInputs>(node, [&](auto &inputs) {
        auto listed = MultiLinks(kind, inputs.Links);
        std::erase(listed, source);
        listed.insert(listed.begin() + std::min<size_t>(index, listed.size()), source);
        SetMultiLinks(kind, inputs.Links, listed);
    });
}
SocketLink InputOf(const state::Registry &r, state::Entity node, uint16_t slot) {
    if (r.all_of<SubgraphNode>(node)) {
        const auto *port = NodePortOf(r, node, slot);
        return port ? port->Source : SocketLink{};
    }
    if (r.all_of<Subgraph>(node)) {
        const auto *port = PortOf(r, node, slot);
        return port ? port->Source() : SocketLink{};
    }
    return r.get<const NodeInputs>(node).Links[slot];
}
std::vector<SocketLink> InputLinks(const state::Registry &r, state::Entity node, uint16_t socket) {
    if (!r.any_of<SubgraphNode, Subgraph>(node) && socket == MultiInput[r.get<const GraphNode>(node).Kind]) return MultiLinks(r.get<const GraphNode>(node).Kind, r.get<const NodeInputs>(node).Links);
    const auto link = InputOf(r, node, socket);
    return link.FromNode == state::Null ? std::vector<SocketLink>{} : std::vector{link};
}
ResolvedSocket ResolveSocket(const state::Registry &r, NodePath path, SocketLink link, bool bypass, const SocketEdit *edit) {
    std::vector<std::pair<NodePath, SocketLink>> visited;
    const auto *at = &RegistryAt(r, path);
    // An edit changes a link of the document, never one of a built-in subgraph.
    const auto edited = [&](state::Entity node, uint16_t socket) { return edit && at == &r && edit->Node == node && edit->Socket == socket; };
    while (link.FromNode != state::Null) {
        const auto from = link.FromNode;
        const bool use = at->all_of<SubgraphNode>(from), subgraph = at->all_of<Subgraph>(from);
        if (!use && !subgraph && !(bypass && at->all_of<NodeBypassed>(from))) return {std::move(path), at, link};
        if (std::ranges::any_of(visited, [&](const auto &v) { return v.second == link && v.first == path; })) return {.Cycle = true};
        visited.emplace_back(path, link);
        if (!use && !subgraph) {
            const auto kind = at->get<const GraphNode>(from).Kind;
            const auto input = BypassInput(kind);
            if (!input || ResultSocket(kind) != link.FromSocket) return {};
            link = edited(from, *input) ? edit->Source : at->get<const NodeInputs>(from).Links[*input];
            continue;
        }
        // The subgraph node whose input port the link reads, and the port.
        auto subgraph_node = from;
        auto port_id = link.FromSocket;
        if (subgraph) {
            // A subgraph's input port reads the input port of the subgraph node the path entered the subgraph through.
            if (path.empty()) return {};
            subgraph_node = path.back();
            path.pop_back();
            at = &RegistryAt(r, path);
        } else {
            const auto *port = PortOf(*at, from, link.FromSocket);
            if (!port) return {};
            if (port->Direction == SocketDirection::Out) {
                if (!bypass || !at->all_of<NodeBypassed>(from)) {
                    const auto inner = SubgraphOf(*at, from);
                    const bool own = inner.Registry == &r && at == &r;
                    link = own && edit && edit->Node == inner.Entity && edit->Socket == port->Id ? edit->Source : port->Source();
                    path.push_back(from);
                    at = inner.Registry;
                    continue;
                }
                const auto input = SubgraphBypassInput(*at, from, port->Id);
                if (!input) return {};
                port_id = *input;
            }
        }
        const auto *port = PortOf(*at, subgraph_node, port_id);
        if (!port) return {};
        const auto *own = NodePortOf(*at, subgraph_node, port_id);
        const auto source = edited(subgraph_node, port_id) ? edit->Source : own ? own->Source :
                                                                                  SocketLink{};
        if (source.FromNode != state::Null) {
            link = source;
            continue;
        }
        if (port->Selection) return own && own->Selection ? ResolvedSocket{.Path = std::move(path), .Registry = at, .SelectionUse = subgraph_node, .SelectionPort = port_id} : ResolvedSocket{};
        if (!ReadsDefault(*port)) return {};
        return {.Path = std::move(path), .Constant = own && own->Value ? own->Value : port->Default, .ValueUse = subgraph_node, .ValuePort = port_id};
    }
    return {};
}
std::optional<NumericValue> ParameterDefault(const state::Registry &r, state::Entity node, uint16_t socket) {
    if (const auto *port = PortOf(r, node, socket)) {
        const auto *own = r.all_of<SubgraphNode>(node) ? NodePortOf(r, node, socket) : nullptr;
        return own && own->Value ? own->Value : port->Default;
    }
    const auto descriptor = SocketOf(r, node, socket);
    // An op input without a literal has the default of the first declaration of its node that has it, unless that reads a geometric property.
    if (const auto *op = r.try_get<const Op>(node)) {
        const std::string_view name{descriptor.Name};
        if (const auto literal = op->Literals.find(name); literal != op->Literals.end()) {
            const auto words = LiteralWords(r, node, literal->first, literal->second);
            if (words.empty() || !(TypeBit(*DataTypeOf(literal->second)) & NumericTypes)) return {};
            NumericValue value{{}, *DataTypeOf(literal->second)};
            std::ranges::copy(words, value.Words.begin());
            return value;
        }
        const auto &lib = mx::Libraries();
        for (const auto d : mx::NodeDeclarations(lib, op->Node)) {
            const auto port = std::ranges::find(lib.Declarations[d].Inputs, name, &mx::Port::Name);
            if (port == lib.Declarations[d].Inputs.end()) continue;
            const auto type = DataTypeOf(port->Type);
            const auto *words = std::get_if<std::vector<uint32_t>>(&port->Default);
            if (port->DefaultGeomProp || !type || !(TypeBit(*type) & NumericTypes) || !words || words->size() != mx::Words(port->Type)) return {};
            NumericValue value{{}, *type};
            std::ranges::copy(*words, value.Words.begin());
            return value;
        }
        return {};
    }
    if (descriptor.Field == NoField) return {};
    std::optional<NumericValue> value;
    VisitKind(r.get<const GraphNode>(node).Kind, [&]<typename K> {
        VisitSocketParameter(r.get<const K>(node), descriptor, [&]<typename F>(const F &field, const FieldSpec &) {
            value = NumericValue{{}, DataType(std::countr_zero(detail::TypesOf<F>()))};
            std::memcpy(value->Words.data(), &field, sizeof(F));
        });
    });
    return value;
}
namespace {
// Drops from each subgraph node and Instance that uses the subgraph its own state of each port the subgraph no longer has.
void DropUsePorts(state::Registry &r, state::Entity subgraph) {
    const auto &ports = r.get<const SubgraphInterface>(subgraph).Ports;
    const auto gone = [&](const auto &own) { return !std::ranges::contains(ports, own.PortId, &SubgraphPort::Id); };
    for (const auto node : SortedEntities(r.view<const SubgraphNode>())) {
        const auto &held = r.get<const SubgraphNode>(node);
        if (UserSubgraph(held.SubgraphId) == subgraph && std::ranges::any_of(held.Ports, gone)) r.patch<SubgraphNode>(node, [&](SubgraphNode &edited) { std::erase_if(edited.Ports, gone); });
    }
    for (const auto instance : SortedEntities(r.view<const Instance>())) {
        const auto &held = r.get<const Instance>(instance);
        if (held.SubgraphEntity == subgraph && std::ranges::any_of(held.PortValues, gone)) r.patch<Instance>(instance, [&](Instance &edited) { std::erase_if(edited.PortValues, gone); });
    }
}
void SetParameterDefault(state::Registry &r, state::Entity node, uint16_t socket, NumericValue value) {
    if (r.all_of<SubgraphNode>(node)) {
        r.patch<SubgraphNode>(node, [&](SubgraphNode &use) {
            auto found = std::ranges::lower_bound(use.Ports, socket, {}, &SubgraphNodePort::PortId);
            if (found == use.Ports.end() || found->PortId != socket) found = use.Ports.insert(found, {.PortId = socket});
            found->Value = value;
        });
        return;
    }
    if (r.all_of<SubgraphInterface>(node)) {
        r.patch<SubgraphInterface>(node, [&](auto &interface) {
            for (auto &port : interface.Ports)
                if (port.Id == socket) port.Default = value;
        });
        return;
    }
    const auto descriptor = SocketOf(r, node, socket);
    if (r.all_of<Op>(node)) {
        if (descriptor.Types & TypeBit(value.Type)) WriteOpLiteral(r, node, descriptor.Name, ValueTypeOf(value.Type), std::span{value.Words}.first(mx::Words(ValueTypeOf(value.Type))));
        return;
    }
    if (descriptor.Field == NoField) return;
    VisitKind(r.get<const GraphNode>(node).Kind, [&]<typename K> {
        if constexpr (field::Walkable<K>) r.patch<K>(node, [&](auto &parameters) { AssignSocketParameter(parameters, descriptor, value.Words); });
    });
}
} // namespace

void OrderGraph(state::Registry &r, state::Entity graph, std::span<const state::Entity> nodes) {
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto next = i + 1u < nodes.size() ? nodes[i + 1u] : state::Null;
        if (const auto &record = r.get<const GraphNode>(nodes[i]); record.Graph != graph || record.Next != next) {
            r.patch<GraphNode>(nodes[i], [&](GraphNode &p) {
                p.Graph = graph;
                p.Next = next;
            });
        }
    }
    r.replace_if_changed(graph, GraphNodes{nodes.empty() ? state::Null : nodes.front(), nodes.empty() ? state::Null : nodes.back()});
}
void AddToGraph(state::Registry &r, state::Entity graph, NodeRecord node, std::vector<SocketLink> links) {
    const auto ends = r.get<const GraphNodes>(graph);
    r.emplace<GraphNode>(node.Entity, graph, state::Null, node.Kind);
    r.emplace<NodeInputs>(node.Entity, std::move(links));
    if (ends.Last != state::Null) r.patch<GraphNode>(ends.Last, [&](GraphNode &record) { record.Next = node.Entity; });
    r.replace<GraphNodes>(graph, ends.First == state::Null ? node.Entity : ends.First, node.Entity);
}
state::Entity CreateSubgraph(state::Registry &r, std::string_view name) {
    const auto subgraph = r.create();
    r.emplace<Subgraph>(subgraph);
    r.emplace<GraphNodes>(subgraph);
    r.emplace<SubgraphInterface>(subgraph);
    EmplaceUniqueName(r, subgraph, name);
    return subgraph;
}
state::Entity CreateSubgraphNode(state::Registry &r, state::Entity graph, SubgraphId subgraph) {
    const auto node = r.create();
    r.emplace<SubgraphNode>(node, subgraph);
    AddToGraph(r, graph, {node, KindOf<SubgraphNode>}, {});
    return node;
}
bool IsNodeGraph(const state::Registry &r, state::Entity graph) { return r.valid(graph) && r.all_of<GraphNodes>(graph); }
state::Entity SurfaceNodeOf(const state::Registry &r, state::Entity material) {
    for (auto node = FirstNode(r, material); node != state::Null; node = r.get<const GraphNode>(node).Next)
        if (r.get<const GraphNode>(node).Kind == KindOf<Surface>) return node;
    return state::Null;
}
state::Entity NodeCopies::operator()(state::Entity source) const {
    const auto at = std::ranges::lower_bound(Items, source, {}, &std::pair<state::Entity, state::Entity>::first);
    return at != Items.end() && at->first == source ? at->second : state::Null;
}
NodeCopies CopyNodes(state::Registry &r, const state::Registry &from, std::span<const state::Entity> originals, const LinkMap &link) {
    NodeCopies copies;
    for (const auto node : originals) copies.Items.emplace_back(node, r.create());
    std::ranges::sort(copies.Items);
    const auto mapped = [&](SocketLink l) { return copies(l.FromNode) != state::Null ? SocketLink{copies(l.FromNode), l.FromSocket} : link(l); };
    for (const auto node : originals) {
        const auto copy = copies(node);
        const auto copy_component = [&]<typename T> {
            const auto *value = from.try_get<const T>(node);
            if (!value) return;
            T copied = *value;
            if constexpr (std::same_as<T, SubgraphNode>) {
                for (auto &port : copied.Ports) port.Source = mapped(port.Source);
            }
            r.emplace<T>(copy, std::move(copied));
        };
        VisitKind(from.get<const GraphNode>(node).Kind, copy_component);
        [&]<typename... T>(TypeList<T...>) { (copy_component.template operator()<T>(), ...); }(TypeList<SourceState, InlineSelection, NodeBypassed, PublishSelection>{});
    }
    return copies;
}
std::vector<SocketLink> CopiedInputs(const state::Registry &from, state::Entity node, const NodeCopies &copies, const LinkMap &link) {
    auto links = from.get<const NodeInputs>(node).Links;
    for (auto &l : links) l = copies(l.FromNode) != state::Null ? SocketLink{copies(l.FromNode), l.FromSocket} : link(l);
    return NormalizedInputs(from.get<const GraphNode>(node).Kind, std::move(links));
}
NodeCopies CopyInto(state::Registry &r, state::Entity graph, const state::Registry &from, std::span<const state::Entity> originals, const LinkMap &link) {
    const auto copies = CopyNodes(r, from, originals, link);
    for (const auto node : originals) AddToGraph(r, graph, {copies(node), from.get<const GraphNode>(node).Kind}, CopiedInputs(from, node, copies, link));
    LayoutGraphValues(r, graph);
    CopyLiterals(r, from, copies);
    return copies;
}
void RewritePaths(state::Registry &r, const std::function<void(NodePath &)> &rewrite) {
    // Each occurrence's state moves to where its rewritten path keeps it.
    const auto move = [&]<typename T> {
        std::vector<std::pair<NodePath, T>> held, moved;
        for (const auto node : SortedEntities(r.view<const T>())) held.emplace_back(NodePath{node}, r.get<const T>(node));
        for (const auto use : SortedEntities(r.view<const Inner<T>>())) {
            for (const auto &[key, value] : r.get<const Inner<T>>(use).Values) {
                NodePath path{use};
                path.append_range(key);
                held.emplace_back(std::move(path), value);
            }
        }
        for (const auto &[path, value] : held) {
            auto to = path;
            auto rewritten = value;
            rewrite(to);
            if constexpr (std::same_as<T, NodeResult>) rewrite(rewritten.Input);
            if (to == path && rewritten == value) continue;
            Drop<T>(r, path);
            // A state whose rewritten path holds no node is of an occurrence that no longer exists.
            bool exists = !std::ranges::contains(to, state::Null);
            if constexpr (std::same_as<T, NodeResult>) exists = exists && !std::ranges::contains(rewritten.Input, state::Null);
            if (exists) moved.emplace_back(std::move(to), std::move(rewritten));
        }
        for (auto &[path, value] : moved) Keep(r, path, std::move(value));
    };
    [&]<typename... T>(TypeList<T...>) { (move.template operator()<T>(), ...); }(OccurrenceStates{});
    for (const auto mesh : SortedEntities(r.view<const EvaluatedMesh>())) {
        auto tip = r.get<const EvaluatedMesh>(mesh).Tip;
        rewrite(tip);
        if (tip != r.get<const EvaluatedMesh>(mesh).Tip) r.patch<EvaluatedMesh>(mesh, [&](EvaluatedMesh &m) { m.Tip = std::move(tip); });
    }
    // A plan's occurrences are rewritten in place, so the next settle pass matches each rewritten occurrence to the one it was.
    // A node's parent is its nearest enclosing occurrence whose rewritten path leads its own, or none.
    // A plan is Derived, so the actions that rewrite paths write it directly.
    for (const auto mesh : SortedEntities(r.view<const CookPlan>())) {
        const auto &plan = r.get<const CookPlan>(mesh);
        std::vector<NodePath> paths(plan.Nodes.size());
        auto nodes = plan.Nodes;
        bool rewritten = false;
        for (uint32_t n = 0; n < nodes.size(); ++n) {
            paths[n] = NodePath{std::from_range, PathOf(plan, n)};
            rewrite(paths[n]);
            const auto leads = [&](uint32_t parent) { return std::ranges::equal(paths[parent], std::span{paths[n]}.first(paths[n].size() - 1u)); };
            auto parent = plan.Nodes[n].Parent;
            while (parent != NoNode && !leads(parent)) parent = plan.Nodes[parent].Parent;
            rewritten = rewritten || nodes[n].Entity != paths[n].back() || nodes[n].Parent != parent;
            nodes[n].Entity = paths[n].back();
            nodes[n].Parent = parent;
        }
        if (rewritten) r.patch<CookPlan>(mesh, [&](CookPlan &p) {
            p.Nodes = std::move(nodes);
            IndexNodes(p);
        });
    }
    for (const auto mesh : SortedEntities(r.view<const GeometryGraph>())) {
        const auto &graph = r.get<const GeometryGraph>(mesh);
        auto shown = DisplayedPath(graph);
        rewrite(shown);
        if (shown == DisplayedPath(graph)) continue;
        r.patch<GeometryGraph>(mesh, [&](GeometryGraph &p) {
            p.DisplayNode = shown.empty() ? state::Null : shown.back();
            p.DisplayPath = shown.empty() ? NodePath{} : NodePath{shown.begin(), shown.end() - 1};
        });
    }
}
} // namespace graph

namespace action::nodes {
namespace {
using namespace graph;

// A subgraph's port identities last while NextId is at most UINT16_MAX, which each action that adds ports checks before it changes anything.
constexpr const char *NoPortIdentities{"The subgraph has no available port identities"};
uint16_t AddPort(state::Registry &r, state::Entity subgraph, Socket descriptor, std::string name, SocketLink source, std::optional<NumericValue> value = {}) {
    const auto next = r.get<const SubgraphInterface>(subgraph).NextId;
    r.patch<SubgraphInterface>(subgraph, [&](auto &interface) {
        interface.Ports.push_back({uint16_t(next), std::move(name), descriptor.Types, descriptor.Selection, descriptor.Direction, source, value});
        ++interface.NextId;
    });
    return uint16_t(next);
}
} // namespace

void Apply(state::Registry &r, const ExposeSocket &a) {
    const auto *node = r.try_get<const GraphNode>(a.Node);
    if (!r.all_of<Subgraph>(a.Subgraph) || !node || node->Graph != a.Subgraph) return Fail(r, "Expose a socket belonging to a node inside the subgraph");
    if (r.all_of<SubgraphNode>(a.Node) ? !PortOf(r, a.Node, a.Socket) : a.Socket >= SocketsOf(r, a.Node).size()) return Fail(r, "The socket does not exist");
    if (r.get<const SubgraphInterface>(a.Subgraph).NextId > UINT16_MAX) return Fail(r, NoPortIdentities);
    const auto descriptor = SocketOf(r, a.Node, a.Socket);
    const auto name = a.Name.empty() ? std::string{descriptor.Name ? descriptor.Name : descriptor.Direction == SocketDirection::In ? "Input" :
                                                                                                                                     "Output"} :
                                       a.Name;
    if (descriptor.Direction == SocketDirection::In && descriptor.Multi) {
        InsertInput(r, a.Node, {a.Subgraph, AddPort(r, a.Subgraph, descriptor, name, {})}, UINT16_MAX);
    } else if (descriptor.Direction == SocketDirection::In) {
        const auto port = AddPort(r, a.Subgraph, descriptor, name, {}, ParameterDefault(r, a.Node, a.Socket));
        SetInput(r, a.Node, a.Socket, {a.Subgraph, port});
    } else AddPort(r, a.Subgraph, descriptor, name, {a.Node, a.Socket});
}

void Apply(state::Registry &r, const Collapse &a) {
    const auto graph = a.Graph;
    if (!IsNodeGraph(r, graph)) return CollapseItems(r, a);
    if (a.Nodes.empty()) return Fail(r, "Collapse one or more nodes");
    auto selected = a.Nodes;
    std::ranges::sort(selected);
    if (std::ranges::adjacent_find(selected) != selected.end()) return Fail(r, "Collapse distinct nodes");
    for (const auto node : selected) {
        if (!r.valid(node) || !r.all_of<GraphNode>(node) || GraphOf(r, node) != graph) return Fail(r, "Collapse nodes of one graph");
        if (r.get<const GraphNode>(node).Kind == KindOf<Surface>) return Fail(r, "Collapse nodes other than the material's Surface node");
    }
    const auto collapsed = [&](state::Entity node) { return std::ranges::binary_search(selected, node); };
    const auto nodes = NodesOf(r, graph);
    // Each link that crosses the subgraph's boundary takes at most one port of the new subgraph.
    size_t crossing = 0u;
    ForEachGraphInput(r, graph, [&](state::Entity node, uint16_t, SocketLink link) { crossing += link.FromNode != state::Null && collapsed(node) != collapsed(link.FromNode); });
    if (crossing > UINT16_MAX) return Fail(r, NoPortIdentities);
    // The moved nodes' literals move to the subgraph's block.
    std::vector<std::pair<state::Entity, OpLiteralWords>> literals;
    for (const auto node : nodes)
        if (collapsed(node) && r.all_of<Op>(node)) literals.emplace_back(node, LiteralsOf(r, node));
    const auto subgraph = CreateSubgraph(r, "Subgraph");
    // The subgraph node takes the first collapsed node's place in the graph's order.
    const auto use = r.create();
    std::vector<state::Entity> kept, moved;
    for (const auto node : nodes) {
        if (collapsed(node) && moved.empty()) kept.push_back(use);
        (collapsed(node) ? moved : kept).push_back(node);
    }
    r.emplace<GraphNode>(use, graph, state::Null, KindOf<SubgraphNode>);
    r.emplace<NodeInputs>(use);
    r.emplace<SubgraphNode>(use, subgraph);
    OrderGraph(r, graph, kept);
    OrderGraph(r, subgraph, moved);
    for (const auto node : moved) ClearGraphPosition(r, graph, node);
    // Links from outside into moved nodes read new input ports, which the subgraph node links to the outside sources.
    for (const auto node : moved) {
        ForEachInput(r, node, [&](uint16_t slot, SocketLink link) {
            if (link.FromNode == state::Null || collapsed(link.FromNode)) return;
            const auto socket = SlotSocketOf(r, node, slot);
            const auto descriptor = SocketOf(r, node, socket);
            const auto port = AddPort(r, subgraph, descriptor, descriptor.Name ? descriptor.Name : "Input", {}, ParameterDefault(r, node, socket));
            SetInput(r, node, slot, {subgraph, port});
            SetInput(r, use, port, link);
        });
    }
    // Links from moved nodes into the rest read output ports bound to the moved nodes' sockets.
    const auto output = [&](SocketLink link) {
        const auto &ports = r.get<const SubgraphInterface>(subgraph).Ports;
        const auto existing = std::ranges::find_if(ports, [&](const auto &p) { return p.Direction == SocketDirection::Out && p.Source() == link; });
        return existing != ports.end() ? existing->Id : AddPort(r, subgraph, SocketOf(r, link.FromNode, link.FromSocket), "Output", link);
    };
    ForEachGraphInput(r, graph, [&](state::Entity node, uint16_t slot, SocketLink link) {
        if (link.FromNode != state::Null && collapsed(link.FromNode)) SetInput(r, node, slot, {use, output(link)});
    });
    if (std::ranges::none_of(r.get<const SubgraphInterface>(subgraph).Ports, [](const auto &port) { return port.Direction == SocketDirection::Out; })) {
        for (const auto node : moved | std::views::reverse) {
            if (const auto socket = ResultSocket(r, node)) {
                output({node, *socket});
                break;
            }
        }
    }
    // Every occurrence of a moved node is now inside the new subgraph node.
    RewritePaths(r, [&](NodePath &path) {
        if (const auto at = std::ranges::find_if(path, collapsed); at != path.end()) path.insert(at, use);
    });
    LayoutSubgraphValues(r, subgraph);
    for (const auto &[node, words] : literals) WriteLiterals(r, node, words);
    LayoutGraphValues(r, graph);
    r.emplace_or_replace<GraphActive>(graph, use, std::vector{use});
}
namespace {
// Replaces the subgraph node with a copy of its subgraph's nodes, and returns the copies in the graph's order.
std::vector<state::Entity> ExpandNode(state::Registry &r, state::Entity use) {
    const auto graph = GraphOf(r, use);
    const auto subgraph = SubgraphOf(r, use);
    const auto ports = r.get<const SubgraphNode>(use).Ports;
    // A read of an input port reads the subgraph node's link into it, and an unlinked port's value moves to each reader.
    const auto port_read = [&](uint16_t id) {
        const auto own = std::ranges::find(ports, id, &SubgraphNodePort::PortId);
        return own != ports.end() ? own->Source : SocketLink{};
    };
    const auto nodes = NodesOf(r, graph);
    const auto position = std::ranges::find(nodes, use);
    const auto &from = *subgraph.Registry;
    const auto originals = NodesOf(from, subgraph.Entity);
    const LinkMap read = [&](SocketLink l) { return l.FromNode == subgraph.Entity ? port_read(l.FromSocket) : SocketLink{}; };
    const auto copies = CopyNodes(r, from, originals, read);
    for (const auto node : originals) AddToGraph(r, graph, {copies(node), from.get<const GraphNode>(node).Kind}, CopiedInputs(from, node, copies, read));
    const auto copied = [&](state::Entity node) { return std::ranges::contains(copies.Items, node, &std::pair<state::Entity, state::Entity>::second); };
    // A read of an output port reads its source's copy, or the link into the input port it passes.
    // A bypassed subgraph's output port reads what the bypass passes.
    const bool bypassed = r.all_of<NodeBypassed>(use);
    const auto out_source = [&](uint16_t id) -> SocketLink {
        if (bypassed) {
            const auto input = SubgraphBypassInput(r, use, id);
            return input ? InputOf(r, use, *input) : SocketLink{};
        }
        const auto *port = PortOf(r, use, id);
        if (!port) return {};
        const auto source = port->Source();
        return source.FromNode == subgraph.Entity ? port_read(source.FromSocket) : SocketLink{copies(source.FromNode), source.FromSocket};
    };
    ForEachGraphInput(r, graph, [&](state::Entity node, uint16_t slot, SocketLink link) {
        if (node != use && !copied(node) && link.FromNode == use) SetInput(r, node, slot, out_source(link.FromSocket));
    });
    // A mesh displaying an output port of the subgraph node, in its own graph or through the subgraph nodes of its display path, displays what the port reads.
    for (const auto mesh : SortedEntities(r.view<const GeometryGraph>())) {
        const auto &graph = r.get<const GeometryGraph>(mesh);
        if (graph.DisplayNode != use || &RegistryAt(r, graph.DisplayPath) != &r) continue;
        const auto shown = graph.DisplayPort ? out_source(*graph.DisplayPort) : SocketLink{};
        r.patch<GeometryGraph>(mesh, [&](GeometryGraph &p) {
            p.DisplayNode = shown.FromNode;
            p.DisplayPort = shown.FromNode != state::Null && r.all_of<SubgraphNode>(shown.FromNode) ? std::optional{shown.FromSocket} : std::nullopt;
            p.DisplayPath = shown.FromNode != state::Null ? p.DisplayPath : NodePath{};
        });
    }
    RewritePaths(r, [&](NodePath &path) {
        const auto at = std::ranges::find(path, use);
        if (at == path.end() || at + 1 == path.end()) return;
        *(at + 1) = copies(*(at + 1));
        path.erase(at);
    });
    // The copies take the subgraph node's place in the graph's order.
    std::vector<state::Entity> order{nodes.begin(), position}, selection;
    for (const auto original : originals) selection.push_back(copies(original));
    order.append_range(selection);
    order.insert(order.end(), position + 1, nodes.end());
    OrderGraph(r, graph, order);
    ClearGraphPosition(r, graph, use);
    LayoutGraphValues(r, graph);
    CopyLiterals(r, from, copies);
    // Each unlinked input port's value then replaces the copied value of each parameter that reads it.
    for (const auto [original, copy] : copies.Items) {
        ForEachInput(from, original, [&](uint16_t slot, SocketLink link) {
            if (link.FromNode != subgraph.Entity || port_read(link.FromSocket).FromNode != state::Null) return;
            const auto resolved = ResolveSocket(r, NodePath{use}, link);
            if (resolved.Constant) SetParameterDefault(r, copy, SlotSocketOf(r, copy, slot), *resolved.Constant);
            if (resolved.SelectionUse != state::Null) r.emplace_or_replace<InlineSelection>(copy, *NodePortOf(r, use, resolved.SelectionPort)->Selection);
        });
    }
    r.destroy(use);
    return selection;
}
} // namespace
void Apply(state::Registry &r, const Expand &a) {
    if (r.valid(a.Node) && r.any_of<SubgraphItems, Instance>(a.Node)) return ExpandItems(r, a);
    const auto expands = [&](state::Entity node) { return r.all_of<SubgraphNode, GraphNode>(node) && SubgraphOf(r, node).Entity != state::Null; };
    if (!r.valid(a.Node) || !r.all_of<SubgraphNode, GraphNode>(a.Node)) return Fail(r, "The subgraph does not exist");
    if (!expands(a.Node)) return Fail(r, "The node's subgraph does not exist");
    const auto graph = GraphOf(r, a.Node);
    auto nodes = ExpandNode(r, a.Node);
    for (size_t i = 0; a.AllLevels && i < nodes.size();) {
        if (!expands(nodes[i])) {
            ++i;
            continue;
        }
        const auto inner = ExpandNode(r, nodes[i]);
        nodes.erase(nodes.begin() + i);
        nodes.insert(nodes.begin() + i, inner.begin(), inner.end());
    }
    // The copy of the subgraph's last node is active.
    const auto active = nodes.empty() ? state::Null : nodes.back();
    std::ranges::sort(nodes);
    r.emplace_or_replace<GraphActive>(graph, active, std::move(nodes));
}
void Apply(state::Registry &r, const MakeUnique &a) {
    if (r.valid(a.Use) && r.all_of<Instance>(a.Use)) return void(MakeInstanceUnique(r, a.Use));
    if (!r.valid(a.Use) || !r.all_of<SubgraphNode>(a.Use)) return Fail(r, "The subgraph node does not exist");
    const auto subgraph = SubgraphOf(r, a.Use);
    if (subgraph.Entity == state::Null) return Fail(r, "The node's subgraph does not exist");
    const auto copy = CreateSubgraph(r, GetName(*subgraph.Registry, subgraph.Entity));
    const auto copies = CopyInto(r, copy, *subgraph.Registry, NodesOf(*subgraph.Registry, subgraph.Entity), [&](SocketLink l) { return l.FromNode == subgraph.Entity ? SocketLink{copy, l.FromSocket} : SocketLink{}; });
    auto interface = subgraph.Registry->get<const SubgraphInterface>(subgraph.Entity);
    for (auto &port : interface.Ports) {
        if (port.Direction != SocketDirection::Out) continue;
        const auto source = port.Source();
        port.Source() = source.FromNode == subgraph.Entity ? SocketLink{copy, source.FromSocket} : SocketLink{copies(source.FromNode), source.FromSocket};
    }
    r.replace<SubgraphInterface>(copy, std::move(interface));
    r.patch<SubgraphNode>(a.Use, [&](SubgraphNode &use) { use.SubgraphId = copy; });
    RewritePaths(r, [&](NodePath &path) {
        const auto at = std::ranges::find(path, a.Use);
        if (at != path.end() && at + 1 != path.end()) *(at + 1) = copies(*(at + 1));
    });
}
bool SubgraphUsed(const state::Registry &r, state::Entity subgraph) {
    return std::ranges::any_of(r.view<const SubgraphNode>(), [&](state::Entity node) { return UserSubgraph(r.get<const SubgraphNode>(node).SubgraphId) == subgraph; });
}
void Apply(state::Registry &r, const DeleteSubgraph &a) {
    if (!r.valid(a.Subgraph) || !r.all_of<Subgraph, GraphNodes>(a.Subgraph)) return Fail(r, "The subgraph does not exist");
    if (SubgraphUsed(r, a.Subgraph)) return Fail(r, "Delete a subgraph that no node uses");
    r.destroy(NodesOf(r, a.Subgraph));
    r.destroy(a.Subgraph);
}
void Apply(state::Registry &r, const SetPortValue &a) {
    const auto *port = PortOf(r, a.Owner, a.Port);
    const auto *field = port ? std::get_if<PromotedField>(&port->Binding) : nullptr;
    if (!field && (!port || !ReadsDefault(*port))) return Fail(r, "This port has no parameter default");
    // An instance holds its own value of a promoted field, and the subgraph's value is the field itself.
    if (field && r.all_of<Instance>(a.Owner)) return SetInstancePortValue(r, a.Owner, a.Port, {a.Words});
    if (field) {
        auto update = action::ReadUpdate(r, field->Entity, field->Component, field->Offset);
        if (!update) return Fail(r, "The promoted field no longer exists");
        std::visit([&]<typename A>(A &value) {
            if constexpr (action::IsUpdate<A>) {
                if constexpr (sizeof(value.Value) <= sizeof(a.Words)) {
                    std::memcpy(&value.Value, a.Words.data(), sizeof(value.Value));
                    action::Apply(r, action::Core{value});
                }
            }
        },
                   *update);
        return;
    }
    const auto type = port->Default ? port->Default->Type : DataType(std::countr_zero(DataTypes(port->Types & NumericTypes)));
    SetParameterDefault(r, a.Owner, a.Port, {a.Words, type});
}
void Apply(state::Registry &r, const ClearPortValue &a) {
    const auto *port = r.any_of<SubgraphNode, Instance>(a.Owner) || r.all_of<Subgraph, GraphNodes>(a.Owner) ? PortOf(r, a.Owner, a.Port) : nullptr;
    if (!port || !(ReadsDefault(*port) || (port->Selection && port->Direction == SocketDirection::In) || (std::holds_alternative<PromotedField>(port->Binding) && r.all_of<Instance>(a.Owner)))) return Fail(r, "Clear the value of an input port of a subgraph node, an instance or a subgraph");
    if (const auto *instance = r.try_get<const Instance>(a.Owner)) {
        if (PortValueOf(*instance, a.Port)) r.patch<Instance>(a.Owner, [&](Instance &held) { std::erase_if(held.PortValues, [&](const PortValue &own) { return own.PortId == a.Port; }); });
    } else if (r.all_of<SubgraphNode>(a.Owner)) {
        const auto *own = NodePortOf(r, a.Owner, a.Port);
        if (!own || (!own->Value && !own->Selection)) return;
        r.patch<SubgraphNode>(a.Owner, [&](SubgraphNode &use) {
            const auto found = std::ranges::find(use.Ports, a.Port, &SubgraphNodePort::PortId);
            if (found->Source.FromNode == state::Null) use.Ports.erase(found);
            else {
                found->Value.reset();
                found->Selection.reset();
            }
        });
    } else if (port->Default) {
        r.patch<SubgraphInterface>(a.Owner, [&](auto &interface) { std::ranges::find(interface.Ports, a.Port, &SubgraphPort::Id)->Default.reset(); });
    }
}
void Apply(state::Registry &r, const RenamePort &a) {
    if (!r.all_of<SubgraphInterface>(a.Subgraph) || !PortOf(r, a.Subgraph, a.Port) || a.Name.empty()) return Fail(r, "Name an existing port of a subgraph");
    r.patch<SubgraphInterface>(a.Subgraph, [&](auto &interface) { for (auto &port : interface.Ports) if (port.Id == a.Port) port.Name = a.Name; });
}
void Apply(state::Registry &r, const MovePort &a) {
    if (!r.all_of<SubgraphInterface>(a.Subgraph)) return Fail(r, "Reorder ports of a subgraph");
    const auto *port = PortOf(r, a.Subgraph, a.Port), *before = PortOf(r, a.Subgraph, a.BeforePort);
    if (!port || !before || port->Direction != before->Direction) return Fail(r, "Reorder ports in the same direction");
    if (a.Port == a.BeforePort) return;
    r.patch<SubgraphInterface>(a.Subgraph, [&](auto &interface) {
        const auto from = std::ranges::find(interface.Ports, a.Port, &SubgraphPort::Id);
        auto moved = std::move(*from);
        interface.Ports.erase(from);
        interface.Ports.insert(std::ranges::find(interface.Ports, a.BeforePort, &SubgraphPort::Id), std::move(moved));
    });
}
void Apply(state::Registry &r, const RemovePort &a) {
    if (!r.all_of<SubgraphInterface>(a.Subgraph) || !PortOf(r, a.Subgraph, a.Port)) return Fail(r, "The port does not exist");
    if (r.all_of<GraphNodes>(a.Subgraph)) {
        // A removed input port's readers keep its value, and every link to a removed port is removed.
        const auto port = *PortOf(r, a.Subgraph, a.Port);
        for (const auto node : NodesOf(r, a.Subgraph)) {
            ForEachInput(r, node, [&](uint16_t slot, SocketLink link) {
                if (link != SocketLink{a.Subgraph, a.Port}) return;
                if (port.Default) SetParameterDefault(r, node, SlotSocketOf(r, node, slot), *port.Default);
                SetInput(r, node, slot, {});
            });
        }
        for (const auto use : SortedEntities(r.view<const SubgraphNode>())) {
            if (UserSubgraph(r.get<const SubgraphNode>(use).SubgraphId) != a.Subgraph) continue;
            ForEachGraphInput(r, GraphOf(r, use), [&](state::Entity node, uint16_t slot, SocketLink link) {
                if (link == SocketLink{use, a.Port}) SetInput(r, node, slot, {});
            });
            if (const auto *graph = r.try_get<const GeometryGraph>(GraphOf(r, use)); graph && graph->DisplayNode == use && graph->DisplayPort == a.Port) {
                r.patch<GeometryGraph>(GraphOf(r, use), [](GeometryGraph &p) { p.DisplayPort.reset(); });
            }
        }
    }
    r.patch<SubgraphInterface>(a.Subgraph, [&](auto &interface) { std::erase_if(interface.Ports, [&](const auto &p) { return p.Id == a.Port; }); });
    DropUsePorts(r, a.Subgraph);
}
} // namespace action::nodes

namespace graph {
NodeView ProjectGraph(const state::Registry &r, state::Entity graph, std::span<const state::Entity> shown) {
    NodeView view{.Graph = graph, .Authored = &r, .InputFirst = {0u}, .SocketFirst = {0u}};
    const auto has = [&](SocketDirection direction) { return r.all_of<Subgraph>(graph) && std::ranges::contains(r.get<const SubgraphInterface>(graph).Ports, direction, &SubgraphPort::Direction); };
    const bool input_ports = has(SocketDirection::In);
    if (input_ports) view.Nodes.push_back({.Item = graph, .Kind = KindOf<SubgraphNode>});
    for (auto node = FirstNode(r, graph); node != state::Null; node = r.get<const GraphNode>(node).Next) view.Nodes.push_back({.Item = node, .Kind = r.get<const GraphNode>(node).Kind});
    if (has(SocketDirection::Out)) view.Nodes.push_back({.Item = graph, .Kind = KindOf<SubgraphNode>});
    std::vector<SocketLink> wires;
    for (uint32_t n = 0; n < view.Nodes.size(); ++n) {
        const auto entity = view.Nodes[n].Item;
        if (view.Nodes[n].Kind == KindOf<SubgraphNode>) {
            // A boundary shows the ports of its direction facing the other way.
            const bool boundary = entity == graph, input_boundary = boundary && input_ports && n == 0u;
            for (const auto direction : {SocketDirection::In, SocketDirection::Out}) {
                for (const auto &port : InterfaceOf(r, entity)->Ports) {
                    const auto facing = boundary ? (port.Direction == SocketDirection::In ? SocketDirection::Out : SocketDirection::In) : port.Direction;
                    if (facing != direction || (boundary && (port.Direction == SocketDirection::In) != input_boundary)) continue;
                    view.Sockets.push_back(PortSocket(r, entity, port, facing));
                    view.SocketIds.push_back(port.Id);
                    if (facing == SocketDirection::In) wires.push_back(InputOf(r, entity, port.Id));
                }
            }
        } else {
            for (uint16_t s = 0; s < SocketsOf(r, entity).size(); ++s) {
                view.Sockets.push_back(SocketOf(r, entity, s));
                view.SocketIds.push_back(s);
            }
            wires.append_range(r.get<const NodeInputs>(entity).Links);
        }
        view.InputFirst.push_back(uint32_t(wires.size()));
        view.SocketFirst.push_back(uint32_t(view.Sockets.size()));
    }
    view.ByItem.resize(view.Nodes.size());
    std::ranges::iota(view.ByItem, 0u);
    std::ranges::stable_sort(view.ByItem, {}, [&](uint32_t n) { return view.Nodes[n].Item; });
    for (const auto &wire : wires) {
        const auto from = NodeOf(view, wire.FromNode);
        const auto ids = from == NoNode ? std::span<const uint16_t>{} : std::span{view.SocketIds}.subspan(view.SocketFirst[from], view.SocketFirst[from + 1] - view.SocketFirst[from]);
        const auto found = std::ranges::find(ids, wire.FromSocket);
        view.Wires.push_back(found == ids.end() ? NodeLink{} : NodeLink{from, uint16_t(found - ids.begin())});
    }
    view.DisplayNode = shown.empty() ? NoNode : NodeOf(view, shown.front());
    return view;
}
} // namespace graph
