#include "action/Core.h"
#include "action/Errors.h"
#include "action/Nodes.h"
#include "mesh/MeshComponents.h"
#include "procedure/NodeView.h"
#include "procedure/OpNodes.h"
#include "scene/Entity.h"
#include "state/Registry.h"

#include <numeric>

namespace procedure {
namespace {
// Whether the value a port passes varies, which its resolved source decides.
bool PortVaries(const state::Registry &r, state::Entity subnet, uint16_t port) {
    const auto resolved = ResolveSocket(r, {subnet, port});
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
    for (const auto subnet : path) at = InstanceDefinition(*at, subnet).Registry;
    return *at;
}

DefinitionRef InstanceDefinition(const state::Registry &r, state::Entity subnet) {
    const auto &instance = r.get<const Subnet>(subnet);
    return instance.DefinitionEntity != state::Null ? DefinitionRef{&r, instance.DefinitionEntity} : BuiltinDefinition(instance.Builtin);
}
state::Entity FirstNode(const state::Registry &r, state::Entity network) { return r.get<const NetworkNodes>(network).First; }
std::vector<state::Entity> NodesOf(const state::Registry &r, state::Entity network) {
    std::vector<state::Entity> nodes;
    for (auto node = FirstNode(r, network); node != state::Null; node = r.get<const ProcedureNode>(node).Next) nodes.push_back(node);
    return nodes;
}
bool Uses(const state::Registry &r, state::Entity network, state::Entity definition) {
    if (network == definition) return true;
    for (auto node = FirstNode(r, network); node != state::Null; node = r.get<const ProcedureNode>(node).Next) {
        if (const auto *subnet = r.try_get<const Subnet>(node); subnet && subnet->DefinitionEntity != state::Null && Uses(r, subnet->DefinitionEntity, definition)) return true;
    }
    return false;
}
std::vector<std::pair<state::Entity, state::Entity>> DefinitionUses(const state::Registry &r) {
    std::vector<std::pair<state::Entity, state::Entity>> uses;
    for (const auto mesh : SortedEntities(r.view<const GeometryProcedure>())) {
        std::vector<state::Entity> pending{mesh}, reached;
        while (!pending.empty()) {
            const auto network = pending.back();
            pending.pop_back();
            for (auto node = FirstNode(r, network); node != state::Null; node = r.get<const ProcedureNode>(node).Next) {
                const auto *subnet = r.try_get<const Subnet>(node);
                if (!subnet || subnet->DefinitionEntity == state::Null || std::ranges::contains(reached, subnet->DefinitionEntity)) continue;
                reached.push_back(subnet->DefinitionEntity);
                pending.push_back(subnet->DefinitionEntity);
                uses.emplace_back(subnet->DefinitionEntity, mesh);
            }
        }
    }
    std::ranges::sort(uses);
    return uses;
}
std::vector<state::Entity> MeshesUsing(const state::Registry &r, std::span<const std::pair<state::Entity, state::Entity>> uses, state::Entity network) {
    if (r.all_of<GeometryProcedure>(network)) return {network};
    const auto [first, last] = std::ranges::equal_range(uses, network, {}, &std::pair<state::Entity, state::Entity>::first);
    return std::ranges::subrange(first, last) | std::views::values | std::ranges::to<std::vector>();
}
std::vector<state::Entity> MeshesUsing(const state::Registry &r, state::Entity network) {
    return r.all_of<GeometryProcedure>(network) ? std::vector{network} : MeshesUsing(r, DefinitionUses(r), network);
}
const InstancePort *InstancePortOf(const state::Registry &r, state::Entity subnet, uint16_t port) {
    const auto &ports = r.get<const Subnet>(subnet).Ports;
    const auto found = std::ranges::find(ports, port, &InstancePort::Id);
    return found == ports.end() ? nullptr : &*found;
}
std::string InstanceName(const state::Registry &r, state::Entity subnet) {
    const auto definition = InstanceDefinition(r, subnet);
    return definition.Entity == state::Null ? std::string{} : GetName(*definition.Registry, definition.Entity);
}
state::Entity NetworkOf(const state::Registry &r, state::Entity node) { return r.get<const ProcedureNode>(node).Network; }
const SubnetInterface *InterfaceOf(const state::Registry &r, state::Entity subnet) {
    if (r.all_of<Subnet>(subnet)) {
        const auto definition = InstanceDefinition(r, subnet);
        return definition.Entity == state::Null ? nullptr : definition.Registry->try_get<const SubnetInterface>(definition.Entity);
    }
    return r.try_get<const SubnetInterface>(subnet);
}
const SubnetPort *PortOf(const state::Registry &r, state::Entity subnet, uint16_t port) {
    const auto *interface = InterfaceOf(r, subnet);
    if (!interface) return nullptr;
    const auto found = std::ranges::find(interface->Ports, port, &SubnetPort::Id);
    return found == interface->Ports.end() ? nullptr : &*found;
}
Socket PortSocket(const state::Registry &r, state::Entity subnet, const SubnetPort &port, SocketDirection facing) {
    return {.Types = port.Types, .Direction = facing, .Varies = facing == SocketDirection::In || PortVaries(r, subnet, port.Id), .Name = port.Name.c_str(), .Selection = port.Selection};
}
Socket SocketOf(const state::Registry &r, state::Entity node, uint16_t socket) {
    if (const auto *port = PortOf(r, node, socket)) return PortSocket(r, node, *port, port->Direction);
    if (!r.all_of<ProcedureNode>(node) || socket >= SocketsOf(r, node).size()) return {DataTypes{}, SocketDirection::In};
    auto descriptor = SocketsOf(r, node)[socket];
    if (descriptor.Direction == SocketDirection::Out && r.all_of<Op>(node)) descriptor.Varies = OpVaries(r, node);
    return descriptor;
}
std::span<const Socket> SocketsOf(const state::Registry &r, state::Entity node) {
    if (const auto *op = r.try_get<const Op>(node)) {
        const auto *definition = DefinitionOf(op->Node);
        return definition ? std::span{definition->Sockets} : std::span<const Socket>{};
    }
    return KindSockets(r.get<const ProcedureNode>(node).Kind);
}
uint16_t InputCountOf(const state::Registry &r, state::Entity node) {
    if (const auto *op = r.try_get<const Op>(node)) {
        const auto *definition = DefinitionOf(op->Node);
        return definition ? definition->Inputs : uint16_t{};
    }
    return InputCount[r.get<const ProcedureNode>(node).Kind];
}
uint16_t SlotSocketOf(const state::Registry &r, state::Entity node, uint16_t slot) { return SlotSocket(r.get<const ProcedureNode>(node).Kind, slot); }
void ForEachInput(const state::Registry &r, state::Entity node, const std::function<void(uint16_t, SocketLink)> &f) {
    if (const auto *subnet = r.try_get<const Subnet>(node)) {
        for (const auto ports = subnet->Ports; const auto &port : ports) f(port.Id, port.Source);
        return;
    }
    const auto links = r.get<const NodeInputs>(node).Links;
    for (uint16_t slot = 0; slot < links.size(); ++slot) f(slot, links[slot]);
}
void ForEachNetworkInput(const state::Registry &r, state::Entity network, const std::function<void(state::Entity, uint16_t, SocketLink)> &f) {
    for (const auto node : NodesOf(r, network)) ForEachInput(r, node, [&](uint16_t slot, SocketLink link) { f(node, slot, link); });
    if (!r.all_of<SubnetDefinition>(network)) return;
    for (const auto ports = r.get<const SubnetInterface>(network).Ports; const auto &port : ports)
        if (port.Direction == SocketDirection::Out) f(network, port.Id, port.Source());
}
std::optional<uint16_t> SubnetBypassInput(const state::Registry &r, state::Entity subnet, uint16_t output) {
    const auto *port = PortOf(r, subnet, output);
    if (!port || port->Direction != SocketDirection::Out) return {};
    std::optional<uint16_t> input;
    for (const auto &p : InterfaceOf(r, subnet)->Ports) {
        if (p.Direction != SocketDirection::In || p.Types != port->Types || p.Selection != port->Selection || PortVaries(r, subnet, p.Id) != PortVaries(r, subnet, output)) continue;
        if (input) return {};
        input = p.Id;
    }
    return input;
}
std::optional<uint16_t> ResultSocket(const state::Registry &r, state::Entity node) {
    if (r.all_of<Subnet>(node)) {
        const auto *interface = InterfaceOf(r, node);
        if (!interface) return {};
        for (const auto &port : interface->Ports) {
            if (port.Direction == SocketDirection::Out && (port.Types & ResultTypes)) return port.Id;
        }
        return {};
    }
    const auto *record = r.try_get<const ProcedureNode>(node);
    return record ? ResultSocket(record->Kind) : std::optional<uint16_t>{};
}
SocketLink DisplayLink(const state::Registry &r, const GeometryProcedure &procedure) {
    if (procedure.DisplayNode == state::Null) return {};
    const auto &at = RegistryAt(r, procedure.DisplayPath);
    const auto socket = procedure.DisplayPort ? procedure.DisplayPort : ResultSocket(at, procedure.DisplayNode);
    return socket ? SocketLink{procedure.DisplayNode, *socket} : SocketLink{};
}
NodePath DisplayedPath(const GeometryProcedure &procedure) {
    if (procedure.DisplayNode == state::Null) return {};
    auto path = procedure.DisplayPath;
    path.push_back(procedure.DisplayNode);
    return path;
}
std::vector<SocketLink> NormalizedInputs(uint16_t kind, std::vector<SocketLink> links) {
    if (MultiInput[kind] != NoSocket) SetMultiLinks(kind, links, MultiLinks(kind, links));
    return links;
}
void SetInput(state::Registry &r, state::Entity node, uint16_t slot, SocketLink source) {
    if (r.all_of<Subnet>(node)) {
        r.patch<Subnet>(node, [&](Subnet &subnet) {
            auto found = std::ranges::lower_bound(subnet.Ports, slot, {}, &InstancePort::Id);
            if (found == subnet.Ports.end() || found->Id != slot) found = subnet.Ports.insert(found, {.Id = slot});
            found->Source = source;
            if (source.FromNode == state::Null && !found->Value && !found->Selection) subnet.Ports.erase(found);
        });
    } else if (r.all_of<SubnetDefinition>(node)) {
        r.patch<SubnetInterface>(node, [&](auto &interface) {
            for (auto &port : interface.Ports)
                if (port.Id == slot) port.Source() = source;
        });
    } else {
        const auto kind = r.get<const ProcedureNode>(node).Kind;
        r.patch<NodeInputs>(node, [&](auto &inputs) {
            inputs.Links[slot] = source;
            inputs.Links = NormalizedInputs(kind, std::move(inputs.Links));
        });
    }
}
void InsertInput(state::Registry &r, state::Entity node, SocketLink source, uint16_t index) {
    const auto kind = r.get<const ProcedureNode>(node).Kind;
    r.patch<NodeInputs>(node, [&](auto &inputs) {
        auto listed = MultiLinks(kind, inputs.Links);
        std::erase(listed, source);
        listed.insert(listed.begin() + std::min<size_t>(index, listed.size()), source);
        SetMultiLinks(kind, inputs.Links, listed);
    });
}
SocketLink InputOf(const state::Registry &r, state::Entity node, uint16_t slot) {
    if (r.all_of<Subnet>(node)) {
        const auto *port = InstancePortOf(r, node, slot);
        return port ? port->Source : SocketLink{};
    }
    if (r.all_of<SubnetDefinition>(node)) {
        const auto *port = PortOf(r, node, slot);
        return port ? port->Source() : SocketLink{};
    }
    return r.get<const NodeInputs>(node).Links[slot];
}
std::vector<SocketLink> InputLinks(const state::Registry &r, state::Entity node, uint16_t socket) {
    if (!r.any_of<Subnet, SubnetDefinition>(node) && socket == MultiInput[r.get<const ProcedureNode>(node).Kind]) return MultiLinks(r.get<const ProcedureNode>(node).Kind, r.get<const NodeInputs>(node).Links);
    const auto link = InputOf(r, node, socket);
    return link.FromNode == state::Null ? std::vector<SocketLink>{} : std::vector{link};
}
ResolvedSocket ResolveSocket(const state::Registry &r, NodePath path, SocketLink link, bool bypass, const SocketEdit *edit) {
    std::vector<std::pair<NodePath, SocketLink>> visited;
    const auto *at = &RegistryAt(r, path);
    // An edit changes a link of the document, never one of a built-in definition.
    const auto edited = [&](state::Entity node, uint16_t socket) { return edit && at == &r && edit->Node == node && edit->Socket == socket; };
    while (link.FromNode != state::Null) {
        const auto from = link.FromNode;
        const bool subnet = at->all_of<Subnet>(from), definition = at->all_of<SubnetDefinition>(from);
        if (!subnet && !definition && !(bypass && at->all_of<NodeBypassed>(from))) return {std::move(path), at, link};
        if (std::ranges::any_of(visited, [&](const auto &v) { return v.second == link && v.first == path; })) return {.Cycle = true};
        visited.emplace_back(path, link);
        if (!subnet && !definition) {
            const auto kind = at->get<const ProcedureNode>(from).Kind;
            const auto input = BypassInput(kind);
            if (!input || ResultSocket(kind) != link.FromSocket) return {};
            link = edited(from, *input) ? edit->Source : at->get<const NodeInputs>(from).Links[*input];
            continue;
        }
        // The subnet node whose input port the link reads, and the port.
        auto instance = from;
        auto port_id = link.FromSocket;
        if (definition) {
            // A definition's input port reads the input port of the subnet node the path entered the definition through.
            if (path.empty()) return {};
            instance = path.back();
            path.pop_back();
            at = &RegistryAt(r, path);
        } else {
            const auto *port = PortOf(*at, from, link.FromSocket);
            if (!port) return {};
            if (port->Direction == SocketDirection::Out) {
                if (!bypass || !at->all_of<NodeBypassed>(from)) {
                    const auto definition_ref = InstanceDefinition(*at, from);
                    const bool own = definition_ref.Registry == &r && at == &r;
                    link = own && edit && edit->Node == definition_ref.Entity && edit->Socket == port->Id ? edit->Source : port->Source();
                    path.push_back(from);
                    at = definition_ref.Registry;
                    continue;
                }
                const auto input = SubnetBypassInput(*at, from, port->Id);
                if (!input) return {};
                port_id = *input;
            }
        }
        const auto *port = PortOf(*at, instance, port_id);
        if (!port) return {};
        const auto *own = InstancePortOf(*at, instance, port_id);
        const auto source = edited(instance, port_id) ? edit->Source : own ? own->Source : SocketLink{};
        if (source.FromNode != state::Null) {
            link = source;
            continue;
        }
        if (port->Selection) return own && own->Selection ? ResolvedSocket{.Path = std::move(path), .Registry = at, .SelectionSubnet = instance, .SelectionPort = port_id} : ResolvedSocket{};
        if (!ReadsDefault(*port)) return {};
        return {.Path = std::move(path), .Constant = own && own->Value ? own->Value : port->Default, .ValueSubnet = instance, .ValuePort = port_id};
    }
    return {};
}
std::optional<PortValue> ParameterDefault(const state::Registry &r, state::Entity node, uint16_t socket) {
    if (const auto *port = PortOf(r, node, socket)) {
        const auto *own = r.all_of<Subnet>(node) ? InstancePortOf(r, node, socket) : nullptr;
        return own && own->Value ? own->Value : port->Default;
    }
    const auto descriptor = SocketOf(r, node, socket);
    // An op input without a literal has the default of the first declaration of its node that has it, unless that reads a geometric property.
    if (const auto *op = r.try_get<const Op>(node)) {
        const std::string_view name{descriptor.Name};
        if (const auto literal = op->Literals.find(name); literal != op->Literals.end()) {
            const auto words = LiteralWords(r, node, literal->first, literal->second);
            if (words.empty() || !(TypeBit(*DataTypeOf(literal->second)) & NumericTypes)) return {};
            PortValue value{{}, *DataTypeOf(literal->second)};
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
            PortValue value{{}, *type};
            std::ranges::copy(*words, value.Words.begin());
            return value;
        }
        return {};
    }
    if (descriptor.Field == NoField) return {};
    std::optional<PortValue> value;
    VisitKind(r.get<const ProcedureNode>(node).Kind, [&]<typename K> {
        VisitSocketParameter(r.get<const K>(node), descriptor, [&]<typename F>(const F &field, const FieldSpec &) {
            value = PortValue{{}, DataType(std::countr_zero(detail::TypesOf<F>()))};
            std::memcpy(value->Words.data(), &field, sizeof(F));
        });
    });
    return value;
}
namespace {
void SetParameterDefault(state::Registry &r, state::Entity node, uint16_t socket, PortValue value) {
    if (r.all_of<Subnet>(node)) {
        r.patch<Subnet>(node, [&](Subnet &subnet) {
            auto found = std::ranges::lower_bound(subnet.Ports, socket, {}, &InstancePort::Id);
            if (found == subnet.Ports.end() || found->Id != socket) found = subnet.Ports.insert(found, {.Id = socket});
            found->Value = value;
        });
        return;
    }
    if (r.all_of<SubnetInterface>(node)) {
        r.patch<SubnetInterface>(node, [&](auto &interface) {
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
    VisitKind(r.get<const ProcedureNode>(node).Kind, [&]<typename K> {
        if constexpr (field::Walkable<K>) r.patch<K>(node, [&](auto &parameters) { AssignSocketParameter(parameters, descriptor, value.Words); });
    });
}
} // namespace

void OrderNetwork(state::Registry &r, state::Entity network, std::span<const state::Entity> nodes) {
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto next = i + 1u < nodes.size() ? nodes[i + 1u] : state::Null;
        if (const auto &record = r.get<const ProcedureNode>(nodes[i]); record.Network != network || record.Next != next) {
            r.patch<ProcedureNode>(nodes[i], [&](ProcedureNode &p) {
                p.Network = network;
                p.Next = next;
            });
        }
    }
    r.replace_if_changed(network, NetworkNodes{nodes.empty() ? state::Null : nodes.front(), nodes.empty() ? state::Null : nodes.back()});
}
void AddToNetwork(state::Registry &r, state::Entity network, NodeRecord node, std::vector<SocketLink> links) {
    const auto ends = r.get<const NetworkNodes>(network);
    r.emplace<ProcedureNode>(node.Entity, network, state::Null, node.Kind);
    r.emplace<NodeInputs>(node.Entity, std::move(links));
    if (ends.Last != state::Null) r.patch<ProcedureNode>(ends.Last, [&](ProcedureNode &record) { record.Next = node.Entity; });
    r.replace<NetworkNodes>(network, ends.First == state::Null ? node.Entity : ends.First, node.Entity);
}
state::Entity CreateDefinition(state::Registry &r, std::string_view name) {
    const auto definition = r.create();
    r.emplace<SubnetDefinition>(definition);
    r.emplace<NetworkNodes>(definition);
    r.emplace<SubnetInterface>(definition);
    EmplaceUniqueName(r, definition, name);
    return definition;
}
state::Entity CreateSubnet(state::Registry &r, state::Entity network, Subnet subnet) {
    const auto node = r.create();
    r.emplace<Subnet>(node, std::move(subnet));
    AddToNetwork(r, network, {node, KindOf<Subnet>}, {});
    return node;
}
bool IsNodeNetwork(const state::Registry &r, state::Entity network) { return r.valid(network) && r.all_of<NetworkNodes>(network); }
state::Entity SurfaceNodeOf(const state::Registry &r, state::Entity material) {
    for (auto node = FirstNode(r, material); node != state::Null; node = r.get<const ProcedureNode>(node).Next)
        if (r.get<const ProcedureNode>(node).Kind == KindOf<Surface>) return node;
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
            if constexpr (std::same_as<T, Subnet>) {
                for (auto &port : copied.Ports) port.Source = mapped(port.Source);
            }
            r.emplace<T>(copy, std::move(copied));
        };
        VisitKind(from.get<const ProcedureNode>(node).Kind, copy_component);
        [&]<typename... T>(TypeList<T...>) { (copy_component.template operator()<T>(), ...); }(TypeList<SourceState, InlineSelection, NodeBypassed, PublishSelection>{});
    }
    return copies;
}
std::vector<SocketLink> CopiedInputs(const state::Registry &from, state::Entity node, const NodeCopies &copies, const LinkMap &link) {
    auto links = from.get<const NodeInputs>(node).Links;
    for (auto &l : links) l = copies(l.FromNode) != state::Null ? SocketLink{copies(l.FromNode), l.FromSocket} : link(l);
    return NormalizedInputs(from.get<const ProcedureNode>(node).Kind, std::move(links));
}
NodeCopies CopyInto(state::Registry &r, state::Entity network, const state::Registry &from, std::span<const state::Entity> originals, const LinkMap &link) {
    const auto copies = CopyNodes(r, from, originals, link);
    for (const auto node : originals) AddToNetwork(r, network, {copies(node), from.get<const ProcedureNode>(node).Kind}, CopiedInputs(from, node, copies, link));
    LayoutNetworkValues(r, network);
    CopyLiterals(r, from, copies);
    return copies;
}
void RewritePaths(state::Registry &r, const std::function<void(NodePath &)> &rewrite) {
    // Each occurrence's state moves to where its rewritten path keeps it.
    const auto move = [&]<typename T> {
        std::vector<std::pair<NodePath, T>> held, moved;
        for (const auto node : SortedEntities(r.view<const T>())) held.emplace_back(NodePath{node}, r.get<const T>(node));
        for (const auto subnet : SortedEntities(r.view<const Inner<T>>())) {
            for (const auto &[key, value] : r.get<const Inner<T>>(subnet).Values) {
                NodePath path{subnet};
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
            moved.emplace_back(std::move(to), std::move(rewritten));
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
    for (const auto mesh : SortedEntities(r.view<const GeometryProcedure>())) {
        const auto &procedure = r.get<const GeometryProcedure>(mesh);
        auto shown = DisplayedPath(procedure);
        rewrite(shown);
        if (shown == DisplayedPath(procedure)) continue;
        r.patch<GeometryProcedure>(mesh, [&](GeometryProcedure &p) {
            p.DisplayNode = shown.empty() ? state::Null : shown.back();
            p.DisplayPath = shown.empty() ? NodePath{} : NodePath{shown.begin(), shown.end() - 1};
        });
    }
}
} // namespace procedure

namespace action::nodes {
namespace {
using namespace procedure;

// A definition's port identities last while NextId is at most UINT16_MAX, which each action that adds ports checks before it changes anything.
constexpr const char *NoPortIdentities{"The subnet has no available port identities"};
uint16_t AddPort(state::Registry &r, state::Entity definition, Socket descriptor, std::string name, SocketLink source, std::optional<PortValue> value = {}) {
    const auto next = r.get<const SubnetInterface>(definition).NextId;
    r.patch<SubnetInterface>(definition, [&](auto &interface) {
        interface.Ports.push_back({uint16_t(next), std::move(name), descriptor.Types, descriptor.Selection, descriptor.Direction, source, value});
        ++interface.NextId;
    });
    return uint16_t(next);
}
} // namespace

void Apply(state::Registry &r, const ExposeSocket &a) {
    const auto *node = r.try_get<const ProcedureNode>(a.Node);
    if (!r.all_of<SubnetDefinition>(a.Subnet) || !node || node->Network != a.Subnet) return Fail(r, "Expose a socket belonging to a node inside the definition");
    if (r.all_of<Subnet>(a.Node) ? !PortOf(r, a.Node, a.Socket) : a.Socket >= SocketsOf(r, a.Node).size()) return Fail(r, "The socket does not exist");
    if (r.get<const SubnetInterface>(a.Subnet).NextId > UINT16_MAX) return Fail(r, NoPortIdentities);
    const auto descriptor = SocketOf(r, a.Node, a.Socket);
    const auto name = a.Name.empty() ? std::string{descriptor.Name ? descriptor.Name : descriptor.Direction == SocketDirection::In ? "Input" :
                                                                                                                                     "Output"} :
                                       a.Name;
    if (descriptor.Direction == SocketDirection::In && descriptor.Multi) {
        InsertInput(r, a.Node, {a.Subnet, AddPort(r, a.Subnet, descriptor, name, {})}, UINT16_MAX);
    } else if (descriptor.Direction == SocketDirection::In) {
        const auto port = AddPort(r, a.Subnet, descriptor, name, {}, ParameterDefault(r, a.Node, a.Socket));
        SetInput(r, a.Node, a.Socket, {a.Subnet, port});
    } else AddPort(r, a.Subnet, descriptor, name, {a.Node, a.Socket});
}

void Apply(state::Registry &r, const GroupNodes &a) {
    const auto network = a.Network;
    if (a.Nodes.empty() || !IsNodeNetwork(r, network)) return Fail(r, "Group nodes of a geometry network or a material's graph");
    auto selected = a.Nodes;
    std::ranges::sort(selected);
    if (std::ranges::adjacent_find(selected) != selected.end()) return Fail(r, "Group distinct nodes");
    for (const auto node : selected) {
        if (!r.valid(node) || !r.all_of<ProcedureNode>(node) || NetworkOf(r, node) != network) return Fail(r, "Group nodes of the same network");
        if (r.get<const ProcedureNode>(node).Kind == KindOf<Surface>) return Fail(r, "Group nodes other than the material's Surface node");
    }
    const auto grouped = [&](state::Entity node) { return std::ranges::binary_search(selected, node); };
    const auto nodes = NodesOf(r, network);
    // Each link that crosses the group's boundary takes at most one port of the new definition.
    size_t crossing = 0u;
    ForEachNetworkInput(r, network, [&](state::Entity node, uint16_t, SocketLink link) { crossing += link.FromNode != state::Null && grouped(node) != grouped(link.FromNode); });
    if (crossing > UINT16_MAX) return Fail(r, NoPortIdentities);
    // The moved nodes' literals move to the definition's block.
    std::vector<std::pair<state::Entity, OpLiteralWords>> literals;
    for (const auto node : nodes)
        if (grouped(node) && r.all_of<Op>(node)) literals.emplace_back(node, LiteralsOf(r, node));
    const auto definition = CreateDefinition(r, "Subnet");
    // The subnet node takes the first grouped node's place in the network's order.
    const auto subnet = r.create();
    std::vector<state::Entity> kept, moved;
    for (const auto node : nodes) {
        if (grouped(node) && moved.empty()) kept.push_back(subnet);
        (grouped(node) ? moved : kept).push_back(node);
    }
    r.emplace<ProcedureNode>(subnet, network, state::Null, KindOf<Subnet>);
    r.emplace<NodeInputs>(subnet);
    r.emplace<Subnet>(subnet, definition);
    OrderNetwork(r, network, kept);
    OrderNetwork(r, definition, moved);
    for (const auto node : moved) ClearNetworkPosition(r, network, node);
    // Links from outside into moved nodes read new input ports, which the subnet node links to the outside sources.
    for (const auto node : moved) {
        ForEachInput(r, node, [&](uint16_t slot, SocketLink link) {
            if (link.FromNode == state::Null || grouped(link.FromNode)) return;
            const auto socket = SlotSocketOf(r, node, slot);
            const auto descriptor = SocketOf(r, node, socket);
            const auto port = AddPort(r, definition, descriptor, descriptor.Name ? descriptor.Name : "Input", {}, ParameterDefault(r, node, socket));
            SetInput(r, node, slot, {definition, port});
            SetInput(r, subnet, port, link);
        });
    }
    // Links from moved nodes into the rest read output ports bound to the moved nodes' sockets.
    const auto output = [&](SocketLink link) {
        const auto &ports = r.get<const SubnetInterface>(definition).Ports;
        const auto existing = std::ranges::find_if(ports, [&](const auto &p) { return p.Direction == SocketDirection::Out && p.Source() == link; });
        return existing != ports.end() ? existing->Id : AddPort(r, definition, SocketOf(r, link.FromNode, link.FromSocket), "Output", link);
    };
    ForEachNetworkInput(r, network, [&](state::Entity node, uint16_t slot, SocketLink link) {
        if (link.FromNode != state::Null && grouped(link.FromNode)) SetInput(r, node, slot, {subnet, output(link)});
    });
    if (std::ranges::none_of(r.get<const SubnetInterface>(definition).Ports, [](const auto &port) { return port.Direction == SocketDirection::Out; })) {
        for (const auto node : moved | std::views::reverse) {
            if (const auto socket = ResultSocket(r, node)) {
                output({node, *socket});
                break;
            }
        }
    }
    // Every occurrence of a moved node is now inside the new subnet node.
    RewritePaths(r, [&](NodePath &path) {
        if (const auto at = std::ranges::find_if(path, grouped); at != path.end()) path.insert(at, subnet);
    });
    LayoutDefinitionValues(r, definition);
    for (const auto &[node, words] : literals) WriteLiterals(r, node, words);
    LayoutNetworkValues(r, network);
    r.emplace_or_replace<NetworkActive>(network, subnet, std::vector{subnet});
}
void Apply(state::Registry &r, const UngroupNodes &a) {
    if (!r.valid(a.Subnet) || !r.all_of<Subnet>(a.Subnet)) return Fail(r, "The subnet does not exist");
    const auto network = NetworkOf(r, a.Subnet);
    const auto definition = InstanceDefinition(r, a.Subnet);
    if (definition.Entity == state::Null) return Fail(r, "The subnet's definition does not exist");
    const auto ports = r.get<const Subnet>(a.Subnet).Ports;
    // A read of an input port reads the subnet node's link into it, and an unlinked port's value moves to each reader.
    const auto port_read = [&](uint16_t id) {
        const auto own = std::ranges::find(ports, id, &InstancePort::Id);
        return own != ports.end() ? own->Source : SocketLink{};
    };
    const auto nodes = NodesOf(r, network);
    const auto position = std::ranges::find(nodes, a.Subnet);
    const auto &from = *definition.Registry;
    const auto originals = NodesOf(from, definition.Entity);
    const LinkMap read = [&](SocketLink l) { return l.FromNode == definition.Entity ? port_read(l.FromSocket) : SocketLink{}; };
    const auto copies = CopyNodes(r, from, originals, read);
    for (const auto node : originals) AddToNetwork(r, network, {copies(node), from.get<const ProcedureNode>(node).Kind}, CopiedInputs(from, node, copies, read));
    const auto copied = [&](state::Entity node) { return std::ranges::contains(copies.Items, node, &std::pair<state::Entity, state::Entity>::second); };
    // A read of an output port reads its source's copy, or the link into the input port it passes.
    // A bypassed subnet's output port reads what the bypass passes.
    const bool bypassed = r.all_of<NodeBypassed>(a.Subnet);
    const auto out_source = [&](uint16_t id) -> SocketLink {
        if (bypassed) {
            const auto input = SubnetBypassInput(r, a.Subnet, id);
            return input ? InputOf(r, a.Subnet, *input) : SocketLink{};
        }
        const auto *port = PortOf(r, a.Subnet, id);
        if (!port) return {};
        const auto source = port->Source();
        return source.FromNode == definition.Entity ? port_read(source.FromSocket) : SocketLink{copies(source.FromNode), source.FromSocket};
    };
    ForEachNetworkInput(r, network, [&](state::Entity node, uint16_t slot, SocketLink link) {
        if (node != a.Subnet && !copied(node) && link.FromNode == a.Subnet) SetInput(r, node, slot, out_source(link.FromSocket));
    });
    // A mesh displaying an output port of the subnet node, in its own network or through the subnet nodes of its display path, displays what the port reads.
    for (const auto mesh : SortedEntities(r.view<const GeometryProcedure>())) {
        const auto &procedure = r.get<const GeometryProcedure>(mesh);
        if (procedure.DisplayNode != a.Subnet || &RegistryAt(r, procedure.DisplayPath) != &r) continue;
        const auto shown = procedure.DisplayPort ? out_source(*procedure.DisplayPort) : SocketLink{};
        r.patch<GeometryProcedure>(mesh, [&](GeometryProcedure &p) {
            p.DisplayNode = shown.FromNode;
            p.DisplayPort = shown.FromNode != state::Null && r.all_of<Subnet>(shown.FromNode) ? std::optional{shown.FromSocket} : std::nullopt;
            p.DisplayPath = shown.FromNode != state::Null ? p.DisplayPath : NodePath{};
        });
    }
    RewritePaths(r, [&](NodePath &path) {
        const auto at = std::ranges::find(path, a.Subnet);
        if (at == path.end() || at + 1 == path.end()) return;
        *(at + 1) = copies(*(at + 1));
        path.erase(at);
    });
    // The copies take the subnet node's place in the network's order.
    std::vector<state::Entity> order{nodes.begin(), position}, selection;
    for (const auto original : originals) selection.push_back(copies(original));
    order.append_range(selection);
    order.insert(order.end(), position + 1, nodes.end());
    OrderNetwork(r, network, order);
    ClearNetworkPosition(r, network, a.Subnet);
    LayoutNetworkValues(r, network);
    CopyLiterals(r, from, copies);
    // Each unlinked input port's value then replaces the copied value of each parameter that reads it.
    for (const auto [original, copy] : copies.Items) {
        ForEachInput(from, original, [&](uint16_t slot, SocketLink link) {
            if (link.FromNode != definition.Entity || port_read(link.FromSocket).FromNode != state::Null) return;
            const auto resolved = ResolveSocket(r, NodePath{a.Subnet}, link);
            if (resolved.Constant) SetParameterDefault(r, copy, SlotSocketOf(r, copy, slot), *resolved.Constant);
            if (resolved.SelectionSubnet != state::Null) r.emplace_or_replace<InlineSelection>(copy, *InstancePortOf(r, a.Subnet, resolved.SelectionPort)->Selection);
        });
    }
    // The copy of the definition's last node is active.
    const auto active = selection.empty() ? state::Null : selection.back();
    std::ranges::sort(selection);
    r.emplace_or_replace<NetworkActive>(network, active, std::move(selection));
    r.destroy(a.Subnet);
}
void Apply(state::Registry &r, const MakeUnique &a) {
    if (!r.valid(a.Subnet) || !r.all_of<Subnet>(a.Subnet)) return Fail(r, "The subnet does not exist");
    const auto definition = InstanceDefinition(r, a.Subnet);
    if (definition.Entity == state::Null) return Fail(r, "The subnet's definition does not exist");
    const auto copy = CreateDefinition(r, GetName(*definition.Registry, definition.Entity));
    const auto copies = CopyInto(r, copy, *definition.Registry, NodesOf(*definition.Registry, definition.Entity), [&](SocketLink l) { return l.FromNode == definition.Entity ? SocketLink{copy, l.FromSocket} : SocketLink{}; });
    auto interface = definition.Registry->get<const SubnetInterface>(definition.Entity);
    for (auto &port : interface.Ports) {
        if (port.Direction != SocketDirection::Out) continue;
        const auto source = port.Source();
        port.Source() = source.FromNode == definition.Entity ? SocketLink{copy, source.FromSocket} : SocketLink{copies(source.FromNode), source.FromSocket};
    }
    r.replace<SubnetInterface>(copy, std::move(interface));
    r.patch<Subnet>(a.Subnet, [&](Subnet &subnet) {
        subnet.DefinitionEntity = copy;
        subnet.Builtin.clear();
    });
    RewritePaths(r, [&](NodePath &path) {
        const auto at = std::ranges::find(path, a.Subnet);
        if (at != path.end() && at + 1 != path.end()) *(at + 1) = copies(*(at + 1));
    });
}
bool DefinitionUsed(const state::Registry &r, state::Entity definition) {
    return std::ranges::any_of(r.view<const Subnet>(), [&](state::Entity node) { return r.get<const Subnet>(node).DefinitionEntity == definition; });
}
void Apply(state::Registry &r, const DeleteDefinition &a) {
    if (!r.valid(a.Definition) || !r.all_of<SubnetDefinition>(a.Definition)) return Fail(r, "The definition does not exist");
    if (DefinitionUsed(r, a.Definition)) return Fail(r, "Delete a definition no subnet uses");
    r.destroy(NodesOf(r, a.Definition));
    r.destroy(a.Definition);
}
void Apply(state::Registry &r, const SetPortValue &a) {
    const auto *port = PortOf(r, a.Subnet, a.Port);
    const auto *field = port ? std::get_if<PromotedField>(&port->Binding) : nullptr;
    if (!field && (!port || !ReadsDefault(*port))) return Fail(r, "This port has no parameter default");
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
    SetParameterDefault(r, a.Subnet, a.Port, {a.Words, type});
}
void Apply(state::Registry &r, const ClearPortValue &a) {
    const auto *port = r.any_of<Subnet, SubnetDefinition>(a.Subnet) ? PortOf(r, a.Subnet, a.Port) : nullptr;
    if (!port || !(ReadsDefault(*port) || (port->Selection && port->Direction == SocketDirection::In))) return Fail(r, "Clear the value of an input port of a subnet node or a definition");
    if (r.all_of<Subnet>(a.Subnet)) {
        const auto *own = InstancePortOf(r, a.Subnet, a.Port);
        if (!own || (!own->Value && !own->Selection)) return;
        r.patch<Subnet>(a.Subnet, [&](Subnet &subnet) {
            const auto found = std::ranges::find(subnet.Ports, a.Port, &InstancePort::Id);
            if (found->Source.FromNode == state::Null) subnet.Ports.erase(found);
            else {
                found->Value.reset();
                found->Selection.reset();
            }
        });
    } else if (port->Default) {
        r.patch<SubnetInterface>(a.Subnet, [&](auto &interface) { std::ranges::find(interface.Ports, a.Port, &SubnetPort::Id)->Default.reset(); });
    }
}
void Apply(state::Registry &r, const RenamePort &a) {
    if (!r.all_of<SubnetInterface>(a.Subnet) || !PortOf(r, a.Subnet, a.Port) || a.Name.empty()) return Fail(r, "Name an existing port of a definition or scene subnet");
    r.patch<SubnetInterface>(a.Subnet, [&](auto &interface) { for (auto &port : interface.Ports) if (port.Id == a.Port) port.Name = a.Name; });
}
void Apply(state::Registry &r, const MovePort &a) {
    if (!r.all_of<SubnetInterface>(a.Subnet)) return Fail(r, "Reorder ports of a definition or scene subnet");
    const auto *port = PortOf(r, a.Subnet, a.Port), *before = PortOf(r, a.Subnet, a.BeforePort);
    if (!port || !before || port->Direction != before->Direction) return Fail(r, "Reorder ports in the same direction");
    if (a.Port == a.BeforePort) return;
    r.patch<SubnetInterface>(a.Subnet, [&](auto &interface) {
        const auto from = std::ranges::find(interface.Ports, a.Port, &SubnetPort::Id);
        auto moved = std::move(*from);
        interface.Ports.erase(from);
        interface.Ports.insert(std::ranges::find(interface.Ports, a.BeforePort, &SubnetPort::Id), std::move(moved));
    });
}
void Apply(state::Registry &r, const RemovePort &a) {
    if (!r.all_of<SubnetInterface>(a.Subnet) || !PortOf(r, a.Subnet, a.Port)) return Fail(r, "The port does not exist");
    if (r.all_of<SubnetDefinition>(a.Subnet)) {
        // A removed input port's readers keep its value, and every link to a removed port is removed.
        const auto port = *PortOf(r, a.Subnet, a.Port);
        for (const auto node : NodesOf(r, a.Subnet)) {
            ForEachInput(r, node, [&](uint16_t slot, SocketLink link) {
                if (link != SocketLink{a.Subnet, a.Port}) return;
                if (port.Default) SetParameterDefault(r, node, SlotSocketOf(r, node, slot), *port.Default);
                SetInput(r, node, slot, {});
            });
        }
        for (const auto subnet : SortedEntities(r.view<const Subnet>())) {
            if (r.get<const Subnet>(subnet).DefinitionEntity != a.Subnet) continue;
            if (InstancePortOf(r, subnet, a.Port)) r.patch<Subnet>(subnet, [&](Subnet &s) { std::erase_if(s.Ports, [&](const auto &p) { return p.Id == a.Port; }); });
            ForEachNetworkInput(r, NetworkOf(r, subnet), [&](state::Entity node, uint16_t slot, SocketLink link) {
                if (link == SocketLink{subnet, a.Port}) SetInput(r, node, slot, {});
            });
            if (const auto *procedure = r.try_get<const GeometryProcedure>(NetworkOf(r, subnet)); procedure && procedure->DisplayNode == subnet && procedure->DisplayPort == a.Port) {
                r.patch<GeometryProcedure>(NetworkOf(r, subnet), [](GeometryProcedure &p) { p.DisplayPort.reset(); });
            }
        }
    }
    r.patch<SubnetInterface>(a.Subnet, [&](auto &interface) { std::erase_if(interface.Ports, [&](const auto &p) { return p.Id == a.Port; }); });
}
} // namespace action::nodes

namespace procedure {
NodeView ProjectNetwork(const state::Registry &r, state::Entity network, std::span<const state::Entity> shown) {
    NodeView view{.Network = network, .Authored = &r, .InputFirst = {0u}, .SocketFirst = {0u}};
    const auto has = [&](SocketDirection direction) { return r.all_of<SubnetDefinition>(network) && std::ranges::contains(r.get<const SubnetInterface>(network).Ports, direction, &SubnetPort::Direction); };
    const bool input_ports = has(SocketDirection::In);
    if (input_ports) view.Nodes.push_back({.Item = network, .Kind = KindOf<Subnet>});
    for (auto node = FirstNode(r, network); node != state::Null; node = r.get<const ProcedureNode>(node).Next) view.Nodes.push_back({.Item = node, .Kind = r.get<const ProcedureNode>(node).Kind});
    if (has(SocketDirection::Out)) view.Nodes.push_back({.Item = network, .Kind = KindOf<Subnet>});
    std::vector<SocketLink> wires;
    for (uint32_t n = 0; n < view.Nodes.size(); ++n) {
        const auto entity = view.Nodes[n].Item;
        if (view.Nodes[n].Kind == KindOf<Subnet>) {
            // A boundary shows the ports of its direction facing the other way.
            const bool boundary = entity == network, input_boundary = boundary && input_ports && n == 0u;
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
} // namespace procedure
