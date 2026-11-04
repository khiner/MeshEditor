#include "action/Core.h"
#include "action/Errors.h"
#include "action/Nodes.h"
#include "procedure/Procedure.h"
#include "scene/Entity.h"
#include "state/Registry.h"

namespace procedure {
state::Entity ContainerOf(const state::Registry &r, state::Entity node) {
    const auto *parent = r.try_get<const NodeContainer>(node);
    return parent ? parent->Parent : state::Null;
}
const SubnetPort *PortOf(const state::Registry &r, state::Entity subnet, uint16_t port) {
    const auto *interface = r.try_get<const SubnetInterface>(subnet);
    if (!interface) return nullptr;
    const auto found = std::ranges::find(interface->Ports, port, &SubnetPort::Id);
    return found == interface->Ports.end() ? nullptr : &*found;
}
Socket SocketOf(const state::Registry &r, state::Entity node, uint16_t socket) {
    if (const auto *port = PortOf(r, node, socket)) return {port->Type, port->Direction, port->Types, false, port->Name.c_str()};
    const auto *record = r.try_get<const ProcedureNode>(node);
    if (!record || socket >= SocketCount[record->Kind]) return {SocketType::Value, SocketDirection::In};
    return KindSockets(record->Kind)[socket];
}
std::optional<uint16_t> SubnetBypassInput(const state::Registry &r, state::Entity subnet, uint16_t output) {
    const auto *port = PortOf(r, subnet, output);
    if (!port || port->Direction != SocketDirection::Out) return {};
    std::optional<uint16_t> input;
    for (const auto &p : r.get<const SubnetInterface>(subnet).Ports) {
        if (p.Direction != SocketDirection::In || p.Type != port->Type || p.Types != port->Types) continue;
        if (input) return {};
        input = p.Id;
    }
    return input;
}
std::optional<uint16_t> ResultSocket(const state::Registry &r, state::Entity node) {
    if (const auto *interface = r.try_get<const SubnetInterface>(node)) {
        for (const auto &port : interface->Ports) {
            if (port.Direction == SocketDirection::Out && (port.Type == SocketType::Geometry || port.Type == SocketType::Instances)) return port.Id;
        }
        return {};
    }
    const auto *record = r.try_get<const ProcedureNode>(node);
    return record ? ResultSocket(record->Kind) : std::optional<uint16_t>{};
}
SocketLink DisplayLink(const state::Registry &r, const GeometryProcedure &procedure) {
    const auto socket = procedure.DisplayPort ? procedure.DisplayPort : ResultSocket(r, procedure.DisplayNode);
    return socket ? SocketLink{procedure.DisplayNode, *socket} : SocketLink{};
}
void SetInput(state::Registry &r, state::Entity node, uint16_t socket, SocketLink source) {
    if (r.all_of<Subnet>(node)) {
        r.patch<SubnetInterface>(node, [&](auto &interface) {
            for (auto &port : interface.Ports)
                if (port.Id == socket) port.Source() = source;
        });
    } else r.patch<NodeInputs>(node, [&](auto &inputs) { inputs.Links[socket] = source; });
}
SocketLink InputOf(const state::Registry &r, state::Entity node, uint16_t socket) {
    if (const auto *port = PortOf(r, node, socket)) return port->Source();
    return r.get<const NodeInputs>(node).Links[socket];
}
ResolvedSocket ResolveSocket(const state::Registry &r, SocketLink link, bool bypass, const SocketEdit *edit) {
    std::vector<SocketLink> visited;
    while (link.FromNode != state::Null) {
        const bool subnet = r.all_of<Subnet>(link.FromNode);
        if (!subnet && !(bypass && r.all_of<NodeBypassed>(link.FromNode))) return {link, {}};
        if (std::ranges::contains(visited, link)) return {{}, {}, true};
        visited.push_back(link);
        if (!subnet) {
            const auto kind = r.get<const ProcedureNode>(link.FromNode).Kind;
            const auto input = BypassInput(kind);
            if (!input || ResultSocket(kind) != link.FromSocket) return {};
            link = edit && edit->Node == link.FromNode && edit->Socket == *input ? edit->Source : r.get<const NodeInputs>(link.FromNode).Links[*input];
            continue;
        }
        const auto *port = PortOf(r, link.FromNode, link.FromSocket);
        if (!port) return {};
        if (bypass && r.all_of<NodeBypassed>(link.FromNode) && port->Direction == SocketDirection::Out) {
            const auto input = SubnetBypassInput(r, link.FromNode, port->Id);
            if (!input) return {};
            link.FromSocket = *input;
            continue;
        }
        const auto source = edit && edit->Node == link.FromNode && edit->Socket == link.FromSocket ? edit->Source : port->Source();
        if (source.FromNode == state::Null) {
            if (port->Direction == SocketDirection::In && (port->Type == SocketType::Value || port->Type == SocketType::Attribute)) return {{}, port->Default};
            return {};
        }
        link = source;
    }
    return {};
}
std::optional<PortValue> ParameterDefault(const state::Registry &r, state::Entity node, uint16_t socket) {
    const auto descriptor = SocketOf(r, node, socket);
    if (const auto *port = PortOf(r, node, socket)) return port->Default;
    if (descriptor.Field == NoField) return {};
    std::optional<PortValue> value;
    VisitKind(r.get<const ProcedureNode>(node).Kind, [&]<typename K> {
        VisitSocketParameter(r.get<const K>(node), descriptor, [&]<typename F>(const F &field) {
            value = PortValue{{}, AttributeType(std::countr_zero(detail::TypesOf<F>()))};
            std::memcpy(value->Words.data(), &field, sizeof(F));
        });
    });
    return value;
}
void SetParameterDefault(state::Registry &r, state::Entity node, uint16_t socket, PortValue value) {
    const auto descriptor = SocketOf(r, node, socket);
    if (r.all_of<Subnet>(node)) {
        r.patch<SubnetInterface>(node, [&](auto &interface) {
            for (auto &port : interface.Ports)
                if (port.Id == socket) port.Default = value;
        });
        return;
    }
    if (descriptor.Field == NoField) return;
    VisitKind(r.get<const ProcedureNode>(node).Kind, [&]<typename K> {
        if constexpr (field::Walkable<K>) r.patch<K>(node, [&](auto &parameters) { AssignSocketParameter(parameters, descriptor, value.Words); });
    });
}
} // namespace procedure

namespace action::nodes {
namespace {
using namespace procedure;

uint16_t AddPort(state::Registry &r, state::Entity subnet, Socket descriptor, std::string name, SocketLink source, PortValue value = {}) {
    const auto next = r.get<const SubnetInterface>(subnet).NextId;
    if (next > UINT16_MAX) throw std::runtime_error("Subnet port identities exhausted");
    r.patch<SubnetInterface>(subnet, [&](auto &interface) {
        interface.Ports.push_back({uint16_t(next), std::move(name), descriptor.Type, descriptor.Direction, descriptor.Types, source, value});
        ++interface.NextId;
    });
    return uint16_t(next);
}
bool Inside(const state::Registry &r, state::Entity node, state::Entity subnet) {
    for (auto parent = ContainerOf(r, node); parent != state::Null; parent = ContainerOf(r, parent))
        if (parent == subnet) return true;
    return false;
}
void RewriteReferences(state::Registry &r, state::Entity owner, state::Entity subnet, std::optional<uint16_t> port_id, bool reconnect) {
    const auto interface = r.get<const SubnetInterface>(subnet);
    for (const auto [node, record] : r.view<const ProcedureNode>().each()) {
        if (record.Procedure != owner || node == subnet) continue;
        const auto rewrite = [&](SocketLink &link, uint16_t target_socket, PortValue *default_value = nullptr) {
            if (link.FromNode != subnet || (port_id && link.FromSocket != *port_id)) return false;
            const auto port = std::ranges::find(interface.Ports, link.FromSocket, &SubnetPort::Id);
            if (port == interface.Ports.end()) {
                link = {};
                return true;
            }
            const auto resolved = ResolveSocket(r, link);
            if (resolved.Constant && port->Direction == SocketDirection::In) {
                if (default_value) *default_value = *resolved.Constant;
                else SetParameterDefault(r, node, target_socket, *resolved.Constant);
            }
            link = reconnect ? port->Source() : SocketLink{};
            return true;
        };
        if (r.all_of<Subnet>(node)) {
            auto ports = r.get<const SubnetInterface>(node);
            bool changed = false;
            for (auto &port : ports.Ports) changed |= rewrite(port.Source(), port.Id, &port.Default);
            if (changed) r.replace<SubnetInterface>(node, std::move(ports));
        } else {
            auto inputs = r.get<const NodeInputs>(node);
            bool changed = false;
            for (uint16_t s = 0; s < inputs.Links.size(); ++s) changed |= rewrite(inputs.Links[s], s);
            if (changed) r.replace<NodeInputs>(node, std::move(inputs));
        }
    }
}
} // namespace

void Apply(state::Registry &r, const ExposeSocket &a) {
    const auto *group = r.try_get<const ProcedureNode>(a.Subnet);
    const auto *node = r.try_get<const ProcedureNode>(a.Node);
    if (!group || !node || !r.all_of<Subnet>(a.Subnet) || group->Procedure != node->Procedure || ContainerOf(r, a.Node) != a.Subnet) return Fail(r, "Expose a socket belonging to a node inside the subnet");
    if (r.all_of<Subnet>(a.Node) ? !PortOf(r, a.Node, a.Socket) : a.Socket >= SocketCount[node->Kind]) return Fail(r, "The socket does not exist");
    const auto descriptor = SocketOf(r, a.Node, a.Socket);
    const auto name = a.Name.empty() ? std::string{descriptor.Name ? descriptor.Name : descriptor.Direction == SocketDirection::In ? "Input" :
                                                                                                                                     "Output"} :
                                       a.Name;
    if (descriptor.Direction == SocketDirection::In) {
        const auto source = InputOf(r, a.Node, a.Socket);
        const auto value = ParameterDefault(r, a.Node, a.Socket).value_or(PortValue{});
        const auto port = AddPort(r, a.Subnet, descriptor, name, source, value);
        SetInput(r, a.Node, a.Socket, {a.Subnet, port});
    } else AddPort(r, a.Subnet, descriptor, name, {a.Node, a.Socket});
}

void Apply(state::Registry &r, const GroupNodes &a) {
    if (a.Nodes.empty() || !r.all_of<GeometryProcedure>(a.MeshEntity)) return Fail(r, "Group nodes in a geometry network");
    const auto parent = ContainerOf(r, a.Nodes.front());
    auto selected = a.Nodes;
    std::ranges::sort(selected);
    if (std::ranges::adjacent_find(selected) != selected.end()) return Fail(r, "Group distinct nodes");
    for (const auto node : selected) {
        const auto *record = r.try_get<const ProcedureNode>(node);
        if (!record || record->Procedure != a.MeshEntity || ContainerOf(r, node) != parent) return Fail(r, "Group nodes in the same network");
    }
    uvec2 position = r.get<const NodeLayout>(selected.front()).Position;
    for (const auto node : selected) {
        const auto p = r.get<const NodeLayout>(node).Position;
        position = {std::min(position.x, p.x), std::min(position.y, p.y)};
    }
    Apply(r, Action{AddNode{a.MeshEntity, KindOf<Subnet>, position, parent}});
    const auto subnet = r.get<const GeometryProcedure>(a.MeshEntity).Last;
    for (const auto node : selected) {
        r.emplace_or_replace<NodeContainer>(node, subnet);
        r.patch<NodeLayout>(node, [&](auto &layout) { layout.Position = layout.Position - position + uvec2{AppendGap, 2u}; });
    }
    const auto nodes = NodesOf(r, r.get<const GeometryProcedure>(a.MeshEntity));
    for (const auto node : nodes) {
        if (node == subnet) continue;
        const bool internal = Inside(r, node, subnet);
        const auto process = [&](uint16_t socket, SocketLink link) {
            if (link.FromNode == state::Null || link.FromNode == subnet) return;
            const bool source_inside = Inside(r, link.FromNode, subnet);
            if (internal && !source_inside) {
                const auto descriptor = SocketOf(r, node, socket);
                const auto port = AddPort(r, subnet, {descriptor.Type, SocketDirection::In, descriptor.Types}, descriptor.Name ? descriptor.Name : "Input", link, ParameterDefault(r, node, socket).value_or(PortValue{}));
                SetInput(r, node, socket, {subnet, port});
            } else if (!internal && source_inside) {
                const auto descriptor = SocketOf(r, link.FromNode, link.FromSocket);
                const auto &ports = r.get<const SubnetInterface>(subnet).Ports;
                const auto existing = std::ranges::find_if(ports, [&](const auto &p) { return p.Direction == SocketDirection::Out && p.Source() == link; });
                const auto port = existing == ports.end() ? AddPort(r, subnet, {descriptor.Type, SocketDirection::Out, descriptor.Types}, "Output", link) : existing->Id;
                SetInput(r, node, socket, {subnet, port});
            }
        };
        if (r.all_of<Subnet>(node)) {
            const auto ports = r.get<const SubnetInterface>(node).Ports;
            for (const auto &port : ports) process(port.Id, port.Source());
        } else {
            const auto inputs = r.get<const NodeInputs>(node).Links;
            for (uint16_t s = 0; s < inputs.size(); ++s) process(s, inputs[s]);
        }
    }
    if (std::ranges::none_of(r.get<const SubnetInterface>(subnet).Ports, [](const auto &port) { return port.Direction == SocketDirection::Out; })) {
        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
            if (!std::ranges::binary_search(selected, *it)) continue;
            if (const auto output = ResultSocket(r, *it)) {
                AddPort(r, subnet, SocketOf(r, *it, *output), "Output", {*it, *output});
                break;
            }
        }
    }
}
void Apply(state::Registry &r, const UngroupNodes &a) {
    const auto *record = r.try_get<const ProcedureNode>(a.Subnet);
    if (!record || !r.all_of<Subnet>(a.Subnet)) return Fail(r, "The subnet does not exist");
    const auto owner = record->Procedure, parent = ContainerOf(r, a.Subnet);
    const auto &procedure = r.get<const GeometryProcedure>(owner);
    if (procedure.DisplayNode == a.Subnet) {
        const auto displayed = ResolveSocket(r, DisplayLink(r, procedure)).Source;
        r.patch<GeometryProcedure>(owner, [&](auto &p) { p.DisplayNode = displayed.FromNode; p.DisplayPort.reset(); });
    }
    RewriteReferences(r, owner, a.Subnet, {}, true);
    for (const auto node : SortedEntities(r.view<const NodeContainer>())) {
        if (ContainerOf(r, node) != a.Subnet) continue;
        const auto position = r.get<const NodeLayout>(a.Subnet).Position;
        r.patch<NodeLayout>(node, [&](auto &layout) {
            layout.Position = {uint32_t(std::max(int64_t(position.x) + layout.Position.x - AppendGap, int64_t{0})), uint32_t(std::max(int64_t(position.y) + layout.Position.y - 2, int64_t{0}))};
        });
        if (parent == state::Null) r.remove<NodeContainer>(node);
        else r.replace<NodeContainer>(node, parent);
    }
    Apply(r, Action{DeleteNodes{owner, {a.Subnet}}});
}
void Apply(state::Registry &r, const SetPortValue &a) {
    const auto *port = PortOf(r, a.Subnet, a.Port);
    if (!port || port->Direction != SocketDirection::In || !(port->Type == SocketType::Value || port->Type == SocketType::Attribute)) return Fail(r, "This port has no parameter default");
    if (const auto *field = std::get_if<PromotedField>(&port->Binding)) {
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
    } else SetParameterDefault(r, a.Subnet, a.Port, {a.Words, port->Default.Type});
}
void Apply(state::Registry &r, const RenamePort &a) {
    if (!PortOf(r, a.Subnet, a.Port) || a.Name.empty()) return Fail(r, "Name an existing subnet port");
    r.patch<SubnetInterface>(a.Subnet, [&](auto &interface) { for (auto &port : interface.Ports) if (port.Id == a.Port) port.Name = a.Name; });
}
void Apply(state::Registry &r, const MovePort &a) {
    const auto *port = PortOf(r, a.Subnet, a.Port), *before = PortOf(r, a.Subnet, a.Before);
    if (!port || !before || port->Direction != before->Direction) return Fail(r, "Reorder ports in the same direction");
    if (a.Port == a.Before) return;
    r.patch<SubnetInterface>(a.Subnet, [&](auto &interface) {
        const auto from = std::ranges::find(interface.Ports, a.Port, &SubnetPort::Id);
        auto moved = std::move(*from);
        interface.Ports.erase(from);
        interface.Ports.insert(std::ranges::find(interface.Ports, a.Before, &SubnetPort::Id), std::move(moved));
    });
}
void Apply(state::Registry &r, const RemovePort &a) {
    if (!PortOf(r, a.Subnet, a.Port)) return Fail(r, "The subnet port does not exist");
    if (const auto *node = r.try_get<const ProcedureNode>(a.Subnet)) RewriteReferences(r, node->Procedure, a.Subnet, a.Port, false);
    r.patch<SubnetInterface>(a.Subnet, [&](auto &interface) { std::erase_if(interface.Ports, [&](const auto &p) { return p.Id == a.Port; }); });
}
} // namespace action::nodes

namespace procedure {
CookPlan ProjectNetwork(const state::Registry &r, state::Entity owner, state::Entity container) {
    CookPlan view{.InputFirst = {0u}, .SocketFirst = {0u}};
    const auto &procedure = r.get<const GeometryProcedure>(owner);
    if (container != state::Null) view.Nodes.push_back({container, KindOf<Subnet>});
    for (auto node = procedure.First; node != state::Null; node = r.get<const ProcedureNode>(node).Next) {
        if (ContainerOf(r, node) == container) view.Nodes.push_back({node, r.get<const ProcedureNode>(node).Kind});
    }
    std::vector<SocketLink> wires;
    for (uint32_t n = 0; n < view.Nodes.size(); ++n) {
        const auto [entity, kind] = view.Nodes[n];
        uint16_t inputs = 0;
        if (kind == KindOf<Subnet>) {
            const bool boundary = entity == container;
            for (const auto direction : {SocketDirection::In, SocketDirection::Out}) {
                for (const auto &port : r.get<const SubnetInterface>(entity).Ports) {
                    const auto facing = boundary ? (port.Direction == SocketDirection::In ? SocketDirection::Out : SocketDirection::In) : port.Direction;
                    if (facing != direction) continue;
                    view.ViewSockets.push_back({port.Type, facing, port.Types, false, port.Name.c_str()});
                    view.ViewSocketIds.push_back(port.Id);
                    if (facing == SocketDirection::In) {
                        ++inputs;
                        wires.push_back(port.Source());
                    }
                }
            }
        } else {
            const auto sockets = KindSockets(kind);
            for (uint16_t s = 0; s < sockets.size(); ++s) {
                view.ViewSockets.push_back(sockets[s]);
                view.ViewSocketIds.push_back(s);
            }
            const auto &links = r.get<const NodeInputs>(entity).Links;
            wires.insert(wires.end(), links.begin(), links.end());
            inputs = uint16_t(links.size());
        }
        view.InputFirst.push_back(view.InputFirst.back() + inputs);
        view.SocketFirst.push_back(uint32_t(view.ViewSockets.size()));
        view.Order.push_back(n);
    }
    IndexNodes(view);
    for (const auto &wire : wires) {
        const auto from = IndexOf(view, wire.FromNode);
        if (from == NoNode) {
            view.Wires.push_back({});
            continue;
        }
        const auto ids = std::span{view.ViewSocketIds}.subspan(view.SocketFirst[from], view.SocketFirst[from + 1] - view.SocketFirst[from]);
        const auto found = std::ranges::find(ids, wire.FromSocket);
        view.Wires.push_back(found == ids.end() ? NodeLink{} : NodeLink{from, uint16_t(found - ids.begin())});
    }
    for (const auto link : view.Wires) view.Inputs.push_back({link.FromNode, link.FromSocket});
    for (auto shown = procedure.DisplayNode; shown != state::Null; shown = ContainerOf(r, shown)) {
        view.DisplayNode = IndexOf(view, shown);
        if (view.DisplayNode != NoNode) break;
    }
    view.ResultNode = view.DisplayNode;
    return view;
}
} // namespace procedure
