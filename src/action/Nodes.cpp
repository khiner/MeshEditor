#include "action/Nodes.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "action/Selection.h"
#include "animation/AnimationData.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "procedure/Cook.h"
#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "state/EntityReferences.h"
#include "viewport/InteractionComponents.h"

namespace action::nodes {
namespace {
using namespace procedure;

bool HandleWritable(const state::Registry &r, state::Entity node, HandleParameter parameter) {
    const auto *record = r.try_get<const ProcedureNode>(node);
    if (!record || r.all_of<NodeBypassed>(node) || uint8_t(parameter) >= HandleSpecs.size()) return false;
    const auto &spec = HandleSpecs[uint8_t(parameter)];
    if (record->Kind != spec.Kind) return false;
    const auto sockets = KindSockets(record->Kind);
    for (uint32_t socket = 0; socket < InputCount[record->Kind]; ++socket) {
        if (!spec.Reads(sockets[socket])) continue;
        const auto input = ResolveSocket(r, InputOf(r, node, uint16_t(socket)), true);
        if (input.Source.FromNode != state::Null || input.Constant) return false;
    }
    return true;
}

struct NodeCopies {
    std::vector<std::pair<state::Entity, state::Entity>> Items;
    state::Entity operator()(state::Entity source) const {
        const auto at = std::ranges::lower_bound(Items, source, {}, &decltype(Items)::value_type::first);
        return at != Items.end() && at->first == source ? at->second : state::Null;
    }
};
NodeCopies CopyNodes(state::Registry &r, std::span<const state::Entity> originals) {
    NodeCopies copies;
    copies.Items.reserve(originals.size());
    for (const auto node : originals) copies.Items.emplace_back(node, r.create());
    std::ranges::sort(copies.Items);
    for (const auto node : originals) {
        const auto entity = copies(node);
        const auto copy_component = [&]<typename T> {
            if (const auto *value = r.try_get<const T>(node)) {
                T copied = *value;
                if constexpr (std::same_as<T, NodeContainer> || std::same_as<T, NetworkLayout> || std::same_as<T, SubnetInterface>) {
                    state::ForEachEntity(copied, [&](auto &e) {
                        const auto copy = copies(e);
                        if (copy != state::Null || std::same_as<T, SubnetInterface>) e = copy;
                    });
                    if constexpr (std::same_as<T, NetworkLayout>) std::ranges::sort(copied.Cells, {}, &NetworkCell::Item);
                }
                r.emplace<T>(entity, std::move(copied));
            }
        };
        VisitKind(r.get<const ProcedureNode>(node).Kind, copy_component);
        [&]<typename... T>(TypeList<T...>) { (copy_component.template operator()<T>(), ...); }(
            TypeList<SourceState, NodeContainer, SubnetInterface, NetworkLayout, Name, NodeLayout, InlineSelection, NodeBypassed, PublishSelection>{}
        );
    }
    return copies;
}
std::vector<SocketLink> CopyInputs(const state::Registry &r, state::Entity node, const NodeCopies &copies) {
    auto links = r.get<const NodeInputs>(node).Links;
    for (auto &link : links) {
        const auto copy = copies(link.FromNode);
        link = copy == state::Null ? SocketLink{} : SocketLink{copy, link.FromSocket};
    }
    return links;
}
std::vector<state::Entity> WithDescendants(const state::Registry &r, const GeometryProcedure &procedure, std::span<const state::Entity> roots) {
    std::vector<state::Entity> nodes;
    for (auto node = procedure.First; node != state::Null; node = r.get<const ProcedureNode>(node).Next) {
        for (auto ancestor = node; ancestor != state::Null; ancestor = ContainerOf(r, ancestor)) {
            if (!std::ranges::contains(roots, ancestor)) continue;
            nodes.push_back(node);
            break;
        }
    }
    return nodes;
}

bool ValidItem(const state::Registry &r, NetworkItem item) {
    if (item.Kind == ItemKind::Entity) return r.valid(item.Entity);
    if (item.Kind == ItemKind::Material) return item.Index < r.Resources.get<const MaterialStore>().Names.size();
    if (item.Kind == ItemKind::Animation) return item.Index < r.get<const Animations>(state::Document).Names.size();
    if (item.Kind == ItemKind::Texture) {
        const auto *textures = r.try_get<const MaterializedTextures>(state::Document);
        return textures && item.Index < textures->Items.size();
    }
    return false;
}

// Whether the node belongs to the mesh's procedure.
bool IsNodeOf(const state::Registry &r, state::Entity mesh_entity, state::Entity node) {
    const auto *record = r.try_get<const ProcedureNode>(node);
    return record && record->Procedure == mesh_entity;
}
// Whether the list holds distinct nodes of the mesh's procedure.
bool ListsNodes(const state::Registry &r, state::Entity mesh_entity, std::span<const state::Entity> nodes) {
    if (!std::ranges::all_of(nodes, [&](state::Entity node) { return IsNodeOf(r, mesh_entity, node); })) return false;
    std::vector<state::Entity> sorted{nodes.begin(), nodes.end()};
    std::ranges::sort(sorted);
    return std::ranges::adjacent_find(sorted) == sorted.end();
}
// The selected nodes of the mesh's procedure.
std::vector<state::Entity> SelectedNodes(const state::Registry &r, state::Entity mesh_entity) {
    std::vector<state::Entity> selected;
    if (const auto *active = r.try_get<const NetworkActive>(mesh_entity))
        for (const auto item : active->Selection)
            if (item.Kind == ItemKind::Entity && IsNodeOf(r, mesh_entity, item.Entity)) selected.push_back(item.Entity);
    return selected;
}
void Select(state::Registry &r, state::Entity mesh, std::span<const state::Entity> nodes, state::Entity active) {
    NetworkActive selection{NetworkItem{active}};
    for (const auto node : nodes) selection.Selection.push_back({node});
    std::ranges::sort(selection.Selection);
    r.emplace_or_replace<NetworkActive>(mesh, std::move(selection));
}

// Appends the node to the end of the procedure's authoring order with the links of its inputs.
void AddToProcedure(state::Registry &r, state::Entity mesh_entity, NodeRecord node, std::vector<SocketLink> links) {
    const auto last = r.get<const GeometryProcedure>(mesh_entity).Last;
    r.emplace<ProcedureNode>(node.Entity, mesh_entity, state::Null, node.Kind);
    r.emplace<NodeInputs>(node.Entity, std::move(links));
    if (last != state::Null) r.patch<ProcedureNode>(last, [&](ProcedureNode &record) { record.Next = node.Entity; });
    r.patch<GeometryProcedure>(mesh_entity, [&](GeometryProcedure &procedure) {
        if (procedure.First == state::Null) procedure.First = node.Entity;
        procedure.Last = node.Entity;
    });
}

// Mark the nodes reading the node, transitively.
// The plan's order visits each node after the nodes it reads.
std::vector<bool> Downstream(const CookPlan &plan, uint32_t node) {
    std::vector<bool> marked(plan.Nodes.size());
    for (const auto n : plan.Order) {
        marked[n] = std::ranges::any_of(plan.Inputs.begin() + plan.InputFirst[n], plan.Inputs.begin() + plan.InputFirst[n + 1u], [&](PlanLink link) { return link.FromNode != NoNode && (link.FromNode == node || marked[link.FromNode]); });
    }
    return marked;
}

std::optional<uint16_t> PrimaryInput(uint16_t kind) {
    if (!ResultSocket(kind)) return {};
    const auto sockets = KindSockets(kind);
    for (uint16_t s = 0; s < InputCount[kind]; ++s)
        if (sockets[s].Type == SocketType::Geometry || sockets[s].Type == SocketType::Instances) return s;
    return {};
}

// Follow the displayed result's primary inputs, including the ports forwarding them through subnets.
bool OnDisplayPath(const state::Registry &r, SocketLink display, state::Entity node, uint16_t socket) {
    while (display.FromNode != state::Null) {
        if (r.all_of<Subnet, NodeBypassed>(display.FromNode) && SocketOf(r, display.FromNode, display.FromSocket).Direction == SocketDirection::Out) {
            const auto input = SubnetBypassInput(r, display.FromNode, display.FromSocket);
            if (!input) return false;
            display.FromSocket = *input;
            continue;
        }
        const auto input = r.all_of<Subnet>(display.FromNode) ? std::optional{display.FromSocket} : PrimaryInput(r.get<const ProcedureNode>(display.FromNode).Kind);
        if (!input) return false;
        if (display.FromNode == node && *input == socket) return true;
        display = InputOf(r, display.FromNode, *input);
    }
    return false;
}

// Advance through the newly connected primary path, stopping at its endpoint or first branch.
SocketLink ExtendedDisplay(const state::Registry &r, const GeometryProcedure &procedure, const CookPlan &plan, const SocketEdit &edit) {
    const auto display = ResolveSocket(r, DisplayLink(r, procedure), false).Source;
    if (display.FromNode == state::Null || ResolveSocket(r, edit.Source, false).Source != display) return {};
    if (OnDisplayPath(r, DisplayLink(r, procedure), edit.Node, edit.Socket)) return {};
    const auto count = uint32_t(plan.Nodes.size()), shown = IndexOf(plan, display.FromNode);
    std::vector<uint32_t> next(count, NoNode);
    uint32_t start = NoNode;
    for (uint32_t n = 0; n < count; ++n) {
        const auto input = PrimaryInput(plan.Nodes[n].Kind);
        if (!input) continue;
        const auto link = plan.Inputs[plan.InputFirst[n] + *input];
        if (link.FromNode == NoNode) continue;
        if (ResultSocket(plan.Nodes[link.FromNode].Kind) != link.FromSocket) continue;
        auto &reader = next[link.FromNode];
        reader = reader == NoNode ? n : count;
        if (link.FromNode == shown && link.FromSocket == display.FromSocket && ResolveSocket(r, InputOf(r, plan.Nodes[n].Entity, *input), false).Source != display)
            start = start == NoNode ? n : count;
    }
    if (start >= count) return {};
    while (next[start] < count) start = next[start];
    const SocketLink result{plan.Nodes[start].Entity, *ResultSocket(plan.Nodes[start].Kind)};
    auto output = result;
    for (auto parent = ContainerOf(r, output.FromNode); parent != state::Null; parent = ContainerOf(r, parent)) {
        std::optional<uint16_t> port;
        for (const auto &p : r.get<const SubnetInterface>(parent).Ports) {
            if (p.Direction != SocketDirection::Out || ResolveSocket(r, {parent, p.Id}, false, &edit).Source != result) continue;
            if (port) return output;
            port = p.Id;
        }
        if (!port) break;
        output = {parent, *port};
    }
    return output;
}

template<typename K> bool Rename(state::Registry &r, state::Entity node, std::string_view name) {
    if (!r.valid(node) || !r.all_of<K>(node)) return false;
    r.patch<K>(node, [&](K &p) {
        p.Name = {};
        std::ranges::copy(name, p.Name.begin());
    });
    return true;
}
} // namespace

void Apply(state::Registry &r, const Action &action) {
    // The mesh entity's procedure, or null after failing the action.
    const auto procedure_of = [&](state::Entity mesh_entity) {
        const auto *procedure = r.try_get<const GeometryProcedure>(mesh_entity);
        if (!procedure) Fail(r, "The mesh has no geometry procedure");
        return procedure;
    };
    const auto kind_of = [&](state::Entity node) { return r.get<const ProcedureNode>(node).Kind; };
    std::visit(
        overloaded{
            [&](const AddNode &a) {
                if (!procedure_of(a.MeshEntity)) return;
                if (a.Kind >= KindCount) return Fail(r, "Unknown node kind");
                if (a.Parent != state::Null && (!IsNodeOf(r, a.MeshEntity, a.Parent) || !r.all_of<Subnet>(a.Parent))) return Fail(r, "The parent network does not exist");
                const auto node = CreateNode(r, a.Kind, a.Position);
                if (a.Parent != state::Null) r.emplace<NodeContainer>(node, a.Parent);
                if (a.Kind == KindOf<Subnet>) {
                    r.emplace<SubnetInterface>(node);
                    EmplaceUniqueName(r, node, "Subnet");
                }
                if (a.Kind == KindOf<Source>) r.emplace<SourceState>(node);
                AddToProcedure(r, a.MeshEntity, {node, a.Kind}, std::vector<SocketLink>(InputCount[a.Kind]));
                Select(r, a.MeshEntity, std::array{node}, node);
            },
            [&](const Link &a) {
                if (!procedure_of(a.MeshEntity)) return;
                if (!IsNodeOf(r, a.MeshEntity, a.FromNode) || !IsNodeOf(r, a.MeshEntity, a.ToNode)) return Fail(r, "A link refers to a missing node");
                if ((r.all_of<Subnet>(a.FromNode) ? !PortOf(r, a.FromNode, a.FromSocket) : a.FromSocket >= SocketCount[kind_of(a.FromNode)]) ||
                    (r.all_of<Subnet>(a.ToNode) ? !PortOf(r, a.ToNode, a.ToSocket) : a.ToSocket >= InputCount[kind_of(a.ToNode)])) return Fail(r, "A link refers to a missing socket");
                if (ContainerOf(r, a.FromNode) != ContainerOf(r, a.ToNode) && ContainerOf(r, a.FromNode) != a.ToNode && ContainerOf(r, a.ToNode) != a.FromNode) return Fail(r, "Connect across network boundaries through their ports");
                auto from = SocketOf(r, a.FromNode, a.FromSocket), to = SocketOf(r, a.ToNode, a.ToSocket);
                if (r.all_of<Subnet>(a.FromNode) && (ContainerOf(r, a.ToNode) == a.FromNode || a.FromNode == a.ToNode)) from.Direction = from.Direction == SocketDirection::In ? SocketDirection::Out : SocketDirection::In;
                if (r.all_of<Subnet>(a.ToNode) && (ContainerOf(r, a.FromNode) == a.ToNode || a.FromNode == a.ToNode)) to.Direction = to.Direction == SocketDirection::In ? SocketDirection::Out : SocketDirection::In;
                if (from.Direction != SocketDirection::Out || to.Direction != SocketDirection::In || !Links(from, to)) return Fail(r, "Link compatible output and input sockets");
                const SocketEdit edit{a.ToNode, a.ToSocket, {a.FromNode, a.FromSocket}};
                const auto plan = Plan(r, r.get<const GeometryProcedure>(a.MeshEntity), false, &edit);
                if (plan.Order.size() != plan.Nodes.size()) return Fail(r, "A link cannot close a cycle");
                const auto display = ExtendedDisplay(r, r.get<const GeometryProcedure>(a.MeshEntity), plan, edit);
                SetInput(r, a.ToNode, a.ToSocket, edit.Source);
                if (display.FromNode != state::Null) Apply(r, Action{SetDisplay{a.MeshEntity, display.FromNode, display.FromSocket}});
            },
            [&](const Unlink &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (!IsNodeOf(r, a.MeshEntity, a.ToNode) || (r.all_of<Subnet>(a.ToNode) ? !PortOf(r, a.ToNode, a.ToSocket) : a.ToSocket >= InputCount[kind_of(a.ToNode)])) return Fail(r, "An unlink refers to a missing input");
                const auto source = InputOf(r, a.ToNode, a.ToSocket);
                auto display = OnDisplayPath(r, DisplayLink(r, *procedure), a.ToNode, a.ToSocket) ? source : SocketLink{};
                while (display.FromNode != state::Null && SocketOf(r, display.FromNode, display.FromSocket).Direction == SocketDirection::In)
                    display = InputOf(r, display.FromNode, display.FromSocket);
                SetInput(r, a.ToNode, a.ToSocket, {});
                if (display.FromNode != state::Null) Apply(r, Action{SetDisplay{a.MeshEntity, display.FromNode, display.FromSocket}});
            },
            [&](const SetDisplay &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                std::optional<uint16_t> port;
                if (a.Node != state::Null) {
                    if (!IsNodeOf(r, a.MeshEntity, a.Node)) return Fail(r, "A display refers to a missing node");
                    const auto socket = a.Port ? a.Port : ResultSocket(r, a.Node);
                    if (!socket) return Fail(r, "Only geometry or instances can be displayed");
                    const auto descriptor = SocketOf(r, a.Node, *socket);
                    if (descriptor.Direction != SocketDirection::Out || (descriptor.Type != SocketType::Geometry && descriptor.Type != SocketType::Instances)) return Fail(r, "Only geometry or instances can be displayed");
                    if (r.all_of<Subnet>(a.Node)) port = socket;
                }
                if (procedure->DisplayNode != a.Node || procedure->DisplayPort != port) r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &edited) { edited.DisplayNode = a.Node; edited.DisplayPort = port; });
            },
            [&](const SelectItems &a) {
                if (!r.valid(a.Network)) return Fail(r, "The network does not exist");
                auto items = a.Items;
                std::ranges::sort(items);
                if (std::ranges::adjacent_find(items) != items.end()) return Fail(r, "The selection must list distinct items");
                if (r.any_of<GeometryProcedure, Subnet>(a.Network)) {
                    const auto owner = r.all_of<Subnet>(a.Network) ? r.get<const ProcedureNode>(a.Network).Procedure : a.Network;
                    const auto valid = [&](NetworkItem item) { return item.Kind == ItemKind::Entity && IsNodeOf(r, owner, item.Entity); };
                    if (!std::ranges::all_of(items, valid) || (a.Active != NetworkItem{} && !valid(a.Active))) return Fail(r, "Select nodes of the procedure");
                    r.emplace_or_replace<NetworkActive>(owner, a.Active != NetworkItem{} ? a.Active : NetworkItem{ActiveNodeOf(r, owner)}, std::move(items));
                    return;
                }
                if (!std::ranges::includes(NetworkItems(r, a.Network), items)) return Fail(r, "The selection contains items outside this network");
                if (a.Active != NetworkItem{} && !std::ranges::binary_search(items, a.Active)) return Fail(r, "The active item must be selected");
                auto active = a.Active;
                if (active == NetworkItem{}) {
                    const auto *previous = r.try_get<const NetworkActive>(a.Network);
                    active = previous && std::ranges::binary_search(items, previous->Item) ? previous->Item : items.empty() ? NetworkItem{} :
                                                                                                                              items.front();
                }
                selection::ApplyTreeSelection selection;
                if (const auto *previous = r.try_get<const NetworkActive>(a.Network)) {
                    for (const auto item : previous->Selection) {
                        if (!std::ranges::binary_search(items, item) && item.Kind == ItemKind::Entity && r.any_of<ObjectKind, BoneIndex>(item.Entity)) selection.Add(item.Entity, false);
                    }
                }
                for (const auto item : items) {
                    if (item.Kind == ItemKind::Entity && r.any_of<ObjectKind, BoneIndex>(item.Entity)) selection.Add(item.Entity, true);
                }
                if (!selection.ToSelect().empty()) selection.Clear = std::ranges::all_of(selection.ToSelect(), [&](auto e) { return r.all_of<BoneIndex>(e); }) ? selection::ApplyTreeSelection::ClearKind::BonesOnly : selection::ApplyTreeSelection::ClearKind::All;
                if (active.Kind == ItemKind::Entity && r.any_of<ObjectKind, BoneIndex>(active.Entity)) selection.NavToActive = active.Entity;
                selection::Apply(r, selection);
                r.emplace_or_replace<NetworkActive>(a.Network, active, std::move(items));
            },
            [&](const MoveNodes &a) {
                if (!r.valid(a.Network)) return Fail(r, "The network does not exist");
                const bool geometry = r.any_of<GeometryProcedure, Subnet>(a.Network);
                const auto items = geometry ? std::vector<NetworkItem>{} : NetworkItems(r, a.Network);
                for (const auto &p : a.Positions) {
                    if (!ValidItem(r, p.Node)) return Fail(r, "A layout refers to a missing item");
                    if (!geometry && !std::ranges::binary_search(items, p.Node)) return Fail(r, "A layout item is outside this network");
                    if (p.Node.Kind != ItemKind::Entity) continue;
                    if (const auto *node = r.try_get<const ProcedureNode>(p.Node.Entity); node && node->Procedure != a.Network && (!r.all_of<Subnet>(a.Network) || r.get<const ProcedureNode>(a.Network).Procedure != node->Procedure)) return Fail(r, "A layout refers to another procedure");
                }
                for (const auto &p : a.Positions) {
                    if (p.Node.Kind == ItemKind::Entity && r.all_of<ProcedureNode>(p.Node.Entity) && p.Node.Entity != a.Network) r.patch<NodeLayout>(p.Node.Entity, [&](auto &layout) { layout.Position = p.Position; });
                    else SetNetworkPosition(r, a.Network, p.Node, p.Position);
                }
            },
            [&](const SetExtrudeOffset &a) {
                if (!r.all_of<mesh::Extrude>(a.Node)) return Fail(r, "The operation is not an extrusion");
                if (!HandleWritable(r, a.Node, HandleParameter::ExtrudeOffset)) return Fail(r, "Disconnect the extrusion offset inputs before editing the offset handle");
                r.patch<mesh::Extrude>(a.Node, [&](auto &p) {
                    p.Offset = a.Value;
                    p.OffsetScale = 1.f;
                    p.AlongNormals = false;
                });
            },
            [&](const SetOperationHandle &a) {
                if (a.Node == state::Null) {
                    r.remove<OperationHandle>(state::Document);
                    return;
                }
                const auto *node = r.try_get<const ProcedureNode>(a.Node);
                const auto *object = r.try_get<const Instance>(a.Object);
                if (!node || !object || object->Entity != node->Procedure || !HandleWritable(r, a.Node, a.Parameter)) {
                    Fail(r, "The operation needs a scene placement and a cooked input for its viewport handle");
                    return;
                }
                PrepareOperationHandle(r, a.Node);
                r.emplace_or_replace<OperationHandle>(state::Document, a.Node, a.Object, a.Parameter);
            },
            [&](const SetBypass &a) {
                const auto *node = r.try_get<const ProcedureNode>(a.Node);
                if (!node) return Fail(r, "The node does not exist");
                if (node->Kind == KindOf<Subnet>) {
                    const auto &ports = r.get<const SubnetInterface>(a.Node).Ports;
                    if (a.Enabled && (std::ranges::none_of(ports, [](const auto &p) { return p.Direction == SocketDirection::Out; }) || std::ranges::any_of(ports, [&](const auto &p) { return p.Direction == SocketDirection::Out && !SubnetBypassInput(r, a.Node, p.Id); }))) return Fail(r, "Each bypassed output needs one compatible input");
                } else if (!BypassInput(node->Kind)) return Fail(r, "The node has no compatible bypass input");
                if (a.Enabled && !r.all_of<NodeBypassed>(a.Node)) r.emplace<NodeBypassed>(a.Node);
                else if (!a.Enabled) r.remove<NodeBypassed>(a.Node);
                r.remove<NodeError>(a.Node);
            },
            [&](const SetAttributeName &a) {
                if (a.Name.empty() || a.Name.size() >= AttributeName{}.size() || a.Name.front() == '.') return Fail(r, "An attribute name needs 1 to 63 characters and no leading '.'");
                if (!Rename<procedure::NamedAttribute>(r, a.Node, a.Name) && !Rename<StoreNamedAttribute>(r, a.Node, a.Name) && !Rename<StoreSelection>(r, a.Node, a.Name)) Fail(r, "Only attribute name nodes take a name");
            },
            [&](const SetInlineSelection &a) {
                const auto *node = r.try_get<const ProcedureNode>(a.Node);
                if (!node || !std::ranges::contains(KindSockets(node->Kind), SocketType::Selection, &Socket::Type)) return Fail(r, "Only a node with a selection input takes an inline selection");
                r.emplace_or_replace<InlineSelection>(a.Node, *a.Selection);
            },
            [&](const CollapseProcedure &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                const auto *evaluated = procedure ? r.try_get<const EvaluatedMesh>(a.MeshEntity) : nullptr;
                if (!evaluated) return;
                // The new Source node takes the evaluated record's state where its history step ends, and the mesh keeps drawing it.
                const auto record = evaluated->StoreId;
                const auto nodes = NodesOf(r, *procedure);
                // Detach published instance lists before destroying the nodes to retain the objects.
                if (r.all_of<ProcedureInstances>(a.MeshEntity)) r.replace<ProcedureInstances>(a.MeshEntity);
                if (r.all_of<ProcedureOwnedMesh>(a.MeshEntity)) r.replace<ProcedureOwnedMesh>(a.MeshEntity, state::Null);
                r.remove<ProcedureInstances, ProcedureOwnedMesh>(a.MeshEntity);
                const auto source = CreateNode(r, KindOf<Source>, {0u, 0u});
                r.emplace<SourceState>(source, store::Hash128{}, record);
                RootProcedure(r, a.MeshEntity, source, KindOf<Source>);
                r.replace<EvaluatedMesh>(a.MeshEntity, record, source);
                r.destroy(nodes);
            },
            [&](const DuplicateNodes &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (a.Nodes.empty() || !ListsNodes(r, a.MeshEntity, a.Nodes)) return Fail(r, "A duplicate lists distinct nodes of the procedure");
                // A copied Source node's state refers to the original's geometry, so a pending state is first kept as it stands.
                FinishSourceStates(r);
                const auto originals = WithDescendants(r, *procedure, a.Nodes);
                const auto active = ActiveNodeOf(r, a.MeshEntity);
                const auto copies = CopyNodes(r, originals);
                std::vector<state::Entity> selected;
                for (const auto node : originals) {
                    const auto copy = copies(node);
                    if (std::ranges::contains(a.Nodes, node)) selected.push_back(copy);
                    AddToProcedure(r, a.MeshEntity, {copy, kind_of(node)}, CopyInputs(r, node, copies));
                }
                const auto copied_active = copies(active);
                Select(r, a.MeshEntity, selected, copied_active != state::Null ? copied_active : copies(originals.back()));
            },
            [&](const DeleteNodes &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (a.Nodes.empty() || !ListsNodes(r, a.MeshEntity, a.Nodes)) return Fail(r, "A delete lists distinct nodes of the procedure");
                const auto nodes = NodesOf(r, *procedure);
                const auto removed_entities = WithDescendants(r, *procedure, a.Nodes);
                auto removed_sorted = removed_entities;
                std::ranges::sort(removed_sorted);
                const auto is_removed = [&](state::Entity node) { return std::ranges::binary_search(removed_sorted, node); };
                const auto through = [&](this const auto &self, SocketLink link) -> SocketLink {
                    if (link.FromNode == state::Null || !is_removed(link.FromNode)) return link;
                    if (const auto *port = PortOf(r, link.FromNode, link.FromSocket)) return self(port->Source());
                    const auto kind = kind_of(link.FromNode);
                    const auto sockets = KindSockets(kind);
                    SocketLink source{};
                    uint32_t sources = 0;
                    const auto &inputs = r.get<const NodeInputs>(link.FromNode).Links;
                    for (uint16_t i = 0; i < inputs.size(); ++i) {
                        const auto read = sockets[i].Type == sockets[link.FromSocket].Type ? self(inputs[i]) : SocketLink{};
                        if (read.FromNode == state::Null) continue;
                        source = read;
                        ++sources;
                    }
                    return sources == 1 ? source : SocketLink{};
                };
                for (const auto node : nodes) {
                    if (is_removed(node)) continue;
                    const auto reconnect = [&](uint16_t socket, SocketLink link) {
                        if (!is_removed(link.FromNode)) return;
                        const auto source = through(link);
                        SetInput(r, node, socket, source.FromNode != state::Null && Links(SocketOf(r, source.FromNode, source.FromSocket), SocketOf(r, node, socket)) ? source : SocketLink{});
                    };
                    if (const auto *interface = r.try_get<const SubnetInterface>(node)) {
                        const auto ports = interface->Ports;
                        for (const auto &port : ports) reconnect(port.Id, port.Source());
                    } else {
                        const auto inputs = r.get<const NodeInputs>(node).Links;
                        for (uint16_t i = 0; i < inputs.size(); ++i) reconnect(i, inputs[i]);
                    }
                }
                // The surviving nodes keep their authoring order.
                auto first = state::Null, last = state::Null;
                for (const auto node : nodes) {
                    if (is_removed(node)) continue;
                    if (last == state::Null) first = node;
                    else if (r.get<const ProcedureNode>(last).Next != node) r.patch<ProcedureNode>(last, [&](ProcedureNode &record) { record.Next = node; });
                    last = node;
                }
                if (last != state::Null && r.get<const ProcedureNode>(last).Next != state::Null) r.patch<ProcedureNode>(last, [](ProcedureNode &record) { record.Next = state::Null; });
                // The display reads the displayed node's result like a consumer and follows its chain to a surviving node.
                // A chain with several inputs, such as a Join's, leaves the display on the node its first input reads.
                auto display = procedure->DisplayNode;
                auto display_port = procedure->DisplayPort;
                if (is_removed(display)) {
                    auto shown = through(DisplayLink(r, *procedure));
                    const auto &inputs = r.get<const NodeInputs>(display).Links;
                    if (shown.FromNode == state::Null && !inputs.empty()) shown = through(inputs.front());
                    display = shown.FromNode;
                    display_port = r.all_of<Subnet>(display) ? std::optional{shown.FromSocket} : std::optional<uint16_t>{};
                }
                // The displayed node, or else the last node, replaces a removed active node.
                auto active = ActiveNodeOf(r, a.MeshEntity);
                if (!std::ranges::contains(nodes, active) || is_removed(active)) active = display != state::Null ? display : last;
                if (const GeometryProcedure edited{first, last, display, display_port}; edited != *procedure) r.replace<GeometryProcedure>(a.MeshEntity, edited);
                auto selected = SelectedNodes(r, a.MeshEntity);
                std::erase_if(selected, is_removed);
                Select(r, a.MeshEntity, selected, active);
                r.destroy(removed_entities);
            },
            [&](const InsertNode &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (!IsNodeOf(r, a.MeshEntity, a.Node) || !IsNodeOf(r, a.MeshEntity, a.ToNode)) return Fail(r, "An insert refers to a missing node or input");
                const auto plan = ProjectNetwork(r, a.MeshEntity, ContainerOf(r, a.Node));
                const auto node = IndexOf(plan, a.Node), to_node = IndexOf(plan, a.ToNode);
                if (node == NoNode || to_node == NoNode) return Fail(r, "Insert within one network");
                uint16_t socket = 0;
                while (socket < InputsOf(plan, to_node) && SocketId(plan, to_node, socket) != a.ToSocket) ++socket;
                if (socket == InputsOf(plan, to_node) || !Inserts(plan, node, to_node, socket)) return Fail(r, "A node inserts on a link through geometry sockets that link with its ends, without closing a cycle");
                // The input's node and the nodes reading it, transitively.
                if (a.Shift) {
                    auto moved = Downstream(plan, to_node);
                    moved[to_node] = true;
                    for (uint32_t n = 0u; n < moved.size(); ++n) {
                        if (moved[n]) r.patch<NodeLayout>(plan.Nodes[n].Entity, [&](NodeLayout &layout) { layout.Position.x += a.Shift; });
                    }
                }
                const auto kind = plan.Nodes[node].Kind;
                const auto input = *MainSocket(kind, SocketDirection::In);
                const auto link = plan.Wires[plan.InputFirst[to_node] + socket];
                if (r.get<const NodeInputs>(a.Node).Links[input].FromNode == state::Null) SetInput(r, a.Node, input, {plan.Nodes[link.FromNode].Entity, SocketId(plan, link.FromNode, link.FromSocket)});
                SetInput(r, a.ToNode, a.ToSocket, {a.Node, *MainSocket(kind, SocketDirection::Out)});
            },
            [&](const auto &a) { Apply(r, a); },
        },
        action
    );
}

SetInlineSelection CaptureSelection(const state::Registry &r, state::Entity mesh_entity, state::Entity node) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto id = *DrawnStoreId(r, mesh_entity);
    const auto *active = r.try_get<const MeshActiveElement>(mesh_entity);
    auto selection = std::make_unique<InlineSelection>(InlineSelection{.Domain = r.get<const EditMode>(state::Document).Value, .Active = active ? active->Handle : InvalidOffset});
    for (const auto [element, captured] : {std::pair{Element::Vertex, &selection->Vertices}, std::pair{Element::Edge, &selection->Edges}, std::pair{Element::Face, &selection->Faces}}) {
        const auto selected = meshes.GetSelectedElements(id, element);
        std::vector<SelectedBlock> blocks;
        selected.ForEachBlock([&](uint32_t block, uint32_t) {
            SelectionBlock words;
            std::ranges::copy(selected.Bits.subspan(block * MeshElementBlockWords, MeshElementBlockWords), words.begin());
            blocks.push_back({selected.Membership[block].Local, words});
        });
        *captured = CompactSelection(std::move(blocks));
    }
    return {node, std::move(selection)};
}

void CaptureActiveSelection(state::Registry &r, state::Entity mesh_entity) {
    const auto *procedure = r.try_get<const GeometryProcedure>(mesh_entity);
    if (const auto node = procedure ? ViewportSelectionNode(r, *procedure) : state::Null; node != state::Null) Apply(r, CaptureSelection(r, mesh_entity, node));
}

state::Entity CopyProcedure(state::Registry &r, state::Entity from, uint32_t drawn) {
    FinishSourceStates(r);
    const auto procedure = r.get<const GeometryProcedure>(from);
    const auto originals = NodesOf(r, procedure);
    const auto copies = CopyNodes(r, originals);
    const auto to = r.create();
    for (const auto node : originals) {
        const auto copy = copies(node);
        auto record = r.get<const ProcedureNode>(node);
        record.Procedure = to;
        record.Next = copies(record.Next);
        r.emplace<ProcedureNode>(copy, record);
        r.emplace<NodeInputs>(copy, CopyInputs(r, node, copies));
        if (const auto *result = r.try_get<const NodeResult>(node)) {
            auto copied = *result;
            copied.Input = copies(result->Input);
            r.emplace<NodeResult>(copy, std::move(copied));
        }
        if (const auto *placement = r.try_get<const BooleanPlacement>(node)) r.emplace<BooleanPlacement>(copy, *placement);
    }
    r.emplace<GeometryProcedure>(to, copies(procedure.First), copies(procedure.Last), copies(procedure.DisplayNode), procedure.DisplayPort);
    auto selected = SelectedNodes(r, from);
    for (auto &node : selected) node = copies(node);
    Select(r, to, selected, copies(ActiveNodeOf(r, from)));
    r.emplace<EvaluatedMesh>(to, drawn, copies(r.get<const EvaluatedMesh>(from).Tip));
    return to;
}

state::Entity EditedMesh(const state::Registry &r) {
    const auto active = FindActiveEntity(r);
    if (active == state::Null) return state::Null;
    return HasMesh(r, active) ? active : GetMeshEntity(r, active);
}

state::Entity AppendNode(state::Registry &r, state::Entity mesh_entity, uint16_t kind) {
    const auto &procedure = r.get<const GeometryProcedure>(mesh_entity);
    const auto display = procedure.DisplayNode;
    const auto result = procedure.DisplayPort ? procedure.DisplayPort : ResultSocket(r, display);
    if (!result || !Links(SocketOf(r, display, *result), KindSockets(kind)[0])) {
        Fail(r, "The displayed node has no geometry result to append to");
        return state::Null;
    }
    const auto displayed = r.get<const NodeLayout>(display).Position;
    const auto plan = Plan(r, procedure);
    const auto moved = Downstream(plan, plan.ResultNode);
    for (uint32_t n = 0u; n < moved.size(); ++n) {
        if (moved[n] && ContainerOf(r, plan.Nodes[n].Entity) == ContainerOf(r, display)) r.patch<NodeLayout>(plan.Nodes[n].Entity, [](NodeLayout &layout) { layout.Position.x += AppendGap; });
    }
    Apply(r, Action{AddNode{mesh_entity, kind, {displayed.x + AppendGap, displayed.y}, ContainerOf(r, display)}});
    const auto entity = r.get<const GeometryProcedure>(mesh_entity).Last;
    if (std::ranges::contains(KindSockets(kind), SocketType::Selection, &Socket::Type)) Apply(r, CaptureSelection(r, mesh_entity, entity));
    Apply(r, Link{mesh_entity, display, *result, entity, 0u});
    Apply(r, SetDisplay{mesh_entity, entity});
    r.emplace<PublishSelection>(entity);
    return entity;
}

state::Entity StagedResultNode(const state::Registry &r, state::Entity mesh_entity, uint16_t kind) {
    const auto display = r.get<const GeometryProcedure>(mesh_entity).DisplayNode;
    return display != state::Null && r.get<const ProcedureNode>(display).Kind == kind && r.all_of<StagedNode>(display) ? display : state::Null;
}

void ForEachResultNode(const state::Registry &r, const std::function<void(state::Entity, state::Entity)> &fn) {
    for (const auto mesh_entity : mesh::SelectedEditMeshes(r)) {
        const auto *procedure = r.try_get<const GeometryProcedure>(mesh_entity);
        if (procedure && procedure->DisplayNode != state::Null) fn(mesh_entity, procedure->DisplayNode);
    }
}
} // namespace action::nodes

namespace action {
state::Entity UpdateTraits<procedure::NodeLayout>::Active(const state::Registry &r) {
    return procedure::ActiveNodeOf(r, nodes::EditedMesh(r));
}
void UpdateTraits<procedure::NodeLayout>::ForEachSelected(state::Registry &r, const std::function<void(state::Entity)> &fn) {
    const auto mesh_entity = nodes::EditedMesh(r);
    if (!r.all_of<procedure::GeometryProcedure>(mesh_entity)) return;
    for (const auto node : nodes::SelectedNodes(r, mesh_entity)) fn(node);
}
} // namespace action
