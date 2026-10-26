#include "action/Nodes.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "procedure/Cook.h"
#include "procedure/Procedure.h"
#include "viewport/InteractionComponents.h"

namespace action::nodes {
namespace {
using namespace procedure;

// Copies the node's parameters, layout, selection and a Source node's state, which refers to the same geometry.
state::Entity CopyNode(state::Registry &r, state::Entity node, uint16_t kind) {
    const auto copy = r.create();
    VisitKind(kind, [&]<typename K> {
        const auto parameters = r.get<const K>(node);
        r.emplace<K>(copy, parameters);
    });
    if (const auto *source = r.try_get<const SourceState>(node)) {
        const auto copied = *source;
        r.emplace<SourceState>(copy, copied);
    }
    const auto layout = r.get<const NodeLayout>(node);
    r.emplace<NodeLayout>(copy, layout);
    if (const auto *selection = r.try_get<const InlineSelection>(node)) {
        auto copied = *selection;
        r.emplace<InlineSelection>(copy, std::move(copied));
    }
    if (r.all_of<PublishSelection>(node)) r.emplace<PublishSelection>(copy);
    return copy;
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
// The node's index in the plan, or NoNode.
uint32_t IndexOf(const CookPlan &plan, state::Entity node) {
    const auto it = std::ranges::find(plan.Nodes, node, &NodeRecord::Entity);
    return it == plan.Nodes.end() ? NoNode : uint32_t(it - plan.Nodes.begin());
}

// The selected nodes of the mesh's procedure.
std::vector<state::Entity> SelectedNodes(const state::Registry &r, state::Entity mesh_entity) {
    std::vector<state::Entity> selected;
    for (const auto node : r.view<const NodeSelected>()) {
        if (IsNodeOf(r, mesh_entity, node)) selected.push_back(node);
    }
    return selected;
}
// Deselects the selected nodes of the mesh's procedure other than the kept ones.
void DeselectNodes(state::Registry &r, state::Entity mesh_entity, std::span<const state::Entity> kept = {}) {
    for (const auto node : SelectedNodes(r, mesh_entity)) {
        if (!std::ranges::contains(kept, node)) r.remove<NodeSelected>(node);
    }
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

// Whether the node reads the target through its links, transitively, or is the target.
bool Reads(const state::Registry &r, state::Entity node, state::Entity target) {
    std::vector<state::Entity> stack{node}, seen{node};
    while (!stack.empty()) {
        const auto e = stack.back();
        stack.pop_back();
        if (e == target) return true;
        for (const auto &link : r.get<const NodeInputs>(e).Links) {
            if (link.FromNode == state::Null) continue;
            const auto at = std::ranges::lower_bound(seen, link.FromNode);
            if (at != seen.end() && *at == link.FromNode) continue;
            seen.insert(at, link.FromNode);
            stack.push_back(link.FromNode);
        }
    }
    return false;
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
    const auto set_input = [&](state::Entity node, uint16_t socket, SocketLink link) {
        r.patch<NodeInputs>(node, [&](NodeInputs &inputs) { inputs.Links[socket] = link; });
    };
    const auto kind_of = [&](state::Entity node) { return r.get<const ProcedureNode>(node).Kind; };
    std::visit(
        overloaded{
            [&](const AddNode &a) {
                if (!procedure_of(a.MeshEntity)) return;
                if (a.Kind >= KindCount) return Fail(r, "Unknown node kind");
                const auto node = CreateNode(r, a.Kind, a.Position);
                if (a.Kind == KindOf<Source>) r.emplace<SourceState>(node);
                DeselectNodes(r, a.MeshEntity);
                r.emplace<NodeSelected>(node);
                AddToProcedure(r, a.MeshEntity, {node, a.Kind}, std::vector<SocketLink>(InputCount[a.Kind]));
                r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &procedure) { procedure.ActiveNode = node; });
            },
            [&](const Link &a) {
                if (!procedure_of(a.MeshEntity)) return;
                if (!IsNodeOf(r, a.MeshEntity, a.FromNode) || !IsNodeOf(r, a.MeshEntity, a.ToNode)) return Fail(r, "A link refers to a missing node");
                const auto from = KindSockets(kind_of(a.FromNode)), to = KindSockets(kind_of(a.ToNode));
                if (a.FromSocket >= from.size() || from[a.FromSocket].Direction != SocketDirection::Out ||
                    a.ToSocket >= to.size() || to[a.ToSocket].Direction != SocketDirection::In) return Fail(r, "A link joins an output socket to an input socket");
                if (!Links(from[a.FromSocket], to[a.ToSocket])) return Fail(r, "Linked sockets differ in type");
                if (Reads(r, a.FromNode, a.ToNode)) return Fail(r, "A link cannot close a cycle");
                set_input(a.ToNode, a.ToSocket, {a.FromNode, a.FromSocket});
            },
            [&](const Unlink &a) {
                if (!procedure_of(a.MeshEntity)) return;
                if (!IsNodeOf(r, a.MeshEntity, a.ToNode) || a.ToSocket >= InputCount[kind_of(a.ToNode)]) return Fail(r, "An unlink refers to a missing input");
                set_input(a.ToNode, a.ToSocket, {});
            },
            [&](const SetDisplay &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (!IsNodeOf(r, a.MeshEntity, a.Node)) return Fail(r, "A display refers to a missing node");
                if (procedure->DisplayNode != a.Node) r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &edited) { edited.DisplayNode = a.Node; });
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
                const auto nodes = Plan(r, *procedure).Nodes;
                // Detach published instance lists before destroying the nodes to retain the objects.
                if (r.all_of<ProcedureInstances>(a.MeshEntity)) r.replace<ProcedureInstances>(a.MeshEntity);
                if (r.all_of<ProcedureOwnedMesh>(a.MeshEntity)) r.replace<ProcedureOwnedMesh>(a.MeshEntity, state::Null);
                r.remove<ProcedureInstances, ProcedureOwnedMesh>(a.MeshEntity);
                const auto source = CreateNode(r, KindOf<Source>, {0u, 0u});
                r.emplace<SourceState>(source, store::Hash128{}, record);
                RootProcedure(r, a.MeshEntity, source, KindOf<Source>);
                r.replace<EvaluatedMesh>(a.MeshEntity, record, source);
                for (const auto &node : nodes) r.destroy(node.Entity);
            },
            [&](const DuplicateNodes &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (a.Nodes.empty() || !ListsNodes(r, a.MeshEntity, a.Nodes)) return Fail(r, "A duplicate lists distinct nodes of the procedure");
                // A copied Source node's state refers to the original's geometry, so a pending state is first kept as it stands.
                FinishSourceStates(r);
                const auto active = procedure->ActiveNode;
                std::vector<state::Entity> copies;
                for (const auto node : a.Nodes) copies.push_back(CopyNode(r, node, kind_of(node)));
                const auto copy_of = [&](state::Entity node) {
                    const auto it = std::ranges::find(a.Nodes, node);
                    return it == a.Nodes.end() ? state::Null : copies[size_t(it - a.Nodes.begin())];
                };
                DeselectNodes(r, a.MeshEntity);
                for (size_t i = 0u; i < copies.size(); ++i) {
                    r.emplace<NodeSelected>(copies[i]);
                    auto links = r.get<const NodeInputs>(a.Nodes[i]).Links;
                    for (auto &link : links) link = link.FromNode != state::Null && copy_of(link.FromNode) != state::Null ? SocketLink{copy_of(link.FromNode), link.FromSocket} : SocketLink{};
                    AddToProcedure(r, a.MeshEntity, {copies[i], kind_of(a.Nodes[i])}, std::move(links));
                }
                const auto copied_active = copy_of(active);
                r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &edited) { edited.ActiveNode = copied_active != state::Null ? copied_active : copies.back(); });
            },
            [&](const DeleteNodes &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (a.Nodes.empty() || !ListsNodes(r, a.MeshEntity, a.Nodes)) return Fail(r, "A delete lists distinct nodes of the procedure");
                const auto plan = Plan(r, *procedure);
                const auto count = uint32_t(plan.Nodes.size());
                std::vector<bool> removed(count);
                for (uint32_t n = 0u; n < count; ++n) removed[n] = std::ranges::contains(a.Nodes, plan.Nodes[n].Entity);
                // Resolve links through removed nodes to surviving outputs.
                const auto through = [&](this const auto &self, PlanLink link) -> PlanLink {
                    if (link.FromNode == NoNode || !removed[link.FromNode]) return link;
                    const auto kind = plan.Nodes[link.FromNode].Kind;
                    const auto sockets = KindSockets(kind);
                    PlanLink source{};
                    uint32_t sources = 0u;
                    for (uint16_t i = 0u; i < InputCount[kind]; ++i) {
                        const auto read = sockets[i].Type == sockets[link.FromSocket].Type ? self(plan.Inputs[plan.InputFirst[link.FromNode] + i]) : PlanLink{};
                        if (read.FromNode == NoNode) continue;
                        source = read;
                        ++sources;
                    }
                    return sources == 1u ? source : PlanLink{};
                };
                // A surviving input reading a removed node reads through its chain, or nothing.
                for (uint32_t n = 0u; n < count; ++n) {
                    if (removed[n]) continue;
                    const auto kind = plan.Nodes[n].Kind;
                    std::optional<std::vector<SocketLink>> relinked;
                    for (uint16_t i = 0u; i < InputCount[kind]; ++i) {
                        const auto link = plan.Inputs[plan.InputFirst[n] + i];
                        if (link.FromNode == NoNode || !removed[link.FromNode]) continue;
                        if (!relinked) relinked = r.get<const NodeInputs>(plan.Nodes[n].Entity).Links;
                        const auto source = through(link);
                        const bool links = source.FromNode != NoNode && Links(KindSockets(plan.Nodes[source.FromNode].Kind)[source.FromSocket], KindSockets(kind)[i]);
                        (*relinked)[i] = links ? SocketLink{EntityOf(plan, source.FromNode), source.FromSocket} : SocketLink{};
                    }
                    if (relinked) r.replace<NodeInputs>(plan.Nodes[n].Entity, std::move(*relinked));
                }
                // The surviving nodes keep their authoring order.
                auto first = state::Null, last = state::Null;
                for (uint32_t n = 0u; n < count; ++n) {
                    if (removed[n]) continue;
                    const auto node = plan.Nodes[n].Entity;
                    if (last == state::Null) first = node;
                    else if (r.get<const ProcedureNode>(last).Next != node) r.patch<ProcedureNode>(last, [&](ProcedureNode &record) { record.Next = node; });
                    last = node;
                }
                if (last != state::Null && r.get<const ProcedureNode>(last).Next != state::Null) r.patch<ProcedureNode>(last, [](ProcedureNode &record) { record.Next = state::Null; });
                // The display reads the displayed node's result like a consumer and follows its chain to a surviving node.
                // A chain with several inputs, such as a Join's, leaves the display on the node its first input reads.
                auto display = procedure->DisplayNode;
                if (const auto displayed = plan.DisplayNode; displayed != NoNode && removed[displayed]) {
                    const auto kind = plan.Nodes[displayed].Kind;
                    const auto result = ResultSocket(kind);
                    auto shown = result ? through({displayed, *result}) : PlanLink{};
                    if (shown.FromNode == NoNode && InputCount[kind]) shown = through(plan.Inputs[plan.InputFirst[displayed]]);
                    display = EntityOf(plan, shown.FromNode);
                }
                // The displayed node, or else the last node, replaces a removed active node.
                auto active = procedure->ActiveNode;
                if (plan.ActiveNode == NoNode || removed[plan.ActiveNode]) active = display != state::Null ? display : last;
                if (const GeometryProcedure edited{first, last, display, active}; edited != *procedure) r.replace<GeometryProcedure>(a.MeshEntity, edited);
                r.destroy(a.Nodes);
            },
            [&](const InsertNode &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (!IsNodeOf(r, a.MeshEntity, a.Node) || !IsNodeOf(r, a.MeshEntity, a.ToNode) || a.ToSocket >= InputCount[kind_of(a.ToNode)]) return Fail(r, "An insert refers to a missing node or input");
                const auto plan = Plan(r, *procedure);
                const auto node = IndexOf(plan, a.Node), to_node = IndexOf(plan, a.ToNode);
                if (!Inserts(plan, node, to_node, a.ToSocket)) return Fail(r, "A node inserts on a link through geometry sockets that link with its ends, without closing a cycle");
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
                if (r.get<const NodeInputs>(a.Node).Links[input].FromNode == state::Null) set_input(a.Node, input, r.get<const NodeInputs>(a.ToNode).Links[a.ToSocket]);
                set_input(a.ToNode, a.ToSocket, {a.Node, *MainSocket(kind, SocketDirection::Out)});
            },
            [&](const SelectNodes &a) {
                const auto *procedure = procedure_of(a.MeshEntity);
                if (!procedure) return;
                if (!ListsNodes(r, a.MeshEntity, a.Nodes)) return Fail(r, "A node selection lists distinct nodes of the procedure");
                if (a.Active != state::Null && !IsNodeOf(r, a.MeshEntity, a.Active)) return Fail(r, "The active node must be a node of the procedure");
                DeselectNodes(r, a.MeshEntity, a.Nodes);
                for (const auto node : a.Nodes) {
                    if (!r.all_of<NodeSelected>(node)) r.emplace<NodeSelected>(node);
                }
                if (a.Active != state::Null && (procedure->ActiveNode != a.Active || (a.Display && procedure->DisplayNode != a.Active))) {
                    r.patch<GeometryProcedure>(a.MeshEntity, [&](GeometryProcedure &edited) {
                        edited.ActiveNode = a.Active;
                        if (a.Display) edited.DisplayNode = a.Active;
                    });
                }
            },
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
    // A copied Source node's state refers to the original's geometry, so a pending state is first kept as it stands.
    FinishSourceStates(r);
    const auto plan = Plan(r, r.get<const GeometryProcedure>(from));
    const auto tip = r.get<const EvaluatedMesh>(from).Tip;
    std::vector<state::Entity> copies;
    for (const auto &[entity, kind] : plan.Nodes) {
        const auto copy = CopyNode(r, entity, kind);
        if (r.all_of<NodeSelected>(entity)) r.emplace<NodeSelected>(copy);
        copies.push_back(copy);
    }
    const auto copy_of = [&](uint32_t node) { return node == NoNode ? state::Null : copies[node]; };
    const auto to = r.create();
    for (uint32_t n = 0u; n < copies.size(); ++n) {
        const auto kind = plan.Nodes[n].Kind;
        std::vector<SocketLink> links;
        for (uint32_t i = plan.InputFirst[n]; i < plan.InputFirst[n + 1u]; ++i) {
            const auto link = plan.Inputs[i];
            links.push_back(link.FromNode != NoNode ? SocketLink{copies[link.FromNode], link.FromSocket} : SocketLink{});
        }
        r.emplace<ProcedureNode>(copies[n], to, n + 1u < copies.size() ? copies[n + 1u] : state::Null, kind);
        r.emplace<NodeInputs>(copies[n], std::move(links));
        // The copy draws the copy of the drawn record, which holds the output of the same nodes.
        if (const auto *input = r.try_get<const EditedInput>(plan.Nodes[n].Entity)) r.emplace<EditedInput>(copies[n], copy_of(IndexOf(plan, input->Input)), input->State, input->StoreId);
    }
    r.emplace<GeometryProcedure>(to, copies.empty() ? state::Null : copies.front(), copies.empty() ? state::Null : copies.back(), copy_of(plan.DisplayNode), copy_of(plan.ActiveNode));
    r.emplace<EvaluatedMesh>(to, drawn, copy_of(IndexOf(plan, tip)));
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
    const auto display_kind = display != state::Null ? r.get<const ProcedureNode>(display).Kind : KindCount;
    const auto result = display != state::Null ? ResultSocket(display_kind) : std::optional<uint16_t>{};
    if (!result || !Links(KindSockets(display_kind)[*result], KindSockets(kind)[0])) {
        Fail(r, "The displayed node has no geometry result to append to");
        return state::Null;
    }
    const auto displayed = r.get<const NodeLayout>(display).Position;
    const auto plan = Plan(r, procedure);
    const auto moved = Downstream(plan, plan.DisplayNode);
    for (uint32_t n = 0u; n < moved.size(); ++n) {
        if (moved[n]) r.patch<NodeLayout>(plan.Nodes[n].Entity, [](NodeLayout &layout) { layout.Position.x += AppendGap; });
    }
    Apply(r, AddNode{mesh_entity, kind, {displayed.x + AppendGap, displayed.y}});
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
    const auto *procedure = r.try_get<const procedure::GeometryProcedure>(nodes::EditedMesh(r));
    return procedure ? procedure->ActiveNode : state::Null;
}
void UpdateTraits<procedure::NodeLayout>::ForEachSelected(state::Registry &r, const std::function<void(state::Entity)> &fn) {
    const auto mesh_entity = nodes::EditedMesh(r);
    if (!r.all_of<procedure::GeometryProcedure>(mesh_entity)) return;
    for (const auto node : nodes::SelectedNodes(r, mesh_entity)) fn(node);
}
} // namespace action
