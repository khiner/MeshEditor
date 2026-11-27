#include "SortUnique.h"
#include "graph/OpNodes.h"
#include "render/ObjectGeometry.h"
#include "scene/Entity.h"
#include "state/Registry.h"
#include <numeric>

namespace graph {
// Check authored object reads before changing any target; a self read retains its existing missing-mesh behavior.
bool ObjectReferencesAcyclic(const state::Registry &r, std::span<const state::Entity> nodes, state::Entity object, state::Entity additional_reader, std::optional<std::pair<state::Entity, state::Entity>> binding) {
    const auto graphs = SortedEntities(r.view<const GeometryGraph>());
    const auto uses = SubgraphUses(r);
    const auto index_of = [&](state::Entity entity) {
        const auto at = std::ranges::lower_bound(graphs, entity);
        return at != graphs.end() && *at == entity ? uint32_t(at - graphs.begin()) : NoNode;
    };
    const auto order = DependencyOrder(uint32_t(graphs.size()), [&](auto visit) {
        for (const auto [node, info] : r.view<const ObjectInfo>().each()) {
            if (!r.all_of<GraphNode>(node)) continue;
            for (const auto reader : MeshesUsing(r, uses, GraphOf(r, node))) {
                const auto target = std::ranges::contains(nodes, node) ? object : info.Object;
                const auto *geometry = r.try_get<const ObjectGeometry>(target);
                const auto mesh = binding && binding->first == target ? binding->second : geometry ? geometry->Entity :
                                                                                                     state::Null;
                if (mesh == state::Null || reader == mesh) continue;
                const auto source = index_of(mesh), read = index_of(reader);
                if (source != NoNode && read != NoNode) visit(source, read);
            }
        }
        if (const auto *geometry = r.try_get<const ObjectGeometry>(object); geometry && additional_reader != geometry->Entity) {
            const auto source = index_of(geometry->Entity), reader = index_of(additional_reader);
            if (source != NoNode && reader != NoNode) visit(source, reader);
        }
    });
    return order.size() == graphs.size();
}

// Order nodes after their dependencies, preserving authoring order among ready nodes.
// An edit whose slot is one past a node's last slot adds a slot.
CookPlan Plan(const state::Registry &r, state::Entity graph, bool bypass, const SocketEdit *edit) {
    CookPlan plan{.InputFirst = {0u}, .SocketFirst = {0u}};
    const auto added = [&](const NodeRecord &node, size_t slots) { return !node.InBuiltinSubgraph && edit && edit->Node == node.Entity && edit->Socket == slots && node.Kind != KindOf<SubgraphNode>; };
    const auto expand = [&](this const auto &self, const state::Registry &at, state::Entity first, uint32_t parent, bool in_builtin_subgraph) -> void {
        for (auto node = first; node != state::Null; node = at.get<const GraphNode>(node).Next) {
            const auto kind = at.get<const GraphNode>(node).Kind;
            const auto n = uint32_t(plan.Nodes.size());
            plan.Nodes.push_back({node, kind, parent, in_builtin_subgraph});
            const bool use = kind == KindOf<SubgraphNode>;
            const auto slots = use ? 0u : at.get<const NodeInputs>(node).Links.size();
            plan.InputFirst.push_back(plan.InputFirst.back() + uint32_t(slots) + uint32_t(added(plan.Nodes.back(), slots)));
            plan.SocketFirst.push_back(plan.SocketFirst.back() + (use ? 0u : uint32_t(SocketsOf(at, node).size())));
            if (!use) continue;
            const auto subgraph = SubgraphOf(at, node);
            if (subgraph.Entity == state::Null) continue;
            if (subgraph.Registry == &r) plan.Subgraphs.push_back(subgraph.Entity);
            self(*subgraph.Registry, FirstNode(*subgraph.Registry, subgraph.Entity), n, in_builtin_subgraph || subgraph.Registry != &r);
        }
    };
    expand(r, FirstNode(r, graph), NoNode, false);
    SortUnique(plan.Subgraphs);
    const auto count = uint32_t(plan.Nodes.size());
    IndexNodes(plan);
    const auto index_of = [&](NodePath path, state::Entity node) {
        path.push_back(node);
        return IndexOf(plan, path);
    };
    // The subgraph nodes enclosing node `n`.
    const auto enclosing = [&](uint32_t n) {
        const auto path = PathOf(plan, n);
        return NodePath{path.begin(), path.end() - 1};
    };
    plan.Inputs.reserve(plan.InputFirst.back());
    for (uint32_t n = 0; n < count; ++n) {
        const auto &node = plan.Nodes[n];
        if (node.Kind == KindOf<SubgraphNode>) continue;
        const auto &inputs = Authored(r, node).get<const NodeInputs>(node.Entity).Links;
        for (uint16_t i = 0; i < inputs.size() + added(node, inputs.size()); ++i) {
            const auto link = !node.InBuiltinSubgraph && edit && edit->Node == node.Entity && edit->Socket == i ? edit->Source : inputs[i];
            const auto resolved = ResolveSocket(r, enclosing(n), link, bypass, edit);
            if (resolved.ValueUse != state::Null) plan.PortReads.push_back({uint32_t(plan.Inputs.size()), index_of(resolved.Path, resolved.ValueUse), resolved.ValuePort});
            plan.Inputs.push_back({
                .FromNode = resolved.Source.FromNode == state::Null ? NoNode : index_of(resolved.Path, resolved.Source.FromNode),
                .FromSocket = resolved.Source.FromSocket,
                .Constant = resolved.Constant,
                .SelectionUse = resolved.SelectionUse == state::Null ? NoNode : index_of(resolved.Path, resolved.SelectionUse),
                .SelectionPort = resolved.SelectionPort,
            });
        }
    }
    plan.Order = DependencyOrder(count, [&](auto visit) {
        for (uint32_t n = 0; n < count; ++n)
            for (uint32_t i = plan.InputFirst[n]; i < plan.InputFirst[n + 1u]; ++i)
                if (const auto link = plan.Inputs[i]; link.FromNode != NoNode) visit(link.FromNode, n);
    });
    // Only an op node's sockets differ from its kind's, so a plan without op nodes reads its kinds' sockets.
    if (std::ranges::any_of(plan.Nodes, [](const NodeRecord &node) { return node.Kind == KindOf<Op>; })) {
        plan.ViewSockets.resize(plan.SocketFirst.back());
        for (uint32_t n = 0; n < count; ++n) {
            if (plan.Nodes[n].Kind != KindOf<SubgraphNode>) std::ranges::copy(SocketsOf(Authored(r, plan.Nodes[n]), plan.Nodes[n].Entity), plan.ViewSockets.begin() + plan.SocketFirst[n]);
        }
    }
    for (const auto n : plan.Order) {
        if (plan.Nodes[n].Kind != KindOf<Op>) continue;
        const auto &op = Authored(r, plan.Nodes[n]).get<const Op>(plan.Nodes[n].Entity);
        const auto *definition = DefinitionOf(op.Node);
        if (!definition) continue;
        const bool varies = OpVaries(*definition, op, [&](uint16_t s) {
            const auto &link = plan.Inputs[plan.InputFirst[n] + s];
            if (link.FromNode != NoNode) return plan.ViewSockets[plan.SocketFirst[link.FromNode] + link.FromSocket].Varies ? OpInput::Varying : OpInput::Single;
            return link.Constant ? OpInput::Single : OpInput::Unset;
        });
        for (auto s = plan.SocketFirst[n] + definition->Inputs; s < plan.SocketFirst[n + 1u]; ++s) plan.ViewSockets[s].Varies = varies;
    }
    // A cycle through ports leaves no order.
    for (uint32_t n = 0; n < count && !plan.Order.empty(); ++n) {
        const auto &node = plan.Nodes[n];
        if (node.Kind != KindOf<SubgraphNode>) continue;
        const auto *interface = InterfaceOf(Authored(r, node), node.Entity);
        for (const auto &port : interface ? std::span<const SubgraphPort>{interface->Ports} : std::span<const SubgraphPort>{}) {
            if (ResolveSocket(r, enclosing(n), {node.Entity, port.Id}, bypass, edit).Cycle) {
                plan.Order.clear();
                break;
            }
        }
    }
    return plan;
}
std::optional<uint16_t> BypassInput(uint16_t kind) {
    const auto output = ResultSocket(kind);
    if (!output) return {};
    const auto sockets = KindSockets(kind);
    for (uint16_t i = 0; i < InputCount[kind]; ++i) {
        if (sockets[i].Types == sockets[*output].Types) return i;
    }
    return {};
}

// A node's parent precedes it.
void IndexNodes(CookPlan &plan) {
    plan.ByEntity.resize(plan.Nodes.size());
    std::iota(plan.ByEntity.begin(), plan.ByEntity.end(), 0u);
    std::ranges::sort(plan.ByEntity, {}, [&](uint32_t n) { return std::tuple{plan.Nodes[n].Parent, plan.Nodes[n].Entity, n}; });
    plan.Paths.clear();
    plan.PathFirst.assign(1u, 0u);
    for (const auto &node : plan.Nodes) {
        if (node.Parent != NoNode) {
            for (auto i = plan.PathFirst[node.Parent]; i < plan.PathFirst[node.Parent + 1]; ++i) plan.Paths.push_back(plan.Paths[i]);
        }
        plan.Paths.push_back(node.Entity);
        plan.PathFirst.push_back(uint32_t(plan.Paths.size()));
    }
}

uint32_t IndexOf(const CookPlan &plan, uint32_t parent, state::Entity node) {
    const std::pair key{parent, node};
    const auto at = std::ranges::lower_bound(plan.ByEntity, key, {}, [&](uint32_t n) { return std::pair{plan.Nodes[n].Parent, plan.Nodes[n].Entity}; });
    return at != plan.ByEntity.end() && plan.Nodes[*at].Parent == parent && plan.Nodes[*at].Entity == node ? *at : NoNode;
}
uint32_t IndexOf(const CookPlan &plan, std::span<const state::Entity> path) {
    auto node = NoNode;
    for (const auto entity : path) {
        node = IndexOf(plan, node, entity);
        if (node == NoNode) return NoNode;
    }
    return node;
}

void BindDisplay(const state::Registry &r, const GeometryGraph &graph, CookPlan &plan) {
    // The node's occurrence in the subgraph nodes of `enclosing`.
    const auto occurrence = [&](std::span<const state::Entity> enclosing, state::Entity node) {
        const auto parent = enclosing.empty() ? NoNode : IndexOf(plan, enclosing);
        return enclosing.empty() || parent != NoNode ? IndexOf(plan, parent, node) : NoNode;
    };
    plan.DisplayNode = occurrence(graph.DisplayPath, graph.DisplayNode);
    const auto result = ResolveSocket(r, graph.DisplayPath, DisplayLink(r, graph), true);
    plan.ResultNode = result.Source.FromNode == state::Null ? NoNode : occurrence(result.Path, result.Source.FromNode);
    Demand(plan, std::array{plan.ResultNode});
}
void Demand(CookPlan &plan, std::span<const uint32_t> nodes) {
    plan.Demanded.assign(plan.Nodes.size(), false);
    for (const auto node : nodes) {
        const auto upstream = Upstream(plan, node, [](uint32_t) { return true; });
        for (uint32_t n = 0; n < plan.Nodes.size(); ++n) plan.Demanded[n] = plan.Demanded[n] || upstream[n];
    }
    plan.Consumers.assign(plan.SocketFirst.back(), 0u);
    for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
        if (!plan.Demanded[n]) continue;
        for (uint32_t i = plan.InputFirst[n]; i < plan.InputFirst[n + 1]; ++i) {
            const auto link = plan.Inputs[i];
            if (link.FromNode != NoNode) ++plan.Consumers[plan.SocketFirst[link.FromNode] + link.FromSocket];
        }
    }
}

namespace {
SubgraphShape ShapeOf(const state::Registry &r, state::Entity use) {
    const auto &node = r.get<const GraphNode>(use);
    const auto &held = r.get<const SubgraphNode>(use);
    SubgraphShape shape{.Graph = node.Graph, .Next = node.Next, .SubgraphId = held.SubgraphId, .Bypassed = r.all_of<NodeBypassed>(use)};
    for (const auto &port : held.Ports) shape.Ports.emplace_back(port.PortId, port.Source, port.Value.has_value(), port.Selection.has_value());
    return shape;
}
} // namespace

CookPlan Compile(const state::Registry &r, state::Entity graph) {
    auto plan = Plan(r, graph, true);
    for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
        const auto &node = plan.Nodes[n];
        const auto &at = Authored(r, node);
        if (const auto *instances = at.try_get<const InstanceOnPoints>(node.Entity); instances && instances->Prototype != state::Null) plan.Inputs[plan.InputFirst[n] + 1u] = {};
        if (at.all_of<NodeBypassed>(node.Entity)) std::fill(plan.Inputs.begin() + plan.InputFirst[n], plan.Inputs.begin() + plan.InputFirst[n + 1], PlanLink{});
        if (node.Parent == NoNode && node.Kind == KindOf<SubgraphNode>) plan.SubgraphNodes.emplace_back(n, ShapeOf(r, node.Entity));
    }
    if (const auto *geometry = r.try_get<const GeometryGraph>(graph)) BindDisplay(r, *geometry, plan);
    return plan;
}

std::optional<CookPlan> Revalued(const state::Registry &r, const CookPlan &plan, std::span<const state::Entity> changed) {
    std::vector<uint32_t> uses;
    for (const auto entity : changed) {
        const auto held = std::ranges::find_if(plan.SubgraphNodes, [&](const auto &use) { return plan.Nodes[use.first].Entity == entity; });
        if (held == plan.SubgraphNodes.end() || !r.all_of<SubgraphNode>(entity) || held->second != ShapeOf(r, entity)) return {};
        uses.push_back(held->first);
    }
    auto revalued = plan;
    // An input that held no constant either reads a port holding no value or default, or belongs to a node whose inputs the compile cleared.
    for (const auto &read : plan.PortReads) {
        auto &constant = revalued.Inputs[read.Input].Constant;
        if (!constant || !std::ranges::contains(uses, read.Use)) continue;
        const auto use = plan.Nodes[read.Use].Entity;
        const auto *own = NodePortOf(r, use, read.Port);
        constant = own && own->Value ? own->Value : PortOf(r, use, read.Port)->Default;
    }
    return revalued;
}

state::Entity ActiveNodeOf(const state::Registry &r, state::Entity graph) {
    const auto *active = r.try_get<const GraphActive>(graph);
    return active ? active->Item : state::Null;
}

} // namespace graph
