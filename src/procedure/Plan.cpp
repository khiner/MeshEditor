#include "procedure/Procedure.h"
#include "render/Instance.h"
#include "scene/Entity.h"
#include "state/Registry.h"
#include <numeric>

namespace procedure {
std::vector<state::Entity> NodesOf(const state::Registry &r, const GeometryProcedure &procedure) {
    std::vector<state::Entity> nodes;
    for (auto node = procedure.First; node != state::Null; node = r.get<const ProcedureNode>(node).Next) nodes.push_back(node);
    return nodes;
}

// Check authored object reads before changing any target; a self read retains its existing missing-mesh behavior.
bool ObjectReferencesAcyclic(const state::Registry &r, std::span<const state::Entity> nodes, state::Entity object, state::Entity additional_reader, std::optional<std::pair<state::Entity, state::Entity>> binding) {
    const auto procedures = SortedEntities(r.view<const GeometryProcedure>());
    const auto index_of = [&](state::Entity entity) {
        const auto at = std::ranges::lower_bound(procedures, entity);
        return at != procedures.end() && *at == entity ? uint32_t(at - procedures.begin()) : NoNode;
    };
    const auto order = DependencyOrder(uint32_t(procedures.size()), [&](auto visit) {
        for (const auto [node, info] : r.view<const ObjectInfo>().each()) {
            const auto *record = r.try_get<const ProcedureNode>(node);
            const auto target = std::ranges::contains(nodes, node) ? object : info.Object;
            const auto *instance = r.try_get<const Instance>(target);
            const auto mesh = binding && binding->first == target ? binding->second : instance ? instance->Entity :
                                                                                                 state::Null;
            if (!record || mesh == state::Null || record->Procedure == mesh) continue;
            const auto source = index_of(mesh), reader = index_of(record->Procedure);
            if (source != NoNode && reader != NoNode) visit(source, reader);
        }
        if (const auto *instance = r.try_get<const Instance>(object); instance && additional_reader != instance->Entity) {
            const auto source = index_of(instance->Entity), reader = index_of(additional_reader);
            if (source != NoNode && reader != NoNode) visit(source, reader);
        }
    });
    return order.size() == procedures.size();
}

// Order nodes after their dependencies, preserving authoring order among ready nodes.
CookPlan Plan(const state::Registry &r, const GeometryProcedure &procedure, bool bypass, const SocketEdit *edit) {
    CookPlan plan{.InputFirst = {0u}, .SocketFirst = {0u}};
    for (auto node = procedure.First; node != state::Null;) {
        const auto &record = r.get<const ProcedureNode>(node);
        plan.Nodes.push_back({node, record.Kind});
        plan.InputFirst.push_back(plan.InputFirst.back() + InputCount[record.Kind]);
        plan.SocketFirst.push_back(plan.SocketFirst.back() + SocketCount[record.Kind]);
        node = record.Next;
    }
    const auto count = uint32_t(plan.Nodes.size());
    IndexNodes(plan);
    plan.Inputs.reserve(plan.InputFirst.back());
    for (const auto &node : plan.Nodes) {
        const auto &inputs = r.get<const NodeInputs>(node.Entity).Links;
        for (uint16_t i = 0; i < inputs.size(); ++i) {
            const auto link = edit && edit->Node == node.Entity && edit->Socket == i ? edit->Source : inputs[i];
            const auto resolved = ResolveSocket(r, link, bypass, edit);
            plan.Inputs.push_back({IndexOf(plan, resolved.Source.FromNode), resolved.Source.FromSocket, resolved.Constant});
            plan.Wires.push_back({IndexOf(plan, link.FromNode), link.FromSocket});
        }
    }
    plan.DisplayNode = IndexOf(plan, procedure.DisplayNode);
    plan.Order = DependencyOrder(count, [&](auto visit) {
        for (uint32_t n = 0; n < count; ++n)
            for (uint32_t i = plan.InputFirst[n]; i < plan.InputFirst[n + 1u]; ++i)
                if (const auto link = plan.Inputs[i]; link.FromNode != NoNode) visit(link.FromNode, n);
    });
    for (const auto &node : plan.Nodes) {
        if (const auto *interface = r.try_get<const SubnetInterface>(node.Entity)) {
            for (const auto &port : interface->Ports) {
                if (ResolveSocket(r, {node.Entity, port.Id}, bypass, edit).Cycle) {
                    plan.Order.clear();
                    break;
                }
            }
        }
    }
    plan.ResultNode = IndexOf(plan, ResolveSocket(r, DisplayLink(r, procedure), bypass).Source.FromNode);
    return plan;
}
std::optional<uint16_t> BypassInput(uint16_t kind) {
    const auto output = ResultSocket(kind);
    if (!output) return {};
    const auto sockets = KindSockets(kind);
    for (uint16_t i = 0; i < InputCount[kind]; ++i) {
        if (sockets[i].Type == sockets[*output].Type) return i;
    }
    return {};
}

void IndexNodes(CookPlan &plan) {
    plan.ByEntity.resize(plan.Nodes.size());
    std::iota(plan.ByEntity.begin(), plan.ByEntity.end(), 0u);
    std::ranges::sort(plan.ByEntity, {}, [&](uint32_t n) { return plan.Nodes[n].Entity; });
}

// The node's index in the plan, or NoNode.
uint32_t IndexOf(const CookPlan &plan, state::Entity node) {
    const auto at = std::ranges::lower_bound(plan.ByEntity, node, {}, [&](uint32_t n) { return plan.Nodes[n].Entity; });
    return at != plan.ByEntity.end() && plan.Nodes[*at].Entity == node ? *at : NoNode;
}

void BindDisplay(const state::Registry &r, const GeometryProcedure &procedure, CookPlan &plan) {
    plan.DisplayNode = IndexOf(plan, procedure.DisplayNode);
    plan.ResultNode = IndexOf(plan, ResolveSocket(r, DisplayLink(r, procedure), true).Source.FromNode);
    plan.Demanded = Upstream(plan, plan.ResultNode, [](uint32_t) { return true; });
    plan.Consumers.assign(plan.SocketFirst.back(), 0u);
    for (uint32_t node = 0; node < plan.Nodes.size(); ++node) {
        if (!plan.Demanded[node]) continue;
        for (uint32_t i = plan.InputFirst[node]; i < plan.InputFirst[node + 1]; ++i) {
            const auto link = plan.Inputs[i];
            if (link.FromNode != NoNode) ++plan.Consumers[plan.SocketFirst[link.FromNode] + link.FromSocket];
        }
    }
}

CookPlan Compile(const state::Registry &r, const GeometryProcedure &procedure) {
    auto plan = Plan(r, procedure, true);
    for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
        if (const auto *instances = r.try_get<const InstanceOnPoints>(plan.Nodes[n].Entity); instances && instances->Prototype != state::Null) plan.Inputs[plan.InputFirst[n] + 1u] = {};
        if (r.all_of<NodeBypassed>(plan.Nodes[n].Entity)) std::fill(plan.Inputs.begin() + plan.InputFirst[n], plan.Inputs.begin() + plan.InputFirst[n + 1], PlanLink{});
    }
    BindDisplay(r, procedure, plan);
    return plan;
}

state::Entity ActiveNodeOf(const state::Registry &r, state::Entity mesh_entity) {
    const auto *active = r.try_get<const NetworkActive>(mesh_entity);
    return active && active->Item.Kind == ItemKind::Entity ? active->Item.Entity : state::Null;
}

uint32_t ActiveIndex(const state::Registry &r, const CookPlan &plan) {
    if (plan.Nodes.empty()) return NoNode;
    const auto active = ActiveNodeOf(r, r.get<const ProcedureNode>(plan.Nodes.front().Entity).Procedure);
    return IndexOf(plan, active);
}
} // namespace procedure
