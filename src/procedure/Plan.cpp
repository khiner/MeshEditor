#include "procedure/Procedure.h"
#include "state/Registry.h"

namespace procedure {
// Order nodes after their dependencies, preserving authoring order among ready nodes.
CookPlan Plan(const state::Registry &r, const GeometryProcedure &procedure) {
    CookPlan plan{.InputFirst = {0u}, .SocketFirst = {0u}};
    for (auto node = procedure.First; node != state::Null;) {
        const auto &record = r.get<const ProcedureNode>(node);
        plan.Nodes.push_back({node, record.Kind});
        plan.InputFirst.push_back(plan.InputFirst.back() + InputCount[record.Kind]);
        plan.SocketFirst.push_back(plan.SocketFirst.back() + SocketCount[record.Kind]);
        node = record.Next;
    }
    const auto count = uint32_t(plan.Nodes.size());
    // Resolve linked nodes to their indices through the entities in ascending order.
    std::vector<std::pair<state::Entity, uint32_t>> indices;
    indices.reserve(count);
    for (uint32_t n = 0u; n < count; ++n) indices.emplace_back(plan.Nodes[n].Entity, n);
    std::ranges::sort(indices);
    const auto index_of = [&](state::Entity e) {
        const auto it = std::ranges::lower_bound(indices, e, {}, &std::pair<state::Entity, uint32_t>::first);
        return it != indices.end() && it->first == e ? it->second : NoNode;
    };
    plan.Inputs.reserve(plan.InputFirst.back());
    for (const auto &node : plan.Nodes) {
        for (const auto &link : r.get<const NodeInputs>(node.Entity).Links) plan.Inputs.push_back({index_of(link.FromNode), link.FromSocket});
    }
    plan.DisplayNode = index_of(procedure.DisplayNode);
    plan.ActiveNode = index_of(procedure.ActiveNode);
    plan.Consumers.assign(plan.SocketFirst.back(), 0u);
    // Track unread inputs per node and consumers as contiguous runs in one list.
    std::vector<uint32_t> pending(count, 0u), consumer_first(count + 1u, 0u);
    const auto for_each_link = [&](auto &&fn) {
        for (uint32_t n = 0u; n < count; ++n) {
            for (uint32_t i = plan.InputFirst[n]; i < plan.InputFirst[n + 1u]; ++i) {
                if (const auto link = plan.Inputs[i]; link.FromNode != NoNode) fn(n, link);
            }
        }
    };
    for_each_link([&](uint32_t n, PlanLink link) {
        ++plan.Consumers[plan.SocketFirst[link.FromNode] + link.FromSocket];
        ++pending[n];
        ++consumer_first[link.FromNode + 1u];
    });
    for (uint32_t n = 0u; n < count; ++n) consumer_first[n + 1u] += consumer_first[n];
    std::vector<uint32_t> consumers(consumer_first.back());
    auto fill = consumer_first;
    for_each_link([&](uint32_t n, PlanLink link) { consumers[fill[link.FromNode]++] = n; });
    for (uint32_t n = 0u; n < count; ++n) {
        if (!pending[n]) plan.Order.push_back(n);
    }
    for (size_t i = 0u; i < plan.Order.size(); ++i) {
        const auto n = plan.Order[i];
        for (uint32_t c = consumer_first[n]; c < consumer_first[n + 1u]; ++c) {
            if (!--pending[consumers[c]]) plan.Order.push_back(consumers[c]);
        }
    }
    return plan;
}
} // namespace procedure
