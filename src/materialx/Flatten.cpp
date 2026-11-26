#include "materialx/Flatten.h"
#include "Variant.h"

#include <array>

namespace mx {
namespace {
// Appends the nodes that `outputs` read, inlining graph definitions, and returns the outputs' sources in `flat`.
// `interface` holds the sources that interface inputs refer to.
std::vector<Source> Inline(const Library &lib, std::span<const GraphNode> nodes, std::span<const Source> outputs, std::span<const Source> interface, FlatGraph &flat) {
    const auto resolve = [&](const Source &source, const auto &expand) {
        return std::visit(
            overloaded{
                [](const Value &value) { return Source{value}; },
                [&](InterfaceInput in) { return interface[in.Input]; },
                [&](NodeOutput out) { return expand(out.Node)[out.Output]; },
            },
            source
        );
    };
    std::vector<std::optional<std::vector<Source>>> node_outputs(nodes.size());
    const auto expand = [&](this const auto &self, uint32_t n) -> const std::vector<Source> & {
        if (node_outputs[n]) return *node_outputs[n];
        const auto &node = nodes[n];
        const auto &ports = lib.Declarations[node.Declaration].Inputs;
        std::vector<std::optional<Source>> set(ports.size());
        for (const auto &input : node.Inputs) set[input.Input] = resolve(input.Source, self);
        const auto unset = [&](const Port &port) {
            if (!port.DefaultGeomProp) return Source{port.Default};
            return Inline(lib, std::span{&lib.GeomProps[*port.DefaultGeomProp], 1}, std::array{Source{NodeOutput{0, 0}}}, {}, flat)[0];
        };
        std::vector<Source> inputs;
        for (size_t i = 0; i < ports.size(); ++i) inputs.push_back(set[i] ? std::move(*set[i]) : unset(ports[i]));
        if (const auto *graph = std::get_if<Graph>(&lib.Implementations[node.Declaration])) {
            node_outputs[n] = Inline(lib, graph->Nodes, graph->Outputs, inputs, flat);
        } else {
            const auto index = uint32_t(flat.Nodes.size());
            const auto output_count = uint32_t(lib.Declarations[node.Declaration].Outputs.size());
            flat.Nodes.push_back({node.Declaration, std::move(inputs)});
            node_outputs[n].emplace();
            for (uint32_t o = 0; o < output_count; ++o) node_outputs[n]->push_back(NodeOutput{index, o});
        }
        return *node_outputs[n];
    };
    std::vector<Source> result;
    for (const auto &output : outputs) result.push_back(resolve(output, expand));
    return result;
}
} // namespace

FlatGraph Flatten(const Library &lib, uint32_t declaration) {
    const auto &d = lib.Declarations[declaration];
    GraphNode root{.Declaration = declaration};
    std::vector<Source> interface;
    for (uint32_t i = 0; i < d.Inputs.size(); ++i) {
        root.Inputs.push_back({i, InterfaceInput{i}});
        interface.push_back(InterfaceInput{i});
    }
    std::vector<Source> outputs;
    for (uint32_t o = 0; o < d.Outputs.size(); ++o) outputs.push_back(NodeOutput{0, o});
    FlatGraph flat;
    flat.Outputs = Inline(lib, std::span{&root, 1}, outputs, interface, flat);
    return flat;
}
} // namespace mx
