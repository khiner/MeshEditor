#include "materialx/MetalSource.h"

#include <algorithm>
#include <format>

namespace mx {
std::string MetalSource(const Library &lib, std::span<const uint32_t> definitions, std::string_view declarations) {
    // A definition uses only earlier ones, so one pass from the last definition marks every definition reached.
    std::vector<bool> used(lib.Definitions.size());
    for (const auto d : definitions) used[d] = true;
    for (auto d = uint32_t(used.size()); d-- > 0u;) {
        if (used[d]) for (const auto use : lib.Definitions[d].Uses) used[use] = true;
    }
    std::string out{"#include <metal_stdlib>\nusing namespace metal;\n"};
    out += declarations;
    for (uint32_t d = 0; d < used.size(); ++d) {
        if (used[d]) out += std::format("#ifndef MX_DEFINITION_{0}\n#define MX_DEFINITION_{0}\n{1}#endif\n", d, lib.Definitions[d].Text);
    }
    return out;
}

std::string Call(const Declaration &d, const InlineExpression &implementation, std::span<const std::string> arguments, std::span<const std::string> results) {
    std::string expression;
    const std::string_view source{implementation.Source};
    size_t pos = 0;
    for (auto open = source.find("{{"); open != std::string_view::npos; open = source.find("{{", pos)) {
        const auto close = source.find("}}", open);
        const auto input = std::ranges::find(d.Inputs, source.substr(open + 2, close - open - 2), &Port::Name) - d.Inputs.begin();
        expression += std::format("{}({})", source.substr(pos, open - pos), arguments[input]);
        pos = close + 2;
    }
    expression += source.substr(pos);
    return std::format("{} = {};", results.front(), expression);
}

std::string Call(const Declaration &, const FunctionCall &implementation, std::span<const std::string> arguments, std::span<const std::string> results) {
    std::string call{implementation.Function + "("};
    for (const auto &argument : arguments) call += argument + ", ";
    for (const auto &result : results) call += result + ", ";
    call.resize(call.size() - 2);
    return call + ");";
}
} // namespace mx
