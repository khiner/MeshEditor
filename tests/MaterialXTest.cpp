#include "Paths.h"
#include "Variant.h"
#include "materialx/MetalSource.h"
#include "metal/AutoreleaseScope.h"
#include "metal/MetalContext.h"
#include "metal/MetalCpp.h"
#include <boost/ut.hpp>

using namespace boost::ut;

namespace {
std::string Join(const std::vector<std::string> &parts) {
    std::string joined;
    for (const auto &part : parts) joined += (joined.empty() ? "" : ", ") + part;
    return joined;
}

// A Metal function named for the declaration that evaluates its native implementation.
// Implementations read no input without a Metal value, so those inputs have no parameter.
std::string Function(const mx::Declaration &d, const auto &implementation) {
    std::vector<std::string> parameters, arguments, results;
    for (const auto &input : d.Inputs) {
        const auto type = mx::MetalType(input.Type);
        if (!type.empty()) parameters.push_back(std::format("{} a_{}", type, input.Name));
        arguments.push_back(type.empty() ? std::string{} : "a_" + input.Name);
    }
    std::string declarations;
    for (uint32_t o = 0; o < d.Outputs.size(); ++o) {
        results.push_back(std::format("r{}", o));
        declarations += std::format("    {} {};\n", mx::MetalType(d.Outputs[o].Type), results.back());
    }
    return std::format("{} {}({}) {{\n{}{}\n    return r0;\n}}\n", mx::MetalType(d.Outputs[0].Type), d.Name, Join(parameters), declarations, mx::Call(d, implementation, arguments, results));
}

std::string CompileError(const mtl::Context &ctx, const std::string &source) {
    const mtl::AutoreleaseScope pool;
    NS::Error *error = nullptr;
    const auto library = NS::TransferPtr(ctx.Device->newLibrary(mtl::Str(source).get(), static_cast<MTL::CompileOptions *>(nullptr), &error));
    return library ? std::string{} : std::string{error->localizedDescription()->utf8String()};
}
} // namespace

const boost::ut::suite MaterialX = [] {
    "every graph definition links outputs and inputs of matching types"_test = [] {
        const auto &lib = mx::Libraries();
        std::vector<std::string> failures;
        for (uint32_t i = 0; i < lib.Declarations.size(); ++i) {
            const auto *graph = std::get_if<mx::Graph>(&lib.Implementations[i]);
            if (!graph) continue;
            const auto &d = lib.Declarations[i];
            const auto matches = [&](const mx::Source &source, mx::Type type) {
                return std::visit(
                    overloaded{
                        [&](const mx::Value &value) { return std::holds_alternative<std::string>(value) == (type == mx::Type::String || type == mx::Type::Filename); },
                        [&](mx::NodeOutput out) { return lib.Declarations[graph->Nodes[out.Node].Declaration].Outputs[out.Output].Type == type; },
                        [&](mx::InterfaceInput in) { return d.Inputs[in.Input].Type == type; },
                    },
                    source
                );
            };
            for (const auto &node : graph->Nodes) {
                const auto &inputs = lib.Declarations[node.Declaration].Inputs;
                for (const auto &input : node.Inputs) {
                    if (!matches(input.Source, inputs[input.Input].Type)) failures.push_back(std::format("{} {}.{}", graph->Name, node.Name, inputs[input.Input].Name));
                }
            }
            for (size_t o = 0; o < graph->Outputs.size(); ++o) {
                if (!matches(graph->Outputs[o], d.Outputs[o].Type)) failures.push_back(std::format("{} output {}", graph->Name, d.Outputs[o].Name));
            }
        }
        expect(eq(Join(failures), std::string{}));
    };

    "every native implementation compiles as Metal with all others in one library"_test = [] {
        const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        const mtl::Context ctx;
        const auto &lib = mx::Libraries();
        std::string all;
        for (uint32_t i = 0; i < lib.Declarations.size(); ++i) {
            std::visit(
                overloaded{
                    [&](const mx::InlineExpression &e) { all += mx::MetalSource(lib, e.Definitions) + Function(lib.Declarations[i], e); },
                    [&](const mx::FunctionCall &call) { all += mx::MetalSource(lib, call.Definitions) + Function(lib.Declarations[i], call); },
                    [](const auto &) {},
                },
                lib.Implementations[i]
            );
        }
        expect(eq(CompileError(ctx, all), std::string{}));
    };
};

int main(int argc, char **argv) {
    if (argc > 1) cfg<override> = {.filter = argv[1]};
    Paths::Init(MESHEDITOR_BUILD_DIR, MESHEDITOR_BUILD_DIR);
    return cfg<>.run();
}
