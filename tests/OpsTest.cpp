#include "OpsFixture.h"

using namespace boost::ut;

namespace {
constexpr std::array AllDomains{ops::Domain::Point, ops::Domain::Vertex, ops::Domain::Edge, ops::Domain::Face, ops::Domain::Corner};
} // namespace

const boost::ut::suite Ops = [] {
    "mixed input types resolve to the declaration with the fewest conversions"_test = [] {
        const auto &lib = mx::Libraries();
        using enum mx::Type;
        struct Case {
            std::string_view Node;
            std::vector<mx::TypedInput> Inputs;
            std::string_view Declaration;
            std::vector<std::optional<std::string_view>> Conversions;
        };
        const std::array cases{
            Case{"add", {{"in1", Vector3}, {"in2", Integer}}, "ND_add_vector3", {{}, "ND_convert_integer_vector3"}},
            Case{"add", {{"in1", Color3}, {"in2", Vector3}}, "ND_add_color3", {{}, "ND_convert_vector3_color3"}},
            Case{"multiply", {{"in1", Boolean}, {"in2", Float}}, "ND_multiply_float", {"ND_convert_boolean_float", {}}},
            Case{"dotproduct", {{"in1", Vector3}, {"in2", Color3}}, "ND_dotproduct_vector3", {{}, "ND_convert_color3_vector3"}},
        };
        for (const auto &c : cases) {
            const auto resolved = ops::Resolve(lib, c.Node, c.Inputs);
            expect(resolved.has_value()) << c.Node;
            if (!resolved) continue;
            expect(lib.Declarations[resolved->Declaration].Name == c.Declaration) << c.Node;
            std::vector<std::optional<std::string_view>> conversions;
            for (const auto &conversion : resolved->Conversions) conversions.push_back(conversion ? std::optional{std::string_view{lib.Declarations[*conversion].Name}} : std::nullopt);
            expect(conversions == c.Conversions) << c.Node;
        }
    };

    "every native node evaluates on the domains it declares, a source reads the one mesh domain that stores it through a split, and the rest fault, as every domain in world space does"_test = [] {
        OpsFixture f;
        const auto &lib = mx::Libraries();
        mtl::ComputeChain chain{f.Buffers};
        for (uint32_t d = 0; d < lib.Declarations.size(); ++d) {
            const auto &declaration = lib.Declarations[d];
            // A native node whose unset inputs read geometric properties, an image node, evaluates where they do.
            if (!std::holds_alternative<mx::Native>(lib.Implementations[d]) || !HasValues(declaration) || std::ranges::any_of(declaration.Inputs, [](const mx::Port &port) { return port.DefaultGeomProp.has_value(); })) continue;
            const auto available = ops::Availability(lib, d), stored = ops::StoredDomains(lib, d);
            for (const auto domain : AllDomains) {
                ops::Graph graph{.Nodes = {{.Op = d}}};
                for (uint32_t o = 0; o < declaration.Outputs.size(); ++o) graph.Outputs.push_back({0u, o});
                const auto compiled = ops::Compile(lib, graph, domain);
                const bool split = !(available & ops::DomainBit(domain)) && domain != ops::Domain::Point && stored;
                expect(compiled.has_value() == (bool(available & ops::DomainBit(domain)) || (split && std::has_single_bit(stored)))) << declaration.Name;
                if (!compiled) {
                    expect(compiled.error().size() == 1u && compiled.error()[0].Fault == (split ? ops::Fault::Ambiguous : ops::Fault::Unavailable)) << declaration.Name;
                    continue;
                }
                // A split source reads its one attribute, which the record lacks, so it reads zero.
                expect(!split || (compiled->Attributes.size() == 1u && compiled->Attributes[0].Source)) << declaration.Name;
                const auto index = f.Index(*compiled);
                if (domain == ops::Domain::Point) ops::RecordPoint(chain, f.Drivers, {*compiled, index}, {}, {}, Frame, Time);
                else f.RecordElements(chain, domain, *compiled, index, {});
            }
            for (const auto *space : {"space", "tospace"}) {
                if (std::ranges::find(declaration.Inputs, space, &mx::Port::Name) == declaration.Inputs.end()) continue;
                const ops::Graph graph{.Nodes = {{.Op = d, .Inputs = {{space, ops::Literal{mx::Type::String, std::string{"world"}}}}}}, .Outputs = {{0u, 0u}}};
                for (const auto domain : AllDomains) expect(!ops::Compile(lib, graph, domain).has_value()) << declaration.Name;
            }
        }
        expect(nothrow([&] { chain.Submit(); }));
    };

    "a named attribute reads its stored values on the domain that holds it, and its default elsewhere"_test = [] {
        OpsFixture f;
        const auto &lib = mx::Libraries();
        mtl::ComputeChain chain{f.Buffers};
        std::mt19937 rng{5u};
        struct Read {
            std::vector<uint32_t> Outputs, Stored;
            ops::Domain Domain;
            std::vector<uint32_t> Default;
        };
        std::vector<Read> reads;
        for (const auto d : mx::NodeDeclarations(lib, "geompropvalue")) {
            const auto type = lib.Declarations[d].Outputs[0].Type;
            const auto storage = ops::StorageOf(type);
            if (!storage) continue;
            for (const auto held : MeshDomains) {
                const auto stored = f.AddAttribute(held, *storage);
                const auto name = std::string{f.Store().GetAttribute(stored).NameView()};
                for (uint32_t handle = 0; handle < f.Count(held); ++handle) test::SetAttributeWords(f.Store(), stored, handle, RandomWords(type, rng));
                const auto default_words = RandomWords(type, rng);
                const ops::Graph graph{.Nodes = {{.Op = d, .Inputs = {{"geomprop", ops::Literal{mx::Type::String, name}}, {"default", ops::Word{type, 0u}}}}}, .Outputs = {{0u, 0u}}};
                for (const auto domain : MeshDomains) {
                    const auto program = *ops::Compile(lib, graph, domain);
                    reads.push_back({f.RecordElements(chain, domain, program, f.Index(program), f.Block(default_words)), domain == held ? std::vector{stored} : std::vector<uint32_t>{}, domain, default_words});
                }
            }
        }
        chain.Submit();
        for (const auto &read : reads) {
            const auto outputs = f.ElementWords(read.Outputs, read.Domain);
            const auto stored = read.Stored.empty() ? std::vector(outputs.size(), read.Default) : f.ElementWords(read.Stored, read.Domain);
            expect(outputs == stored);
        }
    };
};
