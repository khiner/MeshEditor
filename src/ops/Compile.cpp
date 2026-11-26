#include "ops/Compile.h"
#include "Paths.h"
#include "Variant.h"
#include "gpu/OpPoint.h"
#include "materialx/Flatten.h"
#include "materialx/MetalSource.h"
#include "metal/MslSource.h"

#include <algorithm>
#include <bit>
#include <format>

namespace ops {
namespace {
// A native node that reads the point or a stored field.
// Stored holds the domains that store what it reads, and Held the domains whose points hold it: texels and surface points as the weighted values at their triangle's corners, and UV texels as their own texture coordinates.
// A node that reads only fields of the point has the OpNeeds bits of the fields and the code of its value at the point `{}`.
// A native node absent here evaluates on no domain.
struct NativeSource {
    std::string_view Node;
    Domains Stored, Held{};
    uint32_t Needs{};
    std::string_view Code{};
};
constexpr Domains WeightedDomains{DomainBit(Domain::Texel) | DomainBit(Domain::Surface)};
constexpr std::array<NativeSource, 18> NativeSources{{
    {"elementindex", MeshDomains, {}, OpNeedsIndex, "{}.Index"},
    {"position", DomainBit(Domain::Vertex), WeightedDomains, OpNeedsPosition, "{}.Position"},
    {"normal", DomainBit(Domain::Vertex) | DomainBit(Domain::Face) | DomainBit(Domain::Corner), WeightedDomains, OpNeedsNormal, "normalize({}.Normal)"},
    {"tangent", DomainBit(Domain::Corner), WeightedDomains, OpNeedsTangent, "normalize({}.Tangent)"},
    {"texcoord", DomainBit(Domain::Corner), WeightedDomains | DomainBit(Domain::Uv)},
    {"geomcolor", DomainBit(Domain::Vertex) | DomainBit(Domain::Corner), WeightedDomains}, // A record holds its colors on one of them
    {"geompropvalue", MeshDomains, WeightedDomains},
    {"frame", AllDomains, {}, 0u, "{}.Frame"},
    {"time", AllDomains, {}, 0u, "{}.Time"},
    {"transformpoint", AllDomains},
    {"transformvector", AllDomains},
    {"transformnormal", AllDomains},
    {"image", AllDomains},
    {"hextiledimage", AllDomains},
    {"hextilednormalmap", AllDomains},
    {"heighttonormal", AllDomains},
    {"splitlr", AllDomains},
    {"splittb", AllDomains},
}};
const NativeSource *SourceOf(std::string_view node) {
    const auto it = std::ranges::find(NativeSources, node, &NativeSource::Node);
    return it == NativeSources.end() ? nullptr : &*it;
}

// The text of a shader file under an include guard.
std::string Guarded(std::string_view file, std::string_view guard) { return std::format("#ifndef {0}\n#define {0}\n{1}#endif\n", guard, msl::Load(Paths::Shaders(), file).Text); }
// What every program reads, and what a program that samples images reads before the MaterialX functions it calls.
// Each loads on first use, so a process that compiles no program of its kind never reads its file.
const std::string &Prelude() {
    static const std::string prelude = Guarded("OpProgram.metal", "OP_PROGRAM");
    return prelude;
}
const std::string &ImagePrelude() {
    static const std::string prelude = Guarded("OpImage.metal", "OP_IMAGE");
    return prelude;
}

uint32_t Columns(mx::Type type) { return type == mx::Type::Matrix33 ? 3u : type == mx::Type::Matrix44 ? 4u : 1u; }

// A Metal value of the type built from the expressions of its words, where matrix words fill consecutive columns.
std::string ValueCode(mx::Type type, auto &&word) {
    const auto component = [&](uint32_t i) {
        if (type == mx::Type::Boolean) return std::format("({} != 0u)", word(i));
        if (type == mx::Type::Integer) return std::format("as_type<int>({})", word(i));
        return std::format("as_type<float>({})", word(i));
    };
    const auto words = mx::Words(type);
    if (words == 1u) return component(0u);
    const auto columns = Columns(type), rows = words / columns;
    std::string code{mx::MetalType(type)};
    code += '(';
    for (uint32_t c = 0u; c < columns; ++c) {
        if (columns > 1u) code += std::format("{}float{}(", c ? ", " : "", rows);
        for (uint32_t r = 0u; r < rows; ++r) code += (r ? ", " : "") + component(c * rows + r);
        if (columns > 1u) code += ')';
    }
    return code + ')';
}

// Statements writing the value's words to `out` from word `first`, in the order ValueCode reads them.
std::string WriteCode(mx::Type type, const std::string &value, uint32_t first) {
    const auto words = mx::Words(type);
    if (type == mx::Type::Boolean) return std::format("    out[{}] = uint({});\n", first, value);
    if (words == 1u) return std::format("    out[{}] = as_type<uint>({});\n", first, value);
    const auto columns = Columns(type), rows = words / columns;
    std::string code;
    for (uint32_t c = 0u; c < columns; ++c) {
        for (uint32_t r = 0u; r < rows; ++r) {
            code += std::format("    out[{}] = as_type<uint>({}{}[{}]);\n", first + c * rows + r, value, columns > 1u ? std::format("[{}]", c) : "", r);
        }
    }
    return code;
}

// An argument of an op: the Metal expression of a value, the text of a string, or the image table slot of an image, and the recipe that computes it.
struct Arg {
    std::string Code, Text;
    std::optional<uint32_t> Image{};
    Source Recipe;
};

Arg LiteralArg(mx::Type type, const mx::Value &value) {
    if (const auto *text = std::get_if<std::string>(&value)) return {.Text = *text, .Recipe = Literal{type, value}};
    const auto &words = std::get<std::vector<uint32_t>>(value);
    if (words.empty()) return {.Recipe = Literal{type, value}};
    return {.Code = ValueCode(type, [&](uint32_t i) { return std::format("0x{:08x}u", words[i]); }), .Recipe = Literal{type, value}};
}

// Each input's conversion to its port's type, or none when the declaration lacks an input or no conversion reaches its port.
std::optional<std::vector<std::optional<uint32_t>>> Conversions(const mx::Library &lib, uint32_t declaration, std::span<const mx::TypedInput> inputs) {
    const auto &ports = lib.Declarations[declaration].Inputs;
    std::vector<std::optional<uint32_t>> conversions;
    for (const auto &input : inputs) {
        const auto port = std::ranges::find(ports, input.Name, &mx::Port::Name);
        if (port == ports.end()) return {};
        const auto conversion = port->Type == input.Type ? std::nullopt : mx::FindDeclaration(lib, "convert", std::array{mx::TypedInput{"in", input.Type}}, port->Type);
        if (port->Type != input.Type && !conversion) return {};
        conversions.push_back(conversion);
    }
    return conversions;
}

// The program text a graph emits, and what it reads.
// The recipe holds each value the program computes as a declaration node of a graph that computes it alone, which a native source the domain lacks splits off with what it reads.
// On a domain with differentials, a filtered lookup or a derivative emits the recipe of what it differentiates again at the point's neighbors.
struct Emitter {
    const mx::Library &Lib;
    ops::Domain Domain, Colors;
    std::string Body{};
    std::vector<uint32_t> Definitions{}; // The library definitions the program uses
    std::vector<Attribute> Attributes{};
    uint32_t Needs{}, Temporaries{};
    std::optional<Fault> Failed{};
    Graph Recipe{};
    std::vector<uint32_t> RecipeNodes{}; // By recipe node, the graph node whose emission made it
    std::vector<std::vector<std::string>> RecipeCodes{}; // By recipe node, the code of each output at the point
    std::vector<bool> RecipeVaries{}; // By recipe node, whether it reads the point
    std::array<std::map<uint32_t, std::vector<std::string>>, 2> Neighbors{}; // By axis, the code of each output of each recipe node emitted at the neighbor along it
    std::string Point{"p"}; // The point the emission reads
    uint32_t Node{}; // The graph node being emitted
};
// Adds the recipe node and returns its first output.
Link AddRecipe(Emitter &e, Node node) {
    e.Recipe.Nodes.push_back(std::move(node));
    e.RecipeNodes.push_back(e.Node);
    e.RecipeCodes.emplace_back();
    e.RecipeVaries.push_back(false);
    return {uint32_t(e.Recipe.Nodes.size() - 1u), 0u};
}
// The recipe's subgraph that recipe node `root` reads, with `root` as its output.
Split SplitOf(const Emitter &e, uint32_t root, Domain domain) {
    std::vector<bool> kept(root + 1u);
    kept[root] = true;
    for (uint32_t n = root + 1u; n-- > 0u;) {
        if (!kept[n]) continue;
        for (const auto &input : e.Recipe.Nodes[n].Inputs)
            if (const auto *link = std::get_if<Link>(&input.Source)) kept[link->Node] = true;
    }
    Split split{.Graph = {.Colors = e.Colors}, .Domain = domain};
    std::vector<uint32_t> index(root + 1u);
    for (uint32_t n = 0u; n <= root; ++n) {
        if (!kept[n]) continue;
        index[n] = uint32_t(split.Graph.Nodes.size());
        auto node = e.Recipe.Nodes[n];
        for (auto &input : node.Inputs)
            if (auto *link = std::get_if<Link>(&input.Source)) link->Node = index[link->Node];
        split.Graph.Nodes.push_back(std::move(node));
        split.Nodes.push_back(e.RecipeNodes[n]);
    }
    split.Graph.Outputs.push_back({index[root], 0u});
    return split;
}

// The value of a 4-component sample with the type's components.
std::string_view Channels(mx::Type type) {
    const auto words = mx::Words(type);
    return words >= 1u && words <= 3u ? std::string_view{".xyz"}.substr(0u, words + 1u) : std::string_view{};
}
// Declares a temporary for each of the declaration's outputs, and returns their names.
std::vector<std::string> DeclareResults(Emitter &e, const mx::Declaration &d) {
    std::vector<std::string> results;
    for (const auto &output : d.Outputs) {
        results.push_back(std::format("t{}", e.Temporaries++));
        e.Body += std::format("    {} {};\n", mx::MetalType(output.Type), results.back());
    }
    return results;
}

void EmitDeclaration(Emitter &, uint32_t declaration, uint32_t recipe, std::span<const Arg> in, std::span<const std::string> results);
// The code of each output of recipe node `n` at the point's neighbor along `axis`, emitted again there when it reads the point.
std::vector<std::string> NeighborCodes(Emitter &e, uint32_t n, uint32_t axis) {
    if (!e.RecipeVaries[n]) return e.RecipeCodes[n];
    if (const auto found = e.Neighbors[axis].find(n); found != e.Neighbors[axis].end()) return found->second;
    const auto declaration = std::get<uint32_t>(e.Recipe.Nodes[n].Op);
    const auto &d = e.Lib.Declarations[declaration];
    std::vector<Arg> in;
    for (const auto &input : e.Recipe.Nodes[n].Inputs) {
        in.push_back(std::visit(
            overloaded{
                [&](const Literal &literal) { return LiteralArg(literal.Type, literal.Value); },
                [](Word) -> Arg { std::unreachable(); }, // A word reaches the recipe through its dot node, which reads no point
                [&](Link link) { return Arg{.Code = NeighborCodes(e, link.Node, axis)[link.Output], .Recipe = link}; },
                [](ops::Image image) { return Arg{.Image = image.Slot, .Recipe = image}; },
            },
            input.Source
        ));
    }
    auto results = DeclareResults(e, d);
    const auto point = std::exchange(e.Point, std::format("p.Neighbors[{}]", axis));
    EmitDeclaration(e, declaration, n, in, results);
    e.Point = point;
    return e.Neighbors[axis].emplace(n, std::move(results)).first->second;
}
// The codes of the argument's values at the point's neighbors along x and y.
// On a domain with differentials, a value that reads the point is emitted again at each neighbor, and at a neighbor it is its value there plus its difference toward each neighbor at the point.
// On any other domain both are the argument's own code, so its differences are zero.
std::array<std::string, 2> NeighborArgs(Emitter &e, const Arg &arg) {
    const auto *link = std::get_if<Link>(&arg.Recipe);
    if (!Differentiable(e.Domain) || !link || !e.RecipeVaries[link->Node]) return {arg.Code, arg.Code};
    e.Needs |= OpNeedsNeighbors;
    std::array at{NeighborCodes(e, link->Node, 0u)[link->Output], NeighborCodes(e, link->Node, 1u)[link->Output]};
    if (e.Point == "p") return at;
    const auto &point = e.RecipeCodes[link->Node][link->Output];
    for (auto &code : at) code = std::format("({} + ({} - {}))", arg.Code, code, point);
    return at;
}

// Declares the sampler of an image argument with the address modes and filter MaterialX names, where a constant address mode clamps.
// Returns the name of its OpImageSampler, which holds no image without one.
std::string ImageSampler(Emitter &e, const Arg &file, std::string_view u, std::string_view v, std::string_view filter) {
    e.Needs |= OpNeedsImages;
    const auto t = e.Temporaries++;
    const auto address = [](std::string_view mode) { return mode == "periodic" ? "repeat" : mode == "mirror" ? "mirrored_repeat" : "clamp_to_edge"; };
    const auto sampling = filter == "closest" ? "nearest" : "linear";
    e.Body += std::format("    constexpr sampler smp{}(s_address::{}, t_address::{}, filter::{}, mip_filter::{});\n", t, address(u), address(v), sampling, sampling);
    const auto image = file.Image ? std::format("&{}.Images[{}u]", e.Point, *file.Image) : std::string{"nullptr"};
    const auto present = file.Image ? std::format("OpImagePresent({}.Images, {}u)", e.Point, *file.Image) : std::string{"false"};
    e.Body += std::format("    const OpImageSampler img{}{{{}, smp{}, {}}};\n", t, image, t, present);
    return std::format("img{}", t);
}
void AddFunction(Emitter &e, uint32_t declaration) { e.Definitions.append_range(std::get<mx::Native>(e.Lib.Implementations[declaration]).Definitions); }
// The code of the index of a surface point's texture coordinate set that the argument reads directly, or none for an argument other than a texture coordinate node's two-component output.
std::optional<std::string> SurfaceTexcoordSet(const Emitter &e, const Arg &arg) {
    const auto *link = std::get_if<Link>(&arg.Recipe);
    if (e.Domain != Domain::Surface || e.Point != "p" || !link) return {};
    const auto &node = e.Recipe.Nodes[link->Node];
    const auto *declaration = std::get_if<uint32_t>(&node.Op);
    if (!declaration || e.Lib.Declarations[*declaration].Node != "texcoord" || e.Lib.Declarations[*declaration].Outputs[0].Type != mx::Type::Vector2) return {};
    const auto &index = node.Inputs.front().Source;
    if (const auto *from = std::get_if<Link>(&index)) return e.RecipeCodes[from->Node][from->Output];
    const auto &literal = std::get<Literal>(index);
    return LiteralArg(literal.Type, literal.Value).Code;
}
// Emits an image node's lookup into `result`: the default without an image, and else the sample at its coordinates, filtered over their differences toward the point's neighbors.
// Coordinates have V up, and the image's first row is its top.
// A lookup at a surface point's own texture coordinates samples them with V down and filters over their derivatives, as a material's texture does.
void EmitImage(Emitter &e, uint32_t declaration, auto &&input, const std::string &result) {
    AddFunction(e, declaration);
    const auto &d = e.Lib.Declarations[declaration];
    const auto text = [&](std::string_view name) { return std::string_view{input(name)->Text}; };
    const auto image = ImageSampler(e, *input("file"), text("uaddressmode"), text("vaddressmode"), text("filtertype"));
    const auto &texcoord = *input("texcoord");
    const auto transform = [](const std::string &uv) { return std::format("mx_transform_uv({}, float2(1.0), float2(0.0))", uv); };
    const auto uv = std::format("uv{}", e.Temporaries++);
    const auto set = SurfaceTexcoordSet(e, texcoord);
    e.Body += std::format("    const float2 {} = {};\n", uv, set ? std::format("OpSurfaceTexcoord(p, {}, 0u)", *set) : transform(texcoord.Code));
    auto lookup = std::format("texture({}, {})", image, uv);
    if (set) {
        lookup = std::format("textureGrad({0}, {1}, OpSurfaceTexcoord(p, {2}, 1u), OpSurfaceTexcoord(p, {2}, 2u))", image, uv, *set);
    } else if (Differentiable(e.Domain)) {
        const auto neighbors = NeighborArgs(e, texcoord);
        e.Body += std::format("    const float2 {0}x = {1}, {0}y = {2};\n", uv, transform(neighbors[0]), transform(neighbors[1]));
        lookup = std::format("textureGrad({0}, {1}, {1}x - {1}, {1}y - {1})", image, uv);
    }
    auto present = std::format("{}.Present", image);
    if (text("uaddressmode") == "constant") present += std::format(" && {0}.x >= 0.0f && {0}.x <= 1.0f", uv);
    if (text("vaddressmode") == "constant") present += std::format(" && {0}.y >= 0.0f && {0}.y <= 1.0f", uv);
    e.Body += std::format("    {} = {} ? {}{} : {};\n", result, present, lookup, Channels(d.Outputs[0].Type), input("default")->Code);
}
// Emits a native node's MaterialX function into `result`, called on each value input followed by its values at the point's neighbors, and on each image input's sampler.
// A node with an image gives its default without one.
void EmitFunction(Emitter &e, uint32_t declaration, std::span<const Arg> in, auto &&input, const std::string &result) {
    AddFunction(e, declaration);
    const auto &d = e.Lib.Declarations[declaration];
    auto call = std::format("{}(", std::get<mx::Native>(e.Lib.Implementations[declaration]).Function);
    std::string present;
    for (size_t i = 0; i < d.Inputs.size(); ++i) {
        if (d.Inputs[i].Type == mx::Type::Filename) {
            const auto image = ImageSampler(e, in[i], "periodic", "periodic", "linear");
            call += image + ", ";
            present = image + ".Present";
        } else if (mx::Words(d.Inputs[i].Type)) {
            const auto [x, y] = NeighborArgs(e, in[i]);
            call += std::format("{}, {}, {}, ", in[i].Code, x, y);
        }
    }
    call += result + ")";
    e.Body += present.empty() ? std::format("    {};\n", call) : std::format("    if ({}) {};\n    else {} = {};\n", present, call, result, input("default")->Code);
}

// The statement assigning a native declaration's output, whose recipe node is `recipe`, which fails when no domain stores what it reads.
// A source the domain lacks reads its split, evaluated on the domain that stores it, through an attribute.
// No domain has world space, since a mesh has no single world transform.
void EmitNative(Emitter &e, uint32_t declaration, uint32_t recipe, std::span<const Arg> in, const std::string &result) {
    const auto &d = e.Lib.Declarations[declaration];
    const auto input = [&](std::string_view name) -> const Arg * {
        const auto port = std::ranges::find(d.Inputs, name, &mx::Port::Name);
        return port == d.Inputs.end() ? nullptr : &in[port - d.Inputs.begin()];
    };
    const auto world = [&](std::string_view name) {
        const auto *arg = input(name);
        return arg && arg->Text == "world";
    };
    if (world("space") || world("fromspace") != world("tospace")) {
        e.Failed = Fault::Unavailable;
        return;
    }
    const auto type = d.Outputs[0].Type;
    const auto &p = e.Point;
    // A texel reads an integer or boolean field at the corner of its largest weight.
    const auto read = e.Domain == Domain::Texel && (type == mx::Type::Integer || type == mx::Type::Boolean) ? "OpLargestAttribute" : "OpAttribute";
    if (!(Availability(e.Lib, declaration, e.Colors) & DomainBit(e.Domain))) {
        const auto stored = StoredDomains(e.Lib, declaration, e.Colors);
        if (!ReadsFields(e.Domain) || !std::has_single_bit(stored)) {
            e.Failed = ReadsFields(e.Domain) && stored ? Fault::Ambiguous : Fault::Unavailable;
            return;
        }
        auto split = SplitOf(e, recipe, Domain(std::countr_zero(stored)));
        auto it = std::ranges::find(e.Attributes, std::optional{split}, &Attribute::Source);
        if (it == e.Attributes.end()) it = e.Attributes.insert(it, {std::format(".split.{}", e.Attributes.size()), type, std::move(split)});
        e.Body += std::format("    {} = {}({}, {}u, {});\n", result, read, p, uint32_t(it - e.Attributes.begin()), ValueCode(type, [](uint32_t) { return std::string{"0u"}; }));
        e.RecipeVaries[recipe] = true;
        return;
    }
    const std::string_view node{d.Node};
    if (node == "image") return EmitImage(e, declaration, input, result);
    if (!std::get<mx::Native>(e.Lib.Implementations[declaration]).Function.empty()) return EmitFunction(e, declaration, in, input, result);
    const auto *source = SourceOf(node);
    const auto value = [&]() -> std::string {
        if (!source->Code.empty()) {
            e.Needs |= source->Needs;
            return std::vformat(source->Code, std::make_format_args(p));
        }
        if (node == "texcoord") {
            e.Needs |= OpNeedsTexcoords;
            const auto uv = std::format("OpTexcoord({}, {})", p, input("index")->Code);
            return type == mx::Type::Vector2 ? uv : std::format("float3({}, 0.0f)", uv);
        }
        if (node == "geomcolor") {
            e.Needs |= OpNeedsColor;
            return std::format("OpColor({}, {}){}", p, input("index")->Code, Channels(type));
        }
        if (node == "geompropvalue") {
            const Attribute attribute{input("geomprop")->Text, type};
            const auto it = std::ranges::find_if(e.Attributes, [&](const Attribute &a) { return a.Name == attribute.Name && a.Type == attribute.Type; });
            const auto slot = uint32_t(it - e.Attributes.begin());
            if (it == e.Attributes.end()) e.Attributes.push_back(attribute);
            return std::format("{}({}, {}u, {})", read, p, slot, input("default")->Code);
        }
        if (node == "transformnormal") return std::format("normalize({})", input("in")->Code);
        return input("in")->Code; // transformpoint and transformvector
    }();
    e.RecipeVaries[recipe] = e.RecipeVaries[recipe] || node == "elementindex" || source->Held;
    e.Body += std::format("    {} = {};\n", result, value);
}

// Emits the statements assigning a declaration's outputs to `results` from its arguments, whose recipe node is `recipe`.
// The recipe node reads the point when the native node does or an input it reads does.
void EmitDeclaration(Emitter &e, uint32_t declaration, uint32_t recipe, std::span<const Arg> in, std::span<const std::string> results) {
    const auto &d = e.Lib.Declarations[declaration];
    for (const auto &arg : in) {
        if (const auto *link = std::get_if<Link>(&arg.Recipe); link && e.RecipeVaries[link->Node]) e.RecipeVaries[recipe] = true;
    }
    std::vector<std::string> codes;
    for (const auto &arg : in) codes.push_back(arg.Code);
    std::visit(
        overloaded{
            [&](const mx::Native &) { EmitNative(e, declaration, recipe, in, results.front()); },
            [](const mx::Graph &) { std::unreachable(); }, // Flattening inlines every graph
            [&](const auto &call) { // An inline expression or a function call
                e.Definitions.append_range(call.Definitions);
                e.Body += std::format("    {}\n", mx::Call(d, call, codes, results));
            },
        },
        e.Lib.Implementations[declaration]
    );
}

// Emits the declaration's flattened definition on arguments in its input order, and returns its outputs.
std::vector<Arg> Instantiate(Emitter &e, uint32_t declaration, std::span<const Arg> args) {
    const auto flat = mx::Flatten(e.Lib, declaration);
    std::vector<std::vector<Arg>> outputs(flat.Nodes.size());
    const auto arg = [&](const mx::Source &source, mx::Type type) {
        return std::visit(
            overloaded{
                [&](const mx::Value &value) { return LiteralArg(type, value); },
                [&](mx::InterfaceInput in) { return args[in.Input]; },
                [&](mx::NodeOutput out) { return outputs[out.Node][out.Output]; },
            },
            source
        );
    };
    for (uint32_t n = 0; n < flat.Nodes.size(); ++n) {
        const auto &node = flat.Nodes[n];
        const auto &d = e.Lib.Declarations[node.Declaration];
        std::vector<Arg> in;
        Node recipe{.Op = node.Declaration};
        for (size_t i = 0; i < d.Inputs.size(); ++i) {
            in.push_back(arg(node.Inputs[i], d.Inputs[i].Type));
            recipe.Inputs.push_back({d.Inputs[i].Name, in.back().Recipe});
        }
        const auto recipe_node = AddRecipe(e, std::move(recipe)).Node;
        auto results = DeclareResults(e, d);
        EmitDeclaration(e, node.Declaration, recipe_node, in, results);
        if (e.Failed) return {};
        e.RecipeCodes[recipe_node] = results;
        for (uint32_t o = 0; o < results.size(); ++o) outputs[n].push_back({.Code = std::move(results[o]), .Recipe = Link{recipe_node, o}});
    }
    std::vector<Arg> result;
    for (size_t o = 0; o < flat.Outputs.size(); ++o) result.push_back(arg(flat.Outputs[o], e.Lib.Declarations[declaration].Outputs[o].Type));
    return result;
}

// The value of the geometric property an unset input reads.
Arg GeomProp(Emitter &e, const mx::GraphNode &property) {
    const auto &ports = e.Lib.Declarations[property.Declaration].Inputs;
    std::vector<Arg> args;
    for (const auto &port : ports) args.push_back(LiteralArg(port.Type, port.Default));
    for (const auto &input : property.Inputs) args[input.Input] = LiteralArg(ports[input.Input].Type, std::get<mx::Value>(input.Source));
    auto outputs = Instantiate(e, property.Declaration, args);
    return e.Failed ? Arg{} : std::move(outputs.front());
}
} // namespace

std::optional<AttributeType> StorageOf(mx::Type type) {
    if (type == mx::Type::Boolean) return AttributeType::Bool;
    if (type == mx::Type::Integer) return AttributeType::Int;
    // A float value of one to four components.
    constexpr std::array FloatTypes{AttributeType::Float, AttributeType::Vec2, AttributeType::Vec3, AttributeType::Vec4};
    const auto words = mx::Words(type);
    return words >= 1u && words <= 4u ? std::optional{FloatTypes[words - 1u]} : std::nullopt;
}

std::optional<Resolution> Resolve(const mx::Library &lib, std::string_view node, std::span<const mx::TypedInput> inputs, std::optional<mx::Type> output) {
    std::optional<Resolution> best;
    uint32_t best_count{};
    for (const auto declaration : mx::NodeDeclarations(lib, node)) {
        if (!mx::OutputsMatch(lib.Declarations[declaration], output)) continue;
        auto conversions = Conversions(lib, declaration, inputs);
        if (!conversions) continue;
        const auto count = uint32_t(std::ranges::count_if(*conversions, [](const auto &c) { return c.has_value(); }));
        if (count == 0u && inputs.size() == lib.Declarations[declaration].Inputs.size()) return Resolution{declaration, std::move(*conversions)};
        if (best && count >= best_count) continue;
        best = Resolution{declaration, std::move(*conversions)};
        best_count = count;
    }
    return best;
}

Domains StoredDomains(const mx::Library &lib, uint32_t declaration, Domain colors) {
    const auto *source = SourceOf(lib.Declarations[declaration].Node);
    if (!source) return {};
    return source->Node == "geomcolor" ? DomainBit(colors) : source->Stored;
}

Domains Availability(const mx::Library &lib, uint32_t declaration, Domain colors) {
    if (!std::holds_alternative<mx::Native>(lib.Implementations[declaration])) return AllDomains;
    const auto stored = StoredDomains(lib, declaration, colors);
    const auto through = std::popcount(uint32_t(stored & (DomainBit(Domain::Vertex) | DomainBit(Domain::Face)))) == 1;
    const auto *source = SourceOf(lib.Declarations[declaration].Node);
    return Domains((through ? stored | DomainBit(Domain::Corner) : stored) | (source ? source->Held : Domains{}));
}

DomainReduction DefaultReduction(AttributeType type, Domain from, Domain to) {
    if (type != AttributeType::Bool) return DomainReduction::Mean;
    const bool bounding = (from == Domain::Face && (to == Domain::Vertex || to == Domain::Edge)) || (from == Domain::Edge && to == Domain::Vertex);
    return bounding ? DomainReduction::Any : DomainReduction::All;
}

std::expected<Program, std::vector<NodeFault>> Compile(const mx::Library &lib, const Graph &graph, Domain domain) {
    Emitter e{lib, domain, graph.Colors};
    Program program{};
    std::vector<NodeFault> faults;
    // Each node's outputs, and none for a node that faulted or reads one that did.
    std::vector<std::optional<std::vector<Arg>>> node_outputs(graph.Nodes.size());
    std::vector<std::vector<mx::Type>> output_types(graph.Nodes.size());
    uint32_t reads = 0; // The word inputs read so far, which number their names
    const auto read_word = [&](Word word) {
        const auto name = std::format("s{}", reads++);
        e.Body += std::format("    const {} {} = {};\n", mx::MetalType(word.Type), name, ValueCode(word.Type, [&](uint32_t i) { return std::format("v[{}]", word.Offset + i); }));
        // The split of a source reads the word as a dot node of it.
        const auto recipe = AddRecipe(e, {.Op = std::string{"dot"}, .Inputs = {{"in", word}}});
        e.RecipeCodes[recipe.Node] = {name};
        return Arg{.Code = name, .Recipe = recipe};
    };
    for (uint32_t n = 0; n < graph.Nodes.size(); ++n) {
        const auto &node = graph.Nodes[n];
        e.Node = n;
        std::vector<mx::TypedInput> typed;
        bool reads_fault = false;
        for (const auto &input : node.Inputs) {
            const auto type = std::visit(
                overloaded{
                    [](const Literal &literal) { return literal.Type; },
                    [](Word word) { return word.Type; },
                    [&](Link link) {
                        reads_fault |= !node_outputs[link.Node];
                        // An output the source's resolved declaration lacks has no type, which leaves the reading node unresolved.
                        return !reads_fault && link.Output < output_types[link.Node].size() ? output_types[link.Node][link.Output] : mx::Type::MultiOutput;
                    },
                    [](Image) { return mx::Type::Filename; },
                },
                input.Source
            );
            typed.push_back({input.Name, type});
        }
        if (reads_fault) continue;
        const auto resolution = std::visit(
            overloaded{
                [&](const std::string &name) { return Resolve(lib, name, typed, node.Output); },
                [&](uint32_t declaration) {
                    auto conversions = Conversions(lib, declaration, typed);
                    return conversions ? std::optional{Resolution{declaration, std::move(*conversions)}} : std::nullopt;
                },
            },
            node.Op
        );
        if (!resolution) {
            faults.push_back({n, Fault::Unresolved});
            continue;
        }
        const auto &d = lib.Declarations[resolution->Declaration];
        if (std::ranges::any_of(d.Outputs, [](const mx::Port &port) { return mx::MetalType(port.Type).empty(); })) {
            faults.push_back({n, Fault::NoValue});
            continue;
        }
        // The given input at each declaration input, so the program reads its words in declaration input order.
        std::vector<std::optional<size_t>> given(d.Inputs.size());
        for (size_t k = 0; k < node.Inputs.size(); ++k) given[std::ranges::find(d.Inputs, node.Inputs[k].Name, &mx::Port::Name) - d.Inputs.begin()] = k;
        std::vector<Arg> args;
        for (uint32_t i = 0; i < d.Inputs.size(); ++i) {
            const auto &port = d.Inputs[i];
            if (given[i]) {
                const auto arg = std::visit(
                    overloaded{
                        [](const Literal &literal) { return LiteralArg(literal.Type, literal.Value); },
                        read_word,
                        [&](Link link) { return (*node_outputs[link.Node])[link.Output]; },
                        [](Image image) { return Arg{.Image = image.Slot, .Recipe = image}; },
                    },
                    node.Inputs[*given[i]].Source
                );
                const auto conversion = resolution->Conversions[*given[i]];
                args.push_back(conversion ? Instantiate(e, *conversion, std::array{arg}).front() : arg);
            } else if (port.DefaultGeomProp) {
                args.push_back(GeomProp(e, lib.GeomProps[*port.DefaultGeomProp]));
            } else {
                args.push_back(LiteralArg(port.Type, port.Default));
            }
        }
        auto outputs = e.Failed ? std::vector<Arg>{} : Instantiate(e, resolution->Declaration, args);
        if (e.Failed) {
            faults.push_back({n, *std::exchange(e.Failed, {})});
            continue;
        }
        node_outputs[n] = std::move(outputs);
        for (const auto &output : d.Outputs) output_types[n].push_back(output.Type);
    }
    uint32_t output_words = 0;
    std::string writes;
    for (const auto [node, output] : graph.Outputs) {
        if (!node_outputs[node]) continue;
        // An output the node's resolved declaration lacks leaves the node unresolved.
        if (output >= output_types[node].size()) {
            faults.push_back({node, Fault::Unresolved});
            continue;
        }
        const auto type = output_types[node][output];
        const auto output_word_count = mx::Words(type);
        if (domain != Domain::Point && !StorageOf(type)) {
            faults.push_back({node, Fault::Unstored});
        } else if (program.Outputs.size() == OpMaxOutputs || output_words + output_word_count > OpMaxOutputWords) {
            faults.push_back({node, Fault::OutputLimit});
        } else {
            const auto name = std::format("o{}", program.Outputs.size());
            writes += std::format("    const {} {} = {};\n{}", mx::MetalType(type), name, (*node_outputs[node])[output].Code, WriteCode(type, name, output_words));
            program.Outputs.push_back(type);
            output_words += output_word_count;
        }
    }
    if (!faults.empty()) return std::unexpected{std::move(faults)};
    program.Attributes = std::move(e.Attributes);
    program.Needs = e.Needs;
    const auto *images = e.Needs & OpNeedsImages ? &ImagePrelude() : nullptr;
    program.Source = std::format("{}{}[[visible]] void ProgramFunction(const thread OpPoint &p, device const uint *v, thread uint *out) {{\n{}{}}}\n", mx::MetalSource(lib, e.Definitions, images ? std::string_view{*images} : std::string_view{}), Prelude(), e.Body, writes);
    return program;
}

} // namespace ops
