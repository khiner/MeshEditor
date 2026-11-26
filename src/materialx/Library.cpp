#include "materialx/Library.h"
#include "Paths.h"
#include "materialx/Xml.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <fstream>
#include <ranges>

namespace mx {
namespace {
namespace fs = std::filesystem;

struct TypeInfo {
    std::string_view Name;
    uint32_t Words; // Zero for a type with no Metal value
    std::string_view MetalType; // Empty for a type with no Metal value
};
constexpr std::array<TypeInfo, 18> Types{{
    {"boolean", 1, "bool"},
    {"integer", 1, "int"},
    {"float", 1, "float"},
    {"vector2", 2, "float2"},
    {"vector3", 3, "float3"},
    {"vector4", 4, "float4"},
    {"color3", 3, "float3"},
    {"color4", 4, "float4"},
    {"matrix33", 9, "float3x3"},
    {"matrix44", 16, "float4x4"},
    {"string", 0, ""},
    {"filename", 0, ""},
    {"surfaceshader", 0, ""},
    {"displacementshader", 0, ""},
    {"volumeshader", 0, ""},
    {"lightshader", 0, ""},
    {"material", 0, ""},
    {"multioutput", 0, ""},
}};

// Nodes that read the evaluation point, a texture, screen-space derivatives, or a whole stored field.
// MeshEditor implements them in place of any graph definition MaterialX gives.
// Declarations with no graph definition whose implementation reads or produces a string, filename, shader or material are MeshEditor's as well.
constexpr std::array<std::string_view, 30> NativeNodes{
    "adjacent",
    "blur",
    "boundary",
    "convertdomain",
    "curvature",
    "edgeloop",
    "edgering",
    "elementindex",
    "linked",
    "selection",
    "position",
    "normal",
    "tangent",
    "bitangent",
    "texcoord",
    "geomcolor",
    "geompropvalue",
    "geompropvalueuniform",
    "frame",
    "time",
    "viewdirection",
    "transformpoint",
    "transformvector",
    "transformnormal",
    "image",
    "hextiledimage",
    "hextilednormalmap",
    "heighttonormal",
    "splitlr",
    "splittb",
};

constexpr std::array<std::pair<std::string_view, std::string_view>, 15> MetalTypeNames{{
    {"vec2", "float2"},
    {"vec3", "float3"},
    {"vec4", "float4"},
    {"ivec2", "int2"},
    {"ivec3", "int3"},
    {"ivec4", "int4"},
    {"uvec2", "uint2"},
    {"uvec3", "uint3"},
    {"uvec4", "uint4"},
    {"bvec2", "bool2"},
    {"bvec3", "bool3"},
    {"bvec4", "bool4"},
    {"mat2", "float2x2"},
    {"mat3", "float3x3"},
    {"mat4", "float4x4"},
}};

std::unexpected<std::string> Fail(std::string message) { return std::unexpected{std::move(message)}; }

std::string_view Trim(std::string_view s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

std::vector<std::string_view> SplitList(std::string_view text) {
    std::vector<std::string_view> tokens;
    if (Trim(text).empty()) return tokens;
    for (size_t start = 0;;) {
        const auto end = text.find(',', start);
        tokens.push_back(Trim(text.substr(start, end - start)));
        if (end == std::string_view::npos) return tokens;
        start = end + 1;
    }
}

std::optional<Type> ParseType(std::string_view name) {
    const auto it = std::ranges::find(Types, name, &TypeInfo::Name);
    if (it == Types.end()) return {};
    return Type(it - Types.begin());
}

std::expected<std::vector<uint32_t>, std::string> ParseWords(Type type, std::span<const std::string_view> tokens) {
    std::vector<uint32_t> words;
    for (const auto token : tokens) {
        const auto *const end = token.data() + token.size();
        if (type == Type::Boolean) {
            if (token != "true" && token != "false") return Fail(std::format("'{}' is not a boolean", token));
            words.push_back(token == "true" ? 1u : 0u);
        } else if (type == Type::Integer) {
            int32_t value;
            if (std::from_chars(token.data(), end, value).ptr != end) return Fail(std::format("'{}' is not an integer", token));
            words.push_back(std::bit_cast<uint32_t>(value));
        } else {
            float value;
            if (std::from_chars(token.data(), end, value).ptr != end) return Fail(std::format("'{}' is not a float", token));
            words.push_back(std::bit_cast<uint32_t>(value));
        }
    }
    return words;
}

// An empty text is the type's zero value.
std::expected<Value, std::string> ParseValue(Type type, std::string_view text) {
    if (type == Type::String || type == Type::Filename) return Value{std::string{text}};
    const auto tokens = SplitList(text);
    if (tokens.empty()) return Value{std::vector<uint32_t>(Words(type), 0u)};
    if (tokens.size() != Words(type)) return Fail(std::format("'{}' is not a {} value", text, Name(type)));
    return ParseWords(type, tokens).transform([](std::vector<uint32_t> words) { return Value{std::move(words)}; });
}

// A comma-separated list of values, each taking the type's component count.
std::expected<std::vector<Value>, std::string> ParseValues(Type type, std::string_view text) {
    const auto tokens = SplitList(text);
    std::vector<Value> values;
    if (tokens.empty()) return values;
    if (type == Type::String || type == Type::Filename) {
        for (const auto token : tokens) values.emplace_back(std::string{token});
        return values;
    }
    const auto words = Words(type);
    if (words == 0 || tokens.size() % words != 0) return Fail(std::format("'{}' is not a list of {} values", text, Name(type)));
    for (size_t i = 0; i < tokens.size(); i += words) {
        auto value = ParseWords(type, std::span{tokens}.subspan(i, words));
        if (!value) return std::unexpected{std::move(value.error())};
        values.emplace_back(std::move(*value));
    }
    return values;
}

std::expected<std::optional<Value>, std::string> ParseOptionalValue(Type type, std::string_view text) {
    if (text.empty()) return std::optional<Value>{};
    return ParseValue(type, text).transform([](Value value) { return std::optional{std::move(value)}; });
}

// A value's text in a document.
std::string FormatValue(Type type, const Value &value) {
    if (const auto *text = std::get_if<std::string>(&value)) return *text;
    const auto &words = std::get<std::vector<uint32_t>>(value);
    std::string formatted;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i) formatted += ", ";
        if (type == Type::Boolean) formatted += words[i] ? "true" : "false";
        else if (type == Type::Integer) formatted += std::format("{}", std::bit_cast<int32_t>(words[i]));
        else formatted += std::format("{}", std::bit_cast<float>(words[i]));
    }
    return formatted;
}

std::expected<std::string, std::string> ReadFile(const fs::path &path) {
    std::ifstream in{path, std::ios::binary};
    if (!in) return Fail(std::format("cannot read {}", path.string()));
    return std::string{std::istreambuf_iterator<char>{in}, {}};
}

std::optional<uint32_t> PortIndex(const std::vector<Port> &ports, std::string_view name) {
    const auto it = std::ranges::find(ports, name, &Port::Name);
    if (it == ports.end()) return {};
    return uint32_t(it - ports.begin());
}

bool IsIdentifierStart(char c) { return std::isalpha(uint8_t(c)) || c == '_'; }
bool IsIdentifier(char c) { return std::isalnum(uint8_t(c)) || c == '_'; }

struct Document {
    fs::path Path;
    xml::Element Root;
};

std::expected<std::vector<Document>, std::string> ReadDocuments(const fs::path &dir) {
    std::vector<fs::path> paths;
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator{dir, ec}) {
        if (entry.is_regular_file() && entry.path().extension() == ".mtlx") paths.push_back(entry.path());
    }
    std::ranges::sort(paths);
    std::vector<Document> documents;
    for (auto &path : paths) {
        const auto text = ReadFile(path);
        if (!text) return std::unexpected{text.error()};
        auto root = xml::Parse(*text);
        if (!root) return Fail(std::format("{}: {}", path.string(), root.error()));
        documents.push_back({std::move(path), std::move(*root)});
    }
    return documents;
}

std::expected<Port, std::string> ParsePort(const xml::Element &e, bool input, const std::map<std::string, uint32_t, std::less<>> &geomprops) {
    const auto type = ParseType(xml::Find(e, "type"));
    if (!type) return Fail(std::format("port '{}' has unknown type '{}'", xml::Find(e, "name"), xml::Find(e, "type")));
    auto default_value = ParseValue(*type, xml::Find(e, input ? "value" : "default"));
    auto enum_values = ParseValues(*type, xml::Find(e, "enumvalues"));
    auto ui_min = ParseOptionalValue(*type, xml::Find(e, "uimin"));
    auto ui_max = ParseOptionalValue(*type, xml::Find(e, "uimax"));
    auto ui_soft_min = ParseOptionalValue(*type, xml::Find(e, "uisoftmin"));
    auto ui_soft_max = ParseOptionalValue(*type, xml::Find(e, "uisoftmax"));
    for (const auto &error : {default_value.error_or({}), enum_values.error_or({}), ui_min.error_or({}), ui_max.error_or({}), ui_soft_min.error_or({}), ui_soft_max.error_or({})}) {
        if (!error.empty()) return Fail(std::format("port '{}': {}", xml::Find(e, "name"), error));
    }
    const auto geomprop_name = xml::Find(e, "defaultgeomprop");
    const auto geomprop = geomprops.find(geomprop_name);
    if (!geomprop_name.empty() && geomprop == geomprops.end()) return Fail(std::format("port '{}' reads unknown geometric property '{}'", xml::Find(e, "name"), geomprop_name));
    std::vector<std::string> enum_names;
    for (const auto name : SplitList(xml::Find(e, "enum"))) enum_names.emplace_back(name);
    return Port{
        .Name = std::string{xml::Find(e, "name")},
        .Type = *type,
        .Default = std::move(*default_value),
        .Uniform = xml::Find(e, "uniform") == "true",
        .EnumNames = std::move(enum_names),
        .EnumValues = std::move(*enum_values),
        .DefaultGeomProp = geomprop_name.empty() ? std::nullopt : std::optional{geomprop->second},
        .UiName = std::string{xml::Find(e, "uiname")},
        .Doc = std::string{xml::Find(e, "doc")},
        .UiMin = std::move(*ui_min),
        .UiMax = std::move(*ui_max),
        .UiSoftMin = std::move(*ui_soft_min),
        .UiSoftMax = std::move(*ui_soft_max),
    };
}

std::expected<Declaration, std::string> ParseDeclaration(const xml::Element &e, const std::map<std::string, uint32_t, std::less<>> &geomprops) {
    Declaration d{
        .Name = std::string{xml::Find(e, "name")},
        .Node = std::string{xml::Find(e, "node")},
        .NodeGroup = std::string{xml::Find(e, "nodegroup")},
        .Doc = std::string{xml::Find(e, "doc")},
    };
    for (const auto &child : e.Children) {
        if (child.Name != "input" && child.Name != "output") continue;
        const bool input = child.Name == "input";
        auto port = ParsePort(child, input, geomprops);
        if (!port) return Fail(std::format("{}: {}", d.Name, port.error()));
        (input ? d.Inputs : d.Outputs).push_back(std::move(*port));
    }
    if (d.Outputs.empty()) return Fail(std::format("{} has no output", d.Name));
    return d;
}

// The graph of the element defining declaration `d`, whose node elements name their declarations or give their output and input types.
std::expected<Graph, std::string> ParseGraph(const Library &lib, const xml::Element &e, const Declaration &d) {
    Graph graph{.Name = std::string{xml::Find(e, "name")}};
    std::map<std::string_view, uint32_t, std::less<>> node_indices;
    std::vector<const xml::Element *> node_elements;
    for (const auto &child : e.Children) {
        if (child.Name == "output") continue;
        const auto explicit_declaration = xml::Find(child, "nodedef");
        const auto declaration = [&]() -> std::expected<uint32_t, std::string> {
            if (!explicit_declaration.empty()) {
                const auto it = lib.DeclarationsByName.find(explicit_declaration);
                if (it == lib.DeclarationsByName.end()) return Fail(std::format("node '{}' names unknown declaration {}", xml::Find(child, "name"), explicit_declaration));
                return it->second;
            }
            const auto output = ParseType(xml::Find(child, "type"));
            if (!output) return Fail(std::format("node '{}' has unknown type '{}'", xml::Find(child, "name"), xml::Find(child, "type")));
            std::vector<TypedInput> inputs;
            for (const auto &input : child.Children) {
                const auto type = ParseType(xml::Find(input, "type"));
                if (!type) return Fail(std::format("input '{}' of node '{}' has unknown type '{}'", xml::Find(input, "name"), xml::Find(child, "name"), xml::Find(input, "type")));
                inputs.push_back({xml::Find(input, "name"), *type});
            }
            const auto found = FindDeclaration(lib, child.Name, inputs, *output);
            if (!found) return Fail(std::format("node '{}' matches no declaration of <{}>", xml::Find(child, "name"), child.Name));
            return *found;
        }();
        if (!declaration) return std::unexpected{declaration.error()};
        if (!node_indices.emplace(xml::Find(child, "name"), uint32_t(graph.Nodes.size())).second) return Fail(std::format("node name '{}' repeats", xml::Find(child, "name")));
        graph.Nodes.push_back({.Name = std::string{xml::Find(child, "name")}, .Declaration = *declaration});
        node_elements.push_back(&child);
    }
    const auto parse_source = [&](const xml::Element &input, Type type) -> std::expected<Source, std::string> {
        if (const auto interface = xml::Find(input, "interfacename"); !interface.empty()) {
            const auto index = PortIndex(d.Inputs, interface);
            if (!index) return Fail(std::format("'{}' links unknown interface input '{}'", xml::Find(input, "name"), interface));
            return InterfaceInput{*index};
        }
        if (const auto node_name = xml::Find(input, "nodename"); !node_name.empty()) {
            const auto node = node_indices.find(node_name);
            if (node == node_indices.end()) return Fail(std::format("'{}' links unknown node '{}'", xml::Find(input, "name"), node_name));
            const auto &outputs = lib.Declarations[graph.Nodes[node->second].Declaration].Outputs;
            const auto output_name = xml::Find(input, "output");
            if (output_name.empty() && outputs.size() != 1) return Fail(std::format("'{}' links node '{}' with several outputs and names none", xml::Find(input, "name"), node_name));
            const auto output = output_name.empty() ? std::optional{0u} : PortIndex(outputs, output_name);
            if (!output) return Fail(std::format("'{}' links unknown output '{}' of node '{}'", xml::Find(input, "name"), output_name, node_name));
            return NodeOutput{node->second, *output};
        }
        auto value = ParseValue(type, xml::Find(input, "value"));
        if (!value) return Fail(std::format("'{}': {}", xml::Find(input, "name"), value.error()));
        return std::move(*value);
    };
    for (size_t n = 0; n < graph.Nodes.size(); ++n) {
        auto &node = graph.Nodes[n];
        const auto &inputs = lib.Declarations[node.Declaration].Inputs;
        for (const auto &input : node_elements[n]->Children) {
            const auto index = PortIndex(inputs, xml::Find(input, "name"));
            if (!index) return Fail(std::format("node '{}' sets unknown input '{}'", node.Name, xml::Find(input, "name")));
            auto source = parse_source(input, inputs[*index].Type);
            if (!source) return Fail(std::format("node '{}' input {}", node.Name, source.error()));
            node.Inputs.push_back({*index, std::move(*source)});
        }
    }
    std::vector<std::optional<Source>> outputs(d.Outputs.size());
    for (const auto &child : e.Children) {
        if (child.Name != "output") continue;
        const auto index = PortIndex(d.Outputs, xml::Find(child, "name"));
        if (!index) return Fail(std::format("unknown output '{}'", xml::Find(child, "name")));
        auto source = parse_source(child, d.Outputs[*index].Type);
        if (!source) return Fail(std::format("output {}", source.error()));
        outputs[*index] = std::move(*source);
    }
    for (size_t o = 0; o < outputs.size(); ++o) {
        if (!outputs[o]) return Fail(std::format("output '{}' is undefined", d.Outputs[o].Name));
        graph.Outputs.push_back(std::move(*outputs[o]));
    }
    return graph;
}

// The rewrites of the function files of MeshEditor's native texture nodes, for the calls Native describes.
constexpr std::array<std::pair<std::string_view, std::string_view>, 3> NativeRewrites{{
    {"$fileTransformUv", "mx_transform_uv_vflip.glsl"},
    {"$texSamplerSignature", "OpImageSampler tex_sampler"},
    {"$texSamplerSampler2D", "tex_sampler"},
}};
std::string Rewritten(std::string text) {
    for (const auto &[from, to] : NativeRewrites) {
        for (auto at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) text.replace(at, from.size(), to);
    }
    return text;
}

// The GLSL types of the parameters a function that differentiates takes at the point's neighbors.
constexpr std::array<std::string_view, 9> ValueTypeNames{"float", "vec2", "vec3", "vec4", "int", "bool", "mat2", "mat3", "mat4"};
// The suffixes of a parameter's names at the point's neighbors along x and y.
constexpr std::array<std::string_view, 2> NeighborSuffixes{"_nbx", "_nby"};

// The end of the comment that starts at `i`, or `i` where none starts.
size_t SkipComment(std::string_view text, size_t i) {
    if (text.substr(i).starts_with("//")) return std::min(text.find('\n', i), text.size());
    if (text.substr(i).starts_with("/*")) return std::min(text.find("*/", i + 2), text.size() - 2) + 2;
    return i;
}
size_t SkipSpace(std::string_view text, size_t i) {
    while (i < text.size()) {
        if (const auto end = SkipComment(text, i); end != i) i = end;
        else if (std::isspace(uint8_t(text[i]))) ++i;
        else break;
    }
    return i;
}
// The position after the bracket that closes the one at `open`, past comments.
size_t Matching(std::string_view text, size_t open) {
    int depth = 0;
    for (size_t i = open; i < text.size();) {
        if (const auto end = SkipComment(text, i); end != i) {
            i = end;
            continue;
        }
        const auto c = text[i++];
        if (c == '(' || c == '{' || c == '[') ++depth;
        else if ((c == ')' || c == '}' || c == ']') && --depth == 0) return i;
    }
    return text.size();
}
// The text's comma-separated parts outside brackets.
std::vector<std::string_view> SplitArguments(std::string_view text) {
    std::vector<std::string_view> parts;
    size_t start = 0;
    for (size_t i = 0; i < text.size();) {
        if (text[i] == '(' || text[i] == '[') {
            i = Matching(text, i);
        } else if (text[i] == ',') {
            parts.push_back(Trim(text.substr(start, i - start)));
            start = ++i;
        } else {
            ++i;
        }
    }
    if (!Trim(text.substr(start)).empty()) parts.push_back(Trim(text.substr(start)));
    return parts;
}
// Whether the identifier at `i` starts there.
bool StartsIdentifier(std::string_view text, size_t i) { return IsIdentifierStart(text[i]) && (i == 0 || (!IsIdentifier(text[i - 1]) && text[i - 1] != '.')); }
size_t IdentifierEnd(std::string_view text, size_t i) {
    while (i < text.size() && IsIdentifier(text[i])) ++i;
    return i;
}
// The start of the first identifier at or after `i` outside comments, or the text's end.
size_t NextIdentifier(std::string_view text, size_t i) {
    while (i < text.size() && !StartsIdentifier(text, i)) i = std::max(SkipComment(text, i), i + 1);
    return i;
}

// Rewrites GLSL vector and matrix type names to Metal's, and `out T x` and `inout T x` parameters to `thread T &x`.
// Comments stay as they are.
std::string PortToMetal(std::string_view glsl) {
    const auto metal_name = [](std::string_view name) {
        const auto it = std::ranges::find(MetalTypeNames, name, &std::pair<std::string_view, std::string_view>::first);
        return it == MetalTypeNames.end() ? name : it->second;
    };
    std::string out;
    out.reserve(glsl.size() + glsl.size() / 8);
    for (size_t i = 0; i < glsl.size();) {
        const auto start = NextIdentifier(glsl, i);
        out += glsl.substr(i, start - i);
        if (start == glsl.size()) break;
        i = IdentifierEnd(glsl, start);
        const auto word = glsl.substr(start, i - start);
        if (word != "out" && word != "inout") {
            out += metal_name(word);
            continue;
        }
        const auto type = SkipSpace(glsl, i), type_end = IdentifierEnd(glsl, type), name = SkipSpace(glsl, type_end);
        i = IdentifierEnd(glsl, name);
        out += std::format("thread {} &{}", metal_name(glsl.substr(type, type_end - type)), glsl.substr(name, i - name));
    }
    return out;
}

// The value a local variable of the function body takes where it is declared with one and never assigned again.
std::optional<std::string_view> SingleValue(std::string_view body, std::string_view name) {
    std::optional<std::string_view> value;
    uint32_t assignments = 0;
    for (auto at = NextIdentifier(body, 0); at < body.size(); at = NextIdentifier(body, at)) {
        const auto start = at;
        at = IdentifierEnd(body, at);
        if (body.substr(start, at - start) != name) continue;
        auto next = SkipSpace(body, at);
        while (next < body.size() && (body[next] == '.' || body[next] == '[')) next = SkipSpace(body, body[next] == '[' ? Matching(body, next) : IdentifierEnd(body, next + 1));
        const auto rest = body.substr(next);
        if (!(rest.starts_with("=") && !rest.starts_with("==")) && !rest.starts_with("+=") && !rest.starts_with("-=") && !rest.starts_with("*=") && !rest.starts_with("/=") && !rest.starts_with("++") && !rest.starts_with("--")) continue;
        ++assignments;
        const auto type_end = body.find_last_not_of(" \t\r\n", start - 1) + 1;
        const auto type_start = body.find_last_of(" \t\r\n;{}(", type_end - 1) + 1;
        if (next == SkipSpace(body, at) && rest.starts_with("=") && std::ranges::contains(ValueTypeNames, body.substr(type_start, type_end - type_start))) value = Trim(body.substr(next + 1, body.find(';', next) - next - 1));
    }
    return assignments == 1 ? value : std::nullopt;
}
// The expression at the point's neighbor along `axis`: each value parameter's neighbor, and each local the body assigns once at its value there.
std::expected<std::string, std::string> AtNeighbor(std::string_view expression, uint32_t axis, std::span<const std::string_view> parameters, std::string_view body) {
    std::string out;
    for (size_t i = 0; i < expression.size();) {
        const auto start = NextIdentifier(expression, i);
        out += expression.substr(i, start - i);
        if (start == expression.size()) break;
        i = IdentifierEnd(expression, start);
        const auto name = expression.substr(start, i - start);
        const auto after = SkipSpace(expression, i);
        if ((after < expression.size() && expression[after] == '(') || name == "true" || name == "false") {
            out += name;
        } else if (std::ranges::contains(parameters, name)) {
            out += std::format("{}{}", name, NeighborSuffixes[axis]);
        } else if (const auto value = SingleValue(body, name)) {
            auto at = AtNeighbor(*value, axis, parameters, body);
            if (!at) return at;
            out += std::format("({})", *at);
        } else {
            return Fail(std::format("a derivative reads '{}', which is neither a value parameter nor a local assigned once", name));
        }
    }
    return out;
}
// The file with each function that differentiates taking its value parameters' values at the point's two neighbors after them, and `differentiates` listing it.
// A function differentiates when it takes dFdx or dFdy of an expression of its parameters, or calls a function that differentiates.
// dFdx(e) becomes e at the neighbor along x less e, and dFdy(e) the same along y.
// A call to a function that differentiates passes each of its value arguments at the neighbors after it.
using Differentiates = std::map<std::string, std::vector<bool>, std::less<>>; // By function, whether each parameter takes neighbors
std::expected<std::string, std::string> Differentiated(std::string_view text, Differentiates &differentiates) {
    std::string out;
    size_t copied = 0;
    for (auto i = NextIdentifier(text, 0); i < text.size(); i = NextIdentifier(text, i)) {
        const auto name_start = i;
        i = IdentifierEnd(text, i);
        const auto name = text.substr(name_start, i - name_start);
        const auto open = SkipSpace(text, i);
        if (open >= text.size() || text[open] != '(') continue;
        const auto close = Matching(text, open);
        const auto body_open = SkipSpace(text, close);
        i = close;
        if (body_open >= text.size() || text[body_open] != '{') continue;
        const auto body_close = Matching(text, body_open);
        const auto body = text.substr(body_open, body_close - body_open);
        i = body_close;
        const auto calls = [&](std::string_view function) {
            for (auto at = body.find(function); at != std::string_view::npos; at = body.find(function, at + 1)) {
                const auto after = SkipSpace(body, at + function.size());
                if (StartsIdentifier(body, at) && IdentifierEnd(body, at) == at + function.size() && after < body.size() && body[after] == '(') return true;
            }
            return false;
        };
        if (!calls("dFdx") && !calls("dFdy") && std::ranges::none_of(differentiates | std::views::keys, calls)) continue;
        std::vector<std::string_view> values;
        std::vector<bool> takes;
        std::string parameters;
        for (const auto parameter : SplitArguments(text.substr(open + 1, close - open - 2))) {
            const auto name_at = parameter.find_last_of(" \t\r\n") + 1;
            const auto type = Trim(parameter.substr(0, name_at));
            const bool value = std::ranges::contains(ValueTypeNames, type);
            takes.push_back(value);
            parameters += std::format("{}{}", parameters.empty() ? "" : ", ", parameter);
            if (!value) continue;
            values.push_back(parameter.substr(name_at));
            for (const auto suffix : NeighborSuffixes) parameters += std::format(", {} {}{}", type, values.back(), suffix);
        }
        std::string rewritten;
        for (size_t at = 0; at < body.size();) {
            const auto word_start = NextIdentifier(body, at);
            rewritten += body.substr(at, word_start - at);
            if (word_start == body.size()) break;
            at = IdentifierEnd(body, word_start);
            const auto word = body.substr(word_start, at - word_start);
            const auto call_open = SkipSpace(body, at);
            const auto called = differentiates.find(word);
            const bool derivative = word == "dFdx" || word == "dFdy";
            if (call_open >= body.size() || body[call_open] != '(' || (!derivative && called == differentiates.end())) {
                rewritten += word;
                continue;
            }
            const auto call_close = Matching(body, call_open);
            const auto arguments = body.substr(call_open + 1, call_close - call_open - 2);
            at = call_close;
            if (derivative) {
                auto neighbor = AtNeighbor(arguments, word == "dFdx" ? 0u : 1u, values, body);
                if (!neighbor) return std::unexpected{neighbor.error()};
                rewritten += std::format("(({}) - ({}))", *neighbor, arguments);
                continue;
            }
            const auto passed = SplitArguments(arguments);
            if (passed.size() != called->second.size()) return Fail(std::format("{} calls {} with {} arguments", name, word, passed.size()));
            rewritten += std::format("{}(", word);
            for (size_t a = 0; a < passed.size(); ++a) {
                rewritten += std::format("{}{}", a ? ", " : "", passed[a]);
                for (uint32_t axis = 0; axis < 2u && called->second[a]; ++axis) {
                    auto neighbor = AtNeighbor(passed[a], axis, values, body);
                    if (!neighbor) return std::unexpected{neighbor.error()};
                    rewritten += ", " + *neighbor;
                }
            }
            rewritten += ')';
        }
        out += text.substr(copied, open + 1 - copied);
        out += parameters;
        out += text.substr(close - 1, body_open - close + 1);
        out += rewritten;
        copied = body_close;
        differentiates.insert_or_assign(std::string{name}, std::move(takes));
    }
    return out + std::string{text.substr(copied)};
}

// The top-level parts of a function file: each preprocessor line, and each function, structure or constant through its closing brace or semicolon.
std::vector<std::string_view> TopLevelParts(std::string_view text) {
    std::vector<std::string_view> parts;
    for (auto i = SkipSpace(text, 0); i < text.size(); i = SkipSpace(text, i)) {
        const auto start = i;
        if (text[i] == '#') {
            i = std::min(text.find('\n', i), text.size());
        } else {
            while (i < text.size()) {
                if (const auto end = SkipComment(text, i); end != i) {
                    i = end;
                } else if (text[i] == '(' || text[i] == '[') {
                    i = Matching(text, i);
                } else if (text[i] == ';') {
                    ++i;
                    break;
                } else if (text[i] == '{') {
                    i = Matching(text, i);
                    if (const auto after = SkipSpace(text, i); after < text.size() && text[after] == ';') i = after + 1;
                    break;
                } else {
                    ++i;
                }
            }
        }
        parts.push_back(text.substr(start, i - start));
    }
    return parts;
}

// The identifier that ends at `end`, past spaces before it.
std::string_view IdentifierBefore(std::string_view text, size_t end) {
    while (end > 0 && std::isspace(uint8_t(text[end - 1]))) --end;
    auto start = end;
    while (start > 0 && IsIdentifier(text[start - 1])) --start;
    return text.substr(start, end - start);
}
// The name a top-level part defines: a macro's, a structure's, or a function's or constant's.
// An operator is named by the type it returns, so the code that holds values of that type names it.
std::string_view DefinedName(std::string_view part) {
    if (part.starts_with('#')) {
        const auto directive = SkipSpace(part, 1);
        if (!part.substr(directive).starts_with("define")) return {};
        const auto name = SkipSpace(part, directive + 6);
        return part.substr(name, IdentifierEnd(part, name) - name);
    }
    if (part.starts_with("struct")) {
        const auto name = SkipSpace(part, 6);
        return part.substr(name, IdentifierEnd(part, name) - name);
    }
    const auto end = std::min(part.find_first_of("(="), part.size());
    const auto name = IdentifierBefore(part, end);
    return name.empty() ? IdentifierBefore(part, part.find("operator")) : name;
}

// The definitions by name, each in definition order.
using DefinitionNames = std::map<std::string, std::vector<uint32_t>, std::less<>>;

// The definitions among `names` that the identifiers of the text outside comments name, in definition order.
std::vector<uint32_t> Uses(std::string_view text, const DefinitionNames &names) {
    std::vector<uint32_t> uses;
    for (auto i = NextIdentifier(text, 0); i < text.size(); i = NextIdentifier(text, i)) {
        const auto start = i;
        i = IdentifierEnd(text, i);
        if (const auto it = names.find(text.substr(start, i - start)); it != names.end()) uses.append_range(it->second);
    }
    std::ranges::sort(uses);
    const auto [first, last] = std::ranges::unique(uses);
    uses.erase(first, last);
    return uses;
}

// The function files loaded so far and the math library.
struct Loaded {
    std::map<fs::path, std::vector<uint32_t>> Scopes; // By file, its definitions after those of the files it includes
    std::vector<std::string> Names; // By definition, the name it defines
    std::vector<uint32_t> Math; // The math library's definitions, which every function file and expression can use
    Differentiates Differentiating; // The functions loaded so far that differentiate
};

DefinitionNames NamesOf(const Loaded &loaded, std::span<const uint32_t> scope) {
    DefinitionNames names;
    for (const auto d : scope) names[loaded.Names[d]].push_back(d);
    return names;
}

// Loads the file after the files it includes, each relative to the including file, and returns its scope: its definitions after those of the files it includes.
// The files of a native node's implementation take NativeRewrites, and their functions that differentiate take neighbor values.
std::expected<std::vector<uint32_t>, std::string> LoadFile(Library &lib, Loaded &loaded, const fs::path &path, bool native = false) {
    const auto normal = path.lexically_normal();
    if (const auto it = loaded.Scopes.find(normal); it != loaded.Scopes.end()) return it->second;
    auto read = ReadFile(normal);
    if (!read) return std::unexpected{read.error()};
    const auto text = native ? Rewritten(std::move(*read)) : std::move(*read);
    std::vector<uint32_t> scope;
    std::string body;
    for (size_t pos = 0; pos < text.size();) {
        const auto end = std::min(text.find('\n', pos), text.size());
        const std::string_view line{text.data() + pos, end - pos};
        if (Trim(line).starts_with("#include")) {
            const auto open = line.find('"'), close = line.rfind('"');
            if (open == close) return Fail(std::format("{}: malformed include", normal.string()));
            const auto included = LoadFile(lib, loaded, normal.parent_path() / line.substr(open + 1, close - open - 1), native);
            if (!included) return std::unexpected{included.error()};
            for (const auto d : *included) {
                if (!std::ranges::contains(scope, d)) scope.push_back(d);
            }
        } else {
            body.append(line);
            body.push_back('\n');
        }
        pos = end + 1;
    }
    if (native) {
        auto differentiated = Differentiated(body, loaded.Differentiating);
        if (!differentiated) return Fail(std::format("{}: {}", normal.string(), differentiated.error()));
        body = std::move(*differentiated);
    }
    const auto ported = PortToMetal(body);
    auto names = NamesOf(loaded, loaded.Math);
    for (const auto d : scope) names[loaded.Names[d]].push_back(d);
    for (const auto part : TopLevelParts(ported)) {
        const auto d = uint32_t(lib.Definitions.size());
        lib.Definitions.push_back({std::format("{}\n", part), Uses(part, names)});
        const auto &name = loaded.Names.emplace_back(DefinedName(part));
        names[name].push_back(d);
        scope.push_back(d);
    }
    return loaded.Scopes.emplace(normal, std::move(scope)).first->second;
}

// An implementation element of a genglsl file.
struct GlslImplementation {
    std::string Source, Function;
    fs::path File;
};

std::expected<InlineExpression, std::string> ParseExpression(const Declaration &d, std::string_view source, const DefinitionNames &math) {
    const auto ported = PortToMetal(source);
    auto expression = Trim(ported);
    while (expression.ends_with(';')) expression = Trim(expression.substr(0, expression.size() - 1));
    for (size_t open = expression.find("{{"); open != std::string_view::npos; open = expression.find("{{", open + 2)) {
        const auto close = expression.find("}}", open);
        const auto name = expression.substr(open + 2, close - open - 2);
        if (close == std::string_view::npos || !PortIndex(d.Inputs, name)) return Fail(std::format("{}: expression refers to unknown input '{}'", d.Name, name));
    }
    // An expression also names the operators on its port types.
    std::string named{expression};
    for (const auto &port : d.Inputs) named += std::format(" {}", MetalType(port.Type));
    for (const auto &port : d.Outputs) named += std::format(" {}", MetalType(port.Type));
    return InlineExpression{std::string{expression}, Uses(named, math)};
}
} // namespace

std::string_view Name(Type type) { return Types[size_t(type)].Name; }
uint32_t Words(Type type) { return Types[size_t(type)].Words; }
std::string_view MetalType(Type type) { return Types[size_t(type)].MetalType; }

std::expected<Library, std::string> Load(std::span<const std::filesystem::path> folders) {
    std::vector<Document> definitions, implementations;
    for (const auto &folder : folders) {
        if (!fs::is_directory(folder)) return Fail(std::format("no library folder {}", folder.string()));
        auto defined = ReadDocuments(folder);
        if (!defined) return std::unexpected{defined.error()};
        std::ranges::move(*defined, std::back_inserter(definitions));
        auto implemented = ReadDocuments(folder / "genglsl");
        if (!implemented) return std::unexpected{implemented.error()};
        std::ranges::move(*implemented, std::back_inserter(implementations));
    }

    Library lib;
    std::map<std::string, uint32_t, std::less<>> geomprop_indices;
    std::vector<const xml::Element *> geomprop_elements;
    for (const auto &document : definitions) {
        for (const auto &e : document.Root.Children) {
            if (e.Name != "geompropdef") continue;
            if (!geomprop_indices.emplace(xml::Find(e, "name"), uint32_t(geomprop_elements.size())).second) return Fail(std::format("geometric property {} repeats", xml::Find(e, "name")));
            geomprop_elements.push_back(&e);
        }
    }
    for (const auto &document : definitions) {
        for (const auto &e : document.Root.Children) {
            if (e.Name != "nodedef") continue;
            auto declaration = ParseDeclaration(e, geomprop_indices);
            if (!declaration) return Fail(std::format("{}: {}", document.Path.string(), declaration.error()));
            const auto index = uint32_t(lib.Declarations.size());
            lib.DeclarationsByNode[declaration->Node].push_back(index);
            lib.Declarations.push_back(std::move(*declaration));
            if (!lib.DeclarationsByName.emplace(xml::Find(e, "name"), index).second) return Fail(std::format("declaration {} repeats", xml::Find(e, "name")));
        }
    }
    for (const auto *e : geomprop_elements) {
        const auto name = xml::Find(*e, "name");
        const auto type = ParseType(xml::Find(*e, "type"));
        const auto space = xml::Find(*e, "space"), index = xml::Find(*e, "index");
        std::vector<TypedInput> inputs;
        if (!space.empty()) inputs.push_back({"space", Type::String});
        if (!index.empty()) inputs.push_back({"index", Type::Integer});
        const auto declaration = type ? FindDeclaration(lib, xml::Find(*e, "geomprop"), inputs, *type) : std::nullopt;
        if (!declaration) return Fail(std::format("geometric property {} matches no declaration", name));
        GraphNode geomprop{.Name = std::string{name}, .Declaration = *declaration};
        for (const auto &input : inputs) {
            const auto &ports = lib.Declarations[*declaration].Inputs;
            auto value = ParseValue(input.Type, input.Type == Type::String ? space : index);
            if (!value) return Fail(std::format("geometric property {}: {}", name, value.error()));
            geomprop.Inputs.push_back({*PortIndex(ports, input.Name), std::move(*value)});
        }
        lib.GeomProps.push_back(std::move(geomprop));
    }

    std::vector<std::optional<GlslImplementation>> glsl(lib.Declarations.size());
    for (const auto &document : implementations) {
        for (const auto &e : document.Root.Children) {
            if (e.Name != "implementation") continue;
            const auto declaration = lib.DeclarationsByName.find(xml::Find(e, "nodedef"));
            if (declaration == lib.DeclarationsByName.end()) return Fail(std::format("{}: implementation {} names unknown declaration {}", document.Path.string(), xml::Find(e, "name"), xml::Find(e, "nodedef")));
            auto &implementation = glsl[declaration->second];
            if (implementation) return Fail(std::format("{} has several implementations", declaration->first));
            const auto file = xml::Find(e, "file");
            implementation = {
                .Source = std::string{xml::Find(e, "sourcecode")},
                .Function = std::string{xml::Find(e, "function")},
                .File = file.empty() ? fs::path{} : document.Path.parent_path() / file,
            };
        }
    }
    std::vector<std::optional<Graph>> graphs(lib.Declarations.size());
    for (const auto &document : definitions) {
        for (const auto &e : document.Root.Children) {
            if (e.Name != "nodegraph" || xml::Find(e, "nodedef").empty()) continue;
            const auto declaration = lib.DeclarationsByName.find(xml::Find(e, "nodedef"));
            if (declaration == lib.DeclarationsByName.end()) return Fail(std::format("{}: graph {} defines unknown declaration {}", document.Path.string(), xml::Find(e, "name"), xml::Find(e, "nodedef")));
            if (graphs[declaration->second]) return Fail(std::format("{} has several graph definitions", declaration->first));
            auto graph = ParseGraph(lib, e, lib.Declarations[declaration->second]);
            if (!graph) return Fail(std::format("{}: graph {}: {}", document.Path.string(), xml::Find(e, "name"), graph.error()));
            graphs[declaration->second] = std::move(*graph);
        }
    }

    Loaded loaded;
    for (const auto &folder : folders) {
        for (const auto *name : {"mx_math.metal", "mx_matscalaroperators.metal"}) {
            const auto path = folder / "genmsl" / "lib" / name;
            if (!fs::exists(path)) continue;
            auto file = LoadFile(lib, loaded, path);
            if (!file) return std::unexpected{file.error()};
            loaded.Math.append_range(std::move(*file));
        }
    }
    const auto math = NamesOf(loaded, loaded.Math);
    for (size_t i = 0; i < lib.Declarations.size(); ++i) {
        const auto &d = lib.Declarations[i];
        const bool named_native = std::ranges::contains(NativeNodes, d.Node);
        const auto &implementation = glsl[i];
        const bool has_source = implementation && (!implementation->Source.empty() || !implementation->File.empty());
        const auto is_value = [](const Port &p) { return Words(p.Type) > 0; };
        const auto read = [&](const Port &p) { return !has_source || implementation->Source.empty() || implementation->Source.contains("{{" + p.Name + "}}"); };
        const bool passes_non_value = !std::ranges::all_of(d.Outputs, is_value) || std::ranges::any_of(d.Inputs, [&](const Port &p) { return !is_value(p) && read(p); });
        // The overloads of the function an implementation names in its file.
        const auto function = [&](bool native) -> std::expected<std::vector<uint32_t>, std::string> {
            const auto scope = LoadFile(lib, loaded, implementation->File, native);
            if (!scope) return std::unexpected{scope.error()};
            auto overloads = *scope | std::views::filter([&](uint32_t d) { return loaded.Names[d] == implementation->Function; }) | std::ranges::to<std::vector>();
            if (overloads.empty()) return Fail(std::format("{}: no function {}", implementation->File.string(), implementation->Function));
            return overloads;
        };
        auto implemented = [&]() -> std::expected<Implementation, std::string> {
            if (graphs[i] && !named_native) {
                if (has_source) return Fail(std::format("{} has a graph definition and another implementation", d.Name));
                return std::move(*graphs[i]);
            }
            if (named_native && has_source && !implementation->File.empty()) return function(true).transform([&](std::vector<uint32_t> overloads) -> Implementation { return Native{std::move(overloads), implementation->Function}; });
            if (named_native || passes_non_value) return Native{};
            if (!has_source) return Fail(std::format("{} has no implementation", d.Name));
            if (!implementation->Source.empty()) return ParseExpression(d, implementation->Source, math);
            return function(false).transform([&](std::vector<uint32_t> overloads) -> Implementation { return FunctionCall{implementation->Function, std::move(overloads)}; });
        }();
        if (!implemented) return std::unexpected{implemented.error()};
        lib.Implementations.push_back(std::move(*implemented));
    }
    return lib;
}

namespace {
std::vector<fs::path> &AddedFolders() {
    static std::vector<fs::path> folders;
    return folders;
}
} // namespace

void AddLibraryFolder(fs::path folder) { AddedFolders().push_back(std::move(folder)); }

const Library &Libraries() {
    static const Library library = [] {
        const auto standard = Paths::Base() / "materialx";
        std::vector folders{standard / "stdlib", standard / "cmlib", standard / "nprlib", Paths::Res() / "nodes"};
        folders.append_range(AddedFolders());
        auto loaded = Load(folders);
        if (!loaded) throw std::runtime_error(std::format("Failed to load the node libraries: {}", loaded.error()));
        return std::move(*loaded);
    }();
    return library;
}

std::span<const uint32_t> NodeDeclarations(const Library &lib, std::string_view node) {
    const auto it = lib.DeclarationsByNode.find(node);
    return it == lib.DeclarationsByNode.end() ? std::span<const uint32_t>{} : std::span{it->second};
}

bool OutputsMatch(const Declaration &d, std::optional<Type> output) {
    return !output || (*output == Type::MultiOutput ? d.Outputs.size() > 1 : d.Outputs.size() == 1 && d.Outputs[0].Type == *output);
}

std::optional<uint32_t> FindDeclaration(const Library &lib, std::string_view node, std::span<const TypedInput> inputs, std::optional<Type> output) {
    for (const auto i : NodeDeclarations(lib, node)) {
        const auto &d = lib.Declarations[i];
        const auto has_input = [&](const TypedInput &in) { return std::ranges::any_of(d.Inputs, [&](const Port &p) { return p.Name == in.Name && p.Type == in.Type; }); };
        if (OutputsMatch(d, output) && std::ranges::all_of(inputs, has_input)) return i;
    }
    return {};
}

namespace {
// An input element of the name and type, with the attributes of its source.
xml::Element InputElement(std::string name, Type type, std::initializer_list<xml::Attribute> source) {
    xml::Element input{.Name = "input", .Attributes = {{"name", std::move(name)}, {"type", std::string{Name(type)}}}};
    input.Attributes.append_range(source);
    return input;
}

// The element of a node of declaration `d` named `name` with the input elements.
xml::Element NodeElement(const Declaration &d, std::string name, std::vector<xml::Element> inputs = {}) {
    const auto type = d.Outputs.size() == 1u ? Name(d.Outputs.front().Type) : Name(Type::MultiOutput);
    return {.Name = d.Node, .Attributes = {{"name", std::move(name)}, {"type", std::string{type}}, {"nodedef", d.Name}}, .Children = std::move(inputs)};
}

// The element of a node of `node` named `name` taking a vector3 and a float, which maps a normal between [-1, 1] and [0, 1].
xml::Element NormalElement(const Library &lib, std::string_view node, std::string name, std::vector<xml::Element> inputs) {
    const auto first = node == "normalmap" ? "in" : "in1", second = node == "normalmap" ? "scale" : "in2";
    const auto declaration = *FindDeclaration(lib, node, std::array{TypedInput{first, Type::Vector3}, TypedInput{second, Type::Float}}, Type::Vector3);
    return NodeElement(lib.Declarations[declaration], std::move(name), std::move(inputs));
}

// The first child element that satisfies the predicate.
const xml::Element *FindChild(const xml::Element &e, auto &&pred) {
    const auto it = std::ranges::find_if(e.Children, pred);
    return it == e.Children.end() ? nullptr : &*it;
}
} // namespace

std::string WriteMaterial(const Library &lib, const MaterialDocument &material, std::span<const xml::Element> definitions) {
    xml::Element root{.Name = "materialx", .Attributes = {{"version", "1.39"}}};
    root.Children.append_range(definitions);
    const auto &graph = material.Graph;
    xml::Element nodegraph{.Name = "nodegraph", .Attributes = {{"name", graph.Name}}};
    // A link to an output of a node, naming the output when the node has several.
    const auto link = [&](xml::Element &input, const NodeOutput &from) {
        input.Attributes.push_back({"nodename", graph.Nodes[from.Node].Name});
        if (const auto &outputs = lib.Declarations[graph.Nodes[from.Node].Declaration].Outputs; outputs.size() > 1u) input.Attributes.push_back({"output", outputs[from.Output].Name});
    };
    for (const auto &node : graph.Nodes) {
        const auto &d = lib.Declarations[node.Declaration];
        auto element = NodeElement(d, node.Name);
        for (const auto &[index, source] : node.Inputs) {
            auto input = InputElement(d.Inputs[index].Name, d.Inputs[index].Type, {});
            if (const auto *from = std::get_if<NodeOutput>(&source)) link(input, *from);
            else input.Attributes.push_back({"value", FormatValue(d.Inputs[index].Type, std::get<Value>(source))});
            element.Children.push_back(std::move(input));
        }
        nodegraph.Children.push_back(std::move(element));
    }
    for (size_t o = 0; o < material.Outputs.size(); ++o) {
        xml::Element output{.Name = "output", .Attributes = {{"name", material.Outputs[o].first}, {"type", std::string{Name(material.Outputs[o].second)}}}};
        link(output, std::get<NodeOutput>(graph.Outputs[o]));
        nodegraph.Children.push_back(std::move(output));
    }
    root.Children.push_back(std::move(nodegraph));
    const auto shader_name = "SR_" + material.Name;
    xml::Element shader{.Name = "gltf_pbr", .Attributes = {{"name", shader_name}, {"type", "surfaceshader"}, {"nodedef", "ND_gltf_pbr_surfaceshader"}}};
    for (const auto &input : material.Surface) {
        if (const auto *value = std::get_if<Value>(&input.Source)) {
            shader.Children.push_back(InputElement(input.Name, input.Type, {{"value", FormatValue(input.Type, *value)}}));
            continue;
        }
        const xml::Attribute from_graph{"nodegraph", graph.Name}, from_output{"output", material.Outputs[std::get<uint32_t>(input.Source)].first};
        if (!input.TangentNormal) {
            shader.Children.push_back(InputElement(input.Name, input.Type, {from_graph, from_output}));
            continue;
        }
        // The normal maps to [0, 1] as its half plus a half, which normalmap reads in the tangent frame.
        const auto scaled = input.Name + "_scaled", encoded = input.Name + "_encoded", mapped = input.Name + "_map";
        const auto half = InputElement("in2", Type::Float, {{"value", "0.5"}});
        root.Children.push_back(NormalElement(lib, "multiply", scaled, {InputElement("in1", Type::Vector3, {from_graph, from_output}), half}));
        root.Children.push_back(NormalElement(lib, "add", encoded, {InputElement("in1", Type::Vector3, {{"nodename", scaled}}), half}));
        root.Children.push_back(NormalElement(lib, "normalmap", mapped, {InputElement("in", Type::Vector3, {{"nodename", encoded}})}));
        shader.Children.push_back(InputElement(input.Name, input.Type, {{"nodename", mapped}}));
    }
    root.Children.push_back(std::move(shader));
    xml::Element surface_material{.Name = "surfacematerial", .Attributes = {{"name", material.Name}, {"type", "material"}, {"nodedef", "ND_surfacematerial"}}};
    surface_material.Children.push_back(InputElement("surfaceshader", Type::SurfaceShader, {{"nodename", shader_name}}));
    root.Children.push_back(std::move(surface_material));
    return xml::Write(root);
}

std::expected<MaterialDocument, std::string> ReadMaterial(const Library &lib, std::string_view text) {
    const auto root = xml::Parse(text);
    if (!root) return std::unexpected{root.error()};
    const auto *material = FindChild(*root, [](const xml::Element &e) { return e.Name == "surfacematerial"; });
    const auto *shader = FindChild(*root, [](const xml::Element &e) { return e.Name == "gltf_pbr"; });
    const auto *nodegraph = FindChild(*root, [](const xml::Element &e) { return e.Name == "nodegraph" && xml::Find(e, "nodedef").empty(); });
    if (!material || !shader) return Fail("the document holds no surface material of a gltf_pbr shader");
    Declaration d{.Name = nodegraph ? std::string{xml::Find(*nodegraph, "name")} : std::string{}};
    for (const auto &child : nodegraph ? std::span{nodegraph->Children} : std::span<const xml::Element>{}) {
        if (child.Name != "output") continue;
        const auto type = ParseType(xml::Find(child, "type"));
        if (!type) return Fail(std::format("graph output '{}' has unknown type '{}'", xml::Find(child, "name"), xml::Find(child, "type")));
        d.Outputs.push_back({.Name = std::string{xml::Find(child, "name")}, .Type = *type});
    }
    MaterialDocument read{.Name = std::string{xml::Find(*material, "name")}};
    if (nodegraph) {
        auto graph = ParseGraph(lib, *nodegraph, d);
        if (!graph) return std::unexpected{graph.error()};
        read.Graph = std::move(*graph);
    }
    for (const auto &output : d.Outputs) read.Outputs.emplace_back(output.Name, output.Type);
    for (const auto &input : shader->Children) {
        const auto type = ParseType(xml::Find(input, "type"));
        if (!type) return Fail(std::format("shader input '{}' has unknown type '{}'", xml::Find(input, "name"), xml::Find(input, "type")));
        SurfaceInput surface{.Name = std::string{xml::Find(input, "name")}, .Type = *type};
        // A linked input reads a graph output, directly or through the first linked input of each document node it reads.
        const xml::Element *at = &input;
        while (xml::Find(*at, "nodegraph").empty() && !xml::Find(*at, "nodename").empty()) {
            const auto *node = FindChild(*root, [&](const xml::Element &e) { return xml::Find(e, "name") == xml::Find(*at, "nodename"); });
            if (!node) return Fail(std::format("shader input '{}' reads unknown node '{}'", surface.Name, xml::Find(*at, "nodename")));
            surface.TangentNormal |= node->Name == "normalmap";
            at = FindChild(*node, [](const xml::Element &e) { return !xml::Find(e, "nodegraph").empty() || !xml::Find(e, "nodename").empty(); });
            if (!at) return Fail(std::format("shader input '{}' reads no graph output", surface.Name));
        }
        if (const auto output = xml::Find(*at, "output"); !xml::Find(*at, "nodegraph").empty()) {
            const auto index = PortIndex(d.Outputs, output);
            if (!index) return Fail(std::format("shader input '{}' reads unknown graph output '{}'", surface.Name, output));
            surface.Source = *index;
        } else {
            auto value = ParseValue(*type, xml::Find(input, "value"));
            if (!value) return Fail(std::format("shader input '{}': {}", surface.Name, value.error()));
            surface.Source = std::move(*value);
        }
        read.Surface.push_back(std::move(surface));
    }
    return read;
}

std::expected<std::vector<xml::Element>, std::string> OwnDefinitions(const Library &lib, std::span<const uint32_t> declarations) {
    const auto documents = ReadDocuments(Paths::Res() / "nodes");
    if (!documents) return std::unexpected{documents.error()};
    const auto element = [&](std::string_view kind, std::string_view attribute, std::string_view value) -> const xml::Element * {
        for (const auto &document : *documents) {
            if (const auto *found = FindChild(document.Root, [&](const xml::Element &e) { return e.Name == kind && xml::Find(e, attribute) == value; })) return found;
        }
        return nullptr;
    };
    std::vector<xml::Element> definitions;
    std::vector<uint32_t> written;
    for (std::vector<uint32_t> pending{std::from_range, declarations}; !pending.empty();) {
        const auto declaration = pending.back();
        pending.pop_back();
        if (std::ranges::contains(written, declaration)) continue;
        written.push_back(declaration);
        const auto &d = lib.Declarations[declaration];
        const auto *definition = element("nodedef", "name", d.Name);
        if (!definition) continue;
        const auto *graph = element("nodegraph", "nodedef", d.Name);
        const auto *implementation = std::get_if<Graph>(&lib.Implementations[declaration]);
        if (!graph || !implementation) return Fail(std::format("MeshEditor implements node '{}' in its own code, which other MaterialX readers lack", d.Node));
        definitions.push_back(*definition);
        definitions.push_back(*graph);
        for (const auto &node : implementation->Nodes) pending.push_back(node.Declaration);
    }
    return definitions;
}
} // namespace mx
