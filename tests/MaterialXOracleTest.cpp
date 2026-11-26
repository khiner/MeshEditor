#include "OpsFixture.h"
#include "Parallel.h"
#include "materialx/Flatten.h"
#include "metal/AutoreleaseScope.h"
#include "metal/MetalCpp.h"
#include "metal/MslSource.h"
#include "metal/PassChain.h"

#include <MaterialXFormat/Util.h>
#include <MaterialXGenMsl/MslShaderGenerator.h>
#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/Shader.h>

using namespace boost::ut;
namespace ref = MaterialX;

namespace {
// The value sets of a declaration's literals, each its own evaluation.
constexpr uint32_t ValueSets{32};
// The test mesh is a grid of this many quads along each side.
constexpr uint32_t GridSize{16};

// Nodes with value outputs that one side cannot evaluate yet, and why.
const std::array<std::pair<std::vector<std::string_view>, std::string_view>, 5> Skipped{{
    {{"blur"}, "MaterialX passes the input through, which the specification's blur gives only at size zero."},
    {{"bitangent"}, "Reads a bitangent, which no domain stores."},
    {{"viewdirection", "gooch_shade"}, "Reads the view direction, which no domain has."},
    {{"elementindex"}, "MaterialX has no implementation of MeshEditor's own source."},
    {{"convertdomain", "curvature", "boundary", "linked", "adjacent", "edgeloop", "edgering", "selection"}, "MaterialX has no implementation of MeshEditor's own stored-field ops."},
}};

bool IsSkipped(std::string_view node) {
    return std::ranges::any_of(Skipped, [&](const auto &skipped) { return std::ranges::contains(skipped.first, node); });
}
bool IsMatrix(mx::Type type) { return type == mx::Type::Matrix33 || type == mx::Type::Matrix44; }

// Every image input reads the test image, a grid of this many texels along each side.
constexpr uint32_t ImageSize{8};

// The reference library: MaterialX's standard libraries MeshEditor loads and MeshEditor's own nodes, read by MaterialX.
const ref::DocumentPtr &ReferenceLibrary() {
    static const auto library = [] {
        auto doc = ref::createDocument();
        ref::loadLibraries({"libraries/targets", "libraries/stdlib", "libraries/cmlib", "libraries/nprlib"}, ref::FileSearchPath{MESHEDITOR_SOURCE_DIR "/lib/materialx"}, doc);
        ref::loadLibraries({"res/nodes"}, ref::FileSearchPath{MESHEDITOR_SOURCE_DIR}, doc);
        return doc;
    }();
    return library;
}

// The words of a reference value of the type, in file order, or the text of a string.
mx::Value ReferenceValue(std::string_view type_name, const ref::ValuePtr &value) {
    if (type_name == "string" || type_name == "filename") return value ? value->getValueString() : std::string{};
    const auto type = *std::ranges::find_if(std::views::iota(0u, uint32_t(mx::Type::MultiOutput) + 1u), [&](uint32_t t) { return mx::Name(mx::Type(t)) == type_name; });
    std::vector<uint32_t> words(mx::Words(mx::Type(type)), 0u);
    if (!value || words.empty()) return words;
    const auto floats = [&](const auto &v, size_t count) {
        for (size_t i = 0; i < count; ++i) words[i] = std::bit_cast<uint32_t>(float(v[i]));
    };
    const auto matrix = [&](const auto &m) {
        const auto n = m.numRows();
        for (size_t r = 0; r < n; ++r) {
            for (size_t c = 0; c < n; ++c) words[r * n + c] = std::bit_cast<uint32_t>(m[r][c]);
        }
    };
    if (value->isA<bool>()) words[0] = value->asA<bool>() ? 1u : 0u;
    else if (value->isA<int>()) words[0] = std::bit_cast<uint32_t>(value->asA<int>());
    else if (value->isA<float>()) words[0] = std::bit_cast<uint32_t>(value->asA<float>());
    else if (value->isA<ref::Vector2>()) floats(value->asA<ref::Vector2>(), 2);
    else if (value->isA<ref::Vector3>()) floats(value->asA<ref::Vector3>(), 3);
    else if (value->isA<ref::Vector4>()) floats(value->asA<ref::Vector4>(), 4);
    else if (value->isA<ref::Color3>()) floats(value->asA<ref::Color3>(), 3);
    else if (value->isA<ref::Color4>()) floats(value->asA<ref::Color4>(), 4);
    else if (value->isA<ref::Matrix33>()) matrix(value->asA<ref::Matrix33>());
    else if (value->isA<ref::Matrix44>()) matrix(value->asA<ref::Matrix44>());
    return words;
}

// What a declaration states about its ports, comparable between the registry and the reference library.
struct PortStatement {
    std::string Name, Type;
    mx::Value Default;
    bool Uniform;
    std::string GeomProp;
    bool operator==(const PortStatement &) const = default;
};
struct DeclarationStatement {
    std::string Node;
    std::vector<PortStatement> Inputs, Outputs;
    bool operator==(const DeclarationStatement &) const = default;
};

DeclarationStatement Statement(const mx::Library &lib, const mx::Declaration &d) {
    const auto ports = [&](const std::vector<mx::Port> &ports) {
        std::vector<PortStatement> statements;
        for (const auto &port : ports) {
            statements.push_back({port.Name, std::string{mx::Name(port.Type)}, port.Default, port.Uniform, port.DefaultGeomProp ? lib.GeomProps[*port.DefaultGeomProp].Name : std::string{}});
        }
        return statements;
    };
    return {d.Node, ports(d.Inputs), ports(d.Outputs)};
}
DeclarationStatement Statement(const ref::NodeDef &d) {
    DeclarationStatement statement{d.getNodeString()};
    for (const auto &input : d.getActiveInputs()) {
        statement.Inputs.push_back({input->getName(), input->getType(), ReferenceValue(input->getType(), input->getValue()), input->getIsUniform(), input->getDefaultGeomPropString()});
    }
    for (const auto &output : d.getActiveOutputs()) {
        const auto &text = output->getAttribute("default");
        const auto value = text.empty() ? nullptr : ref::Value::createValueFromStrings(text, output->getType());
        statement.Outputs.push_back({output->getName(), output->getType(), ReferenceValue(output->getType(), value), output->getIsUniform(), std::string{}});
    }
    return statement;
}

// The declaration's output type, or the multi-output type for several outputs.
std::string_view OutputTypeName(const mx::Declaration &d) { return d.Outputs.size() == 1 ? mx::Name(d.Outputs[0].Type) : mx::Name(mx::Type::MultiOutput); }

// Whether another declaration of the node takes the same input types to a different output type, so a node stores its output type.
bool StoresOutput(const mx::Library &lib, const mx::Declaration &d) {
    const auto inputs = [](const mx::Declaration &other) { return other.Inputs | std::views::transform([](const mx::Port &p) { return std::pair{p.Name, p.Type}; }) | std::ranges::to<std::vector>(); };
    return std::ranges::any_of(mx::NodeDeclarations(lib, d.Node), [&](uint32_t other) {
        const auto &o = lib.Declarations[other];
        return inputs(o) == inputs(d) && OutputTypeName(o) != OutputTypeName(d);
    });
}

mx::Type TypeOf(AttributeType storage) {
    switch (storage) {
        case AttributeType::Float: return mx::Type::Float;
        case AttributeType::Vec2: return mx::Type::Vector2;
        case AttributeType::Vec3: return mx::Type::Vector3;
        case AttributeType::Vec4: return mx::Type::Vector4;
        case AttributeType::Int: return mx::Type::Integer;
        case AttributeType::Bool: return mx::Type::Boolean;
    }
    std::unreachable();
}

// A grid of quads with random heights, texture coordinates, tangents and colors.
MeshSource OracleMesh() {
    std::mt19937 rng{7u};
    std::uniform_real_distribution unit{-1.f, 1.f};
    constexpr auto n = GridSize + 1u;
    MeshData data;
    MeshVertexAttributes attributes{.Tangents = std::vector<vec4>{}, .Colors0 = std::vector<vec4>{}, .TexCoords0 = std::vector<vec2>{}, .TexCoords1 = std::vector<vec2>{}, .TexCoords2 = std::vector<vec2>{}, .TexCoords3 = std::vector<vec2>{}, .Colors0ComponentCount = 4u};
    for (uint32_t y = 0; y < n; ++y) {
        for (uint32_t x = 0; x < n; ++x) {
            data.Positions.push_back({float(x) - float(GridSize) / 2.f + 0.3f * unit(rng), float(y) - float(GridSize) / 2.f + 0.3f * unit(rng), 2.f * unit(rng)});
            attributes.Tangents->push_back({unit(rng), unit(rng), unit(rng), 1.f});
            attributes.Colors0->push_back({unit(rng) + 0.5f, unit(rng) + 0.5f, unit(rng) + 0.5f, unit(rng) + 0.5f});
            for (auto *uvs : {&attributes.TexCoords0, &attributes.TexCoords1, &attributes.TexCoords2, &attributes.TexCoords3}) (*uvs)->push_back({1.5f * unit(rng) + 0.5f, 1.5f * unit(rng) + 0.5f});
        }
    }
    for (uint32_t y = 0; y < GridSize; ++y) {
        for (uint32_t x = 0; x < GridSize; ++x) data.AddFace(std::array{y * n + x, y * n + x + 1u, (y + 1u) * n + x + 1u, (y + 1u) * n + x});
    }
    return {.Data = std::move(data), .Attrs = std::move(attributes)};
}

// The (input, output) type pairs of the `convert` declarations.
std::vector<std::pair<mx::Type, mx::Type>> ConvertSignatures(const mx::Library &lib) {
    std::vector<std::pair<mx::Type, mx::Type>> signatures;
    for (const auto i : mx::NodeDeclarations(lib, "convert")) {
        const auto &d = lib.Declarations[i];
        signatures.emplace_back(d.Inputs.front().Type, d.Outputs.front().Type);
    }
    std::ranges::sort(signatures);
    const auto [first, last] = std::ranges::unique(signatures);
    signatures.erase(first, last);
    return signatures;
}

// A literal of a test graph: input `Input` of node `Node`, random in each value set or fixed, which the program reads from word `Offset` of its value block.
struct LiteralInput {
    uint32_t Node, Input;
    mx::Type Type;
    std::optional<std::vector<uint32_t>> Fixed;
    uint32_t Offset;
};
// Each image input reads the test image through its slot of the program's image table, and Words are the words of the value block.
struct TestGraph {
    ops::Graph Graph;
    std::vector<LiteralInput> Literals;
    uint32_t Images{}, Words{};
};

// Where a test graph's value inputs come from.
enum class Inputs : uint8_t {
    Literals,
    Attributes, // Named attributes of the domain
    Geometry, // The geometric property an input reads when unset, and named attributes for the rest
};
constexpr std::array<std::string_view, 3> InputsNames{"literals", "attributes", "geometry"};

// The declaration with each value input set from the source, where a uniform input takes its default literal.
// Off one point, a matrix output reaches stored values through the matrix's product with each basis vector.
TestGraph DeclarationGraph(const mx::Library &lib, uint32_t declaration, Inputs inputs, ops::Domain domain, auto &&attribute) {
    const auto &d = lib.Declarations[declaration];
    TestGraph t;
    auto &nodes = t.Graph.Nodes;
    std::map<AttributeType, uint32_t> reads;
    const auto literal = [&](ops::Node &node, uint32_t n, const mx::Port &port, std::optional<std::vector<uint32_t>> fixed) {
        const auto input = uint32_t(&port - lib.Declarations[std::get<uint32_t>(node.Op)].Inputs.data());
        node.Inputs.push_back({port.Name, ops::Word{port.Type, t.Words}});
        t.Literals.push_back({n, input, port.Type, std::move(fixed), t.Words});
        t.Words += mx::Words(port.Type);
    };
    const auto read = [&](mx::Type type) {
        const auto storage = *ops::StorageOf(type);
        const auto g = *mx::FindDeclaration(lib, "geompropvalue", {}, type);
        const auto n = uint32_t(nodes.size());
        nodes.push_back({.Op = g, .Inputs = {{"geomprop", ops::Literal{mx::Type::String, attribute(storage, reads[storage]++)}}}});
        literal(nodes.back(), n, *std::ranges::find(lib.Declarations[g].Inputs, "default", &mx::Port::Name), std::vector<uint32_t>(mx::Words(type)));
        return ops::Link{n, 0u};
    };
    const auto read_matrix = [&](mx::Type type) {
        const auto vector = type == mx::Type::Matrix33 ? mx::Type::Vector3 : mx::Type::Vector4;
        const auto columns = mx::Words(vector);
        std::vector<ops::Input> vectors;
        for (uint32_t c = 0; c < columns; ++c) vectors.push_back({std::format("in{}", c + 1u), read(vector)});
        const std::array typed{mx::TypedInput{"in1", vector}};
        nodes.push_back({.Op = *mx::FindDeclaration(lib, "creatematrix", typed, type), .Inputs = std::move(vectors)});
        return ops::Link{uint32_t(nodes.size() - 1u), 0u};
    };
    std::vector<ops::Input> links;
    std::vector<const mx::Port *> literals;
    for (const auto &port : d.Inputs) {
        if (port.Type == mx::Type::Filename) {
            links.push_back({port.Name, ops::Image{t.Images++}});
        } else if (!HasValue(port.Type)) {
            const auto storage = ops::StorageOf(d.Outputs[0].Type);
            if (port.Name == "geomprop" && storage) links.push_back({port.Name, ops::Literal{mx::Type::String, attribute(*storage, reads[*storage]++)}});
        } else if (inputs == Inputs::Literals || port.Uniform) {
            literals.push_back(&port);
        } else if (inputs == Inputs::Attributes || !port.DefaultGeomProp) {
            links.push_back({port.Name, IsMatrix(port.Type) ? read_matrix(port.Type) : read(port.Type)});
        }
    }
    const auto node = uint32_t(nodes.size());
    nodes.push_back({.Op = declaration, .Inputs = std::move(links)});
    for (const auto *port : literals) literal(nodes[node], node, *port, port->Uniform ? std::optional{std::get<std::vector<uint32_t>>(port->Default)} : std::nullopt);
    for (uint32_t o = 0; o < d.Outputs.size(); ++o) {
        const auto type = d.Outputs[o].Type;
        if (domain == ops::Domain::Point || !IsMatrix(type)) {
            t.Graph.Outputs.push_back({node, o});
            continue;
        }
        const auto vector = type == mx::Type::Matrix33 ? mx::Type::Vector3 : mx::Type::Vector4;
        const std::array typed{mx::TypedInput{"in", vector}, mx::TypedInput{"mat", type}};
        const auto transform = *mx::FindDeclaration(lib, "transformmatrix", typed, vector);
        for (uint32_t c = 0; c < mx::Words(vector); ++c) {
            const auto n = uint32_t(nodes.size());
            nodes.push_back({.Op = transform, .Inputs = {{"mat", ops::Link{node, o}}}});
            std::vector<uint32_t> basis(mx::Words(vector), 0u);
            basis[c] = std::bit_cast<uint32_t>(1.f);
            literal(nodes.back(), n, *std::ranges::find(lib.Declarations[transform].Inputs, "in", &mx::Port::Name), std::move(basis));
            t.Graph.Outputs.push_back({n, 0u});
        }
    }
    return t;
}

// One evaluation of a test graph: its program on a domain, the literal words of each value set, and the per-point attribute and geometry words.
struct Case {
    uint32_t Declaration;
    Inputs Source;
    TestGraph Test;
    ops::Domain Domain;
    ops::Program Program;
    std::vector<std::vector<std::vector<uint32_t>>> Values; // Literal words by value set
    uint32_t Points;
};

// What the reference reads at each evaluation: the words of its vertex data and uniforms.
struct ReferenceInput {
    enum class Kind : uint8_t { Attribute,
                                Geometry,
                                Literal,
                                Constant };
    Kind Kind;
    std::string Name; // The attribute name
    uint32_t Index; // The first geometry probe word, or the literal
    uint32_t Width; // Geometry probe words
    std::vector<uint32_t> Words; // A constant's words
    uint32_t Count; // Words, zero past its source's words
};
struct ReferenceShader {
    std::string Source, Error;
    std::vector<ReferenceInput> Inputs;
    uint32_t InputWords{}, OutputWords{};
};

// The Metal expression of a value of the type from the words `w` at `offset`.
std::string Load(const ref::Syntax &syntax, ref::TypeDesc type, uint32_t offset) {
    if (type.getBaseType() == ref::TypeDesc::BASETYPE_BOOLEAN) return std::format("(w[{}] != 0u)", offset);
    if (type.getBaseType() == ref::TypeDesc::BASETYPE_INTEGER) return std::format("as_type<int>(w[{}])", offset);
    if (type.getSize() == 1) return std::format("as_type<float>(w[{}])", offset);
    std::string components;
    for (uint32_t c = 0; c < type.getSize(); ++c) components += std::format("{}as_type<float>(w[{}])", c ? ", " : "", offset + c);
    return std::format("{}({})", syntax.getTypeName(type), components);
}
// Statements writing the value's words to `oracle_out` from word `first`, matrices by column.
std::string Write(ref::TypeDesc type, const std::string &value, uint32_t first) {
    std::string code = std::format("{{ const auto oracle_value = {};\n", value);
    const auto size = uint32_t(type.getSize());
    const auto columns = size == 9u ? 3u : size == 16u ? 4u : 1u;
    for (uint32_t w = 0; w < size; ++w) {
        const auto component = size == 1u ? std::string{"oracle_value"} : columns > 1u ? std::format("oracle_value[{}][{}]", w / columns, w % columns) : std::format("oracle_value[{}]", w);
        code += type.getBaseType() == ref::TypeDesc::BASETYPE_BOOLEAN ? std::format("oracle_out[{}] = uint({});\n", first + w, component) : std::format("oracle_out[{}] = as_type<uint>({});\n", first + w, component);
    }
    return code + "}\n";
}

// The probe word of each geometric field the reference reads, in the geometry probe's output order.
std::optional<std::pair<uint32_t, uint32_t>> GeometryWords(std::string_view field) {
    if (field == "positionObject") return std::pair{0u, 3u};
    if (field == "normalObject") return std::pair{3u, 3u};
    if (field == "tangentObject") return std::pair{6u, 3u};
    if (field.starts_with("texcoord_") && field.size() == 10 && field[9] >= '0' && field[9] <= '3') return std::pair{9u + 2u * uint32_t(field[9] - '0'), 2u};
    if (field == "color_0") return std::pair{17u, 4u};
    return {};
}

// A program writing the point fields its driver fills in the order GeometryWords gives, in outputs of the probe's types.
constexpr std::array GeometryProbeOutputs{mx::Type::Vector3, mx::Type::Vector3, mx::Type::Vector3, mx::Type::Vector4, mx::Type::Vector4, mx::Type::Vector4};
std::string GeometryProbe() {
    return std::format(
        "#include <metal_stdlib>\nusing namespace metal;\n#ifndef OP_PROGRAM\n#define OP_PROGRAM\n{}#endif\n"
        "[[visible]] void ProgramFunction(const thread OpPoint &p, device const uint *, thread uint *out) {{\n"
        "    uint w = 0u;\n"
        "    for (uint c = 0u; c < 3u; ++c) out[w++] = as_type<uint>(p.Position[c]);\n"
        "    for (uint c = 0u; c < 3u; ++c) out[w++] = as_type<uint>(p.Normal[c]);\n"
        "    for (uint c = 0u; c < 3u; ++c) out[w++] = as_type<uint>(p.Tangent[c]);\n"
        "    for (int s = 0; s < 4; ++s) {{\n"
        "        const float2 uv = OpTexcoord(p, s);\n"
        "        out[w++] = as_type<uint>(uv.x);\n"
        "        out[w++] = as_type<uint>(uv.y);\n"
        "    }}\n"
        "    const float4 color = OpColor(p, 0);\n"
        "    for (uint c = 0u; c < 4u; ++c) out[w++] = as_type<uint>(color[c]);\n"
        "}}\n",
        msl::Load(Paths::Shaders(), "OpProgram.metal").Text
    );
}

// The generated pixel stage's directives up to its own definitions, which a library of several stages holds once.
constexpr std::string_view MetalPreamble{"#include <metal_stdlib>\n#include <simd/simd.h>\nusing namespace metal;\n"};

// MaterialX's Metal shader for the test graph, as one node of a graph definition whose literals are interface inputs.
// Its function `Evaluate` evaluates it at thread i from the reference input words at `InputWords * i`, and writes the outputs' words at `OutputWords * i`.
ReferenceShader GenerateReference(const mx::Library &lib, const TestGraph &t) {
    ReferenceShader result;
    const auto &library = ReferenceLibrary();
    auto doc = ref::createDocument();
    doc->setDataLibrary(library);
    const auto &graph = t.Graph;
    const auto output_type = [&](ops::Link link) { return lib.Declarations[std::get<uint32_t>(graph.Nodes[link.Node].Op)].Outputs[link.Output].Type; };
    const auto output_name = [&](size_t o) { return graph.Outputs.size() == 1 ? std::string{"out"} : std::format("out{}", o); };
    const auto definition = doc->addNodeDef("ND_oracle", graph.Outputs.size() == 1 ? std::string{mx::Name(output_type(graph.Outputs[0]))} : std::string{"multioutput"}, "oracle");
    if (graph.Outputs.size() > 1) {
        for (size_t o = 0; o < graph.Outputs.size(); ++o) definition->addOutput(output_name(o), std::string{mx::Name(output_type(graph.Outputs[o]))});
    }
    auto instance = doc->addNodeInstance(definition, "o");
    auto ng = doc->addNodeGraph("NG_oracle");
    ng->setNodeDefString("ND_oracle");
    const auto interface_name = [](const LiteralInput &l) { return std::format("i{}_{}", l.Node, l.Input); };
    std::vector<ref::NodePtr> nodes;
    for (uint32_t n = 0; n < graph.Nodes.size(); ++n) {
        const auto &node = graph.Nodes[n];
        const auto &d = lib.Declarations[std::get<uint32_t>(node.Op)];
        auto element = ng->addNodeInstance(library->getNodeDef(d.Name), std::format("n{}", n));
        for (const auto &[name, source] : node.Inputs) {
            const auto &port = *std::ranges::find(d.Inputs, name, &mx::Port::Name);
            auto input = element->addInput(name, std::string{mx::Name(port.Type)});
            if (const auto *link = std::get_if<ops::Link>(&source)) {
                input->setConnectedNode(nodes[link->Node]);
                const auto &upstream = lib.Declarations[std::get<uint32_t>(graph.Nodes[link->Node].Op)];
                if (upstream.Outputs.size() > 1) input->setOutputString(upstream.Outputs[link->Output].Name);
            } else if (const auto *image = std::get_if<ops::Image>(&source)) {
                const auto image_name = std::format("image{}", image->Slot);
                if (!definition->getInput(image_name)) {
                    definition->addInput(image_name, "filename");
                    instance->addInput(image_name, "filename");
                }
                input->setInterfaceName(image_name);
            } else if (const auto *literal = std::get_if<ops::Literal>(&source)) {
                input->setValueString(std::get<std::string>(literal->Value));
            } else {
                const auto &l = *std::ranges::find_if(t.Literals, [&](const LiteralInput &l) { return l.Node == n && d.Inputs[l.Input].Name == name; });
                input->setInterfaceName(interface_name(l));
            }
        }
        nodes.push_back(element);
    }
    // The generator replaces a graph-defined node whose first output comes from a constant node with the node's first input, so each output passes through a dot node.
    for (size_t o = 0; o < graph.Outputs.size(); ++o) {
        const auto [n, output] = graph.Outputs[o];
        const auto type_name = std::string{mx::Name(output_type(graph.Outputs[o]))};
        auto dot = ng->addNode("dot", std::format("d{}", o), type_name);
        auto in = dot->addInput("in", type_name);
        in->setConnectedNode(nodes[n]);
        const auto &d = lib.Declarations[std::get<uint32_t>(graph.Nodes[n].Op)];
        if (d.Outputs.size() > 1) in->setOutputString(d.Outputs[output].Name);
        ng->addOutput(output_name(o), type_name)->setConnectedNode(dot);
    }
    std::map<std::string, uint32_t> literal_indices;
    for (uint32_t l = 0; l < t.Literals.size(); ++l) {
        const auto &literal = t.Literals[l];
        const auto type_name = std::string{mx::Name(literal.Type)};
        const auto name = interface_name(literal);
        definition->addInput(name, type_name);
        instance->addInput(name, type_name);
        literal_indices.emplace("o/" + name, l);
    }

    // The generator writes graph literals in fixed notation at the current precision, which keeps every float exact at 60 decimals.
    const ref::ScopedFloatFormatting precision{ref::Value::FloatFormatFixed, 60};
    const auto generator = ref::MslShaderGenerator::create();
    ref::GenContext context{generator};
    // MeshEditor's images have their first row at the top, so texture coordinates flip.
    context.getOptions().fileTextureVerticalFlip = true;
    context.registerSourceCodeSearchPath(ref::FilePath{MESHEDITOR_SOURCE_DIR "/lib/materialx"});
    context.registerSourceCodeSearchPath(ref::FilePath{MESHEDITOR_SOURCE_DIR});
    ref::ShaderPtr shader;
    try {
        shader = generator->generate("Oracle", instance, context);
    } catch (const std::exception &e) {
        result.Error = e.what();
        return result;
    }
    const auto &stage = shader->getStage(ref::Stage::PIXEL);
    const auto &syntax = generator->getSyntax();
    std::string source = shader->getSourceCode(ref::Stage::PIXEL);
    const auto &outputs = stage.getOutputBlock(ref::HW::PIXEL_OUTPUTS).getName();

    std::string writes;
    const auto &shader_graph = shader->getGraph();
    for (size_t o = 0; o < shader_graph.numOutputSockets(); ++o) {
        const auto *socket = shader_graph.getOutputSocket(o);
        const auto value = socket->getConnection() ? socket->getConnection()->getVariable() : syntax.getValue(socket->getType(), *socket->getValue());
        writes += Write(socket->getType(), value, result.OutputWords);
        result.OutputWords += uint32_t(socket->getType().getSize());
    }
    const auto member = outputs + " FragmentMain()";
    const auto member_at = source.find(member);
    const auto return_at = source.find("return " + outputs + "{", member_at);
    const auto entry_at = source.find("fragment " + outputs + " FragmentMain(");
    const auto context_at = source.find("GlobalContext ctx", entry_at);
    if (member_at == std::string::npos || return_at == std::string::npos || entry_at == std::string::npos || context_at == std::string::npos) {
        result.Error = "unexpected pixel stage layout";
        return result;
    }
    const auto context_init = source.substr(context_at, source.find(';', context_at) + 1 - context_at);
    source.resize(entry_at);
    source.insert(return_at, writes);
    source.replace(member_at, member.size(), outputs + " FragmentMain(device uint *oracle_out)");

    std::string loads;
    const auto add_input = [&](ReferenceInput input, ref::TypeDesc type, const std::string &target) {
        loads += std::format("    {} = {};\n", target, Load(syntax, type, result.InputWords));
        input.Count = uint32_t(type.getSize());
        result.InputWords += input.Count;
        result.Inputs.push_back(std::move(input));
    };
    const auto &vertex_data = stage.getInputBlock(ref::HW::VERTEX_DATA);
    for (size_t i = 0; i < vertex_data.size(); ++i) {
        const auto *port = vertex_data[i];
        const auto &field = port->getVariable();
        const auto target = std::format("vd.{}", field);
        if (field.starts_with("i_geomprop_")) {
            add_input({.Kind = ReferenceInput::Kind::Attribute, .Name = field.substr(11)}, port->getType(), target);
        } else if (const auto words = GeometryWords(field)) {
            add_input({.Kind = ReferenceInput::Kind::Geometry, .Index = words->first, .Width = words->second}, port->getType(), target);
        } else if (field.starts_with("color_")) {
            // A mesh holds one color set, so points read zero for the others.
            add_input({.Kind = ReferenceInput::Kind::Constant, .Words = std::vector<uint32_t>(port->getType().getSize(), 0u)}, port->getType(), target);
        } else {
            result.Error = std::format("reads vertex data {}", field);
            return result;
        }
    }
    for (const auto &name : {ref::HW::PRIVATE_UNIFORMS, ref::HW::PUBLIC_UNIFORMS}) {
        const auto &block = stage.getUniformBlock(name);
        if (block.empty()) continue;
        loads += std::format("    {} {};\n", block.getName(), block.getInstance());
        for (size_t i = 0; i < block.size(); ++i) {
            const auto *port = block[i];
            const auto target = std::format("{}.{}", block.getInstance(), port->getVariable());
            // An image reads the test image.
            if (port->getType().getName() == "filename") {
                loads += std::format("    texture2d<float> {0}_tex = oracle_texture;\n    sampler {0}_sampler = oracle_sampler;\n", port->getVariable());
                continue;
            }
            if (const auto found = literal_indices.find(port->getPath()); found != literal_indices.end()) {
                add_input({.Kind = ReferenceInput::Kind::Literal, .Index = found->second}, port->getType(), target);
            } else if (port->getVariable().starts_with("u_geomprop_")) {
                add_input({.Kind = ReferenceInput::Kind::Attribute, .Name = port->getVariable().substr(11)}, port->getType(), target);
            } else if (port->getVariable() == "u_time" || port->getVariable() == "u_frame") {
                add_input({.Kind = ReferenceInput::Kind::Constant, .Words = {std::bit_cast<uint32_t>(port->getVariable() == "u_time" ? Time : Frame)}}, port->getType(), target);
            } else if (port->getValue()) {
                add_input({.Kind = ReferenceInput::Kind::Constant, .Words = std::get<std::vector<uint32_t>>(ReferenceValue(port->getType().getName(), port->getValue()))}, port->getType(), target);
            } else {
                result.Error = std::format("reads uniform {}", port->getVariable());
                return result;
            }
        }
    }
    const auto preamble = source.find(MetalPreamble);
    if (preamble == std::string::npos) {
        result.Error = "unexpected pixel stage directives";
        return result;
    }
    // A point has no neighbors, so the generator's screen-space derivatives are zero.
    for (const auto *derivative : {"dfdx(", "dfdy("}) {
        for (auto at = source.find(derivative); at != std::string::npos; at = source.find(derivative, at)) source.replace(at, 5, "oracle_zero(");
    }
    result.Source = std::format(
        "template<typename T> T oracle_zero(T x) {{ return x - x; }}\n"
        "{}inline void Evaluate(uint oracle_i, device const uint *oracle_in, device uint *oracle_results, texture2d<float> oracle_texture, sampler oracle_sampler) {{\n"
        "    device const uint *w = oracle_in + oracle_i * {}u;\n"
        "    VertexData vd;\n"
        "    vd.pos = float4(0.0f);\n"
        "{}    {}\n"
        "    ctx.FragmentMain(oracle_results + oracle_i * {}u);\n"
        "}}\n",
        source.substr(preamble + MetalPreamble.size()), result.InputWords, loads, context_init, result.OutputWords
    );
    return result;
}
} // namespace

const boost::ut::suite MaterialXOracle = [] {
    "every declaration states what the reference library's definition states"_test = [] {
        const auto &lib = mx::Libraries();
        const auto &library = ReferenceLibrary();
        auto names = lib.Declarations | std::views::transform(&mx::Declaration::Name) | std::ranges::to<std::vector>();
        auto reference_names = library->getNodeDefs() | std::views::transform([](const auto &d) { return d->getName(); }) | std::ranges::to<std::vector>();
        std::ranges::sort(names);
        std::ranges::sort(reference_names);
        expect(names == reference_names);
        for (const auto &d : lib.Declarations) {
            const auto reference = library->getNodeDef(d.Name);
            expect((reference && Statement(lib, d) == Statement(*reference))) << d.Name;
        }
    };

    "every signature resolves to the reference's definition, its own input types with no conversion and an input of a type that converts to its own through it"_test = [] {
        const auto &lib = mx::Libraries();
        auto doc = ref::createDocument();
        doc->setDataLibrary(ReferenceLibrary());
        const auto signatures = ConvertSignatures(lib);
        // The reference definition of the node with the inputs, each through MeshEditor's conversion when it has one.
        const auto reference = [&](const mx::Declaration &d, std::span<const mx::TypedInput> inputs, const ops::Resolution &resolution) {
            const auto &resolved = lib.Declarations[resolution.Declaration];
            auto node = doc->addNode(d.Node, "n", std::string{OutputTypeName(resolved)});
            bool conversions_match = true;
            for (size_t i = 0; i < inputs.size(); ++i) {
                const auto &port = *std::ranges::find(resolved.Inputs, inputs[i].Name, &mx::Port::Name);
                auto input = node->addInput(std::string{inputs[i].Name}, std::string{mx::Name(port.Type)});
                if (!resolution.Conversions[i]) continue;
                auto convert = doc->addNode("convert", std::format("c{}", i), std::string{mx::Name(port.Type)});
                convert->addInput("in", std::string{mx::Name(inputs[i].Type)});
                const auto convert_definition = convert->getNodeDef();
                conversions_match &= convert_definition && convert_definition->getName() == lib.Declarations[*resolution.Conversions[i]].Name;
                input->setConnectedNode(convert);
            }
            const auto definition = node->getNodeDef();
            const auto name = conversions_match && definition ? definition->getName() : std::string{};
            for (const auto &child : doc->getNodes()) doc->removeNode(child->getName());
            return name;
        };
        for (const auto &d : lib.Declarations) {
            const auto output = StoresOutput(lib, d) ? std::optional{d.Outputs.size() == 1 ? d.Outputs[0].Type : mx::Type::MultiOutput} : std::nullopt;
            const auto own = d.Inputs | std::views::transform([](const mx::Port &p) { return mx::TypedInput{p.Name, p.Type}; }) | std::ranges::to<std::vector>();
            // MeshEditor's definition for the inputs, when the reference resolves them through MeshEditor's conversions to it.
            const auto resolve = [&](std::span<const mx::TypedInput> inputs) -> std::optional<std::string> {
                const auto resolved = ops::Resolve(lib, d.Node, inputs, output);
                if (!resolved || reference(d, inputs, *resolved) != lib.Declarations[resolved->Declaration].Name) return {};
                return lib.Declarations[resolved->Declaration].Name;
            };
            expect(resolve(own) == d.Name && std::ranges::none_of(ops::Resolve(lib, d.Node, own, output)->Conversions, [](const auto &c) { return c.has_value(); })) << d.Name;
            for (size_t i = 0; i < own.size(); ++i) {
                for (const auto &[from, to] : signatures) {
                    if (to != own[i].Type) continue;
                    auto inputs = own;
                    inputs[i].Type = from;
                    expect(resolve(inputs).has_value()) << d.Name << own[i].Name << mx::Name(from);
                }
            }
        }
    };

    "every declaration MeshEditor compiles gives the reference generator's words, constant and varying"_test = [] {
        const auto &lib = mx::Libraries();
        // Both sides compile with safe math, since the oracle compares expressions and fast math lets the compiler evaluate two equivalent sources differently.
        OpsFixture f{OracleMesh(), MTL::MathModeSafe};
        mtl::ComputeChain chain{f.Buffers};
        // The test image of random texels, in the programs' image tables and bound to the reference shaders with the sampler their default address modes and filter choose.
        const auto image = mtl::CreateTexture2D(f.Context, MTL::PixelFormatRGBA32Float, {ImageSize, ImageSize}, MTL::TextureUsageShaderRead);
        {
            std::mt19937 rng{3u};
            std::vector<float> texels(4u * ImageSize * ImageSize);
            for (auto &texel : texels) texel = std::uniform_real_distribution{0.f, 1.f}(rng);
            mtl::Upload(image, 0u, std::as_bytes(std::span{texels}), ImageSize * 4u * sizeof(float));
        }
        const OpImageTexture image_texture{(*image)->gpuResourceID()._impl};
        // A reference shader samples with the sampler of the address modes and filter of the first image node its declaration's definition holds, as MaterialX's renderer does.
        std::map<std::array<std::string, 3>, NS::SharedPtr<MTL::SamplerState>> image_samplers;
        const auto image_sampler = [&](uint32_t declaration) {
            std::array<std::string, 3> modes{"periodic", "periodic", "linear"};
            const std::array<std::string_view, 3> slot{"uaddressmode", "vaddressmode", "filtertype"};
            for (const auto &node : mx::Flatten(lib, declaration).Nodes) {
                const auto &d = lib.Declarations[node.Declaration];
                if (d.Node != "image") continue;
                for (uint32_t i = 0; i < d.Inputs.size(); ++i) {
                    const auto *text = std::holds_alternative<mx::Value>(node.Inputs[i]) ? std::get_if<std::string>(&std::get<mx::Value>(node.Inputs[i])) : nullptr;
                    if (const auto at = std::ranges::find(slot, std::string_view{d.Inputs[i].Name}); text && at != slot.end()) modes[at - slot.begin()] = *text;
                }
                break;
            }
            auto &sampler = image_samplers[modes];
            if (!sampler) {
                const auto address = [](std::string_view mode) { return mode == "periodic" ? MTL::SamplerAddressModeRepeat : mode == "mirror" ? MTL::SamplerAddressModeMirrorRepeat : MTL::SamplerAddressModeClampToEdge; };
                const auto filter = modes[2] == "closest" ? MTL::SamplerMinMagFilterNearest : MTL::SamplerMinMagFilterLinear;
                sampler = mtl::CreateSampler(f.Context, {.MinFilter = filter, .MagFilter = filter, .MipFilter = modes[2] == "closest" ? MTL::SamplerMipFilterNearest : MTL::SamplerMipFilterLinear, .AddressS = address(modes[0]), .AddressT = address(modes[1])});
            }
            return sampler.get();
        };

        // Named attributes of random words, by name, created on first use.
        std::map<std::string, std::vector<std::vector<uint32_t>>, std::less<>> attributes;
        std::mt19937 attribute_rng{11u};
        const auto attribute = [&](ops::Domain domain, AttributeType storage, uint32_t k) {
            auto name = std::format("a{}_{}_{}", uint32_t(domain), uint32_t(storage), k);
            if (domain == ops::Domain::Point || attributes.contains(name)) return name;
            const auto id = f.Store().AddAttribute(f.Id, ops::AttributeDomainOf(domain), storage, name).value();
            auto &values = attributes[name];
            for (uint32_t handle = 0; handle < f.Count(domain); ++handle) {
                values.push_back(RandomWords(TypeOf(storage), attribute_rng));
                test::SetAttributeWords(f.Store(), id, handle, values.back());
            }
            return name;
        };

        std::vector<Case> cases;
        for (uint32_t d = 0; d < lib.Declarations.size(); ++d) {
            if (!HasValues(lib.Declarations[d])) continue;
            std::mt19937 rng{d};
            // Literals on the first domain they compile on, and varying inputs from geometry, or else from attributes alone.
            for (const auto &sources : {std::vector{Inputs::Literals}, std::vector{Inputs::Geometry, Inputs::Attributes}}) {
                std::vector<std::pair<Inputs, ops::Domain>> attempts;
                for (const auto source : sources) {
                    if (source == Inputs::Literals) attempts.emplace_back(source, ops::Domain::Point);
                    for (const auto domain : MeshDomains) attempts.emplace_back(source, domain);
                }
                for (const auto [source, domain] : attempts) {
                    auto test = DeclarationGraph(lib, d, source, domain, [&](AttributeType storage, uint32_t k) { return attribute(domain, storage, k); });
                    auto program = ops::Compile(lib, test.Graph, domain);
                    // A split source reads an attribute this harness does not evaluate, so its case is the first domain that holds the source.
                    if (!program || std::ranges::any_of(program->Attributes, [](const ops::Attribute &a) { return a.Source.has_value(); })) continue;
                    // The graph compiles to the same program each time, so a literal edit keeps its program.
                    const auto again = ops::Compile(lib, test.Graph, domain);
                    expect(again && again->Source == program->Source) << lib.Declarations[d].Name;
                    const auto sets = source == Inputs::Literals ? ValueSets : 1u;
                    std::vector<std::vector<std::vector<uint32_t>>> values(sets);
                    for (auto &set : values) {
                        for (const auto &l : test.Literals) set.push_back(l.Fixed.value_or(RandomWords(l.Type, rng)));
                    }
                    cases.push_back({d, source, std::move(test), domain, std::move(*program), std::move(values), domain == ops::Domain::Point ? 1u : f.Count(domain)});
                    break;
                }
            }
        }

        // The point fields each mesh domain's driver fills, read by a program that writes them all.
        std::array<uint32_t, 5> needs{};
        for (uint32_t d = 0; d < lib.Declarations.size(); ++d) {
            if (!std::holds_alternative<mx::Native>(lib.Implementations[d]) || !HasValue(lib.Declarations[d].Outputs[0].Type)) continue;
            for (const auto domain : MeshDomains) {
                if (const auto program = ops::Compile(lib, {.Nodes = {{.Op = d}}, .Outputs = {{0u, 0u}}}, domain)) needs[uint32_t(domain)] |= program->Needs & ~OpNeedsIndex;
            }
        }

        const auto probe_source = GeometryProbe();
        std::vector<std::string> sources{probe_source};
        for (const auto &c : cases) sources.push_back(c.Program.Source);
        const auto indices = mtl::AddPrograms(f.Programs, sources);

        std::array<std::vector<uint32_t>, 5> probe_attributes;
        for (const auto domain : MeshDomains) {
            const ops::Program probe{.Source = probe_source, .Needs = needs[uint32_t(domain)], .Outputs = {GeometryProbeOutputs.begin(), GeometryProbeOutputs.end()}};
            probe_attributes[uint32_t(domain)] = f.RecordElements(chain, domain, probe, indices[0], SlotOffset{});
        }
        // Each case's output words by value set, as point words or stored attributes.
        std::vector<std::vector<std::variant<Range, std::vector<uint32_t>>>> recorded(cases.size());
        for (size_t i = 0; i < cases.size(); ++i) {
            const auto &c = cases[i];
            for (const auto &set : c.Values) {
                std::vector<uint32_t> block(c.Test.Words);
                for (size_t l = 0; l < c.Test.Literals.size(); ++l) std::ranges::copy(set[l], block.begin() + c.Test.Literals[l].Offset);
                const auto values = f.Block(block);
                const std::vector images(c.Test.Images, image_texture);
                if (c.Domain == ops::Domain::Point) recorded[i].push_back(ops::RecordPoint(chain, f.Drivers, {c.Program, indices[i + 1]}, values, images, Frame, Time));
                else recorded[i].push_back(f.RecordElements(chain, c.Domain, c.Program, indices[i + 1], values, images));
            }
        }
        chain.Submit();
        std::array<std::vector<std::vector<uint32_t>>, 5> geometry;
        for (const auto domain : MeshDomains) geometry[uint32_t(domain)] = f.ElementWords(probe_attributes[uint32_t(domain)], domain);
        // Each case's output words at each evaluation, value set major.
        std::vector<std::vector<uint32_t>> mine(cases.size());
        for (size_t i = 0; i < cases.size(); ++i) {
            for (const auto &r : recorded[i]) {
                if (const auto *range = std::get_if<Range>(&r)) {
                    mine[i].append_range(chain.Scratch.Get(*range));
                } else {
                    for (const auto &element : f.ElementWords(std::get<std::vector<uint32_t>>(r), cases[i].Domain)) mine[i].append_range(element);
                }
            }
        }

        std::vector<ReferenceShader> shaders;
        for (const auto &c : cases) shaders.push_back(GenerateReference(lib, c.Test));
        const auto &ctx = f.Context;
        // Consecutive shaders share a library, one per core, each shader in its own namespace, and the library's kernel evaluates the shader it is given.
        const auto library_count = std::min(uint32_t(cases.size()), std::max(std::thread::hardware_concurrency(), 1u));
        const auto first = [&](uint32_t l) { return uint32_t(uint64_t(l) * cases.size() / library_count); };
        std::vector<NS::SharedPtr<MTL::ComputePipelineState>> pipelines(library_count);
        std::vector<uint32_t> library_of(cases.size());
        for (uint32_t l = 0; l < library_count; ++l) std::fill(library_of.begin() + first(l), library_of.begin() + first(l + 1), l);
        {
            const mtl::AutoreleaseScope pool;
            const auto compiler_descriptor = NS::TransferPtr(MTL4::CompilerDescriptor::alloc()->init());
            NS::Error *compiler_error = nullptr;
            const auto compiler = NS::TransferPtr(ctx.Device->newCompiler(compiler_descriptor.get(), &compiler_error));
            const auto options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
            options->setMathMode(MTL::MathModeSafe);
            ParallelFor(library_count, [&](uint32_t l) {
                const mtl::AutoreleaseScope library_pool;
                std::string text{"#define __METAL__\n"}, cases_text;
                text += MetalPreamble;
                for (auto i = first(l); i < first(l + 1); ++i) {
                    if (!shaders[i].Error.empty()) continue;
                    text += std::format("namespace oracle{} {{\n{}}}\n", i, shaders[i].Source);
                    cases_text += std::format("        case {0}u: oracle{0}::Evaluate(oracle_i, oracle_in, oracle_results, oracle_texture, oracle_sampler); break;\n", i);
                }
                text += std::format(
                    "kernel void OracleMain(uint oracle_i [[thread_position_in_grid]], device const uint *oracle_in [[buffer(0)]], device uint *oracle_results [[buffer(1)]], constant uint &oracle_case [[buffer(2)]], texture2d<float> oracle_texture [[texture(0)]], sampler oracle_sampler [[sampler(0)]]) {{\n"
                    "    switch (oracle_case) {{\n{}        default: break;\n    }}\n}}\n",
                    cases_text
                );
                const auto fail = [&](const NS::Error *error) {
                    for (auto i = first(l); i < first(l + 1); ++i) shaders[i].Error = error->localizedDescription()->utf8String();
                };
                NS::Error *error = nullptr;
                const auto library = NS::TransferPtr(ctx.Device->newLibrary(mtl::Str(text).get(), options.get(), &error));
                if (!library) return fail(error);
                const auto function = NS::TransferPtr(MTL4::LibraryFunctionDescriptor::alloc()->init());
                function->setName(mtl::Str("OracleMain").get());
                function->setLibrary(library.get());
                const auto descriptor = NS::TransferPtr(MTL4::ComputePipelineDescriptor::alloc()->init());
                descriptor->setComputeFunctionDescriptor(function.get());
                pipelines[l] = NS::TransferPtr(compiler->newComputePipelineState(descriptor.get(), nullptr, &error));
                if (!pipelines[l]) fail(error);
            });
        }

        std::vector<uint64_t> input_offsets(cases.size()), output_offsets(cases.size());
        uint64_t input_words = 0, output_words = 0;
        for (size_t i = 0; i < cases.size(); ++i) {
            const auto evaluations = uint64_t(cases[i].Values.size()) * cases[i].Points;
            input_offsets[i] = input_words;
            output_offsets[i] = output_words;
            input_words += evaluations * shaders[i].InputWords;
            output_words += evaluations * shaders[i].OutputWords;
        }
        const mtl::ResidentBuffer reference_inputs{ctx, std::max<uint64_t>(input_words, 1u) * 4u}, reference_outputs{ctx, std::max<uint64_t>(output_words, 1u) * 4u};
        auto *const in = reinterpret_cast<uint32_t *>(reference_inputs.Contents().data());
        const auto *const out = reinterpret_cast<const uint32_t *>(reference_outputs.Contents().data());
        auto command = ctx.Begin();
        {
            mtl::PassChain passes{command};
            const auto encoder = passes.BeginCompute("Oracle");
            command.SetTexture(*image, 0u);
            for (size_t i = 0; i < cases.size(); ++i) {
                const auto &c = cases[i];
                const auto &shader = shaders[i];
                if (!shader.Error.empty()) continue;
                auto *words = in + input_offsets[i];
                for (uint32_t set = 0; set < c.Values.size(); ++set) {
                    for (uint32_t point = 0; point < c.Points; ++point) {
                        for (const auto &input : shader.Inputs) {
                            const auto source = [&]() -> std::span<const uint32_t> {
                                switch (input.Kind) {
                                    case ReferenceInput::Kind::Attribute: return attributes.find(input.Name)->second[point];
                                    case ReferenceInput::Kind::Geometry: return std::span{geometry[uint32_t(c.Domain)][point]}.subspan(input.Index, input.Width);
                                    case ReferenceInput::Kind::Literal: return c.Values[set][input.Index];
                                    case ReferenceInput::Kind::Constant: return input.Words;
                                }
                                std::unreachable();
                            }();
                            std::fill_n(words, input.Count, 0u);
                            std::copy_n(source.begin(), std::min<size_t>(input.Count, source.size()), words);
                            words += input.Count;
                        }
                    }
                }
                encoder->setComputePipelineState(pipelines[library_of[i]].get());
                command.SetAddress(reference_inputs.Address() + input_offsets[i] * 4u, 0u);
                command.SetAddress(reference_outputs.Address() + output_offsets[i] * 4u, 1u);
                command.SetBytes(uint32_t(i), 2u);
                command.SetSampler(image_sampler(c.Declaration), 0u);
                const auto threads = uint32_t(c.Values.size()) * c.Points;
                encoder->dispatchThreads(MTL::Size{threads, 1u, 1u}, MTL::Size{std::min(threads, 64u), 1u, 1u});
            }
        }
        expect(ctx.Run(std::move(command)).Error.empty());

        for (uint32_t d = 0; d < lib.Declarations.size(); ++d) {
            const auto &declaration = lib.Declarations[d];
            if (!HasValues(declaration)) continue;
            const auto compared = std::ranges::count_if(std::views::iota(0uz, cases.size()), [&](size_t i) { return cases[i].Declaration == d && shaders[i].Error.empty(); });
            expect(compared == (IsSkipped(declaration.Node) ? 0 : 2)) << declaration.Name;
        }
        for (size_t i = 0; i < cases.size(); ++i) {
            const auto &c = cases[i];
            const auto &d = lib.Declarations[c.Declaration];
            const auto variant = InputsNames[uint32_t(c.Source)];
            if (IsSkipped(d.Node)) continue;
            expect(eq(shaders[i].Error, std::string{})) << d.Name << variant;
            const auto theirs = std::span{out + output_offsets[i], c.Values.size() * c.Points * shaders[i].OutputWords};
            expect(std::ranges::equal(mine[i], theirs)) << d.Name << variant;
        }
    };
};
