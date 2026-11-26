#pragma once

#include "materialx/Xml.h"

#include <expected>
#include <filesystem>
#include <map>
#include <span>
#include <variant>
#include <vector>

// MaterialX node definitions: declarations, implementations and graph definitions.
namespace mx {
enum class Type : uint8_t {
    Boolean,
    Integer,
    Float,
    Vector2,
    Vector3,
    Vector4,
    Color3,
    Color4,
    Matrix33,
    Matrix44,
    String,
    Filename,
    SurfaceShader,
    DisplacementShader,
    VolumeShader,
    LightShader,
    Material,
    // The type of a node with several outputs.
    MultiOutput,
};
std::string_view Name(Type);
// The 32-bit words of a value of the type, or zero for a type with no Metal value.
uint32_t Words(Type);
// The Metal type of a value of the type, or empty for a type with no Metal value.
std::string_view MetalType(Type);

// A literal: the 32-bit words of its components in file order, or the text of a string or filename.
// A boolean is one word holding zero or one.
using Value = std::variant<std::vector<uint32_t>, std::string>;

struct Port {
    std::string Name;
    mx::Type Type;
    // The value an unconnected input takes, or zero when the file gives none.
    Value Default;
    bool Uniform;
    std::vector<std::string> EnumNames;
    std::vector<Value> EnumValues;
    // An unconnected input reads this geometric property instead of `Default`.
    std::optional<uint32_t> DefaultGeomProp; // Index into Library::GeomProps
    std::string UiName, Doc;
    std::optional<Value> UiMin, UiMax, UiSoftMin, UiSoftMax;
};

struct Declaration {
    std::string Name, Node, NodeGroup, Doc;
    std::vector<Port> Inputs, Outputs;
};

// An implementation as one expression, in which `{{input}}` stands for an input's value.
struct InlineExpression {
    std::string Source;
    std::vector<uint32_t> Definitions; // The math library definitions it names and the operators on its port types, as indices into Library::Definitions
};
// An implementation as a function taking the inputs in declaration order, then a reference for each output.
struct FunctionCall {
    std::string Function;
    std::vector<uint32_t> Definitions; // The function's overloads, as indices into Library::Definitions
};
// MeshEditor implements the declaration itself, since it reads the evaluation point, a texture, screen-space derivatives or a whole stored field, or passes a string, filename, shader or material.
// Function and its overloads in Definitions are the MaterialX function that MeshEditor's implementation calls, for a node that has one.
// Their textures are OpImageSampler values, and their texture coordinates have V up over images whose first row is the top.
// A function that takes derivatives takes each value parameter's values at the point's two neighbors after it, which give them.
struct Native {
    std::vector<uint32_t> Definitions{};
    std::string Function{};
};

struct NodeOutput {
    uint32_t Node, Output; // Node index in its graph and output index in the node's declaration
    bool operator==(const NodeOutput &) const = default;
};
struct InterfaceInput {
    uint32_t Input; // Input index in the graph's declaration
    bool operator==(const InterfaceInput &) const = default;
};
using Source = std::variant<Value, NodeOutput, InterfaceInput>;

struct GraphInput {
    uint32_t Input; // Input index in the node's declaration
    mx::Source Source;
    bool operator==(const GraphInput &) const = default;
};
struct GraphNode {
    std::string Name;
    uint32_t Declaration;
    std::vector<GraphInput> Inputs; // The inputs the graph sets, which leaves the rest at their defaults
    bool operator==(const GraphNode &) const = default;
};
// A graph definition in terms of other nodes.
struct Graph {
    std::string Name;
    std::vector<GraphNode> Nodes;
    std::vector<Source> Outputs; // By declaration output index
    bool operator==(const Graph &) const = default;
};

using Implementation = std::variant<Native, InlineExpression, FunctionCall, Graph>;

// A top-level function, structure, constant or macro of a function file ported to Metal.
struct Definition {
    std::string Text;
    std::vector<uint32_t> Uses; // The earlier definitions its text names, as indices into Library::Definitions
};

struct Library {
    std::vector<Declaration> Declarations;
    std::vector<Implementation> Implementations; // By declaration index
    std::map<std::string, std::vector<uint32_t>, std::less<>> DeclarationsByNode; // In load order
    std::map<std::string, uint32_t, std::less<>> DeclarationsByName;
    std::vector<GraphNode> GeomProps; // The node that reads each geometric property, named for the property
    std::vector<Definition> Definitions; // The math library's, then each function file's after those of the files it includes, each file's in text order
};

// Loads the declarations, graph definitions and GLSL implementations of each library folder.
// Fails when a declaration has no implementation, or a graph definition and another implementation.
std::expected<Library, std::string> Load(std::span<const std::filesystem::path> folders);

// MaterialX's stdlib, cmlib and nprlib with MeshEditor's own nodes, and any added folders, loaded on first use.
const Library &Libraries();
// Adds a folder Libraries() loads after MeshEditor's own nodes, when called before its first use.
// Tests add nodes the app does not ship.
void AddLibraryFolder(std::filesystem::path);

struct TypedInput {
    std::string_view Name;
    mx::Type Type;
};
std::span<const uint32_t> NodeDeclarations(const Library &, std::string_view node);
// Whether the declaration's outputs are `output`, or several for MultiOutput, or any when it is not given.
bool OutputsMatch(const Declaration &, std::optional<Type> output);
// The first declaration of `node` that has each given input with its type and, when given, the output type.
std::optional<uint32_t> FindDeclaration(const Library &, std::string_view node, std::span<const TypedInput> inputs, std::optional<Type> output = {});

// A material as a MaterialX document: a gltf_pbr surface shader whose inputs hold values or read the outputs of one node graph, and the surface material of that shader.
struct SurfaceInput {
    std::string Name;
    mx::Type Type;
    std::variant<Value, uint32_t> Source; // A value, or the graph output of that index
    // A graph output holding a normal in the surface's tangent frame with components in [-1, 1], which the input reads through a normalmap of it mapped to [0, 1].
    bool TangentNormal{};
    bool operator==(const SurfaceInput &) const = default;
};
struct MaterialDocument {
    std::string Name;
    mx::Graph Graph; // A node graph whose nodes name their declarations
    std::vector<std::pair<std::string, Type>> Outputs; // The name and type of each graph output
    std::vector<SurfaceInput> Surface; // The inputs the shader sets, which leaves the rest at their defaults
    bool operator==(const MaterialDocument &) const = default;
};
// The document of the material, holding before it each element of `definitions`.
std::string WriteMaterial(const Library &, const MaterialDocument &, std::span<const xml::Element> definitions = {});
// Reads the material of a document WriteMaterial writes, resolving its graph's nodes by declaration name in the library.
std::expected<MaterialDocument, std::string> ReadMaterial(const Library &, std::string_view text);
// The definitions a document needs for the declarations of MeshEditor's own nodes among `declarations`: each declaration, and the graph defining it with the definitions that graph needs.
// Fails for a declaration whose implementation is MeshEditor's own code, which other MaterialX readers lack.
std::expected<std::vector<xml::Element>, std::string> OwnDefinitions(const Library &, std::span<const uint32_t> declarations);
} // namespace mx
