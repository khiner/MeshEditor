#pragma once

#include "gpu/AttributeField.h"
#include "materialx/Library.h"

// Graphs of MaterialX ops compiled to one Metal program each.
namespace ops {
// Where a program evaluates: at one point, at every live element of a mesh record's domain, at every texel of a grid over a record's UV layout, at every texel of a grid over the unit square of texture coordinates, or at each surface point a material shades.
// A texel's point weights the corners of the triangle it lies on.
// A UV texel's point holds the texel's coordinates in every texture coordinate set, and nothing else of a mesh.
// A surface point weights the corners of the drawn triangle under a pixel, a fragment or a ray hit, in its object's space, and reads no stored field.
enum class Domain : uint8_t {
    Point,
    Vertex,
    Edge,
    Face,
    Corner,
    Texel,
    Uv,
    Surface,
};
using Domains = uint8_t; // A bit per domain
constexpr Domains DomainBit(Domain domain) { return Domains(1u << uint32_t(domain)); }
inline constexpr Domains AllDomains{0b11111111};
// The domains that store fields.
inline constexpr Domains MeshDomains = DomainBit(Domain::Vertex) | DomainBit(Domain::Edge) | DomainBit(Domain::Face) | DomainBit(Domain::Corner);
// Whether the domain's points have differentials: the points of the next texel along each axis, over which a lookup filters and a derivative differs.
constexpr bool Differentiable(Domain domain) { return domain == Domain::Texel || domain == Domain::Uv || domain == Domain::Surface; }
// Whether the domain's points read stored fields: the mesh domains', and a texel's through its corners.
constexpr bool ReadsFields(Domain domain) { return domain != Domain::Point && domain != Domain::Uv && domain != Domain::Surface; }
// Whether the domain's points weight the corners of a triangle: texels and surface points.
constexpr bool Weighted(Domain domain) { return domain == Domain::Texel || domain == Domain::Surface; }
// The domain whose stored fields a program on the domain reads at its own elements: corners for weighted points.
constexpr Domain StoringDomain(Domain domain) { return Weighted(domain) ? Domain::Corner : domain; }

// A literal of its own stored type, which the program's code holds.
struct Literal {
    mx::Type Type;
    mx::Value Value;
    bool operator==(const Literal &) const = default;
};
// A value of the type the program reads from its value block, from word `Offset`.
struct Word {
    mx::Type Type;
    uint32_t Offset;
    bool operator==(const Word &) const = default;
};
// Output `Output` of node `Node`.
struct Link {
    uint32_t Node, Output;
    bool operator==(const Link &) const = default;
};
// The image in slot `Slot` of the program's image table.
struct Image {
    uint32_t Slot;
    bool operator==(const Image &) const = default;
};
// An input's value: a literal, a word of the value block, a node's output, or an image.
using Source = std::variant<Literal, Word, Link, Image>;
struct Input {
    std::string Name;
    ops::Source Source;
    bool operator==(const Input &) const = default;
};
// An op names an mx node, resolved from its input types and stored output type, or a declaration.
// An input the node leaves out takes its geometric property, or else the declaration's default.
struct Node {
    std::variant<std::string, uint32_t> Op;
    std::optional<mx::Type> Output; // The stored output type of a node whose inputs do not determine it
    std::vector<Input> Inputs;
    bool operator==(const Node &) const = default;
};
// Each node links only to nodes before it.
struct Graph {
    std::vector<Node> Nodes;
    std::vector<Link> Outputs;
    Domain Colors{Domain::Corner}; // The domain the records the graph evaluates on hold their colors on
    bool operator==(const Graph &) const = default;
};

// A declaration of a node and, for each given input in order, the `convert` declaration that takes it to its port's type, or none for an input of that type.
struct Resolution {
    uint32_t Declaration;
    std::vector<std::optional<uint32_t>> Conversions;
};
// The declaration of `node` with the output type when given that takes the inputs: the first whose inputs are exactly them, or else the one with the fewest conversions, the first in declaration order on a tie.
std::optional<Resolution> Resolve(const mx::Library &, std::string_view node, std::span<const mx::TypedInput> inputs, std::optional<mx::Type> output = {});

// The domains the declaration evaluates on, for records that hold their colors on `colors`.
// A native source evaluates on the domains that store what it reads, and on corners through the corner's vertex or face when exactly one of them stores it.
// Texture coordinates also evaluate on UV texels, and the sources a surface point weights over its triangle's corners on surface points.
Domains Availability(const mx::Library &, uint32_t declaration, Domain colors = Domain::Corner);
// The domains that store what a native node reads, for records that hold their colors on `colors`.
Domains StoredDomains(const mx::Library &, uint32_t declaration, Domain colors = Domain::Corner);
// Whether a point of domain `to` holds the elements of domain `from` related to it, which a program reads a field of `from` at: a corner's vertex or face, or a weighted point's corners' vertices or face.
constexpr bool Through(Domain from, Domain to) { return (to == Domain::Corner || Weighted(to)) && (from == Domain::Vertex || from == Domain::Face); }
// The reduction converting a field of the type between domains where no node chooses one.
// A boolean is any of the related elements toward the vertices and edges bounding its elements, and all of them otherwise, and every other type is their mean.
DomainReduction DefaultReduction(AttributeType, Domain from, Domain to);

enum class Fault : uint8_t {
    Unresolved, // No declaration takes the node's inputs
    NoValue, // The node passes a string, image, shader or material
    Unavailable, // The node reads what the domain lacks
    Ambiguous, // A native source in or under the node reads what the domain lacks and several domains store
    Unstored, // The output has no named attribute type to store it on the domain
    OutputLimit, // The outputs exceed the program's output words
};
struct NodeFault {
    uint32_t Node;
    ops::Fault Fault;
};

// A native source the program's domain lacks, with what it reads, evaluated on the domain that stores it.
// Its nodes read the program's value block at the program's offsets, and its one output is the source's.
struct Split {
    Graph Graph;
    Domain Domain; // The one domain that stores the source
    std::vector<uint32_t> Nodes; // By split node, the graph node whose evaluation reads the source
    bool operator==(const Split &) const = default;
};
// A named attribute the program reads.
// A native source the domain lacks, in the graph or nested in a node, is an attribute holding its split evaluated on the domain that stores it, which the program reads converted to its own.
struct Attribute {
    std::string Name;
    mx::Type Type;
    std::optional<Split> Source{};
};
// One Metal function defining `ProgramFunction`, which the drivers call through their tables.
// The source depends only on the graph's shape, so graphs of the same shape share it.
struct Program {
    std::string Source;
    std::vector<Attribute> Attributes; // By attribute table slot
    uint32_t Needs; // OpNeeds bits of the point fields it reads
    std::vector<mx::Type> Outputs; // Written as consecutive words
};
std::expected<Program, std::vector<NodeFault>> Compile(const mx::Library &, const Graph &, Domain);

// The named attribute type that stores a value of the type, or none for a matrix or a type with no Metal value.
std::optional<AttributeType> StorageOf(mx::Type);
} // namespace ops
