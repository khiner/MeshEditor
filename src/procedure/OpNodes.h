#pragma once

#include "gpu/OpPoint.h"
#include "ops/Compile.h"
#include "procedure/Procedure.h"

namespace MTL {
class Texture;
} // namespace MTL

namespace procedure {
// The data type of a value of the MaterialX type, or none for a string, filename, shader or material.
std::optional<DataType> DataTypeOf(mx::Type);
// The MaterialX type of a numeric or matrix data type.
mx::Type ValueTypeOf(DataType);

// Where a stored-field op's domains come from: its domain input, its source and target inputs, its node's selection domain, or a fixed domain.
enum class StoredDomainRule { Text,
                              SourceTarget,
                              Selection,
                              Vertex,
                              Edge };
// The job a stored-field op records from its field input into its output.
enum class StoredJob { Selection,
                       Curvature,
                       Boundary,
                       Convert,
                       Adjacent,
                       Components };
// A stored-field op's domain rule and job, with the adjacency of its components.
struct StoredOp {
    std::string_view Node;
    StoredDomainRule Domains;
    StoredJob Job;
    FieldAdjacency Adjacency{FieldAdjacency::Domain};
};
// The entry of a stored-field op, or null for another node.
const StoredOp *StoredOpOf(std::string_view node);

// What an op node takes from the declarations of its MaterialX node.
struct OpDefinition {
    std::vector<Socket> Sockets; // Each value input of any declaration, accepting every type a declaration gives it, then each output
    uint16_t Inputs{};
    std::vector<bool> ReadsUnset; // By input socket: an unset input reads a geometric property
    std::vector<std::pair<std::string_view, mx::Type>> Texts; // Each string input
    bool Source{}; // A declaration reads the evaluation point
    bool ReadsFrame{}, ReadsTime{}; // A declaration reads the timeline's frame, or its time
    ops::Domains Domains{}; // The domains some declaration with values evaluates on
    bool StoresOutput{}; // Two declarations take the same input types and differ in output type
    std::vector<mx::Type> OutputTypes; // The output type of each declaration of one output, in declaration order, once each
    bool Stored{}; // A stored-field op, whose value inputs and output are whole fields stored on domains
    bool Filter{}; // An image filter, which filters its first input's image, or its field of texture coordinates stored on a grid, and holds the result as an image
};
// The definition of the MaterialX node, or null for a name no declaration has.
const OpDefinition *DefinitionOf(std::string_view node);
std::optional<uint16_t> InputSocket(const OpDefinition &, std::string_view name);
// Whether the plan node is an op that programs read, which excludes a stored-field op.
// Programs read a filter's image.
bool IsInlineOp(const state::Registry &, const CookPlan &, uint32_t node);
bool IsStoredOp(const state::Registry &, const CookPlan &, uint32_t node);
// Whether the plan node is an image filter op, which a cook runs between programs and which programs read as an image.
bool IsFilterOp(const state::Registry &, const CookPlan &, uint32_t node);
// The text an op input holds: its set text, or else the declaration's default.
std::string_view TextOf(const Op &, std::string_view input, uint32_t declaration);

// An op input's link: none, a value that does not vary, or one that does.
enum class OpInput : uint8_t { Unset,
                               Single,
                               Varying };
// Whether an op node's outputs vary: it is a stored-field op, it reads the evaluation point, an unset input reads a geometric property, or a linked input varies.
// `input(s)` gives the link of each input socket.
bool OpVaries(const OpDefinition &definition, const Op &op, auto &&input) {
    if (definition.Stored || definition.Source) return true;
    for (uint16_t s = 0; s < definition.Inputs; ++s) {
        const auto link = input(s);
        if (link == OpInput::Varying || (link == OpInput::Unset && definition.ReadsUnset[s] && !op.Literals.contains(std::string_view{definition.Sockets[s].Name}))) return true;
    }
    return false;
}
bool OpVaries(const state::Registry &, state::Entity node);

// The declaration each op node of a plan resolves to, and its procedure's op value block.
// It depends on the plan's structure and its op nodes, so plans of one shape share it.
struct OpLayout {
    struct Node {
        std::optional<uint32_t> Declaration; // None for a node that faults, reads one that does, or is in a cycle
        std::vector<mx::Type> Outputs; // By output socket, the declaration's output of that name
        std::vector<uint32_t> Ports; // By output socket, the declaration output index, or NoNode
        uint32_t Values{}; // The word of the node's first slot
    };
    // A slot of a plan node's input.
    struct Slot {
        uint32_t Node;
        std::string Input;
        mx::Type Type;
        OpSlotKind Kind;
        uint32_t Offset; // In words from the block's start
    };
    std::vector<Node> Nodes; // By plan node
    std::vector<Slot> Slots; // By offset
    std::vector<std::vector<uint32_t>> Initial; // By slot, the words of a slot other than a literal
    uint32_t Words{};
};
OpLayout LayoutOf(const state::Registry &, const CookPlan &);
// The type of the value a plan node's output socket produces, or none for an op that faults or a socket with no numeric, matrix or image value.
std::optional<mx::Type> OutputType(const state::Registry &, const CookPlan &, const OpLayout &, uint32_t node, uint16_t socket);

const OpSlot *FindOpSlot(const OpValueBlock &, const NodePath &node, std::string_view input, mx::Type, OpSlotKind);
// Sets the op input to a literal of the type in its network's block, laying the block out again when the input held no literal of that type.
// A literal of a definition's op is also written into each block that holds it for an occurrence, and every mesh holding one cooks from it.
// In the library registry the words go to its literal store.
void WriteOpLiteral(state::Registry &, state::Entity node, std::string_view input, mx::Type, std::span<const uint32_t> words);
// The words of the op input's literal of the type: in its network's block of the op value store, or in the library registry's literal store.
// Empty for an input that holds no literal of the type.
std::span<const uint32_t> LiteralWords(const state::Registry &, state::Entity node, std::string_view input, mx::Type);
// Lays out the procedure's op value block for its plan's layout, keeping each set literal's value and writing every other slot's.
// Writes only the words that differ when the layout is unchanged.
void LayoutOpValues(state::Registry &, state::Entity mesh_entity, const CookPlan &, const OpLayout &);
// Lays out the definition's op value block with a slot for each set literal of its ops, keeping their values.
void LayoutDefinitionValues(state::Registry &, state::Entity definition);
// Lays out the block of a geometry procedure's or a definition's network.
void LayoutNetworkValues(state::Registry &, state::Entity network);
// The type and words of each set literal of an op node, by input name, from its network's block.
using OpLiteralWords = std::map<std::string, std::pair<mx::Type, std::vector<uint32_t>>, std::less<>>;
OpLiteralWords LiteralsOf(const state::Registry &, state::Entity node);
// Writes the words of each literal the op node also holds with the same type into its network's block, which lays them out.
void WriteLiterals(state::Registry &, state::Entity node, const OpLiteralWords &);
// Writes the literals each copied op node's original holds in `from` into the copy's network's block.
void CopyLiterals(state::Registry &, const state::Registry &from, const NodeCopies &);

// A read of a stored field by the graph node that reads it: an output of a node other than an op, or of a stored-field op.
struct OpRead {
    uint32_t Node; // The graph node that reads the field
    uint32_t From; // The plan node of the field
    uint16_t Socket;
};
// The image an image input of a graph node reads: an output of a node other than an op, or the document texture the input holds, or none.
struct OpImage {
    uint32_t From{NoNode}; // The plan node of an image output
    uint16_t Socket{};
    state::Entity Texture{state::Null};
};
// An image a node's output holds: its texture, or none, its texels along each axis, and the components of its values.
struct ImageValue {
    OpImageTexture Texture;
    uint32_t Width, Height, Components;
};
inline constexpr ImageValue NoImage{{}, 0u, 0u, 0u};
OpImageTexture ImageTextureOf(MTL::Texture *);
// The image of a document texture, whose values have four components, or none.
ImageValue DocumentImage(const state::Registry &, state::Entity texture);
// A node reading slot `image` of the program's image table as values of the type at its texture coordinates, clamped at the image's edges, and zero without an image.
ops::Node ImageRead(mx::Type, uint32_t image);

// Each graph node reads its values at its layout offset less Base, so programs of graphs whose values lay out alike from their first are the same.
// Images lists the image of each slot of the program's image table.
struct OpGraph {
    ops::Graph Graph;
    std::vector<uint32_t> Nodes; // The plan node of each graph node, which for a field read is the op node reading it
    std::vector<OpRead> Reads;
    std::vector<OpImage> Images;
    uint32_t Base{};
};
// The op nodes the plan's op outputs read, in plan order, and a read of each stored field they take.
// Each field read has an empty name, which the caller sets.
// A filter of a field that varies reads as its image, and the filters in `unheld`, which hold none, read as their inputs.
OpGraph GraphOf(const state::Registry &, const CookPlan &, const OpLayout &, std::span<const std::pair<uint32_t, uint16_t>> outputs, std::span<const uint32_t> unheld);
// Keeps each fault of the program the graph compiles to on the plan node of its graph node.
void KeepFaults(state::Registry &, const CookPlan &, const OpGraph &, std::span<const ops::NodeFault>);
// Whether the field a filter's first input reads lies over one set of texture coordinates: its texture coordinate nodes hold one index, which is zero where an input of it reads its default coordinates.
// A filter the field reads that is not in `unheld` reads only its own coordinates.
bool ReadsOneCoordinateSet(const state::Registry &, const CookPlan &, const OpLayout &, uint32_t node, std::span<const uint32_t> unheld);
// Makes the graph read its values relative to its first, so graphs whose values lay out alike from their first share a program, and returns the first's word.
uint32_t Rebase(ops::Graph &);
} // namespace procedure
