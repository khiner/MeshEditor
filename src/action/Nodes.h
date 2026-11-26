#pragma once

#include "action/Emit.h"
#include "gpu/Transform.h"
#include "procedure/InlineSelection.h"
#include "procedure/Network.h"
#include "state/Entity.h"

namespace mx {
enum class Type : uint8_t;
} // namespace mx

namespace action::nodes {
using numeric::uvec2;

// Network is a geometry procedure's mesh entity, a user definition, or a material with a node graph, whose network the node actions edit.
// A material's graph holds op nodes and its one Surface node.

// A default node of the kind, where a Source node holds an empty record and a Subnet node instances a new empty user definition.
// Operation, when set, is the starting value of the kind's Operation parameter.
struct KindNode {
    uint16_t Kind;
    std::optional<uint32_t> Operation{};
};
// An op node evaluating the MaterialX node, with unset inputs.
struct OpNode {
    std::string Node;
};
// A subnet node instancing the built-in definition procedure::Macros names.
struct MacroNode {
    uint16_t Macro;
};
// A subnet node instancing the user definition, which may not contain the network.
struct DefinitionNode {
    state::Entity Definition;
};
using NodeSpec = std::variant<KindNode, OpNode, MacroNode, DefinitionNode>;
// Append the node with unlinked inputs to the network and make it the active, sole selected node.
struct AddNode {
    state::Entity Network;
    NodeSpec Node;
};
// Append a subnet node of a macro reading the displayed node's result to the network that holds the displayed node, display it, capture the drawn selection into its selection port, publish its result's selection, and set its value ports in order.
// A later step of a stage sets the values of the subnet node its first step appended.
struct AppendMacro {
    uint16_t Macro;
    std::vector<std::array<uint32_t, 4>> Values{};
};
// Links an output socket into an input socket that procedure::Links admits, replacing the input's link.
// A multi-input socket keeps its links and inserts the new link at Index, or at the end.
// If it already has the source, that link moves to Index.
struct Link {
    state::Entity Network;
    state::Entity FromNode;
    uint16_t FromSocket;
    state::Entity ToNode;
    uint16_t ToSocket;
    uint16_t Index{UINT16_MAX};
};
// Removes the link at Index among the input socket's links.
struct Unlink {
    state::Entity Network;
    state::Entity ToNode;
    uint16_t ToSocket;
    uint16_t Index{};
};
// Displays the result of the node's occurrence in the subnet nodes of Path, or no result when Node is null.
struct SetDisplay {
    state::Entity MeshEntity;
    state::Entity Node{state::Null};
    std::optional<uint16_t> Port;
    std::vector<state::Entity> Path{};
};

// Moves the nodes into a new user definition and replaces them with one subnet node instancing it.
struct GroupNodes {
    state::Entity Network;
    std::vector<state::Entity> Nodes;
};
// Replaces the subnet node with a copy of its definition's nodes.
struct UngroupNodes {
    state::Entity Subnet;
};
// Makes the subnet node instance a new user definition copied from its definition.
struct MakeUnique {
    state::Entity Subnet;
};
// Deletes a user definition no subnet node instances.
struct DeleteDefinition {
    state::Entity Definition;
};
// Exposes a socket of a node of the user definition Subnet as a new port.
struct ExposeSocket {
    state::Entity Subnet;
    state::Entity Node;
    uint16_t Socket;
    std::string Name;
};
// Sets the value an unlinked port reads: a subnet node's own value, a definition's default, or a scene subnet's promoted field.
struct SetPortValue {
    state::Entity Subnet;
    uint16_t Port;
    std::array<uint32_t, 4> Words;
};
// Clears what an unlinked input port reads at the subnet: a subnet node's own value or selection, after which the port reads its definition's default or each reader's inline selection,
// or a definition's default, after which the port reads what the socket it exposes reads.
struct ClearPortValue {
    state::Entity Subnet;
    uint16_t Port;
};
struct RenamePort {
    state::Entity Subnet;
    uint16_t Port;
    std::string Name;
};
struct MovePort {
    state::Entity Subnet;
    uint16_t Port;
    uint16_t BeforePort;
};
struct RemovePort {
    state::Entity Subnet;
    uint16_t Port;
};
void Apply(state::Registry &, const AppendMacro &);
void Apply(state::Registry &, const GroupNodes &);
void Apply(state::Registry &, const UngroupNodes &);
void Apply(state::Registry &, const MakeUnique &);
void Apply(state::Registry &, const DeleteDefinition &);
void Apply(state::Registry &, const ExposeSocket &);
void Apply(state::Registry &, const SetPortValue &);
void Apply(state::Registry &, const ClearPortValue &);
void Apply(state::Registry &, const RenamePort &);
void Apply(state::Registry &, const RemovePort &);
void Apply(state::Registry &, const MovePort &);
// Whether a subnet node uses the definition.
bool DefinitionUsed(const state::Registry &, state::Entity definition);

// Selects nodes of a geometry network, where a definition's network also selects the definition, which its boundaries stand for.
// No active node keeps the network's active node.
struct SelectItems {
    state::Entity Network;
    state::Entity Active;
    std::vector<state::Entity> Items;
};
// Adds a named value port to the scene subnet for a field of an item inside it.
struct ExposePort {
    state::Entity Subnet;
    procedure::PromotedField Field;
    std::unique_ptr<std::string> Name;
};
// Sets the reference to the source, or clears it for no source.
// procedure::ReferenceAccepts says what each kind takes and what clearing it means.
struct AssignReference {
    procedure::SceneReference Reference;
    state::Entity Source;
};
void Apply(state::Registry &, const ExposePort &);
void Apply(state::Registry &, const AssignReference &);
struct GroupItems {
    state::Entity Network;
    std::vector<state::Entity> Items;
};
struct UngroupItems {
    state::Entity Subnet;
};
struct MoveItems {
    state::Entity Network;
    state::Entity Destination;
    std::vector<state::Entity> Items;
};
// Renames an object, scene, subnet, material, texture, or physics resource.
struct RenameItem {
    state::Entity Item;
    std::string Name;
};
// Deletes the items of a scene or document page, and the items inside each subnet among them.
// A deleted material leaves its slots with the default material, and a deleted texture leaves the material fields that held it empty.
// The document, a bone, a hierarchy definition and a subnet definition a subnet node uses stay.
struct DeleteItems {
    std::vector<state::Entity> Items;
};
void Apply(state::Registry &, const DeleteItems &);
void Apply(state::Registry &, const GroupItems &);
void Apply(state::Registry &, const UngroupItems &);
void Apply(state::Registry &, const MoveItems &);
void Apply(state::Registry &, const RenameItem &);
// Output places the output boundary of the definition whose page the network's page is.
struct NodePosition {
    state::Entity Node;
    uvec2 Position;
    bool Output{};
};
// Places each node at its grid cell on the network's page.
struct MoveNodes {
    state::Entity Network;
    std::vector<NodePosition> Positions;
};
// Removes every placed cell of the network's page, which leaves each node where the editor lays it out from the page's connections.
struct AutoLayout {
    state::Entity Network;
};
// Removes the node's placed cell from the network's page, or with Output the cell of the definition's output boundary.
struct UnplaceNode {
    state::Entity Network;
    state::Entity Node;
    bool Output{};
};
struct SetBypass {
    state::Entity Node;
    bool Enabled;
};
// Edits the inline selection of the node's occurrence in the subnet nodes of Path of the object's mesh's procedure over that occurrence's input.
// With Port, edits the own selection of that selection port of the subnet node Node over the geometry its first geometry input port reads.
struct EditSelection {
    state::Entity Node, Object;
    std::vector<state::Entity> Path{};
    std::optional<uint16_t> Port{};
};
struct FinishSelection {};
// Shows the viewport handle of the parameter of the node's occurrence in the subnet nodes of Path of the object's mesh's procedure, or no handle for a null node.
struct SetOperationHandle {
    state::Entity Node{state::Null}, Object{state::Null};
    procedure::HandleParameter Parameter{};
    std::vector<state::Entity> Path{};
};
struct SetExtrudeOffset {
    state::Entity Node;
    vec3 Value;
};

// Names starting with a period are reserved for internal attributes.
struct SetAttributeName {
    state::Entity Node;
    std::string Name;
};

struct SetInlineSelection {
    state::Entity Node;
    std::unique_ptr<procedure::InlineSelection> Selection;
};
// Replaces the mesh's procedure with a displayed Source node holding its evaluated record.
struct CollapseProcedure {
    state::Entity MeshEntity;
};
// Appends a copy of each distinct node, in list order, keeping the links among the copied nodes, and selects the copies.
// A Source node's copy holds a copy of its record.
// The copy of the active node is active, or else the last copy.
struct DuplicateNodes {
    state::Entity Network;
    std::vector<state::Entity> Nodes;
};
// Remove the distinct nodes and reconnect through each removed chain.
// Reconnect through the unique linked input whose source shares a data type with the output and varies as the output does.
// Disconnect ambiguous or incompatible inputs.
// The display follows its node's chain the same way, or else its node's first input.
// The display node, or else the last node, replaces a removed active node.
struct DeleteNodes {
    state::Entity Network;
    std::vector<state::Entity> Nodes;
};

// Insert the node between the input socket's link at Index and that link's source.
// The node's first geometry input and output must be unlinked.
// Move the placed cells of the consumer and its descendants right by Shift cells.
struct InsertNode {
    state::Entity Network;
    state::Entity Node;
    state::Entity ToNode;
    uint16_t ToSocket;
    uint32_t Shift;
    uint16_t Index{};
};

// A literal of a type the op input takes, whose words are its components in MaterialX file order.
struct OpLiteral {
    mx::Type Type;
    std::vector<uint32_t> Words;
};
// What an unlinked op input reads: a literal, a string input's text, or the document texture an image input reads, which a null texture clears.
using OpInputValue = std::variant<OpLiteral, std::string, state::Entity>;
// Sets the op's input of the name.
struct SetOpInput {
    state::Entity Node;
    std::string Input;
    std::unique_ptr<OpInputValue> Value;
};
// Sets the output type of an op whose input types do not determine it.
struct SetOpOutputType {
    state::Entity Node;
    mx::Type Type;
};
// Makes the op evaluate another MaterialX node, keeping the links, literals and texts of the inputs and outputs whose names it also has and whose types it takes.
struct ChangeOp {
    state::Entity Node;
    std::string Op;
};
// Gives the material a node graph holding a Surface node, whose unlinked inputs read the material's values, and makes the Surface node active.
struct AddMaterialGraph {
    state::Entity Material;
};
void Apply(state::Registry &, const SetOpInput &);
void Apply(state::Registry &, const SetOpOutputType &);
void Apply(state::Registry &, const ChangeOp &);

using Action = std::variant<AddNode, AppendMacro, Link, Unlink, SetDisplay, SetBypass, EditSelection, FinishSelection, SetOperationHandle, SetExtrudeOffset, MoveNodes, AutoLayout, UnplaceNode, SelectItems, DeleteItems, GroupItems, UngroupItems, MoveItems, RenameItem, ExposePort, AssignReference, GroupNodes, UngroupNodes, ExposeSocket, SetPortValue, RenamePort, RemovePort, MovePort, SetAttributeName, SetInlineSelection, CollapseProcedure, DuplicateNodes, DeleteNodes, InsertNode, SetOpInput, SetOpOutputType, ChangeOp, MakeUnique, DeleteDefinition, ClearPortValue, AddMaterialGraph>;

void Apply(state::Registry &, const Action &);

// Copies viewport selection changes into the node being edited by the active selection task.
void CaptureActiveSelection(state::Registry &, state::Entity mesh_entity);
// Creates a mesh entity whose procedure copies the one of `from` over new node entities, drawing `drawn`, a copy of the record `from` draws.
state::Entity CopyProcedure(state::Registry &, state::Entity from, uint32_t drawn);
// Append a default node reading the displayed node's result to the network that holds the displayed node, display it, and capture the drawn selection into a node with a selection input.
// Publish the resulting selection and return the node entity, or null on failure.
state::Entity AppendNode(state::Registry &, state::Entity mesh_entity, uint16_t kind);
// The mesh's displayed node when a preview step of the current stage appended it with the kind, or null.
state::Entity StagedResultNode(const state::Registry &, state::Entity mesh_entity, uint16_t kind);
// Visits the displayed node of each edit-mode mesh with a selection in the edit element domain.
void ForEachResultNode(const state::Registry &, const std::function<void(state::Entity node)> &);
} // namespace action::nodes
namespace action {
void Append(Destination, nodes::Action);
}
