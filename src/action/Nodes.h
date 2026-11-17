#pragma once

#include "action/Emit.h"
#include "gpu/Transform.h"
#include "procedure/InlineSelection.h"
#include "procedure/Network.h"
#include "state/Entity.h"

namespace action::nodes {
using numeric::uvec2;

// Append a default node with unlinked inputs and make it the active, sole selected node.
// A Source node holds an empty record.
// Operation, when set, is the starting value of the kind's Operation parameter.
struct AddNode {
    state::Entity MeshEntity;
    uint16_t Kind;
    state::Entity Parent{state::Null};
    std::optional<uint32_t> Operation{};
};
// Append a named subnet built from procedure::Macros and make it the active, sole selected node.
struct AddMacro {
    state::Entity MeshEntity;
    uint16_t Macro;
    state::Entity Parent{state::Null};
};
// Append a macro's subnet reading the displayed node's result, display it, capture the drawn selection into its selection port, and set its value ports in order.
// A later step of a stage sets the values of the subnet its first step appended.
struct AppendMacro {
    uint16_t Macro;
    std::vector<std::array<uint32_t, 4>> Values{};
};
// Links an output socket into an input socket that procedure::Links admits, replacing the input's link.
// A multi-input socket keeps its links and inserts the new link at Index, or at the end.
// If it already has the source, that link moves to Index.
struct Link {
    state::Entity MeshEntity;
    state::Entity FromNode;
    uint16_t FromSocket;
    state::Entity ToNode;
    uint16_t ToSocket;
    uint16_t Index{UINT16_MAX};
};
// Removes the link at Index among the input socket's links.
struct Unlink {
    state::Entity MeshEntity;
    state::Entity ToNode;
    uint16_t ToSocket;
    uint16_t Index{};
};
// Displays the node's result, or no result when Node is null.
struct SetDisplay {
    state::Entity MeshEntity;
    state::Entity Node{state::Null};
    std::optional<uint16_t> Port;
};

struct GroupNodes {
    state::Entity MeshEntity;
    std::vector<state::Entity> Nodes;
};
struct UngroupNodes {
    state::Entity Subnet;
};
struct ExposeSocket {
    state::Entity Subnet;
    state::Entity Node;
    uint16_t Socket;
    std::string Name;
};
struct SetPortValue {
    state::Entity Subnet;
    uint16_t Port;
    std::array<uint32_t, 4> Words;
};
struct RenamePort {
    state::Entity Subnet;
    uint16_t Port;
    std::string Name;
};
struct MovePort {
    state::Entity Subnet;
    uint16_t Port;
    uint16_t Before;
};
struct RemovePort {
    state::Entity Subnet;
    uint16_t Port;
};
void Apply(state::Registry &, const AddMacro &);
void Apply(state::Registry &, const AppendMacro &);
void Apply(state::Registry &, const GroupNodes &);
void Apply(state::Registry &, const UngroupNodes &);
void Apply(state::Registry &, const ExposeSocket &);
void Apply(state::Registry &, const SetPortValue &);
void Apply(state::Registry &, const RenamePort &);
void Apply(state::Registry &, const RemovePort &);
void Apply(state::Registry &, const MovePort &);

// Selects nodes of a geometry procedure, through the procedure or one of its subnets.
// No active node keeps the procedure's active node.
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
// The document, a bone, and a hierarchy definition stay.
struct DeleteItems {
    std::vector<state::Entity> Items;
};
void Apply(state::Registry &, const DeleteItems &);
void Apply(state::Registry &, const GroupItems &);
void Apply(state::Registry &, const UngroupItems &);
void Apply(state::Registry &, const MoveItems &);
void Apply(state::Registry &, const RenameItem &);
struct NodePosition {
    state::Entity Node;
    uvec2 Position;
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
// Removes the node's placed cell from the network's page.
struct UnplaceNode {
    state::Entity Network;
    state::Entity Node;
};
struct SetBypass {
    state::Entity Node;
    bool Enabled;
};
struct EditSelection {
    state::Entity Node, Object;
};
struct FinishSelection {};
struct SetOperationHandle {
    state::Entity Node{state::Null}, Object{state::Null};
    procedure::HandleParameter Parameter{};
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
    state::Entity MeshEntity;
    std::vector<state::Entity> Nodes;
};
// Remove the distinct nodes and reconnect through each removed chain.
// Reconnect through the unique linked input of the output's socket type, and disconnect ambiguous or incompatible inputs.
// The display follows its node's chain the same way, or else its node's first input.
// The display node, or else the last node, replaces a removed active node.
struct DeleteNodes {
    state::Entity MeshEntity;
    std::vector<state::Entity> Nodes;
};

// Insert the node between the input socket's link at Index and that link's source.
// The node's first geometry input and output must be unlinked.
// Move the placed cells of the consumer and its descendants right by Shift cells.
struct InsertNode {
    state::Entity MeshEntity;
    state::Entity Node;
    state::Entity ToNode;
    uint16_t ToSocket;
    uint32_t Shift;
    uint16_t Index{};
};

using Action = std::variant<AddNode, AddMacro, AppendMacro, Link, Unlink, SetDisplay, SetBypass, EditSelection, FinishSelection, SetOperationHandle, SetExtrudeOffset, MoveNodes, AutoLayout, UnplaceNode, SelectItems, DeleteItems, GroupItems, UngroupItems, MoveItems, RenameItem, ExposePort, AssignReference, GroupNodes, UngroupNodes, ExposeSocket, SetPortValue, RenamePort, RemovePort, MovePort, SetAttributeName, SetInlineSelection, CollapseProcedure, DuplicateNodes, DeleteNodes, InsertNode>;

void Apply(state::Registry &, const Action &);

// Copies viewport selection changes into the node being edited by the active selection task.
void CaptureActiveSelection(state::Registry &, state::Entity mesh_entity);
// Creates a mesh entity whose procedure copies the one of `from` over new node entities, drawing `drawn`, a copy of the record `from` draws.
state::Entity CopyProcedure(state::Registry &, state::Entity from, uint32_t drawn);
// The mesh entity whose procedure the editor edits: the active object's mesh, or the active mesh entity.
state::Entity EditedMesh(const state::Registry &);
// Append a default node reading the displayed node's result, display it, and capture the drawn selection into a node with a selection input.
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
