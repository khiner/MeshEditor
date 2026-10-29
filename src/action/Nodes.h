#pragma once

#include "gpu/Transform.h"
#include "procedure/InlineSelection.h"
#include "procedure/Network.h"
#include "state/Entity.h"

// Address procedure nodes by entity and update parameters on their entities.
namespace action::nodes {
using numeric::uvec2;

// Append a default node at the grid cell with unlinked inputs and make it the active, sole selected node.
// A Source node holds an empty record.
struct AddNode {
    state::Entity MeshEntity;
    uint16_t Kind;
    uvec2 Position;
    state::Entity Parent{state::Null};
};
// Links an output socket into an input socket that procedure::Links admits, replacing the input's link.
struct Link {
    state::Entity MeshEntity;
    state::Entity FromNode;
    uint16_t FromSocket;
    state::Entity ToNode;
    uint16_t ToSocket;
};
struct Unlink {
    state::Entity MeshEntity;
    state::Entity ToNode;
    uint16_t ToSocket;
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
void Apply(state::Registry &, const GroupNodes &);
void Apply(state::Registry &, const UngroupNodes &);
void Apply(state::Registry &, const ExposeSocket &);
void Apply(state::Registry &, const SetPortValue &);
void Apply(state::Registry &, const RenamePort &);
void Apply(state::Registry &, const RemovePort &);
void Apply(state::Registry &, const MovePort &);

struct SelectItems {
    state::Entity Network;
    procedure::NetworkItem Active;
    std::vector<procedure::NetworkItem> Items;
};
struct ExposePort {
    state::Entity Subnet;
    procedure::PortBinding Binding;
    std::unique_ptr<std::string> Name;
};
struct AssignReference {
    procedure::SceneReference Reference;
    procedure::NetworkItem Source;
};
void Apply(state::Registry &, const ExposePort &);
void Apply(state::Registry &, const AssignReference &);
struct GroupItems {
    state::Entity Network;
    std::vector<procedure::NetworkItem> Items;
};
struct UngroupItems {
    state::Entity Subnet;
};
struct MoveItems {
    state::Entity Network;
    state::Entity Destination;
    std::vector<procedure::NetworkItem> Items;
};
struct RenameNetwork {
    state::Entity Network;
    std::string Name;
};
void Apply(state::Registry &, const GroupItems &);
void Apply(state::Registry &, const UngroupItems &);
void Apply(state::Registry &, const MoveItems &);
void Apply(state::Registry &, const RenameNetwork &);
struct NodePosition {
    procedure::NetworkItem Node;
    uvec2 Position;
};
struct MoveNodes {
    state::Entity Network;
    std::vector<NodePosition> Positions;
};
struct SetBypass {
    state::Entity Node;
    bool Enabled;
};
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

// Replaces the node's inline selection.
struct SetInlineSelection {
    state::Entity Node;
    std::unique_ptr<procedure::InlineSelection> Selection;
};
// Replaces the mesh's procedure with a displayed Source node holding its evaluated record.
struct CollapseProcedure {
    state::Entity MeshEntity;
};
// Appends a copy of each distinct node, in list order, at its cell, keeping the links among the copied nodes, and selects the copies.
// A Source node's copy holds a copy of its record.
// The copy of the active node is active, or else the last copy.
struct DuplicateNodes {
    state::Entity MeshEntity;
    std::vector<state::Entity> Nodes;
};
// Remove the distinct nodes and reconnect through each removed chain.
// Use the unique linked input with the output socket type; disconnect ambiguous or incompatible inputs.
// The display follows its node's chain the same way, or else its node's first input, and the display node, or else the last node, replaces a removed active node.
struct DeleteNodes {
    state::Entity MeshEntity;
    std::vector<state::Entity> Nodes;
};

// Insert the node between the input and its source using the first geometry sockets.
// Preserve an existing link on the inserted node input.
// Move the consumer and its descendants right by Shift cells.
struct InsertNode {
    state::Entity MeshEntity;
    state::Entity Node;
    state::Entity ToNode;
    uint16_t ToSocket;
    uint32_t Shift;
};

using Action = std::variant<AddNode, Link, Unlink, SetDisplay, SetBypass, SetOperationHandle, SetExtrudeOffset, MoveNodes, SelectItems, GroupItems, UngroupItems, MoveItems, RenameNetwork, ExposePort, AssignReference, GroupNodes, UngroupNodes, ExposeSocket, SetPortValue, RenamePort, RemovePort, MovePort, SetAttributeName, SetInlineSelection, CollapseProcedure, DuplicateNodes, DeleteNodes, InsertNode>;

void Apply(state::Registry &, const Action &);

// The selection of the mesh's drawn record in the viewport's edit domain as an inline selection for the node, by the record's handles, with the mesh's active element.
SetInlineSelection CaptureSelection(const state::Registry &, state::Entity mesh_entity, state::Entity node);
// Sets the inline selection of the mesh's viewport selection node, if any, to the selection of its drawn record.
// Call after each change of the viewport selection, so the viewport selection is the node's selection.
void CaptureActiveSelection(state::Registry &, state::Entity mesh_entity);
// Creates a mesh entity with a copy of the procedure of `from` over new node entities holding copies of its nodes' components, drawing `drawn`, a copy of the record `from` draws.
state::Entity CopyProcedure(state::Registry &, state::Entity from, uint32_t drawn);
// The mesh entity whose procedure the editor edits: the active object's mesh, or the active mesh entity.
state::Entity EditedMesh(const state::Registry &);
// Append a default node reading the displayed node's result, display it, and capture the drawn selection into a node with a selection input.
// The nodes reading the displayed node move right to make room.
// Publish the resulting selection and return the node entity, or null on failure.
state::Entity AppendNode(state::Registry &, state::Entity mesh_entity, uint16_t kind);
// The mesh's displayed node when a preview step of the current stage appended it with the kind, or null.
state::Entity StagedResultNode(const state::Registry &, state::Entity mesh_entity, uint16_t kind);
// Visits each edit-mode mesh with a selection in the edit element domain and its displayed node.
void ForEachResultNode(const state::Registry &, const std::function<void(state::Entity mesh_entity, state::Entity node)> &);
} // namespace action::nodes
