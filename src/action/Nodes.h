#pragma once

#include "mesh/GeometrySelection.h"
#include "state/Entity.h"

namespace state {
struct Scene;
} // namespace state

// Address procedure nodes by index and update parameters on their entities.
namespace action::nodes {
using numeric::uvec2;

// Creates the procedure with a Source node linked into an Output node.
struct AddProcedure {
    state::Entity MeshEntity;
};
// Append a default node at the grid cell with unlinked inputs and make it the active, sole selected node.
struct AddNode {
    state::Entity MeshEntity;
    uint16_t Kind;
    uvec2 Position;
};
// Links an output socket into an input socket that procedure::Links admits, replacing the input's link.
struct Link {
    state::Entity MeshEntity;
    uint32_t FromNode;
    uint16_t FromSocket;
    uint32_t ToNode;
    uint16_t ToSocket;
};
struct Unlink {
    state::Entity MeshEntity;
    uint32_t ToNode;
    uint16_t ToSocket;
};
// Makes an Output node's input the procedure's result.
struct SetOutput {
    state::Entity MeshEntity;
    uint32_t Node;
};

// Names starting with a period are reserved for internal attributes.
struct SetAttributeName {
    state::Entity Node;
    std::string Name;
};

// Supply vertex, edge, and face mask blocks in domain order and ascending owned-block order.
// An empty list selects every element.
struct SetInlineSelection {
    state::Entity Node;
    std::vector<SelectionBlock> Blocks;
    uint32_t Vertices, Edges; // The vertex and edge block counts, with the face blocks after them
};
// Makes the mesh's evaluated record its source and removes its procedure with its nodes.
struct ApplyProcedure {
    state::Entity MeshEntity;
};
// Appends a copy of each node, ascending, at its cell, keeping the links among the copied nodes, and selects the copies.
// The copy of the active node is active, or else the last copy.
struct DuplicateNodes {
    state::Entity MeshEntity;
    std::vector<uint32_t> Nodes;
};
// Replaces the procedure's node selection with the nodes, ascending, and makes Active the node the parameter panel edits.
// The selection is the procedure's own and leaves the entity selection alone.
struct SelectNodes {
    state::Entity MeshEntity;
    std::vector<uint32_t> Nodes;
    uint32_t Active{UINT32_MAX};
};

// Remove ascending node indices and reconnect through each removed chain.
// Use the unique linked input with the output socket type; disconnect ambiguous or incompatible inputs.
struct DeleteNodes {
    state::Entity MeshEntity;
    std::vector<uint32_t> Nodes;
};

// Insert the node between the input and its source using the first geometry sockets.
// Preserve an existing link on the inserted node input.
// Move the consumer and its descendants right by Shift cells.
struct InsertNode {
    state::Entity MeshEntity;
    uint32_t Node;
    uint32_t ToNode;
    uint16_t ToSocket;
    uint32_t Shift;
};

using Action = std::variant<AddProcedure, AddNode, Link, Unlink, SetOutput, SetAttributeName, SetInlineSelection, ApplyProcedure, SelectNodes, DuplicateNodes, DeleteNodes, InsertNode>;

void Apply(state::Scene &, state::Entity viewport, const Action &);

// The selection of the mesh's drawn record as an inline selection for the node, over the blocks the record owns.
SetInlineSelection CaptureSelection(const state::Scene &, state::Entity mesh_entity, state::Entity node);
// Gives `to` a copy of the procedure of `from` over new node entities holding copies of its nodes' components.
// The next settle cooks the copy.
void CopyProcedure(state::Scene &, state::Entity from, state::Entity to);
// The mesh entity whose procedure the editor edits: the active object's mesh, or the active mesh entity.
state::Entity EditedMesh(const state::Scene &);
// Append a default node before the procedure output and capture the drawn selection.
// Publish the resulting selection and return the node entity, or null on failure.
state::Entity AppendNode(state::Scene &, state::Entity viewport, state::Entity mesh_entity, uint16_t kind);
// Visits each edit-mode mesh with a selection in the edit element domain and the node its procedure result reads.
void ForEachResultNode(const state::Scene &, state::Entity viewport, const std::function<void(state::Entity mesh_entity, state::Entity node)> &);
} // namespace action::nodes
