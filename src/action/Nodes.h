#pragma once

#include "mesh/GeometrySelection.h"
#include "numeric/vec2.h"
#include "state/Entity.h"

#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace state {
struct Scene;
} // namespace state

// Geometry procedure editing on a mesh entity, with nodes named by their index in the procedure.
// Node parameters edit through Update on the node entity, and DestroyEntity removes a node.
namespace action::nodes {
using numeric::vec2;

// Creates the procedure with a Source node linked into an Output node.
struct AddProcedure {
    state::Entity MeshEntity;
};
// Appends a node of the kind with its default parameters and unlinked inputs, and makes it the one selected node and the active node.
struct AddNode {
    state::Entity MeshEntity;
    uint16_t Kind;
    vec2 Position;
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

// Names the attribute a Named Attribute node reads or a Store Named Attribute node writes.
// A name starting with '.' is reserved for the cook's own attributes.
struct SetAttributeName {
    state::Entity Node;
    std::string Name;
};

// Sets a node's inline selection to the vertex, edge and face mask blocks in turn, each domain's blocks over the blocks its input record owns in ascending order.
// No blocks select every element.
struct SetInlineSelection {
    state::Entity Node;
    std::vector<SelectionBlock> Blocks;
    uint32_t Vertices, Edges; // The vertex and edge block counts, with the face blocks after them
};
// Makes the mesh's evaluated record its source and removes its procedure with its nodes.
struct ApplyProcedure {
    state::Entity MeshEntity;
};
// Appends a copy of each node, ascending, moved by Offset, keeping the links among the copied nodes, and selects the copies.
// The copy of the active node is active, or else the last copy.
struct DuplicateNodes {
    state::Entity MeshEntity;
    std::vector<uint32_t> Nodes;
    vec2 Offset;
};
// Replaces the procedure's node selection with the nodes, ascending, and makes Active the node the parameter panel edits.
// The selection is the procedure's own and leaves the entity selection alone.
struct SelectNodes {
    state::Entity MeshEntity;
    std::vector<uint32_t> Nodes;
    uint32_t Active{UINT32_MAX};
};

// Removes each node, ascending, and heals each chain through them.
// A remaining node's input reading a removed node's output takes the source of that node's one linked input of the output's socket type, and otherwise unlinks.
// A run of removed nodes heals as one, and an input whose new source does not link into it unlinks.
struct DeleteNodes {
    state::Entity MeshEntity;
    std::vector<uint32_t> Nodes;
};

// Inserts a node on the link into the input: the input reads the node's first geometry output, and the node's first geometry input reads the link's source unless it is already linked.
// The input's node and the nodes reading it, transitively, move right by Shift.
struct InsertNode {
    state::Entity MeshEntity;
    uint32_t Node;
    uint32_t ToNode;
    uint16_t ToSocket;
    float Shift;
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
// Appends a node of the kind with its default parameters to the mesh's procedure, between its result and the node its result read.
// It selects what the mesh's drawn record selects and publishes the selection its edit emits, and the append applies as AddNode, SetInlineSelection and two Links.
// Returns the node's entity, or null after failing the action.
state::Entity AppendNode(state::Scene &, state::Entity viewport, state::Entity mesh_entity, uint16_t kind);
// Visits each edit-mode mesh with a selection in the edit element domain and the node its procedure result reads.
void ForEachResultNode(const state::Scene &, state::Entity viewport, const std::function<void(state::Entity mesh_entity, state::Entity node)> &);
} // namespace action::nodes
