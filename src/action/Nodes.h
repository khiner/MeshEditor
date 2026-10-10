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
// Appends a node of the kind with its default parameters and unlinked inputs.
struct AddNode {
    state::Entity MeshEntity;
    uint16_t Kind;
    vec2 Position;
};
// Links an output socket into an input socket of the same type sharing an attribute type, or a bool attribute into a selection, replacing the input's link.
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

using Action = std::variant<AddProcedure, AddNode, Link, Unlink, SetOutput, SetAttributeName, SetInlineSelection, ApplyProcedure>;

void Apply(state::Scene &, state::Entity viewport, const Action &);

// Appends a node of the kind with its default parameters to the mesh's procedure, between its result and the node its result read.
// It selects what the mesh's drawn record selects and publishes the selection its edit emits, and the append applies as AddNode, SetInlineSelection and two Links.
// Returns the node's entity, or null after failing the action.
state::Entity AppendNode(state::Scene &, state::Entity viewport, state::Entity mesh_entity, uint16_t kind);
// Visits each edit-mode mesh with a selection in the edit element domain and the node its procedure result reads.
void ForEachResultNode(const state::Scene &, state::Entity viewport, const std::function<void(state::Entity mesh_entity, state::Entity node)> &);
} // namespace action::nodes
