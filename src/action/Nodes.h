#pragma once

#include "numeric/vec2.h"
#include "state/Entity.h"

#include <cstdint>
#include <string>
#include <variant>

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

using Action = std::variant<AddProcedure, AddNode, Link, Unlink, SetOutput, SetAttributeName>;

void Apply(state::Scene &, state::Entity viewport, const Action &);
} // namespace action::nodes
