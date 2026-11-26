#pragma once

#include "materialx/Library.h"

namespace mx {
struct FlatNode {
    uint32_t Declaration;
    std::vector<Source> Inputs; // By declaration input index
};

// A declaration's definition with every graph definition inlined, so no node's declaration has a graph implementation.
// Each node follows the nodes it reads, and interface inputs refer to the declaration's inputs.
// An unset input that reads a geometric property reads a node for that property.
struct FlatGraph {
    std::vector<FlatNode> Nodes;
    std::vector<Source> Outputs; // By declaration output index
};

FlatGraph Flatten(const Library &, uint32_t declaration);
} // namespace mx
