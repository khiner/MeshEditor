#pragma once

#include "state/Entity.h"

#include <array>
#include <variant>

namespace state {
struct Registry;
}

namespace graph {
enum class KindGroup : uint8_t;
// A built-in subgraph: its name, the group the editor files it under, its help, and the nodes, links and ports Define builds in an empty subgraph of the registry of built-in subgraphs.
struct BuiltinSubgraphSpec {
    const char *Name;
    KindGroup Group;
    const char *Help;
    void (*Define)(state::Registry &, state::Entity subgraph);
};
// Which built-in subgraph, as its index in BuiltinSubgraphs.
enum class BuiltinSubgraphId : uint16_t { SplitBySolid,
                                          HullOfObjects,
                                          LoopCut,
                                          LoopCutAndSlide,
                                          FillHoles,
                                          BisectFill,
                                          CrossSection,
                                          Symmetrize };
inline constexpr size_t BuiltinSubgraphCount = size_t(BuiltinSubgraphId::Symmetrize) + 1u;
extern const std::array<BuiltinSubgraphSpec, BuiltinSubgraphCount> BuiltinSubgraphs;
// Identifies a subgraph: a user subgraph's entity in the document, or a built-in subgraph.
using SubgraphId = std::variant<state::Entity, BuiltinSubgraphId>;
} // namespace graph
