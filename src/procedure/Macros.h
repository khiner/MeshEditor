#pragma once

#include "state/Entity.h"

#include <array>

namespace state {
struct Registry;
}

namespace procedure {
// A built-in subnet: its name, and the nodes, links and ports Define builds inside a new empty subnet of the mesh entity's procedure.
struct Macro {
    const char *Name;
    void (*Define)(state::Registry &, state::Entity mesh_entity, state::Entity subnet);
};
// Indices into Macros.
enum MacroIndex : uint16_t { SplitBySolidMacro,
                             HullOfObjectsMacro,
                             LoopCutMacro,
                             LoopCutAndSlideMacro,
                             FillHolesMacro,
                             BisectFillMacro,
                             CrossSectionMacro,
                             SymmetrizeMacro,
                             MacroCount };
extern const std::array<Macro, MacroCount> Macros;
} // namespace procedure
