#pragma once

#include "state/Entity.h"

#include <array>

namespace state {
struct Registry;
}

namespace procedure {
// A built-in definition: its name, and the nodes, links and ports Define builds in an empty definition of the library registry.
struct Macro {
    const char *Name;
    void (*Define)(state::Registry &, state::Entity definition);
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
