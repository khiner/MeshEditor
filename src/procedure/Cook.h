#pragma once

#include "procedure/ProcedureSchema.h"

#include <cstdint>

namespace state {
struct Registry;
} // namespace state

// The procedure cooks the settle pass has run.
struct CookCount {
    uint64_t Value{};
};

// Binds the procedure change sets and removes a destroyed node from its procedure.
void RegisterProcedureHandlers(state::Registry &);
// Rebuilds the plans of changed procedures.
// With `cook`, each procedure whose graph, source record, node parameters or inline selections changed then evaluates into a new output record.
void SettleProcedures(state::Registry &, bool cook);
