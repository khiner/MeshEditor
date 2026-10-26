#pragma once

#include "procedure/ProcedureSchema.h"

#include <cstdint>

namespace state {
struct Registry;
enum class Entity : uint32_t;
} // namespace state

// The procedure cooks the settle pass has run.
struct CookCount {
    uint64_t Value{};
};

// Binds the procedure change sets and removes a destroyed node from its procedure.
void RegisterProcedureHandlers(state::Registry &);
// Rebuilds the plans of changed procedures.
// With `cook`, each procedure whose graph, node parameters or inline selections changed then evaluates its result.
void SettleProcedures(state::Registry &, bool cook);
// Records the hash of the mesh storage state a settle pass that cooked one mesh alone ended with.
// The next cook pass restores the whole mesh storage to a kept state only while the storage still holds that state.
void FinishCookPass(state::Registry &);
// Keeps mesh storage as it stands as the state of each Source node with a pending state, which then holds that state's hash.
// Call where a history step ends.
void FinishSourceStates(state::Registry &);
// Ends a history step for the mesh storage states the Source nodes and edited inputs refer to: releases the kept versions of every other state, recording the ones the previous step referred to in the history.
// Call after a history step is recorded.
void EndMeshStates(state::Registry &);
// Records in the history every mesh storage state the Source nodes and edited inputs refer to.
// Call before the history's logs are read or copied.
void RecordMeshStates(state::Registry &);
// Makes the Source node's state pending again where a mesh draws its record without edits, so the step end keeps a write to the record outside the procedure.
void ReopenSourceState(state::Registry &, state::Entity mesh_entity);
// Forgets which nodes a history step's preview steps appended and which kept states restore the whole mesh storage exactly.
// Call where a history step ends.
void EndCookStep(state::Registry &);
