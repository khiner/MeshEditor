#include "action/ActionDrain.h"
#include "action/Emit.h"

using namespace action;

namespace {
std::optional<std::pair<Action, Phase>> Emitted; // This frame's first emitted action and its phase
bool CommitRequested = false; // Standalone commit request

// First action emitted in the frame wins; the rest are ignored.
template<typename ActionType> void Buffer(ActionType a, Phase phase) {
    if (!Emitted) Emitted.emplace(Action{std::move(a)}, phase);
}
} // namespace

namespace action {
template<typename ActionType> void Emit(ActionType a) { Buffer(std::move(a), Phase::Record); }
template<typename ActionType> void EmitStaged(ActionType a) { Buffer(std::move(a), Phase::Stage); }
template<typename ActionType> void EmitCancel(ActionType a) { Buffer(std::move(a), Phase::Cancel); }
void Commit() { CommitRequested = true; }

size_t ActionSize() { return sizeof(Action); }

Drained Drain() {
    Drained drained{std::move(Emitted), CommitRequested};
    Emitted.reset();
    CommitRequested = false;
    return drained;
}
} // namespace action

namespace {
// Force instantiation of every Emit* entry point for every action type so call sites in other TUs link.
using EmitPtr = void (*)();
template<size_t... I>
std::array<EmitPtr, 3 * sizeof...(I)> AllEmits(std::index_sequence<I...>) {
    const auto inst = [](auto fn) { return reinterpret_cast<EmitPtr>(fn); };
    return {
        inst(static_cast<void (*)(std::variant_alternative_t<I, Action>)>(&Emit))...,
        inst(static_cast<void (*)(std::variant_alternative_t<I, Action>)>(&EmitStaged))...,
        inst(static_cast<void (*)(std::variant_alternative_t<I, Action>)>(&EmitCancel))...,
    };
}
const auto _ = AllEmits(std::make_index_sequence<std::variant_size_v<Action>>{});
} // namespace
