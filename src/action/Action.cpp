#include "action/ActionDrain.h"
#include "action/Emit.h"

using namespace action;

namespace {
std::optional<std::pair<Action, Phase>> Emitted; // This frame's first emitted user action and its phase
std::vector<Action> SystemEmitted; // This frame's system-generated actions
bool CommitRequested = false; // Standalone commit request

// First user action emitted in the frame wins; the rest are ignored.
template<typename ActionType> void Buffer(ActionType a, Phase phase) {
    if (!Emitted) Emitted.emplace(MakeAction(std::move(a)), phase);
}
} // namespace

namespace action {
template<typename ActionType> void Emit(ActionType a) { Buffer(std::move(a), Phase::Record); }
template<typename ActionType> void EmitSystem(ActionType a) { SystemEmitted.emplace_back(MakeAction(std::move(a))); }
template<typename ActionType> void EmitStaged(ActionType a) { Buffer(std::move(a), Phase::Stage); }
template<typename ActionType> void EmitCancel(ActionType a) { Buffer(std::move(a), Phase::Cancel); }
void Commit() { CommitRequested = true; }

size_t ActionSize() { return sizeof(Action); }

Drained Drain() { return {std::exchange(Emitted, {}), std::exchange(SystemEmitted, {}), std::exchange(CommitRequested, false)}; }
} // namespace action

namespace {
// Force instantiation of every Emit* entry point for every leaf action type so call sites in other TUs link.
using EmitPtr = void (*)();
template<typename DV> constexpr auto DomainEmits() {
    return []<size_t... I>(std::index_sequence<I...>) {
        const auto inst = [](auto fn) { return reinterpret_cast<EmitPtr>(fn); };
        return std::array<EmitPtr, 4 * sizeof...(I)>{
            inst(static_cast<void (*)(std::variant_alternative_t<I, DV>)>(&Emit))...,
            inst(static_cast<void (*)(std::variant_alternative_t<I, DV>)>(&EmitSystem))...,
            inst(static_cast<void (*)(std::variant_alternative_t<I, DV>)>(&EmitStaged))...,
            inst(static_cast<void (*)(std::variant_alternative_t<I, DV>)>(&EmitCancel))...,
        };
    }(std::make_index_sequence<std::variant_size_v<DV>>{});
}
const auto _ = MapDomains([]<typename DV>() { return DomainEmits<DV>(); });
} // namespace
