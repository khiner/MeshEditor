#include "action/ActionDrain.h"
#include "action/Emit.h"

using namespace action;

namespace {
std::optional<std::pair<Action, Phase>> Emitted;
std::vector<Action> SystemEmitted;
bool CommitRequested = false;

// Retains the first user action emitted during the frame.
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
// Explicit instantiation provides definitions to other translation units.
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
