#include "action/ActionDrain.h"
#include "action/Errors.h"
#include "state/Scene.h"

using namespace action;

namespace {
std::optional<std::pair<Action, Phase>> Emitted;
std::vector<Action> SystemEmitted;
bool CommitRequested = false, CancelRequested = false;
} // namespace

namespace action {
// Retains the first user action emitted during the frame.
template<typename ActionType> void Emit(ActionType a, Phase phase) {
    if (!Emitted) Emitted.emplace(MakeAction(std::move(a)), phase);
}
void Emit(Action a, Phase phase) {
    if (!Emitted) Emitted.emplace(std::move(a), phase);
}
template<typename ActionType> void EmitSystem(ActionType a) { SystemEmitted.emplace_back(MakeAction(std::move(a))); }
void Commit() { CommitRequested = true; }
void Cancel() { CancelRequested = true; }
void Fail(state::Scene &r, std::string message) { r.Context.get<Errors>().Messages.push_back(std::move(message)); }

Drained Drain() {
    return {std::exchange(Emitted, {}), std::exchange(SystemEmitted, {}), std::exchange(CommitRequested, false), std::exchange(CancelRequested, false)};
}
} // namespace action

namespace {
// Explicit instantiation provides definitions to other translation units.
using EmitPtr = void (*)();
template<typename DV> constexpr auto DomainEmits() {
    return []<size_t... I>(std::index_sequence<I...>) {
        const auto inst = [](auto fn) { return reinterpret_cast<EmitPtr>(fn); };
        return std::array<EmitPtr, 2 * sizeof...(I)>{
            inst(static_cast<void (*)(std::variant_alternative_t<I, DV>, Phase)>(&Emit))...,
            inst(static_cast<void (*)(std::variant_alternative_t<I, DV>)>(&EmitSystem))...,
        };
    }(std::make_index_sequence<std::variant_size_v<DV>>{});
}
const auto _ = MapDomains([]<typename DV>() { return DomainEmits<DV>(); });
} // namespace
