#include "action/ActionDrain.h"
#include "action/Errors.h"
#include "state/Registry.h"

using namespace action;

namespace {
std::optional<std::pair<std::vector<Action>, Phase>> Emitted;
bool CommitRequested = false, CancelRequested = false;
} // namespace

namespace action {
template<typename ActionType> void Emit(ActionType a, Phase phase) { Emit(MakeAction(std::move(a)), phase); }
void Emit(Action a, Phase phase) {
    std::vector<Action> actions;
    actions.push_back(std::move(a));
    Emit(std::move(actions), phase);
}
void Emit(std::vector<Action> actions, Phase phase) {
    if (!Emitted && !actions.empty()) Emitted.emplace(std::move(actions), phase);
}
void Commit() { CommitRequested = true; }
void Cancel() { CancelRequested = true; }
void Fail(state::Registry &r, std::string message, bool unsupported_extensions) {
    auto &errors = r.Resources.get<Errors>();
    errors.OnlyUnsupportedExtensions = unsupported_extensions && (errors.Messages.empty() || errors.OnlyUnsupportedExtensions);
    errors.Messages.push_back(std::move(message));
}

Drained Drain() {
    return {std::exchange(Emitted, {}), std::exchange(CommitRequested, false), std::exchange(CancelRequested, false)};
}
} // namespace action

namespace {
using EmitPtr = void (*)();
template<typename DV> constexpr auto DomainEmits() {
    return []<size_t... I>(std::index_sequence<I...>) {
        const auto inst = [](auto fn) { return reinterpret_cast<EmitPtr>(fn); };
        return std::array<EmitPtr, sizeof...(I)>{
            inst(static_cast<void (*)(std::variant_alternative_t<I, DV>, Phase)>(&Emit))...,
        };
    }(std::make_index_sequence<std::variant_size_v<DV>>{});
}
const auto _ = MapDomains([]<typename DV>() { return DomainEmits<DV>(); });
} // namespace
