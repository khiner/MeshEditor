#include "action/Action.h"
#include "action/Emit.h"
#include "action/Log.h"
#include "action/LogSerialize.h"

using namespace action;

namespace {
std::optional<Action> Staged;

std::optional<std::ofstream> LogStream;
std::optional<WriteBehindLog<Action>> Log;

// Apply a merged action by routing its alternative to the owning domain's `Apply`, then return the
// action re-wrapped (its payload was moved into the domain variant to apply, so move it back out).
Action ApplyAction(entt::registry &r, entt::entity viewport, Action &&action) {
    return std::visit(
        [&]<typename T>(T &&a) -> Action {
            using L = std::decay_t<T>;
            auto apply_keep = [&]<typename DomainV>() -> Action {
                DomainV dv{std::move(a)};
                Apply(r, viewport, dv);
                return Action{std::in_place_type<L>, std::move(std::get<L>(dv))};
            };
            if constexpr (is_variant_member_v<L, selection::Action>) return apply_keep.template operator()<selection::Action>();
            else if constexpr (is_variant_member_v<L, object::Action>) return apply_keep.template operator()<object::Action>();
            else if constexpr (is_variant_member_v<L, view::Action>) return apply_keep.template operator()<view::Action>();
            else if constexpr (is_variant_member_v<L, action::physics::Action>) return apply_keep.template operator()<action::physics::Action>();
            else if constexpr (is_variant_member_v<L, audio::Action>) return apply_keep.template operator()<audio::Action>();
            else if constexpr (is_variant_member_v<L, bone::Action>) return apply_keep.template operator()<bone::Action>();
            else if constexpr (is_variant_member_v<L, timeline::Action>) return apply_keep.template operator()<timeline::Action>();
            else if constexpr (is_variant_member_v<L, io::Action>) return apply_keep.template operator()<io::Action>();
            else return apply_keep.template operator()<Core>();
        },
        std::move(action)
    );
}
} // namespace

namespace action {
template<typename ActionType> void Emit(ActionType a) {
    if (!Staged) Staged.emplace(std::move(a)); // First action emitted in the frame wins.
}

void StartLog() {
    LogStream.emplace(OpenLogStream());
    Log.emplace(*LogStream, &SerializeAction);
}
void StopLog() {
    if (Log) Log->Stop();
    Log.reset();
    LogStream.reset(); // flush and close before checking the file on disk
    // The just-closed log is the newest file in the replay dir. Drop it if this session recorded nothing.
    if (auto logs = ListReplayLogs(); !logs.empty()) {
        std::error_code ec;
        if (std::filesystem::file_size(logs.front().Path, ec) == 0 && !ec) std::filesystem::remove(logs.front().Path, ec);
    }
}

void ApplyEmitted(entt::registry &r, entt::entity viewport) {
    if (!Staged) return;

    // Apply the action, then record it to the log unless it is a non-recordable action (e.g. a file save).
    auto recorded = ApplyAction(r, viewport, std::move(*Staged));
    if (Log && std::visit([]<typename T>(const T &) { return Recordable<T>; }, recorded)) Log->Enqueue(std::move(recorded));
    Staged.reset();
}

bool ReplayLog(entt::registry &r, entt::entity viewport, const std::filesystem::path &replay_path, ReplayTick tick) {
    std::ifstream in{replay_path, std::ios::binary};
    if (!in) return false;
    tick(r, viewport);
    StreamActions(in, [&](Action &&a) {
        ApplyAction(r, viewport, std::move(a));
        tick(r, viewport);
    });
    return true;
}

std::size_t ActionSize() { return sizeof(Action); }
} // namespace action

namespace {
// Instantiate Emit for all action types
using EmitPtr = void (*)();
template<std::size_t... I>
std::array<EmitPtr, sizeof...(I)> AllEmits(std::index_sequence<I...>) {
    return {reinterpret_cast<EmitPtr>(static_cast<void (*)(std::variant_alternative_t<I, Action>)>(&Emit))...};
}
const auto _ = AllEmits(std::make_index_sequence<std::variant_size_v<Action>>{});
} // namespace
