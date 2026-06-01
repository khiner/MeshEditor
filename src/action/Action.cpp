#include "action/Action.h"
#include "action/Emit.h"
#include "action/Log.h"
#include "action/LogSerialize.h"

using namespace action;

namespace {
// A buffered action's gesture-grouping role, chosen per emission (not per action type, since the same
// action can be a one-shot edit or a staged gesture step).
enum class Phase {
    Default,
    Stage,
    Cancel,
};

std::optional<std::pair<Action, Phase>> Emitted; // This frame's winning action and its phase.
std::optional<Action> Held; // Last staged step of the in-progress gesture, awaiting commit.
bool CommitRequested = false; // A terminal-less gesture asked to commit its held step.

std::optional<std::ofstream> LogStream;
std::optional<WriteBehindLog<Action>> Log;

// Record a committed change: the .mea log today, the Undo/Redo stack soon.
void RecordCommitted(Action &&a) {
    if (Log && std::visit([]<typename T>(const T &) { return Recordable<std::decay_t<T>>; }, a)) Log->Enqueue(std::move(a));
}

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

namespace {
// First action emitted in the frame wins; the rest are ignored.
template<typename ActionType> void Buffer(ActionType a, Phase phase) {
    if (!Emitted) Emitted.emplace(Action{std::move(a)}, phase);
}
} // namespace

namespace action {
template<typename ActionType> void Emit(ActionType a) { Buffer(std::move(a), Phase::Default); }
template<typename ActionType> void EmitStaged(ActionType a) { Buffer(std::move(a), Phase::Stage); }
template<typename ActionType> void EmitCancel(ActionType a) { Buffer(std::move(a), Phase::Cancel); }
void Commit() { CommitRequested = true; }

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

void CommitHeld() {
    if (Held) {
        RecordCommitted(std::move(*Held));
        Held.reset();
    }
}

void ApplyEmitted(entt::registry &r, entt::entity viewport) {
    if (Emitted) {
        // Apply for live feedback (every phase applies), then group by phase to decide recording.
        auto [action, phase] = std::move(*Emitted);
        Emitted.reset();
        auto recorded = ApplyAction(r, viewport, std::move(action));
        switch (phase) {
            case Phase::Stage: Held = std::move(recorded); break; // Hold; supersede the prior step.
            case Phase::Cancel: Held.reset(); break; // Discard the aborted gesture (revert already applied).
            case Phase::Default:
                CommitHeld();
                RecordCommitted(std::move(recorded));
                break; // Commit gesture first, then record self.
        }
    }
    if (CommitRequested) {
        CommitHeld();
        CommitRequested = false;
    }
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
// Force instantiation of every Emit* entry point for every action type so call sites in other TUs link.
using EmitPtr = void (*)();
template<std::size_t... I>
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
