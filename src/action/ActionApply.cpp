#include "action/ActionApply.h"
#include "action/ActionDrain.h"
#include "action/Log.h"
#include "action/LogSerialize.h"
#include "animation/AnimationTimeline.h"

#include <entt/entity/registry.hpp>

#include <fstream>
#include <optional>

using namespace action;

namespace {
std::optional<Action> Held; // Last staged step of the in-progress gesture, awaiting commit.

std::optional<std::ofstream> LogStream;
std::optional<WriteBehindLog<Action>> Log;
std::filesystem::path LogPath; // Currently-open `.actions` log, empty when none.

// Record a committed change to the .actions log.
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
                DomainV dv{std::forward<T>(a)};
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

void CommitHeld() {
    if (Held) {
        RecordCommitted(std::move(*Held));
        Held.reset();
    }
}
} // namespace

namespace action {
void StartLog() {
    auto [stream, path] = OpenLogStream();
    LogPath = std::move(path);
    LogStream.emplace(std::move(stream));
    Log.emplace(*LogStream, &SerializeAction);
}
std::filesystem::path StopLog() {
    if (Log) Log->Stop();
    Log.reset();
    LogStream.reset(); // flush and close before checking the file on disk
    auto path = std::exchange(LogPath, {});
    if (path.empty()) return {};

    // Drop the just-closed log if nothing was recorded.
    std::error_code ec;
    if (std::filesystem::file_size(path, ec) == 0 && !ec) {
        std::filesystem::remove(path, ec);
        return {};
    }
    return path;
}

void StopPlaybackIfPlaying(entt::registry &r, entt::entity viewport) {
    const auto &playback = r.get<const TimelinePlayback>(viewport);
    if (playback.Playing) RecordCommitted(ApplyAction(r, viewport, Action{timeline::TogglePlay{playback.CurrentFrame}}));
}

void ApplyEmitted(entt::registry &r, entt::entity viewport) {
    auto drained = Drain();
    if (drained.Emitted) {
        auto [action, phase] = std::move(*drained.Emitted);
        auto recorded = ApplyAction(r, viewport, std::move(action));
        switch (phase) {
            case Phase::Stage: Held = std::move(recorded); break;
            case Phase::Cancel:
                Held.reset();
                r.clear<DragFieldStart>();
                break;
            case Phase::Record:
                CommitHeld();
                RecordCommitted(std::move(recorded));
                r.clear<DragFieldStart>();
                break;
        }
    }
    if (drained.CommitRequested) {
        CommitHeld();
        r.clear<DragFieldStart>();
    }
}

bool ReplayLog(entt::registry &r, entt::entity viewport, const std::filesystem::path &replay_path, ReplayTick tick) {
    std::ifstream in{replay_path, std::ios::binary};
    if (!in) return false;

    tick(r, viewport);
    StreamActions(in, [&](Action &&a) {
        RecordCommitted(ApplyAction(r, viewport, std::move(a)));
        r.clear<DragFieldStart>(); // Each replayed action is one committed gesture.
        tick(r, viewport);
    });
    return true;
}
} // namespace action
