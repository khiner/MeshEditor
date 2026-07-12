#include "action/ActionApply.h"
#include "action/ActionDrain.h"
#include "action/ActionIndex.h"
#include "action/Log.h"
#include "action/LogSerialize.h"

#include <entt/entity/registry.hpp>

#include <fstream>
#include <optional>

using namespace action;

namespace {
std::optional<Action> Held; // Last staged step of the in-progress gesture, awaiting commit.

std::optional<std::ofstream> LogStream;
std::optional<WriteBehindLog<Action>> Log;
std::filesystem::path LogPath; // Currently-open `.actions` log, empty when none.

// Advance the action index and append the committed change to the .actions log (when one is open).
void RecordCommitted(entt::registry &r, entt::entity viewport, Action &&a) {
    if (!IsRecordable(a)) return;
    ++r.get_or_emplace<ActionIndex>(viewport).Index;
    if (Log) Log->Enqueue(std::move(a));
}

// Route the action to its domain's Apply.
void ApplyAction(entt::registry &r, entt::entity viewport, const Action &action) {
    std::visit([&](const auto &dv) { Apply(r, viewport, dv); }, action);
}

// Apply and record as one committed action.
void ApplyRecord(entt::registry &r, entt::entity viewport, Action &&a) {
    ApplyAction(r, viewport, a);
    RecordCommitted(r, viewport, std::move(a));
}

void CommitHeld(entt::registry &r, entt::entity viewport) {
    if (Held) {
        RecordCommitted(r, viewport, std::move(*Held));
        Held.reset();
    }
}
} // namespace

namespace action {
void StartLog(std::filesystem::path path, bool append) {
    if (const auto parent = path.parent_path(); !parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }
    LogPath = std::move(path);
    LogStream.emplace(LogPath, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
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

template<typename ActionType> void ApplyNow(entt::registry &r, entt::entity viewport, ActionType a) {
    ApplyRecord(r, viewport, MakeAction(std::move(a)));
}

void ApplyEmitted(entt::registry &r, entt::entity viewport) {
    auto drained = Drain();
    if (drained.Emitted) {
        auto [action, phase] = std::move(*drained.Emitted);
        ApplyAction(r, viewport, action);
        switch (phase) {
            case Phase::Stage: Held = std::move(action); break;
            case Phase::Cancel:
                Held.reset();
                r.clear<DragFieldStart>();
                break;
            case Phase::Record:
                CommitHeld(r, viewport);
                RecordCommitted(r, viewport, std::move(action));
                r.clear<DragFieldStart>();
                break;
        }
    }
    if (drained.CommitRequested) {
        CommitHeld(r, viewport);
        r.clear<DragFieldStart>();
    }
    // System-generated actions apply in addition to the user action, without touching any open gesture.
    for (auto &a : drained.System) ApplyRecord(r, viewport, std::move(a));
}

bool ReplayLog(entt::registry &r, entt::entity viewport, const std::filesystem::path &replay_path, ReplayTick tick, uint64_t skip) {
    std::ifstream in{replay_path, std::ios::binary};
    if (!in) return false;

    // Skip records already captured by the base snapshot.
    for (uint64_t i = 0; i < skip; ++i) {
        uint32_t len;
        if (!in.read(reinterpret_cast<char *>(&len), sizeof len)) return true;
        in.seekg(len, std::ios::cur);
    }

    tick(r, viewport);
    StreamActions(in, [&](Action &&a) {
        ApplyRecord(r, viewport, std::move(a));
        r.clear<DragFieldStart>(); // Each replayed action is one committed gesture.
        tick(r, viewport);
    });
    return true;
}
} // namespace action

namespace {
// Force instantiation of ApplyNow for every leaf action type so call sites in other TUs link.
using ApplyNowPtr = void (*)();
template<typename DV> constexpr auto DomainApplyNows() {
    return []<size_t... I>(std::index_sequence<I...>) {
        const auto inst = [](auto fn) { return reinterpret_cast<ApplyNowPtr>(fn); };
        return std::array<ApplyNowPtr, sizeof...(I)>{
            inst(static_cast<void (*)(entt::registry &, entt::entity, std::variant_alternative_t<I, DV>)>(&ApplyNow))...,
        };
    }(std::make_index_sequence<std::variant_size_v<DV>>{});
}
const auto _ = MapDomains([]<typename DV>() { return DomainApplyNows<DV>(); });
} // namespace
