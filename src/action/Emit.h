#pragma once

#include <entt/entity/fwd.hpp>

#include <filesystem>

namespace action {
// Stage an action to execute at the end of the frame.
// First action emitted per-frame is executed and the rest are ignored.
template<typename ActionType> void Emit(ActionType);

// Apply the first action emitted this frame (if any).
void ApplyEmitted(entt::registry &, entt::entity viewport);

std::size_t ActionSize();

// Open the stream and start its writer thread.
// While running, ApplyEmitted records each applied action.
void StartLog();
// Flush and join the writer.
void StopLog();

// Advances the viewport between replayed actions (deferred events, draw-list rebuild) so actions resolving
// against rendered state settle correctly. Injected so the action layer needn't depend on the renderer.
using ReplayTick = void (*)(entt::registry &, entt::entity viewport);

// Reset to a new project: close the current log, build the default scene, optionally replay a `.mea` log's
// actions on top (synchronously, ticking the viewport via `tick` between them), then open a fresh log.
// Replayed actions are not re-logged.
void NewProject(entt::registry &, entt::entity viewport, const std::filesystem::path &replay_path = {}, ReplayTick tick = nullptr);
} // namespace action
