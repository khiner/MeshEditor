#pragma once

#include "state/Scene.h"

using state::On;

template<typename Change>
auto &reactive(state::Scene &r) { return r.changes(state::Type<Change>()); }

enum class ComponentEventPhase { BeforePose,
                                 AfterPose };
enum class EventPass;
struct ComponentEventHandler {
    void (*Apply)(state::Scene &, EventPass);
    ComponentEventPhase Phase;
};

inline void RegisterComponentEventHandler(
    state::Scene &r, void (*handler)(state::Scene &, EventPass),
    ComponentEventPhase phase = ComponentEventPhase::BeforePose
) {
    r.ctx().emplace<std::vector<ComponentEventHandler>>().push_back({handler, phase});
}

// Run domain setup handlers on the viewport entity.
struct SceneSetupHandlers {
    std::vector<void (*)(state::Scene &, state::Entity)> Handlers;
};

// Run domain clear handlers after scene destruction and before resetting entity identifiers.
struct SceneClearHandlers {
    std::vector<void (*)(state::Scene &)> Handlers;
};

inline void RegisterSceneSetupHandler(state::Scene &r, void (*handler)(state::Scene &, state::Entity)) {
    r.ctx().emplace<SceneSetupHandlers>().Handlers.emplace_back(handler);
}

inline void RegisterSceneClearHandler(state::Scene &r, void (*handler)(state::Scene &)) {
    r.ctx().emplace<SceneClearHandlers>().Handlers.emplace_back(handler);
}
