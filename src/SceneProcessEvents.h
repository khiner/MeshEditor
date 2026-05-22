#pragma once

#include "SceneFrame.h"

#include <entt/entity/fwd.hpp>

// Drains Pending* markers, reactive trackers, and dirty sets accumulated since the last frame.
// Returns the strongest render request triggered by this frame's changes.
RenderRequest ProcessComponentEvents(entt::registry &, entt::entity scene_entity);
