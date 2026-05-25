#pragma once

#include <entt/entity/fwd.hpp>

// ImGui controls that view/edit scene and entity state (rendered outside the viewport image).
void RenderControls(entt::registry &, entt::entity viewport); // Scene tab: shading, lighting, env, object tree, active-entity controls.
void RenderClipPickers(entt::registry &); // Animation-clip pickers.
