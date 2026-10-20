#pragma once

#include "state/Entity.h"

#include <string>

// ImGui controls that view/edit scene and entity state (rendered outside the viewport image).
void RenderControls(state::Registry &); // Scene tab: shading, lighting, env, object tree, active-entity controls.
void DrawViewportStyle(state::Registry &); // Viewport theme colors and sizes.
// The scene's name, or "Scene N" from its source index when the name is empty.
std::string SceneLabel(const state::Registry &, state::Entity scene);
