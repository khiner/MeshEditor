#pragma once

#include "state/Entity.h"

// ImGui controls that view/edit scene and entity state (rendered outside the viewport image).
void RenderControls(state::Registry &, state::Entity viewport); // Scene tab: shading, lighting, env, object tree, active-entity controls.
void DrawViewportStyle(state::Registry &, state::Entity viewport); // Viewport theme colors and sizes.
