#pragma once

#include "state/Entity.h"

// ImGui controls that view/edit scene and entity state (rendered outside the viewport image).
void RenderControls(state::Scene &, state::Entity viewport); // Scene tab: shading, lighting, env, object tree, active-entity controls.
