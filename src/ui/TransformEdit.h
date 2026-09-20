#pragma once

// The transform editor, over any field editor: a properties panel edits an entity's transform and the history editor edits a recorded one.

#include "gpu/Transform.h"

#include <imgui.h>

namespace ui {
template<typename E>
void DrawEditor(E &e, std::type_identity<Transform>, bool scale_locked = false) {
    e.template Drag<&Transform::P>("Position");
    e.template Draw<&Transform::R>("Rotation");
    if (scale_locked) ImGui::BeginDisabled();
    e.template Drag<&Transform::S>(scale_locked ? "Scale (frozen)" : "Scale");
    if (scale_locked) ImGui::EndDisabled();
}
} // namespace ui
