#pragma once

#include "action/Emit.h"
#include "action/Selection.h"
#include "ui/FieldEdit.h"
#include "ui/InterfaceUi.h"

#include <imgui.h>

namespace ui {
// An icon button beside a reference picker that makes the referenced node the active one, which the node editor and the Properties panel then show.
inline void GoToButton(const SvgResource *icon, const char *id, state::Entity target) {
    ImGui::SameLine();
    ImGui::PushID(id);
    const float size = ImGui::GetFrameHeight();
    BeginIconButtonGroup(ImGuiAxis_X, {size, size}, 1, ImGui::GetStyle().FrameRounding);
    if (IconButton("##GoTo", icon, false, target != state::Null)) action::Emit(action::selection::Select{target});
    EndIconButtonGroup();
    Describe("Go to", "Make this node the active node, to edit it in the Properties panel.");
    ImGui::PopID();
}
} // namespace ui
