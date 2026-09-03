#pragma once

#include <imgui.h>

#include <string>

namespace ui {
// Return a preset's literal or string name.
inline const char *PresetLabel(const char *name) { return name; }
inline const char *PresetLabel(const std::string &name) { return name.c_str(); }

// Display named choices and call pick when the selection changes.
void PresetCombo(const char *label, const std::string &current, const auto &choices, auto &&pick) {
    if (!ImGui::BeginCombo(label, current.c_str())) return;
    for (const auto &choice : choices) {
        const auto *name = PresetLabel(choice.Name);
        const bool is_selected = current == name;
        if (ImGui::Selectable(name, is_selected) && !is_selected) pick(choice);
        if (is_selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
}
} // namespace ui
