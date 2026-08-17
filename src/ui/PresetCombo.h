#pragma once

#include <imgui.h>

#include <string>

namespace ui {
// A preset names itself, as either a literal or a string.
inline const char *PresetLabel(const char *name) { return name; }
inline const char *PresetLabel(const std::string &name) { return name.c_str(); }

// Picker listing `choices` by name, calling `pick` with the one the user chooses when it is not already current.
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
