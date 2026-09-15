#pragma once

#include <imgui.h>

#include <string>

namespace ui {
// Combo over `choices` named by `name(choice)`. Calls `pick(choice)` when an entry other than `current` is chosen.
template<typename T, typename Choices, typename Name, typename Pick>
void ChoiceCombo(const char *label, const T &current, const Choices &choices, Name &&name, Pick &&pick) {
    if (!ImGui::BeginCombo(label, std::string{name(current)}.c_str())) return;
    for (const auto &choice : choices) {
        const bool selected = choice == current;
        if (ImGui::Selectable(std::string{name(choice)}.c_str(), selected) && !selected) pick(choice);
        if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
}
} // namespace ui
