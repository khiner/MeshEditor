#pragma once

#include <imgui.h>

#include <concepts>
#include <ranges>
#include <string>

namespace ui {
// Combo over `choices` named by `name(choice)`, drawing the visible entries only.
// `choices` is a random-access range, or a callable that lists one while the popup is open.
// Calls `pick(choice)` when an entry other than `current` is chosen.
template<typename T, typename Choices, typename Name, typename Pick>
void ChoiceCombo(const char *label, const T &current, Choices &&choices, Name &&name, Pick &&pick) {
    if (!ImGui::BeginCombo(label, std::string{name(current)}.c_str())) return;
    const auto draw = [&](const auto &listed) {
        ImGuiListClipper clipper;
        clipper.Begin(int(std::ranges::size(listed)));
        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                const auto &choice = listed[i];
                const bool selected = choice == current;
                if (ImGui::Selectable(std::string{name(choice)}.c_str(), selected) && !selected) pick(choice);
                if (selected) ImGui::SetItemDefaultFocus();
            }
        }
    };
    if constexpr (std::invocable<Choices>) draw(choices());
    else draw(choices);
    ImGui::EndCombo();
}
} // namespace ui
