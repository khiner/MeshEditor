#include "FaustParams.h"

#include "imgui.h"

#include <ranges>
#include <span>

using namespace ImGui;

using std::ranges::find;

bool RadioButtons(const char *label, float *value, const std::span<std::string> names, const std::span<double> values) {
    PushID(label);
    BeginGroup();

    Text("%s", label);
    bool changed = false;
    for (size_t i = 0; i < names.size(); ++i) {
        if (RadioButton(names[i].data(), *value == values[i])) {
            *value = float(values[i]);
            changed = true;
        }
    }
    EndGroup();
    PopID();

    return changed;
}

void FaustParams::DrawItem(const Item &item) {
    const auto type = item.type;
    const auto *label = item.label.c_str();
    using enum ItemType;
    if (!item.items.empty()) {
        if (type == TGroup) {
            BeginTabBar(label);
            for (const auto &child_item : item.items) {
                if (BeginTabItem(child_item.label.c_str())) {
                    DrawItem(child_item);
                    EndTabItem();
                }
            }
            EndTabBar();
        } else {
            // if (!item.label.empty()) SeparatorText(label); // not rendering group titles
            // Treating horizontal groups as vertical.
            // (No horizontal groups are used in the modal audio Faust UI.)
            for (const auto &child_item : item.items) DrawItem(child_item);
        }
        return;
    }
    if (type == Button) {
        ImGui::Button(label);
        if (IsItemActivated() && *item.zone == 0.0) *item.zone = 1.0;
        else if (IsItemDeactivated() && *item.zone == 1.0) *item.zone = 0.0;
    } else if (type == CheckButton) {
        if (auto value = bool(*item.zone); Checkbox(label, &value)) *item.zone = Real(value);
    } else if (type == NumEntry) {
        if (auto value = int(*item.zone); InputInt(label, &value, int(item.step))) *item.zone = std::clamp(Real(value), item.min, item.max);
    } else if (type == Knob || type == HSlider || type == VSlider || type == HBargraph || type == VBargraph) {
        const ImGuiSliderFlags flags = item.logscale ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None;
        if (auto value = float(*item.zone); SliderFloat(label, &value, float(item.min), float(item.max), nullptr, flags)) *item.zone = Real(value);
    } else if (type == HRadioButtons || type == VRadioButtons) {
        if (auto value = float(*item.zone); RadioButtons(item.label.c_str(), &value, NamesForZone.at(item.zone), ValuesForZone.at(item.zone))) *item.zone = Real(value);
    } else if (type == Menu) {
        auto value = *item.zone;
        const auto &names = NamesForZone.at(item.zone);
        const auto &values = ValuesForZone.at(item.zone);
        // todo handle not present
        if (const auto selected_index = find(values, value) - values.begin();
            BeginCombo(label, names[selected_index].c_str())) {
            for (size_t i = 0; i < NamesForZone.size(); ++i) {
                const Real choice_value = values[i];
                const bool is_selected = value == choice_value;
                if (Selectable(names[i].c_str(), is_selected)) *item.zone = Real(choice_value);
            }
            EndCombo();
        }
    }
    if (item.tooltip) {
        SameLine();
        TextDisabled("(?)");
        if (IsItemHovered() && BeginTooltip()) {
            PushTextWrapPos(GetFontSize() * 35);
            TextUnformatted(item.tooltip);
            PopTextWrapPos();
            EndTooltip();
        }
    }
}
