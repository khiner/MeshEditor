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

void FaustParams::DrawItem(const FaustParams::Item &item) {
    const auto type = item.type;
    const auto *label = item.label.c_str();
    if (!item.items.empty()) {
        // if (!item.label.empty()) SeparatorText(label); // turning off to not render model title
        // xxx Relying on the knowledge that no groups are used in the modal audio Faust UI.
        for (const auto &child_item : item.items) DrawItem(child_item);
    }
    if (type == ItemType_Button) {
        Button(label);
        if (IsItemActivated() && *item.zone == 0.0) *item.zone = 1.0;
        else if (IsItemDeactivated() && *item.zone == 1.0) *item.zone = 0.0;
    } else if (type == ItemType_CheckButton) {
        auto value = bool(*item.zone);
        if (Checkbox(label, &value)) *item.zone = Real(value);
    } else if (type == ItemType_NumEntry) {
        auto value = int(*item.zone);
        if (InputInt(label, &value, int(item.step))) *item.zone = std::clamp(Real(value), item.min, item.max);
    } else if (type == ItemType_Knob || type == ItemType_HSlider || type == ItemType_VSlider || type == ItemType_HBargraph || type == ItemType_VBargraph) {
        auto value = float(*item.zone);
        ImGuiSliderFlags flags = item.logscale ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None;
        if (SliderFloat(label, &value, float(item.min), float(item.max), nullptr, flags)) *item.zone = Real(value);
    } else if (type == ItemType_HRadioButtons || type == ItemType_VRadioButtons) {
        auto value = float(*item.zone);
        if (RadioButtons(item.label.c_str(), &value, NamesForZone.at(item.zone), ValuesForZone.at(item.zone))) *item.zone = Real(value);
    } else if (type == ItemType_Menu) {
        auto value = *item.zone;
        const auto &names = NamesForZone.at(item.zone);
        const auto &values = ValuesForZone.at(item.zone);
        // todo handle not present
        const auto selected_index = find(values, value) - values.begin();
        if (BeginCombo(label, names[selected_index].c_str())) {
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
