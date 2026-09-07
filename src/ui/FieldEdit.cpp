#include "ui/FieldEdit.h"
#include "selection/SelectionComponents.h"
#include <imgui_internal.h>

namespace ui {
action::Scope ScopeFromAlt(bool delta_capable) {
    if (!ImGui::GetIO().KeyAlt) return action::Scope::Active;
    return delta_capable ? action::Scope::SelectedDelta : action::Scope::Selected;
}

namespace detail {
action::Scope GestureScope{action::Scope::Active};
std::array<std::byte, 16> GestureStartValue{};
bool GestureTyped{false};
std::function<void()> GestureCancel;

void FieldGesture::Capture() {
    GestureTyped = false;
    std::memcpy(GestureStartValue.data(), Original.data(), Original.size());
    if (Selection) {
        GestureScope = ScopeFromAlt(DeltaCapable);
        // Clear the transient baseline from an interrupted selection drag.
        if (GestureScope == action::Scope::SelectedDelta) R.clear<action::DragFieldStart>();
    }
}

bool FieldGesture::Begin() {
    if (Selection && ImGui::IsItemHovered() && R.view<Selected>().size() > 1) ImGui::SetItemTooltip("Hold Alt to apply to all selected");
    if (!ImGui::IsItemActivated()) return false;
    Capture();
    return true;
}

bool FieldGesture::ShouldStage(bool changed) {
    if (ImGui::TempInputIsActive(ImGui::GetItemID())) GestureTyped = true;
    if (ImGui::IsItemDeactivatedAfterEdit()) return changed;
    if (ImGui::IsItemDeactivated() || !changed) return false;
    // Delay typed edits until commit.
    if (ImGui::IsItemActive() && GestureTyped) return false;
    // Capture the modifier scope for instantaneous widgets.
    if (!ImGui::IsItemActive()) Capture();
    return true;
}

bool FieldGesture::End(bool changed) {
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        action::Commit();
        GestureCancel = nullptr;
    } else if (ImGui::IsItemDeactivated()) {
        if (GestureCancel) {
            GestureCancel();
            GestureCancel = nullptr;
        }
        return false;
    } else if (changed && !ImGui::IsItemActive()) {
        action::Commit();
    }
    return changed;
}
} // namespace detail
} // namespace ui
