#include "ui/FieldEdit.h"
#include "selection/SelectionComponents.h"
#include <imgui_internal.h>

namespace ui {
action::Scope ScopeFromAlt(bool delta_capable) {
    if (!ImGui::GetIO().KeyAlt) return action::Scope::Active;
    return delta_capable ? action::Scope::SelectedDelta : action::Scope::Selected;
}

namespace detail {
namespace {
// Preserve gesture state because ImGui permits one active item.
action::Scope GestureScope{action::Scope::Active};
bool GestureTyped{false};
} // namespace

std::optional<action::Scope> FieldGesture(state::Scene &r, bool changed, bool selection, bool delta_capable) {
    if (selection && ImGui::IsItemHovered() && r.view<Selected>().size() > 1) ImGui::SetItemTooltip("Hold Alt to apply to all selected");
    if (ImGui::IsItemActivated()) {
        GestureScope = ScopeFromAlt(delta_capable);
        GestureTyped = false;
    }
    if (ImGui::TempInputIsActive(ImGui::GetItemID())) GestureTyped = true;
    if (ImGui::IsItemDeactivated() && !ImGui::IsItemDeactivatedAfterEdit()) {
        action::Cancel();
        return {};
    }
    // Widgets that change without staying active (combos, checkboxes) commit at once.
    if (ImGui::IsItemDeactivatedAfterEdit() || (changed && !ImGui::IsItemActive())) action::Commit();
    if (!changed) return {};
    if (!selection) return action::Scope::Entity;
    // An item that was never activated reads the modifier at the change.
    const auto scope = ImGui::IsItemActive() || ImGui::IsItemDeactivated() ? GestureScope : ScopeFromAlt(delta_capable);
    // Alt-typed values copy to the selection instead of offsetting it.
    return scope == action::Scope::SelectedDelta && GestureTyped ? action::Scope::Selected : scope;
}
} // namespace detail
} // namespace ui
