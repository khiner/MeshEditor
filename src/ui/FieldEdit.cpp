#include "ui/FieldEdit.h"
#include "action/Animation.h"
#include "animation/AnimationTimeline.h"
#include "animation/Clips.h"
#include "animation/Fields.h"
#include "selection/SelectionComponents.h"
#include <imgui_internal.h>

namespace ui {
std::optional<animation::ChannelState> QueryKey(const state::Scene &r, state::Entity entity, const ChannelTarget &target) {
    const auto viewport = animation::AnimationsViewport(r);
    if (viewport == state::Null || entity == state::Null) return {};
    std::vector<float> value(target.Count);
    if (value.empty() || !animation::ReadField(r, entity, target, value)) return {};
    return animation::QueryChannel(r, viewport, {entity, target}, animation::FrameSeconds(r, viewport, r.get<const TimelinePlayback>(viewport).CurrentFrame));
}

KeyTint::KeyTint(const std::optional<animation::ChannelState> &state) {
    if (!state || !state->HasChannel) return;
    const auto &d = *state;
    const ImVec4 base = d.Changed ? ImVec4{0.70f, 0.40f, 0.10f, 1.f} : d.KeyAtFrame ? ImVec4{0.58f, 0.52f, 0.12f, 1.f} :
                                                                                      ImVec4{0.22f, 0.44f, 0.28f, 1.f};
    ImGui::PushStyleColor(ImGuiCol_FrameBg, base);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, {base.x * 1.2f, base.y * 1.2f, base.z * 1.2f, 1.f});
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, {base.x * 1.35f, base.y * 1.35f, base.z * 1.35f, 1.f});
    Pushed = 3;
}
KeyTint::~KeyTint() {
    if (Pushed) ImGui::PopStyleColor(Pushed);
}

void KeyDecorator(const state::Scene &r, state::Entity entity, const ChannelTarget &target, const animation::ChannelState &d) {
    ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
    ImGui::PushID(int(uint32_t(target.Component) ^ (uint32_t(target.Offset) << 16) ^ target.Index));
    const float h = ImGui::GetFrameHeight();
    const bool clicked = ImGui::InvisibleButton("##key", {h * 0.7f, h});
    const auto center = (ImGui::GetItemRectMin() + ImGui::GetItemRectMax()) * 0.5f;
    const auto color = ImGui::IsItemHovered() ? ImGui::GetColorU32(ImGuiCol_Text) : ImGui::GetColorU32(ImGuiCol_TextDisabled);
    auto *draw = ImGui::GetWindowDrawList();
    if (!d.HasChannel) {
        draw->AddCircle(center, 3.f, color, 0, 1.5f);
    } else {
        constexpr float half = 4.5f;
        const ImVec2 top{center.x, center.y - half}, right{center.x + half, center.y}, bottom{center.x, center.y + half}, left{center.x - half, center.y};
        if (d.KeyAtFrame) draw->AddQuadFilled(top, right, bottom, left, color);
        else draw->AddQuad(top, right, bottom, left, color, 1.5f);
    }
    ImGui::SetItemTooltip("%s", d.KeyAtFrame ? "Delete keyframe" : d.HasChannel ? "Insert keyframe" :
                                                                                  "Insert keyframe and start animating this property");
    if (clicked) {
        const action::animation::KeyScope keys{entity == FindActiveEntity(r) ? action::Target{action::OnActive{}} : action::Target{entity}, target};
        if (d.KeyAtFrame) action::Emit(action::animation::DeleteKey{keys});
        else action::Emit(action::animation::InsertKey{keys});
    }
    ImGui::PopID();
}

action::Target TargetFromAlt(bool delta_capable) {
    if (!ImGui::GetIO().KeyAlt) return action::OnActive{};
    if (delta_capable) return action::OnSelectedDelta{};
    return action::OnSelected{};
}

namespace detail {
namespace {
// Preserve gesture state because ImGui permits one active item.
action::Target GestureTarget{action::OnActive{}};
bool GestureTyped{false};
} // namespace

std::optional<action::Target> FieldGesture(state::Scene &r, bool changed, bool selection, bool delta_capable) {
    if (selection && ImGui::IsItemHovered() && r.view<Selected>().size() > 1) ImGui::SetItemTooltip("Hold Alt to apply to all selected");
    if (ImGui::IsItemActivated()) {
        GestureTarget = TargetFromAlt(delta_capable);
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
    // An explicit-entity editor records its own target.
    if (!selection) return action::Target{action::OnActive{}};
    // An item that was never activated reads the modifier at the change.
    const auto target = ImGui::IsItemActive() || ImGui::IsItemDeactivated() ? GestureTarget : TargetFromAlt(delta_capable);
    // Alt-typed values copy to the selection instead of offsetting it.
    return std::holds_alternative<action::OnSelectedDelta>(target) && GestureTyped ? action::Target{action::OnSelected{}} : target;
}
} // namespace detail
} // namespace ui
