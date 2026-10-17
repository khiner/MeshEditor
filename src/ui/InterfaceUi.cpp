#include "ui/InterfaceUi.h"
#include "action/View.h"
#include "render/SvgResource.h"
#include "ui/FieldEdit.h"
#include "ui/InterfaceTheme.h"

using namespace ImGui;

namespace ui {
void DrawIconButton(ImVec2 min, ImVec2 max, ImU32 color, float rounding, ImDrawFlags corners, const SvgResource *icon, float icon_dim) {
    static constexpr ImVec2 Inset{0.5f, 0.5f};
    GetWindowDrawList()->AddRectFilled(min + Inset, max - Inset, color, rounding, corners);
    if (!icon) return;
    const auto saved = GetCursorScreenPos();
    SetCursorScreenPos((min + max - ImVec2{icon_dim, icon_dim}) * 0.5f);
    icon->DrawIcon({icon_dim, icon_dim});
    SetCursorScreenPos(saved);
}

void DrawInterfaceStyle(state::Scene &r, state::Entity viewport) {
    Edit f{r, viewport};
    if (Button("Reset")) action::Emit(action::view::ResetInterfaceTheme{});
    SeparatorText("Sizes");
    f.Slider<&InterfaceTheme::IconButtonRounding>(nullptr, 0.f, 12.f, "%.1f");
}
} // namespace ui
