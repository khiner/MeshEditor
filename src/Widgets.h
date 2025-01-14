#pragma once

#include "imgui.h"

// Helper extensions to ImGui widgets.

namespace MeshEditor {
inline bool BeginTable(const char *name, int columns) {
    const auto line_height = ImGui::GetTextLineHeightWithSpacing();
    const bool result = ImGui::BeginTable(name, columns, ImGuiTableFlags_ScrollY, {0, line_height * 8});
    if (result) {
        ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
    }
    return result;
}

inline void HelpMarker(const char *desc) {
    using namespace ImGui;
    SameLine();
    TextDisabled("(?)");
    if (BeginItemTooltip()) {
        PushTextWrapPos(GetFontSize() * 35.0f);
        TextUnformatted(desc);
        PopTextWrapPos();
        EndTooltip();
    }
}
} // namespace MeshEditor
