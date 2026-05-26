#pragma once

// imgui must be included before this header.

namespace MeshEditor {
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
