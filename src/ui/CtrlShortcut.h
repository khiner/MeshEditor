#pragma once

#include <imgui.h>

// A Ctrl chord pressed with Control or Command, which ImGui reports as Super and Ctrl on macOS.
inline bool CtrlShortcut(ImGuiKeyChord chord, ImGuiInputFlags flags = 0) {
    return ImGui::Shortcut(chord, flags) || ImGui::Shortcut((chord & ~ImGuiMod_Ctrl) | ImGuiMod_Super, flags);
}
