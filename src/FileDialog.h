#pragma once

#include <filesystem>
#include <functional>

// Native macOS file dialogs. The completion callback runs asynchronously on the main thread.
namespace FileDialog {
// Invoked with the chosen path. Not called when the user cancels.
using OnPick = std::function<void(const std::filesystem::path &)>;

// Extensions are semicolon-separated and omit dots, e.g. "gltf;glb".
void ShowOpen(const char *extensions, OnPick);
void ShowSave(const char *extensions, const char *default_name, OnPick);
void ShowPickFolder(OnPick);
} // namespace FileDialog
