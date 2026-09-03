#pragma once

#include <filesystem>
#include <functional>

// Present a native macOS file dialog and run its completion callback asynchronously on the main thread.
namespace FileDialog {
// Receive the chosen path; cancellation invokes no callback.
using OnPick = std::function<void(const std::filesystem::path &)>;

// Separate extensions with semicolons and omit dots, for example "gltf;glb".
void ShowOpen(const char *extensions, OnPick);
void ShowSave(const char *extensions, const char *default_name, OnPick);
void ShowPickFolder(OnPick);
} // namespace FileDialog
