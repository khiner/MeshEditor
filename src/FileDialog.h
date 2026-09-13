#pragma once

#include <filesystem>
#include <functional>

// Present a native macOS file dialog and run its completion callback asynchronously on the main thread.
namespace FileDialog {
// Receive the chosen path; cancellation invokes no callback.
using OnPick = std::function<void(const std::filesystem::path &)>;

// Separate extensions with semicolons and omit dots, for example "gltf;glb".
void ShowOpen(const char *extensions, OnPick, bool directories = false);
// A null extensions argument permits names without an extension.
void ShowSave(const char *extensions, const std::filesystem::path &default_path, OnPick);
void ShowPickFolder(OnPick);
} // namespace FileDialog
