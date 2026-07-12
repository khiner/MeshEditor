#pragma once

#include <filesystem>
#include <functional>
#include <span>
#include <string_view>

// Native file dialogs backed by SDL3. The dialogs are asynchronous: the picked
// path arrives on a later `Pump` call, never inline with the Show* call.
namespace FileDialog {
struct Filter {
    std::string_view Name;
    std::string_view Pattern; // Semicolon-separated extensions without dots, e.g. "gltf;glb".
};

// Invoked from `Pump` with the chosen path. Not called when the user cancels or an error occurs.
using OnPick = std::function<void(const std::filesystem::path &)>;

void ShowOpen(std::span<const Filter> filters, OnPick);
void ShowSave(std::span<const Filter> filters, std::string_view default_name, OnPick);
void ShowPickFolder(OnPick);

// Run the callbacks for dialogs that completed since the last call. Call once per frame on the main thread.
void Pump();
} // namespace FileDialog
