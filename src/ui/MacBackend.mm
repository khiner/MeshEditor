#include "ui/MacBackend.h"
#include "MacPlatform.h"
#import <AppKit/AppKit.h>
#include <imgui.h>
#include <imgui_impl_osx.h>
#include <imgui_internal.h>
#include <stdexcept>

namespace ui {
namespace {
void ReleaseInputs() {
    auto &io = ImGui::GetIO();
    for (int button = 0; button < ImGuiMouseButton_COUNT; ++button) io.AddMouseButtonEvent(button, false);
    for (auto key = ImGuiKey_Keyboard_BEGIN; key < ImGuiKey_Keyboard_END; key = ImGuiKey(key + 1)) io.AddKeyEvent(key, false);
    for (const auto mod : {ImGuiMod_Ctrl, ImGuiMod_Shift, ImGuiMod_Alt, ImGuiMod_Super}) io.AddKeyEvent(mod, false);
}
} // namespace

MacBackend::MacBackend(MacPlatform::Window &window) : Window(window) {
    if (!(Initialized = ImGui_ImplOSX_Init((__bridge NSView *)Window.NativeView())))
        throw std::runtime_error("Could not initialize ImGui's macOS backend.");
    auto &io = ImGui::GetIO();
    io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;
    io.ConfigDebugIgnoreFocusLoss = true; // Release inputs in NewFrame on deactivation instead.
}
MacBackend::~MacBackend() { Shutdown(); }
void MacBackend::NewFrame() {
    ImGui_ImplOSX_NewFrame((__bridge NSView *)Window.NativeView());
    Window.UpdateDrawableSize();
    // Release inputs on deactivation, since a release while another app is active is not delivered.
    const bool active = NSApp.isActive;
    if (Active && !active) ReleaseInputs();
    Active = active;
}
void MacBackend::HonorMouseWarp() {
    auto &io = ImGui::GetIO();
    if (!io.WantSetMousePos) return;
    io.WantSetMousePos = false;
    Window.WarpCursor(io.MousePos.x, io.MousePos.y);
}
void MacBackend::Shutdown() {
    if (!Initialized) return;
    ImGui_ImplOSX_Shutdown();
    Initialized = false;
}
} // namespace ui
