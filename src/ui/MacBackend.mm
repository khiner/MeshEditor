#include "ui/MacBackend.h"
#include "MacPlatform.h"
#import <AppKit/AppKit.h>
#include <imgui_impl_osx.h>
#include <imgui_internal.h>

namespace ui {
namespace {
void ReleaseKeys() {
    auto &io = ImGui::GetIO();
    for (auto key = ImGuiKey_Keyboard_BEGIN; key < ImGuiKey_Keyboard_END; key = ImGuiKey(key + 1)) io.AddKeyEvent(key, false);
    for (const auto mod : {ImGuiMod_Ctrl, ImGuiMod_Shift, ImGuiMod_Alt, ImGuiMod_Super}) io.AddKeyEvent(mod, false);
}
} // namespace

MacBackend::MacBackend(MacPlatform::Window &window) : Window(window) {
    if (!(Initialized = ImGui_ImplOSX_Init((__bridge NSView *)Window.NativeView()))) {
        throw std::runtime_error("Could not initialize ImGui's macOS backend.");
    }
    auto &io = ImGui::GetIO();
    io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;
    io.ConfigDebugIgnoreFocusLoss = true; // Release inputs in NewFrame instead.
    // Seed the mouse position, since no mouse event arrives until the cursor moves.
    auto *const view = (__bridge NSView *)Window.NativeView();
    const auto pos = [view convertPoint:view.window.mouseLocationOutsideOfEventStream fromView:nil];
    io.AddMousePosEvent(pos.x, pos.y);
}
MacBackend::~MacBackend() { Shutdown(); }
void MacBackend::NewFrame() {
    ImGui_ImplOSX_NewFrame((__bridge NSView *)Window.NativeView());
    Window.UpdateDrawableSize();
    // Release mouse buttons the hardware reports up, since nested tracking loops like window resizing consume the mouse-up.
    const auto pressed = NSEvent.pressedMouseButtons;
    for (int button = 0; button < ImGuiMouseButton_COUNT; ++button) {
        if (!(pressed & (1 << button))) ImGui::GetIO().AddMouseButtonEvent(button, false);
    }
    // Release keys on deactivation, since a release while another app is active is not delivered.
    const bool active = NSApp.isActive;
    if (Active && !active) ReleaseKeys();
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
