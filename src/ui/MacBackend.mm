#include "ui/MacBackend.h"
#include "MacPlatform.h"
#import <AppKit/AppKit.h>
#include <imgui.h>
#include <imgui_impl_osx.h>
#include <stdexcept>

namespace ui {
MacBackend::MacBackend(MacPlatform::Window &window) : Window(window) {
    if (!(Initialized = ImGui_ImplOSX_Init((__bridge NSView *)Window.NativeView())))
        throw std::runtime_error("Could not initialize ImGui's macOS backend.");
    ImGui::GetIO().BackendFlags |= ImGuiBackendFlags_HasSetMousePos;
}
MacBackend::~MacBackend() { Shutdown(); }
void MacBackend::NewFrame() {
    ImGui_ImplOSX_NewFrame((__bridge NSView *)Window.NativeView());
    Window.UpdateDrawableSize();
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
