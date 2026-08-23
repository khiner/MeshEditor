#pragma once

#include <filesystem>
#include <memory>
#include <vector>

namespace CA {
class MetalLayer;
} // namespace CA

namespace MacPlatform {
void InitPaths();

struct Events {
    bool Quit{false};
    float ScrollX{0};
    float ScrollY{0};
    std::vector<std::filesystem::path> DroppedFiles;
};

class Window {
public:
    struct Impl;

    Window();
    ~Window();

    Window(const Window &) = delete;
    Window &operator=(const Window &) = delete;

    CA::MetalLayer *Layer() const;
    Events PollEvents();

    void InitImGui();
    void NewImGuiFrame();
    // Warps the OS cursor to a position ImGui requested this frame.
    void HonorMouseWarp();
    void ShutdownImGui();

private:
    std::unique_ptr<Impl> Data;
};
} // namespace MacPlatform
