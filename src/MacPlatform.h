#pragma once

#include <filesystem>
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
    bool PreciseScrolling{false}; // The scroll came from a trackpad, at a tenth of its points
    float Magnification{0}; // The pinch's summed magnification
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
    void Show();

    void *NativeView() const;
    void UpdateDrawableSize();
    // Content-view coordinates, with a top-left origin.
    void WarpCursor(float x, float y);

private:
    std::unique_ptr<Impl> Data;
};
} // namespace MacPlatform
