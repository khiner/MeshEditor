#pragma once
namespace MacPlatform {
class Window;
}
namespace ui {
class MacBackend {
public:
    explicit MacBackend(MacPlatform::Window &);
    ~MacBackend();
    MacBackend(const MacBackend &) = delete;
    MacBackend &operator=(const MacBackend &) = delete;
    void NewFrame();
    void HonorMouseWarp();
    void Shutdown();

private:
    MacPlatform::Window &Window;
    bool Initialized{false};
};
} // namespace ui
