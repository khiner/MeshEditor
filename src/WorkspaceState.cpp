#include "WorkspaceState.h"

#include "File.h"
#include "action/SerializeNumeric.h"
#include "viewport/ViewCameraSerialize.h"
#include "viewport/ViewportDisplay.h"

#include <entt/entity/registry.hpp>
#include <imgui.h>
#include <zpp_bits.h>

#include <fstream>

namespace workspace {
namespace {
constexpr uint32_t Version = 1;

constexpr auto SerializeWindowVisibility(auto &archive, auto &visibility) {
    return archive(
        visibility.SceneControls,
        visibility.Viewport,
        visibility.Animation,
        visibility.ImGuiDemo,
        visibility.ImSpinnerDemo,
        visibility.ImPlotDemo,
        visibility.Debug
    );
}
} // namespace

State Capture(const entt::registry &r, entt::entity viewport, const WindowsState &windows) {
    size_t ini_size = 0;
    const char *ini = ImGui::GetCurrentContext() ? ImGui::SaveIniSettingsToMemory(&ini_size) : nullptr;
    return {
        .ViewCamera = GetViewCameraState(r, viewport),
        .ViewportExtent = r.ctx().get<const ViewportExtent>().Value,
        .Windows = GetWindowVisibility(windows),
        .ImGuiIni = ini ? std::string{ini, ini_size} : std::string{},
    };
}

void Apply(entt::registry &r, entt::entity viewport, WindowsState &windows, const State &state) {
    r.ctx().get<ViewportExtent>().Value = state.ViewportExtent;
    SetViewCameraState(r, viewport, state.ViewCamera);
    SetWindowVisibility(windows, state.Windows);
    if (ImGui::GetCurrentContext() && !state.ImGuiIni.empty()) {
        ImGui::LoadIniSettingsFromMemory(state.ImGuiIni.data(), state.ImGuiIni.size());
        windows.LayoutLoaded = true;
    }
}

std::vector<std::byte> Serialize(const State &state) {
    std::vector<std::byte> bytes;
    zpp::bits::out archive{bytes};
    const bool has_saved_view = state.ViewCamera.LookThroughSaved.has_value();
    if (zpp::bits::failure(archive(Version, state.ViewCamera.Active, has_saved_view)) ||
        (has_saved_view && zpp::bits::failure(archive(*state.ViewCamera.LookThroughSaved))) ||
        zpp::bits::failure(archive(state.ViewportExtent)) ||
        zpp::bits::failure(SerializeWindowVisibility(archive, state.Windows)) ||
        zpp::bits::failure(archive(state.ImGuiIni))) {
        return {};
    }
    bytes.resize(archive.position());
    return bytes;
}

std::optional<State> Deserialize(std::span<const std::byte> bytes) {
    zpp::bits::in archive{bytes};
    uint32_t version{};
    State state{
        .ViewCamera = {ViewCamera{vec3{0, 0, 1}, vec3{0}, Camera{}}, std::nullopt},
        .ViewportExtent = {},
        .Windows = {},
        .ImGuiIni = {},
    };
    bool has_saved_view{};
    if (zpp::bits::failure(archive(version, state.ViewCamera.Active, has_saved_view)) || version != Version) {
        return std::nullopt;
    }
    if (has_saved_view) {
        state.ViewCamera.LookThroughSaved.emplace(vec3{0, 0, 1}, vec3{0}, Camera{});
        if (zpp::bits::failure(archive(*state.ViewCamera.LookThroughSaved))) return std::nullopt;
    }
    if (zpp::bits::failure(archive(state.ViewportExtent)) ||
        zpp::bits::failure(SerializeWindowVisibility(archive, state.Windows)) ||
        zpp::bits::failure(archive(state.ImGuiIni))) {
        return std::nullopt;
    }
    return state;
}

std::optional<State> Load(const std::filesystem::path &path) {
    const auto bytes = File::Read(path);
    return bytes ? Deserialize(*bytes) : std::nullopt;
}

bool Save(const std::filesystem::path &path, std::span<const std::byte> bytes) {
    if (bytes.empty()) return false;

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream out{temporary, std::ios::binary | std::ios::trunc};
        out.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
        if (!out) return false;
    }
    std::filesystem::rename(temporary, path, ec);
    if (!ec) return true;
    std::filesystem::remove(temporary, ec);
    return false;
}
} // namespace workspace
