#include "action/Log.h"

#include "Paths.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <optional>
#include <string>

#ifndef RESTORE_SESSION_RETAIN
#define RESTORE_SESSION_RETAIN 5
#endif

namespace action {
namespace {
std::filesystem::path RestoreDir() { return Paths::UserData() / "restore"; }

std::optional<uint32_t> ParseTimestamp(const std::filesystem::path &dir) {
    const auto name = dir.filename().string();
    uint32_t seconds;
    if (auto [_, ec] = std::from_chars(name.data(), name.data() + name.size(), seconds); ec != std::errc{}) return std::nullopt;
    return seconds;
}
} // namespace

std::vector<RestoreSession> ListRestoreSessions() {
    std::vector<RestoreSession> sessions;
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator{RestoreDir(), ec}) {
        if (!entry.is_directory(ec)) continue;
        if (const auto seconds = ParseTimestamp(entry.path())) sessions.emplace_back(entry.path(), *seconds);
    }
    std::ranges::sort(sessions, std::ranges::greater{}, &RestoreSession::UnixSeconds);
    return sessions;
}

std::filesystem::path ReserveRestoreSession() {
    std::filesystem::create_directories(RestoreDir());
    // Keep at most RESTORE_SESSION_RETAIN sessions, including the new one.
    auto sessions = ListRestoreSessions();
    for (size_t i = RESTORE_SESSION_RETAIN - 1; i < sessions.size(); ++i) {
        std::error_code ec;
        std::filesystem::remove_all(sessions[i].Path, ec);
    }
    auto unix_sec = uint32_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    auto dir = RestoreDir() / std::to_string(unix_sec);
    // Bump the timestamp until the name is free (rapid opens can collide within a second).
    for (std::error_code ec; std::filesystem::exists(dir, ec);) dir = RestoreDir() / std::to_string(++unix_sec);
    std::filesystem::create_directories(dir);
    return dir;
}
} // namespace action
