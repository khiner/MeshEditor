#include "project/Sessions.h"

#include "File.h"
#include "Paths.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <optional>
#include <string>
#include <unistd.h>

#ifndef RESTORE_SESSION_RETAIN
#define RESTORE_SESSION_RETAIN 5
#endif

namespace project {
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
        if (!entry.is_directory(ec) || !std::filesystem::exists(entry.path() / "tree.log", ec) || !File::DirectoryLock{entry.path()}) continue;
        if (const auto seconds = ParseTimestamp(entry.path())) sessions.emplace_back(entry.path(), *seconds);
    }
    std::ranges::sort(sessions, std::ranges::greater{}, &RestoreSession::UnixSeconds);
    return sessions;
}

std::filesystem::path ReserveRestoreSession() {
    std::filesystem::create_directories(RestoreDir());
    // Retain closed unnamed projects independently of active projects.
    auto sessions = ListRestoreSessions();
    for (size_t i = RESTORE_SESSION_RETAIN; i < sessions.size(); ++i) {
        const File::DirectoryLock lock{sessions[i].Path};
        if (!lock) continue;
        std::error_code ec;
        std::filesystem::remove_all(sessions[i].Path, ec);
    }
    const auto unix_sec = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    auto name = (RestoreDir() / (std::to_string(unix_sec) + "-XXXXXX")).string();
    return ::mkdtemp(name.data()) ? std::filesystem::path{name} : std::filesystem::path{};
}
} // namespace project
