#include "action/Log.h"

#include "Paths.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <optional>
#include <string>

#ifndef REPLAY_LOG_RETAIN
#define REPLAY_LOG_RETAIN 5
#endif

namespace action {
namespace {
std::filesystem::path ReplayDir() { return Paths::Base() / "replay"; }
constexpr std::string_view LogExt{".actions"};

// Parse the unix-seconds timestamp encoded in a `<unix_seconds>.actions` log's stem.
std::optional<uint32_t> ParseTimestamp(const std::filesystem::path &path) {
    if (path.extension() != LogExt) return std::nullopt;
    const auto stem = path.stem().string();
    uint32_t seconds;
    if (auto [_, ec] = std::from_chars(stem.data(), stem.data() + stem.size(), seconds); ec != std::errc{}) return std::nullopt;
    return seconds;
}
} // namespace

std::vector<ReplayLogFile> ListReplayLogs() {
    std::vector<ReplayLogFile> logs;
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator{ReplayDir(), ec}) {
        if (const auto seconds = ParseTimestamp(entry.path())) logs.emplace_back(entry.path(), *seconds);
    }
    std::ranges::sort(logs, std::ranges::greater{}, &ReplayLogFile::UnixSeconds);
    return logs;
}

std::pair<std::ofstream, std::filesystem::path> OpenLogStream() {
    std::filesystem::create_directories(ReplayDir());
    // Retain only the newest REPLAY_LOG_RETAIN-1 logs so this session's new log brings the total to at most REPLAY_LOG_RETAIN.
    auto logs = ListReplayLogs();
    for (size_t i = REPLAY_LOG_RETAIN - 1; i < logs.size(); ++i) {
        std::error_code ec;
        std::filesystem::remove(logs[i].Path, ec);
    }
    auto unix_sec = uint32_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    auto path = ReplayDir() / (std::to_string(unix_sec) + std::string{LogExt});
    // Filenames are unix-second timestamps. If one already exists (rapid New/Replay within a second),
    // appending would corrupt it, so bump until the name is free — it stays the newest ("Current").
    for (std::error_code ec; std::filesystem::exists(path, ec);) path = ReplayDir() / (std::to_string(++unix_sec) + std::string{LogExt});
    return {std::ofstream{path, std::ios::binary | std::ios::app}, std::move(path)};
}
} // namespace action
