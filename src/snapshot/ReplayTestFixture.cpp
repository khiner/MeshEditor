#include "snapshot/ReplayTestFixture.h"

#include <chrono>
#include <fstream>
#include <string_view>

namespace {
// Set by the debug app via -DREPLAY_FIXTURE_DIR, empty (a no-op) otherwise.
constexpr std::string_view FixtureRoot =
#ifdef REPLAY_FIXTURE_DIR
    REPLAY_FIXTURE_DIR;
#else
    "";
#endif

void WriteBytes(const std::filesystem::path &path, std::span<const std::byte> bytes) {
    std::ofstream out{path, std::ios::binary};
    out.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
}
} // namespace

namespace snapshot {
std::filesystem::path WriteReplayTestFixture(const std::filesystem::path &replay_log, std::span<const std::byte> expected, std::span<const std::byte> actual) {
    if (FixtureRoot.empty()) return {};

    const std::filesystem::path root{FixtureRoot};
    const auto unix_sec = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto dir = root / std::to_string(unix_sec);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::filesystem::copy_file(replay_log, dir / "log.actions", std::filesystem::copy_options::overwrite_existing, ec);
    WriteBytes(dir / "expected.snap", expected);
    WriteBytes(dir / "actual.snap", actual);
    return dir;
}
} // namespace snapshot
