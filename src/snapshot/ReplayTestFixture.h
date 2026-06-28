#pragma once

#include <filesystem>
#include <span>

namespace snapshot {
// Write a replay-test fixture (`log.mea` + `expected.snap` + `actual.snap`) and return its directory.
// Empty path / no-op unless REPLAY_FIXTURE_DIR is defined (debug app only).
std::filesystem::path WriteReplayTestFixture(const std::filesystem::path &replay_log, std::span<const std::byte> expected, std::span<const std::byte> actual);
} // namespace snapshot
