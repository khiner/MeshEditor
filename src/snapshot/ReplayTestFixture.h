#pragma once

#include <filesystem>
#include <span>

namespace snapshot {
// Write a replay-test fixture (`log.mea` + `expected.snap` + `actual.snap`) capturing a case where
// replaying the log didn't reproduce the live scene state. A no-op unless REPLAY_FIXTURE_DIR is defined
// (debug app only).
void WriteReplayTestFixture(const std::filesystem::path &replay_log, std::span<const std::byte> expected, std::span<const std::byte> actual);
} // namespace snapshot
