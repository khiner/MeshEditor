#pragma once

#include <filesystem>
#include <vector>

namespace project {
struct RestoreSession {
    std::filesystem::path Path;
    uint32_t UnixSeconds;
};
// Return inactive unnamed projects newest first.
std::vector<RestoreSession> ListRestoreSessions();
// Create a unique unnamed project directory and prune inactive retained projects.
std::filesystem::path ReserveRestoreSession();

} // namespace project
