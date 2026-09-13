#pragma once

#include <expected>
#include <filesystem>
#include <string>

namespace project {
struct Assets;
}

// Retain OBJ material and texture dependencies alongside the source mesh.
std::expected<std::filesystem::path, std::string> ArchiveMesh(project::Assets &, const std::filesystem::path &);
