#pragma once

#include <expected>
#include <filesystem>
#include <string>

namespace project {
struct Assets;
}

namespace gltf {
// Store the source and its external buffers/images, rewriting resource URIs to the stored files.
std::expected<std::filesystem::path, std::string> ArchiveSource(project::Assets &, const std::filesystem::path &);
} // namespace gltf
