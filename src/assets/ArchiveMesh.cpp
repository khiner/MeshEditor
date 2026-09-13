#include "assets/ArchiveMesh.h"

#include "File.h"
#include "project/Assets.h"

#include <sstream>
#include <tiny_obj_loader.h>
#include <utility>
#include <vector>

namespace {
// Parse mtllib filenames with escaped spaces and backslashes.
std::vector<std::string> MaterialFiles(std::string_view line) {
    std::vector<std::string> files;
    std::string name;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '\\' && i + 1 < line.size() && (line[i + 1] == ' ' || line[i + 1] == '\\')) name += line[++i];
        else if (c != ' ') name += c;
        else if (!name.empty()) files.push_back(std::exchange(name, {}));
    }
    if (!name.empty()) files.push_back(std::move(name));
    return files;
}

std::expected<std::filesystem::path, std::string> ArchiveMaterial(project::Assets &assets, const std::filesystem::path &path, const std::filesystem::path &mesh_dir) {
    auto source = File::ReadAsString(path);
    if (!source) return std::unexpected{source.error()};
    std::istringstream lines{*source};
    std::string output, line;
    while (std::getline(lines, line)) {
        const auto start = line.find_first_not_of(" \t");
        const auto end = start == line.npos ? line.npos : line.find_first_of(" \t", start);
        const auto key = start == line.npos ? std::string_view{} : std::string_view{line}.substr(start, end - start);
        if (key == "map_Kd" || key == "norm" || key == "bump" || key == "map_bump" || key == "map_Bump") {
            std::string name;
            tinyobj::texture_option_t options{};
            if (end != line.npos && tinyobj::ParseTextureNameAndOption(&name, &options, line.c_str() + end)) {
                // MeshImport resolves MTL texture names against the OBJ directory.
                while (!name.empty() && (name.back() == '\r' || name.back() == ' ' || name.back() == '\t')) name.pop_back();
                const auto stored = assets.Store((mesh_dir / name).lexically_normal());
                if (!stored) return std::unexpected{stored.error()};
                line = std::string{key} + " ../" + stored->native().substr(7);
            }
        }
        output += line + '\n';
    }
    return assets.Store(path.filename().string(), std::as_bytes(std::span{output.data(), output.size()}));
}
} // namespace

std::expected<std::filesystem::path, std::string> ArchiveMesh(project::Assets &assets, const std::filesystem::path &path) {
    if (project::Assets::IsReference(path) || path.extension() != ".obj") return assets.Store(path);
    auto source = File::ReadAsString(path);
    if (!source) return std::unexpected{source.error()};
    std::istringstream lines{*source};
    std::string output, line;
    while (std::getline(lines, line)) {
        const auto start = line.find_first_not_of(" \t");
        if (start != line.npos && line.compare(start, 6, "mtllib") == 0 && line.size() > start + 6 && (line[start + 6] == ' ' || line[start + 6] == '\t')) {
            auto names = line.substr(start + 7);
            while (!names.empty() && (names.back() == '\r' || names.back() == ' ' || names.back() == '\t')) names.pop_back();
            line.clear();
            // Use the first existing material file, matching tinyobj.
            for (const auto &name : MaterialFiles(names)) {
                const auto file = path.parent_path() / name;
                if (!std::filesystem::is_regular_file(file)) continue;
                const auto stored = ArchiveMaterial(assets, file, path.parent_path());
                if (!stored) return std::unexpected{stored.error()};
                line = "mtllib ../";
                for (const char c : stored->native().substr(7)) {
                    if (c == ' ' || c == '\\') line += '\\';
                    line += c;
                }
                break;
            }
        }
        output += line + '\n';
    }
    return assets.Store(path.filename().string(), std::as_bytes(std::span{output.data(), output.size()}));
}
