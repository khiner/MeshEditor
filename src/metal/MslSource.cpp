#include "metal/MslSource.h"

#include <format>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace msl {
namespace {
std::string ReadFile(const std::filesystem::path &path) {
    const std::ifstream in{path, std::ios::binary};
    if (!in) throw std::runtime_error(std::format("Failed to open shader source '{}'", path.string()));
    std::ostringstream out;
    out << in.rdbuf();
    return std::move(out).str();
}

std::optional<std::string_view> IncludeTarget(std::string_view line) {
    const auto first = line.find_first_not_of(" \t");
    if (first == std::string_view::npos) return {};
    line.remove_prefix(first);
    if (!line.starts_with("#include")) return {};
    const auto open = line.find('"');
    if (open == std::string_view::npos) return {};
    const auto close = line.find('"', open + 1);
    if (close == std::string_view::npos) return {};
    return line.substr(open + 1, close - open - 1);
}

void Append(Source &out, const std::filesystem::path &root, const std::filesystem::path &relative) {
    for (const auto &seen : out.Files) {
        if (seen == relative) return;
    }
    const auto path = root / relative;
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(std::format("Failed to resolve shader include '{}'", relative.string()));
    }
    out.Files.emplace_back(relative);

    const auto text = ReadFile(path);
    // Use `#line` to map compiler diagnostics to included source files.
    // Metal accepts line numbers but represents the file name through the preceding marker comment.
    out.Text += std::format("\n// ---- {} ----\n", relative.string());
    size_t line_number = 0;
    for (size_t pos = 0; pos <= text.size();) {
        const auto end = text.find('\n', pos);
        const std::string_view line{text.data() + pos, (end == std::string::npos ? text.size() : end) - pos};
        ++line_number;
        if (const auto target = IncludeTarget(line)) {
            Append(out, root, *target);
            // Resume parent-file numbering after each include.
            out.Text += std::format("#line {}\n", line_number + 1);
        } else {
            out.Text.append(line);
            out.Text.push_back('\n');
        }
        if (end == std::string::npos) break;
        pos = end + 1;
    }
}
} // namespace

Source Load(const std::filesystem::path &root, const std::filesystem::path &relative_path, const std::vector<std::string> &defines) {
    Source out;
    for (const auto &define : defines) out.Text += std::format("#define {}\n", define);
    Append(out, root, relative_path);
    return out;
}
} // namespace msl
