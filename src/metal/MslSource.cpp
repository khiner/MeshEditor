#include "metal/MslSource.h"

#include <algorithm>
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

std::string_view Trim(std::string_view line) {
    const auto first = line.find_first_not_of(" \t");
    return first == std::string_view::npos ? std::string_view{} : line.substr(first);
}

// Returns the preprocessor directive name, such as "include" or "endif", or empty for other lines.
std::string_view Directive(std::string_view line) {
    line = Trim(line);
    if (!line.starts_with('#')) return {};
    line = Trim(line.substr(1));
    return line.substr(0, line.find_first_of(" \t"));
}

std::optional<std::string_view> IncludeTarget(std::string_view line) {
    if (Directive(line) != "include") return {};
    const auto open = line.find('"');
    if (open == std::string_view::npos) return {};
    const auto close = line.find('"', open + 1);
    if (close == std::string_view::npos) return {};
    return line.substr(open + 1, close - open - 1);
}

// Conditionals on __METAL_VERSION__ select one side of a header shared with C++.
// Lines on the CPU side pass through verbatim, since the Metal preprocessor discards them.
struct Conditional {
    bool SelectsSide;
    bool CpuSide;
};

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
    std::vector<Conditional> conditionals;
    size_t line_number = 0;
    for (size_t pos = 0; pos <= text.size();) {
        const auto end = text.find('\n', pos);
        const std::string_view line{text.data() + pos, (end == std::string::npos ? text.size() : end) - pos};
        ++line_number;
        const auto directive = Directive(line);
        if (directive == "ifdef" || directive == "ifndef" || directive == "if") {
            const bool selects_side = directive != "if" && line.find("__METAL_VERSION__") != std::string_view::npos;
            conditionals.emplace_back(selects_side, selects_side && directive == "ifndef");
        } else if (directive == "else" && !conditionals.empty() && conditionals.back().SelectsSide) {
            conditionals.back().CpuSide = !conditionals.back().CpuSide;
        } else if (directive == "endif" && !conditionals.empty()) {
            conditionals.pop_back();
        }
        const bool cpu_side = std::ranges::any_of(conditionals, [](const auto &c) { return c.CpuSide; });
        const auto target = cpu_side ? std::nullopt : IncludeTarget(line);
        if (target) {
            Append(out, root, *target);
            // Resume parent-file numbering after each include.
            out.Text += std::format("#line {}\n", line_number + 1);
        } else if (directive == "pragma" && Trim(line).ends_with("once")) {
            // Repeated includes are already omitted, and the flattened source is one file.
            out.Text.push_back('\n');
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
