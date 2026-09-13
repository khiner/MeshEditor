#include "gltf/ArchiveSource.h"

#include "File.h"
#include "project/Assets.h"

#include <fastgltf/types.hpp>
#include <simdjson.h>

#include <cstring>
#include <format>

namespace gltf {
namespace {
std::string Quote(std::string_view value) {
    std::string out{"\""};
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') out += '\\';
        if (c < 0x20) out += std::format("\\u{:04x}", c);
        else out += char(c);
    }
    return out + '"';
}

std::string Uri(const std::filesystem::path &path) {
    std::string out;
    for (const unsigned char c : path.generic_string()) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '/' || c == '.' || c == '-' || c == '_' || c == '~') out += char(c);
        else out += std::format("%{:02X}", c);
    }
    return out;
}
} // namespace

std::expected<std::filesystem::path, std::string> ArchiveSource(project::Assets &assets, const std::filesystem::path &path) {
    if (project::Assets::IsReference(path)) return path;
    auto bytes = File::Read(path);
    if (!bytes) return std::unexpected{bytes.error()};
    const bool binary = path.extension() == ".glb";
    const auto word = [&](size_t offset) {
        uint32_t value;
        std::memcpy(&value, bytes->data() + offset, 4);
        return value;
    };
    if (binary && (bytes->size() < 20 || word(0) != 0x46546c67 || word(4) != 2 || word(8) != bytes->size() || word(16) != 0x4e4f534a || word(12) > bytes->size() - 20)) return std::unexpected{"Invalid GLB header"};
    const size_t json_offset = binary ? 20 : 0;
    const size_t json_size = binary ? word(12) : bytes->size();
    simdjson::dom::parser parser;
    simdjson::dom::object root;
    if (parser.parse(reinterpret_cast<const char *>(bytes->data() + json_offset), json_size).get(root)) return std::unexpected{"Invalid glTF JSON"};
    bool changed = false;
    std::string json{"{"};
    for (const auto field : root) {
        if (json.size() > 1) json += ',';
        json += Quote(field.key) + ':';
        if (field.key != "buffers" && field.key != "images") {
            json += simdjson::minify(field.value);
            continue;
        }
        simdjson::dom::array items;
        if (field.value.get(items)) return std::unexpected{"Invalid glTF resource list"};
        json += '[';
        bool first_item = true;
        for (const auto item : items) {
            simdjson::dom::object object;
            if (item.get(object)) return std::unexpected{"Invalid glTF resource"};
            if (!first_item) json += ',';
            first_item = false;
            json += '{';
            bool first_field = true;
            for (const auto property : object) {
                if (!first_field) json += ',';
                first_field = false;
                json += Quote(property.key) + ':';
                if (property.key != "uri") {
                    json += simdjson::minify(property.value);
                    continue;
                }
                std::string_view source;
                if (property.value.get(source)) return std::unexpected{"Invalid glTF resource URI"};
                const fastgltf::URI uri{source};
                if (!uri.isLocalPath()) {
                    json += Quote(source);
                    continue;
                }
                const auto file = uri.fspath().is_absolute() ? uri.fspath() : path.parent_path() / uri.fspath();
                const auto stored = assets.Store(file.lexically_normal());
                if (!stored) return std::unexpected{stored.error()};
                const auto relative = std::filesystem::path{".."} / stored->native().substr(7);
                json += Quote(Uri(relative));
                changed = true;
            }
            json += '}';
        }
        json += ']';
    }
    json += '}';
    if (!changed) return assets.Store(path.filename().string(), *bytes);
    if (!binary) return assets.Store(path.filename().string(), std::as_bytes(std::span{json.data(), json.size()}));
    while (json.size() % 4) json += ' ';
    const uint32_t size = uint32_t(20 + json.size() + bytes->size() - json_offset - json_size);
    const uint32_t header[]{0x46546c67, 2, size, uint32_t(json.size()), 0x4e4f534a};
    std::vector<std::byte> output(size);
    std::memcpy(output.data(), header, sizeof(header));
    std::memcpy(output.data() + 20, json.data(), json.size());
    std::memcpy(output.data() + 20 + json.size(), bytes->data() + json_offset + json_size, bytes->size() - json_offset - json_size);
    return assets.Store(path.filename().string(), output);
}
} // namespace gltf
