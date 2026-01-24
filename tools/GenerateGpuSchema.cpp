#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

struct Binding {
    std::string Name;
    std::string Kind;
};

struct Field {
    std::string Name;
    std::string Type;
    std::string DefaultValue;
};

struct StructDef {
    std::string Name;
    std::vector<Field> Fields;
};

[[noreturn]] void Fail(std::string_view message) {
    std::cerr << message << "\n";
    std::exit(1);
}

std::string GeneratedComment(const std::filesystem::path &schema_relative_path) {
    return "// Generated from " + schema_relative_path.string() + ". Do not edit by hand.\n\n";
}

std::string_view Trim(std::string_view s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

std::string_view StripComment(std::string_view s) {
    const auto pos = s.find('#');
    return pos == std::string_view::npos ? s : s.substr(0, pos);
}

bool ParseKeyValue(std::string_view line, std::string &key, std::string &value) {
    const auto pos = line.find(':');
    if (pos == std::string_view::npos) return false;

    key = std::string{Trim(line.substr(0, pos))};
    value = std::string{Trim(line.substr(pos + 1))};
    if (value.size() >= 2) {
        const char quote = value.front();
        if ((quote == '"' || quote == '\'') && value.back() == quote) {
            value = value.substr(1, value.size() - 2);
        }
    }
    if (!value.empty() && value.front() == '[' && value.back() == ']') {
        value = value.substr(1, value.size() - 2);
    }
    return !key.empty();
}

enum class Section {
    None,
    Bindings,
    Structs,
};

bool ParseSchema(const std::filesystem::path &path, std::vector<Binding> &bindings, std::vector<StructDef> &structs) {
    std::ifstream in{path};
    if (!in) return false;

    Section section = Section::None;
    Binding current_binding{};
    StructDef current_struct{};
    Field current_field{};
    bool has_binding = false;
    bool has_struct = false;
    bool has_field = false;
    bool in_fields = false;
    size_t fields_parent_indent = 0;

    auto commit_binding = [&]() {
        if (!has_binding) return;
        if (current_binding.Name.empty() || current_binding.Kind.empty()) {
            std::cerr << "Invalid binding entry in schema.\n";
            std::exit(1);
        }
        bindings.push_back(current_binding);
        current_binding = {};
        has_binding = false;
    };

    auto commit_field = [&]() {
        if (!has_field) return;
        if (current_field.Name.empty() || current_field.Type.empty()) {
            std::cerr << "Invalid field entry in schema.\n";
            std::exit(1);
        }
        current_struct.Fields.push_back(current_field);
        current_field = {};
        has_field = false;
    };

    auto commit_struct = [&]() {
        if (!has_struct) return;
        commit_field();
        if (current_struct.Name.empty() || current_struct.Fields.empty()) {
            std::cerr << "Invalid struct entry in schema.\n";
            std::exit(1);
        }
        structs.push_back(current_struct);
        current_struct = {};
        has_struct = false;
    };

    for (std::string line; std::getline(in, line);) {
        const auto stripped = StripComment(line);
        const auto indent = stripped.find_first_not_of(' ');
        const auto indent_count = indent == std::string_view::npos ? stripped.size() : indent;

        auto trimmed = Trim(stripped);
        if (trimmed.empty()) continue;

        if (section == Section::Structs && in_fields && indent_count <= fields_parent_indent) {
            commit_field();
            in_fields = false;
        }

        if (trimmed.starts_with("bindings:")) {
            commit_field();
            commit_struct();
            commit_binding();
            section = Section::Bindings;
            continue;
        }
        if (trimmed.starts_with("structs:")) {
            commit_field();
            commit_struct();
            commit_binding();
            section = Section::Structs;
            continue;
        }

        if (trimmed.starts_with("- ")) {
            if (section == Section::Bindings) {
                commit_binding();
            } else if (section == Section::Structs) {
                if (in_fields) commit_field();
                else commit_struct();
            } else Fail("Item defined outside of a section.");
            trimmed = Trim(trimmed.substr(2));
        }

        std::string key, value;
        if (!ParseKeyValue(trimmed, key, value)) Fail(std::string{"Unrecognized schema line: "} + std::string{trimmed});

        if (section == Section::Bindings) {
            has_binding = true;
            if (key == "name") current_binding.Name = value;
            else if (key == "kind") current_binding.Kind = value;
            else Fail("Unknown bindings key: " + key);
        } else if (section == Section::Structs) {
            if (key == "fields" && value.empty()) {
                in_fields = true;
                fields_parent_indent = indent_count;
                continue;
            }
            if (in_fields) {
                has_field = true;
                if (key == "name") current_field.Name = value;
                else if (key == "type") current_field.Type = value;
                else if (key == "default") current_field.DefaultValue = value;
                else Fail("Unknown field key: " + key);
            } else {
                has_struct = true;
                if (key == "name") current_struct.Name = value;
                else Fail("Unknown struct key: " + key);
            }
        } else Fail("Item defined outside of a section.");
    }

    if (section == Section::Structs && in_fields) commit_field();
    commit_struct();
    commit_binding();
    return true;
}

std::optional<std::string_view> CppTypeFor(std::string_view type) {
    if (type == "u32") return "uint32_t";
    if (type == "float") return "float";
    if (type == "vec3") return "vec3";
    if (type == "vec4") return "vec4";
    if (type == "mat4") return "mat4";
    return {};
}
std::optional<std::string_view> GlslTypeFor(std::string_view type) {
    if (type == "u32") return "uint";
    if (type == "float") return "float";
    if (type == "vec3") return "vec3";
    if (type == "vec4") return "vec4";
    if (type == "mat4") return "mat4";
    return {};
}

std::optional<std::string_view> BindKindEnum(std::string_view kind) {
    if (kind == "Uniform") return "Uniform";
    if (kind == "Image") return "Image";
    if (kind == "Sampler") return "Sampler";
    if (kind == "Buffer") return "Buffer";
    return std::nullopt;
}

std::string ToMacroName(const std::string &name, std::string_view suffix) {
    std::string out;
    out.reserve(name.size() + suffix.size() + 1);
    for (const char ch : name) {
        if (std::isalnum(static_cast<unsigned char>(ch))) out.push_back(char(std::toupper(ch)));
        else out.push_back('_');
    }
    if (!suffix.empty()) {
        out.push_back('_');
        out.append(suffix);
    }
    return out;
}

// args: <binary_dir> <source_dir> <schema_relative_path>
int main(int argc, char **argv) {
    if (argc != 4) return 1;

    const std::filesystem::path build_dir{argv[1]};
    const std::filesystem::path source_dir{argv[2]};
    const std::filesystem::path schema_relative_path{argv[3]};
    const std::filesystem::path schema_path = source_dir / schema_relative_path;
    const auto glsl_dir = build_dir / "shaders";
    const auto cpp_dir = build_dir / "generated";

    std::error_code fs_error;
    std::filesystem::create_directories(glsl_dir, fs_error);
    if (fs_error) return 1;
    std::filesystem::create_directories(cpp_dir, fs_error);
    if (fs_error) return 1;

    std::vector<Binding> bindings;
    std::vector<StructDef> structs;
    if (!ParseSchema(schema_path, bindings, structs)) return 1;

    const auto bindless_glsl_path = glsl_dir / "BindlessBindings.glsl";
    const auto bindless_header_path = cpp_dir / "BindlessBindings.h";
    std::ofstream bindless_glsl{bindless_glsl_path, std::ios::binary};
    std::ofstream bindless_header{bindless_header_path, std::ios::binary};
    if (!bindless_glsl || !bindless_header) return 1;

    bindless_glsl << "#ifndef BINDLESS_BINDINGS_GLSL\n"
                  << "#define BINDLESS_BINDINGS_GLSL\n\n"
                  << GeneratedComment(schema_relative_path);
    for (size_t i = 0; i < bindings.size(); ++i) {
        bindless_glsl << "const uint BINDING_" << bindings[i].Name << " = " << i << ";\n";
    }
    bindless_glsl << "\n\n#endif\n";

    bindless_header << "#pragma once\n\n"
                    << GeneratedComment(schema_relative_path)
                    << "#include <array>\n"
                    << "#include <cstddef>\n"
                    << "#include <cstdint>\n"
                    << "#include <string_view>\n\n"
                    << "enum class BindKind : uint8_t {\n"
                    << "    Uniform,\n"
                    << "    Image,\n"
                    << "    Sampler,\n"
                    << "    Buffer,\n"
                    << "};\n\n"
                    << "enum class SlotType : uint8_t {\n";
    for (const auto &binding : bindings) {
        bindless_header << "    " << binding.Name << ",\n";
    }
    bindless_header << "    Count\n};\n\n"
                    << "constexpr size_t SlotTypeCount{static_cast<size_t>(SlotType::Count)};\n\n"
                    << "struct BindingDef {\n"
                    << "    BindKind Kind;\n"
                    << "    std::string_view Name;\n"
                    << "};\n\n"
                    << "constexpr std::array<BindingDef, SlotTypeCount> BindingDefs{{\n";
    for (const auto &binding : bindings) {
        if (const auto kind = BindKindEnum(binding.Kind); !kind) {
            std::cerr << "Unknown binding kind: " << binding.Kind << "\n";
            return 1;
        } else {
            bindless_header << "    BindingDef{BindKind::" << *kind << ", \"" << binding.Name << "\"},\n";
        }
    }
    bindless_header << "}};\n";

    for (const auto &def : structs) {
        const auto glsl_path = glsl_dir / (def.Name + ".glsl");
        const auto cpp_path = cpp_dir / (def.Name + ".h");
        std::ofstream glsl_out{glsl_path, std::ios::binary};
        std::ofstream cpp_out{cpp_path, std::ios::binary};
        if (!glsl_out || !cpp_out) return 1;

        const auto guard = ToMacroName(def.Name, "GLSL");
        glsl_out << "#ifndef " << guard << "\n"
                 << "#define " << guard << "\n\n"
                 << GeneratedComment(schema_relative_path)
                 << "struct " << def.Name << " {\n";
        for (const auto &field : def.Fields) {
            if (const auto glsl_type = GlslTypeFor(field.Type); !glsl_type) {
                std::cerr << "Unknown type: " << field.Type << "\n";
                return 1;
            } else {
                glsl_out << "    " << *glsl_type << " " << field.Name << ";\n";
            }
        }
        glsl_out << "};\n\n#endif\n";

        bool needs_cstdint = false;
        bool needs_vec3 = false;
        bool needs_vec4 = false;
        bool needs_mat4 = false;
        bool needs_slots = false;
        for (const auto &field : def.Fields) {
            if (field.Type == "u32") needs_cstdint = true;
            if (field.Type == "vec3") needs_vec3 = true;
            if (field.Type == "vec4") needs_vec4 = true;
            if (field.Type == "mat4") needs_mat4 = true;
            if (field.DefaultValue.find("InvalidSlot") != std::string::npos) needs_slots = true;
        }

        cpp_out << "#pragma once\n\n"
                << GeneratedComment(schema_relative_path);
        if (needs_cstdint) cpp_out << "#include <cstdint>\n";
        if (needs_mat4) cpp_out << "#include \"numeric/mat4.h\"\n";
        if (needs_vec3) cpp_out << "#include \"numeric/vec3.h\"\n";
        if (needs_vec4) cpp_out << "#include \"numeric/vec4.h\"\n";
        if (needs_slots) cpp_out << "#include \"vulkan/Slots.h\"\n";
        if (needs_cstdint || needs_mat4 || needs_vec3 || needs_vec4 || needs_slots) cpp_out << "\n";

        cpp_out << "struct " << def.Name << " {\n";
        for (const auto &field : def.Fields) {
            if (const auto cpp_type = CppTypeFor(field.Type); !cpp_type) {
                std::cerr << "Unknown type: " << field.Type << "\n";
                return 1;
            } else if (field.DefaultValue.empty()) {
                cpp_out << "    " << *cpp_type << " " << field.Name << ";\n";
            } else {
                cpp_out << "    " << *cpp_type << " " << field.Name << "{" << field.DefaultValue << "};\n";
            }
        }
        cpp_out << "};\n";
    }

    return 0;
}
