#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

using std::ranges::any_of, std::ranges::find, std::ranges::find_if, std::ranges::find_if_not;

struct Binding {
    std::string Name, Kind;
};

struct Field {
    std::string Name, Type, DefaultValue;
};

struct StructDef {
    std::string Name;
    std::string Binding; // References a binding name (for uniforms)
    bool IsPushConstant = false;
    std::vector<Field> Fields;
};

struct EnumValue {
    std::string Name, Value;
};

struct EnumDef {
    std::string Name, Type; // Underlying type (e.g. u32)
    std::vector<EnumValue> Values;
};

[[noreturn]] void Fail(std::string_view message) {
    std::cerr << message << "\n";
    std::exit(1);
}

std::string GeneratedComment(const std::filesystem::path &schema_relative_path) {
    return "// Generated from " + schema_relative_path.string() + ". Do not edit by hand.\n\n";
}

std::string_view Trim(std::string_view s) {
    constexpr auto is_space = [](unsigned char c) { return std::isspace(c); };
    const auto first = find_if_not(s, is_space);
    const auto last = find_if_not(s | std::views::reverse, is_space).base();
    if (first >= last) return {};
    return {first, last};
}

std::string_view StripComment(std::string_view s) {
    const auto pos = s.find('#');
    return pos == std::string_view::npos ? s : s.substr(0, pos);
}

struct ParsedKeyValue {
    std::string_view Key, Value;
};

std::optional<ParsedKeyValue> ParseKeyValue(std::string_view line) {
    const auto pos = line.find(':');
    if (pos == std::string_view::npos) return std::nullopt;

    auto key = Trim(line.substr(0, pos));
    auto value = Trim(line.substr(pos + 1));
    if (value.size() >= 2) {
        const char quote = value.front();
        if ((quote == '"' || quote == '\'') && value.back() == quote) {
            value = value.substr(1, value.size() - 2);
        }
    }
    if (!value.empty() && value.front() == '[' && value.back() == ']') {
        value = value.substr(1, value.size() - 2);
    }
    if (key.empty()) return std::nullopt;
    return ParsedKeyValue{key, value};
}

enum class Section {
    None,
    Bindings,
    Enums,
    Structs,
};

struct TypeSpec {
    std::string_view Base;
    std::optional<size_t> ArraySize;
};

TypeSpec ParseType(std::string_view type) {
    const auto open = type.find('[');
    if (open == std::string_view::npos) return {.Base = Trim(type), .ArraySize = std::nullopt};

    const auto close = type.find(']', open + 1);
    if (close == std::string_view::npos || close != type.size() - 1) return {.Base = Trim(type), .ArraySize = std::nullopt};

    const auto size_view = Trim(type.substr(open + 1, close - open - 1));
    size_t size = 0;
    const auto *begin = size_view.data();
    const auto *end = size_view.data() + size_view.size();
    if (auto [ptr, ec] = std::from_chars(begin, end, size); ec != std::errc{} || ptr != end) {
        return {.Base = Trim(type), .ArraySize = std::nullopt};
    }
    return {.Base = Trim(type.substr(0, open)), .ArraySize = size};
}

bool ParseSchema(const std::filesystem::path &path, std::vector<Binding> &bindings, std::vector<EnumDef> &enums, std::vector<StructDef> &structs) {
    std::ifstream in{path};
    if (!in) return false;

    std::optional<Binding> current_binding;
    std::optional<EnumDef> current_enum;
    std::optional<EnumValue> current_enum_value;
    std::optional<StructDef> current_struct;
    std::optional<Field> current_field;
    bool in_values{}, in_fields{};

    auto commit_binding = [&]() {
        if (!current_binding) return;
        if (current_binding->Name.empty() || current_binding->Kind.empty()) Fail("Invalid binding entry in schema.");
        bindings.emplace_back(*current_binding);
        current_binding.reset();
    };
    auto commit_enum_value = [&]() {
        if (!current_enum_value) return;
        if (!current_enum) Fail("Invalid enum value entry in schema.");
        if (current_enum_value->Name.empty()) Fail("Invalid enum value entry in schema.");

        if (current_enum_value->Value.empty()) current_enum_value->Value = std::to_string(current_enum->Values.size());
        current_enum->Values.emplace_back(*current_enum_value);
        current_enum_value.reset();
    };
    auto commit_enum = [&]() {
        if (!current_enum) return;
        commit_enum_value();
        if (current_enum->Name.empty() || current_enum->Type.empty() || current_enum->Values.empty()) Fail("Invalid enum entry in schema.");

        enums.emplace_back(*current_enum);
        current_enum.reset();
    };
    auto commit_field = [&]() {
        if (!current_field) return;
        if (!current_struct) Fail("Invalid field entry in schema.");
        if (current_field->Name.empty() || current_field->Type.empty()) Fail("Invalid field entry in schema.");

        current_struct->Fields.emplace_back(*current_field);
        current_field.reset();
    };
    auto commit_struct = [&]() {
        if (!current_struct) return;
        commit_field();
        if (current_struct->Name.empty() || current_struct->Fields.empty()) Fail("Invalid struct entry in schema.");

        structs.emplace_back(*current_struct);
        current_struct.reset();
    };

    Section section = Section::None;
    size_t values_parent_indent{0}, fields_parent_indent{0};
    for (std::string line; std::getline(in, line);) {
        const auto stripped = StripComment(line);
        const auto indent = stripped.find_first_not_of(' ');
        const auto indent_count = indent == std::string_view::npos ? stripped.size() : indent;
        auto trimmed = Trim(stripped);
        if (trimmed.empty()) continue;

        if (section == Section::Enums && in_values && indent_count <= values_parent_indent) {
            commit_enum_value();
            in_values = false;
        }
        if (section == Section::Structs && in_fields && indent_count <= fields_parent_indent) {
            commit_field();
            in_fields = false;
        }

        if (trimmed.starts_with("bindings:")) {
            commit_field();
            commit_struct();
            commit_enum_value();
            commit_enum();
            commit_binding();
            section = Section::Bindings;
            continue;
        }
        if (trimmed.starts_with("enums:")) {
            commit_field();
            commit_struct();
            commit_binding();
            commit_enum_value();
            commit_enum();
            section = Section::Enums;
            continue;
        }
        if (trimmed.starts_with("structs:")) {
            commit_enum_value();
            commit_enum();
            commit_field();
            commit_struct();
            commit_binding();
            section = Section::Structs;
            continue;
        }

        if (trimmed.starts_with("- ")) {
            if (section == Section::Bindings) {
                commit_binding();
            } else if (section == Section::Enums) {
                if (in_values) commit_enum_value();
                else commit_enum();
            } else if (section == Section::Structs) {
                if (in_fields) commit_field();
                else commit_struct();
            } else Fail("Item defined outside of a section.");
            trimmed = Trim(trimmed.substr(2));
        }

        const auto parsed = ParseKeyValue(trimmed);
        if (!parsed) Fail("Unrecognized schema line: " + std::string{trimmed});
        const auto key = parsed->Key;
        const auto value = parsed->Value;

        if (section == Section::Bindings) {
            if (!current_binding) current_binding.emplace();
            if (key == "name") current_binding->Name = value;
            else if (key == "kind") current_binding->Kind = value;
            else Fail("Unknown bindings key: " + std::string{key});
        } else if (section == Section::Enums) {
            if (key == "values" && value.empty()) {
                in_values = true;
                values_parent_indent = indent_count;
                continue;
            }
            if (in_values) {
                if (!current_enum_value) current_enum_value.emplace();
                if (key == "name") current_enum_value->Name = value;
                else if (key == "value") current_enum_value->Value = value;
                else Fail("Unknown enum value key: " + std::string{key});
            } else {
                if (!current_enum) current_enum.emplace();
                if (key == "name") current_enum->Name = value;
                else if (key == "type") current_enum->Type = value;
                else Fail("Unknown enum key: " + std::string{key});
            }
        } else if (section == Section::Structs) {
            if (key == "fields" && value.empty()) {
                in_fields = true;
                fields_parent_indent = indent_count;
                continue;
            }
            if (in_fields) {
                if (!current_field) current_field.emplace();
                if (key == "name") current_field->Name = value;
                else if (key == "type") current_field->Type = value;
                else if (key == "default") current_field->DefaultValue = value;
                else Fail("Unknown field key: " + std::string{key});
            } else {
                if (!current_struct) current_struct.emplace();
                if (key == "name") current_struct->Name = value;
                else if (key == "binding") current_struct->Binding = value;
                else if (key == "push_constant") current_struct->IsPushConstant = (value == "true");
                else Fail("Unknown struct key: " + std::string{key});
            }
        } else Fail("Item defined outside of a section.");
    }

    if (section == Section::Enums && in_values) commit_enum_value();
    if (section == Section::Structs && in_fields) commit_field();
    commit_enum();
    commit_struct();
    commit_binding();
    return true;
}

bool IsStructType(std::string_view type, const std::vector<StructDef> &structs) {
    return any_of(structs, [&](const auto &def) { return def.Name == type; });
}
bool IsEnumType(std::string_view type, const std::vector<EnumDef> &enums) {
    return any_of(enums, [&](const auto &def) { return def.Name == type; });
}

enum class TargetLang {
    Cpp,
    Glsl,
};

std::optional<std::string_view> BuiltinTypeFor(std::string_view type, TargetLang target) {
    if (type == "u8") return "uint8_t";
    if (type == "u32") return target == TargetLang::Cpp ? "uint32_t" : "uint";
    if (type == "quat") return target == TargetLang::Cpp ? "quat" : "vec4";
    if (type == "float" || type == "vec2" || type == "uvec2" || type == "uvec4" || type == "vec3" || type == "vec4" || type == "mat3" || type == "mat4") return type;
    return {};
}

std::optional<std::string_view> CppTypeFor(std::string_view type, const std::vector<StructDef> &structs, const std::vector<EnumDef> &enums) {
    if (const auto builtin = BuiltinTypeFor(type, TargetLang::Cpp)) return builtin;
    if (IsEnumType(type, enums)) return type;
    if (IsStructType(type, structs)) return type;
    return {};
}
std::optional<std::string_view> GlslTypeFor(std::string_view type, const std::vector<StructDef> &structs, const std::vector<EnumDef> &enums) {
    if (const auto builtin = BuiltinTypeFor(type, TargetLang::Glsl)) return builtin;
    if (const auto it = find_if(enums, [&](const auto &def) { return def.Name == type; }); it != enums.end()) {
        return BuiltinTypeFor(it->Type, TargetLang::Glsl);
    }
    if (IsStructType(type, structs)) return type;
    return {};
}

std::optional<std::string_view> BindKindEnum(std::string_view kind) {
    if (kind == "Uniform") return "Uniform";
    if (kind == "Image") return "Image";
    if (kind == "Sampler") return "Sampler";
    if (kind == "Buffer") return "Buffer";
    return std::nullopt;
}

bool BindingExists(const std::vector<Binding> &bindings, std::string_view name) {
    return any_of(bindings, [&](const auto &b) { return b.Name == name; });
}

std::string ToMacroName(std::string_view name, std::string_view suffix) {
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

std::string ToIdentifier(std::string_view name) {
    std::string out;
    out.reserve(name.size());
    for (const char ch : name) {
        const auto uch = static_cast<unsigned char>(ch);
        out.push_back(std::isalnum(uch) ? ch : '_');
    }
    return out;
}

void EmitEnum(
    const EnumDef &def,
    const std::filesystem::path &glsl_dir,
    const std::filesystem::path &cpp_dir,
    const std::filesystem::path &schema_relative_path
) {
    const auto glsl_path = glsl_dir / (def.Name + ".glsl");
    const auto cpp_path = cpp_dir / (def.Name + ".h");
    std::ofstream glsl_out{glsl_path, std::ios::binary};
    std::ofstream cpp_out{cpp_path, std::ios::binary};
    if (!glsl_out || !cpp_out) Fail("Failed to open enum output files for: " + def.Name);

    const auto guard = ToMacroName(def.Name, "GLSL");
    const auto enum_token = ToIdentifier(def.Name);
    const auto cpp_type = BuiltinTypeFor(def.Type, TargetLang::Cpp);
    const auto glsl_type = BuiltinTypeFor(def.Type, TargetLang::Glsl);
    if (!cpp_type || !glsl_type) Fail("Unknown enum underlying type: " + def.Type);

    glsl_out << "#ifndef " << guard << "\n"
             << "#define " << guard << "\n\n"
             << GeneratedComment(schema_relative_path);
    for (const auto &value : def.Values) {
        glsl_out << "const " << *glsl_type << " " << enum_token << "_" << ToIdentifier(value.Name) << " = " << value.Value << ";\n";
    }
    glsl_out << "\n#endif\n";

    cpp_out << "#pragma once\n\n"
            << GeneratedComment(schema_relative_path)
            << "#include <cstdint>\n\n"
            << "enum class " << def.Name << " : " << *cpp_type << " {\n";
    for (const auto &value : def.Values) cpp_out << "    " << value.Name << " = " << value.Value << ",\n";
    cpp_out << "};\n";
}

// args: <binary_dir> <source_dir> <schema_relative_path>
int main(int argc, char **argv) {
    if (argc != 4) return 1;

    const std::filesystem::path build_dir{argv[1]};
    const std::filesystem::path source_dir{argv[2]};
    const std::filesystem::path schema_relative_path{argv[3]};
    const std::filesystem::path schema_path = source_dir / schema_relative_path;
    const auto glsl_dir = build_dir / "shaders";
    const auto cpp_dir = build_dir / "gpu";

    std::error_code fs_error;
    std::filesystem::create_directories(glsl_dir, fs_error);
    if (fs_error) return 1;
    std::filesystem::create_directories(cpp_dir, fs_error);
    if (fs_error) return 1;

    std::vector<Binding> bindings;
    std::vector<EnumDef> enums;
    std::vector<StructDef> structs;
    if (!ParseSchema(schema_path, bindings, enums, structs)) return 1;

    const auto bindless_glsl_path = glsl_dir / "BindlessBindings.glsl";
    const auto bindless_header_path = cpp_dir / "BindlessBindings.h";
    std::ofstream bindless_glsl{bindless_glsl_path, std::ios::binary};
    std::ofstream bindless_header{bindless_header_path, std::ios::binary};
    if (!bindless_glsl || !bindless_header) return 1;

    bindless_glsl << "#ifndef BINDLESS_BINDINGS_GLSL\n\n"
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
        if (const auto kind = BindKindEnum(binding.Kind)) {
            bindless_header << "    BindingDef{BindKind::" << *kind << ", \"" << binding.Name << "\"},\n";
        } else {
            Fail("Unknown binding kind: " + binding.Kind);
        }
    }
    bindless_header << "}};\n";

    for (const auto &def : enums) EmitEnum(def, glsl_dir, cpp_dir, schema_relative_path);

    for (const auto &def : structs) {
        const auto glsl_path = glsl_dir / (def.Name + ".glsl");
        const auto cpp_path = cpp_dir / (def.Name + ".h");
        std::ofstream glsl_out{glsl_path, std::ios::binary};
        std::ofstream cpp_out{cpp_path, std::ios::binary};
        if (!glsl_out || !cpp_out) return 1;

        const auto guard = ToMacroName(def.Name, "GLSL");
        glsl_out << "#ifndef " << guard << "\n"
                 << "#define " << guard << "\n\n"
                 << GeneratedComment(schema_relative_path);
        std::vector<std::string_view> glsl_includes;
        for (const auto &field : def.Fields) {
            const auto spec = ParseType(field.Type);
            if (IsStructType(spec.Base, structs) && spec.Base != def.Name) {
                if (find(glsl_includes, spec.Base) == glsl_includes.end()) glsl_includes.emplace_back(spec.Base);
            }
        }
        // Determine if we need extensions and BindlessBindings.glsl for layout declarations
        const bool is_uniform = !def.Binding.empty();
        if (is_uniform) glsl_out << "#extension GL_EXT_scalar_block_layout : require\n\n";
        if (is_uniform || def.IsPushConstant) glsl_out << "#include \"BindlessBindings.glsl\"\n";
        for (const auto &include : glsl_includes) glsl_out << "#include \"" << include << ".glsl\"\n";
        if (is_uniform || def.IsPushConstant || !glsl_includes.empty()) glsl_out << "\n";

        // Generate layout block or plain struct
        if (def.IsPushConstant) {
            glsl_out << "layout(push_constant) uniform " << def.Name;
        } else if (is_uniform) {
            if (!BindingExists(bindings, def.Binding)) Fail("Unknown binding: " + def.Binding);
            glsl_out << "layout(set = 0, binding = BINDING_" << def.Binding << ", scalar) uniform " << def.Name << "Block";
        } else {
            glsl_out << "struct " << def.Name;
        }
        glsl_out << " {\n";

        for (const auto &field : def.Fields) {
            const auto spec = ParseType(field.Type);
            if (const auto glsl_type = GlslTypeFor(spec.Base, structs, enums)) {
                glsl_out << "    " << *glsl_type << " " << field.Name;
                if (spec.ArraySize) glsl_out << "[" << *spec.ArraySize << "]";
                glsl_out << ";\n";
            } else Fail("Unknown type: " + field.Type);
        }

        if (def.IsPushConstant) glsl_out << "} pc;";
        else if (is_uniform) glsl_out << "} " << def.Name << ";";
        else glsl_out << "};";
        glsl_out << "\n\n#endif\n";

        bool needs_array{false}, needs_cstdint{false},
            needs_uvec2{false}, needs_uvec4{false}, needs_vec2{false}, needs_vec3{false}, needs_vec4{false},
            needs_mat3{false}, needs_mat4{false}, needs_quat{false}, needs_slots{false};
        std::vector<std::string_view> cpp_includes;
        for (const auto &field : def.Fields) {
            const auto spec = ParseType(field.Type);
            if (spec.ArraySize) needs_array = true;
            if (spec.Base == "u32" || spec.Base == "u8") needs_cstdint = true;
            if (spec.Base == "uvec2") needs_uvec2 = true;
            if (spec.Base == "uvec4") needs_uvec4 = true;
            if (spec.Base == "vec2") needs_vec2 = true;
            if (spec.Base == "vec3") needs_vec3 = true;
            if (spec.Base == "vec4") needs_vec4 = true;
            if (spec.Base == "mat3") needs_mat3 = true;
            if (spec.Base == "mat4") needs_mat4 = true;
            if (spec.Base == "quat") needs_quat = true;
            if (field.DefaultValue.find("InvalidSlot") != std::string_view::npos ||
                field.DefaultValue.find("InvalidOffset") != std::string_view::npos) needs_slots = true;
            if (IsStructType(spec.Base, structs) && spec.Base != def.Name) {
                if (find(cpp_includes, spec.Base) == cpp_includes.end()) cpp_includes.emplace_back(spec.Base);
            }
            if (IsEnumType(spec.Base, enums) && spec.Base != def.Name) {
                if (find(cpp_includes, spec.Base) == cpp_includes.end()) cpp_includes.emplace_back(spec.Base);
            }
        }

        cpp_out << "#pragma once\n\n"
                << GeneratedComment(schema_relative_path);
        if (needs_array) cpp_out << "#include <array>\n";
        if (needs_cstdint) cpp_out << "#include <cstdint>\n";
        if (needs_uvec2 || needs_uvec4) cpp_out << "#include \"glm/glm.hpp\"\n";
        if (needs_vec2) cpp_out << "#include \"numeric/vec2.h\"\n";
        if (needs_mat3) cpp_out << "#include \"numeric/mat3.h\"\n";
        if (needs_mat4) cpp_out << "#include \"numeric/mat4.h\"\n";
        if (needs_vec3) cpp_out << "#include \"numeric/vec3.h\"\n";
        if (needs_vec4) cpp_out << "#include \"numeric/vec4.h\"\n";
        if (needs_quat) cpp_out << "#include \"numeric/quat.h\"\n";
        if (needs_slots) cpp_out << "#include \"vulkan/Slots.h\"\n";
        for (const auto &include : cpp_includes) cpp_out << "#include \"gpu/" << include << ".h\"\n";
        if (needs_cstdint || needs_vec2 || needs_mat3 || needs_mat4 || needs_vec3 || needs_vec4 || needs_quat || needs_slots || !cpp_includes.empty()) cpp_out << "\n";

        cpp_out << "struct " << def.Name << " {\n";
        for (const auto &field : def.Fields) {
            const auto spec = ParseType(field.Type);
            if (const auto cpp_type = CppTypeFor(spec.Base, structs, enums)) {
                cpp_out << "    ";
                if (spec.ArraySize) {
                    cpp_out << "std::array<" << *cpp_type << ", " << *spec.ArraySize << "> " << field.Name << "{};\n";
                } else {
                    cpp_out << *cpp_type << " " << field.Name << '{'
                            << (field.DefaultValue.empty() ? std::string_view{} : std::string_view{field.DefaultValue})
                            << "};\n";
                }
            } else Fail("Unknown type: " + field.Type);
        }
        cpp_out << "    bool operator==(const " << def.Name << " &) const = default;\n";
        cpp_out << "};\n";
    }

    return 0;
}
