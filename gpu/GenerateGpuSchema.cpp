#include <algorithm>
#include <charconv>
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
    std::string Binding;
    bool IsPushConstant = false;
    std::vector<Field> Fields;
};

struct EnumValue {
    std::string Name, Value;
};

struct ConstantDef {
    std::string Name, Type;
};

struct ConstantGroup {
    std::string Name;
    std::vector<ConstantDef> Constants;
};

struct EnumDef {
    std::string Name, Type;
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
    FunctionConstants,
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

bool ParseSchema(
    const std::filesystem::path &path, std::vector<Binding> &bindings, std::vector<EnumDef> &enums,
    std::vector<ConstantGroup> &constant_groups, std::vector<StructDef> &structs
) {
    std::ifstream in{path};
    if (!in) return false;

    std::optional<Binding> current_binding;
    std::optional<EnumDef> current_enum;
    std::optional<EnumValue> current_enum_value;
    std::optional<ConstantGroup> current_group;
    std::optional<ConstantDef> current_constant;
    std::optional<StructDef> current_struct;
    std::optional<Field> current_field;
    bool in_values{}, in_constants{}, in_fields{};

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
    auto commit_constant = [&]() {
        if (!current_constant) return;
        if (!current_group) Fail("Invalid function constant entry in schema.");
        if (current_constant->Name.empty() || current_constant->Type.empty()) Fail("Invalid function constant entry in schema.");

        current_group->Constants.emplace_back(*current_constant);
        current_constant.reset();
    };
    auto commit_group = [&]() {
        if (!current_group) return;
        commit_constant();
        if (current_group->Name.empty() || current_group->Constants.empty()) Fail("Invalid function constant group in schema.");

        constant_groups.emplace_back(*current_group);
        current_group.reset();
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
    size_t values_parent_indent{0}, constants_parent_indent{0}, fields_parent_indent{0};
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
        if (section == Section::FunctionConstants && in_constants && indent_count <= constants_parent_indent) {
            commit_constant();
            in_constants = false;
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
            commit_constant();
            commit_group();
            commit_binding();
            section = Section::Bindings;
            continue;
        }
        if (trimmed.starts_with("enums:")) {
            commit_field();
            commit_struct();
            commit_binding();
            commit_constant();
            commit_group();
            commit_enum_value();
            commit_enum();
            section = Section::Enums;
            continue;
        }
        if (trimmed.starts_with("function_constants:")) {
            commit_enum_value();
            commit_enum();
            commit_field();
            commit_struct();
            commit_binding();
            commit_constant();
            commit_group();
            section = Section::FunctionConstants;
            continue;
        }
        if (trimmed.starts_with("structs:")) {
            commit_enum_value();
            commit_enum();
            commit_constant();
            commit_group();
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
            } else if (section == Section::FunctionConstants) {
                if (in_constants) commit_constant();
                else commit_group();
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
        } else if (section == Section::FunctionConstants) {
            if (key == "constants" && value.empty()) {
                in_constants = true;
                constants_parent_indent = indent_count;
                continue;
            }
            if (in_constants) {
                if (!current_constant) current_constant.emplace();
                if (key == "name") current_constant->Name = value;
                else if (key == "type") current_constant->Type = value;
                else Fail("Unknown function constant key: " + std::string{key});
            } else {
                if (!current_group) current_group.emplace();
                if (key == "name") current_group->Name = value;
                else Fail("Unknown function constant group key: " + std::string{key});
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
    if (section == Section::FunctionConstants && in_constants) commit_constant();
    if (section == Section::Structs && in_fields) commit_field();
    commit_enum();
    commit_group();
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

std::optional<std::string_view> MslBuiltinTypeFor(std::string_view type) {
    if (type == "u8") return "uchar";
    if (type == "u32") return "uint";
    if (type == "float") return "float";
    if (type == "vec2") return "packed_float2";
    if (type == "vec3") return "packed_float3";
    if (type == "vec4" || type == "quat") return "packed_float4";
    if (type == "uvec2") return "packed_uint2";
    if (type == "uvec4") return "packed_uint4";
    if (type == "mat3") return "packed_float3x3";
    if (type == "mat4") return "packed_float4x4";
    return {};
}

std::optional<std::string_view> CppBuiltinTypeFor(std::string_view type) {
    if (type == "u8") return "uint8_t";
    if (type == "u32") return "uint32_t";
    if (type == "quat") return "quat";
    if (type == "float" || type == "vec2" || type == "uvec2" || type == "uvec4" || type == "vec3" || type == "vec4" || type == "mat3" || type == "mat4") return type;
    return {};
}

struct Layout {
    size_t Size, Align;
};

std::optional<Layout> BuiltinLayoutFor(std::string_view type) {
    if (type == "u8") return Layout{1, 1};
    if (type == "u32" || type == "float") return Layout{4, 4};
    if (type == "vec2" || type == "uvec2") return Layout{8, 4};
    if (type == "vec3") return Layout{12, 4};
    if (type == "vec4" || type == "uvec4" || type == "quat") return Layout{16, 4};
    if (type == "mat3") return Layout{36, 4};
    if (type == "mat4") return Layout{64, 4};
    return {};
}

constexpr size_t AlignUp(size_t offset, size_t align) { return (offset + align - 1) / align * align; }

std::optional<std::string_view> CppTypeFor(std::string_view type, const std::vector<StructDef> &structs, const std::vector<EnumDef> &enums) {
    if (const auto builtin = CppBuiltinTypeFor(type)) return builtin;
    if (IsEnumType(type, enums)) return type;
    if (IsStructType(type, structs)) return type;
    return {};
}

std::optional<std::string_view> MslTypeFor(std::string_view type, const std::vector<StructDef> &structs, const std::vector<EnumDef> &enums) {
    if (const auto builtin = MslBuiltinTypeFor(type)) return builtin;
    if (const auto it = find_if(enums, [&](const auto &def) { return def.Name == type; }); it != enums.end()) {
        return MslBuiltinTypeFor(it->Type);
    }
    if (IsStructType(type, structs)) return type;
    return {};
}

struct FieldLayout {
    std::string Name;
    size_t Offset;
};

struct StructLayout {
    std::vector<FieldLayout> Fields;
    size_t Size, Align;
};

const StructDef *FindStruct(std::string_view name, const std::vector<StructDef> &structs) {
    const auto it = find_if(structs, [&](const auto &def) { return def.Name == name; });
    return it == structs.end() ? nullptr : &*it;
}

StructLayout ComputeLayout(const StructDef &, const std::vector<StructDef> &, const std::vector<EnumDef> &);

Layout LayoutFor(std::string_view type, const std::vector<StructDef> &structs, const std::vector<EnumDef> &enums) {
    if (const auto builtin = BuiltinLayoutFor(type)) return *builtin;
    if (const auto it = find_if(enums, [&](const auto &def) { return def.Name == type; }); it != enums.end()) {
        if (const auto underlying = BuiltinLayoutFor(it->Type)) return *underlying;
        Fail("Unknown enum underlying type: " + it->Type);
    }
    if (const auto *def = FindStruct(type, structs)) {
        const auto layout = ComputeLayout(*def, structs, enums);
        return {layout.Size, layout.Align};
    }
    Fail("Unknown type in layout: " + std::string{type});
}

StructLayout ComputeLayout(const StructDef &def, const std::vector<StructDef> &structs, const std::vector<EnumDef> &enums) {
    StructLayout out{{}, 0, 1};
    size_t offset = 0;
    for (const auto &field : def.Fields) {
        const auto spec = ParseType(field.Type);
        const auto element = LayoutFor(spec.Base, structs, enums);
        offset = AlignUp(offset, element.Align);
        out.Fields.emplace_back(field.Name, offset);
        offset += element.Size * spec.ArraySize.value_or(1);
        out.Align = std::max(out.Align, element.Align);
    }
    out.Size = AlignUp(offset, out.Align);
    return out;
}

bool IsUniform(std::string_view kind) { return kind == "Uniform" || kind == "UniformDynamic"; }

std::string BufferIndexName(std::string_view binding_name) {
    constexpr std::string_view Suffix{"UBO"};
    const auto stem = binding_name.ends_with(Suffix) ? binding_name.substr(0, binding_name.size() - Suffix.size()) : binding_name;
    return "BufferIndex_" + std::string{stem};
}

std::optional<std::string_view> BindKindEnum(std::string_view kind) {
    if (kind == "Uniform") return "Uniform";
    if (kind == "UniformDynamic") return "UniformDynamic";
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

std::string CppDefaultValue(std::string value) {
    constexpr std::string_view FloatMax{"FloatMax"};
    for (size_t pos; (pos = value.find(FloatMax)) != std::string::npos;) {
        value.replace(pos, FloatMax.size(), "std::numeric_limits<float>::max()");
    }
    return value;
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

void EmitLayoutAsserts(std::ostream &out, std::string_view name, const StructLayout &layout, std::string_view offset_of) {
    for (const auto &field : layout.Fields) {
        out << "static_assert(" << offset_of << "(" << name << ", " << field.Name << ") == " << field.Offset
            << ", \"" << name << "::" << field.Name << " offset\");\n";
    }
    out << "static_assert(sizeof(" << name << ") == " << layout.Size << ", \"" << name << " size\");\n";
}

void EmitEnum(
    const EnumDef &def,
    const std::filesystem::path &msl_dir,
    const std::filesystem::path &cpp_dir,
    const std::filesystem::path &schema_relative_path
) {
    const auto msl_path = msl_dir / (def.Name + ".metal");
    const auto cpp_path = cpp_dir / (def.Name + ".h");
    std::ofstream msl_out{msl_path, std::ios::binary};
    std::ofstream cpp_out{cpp_path, std::ios::binary};
    if (!msl_out || !cpp_out) Fail("Failed to open enum output files for: " + def.Name);

    const auto msl_guard = ToMacroName(def.Name, "MSL");
    const auto enum_token = ToIdentifier(def.Name);
    const auto cpp_type = CppBuiltinTypeFor(def.Type);
    const auto msl_type = MslBuiltinTypeFor(def.Type);
    if (!cpp_type || !msl_type) Fail("Unknown enum underlying type: " + def.Type);

    msl_out << "#ifndef " << msl_guard << "\n"
            << "#define " << msl_guard << "\n\n"
            << GeneratedComment(schema_relative_path);
    for (const auto &value : def.Values) {
        msl_out << "constant " << *msl_type << " " << enum_token << "_" << ToIdentifier(value.Name) << " = " << value.Value << ";\n";
    }
    msl_out << "\n#endif\n";

    cpp_out << "#pragma once\n\n"
            << GeneratedComment(schema_relative_path)
            << "#include <cstdint>\n\n"
            << "enum class " << def.Name << " : " << *cpp_type << " {\n";
    for (const auto &value : def.Values) cpp_out << "    " << value.Name << " = " << value.Value << ",\n";
    cpp_out << "};\n";
}

void EmitFunctionConstants(
    const ConstantGroup &def,
    const std::filesystem::path &msl_dir,
    const std::filesystem::path &cpp_dir,
    const std::filesystem::path &schema_relative_path
) {
    const auto msl_path = msl_dir / (def.Name + ".metal");
    const auto cpp_path = cpp_dir / (def.Name + ".h");
    std::ofstream msl_out{msl_path, std::ios::binary};
    std::ofstream cpp_out{cpp_path, std::ios::binary};
    if (!msl_out || !cpp_out) Fail("Failed to open function constant output files for: " + def.Name);

    msl_out << "#ifndef " << ToMacroName(def.Name, "MSL") << "\n"
            << "#define " << ToMacroName(def.Name, "MSL") << "\n\n"
            << GeneratedComment(schema_relative_path)
            << "#include \"MslPrelude.metal\"\n\n";
    for (size_t i = 0; i < def.Constants.size(); ++i) {
        const auto &constant = def.Constants[i];
        const auto msl_type = constant.Type == "bool" ? std::optional<std::string_view>{"bool"} : MslBuiltinTypeFor(constant.Type);
        if (!msl_type) Fail("Unknown function constant type: " + constant.Type);
        msl_out << "constant " << *msl_type << " " << constant.Name << " [[function_constant(" << i << ")]];\n";
    }
    msl_out << "\n#endif\n";

    cpp_out << "#pragma once\n\n"
            << GeneratedComment(schema_relative_path)
            << "#include <cstdint>\n\n"
            << "enum class " << def.Name << " : uint32_t {\n";
    for (size_t i = 0; i < def.Constants.size(); ++i) cpp_out << "    " << def.Constants[i].Name << " = " << i << ",\n";
    cpp_out << "};\n";
}

int main(int argc, char **argv) {
    if (argc != 4) return 1;

    const std::filesystem::path build_dir{argv[1]}, source_dir{argv[2]}, schema_relative_path{argv[3]}, schema_path{source_dir / schema_relative_path};
    const auto msl_dir{build_dir / "shaders"}, cpp_dir{build_dir / "gpu"};

    std::error_code fs_error;
    std::filesystem::create_directories(msl_dir, fs_error);
    if (fs_error) return 1;
    std::filesystem::create_directories(cpp_dir, fs_error);
    if (fs_error) return 1;

    std::vector<Binding> bindings;
    std::vector<EnumDef> enums;
    std::vector<ConstantGroup> constant_groups;
    std::vector<StructDef> structs;
    if (!ParseSchema(schema_path, bindings, enums, constant_groups, structs)) return 1;
    for (const auto &group : constant_groups) EmitFunctionConstants(group, msl_dir, cpp_dir, schema_relative_path);

    const auto bindless_header_path{cpp_dir / "BindlessBindings.h"};
    std::ofstream bindless_header{bindless_header_path, std::ios::binary};
    if (!bindless_header) return 1;

    bindless_header << "#pragma once\n\n"
                    << GeneratedComment(schema_relative_path)
                    << "#include <array>\n"
                    << "#include <cstdint>\n"
                    << "#include <string_view>\n\n"
                    << "enum class BindKind : uint8_t {\n"
                    << "    Uniform,\n"
                    << "    UniformDynamic,\n"
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
    // Shared capacities keep CPU and GPU slot indices identical.
    // Corpus peaks are 40 buffers, 85 samplers, and 14 images; arena-backed resources do not scale with scene size.
    struct KindCapacity {
        std::string_view Kind;
        uint32_t Capacity;
    };
    constexpr std::array<KindCapacity, 5> Capacities{{
        {"Uniform", 1},
        {"UniformDynamic", 1},
        {"Image", 1024},
        {"Sampler", 1024},
        {"Buffer", 256},
    }};
    const auto capacity_for = [&](std::string_view kind) {
        const auto it = find_if(Capacities, [&](const auto &c) { return c.Kind == kind; });
        if (it == Capacities.end()) Fail("Unknown binding kind: " + std::string{kind});
        return it->Capacity;
    };

    bindless_header << "}};\n";

    struct EntryLayout {
        uint32_t Offset, Stride, Capacity;
    };
    std::vector<EntryLayout> entry_layouts;
    uint32_t table_offset = 0;
    for (const auto &binding : bindings) {
        const auto capacity = capacity_for(binding.Kind);
        if (IsUniform(binding.Kind)) {
            entry_layouts.emplace_back(0u, 0u, capacity);
            continue;
        }
        const uint32_t stride = binding.Kind == "Sampler" ? 16u : 8u;
        entry_layouts.emplace_back(table_offset, stride, capacity);
        table_offset += stride * capacity;
    }
    bindless_header << "\n// Byte layout of the Tier-2 argument buffer, matching the generated MSL BindlessSetT.\n"
                    << "struct BindlessEntryLayout {\n"
                    << "    uint32_t Offset;\n"
                    << "    uint32_t Stride;\n"
                    << "    uint32_t Capacity;\n"
                    << "};\n\n"
                    << "constexpr std::array<BindlessEntryLayout, SlotTypeCount> BindlessLayout{{\n";
    for (size_t i = 0; i < entry_layouts.size(); ++i) {
        const auto &e = entry_layouts[i];
        bindless_header << "    BindlessEntryLayout{" << e.Offset << ", " << e.Stride << ", " << e.Capacity << "}, // " << bindings[i].Name << "\n";
    }
    bindless_header << "}};\n\n"
                    << "constexpr uint32_t BindlessTableSize{" << table_offset << "};\n";

    bindless_header << "\nconstexpr uint32_t BufferIndex_Bindless{0};\n"
                    << "constexpr uint32_t BufferIndex_PushConstants{1};\n";
    uint32_t uniform_index = 2;
    for (const auto &binding : bindings) {
        if (!IsUniform(binding.Kind)) continue;
        bindless_header << "constexpr uint32_t " << BufferIndexName(binding.Name) << "{" << uniform_index++ << "};\n";
    }

    bindless_header << "\nconstexpr uint32_t SlotCapacityFor(BindKind kind) {\n"
                    << "    switch (kind) {\n";
    for (const auto &c : Capacities) bindless_header << "        case BindKind::" << c.Kind << ": return " << c.Capacity << ";\n";
    bindless_header << "    }\n    return 0;\n}\n";

    std::ofstream bindless_msl{msl_dir / "BindlessBindings.metal", std::ios::binary};
    if (!bindless_msl) return 1;
    bindless_msl << "#ifndef BINDLESS_BINDINGS_MSL\n"
                 << "#define BINDLESS_BINDINGS_MSL\n\n"
                 << GeneratedComment(schema_relative_path)
                 << "#include \"MslPrelude.metal\"\n\n";
    for (size_t i = 0; i < bindings.size(); ++i) {
        bindless_msl << "constant uint BINDING_" << bindings[i].Name << " = " << i << ";\n";
    }
    bindless_msl << "\nconstant uint BufferIndex_Bindless = 0;\n"
                 << "constant uint BufferIndex_PushConstants = 1;\n";
    uniform_index = 2;
    for (const auto &binding : bindings) {
        if (!IsUniform(binding.Kind)) continue;
        bindless_msl << "constant uint " << BufferIndexName(binding.Name) << " = " << uniform_index++ << ";\n";
    }
    bindless_msl << "\ntemplate<typename ImageT>\nstruct BindlessSetT {\n";
    for (const auto &binding : bindings) {
        const auto capacity = capacity_for(binding.Kind);
        if (binding.Kind == "Buffer") {
            bindless_msl << "    device const uchar *" << binding.Name << "[" << capacity << "];\n";
        } else if (binding.Kind == "Image") {
            bindless_msl << "    ImageT " << binding.Name << "[" << capacity << "];\n";
        } else if (binding.Kind == "Sampler") {
            bindless_msl << "    BindlessSampler" << (binding.Name == "CubeSampler" ? "Cube" : "2D") << " " << binding.Name << "[" << capacity << "];\n";
        }
    }
    bindless_msl << "};\n\n";
    bindless_msl << "template<typename ImageT> struct BindlessLayoutCheck {\n";
    for (size_t i = 0; i < bindings.size(); ++i) {
        if (IsUniform(bindings[i].Kind)) continue;
        bindless_msl << "    static_assert(__builtin_offsetof(BindlessSetT<ImageT>, " << bindings[i].Name << ") == " << entry_layouts[i].Offset
                     << ", \"" << bindings[i].Name << " argument buffer offset\");\n";
    }
    bindless_msl << "    static_assert(sizeof(BindlessSetT<ImageT>) == " << table_offset << ", \"argument buffer size\");\n"
                 << "};\n\n"
                 << "using BindlessSet = BindlessSetT<texture2d<float, access::read>>;\n"
                 << "using BindlessSetImageWrite = BindlessSetT<texture2d<float, access::write>>;\n"
                 << "using BindlessSetImageUint = BindlessSetT<texture2d<uint, access::read_write>>;\n"
                 << "template struct BindlessLayoutCheck<texture2d<float, access::read>>;\n"
                 << "template struct BindlessLayoutCheck<texture2d<float, access::write>>;\n"
                 << "template struct BindlessLayoutCheck<texture2d<uint, access::read_write>>;\n\n"
                 << "// Writable slots cast away the table's const view.\n"
                 << "#define BindlessBuffer(T, table, slot) reinterpret_cast<device const T *>((table)[(slot)])\n"
                 << "#define BindlessBufferMutable(T, table, slot) reinterpret_cast<device T *>(const_cast<device uchar *>((table)[(slot)]))\n"
                 << "\n#endif\n";

    for (const auto &def : enums) EmitEnum(def, msl_dir, cpp_dir, schema_relative_path);

    for (const auto &def : structs) {
        const auto msl_path = msl_dir / (def.Name + ".metal");
        const auto cpp_path = cpp_dir / (def.Name + ".h");
        std::ofstream msl_out{msl_path, std::ios::binary}, cpp_out{cpp_path, std::ios::binary};
        if (!msl_out || !cpp_out) return 1;

        // Compile-time assertions keep the generated C++ and MSL layouts identical.
        const auto layout = ComputeLayout(def, structs, enums);
        const auto msl_guard = ToMacroName(def.Name, "MSL");
        msl_out << "#ifndef " << msl_guard << "\n"
                << "#define " << msl_guard << "\n\n"
                << GeneratedComment(schema_relative_path)
                << "#include \"MslPrelude.metal\"\n";
        std::vector<std::string_view> msl_includes;
        for (const auto &field : def.Fields) {
            const auto spec = ParseType(field.Type);
            const auto is_nested = (IsStructType(spec.Base, structs) || IsEnumType(spec.Base, enums)) && spec.Base != def.Name;
            if (is_nested && find(msl_includes, spec.Base) == msl_includes.end()) msl_includes.emplace_back(spec.Base);
        }
        for (const auto &include : msl_includes) msl_out << "#include \"" << include << ".metal\"\n";
        msl_out << "\nstruct " << def.Name << " {\n";
        for (const auto &field : def.Fields) {
            const auto spec = ParseType(field.Type);
            if (const auto msl_type = MslTypeFor(spec.Base, structs, enums)) {
                msl_out << "    " << *msl_type << " " << field.Name;
                if (spec.ArraySize) msl_out << "[" << *spec.ArraySize << "]";
                msl_out << ";\n";
            } else Fail("Unknown type: " + field.Type);
        }
        msl_out << "};\n";
        EmitLayoutAsserts(msl_out, def.Name, layout, "__builtin_offsetof");
        msl_out << "\n#endif\n";

        bool needs_array{false}, needs_cstdint{false},
            needs_uvec2{false}, needs_uvec4{false}, needs_vec2{false}, needs_vec3{false}, needs_vec4{false},
            needs_mat3{false}, needs_mat4{false}, needs_quat{false}, needs_slots{false}, needs_range{false}, needs_limits{false};
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
            if (field.DefaultValue.find("InvalidSlot") != std::string_view::npos) needs_slots = true;
            if (field.DefaultValue.find("InvalidOffset") != std::string_view::npos) needs_range = true;
            if (field.DefaultValue.find("FloatMax") != std::string_view::npos) needs_limits = true;
            if (IsStructType(spec.Base, structs) && spec.Base != def.Name) {
                if (find(cpp_includes, spec.Base) == cpp_includes.end()) cpp_includes.emplace_back(spec.Base);
            }
            if (IsEnumType(spec.Base, enums) && spec.Base != def.Name) {
                if (find(cpp_includes, spec.Base) == cpp_includes.end()) cpp_includes.emplace_back(spec.Base);
            }
        }

        cpp_out << "#pragma once\n\n"
                << GeneratedComment(schema_relative_path);
        cpp_out << "#include <cstddef>\n";
        if (needs_array) cpp_out << "#include <array>\n";
        if (needs_cstdint) cpp_out << "#include <cstdint>\n";
        if (needs_limits) cpp_out << "#include <limits>\n";
        if (needs_uvec2) cpp_out << "#include \"numeric/vec2.h\"\n";
        if (needs_uvec4) cpp_out << "#include \"numeric/vec4.h\"\n";
        if (needs_vec2) cpp_out << "#include \"numeric/vec2.h\"\n";
        if (needs_mat3) cpp_out << "#include \"numeric/mat3.h\"\n";
        if (needs_mat4) cpp_out << "#include \"numeric/mat4.h\"\n";
        if (needs_vec3) cpp_out << "#include \"numeric/vec3.h\"\n";
        if (needs_vec4) cpp_out << "#include \"numeric/vec4.h\"\n";
        if (needs_quat) cpp_out << "#include \"numeric/quat.h\"\n";
        if (needs_slots) cpp_out << "#include \"metal/Slots.h\"\n";
        if (needs_range) cpp_out << "#include \"Range.h\"\n";
        for (const auto &include : cpp_includes) cpp_out << "#include \"gpu/" << include << ".h\"\n";
        if (needs_cstdint || needs_limits || needs_vec2 || needs_mat3 || needs_mat4 || needs_vec3 || needs_vec4 || needs_quat || needs_slots || needs_range || !cpp_includes.empty()) cpp_out << "\n";

        cpp_out << "struct " << def.Name << " {\n";
        for (const auto &field : def.Fields) {
            const auto spec = ParseType(field.Type);
            if (const auto cpp_type = CppTypeFor(spec.Base, structs, enums)) {
                cpp_out << "    ";
                const auto default_value = CppDefaultValue(field.DefaultValue);
                if (spec.ArraySize) {
                    cpp_out << "std::array<" << *cpp_type << ", " << *spec.ArraySize << "> " << field.Name << '{';
                    if (!default_value.empty()) {
                        for (size_t i = 0; i < *spec.ArraySize; ++i) cpp_out << (i == 0 ? "" : ", ") << default_value;
                    }
                    cpp_out << "};\n";
                } else {
                    cpp_out << *cpp_type << " " << field.Name << '{' << default_value << "};\n";
                }
            } else Fail("Unknown type: " + field.Type);
        }
        cpp_out << "    bool operator==(const " << def.Name << " &) const = default;\n";
        cpp_out << "};\n";
        EmitLayoutAsserts(cpp_out, def.Name, layout, "offsetof");
    }

    return 0;
}
