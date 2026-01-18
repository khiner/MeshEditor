#include "BindlessBindingFields.h"

#include <fstream>
#include <string_view>

std::string_view KindToCppEnumName(BindKind kind) {
    switch (kind) {
        case BindKind::Uniform: return "Uniform";
        case BindKind::Image: return "Image";
        case BindKind::Sampler: return "Sampler";
        case BindKind::Buffer: return "Buffer";
    }
}

int main(int argc, char **argv) {
    if (argc != 3) return 1;

    std::ofstream glsl_out{argv[1], std::ios::binary};
    if (!glsl_out) return 1;

    std::ofstream header_out{argv[2], std::ios::binary};
    if (!header_out) return 1;

    glsl_out << "#ifndef BINDLESS_BINDINGS_GLSL\n"
             << "#define BINDLESS_BINDINGS_GLSL\n\n"
             << "// Generated from src/BindlessBindingFields.h. Do not edit by hand.\n\n";
    for (size_t i = 0; i < BindlessBindingFields.size(); ++i) {
        glsl_out << "const uint BINDING_" << BindlessBindingFields[i].Name << " = " << i << ";\n";
    }
    glsl_out << "const uint BINDING_COUNT = " << BindlessBindingFields.size() << ";\n\n#endif\n";

    header_out << "#pragma once\n\n"
               << "#include \"BindlessBindingFields.h\"\n\n"
               << "#include <array>\n"
               << "#include <cstddef>\n"
               << "#include <cstdint>\n"
               << "#include <string_view>\n\n"
               << "enum class SlotType : uint8_t {\n";
    for (const auto &field : BindlessBindingFields) {
        header_out << "    " << field.Name << ",\n";
    }
    header_out << "    Count\n};\n\n"
               << "constexpr size_t SlotTypeCount{static_cast<size_t>(SlotType::Count)};\n\n"
               << "struct BindingDef {\n"
               << "    BindKind Kind;\n"
               << "    std::string_view Name;\n"
               << "};\n\n"
               << "constexpr std::array<BindingDef, SlotTypeCount> BindingDefs{{\n";
    for (const auto &field : BindlessBindingFields) {
        header_out << "    BindingDef{BindKind::" << KindToCppEnumName(field.Kind) << ", \"" << field.Name << "\"},\n";
    }
    header_out << "}};\n";

    return 0;
}
