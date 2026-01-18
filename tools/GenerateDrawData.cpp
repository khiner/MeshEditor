#include "DrawDataFields.h"

#include <fstream>

std::string_view CppTypeFor(FieldType type) {
    switch (type) {
        case FieldType::U32: return "uint32_t";
    }
}

std::string_view GlslTypeFor(FieldType type) {
    switch (type) {
        case FieldType::U32: return "uint";
    }
}

int main(int argc, char **argv) {
    if (argc != 3) return 1;

    std::ofstream glsl_out{argv[1], std::ios::binary};
    if (!glsl_out) return 1;

    std::ofstream header_out{argv[2], std::ios::binary};
    if (!header_out) return 1;

    glsl_out << "#ifndef DRAW_DATA_GLSL\n"
             << "#define DRAW_DATA_GLSL\n\n"
             << "// Generated from src/DrawDataFields.h. Do not edit by hand.\n\n"
             << "struct DrawData {\n";
    for (const auto &field : DrawDataFields) {
        glsl_out << "    " << GlslTypeFor(field.Type) << " " << field.Name << ";\n";
    }
    glsl_out << "};\n\n#endif\n";

    header_out << "#pragma once\n\n"
               << "#include \"vulkan/Slots.h\"\n\n"
               << "struct DrawData {\n";
    for (const auto &field : DrawDataFields) {
        header_out << "    " << CppTypeFor(field.Type) << " " << field.Name << "{" << field.DefaultValue << "};\n";
    }
    header_out << "};\n\n"
               << "static_assert(sizeof(DrawData) % 16 == 0, \"DrawData must be 16-byte aligned.\");\n";
    return 0;
}
