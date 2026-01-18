#include "BindlessBindings.h"

#include <fstream>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) {
        return 1;
    }
    const char *output_path = argv[1];
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        return 1;
    }

    out << "#ifndef BINDLESS_BINDINGS_GLSL\n";
    out << "#define BINDLESS_BINDINGS_GLSL\n\n";
    out << "// Generated from src/BindlessBindings.h. Do not edit by hand.\n\n";
    for (size_t i = 0; i < BindingDefs.size(); ++i) {
        out << "const uint BINDING_" << BindingDefs[i].Name << " = " << i << ";\n";
    }
    out << "const uint BINDING_COUNT = " << BindingDefs.size() << ";\n";
    out << "\n#endif\n";
    return 0;
}
