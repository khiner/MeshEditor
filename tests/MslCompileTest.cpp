// Compiles all shaders or the file names supplied on the command line.
// The shared GPU headers assert their struct sizes on the MSL side as well.
#include "TestPaths.h"
#include "metal/MetalCpp.h"
#include "metal/MslSource.h"

#include <algorithm>
#include <print>
#include <string_view>

namespace fs = std::filesystem;

namespace {
struct Failure {
    std::string Name, Message;
};
} // namespace
int main(int argc, char **argv) {
    if (argc >= 2 && std::string_view{argv[1]} == "--check-offline") {
        if (argc < 4) {
            std::println(stderr, "Usage: {} --check-offline SHADER_DIRECTORY SOURCE...", argv[0]);
            return 1;
        }
        const fs::path root = argv[2];
        try {
            for (int i = 3; i < argc; ++i) {
                const fs::path source = argv[i];
                auto binary = root / source;
                binary.replace_extension(".metallib");
                if (!msl::Load(root, source).CanLoadBinary(root, binary)) {
                    std::println(stderr, "MeshEditor would recompile {} from source", (root / source).string());
                    return 1;
                }
            }
        } catch (const std::exception &error) {
            std::println(stderr, "{}", error.what());
            return 1;
        }
        return 0;
    }
    auto *pool = NS::AutoreleasePool::alloc()->init();
    auto *device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::println(stderr, "No Metal device.");
        return 1;
    }

    const auto shaders_dir = ShadersDir(argv[0]);
    if (!fs::is_directory(shaders_dir)) {
        std::println(stderr, "No shaders directory at {}", shaders_dir.string());
        return 1;
    }

    std::vector<fs::path> sources{argv + 1, argv + argc};
    if (sources.empty()) {
        for (const auto &entry : fs::directory_iterator{shaders_dir}) {
            if (entry.path().extension() == ".metal") sources.push_back(entry.path().filename());
        }
    }
    std::ranges::sort(sources);

    std::vector<Failure> failures;
    uint32_t functions = 0;
    const auto names = [](MTL::Library *library) {
        std::vector<std::string> result;
        auto *functions = library->functionNames();
        result.reserve(functions->count());
        for (NS::UInteger i = 0u; i < functions->count(); ++i)
            result.emplace_back(functions->object<NS::String>(i)->utf8String());
        std::ranges::sort(result);
        return result;
    };
    for (const auto &name : sources) {
        std::string text;
        try {
            text = msl::Load(shaders_dir, name).Text;
        } catch (const std::exception &e) {
            failures.emplace_back(name.string(), e.what());
            continue;
        }
        NS::Error *error = nullptr;
        auto *library = device->newLibrary(NS::String::string(text.c_str(), NS::UTF8StringEncoding), msl::CompileOptions().get(), &error);
        if (!library) {
            failures.emplace_back(name.string(), error ? error->localizedDescription()->utf8String() : "unknown compile failure");
            continue;
        }
        functions += uint32_t(library->functionNames()->count());
        auto binary = shaders_dir / name;
        binary.replace_extension(".metallib");
        error = nullptr;
        auto *compiled = device->newLibrary(NS::String::string(binary.string().c_str(), NS::UTF8StringEncoding), &error);
        if (!compiled) {
            failures.emplace_back(name.string(), error ? error->localizedDescription()->utf8String() : "missing offline Metal library");
        } else {
            if (names(compiled) != names(library)) failures.emplace_back(name.string(), "offline library functions differ from source");
            compiled->release();
        }
        library->release();
    }

    for (const auto &failure : failures) std::println(stderr, "FAIL {}\n{}\n", failure.Name, failure.Message);
    std::println("{} shader source(s) compiled, {} entry point(s), {} failure(s)", sources.size(), functions, failures.size());
    pool->release();
    return failures.empty() ? 0 : 1;
}
