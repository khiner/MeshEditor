// Compiles all shaders or the file names supplied on the command line.
// Generated assertions verify CPU and MSL layouts.
#include "TestPaths.h"
#include "metal/MetalCpp.h"
#include "metal/MslSource.h"

#include <algorithm>
#include <filesystem>
#include <print>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
struct Failure {
    std::string Name, Message;
};
} // namespace
int main(int argc, char **argv) {
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

    const std::vector<std::string> requested{argv + 1, argv + argc};
    std::vector<fs::path> sources;
    for (const auto &entry : fs::directory_iterator{shaders_dir}) {
        const auto name = entry.path().filename();
        if (name.extension() != ".metal") continue;
        if (!requested.empty() && std::ranges::find(requested, name.string()) == requested.end()) continue;
        sources.emplace_back(name);
    }
    std::ranges::sort(sources);

    std::vector<Failure> failures;
    uint32_t functions = 0;
    for (const auto &name : sources) {
        std::string text;
        try {
            text = msl::Load(shaders_dir, name).Text;
        } catch (const std::exception &e) {
            failures.emplace_back(name.string(), e.what());
            continue;
        }
        NS::Error *error = nullptr;
        auto *library = device->newLibrary(NS::String::string(text.c_str(), NS::UTF8StringEncoding), static_cast<MTL::CompileOptions *>(nullptr), &error);
        if (!library) {
            failures.emplace_back(name.string(), error ? error->localizedDescription()->utf8String() : "unknown compile failure");
            continue;
        }
        functions += uint32_t(library->functionNames()->count());
        library->release();
    }

    for (const auto &failure : failures) std::println(stderr, "FAIL {}\n{}\n", failure.Name, failure.Message);
    std::println("{} shader source(s) compiled, {} entry point(s), {} failure(s)", sources.size(), functions, failures.size());
    pool->release();
    return failures.empty() ? 0 : 1;
}
