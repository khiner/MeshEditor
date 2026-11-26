#include "metal/Shader.h"
#include "metal/MetalCpp.h"

#include <print>
#include <stdexcept>

// The caller supplies a fresh shader directory and archive to exercise writing, reading and concurrent processes.
// Mode "write" writes the archive, "read" reads it, and "unwind" leaves a write-mode cache by an exception.
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const std::string_view mode{argv[3]};
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    mtl::Context context;
    try {
        mtl::LibraryCache original{context, argv[1], argv[2], mode == "read" ? mtl::ArchiveMode::Read : mtl::ArchiveMode::Write};
        mtl::LibraryCache cache{std::move(original)};
        const mtl::ComputePipeline first{cache, {"CacheTest.metal", "CacheTest"}};
        if (mode == "unwind") throw std::runtime_error("Expected unwind");
        const mtl::ComputePipeline second{cache, {"CacheTest.metal", "CacheTestSecond"}};
        if (mode == "write" && !cache.WriteArchive()) return 3;
        std::println("{{\"hits\":{},\"compiled\":{}}}", cache.ArchiveHitCount(), cache.CompileMissCount());
    } catch (const std::exception &error) {
        if (mode == "unwind" && std::string_view{error.what()} == "Expected unwind") return 0;
        std::println(stderr, "{}", error.what());
        return 1;
    }
}
