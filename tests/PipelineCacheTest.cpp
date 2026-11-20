#include "metal/Shader.h"
#include "metal/MetalCpp.h"

#include <print>
#include <stdexcept>

// The caller supplies a fresh shader directory and archive to exercise cold, warm and concurrent processes.
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    mtl::Context context;
    try {
        mtl::LibraryCache original{context, argv[1], argv[2]};
        mtl::LibraryCache cache{std::move(original)};
        const mtl::ComputePipeline first{cache, {"CacheTest.metal", "CacheTest"}};
        if (std::string_view{argv[3]} == "unwind") throw std::runtime_error("Expected unwind");
        if (!cache.FlushArchive()) return 3;
        const mtl::ComputePipeline second{cache, {"CacheTest.metal", "CacheTestSecond"}};
        if (!cache.FlushArchive()) return 4;
        std::println("{{\"hits\":{},\"compiled\":{}}}", cache.ArchiveHitCount(), cache.CompileMissCount());
    } catch (const std::exception &error) {
        if (std::string_view{argv[3]} == "unwind" && std::string_view{error.what()} == "Expected unwind") return 0;
        std::println(stderr, "{}", error.what());
        return 1;
    }
}
