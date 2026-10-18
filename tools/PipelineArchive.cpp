#include "mesh/MeshPipelines.h"
#include "render/Pipelines.h"

#include <fstream>
#include <print>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::println(stderr, "usage: MeshEditorPipelineArchive <shaders-dir> <output.mtl4a> <stamp>");
        return 2;
    }
    const auto shaders = std::filesystem::path{argv[1]};
    const auto output = std::filesystem::path{argv[2]};
    const auto stamp = std::filesystem::path{argv[3]};
    auto stamp_tmp = stamp;
    stamp_tmp += ".tmp." + std::to_string(getpid());
    try {
        const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        std::filesystem::create_directories(output.parent_path());
        uint32_t compiled = 0u, hits = 0u;
        {
            mtl::Context context;
            mtl::LibraryCache cache{context, shaders, output, {}, false};
            if (!cache.PipelineCompiler()) throw std::runtime_error("Metal 4 pipeline archives are unavailable on this device.");
            MeshPipelines mesh{cache};
            for (uint32_t i = 0; i < uint32_t(MeshPass::Count); ++i)
                (void)mesh[MeshPass(i)];
            Pipelines render{cache};
            render.Main.Compiler.CompilePipelines(0u, false);
            render.Main.Compiler.CompilePipelines(0u, true);
            render.Main.Compiler.CompilePipelines(uint32_t(PbrFeature::Punctual), false);
            render.Main.Compiler.CompilePipelines(uint32_t(PbrFeature::Punctual), true);
            compiled=cache.CompileMissCount(); hits=cache.ArchiveHitCount();
            if (!cache.FlushArchive()) throw std::runtime_error("Metal pipeline archive serialization failed.");
        }
        if (!std::filesystem::exists(output)) throw std::runtime_error("Pipeline archive is missing after compilation.");
        {
            std::ofstream file{stamp_tmp};
            file << "complete\n";
            file.close();
            if (!file) throw std::runtime_error("Pipeline archive build stamp could not be written.");
        }
        std::filesystem::rename(stamp_tmp, stamp);
        std::println(stderr, "Pipeline archive: {} hits, {} compiled.", hits, compiled);
        return 0;
    } catch (const std::exception &error) {
        std::filesystem::remove(stamp_tmp);
        std::println(stderr, "Pipeline archive: {}", error.what());
        return 1;
    }
}
