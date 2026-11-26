#include "mesh/MeshPipelines.h"
#include "metal/ImGuiRenderer.h"
#include "render/Pipelines.h"

#include <fstream>
#include <print>

namespace {
void CreateBuiltinPipelines(mtl::LibraryCache &cache) {
    MeshPipelines mesh{cache};
    for (uint32_t i = 0; i < uint32_t(MeshPass::Count); ++i)
        (void)mesh[MeshPass(i)];
    Pipelines render{cache};
    // Each base variant, and the variant a scene of it takes once a material's graph computes inputs.
    mtl::ProgramRegistry programs{cache.Ctx};
    for (const auto mask : {PbrFeatureMask{}, FeatureBit(PbrFeature::NonTriangleTopology), FeatureBit(PbrFeature::SceneLights), FeatureBit(PbrFeature::SceneLights) | PbrFeature::NonTriangleTopology}) {
        (void)render.Main.Compiler.Variant(mask, nullptr);
        (void)render.Main.Compiler.Variant(mask | PbrFeature::Patterns, &programs);
    }
    (void)mtl::ImGuiPipeline(cache, mtl::Format::Color);
}
} // namespace

// Writes one archive of every built-in pipeline, reusing the binaries of the archive it replaces.
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
        mtl::Context context;
        // An archive that already holds every built-in pipeline stays as it is.
        const bool complete = std::filesystem::exists(output) && [&] {
            mtl::LibraryCache cache{context, shaders, output};
            CreateBuiltinPipelines(cache);
            return cache.CompileMissCount() == 0u;
        }();
        uint32_t written = 0u;
        if (!complete) {
            mtl::LibraryCache cache{context, shaders, output, mtl::ArchiveMode::Write};
            CreateBuiltinPipelines(cache);
            written = cache.CompileMissCount();
            if (!cache.WriteArchive()) throw std::runtime_error("Metal pipeline archive serialization failed.");
        }
        if (!std::filesystem::exists(output)) throw std::runtime_error("Pipeline archive is missing after compilation.");
        {
            std::ofstream file{stamp_tmp};
            file << "complete\n";
            file.close();
            if (!file) throw std::runtime_error("Pipeline archive build stamp could not be written.");
        }
        std::filesystem::rename(stamp_tmp, stamp);
        std::println(stderr, "Pipeline archive: {} pipelines written.", written);
        return 0;
    } catch (const std::exception &error) {
        std::filesystem::remove(stamp_tmp);
        std::println(stderr, "Pipeline archive: {}", error.what());
        return 1;
    }
}
