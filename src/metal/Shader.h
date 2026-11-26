#pragma once

#include "metal/Command.h"
#include "metal/Image.h"
#include "metal/MetalSchema.h"
#include "metal/MslSource.h"

#include <Metal/MTLComputePipeline.hpp>
#include <Metal/MTLLibrary.hpp>
#include <Metal/MTLRenderPipeline.hpp>

namespace MTL4 {
class PipelineDescriptor;
class ComputePipelineDescriptor;
class Compiler;
class CompilerTaskOptions;
class PipelineDataSetSerializer;
class Archive;
} // namespace MTL4

namespace mtl {
inline constexpr uint32_t MaxMeshThreadgroupsPerGrid{1'048'575};

struct FunctionConstant {
    uint32_t Index;
    MTL::DataType Type;
    uint32_t Value; // Stores a bool, uint, or float bit pattern interpreted according to Type.
};

enum class ArchiveMode : uint8_t {
    Read, // Creates pipelines from the archive and compiles the rest.
    ReadOnly, // Creates pipelines from the archive only.
    Write, // Compiles every pipeline, reusing the archive's binaries, for `WriteArchive` to replace the archive with.
};

// Caches shader libraries until their source or included files change, and creates Metal 4 pipelines through one pipeline archive.
struct LibraryCache {
    LibraryCache(const Context &ctx, std::filesystem::path shaders_dir, std::filesystem::path archive = {}, ArchiveMode mode = ArchiveMode::Read);
    ~LibraryCache();
    LibraryCache(const LibraryCache &) = delete;
    LibraryCache &operator=(const LibraryCache &) = delete;
    LibraryCache(LibraryCache &&) noexcept;

    MTL::Library *Get(const std::filesystem::path &relative_path, const std::vector<std::string> &defines = {});
    // Independent read-only cache of the same archive, for background pipeline creation.
    std::unique_ptr<LibraryCache> PrewarmCache() const;
    void Clear();
    // Replaces the archive with every pipeline this write-mode cache created.
    // Call it once, after the last pipeline.
    bool WriteArchive();

    MTL4::Compiler *PipelineCompiler() const { return Compiler.get(); }
    MTL4::CompilerTaskOptions *CompileOptions() const { return Lookup.get(); }
    bool ArchiveOnly() const { return Mode == ArchiveMode::ReadOnly; }
    NS::SharedPtr<MTL::ComputePipelineState> FindComputePipeline(const MTL4::ComputePipelineDescriptor *) const;
    NS::SharedPtr<MTL::RenderPipelineState> FindRenderPipeline(const MTL4::PipelineDescriptor *) const;
    void NotePipelineCreated() { ++CompileMisses; }
    uint32_t ArchiveHitCount() const { return ArchiveHits; }
    uint32_t CompileMissCount() const { return CompileMisses; }
    uint32_t BinaryLibraryLoadCount() const { return BinaryLibraries; }
    uint32_t SourceLibraryCompileCount() const { return SourceLibraries; }

    const Context &Ctx;

private:
    struct Entry {
        NS::SharedPtr<MTL::Library> Library;
        std::vector<std::pair<std::filesystem::path, std::filesystem::file_time_type>> Deps;
    };
    std::filesystem::path ShadersDir;
    std::unordered_map<std::string, Entry> Entries;
    std::filesystem::path ArchivePath;
    ArchiveMode Mode;
    NS::SharedPtr<MTL4::PipelineDataSetSerializer> Serializer;
    NS::SharedPtr<MTL4::Compiler> Compiler;
    NS::SharedPtr<MTL4::Archive> Archive; // Read modes look pipelines up here.
    NS::SharedPtr<MTL4::CompilerTaskOptions> Lookup; // Write mode compiles with the archive as a lookup archive.
    mutable uint32_t ArchiveHits{};
    uint32_t CompileMisses{};
    uint32_t BinaryLibraries{}, SourceLibraries{};
};

struct FunctionRef {
    std::filesystem::path Path; // Relative to the shaders directory.
    std::string Name;
    std::vector<FunctionConstant> Constants{};
    std::vector<std::string> Defines{};
    // The compute or fragment function calls binary functions linked into its pipeline.
    bool BinaryLinking{false};
};

struct PassFormats {
    std::vector<MTL::PixelFormat> Color{};
    MTL::PixelFormat Depth{MTL::PixelFormatInvalid};
    bool operator==(const PassFormats &) const = default;
};

struct BlendState {
    bool Enabled{true};
    bool WriteMask{true}; // False writes no channels, for passes that target one attachment of several.
    MTL::BlendFactor SourceRgb{MTL::BlendFactorSourceAlpha};
    MTL::BlendFactor DestRgb{MTL::BlendFactorOneMinusSourceAlpha};
    MTL::BlendFactor SourceAlpha{MTL::BlendFactorOne};
    MTL::BlendFactor DestAlpha{MTL::BlendFactorOneMinusSourceAlpha};
};

inline constexpr BlendState Blend{};
inline constexpr BlendState NoBlend{.Enabled = false};
inline constexpr BlendState NoWrite{.Enabled = false, .WriteMask = false};
inline constexpr BlendState AdditiveBlend{
    .SourceRgb = MTL::BlendFactorOne, .DestRgb = MTL::BlendFactorOne, .SourceAlpha = MTL::BlendFactorOne, .DestAlpha = MTL::BlendFactorOne
};
inline constexpr BlendState PremultipliedBlend{.SourceRgb = MTL::BlendFactorOne};

struct DepthState {
    bool Test{true};
    bool Write{true};
    MTL::CompareFunction Compare{MTL::CompareFunctionLess};
    bool operator==(const DepthState &) const = default;
};

// A render or mesh pipeline state with the depth-stencil state its passes bind alongside it.
struct RenderPipeline {
    void Bind(RenderEncoder) const;
    uint32_t ImageblockSampleLength() const { return uint32_t(PipelineState->imageblockSampleLength()); }

    NS::SharedPtr<MTL::RenderPipelineState> PipelineState;
    NS::SharedPtr<MTL::DepthStencilState> DepthStencilState;
};

RenderPipeline MakeRenderPipeline(
    LibraryCache &, FunctionRef vertex, std::optional<FunctionRef> fragment, const PassFormats &,
    const std::vector<BlendState> &blends = {}, std::optional<DepthState> depth = {}
);
// A mesh pipeline with binary functions links them into its fragment stage as it compiles.
RenderPipeline MakeMeshPipeline(
    LibraryCache &, FunctionRef mesh, std::optional<FunctionRef> fragment, const PassFormats &,
    const std::vector<BlendState> &blends = {}, std::optional<DepthState> depth = {}, const NS::Array *fragment_functions = nullptr
);

struct ComputePipeline {
    ComputePipeline(LibraryCache &, FunctionRef);

    void Bind(ComputeEncoder) const;

    NS::SharedPtr<MTL::ComputePipelineState> PipelineState;
};

} // namespace mtl
