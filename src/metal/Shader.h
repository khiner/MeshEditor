#pragma once

#include "metal/Image.h"
#include "metal/MslSource.h"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mtl {
inline constexpr uint32_t MaxMeshThreadgroupsPerGrid{1'048'575};

struct FunctionConstant {
    uint32_t Index;
    MTL::DataType Type;
    uint32_t Value; // Stores a bool, uint, or float bit pattern interpreted according to Type.
};

// Caches shader libraries until their source or included files change.
// Uses `pipeline_archive` for deterministic MTL4 pipeline binaries when the device supports Metal 4.
// PipelineCompiler() returns null on older devices, which use the classic APIs without archive caching.
struct LibraryCache {
    LibraryCache(const Context &ctx, std::filesystem::path shaders_dir, std::filesystem::path pipeline_archive = {});
    ~LibraryCache();
    LibraryCache(const LibraryCache &) = delete;
    LibraryCache &operator=(const LibraryCache &) = delete;
    LibraryCache(LibraryCache &&) = default;

    MTL::Library *Get(const std::filesystem::path &relative_path, const std::vector<std::string> &defines = {});
    void Clear() { Entries.clear(); }

    // Returns the archived pipeline for `descriptor`, or null after an archive miss.
    MTL::RenderPipelineState *ArchivedRenderPipeline(const MTL4::PipelineDescriptor *) const;
    MTL::ComputePipelineState *ArchivedComputePipeline(const MTL4::ComputePipelineDescriptor *) const;

    MTL4::Compiler *PipelineCompiler() const { return Compiler.get(); }
    void NotePipelineCreated() { PipelineCreated = true; }

    const Context &Ctx;

private:
    struct Entry {
        NS::SharedPtr<MTL::Library> Library;
        std::vector<std::pair<std::filesystem::path, std::filesystem::file_time_type>> Deps;
    };
    std::filesystem::path ShadersDir;
    std::unordered_map<std::string, Entry> Entries;
    std::filesystem::path ArchivePath;
    NS::SharedPtr<MTL4::PipelineDataSetSerializer> Serializer;
    NS::SharedPtr<MTL4::Compiler> Compiler;
    NS::SharedPtr<MTL4::Archive> LoadedArchive;
    std::optional<uint64_t> SourceFingerprint;
    bool PipelineCreated{false};
};

struct FunctionRef {
    std::filesystem::path Path; // Relative to the shaders directory.
    std::string Name;
    std::vector<FunctionConstant> Constants{};
    std::vector<std::string> Defines{};
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

struct RenderPipeline {
    RenderPipeline(
        LibraryCache &, FunctionRef vertex, std::optional<FunctionRef> fragment, PassFormats,
        std::vector<BlendState> blends = {}, std::optional<DepthState> depth = {}, float depth_bias = 0.f
    );

    void Compile(LibraryCache &);

    MTL::RenderPipelineState *State() const { return PipelineState.get(); }
    float DepthBias() const { return Bias; }

    void Bind(MTL::RenderCommandEncoder *encoder) const {
        encoder->setRenderPipelineState(PipelineState.get());
        encoder->setDepthStencilState(DepthStencilState.get());
    }

private:
    FunctionRef VertexFn;
    std::optional<FunctionRef> FragmentFn;
    PassFormats Formats;
    std::vector<BlendState> Blends;
    std::optional<DepthState> Depth;
    float Bias;
    NS::SharedPtr<MTL::RenderPipelineState> PipelineState;
    NS::SharedPtr<MTL::DepthStencilState> DepthStencilState;
};

struct MeshRenderPipeline {
    MeshRenderPipeline(
        LibraryCache &, FunctionRef mesh, std::optional<FunctionRef> fragment, PassFormats,
        std::vector<BlendState> blends = {}, std::optional<DepthState> depth = {}
    );

    void Compile(LibraryCache &);
    void Bind(MTL::RenderCommandEncoder *encoder) const {
        encoder->setRenderPipelineState(PipelineState.get());
        encoder->setDepthStencilState(DepthStencilState.get());
    }

private:
    FunctionRef MeshFn;
    std::optional<FunctionRef> FragmentFn;
    PassFormats Formats;
    std::vector<BlendState> Blends;
    std::optional<DepthState> Depth;
    NS::SharedPtr<MTL::RenderPipelineState> PipelineState;
    NS::SharedPtr<MTL::DepthStencilState> DepthStencilState;
};

struct ComputePipeline {
    ComputePipeline(LibraryCache &, FunctionRef);

    void Compile(LibraryCache &);

    MTL::ComputePipelineState *State() const { return PipelineState.get(); }
    uint32_t MaxThreadsPerThreadgroup() const { return PipelineState ? uint32_t(PipelineState->maxTotalThreadsPerThreadgroup()) : 0; }

private:
    FunctionRef Fn;
    NS::SharedPtr<MTL::ComputePipelineState> PipelineState;
};

}
