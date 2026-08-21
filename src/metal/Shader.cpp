#include "metal/Shader.h"
#include "metal/MetalContext.h"

#include <format>
#include <stdexcept>

#include <unistd.h>

namespace mtl {
namespace {
bool DepsUnchanged(const std::vector<std::pair<std::filesystem::path, std::filesystem::file_time_type>> &deps, const std::filesystem::path &root) {
    std::error_code ec;
    for (const auto &[relative, mtime] : deps) {
        const auto current = std::filesystem::last_write_time(root / relative, ec);
        if (ec || current != mtime) return false;
    }
    return true;
}

// Classic-path attachments. Classic pipelines carry the depth format, so it configures that too.
template<typename Descriptor>
void ConfigureAttachments(Descriptor *descriptor, const PassFormats &formats, const std::vector<BlendState> &blends) {
    for (size_t i = 0; i < formats.Color.size(); ++i) {
        auto *attachment = descriptor->colorAttachments()->object(i);
        attachment->setPixelFormat(formats.Color[i]);
        const auto blend = i < blends.size() ? blends[i] : NoBlend;
        attachment->setWriteMask(blend.WriteMask ? MTL::ColorWriteMaskAll : MTL::ColorWriteMaskNone);
        attachment->setBlendingEnabled(blend.Enabled);
        if (!blend.Enabled) continue;
        attachment->setSourceRGBBlendFactor(blend.SourceRgb);
        attachment->setDestinationRGBBlendFactor(blend.DestRgb);
        attachment->setRgbBlendOperation(MTL::BlendOperationAdd);
        attachment->setSourceAlphaBlendFactor(blend.SourceAlpha);
        attachment->setDestinationAlphaBlendFactor(blend.DestAlpha);
        attachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
    }
    if (formats.Depth != MTL::PixelFormatInvalid) descriptor->setDepthAttachmentPixelFormat(formats.Depth);
}

void ConfigureAttachments(MTL4::RenderPipelineColorAttachmentDescriptorArray *attachments, const PassFormats &formats, const std::vector<BlendState> &blends) {
    for (size_t i = 0; i < formats.Color.size(); ++i) {
        auto *attachment = attachments->object(i);
        attachment->setPixelFormat(formats.Color[i]);
        const auto blend = i < blends.size() ? blends[i] : NoBlend;
        attachment->setWriteMask(blend.WriteMask ? MTL::ColorWriteMaskAll : MTL::ColorWriteMaskNone);
        attachment->setBlendingState(blend.Enabled ? MTL4::BlendStateEnabled : MTL4::BlendStateDisabled);
        if (!blend.Enabled) continue;
        attachment->setSourceRGBBlendFactor(blend.SourceRgb);
        attachment->setDestinationRGBBlendFactor(blend.DestRgb);
        attachment->setRgbBlendOperation(MTL::BlendOperationAdd);
        attachment->setSourceAlphaBlendFactor(blend.SourceAlpha);
        attachment->setDestinationAlphaBlendFactor(blend.DestAlpha);
        attachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
    }
    // MTL4 pipelines carry no depth format. The pass's depth attachment binds it at encode time.
}

NS::SharedPtr<MTL::DepthStencilState> MakeDepthState(LibraryCache &cache, const std::optional<DepthState> &depth) {
    const auto descriptor = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
    descriptor->setDepthCompareFunction(depth && depth->Test ? depth->Compare : MTL::CompareFunctionAlways);
    descriptor->setDepthWriteEnabled(depth && depth->Write);
    return NS::TransferPtr(cache.Ctx.Device->newDepthStencilState(descriptor.get()));
}

// Materializes `ref` for the classic pipeline descriptors, compiling its library through the cache.
NS::SharedPtr<MTL::Function> MakeFunction(LibraryCache &cache, const FunctionRef &ref) {
    auto *library = cache.Get(ref.Path, ref.Defines);
    if (ref.Constants.empty()) {
        auto function = NS::TransferPtr(library->newFunction(Str(ref.Name)));
        if (!function) throw std::runtime_error(std::format("No function '{}' in '{}'", ref.Name, ref.Path.string()));
        return function;
    }
    const auto values = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    for (const auto &constant : ref.Constants) {
        if (constant.Type == MTL::DataTypeBool) {
            const bool value = constant.Value != 0;
            values->setConstantValue(&value, constant.Type, NS::UInteger(constant.Index));
        } else {
            values->setConstantValue(&constant.Value, constant.Type, NS::UInteger(constant.Index));
        }
    }
    NS::Error *error = nullptr;
    auto function = NS::TransferPtr(library->newFunction(Str(ref.Name), values.get(), &error));
    if (!function) {
        throw std::runtime_error(std::format("Failed to specialize '{}' in '{}':\n{}", ref.Name, ref.Path.string(), error ? error->localizedDescription()->utf8String() : "unknown"));
    }
    return function;
}

// Describes `ref` for the MTL4 compiler, compiling its library through the cache.
NS::SharedPtr<MTL4::FunctionDescriptor> MakeFunctionDescriptor(LibraryCache &cache, const FunctionRef &ref) {
    auto *library = cache.Get(ref.Path, ref.Defines);
    auto function = NS::TransferPtr(MTL4::LibraryFunctionDescriptor::alloc()->init());
    function->setName(Str(ref.Name));
    function->setLibrary(library);
    if (ref.Constants.empty()) return function;

    const auto values = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    for (const auto &constant : ref.Constants) {
        if (constant.Type == MTL::DataTypeBool) {
            const bool value = constant.Value != 0;
            values->setConstantValue(&value, constant.Type, NS::UInteger(constant.Index));
        } else {
            values->setConstantValue(&constant.Value, constant.Type, NS::UInteger(constant.Index));
        }
    }
    auto specialized = NS::TransferPtr(MTL4::SpecializedFunctionDescriptor::alloc()->init());
    specialized->setFunctionDescriptor(function.get());
    specialized->setConstantValues(values.get());
    return specialized;
}
} // namespace

LibraryCache::LibraryCache(const Context &ctx, std::filesystem::path shaders_dir, std::filesystem::path pipeline_archive)
    : Ctx(ctx), ShadersDir(std::move(shaders_dir)), ArchivePath(std::move(pipeline_archive)) {
    // Without a compiler, pipelines build through the classic device APIs and skip archive caching.
    if (!Ctx.Device->supportsFamily(MTL::GPUFamilyMetal4)) return;
    const auto compiler_descriptor = NS::TransferPtr(MTL4::CompilerDescriptor::alloc()->init());
    if (!ArchivePath.empty()) {
        const auto serializer_descriptor = NS::TransferPtr(MTL4::PipelineDataSetSerializerDescriptor::alloc()->init());
        serializer_descriptor->setConfiguration(MTL4::PipelineDataSetSerializerConfigurationCaptureBinaries);
        Serializer = NS::TransferPtr(Ctx.Device->newPipelineDataSetSerializer(serializer_descriptor.get()));
        compiler_descriptor->setPipelineDataSetSerializer(Serializer.get());
    }
    NS::Error *error = nullptr;
    Compiler = NS::TransferPtr(Ctx.Device->newCompiler(compiler_descriptor.get(), &error));
    if (!Compiler) {
        throw std::runtime_error(std::format("Failed to create the Metal pipeline compiler:\n{}", error ? error->localizedDescription()->utf8String() : "unknown"));
    }

    std::error_code ec;
    if (ArchivePath.empty() || !std::filesystem::exists(ArchivePath, ec)) return;
    // A stale or corrupt archive only costs fresh compiles, so a failed load proceeds without one.
    if (auto archive = NS::TransferPtr(Ctx.Device->newArchive(NS::URL::fileURLWithPath(Str(ArchivePath.string())), &error))) {
        LoadedArchive = std::move(archive);
    }
}

// Flush every pipeline this session compiled or looked up back to the archive file.
// The temporary-and-rename keeps concurrent processes from reading a half-written archive.
LibraryCache::~LibraryCache() {
    if (ArchivePath.empty() || !Serializer || !PipelineCreated) return;
    std::error_code ec;
    std::filesystem::create_directories(ArchivePath.parent_path(), ec);
    auto tmp = ArchivePath;
    tmp += std::format(".{}", getpid());
    NS::Error *error = nullptr;
    if (Serializer->serializeAsArchiveAndFlushToURL(NS::URL::fileURLWithPath(Str(tmp.string())), &error)) {
        std::filesystem::rename(tmp, ArchivePath, ec);
    }
    std::filesystem::remove(tmp, ec);
}

MTL::RenderPipelineState *LibraryCache::ArchivedRenderPipeline(const MTL4::PipelineDescriptor *descriptor) const {
    if (!LoadedArchive) return nullptr;
    NS::Error *error = nullptr;
    return LoadedArchive->newRenderPipelineState(descriptor, &error);
}

MTL::ComputePipelineState *LibraryCache::ArchivedComputePipeline(const MTL4::ComputePipelineDescriptor *descriptor) const {
    if (!LoadedArchive) return nullptr;
    NS::Error *error = nullptr;
    return LoadedArchive->newComputePipelineState(descriptor, &error);
}

MTL::Library *LibraryCache::Get(const std::filesystem::path &relative_path, const std::vector<std::string> &defines) {
    auto key = relative_path.string();
    for (const auto &define : defines) key += "|" + define;
    auto &entry = Entries[key];
    if (entry.Library && DepsUnchanged(entry.Deps, ShadersDir)) return entry.Library.get();

    const auto source = msl::Load(ShadersDir, relative_path, defines);
    NS::Error *error = nullptr;
    auto library = NS::TransferPtr(Ctx.Device->newLibrary(Str(source.Text), static_cast<MTL::CompileOptions *>(nullptr), &error));
    if (!library) {
        throw std::runtime_error(std::format("Failed to compile shader '{}':\n{}", relative_path.string(), error ? error->localizedDescription()->utf8String() : "unknown"));
    }
    std::error_code ec;
    decltype(entry.Deps) deps;
    deps.reserve(source.Files.size());
    for (const auto &file : source.Files) deps.emplace_back(file, std::filesystem::last_write_time(ShadersDir / file, ec));
    entry = Entry{std::move(library), std::move(deps)};
    return entry.Library.get();
}

RenderPipeline::RenderPipeline(
    LibraryCache &cache, FunctionRef vertex, std::optional<FunctionRef> fragment, PassFormats formats,
    std::vector<BlendState> blends, std::optional<DepthState> depth, float depth_bias
) : VertexFn(std::move(vertex)), FragmentFn(std::move(fragment)), Formats(std::move(formats)),
    Blends(std::move(blends)), Depth(depth), Bias(depth_bias) {
    Compile(cache);
}

void RenderPipeline::Compile(LibraryCache &cache) {
    if (!cache.PipelineCompiler()) {
        const auto descriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
        const auto vertex_function = MakeFunction(cache, VertexFn);
        descriptor->setVertexFunction(vertex_function.get());
        NS::SharedPtr<MTL::Function> fragment_function;
        if (FragmentFn) {
            fragment_function = MakeFunction(cache, *FragmentFn);
            descriptor->setFragmentFunction(fragment_function.get());
        }
        ConfigureAttachments(descriptor.get(), Formats, Blends);
        NS::Error *error = nullptr;
        PipelineState = NS::TransferPtr(cache.Ctx.Device->newRenderPipelineState(descriptor.get(), &error));
        if (!PipelineState) {
            throw std::runtime_error(std::format("Failed to create the render pipeline for '{}':\n{}", VertexFn.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
        }
        DepthStencilState = MakeDepthState(cache, Depth);
        return;
    }
    const auto descriptor = NS::TransferPtr(MTL4::RenderPipelineDescriptor::alloc()->init());
    const auto vertex_function = MakeFunctionDescriptor(cache, VertexFn);
    descriptor->setVertexFunctionDescriptor(vertex_function.get());
    NS::SharedPtr<MTL4::FunctionDescriptor> fragment_function;
    if (FragmentFn) {
        fragment_function = MakeFunctionDescriptor(cache, *FragmentFn);
        descriptor->setFragmentFunctionDescriptor(fragment_function.get());
    }

    ConfigureAttachments(descriptor->colorAttachments(), Formats, Blends);

    PipelineState = NS::TransferPtr(cache.ArchivedRenderPipeline(descriptor.get()));
    if (!PipelineState) {
        NS::Error *error = nullptr;
        PipelineState = NS::TransferPtr(cache.PipelineCompiler()->newRenderPipelineState(descriptor.get(), nullptr, &error));
        if (!PipelineState) {
            throw std::runtime_error(std::format("Failed to create the render pipeline for '{}':\n{}", VertexFn.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
        }
    }
    cache.NotePipelineCreated();

    DepthStencilState = MakeDepthState(cache, Depth);
}

MeshRenderPipeline::MeshRenderPipeline(
    LibraryCache &cache, FunctionRef mesh, std::optional<FunctionRef> fragment, PassFormats formats,
    std::vector<BlendState> blends, std::optional<DepthState> depth
) : MeshFn(std::move(mesh)), FragmentFn(std::move(fragment)), Formats(std::move(formats)),
    Blends(std::move(blends)), Depth(depth) {
    Compile(cache);
}

void MeshRenderPipeline::Compile(LibraryCache &cache) {
    if (!cache.PipelineCompiler()) {
        const auto descriptor = NS::TransferPtr(MTL::MeshRenderPipelineDescriptor::alloc()->init());
        const auto mesh_function = MakeFunction(cache, MeshFn);
        descriptor->setMeshFunction(mesh_function.get());
        descriptor->setMaxTotalThreadsPerMeshThreadgroup(160);
        descriptor->setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(true);
        descriptor->setMaxTotalThreadgroupsPerMeshGrid(1'048'575);
        NS::SharedPtr<MTL::Function> fragment_function;
        if (FragmentFn) {
            fragment_function = MakeFunction(cache, *FragmentFn);
            descriptor->setFragmentFunction(fragment_function.get());
        }
        ConfigureAttachments(descriptor.get(), Formats, Blends);
        NS::Error *error = nullptr;
        PipelineState = NS::TransferPtr(cache.Ctx.Device->newRenderPipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
        if (!PipelineState) {
            throw std::runtime_error(std::format("Failed to create the mesh render pipeline for '{}':\n{}", MeshFn.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
        }
        DepthStencilState = MakeDepthState(cache, Depth);
        return;
    }
    const auto descriptor = NS::TransferPtr(MTL4::MeshRenderPipelineDescriptor::alloc()->init());
    const auto mesh_function = MakeFunctionDescriptor(cache, MeshFn);
    descriptor->setMeshFunctionDescriptor(mesh_function.get());
    descriptor->setMaxTotalThreadsPerMeshThreadgroup(160);
    descriptor->setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(true);
    descriptor->setMaxTotalThreadgroupsPerMeshGrid(1'048'575);
    NS::SharedPtr<MTL4::FunctionDescriptor> fragment_function;
    if (FragmentFn) {
        fragment_function = MakeFunctionDescriptor(cache, *FragmentFn);
        descriptor->setFragmentFunctionDescriptor(fragment_function.get());
    }
    ConfigureAttachments(descriptor->colorAttachments(), Formats, Blends);

    PipelineState = NS::TransferPtr(cache.ArchivedRenderPipeline(descriptor.get()));
    if (!PipelineState) {
        NS::Error *error = nullptr;
        PipelineState = NS::TransferPtr(cache.PipelineCompiler()->newRenderPipelineState(descriptor.get(), nullptr, &error));
        if (!PipelineState) {
            throw std::runtime_error(std::format("Failed to create the mesh render pipeline for '{}':\n{}", MeshFn.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
        }
    }
    cache.NotePipelineCreated();
    DepthStencilState = MakeDepthState(cache, Depth);
}

ComputePipeline::ComputePipeline(LibraryCache &cache, FunctionRef fn) : Fn(std::move(fn)) { Compile(cache); }

void ComputePipeline::Compile(LibraryCache &cache) {
    if (!cache.PipelineCompiler()) {
        const auto function = MakeFunction(cache, Fn);
        NS::Error *error = nullptr;
        PipelineState = NS::TransferPtr(cache.Ctx.Device->newComputePipelineState(function.get(), &error));
        if (!PipelineState) {
            throw std::runtime_error(std::format("Failed to create the compute pipeline for '{}':\n{}", Fn.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
        }
        return;
    }
    const auto descriptor = NS::TransferPtr(MTL4::ComputePipelineDescriptor::alloc()->init());
    const auto function = MakeFunctionDescriptor(cache, Fn);
    descriptor->setComputeFunctionDescriptor(function.get());
    PipelineState = NS::TransferPtr(cache.ArchivedComputePipeline(descriptor.get()));
    if (!PipelineState) {
        NS::Error *error = nullptr;
        PipelineState = NS::TransferPtr(cache.PipelineCompiler()->newComputePipelineState(descriptor.get(), nullptr, &error));
        if (!PipelineState) {
            throw std::runtime_error(std::format("Failed to create the compute pipeline for '{}':\n{}", Fn.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
        }
    }
    cache.NotePipelineCreated();
}
} // namespace mtl
