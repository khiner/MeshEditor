#include "metal/Shader.h"
#include "metal/AutoreleaseScope.h"

#include "metal/MetalCpp.h"

#include <cstdlib>
#include <format>
#include <objc/message.h>

namespace mtl {
LibraryCache::LibraryCache(LibraryCache &&) noexcept = default;
void LibraryCache::Clear() {
    const AutoreleaseScope pool;
    Entries.clear();
}

void RenderPipeline::Bind(RenderEncoder encoder) const {
    auto &state = encoder.Cmd->State;
    if (state.Pipeline != PipelineState.get()) {
        encoder->setRenderPipelineState(PipelineState.get());
        state.Pipeline = PipelineState.get();
    }
    if (state.DepthStencil != DepthStencilState.get()) {
        encoder->setDepthStencilState(DepthStencilState.get());
        state.DepthStencil = DepthStencilState.get();
    }
}

void ComputePipeline::Bind(ComputeEncoder encoder) const {
    auto &state = encoder.Cmd->State;
    if (state.Pipeline == PipelineState.get()) return;
    encoder->setComputePipelineState(PipelineState.get());
    state.Pipeline = PipelineState.get();
}

namespace {
bool DepsUnchanged(std::span<const std::pair<std::filesystem::path, std::filesystem::file_time_type>> deps, const std::filesystem::path &root) {
    std::error_code ec;
    for (const auto &[relative, mtime] : deps) {
        const auto current = std::filesystem::last_write_time(root / relative, ec);
        if (ec || current != mtime) return false;
    }
    return true;
}

void ConfigureColorAttachments(MTL4::RenderPipelineColorAttachmentDescriptorArray *attachments, const PassFormats &formats, std::span<const BlendState> blends) {
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
}

NS::SharedPtr<MTL::DepthStencilState> MakeDepthState(LibraryCache &cache, const std::optional<DepthState> &depth) {
    const auto descriptor = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
    descriptor->setDepthCompareFunction(depth && depth->Test ? depth->Compare : MTL::CompareFunctionAlways);
    descriptor->setDepthWriteEnabled(depth && depth->Write);
    return NS::TransferPtr(cache.Ctx.Device->newDepthStencilState(descriptor.get()));
}

NS::SharedPtr<MTL::FunctionConstantValues> MakeConstantValues(const FunctionRef &ref) {
    auto values = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    for (const auto &constant : ref.Constants) {
        if (constant.Type == MTL::DataTypeBool) {
            const bool value = constant.Value != 0;
            values->setConstantValue(&value, constant.Type, NS::UInteger(constant.Index));
        } else {
            values->setConstantValue(&constant.Value, constant.Type, NS::UInteger(constant.Index));
        }
    }
    return values;
}

NS::SharedPtr<MTL4::FunctionDescriptor> MakeFunctionDescriptor(LibraryCache &cache, const FunctionRef &ref) {
    const auto *library = cache.Get(ref.Path, ref.Defines);
    auto function = NS::TransferPtr(MTL4::LibraryFunctionDescriptor::alloc()->init());
    function->setName(Str(ref.Name).get());
    function->setLibrary(library);
    if (ref.Constants.empty()) return function;

    const auto values = MakeConstantValues(ref);
    auto specialized = NS::TransferPtr(MTL4::SpecializedFunctionDescriptor::alloc()->init());
    specialized->setFunctionDescriptor(function.get());
    specialized->setConstantValues(values.get());
    return specialized;
}

// Metal builds a serializer's capture in a directory under the user cache and never removes it.
// The first captured pipeline creates the directory, and a private method of the capture's binary archive reports its path.
// The result is empty unless the reported path has the shape of a capture directory.
std::filesystem::path CaptureDirectory(MTL4::PipelineDataSetSerializer *serializer) {
    const auto send = [](void *object, const char *name, auto... args) -> void * {
        auto *const selector = sel_registerName(name);
        if (!object || !class_respondsToSelector(object_getClass(id(object)), selector)) return nullptr;
        return reinterpret_cast<void *(*)(void *, SEL, decltype(args)...)>(objc_msgSend)(object, selector, args...);
    };
    NS::Error *error = nullptr;
    auto *const reported = send(send(serializer, "destinationBinaryArchive"), "getArchiveIDWithError:", &error);
    const auto *const text = static_cast<const char *>(send(reported, "UTF8String"));
    if (!text) return {};
    const auto path = std::filesystem::path{text}.lexically_normal();
    const bool capture = path.filename().string().starts_with("gpuarchiver-") && path.parent_path().filename() == "com.apple.gpuarchiver";
    return capture ? path : std::filesystem::path{};
}
} // namespace

LibraryCache::LibraryCache(const Context &ctx, std::filesystem::path shaders_dir, std::filesystem::path archive, ArchiveMode mode)
    : Ctx(ctx), ShadersDir(std::move(shaders_dir)), ArchivePath(std::move(archive)), Mode(mode) {
    const AutoreleaseScope pool;
    // Shader validation requires live instrumentation and cannot use binary pipeline archives.
    if (const auto *validation = std::getenv("MTL_SHADER_VALIDATION"); validation && std::atoi(validation) != 0) {
        ArchivePath.clear();
        Mode = ArchiveMode::Read;
    }
    NS::Error *error = nullptr;
    auto opened = ArchivePath.empty() ? NS::SharedPtr<MTL4::Archive>{} : NS::TransferPtr(Ctx.Device->newArchive(NS::URL::fileURLWithPath(Str(ArchivePath.string()).get()), &error));
    const auto compiler_descriptor = NS::TransferPtr(MTL4::CompilerDescriptor::alloc()->init());
    if (Mode == ArchiveMode::Write) {
        // A compiler with a serializer reads the system compile cache but never adds to it.
        const auto serializer_descriptor = NS::TransferPtr(MTL4::PipelineDataSetSerializerDescriptor::alloc()->init());
        serializer_descriptor->setConfiguration(MTL4::PipelineDataSetSerializerConfigurationCaptureBinaries);
        Serializer = NS::TransferPtr(Ctx.Device->newPipelineDataSetSerializer(serializer_descriptor.get()));
        compiler_descriptor->setPipelineDataSetSerializer(Serializer.get());
        // The serializer captures pipelines the compiler takes from a lookup archive.
        if (opened) {
            Lookup = NS::TransferPtr(MTL4::CompilerTaskOptions::alloc()->init());
            Lookup->setLookupArchives(NS::Array::array(opened.get()));
        }
    } else {
        Archive = std::move(opened);
    }
    error = nullptr;
    Compiler = NS::TransferPtr(Ctx.Device->newCompiler(compiler_descriptor.get(), &error));
    if (!Compiler) {
        throw std::runtime_error(std::format("Failed to create the Metal pipeline compiler:\n{}", error ? error->localizedDescription()->utf8String() : "unknown"));
    }
}

std::unique_ptr<LibraryCache> LibraryCache::PrewarmCache() const {
    if (!Archive) throw std::runtime_error("Offline pipeline archive is unavailable.");
    return std::make_unique<LibraryCache>(Ctx, ShadersDir, ArchivePath, ArchiveMode::ReadOnly);
}

LibraryCache::~LibraryCache() {
    const AutoreleaseScope pool;
    // Removes the capture of pipelines created since the last write.
    if (Serializer) {
        std::error_code ec;
        std::filesystem::remove_all(CaptureDirectory(Serializer.get()), ec);
    }
    AutoreleaseScope::Release(Entries, Archive, Lookup, Compiler, Serializer);
}

bool LibraryCache::WriteArchive() {
    const AutoreleaseScope pool;
    if (!Serializer) return true;
    std::error_code ec;
    std::filesystem::create_directories(ArchivePath.parent_path(), ec);
    auto tmp = ArchivePath;
    tmp += std::format(".tmp.{}", getpid());
    NS::Error *error = nullptr;
    // Serialization reads the capture directory and moves the serializer to a new one for later pipelines.
    const auto capture_directory = CaptureDirectory(Serializer.get());
    const bool serialized = Serializer->serializeAsArchiveAndFlushToURL(NS::URL::fileURLWithPath(Str(tmp.string()).get()), &error);
    std::filesystem::remove_all(capture_directory, ec);
    if (serialized) std::filesystem::rename(tmp, ArchivePath, ec);
    const bool saved = serialized && !ec;
    std::filesystem::remove(tmp, ec);
    return saved;
}

NS::SharedPtr<MTL::ComputePipelineState> LibraryCache::FindComputePipeline(const MTL4::ComputePipelineDescriptor *descriptor) const {
    if (!Archive) return {};
    const AutoreleaseScope pool;
    NS::Error *error = nullptr;
    auto state = NS::TransferPtr(Archive->newComputePipelineState(descriptor, &error));
    if (state) ++ArchiveHits;
    return state;
}

NS::SharedPtr<MTL::RenderPipelineState> LibraryCache::FindRenderPipeline(const MTL4::PipelineDescriptor *descriptor) const {
    if (!Archive) return {};
    const AutoreleaseScope pool;
    NS::Error *error = nullptr;
    auto state = NS::TransferPtr(Archive->newRenderPipelineState(descriptor, &error));
    if (state) ++ArchiveHits;
    return state;
}

MTL::Library *LibraryCache::Get(const std::filesystem::path &relative_path, const std::vector<std::string> &defines) {
    const AutoreleaseScope pool;
    auto key = relative_path.string();
    for (const auto &define : defines) key += "|" + define;
    auto &entry = Entries[key];
    if (entry.Library && DepsUnchanged(entry.Deps, ShadersDir)) return entry.Library.get();

    const auto source = msl::Load(ShadersDir, relative_path, defines);
    auto binary = ShadersDir / relative_path;
    binary.replace_extension(".metallib");
    const bool fresh = defines.empty() && source.CanLoadBinary(ShadersDir, binary);
    if (Mode == ArchiveMode::ReadOnly && !fresh) throw std::runtime_error(std::format("Offline shader library is stale or missing: '{}'", relative_path.string()));
    NS::Error *error = nullptr;
    auto library = fresh ?
        NS::TransferPtr(Ctx.Device->newLibrary(Str(binary.string()).get(), &error)) :
        NS::TransferPtr(Ctx.Device->newLibrary(Str(source.Text).get(), msl::CompileOptions().get(), &error));
    if (!library) {
        throw std::runtime_error(std::format("Failed to load shader '{}' from {}:\n{}", relative_path.string(), fresh ? binary.string() : "source", error ? error->localizedDescription()->utf8String() : "unknown"));
    }
    if (fresh) ++BinaryLibraries;
    else ++SourceLibraries;
    decltype(entry.Deps) deps;
    deps.reserve(source.Files.size());
    std::error_code ec;
    for (const auto &file : source.Files) deps.emplace_back(file, std::filesystem::last_write_time(ShadersDir / file, ec));
    entry = Entry{std::move(library), std::move(deps)};
    return entry.Library.get();
}

namespace {
// `set_primary` installs the vertex or mesh function and its stage settings on the descriptor.
// A pipeline whose fragment stage links `fragment_functions` compiles with them, outside the archive.
template<typename Descriptor>
RenderPipeline MakeRenderPipeline(
    LibraryCache &cache, const FunctionRef &primary, const std::optional<FunctionRef> &fragment, const PassFormats &formats,
    std::span<const BlendState> blends, std::optional<DepthState> depth, const char *family, auto &&set_primary, const NS::Array *fragment_functions = nullptr
) {
    const AutoreleaseScope pool;
    const auto descriptor = NS::TransferPtr(Descriptor::alloc()->init());
    descriptor->setLabel(Str(fragment ? std::format("{}/{}", primary.Name, fragment->Name) : std::string{primary.Name}).get());
    const auto primary_function = MakeFunctionDescriptor(cache, primary);
    set_primary(descriptor.get(), primary_function.get());
    NS::SharedPtr<MTL4::FunctionDescriptor> fragment_function;
    if (fragment) {
        fragment_function = MakeFunctionDescriptor(cache, *fragment);
        descriptor->setFragmentFunctionDescriptor(fragment_function.get());
        descriptor->setSupportFragmentBinaryLinking(fragment->BinaryLinking);
    }
    ConfigureColorAttachments(descriptor->colorAttachments(), formats, blends);
    NS::Error *error = nullptr;
    auto state = fragment_functions ? NS::SharedPtr<MTL::RenderPipelineState>{} : cache.FindRenderPipeline(descriptor.get());
    const auto archive_hit = bool(state);
    if (fragment_functions) {
        const auto linking = NS::TransferPtr(MTL4::RenderPipelineDynamicLinkingDescriptor::alloc()->init());
        linking->fragmentLinkingDescriptor()->setBinaryLinkedFunctions(fragment_functions);
        state = NS::TransferPtr(cache.PipelineCompiler()->newRenderPipelineState(descriptor.get(), linking.get(), cache.CompileOptions(), &error));
    } else if (!state) state = NS::TransferPtr(cache.PipelineCompiler()->newRenderPipelineState(descriptor.get(), cache.CompileOptions(), &error));
    if (!state) throw std::runtime_error(std::format("Failed to create the {} pipeline for '{}':\n{}", family, primary.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
    if (!archive_hit) cache.NotePipelineCreated();
    return {std::move(state), MakeDepthState(cache, depth)};
}
} // namespace

RenderPipeline MakeRenderPipeline(
    LibraryCache &cache, FunctionRef vertex, std::optional<FunctionRef> fragment, const PassFormats &formats,
    const std::vector<BlendState> &blends, std::optional<DepthState> depth
) {
    return MakeRenderPipeline<MTL4::RenderPipelineDescriptor>(cache, vertex, fragment, formats, blends, depth, "render", [](auto *descriptor, auto *function) {
        descriptor->setVertexFunctionDescriptor(function);
    });
}

RenderPipeline MakeMeshPipeline(
    LibraryCache &cache, FunctionRef mesh, std::optional<FunctionRef> fragment, const PassFormats &formats,
    const std::vector<BlendState> &blends, std::optional<DepthState> depth, const NS::Array *fragment_functions
) {
    const auto set_mesh = [](auto *descriptor, auto *function) {
        descriptor->setMeshFunctionDescriptor(function);
        descriptor->setMaxTotalThreadsPerMeshThreadgroup(160);
        descriptor->setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(true);
        descriptor->setMaxTotalThreadgroupsPerMeshGrid(MaxMeshThreadgroupsPerGrid);
    };
    return MakeRenderPipeline<MTL4::MeshRenderPipelineDescriptor>(cache, mesh, fragment, formats, blends, depth, "mesh render", set_mesh, fragment_functions);
}

ComputePipeline::ComputePipeline(LibraryCache &cache, FunctionRef fn) {
    const AutoreleaseScope pool;
    const auto descriptor = NS::TransferPtr(MTL4::ComputePipelineDescriptor::alloc()->init());
    descriptor->setLabel(Str(fn.Name).get());
    const auto function = MakeFunctionDescriptor(cache, fn);
    descriptor->setComputeFunctionDescriptor(function.get());
    descriptor->setSupportBinaryLinking(fn.BinaryLinking);
    NS::Error *error = nullptr;
    PipelineState = cache.FindComputePipeline(descriptor.get());
    const auto archive_hit = bool(PipelineState);
    if (!PipelineState && cache.ArchiveOnly()) throw std::runtime_error(std::format("Offline compute pipeline is absent: '{}'", fn.Name));
    if (!PipelineState) PipelineState = NS::TransferPtr(cache.PipelineCompiler()->newComputePipelineState(descriptor.get(), cache.CompileOptions(), &error));
    if (!PipelineState) {
        throw std::runtime_error(std::format("Failed to create the compute pipeline for '{}':\n{}", fn.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
    }
    if (!archive_hit) cache.NotePipelineCreated();
}
} // namespace mtl
