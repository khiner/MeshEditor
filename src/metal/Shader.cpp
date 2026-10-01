#include "metal/AutoreleaseScope.h"
#include "metal/Shader.h"

#include "metal/MetalCpp.h"

#include <algorithm>
#include <format>

namespace mtl {
LibraryCache::LibraryCache(LibraryCache &&) noexcept = default;
void LibraryCache::Clear() { const AutoreleaseScope pool; Entries.clear(); }

void RenderPipeline::Bind(MTL::RenderCommandEncoder *encoder) const {
    encoder->setRenderPipelineState(PipelineState.get());
    encoder->setDepthStencilState(DepthStencilState.get());
}

namespace {
bool DepsUnchanged(const std::vector<std::pair<std::filesystem::path, std::filesystem::file_time_type>> &deps, const std::filesystem::path &root) {
    std::error_code ec;
    for (const auto &[relative, mtime] : deps) {
        const auto current = std::filesystem::last_write_time(root / relative, ec);
        if (ec || current != mtime) return false;
    }
    return true;
}

std::vector<std::filesystem::path> PipelineArchives(const std::filesystem::path &base) {
    std::error_code ec;
    std::vector<std::filesystem::path> paths;
    const auto directory = base.parent_path().empty() ? std::filesystem::path{"."} : base.parent_path();
    const auto prefix = base.stem().string() + ".";
    const auto extension = base.extension().string();
    for (std::filesystem::directory_iterator it{directory, ec}, end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        const auto name = it->path().filename().string();
        if (name == base.filename().string() || (name.starts_with(prefix) && name.ends_with(extension))) paths.push_back(it->path());
    }
    std::sort(paths.begin(), paths.end(), [](const auto &a, const auto &b) {
        std::error_code left_error, right_error;
        return std::filesystem::last_write_time(a, left_error) > std::filesystem::last_write_time(b, right_error);
    });
    return paths;
}

void PrunePipelineArchives(const std::filesystem::path &base) {
    constexpr uint32_t MaxDeltaArchives = 64u;
    constexpr uint64_t MaxArchiveBytes = 256u << 20;
    std::error_code ec;
    uint64_t bytes = std::filesystem::file_size(base, ec);
    if (ec) bytes = 0u;
    uint32_t deltas = 0u;
    for (const auto &path : PipelineArchives(base)) {
        if (path.filename() == base.filename()) continue;
        const auto size = std::filesystem::file_size(path, ec);
        if (ec) continue;
        if (deltas == 0u || (deltas < MaxDeltaArchives && bytes + size <= MaxArchiveBytes)) {
            ++deltas;
            bytes += size;
        } else std::filesystem::remove(path, ec);
    }
}

// Both Metal descriptor families share the attachment state; only the blend enable API differs.
template<typename Attachments>
void ConfigureColorAttachments(Attachments *attachments, const PassFormats &formats, const std::vector<BlendState> &blends) {
    for (size_t i = 0; i < formats.Color.size(); ++i) {
        auto *attachment = attachments->object(i);
        attachment->setPixelFormat(formats.Color[i]);
        const auto blend = i < blends.size() ? blends[i] : NoBlend;
        attachment->setWriteMask(blend.WriteMask ? MTL::ColorWriteMaskAll : MTL::ColorWriteMaskNone);
        if constexpr (requires { attachment->setBlendingEnabled(blend.Enabled); }) attachment->setBlendingEnabled(blend.Enabled);
        else attachment->setBlendingState(blend.Enabled ? MTL4::BlendStateEnabled : MTL4::BlendStateDisabled);
        if (!blend.Enabled) continue;
        attachment->setSourceRGBBlendFactor(blend.SourceRgb);
        attachment->setDestinationRGBBlendFactor(blend.DestRgb);
        attachment->setRgbBlendOperation(MTL::BlendOperationAdd);
        attachment->setSourceAlphaBlendFactor(blend.SourceAlpha);
        attachment->setDestinationAlphaBlendFactor(blend.DestAlpha);
        attachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
    }
}

// Classic pipeline descriptors also specify their depth format here.
template<typename Descriptor>
void ConfigureAttachments(Descriptor *descriptor, const PassFormats &formats, const std::vector<BlendState> &blends) {
    ConfigureColorAttachments(descriptor->colorAttachments(), formats, blends);
    if (formats.Depth != MTL::PixelFormatInvalid) descriptor->setDepthAttachmentPixelFormat(formats.Depth);
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

NS::SharedPtr<MTL::Function> MakeFunction(LibraryCache &cache, const FunctionRef &ref) {
    auto *library = cache.Get(ref.Path, ref.Defines);
    NS::Error *error = nullptr;
    auto function = ref.Constants.empty() ?
        NS::TransferPtr(library->newFunction(Str(ref.Name).get())) :
        NS::TransferPtr(library->newFunction(Str(ref.Name).get(), MakeConstantValues(ref).get(), &error));
    if (!function) {
        throw std::runtime_error(std::format("No function '{}' in '{}':\n{}", ref.Name, ref.Path.string(), error ? error->localizedDescription()->utf8String() : "unknown"));
    }
    return function;
}

NS::SharedPtr<MTL4::FunctionDescriptor> MakeFunctionDescriptor(LibraryCache &cache, const FunctionRef &ref) {
    auto *library = cache.Get(ref.Path, ref.Defines);
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
} // namespace

LibraryCache::LibraryCache(const Context &ctx, std::filesystem::path shaders_dir, std::filesystem::path pipeline_archive,
                           std::filesystem::path builtin_archive, bool prune_archive_chunks, bool archive_only)
    : Ctx(ctx), ShadersDir(std::move(shaders_dir)), ArchivePath(std::move(pipeline_archive)),
      BuiltinArchivePath(std::move(builtin_archive)),
      PruneArchiveChunks(prune_archive_chunks), ReadArchiveOnly(archive_only) {
    const AutoreleaseScope pool;
    if (!Ctx.Device->supportsFamily(MTL::GPUFamilyMetal4)) return;
    const auto compiler_descriptor = NS::TransferPtr(MTL4::CompilerDescriptor::alloc()->init());
    NS::SharedPtr<MTL4::PipelineDataSetSerializer> serializer;
    if (!ArchivePath.empty() && !ReadArchiveOnly) {
        const auto serializer_descriptor = NS::TransferPtr(MTL4::PipelineDataSetSerializerDescriptor::alloc()->init());
        serializer_descriptor->setConfiguration(MTL4::PipelineDataSetSerializerConfigurationCaptureBinaries);
        serializer = NS::TransferPtr(Ctx.Device->newPipelineDataSetSerializer(serializer_descriptor.get()));
        compiler_descriptor->setPipelineDataSetSerializer(serializer.get());
    }
    NS::Error *error = nullptr;
    auto compiler = NS::TransferPtr(Ctx.Device->newCompiler(compiler_descriptor.get(), &error));
    if (!compiler) {
        throw std::runtime_error(std::format("Failed to create the Metal pipeline compiler:\n{}", error ? error->localizedDescription()->utf8String() : "unknown"));
    }

    // Archive chunks are immutable. Rewriting a serializer's partial capture
    // would discard pipelines this session did not request.
    std::vector<NS::SharedPtr<MTL4::Archive>> archives;
    const auto load = [&](const std::filesystem::path &path) {
        error = nullptr;
        if (auto archive = NS::TransferPtr(Ctx.Device->newArchive(NS::URL::fileURLWithPath(Str(path.string()).get()), &error))) archives.push_back(std::move(archive));
    };
    if (!ArchivePath.empty()) {
        for (const auto &path : PipelineArchives(ArchivePath)) load(path);
    }
    if (!BuiltinArchivePath.empty() && BuiltinArchivePath != ArchivePath) {
        for (const auto &path : PipelineArchives(BuiltinArchivePath)) load(path);
    }
    Serializer = std::move(serializer);
    Compiler = std::move(compiler);
    Archives = std::move(archives);
}

std::unique_ptr<LibraryCache> LibraryCache::PrewarmCache() const {
    if ((BuiltinArchivePath.empty() || PipelineArchives(BuiltinArchivePath).empty()) &&
        (ArchivePath.empty() || PipelineArchives(ArchivePath).empty()))
        throw std::runtime_error("Offline pipeline archive is unavailable.");
    return std::make_unique<LibraryCache>(Ctx,ShadersDir,ArchivePath,BuiltinArchivePath,false,true);
}

// Write only newly compiled pipelines into an immutable chunk. Existing
// archives keep every pipeline the current process did not use.
LibraryCache::~LibraryCache() {
    const AutoreleaseScope pool;
    (void)FlushArchive();
    AutoreleaseScope::Release(Entries, Archives, Compiler, Serializer);
}

bool LibraryCache::FlushArchive() {
    const AutoreleaseScope pool;
    if (ArchivePath.empty() || !Serializer || !PipelineCreated) return true;
    std::error_code ec;
    std::filesystem::create_directories(ArchivePath.parent_path(), ec);
    if (ec) return false;
    auto destination = ArchivePath;
    if (std::filesystem::exists(destination, ec)) {
        const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
        destination = ArchivePath.parent_path() /
            std::format("{}.{}.{}{}", ArchivePath.stem().string(), stamp, getpid(), ArchivePath.extension().string());
    }
    auto tmp = destination;
    tmp += std::format(".tmp.{}", getpid());
    NS::Error *error = nullptr;
    const bool serialized = Serializer->serializeAsArchiveAndFlushToURL(NS::URL::fileURLWithPath(Str(tmp.string()).get()), &error);
    if (serialized) std::filesystem::rename(tmp, destination, ec);
    const bool saved = serialized && !ec;
    std::filesystem::remove(tmp, ec);
    if (!saved) return false;
    PipelineCreated = false;
    if (PruneArchiveChunks) PrunePipelineArchives(ArchivePath);
    return true;
}

NS::SharedPtr<MTL::ComputePipelineState> LibraryCache::FindComputePipeline(const MTL4::ComputePipelineDescriptor *descriptor) const {
    const AutoreleaseScope pool;
    for (const auto &archive : Archives) {
        NS::Error *error = nullptr;
        if (auto state = NS::TransferPtr(archive->newComputePipelineState(descriptor, &error))) {
            ++ArchiveHits;
            return state;
        }
    }
    return {};
}

NS::SharedPtr<MTL::RenderPipelineState> LibraryCache::FindRenderPipeline(const MTL4::PipelineDescriptor *descriptor) const {
    const AutoreleaseScope pool;
    for (const auto &archive : Archives) {
        NS::Error *error = nullptr;
        if (auto state = NS::TransferPtr(archive->newRenderPipelineState(descriptor, &error))) {
            ++ArchiveHits;
            return state;
        }
    }
    return {};
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
    std::error_code ec;
    const auto binary_time = std::filesystem::last_write_time(binary, ec);
    bool fresh = !ec && defines.empty();
    for (const auto &file : source.Files) {
        std::error_code source_error;
        const auto source_time = std::filesystem::last_write_time(ShadersDir / file, source_error);
        if (source_error || source_time > binary_time) { fresh = false; break; }
    }
    if (ReadArchiveOnly && !fresh) throw std::runtime_error(std::format("Offline shader library is stale or missing: '{}'",relative_path.string()));
    NS::Error *error = nullptr;
    auto library = fresh ?
        NS::TransferPtr(Ctx.Device->newLibrary(Str(binary.string()).get(), &error)) :
        NS::TransferPtr(Ctx.Device->newLibrary(Str(source.Text).get(), static_cast<MTL::CompileOptions *>(nullptr), &error));
    if (!library) {
        throw std::runtime_error(std::format("Failed to load shader '{}' from {}:\n{}", relative_path.string(),
            fresh ? binary.string() : "source", error ? error->localizedDescription()->utf8String() : "unknown"));
    }
    if (fresh) ++BinaryLibraries; else ++SourceLibraries;
    decltype(entry.Deps) deps;
    deps.reserve(source.Files.size());
    for (const auto &file : source.Files) deps.emplace_back(file, std::filesystem::last_write_time(ShadersDir / file, ec));
    entry = Entry{std::move(library), std::move(deps)};
    return entry.Library.get();
}

RenderPipeline MakeRenderPipeline(
    LibraryCache &cache, FunctionRef vertex, std::optional<FunctionRef> fragment, PassFormats formats,
    std::vector<BlendState> blends, std::optional<DepthState> depth
) {
    const AutoreleaseScope pool;
    if (!cache.PipelineCompiler()) {
        const auto descriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
        const auto vertex_function = MakeFunction(cache, vertex);
        descriptor->setVertexFunction(vertex_function.get());
        NS::SharedPtr<MTL::Function> fragment_function;
        if (fragment) {
            fragment_function = MakeFunction(cache, *fragment);
            descriptor->setFragmentFunction(fragment_function.get());
        }
        ConfigureAttachments(descriptor.get(), formats, blends);
        NS::Error *error = nullptr;
        auto state = NS::TransferPtr(cache.Ctx.Device->newRenderPipelineState(descriptor.get(), &error));
        if (!state) {
            throw std::runtime_error(std::format("Failed to create the render pipeline for '{}':\n{}", vertex.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
        }
        return {std::move(state), MakeDepthState(cache, depth)};
    }
    const auto descriptor = NS::TransferPtr(MTL4::RenderPipelineDescriptor::alloc()->init());
    const auto vertex_function = MakeFunctionDescriptor(cache, vertex);
    descriptor->setVertexFunctionDescriptor(vertex_function.get());
    NS::SharedPtr<MTL4::FunctionDescriptor> fragment_function;
    if (fragment) {
        fragment_function = MakeFunctionDescriptor(cache, *fragment);
        descriptor->setFragmentFunctionDescriptor(fragment_function.get());
    }
    ConfigureColorAttachments(descriptor->colorAttachments(), formats, blends);

    NS::Error *error = nullptr;
    auto state = cache.FindRenderPipeline(descriptor.get());
    const bool archive_hit = bool(state);
    if (!state) state = NS::TransferPtr(cache.PipelineCompiler()->newRenderPipelineState(descriptor.get(), nullptr, &error));
    if (!state) {
        throw std::runtime_error(std::format("Failed to create the render pipeline for '{}':\n{}", vertex.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
    }
    if (!archive_hit) cache.NotePipelineCreated();
    return {std::move(state), MakeDepthState(cache, depth)};
}

RenderPipeline MakeMeshPipeline(
    LibraryCache &cache, FunctionRef mesh, std::optional<FunctionRef> fragment, PassFormats formats,
    std::vector<BlendState> blends, std::optional<DepthState> depth
) {
    const AutoreleaseScope pool;
    if (!cache.PipelineCompiler()) {
        const auto descriptor = NS::TransferPtr(MTL::MeshRenderPipelineDescriptor::alloc()->init());
        const auto mesh_function = MakeFunction(cache, mesh);
        descriptor->setMeshFunction(mesh_function.get());
        descriptor->setMaxTotalThreadsPerMeshThreadgroup(160);
        descriptor->setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(true);
        descriptor->setMaxTotalThreadgroupsPerMeshGrid(MaxMeshThreadgroupsPerGrid);
        NS::SharedPtr<MTL::Function> fragment_function;
        if (fragment) {
            fragment_function = MakeFunction(cache, *fragment);
            descriptor->setFragmentFunction(fragment_function.get());
        }
        ConfigureAttachments(descriptor.get(), formats, blends);
        NS::Error *error = nullptr;
        auto state = NS::TransferPtr(cache.Ctx.Device->newRenderPipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
        if (!state) {
            throw std::runtime_error(std::format("Failed to create the mesh render pipeline for '{}':\n{}", mesh.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
        }
        return {std::move(state), MakeDepthState(cache, depth)};
    }
    const auto descriptor = NS::TransferPtr(MTL4::MeshRenderPipelineDescriptor::alloc()->init());
    const auto mesh_function = MakeFunctionDescriptor(cache, mesh);
    descriptor->setMeshFunctionDescriptor(mesh_function.get());
    descriptor->setMaxTotalThreadsPerMeshThreadgroup(160);
    descriptor->setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(true);
    descriptor->setMaxTotalThreadgroupsPerMeshGrid(MaxMeshThreadgroupsPerGrid);
    NS::SharedPtr<MTL4::FunctionDescriptor> fragment_function;
    if (fragment) {
        fragment_function = MakeFunctionDescriptor(cache, *fragment);
        descriptor->setFragmentFunctionDescriptor(fragment_function.get());
    }
    ConfigureColorAttachments(descriptor->colorAttachments(), formats, blends);

    NS::Error *error = nullptr;
    auto state = cache.FindRenderPipeline(descriptor.get());
    const bool archive_hit = bool(state);
    if (!state) state = NS::TransferPtr(cache.PipelineCompiler()->newRenderPipelineState(descriptor.get(), nullptr, &error));
    if (!state) {
        throw std::runtime_error(std::format("Failed to create the mesh render pipeline for '{}':\n{}", mesh.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
    }
    if (!archive_hit) cache.NotePipelineCreated();
    return {std::move(state), MakeDepthState(cache, depth)};
}

ComputePipeline::ComputePipeline(LibraryCache &cache, FunctionRef fn) {
    const AutoreleaseScope pool;
    if (!cache.PipelineCompiler()) {
        const auto function = MakeFunction(cache, fn);
        NS::Error *error = nullptr;
        PipelineState = NS::TransferPtr(cache.Ctx.Device->newComputePipelineState(function.get(), &error));
        if (!PipelineState) {
            throw std::runtime_error(std::format("Failed to create the compute pipeline for '{}':\n{}", fn.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
        }
        return;
    }
    const auto descriptor = NS::TransferPtr(MTL4::ComputePipelineDescriptor::alloc()->init());
    descriptor->setLabel(Str(fn.Name).get());
    const auto function = MakeFunctionDescriptor(cache, fn);
    descriptor->setComputeFunctionDescriptor(function.get());
    NS::Error *error = nullptr;
    PipelineState = cache.FindComputePipeline(descriptor.get());
    const bool archive_hit = bool(PipelineState);
    if (!PipelineState && cache.ArchiveOnly()) throw std::runtime_error(std::format("Offline compute pipeline is absent: '{}'",fn.Name));
    if (!PipelineState) PipelineState = NS::TransferPtr(cache.PipelineCompiler()->newComputePipelineState(descriptor.get(), nullptr, &error));
    if (!PipelineState) {
        throw std::runtime_error(std::format("Failed to create the compute pipeline for '{}':\n{}", fn.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
    }
    if (!archive_hit) cache.NotePipelineCreated();
}
} // namespace mtl
