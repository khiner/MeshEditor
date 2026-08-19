#include "metal/Shader.h"

#include <format>
#include <stdexcept>

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
} // namespace

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

RenderPipeline::RenderPipeline(
    LibraryCache &cache, FunctionRef vertex, std::optional<FunctionRef> fragment, PassFormats formats,
    std::vector<BlendState> blends, std::optional<DepthState> depth, float depth_bias
) : VertexFn(std::move(vertex)), FragmentFn(std::move(fragment)), Formats(std::move(formats)),
    Blends(std::move(blends)), Depth(depth), Bias(depth_bias) {
    Compile(cache);
}

void RenderPipeline::Compile(LibraryCache &cache) {
    const auto descriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
    const auto vertex_function = MakeFunction(cache, VertexFn);
    descriptor->setVertexFunction(vertex_function.get());
    NS::SharedPtr<MTL::Function> fragment_function;
    if (FragmentFn) {
        fragment_function = MakeFunction(cache, *FragmentFn);
        descriptor->setFragmentFunction(fragment_function.get());
    }

    for (size_t i = 0; i < Formats.Color.size(); ++i) {
        auto *attachment = descriptor->colorAttachments()->object(i);
        attachment->setPixelFormat(Formats.Color[i]);
        const auto blend = i < Blends.size() ? Blends[i] : NoBlend;
        attachment->setWriteMask(blend.WriteMask ? MTL::ColorWriteMaskAll : MTL::ColorWriteMaskNone);
        attachment->setBlendingEnabled(blend.Enabled);
        if (blend.Enabled) {
            attachment->setSourceRGBBlendFactor(blend.SourceRgb);
            attachment->setDestinationRGBBlendFactor(blend.DestRgb);
            attachment->setRgbBlendOperation(MTL::BlendOperationAdd);
            attachment->setSourceAlphaBlendFactor(blend.SourceAlpha);
            attachment->setDestinationAlphaBlendFactor(blend.DestAlpha);
            attachment->setAlphaBlendOperation(MTL::BlendOperationAdd);
        }
    }
    if (Formats.Depth != MTL::PixelFormatInvalid) descriptor->setDepthAttachmentPixelFormat(Formats.Depth);

    NS::Error *error = nullptr;
    PipelineState = NS::TransferPtr(cache.Ctx.Device->newRenderPipelineState(descriptor.get(), &error));
    if (!PipelineState) {
        throw std::runtime_error(std::format("Failed to create the render pipeline for '{}':\n{}", VertexFn.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
    }

    const auto depth_descriptor = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
    // No depth state at all means always pass and never write, which is what a depth-less pass wants.
    depth_descriptor->setDepthCompareFunction(Depth && Depth->Test ? Depth->Compare : MTL::CompareFunctionAlways);
    depth_descriptor->setDepthWriteEnabled(Depth && Depth->Write);
    DepthStencilState = NS::TransferPtr(cache.Ctx.Device->newDepthStencilState(depth_descriptor.get()));
}

ComputePipeline::ComputePipeline(LibraryCache &cache, FunctionRef fn) : Fn(std::move(fn)) { Compile(cache); }

void ComputePipeline::Compile(LibraryCache &cache) {
    const auto function = MakeFunction(cache, Fn);
    NS::Error *error = nullptr;
    PipelineState = NS::TransferPtr(cache.Ctx.Device->newComputePipelineState(function.get(), &error));
    if (!PipelineState) {
        throw std::runtime_error(std::format("Failed to create the compute pipeline for '{}':\n{}", Fn.Name, error ? error->localizedDescription()->utf8String() : "unknown"));
    }
}
} // namespace mtl
