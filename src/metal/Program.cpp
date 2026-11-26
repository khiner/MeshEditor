#include "metal/Program.h"
#include "Parallel.h"
#include "metal/AutoreleaseScope.h"

#include "metal/MetalCpp.h"

#include <format>
#include <thread>

namespace mtl {
namespace {
std::string ErrorText(const NS::Error *error) { return error ? error->localizedDescription()->utf8String() : "unknown"; }

// Returns the binary function of each source, in order.
// Consecutive sources share one of at most one library per core, and the libraries compile concurrently.
std::vector<NS::SharedPtr<MTL4::BinaryFunction>> CompilePrograms(const ProgramRegistry &registry, std::span<const std::string *const> sources) {
    const AutoreleaseScope pool;
    const auto count = uint32_t(sources.size());
    const auto library_count = std::min(count, std::max(std::thread::hardware_concurrency(), 1u));
    // Library l holds sources [first(l), first(l + 1)), and source i defines function `Program{i - first(l)}`.
    const auto first = [&](uint32_t l) { return uint32_t(uint64_t(l) * count / library_count); };
    std::vector<NS::SharedPtr<MTL::Library>> libraries(library_count);
    std::vector<std::string> library_errors(library_count);
    const auto options = registry.MathMode ? NS::TransferPtr(MTL::CompileOptions::alloc()->init()) : NS::SharedPtr<MTL::CompileOptions>{};
    if (registry.MathMode) options->setMathMode(*registry.MathMode);
    ParallelFor(library_count, [&](uint32_t l) {
        const AutoreleaseScope library_pool;
        std::string text;
        for (auto i = first(l); i < first(l + 1); ++i) text += std::format("#define ProgramFunction Program{}\n{}\n#undef ProgramFunction\n", i - first(l), *sources[i]);
        NS::Error *error = nullptr;
        libraries[l] = NS::TransferPtr(registry.Ctx.Device->newLibrary(Str(text).get(), options.get(), &error));
        if (!libraries[l]) library_errors[l] = ErrorText(error);
    });
    for (uint32_t l = 0; l < library_count; ++l) {
        if (!libraries[l]) throw std::runtime_error(std::format("Failed to compile programs:\n{}", library_errors[l]));
    }

    std::vector<NS::SharedPtr<MTL4::BinaryFunction>> functions(count);
    std::vector<std::string> function_errors(count);
    auto *const slots = functions.data();
    auto *const errors = function_errors.data();
    const auto group = dispatch_group_create();
    for (uint32_t l = 0; l < library_count; ++l) {
        for (auto i = first(l); i < first(l + 1); ++i) {
            const auto name = Str(std::format("Program{}", i - first(l)));
            const auto function = NS::TransferPtr(MTL4::LibraryFunctionDescriptor::alloc()->init());
            function->setName(name.get());
            function->setLibrary(libraries[l].get());
            const auto descriptor = NS::TransferPtr(MTL4::BinaryFunctionDescriptor::alloc()->init());
            descriptor->setName(name.get());
            descriptor->setFunctionDescriptor(function.get());
            dispatch_group_enter(group);
            (void)NS::TransferPtr(registry.Compiler->newBinaryFunction(descriptor.get(), nullptr, ^(MTL4::BinaryFunction *binary, NS::Error *error) {
                const AutoreleaseScope handler_pool;
                if (binary) slots[i] = NS::RetainPtr(binary);
                else errors[i] = ErrorText(error);
                dispatch_group_leave(group);
            }));
        }
    }
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    dispatch_release(group);
    for (uint32_t i = 0; i < count; ++i) {
        if (!functions[i]) throw std::runtime_error(std::format("Failed to compile a program to a binary function:\n{}", errors[i]));
    }
    return functions;
}

NS::SharedPtr<MTL::ComputePipelineState> Grow(MTL::ComputePipelineState *state, const NS::Array *functions) {
    NS::Error *error = nullptr;
    auto grown = NS::TransferPtr(state->newComputePipelineStateWithBinaryFunctions(functions, &error));
    if (!grown) throw std::runtime_error(std::format("Failed to link programs into a compute driver:\n{}", ErrorText(error)));
    return grown;
}
NS::SharedPtr<MTL::RenderPipelineState> Grow(MTL::RenderPipelineState *state, const NS::Array *functions) {
    const auto descriptor = NS::TransferPtr(MTL4::RenderPipelineBinaryFunctionsDescriptor::alloc()->init());
    descriptor->setFragmentAdditionalBinaryFunctions(functions);
    NS::Error *error = nullptr;
    auto grown = NS::TransferPtr(state->newRenderPipelineState(descriptor.get(), &error));
    if (!grown) throw std::runtime_error(std::format("Failed to link programs into a render driver:\n{}", ErrorText(error)));
    return grown;
}

MTL::FunctionHandle *Handle(MTL::ComputePipelineState *state, const MTL4::BinaryFunction *function) { return state->functionHandle(function); }
MTL::FunctionHandle *Handle(MTL::RenderPipelineState *state, const MTL4::BinaryFunction *function) { return state->functionHandle(function, MTL::RenderStageFragment); }

NS::SharedPtr<MTL::VisibleFunctionTable> NewTable(MTL::ComputePipelineState *state, const MTL::VisibleFunctionTableDescriptor *descriptor) {
    return NS::TransferPtr(state->newVisibleFunctionTable(descriptor));
}
NS::SharedPtr<MTL::VisibleFunctionTable> NewTable(MTL::RenderPipelineState *state, const MTL::VisibleFunctionTableDescriptor *descriptor) {
    return NS::TransferPtr(state->newVisibleFunctionTable(descriptor, MTL::RenderStageFragment));
}

// Links programs from `linked` on into the driver, and replaces its table with one of every program.
template<typename P> void Link(const ProgramRegistry &registry, Driver<P> &driver, uint32_t linked) {
    const AutoreleaseScope pool;
    const auto count = uint32_t(registry.Functions.size());
    auto pipeline = driver.Current->Pipeline;
    if (linked < count) {
        std::vector<const NS::Object *> functions;
        for (auto i = driver.Make ? 0u : linked; i < count; ++i) functions.push_back(registry.Functions[i].get());
        const auto *array = NS::Array::array(functions.data(), functions.size());
        if (driver.Make) pipeline = driver.Make(array);
        else pipeline.PipelineState = Grow(pipeline.PipelineState.get(), array);
    }
    const auto descriptor = NS::TransferPtr(MTL::VisibleFunctionTableDescriptor::alloc()->init());
    descriptor->setFunctionCount(std::max(count, 1u));
    auto table = NewTable(pipeline.PipelineState.get(), descriptor.get());
    std::vector<const MTL::FunctionHandle *> handles;
    for (const auto &function : registry.Functions) handles.push_back(Handle(pipeline.PipelineState.get(), function.get()));
    table->setFunctions(handles.data(), NS::Range{0u, count});
    driver.Current = std::make_shared<const typename Driver<P>::Linked>(std::move(pipeline), Resident{registry.Ctx, std::move(table)});
}
} // namespace

ProgramRegistry::ProgramRegistry(const Context &ctx, std::optional<MTL::MathMode> math_mode) : Ctx(ctx), MathMode(math_mode) {
    const AutoreleaseScope pool;
    const auto descriptor = NS::TransferPtr(MTL4::CompilerDescriptor::alloc()->init());
    NS::Error *error = nullptr;
    Compiler = NS::TransferPtr(Ctx.Device->newCompiler(descriptor.get(), &error));
    if (!Compiler) throw std::runtime_error(std::format("Failed to create the Metal program compiler:\n{}", ErrorText(error)));
}

ProgramRegistry::~ProgramRegistry() { AutoreleaseScope::Release(ComputeDrivers, RenderDrivers, Functions, Compiler); }

std::vector<uint32_t> AddPrograms(ProgramRegistry &registry, std::span<const std::string> sources) {
    const auto linked = uint32_t(registry.Functions.size());
    std::vector<uint32_t> indices;
    indices.reserve(sources.size());
    std::vector<const std::string *> added;
    for (const auto &source : sources) {
        const auto [it, inserted] = registry.Indices.try_emplace(store::HashBytes(std::as_bytes(std::span{source})), linked + uint32_t(added.size()));
        if (inserted) added.push_back(&source);
        indices.push_back(it->second);
    }
    if (added.empty()) return indices;
    for (auto &function : CompilePrograms(registry, added)) registry.Functions.push_back(std::move(function));
    for (auto &driver : registry.ComputeDrivers) Link(registry, driver, linked);
    for (auto &driver : registry.RenderDrivers) Link(registry, driver, linked);
    return indices;
}

ComputeDriver &AddDriver(ProgramRegistry &registry, ComputePipeline pipeline) {
    auto &driver = registry.ComputeDrivers.emplace_back(std::make_shared<const ComputeDriver::Linked>(std::move(pipeline)));
    Link(registry, driver, 0u);
    return driver;
}

RenderDriver &AddDriver(ProgramRegistry &registry, RenderPipeline pipeline) {
    auto &driver = registry.RenderDrivers.emplace_back(std::make_shared<const RenderDriver::Linked>(std::move(pipeline)));
    Link(registry, driver, 0u);
    return driver;
}

RenderDriver &AddDriver(ProgramRegistry &registry, std::function<RenderPipeline(const NS::Array *)> make) {
    const AutoreleaseScope pool;
    std::vector<const NS::Object *> functions;
    for (const auto &function : registry.Functions) functions.push_back(function.get());
    auto &driver = registry.RenderDrivers.emplace_back(std::make_shared<const RenderDriver::Linked>(make(NS::Array::array(functions.data(), functions.size()))), std::move(make));
    Link(registry, driver, uint32_t(functions.size()));
    return driver;
}

void Bind(ComputeEncoder encoder, const ComputeDriver &driver, uint32_t index) {
    driver.Current->Pipeline.Bind(encoder);
    encoder.Cmd->Arguments()->setResource(driver.Current->Table->gpuResourceID(), index);
    encoder.Cmd->Keep(driver.Current);
}

void Bind(RenderEncoder encoder, const RenderDriver &driver, uint32_t index) {
    driver.Current->Pipeline.Bind(encoder);
    encoder.Cmd->FragmentArguments()->setResource(driver.Current->Table->gpuResourceID(), index);
    encoder.Cmd->Keep(driver.Current);
}
} // namespace mtl
