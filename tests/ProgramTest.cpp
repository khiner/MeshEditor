#include "Near.h"
#include "TestPaths.h"
#include "metal/MetalCpp.h"
#include "metal/PassChain.h"
#include "metal/Program.h"
#include "metal/RenderTarget.h"
#include <boost/ut.hpp>

#include <fstream>
#include <functional>

using namespace boost::ut;

namespace {
constexpr uint32_t Width{16};
using Float4 = std::array<float, 4>;

// Every driver evaluates program `y` at `Width` elements `x`.
constexpr std::string_view DriverSource{R"(
#include <metal_stdlib>
using namespace metal;

using Program = float4(float4);
constant uint Width = 16;

float4 Input(uint element) { return float4(float(element), float(element) * 0.5f - 3.0f, 7.0f - float(element), 0.25f); }

kernel void ComputeDriver(uint2 id [[thread_position_in_grid]], device float4 *out [[buffer(0)]], visible_function_table<Program> programs [[buffer(1)]]) {
    out[id.y * Width + id.x] = programs[id.y](Input(id.x));
}

struct Fragment { float4 Position [[position]]; };
vertex Fragment FullScreenVertex(uint vertex_id [[vertex_id]]) {
    const float2 p = float2(float((vertex_id << 1) & 2u), float(vertex_id & 2u));
    return {float4(p * 2.0f - 1.0f, 0.0f, 1.0f)};
}
fragment float4 RenderDriver(Fragment in [[stage_in]], visible_function_table<Program> programs [[buffer(1)]]) {
    const uint2 id = uint2(in.Position.xy);
    return programs[id.y](Input(id.x));
}
)"};
Float4 Input(uint32_t element) { return {float(element), float(element) * 0.5f - 3.f, 7.f - float(element), 0.25f}; }

constexpr std::string_view ProgramHeader{R"(#include <metal_stdlib>
using namespace metal;
#ifndef ProgramTestTwice
#define ProgramTestTwice
inline float4 Twice(float4 x) { return x * 2.0f; }
#endif
)"};

struct ProgramCase {
    std::string Body;
    std::function<float(Float4, size_t)> Host; // Component `i` of the result
};

std::vector<ProgramCase> Cases() {
    std::vector<ProgramCase> cases;
    for (uint32_t k = 1; k <= 8; ++k) {
        const auto f = float(k);
        cases.push_back({std::format("return x * {}.0f + 1.0f;", k), [f](Float4 x, size_t i) { return x[i] * f + 1.f; }});
        cases.push_back({std::format("return max(x, {}.0f) * x.yzwx;", k), [f](Float4 x, size_t i) { return std::max(x[i], f) * x[(i + 1) % 4]; }});
        cases.push_back({std::format("return floor(x * {}.0f * 0.125f) - Twice(x);", k), [f](Float4 x, size_t i) { return std::floor(x[i] * f * 0.125f) - x[i] * 2.f; }});
        cases.push_back({std::format("return select(x, -x * {}.0f, x < 0.0f);", k), [f](Float4 x, size_t i) { return x[i] < 0.f ? -x[i] * f : x[i]; }});
        cases.push_back({std::format("return sqrt(abs(x) + {}.0f);", k), [f](Float4 x, size_t i) { return std::sqrt(std::abs(x[i]) + f); }});
    }
    // A repeated source shares the program of its first occurrence.
    cases.push_back(cases.front());
    return cases;
}

std::vector<uint32_t> Run(const mtl::Context &ctx, const mtl::ComputeDriver &driver, uint32_t programs) {
    const mtl::ResidentBuffer out{ctx, uint64_t(Width) * programs * sizeof(Float4)};
    auto command = ctx.Begin();
    {
        mtl::PassChain chain{command};
        const auto encoder = chain.BeginCompute("Programs");
        mtl::Bind(encoder, driver, 1u);
        command.SetAddress(out.Address(), 0u);
        encoder->dispatchThreads(MTL::Size{Width, programs, 1u}, MTL::Size{Width, 1u, 1u});
    }
    expect(ctx.Run(std::move(command)).Error.empty());
    const auto *words = reinterpret_cast<const uint32_t *>(out.Contents().data());
    return {words, words + size_t(Width) * programs * 4u};
}

std::vector<uint32_t> Run(const mtl::Context &ctx, const mtl::RenderDriver &driver, uint32_t programs) {
    const auto target = mtl::CreateTexture2D(ctx, MTL::PixelFormatRGBA32Float, {Width, programs}, MTL::TextureUsageRenderTarget, 1, MTL::StorageModeShared);
    const std::array colors{mtl::ClearColor(*target)};
    const auto pass = mtl::MakePassDescriptor(colors);
    auto command = ctx.Begin();
    {
        mtl::PassChain chain{command};
        const auto encoder = chain.BeginRender(pass.get(), "Programs");
        mtl::Bind(encoder, driver, 1u);
        encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    }
    expect(ctx.Run(std::move(command)).Error.empty());
    std::vector<uint32_t> words(size_t(Width) * programs * 4u);
    target.Handle->getBytes(words.data(), Width * sizeof(Float4), MTL::Region::Make2D(0, 0, Width, programs), 0);
    return words;
}
} // namespace

const boost::ut::suite Programs = [] {
    "a program gives the same output through every driver"_test = [] {
        const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        mtl::Context ctx;
        const TestDir dir{"MeshEditorProgramTest"};
        std::filesystem::create_directories(dir.Path);
        std::ofstream{dir.Path / "Drivers.metal"} << DriverSource;
        const auto archive = dir.Path / "Drivers.mtl4a";
        const mtl::FunctionRef compute{.Path = "Drivers.metal", .Name = "ComputeDriver", .BinaryLinking = true};
        const auto render = [](mtl::LibraryCache &cache) {
            return mtl::MakeRenderPipeline(cache, {.Path = "Drivers.metal", .Name = "FullScreenVertex"}, mtl::FunctionRef{.Path = "Drivers.metal", .Name = "RenderDriver", .BinaryLinking = true}, {{MTL::PixelFormatRGBA32Float}});
        };

        // The second cache serves both driver pipelines from the archive the first cache writes.
        mtl::LibraryCache compiled{ctx, dir.Path, archive, mtl::ArchiveMode::Write};
        mtl::ComputePipeline compiled_compute{compiled, compute};
        auto compiled_render = render(compiled);
        expect(compiled.WriteArchive());
        mtl::LibraryCache archived{ctx, dir.Path, archive};
        mtl::ComputePipeline archived_compute{archived, compute};
        auto archived_render = render(archived);
        expect(archived.ArchiveHitCount() == 2u);

        mtl::ProgramRegistry programs{ctx};
        std::vector<std::function<std::vector<uint32_t>(uint32_t)>> drivers;
        const auto add_driver = [&](auto pipeline) {
            const auto &driver = mtl::AddDriver(programs, std::move(pipeline));
            drivers.emplace_back([&ctx, &driver](uint32_t count) { return Run(ctx, driver, count); });
        };
        // The compiled drivers register before any program and the archived drivers after the first batch.
        add_driver(std::move(compiled_compute));
        add_driver(std::move(compiled_render));

        const auto cases = Cases();
        std::vector<std::string> sources;
        for (const auto &c : cases) sources.push_back(std::string{ProgramHeader} + "[[visible]] float4 ProgramFunction(float4 x) { " + c.Body + " }\n");
        std::vector<uint32_t> indices;
        for (const auto count : {cases.size() / 2u, cases.size()}) {
            const auto added = mtl::AddPrograms(programs, std::span{sources}.first(count));
            expect(std::ranges::equal(std::span{added}.first(indices.size()), indices));
            indices = added;
            if (drivers.size() == 2u) {
                add_driver(std::move(archived_compute));
                add_driver(std::move(archived_render));
            }

            std::vector<std::vector<uint32_t>> outputs;
            for (const auto &run : drivers) outputs.push_back(run(uint32_t(programs.Functions.size())));
            for (size_t c = 0; c < count; ++c) {
                const auto row = std::span{outputs[0]}.subspan(size_t(indices[c]) * Width * 4u, Width * 4u);
                for (uint32_t element = 0; element < Width; ++element) {
                    for (size_t i = 0; i < 4u; ++i) expect(Near(std::bit_cast<float>(row[element * 4u + i]), cases[c].Host(Input(element), i), 1e-5));
                }
                for (const auto &output : outputs) expect(std::ranges::equal(std::span{output}.subspan(size_t(indices[c]) * Width * 4u, Width * 4u), row));
            }
        }
        expect(programs.Functions.size() == cases.size() - 1u);
    };
};
