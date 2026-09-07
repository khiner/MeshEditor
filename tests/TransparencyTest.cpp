// Compare the production tile compositor with depth-sorted source-over, including reversed draws and overflow.
#include "TestPaths.h"
#include "metal/MetalCpp.h"
#include "metal/MslSource.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <print>
#include <random>
#include <stdexcept>
#include <vector>

struct alignas(16) Layer {
    std::array<float, 4> Color;
    float Depth;
};

int main(int argc, char **argv) {
    (void)argc;
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    auto queue = NS::TransferPtr(device->newCommandQueue());
    auto source = msl::Load(ShadersDir(argv[0]), "Transparency.metal").Text;
    source += R"(
struct TestLayer { float4 Color; float Depth; };
struct TestVaryings { float4 Position [[position]]; float4 Color; };
vertex TestVaryings TestVertex(uint id [[vertex_id]], constant TestLayer &layer [[buffer(0)]]) {
    const float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return {float4(p[id], layer.Depth, 1), layer.Color};
}
fragment TransparencyStore TestFragment(TestVaryings in [[stage_in]], TransparencyValues values [[imageblock_data]]) {
    return StoreTransparency(values, in.Color, in.Position.z);
}
)";
    NS::Error *error = nullptr;
    auto library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nullptr, &error));
    if (!library) throw std::runtime_error(error->localizedDescription()->utf8String());
    const auto pipeline = [&](const char *fragment) {
        auto descriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
        auto vertex = NS::TransferPtr(library->newFunction(NS::String::string("TestVertex", NS::UTF8StringEncoding)));
        auto frag = NS::TransferPtr(library->newFunction(NS::String::string(fragment, NS::UTF8StringEncoding)));
        descriptor->setVertexFunction(vertex.get());
        descriptor->setFragmentFunction(frag.get());
        descriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA16Float);
        auto state = NS::TransferPtr(device->newRenderPipelineState(descriptor.get(), &error));
        if (!state) throw std::runtime_error(error->localizedDescription()->utf8String());
        return state;
    };
    const auto init = pipeline("TransparencyInitFragment");
    const auto insert = pipeline("TestFragment");
    const auto resolve = pipeline("TransparencyResolveFragment");
    auto descriptor = NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
    descriptor->setWidth(1);
    descriptor->setHeight(1);
    descriptor->setPixelFormat(MTL::PixelFormatRGBA16Float);
    descriptor->setStorageMode(MTL::StorageModeShared);
    descriptor->setUsage(MTL::TextureUsageRenderTarget);
    auto target = NS::TransferPtr(device->newTexture(descriptor.get()));
    const auto render = [&](const std::vector<Layer> &layers) {
        auto pass = MTL::RenderPassDescriptor::renderPassDescriptor();
        auto color = pass->colorAttachments()->object(0);
        color->setTexture(target.get());
        color->setLoadAction(MTL::LoadActionClear);
        color->setStoreAction(MTL::StoreActionStore);
        color->setClearColor({0.125, 0.25, 0.5, 1});
        pass->setTileWidth(16);
        pass->setTileHeight(16);
        pass->setImageblockSampleLength(std::max({init->imageblockSampleLength(), insert->imageblockSampleLength(), resolve->imageblockSampleLength()}));
        auto *command = queue->commandBuffer();
        auto *encoder = command->renderCommandEncoder(pass);
        const Layer full{{0, 0, 0, 0}, 0.5f};
        const auto draw = [&](MTL::RenderPipelineState *state, const Layer &layer) {
            encoder->setRenderPipelineState(state);
            encoder->setVertexBytes(&layer, sizeof(layer), 0);
            encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
        };
        draw(init.get(), full);
        for (const auto &layer : layers) draw(insert.get(), layer);
        draw(resolve.get(), full);
        encoder->endEncoding();
        command->commit();
        command->waitUntilCompleted();
        if (command->error()) throw std::runtime_error(command->error()->localizedDescription()->utf8String());
        std::array<_Float16, 4> pixel{};
        target->getBytes(pixel.data(), sizeof(pixel), MTL::Region::Make2D(0, 0, 1, 1), 0);
        return std::array<float, 4>{pixel[0], pixel[1], pixel[2], pixel[3]};
    };
    const auto reference = [](std::vector<Layer> layers) {
        std::ranges::sort(layers, {}, &Layer::Depth);
        std::array<float, 4> color{0.125f, 0.25f, 0.5f, 1};
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            const float alpha = float(_Float16(it->Color[3]));
            for (size_t c = 0; c < 4; ++c) {
                const float value = c == 3 ? alpha : float(_Float16(it->Color[c] * it->Color[3]));
                color[c] = value + (1 - alpha) * color[c];
            }
        }
        return color;
    };
    std::mt19937 rng{0};
    float exact_error = 0, overflow_error = 0, order_error = 0;
    for (uint32_t count : {0u, 1u, 4u, 8u, 32u}) {
        std::vector<Layer> layers;
        for (uint32_t i = 0; i < count; ++i) {
            layers.push_back({{float(i % 3 == 0) * 4, float(i % 3 == 1), float(i % 3 == 2), 0.25f}, 0.2f + float(i) * 1e-5f});
        }
        const auto expected = reference(layers);
        const auto first = render(layers);
        for (uint32_t permutation = 0; permutation < 16; ++permutation) {
            std::shuffle(layers.begin(), layers.end(), rng);
            const auto actual = render(layers);
            for (size_t c = 0; c < 4; ++c) {
                auto &difference = count <= 4 ? exact_error : overflow_error;
                difference = std::max(difference, std::abs(actual[c] - expected[c]));
                order_error = std::max(order_error, std::abs(actual[c] - first[c]));
            }
        }
    }
    std::println("Transparency max error: exact layers {:.6f}; weighted overflow {:.6f}; draw permutations {:.6f}", exact_error, overflow_error, order_error);
    return exact_error <= 0.004f && order_error <= 0.004f && overflow_error <= 0.15f ? 0 : 1;
}
