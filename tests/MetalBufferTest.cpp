#include "RunSuites.h"
#include "metal/Buffer.h"
#include "metal/Dispatch.h"
#include "metal/MetalCpp.h"

#include <array>
#include <optional>

using namespace boost::ut;

int main() {
    "a compute chain owns its recording across calls and intervening submissions"_test = [] {
        const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        mtl::Context ctx;
        mtl::BindlessSet slots{ctx};
        mtl::BufferContext buffers{ctx, slots};
        mtl::Buffer output{buffers, as_bytes(uint32_t{10}), SlotType::Buffer, mtl::BufferLifetime::Workspace};
        NS::Error *error = nullptr;
        const auto library = NS::TransferPtr(ctx.Device->newLibrary(mtl::Str(R"(
            #include <metal_stdlib>
            kernel void add(device uint &value [[buffer(0)]], constant uint &amount [[buffer(1)]]) {
                value += amount;
            }
        )")
                                                                        .get(),
                                                                    nullptr, &error));
        expect(bool(library));
        if (!library) return;
        const auto function = NS::TransferPtr(library->newFunction(mtl::Str("add").get()));
        const auto pipeline = NS::TransferPtr(ctx.Device->newComputePipelineState(function.get(), &error));
        expect(bool(pipeline));
        if (!pipeline) return;

        mtl::ComputeChain chain{buffers};
        const auto add = [&](uint32_t amount) {
            chain.Encode([&](MTL::ComputeCommandEncoder *encoder) {
                encoder->setComputePipelineState(pipeline.get());
                encoder->setBuffer(*output, 0u, 0u);
                encoder->setBytes(&amount, sizeof(amount), 1u);
                encoder->dispatchThreads(MTL::Size{1u, 1u, 1u}, MTL::Size{1u, 1u, 1u});
                encoder->memoryBarrier(MTL::BarrierScopeBuffers);
            });
        };
        add(3u);
        // This committed write executes before the chain and forces its next call to open a new encoder.
        auto *write = ctx.Queue->commandBuffer();
        ctx.OrderAfterGpuWork(write);
        auto *blit = write->blitCommandEncoder();
        blit->fillBuffer(*output, NS::Range::Make(0u, sizeof(uint32_t)), 1u);
        blit->endEncoding();
        write->commit();
        add(5u);
        chain.Submit();
        expect(output.GetSpan<uint32_t>({0u, 1u})[0] == 0x01010109u);

        add(7u);
        chain.Submit();
        expect(output.GetSpan<uint32_t>({0u, 1u})[0] == 0x01010110u);
    };
    "a clone survives source changes and pending GPU reads"_test = [] {
        const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        mtl::Context ctx;
        mtl::BindlessSet slots{ctx};
        mtl::BufferContext buffers{ctx, slots};
        const auto gate = NS::TransferPtr(ctx.Device->newSharedEvent());
        const auto readback = NS::TransferPtr(ctx.Device->newBuffer(4u, MTL::ResourceStorageModeShared));
        std::optional<mtl::Buffer> clone;
        constexpr uint32_t Original = 0x12345678u;
        {
            mtl::Buffer source{buffers, 0u, SlotType::Buffer};
            source.SetUsedSize(mtl::HistoryPageBytes);
            source.Update(as_bytes(Original));
            const std::array pages{0u};
            const std::array footprint{mtl::BufferFootprint{&source, pages}};
            auto copies = mtl::CloneFootprints(buffers, footprint);
            clone.emplace(std::move(copies.front()));
            source.Update(as_bytes(uint32_t{0xabcdef12u}));
            expect(source.GetSpan<uint32_t>({0u, 1u})[0] == 0xabcdef12u);
        }
        buffers.ReclaimRetiredBuffers(true);
        const auto source_slot = slots.Allocate(SlotType::Buffer);

        const auto retired_slot = clone->Slot;
        auto *reader = ctx.Queue->commandBuffer();
        reader->encodeWait(gate.get(), 1u);
        auto *encoder = reader->blitCommandEncoder();
        encoder->copyFromBuffer(**clone, 0u, readback.get(), 0u, 4u);
        encoder->endEncoding();
        ctx.CommitResidency();
        reader->commit();
        clone.reset();
        buffers.ReclaimRetiredBuffers();
        const auto next_slot = slots.Allocate(SlotType::Buffer);
        expect(next_slot != retired_slot);

        gate->setSignaledValue(1u);
        reader->waitUntilCompleted();
        expect(reader->status() == MTL::CommandBufferStatusCompleted);
        expect(*static_cast<const uint32_t *>(readback->contents()) == Original);
        buffers.ReclaimRetiredBuffers(true);
        slots.Release({SlotType::Buffer, source_slot});
        slots.Release({SlotType::Buffer, next_slot});
    };
    return RunSuites();
}
