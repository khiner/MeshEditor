#include "metal/Dispatch.h"
#include "metal/MetalCpp.h"
#include "metal/PassChain.h"
#include <boost/ut.hpp>

using namespace boost::ut;

const boost::ut::suite MetalBuffer = [] {
    "a compute chain owns its recording across calls and intervening submissions"_test = [] {
        const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        mtl::Context ctx;
        mtl::BindlessSet slots{ctx};
        mtl::BufferContext buffers{ctx, slots};
        mtl::Buffer output{buffers, BytesOf(uint32_t{10}), SlotType::Buffer, mtl::BufferLifetime::Workspace};
        NS::Error *error = nullptr;
        const auto library = NS::TransferPtr(ctx.Device->newLibrary(mtl::Str(R"(
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
            chain.Encode([&](mtl::ComputeEncoder encoder) {
                encoder->setComputePipelineState(pipeline.get());
                encoder.Cmd->SetBuffer(*output, 0u, 0u);
                encoder.Cmd->SetBytes(amount, 1u);
                encoder->dispatchThreads(MTL::Size{1u, 1u, 1u}, MTL::Size{1u, 1u, 1u});
                mtl::ComputeBarrier(encoder);
            });
        };
        add(3u);
        // This committed write executes before the chain, whose command waits for every earlier commit.
        {
            auto write = ctx.Begin();
            {
                mtl::PassChain chain{write};
                chain.BeginBlit()->fillBuffer(*output, NS::Range::Make(0u, sizeof(uint32_t)), 1u);
            }
            (void)ctx.Commit(std::move(write));
        }
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
        const mtl::ResidentBuffer readback{ctx, 4u};
        std::optional<mtl::Buffer> clone;
        constexpr uint32_t Original = 0x12345678u;
        {
            mtl::Buffer source{buffers, 0u, SlotType::Buffer};
            source.SetUsedSize(mtl::HistoryPageBytes);
            source.Update(BytesOf(Original));
            const std::array pages{0u};
            const std::array footprint{mtl::BufferFootprint{&source, pages}};
            auto copies = mtl::CloneFootprints(buffers, footprint);
            clone.emplace(std::move(copies.front()));
            source.Update(BytesOf(uint32_t{0xabcdef12u}));
            expect(source.GetSpan<uint32_t>({0u, 1u})[0] == 0xabcdef12u);
        }
        buffers.ReclaimRetiredBuffers(true);
        const auto source_slot = slots.Allocate(SlotType::Buffer);

        const auto retired_slot = clone->Slot;
        auto reader = ctx.Begin();
        reader.WaitEvent(gate.get(), 1u);
        const auto copy = reader.BeginCompute();
        copy->copyFromBuffer(**clone, 0u, *readback, 0u, 4u);
        copy->endEncoding();
        const auto serial = ctx.Commit(std::move(reader));
        clone.reset();
        buffers.ReclaimRetiredBuffers();
        const auto next_slot = slots.Allocate(SlotType::Buffer);
        expect(next_slot != retired_slot);

        gate->setSignaledValue(1u);
        expect(ctx.Wait(serial).Error.empty());
        expect(*reinterpret_cast<const uint32_t *>(readback.Contents().data()) == Original);
        buffers.ReclaimRetiredBuffers(true);
        slots.Release({SlotType::Buffer, source_slot});
        slots.Release({SlotType::Buffer, next_slot});
    };
};
