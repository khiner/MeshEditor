#include "RunSuites.h"
#include "metal/Buffer.h"
#include "metal/MetalCpp.h"

#include <array>
#include <optional>

using namespace boost::ut;

int main() {
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
