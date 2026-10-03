#include "metal/Dispatch.h"

#include "metal/MetalCpp.h"
#include "metal/Shader.h"

#include <atomic>

namespace mtl {
namespace {
// Numbers the submits of every chain, so no two submits share a serial.
std::atomic<uint64_t> NextSubmissionSerial{};
} // namespace

ComputeChain::ComputeChain(BufferContext &buffers, uint32_t scratch_words)
    : Buffers(buffers), Scratch(buffers, SlotType::Buffer, BufferLifetime::Workspace),
      SubmissionSerial(NextSubmissionSerial.fetch_add(1u, std::memory_order_relaxed)) {
    Scratch.ReserveAdditional(scratch_words);
    Scratch.Allocate(1u);
    Scratch.GetMutable({0u, 1u})[0] = 0u;
}

ComputeChain::~ComputeChain() {
    const AutoreleaseScope pool;
    // Recorded passes end with a submit, and only exception unwinding abandons them.
    if (Recording && !std::uncaught_exceptions()) {
        std::fputs("A compute chain was destroyed with unsubmitted passes.\n", stderr);
        std::abort();
    }
    if (const auto encoder = std::exchange(Encoding, {})) encoder->endEncoding();
    Recording.reset();
}

MTL::ComputeCommandEncoder *ComputeChain::Encoder() {
    const auto &ctx = Buffers.Ctx;
    if (!Recording) {
        Recording = NS::RetainPtr(ctx.Queue->commandBuffer());
        ctx.OrderAfterGpuWork(Recording.get());
        RecordingSignals = ctx.ExecutionSignals();
    } else if (ctx.ExecutionSignals() != RecordingSignals) {
        // Work committed since the recording began can write what later passes read, so later passes wait for it.
        if (const auto encoder = std::exchange(Encoding, {})) encoder->endEncoding();
        ctx.OrderAfterGpuWork(Recording.get());
        RecordingSignals = ctx.ExecutionSignals();
    }
    if (!Encoding) {
        Encoding = NS::RetainPtr(Recording->computeCommandEncoder());
        DeclaredResources = ~0ull;
    }
    // Buffers created after the last declaration join it before the next pass reads them.
    if (DeclaredResources != Buffers.Slots.ResourceRevision()) {
        Buffers.Slots.UseResources(Encoding.get());
        DeclaredResources = Buffers.Slots.ResourceRevision();
    }
    Encoding->setBuffer(Buffers.Slots.Table(), 0, BufferIndex_Bindless);
    return Encoding.get();
}

void ComputeChain::Dispatch(const ComputePipeline &pipeline, const void *pc, uint32_t bytes, uint32_t count, uint32_t depth, uint32_t width, bool threads) {
    const AutoreleaseScope pool;
    if (!count || !depth) return;
    auto *encoder = Encoder();
    encoder->setComputePipelineState(pipeline.State());
    encoder->setBytes(pc, bytes, BufferIndex_PushConstants);
    if (threads) encoder->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(width, 1, 1));
    else encoder->dispatchThreadgroups(MTL::Size(count, 1, depth), MTL::Size(width, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
}

void ComputeChain::DispatchIndirect(const ComputePipeline &pipeline, const void *pc, uint32_t bytes, const Buffer &arguments, uint64_t offset, uint32_t width) {
    const AutoreleaseScope pool;
    auto *encoder = Encoder();
    encoder->setComputePipelineState(pipeline.State());
    encoder->setBytes(pc, bytes, BufferIndex_PushConstants);
    encoder->dispatchThreadgroups(*arguments, offset, MTL::Size(width, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
}

void ComputeChain::Retain(Buffer &&buffer) { Retained.push_back(std::move(buffer)); }
void ComputeChain::AfterSubmit(std::function<void()> fn) { Completions.push_back(std::move(fn)); }

void ComputeChain::Submit() {
    const AutoreleaseScope pool;
    // A failed submit drops its completions.
    const auto completions = std::exchange(Completions, {});
    if (Recording) {
        // Buffers retired while recording fence ahead of these passes, so the reclaim after their wait releases them.
        Buffers.ReclaimRetiredBuffers();
        if (const auto encoder = std::exchange(Encoding, {})) encoder->endEncoding();
        // Encoding can allocate buffers, so residency commits after it.
        Buffers.Ctx.CommitResidency();
        const auto command = std::exchange(Recording, {});
        command->commit();
        command->waitUntilCompleted();
        if (command->status() == MTL::CommandBufferStatusError) throw std::runtime_error("GPU compute chain failed.");
        Retained.clear();
        Buffers.ReclaimRetiredBuffers();
    }
    SubmissionSerial = NextSubmissionSerial.fetch_add(1u, std::memory_order_relaxed);
    if (Scratch.Get({0u, 1u})[0]) throw std::invalid_argument("GPU work found invalid canonical ownership or topology.");
    for (const auto &completion : completions) completion();
}
} // namespace mtl
