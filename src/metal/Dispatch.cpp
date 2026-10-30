#include "metal/Dispatch.h"
#include "Profile.h"

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
    EndEncoding();
    Recording.reset();
    // A chain that never committed leaves its scratch unread, so the scratch releases without a fence.
    if (!Committed) {
        const auto first = Buffers.Retired.size();
        Scratch.Buffer = Buffer{Buffers, 0u};
        Buffers.ReleaseRetiredBuffers(first);
    }
}

void ComputeChain::Barrier() {
    if (std::exchange(DeferredBarrier, false) && Encoding) ComputeBarrier({&*Recording, Encoding.get()});
}

void ComputeChain::EndEncoding() {
    Barrier();
    if (const auto encoder = std::exchange(Encoding, {})) encoder->endEncoding();
}

ComputeEncoder ComputeChain::Encoder() {
    if (!Recording) Recording.emplace(Buffers.Ctx.Begin());
    if (!Encoding) {
        const auto encoder = Recording->BeginCompute();
        // The passes read what every earlier command wrote.
        encoder->barrierAfterQueueStages(MTL::StageAll, MTL::StageDispatch | MTL::StageBlit, MTL4::VisibilityOptionDevice);
        Encoding = NS::RetainPtr(encoder.Pass);
    }
    Recording->SetBuffer(Buffers.Slots.Table(), 0, BufferIndex_Bindless);
    return {&*Recording, Encoding.get()};
}

// Binds the pipeline and push constants, runs `dispatch` on the encoder, and orders the passes after it.
void ComputeChain::Record(const ComputePipeline &pipeline, const void *pc, uint32_t bytes, auto &&dispatch) {
    const auto encoder = Encoder();
    pipeline.Bind(encoder);
    encoder.Cmd->SetBytes(std::span{static_cast<const std::byte *>(pc), bytes}, BufferIndex_PushConstants);
    dispatch(encoder);
    if (Deferring) DeferredBarrier = true;
    else ComputeBarrier(encoder);
}

void ComputeChain::Dispatch(const ComputePipeline &pipeline, const void *pc, uint32_t bytes, uint32_t count, uint32_t depth, uint32_t width, bool threads) {
    const AutoreleaseScope pool;
    if (!count || !depth) return;
    if (profile::Enabled) {
        const auto groups = (threads ? (uint64_t(count) + width - 1u) / width : count) * depth;
        profile::RecordCounter("ComputeThreadgroups", groups);
    }
    Record(pipeline, pc, bytes, [&](ComputeEncoder encoder) {
        if (threads) encoder->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(width, 1, 1));
        else encoder->dispatchThreadgroups(MTL::Size(count, 1, depth), MTL::Size(width, 1, 1));
    });
}

void ComputeChain::DispatchIndirect(const ComputePipeline &pipeline, const void *pc, uint32_t bytes, const Buffer &arguments, uint64_t offset, uint32_t width) {
    const AutoreleaseScope pool;
    profile::RecordCounter("ComputeIndirectDispatches", 1u);
    Record(pipeline, pc, bytes, [&](ComputeEncoder encoder) { encoder->dispatchThreadgroups((*arguments)->gpuAddress() + offset, MTL::Size(width, 1, 1)); });
}

void ComputeChain::Retain(Buffer &&buffer) { Retained.push_back(std::move(buffer)); }
void ComputeChain::AfterSubmit(std::function<void()> fn) { Completions.push_back(std::move(fn)); }

void ComputeChain::Submit() {
    const AutoreleaseScope pool;
    // A failed submit drops its completions.
    const auto completions = std::exchange(Completions, {});
    if (Recording) {
        const profile::CpuScope scope{"ComputeChainSubmit"};
        EndEncoding();
        const auto serial = Buffers.Ctx.Commit(std::move(*Recording));
        Recording.reset();
        // The command fences the buffers retired while recording, since it runs after every command committed before it.
        Buffers.FenceRetiredBuffers(serial);
        Committed = true;
        const auto completion = Buffers.Ctx.Wait(serial);
        if (!completion.Error.empty()) throw std::runtime_error("GPU compute chain failed: " + completion.Error);
        if (profile::Enabled) profile::RecordCounter("ComputeGpuMicroseconds", (completion.GpuEndTime - completion.GpuStartTime) * 1e6);
        // The retained buffers and the scratch workspaces growth replaced had their last readers in the command.
        Retained.clear();
        Scratch.Buffer.RetirePreviousWorkspaces();
        Buffers.ReleaseRetiredBuffers();
        Buffers.ReclaimRetiredBuffers();
    }
    SubmissionSerial = NextSubmissionSerial.fetch_add(1u, std::memory_order_relaxed);
    if (Scratch.Get({0u, 1u})[0]) throw std::invalid_argument("GPU work found invalid canonical ownership or topology.");
    for (const auto &completion : completions) completion();
}
} // namespace mtl
