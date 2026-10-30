#pragma once

#include "metal/AutoreleaseScope.h"
#include "metal/BufferArena.h"
#include "metal/Command.h"

#include <functional>
#include <optional>

namespace mtl {
struct ComputePipeline;

// Compute passes recorded in order, each after the buffer writes of the passes before it.
// Passes share one command until the submit, which runs after every command committed before it.
// Submit is the only commit and the only wait.
// It commits the recorded passes, waits for them, and checks the error word.
// Scratch word zero is that error word, which a pass sets when its input violates canonical ownership or topology.
struct ComputeChain {
    // The scratch starts with room for `scratch_words`, so a transaction that knows its size does not grow it mid-recording.
    explicit ComputeChain(BufferContext &, uint32_t scratch_words = 1u);
    // Destroying a chain with unsubmitted passes aborts, except during exception unwinding, which drops them.
    ~ComputeChain();
    ComputeChain(const ComputeChain &) = delete;
    ComputeChain &operator=(const ComputeChain &) = delete;

    // Binds the pipeline, the bindless table and the push constants, then dispatches `groups` threadgroups of `width` threads.
    void Groups(const ComputePipeline &pipeline, const auto &pc, uint32_t groups, uint32_t width = 256u, uint32_t depth = 1u) {
        Dispatch(pipeline, &pc, sizeof(pc), groups, depth, width, false);
    }
    // Dispatches `threads` threads in threadgroups of `width`.
    void Threads(const ComputePipeline &pipeline, const auto &pc, uint32_t threads, uint32_t width = 256u) {
        Dispatch(pipeline, &pc, sizeof(pc), threads, 1u, width, true);
    }
    // Dispatches the threadgroup count in the three words at `offset` bytes into `arguments`.
    // The arguments bind directly, so their buffer keeps its storage until the chain commits.
    void Indirect(const ComputePipeline &pipeline, const auto &pc, const Buffer &arguments, uint64_t offset, uint32_t width = 256u) {
        DispatchIndirect(pipeline, &pc, sizeof(pc), arguments, offset, width);
    }
    // The recording encoder with the bindless table bound, for passes that bind their own arguments.
    // Such a pass ends with its own ComputeBarrier, and the chain's later passes rebind the table.
    // The callback is synchronous: do not escape, end, or submit its encoder.
    template<typename Fn> void Encode(Fn &&fn) {
        const AutoreleaseScope pool;
        std::forward<Fn>(fn)(Encoder());
    }
    // Records the passes of `fn` without barriers between them, followed by one barrier.
    // Require independent reads and writes across concurrent passes.
    template<typename Fn> void Concurrent(Fn &&fn) {
        const bool outer = std::exchange(Deferring, true);
        std::forward<Fn>(fn)();
        Deferring = outer;
        if (!outer) Barrier();
    }
    // Commits the recorded passes, waits for them, reclaims retired buffers, and throws when a pass failed or set the error word.
    // Without recorded passes it only runs the completions.
    void Submit();
    // Identifies the submit that completes the passes recorded now, unique across every chain.
    // A recorder that uploads inputs on the CPU compares it to detect a second recording before the passes of the first complete.
    uint64_t Submission() const { return SubmissionSerial; }
    // Keeps a buffer the recorded passes read or write until the next submit has waited for them.
    void Retain(Buffer &&);
    // Runs `fn` on the host once the next submit completes the passes recorded before it without error.
    void AfterSubmit(std::function<void()> fn);

    // Copies packed parameters into word-aligned scratch, returning their GPU address.
    SlotOffset Upload(std::span<const std::byte> bytes) {
        const auto range = Scratch.Allocate(uint32_t((bytes.size() + 3u) / 4u));
        Scratch.Buffer.Update(bytes, uint64_t(range.Offset) * 4u);
        return {Scratch.Buffer.Slot, range.Offset};
    }

    BufferContext &Buffers;
    BufferArena<uint32_t> Scratch;

private:
    ComputeEncoder Encoder();
    // Records the barrier a concurrent group deferred.
    void Barrier();
    // Ends the compute encoder after the barrier a concurrent group deferred.
    void EndEncoding();
    void Record(const ComputePipeline &, const void *pc, uint32_t bytes, auto &&dispatch);
    void Dispatch(const ComputePipeline &, const void *pc, uint32_t bytes, uint32_t count, uint32_t depth, uint32_t width, bool threads);
    void DispatchIndirect(const ComputePipeline &, const void *pc, uint32_t bytes, const Buffer &arguments, uint64_t offset, uint32_t width);

    std::optional<Command> Recording;
    NS::SharedPtr<MTL4::ComputeCommandEncoder> Encoding;
    uint64_t SubmissionSerial;
    // A submit has committed the chain's passes, which read its scratch.
    bool Committed{false};
    // A concurrent group is recording, and a barrier its passes deferred is pending.
    bool Deferring{false}, DeferredBarrier{false};
    std::vector<Buffer> Retained;
    std::vector<std::function<void()>> Completions;
};
} // namespace mtl
