#include "metal/Dispatch.h"
#include "Profile.h"

#include "metal/MetalCpp.h"
#include "metal/Program.h"

#include <algorithm>
#include <atomic>
#include <exception>

namespace mtl {
namespace {
std::atomic<uint64_t> NextSubmissionSerial{};
} // namespace

ComputeChain::ComputeChain(BufferContext &buffers, uint64_t scratch_words)
    : Buffers(buffers), Scratch(buffers, SlotType::Buffer, BufferLifetime::Workspace),
      SubmissionSerial(NextSubmissionSerial.fetch_add(1u, std::memory_order_relaxed)) {
    Scratch.ReserveAdditional(scratch_words);
    Scratch.Allocate(1u);
    Scratch.GetMutable({0u, 1u})[0] = 0u;
}

ComputeChain::~ComputeChain() {
    const AutoreleaseScope pool;
    if ((Recording || !LaneSteps.empty()) && !std::uncaught_exceptions()) {
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

void ComputeChain::Record(const Pass &pass) {
    if (OpenLane) {
        LaneSteps[*OpenLane].push_back({.Record = [this, pass, constants = std::vector<std::byte>{pass.Constants.begin(), pass.Constants.end()}] {
            auto queued = pass;
            queued.Constants = constants;
            Record(queued);
        }});
        return;
    }
    RecordLanes();
    const AutoreleaseScope pool;
    if (!pass.Arguments && (!pass.Count || !pass.Depth)) return;
    if (profile::Enabled) {
        if (pass.Arguments) profile::RecordCounter("ComputeIndirectDispatches", 1u);
        else profile::RecordCounter("ComputeThreadgroups", (pass.Threads ? (uint64_t(pass.Count) + pass.Width - 1u) / pass.Width : pass.Count) * pass.Depth);
    }
    const auto encoder = Encoder();
    if (pass.Driver) Bind(encoder, *pass.Driver, BufferIndex_Programs);
    else pass.Pipeline->Bind(encoder);
    if (pass.Jobs) encoder.Cmd->SetBuffer(*Scratch.Buffer, uint64_t(*pass.Jobs) * sizeof(uint32_t), BufferIndex_Jobs);
    encoder.Cmd->SetBytes(pass.Constants, BufferIndex_PushConstants);
    if (pass.Arguments) encoder->dispatchThreadgroups((**pass.Arguments)->gpuAddress() + pass.ArgumentsOffset, MTL::Size(pass.Width, 1, 1));
    else if (pass.Threads) encoder->dispatchThreads(MTL::Size(pass.Count, 1, 1), MTL::Size(pass.Width, 1, 1));
    else encoder->dispatchThreadgroups(MTL::Size(pass.Count, 1, pass.Depth), MTL::Size(pass.Width, 1, 1));
    if (Deferring) DeferredBarrier = true;
    else ComputeBarrier(encoder);
}

void ComputeChain::RecordLanes() {
    if (LaneSteps.empty()) return;
    auto lanes = std::exchange(LaneSteps, {});
    KeyedLanes.clear();
    Gathered = 0u;
    size_t steps = 0u;
    for (const auto &lane : lanes) steps = std::max(steps, lane.size());
    for (size_t k = 0u; k < steps; ++k) {
        Concurrent([&] {
            for (auto &lane : lanes) {
                if (k < lane.size() && lane[k].Record) lane[k].Record();
            }
        });
        // Each kind's gathered values record once, in the order the lanes first gathered it.
        std::vector<const void *> kinds;
        for (const auto &lane : lanes) {
            if (k < lane.size() && lane[k].Kind && !std::ranges::contains(kinds, lane[k].Kind)) kinds.push_back(lane[k].Kind);
        }
        for (const auto *kind : kinds) {
            std::vector<std::shared_ptr<void>> values;
            const LaneStep *gathered{};
            for (auto &lane : lanes) {
                if (k >= lane.size() || lane[k].Kind != kind) continue;
                values.push_back(std::move(lane[k].Values));
                gathered = &lane[k];
            }
            gathered->Gathered(values);
        }
    }
}

void ComputeChain::ReleaseScratch() {
    // Fresh scratch reads zero, as a new or recycled workspace does.
    std::ranges::fill(Scratch.Buffer.Contents().first(Scratch.Buffer.UsedSize), std::byte{});
    Scratch.Reset();
    Scratch.Allocate(1u);
}

void ComputeChain::Retain(Buffer &&buffer) { Retained.push_back(std::move(buffer)); }
void ComputeChain::AfterSubmit(std::function<void()> fn) { Completions.push_back(std::move(fn)); }
void ComputeChain::Drain() {
    do Submit();
    while (!Completions.empty() || Recording || !LaneSteps.empty());
}

void ComputeChain::Submit() {
    RecordLanes();
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
        if (profile::Enabled) {
            profile::RecordCounter("ComputeGpuMicroseconds", (completion.GpuEndTime - completion.GpuStartTime) * 1e6);
            profile::RecordCounter("ComputeScratchBytes", double(Scratch.Buffer.UsedSize));
        }
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
