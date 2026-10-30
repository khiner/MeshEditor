#pragma once

#include "metal/Command.h"
#include "metal/PassTimer.h"

namespace mtl {
// `Before` waits for matching `After` stages already encoded on the queue.
struct Barrier {
    MTL::Stages After, Before;
};

// Callers declare cross-pass dependencies, since Metal 4 tracks no resource hazards.
// The first pass of a command waits for every earlier command on the queue, and each later pass for the stages its barriers declare.
// Opening a pass closes the previous encoder, and destruction closes the final encoder.
// Borrows the command and encoders inside the containing operation's AutoreleaseScope.
struct PassChain {
    PassChain(Command &, PassTimer * = nullptr);
    ~PassChain();

    PassChain(const PassChain &) = delete;
    PassChain &operator=(const PassChain &) = delete;

    RenderEncoder BeginRender(const MTL4::RenderPassDescriptor *, std::string_view name, std::initializer_list<Barrier> = {});
    // Dispatches encode without ordering between them, so dependent dispatches need a ComputeBarrier.
    ComputeEncoder BeginCompute(std::string_view name, MTL::Stages after = {});
    // A compute encoder for copies and fills, which run in the blit stage.
    // The pass is untimed, since a compute encoder's timestamps wait for dispatches only.
    ComputeEncoder BeginBlit(MTL::Stages after = {});

    Command &Cmd;

private:
    void ClosePass();
    void OpenPass(MTL4::CommandEncoder *, MTL::Stages own, std::initializer_list<Barrier>);

    PassTimer *Timer;
    MTL4::CommandEncoder *Open{nullptr};
    MTL4::RenderCommandEncoder *OpenRender{nullptr};
    std::optional<uint32_t> OpenSlot;
    bool Encoded{false};
};
} // namespace mtl
