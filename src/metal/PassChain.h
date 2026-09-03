#pragma once

#include "metal/PassTimer.h"

#include <initializer_list>

namespace mtl {
// `Before` waits for matching `After` stages already encoded on the queue.
struct Barrier {
    MTL::Stages After, Before;
};

// Callers must declare cross-pass dependencies for bindless resources because Metal cannot track them automatically.
// Opening a pass closes the previous encoder, and destruction closes the final encoder.
struct PassChain {
    PassChain(MTL::CommandBuffer *, PassTimer * = nullptr);
    ~PassChain();

    PassChain(const PassChain &) = delete;
    PassChain &operator=(const PassChain &) = delete;

    MTL::RenderCommandEncoder *BeginRender(MTL::RenderPassDescriptor *, std::string_view name, std::initializer_list<Barrier> = {});
    // Use concurrent dispatch only when dispatches have no data dependencies.
    MTL::ComputeCommandEncoder *BeginCompute(std::string_view name, MTL::Stages after = {}, MTL::DispatchType dispatch = MTL::DispatchTypeSerial);
    MTL::BlitCommandEncoder *BeginBlit(std::string_view name, MTL::Stages after = {});

private:
    template<typename AttachmentT> void SampleInto(AttachmentT *attachment, std::string_view name);

    void ClosePass();
    void OpenPass(MTL::CommandEncoder *, std::initializer_list<Barrier>);

    MTL::CommandBuffer *CommandBuffer;
    PassTimer *Timer;
    MTL::CommandEncoder *Open{nullptr};
    bool Encoded{false};
};
}
