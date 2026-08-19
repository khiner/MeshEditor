#pragma once

#include "metal/PassTimer.h"

#include <initializer_list>

namespace mtl {
// `Before` waits for matching `After` stages already encoded on the queue.
struct Barrier {
    MTL::Stages After, Before;
};

// Bindless resources are invisible to Metal's automatic hazard tracking, so callers declare their
// cross-pass stage dependencies. Directly bound resources remain tracked by Metal. Opening a pass
// closes the previous encoder; destruction closes the last one.
struct PassChain {
    PassChain(MTL::CommandBuffer *, PassTimer * = nullptr);
    ~PassChain();

    PassChain(const PassChain &) = delete;
    PassChain &operator=(const PassChain &) = delete;

    MTL::RenderCommandEncoder *BeginRender(MTL::RenderPassDescriptor *, std::string_view name, std::initializer_list<Barrier> = {});
    MTL::ComputeCommandEncoder *BeginCompute(std::string_view name, MTL::Stages after = {});
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
} // namespace mtl
