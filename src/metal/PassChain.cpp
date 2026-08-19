#include "metal/PassChain.h"

namespace mtl {
PassChain::PassChain(MTL::CommandBuffer *command_buffer, PassTimer *timer)
    : CommandBuffer(command_buffer), Timer(timer) {}

PassChain::~PassChain() { ClosePass(); }

template<typename AttachmentT> void PassChain::SampleInto(AttachmentT *attachment, std::string_view name) {
    if (!Timer) return;
    const auto pass = Timer->Claim(name);
    if (!pass) return; // The sample buffer is full, so this pass goes untimed.
    attachment->setSampleBuffer(Timer->Buffer());
    attachment->setStartOfEncoderSampleIndex(*pass * 2);
    attachment->setEndOfEncoderSampleIndex(*pass * 2 + 1);
}

void PassChain::ClosePass() {
    if (Open) Open->endEncoding();
    Open = nullptr;
}

void PassChain::OpenPass(MTL::CommandEncoder *encoder, std::initializer_list<Barrier> barriers) {
    if (Encoded) {
        for (const auto &barrier : barriers) {
            if (barrier.After) encoder->barrierAfterQueueStages(barrier.After, barrier.Before);
        }
    }
    Open = encoder;
    Encoded = true;
}

MTL::RenderCommandEncoder *PassChain::BeginRender(MTL::RenderPassDescriptor *pass, std::string_view name, std::initializer_list<Barrier> barriers) {
    if (Timer) {
        if (const auto slot = Timer->Claim(name)) {
            auto *attachment = pass->sampleBufferAttachments()->object(0);
            attachment->setSampleBuffer(Timer->Buffer());
            attachment->setStartOfVertexSampleIndex(*slot * 2);
            attachment->setEndOfFragmentSampleIndex(*slot * 2 + 1);
        }
    }
    ClosePass();
    auto *encoder = CommandBuffer->renderCommandEncoder(pass);
    // Imported and generated geometry both wind front faces counter-clockwise, which Metal's
    // clockwise default would report inverted to [[front_facing]].
    encoder->setFrontFacingWinding(MTL::WindingCounterClockwise);
    OpenPass(encoder, barriers);
    return encoder;
}

MTL::ComputeCommandEncoder *PassChain::BeginCompute(std::string_view name, MTL::Stages after) {
    auto *descriptor = MTL::ComputePassDescriptor::computePassDescriptor();
    SampleInto(descriptor->sampleBufferAttachments()->object(0), name);
    ClosePass();
    auto *encoder = CommandBuffer->computeCommandEncoder(descriptor);
    OpenPass(encoder, {{after, Stage::Dispatch}});
    return encoder;
}

MTL::BlitCommandEncoder *PassChain::BeginBlit(std::string_view name, MTL::Stages after) {
    auto *descriptor = MTL::BlitPassDescriptor::blitPassDescriptor();
    SampleInto(descriptor->sampleBufferAttachments()->object(0), name);
    ClosePass();
    auto *encoder = CommandBuffer->blitCommandEncoder(descriptor);
    OpenPass(encoder, {{after, Stage::Blit}});
    return encoder;
}
} // namespace mtl
