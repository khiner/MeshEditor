#include "metal/PassChain.h"

#include "metal/MetalCpp.h"

namespace mtl {
namespace {
constexpr auto RenderStages = MTL::StageVertex | MTL::StageMesh | MTL::StageObject | MTL::StageFragment;
constexpr auto ComputeStages = MTL::StageDispatch | MTL::StageBlit;

// The start timestamp precedes the earliest stage the barriers hold back, since earlier stages run ahead of the barriers.
MTL::RenderStages StartStage(std::initializer_list<Barrier> barriers) {
    MTL::Stages before{};
    for (const auto &barrier : barriers) before |= barrier.Before;
    if (!before || (before & MTL::StageVertex)) return MTL::RenderStageVertex;
    if (before & MTL::StageObject) return MTL::RenderStageObject;
    if (before & MTL::StageMesh) return MTL::RenderStageMesh;
    return MTL::RenderStageFragment;
}
} // namespace

PassChain::PassChain(Command &command, PassTimer *timer) : Cmd(command), Timer(timer) {}

PassChain::~PassChain() { ClosePass(); }

void PassChain::ClosePass() {
    if (!Open) return;
    if (OpenSlot) {
        if (OpenRender) OpenRender->writeTimestamp(MTL4::TimestampGranularityPrecise, MTL::RenderStageFragment, Timer->Heap(), *OpenSlot * 2 + 1);
        else static_cast<MTL4::ComputeCommandEncoder *>(Open)->writeTimestamp(MTL4::TimestampGranularityPrecise, Timer->Heap(), *OpenSlot * 2 + 1);
    }
    Open->endEncoding();
    Open = nullptr;
    OpenRender = nullptr;
    OpenSlot.reset();
}

void PassChain::OpenPass(MTL4::CommandEncoder *encoder, MTL::Stages own, std::initializer_list<Barrier> barriers) {
    if (Encoded) {
        for (const auto &barrier : barriers) {
            if (barrier.After) encoder->barrierAfterQueueStages(barrier.After, barrier.Before, MTL4::VisibilityOptionDevice);
        }
    } else {
        encoder->barrierAfterQueueStages(MTL::StageAll, own, MTL4::VisibilityOptionDevice);
    }
    Open = encoder;
    Encoded = true;
}

RenderEncoder PassChain::BeginRender(const MTL4::RenderPassDescriptor *pass, std::string_view name, std::initializer_list<Barrier> barriers) {
    ClosePass();
    auto encoder = Cmd.BeginRender(pass);
    // Use counter-clockwise front faces for imported and generated geometry.
    encoder->setFrontFacingWinding(MTL::WindingCounterClockwise);
    const auto start = Encoded ? StartStage(barriers) : MTL::RenderStageVertex;
    OpenPass(encoder.Pass, RenderStages, barriers);
    OpenRender = encoder.Pass;
    if (Timer) {
        if (const auto slot = Timer->Claim(name)) {
            OpenSlot = slot;
            encoder->writeTimestamp(MTL4::TimestampGranularityPrecise, start, Timer->Heap(), *slot * 2);
        }
    }
    return encoder;
}

ComputeEncoder PassChain::BeginCompute(std::string_view name, MTL::Stages after) {
    ClosePass();
    auto encoder = Cmd.BeginCompute();
    OpenPass(encoder.Pass, ComputeStages, {{after, MTL::StageDispatch}});
    if (Timer) {
        if (const auto slot = Timer->Claim(name)) {
            OpenSlot = slot;
            encoder->writeTimestamp(MTL4::TimestampGranularityPrecise, Timer->Heap(), *slot * 2);
        }
    }
    return encoder;
}

ComputeEncoder PassChain::BeginBlit(MTL::Stages after) {
    ClosePass();
    auto encoder = Cmd.BeginCompute();
    OpenPass(encoder.Pass, ComputeStages, {{after, MTL::StageBlit}});
    return encoder;
}
} // namespace mtl
