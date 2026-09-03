#include "metal/RenderTarget.h"

namespace mtl {
MTL::RenderPassDescriptor *MakePassDescriptor(std::span<const ColorAttachment> colors, DepthAttachment depth) {
    auto *descriptor = MTL::RenderPassDescriptor::renderPassDescriptor();
    for (size_t i = 0; i < colors.size(); ++i) {
        const auto &color = colors[i];
        if (!color.Texture) continue;
        auto *attachment = descriptor->colorAttachments()->object(i);
        attachment->setTexture(color.Texture);
        attachment->setLevel(color.Level);
        attachment->setLoadAction(color.Load);
        attachment->setStoreAction(color.Store);
        attachment->setClearColor(color.Clear);
    }
    if (depth.Texture) {
        auto *attachment = descriptor->depthAttachment();
        attachment->setTexture(depth.Texture);
        attachment->setLoadAction(depth.Load);
        attachment->setStoreAction(depth.Store);
        attachment->setClearDepth(depth.Clear);
    }
    return descriptor;
}
}
