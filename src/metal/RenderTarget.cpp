#include "metal/RenderTarget.h"
#include "metal/AutoreleaseScope.h"

namespace mtl {
NS::SharedPtr<MTL::RenderPassDescriptor> MakePassDescriptor(std::span<const ColorAttachment> colors, DepthAttachment depth) {
    const AutoreleaseScope pool;
    auto descriptor = NS::TransferPtr(MTL::RenderPassDescriptor::alloc()->init());
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
} // namespace mtl
