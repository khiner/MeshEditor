#include "metal/Mipmaps.h"
#include "metal/AutoreleaseScope.h"

#include "metal/MetalCpp.h"

namespace mtl {
MipChain CreateMipChain(const Texture &texture) {
    const AutoreleaseScope pool;
    MipChain chain;
    chain.Levels.reserve(texture.MipLevels);
    const auto layers = uint32_t(texture.Handle->arrayLength()) * (texture.Handle->textureType() == MTL::TextureTypeCube ? 6u : 1u);
    for (uint32_t mip = 0; mip < texture.MipLevels; ++mip) chain.Levels.push_back(MipView(texture, mip, MTL::TextureType2DArray, layers));
    return chain;
}

void RecordMipmaps(ComputeEncoder encoder, const ComputePipeline &pipeline, const MipChain &chain) {
    const AutoreleaseScope pool;
    if (chain.Levels.size() < 2) return;
    pipeline.Bind(encoder);
    for (size_t mip = 1; mip < chain.Levels.size(); ++mip) {
        const auto &destination = chain.Levels[mip];
        encoder.Cmd->SetTexture(*chain.Levels[mip - 1], 0);
        encoder.Cmd->SetTexture(*destination, 1);
        encoder->dispatchThreads(MTL::Size(destination.Extent.Width, destination.Extent.Height, destination.Handle->arrayLength()), MTL::Size(8, 8, 1));
        ComputeBarrier(encoder);
    }
}
} // namespace mtl
