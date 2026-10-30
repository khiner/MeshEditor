#pragma once

#include "metal/Command.h"
#include "metal/Image.h"
#include "metal/Shader.h"

namespace mtl {
// Each level of a texture as a 2D array view, for downsampling on the GPU.
// The texture needs shader write usage.
struct MipChain {
    std::vector<Texture> Levels;
};

MipChain CreateMipChain(const Texture &);

// Fills each level after the first with a linear downsample of the level above, and orders the encoder's later dispatches after the last.
// The encoder's own mip generation does not keep each level's mean on HDR content, so every mip chain fills through this.
void RecordMipmaps(ComputeEncoder, const ComputePipeline &, const MipChain &);
} // namespace mtl
