#ifndef MIPDOWNSAMPLE_MSL
#define MIPDOWNSAMPLE_MSL

#include "Varyings.metal"

// Each destination texel centre lands on the corner four source texels share, so the linear filter
// weights them evenly and the level is their average.
fragment float4 MipDownsampleFragment(
    QuadVaryings in [[stage_in]],
    texture2d<float> source [[texture(0)]],
    sampler source_sampler [[sampler(0)]]
) {
    return source.sample(source_sampler, in.TexCoord);
}

#endif
