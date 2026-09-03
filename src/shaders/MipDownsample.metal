#ifndef MIPDOWNSAMPLE_MSL
#define MIPDOWNSAMPLE_MSL

#include "Varyings.metal"

// Sample the shared corner of four source texels to obtain their linear-filtered average.
fragment float4 MipDownsampleFragment(
    QuadVaryings in [[stage_in]],
    texture2d<float> source [[texture(0)]],
    sampler source_sampler [[sampler(0)]]
) {
    return source.sample(source_sampler, in.TexCoord);
}

#endif
