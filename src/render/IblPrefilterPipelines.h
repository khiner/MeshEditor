#pragma once

#include "metal/Shader.h"

// The environment prefilter passes, which run once per loaded environment rather than per frame.
// They bind their source and destination directly instead of going through the bindless table.
struct IblPrefilterPipelines {
    mtl::ComputePipeline EquirectToCubemap, DiffuseIrradiance, SpecularPrefilter;

    explicit IblPrefilterPipelines(mtl::LibraryCache &libraries)
        : EquirectToCubemap{libraries, {"IblPrefilter.metal", "EquirectToCubemapKernel"}},
          DiffuseIrradiance{libraries, {"IblPrefilter.metal", "DiffuseIrradianceKernel"}},
          SpecularPrefilter{libraries, {"IblPrefilter.metal", "SpecularPrefilterKernel"}} {}
};
