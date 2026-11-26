#pragma once
#include "gpu/Types.h"

// Each shader function-constant index is its bit position in the pipeline key.
enum class PbrFeature : uint32_t {
    SceneLights, Transmission, DiffuseTrans, Clearcoat, Sheen, Anisotropy, Iridescence,
    AlphaBlend, OpaqueReceiver, AreaLights, OpaqueShadows, CullShadowBackfaces, TiledLights, MultipleShadowSamples,
    TransmissionPrepass, NonTriangleTopology, TraceDiffuse, TraceReflections, PreparePrimary,
    // A material in view computes inputs, which the points it shades read from the pattern words.
    // A hit kernel with PatternResume shades the samples whose first surface the hit driver evaluated.
    Patterns, PatternResume,
    // These two select pipelines on the CPU.
    BoundedAreaShadows, RayTracing,
};

#ifndef __METAL_VERSION__
using PbrFeatureMask = uint32_t;
constexpr PbrFeatureMask FeatureBit(PbrFeature f) { return 1u << uint32_t(f); }
constexpr PbrFeatureMask operator|(PbrFeatureMask a, PbrFeature b) { return a | FeatureBit(b); }
constexpr PbrFeatureMask &operator|=(PbrFeatureMask &a, PbrFeature b) { return a = a | b; }
constexpr bool HasFeature(PbrFeatureMask mask, PbrFeature f) { return (mask & FeatureBit(f)) != 0; }
// The features that come from a scene's materials.
constexpr auto MeshMaterialFeatures = FeatureBit(PbrFeature::Transmission) | PbrFeature::DiffuseTrans | PbrFeature::Clearcoat | PbrFeature::Sheen | PbrFeature::Anisotropy | PbrFeature::Iridescence;
#endif
