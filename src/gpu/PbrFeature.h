#pragma once
#include "gpu/Types.h"

// Each shader function-constant index is its bit position in the pipeline key.
enum class PbrFeature : uint32_t {
    SceneLights, Transmission, DiffuseTrans, Clearcoat, Sheen, Anisotropy, Iridescence,
    AlphaBlend, OpaqueReceiver, AreaLights, OpaqueShadows, CullShadowBackfaces, TiledLights, MultipleShadowSamples,
    TransmissionPrepass, NonTriangleTopology, TraceDiffuse, TraceReflections, PreparePrimary,
    // These two select pipelines on the CPU.
    BoundedAreaShadows, RayTracing,
};

#ifndef __METAL_VERSION__
using PbrFeatureMask = uint32_t;
constexpr PbrFeatureMask FeatureBit(PbrFeature f) { return 1u << uint32_t(f); }
constexpr PbrFeatureMask operator|(PbrFeatureMask a, PbrFeature b) { return a | FeatureBit(b); }
constexpr PbrFeatureMask &operator|=(PbrFeatureMask &a, PbrFeature b) { return a = a | b; }
constexpr bool HasFeature(PbrFeatureMask mask, PbrFeature f) { return (mask & FeatureBit(f)) != 0; }
// These glTF material extensions use the general PBR shader for secondary hits.
constexpr auto ExtendedPbrMaterials = FeatureBit(PbrFeature::DiffuseTrans) | PbrFeature::Clearcoat | PbrFeature::Sheen | PbrFeature::Anisotropy | PbrFeature::Iridescence;
constexpr auto MeshMaterialFeatures = ExtendedPbrMaterials | PbrFeature::Transmission;
#endif
