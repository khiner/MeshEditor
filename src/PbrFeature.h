#pragma once

#include <cstdint>

enum class PbrFeature : uint32_t {
    Punctual = 1 << 0,
    Transmission = 1 << 1,
    DiffuseTrans = 1 << 2,
    Clearcoat = 1 << 3,
    Sheen = 1 << 4,
    Anisotropy = 1 << 5,
    Iridescence = 1 << 6,
};

using PbrFeatureMask = uint32_t;

inline PbrFeatureMask operator|(PbrFeatureMask a, PbrFeature b) { return a | uint32_t(b); }
inline PbrFeatureMask &operator|=(PbrFeatureMask &a, PbrFeature b) { return a = a | b; }
inline bool HasFeature(PbrFeatureMask mask, PbrFeature f) { return (mask & uint32_t(f)) != 0; }
