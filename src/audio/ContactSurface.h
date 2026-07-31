#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Tangent-space normal map giving a surface's mesoscale structure, mirroring glTF's normalTextureInfo.
struct SurfaceNormalTexture {
    uint32_t Texture{0}; // Index into gltf::SourceAssets::Textures.
    uint32_t TexCoord{0};
    float Scale{1};

    bool operator==(const SurfaceNormalTexture &) const = default;
};

// The finish of a body's surface below the scale of its collision geometry, per mesh entity (KHR_audio_rigid_bodies acoustic surface).
// Lengths are absolute and stay fixed when the node is scaled.
struct ContactSurface {
    std::string Name;
    float Roughness{2e-6f}; // Root-mean-square asperity height sigma, m.
    float CorrelationLength{5e-5f}; // Lateral asperity spacing l, m.
    float SpectralSlope{-1.4f}; // Exponent p of the roughness power spectrum, which varies as q^p.
    std::vector<float> Profile; // Measured heights along a track, m. Empty synthesizes a track from the parameters above.
    float SampleSpacing{0}; // Distance along the surface between consecutive Profile samples, m.
    std::optional<SurfaceNormalTexture> NormalTexture;

    // Whether the authored profile can supply the surface's height track in place of the parameters above.
    bool HasMeasuredProfile() const { return Profile.size() >= 2 && SampleSpacing > 0; }

    bool operator==(const ContactSurface &) const = default;
};

// Content key of a mesh's ContactSurface microscale finish track, derived on edit so a contact adopts its pool slot
// without rehashing a measured profile every frame.
struct SurfaceFinishKey {
    uint64_t Value{0};
};

// A named finish, for authoring by name rather than by measurement.
struct ContactSurfacePreset {
    const char *Name;
    float Roughness, CorrelationLength, SpectralSlope;
};

// Representative finishes from the extension's authoring notes.
namespace surfaces::acoustic {
constexpr std::array All{
    ContactSurfacePreset{"Polished", 1e-7f, 1e-5f, -1.8f},
    ContactSurfacePreset{"Machined", 2e-6f, 5e-5f, -1.4f},
    ContactSurfacePreset{"Sandblasted", 1e-5f, 1e-4f, -1.2f},
    ContactSurfacePreset{"Cast", 1e-4f, 1e-3f, -1.0f},
};
inline constexpr const auto &Default{All[1]}; // Machined, matching ContactSurface's own defaults.
} // namespace surfaces::acoustic

// The surface with a preset's three microscale parameters applied, keeping any authored profile and normal map.
inline ContactSurface WithPreset(const ContactSurface &s, const ContactSurfacePreset &p) {
    auto out = s;
    out.Name = p.Name;
    out.Roughness = p.Roughness;
    out.CorrelationLength = p.CorrelationLength;
    out.SpectralSlope = p.SpectralSlope;
    return out;
}
