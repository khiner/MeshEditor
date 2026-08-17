#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// A body's acoustic surface (KHR_audio_rigid_bodies): scene data describing a surface's finish, authored in glTF and carried on the node.
// Separate from the surface-contact synthesis model in src/audio/surface/, which reads this, so importing, serializing, and editing a surface stay available however SURFACE_AUDIO was configured.

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
    std::string Name{};
    float Roughness{2e-6f}; // Root-mean-square asperity height sigma, m.
    float CorrelationLength{5e-5f}; // Lateral asperity spacing l, m.
    float SpectralSlope{-2.4f}; // Exponent p of the one-dimensional roughness power spectrum, which varies as q^p.
    // Shortest wavelength the spectrum above holds, m: the fine end of the band the surface is described over, as the correlation length is its coarse end.
    // A self-affine surface has no shortest wavelength of its own and both its spectral moments diverge without one, so the density of asperities and the radius of curvature at one exist only against a stated cutoff (Nayak 1971).
    float ShortWavelength{5e-5f / 16}; // Machined, matching the three parameters above.
    // Waviness: the surface's departure from flat at scales above the finish, which ISO 4287 separates from roughness.
    // Two faces meet only where waviness brings them within reach of each other, so it decides how much of a shared area the asperities are available over, and with it how many bear.
    // A surface flat above its finish bears over the whole area it shares, so this is defaulted alongside the finish rather than to zero.
    float Waviness{2e-6f}; // Root-mean-square height of that departure, m.
    float WavinessLength{2e-3f}; // Lateral scale it varies over, m.
    std::vector<float> Profile{}; // Measured heights along a track, m. Empty synthesizes a track from the parameters above.
    float SampleSpacing{0}; // Distance along the surface between consecutive Profile samples, m.
    std::optional<SurfaceNormalTexture> NormalTexture{};

    // Whether the authored profile can supply the surface's height track in place of the parameters above.
    bool HasMeasuredProfile() const { return Profile.size() >= 2 && SampleSpacing > 0; }

    bool operator==(const ContactSurface &) const = default;
};

// Viewport-level sustained-contact controls, for two surfaces driving the same modes for as long as they touch.
// Separate from ModalSoundControls, which the collision path owns.
struct SurfaceSoundControls {
    uint32_t MaxVoices{16}; // Cap on simultaneous sustained-contact voices.
    float SustainLevel{1.f}; // Level of the sustained-contact excitation.
    float AccelNoiseGain{1.f}; // Level of the acceleration noise a body's rigid recoil radiates. Zero disables it.
    float Coupling{1.f}; // How much of an object's own vibration modulates the contact separation.
    float ContactDamping{1.f}; // Scale on the Hunt-Crossley dissipation the physics restitution implies.
    // A persisting contact sounds only once its slip or either sweep speed (m/s) clears its floor.
    float MinSlipSpeed{0.005f}, MinSweepSpeed{0.005f};
    // Silence one modal drive row each, so a feedback loop through the modes can be attributed to a row.
    // The body feed and every other channel stay live.
    bool MuteGeometricDrive{false}, MuteFrictionDrive{false};
};

// Content key of a mesh's ContactSurface microscale finish track, derived on edit so a contact adopts its pool slot without rehashing a measured profile every frame.
struct SurfaceFinishKey {
    uint64_t Value{0};
};

// A named finish, for authoring by name rather than by measurement.
struct ContactSurfacePreset {
    const char *Name;
    float Roughness, CorrelationLength, SpectralSlope;
};

// Band a preset finish is described over, as a ratio of its correlation length to its shortest wavelength.
// This is a budget rather than a measurement: a track holds a fixed sample count, so the band a surface is described over trades against the distance a contact slides before that surface repeats.
// The 3.125 um it gives a machined finish is about the scale a profilometer reports.
constexpr float PresetBandRatio{16};

// Representative finishes from the extension's authoring notes.
namespace surfaces::acoustic {
constexpr std::array All{
    ContactSurfacePreset{"Polished", 1e-7f, 1e-5f, -2.8f},
    ContactSurfacePreset{"Machined", 2e-6f, 5e-5f, -2.4f},
    ContactSurfacePreset{"Sandblasted", 1e-5f, 1e-4f, -2.2f},
    ContactSurfacePreset{"Cast", 1e-4f, 1e-3f, -2.0f},
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
    out.ShortWavelength = p.CorrelationLength / PresetBandRatio;
    return out;
}
