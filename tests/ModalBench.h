#pragma once

// The modal render harness the audio cases share: one bank of identical objects ready to render, the strike that drives them, and the metrics a comparison reads off two renders.
// Nothing here reaches the surface-contact model, so it builds however SURFACE_AUDIO was configured.

#include "audio/ModalAudio.h"
#include "audio/ModalModes.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

constexpr float SampleRate{48'000};
constexpr uint32_t BlockSize{512};
constexpr uint32_t SamplePoints{4};

// Lay the sample points out as a strip of alternating depth, so consecutive point triples form triangles with area, and index them.
inline void SampleStrip(ModalModes &modes) {
    modes.Shapes.resize(SamplePoints);
    modes.Positions.resize(SamplePoints);
    for (uint32_t p = 0; p < SamplePoints; ++p) modes.Positions[p] = vec3{float(p) * 0.01f, 0.f, p % 2 ? 0.02f : 0.f};
    for (uint32_t p = 0; p + 2 < SamplePoints; ++p) modes.Indices.insert(modes.Indices.end(), {p, p + 1, p + 2});
}

inline ModalModes MakeModes(uint32_t mode_count, float longest_t60, float shape_scale = 1.f) {
    ModalModes modes;
    for (uint32_t k = 0; k < mode_count; ++k) {
        modes.Freqs.push_back(40.f * float(k + 1) * 1.031f);
        modes.T60s.push_back(longest_t60 / float(k + 1));
    }
    SampleStrip(modes);
    for (uint32_t p = 0; p < SamplePoints; ++p) {
        for (uint32_t k = 0; k < mode_count; ++k) {
            const float a = float(k + 1) * 0.37f + float(p);
            modes.Shapes[p].push_back(vec3{std::sin(a), std::cos(a * 1.7f), std::sin(a * 2.3f)} * 0.01f * shape_scale);
        }
    }
    return modes;
}

inline ModalEvent ImpactEvent(uint32_t object, float impulse, uint32_t ex_pos = 0, float pulse_step = 1.f / 300.f) {
    return {.Kind = ModalEventKind::Impact, .Object = object, .ExPos = ex_pos, .Jx = impulse, .Jy = 0.5f * impulse, .Jz = 0.f, .PulseStep = pulse_step, .PulseGamma = 20.f, .AccelAmp = 0.f};
}

// One bank of identical objects, ready to render.
struct ModalScene {
    ModalAudio Audio;
    std::vector<uint32_t> Objects;

    ModalScene(uint32_t object_count, uint32_t mode_count, float longest_t60, uint32_t renderers, float rigid_inv_mass = 0.f, float sample_rate = SampleRate, float shape_scale = 1.f)
        : ModalScene(MakeModes(mode_count, longest_t60, shape_scale), object_count, renderers, rigid_inv_mass, sample_rate) {}

    // One bank of whatever modes a case needs to press, for a bench that needs its own frequency.
    ModalScene(const ModalModes &modes, uint32_t object_count, uint32_t renderers, float rigid_inv_mass = 0.f, float sample_rate = SampleRate) {
        Audio.RenderPool.SetSize(renderers);
        ModalBank next;
        next.SampleRate = sample_rate;
        for (uint32_t o = 0; o < object_count; ++o) {
            Objects.push_back(AddModalObject(next, entt::entity{o}, modes));
            TuneModalObject(next, Objects.back(), modes.Freqs, modes.T60s);
            next.OutGain[Objects.back()] = 1.f;
            next.RigidInvMass[Objects.back()] = rigid_inv_mass;
        }
        InstallModalBank(Audio, next);
        // Installing a bank flags queued events as addressing the old slot layout, which the next render clears.
        // One silent block does it, before anything is queued against the new bank.
        std::vector<float> discard(BlockSize, 0.f);
        RenderModal(Audio, discard.data(), BlockSize);
    }

    void Strike(float impulse) {
        for (const auto o : Objects) EnqueueModalEvent(Audio, ImpactEvent(o, impulse));
    }

    std::vector<float> Render(uint32_t blocks, uint32_t frames) {
        std::vector<float> signal(size_t(blocks) * frames, 0.f);
        for (uint32_t block = 0; block < blocks; ++block) RenderModal(Audio, signal.data() + size_t(block) * frames, frames);
        return signal;
    }
};

inline float Peak(std::span<const float> signal) {
    float peak = 0;
    for (const float s : signal) peak = std::max(peak, std::abs(s));
    return peak;
}

// The largest sample-for-sample gap between two renders.
inline float MaxDifference(std::span<const float> a, std::span<const float> b) {
    float worst = 0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
}

inline double Rms(std::span<const float> signal) {
    double energy = 0;
    for (const float s : signal) energy += double(s) * double(s);
    return std::sqrt(energy / double(signal.size()));
}

inline bool AllFinite(std::span<const float> signal) {
    return std::ranges::all_of(signal, [](float s) { return std::isfinite(s); });
}
