#include "ModalAudio.h"
#include "ModalModes.h"

#include <entt/entity/entity.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <thread>

#ifdef __APPLE__
#include <os/workgroup.h>
#endif

namespace {
// Modes are rendered in fixed-width lanes so the sample loop vectorizes across modes.
constexpr uint32_t Lanes{8};
// An object whose gain-weighted state energy falls below this (with no active impacts) is zeroed and skipped.
constexpr float SilentEnergy{1e-12f};
// Distance over which the relief's local mean is removed, m.
// Far longer than any wavelength the contact filter passes, so a settled contact is exactly silent.
constexpr float ReliefDcLength{1e-2f};
// A voice this long without a fresh contact report ends itself.
constexpr float MaxVoiceIdleSeconds{0.1f};

std::optional<uint32_t> FindIndex(const auto &v, const auto &value) {
    const auto it = std::ranges::find(v, value);
    return it != v.end() ? std::optional{uint32_t(std::ranges::distance(v.begin(), it))} : std::nullopt;
}

// Drop element `i` from each parallel column, swapping the last into its place.
void SwapRemove(uint32_t i, auto &...columns) {
    ((columns[i] = columns.back(), columns.pop_back()), ...);
}

void RemoveImpact(ModalBank &b, uint32_t i) {
    SwapRemove(
        i, b.ImpactObject, b.ImpactExPos, b.ImpactSamplesLeft, b.ImpactJx, b.ImpactJy, b.ImpactJz,
        b.ImpactPhaseRe, b.ImpactPhaseIm, b.ImpactRotRe, b.ImpactRotIm,
        b.ImpactGamma, b.ImpactAccelAmp, b.ImpactPrevForce
    );
}

void ActivateImpact(const ModalAudio &m, ModalBank &b, const ModalEvent &e) {
    if (b.ImpactObject.size() >= m.MaxImpacts.load(std::memory_order_relaxed)) return;
    b.ImpactObject.push_back(e.Object);
    b.ImpactExPos.push_back(e.ExPos);
    b.ImpactSamplesLeft.push_back(uint32_t(std::ceil(1.f / e.PulseStep)));
    b.ImpactJx.push_back(e.Jx);
    b.ImpactJy.push_back(e.Jy);
    b.ImpactJz.push_back(e.Jz);
    b.ImpactPhaseRe.push_back(1.f);
    b.ImpactPhaseIm.push_back(0.f);
    const auto theta = std::numbers::pi_v<float> * e.PulseStep;
    b.ImpactRotRe.push_back(std::cos(theta));
    b.ImpactRotIm.push_back(std::sin(theta));
    b.ImpactGamma.push_back(e.PulseGamma);
    b.ImpactAccelAmp.push_back(e.AccelAmp);
    b.ImpactPrevForce.push_back(0.f);
    b.Ringing[e.Object] = 1;
}

void RemoveVoice(ModalBank &b, uint32_t i) {
    SwapRemove(i, b.VoiceId, b.VoiceObject, b.VoiceState, b.VoiceCarry);
}

void SilenceObject(ModalBank &b, uint32_t o) {
    const uint32_t k0 = b.ModeOffset[o], count = b.ModeCount[o];
    std::fill_n(b.StateRe.begin() + k0, count, 0.f);
    std::fill_n(b.StateIm.begin() + k0, count, 0.f);
    b.Ringing[o] = 0;
    // The next strike starts from the whole tuned set again.
    b.LiveModeCount[o] = b.TunedModeCount[o];
    for (uint32_t i = uint32_t(b.ImpactObject.size()); i-- > 0;) {
        if (b.ImpactObject[i] == o) RemoveImpact(b, i);
    }
    for (uint32_t i = uint32_t(b.VoiceId.size()); i-- > 0;) {
        if (b.VoiceObject[i] == o) RemoveVoice(b, i);
    }
}

void DrainEvents(ModalAudio &m, ModalBank &b) {
    auto read = m.EventRead.load(std::memory_order_relaxed);
    const auto write = m.EventWrite.load(std::memory_order_acquire);
    for (; read != write; ++read) {
        const auto &e = m.Events[read % ModalAudio::EventCapacity];
        if (e.Object >= b.Entities.size()) continue;
        switch (e.Kind) {
            case ModalEventKind::Impact:
                if (e.PulseStep > 0) ActivateImpact(m, b, e);
                break;
            case ModalEventKind::Silence:
                SilenceObject(b, e.Object);
                break;
        }
    }
    m.EventRead.store(read, std::memory_order_release);
}

// Bring the bank's voices in line with the newest published set.
// A contact already open keeps its carried state, one the set omits is over, and a new one opens.
// Once publishing stops for MaxVoiceIdleSeconds the set is treated as empty, silencing the scene.
void AdoptVoices(ModalAudio &m, ModalBank &b, uint32_t frame_count) {
    const auto *set = m.PublishedVoices.load(std::memory_order_acquire);
    if (set && set->Frame != m.AdoptedVoiceFrame) {
        m.AdoptedVoiceFrame = set->Frame;
        m.VoiceSetIdleSamples = 0;
    } else {
        m.VoiceSetIdleSamples += frame_count;
    }
    const bool reporting = set != nullptr && m.VoiceSetIdleSamples <= uint32_t(b.SampleRate * MaxVoiceIdleSeconds);
    const auto named = [&](uint64_t id) {
        return reporting && std::ranges::contains(set->Voices, id, &VoiceSet::Voice::Id);
    };
    for (uint32_t v = uint32_t(b.VoiceId.size()); v-- > 0;) {
        if (!named(b.VoiceId[v])) RemoveVoice(b, v);
    }
    if (reporting) {
        for (const auto &voice : set->Voices) {
            // A set built against a bank that has since been replaced can name slots this one does not have.
            if (voice.Object >= b.Entities.size()) continue;
            if (const auto v = FindIndex(b.VoiceId, voice.Id)) {
                b.VoiceObject[*v] = voice.Object;
                b.VoiceState[*v] = voice.State;
            } else {
                b.VoiceId.push_back(voice.Id);
                b.VoiceObject.push_back(voice.Object);
                b.VoiceState.push_back(voice.State);
                b.VoiceCarry.emplace_back();
            }
            b.Ringing[voice.Object] = 1;
        }
    }
    // Published so the main thread never repoints a slot a live voice reads.
    uint64_t mask = 0;
    for (const auto &st : b.VoiceState) {
        for (const auto &t : st.Tracks) {
            if (t.Index >= 0) mask |= 1ull << uint32_t(t.Index);
        }
    }
    m.VoiceTrackMask.store(mask, std::memory_order_release);
}

// The impulse of an impact projected onto `n` mode shapes starting at mode `first`.
void ImpactGainRow(const ModalBank &b, uint32_t impact, uint32_t shape0, uint32_t count, uint32_t first, uint32_t n, float *out) {
    const auto base = shape0 + b.ImpactExPos[impact] * count + first;
    const auto jx = b.ImpactJx[impact], jy = b.ImpactJy[impact], jz = b.ImpactJz[impact];
    for (uint32_t i = 0; i < n; ++i) out[i] = b.ShapeX[base + i] * jx + b.ShapeY[base + i] * jy + b.ShapeZ[base + i] * jz;
}

// Modes advance in `Lanes`-wide chunks with all state in locals, so the per-sample loop is branchless and vectorizes across the chunk.
// Excitation gains hoist out of the sample loop, which a coupled voice cannot do.
void RenderObjectFast(ModalAudio &m, ModalAudio::RenderScratch &w, ModalBank &b, uint32_t o, std::span<const uint32_t> impacts, float *out, uint32_t frame_count) {
    // A shape row is `stride` wide, and only the leading `count` of it still sounds.
    // An impact drives every mode the tuning left, so being struck restores the whole set.
    const auto k0 = b.ModeOffset[o], stride = b.ModeCount[o];
    const auto count = impacts.empty() ? b.LiveModeCount[o] : b.TunedModeCount[o];
    const auto shape0 = b.ShapeOffset[o];
    const auto out_gain = std::atomic_ref{b.OutGain[o]}.load(std::memory_order_relaxed);
    w.Gains.resize(impacts.size() * Lanes);
    float energy = 0.f;
    // The end of the last chunk still carrying audible state.
    uint32_t live = 0;
    for (uint32_t k = 0; k < count; k += Lanes) {
        const auto width = std::min(Lanes, count - k);
        float z_re[Lanes]{}, z_im[Lanes]{}, c_re[Lanes]{}, c_im[Lanes]{};
        for (uint32_t l = 0; l < width; ++l) {
            z_re[l] = b.StateRe[k0 + k + l];
            z_im[l] = b.StateIm[k0 + k + l];
            c_re[l] = b.CoeffRe[k0 + k + l];
            c_im[l] = b.CoeffIm[k0 + k + l];
        }
        for (size_t t = 0; t < impacts.size(); ++t) {
            auto *gain = &w.Gains[t * Lanes];
            ImpactGainRow(b, impacts[t], shape0, stride, k, width, gain);
            std::fill(gain + width, gain + Lanes, 0.f);
        }
        for (uint32_t s = 0; s < frame_count; ++s) {
            float excite[Lanes]{};
            for (size_t t = 0; t < impacts.size(); ++t) {
                const auto force = m.ForceScratch[size_t(impacts[t]) * frame_count + s];
                if (force == 0.f) continue;
                const auto *gain = &w.Gains[t * Lanes];
                for (uint32_t l = 0; l < Lanes; ++l) excite[l] += force * gain[l];
            }
            float acc = 0.f;
            for (uint32_t l = 0; l < Lanes; ++l) {
                const auto re = z_re[l] * c_re[l] - z_im[l] * c_im[l] + excite[l];
                z_im[l] = z_re[l] * c_im[l] + z_im[l] * c_re[l];
                z_re[l] = re;
                acc += z_im[l];
            }
            out[s] += acc * out_gain;
        }
        float chunk = 0.f;
        for (uint32_t l = 0; l < width; ++l) {
            b.StateRe[k0 + k + l] = z_re[l];
            b.StateIm[k0 + k + l] = z_im[l];
            chunk += z_re[l] * z_re[l] + z_im[l] * z_im[l];
        }
        energy += chunk;
        if (chunk * out_gain * out_gain >= SilentEnergy) live = k + width;
    }
    if (impacts.empty() && energy * out_gain * out_gain < SilentEnergy) {
        SilenceObject(b, o);
        return;
    }
    b.Ringing[o] = 1;
    b.LiveModeCount[o] = impacts.empty() ? live : b.TunedModeCount[o];
}

// The excitation a sustained voice contributes this sample (KHR_audio_rigid_bodies Appendix B).
// The normal part is the contact force's fluctuation about the load.
// The tangential part has two mechanisms acting along different directions: the load projected onto each surface's
// own tilt, along the direction the contact travels over it, and Coulomb traction along the slip.
struct VoiceForce {
    float Normal;
    std::array<float, 2> Geometric; // Per surface, in the contact's own surface order, matching SustainedState::SweepDir.
    float Frictional;
};

// tanh by its [5/4] Pade approximant, within a float ULP for |x| <= 1 and 7e-7 by |x| = 2.
// Past 4.9, tanh is 1 to within 1e-4, so it saturates there.
float FastTanh(float x) {
    const float a = std::fabs(x);
    if (a > 4.9f) return std::copysign(1.f, x);
    const float x2 = x * x;
    const float n = x * (10395.f + x2 * (1260.f + x2 * 21.f));
    const float d = 10395.f + x2 * (4725.f + x2 * (210.f + x2));
    return std::clamp(n / d, -1.f, 1.f);
}

// One sample of the reference contact force model.
// `deflection` is the object's modal displacement along the normal at the previous sample.
// That one-sample delay is the explicit discretization of the coupling.
VoiceForce StepVoice(const SustainedState &st, const RoughnessTrack *const *tracks, SustainedCarry &carry, float deflection, float sample_rate) {
    // Each track advances at its own surface's sweep speed, indexed by distance along the surface rather than by time.
    // A voice's first sample has no previous height to difference against, so it contributes no slope.
    const bool priming = !carry.Primed;
    carry.Primed = true;
    // Two surfaces are independent even when their parameters agree, so their tracks start a quarter apart.
    // The offset follows the track's slot, which both voices of a contact fill in the same order, so the two
    // read one surface at one position.
    if (priming) {
        for (uint32_t i = 0; i < SustainedState::TrackCount; ++i) carry.Pos[i] = double(i) * double(TrackSamples) / SustainedState::TrackCount;
    }

    // A track's surface is its index's low bit, so each surface's slope accumulates separately and lands on the
    // direction SustainedState::SweepDir holds for that surface.
    float relief = 0, distance = 0;
    std::array<float, 2> slope{};
    for (uint32_t i = 0; i < SustainedState::TrackCount; ++i) {
        const auto &t = st.Tracks[i];
        if (!tracks[i]) continue;
        carry.Pos[i] += t.Rate;
        const float height = t.Sigma * ReadTrack(*tracks[i], carry.Pos[i], t.Window);
        relief += height;
        // The slope is a difference over the distance travelled, so it is zero at rest and zero-mean while moving.
        if (t.Step > 0 && !priming) slope[i & 1] += (height - carry.PrevHeight[i]) / t.Step;
        carry.PrevHeight[i] = height;
        distance = std::max(distance, t.Step);
    }

    // Removing the relief's local mean leaves a contact at rest sitting exactly at equilibrium, exciting nothing.
    // The load itself is subtracted exactly, below.
    if (priming) {
        carry.ReliefMean = relief;
        carry.Penetration = std::max(st.StaticPenetration, 0.f);
    }
    carry.ReliefMean += (relief - carry.ReliefMean) * std::min(distance / ReliefDcLength, 1.f);

    const float rigid_approach = st.StaticPenetration + relief - carry.ReliefMean;
    // The clamp at zero is the separation nonlinearity that produces micro-collisions and chatter.
    const float separation = std::max(rigid_approach - deflection, 0.f);
    const float separation_rate = (separation - carry.Penetration) * sample_rate;
    carry.Penetration = separation;

    // Hunt and Crossley: f_n = k * delta^(3/2) * (1 + c_d * delta_dot), and a contact never pulls.
    const float force = std::max(st.Stiffness * separation * std::sqrt(separation) * (1 + st.DampingCoeff * separation_rate), 0.f);
    // The excitation is the fluctuation about the load, soft-limited upward by a knee that scales with it and meets
    // the straight part at the same slope. A contact that lifts clear already bottoms out at exactly minus the load,
    // and rounding that off would soften the separation the clamp above produces.
    float normal = force - st.NormalForce;
    if (normal > 0 && st.NormalForce > 0) normal = st.NormalForce * FastTanh(normal / st.NormalForce);
    // The load the tilt projection acts on, damping factor included.
    const float load = st.NormalForce + normal;
    // Coulomb traction rides on the same bounded fluctuation, so the knee above bounds it to mu times the load.
    return {normal, {load * slope[0], load * slope[1]}, st.Friction * normal};
}

// The object's modal displacement along a voice's contact normal, from the state the previous sample left.
// The sum runs the length of the mode count, so it reassociates into independent partial sums rather than one dependency chain.
float ReadDeflection(const float *__restrict gains, const float *__restrict state_im, uint32_t count) {
#pragma clang fp reassociate(on)
    float sum = 0.f;
    for (uint32_t k = 0; k < count; ++k) sum += gains[k] * state_im[k];
    return sum;
}

// A voice's drive rows: the normal, each surface's geometric tangential, and the frictional.
constexpr uint32_t VoiceDrives{4};

// This sample's excitation of every mode from the drives past the first voice's, summed over those that carry force.
void GatherExcitation(const float *__restrict gains, const float *__restrict forces, uint32_t voice_drives, uint32_t drives, float *__restrict excite, uint32_t count) {
    bool seeded = false;
    const auto row = [&](uint32_t d) { return gains + size_t(d) * count; };
    // A voice's four rows all stand behind the same contact, so they enter together in one pass.
    for (uint32_t d = VoiceDrives; d < voice_drives; d += VoiceDrives) {
        const float f0 = forces[d], f1 = forces[d + 1], f2 = forces[d + 2], f3 = forces[d + 3];
        const float *__restrict g0 = row(d), *__restrict g1 = row(d + 1), *__restrict g2 = row(d + 2), *__restrict g3 = row(d + 3);
        if (seeded) {
            for (uint32_t k = 0; k < count; ++k) excite[k] += g0[k] * f0 + g1[k] * f1 + g2[k] * f2 + g3[k] * f3;
        } else {
            for (uint32_t k = 0; k < count; ++k) excite[k] = g0[k] * f0 + g1[k] * f1 + g2[k] * f2 + g3[k] * f3;
            seeded = true;
        }
    }
    // An impact stands on its own row, and carries no force once its pulse is over.
    for (uint32_t d = voice_drives; d < drives; ++d) {
        const float force = forces[d];
        if (force == 0.f) continue;
        const float *__restrict g = row(d);
        if (seeded) {
            for (uint32_t k = 0; k < count; ++k) excite[k] += g[k] * force;
        } else {
            for (uint32_t k = 0; k < count; ++k) excite[k] = g[k] * force;
            seeded = true;
        }
    }
    if (!seeded) std::fill_n(excite, count, 0.f);
}

// Advance every mode one sample and return what the object radiates.
// The first voice's drive rows enter here directly, and its next deflection is read off the state this leaves.
// Both sums run the length of the mode count, so they reassociate into independent partial sums rather than one dependency chain.
float AdvanceModes(
    float *__restrict state_re, float *__restrict state_im, const float *__restrict coeff_re, const float *__restrict coeff_im,
    const float *__restrict gains, const float *__restrict forces, const float *__restrict excite,
    const float *__restrict read_gains, float *__restrict deflection, uint32_t count
) {
#pragma clang fp reassociate(on)
    const float f0 = forces[0], f1 = forces[1], f2 = forces[2], f3 = forces[3];
    const float *__restrict g0 = gains, *__restrict g1 = gains + count, *__restrict g2 = gains + 2 * count, *__restrict g3 = gains + 3 * count;
    float acc = 0.f, read = 0.f;
    for (uint32_t k = 0; k < count; ++k) {
        const auto excitation = g0[k] * f0 + g1[k] * f1 + g2[k] * f2 + g3[k] * f3 + excite[k];
        const auto re = state_re[k] * coeff_re[k] - state_im[k] * coeff_im[k] + excitation;
        state_im[k] = state_re[k] * coeff_im[k] + state_im[k] * coeff_re[k];
        state_re[k] = re;
        acc += state_im[k];
        read += read_gains[k] * state_im[k];
    }
    *deflection = read;
    return acc;
}

// An object holding a sustained voice.
// A voice reads out of and writes into the mode bank at the same sample, so the loop runs sample-outer.
// A sample reads each voice's deflection back, steps its force model, gathers the excitation, then advances the modes.
// Each of those is a flat pass over the modes rather than one nested loop, so each vectorizes.
void RenderObjectCoupled(
    ModalAudio &m, ModalAudio::RenderScratch &w, ModalBank &b, uint32_t o,
    std::span<const uint32_t> impacts, std::span<const uint32_t> voices,
    float *out, uint32_t frame_count
) {
    // A shape row is `stride` wide, and only the leading `count` of it still sounds.
    // A voice drives the object every sample, so it holds every mode the tuning left rather than any decayed prefix.
    const auto k0 = b.ModeOffset[o], stride = b.ModeCount[o], count = b.TunedModeCount[o];
    const auto shape0 = b.ShapeOffset[o];
    const auto out_gain = std::atomic_ref{b.OutGain[o]}.load(std::memory_order_relaxed);
    const auto sample_rate = b.SampleRate;
    const auto coupling = m.Coupling.load(std::memory_order_relaxed);
    // The impact path drives the bank with force * dt worth of impulse per sample, so a sustained force enters the same way.
    // The two channels therefore stand in the physical ratio the spec makes normative, and SustainLevel departs from it.
    const auto sustain_level = m.SustainLevel.load(std::memory_order_relaxed) / sample_rate;

    const uint32_t drives = uint32_t(voices.size()) * VoiceDrives + uint32_t(impacts.size());
    w.DriveGains.resize(size_t(drives) * count);
    w.ReadGains.resize(voices.size() * count);
    w.Forces.resize(drives);
    w.Excite.resize(count);
    const auto drive_row = [&](size_t i) { return &w.DriveGains[i * count]; };
    // Voice state is fixed for the block, so each voice's surface tracks resolve once here.
    // The load orders against the reader generation, so an ended generation means no track read here is still in use.
    w.Tracks.resize(voices.size() * SustainedState::TrackCount);
    for (size_t t = 0; t < voices.size(); ++t) {
        const auto &st = b.VoiceState[voices[t]];
        for (uint32_t i = 0; i < SustainedState::TrackCount; ++i) {
            const auto index = st.Tracks[i].Index;
            w.Tracks[t * SustainedState::TrackCount + i] =
                index < 0 ? nullptr : m.SurfaceTracks[uint32_t(index)].Live.load(std::memory_order_seq_cst);
        }
        auto *gain_n = drive_row(t * VoiceDrives);
        auto *gain_geo0 = drive_row(t * VoiceDrives + 1), *gain_geo1 = drive_row(t * VoiceDrives + 2);
        auto *gain_fric = drive_row(t * VoiceDrives + 3);
        auto *gain_read = &w.ReadGains[t * count];
        const auto base0 = shape0 + st.Blend.Points[0] * stride, base1 = shape0 + st.Blend.Points[1] * stride, base2 = shape0 + st.Blend.Points[2] * stride;
        const float w0 = st.Blend.Weights.x, w1 = st.Blend.Weights.y, w2 = st.Blend.Weights.z;
        for (uint32_t k = 0; k < count; ++k) {
            const vec3 shape{
                w0 * b.ShapeX[base0 + k] + w1 * b.ShapeX[base1 + k] + w2 * b.ShapeX[base2 + k],
                w0 * b.ShapeY[base0 + k] + w1 * b.ShapeY[base1 + k] + w2 * b.ShapeY[base2 + k],
                w0 * b.ShapeZ[base0 + k] + w1 * b.ShapeZ[base1 + k] + w2 * b.ShapeZ[base2 + k],
            };
            gain_n[k] = glm::dot(shape, st.N);
            // Each surface's geometric force acts along the contact's travel over it, and the frictional one along the slip.
            gain_geo0[k] = glm::dot(shape, st.SweepDir[0]);
            gain_geo1[k] = glm::dot(shape, st.SweepDir[1]);
            gain_fric[k] = glm::dot(shape, st.SlipDir);
            // The separation is modulated by the object's own vibration along the same normal.
            gain_read[k] = coupling * gain_n[k] * b.DisplacementScale[k0 + k];
        }
    }
    for (size_t t = 0; t < impacts.size(); ++t) {
        ImpactGainRow(b, impacts[t], shape0, stride, 0, count, drive_row(voices.size() * VoiceDrives + t));
    }

    auto *state_re = &b.StateRe[k0];
    auto *state_im = &b.StateIm[k0];
    const auto *coeff_re = &b.CoeffRe[k0];
    const auto *coeff_im = &b.CoeffIm[k0];
    auto *gains = w.DriveGains.data();
    auto *forces = w.Forces.data();
    auto *excite = w.Excite.data();
    // The first voice's read-out comes out of the advance below, so it is seeded once here and refreshed there.
    auto *read_gains = w.ReadGains.data();
    float first_deflection = ReadDeflection(read_gains, state_im, count);
    // With no drive past the first voice's the gather never runs, so the buffer holds this zero for the whole block.
    const bool gather = drives > VoiceDrives;
    if (!gather) std::fill_n(excite, count, 0.f);
    for (uint32_t s = 0; s < frame_count; ++s) {
        for (size_t t = 0; t < voices.size(); ++t) {
            const auto v = voices[t];
            const auto deflection = t == 0 ? first_deflection : ReadDeflection(read_gains + t * count, state_im, count);
            const auto f = StepVoice(b.VoiceState[v], &w.Tracks[t * SustainedState::TrackCount], b.VoiceCarry[v], deflection, sample_rate);
            forces[t * VoiceDrives] = sustain_level * f.Normal;
            forces[t * VoiceDrives + 1] = sustain_level * f.Geometric[0];
            forces[t * VoiceDrives + 2] = sustain_level * f.Geometric[1];
            forces[t * VoiceDrives + 3] = sustain_level * f.Frictional;
        }
        for (size_t t = 0; t < impacts.size(); ++t) {
            forces[voices.size() * VoiceDrives + t] = m.ForceScratch[size_t(impacts[t]) * frame_count + s];
        }
        if (gather) GatherExcitation(gains, forces, uint32_t(voices.size()) * VoiceDrives, drives, excite, count);
        out[s] += AdvanceModes(state_re, state_im, coeff_re, coeff_im, gains, forces, excite, read_gains, &first_deflection, count) * out_gain;
    }
    b.Ringing[o] = 1;
    b.LiveModeCount[o] = count;
}

// A scheduling group is reference counted, and every holder takes its own reference.
void *RetainWorkgroup([[maybe_unused]] void *workgroup) {
#ifdef __APPLE__
    if (workgroup) return os_retain(static_cast<os_workgroup_t>(workgroup));
#endif
    return workgroup;
}

void ReleaseWorkgroup([[maybe_unused]] void *workgroup) {
#ifdef __APPLE__
    if (workgroup) os_release(static_cast<os_workgroup_t>(workgroup));
#endif
}

// Holds a render thread in the host's scheduling group for as long as it lives, so the scheduler holds it to the
// same deadline as the callback it renders alongside.
struct WorkgroupMembership {
    explicit WorkgroupMembership([[maybe_unused]] void *workgroup) {
#ifdef __APPLE__
        // The membership outlives the device that published the group, so hold a reference of its own.
        if (workgroup && os_workgroup_join(static_cast<os_workgroup_t>(workgroup), &Token) == 0) Workgroup = RetainWorkgroup(workgroup);
#endif
    }
    ~WorkgroupMembership() {
#ifdef __APPLE__
        if (Workgroup) {
            os_workgroup_leave(static_cast<os_workgroup_t>(Workgroup), &Token);
            ReleaseWorkgroup(Workgroup);
        }
#endif
    }
    WorkgroupMembership(const WorkgroupMembership &) = delete;
    WorkgroupMembership &operator=(const WorkgroupMembership &) = delete;

private:
    [[maybe_unused]] void *Workgroup{nullptr};
#ifdef __APPLE__
    os_workgroup_join_token_s Token{};
#endif
};
} // namespace

ModalRenderPool::~ModalRenderPool() {
    const std::scoped_lock lock{ResizeMutex};
    StopWorkers(0);
    ReleaseWorkgroup(Workgroup);
    Workgroup = nullptr;
}

void ModalRenderPool::StopWorkers(size_t keep) {
    if (Threads.size() <= keep) return;
    for (auto i = keep; i < Threads.size(); ++i) Threads[i].request_stop();
    // A parked worker is waiting on the ticket, so it takes a bump to observe the request.
    // Dispatching to nobody keeps the workers that stay from rendering the previous block's context on this wake.
    Dispatch = 0;
    Ticket.fetch_add(1, std::memory_order_release);
    Ticket.notify_all();
    Threads.resize(keep); // jthread's destructor joins each one it drops.
}

void ModalRenderPool::ApplyLocked(uint32_t size, void *workgroup) {
    if (size == Active && workgroup == Workgroup) return;
    // A worker reads the workgroup once, when it starts, so a new group is only picked up by a new worker.
    if (workgroup != Workgroup) {
        StopWorkers(0);
        // The pool can start a worker in the group at any time, so it holds a reference while it names one.
        ReleaseWorkgroup(Workgroup);
        Workgroup = RetainWorkgroup(workgroup);
    } else if (size < Active) {
        StopWorkers(size - 1);
    }
    Active = size;

    // Workers start from the ticket as it stands now, so a run that lands before one is scheduled still wakes it.
    const auto start = Ticket.load(std::memory_order_acquire);
    for (auto i = uint32_t(Threads.size()) + 1; i < size; ++i) {
        Threads.emplace_back([this, i, start](const std::stop_token &stop) {
            const WorkgroupMembership membership{Workgroup};
            for (auto seen = start;;) {
                Ticket.wait(seen, std::memory_order_acquire);
                if (stop.stop_requested()) return;
                seen = Ticket.load(std::memory_order_acquire);
                if (i < Dispatch) {
                    Render(Context, i);
                    if (Remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) Remaining.notify_all();
                }
            }
        });
    }
}

void ModalRenderPool::SetSize(uint32_t count) {
    // More renderers than cores only trade one for another, so the core count is the ceiling.
    const auto cores = std::max(1u, std::thread::hardware_concurrency());
    const std::scoped_lock lock{ResizeMutex};
    ApplyLocked(std::clamp(count, 1u, cores), Workgroup);
}

void ModalRenderPool::SetWorkgroup(void *workgroup) {
    const std::scoped_lock lock{ResizeMutex};
    ApplyLocked(Active, workgroup);
}

ModalRenderPool::Session::Session(ModalRenderPool &pool)
    : Pool(pool), Owns(pool.ResizeMutex.try_lock()), W(Owns ? pool.Active : 1) {}

ModalRenderPool::Session::~Session() {
    if (Owns) Pool.ResizeMutex.unlock();
}

void ModalRenderPool::Session::Run(void (*render)(void *, uint32_t), void *context) {
    if (W <= 1) {
        render(context, 0);
        return;
    }
    Pool.Render = render;
    Pool.Context = context;
    Pool.Dispatch = W;
    Pool.Remaining.store(W - 1, std::memory_order_relaxed);
    Pool.Ticket.fetch_add(1, std::memory_order_release);
    Pool.Ticket.notify_all();
    render(context, 0);
    // The workers finish within the block, so spin for a bounded while before giving the core up.
    for (uint32_t spin = 0; const auto left = Pool.Remaining.load(std::memory_order_acquire); ++spin) {
        if (spin >= 1024) Pool.Remaining.wait(left, std::memory_order_acquire);
    }
}

ModalAudio::ModalAudio() : Live{std::make_unique<ModalBank>()}, Published{Live.get()} {}

void InstallModalBank(ModalAudio &m, ModalBank &next) {
    const auto old = std::move(m.Live);
    m.Live = std::make_unique<ModalBank>(std::move(next));
    m.FlushEvents.store(true, std::memory_order_relaxed); // Before Published so the adopting callback observes it.
    m.Published.store(m.Live.get(), std::memory_order_seq_cst);
    // Wait out a callback still rendering the old bank, so `old` is freed here on the main thread.
    if (const auto seq = m.ReaderSeq.load(std::memory_order_seq_cst); seq & 1) {
        while (m.ReaderSeq.load(std::memory_order_seq_cst) == seq) std::this_thread::yield();
    }
    // The new bank holds no voices, and the wait above leaves no callback reading the pool.
    m.SurfaceTrackSlotByKey.clear();
    for (auto &slot : m.SurfaceTracks) {
        slot.Live.store(nullptr, std::memory_order_relaxed);
        slot.Owned.reset();
        slot.Key = 0;
    }
    m.VoiceTrackMask.store(0, std::memory_order_relaxed);
    m.ReusableSlots = 0;
    m.ActiveVoices.store(0, std::memory_order_relaxed);
    // Every published set addresses the replaced bank's object slots.
    m.PublishedVoices.store(nullptr, std::memory_order_relaxed);
    for (auto &set : m.VoiceSets) set.Voices.clear();
}

VoiceSet &NextVoiceSet(ModalAudio &m) {
    // Two publishes back, which no callback can still be reading by the time a third frame comes round.
    m.VoiceSetWrite = (m.VoiceSetWrite + 1) % uint32_t(m.VoiceSets.size());
    auto &set = m.VoiceSets[m.VoiceSetWrite];
    set.Voices.clear();
    return set;
}

void PublishVoiceSet(ModalAudio &m) {
    auto &set = m.VoiceSets[m.VoiceSetWrite];
    set.Frame = ++m.VoiceFrame;
    m.PublishedVoices.store(&set, std::memory_order_release);
}

uint32_t AddModalObject(ModalBank &b, entt::entity e, const ModalModes &modes) {
    const auto count = uint32_t(modes.Freqs.size());
    const auto slot = uint32_t(b.Entities.size());
    b.Entities.push_back(e);
    b.ModeOffset.push_back(uint32_t(b.CoeffRe.size()));
    b.ModeCount.push_back(count);
    b.TunedModeCount.push_back(count);
    b.LiveModeCount.push_back(count);
    b.ShapeOffset.push_back(uint32_t(b.ShapeX.size()));
    b.OutGain.push_back(0.f);
    b.Ringing.push_back(0);
    b.CoeffRe.resize(b.CoeffRe.size() + count, 0.f);
    b.CoeffIm.resize(b.CoeffIm.size() + count, 0.f);
    b.StateRe.resize(b.StateRe.size() + count, 0.f);
    b.StateIm.resize(b.StateIm.size() + count, 0.f);
    b.DisplacementScale.resize(b.DisplacementScale.size() + count, 0.f);
    for (const auto &row : modes.Shapes) {
        for (const auto &shape : row) {
            b.ShapeX.push_back(shape.x);
            b.ShapeY.push_back(shape.y);
            b.ShapeZ.push_back(shape.z);
        }
    }
    return slot;
}

void TuneModalObject(ModalBank &b, uint32_t object, std::span<const float> freqs, std::span<const float> t60s) {
    const auto k0 = b.ModeOffset[object];
    const auto count = std::min(b.ModeCount[object], uint32_t(std::min(freqs.size(), t60s.size())));
    const float sr = b.SampleRate;
    for (uint32_t k = 0; k < count; ++k) {
        const float freq = freqs[k], t60 = t60s[k];
        if (!std::isfinite(freq) || !std::isfinite(t60) || freq <= 0.f || freq >= sr / 2 - 1 || t60 <= 0.f) {
            b.CoeffRe[k0 + k] = 0.f;
            b.CoeffIm[k0 + k] = 0.f;
            b.DisplacementScale[k0 + k] = 0.f;
            continue;
        }
        const auto decay = std::pow(1e-3f, 1.f / (t60 * sr));
        const auto omega = 2 * std::numbers::pi_v<float> * freq / sr;
        b.CoeffRe[k0 + k] = decay * std::cos(omega);
        b.CoeffIm[k0 + k] = decay * std::sin(omega);
        // Every drive enters as the impulse one sample carries, and a mode's response to an impulse is its
        // displacement times the angular frequency, so dividing the state by that recovers meters.
        b.DisplacementScale[k0 + k] = 1.f / (2 * std::numbers::pi_v<float> * freq);
    }
    // Written after the coefficients, so a callback that sees a longer bank also sees what fills it.
    // Only the trailing muted block is dropped, since a muted mode in the middle still has live modes behind it.
    uint32_t live = b.ModeCount[object];
    while (live > 0 && b.CoeffRe[k0 + live - 1] == 0.f && b.CoeffIm[k0 + live - 1] == 0.f) --live;
    b.TunedModeCount[object] = live;
    // A retuning moves every mode's decay, so the previous audible prefix no longer applies.
    b.LiveModeCount[object] = live;
}

bool SetModalObjectShapes(ModalBank &b, uint32_t object, const ModalModes &modes) {
    const auto begin = b.ShapeOffset[object];
    const auto end = object + 1 < b.ShapeOffset.size() ? b.ShapeOffset[object + 1] : uint32_t(b.ShapeX.size());
    const auto count = uint32_t(modes.Freqs.size());
    if (b.ModeCount[object] != count || end - begin != count * modes.Shapes.size()) return false;
    auto i = begin;
    for (const auto &row : modes.Shapes) {
        for (const auto &shape : row) {
            b.ShapeX[i] = shape.x;
            b.ShapeY[i] = shape.y;
            b.ShapeZ[i] = shape.z;
            ++i;
        }
    }
    return true;
}

std::optional<uint32_t> FindModalObject(const ModalBank &b, entt::entity e) { return FindIndex(b.Entities, e); }

void EnqueueModalEvent(ModalAudio &m, const ModalEvent &e) {
    const auto write = m.EventWrite.load(std::memory_order_relaxed);
    if (write - m.EventRead.load(std::memory_order_acquire) >= ModalAudio::EventCapacity) {
        ++m.EventsDropped;
        return;
    }
    m.Events[write % ModalAudio::EventCapacity] = e;
    m.EventWrite.store(write + 1, std::memory_order_release);
}

namespace {
// Deal this block's ringing objects between the renderers, heaviest first onto whichever is carrying least.
// The deal is a pure function of the bank, so the same block always splits the same way however the renderers are run.
void DealObjects(ModalAudio &m, const ModalBank &b, uint32_t count) {
    m.Renderers.resize(count);
    for (auto &renderer : m.Renderers) renderer.Objects.clear();

    // A voice roughly doubles what an object costs, since the coupled kernel gathers over its modes every sample.
    auto &order = m.RenderOrderScratch;
    order.clear();
    for (uint32_t o = 0; o < uint32_t(b.Entities.size()); ++o) {
        // Only a ringing object has state, an impact, or a voice to render.
        if (!b.Ringing[o]) continue;
        uint64_t voices = 0;
        for (const auto vo : b.VoiceObject) voices += vo == o;
        // An excited object renders its whole tuned set rather than the audible prefix.
        const bool excited = voices > 0 || std::ranges::contains(b.ImpactObject, o);
        order.emplace_back(uint64_t(excited ? b.TunedModeCount[o] : b.LiveModeCount[o]) * (1 + voices), o);
    }
    // One renderer takes every object in bank order, which is what the deal below produces anyway.
    if (count == 1) {
        for (const auto &[cost, o] : order) m.Renderers.front().Objects.push_back(o);
        return;
    }
    // Heaviest first, and by object for equal weights, so the deal never depends on the sort's own tie-breaking.
    std::ranges::sort(order, [](const auto &a, const auto &c) { return a.first != c.first ? a.first > c.first : a.second < c.second; });
    auto &load = m.RenderLoadScratch;
    load.assign(count, 0);
    for (const auto &[cost, o] : order) {
        const auto least = uint32_t(std::ranges::distance(load.begin(), std::ranges::min_element(load)));
        load[least] += cost;
        m.Renderers[least].Objects.push_back(o);
    }
    // Each renderer takes its objects in bank order, which is the order a single renderer would use.
    for (auto &renderer : m.Renderers) std::ranges::sort(renderer.Objects);
}

// What one renderer needs to find its own work, handed to the pool as one pointer.
struct RenderJob {
    ModalAudio &Audio;
    ModalBank &Bank;
    uint32_t FrameCount;
};

// Render the objects this renderer was dealt into its own share of the mix.
void RenderObjects(ModalAudio &m, ModalAudio::RenderScratch &w, ModalBank &b, uint32_t frame_count) {
    const auto impact_count = uint32_t(b.ImpactObject.size());
    for (const auto o : w.Objects) {
        w.Impacts.clear();
        for (uint32_t i = 0; i < impact_count; ++i) {
            if (b.ImpactObject[i] == o) w.Impacts.push_back(i);
        }
        w.Voices.clear();
        for (uint32_t v = 0; v < uint32_t(b.VoiceId.size()); ++v) {
            if (b.VoiceObject[v] == o) w.Voices.push_back(v);
        }
        // An object with no sustained contact takes the hoisted-gain kernel.
        // One holding a voice takes the coupled kernel, which renders that object's impacts too.
        if (w.Voices.empty()) RenderObjectFast(m, w, b, o, w.Impacts, w.Out.data(), frame_count);
        else RenderObjectCoupled(m, w, b, o, w.Impacts, w.Voices, w.Out.data(), frame_count);
    }
}
} // namespace

void RenderModal(ModalAudio &m, float *out, uint32_t frame_count) {
    if (frame_count == 0) return;

    // Mark this callback's generation, then load the live bank. The main thread waits for this generation
    // to advance before freeing a replaced bank, so the pointer is always valid and the audio thread never blocks.
    const auto seq = m.ReaderSeq.load(std::memory_order_relaxed);
    m.ReaderSeq.store(seq + 1, std::memory_order_seq_cst);
    ModalBank &b = *m.Published.load(std::memory_order_seq_cst);
    // A newly published bank invalidates queued events that targeted the old slot layout.
    if (m.FlushEvents.exchange(false, std::memory_order_relaxed)) {
        m.EventRead.store(m.EventWrite.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    DrainEvents(m, b);
    AdoptVoices(m, b, frame_count);
    const auto click_gain = m.ClickGain.load(std::memory_order_relaxed);

    // Per-impact half-sine force curves for this block, plus the acceleration-noise click (the force derivative).
    const auto impact_count = uint32_t(b.ImpactObject.size());
    m.ForceScratch.resize(size_t(impact_count) * frame_count);
    for (uint32_t i = 0; i < impact_count; ++i) {
        float phase_re = b.ImpactPhaseRe[i], phase_im = b.ImpactPhaseIm[i];
        const auto rot_re = b.ImpactRotRe[i], rot_im = b.ImpactRotIm[i];
        const auto gamma = b.ImpactGamma[i];
        const auto click = b.ImpactAccelAmp[i] * click_gain;
        auto prev = b.ImpactPrevForce[i];
        auto left = b.ImpactSamplesLeft[i];
        auto *force = &m.ForceScratch[size_t(i) * frame_count];
        for (uint32_t s = 0; s < frame_count; ++s) {
            float cur{0.f};
            if (left > 0) {
                const float re = phase_re * rot_re - phase_im * rot_im;
                phase_im = phase_re * rot_im + phase_im * rot_re;
                phase_re = re;
                cur = gamma * phase_im;
                --left;
            }
            force[s] = cur;
            out[s] += (cur - prev) * click;
            prev = cur;
        }
        b.ImpactPhaseRe[i] = phase_re;
        b.ImpactPhaseIm[i] = phase_im;
        b.ImpactPrevForce[i] = prev;
        b.ImpactSamplesLeft[i] = left;
    }

    // One width for the whole block, so the deal below and the dispatch under it name the same renderers.
    ModalRenderPool::Session session{m.RenderPool};
    DealObjects(m, b, session.Width());
    for (auto &renderer : m.Renderers) renderer.Out.assign(frame_count, 0.f);
    RenderJob job{m, b, frame_count};
    session.Run(
        [](void *context, uint32_t i) {
            auto &j = *static_cast<RenderJob *>(context);
            RenderObjects(j.Audio, j.Audio.Renderers[i], j.Bank, j.FrameCount);
        },
        &job
    );
    // Each renderer's share is summed in renderer order, so the mix does not depend on which of them reached an object first.
    for (const auto &renderer : m.Renderers) {
        for (uint32_t s = 0; s < frame_count; ++s) out[s] += renderer.Out[s];
    }

    for (uint32_t i = uint32_t(b.ImpactObject.size()); i-- > 0;) {
        if (b.ImpactSamplesLeft[i] == 0) RemoveImpact(b, i);
    }
    m.ActiveVoices.store(uint32_t(b.VoiceId.size()), std::memory_order_relaxed);
    m.ActiveImpacts.store(uint32_t(b.ImpactObject.size()), std::memory_order_relaxed);
    m.ReaderSeq.store(seq + 2, std::memory_order_release);
}
