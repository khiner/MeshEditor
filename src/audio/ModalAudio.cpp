#include "ModalAudio.h"
#include "ModalModes.h"

#include <entt/entity/entity.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>
#include <thread>

namespace {
// Modes are rendered in fixed-width lanes so the sample loop vectorizes across modes.
constexpr uint32_t Lanes{8};
// An object whose gain-weighted state energy falls below this (with no active impacts) is zeroed and skipped.
constexpr float SilentEnergy{1e-12f};
// Absolute level of the sustained-contact channel, which the spec leaves implementation-defined.
// A contact force in newtons sits orders of magnitude below a strike's peak force at the same impulse, so the two channels are scaled apart to share one mix.
constexpr float SustainReference{1e4f};
// Distance over which the relief's local mean is removed, m.
// Far longer than any wavelength the contact filter passes, so a settled contact is exactly silent.
constexpr float ReliefDcLength{1e-2f};
// A voice this long without a fresh contact report ends itself, so a dropped end event cannot leak one.
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
    SwapRemove(i, b.VoiceId, b.VoiceObject, b.VoiceState, b.VoiceCarry, b.VoiceIdleSamples);
}

void StartVoice(const ModalAudio &m, ModalBank &b, const ModalEvent &e) {
    if (b.VoiceId.size() >= m.MaxVoices.load(std::memory_order_relaxed)) return;
    b.VoiceId.push_back(e.ContactId);
    b.VoiceObject.push_back(e.Object);
    b.VoiceState.push_back(e.Contact);
    b.VoiceCarry.emplace_back();
    b.VoiceIdleSamples.push_back(0);
    b.Ringing[e.Object] = 1;
}

void SilenceObject(ModalBank &b, uint32_t o) {
    const uint32_t k0 = b.ModeOffset[o], count = b.ModeCount[o];
    std::fill_n(b.StateRe.begin() + k0, count, 0.f);
    std::fill_n(b.StateIm.begin() + k0, count, 0.f);
    b.Ringing[o] = 0;
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
        if (e.Kind == ModalEventKind::ContactEnd) {
            if (const auto v = FindIndex(b.VoiceId, e.ContactId)) RemoveVoice(b, *v);
            continue;
        }
        if (e.Object >= b.Entities.size()) continue;
        switch (e.Kind) {
            case ModalEventKind::Impact:
                if (e.PulseStep > 0) ActivateImpact(m, b, e);
                break;
            case ModalEventKind::Contact:
                // A voice the audio thread already dropped, at the cap or on going idle, reopens here.
                if (const auto v = FindIndex(b.VoiceId, e.ContactId)) {
                    b.VoiceObject[*v] = e.Object;
                    b.VoiceState[*v] = e.Contact;
                    b.VoiceIdleSamples[*v] = 0;
                    b.Ringing[e.Object] = 1;
                } else {
                    StartVoice(m, b, e);
                }
                break;
            case ModalEventKind::ContactEnd: break; // Handled above, before the object bounds check.
            case ModalEventKind::Silence:
                SilenceObject(b, e.Object);
                break;
        }
    }
    // Publish the slots the voices name before the read position, so a main thread that reads an event as consumed also sees the voice it started.
    uint64_t mask = 0;
    for (const auto &st : b.VoiceState) {
        for (const auto &t : st.Tracks) {
            if (t.Index >= 0) mask |= 1ull << uint32_t(t.Index);
        }
    }
    m.VoiceTrackMask.store(mask, std::memory_order_release);
    m.EventRead.store(read, std::memory_order_release);
}

// The impulse of an impact projected onto `n` mode shapes starting at mode `first`.
void ImpactGainRow(const ModalBank &b, uint32_t impact, uint32_t shape0, uint32_t count, uint32_t first, uint32_t n, float *out) {
    const auto base = shape0 + b.ImpactExPos[impact] * count + first;
    const auto jx = b.ImpactJx[impact], jy = b.ImpactJy[impact], jz = b.ImpactJz[impact];
    for (uint32_t i = 0; i < n; ++i) out[i] = b.ShapeX[base + i] * jx + b.ShapeY[base + i] * jy + b.ShapeZ[base + i] * jz;
}

// Modes advance in `Lanes`-wide chunks with all state in locals, so the per-sample loop is branchless and vectorizes across the chunk.
// Excitation gains are hoisted out of the sample loop, which is what makes this the fast path and what a coupled voice cannot do.
// Chunking also keeps the output sum to `Lanes` terms in registers, short enough that reassociating it buys nothing.
void RenderObjectFast(ModalAudio &m, ModalBank &b, uint32_t o, std::span<const uint32_t> impacts, float *out, uint32_t frame_count) {
    // A shape row is `stride` wide, and only the leading `count` of it still sounds.
    const auto k0 = b.ModeOffset[o], stride = b.ModeCount[o], count = b.LiveModeCount[o];
    const auto shape0 = b.ShapeOffset[o];
    const auto out_gain = std::atomic_ref{b.OutGain[o]}.load(std::memory_order_relaxed);
    m.GainScratch.resize(impacts.size() * Lanes);
    float energy = 0.f;
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
            auto *gain = &m.GainScratch[t * Lanes];
            ImpactGainRow(b, impacts[t], shape0, stride, k, width, gain);
            std::fill(gain + width, gain + Lanes, 0.f);
        }
        for (uint32_t s = 0; s < frame_count; ++s) {
            float excite[Lanes]{};
            for (size_t t = 0; t < impacts.size(); ++t) {
                const auto force = m.ForceScratch[size_t(impacts[t]) * frame_count + s];
                if (force == 0.f) continue;
                const auto *gain = &m.GainScratch[t * Lanes];
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
        for (uint32_t l = 0; l < width; ++l) {
            b.StateRe[k0 + k + l] = z_re[l];
            b.StateIm[k0 + k + l] = z_im[l];
            energy += z_re[l] * z_re[l] + z_im[l] * z_im[l];
        }
    }
    if (impacts.empty() && energy * out_gain * out_gain < SilentEnergy) SilenceObject(b, o);
    else b.Ringing[o] = 1;
}

// Constants of the tangential channel, from micro-collisions with the asperity slopes (KHR_audio_rigid_bodies Appendix B, after Agarwal et al. 2021).
// The exponent is 1, so the drive enters signed, which keeps the excitation zero-mean.
constexpr float TangentialGain{0.05f};

// The excitation a sustained voice contributes this sample: the force's fluctuation about the load along the normal, and the frictional force along the slip.
struct VoiceForce {
    float Normal, Tangential;
};

// tanh by its [5/4] Pade approximant, within a float ULP for |x| <= 1 and 7e-7 by |x| = 2, which covers the knee's working range.
// Past 4.9 tanh is already 1 to within 1e-4, so it saturates there rather than carrying the polynomial out of its accurate range.
float FastTanh(float x) {
    const float a = std::fabs(x);
    if (a > 4.9f) return std::copysign(1.f, x);
    const float x2 = x * x;
    const float n = x * (10395.f + x2 * (1260.f + x2 * 21.f));
    const float d = 10395.f + x2 * (4725.f + x2 * (210.f + x2));
    return std::clamp(n / d, -1.f, 1.f);
}

// One sample of the reference contact force model.
// `deflection` is the object's modal displacement along the normal at the previous sample, and that one-sample delay is the explicit discretization of the coupling, not an approximation of convenience.
VoiceForce StepVoice(const SustainedState &st, const RoughnessTrack *const *tracks, SustainedCarry &carry, float deflection, float sample_rate) {
    // Each track advances at its own surface's sweep speed, indexed by distance along the surface rather than by time.
    // A voice's first sample has no previous height to difference against, so it contributes no slope.
    const bool priming = !carry.Primed;
    carry.Primed = true;
    // Two surfaces are independent realizations even when their parameters agree, so their tracks start a quarter of a track apart and add in quadrature.
    if (priming) {
        for (uint32_t i = 0; i < SustainedState::TrackCount; ++i) carry.Pos[i] = double(i) * double(TrackSamples) / SustainedState::TrackCount;
    }

    float relief = 0, slope = 0, distance = 0;
    for (uint32_t i = 0; i < SustainedState::TrackCount; ++i) {
        const auto &t = st.Tracks[i];
        if (!tracks[i]) continue;
        carry.Pos[i] += t.Rate;
        const float height = t.Sigma * ReadTrack(*tracks[i], carry.Pos[i], t.Window);
        relief += height;
        if (t.Step > 0 && !priming) slope += (height - carry.PrevHeight[i]) / t.Step;
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
    // The clamp at zero is the separation nonlinearity, and is what produces micro-collisions and chatter.
    const float separation = std::max(rigid_approach - deflection, 0.f);
    const float separation_rate = (separation - carry.Penetration) * sample_rate;
    carry.Penetration = separation;

    // Hunt and Crossley: f_n = k * delta^(3/2) * (1 + c_d * delta_dot), and a contact never pulls.
    const float force = std::max(st.Stiffness * separation * std::sqrt(separation) * (1 + st.DampingCoeff * separation_rate), 0.f);
    // The excitation is the fluctuation about the load, soft-limited by a knee that scales with it so it stays within what the load sustains.
    float normal = force - st.NormalForce;
    if (st.NormalForce > 0) normal = st.NormalForce * FastTanh(normal / st.NormalForce);
    return {normal, TangentialGain * slope};
}

// The object's modal displacement along a voice's contact normal, from the state the previous sample left.
// The sum runs the length of the mode count, so it reassociates into independent partial sums rather than one dependency chain.
float ReadDeflection(const float *__restrict gains, const float *__restrict state_im, uint32_t count) {
#pragma clang fp reassociate(on)
    float sum = 0.f;
    for (uint32_t k = 0; k < count; ++k) sum += gains[k] * state_im[k];
    return sum;
}

// The two rows the mode advance folds in itself, which every coupled object has because it always holds a voice.
constexpr uint32_t FusedDrives{2};

// This sample's excitation of every mode from the drives past the fused pair, summed over those that carry force.
// A block whose object has no such drive leaves the buffer at the zero it was filled with.
void GatherExcitation(const float *__restrict gains, const float *__restrict forces, uint32_t voice_drives, uint32_t drives, float *__restrict excite, uint32_t count) {
    bool seeded = false;
    const auto row = [&](uint32_t d) { return gains + size_t(d) * count; };
    // A voice's normal and tangential rows both stand behind the same contact, so they enter together in one pass.
    for (uint32_t d = FusedDrives; d < voice_drives; d += 2) {
        const float fa = forces[d], fb = forces[d + 1];
        const float *__restrict ga = row(d), *__restrict gb = row(d + 1);
        if (seeded) {
            for (uint32_t k = 0; k < count; ++k) excite[k] += ga[k] * fa + gb[k] * fb;
        } else {
            for (uint32_t k = 0; k < count; ++k) excite[k] = ga[k] * fa + gb[k] * fb;
            seeded = true;
        }
    }
    // An impact stands on its own row and drops out once its pulse is over, which is most of the block it ends in.
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
// The first voice's two drive rows enter here rather than through the buffer, so a lone contact never writes an excitation it is about to read back.
// The state this leaves is what that voice reads its deflection from next sample, so that projection rides along too.
// Both sums run the length of the mode count, so they reassociate into independent partial sums rather than one dependency chain.
float AdvanceModes(
    float *__restrict state_re, float *__restrict state_im, const float *__restrict coeff_re, const float *__restrict coeff_im,
    const float *__restrict gains, const float *__restrict forces, const float *__restrict excite,
    const float *__restrict read_gains, float *__restrict deflection, uint32_t count
) {
#pragma clang fp reassociate(on)
    const float f0 = forces[0], f1 = forces[1];
    const float *__restrict g0 = gains, *__restrict g1 = gains + count;
    float acc = 0.f, read = 0.f;
    for (uint32_t k = 0; k < count; ++k) {
        const auto excitation = g0[k] * f0 + g1[k] * f1 + excite[k];
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
// The voice's read-out and write-in both go through the mode bank at the same sample, so the loop runs sample-outer, consuming the object's impacts along the way.
// A sample reads each voice's deflection back, steps its force model, gathers the excitation, then advances the modes.
// Each of those is a flat pass over the modes rather than one nested loop, which is what lets them vectorize.
void RenderObjectCoupled(
    ModalAudio &m, ModalBank &b, uint32_t o,
    std::span<const uint32_t> impacts, std::span<const uint32_t> voices,
    float *out, uint32_t frame_count
) {
    // A shape row is `stride` wide, and only the leading `count` of it still sounds.
    const auto k0 = b.ModeOffset[o], stride = b.ModeCount[o], count = b.LiveModeCount[o];
    const auto shape0 = b.ShapeOffset[o];
    const auto out_gain = std::atomic_ref{b.OutGain[o]}.load(std::memory_order_relaxed);
    const auto sample_rate = b.SampleRate;
    const auto coupling = m.Coupling.load(std::memory_order_relaxed);
    // The impact path drives the bank with force * dt worth of impulse per sample, so a sustained force enters the same way.
    const auto sustain_level = SustainReference * m.SustainLevel.load(std::memory_order_relaxed) / sample_rate;

    const uint32_t drives = uint32_t(voices.size()) * 2 + uint32_t(impacts.size());
    m.CoupledDriveGainScratch.resize(size_t(drives) * count);
    m.CoupledReadGainScratch.resize(voices.size() * count);
    m.CoupledForceScratch.resize(drives);
    m.CoupledExciteScratch.resize(count);
    const auto drive_row = [&](size_t i) { return &m.CoupledDriveGainScratch[i * count]; };
    // Voice state is fixed for the block, so each voice's surface tracks resolve once here.
    // The load is ordered against the generation store above, so a main thread that reads an ended generation knows this callback holds no track it is about to free.
    m.CoupledTrackScratch.resize(voices.size() * SustainedState::TrackCount);
    for (size_t t = 0; t < voices.size(); ++t) {
        const auto &st = b.VoiceState[voices[t]];
        for (uint32_t i = 0; i < SustainedState::TrackCount; ++i) {
            const auto index = st.Tracks[i].Index;
            m.CoupledTrackScratch[t * SustainedState::TrackCount + i] =
                index < 0 ? nullptr : m.SurfaceTracks[uint32_t(index)].Live.load(std::memory_order_seq_cst);
        }
        auto *gain_n = drive_row(t * 2), *gain_t = drive_row(t * 2 + 1);
        auto *gain_read = &m.CoupledReadGainScratch[t * count];
        const auto base1 = shape0 + st.Blend.First * stride, base2 = shape0 + st.Blend.Second * stride;
        const float w1 = st.Blend.Weight, w2 = 1.f - st.Blend.Weight;
        for (uint32_t k = 0; k < count; ++k) {
            const vec3 shape{
                w1 * b.ShapeX[base1 + k] + w2 * b.ShapeX[base2 + k],
                w1 * b.ShapeY[base1 + k] + w2 * b.ShapeY[base2 + k],
                w1 * b.ShapeZ[base1 + k] + w2 * b.ShapeZ[base2 + k],
            };
            gain_n[k] = glm::dot(shape, st.N);
            // Projecting onto the slip velocity rather than its direction carries the slip speed the frictional channel scales with.
            gain_t[k] = glm::dot(shape, st.Slip);
            // The separation is modulated by the object's own vibration along the same normal.
            gain_read[k] = coupling * gain_n[k] * b.DisplacementScale[k0 + k];
        }
    }
    for (size_t t = 0; t < impacts.size(); ++t) {
        ImpactGainRow(b, impacts[t], shape0, stride, 0, count, drive_row(voices.size() * 2 + t));
    }

    auto *state_re = &b.StateRe[k0];
    auto *state_im = &b.StateIm[k0];
    const auto *coeff_re = &b.CoeffRe[k0];
    const auto *coeff_im = &b.CoeffIm[k0];
    auto *gains = m.CoupledDriveGainScratch.data();
    auto *forces = m.CoupledForceScratch.data();
    auto *excite = m.CoupledExciteScratch.data();
    // The first voice's read-out comes out of the advance below, so it is seeded once here and refreshed there.
    auto *read_gains = m.CoupledReadGainScratch.data();
    float first_deflection = ReadDeflection(read_gains, state_im, count);
    // With nothing past the fused pair the buffer is never written, so it holds this zero for the whole block.
    const bool gather = drives > FusedDrives;
    if (!gather) std::fill_n(excite, count, 0.f);
    for (uint32_t s = 0; s < frame_count; ++s) {
        for (size_t t = 0; t < voices.size(); ++t) {
            const auto v = voices[t];
            const auto deflection = t == 0 ? first_deflection : ReadDeflection(read_gains + t * count, state_im, count);
            const auto f = StepVoice(b.VoiceState[v], &m.CoupledTrackScratch[t * SustainedState::TrackCount], b.VoiceCarry[v], deflection, sample_rate);
            forces[t * 2] = sustain_level * f.Normal;
            forces[t * 2 + 1] = sustain_level * f.Tangential;
        }
        for (size_t t = 0; t < impacts.size(); ++t) {
            forces[voices.size() * 2 + t] = m.ForceScratch[size_t(impacts[t]) * frame_count + s];
        }
        if (gather) GatherExcitation(gains, forces, uint32_t(voices.size()) * 2, drives, excite, count);
        out[s] += AdvanceModes(state_re, state_im, coeff_re, coeff_im, gains, forces, excite, read_gains, &first_deflection, count) * out_gain;
    }
    b.Ringing[o] = 1;
}
} // namespace

ModalAudio::ModalAudio() : Live{std::make_unique<ModalBank>()}, Published{Live.get()} {}

void InstallModalBank(ModalAudio &m, ModalBank &next) {
    const auto old = std::move(m.Live);
    m.Live = std::make_unique<ModalBank>(std::move(next));
    m.FlushEvents.store(true, std::memory_order_relaxed); // Before Published so the adopting callback observes it.
    m.Published.store(m.Live.get(), std::memory_order_seq_cst);
    // Wait out a callback still rendering the old bank, then free `old` here on the main thread.
    // Pending events targeted the old layout. The audio thread drops them when it adopts the new bank.
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
}

uint32_t AddModalObject(ModalBank &b, entt::entity e, const ModalModes &modes) {
    const auto count = uint32_t(modes.Freqs.size());
    const auto slot = uint32_t(b.Entities.size());
    b.Entities.push_back(e);
    b.ModeOffset.push_back(uint32_t(b.CoeffRe.size()));
    b.ModeCount.push_back(count);
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
        // The bank drives Re(z) with the excitation, so its state carries the displacement scaled by the sample period over the angular frequency.
        b.DisplacementScale[k0 + k] = 1.f / (sr * 2 * std::numbers::pi_v<float> * freq);
    }
    // Written after the coefficients, so a callback that sees a longer bank also sees what fills it.
    // Only the trailing muted block is dropped, since a muted mode in the middle still has live modes behind it.
    uint32_t live = b.ModeCount[object];
    while (live > 0 && b.CoeffRe[k0 + live - 1] == 0.f && b.CoeffIm[k0 + live - 1] == 0.f) --live;
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

    // An object with no sustained contact takes the hoisted-gain kernel, and one holding a voice takes the coupled kernel, which also consumes that object's impacts.
    for (uint32_t o = 0; o < uint32_t(b.Entities.size()); ++o) {
        auto &obj_impacts = m.ObjectImpactScratch;
        obj_impacts.clear();
        for (uint32_t i = 0; i < impact_count; ++i) {
            if (b.ImpactObject[i] == o) obj_impacts.push_back(i);
        }
        auto &obj_voices = m.ObjectVoiceScratch;
        obj_voices.clear();
        for (uint32_t v = 0; v < uint32_t(b.VoiceId.size()); ++v) {
            if (b.VoiceObject[v] == o) obj_voices.push_back(v);
        }
        if (!b.Ringing[o] && obj_impacts.empty() && obj_voices.empty()) continue;

        if (obj_voices.empty()) RenderObjectFast(m, b, o, obj_impacts, out, frame_count);
        else RenderObjectCoupled(m, b, o, obj_impacts, obj_voices, out, frame_count);
    }

    for (uint32_t i = uint32_t(b.ImpactObject.size()); i-- > 0;) {
        if (b.ImpactSamplesLeft[i] == 0) RemoveImpact(b, i);
    }
    const auto max_idle = uint32_t(b.SampleRate * MaxVoiceIdleSeconds);
    for (uint32_t v = uint32_t(b.VoiceId.size()); v-- > 0;) {
        b.VoiceIdleSamples[v] += frame_count;
        if (b.VoiceIdleSamples[v] > max_idle) RemoveVoice(b, v);
    }
    m.ActiveVoices.store(uint32_t(b.VoiceId.size()), std::memory_order_relaxed);
    m.ActiveImpacts.store(uint32_t(b.ImpactObject.size()), std::memory_order_relaxed);
    m.ReaderSeq.store(seq + 2, std::memory_order_release);
}
