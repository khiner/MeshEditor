#include "ModalAudio.h"
#include "ModalModes.h"

#include <entt/entity/entity.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace {
// Modes are rendered in fixed-width lanes so the sample loop vectorizes across modes.
constexpr uint32_t Lanes{8};
// An object whose gain-weighted state energy falls below this (with no active impacts) is zeroed and skipped.
constexpr float SilentEnergy{1e-12f};

void RemoveImpact(ModalBank &b, uint32_t i) {
    const auto swap_remove = [i](auto &v) {
        v[i] = v.back();
        v.pop_back();
    };
    swap_remove(b.ImpactObject);
    swap_remove(b.ImpactExPos);
    swap_remove(b.ImpactSamplesLeft);
    swap_remove(b.ImpactJx);
    swap_remove(b.ImpactJy);
    swap_remove(b.ImpactJz);
    swap_remove(b.ImpactPhaseRe);
    swap_remove(b.ImpactPhaseIm);
    swap_remove(b.ImpactRotRe);
    swap_remove(b.ImpactRotIm);
    swap_remove(b.ImpactGamma);
    swap_remove(b.ImpactAccelAmp);
    swap_remove(b.ImpactPrevForce);
}

void ActivateImpact(ModalAudio &m, const ModalEvent &e) {
    auto &b = m.Bank;
    if (b.ImpactObject.size() >= m.MaxImpacts) return;
    b.ImpactObject.push_back(e.Object);
    b.ImpactExPos.push_back(e.ExPos);
    b.ImpactSamplesLeft.push_back(uint32_t(std::ceil(1.f / e.PulseStep)));
    b.ImpactJx.push_back(e.Jx);
    b.ImpactJy.push_back(e.Jy);
    b.ImpactJz.push_back(e.Jz);
    b.ImpactPhaseRe.push_back(1.f);
    b.ImpactPhaseIm.push_back(0.f);
    const float theta = std::numbers::pi_v<float> * e.PulseStep;
    b.ImpactRotRe.push_back(std::cos(theta));
    b.ImpactRotIm.push_back(std::sin(theta));
    b.ImpactGamma.push_back(e.PulseGamma);
    b.ImpactAccelAmp.push_back(e.AccelAmp);
    b.ImpactPrevForce.push_back(0.f);
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
}

void DrainEvents(ModalAudio &m) {
    auto read = m.EventRead.load(std::memory_order_relaxed);
    const auto write = m.EventWrite.load(std::memory_order_acquire);
    for (; read != write; ++read) {
        const auto &e = m.Events[read % ModalAudio::EventCapacity];
        if (e.Object >= m.Bank.Entities.size()) continue;
        if (e.Kind == ModalEventKind::Impact) {
            if (e.PulseStep > 0) ActivateImpact(m, e);
        } else {
            SilenceObject(m.Bank, e.Object);
        }
    }
    m.EventRead.store(read, std::memory_order_release);
}
} // namespace

void InstallModalBank(ModalAudio &m, ModalBank &next) {
    const std::scoped_lock lock{m.StructureMutex};
    std::swap(m.Bank, next);
    // Pending events target slots in the old layout, so drop them.
    m.EventRead.store(m.EventWrite.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

uint32_t AddModalObject(ModalBank &b, entt::entity e, const ModalModes &modes) {
    const auto count = uint32_t(modes.Freqs.size());
    const auto slot = uint32_t(b.Entities.size());
    b.Entities.push_back(e);
    b.ModeOffset.push_back(uint32_t(b.CoeffRe.size()));
    b.ModeCount.push_back(count);
    b.ShapeOffset.push_back(uint32_t(b.ShapeX.size()));
    b.OutGain.push_back(0.f);
    b.Ringing.push_back(0);
    b.CoeffRe.resize(b.CoeffRe.size() + count, 0.f);
    b.CoeffIm.resize(b.CoeffIm.size() + count, 0.f);
    b.StateRe.resize(b.StateRe.size() + count, 0.f);
    b.StateIm.resize(b.StateIm.size() + count, 0.f);
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
    const uint32_t k0 = b.ModeOffset[object];
    const uint32_t count = std::min(b.ModeCount[object], uint32_t(std::min(freqs.size(), t60s.size())));
    const float sr = b.SampleRate;
    for (uint32_t k = 0; k < count; ++k) {
        const float freq = freqs[k], t60 = t60s[k];
        if (freq <= 0.f || freq >= sr / 2 - 1 || t60 <= 0.f) {
            b.CoeffRe[k0 + k] = 0.f;
            b.CoeffIm[k0 + k] = 0.f;
            continue;
        }
        const float decay = std::pow(1e-3f, 1.f / (t60 * sr));
        const float omega = 2 * std::numbers::pi_v<float> * freq / sr;
        b.CoeffRe[k0 + k] = decay * std::cos(omega);
        b.CoeffIm[k0 + k] = decay * std::sin(omega);
    }
}

bool SetModalObjectShapes(ModalBank &b, uint32_t object, const ModalModes &modes) {
    const auto begin = b.ShapeOffset[object];
    const auto end = object + 1 < b.ShapeOffset.size() ? b.ShapeOffset[object + 1] : uint32_t(b.ShapeX.size());
    const auto count = uint32_t(modes.Freqs.size());
    if (b.ModeCount[object] != count || end - begin != count * modes.Shapes.size()) return false;
    uint32_t i = begin;
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

std::optional<uint32_t> FindModalObject(const ModalBank &b, entt::entity e) {
    if (auto it = std::ranges::find(b.Entities, e); it != b.Entities.end()) {
        return uint32_t(std::ranges::distance(b.Entities.begin(), it));
    }
    return {};
}

void EnqueueModalEvent(ModalAudio &m, const ModalEvent &e) {
    const auto write = m.EventWrite.load(std::memory_order_relaxed);
    if (write - m.EventRead.load(std::memory_order_acquire) >= ModalAudio::EventCapacity) return;
    m.Events[write % ModalAudio::EventCapacity] = e;
    m.EventWrite.store(write + 1, std::memory_order_release);
}

void RenderModal(ModalAudio &m, float *out, uint32_t frame_count) {
    if (frame_count == 0) return;
    const std::unique_lock lock{m.StructureMutex, std::try_to_lock};
    if (!lock.owns_lock()) return;
    DrainEvents(m);
    auto &b = m.Bank;

    // Per-impact half-sine force curves for this block, plus the acceleration-noise click (the force derivative).
    const auto impact_count = uint32_t(b.ImpactObject.size());
    m.ForceScratch.resize(size_t(impact_count) * frame_count);
    for (uint32_t i = 0; i < impact_count; ++i) {
        float phase_re = b.ImpactPhaseRe[i], phase_im = b.ImpactPhaseIm[i];
        const float rot_re = b.ImpactRotRe[i], rot_im = b.ImpactRotIm[i];
        const float gamma = b.ImpactGamma[i];
        const float click = b.ImpactAccelAmp[i] * m.ClickGain;
        float prev = b.ImpactPrevForce[i];
        auto left = b.ImpactSamplesLeft[i];
        float *force = &m.ForceScratch[size_t(i) * frame_count];
        for (uint32_t s = 0; s < frame_count; ++s) {
            float cur = 0.f;
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

    // Resonator banks. Modes advance in `Lanes`-wide chunks with all state in locals,
    // so the per-sample loop is branchless and vectorizes across the chunk.
    for (uint32_t o = 0; o < uint32_t(b.Entities.size()); ++o) {
        auto &obj_impacts = m.ObjectImpactScratch;
        obj_impacts.clear();
        for (uint32_t i = 0; i < impact_count; ++i) {
            if (b.ImpactObject[i] == o) obj_impacts.push_back(i);
        }
        if (!b.Ringing[o] && obj_impacts.empty()) continue;

        const uint32_t k0 = b.ModeOffset[o], count = b.ModeCount[o];
        const uint32_t shape0 = b.ShapeOffset[o];
        const float out_gain = b.OutGain[o];
        m.GainScratch.resize(obj_impacts.size() * Lanes);
        float energy = 0.f;
        for (uint32_t k = 0; k < count; k += Lanes) {
            const uint32_t width = std::min(Lanes, count - k);
            float z_re[Lanes]{}, z_im[Lanes]{}, c_re[Lanes]{}, c_im[Lanes]{};
            for (uint32_t l = 0; l < width; ++l) {
                z_re[l] = b.StateRe[k0 + k + l];
                z_im[l] = b.StateIm[k0 + k + l];
                c_re[l] = b.CoeffRe[k0 + k + l];
                c_im[l] = b.CoeffIm[k0 + k + l];
            }
            // Excitation gain of each impact for each mode in the chunk: the impulse projected onto the shape.
            for (size_t t = 0; t < obj_impacts.size(); ++t) {
                const auto i = obj_impacts[t];
                const uint32_t base = shape0 + b.ImpactExPos[i] * count + k;
                const float jx = b.ImpactJx[i], jy = b.ImpactJy[i], jz = b.ImpactJz[i];
                float *gain = &m.GainScratch[t * Lanes];
                for (uint32_t l = 0; l < Lanes; ++l) {
                    gain[l] = l < width ? b.ShapeX[base + l] * jx + b.ShapeY[base + l] * jy + b.ShapeZ[base + l] * jz : 0.f;
                }
            }
            for (uint32_t s = 0; s < frame_count; ++s) {
                float excite[Lanes]{};
                for (size_t t = 0; t < obj_impacts.size(); ++t) {
                    const float force = m.ForceScratch[size_t(obj_impacts[t]) * frame_count + s];
                    if (force == 0.f) continue;
                    const float *gain = &m.GainScratch[t * Lanes];
                    for (uint32_t l = 0; l < Lanes; ++l) excite[l] += force * gain[l];
                }
                float acc = 0.f;
                for (uint32_t l = 0; l < Lanes; ++l) {
                    const float re = z_re[l] * c_re[l] - z_im[l] * c_im[l] + excite[l];
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
        if (obj_impacts.empty() && energy * out_gain * out_gain < SilentEnergy) SilenceObject(b, o);
        else b.Ringing[o] = 1;
    }

    for (uint32_t i = uint32_t(b.ImpactObject.size()); i-- > 0;) {
        if (b.ImpactSamplesLeft[i] == 0) RemoveImpact(b, i);
    }
}
