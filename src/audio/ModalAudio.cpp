#include "ModalAudio.h"
#include "ModalModes.h"

#include <entt/entity/entity.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace {
constexpr uint32_t MaxImpacts{1024};
// Modes are rendered in fixed-width lanes so the sample loop vectorizes across modes.
constexpr uint32_t Lanes{8};
// An object whose gain-weighted state energy falls below this (with no active impacts) is zeroed and skipped.
constexpr float SilentEnergy{1e-12f};

void RemoveImpact(ModalAudio &m, uint32_t i) {
    const auto swap_remove = [i](auto &v) {
        v[i] = v.back();
        v.pop_back();
    };
    swap_remove(m.ImpactObject);
    swap_remove(m.ImpactExPos);
    swap_remove(m.ImpactSamplesLeft);
    swap_remove(m.ImpactJx);
    swap_remove(m.ImpactJy);
    swap_remove(m.ImpactJz);
    swap_remove(m.ImpactPhaseRe);
    swap_remove(m.ImpactPhaseIm);
    swap_remove(m.ImpactRotRe);
    swap_remove(m.ImpactRotIm);
    swap_remove(m.ImpactGamma);
    swap_remove(m.ImpactAccelAmp);
    swap_remove(m.ImpactPrevForce);
}

void ActivateImpact(ModalAudio &m, const ModalEvent &e) {
    if (m.ImpactObject.size() >= MaxImpacts) return;
    m.ImpactObject.push_back(e.Object);
    m.ImpactExPos.push_back(e.ExPos);
    m.ImpactSamplesLeft.push_back(uint32_t(std::ceil(1.f / e.PulseStep)));
    m.ImpactJx.push_back(e.Jx);
    m.ImpactJy.push_back(e.Jy);
    m.ImpactJz.push_back(e.Jz);
    m.ImpactPhaseRe.push_back(1.f);
    m.ImpactPhaseIm.push_back(0.f);
    const float theta = std::numbers::pi_v<float> * e.PulseStep;
    m.ImpactRotRe.push_back(std::cos(theta));
    m.ImpactRotIm.push_back(std::sin(theta));
    m.ImpactGamma.push_back(e.PulseGamma);
    m.ImpactAccelAmp.push_back(e.AccelAmp);
    m.ImpactPrevForce.push_back(0.f);
    m.Ringing[e.Object] = 1;
}

void SilenceObject(ModalAudio &m, uint32_t o) {
    const uint32_t k0 = m.ModeOffset[o], count = m.ModeCount[o];
    std::fill_n(m.StateRe.begin() + k0, count, 0.f);
    std::fill_n(m.StateIm.begin() + k0, count, 0.f);
    m.Ringing[o] = 0;
    for (uint32_t i = uint32_t(m.ImpactObject.size()); i-- > 0;) {
        if (m.ImpactObject[i] == o) RemoveImpact(m, i);
    }
}

void DrainEvents(ModalAudio &m) {
    auto read = m.EventRead.load(std::memory_order_relaxed);
    const auto write = m.EventWrite.load(std::memory_order_acquire);
    for (; read != write; ++read) {
        const auto &e = m.Events[read % ModalAudio::EventCapacity];
        if (e.Object >= m.Entities.size()) continue;
        if (e.Kind == ModalEventKind::Impact) {
            if (e.PulseStep > 0) ActivateImpact(m, e);
        } else {
            SilenceObject(m, e.Object);
        }
    }
    m.EventRead.store(read, std::memory_order_release);
}
} // namespace

void ClearModalObjects(ModalAudio &m) {
    m.CoeffRe.clear();
    m.CoeffIm.clear();
    m.StateRe.clear();
    m.StateIm.clear();
    m.ShapeX.clear();
    m.ShapeY.clear();
    m.ShapeZ.clear();
    m.Entities.clear();
    m.ModeOffset.clear();
    m.ModeCount.clear();
    m.ShapeOffset.clear();
    m.OutGain.clear();
    m.Ringing.clear();
    m.ImpactObject.clear();
    m.ImpactExPos.clear();
    m.ImpactSamplesLeft.clear();
    m.ImpactJx.clear();
    m.ImpactJy.clear();
    m.ImpactJz.clear();
    m.ImpactPhaseRe.clear();
    m.ImpactPhaseIm.clear();
    m.ImpactRotRe.clear();
    m.ImpactRotIm.clear();
    m.ImpactGamma.clear();
    m.ImpactAccelAmp.clear();
    m.ImpactPrevForce.clear();
    // Pending events target slots in the old layout, so drop them.
    m.EventRead.store(m.EventWrite.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

uint32_t AddModalObject(ModalAudio &m, entt::entity e, const ModalModes &modes) {
    const auto count = uint32_t(modes.Freqs.size());
    const auto slot = uint32_t(m.Entities.size());
    m.Entities.push_back(e);
    m.ModeOffset.push_back(uint32_t(m.CoeffRe.size()));
    m.ModeCount.push_back(count);
    m.ShapeOffset.push_back(uint32_t(m.ShapeX.size()));
    m.OutGain.push_back(0.f);
    m.Ringing.push_back(0);
    m.CoeffRe.resize(m.CoeffRe.size() + count, 0.f);
    m.CoeffIm.resize(m.CoeffIm.size() + count, 0.f);
    m.StateRe.resize(m.StateRe.size() + count, 0.f);
    m.StateIm.resize(m.StateIm.size() + count, 0.f);
    for (const auto &row : modes.Shapes) {
        for (const auto &shape : row) {
            m.ShapeX.push_back(shape.x);
            m.ShapeY.push_back(shape.y);
            m.ShapeZ.push_back(shape.z);
        }
    }
    return slot;
}

void TuneModalObject(ModalAudio &m, uint32_t object, std::span<const float> freqs, std::span<const float> t60s) {
    const uint32_t k0 = m.ModeOffset[object];
    const uint32_t count = std::min(m.ModeCount[object], uint32_t(std::min(freqs.size(), t60s.size())));
    const float sr = m.SampleRate;
    for (uint32_t k = 0; k < count; ++k) {
        const float freq = freqs[k], t60 = t60s[k];
        if (freq <= 0.f || freq >= sr / 2 - 1 || t60 <= 0.f) {
            m.CoeffRe[k0 + k] = 0.f;
            m.CoeffIm[k0 + k] = 0.f;
            continue;
        }
        const float decay = std::pow(1e-3f, 1.f / (t60 * sr));
        const float omega = 2 * std::numbers::pi_v<float> * freq / sr;
        m.CoeffRe[k0 + k] = decay * std::cos(omega);
        m.CoeffIm[k0 + k] = decay * std::sin(omega);
    }
}

bool SetModalObjectShapes(ModalAudio &m, uint32_t object, const ModalModes &modes) {
    const auto begin = m.ShapeOffset[object];
    const auto end = object + 1 < m.ShapeOffset.size() ? m.ShapeOffset[object + 1] : uint32_t(m.ShapeX.size());
    const auto count = uint32_t(modes.Freqs.size());
    if (m.ModeCount[object] != count || end - begin != count * modes.Shapes.size()) return false;
    uint32_t i = begin;
    for (const auto &row : modes.Shapes) {
        for (const auto &shape : row) {
            m.ShapeX[i] = shape.x;
            m.ShapeY[i] = shape.y;
            m.ShapeZ[i] = shape.z;
            ++i;
        }
    }
    return true;
}

std::optional<uint32_t> FindModalObject(const ModalAudio &m, entt::entity e) {
    if (auto it = std::ranges::find(m.Entities, e); it != m.Entities.end()) {
        return uint32_t(std::ranges::distance(m.Entities.begin(), it));
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

    // Per-impact half-sine force curves for this block, plus the acceleration-noise click (the force derivative).
    const auto impact_count = uint32_t(m.ImpactObject.size());
    m.ForceScratch.resize(size_t(impact_count) * frame_count);
    for (uint32_t i = 0; i < impact_count; ++i) {
        float phase_re = m.ImpactPhaseRe[i], phase_im = m.ImpactPhaseIm[i];
        const float rot_re = m.ImpactRotRe[i], rot_im = m.ImpactRotIm[i];
        const float gamma = m.ImpactGamma[i];
        const float click = m.ImpactAccelAmp[i] * m.ClickGain;
        float prev = m.ImpactPrevForce[i];
        auto left = m.ImpactSamplesLeft[i];
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
        m.ImpactPhaseRe[i] = phase_re;
        m.ImpactPhaseIm[i] = phase_im;
        m.ImpactPrevForce[i] = prev;
        m.ImpactSamplesLeft[i] = left;
    }

    // Resonator banks. Modes advance in `Lanes`-wide chunks with all state in locals,
    // so the per-sample loop is branchless and vectorizes across the chunk.
    for (uint32_t o = 0; o < uint32_t(m.Entities.size()); ++o) {
        auto &obj_impacts = m.ObjectImpactScratch;
        obj_impacts.clear();
        for (uint32_t i = 0; i < impact_count; ++i) {
            if (m.ImpactObject[i] == o) obj_impacts.push_back(i);
        }
        if (!m.Ringing[o] && obj_impacts.empty()) continue;

        const uint32_t k0 = m.ModeOffset[o], count = m.ModeCount[o];
        const uint32_t shape0 = m.ShapeOffset[o];
        const float out_gain = m.OutGain[o];
        m.GainScratch.resize(obj_impacts.size() * Lanes);
        float energy = 0.f;
        for (uint32_t k = 0; k < count; k += Lanes) {
            const uint32_t width = std::min(Lanes, count - k);
            float z_re[Lanes]{}, z_im[Lanes]{}, c_re[Lanes]{}, c_im[Lanes]{};
            for (uint32_t l = 0; l < width; ++l) {
                z_re[l] = m.StateRe[k0 + k + l];
                z_im[l] = m.StateIm[k0 + k + l];
                c_re[l] = m.CoeffRe[k0 + k + l];
                c_im[l] = m.CoeffIm[k0 + k + l];
            }
            // Excitation gain of each impact for each mode in the chunk: the impulse projected onto the shape.
            for (size_t t = 0; t < obj_impacts.size(); ++t) {
                const auto i = obj_impacts[t];
                const uint32_t base = shape0 + m.ImpactExPos[i] * count + k;
                const float jx = m.ImpactJx[i], jy = m.ImpactJy[i], jz = m.ImpactJz[i];
                float *gain = &m.GainScratch[t * Lanes];
                for (uint32_t l = 0; l < Lanes; ++l) {
                    gain[l] = l < width ? m.ShapeX[base + l] * jx + m.ShapeY[base + l] * jy + m.ShapeZ[base + l] * jz : 0.f;
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
                m.StateRe[k0 + k + l] = z_re[l];
                m.StateIm[k0 + k + l] = z_im[l];
                energy += z_re[l] * z_re[l] + z_im[l] * z_im[l];
            }
        }
        if (obj_impacts.empty() && energy * out_gain * out_gain < SilentEnergy) SilenceObject(m, o);
        else m.Ringing[o] = 1;
    }

    for (uint32_t i = uint32_t(m.ImpactObject.size()); i-- > 0;) {
        if (m.ImpactSamplesLeft[i] == 0) RemoveImpact(m, i);
    }
}
