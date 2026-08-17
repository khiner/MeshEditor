#include "ModalAudio.h"
#include "ModalModes.h"

#include <entt/entity/entity.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <span>
#include <thread>

#ifdef __APPLE__
#include <os/workgroup.h>
#endif

namespace {
// An object whose gain-weighted state energy falls below this (with no active impacts) is zeroed and skipped.
constexpr float SilentEnergy{1e-12f};

// Drop impact `i`, swapping the last into its place.
void RemoveImpact(ModalBank &b, uint32_t i) {
    b.Impacts[i] = b.Impacts.back();
    b.Impacts.pop_back();
}

void ActivateImpact(const ModalAudio &m, ModalBank &b, const ModalEvent &e) {
    if (b.Impacts.size() >= m.MaxImpacts.load(std::memory_order_relaxed)) return;
    const auto theta = 2 * std::numbers::pi_v<float> * e.PulseStep;
    b.Impacts.push_back({
        .Object = e.Object,
        .ExPos = e.ExPos,
        .SamplesLeft = uint32_t(std::ceil(1.f / e.PulseStep)),
        .Jx = e.Jx,
        .Jy = e.Jy,
        .Jz = e.Jz,
        .PhaseRe = 1.f,
        .PhaseIm = 0.f,
        .RotRe = std::cos(theta),
        .RotIm = std::sin(theta),
        .Gamma = e.PulseGamma,
        .AccelAmp = e.AccelAmp,
        .ClickB0 = e.ClickB0,
        .ClickA1 = e.ClickA1,
        .ClickA2 = e.ClickA2,
        .ClickZ1 = 0.f,
        .ClickZ2 = 0.f,
    });
    b.Ringing[e.Object] = 1;
}

void SilenceObject(ModalAudio &m, ModalBank &b, uint32_t o) {
    const uint32_t k0 = b.ModeOffset[o], count = b.ModeCount[o];
    std::fill_n(b.StateRe.begin() + k0, count, 0.f);
    std::fill_n(b.StateIm.begin() + k0, count, 0.f);
    b.Ringing[o] = 0;
    // The next strike starts from the whole tuned set again.
    b.LiveModeCount[o] = b.TunedModeCount[o];
    for (uint32_t i = uint32_t(b.Impacts.size()); i-- > 0;) {
        if (b.Impacts[i].Object == o) RemoveImpact(b, i);
    }
    SurfaceSilenceObject(m, o);
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
                SilenceObject(m, b, e.Object);
                break;
        }
    }
    m.EventRead.store(read, std::memory_order_release);
}

// Modes advance in `Lanes`-wide chunks with all state in locals, so the per-sample loop is branchless and vectorizes across the chunk.
// Excitation gains hoist out of the sample loop, which a coupled voice cannot do.
void RenderObjectFast(ModalAudio &m, ModalRenderScratch &w, ModalBank &b, uint32_t o, std::span<const uint32_t> impacts, float *out, uint32_t frame_count) {
    // A shape row is `stride` wide, and only the leading `count` of it still sounds.
    // An impact drives every mode the tuning left, so being struck restores the whole set.
    const auto k0 = b.ModeOffset[o], stride = b.ModeCount[o];
    const auto count = impacts.empty() ? b.LiveModeCount[o] : b.TunedModeCount[o];
    const auto shape0 = b.ShapeOffset[o];
    const auto out_gain = std::atomic_ref{b.OutGain[o]}.load(std::memory_order_relaxed);
    // The listener attenuation scales the output alone, audibility culling staying on the 1 m level, so the bank's live set does not depend on where the camera sits.
    const auto mix_gain = out_gain * std::atomic_ref{b.ListenerGain[o]}.load(std::memory_order_relaxed);
    w.Gains.resize(impacts.size() * Lanes);
    float energy = 0.f;
    // The end of the last chunk still carrying audible state.
    uint32_t live = 0;
    for (uint32_t k = 0; k < count; k += Lanes) {
        const auto width = std::min(Lanes, count - k);
        float z_re[Lanes]{}, z_im[Lanes]{}, c_re[Lanes]{}, c_im[Lanes]{}, p_re[Lanes]{}, p_im[Lanes]{};
        for (uint32_t l = 0; l < width; ++l) {
            z_re[l] = b.StateRe[k0 + k + l];
            z_im[l] = b.StateIm[k0 + k + l];
            c_re[l] = b.CoeffRe[k0 + k + l];
            c_im[l] = b.CoeffIm[k0 + k + l];
            p_im[l] = b.OutPhaseIm[k0 + k + l];
            p_re[l] = b.OutPhaseRe[k0 + k + l];
        }
        for (size_t t = 0; t < impacts.size(); ++t) {
            auto *gain = &w.Gains[t * Lanes];
            ImpactGainRow(b, impacts[t], shape0, stride, k0, k, width, gain);
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
                acc += p_im[l] * z_im[l] + p_re[l] * re;
            }
            out[s] += acc * mix_gain;
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
        SilenceObject(m, b, o);
        return;
    }
    b.Ringing[o] = 1;
    b.LiveModeCount[o] = impacts.empty() ? live : b.TunedModeCount[o];
}

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

// Holds a render thread in the host's scheduling group for as long as it lives, so the scheduler holds it to the same deadline as the callback it renders alongside.
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

ModalAudio::ModalAudio() : Live{std::make_unique<ModalBank>()}, Published{Live.get()}, Surface{MakeSurfaceAudioState()} {}

void InstallModalBank(ModalAudio &m, ModalBank &next) {
    const auto old = std::move(m.Live);
    m.Live = std::make_unique<ModalBank>(std::move(next));
    m.FlushEvents.store(true, std::memory_order_relaxed); // Before Published so the adopting callback observes it.
    m.Published.store(m.Live.get(), std::memory_order_seq_cst);
    // Wait out a callback still rendering the old bank, so `old` is freed here on the main thread.
    if (const auto seq = m.ReaderSeq.load(std::memory_order_seq_cst); seq & 1) {
        while (m.ReaderSeq.load(std::memory_order_seq_cst) == seq) std::this_thread::yield();
    }
    // The new bank holds no sustained contacts, and the wait above leaves no callback reading their pools.
    m.ActiveVoices.store(0, std::memory_order_relaxed);
    SurfaceInstallBank(m);
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
    b.Ringing.push_back(0);
    b.RigidVel.emplace_back(0.f);
    // Per-object columns, by what an untuned object stands at.
    for (auto *col : {&b.OutGain, &b.RigidInvMass, &b.RadiatorB0, &b.AirB0, &b.AirB1, &b.AirB2, &b.RecoilA1, &b.RecoilA2, &b.RadiatorZ1, &b.RadiatorZ2, &b.AirZ1, &b.AirZ2}) col->push_back(0.f);
    for (auto *col : {&b.ListenerGain, &b.DeflectionScale}) col->push_back(1.f);
    // Per-mode columns, which the object's own modes extend.
    for (auto *col : {&b.CoeffRe, &b.CoeffIm, &b.StateRe, &b.StateIm, &b.RadiationGain, &b.DeflectionGain, &b.QuadCompliance, &b.QuadDriveScale}) col->resize(col->size() + count, 0.f);
    b.OutPhaseIm.resize(b.OutPhaseIm.size() + count, 1.f);
    b.OutPhaseRe.resize(b.OutPhaseRe.size() + count, 0.f);
    for (const auto &row : modes.Shapes) {
        for (const auto &shape : row) {
            b.ShapeX.push_back(shape.x);
            b.ShapeY.push_back(shape.y);
            b.ShapeZ.push_back(shape.z);
        }
    }
    // Each mode's radiating strength is the surface integral of its squared normal shape, taken by centroid quadrature over the sample surface.
    // The square makes the winding irrelevant.
    // A model without a sample surface has no radiating area, and its modes stay silent.
    const auto area_offset = b.RadiationGain.size() - count;
    float total_area = 0.f;
    b.RadiationArea.resize(area_offset + count, 0.f);
    for (size_t t = 0; t + 2 < modes.Indices.size(); t += 3) {
        const auto i = modes.Indices[t], j = modes.Indices[t + 1], l = modes.Indices[t + 2];
        const vec3 cr = glm::cross(modes.Positions[j] - modes.Positions[i], modes.Positions[l] - modes.Positions[i]);
        const float doubled = glm::length(cr);
        if (doubled <= 0.f) continue;
        const vec3 n = cr / doubled;
        const float area = doubled / 2;
        total_area += area;
        for (uint32_t k = 0; k < count; ++k) {
            const vec3 shape = (modes.Shapes[i][k] + modes.Shapes[j][k] + modes.Shapes[l][k]) / 3.f;
            const float normal = glm::dot(shape, n);
            b.RadiationArea[area_offset + k] += area * normal * normal;
        }
    }
    b.RadiantRadius.push_back(std::sqrt(total_area / (4 * std::numbers::pi_v<float>)));
    return slot;
}

void TuneModalObject(ModalBank &b, uint32_t object, std::span<const float> freqs, std::span<const float> t60s, float radius_scale) {
    const auto k0 = b.ModeOffset[object];
    const auto count = std::min(b.ModeCount[object], uint32_t(std::min(freqs.size(), t60s.size())));
    const float sr = b.SampleRate;
    const float radius = b.RadiantRadius[object] * radius_scale;
    // The bank's shapes stay at the baked size, so the deflection loop sizes its two shape factors here.
    b.DeflectionScale[object] = 1.f / (radius_scale * radius_scale * radius_scale);
    for (uint32_t k = 0; k < count; ++k) {
        const float freq = freqs[k], t60 = t60s[k];
        if (!std::isfinite(freq) || !std::isfinite(t60) || freq <= 0.f || freq >= sr / 2 - 1 || t60 <= 0.f) {
            b.CoeffRe[k0 + k] = 0.f;
            b.CoeffIm[k0 + k] = 0.f;
            b.RadiationGain[k0 + k] = 0.f;
            b.DeflectionGain[k0 + k] = 0.f;
            b.OutPhaseIm[k0 + k] = 1.f;
            b.OutPhaseRe[k0 + k] = 0.f;
            b.QuadCompliance[k0 + k] = 0.f;
            b.QuadDriveScale[k0 + k] = 0.f;
            continue;
        }
        const auto omega = 2 * std::numbers::pi_v<float> * freq / sr;
        // Every drive enters as the impulse one sample carries, so the state holds mass-normalized modal velocity times this gain.
        // The gain is the mode's far-field pressure per unit of that velocity, rho0*c0*sqrt(sigma*A/(4pi))/r, with radiation efficiency sigma = (ka)^2/(1+(ka)^2) about the body's radiant radius, so the sum over modes leaves the output in Pa at the listener distance.
        const float omega_si = 2 * std::numbers::pi_v<float> * freq;
        const float ka = omega_si * radius / SpeedOfSound;
        const float sigma = ka * ka / (1 + ka * ka);
        // The radiated power drains the mode alongside its structural loss, so the energy leaving as sound is bounded by the energy the mode holds.
        // The surface-intensity balance gives the amplitude rate rho0*c0*sigma*A/2, with the baked-size area brought to the world size, the squared shape carrying scale^-3 and the surface scale^2.
        const float area = b.RadiationArea[k0 + k] / radius_scale;
        const float radiation_rate = AirDensity * SpeedOfSound * sigma * area * 0.5f;
        const auto decay = std::exp(-(Ln1000 / t60 + radiation_rate) / sr);
        b.CoeffRe[k0 + k] = decay * std::cos(omega);
        b.CoeffIm[k0 + k] = decay * std::sin(omega);
        const float gain = AirDensity * SpeedOfSound * std::sqrt(sigma * b.RadiationArea[k0 + k] / (4 * std::numbers::pi_v<float>)) / ListenerDistance;
        b.RadiationGain[k0 + k] = gain;
        // The golden-ratio sequence spreads the phases evenly and deterministically over the circle.
        const float spread = sigma * std::numbers::pi_v<float> * (2.f * std::fmod(0.6180339887f * float(k + 1), 1.0f) - 1.f);
        b.OutPhaseIm[k0 + k] = std::cos(spread);
        b.OutPhaseRe[k0 + k] = std::sin(spread);
        b.DeflectionGain[k0 + k] = gain > 0.f ? 1.f / (gain * omega_si) : 0.f;
        // The central scheme's full-step response, and the drive scale that makes the bank deliver it, the bank's own full-step response being dt * decay * sin(theta) / omega.
        const float dt = 1.f / sr;
        const float central = dt * (1 + decay * decay + 2 * decay * std::cos(omega)) / 4;
        b.QuadCompliance[k0 + k] = central;
        b.QuadDriveScale[k0 + k] = central * omega_si / (decay * std::sin(omega));
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

std::optional<uint32_t> FindModalObject(const ModalBank &b, entt::entity e) {
    const auto it = std::ranges::find(b.Entities, e);
    return it != b.Entities.end() ? std::optional{uint32_t(std::ranges::distance(b.Entities.begin(), it))} : std::nullopt;
}

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

    // A sustained contact roughly doubles what an object costs, since the coupled kernel gathers over its modes every sample.
    auto &order = m.RenderOrderScratch;
    order.clear();
    for (uint32_t o = 0; o < uint32_t(b.Entities.size()); ++o) {
        // Only a ringing object has state, an impact, or a sustained contact to render.
        if (!b.Ringing[o]) continue;
        const uint64_t voices = SurfaceVoiceCount(m, o);
        // An excited object renders its whole tuned set rather than the audible prefix.
        const bool excited = voices > 0 || std::ranges::contains(b.Impacts, o, &ModalBank::ActiveImpact::Object);
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

// Render the objects assigned to this renderer into its own share of the mix.
void RenderObjects(ModalAudio &m, ModalRenderScratch &w, ModalBank &b, uint32_t frame_count) {
    const auto impact_count = uint32_t(b.Impacts.size());
    for (const auto o : w.Objects) {
        w.Impacts.clear();
        for (uint32_t i = 0; i < impact_count; ++i) {
            if (b.Impacts[i].Object == o) w.Impacts.push_back(i);
        }
        // An object holding a sustained contact takes the surface model's coupled kernel, which renders that object's impacts too, and anything else takes the hoisted-gain kernel here.
        if (!SurfaceRenderObject(m, w, b, o, w.Impacts, w.Out.data(), frame_count)) {
            RenderObjectFast(m, w, b, o, w.Impacts, w.Out.data(), frame_count);
        }
    }
}
} // namespace

void RenderModal(ModalAudio &m, float *out, uint32_t frame_count) {
    if (frame_count == 0) return;

    const auto render_start = std::chrono::steady_clock::now();
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
    SurfaceAdoptVoices(m, b, frame_count);
    const auto click_gain = m.ClickGain.load(std::memory_order_relaxed);

    // Per-impact raised-cosine force curves for this block, plus the acceleration-noise click, the coupled recoil filter driven by the pulse as a force in Newtons (see ActiveImpact::ClickB0).
    const auto impact_count = uint32_t(b.Impacts.size());
    m.ForceScratch.resize(size_t(impact_count) * frame_count);
    for (uint32_t i = 0; i < impact_count; ++i) {
        auto &im = b.Impacts[i];
        float phase_re = im.PhaseRe, phase_im = im.PhaseIm;
        const auto rot_re = im.RotRe, rot_im = im.RotIm;
        const auto gamma = im.Gamma;
        const auto amp = im.AccelAmp;
        const auto b0 = im.ClickB0, a1 = im.ClickA1, a2 = im.ClickA2;
        const auto impact_click_gain = click_gain * std::atomic_ref{b.ListenerGain[im.Object]}.load(std::memory_order_relaxed);
        auto z1 = im.ClickZ1, z2 = im.ClickZ2;
        auto left = im.SamplesLeft;
        auto *force = &m.ForceScratch[size_t(i) * frame_count];
        for (uint32_t s = 0; s < frame_count; ++s) {
            float cur{0.f};
            if (left > 0) {
                const float re = phase_re * rot_re - phase_im * rot_im;
                phase_im = phase_re * rot_im + phase_im * rot_re;
                phase_re = re;
                cur = gamma * 0.5f * (1.f - phase_re);
                --left;
            }
            force[s] = cur;
            const float u = amp * cur;
            const float y = b0 * u + z1;
            z1 = -a1 * y + z2;
            z2 = -b0 * u - a2 * y;
            out[s] += y * impact_click_gain;
        }
        im.PhaseRe = phase_re;
        im.PhaseIm = phase_im;
        im.SamplesLeft = left;
        im.ClickZ1 = z1;
        im.ClickZ2 = z2;
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

    for (uint32_t i = uint32_t(b.Impacts.size()); i-- > 0;) {
        // A finished pulse keeps its impact until the click's filter drains, so the click ends without a step.
        const auto &im = b.Impacts[i];
        if (im.SamplesLeft == 0 && std::abs(im.ClickZ1) + std::abs(im.ClickZ2) < 1e-12f) RemoveImpact(b, i);
    }
    // The energy standing in the banks, read back through each mode's own radiation gain so a state in pressure units returns the mechanical energy behind it.
    // A peak that climbs with nothing striking means a channel is feeding the modes rather than damping them.
    double energy = 0;
    for (size_t o = 0; o < b.Entities.size(); ++o) {
        // A silenced object's state was zeroed with its ringing flag, so it contributes nothing here.
        if (!b.Ringing[o]) continue;
        const auto k0 = b.ModeOffset[o], count = b.TunedModeCount[o];
        for (uint32_t k = 0; k < count; ++k) {
            const auto g = b.RadiationGain[k0 + k];
            if (g > 0) energy += 0.5 * (double(b.StateRe[k0 + k]) * b.StateRe[k0 + k] + double(b.StateIm[k0 + k]) * b.StateIm[k0 + k]) / (double(g) * g);
        }
    }
    m.ModalEnergy.store(energy, std::memory_order_relaxed);
    if (double seen = m.PeakModalEnergy.load(std::memory_order_relaxed); energy > seen) {
        m.PeakModalEnergy.store(energy, std::memory_order_relaxed);
    }
    m.ActiveVoices.store(SurfaceActiveVoices(m), std::memory_order_relaxed);
    m.ActiveImpacts.store(uint32_t(b.Impacts.size()), std::memory_order_relaxed);
    m.ReaderSeq.store(seq + 2, std::memory_order_release);

    // What the block cost against the time it had. Past one the device underruns.
    const float seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - render_start).count();
    const float share = b.SampleRate > 0 ? seconds * b.SampleRate / float(frame_count) : 0.f;
    m.RenderSeconds.store(seconds, std::memory_order_relaxed);
    m.RenderShare.store(share, std::memory_order_relaxed);
    if (const float seen = m.PeakRenderShare.load(std::memory_order_relaxed); share > seen) {
        m.PeakRenderShare.store(share, std::memory_order_relaxed);
    }
}
