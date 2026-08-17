#pragma once

#include "SurfaceContact.h"
#include "numeric/vec3.h"

#include <entt/entity/fwd.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <span>
#include <thread>
#include <vector>

struct ModalModes;

// One-shot events the main thread queues for the audio thread.
enum class ModalEventKind : uint32_t {
    Impact, // Start a raised-cosine contact-force pulse on an object.
    Silence, // Clear an object's ringing state and drop its active pulses.
};

// One queued modal synthesis event.
struct ModalEvent {
    ModalEventKind Kind{ModalEventKind::Impact};
    uint32_t Object{0}; // Object slot in the bank
    uint32_t ExPos{0}; // Excitation position index
    float Jx{0}, Jy{0}, Jz{0}; // Node-local impulse vector
    float PulseStep{0}; // Per-sample phase increment of the contact pulse
    float PulseGamma{0}; // Contact pulse amplitude
    float AccelAmp{0}; // Scales the unit-sum pulse to the click filter's input force, N
    float ClickB0{0}, ClickA1{0}, ClickA2{0}; // The click's coupled recoil filter (see ModalBank::ActiveImpact::ClickB0)
};

// The acoustic medium and the output's calibration.
// Output samples are far-field pressure in Pa at ListenerDistance from each object, on axis, summed over the scene's objects.
constexpr float AirDensity{1.204f}; // kg/m^3
constexpr float SpeedOfSound{343.f}; // m/s
constexpr float ListenerDistance{1.f}; // m

// A T60 is the time to fall 60 dB, so the decay rate it states is this over it.
constexpr float Ln1000 = 3 * std::numbers::ln10_v<float>;

// The rigid-recoil radiation filters, bilinear at the bank rate, from the exact translating-sphere solution at the body's volume-equivalent radius (Rienstra and Hirschberg Eq. 6.19-6.20).
// Both share the denominator D(s) = s^2 + 2*wc*s + 2*wc^2 at the corner wc = c0/a.
// Below the corner the radiator is the compact-dipole jerk law 3*rho0*V/(8*pi*c0*r) (Qu and James 2019), and above it the pressure saturates at rho0*c0*(a/r) times the velocity, which bounds the radiated energy by the work done.

// The bilinear denominator both filters share, s^2 + B*wc*s + B*wc^2 at the corner wc.
// B is 2 for a body radiating against the air alone, and 2 + md/m once the body's own inertia is in the loop.
struct RecoilPoles {
    double A0{0}; // The bilinear scale the numerators divide by as well
    float A1{0}, A2{0};
};
inline RecoilPoles RecoilDenominator(double wc, double kk, double beta) {
    const double a0 = kk * kk + beta * wc * kk + beta * wc * wc;
    return {
        .A0 = a0,
        .A1 = float((2 * beta * wc * wc - 2 * kk * kk) / a0),
        .A2 = float((kk * kk - beta * wc * kk + beta * wc * wc) / a0),
    };
}

struct RecoilFilter {
    float RadB0{0}, AirB0{0}, AirB1{0}, AirB2{0}, A1{0}, A2{0};
};
inline RecoilFilter RecoilObjectFilter(double radius, double volume, double sample_rate) {
    if (radius <= 0 || volume <= 0) return {};
    const double wc = SpeedOfSound / radius;
    const double kk = 2 * sample_rate;
    const auto poles = RecoilDenominator(wc, kk, 2);
    const double gp = AirDensity * SpeedOfSound * radius / ListenerDistance;
    // The air load's numerator md*wc*(s^2 + wc*s): added mass md/2 below the corner, resistance rho0*c0*A/3 above.
    const double n2 = AirDensity * volume * wc, n1 = n2 * wc;
    return {
        .RadB0 = float(gp * kk * kk / poles.A0),
        .AirB0 = float((n2 * kk * kk + n1 * kk) / poles.A0),
        .AirB1 = float(-2 * n2 * kk * kk / poles.A0),
        .AirB2 = float((n2 * kk * kk - n1 * kk) / poles.A0),
        .A1 = poles.A1,
        .A2 = poles.A2,
    };
}
// The click's transfer folds the body's inertia into the same solution: pressure per unit contact force rho0*c0*a/(r*m) * s / (s^2 + B*wc*s + B*wc^2) with B = 2 + rho0*V/m.
// A body light for its volume then responds against the air's own load rather than radiating an unbounded jerk.
struct ClickFilter {
    float B0{0}, A1{0}, A2{0};
};
inline ClickFilter RecoilClickFilter(double radius, double volume, double mass, double sample_rate) {
    if (radius <= 0 || mass <= 0) return {};
    const double wc = SpeedOfSound / radius;
    const double kk = 2 * sample_rate;
    const auto poles = RecoilDenominator(wc, kk, 2 + AirDensity * volume / mass);
    const double g = AirDensity * SpeedOfSound * radius / (ListenerDistance * mass);
    return {.B0 = float(g * kk / poles.A0), .A1 = poles.A1, .A2 = poles.A2};
}

// The modal synthesis bank, struct-of-arrays.
// Each mode is a coupled-form (complex one-pole) resonator: z <- z*c + excitation, output Im(z).
struct ModalBank {
    // Per-mode, objects concatenated. Object o owns modes [ModeOffset[o], ModeOffset[o] + ModeCount[o]).
    std::vector<float> CoeffRe, CoeffIm; // Resonator coefficient c = decay * exp(i*2*pi*freq/SR). Zero mutes the mode.
    std::vector<float> StateRe, StateIm; // Resonator state z
    // The mode's far-field radiation gain in Pa per unit of mass-normalized modal velocity, zero for a muted mode.
    // A drive scaled by this leaves the state holding the mode's pressure at the listener distance.
    std::vector<float> RadiationGain;
    // Surface integral of the squared normal shape, m^2/kg at the baked size. Zero when the model has no sample surface.
    std::vector<float> RadiationArea;
    // One over the radiation gain times the angular frequency, which reads modal displacement back out of the state.
    std::vector<float> DeflectionGain;
    // Each mode's far-field phase at the listener, as the cos/sin pair its output contribution is rotated by.
    // Distinct modes of a body larger than the wavelength radiate through their own directivity lobes, so their pressures at any one listener have independent phases and the sum decoheres, while a compact body's modes stay in phase.
    // The phase is deterministic per mode and engages with (ka)^2/(1+(ka)^2), so a small body's coherent onset is untouched.
    std::vector<float> OutPhaseIm, OutPhaseRe;
    // The energy-quadratised exchange's per-mode constants (van Walstijn et al. 2024 Eqs. 28-33).
    // QuadCompliance is the mode's full-step displacement response coefficient dt*(1 + rho^2 + 2*rho*cos(theta))/4.
    // QuadDriveScale converts the bank's impulse-invariant drive to the central scheme's, so a sustained force scaled by it lands exactly the response the quadratised solve assumes.
    std::vector<float> QuadCompliance, QuadDriveScale;
    // Mass-normalized mode shapes. Object o, excitation position p, mode k: index = ShapeOffset[o] + p*ModeCount[o] + k.
    std::vector<float> ShapeX, ShapeY, ShapeZ;

    // Per-object.
    std::vector<entt::entity> Entities;
    std::vector<uint32_t> ModeOffset, ModeCount, ShapeOffset;
    // Modes at or past this index are muted. A mode shape row is still ModeCount wide.
    std::vector<uint32_t> TunedModeCount;
    std::vector<uint32_t> LiveModeCount; // Modes still audible, a prefix of the tuned set
    std::vector<float> OutGain; // Output level
    // Pressure attenuation to the view camera, ListenerDistance/max(distance, ListenerDistance), scaling an object's three output legs.
    std::vector<float> ListenerGain;
    std::vector<float> RadiantRadius; // Radius of the disc holding the sample surface's area, m at the baked size
    // Sizes the deflection read and the modal compliance to the object's world scale: mass-normalized shapes go as scale^(-3/2) and the loop uses two of them, so this is scale^(-3).
    std::vector<float> DeflectionScale;
    std::vector<uint8_t> Ringing; // Nonzero while the object has audible state
    // The object's motion on its own contact springs, as a fluctuation about the trajectory the physics step gives it.
    // A body resting on a contact of stiffness K bounces at sqrt(K/m), which lands in the audio band and which a physics step of 60 to 240 Hz cannot carry.
    // The contacts of one object share this velocity, so summing each contact's stiffness makes the rate independent of how the load divides among them.
    std::vector<float> RigidInvMass; // kg^-1, zero for an object the contact cannot move
    std::vector<vec3> RigidVel; // m/s, fluctuation only, in the object's own frame
    // A body recoiling without changing volume radiates as a compact dipole, through the recoil filters above.
    // All coefficients are zero for a body the world holds in place.
    std::vector<float> RadiatorB0; // Radiator numerator b0. Its b1 = -2*b0 and b2 = b0 (a pure s^2 numerator).
    std::vector<float> AirB0, AirB1, AirB2; // Air-load numerator, N per m/s of rigid velocity.
    std::vector<float> RecoilA1, RecoilA2; // Shared denominator of both recoil filters.
    std::vector<float> RadiatorZ1, RadiatorZ2, AirZ1, AirZ2; // Direct-form-II-transposed states.

    // One in-flight contact pulse. It generates a raised-cosine force curve via a unit-circle rotation:
    // phase <- phase*rot, force = Gamma * (1 - Re(phase)) / 2.
    struct ActiveImpact {
        uint32_t Object, ExPos, SamplesLeft;
        float Jx, Jy, Jz;
        float PhaseRe, PhaseIm;
        float RotRe, RotIm;
        float Gamma, AccelAmp;
        // The click is the same recoil radiator driven by this impact's force pulse, with the body's inertia in the loop (RecoilClickFilter).
        // Its b1 = 0 and b2 = -b0 (a pure s numerator), and AccelAmp scales the unit-sum pulse to the force in Newtons.
        // A finished pulse keeps its impact until the filter states drain, so the click ends without a step.
        float ClickB0, ClickA1, ClickA2, ClickZ1, ClickZ2;
    };
    std::vector<ActiveImpact> Impacts;

    float SampleRate{48'000};
};

// Modes are rendered in fixed-width lanes so the sample loop vectorizes across modes.
constexpr uint32_t Lanes{8};

// Mode `k`'s shape where a contact lands, blended over the three sample points of its own triangle.
// Each base is that point's first mode in the bank's shape columns, and `w` the point's weight.
inline vec3 BlendedShape(const ModalBank &b, uint32_t base0, uint32_t base1, uint32_t base2, vec3 w, uint32_t k) {
    return {
        w.x * b.ShapeX[base0 + k] + w.y * b.ShapeX[base1 + k] + w.z * b.ShapeX[base2 + k],
        w.x * b.ShapeY[base0 + k] + w.y * b.ShapeY[base1 + k] + w.z * b.ShapeY[base2 + k],
        w.x * b.ShapeZ[base0 + k] + w.y * b.ShapeZ[base1 + k] + w.z * b.ShapeZ[base2 + k],
    };
}

// The impulse of an impact projected onto `n` mode shapes starting at mode `first`.
inline void ImpactGainRow(const ModalBank &b, uint32_t impact, uint32_t shape0, uint32_t count, uint32_t k0, uint32_t first, uint32_t n, float *out) {
    const auto &im = b.Impacts[impact];
    const auto base = shape0 + im.ExPos * count + first;
    for (uint32_t i = 0; i < n; ++i) {
        out[i] = b.RadiationGain[k0 + first + i] * (b.ShapeX[base + i] * im.Jx + b.ShapeY[base + i] * im.Jy + b.ShapeZ[base + i] * im.Jz);
    }
}

// Threads that render objects alongside the caller, one index each, the caller taking index 0.
// A render never blocks on anything but its own workers.
struct ModalRenderPool {
    ~ModalRenderPool();

    // Main thread. `count` includes the calling thread, so a count of one spawns nothing.
    void SetSize(uint32_t count);
    // Main thread. Restarts the workers in `workgroup`, since a worker joins one only at startup.
    // Null places them in no group.
    void SetWorkgroup(void *workgroup);

    // Audio thread. One block's use of the pool.
    // Blocks a resize for as long as it lives, and reports a width of one when a resize already holds the pool.
    struct Session {
        explicit Session(ModalRenderPool &);
        ~Session();
        Session(const Session &) = delete;
        Session &operator=(const Session &) = delete;

        // Renderers this block dispatches to, the caller included. Fixed for the session's life.
        uint32_t Width() const { return W; }

        // Runs `render(context, i)` for every index below Width() and returns once they have all finished.
        void Run(void (*render)(void *, uint32_t), void *context);

    private:
        ModalRenderPool &Pool;
        bool Owns;
        uint32_t W;
    };

private:
    // Stop and join every worker past `keep`, leaving the pool at that width. Requires ResizeMutex.
    void StopWorkers(size_t keep);
    void ApplyLocked(uint32_t size, void *workgroup);

    std::vector<std::jthread> Threads;
    // Held by a resize, try-locked by a session, so a resize never overlaps a dispatch and audio never waits on one.
    std::mutex ResizeMutex;
    uint32_t Active{1}; // Renderers a session dispatches to. Guarded by ResizeMutex.
    void *Workgroup{nullptr};
    void (*Render)(void *, uint32_t){nullptr};
    void *Context{nullptr};
    // Workers below this index render on the next ticket, written before the ticket bump and read after the wait.
    // Zero leaves every worker parked.
    uint32_t Dispatch{0};
    std::atomic<uint64_t> Ticket{0}; // Bumped once per run to wake the workers.
    std::atomic<uint32_t> Remaining{0}; // Workers yet to finish this run.
};

// One renderer's working memory, held per renderer so objects can be rendered independently of one another.
// Kept across blocks, so a steady state allocates nothing.
struct ModalRenderScratch {
    std::vector<uint32_t> Objects; // The objects assigned to this renderer, in ascending order.
    std::vector<float> Out; // This renderer's share of the mix, summed into the callback's buffer in renderer order.
    std::vector<float> Gains; // Per-mode gain of each impact, for the hoisted-gain kernel.
    std::vector<uint32_t> Impacts; // The object's own, gathered out of the bank's flat list.
    // The surface-contact model's own working memory, allocated on this renderer's first sustained contact.
    SurfaceRenderScratchPtr Surface;
};

// All modal synthesis state.
// The audio thread reads the live bank via Published under a ReaderSeq generation.
// The main thread publishes a replacement and frees the old bank once that generation advances, never blocking audio.
// Coefficient writes stay plain: in a contracting one-pole a torn read is a one-buffer transient.
struct ModalAudio {
    ModalAudio();

    std::unique_ptr<ModalBank> Live; // Main-thread owner of the currently published bank
    std::atomic<ModalBank *> Published; // The audio thread loads this once per callback
    std::atomic<uint64_t> ReaderSeq{0}; // Audio-thread generation, odd while a callback renders

    std::atomic<float> ClickGain{1}; // Level of the rigid-body acceleration-noise click
    std::atomic<uint32_t> MaxImpacts{1024}; // Cap on simultaneous in-flight contact pulses
    std::atomic<uint32_t> ActiveImpacts{0}; // Published by the audio thread for display
    std::atomic<uint32_t> ActiveVoices{0}; // Sustained contacts the surface model holds, likewise
    // The energy standing in the mode banks, J, and the largest it has reached.
    // A passive scene loses energy between strikes, so a peak that climbs while nothing strikes is a channel feeding the modes.
    std::atomic<double> ModalEnergy{0}, PeakModalEnergy{0};
    // How long the last block took against how long it had, and the worst share since the last reset.
    std::atomic<float> RenderSeconds{0}, RenderShare{0}, PeakRenderShare{0};

    uint64_t EventsDropped{0}; // Main thread only.

    // Single-producer (main thread) single-consumer (audio thread) event queue.
    static constexpr uint32_t EventCapacity{256}; // Power of two
    std::array<ModalEvent, EventCapacity> Events;
    std::atomic<uint32_t> EventWrite{0}, EventRead{0};
    std::atomic<bool> FlushEvents{false}; // Main thread sets on publish. The audio thread drops events that targeted the old layout.

    // Audio-thread scratch shared by every object, kept across blocks.
    std::vector<float> ForceScratch; // Each impact's force curve for the block

    std::vector<ModalRenderScratch> Renderers;
    ModalRenderPool RenderPool;
    // Cost of each ringing object paired with its index, sorted so the heaviest is dealt first. Audio thread only.
    std::vector<std::pair<uint64_t, uint32_t>> RenderOrderScratch;
    std::vector<uint64_t> RenderLoadScratch; // Work already dealt to each renderer, so the next object goes to the lightest.

    // The surface-contact model's state, null in a build without it (see SurfaceContact.h).
    SurfaceAudioStatePtr Surface;
};

// The live bank, for main-thread reads and in-place writes.
inline ModalBank &LiveBank(ModalAudio &m) { return *m.Live; }

// Append an object slot with zeroed state, coefficients, and gain and return its index.
uint32_t AddModalObject(ModalBank &, entt::entity, const ModalModes &);
// Publish a freshly built bank as the live one and free the previous one.
void InstallModalBank(ModalAudio &, ModalBank &next);
// Set an object's resonator coefficients from per-mode frequencies (Hz) and T60s (s).
// Out-of-range and undamped modes are muted. Safe against concurrent rendering.
// `radius_scale` sizes the radiant radius with the object, so the radiation corner stays where the physical body has it.
void TuneModalObject(ModalBank &, uint32_t object, std::span<const float> freqs, std::span<const float> t60s, float radius_scale = 1.f);
// Overwrite an object's mode shapes in place. Returns false when the mode or shape layout differs.
// Safe against concurrent rendering.
bool SetModalObjectShapes(ModalBank &, uint32_t object, const ModalModes &);
// The object slot holding this entity, if any.
std::optional<uint32_t> FindModalObject(const ModalBank &, entt::entity);

// Enqueue an event from the main thread. Dropped when the queue is full.
void EnqueueModalEvent(ModalAudio &, const ModalEvent &);

// Add `frame_count` mono samples of modal synthesis into `out`, on the audio thread. Never blocks.
// The mix is pressure at the view camera: each object scales by its bank ListenerGain.
void RenderModal(ModalAudio &, float *out, uint32_t frame_count);
