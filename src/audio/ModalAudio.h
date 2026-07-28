#pragma once

#include "SurfaceNoise.h"
#include "numeric/vec3.h"

#include <entt/entity/fwd.hpp>

#include <array>
#include <atomic>
#include <bit>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

struct ModalModes;

enum class ModalEventKind : uint32_t {
    Impact, // Start a half-sine contact-force pulse on an object.
    Contact, // Open a sustained-contact voice, or refresh one already open.
    ContactEnd, // Close a sustained voice.
    Silence, // Clear an object's ringing state and drop its active pulses.
};

// The two sample points nearest a contact and the blend between them, so the timbre varies continuously as the contact travels.
struct SamplePointBlend {
    uint32_t First{0}, Second{0};
    float Weight{1};
};

// One surface track a contact rides over, resolved on the main thread.
struct ContactTrack {
    int32_t Index{-1}; // Surface track pool slot, -1 for an unused one.
    float Rate{0}; // Track samples advanced per output sample, from that surface's own sweep.
    float Sigma{0}; // Height scale applied to the track, m.
    float Window{0}; // Width of the contact filter over the track, in track samples.
    float Step{0}; // Distance along the surface per output sample, m.
};

// The contact state a sustained voice renders, refreshed every frame the contact lasts (KHR_audio_rigid_bodies contact state, plus the constants its reference force model derives).
struct SustainedState {
    SamplePointBlend Blend{};
    vec3 N{0}; // Node-local unit normal, directed into the object.
    vec3 Slip{0}; // Node-local relative tangential velocity, m/s. Its direction drives the frictional channel and its magnitude scales it.
    float NormalForce{0}; // N, the load the excitation fluctuates about.
    float Stiffness{0}; // k, N/m^(3/2).
    float StaticPenetration{0}; // delta0, m.
    float DampingCoeff{0}; // Hunt-Crossley c_d, s/m.
    // The tracks the contact rides over: each surface's microscale finish, then its mesoscale relief, which add and each advance at that surface's own sweep.
    static constexpr uint32_t TrackCount{4};
    std::array<ContactTrack, TrackCount> Tracks{};
};

// One queued modal synthesis event.
struct ModalEvent {
    ModalEventKind Kind{ModalEventKind::Impact};
    uint32_t Object{0}; // Object slot in the bank
    uint32_t ExPos{0}; // Excitation position index
    float Jx{0}, Jy{0}, Jz{0}; // Node-local impulse vector
    float PulseStep{0}; // Per-sample phase increment of the contact pulse
    float PulseGamma{0}; // Contact pulse amplitude
    float AccelAmp{0}; // Acceleration-noise click amplitude
    uint64_t ContactId{0}; // Identifies the sustained contact a voice renders
    SustainedState Contact{};
};

// Per-sample carry of a sustained voice: each track's read position, the relief's local mean, and the previous separation.
struct SustainedCarry {
    std::array<double, SustainedState::TrackCount> Pos{};
    std::array<float, SustainedState::TrackCount> PrevHeight{};
    float ReliefMean{0};
    float Penetration{0};
    bool Primed{false}; // The local mean starts at the first sample's relief, so a contact at rest excites nothing.
};

// The modal synthesis bank, struct-of-arrays. Each mode is a coupled-form (complex one-pole)
// resonator: z <- z*c + excitation, output Im(z).
struct ModalBank {
    // Per-mode, objects concatenated. Object o owns modes [ModeOffset[o], ModeOffset[o] + ModeCount[o]).
    std::vector<float> CoeffRe, CoeffIm; // Resonator coefficient c = decay * exp(i*2*pi*freq/SR). Zero mutes the mode.
    std::vector<float> StateRe, StateIm; // Resonator state z
    // Meters of modal displacement per unit of resonator state, 1/(SR * 2*pi*freq), and zero for a muted mode.
    std::vector<float> DisplacementScale;
    // Mass-normalized mode shapes. Object o, excitation position p, mode k: index = ShapeOffset[o] + p*ModeCount[o] + k.
    std::vector<float> ShapeX, ShapeY, ShapeZ;

    // Per-object.
    std::vector<entt::entity> Entities;
    std::vector<uint32_t> ModeOffset, ModeCount, ShapeOffset;
    // Modes at or past this index are muted and radiate nothing, so the render kernels stop there.
    // A mode shape row is still ModeCount wide, which is what indexes it.
    std::vector<uint32_t> LiveModeCount;
    std::vector<float> OutGain; // Output level
    std::vector<uint8_t> Ringing; // Nonzero while the object has audible state

    // Active contact pulses, one per in-flight impact. Each generates a half-sine force curve
    // via a unit-circle rotation: phase <- phase*rot, force = gamma * Im(phase).
    std::vector<uint32_t> ImpactObject, ImpactExPos, ImpactSamplesLeft;
    std::vector<float> ImpactJx, ImpactJy, ImpactJz;
    std::vector<float> ImpactPhaseRe, ImpactPhaseIm;
    std::vector<float> ImpactRotRe, ImpactRotIm;
    std::vector<float> ImpactGamma, ImpactAccelAmp, ImpactPrevForce;

    // Active sustained contacts, each driving one object's modes from the surface tracks riding through the contact and reading that object's deflection back.
    std::vector<uint64_t> VoiceId;
    std::vector<uint32_t> VoiceObject;
    std::vector<SustainedState> VoiceState;
    std::vector<SustainedCarry> VoiceCarry;
    std::vector<uint32_t> VoiceIdleSamples; // Samples since the last refresh. A voice that stops being refreshed ends itself.

    float SampleRate{48'000};
};

// One track in the pool, shared by every surface whose finish or relief hashes to the same content.
// The audio thread loads Live without synchronizing. The main thread repoints a slot only once no voice and no queued event names it,
// so a voice always reads the track it was given and a displaced track is free to destroy on the spot.
struct SurfaceTrackSlot {
    std::atomic<const RoughnessTrack *> Live{nullptr};
    std::shared_ptr<const RoughnessTrack> Owned; // Main thread only. Shared so a track a component already holds is pooled without copying it.
    uint64_t Key{0}; // Content key of Owned. Main thread only.
};

// All modal synthesis state. The audio thread reads the live bank via Published under a ReaderSeq generation.
// The main thread publishes a replacement and frees the old bank once that generation advances, never blocking audio.
// Coefficient writes stay plain (a contracting one-pole keeps a torn read a one-buffer transient).
struct ModalAudio {
    ModalAudio();

    std::unique_ptr<ModalBank> Live; // Main-thread owner of the currently published bank
    std::atomic<ModalBank *> Published; // The audio thread loads this once per callback
    std::atomic<uint64_t> ReaderSeq{0}; // Audio-thread generation, odd while a callback renders

    std::atomic<float> ClickGain{1}; // Level of the rigid-body acceleration-noise click
    std::atomic<uint32_t> MaxImpacts{1024}; // Cap on simultaneous in-flight contact pulses
    std::atomic<float> SustainLevel{1}; // Level of the sustained-contact excitation
    std::atomic<float> Coupling{1}; // How much of the object's own vibration modulates the contact separation
    std::atomic<uint32_t> MaxVoices{16}; // Cap on simultaneous sustained-contact voices
    std::atomic<uint32_t> ActiveVoices{0}, ActiveImpacts{0}; // Published by the audio thread for display

    // Demand that went unmet, so a scene that has gone quiet can be told from one that is starved. Main thread only.
    uint64_t SurfaceTracksRefused{0}, EventsDropped{0};

    // Contact ids a voice was opened for, and the ids this frame's report renewed.
    // Swapped once the frame's events are out, so a contact that stops being reported closes its voice exactly once.
    // Main thread only.
    std::vector<uint64_t> OpenContactVoices, RefreshedContactVoices;

    // Surface tracks sustained voices read, one slot per distinct track, addressed by a content key.
    // A slot the audio thread is done with keeps its track as a cache entry, so a contact that stops and resumes costs a lookup rather than a rebuild.
    static constexpr uint32_t MaxSurfaceTracks{64}; // One bit per slot in VoiceTrackMask, which is what sizes the pool.
    std::array<SurfaceTrackSlot, MaxSurfaceTracks> SurfaceTracks;
    std::unordered_map<uint64_t, uint32_t> SurfaceTrackSlotByKey; // Content key to slot. Main thread only.
    // The slots this callback's voices name, published just before the read position so the main thread can tell what it may repoint.
    std::atomic<uint64_t> VoiceTrackMask{0};
    uint64_t ReusableSlots{0}; // Slots free to repoint this frame, cleared as each is spoken for. Main thread only.

    // Single-producer (main thread) single-consumer (audio thread) event queue.
    static constexpr uint32_t EventCapacity{256}; // Power of two
    std::array<ModalEvent, EventCapacity> Events;
    std::atomic<uint32_t> EventWrite{0}, EventRead{0};
    std::atomic<bool> FlushEvents{false}; // Main thread sets on publish. The audio thread drops events that targeted the old layout.

    // Audio-thread scratch, kept across blocks.
    std::vector<float> ForceScratch, GainScratch;
    std::vector<uint32_t> ObjectImpactScratch, ObjectVoiceScratch;
    // Coupled-kernel scratch. A gain row is a mode shape projected onto a contact direction, which the whole block shares
    // because events drain before rendering, so only the forces multiplying the rows change from sample to sample.
    std::vector<float> CoupledDriveGainScratch; // Per-mode gain of each excitation: two rows per voice (normal channel, tangential channel), then one row per impact.
    std::vector<float> CoupledReadGainScratch; // Per-mode deflection read-out gain, one row per voice.
    std::vector<float> CoupledForceScratch; // This sample's force behind each drive row.
    std::vector<float> CoupledExciteScratch; // This sample's excitation of each mode.
    std::vector<const RoughnessTrack *> CoupledTrackScratch; // Each voice's surface tracks, resolved once per block.
};

// A slot is one bit of the mask the audio thread publishes, which is what fixes the pool at this size.
static_assert(ModalAudio::MaxSurfaceTracks == 8 * sizeof(decltype(ModalAudio::VoiceTrackMask)::value_type));

// Open a frame of track adoption by working out which slots neither a voice nor a queued event names. Main thread only.
inline void BeginSurfaceTrackFrame(ModalAudio &m) {
    // The read position is published after the mask, so an event this reads as consumed has the voice it started already in the mask.
    const auto read = m.EventRead.load(std::memory_order_acquire);
    const auto write = m.EventWrite.load(std::memory_order_relaxed);
    auto named = m.VoiceTrackMask.load(std::memory_order_acquire);
    // An event still queued names slots no voice holds yet.
    for (auto i = read; i != write; ++i) {
        for (const auto &t : m.Events[i % ModalAudio::EventCapacity].Contact.Tracks) {
            if (t.Index >= 0) named |= 1ull << uint32_t(t.Index);
        }
    }
    m.ReusableSlots = ~named;
}

// The pool slot holding `key`'s track, or -1 when every slot is spoken for. Main thread only.
// `make` returns a shared_ptr to the track, so a caller that already holds one hands it over rather than copying it.
inline int32_t AdoptSurfaceTrack(ModalAudio &m, uint64_t key, auto &&make) {
    if (const auto it = m.SurfaceTrackSlotByKey.find(key); it != m.SurfaceTrackSlotByKey.end()) {
        m.ReusableSlots &= ~(1ull << it->second);
        return int32_t(it->second);
    }

    uint32_t index = 0;
    while (index < ModalAudio::MaxSurfaceTracks && m.SurfaceTracks[index].Owned) ++index;
    if (index == ModalAudio::MaxSurfaceTracks) {
        // Every slot holds a track, so take over one whose track nothing names any more.
        if (m.ReusableSlots == 0) {
            ++m.SurfaceTracksRefused;
            return -1;
        }
        index = uint32_t(std::countr_zero(m.ReusableSlots));
        m.SurfaceTrackSlotByKey.erase(m.SurfaceTracks[index].Key);
    }
    auto &slot = m.SurfaceTracks[index];
    slot.Live.store(nullptr, std::memory_order_relaxed);
    slot.Owned = make();
    slot.Key = key;
    slot.Live.store(slot.Owned.get(), std::memory_order_release);
    m.SurfaceTrackSlotByKey.emplace(key, index);
    m.ReusableSlots &= ~(1ull << index);
    return int32_t(index);
}

// The live bank, for main-thread reads and in-place writes.
inline ModalBank &LiveBank(ModalAudio &m) { return *m.Live; }

// Append an object slot with zeroed state, coefficients, and gain and return its index.
uint32_t AddModalObject(ModalBank &, entt::entity, const ModalModes &);
// Publish a freshly built bank as the live one and free the previous one.
void InstallModalBank(ModalAudio &, ModalBank &next);
// Set an object's resonator coefficients from per-mode frequencies (Hz) and T60s (s).
// Out-of-range and undamped modes are muted. Safe against concurrent rendering.
void TuneModalObject(ModalBank &, uint32_t object, std::span<const float> freqs, std::span<const float> t60s);
// Overwrite an object's mode shapes in place. Returns false when the mode or shape layout differs.
// Element-wise writes to stable arrays, safe against concurrent rendering.
bool SetModalObjectShapes(ModalBank &, uint32_t object, const ModalModes &);
// The object slot holding this entity, if any.
std::optional<uint32_t> FindModalObject(const ModalBank &, entt::entity);

// Enqueue an event from the main thread. Dropped when the queue is full.
void EnqueueModalEvent(ModalAudio &, const ModalEvent &);

// Add `frame_count` mono samples of modal synthesis into `out`, on the audio thread. Never blocks.
void RenderModal(ModalAudio &, float *out, uint32_t frame_count);
