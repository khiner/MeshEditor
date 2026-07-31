#pragma once

#include "SurfaceNoise.h"
#include "numeric/vec3.h"

#include <entt/entity/fwd.hpp>

#include <array>
#include <atomic>
#include <bit>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

struct ModalModes;

// One-shot events the main thread queues for the audio thread.
enum class ModalEventKind : uint32_t {
    Impact, // Start a half-sine contact-force pulse on an object.
    Silence, // Clear an object's ringing state and drop its active pulses.
};

// Sample points a contact reads its mode shapes from, barycentric over a triangle of the sample surface.
struct SamplePointBlend {
    std::array<uint32_t, 3> Points{};
    vec3 Weights{1, 0, 0};
};

// One surface track a contact rides over.
struct ContactTrack {
    int32_t Index{-1}; // Surface track pool slot, -1 for an unused one.
    float Rate{0}; // Track samples advanced per output sample, from that surface's own sweep.
    float Sigma{0}; // Height scale applied to the track, m.
    float Window{0}; // Width of the contact filter over the track, in track samples.
    float Step{0}; // Distance along the surface per output sample, m.
};

// KHR_audio_rigid_bodies contact state a sustained voice renders, plus the constants derived for its force model.
struct SustainedState {
    SamplePointBlend Blend{};
    vec3 N{0}; // Node-local unit normal, directed into the object.
    vec3 SlipDir{0}; // Node-local unit slip direction. Zero when nothing slides.
    // Node-local direction each surface's geometric force drives this object, in the contact's own surface order.
    // Signed so that the two objects of one contact are driven apart rather than together.
    std::array<vec3, 2> SweepDir{};
    float NormalForce{0}; // N, the load the excitation fluctuates about.
    float Friction{0}; // Combined friction coefficient.
    float Stiffness{0}; // k, N/m^(3/2).
    float StaticPenetration{0}; // delta0, m.
    float DampingCoeff{0}; // Hunt-Crossley c_d, s/m.
    // Each surface's microscale finish, then its mesoscale relief.
    // A track's surface is its index's low bit, matching SweepDir. Both sides of a contact list them in the same order.
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
};

// What a sustained voice carries from one sample to the next.
struct SustainedCarry {
    std::array<double, SustainedState::TrackCount> Pos{};
    std::array<float, SustainedState::TrackCount> PrevHeight{};
    float ReliefMean{0};
    float Penetration{0};
    bool Primed{false}; // Set once the local mean is seeded from the first sample's relief height.
};

// The modal synthesis bank, struct-of-arrays.
// Each mode is a coupled-form (complex one-pole) resonator: z <- z*c + excitation, output Im(z).
struct ModalBank {
    // Per-mode, objects concatenated. Object o owns modes [ModeOffset[o], ModeOffset[o] + ModeCount[o]).
    std::vector<float> CoeffRe, CoeffIm; // Resonator coefficient c = decay * exp(i*2*pi*freq/SR). Zero mutes the mode.
    std::vector<float> StateRe, StateIm; // Resonator state z
    // Meters of modal displacement per unit of resonator state, 1/(2*pi*freq), and zero for a muted mode.
    std::vector<float> DisplacementScale;
    // Mass-normalized mode shapes. Object o, excitation position p, mode k: index = ShapeOffset[o] + p*ModeCount[o] + k.
    std::vector<float> ShapeX, ShapeY, ShapeZ;

    // Per-object.
    std::vector<entt::entity> Entities;
    std::vector<uint32_t> ModeOffset, ModeCount, ShapeOffset;
    // Modes at or past this index are muted. A mode shape row is still ModeCount wide.
    std::vector<uint32_t> TunedModeCount;
    std::vector<uint32_t> LiveModeCount; // Modes still audible, a prefix of the tuned set
    std::vector<float> OutGain; // Output level
    std::vector<uint8_t> Ringing; // Nonzero while the object has audible state

    // Active contact pulses, one per in-flight impact.
    // Each generates a half-sine force curve via a unit-circle rotation: phase <- phase*rot, force = gamma * Im(phase).
    std::vector<uint32_t> ImpactObject, ImpactExPos, ImpactSamplesLeft;
    std::vector<float> ImpactJx, ImpactJy, ImpactJz;
    std::vector<float> ImpactPhaseRe, ImpactPhaseIm;
    std::vector<float> ImpactRotRe, ImpactRotIm;
    std::vector<float> ImpactGamma, ImpactAccelAmp, ImpactPrevForce;

    // Active sustained contacts, each driving one object's modes and reading its deflection back.
    std::vector<uint64_t> VoiceId;
    std::vector<uint32_t> VoiceObject;
    std::vector<SustainedState> VoiceState;
    std::vector<SustainedCarry> VoiceCarry;

    float SampleRate{48'000};
};

// Every sustained contact of one main-thread frame, published whole.
// A contact missing from the newest set has ended.
struct VoiceSet {
    struct Voice {
        uint64_t Id; // Contact identity, which carries a voice's state across frames.
        uint32_t Object; // Bank object slot, valid only against the bank live when this was published.
        SustainedState State;
    };

    uint64_t Frame{0}; // The frame this was built in.
    std::vector<Voice> Voices;
};

// One track in the pool, shared by every surface whose finish or relief hashes to the same content.
// The audio thread loads Live without synchronizing, and the main thread repoints a slot only once no voice reads it.
struct SurfaceTrackSlot {
    std::atomic<const RoughnessTrack *> Live{nullptr};
    std::shared_ptr<const RoughnessTrack> Owned; // Main thread only. Shared with whatever else holds the track.
    uint64_t Key{0}; // Content key of Owned. Main thread only.
};

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
    std::atomic<float> SustainLevel{1}; // Level of the sustained-contact excitation
    std::atomic<float> Coupling{1}; // How much of the object's own vibration modulates the contact separation
    std::atomic<uint32_t> ActiveVoices{0}, ActiveImpacts{0}; // Published by the audio thread for display

    uint64_t SurfaceTracksRefused{0}, EventsDropped{0}, VoicesRefused{0}; // Main thread only.

    // Sustained contacts, republished whole each main-thread frame and adopted once per callback.
    // Three slots, since a frame-rate publish against shorter callbacks sometimes finds the previous one still being read.
    std::array<VoiceSet, 3> VoiceSets;
    std::atomic<const VoiceSet *> PublishedVoices{nullptr};
    uint32_t VoiceSetWrite{0}; // Main thread only.
    uint64_t VoiceFrame{0}; // Main thread only.
    uint64_t ContactStep{0}; // Main thread only. The simulation step the newest published set was built from.
    uint64_t AdoptedVoiceFrame{~0ull}; // Audio thread only.
    uint32_t VoiceSetIdleSamples{0}; // Audio thread only. Samples since the published frame last advanced.

    // Surface tracks sustained voices read, one slot per distinct track, addressed by a content key.
    // A slot the audio thread is done with keeps its track until another needs the slot.
    static constexpr uint32_t MaxSurfaceTracks{64}; // One bit per slot in VoiceTrackMask.
    std::array<SurfaceTrackSlot, MaxSurfaceTracks> SurfaceTracks;
    std::unordered_map<uint64_t, uint32_t> SurfaceTrackSlotByKey; // Content key to slot. Main thread only.
    // The slots this callback's voices read, so the main thread knows which it may repoint.
    std::atomic<uint64_t> VoiceTrackMask{0};
    uint64_t ReusableSlots{0}; // Slots free to repoint this frame, cleared as each is claimed. Main thread only.

    // Single-producer (main thread) single-consumer (audio thread) event queue.
    static constexpr uint32_t EventCapacity{256}; // Power of two
    std::array<ModalEvent, EventCapacity> Events;
    std::atomic<uint32_t> EventWrite{0}, EventRead{0};
    std::atomic<bool> FlushEvents{false}; // Main thread sets on publish. The audio thread drops events that targeted the old layout.

    // Audio-thread scratch shared by every object, kept across blocks.
    std::vector<float> ForceScratch; // Each impact's force curve for the block

    // One object's working memory, held per renderer so objects can be rendered independently of one another.
    // Kept across blocks, so a steady state allocates nothing.
    struct RenderScratch {
        std::vector<uint32_t> Objects; // The objects this renderer was dealt, in ascending order.
        std::vector<float> Out; // This renderer's share of the mix, summed into the callback's buffer in renderer order.
        std::vector<float> Gains; // Per-mode gain of each impact, for the hoisted-gain kernel.
        std::vector<uint32_t> Impacts, Voices; // The object's own, gathered out of the bank's flat lists.
        // Coupled-kernel scratch. A gain row is a mode shape projected onto a contact direction, fixed for the block.
        // Per-mode gain of each excitation: four rows per voice (normal, each surface's geometric, frictional), then one row per impact.
        std::vector<float> DriveGains;
        std::vector<float> ReadGains; // Per-mode deflection read-out gain, one row per voice.
        std::vector<float> Forces; // This sample's force behind each drive row.
        std::vector<float> Excite; // This sample's excitation of each mode.
        std::vector<const RoughnessTrack *> Tracks; // Each voice's surface tracks, resolved once per block.
    };
    std::vector<RenderScratch> Renderers;
    ModalRenderPool RenderPool;
    // Cost of each ringing object paired with its index, sorted so the heaviest is dealt first. Audio thread only.
    std::vector<std::pair<uint64_t, uint32_t>> RenderOrderScratch;
    std::vector<uint64_t> RenderLoadScratch; // Work already dealt to each renderer, so the next object goes to the lightest.
};

static_assert(ModalAudio::MaxSurfaceTracks == 8 * sizeof(decltype(ModalAudio::VoiceTrackMask)::value_type));

// Mark the slots no voice reads, which this frame may repoint. Main thread only.
inline void BeginSurfaceTrackFrame(ModalAudio &m) {
    auto named = m.VoiceTrackMask.load(std::memory_order_acquire);
    // The main thread cannot tell which of the three sets a callback still holds, so all three count.
    for (const auto &set : m.VoiceSets) {
        for (const auto &v : set.Voices) {
            for (const auto &t : v.State.Tracks) {
                if (t.Index >= 0) named |= 1ull << uint32_t(t.Index);
            }
        }
    }
    m.ReusableSlots = ~named;
}

// The pool slot holding `key`'s track, or -1 when every slot is taken. Main thread only.
// `make` returns a shared_ptr, so a caller already holding the track hands it over rather than copying it.
inline int32_t AdoptSurfaceTrack(ModalAudio &m, uint64_t key, auto &&make) {
    if (const auto it = m.SurfaceTrackSlotByKey.find(key); it != m.SurfaceTrackSlotByKey.end()) {
        m.ReusableSlots &= ~(1ull << it->second);
        return int32_t(it->second);
    }

    uint32_t index = 0;
    while (index < ModalAudio::MaxSurfaceTracks && m.SurfaceTracks[index].Owned) ++index;
    if (index == ModalAudio::MaxSurfaceTracks) {
        // Every slot holds a track, so take over one no voice reads.
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
// Safe against concurrent rendering.
bool SetModalObjectShapes(ModalBank &, uint32_t object, const ModalModes &);
// The object slot holding this entity, if any.
std::optional<uint32_t> FindModalObject(const ModalBank &, entt::entity);

// Enqueue an event from the main thread. Dropped when the queue is full.
void EnqueueModalEvent(ModalAudio &, const ModalEvent &);

// An empty set to write this frame's contacts into, never one a callback may still be reading. Main thread only.
VoiceSet &NextVoiceSet(ModalAudio &);
// Publish the set NextVoiceSet handed out, which ends every contact it omits. Main thread only.
void PublishVoiceSet(ModalAudio &);

// Add `frame_count` mono samples of modal synthesis into `out`, on the audio thread. Never blocks.
void RenderModal(ModalAudio &, float *out, uint32_t frame_count);
