#pragma once

#include <entt/entity/fwd.hpp>

#include <array>
#include <atomic>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

struct ModalModes;

enum class ModalEventKind : uint32_t {
    Impact, // Start a half-sine contact-force pulse on an object.
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
    float AccelAmp{0}; // Acceleration-noise click amplitude
};

// The modal synthesis bank, struct-of-arrays. Each mode is a coupled-form (complex one-pole)
// resonator: z <- z*c + excitation, output Im(z).
struct ModalBank {
    // Per-mode, objects concatenated. Object o owns modes [ModeOffset[o], ModeOffset[o] + ModeCount[o]).
    std::vector<float> CoeffRe, CoeffIm; // Resonator coefficient c = decay * exp(i*2*pi*freq/SR). Zero mutes the mode.
    std::vector<float> StateRe, StateIm; // Resonator state z
    // Mass-normalized mode shapes. Object o, excitation position p, mode k: index = ShapeOffset[o] + p*ModeCount[o] + k.
    std::vector<float> ShapeX, ShapeY, ShapeZ;

    // Per-object.
    std::vector<entt::entity> Entities;
    std::vector<uint32_t> ModeOffset, ModeCount, ShapeOffset;
    std::vector<float> OutGain; // Output level
    std::vector<uint8_t> Ringing; // Nonzero while the object has audible state

    // Active contact pulses, one per in-flight impact. Each generates a half-sine force curve
    // via a unit-circle rotation: phase <- phase*rot, force = gamma * Im(phase).
    std::vector<uint32_t> ImpactObject, ImpactExPos, ImpactSamplesLeft;
    std::vector<float> ImpactJx, ImpactJy, ImpactJz;
    std::vector<float> ImpactPhaseRe, ImpactPhaseIm;
    std::vector<float> ImpactRotRe, ImpactRotIm;
    std::vector<float> ImpactGamma, ImpactAccelAmp, ImpactPrevForce;

    float SampleRate{48'000};
};

// All modal synthesis state.
// The main thread builds a replacement bank off-lock and swaps it into the live bank under StructureMutex.
// The audio thread renders under try_lock and owns the bank's state, the impacts, and the scratch buffers.
// Element-wise coefficient and gain writes to the live bank's stable arrays are safe against concurrent rendering.
struct ModalAudio {
    ModalBank Bank;

    float ClickGain{1}; // Level of the rigid-body acceleration-noise click

    // Single-producer (main thread) single-consumer (audio thread) event queue.
    static constexpr uint32_t EventCapacity{256}; // Power of two
    std::array<ModalEvent, EventCapacity> Events;
    std::atomic<uint32_t> EventWrite{0}, EventRead{0};

    std::mutex StructureMutex;

    // Audio-thread scratch, kept across blocks.
    std::vector<float> ForceScratch;
    std::vector<float> GainScratch;
    std::vector<uint32_t> ObjectImpactScratch;
};

// Append an object slot with zeroed state, coefficients, and gain and return its index.
uint32_t AddModalObject(ModalBank &, entt::entity, const ModalModes &);
// Take StructureMutex and swap a freshly built bank into the live one, dropping pending events.
// The old bank moves into `next` and frees off-lock when `next` goes out of scope.
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

// Add `frame_count` mono samples of modal synthesis into `out`, on the audio thread.
// Skips the block when a structural change holds the lock.
void RenderModal(ModalAudio &, float *out, uint32_t frame_count);
