#pragma once

#include <cstdint>
#include <string>

// Persistent audio output device selection. `SampleRate` is the desired rate (0 = device default).
struct AudioOutputConfig {
    std::string DeviceName;
    uint32_t SampleRate{0};
};

// Persistent audio output level. Changing these applies without re-initializing the device.
struct AudioOutputMix {
    bool On{true};
    bool Muted{false};
    float Volume{1.f};
};

// The monitor limiter's running peak envelope, in full-scale units.
struct MonitorLimiter {
    float Envelope{0};
};

// Viewport-level modal synthesis controls.
struct ModalSoundControls {
    uint32_t RenderThreads{4}; // Threads per block, including the audio callback.
    uint32_t MaxImpacts{1024}; // Cap on simultaneous in-flight contact pulses.
    float ModalLevel{0.5f}; // Gain on every modal object's resonator output.
    float ClickGain{1.f}; // Level of the rigid-body acceleration-noise click.
    float SampleGain{1.f}; // Level of impact-sample playback.
    // A physics collision sounds only when the modal excitation it produces and its approach speed (m/s) clear these floors.
    // The excitation floor sits below the amplitude the render culls a mode at, so it drops only strikes that would render as silence.
    // The speed floor suppresses large support impulses from stationary loaded bodies.
    float MinContactExcitation{1e-7f}, MinContactSpeed{0.01f};
};

enum class SoundVerticesModel {
    // Plays impact recordings sampled at supplied object vertices.
    Samples,
    // Synthesizes impacts from finite-element modes of a tetrahedral volume mesh.
    Modal,
};

// Modal solve inputs beyond the mesh and its acoustic material: tet meshing options and the excitation vertex selection. Per sound entity.
struct ModalSolveSettings {
    uint32_t NumVertices{10};
    float SolveResolution{1}; // Fraction of surface triangles used for the modal solve. Lower is faster and less accurate.
    bool CopySoundVertices{true}; // Solve at the existing excitable vertices when present.
    bool QualityTets{false};
    uint32_t NumModes{30}; // Modes kept from the solve.
    float MinModeFreq{20}, MaxModeFreq{16'000}; // Synthesized frequency band, Hz.
};
