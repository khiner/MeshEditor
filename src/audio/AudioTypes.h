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

// Viewport-level modal synthesis controls.
struct ModalSoundControls {
    float ModalLevel{0.5f}; // Gain on every modal object's resonator output.
    float ClickGain{1.f}; // Level of the rigid-body acceleration-noise click.
    float SampleGain{1.f}; // Level of impact-sample playback.
    uint32_t MaxImpacts{1024}; // Cap on simultaneous in-flight contact pulses.
    // A physics collision sounds only when its impulse (kg·m/s) and approach speed (m/s) clear these floors.
    // They keep settling and micro-jitter contacts from buzzing.
    float MinContactImpulse{0.2f}, MinContactSpeed{0.1f};
};

enum class SoundVerticesModel {
    // Play back recordings of impacts on the object at provided listener points/vertices.
    Samples,
    // Model a rigid body's response to an impact using modal analysis/synthesis:
    // - transform the object's geometry into a tetrahedral volume mesh
    // - use FEM to estimate the mass/sg/damping matrices from the mesh
    // - estimate the dominant modes (frequencies/amplitudes/T60s) using an eigensolver
    // - simulate the object's response to an impact by exciting the modes associated with the impacted vertex
    Modal,
};

// Modal solve inputs beyond the mesh and its acoustic material: tet meshing options and the
// excitation vertex selection. Per sound entity.
struct ModalSolveSettings {
    uint32_t NumVertices{10};
    float SolveResolution{1}; // Fraction of surface triangles used for the modal solve. Lower is faster and less accurate.
    bool CopySoundVertices{true}; // Solve at the existing excitable vertices when present.
    bool QualityTets{false};
    uint32_t NumModes{30}; // Modes kept from the solve.
    float MinModeFreq{20}, MaxModeFreq{16'000}; // Synthesized frequency band, Hz.
};
