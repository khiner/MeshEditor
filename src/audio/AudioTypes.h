#pragma once

#include "Field.h"

#include <FastFEM/Surface2Modes.h>

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
template<> inline constexpr FieldSpec Spec<AudioOutputMix, "Volume">{.Min = 0, .Max = 1};

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
template<> inline constexpr FieldSpec Spec<ModalSoundControls, "ModalLevel">{.Min = 0, .Max = 1};
template<> inline constexpr FieldSpec Spec<ModalSoundControls, "ClickGain">{.Min = 0, .Max = 10};
template<> inline constexpr FieldSpec Spec<ModalSoundControls, "SampleGain">{.Min = 0, .Max = 4};
template<> inline constexpr FieldSpec Spec<ModalSoundControls, "RenderThreads">{.Min = 1, .Max = 16};
template<> inline constexpr FieldSpec Spec<ModalSoundControls, "MaxImpacts">{.Min = 1, .Max = 4096};
template<> inline constexpr FieldSpec Spec<ModalSoundControls, "MinContactExcitation">{.Min = 0, .Max = 1e-3};
template<> inline constexpr FieldSpec Spec<ModalSoundControls, "MinContactSpeed">{.Min = 0, .Max = 5};

enum class SoundVerticesModel {
    // Plays impact recordings sampled at supplied object vertices.
    Samples,
    // Synthesizes impacts from finite-element modes.
    Modal,
};

// Modal solve inputs beyond the surface mesh and its acoustic material. Per sound entity.
struct ModalSolveSettings {
    fastfem::Discretization Discretization{fastfem::Discretization::Tet10};
    fastfem::SurfaceSolveConfig Solve{};
    uint32_t NumVertices{10};
    bool CopySoundVertices{true}; // Solve at the existing excitable vertices when present.
};
template<> inline constexpr FieldSpec Spec<fastfem::SurfaceSolveConfig, "Resolution">{.Min = 1, .Max = 256};
template<> inline constexpr FieldSpec Spec<fastfem::SurfaceSolveConfig, "SurfaceSimplificationRatio">{.Min = 0.25, .Max = 1};
template<> inline constexpr FieldSpec Spec<fastfem::SolverConfig, "NumModes">{.Min = 1, .Max = 512};
template<> inline constexpr FieldSpec Spec<fastfem::SolverConfig, "NumFemModes">{.Min = 1, .Max = 512};
template<> inline constexpr FieldSpec Spec<fastfem::SolverConfig, "MinModeFreq">{.Min = 20, .Max = 20000, .Digits = 0};
template<> inline constexpr FieldSpec Spec<fastfem::SolverConfig, "MaxModeFreq">{.Min = 20, .Max = 20000, .Digits = 0};
template<> inline constexpr FieldSpec Spec<fastfem::SolverConfig, "Tolerance">{.Min = 1e-12, .Max = 1e-3};
template<> inline constexpr FieldSpec Spec<fastfem::SolverConfig, "MaxRestarts">{.Min = 1, .Max = 1000};
template<> inline constexpr FieldSpec Spec<fastfem::FiniteCellConfig, "CutDepth">{.Min = 0, .Max = 8};
template<> inline constexpr FieldSpec Spec<fastfem::FiniteCellConfig, "FictitiousScale">{.Min = 1e-12, .Max = 1e-2};
template<> inline constexpr FieldSpec Spec<fastfem::FiniteCellConfig, "PaddingCells">{.Min = 0, .Max = 2, .Digits = 2};
