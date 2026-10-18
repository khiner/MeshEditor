#pragma once

#include "Field.h"
#include "audio/AudioSchema.h"

#include <FastFEM/Surface2Modes.h>

// Persistent audio output device selection.
// `SampleRate` is the desired rate (0 = device default).
struct AudioOutputConfig {
    std::string DeviceName;
    uint32_t SampleRate{0};
};

// Persistent audio output level.
// Changing these applies without re-initializing the device.
struct AudioOutputMix {
    bool On{true};
    bool Muted{false};
    float Volume{1.f};
};
template<> inline constexpr field::SpecTable<AudioOutputMix> Specs<AudioOutputMix>{
    {&AudioOutputMix::On, {.Help = "Run the audio output device"}},
    {&AudioOutputMix::Muted, {.Help = "Silence output while the device keeps running"}},
    {&AudioOutputMix::Volume, {.Min = 0, .Max = 1, .Help = "Output gain"}},
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
template<> inline constexpr field::SpecTable<ModalSoundControls> Specs<ModalSoundControls>{
    {&ModalSoundControls::RenderThreads, {.Min = 1, .Max = 16, .Help = "Threads that render each audio block. More threads help scenes with many sounding objects."}},
    {&ModalSoundControls::MaxImpacts, {.Min = 1, .Max = 4096, .Help = "Maximum number of contact pulses playing at once"}},
    {&ModalSoundControls::ModalLevel, {.Min = 0, .Max = 1, .Label = "Modal gain", .Help = "Gain on every modal object's output"}},
    {&ModalSoundControls::ClickGain, {.Min = 0, .Max = 10, .Label = "Click", .Help = "Level of the rigid-body acceleration-noise click"}},
    {&ModalSoundControls::SampleGain, {.Min = 0, .Max = 4, .Help = "Level of impact-sample playback"}},
    {&ModalSoundControls::MinContactExcitation, {.Min = 0, .Max = 1e-3, .Scale = FieldScale::Logarithmic, .Format = "%.3g", .Help = "Smallest peak mode amplitude a physics collision must excite to sound"}},
    {&ModalSoundControls::MinContactSpeed, {.Min = 0, .Max = 5, .Unit = FieldUnit::MetersPerSecond, .Label = "Min contact speed", .Help = "Smallest approach speed at which a physics collision sounds. Silences support impulses from resting bodies."}},
};

enum class SoundVerticesModel {
    // Plays impact recordings sampled at supplied object vertices.
    Samples,
    // Synthesizes impacts from finite-element modes.
    Modal,
};

// Modal solve inputs beyond the surface mesh and its acoustic material.
// Per sound entity.
struct ModalSolveSettings {
    fastfem::Discretization Discretization{fastfem::Discretization::Tet10};
    fastfem::SurfaceSolveConfig Solve{};
    uint32_t NumVertices{10};
    bool CopySoundVertices{true}; // Solve at the existing excitable vertices when present.
};
template<> inline constexpr field::SpecTable<ModalSolveSettings> Specs<ModalSolveSettings>{
    {&ModalSolveSettings::Discretization, {.Help = "Tet10 fills the interior with quadratic tetrahedra. Finite cell embeds the surface in a grid of quadratic hexahedral cells."}},
    {&ModalSolveSettings::NumVertices, {.Label = "Vertex count", .Help = "Number of excitation vertices, spread evenly over the mesh"}},
    {&ModalSolveSettings::CopySoundVertices, {.Label = "Reuse existing vertices", .Help = "Solve at the current excitation vertices instead of new evenly spread ones"}},
};
template<> inline constexpr field::SpecTable<fastfem::SurfaceSolveConfig> Specs<fastfem::SurfaceSolveConfig>{
    {&fastfem::SurfaceSolveConfig::Tetrahedralization, {.Label = "Tet10 mesh", .Help = "Surface tetrahedralization settings for the Tet10 discretization"}},
    {&fastfem::SurfaceSolveConfig::FiniteCell, {.Label = "Finite-cell grid", .Help = "Background grid settings for the finite-cell discretization"}},
    {&fastfem::SurfaceSolveConfig::Resolution, {.Min = 1, .Max = 256, .Label = "Target resolution", .Help = "Target divisions along the object's longest axis"}},
    {&fastfem::SurfaceSolveConfig::SurfaceSimplificationRatio, {.Min = 0.25, .Max = 1, .Label = "Surface detail", .Help = "Fraction of surface triangles kept before tetrahedralization. One keeps the original mesh."}},
};
template<> inline constexpr field::SpecTable<fastfem::TetrahedralizationConfig> Specs<fastfem::TetrahedralizationConfig>{
    {&fastfem::TetrahedralizationConfig::Refinement, {.Help = "None only tetrahedralizes and repairs, Quality also improves element shapes, and Quality And Resolution also refines to the target resolution"}},
    {&fastfem::TetrahedralizationConfig::Holes, {.Unit = FieldUnit::Meters, .Label = "Hole seed", .Help = "Point inside a cavity the tetrahedralization leaves empty"}},
};
template<> inline constexpr field::SpecTable<fastfem::SolverConfig> Specs<fastfem::SolverConfig>{
    {&fastfem::SolverConfig::MinModeFreq, {.Min = 20, .Max = 20000, .Unit = FieldUnit::Hertz, .Scale = FieldScale::Logarithmic, .Label = "Min frequency", .Help = "Lowest retained mode frequency"}},
    {&fastfem::SolverConfig::MaxModeFreq, {.Min = 20, .Max = 20000, .Unit = FieldUnit::Hertz, .Scale = FieldScale::Logarithmic, .Label = "Max frequency", .Help = "Highest retained mode frequency"}},
    {&fastfem::SolverConfig::NumModes, {.Min = 1, .Max = 512, .Label = "Retained modes", .Help = "Number of synthesized modes kept from the FEM eigenpairs within the frequency band"}},
    {&fastfem::SolverConfig::NumFemModes, {.Min = 1, .Max = 512, .Label = "FEM eigenpairs", .Help = "Number of eigenpairs computed before filtering by frequency band and mode count"}},
    {&fastfem::SolverConfig::Tolerance, {.Min = 1e-12, .Max = 1e-3, .Scale = FieldScale::Logarithmic, .Label = "Residual tolerance", .Format = "%.1e", .Help = "Eigensolver convergence tolerance"}},
    {&fastfem::SolverConfig::MaxRestarts, {.Min = 1, .Max = 1000, .Label = "Restart limit", .Help = "Maximum number of eigensolver restarts"}},
    {&fastfem::SolverConfig::FundamentalFreq, {.Min = 20, .Max = 20000, .Initial = 440, .Unit = FieldUnit::Hertz, .Scale = FieldScale::Logarithmic, .Label = "Fundamental override", .Help = "Frequency of the lowest mode, scaling all solved modes. Absent uses an active impact recording's estimated fundamental."}},
};
template<> inline constexpr field::SpecTable<fastfem::FiniteCellConfig> Specs<fastfem::FiniteCellConfig>{
    {&fastfem::FiniteCellConfig::CutDepth, {.Min = 0, .Max = 8, .Help = "Subdivision depth of cells the surface cuts. Deeper levels fit the surface better and take longer."}},
    {&fastfem::FiniteCellConfig::FictitiousScale, {.Min = 1e-12, .Max = 1e-2, .Scale = FieldScale::Logarithmic, .Format = "%.1e", .Help = "Stiffness and mass scale on the grid region outside the surface"}},
    {&fastfem::FiniteCellConfig::PaddingCells, {.Min = 0, .Max = 2, .Unit = FieldUnit::Cells, .Label = "Padding", .Help = "Margin of the background grid beyond the surface bounds"}},
    {&fastfem::FiniteCellConfig::GridOffsetCells, {.Speed = 0.01f, .Unit = FieldUnit::Cells, .Label = "Grid offset", .Help = "Offset of the background grid from the surface bounds"}},
};
