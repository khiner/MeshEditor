#pragma once

#include "AcousticMaterial.h"

#include <cstdint>
#include <string>

// Persistent audio output device selection. `SampleRate` is the desired rate (0 = device default).
// Changing either re-initializes the device.
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
    float ClickGain{1.f}; // Level of the rigid-body acceleration-noise click.
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

// If an entity has this component, user has opened the modal model create/edit pane.
struct ModalModelCreateInfo {
    AcousticMaterial Material{materials::acoustic::All.front()};
    uint32_t NumVertices{10};
    bool CopySoundVertices{true}; // Only used if excitable component is already present.
    bool QualityTets{false};
    float SolveResolution{1}; // Fraction of surface triangles used for the modal solve. Lower is faster and less accurate.
};
