#pragma once

#include "AcousticMaterial.h"

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
};
