#pragma once

#include "SoundObject.h"

struct Mesh;

// Model a rigid body's response to an impact using modal analysis/synthesis:
// - transforming the object's geometry into a tetrahedral volume mesh
// - using FEM to estimate the mass/spring/damping matrices from the mesh
// - estimating the dominant modal frequencies and amplitudes using an eigenvalues/eigenvector solver
// - simulating the object's response to an impact by exciting the modes associated with the impacted vertex
struct ModalSoundObject : SoundObject {
    ModalSoundObject(const Mesh &, vec3 listener_position);
    ~ModalSoundObject() override = default;

    void Strike(uint vertex_index, float force = 1.0) override;
    void ProduceAudio(DeviceData, float *output, uint frame_count) override;

    const Mesh &Mesh;
};
