#pragma once

#include <optional>
#include <vector>

#include "numeric/vec3.h"

#include "AudioSource.h"

enum class SoundObjectModel {
    RealImpact,
    // Model a rigid body's response to an impact using modal analysis/synthesis:
    // - transforming the object's geometry into a tetrahedral volume mesh
    // - using FEM to estimate the mass/spring/damping matrices from the mesh
    // - estimating the dominant modal frequencies and amplitudes using an eigenvalues/eigenvector solver
    // - simulating the object's response to an impact by exciting the modes associated with the impacted vertex
    Modal
};

struct RealImpact;
struct RealImpactListenerPoint;
struct Mesh;

namespace SoundObjectData {
struct RealImpact {
    RealImpact(std::vector<std::vector<float>> &&impact_samples) : ImpactSamples(std::move(impact_samples)) {}

    std::vector<std::vector<float>> ImpactSamples;
    uint CurrentVertexIndex{0}, CurrentFrame{0};
};

struct Modal {
    Modal(const Mesh &mesh) : Mesh(mesh) {}

    const Mesh &Mesh;
};
} // namespace SoundObjectData

// Represents a rigid mesh object that generate an audio stream for a listener at a given position
// in response to an impact at a given vertex.
struct SoundObject : AudioSource {
    // RealImpact only
    SoundObject(const RealImpact &, const RealImpactListenerPoint &);
    // Modal only
    SoundObject(const Mesh &, vec3 listener_position);
    // RealImpact and Modal
    SoundObject(const RealImpact &, const RealImpactListenerPoint &, const Mesh &);

    ~SoundObject();

    vec3 ListenerPosition;

    std::optional<SoundObjectData::RealImpact> RealImpactData{};
    std::optional<SoundObjectData::Modal> ModalData{};

    void SetModel(SoundObjectModel);

    void ProduceAudio(DeviceData, float *output, uint frame_count) override;

    void Strike(float force = 1.0);
    void RenderControls();

private:
    SoundObjectModel Model{SoundObjectModel::RealImpact};
};
