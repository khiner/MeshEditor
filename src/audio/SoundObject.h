#pragma once

#include <optional>
#include <vector>

#include "numeric/vec3.h"

#include "AudioSource.h"
#include "MaterialProperties.h"

enum class SoundObjectModel {
    RealImpact,
    // Model a rigid body's response to an impact using modal analysis/synthesis:
    // - transforming the object's geometry into a tetrahedral volume mesh
    // - using FEM to estimate the mass/sg/damping matrices from the mesh
    // - estimating the dominant modal frequencies and amplitudes using an eigenvalues/eigenvector solver
    // - simulating the object's response to an impact by exciting the modes associated with the impacted vertex
    Modal
};

struct RealImpact;
struct RealImpactListenerPoint;
struct Mesh;
struct FaustDSP;

// All model-specific data needed to render audio.
namespace SoundObjectData {
struct RealImpact {
    RealImpact(std::vector<std::vector<float>> &&impact_samples, const std::vector<uint> &vertex_indices)
        : ImpactSamples(std::move(impact_samples)), VertexIndices(vertex_indices) {
            // Start at the end of the first sample, so it doesn't immediately play.
            // All RealImpact samples are the same length.
        if (!ImpactSamples.empty()) CurrentFrame = ImpactSamples.front().size();
    }

    std::vector<std::vector<float>> ImpactSamples;
    std::vector<uint> VertexIndices;
    uint CurrentVertexIndex{0}, CurrentFrame{0};
};

struct Modal {
    Modal(const Mesh &);
    ~Modal();

    const Mesh &Mesh;
    std::unique_ptr<FaustDSP> FaustDsp;
};
} // namespace SoundObjectData

// Represents a rigid mesh object that generate an audio stream for a listener at a given position
// in response to an impact at a given vertex.
struct SoundObject : AudioSource {
    // Modal only
    SoundObject(const Mesh &, vec3 listener_position);
    // RealImpact and Modal
    SoundObject(const Mesh &, const RealImpact &, const RealImpactListenerPoint &);

    ~SoundObject();

    vec3 ListenerPosition;
    MaterialProperties Material{MaterialPresets.at(DefaultMaterialPresetName)};
    std::vector<int> ExcitableVertexIndices;

    std::optional<SoundObjectData::RealImpact> RealImpactData{};
    std::optional<SoundObjectData::Modal> ModalData{};

    void SetModel(SoundObjectModel);

    void ProduceAudio(DeviceData, float *input, float *output, uint frame_count) override;

    void Strike(float force = 1.0);
    void RenderControls();

private:
    SoundObjectModel Model{SoundObjectModel::RealImpact};
};
