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

namespace SoundObjectData {
struct RealImpact {
    RealImpact(std::vector<std::vector<float>> &&impact_samples)
        : ImpactSamples(std::move(impact_samples)) {
        if (!ImpactSamples.empty()) {
            // Start at the end of the first sample, so it doesn't immediately play.
            // Assume all samples have the same size.
            CurrentFrame = ImpactSamples[0].size();
        }
    }

    std::vector<std::vector<float>> ImpactSamples;
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
    // RealImpact only
    SoundObject(const RealImpact &, const RealImpactListenerPoint &);
    // Modal only
    SoundObject(const Mesh &, vec3 listener_position);
    // RealImpact and Modal
    SoundObject(const RealImpact &, const RealImpactListenerPoint &, const Mesh &);

    ~SoundObject();

    inline static const std::string DefaultMaterialPresetName = "Bell";

    vec3 ListenerPosition;
    MaterialProperties Material{MaterialPresets.at(DefaultMaterialPresetName)};

    std::optional<SoundObjectData::RealImpact> RealImpactData{};
    std::optional<SoundObjectData::Modal> ModalData{};

    void SetModel(SoundObjectModel);

    void ProduceAudio(DeviceData, float *input, float *output, uint frame_count) override;

    void Strike(float force = 1.0);
    void RenderControls();

private:
    SoundObjectModel Model{SoundObjectModel::RealImpact};
};
