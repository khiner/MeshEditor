#pragma once

#include <optional>
#include <unordered_map>
#include <vector>

#include "numeric/vec3.h"

#include "AudioSource.h"
#include "MaterialProperties.h"

enum class SoundObjectModel {
    // Play back recordings of impacts on the object at provided listener points/vertices.
    ImpactAudio,
    // Model a rigid body's response to an impact using modal analysis/synthesis:
    // - transforming the object's geometry into a tetrahedral volume mesh
    // - using FEM to estimate the mass/sg/damping matrices from the mesh
    // - estimating the dominant modal frequencies and amplitudes using an eigenvalues/eigenvector solver
    // - simulating the object's response to an impact by exciting the modes associated with the impacted vertex
    Modal,
};

struct FaustDSP;
struct Tets;
template<typename Result> struct Worker;

// All model-specific data needed to render audio.
namespace SoundObjectData {
struct ImpactAudio {
    ImpactAudio(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex)
        : ImpactFramesByVertex(std::move(impact_frames_by_vertex)) {
        // Start at the end of the first sample, so it doesn't immediately play.
        // All samples are the same length.
        if (!ImpactFramesByVertex.empty()) {
            CurrentFrame = ImpactFramesByVertex.begin()->second.size();
        }
    }

    std::unordered_map<uint, std::vector<float>> ImpactFramesByVertex;
    uint CurrentFrame{0};
};

struct Modal {
    Modal();
    ~Modal();

    std::unique_ptr<FaustDSP> FaustDsp;
};
} // namespace SoundObjectData

// Represents a rigid mesh object that generate an audio stream for a listener at a given position
// in response to an impact at a given vertex.
struct SoundObject : AudioSource {
    // All SoundObjects have a modal audio model. If `impact_frames_by_vertex` is non-empty, the object also has an impact audio model.
    SoundObject(
        const Tets &, MaterialProperties &&, vec3 listener_position, uint object_entity_id, uint listener_entity_id,
        std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex = {}
    );

    ~SoundObject();

    const Tets &Tets;
    MaterialProperties Material{MaterialPresets.at(DefaultMaterialPresetName)};
    vec3 ListenerPosition;
    uint ObjectEntityId{0}, ListenerEntityId{0};
    std::vector<uint> ExcitableVertices;
    uint CurrentVertex{0};

    std::optional<SoundObjectData::Modal> ModalData{};
    std::optional<SoundObjectData::ImpactAudio> ImpactAudioData{};

    void SetModel(SoundObjectModel);

    void ProduceAudio(DeviceData, float *input, float *output, uint frame_count) override;

    void Strike(float force = 1.0);
    void RenderControls();

private:
    SoundObjectModel Model{SoundObjectModel::ImpactAudio};
    std::unique_ptr<Worker<std::string>> DspGenerator;
};
