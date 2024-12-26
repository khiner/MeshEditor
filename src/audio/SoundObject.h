#pragma once

#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "AudioSource.h"
#include "MaterialProperties.h"
#include "numeric/vec3.h"

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

struct Tets;
template<typename Result> struct Worker;

struct ModalAudioModel;
struct ImpactAudioModel;
struct Mesh2FaustResult;

// Represents a rigid mesh object that generate an audio stream for a listener at a given position
// in response to an impact at a given vertex.
struct SoundObject : AudioSource {
    // All SoundObjects have a modal audio model.
    // If `impact_frames_by_vertex` is non-empty, the object also has an impact audio model.
    SoundObject(std::string_view name, const Tets &, const std::optional<std::string_view> &material_name, vec3 listener_position, uint listener_entity_id);
    ~SoundObject();

    const std::string Name;
    const Tets &Tets;
    std::string_view MaterialName;
    MaterialProperties Material;
    vec3 ListenerPosition;
    uint ListenerEntityId{0};
    std::vector<uint> ExcitableVertices;
    uint CurrentVertex{0}, CurrentVertexIndicatorEntityId{0};

    void ProduceAudio(DeviceData, float *input, float *output, uint frame_count) override;
    void RenderControls();

    void SetModel(SoundObjectModel);
    void SetImpactFrames(std::unordered_map<uint, std::vector<float>> &&impact_frames_by_vertex);

    void SetVertex(uint);
    void SetVertexForce(float);
    std::optional<uint> FindNearestExcitableVertex(vec3 position);

private:
    std::unique_ptr<ModalAudioModel> ModalModel;
    std::unique_ptr<ImpactAudioModel> ImpactModel;

    SoundObjectModel Model{SoundObjectModel::ImpactAudio};
    std::unique_ptr<Worker<Mesh2FaustResult>> DspGenerator;
};
