#pragma once

#include "AcousticMaterial.h"

#include <entt/entity/fwd.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

enum class SoundObjectModel {
    // Play back recordings of impacts on the object at provided listener points/vertices.
    ImpactAudio,
    // Model a rigid body's response to an impact using modal analysis/synthesis:
    // - transforming the object's geometry into a tetrahedral volume mesh
    // - using FEM to estimate the mass/sg/damping matrices from the mesh
    // - estimating the dominant mode frequencies/amplitudes/T60s using an eigenvalues/eigenvector solver
    // - simulating the object's response to an impact by exciting the modes associated with the impacted vertex
    Modal,
};

struct AudioBuffer;
struct Excitable;
struct FaustDSP;

struct ModalAudioModel;
struct ImpactAudioModel;

// Represents a rigid body that generates an audio stream in response to an impact at a given vertex.
struct SoundObject {
    SoundObject(AcousticMaterial, FaustDSP &);
    ~SoundObject();

    void ProduceAudio(AudioBuffer &, entt::registry &, entt::entity) const;
    void RenderControls(entt::registry &, entt::entity);

    void SetVertex(uint);
    void SetVertexForce(float);

    void SetImpactFrames(std::vector<std::vector<float>> &&impact_frames, std::vector<uint> &&vertex_indices);
    void SetImpactFrames(std::vector<std::vector<float>> &&impact_frames);

private:
    void SetModel(SoundObjectModel, entt::registry &, entt::entity);

    // 1:1 with ImGui controls, cached per-object
    struct ControlState {
        AcousticMaterial Material{materials::acoustic::All.front()};
        uint32_t NumExcitableVertices{10};
        bool UseImpactVertices{true}; // Only used in ImpactAudio mode
        bool QualityTets{false};
    };

    FaustDSP &Dsp;
    ControlState Controls{};
    std::unique_ptr<ModalAudioModel> ModalModel;
    std::unique_ptr<ImpactAudioModel> ImpactModel;
    std::optional<std::vector<uint>> ImpactVertices;

    SoundObjectModel Model{SoundObjectModel::ImpactAudio};
};
