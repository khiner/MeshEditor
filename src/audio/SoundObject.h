#pragma once

#include "CreateSvgResource.h"

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
struct Mesh2FaustResult;

struct ModalAudioModel;
struct ImpactAudioModel;

template<typename Result> struct Worker;

// Represents a rigid body that generates an audio stream in response to an impact at a given vertex.
struct SoundObject {
    SoundObject(CreateSvgResource);
    ~SoundObject();

    void ProduceAudio(AudioBuffer &) const;
    void RenderControls(entt::registry &, entt::entity);

    const Excitable &GetExcitable() const;
    void SetVertex(uint);
    void SetVertexForce(float);

    void SetImpactFrames(std::vector<std::vector<float>> &&impact_frames, std::vector<uint> &&vertex_indices);
    void SetImpactFrames(std::vector<std::vector<float>> &&impact_frames);

private:
    void SetModel(SoundObjectModel, entt::registry &, entt::entity);

    std::unique_ptr<ModalAudioModel> ModalModel;
    std::unique_ptr<ImpactAudioModel> ImpactModel;

    SoundObjectModel Model{SoundObjectModel::ImpactAudio};
    std::unique_ptr<Worker<Mesh2FaustResult>> DspGenerator;
    CreateSvgResource CreateSvg;
};
