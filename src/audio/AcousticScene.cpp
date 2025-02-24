#include "AcousticScene.h"
#include "Excitable.h"
#include "FaustDSP.h"
#include "FaustGenerator.h"
#include "RealImpact.h"
#include "Registry.h"
#include "Scene.h"
#include "SoundObject.h"
#include "Widgets.h" // imgui
#include "Worker.h"
#include "mesh/Mesh.h"
#include "mesh/Primitives.h"

#include <entt/entity/registry.hpp>

#include <format>
#include <ranges>

using std::ranges::to;

// If an entity has this component, it is being listened to by `Listener`.
struct SoundObjectListener {
    entt::entity Listener;
};
struct SoundObjectListenerPoint {
    uint Index; // Index in the root listener point's children.
};

AcousticScene::AcousticScene(entt::registry &r, CreateSvgResource create_svg)
    : R(r), CreateSvg(std::move(create_svg)), Dsp(std::make_unique<FaustDSP>(std::move(create_svg))),
      FaustGenerator(std::make_unique<::FaustGenerator>(r, [this](std::string_view code) { Dsp->SetCode(code); })) {
    R.on_construct<ExcitedVertex>().connect<[](entt::registry &r, entt::entity entity) {
        if (auto *sound_object = r.try_get<SoundObject>(entity)) {
            const auto &excited_vertex = r.get<const ExcitedVertex>(entity);
            sound_object->SetVertex(excited_vertex.Vertex);
            sound_object->SetVertexForce(excited_vertex.Force);
        }
    }>();
    R.on_destroy<ExcitedVertex>().connect<[](entt::registry &r, entt::entity entity) {
        if (auto *sound_object = r.try_get<SoundObject>(entity)) {
            sound_object->SetVertexForce(0.f);
        }
    }>();
}
AcousticScene::~AcousticScene() = default;

void AcousticScene::LoadRealImpact(const fs::path &directory, Scene &scene) const {
    if (!fs::exists(directory)) throw std::runtime_error(std::format("RealImpact directory does not exist: {}", directory.string()));

    scene.ClearMeshes();
    const auto object_entity = scene.AddMesh(
        directory / "transformed.obj",
        {
            .Name = *RealImpact::FindObjectName(directory),
            // RealImpact meshes are oriented with Z up, but MeshEditor uses Y up.
            .Rotation = glm::angleAxis(-float(M_PI_2), vec3{1, 0, 0}) * glm::angleAxis(float(M_PI), vec3{0, 0, 1}),
        }
    );

    std::vector<uint> vertex_indices(RealImpact::NumImpactVertices);
    {
        auto impact_positions = RealImpact::LoadPositions(directory);
        // RealImpact npy file has vertex indices, but the indices may have changed due to deduplication,
        // so we don't even load them. Instead, we look up by position here.
        const auto &mesh = R.get<Mesh>(object_entity);
        for (uint i = 0; i < impact_positions.size(); ++i) {
            vertex_indices[i] = uint(mesh.FindNearestVertex(ToOpenMesh(impact_positions[i])).idx());
        }
    }

    const auto listener_entity = scene.AddMesh(
        Cylinder(0.5f * RealImpact::MicWidthMm / 1000.f, RealImpact::MicLengthMm / 1000.f),
        {
            .Name = std::format("RealImpact Listeners: {}", R.get<Name>(object_entity).Value),
            .Select = false,
            .Visible = false,
        }
    );
    for (const auto &listener_point : RealImpact::LoadListenerPoints(directory)) {
        static const auto rot_z = glm::angleAxis(float(M_PI_2), vec3{0, 0, 1}); // Cylinder is oriended with center along the Y axis.
        const auto listener_instance_entity = scene.AddInstance(
            listener_entity,
            {
                .Name = std::format("RealImpact Listener: {}", listener_point.Index),
                .Position = listener_point.GetPosition(scene.World.Up, true),
                .Rotation = glm::angleAxis(glm::radians(float(listener_point.AngleDeg)), scene.World.Up) * rot_z,
                .Select = false,
            }
        );
        R.emplace<SoundObjectListenerPoint>(listener_instance_entity, listener_point.Index);

        static constexpr uint CenterListenerIndex = 263; // This listener point is roughly centered.
        if (listener_point.Index == CenterListenerIndex) {
            R.emplace<SoundObjectListener>(object_entity, listener_instance_entity);

            static const auto FindMaterial = [](std::string_view name) -> std::optional<AcousticMaterial> {
                for (const auto &material : materials::acoustic::All) {
                    if (material.Name == name) return material;
                }
                return {};
            };
            auto material_name = RealImpact::FindMaterialName(R.get<Name>(object_entity).Value);
            const auto real_impact_material = material_name ? FindMaterial(*material_name) : std::nullopt;
            auto material = real_impact_material ? *real_impact_material : materials::acoustic::All.front();
            auto &sound_object = AddSoundObject(object_entity, std::move(material));
            sound_object.SetImpactFrames(to<std::vector>(RealImpact::LoadSamples(directory, listener_point.Index)), std::move(vertex_indices));
            R.emplace<Excitable>(object_entity, sound_object.GetExcitable());
        }
    }
}

void AcousticScene::ProduceAudio(AudioBuffer buffer) const {
    Dsp->Compute(buffer.FrameCount, &buffer.Input, &buffer.Output);
    for (const auto &audio_source : R.storage<SoundObject>()) {
        audio_source.ProduceAudio(buffer);
    }
}

SoundObject &AcousticScene::AddSoundObject(entt::entity entity, AcousticMaterial material) const {
    R.emplace<Frozen>(entity);
    return R.emplace<SoundObject>(entity, std::move(material), *Dsp);
}

using namespace ImGui;

void AcousticScene::RenderControls(Scene &scene) {
    static const float CharWidth = CalcTextSize("A").x;

    const auto selected_entity = scene.GetSelectedEntity();
    if (!R.storage<SoundObject>().empty() && CollapsingHeader("Sound objects")) {
        if (MeshEditor::BeginTable("Sound objects", 3)) {
            TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
            TableSetupColumn("Name");
            TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, CharWidth * 20);
            TableHeadersRow();
            entt::entity entity_to_select = entt::null, entity_to_delete = entt::null;
            for (const auto &[entity, sound_object] : R.view<const SoundObject>().each()) {
                const bool is_selected = entity == selected_entity;
                PushID(uint(entity));
                TableNextColumn();
                AlignTextToFramePadding();
                if (is_selected) TableSetBgColor(ImGuiTableBgTarget_RowBg0, GetColorU32(ImGuiCol_TextSelectedBg));
                TextUnformatted(IdString(entity).c_str());
                TableNextColumn();
                TextUnformatted(R.get<Name>(entity).Value.c_str());
                TableNextColumn();
                if (is_selected) BeginDisabled();
                if (Button("Select")) entity_to_select = entity;
                if (is_selected) EndDisabled();
                SameLine();
                if (Button("Delete")) entity_to_delete = entity;
                if (const auto *sound_listener = R.try_get<SoundObjectListener>(entity)) {
                    if (Button("Select listener point")) entity_to_select = sound_listener->Listener;
                }
                PopID();
            }
            if (entity_to_select != entt::null) scene.SelectEntity(entity_to_select);
            if (entity_to_delete != entt::null) scene.DestroyEntity(entity_to_delete);
            EndTable();
        }
    }
    if (!R.storage<SoundObjectListenerPoint>().empty() && CollapsingHeader("Listener points")) {
        if (MeshEditor::BeginTable("Listener points", 3)) {
            TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, CharWidth * 10);
            TableSetupColumn("Name");
            TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, CharWidth * 16);
            TableHeadersRow();
            entt::entity entity_to_select = entt::null, entity_to_delete = entt::null;
            for (const auto entity : R.view<SoundObjectListenerPoint>()) {
                const bool is_selected = entity == selected_entity;
                PushID(uint(entity));
                TableNextColumn();
                AlignTextToFramePadding();
                if (is_selected) TableSetBgColor(ImGuiTableBgTarget_RowBg0, GetColorU32(ImGuiCol_TextSelectedBg));
                TextUnformatted(IdString(entity).c_str());
                TableNextColumn();
                TextUnformatted(R.get<Name>(entity).Value.c_str());
                TableNextColumn();
                if (is_selected) BeginDisabled();
                if (Button("Select")) entity_to_select = entity;
                if (is_selected) EndDisabled();
                SameLine();
                if (Button("Delete")) entity_to_delete = entity;
                PopID();
            }
            if (entity_to_select != entt::null) scene.SelectEntity(entity_to_select);
            if (entity_to_delete != entt::null) scene.DestroyEntity(entity_to_delete);
            EndTable();
        }
    }
    if (selected_entity == entt::null) {
        TextUnformatted("No selection");
        return;
    }

    // Display the selected sound object (which could be the object listened to if a listener is selected).
    const auto FindSelectedSoundEntity = [&]() -> entt::entity {
        if (R.all_of<SoundObject>(selected_entity)) return selected_entity;
        if (R.storage<SoundObjectListener>().empty()) return entt::null;
        for (const auto &[entity, listener] : R.view<const SoundObjectListener>().each()) {
            if (listener.Listener == selected_entity) return entity;
        }
        if (R.all_of<SoundObjectListenerPoint>(selected_entity)) return *R.view<const SoundObject>().begin();
        return entt::null;
    };
    const auto sound_entity = FindSelectedSoundEntity();
    if (sound_entity == entt::null) {
        if (Button("Create audio model")) AddSoundObject(selected_entity, materials::acoustic::All.front());
        return;
    }

    SeparatorText("Audio model");

    if (sound_entity != selected_entity && Button("Select sound object")) {
        scene.SelectEntity(sound_entity);
    }

    const auto *listener = R.try_get<SoundObjectListener>(sound_entity);
    if (listener && listener->Listener != selected_entity) {
        if (Button("Select listener point")) {
            scene.SelectEntity(listener->Listener);
        }
    }

    auto &sound_object = R.get<SoundObject>(sound_entity);
    if (const auto *listener_point = R.try_get<SoundObjectListenerPoint>(selected_entity);
        listener_point && (!listener || selected_entity != listener->Listener)) {
        if (Button("Set listener point")) {
            sound_object.SetImpactFrames(to<std::vector>(RealImpact::LoadSamples(R.get<Path>(sound_entity).Value.parent_path(), listener_point->Index)));
            R.emplace_or_replace<SoundObjectListener>(sound_entity, selected_entity);
        }
    }

    sound_object.RenderControls(R, sound_entity);
    Spacing();
    if (Button("Remove audio model")) {
        R.remove<SoundObject, SoundObjectListener, Excitable, Frozen>(sound_entity);
    }
}
