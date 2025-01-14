#include "AcousticScene.h"
#include "AcousticMaterial.h"
#include "RealImpact.h"
#include "Scene.h"
#include "SoundObject.h"
#include "Tets.h"
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

void AcousticScene::LoadRealImpact(const fs::path &directory, entt::registry &r, Scene &scene, CreateSvgResource create_svg) {
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

    static const auto FindMaterial = [](std::string_view name) -> std::optional<AcousticMaterial> {
        for (const auto &material : materials::acoustic::All) {
            if (material.Name == name) return material;
        }
        return {};
    };
    auto material_name = RealImpact::FindMaterialName(r.get<Name>(object_entity).Value);
    const auto real_impact_material = material_name ? FindMaterial(*material_name) : std::nullopt;
    r.emplace<AcousticMaterial>(object_entity, real_impact_material ? *real_impact_material : materials::acoustic::All.front());

    std::vector<uint> vertex_indices(RealImpact::NumImpactVertices);
    {
        auto impact_positions = RealImpact::LoadPositions(directory);
        // RealImpact npy file has vertex indices, but the indices may have changed due to deduplication,
        // so we don't even load them. Instead, we look up by position here.
        const auto &mesh = r.get<Mesh>(object_entity);
        for (uint i = 0; i < impact_positions.size(); ++i) {
            vertex_indices[i] = uint(mesh.FindNearestVertex(impact_positions[i]).idx());
        }
    }

    const auto listener_entity = scene.AddMesh(
        Cylinder(0.5f * RealImpact::MicWidthMm / 1000.f, RealImpact::MicLengthMm / 1000.f),
        {
            .Name = std::format("RealImpact Listeners: {}", r.get<Name>(object_entity).Value),
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
        r.emplace<SoundObjectListenerPoint>(listener_instance_entity, listener_point.Index);

        static constexpr uint CenterListenerIndex = 263; // This listener point is roughly centered.
        if (listener_point.Index == CenterListenerIndex) {
            r.emplace<SoundObjectListener>(object_entity, listener_instance_entity);
            auto &sound_object = r.emplace<SoundObject>(object_entity, create_svg);
            sound_object.SetImpactFrames(to<std::vector>(RealImpact::LoadSamples(directory, listener_point.Index)), std::move(vertex_indices));
            r.emplace<Excitable>(object_entity, sound_object.GetExcitable());
        }
    }
}

AcousticScene::AcousticScene(entt::registry &r) : R(r) {}
AcousticScene::~AcousticScene() = default;

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

    static std::unique_ptr<Worker<Tets>> TetGenerator;
    if (R.all_of<Mesh>(selected_entity) && !R.all_of<Tets>(selected_entity) && !R.all_of<SoundObjectListenerPoint>(selected_entity)) {
        if (TetGenerator) {
            if (auto tets = TetGenerator->Render()) {
                // todo Add an invisible tet mesh to the scene and support toggling between surface/volumetric tet mesh views.
                // scene.AddMesh(tets->CreateMesh(), {.Name = "Tet Mesh",  R.get<Model>(selected_entity).Transform;, .Select = false, .Visible = false});
                TetGenerator.reset();
                R.emplace<Tets>(selected_entity, std::move(*tets));
            }
        } else { // todo conditionally show "Regenerate tet mesh"
            SeparatorText("Tet mesh");

            static bool quality = false;
            Checkbox("Quality", &quality);
            MeshEditor::HelpMarker("Add new Steiner points to the interior of the tet mesh to improve model quality.");
            if (Button("Create audio model")) {
                // We rely on `PreserveSurface` behavior for excitable vertices;
                // Vertex indices on the surface mesh must match vertex indices on the tet mesh.
                // todo display tet mesh in UI and select vertices for debugging (just like other meshes but restrict to edge view)
                TetGenerator = std::make_unique<Worker<Tets>>("Generating tetrahedral mesh...", [&] {
                    return Tets::Generate(R.get<Mesh>(selected_entity), {.PreserveSurface = true, .Quality = quality});
                });
            }
        }
    }

    // Display the selected sound object (which could be the object listened to if a listener is selected).
    if (R.storage<SoundObject>().empty()) return;

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
        TextUnformatted("No sound object selected");
        return;
    }

    SeparatorText("Audio model");

    auto &sound_object = R.get<SoundObject>(sound_entity);
    if (sound_entity != selected_entity && Button("Select sound object")) {
        scene.SelectEntity(sound_entity);
    }

    const auto *listener = R.try_get<SoundObjectListener>(sound_entity);
    if (listener && listener->Listener != selected_entity) {
        if (Button("Select listener point")) {
            scene.SelectEntity(listener->Listener);
        }
    }
    if (const auto *listener_point = R.try_get<SoundObjectListenerPoint>(selected_entity);
        listener_point && (!listener || selected_entity != listener->Listener)) {
        if (Button("Set listener point")) {
            sound_object.SetImpactFrames(to<std::vector>(RealImpact::LoadSamples(R.get<Path>(sound_entity).Value.parent_path(), listener_point->Index)));
            R.emplace_or_replace<SoundObjectListener>(sound_entity, selected_entity);
        }
    }

    if (auto sound_object_action = sound_object.RenderControls(
            GetName(R, sound_entity),
            R.try_get<Tets>(selected_entity),
            R.try_get<AcousticMaterial>(selected_entity)
        )) {
        std::visit(
            Match{
                [&](SoundObjectAction::SelectVertex action) {
                    sound_object.Apply(action);
                    R.remove<ExcitedVertex>(sound_entity);
                },
                [&](SoundObjectAction::SetExciteForce action) {
                    if (action.Force == 0) {
                        R.remove<ExcitedVertex>(sound_entity);
                    } else {
                        R.emplace<ExcitedVertex>(sound_entity, sound_object.GetExcitable().SelectedVertex(), action.Force);
                    }
                },
                [&](SoundObjectAction::SetModel action) {
                    sound_object.Apply(action);
                    R.emplace_or_replace<Excitable>(sound_entity, sound_object.GetExcitable());
                },
                [&](auto action) {
                    sound_object.Apply(action);
                }
            },
            *sound_object_action
        );
    }

    Spacing();
    if (Button("Remove audio model")) {
        R.remove<Excitable, SoundObjectListener, SoundObject, Tets, AcousticMaterial>(sound_entity);
    }
}
