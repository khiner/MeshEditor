#include "numeric/VectorMath.h"
#include "numeric/vec2.h"

#include "Camera.h"
#include "Path.h"
#include "Profile.h"
#include "TransformMath.h"
#include "Variant.h"
#include "action/Audio.h"
#include "action/Bone.h"
#include "action/Object.h"
#include "action/Selection.h"
#include "action/View.h"
#include "animation/Fields.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "audio/AudioDevice.h"
#include "audio/AudioSystem.h"
#include "audio/AudioUi.h"
#include "gizmo/GizmoInteraction.h"
#include "gizmo/TransformGizmo.h"
#include "gltf/GltfScene.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "numeric/Angles.h"
#include "physics/PhysicsUi.h"
#include "render/GpuBufferOps.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MaterialComponents.h"
#include "render/PbrFeature.h"
#include "render/TextureRefs.h"
#include "scene/CameraLens.h"
#include "scene/Defaults.h"
#include "scene/Entity.h"
#include "scene/SceneControlsUi.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionGpu.h"
#include "state/Scene.h"
#include "ui/ChoiceCombo.h"
#include "ui/FieldEdit.h"
#include "ui/HelpMarker.h"
#include "ui/ItemList.h"
#include "ui/MaterialEdit.h"
#include "ui/TransformEdit.h"
#include "viewport/FrameState.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"
#include <imgui_internal.h>

#include <format>

using numeric::Degrees;

using std::ranges::any_of, std::ranges::distance, std::ranges::find, std::ranges::to;

using namespace ImGui;

static void RenderObjectTree(state::Scene &, state::Entity viewport);
static void RenderEntityControls(state::Scene &, state::Entity viewport, state::Entity active_entity);

namespace {
constexpr std::string_view ObjectTypeName(ObjectType type) {
    switch (type) {
        case ObjectType::Empty: return "Empty";
        case ObjectType::Mesh: return "Mesh";
        case ObjectType::Armature: return "Armature";
        case ObjectType::Camera: return "Camera";
        case ObjectType::Light: return "Light";
    }
}

const std::vector<Element> NormalElements{Element::Vertex, Element::Face};

bool SliderUInt(const char *label, uint32_t *v, uint32_t v_min, uint32_t v_max, const char *format = nullptr, ImGuiSliderFlags flags = 0) {
    return ImGui::SliderScalar(label, ImGuiDataType_U32, v, &v_min, &v_max, format, flags);
}

constexpr std::string Capitalize(std::string_view str) {
    if (str.empty()) return {};

    std::string result{str};
    char &c = result[0];
    if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
    return result;
}

// Draw the active primitive's fields, emitting a gesture-grouped update per field on change.
void PrimitiveEditor(const PrimitiveShape &shape) {
    using primitive::MaxSize, primitive::MinSize;
    static constexpr float SizeSpeed = 0.01f;

    // The member pointer addresses the field within the shape alternative.
    const auto field = [&]<typename C, typename F>(bool changed, F C::*member, F value) {
        ui::Gesture(changed, [=] { return action::Update<F>{ui::TargetFromAlt(true), state::Key<PrimitiveShape>(), uint16_t(action::detail::MemPtrOffset(member)), value}; });
    };
    std::visit([&](const auto &s) {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, primitive::Plane>) {
            vec2 size = s.HalfExtents * 2.f;
            field(ui::DragFloat2("Size", &size.x, SizeSpeed, MinSize, MaxSize), &primitive::Plane::HalfExtents, size / 2.f);
        } else if constexpr (std::is_same_v<T, primitive::Circle>) {
            float radius = s.Radius;
            field(ui::DragFloat("Radius", &radius, SizeSpeed, MinSize, MaxSize), &primitive::Circle::Radius, radius);
            uint32_t segments = s.Segments;
            field(SliderUInt("Segments", &segments, 3, 128), &primitive::Circle::Segments, segments);
        } else if constexpr (std::is_same_v<T, primitive::Cuboid>) {
            vec3 size = s.HalfExtents * 2.f;
            field(ui::DragFloat3("Size", &size.x, SizeSpeed, MinSize, MaxSize), &primitive::Cuboid::HalfExtents, size / 2.f);
        } else if constexpr (std::is_same_v<T, primitive::IcoSphere>) {
            float radius = s.Radius;
            field(ui::DragFloat("Radius", &radius, SizeSpeed, MinSize, MaxSize), &primitive::IcoSphere::Radius, radius);
            uint32_t subdivisions = s.Subdivisions;
            field(SliderUInt("Subdivisions", &subdivisions, 1, 6), &primitive::IcoSphere::Subdivisions, subdivisions);
        } else if constexpr (std::is_same_v<T, primitive::UVSphere>) {
            float radius = s.Radius;
            field(ui::DragFloat("Radius", &radius, SizeSpeed, MinSize, MaxSize), &primitive::UVSphere::Radius, radius);
            uint32_t slices = s.Slices, stacks = s.Stacks;
            field(SliderUInt("Slices", &slices, 3, 128), &primitive::UVSphere::Slices, slices);
            field(SliderUInt("Stacks", &stacks, 2, 64), &primitive::UVSphere::Stacks, stacks);
        } else if constexpr (std::is_same_v<T, primitive::Torus>) {
            float major = s.MajorRadius, minor = s.MinorRadius;
            field(ui::DragFloat("Major radius", &major, SizeSpeed, MinSize, MaxSize), &primitive::Torus::MajorRadius, major);
            field(ui::DragFloat("Minor radius", &minor, SizeSpeed, MinSize, s.MajorRadius), &primitive::Torus::MinorRadius, minor);
            uint32_t major_seg = s.MajorSegments, minor_seg = s.MinorSegments;
            field(SliderUInt("Major segments", &major_seg, 3, 256), &primitive::Torus::MajorSegments, major_seg);
            field(SliderUInt("Minor segments", &minor_seg, 3, 256), &primitive::Torus::MinorSegments, minor_seg);
        } else if constexpr (std::is_same_v<T, primitive::Cylinder> || std::is_same_v<T, primitive::Cone>) {
            float radius = s.Radius, height = s.Height;
            field(ui::DragFloat("Radius", &radius, SizeSpeed, MinSize, MaxSize), &T::Radius, radius);
            field(ui::DragFloat("Height", &height, SizeSpeed, MinSize, MaxSize), &T::Height, height);
            uint32_t slices = s.Slices;
            field(SliderUInt("Slices", &slices, 3, 128), &T::Slices, slices);
        }
    },
               shape);
}

std::string to_string(InteractionMode mode) {
    switch (mode) {
        case InteractionMode::Object: return "Object";
        case InteractionMode::Edit: return "Edit";
        case InteractionMode::Excite: return "Excite";
        case InteractionMode::Pose: return "Pose";
    }
}

using namespace he;

// Lens fields shared by the camera object editor and the view camera editor.
// Near and far clip bound each other.
void RenderPerspectiveFields(auto &&fields, const Perspective &perspective) {
    const float far_max = std::max(perspective.NearClip + MinNearFarDelta, MaxFarClip);
    fields.template Slider<&Perspective::FieldOfViewRad>("Field of view");
    fields.template Slider<&Perspective::NearClip>("Near clip", MinNearClip, perspective.HasFarClip() ? std::max(perspective.FarClip - MinNearFarDelta, MinNearClip) : far_max);
    bool infinite_far = !perspective.HasFarClip();
    if (Checkbox("Infinite far clip", &infinite_far)) fields.template Set<&Perspective::FarClip>(infinite_far ? std::numeric_limits<float>::infinity() : far_max);
    if (perspective.HasFarClip()) fields.template Slider<&Perspective::FarClip>("Far clip", perspective.NearClip + MinNearFarDelta, far_max);
}
void RenderOrthographicFields(auto &&fields, const Orthographic &orthographic) {
    fields.template Slider<&Orthographic::Mag, &vec2::x>("X Mag");
    fields.template Slider<&Orthographic::Mag, &vec2::y>("Y Mag");
    fields.template Slider<&Orthographic::NearClip>("Near clip", MinNearClip, std::max(orthographic.FarClip - MinNearFarDelta, MinNearClip));
    fields.template Slider<&Orthographic::FarClip>("Far clip", orthographic.NearClip + MinNearFarDelta, std::max(orthographic.NearClip + MinNearFarDelta, MaxFarClip));
}

// Edits the view camera's lens value. `viewport_aspect` is the aspect of the viewport the camera renders.
bool RenderCameraLensEditor(CameraLens &camera, float distance, float viewport_aspect) {
    bool changed = false;
    int proj_i = std::holds_alternative<Orthographic>(camera) ? 1 : 0;
    const char *const proj_names[]{"Perspective", "Orthographic"};
    if (Combo("Projection", &proj_i, proj_names, IM_ARRAYSIZE(proj_names)) && proj_i != int(camera.index())) {
        if (proj_i == 0) camera = PerspectiveFromOrthographic(std::get<Orthographic>(camera), distance);
        else camera = OrthographicFromPerspective(std::get<Perspective>(camera), distance, viewport_aspect);
        changed = true;
    }
    if (auto *perspective = std::get_if<Perspective>(&camera)) RenderPerspectiveFields(ui::ValueEdit{*perspective, changed}, *perspective);
    else RenderOrthographicFields(ui::ValueEdit{std::get<Orthographic>(camera), changed}, std::get<Orthographic>(camera));
    return changed;
}

// Edits a camera object's lens component through field updates.
void RenderCameraLensFields(state::Scene &r, state::Entity entity) {
    const auto *perspective = r.try_get<const Perspective>(entity);
    int proj_i = perspective ? 0 : 1;
    const char *const proj_names[]{"Perspective", "Orthographic"};
    if (Combo("Projection", &proj_i, proj_names, IM_ARRAYSIZE(proj_names))) action::Emit(action::object::SetProjection{proj_i == 1, ui::TargetFromAlt()});
    ui::Edit fields{r, entity};
    if (perspective) {
        RenderPerspectiveFields(fields, *perspective);
        bool viewport_aspect = !perspective->HasAspectRatio();
        if (Checkbox("Viewport aspect ratio", &viewport_aspect)) fields.Set<&Perspective::AspectRatio>(viewport_aspect ? 0.f : DefaultAspectRatio);
        if (perspective->HasAspectRatio()) fields.Slider<&Perspective::AspectRatio>();
    } else {
        RenderOrthographicFields(fields, r.get<const Orthographic>(entity));
    }
}

std::string NamedOr(const std::string &name, std::string_view fallback, uint32_t i) {
    return name.empty() ? std::format("{}{}", fallback, i) : name;
}

constexpr std::string_view MimeTypeName(gltf::MimeType m) {
    using gltf::MimeType;
    switch (m) {
        case MimeType::None: return "—";
        case MimeType::JPEG: return "image/jpeg";
        case MimeType::PNG: return "image/png";
        case MimeType::KTX2: return "image/ktx2";
        case MimeType::DDS: return "image/vnd-ms.dds";
        case MimeType::GltfBuffer: return "model/gltf-buffer";
        case MimeType::OctetStream: return "application/octet-stream";
        case MimeType::WEBP: return "image/webp";
    }
    return "?";
}

std::string AttributeFlagsString(uint32_t flags) {
    std::string s;
    const auto add = [&](std::string_view tag) {
        if (!s.empty()) s += '|';
        s += tag;
    };
    add("POS"); // Always present.
    if (flags & MeshAttributeBit_Normal) add("NRM");
    if (flags & MeshAttributeBit_Tangent) add("TAN");
    if (flags & MeshAttributeBit_Color0) add("COL0");
    if (flags & MeshAttributeBit_TexCoord0) add("UV0");
    if (flags & MeshAttributeBit_TexCoord1) add("UV1");
    if (flags & MeshAttributeBit_TexCoord2) add("UV2");
    if (flags & MeshAttributeBit_TexCoord3) add("UV3");
    return s;
}

void RenderJsonBlock(const char *label, std::string_view json) {
    SeparatorText(label);
    PushID(label);
    const float row_h = GetTextLineHeightWithSpacing();
    InputTextMultiline(
        "##json", const_cast<char *>(json.data()), json.size() + 1,
        ImVec2{-FLT_MIN, row_h * 6.f}, ImGuiInputTextFlags_ReadOnly
    );
    if (SmallButton("Copy")) SetClipboardText(std::string{json}.c_str());
    PopID();
}

constexpr auto MetadataTableFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;

// Metadata table scaffolding: header columns, then `per_row(i)` fills each row's cells.
void MetaTable(const char *id, std::initializer_list<const char *> cols, size_t rows, auto &&per_row) {
    if (!BeginTable(id, int(cols.size()), MetadataTableFlags)) return;
    for (const auto *col : cols) TableSetupColumn(col);
    TableHeadersRow();
    for (size_t i = 0; i < rows; ++i) {
        TableNextRow();
        per_row(i);
    }
    EndTable();
}
} // namespace

static void RenderEntityControls(state::Scene &r, state::Entity viewport, state::Entity active_entity) {
    auto &meshes = r.Context.get<MeshStore>();
    if (active_entity == state::Null) {
        TextUnformatted("Active object: None");
        return;
    }

    PushID("EntityControls");
    Text("Active entity: %s", GetName(r, active_entity).c_str());
    Indent();

    if (const auto *node = r.try_get<SceneNode>(active_entity)) {
        if (auto parent_entity = node->Parent; parent_entity != state::Null) {
            AlignTextToFramePadding();
            Text("Parent: %s", GetName(r, parent_entity).c_str());
        }
    }

    if (const auto *instance = r.try_get<Instance>(active_entity)) {
        Text("Instance of: %s", GetName(r, instance->Entity).c_str());
    }
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(active_entity)) {
        Text("Armature data: %s", GetName(r, armature_modifier->ArmatureEntity).c_str());
        if (armature_modifier->ArmatureObjectEntity != state::Null) {
            Text("Armature object: %s", GetName(r, armature_modifier->ArmatureObjectEntity).c_str());
        }
    }
    if (const auto *bone_attachment = r.try_get<BoneAttachment>(active_entity)) {
        Text("Attached bone ID: %u", bone_attachment->Bone);
    }
    const auto object_type = r.all_of<ObjectKind>(active_entity) ? r.get<const ObjectKind>(active_entity).Value : ObjectType::Empty;
    Text("Object type: %s", ObjectTypeName(object_type).data());
    const auto active_bone_entity = FindActiveBone(r);
    const auto *active_instance = r.try_get<Instance>(active_entity);
    const bool is_mesh_instance = active_instance && HasMesh(r, active_instance->Entity);
    if (is_mesh_instance) {
        const auto active_mesh_entity = active_instance->Entity;
        const auto &active_mesh = GetMesh(r, active_mesh_entity);
        TextUnformatted(
            std::format("Vertices | Edges | Faces: {:L} | {:L} | {:L}", active_mesh.VertexCount(), active_mesh.EdgeCount(), active_mesh.FaceCount()).c_str()
        );
    } else if (const auto *armature_object = r.try_get<ArmatureObject>(active_entity)) {
        const auto &armature = r.get<const Armature>(armature_object->Entity);
        Text("Bones: %zu", armature.Bones.size());
    }
    Unindent();
    const bool is_bone_edit = r.get<const Interaction>(viewport).Mode == InteractionMode::Edit && active_bone_entity != state::Null && r.all_of<BoneDisplayScale>(active_bone_entity);
    if (CollapsingHeader("Transform")) {
        if (is_bone_edit) {
            const auto &wt = r.get<WorldTransform>(active_bone_entity);
            const float bone_length = r.get<BoneDisplayScale>(active_bone_entity).Value;

            vec3 head = wt.P;
            vec3 tail = head + Rotate(wt.R, vec3{0, bone_length, 0});
            vec3 dir;
            float roll;
            BoneMat3ToVecRoll(ToMat3(wt.R), dir, roll);
            float roll_deg = Degrees(roll);
            float length = bone_length;

            bool changed = ui::DragFloat3("Head", &head[0], 0.01f);
            changed |= ui::DragFloat3("Tail", &tail[0], 0.01f);
            if (ui::DragFloat("Roll", &roll_deg, 1.f)) {
                roll = Radians(roll_deg);
                changed = true;
            }
            if (ui::DragFloat("Length", &length, 0.01f, 0.001f, 0.f)) {
                tail = head + Normalize(tail - head) * std::max(length, 1e-4f);
                changed = true;
            }

            if (changed) {
                const auto new_dir = tail - head;
                if (const auto new_length = Length(new_dir); new_length > 1e-6f) {
                    const auto new_rot = ToQuat(BoneVecRollToMat3(new_dir, roll));
                    const auto pd = ToTransform(GetParentDelta(r, active_bone_entity));
                    action::Emit(action::bone::SetEditHeadTailRoll{
                        .LocalP = Conjugate(pd.R) * ((head - pd.P) / pd.S),
                        .LocalR = Conjugate(pd.R) * new_rot,
                        .DisplayScale = new_length,
                    });
                }
            }
        } else {
            // In Pose mode, edit the active bone rather than the armature.
            const bool is_pose_bone = r.get<const Interaction>(viewport).Mode == InteractionMode::Pose && active_bone_entity != state::Null;
            const auto transform_entity = is_pose_bone ? active_bone_entity : active_entity;
            // A bone edits its rest-relative delta, and an animated node edits its pose.
            const auto draw = [&](auto edit) { ui::DrawEditor(edit, std::type_identity<Transform>{}, r.all_of<ScaleLocked>(transform_entity)); };
            if (r.all_of<BoneDelta>(transform_entity)) draw(ui::Edit{r}.Sub<&BoneDelta::Value>());
            else if (r.all_of<PosedLocal>(transform_entity)) draw(ui::Edit{r}.Sub<&PosedLocal::Value>());
            else draw(ui::Edit{r});
        }
        Spacing();
        {
            AlignTextToFramePadding();
            Text("Mode:");
            SameLine();
            using enum TransformGizmo::Mode;
            ui::Edit gizmo_edit{r, viewport};
            const auto &gizmo_state = r.get<const TransformGizmoState>(viewport);
            if (RadioButton("Local", gizmo_state.Mode == Local)) gizmo_edit.Set<&TransformGizmoState::Mode>(Local);
            SameLine();
            if (RadioButton("World", gizmo_state.Mode == World)) gizmo_edit.Set<&TransformGizmoState::Mode>(World);
            Spacing();
            gizmo_edit.Check<&TransformGizmoState::Config, &TransformGizmo::Config::Snap>();
            if (gizmo_state.Config.Snap) {
                SameLine();
                // todo link/unlink snap values
                gizmo_edit.Drag<&TransformGizmoState::Config, &TransformGizmo::Config::SnapValue>("Snap");
            }
        }
        Spacing();
        if (TreeNode("Debug")) {
            if (const auto label = TransformGizmo::ToString(r.get<const GizmoInteraction>(viewport)); !label.empty()) {
                Text("%s op: %s", r.get<const GizmoInteraction>(viewport).IsUsing() ? "Active" : "Hovered", label.data());
            } else {
                TextUnformatted("Not hovering");
            }
            TreePop();
        }
        if (TreeNode("World transform")) {
            const auto &wt = r.get<WorldTransform>(active_entity);
            Text("Position: %.3f, %.3f, %.3f", wt.P.x, wt.P.y, wt.P.z);
            Text("Rotation: %.3f, %.3f, %.3f, %.3f", wt.R.x, wt.R.y, wt.R.z, wt.R.w);
            Text("Scale: %.3f, %.3f, %.3f", wt.S.x, wt.S.y, wt.S.z);
            TreePop();
        }
    }
    if (active_bone_entity != state::Null && CollapsingHeader("Bone Constraints")) {
        PushID("BoneConstraints");
        const auto *constraints = r.try_get<const BoneConstraints>(active_bone_entity);
        const auto deleted = ui::ItemList(
            constraints ? constraints->Stack.size() : 0,
            [&](uint32_t i) {
                return std::visit([]<typename T>(const T &) {
                    if constexpr (std::is_same_v<T, CopyTransformsData>) return "Copy Transforms";
                    else if constexpr (std::is_same_v<T, ChildOfData>) return "Child Of";
                    else return "?";
                },
                                  constraints->Stack[i].Data);
            },
            [&](uint32_t i) {
                const auto &c = constraints->Stack[i];
                std::vector<state::Entity> targets{state::Null};
                for (const auto te : r.view<const ObjectKind, const Name>())
                    if (!r.any_of<BoneIndex, BoneSubPartOf, BoneJoint, SubElementOf>(te)) targets.push_back(te);
                const auto target_name = [&](state::Entity e) {
                    const auto *name = e != state::Null && r.valid(e) ? r.try_get<const Name>(e) : nullptr;
                    return e == state::Null ? std::string{"None"} : name && !name->Value.empty() ? name->Value :
                                                                                                   IdString(e);
                };
                ui::ChoiceCombo("Target", c.TargetEntity, targets, target_name, [&](state::Entity te) { action::Emit(action::bone::SetConstraintTarget{i, te}); });
                if (std::holds_alternative<ChildOfData>(c.Data)) {
                    if (Button("Set Inverse") && c.TargetEntity != state::Null && r.valid(c.TargetEntity))
                        action::Emit(action::bone::BakeConstraintChildOfInverse{i});
                    SameLine();
                    if (Button("Clear Inverse"))
                        action::Emit(action::bone::ClearConstraintChildOfInverse{i});
                }
                if (float influence = c.Influence; SliderFloat("Influence", &influence, 0.f, 1.f))
                    action::Emit(action::bone::SetConstraintInfluence{i, influence});
            }
        );
        if (deleted) action::Emit(action::bone::DeleteConstraint{*deleted});
        if (Button("Add Copy Transforms")) action::Emit(action::bone::AddConstraint{action::bone::BoneConstraintKind::CopyTransforms});
        SameLine();
        if (Button("Add Child Of")) action::Emit(action::bone::AddConstraint{action::bone::BoneConstraintKind::ChildOf});
        PopID();
    }
    if (is_mesh_instance) {
        const auto active_mesh_entity = active_instance->Entity;
        if (auto *prim_shape = r.try_get<PrimitiveShape>(active_mesh_entity)) {
            const bool frozen = selection::HasScaleLockedInstance(r, active_mesh_entity);
            if (frozen) BeginDisabled();
            if (const auto update_label = std::format("Edit primitive{}", frozen ? " (frozen)" : "");
                CollapsingHeader(update_label.c_str()) && !frozen) {
                PrimitiveEditor(*prim_shape);
            }
            if (frozen) EndDisabled();
        }

        if (CollapsingHeader("Material")) {
            const auto &active_mesh = GetMesh(r, active_mesh_entity);
            auto &material_store = r.Context.get<MaterialStore>();
            const auto texture_refs = GetTextureRefs(r);
            const std::span<const uint32_t> primitive_materials = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(active_mesh.GetStoreId()).PrimitiveMaterials);
            const auto materials = GetMaterials(r);
            const auto material_count = uint32_t(materials.size());
            const auto material_name = [&](uint32_t index) {
                if (index < material_store.Names.size() && !material_store.Names[index].empty()) return std::string{material_store.Names[index]};
                return std::format("Material{}", index);
            };
            if (primitive_materials.empty()) {
                TextUnformatted("No material slots on this mesh.");
            } else if (material_count == 0) {
                TextUnformatted("No materials.");
            } else {
                const uint32_t max_primitive = primitive_materials.size() - 1;
                const auto *existing_slot = r.try_get<const MeshMaterialSlotSelection>(active_mesh_entity);
                uint32_t slot_primitive = existing_slot ? existing_slot->PrimitiveIndex : 0u;
                if (!existing_slot || slot_primitive > max_primitive) {
                    slot_primitive = std::min(slot_primitive, max_primitive);
                    action::Emit(action::object::SetMaterialSlotSelection{slot_primitive});
                }

                BeginChild("MaterialSlots", ImVec2(0, 110), true);
                for (uint32_t primitive_index = 0; primitive_index < primitive_materials.size(); ++primitive_index) {
                    const uint32_t material_index = std::min(primitive_materials[primitive_index], material_count - 1);
                    if (const auto label = std::format("Slot {:L}: {}", primitive_index, material_name(material_index));
                        Selectable(label.c_str(), slot_primitive == primitive_index) && slot_primitive != primitive_index) {
                        action::Emit(action::object::SetMaterialSlotSelection{primitive_index});
                        slot_primitive = primitive_index;
                    }
                }
                EndChild();

                uint32_t material_index = std::min(DisplayedMaterial(r, active_mesh_entity).value_or(primitive_materials[slot_primitive]), material_count - 1);
                if (const auto assigned_material_name = material_name(material_index);
                    BeginCombo("Assigned material", assigned_material_name.c_str())) {
                    for (uint32_t i = 0; i < material_count; ++i) {
                        if (const auto option_name = material_name(i);
                            Selectable(option_name.c_str(), material_index == i)) {
                            action::Emit(action::object::SetMaterialAssignment{slot_primitive, i, ui::TargetFromAlt()});
                            material_index = i;
                        }
                    }
                    EndCombo();
                }

                ui::MaterialEdit fields{r, material_index};
                const auto edit_texture_slot = [&](const char *label, uint32_t &slot) {
                    std::string preview = "None";
                    bool has_match = false;
                    for (const auto &texture : texture_refs) {
                        if (texture.SamplerSlot != slot) continue;
                        preview = texture.Name;
                        has_match = true;
                        break;
                    }
                    if (!has_match && slot != InvalidSlot) preview = std::format("Missing slot {}", slot);

                    bool changed = false;
                    if (BeginCombo(label, preview.c_str())) {
                        if (Selectable("None", slot == InvalidSlot)) {
                            slot = InvalidSlot;
                            changed = true;
                        }
                        for (const auto &texture : texture_refs) {
                            if (Selectable(texture.Name.c_str(), slot == texture.SamplerSlot)) {
                                slot = texture.SamplerSlot;
                                changed = true;
                            }
                        }
                        EndCombo();
                    }
                    return changed;
                };
                // `tex` is a sub-editor over one of the material's TextureInfo fields.
                const auto edit_texture_info = [&](const char *label, auto tex) {
                    tex.template Run<&TextureInfo::Slot>([&](uint32_t &slot) { return edit_texture_slot(std::format("{} texture", label).c_str(), slot); });
                    tex.template Run<&TextureInfo::TexCoord>([&](uint32_t &set) { return SliderUInt(std::format("{} UV set", label).c_str(), &set, 0u, 3u); });
                    tex.template Drag<&TextureInfo::UvOffset>(std::format("{} UV offset", label).c_str());
                    tex.template Drag<&TextureInfo::UvScale>(std::format("{} UV scale", label).c_str());
                    tex.template Drag<&TextureInfo::UvRotation>(std::format("{} UV rotation", label).c_str());
                };
                fields.Color<&PBRMaterial::BaseColorFactor>("Base color");
                fields.Slider<&PBRMaterial::MetallicFactor>("Metallic");
                fields.Slider<&PBRMaterial::RoughnessFactor>("Roughness");
                edit_texture_info("Base color", fields.Sub<&PBRMaterial::BaseColorTexture>());
                edit_texture_info("Metallic-roughness", fields.Sub<&PBRMaterial::MetallicRoughnessTexture>());
                edit_texture_info("Normal", fields.Sub<&PBRMaterial::NormalTexture>());
                fields.Slider<&PBRMaterial::NormalScale>();
                edit_texture_info("Occlusion", fields.Sub<&PBRMaterial::OcclusionTexture>());
                fields.Slider<&PBRMaterial::OcclusionStrength>();
                fields.Color<&PBRMaterial::EmissiveFactor>("Emissive");
                fields.Run<&PBRMaterial::EmissiveStrength>([](float &v) { return ui::DragFloat("Emissive strength", &v, 0.01f, 0.f, FLT_MAX, "%.2f"); }, true);
                edit_texture_info("Emissive", fields.Sub<&PBRMaterial::EmissiveTexture>());

                fields.Enum<&PBRMaterial::AlphaMode>();
                if (materials[material_index].AlphaMode == MaterialAlphaMode::Mask) fields.Slider<&PBRMaterial::AlphaCutoff>();
                fields.Run<&PBRMaterial::DoubleSided>([](uint32_t &v) {
                    bool double_sided = v != 0u;
                    if (!Checkbox("Double sided", &double_sided)) return false;
                    v = double_sided ? 1u : 0u;
                    return true;
                });

                // IOR affects Fresnel reflectance even for non-transmissive dielectrics, so it stays visible.
                fields.Slider<&PBRMaterial::Ior>("IOR");

                const auto pbr_features_mask = r.all_of<PbrMeshFeatures>(active_mesh_entity) ? r.get<const PbrMeshFeatures>(active_mesh_entity).Mask : 0u;
                // Renders the section header when the feature is enabled.
                const auto feature_toggle = [&](const char *label, PbrFeature feature) {
                    bool enabled = HasFeature(pbr_features_mask, feature);
                    if (Checkbox(label, &enabled)) {
                        const auto mask = enabled ? pbr_features_mask | uint32_t(feature) : pbr_features_mask & ~uint32_t(feature);
                        action::Emit(action::object::SetPbrMeshFeaturesMask{mask, ui::TargetFromAlt()});
                    }
                    if (enabled) SeparatorText(label);
                    return enabled;
                };

                if (feature_toggle("Transmission", PbrFeature::Transmission)) {
                    fields.Slider<&PBRMaterial::Transmission, &Transmission::Factor>("Transmission factor");
                    edit_texture_info("Transmission", fields.Sub<&PBRMaterial::Transmission, &Transmission::Texture>());
                    fields.Slider<&PBRMaterial::Dispersion>();
                    // Volume (only meaningful with transmission)
                    fields.Slider<&PBRMaterial::Volume, &Volume::ThicknessFactor>("Thickness");
                    edit_texture_info("Thickness", fields.Sub<&PBRMaterial::Volume, &Volume::ThicknessTexture>());
                    fields.Color<&PBRMaterial::Volume, &Volume::AttenuationColor>("Attenuation color");
                    fields.Run<&PBRMaterial::Volume, &Volume::AttenuationDistance>([](float &v) { return ui::DragFloat("Attenuation distance", &v, 0.01f, 0.f, 0.f, v <= 0.f ? "Infinite" : "%.3f m"); }, true);
                }

                if (feature_toggle("Diffuse transmission", PbrFeature::DiffuseTrans)) {
                    fields.Slider<&PBRMaterial::DiffuseTransmission, &DiffuseTransmission::Factor>("Diffuse transmission factor");
                    edit_texture_info("Diffuse transmission", fields.Sub<&PBRMaterial::DiffuseTransmission, &DiffuseTransmission::Texture>());
                    fields.Color<&PBRMaterial::DiffuseTransmission, &DiffuseTransmission::ColorFactor>("Diffuse transmission color");
                    edit_texture_info("Diffuse transmission color", fields.Sub<&PBRMaterial::DiffuseTransmission, &DiffuseTransmission::ColorTexture>());
                }

                if (feature_toggle("Clearcoat", PbrFeature::Clearcoat)) {
                    fields.Slider<&PBRMaterial::Clearcoat, &Clearcoat::Factor>("Clearcoat factor");
                    edit_texture_info("Clearcoat", fields.Sub<&PBRMaterial::Clearcoat, &Clearcoat::Texture>());
                    fields.Slider<&PBRMaterial::Clearcoat, &Clearcoat::RoughnessFactor>("Clearcoat roughness");
                    edit_texture_info("Clearcoat roughness", fields.Sub<&PBRMaterial::Clearcoat, &Clearcoat::RoughnessTexture>());
                    edit_texture_info("Clearcoat normal", fields.Sub<&PBRMaterial::Clearcoat, &Clearcoat::NormalTexture>());
                    fields.Slider<&PBRMaterial::Clearcoat, &Clearcoat::NormalScale>("Clearcoat normal scale");
                }

                if (feature_toggle("Anisotropy", PbrFeature::Anisotropy)) {
                    fields.Slider<&PBRMaterial::Anisotropy, &Anisotropy::Strength>("Anisotropy strength");
                    fields.Slider<&PBRMaterial::Anisotropy, &Anisotropy::Rotation>("Anisotropy rotation");
                    edit_texture_info("Anisotropy", fields.Sub<&PBRMaterial::Anisotropy, &Anisotropy::Texture>());
                }

                if (feature_toggle("Sheen", PbrFeature::Sheen)) {
                    fields.Color<&PBRMaterial::Sheen, &Sheen::ColorFactor>("Sheen color");
                    edit_texture_info("Sheen color", fields.Sub<&PBRMaterial::Sheen, &Sheen::ColorTexture>());
                    fields.Slider<&PBRMaterial::Sheen, &Sheen::RoughnessFactor>("Sheen roughness");
                    edit_texture_info("Sheen roughness", fields.Sub<&PBRMaterial::Sheen, &Sheen::RoughnessTexture>());
                }

                if (feature_toggle("Iridescence", PbrFeature::Iridescence)) {
                    fields.Slider<&PBRMaterial::Iridescence, &Iridescence::Factor>("Iridescence factor");
                    edit_texture_info("Iridescence", fields.Sub<&PBRMaterial::Iridescence, &Iridescence::Texture>());
                    fields.Slider<&PBRMaterial::Iridescence, &Iridescence::Ior>("Iridescence IOR");
                    fields.Slider<&PBRMaterial::Iridescence, &Iridescence::ThicknessMinimum>("Thickness min", "%.0f nm");
                    fields.Slider<&PBRMaterial::Iridescence, &Iridescence::ThicknessMaximum>("Thickness max", "%.0f nm");
                    edit_texture_info("Iridescence thickness", fields.Sub<&PBRMaterial::Iridescence, &Iridescence::ThicknessTexture>());
                }
            }
        }
    }
    if (HasLens(r, active_entity)) {
        if (CollapsingHeader("Camera")) {
            RenderCameraLensFields(r, active_entity);
            Separator();
            if (LookThroughCameraEntity(r) == active_entity) {
                if (Button("Exit camera view")) action::Emit(action::view::ExitLookThroughCamera{});
            } else {
                if (Button("Look through")) action::Emit(action::view::EnterLookThroughCamera{});
            }
        }
    }
    if (r.all_of<LightIndex>(active_entity) &&
        CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto &light = r.get<const PunctualLight>(active_entity);
        ui::Edit fields{r};
        const char *const type_names[]{"Directional", "Point", "Spot"};
        if (int type_i = int(light.Type); Combo("Type", &type_i, type_names, IM_ARRAYSIZE(type_names))) {
            action::Emit(action::object::SetLightType{PunctualLightType(type_i), ui::TargetFromAlt()});
        }
        fields.Color<&PunctualLight::Color>();
        fields.Slider<&PunctualLight::Intensity>();
        if (light.Type == PunctualLightType::Point || light.Type == PunctualLightType::Spot) {
            bool infinite_range = light.Range <= 0.f;
            if (Checkbox("Infinite range", &infinite_range)) fields.Set<&PunctualLight::Range>(infinite_range ? 0.f : 100.f);
            if (!infinite_range) fields.Slider<&PunctualLight::Range>();
        }
        if (light.Type == PunctualLightType::Spot) {
            constexpr float MaxCone = std::numbers::pi_v<float> / 2.f;
            float outer = std::clamp(light.OuterConeAngle, 0.f, MaxCone);
            const float inner = std::clamp(light.InnerConeAngle, 0.f, outer);
            float blend = outer > 1e-4f ? std::clamp(1.f - inner / outer, 0.f, 1.f) : 0.f;
            const bool size_changed = SliderAngle("Size", &outer, 0.f, 90.f, "%.1f deg");
            ui::KeyDecorator(r, active_entity, animation::Target<&PunctualLight::OuterConeAngle>());
            const bool blend_changed = SliderFloat("Blend", &blend, 0.f, 1.f, "%.2f");
            ui::KeyDecorator(r, active_entity, animation::Target<&PunctualLight::InnerConeAngle>());
            ui::Gesture(size_changed || blend_changed, [&] {
                return action::object::SetSpotCone{std::clamp(outer, 0.f, MaxCone), std::clamp(blend, 0.f, 1.f), ui::TargetFromAlt()};
            });
        }
    }
    if (const auto *instance = r.try_get<Instance>(active_entity); instance && HasMesh(r, instance->Entity)) {
        const bool has_sound = r.all_of<SoundVerticesModel>(active_entity);
        if (CollapsingHeader("Audio", has_sound ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
            DrawObjectAudioControls(r, viewport, active_entity, GetMeshEntity(r, active_entity));
            if (const auto *active_mic = r.try_get<RealImpactActiveMicrophone>(active_entity)) {
                SeparatorText("Microphone");
                Text("Active: %s", GetName(r, active_mic->Entity).c_str());
                if (Button("Select microphone entity")) action::Emit(action::selection::Select{active_mic->Entity});
            }
        }
    } else if (const auto *mic = r.try_get<const RealImpactMicrophone>(active_entity)) {
        if (CollapsingHeader("Audio", ImGuiTreeNodeFlags_DefaultOpen)) {
            Text("Microphone index: %u", mic->Index);
            // Target = sound object currently bound to this mic, else first sound object with a dataset Path.
            auto target = state::Entity{state::Null};
            for (const auto &[e, active] : r.view<const RealImpactActiveMicrophone>().each()) {
                if (active.Entity == active_entity) {
                    target = e;
                    break;
                }
            }
            if (target == state::Null) {
                for (auto [e, _, inst] : r.view<SoundVerticesModel, Instance>().each()) {
                    if (r.all_of<Path>(inst.Entity)) {
                        target = e;
                        break;
                    }
                }
            }
            if (target == state::Null) {
                TextUnformatted("No matching sound object found.");
            } else {
                const auto target_name = GetName(r, target);
                const auto *active = r.try_get<const RealImpactActiveMicrophone>(target);
                const bool is_active = active && active->Entity == active_entity;
                if (is_active) {
                    Text("Active for: %s", target_name.c_str());
                } else if (Button(std::format("Set as active for {}", target_name).c_str())) {
                    action::Emit(action::audio::ActivateRealImpactMicrophone{target});
                }
                if (Button("Select sound object")) action::Emit(action::selection::Select{target});
            }
        }
    }
    physics_ui::RenderEntityProperties(r, active_entity, viewport);

    // glTF metadata: round-trip-only source state on the active entity.
    // TODO: surface per-material source metadata here once material editing UI exists:
    //   - `extras` JSON via SourceAssets::ExtrasByEntity[(Category::Materials, source_index)]
    //   - `MaterialSourceMeta::ExtensionPresence` bits (which extension blocks the source had)
    if (const auto *sa = r.try_get<const gltf::SourceAssets>(viewport)) {
        const auto mesh_entity = active_instance ? active_instance->Entity : state::Null;
        const auto *mesh_layout = mesh_entity != state::Null ? r.try_get<const MeshSourceLayout>(mesh_entity) : nullptr;
        const auto *node = r.try_get<const GltfNode>(active_entity);
        const auto extras = [&](std::optional<uint32_t> index, uint32_t category) -> std::optional<std::string_view> {
            return index ? gltf::GetExtras(*sa, category, *index) : std::nullopt;
        };
        const std::pair<const char *, std::optional<std::string_view>> sections[]{
            {"Extras (Node)", extras(node ? node->Index : std::nullopt, gltf::ExtrasNodes)},
            {"Extras (Mesh)", extras(mesh_layout ? std::optional{mesh_layout->Index} : std::nullopt, gltf::ExtrasMeshes)},
            {"Extras (Camera)", extras(node ? node->Camera : std::nullopt, gltf::ExtrasCameras)},
            {"Extras (Light)", extras(node ? node->Light : std::nullopt, gltf::ExtrasLights)},
        };
        const bool any_extras = std::ranges::any_of(sections, [](const auto &s) { return s.second.has_value(); });
        if ((any_extras || mesh_layout) && CollapsingHeader("glTF metadata")) {
            for (const auto &[label, json] : sections) {
                if (json) RenderJsonBlock(label, *json);
            }
            if (mesh_layout && !mesh_layout->AttributeFlags.empty()) {
                SeparatorText("Mesh source layout");
                Text("Primitives: %zu", mesh_layout->AttributeFlags.size());
                for (size_t i = 0; i < mesh_layout->AttributeFlags.size(); ++i) {
                    const bool indexed = i < mesh_layout->HasSourceIndices.size() && mesh_layout->HasSourceIndices[i];
                    Text("[%zu]%s attrs:%s", i, indexed ? "" : " (non-indexed)", AttributeFlagsString(mesh_layout->AttributeFlags[i]).c_str());
                }
                if (!mesh_layout->MorphTangentDeltas.empty()) {
                    Text("Morph tangent deltas: %zu", mesh_layout->MorphTangentDeltas.size());
                }
            }
        }
    }

    PopID();
}

void RenderControls(state::Scene &r, state::Entity viewport) {
    const profile::CpuScope scope{"SceneControlsUi"};
    if (BeginTabBar("Scene controls")) {
        if (BeginTabItem("Object")) {
            {
                const auto interaction_mode = r.get<const Interaction>(viewport).Mode;
                const auto edit_mode = r.get<const EditMode>(viewport).Value;
                PushID("InteractionMode");
                AlignTextToFramePadding();
                TextUnformatted("Interaction mode:");
                auto interaction_mode_value = int(interaction_mode);
                bool interaction_mode_changed = false;
                const auto active_entity_rc = FindActiveEntity(r);
                const bool active_is_armature_rc = FindArmatureObject(r, active_entity_rc) != state::Null;
                const bool edit_allowed = AllSelectedAreMeshes(r) || active_is_armature_rc;
                const bool pose_allowed = active_is_armature_rc;
                for (const auto mode : r.get<const EnabledInteractionModes>(viewport).Value) {
                    if (mode == InteractionMode::Edit && !edit_allowed) continue;
                    if (mode != InteractionMode::Pose || pose_allowed) {
                        SameLine();
                        interaction_mode_changed |= RadioButton(to_string(mode).c_str(), &interaction_mode_value, int(mode));
                    }
                }
                if (interaction_mode_changed) action::Emit(action::view::SetInteractionMode{.Mode = InteractionMode(interaction_mode_value)});
                ui::Edit viewport_edit{r, viewport};
                if (interaction_mode == InteractionMode::Edit || interaction_mode == InteractionMode::Excite) {
                    viewport_edit.Check<&OrbitToActive::Value>("Orbit to active");
                }
                if (interaction_mode == InteractionMode::Edit && !active_is_armature_rc) {
                    AlignTextToFramePadding();
                    TextUnformatted("Edit mode:");
                    auto type_interaction_mode = int(edit_mode);
                    for (const auto element : Elements) {
                        auto name = Capitalize(label(element));
                        SameLine();
                        if (RadioButton(name.c_str(), &type_interaction_mode, int(element))) action::Emit(action::view::SetEditMode{.Mode = element});
                    }
                    const auto active_entity = FindActiveEntity(r);
                    const auto *active_instance = active_entity != state::Null ? r.try_get<const Instance>(active_entity) : nullptr;
                    const auto active_mesh = active_instance && HasMesh(r, active_instance->Entity) ? active_instance->Entity : state::Null;
                    const auto *active_stats = active_mesh != state::Null ? GetElementSelectionSummary(r, active_mesh, edit_mode) : nullptr;
                    const uint32_t selected_count = active_stats ? active_stats->SelectedCount : 0u;
                    bool any_sharp = false, any_smooth = false;
                    for (const auto entity : r.view<const MeshElementSelection, const MeshHandle>()) {
                        const auto *summary = GetElementSelectionSummary(r, entity, edit_mode);
                        if (!summary) continue;
                        any_sharp |= (summary->SharpnessFlags & 1u) != 0u;
                        any_smooth |= (summary->SharpnessFlags & 2u) != 0u;
                        if (any_sharp && any_smooth) break;
                    }
                    if (active_mesh != state::Null) Text("Editing %s: %u selected", label(edit_mode).data(), selected_count);
                    // Apply face shading or sharp-edge updates to selected elements.
                    // Vertex mode marks every edge incident to a selected vertex.
                    if (edit_mode != Element::None) {
                        if (any_sharp || any_smooth) {
                            const bool mixed = any_sharp && any_smooth;
                            if (mixed) PushItemFlag(ImGuiItemFlags_MixedValue, true);
                            if (edit_mode == Element::Face) {
                                if (bool set_smooth = !any_sharp; Checkbox("Smooth faces", &set_smooth)) action::Emit(action::object::SetSelectedSharp{Element::Face, !set_smooth});
                            } else if (bool set_sharp = any_sharp && !any_smooth; Checkbox(edit_mode == Element::Edge ? "Sharp edges" : "Sharp vertices", &set_sharp)) {
                                action::Emit(action::object::SetSelectedSharp{edit_mode == Element::Edge ? Element::Edge : Element::Vertex, set_sharp});
                            }
                            if (mixed) PopItemFlag();
                        }
                    }
                }
                PopID();
            }
            if (r.view<const Scene>().size() > 1) {
                std::vector<state::Entity> scenes;
                for (const auto e : r.view<const Scene>()) scenes.emplace_back(e);
                std::ranges::sort(scenes, {}, [&](state::Entity e) {
                    const auto *si = r.try_get<const SourceIndex>(e);
                    return si ? si->Value : std::numeric_limits<uint32_t>::max();
                });
                state::Entity active = state::Null;
                for (const auto e : r.view<const ActiveScene>()) active = e;
                const auto scene_label = [&](state::Entity e) {
                    const auto *si = r.try_get<const SourceIndex>(e);
                    return NamedOr(r.get<const Scene>(e).Name, "Scene ", si ? si->Value : 0u);
                };
                if (active != state::Null) ui::ChoiceCombo("Scene", active, scenes, scene_label, [](state::Entity e) { action::Emit(action::view::SetActiveScene{e}); });
            }
            if (CollapsingHeader("Object tree", ImGuiTreeNodeFlags_DefaultOpen)) RenderObjectTree(r, viewport);
            SeparatorText("");
            if (CollapsingHeader("Add object")) {
                static constexpr std::array AllPrimitiveShapes{
                    PrimitiveShape{primitive::Plane{}},
                    PrimitiveShape{primitive::Circle{}},
                    PrimitiveShape{primitive::Cuboid{}},
                    PrimitiveShape{primitive::IcoSphere{}},
                    PrimitiveShape{primitive::UVSphere{}},
                    PrimitiveShape{primitive::Torus{}},
                    PrimitiveShape{primitive::Cylinder{}},
                    PrimitiveShape{primitive::Cone{}},
                };

                for (uint32_t i = 0; i < AllPrimitiveShapes.size(); ++i) {
                    if (i % 4 != 0) SameLine();
                    const auto &shape = AllPrimitiveShapes[i];
                    if (Button(ToString(shape).c_str())) {
                        action::Emit(action::object::AddMeshPrimitive{shape, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Name = ToString(shape)})});
                    }
                }
                Spacing();
                if (Button("Empty")) action::Emit(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
                SameLine();
                if (Button("Armature")) action::Emit(action::object::AddArmature{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
                SameLine();
                if (Button("Camera")) action::Emit(action::object::AddCamera{.Info = std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
                SameLine();
                if (Button("Light")) action::Emit(action::object::AddLight{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
            }
            if (auto *mv = r.try_get<MaterialVariants>(viewport); mv && !mv->Names.empty() && CollapsingHeader("Material variants")) {
                std::vector<std::optional<uint32_t>> variants{std::nullopt};
                for (uint32_t i = 0; i < mv->Names.size(); ++i) variants.emplace_back(i);
                ui::ChoiceCombo(
                    "Active variant", mv->Active, variants,
                    [&](std::optional<uint32_t> v) { return v ? NamedOr(mv->Names[*v], "Variant ", *v) : std::string{"Default"}; },
                    [&](std::optional<uint32_t> v) { action::Emit(action::UpdateOf<&MaterialVariants::Active>(viewport, v)); }
                );
            }
            if (!r.view<const Selected>().empty()) {
                SeparatorText("Selection actions");
                std::vector<state::Entity> selected_mesh_instances;
                for (const auto entity : r.view<const Selected, const Instance>()) {
                    if (!r.all_of<SubElementOf>(entity)) selected_mesh_instances.emplace_back(entity);
                }

                if (!selected_mesh_instances.empty()) {
                    const bool any_visible = any_of(selected_mesh_instances, [&](state::Entity e) { return r.all_of<RenderInstance>(e); });
                    const bool any_hidden = any_of(selected_mesh_instances, [&](state::Entity e) { return !r.all_of<RenderInstance>(e); });
                    const bool mixed_visible = any_visible && any_hidden;
                    if (mixed_visible) PushItemFlag(ImGuiItemFlags_MixedValue, true);
                    if (bool set_visible = any_visible && !any_hidden; Checkbox("Visible", &set_visible)) action::Emit(action::object::SetSelectedVisible{set_visible});
                    if (mixed_visible) PopItemFlag();

                    const auto face_mesh_entities = selection::GetSelectedMeshEntities(r) |
                        std::views::filter([&](state::Entity me) { return GetMesh(r, me).FaceCount() > 0; }) |
                        to<std::vector>();
                    if (!face_mesh_entities.empty()) {
                        // A fully smooth mesh has no sharp faces, while partial sharpness produces a mixed checkbox.
                        bool any_smooth = false, any_sharp = false, any_partial = false;
                        for (const auto me : face_mesh_entities) {
                            const auto &summary = r.get<const MeshShadingSummary>(me);
                            any_smooth |= !summary.AnySharp;
                            any_sharp |= summary.AnySharp;
                            any_partial |= summary.AnySharp && !summary.AllSharp;
                            if ((any_smooth && any_sharp) || any_partial) break;
                        }
                        const bool mixed_smooth = (any_smooth && any_sharp) || any_partial;
                        SameLine();
                        if (mixed_smooth) PushItemFlag(ImGuiItemFlags_MixedValue, true);
                        if (bool set_smooth = any_smooth && !any_sharp; Checkbox("Smooth shading", &set_smooth)) action::Emit(action::object::SetSelectedSmoothShading{set_smooth});
                        if (mixed_smooth) PopItemFlag();
                        if (Button("Smooth by angle")) action::Emit(action::object::ShadeSelectedSmoothByAngle{r.get<const ShadeSmoothAngle>(viewport).Value});
                        SameLine();
                        SetNextItemWidth(GetFontSize() * 6);
                        ui::Edit{r, viewport}.Slider<&ShadeSmoothAngle::Value>("##SmoothByAngle");
                        SameLine();
                        MeshEditor::HelpMarker("Maximum angle between face normals that will be considered as smooth");
                    }
                }
                if (CanDuplicate(r, viewport) && Button("Duplicate")) Duplicate(r, viewport);
                if (CanDuplicateLinked(r, viewport)) {
                    SameLine();
                    if (Button("Duplicate linked")) action::Emit(action::object::DuplicateLinked{}, action::Phase::Stage);
                }
                if (CanDelete(r, viewport) && Button("Delete")) Delete(r, viewport);
                if (r.get<const Interaction>(viewport).Mode == InteractionMode::Pose && !r.view<const BoneSelection>().empty()) {
                    AlignTextToFramePadding();
                    TextUnformatted("Clear transform:");
                    SameLine();
                    if (Button("All")) action::Emit(action::bone::ClearSelectedTransforms{.Position = true, .Rotation = true, .Scale = true});
                    SameLine();
                    if (Button("Position")) action::Emit(action::bone::ClearSelectedTransforms{.Position = true});
                    SameLine();
                    if (Button("Rotation")) action::Emit(action::bone::ClearSelectedTransforms{.Rotation = true});
                    SameLine();
                    if (Button("Scale")) action::Emit(action::bone::ClearSelectedTransforms{.Scale = true});
                }
            }
            RenderEntityControls(r, viewport, FindActiveEntity(r));
            EndTabItem();
        }

        if (BeginTabItem("Render")) {
            ui::Edit f{r, viewport};
            const auto &settings = r.get<const ViewportDisplay>(viewport);
            {
                auto color = settings.ClearColor;
                if (ColorEdit3("Background color", &color.x)) {
                    color.w = 1.f;
                    f.Set<&ViewportDisplay::ClearColor>(color);
                }
            }
            f.Enum<&ViewportDisplay::AnisotropicFilter>("Anisotropic filtering");
            if (CollapsingHeader("Motion blur")) {
                if (bool enabled = settings.MotionBlur.has_value(); Checkbox("Enabled", &enabled)) {
                    f.Set<&ViewportDisplay::MotionBlur>(enabled ? std::optional{MotionBlur{}} : std::optional<MotionBlur>{});
                }
                if (settings.MotionBlur) {
                    auto mb = *settings.MotionBlur;
                    int method = int(mb.Method);
                    bool changed = Combo("Method", &method, "Velocity (fast)\0Full sampling\0");
                    MeshEditor::HelpMarker("Velocity (fast) blurs one rendered image using motion vectors. It is inexpensive, but approximates changing visibility, reflections, and lighting.\n\nFull sampling renders multiple times across the shutter to capture those changes. More samples reduce stepping at higher rendering cost.");
                    mb.Method = MotionBlurMethod(method);
                    changed |= SliderFloat("Shutter (frames)", &mb.Shutter, 0.f, 2.f);
                    if (uint32_t steps = mb.Steps; mb.Method == MotionBlurMethod::FullSampling && SliderUInt("Samples", &steps, 1, 64)) {
                        mb.Steps = uint8_t(steps);
                        changed = true;
                    }
                    if (changed) f.Set<&ViewportDisplay::MotionBlur>(std::optional{mb});
                }
            }
            // Direct mutation outside Apply: not replayable document state.
            if (Button("Recompile shaders")) r.Context.get<FrameState>().RecompileShaders = true;

            if (!r.view<Selected>().empty()) {
                SeparatorText("Selection overlays");
                AlignTextToFramePadding();
                TextUnformatted("Normals");
                for (const auto element : NormalElements) {
                    SameLine();
                    bool show = ElementMaskContains(settings.NormalOverlays, element);
                    if (const auto type_name = Capitalize(label(element));
                        Checkbox(type_name.c_str(), &show)) {
                        auto next_mask = settings.NormalOverlays;
                        SetElementMask(next_mask, element, show);
                        f.Set<&ViewportDisplay::NormalOverlays>(next_mask);
                    }
                }
                f.Check<&ViewportDisplay::ShowBoundingBoxes>("Bounding boxes");
                if (!r.view<const TetBuffers>().empty()) f.Check<&ViewportDisplay::ShowTetWireframe>("Tet wireframe");
            }
            {
                using VC = ViewportThemeColors;
                using AC = AxisThemeColors;
                SeparatorText("Viewport theme");
                const auto &theme = r.get<const ViewportTheme>(viewport);
                if (Button("Reset##ViewportTheme")) action::Emit(action::view::ResetViewportTheme{});
                auto c = f.Sub<&ViewportTheme::Colors>();
                c.Color<&VC::Grid>();
                c.Color<&VC::Wire>();
                c.Color<&VC::WireEdit>();
                c.Color<&VC::ObjectActive>();
                c.Color<&VC::ObjectSelected>();
                c.Color<&VC::Light>();
                c.Color<&VC::Vertex>();
                c.Color<&VC::VertexSelected>();
                c.Color<&VC::EdgeSelectedIncidental>("Edge selected (incidental)");
                c.Color<&VC::EdgeSelected>();
                c.Color<&VC::EdgeSharp>();
                c.Color<&VC::FaceSelectedIncidental>("Face selected (incidental)");
                c.Color<&VC::FaceSelected>();
                c.Color<&VC::ElementActive>();
                c.Color<&VC::ElementExcited>();
                c.Color<&VC::FaceNormal>();
                c.Color<&VC::VertexNormal>();
                c.Color<&VC::BoneSolid>();
                c.Color<&VC::BonePose>();
                c.Color<&VC::BonePoseActive>();
                c.Color<&VC::Transform>();
                SeparatorText("Axis colors");
                auto a = f.Sub<&ViewportTheme::AxisColors>();
                a.Color<&AC::X>("Axis X");
                a.Color<&AC::Y>("Axis Y");
                a.Color<&AC::Z>("Axis Z");
                // UI edits full width and storage is half-width.
                if (float full_width = theme.EdgeWidth * 2.f; SliderFloat("Edge width", &full_width, 0.5f, 4.f))
                    f.Set<&ViewportTheme::EdgeWidth>(full_width / 2.f);
                if (uint32_t v = theme.SilhouetteEdgeWidth; SliderUInt("Silhouette edge width", &v, 1, 4))
                    f.Set<&ViewportTheme::SilhouetteEdgeWidth>(v);
            }
            EndTabItem();
        }

        if (BeginTabItem("Camera")) {
            const auto &camera = r.get<const ViewCamera>(viewport);
            const auto extent = r.Context.get<ViewportExtent>().Value;
            const float viewport_aspect = extent.x == 0 || extent.y == 0 ? 1.f : float(extent.x) / float(extent.y);
            if (Button("Reset##Camera")) action::Emit(action::view::ResetViewCamera{});
            {
                vec3 target = camera.Target;
                ui::Gesture(SliderFloat3("Target", &target.x, -10, 10), [&] { return action::view::SetViewCameraTarget{target}; });
            }
            {
                CameraLens lens = camera.Data;
                ui::Gesture(RenderCameraLensEditor(lens, camera.Distance, viewport_aspect), [&] { return action::view::SetViewCameraLens{lens}; });
            }
            EndTabItem();
        }

        if (BeginTabItem("Physics")) {
            physics_ui::RenderTab(r, viewport);
            EndTabItem();
        }

        if (BeginTabItem("Audio")) {
            if (CollapsingHeader("Device", ImGuiTreeNodeFlags_DefaultOpen)) DrawAudioDeviceControls(r, viewport);
            DrawGlobalSynthControls(r, viewport);
            EndTabItem();
        }

        if (const auto *sa = r.try_get<const gltf::SourceAssets>(viewport)) {
            const bool has_asset = !sa->Generator.empty() || !sa->Copyright.empty() || !sa->MinVersion.empty() || !sa->AssetExtras.empty() || !sa->AssetExtensions.empty();
            const bool has_content = has_asset || !sa->ExtensionsRequired.empty() || !sa->Images.empty() || !sa->Textures.empty() || !sa->Samplers.empty();
            if (has_content && BeginTabItem("glTF metadata")) {
                if (has_asset && CollapsingHeader("Asset", ImGuiTreeNodeFlags_DefaultOpen)) {
                    if (!sa->Generator.empty()) Text("Generator: %s", sa->Generator.c_str());
                    if (!sa->Copyright.empty()) Text("Copyright: %s", sa->Copyright.c_str());
                    if (!sa->MinVersion.empty()) Text("Min version: %s", sa->MinVersion.c_str());
                    if (!sa->AssetExtras.empty()) RenderJsonBlock("asset.extras", sa->AssetExtras);
                    if (!sa->AssetExtensions.empty()) RenderJsonBlock("asset.extensions", sa->AssetExtensions);
                }
                if (!sa->ExtensionsRequired.empty() && CollapsingHeader("Extensions required", ImGuiTreeNodeFlags_DefaultOpen)) {
                    for (const auto &e : sa->ExtensionsRequired) BulletText("%s", e.c_str());
                }
                if (!sa->Images.empty() && CollapsingHeader("Image registry")) {
                    MetaTable("Images", {"#", "Name", "Mime", "Source", "Bytes"}, sa->Images.size(), [&](size_t i) {
                        const auto &img = sa->Images[i];
                        TableNextColumn();
                        Text("%zu", i);
                        TableNextColumn();
                        TextUnformatted(img.Name.c_str());
                        TableNextColumn();
                        TextUnformatted(MimeTypeName(img.MimeType).data());
                        TableNextColumn();
                        if (img.Source == gltf::Image::SourceKind::DataUri) TextUnformatted("data URI");
                        else if (!img.SourcePath.empty()) TextUnformatted(img.SourcePath.c_str());
                        else TextUnformatted("embedded");
                        TableNextColumn();
                        if (img.Bytes.empty()) TextUnformatted(img.SourcePath.empty() ? "—" : "external");
                        else Text("%zu", img.Bytes.size());
                        if (img.IsDirty) {
                            SameLine();
                            TextUnformatted("(dirty)");
                        }
                    });
                }
                if (!sa->Textures.empty() && CollapsingHeader("Texture registry")) {
                    const auto idx_or_dash = [](const std::optional<uint32_t> &i) {
                        return i ? std::format("{}", *i) : std::string{"—"};
                    };
                    MetaTable("Textures", {"#", "Name", "Sampler", "Image (default/WebP/Basisu/DDS)"}, sa->Textures.size(), [&](size_t i) {
                        const auto &t = sa->Textures[i];
                        TableNextColumn();
                        Text("%zu", i);
                        TableNextColumn();
                        TextUnformatted(t.Name.c_str());
                        TableNextColumn();
                        TextUnformatted(idx_or_dash(t.SamplerIndex).c_str());
                        TableNextColumn();
                        Text("%s / %s / %s / %s", idx_or_dash(t.ImageIndex).c_str(), idx_or_dash(t.WebpImageIndex).c_str(), idx_or_dash(t.BasisuImageIndex).c_str(), idx_or_dash(t.DdsImageIndex).c_str());
                    });
                }
                if (!sa->Samplers.empty() && CollapsingHeader("Sampler registry")) {
                    const auto filter_name = [](gltf::Filter f) -> std::string_view {
                        using F = gltf::Filter;
                        switch (f) {
                            case F::Nearest: return "Nearest";
                            case F::Linear: return "Linear";
                            case F::NearestMipMapNearest: return "Nearest/Nearest";
                            case F::LinearMipMapNearest: return "Linear/Nearest";
                            case F::NearestMipMapLinear: return "Nearest/Linear";
                            case F::LinearMipMapLinear: return "Linear/Linear";
                        }
                        return "?";
                    };
                    const auto wrap_name = [](gltf::Wrap w) -> std::string_view {
                        using W = gltf::Wrap;
                        switch (w) {
                            case W::ClampToEdge: return "Clamp";
                            case W::MirroredRepeat: return "Mirror";
                            case W::Repeat: return "Repeat";
                        }
                        return "?";
                    };
                    MetaTable("Samplers", {"#", "Name", "Mag", "Min", "Wrap S/T"}, sa->Samplers.size(), [&](size_t i) {
                        const auto &s = sa->Samplers[i];
                        TableNextColumn();
                        Text("%zu", i);
                        TableNextColumn();
                        TextUnformatted(s.Name.c_str());
                        TableNextColumn();
                        TextUnformatted(s.MagFilter ? filter_name(*s.MagFilter).data() : "—");
                        TableNextColumn();
                        TextUnformatted(s.MinFilter ? filter_name(*s.MinFilter).data() : "—");
                        TableNextColumn();
                        Text("%s / %s", wrap_name(s.WrapS).data(), wrap_name(s.WrapT).data());
                    });
                }
                EndTabItem();
            }
        }
        EndTabBar();
    }
}

static void RenderObjectTree(state::Scene &r, state::Entity viewport) {
    PushStyleVar(ImGuiStyleVar_ItemSpacing, {GetStyle().ItemSpacing.x, 0.f});

    const auto ToSelectionUserData = [](state::Entity e) -> ImGuiSelectionUserData { return ImGuiSelectionUserData(uint32_t(e)); };
    const auto FromSelectionUserData = [&](ImGuiSelectionUserData data) -> state::Entity {
        if (data == ImGuiSelectionUserData_Invalid) return state::Null;
        const auto e = state::Entity(uint32_t(data));
        return r.valid(e) ? e : state::Null;
    };

    const auto GetEntityTypeName = [&](state::Entity e) -> std::string_view {
        if (r.all_of<BoneIndex>(e)) return "Bone";
        if (r.all_of<ObjectKind>(e)) return ObjectTypeName(r.get<const ObjectKind>(e).Value);
        return ObjectTypeName(ObjectType::Empty);
    };
    std::vector<state::Entity> visible_entities;
    // Mutates `out` so begin and end batches fold into a single action.
    using Clear = action::selection::ApplyTreeSelection::ClearKind;
    const auto resolve_into = [&](action::selection::ApplyTreeSelection &out, std::span<const ImGuiSelectionRequest> requests, ImGuiSelectionUserData nav_item) {
        const auto add_target = [&](state::Entity e, bool selected) {
            if (e == state::Null) return;
            out.Add(e, selected);
        };
        for (const auto &request : requests) {
            if (request.Type == ImGuiSelectionRequestType_SetAll) {
                if (request.Selected) {
                    for (const auto e : visible_entities) add_target(e, true);
                    if (const auto nav = FromSelectionUserData(nav_item); nav != state::Null) out.NavToActive = nav;
                } else {
                    const auto nav = FromSelectionUserData(nav_item);
                    out.Clear = nav != state::Null && r.all_of<BoneIndex>(nav) ? Clear::BonesOnly : Clear::All;
                }
                continue;
            }
            if (request.Type != ImGuiSelectionRequestType_SetRange) continue;

            // The request reflects this frame's input. ImGui's NavIdItem catches up next frame.
            if (request.Selected) out.NavToActive = FromSelectionUserData(request.RangeDirection >= 0 ? request.RangeLastItem : request.RangeFirstItem);

            const auto first = FromSelectionUserData(request.RangeFirstItem), last = FromSelectionUserData(request.RangeLastItem);
            const auto first_it = find(visible_entities, first), last_it = find(visible_entities, last);
            if (first_it == visible_entities.end() || last_it == visible_entities.end()) {
                add_target(first, request.Selected);
                add_target(last, request.Selected);
                continue;
            }
            const auto first_i = distance(visible_entities.begin(), first_it);
            const auto last_i = distance(visible_entities.begin(), last_it);
            const auto [i0, i1] = std::minmax(first_i, last_i);
            for (auto i = i0; i <= i1; ++i) add_target(visible_entities[i], request.Selected);
        }
    };

    const int total_selected = r.view<const Selected>().size() + r.view<const BoneSelection>().size();
    auto *ms_begin = BeginMultiSelect(ImGuiMultiSelectFlags_None, total_selected, -1);
    std::vector<ImGuiSelectionRequest> begin_requests;
    begin_requests.reserve(ms_begin->Requests.Size);
    for (const auto &request : ms_begin->Requests) begin_requests.emplace_back(request);
    const auto begin_nav_item = ms_begin->NavIdItem;

    // Build the set of ancestors of any selected entity (for secondary highlight).
    std::unordered_set<state::Entity> ancestor_of_selected;
    const auto mark_ancestors = [&](state::Entity selected_entity) {
        const auto *n = r.try_get<SceneNode>(selected_entity);
        auto parent = n ? n->Parent : state::Null;
        while (parent != state::Null) {
            if (!ancestor_of_selected.insert(parent).second) break; // already inserted, so parents are already covered
            const auto *pn = r.try_get<SceneNode>(parent);
            parent = pn ? pn->Parent : state::Null;
        }
    };
    for (const auto e : r.view<Selected>()) mark_ancestors(e);
    for (const auto e : r.view<BoneSelection>()) mark_ancestors(e);

    const auto render_entity = [&](const auto &self, state::Entity e) -> void {
        const auto *node = r.try_get<SceneNode>(e);
        const bool has_children = node && node->FirstChild != state::Null;
        const bool is_selected = r.any_of<Selected, BoneSelection>(e);
        const bool is_ancestor_selected = !is_selected && ancestor_of_selected.contains(e);

        auto flags =
            ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanFullWidth |
            ImGuiTreeNodeFlags_FramePadding |
            ImGuiTreeNodeFlags_NavLeftJumpsToParent;
        if (!has_children) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        if (is_selected || is_ancestor_selected) flags |= ImGuiTreeNodeFlags_Selected;

        if (is_ancestor_selected) {
            const auto col = GetStyleColorVec4(ImGuiCol_Header);
            PushStyleColor(ImGuiCol_Header, ImVec4{col.x, col.y, col.z, col.w * 0.4f});
            PushStyleColor(ImGuiCol_HeaderHovered, ImVec4{col.x, col.y, col.z, col.w * 0.6f});
        }

        SetNextItemSelectionUserData(ToSelectionUserData(e));
        const bool open = TreeNodeEx(reinterpret_cast<void *>(uintptr_t(uint32_t(e))), flags, "%s", GetName(r, e).c_str());
        SameLine();
        if (const auto type_suffix = GetEntityTypeName(e); r.any_of<Active, BoneActive>(e)) {
            const auto &theme = r.get<const ViewportTheme>(viewport);
            const auto color = r.all_of<BoneActive>(e) ? theme.Colors.BoneActive : theme.Colors.ObjectActive;
            TextColored(ImVec4{color.x, color.y, color.z, 1.f}, "[%s]", type_suffix.data());
        } else {
            TextDisabled("[%s]", type_suffix.data());
        }
        if (is_ancestor_selected) PopStyleColor(2);
        visible_entities.emplace_back(e);
        if (open && has_children) {
            for (const auto child : Children{&r, e}) self(self, child);
            TreePop();
        }
    };

    const auto roots = SortedEntities(
        r.view<const Name>() |
        std::views::filter([&](auto e) {
            const auto *node = r.try_get<const SceneNode>(e);
            return !node || node->Parent == state::Null;
        })
    );
    for (const auto e : roots) render_entity(render_entity, e);
    if (roots.empty()) TextDisabled("No objects");

    // BeginMultiSelect and EndMultiSelect can produce the same selection update.
    action::selection::ApplyTreeSelection tree_selection;
    resolve_into(tree_selection, begin_requests, begin_nav_item);
    auto *ms_end = EndMultiSelect();
    resolve_into(tree_selection, {ms_end->Requests.Data, size_t(ms_end->Requests.Size)}, ms_end->NavIdItem);
    if (!tree_selection.Entities.empty() ||
        tree_selection.Clear != Clear::None || tree_selection.NavToActive != state::Null) {
        action::Emit(std::move(tree_selection));
    }

    PopStyleVar();
}
