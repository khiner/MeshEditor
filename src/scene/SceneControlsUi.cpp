#include "scene/SceneControlsUi.h"
#include "Camera.h"
#include "Path.h"
#include "TransformMath.h"
#include "action/Audio.h"
#include "action/Bone.h"
#include "action/Object.h"
#include "action/Selection.h"
#include "action/View.h"
#include "animation/AnimationData.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "audio/AudioSystem.h"
#include "gizmo/GizmoInteraction.h"
#include "gizmo/TransformGizmo.h"
#include "gltf/GltfScene.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "mesh/Primitives.h"
#include "physics/PhysicsUi.h"
#include "render/GpuBufferAccessors.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/PbrFeature.h"
#include "render/Textures.h"
#include "scene/Defaults.h"
#include "scene/SceneGraph.h"
#include "scene/WorldTransform.h"
#include "selection/Selection.h"
#include "selection/SelectionBitset.h"
#include "selection/SelectionComponents.h"
#include "ui/FieldEdit.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportInteractionState.h"
#include "viewport/ViewportOps.h"

#include <entt/entity/registry.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <imgui_internal.h>

using std::ranges::any_of, std::ranges::distance, std::ranges::find, std::ranges::to;

using namespace ImGui;

static void RenderObjectTree(entt::registry &, entt::entity viewport);
static void RenderEntityControls(entt::registry &, entt::entity viewport, entt::entity active_entity);

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

std::optional<MeshData> PrimitiveEditor(PrimitiveShape &shape) {
    static constexpr float MinSize = 0.01f, MaxSize = 100.f, SizeSpeed = 0.01f;
    return std::visit([](auto &s) -> std::optional<MeshData> {
        using T = std::decay_t<decltype(s)>;
        bool changed = false;
        if constexpr (std::is_same_v<T, primitive::Plane>) {
            vec2 size = s.HalfExtents * 2.f;
            changed = DragFloat2("Size", &size.x, SizeSpeed, MinSize, MaxSize);
            if (changed) s.HalfExtents = size / 2.f;
        } else if constexpr (std::is_same_v<T, primitive::Circle>) {
            changed |= DragFloat("Radius", &s.Radius, SizeSpeed, MinSize, MaxSize);
            int segments = int(s.Segments);
            changed |= SliderInt("Segments", &segments, 3, 128);
            if (changed) s.Segments = uint(segments);
        } else if constexpr (std::is_same_v<T, primitive::Cuboid>) {
            vec3 size = s.HalfExtents * 2.f;
            changed = DragFloat3("Size", &size.x, SizeSpeed, MinSize, MaxSize);
            if (changed) s.HalfExtents = size / 2.f;
        } else if constexpr (std::is_same_v<T, primitive::IcoSphere>) {
            changed |= DragFloat("Radius", &s.Radius, SizeSpeed, MinSize, MaxSize);
            int subdivisions = int(s.Subdivisions);
            changed |= SliderInt("Subdivisions", &subdivisions, 1, 6);
            if (changed) s.Subdivisions = uint(subdivisions);
        } else if constexpr (std::is_same_v<T, primitive::UVSphere>) {
            changed |= DragFloat("Radius", &s.Radius, SizeSpeed, MinSize, MaxSize);
            int slices = int(s.Slices), stacks = int(s.Stacks);
            changed |= SliderInt("Slices", &slices, 3, 128);
            changed |= SliderInt("Stacks", &stacks, 2, 64);
            if (changed) {
                s.Slices = uint(slices);
                s.Stacks = uint(stacks);
            }
        } else if constexpr (std::is_same_v<T, primitive::Torus>) {
            changed |= DragFloat("Major radius", &s.MajorRadius, SizeSpeed, MinSize, MaxSize);
            changed |= DragFloat("Minor radius", &s.MinorRadius, SizeSpeed, MinSize, s.MajorRadius);
            int major_seg = int(s.MajorSegments), minor_seg = int(s.MinorSegments);
            changed |= SliderInt("Major segments", &major_seg, 3, 256);
            changed |= SliderInt("Minor segments", &minor_seg, 3, 256);
            if (changed) {
                s.MajorSegments = uint(major_seg);
                s.MinorSegments = uint(minor_seg);
            }
        } else if constexpr (std::is_same_v<T, primitive::Cylinder> || std::is_same_v<T, primitive::Cone>) {
            changed |= DragFloat("Radius", &s.Radius, SizeSpeed, MinSize, MaxSize);
            changed |= DragFloat("Height", &s.Height, SizeSpeed, MinSize, MaxSize);
            int slices = int(s.Slices);
            changed |= SliderInt("Slices", &slices, 3, 128);
            if (changed) s.Slices = uint(slices);
        }
        if (changed) return primitive::CreateMesh(s);
        return {};
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

float AngleFromCos(float cos_theta) { return std::acos(std::clamp(cos_theta, -1.f, 1.f)); }

// `viewport_aspect` is set when the camera is bound to a viewport that determines its aspect.
bool RenderCameraLensEditor(Camera &camera, float distance, std::optional<float> viewport_aspect = {}) {
    bool lens_changed = false;

    int proj_i = std::holds_alternative<Orthographic>(camera) ? 1 : 0;
    const char *proj_names[]{"Perspective", "Orthographic"};
    if (Combo("Projection", &proj_i, proj_names, IM_ARRAYSIZE(proj_names))) {
        if (proj_i == 0 && !std::holds_alternative<Perspective>(camera)) {
            camera = PerspectiveFromOrthographic(std::get<Orthographic>(camera), distance);
            lens_changed = true;
        } else if (proj_i == 1 && !std::holds_alternative<Orthographic>(camera)) {
            camera = OrthographicFromPerspective(std::get<Perspective>(camera), distance, viewport_aspect);
            lens_changed = true;
        }
    }

    if (auto *perspective = std::get_if<Perspective>(&camera)) {
        float fov_deg = glm::degrees(perspective->FieldOfViewRad);
        const float far_max = std::max(perspective->NearClip + MinNearFarDelta, MaxFarClip);
        if (SliderFloat("Field of view (deg)", &fov_deg, 1.f, 179.f)) {
            perspective->FieldOfViewRad = glm::radians(fov_deg);
            lens_changed = true;
        }
        const float near_max = perspective->FarClip ? std::max(*perspective->FarClip - MinNearFarDelta, MinNearClip) : far_max;
        lens_changed |= SliderFloat("Near clip", &perspective->NearClip, MinNearClip, near_max);
        bool infinite_far = !perspective->FarClip.has_value();
        if (Checkbox("Infinite far clip", &infinite_far)) {
            if (infinite_far) perspective->FarClip.reset();
            else perspective->FarClip = far_max;
            lens_changed = true;
        }
        if (perspective->FarClip) {
            lens_changed |= SliderFloat("Far clip", &*perspective->FarClip, perspective->NearClip + MinNearFarDelta, far_max);
        }
        if (!viewport_aspect) {
            float aspect = perspective->AspectRatio.value_or(DefaultAspectRatio);
            if (SliderFloat("Aspect ratio", &aspect, 0.1f, 5.f)) {
                perspective->AspectRatio = aspect;
                lens_changed = true;
            }
        }
    } else if (auto *orthographic = std::get_if<Orthographic>(&camera)) {
        const float far_max = std::max(orthographic->NearClip + MinNearFarDelta, MaxFarClip);
        lens_changed |= SliderFloat("X Mag", &orthographic->Mag.x, 0.01f, 100.f);
        lens_changed |= SliderFloat("Y Mag", &orthographic->Mag.y, 0.01f, 100.f);
        lens_changed |= SliderFloat("Near clip", &orthographic->NearClip, MinNearClip, orthographic->FarClip - MinNearFarDelta);
        lens_changed |= SliderFloat("Far clip", &orthographic->FarClip, orthographic->NearClip + MinNearFarDelta, far_max);
    }
    return lens_changed;
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
} // namespace

static void RenderEntityControls(entt::registry &r, entt::entity viewport, entt::entity active_entity) {
    auto &meshes = r.ctx().get<MeshStore>();
    auto &textures = r.ctx().get<TextureStore>();
    if (active_entity == entt::null) {
        TextUnformatted("Active object: None");
        return;
    }

    PushID("EntityControls");
    Text("Active entity: %s", GetName(r, active_entity).c_str());
    Indent();

    if (const auto *node = r.try_get<SceneNode>(active_entity)) {
        if (auto parent_entity = node->Parent; parent_entity != entt::null) {
            AlignTextToFramePadding();
            Text("Parent: %s", GetName(r, parent_entity).c_str());
        }
    }

    if (const auto *instance = r.try_get<Instance>(active_entity)) {
        Text("Instance of: %s", GetName(r, instance->Entity).c_str());
    }
    if (const auto *armature_modifier = r.try_get<ArmatureModifier>(active_entity)) {
        Text("Armature data: %s", GetName(r, armature_modifier->ArmatureEntity).c_str());
        if (armature_modifier->ArmatureObjectEntity != entt::null) {
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
    const bool is_mesh_instance = active_instance && r.all_of<Mesh>(active_instance->Entity);
    if (is_mesh_instance) {
        const auto active_mesh_entity = active_instance->Entity;
        const auto &active_mesh = r.get<const Mesh>(active_mesh_entity);
        TextUnformatted(
            std::format("Vertices | Edges | Faces: {:L} | {:L} | {:L}", active_mesh.VertexCount(), active_mesh.EdgeCount(), active_mesh.FaceCount()).c_str()
        );
    } else if (const auto *armature_object = r.try_get<ArmatureObject>(active_entity)) {
        const auto &armature = r.get<const Armature>(armature_object->Entity);
        Text("Bones: %zu", armature.Bones.size());
    }
    Unindent();
    const bool is_bone_edit = r.get<const Interaction>(viewport).Mode == InteractionMode::Edit && active_bone_entity != entt::null && r.all_of<BoneDisplayScale>(active_bone_entity);
    if (CollapsingHeader("Transform")) {
        if (is_bone_edit) {
            const auto &wt = r.get<WorldTransform>(active_bone_entity);
            const float bone_length = r.get<BoneDisplayScale>(active_bone_entity).Value;

            vec3 head = wt.P;
            vec3 tail = head + glm::rotate(wt.R, vec3{0, bone_length, 0});
            vec3 dir;
            float roll;
            BoneMat3ToVecRoll(glm::mat3_cast(wt.R), dir, roll);
            float roll_deg = glm::degrees(roll);
            float length = bone_length;

            bool changed = DragFloat3("Head", &head[0], 0.01f);
            changed |= DragFloat3("Tail", &tail[0], 0.01f);
            if (DragFloat("Roll", &roll_deg, 1.f)) {
                roll = glm::radians(roll_deg);
                changed = true;
            }
            if (DragFloat("Length", &length, 0.01f, 0.001f, 0.f)) {
                tail = head + glm::normalize(tail - head) * std::max(length, 1e-4f);
                changed = true;
            }

            if (changed) {
                const auto new_dir = tail - head;
                if (const auto new_length = glm::length(new_dir); new_length > 1e-6f) {
                    const auto new_rot = glm::quat_cast(BoneVecRollToMat3(new_dir, roll));
                    const auto pd = ToTransform(GetParentDelta(r, active_bone_entity));
                    action::Emit(action::bone::SetEditHeadTailRoll{
                        .LocalP = glm::conjugate(pd.R) * ((head - pd.P) / pd.S),
                        .LocalR = glm::conjugate(pd.R) * new_rot,
                        .DisplayScale = new_length,
                    });
                }
            }
        } else {
            // Standard PRS transform editor (objects, pose mode bones).
            // In Pose mode, edit the active bone rather than the armature.
            const bool is_pose_bone = r.get<const Interaction>(viewport).Mode == InteractionMode::Pose && active_bone_entity != entt::null;
            const auto transform_entity = is_pose_bone ? active_bone_entity : active_entity;
            ui::Edit transform_edit{r, transform_entity};
            transform_edit.Drag<&Transform::P>("Position", 0.01f);
            // Rotation editor (RotationUiVariant is reactively created; may not exist yet on the first frame)
            if (const auto *rotation_ui_ptr = r.try_get<const RotationUiVariant>(transform_entity)) {
                int mode_i = rotation_ui_ptr->index();
                const char *modes[]{"Quat (WXYZ)", "XYZ Euler", "Axis Angle"};
                if (Combo("Rotation mode", &mode_i, modes, IM_ARRAYSIZE(modes)))
                    action::Emit(action::view::SetRotationUiMode{mode_i});
                auto ui_local = *rotation_ui_ptr;
                std::visit(
                    overloaded{
                        [&](RotationQuat &v) {
                            if (DragFloat4("Rotation (quat WXYZ)", &v.Value[0], 0.01f))
                                action::Emit(action::view::SetTransformRotationFromUi{glm::normalize(v.Value), ui_local});
                        },
                        [&](RotationEuler &v) {
                            if (DragFloat3("Rotation (XYZ Euler, deg)", &v.Value[0], 1.f)) {
                                const auto rads = glm::radians(v.Value);
                                action::Emit(action::view::SetTransformRotationFromUi{
                                    glm::normalize(glm::quat_cast(glm::eulerAngleXYZ(rads.x, rads.y, rads.z))),
                                    ui_local,
                                });
                            }
                        },
                        [&](RotationAxisAngle &v) {
                            bool changed = DragFloat3("Rotation axis (XYZ)", &v.Value[0], 0.01f);
                            changed |= DragFloat("Angle (deg)", &v.Value.w, 1.f);
                            if (changed) {
                                const auto axis = glm::normalize(vec3{v.Value});
                                const auto angle = glm::radians(v.Value.w);
                                action::Emit(action::view::SetTransformRotationFromUi{
                                    glm::normalize(quat{std::cos(angle / 2), axis * std::sin(angle / 2)}),
                                    ui_local,
                                });
                            }
                        },
                    },
                    ui_local
                );
            }

            const bool frozen = r.all_of<ScaleLocked>(transform_entity);
            if (frozen) BeginDisabled();
            transform_edit.Drag<&Transform::S>(std::format("Scale{}", frozen ? " (frozen)" : "").c_str(), 0.01f, 0.01f, 10);
            if (frozen) EndDisabled();
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
            gizmo_edit.Check<&TransformGizmoState::Config, &TransformGizmo::Config::Snap>("Snap");
            if (gizmo_state.Config.Snap) {
                SameLine();
                // todo link/unlink snap values
                gizmo_edit.Drag<&TransformGizmoState::Config, &TransformGizmo::Config::SnapValue>("Snap", 1.f, 0.01f, 100.f);
            }
        }
        Spacing();
        if (TreeNode("Debug")) {
            if (const auto label = TransformGizmo::ToString(r.get<const GizmoInteraction>(viewport)); label != "") {
                Text("%s op: %s", TransformGizmo::IsUsing(r, viewport) ? "Active" : "Hovered", label.data());
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
    if (active_bone_entity != entt::null && CollapsingHeader("Bone Constraints")) {
        PushID("BoneConstraints");
        const auto *constraints = r.try_get<const BoneConstraints>(active_bone_entity);
        const size_t stack_size = constraints ? constraints->Stack.size() : 0;
        std::optional<uint32_t> delete_index;
        for (size_t i = 0; i < stack_size; ++i) {
            PushID(int(i));
            const auto &c = constraints->Stack[i];
            const char *type_label = std::visit([]<typename T>(const T &) {
                if constexpr (std::is_same_v<T, CopyTransformsData>) return "Copy Transforms";
                else if constexpr (std::is_same_v<T, ChildOfData>) return "Child Of";
                else return "?";
            },
                                                c.Data);
            const bool expanded = TreeNodeEx("##node", ImGuiTreeNodeFlags_SpanTextWidth, "%s", type_label);
            SameLine();
            if (SmallButton("X")) delete_index = uint32_t(i);
            if (expanded) {
                const auto *cur_name = c.TargetEntity != entt::null && r.valid(c.TargetEntity) ? r.try_get<const Name>(c.TargetEntity) : nullptr;
                const std::string preview = c.TargetEntity == entt::null ? "None" :
                    cur_name && !cur_name->Value.empty()                 ? cur_name->Value :
                                                                           IdString(c.TargetEntity);
                if (BeginCombo("Target", preview.c_str())) {
                    if (Selectable("None", c.TargetEntity == entt::null))
                        action::Emit(action::bone::SetConstraintTarget{uint32_t(i), entt::null});
                    for (auto [te, kind, name] : r.view<const ObjectKind, const Name>().each()) {
                        if (r.any_of<BoneIndex, BoneSubPartOf, BoneJoint, SubElementOf>(te)) continue;
                        const std::string label = name.Value.empty() ? IdString(te) : name.Value;
                        if (Selectable(label.c_str(), te == c.TargetEntity))
                            action::Emit(action::bone::SetConstraintTarget{uint32_t(i), te});
                    }
                    EndCombo();
                }
                if (std::holds_alternative<ChildOfData>(c.Data)) {
                    if (Button("Set Inverse") && c.TargetEntity != entt::null && r.valid(c.TargetEntity)) {
                        // Bake inverse(target_world) * bone_world so current relative pose becomes the new rest.
                        const auto *twt = r.try_get<const WorldTransform>(c.TargetEntity);
                        const auto *bwt = r.try_get<const WorldTransform>(active_bone_entity);
                        if (twt && bwt) {
                            const mat4 inv = glm::inverse(ToMatrix(*twt)) * ToMatrix(*bwt);
                            action::Emit(action::bone::SetConstraintChildOfInverse{uint32_t(i), std::make_unique<mat4>(inv)});
                        }
                    }
                    SameLine();
                    if (Button("Clear Inverse"))
                        action::Emit(action::bone::SetConstraintChildOfInverse{uint32_t(i), std::make_unique<mat4>(I4)});
                }
                if (float influence = c.Influence; SliderFloat("Influence", &influence, 0.f, 1.f))
                    action::Emit(action::bone::SetConstraintInfluence{uint32_t(i), influence});
                TreePop();
            }
            PopID();
        }
        if (delete_index) action::Emit(action::bone::DeleteConstraint{*delete_index});
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
                if (auto primitive_mesh = PrimitiveEditor(*prim_shape)) {
                    action::Emit(action::object::ReplaceMesh{std::make_unique<MeshData>(std::move(*primitive_mesh))});
                }
            }
            if (frozen) EndDisabled();
        }

        if (CollapsingHeader("Material")) {
            const auto &active_mesh = r.get<const Mesh>(active_mesh_entity);
            auto &material_store = r.ctx().get<MaterialStore>();
            auto &texture_store = textures;
            std::span<const uint32_t> primitive_materials = meshes.GetPrimitiveMaterialIndices(active_mesh.GetStoreId());
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
                    action::Emit(action::Replace<MeshMaterialSlotSelection>{active_mesh_entity, {slot_primitive}});
                }

                BeginChild("MaterialSlots", ImVec2(0, 110), true);
                for (uint32_t primitive_index = 0; primitive_index < primitive_materials.size(); ++primitive_index) {
                    const uint32_t material_index = std::min(primitive_materials[primitive_index], material_count - 1);
                    if (const auto label = std::format("Slot {:L}: {}", primitive_index, material_name(material_index));
                        Selectable(label.c_str(), slot_primitive == primitive_index) && slot_primitive != primitive_index) {
                        action::Emit(action::Replace<MeshMaterialSlotSelection>{active_mesh_entity, {primitive_index}});
                        slot_primitive = primitive_index;
                    }
                }
                EndChild();

                const auto *pending_assignment = r.try_get<const MeshMaterialAssignment>(active_mesh_entity);
                uint32_t material_index = pending_assignment && pending_assignment->PrimitiveIndex == slot_primitive ?
                    pending_assignment->MaterialIndex :
                    primitive_materials[slot_primitive];
                material_index = std::min(material_index, material_count - 1);
                if (const auto assigned_material_name = material_name(material_index);
                    BeginCombo("Assigned material", assigned_material_name.c_str())) {
                    for (uint32_t i = 0; i < material_count; ++i) {
                        if (const auto option_name = material_name(i);
                            Selectable(option_name.c_str(), material_index == i)) {
                            action::Emit(action::Replace<MeshMaterialAssignment>{active_mesh_entity, {slot_primitive, i}});
                            material_index = i;
                        }
                    }
                    EndCombo();
                }

                auto &material = materials[material_index];
                const auto edit_texture_slot = [&](const char *label, uint32_t &slot) {
                    std::string preview = "None";
                    bool has_match = false;
                    for (const auto &texture : texture_store.Textures) {
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
                        for (const auto &texture : texture_store.Textures) {
                            if (Selectable(texture.Name.c_str(), slot == texture.SamplerSlot)) {
                                slot = texture.SamplerSlot;
                                changed = true;
                            }
                        }
                        EndCombo();
                    }
                    return changed;
                };
                const auto edit_uv_transform = [&](const char *offset_label, const char *scale_label, const char *rotation_label, vec2 &offset, vec2 &scale, float &rotation) {
                    bool changed = false;
                    changed |= DragFloat2(offset_label, &offset.x, 0.01f);
                    changed |= DragFloat2(scale_label, &scale.x, 0.01f);
                    changed |= DragFloat(rotation_label, &rotation, 0.01f);
                    return changed;
                };
                const auto edit_texture_info = [&](const char *label, TextureInfo &tex) {
                    bool changed = false;
                    changed |= edit_texture_slot(std::format("{} texture", label).c_str(), tex.Slot);
                    changed |= SliderUInt(std::format("{} UV set", label).c_str(), &tex.TexCoord, 0u, 3u);
                    changed |= edit_uv_transform(
                        std::format("{} UV offset", label).c_str(),
                        std::format("{} UV scale", label).c_str(),
                        std::format("{} UV rotation", label).c_str(),
                        tex.UvOffset, tex.UvScale, tex.UvRotation
                    );
                    return changed;
                };
                bool material_changed = false;
                material_changed |= ColorEdit4("Base color", &material.BaseColorFactor.x);
                material_changed |= SliderFloat("Metallic", &material.MetallicFactor, 0.f, 1.f);
                material_changed |= SliderFloat("Roughness", &material.RoughnessFactor, 0.f, 1.f);
                material_changed |= edit_texture_info("Base color", material.BaseColorTexture);
                material_changed |= edit_texture_info("Metallic-roughness", material.MetallicRoughnessTexture);
                material_changed |= edit_texture_info("Normal", material.NormalTexture);
                material_changed |= SliderFloat("Normal scale", &material.NormalScale, -2.f, 2.f);
                material_changed |= edit_texture_info("Occlusion", material.OcclusionTexture);
                material_changed |= SliderFloat("Occlusion strength", &material.OcclusionStrength, 0.f, 1.f);
                material_changed |= ColorEdit3("Emissive", &material.EmissiveFactor.x);
                material_changed |= edit_texture_info("Emissive", material.EmissiveTexture);

                static constexpr std::array alpha_mode_labels{"Opaque", "Mask", "Blend"};
                int alpha_mode = std::clamp<int>(int(material.AlphaMode), 0, int(alpha_mode_labels.size() - 1));
                if (Combo("Alpha mode", &alpha_mode, alpha_mode_labels.data(), int(alpha_mode_labels.size()))) {
                    material.AlphaMode = MaterialAlphaMode(alpha_mode);
                    material_changed = true;
                }
                if (material.AlphaMode == MaterialAlphaMode::Mask) {
                    material_changed |= SliderFloat("Alpha cutoff", &material.AlphaCutoff, 0.f, 1.f);
                }
                bool double_sided = material.DoubleSided != 0u;
                if (Checkbox("Double sided", &double_sided)) {
                    material.DoubleSided = double_sided ? 1u : 0u;
                    material_changed = true;
                }

                // IOR — always visible; affects Fresnel reflectance even for non-transmissive dielectrics.
                material_changed |= SliderFloat("IOR", &material.Ior, 1.0f, 3.0f, "%.3f");

                auto pbr_features_mask = r.all_of<PbrMeshFeatures>(active_mesh_entity) ? r.get<const PbrMeshFeatures>(active_mesh_entity).Mask : 0u;
                bool pbr_features_changed = false;
                const auto feature_toggle = [&](const char *label, PbrFeature feature) {
                    bool enabled = HasFeature(pbr_features_mask, feature);
                    if (Checkbox(label, &enabled)) {
                        if (enabled) pbr_features_mask |= feature;
                        else pbr_features_mask &= ~uint32_t(feature);
                        pbr_features_changed = true;
                    }
                    return enabled;
                };

                if (feature_toggle("Transmission", PbrFeature::Transmission)) {
                    SeparatorText("Transmission");
                    material_changed |= SliderFloat("Transmission factor", &material.Transmission.Factor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Transmission", material.Transmission.Texture);
                    material_changed |= SliderFloat("Dispersion", &material.Dispersion, 0.f, 1.f);
                    // Volume (only meaningful with transmission)
                    material_changed |= SliderFloat("Thickness", &material.Volume.ThicknessFactor, 0.f, 10.f);
                    material_changed |= edit_texture_info("Thickness", material.Volume.ThicknessTexture);
                    material_changed |= ColorEdit3("Attenuation color", &material.Volume.AttenuationColor.x);
                    material_changed |= DragFloat("Attenuation distance", &material.Volume.AttenuationDistance, 0.01f, 0.f, 0.f, material.Volume.AttenuationDistance <= 0.f ? "Infinite" : "%.3f m");
                }

                if (feature_toggle("Diffuse transmission", PbrFeature::DiffuseTrans)) {
                    SeparatorText("Diffuse transmission");
                    material_changed |= SliderFloat("Diffuse transmission factor", &material.DiffuseTransmission.Factor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Diffuse transmission", material.DiffuseTransmission.Texture);
                    material_changed |= ColorEdit3("Diffuse transmission color", &material.DiffuseTransmission.ColorFactor.x);
                    material_changed |= edit_texture_info("Diffuse transmission color", material.DiffuseTransmission.ColorTexture);
                }

                if (feature_toggle("Clearcoat", PbrFeature::Clearcoat)) {
                    SeparatorText("Clearcoat");
                    material_changed |= SliderFloat("Clearcoat factor", &material.Clearcoat.Factor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Clearcoat", material.Clearcoat.Texture);
                    material_changed |= SliderFloat("Clearcoat roughness", &material.Clearcoat.RoughnessFactor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Clearcoat roughness", material.Clearcoat.RoughnessTexture);
                    material_changed |= edit_texture_info("Clearcoat normal", material.Clearcoat.NormalTexture);
                    material_changed |= SliderFloat("Clearcoat normal scale", &material.Clearcoat.NormalScale, -2.f, 2.f);
                }

                if (feature_toggle("Anisotropy", PbrFeature::Anisotropy)) {
                    SeparatorText("Anisotropy");
                    material_changed |= SliderFloat("Anisotropy strength", &material.Anisotropy.Strength, 0.f, 1.f);
                    material_changed |= SliderFloat("Anisotropy rotation", &material.Anisotropy.Rotation, 0.f, 6.2832f, "%.3f rad");
                    material_changed |= edit_texture_info("Anisotropy", material.Anisotropy.Texture);
                }

                if (feature_toggle("Sheen", PbrFeature::Sheen)) {
                    SeparatorText("Sheen");
                    material_changed |= ColorEdit3("Sheen color", &material.Sheen.ColorFactor.x);
                    material_changed |= edit_texture_info("Sheen color", material.Sheen.ColorTexture);
                    material_changed |= SliderFloat("Sheen roughness", &material.Sheen.RoughnessFactor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Sheen roughness", material.Sheen.RoughnessTexture);
                }

                if (feature_toggle("Iridescence", PbrFeature::Iridescence)) {
                    SeparatorText("Iridescence");
                    material_changed |= SliderFloat("Iridescence factor", &material.Iridescence.Factor, 0.f, 1.f);
                    material_changed |= edit_texture_info("Iridescence", material.Iridescence.Texture);
                    material_changed |= SliderFloat("Iridescence IOR", &material.Iridescence.Ior, 1.0f, 5.0f);
                    material_changed |= SliderFloat("Thickness min", &material.Iridescence.ThicknessMinimum, 0.f, 1000.f, "%.0f nm");
                    material_changed |= SliderFloat("Thickness max", &material.Iridescence.ThicknessMaximum, 1.f, 1000.f, "%.0f nm");
                    material_changed |= edit_texture_info("Iridescence thickness", material.Iridescence.ThicknessTexture);
                }

                if (pbr_features_changed) action::Emit(action::object::SetPbrMeshFeaturesMask{pbr_features_mask});
                if (material_changed) action::Emit(action::Replace<MaterialDirty>{viewport, {material_index}});
            }
        }
    }
    if (const auto *cd = r.try_get<const Camera>(active_entity)) {
        if (CollapsingHeader("Camera")) {
            // Use the camera's distance from world origin as the conversion distance.
            const float distance = std::max(glm::length(r.get<WorldTransform>(active_entity).P), 1.f);
            auto edited = *cd;
            if (RenderCameraLensEditor(edited, distance)) action::Emit(action::ReplaceActive<Camera>{edited});
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
        auto light = GetLights(r)[r.get<const LightIndex>(active_entity).Value];
        bool changed{false};

        const char *type_names[]{"Directional", "Point", "Spot"};
        if (int type_i = int(light.Type); Combo("Type", &type_i, type_names, IM_ARRAYSIZE(type_names))) {
            auto next = Defaults::MakePunctualLight(PunctualLightType(type_i));
            next.TransformSlotOffset = light.TransformSlotOffset;
            next.Color = light.Color;
            next.Intensity = light.Intensity;
            light = next;
            changed = true;
        }
        changed |= ColorEdit3("Color", &light.Color.x);
        changed |= SliderFloat("Intensity", &light.Intensity, 0.f, 1000.f, "%.2f");
        if (light.Type == PunctualLightType::Point || light.Type == PunctualLightType::Spot) {
            bool infinite_range = light.Range <= 0.f;
            if (Checkbox("Infinite range", &infinite_range)) {
                light.Range = infinite_range ? 0.f : 100.f;
                changed = true;
            }
            changed |= !infinite_range && SliderFloat("Range", &light.Range, 0.01f, 1000.f, "%.2f");
        }
        if (light.Type == PunctualLightType::Spot) {
            float outer_deg = std::clamp(glm::degrees(AngleFromCos(light.OuterConeCos)), 0.f, 90.f);
            const float inner_deg = std::clamp(glm::degrees(AngleFromCos(light.InnerConeCos)), 0.f, outer_deg);
            float blend = outer_deg > 1e-4f ? std::clamp(1.f - inner_deg / outer_deg, 0.f, 1.f) : 0.f;
            const bool size_changed = SliderFloat("Size", &outer_deg, 0.f, 90.f, "%.1f deg");
            const bool blend_changed = SliderFloat("Blend", &blend, 0.f, 1.f, "%.2f");
            if (size_changed || blend_changed) {
                const float outer_rad = glm::radians(std::clamp(outer_deg, 0.f, 90.f));
                light.OuterConeCos = std::cos(outer_rad);
                light.InnerConeCos = std::cos(outer_rad * (1.f - std::clamp(blend, 0.f, 1.f)));
                changed = true;
            }
        }
        if (changed) action::Emit(action::ReplaceActive<PunctualLight>{light});
    }
    // Audio controls (mesh instance = sound object or eligible to become one; microphone)
    if (const auto *instance = r.try_get<Instance>(active_entity); instance && r.all_of<Mesh>(instance->Entity)) {
        const bool has_sound = r.all_of<SoundVerticesModel>(active_entity);
        if (CollapsingHeader("Audio", has_sound ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
            DrawObjectAudioControls(r, viewport, active_entity, GetMeshEntity(r, active_entity), r.get<const SelectionBitsetRef>(viewport).Value.data());
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
            auto target = entt::entity{entt::null};
            for (const auto &[e, active] : r.view<const RealImpactActiveMicrophone>().each()) {
                if (active.Entity == active_entity) {
                    target = e;
                    break;
                }
            }
            if (target == entt::null) {
                for (auto [e, _, inst] : r.view<SoundVerticesModel, Instance>().each()) {
                    if (r.all_of<Path>(inst.Entity)) {
                        target = e;
                        break;
                    }
                }
            }
            if (target == entt::null) {
                TextUnformatted("No matching sound object found.");
            } else {
                const auto target_name = GetName(r, target);
                const auto *active = r.try_get<const RealImpactActiveMicrophone>(target);
                const bool is_active = active && active->Entity == active_entity;
                if (is_active) {
                    Text("Active for: %s", target_name.c_str());
                } else if (Button(std::format("Set as active for {}", target_name).c_str())) {
                    action::Emit(action::audio::ActivateRealImpactMicrophone{target, active_entity});
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
        const auto mesh_entity = active_instance ? active_instance->Entity : entt::null;
        const auto extras = [&](const auto *src, gltf::ExtrasCategory cat) -> std::optional<std::string_view> {
            return src ? gltf::GetExtras(*sa, cat, src->Value) : std::nullopt;
        };
        const std::pair<const char *, std::optional<std::string_view>> sections[]{
            {"Extras (Node)", extras(r.try_get<const SourceNodeIndex>(active_entity), gltf::ExtrasCategory::Nodes)},
            {"Extras (Mesh)", extras(mesh_entity != entt::null ? r.try_get<const SourceMeshIndex>(mesh_entity) : nullptr, gltf::ExtrasCategory::Meshes)},
            {"Extras (Camera)", extras(r.try_get<const SourceCameraIndex>(active_entity), gltf::ExtrasCategory::Cameras)},
            {"Extras (Light)", extras(r.try_get<const SourceLightIndex>(active_entity), gltf::ExtrasCategory::Lights)},
        };
        const auto *mesh_layout = mesh_entity != entt::null ? r.try_get<const MeshSourceLayout>(mesh_entity) : nullptr;
        const bool any_extras = std::ranges::any_of(sections, [](const auto &s) { return s.second.has_value(); });
        if ((any_extras || mesh_layout) && CollapsingHeader("glTF metadata")) {
            for (const auto &[label, json] : sections) {
                if (json) RenderJsonBlock(label, *json);
            }
            if (mesh_layout && !mesh_layout->VertexCounts.empty()) {
                SeparatorText("Mesh source layout");
                Text("Primitives: %zu", mesh_layout->VertexCounts.size());
                for (size_t i = 0; i < mesh_layout->VertexCounts.size(); ++i) {
                    const uint32_t flags = i < mesh_layout->AttributeFlags.size() ? mesh_layout->AttributeFlags[i] : 0u;
                    const bool indexed = i < mesh_layout->HasSourceIndices.size() && mesh_layout->HasSourceIndices[i];
                    Text("[%zu] verts:%u%s attrs:%s", i, mesh_layout->VertexCounts[i], indexed ? "" : " (non-indexed)", AttributeFlagsString(flags).c_str());
                }
                if (!mesh_layout->MorphTangentDeltas.empty()) {
                    Text("Morph tangent deltas: %zu", mesh_layout->MorphTangentDeltas.size());
                }
            }
        }
    }

    PopID();
}

void RenderControls(entt::registry &r, entt::entity viewport) {
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
                const bool active_is_armature_rc = FindArmatureObject(r, active_entity_rc) != entt::null;
                const bool edit_allowed = AllSelectedAreMeshes(r) || active_is_armature_rc;
                const bool pose_allowed = active_is_armature_rc;
                for (const auto mode : r.get<const EnabledInteractionModes>(viewport).Value) {
                    if (mode == InteractionMode::Edit && !edit_allowed) continue;
                    if (mode == InteractionMode::Pose && !pose_allowed) continue;
                    SameLine();
                    interaction_mode_changed |= RadioButton(to_string(mode).c_str(), &interaction_mode_value, int(mode));
                }
                if (interaction_mode_changed) action::Emit(action::view::SetInteractionMode{.Mode = InteractionMode(interaction_mode_value)});
                if (interaction_mode == InteractionMode::Edit || interaction_mode == InteractionMode::Excite) {
                    bool orbit = r.get<const OrbitToActive>(viewport).Value;
                    if (Checkbox("Orbit to active", &orbit)) r.replace<OrbitToActive>(viewport, orbit);
                }
                if (interaction_mode == InteractionMode::Edit) {
                    bool xray = r.get<const SelectionXRay>(viewport).Value;
                    if (Checkbox("X-ray selection", &xray)) r.replace<SelectionXRay>(viewport, xray);
                }
                if (interaction_mode == InteractionMode::Edit && !active_is_armature_rc) {
                    AlignTextToFramePadding();
                    TextUnformatted("Edit mode:");
                    auto type_interaction_mode = int(edit_mode);
                    for (const auto element : Elements) {
                        auto name = Capitalize(label(element));
                        SameLine();
                        if (RadioButton(name.c_str(), &type_interaction_mode, int(element))) {
                            action::Emit(action::view::SetEditMode{.Mode = element});
                        }
                    }
                    if (const auto active_entity = FindActiveEntity(r); active_entity != entt::null) {
                        if (const auto *instance = r.try_get<Instance>(active_entity); instance && r.all_of<Mesh>(instance->Entity)) {
                            const auto *br = r.try_get<const MeshSelectionBitsetRange>(instance->Entity);
                            const uint32_t selected_count = br ?
                                selection::CountSelected(r.get<const SelectionBitsetRef>(viewport).Value.data(), br->Offset, br->Count) :
                                0;
                            Text("Editing %s: %u selected", label(edit_mode).data(), selected_count);
                        }
                    }
                }
                PopID();
            }
            if (auto *sa = r.try_get<gltf::SourceAssets>(viewport); sa && sa->Scenes.size() > 1) {
                const auto &active_name = sa->Scenes[sa->ActiveSceneIndex].Name;
                const auto preview = NamedOr(active_name, "Scene ", sa->ActiveSceneIndex);
                if (BeginCombo("Scene", preview.c_str())) {
                    for (uint32_t i = 0; i < sa->Scenes.size(); ++i) {
                        const bool selected = i == sa->ActiveSceneIndex;
                        if (Selectable(NamedOr(sa->Scenes[i].Name, "Scene ", i).c_str(), selected) && !selected) {
                            gltf::SwitchActiveScene(r, viewport, i);
                        }
                        if (selected) SetItemDefaultFocus();
                    }
                    EndCombo();
                }
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
                if (Button("Camera")) action::Emit(action::object::AddCamera{.Info = std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive}), .Props = {}});
                SameLine();
                if (Button("Light")) action::Emit(action::object::AddLight{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Select = MeshInstanceCreateInfo::SelectBehavior::Exclusive})});
            }
            if (auto *mv = r.try_get<MaterialVariants>(viewport); mv && !mv->Names.empty() && CollapsingHeader("Material variants")) {
                const auto active = mv->Active;
                const auto preview = active ? NamedOr(mv->Names[*active], "Variant ", *active) : std::string{"Default"};
                const auto set_variant = [&](std::optional<uint32_t> v) {
                    if (active != v) action::Emit(action::UpdateOf<&MaterialVariants::Active>(viewport, v));
                };
                if (BeginCombo("Active variant", preview.c_str())) {
                    if (Selectable("Default", !active)) set_variant({});
                    for (uint32_t i = 0; i < mv->Names.size(); ++i) {
                        if (Selectable(NamedOr(mv->Names[i], "Variant ", i).c_str(), active == i)) set_variant(i);
                    }
                    EndCombo();
                }
            }
            if (!r.storage<Selected>().empty()) {
                SeparatorText("Selection actions");
                std::vector<entt::entity> selected_mesh_instances;
                for (const auto entity : r.view<const Selected, const Instance>()) {
                    if (!r.all_of<SubElementOf>(entity)) selected_mesh_instances.emplace_back(entity);
                }

                if (!selected_mesh_instances.empty()) {
                    const bool any_visible = any_of(selected_mesh_instances, [&](entt::entity e) { return r.all_of<RenderInstance>(e); });
                    const bool any_hidden = any_of(selected_mesh_instances, [&](entt::entity e) { return !r.all_of<RenderInstance>(e); });
                    const bool mixed_visible = any_visible && any_hidden;
                    if (mixed_visible) PushItemFlag(ImGuiItemFlags_MixedValue, true);
                    if (bool set_visible = any_visible && !any_hidden; Checkbox("Visible", &set_visible)) action::Emit(action::object::SetSelectedVisible{set_visible});
                    if (mixed_visible) PopItemFlag();

                    const auto face_mesh_entities = selection::GetSelectedMeshEntities(r) |
                        std::views::filter([&](entt::entity me) { return r.get<const Mesh>(me).FaceCount() > 0; }) |
                        to<std::vector>();
                    if (!face_mesh_entities.empty()) {
                        const bool any_smooth = any_of(face_mesh_entities, [&](entt::entity me) { return r.all_of<SmoothShading>(me); });
                        const bool any_flat = any_of(face_mesh_entities, [&](entt::entity me) { return !r.all_of<SmoothShading>(me); });
                        const bool mixed_smooth = any_smooth && any_flat;
                        SameLine();
                        if (mixed_smooth) PushItemFlag(ImGuiItemFlags_MixedValue, true);
                        if (bool set_smooth = any_smooth && !any_flat; Checkbox("Smooth shading", &set_smooth)) action::Emit(action::object::SetSelectedSmoothShading{set_smooth});
                        if (mixed_smooth) PopItemFlag();
                    }
                }
                if (CanDuplicate(r, viewport) && Button("Duplicate")) Duplicate(r, viewport);
                if (CanDuplicateLinked(r, viewport)) {
                    SameLine();
                    if (Button("Duplicate linked")) action::Emit(action::object::DuplicateLinked{});
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
                    color.a = 1.f;
                    f.Set<&ViewportDisplay::ClearColor>(color);
                }
            }
            if (Button("Recompile shaders")) r.emplace_or_replace<PendingShaderRecompile>(viewport);

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
                if (!r.view<const TetMeshData>().empty()) f.Check<&ViewportDisplay::ShowTetWireframe>("Tet wireframe");
            }
            {
                using VC = ViewportThemeColors;
                using AC = AxisThemeColors;
                SeparatorText("Viewport theme");
                const auto &theme = r.get<const ViewportTheme>(viewport);
                if (Button("Reset##ViewportTheme")) action::Emit(action::view::ResetViewportTheme{});
                auto c = f.Sub<&ViewportTheme::Colors>();
                c.Color<&VC::Grid>("Grid");
                c.Color<&VC::Wire>("Wire");
                c.Color<&VC::WireEdit>("Wire edit");
                c.Color<&VC::ObjectActive>("Object active");
                c.Color<&VC::ObjectSelected>("Object selected");
                c.Color<&VC::Light>("Light");
                c.Color<&VC::Vertex>("Vertex");
                c.Color<&VC::VertexSelected>("Vertex selected");
                c.Color<&VC::EdgeSelectedIncidental>("Edge selected (incidental)");
                c.Color<&VC::EdgeSelected>("Edge selected");
                c.Color<&VC::FaceSelectedIncidental>("Face selected (incidental)");
                c.Color<&VC::FaceSelected>("Face selected");
                c.Color<&VC::ElementActive>("Element active");
                c.Color<&VC::ElementExcited>("Element excited");
                c.Color<&VC::FaceNormal>("Face normal");
                c.Color<&VC::VertexNormal>("Vertex normal");
                c.Color<&VC::BoneSolid>("Bone solid");
                c.Color<&VC::BonePose>("Bone pose");
                c.Color<&VC::BonePoseActive>("Bone pose active");
                c.Color<&VC::Transform>("Transform");
                SeparatorText("Axis colors");
                auto a = f.Sub<&ViewportTheme::AxisColors>();
                a.Color<&AC::X>("Axis X");
                a.Color<&AC::Y>("Axis Y");
                a.Color<&AC::Z>("Axis Z");
                // UI edits "full" width; storage is half-width.
                if (float full_width = theme.EdgeWidth * 2.f; SliderFloat("Edge width", &full_width, 0.5f, 4.f))
                    f.Set<&ViewportTheme::EdgeWidth>(full_width / 2.f);
                if (uint32_t v = theme.SilhouetteEdgeWidth; SliderUInt("Silhouette edge width", &v, 1, 4))
                    f.Set<&ViewportTheme::SilhouetteEdgeWidth>(v);
            }
            EndTabItem();
        }

        if (BeginTabItem("Camera")) {
            const auto &camera = r.get<const ViewCamera>(viewport);
            const auto extent = r.get<const ViewportExtent>(viewport).Value;
            const float viewport_aspect = extent.x == 0 || extent.y == 0 ? 1.f : float(extent.x) / float(extent.y);
            if (Button("Reset##Camera")) action::Emit(action::view::ResetViewCamera{});
            if (vec3 target = camera.Target; SliderFloat3("Target", &target.x, -10, 10))
                action::Emit(action::view::SetViewCameraTarget{target});
            if (Camera lens = camera.Data; RenderCameraLensEditor(lens, camera.Distance, viewport_aspect))
                action::Emit(action::view::SetViewCameraLens{lens});
            EndTabItem();
        }

        if (BeginTabItem("Physics")) {
            physics_ui::RenderTab(r, viewport);
            EndTabItem();
        }

        if (const auto *sa = r.try_get<const gltf::SourceAssets>(viewport)) {
            if (BeginTabItem("glTF metadata")) {
                const bool has_asset = !sa->Generator.empty() || !sa->Copyright.empty() || !sa->MinVersion.empty() || !sa->AssetExtras.empty() || !sa->AssetExtensions.empty();
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
                if (CollapsingHeader("Counts", ImGuiTreeNodeFlags_DefaultOpen)) {
                    Text("Source nodes: %zu", r.view<const SourceNodeIndex>().size());
                    Text("Meshes: %zu", r.view<const MeshName>().size());
                    Text("Materials: %zu", sa->MaterialMetas.size());
                    Text("Textures: %zu", sa->Textures.size());
                    Text("Images: %zu", sa->Images.size());
                    Text("Samplers: %zu", sa->Samplers.size());
                    Text("Animations: %zu", sa->AnimationOrder.size());
                    Text("Skins: %zu", r.view<const SkinName>().size());
                    Text("Cameras: %zu", r.view<const CameraName>().size());
                    Text("Lights: %zu", r.view<const LightName>().size());
                    Text("Physics materials: %zu", r.view<const SourcePhysicsMaterialIndex>().size());
                    Text("Collision filters: %zu", r.view<const SourceCollisionFilterIndex>().size());
                    Text("Physics joints: %zu", r.view<const SourcePhysicsJointDefIndex>().size());
                }
                if (!sa->Images.empty() && CollapsingHeader("Image registry")) {
                    if (BeginTable("Images", 5, MetadataTableFlags)) {
                        TableSetupColumn("#");
                        TableSetupColumn("Name");
                        TableSetupColumn("Mime");
                        TableSetupColumn("Source");
                        TableSetupColumn("Bytes");
                        TableHeadersRow();
                        for (size_t i = 0; i < sa->Images.size(); ++i) {
                            const auto &img = sa->Images[i];
                            TableNextRow();
                            TableNextColumn();
                            Text("%zu", i);
                            TableNextColumn();
                            TextUnformatted(img.Name.c_str());
                            TableNextColumn();
                            TextUnformatted(MimeTypeName(img.MimeType).data());
                            TableNextColumn();
                            if (img.SourceDataUri) TextUnformatted("data URI");
                            else if (!img.SourceAbsPath.empty()) TextUnformatted(img.SourceAbsPath.c_str());
                            else TextUnformatted("embedded");
                            TableNextColumn();
                            if (img.Bytes.empty()) TextUnformatted(img.SourceAbsPath.empty() ? "—" : "external");
                            else Text("%zu", img.Bytes.size());
                            if (img.IsDirty) {
                                SameLine();
                                TextUnformatted("(dirty)");
                            }
                        }
                        EndTable();
                    }
                }
                if (!sa->Textures.empty() && CollapsingHeader("Texture registry")) {
                    if (BeginTable("Textures", 4, MetadataTableFlags)) {
                        TableSetupColumn("#");
                        TableSetupColumn("Name");
                        TableSetupColumn("Sampler");
                        TableSetupColumn("Image (default/WebP/Basisu/DDS)");
                        TableHeadersRow();
                        const auto idx_or_dash = [](const std::optional<uint32_t> &i) {
                            return i ? std::format("{}", *i) : std::string{"—"};
                        };
                        for (size_t i = 0; i < sa->Textures.size(); ++i) {
                            const auto &t = sa->Textures[i];
                            TableNextRow();
                            TableNextColumn();
                            Text("%zu", i);
                            TableNextColumn();
                            TextUnformatted(t.Name.c_str());
                            TableNextColumn();
                            TextUnformatted(idx_or_dash(t.SamplerIndex).c_str());
                            TableNextColumn();
                            Text("%s / %s / %s / %s", idx_or_dash(t.ImageIndex).c_str(), idx_or_dash(t.WebpImageIndex).c_str(), idx_or_dash(t.BasisuImageIndex).c_str(), idx_or_dash(t.DdsImageIndex).c_str());
                        }
                        EndTable();
                    }
                }
                if (!sa->Samplers.empty() && CollapsingHeader("Sampler registry")) {
                    if (BeginTable("Samplers", 5, MetadataTableFlags)) {
                        TableSetupColumn("#");
                        TableSetupColumn("Name");
                        TableSetupColumn("Mag");
                        TableSetupColumn("Min");
                        TableSetupColumn("Wrap S/T");
                        TableHeadersRow();
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
                        for (size_t i = 0; i < sa->Samplers.size(); ++i) {
                            const auto &s = sa->Samplers[i];
                            TableNextRow();
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
                        }
                        EndTable();
                    }
                }
                EndTabItem();
            }
        }
        EndTabBar();
    }
}

void RenderClipPickers(entt::registry &r) {
    static constexpr float ComboWidth = 200.f;
    // Names live on object entities, but ArmatureAnimation lives on the data entity.
    const auto display_name = [&]<typename Anim>(entt::entity entity) {
        if constexpr (std::is_same_v<Anim, ArmatureAnimation>) {
            for (const auto [obj_e, obj] : r.view<const ArmatureObject>().each()) {
                if (obj.Entity == entity) return GetName(r, obj_e);
            }
        }
        return GetName(r, entity);
    };
    const auto clip_picker = [&]<typename Anim>(std::string_view kind) {
        for (auto [entity, anim] : r.view<Anim>().each()) {
            if (anim.Clips.size() < 2) continue;
            const auto active_idx = anim.ActiveClipIndex;
            const auto label = std::format("{}: {}", kind, display_name.template operator()<Anim>(entity));
            PushID(label.c_str());
            SetNextItemWidth(ComboWidth);
            if (BeginCombo("##clip", NamedOr(anim.Clips[active_idx].Name, "Clip ", active_idx).c_str())) {
                for (uint32_t i = 0; i < anim.Clips.size(); ++i) {
                    if (Selectable(NamedOr(anim.Clips[i].Name, "Clip ", i).c_str(), active_idx == i) && active_idx != i) {
                        action::Emit(action::UpdateOf<&Anim::ActiveClipIndex>(entity, i));
                    }
                }
                EndCombo();
            }
            SameLine();
            TextUnformatted(label.c_str());
            PopID();
        }
    };
    clip_picker.template operator()<ArmatureAnimation>("Armature");
    clip_picker.template operator()<MorphWeightAnimation>("Morph");
    clip_picker.template operator()<NodeTransformAnimation>("Node");
}

static void RenderObjectTree(entt::registry &r, entt::entity viewport) {
    PushStyleVar(ImGuiStyleVar_ItemSpacing, {GetStyle().ItemSpacing.x, 0.f});

    const auto ToSelectionUserData = [](entt::entity e) -> ImGuiSelectionUserData { return ImGuiSelectionUserData(uint32_t(e)); };
    const auto FromSelectionUserData = [&](ImGuiSelectionUserData data) -> entt::entity {
        if (data == ImGuiSelectionUserData_Invalid) return entt::null;
        const auto e = entt::entity(uint32_t(data));
        return r.valid(e) ? e : entt::null;
    };

    const auto GetEntityTypeName = [&](entt::entity e) -> std::string_view {
        if (r.all_of<BoneIndex>(e)) return "Bone";
        if (r.all_of<ObjectKind>(e)) return ObjectTypeName(r.get<const ObjectKind>(e).Value);
        return ObjectTypeName(ObjectType::Empty);
    };
    std::vector<entt::entity> visible_entities;
    // Mutates `out` so begin and end batches fold into a single action.
    using Clear = action::selection::ApplyTreeSelection::ClearKind;
    const auto resolve_into = [&](action::selection::ApplyTreeSelection &out, std::span<const ImGuiSelectionRequest> requests, ImGuiSelectionUserData nav_item) {
        const auto add_target = [&](entt::entity e, bool selected) {
            if (e == entt::null) return;
            (selected ? out.ToSelect : out.ToDeselect).push_back(e);
        };
        for (const auto &request : requests) {
            if (request.Type == ImGuiSelectionRequestType_SetAll) {
                if (request.Selected) {
                    for (const auto e : visible_entities) add_target(e, true);
                } else {
                    const auto nav = FromSelectionUserData(nav_item);
                    out.Clear = nav != entt::null && r.all_of<BoneIndex>(nav) ? Clear::BonesOnly : Clear::All;
                }
                continue;
            }
            if (request.Type != ImGuiSelectionRequestType_SetRange) continue;

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
        if (const auto nav = FromSelectionUserData(nav_item); nav != entt::null) out.NavToActive = nav;
    };

    const int total_selected = r.storage<Selected>().size() + r.storage<BoneSelection>().size();
    auto *ms_begin = BeginMultiSelect(ImGuiMultiSelectFlags_None, total_selected, -1);
    std::vector<ImGuiSelectionRequest> begin_requests;
    begin_requests.reserve(ms_begin->Requests.Size);
    for (const auto &request : ms_begin->Requests) begin_requests.emplace_back(request);
    const auto begin_nav_item = ms_begin->NavIdItem;

    // Build the set of ancestors of any selected entity (for secondary highlight).
    std::unordered_set<entt::entity> ancestor_of_selected;
    const auto mark_ancestors = [&](entt::entity selected_entity) {
        const auto *n = r.try_get<SceneNode>(selected_entity);
        auto parent = n ? n->Parent : entt::null;
        while (parent != entt::null) {
            if (!ancestor_of_selected.insert(parent).second) break; // already inserted — parents already covered
            const auto *pn = r.try_get<SceneNode>(parent);
            parent = pn ? pn->Parent : entt::null;
        }
    };
    for (const auto e : r.view<Selected>()) mark_ancestors(e);
    for (const auto e : r.view<BoneSelection>()) mark_ancestors(e);

    const auto render_entity = [&](const auto &self, entt::entity e) -> void {
        const auto *node = r.try_get<SceneNode>(e);
        const bool has_children = node && node->FirstChild != entt::null;
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

    bool has_root = false;
    for (const auto [entity, _] : r.view<const Name>().each()) {
        if (const auto *node = r.try_get<SceneNode>(entity); node && node->Parent != entt::null) continue;
        has_root = true;
        render_entity(render_entity, entity);
    }
    if (!has_root) TextDisabled("No objects");

    // Both BeginMultiSelect and EndMultiSelect carry requests that resolve to the same selection update.
    action::selection::ApplyTreeSelection tree_selection;
    resolve_into(tree_selection, begin_requests, begin_nav_item);
    auto *ms_end = EndMultiSelect();
    resolve_into(tree_selection, {ms_end->Requests.Data, size_t(ms_end->Requests.Size)}, ms_end->NavIdItem);
    if (!tree_selection.ToSelect.empty() || !tree_selection.ToDeselect.empty() ||
        tree_selection.Clear != Clear::None || tree_selection.NavToActive != entt::null) {
        action::Emit(std::move(tree_selection));
    }

    PopStyleVar();
}
