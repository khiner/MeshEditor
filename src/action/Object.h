#pragma once

#include "gpu/Element.h"
#include "gpu/PBRMaterial.h"

#include "CameraTypes.h"
#include "action/Core.h"
#include "gpu/PunctualLight.h"
#include "mesh/MeshData.h"
#include "mesh/PrimitiveType.h"
#include "object/ObjectCreateInfo.h"
#include "viewport/ViewportInteractionState.h"

#include <filesystem>

namespace action::object {
struct SetLightType {
    PunctualLightType Type;
    Scope Scope{Scope::Active};
};
struct SetSpotCone {
    float OuterAngle, Blend;
    Scope Scope{Scope::Active};
};
struct SetCameraLens {
    Camera Value;
    Scope Scope{Scope::Active};
};
struct Delete {};
struct Duplicate {};
struct DuplicateLinked {};
struct ToggleHidden {};
struct SetSelectedVisible {
    bool Visible;
};
struct SetSelectedSmoothShading {
    bool Smooth;
};
// Smooth all faces and mark edges sharp where the dihedral angle exceeds Angle (radians).
struct ShadeSelectedSmoothByAngle {
    float Angle;
};
// Edit-mode element sharpness, resolving the element selection of every edit-mode mesh at apply time.
// Vertex selections apply to every edge touching a selected vertex.
struct SetSelectedSharp {
    Element Element;
    bool Sharp;
};
struct ParentToActive {};
struct ClearParent {};

struct AddEmpty {
    std::unique_ptr<ObjectCreateInfo> Info;
};
struct AddArmature {
    std::unique_ptr<ObjectCreateInfo> Info;
};
struct AddCamera {
    std::unique_ptr<ObjectCreateInfo> Info;
    std::optional<Camera> Props;
};
struct AddLight {
    std::unique_ptr<ObjectCreateInfo> Info;
};
struct AddMeshPrimitive {
    PrimitiveShape Shape;
    std::unique_ptr<MeshInstanceCreateInfo> Info;
};
struct ImportMesh {
    std::filesystem::path Path;
    std::unique_ptr<MeshInstanceCreateInfo> Info;
};
// `Mask=0` removes the component. Targets the mesh entity.
struct SetPbrMeshFeaturesMask {
    uint32_t Mask;
    Scope Scope{Scope::Active};
};
struct UpdateMaterial {
    uint32_t Index;
    std::unique_ptr<PBRMaterial> Value;
    std::optional<uint32_t> Features;
    Scope Scope{Scope::Active};
};
// Choose the material slot shown by the material editor. Targets the mesh entity.
struct SetMaterialSlotSelection {
    uint32_t PrimitiveIndex;
    Scope Scope{Scope::Active};
};
// Assign a material to a primitive slot. Targets the mesh entity.
struct SetMaterialAssignment {
    uint32_t PrimitiveIndex, MaterialIndex;
    Scope Scope{Scope::Active};
};

using Action = std::variant<
    Delete, Duplicate, DuplicateLinked, ToggleHidden, SetSelectedVisible, SetSelectedSmoothShading, ShadeSelectedSmoothByAngle,
    SetSelectedSharp,
    ParentToActive, ClearParent,
    AddEmpty, AddArmature, AddCamera, AddLight, AddMeshPrimitive, ImportMesh,
    SetPbrMeshFeaturesMask, UpdateMaterial, SetMaterialSlotSelection, SetMaterialAssignment,
    SetLightType, SetSpotCone, SetCameraLens>;

void Apply(state::Scene &, state::Entity viewport, const Action &);
} // namespace action::object
