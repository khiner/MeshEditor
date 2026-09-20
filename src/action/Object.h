#pragma once

#include "numeric/vec2.h"

#include "gpu/Element.h"
#include "gpu/PBRMaterial.h"

#include "CameraTypes.h"
#include "action/Core.h"
#include "mesh/MeshData.h"
#include "mesh/PrimitiveType.h"
#include "object/ObjectCreateInfo.h"
#include "render/LightComponents.h"
#include "viewport/ViewportInteractionState.h"

#include <filesystem>

namespace action::object {
struct SetLightType {
    PunctualLightType Type;
    Target Target{OnActive{}};
};
struct SetSpotCone {
    float OuterAngle, Blend;
    Target Target{OnActive{}};
};
// Switches a camera between perspective and orthographic, keeping the view size at the camera's distance from the origin.
struct SetProjection {
    bool Orthographic;
    Target Target{OnActive{}};
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
    std::optional<CameraLens> Props;
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
    Target Target{OnActive{}};
};
// Writes `Value` to the field at byte `Offset` of material `Index` in the material buffer.
template<typename T>
struct UpdateMaterial {
    uint32_t Index;
    uint16_t Offset;
    T Value;
};
template<typename T> inline constexpr bool IsUpdateMaterial = false;
template<typename T> inline constexpr bool IsUpdateMaterial<UpdateMaterial<T>> = true;
// Choose the material slot shown by the material editor. Targets the mesh entity.
struct SetMaterialSlotSelection {
    uint32_t PrimitiveIndex;
    Target Target{OnActive{}};
};
// Assign a material to a primitive slot. Targets the mesh entity.
struct SetMaterialAssignment {
    uint32_t PrimitiveIndex, MaterialIndex;
    Target Target{OnActive{}};
};

using Action = std::variant<
    Delete, Duplicate, DuplicateLinked, ToggleHidden, SetSelectedVisible, SetSelectedSmoothShading, ShadeSelectedSmoothByAngle,
    SetSelectedSharp,
    ParentToActive, ClearParent,
    AddEmpty, AddArmature, AddCamera, AddLight, AddMeshPrimitive, ImportMesh,
    SetPbrMeshFeaturesMask, SetMaterialSlotSelection, SetMaterialAssignment,
    UpdateMaterial<float>, UpdateMaterial<vec2>, UpdateMaterial<vec3>, UpdateMaterial<vec4>, UpdateMaterial<uint32_t>, UpdateMaterial<MaterialAlphaMode>,
    SetLightType, SetSpotCone, SetProjection>;

void Apply(state::Scene &, state::Entity viewport, const Action &);
} // namespace action::object
