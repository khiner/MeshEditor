#pragma once

#include "CameraTypes.h"
#include "Variant.h"
#include "action/Core.h"
#include "gpu/PunctualLight.h"
#include "mesh/MeshData.h"
#include "mesh/PrimitiveType.h"
#include "object/ObjectCreateInfo.h"
#include "render/MaterialComponents.h"

#include <filesystem>

namespace action::object {
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
// Edit-mode element sharpness. Each resolves the element selection of every edit-mode mesh at apply time.
struct SetSelectedFacesSmooth {
    bool Smooth;
};
struct SetSelectedEdgesSharp {
    bool Sharp;
};
// Applies to every edge touching a selected vertex.
struct SetSelectedVertexEdgesSharp {
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

// Update the field at byte `Offset` of the active primitive's current shape alternative.
template<typename Field>
struct UpdatePrimitiveField {
    Scope Scope{Scope::Active};
    uint16_t Offset;
    Field Value, Min, Max;
};

using Actions = std::variant<
    Delete, Duplicate, DuplicateLinked, ToggleHidden, SetSelectedVisible, SetSelectedSmoothShading, ShadeSelectedSmoothByAngle,
    SetSelectedFacesSmooth, SetSelectedEdgesSharp, SetSelectedVertexEdgesSharp,
    ParentToActive, ClearParent,
    AddEmpty, AddArmature, AddCamera, AddLight, AddMeshPrimitive, ImportMesh, SetPbrMeshFeaturesMask,
    UpdatePrimitiveField<float>, UpdatePrimitiveField<vec2>, UpdatePrimitiveField<vec3>, UpdatePrimitiveField<uint32_t>>;

using Action = MergedVariantT<
    Actions,
    Replace<PunctualLight>,
    Replace<MaterialDirty>, Replace<MeshMaterialAssignment>, Replace<MeshMaterialSlotSelection>,
    Update<std::optional<uint32_t>>>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::object
