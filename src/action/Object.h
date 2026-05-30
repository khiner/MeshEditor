#pragma once

#include "CameraTypes.h"
#include "Variant.h"
#include "action/Core.h"
#include "gpu/PunctualLight.h"
#include "mesh/MeshData.h"
#include "mesh/PrimitiveType.h"
#include "object/ObjectCreateInfo.h"
#include "render/MaterialComponents.h"

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
    std::string Path; // not fs::path so the action serializes natively
    std::unique_ptr<MeshInstanceCreateInfo> Info;
};
// `Mask=0` removes the component. Targets the active mesh entity.
struct SetPbrMeshFeaturesMask {
    uint32_t Mask;
};

using Actions = std::variant<
    Delete, Duplicate, DuplicateLinked, ToggleHidden, SetSelectedVisible, SetSelectedSmoothShading,
    ParentToActive, ClearParent,
    AddEmpty, AddArmature, AddCamera, AddLight, AddMeshPrimitive, ImportMesh, SetPbrMeshFeaturesMask>;

using Action = MergedVariantT<
    Actions,
    Replace<PrimitiveShape>,
    Replace<PunctualLight>, ReplaceActive<PunctualLight>,
    Replace<MaterialDirty>, Replace<MeshMaterialAssignment>, Replace<MeshMaterialSlotSelection>,
    Update<std::optional<uint32_t>>>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::object
