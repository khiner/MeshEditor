#pragma once

#include "Camera.h"
#include "ObjectOps.h"
#include "mesh/MeshData.h"
#include "mesh/PrimitiveType.h"

#include <cstdint>
#include <filesystem>
#include <memory>

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
    std::filesystem::path Path;
    std::unique_ptr<MeshInstanceCreateInfo> Info;
};
struct ReplaceMesh {
    std::unique_ptr<MeshData> Data;
};
// `Mask=0` removes the component. Targets the active mesh entity.
struct SetPbrMeshFeaturesMask {
    uint32_t Mask;
};

using Actions = std::variant<
    Delete, Duplicate, DuplicateLinked, ToggleHidden, SetSelectedVisible, SetSelectedSmoothShading,
    ParentToActive, ClearParent,
    AddEmpty, AddArmature, AddCamera, AddLight, AddMeshPrimitive, ImportMesh, ReplaceMesh, SetPbrMeshFeaturesMask>;
} // namespace action::object
