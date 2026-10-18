#pragma once

#include "state/Slot.h"

#include <variant>

struct MeshChanges;
struct RenderInstance;
struct ModelsBuffer;
struct VertexStoreId;
struct MeshHandle;
struct EvaluatedMesh;
struct ResyncState;
struct AuthoredCornerNormals;
struct ObjectExtrasTag;
struct MeshPipelines;
struct MeshStore;
struct MeshEntities;
struct TetBuffers;
namespace primitive {
struct Plane;
struct Circle;
struct Cuboid;
struct IcoSphere;
struct UVSphere;
struct Torus;
struct Cylinder;
struct Cone;
} // namespace primitive

using PrimitiveShape = std::variant<primitive::Plane, primitive::Circle, primitive::Cuboid, primitive::IcoSphere, primitive::UVSphere, primitive::Torus, primitive::Cylinder, primitive::Cone>;

using MeshSchema = state::Schema<state::MeshBase, state::MeshCount, MeshChanges, RenderInstance, ModelsBuffer, VertexStoreId, MeshHandle, EvaluatedMesh, ResyncState, AuthoredCornerNormals, ObjectExtrasTag, MeshPipelines, MeshStore, MeshEntities, primitive::Plane, primitive::Circle, primitive::Cuboid, primitive::IcoSphere, primitive::UVSphere, primitive::Torus, primitive::Cylinder, primitive::Cone, PrimitiveShape, TetBuffers>;
template<typename T>
    requires MeshSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = MeshSchema::Of<T>();
