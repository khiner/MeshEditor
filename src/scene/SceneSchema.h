#pragma once

#include "state/Slot.h"

struct HierarchyInstance;
struct HierarchyGeometry;
struct HierarchyMember;
struct HierarchyDefinition;
struct HierarchyChanges;
struct Orthographic;
struct Perspective;
struct Path;
struct EntityNameCounts;
struct Name;
struct Scene;
struct SceneMembership;
struct Selected;
struct Active;
struct MeshActiveElement;
struct SubElementOf;
struct ScaleLocked;
struct ObjectKind;
struct SceneChanges;
struct SceneParent;
struct SceneChildren;
struct WorldTransform;
struct PosedLocal;

using SceneSchema = state::Schema<state::SceneBase, state::SceneCount, Orthographic, Perspective, Path, EntityNameCounts, Name, Scene, SceneMembership, Selected, Active, MeshActiveElement, SubElementOf, ScaleLocked, ObjectKind, SceneChanges, SceneParent, SceneChildren, WorldTransform, PosedLocal, HierarchyInstance, HierarchyMember, HierarchyDefinition, HierarchyChanges, HierarchyGeometry>;
template<typename T>
    requires SceneSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = SceneSchema::Of<T>();
