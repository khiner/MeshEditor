#pragma once

#include "state/Slot.h"

struct CookCount;
namespace procedure {
struct ProcedureChanges;
struct NodeTheme;
struct Source;
struct Output;
struct Join;
struct Boolean;
struct Position;
struct Normal;
struct Index;
struct NamedAttribute;
struct VectorMath;
struct FloatMath;
struct Compare;
struct BooleanMath;
struct StoreNamedAttribute;
struct FloatValue;
struct IntegerValue;
struct BooleanValue;
struct VectorValue;
struct RotationValue;
struct SetPosition;
struct TransformElements;
struct TransformGeometry;
struct MeshToPoints;
struct InstanceOnPoints;
struct RealizeInstances;
struct ObjectInfo;
struct GeometryProcedure;
struct ProcedureInstances;
struct ProcedureOwnedMesh;
struct NodeLayout;
struct InlineSelection;
struct PublishSelection;
struct NodeSelected;
struct NodeError;
struct CookPlan;
} // namespace procedure

using ProcedureSchema = state::Schema<state::ProcedureBase, state::ProcedureCount, procedure::ProcedureChanges, CookCount, procedure::NodeTheme, procedure::Source, procedure::Output, procedure::Join, procedure::Boolean, procedure::Position, procedure::Normal, procedure::Index, procedure::NamedAttribute, procedure::VectorMath, procedure::FloatMath, procedure::Compare, procedure::BooleanMath, procedure::StoreNamedAttribute, procedure::FloatValue, procedure::IntegerValue, procedure::BooleanValue, procedure::VectorValue, procedure::RotationValue, procedure::SetPosition, procedure::TransformElements, procedure::TransformGeometry, procedure::MeshToPoints, procedure::InstanceOnPoints, procedure::RealizeInstances, procedure::ObjectInfo, procedure::GeometryProcedure, procedure::ProcedureInstances, procedure::ProcedureOwnedMesh, procedure::NodeLayout, procedure::InlineSelection, procedure::PublishSelection, procedure::NodeSelected, procedure::NodeError, procedure::CookPlan>;
template<typename T>
    requires ProcedureSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = ProcedureSchema::Of<T>();
