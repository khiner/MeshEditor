#pragma once

#include "state/Slot.h"

struct CookCount;
namespace procedure {
struct ProcedureChanges;
struct NodeTheme;
struct Source;
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
struct ToSphere;
struct PushPull;
struct Shear;
struct Warp;
struct Spin;
struct Bisect;
struct Bend;
struct VertexSlide;
struct EdgeSlide;
struct Flatten;
struct MarkSharp;
struct Knife;
struct ShadeSmooth;
struct ShadeSmoothByAngle;
struct StoreSelection;
struct MeshToPoints;
struct InstanceOnPoints;
struct RealizeInstances;
struct ObjectInfo;
struct ProcedureNode;
struct NodeInputs;
struct GeometryProcedure;
struct NetworkActive;
struct NodeBypassed;
struct NetworkLayout;
struct NetworkSubnet;
struct Subnet;
struct SubnetInterface;
struct NodeContainer;
struct ProcedureInstances;
struct ProcedureOwnedMesh;
struct NodeLayout;
struct InlineSelection;
struct PublishSelection;
struct NodeError;
struct SourceState;
struct NodeResult;
struct NodeExtrusion;
struct NodeHandle;
struct OperationHandle;
struct BooleanPlacement;
struct StagedNode;
struct CookPass;
struct CookPlan;
} // namespace procedure

using ProcedureSchema = state::Schema<state::ProcedureBase, state::ProcedureCount, procedure::ProcedureChanges, procedure::CookPass, CookCount, procedure::NodeTheme, procedure::Source, procedure::Join, procedure::Boolean, procedure::Position, procedure::Normal, procedure::Index, procedure::NamedAttribute, procedure::VectorMath, procedure::FloatMath, procedure::Compare, procedure::BooleanMath, procedure::StoreNamedAttribute, procedure::FloatValue, procedure::IntegerValue, procedure::BooleanValue, procedure::VectorValue, procedure::RotationValue, procedure::SetPosition, procedure::TransformElements, procedure::TransformGeometry, procedure::ToSphere, procedure::PushPull, procedure::Shear, procedure::Warp, procedure::Bend, procedure::VertexSlide, procedure::EdgeSlide, procedure::Flatten, procedure::MarkSharp, procedure::Knife, procedure::ShadeSmooth, procedure::ShadeSmoothByAngle, procedure::StoreSelection, procedure::MeshToPoints, procedure::InstanceOnPoints, procedure::RealizeInstances, procedure::ObjectInfo, procedure::ProcedureNode, procedure::NodeInputs, procedure::GeometryProcedure, procedure::NetworkActive, procedure::NodeBypassed, procedure::NetworkLayout, procedure::NetworkSubnet, procedure::Subnet, procedure::SubnetInterface, procedure::NodeContainer, procedure::ProcedureInstances, procedure::ProcedureOwnedMesh, procedure::NodeLayout, procedure::InlineSelection, procedure::PublishSelection, procedure::NodeError, procedure::SourceState, procedure::NodeResult, procedure::NodeExtrusion, procedure::NodeHandle, procedure::OperationHandle, procedure::BooleanPlacement, procedure::StagedNode, procedure::CookPlan, procedure::Spin, procedure::Bisect>;
template<typename T>
    requires ProcedureSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = ProcedureSchema::Of<T>();
