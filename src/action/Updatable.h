#pragma once

#include "action/Core.h"
#include "mesh/PrimitiveType.h"

struct AudioOutputConfig;
struct AudioOutputMix;
struct ColliderMaterial;
struct ColliderPolicy;
struct CollisionFilter;
struct MaterialVariants;
struct ModalGain;
struct ModalSoundControls;
struct ModalTuning;
struct Animations;
struct ImageLight;
struct BoneDelta;
struct OrbitToActive;
struct Orthographic;
struct Perspective;
struct PhysicsJoint;
struct PhysicsMaterial;
struct PhysicsMotion;
struct PhysicsSimulationSettings;
struct PhysicsVelocity;
struct PosedLocal;
struct PunctualLight;
struct ShadeSmoothAngle;
struct Striker;
struct SurfaceSoundControls;
struct TransformGizmoState;
struct TimelineNavigation;
struct TriggerNodes;
struct ViewportTheme;
struct InterfaceTheme;
struct TimelineTheme;

namespace action::mesh {
struct Delete;
struct Merge;
struct Extrude;
struct Duplicate;
struct Split;
struct Dissolve;
struct Subdivide;
struct Triangulate;
struct TrisToQuads;
struct Poke;
struct FlipNormals;
struct EdgeSplit;
struct Inset;
struct Solidify;
struct Wireframe;
struct Bevel;
struct SmoothVertices;
struct ShrinkFatten;
struct Randomize;
struct Spin;
struct ExtrudeRepeat;
struct Bisect;
struct Symmetrize;
struct FillHoles;
struct ConvexHull;
struct Decimate;
struct RecalculateNormals;
struct Rip;
} // namespace action::mesh
namespace procedure {
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
struct SetPosition;
struct TransformElements;
struct InstanceOnPoints;
struct ObjectInfo;
struct FloatValue;
struct IntegerValue;
struct BooleanValue;
struct VectorValue;
struct RotationValue;
struct TransformGeometry;
struct NodeLayout;
struct NodeTheme;
} // namespace procedure

namespace action {
template<typename... Cs> struct TypeList {};

// The components Update can address, visited at apply time and checked at each emit site.
using UpdatableComponents = TypeList<
    Transform, PosedLocal, BoneDelta, TransformGizmoState, OrbitToActive, ShadeSmoothAngle,
    ViewportDisplay, ViewportTheme, InterfaceTheme, TimelineTheme, TimelineNavigation, MaterialPreviewLighting, RenderedLighting,
    PunctualLight, ImageLight, Perspective, Orthographic, MaterialVariants, PrimitiveShape,
    Animations,
    PhysicsSimulationSettings, PhysicsMaterial, PhysicsMotion, PhysicsVelocity, ColliderPolicy, CollisionFilter, PhysicsJoint, ColliderMaterial, TriggerNodes,
    ModalGain, ModalTuning, ModalSoundControls, Striker, SurfaceSoundControls, AudioOutputMix, AudioOutputConfig,
    // Geometry procedure node layouts and parameters.
    procedure::NodeLayout, procedure::NodeTheme,
    mesh::Delete, mesh::Merge, mesh::Extrude, mesh::Duplicate, mesh::Split, mesh::Dissolve, mesh::Subdivide, mesh::Triangulate, mesh::TrisToQuads,
    mesh::Poke, mesh::FlipNormals, mesh::EdgeSplit, mesh::Inset, mesh::Solidify, mesh::Wireframe, mesh::Bevel, mesh::SmoothVertices, mesh::ShrinkFatten,
    mesh::Randomize, mesh::Spin, mesh::ExtrudeRepeat, mesh::Bisect, mesh::Symmetrize, mesh::FillHoles, mesh::ConvexHull, mesh::Decimate, mesh::RecalculateNormals,
    mesh::Rip, procedure::TransformElements, procedure::Boolean,
    procedure::Position, procedure::Normal, procedure::Index, procedure::NamedAttribute, procedure::VectorMath, procedure::FloatMath, procedure::Compare,
    procedure::BooleanMath, procedure::StoreNamedAttribute, procedure::SetPosition, procedure::InstanceOnPoints, procedure::ObjectInfo,
    procedure::FloatValue, procedure::IntegerValue, procedure::BooleanValue, procedure::VectorValue, procedure::RotationValue, procedure::TransformGeometry,
    primitive::Plane, primitive::Circle, primitive::Cuboid, primitive::IcoSphere, primitive::UVSphere, primitive::Torus, primitive::Cylinder, primitive::Cone>;

template<typename C, typename List> inline constexpr bool InTypeList = false;
template<typename C, typename... Cs> inline constexpr bool InTypeList<C, TypeList<Cs...>> = (std::same_as<C, Cs> || ...);
template<typename C>
concept Updatable = InTypeList<C, UpdatableComponents>;

// Invokes `f.template operator()<C>()` for the updatable component in `slot`, or nothing for a key outside the list.
void ForUpdatable(state::TypeId slot, auto &&f) {
    [&]<typename... Cs>(TypeList<Cs...>) {
        (void)((state::Type<Cs>() == slot && (f.template operator()<Cs>(), true)) || ...);
    }(UpdatableComponents{});
}
} // namespace action
