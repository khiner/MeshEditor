#pragma once

#include "TypeList.h"
#include "action/Core.h"
#include "mesh/PrimitiveType.h"
#include "procedure/ProcedureSchema.h"

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

namespace action {
using UpdatableComponents = TypeList<
    Transform, PosedLocal, BoneDelta, TransformGizmoState, OrbitToActive, ShadeSmoothAngle,
    ViewportDisplay, AnisotropicFilter, ViewportTheme, InterfaceTheme, TimelineTheme, TimelineNavigation, MaterialPreviewLighting, RenderedLighting,
    PunctualLight, ImageLight, Perspective, Orthographic, MaterialVariants,
    Animations,
    PhysicsSimulationSettings, PhysicsMaterial, PhysicsMotion, PhysicsVelocity, ColliderPolicy, CollisionFilter, PhysicsJoint, ColliderMaterial, TriggerNodes,
    ModalGain, ModalTuning, ModalSoundControls, Striker, SurfaceSoundControls, AudioOutputMix,
    procedure::NodeLayout, procedure::NodeTheme,
    mesh::Delete, mesh::Merge, procedure::Extrude, mesh::Dissolve, mesh::Subdivide, procedure::SubdivisionSurface, procedure::SmoothSurface, mesh::SplitFaces,
    mesh::Poke, mesh::EdgeSplit, mesh::Inset, mesh::Solidify, mesh::Wireframe, mesh::Bevel, mesh::SmoothVertices, mesh::ShrinkFatten,
    mesh::Randomize, procedure::Bisect, procedure::Project, mesh::Simplify,
    mesh::SpaceEvenly, mesh::RelaxEdgeLoops, mesh::CurveBetweenSelected, mesh::Circularize, mesh::MakePlanarFaces, mesh::SnapSymmetry,
    mesh::Fill, mesh::GridFill, mesh::Unsubdivide,
    mesh::CycleCornerAttribute, procedure::MarkSharp, procedure::Knife, procedure::Shade,
    procedure::ToSphere, procedure::PushPull, procedure::Shear, procedure::Warp, procedure::Bend, procedure::VertexSlide, procedure::EdgeSlide, procedure::Flatten,
    procedure::TransformElements, procedure::Boolean, procedure::Minkowski, procedure::LevelSet,
    procedure::Normal, procedure::Curvature, procedure::Boundary, procedure::PlaneSide, procedure::Adjacent, procedure::Index, procedure::NamedAttribute, procedure::VectorMath, procedure::FloatMath, procedure::Compare,
    procedure::BooleanMath, procedure::StoreNamedAttribute, procedure::StoreSelection, procedure::SetPosition, procedure::InstanceOnPoints, procedure::ObjectInfo,
    procedure::FloatValue, procedure::IntegerValue, procedure::BooleanValue, procedure::VectorValue, procedure::RotationValue,
    primitive::Plane, primitive::Circle, primitive::Cube, primitive::IcoSphere, primitive::UVSphere, primitive::QuadSphere, primitive::Torus, primitive::Cylinder>;

template<typename C>
concept Updatable = InTypeList<C, UpdatableComponents>;

// Invokes `f.template operator()<C>()` for the updatable component with `key`, or nothing for a key outside the list.
void ForUpdatable(state::TypeKey key, auto &&f) {
    [&]<typename... Cs>(TypeList<Cs...>) {
        (void)((state::Key<Cs>() == key && (f.template operator()<Cs>(), true)) || ...);
    }(UpdatableComponents{});
}
} // namespace action
