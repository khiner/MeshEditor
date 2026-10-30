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
// The components Update can address, visited at apply time and checked at each emit site.
using UpdatableComponents = TypeList<
    Transform, PosedLocal, BoneDelta, TransformGizmoState, OrbitToActive, ShadeSmoothAngle,
    ViewportDisplay, AnisotropicFilter, ViewportTheme, InterfaceTheme, TimelineTheme, TimelineNavigation, MaterialPreviewLighting, RenderedLighting,
    PunctualLight, ImageLight, Perspective, Orthographic, MaterialVariants,
    Animations,
    PhysicsSimulationSettings, PhysicsMaterial, PhysicsMotion, PhysicsVelocity, ColliderPolicy, CollisionFilter, PhysicsJoint, ColliderMaterial, TriggerNodes,
    ModalGain, ModalTuning, ModalSoundControls, Striker, SurfaceSoundControls, AudioOutputMix,
    // Geometry procedure node layouts and parameters.
    procedure::NodeLayout, procedure::NodeTheme,
    mesh::Delete, mesh::Merge, mesh::Extrude, mesh::Duplicate, mesh::Split, mesh::Dissolve, mesh::Subdivide, mesh::Triangulate, mesh::TrisToQuads,
    mesh::Poke, mesh::FlipNormals, mesh::EdgeSplit, mesh::Inset, mesh::Solidify, mesh::Wireframe, mesh::Bevel, mesh::SmoothVertices, mesh::ShrinkFatten,
    mesh::Randomize, procedure::Spin, mesh::ExtrudeRepeat, procedure::Bisect, mesh::Symmetrize, mesh::FillHoles, mesh::ConvexHull, mesh::Decimate, mesh::RecalculateNormals,
    mesh::Rip, mesh::SpaceEvenly, mesh::RelaxEdgeLoops, mesh::CurveBetweenSelected, mesh::Circularize, mesh::MakePlanarFaces, mesh::SnapSymmetry,
    mesh::SplitNonplanarFaces, mesh::SplitConcaveFaces, mesh::ConnectVertices, mesh::Fill, mesh::BridgeEdgeLoops, mesh::GridFill, mesh::EdgeRotate, mesh::BeautifyFaces, mesh::Unsubdivide, mesh::LoopCut,
    mesh::RotateUVs, mesh::ReverseUVs, mesh::RotateColors, mesh::ReverseColors, procedure::MarkSharp, procedure::Knife, procedure::ShadeSmooth, procedure::ShadeSmoothByAngle,
    procedure::ToSphere, procedure::PushPull, procedure::Shear, procedure::Warp, procedure::Bend, procedure::VertexSlide, procedure::EdgeSlide, procedure::Flatten,
    procedure::TransformElements, procedure::Boolean,
    procedure::Position, procedure::Normal, procedure::Index, procedure::NamedAttribute, procedure::VectorMath, procedure::FloatMath, procedure::Compare,
    procedure::BooleanMath, procedure::StoreNamedAttribute, procedure::StoreSelection, procedure::SetPosition, procedure::InstanceOnPoints, procedure::ObjectInfo,
    procedure::FloatValue, procedure::IntegerValue, procedure::BooleanValue, procedure::VectorValue, procedure::RotationValue, procedure::TransformGeometry,
    primitive::Plane, primitive::Circle, primitive::Cube, primitive::IcoSphere, primitive::UVSphere, primitive::Torus, primitive::Cylinder, primitive::Cone>;

template<typename C>
concept Updatable = InTypeList<C, UpdatableComponents>;

// Invokes `f.template operator()<C>()` for the updatable component with `key`, or nothing for a key outside the list.
void ForUpdatable(state::TypeKey key, auto &&f) {
    [&]<typename... Cs>(TypeList<Cs...>) {
        (void)((state::Key<Cs>() == key && (f.template operator()<Cs>(), true)) || ...);
    }(UpdatableComponents{});
}
} // namespace action
