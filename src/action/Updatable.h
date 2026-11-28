#pragma once

#include "TypeList.h"
#include "action/Core.h"
#include "gpu/Transform.h"
#include "graph/GraphSchema.h"
#include "mesh/PrimitiveType.h"

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
struct SceneLight;
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
    SceneLight, ImageLight, Perspective, Orthographic, MaterialVariants,
    Animations,
    PhysicsSimulationSettings, PhysicsMaterial, PhysicsMotion, PhysicsVelocity, ColliderPolicy, CollisionFilter, PhysicsJoint, ColliderMaterial, TriggerNodes,
    ModalGain, ModalTuning, ModalSoundControls, Striker, SurfaceSoundControls, AudioOutputMix,
    graph::NodeTheme,
    mesh::Delete, mesh::Merge, graph::Extrude, mesh::Dissolve, mesh::Subdivide, graph::SubdivisionSurface, graph::SmoothSurface, mesh::SplitFaces,
    mesh::Poke, mesh::EdgeSplit, mesh::Inset, mesh::Solidify, mesh::Wireframe, mesh::Bevel, mesh::SmoothVertices, mesh::ShrinkFatten,
    mesh::Randomize, graph::Bisect, graph::Project, mesh::Simplify,
    mesh::SpaceEvenly, mesh::RelaxEdgeLoops, mesh::CurveBetweenSelected, mesh::Circularize, mesh::MakePlanarFaces, mesh::SnapSymmetry,
    mesh::Fill, mesh::GridFill, mesh::Unsubdivide,
    mesh::CycleCornerAttribute, graph::MarkSharp, graph::Knife, graph::Shade,
    graph::ToSphere, graph::PushPull, graph::Shear, graph::Warp, graph::Bend, graph::VertexSlide, graph::EdgeSlide, graph::Flatten,
    graph::TransformElements, graph::Boolean, graph::Minkowski, graph::LevelSet,
    graph::StoreNamedAttribute, graph::Capture, graph::StoreSelection, graph::Bake, graph::SetPosition, graph::InstanceOnPoints, graph::ObjectInfo,
    primitive::Plane, primitive::Circle, primitive::Cube, primitive::IcoSphere, primitive::UVSphere, primitive::QuadSphere, primitive::Torus, primitive::Cylinder>;

template<typename C>
concept Updatable = InTypeList<C, UpdatableComponents>;
} // namespace action
