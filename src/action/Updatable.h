#pragma once

#include "TypeList.h"
#include "action/Core.h"
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
struct SpaceEvenly;
struct RelaxEdgeLoops;
struct CurveBetweenSelected;
struct Circularize;
struct MakePlanarFaces;
struct SnapSymmetry;
struct SplitNonplanarFaces;
struct SplitConcaveFaces;
struct ConnectVertices;
struct Fill;
struct BridgeEdgeLoops;
struct GridFill;
struct EdgeRotate;
struct BeautifyFaces;
struct Unsubdivide;
struct LoopCut;
struct RotateUVs;
struct ReverseUVs;
struct RotateColors;
struct ReverseColors;
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
struct ToSphere;
struct PushPull;
struct Shear;
struct Warp;
struct Bend;
struct VertexSlide;
struct EdgeSlide;
struct Flatten;
struct MarkSharp;
struct Knife;
struct ShadeSmooth;
struct ShadeSmoothByAngle;
struct StoreSelection;
struct NodeLayout;
struct NodeTheme;
} // namespace procedure

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
    mesh::Randomize, mesh::Spin, mesh::ExtrudeRepeat, mesh::Bisect, mesh::Symmetrize, mesh::FillHoles, mesh::ConvexHull, mesh::Decimate, mesh::RecalculateNormals,
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
