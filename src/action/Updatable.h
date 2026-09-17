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
struct TriggerNodes;
struct ViewportTheme;

namespace action {
template<typename... Cs> struct TypeList {};

// The components Update can address, visited at apply time and checked at each emit site.
using UpdatableComponents = TypeList<
    Transform, PosedLocal, BoneDelta, TransformGizmoState, OrbitToActive, ShadeSmoothAngle,
    ViewportDisplay, ViewportTheme, MaterialPreviewLighting, RenderedLighting,
    PunctualLight, ImageLight, Perspective, Orthographic, MaterialVariants, PrimitiveShape,
    Animations,
    PhysicsSimulationSettings, PhysicsMaterial, PhysicsMotion, PhysicsVelocity, ColliderPolicy, CollisionFilter, PhysicsJoint, ColliderMaterial, TriggerNodes,
    ModalGain, ModalTuning, ModalSoundControls, Striker, SurfaceSoundControls, AudioOutputMix, AudioOutputConfig>;

template<typename C, typename List> inline constexpr bool InTypeList = false;
template<typename C, typename... Cs> inline constexpr bool InTypeList<C, TypeList<Cs...>> = (std::same_as<C, Cs> || ...);
template<typename C>
concept Updatable = InTypeList<C, UpdatableComponents>;

// Invokes `f.template operator()<C>()` for the updatable component in `slot`, or nothing for a key outside the list.
template<typename F> void ForUpdatable(state::TypeId slot, F &&f) {
    [&]<typename... Cs>(TypeList<Cs...>) {
        (void)((state::Type<Cs>() == slot && (f.template operator()<Cs>(), true)) || ...);
    }(UpdatableComponents{});
}
} // namespace action
