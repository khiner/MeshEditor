#pragma once

#include "Variant.h"
#include "action/Audio.h"
#include "action/Bone.h"
#include "action/Core.h"
#include "action/Io.h"
#include "action/Object.h"
#include "action/Physics.h"
#include "action/Selection.h"
#include "action/Timeline.h"
#include "action/View.h"
#include "viewport/ViewportInteractionState.h"

#include "action/SerializeGlm.h"

namespace action {
using Action = MergedVariantT<
    Core,
    selection::Action, object::Action, view::Action,
    physics::Action, audio::Action, bone::Action, timeline::Action, io::Action>;

// Actions are logged for replay unless they only produce an external artifact.
// E.g. replaying a save would clobber a file.
template<typename T> inline constexpr bool Recordable = true;
template<> inline constexpr bool Recordable<io::SaveGltf> = false;
template<> inline constexpr bool Recordable<io::SaveState> = false;
// Latch state is live-only: the recorded DragGizmo already encodes the resolved transform.
template<> inline constexpr bool Recordable<view::LatchScreenTransform> = false;
template<> inline constexpr bool Recordable<view::ClearScreenTransformLatch> = false;
// View-camera navigation is not recorded. For selection replay correctness, selection actions hold the ViewProj.
// The snapshot still stores the ViewCamera.
template<> inline constexpr bool Recordable<view::OrbitViewCamera> = false;
template<> inline constexpr bool Recordable<view::ZoomViewCamera> = false;
template<> inline constexpr bool Recordable<view::ResetViewCamera> = false;
template<> inline constexpr bool Recordable<view::SetViewCameraTarget> = false;
template<> inline constexpr bool Recordable<view::SetViewCameraLens> = false;
template<> inline constexpr bool Recordable<view::SetViewCameraTargetDirection> = false;
} // namespace action
