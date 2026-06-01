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
} // namespace action
