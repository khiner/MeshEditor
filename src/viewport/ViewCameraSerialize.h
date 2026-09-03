#pragma once

#include "action/SerializeNumeric.h"
#include "viewport/ViewCamera.h"

// Excludes inactive variant storage and transient animation state from the serialized representation.
constexpr auto serialize(auto &archive, const ViewCamera &v) { return archive(v.Data, v.Target, v.Distance, v.Orientation); }
constexpr auto serialize(auto &archive, ViewCamera &v) { return archive(v.Data, v.Target, v.Distance, v.Orientation); }
constexpr auto serialize(auto &archive, const LookingThrough &l) { return archive(l.SavedViewCamera); }
constexpr auto serialize(auto &archive, LookingThrough &l) { return archive(l.SavedViewCamera); }
