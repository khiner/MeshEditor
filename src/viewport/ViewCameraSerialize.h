#pragma once

#include "action/SerializeGlm.h" // glm hooks for the vec/Camera members
#include "viewport/ViewCamera.h"

// A plain memcpy of ViewCamera/LookingThrough would serialize garbage from the Camera variant's inactive alternative and disengaged optionals.
// Encode the persistent public state by-value through zpp instead, dropping the transient animation member (Emplace reconstructs it).
constexpr auto serialize(auto &archive, const ViewCamera &v) { return archive(v.Data, v.Target, v.Distance, v.Orientation); }
constexpr auto serialize(auto &archive, ViewCamera &v) { return archive(v.Data, v.Target, v.Distance, v.Orientation); }
constexpr auto serialize(auto &archive, const LookingThrough &l) { return archive(l.SavedViewCamera); }
constexpr auto serialize(auto &archive, LookingThrough &l) { return archive(l.SavedViewCamera); }
