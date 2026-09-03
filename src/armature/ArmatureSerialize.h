#pragma once

#include "action/SerializeNumeric.h"
#include "armature/Armature.h"

// Excludes derived caches because unordered-map iteration would make snapshots nondeterministic.

constexpr auto serialize(auto &archive, const ArmatureBone &b) { return archive(b.Id, b.ParentBoneId, b.JointNodeIndex, b.Name, b.RestLocal); }
constexpr auto serialize(auto &archive, ArmatureBone &b) { return archive(b.Id, b.ParentBoneId, b.JointNodeIndex, b.Name, b.RestLocal); }

constexpr auto serialize(auto &archive, const Armature &a) { return archive(a.Version, a.NextBoneId, a.Bones, a.Skins); }
constexpr auto serialize(auto &archive, Armature &a) {
    if constexpr (std::remove_cvref_t<decltype(archive)>::kind() == zpp::bits::kind::out) {
        return archive(a.Version, a.NextBoneId, a.Bones, a.Skins);
    } else {
        const auto result = archive(a.Version, a.NextBoneId, a.Bones, a.Skins);
        a.RebuildCaches();
        return result;
    }
}
