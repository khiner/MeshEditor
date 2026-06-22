#pragma once

#include "action/SerializeGlm.h" // glm hooks for Transform/mat4 members
#include "armature/Armature.h"

#include <zpp_bits.h>

// Serialize only the Armature's canonical data, rebuilding the derived caches (BoneIdToIndex, dense topology, RestWorld, JointOrderToBoneIndex) on load.
// This keeps the snapshot deterministic, since BoneIdToIndex is an unordered_map with unstable iteration order.

constexpr auto serialize(auto &archive, const ArmatureBone &b) { return archive(b.Id, b.ParentBoneId, b.JointNodeIndex, b.Name, b.RestLocal); }
constexpr auto serialize(auto &archive, ArmatureBone &b) { return archive(b.Id, b.ParentBoneId, b.JointNodeIndex, b.Name, b.RestLocal); }

constexpr auto serialize(auto &archive, const Armature &a) { return archive(a.Version, a.NextBoneId, a.Bones, a.ImportedSkin); }
constexpr auto serialize(auto &archive, Armature &a) {
    if constexpr (std::remove_cvref_t<decltype(archive)>::kind() == zpp::bits::kind::out) {
        return archive(a.Version, a.NextBoneId, a.Bones, a.ImportedSkin);
    } else {
        const auto result = archive(a.Version, a.NextBoneId, a.Bones, a.ImportedSkin);
        a.RebuildCaches();
        return result;
    }
}
