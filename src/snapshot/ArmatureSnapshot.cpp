#include "animation/AnimationData.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "render/MeshBuffers.h"
#include "scene/WorldTransform.h"
#include "selection/BoneSelection.h"
#include "snapshot/SnapshotRegistration.h"

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

namespace snapshot::detail {

template<> Armature CopyNative(const Armature &source) {
    Armature result;
    result.Version = source.Version;
    result.NextBoneId = source.NextBoneId;
    result.Skins = source.Skins;
    result.Bones.reserve(source.Bones.size());
    for (const auto &b : source.Bones) result.Bones.push_back({b.Id, b.ParentBoneId, b.JointNodeIndex, b.Name, b.RestLocal});
    return result;
}
template<> void PrepareNative(Armature &value) { value.RebuildCaches(); }

void RegisterArmature(Tables &tables) {
    Persistent<
        Armature, ArmatureObject, BoneJointEntities, BoneJoint, BoneSubPartOf, BoneActive, BoneSelection, BoneConstraints,
        ArmatureModifier, BoneIndex, BoneDisplayScale, BoneAttachment, ArmatureAnimation, ArmaturePose>(tables);
    Derived<PosedLocal, BoneAdjacencyIndices, ArmaturePoseState>(tables);
}
} // namespace snapshot::detail
