#include "animation/AnimationData.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "armature/ArmatureSerialize.h"
#include "render/MeshBuffers.h"
#include "scene/WorldTransform.h"
#include "selection/BoneSelection.h"
#include "snapshot/SnapshotRegistration.h"

namespace snapshot::detail {
template<> inline constexpr bool ForceFieldwise<BoneSubPartOf> = true;

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
