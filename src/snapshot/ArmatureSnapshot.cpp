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

void RegisterArmature(Tables &tables) {
    Persistent<
        Armature, ArmatureObject, BoneJointEntities, BoneJoint, BoneSubPartOf, BoneActive, BoneSelection, BoneConstraints,
        ArmatureModifier, BoneIndex, BoneDisplayScale, BoneAttachment, ArmatureAnimation, ArmaturePose>(tables);
    Derived<PosedLocal, BoneAdjacencyIndices, BoneInstanceStateDirty, ArmaturePoseState>(tables);
}
} // namespace snapshot::detail
