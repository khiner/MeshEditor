#pragma once

#include "Armature.h"
#include "entt_fwd.h"
#include "numeric/vec2.h"

#include <optional>
#include <vector>

namespace action::selection {
struct Select {
    entt::entity Entity;
};
struct ToggleSelected {
    entt::entity Entity;
};
struct SelectBone {
    entt::entity Entity;
};
struct ExtendActive {
    entt::entity Entity;
};
struct ExtendBoneActive {
    entt::entity Entity;
};
struct SetBoneSelectionPart {
    entt::entity Entity;
    std::optional<BoneSel> Part;
    bool Additive;
};
struct DeselectAll {};
struct SnapshotBoxSelectBaseline {};
struct ClearBoxSelectBaseline {};
struct BoneHit {
    entt::entity Entity;
    std::optional<BoneSel> Part;
};
struct ApplyBoxSelectObjectHits {
    std::vector<entt::entity> Hits;
    bool Additive;
};
struct ApplyBoxSelectBoneHits {
    std::vector<BoneHit> Hits;
    bool Additive;
};
struct ApplyEditElementClick {
    uvec2 MousePx;
    bool Toggle;
};
struct ApplyTreeSelection {
    enum class ClearKind : uint8_t { None,
                                     BonesOnly,
                                     All };
    std::vector<entt::entity> ToSelect;
    std::vector<entt::entity> ToDeselect;
    entt::entity NavToActive{null_entity};
    ClearKind Clear{ClearKind::None};
};

using Actions = entt::type_list<
    Select, ToggleSelected, SelectBone, ExtendActive, ExtendBoneActive, SetBoneSelectionPart,
    DeselectAll, SnapshotBoxSelectBaseline, ClearBoxSelectBaseline,
    ApplyBoxSelectObjectHits, ApplyBoxSelectBoneHits, ApplyEditElementClick, ApplyTreeSelection>;
} // namespace action::selection
