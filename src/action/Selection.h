#pragma once

#include "entt_fwd.h"
#include "numeric/vec2.h"
#include "selection/BoneSelection.h"

#include <optional>
#include <vector>

namespace action::selection {
struct Select {
    entt::entity Entity;
};
struct ToggleSelected {
    entt::entity Entity;
};
// `Part`/`Additive` describe an optional sub-part merged after the selection state is established.
struct SelectBone {
    entt::entity Entity;
    std::optional<BoneSel> Part{};
    bool Additive{false};
};
struct ExtendActive {
    entt::entity Entity;
};
struct ExtendBoneActive {
    entt::entity Entity;
    std::optional<BoneSel> Part{};
    bool Additive{false};
};
struct SetBoneSelectionPart {
    entt::entity Entity;
    std::optional<BoneSel> Part;
    bool Additive;
};
struct DeselectAll {};
struct SelectAll {};
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
    std::vector<entt::entity> ToSelect, ToDeselect;
    entt::entity NavToActive{null_entity};
    ClearKind Clear{ClearKind::None};
};

using Actions = std::variant<
    Select, ToggleSelected, SelectBone, ExtendActive, ExtendBoneActive, SetBoneSelectionPart,
    DeselectAll, SelectAll, SnapshotBoxSelectBaseline, ClearBoxSelectBaseline,
    ApplyBoxSelectObjectHits, ApplyBoxSelectBoneHits, ApplyEditElementClick, ApplyTreeSelection>;
using Action = Actions;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::selection
