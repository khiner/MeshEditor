#pragma once

#include "entt_fwd.h"
#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "selection/BoneSelection.h"

#include <memory>
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

// Box rectangle in render pixels; the hit entities are resolved against current scene state when applied.
// ViewProj is the view-projection at record time (heap-held to keep the Action variant small),
// so replay resolves pixels against the recorded view.
struct ApplyBoxSelect {
    std::pair<uvec2, uvec2> BoxPx;
    bool Additive;
    std::unique_ptr<mat4> ViewProj;
};
// Object/bone click pick at a pixel; the hit entity is resolved against current scene state when applied.
struct Pick {
    uvec2 MousePx;
    bool Shift;
    std::unique_ptr<mat4> ViewProj;
};
// Re-click at the same spot to cycle to the next overlapping hit under the cursor.
struct PickCycle {
    uvec2 MousePx;
    bool Shift;
    std::unique_ptr<mat4> ViewProj;
};
struct ApplyEditElementClick {
    uvec2 MousePx;
    bool Toggle;
    std::unique_ptr<mat4> ViewProj;
};
struct ApplyTreeSelection {
    enum class ClearKind : uint8_t { None,
                                     BonesOnly,
                                     All };
    std::vector<entt::entity> ToSelect, ToDeselect;
    entt::entity NavToActive{null_entity};
    ClearKind Clear{ClearKind::None};
};

using Action = std::variant<
    Select, ToggleSelected, SelectBone, ExtendActive, ExtendBoneActive, SetBoneSelectionPart,
    DeselectAll, SelectAll, SnapshotBoxSelectBaseline, ClearBoxSelectBaseline,
    ApplyBoxSelect, Pick, PickCycle, ApplyEditElementClick, ApplyTreeSelection>;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::selection
