#pragma once

#include "numeric/uvec2.h"

#include "numeric/vec2.h"
#include "selection/BoneSelection.h"
#include "state/Entity.h"
#include "viewport/RenderView.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace action::selection {
struct Select {
    state::Entity Entity;
};
struct ToggleSelected {
    state::Entity Entity;
};
// `Part`/`Additive` describe an optional sub-part merged after the selection state is established.
struct SelectBone {
    state::Entity Entity;
    std::optional<BoneSel> Part{};
    bool Additive{false};
};
struct ExtendActive {
    state::Entity Entity;
};
struct ExtendBoneActive {
    state::Entity Entity;
    std::optional<BoneSel> Part{};
    bool Additive{false};
};
struct DeselectAll {};
struct SelectAll {};

struct ApplyBoxSelect {
    std::pair<uvec2, uvec2> BoxPx;
    bool Additive;
    std::unique_ptr<RenderView> View;
};
// Object/bone click pick at a pixel; the hit entity is resolved against current scene state when applied.
struct Pick {
    uvec2 MousePx;
    bool Shift;
    std::unique_ptr<RenderView> View;
};
// Re-click at the same spot to cycle to the next overlapping hit under the cursor.
struct PickCycle {
    uvec2 MousePx;
    bool Shift;
    std::unique_ptr<RenderView> View;
};
struct ApplyEditElementClick {
    uvec2 MousePx;
    bool Toggle;
    std::unique_ptr<RenderView> View;
};
struct ApplyTreeSelection {
    enum class ClearKind : uint8_t { None,
                                     BonesOnly,
                                     All };
    std::vector<state::Entity> Entities;
    uint32_t SelectCount{};
    state::Entity NavToActive{state::Null};
    ClearKind Clear{ClearKind::None};

    auto ToSelect() const { return std::span{Entities}.first(SelectCount); }
    auto ToDeselect() const { return std::span{Entities}.subspan(SelectCount); }
    void Add(state::Entity e, bool selected) {
        if (selected) Entities.insert(Entities.begin() + SelectCount++, e);
        else Entities.push_back(e);
    }
};

using Action = std::variant<
    Select, ToggleSelected, SelectBone, ExtendActive, ExtendBoneActive,
    DeselectAll, SelectAll,
    ApplyBoxSelect, Pick, PickCycle, ApplyEditElementClick, ApplyTreeSelection>;

void Apply(state::Scene &, state::Entity viewport, const Action &);
} // namespace action::selection
