#pragma once

#include "selection/Selection.h"
#include "state/Entity.h"

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

// Positions are fractions of the viewport raster extent.
struct ApplyBoxSelect {
    std::pair<vec2, vec2> Box;
    bool Additive;
};
struct Pick {
    bool Shift;
    std::optional<SelectionHit> Hit;
};
struct ApplyEditElementClick {
    bool Toggle;
    std::optional<ElementHit> Hit;
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
    ApplyBoxSelect, Pick, ApplyEditElementClick, ApplyTreeSelection>;

void Apply(state::Registry &, const Action &);
} // namespace action::selection
