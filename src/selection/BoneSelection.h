#pragma once

#include <cstdint>

// Bone selection part identifier.
enum class BoneSel : uint8_t {
    Root,
    Tip,
    Body
};

// Presence means the bone is selected. Fields indicate which parts are selected.
struct BoneSelection {
    bool Root{true}, Tip{true}, Body{true};

    static BoneSelection From(BoneSel part) { return {part != BoneSel::Tip, part != BoneSel::Root, part == BoneSel::Body}; }
    BoneSelection operator|(BoneSelection o) const {
        const bool r = Root || o.Root, t = Tip || o.Tip;
        return {r, t, Body || o.Body || (r && t)};
    }
    bool Has(BoneSel part) const { return part == BoneSel::Root ? Root : part == BoneSel::Tip ? Tip :
                                                                                                Body; }
};
