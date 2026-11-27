#pragma once

#include "gpu/InteractionMode.h"
#include "selection/Selection.h"
#include "selection/SelectionSchema.h"

#include <unordered_set>

// The interaction mode and whether the selection it transforms is bones, bone rest poses, or mesh elements.
struct ModeScope {
    InteractionMode Mode;
    bool BoneEdit, Bone, MeshEdit;
};
ModeScope ScopeOf(const state::Registry &);

// Derived viewport aggregates of the selection in the frame's scene, each recomputed in the settle pass when its own inputs change.
// Per-frame readers take them in O(1).

// Each edited mesh's primary object, in Edit mode only.
struct EditPrimaries {
    selection::PrimaryEditObjectMap All, Transformable;
};

// One selected visible object per mesh, preferring the active object, then the lowest entity ID.
// Transformable chooses among the objects that are not scale-locked.
EditPrimaries ComputeEditPrimaries(const state::Registry &);

// The selected entities a transform moves, and the pivot it turns and scales about.
struct TransformRoots {
    // Selected bones in pose and bone edit modes, else selected objects, each without a selected ancestor.
    // Every selected bone in bone edit mode is a root, since rest-pose edits do not propagate during a drag.
    std::vector<state::Entity> Roots;
    // The roots' mean world position, with a selected tip contributing in bone edit mode.
    vec3 PivotPosition{};
    // The active entity's world rotation, the active bone's in bone modes.
    quat PivotRotation{1, 0, 0, 0};
};

// Whether the shaded elements are smooth, sharp, or both.
enum class SelectionSharpness : uint8_t {
    None, // No faces or elements to shade.
    Smooth,
    Sharp,
    Mixed,
};

struct SelectionFlags {
    std::vector<state::Entity> Meshes; // The meshes that selected objects draw, sorted.
    bool AllMeshes{true}; // Every selected object is a mesh object.
    bool AnyVisible{false}, AnyHidden{false}; // Over selected objects that are not sub-elements.
    bool Transformable{true}; // The selection transforms, which a mesh that a scale-locked object draws blocks in Edit mode.
    bool Scalable{true}; // It also scales, which a scale-locked selected entity blocks.
    bool HasTransformTarget{false}; // The gizmo has something to move.
    SelectionSharpness FaceSharpness{SelectionSharpness::None}; // Of the selected face meshes' faces.
    SelectionSharpness ElementSharpness{SelectionSharpness::None}; // Of the primaries' selected elements, in mesh Edit mode only.
    std::vector<uint8_t> FaceFlags; // The live sharpness of each of Meshes' faces: 1 when any face is sharp, 2 when any is smooth.
    std::array<uint32_t, 4> FaceFlagMeshes{}; // The number of Meshes with each FaceFlags value.
};

// Timeline frames holding a key on any channel of the selected objects' active clips or of the active mesh's displayed material.
// Sorted ascending without duplicates.
struct SelectedKeyframes {
    std::vector<float> Frames;
};

// The outliner's rows in draw order: the scene's sorted named roots, each followed by its open subtree.
struct OutlinerRows {
    struct Row {
        state::Entity Entity;
        uint32_t Depth;
        bool HasChildren;
    };
    std::vector<Row> Rows;
    std::vector<uint32_t> RowOf; // Row index by entity index, UINT32_MAX for an entity without a row.
    std::unordered_set<state::Entity> Open; // Entities whose children show.
    std::unordered_set<state::Entity> SelectedAncestors; // Ancestors of selected objects and bones.
    state::Entity Scene{state::Null}; // The scene the rows were built for.

    // The entity's row, or nothing when it has none.
    std::optional<uint32_t> Find(state::Entity) const;
};

// Rebuilds the rows of the frame's scene's roots from the scene graph and the open set.
void BuildOutlinerRows(const state::Registry &, OutlinerRows &);

// Recomputes the aggregates whose inputs changed this settle, and returns the meshes whose primary edit object changed.
// Runs after the world transforms are current.
std::vector<state::Entity> UpdateSelectionState(state::Registry &);
