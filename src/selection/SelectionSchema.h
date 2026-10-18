#pragma once

#include "state/Slot.h"

struct BoneSelection;
struct SelectionChanges;
struct MeshElementSelection;
struct AdditiveBoxSelectBaseline;
struct ExciteSelectionBaseline;
struct PendingEditElementClick;
struct PendingBoxSelect;
struct PendingPick;
struct BoxSelectState;
struct SelectionSlots;
struct EditPrimaries;
struct TransformRoots;
struct SelectionFlags;
struct SelectedKeyframes;
struct OutlinerRows;

using SelectionSchema = state::Schema<state::SelectionBase, state::SelectionCount, BoneSelection, SelectionChanges, MeshElementSelection, AdditiveBoxSelectBaseline, ExciteSelectionBaseline, PendingEditElementClick, PendingBoxSelect, PendingPick, BoxSelectState, SelectionSlots, EditPrimaries, TransformRoots, SelectionFlags, SelectedKeyframes, OutlinerRows>;
template<typename T>
    requires SelectionSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = SelectionSchema::Of<T>();
