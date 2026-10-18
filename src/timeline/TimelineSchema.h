#pragma once

#include "state/Slot.h"

struct TimelineRange;
struct TimelinePlayback;
struct TimelineNavigation;
struct TimelineView;
struct PlaybackFrame;
struct LastEvaluatedFrame;
struct TimelineTheme;

using TimelineSchema = state::Schema<state::TimelineBase, state::TimelineCount, TimelineRange, TimelinePlayback, TimelineNavigation, TimelineView, PlaybackFrame, LastEvaluatedFrame, TimelineTheme>;
template<typename T>
    requires TimelineSchema::Has<T>
inline constexpr state::TypeId state::Slot<T> = TimelineSchema::Of<T>();
