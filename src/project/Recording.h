#pragma once

#include "action/Action.h"
#include "project/Project.h"

// The parts of a project that take or return the full action type.
namespace project {
// The actions of a history node, decoded for editing.
struct EditDraft {
    uint32_t Node;
    uint64_t Revision; // The history revision the actions were decoded at.
    std::vector<action::Action> RecordedActions;
};

// An action's name, and for a field write the component and field it names.
std::string Label(const action::Action &);
// Applies the action and commits it as its own history step, after finishing any open gesture.
// Returns the committed node, or the present node when nothing was recorded.
std::optional<uint32_t> Do(Project &, action::Action, std::string label = {});
// The draft of `node`, decoded anew when the node or the history changed.
EditDraft &DraftOf(Project &, uint32_t node);
} // namespace project
