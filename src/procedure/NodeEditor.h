#pragma once

#include "Panel.h"

namespace state {
struct Registry;
} // namespace state

namespace procedure {
state::Entity EditorNetwork(const state::Registry &, const NodesPanel &);
void EnterNetwork(NodesPanel &, state::Entity network, state::Entity object = state::Null);
// Turns the editor to the page of the active node when that node changes.
// A geometry page turns to the newly active object's geometry.
void FollowActiveNode(const state::Registry &, NodesPanel &);
void DrawNetworkNavigation(const state::Registry &, NodesPanel &);
void DrawSubnetParameters(state::Registry &, state::Entity subnet, bool interactive);
// Lists the entities and fields the animation has channels for, each entity as a button that reveals it.
void DrawAnimationTargets(const state::Registry &, NodesPanel &, uint32_t animation);
void DrawNodesPanel(state::Registry &, NodesPanel &, bool interactive);
// Draws the parameters of the active node of the geometry network the editor shows.
void DrawNodeParameters(state::Registry &, NodesPanel &, bool interactive);
// Draws the geometry node the editor shows as active, then the selection's actions and the active node.
void DrawPropertiesPanel(state::Registry &, NodesPanel &, Panel &, bool interactive);
void DrawNodeStyle(state::Registry &);
} // namespace procedure
