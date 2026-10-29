#pragma once

#include "Panel.h"

namespace state {
struct Registry;
} // namespace state

namespace procedure {
state::Entity EditorNetwork(const state::Registry &, const NodesPanel &);
void EnterNetwork(NodesPanel &, state::Entity network, state::Entity object = state::Null);
void DrawNetworkNavigation(const state::Registry &, NodesPanel &);
void DrawSubnetParameters(state::Registry &, NodesPanel &, state::Entity subnet, bool interactive);
void DrawNetworkParameters(state::Registry &, NodesPanel &);
void DrawNodesPanel(state::Registry &, NodesPanel &, bool interactive);
void DrawParametersPanel(state::Registry &, NodesPanel &, Panel &, bool interactive);
void DrawNodeStyle(state::Registry &);
} // namespace procedure
