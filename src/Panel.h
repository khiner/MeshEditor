#pragma once

#include "action/Nodes.h"
#include "viewport/ViewportSchema.h"

#include <array>

struct TabSelection {
    uint32_t Bar{}, Tab{};
    bool operator==(const TabSelection &) const = default;
};

struct PanelState {
    struct StorageValue {
        uint32_t Key{}, Value{};
        bool operator==(const StorageValue &) const = default;
    };
    uint32_t Window{};
    float X{}, Y{};
    std::vector<StorageValue> Storage;
    bool operator==(const PanelState &) const = default;
};

struct PendingPanelState {
    PanelState Value;
    bool StorageApplied{false}, ScrollApplied{false};
};

struct PanelVisibility {
    bool SceneControls{true};
    bool Timeline{true};
    bool Style{true};
    bool Info{true};
    bool Properties{true};
    bool ImGuiDemo{false}, ImSpinnerDemo{false}, ImPlotDemo{false};
    bool Debug{false};
    bool History{true};
    bool Nodes{true};
};

struct Panel {
    const char *Name;
    bool PanelVisibility::*Flag; // The panel's flag in the workspace's PanelVisibility
    bool Visible{true};
};

struct HistoryRow {
    uint32_t Node;
    uint32_t Depth; // The original continuation stays inline; later branches nest.
    uint32_t Rails{}; // Ancestor forks with another branch below this row.
};
struct HistoryPanel : Panel {
    bool ValidateRequested{false};
    uint64_t TreeRevision{UINT64_MAX};
    std::vector<HistoryRow> Rows{};
    std::vector<std::optional<uint32_t>> RowByNode{};
    std::optional<uint32_t> EditorNode; // The node whose editor the open state belongs to.
    bool EditorOpen{true};
};

struct GraphCanvas {
    state::Entity Graph{state::Null};
    numeric::vec2 Offset{};
    float Zoom{1};
};
// The page shows Root's graph, or the subgraph of the last subgraph node of Path.
struct NodeCanvas {
    numeric::vec2 Offset{}; // <= 0
    float Zoom{1};
    state::Entity Root{state::Null};
    state::Entity Object{state::Null};
    std::vector<state::Entity> Path; // The subgraph nodes from one in Root's graph to the one whose subgraph's page shows, outermost first
    state::Entity Inspected{state::Null}; // The node of a built-in subgraph's page whose parameters Properties shows read-only
    std::vector<GraphCanvas> Locations;
};
// The entity whose page the canvas shows: its root graph, or the last subgraph node of its path.
inline state::Entity PageGraph(const NodeCanvas &canvas) { return canvas.Path.empty() ? canvas.Root : canvas.Path.back(); }
enum class NodeDrag : uint8_t { None,
                                Pan,
                                Move,
                                Grab,
                                Wire,
                                Box };
namespace graph {
struct UseCounts;
struct NodePage;
} // namespace graph
struct NodesPanel : Panel {
    std::shared_ptr<graph::NodePage> Page; // The displayed page with its layout, kept between frames
    NodeCanvas Canvas{};
    uint64_t Epoch{};
    NodeDrag Drag{NodeDrag::None};
    numeric::vec2 Press{}; // The drag's start on the screen
    numeric::uvec2 MoveStart{}; // The held node's cell when a move started
    numeric::uvec2 AddAt{}; // The cell the Add menu opened at, where the grab of its new node starts
    uint32_t ContextNode{UINT32_MAX};
    uint16_t ContextSocket{UINT16_MAX};
    uint32_t DragNode{UINT32_MAX}; // Node index for the active move, grab, or wire gesture
    uint32_t GrabAfter{}; // Start dragging the new active node after the count exceeds this value
    std::optional<action::nodes::Link> LinkingAdd; // The link a staged linking add makes next frame, whose null end is the node it added
    state::Entity LinkingActive{state::Null}; // The active node of LinkingAdd's graph before the add, which the added node replaces
    uint32_t Expanded{UINT32_MAX}; // The node whose tray is open, or none
    uint32_t PickNode{UINT32_MAX}; // The node whose sockets the port picker lists for the held wire, or none
    uint32_t DetachNode{UINT32_MAX}; // The node whose input link the wire drag picked up, or none
    uint16_t DetachSlot{}; // The input slot of DetachNode that the picked-up link is in
    bool PickTray{false}; // The port picker lists only the node's tray sockets
    uint16_t DragSocket{}; // Socket index for the active wire gesture
    bool Moved{false}; // A move passed the drag threshold and staged its first step
    bool Hovered{false}; // The canvas is under the mouse this frame, which routes the editing shortcuts to it
    bool Linking{false}; // The Add menu lists the sockets that link with the socket a dropped wire held, and links its node through the picked one
    std::array<char, 64> Search{}; // The Add menu's search text
    int SearchHighlight{}; // The search result Enter adds
    std::optional<state::Entity> Followed; // The active node the editor last saw, or none before its first frame
    state::Entity UsersOf{state::Null}; // The item whose users the Properties panel lists
    uint64_t UsersRevision{UINT64_MAX}; // The graph revision Users was read at
    std::vector<graph::SceneReference> Users; // The references that hold UsersOf
    uint64_t UsesRevision{UINT64_MAX}; // The graph revision Uses was counted at
    std::shared_ptr<graph::UseCounts> Uses; // The subgraph nodes that use each subgraph
    std::array<char, 64> NameEdit{}; // The text the active text field edits
    uint32_t NameEditId{}; // The text field editing NameEdit, or zero
    // The textures of an image Properties drew, kept until the UI frame that drew them completes, and the ImGui frame that last drew it.
    struct ShownImage {
        std::shared_ptr<const void> Hold;
        int Frame{};
    };
    ShownImage NodeImage, Preview; // A node's image or output, and a material's preview
};

struct PanelsState {
    Panel
        SceneControls{"Scene controls", &PanelVisibility::SceneControls},
        Timeline{"Timeline", &PanelVisibility::Timeline},
        Style{"Style", &PanelVisibility::Style},
        Info{"Info", &PanelVisibility::Info},
        Properties{"Properties", &PanelVisibility::Properties},
        ImGuiDemo{"Dear ImGui Demo", &PanelVisibility::ImGuiDemo, false},
        ImSpinnerDemo{"ImSpinner Demo", &PanelVisibility::ImSpinnerDemo, false},
        ImPlotDemo{"ImPlot Demo", &PanelVisibility::ImPlotDemo, false},
        Debug{"Debug", &PanelVisibility::Debug, false};
    HistoryPanel History{{"History", &PanelVisibility::History}};
    NodesPanel Nodes{{"Nodes", &PanelVisibility::Nodes}};
    std::string PendingIni;
    std::vector<TabSelection> PendingTabs;
    std::vector<PendingPanelState> PendingPanels;
    bool LayoutLoaded{false};
    bool IniApplied{false}; // The ini loaded at the last frame end, so the windows it places appear this frame.
};

// Every panel in the order the Panels menu lists them.
template<typename Panels> auto AllPanels(Panels &panels) {
    using P = std::conditional_t<std::is_const_v<Panels>, const Panel *, Panel *>;
    return std::to_array<P>({&panels.History, &panels.Nodes, &panels.Info, &panels.Style, &panels.Debug, &panels.ImGuiDemo, &panels.ImSpinnerDemo, &panels.ImPlotDemo, &panels.SceneControls, &panels.Properties, &panels.Timeline});
}

inline PanelVisibility GetPanelVisibility(const PanelsState &panels) {
    PanelVisibility visibility;
    for (const auto *panel : AllPanels(panels)) visibility.*panel->Flag = panel->Visible;
    return visibility;
}

inline void SetPanelVisibility(PanelsState &panels, const PanelVisibility &visibility) {
    for (auto *panel : AllPanels(panels)) panel->Visible = visibility.*panel->Flag;
}
