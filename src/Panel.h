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
    bool Parameters{true};
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

struct NetworkCanvas {
    state::Entity Network{state::Null};
    numeric::vec2 Offset{};
    float Zoom{1};
};
struct NodeCanvas {
    numeric::vec2 Offset{}; // <= 0
    float Zoom{1};
    state::Entity Network{state::Null};
    state::Entity Object{state::Null};
    std::vector<NetworkCanvas> Locations;
};
enum class NodeDrag : uint8_t { None,
                                Pan,
                                Move,
                                Grab,
                                Wire,
                                Box };
namespace procedure {
struct CookPlan;
}
struct NodesPanel : Panel {
    std::shared_ptr<procedure::CookPlan> NetworkView;
    state::Entity ViewedNetwork{state::Null};
    NodeCanvas Canvas{};
    uint64_t Epoch{};
    NodeDrag Drag{NodeDrag::None};
    numeric::vec2 Press{}; // The drag's start on the screen
    numeric::uvec2 MoveStart{}; // The held node's cell when a move started
    numeric::uvec2 AddAt{}; // The cell the Add menu adds a node at
    uint32_t ContextNode{UINT32_MAX};
    uint16_t ContextSocket{UINT16_MAX};
    uint32_t DragNode{UINT32_MAX}; // Node index for the active move, grab, or wire gesture
    uint32_t GrabAfter{}; // Start dragging the new active node after the count exceeds this value
    uint32_t LinkAfter{UINT32_MAX}; // The node count a linking add waits to grow past before it links the new node, or none
    std::vector<action::nodes::Link> PendingLinks; // The links a Make Link press stages one per frame, committing with the last
    uint32_t Expanded{UINT32_MAX}; // The node whose parameters tray is open, with a row per parameter socket, or none
    uint32_t PickNode{UINT32_MAX}; // The node whose sockets in PickRange the port picker offers the wire held at DragNode's DragSocket, or none
    std::pair<uint16_t, uint16_t> PickRange{}; // The first socket the picker offers, through its end
    uint16_t DragSocket{}; // Socket index for the active wire gesture
    uint16_t LinkSocket{}; // The socket of the node a linking add appends that its wire links to
    bool Moved{false}; // A move passed the drag threshold and staged its first step
    bool Hovered{false}; // The canvas is under the mouse this frame, which routes the editing shortcuts to it
    bool Linking{false}; // The Add menu lists the sockets that link with the socket a dropped wire held, and links its node through the picked one
    std::array<char, 64> Search{}; // The Add menu's search text
    int SearchHighlight{}; // The search result Enter adds
    std::array<char, 64> NameEdit{}; // The attribute name the active name field edits
    uint32_t NameEditId{}; // The name field editing NameEdit, or zero
};

struct PanelsState {
    Panel
        SceneControls{"Scene controls", &PanelVisibility::SceneControls},
        Timeline{"Timeline", &PanelVisibility::Timeline},
        Style{"Style", &PanelVisibility::Style},
        Info{"Info", &PanelVisibility::Info},
        Parameters{"Parameters", &PanelVisibility::Parameters},
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
    return std::to_array<P>({&panels.History, &panels.Nodes, &panels.Parameters, &panels.Info, &panels.Style, &panels.Debug, &panels.ImGuiDemo, &panels.ImSpinnerDemo, &panels.ImPlotDemo, &panels.SceneControls, &panels.Timeline});
}

inline PanelVisibility GetPanelVisibility(const PanelsState &panels) {
    PanelVisibility visibility;
    for (const auto *panel : AllPanels(panels)) visibility.*panel->Flag = panel->Visible;
    return visibility;
}

inline void SetPanelVisibility(PanelsState &panels, const PanelVisibility &visibility) {
    for (auto *panel : AllPanels(panels)) panel->Visible = visibility.*panel->Flag;
}
