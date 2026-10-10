#pragma once

#include "action/Nodes.h"
#include "numeric/vec2.h"
#include "state/Entity.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct TabSelection {
    uint32_t Bar{}, Tab{};
    bool operator==(const TabSelection &) const = default;
};

struct WindowState {
    struct StorageValue {
        uint32_t Key{}, Value{};
        bool operator==(const StorageValue &) const = default;
    };
    uint32_t Window{};
    float X{}, Y{};
    std::vector<StorageValue> Storage;
    bool operator==(const WindowState &) const = default;
};

struct PendingWindowState {
    WindowState Value;
    bool StorageApplied{false}, ScrollApplied{false};
};

struct Window {
    const char *Name{""};
    bool Visible{true};
};

struct HistoryRow {
    uint32_t Node;
    uint32_t Depth; // The original continuation stays inline; later branches nest.
    uint32_t Rails{}; // Ancestor forks with another branch below this row.
};
struct HistoryWindow : Window {
    bool ValidateRequested{false};
    uint64_t TreeRevision{UINT64_MAX};
    std::vector<HistoryRow> Rows{};
    std::vector<std::optional<uint32_t>> RowByNode{};
    std::optional<uint32_t> EditorNode; // The node whose editor the open state belongs to.
    bool EditorOpen{true};
};

// The node canvas draws a node position p at the screen point Origin + Offset + p * Zoom, where Origin is the canvas's top left.
struct NodeCanvas {
    numeric::vec2 Offset{40, 40};
    float Zoom{1};
};
enum class NodeDrag : uint8_t { None,
                                Pan,
                                Move,
                                Grab, // The selected nodes follow the mouse until a click confirms
                                Wire,
                                Box };
struct NodesWindow : Window {
    NodeCanvas Canvas{};
    NodeDrag Drag{NodeDrag::None};
    numeric::vec2 Press{}; // The drag's start on the screen
    numeric::vec2 MoveStart{}; // The held node's position when a move started
    numeric::vec2 AddAt{}; // The canvas position the Add menu adds a node at
    uint32_t DragNode{UINT32_MAX}; // The node a move, a grab or a wire holds
    uint32_t GrabAfter{}; // The node count a grab waits to grow past before it holds the new active node
    uint32_t LinkAfter{UINT32_MAX}; // The node count a linking add waits to grow past before it links the new node, or none
    std::vector<action::nodes::Link> PendingLinks; // The links a Make Link press stages one per frame, committing with the last
    uint16_t DragSocket{}; // The socket a wire holds
    bool Moved{false}; // A move passed the drag threshold and staged its first step
    bool Hovered{false}; // The canvas is under the mouse this frame, which routes the editing shortcuts to it
    bool Linking{false}; // The Add menu offers the kinds that link with the socket a dropped wire held, and links its node to that socket
    std::array<char, 64> Search{}; // The Add menu's search text
    int SearchHighlight{}; // The search result Enter adds
    std::array<char, 64> NameEdit{}; // The attribute name the active name field edits
    uint32_t NameEditId{}; // The name field editing NameEdit, or zero
};

struct WindowsState {
    Window
        SceneControls{"Scene controls"},
        Viewport{"Viewport"},
        Animation{"Animation"},
        ImGuiDemo{"Dear ImGui Demo", false},
        ImSpinnerDemo{"ImSpinner Demo", false},
        ImPlotDemo{"ImPlot Demo", false},
        Debug{"Debug", false};
    HistoryWindow History{{"History"}};
    NodesWindow Nodes{{"Nodes"}};
    std::string PendingIni;
    std::vector<TabSelection> PendingTabs;
    std::vector<PendingWindowState> PendingWindows;
    bool LayoutLoaded{false};
};

struct WindowVisibility {
    bool SceneControls{true};
    bool Viewport{true};
    bool Animation{true};
    bool ImGuiDemo{false};
    bool ImSpinnerDemo{false};
    bool ImPlotDemo{false};
    bool Debug{false};
    bool History{true};
    bool Nodes{true};
};

inline WindowVisibility GetWindowVisibility(const WindowsState &windows) {
    return {
        windows.SceneControls.Visible,
        windows.Viewport.Visible,
        windows.Animation.Visible,
        windows.ImGuiDemo.Visible,
        windows.ImSpinnerDemo.Visible,
        windows.ImPlotDemo.Visible,
        windows.Debug.Visible,
        windows.History.Visible,
        windows.Nodes.Visible,
    };
}

inline void SetWindowVisibility(WindowsState &windows, const WindowVisibility &visibility) {
    windows.SceneControls.Visible = visibility.SceneControls;
    windows.Viewport.Visible = visibility.Viewport;
    windows.Animation.Visible = visibility.Animation;
    windows.ImGuiDemo.Visible = visibility.ImGuiDemo;
    windows.ImSpinnerDemo.Visible = visibility.ImSpinnerDemo;
    windows.ImPlotDemo.Visible = visibility.ImPlotDemo;
    windows.Debug.Visible = visibility.Debug;
    windows.History.Visible = visibility.History;
    windows.Nodes.Visible = visibility.Nodes;
}
