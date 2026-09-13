#include "project/HistoryUi.h"

#include "project/Project.h"

#include <imgui.h>

#include <format>

namespace project {
using namespace ImGui;

void HandleHistoryShortcuts(Project &session) {
    if (GetIO().WantTextInput) return;
    auto &history = session.History;
    const int present = history.Present;
    if (Shortcut(ImGuiMod_Ctrl | ImGuiKey_Z, ImGuiInputFlags_RouteGlobal) && history.CanUndo()) session.RequestNavigate(history.Nodes[present].Parent);
    if (Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_Z, ImGuiInputFlags_RouteGlobal) && history.CanRedo()) session.RequestNavigate(history.Nodes[present].Children.back());
}

bool DrawHistoryWindow(Project &session, HistoryWindow &window, bool interactive) {
    if (!window.Visible) return false;
    bool clear = false;
    if (Begin(window.Name, &window.Visible)) {
        auto &history = session.History;
        const auto &nodes = history.Nodes;
        const int present = history.Present;
        BeginDisabled(!history.CanUndo());
        if (Button("Undo") && interactive) session.RequestNavigate(nodes[present].Parent);
        EndDisabled();
        SameLine();
        BeginDisabled(!history.CanRedo());
        if (Button("Redo") && interactive) session.RequestNavigate(nodes[present].Children.back());
        EndDisabled();
        SameLine();
        if (Button("Memory...") && interactive) OpenPopup("History memory");
        if (BeginPopup("History memory")) {
            const auto stats = history.Stats();
            Text("Copied data: %.2f MiB", double(stats.OwnedBytes) / (1 << 20));
            Text("Estimated history memory: %.2f MiB", double(stats.RetainedBytes()) / (1 << 20));
            Text("%zu states cached, %zu require disk reads", stats.HotNodes, stats.ColdNodes);
            int cap = int(session.MemoryCap >> 20);
            if (DragInt("Copied data budget (MiB)", &cap, 1.f, 0, 4096) && interactive) {
                session.MemoryCap = uint64_t(std::max(0, cap)) << 20;
                history.Evict(session.MemoryCap);
            }
            Text("The current state and up to %d recent edits stay cached.", store::History::UndoWindow);
            EndPopup();
        }
#ifdef DEBUG_BUILD
        if (Button("Validate history") && interactive) window.ValidateRequested = true;
        SameLine();
#endif
        clear = Button("Clear history") && interactive;
        SetItemTooltip("Keep the current and last saved states. Discard other undo/redo states.");
        Text("%zu states", nodes.size());
        Separator();
        if (window.TreeRevision != history.Revision) {
            window.Rows.clear();
            std::vector<int> pending;
            if (!nodes.empty()) pending.push_back(0);
            while (!pending.empty()) {
                const auto id = pending.back();
                pending.pop_back();
                window.Rows.push_back(id);
                for (auto it = nodes[id].Children.rbegin(); it != nodes[id].Children.rend(); ++it) pending.push_back(*it);
            }
            window.TreeRevision = history.Revision;
        }
        const float x = GetCursorPosX();
        ImGuiListClipper clipper;
        clipper.Begin(int(window.Rows.size()), GetTextLineHeightWithSpacing());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                const int id = window.Rows[row];
                const auto &node = nodes[id];
                SetCursorPosX(x + float(std::min(node.Depth, 20)) * 12.f);
                if (Selectable(std::format("{}: {}", id, node.Label).c_str(), id == present) && interactive) session.RequestNavigate(id);
            }
        }
    }
    End();
    return clear;
}

} // namespace project
