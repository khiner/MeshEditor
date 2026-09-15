#pragma once

#include <imgui.h>

#include <optional>
#include <string>

namespace ui {
// Renders `count` items as tree nodes, each with a delete button, and returns the index whose delete button was pressed.
// `label(i)` names the node and `body(i)` fills it while expanded.
template<typename Label, typename Body>
std::optional<uint32_t> ItemList(size_t count, Label &&label, Body &&body) {
    std::optional<uint32_t> deleted;
    for (uint32_t i = 0; i < count; ++i) {
        ImGui::PushID(int(i));
        const bool expanded = ImGui::TreeNodeEx("##node", ImGuiTreeNodeFlags_SpanLabelWidth, "%s", std::string{label(i)}.c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("X")) deleted = i;
        if (expanded) {
            body(i);
            ImGui::TreePop();
        }
        ImGui::PopID();
    }
    return deleted;
}
} // namespace ui
