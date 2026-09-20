#include "project/HistoryUi.h"

#include "Field.h"
#include "Variant.h"
#include "action/Dispatch.h"
#include "gpu/PBRMaterial.h"
#include "project/Project.h"
#include "scene/Entity.h"
#include "state/Schema.h"
#include "ui/CtrlShortcut.h"
#include "ui/FieldEdit.h"
#include "ui/TransformEdit.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace project {
using namespace ImGui;

namespace {
template<typename T>
concept Optional = requires(T t) { t.has_value(); t.emplace(); };
template<typename T>
concept Variant = requires { std::variant_size<T>::value; };
template<typename T>
concept Vector = requires(T t) { t.emplace_back(); t.pop_back(); };
template<typename T>
concept Pair = requires(T t) { t.first; t.second; };
template<typename T>
concept UniquePtr = requires { typename T::deleter_type; };

// "TransformSelection" reads "Transform Selection", and "Transform:S" keeps its field.
std::string SpacedName(std::string_view name) {
    std::string out(field::detail::SpacedSize(name), '\0');
    field::detail::SpaceWords(name, out.data());
    return out;
}
std::string TargetName(const state::Scene &r, const action::Target &target) {
    static constexpr const char *Names[]{"Active", "Selected", "Selected Delta", "Viewport"};
    const auto *e = std::get_if<state::Entity>(&target);
    if (!e) return Names[target.index()];
    const auto *name = r.try_get<const Name>(*e);
    return name ? name->Value : std::format("Entity {}", state::Integral(*e));
}
template<typename L> std::string LeafName() { return SpacedName(state::LeafName<L>()); }

template<typename T> void DrawFields(const state::Scene &, T &value, bool &changed, bool &finished);

template<typename T> void DrawValue(const state::Scene &r, const char *label, T &v, const FieldSpec &spec, bool &changed, bool &finished) {
    const auto group = [&](auto &&draw) {
        if (!TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) return;
        draw();
        TreePop();
    };
    if constexpr (std::same_as<T, state::Entity>) {
        const auto *current = v == state::Null ? nullptr : r.try_get<const Name>(v);
        bool edited = false;
        if (BeginCombo(label, current ? current->Value.c_str() : "None")) {
            if (Selectable("None", v == state::Null)) {
                v = state::Null;
                edited = true;
            }
            for (const auto [e, name] : r.view<const Name>().each()) {
                PushID(int(state::Integral(e)));
                if (Selectable(name.Value.c_str(), e == v)) {
                    v = e;
                    edited = true;
                }
                PopID();
            }
            EndCombo();
        }
        changed |= edited;
        finished |= edited;
    } else if constexpr (ui::DrawableField<T>) {
        ui::NoteGesture(ui::DrawField(label, v, spec), changed, finished);
    } else if constexpr (std::same_as<T, std::string>) {
        ui::NoteGesture(InputText(label, &v), changed, finished);
    } else if constexpr (std::same_as<T, std::filesystem::path>) {
        auto text = v.string();
        const bool edited = InputText(label, &text);
        if (edited) v = text;
        ui::NoteGesture(edited, changed, finished);
    } else if constexpr (Optional<T>) {
        PushID(label);
        bool present = v.has_value();
        if (Checkbox(present ? "##present" : label, &present)) {
            if (present) v.emplace();
            else v.reset();
            changed = finished = true;
        }
        if (v) {
            SameLine();
            DrawValue(r, label, *v, spec, changed, finished);
        }
        PopID();
    } else if constexpr (Variant<T>) {
        constexpr size_t N = std::variant_size_v<T>;
        static auto names = []<size_t... Is>(std::index_sequence<Is...>) { return std::array{LeafName<std::variant_alternative_t<Is, T>>()...}; }(std::make_index_sequence<N>{});
        int index = int(v.index());
        PushID(label);
        if (Combo(label, &index, [](void *data, int i) { return static_cast<const std::string *>(data)[i].c_str(); }, names.data(), int(N))) {
            [&]<size_t... Is>(std::index_sequence<Is...>) { (..., (Is == size_t(index) ? void(v.template emplace<Is>()) : void())); }(std::make_index_sequence<N>{});
            changed = finished = true;
        }
        // The alternative draws under its own scope, since it shares the label with the combo.
        PushID(index);
        std::visit([&](auto &alternative) { DrawValue(r, label, alternative, spec, changed, finished); }, v);
        PopID();
        PopID();
    } else if constexpr (UniquePtr<T>) {
        if (v) DrawValue(r, label, *v, spec, changed, finished);
    } else if constexpr (Pair<T>) {
        PushID(label);
        DrawValue(r, std::format("{} First", label).c_str(), v.first, spec, changed, finished);
        DrawValue(r, std::format("{} Second", label).c_str(), v.second, spec, changed, finished);
        PopID();
    } else if constexpr (field::IsArray<T> || Vector<T>) {
        group([&] {
            for (size_t i = 0; i < v.size(); ++i) {
                PushID(int(i));
                DrawValue(r, std::format("{}", i).c_str(), v[i], spec, changed, finished);
                PopID();
            }
            if constexpr (Vector<T>) {
                if (SmallButton("Add")) {
                    v.emplace_back();
                    changed = finished = true;
                }
                if (!v.empty()) {
                    SameLine();
                    if (SmallButton("Remove")) {
                        v.pop_back();
                        changed = finished = true;
                    }
                }
            }
        });
    } else if constexpr (ui::HasEditor<T>) {
        group([&] {
            ui::ValueEdit edit{v, changed, &finished};
            DrawEditor(edit, std::type_identity<T>{});
        });
    } else if constexpr (field::Walkable<T>) {
        group([&] { DrawFields(r, v, changed, finished); });
    } else {
        static_assert(false, "DrawValue: this recorded type has no widget");
    }
}

template<typename T> void DrawFields(const state::Scene &r, T &value, bool &changed, bool &finished) {
    field::ForEach(value, [&]<size_t I>(auto &member, std::integral_constant<size_t, I>) { DrawValue(r, field::Label<T, I>.c_str(), member, Spec<T, field::NameString<T, I>>, changed, finished); });
}

// The node's label names the component and field a write targets, so only its target and value draw here.
template<typename L> void DrawLeaf(const state::Scene &r, L &leaf, bool &changed, bool &finished) {
    if constexpr (action::IsUpdate<L>) {
        Text("Target: %s", TargetName(r, leaf.Target).c_str());
        DrawValue(r, "Value", leaf.Value, action::UpdatedField(leaf.ComponentType, leaf.Offset).Field.Spec, changed, finished);
    } else if constexpr (action::object::IsUpdateMaterial<L>) {
        Text("Material %u", leaf.Index);
        DrawValue(r, "Value", leaf.Value, action::FieldAt<PBRMaterial>(leaf.Offset).Spec, changed, finished);
    } else if constexpr (action::IsPatchFields<L>) {
        [&]<typename C, typename F, size_t N>(action::PatchFields<C, F, N> &patch) {
            for (size_t i = 0; i < N; ++i) {
                const auto &field = action::FieldAt<C>(patch.Offsets[i]);
                PushID(int(i));
                DrawValue(r, field.Path.c_str(), patch.Values[i], field.Spec, changed, finished);
                PopID();
            }
        }(leaf);
    } else if constexpr (ui::HasEditor<L>) {
        ui::ValueEdit edit{leaf, changed, &finished};
        DrawEditor(edit, std::type_identity<L>{});
    } else {
        DrawFields(r, leaf, changed, finished);
    }
}

// A change re-runs the node's actions with the edited values on its parent.
// A release commits them in the node's place.
void DrawNodeEditor(Project &session, int node, bool interactive) {
    auto &draft = session.DraftOf(node);
    size_t leaves = 0;
    for (const auto &recorded_action : draft.RecordedActions) leaves += action::VisitLeaf(recorded_action.Action, []<typename L>(const L &) { return !std::is_empty_v<L>; });
    bool changed = false, finished = false;
    for (size_t i = 0; auto &recorded_action : draft.RecordedActions) {
        PushID(int(i++));
        if (leaves > 1) SeparatorText(SpacedName(Label(recorded_action.Action)).c_str());
        action::VisitLeaf(recorded_action.Action, [&]<typename L>(L &leaf) {
            if constexpr (!std::is_empty_v<L>) DrawLeaf(session.R, leaf, changed, finished);
        });
        PopID();
    }
    if (!interactive) return;
    if (changed) session.RequestRestage();
    if (finished) action::Commit();
}
} // namespace

void HandleHistoryShortcuts(Project &session) {
    if (GetIO().WantTextInput) return;
    auto &history = session.History;
    const int present = history.Present;
    if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_Z, ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_Repeat) && history.CanUndo()) session.RequestNavigate(history.Nodes[present].Parent);
    if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_Z, ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_Repeat) && history.CanRedo()) session.RequestNavigate(history.Nodes[present].Children.back());
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
            Text("Shared node pool: %.2f MiB", double(stats.SharedNodeBytes) / (1 << 20));
            Text("Copied data: %.2f MiB", double(stats.OwnedBytes) / (1 << 20));
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
        Separator();
        if (window.TreeRevision != history.Revision) {
            window.Rows.clear();
            std::vector<int> pending;
            if (!nodes.empty()) pending.push_back(0);
            while (!pending.empty()) {
                const auto id = pending.back();
                pending.pop_back();
                window.Rows.push_back({id, session.Editable(id)});
                for (auto it = nodes[id].Children.rbegin(); it != nodes[id].Children.rend(); ++it) pending.push_back(*it);
            }
            window.TreeRevision = history.Revision;
        }
        // An open edit highlights its node while the present node is its parent.
        // The highlighted node's editor opens below it, and the arrow collapses it.
        const int shown = session.Editing.value_or(present);
        if (window.EditorNode != shown) {
            window.EditorNode = shown;
            window.EditorOpen = true;
        }
        const float x = GetCursorPosX();
        const auto indent = [&](int id) { return float(std::min(nodes[id].Depth, 20)) * 12.f; };
        const auto draw_row = [&](const HistoryRow &row) {
            const int id = row.Node;
            SetCursorPosX(x + indent(id));
            auto flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_NoTreePushOnOpen;
            if (!row.Editable) flags |= ImGuiTreeNodeFlags_Leaf;
            if (id == shown) flags |= ImGuiTreeNodeFlags_Selected;
            SetNextItemOpen(id == shown && window.EditorOpen);
            TreeNodeEx(std::format("{}##{}", nodes[id].Label, id).c_str(), flags);
            if (!interactive) return;
            if (IsItemToggledOpen()) {
                if (id == shown) window.EditorOpen = !window.EditorOpen;
                else session.RequestNavigate(id);
            } else if (IsItemClicked()) {
                session.RequestNavigate(id);
            }
        };
        const auto draw_rows = [&](size_t begin, size_t end) {
            ImGuiListClipper clipper;
            clipper.Begin(int(end - begin), GetFrameHeightWithSpacing());
            while (clipper.Step()) {
                for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) draw_row(window.Rows[begin + size_t(i)]);
            }
        };
        const auto shown_row = size_t(std::ranges::find(window.Rows, shown, &HistoryRow::Node) - window.Rows.begin());
        draw_rows(0, std::min(shown_row, window.Rows.size()));
        if (shown_row < window.Rows.size()) {
            draw_row(window.Rows[shown_row]);
            if (window.EditorOpen && window.Rows[shown_row].Editable) {
                const float editor_indent = indent(shown) + GetTreeNodeToLabelSpacing();
                Indent(editor_indent);
                PushID(shown);
                DrawNodeEditor(session, shown, interactive);
                PopID();
                Unindent(editor_indent);
            }
            draw_rows(shown_row + 1, window.Rows.size());
        }
    }
    End();
    return clear;
}
} // namespace project
