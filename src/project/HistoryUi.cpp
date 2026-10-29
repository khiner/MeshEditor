#include "project/HistoryUi.h"

#include "armature/ArmatureComponents.h"
#include "project/Project.h"
#include "ui/CtrlShortcut.h"
#include "ui/FieldEdit.h"

#include <format>
#include <imgui_stdlib.h>

namespace project {
using namespace ImGui;

namespace {
template<typename T>
concept Optional = requires(T t) { t.has_value(); t.emplace(); };
template<typename T>
concept Variant = requires { std::variant_size<T>::value; }; // NOLINT(modernize-type-traits)
template<typename T>
concept Vector = requires(T t) { t.emplace_back(); t.pop_back(); };
template<typename T>
concept Pair = requires(T t) { t.first; t.second; };
template<typename T>
concept UniquePtr = requires { typename T::deleter_type; };

std::string SpacedName(std::string_view name) {
    std::string out(field::detail::SpacedSize(name), '\0');
    field::detail::SpaceWords(name, out.data());
    return out;
}
std::string TargetName(const state::Registry &r, const action::Target &target) {
    static constexpr const char *Names[]{"Active", "Selected", "Selected Delta", "Viewport"};
    const auto *e = std::get_if<state::Entity>(&target);
    if (!e) return Names[target.index()];
    const auto *name = r.try_get<const Name>(*e);
    return name ? name->Value : std::format("Entity {}", state::Integral(*e));
}
template<typename L> std::string LeafName() { return SpacedName(state::LeafName<L>()); }

template<typename T> void DrawFields(const state::Registry &, T &value, bool &changed, bool &finished);

template<typename T> void DrawValue(const state::Registry &r, const char *label, T &v, const FieldSpec &spec, bool &changed, bool &finished) {
    const auto group = [&](auto &&draw) {
        if (!TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) return;
        draw();
        TreePop();
    };
    if constexpr (std::same_as<T, state::Entity> || state::IsEntityRef<T>) {
        auto &entity = [&]() -> state::Entity & {
            if constexpr (state::IsEntityRef<T>) return v.Value;
            else return v;
        }();
        const auto *current = entity == state::Null ? nullptr : r.try_get<const Name>(entity);
        bool edited = false;
        if (BeginCombo(label, current ? current->Value.c_str() : "None")) {
            if (Selectable("None", entity == state::Null)) {
                entity = state::Null;
                edited = true;
            }
            const auto named = [&] {
                if constexpr (state::IsEntityRef<T>) return r.view<const Name, const typename T::ComponentType>() | std::ranges::to<std::vector>();
                else return r.view<const Name>() | std::ranges::to<std::vector>();
            }();
            ImGuiListClipper clipper;
            clipper.Begin(int(named.size()));
            while (clipper.Step()) {
                for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                    const auto e = named[i];
                    PushID(int(state::Integral(e)));
                    if (Selectable(r.get<const Name>(e).Value.c_str(), e == entity)) {
                        entity = e;
                        edited = true;
                    }
                    PopID();
                }
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
        if (Checkbox("##present", &present)) {
            if (present) v.emplace();
            else v.reset();
            changed = finished = true;
        }
        SameLine();
        if (v) DrawValue(r, label, *v, spec, changed, finished);
        else TextUnformatted(label);
        PopID();
    } else if constexpr (Variant<T>) {
        constexpr size_t N = std::variant_size_v<T>;
        static auto names = []<size_t... Is>(std::index_sequence<Is...>) { return std::array{LeafName<std::variant_alternative_t<Is, T>>()...}; }(std::make_index_sequence<N>{});
        auto index = int(v.index());
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
    } else if constexpr (field::IsArray<T>) {
        group([&] {
            for (size_t i = 0; i < v.size(); ++i) {
                PushID(int(i));
                DrawValue(r, std::format("{}", i).c_str(), v[i], spec, changed, finished);
                PopID();
            }
        });
    } else if constexpr (Vector<T>) {
        // A recorded vector can contain a whole selection, so it opens on request and draws its visible elements.
        if (!TreeNodeEx(label, ImGuiTreeNodeFlags_None, "%s (%zu)", label, v.size())) return;
        ImGuiListClipper clipper;
        clipper.Begin(int(v.size()));
        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                PushID(i);
                DrawValue(r, std::format("{}", i).c_str(), v[i], spec, changed, finished);
                PopID();
            }
        }
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
        TreePop();
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

template<typename T> void DrawFields(const state::Registry &r, T &value, bool &changed, bool &finished) {
    field::ForEach(value, [&]<size_t I>(auto &member, std::integral_constant<size_t, I>) { DrawValue(r, field::ChainLabel<field::At<T, I>{}>.c_str(), member, Spec<T, I>, changed, finished); });
}

// The node's label names the component and field a write targets, so only its target and value draw here.
template<typename L> void DrawLeaf(const state::Registry &r, L &leaf, bool &changed, bool &finished) {
    if constexpr (action::IsUpdate<L>) {
        Text("Target: %s", TargetName(r, leaf.Target).c_str());
        DrawValue(r, "Value", leaf.Value, action::UpdatedField(leaf.ComponentType, leaf.Offset).Field.Spec, changed, finished);
    } else if constexpr (action::object::IsUpdateMaterial<L>) {
        Text("Material %u", leaf.Index);
        DrawValue(r, "Value", leaf.Value, action::FieldAt<PBRMaterial>(leaf.Offset).Spec, changed, finished);
    } else if constexpr (action::IsPatchFields<L>) {
        Text("Target: %s", TargetName(r, leaf.Entity == state::Null ? action::Target{action::OnActive{}} : action::Target{leaf.Entity}).c_str());
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
void DrawNodeEditor(Project &session, uint32_t node, bool interactive) {
    auto &draft = session.DraftOf(node);
    size_t leaves = 0;
    for (const auto &recorded_action : draft.RecordedActions) {
        leaves += action::VisitLeaf(recorded_action, []<typename L>(const L &) { return !std::is_empty_v<L>; });
    }
    bool changed = false, finished = false;
    for (size_t i = 0; auto &recorded_action : draft.RecordedActions) {
        PushID(int(i++));
        if (leaves > 1) SeparatorText(SpacedName(Label(recorded_action)).c_str());
        action::VisitLeaf(recorded_action, [&]<typename L>(L &leaf) {
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
    const auto &history = session.History;
    const auto undo_target = history.UndoTarget();
    const auto redo_target = history.RedoTarget();
    if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiKey_Z, ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_Repeat) && undo_target) session.RequestNavigate(*undo_target);
    if (CtrlShortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_Z, ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_Repeat) && redo_target) session.RequestNavigate(*redo_target);
}

bool DrawHistoryPanel(Project &session, HistoryPanel &panel, bool interactive) {
    if (!panel.Visible) return false;
    bool clear = false;
    if (Begin(panel.Name, &panel.Visible)) {
        auto &history = session.History;
        const auto &nodes = history.Nodes;
        const auto present = history.Present;
        const auto undo_target = history.UndoTarget();
        const auto redo_target = history.RedoTarget();
        BeginDisabled(!undo_target);
        if (Button("Undo") && interactive && undo_target) session.RequestNavigate(*undo_target);
        EndDisabled();
        SameLine();
        BeginDisabled(!redo_target);
        if (Button("Redo") && interactive && redo_target) session.RequestNavigate(*redo_target);
        EndDisabled();
        SameLine();
        if (Button("Memory...") && interactive) OpenPopup("History memory");
        if (BeginPopup("History memory")) {
            const auto stats = history.Stats();
            Text("Shared node pool: %.2f MiB", double(stats.SharedNodeBytes) / (1 << 20));
            Text("Copied data: %.2f MiB", double(stats.OwnedBytes) / (1 << 20));
            Text("%zu states cached, %zu require disk reads", stats.HotNodes, stats.ColdNodes);
            auto cap = int(session.MemoryCap >> 20);
            if (DragInt("Copied data budget (MiB)", &cap, 1.f, 0, 4096) && interactive) {
                session.MemoryCap = uint64_t(std::max(0, cap)) << 20;
                history.Evict(session.MemoryCap);
            }
            Text("The current state and up to %u recent edits stay cached.", store::History::UndoWindow);
            EndPopup();
        }
#ifdef DEBUG_BUILD
        if (Button("Validate history") && interactive) panel.ValidateRequested = true;
        SameLine();
#endif
        clear = Button("Clear history") && interactive;
        ui::Describe("Clear history", "Keep the current and last saved states, and discard other undo and redo states");
        Separator();
        const auto rail_bit = [](uint32_t depth) { return uint32_t{1} << (std::min(depth, 20u) - 1); };
        if (panel.TreeRevision != history.Revision) {
            panel.Rows.clear();
            panel.RowByNode.assign(nodes.size(), std::nullopt);
            std::vector<HistoryRow> pending;
            if (!nodes.empty()) pending.push_back({0, 0});
            while (!pending.empty()) {
                const auto row = pending.back();
                pending.pop_back();
                panel.RowByNode[row.Node] = uint32_t(panel.Rows.size());
                panel.Rows.push_back(row);
                const auto &children = nodes[row.Node].Children;
                for (const auto child : children) {
                    const bool branch = child != children.front();
                    const auto depth = row.Depth + branch;
                    const auto rails = row.Rails | (branch && child != children[1] ? rail_bit(depth) : 0);
                    pending.push_back({child, depth, rails});
                }
            }
            panel.TreeRevision = history.Revision;
        }
        std::vector<uint32_t> lineage;
        for (auto node = present; node; node = nodes[*node].Parent) lineage.push_back(*node);
        std::ranges::reverse(lineage);
        for (auto node = redo_target; node; node = nodes[*node].RedoChild) lineage.push_back(*node);
        const auto active = [&](uint32_t node) { return nodes[node].Depth < lineage.size() && lineage[nodes[node].Depth] == node; };
        struct ActiveRail {
            uint32_t Bit, First, Last;
        };
        std::vector<ActiveRail> active_rails;
        for (const auto node : lineage) {
            const auto parent = nodes[node].Parent;
            if (!parent || node == nodes[*parent].Children.front()) continue;
            const auto first = panel.RowByNode[*parent], last = panel.RowByNode[node];
            if (first && last) active_rails.push_back({rail_bit(panel.Rows[*last].Depth), *first, *last});
        }
        const auto active_mask = [&](size_t row, bool below) {
            auto mask = uint32_t{0};
            for (const auto &rail : active_rails) {
                if (below ? rail.First <= row && row < rail.Last : rail.First < row && row <= rail.Last) mask |= rail.Bit;
            }
            return mask;
        };
        // An open edit highlights its node while the present node is its parent.
        // The highlighted node's editor opens below it, and the arrow collapses it.
        const auto shown = session.Editing ? session.Editing : present;
        if (panel.EditorNode != shown) {
            panel.EditorNode = shown;
            panel.EditorOpen = true;
        }
        const auto x = GetCursorPosX();
        const auto screen_x = GetCursorScreenPos().x;
        const auto indent = [](const HistoryRow &row) { return float(std::min(row.Depth, 20u)) * 12.f; };
        const auto rail_x = [&](uint32_t depth) { return screen_x + float(std::min(depth, 20u)) * 12.f - 6.f; };
        const auto rail_color = GetColorU32(ImGuiCol_TextDisabled);
        const auto active_color = GetColorU32(ImGuiCol_CheckMark);
        const auto draw_rails = [&](uint32_t rails, uint32_t highlighted, float top, float bottom) {
            while (rails) {
                const auto bit = uint32_t{1} << std::countr_zero(rails);
                const auto x = rail_x(std::countr_zero(rails) + 1);
                GetWindowDrawList()->AddLineV(x, top, bottom, highlighted & bit ? active_color : rail_color);
                rails &= rails - 1;
            }
        };
        const auto below_rails = [&](const HistoryRow &row) { return row.Rails | (nodes[row.Node].Children.size() > 1 ? rail_bit(row.Depth + 1) : 0); };
        const auto draw_row = [&](size_t index) {
            const auto &row = panel.Rows[index];
            const auto id = row.Node;
            SetCursorPosX(x + indent(row));
            auto flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_NoTreePushOnOpen;
            if (!nodes[id].Editable) flags |= ImGuiTreeNodeFlags_Leaf;
            if (id == shown) flags |= ImGuiTreeNodeFlags_Selected;
            SetNextItemOpen(id == shown && panel.EditorOpen);
            const auto highlight = active(id) && id != shown;
            if (highlight) PushStyleColor(ImGuiCol_Text, GetStyleColorVec4(ImGuiCol_CheckMark));
            TreeNodeEx(std::format("{}##{}", nodes[id].Label, id).c_str(), flags);
            if (highlight) PopStyleColor();
            const auto min = GetItemRectMin(), max = GetItemRectMax();
            const auto center = (min.y + max.y) * .5f;
            const auto bottom = max.y + GetStyle().ItemSpacing.y;
            const auto above = active_mask(index, false), below = active_mask(index, true);
            draw_rails(row.Rails, above, min.y, center);
            draw_rails(row.Rails, below, center, bottom);
            const auto parent = nodes[id].Parent;
            if (parent && id != nodes[*parent].Children.front()) {
                const auto branch_x = rail_x(row.Depth);
                if (!(row.Rails & rail_bit(row.Depth))) draw_rails(rail_bit(row.Depth), above, min.y, center);
                GetWindowDrawList()->AddLineH(branch_x, min.x - 2.f, center, active(id) ? active_color : rail_color);
            }
            draw_rails(below_rails(row) & ~row.Rails, below, max.y, bottom);
            if (!interactive) return;
            if (IsItemToggledOpen()) {
                if (id == shown) panel.EditorOpen = !panel.EditorOpen;
                else session.RequestNavigate(id);
            } else if (IsItemClicked()) {
                session.RequestNavigate(id);
            }
        };
        const auto draw_rows = [&](size_t begin, size_t end) {
            ImGuiListClipper clipper;
            clipper.Begin(int(end - begin), GetFrameHeightWithSpacing());
            while (clipper.Step()) {
                for (auto i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) draw_row(begin + size_t(i));
            }
        };
        const auto shown_row = shown && panel.RowByNode[*shown] ? size_t(*panel.RowByNode[*shown]) : panel.Rows.size();
        draw_rows(0, std::min(shown_row, panel.Rows.size()));
        if (shown_row < panel.Rows.size()) {
            draw_row(shown_row);
            if (panel.EditorOpen && nodes[*shown].Editable) {
                const auto top = GetCursorScreenPos().y;
                const auto editor_indent = indent(panel.Rows[shown_row]) + GetTreeNodeToLabelSpacing();
                Indent(editor_indent);
                PushID(int(*shown));
                DrawNodeEditor(session, *shown, interactive);
                PopID();
                Unindent(editor_indent);
                draw_rails(below_rails(panel.Rows[shown_row]), active_mask(shown_row, true), top, GetCursorScreenPos().y);
            }
            draw_rows(shown_row + 1, panel.Rows.size());
        }
    }
    End();
    return clear;
}
} // namespace project
