#include "FileDialog.h"
#include "SortUnique.h"
#include "action/Dispatch.h"
#include "action/Io.h"
#include "action/Physics.h"
#include "action/Scene.h"
#include "action/Selection.h"
#include "gltf/SourceAssets.h"
#include "graph/NodeEditor.h"
#include "graph/NodeView.h"
#include "physics/PhysicsUi.h"
#include "render/MaterialComponents.h"
#include "render/ObjectGeometry.h"
#include "render/Textures.h"
#include "scene/SceneControlsUi.h"
#include "ui/MaterialEdit.h"

#include <imgui_stdlib.h>
#include <numeric>

namespace graph {
using namespace ImGui;

state::Entity EditorGraph(const state::Registry &r, const NodesPanel &panel) {
    const auto graph = PageGraph(panel.Canvas);
    if (graph == state::Document || r.valid(graph)) return graph;
    if (graph != state::Null) {
        for (const auto &location : panel.Canvas.Locations | std::views::reverse) {
            if (location.Graph == state::Document || r.valid(location.Graph)) return location.Graph;
        }
    }
    return FrameScene(r);
}

void EnterGraph(NodesPanel &panel, state::Entity graph, state::Entity object, std::vector<state::Entity> path) {
    auto &canvas = panel.Canvas;
    auto at = std::ranges::find(canvas.Locations, PageGraph(canvas), &GraphCanvas::Graph);
    if (at != canvas.Locations.end()) canvas.Locations.erase(at);
    canvas.Locations.push_back({PageGraph(canvas), canvas.Offset, canvas.Zoom});
    canvas.Root = graph;
    canvas.Object = object;
    canvas.Path = std::move(path);
    canvas.Inspected = state::Null;
    at = std::ranges::find(canvas.Locations, PageGraph(canvas), &GraphCanvas::Graph);
    canvas.Offset = at == canvas.Locations.end() ? vec2{} : at->Offset;
    canvas.Zoom = at == canvas.Locations.end() ? 1.f : at->Zoom;
    panel.Drag = NodeDrag::None;
    panel.Expanded = UINT32_MAX;
    panel.LinkingAdd.reset();
}

void DrawGraphNavigation(const state::Registry &r, NodesPanel &panel) {
    const auto graph = EditorGraph(r, panel);
    const auto &subgraphs = panel.Canvas.Path;
    const bool holds_items = r.all_of<SubgraphItems>(graph);
    // The graph whose page this is, or that holds the first subgraph node of the canvas path.
    const auto root = subgraphs.empty() ? graph : panel.Canvas.Root;
    if (Button("Document")) EnterGraph(panel, state::Document);
    // A user subgraph and a material are document items, so a page of their graph follows the document.
    if (!r.all_of<MaterialGraph>(root) && !r.all_of<Subgraph, GraphNodes>(root)) {
        SameLine();
        if (Button(SceneLabel(r, FrameScene(r)).c_str())) EnterGraph(panel, state::Null);
    }
    if (holds_items || !subgraphs.empty() || r.all_of<GraphNodes>(graph) || r.any_of<ArmatureObject, Instance>(graph)) {
        const auto owner = holds_items ? GraphRoot(r, graph) : root;
        const auto object = holds_items || r.all_of<Subgraph>(owner) ? state::Null : panel.Canvas.Object;
        const auto button = [&](state::Entity e, std::string label, state::Entity entered, std::vector<state::Entity> path) {
            SameLine();
            PushID(int(state::Integral(e)));
            if (Button(label.c_str())) EnterGraph(panel, entered, object, std::move(path));
            PopID();
        };
        // A material holds its name in its record.
        if (!holds_items || (owner != state::Document && owner != FrameScene(r))) button(owner, r.all_of<MaterialGraph>(owner) ? ItemName(r, owner) : GetName(r, object != state::Null ? object : owner), owner, {});
        // A subgraph node's page is reached through the subgraph nodes of the canvas path, from the page of the graph that holds the first.
        for (size_t i = 0; i < subgraphs.size(); ++i) button(subgraphs[i], SubgraphNodeName(r, subgraphs[i]), owner, {subgraphs.begin(), subgraphs.begin() + i + 1});
        if (holds_items) {
            std::vector<state::Entity> path;
            for (auto e = graph; e != owner && e != state::Null; e = r.get<const SubgraphItems>(e).ParentEntity) path.push_back(e);
            for (const auto e : path | std::views::reverse) button(e, GetName(r, e), e, {});
        }
    }
}

const char *ItemLabel(const state::Registry &r, state::Entity entity) {
    if (entity == state::Document) return "Document";
    if (r.all_of<Material>(entity)) return "Material";
    if (r.all_of<MaterializedTexture>(entity)) return "Texture";
    if (r.all_of<Subgraph>(entity)) return "Subgraph";
    if (r.all_of<Scene>(entity)) return "Scene";
    if (r.all_of<Instance>(entity)) return "Instance";
    if (r.all_of<PhysicsMaterial>(entity)) return "Physics material";
    if (r.all_of<CollisionFilter>(entity)) return "Collision filter";
    if (r.all_of<CollisionSystem>(entity)) return "Collision system";
    if (r.all_of<JointDescription>(entity)) return "Joint description";
    if (r.all_of<BoneIndex>(entity)) return "Bone";
    if (const auto *object = r.try_get<const ObjectKind>(entity)) {
        constexpr const char *names[]{"Empty", "Mesh", "Armature", "Camera", "Light"};
        return names[uint8_t(object->Value)];
    }
    return "Object";
}

namespace {
// The graph whose page shows the item's node.
state::Entity ItemGraph(const state::Registry &r, state::Entity target) {
    if (const auto *subgraph = r.try_get<const SubgraphItems>(target)) return subgraph->ParentEntity;
    auto graph = state::Document;
    if (r.all_of<BoneIndex>(target)) {
        graph = FindArmatureObject(r, target);
    } else if (r.all_of<ObjectKind>(target)) {
        graph = FrameScene(r);
        if (!InScene(r, target, graph)) {
            for (const auto scene : SortedEntities(r.view<const Scene>()))
                if (InScene(r, target, scene)) {
                    graph = scene;
                    break;
                }
        }
    }
    if (const auto *copy = r.try_get<const CopyOf>(target)) graph = copy->ObjectEntity;
    const auto parent = ParentOrNull(r, target);
    if (const auto *drawn = r.try_get<const ObjectGeometry>(parent); drawn && r.all_of<GraphInstances>(drawn->Entity) && std::ranges::contains(r.get<const GraphInstances>(drawn->Entity).Entities, target)) graph = parent;
    for (;;) {
        const auto representative = GraphRepresentative(r, graph, target);
        if (representative == target || !r.all_of<SubgraphItems>(representative)) break;
        graph = representative;
    }
    return graph;
}
// Selects the item and shows its page, or opens a geometry's graph.
void Reveal(const state::Registry &r, NodesPanel &panel, state::Entity target) {
    if (r.all_of<GeometryGraph>(target)) return EnterGraph(panel, target);
    const bool bone = r.all_of<BoneIndex>(target);
    if (const auto graph = ItemGraph(r, target); EditorGraph(r, panel) != graph) EnterGraph(panel, graph, bone ? FindArmatureObject(r, target) : state::Null);
    if (bone) action::Emit(action::selection::SelectBone{target});
    else action::Emit(action::selection::Select{target});
}
} // namespace

void FollowActiveNode(const state::Registry &r, NodesPanel &panel) {
    const auto active = FindActiveNode(r);
    // The first frame keeps the page the workspace restored.
    const auto followed = std::exchange(panel.Followed, active);
    if (!followed || active == *followed || active == state::Null) return;
    const auto shown = EditorGraph(r, panel);
    // A geometry page turns to the newly active object's geometry.
    if (!panel.Canvas.Path.empty() || r.all_of<GeometryGraph>(shown)) {
        const auto mesh = GetMeshEntity(r, active);
        if (mesh == (panel.Canvas.Path.empty() ? shown : panel.Canvas.Root)) {
            panel.Canvas.Object = active;
            return;
        }
        if (r.all_of<GeometryGraph>(mesh)) return EnterGraph(panel, mesh, active);
    }
    // The page already shows the node, or the node owns the page.
    if (shown == active || ShowsItem(r, shown, active)) return;
    EnterGraph(panel, ItemGraph(r, active));
}

namespace {
// Draws the editor of the field a port promotes, on the entity that holds it: the subgraph's item, or an instance's copy of it.
void DrawPromotedField(const state::Registry &r, const SubgraphPort &port, state::Entity entity, bool interactive) {
    const auto &field = std::get<PromotedField>(port.Binding);
    auto update = action::ReadUpdate(r, entity, field.Component, field.Offset);
    if (!update) return TextDisabled("%s (missing field)", port.Name.c_str());
    const auto &range = action::UpdatedField(field.Component, field.Offset).Field;
    std::visit([&]<typename A>(A &value) {
        if constexpr (action::IsUpdate<A>) {
            if constexpr (std::unsigned_integral<decltype(value.Value)>) {
                if (!range.Labels.empty()) return ui::Gesture(ui::DrawChoice(port.Name.c_str(), value.Value, range.Values, range.Labels) && interactive, [&] { return value; });
            }
            if constexpr (ui::DrawableField<decltype(value.Value)>) ui::Gesture(ui::DrawField(port.Name.c_str(), value.Value, range.Spec) && interactive, [&] { return value; });
        }
    },
               *update);
}
// Draws a scene subgraph's ports.
void DrawSubgraphOfItems(state::Registry &r, state::Entity subgraph, bool interactive) {
    const auto parent = r.get<const SubgraphItems>(subgraph).ParentEntity;
    if (Button("Expand") && interactive) action::Emit(action::nodes::Expand{subgraph});
    ui::Describe("Expand", "Replace this subgraph with its items.");
    SameLine();
    if (Button("Expand all levels") && interactive) action::Emit(action::nodes::Expand{subgraph, true});
    ui::Describe("Expand all levels", "Also replace each instance among the items with copies of its subgraph's items, at every depth.");
    if (IsSceneSubgraph(r, subgraph)) {
        SameLine();
        if (Button("Instance") && interactive) action::Emit(action::object::Instantiate{subgraph});
        ui::Describe("Instance", "Add an object that shows a copy of this subgraph at its own transform.");
    }
    std::optional<uint16_t> previous;
    for (const auto &port : r.get<const SubgraphInterface>(subgraph).Ports) {
        PushID(int(port.Id));
        SeparatorText("Input");
        auto name = port.Name;
        ui::Gesture(InputText("Port name", &name) && interactive, [&] { return action::nodes::RenamePort{subgraph, port.Id, name}; });
        const auto &field = std::get<PromotedField>(port.Binding);
        DrawPromotedField(r, port, field.Entity, interactive);
        if (r.all_of<SubgraphItems>(parent) && Button("Expose on parent") && interactive) action::Emit(action::nodes::ExposePort{parent, field, std::make_unique<std::string>(port.Name)});
        if (previous) {
            if (Button("Move up") && interactive) action::Emit(action::nodes::MovePort{subgraph, port.Id, *previous});
            SameLine();
        }
        previous = port.Id;
        if (Button("Remove port") && interactive) action::Emit(action::nodes::RemovePort{subgraph, port.Id});
        PopID();
    }
}
} // namespace

UseCounts CountSubgraphUses(const state::Registry &r) {
    UseCounts uses;
    std::vector<state::Entity> users;
    for (const auto [_, node] : r.view<const SubgraphNode>().each()) {
        if (const auto *builtin = std::get_if<BuiltinSubgraphId>(&node.SubgraphId)) ++uses.ByBuiltinSubgraph[size_t(*builtin)];
        else if (const auto subgraph = UserSubgraph(node.SubgraphId); subgraph != state::Null) users.push_back(subgraph);
    }
    // An instance uses its scene subgraph.
    for (const auto [_, instance] : r.view<const Instance>().each()) users.push_back(instance.SubgraphEntity);
    uses.Used = users;
    std::erase(uses.Used, state::Null);
    SortUnique(uses.Used);
    std::ranges::sort(users);
    for (const auto subgraph : users) {
        if (uses.User.empty() || uses.User.back().first != subgraph) uses.User.emplace_back(subgraph, 0u);
        ++uses.User.back().second;
    }
    return uses;
}
bool SubgraphUsed(const UseCounts &uses, state::Entity subgraph) { return std::ranges::binary_search(uses.Used, subgraph); }
uint32_t UseCount(const UseCounts &uses, SubgraphId id) {
    if (const auto *builtin = std::get_if<BuiltinSubgraphId>(&id)) return uses.ByBuiltinSubgraph[size_t(*builtin)];
    const auto subgraph = std::get<state::Entity>(id);
    const auto at = std::ranges::lower_bound(uses.User, subgraph, {}, &std::pair<state::Entity, uint32_t>::first);
    return at != uses.User.end() && at->first == subgraph ? at->second : 0u;
}

void DrawAnimationTargets(const state::Registry &r, NodesPanel &panel, uint32_t animation) {
    for (const auto entity : SortedEntities(r.view<const AnimationClips>())) {
        const auto *clip = r.get<const AnimationClips>(entity).Find(animation);
        if (!clip) continue;
        PushID(int(state::Integral(entity)));
        if (entity != state::Document && Button(ItemName(r, entity).c_str())) Reveal(r, panel, entity);
        for (size_t i = 0; i < clip->Channels.size(); ++i) {
            const auto &channel = clip->Channels[i];
            PushID(int(i));
            Text("%s (%zu keys)", animation::FieldLabel(channel.Target).c_str(), channel.Times.size());
            PopID();
        }
        PopID();
    }
}

namespace {
// Lists the references that hold the item, each as a button that reveals its owner.
void DrawUsers(const state::Registry &r, NodesPanel &panel, state::Entity item) {
    if (panel.UsersOf != item || panel.UsersRevision != r.Resources.get<const GraphChanges>().Revision) {
        panel.UsersOf = item;
        panel.UsersRevision = r.Resources.get<const GraphChanges>().Revision;
        panel.Users = ReferencesTo(r, item);
    }
    SeparatorText("Used by");
    if (panel.Users.empty()) TextDisabled("Nothing");
    for (uint32_t i = 0; i < panel.Users.size(); ++i) {
        PushID(int(i));
        if (Button(ItemName(r, panel.Users[i].Owner).c_str())) Reveal(r, panel, panel.Users[i].Owner);
        SameLine();
        TextUnformatted(ReferenceLabel(r, panel.Users[i]).c_str());
        PopID();
    }
}

// Draws an instance's subgraph, and the field each port promotes as the instance's copy holds it.
// An edit of such a field sets the instance's own value of the port.
void DrawInstance(state::Registry &r, NodesPanel &panel, state::Entity instance) {
    const auto &held = r.get<const Instance>(instance);
    const auto subgraph = held.SubgraphEntity;
    Text("Instance of %s", GetName(r, subgraph).c_str());
    if (Button("Reveal subgraph")) Reveal(r, panel, subgraph);
    SameLine();
    if (Button("Make unique")) action::Emit(action::nodes::MakeUnique{instance});
    ui::Describe("Make unique", "Make this instance's copies the items of a new subgraph, which edits then change alone.");
    if (Button("Expand")) action::Emit(action::nodes::Expand{instance});
    ui::Describe("Expand", "Replace this instance with copies of its subgraph's items.");
    SameLine();
    if (Button("Expand all levels")) action::Emit(action::nodes::Expand{instance, true});
    ui::Describe("Expand all levels", "Also replace each instance among the copies, at every depth.");
    for (const auto &port : r.get<const SubgraphInterface>(subgraph).Ports) {
        PushID(int(port.Id));
        DrawPromotedField(r, port, CopyFor(held.Copies, std::get<PromotedField>(port.Binding).Entity), true);
        if (PortValueOf(held, port.Id)) {
            SameLine();
            if (SmallButton("Reset")) action::Emit(action::nodes::ClearPortValue{instance, port.Id});
            ui::Describe("Reset", "Read the subgraph's value in place of this instance's own.");
        }
        PopID();
    }
}

// Draws the node's sections, which its components decide.
void DrawNodeProperties(state::Registry &r, NodesPanel &panel, state::Entity entity) {
    if (entity == state::Document) {
        RenderDocumentControls(r);
    } else if (r.all_of<Material>(entity)) {
        if (r.all_of<MaterialGraph>(entity)) {
            if (Button("Open graph")) EnterGraph(panel, entity);
            ui::Describe("Open graph", "Show the material's node graph in the node editor.");
            SameLine();
            if (Button("Export MaterialX")) {
                FileDialog::ShowSave("mtlx", r.get<const Material>(entity).Name + ".mtlx", [entity](const std::filesystem::path &path) { action::Emit(action::io::SaveMaterialX{entity, path}); });
            }
            ui::Describe("Export MaterialX", "Write the material and its node graph as a MaterialX document.");
        } else {
            if (Button("Add node graph")) {
                action::Emit(action::nodes::AddMaterialGraph{entity});
                EnterGraph(panel, entity);
            }
            ui::Describe("Add node graph", "Compute the material's inputs with a node graph. Unlinked inputs, alpha and transmission keep the values below.");
        }
        DrawMaterialPreview(r, panel, entity, {});
        ui::DrawFields<PBRMaterial>(ui::MaterialEdit{r, entity});
        DrawUsers(r, panel, entity);
    } else if (const auto *texture = r.try_get<const MaterializedTexture>(entity)) {
        if (const auto *source = r.try_get<const gltf::SourceAssets>(state::Document); source && texture->SourceImageIndex < source->Images.size()) {
            const auto &image = source->Images[texture->SourceImageIndex];
            const auto &path = image.SourcePath.empty() ? image.Uri : image.SourcePath;
            if (!path.empty()) TextWrapped("Source: %s", path.c_str());
            else TextUnformatted("Embedded image");
        }
        Text("Color space: %s", texture->Params.ColorSpace == TextureColorSpace::Srgb ? "sRGB" : "Linear");
        DrawUsers(r, panel, entity);
    } else if (r.all_of<SubgraphItems>(entity)) {
        DrawSubgraphOfItems(r, entity, true);
        if (IsSceneSubgraph(r, entity)) DrawUsers(r, panel, entity);
    } else if (r.all_of<Subgraph>(entity)) {
        TextWrapped("%s", UsesText(UseCount(UsesOf(r, panel), entity)).c_str());
        if (Button("Open")) EnterGraph(panel, entity);
        ui::Describe("Open", "Show the subgraph in the node editor.");
        SameLine();
        BeginDisabled(SubgraphUsed(UsesOf(r, panel), entity));
        if (Button("Delete")) action::Emit(action::nodes::DeleteSubgraph{entity});
        EndDisabled();
        ui::Describe("Delete", "Delete this subgraph and its nodes.");
        DrawSubgraphInterface(r, {&r, entity}, true);
    } else if (physics_ui::RenderResourceProperties(r, entity)) {
        DrawUsers(r, panel, entity);
    } else if (r.all_of<Scene>(entity)) {
        if (r.all_of<ImageLight>(entity)) ui::Edit{r, entity}.Draw<&ImageLight::Intensity>();
    } else {
        // An object draws its sections after the buttons that open its source and generated instances in the node editor.
        if (const auto *copy = r.try_get<const CopyOf>(entity)) {
            Text("Copy of %s", GetName(r, copy->SourceEntity).c_str());
            if (Button("Reveal source")) Reveal(r, panel, copy->SourceEntity);
            ui::Describe("Reveal source", "Show the item this copy stands for.");
            // The fields the instance's subgraph has a port for hold the instance's own value, and an edit of any other field edits the source.
            std::string own;
            for (const auto &port : r.get<const SubgraphInterface>(r.get<const Instance>(copy->ObjectEntity).SubgraphEntity).Ports) {
                if (const auto *field = std::get_if<PromotedField>(&port.Binding); field && field->Entity == copy->SourceEntity) own += (own.empty() ? "" : ", ") + port.Name;
            }
            if (own.empty()) TextWrapped("An edit changes %s in every instance.", GetName(r, copy->SourceEntity).c_str());
            else TextWrapped("This instance's own: %s. Any other edit changes %s in every instance.", own.c_str(), GetName(r, copy->SourceEntity).c_str());
        }
        if (r.all_of<Instance>(entity)) DrawInstance(r, panel, entity);
        if (const auto *object = r.try_get<const ObjectGeometry>(entity); object && r.all_of<GraphInstances>(object->Entity)) {
            if (Button("Inspect generated instances")) EnterGraph(panel, entity);
        }
        RenderEntityControls(r, entity);
    }
}
} // namespace

void DrawPropertiesPanel(state::Registry &r, NodesPanel &nodes, Panel &panel, bool interactive) {
    // Properties releases an image it drew before the last frame and has not drawn since.
    for (auto *shown : {&nodes.NodeImage, &nodes.Preview}) {
        if (shown->Hold && shown->Frame + 1 < GetFrameCount()) HoldShownImage(r, *shown, {});
    }
    if (!panel.Visible) return;
    if (Begin(panel.Name, &panel.Visible)) {
        RenderInteractionControls(r);
        const auto geometry = DrawNodeParameters(r, nodes, interactive);
        RenderSelectionActions(r);
        if (const auto active = FindActiveNode(r); active != state::Null && (geometry == state::Null || GetMeshEntity(r, active) == geometry)) {
            auto name = r.all_of<Scene>(active) ? SceneLabel(r, active) : ItemName(r, active);
            SeparatorText(name.c_str());
            PushID("Node");
            if (active != state::Document && !r.all_of<SubElementOf>(active)) ui::Gesture(InputText("Name", &name), [&] { return action::nodes::RenameItem{active, name}; });
            DrawNodeProperties(r, nodes, active);
            PopID();
        }
    }
    End();
}
NodeView ProjectItemGraph(const state::Registry &r, state::Entity graph) {
    NodeView view{.Graph = graph, .InputFirst = {0}, .SocketFirst = {0}};
    const auto items = GraphItems(r, graph);
    const auto drawers = DrawersOf(r);
    const auto count = uint32_t(items.size());
    const auto is_subgraph = [&](state::Entity item) { return item != state::Document && r.all_of<SubgraphItems>(item); };
    // Each leaf the page shows, as its own node or folded into the node of the subgraph that holds it.
    std::vector<std::pair<state::Entity, uint32_t>> leaves;
    for (uint32_t n = 0; n < count; ++n) {
        if (!is_subgraph(items[n])) leaves.emplace_back(items[n], n);
    }
    for (const auto [subgraph, folded] : r.view<const SubgraphItems>().each()) {
        auto child = subgraph;
        while (child != state::Null && r.get<const SubgraphItems>(child).ParentEntity != graph) {
            const auto parent = r.get<const SubgraphItems>(child).ParentEntity;
            child = r.all_of<SubgraphItems>(parent) ? parent : state::Null;
        }
        const auto at = std::ranges::lower_bound(items, child);
        if (child == state::Null || at == items.end() || *at != child) continue;
        for (const auto item : folded.Items) {
            if (!is_subgraph(item)) leaves.emplace_back(item, uint32_t(at - items.begin()));
        }
    }
    std::ranges::sort(leaves);
    const auto node_of = [&](state::Entity item) {
        const auto at = std::ranges::lower_bound(leaves, item, {}, &decltype(leaves)::value_type::first);
        return at != leaves.end() && at->first == item ? at->second : NoNode;
    };
    // A leaf node shows each of its references.
    // A subgraph node shows a reference of an item it holds only when the other end is another node of this page, as an input for the item's reference or an output for the item.
    struct Input {
        SceneReference Reference;
        state::Entity Value;
        uint32_t From;
    };
    std::vector<std::vector<Input>> inputs(count);
    std::vector<std::vector<state::Entity>> outputs(count);
    std::vector<ItemReference> shown;
    for (const auto [leaf, node] : leaves) {
        ShownReferences(r, leaf, drawers, shown);
        for (const auto &[kind, index, value] : shown) {
            const SceneReference reference{leaf, kind, index};
            const auto source = value == state::Null ? NoNode : node_of(value);
            const auto from = source == node ? NoNode : source;
            if (items[node] != leaf && from == NoNode) continue;
            if (from != NoNode && items[from] != value && !std::ranges::contains(outputs[from], value)) outputs[from].push_back(value);
            inputs[node].push_back({reference, value, from});
        }
    }
    const auto socket = [&](std::string label, SocketDirection direction, PortBinding binding) {
        view.Labels.push_back(std::move(label));
        view.Sockets.push_back({TypeBit(DataType::Reference), direction});
        view.ShownTypes.push_back(TypeBit(DataType::Reference));
        view.Bindings.push_back(std::move(binding));
    };
    // A subgraph's node shows its uses.
    const auto uses = graph == state::Document ? CountSubgraphUses(r) : UseCounts{};
    for (uint32_t n = 0; n < count; ++n) {
        const auto *stored = items[n] != state::Document ? r.try_get<const ItemNode>(items[n]) : nullptr;
        auto title = stored ? stored->Title : ItemName(r, items[n]);
        if (items[n] != state::Document && r.all_of<Subgraph, GraphNodes>(items[n])) {
            const auto count = UseCount(uses, items[n]);
            title += count == 1u ? " (1 use)" : std::format(" ({} uses)", count);
        }
        view.Nodes.push_back({.Item = items[n], .Title = std::move(title)});
        for (const auto &input : inputs[n]) {
            auto label = ReferenceLabel(r, input.Reference);
            if (input.Reference.Owner != items[n]) label = ItemName(r, input.Reference.Owner) + ": " + label;
            // A reference to an item on another page reads as the item's name.
            if (input.From == NoNode && input.Value != state::Null) label += ": " + ItemName(r, input.Value);
            socket(std::move(label), SocketDirection::In, input.Reference);
            // A wire leaves a leaf's one output, or the output of a subgraph that stands for the value.
            if (input.From == NoNode) view.Wires.push_back({});
            else view.Wires.push_back({input.From, uint16_t(inputs[input.From].size() + (std::ranges::find(outputs[input.From], input.Value) - outputs[input.From].begin()))});
        }
        view.InputFirst.push_back(view.InputFirst.back() + uint32_t(inputs[n].size()));
        // A leaf's one output stands for the item and has no label.
        if (!is_subgraph(items[n])) socket({}, SocketDirection::Out, items[n]);
        for (const auto item : outputs[n]) socket(ItemName(r, item), SocketDirection::Out, item);
        view.SocketFirst.push_back(uint32_t(view.Sockets.size()));
    }
    // The nodes are in item order.
    view.ByItem.resize(count);
    std::ranges::iota(view.ByItem, 0u);
    return view;
}

void DrawSceneAddMenu(const state::Registry &r, state::Entity graph) {
    const auto root = GraphRoot(r, graph);
    const auto container = r.all_of<SubgraphItems>(graph) ? graph : state::Null;
    if (graph != state::Document && !r.all_of<Scene>(root)) return;
    if (root == state::Document) {
        if (MenuItem("Scene")) action::Emit(action::scene::Add{"Scene"});
        if (MenuItem("Material")) action::Emit(action::object::AddMaterial{});
        // A color texture is decoded from sRGB, and a data texture, such as a normal or roughness map, is read as stored.
        for (const bool color : {true, false}) {
            if (MenuItem(color ? "Color texture" : "Data texture")) FileDialog::ShowOpen("png;jpg;jpeg;webp;ktx2", [color](const std::filesystem::path &path) { action::Emit(action::object::AddTexture{path, color}); });
        }
        if (MenuItem("Physics material")) action::Emit(action::physics::AddPhysicsMaterial{});
        if (MenuItem("Collision system")) action::Emit(action::physics::AddCollisionSystem{});
        if (MenuItem("Collision filter")) action::Emit(action::physics::AddCollisionFilter{});
        if (MenuItem("Joint description")) action::Emit(action::physics::AddJointDescription{});
    } else {
        for (const auto &[label, primitive] : primitive::Presets) {
            if (MenuItem(label)) action::Emit(action::object::AddMeshPrimitive{primitive, std::make_unique<MeshObjectCreateInfo>(MeshObjectCreateInfo{.Scene = root, .BeginTranslate = false, .Graph = container})});
        }
        Separator();
        const auto info = [&] { return std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Scene = root, .BeginTranslate = false, .Graph = container}); };
        if (BeginMenu("Subgraphs")) {
            for (const auto subgraph : SortedEntities(r.view<const SubgraphItems>())) {
                if (!IsSceneSubgraph(r, subgraph)) continue;
                PushID(int(state::Integral(subgraph)));
                if (MenuItem(GetName(r, subgraph).c_str())) action::Emit(action::object::Instantiate{subgraph});
                PopID();
            }
            EndMenu();
        }
        if (MenuItem("Empty")) action::Emit(action::object::AddEmpty{info()});
        if (MenuItem("Armature")) action::Emit(action::object::AddArmature{info()});
        if (MenuItem("Camera")) action::Emit(action::object::AddCamera{info()});
        if (MenuItem("Light")) action::Emit(action::object::AddLight{info()});
    }
}

bool EnterNode(const state::Registry &r, NodesPanel &panel, state::Entity item) {
    if (item == state::Document) return false;
    if (r.all_of<SubgraphNode, GraphNode>(item)) {
        auto path = panel.Canvas.Path;
        path.push_back(item);
        EnterGraph(panel, panel.Canvas.Path.empty() ? GraphOf(r, item) : panel.Canvas.Root, panel.Canvas.Object, std::move(path));
    } else if (r.all_of<GraphNodes>(item) || r.any_of<Scene, ArmatureObject, SubgraphItems, Instance>(item)) {
        EnterGraph(panel, item, state::Null);
    } else if (const auto mesh = GetMeshEntity(r, item); r.all_of<GeometryGraph>(mesh)) {
        EnterGraph(panel, mesh, item);
    } else return false;
    return true;
}

void DrawSceneNodeMenu(const state::Registry &r, NodesPanel &panel, state::Entity entity) {
    const auto graph = EditorGraph(r, panel);
    // The page's selected nodes, sorted.
    auto selection = GraphItems(r, graph);
    std::erase_if(selection, [&](auto e) { return !r.any_of<Selected, BoneSelection>(e); });
    if (entity != state::Document && r.all_of<SubgraphItems>(entity)) {
        if (MenuItem("Enter")) EnterGraph(panel, entity);
        const bool can_move = !selection.empty() && !std::ranges::contains(selection, entity);
        if (MenuItem("Move selected into subgraph", nullptr, false, can_move)) action::Emit(action::nodes::MoveItems{graph, entity, selection});
        if (MenuItem("Expand")) action::Emit(action::nodes::Expand{entity});
        if (MenuItem("Expand all levels")) action::Emit(action::nodes::Expand{entity, true});
        if (MenuItem("Instance", nullptr, false, IsSceneSubgraph(r, entity))) action::Emit(action::object::Instantiate{entity});
    }
    if (entity != state::Document && r.all_of<Instance>(entity)) {
        if (MenuItem("Enter")) EnterGraph(panel, entity);
        if (MenuItem("Make unique")) action::Emit(action::nodes::MakeUnique{entity});
        if (MenuItem("Expand")) action::Emit(action::nodes::Expand{entity});
        if (MenuItem("Expand all levels")) action::Emit(action::nodes::Expand{entity, true});
    }

    if (!selection.empty()) {
        if (MenuItem("Collapse into subgraph")) action::Emit(action::nodes::Collapse{graph, selection});
        if (const auto *subgraph = r.try_get<const SubgraphItems>(graph)) {
            if (MenuItem("Move selected to parent")) action::Emit(action::nodes::MoveItems{graph, subgraph->ParentEntity, selection});
        }
    }
}

void DrawReferenceMenu(const state::Registry &r, NodesPanel &panel, SceneReference reference) {
    const auto value = ReferenceValue(r, reference);
    if (BeginMenu("Assign")) {
        const auto candidates = ReferenceCandidates(r, reference);
        for (uint32_t i = 0; i < candidates.size(); ++i) {
            PushID(int(i));
            if (MenuItem(ItemName(r, candidates[i]).c_str())) action::Emit(action::nodes::AssignReference{reference, candidates[i]});
            PopID();
        }
        EndMenu();
    }
    if (MenuItem("Clear", nullptr, false, value != state::Null)) action::Emit(action::nodes::AssignReference{reference, state::Null});
    if (value != state::Null && MenuItem(("Go to " + ItemName(r, value)).c_str())) Reveal(r, panel, value);
}
} // namespace graph
