#include "FileDialog.h"
#include "SortUnique.h"
#include "action/Dispatch.h"
#include "action/Io.h"
#include "action/Physics.h"
#include "action/Scene.h"
#include "action/Selection.h"
#include "gltf/SourceAssets.h"
#include "physics/PhysicsUi.h"
#include "procedure/NodeEditor.h"
#include "procedure/NodeView.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/SceneControlsUi.h"
#include "ui/MaterialEdit.h"

#include <imgui_stdlib.h>
#include <numeric>

namespace procedure {
using namespace ImGui;

state::Entity EditorNetwork(const state::Registry &r, const NodesPanel &panel) {
    const auto network = PageNetwork(panel.Canvas);
    if (network == state::Document || r.valid(network)) return network;
    if (network != state::Null) {
        for (const auto &location : panel.Canvas.Locations | std::views::reverse) {
            if (location.Network == state::Document || r.valid(location.Network)) return location.Network;
        }
    }
    return FrameScene(r);
}

void EnterNetwork(NodesPanel &panel, state::Entity network, state::Entity object, std::vector<state::Entity> path) {
    auto &canvas = panel.Canvas;
    auto at = std::ranges::find(canvas.Locations, PageNetwork(canvas), &NetworkCanvas::Network);
    if (at != canvas.Locations.end()) canvas.Locations.erase(at);
    canvas.Locations.push_back({PageNetwork(canvas), canvas.Offset, canvas.Zoom});
    canvas.Root = network;
    canvas.Object = object;
    canvas.Path = std::move(path);
    canvas.Inspected = state::Null;
    at = std::ranges::find(canvas.Locations, PageNetwork(canvas), &NetworkCanvas::Network);
    canvas.Offset = at == canvas.Locations.end() ? vec2{} : at->Offset;
    canvas.Zoom = at == canvas.Locations.end() ? 1.f : at->Zoom;
    panel.Drag = NodeDrag::None;
    panel.Expanded = UINT32_MAX;
    panel.LinkingAdd.reset();
}

void DrawNetworkNavigation(const state::Registry &r, NodesPanel &panel) {
    const auto network = EditorNetwork(r, panel);
    const auto &subnets = panel.Canvas.Path;
    const bool scene_subnet = r.all_of<NetworkSubnet>(network);
    // The network whose page this is, or that holds the first subnet node of the canvas path.
    const auto root = subnets.empty() ? network : panel.Canvas.Root;
    if (Button("Document")) EnterNetwork(panel, state::Document);
    // A user definition and a material are document items, so a page of their network follows the document.
    if (!r.any_of<SubnetDefinition, MaterialGraph>(root)) {
        SameLine();
        if (Button(SceneLabel(r, FrameScene(r)).c_str())) EnterNetwork(panel, state::Null);
    }
    if (scene_subnet || !subnets.empty() || r.all_of<NetworkNodes>(network) || r.any_of<ArmatureObject, HierarchyInstance, HierarchyDefinition>(network)) {
        const auto owner = scene_subnet ? NetworkRoot(r, network) : root;
        const auto object = scene_subnet || r.all_of<SubnetDefinition>(owner) ? state::Null : panel.Canvas.Object;
        const auto button = [&](state::Entity e, std::string label, state::Entity entered, std::vector<state::Entity> path) {
            SameLine();
            PushID(int(state::Integral(e)));
            if (Button(label.c_str())) EnterNetwork(panel, entered, object, std::move(path));
            PopID();
        };
        // A material holds its name in its record.
        if (!scene_subnet || (owner != state::Document && owner != FrameScene(r))) button(owner, r.all_of<MaterialGraph>(owner) ? ItemName(r, owner) : GetName(r, object != state::Null ? object : owner), owner, {});
        // A subnet node's page is reached through the subnet nodes of the canvas path, from the page of the network that holds the first.
        for (size_t i = 0; i < subnets.size(); ++i) button(subnets[i], InstanceName(r, subnets[i]), owner, {subnets.begin(), subnets.begin() + i + 1});
        if (scene_subnet) {
            std::vector<state::Entity> path;
            for (auto e = network; e != owner && e != state::Null; e = r.get<const NetworkSubnet>(e).Parent) path.push_back(e);
            for (const auto e : path | std::views::reverse) button(e, GetName(r, e), e, {});
        }
    }
}

const char *ItemLabel(const state::Registry &r, state::Entity entity) {
    if (entity == state::Document) return "Document";
    if (r.all_of<Material>(entity)) return "Material";
    if (r.all_of<MaterializedTexture>(entity)) return "Texture";
    if (r.all_of<NetworkSubnet>(entity)) return "Subnet";
    if (r.all_of<SubnetDefinition>(entity)) return "Subnet definition";
    if (r.all_of<Scene>(entity)) return "Scene";
    if (r.all_of<HierarchyInstance>(entity)) return "Hierarchy instance";
    if (r.all_of<HierarchyDefinition>(entity)) return "Hierarchy definition";
    if (r.all_of<PhysicsMaterial>(entity)) return "Physics material";
    if (r.all_of<CollisionFilter>(entity)) return "Collision filter";
    if (r.all_of<CollisionSystem>(entity)) return "Collision system";
    if (r.all_of<PhysicsJointDef>(entity)) return "Joint definition";
    if (r.all_of<BoneIndex>(entity)) return "Bone";
    if (const auto *object = r.try_get<const ObjectKind>(entity)) {
        constexpr const char *names[]{"Empty", "Mesh", "Armature", "Camera", "Light"};
        return names[uint8_t(object->Value)];
    }
    return "Object";
}

namespace {
// The network whose page shows the item's node.
state::Entity ItemNetwork(const state::Registry &r, state::Entity target) {
    auto network = state::Document;
    if (r.all_of<BoneIndex>(target)) {
        network = FindArmatureObject(r, target);
    } else if (r.all_of<ObjectKind>(target)) {
        network = FrameScene(r);
        if (!InScene(r, target, network)) {
            for (const auto scene : SortedEntities(r.view<const Scene>()))
                if (InScene(r, target, scene)) {
                    network = scene;
                    break;
                }
        }
    }
    if (const auto *member = r.try_get<const HierarchyMember>(target); member && member->Root != target) network = member->Root;
    const auto parent = ParentOrNull(r, target);
    if (const auto *drawn = r.try_get<const Instance>(parent); drawn && r.all_of<ProcedureInstances>(drawn->Entity) && std::ranges::contains(r.get<const ProcedureInstances>(drawn->Entity).Entities, target)) network = parent;
    for (;;) {
        const auto representative = NetworkRepresentative(r, network, target);
        if (representative == target || !r.all_of<NetworkSubnet>(representative)) break;
        network = representative;
    }
    return network;
}
// Selects the item and shows its page, or opens a geometry's network.
void Reveal(const state::Registry &r, NodesPanel &panel, state::Entity target) {
    if (r.all_of<GeometryProcedure>(target)) return EnterNetwork(panel, target);
    const bool bone = r.all_of<BoneIndex>(target);
    if (const auto network = ItemNetwork(r, target); EditorNetwork(r, panel) != network) EnterNetwork(panel, network, bone ? FindArmatureObject(r, target) : state::Null);
    if (bone) action::Emit(action::selection::SelectBone{target});
    else action::Emit(action::selection::Select{target});
}
} // namespace

void FollowActiveNode(const state::Registry &r, NodesPanel &panel) {
    const auto active = FindActiveNode(r);
    // The first frame keeps the page the workspace restored.
    const auto followed = std::exchange(panel.Followed, active);
    if (!followed || active == *followed || active == state::Null) return;
    const auto shown = EditorNetwork(r, panel);
    // A geometry page turns to the newly active object's geometry.
    if (!panel.Canvas.Path.empty() || r.all_of<GeometryProcedure>(shown)) {
        const auto mesh = GetMeshEntity(r, active);
        if (mesh == (panel.Canvas.Path.empty() ? shown : panel.Canvas.Root)) {
            panel.Canvas.Object = active;
            return;
        }
        if (r.all_of<GeometryProcedure>(mesh)) return EnterNetwork(panel, mesh, active);
    }
    // The page already shows the node, or the node owns the page.
    if (shown == active || ShowsItem(r, shown, active)) return;
    EnterNetwork(panel, ItemNetwork(r, active));
}

namespace {
// Draws a scene subnet's ports.
void DrawSubnetParameters(state::Registry &r, state::Entity subnet, bool interactive) {
    const auto parent = r.get<const NetworkSubnet>(subnet).Parent;
    if (Button("Ungroup") && interactive) action::Emit(action::nodes::UngroupItems{subnet});
    std::optional<uint16_t> previous;
    for (const auto &port : r.get<const SubnetInterface>(subnet).Ports) {
        PushID(int(port.Id));
        SeparatorText("Input");
        auto name = port.Name;
        ui::Gesture(InputText("Port name", &name) && interactive, [&] { return action::nodes::RenamePort{subnet, port.Id, name}; });
        const auto &field = std::get<PromotedField>(port.Binding);
        if (auto update = action::ReadUpdate(r, field.Entity, field.Component, field.Offset)) {
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
        } else TextDisabled("%s (missing field)", port.Name.c_str());
        if (r.all_of<NetworkSubnet>(parent) && Button("Expose on parent") && interactive) action::Emit(action::nodes::ExposePort{parent, field, std::make_unique<std::string>(port.Name)});
        if (previous) {
            if (Button("Move up") && interactive) action::Emit(action::nodes::MovePort{subnet, port.Id, *previous});
            SameLine();
        }
        previous = port.Id;
        if (Button("Remove port") && interactive) action::Emit(action::nodes::RemovePort{subnet, port.Id});
        PopID();
    }
}
} // namespace

UseCounts CountDefinitionUses(const state::Registry &r) {
    UseCounts uses;
    std::vector<state::Entity> users;
    for (const auto [node, subnet] : r.view<const Subnet>().each()) {
        if (subnet.DefinitionEntity != state::Null) uses.Used.push_back(subnet.DefinitionEntity);
        if (r.all_of<HierarchyGeometry>(NetworkOf(r, node))) continue;
        if (subnet.DefinitionEntity != state::Null) users.push_back(subnet.DefinitionEntity);
        else if (const auto macro = std::ranges::find(Macros, subnet.Builtin, &Macro::Name); macro != Macros.end()) ++uses.Builtin[size_t(macro - Macros.begin())];
    }
    SortUnique(uses.Used);
    std::ranges::sort(users);
    for (const auto definition : users) {
        if (uses.User.empty() || uses.User.back().first != definition) uses.User.emplace_back(definition, 0u);
        ++uses.User.back().second;
    }
    return uses;
}
bool DefinitionUsed(const UseCounts &uses, state::Entity definition) { return std::ranges::binary_search(uses.Used, definition); }
uint32_t UseCount(const UseCounts &uses, state::Entity definition, std::string_view builtin) {
    if (definition == state::Null) {
        const auto macro = std::ranges::find(Macros, builtin, &Macro::Name);
        return macro == Macros.end() ? 0u : uses.Builtin[size_t(macro - Macros.begin())];
    }
    const auto at = std::ranges::lower_bound(uses.User, definition, {}, &std::pair<state::Entity, uint32_t>::first);
    return at != uses.User.end() && at->first == definition ? at->second : 0u;
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
    if (panel.UsersOf != item || panel.UsersRevision != r.Resources.get<const NetworkChanges>().Revision) {
        panel.UsersOf = item;
        panel.UsersRevision = r.Resources.get<const NetworkChanges>().Revision;
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

// Draws the node's sections, which its components decide.
void DrawNodeProperties(state::Registry &r, NodesPanel &panel, state::Entity entity) {
    if (entity == state::Document) {
        RenderDocumentControls(r);
    } else if (r.all_of<Material>(entity)) {
        if (r.all_of<MaterialGraph>(entity)) {
            if (Button("Open graph")) EnterNetwork(panel, entity);
            ui::Describe("Open graph", "Show the material's node graph in the node editor.");
            SameLine();
            if (Button("Export MaterialX")) {
                FileDialog::ShowSave("mtlx", r.get<const Material>(entity).Name + ".mtlx", [entity](const std::filesystem::path &path) { action::Emit(action::io::SaveMaterialX{entity, path}); });
            }
            ui::Describe("Export MaterialX", "Write the material and its node graph as a MaterialX document.");
        } else {
            if (Button("Add node graph")) {
                action::Emit(action::nodes::AddMaterialGraph{entity});
                EnterNetwork(panel, entity);
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
    } else if (r.all_of<NetworkSubnet>(entity)) {
        DrawSubnetParameters(r, entity, true);
    } else if (r.all_of<SubnetDefinition>(entity)) {
        TextWrapped("%s", UsesText(UseCount(UsesOf(r, panel), entity)).c_str());
        if (Button("Open")) EnterNetwork(panel, entity);
        ui::Describe("Open", "Show the definition's network in the node editor.");
        SameLine();
        BeginDisabled(DefinitionUsed(UsesOf(r, panel), entity));
        if (Button("Delete")) action::Emit(action::nodes::DeleteDefinition{entity});
        EndDisabled();
        ui::Describe("Delete", "Delete this definition and its nodes.");
        DrawDefinitionInterface(r, {&r, entity}, true);
    } else if (physics_ui::RenderResourceProperties(r, entity)) {
        DrawUsers(r, panel, entity);
    } else if (r.all_of<Scene>(entity)) {
        if (r.all_of<ImageLight>(entity)) ui::Edit{r, entity}.Draw<&ImageLight::Intensity>();
    } else {
        // An object draws its sections after the buttons that open its source and generated instances in the node editor.
        if (const auto *member = r.try_get<const HierarchyMember>(entity)) {
            Text("Source: %s", GetName(r, member->Source).c_str());
            if (Button("Reveal source")) Reveal(r, panel, member->Source);
            if (Button("Make independent")) action::Emit(action::object::MakeIndependent{member->Root});
        }
        if (r.all_of<ObjectKind>(entity) && Button("Instance hierarchy")) action::Emit(action::object::InstantiateHierarchy{entity});
        if (const auto *object = r.try_get<const Instance>(entity); object && r.all_of<ProcedureInstances>(object->Entity)) {
            if (Button("Inspect generated instances")) EnterNetwork(panel, entity);
        }
        RenderEntityControls(r, entity);
        if (r.all_of<HierarchyDefinition>(entity)) DrawUsers(r, panel, entity);
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
NodeView SceneNodeDefinitions(const state::Registry &r, state::Entity network) {
    NodeView view{.Network = network, .InputFirst = {0}, .SocketFirst = {0}};
    const auto items = NetworkItems(r, network);
    const auto drawers = DrawersOf(r);
    const auto count = uint32_t(items.size());
    const auto is_subnet = [&](state::Entity item) { return item != state::Document && r.all_of<NetworkSubnet>(item); };
    // Each leaf the page shows, as its own node or folded into the node of the subnet that holds it.
    std::vector<std::pair<state::Entity, uint32_t>> leaves;
    for (uint32_t n = 0; n < count; ++n) {
        if (!is_subnet(items[n])) leaves.emplace_back(items[n], n);
    }
    for (const auto [subnet, folded] : r.view<const NetworkSubnet>().each()) {
        auto child = subnet;
        while (child != state::Null && r.get<const NetworkSubnet>(child).Parent != network) {
            const auto parent = r.get<const NetworkSubnet>(child).Parent;
            child = r.all_of<NetworkSubnet>(parent) ? parent : state::Null;
        }
        const auto at = std::ranges::lower_bound(items, child);
        if (child == state::Null || at == items.end() || *at != child) continue;
        for (const auto member : folded.Members) {
            if (!is_subnet(member)) leaves.emplace_back(member, uint32_t(at - items.begin()));
        }
    }
    std::ranges::sort(leaves);
    const auto node_of = [&](state::Entity item) {
        const auto at = std::ranges::lower_bound(leaves, item, {}, &decltype(leaves)::value_type::first);
        return at != leaves.end() && at->first == item ? at->second : NoNode;
    };
    // A leaf node shows each of its references.
    // A subnet node shows a reference of an item it holds only when the other end is another node of this page, as an input for the item's reference or an output for the item.
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
    // A definition's node shows its uses.
    const auto uses = network == state::Document ? CountDefinitionUses(r) : UseCounts{};
    for (uint32_t n = 0; n < count; ++n) {
        const auto *stored = items[n] != state::Document ? r.try_get<const NetworkNode>(items[n]) : nullptr;
        auto title = stored ? stored->Title : ItemName(r, items[n]);
        if (items[n] != state::Document && r.all_of<SubnetDefinition>(items[n])) {
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
            // A wire leaves a leaf's one output, or the output of a subnet that stands for the value.
            if (input.From == NoNode) view.Wires.push_back({});
            else view.Wires.push_back({input.From, uint16_t(inputs[input.From].size() + (std::ranges::find(outputs[input.From], input.Value) - outputs[input.From].begin()))});
        }
        view.InputFirst.push_back(view.InputFirst.back() + uint32_t(inputs[n].size()));
        // A leaf's one output stands for the item and has no label.
        if (!is_subnet(items[n])) socket({}, SocketDirection::Out, items[n]);
        for (const auto item : outputs[n]) socket(ItemName(r, item), SocketDirection::Out, item);
        view.SocketFirst.push_back(uint32_t(view.Sockets.size()));
    }
    // The nodes are in item order.
    view.ByItem.resize(count);
    std::ranges::iota(view.ByItem, 0u);
    return view;
}

void DrawSceneAddMenu(const state::Registry &r, state::Entity network) {
    const auto root = NetworkRoot(r, network);
    const auto container = r.all_of<NetworkSubnet>(network) ? network : state::Null;
    if (network != state::Document && !r.all_of<Scene>(root)) return;
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
        if (MenuItem("Joint definition")) action::Emit(action::physics::AddJointDef{});
    } else {
        for (const auto &[label, primitive] : primitive::Presets) {
            if (MenuItem(label)) action::Emit(action::object::AddMeshPrimitive{primitive, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Scene = root, .BeginTranslate = false, .Network = container})});
        }
        Separator();
        const auto info = [&] { return std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Scene = root, .BeginTranslate = false, .Network = container}); };
        if (MenuItem("Empty")) action::Emit(action::object::AddEmpty{info()});
        if (MenuItem("Armature")) action::Emit(action::object::AddArmature{info()});
        if (MenuItem("Camera")) action::Emit(action::object::AddCamera{info()});
        if (MenuItem("Light")) action::Emit(action::object::AddLight{info()});
    }
}

bool EnterNode(const state::Registry &r, NodesPanel &panel, state::Entity item) {
    if (item == state::Document) return false;
    if (r.all_of<Subnet>(item)) {
        auto path = panel.Canvas.Path;
        path.push_back(item);
        EnterNetwork(panel, panel.Canvas.Path.empty() ? NetworkOf(r, item) : panel.Canvas.Root, panel.Canvas.Object, std::move(path));
    } else if (r.all_of<NetworkNodes>(item) || r.any_of<Scene, ArmatureObject, NetworkSubnet, HierarchyInstance, HierarchyDefinition>(item)) {
        EnterNetwork(panel, item, state::Null);
    } else if (const auto mesh = GetMeshEntity(r, item); r.all_of<GeometryProcedure>(mesh)) {
        EnterNetwork(panel, mesh, item);
    } else return false;
    return true;
}

void DrawSceneNodeMenu(const state::Registry &r, NodesPanel &panel, state::Entity entity) {
    const auto network = EditorNetwork(r, panel);
    // The page's selected nodes, sorted.
    auto selection = NetworkItems(r, network);
    std::erase_if(selection, [&](auto e) { return !r.any_of<Selected, BoneSelection>(e); });
    if (entity != state::Document && r.all_of<NetworkSubnet>(entity)) {
        if (MenuItem("Enter subnet")) EnterNetwork(panel, entity);
        const bool can_move = !selection.empty() && !std::ranges::contains(selection, entity);
        if (MenuItem("Move selected into subnet", nullptr, false, can_move)) action::Emit(action::nodes::MoveItems{network, entity, selection});
        if (MenuItem("Ungroup")) action::Emit(action::nodes::UngroupItems{entity});
    }
    if (r.all_of<ObjectKind>(entity)) {
        if (MenuItem("Instance hierarchy")) action::Emit(action::object::InstantiateHierarchy{entity});
        if (const auto *member = r.try_get<const HierarchyMember>(entity)) {
            if (MenuItem("Make independent")) action::Emit(action::object::MakeIndependent{member->Root});
            if (MenuItem("Enter hierarchy")) EnterNetwork(panel, member->Root);
        }
    }

    if (!selection.empty()) {
        if (MenuItem("Group selected")) action::Emit(action::nodes::GroupItems{network, selection});
        if (const auto *subnet = r.try_get<const NetworkSubnet>(network)) {
            if (MenuItem("Move selected to parent")) action::Emit(action::nodes::MoveItems{network, subnet->Parent, selection});
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
} // namespace procedure
