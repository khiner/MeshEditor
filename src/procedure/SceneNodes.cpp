#include "FileDialog.h"
#include "SortUnique.h"
#include "action/Dispatch.h"
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

namespace procedure {
using namespace ImGui;

state::Entity EditorNetwork(const state::Registry &r, const NodesPanel &panel) {
    const auto network = panel.Canvas.Network;
    if (network == state::Document || r.valid(network)) return network;
    if (network != state::Null) {
        for (const auto &location : panel.Canvas.Locations | std::views::reverse) {
            if (location.Network == state::Document || r.valid(location.Network)) return location.Network;
        }
    }
    return FrameScene(r);
}

void EnterNetwork(NodesPanel &panel, state::Entity network, state::Entity object) {
    auto &canvas = panel.Canvas;
    auto at = std::ranges::find(canvas.Locations, canvas.Network, &NetworkCanvas::Network);
    if (at != canvas.Locations.end()) canvas.Locations.erase(at);
    canvas.Locations.push_back({canvas.Network, canvas.Offset, canvas.Zoom});
    canvas.Network = network;
    canvas.Object = object;
    at = std::ranges::find(canvas.Locations, network, &NetworkCanvas::Network);
    canvas.Offset = at == canvas.Locations.end() ? vec2{} : at->Offset;
    canvas.Zoom = at == canvas.Locations.end() ? 1.f : at->Zoom;
    panel.Drag = NodeDrag::None;
    panel.Expanded = UINT32_MAX;
    panel.PendingLinks.clear();
    panel.LinkAfter = UINT32_MAX;
}

void DrawNetworkNavigation(const state::Registry &r, NodesPanel &panel) {
    const auto network = EditorNetwork(r, panel);
    if (Button("Document")) EnterNetwork(panel, state::Document);
    SameLine();
    if (Button(SceneLabel(r, FrameScene(r)).c_str())) EnterNetwork(panel, state::Null);
    const bool scene_subnet = r.all_of<NetworkSubnet>(network);
    if (scene_subnet || r.any_of<GeometryProcedure, Subnet, ArmatureObject, HierarchyInstance, HierarchyDefinition>(network)) {
        const auto owner = scene_subnet ? NetworkRoot(r, network) : r.all_of<Subnet>(network) ? r.get<const ProcedureNode>(network).Procedure :
                                                                                                network;
        const auto object = scene_subnet ? state::Null : panel.Canvas.Object;
        const auto button = [&](state::Entity e, state::Entity label) {
            SameLine();
            PushID(int(state::Integral(e)));
            if (Button(GetName(r, label).c_str())) EnterNetwork(panel, e, object);
            PopID();
        };
        if (!scene_subnet || (owner != state::Document && owner != FrameScene(r))) button(owner, object != state::Null ? object : owner);
        std::vector<state::Entity> path;
        for (auto e = network; e != owner && e != state::Null; e = scene_subnet ? r.get<const NetworkSubnet>(e).Parent : ContainerOf(r, e)) path.push_back(e);
        for (const auto e : path | std::views::reverse) button(e, e);
    }
}

const char *ItemLabel(const state::Registry &r, state::Entity entity) {
    if (entity == state::Document) return "Document";
    if (r.all_of<Material>(entity)) return "Material";
    if (r.all_of<MaterializedTexture>(entity)) return "Texture";
    if (r.all_of<NetworkSubnet>(entity)) return "Subnet";
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
    if (r.any_of<GeometryProcedure, Subnet>(shown)) {
        const auto mesh = GetMeshEntity(r, active);
        if (mesh == (r.all_of<Subnet>(shown) ? r.get<const ProcedureNode>(shown).Procedure : shown)) {
            panel.Canvas.Object = active;
            return;
        }
        if (r.all_of<GeometryProcedure>(mesh)) return EnterNetwork(panel, mesh, active);
    }
    // The page already shows the node, or the node owns the page.
    if (shown == active || ShowsItem(r, shown, active)) return;
    EnterNetwork(panel, ItemNetwork(r, active));
}

void DrawSubnetParameters(state::Registry &r, state::Entity subnet, bool interactive) {
    const auto *node = r.try_get<const ProcedureNode>(subnet);
    const auto parent = node ? ContainerOf(r, subnet) : r.get<const NetworkSubnet>(subnet).Parent;
    if (Button("Ungroup") && interactive) {
        if (node) action::Emit(action::nodes::UngroupNodes{subnet});
        else action::Emit(action::nodes::UngroupItems{subnet});
    }
    std::array<std::optional<uint16_t>, 2> previous;
    for (const auto &port : r.get<const SubnetInterface>(subnet).Ports) {
        PushID(int(port.Id));
        SeparatorText(port.Direction == SocketDirection::In ? "Input" : "Output");
        auto name = port.Name;
        ui::Gesture(InputText("Port name", &name) && interactive, [&] { return action::nodes::RenamePort{subnet, port.Id, name}; });
        const auto *field = std::get_if<PromotedField>(&port.Binding);
        if (field) {
            auto update = action::ReadUpdate(r, field->Entity, field->Component, field->Offset);
            if (update) {
                const auto &range = action::UpdatedField(field->Component, field->Offset).Field;
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
        } else if (port.Direction == SocketDirection::In && (port.Type == SocketType::Value || port.Type == SocketType::Attribute)) {
            BeginDisabled(port.Source().FromNode != state::Null);
            auto words = port.Default.Words;
            bool changed;
            if (port.Default.Type == AttributeType::Bool) {
                bool value = words[0] != 0;
                changed = Checkbox("Value", &value);
                words[0] = uint32_t(value);
            } else changed = InputScalarN("Value", port.Default.Type == AttributeType::Int ? ImGuiDataType_U32 : ImGuiDataType_Float, words.data(), int(AttributeComponents(port.Default.Type)));
            ui::Gesture(changed && interactive, [&] { return action::nodes::SetPortValue{subnet, port.Id, words}; });
            EndDisabled();
            if (port.Source().FromNode != state::Null) TextDisabled("Connected");
        }
        if (field && r.all_of<NetworkSubnet>(parent) && Button("Expose on parent") && interactive) action::Emit(action::nodes::ExposePort{parent, *field, std::make_unique<std::string>(port.Name)});
        auto &prior = previous[uint8_t(port.Direction)];
        if (prior) {
            if (Button("Move up") && interactive) action::Emit(action::nodes::MovePort{subnet, port.Id, *prior});
            SameLine();
        }
        prior = port.Id;
        if (node && port.Direction == SocketDirection::Out && (port.Type == SocketType::Geometry || port.Type == SocketType::Instances)) {
            if (Button("Display output") && interactive) action::Emit(action::nodes::SetDisplay{node->Procedure, subnet, port.Id});
            SameLine();
        }
        if (Button("Remove port") && interactive) action::Emit(action::nodes::RemovePort{subnet, port.Id});
        PopID();
    }
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
} // namespace

namespace {
// Draws the node's sections, which its components decide.
void DrawNodeProperties(state::Registry &r, NodesPanel &panel, state::Entity entity) {
    if (entity == state::Document) {
        RenderDocumentControls(r);
    } else if (r.all_of<Material>(entity)) {
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
    if (!panel.Visible) return;
    if (Begin(panel.Name, &panel.Visible)) {
        RenderInteractionControls(r);
        DrawNodeParameters(r, nodes, interactive);
        RenderSelectionActions(r);
        if (const auto active = FindActiveNode(r); active != state::Null) {
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
        view.Sockets.push_back({SocketType::Reference, direction});
        view.Bindings.push_back(std::move(binding));
    };
    for (uint32_t n = 0; n < count; ++n) {
        const auto *stored = items[n] != state::Document ? r.try_get<const NetworkNode>(items[n]) : nullptr;
        view.Nodes.push_back({.Item = items[n], .Title = stored ? stored->Title : ItemName(r, items[n])});
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
    for (size_t s = 0; s < view.Sockets.size(); ++s) view.Sockets[s].Name = view.Labels[s].c_str();
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
    if (r.any_of<Scene, GeometryProcedure, Subnet, ArmatureObject, NetworkSubnet, HierarchyInstance, HierarchyDefinition>(item)) {
        EnterNetwork(panel, item, r.all_of<Subnet>(item) ? panel.Canvas.Object : state::Null);
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
