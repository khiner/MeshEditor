#include "SortUnique.h"
#include "action/Animation.h"
#include "action/Physics.h"
#include "action/Scene.h"
#include "armature/Armature.h"
#include "audio/AudioTypes.h"
#include "audio/AudioUi.h"
#include "gltf/SourceAssets.h"
#include "physics/PhysicsUi.h"
#include "procedure/NodeEditor.h"
#include "procedure/NodeView.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/SceneControlsUi.h"
#include "timeline/Timeline.h"
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
    } else {
        const auto active = FindActiveEntity(r);
        const auto mesh = GetMeshEntity(r, active);
        if (r.all_of<GeometryProcedure>(mesh)) {
            SameLine();
            if (Button("Enter geometry")) EnterNetwork(panel, mesh, active);
        }
    }
}

namespace {
std::string ItemName(const state::Registry &r, NetworkItem item) {
    if (item.Kind == ItemKind::Animation) {
        const auto &names = r.get<const Animations>(state::Document).Names;
        return item.Index < names.size() ? names[item.Index] : "Animation";
    }
    if (item.Kind == ItemKind::Material) {
        const auto &names = r.Resources.get<const MaterialStore>().Names;
        return item.Index < names.size() && !names[item.Index].empty() ? names[item.Index] : std::format("Material {}", item.Index);
    }
    if (item.Kind == ItemKind::Texture) {
        const auto *textures = r.try_get<const MaterializedTextures>(state::Document);
        return textures && item.Index < textures->Items.size() && !textures->Items[item.Index].Params.Name.empty() ? textures->Items[item.Index].Params.Name : std::format("Texture {}", item.Index);
    }
    const auto entity = item.Entity;
    if (entity == state::Document) return "Document";
    if (const auto *m = r.try_get<const PhysicsMaterial>(entity)) return m->Name;
    if (const auto *f = r.try_get<const CollisionFilter>(entity)) return f->Name;
    if (const auto *c = r.try_get<const CollisionSystem>(entity)) return c->Name;
    if (const auto *j = r.try_get<const PhysicsJointDef>(entity)) return j->Name;
    return GetName(r, entity);
}
const char *ItemLabel(const state::Registry &r, NetworkItem item) {
    if (item.Kind == ItemKind::Animation) return "Animation";
    if (item.Kind == ItemKind::Material) return "Material";
    if (item.Kind == ItemKind::Texture) return "Texture";
    const auto entity = item.Entity;
    if (entity == state::Document) return "Settings";
    if (r.all_of<NetworkSubnet>(entity)) return "Subnet";
    if (r.all_of<Scene>(entity)) return "Scene";
    if (r.all_of<HierarchyInstance>(entity)) return "Hierarchy instance";
    if (r.all_of<HierarchyDefinition>(entity)) return "Hierarchy definition";
    if (r.all_of<GeometryProcedure>(entity)) return "Geometry";
    if (r.all_of<Armature>(entity)) return "Armature definition";
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
} // namespace

namespace {
void Reveal(state::Registry &r, NodesPanel &panel, NetworkItem target) {
    auto network = state::Document;
    auto object = state::Null;
    if (target.Kind == ItemKind::Entity && r.all_of<BoneIndex>(target.Entity)) {
        object = FindArmatureObject(r, target.Entity);
        network = object;
    } else if (target.Kind == ItemKind::Entity && r.all_of<ObjectKind>(target.Entity)) {
        network = FrameScene(r);
        if (!InScene(r, target.Entity, network)) {
            for (const auto scene : SortedEntities(r.view<const Scene>()))
                if (InScene(r, target.Entity, scene)) {
                    network = scene;
                    break;
                }
        }
    }
    if (target.Kind == ItemKind::Entity) {
        if (const auto *member = r.try_get<const HierarchyMember>(target.Entity); member && member->Root != target.Entity) network = member->Root;
        const auto parent = ParentOrNull(r, target.Entity);
        if (const auto *object = r.try_get<const Instance>(parent); object && r.all_of<ProcedureInstances>(object->Entity) && std::ranges::contains(r.get<const ProcedureInstances>(object->Entity).Entities, target.Entity)) network = parent;
    }
    for (;;) {
        const auto representative = NetworkRepresentative(r, network, target);
        if (representative == target || !r.all_of<NetworkSubnet>(representative.Entity)) break;
        network = representative.Entity;
    }
    EnterNetwork(panel, network, object);
    action::Emit(action::nodes::SelectItems{network, target, {target}});
}
} // namespace

void DrawSubnetParameters(state::Registry &r, NodesPanel &panel, state::Entity subnet, bool interactive) {
    const auto *node = r.try_get<const ProcedureNode>(subnet);
    const auto parent = node ? ContainerOf(r, subnet) : r.get<const NetworkSubnet>(subnet).Parent;
    if (!node) {
        auto name = GetName(r, subnet);
        ui::Gesture(InputText("Name", &name) && interactive, [&] { return action::nodes::RenameNetwork{subnet, name}; });
    }
    if (Button("Enter subnet") && interactive) EnterNetwork(panel, subnet, node ? panel.Canvas.Object : state::Null);
    SameLine();
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
        const auto *reference = std::get_if<SceneReference>(&port.Binding);
        const auto *item = std::get_if<NetworkItem>(&port.Binding);
        if (field) {
            auto update = action::ReadUpdate(r, field->Entity, field->Component, field->Offset);
            if (update) {
                const auto &spec = action::UpdatedField(field->Component, field->Offset).Field.Spec;
                std::visit([&]<typename A>(A &value) {
                    if constexpr (action::IsUpdate<A>) {
                        if constexpr (ui::DrawableField<decltype(value.Value)>) ui::Gesture(ui::DrawField(port.Name.c_str(), value.Value, spec) && interactive, [&] { return value; });
                    }
                },
                           *update);
            } else TextDisabled("%s (missing field)", port.Name.c_str());
        } else if (port.Type == SocketType::Reference) {
            const auto value = reference ? ReferenceValue(r, *reference) : item ? *item :
                                                                                  NetworkItem{};
            if (value != NetworkItem{} && Button(ItemName(r, value).c_str()) && interactive) Reveal(r, panel, value);
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
        if (r.all_of<NetworkSubnet>(parent) && Button("Expose on parent") && interactive) action::Emit(action::nodes::ExposePort{parent, port.Binding, std::make_unique<std::string>(port.Name)});
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

void DrawNetworkParameters(state::Registry &r, NodesPanel &panel) {
    const auto *active = r.try_get<const NetworkActive>(EditorNetwork(r, panel));
    const auto item = active ? active->Item : NetworkItem{};
    if (item.Kind == ItemKind::Animation) {
        const auto &animations = r.get<const Animations>(state::Document);
        if (item.Index >= animations.Names.size()) return;
        auto name = animations.Names[item.Index];
        ui::Gesture(InputText("Name", &name), [&] { return action::animation::RenameAnimation{item.Index, name}; });
        if (item.Index == animations.Active) TextUnformatted("Active animation");
        else if (Button("Activate animation")) action::Emit(action::animation::SelectAnimation{item.Index});
        if (auto *panels = r.Resources.find<PanelsState>(); panels && Button("Open timeline")) panels->Timeline.Visible = true;
        SeparatorText("Targets");
        for (const auto entity : SortedEntities(r.view<const AnimationClips>())) {
            const auto *clip = r.get<const AnimationClips>(entity).Find(item.Index);
            if (!clip) continue;
            PushID(int(state::Integral(entity)));
            if (entity != state::Document && Button(GetName(r, entity).c_str())) Reveal(r, panel, {entity});
            for (size_t i = 0; i < clip->Channels.size(); ++i) {
                const auto &channel = clip->Channels[i];
                PushID(int(i));
                Text("%s (%zu keys)", animation::FieldLabel(channel.Target).c_str(), channel.Times.size());
                if (channel.Target.Component == state::Key<MaterialStore>() && Button("Inspect material")) Reveal(r, panel, {state::Document, channel.Target.Index, ItemKind::Material});
                PopID();
            }
            PopID();
        }
    } else if (item.Kind == ItemKind::Material) {
        if (item.Index >= r.Resources.get<const MaterialStore>().Names.size()) return;
        SeparatorText(ItemName(r, item).c_str());
        ui::DrawFields<PBRMaterial>(ui::MaterialEdit{r, item.Index});
    } else if (item.Kind == ItemKind::Texture) {
        const auto *textures = r.try_get<const MaterializedTextures>(state::Document);
        if (!textures || item.Index >= textures->Items.size()) return;
        const auto &texture = textures->Items[item.Index];
        SeparatorText(texture.Params.Name.c_str());
        TextUnformatted("Imported texture");
        if (const auto *source = r.try_get<const gltf::SourceAssets>(state::Document); source && texture.SourceImageIndex < source->Images.size()) {
            const auto &image = source->Images[texture.SourceImageIndex];
            const auto &path = image.SourcePath.empty() ? image.Uri : image.SourcePath;
            if (!path.empty()) TextWrapped("Source: %s", path.c_str());
            else TextUnformatted("Embedded image");
        }
        Text("Color space: %s", texture.Params.ColorSpace == TextureColorSpace::Srgb ? "sRGB" : "Linear");
    } else {
        const auto entity = r.valid(item.Entity) ? item.Entity : FindActiveEntity(r);
        if (entity == state::Document) {
            ui::Edit fields{r, state::Document};
            if (CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) ui::DrawFields<PhysicsSimulationSettings>(fields);
            if (r.all_of<ImageLight>(state::Document) && CollapsingHeader("Environment")) ui::DrawFields<ImageLight>(fields);
            if (CollapsingHeader("Audio")) {
                ui::DrawFields<AudioOutputMix>(fields);
                DrawGlobalSynthControls(r);
            }
            if (CollapsingHeader("Timeline")) {
                ui::DrawFields<TimelineNavigation>(fields);
                fields.Draw<&Animations::Record>();
                if (auto *panels = r.Resources.find<PanelsState>(); panels && Button("Open timeline")) panels->Timeline.Visible = true;
            }
            if (const auto *variants = r.try_get<const MaterialVariants>(state::Document); variants && !variants->Names.empty() && CollapsingHeader("Material variants")) {
                if (Selectable("Default", !variants->Active)) fields.Set<&MaterialVariants::Active>({});
                for (uint32_t i = 0; i < variants->Names.size(); ++i) {
                    PushID(int(i));
                    if (Selectable(variants->Names[i].c_str(), variants->Active == i)) fields.Set<&MaterialVariants::Active>(i);
                    PopID();
                }
            }
            if (const auto *source = r.try_get<const gltf::SourceAssets>(state::Document); source && CollapsingHeader("Imported source")) {
                TextWrapped("Generator: %s", source->Generator.c_str());
                TextWrapped("Copyright: %s", source->Copyright.c_str());
                for (const auto &extension : source->ExtensionsRequired) TextUnformatted(extension.c_str());
                if (!source->AssetExtras.empty()) TextWrapped("%s", source->AssetExtras.c_str());
            }
            return;
        }
        if (r.all_of<NetworkSubnet>(entity)) {
            DrawSubnetParameters(r, panel, entity, true);
            return;
        }
        if (const auto *member = r.try_get<const HierarchyMember>(entity)) {
            Text("Source: %s", GetName(r, member->Source).c_str());
            if (Button("Reveal source")) Reveal(r, panel, {member->Source});
            if (Button("Make independent")) action::Emit(action::object::MakeIndependent{member->Root});
            if (member->Root == entity && Button("Enter hierarchy")) EnterNetwork(panel, entity);
        }
        if (r.all_of<ObjectKind>(entity) && Button("Instance hierarchy")) action::Emit(action::object::InstantiateHierarchy{entity});
        if (const auto *object = r.try_get<const Instance>(entity); object && r.all_of<ProcedureInstances>(object->Entity)) {
            if (Button("Inspect generated instances")) EnterNetwork(panel, entity);
        }
        if (physics_ui::RenderResourceProperties(r, entity)) return;
        if (r.all_of<GeometryProcedure>(entity)) {
            SeparatorText(GetName(r, entity).c_str());
            if (Button("Enter geometry")) EnterNetwork(panel, entity);
        } else if (const auto *definition = r.try_get<const Armature>(entity)) {
            SeparatorText("Armature definition");
            Text("%zu bones", definition->Bones.size());
            for (const auto &bone : definition->Bones) TextUnformatted(bone.Name.c_str());
            for (const auto object : SortedEntities(r.view<const ArmatureObject>())) {
                if (r.get<const ArmatureObject>(object).Entity != entity) continue;
                PushID(int(state::Integral(object)));
                if (Button(GetName(r, object).c_str())) EnterNetwork(panel, object, object);
                PopID();
            }
        } else if (r.all_of<Scene>(entity)) {
            auto name = SceneLabel(r, entity);
            ui::Gesture(InputText("Name", &name), [&] { return action::scene::Rename{entity, name}; });
            if (Button("Enter scene")) EnterNetwork(panel, entity);
        } else RenderEntityControls(r, entity, r.all_of<BoneIndex>(entity) ? entity : state::Null);
    }
}
NodeView SceneNodeDefinitions(const state::Registry &r, state::Entity network) {
    NodeView view{.Network = network, .InputFirst = {0}, .SocketFirst = {0}};
    const auto items = NetworkItems(r, network);
    const auto port = [&](std::string label, PortBinding binding) {
        view.Labels.push_back(std::move(label));
        const auto direction = std::holds_alternative<SceneReference>(binding) ? SocketDirection::In : SocketDirection::Out;
        view.Sockets.push_back({SocketType::Reference, direction});
        view.Bindings.push_back(std::move(binding));
    };
    for (const auto item : items) {
        EditorNode node{.Item = item, .Title = ItemName(r, item), .Type = ItemLabel(r, item), .Position = NetworkPosition(r, network, item)};
        uint16_t inputs{};
        if (item.Kind == ItemKind::Entity && r.all_of<NetworkSubnet>(item.Entity)) {
            const auto &ports = r.get<const SubnetInterface>(item.Entity).Ports;
            for (const auto &p : ports)
                if (std::holds_alternative<SceneReference>(p.Binding)) port(p.Name, p.Binding);
            inputs = uint16_t(view.Sockets.size() - view.SocketFirst.back());
            for (const auto &p : ports)
                if (std::holds_alternative<NetworkItem>(p.Binding)) port(p.Name, p.Binding);
        } else {
            for (const auto reference : SceneReferences(r, item)) port(ReferenceLabel(reference), reference);
            inputs = uint16_t(view.Sockets.size() - view.SocketFirst.back());
            port("Reference", item);
        }
        view.InputFirst.push_back(view.InputFirst.back() + inputs);
        view.Nodes.push_back(std::move(node));
        view.SocketFirst.push_back(uint32_t(view.Sockets.size()));
    }
    for (size_t s = 0; s < view.Sockets.size(); ++s) view.Sockets[s].Name = view.Labels[s].c_str();
    view.Wires.resize(view.InputFirst.back());
    const auto index_of = [&](NetworkItem item) -> uint32_t {
        item = NetworkRepresentative(r, network, item);
        const auto at = std::ranges::lower_bound(items, item);
        return at != items.end() && *at == item ? uint32_t(at - items.begin()) : NoNode;
    };
    const auto relationship = [&](NetworkItem from, NetworkItem to) {
        const auto a = index_of(from), b = index_of(to);
        if (a == NoNode || b == NoNode || a == b) return;
        const auto inputs = std::span{view.Wires}.subspan(view.InputFirst[b], InputsOf(view, b));
        if (!std::ranges::any_of(inputs, [&](const auto &link) { return link.FromNode == a; })) view.Relationships.emplace_back(a, b);
    };
    for (uint32_t n = 0; n < items.size(); ++n) {
        for (uint16_t s = 0; s < InputsOf(view, n); ++s) {
            const auto source = ReferenceValue(r, std::get<SceneReference>(BindingOf(view, n, s)));
            const auto from = index_of(source);
            if (from == NoNode || from == n) continue;
            for (uint16_t o = InputsOf(view, from); o < NodeSockets(view, from).size(); ++o) {
                if (std::get<NetworkItem>(BindingOf(view, from, o)) != source) continue;
                view.Wires[view.InputFirst[n] + s] = {from, o};
                break;
            }
            if (view.Wires[view.InputFirst[n] + s].FromNode == NoNode) relationship(source, items[n]);
        }
    }
    if (NetworkRoot(r, network) == state::Document) {
        for (const auto item : NetworkLeaves(r, network)) {
            for (const auto reference : SceneReferences(r, item)) {
                if (reference.Kind == ReferenceKind::Material || reference.Kind == ReferenceKind::Texture) relationship(ReferenceValue(r, reference), item);
            }
            if (item.Kind != ItemKind::Entity) continue;
            if (const auto *filter = r.try_get<const CollisionFilter>(item.Entity)) {
                for (const auto system : filter->Systems) relationship({system}, item);
                for (const auto system : filter->CollideSystems) relationship({system}, item);
            }
            if (r.all_of<Scene>(item.Entity)) {
                for (const auto object : r.view<const ObjectKind>()) {
                    if (!InScene(r, object, item.Entity)) continue;
                    if (const auto *instance = r.try_get<const Instance>(object); instance && r.all_of<GeometryProcedure>(instance->Entity)) relationship({instance->Entity}, item);
                    if (const auto *armature = r.try_get<const ArmatureObject>(object)) relationship({armature->Entity}, item);
                }
            }
        }
    }
    SortUnique(view.Relationships);
    return view;
}

void DrawSceneAddMenu(const state::Registry &r, state::Entity network) {
    const auto root = NetworkRoot(r, network);
    const auto container = r.all_of<NetworkSubnet>(network) ? network : state::Null;
    if (network != state::Document && !r.all_of<Scene>(root)) return;
    if (root == state::Document) {
        if (MenuItem("Scene")) action::Emit(action::scene::Add{"Scene"});
        if (MenuItem("Animation")) action::Emit(action::animation::AddAnimation{"Animation"});
        if (MenuItem("Physics material")) action::Emit(action::physics::AddPhysicsMaterial{});
        if (MenuItem("Collision system")) action::Emit(action::physics::AddCollisionSystem{});
        if (MenuItem("Collision filter")) action::Emit(action::physics::AddCollisionFilter{});
        if (MenuItem("Joint definition")) action::Emit(action::physics::AddJointDef{});
    } else {
        const std::array<PrimitiveParameters, 8> primitives{primitive::Plane{}, primitive::Circle{}, primitive::Cube{}, primitive::UVSphere{}, primitive::IcoSphere{}, primitive::Torus{}, primitive::Cylinder{}, primitive::Cone{}};
        for (const auto &primitive : primitives) {
            if (MenuItem(std::visit([]<typename K>(const K &) { return Title<K>.c_str(); }, primitive))) action::Emit(action::object::AddMeshPrimitive{primitive, std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Scene = root, .BeginTranslate = false, .Network = container})});
        }
        Separator();
        const auto info = [&] { return std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Scene = root, .BeginTranslate = false, .Network = container}); };
        if (MenuItem("Empty")) action::Emit(action::object::AddEmpty{info()});
        if (MenuItem("Armature")) action::Emit(action::object::AddArmature{info()});
        if (MenuItem("Camera")) action::Emit(action::object::AddCamera{info()});
        if (MenuItem("Light")) action::Emit(action::object::AddLight{info()});
    }
}

bool EnterNode(const state::Registry &r, NodesPanel &panel, NetworkItem item) {
    if (item.Kind != ItemKind::Entity) return false;
    if (r.any_of<Scene, GeometryProcedure, Subnet, ArmatureObject, NetworkSubnet, HierarchyInstance, HierarchyDefinition>(item.Entity)) {
        EnterNetwork(panel, item.Entity, r.all_of<Subnet>(item.Entity) ? panel.Canvas.Object : state::Null);
    } else if (const auto mesh = GetMeshEntity(r, item.Entity); r.all_of<GeometryProcedure>(mesh)) {
        EnterNetwork(panel, mesh, item.Entity);
    } else return false;
    return true;
}

void DrawSceneNodeMenu(const state::Registry &r, NodesPanel &panel, NetworkItem item) {
    const auto network = EditorNetwork(r, panel), entity = item.Entity;
    const auto container = r.all_of<NetworkSubnet>(network) ? network : state::Null;
    const auto *inspected = r.try_get<const NetworkActive>(network);
    if (item.Kind == ItemKind::Entity && r.all_of<NetworkSubnet>(entity)) {
        if (MenuItem("Enter subnet")) EnterNetwork(panel, entity);
        const bool can_move = inspected && !inspected->Selection.empty() && !std::ranges::contains(inspected->Selection, item);
        if (MenuItem("Move selected into subnet", nullptr, false, can_move)) action::Emit(action::nodes::MoveItems{network, entity, inspected->Selection});
        if (MenuItem("Ungroup")) action::Emit(action::nodes::UngroupItems{entity});
    } else if (container != state::Null) {
        if (MenuItem("Expose reference output")) action::Emit(action::nodes::ExposePort{container, item});
        for (const auto reference : SceneReferences(r, item)) {
            if (MenuItem(("Expose " + ReferenceLabel(reference)).c_str())) action::Emit(action::nodes::ExposePort{container, reference});
        }
    }
    if (item.Kind == ItemKind::Entity && r.all_of<ObjectKind>(entity)) {
        if (MenuItem("Instance hierarchy")) action::Emit(action::object::InstantiateHierarchy{entity});
        if (const auto *member = r.try_get<const HierarchyMember>(entity)) {
            if (MenuItem("Make independent")) action::Emit(action::object::MakeIndependent{member->Root});
            if (MenuItem("Enter hierarchy")) EnterNetwork(panel, member->Root);
        }
    }

    if (inspected && !inspected->Selection.empty()) {
        if (MenuItem("Group selected")) action::Emit(action::nodes::GroupItems{network, inspected->Selection});
        if (const auto *subnet = r.try_get<const NetworkSubnet>(network)) {
            if (MenuItem("Move selected to parent")) action::Emit(action::nodes::MoveItems{network, subnet->Parent, inspected->Selection});
        }
    }
}

std::string ReferenceDescription(const state::Registry &r, const PortBinding &binding) {
    const auto *reference = std::get_if<SceneReference>(&binding);
    const auto item = reference ? ReferenceValue(r, *reference) : std::get<NetworkItem>(binding);
    return item == NetworkItem{} ? "Unassigned" : ItemName(r, item);
}

void DrawReferenceMenu(const state::Registry &r, state::Entity network, SceneReference reference) {
    const auto container = r.all_of<NetworkSubnet>(network) ? network : state::Null;
    if (BeginMenu("Assign")) {
        for (const auto candidate : ReferenceCandidates(r, reference)) {
            PushID(int(candidate.Kind));
            PushID(int(state::Integral(candidate.Entity)));
            PushID(int(candidate.Index));
            if (MenuItem(ItemName(r, candidate).c_str())) action::Emit(action::nodes::AssignReference{reference, candidate});
            PopID();
            PopID();
            PopID();
        }
        EndMenu();
    }
    if (MenuItem("Disconnect", nullptr, false, ReferenceAccepts(r, reference, {}))) action::Emit(action::nodes::AssignReference{reference, {}});
    if (container != state::Null && MenuItem("Expose on subnet")) action::Emit(action::nodes::ExposePort{container, reference});
}
} // namespace procedure
