#include "procedure/OpNodes.h"
#include "SortUnique.h"
#include "Variant.h"
#include "action/Errors.h"
#include "action/Nodes.h"
#include "materialx/Flatten.h"
#include "procedure/Cook.h"
#include "procedure/MaterialGraphs.h"
#include "render/GpuBuffers.h"
#include "render/Textures.h"
#include "state/Registry.h"

namespace procedure {
namespace {
// Each data type a MaterialX value has, with its MaterialX type.
using TypePair = std::pair<DataType, mx::Type>;
constexpr TypePair ValueTypes[]{
    {DataType::Bool, mx::Type::Boolean},
    {DataType::Int, mx::Type::Integer},
    {DataType::Float, mx::Type::Float},
    {DataType::Vec2, mx::Type::Vector2},
    {DataType::Vec3, mx::Type::Vector3},
    {DataType::Vec4, mx::Type::Vector4},
    {DataType::Color3, mx::Type::Color3},
    {DataType::Color4, mx::Type::Color4},
    {DataType::Matrix33, mx::Type::Matrix33},
    {DataType::Matrix44, mx::Type::Matrix44},
    {DataType::Image, mx::Type::Filename},
};
constexpr std::array StoredOps{
    StoredOp{"selection", StoredDomainRule::Selection, StoredJob::Selection},
    StoredOp{"curvature", StoredDomainRule::Vertex, StoredJob::Curvature},
    StoredOp{"boundary", StoredDomainRule::Text, StoredJob::Boundary},
    StoredOp{"convertdomain", StoredDomainRule::SourceTarget, StoredJob::Convert},
    StoredOp{"adjacent", StoredDomainRule::Text, StoredJob::Adjacent},
    StoredOp{"linked", StoredDomainRule::Text, StoredJob::Components, FieldAdjacency::Domain},
    StoredOp{"edgeloop", StoredDomainRule::Edge, StoredJob::Components, FieldAdjacency::EdgeLoop},
    StoredOp{"edgering", StoredDomainRule::Edge, StoredJob::Components, FieldAdjacency::EdgeRing},
};
// Hashes a name by its characters, so a lookup takes any string view.
struct NameHash {
    using is_transparent = void;
    size_t operator()(std::string_view name) const { return std::hash<std::string_view>{}(name); }
};
// Whether the declaration's output types all have Metal values.
bool HasValues(const mx::Declaration &d) {
    return !d.Outputs.empty() && std::ranges::none_of(d.Outputs, [](const mx::Port &port) { return mx::MetalType(port.Type).empty(); });
}
// Whether each of the declaration's outputs is a value or an image.
bool HasResults(const mx::Declaration &d) {
    return !d.Outputs.empty() && std::ranges::all_of(d.Outputs, [](const mx::Port &port) { return !mx::MetalType(port.Type).empty() || port.Type == mx::Type::Filename; });
}
// Adds what the native nodes with values in the declaration's definition read to the definition, and returns the domains every one evaluates on.
ops::Domains AddNativeReads(const mx::Library &lib, uint32_t declaration, OpDefinition &definition) {
    ops::Domains domains = ops::AllDomains;
    for (const auto &node : mx::Flatten(lib, declaration).Nodes) {
        const auto &native = lib.Declarations[node.Declaration];
        if (!std::holds_alternative<mx::Native>(lib.Implementations[node.Declaration]) || !HasValues(native)) continue;
        domains &= ops::Availability(lib, node.Declaration);
        definition.ReadsFrame |= native.Node == "frame";
        definition.ReadsTime |= native.Node == "time";
    }
    return domains;
}
std::optional<OpDefinition> Define(std::string_view node) {
    const auto &lib = mx::Libraries();
    const auto declarations = mx::NodeDeclarations(lib, node);
    if (declarations.empty()) return {};
    // An input takes a single value where every declaration's input of its name is a stored-field op's uniform input, an image, or a filter's input after its first.
    // Selection's first input is a selection socket, which reads the node's selection while unlinked, which MaterialX has no way to declare.
    struct Port {
        const char *Name;
        DataTypes Types;
        bool Reads{}, Single{true}, Selection{};
    };
    std::vector<Port> inputs, outputs;
    const auto merge = [](std::vector<Port> &ports, const mx::Port &port, DataType type) -> Port & {
        auto found = std::ranges::find(ports, std::string_view{port.Name}, [](const Port &p) { return std::string_view{p.Name}; });
        if (found == ports.end()) found = ports.insert(ports.end(), {port.Name.c_str(), 0u});
        found->Types |= TypeBit(type);
        return *found;
    };
    const bool stored = StoredOpOf(node), filter = node == "blur";
    OpDefinition definition{.Stored = stored, .Filter = filter};
    for (const auto d : declarations) {
        const auto &declaration = lib.Declarations[d];
        // Only a filter gives an image.
        if (!filter && std::ranges::contains(declaration.Outputs, mx::Type::Filename, &mx::Port::Type)) continue;
        for (const auto &port : declaration.Inputs) {
            if (const auto type = DataTypeOf(port.Type)) {
                auto &merged = merge(inputs, port, *type);
                merged.Reads |= port.DefaultGeomProp.has_value();
                merged.Single &= (stored && port.Uniform) || *type == DataType::Image || (filter && &port != &declaration.Inputs.front());
                merged.Selection |= declaration.Node == "selection" && &port == &declaration.Inputs.front();
            } else if (port.Type == mx::Type::String && !std::ranges::contains(definition.Texts, std::string_view{port.Name}, &std::pair<std::string_view, mx::Type>::first)) definition.Texts.emplace_back(port.Name, port.Type);
        }
        for (const auto &port : declaration.Outputs)
            if (const auto type = DataTypeOf(port.Type)) merge(outputs, port, *type);
        const auto domains = stored ? ops::MeshDomains : filter ? ops::AllDomains : AddNativeReads(lib, d, definition);
        definition.Source |= domains != ops::AllDomains;
        if (HasValues(declaration)) definition.Domains |= domains;
    }
    const auto signature = [&](uint32_t d) {
        std::vector<std::pair<std::string_view, mx::Type>> types;
        for (const auto &port : lib.Declarations[d].Inputs) types.emplace_back(port.Name, port.Type);
        return types;
    };
    const auto output_types = [&](uint32_t d) { return lib.Declarations[d].Outputs | std::views::transform(&mx::Port::Type) | std::ranges::to<std::vector>(); };
    for (size_t i = 0; i < declarations.size(); ++i) {
        for (size_t j = i + 1u; j < declarations.size(); ++j) definition.StoresOutput |= signature(declarations[i]) == signature(declarations[j]) && output_types(declarations[i]) != output_types(declarations[j]);
        const auto &outputs = lib.Declarations[declarations[i]].Outputs;
        if (outputs.size() == 1u && !std::ranges::contains(definition.OutputTypes, outputs[0].Type)) definition.OutputTypes.push_back(outputs[0].Type);
    }
    for (const auto &port : inputs) {
        definition.Sockets.push_back({.Types = port.Types, .Direction = SocketDirection::In, .Varies = !port.Single, .Name = port.Name, .Selection = port.Selection});
        definition.ReadsUnset.push_back(port.Reads);
    }
    for (const auto &port : outputs) definition.Sockets.push_back({.Types = port.Types, .Direction = SocketDirection::Out, .Name = port.Name});
    definition.Inputs = uint16_t(inputs.size());
    return definition;
}

std::optional<mx::Type> TextType(const OpDefinition &definition, std::string_view name) {
    const auto found = std::ranges::find(definition.Texts, name, &std::pair<std::string_view, mx::Type>::first);
    return found == definition.Texts.end() ? std::nullopt : std::optional{found->second};
}
// The default of a read of a field of the type.
std::vector<uint32_t> ReadDefault(mx::Type type) {
    const auto &lib = mx::Libraries();
    const std::array inputs{mx::TypedInput{"geomprop", mx::Type::String}};
    const auto &ports = lib.Declarations[*mx::FindDeclaration(lib, "geompropvalue", inputs, type)].Inputs;
    return std::get<std::vector<uint32_t>>(std::ranges::find(ports, "default", &mx::Port::Name)->Default);
}
// Gives the owner a new block of `words` words laid out as the slots, where each literal slot the held block has keeps its value, and every other slot `i` holds the words of `initial(i)`.
void Reblock(state::Registry &r, state::Entity owner, const OpValueBlock *held, std::vector<OpSlot> slots, uint32_t words, auto &&initial) {
    auto &store = r.Resources.get<GpuBuffers>().OpValues;
    std::vector<uint32_t> values(words);
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto &slot = slots[i];
        const auto *kept = held && slot.Kind == OpSlotKind::Literal ? FindOpSlot(*held, slot.Node, slot.Input, slot.Type, slot.Kind) : nullptr;
        std::ranges::copy(kept ? store.Get({held->Block.Offset + kept->Offset, mx::Words(slot.Type)}) : initial(i), values.begin() + slot.Offset);
    }
    const auto block = store.Allocate(std::span<const uint32_t>{values});
    if (held) store.Release(held->Block);
    r.emplace_or_replace<OpValueBlock>(owner, block, std::move(slots));
}
NodeFault FaultOf(ops::Fault fault) {
    switch (fault) {
        case ops::Fault::Unresolved: return NodeFault::Unresolved;
        case ops::Fault::NoValue: return NodeFault::NoValue;
        case ops::Fault::Unavailable: return NodeFault::Unavailable;
        case ops::Fault::Ambiguous: return NodeFault::Ambiguous;
        case ops::Fault::Unstored: return NodeFault::Unstored;
        case ops::Fault::OutputLimit: return NodeFault::OutputLimit;
    }
    std::unreachable();
}
bool IsOp(const CookPlan &plan, uint32_t node) { return node != NoNode && plan.Nodes[node].Kind == KindOf<Op>; }
const Op &OpOf(const state::Registry &r, const CookPlan &plan, uint32_t node) { return Authored(r, plan.Nodes[node]).get<const Op>(plan.Nodes[node].Entity); }
} // namespace

std::optional<uint16_t> InputSocket(const OpDefinition &definition, std::string_view name) {
    for (uint16_t s = 0; s < definition.Inputs; ++s)
        if (definition.Sockets[s].Name == name) return s;
    return {};
}
const StoredOp *StoredOpOf(std::string_view node) {
    const auto found = std::ranges::find(StoredOps, node, &StoredOp::Node);
    return found == StoredOps.end() ? nullptr : &*found;
}
bool IsStoredOp(const state::Registry &r, const CookPlan &plan, uint32_t node) { return IsOp(plan, node) && DefinitionOf(OpOf(r, plan, node).Node)->Stored; }
bool IsInlineOp(const state::Registry &r, const CookPlan &plan, uint32_t node) { return IsOp(plan, node) && !DefinitionOf(OpOf(r, plan, node).Node)->Stored; }
bool IsFilterOp(const state::Registry &r, const CookPlan &plan, uint32_t node) { return IsOp(plan, node) && DefinitionOf(OpOf(r, plan, node).Node)->Filter; }

std::string_view TextOf(const Op &op, std::string_view input, uint32_t declaration) {
    if (const auto set = op.Texts.find(input); set != op.Texts.end()) return set->second;
    const auto &ports = mx::Libraries().Declarations[declaration].Inputs;
    return std::get<std::string>(std::ranges::find(ports, input, &mx::Port::Name)->Default);
}


std::optional<mx::Type> OutputType(const state::Registry &r, const CookPlan &plan, const OpLayout &layout, uint32_t node, uint16_t socket) {
    const auto &[entity, kind, parent, builtin] = plan.Nodes[node];
    if (kind == KindOf<Op>) {
        const auto &resolved = layout.Nodes[node];
        const auto output = socket - DefinitionOf(Authored(r, plan.Nodes[node]).get<const Op>(entity).Node)->Inputs;
        if (!resolved.Declaration || resolved.Ports[output] == NoNode) return {};
        return resolved.Outputs[output];
    }
    const auto types = NodeSockets(plan, node)[socket].Types;
    if (!std::has_single_bit(types) || !(types & (NumericTypes | MatrixTypes | TypeBit(DataType::Image)))) return {};
    return ValueTypeOf(DataType(std::countr_zero(types)));
}

const OpSlot *FindOpSlot(const OpValueBlock &block, const NodePath &node, std::string_view input, mx::Type type, OpSlotKind kind) {
    const auto found = std::ranges::find_if(block.Slots, [&](const OpSlot &slot) { return slot.Node == node && slot.Input == input && slot.Type == type && slot.Kind == kind; });
    return found == block.Slots.end() ? nullptr : &*found;
}

std::optional<DataType> DataTypeOf(mx::Type type) {
    const auto found = std::ranges::find(ValueTypes, type, &TypePair::second);
    return found == std::ranges::end(ValueTypes) ? std::nullopt : std::optional{found->first};
}
mx::Type ValueTypeOf(DataType type) { return std::ranges::find(ValueTypes, type, &TypePair::first)->second; }

const OpDefinition *DefinitionOf(std::string_view node) {
    static std::unordered_map<std::string, std::optional<OpDefinition>, NameHash, std::equal_to<>> definitions;
    auto found = definitions.find(node);
    if (found == definitions.end()) found = definitions.emplace(std::string{node}, Define(node)).first;
    return found->second ? &*found->second : nullptr;
}

bool OpVaries(const state::Registry &r, state::Entity node) {
    std::vector<state::Entity> visited;
    // A node visited again returned false, since a varying one ends the walk.
    const auto varies = [&](this const auto &self, const state::Registry &at, state::Entity n) -> bool {
        if (std::ranges::contains(visited, n)) return false;
        visited.push_back(n);
        const auto &op = at.get<const Op>(n);
        const auto *definition = DefinitionOf(op.Node);
        const auto &links = at.get<const NodeInputs>(n).Links;
        return definition && OpVaries(*definition, op, [&](uint16_t s) {
                   const auto resolved = ResolveSocket(at, links[s], true);
                   const auto from = resolved.Source.FromNode;
                   if (from == state::Null) return resolved.Constant ? OpInput::Single : OpInput::Unset;
                   const auto &source = *resolved.Registry;
                   return (source.all_of<Op>(from) ? self(source, from) : SocketOf(source, from, resolved.Source.FromSocket).Varies) ? OpInput::Varying : OpInput::Single;
               });
    };
    return varies(r, node);
}

OpLayout LayoutOf(const state::Registry &r, const CookPlan &plan) {
    const auto &lib = mx::Libraries();
    OpLayout layout{.Nodes = std::vector<OpLayout::Node>(plan.Nodes.size())};
    // Each op node's given inputs, which resolve its declaration.
    std::vector<std::vector<mx::TypedInput>> given(plan.Nodes.size());
    for (const auto n : plan.Order) {
        if (!IsOp(plan, n)) continue;
        const auto &op = Authored(r, plan.Nodes[n]).get<const Op>(plan.Nodes[n].Entity);
        const auto *definition = DefinitionOf(op.Node);
        if (!definition) continue;
        bool reads_fault = false;
        for (uint16_t s = 0; s < definition->Inputs; ++s) {
            const auto &link = plan.Inputs[plan.InputFirst[n] + s];
            const std::string_view name{definition->Sockets[s].Name};
            std::optional<mx::Type> type;
            if (link.FromNode != NoNode && definition->Sockets[s].Types == TypeBit(DataType::Image)) {
                type = mx::Type::Filename; // An image input reads the image table, which holds no image for a link without one
            } else if (link.FromNode != NoNode) {
                type = OutputType(r, plan, layout, link.FromNode, link.FromSocket);
                reads_fault |= !type;
            } else if (link.Constant) {
                type = ValueTypeOf(link.Constant->Type);
            } else if (op.Images.contains(name)) {
                type = mx::Type::Filename;
            } else if (const auto literal = op.Literals.find(name); literal != op.Literals.end()) {
                type = literal->second;
            }
            if (type) given[n].push_back({name, *type});
        }
        if (reads_fault) continue;
        for (const auto &[name, text] : op.Texts)
            if (const auto type = TextType(*definition, name)) given[n].push_back({name, *type});
        const auto resolution = ops::Resolve(lib, op.Node, given[n], op.Output);
        if (!resolution || !HasResults(lib.Declarations[resolution->Declaration])) continue;
        const auto &declaration = lib.Declarations[resolution->Declaration];
        auto &resolved = layout.Nodes[n];
        resolved.Declaration = resolution->Declaration;
        for (const auto &socket : std::span{definition->Sockets}.subspan(definition->Inputs)) {
            const auto port = std::ranges::find(declaration.Outputs, std::string_view{socket.Name}, &mx::Port::Name);
            resolved.Ports.push_back(port == declaration.Outputs.end() ? NoNode : uint32_t(port - declaration.Outputs.begin()));
            resolved.Outputs.push_back(port == declaration.Outputs.end() ? mx::Type::MultiOutput : port->Type);
        }
    }
    const auto add = [&](uint32_t node, std::string_view input, mx::Type type, OpSlotKind kind, std::vector<uint32_t> initial) {
        layout.Slots.push_back({node, std::string{input}, type, kind, layout.Words});
        initial.resize(mx::Words(type));
        layout.Initial.push_back(std::move(initial));
        layout.Words += mx::Words(type);
    };
    // A literal of a top-level op lives in this block, and a definition's op occurrence holds a copy of its definition's literal.
    const auto literal = [&](uint32_t n, std::string_view input, mx::Type type) {
        const auto &node = plan.Nodes[n];
        if (node.Parent == NoNode) return std::pair{OpSlotKind::Literal, std::vector<uint32_t>{}};
        return std::pair{OpSlotKind::Copy, std::vector<uint32_t>{std::from_range, LiteralWords(Authored(r, node), node.Entity, input, type)}};
    };
    // Each node's slots follow the order the op compiler gives them, then its field reads' defaults, then the default of each read of a native source's output in its place, then its literals no program reads.
    for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
        if (!IsOp(plan, n)) continue;
        const auto &op = Authored(r, plan.Nodes[n]).get<const Op>(plan.Nodes[n].Entity);
        const auto *definition = DefinitionOf(op.Node);
        auto &resolved = layout.Nodes[n];
        resolved.Values = layout.Words;
        std::vector<std::string_view> placed;
        if (resolved.Declaration) {
            for (const auto &port : lib.Declarations[*resolved.Declaration].Inputs) {
                const auto socket = InputSocket(*definition, port.Name);
                const auto in = std::ranges::find(given[n], std::string_view{port.Name}, &mx::TypedInput::Name);
                if (in == given[n].end()) {
                    const auto *value = std::get_if<std::vector<uint32_t>>(&port.Default);
                    if (!port.DefaultGeomProp && value && !value->empty()) add(n, port.Name, port.Type, OpSlotKind::Default, *value);
                    continue;
                }
                if (!socket || in->Type == mx::Type::Filename) continue; // A text compiles into the program, and an image reads the image table
                const auto &link = plan.Inputs[plan.InputFirst[n] + *socket];
                if (link.FromNode == NoNode && link.Constant) {
                    add(n, port.Name, in->Type, OpSlotKind::Constant, {link.Constant->Words.begin(), link.Constant->Words.end()});
                } else if (link.FromNode == NoNode) {
                    auto [kind, initial] = literal(n, port.Name, in->Type);
                    add(n, port.Name, in->Type, kind, std::move(initial));
                    placed.push_back(port.Name);
                }
            }
            for (uint16_t s = 0; s < definition->Inputs && !definition->Stored; ++s) {
                const auto &link = plan.Inputs[plan.InputFirst[n] + s];
                if (link.FromNode == NoNode || IsInlineOp(r, plan, link.FromNode)) continue;
                const auto type = *OutputType(r, plan, layout, link.FromNode, link.FromSocket);
                if (type != mx::Type::Filename) add(n, definition->Sockets[s].Name, type, OpSlotKind::Read, ReadDefault(type));
            }
        }
        for (const auto &[name, type] : op.Literals) {
            if (std::ranges::contains(placed, std::string_view{name})) continue;
            auto [kind, initial] = literal(n, name, type);
            add(n, name, type, kind, std::move(initial));
        }
    }
    return layout;
}


void LayoutOpValues(state::Registry &r, state::Entity mesh_entity, const CookPlan &plan, const OpLayout &layout) {
    auto &store = r.Resources.get<GpuBuffers>().OpValues;
    const auto *held = r.try_get<const OpValueBlock>(mesh_entity);
    if (layout.Slots.empty()) {
        if (held) r.remove<OpValueBlock>(mesh_entity);
        return;
    }
    const auto same = [&](const OpSlot &a, const OpLayout::Slot &b) { return a.Input == b.Input && a.Type == b.Type && a.Kind == b.Kind && a.Offset == b.Offset && std::ranges::equal(a.Node, PathOf(plan, b.Node)); };
    if (held && std::ranges::equal(held->Slots, layout.Slots, same)) {
        for (size_t i = 0; i < layout.Slots.size(); ++i) {
            const auto &slot = layout.Slots[i];
            Range range{held->Block.Offset + slot.Offset, uint32_t(layout.Initial[i].size())};
            if (slot.Kind != OpSlotKind::Literal && !std::ranges::equal(store.Get(range), layout.Initial[i])) store.Update(range, std::span<const uint32_t>{layout.Initial[i]});
        }
        return;
    }
    auto slots = layout.Slots | std::views::transform([&](const OpLayout::Slot &slot) { return OpSlot{{std::from_range, PathOf(plan, slot.Node)}, slot.Input, slot.Type, slot.Kind, slot.Offset}; }) | std::ranges::to<std::vector>();
    Reblock(r, mesh_entity, held, std::move(slots), layout.Words, [&](size_t i) { return std::span<const uint32_t>{layout.Initial[i]}; });
}

void LayoutDefinitionValues(state::Registry &r, state::Entity definition) {
    const auto *held = r.try_get<const OpValueBlock>(definition);
    std::vector<OpSlot> slots;
    uint32_t words = 0;
    for (const auto node : NodesOf(r, definition)) {
        if (const auto *op = r.try_get<const Op>(node)) {
            for (const auto &[name, type] : op->Literals) {
                slots.push_back({{node}, name, type, OpSlotKind::Literal, words});
                words += mx::Words(type);
            }
        }
    }
    if (slots.empty()) {
        if (held) r.remove<OpValueBlock>(definition);
        return;
    }
    if (held && held->Slots == slots) return;
    Reblock(r, definition, held, std::move(slots), words, [](size_t) { return std::span<const uint32_t>{}; });
}

void LayoutNetworkValues(state::Registry &r, state::Entity network) {
    if (r.all_of<SubnetDefinition>(network)) LayoutDefinitionValues(r, network);
    else {
        const auto plan = Compile(r, network);
        LayoutOpValues(r, network, plan, LayoutOf(r, plan));
    }
}

std::span<const uint32_t> LiteralWords(const state::Registry &r, state::Entity node, std::string_view input, mx::Type type) {
    const auto &literals = r.get<const Op>(node).Literals;
    if (const auto literal = literals.find(input); literal == literals.end() || literal->second != type) return {};
    if (const auto *library = r.Resources.find<LibraryLiterals>()) {
        const auto found = library->Words.find(std::pair{node, std::string{input}});
        return found == library->Words.end() ? std::span<const uint32_t>{} : std::span<const uint32_t>{found->second};
    }
    const auto *block = r.try_get<const OpValueBlock>(NetworkOf(r, node));
    const auto *slot = block ? FindOpSlot(*block, NodePath{node}, input, type, OpSlotKind::Literal) : nullptr;
    return slot ? r.Resources.get<const GpuBuffers>().OpValues.Get({block->Block.Offset + slot->Offset, mx::Words(type)}) : std::span<const uint32_t>{};
}

OpLiteralWords LiteralsOf(const state::Registry &r, state::Entity node) {
    OpLiteralWords words;
    for (const auto &[name, type] : r.get<const Op>(node).Literals) {
        const auto held = LiteralWords(r, node, name, type);
        words.emplace(name, std::pair{type, held.empty() ? std::vector<uint32_t>(mx::Words(type)) : std::vector<uint32_t>{std::from_range, held}});
    }
    return words;
}
void CopyLiterals(state::Registry &r, const state::Registry &from, const NodeCopies &copies) {
    for (const auto [original, copy] : copies.Items)
        if (r.all_of<Op>(copy)) WriteLiterals(r, copy, LiteralsOf(from, original));
}
void WriteLiterals(state::Registry &r, state::Entity node, const OpLiteralWords &words) {
    const auto &literals = r.get<const Op>(node).Literals;
    if (literals.empty()) return;
    const auto &block = r.get<const OpValueBlock>(NetworkOf(r, node));
    auto &store = r.Resources.get<GpuBuffers>().OpValues;
    for (const auto &[name, type] : literals) {
        const auto written = words.find(name);
        if (written == words.end() || written->second.first != type) continue;
        const auto *slot = FindOpSlot(block, NodePath{node}, name, type, OpSlotKind::Literal);
        Range range{block.Block.Offset + slot->Offset, mx::Words(type)};
        store.Update(range, std::span<const uint32_t>{written->second.second});
    }
}

// Whether programs read the plan node as an image: a filter of a field that varies, unless it is in `unheld`.
bool ReadsAsImage(const state::Registry &r, const CookPlan &plan, const OpLayout &layout, uint32_t node, std::span<const uint32_t> unheld) {
    const auto &resolved = layout.Nodes[node];
    if (!resolved.Declaration || resolved.Outputs.front() == mx::Type::Filename) return false;
    const auto &definition = *DefinitionOf(OpOf(r, plan, node).Node);
    return definition.Filter && NodeSockets(plan, node)[definition.Inputs].Varies && !std::ranges::contains(unheld, node);
}
// The first texture coordinate node in plan order that `read` marks, or none.
uint32_t FirstTexcoord(const state::Registry &r, const CookPlan &plan, const std::vector<bool> &read) {
    const auto texcoord = std::ranges::find_if(plan.Order, [&](uint32_t n) { return read[n] && IsOp(plan, n) && OpOf(r, plan, n).Node == "texcoord"; });
    return texcoord == plan.Order.end() ? NoNode : *texcoord;
}
// Marks the op nodes the outputs read in `read`, through every input but an image input.
// A filter that programs read as an image reads only the first texture coordinate node its input reads, at which programs read the image.
void MarkRead(const state::Registry &r, const CookPlan &plan, const OpLayout &layout, std::span<const uint32_t> outputs, std::vector<bool> &read, std::span<const uint32_t> unheld) {
    for (std::vector<uint32_t> stack{std::from_range, outputs}; !stack.empty();) {
        const auto n = stack.back();
        stack.pop_back();
        if (read[n]) continue;
        read[n] = true;
        if (ReadsAsImage(r, plan, layout, n, unheld)) {
            std::vector<bool> input(plan.Nodes.size());
            MarkRead(r, plan, layout, std::array{plan.Inputs[plan.InputFirst[n]].FromNode}, input, unheld);
            if (const auto texcoord = FirstTexcoord(r, plan, input); texcoord != NoNode) stack.push_back(texcoord);
            continue;
        }
        const auto sockets = NodeSockets(plan, n);
        for (uint16_t s = 0; s < InputsOf(plan, n); ++s) {
            const auto from = plan.Inputs[plan.InputFirst[n] + s].FromNode;
            if (IsInlineOp(r, plan, from) && sockets[s].Types != TypeBit(DataType::Image)) stack.push_back(from);
        }
    }
}

bool ReadsOneCoordinateSet(const state::Registry &r, const CookPlan &plan, const OpLayout &layout, uint32_t node, std::span<const uint32_t> unheld) {
    const auto &lib = mx::Libraries();
    std::vector<bool> input(plan.Nodes.size());
    MarkRead(r, plan, layout, std::array{plan.Inputs[plan.InputFirst[node]].FromNode}, input, unheld);
    std::optional<uint32_t> set;
    bool defaults = false;
    for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
        if (!input[n] || !IsOp(plan, n)) continue;
        const auto &op = OpOf(r, plan, n);
        const auto &definition = *DefinitionOf(op.Node);
        const auto unset = [&](std::string_view name) {
            const auto &link = plan.Inputs[plan.InputFirst[n] + *InputSocket(definition, name)];
            return link.FromNode == NoNode && !link.Constant && !op.Literals.contains(name);
        };
        if (op.Node == "texcoord") {
            const auto &link = plan.Inputs[plan.InputFirst[n] + *InputSocket(definition, "index")];
            // An index an op gives is known only to the cook.
            if (link.FromNode != NoNode) return false;
            const auto index = link.Constant ? link.Constant->Words[0] : unset("index") ? 0u : LiteralWords(Authored(r, plan.Nodes[n]), plan.Nodes[n].Entity, "index", mx::Type::Integer)[0];
            if (set && *set != index) return false;
            set = index;
        } else if (const auto declaration = layout.Nodes[n].Declaration) {
            for (const auto &port : lib.Declarations[*declaration].Inputs) defaults |= port.DefaultGeomProp && lib.Declarations[lib.GeomProps[*port.DefaultGeomProp].Declaration].Node == "texcoord" && unset(port.Name);
        }
    }
    return !set || *set == 0u || !defaults;
}

OpGraph GraphOf(const state::Registry &r, const CookPlan &plan, const OpLayout &layout, std::span<const std::pair<uint32_t, uint16_t>> outputs, std::span<const uint32_t> unheld) {
    std::vector<bool> read(plan.Nodes.size());
    MarkRead(r, plan, layout, std::vector<uint32_t>{std::from_range, outputs | std::views::keys}, read, unheld);
    OpGraph graph;
    std::vector<uint32_t> index(plan.Nodes.size(), NoNode);
    for (const auto n : plan.Order) {
        if (!read[n]) continue;
        const auto &op = OpOf(r, plan, n);
        const auto *definition = DefinitionOf(op.Node);
        const auto &resolved = layout.Nodes[n];
        ops::Node node{.Output = op.Output};
        if (resolved.Declaration) node.Op = *resolved.Declaration;
        else node.Op = op.Node;
        // A filter that holds an image of a field that varies reads it at the coordinates of the first texture coordinate node its input reads.
        // A filter of a single value, and one in `unheld`, passes its input.
        const auto output_type = resolved.Outputs.empty() ? mx::Type::MultiOutput : resolved.Outputs.front();
        const bool filter = definition->Filter && resolved.Declaration && output_type != mx::Type::Filename;
        const bool image = ReadsAsImage(r, plan, layout, n, unheld);
        if (image) {
            node = ImageRead(output_type, uint32_t(graph.Images.size()));
            graph.Images.push_back({.From = n, .Socket = definition->Inputs});
            std::vector<bool> input(plan.Nodes.size());
            MarkRead(r, plan, layout, std::array{plan.Inputs[plan.InputFirst[n]].FromNode}, input, unheld);
            if (const auto texcoord = FirstTexcoord(r, plan, input); texcoord != NoNode) node.Inputs.push_back({"texcoord", ops::Link{index[texcoord], 0u}});
        } else if (filter) {
            node.Op = std::string{"dot"};
        }
        // The node's slots, which start at its first word.
        const auto first = std::ranges::lower_bound(layout.Slots, resolved.Values, {}, &OpLayout::Slot::Offset);
        const auto slots = std::ranges::subrange{first, std::ranges::find_if(first, layout.Slots.end(), [&](const OpLayout::Slot &slot) { return slot.Node != n; })};
        // The node's slot of the input: its read's default when `read`, and else its value.
        const auto slot_of = [&](std::string_view input, bool read) {
            const auto slot = std::ranges::find_if(slots, [&](const OpLayout::Slot &slot) { return slot.Input == input && (slot.Kind == OpSlotKind::Read) == read; });
            return slot == slots.end() ? nullptr : &*slot;
        };
        for (uint16_t s = 0; s < (image ? 0u : filter ? 1u : definition->Inputs); ++s) {
            const auto &link = plan.Inputs[plan.InputFirst[n] + s];
            const std::string name{filter ? "in" : definition->Sockets[s].Name};
            // Every image input reads a slot of the image table, which holds an image output, the input's texture, or none.
            if (definition->Sockets[s].Types == TypeBit(DataType::Image)) {
                node.Inputs.push_back({name, ops::Image{uint32_t(graph.Images.size())}});
                const auto texture = op.Images.find(name);
                if (link.FromNode != NoNode) graph.Images.push_back({.From = link.FromNode, .Socket = link.FromSocket});
                else graph.Images.push_back({.Texture = texture != op.Images.end() ? texture->second : state::Null});
                continue;
            }
            if (link.FromNode != NoNode) {
                if (IsInlineOp(r, plan, link.FromNode)) {
                    const auto port = layout.Nodes[link.FromNode].Ports.empty() ? 0u : layout.Nodes[link.FromNode].Ports[link.FromSocket - DefinitionOf(OpOf(r, plan, link.FromNode).Node)->Inputs];
                    node.Inputs.push_back({name, ops::Link{index[link.FromNode], port}});
                    continue;
                }
                const auto type = OutputType(r, plan, layout, link.FromNode, link.FromSocket);
                if (!type) continue;
                // A read of a stored field, whose default is the node's Read slot of the input.
                node.Inputs.push_back({name, ops::Link{uint32_t(graph.Graph.Nodes.size()), 0u}});
                graph.Reads.push_back({uint32_t(graph.Graph.Nodes.size()), link.FromNode, link.FromSocket});
                auto &read = graph.Graph.Nodes.emplace_back(ops::Node{.Op = std::string{"geompropvalue"}, .Output = *type, .Inputs = {{"geomprop", ops::Literal{mx::Type::String, std::string{}}}}});
                if (const auto *slot = slot_of(name, true)) read.Inputs.push_back({"default", ops::Word{slot->Type, slot->Offset}});
                graph.Nodes.push_back(n);
            } else if (const auto literal = op.Literals.find(name); link.Constant || literal != op.Literals.end()) {
                const auto type = link.Constant ? ValueTypeOf(link.Constant->Type) : literal->second;
                const auto *slot = slot_of(name, false);
                // A node the layout does not resolve has no slot of a constant, and its program faults.
                node.Inputs.push_back({name, slot ? ops::Source{ops::Word{type, slot->Offset}} : ops::Literal{type, std::vector<uint32_t>(mx::Words(type))}});
            }
        }
        // Each input the node leaves out reads its default from the node's slot of it.
        if (!image && !filter) {
            for (const auto &slot : slots)
                if (slot.Kind == OpSlotKind::Default) node.Inputs.push_back({slot.Input, ops::Word{slot.Type, slot.Offset}});
        }
        // A filter's reader takes none of its texts.
        if (!filter) {
            for (const auto &[name, text] : op.Texts)
                if (const auto type = TextType(*definition, name)) node.Inputs.push_back({name, ops::Literal{*type, text}});
        }
        index[n] = uint32_t(graph.Graph.Nodes.size());
        graph.Graph.Nodes.push_back(std::move(node));
        graph.Nodes.push_back(n);
    }
    for (const auto [n, socket] : outputs) {
        const auto &resolved = layout.Nodes[n];
        const auto output = socket - DefinitionOf(OpOf(r, plan, n).Node)->Inputs;
        graph.Graph.Outputs.push_back({index[n], resolved.Ports.empty() ? 0u : resolved.Ports[output]});
    }
    graph.Base = Rebase(graph.Graph);
    return graph;
}

void KeepFaults(state::Registry &r, const CookPlan &plan, const OpGraph &graph, std::span<const ops::NodeFault> faults) {
    for (const auto &[node, fault] : faults) Keep(r, PathOf(plan, graph.Nodes[node]), NodeError{FaultOf(fault)});
}

OpImageTexture ImageTextureOf(MTL::Texture *texture) { return {texture->gpuResourceID()._impl}; }
ImageValue DocumentImage(const state::Registry &r, state::Entity texture) {
    const auto *materialized = r.valid(texture) ? r.try_get<const MaterializedTexture>(texture) : nullptr;
    const auto &textures = r.Resources.get<const TextureStore>().Textures;
    const auto entry = materialized ? std::ranges::find(textures, materialized->SamplerSlot, &TextureEntry::SamplerSlot) : textures.end();
    return entry == textures.end() ? NoImage : ImageValue{ImageTextureOf(*entry->Image), entry->Image.Extent.Width, entry->Image.Extent.Height, 4u};
}
ops::Node ImageRead(mx::Type type, uint32_t image) {
    return {
        .Op = *mx::FindDeclaration(mx::Libraries(), "image", {}, type),
        .Inputs = {
            {"file", ops::Image{image}},
            {"default", ops::Literal{type, std::vector<uint32_t>(mx::Words(type))}},
            {"uaddressmode", ops::Literal{mx::Type::String, std::string{"clamp"}}},
            {"vaddressmode", ops::Literal{mx::Type::String, std::string{"clamp"}}},
            {"frameoffset", ops::Literal{mx::Type::Integer, std::vector<uint32_t>{0u}}},
        },
    };
}

uint32_t Rebase(ops::Graph &graph) {
    std::optional<uint32_t> base;
    for (const auto &node : graph.Nodes) {
        for (const auto &input : node.Inputs)
            if (const auto *word = std::get_if<ops::Word>(&input.Source)) base = std::min(base.value_or(word->Offset), word->Offset);
    }
    for (auto &node : graph.Nodes) {
        for (auto &input : node.Inputs)
            if (auto *word = std::get_if<ops::Word>(&input.Source)) word->Offset -= *base;
    }
    return base.value_or(0u);
}

void WriteOpLiteral(state::Registry &r, state::Entity node, std::string_view input, mx::Type type, std::span<const uint32_t> words) {
    if (auto *library = r.Resources.find<LibraryLiterals>()) {
        r.patch<Op>(node, [&](Op &op) { op.Literals.insert_or_assign(std::string{input}, type); });
        library->Words.insert_or_assign(std::pair{node, std::string{input}}, std::vector<uint32_t>{std::from_range, words});
        return;
    }
    const auto network = NetworkOf(r, node);
    const auto literal = r.get<const Op>(node).Literals.find(input);
    const bool held = literal != r.get<const Op>(node).Literals.end() && literal->second == type;
    const auto *block = r.try_get<const OpValueBlock>(network);
    const auto *slot = held && block ? FindOpSlot(*block, NodePath{node}, input, type, OpSlotKind::Literal) : nullptr;
    const bool definition = r.all_of<SubnetDefinition>(network);
    if (!slot) {
        if (!held) r.patch<Op>(node, [&](Op &op) { op.Literals.insert_or_assign(std::string{input}, type); });
        LayoutNetworkValues(r, network);
        block = &r.get<const OpValueBlock>(network);
        slot = FindOpSlot(*block, NodePath{node}, input, type, OpSlotKind::Literal);
    }
    auto &store = r.Resources.get<GpuBuffers>().OpValues;
    Range range{block->Block.Offset + slot->Offset, uint32_t(words.size())};
    store.Update(range, words);
    // Each mesh whose plan instances the definition, and each material graph, holds a copy of the literal for each occurrence of the node.
    if (definition) {
        const auto update_copies = [&](const OpValueBlock *held_block) {
            for (const auto &copy : held_block ? std::span{held_block->Slots} : std::span<const OpSlot>{}) {
                if (copy.Kind != OpSlotKind::Copy || copy.Node.back() != node || copy.Input != input || copy.Type != type) continue;
                Range copied{held_block->Block.Offset + copy.Offset, uint32_t(words.size())};
                store.Update(copied, words);
            }
        };
        for (const auto [mesh, plan] : r.view<const CookPlan>().each()) update_copies(std::ranges::binary_search(plan.Definitions, network) ? r.try_get<const OpValueBlock>(mesh) : nullptr);
        for (const auto [material, block] : r.view<const MaterialGraph, const OpValueBlock>().each()) update_copies(&block);
    }
    if (definition || r.all_of<MaterialGraph>(network)) NoteMaterialValues(r);
    InvalidateNode(r, node);
}
} // namespace procedure

namespace action::nodes {
using namespace procedure;
namespace {
// Unlinks each authored link that passes an op output to an input that cannot take it, since the output varies.
// The link is the one in the innermost network holding both the output's occurrence and the input's occurrence.
// A link of a user definition is unlinked only when it fails in each of its occurrences, and an occurrence it fails alone faults when it cooks.
// The plans of the meshes that use the network decide, or the network's own plan when no mesh uses it.
void DropFieldLinks(state::Registry &r, state::Entity network) {
    auto owners = MeshesUsing(r, network);
    if (owners.empty()) owners.push_back(network);
    const auto plans = owners | std::views::transform([&](state::Entity owner) { return Plan(r, owner); }) | std::ranges::to<std::vector>();
    // Each failing link by its node and slot, with the plan and node occurrence it fails in.
    std::vector<std::tuple<state::Entity, uint16_t, size_t, uint32_t>> failing;
    for (size_t p = 0; p < plans.size(); ++p) {
        const auto &plan = plans[p];
        const auto ancestors = [&](uint32_t n) {
            std::vector<uint32_t> enclosing;
            for (auto parent = plan.Nodes[n].Parent; parent != NoNode; parent = plan.Nodes[parent].Parent) enclosing.push_back(parent);
            return enclosing;
        };
        for (uint32_t n = 0; n < plan.Nodes.size(); ++n) {
            if (plan.Nodes[n].Kind == KindOf<Subnet>) continue;
            for (uint16_t slot = 0; slot < InputsOf(plan, n); ++slot) {
                const auto link = plan.Inputs[plan.InputFirst[n] + slot];
                if (link.FromNode == NoNode || plan.Nodes[link.FromNode].Kind != KindOf<Op>) continue;
                if (Links(NodeSockets(plan, link.FromNode)[link.FromSocket], NodeSockets(plan, n)[SlotSocket(plan.Nodes[n].Kind, slot)])) continue;
                // The innermost subnet node occurrence enclosing both ends, whose network holds the link.
                const auto source = ancestors(link.FromNode);
                auto enclosing = NoNode;
                for (const auto a : ancestors(n)) {
                    if (std::ranges::contains(source, a)) {
                        enclosing = a;
                        break;
                    }
                }
                // Each network inside it reads the link through its definition's input port.
                auto node = n;
                auto held = slot;
                while (plan.Nodes[node].Parent != enclosing) {
                    held = InputOf(Authored(r, plan.Nodes[node]), plan.Nodes[node].Entity, held).FromSocket;
                    node = plan.Nodes[node].Parent;
                }
                if (!plan.Nodes[node].Builtin) failing.emplace_back(plan.Nodes[node].Entity, held, p, node);
            }
        }
    }
    SortUnique(failing);
    std::vector<std::pair<state::Entity, uint16_t>> dropped;
    for (auto at = failing.begin(); at != failing.end();) {
        const auto [entity, slot, p, n] = *at;
        const auto end = std::ranges::find_if(at, failing.end(), [&](const auto &f) { return std::get<0>(f) != entity || std::get<1>(f) != slot; });
        const auto occurrences = std::ranges::fold_left(plans, size_t{0}, [&](size_t count, const CookPlan &plan) {
            return count + size_t(std::ranges::count_if(plan.Nodes, [&](const NodeRecord &node) { return !node.Builtin && node.Entity == entity; }));
        });
        if (size_t(end - at) == occurrences) dropped.emplace_back(entity, slot);
        at = end;
    }
    // A multi-input socket's later slots move down when an earlier one unlinks.
    for (const auto &[entity, slot] : dropped | std::views::reverse) SetInput(r, entity, slot, {});
}
} // namespace

void Apply(state::Registry &r, const SetOpInput &a) {
    const auto *op = r.try_get<const Op>(a.Node);
    if (!op) return Fail(r, "The node is not an op");
    const auto &definition = *DefinitionOf(op->Node);
    const auto socket = InputSocket(definition, a.Input);
    std::visit(
        overloaded{
            [&](const OpLiteral &literal) {
                const auto type = DataTypeOf(literal.Type);
                if (!socket || !type || !mx::Words(literal.Type) || !(definition.Sockets[*socket].Types & TypeBit(*type)) || literal.Words.size() != mx::Words(literal.Type)) return Fail(r, "Set the input to a literal of a type it takes");
                // An input that holds a texture reads its literal in place of the texture.
                if (op->Images.contains(a.Input)) r.patch<Op>(a.Node, [&](Op &edited) { edited.Images.erase(a.Input); });
                WriteOpLiteral(r, a.Node, a.Input, literal.Type, literal.Words);
            },
            [&](const std::string &text) {
                if (!TextType(definition, a.Input)) return Fail(r, "The op has no string or filename input of this name");
                // An input with choices takes one that a declaration of the node lists.
                const auto &lib = mx::Libraries();
                bool chooses = false, listed = false;
                for (const auto d : mx::NodeDeclarations(lib, op->Node)) {
                    const auto &ports = lib.Declarations[d].Inputs;
                    const auto port = std::ranges::find(ports, a.Input, &mx::Port::Name);
                    if (port == ports.end() || port->EnumNames.empty()) continue;
                    chooses = true;
                    listed |= std::ranges::contains(port->EnumNames, text);
                }
                if (chooses && !listed) return Fail(r, "Set the input to one of its choices");
                r.patch<Op>(a.Node, [&](Op &edited) { edited.Texts.insert_or_assign(a.Input, text); });
            },
            [&](state::Entity texture) {
                if (!socket || !(definition.Sockets[*socket].Types & TypeBit(DataType::Image))) return Fail(r, "The op has no image input of this name");
                if (texture != state::Null && !(r.valid(texture) && r.all_of<MaterializedTexture>(texture))) return Fail(r, "Choose a texture of the document");
                r.patch<Op>(a.Node, [&](Op &edited) {
                    if (texture == state::Null) edited.Images.erase(a.Input);
                    else edited.Images.insert_or_assign(a.Input, texture);
                });
            },
        },
        *a.Value
    );
}
void Apply(state::Registry &r, const SetOpOutputType &a) {
    const auto *op = r.try_get<const Op>(a.Node);
    if (!op) return Fail(r, "The node is not an op");
    const auto &definition = *DefinitionOf(op->Node);
    if (!definition.StoresOutput || !std::ranges::contains(definition.OutputTypes, a.Type)) return Fail(r, "The op's declarations have no output of this type");
    r.patch<Op>(a.Node, [&](Op &edited) { edited.Output = a.Type; });
    DropFieldLinks(r, NetworkOf(r, a.Node));
}
void Apply(state::Registry &r, const ChangeOp &a) {
    const auto *op = r.try_get<const Op>(a.Node);
    const auto *definition = DefinitionOf(a.Op);
    if (!op) return Fail(r, "The node is not an op");
    if (!definition || definition->Sockets.size() == definition->Inputs) return Fail(r, "Name a MaterialX node with a value output");
    const auto *prior = DefinitionOf(op->Node);
    const auto network = NetworkOf(r, a.Node);
    const auto &lib = mx::Libraries();
    Op changed{.Node = a.Op};
    if (definition->StoresOutput) changed.Output = op->Output && std::ranges::contains(definition->OutputTypes, *op->Output) ? *op->Output : lib.Declarations[mx::NodeDeclarations(lib, a.Op).front()].Outputs.front().Type;
    for (const auto &[name, type] : op->Literals) {
        const auto socket = InputSocket(*definition, name);
        if (socket && definition->Sockets[*socket].Types & TypeBit(*DataTypeOf(type))) changed.Literals.emplace(name, type);
    }
    for (const auto &[name, text] : op->Texts)
        if (TextType(*definition, name)) changed.Texts.emplace(name, text);
    for (const auto &[name, texture] : op->Images) {
        const auto socket = InputSocket(*definition, name);
        if (socket && definition->Sockets[*socket].Types & TypeBit(DataType::Image)) changed.Images.emplace(name, texture);
    }
    const auto prior_links = r.get<const NodeInputs>(a.Node).Links;
    std::vector<SocketLink> links(definition->Inputs);
    for (uint16_t s = 0; s < definition->Inputs; ++s) {
        const auto before = InputSocket(*prior, definition->Sockets[s].Name);
        const auto link = before ? prior_links[*before] : SocketLink{};
        if (link.FromNode != state::Null && Links(SocketOf(r, link.FromNode, link.FromSocket), definition->Sockets[s])) links[s] = link;
    }
    // Each link from an output moves to the changed op's output of the same name, or is removed.
    const auto moved = [&](SocketLink link) -> std::optional<SocketLink> {
        if (link.FromNode != a.Node) return {};
        const std::string_view name{prior->Sockets[link.FromSocket].Name};
        for (uint16_t s = definition->Inputs; s < definition->Sockets.size(); ++s)
            if (definition->Sockets[s].Name == name) return SocketLink{a.Node, s};
        return SocketLink{};
    };
    r.replace<Op>(a.Node, std::move(changed));
    r.replace<NodeInputs>(a.Node, std::move(links));
    ForEachNetworkInput(r, network, [&](state::Entity node, uint16_t slot, SocketLink link) {
        if (const auto to = moved(link)) SetInput(r, node, slot, *to);
    });
    DropFieldLinks(r, network);
}
} // namespace action::nodes
