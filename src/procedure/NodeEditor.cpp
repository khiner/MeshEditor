#include "procedure/NodeEditor.h"
#include "SortUnique.h"
#include "procedure/Macros.h"
#include "scene/Entity.h"
#include "scene/HierarchyInstance.h"
#include "scene/SceneGraph.h"

#include "action/Bone.h"
#include "action/Selection.h"
#include "action/View.h"
#include "armature/ArmatureComponents.h"
#include "mesh/Mesh.h"
#include "metal/ImGuiTexture.h"
#include "metal/MetalContext.h"
#include "procedure/Cook.h"
#include "procedure/MaterialGraphs.h"
#include "procedure/NodeTheme.h"
#include "procedure/NodeView.h"
#include "procedure/OpNodes.h"
#include "render/Instance.h"
#include "ui/ChoiceCombo.h"
#include "ui/FieldEdit.h"
#include "ui/InterfaceUi.h"
#include "viewport/FrameState.h"
#include "viewport/ViewportConsumerFence.h"

#include <imgui_internal.h>
#include <imgui_stdlib.h>

#include <bit>
#include <format>
#include <numeric>

namespace procedure {
using namespace ImGui;
using numeric::uvec2;
using numeric::vec2;

namespace {
constexpr float WireStraightening{3.5f};
constexpr float MinZoom{.25f}, MaxZoom{3.f};
constexpr float TrackpadPoints{10.f}; // The platform reports a trackpad's scroll at a tenth of its points.
constexpr float ZoomStep{1.1f}, TrackpadStepPoints{100.f}; // A zoom step is one mouse wheel notch, or 100 points of trackpad scroll.

constexpr const char *GroupNames[]{"Sources", "Operators", "Combine", "Attributes", "Math", "Generators", "Instances"};
using C = NodeThemeColors;
constexpr vec4 C::*GroupColors[]{&C::SourcesTitle, &C::OperatorsTitle, &C::CombineTitle, &C::AttributesTitle, &C::MathTitle, &C::GeneratorsTitle, &C::InstancesTitle};
constexpr auto KindGroups{[]<typename... Ks>(TypeList<Ks...>) { return std::array{Kind<Ks>.Group...}; }(NodeKinds{})};
constexpr auto KindTitles{[]<typename... Ks>(TypeList<Ks...>) { return std::array{field::TypeLabel<Ks>.c_str()...}; }(NodeKinds{})};
constexpr auto KindHelp{[]<typename... Ks>(TypeList<Ks...>) { return std::array{Kind<Ks>.Help...}; }(NodeKinds{})};
static_assert(std::ranges::none_of(KindHelp, [](const char *help) { return help == nullptr; }), "Every node kind needs a Help sentence.");

constexpr const char *FaultTexts[]{
    "The selection does not fit this operation.",
    "A parameter is invalid.",
    "An attribute input is missing or has the wrong type or domain.",
    "An input mesh is not a closed surface.",
    "A region input leaves the first region's plane.",
    "The boolean produced no geometry.",
    "The node reads no mesh: the Object Info object has none, or the Instance input is unlinked.",
    "No declaration of this node takes its input types.",
    "The node's output is not a value.",
    "The node reads what its consumer's domain lacks.",
    "The node blurs a field of one set of texture coordinates, and its input reads more, so the node passes it unblurred.",
    "The node reads what several domains store, none of them its consumer's. Add a Convert Domain node to choose the domain it reads on.",
    "The output has no attribute type on its consumer's domain.",
    "The consumer reads more outputs than one evaluation writes.",
    "A consumer input does not take the output's type.",
    "No attribute of this name and type. The node outputs its default.",
};
static_assert(std::size(FaultTexts) == field::EnumCount<NodeFault>);
// The fault's text, which names the attribute a geompropvalue op does not find.
std::string FaultText(const state::Registry &a, state::Entity node, NodeFault fault) {
    if (fault != NodeFault::NotFound) return FaultTexts[uint8_t(fault)];
    const auto &texts = a.get<const Op>(node).Texts;
    const auto name = texts.find("geomprop");
    return std::format("No attribute \"{}\" of this type. The node outputs its default.", name != texts.end() ? name->second : std::string{});
}

constexpr const char *DataTypeNames[]{"Float", "2D Vector", "Vector", "4D Vector", "Color", "Color with Alpha", "3x3 Matrix", "4x4 Matrix", "Integer", "Boolean", "Geometry", "Instances", "Image", "Reference"};
static_assert(std::size(DataTypeNames) == field::EnumCount<DataType>);
const char *LabelOf(Socket socket) {
    if (socket.Name) return socket.Name;
    if (socket.Selection) return "Selection";
    if (socket.Types & NumericTypes) return socket.Varies ? "Attribute" : "Value";
    return DataTypeNames[std::countr_zero(socket.Types)];
}
constexpr vec4 C::*TypeColors[]{&C::FloatSocket, &C::MixedSocket, &C::VectorSocket, &C::Vector4Socket, &C::ColorSocket, &C::ColorSocket, &C::MatrixSocket, &C::MatrixSocket, &C::IntegerSocket, &C::BooleanSocket, &C::GeometrySocket, &C::InstancesSocket, &C::ImageSocket, &C::MixedSocket};
static_assert(std::size(TypeColors) == field::EnumCount<DataType>);
vec4 SocketColor(const C &colors, DataTypes types) { return std::has_single_bit(types) ? colors.*TypeColors[std::countr_zero(types)] : colors.MixedSocket; }

// A MaterialX name as the editor shows it: its words split at underscores and capitalized, with a trailing dimension such as "3d" written " 3D".
std::string Presented(std::string_view name) {
    const auto digit = [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; };
    std::string text;
    bool start = true;
    for (size_t i = 0; i < name.size(); ++i) {
        const auto c = name[i];
        if (c == '_') {
            text += ' ';
            start = true;
        } else if (i > 0u && i + 2u == name.size() && digit(c) && !digit(name[i - 1u]) && name[i + 1u] == 'd') {
            text += std::format(" {}D", c);
            break;
        } else {
            text += start ? char(std::toupper(static_cast<unsigned char>(c))) : c;
            start = false;
        }
    }
    return text;
}
// A MaterialX type as the editor shows it, such as "Vector3".
const char *TypeTitle(mx::Type type) {
    static const auto titles = [] {
        std::array<std::string, size_t(mx::Type::MultiOutput) + 1u> titles;
        for (size_t t = 0; t < titles.size(); ++t) titles[t] = Presented(mx::Name(mx::Type(t)));
        return titles;
    }();
    return titles[size_t(type)].c_str();
}

// The title of each MaterialX node whose name presented as a title does not read as one, in node name order.
constexpr std::pair<std::string_view, std::string_view> OpTitles[]{
    {"absval", "Absolute"},
    {"acescg_to_lin_rec709", "ACEScg to Linear Rec. 709"},
    {"acos", "Arccosine"},
    {"adobergb_to_lin_rec709", "Adobe RGB to Linear Rec. 709"},
    {"asin", "Arcsine"},
    {"atan2", "Arctangent 2"},
    {"bitangent", "Surface Bitangent"},
    {"ceil", "Ceiling"},
    {"cellnoise2d", "Cell Noise 2D"},
    {"cellnoise3d", "Cell Noise 3D"},
    {"colorcorrect", "Color Correct"},
    {"combine2", "Combine 2"},
    {"combine3", "Combine 3"},
    {"combine4", "Combine 4"},
    {"convertdomain", "Convert Domain"},
    {"cos", "Cosine"},
    {"cosh", "Hyperbolic Cosine"},
    {"creatematrix", "Create Matrix"},
    {"crossproduct", "Cross Product"},
    {"degrees", "To Degrees"},
    {"disjointover", "Disjoint Over"},
    {"dotproduct", "Dot Product"},
    {"edgeloop", "Edge Loop"},
    {"edgering", "Edge Ring"},
    {"elementindex", "Element Index"},
    {"exp", "Exponent"},
    {"facingratio", "Facing Ratio"},
    {"fract", "Fraction"},
    {"g18_rec709_to_lin_rec709", "Gamma 1.8 Rec. 709 to Linear Rec. 709"},
    {"g22_ap1_to_lin_rec709", "Gamma 2.2 AP1 to Linear Rec. 709"},
    {"g22_rec709_to_lin_rec709", "Gamma 2.2 Rec. 709 to Linear Rec. 709"},
    {"geomcolor", "Geometry Color"},
    {"geompropvalue", "Geometric Property"},
    {"geompropvalueuniform", "Uniform Geometric Property"},
    {"heighttonormal", "Height to Normal"},
    {"hextiledimage", "Hex Tiled Image"},
    {"hextilednormalmap", "Hex Tiled Normal Map"},
    {"hsvadjust", "HSV Adjust"},
    {"hsvtorgb", "HSV to RGB"},
    {"ifequal", "If Equal"},
    {"ifgreater", "If Greater"},
    {"ifgreatereq", "If Greater or Equal"},
    {"inversesqrt", "Inverse Square Root"},
    {"invertmatrix", "Invert Matrix"},
    {"latlongimage", "Lat-Long Image"},
    {"lin_adobergb_to_lin_rec709", "Linear Adobe RGB to Linear Rec. 709"},
    {"lin_displayp3_to_lin_rec709", "Linear Display P3 to Linear Rec. 709"},
    {"ln", "Natural Logarithm"},
    {"max", "Maximum"},
    {"min", "Minimum"},
    {"multiplyadd", "Multiply Add"},
    {"nearlyequal", "Nearly Equal"},
    {"normalmap", "Normal Map"},
    {"pingpong", "Ping-Pong"},
    {"planeside", "Plane Side"},
    {"premult", "Premultiply"},
    {"radians", "To Radians"},
    {"ramp4", "Ramp 4"},
    {"ramplr", "Ramp Left Right"},
    {"ramptb", "Ramp Top Bottom"},
    {"randomcolor", "Random Color"},
    {"randomfloat", "Random Float"},
    {"realpower", "Real Power"},
    {"rec709_display_to_lin_rec709", "Rec. 709 Display to Linear Rec. 709"},
    {"rgbtohsv", "RGB to HSV"},
    {"roundhalfup", "Round Half Up"},
    {"safeacos", "Safe Arccosine"},
    {"safeasin", "Safe Arcsine"},
    {"safedivide", "Safe Divide"},
    {"safemodulo", "Safe Modulo"},
    {"safenormalize", "Safe Normalize"},
    {"safepower", "Safe Power"},
    {"safesqrt", "Safe Square Root"},
    {"selection", "Selection Attribute"},
    {"separate2", "Separate 2"},
    {"separate3", "Separate 3"},
    {"separate4", "Separate 4"},
    {"sin", "Sine"},
    {"sinh", "Hyperbolic Sine"},
    {"smoothmax", "Smooth Maximum"},
    {"smoothmin", "Smooth Minimum"},
    {"smoothstep", "Smooth Step"},
    {"splitlr", "Split Left Right"},
    {"splittb", "Split Top Bottom"},
    {"sqrt", "Square Root"},
    {"srgb_displayp3_to_lin_rec709", "sRGB Display P3 to Linear Rec. 709"},
    {"srgb_texture_to_lin_rec709", "sRGB Texture to Linear Rec. 709"},
    {"surface_unlit", "Unlit Surface"},
    {"surfacematerial", "Surface Material"},
    {"tan", "Tangent"},
    {"tangent", "Surface Tangent"},
    {"tanh", "Hyperbolic Tangent"},
    {"texcoord", "Texture Coordinate"},
    {"tiledcircles", "Tiled Circles"},
    {"tiledcloverleafs", "Tiled Cloverleafs"},
    {"tiledhexagons", "Tiled Hexagons"},
    {"tiledimage", "Tiled Image"},
    {"transformmatrix", "Transform Matrix"},
    {"transformnormal", "Transform Normal"},
    {"transformpoint", "Transform Point"},
    {"transformvector", "Transform Vector"},
    {"trianglewave", "Triangle Wave"},
    {"triplanarprojection", "Triplanar Projection"},
    {"truncatedmodulo", "Truncated Modulo"},
    {"unifiednoise2d", "Unified Noise 2D"},
    {"unifiednoise3d", "Unified Noise 3D"},
    {"unpremult", "Unpremultiply"},
    {"viewdirection", "View Direction"},
    {"volumematerial", "Volume Material"},
    {"worleynoise2d", "Worley Noise 2D"},
    {"worleynoise3d", "Worley Noise 3D"},
    {"xor", "Exclusive Or"},
};
static_assert(std::ranges::is_sorted(OpTitles, {}, &std::pair<std::string_view, std::string_view>::first));

// A MaterialX node with a value output as the editor shows it.
struct OpEntry {
    std::string_view Node;
    std::string Title; // The node's OpTitles entry, or else its name presented as a title
    std::vector<std::string> Labels; // By socket of the node's definition
    std::vector<std::string> TextLabels; // By text input of the node's definition
    uint32_t Group{NoNode}; // The group that offers it, or NoNode for a node no geometry network evaluates
};
struct OpGroup {
    std::string Title;
    std::vector<uint32_t> Ops; // In title order
};
// The ops a geometry network offers, grouped by MaterialX node group.
struct OpCatalog {
    std::vector<OpEntry> Ops; // In node name order
    std::vector<OpGroup> Groups; // In title order
};
// The declaration's input or output of the name, or null.
const mx::Port *FindPort(const mx::Declaration &declaration, std::string_view name, bool input) {
    const auto &ports = input ? declaration.Inputs : declaration.Outputs;
    const auto port = std::ranges::find(ports, name, &mx::Port::Name);
    return port == ports.end() ? nullptr : &*port;
}
const OpCatalog &Ops() {
    static const auto catalog = [] {
        const auto &lib = mx::Libraries();
        // The label of the input or output of a declaration of the node: its first UI name, or else its name.
        const auto label = [&](std::span<const uint32_t> declarations, std::string_view name, bool input) {
            for (const auto d : declarations) {
                if (const auto *port = FindPort(lib.Declarations[d], name, input); port && !port->UiName.empty()) return port->UiName;
            }
            return Presented(name);
        };
        OpCatalog catalog;
        for (const auto &[node, declarations] : lib.DeclarationsByNode) {
            const auto *definition = DefinitionOf(node);
            if (!definition || definition->Sockets.size() == definition->Inputs) continue;
            const auto titled = std::ranges::lower_bound(OpTitles, std::string_view{node}, {}, &std::pair<std::string_view, std::string_view>::first);
            OpEntry entry{.Node = node, .Title = titled != std::ranges::end(OpTitles) && titled->first == node ? std::string{titled->second} : Presented(node)};
            for (uint16_t s = 0; s < definition->Sockets.size(); ++s) {
                // An image input takes an image where MaterialX takes a file name.
                auto text = label(declarations, definition->Sockets[s].Name, s < definition->Inputs);
                entry.Labels.push_back(definition->Sockets[s].Types == TypeBit(DataType::Image) && text == "Filename" ? std::string{"Image"} : std::move(text));
            }
            for (const auto &text : definition->Texts) entry.TextLabels.push_back(label(declarations, text.first, true));
            if (definition->Domains) {
                const auto group = Presented(lib.Declarations[declarations.front()].NodeGroup);
                auto found = std::ranges::find(catalog.Groups, group, &OpGroup::Title);
                if (found == catalog.Groups.end()) found = catalog.Groups.insert(found, {group, {}});
                found->Ops.push_back(uint32_t(catalog.Ops.size()));
            }
            catalog.Ops.push_back(std::move(entry));
        }
        std::ranges::sort(catalog.Groups, {}, &OpGroup::Title);
        for (uint32_t g = 0; g < catalog.Groups.size(); ++g) {
            auto &ops = catalog.Groups[g].Ops;
            std::ranges::sort(ops, {}, [&](uint32_t op) -> const std::string & { return catalog.Ops[op].Title; });
            for (const auto op : ops) catalog.Ops[op].Group = g;
        }
        return catalog;
    }();
    return catalog;
}
const OpEntry *FindOp(std::string_view node) {
    const auto &ops = Ops().Ops;
    const auto found = std::ranges::lower_bound(ops, node, {}, &OpEntry::Node);
    return found != ops.end() && found->Node == node ? &*found : nullptr;
}
// The input or output of the resolved declaration, or else of the node's first declaration that has it.
const mx::Port *DeclaredPort(std::string_view node, std::optional<uint32_t> declaration, std::string_view name, bool input) {
    const auto &lib = mx::Libraries();
    if (const auto *port = declaration ? FindPort(lib.Declarations[*declaration], name, input) : nullptr) return port;
    for (const auto d : mx::NodeDeclarations(lib, node))
        if (const auto *port = FindPort(lib.Declarations[d], name, input)) return port;
    return nullptr;
}
// The doc text of the resolved declaration, or else of the node's first declaration that has one, or null.
const char *OpDoc(std::string_view node, std::optional<uint32_t> declaration) {
    const auto &lib = mx::Libraries();
    if (declaration && !lib.Declarations[*declaration].Doc.empty()) return lib.Declarations[*declaration].Doc.c_str();
    for (const auto d : mx::NodeDeclarations(lib, node))
        if (!lib.Declarations[d].Doc.empty()) return lib.Declarations[d].Doc.c_str();
    return nullptr;
}

std::array<std::string, 2u << field::EnumCount<DataType>> SocketTypeTexts;
// A socket's direction and data types, such as "Input, Float or Vector", built once per direction and types mask.
const char *SocketTypeText(Socket socket) {
    const bool input = socket.Direction == SocketDirection::In;
    auto &text = SocketTypeTexts[(input ? 0u : 1u << field::EnumCount<DataType>) + socket.Types];
    if (!text.empty()) return text.c_str();
    text = input ? "Input, " : "Output, ";
    if (socket.Types == NumericTypes) {
        text += "any type";
        return text.c_str();
    }
    for (uint32_t types = socket.Types; types != 0u;) {
        text += DataTypeNames[std::countr_zero(types)];
        types &= types - 1u;
        if (types != 0u) text += std::has_single_bit(types) ? " or " : ", ";
    }
    return text.c_str();
}

// The help of the parameter field behind each socket of kind K, or null for a fixed socket.
template<typename K> constexpr const char *FieldHelp(Socket socket) {
    const char *help{};
    K value{};
    VisitSocketParameter(value, socket, [&](auto &, const FieldSpec &spec) { help = spec.Help; });
    return help;
}
// Each socket's help, in Sockets order.
constexpr auto SocketHelps{[]<typename... Ks>(TypeList<Ks...>) {
    std::array<const char *, Sockets.size()> helps{};
    (std::ranges::transform(detail::SocketsOf<Ks>(), helps.begin() + SocketFirst[KindOf<Ks>], FieldHelp<Ks>), ...);
    return helps;
}(NodeKinds{})};

ImVec2 Im(vec2 v) { return {v.x, v.y}; }
using ui::U32;
void DrawSocket(ImDrawList &draw, const NodeTheme &theme, ImVec2 at, float radius, float multi_half, Socket socket, vec4 color, bool hovered) {
    const auto fill = U32(color), outline = U32(hovered ? theme.Colors.Hovered : theme.Colors.Outline);
    if (socket.Multi) {
        const ImVec2 half{radius * .8f, multi_half};
        draw.AddRectFilled(at - half, at + half, fill, half.x);
        draw.AddRect(at - half, at + half, outline, half.x, 0, theme.Sizes.OutlineWidth);
        return;
    }
    if (socket.Varies) {
        const float r = radius * 1.2f;
        draw.AddQuadFilled({at.x, at.y - r}, {at.x + r, at.y}, {at.x, at.y + r}, {at.x - r, at.y}, fill);
        draw.AddQuad({at.x, at.y - r}, {at.x + r, at.y}, {at.x, at.y + r}, {at.x - r, at.y}, outline, theme.Sizes.OutlineWidth);
        return;
    }
    draw.AddCircleFilled(at, radius, fill);
    draw.AddCircle(at, radius, outline, 0, theme.Sizes.OutlineWidth);
}

// A node shows inputs down its left column and outputs down its right, one per cell row under a one-cell title.
// Each socket is at the center of its own port cell, and the labels are in the columns between the two port columns.
// Tray sockets share one row, marked by a chevron, while the tray is closed.
// Each tray socket has its own row while the tray is open.
constexpr float MultiHalf{.38f}, MultiSpacing{.22f}; // A multi-input socket's half height and the maximum spacing between its links, in cells.
constexpr float CanvasMargin{.5f}; // The empty space left of and above cell zero, in cells.

// A grid cell's side in canvas points.
float CellOf(const NodeTheme &theme) { return theme.Sizes.Cell.FollowFrameHeight ? GetFrameHeight() : theme.Sizes.Cell.Points; }
uint32_t CellsOf(float cell, float points) { return std::max(1u, uint32_t(std::ceil(points / cell))); }

// An op node is titled by its MaterialX node, and a node with an operation by its kind and the operation.
const char *OperationTitle(const state::Registry &r, state::Entity node, uint16_t kind) {
    if (kind == KindOf<Op>) return FindOp(r.get<const Op>(node).Node)->Title.c_str();
    const char *title = KindTitles[kind];
    VisitKind(kind, [&]<typename K> {
        if constexpr (requires { &K::Operation; }) {
            using Operation = decltype(K::Operation);
            const auto operation = size_t(std::ranges::find(field::EnumValues<Operation>, r.get<const K>(node).Operation) - field::EnumValues<Operation>.begin());
            static const auto titles = [] {
                std::vector<std::string> titles;
                for (const auto *label : field::EnumLabels<Operation>) titles.push_back(std::string{field::TypeLabel<K>.c_str()} + " " + label);
                return titles;
            }();
            title = titles[operation].c_str();
        }
    });
    return title;
}
const char *LabelOf(const NodeView &view, uint32_t node, uint16_t socket) { return view.Labels[view.SocketFirst[node] + socket].c_str(); }
// Resolves each op node of the geometry page against `source`, the plan of the page's root network, where `path` leads from the root network to the page, and sets the data types each socket shows.
void ResolveOps(const state::Registry &r, NodeView &view, std::span<const state::Entity> path, const CookPlan &source) {
    const auto &a = AuthoredOf(r, view);
    view.Declarations.assign(view.Nodes.size(), std::nullopt);
    view.ShownTypes.assign_range(view.Sockets | std::views::transform(&Socket::Types));
    if (std::ranges::none_of(view.Nodes, [](const EditorNode &node) { return node.Kind == KindOf<Op>; })) return;
    const auto &lib = mx::Libraries();
    const auto layout = LayoutOf(r, source);
    for (uint32_t n = 0; n < view.Nodes.size(); ++n) {
        if (view.Nodes[n].Kind != KindOf<Op>) continue;
        const auto entity = view.Nodes[n].Item;
        const auto &op = a.get<const Op>(entity);
        NodePath occurrence{path.begin(), path.end()};
        occurrence.push_back(entity);
        const auto at = IndexOf(source, occurrence);
        const auto declaration = at == NoNode ? std::nullopt : layout.Nodes[at].Declaration;
        view.Declarations[n] = declaration;
        const auto sockets = NodeSockets(view, n);
        for (uint16_t s = 0; s < sockets.size(); ++s) {
            const std::string_view name{sockets[s].Name};
            const bool input = sockets[s].Direction == SocketDirection::In;
            // The resolved declaration's type for the socket, or else the type of its literal.
            const auto type = [&]() -> std::optional<mx::Type> {
                if (const auto *port = declaration ? FindPort(lib.Declarations[*declaration], name, input) : nullptr) return port->Type;
                const auto literal = input ? op.Literals.find(name) : op.Literals.end();
                return literal == op.Literals.end() ? std::nullopt : std::optional{literal->second};
            }();
            if (const auto data = type ? DataTypeOf(*type) : std::nullopt) view.ShownTypes[view.SocketFirst[n] + s] = TypeBit(*data);
        }
    }
}
// The geometry page of the root network, or of the definition's network of the last subnet node of `path`, whose subnet nodes lead from the root network to it.
// A subnet node is titled by its definition's name and, when the definition has several, its uses.
// An op node resolves against the plan of the root network, which is its occurrence's procedure for a geometry procedure.
NodeView GeometryPage(const state::Registry &r, const UseCounts &uses, state::Entity root, std::span<const state::Entity> path) {
    const auto &a = RegistryAt(r, path);
    const auto *procedure = r.try_get<const GeometryProcedure>(root);
    // The display shows on this page's node that is or encloses the displayed occurrence.
    const auto shown = procedure ? DisplayedPath(*procedure) : NodePath{};
    const bool below = shown.size() > path.size() && std::ranges::equal(path, std::span{shown}.first(path.size()));
    auto view = ProjectNetwork(a, path.empty() ? root : InstanceDefinition(r, path.back()).Entity, below ? std::span{shown}.subspan(path.size()) : std::span<const state::Entity>{});
    for (uint32_t n = 0; n < view.Nodes.size(); ++n) {
        auto &[entity, title, type_title, kind] = view.Nodes[n];
        title = kind != KindOf<Subnet> || a.all_of<Subnet>(entity) ? NodeTitle(a, entity) :
            SlotsOf(view, n) > 0u                                  ? std::string{"Subnet Output"} :
                                                                     std::string{"Subnet Input"};
        if (const auto *subnet = a.try_get<const Subnet>(entity)) {
            if (const auto count = UseCount(uses, subnet->DefinitionEntity, subnet->Builtin); count > 1u) title += std::format(" ({} uses)", count);
        }
        type_title = KindTitles[kind];
        const auto *op = kind == KindOf<Op> ? FindOp(a.get<const Op>(entity).Node) : nullptr;
        const auto sockets = NodeSockets(view, n);
        for (uint16_t s = 0; s < sockets.size(); ++s) view.Labels.emplace_back(op ? op->Labels[s] : LabelOf(sockets[s]));
    }
    const auto *compiled = r.try_get<const CookPlan>(root);
    ResolveOps(r, view, path, compiled ? *compiled : Compile(r, root));
    return view;
}
uint32_t ActiveIndex(const state::Registry &r, const NodeView &view) {
    if (view.Authored) return NodeOf(view, ActiveNodeOf(*view.Authored, view.Network));
    // A bone page shows the active bone, and every other page the active node.
    if (const auto bone = NodeOf(view, FindActiveBone(r)); bone != NoNode) return bone;
    return NodeOf(view, FindActiveNode(r));
}
// A geometry node is selected in its network's selection, and any other node by its entity's selection tag.
bool Selected(const state::Registry &r, const NodeView &view, uint32_t node) {
    const auto entity = view.Nodes[node].Item;
    if (!view.Authored) return r.any_of<::Selected, BoneSelection>(entity);
    const auto *selection = view.Authored->try_get<const NetworkActive>(view.Network);
    return selection && selection->Contains(entity);
}
// The distinct items of the nodes, sorted.
std::vector<state::Entity> ItemsOf(const NodeView &view, std::span<const uint32_t> indices) {
    std::vector<state::Entity> items;
    items.reserve(indices.size());
    for (const auto n : indices) items.push_back(view.Nodes[n].Item);
    SortUnique(items);
    return items;
}

bool IsLinked(const NodeView &view, uint32_t node, uint16_t socket) { return view.Wires[view.InputFirst[node] + socket].FromNode != NoNode; }
std::vector<Socket> LinkedParameters(const NodeView &view, uint32_t node) {
    const auto [first, end] = ParameterRange(view.Nodes[node].Kind);
    std::vector<Socket> linked;
    for (auto s = first; s < end; ++s) {
        if (IsLinked(view, node, s)) linked.push_back(NodeSockets(view, node)[s]);
    }
    return linked;
}

constexpr const char *UnlinkedTray{"Parameters"}; // The label of a closed tray with no linked socket
// The labels of the node's linked tray sockets, which its closed tray shows.
std::string TrayLinks(const NodeView &view, uint32_t node) {
    std::string label;
    for (uint16_t s = 0u; s < InputsOf(view, node); ++s) {
        if (InTray(view, node, s) && IsLinked(view, node, s)) label += std::string{label.empty() ? "" : ", "} + LabelOf(view, node, s);
    }
    return label;
}

// The row of each input socket below the node's title, with the tray closed and with it open.
// With the tray closed, every tray socket is on row Shown, the row after the other inputs.
struct InputRows {
    std::vector<uint16_t> Closed, Open;
    uint16_t Shown{}, Tray{};
    uint16_t ClosedCount() const { return Shown + uint16_t(Tray > 0u); }
    uint16_t OpenCount() const { return Shown + Tray; }
};
InputRows RowsOf(uint16_t inputs, auto &&in_tray) {
    InputRows rows{std::vector<uint16_t>(inputs), std::vector<uint16_t>(inputs)};
    for (uint16_t s = 0u; s < inputs; ++s) {
        if (!in_tray(s)) rows.Closed[s] = rows.Open[s] = rows.Shown++;
    }
    for (uint16_t s = 0u; s < inputs; ++s) {
        if (!in_tray(s)) continue;
        rows.Closed[s] = rows.Shown;
        rows.Open[s] = rows.Shown + rows.Tray++;
    }
    return rows;
}

// A node's cells: its top left and size, and whether its left column holds input ports and its right column output ports.
struct Box {
    uint32_t X, Y;
    uint16_t W, H;
    bool In, Out;
};
// The node's box with its tray closed, which shows `tray` on its row.
// Its width also fits the labels of the open tray.
// `label(s)` returns the label of socket s.
// A label is at most MaxLabelCells wide and ends in an ellipsis when it is longer.
Box BoxOf(const NodeTheme &theme, float cell, const char *title, std::span<const Socket> sockets, const InputRows &rows, const char *tray, auto &&label) {
    const auto inputs = uint16_t(rows.Closed.size());
    const uint32_t outputs = uint32_t(sockets.size()) - inputs;
    // A label's cells hold it with the node's margin on both sides, which a row with one label needs.
    const auto text_cells = [&](const char *text, float indent) {
        return std::min(CellsOf(cell, (theme.Sizes.Margin + theme.Sizes.LabelSpacing) * 2.f + indent + CalcTextSize(text).x), uint32_t(theme.Sizes.MaxLabelCells));
    };
    const auto label_cells = [&](uint16_t s) { return text_cells(label(s), s < inputs && rows.Open[s] >= rows.Shown ? theme.Sizes.ParameterIndent : 0.f); };
    std::vector<uint32_t> left(std::max<uint32_t>(rows.OpenCount(), outputs));
    for (uint16_t s = 0u; s < inputs; ++s) left[rows.Open[s]] = label_cells(s);
    if (rows.Tray > 0u) left[rows.Shown] = std::max(left[rows.Shown], text_cells(tray, 0.f));
    const bool in = rows.ClosedCount() > 0u, out = outputs > 0u;
    uint32_t width = CellsOf(cell, CalcTextSize(title).x + (theme.Sizes.Margin + theme.Sizes.TitlePadding) * 2.f);
    for (uint32_t row = 0u; row < left.size(); ++row) width = std::max(width, uint32_t(in) + left[row] + (row < outputs ? label_cells(uint16_t(inputs + row)) : 0u) + uint32_t(out));
    return {0, 0, uint16_t(width), uint16_t(1u + std::max({uint32_t(rows.ClosedCount()), outputs, 1u})), in, out};
}
bool Overlap(const Box &a, const Box &b) { return a.X < b.X + b.W && b.X < a.X + a.W && a.Y < b.Y + b.H && b.Y < a.Y + a.H; }
// Finds the spans of rows that meet a range of rows.
// Level l has cells of RowCell << l rows.
// A span is filed once, by its first row, in the lowest level whose cells are at least as long as the span, so it starts in a cell it meets or in the cell before.
constexpr uint32_t RowCell{8u};
struct RowIndex {
    // A filed span has its first row at or above its last, and links to the other spans of its cell.
    struct Span {
        uint32_t Top{1u}, Bottom{0u}, Next{NoNode}, Prev{NoNode};
    };
    std::vector<Span> Spans; // By id
    std::unordered_map<uint64_t, uint32_t> First; // The first span of each cell that holds one, by CellOf
    uint32_t Levels{};
};
uint32_t LevelOf(uint32_t top, uint32_t bottom) { return uint32_t(std::bit_width((bottom - top) / RowCell)); }
uint64_t CellOf(uint32_t level, uint32_t row) { return uint64_t(level) << 32u | row / (uint64_t(RowCell) << level); }
void RemoveRows(RowIndex &index, uint32_t id) {
    if (id >= index.Spans.size() || index.Spans[id].Top > index.Spans[id].Bottom) return;
    const auto span = std::exchange(index.Spans[id], {});
    if (span.Next != NoNode) index.Spans[span.Next].Prev = span.Prev;
    if (span.Prev != NoNode) index.Spans[span.Prev].Next = span.Next;
    else if (span.Next != NoNode) index.First[CellOf(LevelOf(span.Top, span.Bottom), span.Top)] = span.Next;
    else index.First.erase(CellOf(LevelOf(span.Top, span.Bottom), span.Top));
}
// Files the id under the rows from top to bottom, in place of the rows it was filed under.
void SetRows(RowIndex &index, uint32_t id, uint32_t top, uint32_t bottom) {
    RemoveRows(index, id);
    const auto level = LevelOf(top, bottom);
    index.Levels = std::max(index.Levels, level + 1u);
    if (index.Spans.size() <= id) index.Spans.resize(id + 1u);
    const auto [first, empty] = index.First.try_emplace(CellOf(level, top), id);
    const auto next = empty ? NoNode : std::exchange(first->second, id);
    if (next != NoNode) index.Spans[next].Prev = id;
    index.Spans[id] = {top, bottom, next, NoNode};
}
void SetRows(RowIndex &index, uint32_t node, const Box &box) { SetRows(index, node, box.Y, box.Y + box.H - 1u); }
// Calls `f(id)` once for each span that meets the rows from top to bottom.
void ForEachIn(const RowIndex &index, uint32_t top, uint32_t bottom, auto &&f) {
    for (uint32_t level = 0u; level < index.Levels; ++level) {
        const auto first = CellOf(level, top), last = CellOf(level, bottom);
        for (auto cell = first - uint64_t(uint32_t(first) > 0u); cell <= last; ++cell) {
            const auto at = index.First.find(cell);
            for (auto id = at == index.First.end() ? NoNode : at->second; id != NoNode; id = index.Spans[id].Next) {
                if (index.Spans[id].Top <= bottom && index.Spans[id].Bottom >= top) f(id);
            }
        }
    }
}

// Whether every box of the group, moved by the offset, stays on the canvas and overlaps no indexed box that counts.
bool Fits(std::span<const Box> boxes, const RowIndex &index, std::span<const uint32_t> group, int64_t dx, int64_t dy, auto &&counts) {
    for (const auto m : group) {
        auto box = boxes[m];
        const int64_t x = int64_t(box.X) + dx, y = int64_t(box.Y) + dy;
        if (x < 0 || y < 0) return false;
        box.X = uint32_t(x);
        box.Y = uint32_t(y);
        bool clear = true;
        ForEachIn(index, box.Y, box.Y + box.H - 1u, [&](uint32_t n) { clear = clear && (!counts(n) || !Overlap(box, boxes[n])); });
        if (!clear) return false;
    }
    return true;
}
constexpr int64_t PlaceReach{64}; // The number of rings of cells Place searches around the wanted offset.
// The offset nearest the wanted one where the group fits, searching outward one ring of cells at a time.
// If nothing fits within PlaceReach, the group goes to the right of every box that counts.
std::pair<int64_t, int64_t> Place(std::span<const Box> boxes, const RowIndex &index, std::span<const uint32_t> group, int64_t dx, int64_t dy, auto &&counts) {
    for (int64_t ring = 0; ring <= PlaceReach; ++ring) {
        for (int64_t i = -ring; i <= ring; ++i) {
            for (int64_t j = -ring; j <= ring; ++j) {
                if ((std::abs(i) == ring || std::abs(j) == ring) && Fits(boxes, index, group, dx + i, dy + j, counts)) return {dx + i, dy + j};
            }
        }
    }
    int64_t right = 0, left = INT64_MAX, top = INT64_MAX;
    for (uint32_t n = 0u; n < boxes.size(); ++n) {
        if (counts(n)) right = std::max(right, int64_t(boxes[n].X) + boxes[n].W);
    }
    for (const auto m : group) {
        left = std::min(left, int64_t(boxes[m].X));
        top = std::min(top, int64_t(boxes[m].Y));
    }
    return {right - left, std::max(dy, -top)};
}
// The lanes of an automatic layout, the lanes each wire slot's link crosses from left to right, and the way back of each link that reads leftward.
struct LaneLayout {
    struct Return {
        uint32_t After{NoNode}, Row{NoNode}; // The free column right of the source's column and the free row below the island, or NoNode for a link that reads rightward
    };
    std::vector<Box> Cells; // Each lane's cell row, as wide as its column
    std::vector<uint32_t> SlotFirst, Crossed; // The lanes of slot s are Crossed[SlotFirst[s]] up to Crossed[SlotFirst[s + 1]]
    std::vector<Return> Returns; // By slot
};
// Lays out every box from the view's links alone, with a lane in each column a link passes.
// Connected nodes form an island, and the islands stack top to bottom in node order.
// An island's columns follow its connections left to right: each node sits one column left of its leftmost reader, then each node with inputs sits one column right of its rightmost input.
// A link from column a to column b > a + 1 has a lane in each column between, which it shares with the other links from its socket that pass that column.
// A column stacks its nodes and lanes in the order a walk reaches them, from the island's last node back through each node's inputs in slot order, where a lane comes with the first link that reaches it.
// One free column separates adjacent columns, the nodes and lanes of a column touch, and one free row separates the islands.
// A link that closes a cycle returns leftward along the free row below its island, from the free column right of its source's column.
LaneLayout AutoLayout(const NodeView &view, std::span<Box> boxes) {
    const auto count = uint32_t(boxes.size());
    const auto &wires = view.Wires;
    const auto &slot_first = view.InputFirst;
    std::vector<std::vector<uint32_t>> inputs(count), readers(count);
    std::vector<uint32_t> island(count);
    std::ranges::iota(island, 0u);
    const auto island_of = [&](uint32_t n) {
        while (island[n] != n) n = island[n] = island[island[n]];
        return n;
    };
    const auto connect = [&](uint32_t from, uint32_t to) {
        if (from == NoNode || from == to) return;
        inputs[to].push_back(from);
        readers[from].push_back(to);
        const auto [first, second] = std::minmax({island_of(from), island_of(to)});
        island[second] = first;
    };
    for (uint32_t n = 0u; n < count; ++n) {
        for (auto slot = slot_first[n]; slot < slot_first[n + 1u]; ++slot) connect(wires[slot].FromNode, n);
    }
    // Finish each node after its readers.
    // A link into a node that is still open closes a cycle, and the columns skip it.
    constexpr uint32_t Open{NoNode - 1u};
    std::vector<uint32_t> order, finished(count, NoNode);
    std::vector<int32_t> column(count);
    const auto finish = [&](this const auto &self, uint32_t n) -> void {
        if (finished[n] != NoNode) return;
        finished[n] = Open;
        for (const auto reader : readers[n]) {
            self(reader);
            if (finished[reader] != Open) column[n] = std::min(column[n], column[reader] - 1);
        }
        finished[n] = uint32_t(order.size());
        order.push_back(n);
    };
    for (uint32_t n = 0u; n < count; ++n) finish(n);
    for (const auto n : order | std::views::reverse) {
        int32_t right = INT32_MIN;
        for (const auto input : inputs[n]) {
            if (finished[input] > finished[n]) right = std::max(right, column[input]);
        }
        if (right != INT32_MIN) column[n] = right + 1;
    }
    // A definition's input boundary takes the column before every other node of its island, and its output boundary the column after.
    const auto boundary = [&](uint32_t n) { return view.Authored && view.Nodes[n].Item == view.Network; };
    std::vector<int32_t> first(count, INT32_MAX), last(count, INT32_MIN);
    for (uint32_t n = 0u; n < count; ++n) {
        if (boundary(n)) continue;
        first[island_of(n)] = std::min(first[island_of(n)], column[n]);
        last[island_of(n)] = std::max(last[island_of(n)], column[n]);
    }
    for (uint32_t n = 0u; n < count; ++n) {
        if (boundary(n) && first[island_of(n)] <= last[island_of(n)]) column[n] = OutputBoundary(view, n) ? last[island_of(n)] + 1 : first[island_of(n)] - 1;
    }
    // A link reads forward when its reader finishes first, and every forward link reads from a column left of its reader's.
    LaneLayout lanes;
    lanes.SlotFirst.assign(wires.size() + 1u, 0u);
    for (uint32_t n = 0u; n < count; ++n) {
        for (auto slot = slot_first[n]; slot < slot_first[n + 1u]; ++slot) {
            const auto from = wires[slot].FromNode;
            lanes.SlotFirst[slot + 1u] = from != NoNode && from != n && finished[from] > finished[n] ? uint32_t(column[n] - column[from] - 1) : 0u;
        }
    }
    for (uint32_t slot = 0u; slot < wires.size(); ++slot) lanes.SlotFirst[slot + 1u] += lanes.SlotFirst[slot];
    lanes.Crossed.resize(lanes.SlotFirst.back());
    // Each lane's island, column and walk order, and the lane of each output socket in each column it passes.
    std::vector<std::tuple<uint32_t, int32_t, uint32_t>> lane_places;
    std::map<std::tuple<uint32_t, uint16_t, int32_t>, uint32_t> lane_of;
    std::vector<uint32_t> walked(count, NoNode);
    uint32_t reached = 0u;
    const auto walk = [&](this const auto &self, uint32_t n) -> void {
        if (walked[n] != NoNode) return;
        walked[n] = reached++;
        for (auto slot = slot_first[n]; slot < slot_first[n + 1u]; ++slot) {
            const auto [from, socket] = wires[slot];
            if (from == NoNode || from == n) continue;
            const auto at = reached++;
            for (auto i = lanes.SlotFirst[slot]; i < lanes.SlotFirst[slot + 1u]; ++i) {
                const auto lane_column = column[from] + 1 + int32_t(i - lanes.SlotFirst[slot]);
                const auto [lane, added] = lane_of.try_emplace({from, socket, lane_column}, uint32_t(lane_places.size()));
                if (added) lane_places.emplace_back(island_of(from), lane_column, at);
                lanes.Crossed[i] = lane->second;
            }
            self(from);
        }
    };
    for (const auto n : order) walk(n);
    lanes.Cells.assign(lane_places.size(), {0u, 0u, 0u, 1u, false, false});
    // The nodes, then the lanes at count and after.
    std::vector<uint32_t> sorted(count + lane_places.size());
    std::ranges::iota(sorted, 0u);
    const auto place = [&](uint32_t i) { return i < count ? std::tuple{island_of(i), column[i], walked[i]} : lane_places[i - count]; };
    std::ranges::sort(sorted, {}, place);
    const auto box = [&](uint32_t i) -> Box & { return i < count ? boxes[i] : lanes.Cells[i - count]; };
    // Each node's column's right edge, and each island's bottom.
    std::vector<uint32_t> right(count), below(count);
    uint32_t top = 0u;
    for (const auto items : sorted | std::views::chunk_by([&](uint32_t a, uint32_t b) { return std::get<0>(place(a)) == std::get<0>(place(b)); })) {
        uint32_t left = 0u, bottom = top;
        for (const auto stacked : items | std::views::chunk_by([&](uint32_t a, uint32_t b) { return std::get<1>(place(a)) == std::get<1>(place(b)); })) {
            uint32_t width = 0u, y = top;
            for (const auto i : stacked) {
                box(i).X = left;
                box(i).Y = y;
                y += box(i).H;
                width = std::max(width, uint32_t(box(i).W));
            }
            for (const auto i : stacked) {
                if (i >= count) lanes.Cells[i - count].W = uint16_t(width);
                else right[i] = left + width;
            }
            left += width + 1u;
            bottom = std::max(bottom, y);
        }
        below[std::get<0>(place(items.front()))] = bottom;
        top = bottom + 1u;
    }
    lanes.Returns.assign(wires.size(), {});
    for (uint32_t n = 0u; n < count; ++n) {
        for (auto slot = slot_first[n]; slot < slot_first[n + 1u]; ++slot) {
            const auto from = wires[slot].FromNode;
            if (from != NoNode && from != n && finished[from] <= finished[n]) lanes.Returns[slot] = {right[from], below[island_of(from)]};
        }
    }
    return lanes;
}
} // namespace

std::string NodeTitle(const state::Registry &r, state::Entity node) {
    const auto kind = r.get<const ProcedureNode>(node).Kind;
    return kind == KindOf<Subnet> ? InstanceName(r, node) : std::string{OperationTitle(r, node, kind)};
}
std::vector<SocketTitle> SocketTitles(const state::Registry &r, state::Entity node) {
    std::vector<SocketTitle> titles;
    if (const auto *interface = InterfaceOf(r, node)) {
        for (const auto &port : interface->Ports) titles.push_back({port.Id, port.Name});
        return titles;
    }
    const auto *graph_node = r.try_get<const ProcedureNode>(node);
    if (!graph_node) return titles;
    const auto *op = graph_node->Kind == KindOf<Op> ? FindOp(r.get<const Op>(node).Node) : nullptr;
    const auto sockets = SocketsOf(r, node);
    for (uint16_t s = 0; s < sockets.size(); ++s) titles.push_back({s, op && s < op->Labels.size() ? op->Labels[s] : std::string{LabelOf(sockets[s])}});
    return titles;
}

// The displayed page, kept between frames.
// Its view is built again when the history epoch, the canvas page or a network revision the view reads changes.
// Its content is the view, each node's rows and box, and the automatic layout, measured again when the view or the node theme changes.
// Its arrangement is the layout with the network's placed cells applied and the row indexes, rebuilt when the content or a placed cell changes.
struct NodePage {
    uint64_t Epoch{};
    state::Entity Root{state::Null}; // The network whose page the view shows, or that holds the first subnet node of Path
    std::vector<state::Entity> Path; // The subnet nodes from the root network to a definition's page, outermost first
    state::Entity Mesh{state::Null}; // The mesh whose procedure holds the page's occurrence, or null for a page with no occurrence
    uint64_t Revision{UINT64_MAX}, GraphRevision{UINT64_MAX}; // The network revisions the view was built at
    NodeView View;
    bool Rebuilt{}; // The view changed after the canvas last measured it
    std::vector<InputRows> Rows;
    struct Measured {
        state::Entity Item;
        uint64_t Text;
        Box Size;
    };
    std::vector<Measured> Sizes; // Each node's text and the box measured for it, in item order
    NodeThemeSizes Theme;
    float Cell{}, Font{};
    uint64_t Content{}; // The texts and wires the automatic layout was computed for
    std::vector<Box> Auto; // Each node's box at its automatic cell
    LaneLayout Lanes; // The lanes of the automatic layout
    std::vector<uint32_t> OutFirst, Out; // The wire slots that leave each node
    uint64_t Cells{UINT64_MAX}; // The cells revision the boxes were arranged at
    std::vector<Box> Boxes; // Each node's box where the editor shows it
    std::vector<uint32_t> Moved; // The nodes away from their automatic cells, in order
    RowIndex Nodes, Wires; // The boxes and the wires, by the rows they span
};

namespace {
// The page the canvas shows, whose view, when built again equal to the shown one, keeps the shown one with its layout.
NodePage &PageOf(const state::Registry &r, NodesPanel &panel) {
    if (!panel.Page || panel.Page->Epoch != r.Epoch) panel.Page = std::make_shared<NodePage>(NodePage{.Epoch = r.Epoch});
    auto &page = *panel.Page;
    const auto &canvas = panel.Canvas;
    const auto shown = EditorNetwork(r, panel);
    const auto root = canvas.Path.empty() ? shown : canvas.Root;
    const bool geometry = !canvas.Path.empty() || r.all_of<NetworkNodes>(root);
    const auto &changes = r.Resources.get<const NetworkChanges>();
    // A scene page reads no network graph.
    const auto graph = geometry ? changes.GraphRevision : 0u;
    if (page.Root == root && page.Path == canvas.Path && page.Revision == changes.Revision && page.GraphRevision == graph) return page;
    page.Root = root;
    page.Path = canvas.Path;
    page.Mesh = geometry && r.all_of<GeometryProcedure>(root) ? root : state::Null;
    page.Revision = changes.Revision;
    page.GraphRevision = graph;
    if (auto view = geometry ? GeometryPage(r, UsesOf(r, panel), root, canvas.Path) : SceneNodeDefinitions(r, shown); view != page.View) {
        page.View = std::move(view);
        page.Rebuilt = true;
    }
    return page;
}
// The fault the page's occurrence of the node holds in its mesh's cook, or null for a page with no occurrence.
const NodeFault *FaultOf(const state::Registry &r, const NodePage &page, state::Entity node) {
    if (page.Mesh == state::Null) return nullptr;
    auto path = page.Path;
    path.push_back(node);
    const auto *error = CookedOf<NodeError>(r, path);
    return error ? &error->Fault : nullptr;
}

uint64_t Mix(uint64_t hash, uint64_t value) { return hash ^ (value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u)); }
uint64_t Mix(uint64_t hash, std::string_view text) { return Mix(hash, std::hash<std::string_view>{}(text)); }

// Brings the page's rows, boxes and automatic layout up to date with its view and the theme.
// A node whose text is unchanged keeps its box, and unchanged texts and wires keep the layout.
// Returns whether the layout changed.
bool Measure(NodePage &page, const NodeTheme &theme, float cell, bool restyled) {
    const auto &view = page.View;
    const auto count = uint32_t(view.Nodes.size());
    page.Theme = theme.Sizes;
    page.Cell = cell;
    page.Font = GetFontSize();
    page.Rows.clear();
    std::vector<NodePage::Measured> sizes;
    std::vector<Box> boxes;
    uint64_t content = count;
    for (uint32_t n = 0u; n < count; ++n) {
        const auto &rows = page.Rows.emplace_back(RowsOf(InputsOf(view, n), [&](uint16_t s) { return InTray(view, n, s); }));
        const auto sockets = NodeSockets(view, n);
        const auto tray = TrayLinks(view, n);
        auto text = Mix(Mix(0u, view.Nodes[n].Title), tray);
        for (uint16_t s = 0u; s < sockets.size(); ++s) {
            text = Mix(text, LabelOf(view, n, s));
            if (s < rows.Closed.size()) text = Mix(text, uint64_t(rows.Closed[s]) << 16u | rows.Open[s]);
        }
        const auto item = view.Nodes[n].Item;
        const auto measured = std::ranges::lower_bound(page.Sizes, item, {}, &NodePage::Measured::Item);
        const bool kept = !restyled && measured != page.Sizes.end() && measured->Item == item && measured->Text == text;
        boxes.push_back(kept ? measured->Size : BoxOf(theme, cell, view.Nodes[n].Title.c_str(), sockets, rows, tray.empty() ? UnlinkedTray : tray.c_str(), [&](uint16_t s) { return LabelOf(view, n, s); }));
        sizes.push_back({item, text, boxes.back()});
        content = Mix(content, text);
    }
    const auto &wires = view.Wires;
    for (const auto link : wires) content = Mix(content, uint64_t(link.FromNode) << 16u | link.FromSocket);
    std::ranges::sort(sizes, {}, &NodePage::Measured::Item);
    page.Sizes = std::move(sizes);
    if (!restyled && content == page.Content && page.Auto.size() == count) return false;
    page.Content = content;
    page.Lanes = AutoLayout(view, boxes);
    page.Auto = std::move(boxes);
    page.OutFirst.assign(count + 1u, 0u);
    for (const auto link : wires) {
        if (link.FromNode != NoNode) ++page.OutFirst[link.FromNode + 1u];
    }
    for (uint32_t n = 0u; n < count; ++n) page.OutFirst[n + 1u] += page.OutFirst[n];
    page.Out.resize(page.OutFirst.back());
    auto next = page.OutFirst;
    for (uint32_t slot = 0u; slot < wires.size(); ++slot) {
        if (wires[slot].FromNode != NoNode) page.Out[next[wires[slot].FromNode]++] = slot;
    }
    return true;
}

// The node whose input the wire slot is.
uint32_t ReaderOf(const NodeView &view, uint32_t slot) { return uint32_t(std::ranges::upper_bound(view.InputFirst, slot) - view.InputFirst.begin()) - 1u; }
// Whether the link from node `from` into node `to` runs through the lanes or the return row of the automatic layout, which holds while both nodes are at their automatic cells.
bool Routed(const NodePage &page, uint32_t from, uint32_t to) {
    const auto automatic = [&](uint32_t n) { return page.Boxes[n].X == page.Auto[n].X && page.Boxes[n].Y == page.Auto[n].Y; };
    return automatic(from) && automatic(to);
}
// The lanes the wire slot's link crosses, from left to right.
std::span<const uint32_t> CrossedBy(const NodePage &page, uint32_t slot) {
    const auto &lanes = page.Lanes;
    return std::span{lanes.Crossed}.subspan(lanes.SlotFirst[slot], lanes.SlotFirst[slot + 1u] - lanes.SlotFirst[slot]);
}
// Files the wire in the reader's slot under the rows the boxes of its two nodes and the lanes it runs through span.
void FileWire(NodePage &page, uint32_t slot, uint32_t reader) {
    const auto from = page.View.Wires[slot].FromNode;
    const auto routed = Routed(page, from, reader);
    uint32_t top = std::min(page.Boxes[reader].Y, page.Boxes[from].Y), bottom = std::max(page.Boxes[reader].Y + page.Boxes[reader].H, page.Boxes[from].Y + page.Boxes[from].H);
    for (const auto lane : routed ? CrossedBy(page, slot) : std::span<const uint32_t>{}) {
        top = std::min(top, page.Lanes.Cells[lane].Y);
        bottom = std::max(bottom, page.Lanes.Cells[lane].Y + 1u);
    }
    if (const auto row = page.Lanes.Returns[slot].Row; routed && row != NoNode) bottom = std::max(bottom, row + 1u);
    SetRows(page.Wires, slot, top, bottom - 1u);
}
// Files each wire that touches the node.
void FileWires(NodePage &page, uint32_t n) {
    const auto &view = page.View;
    for (auto slot = view.InputFirst[n]; slot < view.InputFirst[n + 1u]; ++slot) {
        if (view.Wires[slot].FromNode != NoNode) FileWire(page, slot, n);
    }
    for (const auto slot : std::span{page.Out}.subspan(page.OutFirst[n], page.OutFirst[n + 1u] - page.OutFirst[n])) FileWire(page, slot, ReaderOf(view, slot));
}

// Arranges the page's boxes: each at its automatic cell, then each placed one at its cell.
// A placed box takes the cell nearest its own where it overlaps no placed box before it, and an automatic box a placed one overlaps moves to the nearest free cell.
// Every node position the editor uses comes from here, so nodes never overlap.
// Without a new layout, only the boxes away from their automatic cells are arranged again.
void Arrange(const state::Registry &r, NodePage &page, bool relaid) {
    const auto &view = page.View;
    const auto count = uint32_t(view.Nodes.size());
    const auto &wires = view.Wires;
    auto &boxes = page.Boxes;
    // The nodes that were away from their automatic cells before this pass.
    auto changed = std::move(page.Moved);
    if (relaid) {
        boxes = page.Auto;
        page.Nodes = {};
        page.Wires = {};
        for (uint32_t n = 0u; n < count; ++n) SetRows(page.Nodes, n, boxes[n]);
    } else {
        for (const auto n : changed) {
            boxes[n] = page.Auto[n];
            SetRows(page.Nodes, n, boxes[n]);
        }
    }
    page.Moved.clear();
    if (const auto *layout = AuthoredOf(r, view).try_get<const NetworkLayout>(view.Network)) {
        std::vector<uint32_t> placed;
        for (const auto &cell : layout->Cells) {
            const auto n = !cell.Output ? NodeOf(view, cell.Item) : !view.Nodes.empty() && OutputBoundary(view, uint32_t(view.Nodes.size() - 1u)) && view.Nodes.back().Item == cell.Item ? uint32_t(view.Nodes.size() - 1u) :
                                                                                                                                                                          NoNode;
            if (n == NoNode) continue;
            RemoveRows(page.Nodes, n);
            boxes[n].X = cell.Position.x;
            boxes[n].Y = cell.Position.y;
            placed.push_back(n);
        }
        std::ranges::sort(placed);
        const auto is_placed = [&](uint32_t n) { return std::ranges::binary_search(placed, n); };
        const auto settle = [&](uint32_t n, auto &&counts) {
            const auto [dx, dy] = Place(boxes, page.Nodes, std::array{n}, 0, 0, counts);
            boxes[n].X = uint32_t(int64_t(boxes[n].X) + dx);
            boxes[n].Y = uint32_t(int64_t(boxes[n].Y) + dy);
            SetRows(page.Nodes, n, boxes[n]);
        };
        for (const auto n : placed) settle(n, is_placed);
        std::vector<uint32_t> crowded;
        for (const auto n : placed) {
            ForEachIn(page.Nodes, boxes[n].Y, boxes[n].Y + boxes[n].H - 1u, [&](uint32_t other) {
                if (!is_placed(other) && Overlap(boxes[n], boxes[other])) crowded.push_back(other);
            });
        }
        SortUnique(crowded);
        for (const auto n : crowded) {
            RemoveRows(page.Nodes, n);
            settle(n, [](uint32_t) { return true; });
        }
        page.Moved = std::move(placed);
        page.Moved.append_range(crowded);
        std::ranges::sort(page.Moved);
    }
    if (relaid) {
        for (uint32_t n = 0u, slot = 0u; n < count; ++n) {
            for (; slot < view.InputFirst[n + 1u]; ++slot) {
                if (wires[slot].FromNode != NoNode) FileWire(page, slot, n);
            }
        }
    } else {
        changed.append_range(page.Moved);
        SortUnique(changed);
        for (const auto n : changed) FileWires(page, n);
    }
}

// The screen layout of the network, with cell c at Origin + c * Step.
struct Graph {
    const NodeTheme &Theme;
    ImVec2 Origin;
    float Step; // A cell's side in screen points
    float Zoom;
    uint32_t Expanded;
    NodePage &Page;
    std::vector<uint32_t> Shown, ShownWires; // The nodes and wire slots whose rows meet the canvas, in order
    uint32_t DetachNode{NoNode}; // The node whose input link in DetachSlot a wire drag has picked up
    uint16_t DetachSlot{};
    uint32_t DropNode{NoNode}; // The node whose multi-input socket DropSocket leaves a gap at DropIndex for the held wire
    uint16_t DropSocket{}, DropIndex{};
};
uint32_t RowAt(const Graph &graph, float y) { return uint32_t(std::max(0.f, std::floor((y - graph.Origin.y) / graph.Step))); }
// The node's rows, which for the expanded node fit its open tray.
uint32_t HeightOf(const Graph &graph, uint32_t node) {
    const uint32_t outputs = uint32_t(NodeSockets(graph.Page.View, node).size()) - InputsOf(graph.Page.View, node);
    return node == graph.Expanded ? 1u + std::max({uint32_t(graph.Page.Rows[node].OpenCount()), outputs, 1u}) : graph.Page.Boxes[node].H;
}
// The node's rectangle, which the theme's margin separates from the edges of its cells.
ImRect RectOf(const Graph &graph, uint32_t node) {
    const auto &box = graph.Page.Boxes[node];
    const float margin = graph.Theme.Sizes.Margin * graph.Zoom;
    return {graph.Origin + ImVec2{float(box.X), float(box.Y)} * graph.Step + ImVec2{margin, margin}, graph.Origin + ImVec2{float(box.X + box.W), float(box.Y + HeightOf(graph, node))} * graph.Step - ImVec2{margin, margin}};
}
ImVec2 RowAt(const Graph &graph, uint32_t node, bool input, uint32_t row) {
    const auto &box = graph.Page.Boxes[node];
    return graph.Origin + ImVec2{float(box.X) + (input ? .5f : float(box.W) - .5f), float(box.Y) + 1.5f + float(row)} * graph.Step;
}
// The sockets of a closed tray are all at the tray's chevron.
ImVec2 SocketAt(const Graph &graph, uint32_t node, uint16_t socket) {
    const auto inputs = InputsOf(graph.Page.View, node);
    if (socket >= inputs) return RowAt(graph, node, false, socket - inputs);
    const auto &rows = graph.Page.Rows[node];
    return RowAt(graph, node, true, node == graph.Expanded ? rows.Open[socket] : rows.Closed[socket]);
}
ImVec2 TrayAt(const Graph &graph, uint32_t node) { return RowAt(graph, node, true, graph.Page.Rows[node].Shown); }
ImRect TrayRect(const Graph &graph, uint32_t node) {
    const float reach = graph.Theme.Sizes.SocketRadius * 1.2f * graph.Zoom + 3.f, row = graph.Step;
    const auto at = TrayAt(graph, node);
    return {{at.x - reach, at.y - row * .5f}, {at.x + reach, at.y + (float(graph.Page.Rows[node].Tray) - .5f) * row}};
}
// The end point of link `position` of `total` on a multi-input socket.
// The links are evenly spaced and centered on the socket.
ImVec2 MultiAt(const Graph &graph, ImVec2 socket, uint32_t position, uint32_t total) {
    const float spacing = std::min(MultiSpacing, MultiHalf * 2.f / float(std::max(total, 1u))) * graph.Step;
    return {socket.x, socket.y + (float(position) - float(total - 1u) * .5f) * spacing};
}
// The index, among the socket's links, of the link a wire drag has picked up, or none.
std::optional<uint16_t> DetachedIndex(const Graph &graph, uint32_t node, uint16_t socket) {
    if (graph.DetachNode != node || SlotSocket(graph.Page.View, node, graph.DetachSlot) != socket) return {};
    return SlotIndex(graph.Page.View, node, graph.DetachSlot);
}
// The number of links drawn into the socket, which leaves out a picked-up link.
uint32_t ShownLinks(const Graph &graph, uint32_t node, uint16_t socket) { return LinksInto(graph.Page.View, node, socket) - uint32_t(DetachedIndex(graph, node, socket).has_value()); }
// Where the slot's wire ends.
// On a multi-input socket, the links skip a picked-up link and leave a gap where the held wire would drop.
ImVec2 SlotAt(const Graph &graph, uint32_t node, uint16_t slot) {
    const auto &view = graph.Page.View;
    const auto socket = SlotSocket(view, node, slot);
    const auto at = SocketAt(graph, node, socket);
    if (!NodeSockets(view, node)[socket].Multi) return at;
    uint32_t position = SlotIndex(view, node, slot), total = ShownLinks(graph, node, socket);
    if (const auto detached = DetachedIndex(graph, node, socket); detached && *detached < position) --position;
    if (graph.DropNode == node && graph.DropSocket == socket) {
        if (position >= graph.DropIndex) ++position;
        ++total;
    }
    return MultiAt(graph, at, position, total);
}

// A wire's drawn path: its start, then the two control points and the end of each cubic segment.
// A straight segment's control points are its ends.
using Wire = std::span<const ImVec2>;
// Extends the path with a curve that leaves its end and reaches `to` level, with handles that shorten as the curve flattens.
void CurveTo(std::vector<ImVec2> &path, ImVec2 to, float curving) {
    const auto from = path.back();
    const float dx = std::abs(to.x - from.x), dy = std::abs(to.y - from.y);
    const float straightening = dx > 0.f ? std::min(1.f, dy / dx * WireStraightening) : 1.f;
    const ImVec2 handle{curving * dx * straightening, 0.f};
    path.append_range(std::array{from + handle, to - handle, to});
}
void LineTo(std::vector<ImVec2> &path, ImVec2 to) { path.append_range(std::array{path.back(), to, to}); }
ImVec2 NearestOnWire(Wire w, ImVec2 p) {
    auto nearest = w[0];
    for (size_t i = 1u; i < w.size(); i += 3u) {
        if (const auto q = ImBezierCubicClosestPoint(w[i - 1u], w[i], w[i + 1u], w[i + 2u], p, 24); ImLengthSqr(q - p) < ImLengthSqr(nearest - p)) nearest = q;
    }
    return nearest;
}
// Whether the wire crosses the rectangle, testing each of its segments as straight pieces.
bool Crosses(Wire w, const ImRect &rect) {
    constexpr int Pieces{24};
    for (size_t i = 1u; i < w.size(); i += 3u) {
        auto a = w[i - 1u];
        for (int k = 1; k <= Pieces; ++k) {
            const auto b = ImBezierCubicCalc(w[i - 1u], w[i], w[i + 1u], w[i + 2u], float(k) / float(Pieces));
            const auto d = b - a;
            float enter = 0.f, leave = 1.f;
            bool inside = true;
            for (const auto [p, q] : {std::pair{-d.x, a.x - rect.Min.x}, std::pair{d.x, rect.Max.x - a.x}, std::pair{-d.y, a.y - rect.Min.y}, std::pair{d.y, rect.Max.y - a.y}}) {
                if (p == 0.f) inside &= q >= 0.f;
                else if (p < 0.f) enter = std::max(enter, q / p);
                else leave = std::min(leave, q / p);
            }
            if (inside && enter <= leave) return true;
            a = b;
        }
    }
    return false;
}
void DrawWire(ImDrawList &draw, Wire w, ImU32 color, float thickness) {
    draw.PathLineTo(w[0]);
    for (size_t i = 1u; i < w.size(); i += 3u) {
        if (w[i] == w[i - 1u] && w[i + 1u] == w[i + 2u]) draw.PathLineTo(w[i + 2u]);
        else draw.PathBezierCubicCurveTo(w[i], w[i + 1u], w[i + 2u]);
    }
    draw.PathStroke(color, ImDrawFlags_None, thickness);
}
// Calls `f(wire, link, node, slot)` for each drawn link.
// A routed link runs level from its socket to the free column after its node's column, curves across each free column to the next lane or to its reader's column, and runs along the middle of each lane.
// A routed link that returns leaves into the free column right of its source's column, runs left along the free row below the island, and rises in the free column left of its reader.
// Any other link is one curve from socket to socket.
void ForEachWire(const Graph &graph, auto &&f) {
    const auto &view = graph.Page.View;
    const auto &page = graph.Page;
    const float curving = graph.Theme.Sizes.WireCurving;
    const auto x_of = [&](uint32_t cells) { return graph.Origin.x + float(cells) * graph.Step; };
    std::vector<ImVec2> path;
    for (const auto shown : graph.ShownWires) {
        const auto n = ReaderOf(view, shown);
        const auto slot = uint16_t(shown - view.InputFirst[n]);
        const auto link = view.Wires[shown];
        if (n == graph.DetachNode && slot == graph.DetachSlot) continue;
        const auto to = SlotAt(graph, n, slot);
        path.assign(1u, SocketAt(graph, link.FromNode, link.FromSocket));
        const auto back = page.Lanes.Returns[shown];
        if (Routed(page, link.FromNode, n)) {
            if (back.Row != NoNode) {
                const float row = graph.Origin.y + (float(back.Row) + .5f) * graph.Step;
                LineTo(path, {x_of(back.After), path.back().y});
                CurveTo(path, {x_of(back.After) + .5f * graph.Step, row}, curving);
                LineTo(path, {x_of(page.Boxes[n].X) - .5f * graph.Step, row});
            } else {
                for (const auto lane : CrossedBy(page, shown)) {
                    const auto &cell = page.Lanes.Cells[lane];
                    LineTo(path, {x_of(cell.X - 1u), path.back().y});
                    CurveTo(path, {x_of(cell.X), graph.Origin.y + (float(cell.Y) + .5f) * graph.Step}, curving);
                }
                LineTo(path, {x_of(page.Boxes[n].X - 1u), path.back().y});
            }
            CurveTo(path, {x_of(page.Boxes[n].X), to.y}, curving);
            LineTo(path, to);
        } else {
            CurveTo(path, to, curving);
        }
        f(Wire{path}, link, n, slot);
    }
}

enum class HitType : uint8_t { None,
                               Socket,
                               Parameters, // A node's closed tray
                               Node };
struct Hit {
    HitType Type{HitType::None};
    uint32_t Node{NoNode};
    uint16_t Socket{};
    uint16_t Index{}; // On a multi-input socket, the index of the link nearest the mouse
};
// Tests the expanded node first, then sockets before node bodies, with later-drawn nodes first.
Hit HitTest(const Graph &graph, ImVec2 mouse) {
    const auto &view = graph.Page.View;
    const float reach = std::max(graph.Theme.Sizes.SocketRadius * graph.Zoom, 4.f) + 3.f;
    const auto near = [&](ImVec2 at, float half) {
        const auto d = mouse - at;
        return half > 0.f ? std::abs(d.x) <= reach && std::abs(d.y) <= half + 3.f : ImLengthSqr(d) <= reach * reach;
    };
    const auto sockets = [&](uint32_t n) -> std::optional<Hit> {
        const auto &rows = graph.Page.Rows[n];
        const bool closed = n != graph.Expanded;
        const auto inputs = InputsOf(view, n);
        const auto all = NodeSockets(view, n);
        for (uint16_t s = 0u; s < all.size(); ++s) {
            if (s < inputs && closed && InTray(view, n, s)) continue;
            const auto at = SocketAt(graph, n, s);
            if (!near(at, all[s].Multi ? MultiHalf * graph.Step : 0.f)) continue;
            uint16_t index = 0u;
            if (all[s].Multi) {
                const auto links = LinksInto(view, n, s);
                for (uint16_t k = 1u; k < links; ++k) {
                    if (std::abs(MultiAt(graph, at, k, links).y - mouse.y) < std::abs(MultiAt(graph, at, index, links).y - mouse.y)) index = k;
                }
            }
            return Hit{HitType::Socket, n, s, index};
        }
        if (rows.Tray > 0u && closed && near(TrayAt(graph, n), 0.f)) return Hit{HitType::Parameters, n, 0u};
        return {};
    };
    if (graph.Expanded != NoNode) {
        if (const auto hit = sockets(graph.Expanded)) return *hit;
        if (RectOf(graph, graph.Expanded).Contains(mouse)) return {HitType::Node, graph.Expanded, 0u};
    }
    for (const auto n : graph.Shown | std::views::reverse) {
        if (const auto hit = sockets(n)) return *hit;
    }
    for (const auto n : graph.Shown | std::views::reverse) {
        if (RectOf(graph, n).Contains(mouse)) return {HitType::Node, n, 0u};
    }
    return {};
}

bool Joins(Socket held, Socket other) {
    if (held.Direction == other.Direction) return false;
    return held.Direction == SocketDirection::Out ? Links(held, other) : Links(other, held);
}
bool Accepts(const state::Registry &r, const NodeView &view, uint32_t held_node, uint16_t held_socket, uint32_t node, uint16_t socket) {
    if (NodeSockets(view, held_node)[held_socket].Direction == NodeSockets(view, node)[socket].Direction) return false;
    if (view.Authored) return (node != held_node || view.Nodes[node].Kind == KindOf<Subnet>) && Joins(NodeSockets(view, held_node)[held_socket], NodeSockets(view, node)[socket]);
    const auto &a = BindingOf(view, held_node, held_socket), &b = BindingOf(view, node, socket);
    const auto *input = std::get_if<SceneReference>(&a);
    return input ? ReferenceAccepts(r, *input, std::get<state::Entity>(b)) : ReferenceAccepts(r, std::get<SceneReference>(b), std::get<state::Entity>(a));
}
// The node's visible sockets that the held wire can link with.
// With `tray`, only its tray sockets.
std::vector<uint16_t> LinkingSockets(const state::Registry &r, const Graph &graph, uint32_t held_node, uint16_t held_socket, uint32_t node, bool tray) {
    const auto &view = graph.Page.View;
    const auto inputs = InputsOf(view, node);
    std::vector<uint16_t> sockets;
    for (uint16_t s = 0u; s < NodeSockets(view, node).size(); ++s) {
        if (s < inputs ? tray && !InTray(view, node, s) : tray) continue;
        if (Accepts(r, view, held_node, held_socket, node, s)) sockets.push_back(s);
    }
    return sockets;
}
action::nodes::Link LinkOf(const NodeView &view, uint32_t held_node, uint16_t held_socket, uint32_t node, uint16_t socket, uint16_t index = UINT16_MAX) {
    const bool from_held = NodeSockets(view, held_node)[held_socket].Direction == SocketDirection::Out;
    const auto held = view.Nodes[held_node].Item, other = view.Nodes[node].Item;
    return from_held ? action::nodes::Link{view.Network, held, SocketId(view, held_node, held_socket), other, SocketId(view, node, socket), index} : action::nodes::Link{view.Network, other, SocketId(view, node, socket), held, SocketId(view, held_node, held_socket)};
}
action::nodes::Unlink UnlinkOf(const NodeView &view, uint32_t node, uint16_t slot) {
    return {view.Network, view.Nodes[node].Item, SocketId(view, node, SlotSocket(view, node, slot)), SlotIndex(view, node, slot)};
}
action::nodes::InsertNode InsertOf(const NodeView &view, uint32_t node, uint32_t to_node, uint16_t to_slot, uint32_t shift = 0u) {
    return {view.Network, view.Nodes[node].Item, view.Nodes[to_node].Item, SocketId(view, to_node, SlotSocket(view, to_node, to_slot)), shift, SlotIndex(view, to_node, to_slot)};
}
// The assignment that links a scene page's reference socket with an item socket, given in either order.
action::nodes::AssignReference AssignOf(const NodeView &view, uint32_t a_node, uint16_t a_socket, uint32_t b_node, uint16_t b_socket) {
    const auto &a = BindingOf(view, a_node, a_socket), &b = BindingOf(view, b_node, b_socket);
    const auto *input = std::get_if<SceneReference>(&a);
    return input ? action::nodes::AssignReference{*input, std::get<state::Entity>(b)} : action::nodes::AssignReference{std::get<SceneReference>(b), std::get<state::Entity>(a)};
}
// Links the held wire to the socket, at `index` among a multi-input socket's links.
// A wire picked up from an input is unlinked from that input in the same undo step.
// Dropping it back on that input changes nothing, or reorders it on a multi-input socket.
void Connect(const state::Registry &r, NodesPanel &panel, const NodeView &view, uint32_t node, uint16_t socket, uint16_t index = UINT16_MAX) {
    const bool detached = panel.DetachNode != NoNode;
    const auto left_node = panel.DetachNode;
    const auto left_slot = panel.DetachSlot;
    panel.DetachNode = NoNode;
    const bool returns = detached && node == left_node && socket == SlotSocket(view, left_node, left_slot);
    if (returns && !NodeSockets(view, node)[socket].Multi) return;
    if (!action::Begin(action::Phase::Record)) return;
    if (view.Authored) {
        action::AddToGesture(LinkOf(view, panel.DragNode, panel.DragSocket, node, socket, index));
        if (detached && !returns) action::AddToGesture(UnlinkOf(view, left_node, left_slot));
    } else {
        // The reference the wire left clears first, so a list entry dropped back on its own list is added again.
        if (const auto *left = detached && !returns ? std::get_if<SceneReference>(&BindingOf(view, left_node, left_slot)) : nullptr; left && ReferenceAccepts(r, *left, state::Null)) {
            action::AddToGesture(action::nodes::AssignReference{*left, state::Null});
        }
        action::AddToGesture(AssignOf(view, panel.DragNode, panel.DragSocket, node, socket));
    }
}
void Disconnect(const state::Registry &r, const NodeView &view, uint32_t node, uint16_t slot) {
    if (view.Authored) action::Emit(UnlinkOf(view, node, slot));
    else if (const auto *reference = std::get_if<SceneReference>(&BindingOf(view, node, slot)); reference && ReferenceAccepts(r, *reference, state::Null)) action::Emit(action::nodes::AssignReference{*reference, state::Null});
}
// Drops the held wire on a node.
// With one socket that can take it, links to that socket, and with several, opens the port picker.
// Returns whether the picker opened.
bool DropOnNode(const state::Registry &r, NodesPanel &panel, const Graph &graph, uint32_t node, bool tray) {
    const auto sockets = LinkingSockets(r, graph, panel.DragNode, panel.DragSocket, node, tray);
    if (sockets.size() == 1u) Connect(r, panel, graph.Page.View, node, sockets.front());
    if (sockets.size() <= 1u) return false;
    panel.PickNode = node;
    panel.PickTray = tray;
    OpenPopup("##PickPort");
    return true;
}

// Begin moving the selected nodes after the node count exceeds `after`, with the active node starting from the cell.
void Grab(NodesPanel &panel, uint32_t after, uvec2 start) {
    panel.Drag = NodeDrag::Grab;
    panel.DragNode = NoNode;
    panel.GrabAfter = after;
    panel.MoveStart = start;
    panel.Press = vec2{GetIO().MousePos};
    panel.Moved = false;
}

std::vector<uint32_t> SelectedNodes(const state::Registry &r, const NodeView &view) {
    std::vector<uint32_t> selected;
    if (view.Nodes.empty()) return selected;
    if (!view.Authored) {
        // The selected entities are few beside a page's nodes, so each one looks its node up.
        const auto add = [&](state::Entity e) {
            if (const auto n = NodeOf(view, e); n != NoNode) selected.push_back(n);
        };
        for (const auto e : r.view<const ::Selected>()) add(e);
        for (const auto e : r.view<const BoneSelection>()) add(e);
        std::ranges::sort(selected);
        return selected;
    }
    if (const auto *selection = view.Authored->try_get<const NetworkActive>(view.Network)) {
        for (const auto item : selection->Selection) {
            if (const auto n = NodeOf(view, item); n != NoNode) selected.push_back(n);
        }
        if (const auto last = uint32_t(view.Nodes.size() - 1u); OutputBoundary(view, last) && selection->Contains(view.Network)) selected.push_back(last);
    }
    SortUnique(selected);
    return selected;
}

// Connect adjacent selected nodes in horizontal order, using the first compatible input.
// Skip cycles and occupied inputs unless `replace` is set.
std::vector<action::nodes::Link> MakeLinks(const NodePage &page, std::vector<uint32_t> nodes, bool replace) {
    const auto &view = page.View;
    if (!view.Authored) return {};
    std::ranges::stable_sort(nodes, {}, [&](uint32_t n) { return page.Boxes[n].X; });
    const auto &input_first = view.InputFirst;
    auto wires = view.Wires;
    std::vector<action::nodes::Link> links;
    for (size_t i = 0u; i + 1u < nodes.size(); ++i) {
        auto from = nodes[i], to = nodes[i + 1u];
        if (InputsOf(view, to) == 0u || NodeSockets(view, from).size() == InputsOf(view, from)) std::swap(from, to);
        if (Upstream(input_first, std::span{wires}, from, [](uint32_t) { return true; })[to]) continue;
        const auto from_sockets = NodeSockets(view, from), to_sockets = NodeSockets(view, to);
        const auto first_link = [&](bool linked) -> std::optional<action::nodes::Link> {
            for (uint16_t input = 0u; input < InputsOf(view, to); ++input) {
                const auto current = wires[input_first[to] + input];
                // An input this command already linked keeps its new link.
                if ((current.FromNode != NoNode) != linked || current != view.Wires[input_first[to] + input]) continue;
                for (uint16_t output = InputsOf(view, from); output < from_sockets.size(); ++output) {
                    if (Links(from_sockets[output], to_sockets[input]) && (current.FromNode != from || current.FromSocket != output)) return action::nodes::Link{view.Network, view.Nodes[from].Item, output, view.Nodes[to].Item, input};
                }
            }
            return {};
        };
        auto link = first_link(false);
        if (!link && replace) link = first_link(true);
        if (!link) continue;
        wires[input_first[to] + link->ToSocket] = {from, link->FromSocket};
        link->FromSocket = SocketId(view, from, link->FromSocket);
        link->ToSocket = SocketId(view, to, link->ToSocket);
        links.push_back(*link);
    }
    return links;
}

// The link a dragged node would be inserted into, and the cells that link's reader moves right to make room.
struct Insert {
    uint32_t ToNode;
    uint16_t ToSlot;
    uint32_t Shift;
};
// Of the links that cross the node's box and that it can be inserted into, the one passing nearest its main input socket.
std::optional<Insert> InsertTarget(const Graph &graph, uint32_t node) {
    const auto &view = graph.Page.View;
    const auto input = view.Authored ? MainSocket(view.Nodes[node].Kind, SocketDirection::In) : std::nullopt;
    if (!input) return {};
    const auto rect = RectOf(graph, node);
    const auto entry = SocketAt(graph, node, *input);
    std::optional<Insert> insert;
    float nearest = FLT_MAX;
    ForEachWire(graph, [&](Wire wire, NodeLink, uint32_t to_node, uint16_t slot) {
        if (!Crosses(wire, rect) || !Inserts(view, node, to_node, slot)) return;
        const float distance = ImLengthSqr(NearestOnWire(wire, entry) - entry);
        if (distance >= nearest) return;
        nearest = distance;
        const auto &box = graph.Page.Boxes[node], &reader = graph.Page.Boxes[to_node];
        const uint32_t clear = box.X + box.W;
        insert = Insert{to_node, slot, clear > reader.X ? clear - reader.X : 0u};
    });
    return insert;
}

// The links a held node makes when it is dropped with the mouse over another node's port column.
struct Dock {
    uint32_t Node{NoNode}; // The node whose port column is under the mouse
    std::vector<std::pair<uint16_t, uint16_t>> Links; // Each socket of Node with the held node's socket it links with
};
// On the port cell of a socket the held node can link with, the dock links that socket.
// Anywhere else on the column, it links each output with the first input that is unlinked, or whose link the held node can be inserted into.
Dock DockOf(const state::Registry &r, const Graph &graph, uint32_t held, ImVec2 mouse) {
    const auto &view = graph.Page.View;
    const auto cell = (mouse - graph.Origin) / graph.Step;
    if (cell.x < 0.f || cell.y < 0.f) return {};
    const auto x = uint32_t(cell.x), y = uint32_t(cell.y);
    // Whether the cell is in the node's input column, in its output column, or in neither.
    const auto column = [&](uint32_t n) -> std::optional<bool> {
        const auto &box = graph.Page.Boxes[n];
        if (n == held || y < box.Y || y >= box.Y + HeightOf(graph, n)) return {};
        if (box.In && x == box.X) return true;
        if (box.Out && x + 1u == box.X + box.W) return false;
        return {};
    };
    // The expanded node's open tray covers the nodes under it.
    Dock dock;
    for (const auto n : graph.Shown) {
        if (column(n) && (dock.Node == NoNode || n == graph.Expanded)) dock.Node = n;
    }
    const auto node = dock.Node;
    if (node == NoNode) return dock;
    const bool input = *column(node);
    // The links run from the outputs of `from` to the inputs of `to`.
    const auto from = input ? held : node, to = input ? node : held;
    if (view.Authored && Upstream(view.InputFirst, std::span{view.Wires}, from, [](uint32_t) { return true; })[to]) return dock;
    const auto inputs = InputsOf(view, node);
    std::optional<uint16_t> under; // The socket whose port cell is under the mouse
    for (uint16_t s = input ? 0u : inputs; s < (input ? inputs : uint16_t(NodeSockets(view, node).size())); ++s) {
        const bool in_closed_tray = input && node != graph.Expanded && InTray(view, node, s);
        if (!in_closed_tray && RowAt(graph, SocketAt(graph, node, s).y) == y) under = s;
    }
    const auto takes = [&](uint16_t i) {
        return LinksInto(view, to, i) == 0u || NodeSockets(view, to)[i].Multi || (input && view.Authored && Inserts(view, held, node, i));
    };
    // Pairs each output with the first input that takes it, or with `only`, that socket of the node alone, where a linked input also counts.
    const auto pair_up = [&](std::optional<uint16_t> only) {
        std::vector<bool> paired(InputsOf(view, to));
        for (auto o = InputsOf(view, from); o < uint16_t(NodeSockets(view, from).size()); ++o) {
            for (uint16_t i = 0u; i < paired.size(); ++i) {
                if (only && (input ? i : o) != *only) continue;
                if (paired[i] || !(takes(i) || (only && input)) || !Accepts(r, view, from, o, to, i)) continue;
                paired[i] = true;
                dock.Links.push_back(input ? std::pair{i, o} : std::pair{o, i});
                break;
            }
        }
    };
    if (under) pair_up(under);
    if (dock.Links.empty()) pair_up({});
    return dock;
}
// Links the held node as the dock pairs its sockets, in place of the staged move.
// The held node loses its placed cell, so the page lays it out from its connections.
// It is inserted into each link that takes it at the node's socket: the link into an input that is not a multi-input, or every link from an output.
void DropOnPorts(const Graph &graph, uint32_t held, const Dock &dock) {
    const auto &view = graph.Page.View;
    action::Cancel();
    if (!action::Begin(action::Phase::Record)) return;
    action::AddToGesture(action::nodes::UnplaceNode{view.Network, view.Nodes[held].Item, OutputBoundary(view, held)});
    for (const auto [socket, held_socket] : dock.Links) {
        if (!view.Authored) {
            action::AddToGesture(AssignOf(view, held, held_socket, dock.Node, socket));
            continue;
        }
        // The first insert links both of the held node's main sockets, and each later one moves the reader's link to its main output.
        bool inserted = false;
        const auto insert = [&](uint32_t reader, uint16_t slot) {
            if (!Inserts(view, held, reader, slot)) return;
            if (!std::exchange(inserted, true)) return action::AddToGesture(InsertOf(view, held, reader, slot));
            const auto to = SlotSocket(view, reader, slot);
            action::AddToGesture(LinkOf(view, held, *MainSocket(view.Nodes[held].Kind, SocketDirection::Out), reader, to, SlotIndex(view, reader, slot)));
            // A multi-input socket gets the new link before the one it replaces.
            if (!NodeSockets(view, reader)[to].Multi) return;
            auto unlink = UnlinkOf(view, reader, slot);
            ++unlink.Index;
            action::AddToGesture(unlink);
        };
        if (socket >= InputsOf(view, dock.Node)) {
            for (const auto slot : std::span{graph.Page.Out}.subspan(graph.Page.OutFirst[dock.Node], graph.Page.OutFirst[dock.Node + 1u] - graph.Page.OutFirst[dock.Node])) {
                if (const auto reader = ReaderOf(view, slot); view.Wires[slot].FromSocket == socket) insert(reader, uint16_t(slot - view.InputFirst[reader]));
            }
        } else if (!NodeSockets(view, dock.Node)[socket].Multi) {
            insert(dock.Node, socket);
        }
        if (!inserted) action::AddToGesture(LinkOf(view, held, held_socket, dock.Node, socket));
    }
}

void Interact(const state::Registry &r, NodesPanel &panel, Graph &graph, const std::vector<uint32_t> &selected, const Hit &hit, const std::optional<Insert> &insert, const Dock &dock) {
    const auto &io = GetIO();
    const auto &view = graph.Page.View;
    const auto &nodes = view.Nodes;
    const auto active_node = ActiveIndex(r, view);
    const auto mouse = io.MousePos;
    const bool mouse_moved = io.MouseDelta.x != 0.f || io.MouseDelta.y != 0.f;
    // A scene or document page selects its nodes' entities, and no active node keeps the active one.
    const auto select = [&](const std::vector<uint32_t> &selection, uint32_t active) {
        const auto active_item = active == NoNode ? state::Null : nodes[active].Item;
        if (view.Authored) action::Emit(action::nodes::SelectItems{view.Network, active_item, ItemsOf(view, selection)});
        else action::Emit(action::selection::SelectNodes(r, ItemsOf(view, selection), active_item));
    };
    const auto open_add = [&](bool linking) {
        const auto cell = (mouse - graph.Origin) / graph.Step;
        panel.AddAt = {std::max(std::floor(cell.x), 0.f), std::max(std::floor(cell.y), 0.f)};
        panel.Linking = linking;
        OpenPopup("##AddNode");
    };
    // Moves the selected nodes together to the cells nearest the mouse where they fit among the other nodes.
    // The graph takes the staged cells, so this frame draws them.
    const auto follow = [&](bool first) {
        const auto &held = graph.Page.Boxes[panel.DragNode];
        const int64_t wanted_x = int64_t(panel.MoveStart.x) + std::lround((mouse.x - panel.Press.x) / graph.Step), wanted_y = int64_t(panel.MoveStart.y) + std::lround((mouse.y - panel.Press.y) / graph.Step);
        const auto [dx, dy] = Place(graph.Page.Boxes, graph.Page.Nodes, selected, wanted_x - int64_t(held.X), wanted_y - int64_t(held.Y), [&](uint32_t n) { return !std::ranges::binary_search(selected, n); });
        if (first || dx != 0 || dy != 0) {
            action::nodes::MoveNodes move{view.Network, {}};
            for (const auto n : selected) {
                auto &box = graph.Page.Boxes[n];
                box.X = uint32_t(int64_t(box.X) + dx);
                box.Y = uint32_t(int64_t(box.Y) + dy);
                SetRows(graph.Page.Nodes, n, box);
                move.Positions.push_back({nodes[n].Item, {box.X, box.Y}, OutputBoundary(view, n)});
            }
            for (const auto n : selected) FileWires(graph.Page, n);
            action::Emit(std::move(move), action::Phase::Stage);
        }
    };
    const auto end_move = [&](bool commit) {
        if (panel.Moved && commit && !dock.Links.empty()) {
            DropOnPorts(graph, panel.DragNode, dock);
        } else if (panel.Moved && commit) {
            if (insert) action::Emit(InsertOf(view, panel.DragNode, insert->ToNode, insert->ToSlot, insert->Shift), action::Phase::Stage);
            action::Commit();
        } else if (panel.Moved) {
            action::Cancel();
        }
        panel.Drag = NodeDrag::None;
    };
    if (panel.Drag == NodeDrag::Grab) {
        if (panel.DragNode == NoNode) {
            if (nodes.size() <= panel.GrabAfter) {
                panel.Drag = NodeDrag::None;
                return;
            }
            panel.DragNode = active_node;
        }
        if (IsMouseClicked(ImGuiMouseButton_Left)) end_move(true);
        else if (IsKeyPressed(ImGuiKey_Escape) || IsMouseClicked(ImGuiMouseButton_Right)) end_move(false);
        else if (!panel.Moved || mouse_moved) {
            const bool first = !panel.Moved;
            panel.Moved = true;
            follow(first);
        }
        return;
    }
    if (panel.Drag == NodeDrag::None && panel.Hovered) {
        if (IsMouseClicked(ImGuiMouseButton_Left)) {
            panel.Press = vec2{mouse};
            if (hit.Type == HitType::Socket) {
                // Dragging from a linked input picks up that link, and the drag then holds the link's output end.
                panel.Drag = NodeDrag::Wire;
                if (hit.Socket < InputsOf(view, hit.Node) && LinksInto(view, hit.Node, hit.Socket) > 0u) {
                    const auto slot = SlotOf(view, hit.Node, hit.Socket, hit.Index);
                    const auto link = view.Wires[view.InputFirst[hit.Node] + slot];
                    panel.DetachNode = hit.Node;
                    panel.DetachSlot = slot;
                    panel.DragNode = link.FromNode;
                    panel.DragSocket = link.FromSocket;
                } else {
                    panel.DetachNode = NoNode;
                    panel.DragNode = hit.Node;
                    panel.DragSocket = hit.Socket;
                }
            } else if (hit.Type == HitType::Node || hit.Type == HitType::Parameters) {
                if (IsMouseDoubleClicked(ImGuiMouseButton_Left) && nodes[hit.Node].Item != view.Network && EnterNode(r, panel, nodes[hit.Node].Item)) {
                    return;
                }
                const bool is_selected = Selected(r, view, hit.Node), active = hit.Node == active_node;
                if (io.KeyShift && is_selected && active) {
                    auto remaining = selected;
                    std::erase(remaining, hit.Node);
                    select(remaining, NoNode);
                } else {
                    if (!io.KeyShift && !is_selected) {
                        select({hit.Node}, hit.Node);
                    } else if (!active) {
                        auto extended = selected;
                        if (!is_selected) extended.insert(std::ranges::lower_bound(extended, hit.Node), hit.Node);
                        select(extended, hit.Node);
                    }
                    panel.Drag = NodeDrag::Move;
                    panel.DragNode = hit.Node;
                    panel.Moved = false;
                }
            } else {
                panel.Drag = NodeDrag::Box;
            }
        } else if (IsMouseClicked(ImGuiMouseButton_Right)) {
            if (!view.Authored && (hit.Type == HitType::Node || hit.Type == HitType::Socket)) {
                panel.ContextNode = hit.Node;
                panel.ContextSocket = hit.Type == HitType::Socket ? hit.Socket : UINT16_MAX;
                OpenPopup("##NodeMenu");
            } else open_add(false);
        }
    }
    if (panel.Drag == NodeDrag::None && panel.Hovered) {
        constexpr auto Key{ImGuiInputFlags_RouteGlobal};
        if (Shortcut(ImGuiMod_Shift | ImGuiKey_A, Key)) {
            open_add(false);
        } else if (Shortcut(ImGuiKey_X, Key) || Shortcut(ImGuiKey_Delete, Key) || Shortcut(ImGuiKey_Backspace, Key)) {
            auto items = ItemsOf(view, selected);
            // A definition's boundaries stay.
            if (view.Authored) std::erase(items, view.Network);
            if (items.empty()) {
            } else if (view.Authored) action::Emit(action::nodes::DeleteNodes{view.Network, items});
            else if (!r.all_of<BoneIndex>(items.front())) {
                // A definition a subnet node uses stays, and its refused delete is a step of its own, which reports it.
                std::vector<state::Entity> deleted, used;
                for (const auto item : items) (item != state::Document && r.all_of<SubnetDefinition>(item) && DefinitionUsed(UsesOf(r, panel), item) ? used : deleted).push_back(item);
                if (!deleted.empty()) action::Emit(action::nodes::DeleteItems{std::move(deleted)});
                for (const auto definition : used) action::Post(action::nodes::DeleteDefinition{definition});
            }
            // Bones are deleted the way the viewport deletes them, in the armature's Edit mode.
            else if (IsBoneEditMode(r)) action::Emit(action::bone::DeleteSelected{});
        } else if (const bool replace = Shortcut(ImGuiMod_Shift | ImGuiKey_J, Key); replace || Shortcut(ImGuiKey_J, Key)) {
            // The links go into one undo step.
            if (auto links = MakeLinks(graph.Page, selected, replace); !links.empty() && action::Begin(action::Phase::Record)) {
                for (auto &link : links) action::AddToGesture(std::move(link));
            }
        } else if (Shortcut(ImGuiMod_Shift | ImGuiKey_D, Key)) {
            auto copied = ItemsOf(view, selected);
            std::erase(copied, view.Network);
            if (view.Authored && !copied.empty()) {
                // The grab starts the copies from the cell of the node whose copy becomes active.
                const auto &held = graph.Page.Boxes[std::ranges::binary_search(selected, active_node) ? active_node : selected.back()];
                action::Emit(action::nodes::DuplicateNodes{view.Network, std::move(copied)});
                Grab(panel, uint32_t(nodes.size()), {held.X, held.Y});
            }
        } else if (Shortcut(ImGuiMod_Ctrl | ImGuiKey_G, Key)) {
            if (view.Authored) {
                auto grouped = ItemsOf(view, selected);
                std::erase(grouped, view.Network);
                if (!grouped.empty()) action::Emit(action::nodes::GroupNodes{view.Network, std::move(grouped)});
            } else if (!selected.empty()) action::Emit(action::nodes::GroupItems{view.Network, ItemsOf(view, selected)});
        } else if (Shortcut(ImGuiKey_G, Key)) {
            if (active_node != NoNode && Selected(r, view, active_node)) Grab(panel, uint32_t(nodes.size()) - 1u, {graph.Page.Boxes[active_node].X, graph.Page.Boxes[active_node].Y});
        } else if (Shortcut(ImGuiKey_A, Key)) {
            std::vector<uint32_t> all(nodes.size());
            std::ranges::iota(all, 0u);
            if (selected != all) select(all, NoNode);
        } else if (Shortcut(ImGuiMod_Alt | ImGuiKey_A, Key)) {
            if (!selected.empty()) select({}, NoNode);
        } else if (Shortcut(ImGuiKey_L, Key)) {
            if (r.all_of<NetworkLayout>(view.Network)) action::Emit(action::nodes::AutoLayout{view.Network});
        }
    }
    if (panel.Drag == NodeDrag::Move) {
        if (!IsMouseDown(ImGuiMouseButton_Left)) end_move(true);
        else if (panel.Moved && IsKeyPressed(ImGuiKey_Escape)) end_move(false);
        else if (IsMouseDragging(ImGuiMouseButton_Left)) {
            const bool first = !panel.Moved;
            if (first && panel.DragNode == active_node) {
                panel.Moved = true;
                panel.MoveStart = {graph.Page.Boxes[panel.DragNode].X, graph.Page.Boxes[panel.DragNode].Y};
            }
            if (!panel.Moved) panel.Drag = NodeDrag::None;
            else if (first || mouse_moved) follow(first);
        }
    }
    if (panel.Drag == NodeDrag::Wire && IsKeyPressed(ImGuiKey_Escape)) {
        panel.Drag = NodeDrag::None;
        panel.DetachNode = NoNode;
    }
    if (panel.Drag == NodeDrag::Wire && !IsMouseDown(ImGuiMouseButton_Left)) {
        bool picking = false;
        if (hit.Type == HitType::Socket) {
            const bool dropping = graph.DropNode == hit.Node && graph.DropSocket == hit.Socket;
            if (Accepts(r, view, panel.DragNode, panel.DragSocket, hit.Node, hit.Socket)) Connect(r, panel, view, hit.Node, hit.Socket, dropping ? graph.DropIndex : UINT16_MAX);
        } else if (hit.Type == HitType::Parameters || hit.Type == HitType::Node) picking = DropOnNode(r, panel, graph, hit.Node, hit.Type == HitType::Parameters);
        else if (panel.DetachNode != NoNode) Disconnect(r, view, panel.DetachNode, panel.DetachSlot);
        else if (panel.Hovered && view.Authored) open_add(true);
        if (!picking) panel.DetachNode = NoNode;
        panel.Drag = NodeDrag::None;
    }
    if (panel.Drag == NodeDrag::Box && !IsMouseDown(ImGuiMouseButton_Left)) {
        const auto press = Im(panel.Press);
        if (ImLengthSqr(mouse - press) >= io.MouseDragThreshold * io.MouseDragThreshold) {
            const ImRect box{ImMin(press, mouse), ImMax(press, mouse)};
            std::vector<uint32_t> banded;
            ForEachIn(graph.Page.Nodes, RowAt(graph, box.Min.y), RowAt(graph, box.Max.y), [&](uint32_t n) {
                if (box.Overlaps(RectOf(graph, n))) banded.push_back(n);
            });
            if (io.KeyShift) banded.append_range(selected);
            SortUnique(banded);
            if (banded != selected) select(banded, NoNode);
        } else if (!io.KeyShift && !selected.empty()) {
            select({}, NoNode);
        }
        panel.Drag = NodeDrag::None;
    }
}

char Lower(char c) { return char(std::tolower(static_cast<unsigned char>(c))); }
bool Matches(std::string_view label, std::string_view text) { return text.empty() || !std::ranges::search(label, text, {}, Lower, Lower).empty(); }
bool StartsWith(std::string_view label, std::string_view text) { return std::ranges::starts_with(label, text, {}, Lower, Lower); }
// An Add menu entry: the node it adds, its label, and the groups it is filed under, such as "Operators > Boolean >".
struct AddEntry {
    action::nodes::NodeSpec Node;
    const char *Label;
    std::string Groups;
};
// Each kind a node is added by, each followed by one entry per operation.
const auto AddEntries{[]<typename... Ks>(TypeList<Ks...>) {
    std::vector<AddEntry> entries;
    const auto add = [&]<typename K> {
        if constexpr (Kind<K>.Added) {
            const auto groups = std::format("{} >", GroupNames[uint8_t(Kind<K>.Group)]);
            entries.push_back({action::nodes::KindNode{KindOf<K>}, KindTitles[KindOf<K>], groups});
            if constexpr (Operated<K>) {
                for (const auto op : field::EnumValues<decltype(K::Operation)>) entries.push_back({action::nodes::KindNode{KindOf<K>, uint32_t(op)}, field::EnumLabel(op), std::format("{} {} >", groups, KindTitles[KindOf<K>])});
            }
        }
    };
    (add.template operator()<Ks>(), ...);
    return entries;
}(NodeKinds{})};
// The offered ops, by group, each group in title order.
const std::vector<AddEntry> &OpEntries() {
    static const auto entries = [] {
        std::vector<AddEntry> entries;
        for (const auto &group : Ops().Groups) {
            for (const auto op : group.Ops) entries.push_back({action::nodes::OpNode{std::string{Ops().Ops[op].Node}}, Ops().Ops[op].Title.c_str(), std::format("{} >", group.Title)});
        }
        return entries;
    }();
    return entries;
}
// The subnets the network's Add menu offers: each built-in definition, then each user definition that does not contain the network.
std::vector<AddEntry> SubnetEntries(const state::Registry &r, state::Entity network) {
    std::vector<AddEntry> entries;
    for (uint16_t macro = 0u; macro < Macros.size(); ++macro) entries.push_back({action::nodes::MacroNode{macro}, Macros[macro].Name, "Subnets >"});
    for (const auto definition : SortedEntities(r.view<const SubnetDefinition>())) {
        if (!Uses(r, definition, network)) entries.push_back({action::nodes::DefinitionNode{definition}, r.get<const Name>(definition).Value.c_str(), "Subnets >"});
    }
    return entries;
}
// Search labels case-insensitively, placing prefix matches first.
// The kinds and the subnets come first when `kinds` is set, then the offered ops, which also match by MaterialX node name.
std::vector<const AddEntry *> SearchEntries(std::span<const AddEntry> subnets, std::string_view text, bool kinds) {
    std::vector<const AddEntry *> found;
    const auto search = [&](std::span<const AddEntry> entries, bool by_node) {
        for (const auto &entry : entries) {
            if (Matches(entry.Label, text) || (by_node && Matches(std::get<action::nodes::OpNode>(entry.Node).Node, text))) found.push_back(&entry);
        }
    };
    if (kinds) {
        search(AddEntries, false);
        search(subnets, false);
    }
    search(OpEntries(), true);
    std::ranges::stable_partition(found, [&](const AddEntry *entry) {
        const auto *op = std::get_if<action::nodes::OpNode>(&entry->Node);
        return StartsWith(entry->Label, text) || (op && StartsWith(op->Node, text));
    });
    return found;
}

int MoveHighlight(ImGuiInputTextCallbackData *data) {
    *static_cast<int *>(data->UserData) += data->EventKey == ImGuiKey_DownArrow ? 1 : -1;
    return 0;
}
// A menu's search text, the result highlighted before this frame's keys, and whether Enter was pressed in the field.
struct Search {
    std::string_view Text;
    int Previous;
    bool Enter;
};
// Draws a menu's search field, which takes the keyboard as the menu appears.
Search SearchField(NodesPanel &panel) {
    if (IsWindowAppearing()) {
        panel.Search = {};
        panel.SearchHighlight = 0;
        SetKeyboardFocusHere();
    }
    const auto previous = panel.SearchHighlight;
    SetNextItemWidth(GetFontSize() * 14.f);
    const bool enter = InputTextWithHint("##Search", "Search", panel.Search.data(), panel.Search.size(), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackHistory, MoveHighlight, &panel.SearchHighlight);
    if (IsItemEdited()) panel.SearchHighlight = 0;
    return {panel.Search.data(), previous, enter};
}

// A socket of a node an Add menu entry adds, which the Add menu offers to link a dropped wire through.
struct LinkingEntry {
    action::nodes::NodeSpec Node;
    uint16_t Socket;
    procedure::Socket Shown; // The socket as a new node has it
    std::string Groups;
    const char *SocketName;
    std::string Label;
};
// The sockets of every kind a node is added by and of every offered op.
const std::vector<LinkingEntry> &LinkingEntries() {
    static const auto entries = [] {
        std::vector<LinkingEntry> entries;
        ForEachKind([&]<typename K> {
            if constexpr (Kind<K>.Added) {
                const auto *title = KindTitles[KindOf<K>];
                const auto sockets = KindSockets(KindOf<K>);
                for (uint16_t s = 0u; s < uint16_t(sockets.size()); ++s) {
                    const auto *name = LabelOf(sockets[s]);
                    entries.push_back({action::nodes::KindNode{KindOf<K>}, s, sockets[s], std::string{title} + " >", name, std::string{title} + " " + name});
                }
            }
        });
        const auto &ops = Ops().Ops;
        for (const auto &group : Ops().Groups) {
            for (const auto op : group.Ops) {
                const auto &entry = ops[op];
                const auto &definition = *DefinitionOf(entry.Node);
                const bool varies = OpVaries(definition, Op{}, [](uint16_t) { return OpInput::Unset; });
                for (uint16_t s = 0u; s < uint16_t(definition.Sockets.size()); ++s) {
                    auto socket = definition.Sockets[s];
                    if (s >= definition.Inputs) socket.Varies = varies;
                    entries.push_back({action::nodes::OpNode{std::string{entry.Node}}, s, socket, entry.Title + " >", entry.Labels[s].c_str(), entry.Title + " " + entry.Labels[s]});
                }
            }
        }
        return entries;
    }();
    return entries;
}
constexpr int ResultRows{10};
// Draws each row's groups before its label, which texts(i) returns.
// Returns the clicked row, or the highlighted row on Enter, and closes the popup.
std::optional<int> DrawResults(int count, float width, int &highlight, int previous, bool enter, auto &&texts) {
    highlight = std::max(0, std::min(highlight, count - 1));
    const float rows = float(std::min(count, ResultRows));
    BeginChild("##results", {width, rows > 0.f ? rows * GetTextLineHeightWithSpacing() : 1.f}, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
    std::optional<int> picked;
    for (int i = 0; i < count; ++i) {
        PushID(i);
        const auto at = GetCursorScreenPos();
        if (Selectable("##entry", i == highlight)) picked = i;
        PopID();
        SetCursorScreenPos(at);
        const auto [groups, label] = texts(i);
        TextDisabled("%s", groups);
        SameLine();
        TextUnformatted(label);
        if (i == highlight && highlight != previous) SetScrollHereY();
    }
    EndChild();
    if (enter && count > 0) picked = highlight;
    if (picked) CloseCurrentPopup();
    return picked;
}
// Draws the entries that match the search, each after its groups, and returns the node of the one picked.
std::optional<action::nodes::NodeSpec> PickResult(const state::Registry &r, state::Entity network, NodesPanel &panel, const Search &search, bool kinds) {
    const auto subnets = kinds ? SubnetEntries(r, network) : std::vector<AddEntry>{};
    const auto found = SearchEntries(subnets, search.Text, kinds);
    const auto picked = DrawResults(int(found.size()), GetFontSize() * 22.f, panel.SearchHighlight, search.Previous, search.Enter, [&](int i) { return std::pair{found[i]->Groups.c_str(), found[i]->Label}; });
    return picked ? std::optional{found[size_t(*picked)]->Node} : std::nullopt;
}
// Draws a menu item for each op of the group and returns the MaterialX node of the op clicked.
std::optional<std::string> OpItems(const OpGroup &group) {
    std::optional<std::string> picked;
    for (const auto op : group.Ops) {
        if (MenuItem(Ops().Ops[op].Title.c_str())) picked = std::string{Ops().Ops[op].Node};
    }
    return picked;
}
// Draws a menu for each op group and returns the MaterialX node of the op clicked.
// With `kinds`, a group titled as a kind group is left to that group's menu.
std::optional<std::string> OpMenus(bool kinds) {
    std::optional<std::string> picked;
    for (const auto &group : Ops().Groups) {
        if ((kinds && std::ranges::contains(GroupNames, group.Title, [](const char *name) { return std::string_view{name}; })) || !BeginMenu(group.Title.c_str())) continue;
        if (auto op = OpItems(group)) picked = std::move(op);
        EndMenu();
    }
    return picked;
}
// A search field over a menu for each op group, or over the ops matching its text.
// Returns the MaterialX node of the op picked.
std::optional<std::string> DrawOpChoice(const state::Registry &r, NodesPanel &panel) {
    const auto search = SearchField(panel);
    if (search.Text.empty()) return OpMenus(false);
    const auto picked = PickResult(r, state::Null, panel, search, false);
    return picked ? std::optional{std::get<action::nodes::OpNode>(*picked).Node} : std::nullopt;
}

// A material's graph offers only ops.
void DrawAddMenu(const state::Registry &r, NodesPanel &panel, const NodeView &view, bool interactive) {
    if (!BeginPopup("##AddNode")) return;
    const auto network = view.Network;
    const bool kinds = !r.all_of<MaterialGraph>(network);
    const auto count = uint32_t(view.Nodes.size());
    const bool linking = panel.Linking && panel.DragNode < count;
    const auto add = [&](action::nodes::NodeSpec node) {
        if (!interactive) return;
        action::Emit(action::nodes::AddNode{network, std::move(node)});
        Grab(panel, count, panel.AddAt);
    };
    const auto search = SearchField(panel);
    if (linking) {
        const auto held = NodeSockets(view, panel.DragNode)[panel.DragSocket];
        std::vector<const LinkingEntry *> entries;
        for (const auto &entry : LinkingEntries()) {
            if ((kinds || std::holds_alternative<action::nodes::OpNode>(entry.Node)) && Joins(held, entry.Shown) && Matches(entry.Label, search.Text)) entries.push_back(&entry);
        }
        float width = GetFontSize() * 14.f;
        for (const auto *entry : entries) width = std::max(width, CalcTextSize(entry->Label.c_str()).x + GetFontSize() * 2.f);
        const auto picked = DrawResults(int(entries.size()), width, panel.SearchHighlight, search.Previous, search.Enter, [&](int i) { return std::pair{entries[i]->Groups.c_str(), entries[i]->SocketName}; });
        if (picked && interactive) {
            const auto &entry = *entries[*picked];
            action::Emit(action::nodes::AddNode{network, entry.Node}, action::Phase::Stage);
            // The added node becomes the network's active node, which the next frame links through the entry's socket.
            const auto item = view.Nodes[panel.DragNode].Item;
            const auto socket = SocketId(view, panel.DragNode, panel.DragSocket);
            panel.LinkingAdd = held.Direction == SocketDirection::Out ? action::nodes::Link{network, item, socket, state::Null, entry.Socket} : action::nodes::Link{network, state::Null, entry.Socket, item, socket};
            panel.LinkingActive = ActiveNodeOf(r, network);
        }
    } else if (search.Text.empty() && !kinds) {
        if (auto op = OpMenus(false)) add(action::nodes::OpNode{std::move(*op)});
    } else if (search.Text.empty()) {
        for (uint8_t group = 0u; group < uint8_t(std::size(GroupNames)); ++group) {
            if (!BeginMenu(GroupNames[group])) continue;
            for (const auto &entry : AddEntries) {
                const auto &node = std::get<action::nodes::KindNode>(entry.Node);
                if (!node.Operation && KindGroups[node.Kind] == KindGroup(group) && MenuItem(entry.Label)) add(entry.Node);
            }
            // The ops of the MaterialX node group with this title follow the group's kinds.
            if (const auto ops = std::ranges::find(Ops().Groups, std::string_view{GroupNames[group]}, &OpGroup::Title); ops != Ops().Groups.end()) {
                Separator();
                if (auto op = OpItems(*ops)) add(action::nodes::OpNode{std::move(*op)});
            }
            EndMenu();
        }
        if (auto op = OpMenus(true)) add(action::nodes::OpNode{std::move(*op)});
        if (BeginMenu("Subnets")) {
            const auto subnets = SubnetEntries(r, network);
            for (uint32_t i = 0u; i < subnets.size(); ++i) {
                if (i == Macros.size()) Separator();
                PushID(int(i));
                if (MenuItem(subnets[i].Label)) add(subnets[i].Node);
                PopID();
            }
            EndMenu();
        }
    } else if (auto picked = PickResult(r, network, panel, search, kinds)) {
        add(std::move(*picked));
    }
    if (IsKeyPressed(ImGuiKey_Escape)) CloseCurrentPopup();
    EndPopup();
}

void DrawPortPicker(const state::Registry &r, NodesPanel &panel, const Graph &graph, bool interactive) {
    if (panel.PickNode == NoNode) return;
    const auto &view = graph.Page.View;
    if (!BeginPopup("##PickPort")) {
        // Closing the picker without a choice leaves a picked-up link where it was.
        panel.PickNode = NoNode;
        if (panel.Drag != NodeDrag::Wire) panel.DetachNode = NoNode;
        return;
    }
    const auto node = panel.PickNode;
    if (node < view.Nodes.size() && panel.DragNode < view.Nodes.size()) {
        for (const auto s : LinkingSockets(r, graph, panel.DragNode, panel.DragSocket, node, panel.PickTray)) {
            if (MenuItem(LabelOf(view, node, s)) && interactive) Connect(r, panel, view, node, s);
        }
    } else {
        CloseCurrentPopup();
    }
    if (IsKeyPressed(ImGuiKey_Escape)) CloseCurrentPopup();
    EndPopup();
}

void RecordHover(const state::Registry &r, const NodesPanel &panel, const NodePage &page, const Hit &hit) {
    if (hit.Type == HitType::None) return;
    const auto &view = page.View;
    const auto &node = view.Nodes[hit.Node];
    const auto entity = node.Item;
    const auto kind = node.Kind;
    const auto *type = view.Authored ? node.Type : ItemLabel(r, node.Item);
    const auto *op = view.Authored && kind == KindOf<Op> ? &view.Authored->get<const Op>(entity) : nullptr;
    const auto declaration = op ? view.Declarations[hit.Node] : std::nullopt;
    if (hit.Type == HitType::Node) {
        const auto *error = view.Authored ? FaultOf(r, page, entity) : nullptr;
        static std::string status;
        if (error) status = FaultText(*view.Authored, entity, *error);
        const auto *doc = op ? OpDoc(op->Node, declaration) : nullptr;
        ui::Hovered = {.Label = view.Authored ? OperationTitle(*view.Authored, entity, kind) : type, .Path = type, .Type = type, .Help = doc ? doc : view.Authored ? KindHelp[kind] :
                                                                                                                                                        nullptr,
                       .Status = error ? status.c_str() : nullptr};
    } else if (hit.Type == HitType::Parameters) {
        ui::Hovered = {.Label = "Parameters", .Path = type, .Help = "Hover to show the sockets of the node's value parameters. Drop a wire here to link it to one of them."};
    } else {
        const auto socket = NodeSockets(view, hit.Node)[hit.Socket];
        if (const auto *label = LabelOf(view, hit.Node, hit.Socket); !view.Authored && *label) SetTooltip("%s", label);
        const bool held = panel.DragNode == hit.Node && panel.DragSocket == hit.Socket;
        const bool dimmed = panel.Drag == NodeDrag::Wire && !held && !Accepts(r, view, panel.DragNode, panel.DragSocket, hit.Node, hit.Socket);
        const bool linked = socket.Direction == SocketDirection::In && IsLinked(view, hit.Node, hit.Socket);
        const auto *port = op ? DeclaredPort(op->Node, declaration, socket.Name, socket.Direction == SocketDirection::In) : nullptr;
        ui::Hovered = {
            .Label = view.Authored ? LabelOf(view, hit.Node, hit.Socket) : "Reference",
            .Path = type,
            .Type = SocketTypeText(socket),
            // A linked input's parameter value is unused, so its help says how to move the link.
            .Help = !view.Authored ? "Reference to a document item" : socket.Multi ? "Accepts any number of links. They apply in order from top to bottom. Drag a link along the socket to reorder it." :
                linked                                                             ? "Drag to move this link to another input. Drop it on empty canvas to unlink." :
                kind == KindOf<Subnet>                                             ? "Named subnet interface port" :
                op                                                                 ? (port && !port->Doc.empty() ? port->Doc.c_str() : nullptr) :
                                                                                     SocketHelps[SocketFirst[kind] + hit.Socket],
            .Status = dimmed ? "Cannot link with the held wire" : linked ? "Linked" :
                                                                           nullptr,
        };
    }
}

void DrawCanvas(const state::Registry &r, FrameState &frame, NodesPanel &panel, const NodeTheme &theme, NodePage &page, bool interactive) {
    const auto &io = GetIO();
    const auto &view = page.View;
    const auto &nodes = view.Nodes;
    if ((panel.Drag == NodeDrag::Move || panel.Drag == NodeDrag::Wire || (panel.Drag == NodeDrag::Grab && panel.DragNode != NoNode)) && panel.DragNode >= nodes.size()) panel.Drag = NodeDrag::None;
    if (panel.DetachNode != NoNode && (panel.DetachNode >= nodes.size() || panel.DetachSlot >= SlotsOf(view, panel.DetachNode))) {
        panel.DetachNode = NoNode;
        if (panel.Drag == NodeDrag::Wire) panel.Drag = NodeDrag::None;
    }
    const auto origin = GetCursorScreenPos();
    const auto size = GetContentRegionAvail();
    InvisibleButton("##canvas", {std::max(size.x, 1.f), std::max(size.y, 1.f)}, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
    const auto end = GetItemRectMax();
    const auto mouse = io.MousePos;
    panel.Hovered = IsItemHovered() || (IsItemActive() && ImRect{origin, end}.Contains(mouse));
    SetItemKeyOwner(ImGuiKey_MouseWheelY);
    auto &canvas = panel.Canvas;
    if (interactive) {
        if (panel.Hovered) {
            const auto wheel = std::exchange(frame.PreciseWheelDelta, vec2{0});
            const auto magnification = std::exchange(frame.Magnification, 0.f);
            const bool pan = frame.PreciseScrolling && !io.KeyCtrl && !io.KeySuper;
            if (pan) canvas.Offset = vec2{Im(canvas.Offset) + Im(wheel) * TrackpadPoints};
            const float steps = pan ? 0.f : wheel.y * (frame.PreciseScrolling ? TrackpadPoints / TrackpadStepPoints : 1.f);
            if (const auto zoom = std::clamp(canvas.Zoom * std::pow(ZoomStep, steps) * (1.f + magnification), MinZoom, MaxZoom); zoom != canvas.Zoom) {
                const auto at = (mouse - origin - Im(canvas.Offset)) / canvas.Zoom;
                canvas.Offset = vec2{mouse - origin - at * zoom};
                canvas.Zoom = zoom;
            }
        }
        if (panel.Drag == NodeDrag::None && panel.Hovered && IsMouseClicked(ImGuiMouseButton_Middle)) panel.Drag = NodeDrag::Pan;
        if (panel.Drag == NodeDrag::Pan) {
            if (IsMouseDown(ImGuiMouseButton_Middle)) canvas.Offset = vec2{Im(canvas.Offset) + io.MouseDelta};
            else panel.Drag = NodeDrag::None;
        }
    }
    // Clamp the offset so nothing left of or above the margin shows.
    canvas.Offset = {std::min(canvas.Offset.x, 0.f), std::min(canvas.Offset.y, 0.f)};
    const float cell = CellOf(theme);
    const float margin = CanvasMargin * cell * canvas.Zoom;
    const bool restyled = page.Theme != theme.Sizes || page.Cell != cell || page.Font != GetFontSize();
    const bool rebuilt = std::exchange(page.Rebuilt, false);
    const bool relaid = (rebuilt || restyled) && Measure(page, theme, cell, restyled);
    // A rebuilt view can show another network with the same texts and wires, whose own placed cells the boxes take.
    if (const auto cells = r.Resources.get<const NetworkChanges>().CellsRevision; relaid || rebuilt || page.Cells != cells) {
        Arrange(r, page, relaid);
        page.Cells = cells;
    }
    Graph graph{theme, origin + Im(canvas.Offset) + ImVec2{margin, margin}, cell * canvas.Zoom, canvas.Zoom, NoNode, page};
    if (panel.Expanded < nodes.size() && page.Rows[panel.Expanded].Tray > 0u) graph.Expanded = panel.Expanded;
    // The frame draws and tests the nodes and wires whose rows meet the canvas, and the expanded node.
    const auto top = RowAt(graph, origin.y), bottom = RowAt(graph, end.y);
    ForEachIn(page.Nodes, top, bottom, [&](uint32_t n) { graph.Shown.push_back(n); });
    if (graph.Expanded != NoNode) graph.Shown.push_back(graph.Expanded);
    SortUnique(graph.Shown);
    ForEachIn(page.Wires, top, bottom, [&](uint32_t slot) { graph.ShownWires.push_back(slot); });
    // The expanded node's open tray reaches below its box, so the wires into it show with it.
    if (graph.Expanded != NoNode) {
        for (auto slot = view.InputFirst[graph.Expanded]; slot < view.InputFirst[graph.Expanded + 1u]; ++slot) {
            if (view.Wires[slot].FromNode != NoNode) graph.ShownWires.push_back(slot);
        }
    }
    SortUnique(graph.ShownWires);
    const bool wiring = panel.Drag == NodeDrag::Wire;
    if (wiring || panel.PickNode != NoNode) {
        graph.DetachNode = panel.DetachNode;
        graph.DetachSlot = panel.DetachSlot;
    }
    const auto hit = panel.Hovered ? HitTest(graph, mouse) : Hit{};
    // While the held wire is over a multi-input socket it can link with, the mouse height sets the index it would be inserted at.
    std::optional<ImVec2> drop_at;
    if (wiring && hit.Type == HitType::Socket && NodeSockets(view, hit.Node)[hit.Socket].Multi && Accepts(r, view, panel.DragNode, panel.DragSocket, hit.Node, hit.Socket)) {
        const auto at = SocketAt(graph, hit.Node, hit.Socket);
        const auto shown = ShownLinks(graph, hit.Node, hit.Socket);
        graph.DropNode = hit.Node;
        graph.DropSocket = hit.Socket;
        for (uint32_t k = 0u; k < shown; ++k) graph.DropIndex += MultiAt(graph, at, k, shown).y < mouse.y;
        drop_at = MultiAt(graph, at, graph.DropIndex, shown + 1u);
    }
    if (panel.Drag == NodeDrag::None || wiring) RecordHover(r, panel, page, hit);
    // A built-in definition's page is read-only, and selects only the node the panel inspects.
    const bool editable = &AuthoredOf(r, view) == &r;
    const auto inspected = editable ? NoNode : NodeOf(view, panel.Canvas.Inspected);
    const auto selected = [&] {
        if (editable) return SelectedNodes(r, view);
        std::vector<uint32_t> nodes_of;
        for (uint32_t n = 0u; n < nodes.size(); ++n)
            if (panel.Canvas.Inspected != state::Null && nodes[n].Item == panel.Canvas.Inspected) nodes_of.push_back(n);
        return nodes_of;
    }();
    const bool holding = (panel.Drag == NodeDrag::Move && panel.Moved) || (panel.Drag == NodeDrag::Grab && panel.DragNode != NoNode);
    // A held node dropped over a port column links with that node, and one dropped anywhere else is inserted into a link that crosses it.
    const bool holds_one = holding && selected == std::vector{panel.DragNode};
    const auto dock = holds_one && panel.Hovered ? DockOf(r, graph, panel.DragNode, mouse) : Dock{};
    const auto insert = holds_one && dock.Links.empty() ? InsertTarget(graph, panel.DragNode) : std::nullopt;
    if (interactive && editable) Interact(r, panel, graph, selected, hit, insert, dock);
    // A click on a read-only page inspects the node under the mouse, which is panel state.
    if (interactive && !editable && panel.Hovered && IsMouseClicked(ImGuiMouseButton_Left)) panel.Canvas.Inspected = hit.Type == HitType::Node || hit.Type == HitType::Parameters ? nodes[hit.Node].Item : state::Null;
    const bool stays_open = graph.Expanded != NoNode && (TrayRect(graph, graph.Expanded).Contains(mouse) || (wiring && panel.DragNode == graph.Expanded) || panel.PickNode == graph.Expanded);
    if (hit.Type == HitType::Parameters) panel.Expanded = hit.Node;
    else if (!stays_open) panel.Expanded = NoNode;
    const bool idle = panel.Drag == NodeDrag::None && !wiring;
    const auto target = panel.Drag == NodeDrag::None ? std::nullopt : insert;

    auto &draw = *GetWindowDrawList();
    const auto zoom = canvas.Zoom;
    const auto &colors = theme.Colors;
    const auto shown_types = [&](uint32_t n, uint16_t s) { return view.ShownTypes[view.SocketFirst[n] + s]; };
    draw.PushClipRect(origin, end, true);
    draw.AddRectFilled(origin, end, U32(colors.Background));
    const auto grid = U32(colors.Grid), coarse_grid = U32(colors.CoarseGrid);
    const auto coarse = int32_t(std::max<uint8_t>(theme.Sizes.CoarseGridCells, 1u));
    const auto grid_min = ImMax(origin, graph.Origin);
    for (auto i = std::max(0, int32_t(std::floor((origin.x - graph.Origin.x) / graph.Step))); graph.Origin.x + float(i) * graph.Step < end.x; ++i) {
        const float x = graph.Origin.x + float(i) * graph.Step;
        if (x >= origin.x) draw.AddLine({x, grid_min.y}, {x, end.y}, i % coarse == 0 ? coarse_grid : grid);
    }
    for (auto i = std::max(0, int32_t(std::floor((origin.y - graph.Origin.y) / graph.Step))); graph.Origin.y + float(i) * graph.Step < end.y; ++i) {
        const float y = graph.Origin.y + float(i) * graph.Step;
        if (y >= origin.y) draw.AddLine({grid_min.x, y}, {end.x, y}, i % coarse == 0 ? coarse_grid : grid);
    }
    const float thickness = std::max(theme.Sizes.WireWidth * zoom, 1.f);
    PushFont(nullptr, GetStyle().FontSizeBase * zoom);
    const auto font_size = GetFontSize();
    const float spacing = theme.Sizes.LabelSpacing * zoom, label_reach = float(theme.Sizes.MaxLabelCells) * graph.Step;
    const auto draw_label = [&](ImVec2 min, ImVec2 max, const char *text, vec4 color, bool right) {
        if (const float width = CalcTextSize(text).x; right && width <= max.x - min.x) min.x = max.x - width;
        PushStyleColor(ImGuiCol_Text, ImVec4{color.x, color.y, color.z, color.w});
        RenderTextEllipsis(&draw, min, max, max.x, text, nullptr, nullptr);
        PopStyleColor();
    };
    const auto active_node = editable ? ActiveIndex(r, view) : inspected;
    const auto draw_node = [&](uint32_t n) {
        const auto &node = nodes[n];
        const auto entity = node.Item;
        const auto kind = node.Kind;
        const auto [min, max] = RectOf(graph, n);
        // The labels span the columns between the port columns.
        const float labels_min = min.x + (graph.Page.Boxes[n].In ? graph.Step : 0.f), labels_max = max.x - (graph.Page.Boxes[n].Out ? graph.Step : 0.f);
        const auto &rows = graph.Page.Rows[n];
        const bool expanded = n == graph.Expanded;
        const float rounding = theme.Sizes.Rounding * zoom, pad = theme.Sizes.TitlePadding * zoom;
        draw.AddRectFilled(min, max, U32(expanded ? vec4{colors.Node.x, colors.Node.y, colors.Node.z, 1.f} : colors.Node), rounding);
        // The title ends where the node's first cell row does.
        const float title_end = min.y + graph.Step - theme.Sizes.Margin * zoom;
        draw.AddRectFilled(min, {max.x, title_end}, U32(n == view.DisplayNode ? colors.DisplayedTitle : colors.*GroupColors[view.Authored ? uint8_t(KindGroups[kind]) : uint8_t(KindGroup::Sources)]), rounding, ImDrawFlags_RoundTop);
        draw_label({min.x + pad, (min.y + title_end - font_size) * .5f}, {max.x - pad, (min.y + title_end + font_size) * .5f}, node.Title.c_str(), colors.Title, false);
        const auto sockets = NodeSockets(view, n);
        const auto inputs = InputsOf(view, n);
        const auto *fault = view.Authored ? FaultOf(r, page, entity) : nullptr;
        const bool erring = fault != nullptr;
        // The active node shows its outline only while it is selected.
        const bool is_selected = std::ranges::binary_search(selected, n), active = is_selected && active_node != NoNode && entity == nodes[active_node].Item;
        const auto fault_color = fault && *fault == NodeFault::NotFound ? colors.Warning : colors.Error;
        const auto border = active ? colors.Active : is_selected ? colors.Selected :
            erring                                               ? fault_color :
                                                                   colors.Border;
        draw.AddRect(min, max, U32(border), rounding, erring || active || is_selected ? theme.Sizes.HighlightBorderWidth : theme.Sizes.BorderWidth, ImDrawFlags_StrokeCenter);
        if (erring && (active || is_selected)) draw.AddRect(min + ImVec2{3.f, 3.f}, max - ImVec2{3.f, 3.f}, U32(fault_color), rounding, 1.f);
        if (expanded) {
            const auto tray = TrayRect(graph, n);
            draw.AddRectFilled(tray.Min, tray.Max, U32(colors.Tray), rounding);
            draw.AddRect(tray.Min, tray.Max, U32(colors.Outline), rounding, theme.Sizes.OutlineWidth);
        }
        for (uint16_t s = 0u; s < uint16_t(sockets.size()); ++s) {
            const bool parameter = s < inputs && InTray(view, n, s);
            if (parameter && !expanded) continue;
            const auto at = SocketAt(graph, n, s);
            const bool held = wiring && panel.DragNode == n && panel.DragSocket == s;
            const bool dimmed = wiring && !held && !Accepts(r, view, panel.DragNode, panel.DragSocket, n, s);
            const bool hovered = hit.Type == HitType::Socket && hit.Node == n && hit.Socket == s && !dimmed;
            DrawSocket(draw, theme, at, theme.Sizes.SocketRadius * zoom, MultiHalf * graph.Step, sockets[s], dimmed ? colors.Dimmed : SocketColor(colors, shown_types(n, s)), hovered);
            const auto color = dimmed ? colors.DimmedLabel : parameter ? colors.ParameterLabel :
                                                                         colors.Label;
            const float top = at.y - font_size * .5f, bottom = at.y + font_size * .5f;
            if (s < inputs) draw_label({labels_min + spacing + (parameter ? theme.Sizes.ParameterIndent * zoom : 0.f), top}, {std::min(labels_min + label_reach, labels_max) - spacing, bottom}, LabelOf(view, n, s), color, false);
            else draw_label({std::max(labels_max - label_reach, labels_min) + spacing, top}, {labels_max - spacing, bottom}, LabelOf(view, n, s), color, true);
        }
        if (rows.Tray > 0u && !expanded) {
            const auto at = TrayAt(graph, n);
            const auto label = TrayLinks(view, n);
            const bool linked = !label.empty();
            const bool dimmed = wiring && LinkingSockets(r, graph, panel.DragNode, panel.DragSocket, n, true).empty();
            const bool hovered = hit.Type == HitType::Parameters && hit.Node == n && !dimmed;
            const auto color = dimmed ? colors.Dimmed : hovered ? colors.Hovered :
                linked                                          ? colors.ChevronLinked :
                                                                  colors.Chevron;
            const float chevron = theme.Sizes.SocketRadius * zoom;
            const ImVec2 a{at.x - chevron * .7f, at.y - chevron}, b{at.x - chevron * .7f, at.y + chevron}, c{at.x + chevron * .9f, at.y};
            draw.AddTriangleFilled(a, c, b, U32(color));
            draw.AddTriangle(a, c, b, U32(colors.Outline), theme.Sizes.OutlineWidth);
            draw_label({labels_min + spacing, at.y - font_size * .5f}, {std::min(labels_min + label_reach, labels_max) - spacing, at.y + font_size * .5f}, linked ? label.c_str() : UnlinkedTray, color, false);
        }
    };
    for (const auto n : graph.Shown) {
        if (n != graph.Expanded) draw_node(n);
    }
    if (graph.Expanded != NoNode) draw_node(graph.Expanded);
    PopFont();
    // The ports are inside the nodes, so the wires draw over them.
    // Highlight a wire when a dragged node would be inserted into it, when a drag from the hovered input would pick it up, or when it starts at the hovered output.
    const bool over_input = idle && hit.Type == HitType::Socket && hit.Socket < InputsOf(view, hit.Node);
    const bool over_output = idle && hit.Type == HitType::Socket && !over_input;
    const auto hovered_slot = over_input ? SlotOf(view, hit.Node, hit.Socket, hit.Index) : uint16_t{};
    ForEachWire(graph, [&](Wire wire, NodeLink link, uint32_t node, uint16_t slot) {
        const bool lit = (target && target->ToNode == node && target->ToSlot == slot) || (over_input && hit.Node == node && hovered_slot == slot) || (over_output && link == NodeLink{hit.Node, hit.Socket});
        const auto color = lit ? colors.WireHighlight : SocketColor(colors, shown_types(link.FromNode, link.FromSocket));
        DrawWire(draw, wire, U32(color), lit ? thickness * 1.5f : thickness);
    });
    // The port cells a held node would link at.
    for (const auto &link : dock.Links) {
        if (panel.Drag == NodeDrag::None) break;
        const ImVec2 at = SocketAt(graph, dock.Node, link.first), half{graph.Step * .5f, graph.Step * .5f};
        draw.AddRectFilled(at - half, at + half, U32(colors.BoxSelect));
        draw.AddRect(at - half, at + half, U32(colors.BoxSelectBorder));
    }
    if (wiring) {
        const auto socket = SocketAt(graph, panel.DragNode, panel.DragSocket);
        const auto held = NodeSockets(view, panel.DragNode)[panel.DragSocket];
        const bool out = held.Direction == SocketDirection::Out;
        std::vector<ImVec2> wire{out ? socket : mouse};
        CurveTo(wire, out ? drop_at.value_or(mouse) : socket, theme.Sizes.WireCurving);
        DrawWire(draw, wire, U32(SocketColor(colors, shown_types(panel.DragNode, panel.DragSocket))), thickness);
    }
    if (panel.Drag == NodeDrag::Box) {
        const auto press = Im(panel.Press);
        const auto box_min = ImMin(press, mouse), box_max = ImMax(press, mouse);
        draw.AddRectFilled(box_min, box_max, U32(colors.BoxSelect));
        draw.AddRect(box_min, box_max, U32(colors.BoxSelectBorder));
    }
    draw.PopClipRect();
    if (view.Authored) {
        if (editable) DrawAddMenu(r, panel, view, interactive);
    } else {
        BeginDisabled(!interactive);
        if (BeginPopup("##AddNode")) {
            DrawSceneAddMenu(r, view.Network);
            EndPopup();
        }
        if (BeginPopup("##NodeMenu")) {
            if (panel.ContextNode < nodes.size()) {
                if (panel.ContextSocket == UINT16_MAX) DrawSceneNodeMenu(r, panel, nodes[panel.ContextNode].Item);
                else if (panel.ContextSocket < NodeSockets(view, panel.ContextNode).size()) {
                    if (const auto *reference = std::get_if<SceneReference>(&BindingOf(view, panel.ContextNode, panel.ContextSocket))) DrawReferenceMenu(r, panel, *reference);
                }
            }
            EndPopup();
        }
        EndDisabled();
    }
    if (editable) DrawPortPicker(r, panel, graph, interactive);
}

bool ObjectChoice(const state::Registry &r, const char *label, state::Entity &value) {
    bool changed = false;
    const auto current = r.valid(value) ? value : state::Null;
    const auto objects = [&] {
        std::vector<state::Entity> choices{state::Null};
        for (const auto [e, instance] : r.view<const Instance>().each()) {
            if (DrawsMesh(r, instance.Entity)) choices.push_back(e);
        }
        return choices;
    };
    const auto name = [&](state::Entity e) { return e == state::Null ? std::string{"None"} : GetName(r, e); };
    ui::ChoiceCombo(label, current, objects, name, [&](state::Entity e) {
        value = e;
        changed = true;
    });
    return changed;
}

// The panel holds the text of the one active field, and `emit(text)` runs when an edit of it finishes.
void TextField(NodesPanel &panel, const char *label, std::string_view text, auto &&emit) {
    const auto id = GetID(label);
    if (panel.NameEditId != id) {
        panel.NameEdit = {};
        text.copy(panel.NameEdit.data(), panel.NameEdit.size() - 1u);
    }
    InputText(label, panel.NameEdit.data(), panel.NameEdit.size());
    if (IsItemActive()) panel.NameEditId = id;
    if (!IsItemDeactivated()) return;
    panel.NameEditId = 0;
    if (IsItemDeactivatedAfterEdit()) emit(std::string{panel.NameEdit.data()});
}

template<auto M, auto... Nested>
bool ReadsLink(std::span<const Socket> linked) {
    const auto member = [] {
        if constexpr (sizeof...(Nested) == 0) return NoField;
        else return uint8_t(field::IndexOf<field::detail::First<Nested...>>);
    }();
    return std::ranges::any_of(linked, [&](Socket socket) { return socket.Carries(uint8_t(field::IndexOf<M>), member); });
}

template<typename F>
concept LeafParameter = std::same_as<F, AttributeName> || std::same_as<F, state::Entity> || std::same_as<F, mat4>;

// Edits a node's parameter fields through Update actions at their byte offsets in the node component.
template<typename K, auto... Prefix>
struct ParameterEdit : ui::FieldWidgets<Prefix...> {
    state::Registry &R;
    NodesPanel &Panel;
    state::Entity Node;
    const K &Value;
    const std::vector<Socket> &Linked;
    bool Interactive;
    const char *Status;

    template<auto... More>
    ParameterEdit<K, Prefix..., More...> Sub() const { return {{}, R, Panel, Node, Value, Linked, Interactive, nullptr}; }

    template<auto... Ms>
    auto Read() const { return ui::ReadChain<Prefix..., Ms...>(Value); }

    template<auto... Ms>
    bool Run(auto widget, bool delta_capable = false) {
        using F = action::detail::last_field<Prefix..., Ms...>;
        const bool reads_link = ReadsLink<Prefix..., Ms...>(Linked);
        Status = reads_link ? "Linked" : nullptr;
        BeginDisabled(reads_link);
        const bool changed = ui::detail::RunField(R, Node, ChannelTarget{}, Read<Ms...>(), widget, /*selection=*/false, delta_capable && action::DeltaField<F>, [&](action::Target, const F &v) {
            action::Emit(action::Update<action::UpdateValue<F>>{Node, state::Key<K>(), action::detail::FieldOffset<Prefix..., Ms...>(), action::ToUpdateValue(v)}, action::Phase::Stage);
        });
        EndDisabled();
        return changed;
    }

    template<auto M>
        requires LeafParameter<field::Type<M>>
    void Leaf() {
        using F = field::Type<M>;
        const char *label = field::ChainLabel<Prefix..., M>.c_str();
        if constexpr (std::same_as<F, AttributeName>) {
            Status = nullptr;
            TextField(Panel, label, ui::ReadChain<Prefix..., M>(Value).data(), [&](std::string name) {
                if (Interactive) action::Emit(action::nodes::SetAttributeName{Node, std::move(name)});
            });
        } else if constexpr (std::same_as<F, mat4>) {
            Status = nullptr;
            const auto &m = ui::ReadChain<Prefix..., M>(Value);
            for (int row = 0; row < 4; ++row) {
                std::array values{m[0][row], m[1][row], m[2][row], m[3][row]};
                PushID(row);
                InputFloat4(row == 0 ? label : "##row", values.data(), "%.3f", ImGuiInputTextFlags_ReadOnly);
                PopID();
            }
        } else {
            Run<M>([&](state::Entity &v) { return ObjectChoice(R, label, v); });
        }
        if (IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ui::Hovered = {
                .Label = label,
                .Path = field::ChainPath<Prefix..., M>.c_str(),
                .Type = std::same_as<F, AttributeName> ? "string" : field::TypeName<F>.c_str(),
                .Help = field::ChainSpecOf<Prefix..., M>.Help,
                .Status = Status,
            };
        }
    }
};

state::Entity ViewportObject(const state::Registry &r, state::Entity mesh) {
    const auto object = r.Resources.get<const PanelsState>().Nodes.Canvas.Object;
    if (const auto *instance = r.try_get<const Instance>(object); instance && instance->Entity == mesh) return object;
    for (const auto [e, instance] : r.view<const Instance>().each()) {
        if (instance.Entity == mesh && InScene(r, e, FrameScene(r)) && !r.all_of<Hidden>(e)) return e;
    }
    return state::Null;
}

// The elements a selection holds, as counts by domain.
std::string SelectionText(const InlineSelection *selection) {
    if (!selection || selection->All) return "All elements";
    if (selection->Published) return "Published by the input";
    const std::array counts{SelectedCount(selection->Vertices), SelectedCount(selection->Edges), SelectedCount(selection->Faces)};
    if (counts == std::array<uint32_t, 3>{}) return "No elements";
    return std::format("{} vertices, {} edges, {} faces", counts[0], counts[1], counts[2]);
}

// Draws Done and Cancel while the selection task edits the selection of `edit`, or else the button that starts that edit, enabled when its object exists and `ready()` holds.
// Returns whether it drew the button.
bool DrawSelectionTask(const state::Registry &r, action::nodes::EditSelection edit, auto &&ready, bool interactive) {
    const auto *task = r.try_get<const SelectionEditing>(state::Document);
    if (task && task->Node == edit.Node && task->Port == edit.Port) {
        if (Button("Done") && interactive) action::Emit(action::nodes::FinishSelection{});
        SameLine();
        if (Button("Cancel") && interactive) action::Cancel();
        return false;
    }
    BeginDisabled(edit.Object == state::Null || task || !ready());
    if (Button("Edit selection...") && interactive) action::Emit(std::move(edit));
    EndDisabled();
    ui::Describe("Edit selection...", "Select input vertices, edges or faces over the displayed result. Enter commits one undo step; Esc cancels.");
    SameLine();
    return true;
}
// A built-in definition's node, which the library registry holds, shows its selection read-only.
void DrawInlineSelection(const state::Registry &r, const NodePage &page, uint32_t node, bool interactive) {
    const auto &view = page.View;
    if (view.Nodes[node].Kind == KindOf<Subnet>) return;
    const auto sockets = NodeSockets(view, node);
    const auto socket = std::ranges::find(sockets, true, &Socket::Selection);
    if (socket == sockets.end()) return;
    SeparatorText("Selection");
    const auto &a = AuthoredOf(r, view);
    const auto entity = view.Nodes[node].Item;
    const auto *selection = a.try_get<const InlineSelection>(entity);
    const bool linked = IsLinked(view, node, uint16_t(socket - sockets.begin()));
    if (&a != &r) return TextUnformatted(linked ? "Linked" : SelectionText(selection).c_str());
    if (linked) {
        TextDisabled("Linked");
        constexpr std::array Domains{Element::Vertex, Element::Edge, Element::Face};
        ui::ChoiceCombo("Domain", selection ? selection->Domain : NaturalDomainOf(r, entity), Domains, field::EnumLabel<Element>, [&](Element element) {
            auto changed = selection ? *selection : InlineSelection{.All = true};
            changed.Domain = element;
            if (interactive) action::Emit(action::nodes::SetInlineSelection{entity, std::make_unique<InlineSelection>(std::move(changed))});
        });
        ui::Describe("Domain", "Element domain the linked selection evaluates on, the elements the node acts on by default. A field of another domain converts to it.");
        return;
    }
    const bool every = !selection || selection->All;
    TextUnformatted(SelectionText(selection).c_str());
    const auto input_linked = [&] { return ResolveSocket(r, page.Path, InputOf(r, entity, 0u), true).Source.FromNode != state::Null; };
    if (!DrawSelectionTask(r, {entity, ViewportObject(r, page.Mesh), page.Path}, input_linked, interactive)) return;
    BeginDisabled(every);
    if (Button("All") && interactive) action::Emit(action::nodes::SetInlineSelection{entity, std::make_unique<InlineSelection>(InlineSelection{.Domain = selection->Domain, .All = true})});
    EndDisabled();
    ui::Describe("All", "Set the node's selection to all elements");
}

// The literal of type `to` nearest a literal of type `from`.
// One component fills every component, a vector keeps its leading components, a matrix converts through its diagonal, and two matrices share their overlap.
std::vector<uint32_t> ConvertLiteral(mx::Type from, std::span<const uint32_t> words, mx::Type to) {
    const auto columns = [](mx::Type type) {
        if (type == mx::Type::Matrix33) return 3u;
        if (type == mx::Type::Matrix44) return 4u;
        return 0u;
    };
    const auto read = [&](uint32_t i) {
        if (from == mx::Type::Integer) return double(std::bit_cast<int32_t>(words[i]));
        if (from == mx::Type::Boolean) return double(words[i]);
        return double(std::bit_cast<float>(words[i]));
    };
    const auto write = [&](double v) {
        if (to == mx::Type::Boolean) return uint32_t(v != 0.);
        if (to == mx::Type::Integer) return std::bit_cast<uint32_t>(std::isfinite(v) ? int32_t(std::clamp(v, double(INT32_MIN), double(INT32_MAX))) : 0);
        return std::bit_cast<uint32_t>(float(v));
    };
    const auto n = columns(from), m = columns(to);
    std::vector<double> components;
    for (uint32_t i = 0; i < (n ? n : uint32_t(words.size())); ++i) components.push_back(read(n ? i * n + i : i));
    // A missing component is zero, except the alpha of a color and a matrix diagonal, which are one.
    const auto component = [&](uint32_t i) {
        if (components.size() == 1u) return components[0];
        if (i < components.size()) return components[i];
        return to == mx::Type::Color4 || m ? 1. : 0.;
    };
    std::vector<uint32_t> converted(mx::Words(to));
    if (!m) {
        for (uint32_t i = 0; i < converted.size(); ++i) converted[i] = write(component(i));
        return converted;
    }
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t column = 0; column < m; ++column) {
            const bool shared = row < n && column < n;
            converted[row * m + column] = write(shared ? read(row * n + column) : row == column ? component(row) :
                                                                                                  0.);
        }
    }
    return converted;
}

// Draws the widget for an op literal's type over its words and returns whether they changed.
// A color takes the color widget, a matrix edits one row per line, and the port's enum names make a combo.
bool DrawLiteral(const char *label, mx::Type type, std::span<uint32_t> words, const mx::Port &port) {
    const bool described = port.Type == type;
    if (described && !port.EnumNames.empty()) {
        const auto value = [&](size_t i) {
            std::vector<uint32_t> value(words.size());
            if (const auto *listed = i < port.EnumValues.size() ? std::get_if<std::vector<uint32_t>>(&port.EnumValues[i]) : nullptr) std::ranges::copy(*listed | std::views::take(value.size()), value.begin());
            else value[0] = uint32_t(i);
            return value;
        };
        size_t current = port.EnumNames.size();
        for (size_t i = 0; i < port.EnumNames.size(); ++i)
            if (std::ranges::equal(value(i), words)) current = i;
        bool changed = false;
        ui::ChoiceCombo(label, current, std::views::iota(size_t{0}, port.EnumNames.size()), [&](size_t i) { return i < port.EnumNames.size() ? port.EnumNames[i] : std::string{}; }, [&](size_t i) {
            std::ranges::copy(value(i), words.begin());
            changed = true;
        });
        return changed;
    }
    constexpr auto Infinity = std::numeric_limits<double>::infinity();
    const auto bound = [&](const std::optional<mx::Value> &value, double none) {
        const auto *bound_words = described && value ? std::get_if<std::vector<uint32_t>>(&*value) : nullptr;
        if (!bound_words || bound_words->empty()) return none;
        return type == mx::Type::Integer ? double(std::bit_cast<int32_t>(bound_words->front())) : double(std::bit_cast<float>(bound_words->front()));
    };
    const bool color = type == mx::Type::Color3 || type == mx::Type::Color4;
    const FieldSpec spec{
        .Min = bound(port.UiMin, -Infinity),
        .Max = bound(port.UiMax, Infinity),
        .SoftMin = bound(port.UiSoftMin, -Infinity),
        .SoftMax = bound(port.UiSoftMax, Infinity),
        .Speed = type == mx::Type::Integer ? 1.f : .01f,
        .Widget = color ? FieldWidget::Color : FieldWidget::Auto,
    };
    const auto edit = [&]<typename T> {
        T value;
        std::memcpy(&value, words.data(), sizeof(T));
        if (!ui::DrawField(label, value, spec)) return false;
        std::memcpy(words.data(), &value, sizeof(T));
        return true;
    };
    // Matrix words are in row order.
    const auto edit_matrix = [&]<typename M, uint32_t N> {
        M value;
        for (uint32_t row = 0; row < N; ++row)
            for (uint32_t column = 0; column < N; ++column) value[column][row] = std::bit_cast<float>(words[row * N + column]);
        if (!ui::DrawField(label, value, spec)) return false;
        for (uint32_t row = 0; row < N; ++row)
            for (uint32_t column = 0; column < N; ++column) words[row * N + column] = std::bit_cast<uint32_t>(float(value[column][row]));
        return true;
    };
    switch (type) {
        case mx::Type::Boolean: {
            bool on = words[0] != 0u;
            if (!ui::DrawField(label, on, spec)) return false;
            words[0] = uint32_t(on);
            return true;
        }
        case mx::Type::Integer: return edit.template operator()<int32_t>();
        case mx::Type::Float: return edit.template operator()<float>();
        case mx::Type::Vector2: return edit.template operator()<vec2>();
        case mx::Type::Vector3:
        case mx::Type::Color3: return edit.template operator()<vec3>();
        case mx::Type::Vector4:
        case mx::Type::Color4: return edit.template operator()<vec4>();
        case mx::Type::Matrix33: return edit_matrix.template operator()<mat3, 3u>();
        case mx::Type::Matrix44: return edit_matrix.template operator()<mat4, 4u>();
        default: break;
    }
    std::unreachable(); // Only numeric and matrix inputs hold literals
}

// Draws the op node's MaterialX node, its doc text, its stored output type, each input's link, literal or geometric property, and each text input, where the registry `r` holds the node.
void DrawOp(const state::Registry &r, NodesPanel &panel, const NodeView &view, uint32_t node, bool interactive) {
    using namespace action::nodes;
    const auto &lib = mx::Libraries();
    const auto entity = view.Nodes[node].Item;
    const auto &op = r.get<const Op>(entity);
    const auto &definition = *DefinitionOf(op.Node);
    const auto &entry = *FindOp(op.Node);
    const auto declaration = view.Declarations[node];
    const auto set_input = [&](std::string_view input, OpInputValue value) { return SetOpInput{entity, std::string{input}, std::make_unique<OpInputValue>(std::move(value))}; };
    if (BeginCombo("Node", entry.Title.c_str(), ImGuiComboFlags_HeightLarge)) {
        if (auto picked = DrawOpChoice(r, panel); picked && interactive) action::Emit(ChangeOp{entity, std::move(*picked)});
        if (IsKeyPressed(ImGuiKey_Escape)) CloseCurrentPopup();
        EndCombo();
    }
    ui::Describe("Node", "MaterialX node to evaluate. The inputs and outputs it shares by name keep their links and values.");
    if (const auto *doc = OpDoc(op.Node, declaration)) TextWrapped("%s", doc);
    if (definition.StoresOutput && op.Output) {
        const auto types = [&] { return definition.OutputTypes | std::views::filter([](mx::Type type) { return DataTypeOf(type).has_value(); }) | std::ranges::to<std::vector>(); };
        ui::ChoiceCombo("Output type", *op.Output, types, TypeTitle, [&](mx::Type type) {
            if (interactive) action::Emit(SetOpOutputType{entity, type});
        });
        ui::Describe("Output type", "Value type of the output, which the input types leave open");
    }
    for (uint16_t s = 0; s < definition.Inputs; ++s) {
        const auto &socket = definition.Sockets[s];
        // The selection section shows a selection input.
        if (socket.Selection) continue;
        const std::string_view name{socket.Name};
        const auto *label = entry.Labels[s].c_str();
        const auto &port = *DeclaredPort(op.Node, declaration, name, true);
        const auto literal = op.Literals.find(name);
        const auto type = literal != op.Literals.end() ? literal->second : port.Type;
        // An input that takes an image or values reads an image while it holds a texture.
        const bool image = socket.Types == TypeBit(DataType::Image) || op.Images.contains(name);
        const auto hovered = [&](const char *status) {
            if (IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ui::Hovered = {.Label = label, .Path = entry.Title.c_str(), .Type = image ? "Image" : TypeTitle(type), .Help = port.Doc.empty() ? nullptr : port.Doc.c_str(), .Status = status};
        };
        PushID(s);
        if (IsLinked(view, node, s)) {
            TextDisabled("%s", label);
            hovered("Linked");
        } else if (image) {
            const auto held = op.Images.find(name);
            const auto textures = [&] {
                std::vector<state::Entity> choices{state::Null};
                choices.append_range(SortedEntities(r.view<const MaterializedTexture>()));
                return choices;
            };
            ui::ChoiceCombo(label, held == op.Images.end() ? state::Null : held->second, textures, [&](state::Entity e) { return e == state::Null ? std::string{"None"} : ItemName(r, e); }, [&](state::Entity e) {
                if (interactive) action::Emit(set_input(name, e));
            });
            if (IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ui::Hovered = {.Label = label, .Path = entry.Title.c_str(), .Type = "Image", .Help = "Document texture the input reads while unlinked. Without one, the node gives its default."};
        } else if (literal == op.Literals.end() && port.DefaultGeomProp) {
            TextDisabled("%s: %s", label, lib.GeomProps[*port.DefaultGeomProp].Name.c_str());
            hovered("Reads this geometric property while unlinked");
        } else {
            std::vector<uint32_t> words(mx::Words(type));
            if (literal != op.Literals.end()) {
                std::ranges::copy(LiteralWords(r, entity, name, type), words.begin());
            } else if (const auto *value = std::get_if<std::vector<uint32_t>>(&port.Default); value && value->size() == words.size()) {
                words = *value;
            }
            // An input that takes several types shows its literal's type before its value.
            const auto types = DataTypes(socket.Types & (NumericTypes | MatrixTypes));
            const bool typed = !std::has_single_bit(types);
            if (typed) {
                const float width = GetFontSize() * 6.f;
                SetNextItemWidth(width);
                const bool images = socket.Types & TypeBit(DataType::Image);
                if (BeginCombo("##type", TypeTitle(type))) {
                    for (auto bits = uint32_t(types); bits != 0u; bits &= bits - 1u) {
                        const auto choice = ValueTypeOf(DataType(std::countr_zero(bits)));
                        if (Selectable(TypeTitle(choice), choice == type) && choice != type && interactive) action::Emit(set_input(name, OpLiteral{choice, ConvertLiteral(type, words, choice)}));
                    }
                    if (images) {
                        const auto textures = SortedEntities(r.view<const MaterializedTexture>());
                        if (Selectable("Image", false, textures.empty() ? ImGuiSelectableFlags_Disabled : ImGuiSelectableFlags_None) && interactive) action::Emit(set_input(name, textures.front()));
                    }
                    EndCombo();
                }
                if (IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ui::Hovered = {.Label = "Type", .Path = label, .Help = images ? "Value type of the input's literal, or Image to read a texture of the document. Choosing another value type converts the value." : "Value type of the input's literal. Choosing another type converts the value."};
                SameLine();
                PushItemWidth(std::max(1.f, CalcItemWidth() - width - GetStyle().ItemSpacing.x));
            }
            const bool changed = DrawLiteral(label, type, words, port);
            if (typed) PopItemWidth();
            hovered(nullptr);
            if (interactive && ui::detail::FieldGesture(changed, false, false)) action::Emit(set_input(name, OpLiteral{type, std::move(words)}), action::Phase::Stage);
        }
        PopID();
    }
    static const std::string Empty;
    for (uint16_t t = 0; t < definition.Texts.size(); ++t) {
        const auto [name, type] = definition.Texts[t];
        const auto &port = *DeclaredPort(op.Node, declaration, name, true);
        const auto set = op.Texts.find(name);
        const auto *held = set != op.Texts.end() ? &set->second : std::get_if<std::string>(&port.Default);
        const auto &text = held ? *held : Empty;
        const auto *label = entry.TextLabels[t].c_str();
        PushID(int(definition.Inputs + t));
        if (port.EnumNames.empty()) {
            TextField(panel, label, text, [&](std::string edited) {
                if (interactive) action::Emit(set_input(name, std::move(edited)));
            });
        } else {
            ui::ChoiceCombo(label, text, port.EnumNames, std::identity{}, [&](const std::string &choice) {
                if (interactive) action::Emit(set_input(name, choice));
            });
        }
        if (IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ui::Hovered = {.Label = label, .Path = entry.Title.c_str(), .Type = TypeTitle(type), .Help = port.Doc.empty() ? nullptr : port.Doc.c_str()};
        PopID();
    }
}

constexpr const char *BuiltinText{"Built-in definition. Make unique to edit a copy."};
// Draws a numeric port value's components and returns whether they changed.
bool DrawPortValue(const char *label, PortValue &value) {
    if (value.Type == DataType::Bool) {
        bool checked = value.Words[0] != 0u;
        if (!Checkbox(label, &checked)) return false;
        value.Words[0] = uint32_t(checked);
        return true;
    }
    return InputScalarN(label, value.Type == DataType::Int ? ImGuiDataType_U32 : ImGuiDataType_Float, value.Words.data(), int(AttributeComponents(StorageOf(value.Type))));
}
// The value an unlinked input port of a definition passes without a default: the parameter of the first socket of the definition's network that reads it, or zero.
PortValue ExposedValue(const state::Registry &a, state::Entity definition, const SubnetPort &port) {
    const SocketLink read{definition, port.Id};
    for (const auto node : NodesOf(a, definition)) {
        if (const auto *subnet = a.try_get<const Subnet>(node)) {
            for (const auto &own : subnet->Ports) {
                if (own.Source != read) continue;
                if (const auto value = ParameterDefault(a, node, own.Id)) return *value;
            }
            continue;
        }
        const auto &links = a.get<const NodeInputs>(node).Links;
        for (uint16_t slot = 0; slot < links.size(); ++slot) {
            if (links[slot] != read) continue;
            if (const auto value = ParameterDefault(a, node, SlotSocketOf(a, node, slot))) return *value;
        }
    }
    return {{}, DataType(std::countr_zero(DataTypes(port.Types & NumericTypes)))};
}
// Draws a subnet node's uses of its definition and its own state of each port: what each unlinked input port reads, and the display of each output port.
void DrawSubnetNode(state::Registry &r, NodesPanel &panel, const NodePage &page, state::Entity subnet, bool interactive) {
    using namespace action::nodes;
    const auto &node = r.get<const Subnet>(subnet);
    const auto definition = InstanceDefinition(r, subnet);
    const bool builtin = definition.Registry != &r;
    const auto uses = UseCount(UsesOf(r, panel), node.DefinitionEntity, node.Builtin);
    if (Button("Ungroup") && interactive) action::Emit(UngroupNodes{subnet});
    ui::Describe("Ungroup", "Replace this subnet node with a copy of its definition's nodes.");
    SameLine();
    BeginDisabled(!builtin && uses < 2u);
    if (Button("Make unique") && interactive) action::Emit(MakeUnique{subnet});
    EndDisabled();
    ui::Describe("Make unique", "Give this subnet node its own copy of its definition, which its edits then change alone.");
    TextWrapped("%s", builtin ? BuiltinText : UsesText(uses).c_str());
    const auto object = ViewportObject(r, page.Mesh);
    for (const auto &port : InterfaceOf(r, subnet)->Ports) {
        PushID(int(port.Id));
        const auto *own = InstancePortOf(r, subnet, port.Id);
        if (port.Direction == SocketDirection::Out) {
            if (page.Mesh != state::Null && (port.Types & ResultTypes)) {
                SeparatorText(port.Name.c_str());
                if (Button("Display output") && interactive) action::Emit(SetDisplay{page.Mesh, subnet, port.Id, page.Path});
            }
            PopID();
            continue;
        }
        SeparatorText(port.Name.c_str());
        if (own && own->Source.FromNode != state::Null) {
            TextDisabled("Linked");
        } else if (port.Selection) {
            TextUnformatted(own && own->Selection ? SelectionText(&*own->Selection).c_str() : "Each reader's own selection");
            if (DrawSelectionTask(r, {subnet, object, page.Path, port.Id}, [] { return true; }, interactive)) {
                BeginDisabled(!own || !own->Selection);
                if (Button("Clear") && interactive) action::Emit(ClearPortValue{subnet, port.Id});
                EndDisabled();
                ui::Describe("Clear", "Pass each reader's own selection in place of this node's selection.");
            }
        } else if (ReadsDefault(port)) {
            auto value = own && own->Value ? *own->Value : port.Default ? *port.Default : ExposedValue(*definition.Registry, definition.Entity, port);
            ui::Gesture(DrawPortValue("Value", value) && interactive, [&] { return SetPortValue{subnet, port.Id, value.Words}; });
            SameLine();
            BeginDisabled(!own || !own->Value);
            if (Button("Reset") && interactive) action::Emit(ClearPortValue{subnet, port.Id});
            EndDisabled();
            ui::Describe("Reset", "Read the definition's default in place of this node's own value.");
            TextDisabled("%s", own && own->Value ? "Own value" : port.Default ? "Definition default" : "Value of the exposed socket");
        }
        PopID();
    }
}

// Draws the fault the page's occurrence of the node holds, if any, where the registry `a` holds the node.
void DrawFault(const state::Registry &r, const state::Registry &a, const NodePage &page, state::Entity node) {
    const auto *error = FaultOf(r, page, node);
    if (!error) return;
    const auto &colors = r.get<const NodeTheme>(state::Document).Colors;
    const auto color = *error == NodeFault::NotFound ? colors.Warning : colors.Error;
    PushStyleColor(ImGuiCol_Text, {color.x, color.y, color.z, color.w});
    TextWrapped("%s", FaultText(a, node, *error).c_str());
    PopStyleColor();
}
// Draws the image the page's mesh holds for the occurrence of the Bake node or filter, filling the panel's width, or else the status.
void DrawNodeImage(state::Registry &r, NodesPanel &panel, const NodePage &page, state::Entity node, const char *status) {
    SeparatorText("Image");
    auto path = page.Path;
    path.push_back(node);
    const auto image = page.Mesh == state::Null ? std::nullopt : OccurrenceImageOf(r, page.Mesh, path);
    if (!image) {
        TextWrapped("%s", status);
        return;
    }
    HoldShownImage(r, panel.NodeImage, image->Hold);
    const float width = GetContentRegionAvail().x;
    Image(ImTextureRef{mtl::ImGuiTextureId(image->Shown)}, {width, width * float(image->Height) / float(image->Width)});
}
// Draws the node's parameters, op inputs and inline selection, and a built-in definition's node shows them read-only.
void DrawParameters(state::Registry &r, NodesPanel &panel, const NodePage &page, uint32_t node, bool interactive) {
    const auto &view = page.View;
    const auto &a = AuthoredOf(r, view);
    const auto active = view.Nodes[node].Item;
    const auto kind = view.Nodes[node].Kind;
    const bool editable = &a == &r;
    PushID(int(state::Integral(active)));
    SeparatorText(kind == KindOf<Subnet> ? InstanceName(r, active).c_str() : OperationTitle(a, active, kind));
    BeginDisabled(!editable);
    const auto linked = LinkedParameters(view, node);
    const bool displays = editable && page.Mesh != state::Null && ResultSocket(r, active);
    if (displays) {
        auto shown = page.Path;
        shown.push_back(active);
        ui::ActionField("Display", DisplayedPath(r.get<const GeometryProcedure>(page.Mesh)) == shown, [&](bool displayed) { return displayed ? action::nodes::SetDisplay{page.Mesh, active, {}, page.Path} : action::nodes::SetDisplay{page.Mesh}; }, {.Help = "Display this node's result. Turn off to display no result."}, interactive);
    }
    if (editable && (BypassInput(kind) || kind == KindOf<Subnet>)) {
        if (displays) SameLine();
        ui::ActionField("Bypass", r.all_of<NodeBypassed>(active), [&](bool bypassed) { return action::nodes::SetBypass{active, bypassed}; }, {.Help = "Pass the compatible input through without applying this node."}, interactive);
    }
    if (kind == KindOf<Subnet>) DrawSubnetNode(r, panel, page, active, interactive);
    if (editable && r.all_of<SubnetDefinition>(view.Network) && BeginMenu("Expose socket")) {
        for (uint16_t s = 0; s < NodeSockets(view, node).size(); ++s) {
            PushID(s);
            if (MenuItem(LabelOf(view, node, s)) && interactive) action::Emit(action::nodes::ExposeSocket{view.Network, active, SocketId(view, node, s), {}});
            PopID();
        }
        EndMenu();
    }
    if (editable && std::ranges::any_of(HandleSpecs, [&](const auto &spec) { return spec.Kind == kind; })) {
        const auto object = ViewportObject(r, page.Mesh);
        BeginDisabled(object == state::Null);
        if (BeginMenu("Viewport handle")) {
            for (uint8_t i = 0; i < HandleSpecs.size(); ++i) {
                const auto &spec = HandleSpecs[i];
                if (spec.Kind != kind) continue;
                const bool wired = std::ranges::any_of(linked, [&](Socket socket) { return spec.Reads(socket); });
                if (MenuItem(spec.Label, nullptr, false, !wired) && interactive) action::Emit(action::nodes::SetOperationHandle{active, object, HandleParameter(i), page.Path});
            }
            EndMenu();
        }
        EndDisabled();
        if (const auto *handle = r.try_get<const OperationHandle>(state::Document); handle && handle->Node == active && handle->Path == page.Path) {
            SameLine();
            if (Button("Stop handle") && interactive) action::Emit(action::nodes::SetOperationHandle{});
        }
    }
    VisitKind(kind, [&]<typename K> {
        if constexpr (!std::is_empty_v<K> && !std::same_as<K, Op> && !std::same_as<K, Subnet>) ui::DrawFields<K>(ParameterEdit<K>{{}, r, panel, active, a.get<const K>(active), linked, interactive && editable, nullptr});
    });
    if (kind == KindOf<Op>) DrawOp(a, panel, view, node, interactive && editable);
    if (editable && kind == KindOf<Bake>) DrawNodeImage(r, panel, page, active, "The image appears once the geometry and value bake.");
    if (editable && kind == KindOf<Op> && DefinitionOf(r.get<const Op>(active).Node)->Filter && page.Mesh != state::Null) DrawNodeImage(r, panel, page, active, "The image appears once the node blurs an image, or a field that varies, by a size above zero.");
    // A material's graph previews the material at its Surface node and each op's output at the op.
    if (editable && r.all_of<MaterialGraph>(page.Root) && (kind == KindOf<Surface> || kind == KindOf<Op>)) {
        auto path = page.Path;
        if (kind == KindOf<Op>) path.push_back(active);
        DrawMaterialPreview(r, panel, page.Root, path);
    }
    DrawInlineSelection(r, page, node, interactive);
    EndDisabled();
    DrawFault(r, a, page, active);
    PopID();
}
// Turns a page whose subnet nodes no longer lead to it to the page of the nearest enclosing subnet node that still does, or else to its root network's page.
// Each subnet node of the canvas path is in the network of its predecessor's user definition, and the first in the root network.
// A hierarchy instance's copy of a procedure shows as its source's page.
void SettleCanvas(const state::Registry &r, NodesPanel &panel) {
    auto &canvas = panel.Canvas;
    if (canvas.Path.empty()) {
        while (const auto *context = r.valid(canvas.Root) ? r.try_get<const HierarchyGeometry>(canvas.Root) : nullptr) canvas.Root = context->Source;
        return;
    }
    auto network = canvas.Root;
    size_t kept = 0;
    for (; kept < canvas.Path.size() && network != state::Null && r.valid(canvas.Path[kept]) && r.all_of<Subnet>(canvas.Path[kept]) && NetworkOf(r, canvas.Path[kept]) == network; ++kept) {
        network = r.get<const Subnet>(canvas.Path[kept]).DefinitionEntity;
    }
    if (kept == canvas.Path.size()) return;
    canvas.Path.resize(kept);
    canvas.Inspected = state::Null;
    panel.Drag = NodeDrag::None;
    panel.Expanded = UINT32_MAX;
}
} // namespace

void DrawMaterialPreview(state::Registry &r, NodesPanel &panel, state::Entity material, std::span<const state::Entity> path) {
    SeparatorText("Preview");
    const auto image = MaterialPreviewOf(r, material, path);
    if (!image) {
        TextWrapped("The preview appears once the node computes a value.");
        return;
    }
    HoldShownImage(r, path.empty() ? panel.Preview : panel.NodeImage, image->Hold);
    const float width = std::min(GetContentRegionAvail().x, GetFontSize() * 16.f);
    Image(ImTextureRef{mtl::ImGuiTextureId(image->Shown)}, {width, width * float(image->Height) / float(image->Width)});
}
void HoldShownImage(const state::Registry &r, NodesPanel::ShownImage &shown, std::shared_ptr<const void> hold) {
    shown.Frame = GetFrameCount();
    if (shown.Hold == hold) return;
    // The UI frame before this one may draw the held textures.
    (void)r.Resources.get<const mtl::Context>().Wait(r.Resources.get<const ViewportConsumerFence>().Value);
    shown.Hold = std::move(hold);
}

std::string UsesText(uint32_t uses) {
    if (uses == 0u) return "No subnet node uses this definition.";
    if (uses == 1u) return "One subnet node uses this definition.";
    return std::format("{} subnet nodes use this definition. Edits change every use.", uses);
}
const UseCounts &UsesOf(const state::Registry &r, NodesPanel &panel) {
    if (const auto revision = r.Resources.get<const NetworkChanges>().Revision; !panel.Uses || panel.UsesRevision != revision) {
        panel.Uses = std::make_shared<UseCounts>(CountDefinitionUses(r));
        panel.UsesRevision = revision;
    }
    return *panel.Uses;
}

void DrawDefinitionInterface(state::Registry &r, DefinitionRef definition, bool interactive) {
    using namespace action::nodes;
    const auto &a = *definition.Registry;
    const bool editable = &a == &r;
    const auto &ports = a.get<const SubnetInterface>(definition.Entity).Ports;
    SeparatorText("Interface");
    if (ports.empty()) TextWrapped("Expose a socket of a node inside to add a port.");
    BeginDisabled(!editable);
    for (size_t i = 0; i < ports.size(); ++i) {
        const auto &port = ports[i];
        PushID(int(port.Id));
        auto name = port.Name;
        ui::Gesture(InputText(port.Direction == SocketDirection::In ? "Input" : "Output", &name) && interactive && editable && !name.empty(), [&] { return RenamePort{definition.Entity, port.Id, name}; });
        // The neighbouring ports of the same direction, which a move passes.
        const auto neighbour = [&](bool before) -> const SubnetPort * {
            for (auto at = int(i) + (before ? -1 : 1); at >= 0 && at < int(ports.size()); at += before ? -1 : 1) {
                if (ports[size_t(at)].Direction == port.Direction) return &ports[size_t(at)];
            }
            return nullptr;
        };
        const auto *previous = neighbour(true), *next = neighbour(false);
        BeginDisabled(!previous);
        if (Button("Move up") && interactive) action::Emit(MovePort{definition.Entity, port.Id, previous->Id});
        EndDisabled();
        SameLine();
        BeginDisabled(!next);
        if (Button("Move down") && interactive) action::Emit(MovePort{definition.Entity, next->Id, port.Id});
        EndDisabled();
        SameLine();
        if (Button("Remove") && interactive) action::Emit(RemovePort{definition.Entity, port.Id});
        ui::Describe("Remove", "Remove the port. Each node that reads it keeps its value.");
        if (ReadsDefault(port)) {
            auto value = port.Default ? *port.Default : ExposedValue(a, definition.Entity, port);
            ui::Gesture(DrawPortValue("Default", value) && interactive && editable, [&] { return SetPortValue{definition.Entity, port.Id, value.Words}; });
            SameLine();
            BeginDisabled(!port.Default);
            if (Button("Clear") && interactive) action::Emit(ClearPortValue{definition.Entity, port.Id});
            EndDisabled();
            ui::Describe("Clear", "Read the value of the exposed socket while a subnet node leaves this port unlinked.");
            if (!port.Default) TextDisabled("Value of the exposed socket");
        }
        PopID();
    }
    EndDisabled();
}

void DrawNodesPanel(state::Registry &r, NodesPanel &panel, bool interactive) {
    if (panel.Epoch != r.Epoch) {
        panel.Epoch = r.Epoch;
        panel.UsersRevision = UINT64_MAX;
        panel.UsesRevision = UINT64_MAX;
        panel.Drag = NodeDrag::None;
        panel.LinkingAdd.reset();
    }
    SettleCanvas(r, panel);
    FollowActiveNode(r, panel);
    // A linking add links the node it staged, which is its network's active node once the add applies.
    if (interactive && panel.LinkingAdd) {
        auto link = *std::exchange(panel.LinkingAdd, std::nullopt);
        if (const auto added = ActiveNodeOf(r, link.Network); added != panel.LinkingActive) {
            (link.FromNode == state::Null ? link.FromNode : link.ToNode) = added;
            action::Emit(std::move(link), action::Phase::Stage);
        }
        action::Commit();
    }
    panel.Hovered = false;
    if (!panel.Visible) return;
    if (Begin(panel.Name, &panel.Visible)) {
        DrawNetworkNavigation(r, panel);
        auto &page = PageOf(r, panel);
        const auto &view = page.View;
        const auto &a = AuthoredOf(r, view);
        SameLine();
        BeginDisabled(&a != &r || !r.all_of<NetworkLayout>(view.Network));
        if (Button("Auto-layout") && interactive) action::Emit(action::nodes::AutoLayout{view.Network});
        EndDisabled();
        ui::Describe("Auto-layout", "Lay out every node on this page from its connections, removing the cells that nodes were placed at. Press L over the canvas to do the same.");
        // A definition's page states its uses, and offers its subnet node a copy of its own when it shares the definition or the definition is built in.
        if (view.Authored && a.all_of<SubnetDefinition>(view.Network)) {
            const auto subnet = page.Path.empty() ? state::Null : page.Path.back();
            const auto uses = &a == &r ? UseCount(UsesOf(r, panel), view.Network) : 0u;
            AlignTextToFramePadding();
            TextUnformatted(&a == &r ? UsesText(uses).c_str() : BuiltinText);
            if (subnet != state::Null && (&a != &r || uses > 1u)) {
                SameLine();
                if (Button("Make unique") && interactive) action::Emit(action::nodes::MakeUnique{subnet});
                ui::Describe("Make unique", "Give this page's subnet node its own copy of the definition, which its edits then change alone.");
            }
        }
        PushStyleVar(ImGuiStyleVar_WindowPadding, {0.f, 0.f});
        BeginChild("Canvas", {0.f, 0.f}, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoMove);
        PopStyleVar();
        DrawCanvas(r, r.Resources.get<FrameState>(), panel, r.get<const NodeTheme>(state::Document), page, interactive);
        EndChild();
    }
    End();
}

CanvasGeometry DrawnGeometry(const state::Registry &r, const NodesPanel &panel) {
    auto &page = *panel.Page;
    const auto &view = page.View;
    Graph graph{r.get<const NodeTheme>(state::Document), {0.f, 0.f}, 1.f, 1.f, NoNode, page};
    for (uint32_t slot = 0u; slot < view.Wires.size(); ++slot) {
        if (view.Wires[slot].FromNode != NoNode) graph.ShownWires.push_back(slot);
    }
    const auto rect = [](const Box &box) { return CanvasGeometry::Rect{{float(box.X), float(box.Y)}, {float(box.X + box.W), float(box.Y + box.H)}}; };
    CanvasGeometry geometry{page.Boxes | std::views::transform(rect) | std::ranges::to<std::vector>(), page.Lanes.Cells | std::views::transform(rect) | std::ranges::to<std::vector>(), {}};
    ForEachWire(graph, [&](Wire wire, NodeLink link, uint32_t node, uint16_t) {
        geometry.Wires.push_back({link.FromNode, node, wire | std::views::transform([](ImVec2 p) { return vec2{p.x, p.y}; }) | std::ranges::to<std::vector>()});
    });
    return geometry;
}

std::vector<action::nodes::Link> MakeLinksOf(const state::Registry &r, const NodesPanel &panel, bool replace) {
    return MakeLinks(*panel.Page, SelectedNodes(r, panel.Page->View), replace);
}

state::Entity DrawNodeParameters(state::Registry &r, NodesPanel &panel, bool interactive) {
    SettleCanvas(r, panel);
    const auto &page = PageOf(r, panel);
    const auto &view = page.View;
    if (!view.Authored) return state::Null;
    const auto &a = *view.Authored;
    const auto owner = view.Network;
    if (a.all_of<SubnetDefinition>(owner)) {
        PushID("Definition");
        auto name = GetName(a, owner);
        SeparatorText(name.c_str());
        if (&a == &r) ui::Gesture(InputText("Name", &name) && interactive, [&] { return action::nodes::RenameItem{owner, name}; });
        DrawDefinitionInterface(r, {&a, owner}, interactive);
        PopID();
    }
    const bool editable = &a == &r;
    const auto node = editable ? ActiveIndex(r, view) : NodeOf(view, panel.Canvas.Inspected);
    if (editable && node == NoNode) TextWrapped("Right click or press Shift+A to add a node.");
    // A selected boundary stands for the definition, whose interface the page's Properties already shows.
    else if (node != NoNode && view.Nodes[node].Item != owner && (editable || view.Nodes[node].Kind != KindOf<Subnet>)) DrawParameters(r, panel, page, node, interactive);
    // A built-in definition's page is reached through a subnet node.
    return editable ? owner : page.Path.back();
}
void DrawEditor(auto &e, std::type_identity<CellSize>) {
    BeginDisabled(e.template Read<&CellSize::FollowFrameHeight>());
    e.template Draw<&CellSize::Points>();
    EndDisabled();
    SameLine();
    e.template Draw<&CellSize::FollowFrameHeight>();
}
void DrawNodeStyle(state::Registry &r) {
    if (Button("Reset")) action::Emit(action::view::ResetNodeTheme{});
    ui::DrawFields<NodeTheme>(ui::Edit{r, state::Document});
}
} // namespace procedure
