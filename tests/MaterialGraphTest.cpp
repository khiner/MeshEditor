#include "ProcedureFixture.h"
#include "action/Io.h"
#include "gpu/PbrFeature.h"
#include "materialx/Library.h"
#include "procedure/MaterialGraphs.h"
#include "render/MaterialComponents.h"
#include "render/MaterialPreview.h"
#include "viewport/FrameState.h"
#include "viewport/ViewportDisplay.h"

using boost::ut::expect;

namespace {
using namespace procedure;
using action::nodes::AddMaterialGraph, action::nodes::Link;

const auto Textures = std::filesystem::path{MESHEDITOR_SOURCE_DIR} / "external/glTF-Sample-Assets/Models/NormalTangentTest/glTF";

// A scratch folder of the name, in which the fixture begins a project.
struct Scratch {
    TestDir Dir;
    Scratch(ProcedureFixture &f, std::string_view name) : Dir{std::format("/tmp/mesheditor-scratch/material-{}", name).c_str()} { f.Begin(Dir.Path); }
};

// The material whose record was added last.
state::Entity NewestMaterial(const Fixture &f) {
    const auto materials = SortedEntities(f.R.view<const Material>());
    return *std::ranges::max_element(materials, {}, [&](state::Entity e) { return f.R.get<const Material>(e).Index; });
}
// Shades the view's materials without overlays, with anisotropic filtering off so a program's texture lookups filter as the material's do.
void MaterialView(Fixture &f) {
    f.Do(action::view::SetViewportShading{FirstViewport(f.R), ViewportShadingMode::MaterialPreview});
    f.Do(action::UpdateOf<&ViewportDisplay::ShowOverlays>(FirstViewport(f.R), false));
    f.Do(action::UpdateOf<&AnisotropicFilter::Value>(state::Document, AnisotropicFilterLevel::Off));
}
// Shades the view's materials, and adds a material that the active mesh's first primitive draws.
state::Entity ActiveMaterial(Fixture &f, uint32_t features) {
    MaterialView(f);
    if (features) f.Do(action::object::SetPbrMeshFeaturesMask{features});
    f.Do(action::object::AddMaterial{0u});
    return NewestMaterial(f);
}
// A shaded cube, or another shape at the position, whose first primitive draws a new material.
state::Entity MaterialCube(Fixture &f, uint32_t features, PrimitiveParameters shape = primitive::Cube{}, vec3 position = {}) {
    f.Add(action::object::AddMeshPrimitive{std::move(shape), std::make_unique<MeshInstanceCreateInfo>(MeshInstanceCreateInfo{.Transform = {.P = position}})});
    return ActiveMaterial(f, features);
}
template<typename T> void SetField(Fixture &f, state::Entity material, uint16_t offset, T value) { f.Do(action::object::UpdateMaterial<T>{material, offset, value}); }
// Sets the material field to the value, a float or a color.
void SetValue(Fixture &f, state::Entity material, uint16_t offset, const std::vector<float> &value) {
    if (value.size() == 1u) SetField(f, material, offset, value[0]);
    else SetField(f, material, offset, vec3{value[0], value[1], value[2]});
}
// Sets the op's input of the name to the value, a float or a color.
void SetConstant(ProcedureFixture &f, state::Entity node, std::string_view input, const std::vector<float> &value) {
    f.Do(OpInputOf(node, input, action::nodes::OpLiteral{value.size() == 1u ? mx::Type::Float : mx::Type::Color3, value | std::views::transform([](float v) { return std::bit_cast<uint32_t>(v); }) | std::ranges::to<std::vector>()}));
}
state::Entity Constant(ProcedureFixture &f, state::Entity material, const std::vector<float> &value) {
    const auto constant = f.Append(material, "constant");
    SetConstant(f, constant, "value", value);
    return constant;
}
void LinkInput(ProcedureFixture &f, state::Entity material, state::Entity from, PbrInput input) { f.Do(Link{material, from, f.Out(from), SurfaceNodeOf(f.R, material), uint16_t(input)}); }
state::Entity AddTexture(Fixture &f, std::string_view file, bool color) {
    f.Do(action::object::AddTexture{Textures / file, color});
    return FindActiveNode(f.R);
}
// An image op reading the texture's components of the type at the surface's first texture coordinates.
state::Entity Image(ProcedureFixture &f, state::Entity material, state::Entity texture, mx::Type type) {
    const auto node = f.Append(material, "image");
    f.Do(OpInputOf(node, "default", action::nodes::OpLiteral{type, std::vector<uint32_t>(mx::Words(type))}));
    f.Do(OpInputOf(node, "file", texture));
    return node;
}
// The normal a normal texture holds at the surface point, mapped to [-1, 1].
state::Entity DecodedNormal(ProcedureFixture &f, state::Entity material, state::Entity texture) {
    const auto multiply = f.Append(material, "multiply", {From("in1", Image(f, material, texture, mx::Type::Vector3)), Literal("in2", 2.f)});
    return f.Append(material, "subtract", {From("in1", multiply), Literal("in2", 1.f)});
}

constexpr auto Offset(auto member) { return uint16_t(reinterpret_cast<std::size_t>(&(static_cast<const PBRMaterial *>(nullptr)->*member))); }
template<typename E> constexpr uint16_t Nested(auto member, auto inner) { return uint16_t(Offset(member) + reinterpret_cast<std::size_t>(&(static_cast<const E *>(nullptr)->*inner))); }
// A blended material, whose surfaces shade per fragment.
void Blend(Fixture &f, state::Entity material) {
    SetField(f, material, Offset(&PBRMaterial::AlphaMode), MaterialAlphaMode::Blend);
    SetField(f, material, uint16_t(Offset(&PBRMaterial::BaseColorFactor) + 3u * sizeof(float)), .5f);
}

// A material input a constant computes, the material field whose value the material also expresses it with, and another value of the field.
struct ConstantCase {
    const char *Name;
    PbrInput Input;
    uint16_t Field;
    std::vector<float> Value, Other;
    uint32_t Features{};
    std::function<void(Fixture &, state::Entity)> Setup{};
};

std::vector<ConstantCase> ConstantCases() {
    return {
        {"base_color", PbrInput::BaseColor, Offset(&PBRMaterial::BaseColorFactor), {.2f, .5f, .8f}, {.9f, .1f, .1f}},
        {"metallic", PbrInput::Metallic, Offset(&PBRMaterial::MetallicFactor), {.7f}, {.1f}},
        {"roughness", PbrInput::Roughness, Offset(&PBRMaterial::RoughnessFactor), {.35f}, {.9f}},
        {"specular", PbrInput::Specular, Nested<Specular>(&PBRMaterial::Specular, &Specular::Factor), {.4f}, {1.f}},
        {"specular_color", PbrInput::SpecularColor, Nested<Specular>(&PBRMaterial::Specular, &Specular::ColorFactor), {.9f, .6f, .3f}, {1.f, 1.f, 1.f}},
        {"emissive", PbrInput::Emissive, Offset(&PBRMaterial::EmissiveFactor), {.3f, .1f, .05f}, {0.f, 0.f, 0.f}},
        {"blended base_color", PbrInput::BaseColor, Offset(&PBRMaterial::BaseColorFactor), {.2f, .5f, .8f}, {.9f, .1f, .1f}, 0u, Blend},
        {"blended roughness", PbrInput::Roughness, Offset(&PBRMaterial::RoughnessFactor), {.35f}, {.9f}, 0u, Blend},
        {"clearcoat", PbrInput::Clearcoat, Nested<Clearcoat>(&PBRMaterial::Clearcoat, &Clearcoat::Factor), {.6f}, {.1f}, FeatureBit(PbrFeature::Clearcoat)},
        {"clearcoat_roughness", PbrInput::ClearcoatRoughness, Nested<Clearcoat>(&PBRMaterial::Clearcoat, &Clearcoat::RoughnessFactor), {.3f}, {.8f}, FeatureBit(PbrFeature::Clearcoat), [](Fixture &f, state::Entity m) {
             SetField(f, m, Nested<Clearcoat>(&PBRMaterial::Clearcoat, &Clearcoat::Factor), 1.f);
         }},
        {"sheen_color", PbrInput::SheenColor, Nested<Sheen>(&PBRMaterial::Sheen, &Sheen::ColorFactor), {.5f, .2f, .7f}, {.1f, .1f, .1f}, FeatureBit(PbrFeature::Sheen), [](Fixture &f, state::Entity m) {
             SetField(f, m, Nested<Sheen>(&PBRMaterial::Sheen, &Sheen::RoughnessFactor), .5f);
         }},
        {"sheen_roughness", PbrInput::SheenRoughness, Nested<Sheen>(&PBRMaterial::Sheen, &Sheen::RoughnessFactor), {.4f}, {.9f}, FeatureBit(PbrFeature::Sheen), [](Fixture &f, state::Entity m) {
             SetField(f, m, Nested<Sheen>(&PBRMaterial::Sheen, &Sheen::ColorFactor), vec3{.5f, .4f, .3f});
         }},
        {"iridescence", PbrInput::Iridescence, Nested<Iridescence>(&PBRMaterial::Iridescence, &Iridescence::Factor), {.8f}, {.2f}, FeatureBit(PbrFeature::Iridescence)},
        {"iridescence_thickness", PbrInput::IridescenceThickness, Nested<Iridescence>(&PBRMaterial::Iridescence, &Iridescence::ThicknessMaximum), {250.f}, {600.f}, FeatureBit(PbrFeature::Iridescence), [](Fixture &f, state::Entity m) {
             SetField(f, m, Nested<Iridescence>(&PBRMaterial::Iridescence, &Iridescence::Factor), 1.f);
         }},
        {"anisotropy_strength", PbrInput::AnisotropyStrength, Nested<Anisotropy>(&PBRMaterial::Anisotropy, &Anisotropy::Strength), {.6f}, {.1f}, FeatureBit(PbrFeature::Anisotropy), [](Fixture &f, state::Entity m) {
             SetField(f, m, Offset(&PBRMaterial::RoughnessFactor), .4f);
         }},
        {"anisotropy_rotation", PbrInput::AnisotropyRotation, Nested<Anisotropy>(&PBRMaterial::Anisotropy, &Anisotropy::Rotation), {.8f}, {2.f}, FeatureBit(PbrFeature::Anisotropy), [](Fixture &f, state::Entity m) {
             SetField(f, m, Offset(&PBRMaterial::RoughnessFactor), .4f);
             SetField(f, m, Nested<Anisotropy>(&PBRMaterial::Anisotropy, &Anisotropy::Strength), .7f);
         }},
        {"thickness", PbrInput::Thickness, Nested<Volume>(&PBRMaterial::Volume, &Volume::ThicknessFactor), {.5f}, {2.f}, FeatureBit(PbrFeature::DiffuseTrans), [](Fixture &f, state::Entity m) {
             SetField(f, m, Nested<DiffuseTransmission>(&PBRMaterial::DiffuseTransmission, &DiffuseTransmission::Factor), .6f);
             SetField(f, m, Nested<Volume>(&PBRMaterial::Volume, &Volume::AttenuationDistance), .3f);
             SetField(f, m, Nested<Volume>(&PBRMaterial::Volume, &Volume::AttenuationColor), vec3{.3f, .6f, .9f});
         }},
    };
}
// Adds a mirror floor under the unit cube at the origin and traces the view's rays, whose reflections hit the cube.
void MirrorFloor(Fixture &f) {
    const auto floor = MaterialCube(f, 0u, primitive::Plane{.HalfExtents = {4, 4}}, {0, 0, -1.01f});
    SetField(f, floor, Offset(&PBRMaterial::MetallicFactor), 1.f);
    SetField(f, floor, Offset(&PBRMaterial::RoughnessFactor), .05f);
    f.Do(action::UpdateOf<&ViewportDisplay::RayTracing, &RayTracingSettings::Enabled>(FirstViewport(f.R), true));
}
// The view's image after eight ray-traced frames.
std::vector<std::byte> RayImage(Fixture &f) {
    f.P->Settle();
    for (int frame = 0; frame < 8; ++frame) {
        f.R.edit<Viewport>(FirstViewport(f.R)).RenderRequested = true;
        SubmitViewports(f.R);
        WaitForRender(f.R);
    }
    expect(RayTracingActive(f.R.get<const Viewport>(FirstViewport(f.R))));
    const auto &image = f.R.get<const Viewport>(FirstViewport(f.R)).Targets.Screen->FinalColorImage;
    return ReadbackImageRgba8(f.R.Resources.get<const mtl::Context>(), image, 0, 0, image.Extent);
}
// The largest difference of a channel between the view's image and its material's preview rendered with the view's camera, at the preview sphere's pixels.
// A pixel at the sphere's edge differs in coverage, so the sphere's pixels whose four neighbors it also covers compare, of which there are more than a thousand.
int PreviewDifference(Fixture &f, state::Entity material, std::span<const std::byte> image) {
    const auto &viewport = f.R.get<const Viewport>(FirstViewport(f.R));
    const auto extent = viewport.Targets.Screen->FinalColorImage.Extent;
    const auto rendered = RenderMaterialPreview(f.R, viewport.SceneViewUBO.GetSpan<SceneViewUBO>()[0], extent, f.R.get<const Material>(material).Index, {});
    const auto preview = ReadbackImageRgba8(f.R.Resources.get<const mtl::Context>(), rendered->Display, 0, 0, extent);
    const auto covered = [&](size_t x, size_t y) { return x < extent.Width && y < extent.Height && preview[(y * extent.Width + x) * 4u + 3u] == std::byte{255}; };
    int largest = 0;
    uint32_t sphere = 0;
    for (size_t p = 0; p < preview.size(); p += 4u) {
        const size_t x = (p / 4u) % extent.Width, y = (p / 4u) / extent.Width;
        if (!covered(x, y) || !covered(x - 1u, y) || !covered(x + 1u, y) || !covered(x, y - 1u) || !covered(x, y + 1u)) continue;
        ++sphere;
        for (size_t c = 0; c < 3u; ++c) largest = std::max(largest, std::abs(int(preview[p + c]) - int(image[p + c])));
    }
    expect(sphere > 1000u);
    return largest;
}
// The cube's position mapped into [0.25, 0.75] on each axis, by a multiply and an add.
state::Entity PositionColor(ProcedureFixture &f, state::Entity m) {
    const auto multiply = f.Append(m, "multiply", {From("in1", f.Append(m, "position")), Literal("in2", .25f)});
    return f.Append(m, "add", {From("in1", multiply), Literal("in2", .5f)});
}
// A cube whose procedure stores its position as PositionColor maps it as the named attribute "tint" on the domain, displaying the store.
GraphDescription TintCube(AttributeDomain domain) {
    return {primitive::Cube{}, {{"position"}, {"multiply", {Literal("in2", .25f)}}, {"add", {Literal("in2", .5f)}}, {KindOf<StoreNamedAttribute>, {Named("tint"), Set<&StoreNamedAttribute::Domain>(domain)}}}, {{1, 2, 0, "in1"}, {2, 3, 0, "in1"}, {3, 4, 1}, {0, 4, 0}}, 4};
}
// A texture input a graph's image lookup computes, the material's texture slot that expresses it, and the file of the texture.
struct TextureCase {
    const char *Name;
    PbrInput Input;
    const char *File;
    bool Color;
    uint32_t Features{};
    std::function<void(Fixture &, state::Entity material, uint32_t slot)> Express;
    std::function<state::Entity(ProcedureFixture &, state::Entity material, state::Entity texture)> Graph;
    std::function<void(Fixture &, state::Entity)> Setup{};
};
std::vector<TextureCase> TextureCases() {
    // Sets the material's texture slot at the field.
    const auto slot_at = [](uint16_t field) { return [field](Fixture &f, state::Entity m, uint32_t slot) { SetField(f, m, field, slot); }; };
    const auto express_base_color = slot_at(Nested<TextureInfo>(&PBRMaterial::BaseColorTexture, &TextureInfo::Slot));
    const auto graph_base_color = [](ProcedureFixture &f, state::Entity m, state::Entity texture) { return Image(f, m, texture, mx::Type::Color3); };
    return {
        {"base_color", PbrInput::BaseColor, "NormalTangentTest_BaseColor.png", true, 0u, express_base_color, graph_base_color},
        {"normal", PbrInput::Normal, "NormalTangentTest_Normal.png", false, 0u,
         slot_at(Nested<TextureInfo>(&PBRMaterial::NormalTexture, &TextureInfo::Slot)), DecodedNormal},
        {"clearcoat_normal", PbrInput::ClearcoatNormal, "NormalTangentTest_Normal.png", false, FeatureBit(PbrFeature::Clearcoat),
         slot_at(uint16_t(Nested<Clearcoat>(&PBRMaterial::Clearcoat, &Clearcoat::NormalTexture) + offsetof(TextureInfo, Slot))), DecodedNormal, [](Fixture &f, state::Entity m) { SetField(f, m, Nested<Clearcoat>(&PBRMaterial::Clearcoat, &Clearcoat::Factor), 1.f); }},
        {"occlusion", PbrInput::Occlusion, "NormalTangentTest_OcclusionRoughnessMetallic.png", false, 0u,
         [](Fixture &f, state::Entity m, uint32_t slot) {
             SetField(f, m, Offset(&PBRMaterial::OcclusionStrength), .5f);
             SetField(f, m, Nested<TextureInfo>(&PBRMaterial::OcclusionTexture, &TextureInfo::Slot), slot);
         },
         [](ProcedureFixture &f, state::Entity m, state::Entity texture) {
             // The occlusion is one plus its strength times the texture's red channel less one.
             const auto subtract = f.Append(m, "subtract", {From("in1", Image(f, m, texture, mx::Type::Float)), Literal("in2", 1.f)});
             return f.Append(m, "add", {From("in1", f.Append(m, "multiply", {From("in1", subtract), Literal("in2", .5f)})), Literal("in2", 1.f)});
         }},
        {"blended base_color", PbrInput::BaseColor, "NormalTangentTest_BaseColor.png", true, 0u, express_base_color, graph_base_color, Blend},
    };
}

// An edit of one side of a graph equivalence.
using Edit = std::function<void()>;
// One side's material, and its edit or an empty edit.
struct Built {
    state::Entity Material;
    Edit Change;
};
// Builds one side's scene.
using Side = std::function<Built(ProcedureFixture &)>;
// A scene rendered two ways: a reference, and a material's graph that computes the same inputs.
// The two render alike, and after their edits alike again and unlike before.
struct GraphEquivalence {
    std::string Name;
    Side Reference, Graphed;
    // Ray traces the view, with a mirror floor whose reflection rays hit the scene.
    bool Ray{};
    // Each material has a preview, and the two previews are alike too.
    bool Preview{};
    // Each image renders alike again after the shaders recompile.
    bool Recompile{};
    // Each image is within one level of its material's preview rendered with the view's camera.
    bool Lit{};
    // The two images differ by at most one level, since a blended pass samples a texture.
    // A program called through a function table samples it with explicit gradients and the blended material with automatic mip selection, which round differently.
    bool Blended{};
};

// The scene's material holding the case's value, and the material's graph computing the value while the material holds the other value.
// Without ray tracing, each side's edit sets the other value: the reference's field and the graph's constant, whose edit writes the value block the next frame's pattern pass reads.
GraphEquivalence ConstantRow(std::string name, const ConstantCase &c, std::function<state::Entity(ProcedureFixture &)> scene, bool ray) {
    return {
        .Name = std::move(name),
        .Reference = [c, scene, ray](ProcedureFixture &f) {
            const auto material = scene(f);
            SetValue(f, material, c.Field, c.Value);
            return Built{material, ray ? Edit{} : Edit{[&f, material, c] { SetValue(f, material, c.Field, c.Other); }}};
        },
        .Graphed = [c, scene, ray](ProcedureFixture &f) {
            const auto material = scene(f);
            SetValue(f, material, c.Field, c.Other);
            f.Do(AddMaterialGraph{material});
            const auto constant = Constant(f, material, c.Value);
            LinkInput(f, material, constant, c.Input);
            return Built{material, ray ? Edit{} : Edit{[&f, constant, c] { SetConstant(f, constant, "value", c.Other); }}};
        },
        .Ray = ray,
        .Preview = !ray,
    };
}
// The cube of the case's features and setup.
std::function<state::Entity(ProcedureFixture &)> CaseCube(uint32_t features, std::function<void(Fixture &, state::Entity)> setup) {
    return [features, setup](ProcedureFixture &f) {
        const auto material = MaterialCube(f, features);
        if (setup) setup(f, material);
        return material;
    };
}
// The cube holding the case's texture, and the cube's graph computing the input from the texture.
// Without ray tracing, the reference also expects the texture to change the cube's image.
GraphEquivalence TextureRow(std::string name, const TextureCase &c, bool ray) {
    const auto cube = CaseCube(c.Features, c.Setup);
    return {
        .Name = std::move(name),
        .Reference = [c, cube, ray](ProcedureFixture &f) {
            const auto material = cube(f);
            const auto plain = ray ? std::vector<std::byte>{} : f.Image();
            c.Express(f, material, f.R.get<const MaterializedTexture>(AddTexture(f, c.File, c.Color)).SamplerSlot);
            if (!ray) expect(f.Image() != plain);
            return Built{material, {}};
        },
        .Graphed = [c, cube](ProcedureFixture &f) {
            const auto material = cube(f);
            const auto texture = AddTexture(f, c.File, c.Color);
            f.Do(AddMaterialGraph{material});
            LinkInput(f, material, c.Graph(f, material, texture), c.Input);
            return Built{material, {}};
        },
        .Ray = ray,
        .Preview = !ray,
        .Blended = std::string_view{c.Name}.starts_with("blended"),
    };
}
std::vector<GraphEquivalence> GraphEquivalences() {
    std::vector<GraphEquivalence> rows;
    // A graph computing a constant renders as the material holding that constant does, also after a value edit of each.
    for (const auto &c : ConstantCases()) rows.push_back(ConstantRow(std::format("constant {}", c.Name), c, CaseCube(c.Features, c.Setup), false));
    // A blended cube filling the view, whose front and back faces put two transparent fragments at each pixel, computes the inputs of every fragment.
    rows.push_back(ConstantRow("constant double-sided blended base_color", ConstantCases().front(), [](ProcedureFixture &f) {
        const auto material = MaterialCube(f, 0u);
        Blend(f, material);
        SetField(f, material, Offset(&PBRMaterial::DoubleSided), 1u);
        f.Do(action::UpdateOf<&Transform::S>(action::OnActive{}, vec3{6.f}));
        return material; }, false));
    // A smooth sphere filling the view renders as its material's preview does.
    rows.push_back(ConstantRow("constant sphere base_color", ConstantCases().front(), [](ProcedureFixture &f) {
        const auto material = MaterialCube(f, 0u, primitive::UVSphere{});
        f.Do(action::view::ZoomViewCamera{FirstViewport(f.R), .2f});
        f.Do(action::object::ShadeSelected{ShadeMode::Smooth});
        SetField(f, material, Offset(&PBRMaterial::RoughnessFactor), .4f);
        return material; }, false));
    rows.back().Lit = true;
    // A texture input a graph's image lookup computes renders as the material holding the texture does.
    const auto textures = TextureCases();
    for (const auto &c : textures) rows.push_back(TextureRow(std::format("texture {}", c.Name), c, false));
    // A checkerboard over the cube's texture coordinates renders as the checkerboard one cell over with its colors swapped, also after a tiling edit of each.
    // The edit changes the image only where the coordinates vary over the cube.
    const auto checkerboard = [](vec3 color1, vec3 color2, vec2 offset) -> Side {
        return [=](ProcedureFixture &f) {
            const auto material = MaterialCube(f, 0u);
            f.Do(AddMaterialGraph{material});
            const auto node = f.Append(material, "checkerboard", {Literal("uvoffset", offset)});
            for (const auto &[input, color] : {std::pair{"color1", color1}, std::pair{"color2", color2}}) SetConstant(f, node, input, {color.x, color.y, color.z});
            LinkInput(f, material, node, PbrInput::BaseColor);
            return Built{material, [&f, node] { SetLiteral(f, node, "uvtiling", vec2{3.f, 2.f}); }};
        };
    };
    rows.push_back({.Name = "checkerboard base_color", .Reference = checkerboard({.9f, .6f, .2f}, {.1f, .2f, .5f}, {0.f, 0.f}), .Graphed = checkerboard({.1f, .2f, .5f}, {.9f, .6f, .2f}, {1.f, 0.f}), .Preview = true});
    // With ray tracing, a mirror floor's reflection rays hit the cube, whose material's graph computes its inputs at the hits as its pixels' pattern does.
    // Each constant input of an opaque cube and a texture lookup render as the material holding them does, and a graph material's own field holds another value.
    // A reflection ray computes the inputs of its first surface only, so a blended cube's far side reads its data inputs.
    auto ray_cases = ConstantCases();
    // Glass traces through its volume, whose thickness the graph computes.
    ray_cases.push_back({"glass thickness", PbrInput::Thickness, Nested<Volume>(&PBRMaterial::Volume, &Volume::ThicknessFactor), {.5f}, {2.f}, FeatureBit(PbrFeature::Transmission), [](Fixture &f, state::Entity m) {
                             SetField(f, m, Nested<Transmission>(&PBRMaterial::Transmission, &Transmission::Factor), 1.f);
                             SetField(f, m, Nested<Volume>(&PBRMaterial::Volume, &Volume::AttenuationDistance), .3f);
                             SetField(f, m, Nested<Volume>(&PBRMaterial::Volume, &Volume::AttenuationColor), vec3{.3f, .6f, .9f});
                         }});
    for (const auto &c : ray_cases) {
        if (!std::string_view{c.Name}.starts_with("blended")) rows.push_back(ConstantRow(std::format("ray constant {}", c.Name), c, CaseCube(c.Features, c.Setup), true));
    }
    rows.push_back(TextureRow("ray texture base_color", textures.front(), true));
    // With ray tracing, a patterned cube renders as the material holding its computed constant in the ray paths beside the deferred hits.
    // A blended transmissive plane puts one blended surface on each camera ray, whose pixels trace through their own surface.
    // A shadow ray through a transmissive surface tints by its data inputs, so the plane computes its roughness, which no shadow reads.
    rows.push_back(ConstantRow("ray blended transmission", {"roughness", PbrInput::Roughness, Offset(&PBRMaterial::RoughnessFactor), {.35f}, {.9f}}, [](ProcedureFixture &f) {
        const auto material = MaterialCube(f, FeatureBit(PbrFeature::Transmission), primitive::Plane{.HalfExtents = {1.5f, 1.5f}}, {0, 0, .5f});
        Blend(f, material);
        SetField(f, material, Nested<Transmission>(&PBRMaterial::Transmission, &Transmission::Factor), .7f);
        return material; }, true));
    // An opaque cube seen from inside a glass volume around the scene and the camera, which puts the camera inside a medium, leaves its surfaces to the rays.
    rows.push_back(ConstantRow("ray opaque layer", ConstantCases().front(), [](ProcedureFixture &f) {
        const auto material = MaterialCube(f, FeatureBit(PbrFeature::Transmission));
        const auto glass = MaterialCube(f, FeatureBit(PbrFeature::Transmission), primitive::Cube{.HalfExtents = {40, 40, 40}});
        SetField(f, glass, Nested<Transmission>(&PBRMaterial::Transmission, &Transmission::Factor), 1.f);
        SetField(f, glass, Nested<Volume>(&PBRMaterial::Volume, &Volume::ThicknessFactor), 1.f);
        SetField(f, glass, Offset(&PBRMaterial::RoughnessFactor), 0.f);
        SetField(f, glass, Offset(&PBRMaterial::DoubleSided), 1u);
        return material; }, true));
    // A material reading a named attribute a geometry procedure stores on vertices or corners renders as a material computing the attribute's value itself, at pixels, at transparent fragments and at ray hits.
    // A value edit upstream of the store reaches the material's reads.
    // The opaque vertex row also recompiles its patterned materials' programs.
    for (const auto domain : {AttributeDomain::Vertex, AttributeDomain::Corner}) {
        for (const std::string_view mode : {"opaque", "blended", "ray"}) {
            // The cube whose procedure stores the attribute, and its material's graph.
            const auto scene = [domain, mode](ProcedureFixture &f) {
                const auto built = Build(f, TintCube(domain));
                MaterialView(f);
                f.Do(action::object::AddMaterial{0u, built.Mesh});
                const auto material = NewestMaterial(f);
                if (mode == "blended") Blend(f, material);
                f.Do(AddMaterialGraph{material});
                return std::pair{built, material};
            };
            const Side computed = [scene](ProcedureFixture &f) {
                const auto material = scene(f).second;
                const auto color = PositionColor(f, material);
                LinkInput(f, material, color, PbrInput::BaseColor);
                return Built{material, [&f, color] { SetLiteral(f, color, "in2", .4f); }};
            };
            const Side read = [scene](ProcedureFixture &f) {
                const auto [built, material] = scene(f);
                LinkInput(f, material, f.Append(material, "geompropvalue", {Literal("default", vec3{}), Text("geomprop", "tint")}), PbrInput::BaseColor);
                // The procedure stores the position at another offset.
                return Built{material, [&f, built] { SetLiteral(f, built.Nodes[3], "in2", .4f); }};
            };
            rows.push_back({
                .Name = std::format("attribute {} {}", domain == AttributeDomain::Vertex ? "vertex" : "corner", mode),
                .Reference = computed,
                .Graphed = read,
                .Ray = mode == "ray",
                .Recompile = domain == AttributeDomain::Vertex && mode == "opaque",
            });
        }
    }
    return rows;
}
// The images a side renders, before and after its edit, and its material's previews with them.
struct Shown {
    std::vector<std::vector<std::byte>> Images, Previews;
};
// The largest difference of a channel between two images.
int Difference(std::span<const std::byte> a, std::span<const std::byte> b) {
    int largest = 0;
    for (size_t i = 0; i < a.size(); ++i) largest = std::max(largest, std::abs(int(a[i]) - int(b[i])));
    return largest;
}
// Each row's graphed side renders as its reference, before and after their edits, and its reference's edit changes its image.
void Equivalences(ProcedureFixture &f) {
    for (const auto &row : GraphEquivalences()) {
        boost::ut::test(std::format("material graph {}", row.Name)) = [&row, &f] {
            const auto shown = [&row, &f](std::string_view name, const Side &side) {
                const Scratch scratch{f, std::format("{}-{}", row.Name, name)};
                const auto [material, edit] = side(f);
                if (row.Ray) MirrorFloor(f);
                Shown shown;
                const auto show = [&] {
                    const auto image = row.Ray ? RayImage(f) : f.Image();
                    if (row.Recompile) {
                        f.R.Resources.get<FrameState>().RecompileShaders = true;
                        expect(f.Image() == image);
                    }
                    if (row.Lit) expect(PreviewDifference(f, material, image) <= 1);
                    const auto preview = row.Preview ? MaterialPreviewOf(f.R, material, {}) : std::nullopt;
                    expect(preview.has_value() == row.Preview);
                    if (preview) shown.Previews.push_back(ReadbackImageRgba8(f.R.Resources.get<const mtl::Context>(), std::static_pointer_cast<const PreviewImage>(preview->Hold)->Display, 0, 0, {preview->Width, preview->Height}));
                    shown.Images.push_back(image);
                };
                show();
                if (edit) {
                    edit();
                    show();
                }
                return shown;
            };
            const auto reference = shown("reference", row.Reference), graphed = shown("graphed", row.Graphed);
            const auto within = [&row](const std::vector<std::byte> &a, const std::vector<std::byte> &b) { return a.size() == b.size() && Difference(a, b) <= int(row.Blended); };
            expect(std::ranges::equal(graphed.Images, reference.Images, within));
            expect(graphed.Previews == reference.Previews);
            if (reference.Images.size() == 2u) expect(reference.Images[1] != reference.Images[0]);
        };
    }
}

// The texture of the file's name.
state::Entity TextureNamed(const Fixture &f, std::string_view file) {
    for (const auto [e, texture] : f.R.view<const MaterializedTexture>().each())
        if (texture.Params.Name == file) return e;
    return state::Null;
}
// Adds the graph of a MaterialX document to the material through node actions: an op for each node with its values, texts, images and links, then a link from each graph output to the Surface input it feeds.
// A filename reads the document texture of the file's name.
void AddDocumentGraph(ProcedureFixture &f, state::Entity material, const mx::MaterialDocument &document) {
    const auto &lib = mx::Libraries();
    f.Do(AddMaterialGraph{material});
    std::vector<state::Entity> ops;
    // The output socket of the op a document link reads.
    const auto output = [&](mx::NodeOutput from) { return f.OpSocket(ops[from.Node], lib.Declarations[document.Graph.Nodes[from.Node].Declaration].Outputs[from.Output].Name); };
    for (const auto &node : document.Graph.Nodes) {
        const auto &d = lib.Declarations[node.Declaration];
        const auto op = f.Append(material, d.Node);
        if (DefinitionOf(d.Node)->StoresOutput) f.Do(action::nodes::SetOpOutputType{op, d.Outputs.front().Type});
        for (const auto &[port, source] : node.Inputs) {
            const auto &input = d.Inputs[port];
            const auto socket = f.OpSocket(op, input.Name);
            if (const auto *from = std::get_if<mx::NodeOutput>(&source)) {
                f.Do(Link{material, ops[from->Node], output(*from), op, socket});
            } else if (input.Type == mx::Type::Filename) {
                f.Do(OpInputOf(op, input.Name, TextureNamed(f, std::filesystem::path{std::get<std::string>(std::get<mx::Value>(source))}.filename().string())));
            } else if (input.Type == mx::Type::String) {
                Text(input.Name, std::get<std::string>(std::get<mx::Value>(source)))(f, op);
            } else {
                f.Do(OpInputOf(op, input.Name, action::nodes::OpLiteral{input.Type, std::get<std::vector<uint32_t>>(std::get<mx::Value>(source))}));
            }
        }
        ops.push_back(op);
    }
    for (const auto &input : document.Surface) {
        const auto *index = std::get_if<uint32_t>(&input.Source);
        if (!index) continue;
        const auto socket = std::ranges::find(detail::SurfaceSockets, input.Name, [](const Socket &s) { return std::string_view{s.Name}; }) - std::begin(detail::SurfaceSockets);
        const auto from = std::get<mx::NodeOutput>(document.Graph.Outputs[*index]);
        f.Do(Link{material, ops[from.Node], output(from), SurfaceNodeOf(f.R, material), uint16_t(socket)});
    }
}

// The water bottle sample, scaled to the view, whose first primitive draws a new material, and a hidden copy of the bottle, whose mesh draws the material too.
// Each mesh's texture coordinates are unique to each point, which the export bakes over.
state::Entity MaterialBottles(Fixture &f, uint32_t features) {
    MaterialView(f);
    f.Do(action::io::LoadGltf{std::filesystem::path{MESHEDITOR_SOURCE_DIR} / "external/glTF-Sample-Assets/Models/WaterBottle/glTF/WaterBottle.gltf"});
    const auto mesh = *f.R.view<const EvaluatedMesh>().begin();
    const auto object = *std::ranges::find_if(f.R.view<const Instance>(), [&](state::Entity e) { return f.R.get<const Instance>(e).Entity == mesh; });
    f.Do(action::UpdateOf<&Transform::S>(object, vec3{8.f}));
    if (features) f.Do(action::object::SetPbrMeshFeaturesMask{features, mesh});
    f.Do(action::object::AddMaterial{0u, mesh});
    const auto material = NewestMaterial(f);
    f.Do(action::selection::Select{object});
    f.Do(action::object::Duplicate{});
    f.Do(action::object::SetVisible{false});
    return material;
}
// A material whose graph computes one input of each case, a layered graph of several nodes reading the position, a noise and a texture, or a read of a named attribute, on a scene whose meshes draw it.
struct ExportCase {
    std::string Name;
    uint32_t Features{};
    // Sets the material's values and adds the textures its graph reads, in both scenes that build the graph.
    std::function<void(Fixture &, state::Entity)> Values;
    // Builds the graph the harness gives the material, before the image both renders compare.
    std::function<void(ProcedureFixture &, state::Entity)> Graph;
    // The baked copies of the material the export writes: one that every mesh drawing the material shares where its inputs read nothing of a mesh, and else one for each mesh.
    uint32_t Copies{1u};
    // Builds the scene of the features and returns its material.
    std::function<state::Entity(ProcedureFixture &, uint32_t features)> Scene{MaterialBottles};
};
std::vector<ExportCase> ExportCases() {
    std::vector<ExportCase> cases;
    for (const auto &c : ConstantCases()) {
        cases.push_back({c.Name, c.Features, [c](Fixture &f, state::Entity m) {
                             if (c.Setup) c.Setup(f, m);
                             SetValue(f, m, c.Field, c.Other);
                         },
                         [c](ProcedureFixture &f, state::Entity m) { LinkInput(f, m, Constant(f, m, c.Value), c.Input); }});
    }
    const auto textures = [](Fixture &f, state::Entity) {
        AddTexture(f, "NormalTangentTest_BaseColor.png", true);
        AddTexture(f, "NormalTangentTest_Normal.png", false);
    };
    cases.push_back({"position base_color", 0u, {}, [](ProcedureFixture &f, state::Entity m) { LinkInput(f, m, PositionColor(f, m), PbrInput::BaseColor); }, 2u});
    cases.push_back({"image emissive", 0u, textures, [](ProcedureFixture &f, state::Entity m) {
                         LinkInput(f, m, Image(f, m, TextureNamed(f, "NormalTangentTest_BaseColor.png"), mx::Type::Color3), PbrInput::Emissive);
                     }});
    cases.push_back({"image normal", 0u, textures, [](ProcedureFixture &f, state::Entity m) {
                         LinkInput(f, m, DecodedNormal(f, m, TextureNamed(f, "NormalTangentTest_Normal.png")), PbrInput::Normal);
                     }});
    cases.push_back({"layered", 0u, textures, [](ProcedureFixture &f, state::Entity m) {
                         LinkInput(f, m, PositionColor(f, m), PbrInput::BaseColor);
                         LinkInput(f, m, f.Append(m, "noise3d", {Literal("amplitude", .5f)}), PbrInput::Roughness);
                         LinkInput(f, m, Image(f, m, TextureNamed(f, "NormalTangentTest_BaseColor.png"), mx::Type::Color3), PbrInput::Emissive);
                         LinkInput(f, m, DecodedNormal(f, m, TextureNamed(f, "NormalTangentTest_Normal.png")), PbrInput::Normal);
                     },
                     2u});
    // A material's bake reads a named attribute a geometry procedure stores on vertices or corners at each texel's corners, for each mesh.
    for (const auto domain : {AttributeDomain::Vertex, AttributeDomain::Corner}) {
        cases.push_back({
            .Name = std::format("attribute {}", domain == AttributeDomain::Vertex ? "vertex" : "corner"),
            .Graph = [](ProcedureFixture &f, state::Entity m) { LinkInput(f, m, f.Append(m, "geompropvalue", {Literal("default", vec3{}), Text("geomprop", "tint")}), PbrInput::BaseColor); },
            .Copies = 2u,
            .Scene = [domain](ProcedureFixture &f, uint32_t features) {
                const auto material = MaterialBottles(f, features);
                // Each bottle's procedure stores its position as PositionColor maps it as the named attribute "tint" on the domain.
                for (const auto mesh : SortedEntities(f.R.view<const EvaluatedMesh>())) {
                    const auto source = f.Node(mesh, 0u);
                    const auto add = PositionColor(f, mesh);
                    const auto store = f.Append(mesh, KindOf<StoreNamedAttribute>, {Named("tint"), Set<&StoreNamedAttribute::Domain>(domain)});
                    f.Do(Link{mesh, source, f.Out(source), store, 0u});
                    f.Do(Link{mesh, add, f.Out(add), store, 1u});
                    f.Do(action::nodes::SetDisplay{mesh, store});
                }
                return material;
            },
        });
    }
    // A mesh without texture coordinates exports the material's values, which its graph computes too.
    cases.push_back({
        .Name = "unbaked base_color",
        .Values = [](Fixture &f, state::Entity m) { SetValue(f, m, Offset(&PBRMaterial::BaseColorFactor), {.2f, .5f, .8f}); },
        .Graph = [](ProcedureFixture &f, state::Entity m) { LinkInput(f, m, Constant(f, m, {.2f, .5f, .8f}), PbrInput::BaseColor); },
        .Copies = 0u,
        .Scene = [](ProcedureFixture &f, uint32_t features) {
            f.AddSourced({.Data = primitive::CreateMesh(primitive::Cube{}).Data, .FlatShaded = true});
            return ActiveMaterial(f, features);
        },
    });
    return cases;
}

// The names of the materials no primitive of a mesh draws, other than an imported glTF default material, in name order.
std::vector<std::string> UndrawnMaterials(Fixture &f) {
    std::set<uint32_t> drawn;
    for (const auto mesh : f.R.view<const EvaluatedMesh>()) {
        for (const auto index : f.Store().Arenas().PrimitiveMaterials.Get(f.Store().Get(GetMesh(f.R, mesh).GetStoreId()).PrimitiveMaterials)) drawn.insert(index);
    }
    std::vector<std::string> names;
    for (const auto [e, material] : f.R.view<const Material>().each()) {
        const auto *meta = f.R.try_get<const gltf::MaterialSourceMeta>(e);
        if (!drawn.contains(material.Index) && !(meta && meta->ImplicitDefault)) names.push_back(material.Name);
    }
    std::ranges::sort(names);
    return names;
}
// A material's graph exported to glTF bakes its computed inputs to textures, and the imported scene renders within four levels of the material's image.
// The export writes the material's baked copies, and writes the material itself only where a mesh draws it without a bake, so every material the import holds that no primitive draws is one no primitive of the document draws.
// Its graph written as a MaterialX document reads back to the same document, and the graph the read document builds renders the same image.
void ExportRoundTrips(ProcedureFixture &f, ProcedureFixture &g, ProcedureFixture &h) {
    for (const auto &c : ExportCases()) {
        boost::ut::test(std::format("material graph export {}", c.Name)) = [&c, &f, &g, &h] {
            const Scratch scratch{f, std::format("export-{}", c.Name)};
            const auto material = c.Scene(f, c.Features);
            const auto plain = f.Image();
            if (c.Values) c.Values(f, material);
            f.Do(AddMaterialGraph{material});
            c.Graph(f, material);
            const auto image = f.Image();
            expect(image != plain);
            const auto path = scratch.Dir.Path / "export.gltf";
            action::io::Apply(f.R, action::io::SaveGltf{path, 256u});
            // The save reports each mesh without texture coordinates, which exports the material's values.
            expect(std::ranges::all_of(f.R.Resources.get<action::Errors>().Messages, [](const std::string &message) { return message.contains("has no texture coordinates"); }));
            const Scratch read_scratch{g, std::format("export-{}-gltf", c.Name)};
            MaterialView(g);
            g.Do(action::io::LoadGltf{path});
            if (c.Features) {
                for (const auto e : g.R.view<const EvaluatedMesh>()) g.Do(action::object::SetPbrMeshFeaturesMask{c.Features, e});
            }
            expect(Difference(image, g.Image()) <= 4);
            const auto copy = f.R.get<const Material>(material).Name + "_";
            uint32_t copies = 0;
            for (const auto [e, read] : g.R.view<const Material>().each()) copies += read.Name.starts_with(copy);
            expect(copies == c.Copies);
            expect(UndrawnMaterials(g) == UndrawnMaterials(f));
            const auto document = MaterialXOf(f.R, material);
            const auto text = WriteMaterialX(f.R, material);
            expect(document.has_value() && text.has_value());
            if (!document || !text) return;
            const auto read = mx::ReadMaterial(mx::Libraries(), *text);
            expect(read.has_value() && *read == *document);
            if (!read) return;
            const Scratch document_scratch{h, std::format("export-{}-mtlx", c.Name)};
            const auto read_material = c.Scene(h, c.Features);
            if (c.Values) c.Values(h, read_material);
            AddDocumentGraph(h, read_material, *read);
            expect(h.Image() == image);
        };
    }
}
} // namespace

void TestMaterialGraphs() {
    if (!Render) return;
    ProcedureFixture f, g, h;
    Equivalences(f);
    ExportRoundTrips(f, g, h);
}
