#include "numeric/VectorMath.h"
#include "numeric/vec2.h"

#include "gltf/AnimationPointers.h"

#include "CameraTypes.h"
#include "animation/Fields.h"
#include "animation/MorphWeights.h"
#include "armature/ArmatureComponents.h"
#include "render/Instance.h"
#include "render/LightComponents.h"
#include "render/MaterialTextureSlots.h"

#include <charconv>
#include <format>

namespace gltf {
namespace {
using animation::Target;
using animation::TextureTarget;

std::vector<PointerRow> BuildRows() {
    using enum PointerSpace;
    std::vector<PointerRow> rows{
        {"/nodes/{}/translation", Node, Target<&Transform::P>()},
        {"/nodes/{}/rotation", Node, Target<&Transform::R>()},
        {"/nodes/{}/scale", Node, Target<&Transform::S>()},
        {"/nodes/{}/weights", Node, {state::Key<MorphWeightRange>(), 0, 0, 0, ValueKind::Float}},
        {"/nodes/{}/weights/{}", Node, {state::Key<MorphWeightRange>(), 0, 0, 1, ValueKind::Float}},
        {"/nodes/{}/extensions/KHR_node_visibility/visible", Node, {state::Key<Visibility>(), offsetof(Visibility, Visible), 0, 1, ValueKind::Bool}},
        {"/materials/{}/pbrMetallicRoughness/baseColorFactor", Material, Target<&PBRMaterial::BaseColorFactor>()},
        {"/materials/{}/pbrMetallicRoughness/metallicFactor", Material, Target<&PBRMaterial::MetallicFactor>()},
        {"/materials/{}/pbrMetallicRoughness/roughnessFactor", Material, Target<&PBRMaterial::RoughnessFactor>()},
        {"/materials/{}/emissiveFactor", Material, Target<&PBRMaterial::EmissiveFactor>()},
        {"/materials/{}/extensions/KHR_materials_emissive_strength/emissiveStrength", Material, Target<&PBRMaterial::EmissiveStrength>()},
        {"/materials/{}/alphaCutoff", Material, Target<&PBRMaterial::AlphaCutoff>()},
        {"/materials/{}/normalTexture/scale", Material, Target<&PBRMaterial::NormalScale>()},
        {"/materials/{}/occlusionTexture/strength", Material, Target<&PBRMaterial::OcclusionStrength>()},
        {"/materials/{}/extensions/KHR_materials_ior/ior", Material, Target<&PBRMaterial::Ior>()},
        {"/materials/{}/extensions/KHR_materials_dispersion/dispersion", Material, Target<&PBRMaterial::Dispersion>()},
        {"/materials/{}/extensions/KHR_materials_transmission/transmissionFactor", Material, Target<&PBRMaterial::Transmission, &Transmission::Factor>()},
        {"/materials/{}/extensions/KHR_materials_volume/thicknessFactor", Material, Target<&PBRMaterial::Volume, &Volume::ThicknessFactor>()},
        {"/materials/{}/extensions/KHR_materials_volume/attenuationDistance", Material, Target<&PBRMaterial::Volume, &Volume::AttenuationDistance>()},
        {"/materials/{}/extensions/KHR_materials_volume/attenuationColor", Material, Target<&PBRMaterial::Volume, &Volume::AttenuationColor>()},
        {"/materials/{}/extensions/KHR_materials_clearcoat/clearcoatFactor", Material, Target<&PBRMaterial::Clearcoat, &Clearcoat::Factor>()},
        {"/materials/{}/extensions/KHR_materials_clearcoat/clearcoatRoughnessFactor", Material, Target<&PBRMaterial::Clearcoat, &Clearcoat::RoughnessFactor>()},
        {"/materials/{}/extensions/KHR_materials_clearcoat/clearcoatNormalTexture/scale", Material, Target<&PBRMaterial::Clearcoat, &Clearcoat::NormalScale>()},
        {"/materials/{}/extensions/KHR_materials_sheen/sheenColorFactor", Material, Target<&PBRMaterial::Sheen, &Sheen::ColorFactor>()},
        {"/materials/{}/extensions/KHR_materials_sheen/sheenRoughnessFactor", Material, Target<&PBRMaterial::Sheen, &Sheen::RoughnessFactor>()},
        {"/materials/{}/extensions/KHR_materials_specular/specularFactor", Material, Target<&PBRMaterial::Specular, &Specular::Factor>()},
        {"/materials/{}/extensions/KHR_materials_specular/specularColorFactor", Material, Target<&PBRMaterial::Specular, &Specular::ColorFactor>()},
        {"/materials/{}/extensions/KHR_materials_diffuse_transmission/diffuseTransmissionFactor", Material, Target<&PBRMaterial::DiffuseTransmission, &DiffuseTransmission::Factor>()},
        {"/materials/{}/extensions/KHR_materials_diffuse_transmission/diffuseTransmissionColorFactor", Material, Target<&PBRMaterial::DiffuseTransmission, &DiffuseTransmission::ColorFactor>()},
        {"/materials/{}/extensions/KHR_materials_anisotropy/anisotropyStrength", Material, Target<&PBRMaterial::Anisotropy, &Anisotropy::Strength>()},
        {"/materials/{}/extensions/KHR_materials_anisotropy/anisotropyRotation", Material, Target<&PBRMaterial::Anisotropy, &Anisotropy::Rotation>()},
        {"/materials/{}/extensions/KHR_materials_iridescence/iridescenceFactor", Material, Target<&PBRMaterial::Iridescence, &Iridescence::Factor>()},
        {"/materials/{}/extensions/KHR_materials_iridescence/iridescenceIor", Material, Target<&PBRMaterial::Iridescence, &Iridescence::Ior>()},
        {"/materials/{}/extensions/KHR_materials_iridescence/iridescenceThicknessMinimum", Material, Target<&PBRMaterial::Iridescence, &Iridescence::ThicknessMinimum>()},
        {"/materials/{}/extensions/KHR_materials_iridescence/iridescenceThicknessMaximum", Material, Target<&PBRMaterial::Iridescence, &Iridescence::ThicknessMaximum>()},
        {"/cameras/{}/perspective/aspectRatio", Camera, Target<&Perspective::AspectRatio>()},
        {"/cameras/{}/perspective/yfov", Camera, Target<&Perspective::FieldOfViewRad>()},
        {"/cameras/{}/perspective/znear", Camera, Target<&Perspective::NearClip>()},
        {"/cameras/{}/perspective/zfar", Camera, Target<&Perspective::FarClip>()},
        {"/cameras/{}/orthographic/xmag", Camera, Target<&Orthographic::Mag, &vec2::x>()},
        {"/cameras/{}/orthographic/ymag", Camera, Target<&Orthographic::Mag, &vec2::y>()},
        {"/cameras/{}/orthographic/znear", Camera, Target<&Orthographic::NearClip>()},
        {"/cameras/{}/orthographic/zfar", Camera, Target<&Orthographic::FarClip>()},
        {"/extensions/KHR_lights_punctual/lights/{}/color", Light, Target<&PunctualLight::Color>()},
        {"/extensions/KHR_lights_punctual/lights/{}/intensity", Light, Target<&PunctualLight::Intensity>()},
        {"/extensions/KHR_lights_punctual/lights/{}/range", Light, Target<&PunctualLight::Range>()},
        {"/extensions/KHR_lights_punctual/lights/{}/spot/innerConeAngle", Light, Target<&PunctualLight::InnerConeAngle>()},
        {"/extensions/KHR_lights_punctual/lights/{}/spot/outerConeAngle", Light, Target<&PunctualLight::OuterConeAngle>()},
        {"/extensions/EXT_lights_image_based/lights/{}/rotation", ImageLight, Target<&ImageLight::Rotation>()},
        {"/extensions/EXT_lights_image_based/lights/{}/intensity", ImageLight, Target<&ImageLight::Intensity>()},
    };
    for (uint8_t slot = 0; slot < MTS_Count; ++slot) {
        const auto prefix = std::format("/materials/{{}}/{}/extensions/KHR_texture_transform/", MaterialTextureSlots[slot].Pointer);
        rows.emplace_back(prefix + "offset", Material, TextureTarget(slot, &TextureInfo::UvOffset, 0));
        rows.emplace_back(prefix + "scale", Material, TextureTarget(slot, &TextureInfo::UvScale, 0));
        rows.emplace_back(prefix + "rotation", Material, TextureTarget(slot, &TextureInfo::UvRotation, 0));
    }
    return rows;
}

// Whether two targets name the same field, ignoring the material index and a weights target's offset.
// A one-weight target is the single-weight row.
bool SameField(const ChannelTarget &a, const ChannelTarget &b) {
    const auto pose = [](state::TypeKey key) { return key == state::Key<BoneDelta>() ? state::Key<PosedLocal>() : key; };
    if (pose(a.Component) != pose(b.Component)) return false;
    if (a.Component == state::Key<MorphWeightRange>()) return (a.Count == 1) == (b.Count == 1);
    return a.Offset == b.Offset && a.Count == b.Count;
}

// Reads the indices at each "{}" of `tmpl` from `pointer`, matching the literal text between them.
bool MatchTemplate(std::string_view pointer, std::string_view tmpl, std::vector<uint32_t> &indices) {
    indices.clear();
    while (true) {
        const auto brace = tmpl.find("{}");
        const auto literal = tmpl.substr(0, brace);
        if (!pointer.starts_with(literal)) return false;
        pointer.remove_prefix(literal.size());
        if (brace == std::string_view::npos) return pointer.empty();
        tmpl.remove_prefix(brace + 2);
        const auto digits = pointer.substr(0, pointer.find('/'));
        uint32_t index;
        if (const auto [end, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), index); digits.empty() || ec != std::errc{} || end != digits.data() + digits.size()) return false;
        indices.emplace_back(index);
        pointer.remove_prefix(digits.size());
    }
}
} // namespace

const std::vector<PointerRow> &PointerRows() {
    static const auto rows = BuildRows();
    return rows;
}

// The node rows come first in fastgltf's path order.
static_assert(uint8_t(fastgltf::AnimationPath::Translation) == 1 && uint8_t(fastgltf::AnimationPath::Weights) == 4);
const PointerRow &NodePathRow(fastgltf::AnimationPath path) { return PointerRows()[uint8_t(path) - 1]; }
fastgltf::AnimationPath NodePath(const PointerRow &row) { return fastgltf::AnimationPath(&row - PointerRows().data() + 1); }

std::optional<ParsedPointer> ParsePointer(std::string_view pointer) {
    std::vector<uint32_t> indices;
    for (const auto &row : PointerRows()) {
        if (!MatchTemplate(pointer, row.Template, indices)) continue;
        return ParsedPointer{&row, indices[0], indices.size() > 1 ? std::optional{indices[1]} : std::nullopt};
    }
    return {};
}

void ConvertBoneChannel(AnimationChannel &channel, const Transform &rest, bool to_delta) {
    const uint32_t n = channel.Target.Count;
    const bool cubic = channel.Interp == AnimationInterpolation::CubicSpline;
    const auto rotation = to_delta ? Conjugate(rest.R) : rest.R;
    for (size_t i = 0; i < channel.Values.size(); i += n) {
        const bool tangent = cubic && (i / n) % 3 != 1;
        float *v = channel.Values.data() + i;
        if (n == 4) {
            const quat q = rotation * LoadQuat(v);
            StoreQuat(v, tangent ? q : Normalize(q));
        } else if (channel.Target.Offset == offsetof(Transform, P)) {
            const vec3 p{v[0], v[1], v[2]};
            const vec3 out = tangent ? rotation * p : to_delta ? rotation * (p - rest.P) :
                                                                 rest.P + rotation * p;
            v[0] = out.x;
            v[1] = out.y;
            v[2] = out.z;
        } else {
            for (uint32_t c = 0; c < 3; ++c) v[c] = to_delta ? v[c] / rest.S[c] : v[c] * rest.S[c];
        }
    }
}

const PointerRow *RowOf(const ChannelTarget &target) {
    for (const auto &row : PointerRows())
        if (SameField(row.Target, target)) return &row;
    return nullptr;
}
} // namespace gltf
