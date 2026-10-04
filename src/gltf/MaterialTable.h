#pragma once

#include "SourceAssets.h"
#include "gpu/PBRMaterial.h"

#include <fastgltf/types.hpp>

#include <array>
#include <memory>
#include <string_view>
#include <tuple>

// The glTF material extensions MeshEditor keeps, each as the fields it carries in PBRMaterial and in fastgltf under their glTF keys.
// Import, export, the extension tally, and the animation pointer rows derive from this one description.
namespace gltf::detail {
// A field both representations hold, as a value, a color, or a texture.
template<typename O, typename T, typename OF, typename TF>
struct MaterialField {
    OF O::*Ours;
    TF T::*Theirs;
    std::string_view Key;
};
// A normal texture whose scale PBRMaterial keeps beside the texture.
template<typename O, typename T>
struct MaterialNormalField {
    TextureInfo O::*Texture;
    float O::*Scale;
    fastgltf::Optional<fastgltf::NormalTextureInfo> T::*Theirs;
    std::string_view Key;
};
// A distance glTF writes as infinity where PBRMaterial holds zero.
template<typename O, typename T>
struct MaterialDistanceField {
    float O::*Ours;
    fastgltf::num T::*Theirs;
    std::string_view Key;
};
// An extension block, present on a material when its source had it.
template<typename O, typename T, typename FieldTuple>
struct MaterialExtension {
    std::string_view Name;
    uint16_t Bit;
    O PBRMaterial::*Ours;
    std::unique_ptr<T> fastgltf::Material::*Theirs;
    FieldTuple Fields;
};
// An extension holding one number, written when its source had it or the value left its default.
struct MaterialScalarExtension {
    std::string_view Name, Key;
    uint16_t Bit;
    float PBRMaterial::*Ours;
    fastgltf::Optional<fastgltf::num> fastgltf::Material::*Theirs;
    float Default;
};

inline constexpr std::array MaterialScalarExtensions{
    MaterialScalarExtension{"KHR_materials_ior", "ior", MaterialSourceMeta::ExtIor, &PBRMaterial::Ior, &fastgltf::Material::ior, 1.5f},
    MaterialScalarExtension{"KHR_materials_emissive_strength", "emissiveStrength", MaterialSourceMeta::ExtEmissiveStrength, &PBRMaterial::EmissiveStrength, &fastgltf::Material::emissiveStrength, 1.f},
    MaterialScalarExtension{"KHR_materials_dispersion", "dispersion", MaterialSourceMeta::ExtDispersion, &PBRMaterial::Dispersion, &fastgltf::Material::dispersion, 0.f},
};

inline constexpr auto MaterialExtensions = std::tuple{
    MaterialExtension{"KHR_materials_sheen", MaterialSourceMeta::ExtSheen, &PBRMaterial::Sheen, &fastgltf::Material::sheen, std::tuple{
        MaterialField{&::Sheen::ColorFactor, &fastgltf::MaterialSheen::sheenColorFactor, "sheenColorFactor"},
        MaterialField{&::Sheen::RoughnessFactor, &fastgltf::MaterialSheen::sheenRoughnessFactor, "sheenRoughnessFactor"},
        MaterialField{&::Sheen::ColorTexture, &fastgltf::MaterialSheen::sheenColorTexture, "sheenColorTexture"},
        MaterialField{&::Sheen::RoughnessTexture, &fastgltf::MaterialSheen::sheenRoughnessTexture, "sheenRoughnessTexture"},
    }},
    MaterialExtension{"KHR_materials_specular", MaterialSourceMeta::ExtSpecular, &PBRMaterial::Specular, &fastgltf::Material::specular, std::tuple{
        MaterialField{&::Specular::Factor, &fastgltf::MaterialSpecular::specularFactor, "specularFactor"},
        MaterialField{&::Specular::ColorFactor, &fastgltf::MaterialSpecular::specularColorFactor, "specularColorFactor"},
        MaterialField{&::Specular::Texture, &fastgltf::MaterialSpecular::specularTexture, "specularTexture"},
        MaterialField{&::Specular::ColorTexture, &fastgltf::MaterialSpecular::specularColorTexture, "specularColorTexture"},
    }},
    MaterialExtension{"KHR_materials_transmission", MaterialSourceMeta::ExtTransmission, &PBRMaterial::Transmission, &fastgltf::Material::transmission, std::tuple{
        MaterialField{&::Transmission::Factor, &fastgltf::MaterialTransmission::transmissionFactor, "transmissionFactor"},
        MaterialField{&::Transmission::Texture, &fastgltf::MaterialTransmission::transmissionTexture, "transmissionTexture"},
    }},
    MaterialExtension{"KHR_materials_diffuse_transmission", MaterialSourceMeta::ExtDiffuseTransmission, &PBRMaterial::DiffuseTransmission, &fastgltf::Material::diffuseTransmission, std::tuple{
        MaterialField{&::DiffuseTransmission::Factor, &fastgltf::MaterialDiffuseTransmission::diffuseTransmissionFactor, "diffuseTransmissionFactor"},
        MaterialField{&::DiffuseTransmission::ColorFactor, &fastgltf::MaterialDiffuseTransmission::diffuseTransmissionColorFactor, "diffuseTransmissionColorFactor"},
        MaterialField{&::DiffuseTransmission::Texture, &fastgltf::MaterialDiffuseTransmission::diffuseTransmissionTexture, "diffuseTransmissionTexture"},
        MaterialField{&::DiffuseTransmission::ColorTexture, &fastgltf::MaterialDiffuseTransmission::diffuseTransmissionColorTexture, "diffuseTransmissionColorTexture"},
    }},
    MaterialExtension{"KHR_materials_volume", MaterialSourceMeta::ExtVolume, &PBRMaterial::Volume, &fastgltf::Material::volume, std::tuple{
        MaterialField{&::Volume::ThicknessFactor, &fastgltf::MaterialVolume::thicknessFactor, "thicknessFactor"},
        MaterialField{&::Volume::AttenuationColor, &fastgltf::MaterialVolume::attenuationColor, "attenuationColor"},
        MaterialDistanceField{&::Volume::AttenuationDistance, &fastgltf::MaterialVolume::attenuationDistance, "attenuationDistance"},
        MaterialField{&::Volume::ThicknessTexture, &fastgltf::MaterialVolume::thicknessTexture, "thicknessTexture"},
    }},
    MaterialExtension{"KHR_materials_clearcoat", MaterialSourceMeta::ExtClearcoat, &PBRMaterial::Clearcoat, &fastgltf::Material::clearcoat, std::tuple{
        MaterialField{&::Clearcoat::Factor, &fastgltf::MaterialClearcoat::clearcoatFactor, "clearcoatFactor"},
        MaterialField{&::Clearcoat::RoughnessFactor, &fastgltf::MaterialClearcoat::clearcoatRoughnessFactor, "clearcoatRoughnessFactor"},
        MaterialField{&::Clearcoat::Texture, &fastgltf::MaterialClearcoat::clearcoatTexture, "clearcoatTexture"},
        MaterialField{&::Clearcoat::RoughnessTexture, &fastgltf::MaterialClearcoat::clearcoatRoughnessTexture, "clearcoatRoughnessTexture"},
        MaterialNormalField{&::Clearcoat::NormalTexture, &::Clearcoat::NormalScale, &fastgltf::MaterialClearcoat::clearcoatNormalTexture, "clearcoatNormalTexture"},
    }},
    MaterialExtension{"KHR_materials_anisotropy", MaterialSourceMeta::ExtAnisotropy, &PBRMaterial::Anisotropy, &fastgltf::Material::anisotropy, std::tuple{
        MaterialField{&::Anisotropy::Strength, &fastgltf::MaterialAnisotropy::anisotropyStrength, "anisotropyStrength"},
        MaterialField{&::Anisotropy::Rotation, &fastgltf::MaterialAnisotropy::anisotropyRotation, "anisotropyRotation"},
        MaterialField{&::Anisotropy::Texture, &fastgltf::MaterialAnisotropy::anisotropyTexture, "anisotropyTexture"},
    }},
    MaterialExtension{"KHR_materials_iridescence", MaterialSourceMeta::ExtIridescence, &PBRMaterial::Iridescence, &fastgltf::Material::iridescence, std::tuple{
        MaterialField{&::Iridescence::Factor, &fastgltf::MaterialIridescence::iridescenceFactor, "iridescenceFactor"},
        MaterialField{&::Iridescence::Ior, &fastgltf::MaterialIridescence::iridescenceIor, "iridescenceIor"},
        MaterialField{&::Iridescence::ThicknessMinimum, &fastgltf::MaterialIridescence::iridescenceThicknessMinimum, "iridescenceThicknessMinimum"},
        MaterialField{&::Iridescence::ThicknessMaximum, &fastgltf::MaterialIridescence::iridescenceThicknessMaximum, "iridescenceThicknessMaximum"},
        MaterialField{&::Iridescence::Texture, &fastgltf::MaterialIridescence::iridescenceTexture, "iridescenceTexture"},
        MaterialField{&::Iridescence::ThicknessTexture, &fastgltf::MaterialIridescence::iridescenceThicknessTexture, "iridescenceThicknessTexture"},
    }},
};

// Calls `fn` with each extension, in the order the extension tally lists them.
template<typename Fn> void ForEachMaterialExtension(Fn &&fn) {
    std::apply([&](const auto &...extension) { (fn(extension), ...); }, MaterialExtensions);
}
// Calls `fn` with each field of `extension`.
template<typename Extension, typename Fn> void ForEachMaterialField(const Extension &extension, Fn &&fn) {
    std::apply([&](const auto &...field) { (fn(field), ...); }, extension.Fields);
}
} // namespace gltf::detail
