#pragma once

#include "gpu/OpPoint.h"

#ifdef __METAL_VERSION__
#include "gpu/SceneViewUBO.h"
#else
#include <bit>
#endif

// The gltf_pbr inputs a material's node graph computes per surface point, in the order of the Surface node's sockets.
// Each is a bit of a pattern's input mask, and a pattern's words hold its computed inputs in this order.
enum class PbrInput : uint32_t {
    BaseColor,
    Metallic,
    Roughness,
    Normal, // The shading normal in the tangent frame, as a normal texture's texel maps to [-1, 1]
    Occlusion, // The factor on the light beneath the coat
    Specular,
    SpecularColor,
    Iridescence,
    IridescenceThickness, // In nanometers
    SheenColor,
    SheenRoughness,
    Clearcoat,
    ClearcoatRoughness,
    ClearcoatNormal, // In the tangent frame, as Normal
    Emissive,
    Thickness, // In the mesh's space
    AnisotropyStrength,
    AnisotropyRotation, // In radians, counterclockwise from the tangent
    Count,
};
// The inputs of three words, and every other input has one.
GPU_CONSTANT uint32_t PbrInputTriples = (1u << uint32_t(PbrInput::BaseColor)) | (1u << uint32_t(PbrInput::Normal)) | (1u << uint32_t(PbrInput::SpecularColor)) |
    (1u << uint32_t(PbrInput::SheenColor)) | (1u << uint32_t(PbrInput::ClearcoatNormal)) | (1u << uint32_t(PbrInput::Emissive));

// A material's computed inputs: the call of the program that computes them at a surface point, and the inputs it writes.
// The program's index is InvalidOffset for a material without one.
// Its attribute table is the directory of the attribute tables of a program that reads named attributes, which holds the count of mesh records, then each record's table word, and has no slot for a program that reads none.
struct MaterialPattern {
    OpProgramRef Program DEFAULT();
    uint32_t Inputs DEFAULT(); // A bit per PbrInput
    uint32_t Words DEFAULT();
};

// The pattern word buffer holds the words of the computed inputs at each surface point a pattern covers.
// Its first words are a header, then each pixel's words at the stride of the frame's widest pattern, then the words of the ray samples the hit driver evaluates.
// With patterned transparent surfaces, each pixel's list of transparent fragments follows: the pixels' first entries, each entry's key and next entry, and each entry's words.
// The header's first word counts the listed fragments.
GPU_CONSTANT uint32_t PatternWordsHeader = 4u;
// No fragment follows a list's last entry.
GPU_CONSTANT uint32_t PatternListEnd = ~0u;

// The first word of a computed input among the words of a pattern computing `inputs`.
inline uint32_t PbrInputWord(uint32_t inputs, PbrInput input) {
    const uint32_t before = inputs & ((1u << uint32_t(input)) - 1u);
#ifdef __METAL_VERSION__
    return popcount(before) + 2u * popcount(before & PbrInputTriples);
#else
    return uint32_t(std::popcount(before) + 2 * std::popcount(before & PbrInputTriples));
#endif
}

#ifdef __METAL_VERSION__
// The first pattern word of the pixel.
inline uint PatternPixelWord(constant SceneViewUBO &view, uint2 pixel) { return PatternWordsHeader + (pixel.y * view.PatternWidth + pixel.x) * view.PatternStride; }
// The first pattern word of the listed ray sample.
inline uint PatternRingWord(constant SceneViewUBO &view, uint listed) { return view.PatternRing + listed * view.PatternStride; }
#endif
