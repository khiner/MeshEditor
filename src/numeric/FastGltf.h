#pragma once

#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <fastgltf/tools.hpp>

namespace fastgltf {
template<> struct ElementTraits<vec2> : ElementTraitsBase<vec2, AccessorType::Vec2, float> {};
template<> struct ElementTraits<vec3> : ElementTraitsBase<vec3, AccessorType::Vec3, float> {};
template<> struct ElementTraits<vec4> : ElementTraitsBase<vec4, AccessorType::Vec4, float> {};
template<> struct ElementTraits<uvec2> : ElementTraitsBase<uvec2, AccessorType::Vec2, uint32_t> {};
template<> struct ElementTraits<uvec3> : ElementTraitsBase<uvec3, AccessorType::Vec3, uint32_t> {};
template<> struct ElementTraits<uvec4> : ElementTraitsBase<uvec4, AccessorType::Vec4, uint32_t> {};
template<> struct ElementTraits<mat3> : ElementTraitsBase<mat3, AccessorType::Mat3, float> {};
template<> struct ElementTraits<mat4> : ElementTraitsBase<mat4, AccessorType::Mat4, float> {};
} // namespace fastgltf
