#pragma once
#include "mesh/Mesh.h"

vec3 VertexNormal(const Mesh &, uint32_t);
vec3 TiltAlongNormal(vec3, vec2);
vec3 ExciteDirection(const state::Registry &, state::Entity, uint32_t);
