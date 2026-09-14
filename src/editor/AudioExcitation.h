#pragma once
#include "mesh/Mesh.h"
#include "numeric/vec2.h"
#include "state/Entity.h"

extern vec2 ImpulseAngle;
vec3 VertexNormal(const Mesh &, uint32_t);
vec3 TiltAlongNormal(vec3, vec2);
vec3 ExciteDirection(const state::Scene &, state::Entity, uint32_t);
