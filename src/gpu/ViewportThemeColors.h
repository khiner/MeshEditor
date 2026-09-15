#pragma once

#include "gpu/Types.h"

// Mirrors Blender's 3D Viewport theme settings.
struct ViewportThemeColors {
    vec4 Grid DEFAULT();
    vec4 GridLine DEFAULT();
    vec4 GridEmphasis DEFAULT();
    vec3 GridAxisX DEFAULT();
    vec3 GridAxisZ DEFAULT();
    vec3 Wire DEFAULT();
    vec3 WireEdit DEFAULT();
    vec3 ObjectActive DEFAULT();
    vec3 ObjectSelected DEFAULT();
    vec4 Light DEFAULT();
    vec3 Vertex DEFAULT();
    vec3 VertexSelected DEFAULT();
    vec3 EdgeSelectedIncidental DEFAULT();
    vec3 EdgeSelected DEFAULT();
    vec3 EdgeSharp DEFAULT();
    vec4 FaceSelectedIncidental DEFAULT();
    vec4 FaceSelected DEFAULT();
    vec4 ElementActive DEFAULT();
    vec4 ElementExcited DEFAULT();
    vec3 FaceNormal DEFAULT();
    vec3 VertexNormal DEFAULT();
    vec3 BoneSolid DEFAULT();
    vec3 BoneActive DEFAULT();
    vec3 BoneActiveUnsel DEFAULT();
    vec3 BoneSelect DEFAULT();
    vec3 BonePose DEFAULT();
    vec3 BonePoseActive DEFAULT();
    vec3 BonePoseActiveUnsel DEFAULT();
    vec3 Transform DEFAULT();
};
static_assert(sizeof(ViewportThemeColors) == 380, "ViewportThemeColors size");
