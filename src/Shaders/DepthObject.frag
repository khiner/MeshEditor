#version 450
#extension GL_EXT_nonuniform_qualifier : require
#include "Bindless.glsl"

layout(location = 0) flat in uint ObjectId;

layout(location = 0) out vec2 DepthObjectId; // {Depth, ObjectID}

void main() {
    DepthObjectId = vec2(gl_FragCoord.z, float(ObjectId));
}
