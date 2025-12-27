#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "Bindless.glsl"

layout(location = 0) flat in uint InstanceIndex;
layout(location = 0) out vec2 DepthObjectId; // {Depth, ObjectID}

void main() {
    const uint objectId = ObjectIdBuffers[pc.ObjectIdSlot].Ids[pc.FirstInstance + InstanceIndex];
    DepthObjectId = vec2(gl_FragCoord.z, float(objectId));
}
