#version 450

layout(location = 1) flat in uint ObjectId;
layout(location = 0) out vec2 DepthObjectId; // {Depth, ObjectID}

void main() {
    DepthObjectId = vec2(gl_FragCoord.z, float(ObjectId));
}
