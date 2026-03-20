#version 450

#include "SceneUBO.glsl"

layout(location = 0) in vec4 InColor;
layout(location = 1) flat in int Inverted;

layout(location = 0) out vec4 OutColor;
layout(location = 1) out vec4 OutLineData; // No line data for solid fill

void main() {
    // Manual backface cull: discard back-faces (accounting for mirrored instances)
    if ((Inverted == 1) == gl_FrontFacing) discard;

    OutColor = InColor;
    OutLineData = vec4(0); // Not a line
    // X-ray: write near-far-plane depth so fills don't occlude wires (but still pass the eLess depth test against cleared 1.0).
    gl_FragDepth = (SceneViewUBO.BoneXRay != 0u) ? 0.999999 : gl_FragCoord.z;
}
