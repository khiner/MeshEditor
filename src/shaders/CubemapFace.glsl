// OpenGL/Vulkan Y-up cubemap face convention (layer order: +X, -X, +Y, -Y, +Z, -Z).
// uv is in [-1, 1] for the face's s/t axes.
vec3 FaceDirection(uint face, vec2 uv) {
    switch (face) {
        case 0: return normalize(vec3(1.0, -uv.y, -uv.x));
        case 1: return normalize(vec3(-1.0, -uv.y, uv.x));
        case 2: return normalize(vec3(uv.x, 1.0, uv.y));
        case 3: return normalize(vec3(uv.x, -1.0, -uv.y));
        case 4: return normalize(vec3(uv.x, -uv.y, 1.0));
        default: return normalize(vec3(-uv.x, -uv.y, -1.0));
    }
}
