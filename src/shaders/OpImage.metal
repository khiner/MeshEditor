#include <metal_stdlib>
using namespace metal;

// An image a program samples, and no texture for none.
struct OpImageTexture {
    texture2d<float> Texture;
};
// Whether the image table holds a texture at the slot.
inline bool OpImagePresent(device const OpImageTexture *images, uint slot) { return reinterpret_cast<device const ulong *>(images)[slot] != 0ul; }

// The image an image node samples, null for none, with the sampler its inputs choose, and whether the node has an image.
// The image stays in its table, which keeps sources that sample fast to compile.
struct OpImageSampler {
    device const OpImageTexture *Image;
    sampler Sampler;
    bool Present;
};
inline float4 texture(OpImageSampler image, float2 uv) { return image.Image->Texture.sample(image.Sampler, uv); }
inline float4 textureGrad(OpImageSampler image, float2 uv, float2 dx, float2 dy) { return image.Image->Texture.sample(image.Sampler, uv, gradient2d(dx, dy)); }
