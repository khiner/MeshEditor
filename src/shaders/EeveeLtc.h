/* SPDX-FileCopyrightText: 2017-2026 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Metal port of Blender 222853b0d7e eevee_ltc_lib.bsl.hh and eevee_ltc_lut_lib.bsl.hh.
 * LTC: Heitz, Dupuy, Hill and Neubelt, SIGGRAPH 2016.
 * Upstream equations are verified by script/ImportEeveeLut --verify.
 */
#pragma once
#include <metal_stdlib>
using namespace metal;
namespace eevee_ltc {
constexpr sampler linear_sampler(coord::normalized, address::clamp_to_edge, filter::linear);
inline float square(float x) { return x * x; }
inline float safe_divide(float a, float b) { return b == 0.0f ? 0.0f : a / b; }
inline float3 normalize_fallback(float3 v, float3 fallback) { return dot(v, v) > 1e-35f ? normalize(v) : fallback; }
enum class LTCFormFactorType { OneSidedCosineSphereClipped, TwoSidedCosineSphere };
struct LightShape {
    float3 v[4];
    float3 N;
    float disk_radius_projected;
};
struct LTCData {
    float3x3 Minv;
    float attenuation_factor;
    LTCFormFactorType form_factor_type;
};
// Blender gpu_shader_math_matrix_construct.bsl.hh.
float3x3 from_up_axis(float3 up)
{
  /* Duff, Tom, et al. "Building an orthonormal basis, revisited." JCGT 6.1 (2017). */
  float z_sign = up.z >= 0.0f ? 1.0f : -1.0f;
  float a = -1.0f / (z_sign + up.z);
  float b = up.x * up.y * a;

  float3x3 basis;
  basis[0] = float3(1.0f + z_sign * square(up.x) * a, z_sign * b, -z_sign * up.x);
  basis[1] = float3(b, z_sign + square(up.y) * a, -up.y);
  basis[2] = up;
  return basis;
}

/**
 * Create a rotation matrix from an up axis, and an incident vector I.
 * The other axes are chosen to always be orthogonal. The resulting matrix is a basis matrix.
 * \note `forward` and `up` must be normalized.
 * \note This can be used to create a tangent basis from a normal vector and a light/view vector.
 */
float3x3 from_incident_vector(float3 up, float3 forward)
{
  float cos_theta = dot(up, forward);
  if (cos_theta > 0.999999f) {
    /* Mostly for orthographic view and surfel light eval. */
    return from_up_axis(up);
  }

  /* Construct orthonormal basis around N. */
  float3x3 basis;
  basis[0] = normalize(forward - up * cos_theta);
  basis[1] = cross(up, basis[0]);
  basis[2] = up;
  return basis;
}
inline LTCData diffuse(float3 N, float3 V) {
    return {transpose(from_incident_vector(N, V)), 1.0f, LTCFormFactorType::OneSidedCosineSphereClipped};
}
inline LTCData ggx(texture2d_array<float> table, float3 N, float3 V, float roughness) {
    const float2 uv = float2(roughness, sqrt(saturate(1.0f - dot(N, V)))) * (63.0f / 64.0f) + (0.5f / 64.0f);
    const float4 p = table.sample(linear_sampler, uv, 18u, level(0));
    const float3x3 Minv(float3(p.x, 0, p.y), float3(0, 1, 0), float3(p.z, 0, p.w));
    return {Minv * transpose(from_incident_vector(N, V)), saturate((roughness - 0.15f) * 2.5f), LTCFormFactorType::OneSidedCosineSphereClipped};
}
// EEVEE eevee_sampling_lib::sample_circle/sample_disk and shadow_ray_generate_punctual.
inline float2 sample_shape(float2 noise, bool rectangle) {
    if (rectangle) return noise * 2.0f - 1.0f;
    const float phi = (noise.y - 0.5f) * (2.0f * M_PI_F);
    const float cos_phi = cos(phi);
    const float sin_phi = sqrt(1.0f - square(cos_phi)) * sign(phi);
    return float2(cos_phi, sin_phi) * sqrt(noise.x);
}
// Light::shape_radiance_get converts flux to radiance. Half extents are world-space.
inline float area(float3 X, float3 Y, bool rectangle) {
    return length(cross(X, Y)) * (rectangle ? 4.0f : M_PI_F);
}
inline float radiance(float flux, float3 X, float3 Y, bool rectangle) {
    return flux / (M_PI_F * area(X, Y, rectangle));
}
// X and Y are world-space half-extent vectors. Rectangle winding and ellipse
// bounding domain match LightShape::get; emission is toward -cross(X, Y).
inline LightShape shape(float3 center, float3 X, float3 Y, bool rectangle) {
    LightShape result;
    if (rectangle) {
        result.v[0] = center + X - Y;
        result.v[1] = center + X + Y;
        result.v[2] = center - X + Y;
        result.v[3] = center - X - Y;
    } else {
        result.v[0] = center - X - Y;
        result.v[1] = center + X - Y;
        result.v[2] = center + X + Y;
        result.v[3] = center - X + Y;
    }
    result.N = normalize(cross(X, Y));
    result.disk_radius_projected = (rectangle ? length(X + Y) : max(length(X), length(Y))) / length(center);
    return result;
}
namespace detail {

/**
 * Diffuse *clipped* sphere integral. This should equal the irradiance (form factor) of a
 * horizon-clipped sphere w.r.t. a single point. However, the form factor is divided out,
 * resulting in a multiplier defining the clipped sphere only.
 */
float diffuse_sphere_integral(texture2d_array<float> util_tx, float avg_dir_z, float form_factor)
{
  /* use tabulated horizon-clipped sphere */
  float2 uv = float2(avg_dir_z * 0.5f + 0.5f, form_factor);
  uv = uv * (63.0f / 64.0f) + (0.5f / 64.0f);

  return util_tx.sample(linear_sampler, uv, 1u, level(0)).w;
}

/**
 * An extended version of the implementation from [How to solve a cubic equation,
 * revisited](http://momentsingraphics.de/?p=105).
 */
float3 solve_cubic(float4 coefs)
{
  /* Normalize the polynomial */
  coefs.xyz /= coefs.w;
  /* Divide middle coefficients by three */
  coefs.yz /= 3.0f;

  float A = coefs.w;
  float B = coefs.z;
  float C = coefs.y;
  float D = coefs.x;

  /* Compute the Hessian and the discriminant */
  float3 delta = float3(-coefs.zy * coefs.zz + coefs.yx, dot(float2(coefs.z, -coefs.y), coefs.xy));

  /* Discriminant */
  float discr = dot(float2(4.0f * delta.x, -delta.y), delta.zy);

  /* Clamping avoid NaN output on some platform. (see #67060) */
  float sqrt_discr = sqrt(clamp(discr, 0.0f, FLT_MAX));

  float2 xlc, xsc;

  /* Algorithm A */
  {
    float C_a = delta.x;
    float D_a = -2.0f * B * delta.x + delta.y;

    /* Take the cubic root of a normalized complex number */
    float theta = atan2(sqrt_discr, -D_a) / 3.0f;

    float _2_sqrt_C_a = 2.0f * sqrt(-C_a);
    float x_1a = _2_sqrt_C_a * cos(theta);
    float x_3a = _2_sqrt_C_a * cos(theta + (2.0f / 3.0f) * M_PI_F);

    float xl;
    if ((x_1a + x_3a) > 2.0f * B) {
      xl = x_1a;
    }
    else {
      xl = x_3a;
    }

    xlc = float2(xl - B, A);
  }

  /* Algorithm D */
  {
    float C_d = delta.z;
    float D_d = -D * delta.y + 2.0f * C * delta.z;

    /* Take the cubic root of a normalized complex number */
    float theta = atan2(D * sqrt_discr, -D_d) / 3.0f;

    float _2_sqrt_C_d = 2.0f * sqrt(-C_d);
    float x_1d = _2_sqrt_C_d * cos(theta);
    float x_3d = _2_sqrt_C_d * cos(theta + (2.0f / 3.0f) * M_PI_F);

    float xs;
    if (x_1d + x_3d < 2.0f * C) {
      xs = x_1d;
    }
    else {
      xs = x_3d;
    }

    xsc = float2(-D, xs + C);
  }

  float E = xlc.y * xsc.y;
  float F = -xlc.x * xsc.y - xlc.y * xsc.x;
  float G = xlc.x * xsc.x;

  float2 xmc = float2(C * F - B * G, -B * F + C * E);

  float3 root = float3(xsc.x / xsc.y, xmc.x / xmc.y, xlc.x / xlc.y);

  if (root.x < root.y && root.x < root.z) {
    root.xyz = root.yxz;
  }
  else if (root.z < root.x && root.z < root.y) {
    root.xyz = root.xzy;
  }

  return root;
}

/**
 * Approximate edge integral as part of a polygon area integral over a cosine (hemi)sphere.

 * 1. This dates back to Lambert, see: [Geometric Derivation of the Irradiance of  Polygonal
 *    Lights](https://hal.science/hal-01458129/document).
 * 2. This is a fitting that avoids precision issues, see: [Real-Time Area Lighting: a Journey from
 *    Research to Production](https://advances.realtimerendering.com/s2016/s2016_ltc_rnd.pdf)
 */
float3 edge_integral_vec(float3 v1, float3 v2)
{
  float x = dot(v1, v2);
  float y = abs(x);

  float a = 0.8543985f + (0.4965155f + 0.0145206f * y) * y;
  float b = 3.4175940f + (4.1616724f + y) * y;
  float v = a / b;

  // For unit vectors, |cross|^2 == 1 - dot^2; the former avoids cancellation
  // for nearly antipodal corners at a large emitter's boundary.
  const float3 edge = cross(v1, v2);
  float theta_sintheta = (x > 0.0f) ? v : 0.5f * rsqrt(max(dot(edge, edge), 1e-7f)) - v;

  return edge * theta_sintheta;
}

/**
 * Upper parabolic curve of the form `x/(1+x)`, going through `y_1` at `x=1`,
 * with `scale` determining the curve shape.
 */
float upper_parabole(float x, float scale, float y_1)
{
  x = saturate(x / y_1);
  return saturate((x * scale + x) / (scale + x));
}

/**
 * Get the 3rd column of the inverse of 3x3 matrix M^{-1}, normalized. We can avoid
 * computing the reciprocal determinant as we need a unit vector.
 */
float3 dominant_bxdf_direction(float3x3 Minv)
{
  float3 adjoint_z = float3(+(Minv[1][0] * Minv[2][1] - Minv[2][0] * Minv[1][1]),
                            -(Minv[0][0] * Minv[2][1] - Minv[2][0] * Minv[0][1]),
                            +(Minv[0][0] * Minv[1][1] - Minv[1][0] * Minv[0][1]));
  return normalize(adjoint_z);
}

/**
 * Fitted function to attenuation light leakage caused by the sphere integral approximation.
 */
float form_factor_attenuation(LightShape shape, LTCData ltc_data, float3 L)
{
  /* First, find an approximate dominant BxDF direction D. Then, find a vector T on the light
   * plane, coplanar with D and L, i.e. T = cross(cross(L, D), shape.N). */
  float3 D = dominant_bxdf_direction(ltc_data.Minv);
  float3 T = normalize_fallback(D * dot(L, shape.N) - L * dot(D, shape.N), float3(0.0));

  /* Find t where D intersects the line L + t*T. */
  float TD = dot(T, D);
  float t = safe_divide(dot(D, L) * TD - dot(T, L), 1.0f - square(TD));

  if (t >= 0.0 && t < shape.disk_radius_projected) {
    /* If t lies within the disk radius on the positive side, we do not attenuate. */
    return 1.0;
  }
  /* Otherwise, restrict t to within on the disk radius. */
  t = min(abs(t), shape.disk_radius_projected);

  /* Compute L + t*T and warp it into LTC space. This disk point bounds the light shape in LTC
   * space, so we can use its z-component to safely fit an attenuation factor. */
  float attenuation = saturate(normalize(ltc_data.Minv * (L + t * T)).z);
  attenuation = upper_parabole(attenuation, 0.15f, 0.5f);
  attenuation += (1.0f - attenuation) * ltc_data.attenuation_factor;

  return attenuation;
}
}  // namespace detail

/**
 * Evaluate contribution of rectangle light.
 */
float evaluate_quad(texture2d_array<float> util_tx, LightShape shape, float3 light_direction, LTCData ltc_data)
{
  /* Transform the quad corners into LTC space, and project on to sphere. */
  float3 V[4] = {normalize(ltc_data.Minv * shape.v[0]),
                 normalize(ltc_data.Minv * shape.v[1]),
                 normalize(ltc_data.Minv * shape.v[2]),
                 normalize(ltc_data.Minv * shape.v[3])};

  /* Approximation using a sphere with the same form factor as the unclipped quad.
   * Finding a clipped sphere's form factor is easier than clipping the quad. */
  float3 avg_dir;
  avg_dir = detail::edge_integral_vec(V[0], V[1]);
  avg_dir += detail::edge_integral_vec(V[1], V[2]);
  avg_dir += detail::edge_integral_vec(V[2], V[3]);
  avg_dir += detail::edge_integral_vec(V[3], V[0]);

  float form_factor_inv = rsqrt(dot(avg_dir, avg_dir));
  float avg_dir_z = (avg_dir * form_factor_inv).z;
  float form_factor = saturate(1.0f / form_factor_inv);
  /* The form factor should always be finite. Check that the previous saturate works as filter. */
  // assert(!isnan(form_factor) && !isinf(form_factor));

  switch (ltc_data.form_factor_type) {
    case LTCFormFactorType::OneSidedCosineSphereClipped:
      /* The clipped sphere approximation causes light leakage for low roughness;
       * we attenuate with a fitted function that removes some energy below the horizon. */
      form_factor *= detail::form_factor_attenuation(shape, ltc_data, light_direction);
      form_factor *= detail::diffuse_sphere_integral(util_tx, avg_dir_z, form_factor);
      break;
    case LTCFormFactorType::TwoSidedCosineSphere:
      form_factor *= (1.0f / M_PI_F);
      break;
  }

  return form_factor;
}

/**
 * Evaluate contribution of disk light.
 *
 * disk_points are WS vectors from the shading point to the disk "bounding domain".
 */
float evaluate_disk(texture2d_array<float> util_tx, LightShape shape, float3 light_direction, LTCData ltc_data)
{
  /* Intermediate step: init ellipse. */
  float3 C = 0.5f * (shape.v[0] + shape.v[2]);
  float3 V1 = 0.5f * (shape.v[1] - shape.v[2]);
  float3 V2 = 0.5f * (shape.v[1] - shape.v[0]);

  /* Transform ellipse into LTC space. */
  C = ltc_data.Minv * C;
  V1 = ltc_data.Minv * V1;
  V2 = ltc_data.Minv * V2;

  /* Compute eigenvectors of new ellipse. */
  float d11 = dot(V1, V1);
  float d22 = dot(V2, V2);
  float d12 = dot(V1, V2);
  float a, inv_b;                      /* Eigenvalues */
  constexpr float threshold = 0.0007f; /* Can be adjusted. Fix artifacts. */
  if (abs(d12) / sqrt(d11 * d22) > threshold) {
    float tr = d11 + d22;
    float det = -d12 * d12 + d11 * d22;

    /* use sqrt matrix to solve for eigenvalues */
    det = sqrt(det);
    float u = 0.5f * sqrt(tr - 2.0f * det);
    float v = 0.5f * sqrt(tr + 2.0f * det);
    float e_max = (u + v);
    float e_min = (u - v);
    e_max *= e_max;
    e_min *= e_min;

    float3 V1_, V2_;
    if (d11 > d22) {
      V1_ = d12 * V1 + (e_max - d11) * V2;
      V2_ = d12 * V1 + (e_min - d11) * V2;
    }
    else {
      V1_ = d12 * V2 + (e_max - d22) * V1;
      V2_ = d12 * V2 + (e_min - d22) * V1;
    }

    a = 1.0f / e_max;
    inv_b = e_min;
    V1 = normalize(V1_);
    V2 = normalize(V2_);
  }
  else {
    a = 1.0f / d11;
    inv_b = d22;
    V1 *= sqrt(a);
    V2 *= rsqrt(inv_b);
  }

  /* Now find a front facing ellipse with the same solid angle. */

  float3 V3 = normalize(cross(V1, V2));
  if (dot(C, V3) < 0.0f) {
    V3 *= -1.0f;
  }

  float L = dot(V3, C);
  float inv_L = 1.0f / L;
  float x0 = dot(V1, C) * inv_L;
  float y0 = dot(V2, C) * inv_L;

  float ab = a * inv_b;
  inv_b *= square(inv_L);
  float t = 1.0f + x0 * x0;

  /* Compared to the original LTC implementation, we scale the polynomial by `b` to avoid numerical
   * issues when light size is small.
   * i.e., instead of solving `c0 * e^3 + c1 * e^2 + c2 * e + c3 = 0`,
   * we solve `c0/b^3 * (be)^3 + c1/b^2 * (be)^2 + c2/b * be + c3 = 0`. */
  float c0 = ab * inv_b;
  float c1 = ab * (t + y0 * y0) - c0 - inv_b;
  float c2 = inv_b - ab * t - (1.0f + y0 * y0);
  float c3 = 1.0f;

  float3 roots = detail::solve_cubic(float4(c0, c1, c2, c3));
  float e1 = roots.x;
  float e2 = roots.y;
  float e3 = roots.z;

  /* Scale the root back by multiplying `b`.
   * `a * x0 / (a - b * e2)` simplifies to `a/b * x0 / (a/b - e2)`,
   * `b * y0 / (b - b * e2)` simplifies to `y0 / (1.0f - e2)`. */
  float3 avg_dir = float3(ab * x0 / (ab - e2), y0 / (1.0f - e2), 1.0f);
  float3x3 rotate = float3x3(V1, V2, V3);
  avg_dir = rotate * avg_dir;
  avg_dir = normalize(avg_dir);

  /* L1, L2 are the extents of the front facing ellipse. From here, find the sphere form factor. */
  float L1 = rsqrt(-e3 / e2);
  float L2 = rsqrt(-e1 / e2);
  float form_factor = saturate(L1 * L2 * rsqrt((1.0f + L1 * L1) * (1.0f + L2 * L2)));
  /* The form factor should always be finite. Check that the previous saturate works as filter. */
  // assert(!isnan(form_factor) && !isinf(form_factor));

  switch (ltc_data.form_factor_type) {
    case LTCFormFactorType::OneSidedCosineSphereClipped:
      /* The clipped sphere approximation causes light leakage for low roughness;
       * we attenuate with a fitted function that removes some energy below the horizon. */
      form_factor *= detail::form_factor_attenuation(shape, ltc_data, light_direction);
      form_factor *= detail::diffuse_sphere_integral(util_tx, avg_dir.z, form_factor);
      break;
    case LTCFormFactorType::TwoSidedCosineSphere:
      form_factor *= (1.0f / M_PI_F);
      break;
  }

  return form_factor;
}

} // namespace eevee_ltc
