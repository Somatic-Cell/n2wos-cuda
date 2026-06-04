#pragma once

#include <cmath>

#ifndef N2WOS_HD_INLINE
#  if defined(__CUDACC__)
#    define N2WOS_HD_INLINE __host__ __device__ inline
#  else
#    define N2WOS_HD_INLINE inline
#  endif
#endif

namespace n2wos {

// Manufactured constant-coefficient Poisson fields for diagnostics.
// These modes solve
//
//     -Delta u = f,  u|boundary = g = u|boundary,
//
// with no screening/absorption term.  This is therefore an unscreened Poisson
// problem, not a screened Poisson problem.
//
// We deliberately choose harmonic source terms
//
//     f_i(x) = A_i exp(k_i a_i.x) cos/sin(k_i b_i.x + phi_i),   a_i dot b_i = 0,
//
// because the Green integral over a WoS ball at its center is then exactly
//
//     integral_B G(0,y) f_i(x+y) dy = r^2 / 6 * f_i(x)     in 3D.
//
// A corresponding analytic particular solution is
//
//     u_i(x) = -(a_i.x) / (2 k_i) * f_i(x),
//
// since Delta[(a.x) f_i] = 2 k_i f_i when f_i is harmonic and
// directional derivative along a_i equals k_i f_i.
//
// These functions must be evaluated in the same solver-normalized coordinate
// system as the existing boundary modes.

static constexpr int kNcPoissonMultiscaleMode = 30;
static constexpr int kNcPoissonFiglikeHfMode  = 31;

struct NcPoissonTermEval {
    float u;
    float f;
};

template <typename P>
N2WOS_HD_INLINE NcPoissonTermEval nc_poisson_term_exp_cos(
    P p,
    float amp,
    float k,
    float ax, float ay, float az,
    float bx, float by, float bz,
    float phi) {
    const float adot = ax * p.x + ay * p.y + az * p.z;
    const float bdot = bx * p.x + by * p.y + bz * p.z;
    const float f = amp * expf(k * adot) * cosf(k * bdot + phi);
    const float u = -(adot / (2.0f * k)) * f;
    return {u, f};
}

template <typename P>
N2WOS_HD_INLINE NcPoissonTermEval nc_poisson_term_exp_sin(
    P p,
    float amp,
    float k,
    float ax, float ay, float az,
    float bx, float by, float bz,
    float phi) {
    const float adot = ax * p.x + ay * p.y + az * p.z;
    const float bdot = bx * p.x + by * p.y + bz * p.z;
    const float f = amp * expf(k * adot) * sinf(k * bdot + phi);
    const float u = -(adot / (2.0f * k)) * f;
    return {u, f};
}

template <typename P>
N2WOS_HD_INLINE float nc_poisson_multiscale_u(P p) {
    float u = 0.0f;
    // Low/mid frequency source terms visible in the Bunny interior.
    u += nc_poisson_term_exp_cos(p, 0.32f, 1.8f,  1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.15f).u;
    u += nc_poisson_term_exp_sin(p, -0.26f, 2.1f, 0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 1.0f, -0.40f).u;

    const float inv2 = 0.70710678118f;
    u += nc_poisson_term_exp_cos(p, 0.22f, 2.7f,  inv2, inv2, 0.0f, -inv2, inv2, 0.0f, 0.50f).u;
    u += nc_poisson_term_exp_sin(p, 0.18f, 3.0f,  inv2, 0.0f, inv2, -inv2, 0.0f, inv2, -0.10f).u;

    // A small harmonic background / trend.  It has zero source and helps avoid a
    // purely zero-mean manufactured solution.
    u += 0.10f * p.x - 0.07f * p.y + 0.04f * p.z;
    return u;
}

template <typename P>
N2WOS_HD_INLINE float nc_poisson_multiscale_f(P p) {
    float f = 0.0f;
    f += nc_poisson_term_exp_cos(p, 0.32f, 1.8f,  1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.15f).f;
    f += nc_poisson_term_exp_sin(p, -0.26f, 2.1f, 0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 1.0f, -0.40f).f;
    const float inv2 = 0.70710678118f;
    f += nc_poisson_term_exp_cos(p, 0.22f, 2.7f,  inv2, inv2, 0.0f, -inv2, inv2, 0.0f, 0.50f).f;
    f += nc_poisson_term_exp_sin(p, 0.18f, 3.0f,  inv2, 0.0f, inv2, -inv2, 0.0f, inv2, -0.10f).f;
    return f;
}

template <typename P>
N2WOS_HD_INLINE float nc_poisson_figlike_hf_u(P p) {
    float u = 0.0f;
    const float inv2 = 0.70710678118f;
    u += nc_poisson_term_exp_cos(p, 0.28f, 2.4f,  1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.25f).u;
    u += nc_poisson_term_exp_sin(p, -0.22f, 2.7f, 0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 1.0f, -0.30f).u;
    u += nc_poisson_term_exp_cos(p, 0.18f, 3.4f,  inv2, inv2, 0.0f, -inv2, inv2, 0.0f, 0.85f).u;
    u += nc_poisson_term_exp_sin(p, 0.14f, 3.8f,  inv2, 0.0f, inv2, -inv2, 0.0f, inv2, 0.10f).u;
    // yz oblique pair: a=(0,0.8,0.6), b=(0,-0.6,0.8)
    u += nc_poisson_term_exp_cos(p, -0.12f, 4.2f, 0.0f, 0.8f, 0.6f, 0.0f, -0.6f, 0.8f, -0.20f).u;
    u += 0.08f * p.x + 0.05f * p.y - 0.03f * p.z;
    return u;
}

template <typename P>
N2WOS_HD_INLINE float nc_poisson_figlike_hf_f(P p) {
    float f = 0.0f;
    const float inv2 = 0.70710678118f;
    f += nc_poisson_term_exp_cos(p, 0.28f, 2.4f,  1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.25f).f;
    f += nc_poisson_term_exp_sin(p, -0.22f, 2.7f, 0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 1.0f, -0.30f).f;
    f += nc_poisson_term_exp_cos(p, 0.18f, 3.4f,  inv2, inv2, 0.0f, -inv2, inv2, 0.0f, 0.85f).f;
    f += nc_poisson_term_exp_sin(p, 0.14f, 3.8f,  inv2, 0.0f, inv2, -inv2, 0.0f, inv2, 0.10f).f;
    f += nc_poisson_term_exp_cos(p, -0.12f, 4.2f, 0.0f, 0.8f, 0.6f, 0.0f, -0.6f, 0.8f, -0.20f).f;
    return f;
}

template <typename P>
N2WOS_HD_INLINE float nc_poisson_manufactured_u(P p, int mode) {
    if (mode == kNcPoissonMultiscaleMode) return nc_poisson_multiscale_u(p);
    if (mode == kNcPoissonFiglikeHfMode) return nc_poisson_figlike_hf_u(p);
    return 0.0f;
}

template <typename P>
N2WOS_HD_INLINE float nc_poisson_manufactured_f(P p, int mode) {
    if (mode == kNcPoissonMultiscaleMode) return nc_poisson_multiscale_f(p);
    if (mode == kNcPoissonFiglikeHfMode) return nc_poisson_figlike_hf_f(p);
    return 0.0f;
}

N2WOS_HD_INLINE bool nc_poisson_manufactured_is_mode(int mode) {
    return mode == kNcPoissonMultiscaleMode || mode == kNcPoissonFiglikeHfMode;
}

template <typename P>
N2WOS_HD_INLINE float nc_poisson_center_green_contribution(P p, float radius, int mode) {
    if (!nc_poisson_manufactured_is_mode(mode)) return 0.0f;
    // For -Delta u = f in 3D, and for harmonic f, the exact Green integral over
    // a WoS ball centered at p is radius^2 / 6 * f(p).
    return (radius * radius) * (1.0f / 6.0f) * nc_poisson_manufactured_f(p, mode);
}

}  // namespace n2wos


// Higher-frequency manufactured Poisson field. This is a bounded Fourier
// mixture. For Poisson we can prescribe an interior source f=-Delta u, so
// internal spatial frequencies need not be generated from boundary data alone.
//
// The returned value is u(x). The source routine returns f(x)=-Delta u. The
// x5 suffix means that the dominant angular frequencies are roughly five times
// larger than the low/mid-frequency Poisson diagnostic modes.
template <typename P>
N2WOS_HD_INLINE float nc_poisson_figlike_hf_x5_value(P p) {
  const float x = p.x;
  const float y = p.y;
  const float z = p.z;

  // Moderate amplitude keeps comparisons from being dominated by dynamic range.
  float u = 0.0f;
  u += 0.090f * sinf(12.0f * x +  4.0f * y +  1.5f * z + 0.30f);
  u += 0.075f * cosf(-5.0f * x + 13.5f * y +  2.0f * z - 0.70f);
  u += 0.060f * sinf( 7.5f * x -  3.5f * y + 15.0f * z + 1.10f);
  u += 0.050f * cosf(14.0f * x - 10.0f * y +  4.0f * z + 2.20f);
  u += 0.045f * sinf(-9.0f * x +  6.5f * y + 12.0f * z - 1.30f);
  u += 0.035f * cosf(18.0f * x +  2.5f * y -  7.0f * z + 0.80f);

  // Weak low-frequency component to make large-scale bias visible.
  u += 0.030f * (0.60f * x - 0.35f * y + 0.20f * z);
  return u;
}

template <typename P>
N2WOS_HD_INLINE float nc_poisson_figlike_hf_x5_source(P p) {
  const float x = p.x;
  const float y = p.y;
  const float z = p.z;

  float f = 0.0f;
  // For a*sin(k.x+phi) or a*cos(k.x+phi), -Delta gives |k|^2 times the same term.
  f += 0.090f * (12.0f*12.0f +  4.0f* 4.0f +  1.5f* 1.5f) * sinf(12.0f * x +  4.0f * y +  1.5f * z + 0.30f);
  f += 0.075f * ( 5.0f* 5.0f + 13.5f*13.5f +  2.0f* 2.0f) * cosf(-5.0f * x + 13.5f * y +  2.0f * z - 0.70f);
  f += 0.060f * ( 7.5f* 7.5f +  3.5f* 3.5f + 15.0f*15.0f) * sinf( 7.5f * x -  3.5f * y + 15.0f * z + 1.10f);
  f += 0.050f * (14.0f*14.0f + 10.0f*10.0f +  4.0f* 4.0f) * cosf(14.0f * x - 10.0f * y +  4.0f * z + 2.20f);
  f += 0.045f * ( 9.0f* 9.0f +  6.5f* 6.5f + 12.0f*12.0f) * sinf(-9.0f * x +  6.5f * y + 12.0f * z - 1.30f);
  f += 0.035f * (18.0f*18.0f +  2.5f* 2.5f +  7.0f* 7.0f) * cosf(18.0f * x +  2.5f * y -  7.0f * z + 0.80f);
  return f;
}

template <typename P>
N2WOS_HD_INLINE float nc_poisson_figlike_hf_x5_center_green_contribution(P p, float radius) {
  return (radius * radius / 6.0f) * nc_poisson_figlike_hf_x5_source(p);
}

