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

// Analytic manufactured harmonic fields for Neural Cache / 2LMC diagnostics.
// Each non-linear term has the form exp(k a·x) cos/sin(k b·x + phi),
// with a and b orthonormal.  Hence the second derivative in the exponential
// direction is cancelled by the second derivative in the oscillatory direction,
// so each term is harmonic.  Linear terms are harmonic as well.
//
// These functions must be evaluated in the same solver-normalized coordinate
// system as the existing boundary modes.

template <typename P>
N2WOS_HD_INLINE float nc_harmonic_mixture_smooth_value(P p) {
    const float x = p.x;
    const float y = p.y;
    const float z = p.z;

    const float t0 = 0.42f * expf(1.15f * x) * cosf(1.15f * y + 0.15f);
    const float t1 = -0.30f * expf(-1.05f * y) * sinf(1.05f * z - 0.25f);

    // p=(0.6,0,0.8), q=(-0.8,0,0.6), p·q=0.
    const float p2 = 0.6f * x + 0.8f * z;
    const float q2 = -0.8f * x + 0.6f * z;
    const float t2 = 0.24f * expf(1.25f * p2) * cosf(1.25f * q2 + 0.70f);

    // p=(1,1,1)/sqrt(3), q=(1,-1,0)/sqrt(2), p·q=0.
    const float p3 = 0.57735026919f * (x + y + z);
    const float q3 = 0.70710678118f * (x - y);
    const float t3 = 0.18f * expf(-1.10f * p3) * sinf(1.10f * q3 + 1.10f);

    const float lin = 0.12f * x - 0.08f * y + 0.06f * z;
    return 0.72f * (t0 + t1 + t2 + t3 + lin);
}

template <typename P>
N2WOS_HD_INLINE float nc_harmonic_mixture_figlike_value(P p) {
    const float x = p.x;
    const float y = p.y;
    const float z = p.z;

    // Stronger interior-visible variation than smooth, but still analytic
    // and harmonic.  Coefficients are scaled to keep the target roughly O(1)
    // on the normalized Bunny domain.
    const float t0 = 0.36f * expf(1.65f * x) * cosf(1.65f * y + 0.35f);
    const float t1 = -0.30f * expf(-1.45f * y) * cosf(1.45f * z - 0.20f);

    // 45-degree xz pair: p=(1/sqrt(2),0,1/sqrt(2)),
    // q=(-1/sqrt(2),0,1/sqrt(2)), p·q=0.
    const float inv_sqrt2 = 0.70710678118f;
    const float p2 = inv_sqrt2 * (x + z);
    const float q2 = inv_sqrt2 * (-x + z);
    const float t2 = 0.27f * expf(1.85f * p2) * sinf(1.85f * q2 + 0.55f);

    // yz pair: p=(0,0.8,0.6), q=(0,-0.6,0.8), p·q=0.
    const float p3 = 0.8f * y + 0.6f * z;
    const float q3 = -0.6f * y + 0.8f * z;
    const float t3 = 0.18f * expf(-1.30f * p3) * cosf(1.30f * q3 - 0.90f);

    const float lin = 0.10f * x + 0.14f * y - 0.05f * z;
    return 0.58f * (t0 + t1 + t2 + t3 + lin);
}


template <typename P>
N2WOS_HD_INLINE float nc_harmonic_mixture_multiscale_value(P p) {
    const float x = p.x;
    const float y = p.y;
    const float z = p.z;
    const float inv2 = 0.70710678118f;
    const float inv3 = 0.57735026919f;

    // Low / middle frequency content that remains visible in the interior.
    const float t0 = 0.20f * expf(1.35f * x) * cosf(1.35f * y + 0.20f);
    const float t1 = -0.17f * expf(-1.55f * y) * sinf(1.55f * z - 0.45f);
    const float t2 = 0.15f * expf(1.70f * z) * cosf(1.70f * x + 0.80f);

    // Diagonal middle-frequency components.  Each p/q pair is orthonormal.
    const float p3 = inv2 * (x + y);
    const float q3 = inv2 * (-x + y);
    const float t3 = 0.095f * expf(2.55f * p3) * cosf(2.55f * q3 + 0.15f);

    const float p4 = inv2 * (y + z);
    const float q4 = inv2 * (-y + z);
    const float t4 = -0.080f * expf(-2.85f * p4) * sinf(2.85f * q4 - 0.65f);

    const float p5 = inv2 * (x + z);
    const float q5 = inv2 * (-x + z);
    const float t5 = 0.070f * expf(3.15f * p5) * cosf(3.15f * q5 + 1.10f);

    // A 3D oblique pair: a=(1,1,1)/sqrt(3), b=(1,-1,0)/sqrt(2).
    const float p6 = inv3 * (x + y + z);
    const float q6 = inv2 * (x - y);
    const float t6 = 0.060f * expf(-3.35f * p6) * sinf(3.35f * q6 + 0.40f);

    // A small high-ish component.  The coefficient is deliberately small so it
    // creates visible interior texture without dominating the global field.
    const float p7 = 0.81649658093f * x - 0.40824829046f * y - 0.40824829046f * z;
    const float q7 = inv2 * (y - z);
    const float t7 = 0.035f * expf(4.00f * p7) * cosf(4.00f * q7 - 0.35f);

    const float lin = 0.035f * x - 0.025f * y + 0.020f * z;
    return 0.62f * (t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + lin);
}

template <typename P>
N2WOS_HD_INLINE float nc_harmonic_mixture_figlike_hf_value(P p) {
    const float x = p.x;
    const float y = p.y;
    const float z = p.z;
    const float inv2 = 0.70710678118f;

    // More aggressive than multiscale.  This is intended for figure making and
    // stress testing; it may be harder for the neural cache than the smooth and
    // multiscale variants.
    const float t0 = 0.18f * expf(1.65f * x) * cosf(1.65f * y + 0.35f);
    const float t1 = -0.15f * expf(-1.75f * y) * cosf(1.75f * z - 0.20f);
    const float t2 = 0.13f * expf(1.85f * z) * sinf(1.85f * x + 0.70f);

    const float p3 = inv2 * (x + z);
    const float q3 = inv2 * (-x + z);
    const float t3 = 0.095f * expf(2.90f * p3) * sinf(2.90f * q3 + 0.55f);

    const float p4 = inv2 * (x + y);
    const float q4 = inv2 * (-x + y);
    const float t4 = -0.085f * expf(-3.25f * p4) * cosf(3.25f * q4 - 0.75f);

    const float p5 = inv2 * (y + z);
    const float q5 = inv2 * (-y + z);
    const float t5 = 0.070f * expf(3.55f * p5) * sinf(3.55f * q5 + 1.20f);

    // Two small high-frequency interior-visible terms.
    const float p6 = 0.86602540378f * x + 0.5f * z;
    const float q6 = -0.5f * x + 0.86602540378f * z;
    const float t6 = 0.040f * expf(4.20f * p6) * cosf(4.20f * q6 + 0.25f);

    const float p7 = 0.8f * y + 0.6f * z;
    const float q7 = -0.6f * y + 0.8f * z;
    const float t7 = -0.032f * expf(-4.50f * p7) * sinf(4.50f * q7 - 0.30f);

    const float lin = 0.025f * x + 0.020f * y - 0.015f * z;
    return 0.55f * (t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + lin);
}

}  // namespace n2wos
