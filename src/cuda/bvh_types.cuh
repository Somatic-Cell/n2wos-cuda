#pragma once

#include <cfloat>

#ifndef N2WOS_HD
#  ifdef __CUDACC__
#    define N2WOS_HD __host__ __device__
#    define N2WOS_D __device__
#  else
#    define N2WOS_HD
#    define N2WOS_D
#  endif
#endif

namespace n2wos {

struct DeviceVec3 {
  float x;
  float y;
  float z;
};

struct DeviceTriangle {
  DeviceVec3 v0;
  DeviceVec3 v1;
  DeviceVec3 v2;
};

struct DeviceBvhNode {
  DeviceVec3 bbox_min;
  DeviceVec3 bbox_max;
  int left;
  int right;
  int first;
  int count;
};

N2WOS_HD inline DeviceVec3 d_make_vec3(float x, float y, float z) {
  DeviceVec3 out{x, y, z};
  return out;
}

N2WOS_HD inline DeviceVec3 d_add(DeviceVec3 a, DeviceVec3 b) {
  return d_make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

N2WOS_HD inline DeviceVec3 d_sub(DeviceVec3 a, DeviceVec3 b) {
  return d_make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

N2WOS_HD inline DeviceVec3 d_mul(DeviceVec3 a, float s) {
  return d_make_vec3(a.x * s, a.y * s, a.z * s);
}

N2WOS_HD inline float d_dot(DeviceVec3 a, DeviceVec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

N2WOS_HD inline float d_length2(DeviceVec3 a) {
  return d_dot(a, a);
}

N2WOS_HD inline DeviceVec3 d_min(DeviceVec3 a, DeviceVec3 b) {
  return d_make_vec3(a.x < b.x ? a.x : b.x,
                     a.y < b.y ? a.y : b.y,
                     a.z < b.z ? a.z : b.z);
}

N2WOS_HD inline DeviceVec3 d_max(DeviceVec3 a, DeviceVec3 b) {
  return d_make_vec3(a.x > b.x ? a.x : b.x,
                     a.y > b.y ? a.y : b.y,
                     a.z > b.z ? a.z : b.z);
}

N2WOS_HD inline float d_clamp(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

N2WOS_HD inline float d_aabb_distance2(DeviceVec3 p, DeviceVec3 bmin, DeviceVec3 bmax) {
  float dx = 0.0f;
  if (p.x < bmin.x) dx = bmin.x - p.x;
  if (p.x > bmax.x) dx = p.x - bmax.x;
  float dy = 0.0f;
  if (p.y < bmin.y) dy = bmin.y - p.y;
  if (p.y > bmax.y) dy = p.y - bmax.y;
  float dz = 0.0f;
  if (p.z < bmin.z) dz = bmin.z - p.z;
  if (p.z > bmax.z) dz = p.z - bmax.z;
  return dx * dx + dy * dy + dz * dz;
}

}  // namespace n2wos
