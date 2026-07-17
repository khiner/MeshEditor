#ifndef VELOCITY_GLSL
#define VELOCITY_GLSL

// Screen motion is stored at 1/100 scale so extreme projections stay within half-float range.
vec4 PackVelocity(vec4 motion) { return motion * 0.01; }
vec4 UnpackVelocity(vec4 motion) { return motion * 100.0; }

#endif
