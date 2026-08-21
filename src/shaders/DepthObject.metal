#ifndef DEPTHOBJECT_MSL
#define DEPTHOBJECT_MSL

#include "Varyings.metal"

fragment float2 DepthObjectFragment(ObjectIdFragmentVaryings in [[stage_in]]) {
    return float2(in.Position.z, float(in.ObjectId));
}

#endif
