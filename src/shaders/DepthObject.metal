#ifndef DEPTHOBJECT_MSL
#define DEPTHOBJECT_MSL

#include "Varyings.metal"

fragment float2 DepthObjectFragment(ObjectIdVaryings in [[stage_in]]) {
    return float2(in.Position.z, float(in.ObjectId));
}

#endif
