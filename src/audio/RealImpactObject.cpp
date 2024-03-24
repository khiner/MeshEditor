#include "RealImpactObject.h"

#include "npy.h"

/**
All files (present for each object):
  - angle.npy (24K)
  - deconvolved_0db.npy (2.3G) // Not part of `ListenerPoint`.
  - distance.npy (24K)
  - listenerXYZ.npy (72K)
  - material_0.mtl (4.0K)
  - material_0.png (1.2M)
  - micID.npy (24K)
  - transformed.obj (3.4M)
  - vertexID.npy (24K)
  - vertexXYZ.npy (72K)
*/

std::vector<RealImpactObject::ListenerPoint> LoadListenerPoints(const fs::path &directory) {
    const auto angles = npy::read_npy<long>(directory / "angle.npy");
    const auto distances = npy::read_npy<long>(directory / "distance.npy");
    const auto positions = npy::read_npy<double>(directory / "listenerXYZ.npy");

    const size_t num_listeners = angles.shape[0];
    std::vector<RealImpactObject::ListenerPoint> listener_points;
    listener_points.reserve(num_listeners);
    for (uint i = 0; i < num_listeners; ++i) {
        listener_points.push_back({
            .Index = i,
            .AngleDeg = angles.data[i],
            .DistanceMm = distances.data[i],
            .Position = {positions.data[i * 3], positions.data[i * 3 + 1], positions.data[i * 3 + 2]},
        });
    }
    return listener_points;
}

RealImpactObject::RealImpactObject(const fs::path &directory)
    : Directory(std::move(directory)), ObjPath(directory / "transformed.obj"), ListenerPoints(LoadListenerPoints(directory)) {}

std::vector<float> RealImpactObject::LoadSamples(const ListenerPoint &) const {
    // todo add offset to `read_npy` to skip the first `Index` samples
    return {};
}
