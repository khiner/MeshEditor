#include "RealImpact.h"

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

std::vector<RealImpact::ListenerPoint> LoadListenerPoints(const fs::path &directory) {
    // static const uint NumVertexIds = 5; // Number of unique vertex IDs
    static const uint NumListenerPositions = 600; // Number of unique listener positions

    // todo only read the first `NumListenerPositions` listener positions
    const auto mic_ids = npy::read_npy<long>(directory / "micID.npy");
    const auto angles = npy::read_npy<long>(directory / "angle.npy");
    const auto distances = npy::read_npy<long>(directory / "distance.npy");

    std::vector<RealImpact::ListenerPoint> listener_points;
    listener_points.reserve(NumListenerPositions);
    for (uint i = 0; i < NumListenerPositions; ++i) {
        listener_points.push_back(RealImpact::ListenerPoint{
            .MicId = mic_ids.data[i],
            .DistanceMm = distances.data[i],
            .AngleDeg = angles.data[i],
        });
    }
    return listener_points;
}

RealImpact::RealImpact(const fs::path &directory)
    : Directory(std::move(directory)), ObjPath(directory / "transformed.obj"), ListenerPoints(LoadListenerPoints(directory)) {}

std::vector<float> RealImpact::LoadSamples(const ListenerPoint &) const {
    // todo add offset to `read_npy` to skip the first `Index` samples
    return {};
}
