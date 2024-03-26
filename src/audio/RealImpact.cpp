#include "RealImpact.h"

#include "npy.h"
#include <glm/gtc/matrix_transform.hpp>

#include "numeric/vec4.h"

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

/*
See https://github.com/samuel-clarke/RealImpact/blob/main/preprocess_measurements.py
Here, we reproduce the `get_mic_world_space` function, to avoid redundantly storing positions.
The only difference is we use Y-up instead of Z-up.
    MIC_BAR_LENGTH = 1890 - 70
    def get_mic_world_space(angle, distance, ind):
        mic_z = -(MIC_BAR_LENGTH/2) + ind/14 * MIC_BAR_LENGTH
        mic_x = 230 + distance
        mic_y = -((45/2) + 20.95) * np.ones_like(angle)
        mic_points = np.vstack((mic_x, mic_y, mic_z)).transpose()
        rot = Rotation.from_euler('z', angle, degrees=True)
        pos_meters = rot.apply(mic_points) / 1000.0
        return pos_meters
*/
vec3 RealImpact::GetPosition(const ListenerPoint &p, vec3 world_up, bool mic_center) {
    const float angle = glm::radians(float(p.AngleDeg)), dist = float(p.DistanceMm);
    const vec3 pos{
        // 230 I believe is for the gantry (where the object is placed)
        230 + dist + (mic_center ? MicLengthMm / 2 : 0),
        -(MicBarLengthMm / 2) + (float(p.MicId) / (NumMics - 1)) * MicBarLengthMm,
        // I beleive these offseta are to accurately reflect the mic positions attached to _one side_ of microphone array bar.
        // You can see the same offsets in https://samuelpclarke.com/realimpact/ and https://www.youtube.com/watch?v=OeZMeze-oIs
        ((45.f / 2.f) + 20.95f),
    };
    return vec3{glm::rotate({1}, angle, world_up) * vec4{pos, 1}} / 1000.f;
}

RealImpact::RealImpact(const fs::path &directory)
    : Directory(directory), ObjPath(directory / "transformed.obj"), ListenerPoints(LoadListenerPoints(directory)) {}

std::vector<float> RealImpact::LoadSamples(const ListenerPoint &) const {
    // todo add offset to `read_npy` to skip the first `Index` samples
    return {};
}
