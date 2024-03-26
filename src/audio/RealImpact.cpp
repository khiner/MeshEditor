#include "RealImpact.h"

#include "npy.h"
#include <glm/gtc/matrix_transform.hpp>

#include "numeric/vec4.h"

/**
All files (present for each object):
  - angle.npy (24K)
  - deconvolved_0db.npy (2.3G) // Loaded dynamically
  - distance.npy (24K)
  - listenerXYZ.npy (72K)
  - material_0.mtl (4.0K)
  - material_0.png (1.2M)
  - micID.npy (24K)
  - transformed.obj (3.4M)
  - vertexID.npy (24K)
  - vertexXYZ.npy (72K)
*/

RealImpact::RealImpact(const fs::path &directory)
    : Directory(directory), ObjPath(directory / "transformed.obj") {}

std::vector<RealImpactListenerPoint> RealImpact::LoadListenerPoints() const {
    // static const uint NumVertexIds = 5; // Number of unique vertex IDs
    static const uint NumListenerPositions = 600; // Number of unique listener positions

    const auto mic_ids = npy::read_npy<long>(Directory / "micID.npy");
    const auto angles = npy::read_npy<long>(Directory / "angle.npy");
    const auto distances = npy::read_npy<long>(Directory / "distance.npy");

    std::vector<RealImpactListenerPoint> listener_points;
    listener_points.reserve(NumListenerPositions);
    for (uint i = 0; i < NumListenerPositions; ++i) {
        listener_points.push_back(RealImpactListenerPoint{
            .Index = i,
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
vec3 RealImpactListenerPoint::GetPosition(vec3 world_up, bool mic_center) const {
    const float angle = glm::radians(float(AngleDeg)), dist = float(DistanceMm);
    const vec3 pos{
        // 230 I believe is for the gantry (where the object is placed)
        230 + dist + (mic_center ? RealImpact::MicLengthMm / 2 : 0),
        -(RealImpact::MicBarLengthMm / 2) + (float(MicId) / (RealImpact::NumMics - 1)) * RealImpact::MicBarLengthMm,
        // I beleive these offseta are to accurately reflect the mic positions attached to _one side_ of microphone array bar.
        // You can see the same offsets in https://samuelpclarke.com/realimpact/ and https://www.youtube.com/watch?v=OeZMeze-oIs
        ((45.f / 2.f) + 20.95f),
    };
    return vec3{glm::rotate({1}, angle, world_up) * vec4{pos, 1}} / 1000.f;
}

std::vector<std::vector<float>> RealImpactListenerPoint::LoadImpactSamples(const RealImpact &parent) const {
    std::vector<std::vector<float>> all_samples;
    all_samples.reserve(RealImpact::NumImpactVertices);
    for (uint i = 0; i < RealImpact::NumImpactVertices; ++i) {
        const size_t offset = i * RealImpact::NumListenerPoints + Index;
        const size_t size = 209549; // todo get from shape
        all_samples.emplace_back(npy::read_npy<float>(parent.Directory / "deconvolved_0db.npy", offset, size).data);
    }
    return all_samples;
}
