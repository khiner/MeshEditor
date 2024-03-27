#include "ModalSoundObject.h"

#include "mesh/Mesh.h"

// todo

ModalSoundObject::ModalSoundObject(const ::Mesh &mesh, vec3 listener_position) : SoundObject(listener_position), Mesh(mesh) {}

void ModalSoundObject::ProduceAudio(DeviceData, float *output, uint frame_count) {
    for (uint i = 0; i < frame_count; ++i) output[i] += 0.0f;
}

void ModalSoundObject::Strike(uint vertex_index, float force) {
}
