// Measures the cost of solved-contact reporting on the same RBP workload.
#include "Solver.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    const uint32_t bodies = argc > 1 ? std::max(1, std::atoi(argv[1])) : 128;
    const uint32_t steps = argc > 2 ? std::max(1, std::atoi(argv[2])) : 600;
    rbp::mtl::Context context;
    rbp::Solver solver{context};
    for (bool reporting : {false, true}) {
        rbp::World world{context, {.Bodies = bodies + 1, .Shapes = 2}};
        rbp::Shape plane{}, box{};
        plane.Kind = rbp::ShapePlane;
        plane.Normal = {0, 1, 0};
        box.Kind = rbp::ShapeBox;
        box.HalfExtents = {0.5f, 0.5f, 0.5f};
        rbp::BodyDesc floor{};
        floor.Shape = world.AddShape(plane);
        world.AddBody(floor);
        const auto shape = world.AddShape(box);
        for (uint32_t i = 0; i < bodies; ++i) {
            rbp::BodyDesc body{};
            body.Shape = shape;
            body.Pose = rbp::At(rbp::float3{2 * float(i % 16), 0.5f, 2 * float(i / 16)});
            world.AddBody(body);
        }
        rbp::StepSettings settings;
        settings.SleepSteps = UINT32_MAX;
        for (int i = 0; i < 60; ++i) solver.Step(world, settings);
        world.TrackContacts = reporting;
        size_t events = 0;
        const auto began = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < steps; ++i) {
            solver.Step(world, settings);
            if (reporting) events += world.TakeContactChanges().size();
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        std::printf("reporting=%d bodies=%u steps=%u events=%zu ms_per_step=%.4f\n", int(reporting), bodies, steps, events, seconds * 1000 / steps);
    }
}
