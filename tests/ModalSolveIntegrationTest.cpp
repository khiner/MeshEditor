#include "audio/ModalSolve.h"

#include <FastFEM/SolveMonitor.h>
#include <boost/ut.hpp>

#include <array>
#include <type_traits>
#include <vector>

using namespace boost::ut;

static_assert(std::is_same_v<vec3, fastfem::Vec3>);
static_assert(std::is_same_v<dvec3, fastfem::DVec3>);
static_assert(std::is_same_v<uvec3, fastfem::UVec3>);
static_assert(std::is_same_v<quat, fastfem::Quat>);
static_assert(std::is_same_v<AcousticMaterialProperties, fastfem::AcousticMaterialProperties>);
static_assert(std::is_same_v<MassProperties, fastfem::MassProperties>);
static_assert(std::is_base_of_v<fastfem::ModalModes, ModalModes>);
static_assert(std::is_base_of_v<fastfem::ModalEigenSummary, ModalEigenSummary>);

int main() {
    "FastFEM surface solve integrates with modal components"_test = [] {
        const std::vector<vec3> positions{
            {0, 0, 0},
            {0.2f, 0, 0},
            {0.2f, 0.1f, 0},
            {0, 0.1f, 0},
            {0, 0, 0.08f},
            {0.2f, 0, 0.08f},
            {0.2f, 0.1f, 0.08f},
            {0, 0.1f, 0.08f},
        };
        const std::vector<uint32_t> triangles{
            0,
            2,
            1,
            0,
            3,
            2,
            4,
            5,
            6,
            4,
            6,
            7,
            0,
            1,
            5,
            0,
            5,
            4,
            1,
            2,
            6,
            1,
            6,
            5,
            2,
            3,
            7,
            2,
            7,
            6,
            3,
            0,
            4,
            3,
            4,
            7,
        };
        constexpr AcousticMaterialProperties material{
            .Density = 1000,
            .YoungModulus = 1e7,
            .PoissonRatio = 0.2,
            .Alpha = 1,
            .Beta = 1e-7,
        };
        fastfem::SolveMonitor monitor;
        const auto result = modal::SolveSurfaceModes(
            positions, triangles, material, positions, vec3{1}, fastfem::Discretization::Tet10,
            {.Modal = {.MinModeFreq = 1, .MaxModeFreq = 100'000, .NumModes = 4, .NumFemModes = 12, .MaxRestarts = 150}},
            {.KeepBasis = true}, &monitor
        );
        expect(bool(result)) << (result ? "" : result.error());
        if (!result) return;
        expect(!result->Modes.Freqs.empty());
        expect(result->Modes.Shapes.size() == positions.size());
        expect(result->Summary.Eigenvalues.size() == 12_u);
        expect(result->Mass.Mass > 0.0);
        expect(!result->Tetrahedra.Positions.empty());
        expect(!result->Tetrahedra.EdgeIndices.empty());
        expect(bool(result->Basis));
        expect(monitor.Progress.load(std::memory_order_relaxed) == 1.f);

        auto current = result->Modes;
        current.Vertices = {0, 1, 2};
        current.Indices = {0, 1, 2};
        auto rescaled = modal::RescaleModes(result->Summary, current, {.Density = 2000, .YoungModulus = 2e7, .PoissonRatio = 0.2, .Alpha = 1, .Beta = 1e-7});
        expect(bool(rescaled));
        expect(rescaled->Vertices == current.Vertices);
        expect(rescaled->Indices == current.Indices);
    };
}
