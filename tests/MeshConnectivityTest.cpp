#include "mesh/Mesh.h"

#include "RunSuites.h"

#include <boost/ut.hpp>

using namespace boost::ut;

int main() {
    "shared edges preserve reciprocal halfedges and terminating vertex fans"_test = [] {
        // A closed tetrahedron with one or two duplicated faces adds third and fourth edge incidences.
        constexpr uint32_t corners[]{0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3, 0, 2, 1, 0, 1, 3};
        for (uint32_t count : {12u, 15u, 18u}) {
            std::vector<he::HH> outgoing(4), opposites(count);
            std::vector<uint32_t> bits(1), ranks(1), samples(1);
            const auto built = BuildConnectivity({}, std::span{corners}.first(count), 4, {outgoing, opposites, bits, ranks, samples, {}});
            const MeshConnectivity connectivity{
                .VertexCount = 4,
                .OutgoingHalfedges = outgoing,
                .Opposites = opposites,
                .EdgeFirstBits = bits,
                .EdgeFirstRanks = ranks,
                .HalfedgeToEdge = built.HalfedgeToEdge,
                .EdgeCount = built.EdgeCount,
                .Edges = built.Edges,
                .EdgeSamples = samples,
                .FaceCount = count / 3,
                .Faces = {},
            };
            expect(built.EdgeCount == 6_u);
            for (uint32_t h = 0; h < count; ++h) {
                if (const auto opposite = opposites[h]) expect(opposites[*opposite] == he::HH(h));
                const auto from = corners[*connectivity.Previous(he::HH(h))], to = corners[h];
                for (uint32_t other = 0; other < h; ++other) {
                    const auto other_from = corners[*connectivity.Previous(he::HH(other))], other_to = corners[other];
                    if ((from == other_from && to == other_to) || (from == other_to && to == other_from)) {
                        expect(connectivity.Edge(he::HH(h)) == connectivity.Edge(he::HH(other)));
                    }
                }
                auto current = he::HH(h);
                uint32_t visited = 0;
                do {
                    const auto opposite = opposites[*current];
                    current = opposite ? connectivity.Next(opposite) : he::HH{};
                    ++visited;
                } while (current && current != he::HH(h) && visited <= count);
                expect(visited <= count);
            }
        }
    };
    return RunSuites();
}
