#ifndef BOXWIRE_MSL
#define BOXWIRE_MSL

// Corner indices for the 12 box edges. Corner bits select min/max per axis: x=1, y=2, z=4.
constant uint BoxEdgeCorners[24] = {
    0u, 1u, 1u, 3u, 3u, 2u, 2u, 0u, // bottom ring
    4u, 5u, 5u, 7u, 7u, 6u, 6u, 4u, // top ring
    0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u // verticals
};

#endif
