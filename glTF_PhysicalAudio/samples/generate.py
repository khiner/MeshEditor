#!/usr/bin/env python3
"""Generate the KHR_audio_rigid_bodies sample scenes.

The root of samples/ holds the full-experience scenes, each a complete listening experience with
every mechanism sounding at once. Pile is the first: five bodies of stepped geometric complexity
dropped together onto a radiating platform.

test/ holds the isolation corpus. Each phenomenon there is a directory of sibling scenes, one
configuration sounding per scene, named in a/b/c ladder order, so every effect is heard and measured
in isolation rather than superimposed in one mix. Each scene isolates one prediction the extension
makes, so that what you hear either matches it or does not. Bodies start just above a static ground
and are dropped, giving an impact followed by the settling and resting the contact model governs.

Written against the extension text rather than exported from any implementation, so a renderer that
disagrees with these is disagreeing with the specification.

Run with scene name fragments as arguments to regenerate only the scenes naming them.
"""

import base64
import cmath
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

OUT = Path(__file__).parent

# Scene name fragments to regenerate, empty regenerating everything.
# Set from the command line when run as a script, so an edit to one scene's authoring re-emits that scene alone.
Only = frozenset()


def selected(name):
    """Whether this run regenerates the scene called `name`."""
    return not Only or any(o in name for o in Only)


# Ceramic, matching the extension's own example, and steel for a body stiff enough to ring past hearing.
# Their Rayleigh constants differ, so a scene using both hears the damping the material carries.
CERAMIC = {"name": "Ceramic", "density": 2700, "youngsModulus": 7.2e10, "poissonRatio": 0.19, "alpha": 6, "beta": 1e-7}
STEEL = {"name": "Steel", "density": 7850, "youngsModulus": 2.0e11, "poissonRatio": 0.29, "alpha": 5, "beta": 3e-8}
# Low-loss glass for bodies whose free ring must outlast contact-borne dissipation, so a scene can hear the contact choke the ring.
# Loss factor ~4e-4 at 5 kHz, the measured range for glass.
GLASS = {"name": "Glass", "density": 2500, "youngsModulus": 7.0e10, "poissonRatio": 0.22, "alpha": 3, "beta": 1e-8}
ACOUSTIC_MATERIALS = [CERAMIC, STEEL]

# The extension's representative finishes, matching the table in its Authoring Notes.
# Roughness sets the rate a body oscillates on its contact at, so these are the ladder the second scene sweeps.
# Each carries the waviness of its own process, the mould's form for a casting and the tool path for a cut face, within the ordering the extension requires of the two scales.
FINISHES = {
    "Polished": {"roughness": 1e-7, "correlationLength": 1e-5, "spectralSlope": -2.8,
                 "waviness": 2e-7, "wavinessLength": 5e-3},
    "Machined": {"roughness": 2e-6, "correlationLength": 5e-5, "spectralSlope": -2.4,
                 "waviness": 2e-6, "wavinessLength": 2e-3},
    "Cast": {"roughness": 1e-4, "correlationLength": 1e-3, "spectralSlope": -2.0,
             "waviness": 2e-4, "wavinessLength": 2e-2},
}
# Where each finish lands in the surface table a scene carries by default.
FinishIndex = {name: i for i, name in enumerate(FINISHES)}


# A body at rest on a surface it is not moving over excites nothing, which the extension requires, so these scenes slide instead of settling.
# Each body starts barely touching its lane and holds its speed, giving one continuous contact at one operating point.
# Contact is a train of shocks between opposing asperities arriving at the sliding speed over the correlation length, so the speed decides whether the mechanism is audible at all.
# This is Dang 2013's middle speed, which on the Machined finish's 50 um correlation length puts that rate at 2 kHz.
# The lanes are frictionless because the rate has to hold: a body slowing under a realistic friction coefficient sweeps its own operating point, and one that holds an audible rate for even half a second has to start above hearing.
# The lanes' own material: frictionless so a sliding body holds its speed, and dead so it does not bounce.
Frictionless = {"staticFriction": 0.0, "dynamicFriction": 0.0, "restitution": 0.0}
SlideSpeed = 0.1  # m/s
StartX = -0.45
LaneZ = (-0.4, 0.0, 0.4, 0.8)


# Every box body is this bar, whose solved bending modes land in the audible band (first near 1.8 kHz in ceramic at reference scale).
# Only what the scene is varying differs between bodies, so the modal model, the geometry, and the sample points are shared and cannot explain any difference you hear.
# A body's mass picks its node scale on the shared model: mass = BarMass * scale^3, and the engine retunes frequencies by 1/scale and shape amplitudes by scale^(-3/2) from there.
BarHalf = (0.12, 0.03, 0.01)
SphereR = 0.045  # Bodies whose scene needs a point contact stay spheres at this radius.
GroundHalf = (0.6, 0.04, 0.12)  # A lane long enough for the whole skid.
# The lane a body that keeps its speed for the whole render needs: a rolling sphere is not slowed by friction, and a decelerating one has to start fast.
LongLane = {"lane_half": (4.0, 0.04, 0.12), "start_x": -3.6}


def box(hx, hy, hz):
    """Positions, normals, texture coordinates, and triangle indices for a box, with flat-shaded faces.

    Each face carries the unit square, so a mesoscale relief map tiles once per face and its lengths in
    mesh coordinates are the face's own."""
    faces = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
             ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
             ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))]
    h = (hx, hy, hz)
    pos, nrm, uvs, idx = [], [], [], []
    for n, u, v in faces:
        base = len(pos)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            pos.append(tuple(n[i] * h[i] + su * u[i] * h[i] + sv * v[i] * h[i] for i in range(3)))
            nrm.append(n)
            uvs.append((0.5 * (su + 1), 0.5 * (sv + 1)))
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return pos, nrm, uvs, idx


def sphere(r, segments=24, rings=12):
    """Positions, normals, and triangle indices for a closed UV sphere, smooth shaded.

    Each pole is a single vertex and the seam column appears once, so every position is unique and
    every edge carries two faces."""
    def unit(theta, phi):
        st = math.sin(theta)
        return (st * math.cos(phi), math.cos(theta), st * math.sin(phi))

    # Poles first, then one block of `segments` vertices per interior ring.
    pos, nrm = [], []
    def add(n):
        nrm.append(n)
        pos.append((r * n[0], r * n[1], r * n[2]))
    add((0.0, 1.0, 0.0))
    add((0.0, -1.0, 0.0))
    for i in range(1, rings):
        for j in range(segments):
            add(unit(math.pi * i / rings, 2 * math.pi * j / segments))
    north, south = 0, 1
    def ring(i, j):
        return 2 + (i - 1) * segments + (j % segments)

    idx = []
    for j in range(segments):
        idx += [north, ring(1, j + 1), ring(1, j)]
        idx += [south, ring(rings - 1, j), ring(rings - 1, j + 1)]
    for i in range(1, rings - 1):
        for j in range(segments):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            idx += [a, b, c, b, d, c]
    # Poles take the seam's coordinate, which is enough for a relief map the sphere never carries.
    uvs = [(0.5 * (1 + n[0]), 0.5 * (1 + n[2])) for n in nrm]
    return pos, nrm, uvs, idx


def normal_map_png(size=64, bumps=4, slope=0.25):
    """A tangent-space normal map of a sinusoidal egg-crate, as a PNG data URI.

    The height is slope/(2*pi*bumps) times sin(2*pi*bumps*u)*sin(2*pi*bumps*v) over the unit square, so
    `slope` is the peak surface gradient the map describes."""
    pixels = []
    for y in range(size):
        for x in range(size):
            u, v = (x + 0.5) / size, (y + 0.5) / size
            du = slope * math.cos(2 * math.pi * bumps * u) * math.sin(2 * math.pi * bumps * v)
            dv = slope * math.sin(2 * math.pi * bumps * u) * math.cos(2 * math.pi * bumps * v)
            n = (-du, -dv, 1.0)
            length = math.sqrt(sum(c * c for c in n))
            pixels.append(tuple(int(round(255 * (0.5 + 0.5 * c / length))) for c in n))
    raw = b"".join(b"\x00" + bytes(c for p in pixels[row * size:(row + 1) * size] for c in p) for row in range(size))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _fft(values):
    """Iterative radix-2 Cooley-Tukey over python complex, for the profile synthesis alone."""
    n = len(values)
    out = list(values)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            out[i], out[j] = out[j], out[i]
    length = 2
    while length <= n:
        step = cmath.exp(-2j * math.pi / length)
        for start in range(0, n, length):
            w = 1 + 0j
            for k in range(start, start + length // 2):
                u, v = out[k], out[k + length // 2] * w
                out[k], out[k + length // 2] = u + v, u - v
                w *= step
        length <<= 1
    return out


def measured_profile(count=65536, correlation=5e-5, slope=-2.4, roughness=2e-6, seed=12345):
    """A stored height track, in meters, standing in for a profilometer trace of the machined finish.

    Drawn from the same spectrum the synthesized track carries, flat below the corner the correlation
    length sets and falling as q^slope above it, at the same lc/64 spacing, with independent phases.
    So a scene shipping this shares the statistics of one synthesized from the parameters and not the
    realization. Deterministic, so every renderer reading this file reads exactly the same surface,
    which is the whole point of shipping a profile rather than three statistics."""
    spacing = correlation / 64
    q0 = 1.0 / correlation
    dq = 1.0 / (count * spacing)
    state = seed
    spectrum = [0j] * count
    for i in range(1, count // 2 + 1):
        state = (1103515245 * state + 12345) % (1 << 31)
        phase = 2 * math.pi * state / (1 << 31)
        q = i * dq
        amplitude = (q / q0) ** (slope * 0.5) if q > q0 else 1.0
        bin = amplitude * cmath.exp(1j * phase)
        # The top bin is its own mirror, so it must be real for the heights to be.
        spectrum[i] = complex(bin.real, 0.0) if i == count // 2 else bin
        if i != count // 2:
            spectrum[count - i] = bin.conjugate()
    heights = [v.real for v in _fft([s.conjugate() for s in spectrum])]
    mean = sum(heights) / count
    heights = [h - mean for h in heights]
    rms = math.sqrt(sum(h * h for h in heights) / count)
    return [roughness * h / rms for h in heights], spacing


class GridSurface:
    """A surface mesh gathered off a grid, welding each grid corner the first time a face names it.

    `position` places a corner (i, j, k), and quads are emitted as the triangle pair their corner
    order winds. A face's corners run outward, toward the missing neighbour on its far side."""

    def __init__(self, position):
        self.position = position
        self.verts, self.ids, self.tris = [], {}, []

    def v(self, i, j, k):
        key = (i, j, k)
        if key not in self.ids:
            self.ids[key] = len(self.verts)
            self.verts.append(self.position(i, j, k))
        return self.ids[key]

    def quad(self, a, b, c, d):
        self.tris.append((a, b, c))
        self.tris.append((a, c, d))


def _grid_box(hx, hy, hz, spacing):
    """A welded box surface mesh with each face gridded near `spacing`, as (vertices, triangles).

    The solver samples its excitation positions at these vertices, so the grid pitch decides how densely
    the surface carries sample points for contact blending and the radiating-area quadrature."""
    nx, ny, nz = (max(1, round(2 * h / spacing)) for h in (hx, hy, hz))
    s = GridSurface(lambda i, j, k: (2 * hx * i / nx - hx, 2 * hy * j / ny - hy, 2 * hz * k / nz - hz))
    v, quad = s.v, s.quad
    for i in range(nx):
        for j in range(ny):
            quad(v(i, j, 0), v(i, j + 1, 0), v(i + 1, j + 1, 0), v(i + 1, j, 0))
            quad(v(i, j, nz), v(i + 1, j, nz), v(i + 1, j + 1, nz), v(i, j + 1, nz))
    for i in range(nx):
        for k in range(nz):
            quad(v(i, 0, k), v(i + 1, 0, k), v(i + 1, 0, k + 1), v(i, 0, k + 1))
            quad(v(i, ny, k), v(i, ny, k + 1), v(i + 1, ny, k + 1), v(i + 1, ny, k))
    for j in range(ny):
        for k in range(nz):
            quad(v(0, j, k), v(0, j, k + 1), v(0, j + 1, k + 1), v(0, j + 1, k))
            quad(v(nx, j, k), v(nx, j + 1, k), v(nx, j + 1, k + 1), v(nx, j, k + 1))
    return s.verts, s.tris


_solve_cache = {}


def _solve_obj(obj_path, material, modes, min_freq, max_freq):
    tool = os.environ.get("MODAL_SOLVE", os.path.join(os.path.dirname(__file__), "..", "..", "build", "tests", "MeshEditorModalSolve"))
    args = [tool, obj_path,
            "--density", str(material["density"]), "--young", str(material["youngsModulus"]),
            "--poisson", str(material["poissonRatio"]), "--alpha", str(material["alpha"]), "--beta", str(material["beta"]),
            "--min-freq", str(min_freq), "--max-freq", str(max_freq), "--modes", str(modes)]
    # The solve runs single-threaded so its output is bit-identical run to run.
    # Threaded Accelerate varies summation order, and near-degenerate mode pairs rotate under that noise (shapes moved up to 1.3% between runs at bit-identical frequencies), which broke the harnesses' bit-reproducibility across invocations.
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS="1")
    out = subprocess.run(args, capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"modal solve failed for {obj_path}: {out.stderr.strip()}")
    return json.loads(out.stdout)


def _material_key(material):
    return (material["density"], material["youngsModulus"], material["poissonRatio"], material["alpha"], material["beta"])


def _cached_solve(key, material, modes, min_freq, max_freq, build):
    """Solves the mesh `build` returns as (vertices, zero-based triangles), memoized on `key`.

    The build runs only on a miss, so a repeated configuration costs neither a mesh nor a solve."""
    if key in _solve_cache:
        return _solve_cache[key]
    verts, tris = build()
    with tempfile.NamedTemporaryFile(suffix=".obj", mode="w", delete=False) as f:
        for p in verts:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
        for a, b, c in tris:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")
        obj_path = f.name
    result = _solve_obj(obj_path, material, modes, min_freq, max_freq)
    os.unlink(obj_path)
    _solve_cache[key] = result
    return result


def solved_modes(half, material, *, spacing=0.02, modes=30, min_freq=20.0, max_freq=16000.0):
    """Modes of a box body from the real solver chain, as the dict the model embed reads.

    `half` is the box's half extents in meters and `material` is one of the scene material dicts
    (density, youngsModulus, poissonRatio, alpha, beta, in SI). The solver binary is found through
    MODAL_SOLVE, defaulting to the build tree's MeshEditorModalSolve. Solves are cached by their
    full configuration."""
    key = ("box", tuple(half), *_material_key(material), spacing, modes, min_freq, max_freq)
    return _cached_solve(key, material, modes, min_freq, max_freq, lambda: _grid_box(*half, spacing))


def solved_modes_sphere(r, material, *, modes=30, min_freq=20.0, max_freq=16000.0):
    """Modes of a sphere body from the real solver chain, over the same mesh the scene renders."""

    def build():
        positions, _, _, indices = sphere(r)
        return positions, [tuple(indices[t:t + 3]) for t in range(0, len(indices), 3)]

    key = ("sphere", r, *_material_key(material), modes, min_freq, max_freq)
    return _cached_solve(key, material, modes, min_freq, max_freq, build)




class Buffers:
    """Accumulates float and index data into one embedded buffer."""

    def __init__(self):
        self.blob = bytearray()
        self.views = []
        self.accessors = []

    def _view(self, data: bytes):
        while len(self.blob) % 4:
            self.blob.append(0)
        self.views.append({"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)})
        self.blob += data
        return len(self.views) - 1

    def _accessor(self, data, component_type, count, kind, columns):
        """One accessor over `data`, its bounds taken per component from `columns`."""
        self.accessors.append({"bufferView": self._view(data), "componentType": component_type, "count": count,
                               "type": kind, "min": [min(c) for c in columns], "max": [max(c) for c in columns]})
        return len(self.accessors) - 1

    def scalars(self, values):
        return self._accessor(struct.pack(f"<{len(values)}f", *values), 5126, len(values), "SCALAR", [values])

    def ushorts(self, values):
        return self._accessor(struct.pack(f"<{len(values)}H", *values), 5123, len(values), "SCALAR", [values])

    def vecs(self, values):
        """Vectors of as many components as the first one carries."""
        n = len(values[0])
        flat = [c for v in values for c in v]
        return self._accessor(struct.pack(f"<{len(flat)}f", *flat), 5126, len(values), f"VEC{n}",
                              [[v[i] for v in values] for i in range(n)])

    def uri(self):
        return "data:application/octet-stream;base64," + base64.b64encode(bytes(self.blob)).decode()


def body(label, mass, lane, surface, shape="box", *, physics_material=0, rolls=False, drop=0.0,
         speed=None, driven=False, drive_damping=1000.0, on=None, silent=False):
    """One dynamic body. Its mass picks its size on the scene's solved model, mass = solved mass * s^3.
    `rolls` spins it to match its travel, so its contact slips not at all.
    `drop` is how far it falls before touching, which sets the impulse of the impact that starts the scene.
    A body with no drop is authored resting on its surface and starts in silence.
    `on` is the index of the body this one stands on rather than the ground, which is what strikes a
    body that is already seated: the tap presses it into its support instead of landing it.
    `silent` gives the body its finish and no modal model, which the extension allows, so a striker
    carries its contact into the sound without adding a ring of its own.
    `speed` overrides the scene's, so one scene can hold a ladder of sliding speeds.
    `driven` holds the body's speed against friction with a velocity motor jointed to its lane, and locks
    its rotation, so a sphere can keep sliding where friction would otherwise spin it up into rolling.
    `drive_damping` is the motor's velocity gain. Friction droops the held speed by force over gain,
    so a drive at millimeters per second takes a far stiffer hold than the default."""
    return {"label": label, "mass": mass, "lane": lane, "surface": surface, "shape": shape,
            "physics_material": physics_material, "rolls": rolls, "drop": drop, "speed": speed,
            "driven": driven, "drive_damping": drive_damping, "on": on, "silent": silent}


def ground(label, surface, lane, *, probe=None):
    """`probe` gives the ground its own modal model, so it radiates as well as being scraped."""
    return {"label": label, "surface": surface, "lane": lane, "probe": probe}


def box_shape(half):
    return {"type": "box", "box": {"size": [2 * h for h in half]}}


def sphere_shape(radius):
    return {"type": "sphere", "sphere": {"radius": radius}}


def finish_ground(finish, lane=1, *, probe=None):
    """The ground of a scene pairing one of the preset finishes against itself."""
    return ground(finish, FinishIndex[finish], lane, probe=probe)


def _modal_model(buf, label, solved, material=0):
    """One modal model entry, with its arrays landed in `buf`."""
    return {
        "name": label,
        "frequencies": buf.scalars(solved["frequencies"]),
        "decayRates": buf.scalars(solved["decayRates"]),
        "positions": buf.vecs(solved["positions"]),
        "shapes": buf.vecs(solved["shapes"]),
        "indices": buf.ushorts(solved["indices"]),
        "material": material,
        # The mass distribution the model was solved at.
        # A body of another mass sizes itself on the model as a scale change, mass = solved mass * scale^3.
        "massProperties": {
            "mass": solved["mass"],
            "centerOfMass": solved["centerOfMass"],
            "inertiaDiagonal": solved["inertiaDiagonal"],
        },
    }


def _add_mesh(buf, meshes, pos, nrm, uvs, idx):
    """Appends one primitive to `meshes`, with its attributes landed in `buf`."""
    attributes = {"POSITION": buf.vecs(pos), "NORMAL": buf.vecs(nrm), "TEXCOORD_0": buf.vecs(uvs)}
    meshes.append({"primitives": [{"attributes": attributes, "indices": buf.ushorts(idx)}]})
    return len(meshes) - 1


def _write_doc(name, description, buf, *, nodes, scene_nodes, meshes, shapes, physics_materials,
               acoustic_materials, surfaces, models, physics_joints=(), relief_surfaces=()):
    """Assembles the document those parts make, finalizes `buf`, and writes the scene.

    `relief_surfaces` names the surface indices carrying the shared relief map."""
    # A stored profile is an accessor, so it lands in the same buffer as everything else.
    for surface in surfaces:
        if "profileHeights" in surface:
            surface["profile"] = buf.scalars(surface.pop("profileHeights"))
    # The buffer and its views are only final once every accessor above has been added.
    doc = {
        "asset": {"version": "2.0", "generator": "KHR_audio_rigid_bodies samples", "copyright": description},
        "extensionsUsed": ["KHR_implicit_shapes", "KHR_physics_rigid_bodies", "KHR_audio_rigid_bodies"],
        "scene": 0,
        "scenes": [{"name": name, "nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": meshes,
        "buffers": [{"byteLength": len(buf.blob), "uri": buf.uri()}],
        "bufferViews": buf.views,
        "accessors": buf.accessors,
        "extensions": {
            "KHR_implicit_shapes": {"shapes": shapes},
            "KHR_physics_rigid_bodies": {"physicsMaterials": physics_materials} | ({"physicsJoints": physics_joints} if physics_joints else {}),
            "KHR_audio_rigid_bodies": {
                "acousticMaterials": acoustic_materials,
                "acousticSurfaces": surfaces,
                "modalModels": models,
            },
        },
    }
    if relief_surfaces:
        doc["images"] = [{"uri": normal_map_png(), "mimeType": "image/png"}]
        doc["samplers"] = [{"wrapS": 10497, "wrapT": 10497}]
        doc["textures"] = [{"source": 0, "sampler": 0}]
        for i in relief_surfaces:
            surfaces[i]["normalTexture"] = {"index": 0, "texCoord": 0}
    path = OUT / f"{name}.gltf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{name}.gltf: {len(nodes)} nodes, {len(buf.blob)} bytes of buffer")


def scene(name, description, bodies, grounds, *, lane_half=None, start_x=None, speed=None,
          probe=None, probe_material=0, physics_materials=None, surfaces=None, relief_surfaces=(),
          bar_half=None, acoustic_materials=None):
    """Each scene owns its lane, probe, materials, and surfaces, so a default that suits one cannot
    silently constrain another. `relief_surfaces` names surface indices that carry the shared relief map.
    `bar_half` is the box bodies' half extents, defaulting to the corpus bar.
    `acoustic_materials` overrides the scene's material table, and `probe_material` indexes the body
    models' material in it, so a probe solved on another material can reference that material."""
    if not selected(name):
        return
    lane_half = lane_half or GroundHalf
    bar_half = bar_half or BarHalf
    start_x = StartX if start_x is None else start_x
    speed = SlideSpeed if speed is None else speed
    physics_materials = physics_materials or [Frictionless]
    buf = Buffers()
    body_shapes = list(dict.fromkeys(b["shape"] for b in bodies))
    # One solved model per body shape the scene uses.
    # `probe` overrides them all when a scene needs a solve of its own (another material or frequency window).
    # A ground carrying its own solve appends one and refers to it by index.
    body_solved = {g: probe or (solved_modes(bar_half, CERAMIC) if g == "box" else solved_modes_sphere(SphereR, CERAMIC, max_freq=60000.0)) for g in body_shapes}
    models, body_model = [], {}
    for g in body_shapes:
        body_model[g] = len(models)
        models.append(_modal_model(buf, f"Solved {g}", body_solved[g], probe_material))
    ground_model = {}
    for i, g in enumerate(grounds):
        if g["probe"] is not None:
            models.append(_modal_model(buf, f"Ground {g['label']}", g["probe"]))
            ground_model[i] = len(models) - 1

    # One mesh per distinct size, so a body's own geometry is never a variable between scenes.
    meshes = []

    def add_mesh(*args):
        return _add_mesh(buf, meshes, *args)

    ground_mesh = add_mesh(*box(*lane_half))

    nodes, scene_nodes = [], []
    shapes_decl = [box_shape(lane_half)]
    # A body's mesh and its collider, one pair per shape and scale the scene actually uses.
    body_mesh, body_shape = {}, {}
    for g in body_shapes:
        if g == "box":
            body_mesh[g] = add_mesh(*box(*bar_half))
            shapes_decl.append(box_shape(bar_half))
        else:
            body_mesh[g] = add_mesh(*sphere(SphereR))
            shapes_decl.append(sphere_shape(SphereR))
        body_shape[g] = len(shapes_decl) - 1
    for i, g in enumerate(grounds):
        audio = {"acousticSurface": g["surface"]}
        if i in ground_model:
            audio |= {"modalModel": ground_model[i], "gain": 1.0}
        scene_nodes.append(len(nodes))
        nodes.append({
            "name": f"Ground {g['label']}",
            "mesh": ground_mesh,
            "translation": [0.0, -lane_half[1], LaneZ[g["lane"]]],
            "extensions": {
                "KHR_physics_rigid_bodies": {"collider": {"geometry": {"shape": 0}, "physicsMaterial": 0}},
                "KHR_audio_rigid_bodies": audio,
            },
        })
    # Every body's own half height, so one standing on another is placed on top of it.
    body_half = [(bar_half[1] if b["shape"] == "box" else SphereR) * (b["mass"] / body_solved[b["shape"]]["mass"]) ** (1.0 / 3.0)
                 for b in bodies]
    physics_joints = []
    for b_index, b in enumerate(bodies):
        geom, mass = b["shape"], b["mass"]
        solved = body_solved[geom]
        # The body's mass picks its size on the solved model, mass = solved mass * s^3, so the engine's scale retuning keeps the modes, the shapes, and the rigid dynamics of one body of this mass.
        s = (mass / solved["mass"]) ** (1.0 / 3.0)
        inertia = [i * (mass / solved["mass"]) ** (5.0 / 3.0) for i in solved["inertiaDiagonal"]]
        y_half = body_half[b_index]
        # A body standing on another rests on its top face, and its drop is measured from there.
        stand = 2 * body_half[b["on"]] if b["on"] is not None else 0.0
        body_speed = speed if b["speed"] is None else b["speed"]
        # A body that rolls carries the spin its travel implies, so the contact point stands still.
        motion = {"mass": mass, "inertiaDiagonal": inertia, "linearVelocity": [body_speed, 0.0, 0.0]}
        if b["rolls"]:
            motion["angularVelocity"] = [0.0, 0.0, -body_speed / y_half]
        rigid_body = {
            "motion": motion,
            "collider": {"geometry": {"shape": body_shape[geom]}, "physicsMaterial": b["physics_material"]},
        }
        # A driven body is jointed to its lane: a velocity motor along the travel holds its speed against friction, its rotation is locked, and every other axis stays free so it follows the contact.
        # The drive targets the connected node's velocity in joint space per the KHR spec, so holding this body at +v against its lane takes a target of -v.
        if b["driven"]:
            physics_joints.append({
                "limits": [{"angularAxes": [0, 1, 2], "min": 0.0, "max": 0.0}],
                "drives": [{"type": "linear", "mode": "force", "axis": 0, "maxForce": 100.0,
                            "velocityTarget": -body_speed, "damping": b["drive_damping"]}],
            })
            lane_ground_node = next(i for i, g in enumerate(grounds) if g["lane"] == b["lane"])
            rigid_body["joint"] = {"connectedNode": lane_ground_node, "joint": len(physics_joints) - 1,
                                   "enableCollision": True}
        node = {
            "name": b["label"],
            "mesh": body_mesh[geom],
            "translation": [start_x, stand + y_half + b["drop"], LaneZ[b["lane"]]],
            "extensions": {
                "KHR_physics_rigid_bodies": rigid_body,
                # A silent body carries its finish and no modal model, which the extension allows.
                "KHR_audio_rigid_bodies": ({} if b["silent"] else {"modalModel": body_model[geom]})
                    | {"acousticSurface": b["surface"], "gain": 1.0},
            },
        }
        if s != 1.0:
            node["scale"] = [s, s, s]
        scene_nodes.append(len(nodes))
        nodes.append(node)

    _write_doc(
        name, description, buf, nodes=nodes, scene_nodes=scene_nodes, meshes=meshes, shapes=shapes_decl,
        physics_materials=physics_materials, physics_joints=physics_joints,
        acoustic_materials=acoustic_materials or ACOUSTIC_MATERIALS,
        surfaces=list(surfaces or _surfaces()), models=models, relief_surfaces=relief_surfaces
    )


def test_scene(name, *args, **kwargs):
    """One phenomenon scene of the isolation corpus, under test/.

    The root of samples/ holds the full-experience scenes, and test/ holds the ladder corpus whose
    scenes each isolate one prediction."""
    scene(f"test/{name}", *args, **kwargs)


def surface_scene(name, *args, **kwargs):
    """One scene of the isolation corpus whose prediction is a sustained contact's, under test/surface/.

    These sound only where the surface-contact model renders them, so they are kept apart from the
    scenes a collision alone voices."""
    test_scene(f"surface/{name}", *args, **kwargs)


def union_surface(boxes, spacing):
    """Welded watertight surface mesh of a union of axis-aligned boxes, gridded near `spacing`.

    `boxes` is a list of (center, half) pairs. Grid cells count as solid where their centre falls in
    any box, and every solid cell face against the outside is emitted, so any union closes."""
    lo = [min(c[i] - h[i] for c, h in boxes) for i in range(3)]
    hi = [max(c[i] + h[i] for c, h in boxes) for i in range(3)]
    counts = [max(1, round((hi[i] - lo[i]) / spacing)) for i in range(3)]
    step = [(hi[i] - lo[i]) / counts[i] for i in range(3)]

    def solid(i, j, k):
        if not (0 <= i < counts[0] and 0 <= j < counts[1] and 0 <= k < counts[2]):
            return False
        p = [lo[a] + ([i, j, k][a] + 0.5) * step[a] for a in range(3)]
        return any(all(abs(p[a] - c[a]) <= h[a] + 1e-12 for a in range(3)) for c, h in boxes)

    s = GridSurface(lambda i, j, k: (lo[0] + i * step[0], lo[1] + j * step[1], lo[2] + k * step[2]))
    v, quad = s.v, s.quad
    for i in range(counts[0]):
        for j in range(counts[1]):
            for k in range(counts[2]):
                if not solid(i, j, k):
                    continue
                if not solid(i - 1, j, k):
                    quad(v(i, j, k), v(i, j, k + 1), v(i, j + 1, k + 1), v(i, j + 1, k))
                if not solid(i + 1, j, k):
                    quad(v(i + 1, j, k), v(i + 1, j + 1, k), v(i + 1, j + 1, k + 1), v(i + 1, j, k + 1))
                if not solid(i, j - 1, k):
                    quad(v(i, j, k), v(i + 1, j, k), v(i + 1, j, k + 1), v(i, j, k + 1))
                if not solid(i, j + 1, k):
                    quad(v(i, j + 1, k), v(i, j + 1, k + 1), v(i + 1, j + 1, k + 1), v(i + 1, j + 1, k))
                if not solid(i, j, k - 1):
                    quad(v(i, j, k), v(i, j + 1, k), v(i + 1, j + 1, k), v(i + 1, j, k))
                if not solid(i, j, k + 1):
                    quad(v(i, j, k + 1), v(i + 1, j, k + 1), v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1))
    return s.verts, s.tris


def flat_mesh(verts, tris):
    """Flat-shaded positions, normals, texture coordinates, and indices from a welded surface.

    Each face's texture coordinates are its positions in its own plane, so a relief map keeps mesh
    lengths, as the box builder's faces do."""
    pos, nrm, uvs, idx = [], [], [], []
    for a, b, c in tris:
        pa, pb, pc = verts[a], verts[b], verts[c]
        u = [pb[i] - pa[i] for i in range(3)]
        w = [pc[i] - pa[i] for i in range(3)]
        n = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
        length = math.sqrt(sum(x * x for x in n)) or 1.0
        n = tuple(x / length for x in n)
        axis = max(range(3), key=lambda i: abs(n[i]))
        ua, va = [i for i in range(3) if i != axis]
        for p in (pa, pb, pc):
            pos.append(p)
            nrm.append(n)
            uvs.append((p[ua], p[va]))
        idx += [len(pos) - 3, len(pos) - 2, len(pos) - 1]
    return pos, nrm, uvs, idx


def solved_modes_union(boxes, material, *, spacing=0.005, modes=30, min_freq=20.0, max_freq=16000.0):
    """Modes of a union-of-boxes body from the real solver chain, over the same mesh the scene renders."""
    key = ("union", tuple((tuple(c), tuple(h)) for c, h in boxes), *_material_key(material), spacing, modes, min_freq, max_freq)
    return _cached_solve(key, material, modes, min_freq, max_freq, lambda: union_surface(boxes, spacing))


def _axis_quat(axis, degrees):
    half = math.radians(degrees) / 2
    s = math.sin(half)
    return tuple(a * s for a in axis) + (math.cos(half),)


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


# The bracket body: a steel angle of two boxes, sharing a corner, so its surface is concave.
BracketBoxes = (((0.0, 0.01, 0.0), (0.05, 0.01, 0.02)), ((-0.04, 0.035, 0.0), (0.01, 0.035, 0.02)))


def pile():
    """The full-experience scene: five bodies of stepped geometric complexity dropped together onto
    one platform, with every major audio mechanism sounding at once.

    Unlike the test corpus this isolates nothing: it is the qualitative listening scene and the
    performance target, cluttered and rich, with natural collision and surface-settling noise.
    Ceramic, steel, and glass bodies from a plain cube up to a concave bracket land under gravity
    with friction and restitution, on a radiating cast platform carrying the relief map, one body
    reading a stored profile, and every body carrying its own solved modal model."""
    if not selected("Pile"):
        return
    buf = Buffers()
    profile_heights, profile_spacing = measured_profile()
    surfaces = [
        {"name": "Polished steel", "material": 1, **FINISHES["Polished"]},
        {"name": "Polished glass", "material": 2, **FINISHES["Polished"]},
        {"name": "Machined ceramic", "material": 0, **FINISHES["Machined"]},
        {"name": "Cast steel", "material": 1, **FINISHES["Cast"]},
        {"name": "Measured machined", "material": 0, "profileHeights": profile_heights,
         "sampleSpacing": profile_spacing, **FINISHES["Machined"]},
        {"name": "Cast ground", "material": 0, **FINISHES["Cast"]},
    ]
    platform_half = (0.3, 0.03, 0.3)
    cube_half = (0.04, 0.04, 0.04)
    slab_half = (0.07, 0.012, 0.05)
    bracket_verts, bracket_tris = union_surface(BracketBoxes, 0.005)

    def modal_model(label, solved, material):
        return _modal_model(buf, label, solved, material)

    meshes = []

    def add_mesh(*args):
        return _add_mesh(buf, meshes, *args)

    # Every body at its own solved geometry and mass, so nothing is a scaled stand-in.
    # Compact bodies ring above hearing, so their windows run ultrasonic and the engine's own audibility gate decides what sounds: a small hard cube clicks through what it strikes.
    # (label, solve, material index, surface index, mesh, implicit shape or None for the bracket,
    #  clear radius about the origin, drop, x, z, rotation)
    bodies = [
        ("Cube", solved_modes(cube_half, CERAMIC, spacing=0.01, max_freq=60000.0), 0, 2,
         add_mesh(*box(*cube_half)), box_shape(cube_half),
         math.sqrt(3) * cube_half[0], 0.25, 0.06, 0.05, _axis_quat((0, 1, 0), 20)),
        ("Bar", solved_modes(BarHalf, STEEL), 1, 0,
         add_mesh(*box(*BarHalf)), box_shape(BarHalf),
         math.sqrt(sum(h * h for h in BarHalf)), 0.35, -0.04, -0.03,
         _quat_mul(_axis_quat((0, 1, 0), -35), _axis_quat((0, 0, 1), 8))),
        ("Marble", solved_modes_sphere(SphereR, GLASS, max_freq=60000.0), 2, 1,
         add_mesh(*sphere(SphereR)), sphere_shape(SphereR),
         SphereR, 0.5, 0.12, -0.09, None),
        ("Slab", solved_modes(slab_half, CERAMIC, spacing=0.012, max_freq=60000.0), 0, 4,
         add_mesh(*box(*slab_half)), box_shape(slab_half),
         math.sqrt(sum(h * h for h in slab_half)), 0.18, -0.09, 0.07,
         _quat_mul(_axis_quat((0, 1, 0), 55), _axis_quat((1, 0, 0), 6))),
        ("Bracket", solved_modes_union(BracketBoxes, STEEL, max_freq=60000.0), 1, 3,
         add_mesh(*flat_mesh(bracket_verts, bracket_tris)), None,
         0.09, 0.75, 0.0, 0.0, _quat_mul(_axis_quat((0, 1, 0), 100), _axis_quat((0, 0, 1), 12))),
    ]
    platform_mesh = add_mesh(*box(*platform_half))
    platform_solved = solved_modes(platform_half, CERAMIC, spacing=0.05)

    models = [modal_model(label, solved, material) for label, solved, material, *_ in bodies]
    models.append(modal_model("Platform", platform_solved, 0))

    shapes_decl = [box_shape(platform_half)]
    nodes, scene_nodes = [], []
    scene_nodes.append(len(nodes))
    nodes.append({
        "name": "Platform",
        "mesh": platform_mesh,
        "translation": [0.0, -platform_half[1], 0.0],
        "extensions": {
            "KHR_physics_rigid_bodies": {"collider": {"geometry": {"shape": 0}, "physicsMaterial": 0}},
            "KHR_audio_rigid_bodies": {"acousticSurface": 5, "modalModel": len(models) - 1, "gain": 1.0},
        },
    })
    for index, (label, solved, material, surface, mesh, shape, clear, drop, x, z, rotation) in enumerate(bodies):
        motion = {"mass": solved["mass"], "inertiaDiagonal": solved["inertiaDiagonal"]}
        rigid_body = {"motion": motion}
        node = {
            "name": label,
            "mesh": mesh,
            "translation": [x, drop + clear, z],
            "extensions": {
                "KHR_physics_rigid_bodies": rigid_body,
                "KHR_audio_rigid_bodies": {"modalModel": index, "acousticSurface": surface, "gain": 1.0},
            },
        }
        if rotation is not None:
            node["rotation"] = list(rotation)
        if shape is not None:
            shapes_decl.append(shape)
            rigid_body["collider"] = {"geometry": {"shape": len(shapes_decl) - 1}, "physicsMaterial": 0}
        else:
            # The bracket's contact geometry is the union its surface is: one box collider per arm on child nodes, which the engine composes into one compound body.
            children = []
            for center, half in BracketBoxes:
                shapes_decl.append(box_shape(half))
                children.append(len(nodes))
                nodes.append({
                    "name": f"{label} arm {len(children)}",
                    "translation": list(center),
                    "extensions": {
                        "KHR_physics_rigid_bodies": {
                            "collider": {"geometry": {"shape": len(shapes_decl) - 1}, "physicsMaterial": 0}
                        },
                    },
                })
            node["children"] = children
        scene_nodes.append(len(nodes))
        nodes.append(node)

    # The cast ground carries the relief map.
    _write_doc(
        "Pile",
        "Five bodies, a cube through a concave bracket, dropped together into a pile on a radiating platform. Every major audio mechanism sounds at once.",
        buf, nodes=nodes, scene_nodes=scene_nodes, meshes=meshes, shapes=shapes_decl,
        physics_materials=[{"staticFriction": 0.5, "dynamicFriction": 0.4, "restitution": 0.3}],
        acoustic_materials=[CERAMIC, STEEL, GLASS], surfaces=surfaces, models=models, relief_surfaces=(5,)
    )


def _surfaces():
    return [{"name": k, "material": 0, **v} for k, v in FINISHES.items()]


def main():
    pile()

    # Each phenomenon is a directory of sibling scenes, one configuration sounding per scene, named in a/b/c ladder order.
    # Siblings put their body and ground on the centre lane and share their lane geometry, so every one frames identically and their levels compare across files.

    # Every body oscillates on its contact at sqrt(K/m), and K is proportional to the load, so the rate is the same for all three however much they weigh.
    # They should settle onto one shared pitch.
    for tag, label, mass in (("a_Light5g", "Light 5 g", 0.005),
                             ("b_Middling500g", "Middling 500 g", 0.5),
                             ("c_Heavy50kg", "Heavy 50 kg", 50.0)):
        surface_scene(
            f"ContactPitch/{tag}",
            "One of three identical bodies of very different mass on one finish. The rate each rides its contact at does not depend on its mass.",
            bodies=[body(label, mass, 1, FinishIndex["Machined"])],
            grounds=[finish_ground("Machined")],
        )

    # A contact excites both solids that make it.
    # Every other scene leaves its ground silent, so this is the one where the scraped surface rings back, in a band of its own so it can be heard doing it.
    surface_scene(
        "SurfaceRadiates/a_Scrape",
        "A body scraping a surface that rings back. Both solids of a contact radiate.",
        # 2 kg keeps the scraper bearing on the bed (~11% flight) rather than skipping off it, which is the difference between a grainy scrape and sparse plucks.
        bodies=[body("Scraper", 2.0, 1, FinishIndex["Machined"])],
        grounds=[finish_ground("Machined", probe=solved_modes(GroundHalf, CERAMIC, spacing=0.04))],
    )

    # The same body on three finishes.
    # The rate goes as one over the square root of the roughness, so these should sit roughly five octaves apart, roughest lowest.
    for tag, finish in (("a_Polished", "Polished"), ("b_Machined", "Machined"), ("c_Cast", "Cast")):
        surface_scene(
            f"FinishLadder/{tag}",
            "One body on one of three finishes. The rate it rides its contact at goes as one over the square root of the roughness.",
            bodies=[body(f"On {finish}", 2.0, 1, FinishIndex[finish])],
            grounds=[finish_ground(finish)],
        )

    # Asperity shocks arrive at the sliding speed over the correlation length, and that rate is what decides whether contact is heard as separate events or as hiss.
    # Sliding speed cannot reach the band where the difference is audible: on a machined finish even a crawl arrives at hundreds of events a second, and bodies overlap them faster than the modes decay.
    # The surface's own length scale can, so these three lanes hold the sliding speed and step the correlation length instead, putting the rate at 100, 500 and 2000 events a second.
    # Roughness is scaled with the correlation length so every lane carries the same surface gradient, which is what sets the asperity pressure and so the real contact area.
    # Only the length scale differs, so the three lanes differ in the rate their shocks arrive at and in nothing else.
    for tag, label, lc in (("a_100PerSecond", "100 shocks/s", 1e-3),
                           ("b_500PerSecond", "500 shocks/s", 2e-4),
                           ("c_2000PerSecond", "2000 shocks/s", 5e-5)):
        surface_scene(
            f"ShockRate/{tag}",
            "One body at one speed over one of three finishes of the same gradient and different length scale. Asperity shocks arrive at the speed over the correlation length, so the coarsest scene is heard as separate events and the finest as hiss.",
            bodies=[body(label, 2.0, 1, 0, drop=0.0)],
            grounds=[ground(label, 0, 1)],
            # Scaled with the finish so every scene keeps one waviness-to-roughness ratio too.
            surfaces=[{"name": label, "material": 0, "roughness": 0.04 * lc,
                       "correlationLength": lc, "spectralSlope": -2.4,
                       "waviness": 0.04 * lc, "wavinessLength": 40 * lc}],
            lane_half=(2.5, 0.04, 0.12), start_x=-2.3,
        )

    # Coulomb traction acts on the same force fluctuation the normal direction carries, scaled by the friction coefficient and applied along the slip.
    # It is the one term of the contact force no other scene can catch the absence of, since every other lane in the corpus is frictionless so that its sliding speed holds.
    # Here the coefficient is the only thing that differs, and a renderer that leaves traction out renders the second and third lanes as quiet as the first while they slide.
    # Friction also slows a body, so the lanes stop at different times.
    # What discriminates is the level over the window they are all still moving, which the fastest to stop holds for about half a second.
    for tag, label, static, dynamic in (("a_Frictionless", "Frictionless", 0.0, 0.0),
                                        ("b_LightGrip", "Light grip", 0.055, 0.05),
                                        ("c_FullGrip", "Full grip", 0.16, 0.15)):
        surface_scene(
            f"FrictionLadder/{tag}",
            "One body on one finish at one of three friction coefficients. Coulomb traction drives the surface along the slip in proportion to the coefficient, so a frictionless scene and a gripping one differ while both still slide.",
            bodies=[body(label, 2.0, 1, 0, drop=0.0, physics_material=1)],
            grounds=[ground("Coarse", 0, 1)],
            # A coarse finish so the shocks land in the band at the speed a decelerating body needs to start at.
            surfaces=[{"name": "Coarse", "material": 0, "roughness": 4e-5,
                       "correlationLength": 1e-3, "spectralSlope": -2.4,
                       "waviness": 4e-5, "wavinessLength": 4e-2}],
            physics_materials=[Frictionless, {"staticFriction": static, "dynamicFriction": dynamic, "restitution": 0.0}],
            speed=1.0, **LongLane,
        )

    # At millimeters per second the shock rate collapses and the frictional channel carries the scene: the patch shear spring sticks, loads with the travel, and slips in audio-band cycles.
    # Three speeds ladder the slip-event rate.
    # The stiff drive holds the creep against friction, whose droop at the default gain would swamp a millimeter-per-second target.
    for tag, label, creep_speed in (("a_Creep", "Creep", 0.001), ("b_Slow", "Slow", 0.003), ("c_Fast", "Fast", 0.010)):
        surface_scene(
            f"Squeak/{tag}",
            "A gripping body driven at millimeters per second. Friction sticks and slips through the patch's shear compliance, so the scrape squeaks instead of hissing.",
            bodies=[body(label, 2.0, 1, FinishIndex["Machined"], driven=True, speed=creep_speed, physics_material=1, drive_damping=1e6)],
            grounds=[finish_ground("Machined")],
            physics_materials=[Frictionless, {"staticFriction": 0.4, "dynamicFriction": 0.4, "restitution": 0.0}],
            start_x=0.0,
        )

    # Two faces meeting share an area their geometry fixes, while a sphere touches at a point whose patch grows with the load instead.
    # The extension makes that a difference of contact law, not of degree.
    # Each body carries its own solved model, since each is its own geometry, at one mass and one finish.
    for tag, label, shape in (("a_Face", "Box on a face", "box"), ("b_Point", "Sphere on a point", "sphere")):
        surface_scene(
            f"FaceVsPoint/{tag}",
            "A box on its face, or the comparison sphere on its point. The box's contact patch is fixed by its geometry and the sphere's grows with its load.",
            bodies=[body(label, 0.5, 1, FinishIndex["Machined"], shape)],
            grounds=[finish_ground("Machined")],
            **LongLane,
        )

    # A rolling body has no slip at all, and the literature's habit of classifying a contact by its slip routes it out of the surface model entirely and silences it.
    # These two scenes hold one sphere each at one speed over one finish and the same real friction, and the only difference is the spin.
    # Rolling's sphere carries the spin its travel implies and friction keeps it locked there, so its frictional channel is silent by physics rather than by an authored material.
    # Sliding's sphere is motor-held at the same speed with its rotation locked, so friction excites it the whole way.
    # Listen to them one after the other rather than mixed.
    sphere_friction = [{"staticFriction": 0.4, "dynamicFriction": 0.3, "restitution": 0.0}]
    surface_scene(
        "RollingVsSliding/a_Rolling",
        "A sphere rolling without slipping. Rolling has no slip and sounds anyway, because each surface is traversed at its own rate.",
        bodies=[body("Rolling", 0.5, 1, FinishIndex["Machined"], "sphere", rolls=True)],
        grounds=[finish_ground("Machined")],
        physics_materials=sphere_friction,
        **LongLane,
    )
    surface_scene(
        "RollingVsSliding/b_Sliding",
        "The same sphere sliding without spinning, the comparison for Rolling. Friction excites its contact the whole way, and a motor holds its speed.",
        bodies=[body("Sliding", 0.5, 1, FinishIndex["Machined"], "sphere", driven=True)],
        grounds=[finish_ground("Machined")],
        physics_materials=sphere_friction,
        **LongLane,
    )
    # Rolling is the limit where both surfaces are traversed at one rate, and that is the only limit in which a single composite of the pair's roughness swept at a single rate is exact.
    # Reading it off Rolling above does not work: the rate a contact point traverses a sphere is its spin times the radius it stands at, which the load presses inside the sphere's own, so that scene's two rates part by about a tenth of a percent and walk tens of elements of relative slip over a render.
    # This one holds them together instead, by making that depth a smaller share of the radius it is taken from.
    # Friction is what keeps a rolling body rolling, so it keeps the same real friction the two above carry.
    surface_scene(
        "RollingVsSliding/c_Locked",
        "A large sphere rolling, the limit where both surfaces are traversed at one rate. Its two sweeps hold together where a smaller sphere's part.",
        bodies=[body("Locked", 50.0, 1, FinishIndex["Machined"], "sphere", rolls=True)],
        grounds=[finish_ground("Machined")],
        physics_materials=sphere_friction,
        **LongLane,
    )

    # Which side of a pair carries the roughness decides how much of the gap fluctuates, because only a surface the contact travels along brings fresh material under it.
    # A body sliding flat on a ground sweeps the ground's surface and sits at one place on its own face, so the same pair roughness sweeps when it is on the ground and stands still when it is on the body.
    # Every other scene pairs a finish against itself, where the two sides split the fluctuation evenly and the contrast cannot appear.
    for tag, label, sliding_face, swept in (("a_Even", "Even", "Machined", "Machined"),
                                            ("b_SweptRough", "Swept rough", "Polished", "Cast"),
                                            ("c_UnsweptRough", "Unswept rough", "Cast", "Polished")):
        surface_scene(
            f"SlidingPair/{tag}",
            "A body sliding on a ground, with the pair's roughness on the surface the contact sweeps, on the face it never moves across, or split evenly. Only a surface the contact travels along brings fresh material under it.",
            bodies=[body(label, 2.0, 1, FinishIndex[sliding_face])],
            grounds=[finish_ground(swept)],
        )

    # A rolling body on a coarse ground has two regimes in speed: at a crawl its footprint always finds the next supporting crest and it follows the surface quasi-statically, and as the speed rises the terrain launches it faster than gravity can return it, so contact gives way to ballistic hops and bounce trains.
    # The middle rung rolls at the corpus speed, so it doubles as the standing sphere-on-cast probe scene.
    for tag, label, v in (("a_Creep", "Creep", 0.02), ("b_Reference", "Reference", 0.1), ("c_Brisk", "Brisk", 0.7)):
        surface_scene(
            f"BounceOnset/{tag}",
            "A sphere rolling on a coarse casting at one of three speeds. The same ground turns from a quiet ride into bounce trains as the speed rises past what its hills can support.",
            bodies=[body(label, 0.5, 1, FinishIndex["Machined"], "sphere", rolls=True, speed=v)],
            grounds=[finish_ground("Cast")],
            physics_materials=sphere_friction,
            **LongLane,
        )

    # A dropped body's bounces arrive as discrete impacts, so this pair voices the strike channel's surface response: the finish sets each landing's contact time and with it the strike's brightness.
    # The drop and restitution are sized so the bounce train sweeps the heavy landing class from hundreds of millimeters per second down.
    landing_material = [{"staticFriction": 0.4, "dynamicFriction": 0.3, "restitution": 0.5}]
    for tag, label, finish in (("a_Machined", "On machined", "Machined"), ("b_Cast", "On cast", "Cast")):
        surface_scene(
            f"Landing/{tag}",
            "The same sphere dropped onto one of two finishes. The finish sets each landing's contact time, so the two grounds differ in the brightness of every bounce.",
            bodies=[body(label, 0.5, 1, FinishIndex["Machined"], "sphere", drop=0.005, speed=0.0)],
            grounds=[finish_ground(finish)],
            physics_materials=landing_material,
        )

    # Node scale sizes the mesoscale relief, which lives in mesh coordinates, and leaves the microscale finish alone, which is absolute.
    # Three copies of one body at three scales therefore detune together while their relief stretches and their polish does not.
    for tag, label, mass in (("a_HalfSize", "Half size", 0.0972),
                             ("b_Reference", "Reference", 0.7776),
                             ("c_DoubleSize", "Double size", 6.2208)):
        surface_scene(
            f"ReliefAndScale/{tag}",
            "One body at one of three scales on a ground carrying a relief map. Scale sizes the relief and detunes the model, and leaves the absolute finish alone.",
            bodies=[body(label, mass, 1, 1)],
            grounds=[ground("Relief", 0, 1)],
            # Built fresh per scene: the writer annotates these dicts with the relief texture.
            surfaces=[{"name": "Relief", "material": 0, **FINISHES["Machined"]},
                      {"name": "Machined", "material": 0, **FINISHES["Machined"]}],
            relief_surfaces=(0,),
        )

    # Restitution is the only authored input to the contact's dissipation, so it is the one control over how sharply a contact rings.
    # Everything else here is held.
    for tag, label, restitution in (("a_Dead", "Dead", 0.0), ("b_Middling", "Middling", 0.5), ("c_Lively", "Lively", 0.9)):
        surface_scene(
            f"RestitutionLadder/{tag}",
            "One body on one finish at one of three restitutions. Restitution sets the contact's dissipation and nothing else differs.",
            bodies=[body(label, 0.5, 1, FinishIndex["Machined"], physics_material=0)],
            grounds=[finish_ground("Machined")],
            physics_materials=[{"staticFriction": 0.0, "dynamicFriction": 0.0, "restitution": restitution}],
        )

    # A small stiff body's modes lie above hearing, so the extension's Nyquist rule culls every one of them and the recoil is the whole sound.
    # A renderer that has not implemented acceleration noise produces silence here, which is the point: nothing else in the corpus separates the two.
    # It is dropped rather than slid, because the recoil comes from an impact.
    test_scene(
        "AccelerationNoise/a_SteelBead",
        "A small steel body whose modes are all ultrasonic, dropped. Every mode is culled, so what remains is the rigid-body recoil alone.",
        bodies=[body("Steel bead", 0.1, 1, 0, "sphere", drop=0.25)],
        grounds=[ground("Steel", 0, 1)],
        probe=solved_modes_sphere(SphereR, STEEL, max_freq=60000.0),
        surfaces=[{"name": "Steel machined", "material": 1, **FINISHES["Machined"]}],
        speed=0.0,
    )

    # A contact that separates applies no force, and that clamp at zero is what turns a bounce into a train of impacts rather than one soft landing.
    # Two bodies dropped from one height, differing only in the restitution of the material they land on.
    for tag, label, restitution in (("a_Bouncing", "Bouncing", 0.9), ("b_Dead", "Dead", 0.0)):
        test_scene(
            f"Separation/{tag}",
            "One body dropped, bouncing or landing dead by its restitution. A contact that parts applies no force, so the bouncing one strikes again and again.",
            bodies=[body(label, 0.5, 1, FinishIndex["Machined"], physics_material=0, drop=0.25)],
            grounds=[finish_ground("Machined")],
            physics_materials=[{"staticFriction": 0.22, "dynamicFriction": 0.2, "restitution": restitution}],
            speed=0.0,
        )

    # A loaded frictional interface damps the modes pressed against it: sub-cone micro-slip at the asperity junction bleeds each ring cycle, so grip shortens the decay of a struck body at rest where a frictionless support leaves it ringing free.
    # One strike, one finish, one load, and only the friction coefficient steps, so the ring's decay is the one thing that moves.
    # The body is low-loss glass: its free ring must outlast contact-borne dissipation by several times for the choke to read, and the ceramic bar's own material damping consumes its ring first.
    # The ring is authored already resting and a small striker is dropped onto it, which is what lets the body carry glass's own restitution: a landing at that restitution bounces for the whole render and re-excites the ring it is supposed to be choking, where a tap from above presses the body into its support and leaves the seated junction the only thing acting on the decay.
    # The striker carries a finish and no modal model, which the extension allows for exactly this, so it sounds only through the contact it makes, and its junction is frictionless and dead, so it neither rings nor grips and only the ground junction steps across the ladder.
    for tag, label, friction in (("a_NoGrip", "No grip", 0.0), ("b_Grip", "Grip", 0.3), ("c_FullGrip", "Full grip", 0.9)):
        surface_scene(
            f"PressedRing/{tag}",
            "One resting glass body tapped from above, on one finish at one of three friction coefficients. Interface micro-slip damps the pressed ring, so the decay shortens as grip grows.",
            bodies=[body(label, 0.5, 1, FinishIndex["Machined"]),
                    body("Striker", 0.05, 1, FinishIndex["Machined"], physics_material=1, drop=0.25, on=0, silent=True)],
            grounds=[finish_ground("Machined")],
            probe=solved_modes(BarHalf, GLASS),
            probe_material=2,
            acoustic_materials=ACOUSTIC_MATERIALS + [GLASS],
            physics_materials=[{"staticFriction": friction, "dynamicFriction": friction, "restitution": 0.9},
                               Frictionless | {"frictionCombine": "multiply", "restitutionCombine": "multiply"}],
            speed=0.0,
        )

    # One modal model, three instances, and only one of them struck.
    # Each instance carries its own resonator state, so the two at rest must stay silent however hard their neighbour is hit.
    # The silent co-present instances are the phenomenon, so this scene keeps all three bodies.
    test_scene(
        "StrikeOne/a_ThreeInstances",
        "Three instances of one model, one dropped and two already at rest. Only the struck instance rings.",
        bodies=[body("At rest", 0.5, 0, FinishIndex["Machined"], drop=0.0),
                body("Struck", 0.5, 1, FinishIndex["Machined"], drop=0.25),
                body("At rest too", 0.5, 2, FinishIndex["Machined"], drop=0.0)],
        grounds=[finish_ground("Machined", 0),
                 finish_ground("Machined", 1),
                 finish_ground("Machined", 2)],
        speed=0.0,
    )

    # A contact applies its force over a finite time, and each mode is excited by that pulse's spectrum at its own frequency, so a mode well above one over the contact time is barely reached.
    # The Hertz contact time goes as mass^(2/5) and only as speed^(-1/5), so mass is the lever that moves it: a thousandfold in mass is sixteenfold in contact time, where a thousandfold in drop height is under fourfold.
    # The heavy body should land duller as well as louder.
    # Every variant drops from one height, so the contact time answers to the mass alone.
    for tag, label, mass in (("a_5g", "5 g", 0.005), ("b_500g", "500 g", 0.5), ("c_5kg", "5 kg", 5.0)):
        test_scene(
            f"ContactDuration/{tag}",
            "One of three bodies of very different mass dropped onto one finish. The heavier the body the longer its contact, so the fewer of the high modes it reaches.",
            bodies=[body(label, mass, 1, FinishIndex["Machined"], drop=0.25)],
            grounds=[finish_ground("Machined")],
            speed=0.0,
        )

    # The contact time answers to the approach speed only while the patch is still growing under the load.
    # A sphere's grows for the whole collision, so its contact goes as speed^(-1/5).
    # A box lands on its face, which fills the patch at first touch, so its contact is a flat punch of fixed stiffness and lasts the same however hard it lands.
    # This is why ContactDuration reaches for mass instead of drop height: on a face, drop height moves nothing at all.
    # Two heights a hundredfold apart, which is tenfold in speed.
    for tag, label, shape, drop in (("a_SphereLight", "Sphere, light landing", "sphere", 0.005),
                                    ("b_SphereHard", "Sphere, hard landing", "sphere", 0.5),
                                    ("c_BoxLight", "Box, light landing", "box", 0.005),
                                    ("d_BoxHard", "Box, hard landing", "box", 0.5)):
        test_scene(
            f"PatchGrowth/{tag}",
            "A sphere or a box dropped from one of two heights. The sphere's patch grows under its load, so its contact shortens as it lands harder, and the box's is filled by its face, so its does not.",
            bodies=[body(label, 0.5, 1, FinishIndex["Machined"], shape, drop=drop)],
            grounds=[finish_ground("Machined")],
            speed=0.0,
        )

    # A stored profile is the only way two renderers read the same surface sample for sample.
    # The two lanes carry one roughness by two routes, so what they share is the statistics and what they do not share is the realization.
    measured, spacing = measured_profile()
    surface_scene(
        "MeasuredProfile/a_Stored",
        "One body on a stored profile. Only a stored profile is reproducible sample for sample across renderers.",
        bodies=[body("On a stored profile", 2.0, 1, 0)],
        grounds=[ground("Measured", 0, 1)],
        surfaces=[{"name": "Measured", "material": 0, "profileHeights": measured, "sampleSpacing": spacing,
                   **FINISHES["Machined"]}],
    )
    surface_scene(
        "MeasuredProfile/b_Synthesized",
        "The same body on the statistics the stored profile was drawn from. It shares the statistics and not the realization.",
        bodies=[body("On the statistics", 2.0, 1, 0)],
        grounds=[ground("Synthesized", 0, 1)],
        surfaces=[{"name": "Synthesized", "material": 0, **FINISHES["Machined"]}],
    )


if __name__ == "__main__":
    Only = frozenset(sys.argv[1:])
    main()
