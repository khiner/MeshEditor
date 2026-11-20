"""Measure sustained display-space convergence on the three sphere silhouettes."""

import hashlib
import json
from statistics import median

import numpy as np
from PIL import Image


def sphere_mask(scene_file, pose):
    scene = json.loads(scene_file.read_text())
    width, height = pose["width"], pose["height"]
    y, x = np.ogrid[:height, :width]
    projection = pose["projection"]
    dx = ((x + 0.5) * 2 / width - 1) / projection[0][0]
    dy = (1 - (y + 0.5) * 2 / height) / projection[1][1]
    squared_length = dx * dx + dy * dy + 1
    view = np.linalg.inv(pose["cameraToWorld"])
    mask = np.zeros((height, width), dtype=bool)
    for node in scene["nodes"]:
        if "glass" not in node.get("name", "").lower():
            continue
        primitive = scene["meshes"][node["mesh"]]["primitives"][0]
        bounds = scene["accessors"][primitive["attributes"]["POSITION"]]
        assert bounds["min"] == [-1] * 3 and bounds["max"] == [1] * 3
        cx, cy, cz, _ = view @ [*node["translation"], 1]
        # Intersect pixel-center camera rays with the analytic unit sphere.
        dot = dx * cx + dy * cy - cz
        mask |= (dot > 0) & (dot * dot >= squared_length * (cx * cx + cy * cy + cz * cz - 1))
    assert mask.any()
    return mask


def summarize(out, scene_file):
    camera_log = (out / "recovery-0.jsonl").read_bytes()
    poses = [json.loads(line) for line in camera_log.splitlines()]
    mask = sphere_mask(scene_file, poses[-1])
    blender = json.loads((out / "recovery-eevee-trials.json").read_text())
    result = {
        "orbitFrames": 32,
        "holdFrames": 72,
        "cameraSha256": hashlib.sha256(camera_log).hexdigest(),
        "nativeExecutableSha256": (out / "recovery-build.txt").read_text().strip(),
        "blenderRevision": blender["revision"],
        "glassPixels": int(mask.sum()),
    }
    thresholds = [4, 2, 1]
    for engine in ["mesh", "eevee"]:
        trials = []
        for trial in range(3):

            def pixels(frame):
                with Image.open(out / f"{engine}-{trial}-{frame:03}.png") as image:
                    return np.asarray(image.convert("RGB"))

            reference = pixels(72)
            assert reference.shape[:2] == mask.shape
            errors = []
            settled_frame = 0
            for index in range(73):
                delta = pixels(index).astype(np.float32) - reference
                errors.append(float(np.sqrt(np.mean(delta[mask] ** 2, dtype=np.float64))))
                if delta.any():
                    settled_frame = index + 1
            if settled_frame > 64:
                raise ValueError(f"{engine} trial {trial} still changing after 64 frames")
            times = np.loadtxt(out / f"{engine}-{trial}.ms")
            assert times.shape == (73,) and times[0] == 0 and np.all(np.diff(times) >= 0)
            # A crossing counts only when every remaining frame satisfies the threshold.
            sustained = np.maximum.accumulate(errors[::-1])[::-1]
            frames = [int(np.flatnonzero(sustained <= threshold)[0]) for threshold in thresholds]
            trials.append(
                dict(
                    frames=frames,
                    settledFrame=settled_frame,
                    rmse=[round(x, 4) for x in errors],
                    elapsedMs=times.round(3).tolist(),
                )
            )
        crossings = []
        for i, threshold in enumerate(thresholds):
            frames = [t["frames"][i] for t in trials]
            elapsed = [t["elapsedMs"][f] for t, f in zip(trials, frames)]
            crossings.append(
                dict(rmseThreshold=threshold, frames=frames, trialsMs=elapsed, medianMs=median(elapsed))
            )
        result[engine] = dict(crossings=crossings, trials=trials)
    (out / "recovery.json").write_text(json.dumps(result, indent=2) + "\n")
    y, x = np.nonzero(mask)
    crop = (int(x.min()) - 12, int(y.min()) - 12, int(x.max()) + 13, int(y.max()) + 13)
    w, h = crop[2] - crop[0], crop[3] - crop[1]
    plate = Image.new("RGB", (2 * w, 2 * h))
    for column, engine in enumerate(["mesh", "eevee"]):
        for row, frame in enumerate([0, 72]):
            with Image.open(out / f"{engine}-0-{frame:03}.png") as image:
                plate.paste(image.crop(crop), (column * w, row * h))
    plate.save(out / "recovery.webp", lossless=True)
    plate.thumbnail((960, 960), Image.Resampling.LANCZOS)
    plate.save(out / "recovery-preview.webp", quality=90)
