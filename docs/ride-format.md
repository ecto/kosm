# `ride.json` — a recorded rollout, for the window

Written by an ipse rollout (`crates/ipse-sim/examples/k1_skatepark_ride.rs`),
read by `kosm-view --ride <ride.json>`. Metres, z up, quaternions as
`[w, x, y, z]`, rotations map local to world. Every path is absolute.

```json
{
  "name": "skatepark — mini ramp, skate-cem-p96x48-s9",
  "dt": 0.016666,                       // seconds between frames
  "meshes": [                           // shapes, referenced by index
    {"kind": "stl", "path": "/abs/K1/meshes/Trunk.STL", "scale": 1.0},
    {"kind": "box", "half": [0.39, 0.098, 0.0055]},
    {"kind": "sphere", "radius": 0.027},
    {"kind": "cylinder", "radius": 0.008, "half_length": 0.09, "axis": [0, 1, 0]}
  ],
  "fixed": [                            // things that never move
    {"mesh": 0, "pos": [0, 0, 0], "quat": [1, 0, 0, 0], "colour": [0.6, 0.6, 0.62], "label": "park"}
  ],
  "actors": [                           // things with a pose per frame
    {"mesh": 1, "colour": [0.75, 0.76, 0.78], "label": "Trunk",
     "offset_pos": [0, 0, 0], "offset_quat": [1, 0, 0, 0]}   // mesh origin in the body, applied after the frame pose
  ],
  "track": 3,                           // actor index the camera follows (the trunk)
  "frames": [
    {"t": 0.0, "poses": [[[x, y, z], [w, x, y, z]], ...]}   // one [pos, quat] per actor, in `actors` order
  ]
}
```

`scale` on an `stl` mesh multiplies its vertices (the K1's STLs are already in metres).
`fixed` is drawn once; `actors[i]` is drawn every frame at
`frame.poses[i]` composed with its offset: `world = pose ∘ offset`.
Colours are linear RGB in 0..1. A reader ignores keys it does not know.

## Streaming: `--stream`

For a live window the same recorder writes the ride to stdout as it runs,
one JSON object per line, flushed per frame:

1. the header: the `ride.json` object with `"frames": []`;
2. then one frame object per line, `{"t": 0.017, "poses": [...]}`, in order;
3. EOF when the rollout ends (or the process is killed).

Diagnostics go to stderr, never stdout. A reader appends frames as they
arrive and may keep following the newest one.
