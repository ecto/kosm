# The Warehouse: a THPS1-shaped park for the K1

Date: 2026-09-05. Status: agreed (scope C: level + building, rendered by the
window's Lambert tier; the path tracer is another agent's).

## Shape

Tony Hawk's Pro Skater 1's first level, scaled to the K1 (about two thirds
of a skater). `levels/warehouse.loon`, all knobs `defparam`s, Z-up, mm.
`levels/skatepark.loon` (the mini ramp alone) stays.

| piece | where | size |
|---|---|---|
| room | inner 16 × 9 m, walls 4.5 m, x along the length | |
| half pipe | −x end, the mini ramp: two facing transitions | R 1.2 m, lip 0.6 m, width 3 m, flat 3 m |
| mezzanine | above the half pipe, the "secret room" | floor at z 2.6 m, 3 × 3 m, a rail along its edge |
| quarter pipes | +x end, either side of the door | R 1.2 m, lip 0.6 m, width 2.5 m each |
| platform + qp | +x end, one side wall: a qp up to a raised platform with a small qp on top | platform 0.8 m tall, 3 × 2.5 m |
| rail | middle of the floor, bent 15° at its centre | steel Ø 50 mm, 0.3 m high, 2 × 2 m |
| kickers | facing each other across a gap on the flat | 0.9 m long, 0.3 m tall wedge (a box rotated about y, sunk into the slab), 1.2 m wide, 1.5 m gap |
| box piles | five, by the walls | stacks of 0.5 m cubes, 2–3 high |
| ledge | along one side wall | 0.35 m tall, 3 m long, 0.4 m deep, steel edge |
| door | +x end wall, centred, open | 3 m wide, 3 m tall |
| clerestory | high on the +y wall | a row of windows, sill 3.2 m, 1 m tall |
| skylight | along the roof ridge | 1.5 m wide strip |
| roof | gabled, on trusses | ridge 5.5 m; 5 trusses |
| lamps | hung from the trusses | 6 high-bay discs |

The K1 spawns on the flat between the kickers, facing +x toward the rail.

## Two sets of roots

Every root has a material name. Roots whose material is `no-collide`
(roof, trusses, skylight, lamps, the walls above 1 m, window frames, door
frame) are drawn but not baked; everything else is the collision set. The
walls' bottom metre is a `brick` root in the collision set, so the K1 cannot
leave the building without the SDF growing to the roof. Bake cell defaults
to 20 mm for this level (the volume is 60× the mini ramp's).

## What the bake writes, beyond the map

`out/maps/<level stem>/parts/<root>.stl` per root (metres, drawn geometry,
both sets) and `parts.json`:

```json
[{"name": "walls", "material": "brick", "path": "/abs/parts/walls.stl",
  "colour": [0.58, 0.32, 0.25], "collide": true}]
```

Colours come from one table keyed by material name (`plywood`, `masonite`,
`steel`, `concrete`, `brick`, `galvanized`, `glass`, `lamp`, `cardboard`,
`no-collide` falls back to the root name's own material where given). The
ride recorder, when `parts.json` exists beside the scenario's map, emits one
`fixed` instance per part with that colour instead of the single grey
`mesh.stl`; otherwise as today.

## Physics check

The mini ramp check runs as before on the half pipe. Added: the wheel
released on the +x quarter pipe reaches the flat at the same 10/7·g·Δ; a
wheel rolled at the kicker leaves it (apex above the kicker's lip); the SDF
at the rail's top reads Ø/2 from its axis.
