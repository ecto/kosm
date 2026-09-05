# Skatepark: a mini ramp the K1 can stand on

Date: 2026-09-05. Status: agreed, building.

## Purpose

A training level for the Booster K1 in `../ipse`. ipse already has the
skateboard rig (`ipse-sim/src/skate.rs`) and a K1 that rides it on a flat
floor; its terrain comes from an ipse-map directory (collision mesh + baked
SDF) that a scenario file points at. Kosm's job is the park: author it as
geometry, bake it into a map ipse-sim can stand on, and check the physics of
that map against closed-form numbers before the robot sees it.

## Level: `levels/skatepark.loon`

A mini ramp. Z-up, millimetres, the court's conventions. Knobs:

| knob | default | meaning |
|---|---|---|
| `tr_r_mm` | 1200 | transition radius |
| `lip_mm` | 600 | lip height above the flat |
| `width_mm` | 2400 | ramp width (y) |
| `flat_mm` | 3000 | flat bottom length (x) |
| `deck_mm` | 600 | platform depth behind each lip |
| `coping_r_mm` | 30 | coping rod radius |
| `slab_t_mm` | 40 | slab under everything, top at z = 0 |
| `second_side` | 1 | 0 makes it a single quarterpipe |
| `sdf_cell_mm` | 10 | bake sample spacing |

A quarterpipe is a box minus a cylinder whose axis runs along y, one radius
above the end of the flat, so `lip_mm < tr_r_mm` is under-vert. A deck box
sits behind the lip; a coping cylinder rides the edge. Sized for a K1
(~1.2 m tall), not a person.

## Bake: `skatepark.rs`, `kosm-spike --skatepark`

Evaluate the document, collect the part meshes in metres, bake with
`ipse_map::SdfGrid::bake`, and write a real map directory under
`out/maps/skatepark/`: `mesh.stl`, `sdf.bin`, `map.toml`, plus an isometric
SVG. Fidelity check: sample the SDF along the ideal arc and report the max
error against the analytic circle. The wheels are 27 mm radius, so the cell
defaults to 10 mm.

## Physics check

The court's e² test, for a ramp: a wheel-sized solid sphere set on one
transition at height h, released, rolled through the same SDF contact path
the K1's feet use (`garage.rs`'s step). Closed form on the flat:
`v² = 10/7 · g · h` for rolling without slip; it should climb the far wall
back to h minus rolling losses. Printed and pinned by `tests/skatepark.rs`:
exit speed vs prediction, far-wall apex vs h, lateral drift (a leaning SDF
normal shows up here).

## ipse hookup

The bake also writes `scenario.toml` next to the map: the K1 on its board on
the flat with a sagittal shove toward the transition, so ipse's scenario
runner can use the park with no ipse code change.

## Out of scope

Board physics in kosm, rendering beyond the SVG, bowls, rails, stairs.
