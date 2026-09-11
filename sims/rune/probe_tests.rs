//! The cove's light, pinned.
//!
//! [`bake::run_light`] bakes two hundred thousand probes; this bakes three,
//! at the same seed and the same rays, and snapshots what they read. A
//! changed number means changed code — a moved sun, a different sand, a
//! different projection — which is the whole of the claim.
//!
//! The tolerance is 0.1 in the tracer's radiance units, against a sunlit
//! sand reading of about 2.3. That is four per cent, and the bake's own
//! seed-to-seed spread at 64 rays is 1.4 per cent, so the snapshot is loose
//! enough for the Monte Carlo and tight enough for the level.
//!
//! Every bake here decides which probes are rock the way the level's own bake
//! does, with [`bake::inside`] over the cove's field — never with the
//! crossing-count fallback, which the ground's fifty overlapping shells fool.

use std::sync::OnceLock;

use kosm::light::probes::{self, BakeSpec, VolumeSpec};
use kosm::snapshot;
use kosm_scan::SdfGrid;

use super::{CoveScene, bake, render};

/// The cove's field, baked once for every test in here: eight seconds each
/// is a price the second test should not pay again.
fn field() -> &'static SdfGrid {
    static FIELD: OnceLock<SdfGrid> = OnceLock::new();
    FIELD.get_or_init(|| {
        let scene = CoveScene::bundled().expect("the cove builds");
        bake::field(&scene).expect("the cove's field bakes")
    })
}

/// One probe, at a point in the cove, with everything else the bake's.
fn probe_at(scene: &CoveScene, lit: &render::Picture, p: [f64; 3], rays: usize) -> [f32; probes::BANDS] {
    let volume = VolumeSpec {
        origin: p,
        spacing: 1.0,
        dims: [1, 1, 1],
        scene_per_metre: render::PER_M,
    };
    let d = scene.sun_dir();
    let rock = bake::Rock::new(scene, field()).expect("the cove's parts evaluate");
    let inside = |p: [f64; 3]| rock.contains(p);
    let v = probes::bake_with(
        lit,
        &BakeSpec {
            volume,
            suns: vec![[d.x, d.y, d.z]],
            rays,
            seed: super::bake::LIGHT_SEED,
            inside: Some(&inside),
            ..BakeSpec::default()
        },
    );
    v.sample(0.0, p, [0.0, 0.0, 1.0])
}

/// Three probes, pinned.
///
/// **Re-recorded 2026-09-11.** The numbers moved with the sky: it is
/// Preetham's now (06c557b, "the cove gets a sky") where it was a two-colour
/// gradient, bluer overhead and warmer at the horizon, so every probe lost a
/// tenth to a fifth in the blue pair and gained up to a tenth in the red — the
/// door's sand went from 1.86 / 2.33 / 2.73 to 1.69 / 2.29 / 2.80.
///
/// The same recording pins the bake as the level now does it. Which probes are
/// rock is [`bake::Rock`]'s call — the field's sign, then the level's closed
/// parts where the sign cannot say — rather than a three-ray crossing count,
/// over a collision mesh welded per part whose sign ties are voted rather than
/// first-wins (`skatepark::collision_mesh`, `kosm_scan::TriMesh`). The weld and
/// the inside test move which of the cove's probes are rock — whole rows of
/// them along the bedding planes — and not these three, which are in the air
/// and read the same digits under either; a snapshot is of a tree, and this is
/// the tree it is of.
#[test]
fn three_probes_of_the_cove_are_pinned() -> anyhow::Result<()> {
    let scene = CoveScene::bundled()?;
    let mut picture = render::Scene::new(&scene)?;
    // The hero walks; its shadow is not the level's light.
    picture.set_being_visible(false);
    let placement = render::Placement::standing(&scene, scene.spawn_x, scene.spawn_y, 0.0);
    let lit = picture.at(&placement);

    let door = scene.door_frame().origin;
    let points = [
        // the sand two metres off the door, where the hero stands to solve it
        [door.x, door.y - 2.0, scene.sand_z_at(door.x, door.y - 2.0) + 0.05],
        // out on the open beach, a metre up
        [0.0, -5.0, scene.sand_z_at(0.0, -5.0) + 1.0],
        // and three metres over the sea, which is sky and water and no sand
        [0.0, -25.0, 3.0],
    ];
    let mut values = Vec::new();
    for p in points {
        let e = probe_at(&scene, &lit, p, 64);
        println!(
            "cove probe ({:+.1}, {:+.1}, {:+.1}): {:.3} {:.3} {:.3} {:.3} {:.3} {:.3}",
            p[0], p[1], p[2], e[0], e[1], e[2], e[3], e[4], e[5]
        );
        values.extend(e.iter().map(|v| *v as f64));
    }
    snapshot::assert_close("rune/probes", &values, 0.1)
}

#[test]
fn the_rock_is_inside_and_the_sand_is_lit() -> anyhow::Result<()> {
    // Two claims about the bake, at its own defaults. The sun reaches the
    // open sand — the cove's sun is low over the sea and rakes the beach,
    // which is why the door's own face is lit too — and a probe buried in the
    // rock is *marked* rather than baked, so a sampler never drags the inside
    // of the rock onto the face of it.
    //
    // The first is about `BakeSpec::sun_direct`, which is true here and is
    // **false** in `bake::run_light`: the volume the raster tier reads leaves
    // the direct term to the shader and its shadow map, and keeps every
    // bounce. This test is the one that says the term is there to leave out.
    //
    // The second is about `BakeSpec::inside`, and it is the one that failed
    // while the bake asked `probes::inside_solid`'s three-ray crossing count:
    // half a metre under the sand in front of the door, the count read air.
    // The ground is fifty closed shells that overlap, and a crossing count is
    // only an inside test for one. The level's bake asks the field instead.
    let scene = CoveScene::bundled()?;
    let mut picture = render::Scene::new(&scene)?;
    picture.set_being_visible(false);
    let placement = render::Placement::standing(&scene, scene.spawn_x, scene.spawn_y, 0.0);
    let lit = picture.at(&placement);
    let door = scene.door_frame().origin;
    let rock = bake::Rock::new(&scene, field())?;
    let inside = |p: [f64; 3]| rock.contains(p);
    let marked = |p: [f64; 3]| {
        let volume = VolumeSpec { origin: p, spacing: 1.0, dims: [1, 1, 1], scene_per_metre: render::PER_M };
        probes::bake_with(&lit, &BakeSpec { volume, rays: 8, inside: Some(&inside), ..BakeSpec::default() })
            .is_inside(0, 0, 0)
    };

    let open_at = [door.x - 6.0, door.y - 6.0, scene.sand_z_at(0.0, door.y - 6.0) + 0.5];
    let open = probe_at(&scene, &lit, open_at, 64);
    println!("cove sun: {:.3} on the open sand", open[4]);
    anyhow::ensure!(open[4] > 2.0, "the open sand should carry the sun, not {}", open[4]);

    // Two metres into the cliff's beds over the door, and half a metre under
    // the sand two metres off it.
    for p in [
        [door.x, scene.cliff_face_y() + 2.0, scene.door_sill() + 2.0],
        [door.x, door.y - 2.0, scene.sand_z_at(door.x, door.y - 2.0) - 0.5],
    ] {
        anyhow::ensure!(marked(p), "({:+.1}, {:+.1}, {:+.1}) is inside the rock", p[0], p[1], p[2]);
    }
    // And the air is not: the open sand's probe, and a hand's breadth over the
    // sand at the door, where the hero stands.
    for p in [open_at, [door.x, door.y - 2.0, scene.sand_z_at(door.x, door.y - 2.0) + 0.05]] {
        anyhow::ensure!(!marked(p), "({:+.1}, {:+.1}, {:+.1}) is in the air", p[0], p[1], p[2]);
    }
    Ok(())
}
