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

use kosm::light::probes::{self, BakeSpec, VolumeSpec};
use kosm::snapshot;

use super::{CoveScene, render};

/// One probe, at a point in the cove, with everything else the bake's.
fn probe_at(scene: &CoveScene, lit: &render::Picture, p: [f64; 3], rays: usize) -> [f32; probes::BANDS] {
    let volume = VolumeSpec {
        origin: p,
        spacing: 1.0,
        dims: [1, 1, 1],
        scene_per_metre: render::PER_M,
    };
    let d = scene.sun_dir();
    let v = probes::bake_with(
        lit,
        &BakeSpec {
            volume,
            suns: vec![[d.x, d.y, d.z]],
            rays,
            seed: super::bake::LIGHT_SEED,
            ..BakeSpec::default()
        },
    );
    v.sample(0.0, p, [0.0, 0.0, 1.0])
}

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
    // Two claims the raster tier leans on: the sun reaches the open sand —
    // the cove's sun is low over the sea and rakes the beach, which is why
    // the door's own face is lit too — and a probe buried in the cliff is
    // *marked* rather than baked, so a sampler never drags the inside of the
    // rock onto the face of it.
    let scene = CoveScene::bundled()?;
    let mut picture = render::Scene::new(&scene)?;
    picture.set_being_visible(false);
    let placement = render::Placement::standing(&scene, scene.spawn_x, scene.spawn_y, 0.0);
    let lit = picture.at(&placement);
    let door = scene.door_frame().origin;

    let open = probe_at(&scene, &lit, [door.x - 6.0, door.y - 6.0, scene.sand_z_at(0.0, door.y - 6.0) + 0.5], 64);
    println!("cove sun: {:.3} on the open sand", open[4]);
    anyhow::ensure!(open[4] > 2.0, "the open sand should carry the sun, not {}", open[4]);

    // A metre into the cliff, and a metre under the sand at the door.
    for p in [
        [door.x, scene.cliff_face_y() + 2.0, scene.door_sill() + 2.0],
        [door.x, door.y - 2.0, scene.sand_z_at(door.x, door.y - 2.0) - 0.5],
    ] {
        let volume = VolumeSpec { origin: p, spacing: 1.0, dims: [1, 1, 1], scene_per_metre: render::PER_M };
        let v = probes::bake_with(&lit, &BakeSpec { volume, rays: 8, ..BakeSpec::default() });
        anyhow::ensure!(v.is_inside(0, 0, 0), "({:+.1}, {:+.1}, {:+.1}) is inside the rock", p[0], p[1], p[2]);
    }
    Ok(())
}
