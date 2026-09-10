//! The material library, end to end: a constant fitted out of a caustic, and
//! two datasheets pinned.
//!
//! `cargo test -p kosm --release --test material`. Release, because the fit
//! traces a hundred thousand rays per loss evaluation and the datasheet
//! renders two material balls.

use std::path::PathBuf;

use kosm::material::{self, Material};
use kosm::prelude::*;
use kosm::{glass, light, snapshot};
use tang::Vec3;

/// `out/materials/`, found from this crate rather than from the test's cwd.
fn out_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out/materials")
}

/// `N-BK7` -> `n_bk7`.
fn slug(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect()
}

// ── the fit ───────────────────────────────────────────────────────────────

/// The caustic a sphere of glass with index `n_d` throws onto the plate
/// below it. A small budget: 32 cells over a 20 mm window, 20 000 rays a
/// band. The lattice jitter in `light::trace` is deterministic, so this is a
/// deterministic function of `n_d` and a central difference over it means
/// something.
fn caustic(n_d: f64) -> light::Caustic<f64> {
    const R: f64 = 0.010;
    let lamp = Vec3::new(0.0, 0.0, 0.15);
    let shape = glass::Shape::Sphere { centre: Vec3::new(0.0, 0.0, R), r: R };
    light::trace::<f64>(lamp, &shape, n_d, 0.02, 32, 20_000)
}

/// `Material::fit` recovers N-BK7's index from the caustic it throws.
///
/// The same demonstration `sims/marble` makes with duals, made here with the
/// zeroth-order path and over a *substance* rather than a bare scalar: the
/// thing being moved is `named("N-BK7").n_d`, one of the material's own
/// constants, and everything else about the glass is left exactly where the
/// datasheet put it.
#[test]
fn n_d_is_recovered_from_a_caustic() {
    let truth = material::named("N-BK7").expect("N-BK7 is in the library");
    let n_true = truth.n_d().expect("N-BK7 has an index");
    let target = caustic(n_true);

    // Start well below it, the way the marble's fit does.
    let start = truth.with_params(&[Param::new("N-BK7.n_d", 1.42)]);
    assert_eq!(start.n_d(), Some(1.42));

    let fitted = start.fit(|m| light::loss(&caustic(m.n_d().unwrap()), &target), &["n_d"], 30);
    let got = fitted.n_d().expect("still a dielectric");
    println!("fit    n_d {:.4} -> {got:.4} (true {n_true})", 1.42);
    assert!(
        (got - n_true).abs() < 5e-3,
        "recovered n_d = {got}, want {n_true} (started at 1.42)"
    );

    // The fit moved the constant it was told to and nothing else: the density,
    // the loss factor, the Sellmeier pair and the provenance are the
    // datasheet's still.
    assert_eq!(fitted.density, truth.density);
    assert_eq!(fitted.loss, truth.loss);
    assert_eq!(fitted.provenance, truth.provenance);
    assert!(fitted.pbr().sellmeier.is_some());

    // And a fit that starts at the answer stays there.
    let held = truth.fit(|m| light::loss(&caustic(m.n_d().unwrap()), &target), &["n_d"], 5);
    assert!((held.n_d().unwrap() - n_true).abs() < 1e-6);
}

// ── the datasheets ────────────────────────────────────────────────────────

fn sheet(name: &str) -> (Material, kosm::material::Datasheet) {
    let m = material::named(name).unwrap_or_else(|| panic!("`{name}` is in the library"));
    let dir = out_dir().join(slug(&m.name));
    let sheet = material::datasheet(&m, &dir).expect("datasheet");
    println!(
        "sheet  {:16} rho {:7.0}  E {:8.3e}  mu {:.2}  e {:.2}  bounce {:?}  ring {:?}  -> {}",
        sheet.name,
        m.density,
        m.young,
        m.friction,
        m.restitution,
        sheet.bounce.iter().map(|b| (b.ratio * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        sheet.ring_hz.iter().map(|h| (h / 1000.0).round()).collect::<Vec<_>>(),
        dir.display()
    );
    (m, sheet)
}

/// N-BK7's and brass's datasheets, pinned.
///
/// Every number here is a phyz rollout or a closed form over the library's own
/// constants, so a changed number means changed code — either the constants
/// moved, the contact solve moved, or the modal bank moved. All three are
/// worth being told about.
#[test]
fn the_datasheets_of_glass_and_brass_are_pinned() -> anyhow::Result<()> {
    let (glass, glass_sheet) = sheet("N-BK7");
    let (brass, brass_sheet) = sheet("brass");

    // The constants are the datasheet's, and they are marked measured.
    assert_eq!(glass.density, 2510.0);
    assert!(glass.provenance.optics.is_measured());
    assert!(brass.provenance.mechanics.is_measured());

    // Glass is the harder, bouncier one; brass is the heavier, deader one.
    for (g, b) in glass_sheet.bounce.iter().zip(&brass_sheet.bounce) {
        assert!(g.rebound_m > b.rebound_m, "glass {g:?} vs brass {b:?}");
    }
    // Both ring, and glass rings higher: it is three times lighter than brass
    // for a comparable modulus.
    assert_eq!(glass_sheet.ring_hz.len(), 5);
    assert_eq!(brass_sheet.ring_hz.len(), 5);
    assert!(glass_sheet.ring_hz[0] > brass_sheet.ring_hz[0]);
    // Neither is a liquid.
    assert!(glass_sheet.settle_mps.is_none() && brass_sheet.settle_mps.is_none());

    snapshot::assert_close("_materials/n_bk7", &glass_sheet.numbers(), 1e-6)?;
    snapshot::assert_close("_materials/brass", &brass_sheet.numbers(), 1e-6)?;

    // The balls were written where the report says they are.
    for name in ["n_bk7", "brass"] {
        let png = out_dir().join(name).join("ball.png");
        assert!(png.exists(), "{} was not written", png.display());
        let image = image::open(&png)?.to_rgba8();
        assert_eq!(image.dimensions(), (320, 240));
        // Not a blank frame: the studio rig lit something.
        let mean: f64 =
            image.as_raw().iter().map(|c| *c as f64).sum::<f64>() / image.as_raw().len() as f64;
        assert!(mean > 5.0, "{name}'s ball is black ({mean:.1})");
    }

    // And the JSON beside it round-trips.
    let json = std::fs::read_to_string(out_dir().join("brass").join("datasheet.json"))?;
    let back: kosm::material::Datasheet = serde_json::from_str(&json)?;
    assert_eq!(back.name, "brass");
    assert_eq!(back.material.provenance, brass.provenance);
    Ok(())
}

// ── authoring ─────────────────────────────────────────────────────────────

/// A body authored by value hands its constants back, and a body authored by
/// name resolves them through the library — one table either way.
#[test]
fn a_built_body_knows_what_it_is_made_of() -> anyhow::Result<()> {
    let bronze = material::named("bell bronze").expect("bell bronze");
    let by_value = bronze.clone();
    let built = build(&Params::default(), move |b| {
        b.body("plate").material("granite").boxed(300.0, 300.0, 20.0);
        b.body("bell").substance(&by_value).sphere(30.0).dynamic(1.0).at(0.0, 0.0, 100.0);
    })?;

    let plate = built.bodies[0].substance().expect("granite is in the library");
    assert_eq!(plate.name, "granite");
    assert!(plate.contact().friction > 0.6);

    let bell = built.bodies[1].substance().expect("authored by value");
    assert_eq!(bell, bronze);
    assert!(bell.modal().loss < 1e-3);
    // the document carries the substance's name, so the colour path is
    // untouched by having authored it by value
    assert_eq!(built.bodies[1].material, "bell bronze");

    // A substance the library has never heard of still comes back, because the
    // author had it.
    let invented = Material { name: "unobtainium".into(), density: 42.0, ..Material::default() };
    let odd = invented.clone();
    let built = build(&Params::default(), move |b| {
        b.body("ingot").substance(&odd).boxed(20.0, 20.0, 20.0);
    })?;
    assert_eq!(built.bodies[0].substance(), Some(invented));
    Ok(())
}
