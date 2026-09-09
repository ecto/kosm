//! The net: it hangs, and a made shot goes through it.
use crate::court::{Court, CourtScene};

/// Left alone, the net hangs from the rim at the length it was cut to: the
/// lowest node within a few centimetres of rim − net_length, no cord stretched
/// far, nothing gone to NaN.
#[test]
fn the_net_hangs_from_the_rim() {
    let mut scene = CourtScene::bundled().unwrap();
    scene.shot = None;
    scene.n_balls = 1; // dropped far down the court, nowhere near the hoop
    let mut court = Court::from_scene(&scene).unwrap();
    while court.time() < 2.0 {
        court.step();
    }
    let net = court.net.as_ref().expect("the level's net");
    assert!(net.nodes.iter().all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite()), "a node went to NaN");
    assert!(net.vel.iter().all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite()), "a velocity went to NaN");

    let length = scene.authored.millimetres("net_length_mm").unwrap();
    let rod = scene.authored.millimetres("rim_rod_mm").unwrap();
    let want = scene.hoop.rim_centre.z - 0.5 * rod - length;
    let low = net.lowest();
    println!("net hangs to {low:.4} m, cut for {want:.4} m; worst stretch {:+.3}%", net.worst_stretch() * 100.0);
    assert!((low - want).abs() < 0.03, "the net hangs to {low:.4} m, expected about {want:.4} m");
    assert!(net.worst_stretch() < 0.05, "a cord stretched {:.1}%", net.worst_stretch() * 100.0);
}

/// The level's own free throw, with the net on: it still goes in, and the net
/// gets out of the way as it does.
#[test]
fn a_made_shot_goes_through_the_net() {
    let mut scene = CourtScene::bundled().unwrap();
    scene.n_balls = 0; // the shot alone
    let mut court = Court::from_scene(&scene).unwrap();
    let rest: Vec<_> = court.net.as_ref().unwrap().nodes.clone();

    let mut moved: f64 = 0.0;
    while court.time() < 3.0 {
        court.step();
        let net = court.net.as_ref().unwrap();
        for (p, q) in net.nodes.iter().zip(&rest) {
            moved = moved.max((*p - *q).norm());
        }
    }
    let made = court.made_at.expect("the level's free throw should still go in with a net on the rim");
    println!("made at {made:.3} s; the net moved {:.3} m at most", moved);
    assert!(moved > 0.05, "the net barely moved ({moved:.4} m): the ball did not pass through it");
    let net = court.net.as_ref().unwrap();
    assert!(net.nodes.iter().all(|p| p.z.is_finite()), "a node went to NaN");
}

/// A net let go from its cut shape swings, and then it hangs still. The
/// length pass used to leave velocities behind and pump the swing forever.
#[test]
fn the_net_comes_to_rest() {
    let mut scene = CourtScene::bundled().unwrap();
    // one ball, dropped far from the hoop, so the net is left alone
    scene.n_balls = 1;
    scene.shot = None;
    let mut court = Court::from_scene(&scene).unwrap();
    while court.time() < 8.0 {
        court.step();
    }
    let net = court.net.as_ref().expect("the level asks for a net");
    let speed = net.max_speed();
    println!("fastest node after 8 s: {:.2e} m/s", speed);
    assert!(speed < 2e-3, "the net should be still after 8 s, fastest node moves at {speed:.3e} m/s");
}
