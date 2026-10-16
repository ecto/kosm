//! Training throughput of the denoiser, on a fixed synthetic dataset.
//!
//! ```text
//! cargo run --release -p kosm --example denoise_bench -- [--tiles 64] [--tile 64] [--hidden 32] [--epochs 3]
//! ```
//!
//! No file, no GPU: the tiles are a deterministic hash of their coordinates,
//! every one paired with a successor frame, so the number this prints is the
//! cost of the fit and nothing else — one forward and backward over both
//! frames of a tile is one *tile-pass*. The loss curve is printed too, so a
//! change that made the fit faster by making it wrong shows up here first.

use std::time::Instant;

use kosm::denoise::{
    dataset::{Tile, TileNext},
    kpn::Kpn,
    train,
};

fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn hash(mut z: u64) -> f32 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
}

/// A tile with structure a filter can learn something from: a smooth
/// reference, a noisy mean around it, a few fireflies, an edge in depth.
fn tile(t: u64, s: usize) -> Tile {
    let n = s * s;
    let h = |p: usize, k: u64| hash(t * 1_000_003 + p as u64 * 31 + k);
    let mut reference = vec![0.0f32; 3 * n];
    let mut mean = vec![0.0f32; 3 * n];
    let mut next_mean = vec![0.0f32; 3 * n];
    let mut albedo = vec![0.0f32; 3 * n];
    let mut normal = vec![0.0f32; 3 * n];
    let mut depth = vec![0.0f32; n];
    let mut variance = vec![0.0f32; n];
    let mut next_var = vec![0.0f32; n];
    let mut id = vec![0.0f32; n];
    let count: Vec<f32> = (0..n).map(|p| 1.0 + (h(p, 9) * 8.0).floor()).collect();
    let next_count: Vec<f32> = (0..n)
        .map(|p| if h(p, 10) < 0.8 { count[p] + 1.0 } else { 1.0 })
        .collect();
    for p in 0..n {
        let (x, y) = ((p % s) as f32 / s as f32, (p / s) as f32 / s as f32);
        let edge = x + 0.3 * y > 0.5 + 0.1 * (t % 5) as f32;
        depth[p] = if edge { 900.0 } else { 2600.0 };
        normal[p * 3 + if edge { 1 } else { 2 }] = 1.0;
        id[p] = if edge { 3.0 } else { 7.0 };
        for c in 0..3 {
            albedo[p * 3 + c] = 0.1 + 0.6 * h(p, c as u64 + 20);
            let r = 0.2 + 0.5 * (x * (1.0 + c as f32)).sin().abs() + if edge { 0.6 } else { 0.0 };
            reference[p * 3 + c] = r * albedo[p * 3 + c];
            let fire = if h(p, 30) < 0.01 { 40.0 } else { 1.0 };
            mean[p * 3 + c] = reference[p * 3 + c] * (0.4 + 1.2 * h(p, c as u64 + 40)) * fire;
            next_mean[p * 3 + c] = reference[p * 3 + c] * (0.4 + 1.2 * h(p, c as u64 + 50));
        }
        variance[p] = 0.05 * h(p, 60);
        next_var[p] = 0.05 * h(p, 61);
    }
    Tile {
        size: s,
        tier: 4,
        mean,
        variance,
        count,
        normal,
        depth,
        albedo,
        id,
        reference,
        next: Some(Box::new(TileNext {
            mean: next_mean,
            variance: next_var,
            count: next_count,
        })),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tiles: usize = num(&args, "--tiles", 64);
    let s: usize = num(&args, "--tile", 64);
    let hidden: usize = num(&args, "--hidden", 32);
    let epochs: usize = num(&args, "--epochs", 3);
    let batch: usize = num(&args, "--batch", 16);

    let train: Vec<train::Prepared> = (0..tiles as u64)
        .map(|t| train::Prepared::of(&tile(t, s)))
        .collect();
    let mut net = Kpn::new(hidden, 1);
    let cfg = train::Fit {
        epochs,
        batch,
        lr: 2e-3,
        seed: 1,
    };
    println!(
        "{tiles} tiles of {s}x{s}, hidden {hidden}, batch {batch}, {} threads",
        rayon::current_num_threads()
    );
    let start = Instant::now();
    let mut last = start;
    let mut first_epoch = 0.0;
    let curve = train::fit(&mut net, &train, &cfg, |e, l| {
        let dt = last.elapsed().as_secs_f64();
        last = Instant::now();
        if e == 0 {
            first_epoch = dt;
        }
        println!(
            "  epoch {e:2}  loss {l:.6}  {dt:.2} s  {:.1} tile-passes/s",
            tiles as f64 / dt
        );
    });
    let total = start.elapsed().as_secs_f64();
    let steady = if epochs > 1 {
        (total - first_epoch) / (epochs - 1) as f64
    } else {
        total
    };
    println!(
        "total {total:.2} s; steady {steady:.2} s/epoch = {:.1} tile-passes/s; loss {} -> {}",
        tiles as f64 / steady,
        curve.first().unwrap(),
        curve.last().unwrap()
    );
}
