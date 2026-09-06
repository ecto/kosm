//! Fit the denoiser to a v2 dataset, and score it against à-trous.
//!
//! ```text
//! # the expensive half, and it needs a GPU, so it lives in the viewer:
//! cargo run --release -p kosm-view -- --dump-dataset target/denoise-v2.bin
//!
//! cargo run --release -p kosm-spike --example denoise_dataset -- \
//!     train --data target/denoise-v2.bin --out target/denoise-court.bin
//! ```
//!
//! **Generation is not here any more.** A v1 dataset was CPU films of
//! independent passes and this example could make one; a v2 dataset is the
//! *device history* read back after the same passes the window runs, so the
//! only thing that can produce one is the thing that drives the device — see
//! `kosm-view`'s `--dump-dataset`. That move is the whole point of v2: the
//! v1 network was scored on a distribution nothing ever handed it.

use std::time::Instant;

use kosm_spike::court::denoise::{dataset, kpn::Kpn, train};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    arg(args, name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("train") => fit(&args),
        _ => {
            eprintln!(
                "usage: denoise_dataset train --data <v2.bin> --out <weights.bin>\n\
                 (make the dataset with: kosm-view --dump-dataset <v2.bin>)"
            );
            Ok(())
        }
    }
}

fn fit(args: &[String]) -> anyhow::Result<()> {
    let path = arg(args, "--data").unwrap_or_else(|| "target/denoise-v2.bin".into());
    let out = arg(args, "--out").unwrap_or_else(|| "target/denoise-court.bin".into());
    let size: usize = num(args, "--tile", 64);
    let hidden: usize = num(args, "--hidden", 32);
    let limit: usize = num(args, "--tiles", 2400);
    let data = dataset::Dataset::load(&path)?;
    let sizes: std::collections::BTreeSet<(u32, u32)> =
        data.samples.iter().map(|s| (s.width, s.height)).collect();
    println!(
        "{} sequences at {}",
        data.samples.len(),
        sizes
            .iter()
            .map(|(w, h)| format!("{w}x{h}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // A held-out fifth, taken as every fifth sequence rather than the tail:
    // the generator walks the resolutions in order, and a contiguous split
    // would hold out one resolution entirely.
    let held: Vec<usize> = (0..data.samples.len()).filter(|i| i % 5 == 4).collect();
    let kept: Vec<usize> = (0..data.samples.len()).filter(|i| i % 5 != 4).collect();

    let start = Instant::now();
    let train_tiles = train::prepare(&data, size, &kept, limit);
    let paired = train_tiles.iter().filter(|t| t.next.is_some()).count();
    let mut net = Kpn::new(hidden, num(args, "--seed", 1));
    println!(
        "{} training tiles of {size}x{size} ({paired} with a successor frame), \
         {} sequences held out; {} parameters",
        train_tiles.len(),
        held.len(),
        net.parameters()
    );

    let cfg = train::Fit {
        epochs: num(args, "--epochs", 30),
        batch: num(args, "--batch", 16),
        lr: num(args, "--lr", 2e-3),
        seed: num(args, "--seed", 1),
    };
    let curve = train::fit(&mut net, &train_tiles, &cfg, |e, l| {
        if e % 2 == 0 || e + 1 == cfg.epochs {
            println!(
                "  epoch {e:3}  loss {l:.6}  ({:.0}s)",
                start.elapsed().as_secs_f64()
            );
        }
    });
    net.save(&out)?;
    println!(
        "trained in {:.0} s, {} -> {} ; weights in {out} ({} bytes)",
        start.elapsed().as_secs_f64(),
        curve.first().copied().unwrap_or(0.0),
        curve.last().copied().unwrap_or(0.0),
        24 + net.parameters() * 4
    );

    let opts = kosm_render::pathtrace::PathTraceOptions::default();
    println!("\nheld-out RMSE on device-history tiles, through x/(1+x):");
    println!("  history     raw    à-trous   neural");
    for s in train::evaluate(&net, &data, &held, size, &opts) {
        println!(
            "  {:>7}  {:.5}  {:.5}  {:.5}   ({} tiles)",
            s.count, s.raw_tone, s.atrous_tone, s.neural_tone, s.tiles
        );
    }
    Ok(())
}
