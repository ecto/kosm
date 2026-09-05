//! Make the training set, fit the denoiser, and score it against à-trous.
//!
//! ```text
//! cargo run --release -p kosm-spike --example denoise_dataset -- \
//!     generate --out target/denoise.bin --samples 50
//! cargo run --release -p kosm-spike --example denoise_dataset -- \
//!     train --data target/denoise.bin --out target/denoise-court.bin
//! ```
//!
//! `generate` is the expensive half — a 1024-spp reference per sample — and
//! `train` reads the file it writes, so the two are separate subcommands and
//! not one run.

use std::time::Instant;

use kosm_spike::court::CourtScene;
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
        Some("generate") => generate(&args),
        Some("train") => fit(&args),
        _ => {
            eprintln!("usage: denoise_dataset <generate|train> [options]");
            Ok(())
        }
    }
}

fn generate(args: &[String]) -> anyhow::Result<()> {
    let out = arg(args, "--out").unwrap_or_else(|| "target/denoise.bin".into());
    let cfg = dataset::Config {
        width: num(args, "--width", 320),
        height: num(args, "--height", 180),
        samples: num(args, "--samples", 50),
        reference_spp: num(args, "--ref-spp", 1024),
        t_end: num(args, "--t-end", 4.0),
        seed: num(args, "--seed", 0x5EED_C0FF_EE12_3456),
    };
    let scene = CourtScene::bundled()?;
    let start = Instant::now();
    let mut last = Instant::now();
    let data = dataset::generate(&scene, &cfg, |i, n| {
        if i > 0 {
            let per = start.elapsed().as_secs_f64() / i as f64;
            eprintln!(
                "  sample {i}/{n}  {:.1}s each, {:.0}s left",
                last.elapsed().as_secs_f64(),
                per * (n - i) as f64
            );
        }
        last = Instant::now();
    })?;
    let secs = start.elapsed().as_secs_f64();
    data.save(&out)?;
    println!(
        "{} samples of {}x{} at {} spp reference -> {} ({:.1} MB) in {:.1} s",
        data.samples.len(),
        cfg.width,
        cfg.height,
        cfg.reference_spp,
        out,
        data.bytes() as f64 / 1e6,
        secs
    );
    Ok(())
}

fn fit(args: &[String]) -> anyhow::Result<()> {
    let path = arg(args, "--data").unwrap_or_else(|| "target/denoise.bin".into());
    let out = arg(args, "--out").unwrap_or_else(|| "target/denoise-court.bin".into());
    let size: usize = num(args, "--tile", 64);
    let hidden: usize = num(args, "--hidden", 32);
    let data = dataset::Dataset::load(&path)?;
    println!(
        "{} samples of {}x{}",
        data.samples.len(),
        data.width,
        data.height
    );

    // A held-out fifth, taken as every fifth sample rather than the tail:
    // consecutive samples share a simulated instant's neighbourhood and a
    // contiguous split would hand the network the easy half.
    let held: Vec<usize> = (0..data.samples.len()).filter(|i| i % 5 == 4).collect();
    let kept: Vec<usize> = (0..data.samples.len()).filter(|i| i % 5 != 4).collect();

    let start = Instant::now();
    let train_tiles = train::prepare(&data, size, &kept);
    let mut net = Kpn::new(hidden, num(args, "--seed", 1));
    println!(
        "{} training tiles of {size}x{size}, {} held out; {} parameters",
        train_tiles.len(),
        held.len(),
        net.parameters()
    );

    let cfg = train::Fit {
        epochs: num(args, "--epochs", 60),
        batch: num(args, "--batch", 8),
        lr: num(args, "--lr", 2e-3),
        seed: num(args, "--seed", 1),
    };
    let curve = train::fit(&mut net, &train_tiles, &cfg, |e, l| {
        if e % 2 == 0 || e + 1 == cfg.epochs {
            println!("  epoch {e:3}  loss {l:.6}  ({:.0}s)", start.elapsed().as_secs_f64());
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
    println!("\nheld-out RMSE, through x/(1+x):");
    println!("  spp     raw    à-trous   neural");
    for s in train::evaluate(&net, &data, &held, size, &opts) {
        println!(
            "  {:>3}  {:.5}  {:.5}  {:.5}   ({} tiles)",
            s.count, s.raw_tone, s.atrous_tone, s.neural_tone, s.tiles
        );
    }
    Ok(())
}
