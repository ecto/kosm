//! `kosm run <path>`, `kosm list`.
//!
//! The registry is `kosm_cli`'s, generated from the tree under `sims/`.

use kosm_cli::SIMS;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("list") | None => {
            list();
            Ok(())
        }
        Some("run") => {
            let Some(path) = args.get(1) else {
                list();
                anyhow::bail!("`kosm run` needs a sim path");
            };
            kosm_cli::dispatch(path, &args[2..])
        }
        Some(other) => {
            list();
            anyhow::bail!("unknown command `{other}`; try `kosm run <path>` or `kosm list`")
        }
    }
}

fn list() {
    println!("sims:");
    for (path, _) in SIMS {
        println!("  {path}");
    }
}
