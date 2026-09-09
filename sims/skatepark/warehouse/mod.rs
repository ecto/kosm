//! The warehouse: the skatepark's code over a bigger level.

pub mod scene;

#[cfg(test)]
mod tests;

/// `kosm run skatepark/warehouse`.
pub fn run(args: &kosm_cli::Args) -> anyhow::Result<()> {
    let built = scene::scene(&kosm::build::Params::default())?;
    super::run_level("warehouse", built, args.out())
}
