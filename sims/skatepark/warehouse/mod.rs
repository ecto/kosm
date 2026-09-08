//! The warehouse: the skatepark's code over `warehouse.loon`.

#[cfg(test)]
mod tests;

/// `kosm run skatepark/warehouse`.
pub fn run(args: &kosm_cli::Args) -> anyhow::Result<()> {
    let level = kosm::scene::AuthoredScene::bundled_path("warehouse.loon");
    super::run_level(&level, args.out())
}
