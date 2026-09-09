//! Command selection for the `kosm-spike` binary.
//!
//! Keep process arguments at the composition root. Simulation code receives a
//! typed command instead of repeatedly querying `std::env::args`.

use std::path::PathBuf;

const DEFAULT_LEVEL: &str = "levels/marble.loon";
const DEFAULT_FRAMES: usize = 150;
const DEFAULT_SKATEPARK: &str = "levels/skatepark.loon";
const DEFAULT_COURT_LEVEL: &str = "levels/court.loon";
const DEFAULT_COVE: &str = "levels/cove.loon";

pub use crate::pool::PoolRenderer;

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Marble { level: PathBuf },
    Court { frames: Option<usize> },
    Pool { frames: usize, renderer: PoolRenderer },
    Splash { frames: usize },
    Splat { ply: PathBuf },
    Skatepark { level: PathBuf },
    Cove { level: PathBuf },
    CourtBake { level: PathBuf },
}

impl Command {
    pub fn from_env() -> anyhow::Result<Self> {
        Self::parse(std::env::args().skip(1))
    }

    fn parse(args: impl IntoIterator<Item = String>) -> anyhow::Result<Self> {
        let mut args = args.into_iter();
        let Some(first) = args.next() else {
            return Ok(Self::Marble {
                level: DEFAULT_LEVEL.into(),
            });
        };

        match first.as_str() {
            "--court" => Ok(Self::Court {
                frames: args.next().and_then(|v| v.parse().ok()),
            }),
            "--pool" => {
                let rest: Vec<String> = args.collect();
                let mut count = None;
                let mut renderer = PoolRenderer::default();
                let mut it = rest.iter();
                while let Some(arg) = it.next() {
                    if arg == "--pool-render" {
                        let value = it
                            .next()
                            .ok_or_else(|| anyhow::anyhow!("--pool-render needs a renderer"))?;
                        renderer = PoolRenderer::parse(value)?;
                    } else if count.is_none() {
                        count = Some(arg.clone());
                    }
                }
                Ok(Self::Pool {
                    frames: frames(count),
                    renderer,
                })
            }
            "--splash" => Ok(Self::Splash {
                frames: frames(args.next()),
            }),
            "--skatepark" => Ok(Self::Skatepark {
                level: args.next().map(PathBuf::from).unwrap_or_else(|| DEFAULT_SKATEPARK.into()),
            }),
            "--cove" => Ok(Self::Cove {
                level: args.next().map(PathBuf::from).unwrap_or_else(|| DEFAULT_COVE.into()),
            }),
            "--court-bake" => Ok(Self::CourtBake {
                level: args.next().map(PathBuf::from).unwrap_or_else(|| DEFAULT_COURT_LEVEL.into()),
            }),
            "--splat" => {
                let ply = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--splat needs a .ply"))?;
                Ok(Self::Splat { ply: ply.into() })
            }
            _ => Ok(Self::Marble {
                level: first.into(),
            }),
        }
    }
}

fn frames(value: Option<String>) -> usize {
    value.and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_FRAMES)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> anyhow::Result<Command> {
        Command::parse(args.iter().map(|v| (*v).to_owned()))
    }

    #[test]
    fn no_arguments_runs_the_default_marble_level() {
        assert_eq!(
            parse(&[]).unwrap(),
            Command::Marble {
                level: DEFAULT_LEVEL.into()
            }
        );
    }

    #[test]
    fn an_explicit_level_is_a_marble_command() {
        assert_eq!(
            parse(&["levels/other.loon"]).unwrap(),
            Command::Marble {
                level: "levels/other.loon".into()
            }
        );
    }

    #[test]
    fn pool_commands_keep_the_existing_frame_defaults() {
        assert_eq!(
            parse(&["--pool"]).unwrap(),
            Command::Pool { frames: 150, renderer: PoolRenderer::Legacy }
        );
        assert_eq!(
            parse(&["--splash", "24"]).unwrap(),
            Command::Splash { frames: 24 }
        );
        assert_eq!(
            parse(&["--pool", "not-a-number"]).unwrap(),
            Command::Pool { frames: 150, renderer: PoolRenderer::Legacy }
        );
    }

    #[test]
    fn the_pool_renderer_is_legacy_unless_asked_for() {
        assert_eq!(
            parse(&["--pool", "3", "--pool-render", "kosm"]).unwrap(),
            Command::Pool { frames: 3, renderer: PoolRenderer::Kosm }
        );
        assert_eq!(
            parse(&["--pool", "--pool-render", "legacy"]).unwrap(),
            Command::Pool { frames: 150, renderer: PoolRenderer::Legacy }
        );
        assert!(parse(&["--pool", "--pool-render", "opengl"]).is_err());
        assert!(parse(&["--pool", "--pool-render"]).is_err());
    }

    #[test]
    fn cove_defaults_to_the_bundled_level() {
        assert_eq!(parse(&["--cove"]).unwrap(), Command::Cove { level: DEFAULT_COVE.into() });
        assert_eq!(parse(&["--cove", "levels/other.loon"]).unwrap(), Command::Cove { level: "levels/other.loon".into() });
    }

    #[test]
    fn court_bake_defaults_to_the_bundled_level() {
        assert_eq!(parse(&["--court-bake"]).unwrap(), Command::CourtBake { level: DEFAULT_COURT_LEVEL.into() });
        assert_eq!(parse(&["--court-bake", "levels/x.loon"]).unwrap(), Command::CourtBake { level: "levels/x.loon".into() });
    }

    #[test]
    fn court_frames_default_to_the_scene() {
        assert_eq!(parse(&["--court"]).unwrap(), Command::Court { frames: None });
        assert_eq!(parse(&["--court", "30"]).unwrap(), Command::Court { frames: Some(30) });
    }

    #[test]
    fn splat_requires_a_path() {
        assert_eq!(
            parse(&["--splat", "scan.ply"]).unwrap(),
            Command::Splat {
                ply: "scan.ply".into()
            }
        );
        assert_eq!(
            parse(&["--splat"]).unwrap_err().to_string(),
            "--splat needs a .ply"
        );
    }
}
