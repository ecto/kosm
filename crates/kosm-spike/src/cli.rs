//! Command selection for the `kosm-spike` binary.
//!
//! Keep process arguments at the composition root. Simulation code receives a
//! typed command instead of repeatedly querying `std::env::args`.

use std::path::PathBuf;

const DEFAULT_LEVEL: &str = "levels/marble.loon";
const DEFAULT_FRAMES: usize = 150;

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Marble { level: PathBuf },
    Court { frames: Option<usize> },
    Pool { frames: usize },
    Splash { frames: usize },
    Splat { ply: PathBuf },
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
            "--pool" => Ok(Self::Pool {
                frames: frames(args.next()),
            }),
            "--splash" => Ok(Self::Splash {
                frames: frames(args.next()),
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
        assert_eq!(parse(&["--pool"]).unwrap(), Command::Pool { frames: 150 });
        assert_eq!(
            parse(&["--splash", "24"]).unwrap(),
            Command::Splash { frames: 24 }
        );
        assert_eq!(
            parse(&["--pool", "not-a-number"]).unwrap(),
            Command::Pool { frames: 150 }
        );
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
