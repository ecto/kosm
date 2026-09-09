//! The room journal — an append-only log of observations of one place.
//!
//! The map is not a snapshot; it is this journal, replayed. Every entry is an
//! observation event (today: a head-scan at a standing position, with the
//! measured head pose per frame). Fusion is a pure function of the journal;
//! waking up is opening the journal and appending; crashing loses at most the
//! entry in flight. Ported from `scripts/room_journal.py` after R1/R2 proved
//! the design on the robot (see memory `room-r1-two-stop`).
//!
//! # Layout
//!
//! ```text
//! room/
//!   journal.jsonl          one JSON entry per line, append-only
//!   blobs/<seq 04d>/<name> raw frames, immutable once written
//! ```
//!
//! Rules, unchanged from the prototype:
//! * append-only; a bad entry is superseded by a later one, never edited;
//! * blobs are written **before** the journal line that references them — the
//!   line is the commit point (and is fsynced);
//! * anything derived (poses, clouds, TSDF) lives *outside* the journal
//!   directory and can always be regenerated.

use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One captured vantage: a stereo frame + depth + the *measured* head pose.
///
/// `cmd_pitch`/`cmd_yaw` are `None` for continuous scans (there is no single
/// command per frame) and informational otherwise. The pose is authoritative:
/// the head is a slow, ~80%-gain tracker and commanded angles routinely miss
/// by tenths of a radian.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Vantage {
    pub name: String,
    pub cmd_pitch: Option<f64>,
    pub cmd_yaw: Option<f64>,
    pub head_pose: Option<HeadPoseRecord>,
    /// Blob-relative path of the NV12 combined stereo frame.
    pub rgb: String,
    /// Blob-relative path of the u16 depth frame.
    pub depth: String,
    pub rgb_wh: [u32; 2],
    pub depth_wh: [u32; 2],
    /// Camera header stamp, seconds.
    pub stamp: f64,
}

/// Measured head pose: position metres, orientation quaternion xyzw.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HeadPoseRecord {
    pub pos: [f64; 3],
    pub quat: [f64; 4],
}

impl HeadPoseRecord {
    /// Yaw (+left) and pitch (+down at the neck's convention) in radians,
    /// straight from the quaternion — the audit primitive that caught the
    /// phantom "yaw clamp".
    pub fn yaw_pitch(&self) -> (f64, f64) {
        let [x, y, z, w] = self.quat;
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
        (yaw, pitch)
    }
}

/// One journal entry. `stop` labels the standing position (`s0`, `s1`, …);
/// `odom` is the planar base odometry (`rt/odometer_state`: x, y, theta as
/// `pos = [x, y, 0]`, `quat` = yaw) at capture, or `None` when no message had
/// arrived — every pre-2026-08-17 entry is `None` because the subscribers
/// declared the wrong ROS type (`nav_msgs/Odometry`), not because the topic
/// was silent.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Entry {
    pub seq: u64,
    #[serde(rename = "type")]
    pub kind: String,
    pub stop: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scan: Option<String>,
    pub odom: Option<HeadPoseRecord>,
    pub vantages: Vec<Vantage>,
    #[serde(default)]
    pub wall: f64,
}

/// The journal on disk. Cheap to open; state lives in the files.
pub struct RoomJournal {
    root: PathBuf,
}

impl RoomJournal {
    pub fn open(root: impl Into<PathBuf>) -> std::io::Result<Self> {
        let root = root.into();
        create_dir_all(root.join("blobs"))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn journal_path(&self) -> PathBuf {
        self.root.join("journal.jsonl")
    }

    /// Every committed entry, in order. Malformed lines are an error, not a
    /// skip: a journal that cannot be read completely should be looked at,
    /// not silently truncated.
    pub fn entries(&self) -> std::io::Result<Vec<Entry>> {
        let path = self.journal_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for line in BufReader::new(File::open(path)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            out.push(serde_json::from_str(&line).map_err(std::io::Error::other)?);
        }
        Ok(out)
    }

    pub fn next_seq(&self) -> std::io::Result<u64> {
        Ok(self.entries()?.last().map_or(0, |e| e.seq + 1))
    }

    /// The blob directory for entry `seq`, created if absent. Write blobs
    /// here first; then commit the entry referencing them.
    pub fn blob_dir(&self, seq: u64) -> std::io::Result<PathBuf> {
        let d = self.root.join("blobs").join(format!("{seq:04}"));
        create_dir_all(&d)?;
        Ok(d)
    }

    /// Commit an entry: one line appended, flushed, fsynced. The blobs it
    /// references must already exist.
    pub fn append(&self, entry: &Entry) -> std::io::Result<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.journal_path())?;
        let mut line = serde_json::to_string(entry).map_err(std::io::Error::other)?;
        line.push('\n');
        f.write_all(line.as_bytes())?;
        f.flush()?;
        f.sync_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vantage(name: &str) -> Vantage {
        Vantage {
            name: name.into(),
            cmd_pitch: Some(0.25),
            cmd_yaw: Some(0.5),
            head_pose: Some(HeadPoseRecord {
                pos: [0.06, 0.01, 0.876],
                quat: [0.0, 0.0, 0.0, 1.0],
            }),
            rgb: format!("{name}.nv12"),
            depth: format!("{name}.u16"),
            rgb_wh: [544, 896],
            depth_wh: [544, 448],
            stamp: 123.4,
        }
    }

    #[test]
    fn round_trips_and_appends() {
        let dir = std::env::temp_dir().join(format!("room-journal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let j = RoomJournal::open(&dir).unwrap();
        assert_eq!(j.next_seq().unwrap(), 0);

        let e = Entry {
            seq: 0,
            kind: "sweep".into(),
            stop: "s0".into(),
            scan: None,
            odom: None,
            vantages: vec![vantage("up_L")],
            wall: 1.0,
        };
        j.append(&e).unwrap();
        let back = j.entries().unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].stop, "s0");
        assert_eq!(back[0].vantages[0].rgb, "up_L.nv12");
        assert_eq!(j.next_seq().unwrap(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn python_journal_lines_parse() {
        // A line as scripts/room_sweep.py wrote it on the robot (trimmed
        // vantage list). The Rust port must read the Python journal.
        let line = r#"{"seq": 0, "type": "sweep", "stop": "s0", "odom": null, "vantages": [{"name": "up_L", "cmd_pitch": -0.1, "cmd_yaw": 0.5, "head_pose": {"pos": [0.06, 0.01, 0.87], "quat": [0.02, -0.03, 0.19, 0.97]}, "rgb": "up_L.nv12", "depth": "up_L.u16", "rgb_wh": [544, 896], "depth_wh": [544, 448], "stamp": 1786.5}], "wall": 1786809.0}"#;
        let e: Entry = serde_json::from_str(line).unwrap();
        assert_eq!(e.kind, "sweep");
        let (yaw, _pitch) = e.vantages[0].head_pose.as_ref().unwrap().yaw_pitch();
        assert!(yaw > 0.3, "up_L should look left, got {yaw}");
    }

    #[test]
    fn yaw_pitch_identity() {
        let p = HeadPoseRecord {
            pos: [0.0; 3],
            quat: [0.0, 0.0, 0.0, 1.0],
        };
        let (y, pi) = p.yaw_pitch();
        assert!(y.abs() < 1e-9 && pi.abs() < 1e-9);
    }
}
