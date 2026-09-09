//! `diff(a, b)`, per column.
//!
//! Because worlds are columns, this one function is three things: a snapshot
//! test when `b` is the stored frame, a sim2real gap when `a` is fitted and
//! `b` is the CAD, and a curriculum step when `b` is the next rung.
//!
//! ```
//! use kosm::prelude::*;
//! # fn main() -> anyhow::Result<()> {
//! let (model, state) = kosm::world::demo_marble();
//! let a = World::from_phyz(model, state).with_params(vec![Param::new("tilt", 0.05)]);
//! let b = PhyzStep::new(1e-3).step(&a, &Action::none()).with(&[("tilt", 0.06)]);
//! let d = diff(&a, &b);
//! assert!(!d.is_within(1e-9), "the marble moved");
//! assert!(d.is_within(1.0));
//! assert!((d.param("tilt").unwrap() - 0.01).abs() < 1e-12);
//! assert!(d.column("q").unwrap().max_abs > 0.0);
//! # Ok(()) }
//! ```

use crate::world::World;

/// One column's summary.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ColumnDiff {
    pub name: String,
    /// The largest absolute difference over the column. `f64::INFINITY` when
    /// the two columns are not the same length — a shape change is not a
    /// small number.
    pub max_abs: f64,
    pub mean_abs: f64,
    pub len: (usize, usize),
}

/// Every column, plus every knob.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Diff {
    pub columns: Vec<ColumnDiff>,
    /// `(name, b − a)`. A knob only one side has reads as its own value.
    pub params: Vec<(String, f64)>,
}

impl Diff {
    pub fn column(&self, name: &str) -> Option<&ColumnDiff> {
        self.columns.iter().find(|c| c.name == name)
    }

    pub fn param(&self, name: &str) -> Option<f64> {
        self.params.iter().find(|(n, _)| n == name).map(|(_, d)| *d)
    }

    /// The worst number in the whole diff, columns and knobs alike.
    pub fn max_abs(&self) -> f64 {
        self.columns
            .iter()
            .map(|c| c.max_abs)
            .chain(self.params.iter().map(|(_, d)| d.abs()))
            .fold(0.0, f64::max)
    }

    /// Whether every column and every knob agrees to `tol`.
    pub fn is_within(&self, tol: f64) -> bool {
        self.max_abs() <= tol
    }
}

/// The per-column difference between two worlds.
pub fn diff(a: &World, b: &World) -> Diff {
    let (ca, cb) = (a.columns(), b.columns());
    let mut columns = Vec::with_capacity(ca.len().max(cb.len()));
    for (name, va) in &ca {
        let Some((_, vb)) = cb.iter().find(|(n, _)| n == name) else {
            columns.push(ColumnDiff {
                name: name.clone(),
                max_abs: f64::INFINITY,
                mean_abs: f64::INFINITY,
                len: (va.len(), 0),
            });
            continue;
        };
        columns.push(summarise(name, va, vb));
    }
    for (name, vb) in &cb {
        if !ca.iter().any(|(n, _)| n == name) {
            columns.push(ColumnDiff {
                name: name.clone(),
                max_abs: f64::INFINITY,
                mean_abs: f64::INFINITY,
                len: (0, vb.len()),
            });
        }
    }

    let mut params: Vec<(String, f64)> = Vec::new();
    for p in &b.params {
        params.push((p.name.clone(), p.value - a.param(&p.name).unwrap_or(0.0)));
    }
    for p in &a.params {
        if !params.iter().any(|(n, _)| *n == p.name) {
            params.push((p.name.clone(), -p.value));
        }
    }
    params.sort_by(|x, y| x.0.cmp(&y.0));

    Diff { columns, params }
}

fn summarise(name: &str, a: &[f64], b: &[f64]) -> ColumnDiff {
    if a.len() != b.len() {
        return ColumnDiff {
            name: name.to_owned(),
            max_abs: f64::INFINITY,
            mean_abs: f64::INFINITY,
            len: (a.len(), b.len()),
        };
    }
    let mut max_abs = 0.0f64;
    let mut sum = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (y - x).abs();
        max_abs = max_abs.max(d);
        sum += d;
    }
    ColumnDiff {
        name: name.to_owned(),
        max_abs,
        mean_abs: if a.is_empty() { 0.0 } else { sum / a.len() as f64 },
        len: (a.len(), b.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Param, World, demo_marble};

    fn world() -> World {
        let (m, s) = demo_marble();
        World::from_phyz(m, s)
    }

    #[test]
    fn a_world_does_not_differ_from_itself() {
        let w = world();
        let d = diff(&w, &w);
        assert!(d.is_within(0.0), "{d:?}");
        assert_eq!(d.max_abs(), 0.0);
    }

    #[test]
    fn a_moved_column_shows_up_by_name() {
        let a = world();
        let mut b = a.clone();
        b.state_mut().q[5] += 0.25;
        let d = diff(&a, &b);
        assert_eq!(d.column("q").unwrap().max_abs, 0.25);
        assert_eq!(d.column("v").unwrap().max_abs, 0.0);
        assert!(!d.is_within(0.1));
        assert!(d.is_within(0.3));
    }

    #[test]
    fn knobs_diff_too_and_a_missing_one_reads_as_its_own_value() {
        let a = world().with_params(vec![Param::new("tilt", 0.05)]);
        let b = a.with(&[("tilt", 0.07), ("roll", 0.5)]);
        let d = diff(&a, &b);
        assert!((d.param("tilt").unwrap() - 0.02).abs() < 1e-12);
        assert_eq!(d.param("roll"), Some(0.5));
    }

    #[test]
    fn a_shape_change_is_not_a_small_number() {
        let a = world();
        let mut b = a.clone();
        b.lights.push(crate::world::Light { name: "key".into(), pos: [0.0; 3], power: 1.0 });
        assert!(!diff(&a, &b).is_within(1e9));
    }
}
