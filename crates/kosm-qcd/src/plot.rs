//! Small painter-drawn plots: axes with ticks, points with error bars,
//! curves, histograms, and a complex-plane scatter.

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Stroke, pos2, vec2};

pub const INK: Color32 = Color32::from_gray(200);
pub const FAINT: Color32 = Color32::from_gray(90);
pub const ACCENT: Color32 = Color32::from_rgb(255, 140, 60);
pub const COOL: Color32 = Color32::from_rgb(90, 160, 255);

pub struct Axes {
    pub rect: Rect,
    pub x: (f64, f64),
    pub y: (f64, f64),
    pub painter: egui::Painter,
}

impl Axes {
    /// Allocate a plot of `height` and draw its frame, ticks and labels.
    pub fn new(
        ui: &mut egui::Ui,
        height: f32,
        x: (f64, f64),
        y: (f64, f64),
        x_label: &str,
        y_label: &str,
    ) -> Self {
        let (outer, _) =
            ui.allocate_exact_size(vec2(ui.available_width(), height), egui::Sense::hover());
        let rect = Rect::from_min_max(outer.min + vec2(38.0, 6.0), outer.max - vec2(6.0, 22.0));
        let painter = ui.painter_at(outer);
        let axes = Self {
            rect,
            x,
            y,
            painter,
        };
        axes.painter
            .rect_stroke(rect, 2.0, Stroke::new(1.0, FAINT), egui::StrokeKind::Inside);
        let font = FontId::proportional(10.0);
        for t in ticks(x) {
            let p = axes.to(t, y.0);
            axes.painter
                .line_segment([p, p - vec2(0.0, 4.0)], Stroke::new(1.0, FAINT));
            axes.painter.text(
                p + vec2(0.0, 2.0),
                Align2::CENTER_TOP,
                fmt(t),
                font.clone(),
                INK,
            );
        }
        for t in ticks(y) {
            let p = axes.to(x.0, t);
            axes.painter
                .line_segment([p, p + vec2(4.0, 0.0)], Stroke::new(1.0, FAINT));
            axes.painter.text(
                p - vec2(3.0, 0.0),
                Align2::RIGHT_CENTER,
                fmt(t),
                font.clone(),
                INK,
            );
        }
        axes.painter.text(
            pos2(rect.center().x, outer.max.y - 1.0),
            Align2::CENTER_BOTTOM,
            x_label,
            font.clone(),
            INK,
        );
        axes.painter.text(
            rect.min + vec2(4.0, 2.0),
            Align2::LEFT_TOP,
            y_label,
            font,
            INK,
        );
        axes
    }

    pub fn to(&self, x: f64, y: f64) -> Pos2 {
        let fx = ((x - self.x.0) / (self.x.1 - self.x.0)) as f32;
        let fy = ((y - self.y.0) / (self.y.1 - self.y.0)) as f32;
        pos2(
            self.rect.left() + fx * self.rect.width(),
            self.rect.bottom() - fy * self.rect.height(),
        )
    }

    pub fn point(&self, x: f64, y: f64, err: f64, color: Color32) {
        let p = self.to(x, y);
        if err > 0.0 {
            let (a, b) = (self.to(x, y - err), self.to(x, y + err));
            self.painter.line_segment([a, b], Stroke::new(1.0, color));
        }
        self.painter.circle_filled(p, 2.5, color);
    }

    pub fn curve(&self, f: impl Fn(f64) -> f64, color: Color32) {
        let pts: Vec<Pos2> = (0..=80)
            .map(|i| self.x.0 + (self.x.1 - self.x.0) * i as f64 / 80.0)
            .map(|x| self.to(x, f(x)))
            .filter(|p| self.rect.expand(1.0).contains(*p))
            .collect();
        self.painter
            .add(egui::Shape::line(pts, Stroke::new(1.5, color)));
    }

    pub fn vline(&self, x: f64, color: Color32) {
        self.painter.line_segment(
            [self.to(x, self.y.0), self.to(x, self.y.1)],
            Stroke::new(1.0, color),
        );
    }

    /// Histogram bars over `bins` equal bins spanning the x range.
    pub fn bars(&self, counts: &[f64], color: Color32) {
        let w = (self.x.1 - self.x.0) / counts.len() as f64;
        for (i, &c) in counts.iter().enumerate() {
            let x0 = self.x.0 + i as f64 * w;
            let r = Rect::from_two_pos(self.to(x0 + 0.08 * w, 0.0), self.to(x0 + 0.92 * w, c));
            self.painter.rect_filled(r, 0.0, color);
        }
    }
}

pub fn histogram(data: impl Iterator<Item = f64>, lo: f64, hi: f64, bins: usize) -> Vec<f64> {
    let mut h = vec![0.0; bins];
    for v in data {
        let i = ((v - lo) / (hi - lo) * bins as f64).floor();
        if i >= 0.0 && (i as usize) < bins {
            h[i as usize] += 1.0;
        }
    }
    h
}

fn ticks((lo, hi): (f64, f64)) -> Vec<f64> {
    let span = hi - lo;
    if span <= 0.0 || !span.is_finite() {
        return vec![];
    }
    let raw = span / 4.0;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|m| m * mag)
        .find(|s| span / s <= 5.0)
        .unwrap_or(10.0 * mag);
    let mut t = (lo / step).ceil() * step;
    let mut out = vec![];
    while t <= hi + 1e-9 * span {
        out.push(t);
        t += step;
    }
    out
}

fn fmt(v: f64) -> String {
    if v == 0.0 {
        "0".into()
    } else if v.abs() >= 100.0 {
        format!("{v:.0}")
    } else if v.abs() >= 1.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}
