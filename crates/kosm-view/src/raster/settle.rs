//! Settling into the reference.
//!
//! The raster tier is what shows while you move. The path tracer is what the
//! picture *becomes* when you stop — and the crossing between them is one
//! number.
//!
//! ```text
//! blend = clamp((spp − 4) / 24, 0, 1)
//! shown = lerp(raster, traced, blend)
//! ```
//!
//! Four samples a pixel is where the tracer's frame stops being noise; twenty
//! eight is where it stops being noticeably noisier than the raster's. Between
//! them the reference fades up. **Any move resets it to zero on the frame it
//! happens** — not over a few frames, not eased — because a blend that decayed
//! would smear a converged still of the old pose across the new one, which is
//! exactly the ghost the whole tier exists to avoid. What the player sees the
//! instant they touch the mouse is the raster frame, and it is sharp.
//!
//! ```
//! use kosm_view::raster::Settle;
//! let mut s = Settle::default();
//! assert_eq!(s.blend(), 0.0);
//! s.traced(30.0);                 // the tracer reports 30 samples a pixel
//! assert_eq!(s.blend(), 1.0);
//! s.moved();                      // the player turns
//! assert_eq!(s.blend(), 0.0, "a move is felt on the frame it happens");
//! ```

/// Where the blend starts and where it finishes, in samples a pixel.
pub const SPP_FLOOR: f32 = 4.0;
pub const SPP_FULL: f32 = 28.0;

/// The crossing between the two tiers.
#[derive(Clone, Copy, Debug, Default)]
pub struct Settle {
    spp: f32,
    /// Whether the blend is switched off entirely — `--no-settle`, and the
    /// state a level under a projection the raster cannot draw is in.
    off: bool,
    /// How many frames in a row nothing has moved.
    still_frames: u64,
    /// Simulated seconds since the last move.
    still_s: f64,
}

impl Settle {
    /// A blend that never rises: `--no-settle`.
    pub fn disabled() -> Self {
        Self { off: true, ..Self::default() }
    }

    /// The tracer has resolved a frame at this many samples a pixel.
    pub fn traced(&mut self, spp: f32) {
        if !self.off {
            self.spp = spp;
        }
    }

    /// The camera or the world moved. Everything the tracer has accumulated
    /// is about a pose that is gone.
    pub fn moved(&mut self) {
        self.spp = 0.0;
        self.still_frames = 0;
        self.still_s = 0.0;
    }

    /// One frame passed with nothing moving.
    pub fn still(&mut self, dt: f64) {
        self.still_frames += 1;
        self.still_s += dt.max(0.0);
    }

    /// How much of the presented frame is the reference's.
    pub fn blend(&self) -> f32 {
        if self.off {
            return 0.0;
        }
        ((self.spp - SPP_FLOOR) / (SPP_FULL - SPP_FLOOR)).clamp(0.0, 1.0)
    }

    /// Whether the presented frame is the raster's alone — which is when it
    /// can be handed over as a texture with nothing crossing the bus.
    pub fn is_raster_only(&self) -> bool {
        self.blend() <= 0.0
    }

    /// Whether the reference has fully arrived.
    pub fn is_settled(&self) -> bool {
        self.blend() >= 1.0
    }

    /// How long the picture has been still, simulated seconds.
    pub fn still_for(&self) -> f64 {
        self.still_s
    }

    /// The samples the tracer last reported.
    pub fn spp(&self) -> f32 {
        self.spp
    }

    /// The line the pace log prints.
    pub fn report(&self, fps: f64, latency_ms: f64) -> SettleReport {
        SettleReport { fps, latency_ms, blend: self.blend(), spp: self.spp }
    }
}

/// What the pace line says about the crossing.
#[derive(Clone, Copy, Debug)]
pub struct SettleReport {
    pub fps: f64,
    pub latency_ms: f64,
    pub blend: f32,
    pub spp: f32,
}

impl std::fmt::Display for SettleReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "raster {:.1} fps, {:.0} ms presented, blend {:.2} at {:.1} spp",
            self.fps, self.latency_ms, self.blend, self.spp
        )
    }
}

/// One presented frame: the raster's bytes, the reference's, and the blend.
///
/// Both are already sRGB-encoded, and the mix is done **there** rather than in
/// linear light on purpose: the two tiers tonemap with the same ACES curve and
/// the same sRGB transfer, so a pixel that agrees agrees at every blend and a
/// pixel that does not fades between two displayable values. Mixing in linear
/// and re-encoding would be one more place for the two to disagree.
///
/// `traced` may be smaller than `raster` — the tracer runs at the budget's
/// size and the raster at the window's — in which case it is sampled with
/// bilinear filtering on the way in.
pub fn present(
    raster: &[u8],
    raster_size: (u32, u32),
    traced: &[u8],
    traced_size: (u32, u32),
    blend: f32,
) -> Vec<u8> {
    let (w, h) = raster_size;
    let mut out = raster.to_vec();
    let b = blend.clamp(0.0, 1.0);
    if b <= 0.0 || traced.is_empty() || traced_size.0 == 0 || traced_size.1 == 0 {
        return out;
    }
    let (tw, th) = traced_size;
    for y in 0..h {
        // pixel centres, so a 1:1 upscale is the identity
        let sy = ((y as f32 + 0.5) * th as f32 / h as f32 - 0.5).clamp(0.0, th as f32 - 1.0);
        let (y0, fy) = (sy.floor() as u32, sy - sy.floor());
        let y1 = (y0 + 1).min(th - 1);
        for x in 0..w {
            let sx = ((x as f32 + 0.5) * tw as f32 / w as f32 - 0.5).clamp(0.0, tw as f32 - 1.0);
            let (x0, fx) = (sx.floor() as u32, sx - sx.floor());
            let x1 = (x0 + 1).min(tw - 1);
            let at = |xx: u32, yy: u32, c: usize| {
                traced[((yy * tw + xx) as usize) * 4 + c] as f32
            };
            let dst = ((y * w + x) as usize) * 4;
            for c in 0..3 {
                let top = at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
                let bot = at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
                let t = top * (1.0 - fy) + bot * fy;
                let r = out[dst + c] as f32;
                out[dst + c] = (r + (t - r) * b).round().clamp(0.0, 255.0) as u8;
            }
            out[dst + 3] = 255;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The design's curve: nothing under four samples, everything at
    /// twenty-eight, linear between.
    #[test]
    fn the_blend_is_the_designs_curve() {
        let mut s = Settle::default();
        for (spp, want) in [(0.0, 0.0), (4.0, 0.0), (16.0, 0.5), (28.0, 1.0), (200.0, 1.0)] {
            s.traced(spp);
            assert!((s.blend() - want).abs() < 1e-6, "{spp} spp gave {}", s.blend());
        }
    }

    /// **A move is felt on the frame it happens**, not eased away over
    /// several. This is the whole reason the tier exists.
    #[test]
    fn a_move_drops_the_blend_in_one_frame() {
        let mut s = Settle::default();
        s.traced(120.0);
        assert!(s.is_settled());
        s.moved();
        assert_eq!(s.blend(), 0.0);
        assert!(s.is_raster_only());
        // and the still clock started over
        s.still(0.1);
        assert!((s.still_for() - 0.1).abs() < 1e-9);
    }

    /// **The design's two deadlines, in seconds.**
    ///
    /// Standing at sixty frames a second the tracer folds one pass a frame —
    /// which is what `raster_worker` does: a raster frame, then one
    /// `Tracer::pass_at` while nothing is moving — so the blend must reach
    /// one inside three seconds of stillness and drop to zero inside one
    /// frame of a walk. The loop is written out rather than asserted on the
    /// curve, because the deadline is about the *rate* the samples arrive at
    /// and not about where the curve crosses.
    #[test]
    fn the_blend_settles_in_three_seconds_and_breaks_in_one_frame() {
        let dt = 1.0 / 60.0;
        let mut s = Settle::default();
        let mut spp = 0.0f32;
        let mut settled_at = None;
        for frame in 0..(3.0 / dt) as u32 {
            spp += 1.0; // one pass folded per still frame
            s.still(dt);
            s.traced(spp);
            if s.is_settled() && settled_at.is_none() {
                settled_at = Some(s.still_for());
            }
            assert!(frame < 1000);
        }
        let at = settled_at.expect("the blend never reached one in three seconds");
        assert!(at <= 3.0, "the reference took {at:.2} s to arrive, over the design's three");
        // and it is well inside it: twenty-eight passes at sixty a second
        assert!(at < 0.6, "{at:.2} s is later than twenty-eight frames");

        // one step of a walk, and the reference is gone on that frame
        s.moved();
        assert_eq!(s.blend(), 0.0, "a walk did not drop the blend on its own frame");
        assert!(s.is_raster_only());
    }

    /// `--no-settle` never leaves the raster.
    #[test]
    fn the_blend_can_be_switched_off() {
        let mut s = Settle::disabled();
        s.traced(1000.0);
        assert_eq!(s.blend(), 0.0);
        assert!(s.is_raster_only());
    }

    /// At blend zero the presented frame is the raster's bytes exactly; at
    /// one it is the reference's; and a same-size upscale is the identity.
    #[test]
    fn the_present_is_a_lerp_that_reaches_both_ends() {
        let raster = vec![10u8; 4 * 4 * 4];
        let traced = vec![250u8; 4 * 4 * 4];
        assert_eq!(present(&raster, (4, 4), &traced, (4, 4), 0.0), raster);
        let full = present(&raster, (4, 4), &traced, (4, 4), 1.0);
        assert!(full.chunks(4).all(|p| p[0] == 250 && p[3] == 255));
        let half = present(&raster, (4, 4), &traced, (4, 4), 0.5);
        assert!(half.chunks(4).all(|p| p[0] == 130), "half way is {:?}", &half[..4]);
    }

    /// A smaller traced frame is upscaled bilinearly, and a constant one
    /// stays constant — which is the check that the sample positions are
    /// pixel centres and not corners.
    #[test]
    fn a_smaller_reference_upscales_without_a_ramp() {
        let raster = vec![0u8; 8 * 8 * 4];
        let traced = vec![200u8; 4 * 4 * 4];
        let out = present(&raster, (8, 8), &traced, (4, 4), 1.0);
        for p in out.chunks(4) {
            assert_eq!(p[0], 200, "an upscaled constant is constant");
        }
    }
}
