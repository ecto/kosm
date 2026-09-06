//! A denoiser trained on the court's own reference renders.
//!
//! The à-trous filter in [`kosm_render::pathtrace::denoise`] and its device
//! twin in `kosm_render::gpu::history` are *general*: five hand-tuned
//! constants that have to be right for a kitchen, a skatepark and a gym at
//! once. This module is the other option. The court is a level we own, its
//! path tracer is a level we own, and a reference render of it is exact
//! ground truth we can make as much of as we like. So: sample the level,
//! render each sample noisy and converged, and fit a small kernel-predicting
//! network to the difference.
//!
//! Three pieces, in the order they run:
//!
//! * [`dataset`] — the file format, and the tiles cut out of it. The
//!   *generation* is `kosm-view`'s `--dump-dataset`, not this crate's: a v2
//!   sample is the device history read back after the same passes the window
//!   runs, and only the thing that drives the device can make one. That move
//!   is the whole lesson of v1 — see [`dataset`]'s module docs.
//! * [`kpn`] — the network: three 3x3 convolutions predicting a normalised
//!   5x5 filter kernel per pixel, applied to the *demodulated* illumination
//!   exactly where the à-trous filter would have run.
//! * [`train`] — the fit, with `tang_train`'s `Parameter` and `ModuleAdam`
//!   over hand-written convolution kernels; see [`kpn`] for why the layers
//!   are ours rather than `tang_train::Conv2d`'s. The loss is L1 through a
//!   tone curve, plus gradients, plus a temporal consistency term that only
//!   a paired dataset can express.
//!
//! The weights come out as the `.bin` `kosm_render::gpu::neural::Weights`
//! loads, so the thing trained here is the thing the viewport runs.

pub mod dataset;
pub mod kpn;
pub mod train;
