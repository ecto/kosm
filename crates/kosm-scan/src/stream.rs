//! The map, on the wire: 16³-voxel bricks of surface, versioned and paged.
//!
//! A map is not a frame. Pose streams at 60 Hz and every packet replaces the
//! last one, so the dome's idiom — repaint wholesale, lose one, the next is
//! along in 16 ms — costs nothing. A lab-sized mesh is tens of megabytes and
//! *accumulates*: the interesting case is a robot standing still while its
//! camera adds a corner of the room every few seconds, and repainting the
//! whole room to deliver a corner is the wrong shape by three orders of
//! magnitude.
//!
//! So the unit is a **brick**: a cube of [`BRICK`]³ voxels, meshed on its own.
//! Fusion marks a brick dirty when a voxel inside it changes; a dirty brick
//! re-meshes locally and ships. Everything else on the wire follows from
//! wanting that to be reliable without an ACK channel:
//!
//! * **Versions, not sequence numbers.** Each brick carries its own
//!   monotonically increasing version. A page from an older version of a
//!   brick that has since been re-meshed is discarded on arrival rather than
//!   assembled into a mesh nobody asked for.
//! * **Pages, because a brick is bigger than a datagram.** A dense brick runs
//!   to several KB; UDP will carry that and IP will fragment it, but one lost
//!   fragment silently destroys the whole datagram, so the loss unit gets to
//!   be ours and stays under a typical MTU. See [`PAGE`].
//! * **No ACKs, by design.** Dirty bricks send immediately, and a slow
//!   background walk re-sends every non-empty brick continuously
//!   ([`BrickWalker`]). A packet lost at 2% is healed by the next walk instead
//!   of by a retransmit protocol we would have to write, test and debug on a
//!   link where the *consequence* of loss is a hole in a wall for four
//!   seconds.
//! * **Tombstones.** A brick whose surface has been carved away sends a
//!   zero-count page, because "no packet" is indistinguishable from "no route"
//!   and space that has been observed to be empty is a real observation.
//!
//! The receiver is `MapReceiver` in Dojo; [`decode_brick`] and this
//! module's tests are the fixtures its Swift twin is checked against, the way
//! `Wire.swift` is checked against `view.rs`.

use std::collections::BTreeMap;

use phyz_math::Vec3;

use crate::mesh::TriMesh;

/// Voxels along a brick edge. 16 is the fusion grid's own natural block and
/// puts a fully-dense brick at a few KB — a handful of pages, not hundreds.
pub const BRICK: usize = 16;

/// Payload bytes per page.
///
/// Sized so header + payload clears a 1500-byte Ethernet MTU with room for
/// IPv6 and any tunnel a laptop might be behind. Bigger pages would mean
/// fewer packets and IP fragmentation doing the paging for us — which is
/// exactly the arrangement where one lost fragment costs a whole brick.
pub const PAGE: usize = 1200;

/// `"BRIK"`, little-endian, the way every tag in this house is written.
pub const BRICK_TAG: u32 = 0x4B49_5242;

/// `"MAPI"` — the map's own description, sent before and among the bricks.
pub const INFO_TAG: u32 = 0x4950_414D;

/// Where a brick sits, in bricks.
///
/// `i16`, so the addressable world is ±32767 bricks — about ±10 km at a 2 cm
/// cell. The alternative (`i32`) doubles the field for a range no captured
/// place will ever want.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BrickKey {
    pub i: i16,
    pub j: i16,
    pub k: i16,
}

impl BrickKey {
    /// The brick containing a point, given the brick span in metres.
    pub fn containing(p: Vec3, span: f64) -> BrickKey {
        BrickKey {
            i: (p.x / span).floor() as i16,
            j: (p.y / span).floor() as i16,
            k: (p.z / span).floor() as i16,
        }
    }

    /// The low corner of the brick, in metres.
    pub fn origin(&self, span: f64) -> Vec3 {
        Vec3::new(self.i as f64 * span, self.j as f64 * span, self.k as f64 * span)
    }
}

/// One brick's surface, quantized and ready to send.
///
/// Vertices are `u16` in a window that is **twice** the brick span, anchored
/// half a brick below the brick's own origin. The doubling is what lets a
/// triangle assigned to this brick reach into its neighbours without being
/// clipped: assignment is by centroid, so a triangle can overhang by up to
/// half its own size in any direction, and [`brick_trimesh`] guarantees no
/// edge exceeds one brick span. At a 2 cm cell the window is 0.64 m over
/// 65536 steps — 10 µm, well under the sub-mm the format promises.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BrickMesh {
    pub verts: Vec<[u16; 3]>,
    pub tris: Vec<[u16; 3]>,
}

impl BrickMesh {
    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    /// Quantization window: low corner, and metres per step.
    ///
    /// One function so the encoder and every decoder — including the Swift
    /// one — are reading the same two lines rather than two descriptions of
    /// them.
    pub fn window(key: BrickKey, span: f64) -> (Vec3, f64) {
        let lo = key.origin(span) - Vec3::splat(span * 0.5);
        (lo, span * 2.0 / 65535.0)
    }

    /// Dequantize back to metres. Used by the tests, and by any Rust consumer
    /// that wants the mesh the viewer is drawing.
    pub fn vertices(&self, key: BrickKey, span: f64) -> Vec<Vec3> {
        let (lo, step) = Self::window(key, span);
        self.verts
            .iter()
            .map(|v| {
                lo + Vec3::new(v[0] as f64 * step, v[1] as f64 * step, v[2] as f64 * step)
            })
            .collect()
    }
}

/// A whole map, cut into bricks.
#[derive(Debug, Clone, Default)]
pub struct Bricked {
    /// Metres per brick edge — `BRICK` × the grid cell.
    pub span: f64,
    /// Non-empty bricks only. Empty space is not stored and not sent; a brick
    /// that *becomes* empty is sent once as a tombstone and then forgotten.
    pub bricks: BTreeMap<BrickKey, BrickMesh>,
}

impl Bricked {
    pub fn len(&self) -> usize {
        self.bricks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bricks.is_empty()
    }
}

/// Cut a triangle mesh into bricks.
///
/// Two steps, and the first is the one that is easy to skip and expensive to
/// skip. Triangles are assigned to bricks **by centroid**, which is only
/// sound if a triangle is small relative to a brick — and the meshes arriving
/// here are not uniformly small. Surface nets output is (a cell or two), but
/// a phone scan's floor can be two enormous triangles, and
/// [`crate::props::box_mesh`]-style props certainly are. An unsplit 6 m floor
/// triangle lands in exactly one brick, quantizes to garbage against a 0.64 m
/// window, and disappears — a map with no floor, from a mesh that has one.
///
/// So: split first, at the longest edge, until no edge exceeds one brick
/// span; then bucket by centroid. The split is midpoint subdivision, which
/// keeps the surface exactly (the midpoint of an edge is on the edge) and
/// keeps the winding, which the pseudonormal sign convention downstream
/// depends on.
pub fn brick_trimesh(mesh: &TriMesh, cell: f64) -> Bricked {
    let span = cell * BRICK as f64;
    let mut bricks: BTreeMap<BrickKey, BrickMesh> = BTreeMap::new();
    // Per-brick vertex dedupe, so a shared edge does not become two vertices
    // and double the wire cost of every brick.
    let mut index: BTreeMap<BrickKey, BTreeMap<[u16; 3], u16>> = BTreeMap::new();

    let mut push = |a: Vec3, b: Vec3, c: Vec3| {
        let centroid = (a + b + c) / 3.0;
        let key = BrickKey::containing(centroid, span);
        let (lo, step) = BrickMesh::window(key, span);
        let brick = bricks.entry(key).or_default();
        let seen = index.entry(key).or_default();
        let mut ids = [0u16; 3];
        for (n, p) in [a, b, c].into_iter().enumerate() {
            let q = [
                quantize(p.x - lo.x, step),
                quantize(p.y - lo.y, step),
                quantize(p.z - lo.z, step),
            ];
            let next = seen.len();
            // 65535 vertices per brick is far past anything a 16³ block can
            // hold; the guard exists so a pathological input drops triangles
            // rather than aliasing indices into the wrong vertices.
            if next >= u16::MAX as usize {
                return;
            }
            ids[n] = *seen.entry(q).or_insert_with(|| {
                brick.verts.push(q);
                next as u16
            });
        }
        if ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2] {
            brick.tris.push(ids);
        }
    };

    for t in &mesh.triangles {
        let tri = [
            mesh.vertices[t[0] as usize],
            mesh.vertices[t[1] as usize],
            mesh.vertices[t[2] as usize],
        ];
        split_to(tri, span, &mut push);
    }
    bricks.retain(|_, b| !b.is_empty());
    Bricked { span, bricks }
}

/// Cut a fusion grid's current surface into bricks.
///
/// The live path. [`crate::FusedGrid::extract_mesh`] already returns the
/// surface-nets soup for the whole grid, so this is that, welded and bricked
/// — correct from the first day, and the place where per-brick incremental
/// re-meshing goes when a stance sweep's cost makes it worth the bookkeeping.
/// Today a sweep re-meshes the grid it just changed, which is milliseconds at
/// lab size and a bridge that cannot get out of step with itself.
pub fn brick_fused(grid: &crate::FusedGrid) -> Bricked {
    let soup = grid.extract_mesh();
    let flat: Vec<crate::stl::SoupTri> = soup
        .iter()
        .map(|t| {
            [
                [t[0].x as f32, t[0].y as f32, t[0].z as f32],
                [t[1].x as f32, t[1].y as f32, t[1].z as f32],
                [t[2].x as f32, t[2].y as f32, t[2].z as f32],
            ]
        })
        .collect();
    brick_trimesh(&TriMesh::from_soup(&flat), grid.cell)
}

fn quantize(v: f64, step: f64) -> u16 {
    (v / step).round().clamp(0.0, 65535.0) as u16
}

/// Midpoint-subdivide until every edge is within `max_edge`, calling `out`
/// for each surviving triangle.
fn split_to(tri: [Vec3; 3], max_edge: f64, out: &mut impl FnMut(Vec3, Vec3, Vec3)) {
    // Iterative, with an explicit stack. Recursion here is bounded by the
    // ratio of the biggest triangle to a brick, which for a room-sized floor
    // slab at 2 cm cells is about eight levels — fine on the stack, but the
    // depth is a property of *input data* and this runs on a bridge thread.
    let mut stack = vec![tri];
    // A triangle 2^24 times a brick span is not a scan, it is a corrupt file;
    // the cap turns that into dropped geometry instead of an OOM.
    let mut budget = 1 << 24;
    while let Some([a, b, c]) = stack.pop() {
        if budget == 0 {
            return;
        }
        budget -= 1;
        let edges = [(a - b).norm(), (b - c).norm(), (c - a).norm()];
        let longest = edges.iter().cloned().fold(0.0_f64, f64::max);
        if longest <= max_edge || !longest.is_finite() {
            out(a, b, c);
            continue;
        }
        // Split the longest edge and keep the winding: replacing edge (p,q)
        // with midpoint m yields (p, m, r) and (m, q, r), both wound the same
        // way as the parent.
        let (p, q, r) = if edges[0] >= edges[1] && edges[0] >= edges[2] {
            (a, b, c)
        } else if edges[1] >= edges[2] {
            (b, c, a)
        } else {
            (c, a, b)
        };
        let m = (p + q) * 0.5;
        stack.push([p, m, r]);
        stack.push([m, q, r]);
    }
}

// MARK: - The wire

/// Header bytes on every brick page.
///
/// tag(4) + coords(6) + version(4) + page(2) + pages(2) + offset(2) +
/// len(2) + payload_len(4).
pub const BRICK_HEADER: usize = 26;

/// Serialize a brick's payload — the thing pages are cut from.
///
/// `u16` counts, then `u16`×3 per vertex, then `u16`×3 per triangle. Indices
/// are into this brick's own vertex list and nothing else; a brick is a
/// closed little world so a lost neighbour cannot corrupt it.
pub fn brick_payload(mesh: &BrickMesh) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + mesh.verts.len() * 6 + mesh.tris.len() * 6);
    out.extend_from_slice(&(mesh.verts.len() as u16).to_le_bytes());
    out.extend_from_slice(&(mesh.tris.len() as u16).to_le_bytes());
    for v in &mesh.verts {
        for c in v {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    for t in &mesh.tris {
        for c in t {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    out
}

/// Every page of one brick, ready to `send_to`.
///
/// An empty brick produces exactly one page with a zero-length payload — the
/// tombstone. It is a real message and not an absence, which is the whole
/// reason carved space can disappear from the viewer.
pub fn encode_brick(key: BrickKey, version: u32, mesh: &BrickMesh) -> Vec<Vec<u8>> {
    // A brick with no *triangles* is empty however many stray vertices it
    // carries, and it goes out as a zero-length payload rather than as a
    // four-byte header saying zero twice. The distinction matters because
    // `payload_len == 0` is the tombstone test on the receiving end, and a
    // tombstone that does not read as one leaves carved space on screen.
    let payload = if mesh.tris.is_empty() { Vec::new() } else { brick_payload(mesh) };
    let pages = payload.len().div_ceil(PAGE).max(1);
    (0..pages)
        .map(|p| {
            let start = p * PAGE;
            let end = (start + PAGE).min(payload.len());
            let slice = &payload[start.min(payload.len())..end.max(start.min(payload.len()))];
            let mut pkt = Vec::with_capacity(BRICK_HEADER + slice.len());
            pkt.extend_from_slice(&BRICK_TAG.to_le_bytes());
            pkt.extend_from_slice(&key.i.to_le_bytes());
            pkt.extend_from_slice(&key.j.to_le_bytes());
            pkt.extend_from_slice(&key.k.to_le_bytes());
            pkt.extend_from_slice(&version.to_le_bytes());
            pkt.extend_from_slice(&(p as u16).to_le_bytes());
            pkt.extend_from_slice(&(pages as u16).to_le_bytes());
            pkt.extend_from_slice(&(start as u16).to_le_bytes());
            pkt.extend_from_slice(&(slice.len() as u16).to_le_bytes());
            pkt.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            pkt.extend_from_slice(slice);
            pkt
        })
        .collect()
}

/// One page, as it came off the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrickPage {
    pub key: BrickKey,
    pub version: u32,
    pub page: u16,
    pub pages: u16,
    pub offset: usize,
    pub payload_len: usize,
    pub bytes: Vec<u8>,
}

impl BrickPage {
    /// A brick with no surface. See [`encode_brick`] on why this is a message.
    pub fn is_tombstone(&self) -> bool {
        self.payload_len == 0
    }
}

/// Parse a page, or `None` if it is not one.
///
/// Every length is checked against the buffer that actually arrived rather
/// than against the header's claim about itself, because the header is
/// attacker-or-bug-controlled and this runs on a socket anyone on the LAN can
/// send to.
pub fn decode_brick(bytes: &[u8]) -> Option<BrickPage> {
    if bytes.len() < BRICK_HEADER || u32::from_le_bytes(bytes[0..4].try_into().ok()?) != BRICK_TAG {
        return None;
    }
    let i16at = |o: usize| i16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let u16at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let key = BrickKey { i: i16at(4), j: i16at(6), k: i16at(8) };
    let version = u32::from_le_bytes(bytes[10..14].try_into().ok()?);
    let page = u16at(14);
    let pages = u16at(16);
    let offset = u16at(18) as usize;
    let len = u16at(20) as usize;
    let payload_len = u32::from_le_bytes(bytes[22..26].try_into().ok()?) as usize;
    if pages == 0 || page >= pages || bytes.len() < BRICK_HEADER + len {
        return None;
    }
    if offset + len > payload_len {
        return None;
    }
    Some(BrickPage {
        key,
        version,
        page,
        pages,
        offset,
        payload_len,
        bytes: bytes[BRICK_HEADER..BRICK_HEADER + len].to_vec(),
    })
}

/// Parse an assembled payload back into a brick mesh.
///
/// Shares its bounds discipline with [`decode_brick`], and additionally
/// rejects a triangle whose indices do not name vertices this brick sent —
/// the one way a truncated-but-well-formed payload could reach a renderer and
/// crash it.
pub fn decode_payload(bytes: &[u8]) -> Option<BrickMesh> {
    if bytes.is_empty() {
        return Some(BrickMesh::default());
    }
    if bytes.len() < 4 {
        return None;
    }
    let nv = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
    let nt = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
    if bytes.len() < 4 + nv * 6 + nt * 6 {
        return None;
    }
    let at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let verts: Vec<[u16; 3]> =
        (0..nv).map(|n| [at(4 + n * 6), at(6 + n * 6), at(8 + n * 6)]).collect();
    let base = 4 + nv * 6;
    let tris: Vec<[u16; 3]> = (0..nt)
        .map(|n| [at(base + n * 6), at(base + 2 + n * 6), at(base + 4 + n * 6)])
        .collect();
    if tris.iter().flatten().any(|&i| i as usize >= nv) {
        return None;
    }
    Some(BrickMesh { verts, tris })
}

// MARK: - The map's own description

/// What the viewer needs before a brick means anything: the span its
/// coordinates are in, and where to find the layers this stream does not
/// carry.
///
/// Sent repeatedly rather than once at the top. A viewer that joins late — a
/// tab opened after the bridge started, which is the normal case — would
/// otherwise sit on a pile of bricks it cannot place.
#[derive(Debug, Clone, PartialEq)]
pub struct MapInfo {
    pub name: String,
    /// Metres per brick edge. Everything on the wire is in these units.
    pub span: f64,
    /// The SDF cell, for the viewer to report what it can and cannot resolve.
    pub cell: f64,
    pub lo: Vec3,
    pub hi: Vec3,
    /// Non-empty bricks the sender knows about, so the viewer can say "412 of
    /// 900" instead of spinning.
    pub bricks: u32,
    /// Absolute path to the splat `.ply`, empty when the map has no
    /// appearance layer yet. A path and not the splat itself: a lab splat is
    /// hundreds of megabytes and the viewer is on the same machine as the
    /// file in every case that exists today.
    pub splat: String,
    /// The map's `[align]`: the transform from the capture frame into this
    /// map frame, as rotation-then-translation.
    ///
    /// On the wire because the splat needs it and nothing else does. The
    /// physics layer was baked *after* alignment, so its bricks arrive
    /// already in the map frame; the splat trains in the raw capture frame
    /// and is bent by the renderer. Sending the transform rather than having
    /// the viewer read `map.toml` keeps one parser for the format.
    ///
    /// `rotate` is an enum, not a matrix: 0 none, 1 y-up → z-up
    /// `(x, y, z) → (x, −z, y)`, 2 y-down → z-up `(x, y, z) → (x, z, −y)`.
    /// A free-form matrix on the wire would invite a transform nothing bakes
    /// and nobody can verify against the manifest.
    pub align_rotate: u32,
    pub align_translate: [f64; 3],
}

pub fn encode_info(info: &MapInfo) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + info.name.len() + info.splat.len());
    out.extend_from_slice(&INFO_TAG.to_le_bytes());
    out.extend_from_slice(&(info.span as f32).to_le_bytes());
    out.extend_from_slice(&(info.cell as f32).to_le_bytes());
    for v in [info.lo.x, info.lo.y, info.lo.z, info.hi.x, info.hi.y, info.hi.z] {
        out.extend_from_slice(&(v as f32).to_le_bytes());
    }
    out.extend_from_slice(&info.bricks.to_le_bytes());
    out.extend_from_slice(&info.align_rotate.to_le_bytes());
    for v in info.align_translate {
        out.extend_from_slice(&(v as f32).to_le_bytes());
    }
    for s in [&info.name, &info.splat] {
        let b = s.as_bytes();
        out.extend_from_slice(&(b.len() as u16).to_le_bytes());
        out.extend_from_slice(b);
    }
    out
}

pub fn decode_info(bytes: &[u8]) -> Option<MapInfo> {
    if bytes.len() < 56 || u32::from_le_bytes(bytes[0..4].try_into().ok()?) != INFO_TAG {
        return None;
    }
    let f = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as f64;
    let bricks = u32::from_le_bytes(bytes[36..40].try_into().ok()?);
    let align_rotate = u32::from_le_bytes(bytes[40..44].try_into().ok()?);
    let align_translate = [f(44), f(48), f(52)];
    let mut at = 56;
    let mut text = || {
        if bytes.len() < at + 2 {
            return None;
        }
        let n = u16::from_le_bytes([bytes[at], bytes[at + 1]]) as usize;
        if bytes.len() < at + 2 + n {
            return None;
        }
        let s = String::from_utf8_lossy(&bytes[at + 2..at + 2 + n]).into_owned();
        at += 2 + n;
        Some(s)
    };
    let name = text()?;
    let splat = text()?;
    Some(MapInfo {
        name,
        span: f(4),
        cell: f(8),
        lo: Vec3::new(f(12), f(16), f(20)),
        hi: Vec3::new(f(24), f(28), f(32)),
        bricks,
        splat,
        align_rotate,
        align_translate,
    })
}

// MARK: - Sending

/// What to send next, and how often to send it again.
///
/// Two queues rather than one. Dirty bricks jump — a corner of the room that
/// just appeared should appear — and everything else is walked continuously
/// at a rate that heals loss without ever being the reason a link is busy.
/// The walk is the whole error-recovery story, so its rate is the one number
/// worth arguing about: at 32 bricks/s a 900-brick lab re-sends completely
/// every 28 s, and any single lost page is healed within that.
pub struct BrickWalker {
    bricked: Bricked,
    /// Version last *sent* per brick, so a re-mesh bumps and a re-walk does
    /// not.
    versions: BTreeMap<BrickKey, u32>,
    /// Bricks changed since the last send, in the order they changed.
    dirty: Vec<BrickKey>,
    /// Bricks that went empty and owe a tombstone.
    doomed: Vec<BrickKey>,
    /// Rolling position of the background walk.
    cursor: usize,
}

impl BrickWalker {
    /// A walker over a map that is not going to change — the static case
    /// `viewbridge --map` serves, and the one the Swift renderer is built
    /// against before live fusion exists.
    pub fn new(bricked: Bricked) -> BrickWalker {
        let versions = bricked.bricks.keys().map(|k| (*k, 1u32)).collect();
        let dirty = bricked.bricks.keys().copied().collect();
        BrickWalker { bricked, versions, dirty, doomed: Vec::new(), cursor: 0 }
    }

    pub fn span(&self) -> f64 {
        self.bricked.span
    }
    pub fn len(&self) -> usize {
        self.bricked.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bricked.is_empty()
    }

    /// Replace the surface, marking as dirty exactly the bricks whose mesh
    /// actually changed.
    ///
    /// Comparing meshes rather than trusting a dirty flag from fusion is
    /// deliberate at this stage: it makes the stream correct regardless of
    /// whether the producer's bookkeeping is, and a wrong dirty flag is a
    /// stale wall on screen that nobody can debug from the viewer end. When
    /// fusion's own per-brick tracking earns its keep, this becomes the
    /// fallback rather than the path.
    ///
    /// The honest limit: this sees *tessellation*, not surface. A producer
    /// whose triangles change shape when the scene grows — a re-baked phone
    /// scan, where [`split_to`] subdivides one enormous floor triangle
    /// differently each time — dirties every brick it touches, and the
    /// bandwidth win disappears. Fusion is not such a producer; surface nets
    /// is local. Pinned by `a_reshaped_tessellation_dirties_everything`.
    pub fn replace(&mut self, next: Bricked) {
        for (key, mesh) in &next.bricks {
            if self.bricked.bricks.get(key) != Some(mesh) {
                let v = self.versions.entry(*key).or_insert(0);
                *v += 1;
                self.dirty.push(*key);
            }
        }
        for key in self.bricked.bricks.keys() {
            if !next.bricks.contains_key(key) {
                let v = self.versions.entry(*key).or_insert(0);
                *v += 1;
                self.doomed.push(*key);
            }
        }
        self.bricked = next;
    }

    /// Whether anything is waiting to go out ahead of the background walk.
    pub fn has_pending(&self) -> bool {
        !self.dirty.is_empty() || !self.doomed.is_empty()
    }

    /// The next batch of packets: up to `burst` dirty bricks, then `walk`
    /// bricks of background repair.
    ///
    /// Returns whole bricks — never a partial one — so a caller that stops
    /// early never leaves a half-sent brick that the receiver will hold
    /// incomplete until the next walk comes round.
    ///
    /// `burst` exists because the first batch of a freshly loaded map is the
    /// *entire* map: a lab is hundreds of bricks, and handing a socket two
    /// thousand datagrams in one go is how a burst that is nothing on
    /// loopback becomes a burst that overruns a receive buffer on wifi. The
    /// caller spreads it over ticks; the receiver draws a map that fills in.
    pub fn next_batch(&mut self, burst: usize, walk: usize) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let stones = self.doomed.len().min(burst);
        for key in self.doomed.drain(..stones).collect::<Vec<_>>() {
            let v = self.versions.get(&key).copied().unwrap_or(1);
            out.extend(encode_brick(key, v, &BrickMesh::default()));
            self.versions.remove(&key);
        }
        let fresh = self.dirty.len().min(burst.saturating_sub(stones));
        for key in self.dirty.drain(..fresh).collect::<Vec<_>>() {
            if let Some(mesh) = self.bricked.bricks.get(&key) {
                let v = self.versions.get(&key).copied().unwrap_or(1);
                out.extend(encode_brick(key, v, mesh));
            }
        }
        if self.bricked.is_empty() {
            return out;
        }
        for _ in 0..walk.min(self.bricked.len()) {
            let idx = self.cursor % self.bricked.len();
            self.cursor = self.cursor.wrapping_add(1);
            if let Some((key, mesh)) = self.bricked.bricks.iter().nth(idx) {
                let v = self.versions.get(key).copied().unwrap_or(1);
                out.extend(encode_brick(*key, v, mesh));
            }
        }
        out
    }

    /// The description that goes with this stream.
    ///
    /// `align` is the manifest's, passed through untouched — see
    /// [`MapInfo::align_rotate`] for why the splat needs it and the bricks
    /// do not.
    pub fn info(
        &self,
        name: &str,
        cell: f64,
        splat: Option<&std::path::Path>,
        align: Option<&crate::manifest::Align>,
        fallback: Option<&crate::manifest::Extent>,
    ) -> MapInfo {
        let mut lo = Vec3::splat(f64::INFINITY);
        let mut hi = Vec3::splat(f64::NEG_INFINITY);
        for (key, mesh) in &self.bricked.bricks {
            for v in mesh.vertices(*key, self.bricked.span) {
                lo = lo.component_min(v);
                hi = hi.component_max(v);
            }
        }
        if !lo.x.is_finite() {
            // No bricks: an appearance-only map. Its extent was measured at
            // import and lives in the manifest, and without it the viewer
            // cannot tell a cupboard from a warehouse — see `MapManifest`.
            match fallback {
                Some(e) => {
                    lo = Vec3::new(e.lo[0], e.lo[1], e.lo[2]);
                    hi = Vec3::new(e.hi[0], e.hi[1], e.hi[2]);
                }
                None => {
                    lo = Vec3::zeros();
                    hi = Vec3::zeros();
                }
            }
        }
        MapInfo {
            name: name.to_string(),
            span: self.bricked.span,
            cell,
            lo,
            hi,
            bricks: self.bricked.len() as u32,
            splat: splat.map(|p| p.display().to_string()).unwrap_or_default(),
            align_rotate: match align.map(|a| a.rotate.as_str()) {
                Some("y-up-to-z-up") => 1,
                Some("y-down-to-z-up") => 2,
                _ => 0,
            },
            align_translate: align.map(|a| a.translate).unwrap_or([0.0; 3]),
        }
    }
}

// MARK: - Receiving (the Rust twin of Dojo's assembler)

/// Reassembles pages into bricks. The Swift `MapReceiver` is this, in Swift;
/// keeping a Rust one costs thirty lines and buys a test that exercises the
/// whole loss-and-version story without a simulator.
#[derive(Debug, Default)]
pub struct BrickAssembler {
    partial: BTreeMap<BrickKey, (u32, Vec<u8>, Vec<bool>)>,
    pub bricks: BTreeMap<BrickKey, (u32, BrickMesh)>,
}

impl BrickAssembler {
    /// Feed one page. Returns the brick's key when that page completed it.
    pub fn accept(&mut self, page: &BrickPage) -> Option<BrickKey> {
        // A brick we already hold at this version or newer has nothing to
        // learn from this page. This is what makes the background walk free:
        // the steady state is every page arriving and being dropped here.
        if let Some((v, _)) = self.bricks.get(&page.key) {
            if *v >= page.version {
                return None;
            }
        }
        if page.is_tombstone() {
            self.partial.remove(&page.key);
            self.bricks.remove(&page.key);
            return Some(page.key);
        }
        let slot = self.partial.entry(page.key).or_insert_with(|| {
            (page.version, vec![0u8; page.payload_len], vec![false; page.pages as usize])
        });
        // A newer version restarts assembly. Mixing pages from two versions
        // of a brick is the one way to build a mesh that never existed.
        if slot.0 != page.version {
            if page.version < slot.0 {
                return None;
            }
            *slot = (page.version, vec![0u8; page.payload_len], vec![false; page.pages as usize]);
        }
        if slot.1.len() != page.payload_len || page.offset + page.bytes.len() > slot.1.len() {
            return None;
        }
        slot.1[page.offset..page.offset + page.bytes.len()].copy_from_slice(&page.bytes);
        if let Some(f) = slot.2.get_mut(page.page as usize) {
            *f = true;
        }
        if !slot.2.iter().all(|f| *f) {
            return None;
        }
        let (v, bytes, _) = self.partial.remove(&page.key)?;
        let mesh = decode_payload(&bytes)?;
        self.bricks.insert(page.key, (v, mesh));
        Some(page.key)
    }
}

// ---------------------------------------------------------------------------
// Things in the scene
// ---------------------------------------------------------------------------
//
// Objects travel on the *same port* as the map, because they are the same
// question — "what is in this place?" — and one port means one receiver on
// the Swift side deciding what a packet is by its tag, rather than two
// sockets racing to describe one room. Two tags, for the reason `MAPI` and
// `BRIK` are two: a roster is slow and repeated, a pose is fast and
// disposable.

/// `"OBJS"` — the roster: what things are here, and where their files are.
pub const THING_TAG: u32 = 0x534A_424F;

/// `"OBJP"` — poses only, at frame rate, once physics is moving them.
pub const THING_POSE_TAG: u32 = 0x504A_424F;

/// One thing in the scene, as the viewer needs to know it.
///
/// Paths, not geometry. An object's mesh is a few hundred kilobytes and the
/// viewer is on the same machine as the file in every case that exists today
/// — the same call the map's splat layer makes. Bricks exist because a *map*
/// grows while you watch it; an object is finished before it is placed.
#[derive(Debug, Clone, PartialEq)]
pub struct ThingInfo {
    /// Index in the scene's roster, stable for the life of the bridge. The
    /// pose stream refers to things by this and nothing else.
    pub id: u16,
    /// How many things the scene has, repeated in every page so a receiver
    /// can tell a partial roster from a complete one, and can drop anything
    /// left over from a scene that had more.
    pub total: u16,
    pub name: String,
    /// What to draw, as `(part name, absolute mesh path)` — one entry for a
    /// captured thing, one per body for a mechanism.
    ///
    /// The order is the object manifest's part order and the index the pose
    /// stream uses, so a wheel is referred to by a number rather than by a
    /// name a receiver would have to match. A part with an empty path is a
    /// body with nothing to draw, which is a real state and not an error.
    pub parts: Vec<(String, String)>,
    /// Absolute path to the splat, empty if there is no appearance layer.
    pub splat: String,
    /// The object's `[align]`, same encoding as [`MapInfo::align_rotate`] —
    /// what a renderer must apply to bring the splat into the object frame.
    pub align_rotate: u32,
    pub align_translate: [f64; 3],
    /// Where it is now: position and orientation `[x, y, z, w]` in the map
    /// frame. Repeated here as well as in the pose stream so a tab that
    /// opens into a scene with nothing running still knows where things are.
    pub pos: Vec3,
    pub quat: [f64; 4],
    /// Height above its own base, metres — what a placement UI puts in a row.
    pub height: f64,
    /// Kilograms. On the wire because "how heavy is that" is the question a
    /// person asks before they put a robot next to it.
    pub mass: f64,
    /// Whether physics is moving it. False in an explore tab, where the pose
    /// above is the whole truth and no pose stream is coming.
    pub live: bool,
}

/// Where one *part* of one thing is, at frame rate.
///
/// Keyed by `(id, part)` rather than by id alone because a mechanism moves in
/// pieces: a board's deck leans while its trucks steer and its wheels spin,
/// and a stream that carried only the base would draw a rigid plank.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThingPose {
    pub id: u16,
    /// Index into [`ThingInfo::parts`].
    pub part: u16,
    pub pos: Vec3,
    /// `[x, y, z, w]`.
    pub quat: [f64; 4],
}

/// Encode the roster, paged to stay under [`PAGE`].
///
/// Returns one datagram per page; every page carries `total`, so a receiver
/// that has seen ids `0..total` has the whole scene and one that sees fewer
/// knows to keep waiting rather than to draw a half-empty room.
pub fn encode_things(things: &[ThingInfo]) -> Vec<Vec<u8>> {
    let mut pages = Vec::new();
    let mut page: Vec<u8> = Vec::new();
    let mut count = 0u16;
    let flush = |page: &mut Vec<u8>, count: &mut u16, pages: &mut Vec<Vec<u8>>| {
        if *count == 0 {
            return;
        }
        let mut out = Vec::with_capacity(page.len() + 8);
        out.extend_from_slice(&THING_TAG.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&(things.len() as u16).to_le_bytes());
        out.extend_from_slice(page);
        pages.push(out);
        page.clear();
        *count = 0;
    };
    for t in things {
        let mut one = Vec::with_capacity(96 + t.name.len() + t.splat.len() + 64 * t.parts.len());
        one.extend_from_slice(&t.id.to_le_bytes());
        one.push(t.live as u8);
        one.push(0); // pad, so the floats that follow start on a multiple of 4
        for v in [t.pos.x, t.pos.y, t.pos.z] {
            one.extend_from_slice(&(v as f32).to_le_bytes());
        }
        for v in t.quat {
            one.extend_from_slice(&(v as f32).to_le_bytes());
        }
        one.extend_from_slice(&(t.height as f32).to_le_bytes());
        one.extend_from_slice(&(t.mass as f32).to_le_bytes());
        one.extend_from_slice(&t.align_rotate.to_le_bytes());
        for v in t.align_translate {
            one.extend_from_slice(&(v as f32).to_le_bytes());
        }
        let mut text = |one: &mut Vec<u8>, s: &str| {
            let b = s.as_bytes();
            one.extend_from_slice(&(b.len() as u16).to_le_bytes());
            one.extend_from_slice(b);
        };
        text(&mut one, &t.name);
        text(&mut one, &t.splat);
        one.extend_from_slice(&(t.parts.len() as u16).to_le_bytes());
        for (name, mesh) in &t.parts {
            text(&mut one, name);
            text(&mut one, mesh);
        }
        if !page.is_empty() && page.len() + one.len() + 8 > PAGE {
            flush(&mut page, &mut count, &mut pages);
        }
        page.extend_from_slice(&one);
        count += 1;
    }
    flush(&mut page, &mut count, &mut pages);
    pages
}

/// Decode one roster page. `None` for anything that is not one, or that is
/// truncated — a half-read path would name a file that does not exist.
pub fn decode_things(bytes: &[u8]) -> Option<Vec<ThingInfo>> {
    if bytes.len() < 8 || u32::from_le_bytes(bytes[0..4].try_into().ok()?) != THING_TAG {
        return None;
    }
    let count = u16::from_le_bytes(bytes[4..6].try_into().ok()?) as usize;
    let total = u16::from_le_bytes(bytes[6..8].try_into().ok()?);
    let mut at = 8;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if bytes.len() < at + 56 {
            return None;
        }
        let f = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as f64;
        let id = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        let live = bytes[at + 2] != 0;
        let pos = Vec3::new(f(at + 4), f(at + 8), f(at + 12));
        let quat = [f(at + 16), f(at + 20), f(at + 24), f(at + 28)];
        let height = f(at + 32);
        let mass = f(at + 36);
        let align_rotate = u32::from_le_bytes(bytes[at + 40..at + 44].try_into().ok()?);
        let align_translate = [f(at + 44), f(at + 48), f(at + 52)];
        at += 56;
        // A free function rather than a closure: the part list needs to read
        // its own count between strings, and a closure holding `at` mutably
        // would own it for the rest of the loop.
        fn text(bytes: &[u8], at: &mut usize) -> Option<String> {
            if bytes.len() < *at + 2 {
                return None;
            }
            let n = u16::from_le_bytes([bytes[*at], bytes[*at + 1]]) as usize;
            if bytes.len() < *at + 2 + n {
                return None;
            }
            let s = String::from_utf8_lossy(&bytes[*at + 2..*at + 2 + n]).into_owned();
            *at += 2 + n;
            Some(s)
        }
        let name = text(bytes, &mut at)?;
        let splat = text(bytes, &mut at)?;
        if bytes.len() < at + 2 {
            return None;
        }
        let n = u16::from_le_bytes([bytes[at], bytes[at + 1]]) as usize;
        at += 2;
        let mut parts = Vec::with_capacity(n.min(64));
        for _ in 0..n {
            let pname = text(bytes, &mut at)?;
            let pmesh = text(bytes, &mut at)?;
            parts.push((pname, pmesh));
        }
        out.push(ThingInfo {
            id,
            total,
            name,
            parts,
            splat,
            align_rotate,
            align_translate,
            pos,
            quat,
            height,
            mass,
            live,
        });
    }
    Some(out)
}

/// Encode a frame of poses. `seq` wraps; a receiver keeps the newest and
/// drops anything older, the way the world-frame stream does.
///
/// One datagram: 32 bytes a part means 37 fit under [`PAGE`] — five
/// skateboards, or thirty-seven buckets, which is more than anybody is
/// dragging around a room. Beyond it the tail is dropped rather than paged,
/// because a pose is worthless the moment the next frame exists — and the
/// roster, which is paged, is what tells the viewer those things are there at
/// all.
pub fn encode_thing_poses(seq: u32, poses: &[ThingPose]) -> Vec<u8> {
    let room = (PAGE - 10) / THING_POSE_BYTES;
    let n = poses.len().min(room);
    let mut out = Vec::with_capacity(10 + n * THING_POSE_BYTES);
    out.extend_from_slice(&THING_POSE_TAG.to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&(n as u16).to_le_bytes());
    for p in &poses[..n] {
        out.extend_from_slice(&p.id.to_le_bytes());
        out.extend_from_slice(&p.part.to_le_bytes());
        for v in [p.pos.x, p.pos.y, p.pos.z] {
            out.extend_from_slice(&(v as f32).to_le_bytes());
        }
        for v in p.quat {
            out.extend_from_slice(&(v as f32).to_le_bytes());
        }
    }
    out
}

/// `id` + part index + position + quaternion, all f32 after the two indices.
const THING_POSE_BYTES: usize = 2 + 2 + 12 + 16;

pub fn decode_thing_poses(bytes: &[u8]) -> Option<(u32, Vec<ThingPose>)> {
    if bytes.len() < 10 || u32::from_le_bytes(bytes[0..4].try_into().ok()?) != THING_POSE_TAG {
        return None;
    }
    let seq = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    let n = u16::from_le_bytes(bytes[8..10].try_into().ok()?) as usize;
    if bytes.len() < 10 + n * THING_POSE_BYTES {
        return None;
    }
    let f = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let at = 10 + i * THING_POSE_BYTES;
        out.push(ThingPose {
            id: u16::from_le_bytes([bytes[at], bytes[at + 1]]),
            part: u16::from_le_bytes([bytes[at + 2], bytes[at + 3]]),
            pos: Vec3::new(f(at + 4), f(at + 8), f(at + 12)),
            quat: [f(at + 16), f(at + 20), f(at + 24), f(at + 28)],
        });
    }
    Some((seq, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A floor tiled out of small quads — the shape surface nets and TSDF
    /// fusion actually produce, and therefore the one the diff is designed
    /// for. `slab` deliberately is not: see `replace_dirties_only_what_changed`.
    fn tiles(n: usize) -> TriMesh {
        let step = 0.08; // a quarter of a brick
        let mut verts = Vec::new();
        let mut tris = Vec::new();
        for iy in 0..=n {
            for ix in 0..=n {
                verts.push(Vec3::new(ix as f64 * step, iy as f64 * step, 0.0));
            }
        }
        let row = (n + 1) as u32;
        for iy in 0..n as u32 {
            for ix in 0..n as u32 {
                let a = iy * row + ix;
                tris.push([a, a + 1, a + row + 1]);
                tris.push([a, a + row + 1, a + row]);
            }
        }
        TriMesh::new(verts, tris)
    }

    fn slab(size: f64) -> TriMesh {
        // One big square in the z = 0 plane: two triangles, deliberately far
        // larger than a brick, because that is the case centroid bucketing
        // gets wrong without the split.
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(size, 0.0, 0.0),
            Vec3::new(size, size, 0.0),
            Vec3::new(0.0, size, 0.0),
        ];
        TriMesh::new(v, vec![[0, 1, 2], [0, 2, 3]])
    }

    #[test]
    fn big_triangles_are_split_across_bricks() {
        let cell = 0.02;
        let span = cell * BRICK as f64; // 0.32 m
        let b = brick_trimesh(&slab(3.2), cell);
        assert_eq!(b.span, span);
        // A 3.2 m square over 0.32 m bricks: 10×10 in the plane. Centroid
        // bucketing without the split would give exactly 1 or 2.
        assert!(b.len() >= 90, "only {} bricks — the split did not run", b.len());
        // Every vertex must land back within a millimetre of the plane it
        // came from, which is the property the quantization window exists to
        // provide.
        for (key, mesh) in &b.bricks {
            for v in mesh.vertices(*key, span) {
                assert!(v.z.abs() < 1e-3, "vertex off the plane by {}", v.z);
                assert!((-0.01..=3.21).contains(&v.x), "x escaped: {}", v.x);
            }
        }
    }

    #[test]
    fn quantization_holds_sub_millimetre() {
        let cell = 0.02;
        let span = cell * BRICK as f64;
        let key = BrickKey { i: 3, j: -2, k: 1 };
        let (lo, step) = BrickMesh::window(key, span);
        // Worst case is half a step, everywhere in the window.
        assert!(step / 2.0 < 1e-4, "step {step} is coarser than promised");
        let p = lo + Vec3::new(span * 1.3, span * 0.2, span * 1.9);
        let q = [
            quantize(p.x - lo.x, step),
            quantize(p.y - lo.y, step),
            quantize(p.z - lo.z, step),
        ];
        let back = BrickMesh { verts: vec![q], tris: vec![] }.vertices(key, span)[0];
        assert!((back - p).norm() < 1e-4, "round trip lost {} m", (back - p).norm());
    }

    #[test]
    fn page_roundtrip_through_the_assembler() {
        let b = brick_trimesh(&slab(3.2), 0.02);
        let (key, mesh) = b.bricks.iter().next().unwrap();
        let pages = encode_brick(*key, 7, mesh);
        let mut asm = BrickAssembler::default();
        for p in &pages {
            assert!(p.len() <= BRICK_HEADER + PAGE, "page over MTU budget: {}", p.len());
            let page = decode_brick(p).expect("decodes");
            asm.accept(&page);
        }
        let (v, got) = asm.bricks.get(key).expect("assembled");
        assert_eq!(*v, 7);
        assert_eq!(got, mesh);
    }

    #[test]
    fn a_dense_brick_actually_pages() {
        // The paging path is only exercised by a brick over PAGE bytes, and
        // the slab's flat bricks are not. Build one that is.
        let mut mesh = BrickMesh::default();
        for n in 0..400u16 {
            mesh.verts.push([n, n.wrapping_mul(3), n.wrapping_mul(7)]);
        }
        for n in 0..390u16 {
            mesh.tris.push([n, n + 1, n + 2]);
        }
        let key = BrickKey { i: 0, j: 0, k: 0 };
        let pages = encode_brick(key, 2, &mesh);
        assert!(pages.len() > 1, "expected paging, got {} page", pages.len());
        let mut asm = BrickAssembler::default();
        // Out of order, because UDP does not promise order and the assembler
        // must not depend on it.
        for p in pages.iter().rev() {
            asm.accept(&decode_brick(p).unwrap());
        }
        assert_eq!(asm.bricks.get(&key).map(|(_, m)| m), Some(&mesh));
    }

    #[test]
    fn a_lost_page_leaves_the_brick_absent_until_the_walk_repeats() {
        let mut mesh = BrickMesh::default();
        for n in 0..400u16 {
            mesh.verts.push([n, n, n]);
        }
        for n in 0..390u16 {
            mesh.tris.push([n, n + 1, n + 2]);
        }
        let key = BrickKey { i: 1, j: 1, k: 1 };
        let pages = encode_brick(key, 1, &mesh);
        let mut asm = BrickAssembler::default();
        for p in pages.iter().skip(1) {
            asm.accept(&decode_brick(p).unwrap());
        }
        assert!(asm.bricks.get(&key).is_none(), "incomplete brick must not render");
        // The walk comes round again; this time nothing is dropped.
        for p in &pages {
            asm.accept(&decode_brick(p).unwrap());
        }
        assert!(asm.bricks.contains_key(&key), "the re-walk did not heal it");
    }

    #[test]
    fn a_stale_page_cannot_corrupt_a_newer_brick() {
        let key = BrickKey { i: 0, j: 0, k: 0 };
        let old = BrickMesh { verts: vec![[1, 1, 1], [2, 2, 2], [3, 3, 3]], tris: vec![[0, 1, 2]] };
        let new = BrickMesh { verts: vec![[9, 9, 9], [8, 8, 8], [7, 7, 7]], tris: vec![[0, 1, 2]] };
        let mut asm = BrickAssembler::default();
        for p in encode_brick(key, 5, &new) {
            asm.accept(&decode_brick(&p).unwrap());
        }
        for p in encode_brick(key, 4, &old) {
            asm.accept(&decode_brick(&p).unwrap());
        }
        assert_eq!(asm.bricks.get(&key).map(|(_, m)| m), Some(&new));
    }

    #[test]
    fn a_tombstone_removes_the_brick() {
        let key = BrickKey { i: 2, j: 0, k: 0 };
        let mesh = BrickMesh { verts: vec![[1, 1, 1], [2, 2, 2], [3, 3, 3]], tris: vec![[0, 1, 2]] };
        let mut asm = BrickAssembler::default();
        for p in encode_brick(key, 1, &mesh) {
            asm.accept(&decode_brick(&p).unwrap());
        }
        assert!(asm.bricks.contains_key(&key));
        let stone = encode_brick(key, 2, &BrickMesh::default());
        assert_eq!(stone.len(), 1);
        let page = decode_brick(&stone[0]).unwrap();
        assert!(page.is_tombstone());
        asm.accept(&page);
        assert!(!asm.bricks.contains_key(&key), "carved space did not disappear");
    }

    #[test]
    fn the_walker_sends_everything_then_repairs() {
        let b = brick_trimesh(&slab(1.6), 0.02);
        let n = b.len();
        let mut w = BrickWalker::new(b);
        let first = w.next_batch(usize::MAX, 0);
        let mut asm = BrickAssembler::default();
        for p in &first {
            asm.accept(&decode_brick(p).unwrap());
        }
        assert_eq!(asm.bricks.len(), n, "first batch did not carry the whole map");
        // A steady walk re-sends and the assembler drops it all — the
        // property that makes no-ACK repair free in the common case.
        let again = w.next_batch(usize::MAX, 8);
        assert!(!again.is_empty());
        assert!(again.iter().all(|p| asm.accept(&decode_brick(p).unwrap()).is_none()));
    }

    #[test]
    fn replace_dirties_only_what_changed() {
        let mut w = BrickWalker::new(brick_trimesh(&tiles(20), 0.02));
        let _ = w.next_batch(usize::MAX, 0);
        // The same surface again: nothing changed, so a zero-walk batch is
        // empty. If this ever fails, every stance sweep re-sends the room.
        w.replace(brick_trimesh(&tiles(20), 0.02));
        assert!(w.next_batch(usize::MAX, 0).is_empty(), "an unchanged map re-sent itself");
        // More floor arrives. The bricks that already had their surface must
        // stay quiet — this is the property the whole brick idea is for.
        w.replace(brick_trimesh(&tiles(30), 0.02));
        let batch = w.next_batch(usize::MAX, 0);
        assert!(!batch.is_empty());
        let touched: std::collections::BTreeSet<_> =
            batch.iter().filter_map(|p| decode_brick(p)).map(|p| p.key).collect();
        assert!(
            touched.len() < w.len(),
            "growing the map re-sent all of it: {} of {}",
            touched.len(),
            w.len()
        );
    }

    /// The honest edge on [`BrickWalker::replace`]: the diff can only see
    /// what the *tessellation* does, and `split_to` subdivides a triangle
    /// globally, so a producer whose triangles change shape when the scene
    /// grows dirties everything. Fusion is not such a producer — surface nets
    /// is local — but a re-baked phone scan is, and this pins which is which
    /// rather than leaving it to be discovered as a bandwidth mystery.
    #[test]
    fn a_reshaped_tessellation_dirties_everything() {
        let mut w = BrickWalker::new(brick_trimesh(&slab(1.6), 0.02));
        let _ = w.next_batch(usize::MAX, 0);
        w.replace(brick_trimesh(&slab(2.4), 0.02));
        let touched: std::collections::BTreeSet<_> =
            w.next_batch(usize::MAX, 0).iter().filter_map(|p| decode_brick(p)).map(|p| p.key).collect();
        assert!(touched.len() >= w.len(), "the note above is stale — the diff got smarter");
    }

    #[test]
    fn a_vanished_brick_is_tombstoned() {
        let mut w = BrickWalker::new(brick_trimesh(&tiles(30), 0.02));
        let _ = w.next_batch(usize::MAX, 0);
        w.replace(brick_trimesh(&tiles(20), 0.02));
        let batch = w.next_batch(usize::MAX, 0);
        let stones: Vec<_> = batch
            .iter()
            .filter_map(|p| decode_brick(p))
            .filter(|p| p.is_tombstone())
            .collect();
        assert!(!stones.is_empty(), "shrinking the map left orphan bricks on screen");
    }

    #[test]
    fn info_roundtrips() {
        let w = BrickWalker::new(brick_trimesh(&slab(1.6), 0.02));
        let info = w.info(
            "lab",
            0.02,
            Some(std::path::Path::new("/maps/lab/splat.ply")),
            Some(&crate::manifest::Align {
                rotate: "y-up-to-z-up".into(),
                translate: [0.0, 0.0, -1.31],
            }),
            None,
        );
        let back = decode_info(&encode_info(&info)).expect("decodes");
        assert_eq!(back.name, "lab");
        assert_eq!(back.splat, "/maps/lab/splat.ply");
        assert_eq!(back.align_rotate, 1, "the splat would render on its side");
        assert!((back.align_translate[2] + 1.31).abs() < 1e-4);
        assert_eq!(back.bricks, info.bricks);
        assert!((back.span - 0.32).abs() < 1e-6);
        assert!((back.hi.x - 1.6).abs() < 1e-3, "extent lost: {}", back.hi.x);
    }

    fn a_thing(id: u16, total: u16, name: &str) -> ThingInfo {
        ThingInfo {
            id,
            total,
            name: name.into(),
            parts: vec![(name.into(), format!("/objects/{name}/mesh.stl"))],
            splat: format!("/objects/{name}/splat.ply"),
            align_rotate: 2,
            align_translate: [0.1, -0.2, -1.03],
            pos: Vec3::new(1.5, -0.25, 0.31),
            quat: [0.0, 0.0, 0.3826834, 0.9238795],
            height: 0.368,
            mass: 1.2,
            live: true,
        }
    }

    #[test]
    fn a_roster_round_trips() {
        let things = vec![a_thing(0, 2, "bucket"), a_thing(1, 2, "crate")];
        let pages = encode_things(&things);
        assert_eq!(pages.len(), 1, "two small things should be one datagram");
        let back = decode_things(&pages[0]).expect("roster must decode");
        assert_eq!(back.len(), 2);
        assert_eq!(back[1].name, "crate");
        assert_eq!(back[1].total, 2, "a page that does not say how many is a half-scene");
        assert_eq!(back[0].parts[0].1, "/objects/bucket/mesh.stl");
        assert_eq!(back[0].align_rotate, 2, "the splat would render upside down");
        assert!((back[0].align_translate[2] + 1.03).abs() < 1e-5);
        assert!((back[0].pos - things[0].pos).norm() < 1e-5);
        assert!((back[0].quat[3] - things[0].quat[3]).abs() < 1e-6);
        assert!((back[0].height - 0.368).abs() < 1e-5);
        assert!(back[0].live);
    }

    #[test]
    fn a_mechanism_carries_one_part_per_body() {
        // Seven bodies, three distinct shapes — a board as `rigbake` writes
        // it. What must survive is the *count and order*, because the pose
        // stream refers to a wheel by its index in this list.
        let board = ThingInfo {
            parts: [
                ("skate_deck", "/objects/board/skate_deck.stl"),
                ("skate_truck_front", "/objects/board/skate_truck_front.stl"),
                ("skate_truck_rear", "/objects/board/skate_truck_front.stl"),
                ("skate_wheel_fl", "/objects/board/skate_wheel_fl.stl"),
                ("skate_wheel_fr", "/objects/board/skate_wheel_fl.stl"),
                ("skate_wheel_rl", "/objects/board/skate_wheel_fl.stl"),
                ("skate_wheel_rr", "/objects/board/skate_wheel_fl.stl"),
            ]
            .iter()
            .map(|(n, m)| (n.to_string(), m.to_string()))
            .collect(),
            ..a_thing(0, 1, "skateboard")
        };
        let pages = encode_things(&[board.clone()]);
        assert_eq!(pages.len(), 1);
        let back = decode_things(&pages[0]).unwrap();
        assert_eq!(back[0].parts.len(), 7);
        assert_eq!(back[0].parts, board.parts, "a part moved or a name was truncated");
    }

    #[test]
    fn a_big_roster_pages_and_every_page_says_the_total() {
        // Long paths are the realistic case — a repo checkout is deep — and
        // they are what pushes a roster over one datagram.
        let things: Vec<ThingInfo> = (0..40)
            .map(|i| {
                let mut t = a_thing(i, 40, "a-thing-with-a-long-enough-name-to-matter");
                t.parts = vec![(
                    "part".into(),
                    format!("/Users/somebody/Developer/ipse/objects/thing-{i}/mesh.stl"),
                )];
                t
            })
            .collect();
        let pages = encode_things(&things);
        assert!(pages.len() > 1, "40 things should not fit in one datagram");
        let mut seen = Vec::new();
        for p in &pages {
            assert!(p.len() <= PAGE + 8, "page over the MTU budget: {}", p.len());
            let got = decode_things(p).expect("every page must decode on its own");
            for t in got {
                assert_eq!(t.total, 40);
                seen.push(t.id);
            }
        }
        seen.sort();
        assert_eq!(seen, (0..40).collect::<Vec<u16>>(), "a thing was lost between pages");
    }

    #[test]
    fn poses_round_trip_and_the_tail_is_dropped_not_mangled() {
        let poses: Vec<ThingPose> = (0..3)
            .map(|i| ThingPose {
                id: i,
                part: i * 2,
                pos: Vec3::new(i as f64, 0.5, 0.25),
                quat: [0.0, 0.0, 0.0, 1.0],
            })
            .collect();
        let (seq, back) = decode_thing_poses(&encode_thing_poses(7, &poses)).unwrap();
        assert_eq!(seq, 7);
        assert_eq!(back.len(), 3);
        assert!((back[2].pos.x - 2.0).abs() < 1e-6);
        assert_eq!(back[2].part, 4, "the part index is what makes a wheel a wheel");
        assert!((back[1].quat[3] - 1.0).abs() < 1e-6);

        // More things than a datagram holds: the packet stays valid and short.
        let many: Vec<ThingPose> = (0..200).map(|i| ThingPose { id: i, ..poses[0] }).collect();
        let pkt = encode_thing_poses(1, &many);
        assert!(pkt.len() <= PAGE, "pose packet over budget: {}", pkt.len());
        let (_, back) = decode_thing_poses(&pkt).unwrap();
        assert!(back.len() < 200 && !back.is_empty());
    }

    #[test]
    fn a_thing_packet_is_not_a_brick_and_the_other_way_round() {
        // The two tags share a port, so each decoder must refuse the other's
        // packets rather than read a pose as a brick key.
        let roster = encode_things(&[a_thing(0, 1, "x")]).remove(0);
        assert!(decode_brick(&roster).is_none());
        assert!(decode_info(&roster).is_none());
        assert!(decode_thing_poses(&roster).is_none());
        let brick = encode_brick(BrickKey { i: 1, j: 2, k: 3 }, 1, &BrickMesh::default())
            .remove(0);
        assert!(decode_things(&brick).is_none());
        assert!(decode_thing_poses(&brick).is_none());
    }

    #[test]
    fn garbage_is_refused_rather_than_trusted() {
        assert!(decode_brick(&[]).is_none());
        assert!(decode_brick(&[0; 40]).is_none());
        let mut ok = encode_brick(BrickKey { i: 0, j: 0, k: 0 }, 1, &BrickMesh::default())[0].clone();
        // A header claiming more payload than arrived.
        ok[20] = 0xFF;
        ok[21] = 0xFF;
        assert!(decode_brick(&ok).is_none());
        // Triangles indexing vertices that were never sent.
        let bad = {
            let mut p = Vec::new();
            p.extend_from_slice(&1u16.to_le_bytes());
            p.extend_from_slice(&1u16.to_le_bytes());
            p.extend_from_slice(&[0u8; 6]);
            p.extend_from_slice(&[0u8, 0, 5, 0, 9, 0]);
            p
        };
        assert!(decode_payload(&bad).is_none());
    }
}
