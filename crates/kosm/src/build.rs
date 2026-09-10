//! Geometry in Rust: a builder over vcad's IR.
//!
//! The level used to be a `.loon` file. It is a Rust function now, and this
//! is the vocabulary it is written in. What comes out is the same
//! [`vcad_ir::Document`] the loon evaluator used to produce, so everything
//! downstream — [`crate::colliders`] for phyz, [`crate::brep`] for the
//! picture, `vcad_eval` for the STL — is untouched.
//!
//! ```
//! use kosm::build::{build, Params};
//! let built = build(&Params::default(), |b| {
//!     let tilt = b.param("tilt", 5.0);
//!     b.body("plate").boxed(400.0, 400.0, 10.0).rotate_y(tilt);
//!     b.body("marble").sphere(8.0).glass().dynamic(0.012).at(0.0, 0.0, 30.0);
//! }).unwrap();
//! assert_eq!(built.world.param("tilt"), Some(5.0));
//! let steeper = built.with(&[("tilt", 9.0)]).unwrap();
//! assert_eq!(steeper.world.param("tilt"), Some(9.0));
//! ```
//!
//! **Units.** Everything authored here is millimetres and degrees, vcad's
//! own units and the ones the `.loon` files used. The conversion to phyz's
//! metres happens exactly where it always did: [`crate::scene::MM`], applied
//! by [`crate::colliders`] on the way into the model, and by
//! [`Builder::mm`] for a length a sim needs in metres.
//!
//! **Origins**, also vcad's: [`Builder::cube`] has a corner at the origin,
//! [`Builder::cylinder`]'s base is at `z = 0`, [`Builder::sphere`] is
//! centred. [`Builder::boxed`] is the centred cube every level wrote by hand.
//!
//! **Rebuilding.** [`build`] returns a [`Built`] that keeps the closure, so
//! [`Built::with`] re-runs it with an overridden knob and gets a world that
//! is consistent all the way down — new document, new colliders, new params.
//! That is the honest version of the old "rewrite the `defparam` and
//! re-evaluate the file".

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use phyz_math::{GRAVITY, Mat3, SpatialInertia, SpatialTransform, Vec3};
use phyz_model::{Geometry, ModelBuilder};
use vcad_ir::{CsgOp, Document, Node, NodeId, SceneEntry};

use crate::colliders::{Derived, colliders_from_document};
use crate::material::Material as Substance;
use crate::scene::MM;
use crate::world::{Material, Param, World};

/// Knob overrides for a build. The sim's `scene(params: &Params) -> Built`
/// takes one of these; an empty one means "every knob at its default".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Params(HashMap<String, f64>);

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, name: &str) -> Option<f64> {
        self.0.get(name).copied()
    }

    pub fn set(&mut self, name: impl Into<String>, value: f64) -> &mut Self {
        self.0.insert(name.into(), value);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<&[(&str, f64)]> for Params {
    fn from(pairs: &[(&str, f64)]) -> Self {
        let mut p = Params::new();
        for (name, value) in pairs {
            p.set(*name, *value);
        }
        p
    }
}

impl FromIterator<(String, f64)> for Params {
    fn from_iter<I: IntoIterator<Item = (String, f64)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// A source of authored geometry: the document, what its roots are called,
/// and where it came from. The bake and the parts walk take one of these
/// rather than a `Built`, so a caller with a document from somewhere else —
/// a URDF import, a scan — can hand it over without going through the
/// builder.
pub trait Authored {
    fn document(&self) -> &Document;
    /// The roots' names, in document order. vcad's `SceneEntry` carries a
    /// material and nothing else, so the name lives with the author.
    fn root_names(&self) -> Vec<String>;
    /// What to record in a baked map's provenance.
    fn origin(&self) -> String;
}

impl Authored for Built {
    fn document(&self) -> &Document {
        &self.document
    }
    fn root_names(&self) -> Vec<String> {
        self.bodies.iter().map(|b| b.name.clone()).collect()
    }
    fn origin(&self) -> String {
        format!("{} bodies built in rust", self.bodies.len())
    }
}

/// What a build produced: the CAD, the knobs, the colliders, the world, and
/// the recipe to do it again with a knob turned.
pub struct Built {
    /// The authored document — millimetres, exactly what a `.loon` used to
    /// evaluate to. `colliders_from_document`, `brep::Scene` and
    /// `vcad_eval::evaluate_document` all take this.
    pub document: Document,
    /// Every knob the closure asked for, with the value it resolved to.
    pub params: Vec<Param>,
    /// Per body, in declaration order.
    pub bodies: Vec<BuiltBody>,
    /// A world over the bodies above: static bodies fixed, dynamic bodies
    /// free, colliders derived from the document. A sim whose physics needs
    /// more than that takes `document` and builds its own rig.
    pub world: World,
    /// Collider warnings from every body, one line each: a `Difference` that
    /// fell back to a hull says so here.
    pub warnings: Vec<String>,
    /// Informational lines about decompositions that succeeded.
    pub notes: Vec<String>,
    recipe: Recipe,
}

/// One authored body, after the build.
pub struct BuiltBody {
    pub name: String,
    pub material: String,
    /// The substance this body was authored with, when it was authored by
    /// value through [`Body::substance`] rather than by name. [`Self::substance`]
    /// is what a caller should ask; this is where the answer is kept when the
    /// author had one that the library does not.
    pub authored_substance: Option<Substance>,
    /// `None` for a static body.
    pub mass: Option<f64>,
    /// Where a dynamic body starts, metres. Static bodies are at the origin
    /// and carry their placement in the document instead.
    pub origin: [f64; 3],
    /// This body's root node in [`Built::document`].
    pub root: NodeId,
    /// Colliders derived from that root, in metres.
    pub colliders: Derived,
}

impl BuiltBody {
    /// This body's substance: the one it was authored with, or the library's
    /// entry for its material name.
    ///
    /// The `no-collide` marker is stripped first, so a roof authored as
    /// `"no-collide galvanized"` still answers "galvanized". A name the
    /// library does not know gives `None` — no fallback substance, because
    /// guessing a density is worse than admitting there isn't one.
    ///
    /// ```
    /// use kosm::prelude::*;
    /// let built = build(&Params::default(), |b| {
    ///     b.body("hoop").material("brass").cylinder(10.0, 4.0);
    /// })?;
    /// let brass = built.bodies[0].substance().expect("brass is in the library");
    /// assert_eq!(brass.name, "brass");
    /// assert!(brass.contact().friction > 0.0);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn substance(&self) -> Option<Substance> {
        self.authored_substance
            .clone()
            .or_else(|| crate::material::named(crate::materials::split(&self.material).1))
    }
}

type Closure = Arc<dyn Fn(&Builder) + Send + Sync>;

#[derive(Clone)]
struct Recipe {
    closure: Closure,
    params: Params,
}

impl Built {
    /// A knob's resolved value.
    pub fn param(&self, name: &str) -> Option<f64> {
        self.params.iter().find(|p| p.name == name).map(|p| p.value)
    }

    /// The same, in metres: the authored units are millimetres.
    pub fn mm(&self, name: &str) -> Option<f64> {
        self.param(name).map(|v| v * MM)
    }

    /// A knob, or an error naming it.
    pub fn parameter(&self, name: &str) -> anyhow::Result<f64> {
        self.param(name).ok_or_else(|| anyhow::anyhow!("the level has no `{name}`"))
    }

    pub fn parameter_or(&self, name: &str, default: f64) -> f64 {
        self.param(name).unwrap_or(default)
    }

    /// A knob's length in metres: the authored units are millimetres, and
    /// this is the one place they cross.
    pub fn millimetres(&self, name: &str) -> anyhow::Result<f64> {
        Ok(self.parameter(name)? * MM)
    }

    /// Re-run the closure with these knobs turned. Everything downstream —
    /// the document, the colliders, the world's params — is rebuilt from it,
    /// so nothing can be left describing the old value.
    pub fn with(&self, updates: &[(&str, f64)]) -> anyhow::Result<Built> {
        let mut params = self.recipe.params.clone();
        for (name, value) in updates {
            params.set(*name, *value);
        }
        run(self.recipe.closure.clone(), params)
    }

}

/// Author a world in Rust.
///
/// The closure runs once now and again on every [`Built::with`], so it must
/// be a function of its `Builder` and nothing else. Knobs come from
/// [`Builder::param`], which returns the override in `params` when there is
/// one and the default otherwise.
pub fn build<F>(params: &Params, closure: F) -> anyhow::Result<Built>
where
    F: Fn(&Builder) + Send + Sync + 'static,
{
    run(Arc::new(closure), params.clone())
}

fn run(closure: Closure, params: Params) -> anyhow::Result<Built> {
    let builder = Builder {
        inner: Rc::new(RefCell::new(Inner {
            doc: Document::new(),
            next: 1,
            params: Vec::new(),
            overrides: params.clone(),
            bodies: Vec::new(),
        })),
    };
    closure(&builder);
    let inner = Rc::try_unwrap(builder.inner)
        .map_err(|_| anyhow::anyhow!("a Shape outlived the build closure"))?
        .into_inner();
    finish(inner, Recipe { closure, params })
}

fn finish(inner: Inner, recipe: Recipe) -> anyhow::Result<Built> {
    let Inner { mut doc, params, bodies, .. } = inner;
    doc.roots.clear();
    for body in &bodies {
        doc.roots.push(SceneEntry { root: body.root, material: body.material.clone(), visible: None });
    }

    let mut built_bodies = Vec::new();
    for body in &bodies {
        // A body that is only drawn — the `no-collide` marker, or an explicit
        // `decorative()` — is never walked for colliders: a decomposition it
        // would only warn about is a decomposition nobody asked for.
        let colliders = if body.drawn_only || !crate::materials::split(&body.material).0 {
            Derived { colliders: Vec::new(), warnings: Vec::new(), notes: Vec::new(), removed: Vec::new() }
        } else {
            let mut one = Document::new();
            one.nodes = doc.nodes.clone();
            one.roots = vec![SceneEntry { root: body.root, material: body.material.clone(), visible: None }];
            colliders_from_document(&one)?
        };
        built_bodies.push(BuiltBody {
            name: body.name.clone(),
            material: body.material.clone(),
            authored_substance: body.substance.clone(),
            mass: body.mass,
            origin: [body.origin[0] * MM, body.origin[1] * MM, body.origin[2] * MM],
            root: body.root,
            colliders,
        });
    }

    let world = world_of(&built_bodies, &params);
    let warnings = built_bodies.iter().flat_map(|b| b.colliders.warnings.iter().cloned()).collect();
    let notes = built_bodies.iter().flat_map(|b| b.colliders.notes.iter().cloned()).collect();
    Ok(Built { document: doc, params, bodies: built_bodies, world, warnings, notes, recipe })
}

/// A phyz rig over the authored bodies: dynamic bodies free, static bodies
/// fixed with the document's colliders on them.
fn world_of(bodies: &[BuiltBody], params: &[Param]) -> World {
    let mut mb = ModelBuilder::new().gravity(Vec3::new(0.0, 0.0, -GRAVITY)).dt(1e-3);
    let static_inertia = SpatialInertia::new(1.0, Vec3::zeros(), Mat3::identity() * 0.01);
    for body in bodies {
        match body.mass {
            Some(m) => {
                let i = inertia_of(body, m);
                mb = mb.add_free_body(&body.name, -1, SpatialTransform::identity(), i);
            }
            None => {
                mb = mb.add_fixed_body(&body.name, -1, SpatialTransform::identity(), static_inertia);
            }
        }
    }
    let mut model = mb.build();
    let mut index = 0usize;
    for body in bodies {
        match body.mass {
            Some(_) => {
                if let Some(g) = body.colliders.colliders.first() {
                    model.bodies[index].geometry = Some(g.geometry.clone());
                }
            }
            None => {
                model.bodies[index].collisions = body.colliders.colliders.clone();
                model.bodies[index].visuals = body.colliders.colliders.clone();
            }
        }
        index += 1;
    }
    let mut state = model.default_state();
    // A free joint's q is [wx wy wz x y z]; put each dynamic body where it
    // was authored.
    let mut q = 0usize;
    for (body, joint) in bodies.iter().zip(model.joints.iter()) {
        let n = joint.ndof();
        if body.mass.is_some() && n >= 6 {
            state.q[q + 3] = body.origin[0];
            state.q[q + 4] = body.origin[1];
            state.q[q + 5] = body.origin[2];
        }
        q += n;
    }
    let materials = bodies
        .iter()
        .map(|b| {
            // The authored substance first: a body given a `Material` by value
            // carries constants the library may not have, and its albedo is
            // the one the author meant.
            let s = b.substance();
            Material {
                name: b.material.clone(),
                albedo: s
                    .as_ref()
                    .map(|s| s.colour())
                    .unwrap_or_else(|| crate::materials::colour(crate::materials::split(&b.material).1)),
                roughness: s.as_ref().map(|s| s.roughness).unwrap_or(0.5),
                metallic: s
                    .as_ref()
                    .map(|s| matches!(s.optics, crate::material::Optics::Conductor) as u8 as f64)
                    .unwrap_or(0.0),
                ..Material::default()
            }
        })
        .collect();
    let mut world = World::from_phyz(model, state).with_params(params.to_vec());
    world.materials = materials;
    world
}

/// The inertia of a body's first collider, scaled to `m`. A sphere, a box and
/// a cylinder have closed forms; anything else is treated as its bounding box.
fn inertia_of(body: &BuiltBody, m: f64) -> SpatialInertia {
    let d = match body.colliders.colliders.first().map(|g| &g.geometry) {
        Some(Geometry::Sphere { radius }) => {
            let i = 0.4 * m * radius * radius;
            Vec3::new(i, i, i)
        }
        Some(Geometry::Box { half_extents }) => {
            let (x, y, z) = (2.0 * half_extents.x, 2.0 * half_extents.y, 2.0 * half_extents.z);
            Vec3::new(m * (y * y + z * z) / 12.0, m * (x * x + z * z) / 12.0, m * (x * x + y * y) / 12.0)
        }
        Some(Geometry::Cylinder { radius, height }) => {
            let side = m * (3.0 * radius * radius + height * height) / 12.0;
            Vec3::new(side, side, 0.5 * m * radius * radius)
        }
        _ => Vec3::new(m * 1e-3, m * 1e-3, m * 1e-3),
    };
    SpatialInertia::new(m, Vec3::zeros(), Mat3::from_diagonal(&d))
}

// ---- the builder ------------------------------------------------------------

struct BodyDef {
    name: String,
    material: String,
    substance: Option<Substance>,
    mass: Option<f64>,
    origin: [f64; 3],
    root: NodeId,
    drawn_only: bool,
}

struct Inner {
    doc: Document,
    next: NodeId,
    params: Vec<Param>,
    overrides: Params,
    bodies: Vec<BodyDef>,
}

impl Inner {
    fn add(&mut self, name: Option<String>, op: CsgOp) -> NodeId {
        let id = self.next;
        self.next += 1;
        self.doc.nodes.insert(id, Node { id, name, op });
        id
    }
}

/// The authoring handle. Every method takes `&self`, so a shape and a body
/// can be built side by side without fighting the borrow checker.
#[derive(Clone)]
pub struct Builder {
    inner: Rc<RefCell<Inner>>,
}

fn v3(x: f64, y: f64, z: f64) -> vcad_ir::Vec3 {
    vcad_ir::Vec3::new(x, y, z)
}

impl Builder {
    /// A named knob. Returns the override when this build has one for it,
    /// the default otherwise, and registers it on the resulting world either
    /// way — so `world.param(name)` and `built.with(&[(name, v)])` agree.
    pub fn param(&self, name: &str, default: f64) -> f64 {
        let mut inner = self.inner.borrow_mut();
        let value = inner.overrides.get(name).unwrap_or(default);
        if let Some(existing) = inner.params.iter_mut().find(|p| p.name == name) {
            existing.value = value;
        } else {
            inner.params.push(Param::new(name, value));
        }
        value
    }

    /// A knob in metres, for a length the physics wants: `param * MM`.
    pub fn mm(&self, name: &str, default: f64) -> f64 {
        self.param(name, default) * MM
    }

    fn shape(&self, op: CsgOp) -> Shape {
        let id = self.inner.borrow_mut().add(None, op);
        Shape { inner: self.inner.clone(), id }
    }

    /// vcad's cube: a corner at the origin, extending into +x +y +z.
    pub fn cube(&self, x: f64, y: f64, z: f64) -> Shape {
        self.shape(CsgOp::Cube { size: v3(x, y, z) })
    }

    /// A box centred on the origin — the `box-c` every `.loon` defined.
    pub fn boxed(&self, x: f64, y: f64, z: f64) -> Shape {
        self.cube(x, y, z).translate(-0.5 * x, -0.5 * y, -0.5 * z)
    }

    /// Along +z, base at `z = 0`.
    pub fn cylinder(&self, radius: f64, height: f64) -> Shape {
        self.shape(CsgOp::Cylinder { radius, height, segments: 0 })
    }

    /// Centred on the origin, axis along z.
    pub fn torus(&self, major_radius: f64, minor_radius: f64) -> Shape {
        self.shape(CsgOp::Torus { major_radius, minor_radius, segments: 0 })
    }

    /// Centred on the origin.
    pub fn sphere(&self, radius: f64) -> Shape {
        self.shape(CsgOp::Sphere { radius, segments: 0 })
    }

    /// Along +z, base at `z = 0`; `radius_top` of zero is a point.
    pub fn cone(&self, radius_bottom: f64, radius_top: f64, height: f64) -> Shape {
        self.shape(CsgOp::Cone { radius_bottom, radius_top, height, segments: 0 })
    }

    /// The box spanning `[x0, x1] × [y0, y1] × [z0, z1]`. Two opposite
    /// corners, which is how a level usually knows where a thing goes.
    pub fn box_at(&self, x: [f64; 2], y: [f64; 2], z: [f64; 2]) -> Shape {
        self.cube(x[1] - x[0], y[1] - y[0], z[1] - z[0]).at(x[0], y[0], z[0])
    }

    /// A cylinder of radius `r` and length `l` lying along x, centred on the
    /// origin.
    pub fn rod_x(&self, r: f64, l: f64) -> Shape {
        self.cylinder(r, l).rotate_y(90.0).at(-0.5 * l, 0.0, 0.0)
    }

    /// The same, along y.
    pub fn rod_y(&self, r: f64, l: f64) -> Shape {
        self.cylinder(r, l).rotate_x(90.0).at(0.0, 0.5 * l, 0.0)
    }

    /// The same, along z.
    pub fn rod_z(&self, r: f64, l: f64) -> Shape {
        self.cylinder(r, l).at(0.0, 0.0, -0.5 * l)
    }

    /// Nothing — the identity for union, and a body that is switched off.
    pub fn empty(&self) -> Shape {
        self.shape(CsgOp::Empty)
    }

    /// Declare a body. Static and clay-grey until told otherwise.
    pub fn body(&self, name: &str) -> Body {
        let root = self.inner.borrow_mut().add(Some(name.to_owned()), CsgOp::Empty);
        let mut inner = self.inner.borrow_mut();
        inner.bodies.push(BodyDef {
            name: name.to_owned(),
            material: "clay".into(),
            substance: None,
            mass: None,
            origin: [0.0; 3],
            root,
            drawn_only: false,
        });
        let index = inner.bodies.len() - 1;
        drop(inner);
        Body { inner: self.inner.clone(), index }
    }
}

/// A CSG subtree under construction. Millimetres and degrees.
#[derive(Clone)]
pub struct Shape {
    inner: Rc<RefCell<Inner>>,
    id: NodeId,
}

impl Shape {
    fn wrap(self, op: impl FnOnce(NodeId) -> CsgOp) -> Shape {
        let op = op(self.id);
        let id = self.inner.borrow_mut().add(None, op);
        Shape { inner: self.inner, id }
    }

    /// Name this node — it is what a collider warning calls itself.
    pub fn named(self, name: &str) -> Shape {
        if let Some(node) = self.inner.borrow_mut().doc.nodes.get_mut(&self.id) {
            node.name = Some(name.to_owned());
        }
        self
    }

    pub fn translate(self, x: f64, y: f64, z: f64) -> Shape {
        self.wrap(|child| CsgOp::Translate { child, offset: v3(x, y, z) })
    }

    /// `translate`, spelled the way a placement reads.
    pub fn at(self, x: f64, y: f64, z: f64) -> Shape {
        self.translate(x, y, z)
    }

    /// Euler XYZ, degrees: X first, then Y, then Z — vcad's convention.
    pub fn rotate(self, x: f64, y: f64, z: f64) -> Shape {
        self.wrap(|child| CsgOp::Rotate { child, angles: v3(x, y, z) })
    }

    pub fn rotate_x(self, deg: f64) -> Shape {
        self.rotate(deg, 0.0, 0.0)
    }

    pub fn rotate_y(self, deg: f64) -> Shape {
        self.rotate(0.0, deg, 0.0)
    }

    pub fn rotate_z(self, deg: f64) -> Shape {
        self.rotate(0.0, 0.0, deg)
    }

    pub fn scale(self, x: f64, y: f64, z: f64) -> Shape {
        self.wrap(|child| CsgOp::Scale { child, factor: v3(x, y, z) })
    }

    pub fn union(self, other: Shape) -> Shape {
        let right = other.id;
        self.wrap(|left| CsgOp::Union { left, right })
    }

    /// `self` minus `tool`. The collider path decomposes this into convex
    /// pieces that stay out of what was removed; see [`crate::colliders`].
    pub fn difference(self, tool: Shape) -> Shape {
        let right = tool.id;
        self.wrap(|left| CsgOp::Difference { left, right })
    }

    pub fn intersection(self, other: Shape) -> Shape {
        let right = other.id;
        self.wrap(|left| CsgOp::Intersection { left, right })
    }

    /// `count` copies including this one, each `spacing` further along
    /// `direction` than the last.
    pub fn linear_pattern(self, direction: [f64; 3], count: u32, spacing: f64) -> Shape {
        self.wrap(|child| CsgOp::LinearPattern {
            child,
            direction: v3(direction[0], direction[1], direction[2]),
            count,
            spacing,
        })
    }

    /// `count` copies including this one, sweeping `angle_deg` in total about
    /// the axis through `origin` along `axis`.
    pub fn circular_pattern(self, origin: [f64; 3], axis: [f64; 3], count: u32, angle_deg: f64) -> Shape {
        self.wrap(|child| CsgOp::CircularPattern {
            child,
            axis_origin: v3(origin[0], origin[1], origin[2]),
            axis_dir: v3(axis[0], axis[1], axis[2]),
            count,
            angle_deg,
        })
    }

    pub fn mirror(self, origin: [f64; 3], normal: [f64; 3]) -> Shape {
        self.wrap(|child| CsgOp::Mirror {
            child,
            plane_origin: v3(origin[0], origin[1], origin[2]),
            plane_normal: v3(normal[0], normal[1], normal[2]),
        })
    }
}

/// A named body: one root in the document, one body in the model.
///
/// Shapes handed to [`Body::add`] are unioned onto the body's root. The
/// transform methods turn the whole body: for a static body that is a
/// document transform (the colliders come out of the document already
/// placed), and for a dynamic body [`Body::at`] is where it starts instead,
/// since a moving body's pose is state and not CAD.
#[derive(Clone)]
pub struct Body {
    inner: Rc<RefCell<Inner>>,
    index: usize,
}

impl Body {
    fn map_root(&self, f: impl FnOnce(&mut Inner, NodeId) -> NodeId) -> &Self {
        let mut inner = self.inner.borrow_mut();
        let root = inner.bodies[self.index].root;
        let new = f(&mut inner, root);
        inner.bodies[self.index].root = new;
        self
    }

    fn is_dynamic(&self) -> bool {
        self.inner.borrow().bodies[self.index].mass.is_some()
    }

    /// Union a shape onto this body.
    pub fn add(&self, shape: Shape) -> &Self {
        let right = shape.id;
        self.map_root(|inner, left| {
            if matches!(inner.doc.nodes.get(&left).map(|n| &n.op), Some(CsgOp::Empty)) {
                let name = inner.doc.nodes.get(&left).and_then(|n| n.name.clone());
                if let Some(node) = inner.doc.nodes.get_mut(&right) {
                    node.name = name;
                }
                right
            } else {
                inner.add(None, CsgOp::Union { left, right })
            }
        })
    }

    /// Replace this body's shape with `self.shape() - tool`.
    pub fn difference(&self, tool: Shape) -> &Self {
        let right = tool.id;
        self.map_root(|inner, left| inner.add(None, CsgOp::Difference { left, right }))
    }

    fn builder(&self) -> Builder {
        Builder { inner: self.inner.clone() }
    }

    pub fn cube(&self, x: f64, y: f64, z: f64) -> &Self {
        self.add(self.builder().cube(x, y, z))
    }

    pub fn boxed(&self, x: f64, y: f64, z: f64) -> &Self {
        self.add(self.builder().boxed(x, y, z))
    }

    pub fn cylinder(&self, radius: f64, height: f64) -> &Self {
        self.add(self.builder().cylinder(radius, height))
    }

    pub fn sphere(&self, radius: f64) -> &Self {
        self.add(self.builder().sphere(radius))
    }

    /// Where the body sits. A static body's placement goes into the document;
    /// a dynamic body's is its initial position instead. Millimetres.
    pub fn at(&self, x: f64, y: f64, z: f64) -> &Self {
        if self.is_dynamic() {
            self.inner.borrow_mut().bodies[self.index].origin = [x, y, z];
            self
        } else {
            self.map_root(|inner, child| inner.add(None, CsgOp::Translate { child, offset: v3(x, y, z) }))
        }
    }

    /// Euler XYZ, degrees.
    pub fn rotate(&self, x: f64, y: f64, z: f64) -> &Self {
        self.map_root(|inner, child| inner.add(None, CsgOp::Rotate { child, angles: v3(x, y, z) }))
    }

    pub fn rotate_x(&self, deg: f64) -> &Self {
        self.rotate(deg, 0.0, 0.0)
    }

    pub fn rotate_y(&self, deg: f64) -> &Self {
        self.rotate(0.0, deg, 0.0)
    }

    pub fn rotate_z(&self, deg: f64) -> &Self {
        self.rotate(0.0, 0.0, deg)
    }

    pub fn scale(&self, x: f64, y: f64, z: f64) -> &Self {
        self.map_root(|inner, child| inner.add(None, CsgOp::Scale { child, factor: v3(x, y, z) }))
    }

    /// The material name. [`crate::materials`] turns it into a colour, and
    /// the `no-collide` prefix still means what it meant.
    pub fn material(&self, name: &str) -> &Self {
        self.inner.borrow_mut().bodies[self.index].material = name.to_owned();
        self
    }

    /// The material, by value: a [`Substance`] rather than a name.
    ///
    /// The body takes the substance's own name, so the document, the colour
    /// path and the `no-collide` rule all behave exactly as if
    /// [`Body::material`] had been called with it — and
    /// [`BuiltBody::substance`] hands the constants back afterwards, so a sim
    /// can ask this body for its `contact()` or its `modal()` without a
    /// second table.
    ///
    /// ```
    /// use kosm::prelude::*;
    /// let bronze = material::named("bell bronze").expect("bell bronze");
    /// let built = build(&Params::default(), move |b| {
    ///     b.body("bell").substance(&bronze).sphere(30.0).dynamic(1.0);
    /// })?;
    /// let m = built.bodies[0].substance().expect("authored by value");
    /// assert_eq!(m.name, "bell bronze");
    /// assert!(m.loss < 1e-3, "a bell is a low loss factor");
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn substance(&self, substance: &Substance) -> &Self {
        let mut inner = self.inner.borrow_mut();
        inner.bodies[self.index].material = substance.name.clone();
        inner.bodies[self.index].substance = Some(substance.clone());
        self
    }

    pub fn glass(&self) -> &Self {
        self.material("glass")
    }

    /// Fixed to the world. The default.
    pub fn static_(&self) -> &Self {
        self.inner.borrow_mut().bodies[self.index].mass = None;
        self
    }

    /// Drawn, never collided: the world leaves it out and no colliders are
    /// derived for it. A `no-collide` material means the same thing.
    pub fn decorative(&self) -> &Self {
        self.inner.borrow_mut().bodies[self.index].drawn_only = true;
        self
    }

    /// A free body of this mass, in kilogrammes.
    pub fn dynamic(&self, mass: f64) -> &Self {
        self.inner.borrow_mut().bodies[self.index].mass = Some(mass);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_body_is_a_root_and_a_collider() {
        let built = build(&Params::default(), |b| {
            b.body("plate").boxed(300.0, 200.0, 10.0).material("pla");
        })
        .expect("builds");
        assert_eq!(built.document.roots.len(), 1);
        assert_eq!(built.bodies[0].colliders.colliders.len(), 1);
        assert_eq!(built.world.model().bodies.len(), 1);
        match &built.bodies[0].colliders.colliders[0].geometry {
            Geometry::Box { half_extents } => {
                assert!((half_extents.x - 0.150).abs() < 1e-12, "millimetres cross once");
                assert!((half_extents.z - 0.005).abs() < 1e-12);
            }
            other => panic!("expected a box, got {other:?}"),
        }
    }

    #[test]
    fn knobs_resolve_and_rebuild() {
        let built = build(&Params::default(), |b| {
            let r = b.param("marble_r", 10.0);
            b.body("marble").sphere(r).glass().dynamic(0.012);
        })
        .expect("builds");
        assert_eq!(built.param("marble_r"), Some(10.0));
        assert_eq!(built.world.param("marble_r"), Some(10.0));

        let bigger = built.with(&[("marble_r", 20.0)]).expect("rebuilds");
        assert_eq!(bigger.param("marble_r"), Some(20.0));
        match &bigger.bodies[0].colliders.colliders[0].geometry {
            Geometry::Sphere { radius } => assert!((radius - 0.020).abs() < 1e-12),
            other => panic!("expected a sphere, got {other:?}"),
        }
    }

    #[test]
    fn a_difference_goes_through_the_decomposition() {
        // A tube: 25 mm outside, bored to 20, right through.
        let built = build(&Params::default(), |b| {
            let bore = b.cylinder(20.0, 40.0).at(0.0, 0.0, -5.0);
            b.body("tube").add(b.cylinder(25.0, 30.0).difference(bore));
        })
        .expect("builds");
        let d = &built.bodies[0].colliders;
        assert!(d.warnings.is_empty(), "should not have fallen back: {:?}", d.warnings);
        assert!(d.colliders.len() >= 6, "expected a ring of pieces, got {}", d.colliders.len());
        assert!(crate::colliders::verify_no_intrusion(d) < 1e-3);
    }

    #[test]
    fn a_pattern_is_a_union_of_copies() {
        let built = build(&Params::default(), |b| {
            b.body("posts").add(b.cylinder(2.0, 10.0).linear_pattern([1.0, 0.0, 0.0], 4, 25.0));
        })
        .expect("builds");
        assert_eq!(built.bodies[0].colliders.colliders.len(), 4);
    }

    #[test]
    fn a_dynamic_body_starts_where_it_was_put() {
        let built = build(&Params::default(), |b| {
            b.body("marble").sphere(8.0).dynamic(0.012).at(10.0, 0.0, 30.0);
        })
        .expect("builds");
        let q = &built.world.state().q;
        assert!((q[3] - 0.010).abs() < 1e-12);
        assert!((q[5] - 0.030).abs() < 1e-12);
    }
}
