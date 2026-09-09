# Publishing

kosm is a workspace of crates that are meant to be depended on by version, not
by path. Today most of what it depends on is not on crates.io yet, so the root
`Cargo.toml` holds git revs. This file says how those become versions and in
what order.

## The rule

- A public crate depends on another public crate by a crates.io `version`.
- A git dependency is a placeholder for something not yet published. It always
  carries a `rev` (never a bare `branch`) and a one-line comment naming the
  version that will replace it.
- Per-developer local checkouts go in an **untracked** `.cargo/config.toml`
  (copy `.cargo/config.toml.example`). The manifest is what CI and a clean
  clone build; that file is one machine's convenience.

## Dependency order

Nothing downstream can be published before everything upstream of it is:

```
tang  →  phyz  →  kosm-render  →  vcad  →  kosm-scan  →  kosm  →  kosm-train
```

`kosm-render` sits between phyz and vcad because vcad depends on it — that is
the back-edge the root manifest's `[patch."https://github.com/ecto/kosm"]`
exists to close while vcad's pin is a git rev. `kosm-mpm` and `kosm-registry`
have no unpublished dependencies and can go out at any time.

## What each repo owes

| dependency | on crates.io | what kosm needs | becomes |
| --- | --- | --- | --- |
| `tang` | 0.2.0 | 18 commits past the 0.2.0 publish: SIMD moments, the `algebraic` reduction feature, row-major GEMM | **tang 0.2.1** |
| `tang-la`, `tang-ad`, `tang-expr` | 0.1.0 | the same tree as above (they are patched only to keep one tang in the graph) | **0.1.1** |
| `tang-tensor`, `tang-train` | — | `Parameter`, `ModuleAdam` | **0.1.0** (first publish) |
| `tang-3dgs` | — | the splat renderer | **0.1.0** (first publish) |
| `phyz` and `phyz-{math,model,rigid,contact,diff,world,collision}` | 0.3.0 / 0.1.0 | local 0.3.1 plus the free-joint Coriolis fix on `claude/rigid-impact-restitution`, unmerged | **phyz 0.4** across the family |
| `phyz-camera` | — | the camera rig | **0.4** (first publish) |
| `vcad` | 0.1.0 | local 0.9.4 on `claude/wgpu-30` (PR #864) | **vcad 0.10** |
| `vcad-kernel`, `-eval`, `-ir`, `-loon`, `-render`, `-kernel-{export,acoustics,optics,raytrace,math,primitives,gpu}` | — | all of them | **0.10** (first publish) |
| `kosm-render` | — | held at 0.1.x while vcad's pinned rev requires `0.1.0` | **0.2.0** in the same commit that bumps the vcad rev |

Each row that lands flips a `git = …, rev = …` entry in the root
`[workspace.dependencies]` into `version = "…"`. Nothing else in the workspace
changes.

## Publish flags

| crate | `publish` | why |
| --- | --- | --- |
| `kosm-render`, `kosm`, `kosm-scan`, `kosm-train`, `kosm-registry`, `kosm-mpm` | default (true) | library crates with full metadata; publishable once their git deps are versions |
| `kosm-cli` | `false` | its `build.rs` walks `sims/` at the workspace root, and `cargo package` cannot carry files from outside the crate directory |
| `kosm-view` | `false` | depends on `vcad-kernel`, `-raytrace`, `-gpu` and `-math`, which are git deps; flip once vcad 0.10 exists |

## The per-developer override

```bash
cp .cargo/config.toml.example .cargo/config.toml
$EDITOR .cargo/config.toml    # point the paths at your checkouts
```

The example uses sibling paths (`../phyz/crates/phyz`, `../tang/crates/tang`),
which is right if your repos live next to each other. A `[patch]` in a cargo
config overrides a `[patch]` for the same source in `Cargo.toml` — silently,
and workspace-wide.

Cargo rewrites `Cargo.lock` to those paths on the first build. Do not commit
that: the committed lock is the clean-clone form, where phyz and tang resolve
from their git revs. `git checkout -- Cargo.lock` before committing, or resolve
once with the config moved aside.

To see exactly what CI and a clean clone see:

```bash
mv .cargo/config.toml /tmp/ && cargo metadata --format-version 1 >/dev/null; mv /tmp/config.toml .cargo/
```
