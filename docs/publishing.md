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

### Why the versions are not all there already

`cargo publish` reads a dependency's `version` and drops its `git` key from the
packaged manifest, so a `version` sitting beside a `rev` is the published form
pre-declared — free to write, and inert until the upstream ships. But cargo
also treats a git dependency's `version` as a real requirement against the
version *in that git tree*. `tang-3dgs`, `tang-tensor` and `tang-train` are
0.1.0 in the pinned rev, so `version = "0.1"` is in the manifest today. phyz's
tree says 0.3.1 and vcad's says 0.9.4; writing `"0.4"` / `"0.10"` there makes
`cargo metadata` fail outright:

```
error: failed to select a version for the requirement `phyz = "^0.4"`
candidate versions found which didn't match: 0.3.1
```

So those two families get their `version` key in the same commit that bumps
their `rev` to the tag carrying the new number. Same for `tang` itself: it
already says `version = "0.2"`, which 0.2.1 satisfies without an edit.

## Publish order

`scripts/publish.sh` is the whole procedure. It publishes, with `sleep 30`
between each so the index catches up, and stops on the first failure:

```
kosm-render → kosm-registry → kosm-scan → kosm-mpm → kosm → kosm-train
```

`kosm-mpm` and `kosm-registry` have no unpublished dependencies and could go
out earlier; they sit here so one run does everything. `kosm-cli` and
`kosm-view` are `publish = false` (see below).

### Preflight

The script refuses to start until the three upstreams that gate everything are
actually on crates.io, and prints which are missing:

| crate | version |
| --- | --- |
| `tang` | 0.2.1 |
| `phyz` | 0.4.0 |
| `vcad-kernel` | 0.10.0 |

It checks with `cargo info <crate>@<version> --registry crates-io`, falling
back to `curl https://crates.io/api/v1/crates/<crate>/<version>`. Those three
stand in for their families — the subcrates ship in the same release.

```bash
scripts/publish.sh --dry-run   # preflight + cargo publish --dry-run
scripts/publish.sh
```

## After ecto/vcad#867

vcad PR #867 is what lets `kosm-render` leave 0.1.x. Once it is on vcad `main`,
one commit here does three things:

1. bump every vcad git dep's `rev` to that vcad `main` commit,
2. set `crates/kosm-render/Cargo.toml` to `version = "0.2.0"` (and its
   in-workspace consumers to `version = "0.2"`),
3. delete the `[patch."https://github.com/ecto/kosm"]` table — it exists only
   to close the back-edge while vcad pins an old kosm-render, and at 0.2.0 it
   would stop applying anyway (a patch must *satisfy* the requirement it
   replaces, which is how `main` broke once already; see commit b8e9cb2).

It is written out as [`after-vcad-867.patch`](after-vcad-867.patch):

```bash
git -C ../vcad ls-remote origin main   # re-check the sha first
git apply docs/after-vcad-867.patch
```

The sha baked into the patch is vcad `origin/main` as of 2026-09-09
(`e9697505b1e0dbd329089b60b645daa6626570bf`) and **must be re-checked at apply
time** — if #867 merged after that, use the merge commit. Then confirm
`cargo metadata` shows exactly one `kosm-render` and tag `kosm-render-v0.2.0`.

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
