# Publishing

kosm is a workspace of crates that are meant to be depended on by version, not
by path. As of 2026-09-10 every external it depends on is on crates.io, so the
root `Cargo.toml` holds versions only. This file says what those are and in
what order kosm's own crates go out.

## The rule

- A public crate depends on another public crate by a crates.io `version`.
- A git dependency is a placeholder for something not yet published. There are
  none left; if one comes back it carries a `rev` (never a bare `branch`) and a
  one-line comment naming the version that will replace it.
- Per-developer local checkouts go in an **untracked** `.cargo/config.toml`
  (copy `.cargo/config.toml.example`). The manifest is what CI and a clean
  clone build; that file is one machine's convenience.

## Dependency order

Nothing downstream can be published before everything upstream of it is:

```
tang  →  phyz  →  kosm-render  →  vcad  →  kosm-scan  →  kosm  →  kosm-train
```

`kosm-render` sits between phyz and vcad because vcad depends on it. vcad 0.10
tracks the published `kosm-render`, so that back-edge no longer needs a
`[patch]` in the root manifest. `kosm-mpm` and `kosm-registry` have no
unpublished dependencies and can go out at any time.

## What each repo owes

Nothing. Every row landed on 2026-09-10: tang 0.2.1; tang-la, tang-ad,
tang-expr 0.1.1; tang-tensor, tang-train, tang-3dgs 0.1.0; the phyz family
(including the first `phyz-camera`) 0.4.0; the vcad family 0.10.0. The root
`[workspace.dependencies]` names those versions, `kosm-render` is 0.2.0, and
the `[patch]` tables that unified the git trees are gone.

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

## After ecto/vcad#867 (done)

vcad #867 is what let `kosm-render` leave 0.1.x. It merged, vcad 0.10.0 shipped
tracking `kosm-render` 0.2, and the flip landed here on 2026-09-10: every git
dep became a version, `crates/kosm-render` went to 0.2.0 (its in-workspace
consumers to `"0.2"`), and the `[patch."https://github.com/ecto/kosm"]` table
was deleted — it existed only to close the back-edge while vcad pinned an old
kosm-render, and at 0.2.0 it would have stopped applying anyway (a patch must
*satisfy* the requirement it replaces, which is how `main` broke once already;
see commit b8e9cb2). `cargo metadata` shows exactly one `kosm-render`.

## Publish flags

| crate | `publish` | why |
| --- | --- | --- |
| `kosm-render`, `kosm`, `kosm-scan`, `kosm-train`, `kosm-registry`, `kosm-mpm` | default (true) | library crates with full metadata; publishable once their git deps are versions |
| `kosm-cli` | `false` | its `build.rs` walks `sims/` at the workspace root, and `cargo package` cannot carry files from outside the crate directory |
| `kosm-view` | `false` | the `view` feature's eframe/egui/wgpu stack is the heaviest thing in the graph and nothing depends on it by version; its deps are all published, so this can flip whenever it earns its keep |

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
that: the committed lock is the clean-clone form, where phyz, tang and vcad
resolve from crates.io. `git checkout -- Cargo.lock` before committing, or
resolve once with the config moved aside.

To see exactly what CI and a clean clone see:

```bash
mv .cargo/config.toml /tmp/ && cargo metadata --format-version 1 >/dev/null; mv /tmp/config.toml .cargo/
```
