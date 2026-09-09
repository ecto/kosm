#!/usr/bin/env bash
# Publish kosm's crates to crates.io, in dependency order.
#
# Everything upstream must be on crates.io first — cargo refuses to publish a
# crate with a git dependency, and the root manifest's phyz/vcad/tang entries
# are git revs until those repos ship. The preflight below checks the three
# that gate everything else; see docs/publishing.md for the whole table.
#
#   scripts/publish.sh            # preflight, then publish
#   scripts/publish.sh --dry-run  # add --dry-run to every cargo publish
set -euo pipefail

DRY=()
[[ "${1:-}" == "--dry-run" ]] && DRY=(--dry-run)

# name:version pairs that must already exist on crates.io.
UPSTREAMS=(
  "tang:0.2.1"
  "phyz:0.4.0"
  "vcad-kernel:0.10.0"
)

# Publish order. Each one's dependencies are published before it.
CRATES=(
  kosm-render
  kosm-registry
  kosm-scan
  kosm-mpm
  kosm
  kosm-train
)

exists() {  # exists <crate> <version>
  local crate="$1" version="$2"
  if command -v cargo-info >/dev/null 2>&1 || cargo info --help >/dev/null 2>&1; then
    cargo info "${crate}@${version}" --registry crates-io >/dev/null 2>&1 && return 0
  fi
  # Fallback: the crates.io API. A published version has an entry in `versions`.
  curl -sfA "kosm-publish (cam@campedersen.com)" \
    "https://crates.io/api/v1/crates/${crate}/${version}" \
    | grep -q '"num"' && return 0
  return 1
}

echo "preflight: upstream crates on crates.io"
missing=()
for pair in "${UPSTREAMS[@]}"; do
  crate="${pair%%:*}"; version="${pair##*:}"
  if exists "$crate" "$version"; then
    echo "  ok      ${crate} ${version}"
  else
    echo "  MISSING ${crate} ${version}"
    missing+=("${crate} ${version}")
  fi
done

if ((${#missing[@]})); then
  echo
  echo "not publishing: ${#missing[@]} upstream(s) missing:"
  printf '  - %s\n' "${missing[@]}"
  echo "publish those first (docs/publishing.md), then rerun."
  exit 1
fi

echo
for crate in "${CRATES[@]}"; do
  echo "publishing ${crate}"
  cargo publish -p "$crate" "${DRY[@]}"
  if [[ "$crate" != "${CRATES[-1]}" ]]; then
    echo "  waiting 30s for the index"
    sleep 30
  fi
done

echo "done: ${CRATES[*]}"
