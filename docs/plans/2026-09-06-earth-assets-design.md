# Earth asset distribution

The orbital renderer needs a 151 MB derived Earth cache: tiled Blue Marble albedo, cloud coverage, water mask, and original cloud erosion noise. Tracking it in Git makes every clone, branch, and pull request carry binary payload unrelated to code review.

The repository instead tracks the renderer, `scripts/fetch-earth-assets.py`, and `scripts/prepare-earth.py`. The fetch script downloads the three attributed sources, verifies fixed SHA-256 digests, and invokes the preparation script. Its output is `assets/earth`, which is ignored by Git. The viewer defaults to this cache and accepts `KOSM_EARTH_ASSETS` for a shared or pre-existing cache.

At launch, the renderer verifies the fallback map, cloud map, erosion volume, and surface manifest before allocating GPU resources. A missing cache produces a direct setup command rather than an opaque texture or shader error. Runtime image-tile decoding remains bounded and local; neither the viewer nor the renderer makes network requests.

The fetch script is intentionally explicit rather than a first-run downloader. It keeps program startup deterministic, allows review of provenance and hashes before a network transfer, and preserves offline behavior after setup. Tests cover the renderer’s numerical asset processing; a cache-backed GPU capture validates path resolution and tile loading.
