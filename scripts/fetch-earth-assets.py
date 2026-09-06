#!/usr/bin/env python3
"""Download, verify, and prepare the optional orbital Earth asset cache."""
from hashlib import sha256
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "earth"
SOURCES = (
    ("earth-21600.jpg", "https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/bmng-base/july/world.200407.3x21600x10800.jpg", "dea8b4dc8a4f93f5f8bce0c8c85a508a178e7901e9ed8e6bf86e6ce7ef6d61e2"),
    ("clouds-8192.jpg", "https://www.solarsystemscope.com/textures/download/8k_earth_clouds.jpg", "c792eca228989d36ebb45d3ea6ff1198be5e21a25d70d2fbcb2124ffd14ba7f5"),
    ("water-mask.tif", "https://www.solarsystemscope.com/textures/download/8k_earth_specular_map.tif", "194e67ef7f14dd50b402a1a6f8c7ed55810e16515b95f7736091951efe8dc3fb"),
)

def fetch(directory: Path, name: str, url: str, digest: str) -> Path:
    path = directory / name
    print(f"Downloading {name}", flush=True)
    urlretrieve(url, path)
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {name}: {actual}")
    return path

def main() -> int:
    with TemporaryDirectory(prefix="kosm-earth-") as temporary:
        source_dir = Path(temporary)
        paths = [fetch(source_dir, *source) for source in SOURCES]
        ASSETS.parent.mkdir(parents=True, exist_ok=True)
        run([sys.executable, str(ROOT / "scripts" / "prepare-earth.py"), *map(str, paths), str(ASSETS)], check=True)
    print(f"Earth assets ready at {ASSETS}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
