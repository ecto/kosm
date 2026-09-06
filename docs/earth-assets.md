# Earth asset cache

The orbital viewer keeps generated imagery outside Git. Run `python3 scripts/fetch-earth-assets.py` once from the repository root. It downloads the NASA Blue Marble July 2004 map and Solar System Scope cloud/specular maps, verifies their SHA-256 digests, then invokes `scripts/prepare-earth.py` to create `assets/earth`.

The generated cache contains 200 1028 × 1028 surface tiles, an 8K cloud coverage map, a 2K global fallback, a small deterministic cloud-erosion volume, and a manifest. It is roughly 151 MB and is ignored by Git. The viewer reads this cache at launch; set `KOSM_EARTH_ASSETS` to point it at another prepared directory.

NASA Blue Marble Next Generation, July 2004: Reto Stöckli / NASA Earth Observatory. Source: https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/bmng-base/july/world.200407.3x21600x10800.jpg

Cloud coverage and water/specular mask: Solar System Scope / INOVE, based on NASA data, under CC BY 4.0. Sources: https://www.solarsystemscope.com/textures/download/8k_earth_clouds.jpg and https://www.solarsystemscope.com/textures/download/8k_earth_specular_map.tif. Attribution: https://www.solarsystemscope.com/textures/
