#!/usr/bin/env python3
"""Build orbital Earth assets from attributed source images. No network at runtime.
Usage: python prepare-earth.py earth-21600.jpg clouds-8192.jpg water-mask.tif [output-directory]
Requires Pillow and numpy. Surface PNGs carry linear water coverage in alpha.
"""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=300_000_000
root=Path(sys.argv[4]) if len(sys.argv) == 5 else Path(__file__).resolve().parents[1]/'assets/earth'
(root/'surface').mkdir(exist_ok=True)
earth=Image.open(sys.argv[1]).convert('RGB').resize((20480,10240),Image.Resampling.LANCZOS)
water=Image.open(sys.argv[3]).convert('L')
# Solar System Scope's white specular regions designate water.
low=earth.resize((2048,1024),Image.Resampling.LANCZOS).convert('RGBA')
low.putalpha(water.resize(low.size,Image.Resampling.LANCZOS));low.save(root/'earth-low.png')
for y in range(10):
 for x in range(20):
  # Two-pixel gutters, wrapping at the dateline and clamping at the poles.
  xs=np.arange(x*1024-2,(x+1)*1024+2)%20480
  ys=np.clip(np.arange(y*1024-2,(y+1)*1024+2),0,10239)
  # Restrict decoding/copying to this row band instead of duplicating the globe.
  band=np.asarray(earth.crop((0,int(ys.min()),20480,int(ys.max())+1)))
  rgb=band[ys-int(ys.min())][:,xs]
  tile=Image.fromarray(rgb).convert('RGBA')
  mw,mh=water.size
  # Resample the same geographic gutter; edge wrap also applies to the mask.
  mx=(xs+.5)*mw/20480-.5;my=(ys+.5)*mh/10240-.5
  arr=np.asarray(water);ix=np.floor(mx).astype(int);iy=np.floor(my).astype(int)
  fx=mx-ix;fy=my-iy
  a=arr[np.clip(iy,0,mh-1)[:,None],(ix%mw)[None,:]]
  b=arr[np.clip(iy,0,mh-1)[:,None],((ix+1)%mw)[None,:]]
  c=arr[np.clip(iy+1,0,mh-1)[:,None],(ix%mw)[None,:]]
  d=arr[np.clip(iy+1,0,mh-1)[:,None],((ix+1)%mw)[None,:]]
  mask=(a*(1-fx)+b*fx)*(1-fy[:,None])+(c*(1-fx)+d*fx)*fy[:,None]
  tile.putalpha(Image.fromarray(np.uint8(np.clip(mask,0,255))))
  tile.save(root/'surface'/f'{y*20+x:03}.png',compress_level=4)
 print(f'surface row {y+1}/10',flush=True)
Image.open(sys.argv[2]).convert('L').save(root/'clouds-8k.png')
# Seamless deterministic 3-D fractal noise, sampled trilinearly in the shader.
rng=np.random.default_rng(62026);n=64;total=np.zeros((n,n,n),dtype=np.float32)
for frequency,weight in [(4,.5),(8,.25),(16,.15),(32,.1)]:
 grid=rng.random((frequency,frequency,frequency),dtype=np.float32)
 p=np.arange(n)*frequency/n;i=np.floor(p).astype(int);f=p-i;f=f*f*(3-2*f)
 layer=np.zeros_like(total)
 for dz in [0,1]:
  for dy in [0,1]:
   for dx in [0,1]:
    layer+=grid[((i+dz)%frequency)[:,None,None],((i+dy)%frequency)[None,:,None],((i+dx)%frequency)[None,None,:]]*((f if dz else 1-f)[:,None,None]*(f if dy else 1-f)[None,:,None]*(f if dx else 1-f)[None,None,:])
 total+=layer*weight
(root/'cloud-noise.raw').write_bytes(np.uint8(np.clip(total,0,1)*255).tobytes())
(root/'surface'/'manifest.json').write_text(json.dumps({'width':20480,'height':10240,'columns':20,'rows':10,'tile_core':1024,'gutter':2,'tiles':200,'source':'NASA BMNG July 2004, 21600x10800','water_mask':'Solar System Scope / INOVE CC BY 4.0'},indent=2))
print('Earth assets ready',flush=True)
