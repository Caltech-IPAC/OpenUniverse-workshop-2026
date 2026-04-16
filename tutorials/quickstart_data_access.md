---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Quickstart: Accessing OpenUniverse2024 Data

## Learning Goals

By the end of this tutorial, you will be able to:

- Browse the OpenUniverse2024 data directories on S3
- Explore the structure of Roman and Rubin FITS image files
- Read the OpenUniverse2024 parquet catalogs (transient, galaxy, and galaxy flux)
- Query Roman and Rubin images covering a sky position using the IRSA SIA service

## Introduction

The [OpenUniverse2024](https://arxiv.org/abs/2501.05632) simulation suite delivers ~70 deg² of matched optical/infrared imagery for both the LSST Wide-Fast-Deep (WFD) and the Nancy Grace Roman Space Telescope high-latitude survey, producing roughly 400 TB of publicly available synthetic imaging and catalogs. All data are stored in the cloud (AWS S3) and can be accessed anonymously without any credentials.

This tutorial is a focused introduction to data access only. It covers the three main types of data products:

1. **FITS images** — Roman and Rubin simulated science images stored in S3
2. **Parquet catalogs** — transient (SNANA), galaxy, and galaxy-flux tables, indexed by HEALPix sky region
3. **Image search via SIA** — querying which images cover a given sky position using astroquery and the IRSA Simple Image Access service

No astrophysical analysis is performed here. For science workflows that build on these access patterns, see the [TDE Light Curve](TDE_light_curve) and [SED Fitting](SED_fit) tutorials in this repository.

### Instructions

This notebook is designed to be run sequentially from top to bottom. All code is self-contained and relies on publicly accessible data.

### Input

- OpenUniverse2024 Roman and Rubin images and catalogs on AWS S3 (`s3://nasa-irsa-simulations/`)

### Output

- A gallery of example Roman FITS images
- Summary of parquet catalog structure and contents
- A table of image files overlapping a chosen sky position

## Imports

```{code-cell} ipython3
# Uncomment the next line to install dependencies if needed.
# !pip install numpy astropy s3fs photutils matplotlib pyarrow hpgeom astroquery
```

```{code-cell} ipython3
from astropy.io import fits
import numpy as np
import s3fs
from matplotlib import pyplot as plt
import pyarrow.fs
import pyarrow.parquet as pq
import hpgeom
import json
from astroquery.ipac.irsa import Irsa
```

## 1. Explore the OpenUniverse2024 Data Directories on S3

The OpenUniverse2024 data live in a public AWS S3 bucket and can be accessed anonymously using `s3fs`. This section shows how to establish that connection, navigate the directory tree, and inspect the contents of a FITS image file.

In the path below, `simple_model` refers to the simulated images with noise and realistic instrument effects, as opposed to `truth` images which are noise-free. The `full` simulation covers the complete survey footprint; a smaller `preview` subset is also available. See the [OpenUniverse2024 paper](https://arxiv.org/abs/2501.05632) for details on the differences.

```{code-cell} ipython3
# Create an anonymous (public read-only) connection to the NASA IRSA S3 bucket.
s3 = s3fs.S3FileSystem(anon=True)

# Top-level path components
BUCKET_NAME = "nasa-irsa-simulations"
OU_PREFIX = "openuniverse2024"
ROMAN_TDS_PREFIX = "roman/full/RomanTDS/images/simple_model"
CATALOG_NAME = "roman_rubin_cats_v1.1.2_faint"

# Pick one band and pointing to explore
BAND = "J129"
POINTING = "10190"

image_directory = f"{BUCKET_NAME}/{OU_PREFIX}/{ROMAN_TDS_PREFIX}/{BAND}/{POINTING}"

# List files in this directory
s3.ls(image_directory)
```

```{code-cell} ipython3
# How many files are in this directory?
files = [f"s3://{f}" for f in s3.ls(image_directory)]
print(f"Found {len(files)} files")

# Open one FITS file and inspect its extensions
fname = files[0]
with fits.open(fname, use_fsspec=True, fsspec_kwargs={"anon": True}, memmap=False) as hdul:
    print(f"File: {fname}")
    print(f"Number of extensions: {len(hdul)}\n")
    hdul.info()
```

Each Roman TDS FITS file contains four extensions: a primary header with no data, followed by three 4088×4088 pixel planes — `SCI` (science image), `ERR` (per-pixel uncertainty), and `DQ` (data quality mask).

+++

Let's display a gallery of example images to get a sense of the data.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def show_gallery(files, max_images=9):
    """
    Display a gallery of FITS images.

    Parameters
    ----------
    files : list of str
        List of S3 URIs to FITS files.
    max_images : int, optional
        Maximum number of images to display (default: 9).
    """
    n_images = min(len(files), max_images)
    ncols = n_images if n_images < 4 else 3
    nrows = (n_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, f in enumerate(files[:n_images]):
        with fits.open(f, fsspec_kwargs={"anon": True}, memmap=False) as hdul:
            data = hdul[1].data
            vmin, vmax = np.nanpercentile(data, [5, 99])
            axes[i].imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
            axes[i].set_title(f.split("/")[-1], fontsize=8)
            axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
# Display up to 6 images from the selected directory.
show_gallery(files, max_images=6)
```

## 2. Access the Parquet Catalogs

The OpenUniverse2024 catalogs are stored as Parquet files, partitioned by HEALPix sky region (nside=32, RING ordering). Each region has three file types:

1. `snana_{region}.parquet` — one row per simulated transient event (supernovae, TDEs, etc.), with event type (`model_name`) and host galaxy ID (`host_id`)
2. `galaxy_{region}.parquet` — host galaxy positions and physical properties
3. `galaxy_flux_{region}.parquet` — multi-band Roman and Rubin photometry for each galaxy

The `region` number in each filename is the HEALPix pixel index. We can convert sky coordinates to a region index using `hpgeom`.

```{code-cell} ipython3
# The Roman Time-Domain Survey is centered near the LSST ELAIS-S1 Deep Drilling Field.
ra = 9.45
dec = -44.02

# Convert sky coordinates to a HEALPix region index (nside=32, RING ordering)
nside = 32
region = hpgeom.angle_to_pixel(nside, ra, dec, lonlat=True, nest=False)
print(f"HEALPix region for RA={ra}, Dec={dec}: {region}")
```

```{code-cell} ipython3
# Build the S3 paths for this region's catalog files
catalog_prefix = f"{BUCKET_NAME}/{OU_PREFIX}/roman/full/{CATALOG_NAME}"

snana_path    = f"{catalog_prefix}/snana_{region}.parquet"
galaxy_path   = f"{catalog_prefix}/galaxy_{region}.parquet"
gal_flux_path = f"{catalog_prefix}/galaxy_flux_{region}.parquet"

print("SNANA file:       ", snana_path)
print("Galaxy info file: ", galaxy_path)
print("Galaxy flux file: ", gal_flux_path)
```

### 2.1 Inspect the SNANA Transient Catalog

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def inspect_parquet_columns(s3_path, max_rows=0):
    """
    Read a Parquet file from S3 and print its structure.

    Parameters
    ----------
    s3_path : str
        S3 path to the Parquet file (without the s3:// prefix).
    max_rows : int, optional
        If > 0, also print the first `max_rows` rows. Default is 0.

    Returns
    -------
    pandas.DataFrame
        The full DataFrame loaded from the file.
    """
    fs = pyarrow.fs.S3FileSystem(anonymous=True)
    df = pq.read_table(s3_path, filesystem=fs).to_pandas()

    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nColumn names:")
    for c in df.columns:
        print("  ", c)

    if max_rows > 0:
        print(f"\nFirst {max_rows} rows:")
        print(df.head(max_rows))

    return df
```

```{code-cell} ipython3
df_snana = inspect_parquet_columns(snana_path)
```

```{code-cell} ipython3
# What transient model types are included in this region?
df_snana["model_name"].unique()
```

### 2.2 Inspect the Galaxy Info Catalog

```{code-cell} ipython3
df_galaxy = inspect_parquet_columns(galaxy_path)
```

### 2.3 Inspect the Galaxy Flux Catalog

```{code-cell} ipython3
df_galaxy_flux = inspect_parquet_columns(gal_flux_path)
```

### 2.4 Join Transient Events to Their Host Galaxies

A common operation is to take a transient from the SNANA file and retrieve its host galaxy's sky position from the galaxy info file, using the `host_id` / `galaxy_id` key.

```{code-cell} ipython3
fs = pyarrow.fs.S3FileSystem(anonymous=True)

# Pick one transient — here we grab the first row as an example
example_transient = df_snana.iloc[0]
print("Example transient:")
print(example_transient[["model_name", "host_id", "start_mjd", "end_mjd"]])

# Look up its host galaxy
host = pq.read_table(
    galaxy_path,
    filesystem=fs,
    filters=[("galaxy_id", "==", example_transient["host_id"])]
).to_pandas()

print("\nHost galaxy info:")
print(host)
```

## 3. Query Images by Sky Position Using SIA

Given a sky position (e.g., the host galaxy coordinates from Section 2), we can search for all Roman or Rubin images that cover that position using the IRSA Simple Image Access (SIA) service via `astroquery`.

```{code-cell} ipython3
from astropy.coordinates import SkyCoord
from astropy import units as u

# Point the astroquery IRSA client to the simulated VO endpoints.
# Must be connected to the IPAC VPN or local network to access these endpoints.
# TODO: replace irsadev with irsa when the simulated SIA is deployed to production.
Irsa.sia_url = "https://irsadev.ipac.caltech.edu/simulated/SIA"
Irsa.tap_url = "https://irsadev.ipac.caltech.edu/simulated/TAP"

# List all available simulated image collections
Irsa.list_collections(servicetype='SIA')
```

```{code-cell} ipython3
# Collection names for OpenUniverse2024
OU_ROMAN_SIA_COLLECTION = 'simulated_roman_openuniverse2024'
OU_RUBIN_SIA_COLLECTION = 'simulated_rubin_openuniverse2024'
```

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
def get_s3_fpath(cloud_access):
    """Extract the S3 URI from the cloud_access JSON string in an SIA result."""
    cloud_info = json.loads(cloud_access)
    bucket_name = cloud_info['aws']['bucket_name']
    key = cloud_info['aws']['key']
    return f's3://{bucket_name}/{key}'
```

```{code-cell} ipython3
# Use the host galaxy position from Section 2 (or set any RA/Dec you want to query).
host_ra  = float(host.iloc[0]["ra"])
host_dec = float(host.iloc[0]["dec"])
search_radius = 1 * u.arcsec  # small radius: we just need images that contain this point

coords = SkyCoord(host_ra, host_dec, unit='deg')

# Query Roman TDS images in the J129 band
sia_results = Irsa.query_sia(pos=(coords, search_radius.to(u.deg)),
                             collection=OU_ROMAN_SIA_COLLECTION)

# Keep only J129 simple_model images and attach the S3 URI
bandname = "J129"
roman_images = sia_results[
    ['TDS_simple_model' in r['obs_id'] and bandname in r['energy_bandpassname']
     for r in sia_results]
]
roman_images['s3_uri'] = [get_s3_fpath(r['cloud_access']) for r in roman_images]

print(f"Found {len(roman_images)} Roman {bandname} images at RA={host_ra:.4f}, Dec={host_dec:.4f}")
roman_images['obs_id', 't_min', 't_max', 's3_uri']
```

```{code-cell} ipython3
# The same search works for Rubin images — just swap the collection name and band filter.
rubin_band = "r"
rubin_results = Irsa.query_sia(pos=(coords, search_radius.to(u.deg)),
                               collection=OU_RUBIN_SIA_COLLECTION)

rubin_images = rubin_results[
    [rubin_band in r['energy_bandpassname'] for r in rubin_results]
]
rubin_images['s3_uri'] = [get_s3_fpath(r['cloud_access']) for r in rubin_images]

print(f"Found {len(rubin_images)} Rubin {rubin_band}-band images at RA={host_ra:.4f}, Dec={host_dec:.4f}")
rubin_images['obs_id', 't_min', 't_max', 's3_uri']
```

You now have S3 URIs for all Roman and Rubin images covering your target position. To open any of these images, pass the URI to `astropy.io.fits.open` with `fsspec_kwargs={"anon": True}` as shown in Section 1.

## Acknowledgements

- [IPAC-IRSA](https://irsa.ipac.caltech.edu/)
- This work made use of Astropy:\footnote{http://www.astropy.org} a community-developed core Python package and an ecosystem of tools and resources for astronomy.

## About this notebook

**Authors:** IRSA Data Science Team, including Jessica Krick, Jaladh Singhal, Troy Raen, Brigitta Sipőcz,
Andreas Faisst, Vandana Desai

**Updated:** 2026-04-16

**Contact:** [IRSA Helpdesk](https://irsa.ipac.caltech.edu/docs/help_desk.html) with questions
or problems.

**Runtime:** As of the date above, this notebook takes about 60s to run to completion on
a machine with 8GB RAM and 4 CPU.

**AI Acknowledgement:**

This tutorial was developed with the assistance of AI tools

**References:**

- [Robitaille et al., 2013](https://www.aanda.org/articles/aa/full_html/2013/10/aa22068-13/aa22068-13.html)

- [Astropy Collaboration et al., 2018](https://arxiv.org/abs/1801.02634)

- [Astropy Collaboration et al., 2022](https://arxiv.org/abs/2206.14220)

- [OpenUniverse et al., 2025](https://arxiv.org/abs/2501.05632)
