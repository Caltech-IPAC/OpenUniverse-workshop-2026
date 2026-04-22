# OpenUniverse 2024 Demo Tutorials

The OpenUniverse2024 dataset is a suite of large-scale cosmological simulations designed to support joint survey planning between the Nancy Grace Roman Space Telescope and the Vera C. Rubin Observatory (LSST). Covering roughly 70 square degrees of sky with matched optical and infrared imaging, it provides realistic synthetic catalogs and images incorporating detailed extragalactic modeling, transient populations, and instrument effects.

## Tutorials

This repository contains three main tutorial notebooks using the OpenUniverse2024 dataset — from data access and visualization to photometric analysis and spectral energy distribution (SED) fitting.

### 1. Quickstart Tutorial — Accessing OpenUniverse2024 Data

The Quickstart tutorial is a focused introduction to the three main data access patterns in OpenUniverse2024: browsing the S3 directory structure for Roman and Rubin FITS images, reading the parquet catalogs (transient, galaxy, and galaxy-flux tables), and querying which images cover a given sky position using the IRSA Simple Image Access (SIA) service.

#### Goals:

- Connect to the public S3 bucket and navigate the Roman image directory tree
- Inspect the structure of a Roman FITS file and display a gallery of images
- Load SNANA, galaxy, and galaxy-flux parquet catalogs using HEALPix region indexing
- Query Roman and Rubin images overlapping a sky position via astroquery and SIA

### 2. SED_fit Tutorial — Fitting Galaxy SEDs with Prospector

The SED_fit tutorial demonstrates how to build a full science workflow: starting from OpenUniverse2024 photometric catalogs, constructing spectral energy distributions (SEDs), and fitting them using the Prospector Bayesian SED fitting code. This example focuses on supernova host galaxies, comparing stellar populations between Type Ia and core-collapse supernovae.

#### Goals:

- Access multiband photometry from OpenUniverse2024 (Roman + Rubin)
- Convert simulated fluxes to physical units and plot broadband SEDs
- Perform SED fitting using Prospector and FSPS
- Compare host galaxy properties across supernova types
- Visualize results as SED plots and histograms of fitted stellar masses

### 3. TDE Light Curve Tutorial — Building a Multi-Epoch Light Curve

The TDE Light Curve tutorial demonstrates an end-to-end science workflow for transient astronomy: locating a simulated Tidal Disruption Event (TDE) in the OpenUniverse2024 transient catalog, identifying its host galaxy, retrieving Roman images via the IRSA SIA service, and performing aperture photometry to construct a multi-epoch light curve.

#### Goals:

- Select a TDE from the SNANA parquet catalog and locate its host galaxy
- Query Roman TDS images covering the host position using astroquery and SIA
- Perform aperture photometry on individual Roman images
- Build and visualize a multi-epoch infrared light curve
- Display full images and cutouts centered on the TDE host

## Citation

If you use these tutorials or the OpenUniverse2024 dataset in your work, please follow these [instructions](https://registry.opendata.aws/openuniverse2024/) under "How to Cite".


## Contributing

When creating a new notebook, please start with the IRSA notebooks template and follow the formatting guidelines described at [IRSA Python Tutorial Notebooks](https://confluence.ipac.caltech.edu/spaces/IRSA/pages/749091763/Python+Tutorial+Notebooks).

## Contact

Contact [IRSA Help Desk](https://irsa.ipac.caltech.edu/docs/help_desk.html) with questions.
