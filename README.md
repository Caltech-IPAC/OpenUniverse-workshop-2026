# OpenUniverse 2024 Demo Tutorials

The OpenUniverse2024 dataset is a suite of large-scale cosmological simulations designed to support joint survey planning between the Nancy Grace Roman Space Telescope and the Vera C. Rubin Observatory (LSST). Covering roughly 70 square degrees of sky with matched optical and infrared imaging, it provides realistic synthetic catalogs and images incorporating detailed extragalactic modeling, transient populations, and instrument effects.

This repository contains three main tutorial notebooks using the OpenUniverse2024 dataset — from data access and visualization to photometric analysis and spectral energy distribution (SED) fitting.

## 1. Quickstart Tutorial (coming soon)

The Quickstart notebook will provide a lightweight introduction to the OpenUniverse2024 dataset and its data structure. It will guide users through connecting to the public S3 bucket, exploring catalog files, and visualizing a subset of the simulated images.

## 2. SED_fit Tutorial — Fitting Galaxy SEDs with Prospector

The SED_fit tutorial demonstrates how to build a full science workflow: starting from OpenUniverse2024 photometric catalogs, constructing spectral energy distributions (SEDs), and fitting them using the Prospector Bayesian SED fitting code. This example focuses on supernova host galaxies, comparing stellar populations between Type Ia and core-collapse supernovae.

### Goals:

- Access multiband photometry from OpenUniverse2024 (Roman + Rubin)
- Convert simulated fluxes to physical units and plot broadband SEDs
- Perform SED fitting using Prospector and FSPS
- Compare host galaxy properties across supernova types
- Visualize results as SED plots and histograms of fitted stellar masses

## 3. GW_host Tutorial — Identifying Gravitational-Wave Host Galaxies

The GW_host tutorial explores how to identify candidate host galaxies for gravitational-wave events within the OpenUniverse2024 simulations. It introduces an end-to-end workflow that includes accessing sky-localization maps, performing cone searches in simulated catalogs, retrieving corresponding images, and performing photometric measurements.

### Goals:

- Parse gravitational-wave alert information
- Perform cone searches within OpenUniverse2024 Roman/Rubin catalogs
- Retrieve and inspect overlapping simulated images from S3
- Perform aperture photometry and build light curves for host candidates
- Visualize cutouts and light curves to identify potential GW hosts

## Citation

If you use these tutorials or the OpenUniverse2024 dataset in your work, please follow these [instructions](https://registry.opendata.aws/openuniverse2024/) under "How to Cite"


## Contributing

When creating a new notebook, please start with the IRSA notebooks template and follow the formatting guidelines described at [IRSA Python Tutorial Notebooks](https://confluence.ipac.caltech.edu/spaces/IRSA/pages/749091763/Python+Tutorial+Notebooks).

## Contact

Contact [IRSA Help Desk](https://irsa.ipac.caltech.edu/docs/help_desk.html) with questions.
