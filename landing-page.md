# Hands-On Science with OpenUniverse2024 Roman and Rubin Simulations

[OpenUniverse2024](https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/overview.html) is a collection of large-scale simulated observations designed to resemble expected data products from the Nancy Grace Roman Space Telescope and the Vera C. Rubin Observatory.
The dataset includes simulated wide-field imaging, time-domain observations, and mock source catalogs covering roughly 70 square degrees of sky.

## What Is OpenUniverse2024?

Video:

Slides:

## Introduction to Fornax

Video:

Slides:

## Accessing OpenUniverse2024 Data

Video:

Slides:

## Visualizing OpenUniverse2024 Data

Video:

Slides:

## Tutorials

Workshop participants will explore the following tutorials.

If you use these tutorials or the OpenUniverse2024 dataset in your work, please follow these [instructions](https://registry.opendata.aws/openuniverse2024/) under "How to Cite".

### Quickstart

[OpenUniverse2024 Quickstart](https://caltech-ipac.github.io/irsa-tutorials/openuniverse2024-quickstart): Access OpenUniverse2024 Roman and Rubin images and catalogs.

The Quickstart tutorial is a focused introduction to the three main data access patterns in OpenUniverse2024: browsing the S3 directory structure for Roman and Rubin FITS images, reading the parquet catalogs (transient, galaxy, and galaxy-flux tables), and querying which images cover a given sky position using the IRSA Simple Image Access (SIA) service.

Goals:

- Connect to the public S3 bucket and navigate the Roman image directory tree
- Inspect the structure of a Roman FITS file and display a gallery of images
- Load SNANA, galaxy, and galaxy-flux parquet catalogs using HEALPix region indexing
- Query Roman and Rubin images overlapping a sky position via astroquery and SIA

### Visualization

[OpenUniverse2024 Visualization](https://caltech-ipac.github.io/irsa-tutorials/openuniverse2024preview-firefly): Use Firefly to get an overview of survey structure and visualize content.

### TDE Light Curve

[OpenUniverse2024 TDE Light Curve](https://caltech-ipac.github.io/irsa-tutorials/openuniverse2024-tde-light-curve): Locate a simulated TDE, retrieve Roman images, and build a multi-epoch light curve.

The TDE Light Curve tutorial demonstrates an end-to-end science workflow for transient astronomy: locating a simulated Tidal Disruption Event (TDE) in the OpenUniverse2024 transient catalog, identifying its host galaxy, retrieving Roman images via the IRSA SIA service, and performing aperture photometry to construct a multi-epoch light curve.

Goals:

- Select a TDE from the SNANA parquet catalog and locate its host galaxy
- Query Roman TDS images covering the host position using astroquery and SIA
- Perform aperture photometry on individual Roman images
- Build and visualize a multi-epoch infrared light curve
- Display full images and cutouts centered on the TDE host

### SED Fitting

[OpenUniverse2024 SED Fitting](https://caltech-ipac.github.io/irsa-tutorials/openuniverse2024-sed-fit): Fit spectral energy distributions for supernova host galaxies using Prospector.

The SED_fit tutorial demonstrates how to build a full science workflow: starting from OpenUniverse2024 photometric catalogs, constructing spectral energy distributions (SEDs), and fitting them using the Prospector Bayesian SED fitting code. This example focuses on supernova host galaxies, comparing stellar populations between Type Ia and core-collapse supernovae.

Goals:

- Access multiband photometry from OpenUniverse2024 (Roman + Rubin)
- Convert simulated fluxes to physical units and plot broadband SEDs
- Perform SED fitting using Prospector and FSPS
- Compare host galaxy properties across supernova types
- Visualize results as SED plots and histograms of fitted stellar masses

### Roman Coadds

[OpenUniverse2024 Roman Coadds](https://caltech-ipac.github.io/irsa-tutorials/openuniverse2024-roman-simulated-wideareasurvey): Access OpenUniverse2024 wide-area simulated survey data.

### Time Domain

[OpenUniverse2024 Time Domain](https://caltech-ipac.github.io/irsa-tutorials/openuniverse2024-roman-simulated-timedomainsurvey): Access and analyze the simulated time-domain OpenUniverse2024 survey.
