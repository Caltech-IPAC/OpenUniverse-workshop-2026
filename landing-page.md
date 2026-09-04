# Hands-On Science with OpenUniverse2024 Roman and Rubin Simulations

[OpenUniverse2024](https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/overview.html) is a collection of large-scale simulated observations designed to resemble the expected data products of the Nancy Grace Roman Space Telescope and the Vera C. Rubin Observatory.
Both observatories are simulated over the same roughly 70 square degree patch of sky, so the same objects appear in Roman's high-resolution near-infrared imaging and in Rubin's wide optical *ugrizy* imaging, and the two can be analyzed together.
The dataset includes wide-area imaging, a time-domain survey that revisits a smaller footprint many times, and mock source catalogs.
For Roman there are truth catalogs of the input object properties, noiseless truth images, calibrated single-epoch images, and coadds.
For Rubin there are raw pixel data, calibrated exposures, coadds, and photometric catalogs.


All data is public, hosted on Amazon S3 in `s3://nasa-irsa-simulations/openuniverse2024/`, and requires no credentials. Catalogs are stored in Parquet format.  Images are multi-extension FITS.


The below materials are intended as a free workshop with a hands-on introduction to working with the OU dataset.
Rather than surveying the simulations from a distance, we work through Python Jupyter notebooks that access the data directly in the cloud and carry it through to a scientific result — locating a transient, building a light curve, fitting a spectral energy distribution.
The emphasis throughout is on combining complementary survey data: what you can measure by using Roman and Rubin together that you cannot get from either one alone, and how to structure that work as a scalable cloud workflow rather than a local download.


All of the material runs in the Fornax Science Console, so participants can follow along and keep experimenting afterward with no local setup.

## Before You Start

The workshop runs entirely in the [Fornax Science Console](https://docs.fornax.sciencecloud.nasa.gov/), NASA's cloud-based science platform, so there is nothing to install on your own machine.
The [Fornax Quick Start Guide](https://docs.fornax.sciencecloud.nasa.gov/quick-start/) walks through the five steps of getting an account, logging in, starting a server session, opening a notebook, and shutting the server down again.

- **Registered workshop participants** already have a Fornax account and can skip step 1 and start at step 2, "Log In".
- **Everyone else** should begin at step 1, "Get an Account", and register at [signup.fornax.sciencecloud.nasa.gov](https://signup.fornax.sciencecloud.nasa.gov). Requests made with a `nasa.gov` address are approved immediately; other institutional addresses are reviewed by Fornax staff and can take up to two business days, so allow time before you plan to work through the material.

The tutorials assume some familiarity with Python and Jupyter notebooks, but not with Roman, Rubin, or cloud data access.

## Agenda

Slides and recordings will be posted here after the workshop.

- **What Is OpenUniverse2024?** (15 min)
- **Introduction to Fornax** (15 min)
- **Accessing OpenUniverse2024 Data** (20 min)
- **Visualizing OpenUniverse2024 Data** (15 min)
- **Self-paced tutorial exploration** (25 min)

## Tutorials

Workshop participants will explore the following tutorials, all part of the [IRSA simulated data tutorials](https://caltech-ipac.github.io/irsa-tutorials/simulated/) and runnable at any time in Fornax.

### [Quickstart](https://caltech-ipac.github.io/irsa-tutorials/openuniverse2024-quickstart)

Access OpenUniverse2024 Roman and Rubin images and catalogs.

- Connect to the public S3 bucket and navigate the Roman image directory tree
- Inspect the structure of a Roman FITS file and display a gallery of images
- Load SNANA, galaxy, and galaxy-flux parquet catalogs using HEALPix region indexing
- Query Roman and Rubin images overlapping a sky position via astroquery and SIA

### [Firefly Visualization](https://caltech-ipac.github.io/irsa-tutorials/openuniverse2024-firefly-visualization)

Use Firefly to explore the structure of the simulations and visualize their content.

- Access cloud-hosted Roman and Rubin simulated images and find those covering a sky position with SIA
- Launch an interactive Firefly instance inside JupyterLab
- Display a Rubin image and overplot ds9 regions showing where the Roman data fall
- Overplot Parquet truth catalogs and filter them to pick out high-redshift galaxies
- Build three-color images and control stretch, pan, and zoom from Python

### [Roman Time Domain Survey Supernova](https://caltech-ipac.github.io/irsa-tutorials/openuniverse2024-roman-tds-supernova)

Explore the Roman Time Domain Survey observation sequence and animate a simulated SN Ia across epochs.

- Read the Roman TDS Observation Sequence File to see how the survey is laid out in space and time
- Filter the SNANA transient catalog for a SN Ia peaking during an observed window
- Query SIA for every Roman image covering the supernova during its bright phase
- Extract aligned cutouts from 31 epochs, correcting for the varying focal plane orientation
- Animate the cutouts into a GIF showing the supernova brighten and fade

### [TDE Light Curve](https://caltech-ipac.github.io/irsa-tutorials/openuniverse2024-tde-light-curve)

Locate a simulated TDE, retrieve Roman images, and build a multi-epoch light curve.

- Select a TDE from the SNANA parquet catalog and locate its host galaxy
- Query Roman TDS images covering the host position using astroquery and SIA
- Perform aperture photometry on individual Roman images
- Build and visualize a multi-epoch infrared light curve
- Display full images and cutouts centered on the TDE host

### [SED Fitting](https://caltech-ipac.github.io/irsa-tutorials/openuniverse2024-sed-fit)

Fit spectral energy distributions for supernova host galaxies using Prospector.

- Access multiband photometry from OpenUniverse2024 (Roman + Rubin)
- Convert simulated fluxes to physical units and plot broadband SEDs
- Perform SED fitting using Prospector and FSPS
- Compare host galaxy properties across supernova types
- Visualize results as SED plots and histograms of fitted stellar masses

## The Dataset and How to Cite It

- **Documentation:** [OpenUniverse2024 overview at IRSA](https://irsa.ipac.caltech.edu/data/theory/openuniverse2024/overview.html)
- **Data:** [OpenUniverse2024 on the AWS Open Data Registry](https://registry.opendata.aws/openuniverse2024/), in the `us-east-1` buckets `s3://nasa-irsa-simulations/openuniverse2024/roman/` and `s3://nasa-irsa-simulations/openuniverse2024/rubin/`

If you use these tutorials or the OpenUniverse2024 dataset in your work, please follow the ["How to Cite" instructions](https://registry.opendata.aws/openuniverse2024/): cite DOI [10.26131/IRSA569](https://doi.org/10.26131/IRSA569) for the Data Preview or DOI [10.26131/IRSA596](https://doi.org/10.26131/IRSA596) for the Full Release, and follow the OpenUniverse2024 citation guidelines and the IRSA acknowledgement guidelines.

## Questions

Contact the [IRSA Help Desk](https://irsa.ipac.caltech.edu/docs/help_desk.html).
