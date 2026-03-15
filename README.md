# Solar Flare Prediction Using Solar Active Region Data

This project is the first stage of a scientific data-analysis workflow for solar flare prediction. It focuses on downloading, cleaning, aligning, and exploring solar flare observations together with solar active-region physical parameters before moving to machine learning.

## Project Goal

The objective is to build a clean exploratory pipeline for:

- loading NOAA solar flare event data
- loading SHARP active-region parameters
- matching flare events to active regions
- creating a binary `flare_next_24h` target
- generating visual and statistical summaries

## Data Sources

- NOAA SWPC archived daily solar event reports
  - flare event timing
  - flare class
  - NOAA active region number
- JSOC SHARP parameter data
  - magnetic flux (`USFLUX`)
  - active region area (`AREA_ACR`)
  - current helicity and related magnetic complexity features

## Folder Structure

```text
solar_flare_prediction/
    data/
    figures/
    notebooks/
    src/
    requirements.txt
    README.md
```

## Main Files

- `src/data_download.py`
  - downloads NOAA flare reports
  - downloads SHARP parameter tables
- `src/preprocessing.py`
  - loads datasets
  - cleans timestamps and missing values
  - normalizes and aligns active region IDs
  - creates `flare_next_24h`
- `src/exploration.py`
  - dataset summary
  - plots
  - Pearson correlations
  - mutual information
- `notebooks/solar_flare_prediction_stage1.ipynb`
  - main exploratory notebook with outputs

## Features Used

The exploratory stage currently works with SHARP-style physical parameters such as:

- `USFLUX`
- `AREA_ACR`
- `TOTUSJH`
- `TOTUSJZ`
- `ABSNJZH`
- `SAVNCPP`
- `MEANPOT`
- `R_VALUE`
- `MEANGBT`
- `MEANGBZ`
- `MEANGBH`

## Label Definition

The binary label is defined as:

```text
flare_next_24h = 1
if a flare occurs in the same active region within the next 24 hours
otherwise 0
```

## Visualizations Produced

The pipeline generates:

- histogram of flare classes
- distribution of magnetic field strength
- distribution of active region area
- correlation heatmap of physical parameters
- scatter plot of magnetic flux vs flare occurrence

Saved figures are written to the `figures/` folder.

## Statistical Analysis

The project computes:

- Pearson correlation between numeric features and `flare_next_24h`
- mutual information between physical parameters and `flare_next_24h`

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

The main workflow is in the notebook:

```text
notebooks/solar_flare_prediction_stage1.ipynb
```

Open the notebook in VS Code or Jupyter and run the cells. The notebook can:

- download a compact NOAA + SHARP dataset
- load and clean the data
- create the flare label
- display summary tables
- generate figures

## Notes on Data Alignment

Two practical preprocessing issues are handled in the code:

- SHARP region identifiers can appear in a five-digit format such as `14366`, while NOAA flare reports may list the same region as `4366`
- SHARP rows can contain multiple NOAA region IDs, so these rows are expanded before matching to flare events

## Current Stage

This repository is intentionally focused on exploratory analysis and label creation. The next stage would be to:

- select features
- split training and test sets
- train baseline machine-learning models
- evaluate prediction skill
