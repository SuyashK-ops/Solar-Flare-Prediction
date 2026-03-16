# Solar Flare Prediction Using Solar Active Region Data

## Summary

This project studies whether physical properties of solar active regions can be used to predict whether a solar flare will occur in the next 24 hours. The motivation is that solar flares affect space weather, which matters for satellites, communications, navigation, and space operations. The work is organized in phases: download NOAA flare-event reports and SHARP magnetic data, clean and align the datasets by active region and time, create forecasting labels, explore the physical trends with plots and statistics, improve the forecasting setup with time-aware features and chronological splits, and finally apply machine-learning models to test whether these solar magnetic features contain usable predictive signal.

This project is a phased scientific data-analysis workflow for solar flare prediction, starting from data acquisition and exploratory analysis and ending with a final machine-learning benchmark.

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
  - exploratory analysis notebook with outputs
- `notebooks/solar_flare_prediction_stage2_modeling.ipynb`
  - baseline machine-learning notebook with saved results
- `notebooks/solar_flare_prediction_stage3_forecasting_design.ipynb`
  - forecasting-design notebook with time-aware features and chronological splitting
- `notebooks/solar_flare_prediction_stage4_final_ml.ipynb`
  - final machine-learning notebook with tuned models and held-out chronological evaluation

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

## Baseline Modeling Stage

The second notebook adds a first machine-learning step using:

- logistic regression
- decision tree
- random forest

The goal here is to establish whether the SHARP magnetic parameters contain predictive signal for `flare_next_24h`, not to claim a final operational forecast model.

## Forecasting Design Stage

The third notebook strengthens the scientific setup before any serious ML stage by adding:

- temporal features such as snapshot-to-snapshot change
- rolling means and rolling standard deviations within each active region
- a stronger-flare label for `M` and `X` class events
- a chronological train/test split instead of a random split

This phase is meant to improve the realism of the prediction problem itself.

## Final ML Stage

The final notebook applies machine learning to the forecasting dataset by:

- combining the original SHARP features with temporal forecasting features
- training several candidate classifiers
- tuning models using only the earlier training period
- evaluating the final models on a later held-out period
- interpreting the best model with feature-importance analysis

This is the final benchmark stage of the project.

## Next Stage

Natural extensions after this baseline are:

- expand the dataset to a longer historical interval
- use longer-term temporal validation across broader date ranges
- improve physically motivated feature engineering
- compare event definitions and forecast windows
- test stronger models on a larger dataset
- study feature stability across solar cycles
