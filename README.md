# Solar Flare Prediction Using Solar Active Region Data

## Summary

This project investigates whether physical properties of solar active regions can be used to predict if a solar flare will occur within the next 24 hours. The motivation is space weather: solar flares can disrupt satellites, communications, navigation systems, and human activity in space. The workflow is built in phases, beginning with NOAA flare-event reports and SHARP magnetic-field data, then cleaning and aligning the datasets, creating forecasting labels, exploring physical trends statistically, improving the forecasting design with time-aware features, and finally applying machine-learning models to test whether these solar magnetic signals contain useful predictive information.

## Why It Matters

Solar flare prediction is important for:

- satellite operations
- space mission planning
- radio communication reliability
- GPS and navigation resilience
- broader space-weather forecasting

## Project Highlights

- Uses real NOAA SWPC flare-event reports and JSOC SHARP active-region parameters
- Builds a full pipeline from raw data download to final ML benchmarking
- Handles real solar-data alignment issues, including region-ID normalization across datasets
- Uses physically meaningful magnetic features rather than abstract synthetic data
- Improves the project in clear stages instead of jumping directly to ML

## Project Phases

### Phase 1: Data and Exploration

- Download flare-event reports and SHARP parameters
- Clean timestamps, missing values, and active-region IDs
- Create the binary target `flare_next_24h`
- Perform exploratory data analysis and statistics

Main notebook:
- `notebooks/solar_flare_prediction_stage1.ipynb`

### Phase 2: Baseline Prediction Setup

- Organize the labeled dataset into a prediction-ready feature table
- Compare simple baseline models
- Produce first performance comparisons

Main notebook:
- `notebooks/solar_flare_prediction_stage2_modeling.ipynb`

### Phase 3: Forecasting Design Improvements

- Add time-aware features such as change, rolling mean, and rolling variability
- Add a stronger-flare label for `M` and `X` class events
- Use chronological splitting instead of a random split

Main notebook:
- `notebooks/solar_flare_prediction_stage3_forecasting_design.ipynb`

### Phase 4: Final ML Benchmark

- Train and tune multiple models on the forecasting dataset
- Evaluate models on a later held-out time period
- Compare final performance with precision-recall and ROC analysis
- Interpret the strongest model using feature-importance analysis

Main notebook:
- `notebooks/solar_flare_prediction_stage4_final_ml.ipynb`

## Data Sources

- NOAA SWPC archived daily solar event reports
  - flare timing
  - flare class
  - NOAA active region number
- JSOC SHARP parameter data
  - magnetic flux
  - active-region area
  - current helicity and related magnetic-complexity measures

## Main Physical Features

Examples of features used in the project:

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

## Labels

Primary forecasting label:

```text
flare_next_24h = 1
if a flare occurs in the same active region within the next 24 hours
otherwise 0
```

Phase 3 also adds:

```text
strong_flare_next_24h = 1
if an M-class or X-class flare occurs within the next 24 hours
otherwise 0
```

## Current Final-Phase Result

On the current downloaded sample, the best final model is a random forest using time-aware magnetic features and chronological evaluation.

- Best model: `Random Forest`
- Best parameters: `max_depth=4`, `n_estimators=200`
- Average precision: about `0.700`
- ROC AUC: about `0.811`
- F1 score: about `0.581`

These results should be interpreted as a compact scientific benchmark, not as a finished operational flare-warning system.

## Project Structure

```text
solar_flare_prediction/
    data/
    figures/
    notebooks/
    src/
    requirements.txt
    README.md
```

## Key Files

- `src/data_download.py`
  - downloads NOAA and SHARP data
- `src/preprocessing.py`
  - loads, cleans, aligns, and labels the datasets
- `src/exploration.py`
  - exploratory plots and statistics
- `src/modeling.py`
  - baseline prediction utilities
- `src/forecasting.py`
  - forecasting-design improvements and time-aware features
- `src/final_ml.py`
  - final ML training, tuning, evaluation, and interpretation

## Practical Data Notes

The code handles two real solar-data issues:

- SHARP may store NOAA active-region identifiers in five-digit form such as `14366`, while NOAA flare reports list the same region as `4366`
- SHARP rows can contain multiple NOAA region IDs, so those rows are expanded before flare matching

## Visual and Statistical Outputs

The project generates:

- flare-class histograms
- magnetic-feature distributions
- correlation heatmaps
- magnetic-flux vs flare-occurrence scatter plots
- model-comparison plots
- precision-recall and ROC curves
- feature-importance plots

Saved figures are written to `figures/`.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

Open the notebooks in VS Code or Jupyter and run them in order:

1. `notebooks/solar_flare_prediction_stage1.ipynb`
2. `notebooks/solar_flare_prediction_stage2_modeling.ipynb`
3. `notebooks/solar_flare_prediction_stage3_forecasting_design.ipynb`
4. `notebooks/solar_flare_prediction_stage4_final_ml.ipynb`

## What This Project Demonstrates

This project demonstrates:

- scientific data cleaning and integration
- physics-motivated feature analysis
- forecasting-aware dataset design
- model evaluation under chronological testing
- clear staged project development from raw data to final benchmark

## Future Improvements

Natural next extensions would be:

- use a much longer historical dataset
- evaluate across multiple solar-cycle periods
- compare different forecast windows
- focus specifically on stronger flares
- test stronger models on larger datasets
