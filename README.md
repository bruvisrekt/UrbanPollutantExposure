# Quantitative Assessment of Urban Pollutant Exposure Across Economic Strata

> A data science project investigating the relationship between neighbourhood-level median household income and fine particulate matter (PM2.5) concentrations across urban areas in the United States.

**Author:** Aakash Rout (24BDS1074)
**Institution:** Vellore Institute of Technology, Chennai Campus
**Course:** Foundations of Data Science (FDS)

---

## Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [ML Model](#ml-model)
- [Pipeline Modules](#pipeline-modules)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Known Limitations](#known-limitations)
- [Bibliography](#bibliography)

---

## Overview

Air quality is not distributed equally across urban populations. This project quantitatively examines the extent to which a resident's economic standing — measured by median household income — determines the level of air pollution they are exposed to. Using real-world sensor data from the OpenAQ platform and socio-economic data from the U.S. Census Bureau, the pipeline constructs a unified, ZIP-code-level dataset that serves as the foundation for statistical analysis and predictive modelling.

---

## Research Motivation

A growing body of evidence suggests that low-income neighbourhoods disproportionately bear the burden of urban air pollution, owing to historical patterns in industrial zoning, infrastructure development, and urban planning. The consequences are measurable: higher rates of childhood asthma, cardiovascular disease, and reduced educational attainment in disadvantaged areas.

This project aims to:

- Quantify the income–pollution disparity using open civic data
- Build a reproducible, auditable data pipeline for environmental justice research
- Produce findings that are communicable to non-technical stakeholders such as city planners and public health officials

---

## Project Structure

```
urban-pollution-income/
│
├── data/
│   ├── raw/
│   │   ├── openaq_pm25_2023.json        # Raw API response from OpenAQ
│   │   └── acs_median_income_by_zip.csv # Census ACS Table B19013
│   └── processed/
│       └── unified_pollution_income.csv # Final merged dataset
│
├── notebooks/
│   ├── 01_acquisition_cleaning.ipynb    # Module 1: Data Acquisition & Cleaning
│   ├── 02_integration.ipynb             # Module 2: Data Integration
│   └── 03_modelling.ipynb               # Module 3: XGBoost Modelling (upcoming)
│
├── src/
│   ├── acquisition.py                   # OpenAQ fetch + Census load + cleaning
│   └── integration.py                   # Reverse geocoding + merge + feature engineering
│
├── outputs/
│   └── figures/                         # Visualisations (scatter plots, bar charts)
│
├── docs/
│   └── technical_documentation.docx    # Full project documentation
│
├── requirements.txt
└── README.md
```

---

## Datasets

| # | Dataset | Source | Format | Description |
|---|---------|--------|--------|-------------|
| 1 | OpenAQ v3 Measurements | [api.openaq.org](https://api.openaq.org/v3/) | JSON (API) | PM2.5 readings (µg/m³) from fixed monitoring stations across the US, filtered for calendar year 2023 |
| 2 | ACS 5-Year Estimates — Table B19013 | [data.census.gov](https://data.census.gov/table/ACSDT5Y2023.B19013) | CSV | Median household income by 5-digit ZIP Code Tabulation Area (ZCTA), U.S. Census Bureau |

> **Note:** An OpenAQ API key is required for data acquisition. Register for a free key at [openaq.org](https://openaq.org).

---

## Methodology

The pipeline follows four sequential stages:

1. **Data Acquisition & Cleaning** — Fetch PM2.5 readings from the OpenAQ API and load Census income data. Remove sensor faults (readings < 0 or > 500 µg/m³), standardise ZIP code formatting, and drop null income records.

2. **Data Integration** — Reverse-geocode sensor coordinates (lat/lon) to ZIP codes using Nominatim. Aggregate PM2.5 to annual mean per ZIP code. Perform an inner join with Census income data and engineer an income quintile feature (1 = lowest 20%, 5 = highest 20%).

3. **Statistical Analysis** *(upcoming)* — Compute the Pearson Correlation Coefficient between median income and mean PM2.5. Conduct a one-way ANOVA across income quintiles to test for statistically significant pollution disparities.

4. **Predictive Modelling** *(upcoming)* — Train and evaluate an XGBoost Regressor with 5-fold cross-validated hyperparameter tuning. Extract SHAP values for feature importance interpretation.

---

## ML Model

**Selected Model: XGBoost Regressor**

| Candidate | Reason Considered | Decision |
|-----------|------------------|----------|
| Linear Regression | Simple baseline; interpretable coefficients | Baseline only — assumes linearity, sensitive to outliers |
| Random Forest | Handles non-linearity; robust to missing data | Strong alternative; lacks SHAP efficiency of XGBoost |
| **XGBoost** | **Captures non-linear interactions; native missing value handling; produces SHAP feature importances** | ✅ **Selected** |

**Evaluation Metrics:** Root Mean Squared Error (RMSE), R² coefficient of determination
**Train / Test Split:** 80/20, stratified by income quintile

---

## Pipeline Modules

### Module 1 — Data Acquisition & Cleaning (`src/acquisition.py`)

```python
# Key steps performed:
# 1. Fetch PM2.5 data from OpenAQ v3 API (up to 10,000 records, year 2023)
# 2. Flatten nested JSON fields (coordinates, date)
# 3. Remove outliers: drop records where pm25 < 0 or pm25 > 500 µg/m³
# 4. Load Census ACS CSV; zero-pad ZIP codes to 5 digits
# 5. Drop rows with null or zero median income values
```

### Module 2 — Data Integration (`src/integration.py`)

```python
# Key steps performed:
# 1. Deduplicate sensor locations; reverse-geocode lat/lon → ZIP code (Nominatim)
# 2. Merge ZIP codes back into the air quality DataFrame
# 3. Aggregate PM2.5 to annual mean, count, and std dev per ZIP code
# 4. Inner join aggregated AQ data with Census income data on zip_code
# 5. Engineer income_quintile feature using pandas.qcut (5 equal-population bands)
# 6. Export final dataset to data/processed/unified_pollution_income.csv
```

---

## Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
pandas>=2.1.0
numpy>=1.26.0
requests>=2.31.0
geopy>=2.4.0
xgboost>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.8.0
shap>=0.44.0
jupyter>=1.0.0
```

---

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/urban-pollution-income.git
cd urban-pollution-income
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Open `src/acquisition.py` and replace the placeholder with your OpenAQ API key:

```python
headers = {'X-API-Key': 'YOUR_OPENAQ_API_KEY'}
```

### 4. Run the pipeline

```bash
# Step 1 — Acquire and clean data
python src/acquisition.py

# Step 2 — Integrate datasets
python src/integration.py
```

The final unified dataset will be saved to `data/processed/unified_pollution_income.csv`.

---

## Known Limitations

| Limitation | Description | Planned Resolution |
|------------|-------------|-------------------|
| ZIP vs ZCTA boundary mismatch | Nominatim returns postal ZIP codes which may not align perfectly with Census ZCTAs | Replace Nominatim with Census TIGER/Line Shapefile spatial joins |
| Annual mean aggregation | Averaging PM2.5 over a full year may obscure seasonal spikes from wildfires or winter heating | Introduce seasonal decomposition in a future pipeline version |
| Static sensor coverage | OpenAQ stations are fixed; mobile exposure during commutes is not captured | Flag as a research gap; not addressed in current scope |
| Census data lag | ACS income estimates are updated annually but reflect a 5-year rolling average | Acceptable for this study; noted as a temporal limitation |

---

## Bibliography

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

Brulle, R. J., & Pellow, D. N. (2006). Environmental justice: Human health and environmental inequalities. *Annual Review of Public Health*, 27, 103–124. https://doi.org/10.1146/annurev.publhealth.27.021405.102124

Hajat, A., Hsia, C., & O'Neill, M. S. (2015). Socioeconomic disparities and air pollution exposure: A global review. *Current Environmental Health Reports*, 2(4), 440–450. https://doi.org/10.1007/s40572-015-0069-5

Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

McKinney, W., & the pandas development team. (2023). *pandas: Powerful Python data analysis toolkit* (Version 2.1) [Software]. https://doi.org/10.5281/zenodo.3509134

OpenAQ. (2023). *OpenAQ open air quality data platform* (Version 3) [Data set]. OpenAQ. https://api.openaq.org/v3/

Pearson, K. (1895). Notes on regression and inheritance in the case of two parents. *Proceedings of the Royal Society of London*, 58, 240–242. https://doi.org/10.1098/rspl.1895.0041

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

United States Census Bureau. (2023). *American Community Survey 5-year estimates, Table B19013: Median household income in the past 12 months* [Data set]. U.S. Department of Commerce. https://data.census.gov/table/ACSDT5Y2023.B19013

United States Environmental Protection Agency. (2023). *EJScreen: Environmental justice screening and mapping tool* [Interactive database]. U.S. EPA. https://www.epa.gov/ejscreen

World Health Organization. (2021). *WHO global air quality guidelines: Particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide*. WHO Press. https://doi.org/10.2471/BKL.21.15

---

*Submitted in partial fulfilment of the requirements for the Foundations of Data Science course, Vellore Institute of Technology, Chennai Campus.*
