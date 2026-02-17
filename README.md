# Demand Forecasting System

Weekly demand forecasting for beverage distribution clients, split into frequent (regular orders) and rare (intermittent orders) product families.

## Models

**Frequent families** — three approaches compared per client:
- **Model 1**: Global hurdle XGBoost trained on aggregated family data, outputs weekly coefficients applied to each client's 52-week rolling mean
- **Model 2**: Per-client hurdle model (logistic gate + XGBoost) using temporal, weather, and event features
- **Model 3**: Historical coefficient — averages same-week ratios from prior years, no ML

**Rare families** — two approaches:
- **Same-week reproduction**: copies the value from the same week one year prior
- **Croston**: exponential smoothing on demand sizes and inter-arrival intervals

## Usage

```bash
pip install -r requirements.txt
python main.py              # runs both frequent and rare
python main.py --mode frequent
python main.py --mode rare
```

## Structure

```
├── main.py              # entry point
├── src/
│   ├── config.py        # constants, hyperparameters, paths
│   ├── data.py          # data loading (weather, events, clients)
│   ├── features.py      # feature engineering (global + per-client)
│   ├── model.py         # HurdleModel, CoefficientModel, Croston, RareFamiliesModel
│   ├── comparison.py    # model comparison and plot generation
│   └── utils.py         # plotting, file I/O helpers
├── requirements.txt
└── .gitignore
```

Data is expected under `data/` with per-client subdirectories. See `data/{client_id}/frequentes.csv` and `rares.csv` as inputs. Outputs (predictions, coefficients, plots) are written alongside the inputs.
