from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    PLOT_DPI,
    PLOT_FIGSIZE,
    PLOT_TICK_STEP_DIVISOR,
    TEST_END,
    TEST_FREQ,
    TEST_START,
    VALID_FAMILIES,
)


def sanitize_filename(name: str) -> str:
    return re.sub(r'[/\\:*?"<>| ]', "_", name)


def get_client_ids() -> list[str]:
    return [d for d in os.listdir(DATA_DIR) if d.isdigit()]


def load_csv_safe(path: str, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        if parse_dates:
            for col in parse_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        return df
    except Exception:
        return None


def get_real_values(
    client_data: pd.DataFrame, famille: str
) -> pd.DataFrame:
    fd = client_data[["date", famille]].rename(columns={famille: "quantity"}).copy()
    fd_year = fd[fd["date"].dt.year == 2024]
    test_dates = pd.date_range(start=TEST_START, end=TEST_END, freq=TEST_FREQ)

    rows: list[dict] = []
    for td in test_dates:
        week = fd_year[
            (fd_year["date"] >= td) & (fd_year["date"] <= td + pd.Timedelta(days=6))
        ]
        rows.append({
            "date": td.strftime("%Y-W%W"),
            "real": week["quantity"].iloc[0] if len(week) > 0 else 0,
        })
    return pd.DataFrame(rows)


def calculate_rolling_predictions(
    client_data: pd.DataFrame,
    famille: str,
    coefficients: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    fd = client_data[["date", famille]].rename(columns={famille: "quantity"}).copy()
    fd = fd[fd["date"] < TEST_START]
    test_dates = pd.date_range(start=TEST_START, end=TEST_END, freq=TEST_FREQ)

    rows: list[dict] = []
    for td in test_dates:
        match = coefficients[
            (coefficients["date"] == td) & (coefficients["famille"] == famille)
        ]
        coeff = match["coefficient"].iloc[0] if len(match) > 0 else 1.0
        hist = fd[fd["date"] < td]

        if len(hist) >= 52:
            mean = hist["quantity"].tail(52).mean()
        elif len(hist) > 0:
            mean = hist["quantity"].mean()
        else:
            mean = 0

        rows.append({"date": td, "prediction": max(0, coeff * mean), "model": model_name})
    return pd.DataFrame(rows)


def plot_frequent_comparison(
    results: dict,
    client_id: str,
    famille: str,
    available_models: list[str],
) -> str:
    fig, ax = plt.subplots(1, 1, figsize=PLOT_FIGSIZE)
    x = range(len(results["dates"]))

    ax.plot(x, results["real"], "o-", label="Actual", linewidth=3, markersize=8, color="black")

    model_styles = {
        "model1": ("s-", "Predictive Hurdle (Global)", "red"),
        "model2": ("^-", "Direct Hurdle (Client)", "blue"),
        "model3": ("d-", "Historical Coefficient", "green"),
    }

    for m in available_models:
        key = f"{m}_pred"
        if key in results:
            marker, label, color = model_styles[m]
            ax.plot(x, results[key], marker, label=label, linewidth=2, color=color, alpha=0.8)

    tag = " + ".join(m.upper() for m in available_models)
    ax.set_title(f"Comparison ({tag}) — Client {client_id} — {famille}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Weeks 2024", fontsize=12)
    ax.set_ylabel("Quantity", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    step = max(1, len(x) // PLOT_TICK_STEP_DIVISOR)
    ax.set_xticks(list(x)[::step])
    ax.set_xticklabels(
        [results["dates"][i] for i in range(0, len(results["dates"]), step)], rotation=45
    )

    plt.tight_layout()
    out_dir = f"data/{client_id}/frequente_prediction"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/comparison_{client_id}_{sanitize_filename(famille)}.png"
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    return path


def plot_rare_comparison(
    results: dict, client_id: str, famille: str
) -> str:
    fig, ax = plt.subplots(1, 1, figsize=PLOT_FIGSIZE)
    x = range(len(results["dates"]))

    ax.plot(x, results["real"], "o-", label="Actual", linewidth=3, markersize=8, color="black")

    if results.get("same_week_pred"):
        ax.plot(x, results["same_week_pred"], "s-", label="Same Week Last Year",
                linewidth=2, color="orange", alpha=0.8)
    if results.get("croston_pred"):
        ax.plot(x, results["croston_pred"], "^-", label="Croston",
                linewidth=2, color="purple", alpha=0.8)

    ax.set_title(f"Rare Families — Client {client_id} — {famille}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Weeks 2024", fontsize=12)
    ax.set_ylabel("Quantity", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    step = max(1, len(x) // PLOT_TICK_STEP_DIVISOR)
    ax.set_xticks(list(x)[::step])
    ax.set_xticklabels(
        [results["dates"][i] for i in range(0, len(results["dates"]), step)], rotation=45
    )

    plt.tight_layout()
    out_dir = f"data/{client_id}/rares_prediction"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/rare_comparison_{client_id}_{sanitize_filename(famille)}.png"
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    return path
