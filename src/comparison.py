from __future__ import annotations

import pandas as pd

from src.config import DATA_DIR, TEST_END, TEST_FREQ, TEST_START, VALID_FAMILIES
from src.utils import (
    calculate_rolling_predictions,
    get_client_ids,
    get_real_values,
    load_csv_safe,
    plot_frequent_comparison,
    plot_rare_comparison,
)


class FrequentModelComparison:
    def __init__(self) -> None:
        self.model1_coefficients: pd.DataFrame | None = None
        self.model3_cache: dict[str, pd.DataFrame] = {}

    def run(self) -> None:
        self._load_global_coefficients()
        for client_id in get_client_ids():
            self._compare_client(client_id)

    def _load_global_coefficients(self) -> None:
        self.model1_coefficients = load_csv_safe(
            "data/families_w/coefficients.csv", parse_dates=["date"]
        )

    def _compare_client(self, client_id: str) -> None:
        client_data = load_csv_safe(f"data/{client_id}/frequentes.csv", parse_dates=["date"])
        predictions = load_csv_safe(f"data/{client_id}/predictions.csv")
        coefficients = load_csv_safe(f"data/{client_id}/coef_families.csv", parse_dates=["date"])

        if client_data is None:
            return
        if predictions is not None:
            predictions["date"] = predictions["date"].astype(str)

        families = [c for c in client_data.columns if c in VALID_FAMILIES]
        test_dates = pd.date_range(start=TEST_START, end=TEST_END, freq=TEST_FREQ)

        for famille in families:
            real = get_real_values(client_data, famille)
            results: dict = {"dates": [], "real": []}
            for td in test_dates:
                row = real[real["date"] == td.strftime("%Y-W%W")]
                if len(row) > 0:
                    results["dates"].append(td.strftime("%Y-W%W"))
                    results["real"].append(row["real"].iloc[0])

            if not results["dates"]:
                continue

            available: list[str] = []

            if self.model1_coefficients is not None:
                m1 = calculate_rolling_predictions(client_data, famille, self.model1_coefficients, "model1")
                if len(m1) > 0:
                    available.append("model1")
                    results["model1_pred"] = self._align(m1, test_dates)

            if predictions is not None:
                m2 = predictions[predictions["famille"] == famille]
                if len(m2) > 0:
                    available.append("model2")
                    results["model2_pred"] = [
                        m2.loc[m2["date"] == td.strftime("%Y-W%W"), "prediction"].iloc[0]
                        if len(m2[m2["date"] == td.strftime("%Y-W%W")]) > 0 else 0
                        for td in test_dates
                    ]

            if coefficients is not None:
                m3 = calculate_rolling_predictions(client_data, famille, coefficients, "model3")
                if len(m3) > 0:
                    available.append("model3")
                    results["model3_pred"] = self._align(m3, test_dates)

            if available:
                plot_frequent_comparison(results, client_id, famille, available)

    @staticmethod
    def _align(preds: pd.DataFrame, test_dates: pd.DatetimeIndex) -> list[float]:
        return [
            preds.loc[preds["date"] == td, "prediction"].iloc[0]
            if len(preds[preds["date"] == td]) > 0 else 0
            for td in test_dates
        ]


class RareModelComparison:
    def run(self) -> None:
        for client_id in get_client_ids():
            self._compare_client(client_id)

    def _compare_client(self, client_id: str) -> None:
        client_data = load_csv_safe(f"data/{client_id}/rares.csv", parse_dates=["date"])
        preds = load_csv_safe(f"data/{client_id}/predictions_rares.csv")
        if client_data is None or preds is None:
            return

        families = [c for c in client_data.columns if c != "date"]
        test_dates = pd.date_range(start=TEST_START, end=TEST_END, freq=TEST_FREQ)

        for famille in families:
            fp = preds[preds["famille"] == famille]
            if fp.empty:
                continue
            real = get_real_values(client_data, famille)

            results: dict = {"dates": [], "real": [], "same_week_pred": [], "croston_pred": []}
            for td in test_dates:
                ds = td.strftime("%Y-W%W")
                r = real[real["date"] == ds]
                p = fp[fp["date"] == ds]
                if r.empty or p.empty:
                    continue
                results["dates"].append(ds)
                results["real"].append(r["real"].iloc[0])
                results["same_week_pred"].append(p["same_week_pred"].iloc[0])
                results["croston_pred"].append(p["croston_pred"].iloc[0])

            if results["dates"]:
                plot_rare_comparison(results, client_id, famille)
