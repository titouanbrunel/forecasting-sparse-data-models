from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.config import (
    CROSTON_ALPHA,
    LOGISTIC_MAX_ITER,
    MIN_FEATURE_SAMPLES,
    MIN_TRAIN_SAMPLES,
    ROLLING_WINDOW,
    SAME_WEEK_WINDOW_DAYS,
    TEST_END,
    TEST_START,
    TEST_FREQ,
    TEST_YEAR,
    XGBOOST_PARAMS,
)
from src.data import DataLoader
from src.features import ClientFeatureBuilder, GlobalFeatureBuilder


class HurdleModel:
    def __init__(self) -> None:
        self.classifier = LogisticRegression(random_state=42, max_iter=LOGISTIC_MAX_ITER)
        self.regressor = xgb.XGBRegressor(**XGBOOST_PARAMS)
        self.scaler_cls = StandardScaler()
        self.scaler_reg = StandardScaler()
        self.fitted = False
        self.hurdle_mode = True

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> None:
        if len(X) == 0:
            return
        y_bin = (y > 0).astype(int)

        if y_bin.sum() == len(y_bin):
            self.hurdle_mode = False
            self.regressor.fit(X, y, verbose=False)
            self.fitted = True
            return

        self.hurdle_mode = True
        X_scaled = self.scaler_cls.fit_transform(X)
        self.classifier.fit(X_scaled, y_bin)

        pos = y > 0
        if pos.sum() > 10:
            X_pos = self.scaler_reg.fit_transform(X[pos])
            self.regressor.fit(X_pos, y[pos], verbose=False)
            self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([0])

        if not self.hurdle_mode:
            return np.maximum(self.regressor.predict(X), 0)

        prob = self.classifier.predict_proba(self.scaler_cls.transform(X))[:, 1]
        if self.fitted:
            count = np.maximum(self.regressor.predict(self.scaler_reg.transform(X)), 0)
        else:
            count = np.ones(len(X))
        return prob * count


class CoefficientModel:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.feature_builder = GlobalFeatureBuilder()

    def generate_global_coefficients(self) -> pd.DataFrame:
        rows: list[dict] = []
        test_dates = pd.date_range(start=TEST_START, end=TEST_END, freq=TEST_FREQ)

        for name, fam_df in self.loader.families.items():
            series = fam_df.groupby("date")[name].sum()
            for td in test_dates:
                _, coeff = self._predict_coefficient(series, td)
                rows.append({"date": td, "famille": name, "coefficient": coeff})

        df = pd.DataFrame(rows)
        df.to_csv(str(self.loader.families_dir_path / "coefficients.csv") if hasattr(self.loader, "families_dir_path") else "data/families_w/coefficients.csv", index=False)
        return df

    def generate_client_coefficients(self) -> None:
        import os
        from src.config import DATA_DIR

        clients = [d for d in os.listdir(DATA_DIR) if d.isdigit()]
        test_dates = pd.date_range(start=TEST_START, end=TEST_END, freq=TEST_FREQ)

        for client_id in clients:
            client_data = self.loader.load_client(client_id)
            if client_data is None:
                continue

            families = [c for c in client_data.columns if c != "date"]
            rows: list[dict] = []

            for famille in families:
                hist = self._historical_coefficients(client_data, famille)
                for td in test_dates:
                    week_num = td.isocalendar().week
                    coeff = self._predict_week_coefficient(hist, week_num)
                    rows.append({"date": td, "famille": famille, "coefficient": coeff})

            if rows:
                pd.DataFrame(rows).to_csv(f"data/{client_id}/coef_families.csv", index=False)

    def _predict_coefficient(
        self, series: pd.Series, test_date: pd.Timestamp
    ) -> tuple[float, float]:
        train = series[series.index < test_date]

        if len(train) < ROLLING_WINDOW:
            return train.mean() if len(train) > 0 else 0, 1.0

        features = self.feature_builder.transform(train)
        if len(features) < MIN_FEATURE_SAMPLES:
            return train.mean(), 1.0

        X = features.drop("val", axis=1)
        y = features["val"]

        if (y == 0).sum() == 0:
            model = xgb.XGBRegressor(**XGBOOST_PARAMS)
            model.fit(X, y, verbose=False)
        else:
            model = HurdleModel()
            model.fit(X.values, y.values, feature_names=X.columns.tolist())

        extended = pd.concat([train, pd.Series([np.nan], index=[test_date])])
        ext_feat = self.feature_builder.transform(extended)
        if len(ext_feat) == 0:
            return train.mean(), 1.0

        X_pred = ext_feat.drop("val", axis=1).iloc[[-1]]
        if isinstance(model, HurdleModel):
            pred = max(0, model.predict(X_pred.values)[0])
        else:
            pred = max(0, model.predict(X_pred)[0])

        rolling = train.tail(ROLLING_WINDOW).mean()
        coeff = pred / rolling if rolling > 0 else 1.0
        return pred, coeff

    def _historical_coefficients(
        self, client_data: pd.DataFrame, famille: str
    ) -> dict[int, dict[int, float]]:
        fd = client_data[["date", famille]].rename(columns={famille: "quantity"}).copy()
        fd["year"] = fd["date"].dt.year
        fd["week"] = fd["date"].dt.isocalendar().week

        coefficients: dict[int, dict[int, float]] = {}
        for _, row in fd.iterrows():
            prev = fd[fd["date"] < row["date"]]
            if len(prev) < ROLLING_WINDOW:
                continue
            rolling = prev["quantity"].tail(ROLLING_WINDOW).mean()
            if rolling > 0:
                week_key = int(row["week"])
                coefficients.setdefault(week_key, {})[int(row["year"])] = row["quantity"] / rolling
        return coefficients

    @staticmethod
    def _predict_week_coefficient(
        hist: dict[int, dict[int, float]], week_number: int
    ) -> float:
        if week_number not in hist:
            return 1.0
        vals = [v for y, v in hist[week_number].items() if y != TEST_YEAR]
        return np.mean(vals) if vals else 1.0


class HurdleClientModel:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.feature_builder = ClientFeatureBuilder(loader.weekly)

    def predict_client(self, client_id: str) -> pd.DataFrame | None:
        client_data = self.loader.load_client(client_id)
        if client_data is None:
            return None

        df_full, feat_cols = self.feature_builder.transform(client_data)
        families = df_full["famille"].unique()
        predictions: list[dict] = []

        for famille in families:
            fd = df_full[df_full["famille"] == famille].sort_values(["year", "week"])
            test_fd = fd[fd["year"] == TEST_YEAR]
            if test_fd.empty:
                continue

            for _, row in test_fd.iterrows():
                train = df_full[
                    (df_full["year"] < row["year"])
                    | ((df_full["year"] == row["year"]) & (df_full["week"] < row["week"]))
                ]
                if len(train) < MIN_TRAIN_SAMPLES:
                    continue

                model = HurdleModel()
                model.fit(train[feat_cols].values, train["quantity"].values, feature_names=feat_cols)
                pred = model.predict(row[feat_cols].values.reshape(1, -1))[0]

                predictions.append({
                    "date": f"{int(row['year'])}-W{int(row['week']):02d}",
                    "famille": famille,
                    "prediction": max(0, pred),
                })

        df = pd.DataFrame(predictions)
        df.to_csv(f"data/{client_id}/predictions.csv", index=False)
        return df


class CrostonModel:
    def __init__(self, alpha: float = CROSTON_ALPHA) -> None:
        self.alpha = alpha
        self.demand_forecast: float | None = None
        self.interval_forecast: float | None = None

    def fit(self, series: np.ndarray) -> None:
        demands = [v for v in series if v > 0]
        if not demands:
            self.demand_forecast = 0
            self.interval_forecast = len(series)
            return

        intervals: list[int] = []
        last_idx = -1
        for i, v in enumerate(series):
            if v > 0:
                if last_idx >= 0:
                    intervals.append(i - last_idx)
                last_idx = i

        if not intervals:
            self.demand_forecast = np.mean(demands)
            self.interval_forecast = len(series)
            return

        d_forecast = float(demands[0])
        i_forecast = float(intervals[0])
        for d in demands[1:]:
            d_forecast = self.alpha * d + (1 - self.alpha) * d_forecast
        for iv in intervals[1:]:
            i_forecast = self.alpha * iv + (1 - self.alpha) * i_forecast

        self.demand_forecast = d_forecast
        self.interval_forecast = max(i_forecast, 1)

    def predict(self) -> float:
        if self.demand_forecast is None or self.interval_forecast is None:
            return 0
        return self.demand_forecast / self.interval_forecast


class RareFamiliesModel:
    def predict_client(self, client_id: str) -> pd.DataFrame | None:
        from src.config import DATA_DIR

        path = DATA_DIR / client_id / "rares.csv"
        if not path.exists():
            return None
        client_data = pd.read_csv(path)
        client_data["date"] = pd.to_datetime(client_data["date"])

        families = [c for c in client_data.columns if c != "date"]
        test_dates = pd.date_range(start=TEST_START, end=TEST_END, freq=TEST_FREQ)
        rows: list[dict] = []

        for famille in families:
            for td in test_dates:
                rows.append({
                    "date": td.strftime("%Y-W%W"),
                    "famille": famille,
                    "same_week_pred": max(0, self._same_week_last_year(client_data, famille, td)),
                    "croston_pred": max(0, self._croston_predict(client_data, famille, td)),
                })

        df = pd.DataFrame(rows)
        df.to_csv(f"data/{client_id}/predictions_rares.csv", index=False)
        return df

    @staticmethod
    def _same_week_last_year(
        data: pd.DataFrame, famille: str, test_date: pd.Timestamp
    ) -> float:
        fd = data[["date", famille]].rename(columns={famille: "quantity"})
        target = test_date - pd.Timedelta(days=365)
        window = pd.Timedelta(days=SAME_WEEK_WINDOW_DAYS)

        match = fd[(fd["date"] >= target - window) & (fd["date"] <= target + window)]
        if len(match) > 0:
            return match["quantity"].iloc[0]

        hist = fd[fd["date"] < test_date]
        if hist.empty:
            return 0
        orders = hist[hist["quantity"] > 0]
        if orders.empty:
            return 0

        freq = len(orders) / len(hist)
        return orders["quantity"].mean() if np.random.random() < freq else 0

    @staticmethod
    def _croston_predict(
        data: pd.DataFrame, famille: str, test_date: pd.Timestamp
    ) -> float:
        fd = data[["date", famille]].rename(columns={famille: "quantity"})
        hist = fd[fd["date"] < test_date]
        if len(hist) < 4:
            return 0
        model = CrostonModel()
        model.fit(hist["quantity"].values)
        return model.predict()
