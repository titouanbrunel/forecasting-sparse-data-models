from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    GLOBAL_LAG_STEPS,
    GLOBAL_ROLLING_WINDOWS,
    HOLIDAY_SEASON_MONTHS,
    LAG_STEPS,
    PEAK_SUMMER_MONTHS,
    ROLLING_WINDOWS,
    SEASON_MAP,
)


class GlobalFeatureBuilder:
    @staticmethod
    def transform(series: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({"val": series})
        df["week"] = series.index.isocalendar().week
        df["month"] = series.index.month
        df["quarter"] = series.index.quarter

        for lag in GLOBAL_LAG_STEPS:
            df[f"lag{lag}"] = df["val"].shift(lag)

        for window in GLOBAL_ROLLING_WINDOWS:
            shifted = df["val"].shift(1)
            df[f"avg{window}"] = shifted.rolling(window=window).mean()
            df[f"std{window}"] = shifted.rolling(window=window).std()

        df["diff"] = df["val"].shift(1) - df["val"].shift(2)
        df["accel"] = df["diff"] - (df["val"].shift(2) - df["val"].shift(3))
        df["ratio"] = df["lag1"] / (df["avg4"] + 1)
        df["trend"] = range(len(df))

        return df.dropna()


class ClientFeatureBuilder:
    def __init__(self, weekly_data: pd.DataFrame) -> None:
        self.weekly = weekly_data

    @property
    def feature_columns(self) -> list[str]:
        return [
            "trend", "week_sin", "week_cos", "month_sin", "month_cos",
            "week", "month", "quarter",
            "is_spring", "is_summer", "is_autumn", "is_winter",
            "is_peak_summer", "is_holiday_season",
            *[f"lag_{s}" for s in LAG_STEPS],
            *[f"ma_{w}" for w in ROLLING_WINDOWS],
            *[f"std_{w}" for w in ROLLING_WINDOWS],
            "volatility_4w", "max_4w", "min_4w",
            "had_order_last_week", "client_avg_quantity", "client_total_orders",
            "client_order_frequency", "client_summer_avg", "client_winter_avg",
            "avg_temp_c", "temp_range", "sun_hours", "lunch_temp", "lunch_precip",
            "dinner_temp", "dinner_precip", "is_rainy_day", "is_hot_day", "is_cold_day",
            "high_precip", "perfect_weather", "weather_quality",
            "is_jour_ferie", "vacation_intensity", "is_weekend",
            "event_impact_mean", "event_impact_max", "event_impact_min", "event_decay_mean",
            "has_lockdown", "has_sport_event", "has_cultural_event", "has_strike", "has_reopening",
        ]

    def transform(self, df_client: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        df = df_client.copy()
        df["year"] = df["date"].dt.year
        df["week"] = df["date"].dt.isocalendar().week
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter

        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        for name, months in SEASON_MAP.items():
            df[f"is_{name}"] = df["month"].isin(months).astype(int)
        df["is_peak_summer"] = df["month"].isin(PEAK_SUMMER_MONTHS).astype(int)
        df["is_holiday_season"] = df["month"].isin(HOLIDAY_SEASON_MONTHS).astype(int)

        merged = df.merge(self.weekly, on=["year", "week"], how="left").fillna(0)

        meta_cols = {
            "date", "year", "week", "month", "quarter",
            "week_sin", "week_cos", "month_sin", "month_cos",
            "is_spring", "is_summer", "is_autumn", "is_winter",
            "is_peak_summer", "is_holiday_season",
            *self.weekly.columns,
        }
        families = [c for c in merged.columns if c not in meta_cols]

        parts: list[pd.DataFrame] = []
        for family in families:
            cols = (
                ["date", "year", "week", "month", "quarter",
                 "week_sin", "week_cos", "month_sin", "month_cos",
                 "is_spring", "is_summer", "is_autumn", "is_winter",
                 "is_peak_summer", "is_holiday_season", family]
                + [c for c in self.weekly.columns if c not in ("year", "week")]
            )
            fd = merged[cols].copy().rename(columns={family: "quantity"})
            fd["famille"] = family
            fd = fd.sort_values(["year", "week"])

            for lag in LAG_STEPS:
                fd[f"lag_{lag}"] = fd["quantity"].shift(lag)

            shifted = fd["quantity"].shift(1)
            for w in ROLLING_WINDOWS:
                fd[f"ma_{w}"] = shifted.rolling(w).mean()
                fd[f"std_{w}"] = shifted.rolling(w).std()

            fd["trend"] = range(len(fd))
            fd["volatility_4w"] = shifted.rolling(4).std()
            fd["max_4w"] = shifted.rolling(4).max()
            fd["min_4w"] = shifted.rolling(4).min()

            fd["had_order_last_week"] = (fd["lag_1"] > 0).astype(int)
            fd["client_avg_quantity"] = fd["quantity"].expanding().mean().shift(1)
            fd["client_total_orders"] = (fd["quantity"] > 0).expanding().sum().shift(1)
            fd["client_order_frequency"] = fd["client_total_orders"] / (fd["trend"] + 1)

            for season in ("summer", "winter"):
                mask = fd[f"is_{season}"] == 1
                fd[f"client_{season}_avg"] = (
                    fd.loc[mask, "quantity"].expanding().mean().shift(1)
                )

            parts.append(fd)

        result = pd.concat(parts, ignore_index=True).fillna(0)

        for col in self.feature_columns:
            if col not in result.columns:
                result[col] = 0

        return result, self.feature_columns
