from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.config import (
    COLD_THRESHOLD,
    DATA_DIR,
    EVENTS_FILE,
    FAMILIES_DIR,
    HIGH_PRECIP_THRESHOLD,
    HOT_THRESHOLD,
    PERFECT_WEATHER_MAX_PRECIP,
    PERFECT_WEATHER_MIN_SUN,
    PERFECT_WEATHER_TEMP_RANGE,
    WEATHER_FILE,
    WEATHER_QUALITY_WEIGHTS,
)


class DataLoader:
    def __init__(self) -> None:
        self.weather: pd.DataFrame | None = None
        self.events: pd.DataFrame | None = None
        self.weekly: pd.DataFrame | None = None
        self.families: dict[str, pd.DataFrame] = {}
        self.client_data: pd.DataFrame | None = None

    def load_all(self) -> None:
        self.weather = self._load_weather()
        self.events = self._load_events()
        self.weekly = self._aggregate_weekly()
        self.families = self._load_families(FAMILIES_DIR)

    def load_client(self, client_id: str, kind: str = "frequentes") -> pd.DataFrame | None:
        path = DATA_DIR / client_id / f"{kind}.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        self.client_data = df
        return df

    def _load_weather(self) -> pd.DataFrame:
        with open(WEATHER_FILE) as f:
            raw = json.load(f)

        df = pd.DataFrame(raw["daily_data"])
        df["date"] = pd.to_datetime(df["date"])

        df["lunch_temp"] = df["lunch_period"].apply(lambda x: x["avg_temp_c"])
        df["lunch_precip"] = df["lunch_period"].apply(lambda x: x["total_precip_mm"])
        df["dinner_temp"] = df["dinner_period"].apply(lambda x: x["avg_temp_c"])
        df["dinner_precip"] = df["dinner_period"].apply(lambda x: x["total_precip_mm"])

        df["temp_range"] = df["max_temp_c"] - df["min_temp_c"]
        df["is_hot_day"] = (df["avg_temp_c"] > HOT_THRESHOLD).astype(int)
        df["is_cold_day"] = (df["avg_temp_c"] < COLD_THRESHOLD).astype(int)
        total_precip = df["lunch_precip"] + df["dinner_precip"]
        df["high_precip"] = (total_precip > HIGH_PRECIP_THRESHOLD).astype(int)
        df["perfect_weather"] = (
            df["avg_temp_c"].between(*PERFECT_WEATHER_TEMP_RANGE)
            & (total_precip < PERFECT_WEATHER_MAX_PRECIP)
            & (df["sun_hours"] >= PERFECT_WEATHER_MIN_SUN)
        ).astype(int)

        return df

    def _load_events(self) -> pd.DataFrame:
        with open(EVENTS_FILE) as f:
            raw = json.load(f)

        event_list = raw.get("events", raw) if isinstance(raw, dict) else raw
        if not isinstance(event_list, list):
            return pd.DataFrame()

        rows: list[dict] = []
        for event in event_list:
            if not isinstance(event, dict):
                continue
            start = pd.to_datetime(event["date"])
            duration = event["duration_days"]
            impact = event["impact"]
            for i in range(duration):
                rows.append({
                    "date": start + pd.Timedelta(days=i),
                    "event_name": event["name"],
                    "event_type": event["type"],
                    "event_impact": impact,
                    "event_decay": impact * (1 - i / duration),
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        agg = df.groupby("date").agg({
            "event_impact": "sum",
            "event_decay": "sum",
            "event_name": lambda x: " + ".join(x),
            "event_type": lambda x: " + ".join(x),
        }).reset_index()

        for label in ["lockdown", "sport_event", "cultural_event", "strike", "reopening"]:
            agg[f"has_{label}"] = agg["event_type"].str.contains(label).astype(int)

        return agg

    def _aggregate_weekly(self) -> pd.DataFrame:
        weather = self.weather.copy()
        weather["year"] = weather["date"].dt.year
        weather["week"] = weather["date"].dt.isocalendar().week

        weather_agg = weather.groupby(["year", "week"]).agg({
            "avg_temp_c": "mean", "max_temp_c": "max", "min_temp_c": "min",
            "temp_range": "mean", "sun_hours": "sum", "uv_index": "mean",
            "lunch_temp": "mean", "lunch_precip": "sum",
            "dinner_temp": "mean", "dinner_precip": "sum",
            "is_rainy_day": "sum", "is_windy_day": "sum",
            "is_hot_day": "sum", "is_cold_day": "sum",
            "high_precip": "sum", "perfect_weather": "sum",
            "is_jour_ferie": "sum", "is_vacances_zone_a": "sum", "is_weekend": "sum",
        }).reset_index()

        event_cols = [
            "event_impact_mean", "event_impact_max", "event_impact_min",
            "event_decay_mean", "has_lockdown", "has_sport_event",
            "has_cultural_event", "has_strike", "has_reopening",
        ]

        if self.events.empty or "date" not in self.events.columns:
            for col in event_cols:
                weather_agg[col] = 0
            weekly = weather_agg
        else:
            ev = self.events.copy()
            ev["year"] = ev["date"].dt.year
            ev["week"] = ev["date"].dt.isocalendar().week
            ev_agg = ev.groupby(["year", "week"]).agg({
                "event_impact": ["mean", "max", "min"],
                "event_decay": "mean",
                "has_lockdown": "max", "has_sport_event": "max",
                "has_cultural_event": "max", "has_strike": "max", "has_reopening": "max",
            }).reset_index()
            ev_agg.columns = ["year", "week"] + event_cols
            weekly = weather_agg.merge(ev_agg, on=["year", "week"], how="left").fillna(0)

        w = WEATHER_QUALITY_WEIGHTS
        weekly["vacation_intensity"] = weekly["is_vacances_zone_a"] / 7
        weekly["weather_quality"] = (
            weekly["perfect_weather"] / 7 * w["perfect"]
            + (7 - weekly["is_rainy_day"]) / 7 * w["rain_free"]
            + weekly["avg_temp_c"] / 30 * w["temp"]
        )

        return weekly

    def _load_families(self, folder: os.PathLike) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        folder_path = str(folder)
        for fname in os.listdir(folder_path):
            if not fname.endswith(".csv"):
                continue
            name = fname.replace(".csv", "")
            df = pd.read_csv(os.path.join(folder_path, fname))
            df["date"] = pd.to_datetime(df["date"])
            result[name] = df
        return result
