from pathlib import Path

DATA_DIR = Path("data")
FAMILIES_DIR = DATA_DIR / "families_w"
WEATHER_FILE = DATA_DIR / "weather.json"
EVENTS_FILE = DATA_DIR / "events.json"

TEST_YEAR = 2024
TEST_START = f"{TEST_YEAR}-01-01"
TEST_END = f"{TEST_YEAR}-12-30"
TEST_FREQ = "W-MON"

ROLLING_WINDOW = 52
MIN_TRAIN_SAMPLES = 100
MIN_FEATURE_SAMPLES = 10

XGBOOST_PARAMS: dict = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "reg:squarederror",
}

CROSTON_ALPHA = 0.2
LOGISTIC_MAX_ITER = 1000

LAG_STEPS = [1, 2, 4, 8, 12]
ROLLING_WINDOWS = [4, 8, 12, 26]
GLOBAL_LAG_STEPS = [1, 2, 3, 4]
GLOBAL_ROLLING_WINDOWS = [4, 8]

HOT_THRESHOLD = 25
COLD_THRESHOLD = 10
HIGH_PRECIP_THRESHOLD = 5
PERFECT_WEATHER_TEMP_RANGE = (18, 25)
PERFECT_WEATHER_MAX_PRECIP = 1
PERFECT_WEATHER_MIN_SUN = 4

SEASON_MAP: dict[str, list[int]] = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
    "winter": [12, 1, 2],
}
PEAK_SUMMER_MONTHS = [7, 8]
HOLIDAY_SEASON_MONTHS = [7, 8, 12]

SAME_WEEK_WINDOW_DAYS = 3

VALID_FAMILIES = [
    "ALCOOLS", "APERITIF SANS ALCOOL", "ARTE DE LA TABLE", "BIERE BOITE",
    "BIERE BOUTEILLE", "BIERE FUT", "BOUTEILLE GAZ", "CAFE", "CAISSE CONSIGNEE",
    "CHAMPAGNE", "CIDRE", "COFFRET", "EAUX", "EMBALLAGE", "EPICES", "EQUIPEMENT",
    "FAMILLE_NON_DEFINIE", "FOOD", "GLACES", "HYGIENE", "JUS DE FRUIT", "KIT",
    "LAIT", "LOCATION", "N/Q", "PROSECCO", "PUB", "SAN BITTER", "SIROP",
    "SODA & LIMO", "TABASCO", "THE", "VINS",
]

PLOT_FIGSIZE = (16, 8)
PLOT_DPI = 300
PLOT_TICK_STEP_DIVISOR = 10

WEATHER_QUALITY_WEIGHTS = {"perfect": 0.5, "rain_free": 0.3, "temp": 0.2}
