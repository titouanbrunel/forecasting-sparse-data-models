import argparse
import os

from tqdm import tqdm

from src.data import DataLoader
from src.model import CoefficientModel, HurdleClientModel, RareFamiliesModel
from src.comparison import FrequentModelComparison, RareModelComparison
from src.utils import get_client_ids


def run_frequent(loader: DataLoader) -> None:
    coeff = CoefficientModel(loader)
    hurdle = HurdleClientModel(loader)

    coeff.generate_global_coefficients()
    coeff.generate_client_coefficients()

    for cid in tqdm(get_client_ids(), desc="Frequent predictions"):
        if os.path.exists(f"data/{cid}/predictions.csv"):
            continue
        hurdle.predict_client(cid)

    FrequentModelComparison().run()


def run_rare() -> None:
    model = RareFamiliesModel()
    for cid in tqdm(get_client_ids(), desc="Rare predictions"):
        model.predict_client(cid)
    RareModelComparison().run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Demand Forecasting")
    parser.add_argument("--mode", choices=["frequent", "rare", "all"], default="all")
    args = parser.parse_args()

    if args.mode in ("frequent", "all"):
        loader = DataLoader()
        loader.load_all()
        run_frequent(loader)

    if args.mode in ("rare", "all"):
        run_rare()


if __name__ == "__main__":
    main()
