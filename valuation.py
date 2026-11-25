"""Dataset validation, apartment valuation, and comparable statistics."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATASET_PATH = Path(__file__).resolve().with_name("apartment_prices.csv")
NUMERIC_FEATURES = ["area_sqm", "rooms", "floor"]
FEATURES = [*NUMERIC_FEATURES, "city"]
TARGET = "price_usd"
REQUIRED_COLUMNS = {"apartment_id", *FEATURES, TARGET}
COMPARABLE_COLUMNS = ["city", *NUMERIC_FEATURES, TARGET, "price_per_sqm"]
TEST_FRACTION = 0.20
RANDOM_SEED = 42


def load_dataset(path=DATASET_PATH):
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {', '.join(sorted(missing))}")
    if len(data) < 6:
        raise ValueError("At least six apartments are required for training and evaluation.")
    if data["apartment_id"].isna().any() or data["apartment_id"].duplicated().any():
        raise ValueError("Apartment IDs must be present and unique.")
    data["city"] = data["city"].astype("string").str.strip()
    if data["city"].isna().any() or data["city"].eq("").any():
        raise ValueError("City names must be present.")
    numeric = [*NUMERIC_FEATURES, TARGET]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    values = data[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Area, room count, floor, and prices must be finite and positive.")
    if (data[NUMERIC_FEATURES] % 1 != 0).any().any():
        raise ValueError("Area, room count, and floor must be whole numbers.")
    return data


def train_model(data):
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=TEST_FRACTION, random_state=RANDOM_SEED
    )
    encoder = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
        ]
    )
    model = Pipeline([
        ("features", encoder),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }
    return model, metrics


def predict_price(model, city, area, rooms, floor):
    apartment = pd.DataFrame([{
        "city": city, "area_sqm": area, "rooms": rooms, "floor": floor,
    }])
    return float(model.predict(apartment)[0])


def find_similar_apartments(data, city, area, rooms):
    similar = data.loc[
        data["city"].eq(city)
        & data["area_sqm"].between(area * 0.8, area * 1.2)
        & data["rooms"].eq(rooms)
    ].copy()
    similar["price_per_sqm"] = (similar[TARGET] / similar["area_sqm"]).round(2)
    similar["area_difference"] = (similar["area_sqm"] - area).abs()
    return similar.sort_values("area_difference", kind="stable")


def get_statistics(data):
    if data.empty:
        return {"count": 0}
    prices = data[TARGET]
    return {
        "count": len(data),
        "average_price_usd": round(float(prices.mean()), 2),
        "median_price_usd": round(float(prices.median()), 2),
        "minimum_price_usd": round(float(prices.min()), 2),
        "maximum_price_usd": round(float(prices.max()), 2),
        "average_area_sqm": round(float(data["area_sqm"].mean()), 2),
        "average_price_per_sqm": round(float((prices / data["area_sqm"]).mean()), 2),
    }


if __name__ == "__main__":
    dataset = load_dataset()
    _, evaluation = train_model(dataset)
    print(f"Apartments: {len(dataset)} ({evaluation['train_rows']} train / "
          f"{evaluation['test_rows']} test)")
    print(f"R2: {evaluation['r2']:.6f}")
    print(f"MAE: ${evaluation['mae']:,.2f}")
    print(f"RMSE: ${evaluation['rmse']:,.2f}")
