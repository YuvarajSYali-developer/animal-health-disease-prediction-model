import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "Animal_Type",
    "Disease_Prediction",
    "Body_Temperature",
    "Heart_Rate",
    "Respiratory_Rate",
    "Activity_Level",
}


def load_sources(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Copy data_sources.example.json to data_sources.json "
            "and update it with your dataset URLs."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_dataset(df: pd.DataFrame, source: dict) -> pd.DataFrame:
    column_map = source.get("column_map", {})
    df = df.rename(columns=column_map)

    symptom_columns = source.get("symptom_columns", [])
    for symptom in symptom_columns:
        if symptom not in df.columns:
            df[symptom] = 0

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            default_value = source.get("defaults", {}).get(col)
            if default_value is None:
                raise ValueError(f"Missing required column '{col}' in source {source.get('name')}")
            df[col] = default_value

    keep_columns = list(REQUIRED_COLUMNS) + symptom_columns
    df = df[keep_columns].copy()
    return df


def ingest_sources(sources):
    frames = []
    for source in sources:
        name = source.get("name", "unknown")
        url = source.get("url")
        if not url:
            raise ValueError(f"Source '{name}' is missing a URL.")
        print(f"[+] Loading source: {name}")
        df = pd.read_csv(url)
        frames.append(normalize_dataset(df, source))
    return pd.concat(frames, ignore_index=True)


def main():
    config_path = Path("data_sources.json")
    sources = load_sources(config_path)
    combined = ingest_sources(sources)
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "combined_dataset.csv"
    combined.to_csv(output_path, index=False)
    print(f"[[OK]] Combined dataset saved to {output_path}")


if __name__ == "__main__":
    main()
