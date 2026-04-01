import pandas as pd

from ..database import STATIONS_TABLE, STATION_RIVERS_TABLE, read_dataframe_table


def _normalise_station_code(value) -> str | None:
    """Return a stable string station code.

    DuckDB and pandas can round-trip identifier-looking columns as integers,
    floats, or strings depending on the source payload. Converting once here
    keeps filtering and resume logic consistent.
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalise_station_name(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_queryable_stations(root_dir="."):
    # The station preprocess step prepares a single curated station-to-trench
    # table. The archive API can expose multiple sensor families per station, so
    # every curated station remains eligible for scraping.
    curated_stations = read_dataframe_table(root_dir, STATION_RIVERS_TABLE)
    curated_stations = curated_stations.loc[:, ["station_code"]].copy()
    curated_stations["station_code"] = curated_stations["station_code"].map(_normalise_station_code)
    curated_stations = curated_stations.dropna(subset=["station_code"])
    curated_stations = curated_stations.reset_index(drop=True).drop_duplicates(subset=["station_code"])

    station_inventory = read_dataframe_table(root_dir, STATIONS_TABLE)
    station_inventory = station_inventory.loc[:, ["station_code", "station_name"]].copy()
    station_inventory["station_code"] = station_inventory["station_code"].map(_normalise_station_code)
    station_inventory["station_name"] = station_inventory["station_name"].map(_normalise_station_name)
    station_inventory = station_inventory.dropna(subset=["station_code", "station_name"])
    station_inventory = station_inventory.drop_duplicates(subset=["station_code"], keep="first")

    stations = curated_stations.merge(
        station_inventory,
        on="station_code",
        how="left",
        validate="one_to_one",
    )
    missing_names = stations["station_name"].isna()
    if missing_names.any():
        missing_codes = stations.loc[missing_names, "station_code"].tolist()
        raise KeyError(
            "Missing station_name values in stations table for station code(s): "
            + ", ".join(missing_codes[:10])
        )
    return stations.set_index("station_code")
