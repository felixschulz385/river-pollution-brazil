import io
import logging
import re
import unicodedata
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    ACTION_MEASURES,
    ACTIONS,
    ADM2_COLUMNS,
    CLASSES,
    COVERAGE_START,
    DEFAULT_ADM2_OUTPUT_PATH,
    DEFAULT_PRIORITY_OUTPUT_PATH,
    DEFAULT_RECORDS_OUTPUT_PATH,
    ENCODING,
    FIRST_DATE,
    GARIMPO_PATTERN,
    ITEM_TABLES,
    LAT_RANGE,
    LON_RANGE,
    MINING_PATTERN,
    MUN_ID_COLUMN,
    PLACE_NAME_PATTERN,
    PRIORITY_BIOME,
    PRIORITY_COLUMNS,
    PRIORITY_STATE_CORRECTIONS,
    RECORD_COLUMNS,
    SEPARATOR,
    SNAPSHOT_PREFIX,
    STATE_CODES,
    TABLES,
    YEAR_COLUMN,
    action_columns,
    measure_column,
    priority_path,
    snapshot_path,
)
from src.data.sources.gadm.constants import DEFAULT_ADM2_LAYER, DEFAULT_SIMPLIFIED_GADM_PATH


logger = logging.getLogger(__name__)


def read_table(path, columns):
    """A raw table as text columns; a zip is read as the concatenation of its CSVs (fines: one per year)."""
    options = dict(sep=SEPARATOR, usecols=lambda column: column in columns, dtype=str,
                   keep_default_na=False, na_values=[""])
    path = Path(path)
    if path.suffix != ".zip":
        return pd.read_csv(path, encoding=ENCODING, **options)
    frames = []
    with zipfile.ZipFile(path) as bundle:
        for name in sorted(bundle.namelist()):
            if name.lower().endswith(".csv"):
                with bundle.open(name) as handle:
                    frames.append(pd.read_csv(io.TextIOWrapper(handle, encoding=ENCODING), **options))
    return pd.concat(frames, ignore_index=True)


_NAMES = re.compile(PLACE_NAME_PATTERN)


def _fold(text):
    return _NAMES.sub(" ", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode())


def fold_text(frame, columns):
    """The columns joined, lower-cased and accent-folded to ASCII, with place and other names removed."""
    joined = frame[columns[0]].fillna("")
    for column in columns[1:]:
        joined = joined + " | " + frame[column].fillna("")
    # Python's `re`: the name pattern uses a lookahead, which pyarrow's RE2 lacks.
    return joined.str.lower().map(_fold)


def classify(frame, columns):
    """`mining` and `garimpo` flags from the description columns."""
    text = fold_text(frame, columns)
    return pd.DataFrame({
        "mining": text.str.contains(MINING_PATTERN, regex=True),
        "garimpo": text.str.contains(GARIMPO_PATTERN, regex=True),
    }, index=frame.index)


def normalise_id(values):
    """IDs as clean text: trimmed, upper-cased, float formatting ("2126006.0000000000") removed."""
    cleaned = values.str.strip().str.upper().str.replace(r"\.0+$", "", regex=True)
    return cleaned.where(cleaned.str.len() > 0)


def _code_key(code, series):
    return normalise_id(code) + "-" + normalise_id(series).fillna("")


def record_ids(frame, spec):
    """`seq:<system ID>`, or `code:<printed code>-<series>` for records without one (AIe app)."""
    seq = normalise_id(frame[spec["seq"]])
    return ("seq:" + seq).fillna("code:" + _code_key(frame[spec["code"]], frame[spec["series"]]))


def to_number(values):
    """Numbers written with a decimal comma or point."""
    return pd.to_numeric(values.str.strip().str.replace(",", ".", regex=False), errors="coerce")


def to_date(values, last_date):
    """Dates from the leading `YYYY-MM-DD`; impossible ones (before 1977, after the snapshot) are missing."""
    dates = pd.to_datetime(values.str[:10], format="%Y-%m-%d", errors="coerce")
    return dates.where(dates.between(pd.Timestamp(FIRST_DATE), pd.Timestamp(last_date)))


def clean_coordinates(lon, lat):
    """Coordinates inside Brazil's bounding box; swapped pairs are swapped back, others dropped."""
    lon, lat = to_number(lon), to_number(lat)

    def inside(x, y):
        return x.between(*LON_RANGE) & y.between(*LAT_RANGE)

    keep, swap = inside(lon, lat), ~inside(lon, lat) & inside(lat, lon)
    clean_lon = lon.where(keep, lat.where(swap))
    clean_lat = lat.where(keep, lon.where(swap))
    return clean_lon, clean_lat


def item_flags(items, spec):
    """Item counts and mining and garimpo flags per term, keyed by system ID and by printed code."""
    flags = classify(items, spec["text"])
    seq_keys = "seq:" + normalise_id(items[spec["seq"]])
    code_keys = "code:" + _code_key(items[spec["code"]], items[spec["series"]])
    return tuple(
        flags.assign(key=keys, items=1).dropna(subset=["key"]).groupby("key").agg(
            items=("items", "sum"), mining=("mining", "any"), garimpo=("garimpo", "any"))
        for keys in (seq_keys, code_keys)
    )


def _first_filled(frame, columns):
    values = pd.Series(np.nan, index=frame.index, dtype=object)
    for column in columns:
        values = values.fillna(frame[column].str.strip().where(lambda text: text.str.len() > 0))
    return values


def action_records(action, table, items, last_date):
    """The record table rows of one action: non-cancelled records, cleaned, with mining flags."""
    spec = TABLES[action]
    table = table[table[spec["cancelled"]].str.strip().str.upper() == "N"].reset_index(drop=True)
    flags = classify(table, spec["text"])

    record = pd.DataFrame(index=table.index)
    record["action"] = action
    record["record_id"] = record_ids(table, spec)
    record["date"] = to_date(table[spec["date"]], last_date)
    record[YEAR_COLUMN] = record["date"].dt.year.astype("Int64")
    municipality = table[spec["municipality"]].str.strip().str.replace(r"\.0+$", "", regex=True)
    record[MUN_ID_COLUMN] = municipality.str[:6].where(municipality.str.fullmatch(r"\d{7}", na=False))
    record["uf"] = table[spec["uf"]].str.strip().str.upper()
    record["lon"], record["lat"] = clean_coordinates(table[spec["lon"]], table[spec["lat"]])
    record["value_brl"] = to_number(table[spec["value"]]) if spec["value"] else np.nan
    record["area_ha"] = to_number(table[spec["area"]]) if spec["area"] else np.nan
    record["lifted_date"] = to_date(table[spec["lifted"]], last_date) if spec["lifted"] else pd.NaT
    record["items"] = np.nan
    record["operation"] = _first_filled(table, spec["operation"])
    record["operation_code"] = normalise_id(table[spec["operation_code"]])
    record["operation_seq"] = normalise_id(table[spec["operation_seq"]])
    record["fine_seq"] = normalise_id(table[spec["fine_seq"]])
    record["fine_code"] = normalise_id(table[spec["fine_code"]])
    record["mining"], record["garimpo"] = flags["mining"], flags["garimpo"]

    if items is not None:
        by_seq, by_code = item_flags(items, ITEM_TABLES[action])
        seq_keys = "seq:" + normalise_id(table[spec["seq"]])
        code_keys = "code:" + _code_key(table[spec["code"]], table[spec["series"]])
        # A term takes its items by system ID, else by printed code.
        linked = by_seq.reindex(seq_keys).reset_index(drop=True)
        linked = linked.fillna(by_code.reindex(code_keys).reset_index(drop=True))
        record["items"] = linked["items"].fillna(0).to_numpy()
        record["mining"] |= linked["mining"].fillna(False).astype(bool).to_numpy()
        record["garimpo"] |= linked["garimpo"].fillna(False).astype(bool).to_numpy()
    record["mining"] |= record["garimpo"]
    return record[list(RECORD_COLUMNS)]


def snapshot_date(snapshot):
    """The download date in the snapshot folder's name (`ibama_<YYYY-MM-DD>`)."""
    return date.fromisoformat(Path(snapshot).name.removeprefix(SNAPSHOT_PREFIX))


def build_records(snapshot):
    """One row per fine, embargo, seizure and destruction term in the snapshot."""
    last_date = snapshot_date(snapshot)
    frames = []
    for action in ACTIONS:
        spec = TABLES[action]
        columns = {value for key, value in spec.items() if key not in ("file", "text", "operation") and value}
        columns |= {*spec["text"], *spec["operation"]}
        table = read_table(snapshot / spec["file"], columns)
        items = None
        if action in ITEM_TABLES:
            item_spec = ITEM_TABLES[action]
            items = read_table(snapshot / item_spec["file"],
                               {item_spec["seq"], item_spec["code"], item_spec["series"], *item_spec["text"]})
        records = action_records(action, table, items, last_date)
        logger.info("%s: %d records (%d dropped as cancelled), %d mining, %d garimpo", action, len(records),
                    len(table) - len(records), int(records["mining"].sum()), int(records["garimpo"].sum()))
        frames.append(records)
    return pd.concat(frames, ignore_index=True)


def adm2_enforcement(records, last_year):
    """Municipality x year counts and amounts by action and class, for municipality-years with a record.

    Each action's columns are zero from its `COVERAGE_START` to `last_year`
    and missing before.
    """
    first_year = min(COVERAGE_START.values())
    located = records.dropna(subset=[MUN_ID_COLUMN, YEAR_COLUMN])
    located = located[located[YEAR_COLUMN].between(first_year, last_year)].astype({YEAR_COLUMN: int})
    keys = [MUN_ID_COLUMN, YEAR_COLUMN]
    index = pd.MultiIndex.from_frame(located[keys].drop_duplicates().sort_values(keys))

    result = pd.DataFrame(index=index)
    for action in ACTIONS:
        rows = located[located["action"] == action]
        for group in CLASSES:
            members = rows if group == "all" else rows[rows[group]]
            grouped = members.groupby(keys)
            for measure, amount in ACTION_MEASURES[action]:
                values = grouped.size() if amount is None else grouped[amount].sum(min_count=1).fillna(0.0)
                result[measure_column(measure, group)] = values.reindex(index).fillna(0.0).astype(float)
        before = result.index.get_level_values(YEAR_COLUMN) < COVERAGE_START[action]
        result.loc[before, list(action_columns(action))] = np.nan
    return result.reset_index()[list(ADM2_COLUMNS)]


def build_enforcement(root_dir=".", snapshot=None, records_output_path=None, adm2_output_path=None):
    """Read the newest IBAMA snapshot and write the record and municipality x year tables."""
    snapshot = Path(snapshot) if snapshot else snapshot_path(root_dir)
    if snapshot is None:
        raise FileNotFoundError(
            "No IBAMA snapshot under data/enforcement/raw/ibama_<YYYY-MM-DD>/ with every table. "
            "Download the Fiscalização datasets from dadosabertos.ibama.gov.br (the portal blocks requests "
            "from outside Brazil) and place them, with their `<dataset>__` prefix, in a dated snapshot folder."
        )
    logger.info("Reading IBAMA tables from %s", snapshot)
    records = build_records(snapshot)
    last_year = snapshot_date(snapshot).year - 1
    adm2 = adm2_enforcement(records, last_year)

    outputs = []
    for frame, path in ((records, records_output_path or DEFAULT_RECORDS_OUTPUT_PATH),
                        (adm2, adm2_output_path or DEFAULT_ADM2_OUTPUT_PATH)):
        resolved = Path(root_dir) / path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(resolved, index=False)
        outputs.append(resolved)
    logger.info("Wrote %d records to %s and the ADM2 table %s to %s",
                len(records), outputs[0], adm2.shape, outputs[1])
    return records, adm2


def fold_name(name):
    """A municipality name lower-cased and accent-folded, with apostrophe variants and hyphens unified."""
    text = unicodedata.normalize("NFKD", name.replace("’", "'")).encode("ascii", "ignore").decode()
    return " ".join(text.lower().replace("-", " ").split())


def gadm_municipalities(gadm_path):
    """(IBGE state code, folded name) key -> 6-digit `mun_id`, from GADM's municipalities."""
    import pyogrio

    frame = pyogrio.read_dataframe(gadm_path, layer=DEFAULT_ADM2_LAYER, columns=["NAME_2", "CC_2"], read_geometry=False)
    codes = frame["CC_2"]
    codes = codes.round().astype("Int64").astype(str) if pd.api.types.is_float_dtype(codes) else codes.astype(str)
    keys = codes.str[:2] + "|" + frame["NAME_2"].map(fold_name)
    if keys.duplicated().any():
        raise ValueError(f"Municipality names repeated within a state in GADM: {sorted(keys[keys.duplicated()])}")
    return pd.Series(codes.str[:6].to_numpy(), index=keys.to_numpy())


def priority_list_adm2(table, municipalities):
    """Municipality x year flags of the Amazon Priority List and the monitored list.

    Rows cover every municipality on either list in any year, for every year
    of the table, with 0 where it is off the list. Rows are matched to
    `municipalities` on state and name; an unmatched row raises.
    """
    rows = table[table["biome"] == PRIORITY_BIOME].copy()
    states = [PRIORITY_STATE_CORRECTIONS.get((state, name), state) for state, name in zip(rows["uf"], rows["name_muni"])]
    keys = pd.Series(states, index=rows.index).map(STATE_CODES) + "|" + rows["name_muni"].map(fold_name)
    rows[MUN_ID_COLUMN] = keys.map(municipalities)
    unmatched = rows[rows[MUN_ID_COLUMN].isna()]
    if len(unmatched):
        raise ValueError("Priority List rows without a GADM municipality: "
                         f"{sorted(set(zip(unmatched['uf'], unmatched['name_muni'])))}")
    recoded = rows["code_muni"].astype(str).str[:6] != rows[MUN_ID_COLUMN]
    logger.info("Priority List: %d Amazon rows, %d recoded from the record's IBGE code", len(rows), int(recoded.sum()))

    keys = [MUN_ID_COLUMN, YEAR_COLUMN]
    flags = rows.groupby(keys)[["priority", "monitored"]].max()
    both = flags[(flags["priority"] > 0) & (flags["monitored"] > 0)]
    if len(both):
        raise ValueError(f"Municipality-years on both lists: {both.index.tolist()}")
    years = range(int(rows[YEAR_COLUMN].min()), int(rows[YEAR_COLUMN].max()) + 1)
    index = pd.MultiIndex.from_product([sorted(rows[MUN_ID_COLUMN].unique()), years], names=keys)
    flags = flags.reindex(index, fill_value=0).astype(float)
    result = flags.rename(columns={"priority": "priority_list", "monitored": "priority_monitored"}).reset_index()
    return result[list(PRIORITY_COLUMNS)]


def build_priority_list(root_dir=".", path=None, gadm_path=None, output_path=None):
    """Read the downloaded Priority List and write its municipality x year table."""
    path = Path(path) if path else priority_path(root_dir)
    if not path.exists():
        raise FileNotFoundError(f"No Priority List at {path}; run `data fetch --source enforcement`.")
    table = pd.read_csv(path, dtype={"code_muni": str})
    adm2 = priority_list_adm2(table, gadm_municipalities(gadm_path or Path(root_dir) / DEFAULT_SIMPLIFIED_GADM_PATH))
    resolved = Path(root_dir) / (output_path or DEFAULT_PRIORITY_OUTPUT_PATH)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    adm2.to_parquet(resolved, index=False)
    logger.info("Wrote the Priority List table %s to %s", adm2.shape, resolved)
    return adm2
