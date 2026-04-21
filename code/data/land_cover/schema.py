import pandas as pd

from .constants import (
    LAND_COVER_CLASS_PREFIX,
    LAND_COVER_TOTAL_COLUMN,
    TRENCH_ID_COLUMN,
    YEAR_COLUMN,
)


def validate_required_columns(frame, required_columns, frame_name):
    """Raise a clear error if a table is missing required columns."""
    missing_columns = set(required_columns).difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing required columns: {sorted(missing_columns)}."
        )


def validate_land_cover_output_columns(land_cover_df):
    """Validate the expected output schema before downstream aggregation."""
    required_columns = {TRENCH_ID_COLUMN, YEAR_COLUMN}
    missing_columns = required_columns.difference(land_cover_df.columns)
    if missing_columns:
        raise ValueError(
            "Land-cover input is missing required columns: "
            f"{sorted(missing_columns)}. Expected schema uses "
            f"`{TRENCH_ID_COLUMN}` and `{YEAR_COLUMN}`."
        )


def land_cover_assembly_columns(land_cover_df):
    """Return the preprocessed land-cover columns used by assembly outputs."""
    validate_land_cover_output_columns(land_cover_df)
    class_columns = [
        column
        for column in land_cover_df.columns
        if column.startswith(LAND_COVER_CLASS_PREFIX)
    ]
    if not class_columns:
        raise ValueError(
            "Land-cover input does not include any preprocessed "
            f"`{LAND_COVER_CLASS_PREFIX}*` columns."
        )
    lc_columns = [LAND_COVER_TOTAL_COLUMN, *class_columns]
    missing_columns = [
        column for column in lc_columns if column not in land_cover_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Land-cover input is missing expected preprocessed columns: "
            f"{missing_columns}."
        )
    return lc_columns


def normalize_optional_int(value):
    """Convert optional integer-like legend values to Python ints or None."""
    if pd.isna(value) or value == "":
        return None
    return int(value)


def subclass_summary_id(class_id, subclass_id):
    """Build a globally unique subclass summary id from class and subclass."""
    normalized_subclass = normalize_optional_int(subclass_id)
    if normalized_subclass is None:
        return None
    return int(class_id) * 10 + normalized_subclass


def get_output_columns(legend_path):
    """Derive preprocess output columns from the legend file."""
    legend = pd.read_excel(legend_path)
    class_ids = sorted(
        {int(class_id) for class_id in legend["Class"].dropna().tolist()}
    )
    subclass_ids = sorted(
        {
            summary_id
            for class_id, subclass_id in legend[["Class", "Subclass"]].itertuples(index=False)
            for summary_id in [subclass_summary_id(class_id, subclass_id)]
            if summary_id is not None
        }
    )
    return (
        [LAND_COVER_TOTAL_COLUMN]
        + [f"{LAND_COVER_CLASS_PREFIX}{int(class_id)}" for class_id in class_ids]
        + [f"{LAND_COVER_CLASS_PREFIX}{int(class_id)}" for class_id in subclass_ids]
    )
