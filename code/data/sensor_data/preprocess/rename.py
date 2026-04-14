from .settings import PORTUGUESE_TO_ENGLISH_COLUMNS


def rename_portuguese_fields(frame, column_map=None):
    """Rename source fields to English names when mappings are configured."""
    mapping = PORTUGUESE_TO_ENGLISH_COLUMNS if column_map is None else column_map
    return frame.rename(columns={k: v for k, v in mapping.items() if k in frame.columns})
