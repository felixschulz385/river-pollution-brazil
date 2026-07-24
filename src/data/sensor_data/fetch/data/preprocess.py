import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from time import monotonic

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    class _TqdmFallback:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])

        def update(self, count=1):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable, *args, **kwargs)

from ..database import (
    RAW_ARCHIVES_TABLE,
    SENSOR_ARCHIVE_FILES_TABLE,
    append_dataframe_table,
    write_dataframe_table,
)
from ..paths import (
    ensure_water_quality_dirs,
    get_download_log_database_path,
    get_raw_dir,
    get_sensor_database_path,
)
from .access_reader import read_archive_payload

logger = logging.getLogger(__name__)
LOG_EVERY_N_ARCHIVES = 500
LOG_EVERY_N_TABLES_READ = 1000
FLUSH_EVERY_N_ARCHIVES = 500
DEFAULT_PREPROCESS_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))
DEFAULT_PREPROCESS_BACKEND = "thread"
PREPROCESS_METADATA_FILENAME = "sensor_data_preprocess_metadata.md"

ARCHIVE_TABLE_RECORD_COLUMNS = [
    "station_code",
    "source_archive_name",
    "source_mdb_name",
    "source_table_name",
    "table_name",
    "row_count",
    "is_empty_table",
]


def _load_name_mapping() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Return empty mappings because raw database names are preserved."""
    return {}, {}


def list_valid_raw_archives(root_dir="."):
    """Identify downloaded MDB archives that should be preprocessed."""
    raw_dir = get_raw_dir(root_dir)
    files = pd.DataFrame({"filename": [path.name for path in raw_dir.iterdir() if path.is_file()]})
    if files.empty:
        return files

    files["file_size_bytes"] = files["filename"].apply(lambda name: (raw_dir / name).stat().st_size)
    files["station_code"] = files["filename"].str.extract(r"(^\d+)").iloc[:, 0]
    files = files.loc[
        (files["file_size_bytes"] > 22)
        & files["station_code"].notna()
        & files["filename"].str.contains(r"_mdb.*\.zip$", case=False, regex=True),
        :,
    ].copy()
    if files.empty:
        return files

    return files.reset_index(drop=True)


def _append_pending_frame(
    root_dir: str,
    table_name: str,
    frames: list[pd.DataFrame],
    written_tables: set[str],
) -> tuple[int, int]:
    """Flush one table batch to DuckDB and return written rows plus frame count."""
    if not frames:
        return 0, 0

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if table_name in written_tables:
        append_dataframe_table(root_dir, table_name, combined)
    else:
        write_dataframe_table(root_dir, table_name, combined)
        written_tables.add(table_name)
    return len(combined), len(frames)


def _parse_source_tables(source_tables: str | list[str] | None) -> list[str] | None:
    if source_tables is None:
        return None
    if isinstance(source_tables, str):
        parsed = [table.strip() for table in source_tables.split(",")]
    else:
        parsed = [str(table).strip() for table in source_tables]
    return [table for table in parsed if table]


def _format_markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_None._"

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def _write_preprocess_metadata(
    root_dir: str,
    *,
    raw_dir: Path,
    started_at_iso: str,
    elapsed_minutes: float,
    total_input_archives: int,
    single_station: str | None,
    requested_source_tables: list[str] | None,
    tables_to_read: list[str] | None,
    worker_count: int,
    preprocess_backend: str,
    log_every_tables: int,
    processed_archives: int,
    archives_without_mdb: int,
    archives_with_only_empty_tables: int,
    extracted_mdb_count: int,
    source_table_count: int,
    nonempty_source_table_count: int,
    parsed_row_count: int,
    flush_count: int,
    written_row_counts: dict[str, int],
    written_frame_counts: dict[str, int],
) -> Path:
    """Write a human-readable run summary beside the main sensor database."""
    sensor_database_path = get_sensor_database_path(root_dir)
    download_log_database_path = get_download_log_database_path(root_dir)
    metadata_path = sensor_database_path.with_name(PREPROCESS_METADATA_FILENAME)
    completed_at_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    source_table_mode = "all discovered source tables" if requested_source_tables is None else "requested source tables"
    if tables_to_read is None:
        source_table_preview = "discovered per MDB"
    else:
        source_table_preview = ", ".join(tables_to_read[:20])
        if len(tables_to_read) > 20:
            source_table_preview += f", ... ({len(tables_to_read)} total)"

    output_rows = [
        [table_name, row_count, written_frame_counts.get(table_name, 0)]
        for table_name, row_count in sorted(written_row_counts.items())
    ]

    content = f"""# Sensor Data Preprocess Metadata

## Run

| Field | Value |
| --- | --- |
| Started at | {started_at_iso} |
| Completed at | {completed_at_iso} |
| Elapsed minutes | {elapsed_minutes:.2f} |
| Root directory | {Path(root_dir).resolve()} |
| Raw archive directory | {raw_dir.resolve()} |
| Main database | {sensor_database_path.resolve()} |
| Download log database | {download_log_database_path.resolve()} |

## Options

| Field | Value |
| --- | --- |
| Single station | {single_station or ""} |
| Source table mode | {source_table_mode} |
| Source tables | {source_table_preview} |
| Worker count | {worker_count} |
| Backend | {preprocess_backend} |
| Log every tables | {log_every_tables} |

## Counts

| Field | Value |
| --- | --- |
| Input archives | {total_input_archives} |
| Processed archives | {processed_archives} |
| Archives without MDB | {archives_without_mdb} |
| Archives with only empty tables | {archives_with_only_empty_tables} |
| Extracted MDB files | {extracted_mdb_count} |
| Source tables read | {source_table_count} |
| Non-empty source tables | {nonempty_source_table_count} |
| Parsed rows | {parsed_row_count} |
| DuckDB flush batches | {flush_count} |
| Output tables | {len(written_row_counts)} |

## Output Tables

{_format_markdown_table(["Table", "Rows", "Source frames"], output_rows)}
"""
    metadata_path.write_text(content, encoding="utf-8")
    return metadata_path


def preprocess_station_data(
    root_dir=".",
    single_station=None,
    preprocess_workers: int | None = None,
    source_tables: str | list[str] | None = None,
    preprocess_backend: str = DEFAULT_PREPROCESS_BACKEND,
    log_every_tables: int = LOG_EVERY_N_TABLES_READ,
):
    """Parse downloaded MDB archives and persist raw Access tables into DuckDB."""
    started_at = monotonic()
    started_at_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    water_quality_dir, raw_dir = ensure_water_quality_dirs(root_dir)
    files = list_valid_raw_archives(root_dir)
    if single_station is not None:
        files = files.loc[files["station_code"].astype(str) == str(single_station)].copy()
        logger.info("Filtering preprocess input to station %s.", single_station)
    if files.empty:
        raise ValueError(f"No valid raw station MDB archives found in {raw_dir}.")

    table_map, column_map = _load_name_mapping()
    requested_source_tables = _parse_source_tables(source_tables)
    tables_to_read = requested_source_tables
    worker_count = preprocess_workers or DEFAULT_PREPROCESS_WORKERS
    worker_count = max(1, int(worker_count))
    executor_class = ProcessPoolExecutor if preprocess_backend == "process" else ThreadPoolExecutor
    extract_root = water_quality_dir / "_mdb_extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Preprocessing %s raw MDB archive(s) from %s into %s.",
        len(files),
        raw_dir,
        water_quality_dir,
    )
    logger.info(
        "Raw database table and field names will be preserved.",
    )
    if requested_source_tables is None:
        logger.info(
            "Discovering Access source tables with %s %s worker(s).",
            worker_count,
            preprocess_backend,
        )
    else:
        logger.info(
            "Reading only requested source table(s) with %s %s worker(s): %s.",
            worker_count,
            preprocess_backend,
            ", ".join(requested_source_tables),
        )

    archive_metadata = files.copy()
    archive_metadata["archive_kind"] = "mdb"
    archive_metadata["is_empty_archive"] = 0

    archive_table_records: list[dict[str, object]] = []
    pending_tables: dict[str, list[pd.DataFrame]] = {}
    written_tables: set[str] = set()
    written_row_counts: dict[str, int] = {}
    written_frame_counts: dict[str, int] = {}
    processed_archives = 0
    archives_without_mdb = 0
    archives_with_only_empty_tables = 0
    extracted_mdb_count = 0
    source_table_count = 0
    nonempty_source_table_count = 0
    parsed_row_count = 0
    flush_count = 0
    last_flush_at = monotonic()
    next_table_log_at = max(1, int(log_every_tables or LOG_EVERY_N_TABLES_READ))

    def _write_run_metadata() -> Path:
        return _write_preprocess_metadata(
            root_dir,
            raw_dir=raw_dir,
            started_at_iso=started_at_iso,
            elapsed_minutes=(monotonic() - started_at) / 60,
            total_input_archives=len(files),
            single_station=single_station,
            requested_source_tables=requested_source_tables,
            tables_to_read=tables_to_read,
            worker_count=worker_count,
            preprocess_backend=preprocess_backend,
            log_every_tables=max(1, int(log_every_tables or LOG_EVERY_N_TABLES_READ)),
            processed_archives=processed_archives,
            archives_without_mdb=archives_without_mdb,
            archives_with_only_empty_tables=archives_with_only_empty_tables,
            extracted_mdb_count=extracted_mdb_count,
            source_table_count=source_table_count,
            nonempty_source_table_count=nonempty_source_table_count,
            parsed_row_count=parsed_row_count,
            flush_count=flush_count,
            written_row_counts=written_row_counts,
            written_frame_counts=written_frame_counts,
        )

    def _flush_pending_tables(reason: str) -> None:
        nonlocal flush_count, last_flush_at
        pending_frame_count = sum(len(frames) for frames in pending_tables.values())
        if pending_frame_count == 0:
            return

        flush_count += 1
        pending_table_count = len(pending_tables)
        logger.info(
            "Flushing batch %s to DuckDB (%s): %s frame(s) across %s table(s).",
            flush_count,
            reason,
            pending_frame_count,
            pending_table_count,
        )
        flush_started_at = monotonic()
        for table_name, frames in tqdm(
            list(pending_tables.items()),
            total=pending_table_count,
            desc=f"Flushing batch {flush_count}",
        ):
            row_count, frame_count = _append_pending_frame(
                root_dir,
                table_name,
                frames,
                written_tables,
            )
            written_row_counts[table_name] = written_row_counts.get(table_name, 0) + row_count
            written_frame_counts[table_name] = written_frame_counts.get(table_name, 0) + frame_count
            frames.clear()

        pending_tables.clear()
        last_flush_at = monotonic()
        logger.info(
            "Finished DuckDB batch %s in %.1f min; %s total output table(s), %s total row(s) written so far.",
            flush_count,
            (last_flush_at - flush_started_at) / 60,
            len(written_tables),
            sum(written_row_counts.values()),
        )

    def _log_archive_progress() -> None:
        elapsed_minutes = (monotonic() - started_at) / 60
        logger.info(
            "Parsed %s/%s archive(s) in %.1f min: %s MDB file(s), %s source table(s), %s non-empty table(s), %s row(s), %s output table(s), %.1f min since last flush.",
            processed_archives,
            len(files),
            elapsed_minutes,
            extracted_mdb_count,
            source_table_count,
            nonempty_source_table_count,
            parsed_row_count,
            len(written_tables.union(pending_tables.keys())),
            (monotonic() - last_flush_at) / 60,
        )

    def _log_table_progress() -> None:
        elapsed_seconds = max(monotonic() - started_at, 0.001)
        elapsed_minutes = elapsed_seconds / 60
        logger.info(
            "Read %s source table(s) from %s archive(s) in %.1f min: %s non-empty table(s), %s row(s), %.1f table(s)/sec.",
            source_table_count,
            processed_archives,
            elapsed_minutes,
            nonempty_source_table_count,
            parsed_row_count,
            source_table_count / elapsed_seconds,
        )

    def _record_archive_result(result: dict[str, object]) -> None:
        nonlocal processed_archives
        nonlocal archives_without_mdb
        nonlocal archives_with_only_empty_tables
        nonlocal extracted_mdb_count
        nonlocal source_table_count
        nonlocal nonempty_source_table_count
        nonlocal parsed_row_count
        nonlocal next_table_log_at

        processed_archives += 1
        archive_name = str(result["archive_name"])
        if result["without_mdb"]:
            archives_without_mdb += 1
            logger.warning("Skipping archive without MDB members: %s", archive_name)
            archive_metadata.loc[archive_metadata["filename"] == archive_name, "is_empty_archive"] = 1
        else:
            extracted_mdb_count += int(result["mdb_members"])
            member_records = result["archive_records"]
            archive_table_records.extend(member_records)
            source_table_count += len(member_records)
            nonempty_source_table_count += sum(
                1 for record in member_records if not record["is_empty_table"]
            )
            parsed_row_count += sum(int(record["row_count"]) for record in member_records)

            if not result["has_nonempty_table"]:
                archives_with_only_empty_tables += 1
                logger.debug("Archive %s contains only empty MDB tables.", archive_name)
                archive_metadata.loc[archive_metadata["filename"] == archive_name, "is_empty_archive"] = 1

            for table_name, frames in result["parsed_tables"].items():
                pending_tables.setdefault(table_name, []).extend(frames)

        if source_table_count >= next_table_log_at:
            _log_table_progress()
            while source_table_count >= next_table_log_at:
                next_table_log_at += max(1, int(log_every_tables or LOG_EVERY_N_TABLES_READ))
        if processed_archives % LOG_EVERY_N_ARCHIVES == 0 or processed_archives == len(files):
            _log_archive_progress()
        if processed_archives % FLUSH_EVERY_N_ARCHIVES == 0:
            _flush_pending_tables(f"{processed_archives} archive(s) parsed")

    async def _read_archives() -> None:
        loop = asyncio.get_running_loop()
        archive_records = files.loc[:, ["filename", "station_code"]].to_dict("records")
        batch_size = max(worker_count * 4, 1)

        with executor_class(max_workers=worker_count) as executor:
            progress = tqdm(total=len(archive_records), desc="Reading MDB archives")
            try:
                for batch_start in range(0, len(archive_records), batch_size):
                    batch = archive_records[batch_start : batch_start + batch_size]
                    futures = [
                        loop.run_in_executor(
                            executor,
                            read_archive_payload,
                            str(raw_dir / record["filename"]),
                            record["filename"],
                            str(record["station_code"]),
                            table_map,
                            column_map,
                            tables_to_read,
                            str(extract_root),
                        )
                        for record in batch
                    ]
                    for future in asyncio.as_completed(futures):
                        result = await future
                        _record_archive_result(result)
                        progress.update(1)
            finally:
                progress.close()

    asyncio.run(_read_archives())
    _flush_pending_tables("final parse batch")

    logger.info(
        "Finished parsing archives: %s processed, %s without MDB files, %s with only empty tables, %s MDB file(s), %s source table(s), %s non-empty source table(s), %s parsed row(s), %s output table(s).",
        processed_archives,
        archives_without_mdb,
        archives_with_only_empty_tables,
        extracted_mdb_count,
        source_table_count,
        nonempty_source_table_count,
        parsed_row_count,
        len(written_tables),
    )
    logger.info("Writing archive metadata to DuckDB table %s.", RAW_ARCHIVES_TABLE)
    write_dataframe_table(root_dir, RAW_ARCHIVES_TABLE, archive_metadata)
    logger.info(
        "Writing %s archive table record(s) to DuckDB table %s.",
        len(archive_table_records),
        SENSOR_ARCHIVE_FILES_TABLE,
    )
    write_dataframe_table(
        root_dir,
        SENSOR_ARCHIVE_FILES_TABLE,
        pd.DataFrame(archive_table_records, columns=ARCHIVE_TABLE_RECORD_COLUMNS),
    )

    if not written_tables:
        logger.info("All processed MDB archives were empty. No data tables were written.")
        metadata_path = _write_run_metadata()
        logger.info("Wrote preprocess metadata to %s.", metadata_path)
        return {}

    logger.info(
        "Finished writing %s parsed output table(s) to DuckDB across %s batch(es).",
        len(written_tables),
        flush_count,
    )
    for table_name, row_count in sorted(written_row_counts.items()):
        logger.info(
            "Output table %s: %s row(s) from %s source frame(s).",
            table_name,
            row_count,
            written_frame_counts.get(table_name, 0),
        )

    logger.info(
        "Water-quality preprocess finished in %.1f min.",
        (monotonic() - started_at) / 60,
    )
    metadata_path = _write_run_metadata()
    logger.info("Wrote preprocess metadata to %s.", metadata_path)
    return written_row_counts
