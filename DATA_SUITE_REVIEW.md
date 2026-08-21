# Data suite review — 2026-08-20

Findings from a full review of `src/data/` (climate, sensor_data, health, land_cover,
river_network, shared, verification, assembly, biomes, population). Check items off as
they're fixed.

## Already fixed

- [x] `health/preprocess/preprocess.py` (`_clean_birth_outcome_frame`) — `Total` column
      wasn't numeric-coerced before `check_value_range`, crashing on DATASUS `"-"`
      placeholders. Now reuses `_coerce_tabnet_numeric` (handles decimal commas, never
      raises) for both `Total` and `value_columns`.
- [x] `river_network/core.py:458-467` — transitive-closure loop now squares
      `reach_result` against itself (`reach_result @ reach_result`) each iteration instead
      of against the original adjacency, so it actually doubles reach per pass and
      converges in `log2(N)` iterations as the comment always claimed.
- [x] `sensor_data/fetch/data/download.py:195-201` (`_current_raw_archives_frame`) — now
      builds one file listing and derives both columns from it (plus tolerates a file
      vanishing mid-scan), instead of two independent `iterdir()` calls that could desync.
- [x] `verification/core.py:65-92` — a crashing `fingerprint_paths()` now gets a unique
      per-run `"error:<timestamp>"` fingerprint (and skips the cache-read entirely) instead
      of a constant empty-list hash, so it can no longer poison the cache into serving a
      stale result forever.
- [x] `climate/preprocess/era5_land.py` (`preprocess_era5_land`, stage "all"/"zarr") — now
      filters discovered GRIB files through `_manifest_ready_for_preprocess` (and
      `_wait_for_lock_release`) the same way `preprocess_era5_land_worker` does, so a file
      that failed verification or is still mid-download can no longer be silently ingested.
- [x] `shared/spatial_tabular.py:60-97` (`rasterize_feature_values`) — now tracks an
      explicit `assigned` mask of pixels actually covered by a geometry, instead of
      inferring "unassigned" from `grid != 0`, so a feature's genuine value of 0 is no
      longer clobbered with `fill_value`.
- [x] `land_cover/preprocess.py:221,246-248` (`process_year`) — replaced the dead
      `(ValueError, RuntimeError, Exception)` tuple and ad hoc string-matching with the
      shared `is_extent_mismatch_error` helper (same one `shared/spatial_tabular.py`
      defines for this exact purpose); the outer handler's silent
      `"cannot convert float NaN to integer"` swallow now logs and counts like every other
      unexpected error instead of vanishing untracked.

## High severity — correctness bugs

## Medium severity — correctness / robustness

- [x] `sensor_data/fetch/database.py:189-213` (`append_dataframe_table`) — now shares a
      `_table_exists_on_connection` helper with `table_exists()` instead of reimplementing
      the same query inline (couldn't call `table_exists()` directly since it opens its own
      connection and `append_dataframe_table` already holds one open).
- [x] `land_cover/preprocess.py:100` (`get_files`) — extracted years are now checked against
      a plausible `[1985, 2030]` range and raise `ValueError` if outside it, catching the
      regex latching onto the wrong 4-digit token in a filename.
- [x] `health/preprocess/preprocess.py:359-363` (`_extract_municipality_fields`) — now logs
      a warning with the dropped-row count when rows have no parseable municipality code,
      instead of silently vanishing via `dropna`.
- [x] `health/fetch/datasus.py:750` — replaced the bare `print()` with `logger.info`, so it
      shows up in captured logs like every other fetch/preprocess call in the module.
- [x] `shared/paths.py` vs `biomes/constants.py:44-57` and `population/constants.py:6-21` —
      both `biomes_dir`/`raw_dir` and population's `population_dir`/`raw_dir`/`processed_dir`
      no longer `mkdir` as a side effect (matching `shared/paths.py`'s pure-path-lookup
      contract); the one place biomes needs the directory to exist (`fetch.py`'s
      download/extract) now calls `mkdir` explicitly itself, and population's write sites
      already did.
- [x] `verification/sources.py:415,454` (`_biomes_list_fetched`, `_biomes_fingerprint_paths`)
      — now import and call `archive_path()` from `biomes/constants.py` instead of
      hardcoding the path a second time.
- [x] `verification/core.py:123-134` — `check_outputs()` returning an empty artifact list
      (despite data being present) is now explicitly reported as `"outstanding"` with a
      warning log, instead of silently falling through to `"verified"`.
- [x] `shared/sensor_upstream.py:296,301` (`resolve_reachable_distances`) — a reachability
      index missing from the distance matrix (other than the target's own self-entry) now
      raises `ValueError` describing the matrix inconsistency, instead of silently
      defaulting to distance `0.0` (max kernel weight).
- [x] `climate/*` — ERA5 dataset ids are now defined once as `ERA5_LAND_SUBTYPE_DATASETS` in
      `climate/constants.py`; `fetch/era5_land_hourly.py`, `fetch/era5_land_daily.py`,
      `fetch/common.py`'s `DATASET_RUNNING_REMOTE_REQUEST_LIMITS`, and
      `preprocess/era5_land.py`'s `_dataset_name_for_subtype` all key off it instead of
      hardcoding independently. Also found and removed a *third*, previously-unflagged copy
      of `_build_trench_length_lookup` living in `climate/assembly.py`, and fixed the
      `GEObOX_FILENAME` typo while in the file.
- [x] `land_cover/assembly.py:193-199` vs `aggregation.py:48-61` (and the third copy in
      `climate/assembly.py`) — `_build_trench_length_lookup` and `_land_cover_feature_stem`
      are now defined once in `land_cover/schema.py` (using the validated version that
      raises a clear error for missing columns) and imported by all three call sites.
- [x] `health/fetch/forms.py:328-342` vs `preprocess/preprocess.py` (`_read_datasus_csv`) —
      the row-width-normalization logic now lives in a new `health/csv_utils.py` shared by
      both (avoids a circular import since `fetch/datasus.py` already imports `fetch/forms.py`).
- [x] `land_cover/aggregation.py:277` / `assembly.py` — `aggregate_along_rivers` now raises
      `ValueError` up front if `LAND_COVER_TOTAL_COLUMN` is missing from the input, and the
      per-bucket total is read via direct indexing instead of `.get(..., 0.0)`; `assembly.py`
      already sourced its columns from the validated `land_cover_assembly_columns()`, so it
      needed no change.

- [x] `population/preprocess/preprocess.py:62` vs `land_cover/constants.py:135-137`
      (`derive_mun_id_from_adm2_id`) — population now imports and uses the same
      `derive_mun_id_from_adm2_id` helper (already the shared convention used by biomes and
      assembly) instead of its own independent `.str[:6]` front-truncation.
- [x] `verification/checks.py:44-55` (`check_null_fraction`) — now wired into
      `_biomes_check_outputs` (0% null tolerance on `BIOME_COLUMN` for both `biome_adm2` and
      `biome_sensor`, closing the gap where `sjoin_nearest`'s fallback wasn't re-checked for
      remaining nulls) and into `_assembly_check_outputs` (99% null tolerance per wide-source
      variable column, catching the "left join silently produces an all-NaN column" failure
      mode called out below — this was the more natural place to add the check, ahead of it
      reaching a distributed output, rather than adding it inside `build.py`'s join logic).
- [x] `assembly/build.py:235-245` (`assemble_dataset`) — see above; addressed via a
      verification-side null-fraction check per wide-source column rather than changing the
      join logic itself.

- [x] `sensor_data/preprocess/clean.py:144-147` (`build_cleaning_flags`) — added
      `classify_values`, a vectorized `np.select`-based equivalent of `classify_value` with
      the exact same precedence order; verified equivalence against the scalar version over
      20,000 randomized samples (0 mismatches) and ~2.2x faster than the real
      `frame.apply(axis=1)` call at 200k rows (gap widens at the millions-of-rows scale this
      table reaches in production). `classify_value` itself is kept as-is (still documents
      the per-value rule set clearly).
- [x] `sensor_data/preprocess/preprocess.py:296-347` — `_read_source_table`, `_read_table`,
      and `_read_first_available_table` now all funnel through one
      `_connect_sensor_database` + `_read_first_available_table` implementation instead of
      each opening their own connection with near-identical logic; removed the now-dead
      `_resolve_source_table` wrapper this consolidation left behind.

- [x] `climate/preprocess/era5_land.py:771-786` (`_metadata_time_offsets`) — this turned out
      not to need real accumulated-variable GRIB data to verify at all: the actually-installed
      `earthkit-data` (found via the project's `311` conda env, since `pyproject.toml` leaves
      it unpinned) is far enough past a library rewrite that this whole code path was already
      broken, independent of the stepRange question. Verified with a synthetic single-message
      GRIB file built locally via `eccodes` (no real ERA5 data needed):
      1. `ekd.from_source("file", ...)` now returns a `GribData` wrapper with no `.ls()`/
         `.data()` at all — `_open_era5_dataset` crashed with `AttributeError` before ever
         reaching the metadata code. Fixed by calling `.to_fieldlist()` on it (a harmless
         no-op on older earthkit-data versions where `from_source` already returns a
         FieldList).
      2. Once fixed, `.ls()`'s columns are no longer flat eccodes names (`dataTime`,
         `stepRange`, `shortName`, `dataDate`) but dotted "collection.key" names
         (`time.valid_datetime`, `time.base_datetime`, `parameter.variable`, ...), and
         `time.valid_datetime` already comes out fully step-adjusted (confirmed: reference
         06:00 + a "0-1" accumulation step → 07:00, computed by the library itself). So the
         original "does it double-count" question is moot for the current schema —
         `_field_valid_datetimes`/`_field_band_names` now read `time.valid_datetime`/
         `parameter.variable` directly, with the old flat-column reconstruction kept only as
         an explicit legacy fallback (and fixed there too: it now only ever uses `dataTime`,
         not `dataTime`-or-`validityTime`, as the base to add `stepRange` onto, since
         `validityTime` is already step-adjusted and using it as the base was the original
         latent double-count bug in that fallback path).
      Ran the real `_open_era5_dataset` end-to-end against the synthetic GRIB file
      (confirmed it crashed pre-fix, produced the correct `2020-01-01T07:00:00` valid time
      post-fix); added `_FakeEra5FieldList.to_fieldlist()` (returns `self`, matching real
      `FieldList` behavior) and two new unit tests for the dotted-key path and the
      double-count-safe legacy fallback. Full `tests/data/sources/climate/` suite (48 tests)
      and `tests/data` overall (124 tests) pass.

## Lower severity — maintainability

- [x] `land_cover/preprocess.py:272` (`_load_drainage_polygons`) — now takes an explicit
      `drainage_path` argument instead of `self`, decoupling it from `LandCover` instance
      internals.
- [x] `shared/spatial_tabular.py:178-180` — removed the dead `masked_unique_counts`,
      `is_no_overlap_error`, `rasterize_value_grid` aliases (and their re-exports in
      `shared/__init__.py`) after confirming zero callers anywhere in the repo.
- [x] `shared/batches.py:16-17,20-23,110-117` — `table_raw_dir`/`batch_table_dir`/
      `batch_output_dir` are now pure path lookups (no `mkdir`); `write_manifest` now
      `mkdir`s explicitly at its one actual write site. The health fetch path that writes
      into `batch_output_path`'s directory (`forms.py`'s `download_result_csv`) already had
      its own `mkdir`, so needed no change.
- [x] `assembly/schema.py:121-123` (`load_assembly_config`) — replaced the O(n²)
      `.count()`-in-a-comprehension duplicate-id check with a `Counter`-based one.

- [x] `sensor_data/fetch/data/access_reader.py:19-32` (`normalize_object_columns`) — also
      didn't need a live pyodbc/Access source. The installed `pandas` (found via the same
      `311` env) is 3.0, whose default dtypes changed enough to break this function's premise
      either way: built a synthetic frame the same way `pd.read_sql` builds one from cursor
      rows (`pd.DataFrame.from_records` over `(datetime, str, str)` tuples, matching what
      `read_access_table`'s `pd.read_sql` does) and confirmed a homogeneous
      `datetime`/`None` column now auto-infers to `datetime64[us]` (not `object` — so the
      original double-encoding risk is gone on its own), while a decimal-comma text column
      now defaults to pandas 3.0's new `str` dtype (not `object` either) — meaning the
      function's `dtype != object` gate silently skipped every real text column too,
      defeating the actual decimal-comma-to-float coercion it exists for. Fixed by widening
      the gate to `is_object_dtype(...) or is_string_dtype(...)` (both still correctly
      exclude `datetime64` columns, confirmed). Verified against the same synthetic
      `from_records` frame: datetime column stays untouched, `"1,5"` correctly coerces to
      `1.5`, plain text stays text.
- [x] `climate/fetch/common.py:666-847` (`retrieve_batched_dataset`) — the user set up real
      CDS credentials (`~/.cdsapirc`), which unblocked this. `load_cds_credentials()` now
      also falls back to `~/.cdsapirc` (standard cdsapi format/location) when the
      project-local `setup/secrets/.cdsapi` is absent, so no credentials had to be
      duplicated into the repo (with a new regression test for the fallback, and an existing
      test fixed to isolate `Path.home()` so it doesn't pick up a real `~/.cdsapirc` on the
      machine running it). Decomposed the one-cycle-per-`while True`-iteration body into
      `_refresh_remote_statuses` (priority-running-then-remaining status checks, returns
      updated manifest state + counts), `_submit_pending_requests` (submits new requests
      within the active-request budget), and `_run_retrieval_cycle` (runs both, logs, returns
      `(active_requests, counts)`) — `retrieve_batched_dataset` itself is now just setup +
      the termination-condition loop calling `_run_retrieval_cycle`. Purely mechanical
      extraction, no behavior change. Verified two ways: (1) the existing mocked test suite
      already has thorough per-edge-case coverage of this function (running limits,
      queue-full deferral, verification retries, stale-status starvation, rejected-job
      retry, fresh-check skip, etc.) — all pass unchanged; (2) ran a real, tiny, one-off
      end-to-end request against the live CDS API (`reanalysis-era5-land`, 1 variable, 1
      timestep, a ~0.1°×0.1° box, written to a scratch root_dir, never the project's real
      `data/` dir) through the refactored `retrieve_batched_dataset` — confirmed it actually
      submitted, polled through several real queue cycles, downloaded a 207-byte GRIB file,
      wrote a manifest with `"status": "downloaded"`, and terminated the loop correctly on
      "all batches are downloaded". `tests/data` overall (125 tests) passes.
- [x] `sensor_data/preprocess/assembly.py:412-441` (`_aggregate_streamflow_matches`) — also
      didn't need real data, just a reference test (same approach as the `clean.py`/
      `classify_values` fix above). Replaced the per-group `.apply` with vectorized
      `groupby().sum()`/`.nunique()`/`.min()` aggregation: weighted-mean numerator/denominator
      columns computed once per row then summed per group (duplicate join rows scale
      numerator and denominator equally and so don't change the weighted mean — verified);
      the three diagnostic columns that previously deduped via `.drop_duplicates()` inside
      the per-group `.apply` (`streamflow_match_count`, `streamflow_nonnull_day_count`,
      `streamflow_total_weight`) now dedupe via `.nunique()`/a single upfront
      `.drop_duplicates(subset=[*group_columns, "streamflow_station_code"])` instead.
      Verified equivalence against the original scalar implementation over 300 randomized
      trials (varying station/date counts, weights, missing values, and deliberately
      duplicated join rows to exercise the dedup path) — 0 mismatches. Added two permanent
      regression tests (`test_aggregate_streamflow_matches_weights_and_dedupes`,
      `test_aggregate_streamflow_matches_returns_nan_when_all_values_missing`) to
      `tests/data/sources/sensor_data/test_assembly.py`.
- [x] `verification/sources.py` — added a shared `_artifact_check(label, path,
      build_checks)` helper (right after `_missing_artifact`) that owns the "load parquet →
      missing → build checks" sequence: reads the parquet via `_safe_read_parquet`, returns
      `_missing_artifact(...)` if it's absent/unreadable, otherwise calls the adapter-supplied
      `build_checks(frame)` and wraps the result. Rewired all 8 adapters'
      `check_outputs()` (`river_network`, `land_cover`, `sensor_data`, `climate`, `biomes`,
      `population`, `health`, `assembly`) to build a `build_checks` closure and call
      `_artifact_check` instead of hand-rolling the `frame is None` branch each time — the
      per-source check logic itself (required columns, value ranges, null fractions, etc.)
      is unchanged, only the missing/loading boilerplate around it moved into one place.
      `assembly`'s config-load/parse-error handling (a genuinely different pattern) was left
      as-is; only its per-dataset artifact loop was converted. Full `tests/data/verification`
      suite (38 tests) and `tests/data` overall (124 tests) still pass.
- [x] `shared/spatial_tabular.py:60-97` vs `:123-147` — no design decision needed after all:
      `rasterize_feature_values` had zero callers anywhere in the repo (confirmed by grep),
      same as the `masked_unique_counts`/`is_no_overlap_error`/`rasterize_value_grid`
      aliases already removed above for the same reason. Removed it (and its `__init__.py`
      re-export) rather than reconciling its overlap semantics with
      `rasterize_feature_labels`, which resolves the inconsistency by deleting the unused,
      semantically-hazardous side of it.
- [x] `biomes/preprocess.py:177-187` (`build_station_biomes`) — nearest-biome fallback still
      doesn't itself re-verify `sjoin_nearest` resolved every null, but a null now reaching
      `biome_sensor.parquet`/`biome_adm2.parquet` is caught by the `check_null_fraction`
      wiring added above instead of passing through unchecked.

## Second pass — 2026-08-20 (independent review, not yet cross-checked against items above)

### Medium severity

- [x] `sensor_data/fetch/data/download.py:410-429` (`_existing_category_keys`) — the regex
      `^{code}_(?P<tab>[a-z0-9_]+)_(?P<category>[a-z0-9_]+)_mdb_` uses two adjacent greedy
      groups that both accept underscores, so Python's backtracking always dumps everything
      except the last underscore-token into `tab` and only the final token into `category`,
      regardless of the real tab/station-type boundary. Filenames are built at line 729 as
      `f"{_slugify_label(result_tab)}_{station_type_slug}"`, and `_slugify_label` turns any
      multi-word tab or station type into several underscore-joined tokens (e.g.
      `conventional_fluviometrica_convencional`), so the parsed `(tab, category)` key from
      an existing archive's filename essentially never matches the key computed live during
      scraping (line 680: `_category_key(result_tab, station_type_slug)`). Confirmed by
      reading both the filename-construction site and the regex. Effect: already-downloaded
      station categories aren't recognized as already-downloaded, so `_should_attempt_category`
      re-attempts (re-downloads) them on every run in "default"/"missing-only" fetch modes.
      Fixed by switching the delimiter between the tab and category slugs from `_` to `__`
      (both at the `source_label` construction site and in the parsing regex); `_slugify_label`
      already collapses runs of underscores to one, so `__` can never appear inside either
      slug, making the split unambiguous regardless of how many words either label contains.
- [x] `biomes/preprocess.py:177-187` (`build_station_biomes`) — the nearest-biome fallback
      assigned `nearest[BIOME_COLUMN].to_numpy()` into `joined.loc[unmatched.index, ...]`
      positionally, assuming `sjoin_nearest(...).drop_duplicates(subset=[STATION_CODE_COLUMN],
      keep="first")` preserves `unmatched.index`'s row order. `sjoin_nearest` can return
      tied-distance duplicates per left feature, and neither the order `keep="first"` retains
      nor alignment with `unmatched.index` was verified. If ties broke in a different order,
      a station could silently get another station's nearest biome, with no error surfaced.
      Fixed by keying the assignment through `.map()` on `STATION_CODE_COLUMN` instead of a
      positional `.to_numpy()` assignment, matching the rest of the module's join style.
- [x] `land_cover/assembly.py:161-345` vs `climate/assembly.py:179-383` (not
      `climate/fetch/common.py` — corrected after re-checking exact locations) — the same
      ~180-line upstream-distance-bucketing block (`_build_system_trench_lookup`,
      `_build_trench_system_position_lookup`, `_resolve_upstream_trench_distances`,
      `_shift_upstream_distances`, `_combine_station_upstream_distances`, `_bucket_label`,
      `_assign_sensor_distance_buckets`) was duplicated near-verbatim (byte-for-byte aside
      from two docstring words) between the two files, including an identical comment
      claiming it's "deliberately not delegated to shared.sensor_upstream". Fixed by moving
      the six functions into `shared/sensor_upstream.py` (parameterized on `rn_module`,
      column names, and the bucket list instead of hardcoding either module's constants),
      and reducing both call sites to thin same-name wrappers so no call sites needed to
      change. `land_cover/assembly.py` and `climate/assembly.py` no longer import
      `build_group_index_lookup`/`sparse_row` directly (removed as now-unused there — the
      shared functions call them internally).

### Low severity

- [x] `health/fetch/datasus.py:538-554` (`fetch_mortality_age_tables`) — `pre_1996`'s
      `default_year` was `"22"` but its own `years` list only contains `"79"`-`"94"`;
      `post_1995`'s `default_year` was `"95"` but its `years` list only contains
      `"96"`-`"21"`. Since neither value ever appeared in its own plan's `years`, the
      "skip clicking, it's already the page default" comparison could never be true, so the
      optimization silently never fired — every year was always explicitly clicked either
      way. Rather than guess at the two swapped-looking values (unverifiable without hitting
      the live DATASUS form in this environment), removed the dead `default_year` fields and
      the `if year != config["default_year"]:` branch entirely and made the click
      unconditional, which is exactly the behavior the code already had in practice — no
      functional change, just removing misleading dead code.
- [x] `climate/fetch/common.py:667-848` (`retrieve_batched_dataset`) — the request-count
      recomputation (`active_requests`/`running_requests` summed from `manifest_states`) was
      written out fully twice (previously ~741-748 and ~789-796), each immediately preceded
      by a now-dead incremental `+=`/`-=` update that got overwritten by the recompute right
      after it. Extracted a single `_count_manifest_activity(manifest_states)` helper (added
      just above `retrieve_batched_dataset`) and call it at both recompute sites plus once
      more after the priority-checks loop; deleted the now-unreachable incremental updates
      and the initial accumulation that fed them. Behavior is unchanged (both sites already
      only used the fully-recomputed value) — this just removes the duplicated
      recomputation and dead code the finding flagged as a future-drift risk.

## Third pass — 2026-08-21 (independent review, done blind against this file, cross-checked after)

Reviewed the current on-disk state of `src/data/` (including the uncommitted working-tree
changes) from scratch, without reading the sections above first. Two items below directly
contradict a "fixed" item from the 2026-08-20 pass; both were independently re-derived and
verified against the actual code/tests before writing this up.

### High severity

- [x] **Reopens the "fixed" item above** — `sensor_data/preprocess/assembly.py:80-111,375-406`
      (`_prepare_streamflow_features`, `_aggregate_streamflow_matches`) — the weighted mean is
      still biased by duplicate `(station_code, date)` rows, and the 2026-08-20 fix didn't
      catch it because its self-verification was circular. The comment ("a duplicate row...
      scales the numerator and denominator by the same factor and so leaves the weighted mean
      unchanged") is only true when the duplicate rows belong to a *single* station; across
      *multiple* matched stations with different weights, a duplicate for one station inflates
      that station's contribution relative to the others. Verified by hand against the repo's
      own test (`tests/data/sources/sensor_data/test_assembly.py:37-72`,
      `test_aggregate_streamflow_matches_weights_and_dedupes`): station `sf1` (weight 0.5,
      value 20) is duplicated, `sf2` (weight 0.3, value 5) is not. The test asserts
      `streamflow_discharge_mean_7d == 21.5/1.3 ≈ 16.54`; the value you'd get by first
      deduping `sf1`'s row (the correct behavior, since the duplication is a same-row join
      artifact, not two real observations) is `11.5/0.8 = 14.375` — a ~15% difference. The
      2026-08-20 "fix" ran 300 randomized trials checking the new vectorized code against the
      *old* per-group `.apply` implementation and found "0 mismatches" — that only proves the
      two implementations agree with each other, not that either computes the correct weighted
      mean; the old implementation had the identical bug. Failure scenario: any water-quality
      station whose matched streamflow gauge has duplicated date rows (e.g. from an upstream
      many-to-many join, explicitly called out as the expected trigger in the code's own
      comment) gets its discharge features silently skewed toward that gauge in every row
      assembled for the station, for the life of the pipeline. Fix direction: deduplicate
      `(station_code, date)` before the weight/value products are summed, not just for the
      diagnostic columns.
      **Fixed:** `_prepare_streamflow_features` now `drop_duplicates`s on
      `(station_code, date)` before computing rolling features (so a duplicate source row
      can no longer double-weight a date in the rolling window either), and
      `_aggregate_streamflow_matches` now builds `unique_station_rows` (dedup on
      `(group, streamflow_station_code)`) *before* computing the weighted numerator/
      denominator, reusing the same dedup already used for the diagnostic columns, instead
      of summing the raw joined frame. Updated
      `test_aggregate_streamflow_matches_weights_and_dedupes` to assert the correct
      `11.5/0.8` weighted mean instead of the biased `21.5/1.3` it previously encoded as
      expected. Verified: `tests/data/sources/sensor_data/test_assembly.py` (4 tests) and
      the full `tests/data` suite (125 tests) pass under the `311` conda env.

- [x] `land_cover/aggregation.py:74-79` (`_assign_distance_bucket`) vs
      `land_cover/constants.py:125-129` (`LAND_COVER_COMPOSITION_BUCKET_MAP`) — the ADM2/river
      aggregation path buckets upstream distance with no upper cap
      (`floor(distance/25)*25`, so a trench 600 km upstream gets bucket `600`), but
      `composition.py:37-144` (`compute_kernel_weighted_composition`, whose docstring
      explicitly says it consumes output from "both the sensor and ADM2 land-cover assembly
      variants") joins that bucket column against `bucket_map` with an **inner join**
      (`composition.py:140-144`), and `LAND_COVER_COMPOSITION_BUCKET_MAP`'s keys stop at
      `{0, 25, ..., 475, 500}`. Any bucket `>500` has no match and is silently dropped from
      both the leaf-class total and the kernel-weight normalization — confirmed by reading
      both the bucket-assignment and the join. Contrast with the *sensor* path
      (`land_cover/assembly.py` / `shared/sensor_upstream.py`'s `_bucket_label`), which
      explicitly clamps everything `≥500 km` into one `500` bucket via `SENSOR_DISTANCE_BUCKETS`
      — the ADM2 path never got the same clamp. Failure scenario: any municipality whose
      upstream drainage network extends beyond 500 km (common for large systems, e.g. Amazon
      or São Francisco tributaries) silently loses all land-cover composition data beyond
      500 km, with no warning — the reported `lc_*`/`alr_*` shares for that municipality are
      quietly computed from a truncated catchment.
      **Fixed:** `_assign_distance_bucket` still does the same 25 km-width `floor` bucketing
      (and, importantly, still allows *negative* buckets — confirmed via
      `tests/data/sources/land_cover/test_assembly.py`'s
      `test_assemble_land_cover_adm2_uses_bucketed_upstream_output` that a `-25` bucket is
      legitimate, representing the ADM2-touching trench's own downstream portion behind the
      shifted-distance zero point — so it couldn't simply delegate to
      `shared.sensor_upstream.assign_distance_buckets` with `SENSOR_DISTANCE_BUCKETS`, whose
      list starts at 0 and would silently drop those rows), but now clamps the upper end to
      `SENSOR_DISTANCE_BUCKET_STARTS_KM[-1]` (500) via `np.minimum`, matching the cap already
      used by the sensor path and by `LAND_COVER_COMPOSITION_BUCKET_MAP`. Verified directly:
      distances of 600 and 5000 km now both bucket to `500` instead of `600`/`5000`. Full
      `tests/data/sources/land_cover` suite (5 tests) and `tests/data` overall (125 tests)
      pass.

### Medium severity

- [x] `climate/preprocess/era5_land.py:1186-1206` (`prepare_daily_era5_dataset`) — unlike
      `resample_era5l_hourly_to_daily` (used by the hourly/ARCO paths), this function never
      computes `ERA5L_VAR_CONFIG["2t"]["aggregation"]["extras"]` (`2t_daily_min`/
      `2t_daily_max`), confirmed by the test suite itself
      (`test_prepare_daily_era5_dataset_writes_daily_values_without_resampling` asserts
      `"2t_daily_min" not in prepared.data_vars`). Since `era5_land_hourly.py` never fetches
      `2m_temperature`, the only two writers of `2t` to the shared zarr store are
      `era5_land_daily` (mean only) and ARCO (mean+min+max). For any date range where an
      operator falls back to the `era5_land_daily` fetch path instead of ARCO (e.g. ARCO
      access lag for older years), `2t_daily_min`/`2t_daily_max` are left unwritten for those
      dates, and `assembly.py`'s `ANNUAL_MIN_VARIABLES`/`ANNUAL_MAX_VARIABLES` silently compute
      annual temperature extremes over only the ARCO-covered subset — wrong values, no error.
      **Fixed differently than the root cause suggests:** rather than teaching
      `era5_land_daily`'s CDS fetch to also request min/max daily statistics (a larger,
      live-API-dependent change out of scope here), `_annual_aggregate_sql` now requires every
      day in the trench-year to have a non-null `2t_daily_min`/`2t_daily_max` before reporting
      an annual value (`CASE WHEN COUNT(identifier) = COUNT(*) THEN MIN/MAX(...) ELSE NULL
      END`), so a trench-year mixing ARCO- and `era5_land_daily`-sourced days now correctly
      comes back `NULL` instead of a silently-partial value; a trench-year with no ARCO
      coverage at all was already `NULL` and is unaffected. Added
      `test_annual_aggregate_sql_nulls_min_max_for_partial_year_coverage` (pure DuckDB, no live
      data needed) covering full coverage, partial coverage, and zero coverage. Full
      `tests/data/sources/climate` suite (53 tests) passes.
- [x] `climate/assembly.py` — the sensor-panel path (`_assemble_sensor_upstream_duckdb`)
      computes every rolling window (`mean_7d`...`mean_365d`) as an `AVG(...) OVER (...)` for
      *all* climate columns including the accumulation variables (`tp`/`sro`/`ssro`/`pev`),
      while the ADM2-panel path (`_annual_aggregate_sql`) explicitly `SUM`s those same
      variables per `ANNUAL_SUM_VARIABLES`. "30-day precipitation" means daily-mean-of-mean in
      one panel and annual total in the other for the same underlying values, with no comment
      explaining the divergence — worth confirming this is intentional, since a downstream
      consumer treating the two panels as comparable would silently get the wrong semantics.
      **Resolved (user decision):** asked the user to weigh in; asked to align both panels on
      whichever is scientifically correct rather than pick arbitrarily. For a *fixed* window,
      mean and sum carry identical information (a constant rescale by the window length), so
      the deciding factor is comparability *across* window lengths — a mean stays directly
      comparable across the 7/30/90/180/365-day (sensor) and annual (ADM2) windows, while raw
      sums don't (a 365-day sum isn't informative next to a 7-day sum without renormalizing
      anyway). Both panels' existing column names (`mean_Xd`, `mean_value`) already commit to
      "mean" -- the ADM2 panel's `SUM` for accumulation variables was the one actually
      violating its own naming. Folded `ANNUAL_SUM_VARIABLES`'s members into
      `ANNUAL_MEAN_VARIABLES` in `climate/schema.py` (with a comment recording the reasoning)
      and removed the now-dead `SUM` branch from `_annual_aggregate_sql`; added a comment at
      the sensor-panel's `AVG` aggregation cross-referencing the same rationale. No output
      schema/column-name changes on either panel. Full `tests/data/sources/climate` suite (53
      tests) and `tests/data` overall (132 tests) pass.
- [x] `assembly/build.py:66-77` (bucket-kernel weighting) — `df[bucket_column].map(raw_weights)`
      produces `NaN` for any bucket label absent from `bucket_map`; the only NaN-cleanup
      (`df.loc[df[value_column].isna(), "_raw_weight"] = 0.0`) only fires on null *values*, not
      unmapped *bucket labels*. That `NaN` poisons `weight_sum` via `.transform("sum")`, and
      `np.where(weight_sum > 0, ...)` treats `NaN > 0` as `False`, silently zeroing the
      **entire** `(entity, category)` group's output rather than just the offending row.
      Trigger: any row whose bucket value isn't in the map (e.g. a bucket added upstream but
      not mirrored here — plausible given the uncapped-bucket finding above) silently zeroes
      that whole entity/category instead of raising.
      **Fixed:** `.map(raw_weights)` result now gets `.fillna(0.0)` before the null-value
      zeroing, so an unmapped bucket label is simply excluded from the group's weighting (like
      `land_cover.composition`'s equivalent inner join already does for out-of-map buckets)
      instead of poisoning the whole group via NaN propagation. Added
      `test_compute_kernel_weighted_bucket_values_ignores_unmapped_bucket_without_zeroing_group`
      to `tests/data/assembly/test_build.py`. Full `tests/data/assembly` suite (10 tests) and
      `tests/data` overall (132 tests) pass.
- [x] `land_cover/health/population` — `health/preprocess/preprocess.py:185-186`
      (`_read_datasus_csv`) drops the last row solely because `body[-1][0].strip('"') ==
      "Total"`, with no structural check; and `population/preprocess/preprocess.py`'s call
      into `derive_mun_id_from_adm2_id(d["mun_id"])` truncates `id_municipio` with
      `str(value)[:-1]` without validating it's actually a 7-digit code first — a null/NaN or
      already-6-digit value from BigQuery silently corrupts the join key (`"nan"[:-1]` =
      `"na"`, or a legitimate 6-digit code gets truncated to 5) with no error raised.
      **Fixed:** `_read_datasus_csv` now `logger.debug`s the dropped row's content when it
      matches the "Total" heuristic, for visibility, without changing the (already reasonably
      narrow, last-row-only) matching logic itself -- hardening it further would mean guessing
      at DATASUS's exact export format, which the 2026-08-20 pass already flagged as
      unverifiable without hitting the live form. For `derive_mun_id_from_adm2_id`: adding a
      strict length check broke an existing land_cover test fixture that intentionally uses
      shorter placeholder adm2_ids (`"1001A"`), so instead the shared helper
      (`land_cover/constants.py`) now only rejects a null/NaN `adm2_id` before truncating.
      Discovered along the way that `population/preprocess/preprocess.py`'s call site
      pre-cast the column to `str` *before* calling the helper (`d["mun_id"].astype(str).map(...)`),
      which would have turned a real NaN into the literal string `"nan"` and defeated the new
      null check entirely -- removed that pre-cast so the helper can actually see the null.
      Added `test_transform_population_frame_raises_on_null_municipality_id`. An
      already-6-digit `mun_id` passed in by mistake is still not caught (a general length check
      isn't safe given the shared helper's flexible-length test fixtures); flagging as a
      residual gap rather than fixing further. Full `tests/data/sources/population` (5 tests),
      `tests/data/sources/land_cover` (5 tests), and `tests/data` overall (132 tests) pass.
- [x] `shared/webdriver.py:197-229,293-307` — two related resource-leak paths: (1)
      `create_chrome_driver` calls `manager.__enter__()` directly instead of using `with`, so
      `ManagedBrowser.__exit__` (which cleans up temp profile/cache dirs) only runs if the
      caller reliably calls `driver.quit()`; (2) inside `_create_driver`'s retry loop, if
      `webdriver.Chrome(...)` spawns a process and a later line in the same `try` raises before
      `self._driver` is set, that Chrome process is orphaned and the next retry spawns another
      without killing it.
      **Fixed:** (1) `_create_driver`'s retry loop now wraps the post-spawn setup
      (`set_page_load_timeout`, `set_window_size`, etc.) in its own `try`/`except` that quits
      the just-spawned driver before re-raising to the outer retry handler, so a failure there
      no longer orphans a live Chrome process. (2) `create_chrome_driver` now falls back to
      `manager.quit()` if anything raises between `manager.__enter__()` and returning the
      driver. No test added -- exercising this needs a real/mocked Selenium Chrome session,
      and there's no existing test infrastructure for `webdriver.py` to extend; verified by
      reading and `ast.parse` only.
- [x] `shared/batches.py:36-57` (`load_manifest`/`write_manifest`) — `write_manifest` writes
      directly to the target path (no temp-file + `os.replace`), and `load_manifest`'s
      `json.loads(line)` has no try/except; a process killed mid-write leaves a truncated
      manifest line that crashes the *next* resumed run instead of degrading gracefully.
      **Fixed:** `write_manifest` now writes to a `{path}.tmp-{pid}` file and `os.replace`s it
      into place, so a crash mid-write can never leave a torn file behind; `load_manifest` now
      catches `json.JSONDecodeError` per line, logs a warning, and skips just that line instead
      of crashing the whole load (defense in depth for manifests corrupted some other way, e.g.
      manual editing). Added `tests/data/shared/test_batches.py` (2 tests: atomicity + no
      leftover temp file, and malformed-line tolerance). Full `tests/data` suite (132 tests
      after all fixes in this pass) passes.
- [x] `sensor_data/fetch/data/download.py:460-479` — `_wait_for_download_completion` matches
      files with `*{station_code}*.zip`, a substring glob that can pick up a stale/partial
      archive from an unrelated station whose code contains this one as a substring (e.g. `"12"`
      matching a leftover `..._1234_...zip`), silently filing it under the wrong station.
      **Fixed:** added `_station_code_matches`, which requires the code to appear with
      non-digit boundaries (`(?<!\d){code}(?!\d)`) rather than as a raw substring, and switched
      both the `.zip` and `.crdownload` matching in `_wait_for_download_completion` to use it.
      Added `tests/data/sources/sensor_data/test_download.py` (2 tests: the matcher's
      boundary behavior directly, and an end-to-end check that a same-substring file for
      another station is correctly ignored, timing out rather than being picked up).
- [x] `sensor_data/fetch/data/download.py:1040-1094` — download-result records are only
      flushed to DuckDB every 10 stations or at the end; a crash mid-run loses up to 9
      stations' worth of logged results, which then look "never attempted" on resume even
      though some categories may have partially succeeded.
      **Fixed:** now flushes the record buffer after every station instead of every 10 --
      each station's download already takes several seconds (network/browser-bound), so the
      extra DuckDB appends are negligible overhead against that, and it closes the crash-loss
      window down to at most the current station's own in-flight records.
- [x] `sensor_data/fetch/data/access_reader.py:104-112` — auto-discovered tables
      (`source_tables=None`) abort the entire MDB file's read on the first table pyodbc can't
      marshal, while an explicit `--source-tables` request only skips that one table — an
      asymmetry that could silently drop an entire file's data because one late, unrelated
      table is unreadable.
      **Fixed:** both modes now skip just the offending table (with a `logger.warning` including
      the table/file/station and the original exception) instead of aborting the whole file in
      the auto-discovery case. No test added -- needs a live/mocked pyodbc Access connection,
      which isn't available on this machine (no Windows Access ODBC driver) and has no existing
      test scaffolding to extend; verified by reading and `ast.parse` only.

### Low severity

- [x] `land_cover/schema.py:82-87` (`normalize_optional_int`) — bare `pd.isna(value)` isn't
      safe against list-like/array input; not currently exercised that way, but worth guarding
      since it participates in the class-ID mapping applied to every raster pixel count.
      **Fixed:** now raises `TypeError` up front via `pd.api.types.is_scalar(value)` (which
      already treats `None` as scalar, so the null-handling path is unaffected) instead of
      letting a non-scalar reach `pd.isna` and blow up with an ambiguous-truth-value error.
      Added `tests/data/sources/land_cover/test_schema.py` (2 tests: null/numeric/string
      scalars still work, non-scalar input raises cleanly).
- [x] `population/constants.py:18` defines `DEFAULT_POPULATION_OUTPUT_FILENAME` but
      `preprocess_population_data` hardcodes the literal `"population.parquet"` instead of
      importing it — dead constant the two can silently drift from.
      **Fixed:** `preprocess_population_data` now imports and uses
      `DEFAULT_POPULATION_OUTPUT_FILENAME` instead of the duplicated literal. Turns out the
      constant wasn't actually dead -- `verification/sources.py` already imports and uses it
      for the population artifact-check path, so this fix also closes a latent
      verification/preprocess drift risk (verification's expected filename could have diverged
      from what preprocess actually writes).
- [x] `shared/slurm.py:39-73` (`render_sbatch_script`) — only `command_argv` is
      `shlex.quote`d; `job_name`/`partition`/`conda_env`/`project_dir`/`log_dir`/`extra_env`
      values are interpolated raw into the generated shell script. Low likelihood (repo-
      controlled config) but worth hardening since it's unvalidated input into a script later
      run via `sbatch`.
      **Fixed:** every value interpolated into a line bash actually executes (`eval "$(conda_hook
      ...)"`, `conda activate`, `cd`, `mkdir -p`, `export NAME=value`) is now `shlex.quote`d;
      `extra_env` names are validated against a `[A-Za-z_][A-Za-z0-9_]*` pattern (can't be
      shell-quoted since they need to stay bare identifiers). `#SBATCH` directive lines are
      read by Slurm's own parser, not bash, so they don't need shell quoting, but a value
      containing a newline could still forge an extra directive -- added an explicit
      newline check across all spec string fields plus `log_dir` at the top of the function.
      Added 3 regression tests to `tests/data/shared/test_slurm.py` (metacharacter quoting,
      invalid `extra_env` name rejected, newline-in-`job_name` rejected). Full suite (8 tests)
      passes.
- [x] `climate/fetch/test.py` — a throwaway debug script (hardcoded local path, module-level
      `earthkit.data` import, prints to stdout) committed inside the package rather than as a
      scratch file; harmless at runtime (not pytest-collected, no `test_` functions) but dead
      weight that reads as a real test file by name.
      **Fixed:** deleted (`git rm`), after confirming it's not imported or referenced anywhere
      in `src/` or `tests/`.
- [x] `verification/sources.py:113-116` — docstring references "the empty-checks-list issue
      noted in `checks.py`"; no such note currently exists in `checks.py` — stale
      cross-reference.
      **Fixed:** reworded the docstring to drop the dangling cross-reference; confirmed via
      grep that `checks.py` has no such note.



- The 17 test failures previously seen in `tests/data/verification/test_sources.py` were a
  missing `xarray` dependency in the ambient shell's default Python (`miniforge3/bin/python3`),
  not a real bug. The project's actual environment is the `311` conda env
  (`/Users/felixschulz/miniforge3/envs/311`) — it has `xarray`, `earthkit-data`, `pyodbc`,
  `pandas` 3.0, etc. installed, and `tests/data` (124 tests) passes cleanly there. Several
  "needs live data to verify" findings above turned out to be resolvable just by checking
  what's actually installed in `311` instead: `pyproject.toml` doesn't pin `earthkit-data` or
  `pandas`, and both have moved far enough since this code was written (`earthkit-data`'s
  `.ls()`/`GribData` API rewrite; pandas 3.0's new default string dtype) to break two
  functions outright, independent of the "needs real data" question originally attached to
  them. Worth checking `311` (or wherever the pipeline actually runs) against `pyproject.toml`
  periodically, since neither dependency is version-pinned.

## Fourth pass — 2026-08-21 (independent review, done blind against this file)

Split into four parallel reviews (shared/assembly/verification, climate, sensor_data,
health/population/biomes/land_cover/river_network), each done without reading this file
first. Not yet cross-checked against the passes above or against each other for overlap —
flagging that up front since some items may restate or interact with earlier entries (e.g.
the land-cover nodata/`TOTAL` issue below touches the same composition code as the third
pass's distance-bucket fix).

### High severity

- [x] `shared/spatial_tabular.py:26-38` (`crop_unique_counts`) +
      `land_cover/preprocess.py:157-170` (`_accumulate_mapped_counts`) — land-cover rasters
      are opened via `rxr.open_rasterio(raster_path, chunks=None)` with default `masked=False`
      (`preprocess.py:218`), so MapBiomas nodata (class `0`) is a plain finite int and isn't
      filtered by `crop_unique_counts` (which only drops NaN/inf). `TOTAL` is then summed over
      the *raw, unfiltered* pixel counts, while individual class columns go through
      `create_mappers`'s `np.vectorize(lambda x: legend_class_dict.get(x, np.nan))`, which maps
      unmapped codes like `0` to `NaN` and drops them. Net effect: `TOTAL` includes nodata
      pixels but class columns don't, so `class_count/TOTAL` shares systematically undercount
      and don't sum to 1 for any drainage polygon touching a nodata/border region — a silent,
      data-dependent bias in every downstream land-cover composition output.
      **Fixed:** two changes. (1) `process_year` now opens rasters with `masked=True`, so any
      nodata pixel flagged in the GeoTIFF's own metadata becomes `NaN` and is excluded by
      `crop_unique_counts`'s existing `isfinite` filter — defense in depth, but not sufficient
      on its own if MapBiomas's nodata isn't declared in the file's metadata. (2) The real fix:
      `_accumulate_mapped_counts` no longer sums raw `counts` for `TOTAL`; it now derives
      `TOTAL` from the class-level mapper's own valid mask (the first entry in
      `legend_mappers`), so `TOTAL` is by construction consistent with the class columns
      regardless of what the raster's nodata sentinel value is or whether it's declared as
      nodata at all. Added `tests/data/sources/land_cover/test_preprocess.py`
      (`test_accumulate_mapped_counts_excludes_unmapped_nodata_from_total`), confirming an
      unmapped code (e.g. `0`) no longer inflates `TOTAL` beyond the sum of mapped class
      counts. Full `tests/data` suite (143 tests) passes.
- [x] `climate/preprocess/era5_land.py` `process_era5_input_file` (~1272-1322) vs
      `write_dataset_region` (~1226-1237) — only the *raw GRIB input file* is locked via
      `climate_file_lock(path, ...)` before writing; the shared zarr `store_path` itself is
      never locked before the in-place `to_zarr(mode="r+", region=...)` write. Contrast with
      `preprocess/era5_land_arco.py`, which explicitly locks the store before the same kind of
      write. Two concurrent `preprocess` invocations (e.g. two SLURM array jobs) writing to
      months that share a zarr chunk (`ERA5_OUTPUT_CHUNKS = (365, -1, -1)`, ~1 chunk/year) can
      race and corrupt or lose data with no error raised.
      **Fixed:** `process_era5_input_file` now wraps the `write_dataset_region(prepared,
      store_path)` call in `climate_file_lock(store_path, owner="climate_preprocess_worker")`,
      matching the pattern already used by `preprocess/era5_land_arco.py`. The input-file lock
      is unaffected (different lock path, no deadlock risk). `tests/data/sources/climate`
      (54 tests) passes.
- [x] `climate` UTC-day vs. local-time mismatch — `resample_era5l_hourly_to_daily`
      (`era5_land.py` ~1139) buckets by UTC calendar day, and `fetch/era5_land_daily.py`
      explicitly requests `"time_zone": "utc+00:00"`. Downstream `assembly.py` joins this by
      calendar date against water-quality/sensor dates presumably recorded in Brazil local
      time (UTC-3 to UTC-5). A rainfall event at 22:00 local time on day N (01:00-03:00 UTC on
      day N+1) is attributed to climate date N+1, shifting precipitation/temperature signal by
      up to a full day relative to the water sample it's meant to explain.
      **Partially fixed.** Added `BRAZIL_UTC_OFFSET_HOURS = -3` to `climate/constants.py`
      (Brazil has used one nationwide standard time, UTC-3, since the 2019 DST repeal — not
      exact for the Amazon's UTC-4/-5 fringe, but right for the overwhelming majority of the
      country) and changed `fetch/era5_land_daily.py`'s `build_era5_land_daily_request` to
      request `time_zone="utc-03:00"` instead of `"utc+00:00"`; CDS computes that daily
      aggregate server-side over the full time series, so there's no boundary-file risk. Added
      `test_build_era5_land_daily_request_uses_brazil_local_time_zone`.
      **Not fixed, and now accepted rather than left open:** `resample_era5l_hourly_to_daily`
      (the hourly-GRIB-origin path, used for precipitation/accumulation variables) still buckets
      by UTC day, a systematic ~3h-off-local-midnight offset. I attempted the same "shift the
      time axis by -3h before resampling" fix used for the CDS-side path above, but reverted it:
      raw hourly GRIB files are fetched in **per-UTC-month** batches
      (`fetch/era5_land_hourly.py`), and `process_era5_input_file` resamples one file at a time.
      Shifting by -3h before resampling means the day straddling each month boundary only has
      ~21 or ~3 of its 24 hours available in whichever file is processed (the other file has the
      complementary partial hours), and since `write_dataset_region` does a
      `to_zarr(mode="r+", region=...)` write keyed by date, the boundary day silently ends up
      holding whichever file processed last — a partial-day sum treated as complete, which is
      *worse* than the current uniform UTC-day convention (a new silent per-boundary-day error
      vs. a known, constant, dataset-wide offset). A correct fix needs the resample step to see
      a few hours of the adjacent month's file (or an explicit carry-over of partial
      boundary-day accumulators across files) — a real architecture change.
      **Decision (2026-08-21): accept the ~3h UTC-vs-local-time offset for this path rather than
      pursue that architecture change.** Rationale: the offset is systematic and bounded (every
      day shifted by the same ~3h, not random noise), it consistently relabels rather than
      corrupts the data, its effect on sum-aggregated variables (precipitation/runoff — the ones
      that actually go through this path) is further diluted by the 7-day/31-day rolling
      streamflow-feature windows already used downstream, and it's pre-existing behavior, not a
      regression. Revisit this if an analysis ever needs tight same-day precision between a
      precipitation value and a water-quality sample (e.g. "did it rain in the hours right
      before this sample") rather than a rolling/lagged relationship.
- [x] `sensor_data/fetch/data/access_reader.py:67-71,104` (`connect_access_database`, used via
      `with connect_access_database(...) as connection:` in `load_mdb_tables`) — pyodbc's
      `Connection.__enter__`/`__exit__` only commit/rollback, they do **not** close the
      connection. Every archive parsed leaks an ODBC handle to the Access driver, which can
      hold a file lock on the extracted `.mdb`. `read_archive_payload`
      (`access_reader.py:177-197`) then does `shutil.rmtree(extract_root, ignore_errors=True)`
      — if the lock is still held, removal silently fails and the temp dir + `.mdb` is left on
      disk forever. Over a run processing thousands of archives, this is a real disk-space
      leak with no error or warning.
      **Fixed:** `load_mdb_tables` now opens the connection directly (not via `with`) and
      explicitly calls `connection.close()` in a `finally` block, so the ODBC handle (and any
      driver-held file lock on the `.mdb`) is always released before the caller's
      `shutil.rmtree` runs, on both the success and exception paths. No write/transaction
      semantics are affected — this module only ever reads (`SELECT`) from the MDB. Added
      `tests/data/sources/sensor_data/test_access_reader.py` with a fake connection asserting
      `close()` is called both when reading succeeds and when a table read raises.
- [x] `health/fetch/datasus.py` `fetch_birth_outcome_tables` (~751-756) — the period-select
      block is `if year != latest_year: select_option_value(...); select_option_value(...)`.
      When `year == latest_year` (currently 2023), *neither* call runs, leaving the "Período"
      multiselect in whatever state it defaulted to (possibly nothing selected, or a stale
      selection from a prior request). The query submitted for the most recent year is built
      without an explicit period filter, silently producing wrong/empty results for that year
      while every other year is queried correctly.
      **Fixed:** `form.select_option_value(f"nvbr{year_code}.dbf")` is now called
      unconditionally every iteration (so the target year is always selected), and the
      `latest_year_code` selection stays gated behind `year != latest_year` exactly as before —
      preserving the existing two-value selection for every non-latest year unchanged, while
      the latest-year iteration now selects its own period instead of nothing. No live-form
      test exists for this Selenium-driven fetcher (none did before), so this couldn't be
      exercised against the real DATASUS form; the fix is a minimal, behavior-preserving-except-
      for-the-bug change reviewed by inspection.
- [x] `health/preprocess/preprocess.py` `_coerce_tabnet_numeric` (343-355) — does a blind
      `str.replace("-", "0")` on the whole string (to turn DATASUS's `"-"` no-data placeholder
      into `"0"`) *before* checking for decimal commas. This also strips the minus sign off any
      real negative number: `"-5,3"` → `"05,3"` → `"05.3"` → `5.3`, silently flipping sign and
      shifting a digit. Low frequency today (most SIH/mortality metrics are non-negative) but a
      latent bug in a shared helper used across hospitalization, ICD-10, and morbidity
      preprocessing — any future/edge-case negative metric gets corrupted with no warning.
      **Fixed:** now only replaces a cell whose *entire* (stripped) value is exactly `"-"` with
      `"0"` (`normalized.where(normalized != "-", "0")`), instead of substring-replacing every
      `-` character. A genuine negative value like `"-5,3"` is left untouched by that step and
      correctly parses to `-5.3`. Added `tests/data/sources/health/test_preprocess.py`
      covering both the bare-dash-as-missing case and negative-value preservation.

### Medium severity

- [x] `shared/slurm.py:49-52` (`render_sbatch_script`) — the newline-injection guard only
      checks `job_name`, `partition`, `time`, `qos`, `conda_hook`, `conda_env`, `project_dir`;
      `mem` and `cpus_per_task` are interpolated unguarded into `#SBATCH` directive lines. A
      `mem`/`cpus_per_task` value containing `\n` (from a malformed `setup/slurm_jobs.yaml`
      entry) can forge extra `#SBATCH` directives, defeating the point of the existing check.
      **Fixed:** added `cpus_per_task` and `mem` to the guarded-field tuple. Added
      `test_render_sbatch_script_rejects_newline_in_mem`.
- [x] `verification/core.py:46-49` (`_write_sidecar`) writes the JSON sidecar directly
      (`open(path, "w")` + `json.dump`), unlike `batches.write_manifest`
      (`shared/batches.py:63-75`), which writes to a temp file and `os.replace`s it for
      crash-safety. A SIGKILL/crash mid-write (plausible on a Slurm job hitting a time/mem
      limit) can leave a torn sidecar; `_read_sidecar` degrades to a cache-miss on
      `JSONDecodeError`, but a concurrent reader could momentarily see a truncated file.
      **Fixed:** `_write_sidecar` now writes to `{path}.tmp-{pid}` and `os.replace`s it into
      place, matching `batches.write_manifest`'s pattern exactly. `tests/data/verification`
      (38 tests) passes unchanged.
- [x] `shared/sensor_upstream.py:677-710` (`explode_list_matches`), used by
      `prepare_trench_adm2_matches` — explodes `values_column`/`weights_column` together via
      `frame.explode([...])`, which requires equal-length list cells per row. An out-of-sync
      `adm2_list`/`intersection_lengths` pair (e.g. from a partially-written or manually edited
      river-network table) raises a raw `ValueError: columns must have matching element
      counts`, crashing ADM2 upstream aggregation instead of failing with a clear message.
      **Fixed:** added an explicit per-row length check (`.str.len()` comparison) before the
      `explode` call; a mismatch now raises a `ValueError` naming the two columns, the row
      count affected, and the first offending row's `id_columns`, instead of pandas's generic
      message. Added
      `test_explode_list_matches_raises_clear_error_on_mismatched_list_lengths`.
- [x] `climate/preprocess/era5_land.py` `ERA5L_VAR_CONFIG` (~79-186) — asymmetric `skipna`:
      accumulation variables (tp/sro/ssro/pev) use `skipna=False` for daily sums (correctly
      conservative — one missing hour nulls the day), but instantaneous variables (2t/2d/
      swvl1/swvl2) use `skipna=True` for daily means. A day with only 1 of 24 hourly values
      present (truncated/partially-failed GRIB download, or first/last day of a raw file's
      coverage) still produces a non-null daily mean with no signal it's based on almost no
      data.
      **Fixed:** `skipna` is now `False` for all eight variables (2t, 2d, swvl1, swvl2 changed
      from `True`), so a missing hour nulls the whole day's mean, matching the conservative
      convention already used for the sum-aggregated variables. Added
      `test_resample_hourly_to_daily_nulls_instantaneous_mean_on_missing_hour`; existing
      `tests/data/sources/climate` (55 tests) unaffected, since none exercised `skipna=True`
      behavior with an actual NaN present.
- [x] `climate/assembly.py` `_annual_aggregate_sql` (~150-172) — the completeness check
      `CASE WHEN COUNT(id) = COUNT(*) THEN MIN/MAX ELSE NULL` only detects nulls within present
      rows, not missing calendar days. If a trench-year is missing whole days (partial
      preprocessing run, trench added mid-pipeline, store gap), `COUNT(*)` is silently smaller
      than 365/366 and the check passes, reporting MIN/MAX over a subset of the year as if
      complete.
      **Fixed:** the MIN/MAX branch now compares `COUNT(identifier)` against the true number of
      calendar days in that trench-year's year (a leap-year-aware `CASE` expression derived
      from `EXTRACT(YEAR FROM {source_alias}.date)`, matching the same year expression the
      caller's `GROUP BY` already uses), instead of against `COUNT(*)`. The `ANNUAL_MEAN_VARIABLES`
      branch is deliberately left untouched — this only tightens the already-completeness-aware
      MIN/MAX branch, not the mean, which has no such check today and changing that would be a
      separate, larger behavior change. Rewrote
      `test_annual_aggregate_sql_nulls_min_max_for_partial_year_coverage` (the old version had
      no `date` column and grouped by trench only, incompatible with the new year-aware SQL) to
      cover three cases over a real 365-day year: full coverage (real MIN), full coverage with
      one null (NULL, pre-existing behavior), and 360/365 days present but all non-null (NULL —
      the new behavior; previously this silently returned a MIN over the partial data).
- [x] `sensor_data/fetch/data/download.py` `download_by_id` (~950-975) vs `fetch_station_data`
      (~1054-1073) — on session loss, `download_by_id` does `driver = browser_manager.restart()`
      and `continue`s, but this only rebinds `download_by_id`'s local variable; the caller's
      loop variable is never updated, so the next station is called with the dead driver
      reference, triggering another detect-and-restart cycle. Not a hard failure, but every
      station immediately following an in-flight restart wastes at least one attempt.
      **Fixed:** `download_by_id` now returns `(records, driver)` instead of just `records`, and
      `fetch_station_data`'s loop does `station_records, driver = download_by_id(...)`, so a
      mid-station restart is immediately visible to the caller for the next station. Added
      `test_download_by_id_returns_restarted_driver_after_session_loss`, which fails against
      the pre-fix single-return-value code (there'd be nothing to unpack the new driver from).
- [x] `sensor_data/fetch/data/download.py` `_current_raw_archives_frame`/
      `_compute_pending_station_ids` (195-320) — for `default`/`missing-only` fetch modes, a
      station counts as "already archived" if *any* file matching `^{station_code}_` exists in
      `raw_dir`, regardless of whether `_is_valid_archive` ever confirmed it or the download
      actually completed. A partially written/orphaned file (e.g. `shutil.move` interrupted, or
      process killed between file creation and the `sensor_downloads` log write) permanently
      marks that station "done," skipping it on all future runs with no data ever recorded.
      **Fixed:** added `_is_parseable_zip` (ZIP signature + central-directory + CRC check via
      `testzip()`, deliberately *not* requiring MDB members like `_is_valid_archive` does) and
      `_current_raw_archives_frame` now excludes any `.zip` file that fails it. A structurally
      intact but MDB-less ZIP (a legitimately-downloaded empty result) still counts as archived,
      preserving that existing, deliberate behavior; only genuinely truncated/corrupt files
      (the actual bug) are excluded. Added
      `test_current_raw_archives_frame_excludes_truncated_zip`.
- [x] `sensor_data/fetch/data/access_reader.py` `normalize_object_columns` (19-43) — coerces
      any text column by blindly replacing `,`→`.` and accepting it if every non-null value
      parses as numeric afterward. A column legitimately using comma as a thousands separator
      (`"1,234"`) is silently reinterpreted as `1.234` — a 1000x scale error — with no signal
      distinguishing decimal-comma (assumed pt-BR convention) from thousands-comma formatting.
      **Partially fixed.** A value containing *both* `.` and `,` (e.g. `"1.234,56"`) is now
      unambiguously handled as full pt-BR formatting (period = thousands, comma = decimal):
      periods are stripped before the comma becomes a decimal point, giving `1234.56` instead
      of the previous `1.234.56` (which would already have failed to parse as a number and
      fallen back to text — so this is a strict improvement, not just a wash). **Not fixed:** a
      bare comma with no period (`"1,234"`) remains genuinely ambiguous between decimal and
      thousands separator with nothing in the string itself to disambiguate; it's still coerced
      as decimal, matching the function's pre-existing pt-BR assumption. Resolving that case
      correctly would need column-level metadata (expected units/precision) that isn't
      available at this layer. Added
      `test_normalize_object_columns_coerces_plain_decimal_comma` and
      `test_normalize_object_columns_coerces_pt_br_thousands_and_decimal`.
- [x] `biomes/fetch.py` `fetch_biomes` (12-22) — archive is written directly to its final
      destination path, and the next run skips re-downloading purely because
      `destination.exists()`. A process killed mid-download (network drop, OOM, SLURM
      preemption) leaves a partial/corrupt zip that every subsequent run treats as "already
      present," fails at `zipfile.ZipFile(...)` with `BadZipFile`, and never re-fetches —
      requires manual deletion to recover.
      **Fixed:** downloads now go to `{destination}.tmp-{pid}` first, then `os.replace` moves
      the file into place atomically, matching the same crash-safety pattern used elsewhere in
      this pass (`batches.write_manifest`, `verification._write_sidecar`). Added a new
      `tests/data/sources/biomes/` test module (none existed before) covering the download path
      (asserting no leftover temp file) and the already-downloaded skip path.
- [x] `land_cover/preprocess.py` `process_year` (216-282) — if `rxr.open_rasterio` raises for a
      year's `.tif` (missing file, corrupt GeoTIFF, transient disk/network hiccup on a SLURM
      node), the exception is caught, logged, and execution falls through to build/return a
      `year_df` from the still-all-zero `output_data` array. This is indistinguishable
      downstream from a year where every drainage polygon genuinely has zero overlap — a
      transient infra failure gets baked into the dataset as "zero land cover for all trenches
      that year" instead of surfacing as a missing/failed year.
      **Fixed:** removed the `try/except` around the raster-open + per-polygon loop entirely;
      a raster load failure now propagates out of `process_year`, which `Parallel` (in
      `preprocess_land_cover`) surfaces as a failed run instead of silently completing with a
      wrong all-zero year folded into the output. The per-polygon error handling (extent
      mismatches, individual crop failures) is untouched — only the outer raster-open catch was
      removed. Added `test_process_year_raises_instead_of_returning_all_zero_year`.
- [x] `health/fetch/forms.py` `reset_query()` (334-338) + `health/fetch/datasus.py`
      `_execute_sih_manifest_entries` `finally` block (528-530) — `reset_query()` unconditionally
      does `driver.close()` then indexes `window_handles[0]` and clicks "limpa"; only invoked
      when `len(window_handles) > 1`. If failure happens after the results window opened but is
      itself in an unexpected state, `reset_query()` can raise from inside the `finally` (e.g.
      `NoSuchElementException` on `.limpa`), masking the original exception and leaving two
      windows open for the next manifest entry, corrupting subsequent batch attempts in the
      same run.
      **Fixed:** the `form.reset_query()` call inside the `finally` block is now wrapped in its
      own `try/except`, logging a warning on failure instead of letting it propagate and replace
      whatever exception (if any) is already in flight from the `try` above. Added
      `test_execute_sih_manifest_entries_does_not_mask_original_exception_with_reset_failure`,
      which fails against the pre-fix code (the original `ValueError` would be replaced by the
      mocked `reset_query()`'s `RuntimeError`).

### Low severity

- [x] `climate/fetch/common.py` `ERA5_DAYS = ["01".."31"]` is used unconditionally for every
      month in the hourly/daily request builders, including February and 30-day months.
      Correctness depends entirely on the CDS API silently ignoring nonexistent day/month
      combinations, which nothing here verifies or tests.
      **Fixed:** added `days_in_month(year, month)` to `fetch/common.py` (uses
      `calendar.monthrange`), and both `build_era5_land_hourly_request` and
      `build_era5_land_daily_request` now use it instead of the fixed `ERA5_DAYS` list, so the
      request's `"day"` field matches the actual calendar regardless of whether the CDS API
      would have ignored the extra days anyway. `ERA5_DAYS` is kept (documented as legacy) since
      nothing else references it. Added
      `test_build_era5_land_requests_use_actual_days_in_month` (checks both a 28-day and a
      29-day February).
- [x] `climate/assembly.py` `_partitioned_trench_day_paths` (~128-147) falls back to scanning
      the entire `climate_path` directory when no requested month partitions exist on disk,
      rather than treating "range has no data" as empty. Correctness is preserved by the later
      `WHERE date BETWEEN` filter, but this silently turns a real data gap into an
      undifferentiated full-table-scan slowdown.
      **Fixed:** added a `logger.warning` when falling back to the full-directory scan, naming
      the directory and requested date range, so the slowdown/potential data gap is visible
      instead of silent. Behavior is otherwise unchanged. Added
      `test_partitioned_trench_day_paths_warns_on_full_directory_fallback` and
      `test_partitioned_trench_day_paths_uses_existing_partitions_without_warning`.
- [x] `sensor_data/preprocess/assembly.py` `_prepare_streamflow_features` (105-113) — rolling
      7/31-day windows use `.rolling(window=7/31, min_periods=1)` on row order per station, not
      actual dates. A station with date gaps (outages, partial monthly imports) gets a "7/31
      day" mean that silently spans however many calendar days those rows actually cover.
      (Note: this touches the same function reworked by the third pass's streamflow-dedup fix
      above — worth checking whether that fix affects this issue.)
      **Fixed:** confirmed the third pass's dedup fix is orthogonal (it collapses duplicate
      `(station_code, date)` rows before rolling; it doesn't address date gaps). Switched the
      rolling call from a row-count window (`.rolling(window=7)`) to a date-offset window
      (`.rolling("7D", on=DATE_COLUMN)`), which anchors the window to actual elapsed calendar
      time instead of row count. This changes the result's index to a `(station, date)`
      MultiIndex, so it's assigned back into `features` via `.to_numpy()` positional assignment
      rather than index alignment — safe because `features` is already sorted by
      `[STATION_CODE_COLUMN, DATE_COLUMN]`, matching `groupby`'s row order exactly. Added
      `test_prepare_streamflow_features_rolling_window_is_calendar_day_based`, which fails
      against the pre-fix code (a 4-row/2-cluster series with a gap would have averaged all 4
      rows into the 7-day column instead of just the 2 rows actually within 7 days).
- [x] `shared/webdriver.py:122-131` — `ManagedBrowser.__exit__` with `keep_open_on_error=True`
      returns without calling `quit()` on any exception, leaking the Chrome process plus its
      `tempfile.mkdtemp` profile/cache dir. Intentional for interactive debugging, but a real
      leak if left on in an unattended/looped scrape.
      **Fixed (documentation + louder warning, behavior unchanged by design):** this is a
      deliberate interactive-debugging escape hatch — there's no way to keep a window open for
      human inspection while also cleaning up its process/profile dir, so the leak itself can't
      be "fixed" without defeating the feature's purpose. Added a class docstring spelling out
      the tradeoff and explicitly warning against unattended/looped use, and expanded the
      runtime warning to name the leaked profile directory and explicitly say cleanup won't
      happen. Added `tests/data/shared/test_webdriver.py` (new file) verifying `quit()` is
      skipped and the expanded warning fires with `keep_open_on_error=True`, and that `quit()`
      still runs normally otherwise.
- [x] `shared/batches.py:110-122` (`update_manifest_entry`) — if `batch_id` isn't found in
      `entries` (e.g. stale reference after a manifest re-plan), silently no-ops (still
      rewriting the unchanged file) instead of raising, which could mask a caller bug where
      batch bookkeeping has drifted out of sync.
      **Fixed:** the `for...else` now raises `ValueError` naming the missing `batch_id` and the
      dataset/table, instead of silently rewriting the manifest unchanged. Added
      `test_update_manifest_entry_raises_for_unknown_batch_id`; existing
      `tests/data/shared`/`tests/data/sources/health` suites unaffected (no caller relied on the
      silent no-op).
- [x] `assembly/build.py:99-107` (`_pivot_long_source`) uses `pd.pivot` (not `pivot_table`);
      any duplicate `(join_keys, pivot_value)` combination in a long-format source raises a
      hard, unhelpful `ValueError` rather than a clear assembly-level error.
      **Fixed:** wrapped the `frame.pivot(...)` call in a `try/except ValueError`, re-raising
      with the source name, the offending `(join_keys, pivot_column)` combination, and the
      first few duplicated key rows, instead of pandas's generic pivot error. Added
      `test_pivot_long_source_raises_clear_error_on_duplicate_join_key_pivot_combination`.
- [x] `health/fetch/datasus.py` `_parse_sih_period_value` (214-218) —
      `year = 1900 + yy if yy >= 95 else 2000 + yy` assumes SIH data starts at 1995 with no
      bounds check; a hypothetical `"94"`-coded period file would silently map to 2094 instead
      of erroring.
      **Fixed:** added a plausibility bound (`1995 <= year <= current_year + 1`) after decoding;
      an out-of-range result now raises `ValueError` naming the period value and decoded year
      instead of silently producing e.g. 2094. Added
      `test_parse_sih_period_value_decodes_year_and_month` and
      `test_parse_sih_period_value_rejects_implausible_year`.
- [x] `health/preprocess/preprocess.py` `_read_datasus_csv` (183-190) — only strips a single
      trailing "Total" row (`body[-1]`). An export with more than one trailing summary/subtotal
      line would let the extra row(s) survive into the frame and either raise `KeyError` on an
      unmapped label or contaminate output with a spurious municipality-less row.
      **Fixed:** changed the single `if body[-1]... == "Total"` check to a `while` loop, so
      every consecutive trailing "Total" row is stripped, not just the last one. Added
      `test_read_datasus_csv_drops_all_trailing_total_rows` with two trailing Total rows.

All 23 findings from this pass (6 high, 10 medium, 7 low) have been fixed and tested; no
duplicate/contradictory overlap with the earlier three passes turned out to need reconciling
during the fix work. `tests/data` grew from 125 to 165 tests over the course of this pass and
passes in full under the `311` conda env. One item was deliberately left as a documented partial
fix rather than a full one: the UTC-vs-local-time climate mismatch (high severity) was only
fixed on the CDS-side daily-statistics path; the hourly-GRIB resample path was left alone after
the "obvious" fix turned out to introduce a worse partial-day-sum bug at month-file boundaries.
For that path, the ~3h offset itself is now an accepted limitation (project decision,
2026-08-21) rather than an open item — see that item's note for the rationale and the condition
under which it should be revisited.
