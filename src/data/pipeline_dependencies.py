"""Structured cross-source dependency graph gating `data fetch`/`data preprocess`/`data assemble`.

Real data dependencies exist between sources (e.g. `sensor_data.preprocess()`
hard-requires `river_network` and `gadm`, deep inside `preprocess/assembly.py`)
but nothing in `src/cli.py` checks them before
dispatching -- running a source out of order today just crashes inside its
own preprocessing code. This module declares those dependencies explicitly,
answers "what's blocking this stage" (`unmet_prerequisites`), and builds a
topologically-ordered run plan for `--chain` (`build_chain`).

"Fetched" is read from `SourceReport.fetch_status` (verification's raw
fetched-artifact check): anything other than "not_present_locally" counts.
"Preprocessed" is read from `SourceReport.preprocess_complete`: True iff
every declared output artifact exists, regardless of whether its content
passes checks -- a source with all files present but a failed value-range
check still counts as "preprocessed" for gating purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StagePrerequisites:
    requires_fetched: tuple[str, ...] = ()
    requires_preprocessed: tuple[str, ...] = ()


# Explicit, hand-verified cross-source edges -- keyed by (source, verb) for
# verb in {"fetch", "preprocess"}. A pair with no entry has no *cross-source*
# prerequisites; every source's own "preprocess" verb additionally always
# requires that same source's own raw data to be present, a universal rule
# applied by `unmet_prerequisites`/`build_chain` themselves, not repeated here.
#
# - sensor_data.preprocess() reads river_network's processed trenches while
#   matching stations to the river network, and separately reads gadm's
#   simplified boundary directly to filter stations to within Brazil (both in
#   preprocess/assembly.py, called as part of its single preprocess step).
#   fetch has no such dependency -- only scraping happens there.
# - land_cover.preprocess() and climate.preprocess() both read
#   river_network's processed drainage areas; their aggregate/assemble
#   sub-phase additionally reads sensor_data's fully-preprocessed output.
#   Since dependencies are declared per verb (not per --phase), the whole
#   "preprocess" verb is conservatively gated on both.
# - biomes.preprocess() reads the "stations" table populated by
#   sensor_data.fetch() (not sensor_data.preprocess()), and separately reads
#   gadm's simplified boundary directly for ADM2 mapping.
# - river_network.preprocess() (RiverNetwork.generate()) now always
#   annotates drainage areas and builds the trench-ADM2 table from gadm's
#   simplified output -- there is no longer a partial/un-annotated output.
# `gadm` itself is the shared, manually-placed ADM2 boundary geopackage --
# its own pseudo-source (see `src/data/verification/sources.py`) with a real
# preprocessing step (simplification, `src/data/sources/gadm`), not owned by
# any of the three sources that read its simplified output.
PIPELINE_DEPENDENCIES: dict[tuple[str, str], StagePrerequisites] = {
    ("river_network", "preprocess"): StagePrerequisites(requires_preprocessed=("gadm",)),
    ("sensor_data", "preprocess"): StagePrerequisites(requires_preprocessed=("river_network", "gadm")),
    ("land_cover", "preprocess"): StagePrerequisites(requires_preprocessed=("river_network", "sensor_data")),
    ("climate", "preprocess"): StagePrerequisites(requires_preprocessed=("river_network", "sensor_data")),
    ("biomes", "preprocess"): StagePrerequisites(
        requires_fetched=("sensor_data",), requires_preprocessed=("gadm",)
    ),
}

# Pseudo-source name for the global `data assemble` step, reused as a graph
# node alongside the 7 real SOURCE_REGISTRY names.
ASSEMBLY_NODE = "assembly"


def assembly_prerequisites(root_dir: str = ".") -> StagePrerequisites:
    """Derive assembly's prerequisite sources from `setup/assembly_datasets.yaml`.

    Every source name that appears as a `data/<source>/...` path prefix among
    the config's declared source paths is required to be preprocessed --
    derived dynamically (not hardcoded) so this can't silently drift from the
    config, e.g. once health is wired in it starts showing up here for free.
    Degrades to no prerequisites if the config is missing/unparseable, mirroring
    how `src/data/verification/sources.py`'s `_load_assembly_datasets_safe`
    already treats those cases.
    """
    from src.cli import SOURCE_REGISTRY
    from src.data.assembly.constants import DEFAULT_CONFIG_PATH
    from src.data.assembly.schema import load_assembly_config

    config_path = Path(root_dir) / DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return StagePrerequisites()
    try:
        datasets = load_assembly_config(config_path)
    except Exception:
        return StagePrerequisites()

    known_sources = set(SOURCE_REGISTRY)
    referenced: set[str] = set()
    for dataset in datasets.values():
        for source in dataset.sources:
            parts = Path(source.path).parts
            if len(parts) >= 2 and parts[0] == "data" and parts[1] in known_sources:
                referenced.add(parts[1])
    return StagePrerequisites(requires_preprocessed=tuple(sorted(referenced)))


def _prerequisites_for(source: str, verb: str, root_dir: str) -> StagePrerequisites:
    if source == ASSEMBLY_NODE:
        return assembly_prerequisites(root_dir)
    return PIPELINE_DEPENDENCIES.get((source, verb), StagePrerequisites())


def _has_automated_fetch(name: str) -> bool:
    """Whether `name` is a `data fetch`-dispatchable source. Nodes outside
    `SOURCE_REGISTRY` entirely (e.g. `gadm`, a shared file with no owning
    source module) are treated the same as a registered `fetch=False`
    source: manual placement only."""
    from src.cli import SOURCE_REGISTRY

    return SOURCE_REGISTRY.get(name, {}).get("fetch", False)


def _self_fetch_problem(source: str, root_dir: str, verification) -> str | None:
    """The universal rule: preprocessing a source requires that source's own
    raw data to already be present. Returns a human-readable problem
    description, or None if satisfied."""
    report = verification.verify(source=source)[source]
    if report.fetch_status != "not_present_locally":
        return None
    if _has_automated_fetch(source):
        return f"'{source}' has no raw data fetched yet (run: python -m src.cli data fetch --source {source})"
    return f"'{source}' has no raw data present locally -- it must be placed manually (see readme.md)."


def unmet_prerequisites(source: str, verb: str, root_dir: str = ".") -> list[str]:
    """Human-readable descriptions of everything blocking `(source, verb)`."""
    from src.data.verification.core import Verification

    verification = Verification(root_dir)
    problems: list[str] = []

    if verb == "preprocess" and source != ASSEMBLY_NODE:
        self_problem = _self_fetch_problem(source, root_dir, verification)
        if self_problem is not None:
            problems.append(self_problem)

    prereqs = _prerequisites_for(source, verb, root_dir)
    prereq_names = set(prereqs.requires_fetched) | set(prereqs.requires_preprocessed)
    reports = {name: verification.verify(source=name)[name] for name in prereq_names}

    for name in prereqs.requires_fetched:
        if reports[name].fetch_status != "not_present_locally":
            continue
        if _has_automated_fetch(name):
            problems.append(f"'{name}' has not been fetched yet (run: python -m src.cli data fetch --source {name})")
        else:
            problems.append(f"'{name}' has no raw data present locally -- it must be placed manually (see readme.md).")
    for name in prereqs.requires_preprocessed:
        if not reports[name].preprocess_complete:
            problems.append(
                f"'{name}' has not finished preprocessing yet (run: python -m src.cli data preprocess --source {name})"
            )
    return problems


def build_chain(source: str, verb: str, root_dir: str = ".") -> list[tuple[str, str]]:
    """A topologically-ordered `[(source, verb), ...]` run plan ending with
    `(source, verb)` itself, covering every transitively unmet prerequisite.

    Raises `ValueError` if a prerequisite has no automated fetch and its raw
    data isn't present locally (river_network, gadm) -- that can't be
    auto-resolved, only manually placed.
    """
    from src.data.verification.core import Verification

    verification = Verification(root_dir)
    chain: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def resolve(node_source: str, node_verb: str, path: tuple[tuple[str, str], ...]) -> None:
        key = (node_source, node_verb)
        if key in seen:
            return
        if key in path:
            cycle = " -> ".join(f"{s}.{v}" for s, v in (*path, key))
            raise ValueError(f"Dependency cycle detected: {cycle}")
        next_path = (*path, key)

        if node_verb == "preprocess" and node_source != ASSEMBLY_NODE:
            self_problem = _self_fetch_problem(node_source, root_dir, verification)
            if self_problem is not None:
                if not _has_automated_fetch(node_source):
                    raise ValueError(self_problem)
                resolve(node_source, "fetch", next_path)

        prereqs = _prerequisites_for(node_source, node_verb, root_dir)
        for name in prereqs.requires_fetched:
            report = verification.verify(source=name)[name]
            if report.fetch_status != "not_present_locally":
                continue
            if not _has_automated_fetch(name):
                raise ValueError(
                    f"'{name}' has no automated fetch step and its raw data is not present "
                    "locally -- it must be placed manually (see readme.md)."
                )
            resolve(name, "fetch", next_path)
        for name in prereqs.requires_preprocessed:
            report = verification.verify(source=name)[name]
            if not report.preprocess_complete:
                resolve(name, "preprocess", next_path)

        chain.append(key)
        seen.add(key)

    resolve(source, verb, ())
    return chain


__all__ = [
    "ASSEMBLY_NODE",
    "PIPELINE_DEPENDENCIES",
    "StagePrerequisites",
    "assembly_prerequisites",
    "build_chain",
    "unmet_prerequisites",
]
