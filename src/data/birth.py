from health.fetch import fetch_birth_outcome_tables
from health.preprocess import preprocess_birth_outcome_tables


BIRTH_SUBTYPE_MAP = {
    "all": ["gestational_duration", "birth_weight"],
    "gestation": ["gestational_duration"],
    "weight": ["birth_weight"],
}


class birth:
    """Compatibility wrapper for the birth subset of the health suite."""

    def __init__(self, root_dir=".", headless=False, download_dir=None):
        self.root_dir = root_dir
        self.headless = headless
        self.download_dir = download_dir

    def _resolve_outcomes(self, subtype):
        try:
            return BIRTH_SUBTYPE_MAP[subtype]
        except KeyError as exc:
            raise ValueError(
                f"Invalid subtype: {subtype}. Choose from: {', '.join(BIRTH_SUBTYPE_MAP)}"
            ) from exc

    def fetch(self, subtype="all"):
        """Fetch raw birth tables into `data/health/raw` via the health suite."""
        return fetch_birth_outcome_tables(
            root_dir=self.root_dir,
            headless=self.headless,
            download_dir=self.download_dir,
            outcome_names=self._resolve_outcomes(subtype),
        )

    def preprocess(self, subtype="all"):
        """Preprocess birth outputs via the integrated health suite."""
        return preprocess_birth_outcome_tables(
            root_dir=self.root_dir,
            outcome_names=self._resolve_outcomes(subtype),
        )


__all__ = ["birth"]
