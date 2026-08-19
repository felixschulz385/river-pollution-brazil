class Health:
    """Run health-data fetching and preprocessing workflows."""

    def __init__(self, root_dir=".", headless=False, download_dir=None):
        self.root_dir = root_dir
        self.headless = headless
        self.download_dir = download_dir

    def fetch(self, subtype="all"):
        """Fetch raw health tables from DATASUS TABNET into `data/health/raw`."""
        from .fetch import fetch_health_data

        return fetch_health_data(
            root_dir=self.root_dir,
            subtype=subtype,
            headless=self.headless,
            download_dir=self.download_dir,
        )

    def preprocess(self, subtype="all"):
        """Transform raw health tables into analysis-ready files under `data/health`."""
        from .preprocess import preprocess_health_data

        return preprocess_health_data(root_dir=self.root_dir, subtype=subtype)


__all__ = ["Health"]
