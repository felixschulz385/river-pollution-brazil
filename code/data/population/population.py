class population:
    """Run population-data fetching and preprocessing workflows."""

    def __init__(self, root_dir=".", billing_project="river-pollution-499210"):
        self.root_dir = root_dir
        self.billing_project = billing_project

    def fetch(self):
        """Fetch raw population tables from BigQuery into `data/population/raw`."""
        from .fetch import fetch_population_data

        return fetch_population_data(
            root_dir=self.root_dir,
            billing_project=self.billing_project,
        )

    def preprocess(self):
        """Transform raw population tables into analysis-ready files under `data/population`."""
        from .preprocess import preprocess_population_data

        return preprocess_population_data(root_dir=self.root_dir)


__all__ = ["population"]
