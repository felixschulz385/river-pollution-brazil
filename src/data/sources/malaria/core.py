class Malaria:
    """Fetch SIVEP-Malária notifications and aggregate them to ADM2 panels and garimpo flows."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def fetch(self):
        """Download the yearly SIVEP parquet files from Zenodo, verified by md5."""
        from .fetch import fetch_sivep

        return fetch_sivep(root_dir=self.root_dir)

    def preprocess(self, adm2_output_path=None, flows_output_path=None, monthly_output_path=None):
        """Build the ADM2 case counts (by year and by month) and the garimpo residence-to-infection flows."""
        from .preprocess import build_malaria_panels

        return build_malaria_panels(
            root_dir=self.root_dir,
            adm2_output_path=adm2_output_path,
            flows_output_path=flows_output_path,
            monthly_output_path=monthly_output_path,
        )


__all__ = ["Malaria"]
