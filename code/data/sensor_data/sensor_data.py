class sensor_data:
    """Thin wrapper around the water-quality scraping pipeline.

    The heavy lifting lives in `scraper/`. This wrapper keeps the public module
    interface small and makes CLI integration straightforward.
    """

    def __init__(
        self,
        root_dir=".",
        download_dir=None,
        headless=False,
        keep_browser_on_error=False,
        single_station=None,
        fetch_mode="default",
    ):
        self.root_dir = root_dir
        self.download_dir = download_dir
        self.headless = headless
        self.keep_browser_on_error = keep_browser_on_error
        self.single_station = single_station
        self.fetch_mode = fetch_mode

    def fetch(self):
        """Download raw station archives for prepared sensor stations."""
        from .scraper.data.download import fetch_station_data

        fetch_station_data(
            root_dir=self.root_dir,
            download_dir=self.download_dir,
            headless=self.headless,
            keep_browser_on_error=self.keep_browser_on_error,
            single_station=self.single_station,
            fetch_mode=self.fetch_mode,
        )

    def preprocess(self):
        """Parse downloaded station archives into DuckDB tables."""
        from .scraper.data.preprocess import preprocess_station_data

        preprocess_station_data(
            root_dir=self.root_dir,
            single_station=self.single_station,
        )
