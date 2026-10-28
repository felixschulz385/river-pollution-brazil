class CommodityPrices:
    """Fetch the gold price and the BRL/USD exchange rate and build gold prices in reais."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def fetch(self):
        """Download the World Bank Pink Sheet and BCB SGS series 1."""
        from .fetch import fetch_commodity_prices

        return fetch_commodity_prices(root_dir=self.root_dir)

    def preprocess(self, monthly_output_path=None, annual_output_path=None):
        """Build the monthly and annual gold price tables in USD and BRL."""
        from .preprocess import build_gold_price_tables

        return build_gold_price_tables(
            root_dir=self.root_dir,
            monthly_output_path=monthly_output_path,
            annual_output_path=annual_output_path,
        )


__all__ = ["CommodityPrices"]
