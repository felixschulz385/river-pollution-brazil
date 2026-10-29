class Enforcement:
    """Build IBAMA enforcement records and municipality x year measures from the Fiscalização open data."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def preprocess(self, snapshot=None, records_output_path=None, adm2_output_path=None):
        """Clean and classify fines, embargoes, seizures and destruction terms, and count them by municipality and year."""
        from .preprocess import build_enforcement

        return build_enforcement(root_dir=self.root_dir, snapshot=snapshot,
                                 records_output_path=records_output_path, adm2_output_path=adm2_output_path)


__all__ = ["Enforcement"]
