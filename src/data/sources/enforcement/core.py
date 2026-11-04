class Enforcement:
    """Build IBAMA enforcement records and municipality x year measures from the Fiscalização open data, and the
    Ministry of the Environment's Priority List by municipality and year."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def fetch(self):
        """Download the Priority List from Zenodo (the IBAMA tables are placed by hand)."""
        from .fetch import fetch_priority_list

        return fetch_priority_list(root_dir=self.root_dir)

    def preprocess(self, snapshot=None, records_output_path=None, adm2_output_path=None):
        """Clean and classify fines, embargoes, seizures and destruction terms, and count them by municipality and year."""
        from .preprocess import build_enforcement

        return build_enforcement(root_dir=self.root_dir, snapshot=snapshot,
                                 records_output_path=records_output_path, adm2_output_path=adm2_output_path)

    def preprocess_priority_list(self, path=None, gadm_path=None, output_path=None):
        """Recode the Priority List to municipalities and flag each municipality and year on either list."""
        from .preprocess import build_priority_list

        return build_priority_list(root_dir=self.root_dir, path=path, gadm_path=gadm_path, output_path=output_path)


__all__ = ["Enforcement"]
