class Mining:
    """Build municipality x year mining-permit (PLG) measures from the ANM Cadastro Mineiro microdata."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def preprocess(self, archive=None, adm2_output_path=None, monthly_output_path=None):
        """Count PLG requests, grants and PLGs in force by municipality and year, and requests by month."""
        from .preprocess import build_adm2_permits

        return build_adm2_permits(root_dir=self.root_dir, archive=archive, output_path=adm2_output_path,
                                  monthly_output_path=monthly_output_path)


__all__ = ["Mining"]
