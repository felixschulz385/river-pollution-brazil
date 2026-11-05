class CloudCover:
    """Measure INPE's DETER cloud cover (2005-2016) and count DETER-B alerts (from 2016) per Legal Amazon municipality."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def preprocess(self, path=None, gadm_path=None, layer=None, adm2_id_column=None, month_output_path=None,
                   adm2_output_path=None, workers=None):
        """Share of each municipality under DETER's cloud mask per month, with calendar- and PRODES-year means."""
        from .preprocess import build_clouds

        return build_clouds(root_dir=self.root_dir, path=path, gadm_path=gadm_path, layer=layer,
                            adm2_id_column=adm2_id_column, month_output_path=month_output_path,
                            adm2_output_path=adm2_output_path, workers=workers)

    def preprocess_alerts(self, path=None, gadm_path=None, layer=None, adm2_id_column=None, month_output_path=None,
                          adm2_output_path=None):
        """DETER-B alert counts and areas per municipality and month, for all classes, deforestation and mining."""
        from .preprocess import build_alerts

        return build_alerts(root_dir=self.root_dir, path=path, gadm_path=gadm_path, layer=layer,
                            adm2_id_column=adm2_id_column, month_output_path=month_output_path,
                            adm2_output_path=adm2_output_path)


__all__ = ["CloudCover"]
