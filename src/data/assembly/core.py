import logging

from .build import assemble_dataset as _assemble_dataset
from .build import write_dataset as _write_dataset
from .constants import DEFAULT_CONFIG_PATH
from .schema import get_dataset_config


logger = logging.getLogger(__name__)


class Assembly:
    """Join per-domain data outputs into analysis-ready sensor- or ADM2-indexed tables."""

    def __init__(self, root_dir=".", config_path=None):
        self.root_dir = root_dir
        self.config_path = config_path or DEFAULT_CONFIG_PATH

    def assemble(self, dataset_id, config_path=None, output_path=None):
        """Assemble the dataset identified by `dataset_id` in the assembly config."""
        resolved_config_path = config_path or self.config_path
        dataset_config = get_dataset_config(resolved_config_path, dataset_id)
        logger.info(
            "Assembling dataset '%s' (mode=%s) from %s",
            dataset_config.id,
            dataset_config.mode,
            resolved_config_path,
        )
        merged = _assemble_dataset(dataset_config, root_dir=self.root_dir)
        resolved_output_path = output_path or dataset_config.output_path
        _write_dataset(merged, resolved_output_path)
        return merged
