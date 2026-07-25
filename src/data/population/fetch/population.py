from __future__ import annotations

from pathlib import Path

from ..constants import raw_dir as _raw_dir


POPULATION_QUERY = """
SELECT
    dados.ano AS ano,
    dados.id_municipio AS id_municipio,
    diretorio_id_municipio.nome AS id_municipio_nome,
    dados.sexo AS sexo,
    dados.grupo_idade AS grupo_idade,
    dados.populacao AS populacao
FROM `basedosdados.br_ms_populacao.municipio` AS dados
LEFT JOIN (
    SELECT DISTINCT
        id_municipio,
        nome
    FROM `basedosdados.br_bd_diretorios_brasil.municipio`
) AS diretorio_id_municipio
    ON dados.id_municipio = diretorio_id_municipio.id_municipio
"""




def fetch_population_data(
    root_dir: str | Path = ".",
    billing_project: str = "river-pollution-499210",
    output_path: str | Path | None = None,
) -> Path:
    """Query municipality population data from BigQuery and persist the raw extract."""

    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ImportError(
            "google-cloud-bigquery is required to fetch population data."
        ) from exc

    destination = Path(output_path) if output_path else _raw_dir(root_dir) / "population_raw.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)

    client = bigquery.Client(project=billing_project)
    job_config = bigquery.QueryJobConfig(use_legacy_sql=False)
    frame = client.query(
        POPULATION_QUERY,
        job_config=job_config,
        project=billing_project,
    ).to_dataframe(create_bqstorage_client=True)

    frame.to_parquet(destination, index=False)
    return destination
