# Slurm scripts

The per-source data-pipeline jobs that used to live here as hand-written
`.sh` files (`extract_climate.sh`, `assemble_climate.sh`,
`extract_land_cover.sh`, `assemble_land_cover_adm2.sh`,
`assemble_land_cover_sensor.sh`, `assemble_sensor_data.sh`,
`generate_river_network.sh`) are now submitted directly by the CLI:

```bash
python -m src.cli data fetch      --source <source> --slurm [source flags...]
python -m src.cli data preprocess --source <source> --phase <extract|aggregate> --slurm [source flags...]
python -m src.cli data assemble   --dataset <id> --slurm
```

`--slurm` renders and submits an `sbatch` job whose body is the same
command minus `--slurm`. Resource specs (partition/time/mem/cpus/conda env)
live in `setup/slurm_jobs.yaml`, keyed by `<source>.<verb>[.<phase>]`.

The remaining scripts in this directory are unrelated to `src.cli data`:
`run_sensor_analysis.sh`, `merge_sensor_analysis.sh`, and
`submit_sensor_analysis_shards.sh` belong to the `analysis sensor-data`
module; `vscode-server.sh` starts a dev server. `compute_extraction_matrices.sh`,
`compute_reachability_graph.sh`, `extract_drainage_polygons.sh`, and
`fix_drainage_polygons.sh` are orphaned/non-functional legacy scripts, kept
pending a decision on whether to retire them.
