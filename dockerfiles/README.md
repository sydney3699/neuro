# Container images

One image per conda environment (their deps conflict — JAX vs multiple torch-cu121
builds vs scgpt's torch 2.3/py3.10 — so a single image is infeasible). Each image
creates its env from the committed `envs/<env>.yml` via micromamba, then
`pip install --no-deps -e .` so the shared `neurospatial` package imports cleanly.
GPU images use the `nvidia/cuda:12.1.1-cudnn8-runtime` base to match the `cu121`
wheel pins; CPU images use `ubuntu:22.04`.

Build all from the repo root: `dockerfiles/build.sh` (see the script header for
`REGISTRY`/`TAG`/`PUSH` overrides and the ECR login snippet for AWS Batch).

| Image (`neuro-<stage>`) | Env spec | GPU | Nextflow processes it backs |
|---|---|---|---|
| `neuro-envi`     | `envs/neuro.yml`      | yes | ENVI imputation (`run_envi.py`, incl. GW34) |
| `neuro-scvi`     | `envs/scvi-annot.yml` | yes | scVI reference annotation (`annotate_cells.py --embedding scvi`) |
| `neuro-scgpt`    | `envs/scgpt.yml`      | yes | scGPT zero-shot embedding (`scgpt_embed.py`) |
| `neuro-stagate`  | `envs/stagate.yml`    | yes | STAGATE spatial domains (`stagate_domains.py`) |
| `neuro-analysis` | `envs/neuro.yml`      | no  | COMMOT (`run_commot.py`/`merge_commot.py`), all niche stages (`profile_niches`, `compare_niches`, `signaling_diff`, `sp_niche_analysis`, `permutation_composition`, `layer_recovery_eval`), GW34 (`harmonize_h1`, `persistence_compare`, `build_gw34_envi_input`), annotation post-proc (`recalibrate_confidence`, `build_coarse_celltypes`), viz |
| `neuro-banksy`   | `envs/banksy.yml`     | no  | Banksy spatial domains (`banksy_domains.py`) — isolated `pybanksy` env |

Note: `neuro-analysis` intentionally excludes `pybanksy` (kept in its own image,
mirroring the isolated cluster env); everything else CPU-only runs in `neuro-analysis`.
