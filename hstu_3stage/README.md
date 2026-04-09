# HSTU Three-Stage ML-1M Pipeline

This directory contains the runnable HSTU-based three-stage pipeline used together with the released ML-1M augmentation data.

## What is included

- `main.py`: the HSTU training entry point
- `generative_recommenders/`: the required HSTU package subset
- `scripts/ml-1m/build_coarse_rank_dataset.py`: recall-to-coarse conversion
- `scripts/ml-1m/build_fine_rank_dataset.py`: coarse-to-fine conversion
- `configs/ml-1m/release-*.gin`: relative-path ML-1M configs for the three stages
- `scripts/ml-1m/run_three_stage_release.sh`: a Linux example that chains the full three-stage process

## Three-stage protocol in this release

The included pipeline is:

1. `recall`: train/export top-1000 candidates
2. `coarse ranking`: rebuild a candidate dataset from the recall export and score the top-1000 pool
3. `fine ranking`: rebuild a smaller candidate dataset from the coarse export and score the top-`m` pool

The provided example uses `top_m=50`, because this is the denser reranking setting used in the released cascade experiments.

If you want a stricter three-stage version, the fine-stage builder also supports:

- `--top_m 1`

## Required environment

Install the HSTU-side dependencies in a Linux environment:

```bash
pip install -r requirements-hstu.txt
```

At minimum you need:

- `torch`
- `gin-config`
- `absl-py`
- `pandas`
- `tensorboard`
- `torchrec`
- `fbgemm_gpu`

## Quick verification

These commands are lightweight and useful to verify the release structure before launching training:

```bash
python main.py --helpfull
python scripts/ml-1m/build_coarse_rank_dataset.py --help
python scripts/ml-1m/build_fine_rank_dataset.py --help
python scripts/ml-1m/verify_builder_smoke.py
```

## End-to-end example

From the repository root:

```bash
cd hstu_3stage
bash scripts/ml-1m/run_three_stage_release.sh
```

The script expects the released ML-1M data already present at:

- `../data/ml-1m/regen-050/sasrec_format_final_recall.csv`

and writes generated outputs under:

- `hstu_3stage/artifacts/exports/`

## Notes

- All paths are relative to the release repository.
- The current example is configured for `regen-050`.
- The fine-stage builder supports both `top_m=50` and `top_m=1`.
- This directory is intended to be runnable. It is not only a collection of placeholder scripts.
