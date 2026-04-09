# OFFER ML-1M Release

This repository is a compact public release for the ML-1M part of the paper. It contains:

- the released ML-1M sequence files used by the method
- the released regenerated ML-1M outputs for four entropy settings
- the minimal generation code needed to understand how the released data was produced
- a small tokenizer support package and the ML-1M tokenizer checkpoint used in the release
- a runnable HSTU-based three-stage pipeline for ML-1M

This repository is intentionally limited to the paper content. It does not include the rebuttal-only evaluation pipelines.

## Repository layout

```text
offer_ml1m_release_P1/
+- data/
|  +- ml-1m/
|     +- origin/
|     |  +- sasrec_format_origin.csv
|     +- regen-005/
|     +- regen-030/
|     +- regen-050/
|     +- regen-070/
+- code/
|  +- 1.Build_pretraining_dataset.py
|  +- 2.Pretrain_regenerator.py
|  +- 3.Hybrid_inference.py
|  +- pre2_convert_sasrec_format.py
|  +- pre4_tokenizer.py
|  +- fin3_replace_sasrec_format.py
|  +- module/
|  +- utils/
|  +- data/
|  +- tokenizer_support/
+- artifacts/
|  +- tokenizer/
|     +- ml-1m/
+- hstu_3stage/
|  +- main.py
|  +- generative_recommenders/
|  +- configs/ml-1m/
|  +- scripts/ml-1m/
|  +- README.md
+- configs/
```

## What is already released

The main released assets are the ready-to-use ML-1M sequence files under `data/ml-1m/`.

For each entropy setting, two final CSV files are provided:

- `sasrec_format_final_recall.csv`
- `sasrec_format_final_rerank.csv`

and one pattern dictionary:

- `pattern_mappings.json`

The available released settings are:

- `regen-005`
- `regen-030`
- `regen-050`
- `regen-070`

The original sequence file is:

- `data/ml-1m/origin/sasrec_format_origin.csv`

If you only need the released paper data, you can use these files directly and skip the generation code.

## Minimal environment

The code was organized for Python 3.10+.

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

## Data format

Each sequence CSV follows the same row format:

- `index`
- `user_id`
- `sequence_item_ids`
- `sequence_ratings`
- `sequence_timestamps`

The `*_recall.csv` file is the released sequence file for the recall stage.

The `*_rerank.csv` file is the released sequence file for the reranking stage.

`pattern_mappings.json` stores the released token-to-pattern mapping used by the augmentation pipeline.

## Simple generation view

The original internal workflow is long. For the public release, it can be understood in three steps.

### Step 1: Mine low-entropy token patterns

Use `pre4_tokenizer.py` to mine candidate low-entropy token patterns from ML-1M. The repository already includes:

- the ML-1M tokenizer text file in `code/tokenizer_support/data/ml-1m.txt`
- the tokenizer checkpoint in `artifacts/tokenizer/ml-1m/`

Example:

```bash
python code/pre4_tokenizer.py \
  --dataset ml-1m \
  --train_dir release_run \
  --tokenize_only true \
  --state_dict_path artifacts/tokenizer/ml-1m/SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth
```

This step outputs a token JSON file under `artifacts/tokenizer_runs/`.

### Step 2: Build sequence-pattern pairs and train the regenerator

Use:

- `1.Build_pretraining_dataset.py`
- `2.Pretrain_regenerator.py`

These scripts consume intermediate tensors stored in a working directory, typically:

- `seq2pat_data.pth`
- `train_ori.pth`
- `pre-trained_embedding.ckpt`

Example:

```bash
python code/1.Build_pretraining_dataset.py --root_path ./workdir/ml-1m --max_seq_len 200
python code/2.Pretrain_regenerator.py --root_path ./workdir/ml-1m --max_seq_len 200 --epochs 40
```

This stage produces:

- `patterns.pth`
- `seq-pat-pair.pth`
- a regenerator checkpoint

### Step 3: Batched hybrid inference and final CSV export

Use:

- `3.Hybrid_inference.py`
- `fin3_replace_sasrec_format.py`

Example:

```bash
python code/3.Hybrid_inference.py \
  --root_path ./workdir/ml-1m \
  --max_seq_len 402 \
  --domain_count 2 \
  --loops 1 \
  --batch_size 512 \
  --output_name train_regen.pth

python code/fin3_replace_sasrec_format.py \
  --root_path ./workdir/ml-1m \
  --origin_path data/ml-1m/origin \
  --convert_type recall

python code/fin3_replace_sasrec_format.py \
  --root_path ./workdir/ml-1m \
  --origin_path data/ml-1m/origin \
  --convert_type rerank
```

This stage produces the final released sequence CSV files.

## Recommended way to use this release

For most readers, the easiest path is:

1. Start from `data/ml-1m/origin/sasrec_format_origin.csv`
2. Compare it with the released regenerated outputs under `regen-005`, `regen-030`, `regen-050`, and `regen-070`
3. Use `code/` only if you want to inspect how the released data was generated
4. Use `hstu_3stage/` if you want to run the released three-stage HSTU pipeline on ML-1M

## HSTU three-stage pipeline

The repository also includes a runnable HSTU-based cascade pipeline under `hstu_3stage/`.

Its default example is:

1. recall top-1000 export
2. coarse ranking on the recall candidate pool
3. fine ranking on the top-50 coarse pool

See:

- `hstu_3stage/README.md`
- `hstu_3stage/scripts/ml-1m/run_three_stage_release.sh`

## Notes

- Only ML-1M is kept in this release.
- The repository uses relative paths only.
- Personal paths, user names, and external repository links were removed from the public-facing code copy.
