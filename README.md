# NMR-Frag (Fragment Selection and Assembly with Spectrum-Guided Candidate Ranking for NMR Structure Elucidation)

![NMR-Frag selects and assembles fragments from NMR spectra.](./images/TOC.png)

Nuclear magnetic resonance (NMR) spectroscopy reveals local chemical environments that chemists use to identify molecular fragments and infer their connectivity, yet existing models rarely mirror this stepwise chemical reasoning. NMR-Frag was developed to model this process in three stages: constructing an unordered fragment multiset from a library of 3110 structural units, assembling the selected fragments into chemically valid candidate structures by predicting bonds between their attachment ports, and ranking candidates directly based on the input spectra without requiring an auxiliary spectral prediction model. Pretraining on PubChem enables NMR-Frag to capture fragment composition and assembly rules prior to finetuning on experimental spectra from NMRexp. Individual spectra, the molecular formula, or proton peak annotations were randomly omitted during training to broaden its applicability. Evaluation of the full model on NMRexp yielded a hit@1 of 76.47% and a hit@10 of 90.80%. Case studies demonstrate that fragment selection is supported by peaks at the expected chemical shifts even without explicit peak assignments. Overall, NMR-Frag elucidates molecular structures from NMR spectra with high accuracy while guiding chemists by highlighting selected substructures and the contributions of individual peaks.

[Data and checkpoints](#data-and-checkpoints) · [Checkpoint verification](#checkpoint-verification) · [Dataset preparation](#dataset-preparation) · [Training](#training) · [Evaluation](#evaluation)

## Data and checkpoints

**Please download datasets and checkpoints under `data/`.  [NMR-Frag on Hugging Face](https://huggingface.co/CROWD-lpy/NMR-Frag)**

| Local path | Contents | Needed for |
| --- | --- | --- |
| `data/checkpoints/nmrexp/last.pt` | A trained NMRexp checkpoint. An existing `best.pt` can also be supplied by its own path. | Spectrum evaluation and offline hard-negative mining |
| `data/checkpoints/pubchem/last.pt` | A PubChem reconstruction checkpoint | Starting NMRexp training from existing PubChem weights |
| `data/processed/overall_3000/` | Original processed dataset with `manifest.json`, `tokenizer.json`, `catalog.pt` and partitioned `examples/` | Evaluation, mining and PubChem training |
| `data/processed/overall_3000_hard_candidates/` | Same layout with per-record candidate graphs in the NMRexp partitions | Main NMRexp training and evaluation |
| `data/processed/fragments/tokenizer.json` | Complete fragment vocabulary and merge rules associated with an existing checkpoint | Rebuilding examples for that checkpoint from raw data |
| `data/raw/NMRexp/` | One NMRexp release or one set of split files, in Parquet, LMDB, JSONL or CSV format | Building from raw spectra. [NMRexp dataset](https://zenodo.org/records/17296666) |
| `data/raw/PubChem/CID-SMILES.gz` | Two columns, PubChem CID and SMILES, separated by whitespace | Building from raw structures. [PubChem CID–SMILES](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz) |

### Installation

```bash
conda env create -n NMRFrag_py311 -f environment.yml
conda activate NMRFrag_py311
```

## Checkpoint verification

With an existing NMRexp checkpoint and its processed dataset, generate candidates for one validation record.

```bash
python src/05_evaluate.py --stage nmrexp --split valid \
  --checkpoint data/checkpoints/nmrexp/last.pt \
  --data data/processed/overall_3000 --limit 1 \
  --out data/evaluation/checkpoint_check.jsonl --device cuda:0 --threads 1
```

## Dataset preparation

The following steps cover dataset preprocessing and restructuring. If you have already downloaded the preprocessed data and model weights, please skip this step.

### 1. Standardize records and assign splits

```bash
python src/01_prepare_data.py --nmrexp data/raw/NMRexp \
  --pubchem data/raw/PubChem/CID-SMILES.gz --out data/processed/uniform \
  --cid-mod 100 --max-heavy 100 --seed 42
```

PubChem pretraining excludes NMRexp validation and test structures. The PubChem example selects CIDs divisible by 100. Recreating a specific benchmark cohort requires its original split assignments and data snapshots.

### 2. Build or reuse the fragment vocabulary

To generate a new vocabulary, fit the merge rules on training structures and retain up to 3,000 frequent fragment tokens in addition to the atomic fallback templates.

```bash
python src/02_build_fragments.py --data data/processed/uniform \
  --out data/processed/fragments --top-k 3000 --min-count 1000 \
  --merges 256 --mine-per-source 5000 --max-atoms 12
```

When using existing weights, place their original `tokenizer.json` in `data/processed/fragments/` and go directly to step 3.

### 3. Encode and verify the training examples

```bash
python src/03_prepare_data.py --data data/processed/uniform \
  --fragments data/processed/fragments --out data/processed/overall_3000 \
  --hard-negatives 4
```

## Training

### PubChem pretraining

```bash
python src/04_train.py --stage pubchem --data data/processed/overall_3000 \
  --run-dir data/runs/pubchem --fragment-repr hybrid --epochs 25 \
  --batch-size 64 --tasks-per-record 2 --lr 2e-4 --seed 42 --device cuda:0
```

### Prepare and merge hard negatives with an existing NMRexp checkpoint

```bash
python src/07_prepare_hard_negatives.py --checkpoint data/checkpoints/nmrexp/last.pt \
  --data data/processed/overall_3000 --out data/hard/train.jsonl \
  --cohort available --limit 0 --max-negatives 8 --beam 32 --connect-beam 8 \
  --seed 42 --device cuda:0

python src/08_merge_hard_candidates.py --data data/processed/overall_3000 \
  --hard-cache data/hard/train.jsonl \
  --out data/processed/overall_3000_hard_candidates
```

<details>
<summary>Starting with a new vocabulary and no compatible NMRexp checkpoint</summary>

After PubChem pretraining, fit a seed NMRexp model using the local negatives from step 3. Use its `last.pt` as the checkpoint in step 07. The mining, merging and final fine-tuning commands then follow the same workflow. The resulting pools depend on this seed model.

```bash
python src/04_train.py --stage nmrexp --bootstrap --data data/processed/overall_3000 \
  --init-from data/runs/pubchem/last.pt --run-dir data/runs/nmrexp_seed \
  --epochs 25 --batch-size 64 --tasks-per-record 2 --lr 2e-4 \
  --lr-decay-scale 2 --rank-weight 0.5 --device cuda:0
```

</details>

### NMRexp fine-tuning

```bash
python src/04_train.py --stage nmrexp \
  --data data/processed/overall_3000_hard_candidates \
  --init-from data/runs/pubchem/last.pt --run-dir data/runs/nmrexp \
  --epochs 30 --batch-size 64 --tasks-per-record 2 --lr 2e-4 --lr-decay-scale 2 \
  --rank-weight 0.5 --hard-negatives 4 --eval-valid 500 \
  --eval-beam 32 --eval-connect-beam 8 --eval-connect-normalization pair-mask \
  --eval-score-mode joint --eval-alpha 0.5 --eval-connect-weight 0.25 \
  --eval-rank-mix 0.9 --eval-pool-size 0 --seed 42 --device cuda:0
```

For a supplied PubChem checkpoint, use `--init-from data/checkpoints/pubchem/last.pt`. 

For NMRexp training from scratch on the same merged data, omit `--init-from`, specify `--fragment-repr hybrid` and choose a new run directory. This run omits graph-teacher alignment. 

`--rank-weight` weights the training loss. 

`--eval-rank-mix` weights Rank during generation validation.

### Resume a training run

```bash
python src/04_train.py --stage nmrexp \
  --data data/processed/overall_3000_hard_candidates \
  --resume data/runs/nmrexp/last.pt --run-dir data/runs/nmrexp \
  --epochs 30 --device cuda:0
```

`--epochs` is the total epoch count.

## Evaluation

```bash
python src/05_evaluate.py --stage nmrexp --split test \
  --checkpoint data/runs/nmrexp/last.pt --data data/processed/overall_3000_hard_candidates \
  --mode hc --cohort paired --formula --annotations full --beam 64 --connect-beam 16 \
  --connect-normalization pair-mask --score-mode joint --alpha 0.5 \
  --connect-weight 0.25 --rank-mix 0.9 --pool-size 0 --limit 0 \
  --out data/evaluation/nmrexp_test.jsonl --device cuda:0
```

The reference generation score is $(A+0.25C)/\sqrt{n_A+n_C}$, where $A$ and $C$ are the cumulative Add and Connect log scores. $n_A$ counts selected fragments plus the stop action and $n_C$ counts connections. Pair-mask normalization scores each Connect choice over its allowed port partners and expands feasible continuations. Each unique structure retains its best path. The final score mixes the standardized generation score (10%) and Rank score (90%) over the full set of unique candidates. `--pool-size 0` retains that entire set.

| Evaluation setting | Standalone flags |
| --- | --- |
| Proton or carbon spectrum only | `--mode h` or `--mode c`. Use `--cohort paired` for the same molecules as the two-spectrum evaluation, or `--cohort available` for all records with the requested nucleus. |
| Shifts only or missing formula | `--annotations shift` or `--no-formula` |
| Smaller search | `--beam 32 --connect-beam 8` or `--beam 16 --connect-beam 4` |
| Generation score alone | `--rank-mix 0` |
| Given the reference fragment multiset | `--diagnostic oracle-bag` |
