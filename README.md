# A2G-DiffRec

## Implementation of **[Adaptive Autoguidance for Item-Side Fairness in Diffusion Recommender Systems](https://arxiv.org/abs/2602.14706)** (SIGIR 2026)

## Table of Contents

- [Results](#results)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Baselines](#baselines)
- [Experiment Tracking](#experiment-tracking)
- [Citation](#citation)
- [License](#license)

---

## Results

Full performance tables at cutoffs K = 10, 20, and 100 are in [RESULTS.md](RESULTS.md).

---

## Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

Key packages:


| Package | Version           |
| ------- | ----------------- |
| Python  | 3.8.20            |
| PyTorch | 1.12.0            |
| NumPy   | 1.24.4            |
| WandB   | 0.22.3 (optional) |


---

## Dataset Preparation

### Step 1 — Download datasets


| Dataset         | Source                                                                                                                                                                      |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MovieLens 1M    | [https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/)                                                                                |
| Foursquare FTKY | [https://sites.google.com/site/yangdingqi/home/foursquare-dataset](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) (User Profile + Global-scale Check-in) |
| Music4All-Onion | [https://zenodo.org/records/6609677](https://zenodo.org/records/6609677)                                                                                                    |


### Step 2 — Organize files

Place each downloaded dataset under `dataset/<dataset_name>/`.

Create a `.env` file in the project root with the absolute path to the dataset directory:

```bash
DATASET_PATH=/absolute/path/to/A2G-DiffRec/dataset/
```

### Step 3 — Preprocess

**3a.** Run the corresponding notebook to generate `data.csv`:

```
data/preprocess_<dataset_name>.ipynb
```

**3b.** Generate train/validation/test splits and fairness metadata:

```bash
bash scripts/run_process_data.sh
```

This produces the following files under `dataset/<dataset_name>/C5_[7,1,2]/`:

```
train_list.npy
valid_list.npy
test_list.npy
fairness/item_groups.npy
fairness/popularity_bins_mass.npy
```

It also generates RecBole-format files in the same directory for use with baseline models.

---

## Training

A2G-DiffRec is trained in two stages:


| Stage       | Description                                                   |
| ----------- | ------------------------------------------------------------- |
| **Stage 1** | Train a vanilla DiffRec base model                            |
| **Stage 2** | Train with adaptive autoguidance for fairness (`--train_a2g`) |


### Stage 1 — Base DiffRec

```bash
python guided_main.py \
  --cuda \
  --dataset ml-1m \
  --data_path ./dataset/ml-1m/C5_[7,1,2]/ \
  --mean_type x0 \
  --steps 20 \
  --batch_size 768
```

The trained checkpoint will be saved to `saved_models/`.

### Stage 2 — A2G-DiffRec

Requires the Stage 1 checkpoint. Place it at `saved_models/g_ckpt/<dataset>_ep<N>.pth`, then run:

```bash
python guided_main.py \
  --cuda \
  --dataset ml-1m \
  --data_path ./dataset/ml-1m/C5_[7,1,2]/ \
  --g_ckpt saved_models/g_ckpt/ml-1m_ep2.pth \
  --train_a2g
```

### Alternative guidance modes

> **Note:** Enable only one guidance mode at a time.

**CFG (Classifier-Free Guidance):**

```bash
python guided_main.py \
  --dataset ml-1m \
  --data_path ./dataset/ml-1m/C5_[7,1,2]/ \
  --use_cfg \
  --cfg_scale <scale>
```

**AG (Autoguidance):**

```bash
python guided_main.py \
  --dataset ml-1m \
  --data_path ./dataset/ml-1m/C5_[7,1,2]/ \
  --use_ag \
  --g_ckpt <path_to_checkpoint>
```

---

## Baselines

LightGCN and MultiVAE baselines are in `baselines/` and use the [RecBole](https://github.com/RUCAIBox/RecBole) framework.

**Install RecBole from source:**

```bash
git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole
pip install -e . --verbose
```

**Run baselines:**

```bash
python search_recbole_models.py \
  --datasets ml-1m \
  --split C5_[7,1,2] \
  --models MultiVAE,LightGCN \
  --gpus 0,1,2,3 \
  --max_parallel 4
```

**Evaluate** (accuracy + fairness metrics) using the scripts in `baselines/`. The evaluation protocol matches that of the DiffRec-based models for a fair comparison.

---

## Experiment Tracking

[Weights & Biases](https://wandb.ai) is supported for tracking experiments. Configure your W&B account first, then add `--use_wandb` to any training command:

```bash
python guided_main.py \
  --use_wandb \
  ...
```

---

## Citation

(to be updated)

If you find this work useful, please cite:

```bibtex
@article{li2026a2gdiffrec,
  title={Adaptive Autoguidance for Item-Side Fairness in Diffusion Recommender Systems},
  author={Li, Zihan and Escobedo, Gustavo and Moscati, Marta and Lesota, Oleg and Schedl, Markus},
  journal={arXiv preprint arXiv:2602.14706},
  year={2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).