# partly adapted from https://github.com/yiyanxu/diffrec
import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

import data_utils
import evaluate_utils
import models.gaussian_diffusion as gd
from calculate_fairness import compute_all_provider_metrics
from models.guided_DNN import AdaptiveAutoGuideNet, DNN, SelfGuidanceNet

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a diffusion model for recommendation"
    )

    # --- Dataset & Paths ---
    data_group = parser.add_argument_group("Dataset & Paths")
    data_group.add_argument("--dataset", type=str, default="ml-1m", help="dataset name")
    data_group.add_argument("--data_path", type=str, default="../dataset/ml-1m/C5_[7,1,2]/", help="data path")
    data_group.add_argument("--save_path", type=str, default="./saved_models/", help="save model path")
    data_group.add_argument("--log_name", type=str, default="log", help="log name")
    data_group.add_argument("--log_dir", type=str, default="./logs/", help="log directory")

    # --- Training ---
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.0)
    train_group.add_argument("--batch_size", type=int, default=400)
    train_group.add_argument("--epochs", type=int, default=500, help="upper epoch limit")
    train_group.add_argument("--epoch_limit", type=int, default=1000, help="epoch limit")
    train_group.add_argument("--patience", type=int, default=30, help="patience for early stopping")
    train_group.add_argument("--random_seed", type=int, default=1, help="random seed (single seed)")
    train_group.add_argument("--random_seeds", type=str, default=None, help="comma-separated random seeds for multiple runs (e.g., 1,42,2025 or '1,42,2025'). Overrides --random_seed")
    train_group.add_argument("--round", type=int, default=1, help="experiment round")
    train_group.add_argument("--amp", action="store_true", help="use mixed precision training")
    train_group.add_argument("--save_every", type=int, default=0, help="save checkpoint every N epochs")

    # --- Device ---
    device_group = parser.add_argument_group("Device")
    device_group.add_argument("--cuda", action="store_true", help="use CUDA")
    device_group.add_argument("--gpu", type=str, default="0", help="gpu card ID")

    # --- Model Architecture ---
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--time_type", type=str, default="cat", help="cat or add")
    model_group.add_argument("--dims", type=str, default="[1000]", help="DNN dims")
    model_group.add_argument("--norm", type=lambda x: x.lower() == "true", default=False, help="normalize input")
    model_group.add_argument("--emb_size", type=int, default=10, help="timestep embedding size")

    # --- Diffusion ---
    diff_group = parser.add_argument_group("Diffusion")
    diff_group.add_argument("--mean_type", type=str, default="x0", help="MeanType: x0, eps")
    diff_group.add_argument("--steps", type=int, default=100, help="diffusion steps")
    diff_group.add_argument("--noise_schedule", type=str, default="linear-var", help="noise schedule")
    diff_group.add_argument("--noise_scale", type=float, default=0.1, help="noise scale")
    diff_group.add_argument("--noise_min", type=float, default=0.0001, help="noise lower bound")
    diff_group.add_argument("--noise_max", type=float, default=0.02, help="noise upper bound")
    diff_group.add_argument("--sampling_noise", type=lambda x: x.lower() == "true", default=False)
    diff_group.add_argument("--sampling_steps", type=int, default=-1, help="sampling steps")
    diff_group.add_argument("--sampling_ratio", type=float, default=-1.0, help="sampling ratio")
    diff_group.add_argument("--reweight", type=lambda x: x.lower() == "true", default=True)

    # --- Self-Guidance (cdiff) ---
    cdiff_group = parser.add_argument_group("Self-Guidance (cdiff)")
    cdiff_group.add_argument("--use_cdiff", action="store_true", help="enable self-guidance")
    cdiff_group.add_argument("--cdiff_w", type=float, default=0.1, help="self-guidance weight")

    # --- CFG Guidance ---
    cfg_group = parser.add_argument_group("CFG Guidance")
    cfg_group.add_argument("--use_cfg", action="store_true", help="use CFG guidance")
    cfg_group.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale")
    cfg_group.add_argument("--cond_drop_prob", type=float, default=0.1, help="condition dropout prob")

    # --- Auto Guidance (AG) ---
    ag_group = parser.add_argument_group("Auto Guidance")
    ag_group.add_argument("--use_ag", action="store_true", help="enable auto guidance")
    ag_group.add_argument("--ag_w", type=float, default=2.0, help="auto guidance weight")
    ag_group.add_argument("--g_ckpt", type=str, default="", help="weak model checkpoint path")
    ag_group.add_argument("--use_same_arch_for_d0", action="store_true", help="D0 uses same arch as D1")
    ag_group.add_argument("--use_cond", type=lambda x: x.lower() == "true", default=False)
    ag_group.add_argument("--weak_dims", type=str, required=False, default=None, help="dims for weak D0")

    # --- Adaptive Autoguidance (A2G) ---
    a2g_group = parser.add_argument_group("Adaptive Auto Guidance (A2G)")
    a2g_group.add_argument("--train_a2g", action="store_true", help="enable a2g training")
    a2g_group.add_argument("--ag_w_max", type=float, default=3, help="max guidance weight")
    a2g_group.add_argument("--a2g_hidden", type=int, default=32, help="GuideNet hidden size")
    a2g_group.add_argument("--lambda_ag", type=float, default=0.2, help="AG loss weight")
    a2g_group.add_argument("--lambda_pop", type=float, default=0.2, help="popularity loss weight")
    a2g_group.add_argument("--a2g_tau", type=float, default=2.0, help="a2g tau")
    a2g_group.add_argument("--use_stats", type=lambda x: x.lower() == "true", default=True)
    a2g_group.add_argument("--use_item_groups", type=lambda x: x.lower() == "true", default=True)
    a2g_group.add_argument("--use_bins", type=lambda x: x.lower() == "true", default=False,
                           help="A2G: use popularity_bins (1=head,2=mid,3=tail) to define tail for guide; tail=3 -> group 2.")
    a2g_group.add_argument("--pop_global_Q_head", type=float, default=0.2)
    a2g_group.add_argument("--pop_global_Q_tail", type=float, default=0.2)
    a2g_group.add_argument("--bonus_coefficient", type=float, default=0.2)
    a2g_group.add_argument("--pop_K_candidate", type=int, default=50)

    # --- Evaluation ---
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--topN", type=str, default="[10, 20, 50, 100]")
    eval_group.add_argument("--tst_w_val", action="store_true", help="test with validation")
    eval_group.add_argument("--fairness_on_update", action="store_true", default=False)
    eval_group.add_argument("--no_fairness_on_update", dest="fairness_on_update", action="store_false")

    # --- Inference ---
    infer_group = parser.add_argument_group("Inference")
    infer_group.add_argument("--inference_only", action="store_true")
    infer_group.add_argument("--best_model_path", type=str, default="", help="best model checkpoint")
    infer_group.add_argument("--ablation_type", type=str, default="none", help="ablation type")

    # --- Small Test Mode ---
    test_group = parser.add_argument_group("Small Test Mode")
    test_group.add_argument("--small_test", action="store_true", help="enable small-scale test mode")
    test_group.add_argument("--small_test_users", type=int, default=1000, help="number of users to use in small test mode")

    # --- Weights & Biases ---
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument("--use_wandb", action="store_true", help="use W&B logging")
    wandb_group.add_argument("--wandb_project", type=str, default="test_public")
    wandb_group.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (default: use logged-in user)")
    wandb_group.add_argument("--wandb_run_name", type=str, default=None)
    wandb_group.add_argument("--wandb_group", type=str, default=None)
    wandb_group.add_argument("--wandb_tags", type=str, default="")
    wandb_group.add_argument("--wandb_notes", type=str, default="")

    return parser


@dataclass
class TrainingContext:
    # Models
    model: torch.nn.Module = None
    diffusion: gd.GaussianDiffusion = None
    model_d0: Optional[torch.nn.Module] = None
    guide_net: Optional[torch.nn.Module] = None
    self_guidance_net: Optional[torch.nn.Module] = None

    # Optimizer
    optimizer: torch.optim.Optimizer = None

    # Data
    train_loader: DataLoader = None
    test_loader: DataLoader = None
    test_twv_loader: Optional[DataLoader] = None
    train_data: Any = None
    valid_y_data: Any = None
    test_y_data: Any = None
    mask_tv: Any = None

    # Dataset info
    n_user: int = 0
    n_item: int = 0

    item_groups: Optional[np.ndarray] = None
    user_groups: Optional[np.ndarray] = None
    popularity_bins: Optional[np.ndarray] = None
    a2g_item_groups: Optional[np.ndarray] = None

    in_dims: List[int] = field(default_factory=list)
    out_dims: List[int] = field(default_factory=list)

    best_recall: float = -100.0
    best_epoch: int = 0
    best_results: Any = None
    best_ckpt: Dict = None

    model_save_root: Path = None
    hparams: Dict = field(default_factory=dict)

    device: torch.device = None


class LoggerWriter:
    def __init__(self, level_func):
        self.level_func = level_func
        self._buf = ""

    def write(self, message):
        if not message or message == "\n":
            return
        self._buf += message
        if "\n" in self._buf:
            for line in self._buf.splitlines():
                if line.strip():
                    self.level_func(line)
            self._buf = ""

    def flush(self):
        if self._buf.strip():
            self.level_func(self._buf.strip())
            self._buf = ""

    def isatty(self):
        return False

    def fileno(self):
        try:
            return sys.__stdout__.fileno()
        except Exception:
            return 1

    @property
    def encoding(self):
        return getattr(sys.__stdout__, "encoding", "utf-8")


def setup_logging(log_dir: Path, log_name: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.__stdout__),
        ],
    )

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.info)

    return log_file


def log_metrics(epoch: int, phase: str, metrics_dict: Dict, use_wandb: bool = False):
    prefix = f"[Epoch {epoch:03d}] {phase}"

    metric_strs = []
    for key, values in metrics_dict.items():
        if isinstance(values, (list, tuple)) and len(values) == 4:
            metric_strs.append(f"{key}: {values[0]:.4f}/{values[1]:.4f}/{values[2]:.4f}/{values[3]:.4f}")
        else:
            metric_strs.append(f"{key}: {values:.4f}")

    logging.info("%s | %s", prefix, " | ".join(metric_strs))

    if use_wandb and WANDB_AVAILABLE:
        wandb_dict = {}
        for key, values in metrics_dict.items():
            if isinstance(values, (list, tuple)) and len(values) == 4:
                for i, k in enumerate([10, 20, 50, 100]):
                    wandb_dict[f"{phase}/{key}@{k}"] = values[i]
            else:
                wandb_dict[f"{phase}/{key}"] = values
        wandb.log(wandb_dict, step=epoch)


def try_load_array(paths: List[str]) -> Optional[np.ndarray]:
    for p in paths:
        if os.path.exists(p):
            if p.endswith(".npy"):
                return np.load(p)
            else:
                with open(p, "r") as f:
                    return np.array([int(x.strip()) for x in f if x.strip() != ""])
    return None


def load_fairness_groups(data_path: str, n_user: int, n_item: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    fairness_dir = os.path.join(data_path, "fairness")

    user_group_candidates = [
        os.path.join(fairness_dir, "fairness_user_groups.npy"),
        os.path.join(fairness_dir, "fairness_user_groups.txt"),
        os.path.join(fairness_dir, "user_groups.npy"),
        os.path.join(fairness_dir, "user_groups.txt"),
    ]
    item_group_candidates = [
        os.path.join(fairness_dir, "fairness_item_groups.npy"),
        os.path.join(fairness_dir, "fairness_item_groups.txt"),
        os.path.join(fairness_dir, "item_groups.npy"),
        os.path.join(fairness_dir, "item_groups.txt"),
    ]
    popularity_bin_candidates = [
        os.path.join(fairness_dir, "popularity_bins_mass.npy"),
        os.path.join(fairness_dir, "popularity_bins_mass.txt"),
    ]

    popularity_bins = try_load_array(popularity_bin_candidates)
    user_groups = try_load_array(user_group_candidates)
    item_groups = try_load_array(item_group_candidates)

    if popularity_bins is not None:
        if len(popularity_bins) != n_item:
            logging.warning("[Fairness] popularity_bins size mismatch: %d != %d", len(popularity_bins), n_item)
        logging.info("popularity_bins shape: %s", popularity_bins.shape)
    else:
        logging.info("popularity_bins: None")

    if user_groups is not None:
        if len(user_groups) != n_user:
            logging.warning("[Fairness] user_groups size mismatch: %d != %d", len(user_groups), n_user)
        logging.info("user_groups shape: %s", user_groups.shape)
    else:
        logging.info("user_groups: None")

    if item_groups is not None:
        if len(item_groups) != n_item:
            logging.warning("[Fairness] item_groups size mismatch: %d != %d", len(item_groups), n_item)
        logging.info("item_groups shape: %s", item_groups.shape)
    else:
        logging.info("item_groups: None")

    return user_groups, item_groups, popularity_bins


def derive_item_groups_from_pop_bins(pop_bins: np.ndarray) -> np.ndarray:
    pop_bins = np.asarray(pop_bins, dtype=np.int64)
    out = np.ones_like(pop_bins, dtype=np.int64)
    out[pop_bins == 3] = 2
    return out


def load_complete_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    load_guide: bool = True,
    load_weak: bool = True
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    hparams = ckpt["hparams"]
    dims = eval(hparams["dims"])
    emb_size = hparams["emb_size"]
    n_item = hparams["n_item"]
    norm = hparams.get("norm", False)
    time_type = hparams.get("time_type", "cat")
    out_dims = dims + [n_item]
    in_dims = out_dims[::-1]

    # Main model
    model = DNN(in_dims, out_dims, emb_size, time_type=time_type, norm=norm).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Diffusion
    mean_type = gd.ModelMeanType.START_X if hparams["mean_type"] == "x0" else gd.ModelMeanType.EPSILON
    diffusion = gd.GaussianDiffusion(
        mean_type,
        hparams.get("noise_schedule", "linear-var"),
        hparams.get("noise_scale", 0.1),
        hparams.get("noise_min", 0.0001),
        hparams.get("noise_max", 0.02),
        hparams["steps"],
        device
    ).to(device)

    guide_net = None
    self_guidance_net = None
    model_d0 = None

    # Load guide_net if needed
    if load_guide and hparams.get("train_a2g", False):
        guide_path = hparams.get("guide_path", None)
        if guide_path is not None and os.path.exists(guide_path):
            guide_ckpt = torch.load(guide_path, map_location=device)
            guide_net = AdaptiveAutoGuideNet(
                item_dim=n_item,
                hidden=hparams.get("a2g_hidden", 32),
                w_max=hparams.get("ag_w_max", 3.5),
                tau=hparams.get("a2g_tau", 2.0),
                use_stats=hparams.get("use_stats", True),
                use_item_groups=hparams.get("use_item_groups", True),
                bonus_coefficient=hparams.get("bonus_coefficient", 0.2),
            ).to(device)
            guide_net.load_state_dict(guide_ckpt["state_dict"])
            guide_net.eval()
            logging.info("GuideNet loaded from: %s", guide_path)

    if hparams.get("use_cdiff", False):
        ckpt_path_obj = Path(ckpt_path)
        sg_path = str(ckpt_path_obj.parent / (ckpt_path_obj.stem + "__self_guidance.pth"))
        if os.path.exists(sg_path):
            sg_ckpt = torch.load(sg_path, map_location=device)
            self_guidance_net = SelfGuidanceNet(
                in_dims=in_dims,
                out_dims=out_dims,
                emb_size=emb_size,
                norm=norm
            ).to(device)
            self_guidance_net.load_state_dict(sg_ckpt["state_dict"])
            self_guidance_net.eval()
            logging.info("Self-GuidanceNet loaded from: %s", sg_path)

    # Load weak model if needed
    if load_weak and (hparams.get("use_ag", False) or hparams.get("train_a2g", False)):
        if hparams.get("use_same_arch_for_d0", False):
            d0_in_dims, d0_out_dims = in_dims, out_dims
        else:
            d0_dims = eval(hparams.get("weak_dims", "[1000]"))
            d0_out_dims = d0_dims + [n_item]
            d0_in_dims = d0_out_dims[::-1]

        model_d0 = DNN(d0_in_dims, d0_out_dims, emb_size, time_type=time_type, norm=norm).to(device)
        if hparams.get("g_ckpt") and os.path.exists(hparams["g_ckpt"]):
            weak_ckpt = torch.load(hparams["g_ckpt"], map_location=device)
            state_dict = weak_ckpt.get("state_dict", weak_ckpt)
            model_d0.load_state_dict(state_dict, strict=False)
            logging.info("Weak model dims: %s -> %s", d0_in_dims, d0_out_dims)
            logging.info("Weak model loaded from: %s", hparams["g_ckpt"])
        model_d0.eval()

    return {
        "model": model,
        "diffusion": diffusion,
        "guide_net": guide_net,
        "self_guidance_net": self_guidance_net,
        "model_d0": model_d0,
        "hparams": hparams
    }


def save_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    tag: str,
    save_dir: Path,
    hparams: Dict,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> str:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{hparams['dataset']}_{tag}.pth"

    ckpt = {
        "state_dict": model.state_dict(),
        "hparams": hparams,
        "model_class": model.__class__.__name__,
        "epoch": epoch,
        "code_version": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        },
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()

    path = save_dir / fname
    torch.save(ckpt, path)
    logging.info("Checkpoint saved: %s", path)
    return str(path)


# Evaluation 
def evaluate_with_lists(
    ctx: TrainingContext,
    args: argparse.Namespace,
    data_loader: DataLoader,
    data_te: Any,
    mask_his: Any,
    topN: List[int]
) -> Tuple[Tuple, List[List[int]], List[List[int]]]:
    ctx.model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]
    topk = topN[-1]

    predict_items = []
    target_items = []

    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx * args.batch_size:batch_idx * args.batch_size + len(batch)]]
            batch = batch.to(ctx.device)
            cond = batch if args.use_cond else None

            if args.use_cdiff:
                prediction = ctx.diffusion.p_sample_self_guidance(
                    ctx.model,
                    ctx.self_guidance_net,
                    batch,
                    args.sampling_steps,
                    guidance_weight=args.cdiff_w,
                    sampling_noise=args.sampling_noise
                )
            elif args.train_a2g:
                prediction = ctx.diffusion.p_sample_a2g(
                    model_main=ctx.model,
                    model_weak=ctx.model_d0,
                    guide_net=ctx.guide_net,
                    x_start=batch,
                    steps=args.sampling_steps,
                    cond=cond,
                    sampling_noise=args.sampling_noise,
                    item_groups=ctx.a2g_item_groups
                )
            elif args.use_cfg:
                prediction = ctx.diffusion.p_sample_cfg(
                    ctx.model,
                    batch,
                    args.sampling_steps,
                    cfg_scale=args.cfg_scale,
                    sampling_noise=args.sampling_noise
                )
            elif args.use_ag:
                prediction = ctx.diffusion.p_sample_ag(
                    ctx.model,
                    ctx.model_d0,
                    batch,
                    args.sampling_steps,
                    cond=cond,
                    w=args.ag_w,
                    sampling_noise=args.sampling_noise
                )
            else:
                prediction = ctx.diffusion.p_sample(
                    ctx.model,
                    batch,
                    args.sampling_steps,
                    sampling_noise=args.sampling_noise
                )
            prediction[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(prediction, topk)
            predict_items.extend(indices.cpu().numpy().tolist())

    metrics = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)
    return metrics, predict_items, target_items


def evaluate(
    ctx: TrainingContext,
    args: argparse.Namespace,
    data_loader: DataLoader,
    data_te: Any,
    mask_his: Any,
    topN: List[int]
) -> Tuple:
    metrics, _, _ = evaluate_with_lists(ctx, args, data_loader, data_te, mask_his, topN)
    return metrics


def evaluate_popularity(data_te, mask_his, topN: List[int]) -> Tuple:
    import scipy.sparse as sp

    if sp.issparse(mask_his):
        item_pop = np.asarray(mask_his.sum(axis=0)).ravel()
        his_bool = mask_his.toarray().astype(bool)
    else:
        item_pop = np.array(mask_his.sum(axis=0)).ravel()
        his_bool = mask_his.astype(bool)

    if sp.issparse(data_te):
        te_bool = data_te.toarray().astype(bool)
    else:
        te_bool = data_te.astype(bool)

    pop_order = np.argsort(-item_pop)
    num_users, num_items = his_bool.shape
    Kmax = topN[-1]

    predict_items, target_items = [], []

    for u in range(num_users):
        mask_u = his_bool[u]
        allowed_idx_in_order = np.nonzero(~mask_u[pop_order])[0][:Kmax]
        rec = pop_order[allowed_idx_in_order].tolist()
        predict_items.append(rec)
        tgt = np.flatnonzero(te_bool[u]).tolist()
        target_items.append(tgt)

    return evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)


def evaluate_fairness(
    predict_items: List[List[int]],
    ground_truth: List[List[int]],
    data_path: str,
    dataset_name: str,
    n_user: int,
    n_item: int,
    topN: List[int],
    item_groups: Optional[np.ndarray],
    user_groups: Optional[np.ndarray],
    use_wandb: bool = False,
    epoch: Optional[int] = None
) -> Dict:
    provider_results = None

    if item_groups is not None:
        k_max = max(topN)
        predicted_array = np.zeros((len(predict_items), k_max), dtype=int)
        for i, recs in enumerate(predict_items):
            if recs:
                predicted_array[i, :min(len(recs), k_max)] = recs[:k_max]

        actual_short_head_ratio = float((item_groups == 1).sum()) / len(item_groups)
        provider_results = compute_all_provider_metrics(
            predicted_indices=predicted_array,
            item_groups=item_groups,
            short_head_ratio=actual_short_head_ratio,
            topN=topN,
        )

    # Aggregate and log
    lines = ["\n=== Fairness Metrics Summary ==="]
    wandb_payload = {}

    for k in topN:
        aplt = (provider_results or {}).get(f"APLT@{k}")
        dexp = (provider_results or {}).get(f"DeltaExposure@{k}")
        coverage = (provider_results or {}).get(f"ItemCoverage@{k}")
        longtail_coverage = (provider_results or {}).get(f"LongtailCoverage@{k}")
        gini = (provider_results or {}).get(f"Gini@{k}")
        entropy = (provider_results or {}).get(f"Entropy@{k}")

        line = f"@{k} → " + \
               (f"APLT={aplt:.4f}" if aplt is not None else "APLT=N/A") + " | " + \
               (f"ΔExp={dexp:.4f}" if dexp is not None else "ΔExp=N/A") + " | " + \
               (f"ItemCoverage={coverage:.4f}" if coverage is not None else "ItemCoverage=N/A") + " | " + \
               (f"LTCoverage={longtail_coverage:.4f}" if longtail_coverage is not None else "LTCoverage=N/A") + " | " + \
               (f"Gini={gini:.4f}" if gini is not None else "Gini=N/A") + " | " + \
               (f"Entropy={entropy:.4f}" if entropy is not None else "Entropy=N/A")
        lines.append(line)

        wandb_payload.update({
            f"Fairness/APLT@{k}": aplt,
            f"Fairness/ΔExp@{k}": dexp,
            f"Fairness/ItemCoverage@{k}": coverage,
            f"Fairness/LTCoverage@{k}": longtail_coverage,
            f"Fairness/Gini@{k}": gini,
            f"Fairness/Entropy@{k}": entropy,
        })

    logging.info("\n".join(lines))

    if use_wandb and WANDB_AVAILABLE:
        if epoch is not None:
            wandb.log(wandb_payload, step=epoch)
        else:
            final_payload = {f"Final/{k}": v for k, v in wandb_payload.items()}
            wandb.log(final_payload)

    logging.info("Fairness metrics logged")
    return {
        "provider": provider_results,
        "wandb_logged": wandb_payload,
    }



def compute_training_loss(
    ctx: TrainingContext,
    args: argparse.Namespace,
    batch: torch.Tensor
) -> Dict[str, torch.Tensor]:
    cond = batch if args.use_cond else None

    if args.use_cdiff:
        return ctx.diffusion.training_losses_self_guidance(
            ctx.model,
            ctx.self_guidance_net,
            batch,
            guidance_weight=args.cdiff_w,
            reweight=args.reweight
        )
    elif args.train_a2g:
        return ctx.diffusion.training_losses_a2g(
            main_model=ctx.model,
            weak_model=ctx.model_d0,
            guide_net=ctx.guide_net,
            x_start=batch,
            cond=cond,
            pop_global_Q_head=args.pop_global_Q_head,
            pop_global_Q_tail=args.pop_global_Q_tail,
            lambda_ag=args.lambda_ag,
            lambda_pop=args.lambda_pop,
            reweight=args.reweight,
            item_groups=ctx.a2g_item_groups,
            popularity_bins=ctx.popularity_bins
        )
    elif args.use_cfg:
        return ctx.diffusion.training_losses_cfg(
            ctx.model,
            batch,
            cond_drop_prob=args.cond_drop_prob,
            reweight=args.reweight
        )
    elif args.use_ag and args.use_cond:
        return ctx.diffusion.training_losses_cfg(
            ctx.model,
            batch,
            cond_drop_prob=args.cond_drop_prob,
            reweight=args.reweight
        )
    else:
        return ctx.diffusion.training_losses(
            ctx.model,
            batch,
            reweight=args.reweight,
        )

def train_epoch(
    ctx: TrainingContext,
    args: argparse.Namespace,
    scaler: torch.cuda.amp.GradScaler
) -> Dict[str, float]:
    ctx.model.train()

    batch_count = 0
    total_loss = 0.0
    total_pop_loss = 0.0
    total_ag_loss = 0.0
    total_base_loss = 0.0

    for batch in ctx.train_loader:
        batch = batch.to(ctx.device, non_blocking=True)
        batch_count += 1
        ctx.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.amp):
            losses = compute_training_loss(ctx, args, batch)
            loss = losses["loss"].mean()

            if args.train_a2g:
                pop_loss = losses["loss_pop"].mean()
                ag_loss = losses["loss_ag"].mean()
                base_loss = losses["loss_base"].mean()

        total_loss += float(loss.detach())
        if args.train_a2g:
            total_pop_loss += float(pop_loss.detach())
            total_ag_loss += float(ag_loss.detach())
            total_base_loss += float(base_loss.detach())

        scaler.scale(loss).backward()
        scaler.step(ctx.optimizer)
        scaler.update()

    return {
        "loss": total_loss / batch_count,
        "pop_loss": total_pop_loss / batch_count if args.train_a2g else 0.0,
        "ag_loss": total_ag_loss / batch_count if args.train_a2g else 0.0,
        "base_loss": total_base_loss / batch_count if args.train_a2g else 0.0,
    }


def save_best_model(
    ctx: TrainingContext,
    args: argparse.Namespace,
    epoch: int,
    model_save_root: Path
) -> str:
    BEST_TAG = "best"
    BEST_PTH = model_save_root / f"{args.dataset}_{BEST_TAG}.pth"
    BEST_GUIDE = model_save_root / f"{args.dataset}_{BEST_TAG}__guide.pth"
    BEST_SELF_GUIDANCE = model_save_root / f"{args.dataset}_{BEST_TAG}__self_guidance.pth"

    best_ckpt = {
        "state_dict": ctx.model.state_dict(),
        "hparams": ctx.hparams,
        "model_class": ctx.model.__class__.__name__,
        "epoch": epoch,
        "code_version": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "optimizer": ctx.optimizer.state_dict(),
    }

    if ctx.self_guidance_net is not None:
        self_guidance_ckpt = {
            "state_dict": ctx.self_guidance_net.state_dict(),
            "class": ctx.self_guidance_net.__class__.__name__,
        }
        best_ckpt["self_guidance_net"] = self_guidance_ckpt
        torch.save(self_guidance_ckpt, BEST_SELF_GUIDANCE)
        logging.info("Self-GuidanceNet saved to: %s", BEST_SELF_GUIDANCE)

    if ctx.guide_net is not None:
        ctx.hparams["guide_path"] = str(BEST_GUIDE)
        guide_ckpt = {
            "state_dict": ctx.guide_net.state_dict(),
            "class": ctx.guide_net.__class__.__name__,
            "hparams": {
                "w_max": args.ag_w_max,
                "hidden": args.a2g_hidden,
                "lambda_ag": args.lambda_ag,
                "lambda_pop": args.lambda_pop,
                "a2g_tau": args.a2g_tau,
                "use_stats": args.use_stats,
                "use_item_groups": args.use_item_groups,
                "guide_path": str(BEST_GUIDE),
                "g_ckpt": str(args.g_ckpt),
                "use_same_arch_for_d0": str(args.use_same_arch_for_d0),
                "weak_dims": str(args.weak_dims),
            },
        }
        best_ckpt["guide_net"] = guide_ckpt
        torch.save(guide_ckpt, BEST_GUIDE)
        logging.info("GuideNet saved to: %s", BEST_GUIDE)

    # Save main checkpoint
    torch.save(best_ckpt, BEST_PTH)
    logging.info("Model saved to: %s", BEST_PTH)

    ctx.best_ckpt = best_ckpt
    return str(BEST_PTH)


def run_training(ctx: TrainingContext, args: argparse.Namespace, run_ts: str) -> Optional[str]:
    logging.info("=" * 80)
    logging.info("TRAINING STARTED")
    logging.info("=" * 80)

    iters_per_epoch = len(ctx.train_loader)
    max_epochs = min(args.epochs, args.epoch_limit)

    logging.info("Training for %d epochs with patience=%d", max_epochs, args.patience)
    logging.info("Iters per epoch: %d", iters_per_epoch)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model_save_root = Path(args.save_path) / args.dataset / run_ts
    model_save_root.mkdir(parents=True, exist_ok=True)
    ctx.model_save_root = model_save_root

    BEST_JSON = model_save_root / f"{args.dataset}_best.json"
    TOPN = eval(args.topN)
    best_model_path = None

    for epoch in range(1, max_epochs + 1):
        # Early stopping
        if epoch - ctx.best_epoch >= args.patience:
            logging.info("=" * 80)
            logging.info("EARLY STOPPING at epoch %d (best: %d)", epoch, ctx.best_epoch)
            logging.info("=" * 80)
            break

        start_time = time.time()
        losses = train_epoch(ctx, args, scaler)
        epoch_time = time.time() - start_time

        logging.info("[Epoch %03d] Train Loss: %.4f | Time: %s",
                    epoch, losses["loss"], time.strftime("%H:%M:%S", time.gmtime(epoch_time)))

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "train/loss": losses["loss"],
                "train/pop_loss": losses["pop_loss"],
                "train/ag_loss": losses["ag_loss"],
                "train/base_loss": losses["base_loss"],
                "train/epoch_time": epoch_time,
            }, step=epoch)

        # Evaluation
        if epoch < 5 or epoch % 5 == 0: # TODO change this 
        # if epoch>0: 
            logging.info("-" * 80)

            # Validation
            valid_results, valid_predict_items, valid_target_items = evaluate_with_lists(
                ctx, args,
                data_loader=ctx.test_loader,
                data_te=ctx.valid_y_data,
                mask_his=ctx.train_data,
                topN=TOPN,
            )
            valid_precision, valid_recall, valid_ndcg, valid_mrr = valid_results

            log_metrics(epoch, "Valid", {
                "Precision": valid_precision,
                "Recall": valid_recall,
                "NDCG": valid_ndcg,
                "MRR": valid_mrr
            }, use_wandb=args.use_wandb)
            # Check for best model
            if valid_recall[1] > ctx.best_recall:
                ctx.best_recall = valid_recall[1]
                ctx.best_epoch = epoch
                ctx.best_results = valid_results

                if args.fairness_on_update:
                    evaluate_fairness(
                        predict_items=valid_predict_items,
                        ground_truth=valid_target_items,
                        data_path=args.data_path,
                        dataset_name=args.dataset,
                        n_user=ctx.n_user,
                        n_item=ctx.n_item,
                        topN=TOPN,
                        item_groups=ctx.item_groups,
                        user_groups=ctx.user_groups,
                        use_wandb=args.use_wandb,
                        epoch=epoch
                    )

                logging.info(">>> NEW BEST MODEL at epoch %d (Recall@20: %.4f) <<<", epoch, ctx.best_recall)
                best_model_path = save_best_model(ctx, args, epoch, model_save_root)

                # Save manifest
                manifest = {
                    "total_epoch": epoch,
                    "best_epoch": ctx.best_epoch,
                    "iters_per_epoch": iters_per_epoch,
                    "best_epoch_iter": ctx.best_epoch * iters_per_epoch,
                    "best_valid_results": ctx.best_results,
                    "args": vars(args),
                }
                with open(BEST_JSON, "w") as f:
                    json.dump(manifest, f, indent=2)

                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.run.summary["best_epoch"] = ctx.best_epoch
                    wandb.run.summary["best_valid_recall@20"] = ctx.best_recall
                    wandb.run.summary["iters_per_epoch"] = iters_per_epoch
                    artifact = wandb.Artifact(
                        name=f"model-{wandb.run.id}",
                        type="model",
                        description=f"Best model at epoch {ctx.best_epoch}"
                    )
                    artifact.add_file(str(BEST_JSON))
                    wandb.log_artifact(artifact)

            logging.info("-" * 80)

        # Periodic checkpointing
        if args.save_every > 0 and epoch % args.save_every == 0:
            periodic_path = save_checkpoint(
                ctx.model, epoch, f"ep{epoch}", model_save_root, ctx.hparams, ctx.optimizer
            )
            logging.info("[Periodic Save] epoch=%d path=%s", epoch, periodic_path)

    logging.info("=" * 80)
    logging.info("TRAINING COMPLETED")
    logging.info("Best Epoch: %03d | Best Recall@20: %.4f", ctx.best_epoch, ctx.best_recall)
    logging.info("=" * 80)

    return best_model_path

# Initialization Functions
def setup_device(args: argparse.Namespace) -> torch.device:
    if getattr(args, "gpu", None) and args.gpu not in ("env", "auto", ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    use_cuda = bool(args.cuda) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        frac = float(os.environ.get("PER_PROCESS_MEM_FRAC", "0.90"))
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    return device


def set_random_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate_guidance_args(args: argparse.Namespace):
    if args.use_cdiff:
        if args.use_cfg or args.use_ag or args.train_a2g:
            logging.info("[cdiff] enabled → disabling CFG/AG/A2G")
        args.use_cfg = False
        args.use_ag = False
        args.train_a2g = False

    active_guiders = sum([
        bool(args.use_cfg),
        bool(args.use_ag),
        bool(args.train_a2g),
        bool(args.use_cdiff)
    ])
    assert active_guiders <= 1, "Only one guidance method can be enabled"


def build_hparams(args: argparse.Namespace, n_user: int, n_item: int) -> Dict:
    return {
        "dataset": args.dataset,
        "dims": args.dims,
        "emb_size": args.emb_size,
        "n_user": n_user,
        "n_item": n_item,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "mean_type": args.mean_type,
        "steps": args.steps,
        "noise_scale": args.noise_scale,
        "noise_min": args.noise_min,
        "noise_max": args.noise_max,
        "noise_schedule": args.noise_schedule,
        "sampling_steps": args.sampling_steps,
        "sampling_noise": args.sampling_noise,
        "sampling_ratio": args.sampling_ratio,
        "topN": args.topN,
        "reweight": args.reweight,
        "batch_size": args.batch_size,
        "random_seed": args.random_seed,
        "norm": args.norm,
        "time_type": args.time_type,
        # CFG
        "cfg_scale": args.cfg_scale,
        "cond_drop_prob": args.cond_drop_prob,
        "use_cfg": args.use_cfg,
        # cdiff
        "cdiff_w": args.cdiff_w,
        "use_cdiff": args.use_cdiff,
        # AG
        "use_ag": args.use_ag,
        "ag_w": args.ag_w,
        "g_ckpt": args.g_ckpt,
        "use_same_arch_for_d0": args.use_same_arch_for_d0,
        "weak_dims": args.weak_dims,
        "use_cond": args.use_cond,
        # a2g
        "train_a2g": args.train_a2g,
        "ag_w_max": args.ag_w_max,
        "lambda_ag": args.lambda_ag,
        "lambda_pop": args.lambda_pop,
        "a2g_hidden": args.a2g_hidden,
        "a2g_tau": args.a2g_tau,
        "use_stats": args.use_stats,
        "use_item_groups": args.use_item_groups,
        "use_bins": args.use_bins,
        "bonus_coefficient": args.bonus_coefficient,
        # Small test mode
        "small_test": args.small_test,
        "small_test_users": args.small_test_users if args.small_test else None,
    }


def initialize_wandb(args: argparse.Namespace, run_id: str):
    if not args.use_wandb or not WANDB_AVAILABLE:
        return
    if args.g_ckpt:
        fname = args.g_ckpt.split("/")[-1]
        m = re.search(r"_ep(\d+)", fname)
        if m is None:
            raise ValueError(f"Cannot parse epoch from checkpoint name: {fname}")
        g_ckpt_ep = int(m.group(1))
    else:
        g_ckpt_ep = None
    wandb_config = {
        "dataset": args.dataset,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "epoch_limit": args.epoch_limit,
        "patience": args.patience,
        "random_seed": args.random_seed,
        "dims": args.dims,
        "emb_size": args.emb_size,
        "time_type": args.time_type,
        "norm": args.norm,
        "mean_type": args.mean_type,
        "steps": args.steps,
        "noise_schedule": args.noise_schedule,
        "noise_scale": args.noise_scale,
        "noise_min": args.noise_min,
        "noise_max": args.noise_max,
        "sampling_noise": args.sampling_noise,
        "sampling_steps": args.sampling_steps,
        "reweight": args.reweight,
        "use_cfg": args.use_cfg,
        "cfg_scale": args.cfg_scale,
        "cond_drop_prob": args.cond_drop_prob,
        "use_ag": args.use_ag,
        "ag_w": args.ag_w,
        "g_ckpt": args.g_ckpt,
        "g_ckpt_ep": g_ckpt_ep,
        "train_a2g": args.train_a2g,
        "ag_w_max": args.ag_w_max,
        "a2g_hidden": args.a2g_hidden,
        "lambda_ag": args.lambda_ag,
        "lambda_pop": args.lambda_pop,
        "a2g_tau": args.a2g_tau,
        "use_cdiff": args.use_cdiff,
        "cdiff_w": args.cdiff_w,
        "topN": args.topN,
    }

    wandb_tags = [tag.strip() for tag in args.wandb_tags.split(",")] if args.wandb_tags else []
    wandb_tags.append(args.dataset)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        notes=args.wandb_notes,
        name=args.wandb_run_name or args.log_name or run_id,
        config=wandb_config,
        tags=wandb_tags,
        group=args.wandb_group,
    )
    logging.info("Weights & Biases initialized: %s", wandb.run.url)


def build_models(
    args: argparse.Namespace,
    n_item: int,
    in_dims: List[int],
    out_dims: List[int],
    device: torch.device
) -> Tuple[DNN, gd.GaussianDiffusion, Optional[DNN], Optional[AdaptiveAutoGuideNet], Optional[SelfGuidanceNet]]:
    # Diffusion
    if args.mean_type == "x0":
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == "eps":
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError(f"Unknown mean type: {args.mean_type}")

    diffusion = gd.GaussianDiffusion(
        mean_type, args.noise_schedule,
        args.noise_scale, args.noise_min, args.noise_max,
        args.steps, device
    ).to(device)
    # Weak model for AG/A2G
    model_d0 = None
    if args.use_ag or args.train_a2g:
        assert args.g_ckpt, "AG/a2g requires --g_ckpt"

        if args.use_same_arch_for_d0:
            d0_out_dims = out_dims
            d0_in_dims = in_dims
            args.weak_dims = args.dims
        else:
            d0_base_dims = eval(args.weak_dims)
            d0_out_dims = d0_base_dims + [n_item]
            d0_in_dims = d0_out_dims[::-1]

        model_d0 = DNN(d0_in_dims, d0_out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
        ckpt = torch.load(args.g_ckpt, map_location=device)
        if isinstance(ckpt, dict) and "model_obj" in ckpt:
            model_d0.load_state_dict(ckpt["model_obj"].state_dict(), strict=False)
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model_d0.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            model_d0.load_state_dict(ckpt, strict=False)

        for p in model_d0.parameters():
            p.requires_grad = False
        model_d0.eval()
        logging.info("Weak model loaded and frozen")

    # Main model
    model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)

    # GuideNet for a2g
    guide_net = None
    if args.train_a2g:
        guide_net = AdaptiveAutoGuideNet(
            item_dim=n_item,
            hidden=args.a2g_hidden,
            w_max=args.ag_w_max,
            tau=args.a2g_tau,
            use_stats=args.use_stats,
            use_item_groups=args.use_item_groups,
            bonus_coefficient=args.bonus_coefficient,
        ).to(device)

    # Self-guidance net for cdiff
    self_guidance_net = None
    if args.use_cdiff:
        self_guidance_net = SelfGuidanceNet(
            in_dims=in_dims, out_dims=out_dims, emb_size=args.emb_size, norm=args.norm
        ).to(device)
        logging.info("Self-Guidance Net loaded")

    return model, diffusion, model_d0, guide_net, self_guidance_net


# Main Function
def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Timestamps and IDs
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{args.dataset}_{run_ts}"

    # Validate args
    if args.inference_only:
        assert args.best_model_path, "Provide --best_model_path for inference-only mode"

    inference_mode = bool(args.best_model_path)
    train_mode = not inference_mode

    if args.sampling_ratio != -1.0:
        args.sampling_steps = max(1, int(round(args.steps * args.sampling_ratio)))

    if not args.log_name or args.log_name == "log":
        args.log_name = run_id
    logs_dir = Path(args.log_dir) / args.dataset / run_ts
    log_file = setup_logging(logs_dir, args.log_name)

    logging.info("=" * 80)
    logging.info("EXPERIMENT STARTED")
    logging.info("=" * 80)
    logging.info("Arguments: %s", vars(args))
    logging.info("Log file: %s", log_file)

    device = setup_device(args)
    logging.info("Device: %s", device)

    set_random_seeds(args.random_seed)

    validate_guidance_args(args)

    initialize_wandb(args, run_id)

    # Data path
    if args.data_path[-1] != "/":
        args.data_path += "/"

    # Load data
    logging.info("-" * 80)
    logging.info("LOADING DATA")
    logging.info("-" * 80)

    train_path = args.data_path + "train_list.npy"
    valid_path = args.data_path + "valid_list.npy"
    test_path = args.data_path + "test_list.npy"

    train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(
        train_path, valid_path, test_path
    )

    if args.small_test:
        original_n_user = n_user
        n_user_limit = min(args.small_test_users, n_user)
        logging.info("=" * 80)
        logging.info("SMALL TEST MODE ENABLED")
        logging.info("Limiting users from %d to %d", original_n_user, n_user_limit)
        logging.info("=" * 80)
        
        train_data = train_data[:n_user_limit]
        valid_y_data = valid_y_data[:n_user_limit]
        test_y_data = test_y_data[:n_user_limit]
        n_user = n_user_limit
        
        if args.epochs > 50:
            logging.info("Reducing epochs from %d to 50 for small test mode", args.epochs)
            args.epochs = 50
        if args.epoch_limit > 50:
            args.epoch_limit = 50
        if args.patience > 10:
            logging.info("Reducing patience from %d to 10 for small test mode", args.patience)
            args.patience = 10

    logging.info("Dataset: %s", args.dataset)
    logging.info("Users: %d | Items: %d", n_user, n_item)
    logging.info("Train: %d | Valid: %d | Test: %d",
                train_data.nnz, valid_y_data.nnz, test_y_data.nnz)

    num_workers = int(os.environ.get("DL_NUM_WORKERS", "12"))
    prefetch = int(os.environ.get("DL_PREFETCH", "4"))

    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)

    train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, pin_memory=True,
        shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn,
        persistent_workers=True, prefetch_factor=prefetch if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
        num_workers=num_workers, worker_init_fn=worker_init_fn,
        persistent_workers=True, prefetch_factor=prefetch if num_workers > 0 else None,
    )

    test_twv_loader = None
    if args.tst_w_val:
        tv_dataset = data_utils.DataDiffusion(
            torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A)
        )
        test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)

    mask_tv = train_data + valid_y_data

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.config.update({
            "n_user": n_user,
            "n_item": n_item,
            "iters_per_epoch": len(train_loader),
        }, allow_val_change=True)

    user_groups, item_groups, popularity_bins = load_fairness_groups(args.data_path, n_user, n_item)

    if args.small_test:
        if user_groups is not None and len(user_groups) > n_user:
            logging.info("Limiting user_groups from %d to %d", len(user_groups), n_user)
            user_groups = user_groups[:n_user]
        if popularity_bins is not None and len(popularity_bins) > n_item:
            logging.info("Limiting popularity_bins from %d to %d", len(popularity_bins), n_item)
            popularity_bins = popularity_bins[:n_item]
        if item_groups is not None and len(item_groups) > n_item:
            logging.info("Limiting item_groups from %d to %d", len(item_groups), n_item)
            item_groups = item_groups[:n_item]

    if getattr(args, "use_bins", False) and popularity_bins is not None:
        item_groups = derive_item_groups_from_pop_bins(popularity_bins)
        logging.info("use_bins=True → item_groups overridden from popularity_bins (tail=3→2, rest→1), shape=%s", item_groups.shape)
    if item_groups is not None and popularity_bins is not None:
        
        #print tail num (tail = 2), head = 1
        tail_num = np.sum(item_groups == 2)
        head_num = np.sum(item_groups == 1)
        logging.info("Item groups distribution (from popularity_bins): head=1 → %d items, tail=2 → %d items", head_num, tail_num)

    if args.use_item_groups and item_groups is None and not getattr(args, "use_bins", False):
        logging.warning("use_item_groups=True but item_groups is None (and not use_bins), setting to False")
        args.use_item_groups = False

    logging.info("-" * 80)
    logging.info("BUILDING MODEL")
    logging.info("-" * 80)

    out_dims = eval(args.dims) + [n_item]
    in_dims = out_dims[::-1]

    model, diffusion, model_d0, guide_net, self_guidance_net = build_models(
        args, n_item, in_dims, out_dims, device
    )

    parameters = list(model.parameters())
    if guide_net is not None:
        parameters += list(guide_net.parameters())
    if self_guidance_net is not None:
        parameters += list(self_guidance_net.parameters())

    optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

    mlp_num = sum(p.nelement() for p in model.parameters())
    diff_num = sum(p.nelement() for p in diffusion.parameters())
    guide_num = sum(p.nelement() for p in guide_net.parameters()) if guide_net else 0

    logging.info("Model: in_dims=%s, out_dims=%s, emb_size=%d", in_dims, out_dims, args.emb_size)
    logging.info("Parameters: MLP=%d, Diffusion=%d, Total=%d", mlp_num, diff_num, mlp_num + diff_num + guide_num)

    if args.train_a2g:
        logging.info("Using Adaptive Autoguidance (A2G)")
    elif args.use_ag:
        logging.info("Using Autoguidance (AG) with w=%f", args.ag_w)
    elif args.use_cfg:
        logging.info("Using CFG with scale=%f", args.cfg_scale)
    elif args.use_cdiff:
        logging.info("Using Self-Guidance (cdiff) with w=%f", args.cdiff_w)
    else:
        logging.info("Using standard diffusion")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.config.update({"total_parameters": sum(p.nelement() for p in parameters)})
        try:
            wandb.watch(model, log="all", log_freq=2000)
            if guide_net is not None:
                wandb.watch(guide_net, log="all", log_freq=1000)
        except ValueError as e:
            logging.warning("wandb.watch skipped: %s", e)

    ctx = TrainingContext(
        model=model,
        diffusion=diffusion,
        model_d0=model_d0,
        guide_net=guide_net,
        self_guidance_net=self_guidance_net,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        test_twv_loader=test_twv_loader,
        train_data=train_data,
        valid_y_data=valid_y_data,
        test_y_data=test_y_data,
        mask_tv=mask_tv,
        n_user=n_user,
        n_item=n_item,
        item_groups=item_groups,
        user_groups=user_groups,
        popularity_bins=popularity_bins,
        in_dims=in_dims,
        out_dims=out_dims,
        device=device,
        hparams=build_hparams(args, n_user, n_item),
    )

    if args.train_a2g:
        ctx.a2g_item_groups = ctx.item_groups if args.use_item_groups else None
        if ctx.a2g_item_groups is not None:
            logging.info("a2g using item_groups (already derived from popularity_bins if use_bins=True)")

    # Training or inference
    best_model_path = None
    if train_mode:
        best_model_path = run_training(ctx, args, run_ts)
    else:
        logging.info("=" * 80)
        logging.info("SKIP TRAINING (INFERENCE MODE)")
        logging.info("Loading checkpoint: %s", args.best_model_path)
        logging.info("=" * 80)
        best_model_path = args.best_model_path

    # Load best model for final eval
    if not best_model_path or not os.path.exists(best_model_path):
        logging.error("Best model path invalid: %s", best_model_path)
        raise FileNotFoundError(f"Invalid path: {best_model_path}")

    logging.info("Loading best checkpoint: %s", best_model_path)
    components = load_complete_model_from_checkpoint(
        best_model_path, device, load_guide=True, load_weak=True
    )

    # Update context with loaded models
    ctx.model = components["model"]
    ctx.diffusion = components["diffusion"]
    ctx.guide_net = components["guide_net"]
    ctx.self_guidance_net = components["self_guidance_net"]
    ctx.model_d0 = components["model_d0"]
    saved_hparams = components["hparams"]

    if saved_hparams.get("train_a2g"):
        ctx.a2g_item_groups = ctx.item_groups if saved_hparams.get("use_item_groups", True) else None

    if "steps" in saved_hparams:
        args.steps = int(saved_hparams["steps"])
    if "sampling_steps" in saved_hparams:
        args.sampling_steps = int(saved_hparams["sampling_steps"])
    elif saved_hparams.get("sampling_ratio", -1.0) != -1.0:
        args.sampling_steps = max(1, int(round(args.steps * saved_hparams["sampling_ratio"])))

    if args.sampling_steps < 1 or args.sampling_steps > ctx.diffusion.steps:
        logging.warning("Clamping sampling_steps=%d to %d", args.sampling_steps, ctx.diffusion.steps)
        args.sampling_steps = ctx.diffusion.steps

    logging.info("Final config: steps=%d, sampling_steps=%d", args.steps, args.sampling_steps)

    if train_mode and ctx.best_results:
        logging.info("-" * 80)
        logging.info("BEST VALIDATION RESULTS:")
        valid_precision, valid_recall, valid_ndcg, valid_mrr = ctx.best_results
        logging.info(
            "Precision: %.4f/%.4f/%.4f/%.4f | Recall: %.4f/%.4f/%.4f/%.4f | "
            "NDCG: %.4f/%.4f/%.4f/%.4f | MRR: %.4f/%.4f/%.4f/%.4f",
            *valid_precision, *valid_recall, *valid_ndcg, *valid_mrr
        )

    # Final evaluation
    TOPN = eval(args.topN)
    try:
        if args.tst_w_val:
            test_metrics, predict_items, target_items = evaluate_with_lists(
                ctx, args, ctx.test_twv_loader, ctx.test_y_data, ctx.mask_tv, TOPN
            )
        else:
            test_metrics, predict_items, target_items = evaluate_with_lists(
                ctx, args, ctx.test_loader, ctx.test_y_data, ctx.mask_tv, TOPN
            )

        test_precision, test_recall, test_ndcg, test_mrr = test_metrics
        logging.info("Final Test: Precision=%s, Recall=%s, NDCG=%s, MRR=%s",
                    "-".join(f"{x:.4f}" for x in test_precision),
                    "-".join(f"{x:.4f}" for x in test_recall),
                    "-".join(f"{x:.4f}" for x in test_ndcg),
                    "-".join(f"{x:.4f}" for x in test_mrr))

        if args.use_wandb and WANDB_AVAILABLE:
            final_metrics = {}
            for i, k in enumerate(TOPN):
                final_metrics[f"Final/Test/Precision@{k}"] = test_precision[i]
                final_metrics[f"Final/Test/Recall@{k}"] = test_recall[i]
                final_metrics[f"Final/Test/NDCG@{k}"] = test_ndcg[i]
                final_metrics[f"Final/Test/MRR@{k}"] = test_mrr[i]
            wandb.log(final_metrics)

        evaluate_fairness(
            predict_items=predict_items,
            ground_truth=target_items,
            data_path=args.data_path,
            dataset_name=args.dataset,
            n_user=ctx.n_user,
            n_item=ctx.n_item,
            topN=TOPN,
            item_groups=ctx.item_groups,
            user_groups=ctx.user_groups,
            use_wandb=args.use_wandb,
            epoch=None,
        )
    except Exception as e:
        logging.error("Final evaluation failed: %s", e)
        raise

    logging.info("-" * 80)
    logging.info("End time: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("=" * 80)

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        logging.info("Weights & Biases run finished")


if __name__ == "__main__":
    parser = create_argument_parser()
    temp_args = parser.parse_args()
    
    if temp_args.random_seeds is not None:
        seeds = [int(s.strip()) for s in temp_args.random_seeds.split(',')]
        print(f"Running with multiple seeds: {seeds}")
        print("=" * 80)
        
        for seed_idx, seed in enumerate(seeds, 1):
            print(f"\n{'='*80}")
            print(f"RUNNING SEED {seed_idx}/{len(seeds)}: seed={seed}")
            print(f"{'='*80}\n")
            
            original_argv = sys.argv.copy()
            
            new_argv = []
            skip_next = False
            for i, arg in enumerate(original_argv):
                if skip_next:
                    skip_next = False
                    continue
                if arg == '--random_seeds':
                    skip_next = True
                    continue
                new_argv.append(arg)
            
            if '--random_seed' in new_argv:
                seed_idx_in_argv = new_argv.index('--random_seed')
                new_argv[seed_idx_in_argv + 1] = str(seed)
            else:
                new_argv.extend(['--random_seed', str(seed)])
            
            sys.argv = new_argv
            
            try:
                main()
            except Exception as e:
                print(f"\n[ERROR] Seed {seed} failed: {e}")
                import traceback
                traceback.print_exc()
                print(f"Continuing with next seed...\n")
            finally:
                sys.argv = original_argv
        
        print(f"\n{'='*80}")
        print(f"ALL SEEDS COMPLETED ({len(seeds)} runs)")
        print(f"{'='*80}\n")
    else:
        # Single seed run
        main()
