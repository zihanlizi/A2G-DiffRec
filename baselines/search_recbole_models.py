#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage:
    python baselines/search_recbole_models.py \
  --datasets ml-1m --split C5_[7,1,2] --models LightGCN \
  --gpus 0,1,2 --trials_per_gpu 6 --output_root ./baselines/recbole_results_GCN --project_root ./
"""

import os
import sys
import gc
import json
import argparse
import itertools
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import time
import multiprocessing as mp
from multiprocessing import Manager
import logging
import yaml

import torch


DEFAULT_EPOCHS = 100
DEFAULT_STOPPING_STEP = 20
DEFAULT_TRAIN_BATCH_SIZE = 512
DEFAULT_EVAL_BATCH_SIZE = 512
DEFAULT_SEED = 2025
MIN_BATCH_SIZE_FLOOR = 64

RESULT_FILENAME = "trial_result.json"
CONFIG_FILENAME = "config.json"


def get_model_class(model_name: str):
    from recbole.model.general_recommender import (
        MultiVAE,
        BPR,
        LightGCN,
        Pop,
        Random,
    )
    
    model_map = {
        "MultiVAE": MultiVAE,
        "BPR": BPR,
        "LightGCN": LightGCN,
        "Pop": Pop,
        "Random": Random,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    return model_map[model_name]


@dataclass
class TrialConfig:
    """for a single trial"""
    trial_id: int
    model_name: str
    dataset_name: str
    split: str
    params: Dict[str, Any]
    project_root: str
    output_root: str
    params_hash: str = field(default="")
    
    def __post_init__(self):
        if not self.params_hash:
            self.params_hash = self._compute_params_hash()
    
    def _compute_params_hash(self) -> str:
        params_str = json.dumps(self.params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:12]
    
    def get_trial_name(self) -> str:
        return (
            f"{self.model_name}_{self.dataset_name}_"
            f"{self.split}_trial{self.trial_id}_{self.params_hash}"
        )


@dataclass
class TrialResult:
    trial_id: int
    model_name: str
    dataset_name: str
    split: str
    params_hash: str
    success: bool
    valid_score: Optional[float] = None
    valid_result: Optional[Dict[str, float]] = None
    test_result: Optional[Dict[str, float]] = None
    config_dict: Optional[Dict[str, Any]] = None
    model_dir: Optional[str] = None
    gpu_id: Optional[str] = None
    error_message: Optional[str] = None
    attempts: int = 1
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialResult":
        return cls(**data)


DEFAULT_SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "MultiVAE": {
        "learning_rate": [3e-3, 1e-3, 3e-4],
        "mlp_hidden_size": [[200, 600]],
        "reg_weight": [1e-6],
        "dropout_prob": [0.0, 0.3],
        "beta": [0.05, 0.1, 0.2, 0.3],
        "total_anneal_steps": [20000],
        "train_batch_size": [512],
        "seed": [42]
    },
    "LightGCN": {
        "learning_rate": [1e-3, 1e-4],
        "embedding_size": [64, 128],
        "n_layers": [2],
        "reg_weight": [1e-6, 1e-5, 1e-4],
        "dropout_prob": [0.1, 0.3],
        "train_batch_size": [1024],
        "seed": [42],
    },
}

def load_search_space(
    model_name: str,
    search_space_file: Optional[str] = None,
) -> Dict[str, List[Any]]:
    if search_space_file and os.path.exists(search_space_file):
        with open(search_space_file, "r") as f:
            custom_spaces = yaml.safe_load(f)
            if model_name in custom_spaces:
                return custom_spaces[model_name]
    
    return DEFAULT_SEARCH_SPACES.get(model_name, {})


def iter_param_grid(space: Dict[str, List[Any]]):
    """Generate all hyperparameter combinations."""
    if not space:
        yield {}
        return
    
    keys = list(space.keys())
    values_list = [space[k] for k in keys]
    
    for combo in itertools.product(*values_list):
        yield dict(zip(keys, combo))


def count_param_combinations(space: Dict[str, List[Any]]) -> int:
    """Count total number of parameter combinations."""
    if not space:
        return 1
    
    count = 1
    for values in space.values():
        count *= len(values)
    return count


class GPUManager:
    def __init__(self, gpu_ids: List[str], manager: mp.Manager):
        self.gpu_queue = manager.Queue()
        self.gpu_ids = gpu_ids
        
        for gpu_id in gpu_ids:
            self.gpu_queue.put(gpu_id)
    
    def acquire(self, timeout: float = 300.0) -> Optional[str]:
        try:
            return self.gpu_queue.get(timeout=timeout)
        except Exception:
            return None
    
    def release(self, gpu_id: str):
        if gpu_id:
            self.gpu_queue.put(gpu_id)


def get_trial_result_path(trial_config: TrialConfig) -> str:
    trial_name = trial_config.get_trial_name()
    model_dir = os.path.join(
        trial_config.output_root,
        trial_config.dataset_name,
        trial_config.model_name,
        "checkpoints",
        trial_name,
    )
    return os.path.join(model_dir, RESULT_FILENAME)


def is_trial_completed(trial_config: TrialConfig) -> Tuple[bool, Optional[TrialResult]]:
    result_path = get_trial_result_path(trial_config)
    
    if not os.path.exists(result_path):
        return False, None
    
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
        result = TrialResult.from_dict(data)
        return True, result
    except Exception:
        return False, None


def save_trial_result(result: TrialResult, trial_config: TrialConfig):
    """Save trial result to disk."""
    result_path = get_trial_result_path(trial_config)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def build_base_config(
    project_root: str,
    dataset_name: str,
    split: str,
    device: str,
    save_root: str,
    log_root: str,
    gpu_id: int = 0,
    epochs: int = DEFAULT_EPOCHS,
    stopping_step: int = DEFAULT_STOPPING_STEP,
) -> Dict[str, Any]:
    """
    Build base RecBole configuration.
    """
    data_path = os.path.join(project_root, "dataset", dataset_name, split, "recbole")
    use_gpu = device.startswith("cuda")
    
    return {
        # Data paths
        "data_path": data_path + "/",
        "benchmark_filename": ["train", "valid", "test"],
        
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating",
        "TIME_FIELD": "timestamp",
        "load_col": {
            "inter": ["user_id", "item_id", "rating", "timestamp"],
        },
        
        "epochs": epochs,
        "stopping_step": stopping_step,
        "train_batch_size": DEFAULT_TRAIN_BATCH_SIZE,
        "eval_batch_size": DEFAULT_EVAL_BATCH_SIZE,
        
        "seed": DEFAULT_SEED,
        "reproducibility": True,
        "show_progress": False,
        "use_gpu": use_gpu,
        "gpu_id": gpu_id if use_gpu else -1,
        "device": device,
        
        "topk": [10, 20, 50, 100],
        "valid_metric": "Recall@20",
        
        "log_root": log_root,
        "checkpoint_dir": save_root,
        "save_dataset": False,
        "save_dataloaders": False,
    }


def setup_trial_logging(log_file_path: str, trial_config: TrialConfig, gpu_id: Optional[str]):
    """
    Setup logging for a single trial.
    """
    logger = logging.getLogger(f"trial_{trial_config.trial_id}")
    logger.setLevel(logging.INFO)
    
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Log trial information
    logger.info("=" * 80)
    logger.info(f"Trial {trial_config.trial_id}: {trial_config.model_name} on {trial_config.dataset_name}")
    logger.info(f"Split: {trial_config.split}")
    logger.info(f"GPU: {gpu_id if gpu_id else 'CPU'}")
    logger.info(f"Parameters: {json.dumps(trial_config.params, indent=2)}")
    logger.info("=" * 80)


def cleanup_memory(device: Optional[str] = None):
    gc.collect()
    
    # CUDA cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if device and device.startswith("cuda"):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass


def run_with_memory_cleanup(func):
    """Decorator to ensure memory cleanup after function execution."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            cleanup_memory()
    return wrapper


def setup_process_environment(gpu_id: Optional[str]) -> Tuple[str, int]:
    if gpu_id is None or gpu_id == "":
        return "cpu", -1

    gid = int(gpu_id)
    if not torch.cuda.is_available():
        return "cpu", -1

    torch.cuda.set_device(gid)
    return f"cuda:{gid}", gid


def run_single_trial_impl(
    trial_config: TrialConfig,
    gpu_id: Optional[str],
    max_oom_retries: int = 3,
    batch_shrink: float = 0.5,
    min_batch_size: int = MIN_BATCH_SIZE_FLOOR,
) -> TrialResult:
    """
    Execute a single trial.
    """
    start_time = time.time()
    
    device, recbole_gpu_id = setup_process_environment(gpu_id)
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.trainer import Trainer
    from recbole.utils import init_seed, init_logger
    trial_name = trial_config.get_trial_name()
    
    model_dir = os.path.join(
        trial_config.output_root,
        trial_config.dataset_name,
        trial_config.model_name,
        "checkpoints",
        trial_name,
    )
    log_dir = os.path.join(
        trial_config.output_root,
        trial_config.dataset_name,
        trial_config.model_name,
        "logs",
        trial_name,
    )
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    base_config = build_base_config(
        project_root=trial_config.project_root,
        dataset_name=trial_config.dataset_name,
        split=trial_config.split,
        device=device,
        save_root=os.path.dirname(model_dir),
        log_root=os.path.dirname(log_dir),
        gpu_id=recbole_gpu_id,
    )
    
    # Get initial batch sizes
    base_train_bs = trial_config.params.get(
        "train_batch_size",
        base_config.get("train_batch_size", DEFAULT_TRAIN_BATCH_SIZE),
    )
    base_eval_bs = base_config.get("eval_batch_size", DEFAULT_EVAL_BATCH_SIZE)
    
    current_train_bs = base_train_bs
    current_eval_bs = base_eval_bs
    
    for attempt in range(1, max_oom_retries + 1):
        cleanup_memory(device)
        
        try:
            config_dict = dict(base_config)
            config_dict.update(trial_config.params)
            config_dict.update({
                "device": device,
                "gpu_id": recbole_gpu_id,
                "checkpoint_dir": model_dir,
                "log_root": log_dir,
                "train_batch_size": int(current_train_bs),
                "eval_batch_size": int(current_eval_bs),
            })
            
            config = Config(
                model=trial_config.model_name,
                dataset=trial_config.dataset_name,
                config_dict=config_dict,
            )
            
            init_seed(config["seed"], config["reproducibility"])
            
            log_file_path = os.path.join(log_dir, f"{trial_name}.log")
            setup_trial_logging(log_file_path, trial_config, gpu_id)
            
            config_dict["log_file"] = log_file_path
            
            init_logger(config)
            
            trial_logger = logging.getLogger(f"trial_{trial_config.trial_id}")
            
            trial_logger.info("Loading dataset...")
            trial_logger.info(f"Data path: {config['data_path']}")
            trial_logger.info(f"Benchmark files: {config['benchmark_filename']}")
            trial_logger.info(f"Fields to load: {config['load_col']}")
            
            recbole_dataset = create_dataset(config)
            
            # Log dataset statistics
            trial_logger.info("=" * 80)
            trial_logger.info("Dataset Statistics:")
            trial_logger.info(f"  Number of users: {recbole_dataset.user_num}")
            trial_logger.info(f"  Number of items: {recbole_dataset.item_num}")
            trial_logger.info(f"  Number of interactions: {recbole_dataset.inter_num}")
            if recbole_dataset.user_num > 0 and recbole_dataset.item_num > 0:
                sparsity = (1 - recbole_dataset.inter_num / (recbole_dataset.user_num * recbole_dataset.item_num)) * 100
                trial_logger.info(f"  Sparsity: {sparsity:.2f}%")
            trial_logger.info("=" * 80)
            
            train_data, valid_data, test_data = data_preparation(config, recbole_dataset)
            
            # Log data split statistics
            trial_logger.info("Data Split Statistics:")
            try:
                # Get interaction counts from dataloaders
                train_count = len(train_data.dataset) if hasattr(train_data, 'dataset') else 'N/A'
                valid_count = len(valid_data.dataset) if hasattr(valid_data, 'dataset') else 'N/A'
                test_count = len(test_data.dataset) if hasattr(test_data, 'dataset') else 'N/A'
                
                trial_logger.info(f"  Train set: {train_count} interactions")
                trial_logger.info(f"  Valid set: {valid_count} interactions")
                trial_logger.info(f"  Test set: {test_count} interactions")
                
                # Log sample data from train set if available
                if hasattr(train_data, 'dataset') and hasattr(train_data.dataset, 'inter_feat'):
                    inter_feat = train_data.dataset.inter_feat
                    if inter_feat is not None and len(inter_feat) > 0:
                        trial_logger.info("Sample training interactions (first 5):")
                        sample_size = min(5, len(inter_feat))
                        user_field = config['USER_ID_FIELD']
                        item_field = config['ITEM_ID_FIELD']
                        rating_field = config.get('RATING_FIELD', None)
                        time_field = config.get('TIME_FIELD', None)
                        
                        try:
                            if hasattr(inter_feat, user_field):
                                user_ids = inter_feat[user_field][:sample_size]
                                item_ids = inter_feat[item_field][:sample_size]
                                
                                if hasattr(user_ids, 'cpu'):
                                    user_ids = user_ids.cpu().numpy()
                                elif hasattr(user_ids, 'values'):
                                    user_ids = user_ids.values
                                elif hasattr(user_ids, 'tolist'):
                                    user_ids = user_ids.tolist()
                                    
                                if hasattr(item_ids, 'cpu'):
                                    item_ids = item_ids.cpu().numpy()
                                elif hasattr(item_ids, 'values'):
                                    item_ids = item_ids.values
                                elif hasattr(item_ids, 'tolist'):
                                    item_ids = item_ids.tolist()
                                
                                ratings = None
                                timestamps = None
                                if rating_field and hasattr(inter_feat, rating_field):
                                    ratings = inter_feat[rating_field][:sample_size]
                                    if hasattr(ratings, 'cpu'):
                                        ratings = ratings.cpu().numpy()
                                    elif hasattr(ratings, 'values'):
                                        ratings = ratings.values
                                    elif hasattr(ratings, 'tolist'):
                                        ratings = ratings.tolist()
                                        
                                if time_field and hasattr(inter_feat, time_field):
                                    timestamps = inter_feat[time_field][:sample_size]
                                    if hasattr(timestamps, 'cpu'):
                                        timestamps = timestamps.cpu().numpy()
                                    elif hasattr(timestamps, 'values'):
                                        timestamps = timestamps.values
                                    elif hasattr(timestamps, 'tolist'):
                                        timestamps = timestamps.tolist()
                                
                                # Log samples
                                for i in range(sample_size):
                                    user_id = user_ids[i] if isinstance(user_ids, (list, tuple)) or hasattr(user_ids, '__getitem__') else 'N/A'
                                    item_id = item_ids[i] if isinstance(item_ids, (list, tuple)) or hasattr(item_ids, '__getitem__') else 'N/A'
                                    rating = ratings[i] if ratings is not None and (isinstance(ratings, (list, tuple)) or hasattr(ratings, '__getitem__')) else 'N/A'
                                    timestamp = timestamps[i] if timestamps is not None and (isinstance(timestamps, (list, tuple)) or hasattr(timestamps, '__getitem__')) else 'N/A'
                                    trial_logger.info(f"  User {user_id}, Item {item_id}, Rating {rating}, Time {timestamp}")
                        except Exception as e:
                            trial_logger.debug(f"Could not log sample data: {e}")
                
            except Exception as e:
                trial_logger.warning(f"Could not log detailed data statistics: {e}")
                import traceback
                trial_logger.debug(traceback.format_exc())
            
            trial_logger.info("Dataset loaded successfully.")
            
            model_class = get_model_class(trial_config.model_name)
            model = model_class(config, train_data.dataset).to(config["device"])
            
            trainer = Trainer(config, model)
            
            trial_logger = logging.getLogger(f"trial_{trial_config.trial_id}")
            
            # Training
            trial_logger.info("Starting training...")
            best_valid_score, best_valid_result = trainer.fit(
                train_data, valid_data
            )
            trial_logger.info(f"Training completed. Best valid score: {best_valid_score:.6f}")
            trial_logger.info(f"Best valid metrics: {json.dumps(best_valid_result, indent=2)}")
            
            # Evaluation
            trial_logger.info("Starting evaluation on test set...")
            best_test_result = trainer.evaluate(
                test_data, load_best_model=True
            )
            trial_logger.info(f"Test metrics: {json.dumps(best_test_result, indent=2)}")
            
            # Log final summary
            trial_logger.info("=" * 80)
            trial_logger.info("Trial completed successfully!")
            trial_logger.info(f"Best valid score: {best_valid_score:.6f}")
            trial_logger.info(f"Duration: {time.time() - start_time:.2f} seconds")
            trial_logger.info("=" * 80)
            
            # Save actual configuration used
            actual_config = {
                k: v for k, v in config.final_config_dict.items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool, list, dict, type(None)))
            }
            
            config_save_path = os.path.join(model_dir, CONFIG_FILENAME)
            with open(config_save_path, "w") as f:
                json.dump(actual_config, f, indent=2, default=str)
            
            del model, trainer, train_data, valid_data, test_data, recbole_dataset
            cleanup_memory(device)
            
            duration = time.time() - start_time
            
            result = TrialResult(
                trial_id=trial_config.trial_id,
                model_name=trial_config.model_name,
                dataset_name=trial_config.dataset_name,
                split=trial_config.split,
                params_hash=trial_config.params_hash,
                success=True,
                valid_score=float(best_valid_score),
                valid_result={k: float(v) for k, v in best_valid_result.items()},
                test_result={k: float(v) for k, v in best_test_result.items()},
                config_dict=actual_config,
                model_dir=model_dir,
                gpu_id=gpu_id,
                attempts=attempt,
                duration_seconds=duration,
            )
            
            # Save result
            save_trial_result(result, trial_config)
            
            return result
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            trial_logger = logging.getLogger(f"trial_{trial_config.trial_id}")
            
            if "out of memory" in error_msg or "cuda" in error_msg:
                trial_logger.warning(f"OOM error on attempt {attempt}: {str(e)}")
                trial_logger.info(f"Current batch sizes: train={current_train_bs}, eval={current_eval_bs}")
                cleanup_memory(device)
                
                new_train_bs = int(current_train_bs * batch_shrink)
                new_eval_bs = int(current_eval_bs * batch_shrink)
                
                if new_train_bs < min_batch_size:
                    duration = time.time() - start_time
                    trial_logger.error(f"OOM: batch size below minimum ({min_batch_size}). Trial failed.")
                    return TrialResult(
                        trial_id=trial_config.trial_id,
                        model_name=trial_config.model_name,
                        dataset_name=trial_config.dataset_name,
                        split=trial_config.split,
                        params_hash=trial_config.params_hash,
                        success=False,
                        error_message=f"OOM: batch size below minimum ({min_batch_size})",
                        attempts=attempt,
                        duration_seconds=duration,
                        gpu_id=gpu_id,
                    )
                
                trial_logger.info(f"Retrying with reduced batch sizes: train={new_train_bs}, eval={new_eval_bs}")
                current_train_bs = new_train_bs
                current_eval_bs = new_eval_bs
                continue
            else:
                # Other runtime error
                duration = time.time() - start_time
                trial_logger.error(f"RuntimeError on attempt {attempt}: {str(e)}")
                cleanup_memory(device)
                
                return TrialResult(
                    trial_id=trial_config.trial_id,
                    model_name=trial_config.model_name,
                    dataset_name=trial_config.dataset_name,
                    split=trial_config.split,
                    params_hash=trial_config.params_hash,
                    success=False,
                    error_message=f"RuntimeError: {str(e)}",
                    attempts=attempt,
                    duration_seconds=duration,
                    gpu_id=gpu_id,
                )
                
        except Exception as e:
            duration = time.time() - start_time
            trial_logger = logging.getLogger(f"trial_{trial_config.trial_id}")
            trial_logger.error(f"Exception on attempt {attempt}: {type(e).__name__}: {str(e)}", exc_info=True)
            cleanup_memory(device)
            
            return TrialResult(
                trial_id=trial_config.trial_id,
                model_name=trial_config.model_name,
                dataset_name=trial_config.dataset_name,
                split=trial_config.split,
                params_hash=trial_config.params_hash,
                success=False,
                error_message=f"{type(e).__name__}: {str(e)}",
                attempts=attempt,
                duration_seconds=duration,
                gpu_id=gpu_id,
            )
    
    duration = time.time() - start_time
    return TrialResult(
        trial_id=trial_config.trial_id,
        model_name=trial_config.model_name,
        dataset_name=trial_config.dataset_name,
        split=trial_config.split,
        params_hash=trial_config.params_hash,
        success=False,
        error_message="Max OOM retries exceeded",
        attempts=max_oom_retries,
        duration_seconds=duration,
        gpu_id=gpu_id,
    )


def worker_run_trial(
    trial_config: TrialConfig,
    gpu_queue: mp.Queue,
    max_oom_retries: int,
    batch_shrink: float,
    min_batch_size: int,
    gpu_timeout: float = 300.0,
) -> TrialResult:
    gpu_id = None
    
    try:
        try:
            gpu_id = gpu_queue.get(timeout=gpu_timeout)
        except Exception:
            return TrialResult(
                trial_id=trial_config.trial_id,
                model_name=trial_config.model_name,
                dataset_name=trial_config.dataset_name,
                split=trial_config.split,
                params_hash=trial_config.params_hash,
                success=False,
                error_message="Timeout waiting for GPU",
            )
        
        result = run_single_trial_impl(
            trial_config=trial_config,
            gpu_id=gpu_id,
            max_oom_retries=max_oom_retries,
            batch_shrink=batch_shrink,
            min_batch_size=min_batch_size,
        )
        
        return result
        
    finally:
        if gpu_id is not None:
            try:
                gpu_queue.put(gpu_id)
            except Exception:
                pass
        
        cleanup_memory()


def generate_all_trials(
    models: List[str],
    datasets: List[str],#
    split: str,
    project_root: str,
    output_root: str,
    search_space_file: Optional[str] = None,
) -> List[TrialConfig]:
    all_trials: List[TrialConfig] = []
    trial_id = 0
    
    for dataset_name in datasets:
        for model_name in models:
            # Validate model
            try:
                _ = get_model_class(model_name)
            except ValueError as e:
                print(f"[WARN] {e}, skipping")
                continue
            
            # Get search space
            search_space = load_search_space(model_name, search_space_file)
            
            # Generate trials
            for params in iter_param_grid(search_space):
                trial_id += 1
                
                trial = TrialConfig(
                    trial_id=trial_id,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    split=split,
                    params=params,
                    project_root=project_root,
                    output_root=output_root,
                )
                
                all_trials.append(trial)
    
    return all_trials


# Parallel Execution
def run_trials_parallel(
    trials: List[TrialConfig],
    gpu_ids: List[str],
    max_workers: int,
    max_oom_retries: int,
    batch_shrink: float,
    min_batch_size: int,
    resume: bool = True,
    trials_per_gpu: int = 1,
) -> List[TrialResult]:

    print(f"\n{'=' * 80}")
    print(f"EXECUTING {len(trials)} TRIALS")
    print(f"Max workers: {max_workers}")
    print(f"GPUs: {gpu_ids if gpu_ids else 'CPU-only'}")
    print(f"Trials per GPU: {trials_per_gpu}")
    print(f"Resume mode: {'ON' if resume else 'OFF'}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")
    
    results: List[TrialResult] = []
    trials_to_run: List[TrialConfig] = []
    
    if resume:
        for trial in trials:
            completed, existing_result = is_trial_completed(trial)
            if completed and existing_result:
                print(
                    f"[SKIP] Trial {trial.trial_id} already completed: "
                    f"{trial.model_name} on {trial.dataset_name}"
                )
                results.append(existing_result)
            else:
                trials_to_run.append(trial)
        
        print(f"\nSkipped {len(results)} completed trials")
        print(f"Running {len(trials_to_run)} remaining trials\n")
    else:
        trials_to_run = trials
    
    if not trials_to_run:
        print("All trials already completed!")
        return results
    
    start_time = time.time()
    
    manager = Manager()
    gpu_queue = manager.Queue()
    
    effective_gpus = gpu_ids if gpu_ids else [""]
    slots_per_gpu = max(1, trials_per_gpu) if effective_gpus and effective_gpus[0] != "" else 1
    for slot_idx in range(len(effective_gpus) * slots_per_gpu):
        gpu_queue.put(effective_gpus[slot_idx % len(effective_gpus)])
    
    total_slots = len(effective_gpus) * slots_per_gpu
    effective_workers = min(max_workers, len(trials_to_run), total_slots)
    
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        future_to_trial = {
            executor.submit(
                worker_run_trial,
                trial,
                gpu_queue,
                max_oom_retries,
                batch_shrink,
                min_batch_size,
            ): trial
            for trial in trials_to_run
        }
        
        completed = len(results)
        total = len(trials)
        
        for future in as_completed(future_to_trial):
            trial = future_to_trial[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                
                if result.success:
                    print(
                        f"[{completed:3d}/{total}] ✓ "
                        f"{result.dataset_name:10s} | {result.model_name:10s} | "
                        f"Trial {result.trial_id:3d} | GPU {result.gpu_id or 'CPU':4s} | "
                        f"Score: {result.valid_score:.4f} | "
                        f"Time: {result.duration_seconds:.1f}s"
                    )
                else:
                    print(
                        f"[{completed:3d}/{total}] ✗ "
                        f"{result.dataset_name:10s} | {result.model_name:10s} | "
                        f"Trial {result.trial_id:3d} | GPU {result.gpu_id or 'CPU':4s} | "
                        f"FAILED: {result.error_message}"
                    )
                    
            except Exception as e:
                print(
                    f"[{completed:3d}/{total}] ✗ "
                    f"Trial {trial.trial_id} | "
                    f"EXCEPTION: {repr(e)}"
                )
                
                failed_result = TrialResult(
                    trial_id=trial.trial_id,
                    model_name=trial.model_name,
                    dataset_name=trial.dataset_name,
                    split=trial.split,
                    params_hash=trial.params_hash,
                    success=False,
                    error_message=f"Future exception: {repr(e)}",
                )
                results.append(failed_result)
    
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r.success)
    
    print(f"\n{'=' * 80}")
    print(f"ALL TRIALS COMPLETED")
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"Successful: {successful}/{len(trials)}")
    print(f"Failed: {len(trials) - successful}/{len(trials)}")
    print(f"{'=' * 80}\n")
    
    return results


def save_results_summary(
    results: List[TrialResult],
    datasets: List[str],
    models: List[str],
    split: str,
    output_root: str,
):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "datasets": datasets,
        "models": models,
        "split": split,
        "total_trials": len(results),
        "successful_trials": sum(1 for r in results if r.success),
        "failed_trials": sum(1 for r in results if not r.success),
        "total_duration_seconds": sum(r.duration_seconds for r in results),
        "results_by_dataset": {},
    }
    
    for dataset in datasets:
        summary["results_by_dataset"][dataset] = {}
        
        for model in models:
            model_results = [
                r for r in results
                if r.dataset_name == dataset
                and r.model_name == model
                and r.success
            ]
            
            all_model_trials = [
                r for r in results
                if r.dataset_name == dataset
                and r.model_name == model
            ]
            
            if not model_results:
                summary["results_by_dataset"][dataset][model] = {
                    "status": "no_successful_trials",
                    "total_trials": len(all_model_trials),
                    "failed_trials": len(all_model_trials),
                }
                continue
            
            # Find best trial
            best = max(model_results, key=lambda x: x.valid_score or float("-inf"))
            
            # Filter config for saving
            filtered_config = {}
            if best.config_dict:
                exclude_keys = {
                    "data_path", "checkpoint_dir", "log_root", "device",
                    "gpu_id", "use_gpu", "benchmark_filename",
                }
                filtered_config = {
                    k: v for k, v in best.config_dict.items()
                    if k not in exclude_keys
                }
            
            best_info = {
                "status": "success",
                "best_trial_id": best.trial_id,
                "best_params_hash": best.params_hash,
                "best_valid_score": best.valid_score,
                "best_valid_result": best.valid_result,
                "best_test_result": best.test_result,
                "best_model_dir": best.model_dir,
                "total_trials": len(all_model_trials),
                "successful_trials": len(model_results),
                "best_params": filtered_config,
            }
            
            summary["results_by_dataset"][dataset][model] = best_info
            
            # Save individual best config
            model_dir = os.path.join(output_root, dataset, model)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save best config
            best_config_path = os.path.join(
                model_dir, f"best_config_{dataset}_{split}.json"
            )
            with open(best_config_path, "w") as f:
                json.dump(best_info, f, indent=2)
            
            # Save best model path
            best_model_path = os.path.join(
                model_dir, f"best_model_{dataset}_{split}.txt"
            )
            with open(best_model_path, "w") as f:
                f.write((best.model_dir or "") + "\n")
            
            # Save all trials for this model
            all_trials_path = os.path.join(
                model_dir, f"all_trials_{dataset}_{split}.json"
            )
            with open(all_trials_path, "w") as f:
                json.dump(
                    [r.to_dict() for r in all_model_trials],
                    f,
                    indent=2,
                )
    
    # Save overall summary
    summary_path = os.path.join(output_root, f"search_summary_{timestamp}.json")
    os.makedirs(output_root, exist_ok=True)
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    latest_summary_path = os.path.join(output_root, "search_summary_latest.json")
    with open(latest_summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nOverall summary saved to: {summary_path}")
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("SEARCH RESULTS SUMMARY")
    print(f"{'=' * 80}")
    
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        print("-" * 60)
        
        for model in models:
            info = summary["results_by_dataset"][dataset].get(model, {})
            
            if info.get("status") == "success":
                print(
                    f"  {model:12s}: "
                    f"Valid={info['best_valid_score']:.4f}, "
                    f"Trials={info['successful_trials']}/{info['total_trials']}"
                )
                
                # Print top test metrics
                if info.get("best_test_result"):
                    test_metrics = info["best_test_result"]
                    key_metrics = ["recall@20", "ndcg@20", "hit@20"]
                    metric_strs = []
                    for metric in key_metrics:
                        for k, v in test_metrics.items():
                            if metric in k.lower():
                                metric_strs.append(f"{k}={v:.4f}")
                                break
                    if metric_strs:
                        print(f"               Test: {', '.join(metric_strs)}")
            else:
                total = info.get("total_trials", 0)
                print(f"  {model:12s}: No successful trials (0/{total})")
    
    print(f"\n{'=' * 80}")


def validate_inputs(
    project_root: str,
    datasets: List[str],
    split: str,
    gpu_ids: List[str],
    max_parallel: Optional[int],
    trials_per_gpu: int = 1,
) -> Tuple[List[str], int]:
    for dataset in datasets:
        data_path = os.path.join(project_root, "dataset", dataset, split, "recbole")
        if not os.path.isdir(data_path):
            raise FileNotFoundError(
                f"Dataset path not found: {data_path}\n"
                f"Please check --project_root, --datasets, and --split arguments."
            )
    
    cuda_available = torch.cuda.is_available()
    
    if not cuda_available or not gpu_ids:
        if gpu_ids:
            print("[WARN] CUDA not available, falling back to CPU.")
        use_gpu_ids: List[str] = []
        max_workers_final = max_parallel or 1
    else:
        valid_gpu_ids = []
        
        for g in gpu_ids:
            try:
                int(g)
                valid_gpu_ids.append(g)
            except ValueError:
                print(f"[WARN] Invalid GPU ID: {g}")
        
        if not valid_gpu_ids:
            print("[WARN] No valid GPUs found, falling back to CPU.")
            use_gpu_ids = []
            max_workers_final = max_parallel or 1
        else:
            use_gpu_ids = valid_gpu_ids
            max_workers_final = max_parallel or (len(valid_gpu_ids) * max(1, trials_per_gpu))
    
    return use_gpu_ids, max_workers_final


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimized hyperparameter search for RecBole models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU search
  python search_recbole_models.py --datasets ml-1m --split C5_[7,1,2] --models MultiVAE --gpus 0

  # Single GPU, multiple trials in parallel (2 trials on same GPU)
  python search_recbole_models.py --datasets ml-1m --split C5_[7,1,2] --models MultiVAE --gpus 0 --trials_per_gpu 2

  # Multi-GPU search
  python search_recbole_models.py --datasets ml-1m --split C5_[7,1,2] --models MultiVAE,LightGCN --gpus 0,1,2,3 --max_parallel 4


  # Resume interrupted search
  python search_recbole_models.py --datasets ml-1m --split C5_[7,1,2] --models MultiVAE --gpus 0 --resume
        """,
    )
    
    parser.add_argument(
        "--project_root",
        type=str,
        default="..",
        help="Project root directory (default: ..)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Dataset names (e.g., 'ml-1m')",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Split name (e.g., 'C5_[7,1,2]')",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="MultiVAE,BPR,LightGCN",
        help="Comma-separated model names (default: MultiVAE,BPR,LightGCN)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs; empty string for CPU-only (default: 0)",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=None,
        help="Max parallel workers (default: GPUs × trials_per_gpu or 1 if CPU)",
    )
    parser.add_argument(
        "--trials_per_gpu",
        type=int,
        default=1,
        help="Max concurrent trials per GPU (default: 1). >1 allows multiple trials on same GPU (higher OOM risk).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./recbole_results",
        help="Output directory (default: ./recbole_results)",
    )
    parser.add_argument(
        "--max_oom_retries",
        type=int,
        default=3,
        help="Max OOM retry attempts (default: 3)",
    )
    parser.add_argument(
        "--batch_shrink",
        type=float,
        default=0.5,
        help="Batch size shrink factor on OOM (default: 0.5)",
    )
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=MIN_BATCH_SIZE_FLOOR,
        help=f"Minimum batch size (default: {MIN_BATCH_SIZE_FLOOR})",
    )
    parser.add_argument(
        "--search_space_file",
        type=str,
        default=None,
        help="Path to YAML file with custom search spaces",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from completed trials (default: True)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume mode (re-run all trials)",
    )
    
    return parser.parse_args()


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    args = parse_args()
    
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else []
    
    resume = args.resume and not args.no_resume
    
    try:
        gpu_ids, max_workers = validate_inputs(
            project_root=args.project_root,
            datasets=datasets,
            split=args.split,
            gpu_ids=gpu_ids,
            max_parallel=args.max_parallel,
            trials_per_gpu=args.trials_per_gpu,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    print(f"\n{'=' * 80}")
    print("RECBOLE HYPERPARAMETER SEARCH (Fixed Version)")
    print(f"{'=' * 80}")
    print(f"Project root:    {args.project_root}")
    print(f"Datasets:        {datasets}")
    print(f"Split:           {args.split}")
    print(f"Models:          {models}")
    print(f"GPUs:            {gpu_ids if gpu_ids else 'CPU-only'}")
    print(f"Trials per GPU:  {args.trials_per_gpu}")
    print(f"Max workers:     {max_workers}")
    print(f"Output root:     {args.output_root}")
    print(f"Resume mode:     {'ON' if resume else 'OFF'}")
    print(f"OOM retries:     {args.max_oom_retries}")
    print(f"Batch shrink:    {args.batch_shrink}")
    print(f"Min batch size:  {args.min_batch_size}")
    if args.search_space_file:
        print(f"Search space:    {args.search_space_file}")
    print(f"{'=' * 80}\n")
    
    # Generate trials
    all_trials = generate_all_trials(
        models=models,
        datasets=datasets,
        split=args.split,
        project_root=args.project_root,
        output_root=args.output_root,
        search_space_file=args.search_space_file,
    )
    
    print(f"Generated {len(all_trials)} trials:")
    for dataset in datasets:
        dataset_trials = [t for t in all_trials if t.dataset_name == dataset]
        print(f"  {dataset}:")
        for model in models:
            model_trials = [t for t in dataset_trials if t.model_name == model]
            if model_trials:
                search_space = load_search_space(model, args.search_space_file)
                combos = count_param_combinations(search_space)
                print(f"    {model}: {len(model_trials)} trials ({combos} param combinations)")
    print()
    
    # Run trials
    results = run_trials_parallel(
        trials=all_trials,
        gpu_ids=gpu_ids,
        max_workers=max_workers,
        max_oom_retries=args.max_oom_retries,
        batch_shrink=args.batch_shrink,
        min_batch_size=args.min_batch_size,
        resume=resume,
        trials_per_gpu=args.trials_per_gpu,
    )
    
    save_results_summary(
        results=results,
        datasets=datasets,
        models=models,
        split=args.split,
        output_root=args.output_root,
    )
    
    print("\n" + "=" * 80)
    print("SEARCH COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()