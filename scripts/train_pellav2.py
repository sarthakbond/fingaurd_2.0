#!/usr/bin/env python3
"""
FinGuard 2.0 — Pellav2 Stage 1 Fine-Tuning Pipeline
Target Hardware: NVIDIA RTX 4050 Laptop GPU (6GB VRAM), Windows x64

Stage 1 Architecture & Training Principles:
  - Backbone: Wav2Vec2-XLS-R-300M (315,438,720 parameters) — 100% FROZEN in eval mode.
  - Trainable Head: 25-layer softmax attention weights + Linear(1024, 1) head (1,050 parameters).
  - Loss: Binary Cross-Entropy with Logits (BCEWithLogitsLoss).
  - Optimizer: AdamW (lr=1e-4, weight_decay=0.01) with gradient clipping.
  - Precision: AMP float16 with GradScaler for numerical stability.
  - Split Isolation: Strictly train_manifest.json and val_manifest.json.
  - Checkpoint Protection: Base weights at E:\\project models\\pellav2_detector.pt are read-only.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2Model, Wav2Vec2Config

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG_PATH = Path(r"E:\fingaurd2.0\configs\train_pellav2_stage1.yaml")


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_metrics(y_true: np.ndarray, y_logits: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification metrics:
    ROC-AUC, Precision, Recall, Specificity, F1, Balanced Accuracy,
    Accuracy, Confusion Matrix, FPR, and FNR.
    """
    y_probs = 1.0 / (1.0 + np.exp(-y_logits))
    y_pred = (y_probs >= threshold).astype(int)

    # Confusion matrix
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    balanced_acc = float((recall + specificity) / 2.0)
    acc = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC-AUC via rank statistic
    pos_scores = y_probs[y_true == 1]
    neg_scores = y_probs[y_true == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        try:
            u, _ = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")
            roc_auc = float(u / (len(pos_scores) * len(neg_scores)))
        except Exception:
            roc_auc = 0.5
    else:
        roc_auc = 0.5

    return {
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "accuracy": round(acc, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


class Pellav2Detector(nn.Module):
    """Official Pellav2 Detector architecture."""
    def __init__(self, config_id: str = "facebook/wav2vec2-xls-r-300m"):
        super().__init__()
        self.config = Wav2Vec2Config.from_pretrained(config_id)
        self.backbone = Wav2Vec2Model(self.config)
        self.layer_weights = nn.Parameter(torch.zeros(self.config.num_hidden_layers + 1))
        self.head = nn.Linear(self.config.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In Stage 1, backbone parameters are frozen and run under torch.no_grad()
        with torch.no_grad():
            hs = self.backbone(x, output_hidden_states=True).hidden_states
        w = torch.softmax(self.layer_weights, dim=0)
        h = torch.stack(hs).mul(w[:, None, None, None]).sum(0).mean(dim=1)
        return self.head(h).squeeze(-1)


def preprocess_audio(file_path: Path, target_sr: int = 16000, target_length: int = 64000) -> torch.Tensor:
    """Standard Pellav2 preprocessing: 16kHz mono, 4s center-crop/pad, per-clip normalisation."""
    wav, sr = torchaudio.load(file_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        wav = resampler(wav)

    L = wav.shape[-1]
    if L >= target_length:
        start = (L - target_length) // 2
        wav = wav[:, start:start + target_length]
    else:
        wav = torch.nn.functional.pad(wav, (0, target_length - L))

    wav = (wav - wav.mean()) / (wav.std() + 1e-7)
    return wav.squeeze(0).to(torch.float32)


class Pellav2Dataset(Dataset):
    """Manifest-driven PyTorch Dataset for Pellav2 training and validation."""
    def __init__(self, manifest_path: Path, dataset_root: Path, target_sr: int = 16000, target_length: int = 64000):
        self.dataset_root = dataset_root
        self.target_sr = target_sr
        self.target_length = target_length
        self.resamplers: Dict[int, T.Resample] = {}

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = data["samples"]

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, entry: Dict[str, Any]) -> Path:
        if entry["source_dataset"] == "garystafford":
            return self.dataset_root / "garystafford" / entry["filename"]
        else:
            return self.dataset_root / "in_the_wild" / entry["filename"].replace("release_in_the_wild/", "")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.samples[idx]
        file_path = self._resolve_path(entry)
        label_val = 1.0 if entry["label"] == "fake" else 0.0

        try:
            wav, sr = torchaudio.load(file_path)
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            if sr != self.target_sr:
                if sr not in self.resamplers:
                    self.resamplers[sr] = T.Resample(sr, self.target_sr)
                wav = self.resamplers[sr](wav)

            L = wav.shape[-1]
            if L >= self.target_length:
                start = (L - self.target_length) // 2
                wav = wav[:, start:start + self.target_length]
            else:
                wav = torch.nn.functional.pad(wav, (0, self.target_length - L))

            wav = (wav - wav.mean()) / (wav.std() + 1e-7)
            wav = wav.squeeze(0).to(torch.float32)
        except Exception as ex:
            # Fallback zero-tensor in the event of I/O glitch
            print(f"[WARN] Error reading {file_path}: {ex}", file=sys.stderr)
            wav = torch.zeros(self.target_length, dtype=torch.float32)

        return wav, torch.tensor(label_val, dtype=torch.float32)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, use_amp: bool) -> Tuple[float, Dict[str, Any]]:
    """Evaluates model over dataloader and returns average loss and metrics."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            all_logits.extend(logits.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / len(all_targets) if all_targets else 0.0
    metrics = compute_metrics(np.array(all_targets), np.array(all_logits), threshold=0.50)
    metrics["loss"] = round(avg_loss, 4)
    return avg_loss, metrics


def train_stage1(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Deterministic Seeding
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    print("=" * 78)
    print(" FinGuard 2.0 — Pellav2 Stage 1 Fine-Tuning Execution")
    print("=" * 78)
    print(f"Seed:               {seed}")
    print(f"Batch Size:         {cfg['training']['batch_size']}")
    print(f"Learning Rate:      {cfg['training']['learning_rate']}")
    print(f"Epochs:             {cfg['training']['epochs']}")
    print(f"Mixed Precision:    {cfg['training']['use_amp']}")

    # 2. Strict Asset & Checkpoint Protection
    base_ckpt = Path(cfg["model"]["base_checkpoint"])
    if not base_ckpt.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt}")
    orig_hash = compute_sha256(base_ckpt)
    print(f"Base Checkpoint:    {base_ckpt} (SHA-256: {orig_hash[:16]}...)")

    output_dir = Path(cfg["experiments"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output Directory:   {output_dir}")

    # 3. Split Isolation Enforcement
    train_mf = Path(cfg["data"]["train_manifest"])
    val_mf = Path(cfg["data"]["val_manifest"])
    dataset_root = Path(cfg["data"]["dataset_root"])

    print(f"\nLoading Training Manifest:   {train_mf}")
    print(f"Loading Validation Manifest: {val_mf}")
    train_dataset = Pellav2Dataset(train_mf, dataset_root, cfg["data"]["target_sample_rate"], cfg["data"]["target_length_samples"])
    val_dataset = Pellav2Dataset(val_mf, dataset_root, cfg["data"]["target_sample_rate"], cfg["data"]["target_length_samples"])
    print(f"Train Dataset Samples: {len(train_dataset)}")
    print(f"Val Dataset Samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=(torch.cuda.is_available()),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=(torch.cuda.is_available()),
    )

    # 4. Model Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model = Pellav2Detector(cfg["model"]["wav2vec2_config_id"]).to(device)
    state = torch.load(str(base_ckpt), map_location=device)
    model.load_state_dict(state, strict=True)
    print("[PASS] Checkpoint state loaded into Detector.")

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    trainable_params = [model.layer_weights, *model.head.parameters()]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable Parameters: {trainable_count:,} (layer_weights: 25, head: 1,025)")

    # 5. Optimizer, Loss, and Scaler
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    pos_weight = torch.tensor([float(cfg["training"].get("pos_weight", 1.0))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler(enabled=(cfg["training"]["use_amp"] and device.type == "cuda"))

    # Initial Baseline Validation
    print("\n[EVAL] Evaluating baseline checkpoint performance on validation set...")
    init_loss, init_metrics = evaluate(model, val_loader, criterion, device, cfg["training"]["use_amp"])
    print(f"  Baseline Val Loss:     {init_metrics['loss']:.4f}")
    print(f"  Baseline Val ROC-AUC:  {init_metrics['roc_auc']:.4f}")
    print(f"  Baseline Val Bal-Acc:  {init_metrics['balanced_accuracy']:.4f}")
    print(f"  Baseline Val F1:       {init_metrics['f1']:.4f}")

    # 6. Training Loop
    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"]["early_stopping_patience"])
    best_roc_auc = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    print("\n" + "=" * 78)
    print(" STARTING STAGE 1 TRAINING LOOP")
    print("=" * 78)

    for epoch in range(1, epochs + 1):
        model.layer_weights.requires_grad_(True)
        model.head.train()
        model.backbone.eval()

        epoch_loss = 0.0
        start_t = time.time()

        try:
            for step, (bx, by) in enumerate(train_loader):
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()

                if cfg["training"]["use_amp"] and device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(bx)
                        loss = criterion(logits, by)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(cfg["training"]["gradient_clip_norm"]))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(cfg["training"]["gradient_clip_norm"]))
                    optimizer.step()

                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Encountered NaN or Inf loss at epoch {epoch}, step {step}: {loss.item()}")

                if (step + 1) % 120 == 0 or (step + 1) == len(train_loader):
                    print(f"  Step [{step+1:03d}/{len(train_loader):03d}] — Step Loss: {loss.item():.4f}", flush=True)

                epoch_loss += loss.item() * len(by)
        except torch.cuda.OutOfMemoryError as oom:
            print(f"\n[ERROR] CUDA Out-of-Memory detected: {oom}", file=sys.stderr, flush=True)
            print("Consider reducing batch_size in configuration.", file=sys.stderr, flush=True)
            raise oom

        avg_train_loss = epoch_loss / len(train_dataset)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, cfg["training"]["use_amp"])
        elapsed = time.time() - start_t

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] — "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
            f"Bal-Acc: {val_metrics['balanced_accuracy']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"FPR: {val_metrics['fpr']:.4f} | "
            f"FNR: {val_metrics['fnr']:.4f}",
            flush=True
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "elapsed_sec": round(elapsed, 2),
        }
        history.append(epoch_record)

        # Checkpoint Saving & Early Stopping
        current_auc = val_metrics["roc_auc"]
        if current_auc > best_roc_auc:
            best_roc_auc = current_auc
            best_epoch = epoch
            patience_counter = 0

            best_ckpt_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg,
                "base_checkpoint_hash": orig_hash,
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val ROC-AUC: {best_roc_auc:.4f})", flush=True)
        else:
            patience_counter += 1
            print(f"  --> Early stopping counter: {patience_counter}/{patience}", flush=True)
            if patience_counter >= patience:
                print(f"[EARLY STOPPING] Validation ROC-AUC did not improve for {patience} consecutive epochs.", flush=True)
                break

    # 7. Experiment Artifacts & Metadata Summary
    total_train_time = time.time() - (start_t - elapsed)
    peak_vram_mb = (torch.cuda.max_memory_allocated(0) / (1024 ** 2)) if torch.cuda.is_available() else 0.0

    metadata = {
        "title": "FinGuard 2.0 Pellav2 Stage 1 Fine-Tuning Experiment",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_checkpoint": str(base_ckpt),
        "base_checkpoint_sha256": orig_hash,
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_roc_auc,
        "baseline_val_metrics": init_metrics,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "total_runtime_sec": round(total_train_time, 2),
        "history": history,
        "software_versions": {
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "transformers": Wav2Vec2Model.__module__.split(".")[0],
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        },
        "config": cfg,
    }
    meta_path = output_dir / "experiment_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[DONE] Experiment metadata recorded: {meta_path}", flush=True)

    # 8. Render final side-by-side comparison table
    best_record = torch.load(str(output_dir / "best_model.pt"), map_location=device)
    from scripts.evaluate_pellav2 import print_comparison_table
    print_comparison_table(init_metrics, best_record["val_metrics"], split_name="Validation")
    print(f"Best Epoch:           {best_epoch}")
    print(f"Peak VRAM:            {peak_vram_mb:.0f} MB")
    print(f"Best Checkpoint:      {output_dir / 'best_model.pt'}")
    print("=" * 78, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Pellav2 Stage 1 Fine-Tuning Script")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to YAML training configuration")
    parser.add_argument("--dry-run", action="store_true", help="Perform preflight dry-run check without training")
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.dry_run:
        print("[DRY-RUN] Validating training setup only...")
        from scripts.validate_training_setup import main as run_validation
        run_validation()
        return

    train_stage1(config_path)


if __name__ == "__main__":
    main()
