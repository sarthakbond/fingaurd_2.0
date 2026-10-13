#!/usr/bin/env python3
"""
FinGuard 2.0 — Pellav2 Side-by-Side Model Evaluation Script
Compares the original baseline checkpoint and fine-tuned checkpoint on identical validation audio.

Guarantees:
  - Reports metrics side-by-side at the standard 0.50 threshold.
  - Never reports validation results as final test results.
  - Does NOT access locked_test_manifest.json.
  - Supports separate diagnostic probe evaluation via `--diagnostic`.
"""

import sys
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Config

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_pellav2 import Pellav2Detector, Pellav2Dataset, compute_metrics, evaluate

DEFAULT_CONFIG_PATH = Path(r"E:\fingaurd2.0\configs\train_pellav2_stage1.yaml")


def load_model(checkpoint_path: Path, config_id: str, device: torch.device) -> nn.Module:
    """Loads Pellav2 detector from a raw state_dict or training checkpoint dictionary."""
    model = Pellav2Detector(config_id).to(device)
    raw = torch.load(str(checkpoint_path), map_location=device)
    state_dict = raw.get("model_state_dict", raw)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def print_comparison_table(metrics_base: Dict[str, Any], metrics_ft: Dict[str, Any], split_name: str = "Validation"):
    """Renders formatted side-by-side comparison table."""
    print("\n" + "=" * 80)
    print(f" SIDE-BY-SIDE METRIC COMPARISON ({split_name.upper()} SET — THRESHOLD = 0.50)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Original Baseline':<18} | {'Fine-Tuned Stage 1':<18} | {'Delta':<12}")
    print("-" * 80)

    metric_keys = [
        ("Loss", "loss"),
        ("ROC-AUC", "roc_auc"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("Precision", "precision"),
        ("Recall (Sensitivity)", "recall"),
        ("Specificity (TNR)", "specificity"),
        ("F1-Score", "f1"),
        ("Accuracy", "accuracy"),
        ("False Positive Rate (FPR)", "fpr"),
        ("False Negative Rate (FNR)", "fnr"),
    ]

    for label, key in metric_keys:
        v_base = metrics_base.get(key, 0.0)
        v_ft = metrics_ft.get(key, 0.0)
        delta = v_ft - v_base
        delta_str = f"{delta:+.4f}" if isinstance(delta, float) else "N/A"
        print(f"{label:<25} | {v_base:<18.4f} | {v_ft:<18.4f} | {delta_str:<12}")

    print("-" * 80)
    cm_base = metrics_base["confusion_matrix"]
    cm_ft = metrics_ft["confusion_matrix"]
    print(f"{'Confusion Matrix':<25} | TN={cm_base['tn']}, FP={cm_base['fp']} | TN={cm_ft['tn']}, FP={cm_ft['fp']} |")
    print(f"{'':<25} | FN={cm_base['fn']}, TP={cm_base['tp']} | FN={cm_ft['fn']}, TP={cm_ft['tp']} |")
    print("=" * 80)
    print("NOTE: Validation performance does not guarantee generalisation to out-of-domain distribution.")
    print("Locked test set remains quarantined and was NOT evaluated.")


def evaluate_diagnostic(base_model: nn.Module, ft_model: nn.Module, diag_mf: Path, dataset_root: Path, device: torch.device):
    """Evaluates targeted diagnostic probes individually."""
    print("\n" + "=" * 80)
    print(" TARGETED DIAGNOSTIC PROBE EVALUATION (Quarantined Controls)")
    print("=" * 80)

    with open(diag_mf, "r", encoding="utf-8") as f:
        probes = json.load(f)

    from scripts.train_pellav2 import preprocess_audio
    print(f"{'ID':<12} | {'Label':<6} | {'Probe Type':<25} | {'Base Score':<10} | {'FT Score':<10}")
    print("-" * 80)

    for p in probes:
        file_path = Path(r"E:\fingaurd2.0") / p["path"]
        if not file_path.is_file():
            print(f"{p['id']:<12} | {p['label']:<6} | {p['type']:<25} | [MISSING FILE] | [MISSING FILE]")
            continue

        wav = preprocess_audio(file_path).unsqueeze(0).to(device)
        with torch.no_grad():
            s_base = torch.sigmoid(base_model(wav)).item()
            s_ft = torch.sigmoid(ft_model(wav)).item()

        print(f"{p['id']:<12} | {p['label']:<6} | {p['type']:<25} | {s_base:<10.4f} | {s_ft:<10.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Pellav2 Side-by-Side Model Evaluator")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to YAML training configuration")
    parser.add_argument("--fine-tuned-checkpoint", type=str, default=None, help="Path to fine-tuned checkpoint (.pt)")
    parser.add_argument("--diagnostic", action="store_true", help="Run diagnostic probe evaluation")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" FinGuard 2.0 — Pellav2 Side-by-Side Model Evaluation")
    print("=" * 80)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    base_ckpt = Path(cfg["model"]["base_checkpoint"])
    ft_ckpt = Path(args.fine_tuned_checkpoint) if args.fine_tuned_checkpoint else Path(cfg["experiments"]["output_dir"]) / "best_model.pt"

    print(f"Original Checkpoint:   {base_ckpt}")
    print(f"Fine-Tuned Checkpoint: {ft_ckpt}")

    if not base_ckpt.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt}")
    if not ft_ckpt.is_file():
        print(f"\n[INFO] Fine-tuned checkpoint not found at: {ft_ckpt}")
        print("To run side-by-side evaluation, train a checkpoint first or specify `--fine-tuned-checkpoint`.")
        print("Evaluating baseline checkpoint only...")

    base_model = load_model(base_ckpt, cfg["model"]["wav2vec2_config_id"], device)
    ft_model = load_model(ft_ckpt, cfg["model"]["wav2vec2_config_id"], device) if ft_ckpt.is_file() else None

    # Load Validation Set
    val_mf = Path(cfg["data"]["val_manifest"])
    dataset_root = Path(cfg["data"]["dataset_root"])
    val_dataset = Pellav2Dataset(val_mf, dataset_root, cfg["data"]["target_sample_rate"], cfg["data"]["target_length_samples"])
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    print("\n[EVAL] Running baseline model over validation set...")
    _, base_metrics = evaluate(base_model, val_loader, criterion, device, cfg["training"]["use_amp"])

    if ft_model is not None:
        print("[EVAL] Running fine-tuned model over validation set...")
        _, ft_metrics = evaluate(ft_model, val_loader, criterion, device, cfg["training"]["use_amp"])
        print_comparison_table(base_metrics, ft_metrics, split_name="Validation")
    else:
        print("\n" + "=" * 60)
        print(" BASELINE MODEL VALIDATION RESULTS (THRESHOLD = 0.50)")
        print("=" * 60)
        for k, v in base_metrics.items():
            print(f"  {k:<20}: {v}")
        print("=" * 60)

    if args.diagnostic and ft_model is not None:
        diag_mf = Path(cfg["data"]["diagnostic_manifest"])
        evaluate_diagnostic(base_model, ft_model, diag_mf, dataset_root, device)


if __name__ == "__main__":
    main()
