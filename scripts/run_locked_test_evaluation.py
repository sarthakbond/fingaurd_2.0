#!/usr/bin/env python3
"""
FinGuard 2.0 — Stage 1 Final Locked-Test Evaluation
Evaluates original Pellav2 checkpoint vs. Stage 1 fine-tuned checkpoint on the locked test manifest.

Guarantees & Constraints:
  - Fixed 0.50 threshold (strictly no threshold tuning).
  - Identical audio preprocessing (16kHz mono, 64k crop/pad, per-clip normalisation).
  - Evaluates full 1,200 samples (600 genuine, 600 synthetic).
  - Per-generator breakdown: Kokoro (68), Hume AI (116), Speechify (211), In-the-Wild (205).
  - Diagnostic probes evaluated separately and quarantined.
  - Saves reproducible JSON and CSV evaluation reports.
  - Zero modification to production code or checkpoints.
"""

import os
import sys
import csv
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2Model, Wav2Vec2Config

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_pellav2 import Pellav2Detector, Pellav2Dataset, compute_metrics, preprocess_audio

DATASET_ROOT = Path(r"E:\datasets")
LOCKED_TEST_MANIFEST = DATASET_ROOT / "splits" / "locked_test_manifest.json"
DIAGNOSTIC_MANIFEST = DATASET_ROOT / "splits" / "diagnostic_manifest.json"
BASE_CHECKPOINT = Path(r"E:\project models\pellav2_detector.pt")
FT_CHECKPOINT = Path(r"E:\datasets\experiments\pellav2_stage1\best_model.pt")
OUTPUT_DIR = Path(r"E:\datasets\experiments\pellav2_stage1")


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_detector(checkpoint_path: Path, config_id: str, device: torch.device) -> nn.Module:
    model = Pellav2Detector(config_id).to(device)
    raw = torch.load(str(checkpoint_path), map_location=device)
    state_dict = raw.get("model_state_dict", raw)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def evaluate_model_on_manifest(
    model: nn.Module,
    samples: List[Dict[str, Any]],
    dataset_root: Path,
    device: torch.device,
    batch_size: int = 8,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Runs model over sample list and returns logits, probabilities, and error logs."""
    target_sr = 16000
    target_length = 64000
    resamplers = {}

    all_logits = []
    sample_records = []
    error_logs = []

    model.eval()
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]
        batch_tensors = []

        for entry in batch_samples:
            if entry["source_dataset"] == "garystafford":
                fpath = dataset_root / "garystafford" / entry["filename"]
            else:
                fpath = dataset_root / "in_the_wild" / entry["filename"].replace("release_in_the_wild/", "")

            try:
                wav, sr = torchaudio.load(fpath)
                if wav.shape[0] > 1:
                    wav = torch.mean(wav, dim=0, keepdim=True)
                if sr != target_sr:
                    if sr not in resamplers:
                        resamplers[sr] = T.Resample(sr, target_sr)
                    wav = resamplers[sr](wav)

                L = wav.shape[-1]
                if L >= target_length:
                    start = (L - target_length) // 2
                    wav = wav[:, start : start + target_length]
                else:
                    wav = torch.nn.functional.pad(wav, (0, target_length - L))

                wav = (wav - wav.mean()) / (wav.std() + 1e-7)
                wav = wav.squeeze(0).to(torch.float32)
            except Exception as ex:
                error_logs.append({
                    "sample_id": entry.get("id"),
                    "filename": str(fpath),
                    "error": str(ex)
                })
                wav = torch.zeros(target_length, dtype=torch.float32)

            batch_tensors.append(wav)

        x = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
            else:
                logits = model(x)

        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        all_logits.extend(logits.detach().cpu().numpy().tolist())

    logits_arr = np.array(all_logits, dtype=np.float32)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))
    return logits_arr, probs_arr, error_logs


def compute_slice_metrics(y_true: np.ndarray, y_logits: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    y_probs = 1.0 / (1.0 + np.exp(-y_logits))
    y_pred = (y_probs >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    n_total = len(y_true)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    balanced_acc = float((recall + specificity) / 2.0) if (n_pos > 0 and n_neg > 0) else float((tp + tn) / n_total)
    acc = float((tp + tn) / n_total) if n_total > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC-AUC
    pos_scores = y_probs[y_true == 1]
    neg_scores = y_probs[y_true == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        try:
            u, _ = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")
            roc_auc = float(u / (len(pos_scores) * len(neg_scores)))
        except Exception:
            roc_auc = 0.5
    else:
        roc_auc = None  # Single-class slice

    return {
        "count": n_total,
        "n_real": n_neg,
        "n_fake": n_pos,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A (single class)",
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def evaluate_diagnostic_probes(
    base_model: nn.Module,
    ft_model: nn.Module,
    diag_manifest_path: Path,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Evaluates separate diagnostic probes without using them for model selection."""
    with open(diag_manifest_path, "r", encoding="utf-8") as f:
        probes = json.load(f)

    results = []
    # Known alternative paths for probes
    alt_paths = {
        "Downloads/ElevenLabs_Roger.mpeg": Path(r"C:\Users\sarthak\Downloads\ElevenLabs_2026-10-11T14_57_02_Roger - Laid-Back, Casual, Resonant_pre_sp100_s50_sb75_v4.mp3.mpeg")
    }

    base_model.eval()
    ft_model.eval()

    for p in probes:
        rel_p = p["path"]
        fpath = Path(r"E:\fingaurd2.0") / rel_p
        if not fpath.is_file() and rel_p in alt_paths and alt_paths[rel_p].is_file():
            fpath = alt_paths[rel_p]

        if not fpath.is_file():
            results.append({
                "id": p["id"],
                "label": p["label"],
                "type": p["type"],
                "description": p.get("description", ""),
                "status": "MISSING_FILE",
                "path": str(fpath),
                "base_score": None,
                "ft_score": None,
                "base_pred": None,
                "ft_pred": None,
            })
            continue

        try:
            wav = preprocess_audio(fpath).unsqueeze(0).to(device)
            with torch.no_grad():
                b_logit = base_model(wav).item()
                f_logit = ft_model(wav).item()
                b_score = 1.0 / (1.0 + np.exp(-b_logit))
                f_score = 1.0 / (1.0 + np.exp(-f_logit))

            results.append({
                "id": p["id"],
                "label": p["label"],
                "type": p["type"],
                "description": p.get("description", ""),
                "status": "EVALUATED",
                "path": str(fpath),
                "base_logit": round(b_logit, 4),
                "base_score": round(float(b_score), 4),
                "base_pred": "fake" if b_score >= 0.5 else "real",
                "ft_logit": round(f_logit, 4),
                "ft_score": round(float(f_score), 4),
                "ft_pred": "fake" if f_score >= 0.5 else "real",
            })
        except Exception as ex:
            results.append({
                "id": p["id"],
                "label": p["label"],
                "type": p["type"],
                "description": p.get("description", ""),
                "status": f"ERROR: {ex}",
                "path": str(fpath),
                "base_score": None,
                "ft_score": None,
                "base_pred": None,
                "ft_pred": None,
            })

    return results


def main():
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" FINGUARD 2.0 — STAGE 1 LOCKED-TEST EVALUATION")
    print("=" * 80)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Hashes and Paths
    base_hash = compute_sha256(BASE_CHECKPOINT)
    ft_hash = compute_sha256(FT_CHECKPOINT)
    print(f"\n[1] Checkpoints:")
    print(f"  Baseline:   {BASE_CHECKPOINT} (SHA-256: {base_hash})")
    print(f"  Fine-Tuned: {FT_CHECKPOINT} (SHA-256: {ft_hash})")

    # 2. Load Manifest
    with open(LOCKED_TEST_MANIFEST, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    samples = manifest_data["samples"]
    print(f"\n[2] Locked Test Samples: {len(samples)} (Real: 600, Fake: 600)")

    # 3. Load Models
    print("\n[3] Loading Models...")
    config_id = "facebook/wav2vec2-xls-r-300m"
    base_model = load_detector(BASE_CHECKPOINT, config_id, device)
    ft_model = load_detector(FT_CHECKPOINT, config_id, device)
    print("  [PASS] Both models loaded with strict=True.")

    # 4. Run Evaluations
    print("\n[4] Running Baseline Inference over 1,200 Locked Test Samples...")
    base_logits, base_probs, base_errors = evaluate_model_on_manifest(
        base_model, samples, DATASET_ROOT, device, batch_size=8, use_amp=True
    )
    print(f"  Baseline complete. Errors encountered: {len(base_errors)}")

    print("\n[5] Running Fine-Tuned Stage 1 Inference over 1,200 Locked Test Samples...")
    ft_logits, ft_probs, ft_errors = evaluate_model_on_manifest(
        ft_model, samples, DATASET_ROOT, device, batch_size=8, use_amp=True
    )
    print(f"  Fine-Tuned complete. Errors encountered: {len(ft_errors)}")

    # 5. Build Sample-Level Predictions Table & CSV
    y_true = np.array([1 if s["label"] == "fake" else 0 for s in samples], dtype=int)
    predictions_rows = []
    csv_path = OUTPUT_DIR / "locked_test_predictions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "id", "source_dataset", "filename", "speaker_or_channel", "generator",
            "ground_truth_label", "ground_truth_int",
            "base_logit", "base_prob", "base_pred", "base_correct",
            "ft_logit", "ft_prob", "ft_pred", "ft_correct"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, s in enumerate(samples):
            b_p = float(base_probs[idx])
            f_p = float(ft_probs[idx])
            b_pred = 1 if b_p >= 0.50 else 0
            f_pred = 1 if f_p >= 0.50 else 0
            gt = y_true[idx]

            row = {
                "id": s["id"],
                "source_dataset": s["source_dataset"],
                "filename": s["filename"],
                "speaker_or_channel": s["speaker_or_channel"],
                "generator": s.get("generator", "none_genuine"),
                "ground_truth_label": s["label"],
                "ground_truth_int": gt,
                "base_logit": round(float(base_logits[idx]), 4),
                "base_prob": round(b_p, 4),
                "base_pred": "fake" if b_pred == 1 else "real",
                "base_correct": bool(b_pred == gt),
                "ft_logit": round(float(ft_logits[idx]), 4),
                "ft_prob": round(f_p, 4),
                "ft_pred": "fake" if f_pred == 1 else "real",
                "ft_correct": bool(f_pred == gt),
            }
            writer.writerow(row)
            predictions_rows.append(row)

    print(f"\n[PASS] Saved sample-level predictions to: {csv_path}")

    # 6. Overall Metrics
    overall_base = compute_slice_metrics(y_true, base_logits, threshold=0.50)
    overall_ft = compute_slice_metrics(y_true, ft_logits, threshold=0.50)

    # 7. Slices / Generator Breakdown
    generators = sorted(list(set(s.get("generator", "none_genuine") for s in samples)))
    generator_metrics = {}

    for gen in generators:
        idx_sub = [i for i, s in enumerate(samples) if s.get("generator", "none_genuine") == gen]
        y_sub = y_true[idx_sub]
        b_log_sub = base_logits[idx_sub]
        f_log_sub = ft_logits[idx_sub]

        generator_metrics[gen] = {
            "baseline": compute_slice_metrics(y_sub, b_log_sub, threshold=0.50),
            "fine_tuned": compute_slice_metrics(y_sub, f_log_sub, threshold=0.50),
        }

    # Sources Breakdown
    sources = sorted(list(set(s["source_dataset"] for s in samples)))
    source_metrics = {}
    for src in sources:
        idx_sub = [i for i, s in enumerate(samples) if s["source_dataset"] == src]
        y_sub = y_true[idx_sub]
        b_log_sub = base_logits[idx_sub]
        f_log_sub = ft_logits[idx_sub]
        source_metrics[src] = {
            "baseline": compute_slice_metrics(y_sub, b_log_sub, threshold=0.50),
            "fine_tuned": compute_slice_metrics(y_sub, f_log_sub, threshold=0.50),
        }

    # Unseen Generators Combined Breakdown (Kokoro + Hume AI + Speechify vs GS Genuine Test)
    gs_test_idx = [i for i, s in enumerate(samples) if s["source_dataset"] == "garystafford"]
    y_gs = y_true[gs_test_idx]
    b_gs = base_logits[gs_test_idx]
    f_gs = ft_logits[gs_test_idx]
    unseen_gs_metrics = {
        "baseline": compute_slice_metrics(y_gs, b_gs, threshold=0.50),
        "fine_tuned": compute_slice_metrics(y_gs, f_gs, threshold=0.50),
    }

    # 8. Evaluate Diagnostic Probes
    print("\n[6] Evaluating Separate Diagnostic Probes...")
    diag_results = evaluate_diagnostic_probes(base_model, ft_model, DIAGNOSTIC_MANIFEST, device)

    # 9. JSON Summary Output
    report = {
        "title": "FinGuard 2.0 Stage 1 Locked-Test Final Evaluation Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        },
        "software": {
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "python": sys.version.split()[0],
        },
        "checkpoints": {
            "baseline": {
                "path": str(BASE_CHECKPOINT),
                "sha256": base_hash,
            },
            "fine_tuned_stage1": {
                "path": str(FT_CHECKPOINT),
                "sha256": ft_hash,
            },
        },
        "preprocessing": {
            "sample_rate": 16000,
            "crop_pad_samples": 64000,
            "crop_strategy": "center-crop or zero-pad",
            "normalisation": "per-clip zero-mean unit-variance",
            "threshold": 0.50,
        },
        "overall_evaluation": {
            "total_samples": len(samples),
            "n_real": 600,
            "n_fake": 600,
            "baseline": overall_base,
            "fine_tuned": overall_ft,
        },
        "held_out_unseen_generators_combined": unseen_gs_metrics,
        "per_generator_metrics": generator_metrics,
        "per_source_metrics": source_metrics,
        "inference_errors": {
            "baseline_errors": base_errors,
            "fine_tuned_errors": ft_errors,
        },
        "diagnostic_probes": diag_results,
    }

    json_path = OUTPUT_DIR / "locked_test_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[PASS] Saved full JSON report to: {json_path}")

    # 10. Print Visual Summary Table
    print("\n" + "=" * 80)
    print(" LOCKED-TEST OVERALL COMPARISON (1,200 SAMPLES — FIXED THRESHOLD = 0.50)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Original Baseline':<18} | {'Fine-Tuned Stage 1':<18} | {'Delta':<12}")
    print("-" * 80)
    metric_keys = [
        ("ROC-AUC", "roc_auc"),
        ("Accuracy", "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("Precision", "precision"),
        ("Recall (Sensitivity)", "recall"),
        ("Specificity (TNR)", "specificity"),
        ("F1-Score", "f1"),
        ("False Positive Rate (FPR)", "fpr"),
        ("False Negative Rate (FNR)", "fnr"),
    ]
    for label, k in metric_keys:
        b_v = overall_base[k]
        f_v = overall_ft[k]
        delta = f_v - b_v
        print(f"{label:<25} | {b_v:<18.4f} | {f_v:<18.4f} | {delta:<+12.4f}")
    print("-" * 80)
    cm_b = overall_base["confusion_matrix"]
    cm_f = overall_ft["confusion_matrix"]
    print(f"{'Confusion Matrix':<25} | TN={cm_b['tn']}, FP={cm_b['fp']} | TN={cm_f['tn']}, FP={cm_f['fp']} |")
    print(f"{'':<25} | FN={cm_b['fn']}, TP={cm_b['tp']} | FN={cm_f['fn']}, TP={cm_f['tp']} |")
    print("=" * 80)

    # Print Held-Out Breakdown Table
    print("\n" + "=" * 80)
    print(" HELD-OUT SYNTHETIC GENERATORS ACCURACY (AT THRESHOLD 0.50)")
    print("=" * 80)
    print(f"{'Generator':<25} | {'Count':<6} | {'Baseline Recall':<16} | {'FT Stage 1 Recall':<18} | {'Delta':<10}")
    print("-" * 80)
    target_gens = [
        ("Kokoro (hexgrad_kokoro)", "hexgrad_kokoro"),
        ("Hume AI (hume_ai)", "hume_ai"),
        ("Speechify (speechify)", "speechify"),
        ("In-The-Wild Voice Clone", "in_the_wild_clone"),
    ]
    for label, key in target_gens:
        m_b = generator_metrics[key]["baseline"]
        m_f = generator_metrics[key]["fine_tuned"]
        b_rec = m_b["recall"]
        f_rec = m_f["recall"]
        print(f"{label:<25} | {m_b['count']:<6} | {b_rec:<16.4f} | {f_rec:<18.4f} | {f_rec - b_rec:<+10.4f}")
    print("=" * 80)

    # Print Genuine Breakdown Table
    print("\n" + "=" * 80)
    print(" GENUINE SPEECH TEST SUBSETS SPECIFICITY / FPR (AT THRESHOLD 0.50)")
    print("=" * 80)
    print(f"{'Genuine Subset':<25} | {'Count':<6} | {'Base TNR (Spec)':<16} | {'FT TNR (Spec)':<16} | {'Base FPR':<10} | {'FT FPR':<10}")
    print("-" * 80)
    gen_subsets = [
        ("Gary Stafford Real", "garystafford"),
        ("In-The-Wild Real", "in_the_wild"),
    ]
    for label, src_key in gen_subsets:
        m_b = source_metrics[src_key]["baseline"]
        m_f = source_metrics[src_key]["fine_tuned"]
        print(f"{label:<25} | {m_b['n_real']:<6} | {m_b['specificity']:<16.4f} | {m_f['specificity']:<16.4f} | {m_b['fpr']:<10.4f} | {m_f['fpr']:<10.4f}")
    print("=" * 80)

    print(f"\nTotal Evaluation Elapsed Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
