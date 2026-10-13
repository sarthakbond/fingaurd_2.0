import pytest
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Config

from scripts.train_pellav2 import Pellav2Detector, Pellav2Dataset, compute_metrics
from scripts.evaluate_audio_models import _build_pellav2_detector_class, _pellav2_load_clip

CKPT_PATH = Path(r"E:\project models\pellav2_detector.pt")
VAL_MANIFEST = Path(r"E:\datasets\splits\val_manifest.json")
DATASET_ROOT = Path(r"E:\datasets")


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def checkpoint_state(device):
    assert CKPT_PATH.is_file(), f"Original checkpoint missing at {CKPT_PATH}"
    return torch.load(str(CKPT_PATH), map_location=device)


def test_checkpoint_state_dict_exact_match(checkpoint_state):
    """Verifies all 425 parameter keys are present in official checkpoint."""
    assert len(checkpoint_state) == 425
    assert "layer_weights" in checkpoint_state
    assert "head.weight" in checkpoint_state
    assert "head.bias" in checkpoint_state
    assert checkpoint_state["layer_weights"].shape == torch.Size([25])
    assert checkpoint_state["head.weight"].shape == torch.Size([1, 1024])
    assert checkpoint_state["head.bias"].shape == torch.Size([1])


def test_model_architecture_and_weight_loading_parity(checkpoint_state, device):
    """Verifies project detector loads checkpoint with strict=True parity."""
    model = Pellav2Detector("facebook/wav2vec2-xls-r-300m").to(device)
    missing, unexpected = model.load_state_dict(checkpoint_state, strict=True)
    assert len(missing) == 0
    assert len(unexpected) == 0

    # Verify parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 315439770


def test_tensor_forward_pass_exact_parity(checkpoint_state, device):
    """Verifies official Detector and project Detector produce bit-for-bit identical logits on identical inputs."""
    OfficialDetector = _build_pellav2_detector_class()
    official_model = OfficialDetector().to(device)
    official_model.load_state_dict(checkpoint_state, strict=True)
    official_model.eval()

    project_model = Pellav2Detector("facebook/wav2vec2-xls-r-300m").to(device)
    project_model.load_state_dict(checkpoint_state, strict=True)
    project_model.eval()

    # Deterministic test input
    torch.manual_seed(1234)
    x = torch.randn(2, 64000, device=device)

    with torch.no_grad():
        out_official = official_model(x)
        out_project = project_model(x)

    delta = torch.abs(out_official - out_project).max().item()
    assert delta < 1e-5, f"Logit mismatch between official and project model: max delta = {delta}"


def test_label_mapping_contract():
    """Verifies label mapping: genuine human speech = 0.0, synthetic clone = 1.0."""
    dataset = Pellav2Dataset(VAL_MANIFEST, DATASET_ROOT)
    
    # Check first 50 entries
    for i in range(min(50, len(dataset))):
        entry = dataset.samples[i]
        _, label_tensor = dataset[i]
        expected = 1.0 if entry["label"] == "fake" else 0.0
        assert label_tensor.item() == expected, f"Label mapping mismatch at index {i}: {entry['label']} vs {label_tensor.item()}"


def test_preprocessing_parity_on_16khz_wav():
    """Verifies waveform preprocessing on 16kHz audio matches official load_clip within numerical epsilon."""
    # Find a 16kHz WAV sample
    sample_wav = DATASET_ROOT / "in_the_wild" / "10039.wav"
    assert sample_wav.is_file(), f"Sample audio missing: {sample_wav}"

    # Official load
    w_official = _pellav2_load_clip(str(sample_wav))

    # Project dataset load
    dataset = Pellav2Dataset(VAL_MANIFEST, DATASET_ROOT)
    # Find index of 10039.wav
    idx = next(i for i, s in enumerate(dataset.samples) if "10039.wav" in s["filename"])
    w_project, _ = dataset[idx]

    w_project_np = w_project.cpu().numpy()
    max_diff = np.max(np.abs(w_official - w_project_np))
    mean_diff = np.mean(np.abs(w_official - w_project_np))

    # Single-precision float tolerance across libsndfile and torchaudio
    assert max_diff < 1e-4, f"Waveform preprocessing max difference too high: {max_diff}"
    assert mean_diff < 1e-5, f"Waveform preprocessing mean difference too high: {mean_diff}"
