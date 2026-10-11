from src.core.config import config

def test_config_loaded():
    assert config is not None
    assert "thresholds" in config._raw_config
    assert "benchmarks" in config._raw_config
    assert "engines" in config._raw_config

def test_benchmarks_metadata():
    benchmarks = config.benchmarks
    assert "faceforensics_pp" in benchmarks
    assert "dfdc" in benchmarks
    assert benchmarks["faceforensics_pp"]["name"] == "FaceForensics++"
    assert benchmarks["dfdc"]["name"] == "Deepfake Detection Challenge (DFDC)"

def test_thresholds():
    assert float(config.thresholds.get("vision_fake_threshold", 0)) == 0.50
    assert float(config.thresholds.get("audio_fake_threshold", 0)) == 0.50
