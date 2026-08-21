from __future__ import annotations

import tempfile
from pathlib import Path

from benchmark.config import BenchmarkConfig, TrajectoryConfig, config_hash, load_config


def test_load_config_merges_onto_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.yaml"
        path.write_text("name: my_test\nn_runs: 7\ntrajectory:\n  kind: aggressive\n")
        config = load_config(path)
        assert config.name == "my_test"
        assert config.n_runs == 7
        assert config.trajectory.kind == "aggressive"
        # untouched fields keep their dataclass defaults
        assert config.trajectory.dt == TrajectoryConfig().dt
        assert config.noise.gyro_noise == BenchmarkConfig().noise.gyro_noise


def test_config_hash_is_deterministic_and_sensitive_to_changes():
    a = BenchmarkConfig()
    b = BenchmarkConfig()
    assert config_hash(a) == config_hash(b)

    c = BenchmarkConfig(n_runs=BenchmarkConfig().n_runs + 1)
    assert config_hash(a) != config_hash(c)

    d = BenchmarkConfig(trajectory=TrajectoryConfig(kind="aggressive"))
    assert config_hash(a) != config_hash(d)


def test_config_hash_ignores_field_order():
    a = BenchmarkConfig(estimators=["ekf", "eskf"])
    b = BenchmarkConfig(estimators=["ekf", "eskf"])
    assert config_hash(a) == config_hash(b)
