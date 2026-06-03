from __future__ import annotations

import json
from pathlib import Path

import torch

from fate_oia.models.psr_specialist_registry import SpecialistRegistry


def write_candidate(path: Path, name: str = "cand") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.save(torch.randn(6, 4), path / "logits_action_fused_best_test.pt")
    torch.save(torch.randn(6, 21), path / "logits_reason_best_test.pt")
    torch.save(torch.randint(0, 2, (6, 4)).float(), path / "labels_action_best_test.pt")
    torch.save(torch.randint(0, 2, (6, 21)).float(), path / "labels_reason_best_test.pt")
    (path / "file_names_best_test.json").write_text(json.dumps([f"{i}.jpg" for i in range(6)]), encoding="utf-8")
    return path


def write_registry(path: Path, action_dir: Path, exp_dir: Path) -> Path:
    text = f"""
config_version: psr_oia_v2_registry
test_only_evaluation: true
feature_cache_enabled: false
alignment:
  action_dim: 4
  reason_dim: 21
candidates:
  action_specialists:
    - name: action_a
      required: true
      search:
        run_dir_glob: ['{action_dir.as_posix()}']
  explanation_specialists:
    - name: exp_e
      required: true
      search:
        run_dir_glob: ['{exp_dir.as_posix()}']
  calibration_specialists: []
"""
    path.write_text(text, encoding="utf-8")
    return path


def test_specialist_registry_discovers_and_loads(tmp_path):
    a = write_candidate(tmp_path / "a")
    e = write_candidate(tmp_path / "e")
    cfg = write_registry(tmp_path / "registry.yaml", a, e)
    registry = SpecialistRegistry(cfg)
    loaded, report = registry.aligned_available(tmp_path / "out")
    assert len(loaded) == 2
    assert report["aligned"] is True
    assert (tmp_path / "out" / "specialist_manifest.json").exists()
