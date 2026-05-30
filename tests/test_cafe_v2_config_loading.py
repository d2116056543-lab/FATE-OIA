from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from fate_oia.engine.train_cafe_oia import build_parser
from fate_oia.utils.config_io import parse_args_with_config


def test_yaml_overrides_and_cli_wins(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("config_version: cafe_oia_v2_evidence_fixed\nloss:\n  direct_effect: 0.031\ntraining:\n  best_selection_split: test\n", encoding="utf-8")
    parser = build_parser()
    args = parse_args_with_config(parser, ["--config", str(cfg), "--output_dir", str(tmp_path / "out")], "cafe_oia_v2_evidence_fixed")
    assert args.loss_direct_effect == pytest.approx(0.031)
    parser = build_parser()
    args = parse_args_with_config(parser, ["--config", str(cfg), "--output_dir", str(tmp_path / "out"), "--loss_direct_effect", "0.044"], "cafe_oia_v2_evidence_fixed")
    assert args.loss_direct_effect == pytest.approx(0.044)


def test_bad_config_version_blocks(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("config_version: cafe_oia_v1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_args_with_config(build_parser(), ["--config", str(cfg), "--output_dir", str(tmp_path / "out")], "cafe_oia_v2_evidence_fixed")
