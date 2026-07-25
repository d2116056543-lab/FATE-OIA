"""Contract tests for the frozen RAEL DINO ViT-S/8 field.

These tests use a tiny fake implementation for most behaviors and always load
the project-local official DINO checkpoint on CPU. They never download weights
and never require CUDA.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fate_oia.models.rael_dino_field import RAELDinoFieldExtractor


class _RecordingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.attention_map: torch.Tensor | None = None
        self.attn_gradients: torch.Tensor | None = None
        self.attention: torch.Tensor | None = None
        self.input: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self.input = tokens
        self.attention_map = tokens[..., :1]
        self.attn_gradients = tokens[..., :1]
        self.attention = tokens[..., :1]
        self.v = tokens[..., :1]
        return tokens + self.scale * 0.0


class _ImmediateReleaseBlock(_RecordingBlock):
    def __init__(self, previous: _RecordingBlock | None) -> None:
        super().__init__()
        self.previous = previous

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.previous is not None:
            assert self.previous.attention_map is None
            assert self.previous.attn_gradients is None
            assert self.previous.input is None
            assert self.previous.v is None
        return super().forward(tokens)


class _FakeDino(nn.Module):
    """Imitates the DINO ViT token API without network or GPU dependencies."""

    def __init__(self, token_count: int = 3601, embed_dim: int = 384) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.token_count = token_count
        self.blocks = nn.ModuleList([_RecordingBlock() for _ in range(12)])
        self.norm = nn.Identity()
        self.seed = nn.Parameter(torch.ones(embed_dim))
        self.prepare_tokens_calls = 0

    def prepare_tokens(self, images: torch.Tensor) -> torch.Tensor:
        self.prepare_tokens_calls += 1
        batch = images.shape[0]
        return self.seed.view(1, 1, -1).expand(batch, self.token_count, -1).clone()


class _ImmediateReleaseDino(_FakeDino):
    def __init__(self) -> None:
        super().__init__()
        blocks: list[_ImmediateReleaseBlock] = []
        previous: _RecordingBlock | None = None
        for _ in range(12):
            block = _ImmediateReleaseBlock(previous)
            blocks.append(block)
            previous = block
        self.blocks = nn.ModuleList(blocks)


def _field(backbone: _FakeDino | None = None) -> RAELDinoFieldExtractor:
    return RAELDinoFieldExtractor(
        arch="vit_small",
        patch_size=8,
        selected_layers=(3, 6, 9, 12),
        checkpoint_key="teacher",
        backbone=backbone or _FakeDino(),
    )


def test_contract_returns_four_frozen_dense_layers_and_single_batch_call() -> None:
    backbone = _FakeDino()
    field = _field(backbone)
    images = torch.randn(2, 3, 360, 640)

    output = field(images)

    assert output["patch_tokens_by_layer"].shape == (2, 4, 3600, 384)
    assert output["cls_tokens_by_layer"].shape == (2, 4, 384)
    assert output["grid_hw"] == (45, 80)
    assert output["original_tokens"] == 3601
    assert output["dino_call_count"] == 1
    assert backbone.prepare_tokens_calls == 1
    assert not output["patch_tokens_by_layer"].requires_grad
    assert not output["cls_tokens_by_layer"].requires_grad


def test_dino_call_count_is_per_forward_and_lifetime_count_is_monotonic() -> None:
    backbone = _FakeDino()
    field = _field(backbone)

    first = field(torch.randn(1, 3, 360, 640))
    second = field(torch.randn(1, 3, 360, 640))

    assert first["dino_call_count"] == 1
    assert second["dino_call_count"] == 1
    assert first["lifetime_dino_call_count"] == 1
    assert second["lifetime_dino_call_count"] == 2
    assert field.lifetime_dino_call_count == 2
    assert backbone.prepare_tokens_calls == 2


def test_mirror_contract_concatenates_before_one_extractor_call() -> None:
    backbone = _FakeDino()
    field = _field(backbone)
    canonical = torch.randn(2, 3, 360, 640)
    mirror = canonical.flip(-1)

    combined = field.concat_canonical_and_mirror(canonical, mirror)
    output = field(combined)

    assert combined.shape == (4, 3, 360, 640)
    assert output["patch_tokens_by_layer"].shape[0] == 4
    assert output["dino_call_count"] == 1
    assert backbone.prepare_tokens_calls == 1


def test_backbone_is_frozen_and_stays_eval_when_parent_enters_train_mode() -> None:
    backbone = _FakeDino()
    field = _field(backbone)

    field.train()
    assert field.training
    assert not field.backbone.training
    assert all(not parameter.requires_grad for parameter in field.backbone.parameters())

    field(torch.randn(1, 3, 360, 640))
    assert not field.backbone.training


def test_downstream_projection_gets_grad_but_frozen_dino_never_does() -> None:
    field = _field()
    projection = nn.Linear(384, 1)

    output = field(torch.randn(1, 3, 360, 640))
    loss = projection(output["patch_tokens_by_layer"]).sum()
    loss.backward()

    assert projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in field.backbone.parameters())


@pytest.mark.parametrize(
    ("images", "message"),
    [
        (torch.randn(1, 3, 224, 224), "360x640"),
        (torch.randn(1, 1, 360, 640), "3-channel"),
        (torch.randn(3, 360, 640), "rank-4"),
    ],
)
def test_rejects_wrong_image_contract(images: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _field()(images)


def test_rejects_wrong_layers_patch_size_and_cache_arguments() -> None:
    with pytest.raises(ValueError, match="selected_layers"):
        RAELDinoFieldExtractor(backbone=_FakeDino(), selected_layers=(3, 7, 11))
    with pytest.raises(ValueError, match="patch_size"):
        RAELDinoFieldExtractor(backbone=_FakeDino(), patch_size=16)
    with pytest.raises(TypeError):
        RAELDinoFieldExtractor(backbone=_FakeDino(), cache_path="forbidden")


def test_rejects_backbone_with_wrong_token_count() -> None:
    with pytest.raises(RuntimeError, match="3601"):
        _field(_FakeDino(token_count=3600))(torch.randn(1, 3, 360, 640))


def test_recursively_clears_transient_attention_input_and_value_tensors() -> None:
    backbone = _FakeDino()
    field = _field(backbone)
    field(torch.randn(1, 3, 360, 640))

    for block in backbone.blocks:
        assert block.attention_map is None
        assert block.attn_gradients is None
        assert block.attention is None
        assert block.input is None
        assert block.v is None


def test_each_real_attention_field_is_released_before_the_next_block_runs() -> None:
    backbone = _ImmediateReleaseDino()
    field = _field(backbone)

    output = field(torch.randn(1, 3, 360, 640))

    assert output["patch_tokens_by_layer"].shape == (1, 4, 3600, 384)
    for block in backbone.blocks:
        assert block.attention_map is None
        assert block.attn_gradients is None
        assert block.input is None
        assert block.v is None


def test_cuda_bf16_autocast_is_used_only_when_the_device_supports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fate_oia.models.rael_dino_field as module

    calls: list[tuple[str, torch.dtype]] = []
    fake_cuda_image = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "is_bf16_supported", lambda: True)

    def _recording_autocast(*, device_type: str, dtype: torch.dtype):
        calls.append((device_type, dtype))
        return nullcontext()

    monkeypatch.setattr(module.torch, "autocast", _recording_autocast)
    with RAELDinoFieldExtractor._autocast_context(fake_cuda_image):
        pass

    assert calls == [("cuda", torch.bfloat16)]


@pytest.mark.parametrize(
    ("cuda_available", "bf16_supported"),
    [(False, False), (True, False)],
)
def test_cuda_without_bf16_support_falls_back_without_autocast(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    bf16_supported: bool,
) -> None:
    import fate_oia.models.rael_dino_field as module

    fake_cuda_image = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(
        module.torch.cuda,
        "is_bf16_supported",
        lambda: bf16_supported,
        raising=False,
    )

    def _unexpected_autocast(**_: object):
        raise AssertionError("unsupported CUDA path must not enter bf16 autocast")

    monkeypatch.setattr(module.torch, "autocast", _unexpected_autocast)
    with RAELDinoFieldExtractor._autocast_context(fake_cuda_image):
        pass


def test_cpu_autocast_context_is_a_safe_fp32_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fate_oia.models.rael_dino_field as module

    fake_cpu_image = SimpleNamespace(device=SimpleNamespace(type="cpu"))

    def _unexpected_autocast(**_: object):
        raise AssertionError("CPU contract must not require bf16 autocast")

    monkeypatch.setattr(module.torch, "autocast", _unexpected_autocast)
    with RAELDinoFieldExtractor._autocast_context(fake_cpu_image):
        pass


def test_official_teacher_checkpoint_loader_is_strict_about_bad_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fate_oia.models.rael_dino_field as module

    reference = _FakeDino()
    teacher_state = {
        f"module.backbone.{key}": value for key, value in reference.state_dict().items()
    }
    checkpoint_path = tmp_path / "official_teacher.pth"
    torch.save(
        {
            "teacher": {
                **teacher_state,
                "module.head.last_layer.weight": torch.ones(2, 2),
            }
        },
        checkpoint_path,
    )

    monkeypatch.setattr(
        module,
        "vits",
        SimpleNamespace(vit_small=lambda patch_size, num_classes: _FakeDino()),
    )
    field = RAELDinoFieldExtractor(pretrained_weights=str(checkpoint_path))
    assert field.checkpoint_key == "teacher"
    assert field.pretrained_weights == str(checkpoint_path)
    assert field.weight_load_report is not None
    assert field.weight_load_report["checkpoint_format"] == "teacher_wrapper"

    raw_path = tmp_path / "raw_backbone_state.pth"
    torch.save(teacher_state, raw_path)
    raw_field = RAELDinoFieldExtractor(pretrained_weights=str(raw_path))
    assert raw_field.weight_load_report is not None
    assert raw_field.weight_load_report["checkpoint_format"] == "raw_backbone_state_dict"

    bad_path = tmp_path / "bad_teacher.pth"
    torch.save({"student": reference.state_dict()}, bad_path)
    with pytest.raises(KeyError, match="teacher"):
        RAELDinoFieldExtractor(pretrained_weights=str(bad_path))

    missing_path = tmp_path / "missing_teacher_weight.pth"
    missing_state = dict(teacher_state)
    missing_state.pop("module.backbone.seed")
    torch.save({"teacher": missing_state}, missing_path)
    with pytest.raises(RuntimeError, match="missing"):
        RAELDinoFieldExtractor(pretrained_weights=str(missing_path))

    unexpected_path = tmp_path / "unexpected_teacher_weight.pth"
    unexpected_state = dict(teacher_state)
    unexpected_state["module.backbone.not_a_real_weight"] = torch.ones(1)
    torch.save({"teacher": unexpected_state}, unexpected_path)
    with pytest.raises(RuntimeError, match="unexpected"):
        RAELDinoFieldExtractor(pretrained_weights=str(unexpected_path))

    raw_missing_path = tmp_path / "raw_missing_teacher_weight.pth"
    torch.save(missing_state, raw_missing_path)
    with pytest.raises(RuntimeError, match="missing"):
        RAELDinoFieldExtractor(pretrained_weights=str(raw_missing_path))

    raw_unexpected_path = tmp_path / "raw_unexpected_teacher_weight.pth"
    torch.save(unexpected_state, raw_unexpected_path)
    with pytest.raises(RuntimeError, match="unexpected"):
        RAELDinoFieldExtractor(pretrained_weights=str(raw_unexpected_path))

    ambiguous_raw_path = tmp_path / "ambiguous_raw_state.pth"
    ambiguous_raw_state = dict(list(teacher_state.items())[:7])
    torch.save(ambiguous_raw_state, ambiguous_raw_path)
    with pytest.raises(KeyError, match="neither"):
        RAELDinoFieldExtractor(pretrained_weights=str(ambiguous_raw_path))


def test_does_not_import_an_old_acpr_main_path() -> None:
    source = Path(__file__).resolve().parents[1] / "fate_oia/models/rael_dino_field.py"
    text = source.read_text(encoding="utf-8")
    assert "acpr_oia_model" not in text
    assert "ACPROIAModel" not in text


def test_real_local_weight_cpu_load_contract() -> None:
    weight_path = os.environ.get(
        "RAEL_DINO_WEIGHTS",
        "E:/sbw/FATE_Drive/fate_oia_worktree/ckp/reference/dino_deitsmall8_pretrain.pth",
    )
    assert Path(weight_path).is_file(), f"required local DINO weight is missing: {weight_path}"
    field = RAELDinoFieldExtractor(pretrained_weights=weight_path)
    assert field.weight_load_report is not None
    assert field.weight_load_report["checkpoint_format"] == "raw_backbone_state_dict"
    assert all(not parameter.requires_grad for parameter in field.backbone.parameters())
    if os.environ.get("RAEL_REAL_DINO_FORWARD") == "1":
        output = field(torch.zeros(1, 3, 360, 640))
        assert output["patch_tokens_by_layer"].shape == (1, 4, 3600, 384)
