from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .psi_damo_dataset import (
    _first,
    _infer_text_action,
    _letterbox_tensor,
    _load_exp29_records,
    _load_records,
    _mask_has_positive,
    _text_matches_action,
)
from .psi_frame_resolver import PSIFrameResolver, assert_target_not_in_inputs


ACTION_NAME_TO_ID = {"maintain_speed": 0, "reduce_speed": 1, "stop_car": 2}


class PSISanityDataset(Dataset):
    """A protocol sanity dataset with explicit input modes.

    Modes:
    - target_frame: use the supervised target frame image.
    - last_observed: use input_frames[-1] only, without target leakage.
    - clip15: use all 15 observed frames, matching the old no-target protocol.
    - k_current_15: use target_frame-14..target_frame, matching the current PSI protocol.
    """

    VALID_MODES = {"target_frame", "last_observed", "clip15", "k_current_15"}

    def __init__(
        self,
        package_root: str | Path,
        split: str,
        input_mode: str,
        frames_root: str | Path,
        image_size: tuple[int, int] = (224, 384),
        max_samples: int | None = None,
        max_sample_strategy: str = "head",
        max_sample_seed: int = 7,
        protocol_index_dir: str | Path | None = None,
        protocol_name: str | None = None,
        exp_supervision_policy: str = "record_mask",
        exp_near_keyframe_max_gap: int = 30,
        use_decision_group_weight: bool = False,
    ) -> None:
        if input_mode not in self.VALID_MODES:
            raise ValueError(f"input_mode must be one of {sorted(self.VALID_MODES)}, got {input_mode!r}")
        self.package_root = Path(package_root)
        self.split = split
        self.input_mode = input_mode
        self.image_size = image_size
        self.max_sample_strategy = str(max_sample_strategy)
        self.max_sample_seed = int(max_sample_seed)
        self.protocol_index_dir = Path(protocol_index_dir) if protocol_index_dir else None
        self.protocol_name = str(protocol_name) if protocol_name else None
        self.exp_supervision_policy = str(exp_supervision_policy)
        self.exp_near_keyframe_max_gap = int(exp_near_keyframe_max_gap)
        self.use_decision_group_weight = bool(use_decision_group_weight)
        split_path = self.package_root / "samples" / f"{split}.pkl"
        all_records = _load_records(split_path)
        if self.protocol_index_dir is not None:
            selected_indices, self.protocol_index_rows = self._load_protocol_index(split)
        else:
            selected_indices = list(range(len(all_records)))
            self.protocol_index_rows = [{} for _ in selected_indices]
        if max_samples is not None:
            selected_indices, self.protocol_index_rows = self._limit_indices(
                all_records,
                selected_indices,
                self.protocol_index_rows,
                int(max_samples),
            )
        self.record_indices = selected_indices
        self.records = [all_records[idx] for idx in selected_indices]
        self.decision_group_counts = Counter(self._decision_group_key(record) for record in self.records)
        self.exp_records: list[dict[str, Any]] | None = None
        exp_path = self.package_root / "reason_exp29" / f"{split}.pkl"
        if exp_path.exists():
            self.exp_records = _load_exp29_records(exp_path)
        self.resolver = PSIFrameResolver(frames_root)

    def __len__(self) -> int:
        return len(self.records)

    def _load_protocol_index(self, split: str) -> tuple[list[int], list[dict[str, Any]]]:
        candidates = []
        if self.protocol_name:
            candidates.append(self.protocol_index_dir / self.protocol_name / f"{split}_indices.jsonl")
        candidates.append(self.protocol_index_dir / f"{split}_indices.jsonl")
        index_path = next((path for path in candidates if path.exists()), None)
        if index_path is None:
            raise FileNotFoundError(f"No protocol index for split={split!r}; tried {candidates}")
        selected = []
        rows = []
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                selected.append(int(row["source_index"]))
                rows.append(row)
        return selected, rows

    def _record_action_id(self, record: dict[str, Any], protocol_row: dict[str, Any] | None = None) -> int:
        if protocol_row is not None and "action_hard" in protocol_row:
            return int(protocol_row["action_hard"])
        soft = torch.tensor(
            _first(record, ("action_soft_target", "action_soft", "action_distribution", "soft_action"), [1 / 3, 1 / 3, 1 / 3]),
            dtype=torch.float32,
        )
        majority = _first(record, ("action_majority", "action_label", "majority_action"), None)
        if majority is not None:
            return int(majority)
        return ACTION_NAME_TO_ID.get(str(_first(record, ("action_name",), "")), int(soft.argmax().item()))

    def _decision_group_key(self, record: dict[str, Any]) -> tuple[str, int]:
        decision_keyframe = _first(record, ("decision_keyframe", "decision_frame", "decision_kf"), None)
        if decision_keyframe is None:
            target = _first(record, ("target_frame",), 0)
            decision_keyframe = target
        return (str(_first(record, ("video_id", "video", "vid"), "")), int(decision_keyframe))

    def _limit_indices(
        self,
        all_records: list[dict[str, Any]],
        selected_indices: list[int],
        protocol_rows: list[dict[str, Any]],
        max_samples: int,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        if len(selected_indices) <= max_samples:
            return selected_indices, protocol_rows
        if self.max_sample_strategy == "head":
            return selected_indices[:max_samples], protocol_rows[:max_samples]
        if self.max_sample_strategy != "balanced_action":
            raise ValueError(f"Unsupported PSI sanity max_sample_strategy: {self.max_sample_strategy}")
        rng = random.Random(self.max_sample_seed)
        pairs = list(zip(selected_indices, protocol_rows))
        by_class: dict[int, list[tuple[int, dict[str, Any]]]] = {0: [], 1: [], 2: []}
        for pair in pairs:
            by_class.setdefault(self._record_action_id(all_records[pair[0]], pair[1]), []).append(pair)
        selected_pairs: list[tuple[int, dict[str, Any]]] = []
        per_class = max(1, max_samples // 3)
        for cls in range(3):
            candidates = list(by_class.get(cls, []))
            rng.shuffle(candidates)
            selected_pairs.extend(candidates[:per_class])
        if len(selected_pairs) < max_samples:
            selected_set = {idx for idx, _ in selected_pairs}
            remaining = [pair for pair in pairs if pair[0] not in selected_set]
            rng.shuffle(remaining)
            selected_pairs.extend(remaining[: max_samples - len(selected_pairs)])
        selected_pairs = selected_pairs[:max_samples]
        rng.shuffle(selected_pairs)
        return [idx for idx, _ in selected_pairs], [row for _, row in selected_pairs]

    def _target_value(self, record: dict[str, Any], protocol_row: dict[str, Any] | None = None) -> Any:
        if protocol_row is not None:
            target = _first(protocol_row, ("target_frame", "target_frame_path"), None)
            if target is not None:
                return target
        return _first(record, ("target_frame", "target_frame_path"), None)

    def _target_path(self, record: dict[str, Any], video_id: str, protocol_row: dict[str, Any] | None = None) -> Path:
        target = self._target_value(record, protocol_row)
        if target is None:
            raise KeyError("sample missing target_frame")
        if isinstance(target, int) or (isinstance(target, str) and str(target).isdigit()):
            return self.resolver.resolve_frame_id(video_id, target)
        return self.resolver.resolve(str(target))

    def _k_current_paths(self, record: dict[str, Any], video_id: str, protocol_row: dict[str, Any] | None = None) -> list[Path]:
        target = self._target_value(record, protocol_row)
        if target is None:
            raise KeyError("sample missing target_frame")
        target_id = int(target)
        return [self.resolver.resolve_frame_id(video_id, frame_id) for frame_id in range(target_id - 14, target_id + 1)]

    def _exp_is_near_keyframe(self, record: dict[str, Any]) -> bool:
        explanation_keyframe = _first(record, ("explanation_keyframe",), None)
        if explanation_keyframe is None:
            return False
        gaps = []
        for key in ("decision_keyframe", "target_frame"):
            value = _first(record, (key,), None)
            if value is not None:
                gaps.append(abs(int(explanation_keyframe) - int(value)))
        return bool(gaps) and min(gaps) <= self.exp_near_keyframe_max_gap

    def _semantic_exp_is_valid(self, record: dict[str, Any], mask: torch.Tensor) -> bool:
        if not _mask_has_positive(mask):
            return False
        text = (
            str(_first(record, ("explanation_text", "explanation", "description"), ""))
            + "\n"
            + str(_first(record, ("reasoning_text", "reason", "reasoning"), ""))
        ).strip()
        if not text or not _text_matches_action(str(_first(record, ("action_name",), "")), _infer_text_action(text)):
            return False
        return self._exp_is_near_keyframe(record)

    def _exp29(self, idx: int, record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        source = record
        source_idx = self.record_indices[idx] if hasattr(self, "record_indices") else idx
        if self.exp_records is not None and source_idx < len(self.exp_records):
            source = {**record, **self.exp_records[source_idx]}
        values = _first(source, ("exp29", "reason_exp29", "reason_labels", "explanation_labels"), [0.0] * 29)
        exp = torch.tensor(values, dtype=torch.float32)
        if exp.numel() != 29:
            raise ValueError(f"Exp29 target must have 29 labels, got {exp.numel()}")
        mask_values = _first(source, ("exp29_mask", "reason_mask", "explanation_mask"), None)
        if mask_values is None:
            mask = torch.ones_like(exp)
            if float(exp.sum()) == 0.0:
                mask.zero_()
        elif isinstance(mask_values, (int, float)):
            mask = torch.full_like(exp, float(mask_values))
        else:
            mask = torch.tensor(mask_values, dtype=torch.float32)
        if float(exp.sum()) == 0.0:
            mask = torch.zeros_like(mask)
        if self.exp_supervision_policy == "semantic_consistent_near_keyframe":
            if not self._semantic_exp_is_valid(source, mask):
                mask = torch.zeros_like(mask)
        elif self.exp_supervision_policy == "near_keyframe_raw_mask":
            if not _mask_has_positive(mask) or not self._exp_is_near_keyframe(source):
                mask = torch.zeros_like(mask)
        elif self.exp_supervision_policy == "protocol_index_mask":
            protocol_row = self.protocol_index_rows[idx] if idx < len(self.protocol_index_rows) else {}
            if not bool(protocol_row.get("exp29_supervised", False)):
                mask = torch.zeros_like(mask)
        elif self.exp_supervision_policy != "record_mask":
            raise ValueError(f"Unsupported exp_supervision_policy: {self.exp_supervision_policy}")
        return exp, mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        protocol_row = self.protocol_index_rows[idx] if idx < len(self.protocol_index_rows) else {}
        video_id = str(_first(record, ("video_id", "video", "vid"), ""))
        frame_values = list(_first(record, ("input_frames", "frames", "observed_frames"), []))
        frame_paths = self.resolver.resolve_sequence(frame_values, expected_count=15, video_id=video_id)
        target_path = self._target_path(record, video_id, protocol_row)
        if self.input_mode in {"last_observed", "clip15"}:
            assert_target_not_in_inputs(frame_paths, target_path)
        if self.input_mode == "target_frame":
            images = _letterbox_tensor(target_path, self.image_size)
            selected_frame_id = int(self._target_value(record, protocol_row) or 0)
        elif self.input_mode == "last_observed":
            images = _letterbox_tensor(frame_paths[-1], self.image_size)
            selected_frame_id = int(frame_values[-1])
        else:
            if self.input_mode == "k_current_15":
                frame_paths = self._k_current_paths(record, video_id, protocol_row)
                selected_frame_id = int(self._target_value(record, protocol_row) or 0)
            else:
                selected_frame_id = int(frame_values[-1])
            images = torch.stack([_letterbox_tensor(path, self.image_size) for path in frame_paths], dim=0)
        soft_source = _first(protocol_row, ("action_soft_target", "action_soft", "soft_action"), None)
        if soft_source is None:
            soft_source = _first(record, ("action_soft_target", "action_soft", "soft_action"), [1 / 3, 1 / 3, 1 / 3])
        soft = torch.tensor(soft_source, dtype=torch.float32)
        if soft.numel() != 3:
            raise ValueError(f"PSI action soft target must have 3 classes, got {soft.numel()}")
        majority = protocol_row.get("action_hard", None)
        if majority is None:
            majority = _first(record, ("action_majority", "action_label", "majority_action"), None)
        if majority is None:
            action_name = str(_first(record, ("action_name",), ""))
            majority = ACTION_NAME_TO_ID.get(action_name, int(soft.argmax().item()))
        exp29, exp29_mask = self._exp29(idx, record)
        protocol_weight = float(protocol_row.get("action_weight", 1.0))
        group_weight = 1.0
        if self.use_decision_group_weight:
            group_count = max(int(self.decision_group_counts.get(self._decision_group_key(record), 1)), 1)
            group_weight = 1.0 / float(group_count)
        return {
            "images": images,
            "action_soft": soft,
            "action_majority": torch.tensor(int(majority), dtype=torch.long),
            "exp29": exp29,
            "exp29_mask": exp29_mask,
            "paper_effective_weight": torch.tensor(
                float(_first(record, ("paper_effective_weight", "sample_weight", "weight"), 1.0)) * protocol_weight * group_weight,
                dtype=torch.float32,
            ),
            "video_id": video_id,
            "sample_id": str(_first(record, ("sample_id", "id"), f"{self.split}_{idx}")),
            "input_mode": self.input_mode,
            "selected_frame_id": torch.tensor(selected_frame_id, dtype=torch.long),
            "target_frame_path": str(target_path),
            "frame_paths": [str(path) for path in frame_paths],
        }


def psi_sanity_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["images"] for item in items], dim=0),
        "action_soft": torch.stack([item["action_soft"] for item in items], dim=0),
        "action_majority": torch.stack([item["action_majority"] for item in items], dim=0),
        "exp29": torch.stack([item["exp29"] for item in items], dim=0),
        "exp29_mask": torch.stack([item["exp29_mask"] for item in items], dim=0),
        "paper_effective_weight": torch.stack([item["paper_effective_weight"] for item in items], dim=0),
        "video_id": [item["video_id"] for item in items],
        "sample_id": [item["sample_id"] for item in items],
        "input_mode": items[0]["input_mode"] if items else None,
        "selected_frame_id": torch.stack([item["selected_frame_id"] for item in items], dim=0),
        "target_frame_path": [item["target_frame_path"] for item in items],
        "frame_paths": [item["frame_paths"] for item in items],
    }

