"""Streaming, fail-closed RAEL P19 case export from formal model outputs."""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from fate_oia.models.rael_oia_model import BRANCH_NAMES
from fate_oia.utils.rael_artifacts import _validate_epoch_jsonl
from fate_oia.utils.rael_posthoc_calibration import (
    apply_posthoc_calibration,
    serialize_calibration_result,
)


_ACTION_COUNT = 4
_REASON_COUNT = 21
_SLOT_COUNT = 20
_NAMED_SLOT_COUNT = 12
_PAIR_COUNT = _SLOT_COUNT * (_SLOT_COUNT - 1) // 2
_GRID_HW = (45, 80)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PLACEHOLDER_NAME = re.compile(r"^(?:action|reason)[_ -]?\d+$|^(?:unknown|tbd|placeholder)$", re.IGNORECASE)
_EXPECTED_PAIR_INDICES = torch.triu_indices(_SLOT_COUNT, _SLOT_COUNT, offset=1).transpose(0, 1).contiguous()


class RAELCaseExportCollector:
    """Collect bounded failure/evidence examples from one formal decoder batch."""

    def __init__(
        self,
        *,
        max_failure_cases: int,
        max_evidence_cases: int,
        top_slots: int,
        reconstruction_tolerance: float = 1.0e-5,
    ) -> None:
        self.max_failure_cases = self._positive_int("max_failure_cases", max_failure_cases)
        self.max_evidence_cases = self._positive_int("max_evidence_cases", max_evidence_cases)
        self.top_slots = self._positive_int("top_slots", top_slots)
        if self.top_slots > _SLOT_COUNT:
            raise ValueError(f"top_slots must be <= {_SLOT_COUNT}")
        if not isinstance(reconstruction_tolerance, (float, int)) or isinstance(reconstruction_tolerance, bool):
            raise TypeError("reconstruction_tolerance must be a positive finite number")
        self.reconstruction_tolerance = float(reconstruction_tolerance)
        if not math.isfinite(self.reconstruction_tolerance) or self.reconstruction_tolerance <= 0.0:
            raise ValueError("reconstruction_tolerance must be a positive finite number")
        self._failure_cases: list[dict[str, Any]] = []
        self._evidence_cases: list[dict[str, Any]] = []
        self._seen_file_names: set[str] = set()
        self._sample_count = 0
        self._finalized = False

    @staticmethod
    def _positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def consume(
        self,
        *,
        file_names: Sequence[str],
        action_targets: Tensor,
        reason_targets: Tensor,
        outputs: Mapping[str, Any],
        action_calibration: Mapping[str, Any],
        reason_calibration: Mapping[str, Any],
        action_names: Sequence[str],
        reason_names: Sequence[str],
    ) -> None:
        self._require_open()
        batch_size = self._validate_file_names(file_names)
        action_names = self._validate_names("action_names", action_names, _ACTION_COUNT)
        reason_names = self._validate_names("reason_names", reason_names, _REASON_COUNT)
        action_target = self._binary_tensor("action_targets", action_targets, (batch_size, _ACTION_COUNT))
        reason_target = self._binary_tensor("reason_targets", reason_targets, (batch_size, _REASON_COUNT))
        self._validate_calibration("action_calibration", action_calibration, _ACTION_COUNT)
        self._validate_calibration("reason_calibration", reason_calibration, _REASON_COUNT)
        formal = self._validate_formal_outputs(outputs, batch_size=batch_size)

        action_deploy = apply_posthoc_calibration(formal["action_final"], action_calibration)
        reason_deploy = apply_posthoc_calibration(formal["reason_final"], reason_calibration)
        action_raw = formal["action_final"] > 0.0
        reason_raw = formal["reason_final"] > 0.0
        action_deploy = self._require_tensor(
            "action calibration decision", action_deploy["decision"], (batch_size, _ACTION_COUNT), floating=False
        ).to(dtype=torch.bool)
        reason_deploy = self._require_tensor(
            "reason calibration decision", reason_deploy["decision"], (batch_size, _REASON_COUNT), floating=False
        ).to(dtype=torch.bool)

        for index, file_name in enumerate(file_names):
            if len(self._failure_cases) < self.max_failure_cases and self._is_failure(
                action_target[index], reason_target[index], action_raw[index], reason_raw[index], action_deploy[index], reason_deploy[index]
            ):
                self._failure_cases.append(
                    self._failure_row(
                        file_name=file_name, index=index, action_target=action_target, reason_target=reason_target,
                        action_raw=action_raw, reason_raw=reason_raw, action_deploy=action_deploy,
                        reason_deploy=reason_deploy, formal=formal,
                    )
                )
            if len(self._evidence_cases) < self.max_evidence_cases:
                for target_type, target_id in self._evidence_targets(
                    action_target[index], reason_target[index], action_deploy[index], reason_deploy[index]
                ):
                    if len(self._evidence_cases) >= self.max_evidence_cases:
                        break
                    target_name = action_names[target_id] if target_type == "action" else reason_names[target_id]
                    self._evidence_cases.append(
                        self._evidence_row(
                            file_name=file_name, index=index, target_type=target_type, target_id=target_id,
                            target_name=target_name, formal=formal,
                        )
                    )
        self._seen_file_names.update(file_names)
        self._sample_count += batch_size

    def finalize(self, provenance: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        self._require_open()
        if not self._failure_cases:
            raise ValueError("failure_cases.jsonl would be empty; no fake failure case may be exported")
        if not self._evidence_cases:
            raise ValueError("evidence_cases.jsonl would be empty; no fake evidence case may be exported")
        shared = self._validate_provenance(provenance, sample_count=self._sample_count)
        result = {
            "failure_cases.jsonl": [{**shared, **row} for row in self._failure_cases],
            "evidence_cases.jsonl": [{**shared, **row} for row in self._evidence_cases],
        }
        _validate_epoch_jsonl("failure_cases.jsonl", result["failure_cases.jsonl"], epoch=shared["epoch"])
        _validate_epoch_jsonl("evidence_cases.jsonl", result["evidence_cases.jsonl"], epoch=shared["epoch"])
        self._finalized = True
        return copy.deepcopy(result)

    def _require_open(self) -> None:
        if self._finalized:
            raise RuntimeError("RAELCaseExportCollector is finalized and immutable")

    def _validate_file_names(self, file_names: Sequence[str]) -> int:
        if not isinstance(file_names, Sequence) or isinstance(file_names, (str, bytes, bytearray)) or not file_names:
            raise ValueError("file_names must be a nonempty sequence")
        names = list(file_names)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("file_names must be nonempty strings")
        if len(set(names)) != len(names) or any(name in self._seen_file_names for name in names):
            raise ValueError("duplicate file_name is forbidden")
        return len(names)

    @staticmethod
    def _validate_names(name: str, names: Sequence[str], expected: int) -> tuple[str, ...]:
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes, bytearray)) or len(names) != expected:
            raise ValueError(f"{name} must contain exactly {expected} names")
        normalized = tuple(item.strip() if isinstance(item, str) else "" for item in names)
        if any(not item or _PLACEHOLDER_NAME.fullmatch(item) for item in normalized) or len(set(normalized)) != expected:
            raise ValueError(f"{name} must be unique, nonempty, and semantically named")
        return normalized

    @staticmethod
    def _require_tensor(name: str, value: Any, shape: tuple[int, ...], *, floating: bool = True) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
        if value.is_complex():
            raise TypeError(f"{name} must not be complex")
        if floating and not torch.is_floating_point(value):
            raise TypeError(f"{name} must be floating point")
        if torch.is_floating_point(value) and not bool(torch.isfinite(value.detach()).all()):
            raise ValueError(f"{name} must contain only finite values")
        return value.detach()

    @staticmethod
    def _require_pair_indices(value: Any, *, name: str) -> Tensor:
        tensor = RAELCaseExportCollector._require_tensor(name, value, (_PAIR_COUNT, 2), floating=False)
        if torch.is_floating_point(tensor) or tensor.dtype == torch.bool:
            raise TypeError(f"{name} must use an integer dtype")
        indices = tensor.to(device="cpu", dtype=torch.long)
        if not torch.equal(indices, _EXPECTED_PAIR_INDICES):
            raise ValueError(f"{name} must contain every unordered pair of {_SLOT_COUNT} slots exactly once")
        return indices.clone()

    def _binary_tensor(self, name: str, value: Any, shape: tuple[int, int]) -> Tensor:
        tensor = self._require_tensor(name, value, shape)
        if not bool(torch.logical_or(tensor == 0, tensor == 1).all()):
            raise ValueError(f"{name} must be binary")
        return tensor.to(device="cpu", dtype=torch.float32).clone()

    @staticmethod
    def _validate_calibration(name: str, result: Mapping[str, Any], targets: int) -> None:
        if not isinstance(result, Mapping):
            raise TypeError(f"{name} must be a P15 mapping")
        if result.get("fit_split") != "train_calib":
            raise ValueError(f"{name} must be fit on train_calib")
        if result.get("targets") != targets:
            raise ValueError(f"{name} target count mismatch")
        serialize_calibration_result(result)

    def _validate_formal_outputs(self, outputs: Mapping[str, Any], *, batch_size: int) -> dict[str, Any]:
        if not isinstance(outputs, Mapping):
            raise TypeError("outputs must be a mapping")
        action_final = self._require_tensor("action_logits_final", outputs.get("action_logits_final"), (batch_size, _ACTION_COUNT))
        reason_final = self._require_tensor("reason_logits_final", outputs.get("reason_logits_final"), (batch_size, _REASON_COUNT))
        branches = self._validate_branches(outputs.get("branch_logits"), batch_size, action_final, reason_final)
        masks = self._require_tensor("slot_masks", outputs.get("slot_masks"), (batch_size, _SLOT_COUNT, *_GRID_HW))
        if not bool(torch.any(masks.abs() > 0.0)):
            raise ValueError("slot_masks have fabricated zero signal")
        attributes = self._validate_attributes(outputs, batch_size=batch_size)
        action = self._validate_contribution_family(outputs, "action", batch_size, _ACTION_COUNT, action_final)
        reason = self._validate_contribution_family(outputs, "reason", batch_size, _REASON_COUNT, reason_final)
        if not torch.equal(action["pair_indices"], reason["pair_indices"]):
            raise ValueError("action_pair_indices and reason_pair_indices must be identical")
        return {
            "action_final": action_final.to(device="cpu", dtype=torch.float32).clone(),
            "reason_final": reason_final.to(device="cpu", dtype=torch.float32).clone(),
            "branches": branches,
            "masks": masks.to(device="cpu", dtype=torch.float32).clone(),
            "attributes": attributes,
            "action": action,
            "reason": reason,
        }

    def _validate_branches(self, value: Any, batch_size: int, action_final: Tensor, reason_final: Tensor) -> dict[str, dict[str, Tensor]]:
        if not isinstance(value, Mapping) or set(value) != set(BRANCH_NAMES) or len(value) != len(BRANCH_NAMES):
            raise ValueError("branch_logits must have the exact formal branch key set")
        sealed: dict[str, dict[str, Tensor]] = {}
        for branch_name in tuple(BRANCH_NAMES):
            branch = value[branch_name]
            if not isinstance(branch, Mapping) or set(branch) != {"action", "reason"}:
                raise ValueError(f"branch {branch_name} must contain exactly action/reason")
            sealed[branch_name] = {
                "action": self._require_tensor(f"branch {branch_name}.action", branch["action"], (batch_size, _ACTION_COUNT)),
                "reason": self._require_tensor(f"branch {branch_name}.reason", branch["reason"], (batch_size, _REASON_COUNT)),
            }
        if not torch.equal(sealed["full"]["action"], action_final) or not torch.equal(sealed["full"]["reason"], reason_final):
            raise ValueError("full branch must exactly equal final logits")
        return {
            name: {key: item.to(device="cpu", dtype=torch.float32).clone() for key, item in branch.items()}
            for name, branch in sealed.items()
        }

    def _validate_attributes(self, outputs: Mapping[str, Any], *, batch_size: int) -> dict[str, Any]:
        scalar_names = ("slot_area", "slot_scale", "slot_activity", "slot_presence", "slot_observability", "slot_reliability")
        sealed: dict[str, Any] = {}
        for name in scalar_names:
            sealed[name] = self._require_tensor(name, outputs.get(name), (batch_size, _SLOT_COUNT)).to(device="cpu", dtype=torch.float32).clone()
        sealed["slot_centroid"] = self._require_tensor(
            "slot_centroid", outputs.get("slot_centroid"), (batch_size, _SLOT_COUNT, 2)
        ).to(device="cpu", dtype=torch.float32).clone()
        sealed["slot_type_probs"] = self._probability_tensor(
            "slot_type_probs", outputs.get("slot_type_probs"), (batch_size, _NAMED_SLOT_COUNT, 6)
        )
        sealed["slot_state_probs"] = self._probability_tensor(
            "slot_state_probs", outputs.get("slot_state_probs"), (batch_size, _NAMED_SLOT_COUNT, 4)
        )
        sector = outputs.get("slot_sector_probs")
        if not isinstance(sector, Mapping) or set(sector) != {"horizontal", "depth"}:
            raise ValueError("slot_sector_probs must contain exactly horizontal/depth")
        sealed["slot_sector_probs"] = {
            name: self._probability_tensor(f"slot_sector_probs.{name}", sector[name], (batch_size, _SLOT_COUNT, 3))
            for name in ("horizontal", "depth")
        }
        return sealed

    def _probability_tensor(self, name: str, value: Any, shape: tuple[int, ...]) -> Tensor:
        tensor = self._require_tensor(name, value, shape)
        if bool(torch.any(tensor < 0.0)) or bool(torch.any(tensor > 1.0)):
            raise ValueError(f"{name} must be probabilities in [0, 1]")
        if not bool(torch.allclose(tensor.sum(dim=-1), torch.ones_like(tensor[..., 0]), atol=self.reconstruction_tolerance, rtol=0.0)):
            raise ValueError(f"{name} probability vectors must sum to one")
        return tensor.to(device="cpu", dtype=torch.float32).clone()

    def _validate_contribution_family(
        self, outputs: Mapping[str, Any], prefix: str, batch_size: int, width: int, final: Tensor
    ) -> dict[str, Tensor]:
        global_value = self._require_tensor(f"{prefix}_global_contribution", outputs.get(f"{prefix}_global_contribution"), (batch_size, width))
        unary = self._require_tensor(f"{prefix}_unary_contributions", outputs.get(f"{prefix}_unary_contributions"), (batch_size, width, _SLOT_COUNT))
        pairwise = self._require_tensor(f"{prefix}_pairwise_contributions", outputs.get(f"{prefix}_pairwise_contributions"), (batch_size, width, _PAIR_COUNT))
        incident = self._require_tensor(
            f"{prefix}_pairwise_incident_contributions", outputs.get(f"{prefix}_pairwise_incident_contributions"), (batch_size, width, _SLOT_COUNT)
        )
        deletion = self._require_tensor(f"{prefix}_analytical_deletion", outputs.get(f"{prefix}_analytical_deletion"), (batch_size, width, _SLOT_COUNT))
        pair_indices = self._require_pair_indices(outputs.get(f"{prefix}_pair_indices"), name=f"{prefix}_pair_indices")
        expected_incident = self._pairwise_incident(pairwise, pair_indices)
        if (incident - expected_incident).abs().max().item() > self.reconstruction_tolerance:
            raise ValueError(f"{prefix} pairwise incident contributions must equal the indexed pairwise sum")
        if (deletion - (unary + incident)).abs().max().item() > self.reconstruction_tolerance:
            raise ValueError(f"{prefix} analytical deletion must equal unary plus incident pairwise")
        reconstruction = global_value + unary.sum(dim=-1) + pairwise.sum(dim=-1)
        if (reconstruction - final).abs().max().item() > self.reconstruction_tolerance:
            raise ValueError(f"{prefix} global plus unary plus pairwise must reconstruct final")
        return {
            "global": global_value.to(device="cpu", dtype=torch.float32).clone(),
            "unary": unary.to(device="cpu", dtype=torch.float32).clone(),
            "pairwise": pairwise.to(device="cpu", dtype=torch.float32).clone(),
            "incident": incident.to(device="cpu", dtype=torch.float32).clone(),
            "deletion": deletion.to(device="cpu", dtype=torch.float32).clone(),
            "pair_indices": pair_indices,
        }

    @staticmethod
    def _pairwise_incident(pairwise: Tensor, pair_indices: Tensor) -> Tensor:
        incident = pairwise.new_zeros(pairwise.shape[0], pairwise.shape[1], _SLOT_COUNT)
        indices = pair_indices.to(device=pairwise.device)
        for side in (0, 1):
            incident.scatter_add_(2, indices[:, side].view(1, 1, -1).expand_as(pairwise), pairwise)
        return incident

    @staticmethod
    def _is_failure(action_target: Tensor, reason_target: Tensor, action_raw: Tensor, reason_raw: Tensor, action_deploy: Tensor, reason_deploy: Tensor) -> bool:
        return bool(
            torch.any(action_target.to(dtype=torch.bool) != action_raw)
            or torch.any(reason_target.to(dtype=torch.bool) != reason_raw)
            or torch.any(action_target.to(dtype=torch.bool) != action_deploy)
            or torch.any(reason_target.to(dtype=torch.bool) != reason_deploy)
        )

    @staticmethod
    def _case_id(kind: str, file_name: str, target_type: str | None = None, target_id: int | None = None) -> str:
        components = ["rael-p19", kind, file_name]
        if target_type is not None:
            components.extend((target_type, str(target_id)))
        return hashlib.sha256("\0".join(components).encode("utf-8")).hexdigest()

    def _failure_row(self, *, file_name: str, index: int, action_target: Tensor, reason_target: Tensor, action_raw: Tensor, reason_raw: Tensor, action_deploy: Tensor, reason_deploy: Tensor, formal: Mapping[str, Any]) -> dict[str, Any]:
        branch_deltas = {
            name: {
                "action": self._float_list(formal["branches"][name]["action"][index] - formal["action_final"][index]),
                "reason": self._float_list(formal["branches"][name]["reason"][index] - formal["reason_final"][index]),
            }
            for name in tuple(BRANCH_NAMES)
        }
        return {
            "file_name": file_name,
            "case_id": self._case_id("failure", file_name),
            "data": {
                "labels": {"action": self._float_list(action_target[index]), "reason": self._float_list(reason_target[index])},
                "raw_predictions": {"action": self._bool_list(action_raw[index]), "reason": self._bool_list(reason_raw[index])},
                "deploy_predictions": {"action": self._bool_list(action_deploy[index]), "reason": self._bool_list(reason_deploy[index])},
                "branch_deltas": branch_deltas,
            },
        }

    @staticmethod
    def _evidence_targets(action_target: Tensor, reason_target: Tensor, action_deploy: Tensor, reason_deploy: Tensor) -> tuple[tuple[str, int], ...]:
        targets: list[tuple[str, int]] = []
        for target_type, labels, decision in (("action", action_target, action_deploy), ("reason", reason_target, reason_deploy)):
            positives = [index for index, value in enumerate(labels.tolist()) if value == 1.0]
            targets.extend((target_type, index) for index in (positives if positives else [index for index, value in enumerate(decision.tolist()) if bool(value)]))
        return tuple(targets)

    def _evidence_row(self, *, file_name: str, index: int, target_type: str, target_id: int, target_name: str, formal: Mapping[str, Any]) -> dict[str, Any]:
        family = formal[target_type]
        deletion = family["deletion"][index, target_id]
        selected_slots = sorted(range(_SLOT_COUNT), key=lambda slot: (-float(deletion[slot].abs()), slot))[: self.top_slots]
        if not selected_slots or float(deletion[selected_slots[0]].abs()) <= 0.0:
            return {
                "file_name": file_name,
                "case_id": self._case_id("evidence-unavailable", file_name, target_type, target_id),
                "data": {
                    "available": False,
                    "unavailable_reason": "zero_analytical_deletion",
                    "target": {"type": target_type, "id": target_id, "name": target_name},
                },
            }
        masks = {str(slot): self._nested_floats(formal["masks"][index, slot]) for slot in selected_slots}
        attributes = {str(slot): self._slot_attributes(formal["attributes"], index, slot) for slot in selected_slots}
        unary = {str(slot): self._scalar(family["unary"][index, target_id, slot]) for slot in selected_slots}
        incident = {str(slot): self._scalar(family["incident"][index, target_id, slot]) for slot in selected_slots}
        analytical = {str(slot): self._scalar(family["deletion"][index, target_id, slot]) for slot in selected_slots}
        for slot in selected_slots:
            if abs(analytical[str(slot)] - (unary[str(slot)] + incident[str(slot)])) > self.reconstruction_tolerance:
                raise ValueError("selected analytical deletion identity failed")
        final = self._scalar(formal[f"{target_type}_final"][index, target_id])
        reconstructed = self._scalar(family["global"][index, target_id] + family["unary"][index, target_id].sum() + family["pairwise"][index, target_id].sum())
        reconstruction_error = final - reconstructed
        if abs(reconstruction_error) > self.reconstruction_tolerance:
            raise ValueError("selected contribution reconstruction failed")
        return {
            "file_name": file_name,
            "case_id": self._case_id("evidence", file_name, target_type, target_id),
            "data": {
                "available": True,
                "unavailable_reason": None,
                "target": {"type": target_type, "id": target_id, "name": target_name},
                "selected_slots": selected_slots,
                "masks": masks,
                "attributes": attributes,
                "contributions": {
                    "global": self._scalar(family["global"][index, target_id]),
                    "unary": unary,
                    "pairwise": self._float_list(family["pairwise"][index, target_id]),
                    "pair_indices": self._nested_ints(family["pair_indices"]),
                    "pairwise_incident": incident,
                    "analytical_deletion": analytical,
                    "final": final,
                    "reconstruction_error": reconstruction_error,
                },
            },
        }

    @staticmethod
    def _slot_attributes(attributes: Mapping[str, Any], index: int, slot: int) -> dict[str, Any]:
        named = slot < _NAMED_SLOT_COUNT
        return {
            "area": RAELCaseExportCollector._scalar(attributes["slot_area"][index, slot]),
            "centroid": RAELCaseExportCollector._float_list(attributes["slot_centroid"][index, slot]),
            "scale": RAELCaseExportCollector._scalar(attributes["slot_scale"][index, slot]),
            "activity": RAELCaseExportCollector._scalar(attributes["slot_activity"][index, slot]),
            "presence": RAELCaseExportCollector._scalar(attributes["slot_presence"][index, slot]),
            "observability": RAELCaseExportCollector._scalar(attributes["slot_observability"][index, slot]),
            "reliability": RAELCaseExportCollector._scalar(attributes["slot_reliability"][index, slot]),
            "type_probabilities": RAELCaseExportCollector._float_list(attributes["slot_type_probs"][index, slot]) if named else None,
            "state_probabilities": RAELCaseExportCollector._float_list(attributes["slot_state_probs"][index, slot]) if named else None,
            "sector_probabilities": {
                "horizontal": RAELCaseExportCollector._float_list(attributes["slot_sector_probs"]["horizontal"][index, slot]),
                "depth": RAELCaseExportCollector._float_list(attributes["slot_sector_probs"]["depth"][index, slot]),
            },
        }

    @staticmethod
    def _scalar(value: Tensor) -> float:
        result = float(value.detach().cpu().item())
        if not math.isfinite(result):
            raise ValueError("case export values must be finite")
        return result

    @staticmethod
    def _float_list(value: Tensor) -> list[float]:
        result = [float(item) for item in value.detach().cpu().tolist()]
        if not all(math.isfinite(item) for item in result):
            raise ValueError("case export values must be finite")
        return result

    @staticmethod
    def _nested_ints(value: Tensor) -> list[list[int]]:
        return [[int(item) for item in row] for row in value.detach().cpu().tolist()]

    @staticmethod
    def _bool_list(value: Tensor) -> list[bool]:
        return [bool(item) for item in value.detach().cpu().tolist()]

    @staticmethod
    def _nested_floats(value: Tensor) -> list[list[float]]:
        if tuple(value.shape) != _GRID_HW:
            raise ValueError("selected slot mask must retain the exact 45x80 grid")
        nested = value.detach().cpu().tolist()
        if not all(math.isfinite(float(item)) for row in nested for item in row):
            raise ValueError("selected slot masks must be finite")
        return [[float(item) for item in row] for row in nested]

    @staticmethod
    def _validate_provenance(provenance: Mapping[str, Any], *, sample_count: int) -> dict[str, Any]:
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        required = ("schema_version", "producer", "source_fingerprint_sha256", "config_sha256", "epoch", "sample_count")
        if any(field not in provenance for field in required):
            raise ValueError("provenance is incomplete for P18")
        if provenance["schema_version"] != "rael-artifact-v1":
            raise ValueError("provenance schema_version must match P18")
        if not isinstance(provenance["producer"], str) or not provenance["producer"].strip():
            raise ValueError("provenance producer is required")
        for field in ("source_fingerprint_sha256", "config_sha256"):
            if not isinstance(provenance[field], str) or not _SHA256.fullmatch(provenance[field]):
                raise ValueError(f"provenance {field} must be lowercase sha256")
        if isinstance(provenance["epoch"], bool) or not isinstance(provenance["epoch"], int) or provenance["epoch"] < 0:
            raise ValueError("provenance epoch must be a nonnegative integer")
        if provenance["sample_count"] != sample_count:
            raise ValueError("provenance sample_count must equal consumed formal samples")
        return {field: copy.deepcopy(provenance[field]) for field in required}
