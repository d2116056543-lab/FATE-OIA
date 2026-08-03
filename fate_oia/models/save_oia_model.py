from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fate_oia.losses import save_faithfulness_losses as save_utility_bridge

from .meter_calalign_foundation import METERCalAlignFoundation
from .save_action_evidence import (
    SAVEActionEvidence,
    _FoundationFirewalledEvidence,
    _direction_preserving_cap,
    evidence_ramp,
)
from .save_multiscale_field import SAVEMultiscaleField
from .save_predicate_measurement import SAVEPredicateMeasurement
from .save_reason_decoder import SAVEReasonDecoder
from .save_utility_bridge import SAVEUtilityBridge


class SAVEOIAModel(nn.Module):
    """Direct-image SAVE decoder over one frozen CalAlign/DINO field."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        use_mock_dino: bool = False,
        schema_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.foundation = METERCalAlignFoundation(
            dim=dim,
            action_dim=action_dim,
            reason_dim=reason_dim,
            selected_layers=selected_layers,
            pretrained_weights=pretrained_weights,
            use_mock_dino=use_mock_dino,
        )
        self.multiscale_field = SAVEMultiscaleField(
            dim=dim,
            input_dim=dim,
            selected_layers=selected_layers,
        )
        resolved_schema_path = (
            Path(schema_path)
            if schema_path is not None
            else Path(__file__).resolve().parents[2]
            / "configs"
            / "save_factor_schema.yaml"
        )
        self.predicate_measurement = SAVEPredicateMeasurement(
            dim=dim,
            factor_dim=reason_dim,
            num_layers=len(selected_layers),
            schema_path=resolved_schema_path,
        )
        self.predicate_measurement.schema_path = resolved_schema_path
        self.action_evidence = SAVEActionEvidence(
            dim=dim,
            action_dim=action_dim,
        )
        self.utility_bridge = SAVEUtilityBridge(
            dim=dim,
            action_dim=action_dim,
            factor_dim=reason_dim,
        )
        self.reason_decoder = SAVEReasonDecoder(
            dim=dim,
            reason_dim=reason_dim,
            action_dim=action_dim,
            foundation=self.foundation,
        )
        self.encode_call_count = 0

    def train(self, mode: bool = True) -> "SAVEOIAModel":
        super().train(mode)
        self.foundation.dino.eval()
        return self

    def encode_images(self, images: Tensor) -> dict[str, Any]:
        """Encode images exactly once through the frozen DINO field."""
        if not isinstance(images, Tensor):
            raise TypeError("images must be a tensor")
        self.encode_call_count += 1
        self.foundation.dino.eval()
        with torch.no_grad():
            encoded = self.foundation.encode_images(images)
        visual = self.multiscale_field(encoded["patch_tokens_by_layer"])
        return {**encoded, **visual, "dino_field_frozen": True}

    @staticmethod
    def _field_value(field: Mapping[str, Any], name: str) -> Tensor:
        value = field.get(name)
        if not isinstance(value, Tensor):
            raise ValueError(f"SAVE field is missing tensor {name!r}")
        return value

    def _pre_detail_action_contribution(
        self,
        global_read: Mapping[str, Tensor],
        detail_field: Tensor,
    ) -> Tensor:
        """Build a real patch contribution before the utility/detail stage.

        The utility teacher must select patches before the formal detail read,
        but it still needs an action-conditioned contribution.  The global
        inquiry and learned patch-value route provide that contribution without
        materializing a second field or invoking the detail inquiry twice.
        """
        action_global_token = global_read["action_global_token"]
        global_attention = global_read["action_global_attention"]
        action_value = self.action_evidence.patch_action_value(action_global_token)
        patch_value = self.action_evidence.patch_value(detail_field)
        patch_score = torch.einsum(
            "bad,bnd->ban", action_value, patch_value
        ) / (self.action_evidence.dim**0.5)
        return global_attention * patch_score

    @staticmethod
    def _teacher_sparse_predicate_map(predicate_map: Tensor) -> Tensor:
        """Keep real predicate mass in one deterministic spatial sector.

        The matched-control contract needs a bounded local deletion.  This is
        a sparse view of the measured map, not a target: values are unchanged,
        while mass outside the best measured sector is excluded from teacher
        candidate selection.
        """
        batch, factors, patches = predicate_map.shape
        height, width = 45, 80
        if patches != height * width:
            raise ValueError("SAVE teacher map requires the 45x80 patch grid")
        indices = torch.arange(patches, device=predicate_map.device)
        rows = torch.div(indices, width, rounding_mode="floor")
        columns = indices.remainder(width)
        sectors = (columns * 3 // width).clamp_max(2) * 5 + (
            rows * 5 // height
        ).clamp_max(4)
        sparse = predicate_map.new_zeros(predicate_map.shape)
        keep = min(32, patches)
        for sample in range(batch):
            for factor in range(factors):
                values = predicate_map[sample, factor].float().clamp_min(0.0)
                sector_mass = torch.zeros(15, device=values.device)
                sector_mass.scatter_add_(0, sectors, values)
                sector = int(torch.argmax(sector_mass).item())
                candidates = torch.nonzero(sectors == sector, as_tuple=False).flatten()
                selected = candidates[
                    torch.topk(values[candidates], k=min(keep, candidates.numel())).indices
                ]
                sparse[sample, factor, selected] = predicate_map[sample, factor, selected]
        return sparse

    def _teacher_decoder(
        self,
        *,
        global_read: Mapping[str, Tensor],
        detail_field: Tensor,
        base_action_logits: Tensor,
        label_attention: Tensor,
        predicate_map: Tensor,
        predicate_reliability: Tensor,
    ) -> tuple[Any, list[dict[str, Any]]]:
        calls: list[dict[str, Any]] = []

        def decode(
            teacher_field: Tensor,
            deleted_patches: Tensor,
            *,
            sample_index: int,
            action_index: int,
            factor_index: int,
            variant: str,
        ) -> dict[str, Tensor | str]:
            if teacher_field.ndim != 3 or teacher_field.shape[0] != 1:
                raise ValueError("SAVE teacher field must be [1,N,D]")
            sample = int(sample_index)
            deleted = torch.as_tensor(
                deleted_patches, device=teacher_field.device, dtype=torch.long
            )
            variant_field = teacher_field.clone()
            if deleted.numel() > 0:
                variant_field[:, deleted, :] = 0.0
            global_sample = {
                key: value[sample : sample + 1]
                for key, value in global_read.items()
            }
            detail = self.action_evidence._read_detail(
                global_sample,
                variant_field,
                calalign_action_attention=label_attention[sample : sample + 1],
                predicate_map=predicate_map[sample : sample + 1],
                predicate_candidate_weight=None,
                predicate_reliability=predicate_reliability[sample : sample + 1],
                predicate_gain=None,
            )
            raw = detail["action_evidence_raw"]
            kappa = self.action_evidence._kappa(base_action_logits[sample : sample + 1])
            bounded = kappa.view(1, -1) * torch.tanh(
                raw / kappa.view(1, -1).clamp_min(torch.finfo(raw.dtype).tiny)
            )
            calls.append(
                {
                    "variant": variant,
                    "sample_index": sample,
                    "action_index": int(action_index),
                    "factor_index": int(factor_index),
                    "field_data_ptr": int(teacher_field.data_ptr()),
                    "source_detail_data_ptr": int(detail_field.data_ptr()),
                }
            )
            return {
                "action_logits": base_action_logits[sample : sample + 1] + bounded,
                "action_evidence_raw": raw,
                "variant": variant,
            }

        return decode, calls

    def _staged_action_evidence(
        self,
        *,
        action_nodes: Tensor,
        action_logits_base: Tensor,
        label_attention: Tensor,
        global_field: Tensor,
        detail_field: Tensor,
        predicate: Mapping[str, Any],
        progress: float,
        optimizer_update: int | None,
        action_targets: Tensor | None,
        run_teacher: bool | None,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        # The order is contractual: global read, utility, then detail read.
        global_read = self.action_evidence.read_global(action_nodes, global_field)
        predicate_token = predicate["predicate_token_action"]
        predicate_map = predicate["predicate_map_action"]
        predicate_state_prob = predicate["predicate_state_prob_action"]
        predicate_reliability = predicate["predicate_reliability_action"]
        teacher_predicate_map = self._teacher_sparse_predicate_map(predicate_map)
        global_attention = global_read["action_global_attention"]
        overlap = torch.einsum("ban,bfn->baf", global_attention, predicate_map)
        teacher_overlap = torch.einsum(
            "ban,bfn->baf", global_attention, teacher_predicate_map
        )
        detail_summary = detail_field.mean(dim=1)
        query = global_read["action_global_token"] + detail_summary.unsqueeze(1)
        similarity = F.cosine_similarity(
            query.unsqueeze(2), predicate_token.unsqueeze(1), dim=-1
        )
        teacher_decoder, teacher_calls = self._teacher_decoder(
            global_read=global_read,
            detail_field=detail_field,
            base_action_logits=action_logits_base,
            label_attention=label_attention,
            predicate_map=teacher_predicate_map,
            predicate_reliability=predicate_reliability,
        )
        utility = self.utility_bridge(
            # Utility supervision is a private bridge: it must not update the
            # CalAlign foundation through shared action/global inputs.
            global_read["action_global_token"].detach(),
            predicate_token,
            predicate_state_prob=predicate_state_prob,
            predicate_reliability=predicate_reliability,
            base_predicate_overlap=teacher_overlap.detach(),
            global_detail_query_similarity=similarity.detach(),
            detail_field=detail_field,
            predicate_map=teacher_predicate_map,
            action_contribution=self._pre_detail_action_contribution(
                global_read, detail_field
            ),
            base_action_logits=action_logits_base,
            action_targets=action_targets,
            optimizer_update=optimizer_update,
            run_teacher=run_teacher,
            teacher_decoder=teacher_decoder,
        )
        detail_read = self.action_evidence.read_detail(
            global_read,
            detail_field,
            calalign_action_attention=label_attention,
            predicate_map=predicate_map,
            predicate_candidate_weight=utility["predicate_candidate_weight"],
            predicate_reliability=predicate_reliability,
        )
        contribution = save_utility_bridge.compute_named_unnamed_contributions(
            detail_read["action_patch_contribution"],
            utility["predicate_candidate_weight"],
            predicate_map,
            named_eligibility=predicate["predicate_named_mask"]
            if "predicate_named_mask" in predicate
            else self.predicate_measurement.named_mask,
        )
        raw_evidence = detail_read["action_evidence_raw"].float()
        named_contribution = contribution["action_named_contribution"]
        unnamed_contribution = raw_evidence - named_contribution.sum(-1)
        named_responsibility = contribution["action_named_responsibility"]
        unnamed_responsibility = 1.0 - named_responsibility.sum(2)
        contribution["action_unnamed_contribution"] = unnamed_contribution
        contribution["action_unnamed_responsibility"] = unnamed_responsibility
        contribution["action_responsibility_sum"] = (
            named_responsibility.sum(2) + unnamed_responsibility
        )
        contribution["action_conservation_error"] = (
            named_contribution.sum(-1) + unnamed_contribution - raw_evidence
        )

        kappa = self.action_evidence._kappa(action_logits_base)
        bounded = kappa.view(1, -1) * torch.tanh(
            detail_read["action_evidence_raw"]
            / kappa.view(1, -1).clamp_min(torch.finfo(action_logits_base.dtype).tiny)
        )
        ramp = evidence_ramp(progress)
        gain = torch.sigmoid(self.action_evidence.evidence_gain_raw).to(
            action_logits_base
        ).view(1, -1)
        delta = ramp * gain * bounded
        uncapped = action_logits_base + delta
        final = _direction_preserving_cap(
            uncapped,
            action_logits_base,
            ramp=ramp,
            cap=self.action_evidence.action_logit_cap,
        )
        auxiliary_raw = _FoundationFirewalledEvidence.apply(
            detail_read["action_evidence_raw"],
            action_nodes,
            global_field,
            detail_field,
            label_attention,
            predicate_map,
            utility["predicate_candidate_weight"],
            predicate_reliability,
            None,
            self.action_evidence,
            *self.action_evidence._evidence_parameters(),
        )
        auxiliary_bounded = kappa.view(1, -1) * torch.tanh(
            auxiliary_raw
            / kappa.view(1, -1).clamp_min(torch.finfo(action_logits_base.dtype).tiny)
        )
        output = {
            **detail_read,
            **utility,
            **contribution,
            "action_logits_base": action_logits_base,
            "action_logits_visual": action_logits_base,
            "action_evidence_bounded": bounded,
            "action_evidence_delta_unramped": bounded,
            "action_evidence_delta": delta,
            "action_logits_final": final,
            "action_logits_evidence_aux": action_logits_base.detach() + auxiliary_bounded,
            "action_logits_evidence_auxiliary": action_logits_base.detach() + auxiliary_bounded,
            "action_evidence_aux_raw": auxiliary_raw,
            "action_evidence_aux_bounded": auxiliary_bounded,
            "action_correction_kappa": kappa.view(1, -1),
            "action_evidence_gain": gain,
            "action_credit_ramp": action_logits_base.new_tensor(ramp),
            "action_logit_uncapped_final": uncapped,
            "action_predicate_overlap": overlap,
            "global_detail_query_similarity": similarity,
        }
        teacher_plan = utility.get("teacher_plan")
        if isinstance(teacher_plan, dict):
            teacher_plan["teacher_decoder_calls"] = teacher_calls
        return output, {"global_read": global_read, "utility": utility, "detail_read": detail_read}

    def decode_from_field(
        self,
        field: Mapping[str, Any],
        *,
        progress: float = 1.0,
        diagnostic_modes: tuple[str, ...] = (),
        optimizer_update: int | None = None,
        action_targets: Tensor | None = None,
        run_teacher: bool | None = None,
    ) -> dict[str, Any]:
        patch_tokens = self._field_value(field, "patch_tokens_by_layer")
        decoded = self.foundation.decode_foundation(field)
        visual = self.multiscale_field(patch_tokens)
        predicate = self.predicate_measurement(
            decoded["factor_base_nodes"],
            patch_tokens,
            progress,
        )
        evidence, staged = self._staged_action_evidence(
            action_nodes=decoded["action_nodes"],
            action_logits_base=decoded["action_logits_calalign"],
            label_attention=decoded["label_attention"],
            global_field=visual["global_field"],
            detail_field=visual["detail_field"],
            predicate=predicate,
            progress=progress,
            optimizer_update=optimizer_update,
            action_targets=action_targets,
            run_teacher=run_teacher,
        )
        reason = self.reason_decoder(
            reason_logits_calalign=decoded["reason_logits_calalign"],
            reason_nodes=decoded["factor_base_nodes"],
            global_field=visual["global_field"],
            detail_field=visual["detail_field"],
            factor_measurement_token=predicate["predicate_token_action"],
            factor_evidence_map=predicate["predicate_map_action"],
            factor_reliability=predicate["predicate_reliability_action"],
            predicate_token=predicate["predicate_token_action"],
            predicate_map=predicate["predicate_map_action"],
            predicate_state_prob=predicate["predicate_state_prob_action"],
            action_evidence_overlap=evidence["action_predicate_overlap"],
            progress=progress,
            update_running_stats=False,
        )
        clean_action_kappa = evidence["action_correction_kappa"]
        clean_action_bounded = clean_action_kappa * torch.tanh(
            reason["action_logits_clean"]
            / clean_action_kappa.clamp_min(
                torch.finfo(reason["action_logits_clean"].dtype).tiny
            )
        )
        clean_action_ramp = evidence_ramp(progress)
        clean_action_delta = clean_action_ramp * clean_action_bounded
        action_uncapped_final = evidence["action_logits_final"] + clean_action_delta
        action_logits_final = _direction_preserving_cap(
            action_uncapped_final,
            decoded["action_logits_calalign"],
            ramp=clean_action_ramp,
            cap=self.action_evidence.action_logit_cap,
        )
        output: dict[str, Any] = {
            **decoded,
            **visual,
            **predicate,
            **evidence,
            **reason,
            "patch_tokens_by_layer": patch_tokens,
            "action_logits_evidence_final": evidence["action_logits_final"],
            "action_logits_final": action_logits_final,
            "action_clean_reason_bounded": clean_action_bounded,
            "action_clean_reason_delta": clean_action_delta,
            "action_clean_reason_kappa": clean_action_kappa,
            "action_clean_reason_ramp": decoded["action_logits_calalign"].new_tensor(
                clean_action_ramp
            ),
            "action_logit_uncapped_final": action_uncapped_final,
            "reason_logits_pu_private": reason["reason_logits_private_direct"],
            "branch_logits": {
                "base": decoded["action_logits_calalign"],
                "final": action_logits_final,
                "evidence_aux": evidence["action_logits_evidence_aux"],
            },
            "audit_outputs": {
                "field_reused": True,
                "one_dino_call": self.encode_call_count == 1,
                "dino_field_frozen": True,
                "action_anchor": "calalign",
                "reason_anchor": "calalign",
                "evidence_order": ("read_global", "utility_bridge", "read_detail"),
                "evidence_forward_called": False,
                "diagnostic_modes": tuple(diagnostic_modes),
            },
            "utility_teacher_plan": evidence.get("teacher_plan"),
            "counterfactual_teacher": evidence.get("teacher_plan"),
            "test_forward_image_only": False,
            "staged_action_evidence": staged,
        }
        return output

    def forward_test(self, images: Tensor, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if args or kwargs:
            raise ValueError("SAVE test forward is image-only")
        field = self.encode_images(images)
        output = self.decode_from_field(field, progress=1.0)
        output["test_forward_image_only"] = True
        return output

    def forward(
        self,
        images: Tensor,
        *,
        progress: float = 1.0,
        action_targets: Tensor | None = None,
        optimizer_update: int | None = None,
        run_teacher: bool | None = None,
    ) -> dict[str, Any]:
        field = self.encode_images(images)
        output = self.decode_from_field(
            field,
            progress=progress,
            optimizer_update=optimizer_update,
            action_targets=action_targets,
            run_teacher=run_teacher,
        )
        output["test_forward_image_only"] = False
        if action_targets is not None:
            output["losses"] = save_utility_bridge.save_faithfulness_losses(
                output, action_targets
            )
        return output


__all__ = ["SAVEOIAModel"]
