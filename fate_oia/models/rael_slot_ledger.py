"""Two-iteration competitive evidence ledger for RAEL-OIA P5.

The ledger binds a fixed internal set of 21 slots to the P3 visual field.
Competition is intentionally normalized over slots for every patch, so each
patch is allocated once rather than letting every slot independently attend to
the whole image.  It reads the P3 precomputed K/V tensors layer by layer and
never materializes a ``[B, 21, 4, N, D]`` tensor.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any
import weakref

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LedgerSlotSpec:
    """Immutable identity metadata for an internal evidence slot."""

    index: int
    family: str
    name: str
    public: bool
    fixed_identity: bool


_PUBLIC_EVIDENCE_ISSUER = object()
_INTERNAL_DIAGNOSTIC_ISSUER = object()


class PublicEvidenceView:
    """Ledger-issued public evidence with a sealed, versioned payload.

    The three public tensors are independent ``clone`` operations rather than
    views into the ledger's internal state.  Cloning preserves their autograd
    path.  Detached private snapshots catch ``.data`` writes that bypass a
    Tensor's ``_version`` counter; the ledger registry also records their
    identities so replacement cannot disable that integrity check.
    """

    __slots__ = (
        "__tokens",
        "__masks",
        "__valid_mask",
        "__slot_indices",
        "__provenance",
        "__token_integrity_snapshot",
        "__mask_integrity_snapshot",
        "__valid_mask_integrity_snapshot",
        "__sealed",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        _issuer: object,
        tokens: Tensor,
        masks: Tensor,
        valid_mask: Tensor,
        slot_indices: tuple[int, ...],
        provenance: object,
    ) -> None:
        if _issuer is not _PUBLIC_EVIDENCE_ISSUER:
            raise TypeError("PublicEvidenceView can only be issued by RAELSlotLedger")
        object.__setattr__(self, "_PublicEvidenceView__tokens", tokens)
        object.__setattr__(self, "_PublicEvidenceView__masks", masks)
        object.__setattr__(self, "_PublicEvidenceView__valid_mask", valid_mask)
        object.__setattr__(self, "_PublicEvidenceView__slot_indices", slot_indices)
        object.__setattr__(self, "_PublicEvidenceView__provenance", provenance)
        object.__setattr__(self, "_PublicEvidenceView__token_integrity_snapshot", tokens.detach().clone())
        object.__setattr__(self, "_PublicEvidenceView__mask_integrity_snapshot", masks.detach().clone())
        object.__setattr__(
            self,
            "_PublicEvidenceView__valid_mask_integrity_snapshot",
            valid_mask.detach().clone(),
        )
        object.__setattr__(self, "_PublicEvidenceView__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        del value
        raise AttributeError(f"{type(self).__name__} has sealed read-only fields: {name}")

    @property
    def tokens(self) -> Tensor:
        return object.__getattribute__(self, "_PublicEvidenceView__tokens")

    @property
    def masks(self) -> Tensor:
        return object.__getattribute__(self, "_PublicEvidenceView__masks")

    @property
    def valid_mask(self) -> Tensor:
        return object.__getattribute__(self, "_PublicEvidenceView__valid_mask")

    @property
    def slot_indices(self) -> tuple[int, ...]:
        return object.__getattribute__(self, "_PublicEvidenceView__slot_indices")


def _public_view_private_payload(
    evidence: PublicEvidenceView,
) -> tuple[Tensor, Tensor, Tensor, tuple[int, ...], object]:
    """Read sealed payload only inside the ledger's validation boundary."""

    return (
        object.__getattribute__(evidence, "_PublicEvidenceView__tokens"),
        object.__getattribute__(evidence, "_PublicEvidenceView__masks"),
        object.__getattribute__(evidence, "_PublicEvidenceView__valid_mask"),
        object.__getattribute__(evidence, "_PublicEvidenceView__slot_indices"),
        object.__getattribute__(evidence, "_PublicEvidenceView__provenance"),
    )


def _public_view_integrity_snapshots(evidence: PublicEvidenceView) -> tuple[Tensor, Tensor, Tensor]:
    """Read immutable detached snapshots only inside the ledger boundary."""

    return (
        object.__getattribute__(evidence, "_PublicEvidenceView__token_integrity_snapshot"),
        object.__getattribute__(evidence, "_PublicEvidenceView__mask_integrity_snapshot"),
        object.__getattribute__(evidence, "_PublicEvidenceView__valid_mask_integrity_snapshot"),
    )


class InternalLedgerDiagnostics:
    """Detached audit-only diagnostics that cannot alias live ledger tensors."""

    __slots__ = (
        "__slot_masks",
        "__slot_tokens",
        "__slot_activity",
        "__slot_area",
        "__slot_centroid",
        "__slot_scale",
        "__slot_nonempty",
        "__layer_weights_one",
        "__layer_weights_two",
        "__visual_logits_one",
        "__assignment_one",
        "__visual_logits_two",
        "__logits_two",
        "__assignment_two",
        "__slot_specs",
        "__provenance",
        "__sealed",
    )

    def __init__(self, *, _issuer: object, provenance: object, **values: Any) -> None:
        if _issuer is not _INTERNAL_DIAGNOSTIC_ISSUER:
            raise TypeError("InternalLedgerDiagnostics is only available through RAELSlotLedger.audit_diagnostics")
        tensor_names = (
            "slot_masks",
            "slot_tokens",
            "slot_activity",
            "slot_area",
            "slot_centroid",
            "slot_scale",
            "slot_nonempty",
            "layer_weights_one",
            "layer_weights_two",
            "visual_logits_one",
            "assignment_one",
            "visual_logits_two",
            "logits_two",
            "assignment_two",
        )
        for name in tensor_names:
            value = values[name]
            if not torch.is_tensor(value):
                raise TypeError(f"diagnostic {name} must be a Tensor")
            object.__setattr__(self, f"_InternalLedgerDiagnostics__{name}", value.detach().clone())
        object.__setattr__(self, "_InternalLedgerDiagnostics__slot_specs", tuple(values["slot_specs"]))
        object.__setattr__(self, "_InternalLedgerDiagnostics__provenance", provenance)
        object.__setattr__(self, "_InternalLedgerDiagnostics__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        del value
        raise AttributeError(f"{type(self).__name__} has sealed audit fields: {name}")

    @property
    def allow_contribution(self) -> bool:
        return False

    @property
    def allow_cf(self) -> bool:
        return False

    @property
    def slot_masks(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__slot_masks")

    @property
    def slot_tokens(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__slot_tokens")

    @property
    def slot_activity(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__slot_activity")

    @property
    def slot_area(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__slot_area")

    @property
    def slot_centroid(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__slot_centroid")

    @property
    def slot_scale(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__slot_scale")

    @property
    def slot_nonempty(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__slot_nonempty")

    @property
    def layer_weights_iteration1(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__layer_weights_one")

    @property
    def layer_weights_iteration2(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__layer_weights_two")

    @property
    def iteration1_visual_logits(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__visual_logits_one")

    @property
    def iteration1_assignment(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__assignment_one")

    @property
    def iteration2_visual_logits(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__visual_logits_two")

    @property
    def iteration2_logits(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__logits_two")

    @property
    def iteration2_assignment(self) -> Tensor:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__assignment_two")

    @property
    def slot_specs(self) -> tuple[LedgerSlotSpec, ...]:
        return object.__getattribute__(self, "_InternalLedgerDiagnostics__slot_specs")


@dataclass(frozen=True)
class _IssuedPublicEvidence:
    tokens: Tensor
    masks: Tensor
    valid_mask: Tensor
    slot_indices: tuple[int, ...]
    provenance: object
    token_integrity_snapshot: Tensor
    mask_integrity_snapshot: Tensor
    valid_mask_integrity_snapshot: Tensor
    token_version: int
    mask_version: int
    valid_mask_version: int


class RAELSlotLedger(nn.Module):
    """Bind P3 shared K/V fields into exactly two competitive slot updates."""

    FORMAL_DIM = 384
    FORMAL_GRID_HW = (45, 80)
    FORMAL_NUM_LAYERS = 4
    ENTITY_SLOT_COUNT = 12
    ROAD_SLOT_NAMES = (
        "drivable_left",
        "drivable_center",
        "drivable_right",
        "boundary_left",
        "boundary_right",
    )
    LATENT_SLOT_COUNT = 3
    LATENT_START = ENTITY_SLOT_COUNT + len(ROAD_SLOT_NAMES)
    BACKGROUND_INDEX = 20
    INTERNAL_SLOT_COUNT = 21
    PUBLIC_SLOT_COUNT = 20
    ITERATIONS = 2
    MASK_BIAS = 0.5
    parameter_owner = "slot_ledger"

    def __init__(
        self,
        dim: int = FORMAL_DIM,
        num_layers: int = FORMAL_NUM_LAYERS,
        eps: float = 1.0e-6,
        mask_bias: float = MASK_BIAS,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_layers != self.FORMAL_NUM_LAYERS:
            raise ValueError("RAEL ledger requires exactly four P3 layers")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if abs(float(mask_bias) - self.MASK_BIAS) > 1.0e-12:
            raise ValueError("RAEL ledger fixes lambda_mask_bias at 0.5")

        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.eps = float(eps)
        self.mask_bias = float(mask_bias)
        self.slot_specs = self._build_slot_specs()

        self.slot_queries = nn.Parameter(torch.empty(self.INTERNAL_SLOT_COUNT, self.dim))
        self.query_norm = nn.LayerNorm(self.dim)
        self.layer_query_projection = nn.Linear(self.dim, self.dim, bias=False)
        self.layer_global_projection = nn.Linear(self.dim, self.dim, bias=False)
        self.layer_score = nn.Linear(self.dim, 1, bias=False)
        self.visual_query_projection = nn.Linear(self.dim, self.dim, bias=False)
        self.slot_gru = nn.GRUCell(self.dim, self.dim)

        # Global context is a separate readout, never an additional public
        # evidence slot and never part of the 21-way competitive softmax.
        self.global_context_projection = nn.Linear(self.dim, self.dim, bias=False)
        self.global_context_score = nn.Linear(self.dim, 1, bias=False)
        self.global_context_norm = nn.LayerNorm(self.dim)

        self.register_buffer(
            "mirror_slot_permutation",
            torch.tensor(self._mirror_slot_permutation(), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "road_slot_indices",
            torch.tensor(tuple(range(self.ENTITY_SLOT_COUNT, self.ENTITY_SLOT_COUNT + 5)), dtype=torch.long),
            persistent=True,
        )
        self._issued_public_views: weakref.WeakKeyDictionary[PublicEvidenceView, _IssuedPublicEvidence] = weakref.WeakKeyDictionary()
        self._diagnostics_by_public_view: weakref.WeakKeyDictionary[PublicEvidenceView, InternalLedgerDiagnostics] = weakref.WeakKeyDictionary()
        self.reset_parameters()

    @classmethod
    def _build_slot_specs(cls) -> tuple[LedgerSlotSpec, ...]:
        specs: list[LedgerSlotSpec] = []
        for index in range(cls.ENTITY_SLOT_COUNT):
            specs.append(
                LedgerSlotSpec(
                    index=index,
                    family="entity",
                    name=f"entity_slot_{index:02d}",
                    public=True,
                    fixed_identity=False,
                )
            )
        for offset, name in enumerate(cls.ROAD_SLOT_NAMES):
            specs.append(
                LedgerSlotSpec(
                    index=cls.ENTITY_SLOT_COUNT + offset,
                    family="road",
                    name=name,
                    public=True,
                    fixed_identity=True,
                )
            )
        for offset in range(cls.LATENT_SLOT_COUNT):
            specs.append(
                LedgerSlotSpec(
                    index=cls.LATENT_START + offset,
                    family="latent",
                    name=f"latent_slot_{offset:02d}",
                    public=True,
                    fixed_identity=False,
                )
            )
        specs.append(
            LedgerSlotSpec(
                index=cls.BACKGROUND_INDEX,
                family="background",
                name="background_sink",
                public=False,
                fixed_identity=True,
            )
        )
        if len(specs) != cls.INTERNAL_SLOT_COUNT:
            raise RuntimeError("RAEL internal slot schema is malformed")
        return tuple(specs)

    @classmethod
    def _mirror_slot_permutation(cls) -> tuple[int, ...]:
        permutation = list(range(cls.INTERNAL_SLOT_COUNT))
        road_start = cls.ENTITY_SLOT_COUNT
        permutation[road_start + 0] = road_start + 2
        permutation[road_start + 2] = road_start + 0
        permutation[road_start + 3] = road_start + 4
        permutation[road_start + 4] = road_start + 3
        return tuple(permutation)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)
        for projection in (
            self.layer_query_projection,
            self.layer_global_projection,
            self.layer_score,
            self.visual_query_projection,
            self.global_context_projection,
            self.global_context_score,
        ):
            nn.init.xavier_uniform_(projection.weight)
        for norm in (self.query_norm, self.global_context_norm):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

    @classmethod
    def formal_metadata(cls) -> dict[str, int | tuple[int, int] | tuple[str, ...]]:
        """Return the immutable full-resolution ledger contract."""

        return {
            "dim": cls.FORMAL_DIM,
            "grid_hw": cls.FORMAL_GRID_HW,
            "num_layers": cls.FORMAL_NUM_LAYERS,
            "internal_slot_count": cls.INTERNAL_SLOT_COUNT,
            "public_slot_count": cls.PUBLIC_SLOT_COUNT,
            "iterations": cls.ITERATIONS,
            "road_slots": cls.ROAD_SLOT_NAMES,
        }

    def owned_parameter_names(self) -> tuple[str, ...]:
        """Expose a stable P13 ownership boundary without training behavior."""

        return tuple(name for name, _ in self.named_parameters())

    @staticmethod
    def _uses_native_cuda_bfloat16(value: Tensor) -> bool:
        return (
            value.device.type == "cuda"
            and value.dtype == torch.bfloat16
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )

    @classmethod
    def _working_autocast_context(cls, value: Tensor):
        if cls._uses_native_cuda_bfloat16(value):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _working_tensors(self, prepared: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor, tuple[int, int]]:
        required = ("keys_by_layer", "values_by_layer", "layer_global_tokens", "grid_hw")
        missing = [key for key in required if key not in prepared]
        if missing:
            raise KeyError(f"P3 prepared field is missing {missing}")
        keys = prepared["keys_by_layer"]
        values = prepared["values_by_layer"]
        globals_ = prepared["layer_global_tokens"]
        grid_hw = prepared["grid_hw"]
        if not all(torch.is_tensor(value) for value in (keys, values, globals_)):
            raise TypeError("P3 keys, values, and globals must remain Tensor values")
        if not isinstance(grid_hw, tuple) or len(grid_hw) != 2:
            raise TypeError("P3 grid_hw must be a two-item tuple")
        height, width = (int(grid_hw[0]), int(grid_hw[1]))
        if height <= 0 or width <= 0:
            raise ValueError("P3 grid_hw must be positive")
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("P3 keys_by_layer and values_by_layer must be [B,4,N,D]")
        batch, layers, tokens, dim = keys.shape
        if layers != self.num_layers or dim != self.dim or tokens != height * width:
            raise ValueError("P3 K/V shape does not satisfy the ledger contract")
        if globals_.shape != (batch, self.num_layers, self.dim):
            raise ValueError("P3 layer_global_tokens must be [B,4,D]")
        if not bool(torch.isfinite(keys).all() and torch.isfinite(values).all() and torch.isfinite(globals_).all()):
            raise ValueError("P3 field tensors must be finite")

        parameter_dtype = self.slot_queries.dtype
        if keys.dtype != parameter_dtype and not self._uses_native_cuda_bfloat16(keys):
            keys = keys.to(dtype=parameter_dtype)
            values = values.to(dtype=parameter_dtype)
            globals_ = globals_.to(dtype=parameter_dtype)
        return keys, values, globals_, (height, width)

    def _layer_weights(self, slot_state: Tensor, layer_globals: Tensor) -> Tensor:
        query = self.layer_query_projection(self.query_norm(slot_state))
        global_tokens = self.layer_global_projection(layer_globals)
        scores = self.layer_score(
            torch.tanh(query.unsqueeze(2) + global_tokens.unsqueeze(1))
        ).squeeze(-1)
        return torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)

    def _visual_logits(
        self,
        slot_state: Tensor,
        keys_by_layer: Tensor,
        layer_globals: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Score each slot against each patch using one layer loop.

        Keeping the loop over the four layers prevents allocation of a
        `[B,21,4,N,D]` intermediate while still using every layer-specific
        P3 key tensor.
        """

        layer_weights = self._layer_weights(slot_state, layer_globals)
        visual_queries = self.visual_query_projection(self.query_norm(slot_state))
        logits = torch.zeros(
            visual_queries.shape[0],
            self.INTERNAL_SLOT_COUNT,
            keys_by_layer.shape[2],
            device=visual_queries.device,
            dtype=visual_queries.dtype,
        )
        scale = self.dim**-0.5
        for layer_index in range(self.num_layers):
            score = torch.einsum(
                "bjd,bnd->bjn",
                visual_queries,
                keys_by_layer[:, layer_index],
            ) * scale
            logits = logits + layer_weights[:, :, layer_index].unsqueeze(-1) * score
        return logits, layer_weights

    @staticmethod
    def _slot_competition(logits: Tensor) -> Tensor:
        """Assign every patch across slots, never across patches."""

        if logits.ndim != 3:
            raise ValueError("competitive logits must be [B,21,N]")
        return torch.softmax(logits.float(), dim=1).to(dtype=logits.dtype)

    def _pooled_values(self, assignments: Tensor, values_by_layer: Tensor, layer_weights: Tensor) -> Tensor:
        denominator = assignments.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        pooled = torch.zeros(
            assignments.shape[0],
            self.INTERNAL_SLOT_COUNT,
            self.dim,
            device=assignments.device,
            dtype=assignments.dtype,
        )
        for layer_index in range(self.num_layers):
            selected = torch.einsum(
                "bjn,bnd->bjd",
                assignments,
                values_by_layer[:, layer_index],
            )
            pooled = pooled + layer_weights[:, :, layer_index].unsqueeze(-1) * selected
        return pooled / denominator

    @staticmethod
    def geometry_from_masks(masks: Tensor, eps: float = 1.0e-6) -> dict[str, Tensor]:
        """Compute finite mask geometry, including safe all-zero fallback."""

        if masks.ndim != 4:
            raise ValueError("slot masks must be [B,J,H,W]")
        batch, slots, height, width = masks.shape
        if min(batch, slots, height, width) <= 0:
            raise ValueError("slot masks must have positive extents")
        if not bool(torch.isfinite(masks).all()):
            raise ValueError("slot masks must be finite")
        if bool((masks < 0.0).any() or (masks > 1.0).any()):
            raise ValueError("slot masks must lie in [0,1]")
        mass = masks.sum(dim=(-1, -2))
        area = mass / float(height * width)
        if bool((mass < 0.0).any() or (area < 0.0).any()):
            raise ValueError("slot mass and area must be nonnegative")
        x = torch.linspace(-1.0, 1.0, width, device=masks.device, dtype=masks.dtype).view(1, 1, 1, width)
        y = torch.linspace(-1.0, 1.0, height, device=masks.device, dtype=masks.dtype).view(1, 1, height, 1)
        safe_mass = mass.clamp_min(eps)
        centroid_x = (masks * x).sum(dim=(-1, -2)) / safe_mass
        centroid_y = (masks * y).sum(dim=(-1, -2)) / safe_mass
        valid = mass > eps
        centroid = torch.stack((centroid_x, centroid_y), dim=-1)
        centroid = torch.where(valid.unsqueeze(-1), centroid, torch.zeros_like(centroid))
        scale = area.clamp_min(0.0).sqrt()
        return {
            "mass": mass,
            "activity": area,
            "area": area,
            "centroid": centroid,
            "scale": scale,
            "nonempty": valid,
        }

    def _issue_public_evidence(
        self,
        *,
        tokens: Tensor,
        masks: Tensor,
        valid_mask: Tensor,
        diagnostics: InternalLedgerDiagnostics,
        provenance: object,
    ) -> PublicEvidenceView:
        # Public consumers receive their own autograd-preserving clone.  It
        # cannot share storage with either normal forward outputs or the
        # detached internal audit snapshot.
        issued_tokens = tokens.clone()
        issued_masks = masks.clone()
        issued_valid_mask = valid_mask.clone()
        slot_indices = tuple(range(self.PUBLIC_SLOT_COUNT))
        view = PublicEvidenceView(
            _issuer=_PUBLIC_EVIDENCE_ISSUER,
            tokens=issued_tokens,
            masks=issued_masks,
            valid_mask=issued_valid_mask,
            slot_indices=slot_indices,
            provenance=provenance,
        )
        token_snapshot, mask_snapshot, valid_mask_snapshot = _public_view_integrity_snapshots(view)
        self._issued_public_views[view] = _IssuedPublicEvidence(
            tokens=issued_tokens,
            masks=issued_masks,
            valid_mask=issued_valid_mask,
            slot_indices=slot_indices,
            provenance=provenance,
            token_integrity_snapshot=token_snapshot,
            mask_integrity_snapshot=mask_snapshot,
            valid_mask_integrity_snapshot=valid_mask_snapshot,
            token_version=issued_tokens._version,
            mask_version=issued_masks._version,
            valid_mask_version=issued_valid_mask._version,
        )
        self._diagnostics_by_public_view[view] = diagnostics
        return view

    def _require_public_evidence(self, evidence: PublicEvidenceView) -> _IssuedPublicEvidence:
        if not isinstance(evidence, PublicEvidenceView):
            raise TypeError("P5 contribution/evidence interfaces require a ledger-issued PublicEvidenceView")
        issued = self._issued_public_views.get(evidence)
        if issued is None:
            raise ValueError("PublicEvidenceView was not issued by this RAELSlotLedger")
        try:
            tokens, masks, valid_mask, slot_indices, provenance = _public_view_private_payload(evidence)
            token_snapshot, mask_snapshot, valid_mask_snapshot = _public_view_integrity_snapshots(evidence)
        except AttributeError as error:
            raise ValueError("PublicEvidenceView sealed payload is incomplete") from error
        if (
            provenance is not issued.provenance
            or tokens is not issued.tokens
            or masks is not issued.masks
            or valid_mask is not issued.valid_mask
            or slot_indices != issued.slot_indices
        ):
            raise ValueError("PublicEvidenceView provenance or public tensors were modified")
        if (
            tokens._version != issued.token_version
            or masks._version != issued.mask_version
            or valid_mask._version != issued.valid_mask_version
        ):
            raise ValueError("PublicEvidenceView tensor version changed after issuance")
        if (
            token_snapshot is not issued.token_integrity_snapshot
            or mask_snapshot is not issued.mask_integrity_snapshot
            or valid_mask_snapshot is not issued.valid_mask_integrity_snapshot
        ):
            raise ValueError("PublicEvidenceView integrity snapshot was modified")
        if not (
            torch.equal(tokens.detach(), token_snapshot)
            and torch.equal(masks.detach(), mask_snapshot)
            and torch.equal(valid_mask.detach(), valid_mask_snapshot)
        ):
            raise ValueError("PublicEvidenceView integrity snapshot mismatch")
        if issued.slot_indices != tuple(range(self.PUBLIC_SLOT_COUNT)):
            raise ValueError("PublicEvidenceView must contain exactly public indices 0..19")
        if tokens.ndim != 3 or tokens.shape[1:] != (self.PUBLIC_SLOT_COUNT, self.dim):
            raise ValueError("public evidence tokens must be exactly [B,20,D]")
        if masks.ndim != 4 or masks.shape[0] != tokens.shape[0] or masks.shape[1] != self.PUBLIC_SLOT_COUNT:
            raise ValueError("public evidence masks must be exactly [B,20,H,W]")
        if valid_mask.dtype != torch.bool or valid_mask.shape != tokens.shape[:2]:
            raise ValueError("public evidence validity must be bool [B,20]")
        if not bool(torch.isfinite(tokens).all() and torch.isfinite(masks).all()):
            raise ValueError("public evidence tensors must be finite")
        if bool((masks < 0.0).any() or (masks > 1.0).any()):
            raise ValueError("public evidence masks must lie in [0,1]")
        mass = masks.sum(dim=(-1, -2))
        area = mass / float(masks.shape[-1] * masks.shape[-2])
        if bool((mass < 0.0).any() or (area < 0.0).any()):
            raise ValueError("public evidence mass and area must be nonnegative")
        return issued

    def _global_context(self, layer_globals: Tensor) -> tuple[Tensor, Tensor]:
        projected = self.global_context_projection(layer_globals)
        scores = self.global_context_score(torch.tanh(projected)).squeeze(-1)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        context = (weights.unsqueeze(-1) * layer_globals).sum(dim=1)
        return self.global_context_norm(context), weights

    def forward(self, prepared_field: Mapping[str, Any]) -> dict[str, Any]:
        """Run exactly two competitive binding updates from one P3 prepared field."""

        keys, values, layer_globals, grid_hw = self._working_tensors(prepared_field)
        batch = keys.shape[0]
        initial_slots = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)

        # Keep full-resolution K/V in their BF16 field storage. CUDA autocast
        # handles FP32 master weights; only small normalization scores widen.
        with self._working_autocast_context(keys):
            visual_logits_one, layer_weights_one = self._visual_logits(initial_slots, keys, layer_globals)
            assignment_one = self._slot_competition(visual_logits_one)
            pooled_one = self._pooled_values(assignment_one, values, layer_weights_one)
            slots_one = self.slot_gru(
                pooled_one.reshape(-1, self.dim),
                initial_slots.reshape(-1, self.dim),
            ).reshape(batch, self.INTERNAL_SLOT_COUNT, self.dim)

            visual_logits_two, layer_weights_two = self._visual_logits(slots_one, keys, layer_globals)
            logits_two = visual_logits_two + self.mask_bias * torch.log(assignment_one.clamp_min(self.eps))
            assignment_two = self._slot_competition(logits_two)
            pooled_two = self._pooled_values(assignment_two, values, layer_weights_two)
            slots_two = self.slot_gru(
                pooled_two.reshape(-1, self.dim),
                slots_one.reshape(-1, self.dim),
            ).reshape(batch, self.INTERNAL_SLOT_COUNT, self.dim)

        masks = assignment_two.reshape(batch, self.INTERNAL_SLOT_COUNT, grid_hw[0], grid_hw[1])
        geometry = self.geometry_from_masks(masks, eps=self.eps)
        global_context, global_layer_weights = self._global_context(layer_globals)
        # Clone public output tensors so normal forward consumers do not share
        # storage with the 21-slot internal ledger state.
        public_masks = masks[:, : self.PUBLIC_SLOT_COUNT].clone()
        public_tokens = slots_two[:, : self.PUBLIC_SLOT_COUNT].clone()
        public_activity = geometry["activity"][:, : self.PUBLIC_SLOT_COUNT].clone()
        public_area = geometry["area"][:, : self.PUBLIC_SLOT_COUNT].clone()
        public_centroid = geometry["centroid"][:, : self.PUBLIC_SLOT_COUNT].clone()
        public_scale = geometry["scale"][:, : self.PUBLIC_SLOT_COUNT].clone()
        public_nonempty = geometry["nonempty"][:, : self.PUBLIC_SLOT_COUNT].clone()
        public_valid = public_activity > self.eps
        no_valid_public = ~public_valid.any(dim=1)
        if bool(no_valid_public.any()):
            fallback_indices = public_activity.argmax(dim=1)
            public_valid = public_valid.clone()
            public_valid[no_valid_public, fallback_indices[no_valid_public]] = True

        provenance = object()
        diagnostics = InternalLedgerDiagnostics(
            _issuer=_INTERNAL_DIAGNOSTIC_ISSUER,
            provenance=provenance,
            slot_masks=masks,
            slot_tokens=slots_two,
            slot_activity=geometry["activity"],
            slot_area=geometry["area"],
            slot_centroid=geometry["centroid"],
            slot_scale=geometry["scale"],
            slot_nonempty=geometry["nonempty"],
            layer_weights_one=layer_weights_one,
            layer_weights_two=layer_weights_two,
            visual_logits_one=visual_logits_one,
            assignment_one=assignment_one,
            visual_logits_two=visual_logits_two,
            logits_two=logits_two,
            assignment_two=assignment_two,
            slot_specs=self.slot_specs,
        )
        public_evidence = self._issue_public_evidence(
            tokens=public_tokens,
            masks=public_masks,
            valid_mask=public_valid,
            diagnostics=diagnostics,
            provenance=provenance,
        )

        return {
            "iterations": self.ITERATIONS,
            "public_evidence": public_evidence,
            "slot_masks": public_masks,
            "slot_tokens": public_tokens,
            "public_slot_masks": public_masks,
            "public_slot_tokens": public_tokens,
            "slot_activity": public_activity,
            "slot_area": public_area,
            "slot_centroid": public_centroid,
            "slot_scale": public_scale,
            "slot_nonempty": public_nonempty,
            "public_slot_valid": public_valid,
            "background_mask": masks[:, self.BACKGROUND_INDEX : self.BACKGROUND_INDEX + 1],
            "background_token": slots_two[:, self.BACKGROUND_INDEX],
            "background_activity": geometry["activity"][:, self.BACKGROUND_INDEX],
            "background_contract": {
                "allow_contribution": False,
                "allow_cf": False,
                "allow_explanation": False,
            },
            "global_context": global_context,
            "global_context_layer_weights": global_layer_weights,
        }

    def public_contribution_view(self, evidence: PublicEvidenceView) -> PublicEvidenceView:
        """Return the only P5 boundary permitted to contribution consumers."""

        self._require_public_evidence(evidence)
        return evidence

    def to_evidence_read_bundle(self, evidence: PublicEvidenceView) -> Any:
        """Adapt the 20 public image-derived slots to P4 without P4 edits."""

        from fate_oia.models.rael_semantic_reason import EvidenceReadBundle

        public = self.public_contribution_view(evidence)
        return EvidenceReadBundle(tokens=public.tokens, valid_mask=public.valid_mask)

    def latent_training_view(self, evidence: PublicEvidenceView) -> dict[str, Tensor]:
        """Expose unnamed latent slots only to future task/view/diversity losses.

        The returned tensors intentionally carry no reason schema, named
        evidence family, or action-compatibility annotation.  P6/P13 can use
        this boundary for task utility, mirror/view consistency, and diversity
        regularization without turning latent slots into human-named evidence.
        """

        public = self._require_public_evidence(evidence)
        tokens = public.tokens
        masks = public.masks
        activity = masks.sum(dim=(-1, -2)) / float(masks.shape[-1] * masks.shape[-2])
        latent_slice = slice(self.LATENT_START, self.BACKGROUND_INDEX)
        return {
            "task_tokens": tokens[:, latent_slice],
            "view_masks": masks[:, latent_slice],
            "diversity_activity": activity[:, latent_slice],
        }

    def audit_diagnostics(self, evidence: PublicEvidenceView) -> InternalLedgerDiagnostics:
        """Return 21-slot state solely to audit code holding an issued view."""

        self._require_public_evidence(evidence)
        diagnostics = self._diagnostics_by_public_view.get(evidence)
        if diagnostics is None:
            raise ValueError("internal ledger diagnostics are unavailable for this evidence provenance")
        return diagnostics

    def mirror_geometry_consistency(
        self,
        canonical_evidence: PublicEvidenceView,
        mirrored_evidence: PublicEvidenceView,
    ) -> dict[str, Tensor]:
        """Compare canonical geometry with horizontally flipped mirror output."""

        canonical_masks = self.audit_diagnostics(canonical_evidence).slot_masks
        mirrored_masks = self.audit_diagnostics(mirrored_evidence).slot_masks
        if canonical_masks.shape != mirrored_masks.shape or canonical_masks.ndim != 4:
            raise ValueError("canonical and mirrored masks must share [B,21,H,W]")
        permutation = self.mirror_slot_permutation.to(device=canonical_masks.device)
        mirror_aligned = mirrored_masks.index_select(1, permutation).flip(dims=(-1,))
        # Entity and latent slots are deliberately non-identifying.  Mirror
        # diagnostics therefore compare their aggregate occupancy, while road
        # identities retain their explicit left/right permutation.
        fixed_indices = torch.cat(
            (
                self.road_slot_indices.to(device=canonical_masks.device),
                torch.tensor([self.BACKGROUND_INDEX], device=canonical_masks.device),
            )
        )
        canonical_fixed = canonical_masks.index_select(1, fixed_indices)
        mirror_fixed = mirror_aligned.index_select(1, fixed_indices)
        canonical_geometry = self.geometry_from_masks(canonical_fixed, eps=self.eps)
        mirror_geometry = self.geometry_from_masks(mirror_fixed, eps=self.eps)
        entity_union = canonical_masks[:, : self.ENTITY_SLOT_COUNT].sum(dim=1, keepdim=True)
        mirror_entity_union = mirror_aligned[:, : self.ENTITY_SLOT_COUNT].sum(dim=1, keepdim=True)
        latent_union = canonical_masks[:, self.LATENT_START : self.BACKGROUND_INDEX].sum(dim=1, keepdim=True)
        mirror_latent_union = mirror_aligned[:, self.LATENT_START : self.BACKGROUND_INDEX].sum(dim=1, keepdim=True)
        return {
            "mask_l1": (canonical_fixed - mirror_fixed).abs().mean(dim=(-1, -2)),
            "area_l1": (canonical_geometry["area"] - mirror_geometry["area"]).abs(),
            "centroid_l1": (canonical_geometry["centroid"] - mirror_geometry["centroid"]).abs().mean(dim=-1),
            "entity_union_l1": (entity_union - mirror_entity_union).abs().mean(dim=(-1, -2)),
            "latent_union_l1": (latent_union - mirror_latent_union).abs().mean(dim=(-1, -2)),
            "finite": torch.isfinite(canonical_fixed).all(dim=(-1, -2))
            & torch.isfinite(mirror_fixed).all(dim=(-1, -2)),
        }
