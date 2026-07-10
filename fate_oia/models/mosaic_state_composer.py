from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class MOSAICSupportVetoComposer(nn.Module):
    def __init__(
        self,
        factor_names: Sequence[str],
        states: dict[str, Any],
        *,
        dim: int = 384,
        state_residual_cap: float = 0.20,
    ) -> None:
        super().__init__()
        self.factor_names = tuple(factor_names)
        self.factor_index = {name: index for index, name in enumerate(self.factor_names)}
        self.state_names = tuple(states)
        self.state_index = {name: index for index, name in enumerate(self.state_names)}
        self.dim = dim
        if len(self.factor_index) != len(self.factor_names) or not self.factor_names:
            raise ValueError("state composer requires unique non-empty factor names")
        if not states or not 0.0 <= state_residual_cap <= 0.20:
            raise ValueError("state composer requires states and residual cap <= 0.20")
        self.state_residual_cap = float(state_residual_cap)

        dependencies: dict[str, set[str]] = {name: set() for name in self.state_names}
        support_specs: dict[str, list[tuple[tuple[str, ...], int]]] = {name: [] for name in self.state_names}
        veto_specs: dict[str, tuple[tuple[str, ...], int | None]] = {}
        self.support_weights = nn.ParameterList()
        self.veto_weights = nn.ParameterList()
        initial_weight = _inverse_softplus(1.0)
        for state_name, specification in states.items():
            if not isinstance(specification, dict) or set(specification) != {"required_groups", "veto"}:
                raise ValueError(f"state {state_name} has invalid support/veto fields")
            groups = specification["required_groups"]
            veto = specification["veto"]
            if not isinstance(groups, list) or not groups or not isinstance(veto, list):
                raise ValueError(f"state {state_name} has invalid support/veto lists")
            for group in groups:
                if not isinstance(group, dict) or set(group) != {"any_of"} or not isinstance(group["any_of"], list):
                    raise ValueError(f"state {state_name} has invalid required group")
                references = tuple(group["any_of"])
                if not references:
                    raise ValueError(f"state {state_name} has empty required group")
                self._validate_references(state_name, references, dependencies)
                parameter_index = len(self.support_weights)
                self.support_weights.append(nn.Parameter(torch.full((len(references),), initial_weight)))
                support_specs[state_name].append((references, parameter_index))
            veto_references = tuple(veto)
            self._validate_references(state_name, veto_references, dependencies)
            if veto_references:
                parameter_index = len(self.veto_weights)
                self.veto_weights.append(nn.Parameter(torch.full((len(veto_references),), initial_weight)))
            else:
                parameter_index = None
            veto_specs[state_name] = (veto_references, parameter_index)

        self._topological_order = self._topological_sort(dependencies)
        self._support_specs = support_specs
        self._veto_specs = veto_specs
        self.raw_gamma = nn.Parameter(torch.full((len(self.state_names),), _inverse_softplus(0.25)))
        self.state_queries = nn.Parameter(torch.randn(len(self.state_names), dim) * 0.02)
        self.context_key = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.context_value = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.residual_projection = nn.Linear(dim, len(self.state_names), bias=False)

    def _validate_references(
        self,
        state_name: str,
        references: tuple[str, ...],
        dependencies: dict[str, set[str]],
    ) -> None:
        for reference in references:
            if reference in self.state_index:
                dependencies[state_name].add(reference)
            elif reference not in self.factor_index:
                raise ValueError(f"state {state_name} references unknown factor/state {reference}")

    @staticmethod
    def _topological_sort(dependencies: dict[str, set[str]]) -> tuple[str, ...]:
        visiting: set[str] = set()
        visited: set[str] = set()
        result: list[str] = []

        def visit(state_name: str) -> None:
            if state_name in visiting:
                raise ValueError(f"state dependency cycle detected at {state_name}")
            if state_name in visited:
                return
            visiting.add(state_name)
            for dependency in sorted(dependencies[state_name]):
                visit(dependency)
            visiting.remove(state_name)
            visited.add(state_name)
            result.append(state_name)

        for state_name in dependencies:
            visit(state_name)
        return tuple(result)

    def _reference_values(
        self,
        references: tuple[str, ...],
        factor_activation: torch.Tensor,
        computed_states: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        values = [
            computed_states[reference]
            if reference in self.state_index
            else factor_activation[:, self.factor_index[reference]]
            for reference in references
        ]
        return torch.stack(values, dim=-1)

    @staticmethod
    def _bounded_nonnegative_weights(raw_weights: torch.Tensor) -> torch.Tensor:
        positive = F.softplus(raw_weights)
        return positive / (1.0 + positive)

    def _bounded_gamma(self) -> torch.Tensor:
        positive = F.softplus(self.raw_gamma)
        return 5.0 * positive / (1.0 + positive)

    def _visual_residual(self, context: torch.Tensor, residual_scale: float) -> torch.Tensor:
        batch_size = context.shape[0]
        compute_context = context.to(dtype=self.context_key.weight.dtype)
        keys = self.context_key(compute_context).flatten(2)
        values = self.context_value(compute_context).flatten(2).transpose(1, 2)
        attention_logits = torch.einsum("kd,bdn->bkn", self.state_queries, keys) / math.sqrt(self.dim)
        attention = torch.softmax(attention_logits, dim=-1)
        attended = torch.einsum("bkn,bnd->bkd", attention, values)
        projected = self.residual_projection(attended)
        diagonal = projected.diagonal(dim1=1, dim2=2).reshape(batch_size, len(self.state_names))
        return self.state_residual_cap * residual_scale * torch.tanh(diagonal)

    @staticmethod
    def _logit(probability: torch.Tensor) -> torch.Tensor:
        probability = probability.clamp(1e-5, 1.0 - 1e-5)
        return probability.log() - (1.0 - probability).log()

    @staticmethod
    def _binary_entropy(probability: torch.Tensor) -> torch.Tensor:
        probability = probability.clamp(1e-6, 1.0 - 1e-6)
        return -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log()) / math.log(2.0)

    def forward(
        self,
        positive_evidence: torch.Tensor,
        negative_evidence: torch.Tensor,
        uncertainty: torch.Tensor,
        context: torch.Tensor,
        *,
        residual_scale: float = 1.0,
    ) -> dict[str, Any]:
        if type(residual_scale) not in {int, float} or not 0.0 <= float(residual_scale) <= 1.0:
            raise ValueError("residual_scale must be in [0,1]")
        batch_size = positive_evidence.shape[0] if positive_evidence.ndim else -1
        factor_shape = (batch_size, len(self.factor_names))
        if (
            tuple(positive_evidence.shape) != factor_shape
            or tuple(negative_evidence.shape) != factor_shape
            or tuple(uncertainty.shape) != factor_shape
            or tuple(context.shape) != (batch_size, self.dim, 12, 20)
        ):
            raise ValueError("MOSAIC state composer shape contract is invalid")
        if not all(value.is_floating_point() for value in (positive_evidence, negative_evidence, uncertainty, context)):
            raise ValueError("MOSAIC state composer requires floating-point inputs")

        residual = self._visual_residual(context, float(residual_scale))
        factor_activation = (
            positive_evidence.clamp(0.0, 1.0)
            * (1.0 - negative_evidence.clamp(0.0, 1.0))
            * (1.0 - uncertainty.clamp(0.0, 1.0))
        )
        computed_probabilities: dict[str, torch.Tensor] = {}
        computed_support: dict[str, torch.Tensor] = {}
        computed_veto: dict[str, torch.Tensor] = {}
        computed_logits: dict[str, torch.Tensor] = {}
        gamma = self._bounded_gamma()
        for state_name in self._topological_order:
            group_support: list[torch.Tensor] = []
            for references, parameter_index in self._support_specs[state_name]:
                evidence = self._reference_values(references, factor_activation, computed_probabilities)
                weights = self._bounded_nonnegative_weights(self.support_weights[parameter_index])
                contribution = weights.unsqueeze(0) * evidence
                group_support.append(1.0 - torch.prod(1.0 - contribution, dim=-1))
            support = torch.stack(group_support, dim=-1).prod(dim=-1)

            veto_references, veto_parameter_index = self._veto_specs[state_name]
            if veto_parameter_index is None:
                veto = torch.zeros_like(support)
            else:
                veto_evidence = self._reference_values(veto_references, factor_activation, computed_probabilities)
                veto_weights = self._bounded_nonnegative_weights(self.veto_weights[veto_parameter_index])
                veto_contribution = veto_weights.unsqueeze(0) * veto_evidence
                veto = 1.0 - torch.prod(1.0 - veto_contribution, dim=-1)

            state_index = self.state_index[state_name]
            veto_penalty = -torch.log1p(-veto.clamp(max=1.0 - 1e-5))
            state_logit = self._logit(support) - gamma[state_index] * veto_penalty + residual[:, state_index]
            state_probability = torch.sigmoid(state_logit)
            computed_support[state_name] = support
            computed_veto[state_name] = veto
            computed_logits[state_name] = state_logit
            computed_probabilities[state_name] = state_probability

        logits = torch.stack([computed_logits[name] for name in self.state_names], dim=-1)
        probabilities = torch.stack([computed_probabilities[name] for name in self.state_names], dim=-1)
        support = torch.stack([computed_support[name] for name in self.state_names], dim=-1)
        veto = torch.stack([computed_veto[name] for name in self.state_names], dim=-1)
        state_uncertainty = self._binary_entropy(probabilities)
        return {
            "decision_state_logits": logits,
            "decision_state_prob": probabilities,
            "decision_state_support": support,
            "decision_state_veto": veto,
            "decision_state_residual": residual,
            "decision_state_uncertainty": state_uncertainty,
            "state_stats": {
                "support_weight_mean": torch.stack(
                    [self._bounded_nonnegative_weights(weight).mean() for weight in self.support_weights]
                ).mean().detach(),
                "veto_weight_mean": torch.stack(
                    [self._bounded_nonnegative_weights(weight).mean() for weight in self.veto_weights]
                ).mean().detach(),
                "gamma_mean": gamma.mean().detach(),
                "input_negative_evidence_mean": negative_evidence.mean().detach(),
                "input_uncertainty_mean": uncertainty.mean().detach(),
            },
        }
