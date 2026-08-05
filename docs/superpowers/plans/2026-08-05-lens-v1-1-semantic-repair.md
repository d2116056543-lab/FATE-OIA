# LENS V1.1 Semantic Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the latent-state/emission axis contract while retaining the
pilot-proven LENS transport paths and ranking later full-run checkpoints by
real metrics.

**Architecture:** The visual latent state stays `[positive, counter, unknown]`.
The annotation emission stays `[counter, unknown, positive]`. A named,
single-purpose permutation mediates every interaction between the two spaces;
all model outputs declare which space they use.

**Tech Stack:** PyTorch, pytest, frozen DINO ViT-S/8, existing LENS trainer and
audit artifacts.

---

### Task 1: Lock the semantic contract with failing tests

**Files:**
- Create: `tests/test_lens_emission_axis_contract.py`
- Modify: `fate_oia/models/lens_annotation_emission.py`
- Modify: `fate_oia/losses/lens_latent_losses.py`

- [ ] **Step 1: Write the failing identity-emission test**

```python
import torch

def test_identity_emission_interprets_state_order_explicitly():
    emission = LENSAnnotationEmission(reason_dim=1)
    states = torch.eye(3).view(3, 1, 3)  # positive, counter, unknown
    result = emission(states, torch.zeros(3, 1), progress=0.0)
    assert torch.allclose(result["reason_prob_latent"].flatten(),
                          torch.tensor([1.0 - 1e-6, 1e-6, 0.5]), atol=2e-6)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_lens_emission_axis_contract.py::test_identity_emission_interprets_state_order_explicitly -q`

Expected: FAIL because the unconverted direct contraction produces
`[1e-6, 0.5, 1.0 - 1e-6]`.

- [ ] **Step 3: Write two failing loss-axis tests**

```python
import torch

def test_responsibility_returns_named_state_and_emission_orders():
    state_prob = torch.tensor([[[0.70, 0.20, 0.10]]])
    emission_prob = torch.tensor([[0.05, 0.50, 0.95]])
    observed_reason = torch.ones(1, 1)
    action_state_logits = torch.zeros(1, 1, 3, 4)
    action_targets = torch.zeros(1, 4)
    payload = conflict_discounted_responsibility(
        state_prob, emission_prob, observed_reason, action_state_logits,
        action_targets, lambda_action=1.0,
    )
    assert torch.allclose(payload["gamma_state_order"],
                          payload["gamma_emission_order"][..., [2, 0, 1]])

def test_conflict_safe_logits_use_emission_axis_conversion():
    state_prob = torch.tensor([[[1.0, 0.0, 0.0]]])
    emission_prob = torch.tensor([[0.0, 0.5, 1.0]])
    result = conflict_safe_reason_logits(
        state_prob, torch.zeros(1, 1), emission_prob, torch.ones(1, 1),
        torch.tensor([[[1.0, 0.0, 0.0]]]), torch.zeros(1, 1),
        alpha_reason=1.0,
    )
    assert result["reason_logits_latent_train"].item() > 10.0
```

- [ ] **Step 4: Run the loss tests and verify RED**

Run: `python -m pytest tests/test_lens_emission_axis_contract.py -q`

Expected: FAIL because the named output does not exist and the direct
contraction treats a positive state as counter evidence.

### Task 2: Implement one explicit conversion

**Files:**
- Modify: `fate_oia/models/lens_annotation_emission.py`
- Modify: `fate_oia/losses/lens_latent_losses.py`

- [ ] **Step 1: Define the only allowed conversion**

```python
STATE_TO_EMISSION = (1, 2, 0)
EMISSION_TO_STATE = (2, 0, 1)

def state_to_emission_order(state_prob: Tensor) -> Tensor:
    return state_prob[..., list(STATE_TO_EMISSION)]

def emission_to_state_order(value: Tensor) -> Tensor:
    return value[..., list(EMISSION_TO_STATE)]
```

- [ ] **Step 2: Apply it before every state/emission contraction**

```python
state_for_emission = state_to_emission_order(state_prob)
latent_prob = torch.einsum("brs,rs->br", state_for_emission, effective)
```

`conflict_discounted_responsibility` computes annotation likelihood and gamma
in emission order, returns both `gamma_emission_order` and
`gamma_state_order`, and `state_loss` receives only `gamma_state_order`.
`emission_loss` receives only `gamma_emission_order`.

- [ ] **Step 3: Run targeted tests and verify GREEN**

Run: `python -m pytest tests/test_lens_emission_axis_contract.py -q`

Expected: PASS.

### Task 3: Verify no branch semantics regress

**Files:**
- Modify: `tests/test_lens_oia_model.py` or the existing LENS model test file
- Modify: `fate_oia/engine/audit_lens_oia_implementation.py` only if its
  dynamic contract omits named gamma orders

- [ ] **Step 1: Add a model-output contract assertion**

```python
assert output["state_prob"].shape[-1] == 3
assert output["emission_prob"].shape[-1] == 3
assert output["gamma_state_order"].shape == output["state_prob"].shape
assert output["gamma_emission_order"].shape == output["state_prob"].shape
```

- [ ] **Step 2: Compile and run all LENS tests**

Run: `python -m py_compile fate_oia/models/lens_annotation_emission.py fate_oia/losses/lens_latent_losses.py && python -m pytest tests/test_lens_*.py -q`

Expected: PASS with no frozen-DINO, action firewall, or direct-image protocol
regression.

### Task 4: Re-audit and make the metric-first full-run decision

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

- [ ] **Step 1: Run the official-DINO audit and a short smoke**

The smoke must write branch metrics for action source/base/final and reason
source/latent/formal, emission order diagnostics, selected/control deletion,
and all owner gradient values.

- [ ] **Step 2: Record the decision rule**

Record that full training is allowed when the semantic tests and safety
protocol pass, and branch deltas are bounded/non-harmful. Numerical gates are
reported diagnostically; checkpoint ranking remains metric-first.

- [ ] **Step 3: Commit the repair and documentation**

```bash
git add fate_oia tests docs/superpowers task_plan.md findings.md progress.md
git commit -m "fix: align LENS latent state and emission axes"
```
