from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml


DEFAULT_OFFLINE_TEXT_ENCODER = "artifacts/heca/frozen_bert_base_uncased"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def export_prototypes(
    schema_path: str | Path,
    output_dir: str | Path,
    *,
    encoder_id: str = DEFAULT_OFFLINE_TEXT_ENCODER,
) -> dict[str, object]:
    # The text tower exists only in this offline command. Runtime model files
    # contain no tokenizer/model imports and only load the emitted tensors.
    # This environment includes an incompatible optional TensorFlow install;
    # explicitly selecting the PyTorch backend keeps the one-time export local.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    from transformers import AutoModel, AutoTokenizer

    schema = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))
    rows = list(schema["factors"])
    # Ontology text is a one-time, reproducible offline artifact. A network
    # fallback would make Gate results depend on host connectivity.
    tokenizer = AutoTokenizer.from_pretrained(encoder_id, local_files_only=True)
    encoder = AutoModel.from_pretrained(encoder_id, local_files_only=True).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    def encode(prompts: list[str]) -> torch.Tensor:
        batch = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        hidden = encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return torch.nn.functional.normalize(pooled, dim=-1).cpu()

    factor_prompts = [str(row["text_prompt"]) for row in rows]
    state_prompts = [list(map(str, row["state_prompts"])) for row in rows]
    factor = encode(factor_prompts)
    flat_state = encode([prompt for prompts in state_prompts for prompt in prompts])
    max_states = max(map(len, state_prompts))
    state = torch.zeros(len(rows), max_states, flat_state.shape[-1])
    cursor = 0
    for factor_id, prompts in enumerate(state_prompts):
        state[factor_id, : len(prompts)] = flat_state[cursor : cursor + len(prompts)]
        cursor += len(prompts)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    factor_path = root / "factor_text_prototype.pt"
    state_path = root / "state_text_prototype.pt"
    torch.save(factor, factor_path)
    torch.save(state, state_path)
    manifest = {
        "schema_version": 1,
        "factor_count": len(rows),
        "state_count": max_states,
        "encoder_id": encoder_id,
        "offline_only": True,
        "local_files_only": True,
        "schema_path": str(schema_path),
        "schema_sha256": _sha256(Path(schema_path)),
        "factor_prompts": factor_prompts,
        "state_prompts": state_prompts,
        "factor_shape": list(factor.shape),
        "state_shape": list(state.shape),
        "dtype": str(factor.dtype),
        "factor_path": str(factor_path),
        "state_path": str(state_path),
        "factor_sha256": _sha256(factor_path),
        "state_sha256": _sha256(state_path),
    }
    manifest["sha256"] = hashlib.sha256(
        (manifest["factor_sha256"] + manifest["state_sha256"] + manifest["schema_sha256"]).encode()
    ).hexdigest()
    (root / "ontology_prototype_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "heca_ontology_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="configs/meter_factor_schema.yaml")
    parser.add_argument("--output_dir", default="artifacts/heca")
    parser.add_argument("--encoder_id", default=DEFAULT_OFFLINE_TEXT_ENCODER)
    args = parser.parse_args()
    print(json.dumps(export_prototypes(args.schema, args.output_dir, encoder_id=args.encoder_id), indent=2))


if __name__ == "__main__":
    main()
