from __future__ import annotations

import pytest
import torch

from fate_oia.models.psr_specialist_registry import LoadedSpecialistLogits, validate_alignment


def loaded(name: str, files=None, flip=False):
    labels_a = torch.zeros(3, 4)
    labels_r = torch.zeros(3, 21)
    if flip:
        labels_a[0, 0] = 1
    return LoadedSpecialistLogits(
        name=name,
        role="x",
        action_logits=torch.zeros(3, 4),
        reason_logits=torch.zeros(3, 21),
        labels_action=labels_a,
        labels_reason=labels_r,
        file_names=files or ["a", "b", "c"],
        source_dir=None,
    )


def test_alignment_rejects_file_and_label_mismatch():
    validate_alignment(loaded("a"), loaded("b"))
    with pytest.raises(ValueError, match="file_names mismatch"):
        validate_alignment(loaded("a"), loaded("bad_files", ["a", "x", "c"]))
    with pytest.raises(ValueError, match="action labels mismatch"):
        validate_alignment(loaded("a"), loaded("bad_labels", flip=True))
    with pytest.raises(ValueError, match="duplicated file names"):
        validate_alignment(loaded("a"), loaded("dupe", ["a", "a", "c"]))
